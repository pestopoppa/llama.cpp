#!/usr/bin/env python3
"""EXL3 safetensors -> GGUF importer for qwen4exp (Qwen3.8-Flash-Next), EXL3-6 CPU path.

Takes the EXL3-quantized linears from a turboderp exl3 checkpoint and everything else (KV,
tokenizer, non-EXL3 tensors) byte-for-byte from an existing qwen4exp GGUF (the unsloth UD
release). Output is a 2-split model: split 1 = everything but the n-gram table, split 2 =
`per_layer_token_embd` alone, written once and shared by symlink (EXL3-CPU-DESIGN.md s.6).

modes:
  native       EXL3 tensors as GGML_TYPE_EXL3_<K> (band8 layout) + <base>.exl3_suh / .exl3_svh (F16)
  materialize  the same matrices reconstructed to F16 (canonical stage roundings) -- the
               reference artifact for model-level comparisons

examples:
  exl3_import.py --mode native --layers 4 --keep-ud-head --out /…/native-L4
  exl3_import.py --mode materialize --layers 4 --keep-ud-head --out /…/mat-L4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gguf-py"))
sys.path.insert(0, HERE)

import gguf  # noqa: E402
import gguf.quants  # noqa: E402
from gguf import GGMLQuantizationType as QT  # noqa: E402
import exl3np as X  # noqa: E402

DEF_EXL3 = "/mnt/raid0/llm/models/turboderp/Qwen3.8-Flash-Next-exl3-3.05bpw"
DEF_UD = "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
DEF_FILES = "/mnt/raid0/llm/tmp/exl3-q38fn-dl/files.txt"
HF = "model.language_model."
EXL3_TYPES = {k: getattr(QT, f"EXL3_{k}") for k in range(1, 9)}
PER_LAYER_ARRAYS = {"qwen4exp.attention.compress_ratios"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------------------------

def layer_map(il: int, hp: dict):
    """GGUF base name -> (hf base or list of per-expert hf bases, vreorder spec or None)."""
    p = f"{HF}layers.{il}."
    nk, nv, hd = hp["linear_num_key_heads"], hp["linear_num_value_heads"], hp["linear_value_head_dim"]
    kd = hp["linear_key_head_dim"] * nk
    assert hd == 128, "V-head reorder as a 128-block permutation needs head_dim == 128"
    m = {
        f"blk.{il}.attn_qkv": (p + "linear_attn.in_proj_qkv", ("out", (2 * kd) // 128, nk, nv)),
        f"blk.{il}.attn_gate": (p + "linear_attn.in_proj_z", ("out", 0, nk, nv)),
        f"blk.{il}.ssm_out": (p + "linear_attn.out_proj", ("in", 0, nk, nv)),
        f"blk.{il}.attn_q": (p + "self_attn.q_proj", None),
        f"blk.{il}.attn_k": (p + "self_attn.k_proj", None),
        f"blk.{il}.attn_v": (p + "self_attn.v_proj", None),
        f"blk.{il}.attn_output": (p + "self_attn.o_proj", None),
        f"blk.{il}.ffn_gate_shexp": (p + "mlp.shared_expert.gate_proj", None),
        f"blk.{il}.ffn_up_shexp": (p + "mlp.shared_expert.up_proj", None),
        f"blk.{il}.ffn_down_shexp": (p + "mlp.shared_expert.down_proj", None),
    }
    ne = hp["num_experts"]
    for proj in ("gate", "up", "down"):
        m[f"blk.{il}.ffn_{proj}_exps"] = ([p + f"mlp.experts.{e}.{proj}_proj" for e in range(ne)], None)
    return m


def head_base(st):
    for c in ("lm_head", HF + "lm_head", "model.lm_head"):
        if c + ".trellis" in st:
            return c
    return None


# ---------------------------------------------------------------------------------------------
# one EXL3 matrix
# ---------------------------------------------------------------------------------------------

class Exl3Matrix:
    def __init__(self, st, base):
        self.base = base
        self.trellis = st.get(base + ".trellis")
        self.suh = np.asarray(st.get(base + ".suh"))
        self.svh = np.asarray(st.get(base + ".svh"))
        mul = int(np.asarray(st.get(base + ".mul1")).astype(np.int64)) & 0xFFFFFFFF
        if mul != X.MUL1_MULT:
            raise ValueError(f"{base}: mul1 marker {mul:#x} != {X.MUL1_MULT:#x} (unsupported codebook)")
        for extra in (".mcg", ".bias", ".svh_bias"):
            if base + extra in st:
                raise ValueError(f"{base}: unsupported extra tensor {extra}")
        tk, tn, words = self.trellis.shape
        if words % 16:
            raise ValueError(f"{base}: trellis words {words}")
        self.K = words // 16
        self.k, self.n = tk * 16, tn * 16
        if self.k % 128 or self.n % 128:
            raise ValueError(f"{base}: padded/non-128 shape {self.k}x{self.n} not supported in v1")
        if self.suh.shape != (self.k,) or self.svh.shape != (self.n,):
            raise ValueError(f"{base}: suh/svh shape {self.suh.shape}/{self.svh.shape} vs {self.k}x{self.n}")
        if self.suh.dtype != np.float16 or self.svh.dtype != np.float16:
            raise ValueError(f"{base}: suh/svh not f16")
        if not (np.isfinite(self.suh).all() and np.isfinite(self.svh).all()):
            raise ValueError(f"{base}: nonfinite suh/svh")

    def native(self, vr):
        """band8 bytes + suh + svh with the V-head reorder applied as block permutations."""
        b8 = X.native_to_band8(np.asarray(self.trellis))
        suh, svh = self.suh.copy(), self.svh.copy()
        if vr is not None:
            side, blk0, nk, nv = vr
            if side == "out":
                perm = X.vhead_block_perm(self.n // 128, blk0, nk, nv)
                b8 = X.band8_permute_out(b8, perm)
                svh = svh.reshape(-1, 128)[perm].reshape(-1)
            else:
                perm = X.vhead_block_perm(self.k // 128, blk0, nk, nv)
                b8 = X.band8_permute_in(b8, perm)
                suh = suh.reshape(-1, 128)[perm].reshape(-1)
        return b8, suh, svh

    def vectors(self, vr):
        """suh, svh after the block permutations (no trellis work)."""
        suh, svh = self.suh.copy(), self.svh.copy()
        if vr is not None:
            side, blk0, nk, nv = vr
            if side == "out":
                svh = svh.reshape(-1, 128)[X.vhead_block_perm(self.n // 128, blk0, nk, nv)].reshape(-1)
            else:
                suh = suh.reshape(-1, 128)[X.vhead_block_perm(self.k // 128, blk0, nk, nv)].reshape(-1)
        return suh, svh

    def materialized(self, vr):
        """W [k, n] f16 with the reorder applied to plain rows/columns (independent of native())."""
        W = X.materialize(np.asarray(self.trellis), self.suh, self.svh, self.K)
        if vr is not None:
            side, blk0, nk, nv = vr
            if side == "out":
                perm = X.vhead_block_perm(self.n // 128, blk0, nk, nv)
                W = W.reshape(self.k, -1, 128)[:, perm].reshape(self.k, self.n)
            else:
                perm = X.vhead_block_perm(self.k // 128, blk0, nk, nv)
                W = W.reshape(-1, 128, self.n)[perm].reshape(self.k, self.n)
        return np.ascontiguousarray(W)


def native_roundtrip_ok(mx: Exl3Matrix, vr) -> bool:
    """materialize(band8->native after block perms) must equal materialized() bit for bit."""
    b8, suh, svh = mx.native(vr)
    W1 = X.materialize(X.band8_to_native(b8), suh, svh, mx.K)
    return np.array_equal(W1.view(np.uint16), mx.materialized(vr).view(np.uint16))


def ud_dequant(t) -> np.ndarray:
    """UD GGUF tensor -> float32 numpy in [.., n, k] order."""
    from gguf import quants
    arr = quants.dequantize(t.data, t.tensor_type)
    return np.asarray(arr, dtype=np.float32)


def corr_check(W_kn: np.ndarray, ud_nk: np.ndarray, what: str):
    """Per-128-output-block correlation of EXL3 W vs the UD dequant; a wrong mapping/perm -> ~0."""
    a = W_kn.astype(np.float32).T             # [n, k]
    b = ud_nk.reshape(a.shape)
    nb = a.shape[0] // 128
    cs = []
    for j in range(nb):
        x = a[j * 128:(j + 1) * 128].ravel()
        y = b[j * 128:(j + 1) * 128].ravel()
        cs.append(float(np.corrcoef(x, y)[0, 1]))
    cs = np.array(cs)
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    log(f"  check {what}: block corr min {cs.min():.4f} median {np.median(cs):.4f}; rel-L2 vs UD {rel:.4f}")
    if cs.min() < 0.8:
        raise ValueError(f"{what}: block correlation {cs.min():.3f} < 0.8 -- mapping or permutation wrong")
    return cs.min(), rel


# ---------------------------------------------------------------------------------------------
# GGUF plumbing
# ---------------------------------------------------------------------------------------------

def open_ud(first: str):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
    files = [first] if not m else [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, int(m.group(3)) + 1)]
    return [gguf.GGUFReader(f) for f in files]


def copy_kv(w: gguf.GGUFWriter, r0: gguf.GGUFReader, n_layer_new: int | None):
    n_layer_old = None
    for f in r0.fields.values():
        if f.name.endswith(".block_count"):
            n_layer_old = int(f.contents())
    for f in r0.fields.values():
        name = f.name
        if name.startswith("GGUF.") or name.startswith("split.") or name in ("general.architecture", "general.file_type", "general.name"):
            continue
        vtype = f.types[0]
        val = f.contents()
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        if n_layer_new is not None:
            if name.endswith(".block_count"):
                val = n_layer_new
            elif name in PER_LAYER_ARRAYS:
                assert len(val) == n_layer_old, name
                val = val[:n_layer_new]
        w.add_key_value(name, val, vtype, sub_type=sub)
    return n_layer_old


def tensor_bytes_view(t) -> np.ndarray:
    """ReaderTensor -> uint8 view with numpy byte shape (last dim = row bytes)."""
    ne = [int(x) for x in t.shape]
    data = np.asarray(t.data).view(np.uint8)
    rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    row_bytes = data.size // rows
    shape = list(reversed(ne[1:])) + [row_bytes]
    return data.reshape(shape)


class Plan:
    """Ordered (name, nbytes, raw_type, byte_shape, producer) list; producer() -> ndarray."""

    def __init__(self):
        self.items = []

    def add(self, name, byte_shape, raw_type, nbytes, producer):
        self.items.append((name, list(byte_shape), raw_type, nbytes, producer))


def write_gguf(path, kv_fn, plan: Plan, split_no: int, split_count: int, tensors_total: int):
    w = gguf.GGUFWriter(path, "qwen4exp")
    kv_fn(w)
    if split_count > 1:
        w.add_uint16("split.no", split_no)
        w.add_uint16("split.count", split_count)
        w.add_int32("split.tensors.count", tensors_total)
    for name, bshape, rtype, nbytes, _ in plan.items:
        w.add_tensor_info(name, bshape, np.dtype(np.uint8), nbytes, raw_dtype=rtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    t0 = time.time()
    done = 0
    for i, (name, bshape, rtype, nbytes, prod) in enumerate(plan.items):
        arr = prod()
        arr = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        if arr.nbytes != nbytes:
            raise ValueError(f"{name}: produced {arr.nbytes} bytes, planned {nbytes}")
        w.write_tensor_data(arr)
        done += nbytes
        if i % 50 == 0 or nbytes > (1 << 30):
            log(f"  [{i + 1}/{len(plan.items)}] {name} {nbytes / 2**20:.1f} MiB ({done / 2**30:.2f} GiB, {time.time() - t0:.0f}s)")
    w.close()


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exl3", default=DEF_EXL3)
    ap.add_argument("--ud", default=DEF_UD, help="first split of the source qwen4exp GGUF")
    ap.add_argument("--out", required=True, help="output directory (gets model-0000{1,2}-of-00002.gguf)")
    ap.add_argument("--mode", choices=("native", "materialize"), default="native")
    ap.add_argument("--layers", type=int, default=None, help="truncate to the first N layers (test artifacts)")
    ap.add_argument("--keep-ud-head", action="store_true", help="keep the UD output.weight instead of EXL3 lm_head")
    ap.add_argument("--ple", default="/mnt/raid0/llm/models/exl3-gguf/qwen4exp-ple-table-ud-iq4nl.split2of2.gguf",
                    help="shared split-2 file holding per_layer_token_embd (created if missing)")
    ap.add_argument("--check", action="store_true", help="correlate every EXL3 tensor against the UD tensor")
    ap.add_argument("--check-experts", default="0,511", help="experts sampled by --check")
    ap.add_argument("--check-only", action="store_true", help="build the plan and run --check, write nothing")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.exl3, "config.json")))
    hp = cfg["text_config"]
    qcfg = cfg.get("quantization_config", {})
    if qcfg.get("quant_method") != "exl3" or qcfg.get("codebook") != "mul1":
        raise SystemExit(f"unsupported quantization_config {qcfg}")
    st = X.SafeTensorsDir(args.exl3)
    log(f"exl3: {len(st.files)} safetensors files, {len(st.where)} tensors")
    readers = open_ud(args.ud)
    r0 = readers[0]
    ud = {}
    order = []
    for r in readers:
        for t in r.tensors:
            ud[t.name] = t
            order.append(t.name)
    n_layer_old = int(next(f for f in r0.fields.values() if f.name.endswith(".block_count")).contents())
    n_layer = args.layers or n_layer_old

    # EXL3 replacement map
    repl = {}
    for il in range(n_layer):
        for g, spec in layer_map(il, hp).items():
            if g + ".weight" in ud:       # GDN vs full-attention layers carry different sets
                repl[g] = spec
    hb = head_base(st)
    if not args.keep_ud_head:
        if hb is None:
            raise SystemExit("lm_head not found in the EXL3 checkpoint yet (use --keep-ud-head for tests)")
        repl["output"] = (hb, None)

    sample_experts = [int(x) for x in args.check_experts.split(",")] if (args.check or args.check_only) else []

    def keep(name):
        m = re.match(r"blk\.(\d+)\.", name)
        return not m or int(m.group(1)) < n_layer

    plan = Plan()
    ple_name = "per_layer_token_embd.weight"
    stats = {"exl3_bytes": 0, "copied_bytes": 0, "n_exl3": 0}
    for name in order:
        if not keep(name) or name == ple_name:
            continue
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        if base in repl:
            hfb, vr = repl[base]
            t_ud = ud[name]
            ne_ud = [int(x) for x in t_ud.shape]
            if isinstance(hfb, list):
                missing = [b for b in hfb if b + ".trellis" not in st]
                if missing:
                    raise SystemExit(f"{name}: {len(missing)} experts missing in the EXL3 checkpoint (e.g. {missing[0]})")
                m0 = Exl3Matrix(st, hfb[0])
                E = len(hfb)
                if ne_ud != [m0.k, m0.n, E]:
                    raise SystemExit(f"{name}: UD shape {ne_ud} vs EXL3 [{m0.k},{m0.n},{E}]")
                if (args.check or args.check_only):
                    for e in sample_experts:
                        mx = Exl3Matrix(st, hfb[e])
                        W_ud = np.asarray(gguf.quants.dequantize(np.asarray(t_ud.data)[e], t_ud.tensor_type), dtype=np.float32)
                        corr_check(mx.materialized(vr), W_ud, f"{name}[e{e}] K{mx.K}")
                        if args.mode == "native" and e == sample_experts[0]:
                            assert native_roundtrip_ok(mx, vr), f"{name}: band8 roundtrip mismatch"
                if args.mode == "native":
                    K = m0.K
                    row_bytes = m0.k // 16 * 2 * K
                    per = m0.n * row_bytes

                    def prod_tr(hfb=hfb, vr=vr, K=K, per=per):
                        out = np.empty(len(hfb) * per, dtype=np.uint8)
                        for e, b in enumerate(hfb):
                            mx = Exl3Matrix(st, b)
                            if mx.K != K:
                                raise ValueError(f"{b}: mixed K inside one expert tensor ({mx.K} vs {K})")
                            b8, _, _ = mx.native(vr)
                            out[e * per:(e + 1) * per] = b8.view(np.uint8).reshape(-1)
                        return out

                    def prod_vec(which, hfb=hfb, vr=vr):
                        rows = []
                        for b in hfb:
                            suh, svh = Exl3Matrix(st, b).vectors(vr)
                            rows.append(suh if which == "suh" else svh)
                        return np.stack(rows).astype(np.float16)

                    plan.add(name, [E, m0.n, row_bytes], EXL3_TYPES[K], E * per, prod_tr)
                    plan.add(base + ".exl3_suh", [E, m0.k * 2], QT.F16, E * m0.k * 2, lambda f=prod_vec: f("suh"))
                    plan.add(base + ".exl3_svh", [E, m0.n * 2], QT.F16, E * m0.n * 2, lambda f=prod_vec: f("svh"))
                    stats["exl3_bytes"] += E * per
                else:
                    def prod_mat(hfb=hfb, vr=vr, m0=m0):
                        out = np.empty((len(hfb), m0.n, m0.k), dtype=np.float16)
                        for e, b in enumerate(hfb):
                            out[e] = Exl3Matrix(st, b).materialized(vr).T
                        return out
                    plan.add(name, [E, m0.n, m0.k * 2], QT.F16, E * m0.n * m0.k * 2, prod_mat)
                stats["n_exl3"] += 1
            else:
                if hfb + ".trellis" not in st:
                    raise SystemExit(f"{name}: {hfb}.trellis missing in the EXL3 checkpoint")
                mx = Exl3Matrix(st, hfb)
                if ne_ud != [mx.k, mx.n]:
                    raise SystemExit(f"{name}: UD shape {ne_ud} vs EXL3 [{mx.k},{mx.n}]")
                if (args.check or args.check_only):
                    corr_check(mx.materialized(vr), ud_dequant(t_ud), f"{name} K{mx.K}")
                    if args.mode == "native":
                        assert native_roundtrip_ok(mx, vr), f"{name}: band8 roundtrip mismatch"
                if args.mode == "native":
                    row_bytes = mx.k // 16 * 2 * mx.K
                    plan.add(name, [mx.n, row_bytes], EXL3_TYPES[mx.K], mx.n * row_bytes,
                             lambda mx=mx, vr=vr: mx.native(vr)[0])
                    plan.add(base + ".exl3_suh", [mx.k * 2], QT.F16, mx.k * 2, lambda mx=mx, vr=vr: mx.vectors(vr)[0])
                    plan.add(base + ".exl3_svh", [mx.n * 2], QT.F16, mx.n * 2, lambda mx=mx, vr=vr: mx.vectors(vr)[1])
                    stats["exl3_bytes"] += mx.n * row_bytes
                else:
                    plan.add(name, [mx.n, mx.k * 2], QT.F16, mx.n * mx.k * 2, lambda mx=mx, vr=vr: mx.materialized(vr).T)
                stats["n_exl3"] += 1
            continue
        t = ud[name]
        bv = tensor_bytes_view(t)
        plan.add(name, bv.shape, t.tensor_type, bv.nbytes, lambda bv=bv: bv)
        stats["copied_bytes"] += bv.nbytes

    if stats["n_exl3"] != len(repl):
        raise SystemExit(f"replaced {stats['n_exl3']} tensors but the map has {len(repl)}")

    if args.check_only:
        log(f"check-only: {len(plan.items)} tensors planned, {stats['n_exl3']} EXL3, nothing written")
        return

    sha = {}
    if os.path.exists(DEF_FILES):
        for line in open(DEF_FILES):
            f, s = line.split()
            if f.endswith(".safetensors"):
                sha[f] = s
    # the model shards (the only files EXL3 tensors are read from; ngram/mtp files are not used)
    used = sorted({os.path.basename(v.path) for v in st.files.values()
                   if os.path.basename(v.path).startswith("model-") and v.path.endswith(".safetensors")})

    def kv(w):
        copy_kv(w, r0, args.layers)
        w.add_string("general.name", f"Qwen3.8 Flash Next EXL3 {qcfg.get('bits')}bpw ({args.mode}"
                     + (f", first {args.layers} layers" if args.layers else "") + ")")
        w.add_uint32("exl3.version", 1)
        w.add_string("exl3.mode", args.mode)
        w.add_string("exl3.codebook", "mul1")
        if args.mode == "native":
            w.add_string("exl3.layout", "band8-v1")
        w.add_string("exl3.source.repo", "turboderp/Qwen3.8-Flash-Next-exl3")
        w.add_string("exl3.source.branch", f"{qcfg.get('bits')}bpw")
        w.add_string("exl3.source.quantization_config", json.dumps(qcfg, sort_keys=True))
        w.add_array("exl3.source.shards", [f"{f}:{sha.get(f, '?')}" for f in used])
        w.add_string("exl3.source.trunk_gguf", os.path.basename(args.ud))
        w.add_bool("exl3.head_from_trunk", bool(args.keep_ud_head))

    os.makedirs(args.out, exist_ok=True)
    main_path = os.path.join(args.out, "model-00001-of-00002.gguf")
    ple_link = os.path.join(args.out, "model-00002-of-00002.gguf")
    if not os.path.exists(args.ple):
        log(f"writing shared PLE split {args.ple}")
        os.makedirs(os.path.dirname(args.ple), exist_ok=True)
        pp = Plan()
        bv = tensor_bytes_view(ud[ple_name])
        pp.add(ple_name, bv.shape, ud[ple_name].tensor_type, bv.nbytes, lambda bv=bv: bv)
        write_gguf(args.ple + ".tmp", lambda w: w.add_string("general.name", "qwen4exp PLE table (split 2 of 2)"),
                   pp, 1, 2, 1)
        os.rename(args.ple + ".tmp", args.ple)
    log(f"plan: {len(plan.items)} tensors, EXL3 {stats['n_exl3']} ({stats['exl3_bytes'] / 2**30:.2f} GiB packed), "
        f"copied {stats['copied_bytes'] / 2**30:.2f} GiB, mode {args.mode}, layers {n_layer}")
    write_gguf(main_path + ".tmp", kv, plan, 0, 2, len(plan.items) + 1)
    os.rename(main_path + ".tmp", main_path)
    if os.path.lexists(ple_link):
        os.remove(ple_link)
    os.symlink(os.path.abspath(args.ple), ple_link)
    log(f"done: {main_path} (+ {ple_link} -> {args.ple})")


if __name__ == "__main__":
    main()
