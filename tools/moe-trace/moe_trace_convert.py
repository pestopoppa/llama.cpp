#!/usr/bin/env python3
"""Verify and convert llama-moe-trace runs (epyc.moe.routing_trace.v1). INF-72 RT-1.

Port of routing-lab/06_convert_gguf_trace.py (MIT, Copyright (c) 2026 Shriniwas Ramesh Suram,
github.com/Shriniwas410/cacheable-by-design @ 03366ef1). That script converted a raw int32 record stream
to an (L, T, k) npz; here the tool already writes the RT-2.1 contract directly, so this script
  * `verify`  proves a run directory is a valid, untampered, hazard-free trace, and
  * `npz`     keeps the original job: (L, T, k) int16 + meta, for routing-lab 02/03/04.

`verify` is the offline half of RT-3: it re-runs the stride-hazard guard on the STORED ids (every
expert appearing exactly the same number of times in a prefill forward, near-uniform counts with
sub-chance reuse), so a run produced by any other tool, or an old run, cannot be trusted by default.

Layout (see tools/moe-trace/moe-trace-core.h, mirrored by trace_shape.load_run in
epyc-inference-research/scripts/kernel_rnd/moe_routing_tap):
  routing_trace.json         manifest, schema epyc.moe.routing_trace.v1
  routing_trace.rows.bin     16-byte rows: u32 sample_id, u32 pos, u32 step, u16 spec_block, u8 phase, u8 spec_accepted
  routing_trace.ids.i16      int16 [rows, n_layers, max_top_k], -1 pad

Usage:
  python3 moe_trace_convert.py verify RUN_DIR [--no-guard]
  python3 moe_trace_convert.py npz RUN_DIR --out FILE.npz
  python3 moe_trace_convert.py summary RUN_DIR
Exit: 0 ok, 1 problems found, 2 usage / unreadable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA_ID = "epyc.moe.routing_trace.v1"
COLUMNS = ["sample_id", "pos", "phase", "step", "spec_block", "spec_accepted"]
ROW_DTYPE = np.dtype([("sample_id", "<u4"), ("pos", "<u4"), ("step", "<u4"),
                      ("spec_block", "<u2"), ("phase", "u1"), ("acc", "u1")])
assert ROW_DTYPE.itemsize == 16
PREFILL, DECODE, ACC_NULL = 0, 1, 255
FILES = ("routing_trace.json", "routing_trace.rows.bin", "routing_trace.ids.i16")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_run(run_dir):
    """-> (manifest dict, rows structured array [R], ids int16 array [R, n_layers, max_k])"""
    d = Path(run_dir)
    m = json.loads((d / FILES[0]).read_text())
    rows = np.fromfile(d / FILES[1], dtype=ROW_DTYPE)
    nl = len(m["layers"])
    mk = max(l["top_k"] for l in m["layers"])
    flat = np.fromfile(d / FILES[2], dtype="<i2")
    if flat.size != rows.size * nl * mk:
        raise ValueError(f"ids.i16 holds {flat.size} int16, expected {rows.size}*{nl}*{mk}={rows.size * nl * mk}")
    return m, rows, flat.reshape(rows.size, nl, mk)


def check_manifest(m: dict) -> list[str]:
    p: list[str] = []
    if m.get("schema") != SCHEMA_ID:
        p.append(f"schema id {m.get('schema')!r}")
    for k in ("model", "arch", "batch_size", "layers", "samples", "columns", "ids", "rows_sha256"):
        if k not in m:
            p.append(f"manifest missing {k!r}")
    if p:
        return p
    layers = m["layers"]
    if [l.get("layer") for l in layers] != list(range(len(layers))):
        p.append("layers must be listed in order 0..n-1")
    for l in layers:
        if l.get("id_dtype") != "int16":
            p.append(f"layer {l.get('layer')}: id_dtype must be int16, not {l.get('id_dtype')!r}")
        if not 1 <= l.get("top_k", 0) <= l.get("n_experts", 0):
            p.append(f"layer {l.get('layer')}: top_k outside 1..n_experts")
        if l.get("n_experts", 0) > 32767:
            p.append(f"layer {l.get('layer')}: n_experts exceeds int16")
    if m["columns"] != COLUMNS:
        p.append("columns must be exactly " + ",".join(COLUMNS))
    if (m["ids"] or {}).get("dtype") != "int16":
        p.append("ids.dtype must be int16")
    nxt = 0
    for s in sorted(m["samples"], key=lambda y: y.get("start_row", 0)):
        if s.get("start_row") != nxt:
            p.append(f"sample {s.get('sample_id')}: rows not contiguous at {nxt}")
        nxt = s.get("start_row", 0) + s.get("n_rows", 0)
    return p


def check_hashes(d: Path, m: dict) -> list[str]:
    p = []
    if sha256_file(d / FILES[1]) != m["rows_sha256"]:
        p.append("rows.bin sha256 does not match the manifest")
    if sha256_file(d / FILES[2]) != m["ids"]["sha256"]:
        p.append("ids.i16 sha256 does not match the manifest")
    return p


def check_rows(m: dict, rows: np.ndarray) -> list[str]:
    p: list[str] = []
    total = sum(s["n_rows"] for s in m["samples"])
    if rows.size != total:
        return [f"rows.bin holds {rows.size} rows, the sample boundaries declare {total}"]
    for s in m["samples"]:
        seg = rows[s["start_row"]:s["start_row"] + s["n_rows"]]
        sid = s["sample_id"]
        if (seg["sample_id"] != sid).any():
            p.append(f"sample {sid}: foreign row inside its boundary")
        if seg.size > 1 and (np.diff(seg["pos"].astype(np.int64)) <= 0).any():
            p.append(f"sample {sid}: token order broken (pos not strictly increasing)")
        if (seg["phase"] > DECODE).any():
            p.append(f"sample {sid}: phase value outside prefill/decode")
        if seg.size > 1 and (np.diff(seg["phase"].astype(np.int8)) < 0).any():
            p.append(f"sample {sid}: prefill after decode")
        if seg.size > 1 and (np.diff(seg["step"].astype(np.int64)) < 0).any():
            p.append(f"sample {sid}: step not monotonic")
            continue
        _, idx, cnt = np.unique(seg["step"], return_index=True, return_counts=True)
        blk = np.repeat(cnt, cnt)
        if (seg["spec_block"] != blk).any():
            p.append(f"sample {sid}: spec_block != rows in the forward")
        if (np.maximum.reduceat(seg["phase"], idx) != np.minimum.reduceat(seg["phase"], idx)).any():
            p.append(f"sample {sid}: mixed phases in one forward")
        if ((seg["phase"] == PREFILL) & (seg["acc"] != ACC_NULL)).any():
            p.append(f"sample {sid}: prefill row carries spec_accepted")
        if ((seg["spec_block"] == 1) & (seg["acc"] != ACC_NULL)).any():
            p.append(f"sample {sid}: spec_accepted on a non-speculative row")
        if ((seg["acc"] != ACC_NULL) & (seg["acc"] > 1)).any():
            p.append(f"sample {sid}: bad spec_accepted value")
    return p


def check_ids(m: dict, ids: np.ndarray) -> list[str]:
    p: list[str] = []
    for li, l in enumerate(m["layers"]):
        k, ne = l["top_k"], l["n_experts"]
        sel = ids[:, li, :k].astype(np.int32)
        if (sel < 0).any() or (sel >= ne).any():
            p.append(f"layer {li}: expert id outside 0..{ne - 1}")
            continue
        if k > 1 and (np.diff(np.sort(sel, axis=1), axis=1) == 0).any():
            p.append(f"layer {li}: a row selects the same expert twice (need {k} distinct)")
        if k < ids.shape[2] and (ids[:, li, k:] != -1).any():
            p.append(f"layer {li}: slots beyond top_k={k} must be -1")
    return p


def hazard(sel: np.ndarray, n_experts: int) -> dict:
    """Mirror of moe_trace::hazard_check (C++). sel: [n_tokens, k] for one layer of one forward."""
    n, k = sel.shape
    r = {"evaluated": False, "fatal": False, "uniform": False, "sub_chance": False, "why": "",
         "chance": k / n_experts if n_experts else 0.0}
    if n_experts < 16 or k * 4 > n_experts or n < 8 or n * k < 2 * n_experts:
        return r
    r["evaluated"] = True
    cnt = np.bincount(sel.ravel().astype(np.int64), minlength=n_experts)
    mean = n * k / n_experts
    cv = float(cnt.std() / mean) if mean > 0 else 0.0
    ov = (sel[1:, :, None] == sel[:-1, None, :]).any(axis=2).sum()
    ir = float(ov) / ((n - 1) * k)
    r.update(count_min=int(cnt.min()), count_max=int(cnt.max()), mean=mean, cv=cv, ir_mean=ir)
    r["uniform"] = int(cnt.max() - cnt.min()) <= 1
    near = cv < 0.10 and mean >= 4.0
    r["sub_chance"] = ir < r["chance"]
    r["fatal"] = r["uniform"] or (near and r["sub_chance"])
    if r["fatal"]:
        r["why"] = (f"STRIDE-HAZARD SIGNATURE: per-expert counts min={cnt.min()} max={cnt.max()} (mean {mean:.2f}, cv {cv:.3f})"
                    f"{' = every expert selected the same number of times' if r['uniform'] else ''}; "
                    f"adjacent-token reuse IR_t={ir:.4f} vs chance {r['chance']:.4f}{' = SUB-CHANCE' if r['sub_chance'] else ''}")
    return r


def check_guard(m: dict, rows: np.ndarray, ids: np.ndarray) -> list[str]:
    """RT-3 offline: every prefill forward, every layer."""
    p: list[str] = []
    evaluated = 0
    for s in m["samples"]:
        seg = rows[s["start_row"]:s["start_row"] + s["n_rows"]]
        base = s["start_row"]
        if seg.size == 0:
            continue
        _, idx, cnt = np.unique(seg["step"], return_index=True, return_counts=True)
        for i0, c in zip(idx, cnt):
            if seg["phase"][i0] != PREFILL:
                continue
            for li, l in enumerate(m["layers"]):
                h = hazard(ids[base + i0:base + i0 + c, li, :l["top_k"]], l["n_experts"])
                evaluated += h["evaluated"]
                if h["fatal"]:
                    p.append(f"sample {s['sample_id']} step {int(seg['step'][i0])} layer {li}: {h['why']}")
                    if len(p) >= 8:
                        return p
    if evaluated == 0:
        p.append("hazard guard evaluated NOTHING (no prefill forward with >= 2*n_experts selections): "
                 "the run cannot be shown free of the stride hazard")
    return p


def verify(run_dir, guard: bool = True) -> list[str]:
    d = Path(run_dir)
    for f in FILES:
        if not (d / f).is_file():
            return [f"missing {f} (an interrupted or failed run has no manifest)"]
    m = json.loads((d / FILES[0]).read_text())
    p = check_manifest(m)
    if p:
        return p
    p += check_hashes(d, m)
    try:
        m, rows, ids = load_run(d)
    except ValueError as e:
        return p + [str(e)]
    p += check_rows(m, rows)
    if not p:
        p += check_ids(m, ids)
    if guard and not p:
        p += check_guard(m, rows, ids)
    return p


def to_npz(run_dir, out: str) -> str:
    """Original 06_convert job: experts (L, T, k) int16 + meta json. Needs one top_k across layers."""
    m, rows, ids = load_run(run_dir)
    ks = {l["top_k"] for l in m["layers"]}
    if len(ks) != 1:
        raise ValueError(f"layers have different top_k {sorted(ks)}; npz (L,T,k) needs one")
    k = ks.pop()
    experts = np.ascontiguousarray(np.transpose(ids[:, :, :k], (1, 0, 2))).astype(np.int16)
    meta = {"model": m["model"], "arch": m["arch"], "top_k": k, "k": k,
            "num_experts": max(l["n_experts"] for l in m["layers"]),
            "n_experts_per_layer": [l["n_experts"] for l in m["layers"]],
            "n_moe_layers": len(m["layers"]), "rows": int(rows.size),
            "phase": rows["phase"].astype(np.uint8), "sample_id": rows["sample_id"], "pos": rows["pos"], "step": rows["step"],
            "source": "llama-moe-trace eval-callback (GGUF, weights untouched), epyc.moe.routing_trace.v1"}
    arrays = {k_: v for k_, v in meta.items() if isinstance(v, np.ndarray)}
    meta = {k_: v for k_, v in meta.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(out, experts=experts, meta=json.dumps(meta), **arrays)
    return f"{out}: experts {experts.shape}, num_experts={meta['num_experts']}"


def summary(run_dir) -> str:
    m, rows, ids = load_run(run_dir)
    n_pre = int((rows["phase"] == PREFILL).sum())
    lines = [f"model {m['model']}  arch {m['arch']}  batch {m['batch_size']}",
             f"rows {rows.size} (prefill {n_pre}, decode {rows.size - n_pre}) in {len(m['samples'])} samples; "
             f"{len(m['layers'])} layers, n_experts {sorted({l['n_experts'] for l in m['layers']})}, top_k {sorted({l['top_k'] for l in m['layers']})}"]
    eor = Path(run_dir) / "routing_trace.eor.json"
    if eor.is_file():
        e = json.loads(eor.read_text())
        lines.append(f"IR_t macro mean over layers: {e['macro_mean_over_layers']}  (chance K/N per layer in the sidecar)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("run_dir"); v.add_argument("--no-guard", action="store_true")
    n = sub.add_parser("npz"); n.add_argument("run_dir"); n.add_argument("--out", required=True)
    s = sub.add_parser("summary"); s.add_argument("run_dir")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "verify":
            probs = verify(a.run_dir, guard=not a.no_guard)
            if probs:
                print(f"{a.run_dir}: INVALID", file=sys.stderr)
                for x in probs:
                    print("  - " + x, file=sys.stderr)
                return 1
            print(f"{a.run_dir}: ok" + ("" if a.no_guard else " (hash-bound, schema-valid, hazard guard passed)"))
            return 0
        if a.cmd == "npz":
            probs = verify(a.run_dir)
            if probs:
                print("refusing to convert an invalid run:\n  - " + "\n  - ".join(probs), file=sys.stderr)
                return 1
            print(to_npz(a.run_dir, a.out))
            return 0
        print(summary(a.run_dir))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"unreadable run: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
