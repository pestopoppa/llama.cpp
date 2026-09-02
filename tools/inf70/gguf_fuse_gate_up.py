#!/usr/bin/env python3
"""Offline fusion of per-layer MoE ffn_gate_exps + ffn_up_exps into ffn_gate_up_exps.

Rewrites a GGUF, copying every KV field and every tensor byte-for-byte, except that for
each layer the two split expert tensors

    blk.N.ffn_gate_exps.weight   ne = [n_embd, n_ff, n_expert]
    blk.N.ffn_up_exps.weight     ne = [n_embd, n_ff, n_expert]

are replaced by a single

    blk.N.ffn_gate_up_exps.weight   ne = [n_embd, n_ff*2, n_expert]

whose rows are, per expert, the n_ff gate rows followed by the n_ff up rows (concatenation
along ne[1]).  This is the layout `create_tensor_gate_up_exps` (src/llama-model.cpp) loads
and the merged branch of `build_moe_ffn` (src/llama-graph.cpp) consumes: it runs one
mul_mat_id producing [n_ff*2, n_expert_used, n_tokens] and takes gate = view at offset 0,
up = view at offset n_ff*nb[0].  It matches the order produced online by
convert_hf_to_gguf.py --fuse-gate-up-exps (conversion/base.py: torch.cat([gate, up], dim=1)
on tensors shaped (n_expert, n_ff, n_embd)).

Both source tensors must have the same ggml type and shape; the concatenation is a pure
byte-level splice of whole quantized rows, so the result is bit-exact.

Data is streamed: nothing larger than one expert's rows is held in memory.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402

logger = logging.getLogger("gguf-fuse-gate-up")

GATE_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")
UP_RE = re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$")


def align(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    reader = GGUFReader(args.input, "r")

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    assert arch_field is not None, "input has no general.architecture"
    arch = arch_field.contents()
    logger.info("arch=%s tensors=%d fields=%d", arch, len(reader.tensors), len(reader.fields))

    by_name = {t.name: t for t in reader.tensors}
    gates = {int(m.group(1)): t for t in reader.tensors if (m := GATE_RE.match(t.name))}
    ups = {int(m.group(1)): t for t in reader.tensors if (m := UP_RE.match(t.name))}
    if set(gates) != set(ups):
        raise SystemExit(f"gate/up layer sets differ: {sorted(set(gates) ^ set(ups))}")
    if not gates:
        raise SystemExit("no blk.N.ffn_{gate,up}_exps.weight tensors found")
    logger.info("fusing %d layers: %s", len(gates), sorted(gates))

    for bid in sorted(gates):
        g, u = gates[bid], ups[bid]
        if g.tensor_type != u.tensor_type:
            raise SystemExit(f"layer {bid}: type mismatch {g.tensor_type.name} vs {u.tensor_type.name}")
        if tuple(g.shape) != tuple(u.shape):
            raise SystemExit(f"layer {bid}: shape mismatch {list(g.shape)} vs {list(u.shape)}")
        if len(g.shape) != 3:
            raise SystemExit(f"layer {bid}: expected a 3-D expert tensor, got {list(g.shape)}")
        if g.data.ndim != 3:
            raise SystemExit(f"layer {bid}: expected 3-D raw data, got {g.data.shape}")

    # ---- key/value metadata: copy verbatim -------------------------------------------
    writer = GGUFWriter(args.output, arch, use_temp_file=False)
    n_kv = 0
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue  # written by GGUFWriter itself
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if val is None:
            logger.warning("field %s has no readable contents, skipping", field.name)
            continue
        writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
        n_kv += 1
    logger.info("copied %d kv fields (+ general.architecture)", n_kv)

    # ---- tensor info -----------------------------------------------------------------
    # The fused tensor takes the gate tensor's slot in the tensor order; the up tensor is dropped.
    plan: list[tuple[str, object]] = []  # (kind, payload)
    for t in reader.tensors:
        m = GATE_RE.match(t.name)
        if m is not None:
            bid = int(m.group(1))
            g, u = gates[bid], ups[bid]
            # raw byte shape, numpy (reversed-ne) order: (n_expert, n_ff, row_bytes)
            byte_shape = (g.data.shape[0], g.data.shape[1] + u.data.shape[1], g.data.shape[2])
            nbytes = g.data.nbytes + u.data.nbytes
            name = f"blk.{bid}.ffn_gate_up_exps.weight"
            writer.add_tensor_info(name, byte_shape, g.data.dtype, nbytes, raw_dtype=g.tensor_type)
            plan.append(("fuse", bid))
            continue
        if UP_RE.match(t.name) is not None:
            continue  # folded into the fused tensor
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
        plan.append(("copy", t.name))

    n_out = sum(len(d) for d in writer.tensors)
    logger.info("output tensor count %d (input %d, -%d)", n_out, len(reader.tensors), len(reader.tensors) - n_out)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    fout = writer.fout[0]
    alignment = writer.data_alignment
    total = 0
    for kind, payload in plan:
        # align the start of every tensor's data exactly like GGUFWriter.write_tensor_data
        pad = align(fout.tell(), alignment) - fout.tell()
        if pad:
            fout.write(b"\0" * pad)
        if kind == "copy":
            t = by_name[payload]
            t.data.tofile(fout)
            n = t.data.nbytes
        else:
            bid = payload
            g, u = gates[bid], ups[bid]
            gd, ud = g.data, u.data
            n_expert = gd.shape[0]
            for e in range(n_expert):
                gd[e].tofile(fout)
                ud[e].tofile(fout)
            n = gd.nbytes + ud.nbytes
            logger.info("layer %d fused: %s %s + %s -> %s (%d bytes)", bid, g.tensor_type.name,
                        list(g.shape), list(u.shape),
                        [int(g.shape[0]), int(g.shape[1]) * 2, int(g.shape[2])], n)
        total += n
        pad = align(n, alignment) - n
        if pad:
            fout.write(b"\0" * pad)
    writer.flush()
    writer.close()
    logger.info("wrote %d tensor bytes to %s (%d bytes on disk)", total, args.output, args.output.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
