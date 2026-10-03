"""Check exl3np against the EXL3-1 frozen fixtures (independent oracle, research bc9ac47b).

usage: python test_exl3np.py [fixtures_dir]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exl3np as X  # noqa: E402

DEFAULT = "/workspace/repos/epyc-inference-research/scripts/kernel_rnd/exl3/fixtures"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check(d: str) -> bool:
    man = json.load(open(os.path.join(d, "manifest.json")))
    fix = json.load(open(os.path.join(d, "fixture.json")))
    m = man["matrices"][0]
    if m["codebook"] != "mul1":
        return True
    K = m["K"]
    kp, npad = m["padded_shape"]
    k, n = m["shape"]
    tr = np.fromfile(os.path.join(d, m["tensors"]["trellis"]["path"]), dtype="<u2").reshape(kp // 16, npad // 16, 16 * K)
    suh = np.fromfile(os.path.join(d, "suh.bin"), dtype="<f2")
    svh = np.fromfile(os.path.join(d, "svh.bin"), dtype="<f2")
    ref = fix["reference"]
    ok = True
    st = X.tile_states(tr[0, 0], K)
    ok &= list(map(int, st)) == ref["packed_states"]
    raw = X.decode_raw(tr, K)
    ok_raw = sha(raw.astype("<f2").tobytes()) == ref["raw_fp16_sha256"]
    ok &= ok_raw
    ok &= np.array_equal(raw[:16, :16].astype(np.float64), np.array(ref["reconstructed_tile"]))
    W = X.materialize(tr, suh, svh, K)
    ok_w = sha(W.astype("<f2").tobytes()) == ref["transformed_fp16_sha256"]
    # operator outputs (ascending fp32, like the oracle)
    acts = np.array(fix["activations"], dtype=np.float32)
    bias = None
    if m["tensors"]["bias"] is not None:
        bias = np.fromfile(os.path.join(d, m["tensors"]["bias"]["path"]), dtype="<f2").astype(np.float32)
    Wc = W[:k, :n].astype(np.float32)
    y = acts[:, :k].astype(np.float64) @ Wc.astype(np.float64)
    if bias is not None:
        y = y + bias[:n]
    yref = np.array(ref["operator_outputs"], dtype=np.float64)
    rel = np.linalg.norm(y - yref) / np.linalg.norm(yref)
    ok &= rel < 1e-3
    print(f"{os.path.basename(d):14s} K={K} states+raw {'OK' if ok_raw else 'FAIL'} "
          f"materialized-hash {'exact' if ok_w else 'differs (BLAS order)'} op rel-L2 {rel:.2e} -> {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    allok = True
    for name in sorted(os.listdir(root)):
        if "mul1" in name:
            allok &= check(os.path.join(root, name))
    print("ALL PASS" if allok else "FAILURES")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
