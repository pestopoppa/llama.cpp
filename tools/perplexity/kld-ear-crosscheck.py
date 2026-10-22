#!/usr/bin/env python3
"""Recompute llama-perplexity's EAR lines from saved --kl-divergence-base files (stdlib only).

    kld-ear-crosscheck.py BASE.kld QUANT.kld [--log KLD_STDOUT] [--max-positions N] [--tol T]

BASE.kld   the --kl-divergence-base file the KLD run compared against (p).
QUANT.kld  a --kl-divergence-base file written by the evaluated model over the SAME tokens (q).
--log      stdout of `llama-perplexity --kl-divergence` for the evaluated model against BASE.kld;
           its "EAR full vocab" / "EAR top10 renorm" lines are checked against the recomputation.

The C++ side sees q as exact float logits; the QUANT file stores q as uint16 log-probs whose codes
are clamped to 0 for tokens more than 16 nats below the mode.  The recomputation therefore brackets
the true value: clamped q entries contribute 0 (lower bound) or their floor value (upper bound), and
the encoding step (16/65535 nats) is added as tolerance.  PASS = the logged value lies in
[lo - tol, hi + tol] for both variants.  With BASE == QUANT the recomputation must give EAR = 1.

Exit: 0 pass, 1 mismatch, 2 bad input.
"""
from __future__ import annotations

import argparse
import array
import heapq
import math
import re
import struct
import sys

TOP_K = 10
QUANT_STEP = 16.0 / 65535.0


class KldFile:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        if self.f.read(8) != b"_logits_":
            raise ValueError(f"{path}: not a llama-perplexity log-prob file")
        self.n_ctx, self.n_vocab, self.n_chunk = struct.unpack("<Iii", self.f.read(12))
        self.tokens = self.f.read(4 * self.n_ctx * self.n_chunk)
        if len(self.tokens) != 4 * self.n_ctx * self.n_chunk:
            raise ValueError(f"{path}: truncated token block")
        self.nv = 2 * ((self.n_vocab + 1) // 2) + 4
        self.per_chunk = self.n_ctx - 1 - self.n_ctx // 2

    def records(self):
        rec_bytes = 2 * self.nv
        for _ in range(self.n_chunk * self.per_chunk):
            buf = self.f.read(rec_bytes)
            if len(buf) != rec_bytes:
                return
            scale, min_lp = struct.unpack("<ff", buf[:8])
            codes = array.array("H")
            codes.frombytes(buf[8:8 + 2 * self.n_vocab])
            if sys.byteorder != "little":
                codes.byteswap()
            yield scale, min_lp, codes


def position_ear(pa, pb, n_vocab):
    """Returns (full_lo, full_hi, top_lo, top_hi) for one position."""
    sa, ma, ca = pa
    sb, mb, cb = pb
    exp = math.exp
    full_lo = full_hi = 0.0
    for i in range(n_vocab):
        lp = sa * ca[i] + ma
        lq = sb * cb[i] + mb
        m = exp(lp if lp < lq else lq)
        full_hi += m
        if cb[i]:
            full_lo += m
    # base top-K, ties -> lower index (stable), as in kld-ear.h
    k = min(TOP_K, n_vocab)
    top = heapq.nlargest(k, range(n_vocab), key=ca.__getitem__)  # == sorted(reverse=True)[:k], stable
    lps = [sa * ca[i] + ma for i in top]
    lp_max = max(lps)
    zp = sum(exp(v - lp_max) for v in lps)
    p = [exp(v - lp_max) / zp for v in lps]
    outs = []
    for zero_clamped in (True, False):
        qs = [0.0 if (zero_clamped and cb[i] == 0) else exp(sb * cb[i] + mb) for i in top]
        zq = sum(qs)
        outs.append(sum(min(pk, qk / zq) for pk, qk in zip(p, qs)) if zq > 0 else 0.0)
    top_lo, top_hi = min(outs), max(outs)
    return min(full_lo, 1.0), min(full_hi, 1.0), top_lo, top_hi


def parse_log(path: str):
    vals = {}
    pat = re.compile(r"^EAR (full vocab|top\d+ renorm)\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)\s*%")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                vals["full" if m.group(1) == "full vocab" else "top"] = float(m.group(2)) / 100.0
    return vals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("quant")
    ap.add_argument("--log")
    ap.add_argument("--max-positions", type=int, default=0, help="0 = all (the log is only comparable with all)")
    ap.add_argument("--tol", type=float, default=QUANT_STEP)
    a = ap.parse_args(argv)
    try:
        fa, fb = KldFile(a.base), KldFile(a.quant)
    except (OSError, ValueError, struct.error) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    if (fa.n_ctx, fa.n_vocab, fa.n_chunk) != (fb.n_ctx, fb.n_vocab, fb.n_chunk) or fa.tokens != fb.tokens:
        print("bad input: the two files do not cover the same tokens (n_ctx/n_vocab/n_chunk/tokens differ)", file=sys.stderr)
        return 2
    n = 0
    s = [0.0, 0.0, 0.0, 0.0]
    for pa, pb in zip(fa.records(), fb.records()):
        for j, v in enumerate(position_ear(pa, pb, fa.n_vocab)):
            s[j] += v
        n += 1
        if a.max_positions and n >= a.max_positions:
            break
    if n == 0:
        print("bad input: no positions", file=sys.stderr)
        return 2
    full_lo, full_hi, top_lo, top_hi = (x / n for x in s)
    print(f"positions={n} n_vocab={fa.n_vocab}")
    print(f"recomputed EAR full vocab  : [{100*full_lo:.6f}, {100*full_hi:.6f}] %")
    print(f"recomputed EAR top{TOP_K} renorm: [{100*top_lo:.6f}, {100*top_hi:.6f}] %")
    if a.base == a.quant or fa.f.name == fb.f.name:
        ok = full_hi >= 1.0 - a.tol and top_hi >= 1.0 - a.tol
        print(f"self-check EAR(base, base) == 1: {'pass' if ok else 'FAIL'}")
        return 0 if ok else 1
    if not a.log:
        return 0
    logged = parse_log(a.log)
    if set(logged) != {"full", "top"}:
        print(f"bad input: EAR lines not found in {a.log}", file=sys.stderr)
        return 2
    ok = True
    for key, lo, hi in (("full", full_lo, full_hi), ("top", top_lo, top_hi)):
        v = logged[key]
        good = lo - a.tol <= v <= hi + a.tol
        ok &= good
        print(f"logged {key:4s} {100*v:.6f} % in [{100*(lo-a.tol):.6f}, {100*(hi+a.tol):.6f}] %: {'pass' if good else 'FAIL'}")
    print(f"crosscheck: {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
