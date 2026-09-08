#!/usr/bin/env python3
"""Capture native-MTP counters and P-BENCH-4-shaped server responses.

The enclosing launcher owns server lifecycle, locks, linkage, affinity, and
contention evidence.  This client only issues fixed requests and preserves raw
responses.  Its five-request speed block is an observation unless the exact
ratified P-BENCH-4 instrument and host gates attest it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import urllib.request
from pathlib import Path


def post(port: int, route: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def response_row(kind: str, ordinal: int, raw: bytes, doc: dict) -> dict:
    timings = doc.get("timings") or {}
    choice = (doc.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or doc.get("stop_type")
    row = {
        "kind": kind,
        "ordinal": ordinal,
        "predicted_n": timings.get("predicted_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "draft_n": timings.get("draft_n", 0),
        "draft_n_accepted": timings.get("draft_n_accepted", 0),
        "finish_reason": finish,
        "canonical_response_sha256": digest_bytes(raw),
    }
    for key in ("predicted_n", "predicted_per_second"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"response missing numeric timings.{key}")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise RuntimeError(f"invalid timings.{key}={value!r}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--prompt", default="Explain how a CPU cache hierarchy works, with a concrete example.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    base = {"prompt": args.prompt, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": False, "stream": False,
            "ignore_eos": True, "return_tokens": True}
    (args.out / "request-base.json").write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    rows = []
    ordinal = 0
    warm_rates = []
    for attempt in range(1, 9):
        ordinal += 1
        body = {**base, "n_predict": 64}
        (args.out / f"request-warmup-{attempt}.json").write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n")
        doc = post(args.port, "/completion", body, args.timeout)
        raw = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (args.out / f"response-warmup-{attempt}.json").write_bytes(raw)
        row = response_row("warmup", ordinal, raw, doc)
        rows.append(row)
        warm_rates.append(float(row["predicted_per_second"]))
        if len(warm_rates) >= 3:
            tail = warm_rates[-3:]
            med = statistics.median(tail)
            if max(abs(x - med) / med for x in tail) <= 0.05:
                break
    else:
        raise RuntimeError("warmup did not produce three consecutive rates within 5%")

    for rep in range(1, 6):
        ordinal += 1
        body = {**base, "n_predict": 512}
        (args.out / f"request-measure-{rep}.json").write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n")
        doc = post(args.port, "/completion", body, args.timeout)
        raw = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (args.out / f"response-measure-{rep}.json").write_bytes(raw)
        rows.append(response_row("measure", ordinal, raw, doc))
    (args.out / "requests.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"requests": len(rows), "drafted": sum(int(r["draft_n"] or 0) for r in rows),
                      "accepted": sum(int(r["draft_n_accepted"] or 0) for r in rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
