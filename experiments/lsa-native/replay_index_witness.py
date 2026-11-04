#!/usr/bin/env python3
"""Offline, model-free replay of one native HIP selection against the pinned author oracle.

The runner must independently pin exact rendered development token IDs. This
file contains no token IDs and cannot establish request identity by itself.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

ORACLE_SHA256 = "96e44dc399114e0709a0f6c0820c0ddcc2c62cb28d69f7a675389ae17db68ac2"
HEADER = struct.Struct("<8s9I2Q64s64s64s")
MAX_BYTES = HEADER.size + (64 + 5132 * 64) * 4 + 4612 * 4


def read_witness(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size or len(data) > MAX_BYTES:
        raise ValueError("witness size outside bounds")
    magic, count, capacity, length, layer, position, top_k, top_p, window, sink, slot, epoch, model, map_hash, map_source = HEADER.unpack_from(data)
    if magic != b"LSAWIT1\0" or count != 5132 or layer != 3 or position != 5131:
        raise ValueError("witness target mismatch")
    if not (0 <= length <= capacity <= 4612 and top_k + window + sink == capacity and top_p <= top_k):
        raise ValueError("witness selection bounds mismatch")
    if (top_k, top_p, window, sink) != (4096, 1024, 512, 4):
        raise ValueError("witness budgets differ from pinned author-oracle defaults")
    expected = HEADER.size + (64 + count * 64 + length) * 4
    if len(data) != expected:
        raise ValueError("witness payload length mismatch")
    digests = [v.decode("ascii") for v in (model, map_hash, map_source)]
    if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in digests):
        raise ValueError("invalid provenance digest")
    offset = HEADER.size
    query = struct.unpack_from("<64f", data, offset)
    offset += 64 * 4
    keys = struct.unpack_from(f"<{count * 64}f", data, offset)
    offset += count * 64 * 4
    indices = list(struct.unpack_from(f"<{length}i", data, offset))
    if not all(math.isfinite(v) for v in query + keys) or any(i < 0 or i >= count for i in indices):
        raise ValueError("nonfinite vector or invalid index")
    return {"query": query, "keys": keys, "indices": indices, "position": position,
            "budgets": [top_k, top_p, window, sink], "slot": slot, "epoch": epoch,
            "model_sha256": digests[0], "map_sha256": digests[1], "map_source_sha256": digests[2]}


def compare_rows(native, author):
    native, author = list(native), list(author)
    if len(native) != len(set(native)) or len(author) != len(set(author)):
        raise ValueError("duplicate selected index")
    ordered_equal = native == author
    set_equal = set(native) == set(author)
    disposition = ("ordered_exact" if ordered_equal else
                   "inconclusive_order_only" if set_equal else "membership_mismatch")
    return {"disposition": disposition, "ordered_equal": ordered_equal, "set_equal": set_equal,
            "native_indices": native, "author_indices": author,
            "native_length": len(native), "author_length": len(author),
            "first_order_mismatch": next((i for i, pair in enumerate(zip(native, author)) if pair[0] != pair[1]),
                                         min(len(native), len(author)) if len(native) != len(author) else None),
            "native_only": sorted(set(native) - set(author)),
            "author_only": sorted(set(author) - set(native))}


def disposition_exit_code(disposition):
    return {"ordered_exact": 0, "membership_mismatch": 1,
            "inconclusive_order_only": 2}[disposition]


def replay(witness, oracle_path):
    if hashlib.sha256(Path(oracle_path).read_bytes()).hexdigest() != ORACLE_SHA256:
        raise ValueError("author oracle digest mismatch")
    import torch
    spec = importlib.util.spec_from_file_location("lsa_author_oracle", oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    torch.set_num_threads(1)
    query = torch.tensor(witness["query"], dtype=torch.float32).reshape(1, 64)
    keys = torch.tensor(witness["keys"], dtype=torch.float32).reshape(5132, 64)
    positions = torch.tensor([witness["position"]], dtype=torch.int32)
    selected = oracle.select(query, keys, positions)[0]
    author = [int(v) for v in selected.tolist() if v >= 0]
    return compare_rows(witness["indices"], author)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("witness", type=Path)
    p.add_argument("--oracle", required=True, type=Path)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--map-sha256", required=True)
    p.add_argument("--map-source-sha256", required=True)
    args = p.parse_args()
    witness = read_witness(args.witness)
    for field in ("model_sha256", "map_sha256", "map_source_sha256"):
        if witness[field] != getattr(args, field):
            raise ValueError(f"{field} differs from external pin")
    result = replay(witness, args.oracle)
    print(json.dumps({"schema": "lsa-index-witness-replay-v1", "target": "layer3-position5131",
                      "witness_sha256": hashlib.sha256(args.witness.read_bytes()).hexdigest(),
                      **{k: witness[k] for k in ("model_sha256", "map_sha256", "map_source_sha256", "budgets", "slot", "epoch")},
                      **result}, sort_keys=True))
    return disposition_exit_code(result["disposition"])


if __name__ == "__main__":
    raise SystemExit(main())
