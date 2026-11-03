"""Slow isolated reference mask worker. Protocol is private to the experimental graph."""
import os
import struct
import sys
from pathlib import Path

import torch

from reference import load_maps, project, select


def compute(data, layer, mode, maps):
    n_query, n_kv = struct.unpack_from("=II", data)
    floats = torch.frombuffer(data, dtype=torch.float32, offset=8)
    q_end = n_query * 24 * 256
    k_end = q_end + n_kv * 4 * 256
    q = floats[:q_end].reshape(n_query, 24, 256)
    k = floats[q_end:k_end].reshape(n_kv, 4, 256)
    mask = floats[k_end:].reshape(n_query, n_kv).clone()
    positions = []
    for row in mask:
        allowed = torch.isfinite(row)
        n = int(allowed.sum())
        if n == 0 or not allowed[:n].all() or allowed[n:].any() or (row[allowed] != 0).any():
            raise ValueError("LSA requires one contiguous causal sequence starting at slot zero")
        positions.append(n - 1)
    if mode == "positional":
        mask.fill_(-float("inf"))
        for row, position in enumerate(positions):
            start = max(0, position + 1 - 512)
            mask[row, max(4, start - 4096):position + 1] = 0
            mask[row, :min(4, position + 1)] = 0
    elif mode != "all":
        iq, ik = project(q, k[:max(positions) + 1], maps[f"{layer}.W_Q_idx"], maps[f"{layer}.W_K_idx"])
        indices = select(iq, ik, torch.tensor(positions))
        mask.fill_(-float("inf"))
        for row, keys in enumerate(indices):
            mask[row, keys[keys >= 0].long()] = 0
    return mask.numpy().tobytes()


def read_exact(stream, length):
    chunks = []
    while length:
        part = stream.read(length)
        if not part:
            raise EOFError("short worker input")
        chunks.append(part)
        length -= len(part)
    return b"".join(chunks)


def main():
    torch.set_num_threads(int(os.environ.get("LSA_THREADS", "1")))
    if sys.argv[1] != "--serve":
        source, dest, layer, mode, map_path = sys.argv[1:]
        maps = load_maps(map_path, mode) if mode in ("pca", "distilled") else None
        Path(dest).write_bytes(compute(bytearray(Path(source).read_bytes()), layer, mode, maps))
        return
    mode, map_path = sys.argv[2:]
    maps = load_maps(map_path, mode) if mode in ("pca", "distilled") else None
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = inp.read(12)
        if not header:
            return
        try:
            if len(header) != 12:
                raise EOFError("short worker header")
            nq, nk, layer = struct.unpack("=III", header)
            if not nq or not nk or nq > nk or layer not in range(3, 64, 4):
                raise ValueError("invalid worker dimensions or layer")
            size = (nq * 24 * 256 + nk * 4 * 256 + nq * nk) * 4
            payload = read_exact(inp, size)
            result = compute(bytearray(struct.pack("=II", nq, nk) + payload), str(layer), mode, maps)
            out.write(struct.pack("=i", 0) + result)
            out.flush()
        except Exception as exc:
            print(f"LSA worker: {exc}", file=sys.stderr, flush=True)
            out.write(struct.pack("=i", -1))
            out.flush()
            return


if __name__ == "__main__":
    main()
