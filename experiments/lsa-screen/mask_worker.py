"""Slow isolated reference mask worker. Protocol is private to the experimental graph."""
import os
import math
import struct
import sys
from pathlib import Path

import torch

from reference import load_maps, project, select

ORACLE_TOP_K = 4096
ORACLE_WINDOW = 512
ORACLE_SINK = 4


def oracle_mass(q, k, position, scale, softcap):
    """Mean of per-query-head dense attention probabilities over the causal prefix."""
    if q.shape != (24, 256) or k.ndim != 3 or k.shape[1:] != (4, 256):
        raise ValueError("oracle Q/K geometry")
    if not (0 <= position < len(k)) or not math.isfinite(scale) or scale <= 0 or not math.isfinite(softcap) or softcap < 0:
        raise ValueError("oracle position or attention configuration")
    if not torch.isfinite(q).all() or not torch.isfinite(k[:position + 1]).all():
        raise ValueError("oracle nonfinite Q/K")
    end = position + 1
    # Six query heads share each KV head. The model's softcap acts on scaled
    # logits before the per-head softmax, matching ggml_flash_attn_ext.
    logits = torch.einsum('ghd,kgd->ghk', q.float().reshape(4, 6, 256), k[:end].float()) * scale
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    return torch.softmax(logits.reshape(24, end), dim=-1).mean(dim=0)


def oracle_indices(q, k, position, scale, softcap):
    """Physical-RoPE dense teacher; top distant keys + causal window + sinks."""
    mass = oracle_mass(q, k, position, scale, softcap)
    end = position + 1
    start = max(0, end - ORACLE_WINDOW)
    distant = mass[ORACLE_SINK:start]
    rank = torch.argsort(distant, descending=True, stable=True)[:ORACLE_TOP_K] + ORACLE_SINK
    window = torch.arange(start, end)
    sink = torch.arange(min(ORACLE_SINK, start))
    result = torch.cat((rank, window, sink)).long()
    if result.numel() > ORACLE_TOP_K + ORACLE_WINDOW + ORACLE_SINK or result.unique().numel() != result.numel():
        raise ValueError("oracle budget or duplicate index")
    return result


def oracle_config(mode):
    if mode != 'oracle_topk':
        return None
    try:
        scale = float(os.environ['LLAMA_LSA_ORACLE_SCALE'])
        softcap = float(os.environ['LLAMA_LSA_ORACLE_SOFTCAP'])
    except (KeyError, ValueError) as exc:
        raise ValueError('oracle scale/softcap missing or invalid') from exc
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(softcap) or softcap < 0:
        raise ValueError('oracle scale/softcap invalid')
    return scale, softcap


def compute(data, layer, mode, maps, oracle=None):
    if mode not in ('all', 'positional', 'pca', 'distilled', 'oracle_topk'):
        raise ValueError('unknown LSA control mode')
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
    elif mode == 'oracle_topk':
        if oracle is None:
            raise ValueError('oracle attention configuration missing')
        mask.fill_(-float('inf'))
        for row, position in enumerate(positions):
            selected = oracle_indices(q[row], k, position, *oracle)
            mask[row, selected] = 0
    elif mode in ('pca', 'distilled'):
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
        Path(dest).write_bytes(compute(bytearray(Path(source).read_bytes()), layer, mode, maps, oracle_config(mode)))
        return
    mode, map_path = sys.argv[2:]
    maps = load_maps(map_path, mode) if mode in ("pca", "distilled") else None
    oracle = oracle_config(mode)
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
            result = compute(bytearray(struct.pack("=II", nq, nk) + payload), str(layer), mode, maps, oracle)
            out.write(struct.pack("=i", 0) + result)
            out.flush()
        except Exception as exc:
            print(f"LSA worker: {exc}", file=sys.stderr, flush=True)
            out.write(struct.pack("=i", -1))
            out.flush()
            return


if __name__ == "__main__":
    main()
