"""Standalone Qwen3.8 flat index reference; not connected to llama inference."""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

HASHES = {
    "pca": "baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65",
    "distilled": "24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b",
}
LAYERS = tuple(range(3, 64, 4))


def load_maps(path, kind):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[kind]:
        raise ValueError("unrecognized map digest")
    maps = torch.load(path, map_location="cpu", weights_only=True)
    names = {f"{i}.W_{q}_idx" for i in LAYERS for q in ("Q", "K")}
    if set(maps) != names or any(x.shape != (768, 64) for x in maps.values()):
        raise ValueError("unexpected map coverage or shape")
    return maps


def project(q, k, wq, wk):
    """q [T,24,256], k [T,4,256], after Q/K norm; rotary prefix excluded."""
    if q.shape[1:] != (24, 256) or k.shape[1:] != (4, 256):
        raise ValueError("unsupported Qwen3.8 geometry")
    group = q[:, :, 64:].reshape(-1, 4, 6, 192).sum(2, dtype=torch.float32).reshape(-1, 768)
    keys = k[:, :, 64:].reshape(-1, 768).to(torch.bfloat16)
    iq = F.normalize((group.to(torch.bfloat16) @ wq.to(torch.bfloat16)).float(), dim=-1)
    ik = F.normalize((keys @ wk.to(torch.bfloat16)).float(), dim=-1).to(torch.bfloat16)
    return iq, ik


def buckets(index_keys):
    """Pool four consecutive keys, Ward-merge into 64 capped buckets per 512 keys."""
    blocks = index_keys.shape[0] // 512
    vectors = index_keys[:blocks * 512].float().reshape(blocks, 128, 4, 64).sum(2)
    sums = vectors.clone()
    sizes = torch.full((blocks, 128), 4.0)
    alive = torch.ones((blocks, 128), dtype=torch.bool)
    parent = torch.arange(128).expand(blocks, -1).clone()
    ix = torch.arange(128).expand(blocks, -1)
    diag = torch.eye(128, dtype=torch.bool)
    for _ in range(256):
        remaining = alive.sum(1) - 64
        if not (remaining > 0).any():
            break
        gram = sums @ sums.transpose(1, 2)
        sq = gram.diagonal(dim1=1, dim2=2)
        norms = sq.clamp_min(0).sqrt()
        cost = norms[:, :, None] + norms[:, None, :] - (sq[:, :, None] + sq[:, None, :] + 2 * gram).clamp_min(0).sqrt()
        allowed = alive[:, :, None] & alive[:, None, :] & ~diag
        allowed &= sizes[:, :, None] + sizes[:, None, :] <= 16
        best_cost, best = cost.masked_fill(~allowed, float("inf")).min(2)
        mutual = torch.isfinite(best_cost) & (best.gather(1, best) == ix) & (ix < best)
        rank = best_cost.masked_fill(~mutual, float("inf")).argsort(1).argsort(1)
        mutual &= rank < remaining.clamp_min(0)[:, None]
        if not mutual.any():
            raise ValueError("Ward failed to reach target bucket count")
        block, kept = mutual.nonzero(as_tuple=True)
        absorbed = best[block, kept]
        sums[block, kept] += sums[block, absorbed]
        sizes[block, kept] += sizes[block, absorbed]
        alive[block, absorbed] = False
        parent[block, absorbed] = kept
    if (alive.sum(1) != 64).any():
        raise ValueError("Ward round limit exceeded")
    for _ in range(8):
        parent = parent.gather(1, parent)
    labels = (alive.long().cumsum(1) - 1).gather(1, parent).repeat_interleave(4, 1)
    members = torch.full((blocks * 64, 16), -1, dtype=torch.int32)
    centroids = torch.empty((blocks * 64, 64))
    for block in range(blocks):
        for bucket in range(64):
            idx = (labels[block] == bucket).nonzero().flatten() + block * 512
            members[block * 64 + bucket, :len(idx)] = idx.int()
            vec = index_keys[idx].float().sum(0)
            centroids[block * 64 + bucket] = vec / vec.norm().clamp_min(1e-30)
    return centroids, members


def select(iq, ik, positions, top_k=4096, top_p=1024, window=512, sink=4):
    """Flat Ward centroid selection, expanded in bucket-score order, with causal window and sinks."""
    if positions.shape != (iq.shape[0],) or positions.numel() and (positions.min() < 0 or positions.max() >= len(ik)):
        raise ValueError("invalid query positions")
    centroids, members = buckets(ik)
    scores = iq.float() @ centroids.T
    out = torch.full((len(iq), top_k + window + sink), -1, dtype=torch.int32)
    for row, pos in enumerate(positions.tolist()):
        visible = min((pos // 512) * 64, len(centroids))
        picked = torch.argsort(scores[row, :visible], descending=True, stable=True)[:top_p]
        start = max(0, pos + 1 - window)
        keys = members[picked].flatten()
        keys = keys[(keys >= sink) & (keys < start)][:top_k]
        keys = torch.cat((keys, torch.arange(start, pos + 1), torch.arange(min(sink, start, pos + 1)))).int()
        out[row, :len(keys)] = keys
    return out
