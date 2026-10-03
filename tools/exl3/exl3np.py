"""EXL3 (exllamav3 trellis, mul1 codebook) numpy helpers for the GGUF importer.

Independent numpy re-implementation of the facts frozen by EXL3-1
(epyc-inference-research scripts/kernel_rnd/exl3, commit bc9ac47b); test_exl3np.py checks it
against that oracle's real fixtures. Nothing here is imported from exllamav3.

Native trellis: int16 [k/16, n/16, 16*K], tile (kt, nt) at (kt*n/16 + nt)*16K words.
band8 layout (EXL3-CPU-DESIGN.md section 1): tile (kt, nt) at ((nt/8)*(k/16)*8 + kt*8 + nt%8)*16K.
"""
from __future__ import annotations

import json
import math
import os
import struct

import numpy as np

MUL1_MULT = 0x83DCD12D
K_INV = 1774.0 / 262144.0            # fp16 0x1EEE
C_HALF = -10.3828125                 # fp16 0xC931
HAD_SCALE = np.float32(1.0 / math.sqrt(128.0))


# ----------------------------------------------------------------------------------------------
# safetensors (minimal, memmap; tolerates a growing .part file for byte ranges already present)
# ----------------------------------------------------------------------------------------------

_ST_DTYPES = {"F16": np.float16, "BF16": np.uint16, "F32": np.float32, "I16": np.int16,
              "I32": np.int32, "I64": np.int64, "U8": np.uint8, "I8": np.int8}


class SafeTensors:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        self.header = hdr
        self.base = 8 + n

    def names(self):
        return self.header.keys()

    def info(self, name):
        return self.header[name]

    def get(self, name) -> np.ndarray:
        h = self.header[name]
        a, b = h["data_offsets"]
        size = os.path.getsize(self.path)
        if self.base + b > size:
            raise IOError(f"{name}: bytes not yet present in {self.path} ({self.base + b} > {size})")
        dt = _ST_DTYPES[h["dtype"]]
        shape = tuple(h["shape"])
        count = int(np.prod(shape)) if shape else 1
        arr = np.memmap(self.path, dtype=dt, mode="r", offset=self.base + a, shape=(count,))
        return arr.reshape(shape) if shape else arr.reshape(())


class SafeTensorsDir:
    """All *.safetensors (and *.safetensors.part) files of a directory, by tensor name."""

    def __init__(self, directory: str, allow_part: bool = True):
        self.files = {}
        self.where = {}
        for fn in sorted(os.listdir(directory)):
            if fn.endswith(".safetensors") or (allow_part and fn.endswith(".safetensors.part")):
                full = os.path.join(directory, fn)
                if fn.endswith(".part") and os.path.exists(full[:-5]):
                    continue
                try:
                    st = SafeTensors(full)
                except Exception:
                    continue      # header not yet downloaded
                self.files[fn] = st
                for k in st.names():
                    self.where[k] = st

    def __contains__(self, name):
        return name in self.where

    def get(self, name):
        return self.where[name].get(name)

    def info(self, name):
        return self.where[name].info(name)


# ----------------------------------------------------------------------------------------------
# decode
# ----------------------------------------------------------------------------------------------

def tc_perm() -> np.ndarray:
    """state index t -> tile position r*16 + c (r = input row, c = output column)."""
    p = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0, c0 = (t % 4) * 2, t // 4
        r1, r2, r3, c1 = r0 + 1, r0 + 8, r0 + 9, c0 + 8
        p[t * 8:t * 8 + 8] = [r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
                              r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1]
    return p


TC_PERM = tc_perm()


def mul1_lut() -> np.ndarray:
    s = np.arange(65536, dtype=np.uint64)
    x = (s * MUL1_MULT) & 0xFFFFFFFF
    bs = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    # exact pre-rounding value in float64, then ONE rounding to binary16 (RNE)
    v = (1024.0 + bs.astype(np.float64)) * K_INV + C_HALF
    return v.astype(np.float16)


MUL1_LUT = mul1_lut()


def tile_states(tiles_u16: np.ndarray, K: int) -> np.ndarray:
    """tiles_u16: [..., 16*K] uint16/int16 packed tiles -> [..., 256] uint16 states (state order)."""
    t = np.ascontiguousarray(tiles_u16).view(np.uint16)
    lead = t.shape[:-1]
    w = t.reshape(-1, 16 * K).view("<u4").astype(np.uint64)      # [T, 8K] little-endian u32
    nw = 8 * K
    i = np.arange(256, dtype=np.int64)
    start = ((i + 257) * K - 16) % (256 * K)
    a = start // 32
    off = start % 32
    b = (a + 1) % nw
    hi = w[:, a]
    lo = w[:, b]
    v = ((hi << np.uint64(32)) | lo) >> (np.uint64(48) - off.astype(np.uint64))
    return (v & np.uint64(0xFFFF)).astype(np.uint16).reshape(*lead, 256)


def decode_raw(trellis: np.ndarray, K: int) -> np.ndarray:
    """native trellis [k/16, n/16, 16K] -> raw codebook matrix Q [k, n] float16."""
    tk, tn, words = trellis.shape
    assert words == 16 * K, (trellis.shape, K)
    st = tile_states(trellis, K)                                   # [tk, tn, 256]
    vals = MUL1_LUT[st]                                            # state order
    tiles = np.empty_like(vals)
    tiles[..., TC_PERM] = vals                                     # position order r*16+c
    q = tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)
    return q


def hadamard128() -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return (h * float(HAD_SCALE)).astype(np.float32)


H128 = hadamard128()


def materialize(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, K: int) -> np.ndarray:
    """Canonical-order reconstruction W [k, n] (float16), numpy fp32 matmuls for the H128 products.

    Same stage roundings as materialized_weight_fp32_fma_v1 (fp16 RNE after each of the four
    stages); the fp32 summation order of the H products is BLAS order, not ascending FMA, so a
    rare element can differ by one fp16 ulp from the canonical oracle (measured in test_exl3np).
    """
    q = decode_raw(trellis, K).astype(np.float32)
    k, n = q.shape
    L = np.einsum("ij,bjn->bin", H128, q.reshape(k // 128, 128, n), optimize=True).reshape(k, n)
    L = L.astype(np.float16).astype(np.float32)
    L = (L * suh.astype(np.float32)[:, None]).astype(np.float16).astype(np.float32)
    W = np.einsum("kbj,ji->kbi", L.reshape(k, n // 128, 128), H128, optimize=True).reshape(k, n)
    W = W.astype(np.float16).astype(np.float32)
    W = (W * svh.astype(np.float32)[None, :]).astype(np.float16)
    return W


# ----------------------------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------------------------

def native_to_band8(trellis: np.ndarray) -> np.ndarray:
    """[k/16, n/16, 16K] native -> band8 [n/128, k/16, 8, 16K] (contiguous)."""
    tk, tn, words = trellis.shape
    assert tn % 8 == 0
    return np.ascontiguousarray(trellis.reshape(tk, tn // 8, 8, words).transpose(1, 0, 2, 3))


def band8_permute_out(b8: np.ndarray, block_perm: np.ndarray) -> np.ndarray:
    """new band j = old band block_perm[j]."""
    return np.ascontiguousarray(b8[block_perm])


def band8_permute_in(b8: np.ndarray, block_perm: np.ndarray) -> np.ndarray:
    """new 128-input block j = old block block_perm[j] (8 kt tiles per block)."""
    nb, tk, eight, words = b8.shape
    x = b8.reshape(nb, tk // 8, 8, eight, words)
    return np.ascontiguousarray(x[:, block_perm]).reshape(nb, tk, eight, words)


def band8_to_native(b8: np.ndarray) -> np.ndarray:
    nb, tk, eight, words = b8.shape
    return np.ascontiguousarray(b8.transpose(1, 0, 2, 3)).reshape(tk, nb * 8, words)


def vhead_block_perm(n_blocks_total: int, seg_block0: int, num_k: int, num_v: int) -> np.ndarray:
    """llama.cpp _LinearAttentionVReorderBase on 128-wide heads: grouped [k, r] -> tiled [r, k].

    Returns perm over all blocks with new[j] = old[perm[j]]; blocks outside the V segment fixed."""
    r = num_v // num_k
    perm = np.arange(n_blocks_total)
    for p in range(num_v):
        rr, kh = divmod(p, num_k)
        perm[seg_block0 + p] = seg_block0 + kh * r + rr
    return perm
