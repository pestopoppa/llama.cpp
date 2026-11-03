#!/usr/bin/env python3
"""Offline, explicit-binding exporter. Torch is never imported by native runtime."""
import argparse
import hashlib
import math
from pathlib import Path
import struct


def encode_maps(maps, *, model_identity, source_sha256, identity, q_heads, kv_heads,
                head_dim, rope_dim, layers):
    if (q_heads, kv_heads, head_dim, rope_dim) not in ((24, 4, 256, 64), (16, 2, 256, 64)):
        raise ValueError("unsupported released/35B geometry")
    if not layers or len(set(layers)) != len(layers) or any(x < 0 or x > 4095 for x in layers):
        raise ValueError("invalid global layer coverage")
    for text in (model_identity, identity):
        if not text or '\0' in text or len(text.encode()) > 4096:
            raise ValueError("invalid identity")
    if len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError("expected lowercase source sha256")
    expected = {f"{i}.W_{axis}_idx" for i in layers for axis in ('Q', 'K')}
    if set(maps) != expected:
        raise ValueError("missing, extra, or recurrent-layer maps")
    # Import only for offline PT conversion, and never initialize a device.
    import torch
    payload = bytearray(model_identity.encode() + identity.encode())
    rows = kv_heads * (head_dim - rope_dim)
    for layer in sorted(layers):
        payload += struct.pack('<I', layer)
        for axis in ('Q', 'K'):
            value = maps[f"{layer}.W_{axis}_idx"]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows, 64):
                raise ValueError(f"invalid {layer} {axis} shape, expected {(rows, 64)}")
            if value.layout != torch.strided or value.is_complex() or not bool(torch.isfinite(value).all()):
                raise ValueError("map is not finite real strided tensor")
            value = value.detach().cpu().to(torch.bfloat16).contiguous()
            if not bool(torch.isfinite(value).all()):
                raise ValueError("BF16 conversion overflow")
            # Explicit LE representation, independent of host endianness.
            words = value.view(torch.int16).reshape(-1).tolist()
            payload += struct.pack('<' + 'H' * len(words), *(x & 65535 for x in words))
    header = b'LSAMAP1\0' + struct.pack('<9I', 1, q_heads, kv_heads, head_dim,
                                          rope_dim, 64, len(layers),
                                          len(model_identity.encode()), len(identity.encode()))
    return header + bytes.fromhex(source_sha256) + hashlib.sha256(payload).digest() + payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--source-sha256', required=True, help='approved PT digest; conversion refuses mismatch')
    p.add_argument('--model-identity', required=True, help='exact model binding, preferably name@sha256:GGUF_DIGEST')
    p.add_argument('--identity', required=True, help='training/release identity')
    p.add_argument('--geometry', choices=['27b', '35b'], required=True)
    p.add_argument('--layers', required=True, help='exact comma-separated global attention layer ids')
    args = p.parse_args()
    source = args.source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != args.source_sha256:
        p.error('PT digest differs from approved source')
    import torch
    maps = torch.load(args.source, map_location='cpu', weights_only=True)
    shape = (24, 4, 256, 64) if args.geometry == '27b' else (16, 2, 256, 64)
    blob = encode_maps(maps, model_identity=args.model_identity, source_sha256=digest,
                       identity=args.identity, q_heads=shape[0], kv_heads=shape[1],
                       head_dim=shape[2], rope_dim=shape[3],
                       layers=[int(x) for x in args.layers.split(',')])
    # Exclusive create: a map already bound by a manifest is never overwritten.
    with args.output.open('xb') as out:
        out.write(blob)
    print(f'{args.output}: {len(blob)} bytes sha256={hashlib.sha256(blob).hexdigest()}')


if __name__ == '__main__':
    main()
