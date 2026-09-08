#!/usr/bin/env python3
"""Generate deterministic tiny GLM5Next alias/default test fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from glm53_artifact import expected_tensor_shapes, fixture_contract, load_writer


def metadata(prefix: str, explicit: bool) -> dict[str, tuple[str, object]]:
    d = fixture_contract()["dimensions"]
    values: dict[str, tuple[str, object]] = {
        f"{prefix}.block_count": ("u32", 5), f"{prefix}.nextn_predict_layers": ("u32", 1),
        f"{prefix}.context_length": ("u32", 128), f"{prefix}.embedding_length": ("u32", d["embedding"]),
        f"{prefix}.vocab_size": ("u32", d["vocab"]), f"{prefix}.feed_forward_length": ("u32", d["dense_ff"]),
        f"{prefix}.attention.head_count": ("u32", d["heads"]),
        f"{prefix}.attention.head_count_kv": ("array", [0, 0, 0, 1, 1]),
        f"{prefix}.attention.key_length": ("u32", d["mla_key_dim"]),
        f"{prefix}.attention.value_length": ("u32", d["mla_value_dim"]),
        f"{prefix}.attention.key_length_mla": ("u32", d["mla_key_dim"]),
        f"{prefix}.attention.value_length_mla": ("u32", d["mla_value_dim"]),
        f"{prefix}.attention.q_lora_rank": ("u32", d["q_lora_rank"]),
        f"{prefix}.attention.kv_lora_rank": ("u32", d["kv_lora_rank"]),
        f"{prefix}.attention.layer_norm_rms_epsilon": ("f32", 1e-5),
        f"{prefix}.attention.layer_norm_epsilon": ("f32", 1e-6),
        f"{prefix}.rope.dimension_count": ("u32", 0), f"{prefix}.ssm.conv_kernel": ("u32", 4),
        f"{prefix}.kda.head_dim": ("u32", d["kda_head_dim"]),
        f"{prefix}.kda.gate_lower_bound": ("f32", -5.0),
        f"{prefix}.expert_count": ("u32", d["experts"]), f"{prefix}.expert_used_count": ("u32", 2),
        f"{prefix}.expert_shared_count": ("u32", 1),
        f"{prefix}.expert_feed_forward_length": ("u32", d["expert_ff"]),
        f"{prefix}.leading_dense_block_count": ("u32", 3),
        f"{prefix}.expert_weights_scale": ("f32", 2.5), f"{prefix}.expert_weights_norm": ("bool", True),
        f"{prefix}.expert_gating_func": ("u32", 2),
        f"{prefix}.attention.indexer.head_count": ("u32", d["index_heads"]),
        f"{prefix}.attention.indexer.key_length": ("u32", d["index_head_dim"]),
        f"{prefix}.attention.indexer.top_k": ("u32", d["index_top_k"]),
        f"{prefix}.attention.indexer.kpool": ("u32", 4),
        f"{prefix}.hyper_connection.count": ("u32", 4),
        f"{prefix}.hyper_connection.sinkhorn_iterations": ("u32", 2),
        f"{prefix}.hyper_connection.epsilon": ("f32", 1e-6),
        "tokenizer.ggml.model": ("str", "none"),
    }
    if explicit:
        values[f"{prefix}.attention.indexer.kpool_select_tail"] = ("bool", True)
        values[f"{prefix}.attention.indexer.index_share_mtp"] = ("bool", True)
    return values


def tensor(name: str, ggml_shape: list[int], seed: int) -> np.ndarray:
    local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
    value = np.random.default_rng(local_seed).normal(0, 0.02, list(reversed(ggml_shape))).astype(np.float32)
    if name.endswith("norm.weight"): value += 1.0
    return value


def generate(output: Path, gguf_py: Path | None = None, seed: int = 0x53A5) -> list[Path]:
    writer_cls = load_writer(gguf_py)
    output.mkdir(parents=True, exist_ok=True)
    shapes = expected_tensor_shapes(fixture_contract())
    paths: list[Path] = []
    records = []
    for arch in ("glm5next", "glm5-next"):
        for explicit in (False, True):
            path = output / f"tiny-{arch}-{'explicit' if explicit else 'defaults'}.gguf"
            writer = writer_cls(path, arch)
            for key, (kind, value) in metadata(arch, explicit).items():
                {"u32": writer.add_uint32, "f32": writer.add_float32, "bool": writer.add_bool,
                 "str": writer.add_string, "array": writer.add_array}[kind](key, value)
            for name, shape in sorted(shapes.items()): writer.add_tensor(name, tensor(name, shape, seed))
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            paths.append(path)
            records.append({"file": path.name, "architecture": arch, "explicit_flags": explicit,
                            "seed": seed, "tensor_count": len(shapes)})
    (output / "fixtures.json").write_text(json.dumps(records, indent=2) + "\n")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf-py", type=Path)
    parser.add_argument("--seed", type=int, default=0x53A5)
    args = parser.parse_args()
    generate(args.output, args.gguf_py, args.seed)
    return 0


if __name__ == "__main__": raise SystemExit(main())
