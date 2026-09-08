#!/usr/bin/env python3
"""Portable, read-only GLM-5.3-Flash GGUF header validator."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

SAMPLE_BYTES = 1024 * 1024
MLA_BLOCKS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]


def resolve_gguf_py(explicit: Path | None = None) -> Path | None:
    """Resolve gguf-py from CLI, GGUF_PY, or a containing llama.cpp tree."""
    if explicit is not None:
        return explicit.resolve()
    if value := os.environ.get("GGUF_PY"):
        return Path(value).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "gguf-py"
        if (candidate / "gguf" / "gguf_reader.py").is_file():
            return candidate
    return None


def load_reader(gguf_py: Path | None = None):
    path = resolve_gguf_py(gguf_py)
    if path is not None:
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module("gguf").GGUFReader
    except (ImportError, AttributeError) as exc:
        hint = "set GGUF_PY or pass --gguf-py" if path is None else f"checked {path}"
        raise RuntimeError(f"cannot import GGUFReader ({hint}): {exc}") from exc


def load_writer(gguf_py: Path | None = None):
    path = resolve_gguf_py(gguf_py)
    if path is not None:
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module("gguf").GGUFWriter
    except (ImportError, AttributeError) as exc:
        hint = "set GGUF_PY or pass --gguf-py" if path is None else f"checked {path}"
        raise RuntimeError(f"cannot import GGUFWriter ({hint}): {exc}") from exc


def _normalise(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, (float, int)):
        return abs(actual - expected) <= max(1e-12, abs(float(expected)) * 1e-6)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _sample_sha256(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(b"glm53-bounded-sample-v1\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(stream.read(SAMPLE_BYTES))
        if size > SAMPLE_BYTES:
            stream.seek(max(SAMPLE_BYTES, size - SAMPLE_BYTES))
            digest.update(stream.read(SAMPLE_BYTES))
    return digest.hexdigest()


def scan(paths: Iterable[Path], reader_cls: Any) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for path in sorted(paths):
        reader = reader_cls(str(path), "r")
        shard_metadata = {key: _normalise(field.contents()) for key, field in sorted(reader.fields.items())}
        descriptors = [
            {
                "name": tensor.name,
                "shape": [int(dim) for dim in tensor.shape],
                "type": tensor.tensor_type.name,
                "n_elements": int(tensor.n_elements),
                "n_bytes": int(tensor.n_bytes),
                "data_offset": int(tensor.data_offset),
            }
            for tensor in reader.tensors
        ]
        stat = path.stat()
        shards.append({
            "file": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sample_sha256": _sample_sha256(path),
            "data_offset": int(reader.data_offset),
            "metadata_sha256": _digest(shard_metadata),
            "tensor_header_sha256": _digest(descriptors),
            "tensor_count": len(descriptors),
        })
        for key, value in shard_metadata.items():
            if key.startswith(("GGUF.", "split.")):
                continue
            if key in metadata and metadata[key] != value:
                raise ValueError(f"metadata conflict for {key!r} across shards")
            metadata[key] = value
        tensors.extend({"shard": path.name, **item} for item in descriptors)
    tensors.sort(key=lambda item: item["name"])
    names = [item["name"] for item in tensors]
    seen: set[str] = set()
    duplicates = sorted({name for name in names if name in seen or seen.add(name)})
    return {
        "shards": shards,
        "metadata": metadata,
        "metadata_sha256": _digest(metadata),
        "tensors": tensors,
        "tensor_set_sha256": _digest(tensors),
        "tensor_count": len(tensors),
        "duplicate_tensor_names": duplicates,
    }


def real_contract() -> dict[str, Any]:
    head_kv = [1 if block in MLA_BLOCKS else 0 for block in range(45)] + [1]
    return {
        "accepted_disk_architectures": ["glm5next", "glm5-next"],
        "dimensions": {
            "embedding": 4096, "vocab": 154880, "heads": 64,
            "kda_head_dim": 128, "conv_kernel": 4,
            "q_lora_rank": 1536, "kv_lora_rank": 512,
            "mla_key_dim": 256, "mla_value_dim": 256,
            "index_heads": 32, "index_head_dim": 128,
            "index_top_k": 2048, "index_kpool": 4,
            "experts": 288, "expert_ff": 2048, "dense_ff": 12288,
            "shared_experts": 1, "hc_streams": 4, "dense_lead": 3,
            "trunk_blocks": 45, "nextn_blocks": 1, "total_blocks": 46,
            "mla_blocks": MLA_BLOCKS,
        },
        "required_metadata": {
            "glm5next.block_count": 46,
            "glm5next.nextn_predict_layers": 1,
            "glm5next.context_length": 1048576,
            "glm5next.embedding_length": 4096,
            "glm5next.vocab_size": 154880,
            "glm5next.feed_forward_length": 12288,
            "glm5next.attention.head_count": 64,
            "glm5next.attention.head_count_kv": head_kv,
            "glm5next.attention.key_length": 512,
            "glm5next.attention.value_length": 512,
            "glm5next.attention.key_length_mla": 256,
            "glm5next.attention.value_length_mla": 256,
            "glm5next.attention.q_lora_rank": 1536,
            "glm5next.attention.kv_lora_rank": 512,
            "glm5next.attention.layer_norm_rms_epsilon": 1e-5,
            "glm5next.rope.dimension_count": 0,
            "glm5next.ssm.conv_kernel": 4,
            "glm5next.kda.head_dim": 128,
            "glm5next.kda.gate_lower_bound": -5.0,
            "glm5next.expert_count": 288,
            "glm5next.expert_used_count": 8,
            "glm5next.expert_shared_count": 1,
            "glm5next.expert_feed_forward_length": 2048,
            "glm5next.leading_dense_block_count": 3,
            "glm5next.expert_weights_scale": 2.5,
            "glm5next.expert_weights_norm": True,
            "glm5next.expert_gating_func": 2,
            "glm5next.attention.indexer.head_count": 32,
            "glm5next.attention.indexer.key_length": 128,
            "glm5next.attention.indexer.top_k": 2048,
            "glm5next.attention.indexer.kpool": 4,
            "glm5next.hyper_connection.count": 4,
            "glm5next.hyper_connection.sinkhorn_iterations": 20,
            "glm5next.hyper_connection.epsilon": 1e-6,
        },
        "optional_metadata": {
            "glm5next.attention.layer_norm_epsilon": {"expected": 1e-6, "default": 0.0},
            "glm5next.swiglu_clamp_exp": {"expected": [10.0] * 46, "default": []},
            "glm5next.swiglu_clamp_shexp": {"expected": [10.0] * 46, "default": "clamp_exp"},
        },
        "semantic_defaults": {
            "glm5next.attention.indexer.kpool_select_tail": True,
            "glm5next.attention.indexer.index_share_mtp": True,
        },
        "fallbacks": {
            "blk.45.nextn.embed_tokens.weight": "token_embd.weight",
            "blk.45.nextn.shared_head_head.weight": "output.weight",
        },
    }


def fixture_contract() -> dict[str, Any]:
    result = real_contract()
    result["dimensions"] = {
        "embedding": 32, "vocab": 64, "heads": 4,
        "kda_head_dim": 8, "conv_kernel": 4,
        "q_lora_rank": 16, "kv_lora_rank": 8,
        "mla_key_dim": 8, "mla_value_dim": 8,
        "index_heads": 2, "index_head_dim": 4,
        "index_top_k": 8, "index_kpool": 4,
        "experts": 4, "expert_ff": 16, "dense_ff": 48,
        "shared_experts": 1, "hc_streams": 4, "dense_lead": 3,
        "trunk_blocks": 4, "nextn_blocks": 1, "total_blocks": 5,
        "mla_blocks": [3],
    }
    result["required_metadata"] = {}
    result["optional_metadata"] = {}
    result["semantic_defaults"] = {}
    result["fallbacks"] = {
        "blk.4.nextn.embed_tokens.weight": "token_embd.weight",
        "blk.4.nextn.shared_head_head.weight": "output.weight",
    }
    return result


def expected_tensor_shapes(contract: dict[str, Any]) -> dict[str, list[int]]:
    """Derive every required tensor from #27917 loader construction."""
    d = contract["dimensions"]
    e, vocab = d["embedding"], d["vocab"]
    heads, kd, conv = d["heads"], d["kda_head_dim"], d["conv_kernel"]
    inner = heads * kd
    qr, kr = d["q_lora_rank"], d["kv_lora_rank"]
    mk, mv = d["mla_key_dim"], d["mla_value_dim"]
    ih, idim, pool = d["index_heads"], d["index_head_dim"], d["index_kpool"]
    nex, ffe, ffd = d["experts"], d["expert_ff"], d["dense_ff"]
    shared, hc = d["shared_experts"], d["hc_streams"]
    mix = (2 + hc) * hc
    trunk, total, lead = d["trunk_blocks"], d["total_blocks"], d["dense_lead"]
    mla = set(d["mla_blocks"])
    out = {"token_embd.weight": [e, vocab], "output_norm.weight": [e], "output.weight": [e, vocab]}

    def put(block: int, name: str, shape: list[int]) -> None:
        out[f"blk.{block}.{name}"] = shape

    for block in range(total):
        put(block, "attn_norm.weight", [e]); put(block, "ffn_norm.weight", [e])
        if block < trunk:
            for part in ("attn", "ffn"):
                put(block, f"hc_{part}_fn.weight", [hc * e, mix])
                put(block, f"hc_{part}_base.weight", [mix])
                put(block, f"hc_{part}_scale.weight", [3])
        if block in mla or block >= trunk:
            for name, shape in {
                "attn_q_a_norm.weight": [qr], "attn_kv_a_norm.weight": [kr],
                "attn_q_a.weight": [e, qr], "attn_q_b.weight": [qr, heads * mk],
                "attn_kv_a_mqa.weight": [e, kr], "attn_k_b.weight": [mk, kr, heads],
                "attn_v_b.weight": [kr, mv, heads], "attn_output.weight": [heads * mv, e],
                "indexer.k_norm.weight": [idim], "indexer.k_norm.bias": [idim],
                "indexer.proj.weight": [e, ih], "indexer.attn_k.weight": [e, idim],
                "indexer.attn_q_b.weight": [qr, ih * idim],
                "indexer_compressor_gate.weight": [e, idim],
                "indexer_compressor_ape.weight": [idim, pool],
            }.items(): put(block, name, shape)
        else:
            for qkv in ("q", "k", "v"):
                put(block, f"attn_{qkv}.weight", [e, inner])
                put(block, f"ssm_conv1d_{qkv}.weight", [conv, 1, inner])
            for name, shape in {
                "ssm_f_a.weight": [e, kd], "ssm_f_b.weight": [kd, inner],
                "ssm_beta.weight": [e, heads], "ssm_a": [heads], "ssm_dt.bias": [inner],
                "ssm_g_a.weight": [e, kd], "ssm_g_b.weight": [kd, inner],
                "ssm_norm.weight": [kd], "attn_output.weight": [inner, e],
            }.items(): put(block, name, shape)
        if block < lead:
            put(block, "ffn_gate.weight", [e, ffd]); put(block, "ffn_down.weight", [ffd, e]); put(block, "ffn_up.weight", [e, ffd])
        else:
            for name, shape in {
                "ffn_gate_inp.weight": [e, nex], "exp_probs_b.bias": [nex],
                "ffn_gate_exps.weight": [e, ffe, nex], "ffn_down_exps.weight": [ffe, e, nex],
                "ffn_up_exps.weight": [e, ffe, nex], "ffn_gate_shexp.weight": [e, ffe * shared],
                "ffn_down_shexp.weight": [ffe * shared, e], "ffn_up_shexp.weight": [e, ffe * shared],
            }.items(): put(block, name, shape)
    put(trunk, "nextn.eh_proj.weight", [2 * e, e])
    put(trunk, "nextn.enorm.weight", [e]); put(trunk, "nextn.hnorm.weight", [e])
    put(trunk, "nextn.shared_head_norm.weight", [e])
    return out


def validate_loader(actual: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    requirements: list[str] = []
    md = actual["metadata"]
    arch = md.get("general.architecture")
    if arch not in contract["accepted_disk_architectures"]:
        errors.append(f"unsupported architecture: {arch!r}")
    canonical = dict(md)
    if arch == "glm5-next":
        canonical.update({"glm5next." + k[len("glm5-next."):]: v for k, v in md.items() if k.startswith("glm5-next.")})
    for key, expected in contract["required_metadata"].items():
        if key not in canonical: errors.append(f"missing metadata: {key}")
        elif not _equal(canonical[key], expected): errors.append(f"metadata mismatch {key}")
    for key, rule in contract["optional_metadata"].items():
        if key not in canonical: requirements.append(f"optional {key} absent; loader default={rule['default']!r}")
        elif not _equal(canonical[key], rule["expected"]): errors.append(f"optional metadata mismatch {key}")
    for key, default in contract["semantic_defaults"].items():
        if key not in canonical: requirements.append(f"loader must default absent {key} to {default!r}")
        elif not _equal(canonical[key], default): errors.append(f"semantic metadata mismatch {key}")
    expected = expected_tensor_shapes(contract)
    got = {tensor["name"]: tensor["shape"] for tensor in actual["tensors"]}
    for name in sorted(expected.keys() - got.keys()): errors.append(f"missing tensor: {name}")
    for name in sorted(got.keys() - expected.keys() - contract["fallbacks"].keys()): errors.append(f"unexpected tensor: {name}")
    for name in sorted(expected.keys() & got.keys()):
        if expected[name] != got[name]: errors.append(f"tensor shape mismatch {name}: expected {expected[name]}, got {got[name]}")
    for name, fallback in contract["fallbacks"].items():
        if name not in got: requirements.append(f"{name} absent; loader must use {fallback}")
    if actual["duplicate_tensor_names"]: errors.append(f"duplicate tensors: {actual['duplicate_tensor_names']}")
    return {"loader_schema_compatible": not errors, "errors": errors, "loader_requirements": requirements,
            "loader_required_tensor_count": len(expected), "observed_tensor_set_sha256": actual["tensor_set_sha256"]}


def validate_identity(actual: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    expected = manifest.get("inventory", manifest)
    errors: list[str] = []
    exp_shards = {item["file"]: item for item in expected["shards"]}
    got_shards = {item["file"]: item for item in actual["shards"]}
    if exp_shards.keys() != got_shards.keys(): errors.append("shard filename set mismatch")
    for name in sorted(exp_shards.keys() & got_shards.keys()):
        for key in ("size", "sample_sha256", "metadata_sha256", "tensor_header_sha256", "tensor_count"):
            if exp_shards[name][key] != got_shards[name][key]: errors.append(f"identity mismatch {name} {key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--gguf-py", type=Path)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--result-out", type=Path)
    args = parser.parse_args()
    paths = sorted(args.model_dir.glob("*.gguf"))
    if not paths: raise SystemExit(f"no .gguf files under {args.model_dir}")
    actual = scan(paths, load_reader(args.gguf_py))
    result = validate_loader(actual, real_contract())
    if args.identity_manifest:
        result["identity_errors"] = validate_identity(actual, json.loads(args.identity_manifest.read_text()))
        result["identity_matches"] = not result["identity_errors"]
    if args.result_out: args.result_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["loader_schema_compatible"] and not result.get("identity_errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
