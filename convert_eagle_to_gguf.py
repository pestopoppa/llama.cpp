#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert EAGLE draft head checkpoints to GGUF format.

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) uses
a lightweight draft head (single transformer layer) attached to the target
model's hidden states for speculative decoding.

Usage:
    python convert_eagle_to_gguf.py <eagle_checkpoint_dir> --outfile <output.gguf>

Example:
    python convert_eagle_to_gguf.py /path/to/EAGLE-LLaMA2-Chat-7B --outfile eagle-llama2-7b.gguf
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Add gguf-py to path
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf


# EAGLE tensor name mapping: PyTorch name -> GGUF name
# EAGLE-1 structure:
#   - embed_tokens: token embeddings (same vocab as base model)
#   - fc: fusion layer (combines hidden state + embedded token)
#   - layers.0.*: single decoder layer (attention + FFN + norms)
EAGLE_TENSOR_MAP = {
    # Token embeddings
    "embed_tokens.weight": "eagle.embed_tokens.weight",
    "model.embed_tokens.weight": "eagle.embed_tokens.weight",

    # Fusion layer (fc combines hidden states with embeddings)
    "fc.weight": "eagle.fc.weight",
    "fc.bias": "eagle.fc.bias",
    "model.fc.weight": "eagle.fc.weight",
    "model.fc.bias": "eagle.fc.bias",

    # Layer 0 attention norm
    "layers.0.input_layernorm.weight": "eagle.blk.0.attn_norm.weight",
    "model.layers.0.input_layernorm.weight": "eagle.blk.0.attn_norm.weight",

    # Layer 0 attention (Q, K, V, O projections)
    "layers.0.self_attn.q_proj.weight": "eagle.blk.0.attn_q.weight",
    "layers.0.self_attn.k_proj.weight": "eagle.blk.0.attn_k.weight",
    "layers.0.self_attn.v_proj.weight": "eagle.blk.0.attn_v.weight",
    "layers.0.self_attn.o_proj.weight": "eagle.blk.0.attn_output.weight",
    "model.layers.0.self_attn.q_proj.weight": "eagle.blk.0.attn_q.weight",
    "model.layers.0.self_attn.k_proj.weight": "eagle.blk.0.attn_k.weight",
    "model.layers.0.self_attn.v_proj.weight": "eagle.blk.0.attn_v.weight",
    "model.layers.0.self_attn.o_proj.weight": "eagle.blk.0.attn_output.weight",

    # Layer 0 FFN norm
    "layers.0.post_attention_layernorm.weight": "eagle.blk.0.ffn_norm.weight",
    "model.layers.0.post_attention_layernorm.weight": "eagle.blk.0.ffn_norm.weight",

    # Layer 0 FFN (gate, up, down projections) - SwiGLU/SiLU activation
    "layers.0.mlp.gate_proj.weight": "eagle.blk.0.ffn_gate.weight",
    "layers.0.mlp.up_proj.weight": "eagle.blk.0.ffn_up.weight",
    "layers.0.mlp.down_proj.weight": "eagle.blk.0.ffn_down.weight",
    "model.layers.0.mlp.gate_proj.weight": "eagle.blk.0.ffn_gate.weight",
    "model.layers.0.mlp.up_proj.weight": "eagle.blk.0.ffn_up.weight",
    "model.layers.0.mlp.down_proj.weight": "eagle.blk.0.ffn_down.weight",
}


def load_config(dir_model: Path) -> Dict[str, Any]:
    """Load EAGLE config.json."""
    config_path = dir_model / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {dir_model}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pytorch_model(dir_model: Path) -> Dict[str, torch.Tensor]:
    """Load pytorch_model.bin weights."""
    model_path = dir_model / "pytorch_model.bin"
    if not model_path.exists():
        # Try safetensors
        safetensors_path = dir_model / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file
            return load_file(safetensors_path)
        raise FileNotFoundError(f"No pytorch_model.bin or model.safetensors found in {dir_model}")

    return torch.load(model_path, map_location="cpu", weights_only=True)


def convert_tensor(tensor: torch.Tensor, ftype: gguf.LlamaFileType) -> tuple:
    """Convert tensor to numpy with appropriate dtype."""
    # Convert to float32 first
    tensor = tensor.float()
    data = tensor.numpy()

    # Determine output dtype based on ftype
    if ftype == gguf.LlamaFileType.ALL_F32:
        return data, gguf.GGMLQuantizationType.F32
    elif ftype == gguf.LlamaFileType.MOSTLY_F16:
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16
    elif ftype == gguf.LlamaFileType.MOSTLY_BF16:
        # BF16 - use view trick
        data_f32 = data.astype(np.float32)
        data_bf16 = data_f32.view(np.uint32) >> 16
        return data_bf16.astype(np.uint16), gguf.GGMLQuantizationType.BF16
    else:
        # Default to F16
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16


def main():
    parser = argparse.ArgumentParser(
        description="Convert EAGLE draft head checkpoint to GGUF format"
    )
    parser.add_argument(
        "dir_model",
        type=Path,
        help="Path to EAGLE checkpoint directory (contains config.json and pytorch_model.bin)",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output GGUF file path (default: <dir_model>/eagle.gguf)",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        default="f16",
        choices=["f32", "f16", "bf16"],
        help="Output data type (default: f16)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    args = parser.parse_args()

    # Validate input directory
    if not args.dir_model.is_dir():
        print(f"Error: {args.dir_model} is not a directory")
        sys.exit(1)

    # Set output file
    if args.outfile is None:
        args.outfile = args.dir_model / "eagle.gguf"

    # Map output type
    ftype_map = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    }
    ftype = ftype_map[args.outtype]

    print(f"Loading EAGLE checkpoint from {args.dir_model}")

    # Load config
    config = load_config(args.dir_model)
    if args.verbose:
        print(f"Config: {json.dumps(config, indent=2)}")

    # Load weights
    state_dict = load_pytorch_model(args.dir_model)
    print(f"Loaded {len(state_dict)} tensors")

    if args.verbose:
        print("Available tensors:")
        for name in sorted(state_dict.keys()):
            shape = list(state_dict[name].shape)
            print(f"  {name}: {shape}")

    # Create GGUF writer with EAGLE architecture
    # Note: We use a custom "eagle" architecture
    writer = gguf.GGUFWriter(args.outfile, arch="eagle")

    # Write metadata
    writer.add_name(args.dir_model.name)

    # EAGLE hyperparameters from config
    hidden_size = config.get("hidden_size", 4096)
    intermediate_size = config.get("intermediate_size", 11008)
    num_attention_heads = config.get("num_attention_heads", 32)
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    num_hidden_layers = config.get("num_hidden_layers", 1)  # EAGLE typically has 1 layer
    vocab_size = config.get("vocab_size", 32000)
    max_position_embeddings = config.get("max_position_embeddings", 4096)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 10000.0)

    # Write hyperparameters using custom keys
    writer.add_uint32("eagle.embedding_length", hidden_size)
    writer.add_uint32("eagle.feed_forward_length", intermediate_size)
    writer.add_uint32("eagle.attention.head_count", num_attention_heads)
    writer.add_uint32("eagle.attention.head_count_kv", num_key_value_heads)
    writer.add_uint32("eagle.block_count", num_hidden_layers)
    writer.add_uint32("eagle.vocab_size", vocab_size)
    writer.add_uint32("eagle.context_length", max_position_embeddings)
    writer.add_float32("eagle.attention.layer_norm_rms_epsilon", rms_norm_eps)
    writer.add_float32("eagle.rope.freq_base", rope_theta)

    # Map and write tensors
    tensors_written = 0
    for pt_name, tensor in state_dict.items():
        # Find GGUF name
        gguf_name = EAGLE_TENSOR_MAP.get(pt_name)

        if gguf_name is None:
            # Try without 'model.' prefix
            alt_name = pt_name.replace("model.", "", 1) if pt_name.startswith("model.") else f"model.{pt_name}"
            gguf_name = EAGLE_TENSOR_MAP.get(alt_name)

        if gguf_name is None:
            print(f"Warning: Unknown tensor {pt_name}, skipping")
            continue

        # Convert tensor
        data, qtype = convert_tensor(tensor, ftype)

        if args.verbose:
            print(f"  {pt_name} -> {gguf_name}: {list(tensor.shape)} ({qtype.name})")

        writer.add_tensor(gguf_name, data, raw_dtype=qtype)
        tensors_written += 1

    print(f"Wrote {tensors_written} tensors to {args.outfile}")

    # Write file
    writer.write_header_to_file(args.outfile)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully converted EAGLE checkpoint to {args.outfile}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  KV heads: {num_key_value_heads}")
    print(f"  FFN size: {intermediate_size}")
    print(f"  Layers: {num_hidden_layers}")
    print(f"  Vocab size: {vocab_size}")


if __name__ == "__main__":
    main()
