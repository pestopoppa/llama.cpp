#!/usr/bin/env python3
"""
Convert DFlash block diffusion drafter models from HuggingFace safetensors to GGUF.

DFlash models are small (0.5-1B) transformers trained for block diffusion speculation.
They share embedding + lm_head with the target model at runtime. Architecture is
standard Qwen3-like attention+FFN with two extra tensors:
  - fc.weight:          conditioning projection (concatenated target hidden states -> drafter dim)
  - hidden_norm.weight: RMS norm for conditioning signal

For Phase 1 (load test), we generate minimal dummy embed_tokens and lm_head so the
model loads as a self-contained Qwen3-like model. In production (Phase 2+), these
would come from the target model at runtime.

Usage:
  python3 convert_dflash_to_gguf.py /path/to/DFlash-model --outfile output.gguf [--outtype f16|f32]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Add gguf-py to path
sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))
import gguf


def read_safetensors_metadata(path: Path) -> dict:
    """Read tensor metadata from safetensors file without loading tensor data."""
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        with open(f, "rb") as fp:
            header_size = struct.unpack("<Q", fp.read(8))[0]
            header = json.loads(fp.read(header_size))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            tensors[name] = {
                "shape": info["shape"],
                "dtype": info["dtype"],
                "file": str(f),
                "data_offsets": info["data_offsets"],
                "header_size": header_size,
            }
    return tensors


def read_safetensors_tensor(info: dict) -> np.ndarray:
    """Load a single tensor from safetensors file."""
    shape = info["shape"]
    raw_dtype = info["dtype"]
    start, end = info["data_offsets"]

    with open(info["file"], "rb") as f:
        f.seek(8 + info["header_size"] + start)
        data = f.read(end - start)

    if raw_dtype == "BF16":
        # BF16 -> F32: zero-pad each bf16 to 32 bits
        bf16 = np.frombuffer(data, dtype=np.uint16)
        f32 = np.zeros(len(bf16), dtype=np.uint32)
        f32[:] = bf16.astype(np.uint32) << 16
        arr = f32.view(np.float32).reshape(shape)
        return arr
    elif raw_dtype == "F16":
        return np.frombuffer(data, dtype=np.float16).reshape(shape)
    elif raw_dtype == "F32":
        return np.frombuffer(data, dtype=np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported dtype: {raw_dtype}")


# HuggingFace -> GGUF tensor name mapping for DFlash (Qwen3-like layers)
# DFlash tensors have NO "model." prefix (unlike standard Qwen3)
TENSOR_MAP = {
    # Layer tensors (with block ID substitution)
    "layers.{bid}.self_attn.q_proj.weight": "blk.{bid}.attn_q.weight",
    "layers.{bid}.self_attn.k_proj.weight": "blk.{bid}.attn_k.weight",
    "layers.{bid}.self_attn.v_proj.weight": "blk.{bid}.attn_v.weight",
    "layers.{bid}.self_attn.o_proj.weight": "blk.{bid}.attn_output.weight",
    "layers.{bid}.self_attn.q_norm.weight": "blk.{bid}.attn_q_norm.weight",
    "layers.{bid}.self_attn.k_norm.weight": "blk.{bid}.attn_k_norm.weight",
    "layers.{bid}.mlp.gate_proj.weight":    "blk.{bid}.ffn_gate.weight",
    "layers.{bid}.mlp.up_proj.weight":      "blk.{bid}.ffn_up.weight",
    "layers.{bid}.mlp.down_proj.weight":    "blk.{bid}.ffn_down.weight",
    "layers.{bid}.input_layernorm.weight":  "blk.{bid}.attn_norm.weight",
    "layers.{bid}.post_attention_layernorm.weight": "blk.{bid}.ffn_norm.weight",
    # Global tensors
    "norm.weight": "output_norm.weight",
    # DFlash-specific conditioning tensors
    "fc.weight": "dflash.fc.weight",
    "hidden_norm.weight": "dflash.hidden_norm.weight",
}


def map_tensor_name(hf_name: str, n_layers: int) -> str | None:
    """Map HF tensor name to GGUF tensor name."""
    # Try global tensors first
    if hf_name in TENSOR_MAP:
        return TENSOR_MAP[hf_name]

    # Try block tensors
    for bid in range(n_layers):
        for hf_pattern, gguf_pattern in TENSOR_MAP.items():
            if "{bid}" not in hf_pattern:
                continue
            hf_expected = hf_pattern.format(bid=bid)
            if hf_name == hf_expected:
                return gguf_pattern.format(bid=bid)

    return None


def main():
    parser = argparse.ArgumentParser(description="Convert DFlash drafter to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to DFlash HF model directory")
    parser.add_argument("--outfile", type=Path, required=True, help="Output GGUF file path")
    parser.add_argument("--outtype", choices=["f16", "f32"], default="f16", help="Output data type")
    parser.add_argument("--target-gguf", type=Path, help="Target model GGUF to copy embed_tokens/lm_head from (instead of dummy zeros)")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found in {model_dir}")
        sys.exit(1)

    config = json.loads(config_path.read_text())

    # Extract parameters
    n_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    head_dim = config.get("head_dim", hidden_size // n_heads)
    max_position_embeddings = config.get("max_position_embeddings", 40960)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 1000000.0)

    # DFlash-specific config
    dflash_config = config.get("dflash_config", {})
    block_size = config.get("block_size", 16)
    target_layer_ids = dflash_config.get("target_layer_ids", [])
    mask_token_id = dflash_config.get("mask_token_id", 151669)

    model_name = model_dir.name

    print(f"Converting DFlash model: {model_name}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Heads: {n_heads} (KV: {n_kv_heads})")
    print(f"  Intermediate: {intermediate_size}")
    print(f"  Vocab: {vocab_size}")
    print(f"  Block size: {block_size}")
    print(f"  Target layer taps: {target_layer_ids}")
    print(f"  Mask token ID: {mask_token_id}")
    print(f"  Output type: {args.outtype}")
    print()

    # Read tensor metadata
    tensor_meta = read_safetensors_metadata(model_dir)
    print(f"Found {len(tensor_meta)} tensors in safetensors files")

    # Initialize GGUF writer
    # Use "dflash" architecture — requires LLM_ARCH_DFLASH registered in C++
    writer = gguf.GGUFWriter(str(args.outfile), "dflash")

    # Add standard model parameters
    writer.add_block_count(n_layers)
    writer.add_embedding_length(hidden_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_context_length(max_position_embeddings)
    writer.add_vocab_size(vocab_size)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_rope_dimension_count(head_dim)
    writer.add_causal_attention(True)
    writer.add_name(f"DFlash-{model_name}")
    # Explicit key/value head dimensions — critical for models where
    # hidden_size != n_heads * head_dim (e.g., production DFlash drafter)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)

    # Add DFlash-specific metadata as custom KV pairs
    writer.add_uint32("dflash.block_size", block_size)
    writer.add_uint32("dflash.mask_token_id", mask_token_id)
    writer.add_uint32("dflash.n_target_layers", len(target_layer_ids))
    writer.add_array("dflash.target_layer_ids", target_layer_ids)
    # Phase 1: DFlash-specific tensors (fc, hidden_norm) are NOT included in the GGUF
    # because the Qwen3 loader rejects unknown tensor names. Store their shapes as metadata
    # so Phase 2 can add proper DFlash architecture support.
    fc_meta = tensor_meta.get("fc.weight", {})
    if fc_meta:
        writer.add_array("dflash.fc_shape", fc_meta["shape"])

    # Add vocabulary from tokenizer.json
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    if tokenizer_path.exists():
        print("Found tokenizer.json - extracting vocabulary")
        tokenizer = json.loads(tokenizer_path.read_text())

        # Extract token list from the tokenizer model vocabulary
        vocab = tokenizer.get("model", {}).get("vocab", {})
        merges = tokenizer.get("model", {}).get("merges", [])

        if vocab:
            # Sort by ID
            tokens_by_id = sorted(vocab.items(), key=lambda x: x[1])
            token_list = [t[0].encode("utf-8") for t in tokens_by_id]
            scores = [0.0] * len(token_list)
            token_types = [gguf.TokenType.NORMAL] * len(token_list)

            # Pad to vocab_size if needed
            while len(token_list) < vocab_size:
                token_list.append(f"<unused_{len(token_list)}>".encode("utf-8"))
                scores.append(0.0)
                token_types.append(gguf.TokenType.UNUSED)

            writer.add_tokenizer_model("gpt2")
            writer.add_token_list(token_list[:vocab_size])
            writer.add_token_scores(scores[:vocab_size])
            writer.add_token_types(token_types[:vocab_size])

            if merges:
                # Merges can be ["a b"] strings or [["a", "b"]] pairs
                merge_strs = []
                for m in merges:
                    if isinstance(m, list):
                        merge_strs.append(" ".join(m).encode("utf-8"))
                    else:
                        merge_strs.append(m.encode("utf-8"))
                writer.add_token_merges(merge_strs)

            # Add special tokens from tokenizer_config
            if tokenizer_config_path.exists():
                tc = json.loads(tokenizer_config_path.read_text())
                if "bos_token" in tc:
                    bos = tc["bos_token"]
                    if isinstance(bos, dict):
                        bos = bos.get("content", "")
                    bos_id = vocab.get(bos, -1)
                    if bos_id >= 0:
                        writer.add_bos_token_id(bos_id)
                if "eos_token" in tc:
                    eos = tc["eos_token"]
                    if isinstance(eos, dict):
                        eos = eos.get("content", "")
                    eos_id = vocab.get(eos, -1)
                    if eos_id >= 0:
                        writer.add_eos_token_id(eos_id)

            print(f"  Added {min(len(token_list), vocab_size)} tokens")
        else:
            print("  WARNING: No vocab in tokenizer.json, using dummy vocab")
            _add_dummy_vocab(writer, vocab_size)
    else:
        print("No tokenizer.json found - using dummy vocabulary")
        _add_dummy_vocab(writer, vocab_size)

    # Convert and add tensors
    use_f16 = args.outtype == "f16"

    embd_dtype = gguf.GGMLQuantizationType.F16 if use_f16 else gguf.GGMLQuantizationType.F32

    # 1. Add embedding (token_embd)
    target_embd = None
    target_output = None
    if args.target_gguf and args.target_gguf.exists():
        print(f"\nExtracting embed/lm_head from target GGUF: {args.target_gguf}")
        target_reader = gguf.GGUFReader(str(args.target_gguf))
        for tensor in target_reader.tensors:
            if tensor.name == "token_embd.weight":
                # Dequantize from target's quantization format to f32, then convert to f16
                target_embd = gguf.dequantize(tensor.data, tensor.tensor_type)
                if use_f16:
                    target_embd = target_embd.astype(np.float16)
                print(f"  token_embd.weight: {tensor.shape} (quantized) -> {target_embd.shape} {target_embd.dtype}")
            elif tensor.name == "output.weight":
                target_output = gguf.dequantize(tensor.data, tensor.tensor_type)
                if use_f16:
                    target_output = target_output.astype(np.float16)
                print(f"  output.weight: {tensor.shape} (quantized) -> {target_output.shape} {target_output.dtype}")

    if target_embd is not None:
        print("Using target model's token_embd (dequantized)")
        writer.add_tensor("token_embd.weight", target_embd, raw_dtype=embd_dtype)
    else:
        print("\nGenerating dummy token_embd (shared with target at runtime)...")
        dummy_embd = np.zeros((vocab_size, hidden_size), dtype=np.float16 if use_f16 else np.float32)
        writer.add_tensor("token_embd.weight", dummy_embd, raw_dtype=embd_dtype)

    # 2. Convert model tensors
    print("Converting model tensors...")
    converted = 0
    skipped = []

    for hf_name, meta in sorted(tensor_meta.items()):
        gguf_name = map_tensor_name(hf_name, n_layers)
        if gguf_name is None:
            skipped.append(hf_name)
            continue

        # Load tensor
        arr = read_safetensors_tensor(meta)

        # Convert to target dtype
        if use_f16:
            # 1D tensors (norms) stay f32 for precision
            if len(arr.shape) == 1:
                arr = arr.astype(np.float32)
                writer.add_tensor(gguf_name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            else:
                arr = arr.astype(np.float16)
                writer.add_tensor(gguf_name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            arr = arr.astype(np.float32)
            writer.add_tensor(gguf_name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)

        converted += 1
        print(f"  {hf_name:50s} -> {gguf_name:40s} {meta['shape']}")

    # 3. Add output (lm_head)
    if target_output is not None:
        print("\nUsing target model's output head (lm_head, dequantized)")
        writer.add_tensor("output.weight", target_output, raw_dtype=embd_dtype)
    else:
        print("\nGenerating dummy output head (shared with target at runtime)...")
        dummy_output = np.zeros((vocab_size, hidden_size), dtype=np.float16 if use_f16 else np.float32)
        writer.add_tensor("output.weight", dummy_output, raw_dtype=embd_dtype)

    if skipped:
        print(f"\nSkipped {len(skipped)} unrecognized tensors:")
        for s in skipped:
            print(f"  {s}")

    print(f"\nConverted {converted} tensors + 2 dummy (embed, output)")
    print(f"Writing GGUF to: {args.outfile}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = args.outfile.stat().st_size
    print(f"Done! File size: {file_size / 1024 / 1024:.1f} MB")


def _add_dummy_vocab(writer: gguf.GGUFWriter, vocab_size: int):
    """Add minimal dummy vocabulary for models without tokenizer.json."""
    tokens = [f"<t{i}>".encode("utf-8") for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    token_types = [gguf.TokenType.NORMAL] * vocab_size
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)


if __name__ == "__main__":
    main()
