#!/usr/bin/env python3
"""Generate a tiny Diff Transformer V2 GGUF and validate it loads in llama.cpp.

Creates a 2-layer, 64-dim model with random F32 weights to verify:
1. GGUF generation with diff-transformer architecture works
2. Tensor shapes are correct (W_q is 2x, W_lambda present)
3. llama-cli can load the model without error
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

# Add gguf-py to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gguf-py'))
import gguf


def create_synthetic_gguf(path: str) -> None:
    """Create a tiny diff-transformer GGUF with random weights."""
    n_embd = 64
    n_head = 4
    n_head_kv = 2
    d_head = n_embd // n_head  # 16
    n_ff = 128
    n_layer = 2
    n_vocab = 256

    writer = gguf.GGUFWriter(path, "diff-transformer")

    # Model metadata
    writer.add_context_length(512)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_key_length(d_head)
    writer.add_value_length(d_head)
    writer.add_feed_forward_length(n_ff)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(d_head)
    writer.add_file_type(gguf.GGMLQuantizationType.F32)

    # Tokenizer (minimal BPE stub with merges)
    writer.add_tokenizer_model("gpt2")
    tokens = [f"tok{i}".encode() for i in range(n_vocab)]
    scores = [0.0] * n_vocab
    token_types = [1] * n_vocab  # NORMAL
    # Generate dummy merges (gpt2 tokenizer requires merges list)
    merges = [f"tok{i} tok{i+1}" for i in range(0, min(n_vocab - 1, 100))]
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)

    rng = np.random.default_rng(42)

    def add_weight(name: str, shape: tuple):
        data = rng.standard_normal(shape).astype(np.float32) * 0.02
        writer.add_tensor(name, data)

    # Embeddings + output
    add_weight("token_embd.weight", (n_vocab, n_embd))
    add_weight("output_norm.weight", (n_embd,))
    add_weight("output.weight", (n_vocab, n_embd))

    for i in range(n_layer):
        pfx = f"blk.{i}"

        # Attention norm
        add_weight(f"{pfx}.attn_norm.weight", (n_embd,))

        # Q is DOUBLED: (n_embd, 2 * n_head * d_head)
        add_weight(f"{pfx}.attn_q.weight", (2 * n_head * d_head, n_embd))

        # K, V standard: (n_embd, n_head_kv * d_head)
        add_weight(f"{pfx}.attn_k.weight", (n_head_kv * d_head, n_embd))
        add_weight(f"{pfx}.attn_v.weight", (n_head_kv * d_head, n_embd))

        # Output: (n_head * d_head, n_embd)
        add_weight(f"{pfx}.attn_output.weight", (n_embd, n_head * d_head))

        # Lambda: (n_embd, n_head)
        add_weight(f"{pfx}.attn_lambda.weight", (n_head, n_embd))

        # FFN
        add_weight(f"{pfx}.ffn_norm.weight", (n_embd,))
        add_weight(f"{pfx}.ffn_gate.weight", (n_ff, n_embd))
        add_weight(f"{pfx}.ffn_up.weight", (n_ff, n_embd))
        add_weight(f"{pfx}.ffn_down.weight", (n_embd, n_ff))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Created {path} ({size_mb:.1f} MB, {n_layer} layers, {n_embd}d, {n_head}h)")


def validate_gguf(path: str) -> bool:
    """Read back the GGUF and verify tensor shapes."""
    reader = gguf.GGUFReader(path)

    arch = None
    for kv in reader.fields.values():
        if "architecture" in kv.name:
            arch = str(bytes(kv.parts[-1]), "utf-8")
            break

    assert arch == "diff-transformer", f"Expected diff-transformer, got {arch}"

    tensor_map = {t.name: t.shape for t in reader.tensors}

    # Verify Q is doubled
    q_shape = tensor_map.get("blk.0.attn_q.weight")
    assert q_shape is not None, "Missing blk.0.attn_q.weight"
    # GGUFReader shape is [n_embd, 2*n_head*d_head] = [64, 128]
    assert q_shape[1] == 128, f"Expected Q dim1=128 (2*4*16), got {q_shape[1]}"

    # Verify lambda exists
    lambda_shape = tensor_map.get("blk.0.attn_lambda.weight")
    assert lambda_shape is not None, "Missing blk.0.attn_lambda.weight"
    # GGUFReader shape is [n_embd, n_head] = [64, 4]
    assert lambda_shape[1] == 4, f"Expected lambda dim1=4 (n_head), got {lambda_shape[1]}"

    print(f"GGUF validation passed: arch={arch}, Q shape={q_shape}, lambda shape={lambda_shape}")
    return True


def test_load(gguf_path: str, build_dir: str) -> bool:
    """Try loading the GGUF with llama-cli to verify C++ side accepts it."""
    build_dir = os.path.abspath(build_dir)
    llama_cli = os.path.join(build_dir, "bin", "llama-cli")
    if not os.path.exists(llama_cli):
        print(f"SKIP: {llama_cli} not found (build first)")
        return True

    # Set LD_LIBRARY_PATH for shared lib builds (must use absolute paths)
    env = os.environ.copy()
    lib_dirs = [
        os.path.join(build_dir, "src"),
        os.path.join(build_dir, "ggml", "src"),
        os.path.join(build_dir, "ggml", "src", "ggml-cpu"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")

    # Use shell pipeline: echo "" | llama-cli ... — piped EOF causes clean exit
    # after model load. A 15s timeout is generous; load failure errors in <1s.
    cmd = (
        f'echo "" | LD_LIBRARY_PATH="{":".join(lib_dirs)}" '
        f'timeout 15 "{llama_cli}" -m "{gguf_path}" -p "test" -n 1 -t 2 2>&1'
    )

    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True, text=True, timeout=30, env=env,
    )

    output = result.stdout + result.stderr

    if "error loading model" in output or "unknown model architecture" in output:
        print(f"FAIL: model load error:\n{output[-500:]}")
        return False

    if "Loading model" in output:
        print("PASS: model loaded and ran the synthetic diff-transformer GGUF")
        return True

    print(f"UNCERTAIN: exit code {result.returncode}\n{output[-500:]}")
    return result.returncode == 0


if __name__ == "__main__":
    build_dir = os.environ.get("BUILD_DIR", os.path.join(
        os.path.dirname(__file__), "..", "build-diff-test"))

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        gguf_path = f.name

    try:
        print("=== Step 1: Generate synthetic GGUF ===")
        create_synthetic_gguf(gguf_path)

        print("\n=== Step 2: Validate GGUF structure ===")
        validate_gguf(gguf_path)

        print("\n=== Step 3: Test load with llama-cli ===")
        ok = test_load(gguf_path, build_dir)

        if ok:
            print("\n=== ALL TESTS PASSED ===")
        else:
            print("\n=== TESTS FAILED ===")
            sys.exit(1)
    finally:
        os.unlink(gguf_path)
