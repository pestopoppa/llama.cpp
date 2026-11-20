#pragma once

#include "common.cuh"

// AK-ASM-H10: GQA-packed multi-token verify attention (head_dim 256, f16 or q8_0 KV, n_q 2..16) for CDNA (MFMA).
// All q heads sharing one KV head x all query tokens are packed into 16-row MFMA tiles, so each KV row is read once per
// KV head; q8_0 K/V is dequantized in registers on its way to LDS (no full KV->f16 conversion); split-KV across
// workgroups with an LSE merge. Kill switch: GGML_CUDA_FA_GQA_VERIFY=0 (falls back to the C6 kernel selection).

bool ggml_cuda_fattn_gqa_verify_applicable(int device, const ggml_tensor * dst);

void ggml_cuda_fattn_gqa_verify(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// DTREE-ATTN-ROUTE v2: tree-verify attention (GGML_HIP_TREE_FA=1, default off), head_dim 256, f16 or q8_0 KV, n_q 4..16,
// gqa*n_q <= 128, single sequence, CDNA. Two launches (the stock gv count): a KV-split kernel and an LSE merge, no host read
// of the mask. The split kernel classifies the mask per KV chunk while staging it (DEAD: skipped, OPEN: no mask applied,
// BITS: one uint32 ancestor word per query row applied to the QK^T accumulators in registers, FLOAT: additive mask), so the
// tree block lives inside the last KV-split chunk(s) and any other mask is computed by the same kernel exactly.
bool ggml_cuda_fattn_tree_applicable(int device, const ggml_tensor * dst);

void ggml_cuda_fattn_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
