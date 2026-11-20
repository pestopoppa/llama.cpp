#pragma once

#include "common.cuh"

// AK-ASM-H10: GQA-packed multi-token verify attention (head_dim 256, f16 or q8_0 KV, n_q 2..16) for CDNA (MFMA).
// All q heads sharing one KV head x all query tokens are packed into 16-row MFMA tiles, so each KV row is read once per
// KV head; q8_0 K/V is dequantized in registers on its way to LDS (no full KV->f16 conversion); split-KV across
// workgroups with an LSE merge. Kill switch: GGML_CUDA_FA_GQA_VERIFY=0 (falls back to the C6 kernel selection).

bool ggml_cuda_fattn_gqa_verify_applicable(int device, const ggml_tensor * dst);

void ggml_cuda_fattn_gqa_verify(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// DTREE-ATTN-ROUTE: tree-verify attention (GGML_HIP_TREE_FA=1, default off), head_dim 256, f16 or q8_0 KV, n_q 4..16,
// single sequence, CDNA. Three device stages on one stream, no host read of the mask:
//   0. tfa_analyze: one pass over the KQ mask -> per 256-column block (last live column, first column whose rows differ,
//      non-binary flag). Every later kernel folds these into (ok, t0): the mask is tree-shaped when it is binary, every
//      column below t0 = last_live - 32 is identical across the query rows, and the tree lives in the 32-column tail.
//   1. prefix: the gv_fattn KV-split kernel over [0, t0) reading ONE mask row (broadcast) instead of n_q rows;
//   2. tail: a masked MFMA 16x16x16 kernel over the 32 tail columns with a per-row uint32 ancestor word in registers,
//      applied to the QK^T accumulators;
//   3. an LSE merge of the prefix splits and the tail.
// When the mask is not tree-shaped (ok = 0) stage 1 runs the full dense-mask gv loop and stage 2 contributes nothing,
// so the result is the gv result for any mask.
bool ggml_cuda_fattn_tree_applicable(int device, const ggml_tensor * dst);

void ggml_cuda_fattn_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
