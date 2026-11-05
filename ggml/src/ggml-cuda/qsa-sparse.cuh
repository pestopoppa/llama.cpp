#pragma once

#include "common.cuh"

// AK-ASM-H12: fused QSA sparse attention.
//
// Qwen4exp (Q38FN) QSA layers build their attention as a dense FLASH_ATTN_EXT over the whole KV
// cache with a mask that is -inf everywhere except at the cells named by the indexer top-k
// (src/models/qwen4exp.cpp build_attn_qsa):
//
//   fill(-inf) -> set_rows(zeros, top_k) -> add(kq_mask) -> FLASH_ATTN_EXT(q, k, v, mask)
//
// ggml_cuda_qsa_sparse_try() recognises that mask chain at the FLASH_ATTN_EXT node and, instead of
// the dense FA, runs one kernel that gathers only the top-k K/V rows and does GQA attention over
// them (one K/V read shared by all q heads of a kv head). Returns false (caller runs the normal FA)
// when the pattern does not match, the shapes/types are unsupported, or GGML_CUDA_QSA_SPARSE=0.
bool ggml_cuda_qsa_sparse_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
