#pragma once

// AK-RF-A3B-ROUTER-FUSE: one kernel for the MoE router GEMV (F32 ffn_gate_inp [n_embd, n_expert] x F32 cur
// [n_embd, n_rows]) and the softmax -> top-k -> get_rows (-> sum_rows/clamp/div renorm) (-> scale) chain that the
// topk-moe fusion covers. Default OFF: GGML_HIP_ROUTER_FUSE=1 enables it for 1 <= n_rows <= 8 (the MTP / tree verify
// widths); GGML_HIP_ROUTER_FUSE_MIN_N=N (default 1) raises the lower bound, e.g. =2 leaves n=1 decode/draft rows on the
// existing MMVF + topk-moe (AITER asm) route. Unset / 0 = upstream route, nothing in this file runs.
//
// Shape: every block computes ROUTER_FUSE_WARPS experts (one 32-lane warp per expert) for all n_rows and writes the
// logits to a pool scratch (n_rows x n_expert floats, L2 resident); the last block to finish (device-scope counter,
// self-resetting) runs the top-k for every row. One launch replaces MUL_MAT + the topk-moe launch (or, at n >= 2 where
// the topk-moe memory-range check rejects the chain on C21, MUL_MAT + soft_max + argsort + get_rows + norm launches).

#include "common.cuh"

#include <cstdint>

#define ROUTER_FUSE_MAX_ROWS 8

// knob + row-count gate (cheap; reads the environment once)
bool ggml_cuda_router_fuse_enabled();
bool ggml_cuda_router_fuse_rows_ok(int64_t n_rows);

// Runs the fused router and returns true, or returns false having enqueued nothing (the caller then runs the
// unfused nodes). mul_mat = the MUL_MAT router node (src0 = gate_inp, src1 = cur); ids = the top-k VIEW of the
// argsort; weights = the last node of the chain; clamp / scale = the CLAMP / SCALE nodes or nullptr.
bool ggml_cuda_op_router_fuse(ggml_backend_cuda_context & ctx,
                              const ggml_tensor *         mul_mat,
                              ggml_tensor *               weights,
                              ggml_tensor *               ids,
                              const ggml_tensor *         clamp,
                              const ggml_tensor *         scale);

// engagement witness: fused launches enqueued since process start (host side; a captured graph counts once)
uint64_t ggml_cuda_router_fuse_n_calls();
