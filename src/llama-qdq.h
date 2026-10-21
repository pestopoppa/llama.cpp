#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

//
// Block quantize-dequantize ("fake quant", QDQ) of an F32 activation, as a graph op.
//
// Emulates the activation round trips of the DeepSeek-V4.1-Flash reference, which quantizes some
// activations to a low-precision format and immediately dequantizes them back, so that inference
// sees exactly the rounding the model was trained with. The port stores and multiplies those
// activations at full precision; this op lets an experiment put the round trip back.
//
// Normative: DeepSeek-V4.1-Flash-1017 inference/kernel.py
//   act_quant_kernel  (inplace=True, round_scale=True)            -> LLAMA_QDQ_FP8_E4M3_UE8M0_B32
//   fp4_quant_kernel  (inplace=True, scale_dtype=FE8M0, block 32) -> LLAMA_QDQ_FP4_E2M1_UE8M0_B32
//   fp4_quant_kernel  (inplace=True, scale_dtype=FP8,   block 16) -> LLAMA_QDQ_FP4_E2M1_E4M3_B16
//
// Per block of B consecutive elements along ne[0] (all arithmetic in f32, as the reference's
// compute_dtype):
//
//   FP8, ue8m0 scale:  amax = max(absmax, 1e-4)
//                      s    = 2^ceil_log2(amax * f32(1/448))     (fast_round_scale: IEEE bits)
//                      y    = e4m3_rne(clamp(x / s, -448, 448)) * s
//   FP4, ue8m0 scale:  amax = max(absmax, 6 * 2^-126)
//                      s    = 2^ceil_log2(amax * f32(1/6))
//                      y    = e2m1_rne(clamp(x / s, -6, 6)) * s
//   FP4, e4m3 scale:   amax = max(absmax, 6 * 2^-9)
//                      s    = e4m3_rne_satfinite(amax / 6)
//                      y    = e2m1_rne(clamp(x / s, -6, 6)) * s
//
// The products are exact in f32 (and in bf16, the reference's storage type), so the op output is
// the reference's dequantized value bit for bit GIVEN THE SAME INPUT. One delta it cannot remove:
// the reference's input is bf16 and ours is f32, so the block scale and the rounding see the
// unrounded f32 activation. Record that next to any parity claim made with this op.
//
// Implemented with ggml_map_custom1 (an existing op), so ggml's op table, every backend and the
// RPC wire format are untouched. It runs on the CPU backend only; a graph on another backend
// gets it scheduled to the CPU. The op is never inserted unless a caller asks for it: with every
// site disabled the graph is byte-identical to a build without this file.
//

enum llama_qdq_mode {
    LLAMA_QDQ_NONE = 0,
    LLAMA_QDQ_FP8_E4M3_UE8M0_B32,  // window KV, attention output before wo_a (model.py L708/L787/L1044/L1064/L1073)
    LLAMA_QDQ_FP4_E2M1_UE8M0_B32,  // indexer K and Q                         (model.py L546/L553)
    LLAMA_QDQ_FP4_E2M1_E4M3_B16,   // compressed KV latent                    (model.py L761)
};

// block size of a mode (0 for NONE)
int64_t llama_qdq_block_size(enum llama_qdq_mode mode);

const char * llama_qdq_mode_name(enum llama_qdq_mode mode);

// Scalar reference: QDQ of n contiguous floats, n % block_size == 0. x == y is allowed.
// This is what the graph op runs, per block; exposed for tests.
void llama_qdq_ref(enum llama_qdq_mode mode, const float * x, float * y, int64_t n);

// element-level helpers, exposed for tests. Inputs are already clamped to the format's range.
float llama_qdq_round_e4m3(float v); // RNE onto the e4m3fn grid (|v| <= 448)
float llama_qdq_round_e2m1(float v); // RNE onto the e2m1 grid   (|v| <= 6)

// Graph op. Returns x unchanged (no node) for LLAMA_QDQ_NONE. x must be F32 with
// ne[0] % block_size == 0; a non-contiguous x is made contiguous first.
ggml_tensor * llama_build_qdq(ggml_context * ctx, ggml_tensor * x, enum llama_qdq_mode mode);

//
// DS41 site switches. Env LLAMA_DSV41_QDQ, a comma list, default empty (everything off):
//
//   wo_a     attention output before wo_a, FP8 (DS41-W5.1a; main model AND DSpark drafter)
//   kv_win   sliding-window KV, FP8 (DS41-T3.2; main model AND DSpark ring/block KV)
//   kv_comp  compressed KV latent, FP4 + e4m3/16 (DS41-T3.2)
//   idx_k    indexer key, FP4 + ue8m0/32 (DS41-T3.2)
//   idx_q    indexer query, FP4 + ue8m0/32 (DS41-T3.2)
//   kv       = kv_win,kv_comp,idx_k,idx_q (the whole DS41-T3.2 set)
//   all      = everything above
//   none / 0 / empty = off
//
// An unknown token aborts: a misspelt arm must never silently run as the baseline. The parsed
// set is logged once (INFO, "llama_dsv41_qdq: sites = ...") the first time it is read, so a run
// log proves which arm it was.
//
struct llama_dsv41_qdq_sites {
    bool wo_a    = false;
    bool kv_win  = false;
    bool kv_comp = false;
    bool idx_k   = false;
    bool idx_q   = false;

    bool any() const { return wo_a || kv_win || kv_comp || idx_k || idx_q; }
};

// parse a spec string (nullptr == empty). Returns false and sets *bad_token on an unknown token.
bool llama_dsv41_qdq_parse(const char * spec, llama_dsv41_qdq_sites & out, std::string * bad_token = nullptr);

// process-wide sites from LLAMA_DSV41_QDQ, read once
const llama_dsv41_qdq_sites & llama_dsv41_qdq_get();
