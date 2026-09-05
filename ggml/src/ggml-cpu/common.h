#pragma once

#include "ggml.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#include <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static inline ggml_fp16_t f32_to_f16(float x) {
    return GGML_CPU_FP32_TO_FP16(x);
}

static inline float f16_to_f32(ggml_fp16_t x) {
    return GGML_CPU_FP16_TO_FP32(x);
}

static inline ggml_bf16_t f32_to_bf16(float x) {
    return GGML_FP32_TO_BF16(x);
}

static inline float bf16_to_f32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);
}

static inline float i32_to_f32(int32_t x) {
    return x;
}

static inline int32_t f32_to_i32(float x) {
    return x;
}

static inline float f32_to_f32(float x) {
    return x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
struct type_conversion_table;

template <>
struct type_conversion_table<ggml_fp16_t> {
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
struct type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
struct type_conversion_table<ggml_bf16_t> {
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;
};

template <>
struct type_conversion_table<int32_t> {
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;
};

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    return {ir0, ir1};
}

// ---------------------------------------------------------------------------
// INF-70 SYNC-2: column split for the degenerate single-row case.
//
// get_thread_range() above splits work over ROWS.  At batch 1 almost every elementwise tensor
// in this model is [n, 1] -- one row -- so dr = 1, thread 0 gets the whole tensor and the other
// 47 threads get an empty range and go straight to the barrier.  Measured on qwen4exp at t48
// (D0-b per-node profile): 7.4 ms/token of row-partitioned elementwise work executes on ONE
// thread, 4.7 ms of it in 134 UNARY [10240,1] nodes (the hyper-connection sigmoids).
//
// Elementwise ops have no cross-element dependency, so splitting the single row into column
// chunks is BIT-IDENTICAL -- every output element is computed by the same expression on the
// same input, only on a different thread.  This is the same shape of fix as INF-70 D8's
// (row, column-chunk) split for GET_ROWS.
//
// Opt-in: GGML_ELEM_COLSPLIT=1.  GGML_ELEM_COLSPLIT_MIN (default 1024) is the minimum row
// length -- below it the per-thread dispatch costs more than the work.
static inline bool ggml_elem_colsplit_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_ELEM_COLSPLIT");
        v = (s != NULL && atoi(s) == 1) ? 1 : 0;
    }
    return v != 0;
}

static inline int64_t ggml_elem_colsplit_min(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_ELEM_COLSPLIT_MIN");
        v = s ? atoll(s) : 1024;
        if (v < 1) {
            v = 1;
        }
    }
    return v;
}

// column range for this thread over a single contiguous row of nc elements
static inline std::pair<int64_t, int64_t> get_col_range(const struct ggml_compute_params * params, int64_t nc) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;
    const int64_t dc  = (nc + nth - 1)/nth;
    const int64_t c0  = dc*ith;
    return {c0, MIN(c0 + dc, nc)};
}

// is this node the degenerate one-row elementwise case the column split applies to?
static inline bool ggml_elem_colsplit_applies(const struct ggml_compute_params * params,
                                              const struct ggml_tensor * src0,
                                              const struct ggml_tensor * dst) {
    return ggml_elem_colsplit_enabled()
        && params->nth > 1
        && ggml_nrows(src0) == 1
        && src0->ne[0] >= ggml_elem_colsplit_min()
        && ggml_is_contiguous(src0)
        && ggml_is_contiguous(dst);
}

struct ggml_fa_tile_config {
    static constexpr size_t Q  = GGML_FA_TILE_Q;
    static constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
