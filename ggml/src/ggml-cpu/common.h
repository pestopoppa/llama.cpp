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
#include <cstdlib>

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
// INF-70 SYNC-10: (row, column-chunk) work split for elementwise kernels.
//
// get_thread_range() above splits ONLY over rows. At batch 1 every elementwise
// tensor in the decode graph is [nc,1,1], so nr = 1 and thread 0 does all the
// work while the other nth-1 threads fall straight through to the barrier.
// Measured on this tree (INF-70 SYNC-1, per-(node,thread) profiler, tip
// c51e4dabf, -t 48): 145 UNARY + 144 MUL nodes/token burn 6.29 ms of critical
// path with thr_mean/thr_max < 0.1.
//
// Elementwise ops compute each output element from the input element(s) at the
// same index, so cutting a row into column chunks and dealing the (row, chunk)
// pairs out to the threads is BIT-IDENTICAL to the single-threaded result by
// construction. This is the same shape as the D8 GET_ROWS fix (bc2834a9b).
//
// INF-70 CHAMPION-1: DEFAULT ON. Bit-identical by construction (see above), measured
// +3.05% served / +8.42% plain. Escape hatch: GGML_ROWCOL_SPLIT=0 restores the upstream
// row-only split with no rebuild. GGML_ROWCOL_MIN_ELEMS sets the smallest row worth cutting.

static inline bool ggml_rowcol_split_enabled(void) {
    static const bool v = [] {
        const char * s = getenv("GGML_ROWCOL_SPLIT");
        return (s == NULL || *s == '\0') ? true : (atoi(s) != 0);
    }();
    return v;
}

// INF-70 SYNC-17 FIX-1: GGML_SCALE_SPLIT gates routing scale_f32 through get_rowcol_split.
// Bit-identical by construction (elementwise), DEFAULT ON; =0 restores the upstream row split.
static inline bool ggml_scale_split_enabled(void) {
    static const bool v = [] {
        const char * s = getenv("GGML_SCALE_SPLIT");
        return (s == NULL || *s == '\0') ? true : (atoi(s) != 0);
    }();
    return v;
}

static inline int64_t ggml_rowcol_min_elems(void) {
    static const int64_t v = []() -> int64_t {
        const char * s = getenv("GGML_ROWCOL_MIN_ELEMS");
        const int64_t d = s ? atoll(s) : 512;
        return d < 0 ? 0 : d;
    }();
    return v;
}

// INF-70 SYNC-10: route sigmoid through the SIMD ggml_vec_sigmoid_f32 instead of a scalar
// libm expf per element. Separate knob from GGML_ROWCOL_SPLIT so the vectorisation and the
// threading can be attributed independently. NOT bit-identical to libm expf -- default OFF.
static inline bool ggml_vec_sigmoid_enabled(void) {
    static const bool v = [] {
        const char * s = getenv("GGML_VEC_SIGMOID");
        return s != NULL && atoi(s) != 0;
    }();
    return v;
}

struct ggml_rowcol_split {
    int64_t ncc;    // column chunks per row
    int64_t cstep;  // elements per column chunk
    int64_t t0;     // first (row, chunk) task of this thread
    int64_t t1;     // one past the last

    // decode task t into (row, [c0,c1))
    inline void unpack(int64_t t, int64_t nc, int64_t & ir, int64_t & c0, int64_t & c1) const {
        ir = t/ncc;
        c0 = (t - ir*ncc)*cstep;
        c1 = MIN(c0 + cstep, nc);
    }
};

// nr: rows to process, nc: elements per row, esz: element size in bytes of the
// widest tensor touched (chunks are aligned to a cache line of it so two
// threads never write the same line).
static inline ggml_rowcol_split get_rowcol_split(const struct ggml_compute_params * params,
                                                 int64_t nr, int64_t nc, size_t esz) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    int64_t ncc   = 1;
    int64_t cstep = nc;

    if (nth > 1 && nr < nth && nc >= ggml_rowcol_min_elems() && ggml_rowcol_split_enabled()) {
        // one cache line of the element type, at least 16 elements
        int64_t align = esz ? (int64_t)(64/esz) : 16;
        if (align < 1) {
            align = 1;
        }
        const int64_t min_chunk = MAX(align, (int64_t) 64);

        ncc   = (nth + nr - 1)/nr;                       // enough chunks to occupy every thread
        cstep = (nc + ncc - 1)/ncc;
        cstep = MAX(cstep, min_chunk);
        cstep = ((cstep + align - 1)/align)*align;       // keep chunks cache-line aligned
        ncc   = (nc + cstep - 1)/cstep;                  // drop chunks the rounding emptied
        if (ncc < 1) {
            ncc   = 1;
            cstep = nc;
        }
    }

    const int64_t nt = nr*ncc;
    const int64_t dt = (nt + nth - 1)/nth;

    ggml_rowcol_split split;
    split.ncc   = ncc;
    split.cstep = cstep;
    split.t0    = MIN(dt*ith, nt);
    split.t1    = MIN(split.t0 + dt, nt);

    return split;
}

struct ggml_fa_tile_config {
    static constexpr size_t Q  = GGML_FA_TILE_Q;
    static constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
