#pragma once

// C15-EPI-Q8: producer-epilogue q8_1 write (GGML_HIP_EPILOGUE_Q8=1, default off).
// A producer kernel whose output feeds a q8_1 MMVQ also writes the q8_1 blocks of that output, so the MMVQ
// finds them in its graph-capture cache and skips its own quantize_q8_1 launch. The producer still writes f32.
//
// Contract: the calling thread holds output element i, and the 32 lanes holding elements
// [32*(i/32), 32*(i/32)+32) are 32 consecutive lanes of one warp/wave that all reach this call together.
// The math is the same as quantize_q8_1(_1d)<false>: d = amax/127, q = roundf(x/d), ds = (d, float sum).
// amax is an exact max, so d and q (all a q8_0/q6_k MMVQ reads) are bit-identical to the standalone quantize.

#include "common.cuh"

static __device__ __forceinline__ void epilogue_q8_1_store(const float xi, block_q8_1 * y, const int64_t i) {
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float sum  = warp_reduce_sum<QK8_1>(xi);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    const int64_t ib  = i / QK8_1;
    const int     iqs = i % QK8_1;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
}
