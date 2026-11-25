#pragma once
// HBM-knee helper (ak-q8-hbm-knee-map-gpu-20261125, MI210): HBM read streams peak at ~16 KB per CU in flight
// (1512-1536 GB/s, 92-94 % of 1638) and lose 17-29 % at 64 KB/CU; GEMV-sized (33-95 MB) coherent or per-WG-range
// streams reach 1350-1486 GB/s at 16 KB/CU; a 4-row strided front costs ~6 %. Under-filled launches (few WGs, scalar
// loads) sit far below the knee instead. This helper sizes a persistent grid so that
//     waves_per_cu x bytes_in_flight_per_wave ~= GGML_HIP_KNEE_KB (default 16) KB
// and hands each workgroup ONE contiguous range of the linear work index (no grid-stride interleave: every WG sweeps its
// own front, so the GPU reads ~grid fronts, not grid x blockDim scattered addresses).
// Each op that uses it has its own default-off knob; this header only computes the shape.

#include "common.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

static inline int ggml_cuda_knee_kb() {
    static const int kb = [] {
        const char * v = getenv("GGML_HIP_KNEE_KB");
        const int    k = (v && *v) ? atoi(v) : 16;
        return std::max(1, std::min(256, k));
    }();
    return kb;
}

// knob helper: GGML_HIP_KNEE_<OP>=1 enables the op's knee variant (default 0 = stock path, byte-identical)
static inline bool ggml_cuda_knee_env(const char * name) {
    const char * v = getenv(name);
    return v && *v && atoi(v) != 0;
}

// Persistent workgroup count for a launch of `block_threads`-thread WGs where each wave keeps `wave_bytes_in_flight`
// bytes of HBM loads outstanding. Capped by `n_work_wg` (never more WGs than there is work for one WG each).
static inline int64_t ggml_cuda_knee_grid(int device, int block_threads, int64_t wave_bytes_in_flight, int64_t n_work_wg) {
    const int     nsm        = ggml_cuda_info().devices[device].nsm;
    const int     wsize      = ggml_cuda_info().devices[device].warp_size;
    const int64_t waves_wg   = std::max<int64_t>(1, (block_threads + wsize - 1) / wsize);
    const int64_t budget     = (int64_t) ggml_cuda_knee_kb() * 1024;
    const int64_t waves_cu   = std::max<int64_t>(1, budget / std::max<int64_t>(1, wave_bytes_in_flight));
    const int64_t wg_per_cu  = std::max<int64_t>(1, waves_cu / waves_wg);
    return std::max<int64_t>(1, std::min<int64_t>(n_work_wg, (int64_t) nsm * wg_per_cu));
}

// device: the contiguous [i0, i1) slice of n work items owned by workgroup `wg` of `nwg`
static __device__ __forceinline__ void ggml_cuda_knee_range(const int64_t n, const int64_t wg, const int64_t nwg,
                                                            int64_t & i0, int64_t & i1) {
    i0 = wg * n / nwg;
    i1 = (wg + 1) * n / nwg;
}
