#pragma once

// AK-ASM-H5-PORT: FLASH_ATTN_EXT fast path on gfx90a (MI210) through the retargeted AITER FA-v3 fwd hd128 bf16 asm
// kernels (ggml/src/ggml-hip/fa-asm-gfx90a/README.md). Compiled only with GGML_USE_HIP && GGML_HIP_FA_ASM_GFX90A;
// otherwise every entry point is an inline no-op and the existing FA selection is unchanged.

#include "common.cuh"

#include <cstdint>

#if defined(GGML_USE_HIP) && defined(GGML_HIP_FA_ASM_GFX90A)
#define GGML_CUDA_FA_ASM_GFX90A_AVAILABLE

// Shape/type/param predicate only (no mask contents, no device work, no sync). True when the node COULD take the fast
// path: gfx90a device, D 128, supported types, no ALiBi/softcap/sinks, single sequence, enough query rows. Used by
// the CUDA-graph compatibility check: a node for which this is true may synchronize the stream (mask classification),
// so the graph that contains it is never captured.
bool ggml_cuda_fattn_asm_gfx90a_may_apply(int device, const ggml_tensor * dst);

// Runs the fast path and returns true, or returns false having enqueued nothing (the caller then runs the existing
// kernel selection unchanged). Classifies the KQ mask on the device (bottom-right causal / full, with a trailing run
// of -INF columns = KV padding), which synchronizes the stream once per (graph compute, mask tensor).
bool ggml_cuda_fattn_asm_gfx90a_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Test witnesses (exported through the backend reg's get_proc_address; no behaviour depends on them).
uint64_t ggml_cuda_fattn_asm_gfx90a_n_calls();      // FA ops the asm kernel computed since process start
int64_t  ggml_cuda_fattn_asm_gfx90a_min_batch();    // query-row threshold in effect (GGML_HIP_FA_ASM_MIN_BATCH)
bool     ggml_cuda_fattn_asm_gfx90a_usable(int device); // enabled, gfx90a, and both code objects loaded
#else
static inline bool ggml_cuda_fattn_asm_gfx90a_may_apply(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return false;
}
static inline bool ggml_cuda_fattn_asm_gfx90a_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    return false;
}
#endif
