#pragma once

// EXL3 (exllamav3 trellis, mul1 codebook) matrix executor for the CPU backend.
// Runs MUL_MAT / MUL_MAT_ID whose src0 is GGML_TYPE_EXL3_<K>; the per-tensor Hadamard
// vectors ride on the op as extra sources (MUL_MAT: src2 = svh, src3 = suh;
// MUL_MAT_ID: src3 = svh, src4 = suh). See EXL3-CPU-DESIGN.md at the repository root.

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

bool   ggml_cpu_exl3_supports(const struct ggml_tensor * op);
size_t ggml_cpu_exl3_work_size(const struct ggml_tensor * dst, int n_threads);
void   ggml_cpu_exl3_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// test hooks: active ISA tier name ("scalar", "vnni", "vbmi") and an override
// (also settable with env GGML_EXL3_ISA=scalar|vnni|vbmi, which can only lower the tier)
GGML_BACKEND_API const char * ggml_cpu_exl3_isa(void);
GGML_BACKEND_API void ggml_cpu_exl3_set_isa(const char * name);
// test hook: largest per-expert instance count still routed to the int8 GEMV path (default 8)
GGML_BACKEND_API void ggml_cpu_exl3_set_gemv_max_rows(int rows);
// test hook: GEMV split-K target unit count (default 64: ops with < 32 units split; env GGML_EXL3_SPLITK_TARGET); 1 disables
GGML_BACKEND_API void ggml_cpu_exl3_set_splitk_target(int target);

#ifdef __cplusplus
}
#endif
