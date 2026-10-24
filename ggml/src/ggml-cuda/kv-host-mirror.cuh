#pragma once

// HSF-5.P (experimental): CUDA/HIP implementation of ggml-kv-mirror.h.
// Only reachable through ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_mirror_get_ops").

#include "ggml-kv-mirror.h"

const struct ggml_kv_mirror_ops * ggml_backend_cuda_kv_mirror_get_ops(void);
