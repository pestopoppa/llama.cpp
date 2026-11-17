#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// UP-FAROWS-HIP C13 side channel: upload the FA row list of `key` (proc "ggml_backend_fa_rows_set",
// ggml_backend_fa_rows_set_t in ggml.h). Synchronous; see fattn.cu.
bool ggml_cuda_fa_rows_set(int32_t key, const int32_t * rows, int64_t n_rows, int64_t n_slices);
