#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// AK-C22-SMALLOP-FUSE F1 (GGML_CUDA_AK_CPY_BATCH=1, default off): batch a run of same-layout CPY nodes into one launch.
// Returns the number of graph nodes to skip after node i (0 = not fused).
bool     ggml_cuda_ak_cpy_batch_enabled();
int      ggml_cuda_ak_cpy_batch_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
uint64_t ggml_cuda_ak_cpy_batch_n_calls();
