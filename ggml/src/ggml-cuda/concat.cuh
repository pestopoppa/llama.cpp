#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// AK-C22-SMALLOP-FUSE-2 C2A test witness: flat non-contiguous concat launches since process start
uint64_t ggml_cuda_ak_concat_flat_n_calls();
