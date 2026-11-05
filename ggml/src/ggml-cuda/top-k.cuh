#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// AK-ASM-H12: without CUB (HIP/MUSA) TOP_K was limited to ncols <= 1024 (bitonic in shared memory), so the
// Q38FN QSA indexer top-k over n_kv cells fell back to the CPU with a D2H/H2D round trip per QSA layer.
// A per-row radix select (after AITER topk_per_row) lifts the limit. GGML_CUDA_TOPK_RADIX=0 restores it.
bool ggml_cuda_top_k_radix_enabled();
