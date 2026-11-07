#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// [C8-6-GDN-CHUNKED] chunked WY prefill path: launches since process start, effective n_tokens threshold
// (INT32_MAX when GGML_CUDA_GDN_CHUNKED=0), and whether the device can run it (CDNA MFMA). Test witnesses.
uint64_t ggml_cuda_gdn_chunked_n_calls();
int32_t  ggml_cuda_gdn_chunked_min_tokens();
bool     ggml_cuda_gdn_chunked_usable(int device);
