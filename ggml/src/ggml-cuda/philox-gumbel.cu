// [GPU-RESIDENT-ROUND-2f] device Philox-Gumbel noise for the DFlash2 lattice selector (see ggml.h ggml_philox_gumbel)
#include "philox-gumbel.cuh"
#include "../ggml-philox.h"

static __global__ void k_philox_gumbel(const int32_t * key, size_t key_nb1, const char * ctr, size_t ctr_s,
                                       float * dst, size_t nb1, size_t nb2, int64_t ne0, int64_t ne1, int64_t n_b, int32_t skip) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne0*ne1*n_b) {
        return;
    }
    const int64_t i0 = i % ne0, i1 = (i / ne0) % ne1, b = i / (ne0*ne1);
    const int32_t * k = (const int32_t *) ((const char *) key + b*key_nb1);
    const float     c = *(const float *) (ctr + b*ctr_s);
    *(float *) ((char *) dst + i0*sizeof(float) + i1*nb1 + b*nb2) =
        ggml_philox_gumbel_at((uint32_t) k[0], (uint32_t) k[1], c, i0 + i1*ne0, skip);
}

void ggml_cuda_op_philox_gumbel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * key = dst->src[0];
    const ggml_tensor * ctr = dst->src[1];
    const int32_t skip = ggml_get_op_params_i32(dst, 0);
    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], n_b = dst->ne[2];
    const size_t  cs  = ctr->ne[0] == n_b ? ctr->nb[0] : ctr->nb[1];
    const int64_t n   = ne0*ne1*n_b;
    if (n == 0) {
        return;
    }
    const int bs = 256;
    k_philox_gumbel<<<(n + bs - 1)/bs, bs, 0, ctx.stream()>>>((const int32_t *) key->data, key->nb[1], (const char *) ctr->data, cs,
        (float *) dst->data, dst->nb[1], dst->nb[2], ne0, ne1, n_b, skip);
}
