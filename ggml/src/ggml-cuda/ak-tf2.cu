// AK-RF-27B-TAIL-FUSE-2: knobs, witnesses and the elementwise fused kernels (see ak-tf2.cuh).
#include "ak-tf2.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>

#define AK_TF2_NG 9

static std::atomic<uint64_t> g_ak_tf2_n_calls[AK_TF2_NG];

bool ggml_cuda_ak_tf2_enabled(int group) {
    static const int mask = [] {
        int m = 0;
        for (int g = 0; g < AK_TF2_NG; ++g) {
            char name[32];
            snprintf(name, sizeof(name), "GGML_CUDA_AK_TF2_G%d", g);
            const char * e = getenv(name);
            if (e != nullptr && atoi(e) == 1) {
                m |= 1 << g;
            }
        }
        return m;
    }();
    return group >= 0 && group < AK_TF2_NG && (mask >> group) & 1;
}

uint64_t ggml_cuda_ak_tf2_n_calls(int group) {
    return group >= 0 && group < AK_TF2_NG ? g_ak_tf2_n_calls[group].load() : 0;
}

void ggml_cuda_ak_tf2_note(int group, const char * what) {
    if (g_ak_tf2_n_calls[group].fetch_add(1) == 0) {
        GGML_LOG_INFO("ggml_cuda_ak_tf2: AK-RF-27B-TAIL-FUSE-2 G%d %s fused path ran (call 1, GGML_CUDA_AK_TF2_G%d=1)\n",
                      group, what, group);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// G3: gate = softplus(alpha + dt) * a (qwen35 GDN: alpha_biased = ADD(alpha, ssm_dt), SOFTPLUS, MUL(., ssm_a)).
// Unfused: k_bin_bcast add (fp32 a + b), unary_op_kernel op_softplus, k_bin_bcast mul (fp32 a * b). Same expressions here.
static __device__ __forceinline__ float ak_tf2_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));   // = unary.cu op_softplus
}

static __global__ void ak_tf2_g3_kernel(const float * __restrict__ x, const float * __restrict__ dt,
                                        const float * __restrict__ a, float * __restrict__ dst, const int64_t n,
                                        const int nc, const bool a_first, const bool dt_first) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int   c  = (int) (i % nc);
    const float s  = dt_first ? dt[c] + x[i] : x[i] + dt[c];
    const float sp = ak_tf2_softplus(s);
    dst[i] = a_first ? a[c] * sp : sp * a[c];
}

// add = ADD(x, dt) (either operand order), sp = SOFTPLUS(add), mul = MUL(sp, a) (either order). x and mul contiguous F32
// of one shape; dt and a contiguous F32 single rows of ne[0] = x->ne[0] (the per-head broadcast). Uses are checked by the
// caller (ggml_can_fuse_subgraph, only mul is an output).
bool ggml_cuda_ak_tf2_g3_supported(const ggml_tensor * add, const ggml_tensor * sp, const ggml_tensor * mul) {
    if (add->op != GGML_OP_ADD || sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS ||
        mul->op != GGML_OP_MUL || sp->src[0] != add) {
        return false;
    }
    const ggml_tensor * a = mul->src[0] == sp ? mul->src[1] : (mul->src[1] == sp ? mul->src[0] : nullptr);
    if (a == nullptr || a == sp) {
        return false;
    }
    const bool          dt_first = ggml_nrows(add->src[0]) == 1 && ggml_nrows(add->src[1]) != 1;
    const ggml_tensor * x        = dt_first ? add->src[1] : add->src[0];
    const ggml_tensor * dt       = dt_first ? add->src[0] : add->src[1];
    for (const ggml_tensor * t : { x, dt, a, add, sp, mul }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    const int64_t nc = x->ne[0];
    if (!ggml_are_same_shape(x, add) || !ggml_are_same_shape(sp, add) || !ggml_are_same_shape(mul, add) ||
        dt->ne[0] != nc || ggml_nrows(dt) != 1 || a->ne[0] != nc || ggml_nrows(a) != 1 || nc > INT32_MAX) {
        return false;
    }
    // mul may reuse x element-for-element (in-place allocation); any other overlap with an input is refused
    const int64_t mlo = (int64_t) mul->data, mhi = mlo + (int64_t) ggml_nbytes(mul);
    for (const ggml_tensor * t : { dt, a }) {
        const int64_t lo = (int64_t) t->data, hi = lo + (int64_t) ggml_nbytes(t);
        if (!(mhi <= lo || hi <= mlo)) {
            return false;
        }
    }
    const int64_t xlo = (int64_t) x->data, xhi = xlo + (int64_t) ggml_nbytes(x);
    return mhi <= xlo || xhi <= mlo || (mlo == xlo && mhi == xhi);
}

void ggml_cuda_ak_tf2_g3(ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * sp, ggml_tensor * mul) {
    const bool          dt_first = ggml_nrows(add->src[0]) == 1 && ggml_nrows(add->src[1]) != 1;
    const ggml_tensor * x        = dt_first ? add->src[1] : add->src[0];
    const ggml_tensor * dt       = dt_first ? add->src[0] : add->src[1];
    const bool          a_first  = mul->src[1] == sp;
    const ggml_tensor * a        = a_first ? mul->src[0] : mul->src[1];
    const int64_t       n        = ggml_nelements(mul);
    ggml_cuda_ak_tf2_note(3, "gdn gate add+softplus+mul");
    const int nb = (int) ((n + 255) / 256);
    ak_tf2_g3_kernel<<<nb, 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) dt->data,
        (const float *) a->data, (float *) mul->data, n, (int) x->ne[0], a_first, dt_first);
    CUDA_CHECK(cudaGetLastError());
}
