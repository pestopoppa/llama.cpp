// AK-ASM-H13b: hyper-connection seam (qwen4exp build_hc_mix / build_hc_combine) on the device. See hc-fuse.cuh.
// Every fused kernel evaluates the same scalar expressions, in the same order, as the unfused kernels it replaces
// (scale.cu: s*x+b, unary.cu: sigmoid 1/(1+expf(-x)), silu x/(1+expf(-x)), binbcast mul/add, CPU mean_d1: sequential
// sum over ne1 then / ne1).

#include "hc-fuse.cuh"
#include "norm.cuh"

#include <atomic>
#include <cstdlib>

static bool hc_env_flag(const char * name, bool def) {
    const char * e = getenv(name);
    return e ? atoi(e) != 0 : def;
}

bool ggml_cuda_hc_fuse_enabled() {
    static const bool v = hc_env_flag("GGML_CUDA_HC_FUSE", true);
    return v;
}

bool ggml_cuda_hc_rms_fuse_enabled() {
    static const bool v = hc_env_flag("GGML_CUDA_HC_RMS_FUSE", true);  // H13c rms_norm*gamma fusion
    return v;
}

bool ggml_cuda_mean_d1_enabled() {
    static const bool v = hc_env_flag("GGML_CUDA_MEAN_D1", true);
    return v;
}

static std::atomic<uint64_t> g_hc_n_calls{0};

uint64_t ggml_cuda_hc_fuse_n_calls() {
    return g_hc_n_calls.load(std::memory_order_relaxed);
}

static __device__ __forceinline__ float hc_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- MEAN_D1: dst[i0, 0, i2, i3] = (sum_{i1} src[i0, i1, i2, i3]) / ne01, arbitrary f32 strides
static __global__ void k_mean_d1(const char * __restrict__ src, float * __restrict__ dst, const int64_t ne00,
                                 const int64_t ne01, const int64_t ne02, const int64_t ne03, const size_t nb00,
                                 const size_t nb01, const size_t nb02, const size_t nb03, const size_t nb0,
                                 const size_t nb2, const size_t nb3) {
    const int64_t n = ne00 * ne02 * ne03;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t i00 = i % ne00;
        const int64_t i02 = (i / ne00) % ne02;
        const int64_t i03 = i / (ne00 * ne02);
        const char * s = src + i00 * nb00 + i02 * nb02 + i03 * nb03;
        float sum = 0.0f;
        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            sum += *(const float *) (s + i01 * nb01);
        }
        *(float *) ((char *) dst + i00 * nb0 + i02 * nb2 + i03 * nb3) = sum / (float) ne01;
    }
}

static unsigned hc_blocks(int64_t n) {
    const int64_t b = (n + 255) / 256;
    return (unsigned) (b < 65535 ? (b > 0 ? b : 1) : 65535);
}

void ggml_cuda_op_mean_d1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const int64_t n = src->ne[0] * src->ne[2] * src->ne[3];
    k_mean_d1<<<hc_blocks(n), 256, 0, ctx.stream()>>>((const char *) src->data, (float *) dst->data, src->ne[0], src->ne[1],
                                                      src->ne[2], src->ne[3], src->nb[0], src->nb[1], src->nb[2], src->nb[3],
                                                      dst->nb[0], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

// ---- combine: out[e,h,t] = res[e,h,t] + b[e,t] * (s2 * sigmoid(s1 * inj[h,t] + b1) + b2)   (all contiguous)
static __global__ void k_hc_combine(const float * __restrict__ res, const float * __restrict__ bo, const float * __restrict__ inj,
                                    float * __restrict__ out, const int64_t E, const int64_t H, const int64_t T,
                                    const float s1, const float b1, const float s2, const float b2) {
    const int64_t n = E * H * T;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t e = i % E;
        const int64_t h = (i / E) % H;
        const int64_t t = i / (E * H);
        const float   w = s2 * hc_sigmoid(s1 * inj[t * H + h] + b1) + b2;
        out[i] = res[i] + bo[t * E + e] * w;
    }
}

// ---- mix tail: out[e,t] = (sum_h xn[e + h*E, t] * sigmoid(u[e + h*E, t])) / H
static __global__ void k_hc_mix_tail(const float * __restrict__ xn, const float * __restrict__ u, float * __restrict__ out,
                                     const int64_t E, const int64_t H, const int64_t T) {
    const int64_t n = E * T;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t e = i % E;
        const int64_t t = i / E;
        float sum = 0.0f;
        for (int64_t h = 0; h < H; ++h) {
            const int64_t j = t * E * H + h * E + e;
            sum += xn[j] * hc_sigmoid(u[j]);
        }
        out[i] = sum / (float) H;
    }
}

// ---- gate: out = silu(s * x + b)
static __global__ void k_hc_scale_silu(const float * __restrict__ x, float * __restrict__ out, const int64_t n, const float s,
                                       const float b) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const float y = s * x[i] + b;
        out[i] = y / (1.0f + expf(-y));
    }
}

void ggml_cuda_hc_fuse_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_hc_anchor & an) {
    cudaStream_t stream = ctx.stream();
    switch (an.kind) {
        case HC_FUSE_COMBINE: {
            const int64_t E = dst->ne[0], H = dst->ne[1], T = dst->ne[2];
            k_hc_combine<<<hc_blocks(E * H * T), 256, 0, stream>>>((const float *) an.a->data, (const float *) an.b->data,
                (const float *) an.c->data, (float *) dst->data, E, H, T, an.s1, an.b1, an.s2, an.b2);
        } break;
        case HC_FUSE_MIX_TAIL: {
            const int64_t E = dst->ne[0], T = dst->ne[2];
            const int64_t H = ggml_nelements(an.a) / (E * T);
            k_hc_mix_tail<<<hc_blocks(E * T), 256, 0, stream>>>((const float *) an.a->data, (const float *) an.b->data,
                (float *) dst->data, E, H, T);
        } break;
        case HC_FUSE_RMS_GAMMA: {
            // the existing fused rms_norm*mul kernel, given views in the RMS_NORM layout: output [E,hc,T] contiguous is
            // the MUL's [hc*E, T]; gamma [hc*E] is viewed [E,hc,1,1] so it broadcasts over T
            ggml_tensor rms = *an.c;
            ggml_tensor gam = *an.b;
            gam.ne[0] = rms.ne[0]; gam.ne[1] = rms.ne[1]; gam.ne[2] = 1; gam.ne[3] = 1;
            gam.nb[1] = gam.nb[0] * gam.ne[0]; gam.nb[2] = gam.nb[1] * gam.ne[1]; gam.nb[3] = gam.nb[2];
            ggml_tensor mul = *dst;
            for (int d = 0; d < 4; ++d) { mul.ne[d] = rms.ne[d]; }
            mul.nb[0] = sizeof(float);
            for (int d = 1; d < 4; ++d) { mul.nb[d] = mul.nb[d - 1] * mul.ne[d - 1]; }
            mul.src[0] = &rms;
            mul.src[1] = &gam;
            ggml_cuda_op_rms_norm_fused(ctx, &rms, &mul);
        } break;
        case HC_FUSE_SCALE_SILU: {
            const int64_t n = ggml_nelements(dst);
            k_hc_scale_silu<<<hc_blocks(n), 256, 0, stream>>>((const float *) an.a->data, (float *) dst->data, n, an.s1, an.b1);
        } break;
    }
    CUDA_CHECK(cudaGetLastError());
    const uint64_t c = g_hc_n_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (c == 1) {
        GGML_LOG_INFO("ggml_cuda_hc_fuse: hyper-connection fused path ran (call 1) kind=%d node=%s\n", (int) an.kind, dst->name);
    }
}

// ---------------------------------------------------------------------------------------------------- planning
// the planner is host-only C++ in hc-fuse-plan.h (unit-tested on the CPU by tests/test-hc-fuse-plan.cpp)
static std::atomic<uint64_t> g_hc_n_unsafe{0};

void ggml_cuda_hc_fuse_plan(const ggml_cgraph * cgraph, ggml_cuda_hc_plan & plan) {
    plan.skip.clear();
    plan.anchors.clear();
    plan.chains.clear();
    plan.n_unsafe = 0;
    if (!ggml_cuda_hc_fuse_enabled()) {
        return;
    }
    ggml_hc_fuse_plan_build(cgraph, ggml_cuda_hc_rms_fuse_enabled(), plan);
    if (plan.n_unsafe > 0 && g_hc_n_unsafe.fetch_add((uint64_t) plan.n_unsafe, std::memory_order_relaxed) == 0) {
        GGML_LOG_INFO("ggml_cuda_hc_fuse: buffer-lifetime guard kept %d hc fusion(s) unfused (first graph)\n", plan.n_unsafe);
    }
}
