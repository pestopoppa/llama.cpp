#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cinttypes>
#include <cstdlib>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

#if defined(__gfx90a__)
static __device__ __forceinline__ float reduce_q4_K_halfwave_gfx90a(float value) {
    // DPP reads need two wait states after a VGPR write on gfx90a.
    asm volatile(
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0"
        : "+v"(value));

    return value;
}

static __device__ __forceinline__ void reduce_q4_K_halfwave_gfx90a(float & value, float & gate) {
    // Interleave independent accumulators to provide the two DPP wait states.
    asm volatile(
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0"
        : "+v"(value), "+v"(gate));
}

static __device__ __forceinline__ float reduce_q6_K_wave_gfx90a(float value) {
    asm volatile(
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:0"
        : "+v"(value));

    return value;
}

static __device__ __forceinline__ void reduce_q6_K_wave_gfx90a(float & value, float & gate) {
    asm volatile(
        "s_nop 1\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0\n\t"
        "v_add_f32_dpp %0, %0, %0 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:0\n\t"
        "s_nop 0\n\t"
        "v_add_f32_dpp %1, %1, %1 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:0"
        : "+v"(value), "+v"(gate));
}
#endif // defined(__gfx90a__)

static __device__ __forceinline__ float2 vec_dot_q4_K_q8_1_dual(
        const void * __restrict__ vbq, const void * __restrict__ vgate,
        const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_K * bq4_K      = (const block_q4_K *) vbq   + kbx;
    const block_q4_K * bq4_K_gate = (const block_q4_K *) vgate + kbx;

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int q4_offset = 16 * bq8_offset + 4 * ((iqs/2)%4);

    const int * q4      = (const int *) (bq4_K->qs      + q4_offset);
    const int * q4_gate = (const int *) (bq4_K_gate->qs + q4_offset);
    const int v0 = q4[0];
    const int v1 = q4[4];
    const int v0_gate = q4_gate[0];
    const int v1_gate = q4_gate[4];

    const uint16_t * scales      = (const uint16_t *) bq4_K->scales;
    const uint16_t * scales_gate = (const uint16_t *) bq4_K_gate->scales;
    const int j = bq8_offset/2;
    const uint8_t * sc;
    const uint8_t * sc_gate;
#if defined(GGML_USE_HIP)
    uint32_t aux = 0;
    uint32_t aux_gate = 0;
#if defined(__gfx90a__)
    if ((threadIdx.x & 3) == 0) {
#endif
        const int is = j & 1;
        const uint32_t s0 = scales[is + 0];
        const uint32_t s1 = scales[is + 2];
        const uint32_t s2 = scales[is + 4];
        const uint32_t s01 = __builtin_amdgcn_perm(s1, s0, 0x05040100);
        const uint32_t aux_low = s01 & 0x3f3f3f3f;
        const uint32_t aux_high = (__builtin_amdgcn_perm(s2 >> 4, s2, 0x05040100) & 0x0f0f0f0f) |
                                  ((s01 & 0xc0c0c0c0) >> 2);
        aux = j < 2 ? aux_low : aux_high;

        const uint32_t s0_gate = scales_gate[is + 0];
        const uint32_t s1_gate = scales_gate[is + 2];
        const uint32_t s2_gate = scales_gate[is + 4];
        const uint32_t s01_gate = __builtin_amdgcn_perm(s1_gate, s0_gate, 0x05040100);
        const uint32_t aux_low_gate = s01_gate & 0x3f3f3f3f;
        const uint32_t aux_high_gate =
            (__builtin_amdgcn_perm(s2_gate >> 4, s2_gate, 0x05040100) & 0x0f0f0f0f) |
            ((s01_gate & 0xc0c0c0c0) >> 2);
        aux_gate = j < 2 ? aux_low_gate : aux_high_gate;
#if defined(__gfx90a__)
    }
    aux = __builtin_amdgcn_mov_dpp(aux, 0x00, 0xf, 0xf, false);
    aux_gate = __builtin_amdgcn_mov_dpp(aux_gate, 0x00, 0xf, 0xf, false);
#endif
    sc = (const uint8_t *) &aux;
    sc_gate = (const uint8_t *) &aux_gate;
#else
    uint16_t aux[2];
    uint16_t aux_gate[2];
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
        aux_gate[0] = scales_gate[j+0] & 0x3f3f;
        aux_gate[1] = scales_gate[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
        aux_gate[0] = ((scales_gate[j+2] >> 0) & 0x0f0f) | ((scales_gate[j-2] & 0xc0c0) >> 2);
        aux_gate[1] = ((scales_gate[j+2] >> 4) & 0x0f0f) | ((scales_gate[j-0] & 0xc0c0) >> 2);
    }
    sc = (const uint8_t *) aux;
    sc_gate = (const uint8_t *) aux_gate;
#endif
    const uint8_t * m      = sc      + 2;
    const uint8_t * m_gate = sc_gate + 2;
    const bool sum_lane = iqs % QI8_1 == 0;

#if defined(__gfx90a__)
    using floatx2_t = __attribute__((ext_vector_type(2))) float;
    floatx2_t sumf_d = {0.0f, 0.0f};
    floatx2_t sumf_m = {0.0f, 0.0f};
#else
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    float sumf_d_gate = 0.0f;
    float sumf_m_gate = 0.0f;
#endif

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        const half2 ds8 = bq8i->ds;
        const float d8 = __low2float(ds8);
        const float s8 = sum_lane ? __high2float(ds8) : 0.0f;
        const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);
        const int u0 = q8[0];
        const int u1 = q8[4];

        const int dot0 = ggml_cuda_dp4a((v0 >> (4*i)) & 0x0F0F0F0F, u0, 0);
        const int dot1 = ggml_cuda_dp4a((v1 >> (4*i)) & 0x0F0F0F0F, u1, 0);
        const int dot0_gate = ggml_cuda_dp4a((v0_gate >> (4*i)) & 0x0F0F0F0F, u0, 0);
        const int dot1_gate = ggml_cuda_dp4a((v1_gate >> (4*i)) & 0x0F0F0F0F, u1, 0);
#if defined(__gfx90a__)
        const floatx2_t d8_packed = {d8, d8};
        const floatx2_t s8_packed = {s8, s8};
        const floatx2_t dot_packed = {
            float((dot0 + dot1) * sc[i]), float((dot0_gate + dot1_gate) * sc_gate[i])};
        const floatx2_t min_packed = {float(m[i]), float(m_gate[i])};
        sumf_d = __builtin_elementwise_fma(d8_packed, dot_packed, sumf_d);
        sumf_m = __builtin_elementwise_fma(s8_packed, min_packed, sumf_m);
#else
        sumf_d += d8 * ((dot0 + dot1) * sc[i]);
        sumf_m += s8 * m[i];
        sumf_d_gate += d8 * ((dot0_gate + dot1_gate) * sc_gate[i]);
        sumf_m_gate += s8 * m_gate[i];
#endif
    }

#if defined(__gfx90a__)
    union {
        half2 h;
        uint32_t u;
    } dm4_packed = {}, dm4_gate_packed = {};
    if ((threadIdx.x & 3) == 0) {
        dm4_packed.h = bq4_K->dm;
        dm4_gate_packed.h = bq4_K_gate->dm;
    }
    dm4_packed.u = __builtin_amdgcn_mov_dpp(dm4_packed.u, 0x00, 0xf, 0xf, false);
    dm4_gate_packed.u = __builtin_amdgcn_mov_dpp(dm4_gate_packed.u, 0x00, 0xf, 0xf, false);
    const float2 dm4 = __half22float2(dm4_packed.h);
    const float2 dm4_gate = __half22float2(dm4_gate_packed.h);
    return make_float2(
        dm4.x*sumf_d[0] - dm4.y*sumf_m[0],
        dm4_gate.x*sumf_d[1] - dm4_gate.y*sumf_m[1]);
#else
    const float2 dm4 = __half22float2(bq4_K->dm);
    const float2 dm4_gate = __half22float2(bq4_K_gate->dm);
    return make_float2(
        dm4.x*sumf_d - dm4.y*sumf_m,
        dm4_gate.x*sumf_d_gate - dm4_gate.y*sumf_m_gate);
#endif
}

struct q4_K_b4_weight {
    int q[2*QR4_K];
    int sc[QR4_K];
    int m[QR4_K];
    float2 dm;
};

static __device__ __forceinline__ q4_K_b4_weight load_q4_K_b4_weight(
        const block_q4_K * __restrict__ bq4_K, const int & iqs) {
    q4_K_b4_weight weight;

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int q4_offset = 16 * bq8_offset + 4 * ((iqs/2)%4);
    const int * q4 = (const int *) (bq4_K->qs + q4_offset);
    const int v0 = q4[0];
    const int v1 = q4[4];

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        weight.q[2*i + 0] = (v0 >> (4*i)) & 0x0F0F0F0F;
        weight.q[2*i + 1] = (v1 >> (4*i)) & 0x0F0F0F0F;
    }

    const uint16_t * scales = (const uint16_t *) bq4_K->scales;
    const int j = bq8_offset/2;
    uint32_t aux = 0;
#if defined(GGML_USE_HIP)
#if defined(__gfx90a__)
    if ((threadIdx.x & 3) == 0) {
#endif
        const int is = j & 1;
        const uint32_t s0 = scales[is + 0];
        const uint32_t s1 = scales[is + 2];
        const uint32_t s2 = scales[is + 4];
        const uint32_t s01 = __builtin_amdgcn_perm(s1, s0, 0x05040100);
        const uint32_t aux_low = s01 & 0x3f3f3f3f;
        const uint32_t aux_high = (__builtin_amdgcn_perm(s2 >> 4, s2, 0x05040100) & 0x0f0f0f0f) |
                                  ((s01 & 0xc0c0c0c0) >> 2);
        aux = j < 2 ? aux_low : aux_high;
#if defined(__gfx90a__)
    }
    aux = __builtin_amdgcn_mov_dpp(aux, 0x00, 0xf, 0xf, false);
#endif
#else
    uint16_t aux16[2];
    if (j < 2) {
        aux16[0] = scales[j+0] & 0x3f3f;
        aux16[1] = scales[j+2] & 0x3f3f;
    } else {
        aux16[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux16[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    aux = uint32_t(aux16[0]) | (uint32_t(aux16[1]) << 16);
#endif

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        weight.sc[i] = (aux >> (8*i)) & 0xff;
        weight.m[i] = (aux >> (8*(i + 2))) & 0xff;
    }
    weight.dm = __half22float2(bq4_K->dm);
    return weight;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_b4(
        const q4_K_b4_weight & weight, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int q8_offset = (iqs/2)%4;
    const bool sum_lane = iqs % QI8_1 == 0;
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        const int * q8 = (const int *) bq8i->qs + q8_offset;
        const int dot = ggml_cuda_dp4a(
            weight.q[2*i + 1], q8[4], ggml_cuda_dp4a(weight.q[2*i + 0], q8[0], 0));
        const half2 ds8 = bq8i->ds;
        sumf_d += __low2float(ds8) * (dot * weight.sc[i]);
        if (sum_lane) {
            sumf_m += __high2float(ds8) * weight.m[i];
        }
    }

    return weight.dm.x*sumf_d - weight.dm.y*sumf_m;
}

static bool ggml_cuda_log_mmvq_route_enabled() {
    static const bool enabled = []() {
        const char * s = getenv("GGML_CUDA_LOG_MMVQ_ROUTE");
        return s != nullptr && atoi(s) != 0;
    }();
    return enabled;
}

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    const auto log_decision = [type, cc, ne11](bool decision) {
        if (ggml_cuda_log_mmvq_route_enabled() && type == GGML_TYPE_Q8_0) {
            GGML_LOG_INFO("GGML_CUDA_MMVQ_ROUTE_DECISION type=%s cc=%d ne11=%" PRId64 " use_mmvq=%d\n",
                ggml_type_name(type), cc, ne11, decision ? 1 : 0);
        }
        return decision;
    };

    if (!ggml_is_quantized(type)) {
        return log_decision(false);
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return log_decision(ne11 <= 7);
                case GGML_TYPE_Q5_1:
                    return log_decision(ne11 <= 7);
                case GGML_TYPE_Q8_0:
                    return log_decision(ne11 <= 6);
                case GGML_TYPE_Q2_K:
                    return log_decision(ne11 <= 4);
                case GGML_TYPE_Q3_K:
                    return log_decision(ne11 <= 3);
                case GGML_TYPE_Q4_K:
                    return log_decision(ne11 <= 2);
                case GGML_TYPE_Q5_K:
                    return log_decision(ne11 <= 3);
                case GGML_TYPE_Q6_K:
                    return log_decision(ne11 <= 4);
                case GGML_TYPE_IQ1_S:
                    return log_decision(ne11 <= 5);
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return log_decision(ne11 <= 6);
                default:
                    return log_decision(ne11 <= MMVQ_MAX_BATCH_SIZE);
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return log_decision(ne11 <= 5);
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q5_K:
                return log_decision(ne11 <= 3);
            case GGML_TYPE_Q4_K:
                return log_decision(ne11 <= MMVQ_MAX_BATCH_SIZE);
            case GGML_TYPE_Q6_K:
                return log_decision(ne11 <= 5);
            // akm-cdna2-q8-b4-mmvq-route: route Q8_0 at ne11<=4 (MTP verify width) through the
            // ncols_dst<=4 mul_mat_vec_q templates instead of mul_mat_q. At ne11=4 MMQ pads to
            // J=8 and runs occupancy-1 128-row stream-k tiles for 4 real columns, leaving the
            // dec-b4 surface at ~29% of HBM bandwidth; the multi-column MMVQ shares each weight
            // read across 4 register accumulators. ne11>=5 stays on MMQ (the July crossover).
            case GGML_TYPE_Q8_0:
                return log_decision(ne11 <= 4);
            default:
                return log_decision(ne11 <= MMVQ_MAX_BATCH_SIZE);
        }
    }
    return log_decision(ne11 <= MMVQ_MAX_BATCH_SIZE);
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(
        ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool fixed_1536_cdna2 = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        if (ncols_dst == 1 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) && fixed_1536_cdna2) {
            return 4;
        }
        // CDNA2 single-stream experiment (mi210-q8-dequant handoff, lever 2/3): batch-1 Q8_0 GEMV is
        // achieved-BW/occupancy-limited at nwarps=2 (128 thr/block). Raise warps-per-block to put more
        // weight-load requests in flight (Little's law). RDNA4 already uses nwarps=8 for Q8_0. Same
        // dp4a math, but the fp cross-warp reduction is split more ways, so numerically-valid-not-
        // bit-exact (test-backend-ops MUL_MAT 1103/1103 pass). Measured +4.6% (28.99->30.32 t/s, 27B
        // Q8) single-stream tg128; nwarps=4 beat 2 (baseline) and 8 (reduction-overhead-bound).
        if (ncols_dst == 1 && type == GGML_TYPE_Q8_0) {
            return 4;
        }
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    // akm-cdna2-q8-b4-y-stream-amortize: at ncols_dst=4 the 4-column Q8_1 activation row does not
    // depend on the row index, so each block re-reads it from L2/L1 once per row group. 4 rows per
    // block halves that y re-read traffic versus 2 (y ~1.06x of the weight bytes at 4 rows vs
    // ~2.1x at 2), raising the kernel's L2 read ceiling toward the ncols_dst=1 kernel's. GCN
    // (CDNA2) only; the small_k instantiation keeps rows_per_cuda_block=2 as the host-side
    // fallback for row counts that are not multiples of 4.
    if (table_id == MMVQ_PARAMETERS_GCN && type == GGML_TYPE_Q8_0 && ncols_dst == 4 && !small_k) {
        return 4;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// ============================================================================================
// CDNA2 (gfx90a) batch-1 Q8_0 GEMV: async weight-prefetch / LDS double-buffering.
//   Experimental — mi210-q8-dequant handoff, converged do-first lever. All numbers OBSERVATIONS.
//   Premise: the Q8_0 GEMV is already int8-native (dp4a); the 47->62% roofline gap is a batch-1
//   memory-level-parallelism wall, not a dequant-compute gap. Lever: software-pipeline the next
//   weight tile's global->LDS DMA under the current tile's dp4a to keep more HBM requests in
//   flight (Little's law). Uses the MUBUF direct-to-LDS path (llvm.amdgcn.raw.buffer.load.lds,
//   supported on gfx9/CDNA2) — NOT __builtin_amdgcn_global_load_lds (gfx940-gated, mis-encodes
//   on gfx90a). CP_ASYNC_AVAILABLE is CUDA-only, so there is no cp.async on the AMD path today.
//   Gated to Q8_0 + ncols_dst==1 + dense(ids==null) + no-fusion + CDNA2; every other path is
//   byte-identical. Runtime-selected by GGML_CUDA_Q8_PREFETCH (0=off, 1=on cached, 2=on SLC).
// ============================================================================================
#if defined(GGML_USE_HIP)
#define Q8_LDS_PREFETCH_COMPILED 1

typedef int __attribute__((ext_vector_type(4))) q8pf_i32x4_t;

// gfx9 raw buffer resource (V#): [base:64][num_records:32][cfg:32].
// cfg = 0x00020000 for __gfx9__/CDNA per CK ck.hpp:76-77 (CK_BUFFER_RESOURCE_3RD_DWORD).
static __device__ __forceinline__ q8pf_i32x4_t q8pf_make_rsrc(const void * base, uint32_t num_bytes) {
    union { q8pf_i32x4_t v; struct { const void * p; uint32_t range; uint32_t cfg; } s; } u;
    u.s.p     = base;
    u.s.range = num_bytes;
    u.s.cfg   = 0x00020000u;
    return u.v;
}

// LLVM intrinsic: global->LDS DMA, one DWORD per lane, bypasses VGPRs.
// args: (rsrc, lds_ptr[addrspace 3], size, voffset, soffset, imm_offset, aux/cachepolicy)
__device__ void q8pf_raw_buffer_load_lds(
    q8pf_i32x4_t rsrc, __attribute__((address_space(3))) uint32_t * lds_ptr,
    int size, int voffset, int soffset, int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

// One CUDA block = one output row (rows_per_cuda_block == 1). nwarps warps * 64 lanes.
// WAUX: cachepolicy for the weight DMA (0 = default-cached, 2 = SLC/streaming-nontemporal).
template <int nwarps, int WAUX>
__launch_bounds__(nwarps*64, 1)
static __global__ void mul_mat_vec_q8_0_prefetch(
        const void * __restrict__ vx_ptr, const void * __restrict__ vy_ptr, float * __restrict__ dst_ptr,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(CDNA2)
    constexpr int warp_size    = 64;
    constexpr int qk           = QK8_0;                        // 32
    constexpr int qi           = QI8_0;                        // 8
    constexpr int vdr          = VDR_Q8_0_Q8_1_MMVQ;           // 2
    constexpr int nthreads     = nwarps*warp_size;             // 256 for nwarps=4
    constexpr int blk_per_iter = vdr*nthreads/qi;              // 64
    constexpr int tile_bytes   = blk_per_iter*(int)sizeof(block_q8_0);  // 64*34 = 2176
    constexpr int tile_dwords  = tile_bytes/4;                 // 544
    constexpr int n_passes     = (tile_dwords + nthreads - 1)/nthreads; // 3
    constexpr int grp          = qi/vdr;                       // 4

    const int tid  = warp_size*threadIdx.y + threadIdx.x;      // 0..nthreads-1
    const int row0 = blockIdx.x;                               // rows_per_cuda_block == 1

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    ggml_cuda_pdl_sync();

    const int blocks_per_row_x = ncols_x / qk;
    const int n_iter = (blocks_per_row_x + blk_per_iter - 1) / blk_per_iter;

    // weight row base; buffer resource bounded to this row's byte extent so a partial last tile
    // reads 0 for out-of-range dwords (they are never consumed — guarded below).
    const block_q8_0 * x_row = (const block_q8_0 *) vx_ptr
        + sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    // block_q8_0 is 34 B (not DWORD-aligned): round the buffer num_records UP to a DWORD so the
    // last block's final straddling DWORD is in-bounds (else HW zeroes it -> qs[30..31] corrupt).
    // The extra <=3 B are read into LDS but never consumed (guard: gblock < blocks_per_row_x).
    const uint32_t row_bytes = (uint32_t) blocks_per_row_x * (uint32_t) sizeof(block_q8_0);
    const q8pf_i32x4_t rsrc = q8pf_make_rsrc(x_row, (row_bytes + 3u) & ~3u);

    // activation vector base (block_q8_1); default-cached (temporal reuse across rows).
    const block_q8_1 * y = (const block_q8_1 *) vy_ptr + sample_y*stride_sample_y + channel_y*stride_channel_y;

    __shared__ uint32_t lds_w[2][tile_dwords];

    auto dma_tile = [&](int it, int buf) {
        __attribute__((address_space(3))) uint32_t * lds_base =
            (__attribute__((address_space(3))) uint32_t *) &lds_w[buf][0];
        const uint32_t byte_base = (uint32_t) it * (uint32_t) tile_bytes;
#pragma unroll
        for (int p = 0; p < n_passes; ++p) {
            const int d = p*nthreads + tid;
            if (d < tile_dwords) {
                q8pf_raw_buffer_load_lds(rsrc, lds_base + d, 4, (int)(byte_base + (uint32_t) d * 4u), 0, 0, WAUX);
            }
        }
    };

    dma_tile(0, 0); // prologue: issue tile 0

    const int g0  = tid / grp;              // local block index within tile
    const int kqs = vdr * (tid % grp);      // 0,2,4,6

    float acc = 0.0f;
    for (int it = 0; it < n_iter; ++it) {
        // drain this lane's outstanding vmem (current tile DMA + prev consume's y-loads),
        // then a scheduling barrier (MANDATORY — else the compiler hoists the LDS consumer
        // above the wait) and a block barrier so all lanes see the whole tile in LDS.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();

        // issue the next tile so its DMA overlaps the current tile's compute below.
        if (it + 1 < n_iter) {
            dma_tile(it + 1, (it + 1) & 1);
        }
        __builtin_amdgcn_sched_barrier(0);

        const int gblock = it*blk_per_iter + g0;
        if (gblock < blocks_per_row_x) {
            const block_q8_0 * wb =
                (const block_q8_0 *) ((const char *) &lds_w[it & 1][0] + g0*(int) sizeof(block_q8_0));
            int v[vdr];
            int u[vdr];
#pragma unroll
            for (int i = 0; i < vdr; ++i) {
                v[i] = get_int_b2(wb->qs, kqs + i);       // weights from LDS
                u[i] = get_int_b4(y[gblock].qs, kqs + i); // activation from global (cached)
            }
            acc += vec_dot_q8_0_q8_1_impl<float, vdr>(v, u, wb->d, __low2half(y[gblock].ds));
        }
    }

    // cross-warp reduction (rows_per_cuda_block == 1), identical structure to mul_mat_vec_q.
    __shared__ float tmp_shared[nwarps > 1 ? nwarps-1 : 1][warp_size];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y-1][threadIdx.x] = acc;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int l = 0; l < nwarps-1; ++l) {
        acc += tmp_shared[l][threadIdx.x];
    }
    acc = warp_reduce_sum<warp_size>(acc);

    if (threadIdx.x == 0) {
        dst_ptr[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0] = acc;
    }
#else
    GGML_UNUSED_VARS(vx_ptr, vy_ptr, dst_ptr, ncols_x, stride_row_x, stride_col_dst, channel_ratio,
        stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio,
        stride_sample_x, stride_sample_y, stride_sample_dst);
    NO_DEVICE_CODE;
#endif // __HIP_DEVICE_COMPILE__ && CDNA2
}
#endif // GGML_USE_HIP

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, int ncols_x_fixed = 0,
          bool gate_only_swiglu = false, bool bias_only = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), ncols_x_fixed == 1536)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    static_assert(!gate_only_swiglu || has_fusion, "gate-only SwiGLU requires fusion");
    static_assert(!bias_only || (has_fusion && type == GGML_TYPE_Q4_K), "bias-only fusion requires Q4_K");
    static_assert(!bias_only || !gate_only_swiglu, "bias-only fusion and gate-only SwiGLU are mutually exclusive");
    static_assert(ncols_x_fixed == 0 ||
        (ncols_x_fixed == 1536 && ncols_dst == 1 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K)),
        "fixed-width MMVQ specialization is only available for Q4_K/Q6_K batch-1 at 1536 columns");

    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, ncols_x_fixed == 1536);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
#if defined(CDNA2)
    constexpr bool halfwave_rows = type == GGML_TYPE_Q4_K && ncols_dst == 1 && small_k && nwarps >= 2;
#else
    constexpr bool halfwave_rows = false;
#endif
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps) *
        (halfwave_rows ? 2 : 1);
    constexpr int rows_per_thread = halfwave_rows ? 1 : rows_per_cuda_block;
    constexpr int reduction_width = halfwave_rows ? warp_size/2 : warp_size;
    constexpr int k_part_count = halfwave_rows ? 2 : 1;
#if defined(__gfx90a__)
    constexpr bool dpp_halfwave_reduce = halfwave_rows && ncols_x_fixed == 1536;
    constexpr bool dpp_q6_K_reduce =
        type == GGML_TYPE_Q6_K && ncols_dst == 1 && ncols_x_fixed == 1536 && rows_per_thread == 1;
#else
    constexpr bool dpp_halfwave_reduce = false;
    constexpr bool dpp_q6_K_reduce = false;
#endif
    constexpr int result_lane = dpp_halfwave_reduce ? reduction_width - 1 : dpp_q6_K_reduce ? warp_size - 1 : 0;

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int lane = halfwave_rows ? threadIdx.x % reduction_width : threadIdx.x;
    const     int k_tid_base = halfwave_rows ? lane : tid;
    const     int row = halfwave_rows ? 2*threadIdx.y + threadIdx.x/reduction_width : 0;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x_fixed == 0 ? ncols_x / qk : ncols_x_fixed / qk;
    constexpr int blocks_per_iter = vdr * (halfwave_rows ? 1 : nwarps)*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;

    if constexpr (bias_only) {
        x_bias = (const float *) fusion.x_bias;
    } else if constexpr (gate_only_swiglu) {
        vgate = fusion.gate;
    } else if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (bias_only) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (lane >= result_lane && lane < result_lane + rows_per_thread &&
            (halfwave_rows || threadIdx.y == 0) &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + row + lane - result_lane) < stride_col_dst)) {
            x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                x_biases[j] = x_bias[j * stride_col_dst + row + lane - result_lane];
            }
        }
    } else if constexpr (has_fusion && !gate_only_swiglu) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (lane >= result_lane && lane < result_lane + rows_per_thread &&
            (halfwave_rows || threadIdx.y == 0) &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + row + lane - result_lane) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + row + lane - result_lane];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + row + lane - result_lane];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_thread][k_part_count] = {{{0.0f}}};
    float tmp_gate[ncols_dst][rows_per_thread][k_part_count] = {{{0.0f}}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    constexpr bool converged_q4_K_dual =
        type == GGML_TYPE_Q4_K && ncols_dst == 1 && small_k && gate_only_swiglu;
    const bool row_in_bounds = !halfwave_rows || uint32_t(row0 + row) < stride_col_dst;
    if (row_in_bounds || converged_q4_K_dual) {
#pragma unroll
        for (int k_part = 0; k_part < k_part_count; ++k_part) {
            const int k_tid = k_tid_base + k_part*reduction_width;
            const int kbx0 = k_tid / (qi/vdr);
            if constexpr (ncols_x_fixed != 0) {
                constexpr int fixed_blocks_per_row_x = ncols_x_fixed / qk;
                constexpr int kbx_iters = (fixed_blocks_per_row_x + blocks_per_iter - 1) / blocks_per_iter;
#pragma unroll
                for (int kbx_iter = 0; kbx_iter < kbx_iters; ++kbx_iter) {
                    const int kbx = kbx0 + kbx_iter*blocks_per_iter;
                    if (kbx >= fixed_blocks_per_row_x) {
                        continue;
                    }
                    const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

                    // x block quant index when casting the quants to int
                    const int kqs = vdr * (k_tid % (qi/vdr));

#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                        for (int i = 0; i < rows_per_thread; ++i) {
                            const int row_i = converged_q4_K_dual && !row_in_bounds ? 0 : row + i;
                            const int kbx_row = kbx_offset + row_i*stride_row_x + kbx;
                            if constexpr (converged_q4_K_dual) {
                                const float2 dots = vec_dot_q4_K_q8_1_dual(
                                    vx, vgate, &y[j*stride_col_y + kby], kbx_row, kqs);
                                if (row_in_bounds) {
                                    tmp[j][i][k_part] += dots.x;
                                    tmp_gate[j][i][k_part] += dots.y;
                                }
                            } else {
                                tmp[j][i][k_part] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_row, kqs);
                                if constexpr (gate_only_swiglu) {
                                    tmp_gate[j][i][k_part] += vec_dot_q_cuda(vgate, &y[j*stride_col_y + kby], kbx_row, kqs);
                                } else if constexpr (has_fusion && !bias_only) {
                                    if (use_gate) {
                                        tmp_gate[j][i][k_part] += vec_dot_q_cuda(vgate, &y[j*stride_col_y + kby], kbx_row, kqs);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for (int kbx = kbx0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
                    const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

                    // x block quant index when casting the quants to int
                    const int kqs = vdr * (k_tid % (qi/vdr));

                    if constexpr (type == GGML_TYPE_Q4_K && ncols_dst == 4 && !has_fusion) {
#pragma unroll
                        for (int i = 0; i < rows_per_thread; ++i) {
                            const int kbx_row = kbx_offset + (row + i)*stride_row_x + kbx;
                            const q4_K_b4_weight weight = load_q4_K_b4_weight(
                                (const block_q4_K *) vx + kbx_row, kqs);
#pragma unroll
                            for (int j = 0; j < ncols_dst; ++j) {
                                tmp[j][i][k_part] += vec_dot_q4_K_q8_1_b4(
                                    weight, &y[j*stride_col_y + kby], kqs);
                            }
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                            for (int i = 0; i < rows_per_thread; ++i) {
                                const int row_i = converged_q4_K_dual && !row_in_bounds ? 0 : row + i;
                                const int kbx_row = kbx_offset + row_i*stride_row_x + kbx;
                                if constexpr (converged_q4_K_dual) {
                                    const float2 dots = vec_dot_q4_K_q8_1_dual(
                                        vx, vgate, &y[j*stride_col_y + kby], kbx_row, kqs);
                                    if (row_in_bounds) {
                                        tmp[j][i][k_part] += dots.x;
                                        tmp_gate[j][i][k_part] += dots.y;
                                    }
                                } else {
                                    tmp[j][i][k_part] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_row, kqs);
                                    if constexpr (gate_only_swiglu) {
                                        tmp_gate[j][i][k_part] += vec_dot_q_cuda(vgate, &y[j*stride_col_y + kby], kbx_row, kqs);
                                    } else if constexpr (has_fusion && !bias_only) {
                                        if (use_gate) {
                                            tmp_gate[j][i][k_part] += vec_dot_q_cuda(vgate, &y[j*stride_col_y + kby], kbx_row, kqs);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if constexpr (halfwave_rows) {
        tmp[0][0][0] += tmp[0][0][1];
        if constexpr (gate_only_swiglu) {
            tmp_gate[0][0][0] += tmp_gate[0][0][1];
        } else if constexpr (has_fusion && !bias_only) {
            if (use_gate) {
                tmp_gate[0][0][0] += tmp_gate[0][0][1];
            }
        }
    } else {
        __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_thread][warp_size];
        [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && !bias_only && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_thread][warp_size];

        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_thread; ++i) {
                    tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i][0];
                    if constexpr (gate_only_swiglu) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i][0];
                    } else if constexpr (has_fusion && !bias_only) {
                        if (use_gate) {
                            tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i][0];
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }

        // sum up partial sums
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_thread; ++i) {
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][i][0] += tmp_shared[l][j][i][threadIdx.x];
                    if constexpr (gate_only_swiglu) {
                        tmp_gate[j][i][0] += tmp_shared_gate[l][j][i][threadIdx.x];
                    } else if constexpr (has_fusion && !bias_only) {
                        if (use_gate) {
                            tmp_gate[j][i][0] += tmp_shared_gate[l][j][i][threadIdx.x];
                        }
                    }
                }
            }
        }
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // finish the per-row reduction and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_thread; ++i) {
#if defined(__gfx90a__)
            if constexpr (dpp_halfwave_reduce) {
                if constexpr (gate_only_swiglu) {
                    reduce_q4_K_halfwave_gfx90a(tmp[j][i][0], tmp_gate[j][i][0]);
                } else if constexpr (has_fusion && !bias_only) {
                    if (use_gate) {
                        reduce_q4_K_halfwave_gfx90a(tmp[j][i][0], tmp_gate[j][i][0]);
                    } else {
                        tmp[j][i][0] = reduce_q4_K_halfwave_gfx90a(tmp[j][i][0]);
                    }
                } else {
                    tmp[j][i][0] = reduce_q4_K_halfwave_gfx90a(tmp[j][i][0]);
                }
            } else if constexpr (dpp_q6_K_reduce) {
                if constexpr (gate_only_swiglu) {
                    reduce_q6_K_wave_gfx90a(tmp[j][i][0], tmp_gate[j][i][0]);
                } else if constexpr (has_fusion && !bias_only) {
                    if (use_gate) {
                        reduce_q6_K_wave_gfx90a(tmp[j][i][0], tmp_gate[j][i][0]);
                    } else {
                        tmp[j][i][0] = reduce_q6_K_wave_gfx90a(tmp[j][i][0]);
                    }
                } else {
                    tmp[j][i][0] = reduce_q6_K_wave_gfx90a(tmp[j][i][0]);
                }
            } else
#endif // defined(__gfx90a__)
            {
                tmp[j][i][0] = warp_reduce_sum<reduction_width>(tmp[j][i][0]);
                if constexpr (gate_only_swiglu) {
                    tmp_gate[j][i][0] = warp_reduce_sum<reduction_width>(tmp_gate[j][i][0]);
                } else if constexpr (has_fusion && !bias_only) {
                    if (use_gate) {
                        tmp_gate[j][i][0] = warp_reduce_sum<reduction_width>(tmp_gate[j][i][0]);
                    }
                }
            }

            if (lane == result_lane + (dpp_q6_K_reduce ? 0 : i) &&
                    (rows_per_cuda_block == 1 || uint32_t(row0 + row + i) < stride_col_dst)) {
                float result = tmp[j][i][0];
                if constexpr (bias_only) {
                    result += x_biases[j];
                } else if constexpr (gate_only_swiglu) {
                    result *= ggml_cuda_op_silu_single(tmp_gate[j][i][0]);
                } else if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i][0];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + row + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (gate_only_swiglu) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias);
    }
    if constexpr (bias_only) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, vgate, gate_bias, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false,
        const bool halfwave_rows = false, const bool fixed_1536_cdna2 = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, fixed_1536_cdna2);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps) * (halfwave_rows ? 2 : 1);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

#ifdef Q8_LDS_PREFETCH_COMPILED
// GGML_CUDA_Q8_PREFETCH: 0=off (default), 1=on default-cached weight DMA, 2=on SLC/streaming.
static int q8pf_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_Q8_PREFETCH");
        return e ? atoi(e) : 0;
    }();
    return mode;
}

static void mul_mat_vec_q8_0_prefetch_launch(
        const void * vx, const void * vy, float * dst,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, cudaStream_t stream, int mode) {
    // nwarps == 4 matches calc_nwarps(Q8_0, ncols_dst=1, GCN table) for CDNA2.
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    if (mode == 2) {
        ggml_cuda_kernel_launch(mul_mat_vec_q8_0_prefetch<4, 2>, lp,
            vx, vy, dst, ncols_x, stride_row_x, stride_col_dst, channel_ratio, stride_channel_x,
            stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q8_0_prefetch<4, 0>, lp,
            vx, vy, dst, ncols_x, stride_row_x, stride_col_dst, channel_ratio, stride_channel_x,
            stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    }
}
#endif // Q8_LDS_PREFETCH_COMPILED

template<ggml_type type, int c_ncols_dst, bool small_k = false, int ncols_x_fixed = 0>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            if constexpr (type == GGML_TYPE_Q4_K && c_ncols_dst == 1) {
                const bool use_bias_only = fusion.gate == nullptr && fusion.x_bias != nullptr &&
                    fusion.gate_bias == nullptr && fusion.x_scale == nullptr && fusion.gate_scale == nullptr;
                if (use_bias_only) {
                    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, ncols_x_fixed, false, true>, launch_params,
                         vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                         channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                         sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
                    return;
                }
                const bool use_gate_only_swiglu = fusion.gate != nullptr && fusion.x_bias == nullptr &&
                    fusion.gate_bias == nullptr && fusion.x_scale == nullptr && fusion.gate_scale == nullptr &&
                    fusion.glu_op == GGML_GLU_OP_SWIGLU;
                if (use_gate_only_swiglu) {
                    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, ncols_x_fixed, true>, launch_params,
                         vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                         channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                         sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
                    return;
                }
            }
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, ncols_x_fixed>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, ncols_x_fixed>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block>, launch_params,
        vx, vy, ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

#ifdef Q8_LDS_PREFETCH_COMPILED
            // Experimental CDNA2 async weight-prefetch specialization (dense Q8_0 batch-1 GEMV).
            if constexpr (type == GGML_TYPE_Q8_0) {
                if (!has_ids && !has_fusion && GGML_CUDA_CC_IS_CDNA(cc) && warp_size == 64 && q8pf_mode() != 0) {
                    std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                            nsamples_dst, warp_size, table_id);
                    mul_mat_vec_q8_0_prefetch_launch(
                        vx, vy, dst, ncols_x, stride_row_x, stride_col_dst, channel_ratio_fd, stride_channel_x,
                        stride_channel_y, stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y,
                        stride_sample_dst, dims.first, dims.second, stream, q8pf_mode());
                    return;
                }
            }
#endif // Q8_LDS_PREFETCH_COMPILED

            bool use_small_k = should_use_small_k(c_ncols_dst);

#ifdef GGML_USE_HIP
            if constexpr (type == GGML_TYPE_Q4_K) {
                if (ncols_x == 1536 && cc == GGML_CUDA_CC_CDNA2) {
                    constexpr bool fixed_small_k = true;
                    constexpr bool fixed_1536_cdna2 = true;
                    std::pair<dim3, dim3> dims = calc_launch_params<type>(
                        c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, fixed_small_k,
                        fixed_small_k && calc_nwarps(type, c_ncols_dst, table_id, fixed_1536_cdna2) >= 2,
                        fixed_1536_cdna2);
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, fixed_small_k, 1536>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                        stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                        stream);
                    return;
                }
            }
            if constexpr (type == GGML_TYPE_Q6_K) {
                if (!has_ids && ncols_x == 1536 && cc == GGML_CUDA_CC_CDNA2) {
                    constexpr bool fixed_small_k = false;
                    constexpr bool fixed_1536_cdna2 = true;
                    std::pair<dim3, dim3> dims = calc_launch_params<type>(
                        c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, fixed_small_k,
                        false, fixed_1536_cdna2);
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, fixed_small_k, 1536>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                        stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                        stream);
                    return;
                }
            }
#endif // GGML_USE_HIP

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                    nsamples_dst, warp_size, table_id, true,
                    type == GGML_TYPE_Q4_K && c_ncols_dst == 1 &&
                        calc_nwarps(type, c_ncols_dst, table_id) == 2 && cc == GGML_CUDA_CC_CDNA2);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
            const int rpb = calc_rows_per_block(type, c_ncols_dst, table_id, false, nwarps);
            if (rpb == 4 && nrows_x % rpb != 0) {
                if constexpr (type == GGML_TYPE_Q8_0) {
                    // akm-cdna2-q8-b4-y-stream-amortize: R=4 row tiling needs nrows_x % 4 == 0;
                    // launch the small_k instantiation (rows_per_cuda_block=2) instead, so a
                    // non-multiple row count never reads past the last row of x.
                    constexpr bool r2_fallback = true;
                    std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, r2_fallback);
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, r2_fallback>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                         channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                         sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                         dims.first, dims.second, 0, ids_stride, stream);
                    break;
                }
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#if defined(GGML_USE_HIP) && defined(GGML_HIP_GRAPHS)
struct mmvq_q8_1_graph_cache_entry {
    const ggml_tensor * src1 = nullptr;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {};
    size_t nb[GGML_MAX_DIMS] = {};
    int64_t ne10_padded = 0;
    size_t q8_1_size = 0;
    std::unique_ptr<ggml_cuda_pool_alloc<char>> q8_1;
};

struct mmvq_q8_1_graph_cache {
    ggml_backend_cuda_context * ctx = nullptr;
    ggml_cuda_pool * pool = nullptr;
    unsigned long long capture_id = 0;
    std::vector<mmvq_q8_1_graph_cache_entry> entries;

    void clear() {
        while (!entries.empty()) {
            entries.pop_back();
        }
        ctx = nullptr;
        pool = nullptr;
        capture_id = 0;
    }
};

struct mmvq_q8_1_graph_cache_registry {
    std::mutex mutex;
    std::unordered_map<ggml_backend_cuda_context *, std::unique_ptr<mmvq_q8_1_graph_cache>> caches;
};

static mmvq_q8_1_graph_cache_registry q8_1_graph_cache_registry;

void ggml_cuda_mmvq_q8_1_graph_cache_clear(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(q8_1_graph_cache_registry.mutex);
    const auto it = q8_1_graph_cache_registry.caches.find(ctx);
    if (it != q8_1_graph_cache_registry.caches.end()) {
        it->second->clear();
        q8_1_graph_cache_registry.caches.erase(it);
    }
}

static bool mmvq_q8_1_graph_cache_matches(
        const mmvq_q8_1_graph_cache_entry & entry, const ggml_tensor * src1,
        const int64_t ne10_padded, const size_t q8_1_size) {
    if (entry.src1 != src1 || entry.type != src1->type ||
        entry.ne10_padded != ne10_padded || entry.q8_1_size != q8_1_size) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (entry.ne[i] != src1->ne[i] || entry.nb[i] != src1->nb[i]) {
            return false;
        }
    }
    return true;
}
#endif

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_alloc(ctx.pool());
    void * src1_q8_1 = nullptr;

    bool reuse_src1_q8_1 = false;
#if defined(GGML_USE_HIP) && defined(GGML_HIP_GRAPHS)
    const bool cache_eligible = ctx.curr_stream_no == 0 && ids == nullptr &&
        (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) && ne11 == 1 && ne12 == 1 && ne13 == 1;
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    if (cache_eligible) {
        CUDA_CHECK(hipStreamGetCaptureInfo_v2(
            stream, &capture_status, &capture_id, nullptr, nullptr, nullptr));
    }
    const bool capture_active = capture_status == hipStreamCaptureStatusActive;
    if (cache_eligible && capture_active) {
        std::lock_guard<std::mutex> lock(q8_1_graph_cache_registry.mutex);
        std::unique_ptr<mmvq_q8_1_graph_cache> & cache_ptr = q8_1_graph_cache_registry.caches[&ctx];
        if (cache_ptr == nullptr) {
            cache_ptr = std::make_unique<mmvq_q8_1_graph_cache>();
        }
        mmvq_q8_1_graph_cache & cache = *cache_ptr;
        ggml_cuda_pool * const pool = &ctx.pool();
        if (cache.ctx != &ctx || cache.pool != pool) {
            cache.clear();
            cache.ctx = &ctx;
            cache.pool = pool;
            cache.capture_id = capture_id;
        } else if (cache.capture_id != capture_id) {
            cache.clear();
            cache.ctx = &ctx;
            cache.pool = pool;
            cache.capture_id = capture_id;
        }

        for (const mmvq_q8_1_graph_cache_entry & entry : cache.entries) {
            if (mmvq_q8_1_graph_cache_matches(entry, src1, ne10_padded, src1_q8_1_size)) {
                src1_q8_1 = entry.q8_1->get();
                reuse_src1_q8_1 = true;
                break;
            }
        }

        if (!reuse_src1_q8_1) {
            cache.entries.emplace_back();
            mmvq_q8_1_graph_cache_entry & entry = cache.entries.back();
            entry.src1 = src1;
            entry.type = src1->type;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                entry.ne[i] = src1->ne[i];
                entry.nb[i] = src1->nb[i];
            }
            entry.ne10_padded = ne10_padded;
            entry.q8_1_size = src1_q8_1_size;
            entry.q8_1 = std::make_unique<ggml_cuda_pool_alloc<char>>(*pool, src1_q8_1_size);
            src1_q8_1 = entry.q8_1->get();
        }
    }
#endif

    if (src1_q8_1 == nullptr) {
        src1_q8_1 = src1_q8_1_alloc.alloc(src1_q8_1_size);
    }

    if (!reuse_src1_q8_1) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
