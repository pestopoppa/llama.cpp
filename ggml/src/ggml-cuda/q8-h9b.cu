#include "q8-h9b.cuh"

#include <atomic>
#include <cstdlib>

// AK-ASM-H9B: one fused kernel for dense Q8_0 x f32 at large batch on gfx90a. Weights are dequantized to f16 exactly as
// C6's dequantize_block<q8_0> does ((half) ((float) q * (float) d)), activations rounded f32 -> f16 exactly as C6's
// to_fp16 does, both while staging into LDS; v_mfma_f32_32x32x8f16 with f32 accumulation. Replaces C6's two converts +
// rocBLAS with no per-block rescale (the H9 int8 path's VALU cost).

static bool q8h9b_enabled() {
    static const bool v = [] {
        const char * s = getenv("GGML_HIP_Q8_H9B");
        return !(s && s[0] == '0' && s[1] == '\0');
    }();
    return v;
}

static int64_t q8h9b_min_n() {
    static const int64_t v = [] {
        const char * s = getenv("GGML_HIP_Q8_H9B_MIN_N");
        return s && s[0] ? (int64_t) atoll(s) : (int64_t) 129;
    }();
    return v;
}

static std::atomic<int> q8h9b_n_calls{0};

#define Q8H9B_BM 128
#define Q8H9B_BN 128
#define Q8H9B_NT 256
#define Q8H9B_RW 20 // LDS words per tile row: 32 halves (one 32-k stage) = 16 words + 4 pad

#if defined(__gfx90a__)
#define Q8H9B_DEVICE 1
typedef int32_t  q8h9b_i32x4  __attribute__((ext_vector_type(4)));
typedef float    q8h9b_f32x4  __attribute__((ext_vector_type(4)));
typedef float    q8h9b_f32x16 __attribute__((ext_vector_type(16)));
typedef _Float16 q8h9b_f16x4  __attribute__((ext_vector_type(4)));

extern "C" __device__ int32_t q8h9b_buffer_load_i32(q8h9b_i32x4 rsrc, int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.i32");
extern "C" __device__ q8h9b_f32x4 q8h9b_buffer_load_f32x4(q8h9b_i32x4 rsrc, int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.v4f32");

static __device__ __forceinline__ q8h9b_i32x4 q8h9b_make_rsrc(const void * base, const uint32_t num_bytes) {
    const uint64_t a = (uint64_t) base;
    q8h9b_i32x4 r;
    r.x = __builtin_amdgcn_readfirstlane((int32_t) (uint32_t) a);
    r.y = __builtin_amdgcn_readfirstlane((int32_t) (uint32_t) (a >> 32));
    r.z = __builtin_amdgcn_readfirstlane((int32_t) num_bytes);
    r.w = 0x00020000;
    return r;
}

static __device__ __forceinline__ uint32_t q8h9b_pack(const float a, const float b) {
    const _Float16 ha = (_Float16) a;
    const _Float16 hb = (_Float16) b;
    return (uint32_t) __builtin_bit_cast(uint16_t, ha) | ((uint32_t) __builtin_bit_cast(uint16_t, hb) << 16);
}
#endif

// dst[n][m] = sum_k half(d_w*q_w)[m][k] * half(X[n][k]); W Q8_0 rows (row_bytes_x apart), X f32 rows (row_bytes_y apart).
__launch_bounds__(Q8H9B_NT, 1)
static __global__ void mul_mat_q8_0_f16_h9b(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst,
        const int M, const int N, const int nstage,
        const int row_bytes_x, const int row_bytes_y, const int64_t stride_dst) {
#if defined(Q8H9B_DEVICE)
    __shared__ __attribute__((aligned(16))) uint32_t s_a[2][Q8H9B_BM*Q8H9B_RW];
    __shared__ __attribute__((aligned(16))) uint32_t s_b[2][Q8H9B_BN*Q8H9B_RW];

    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid % 64;
    const int l32  = lane % 32;
    const int h    = lane / 32;

    const int n0 = blockIdx.x * Q8H9B_BN;
    const int m0 = blockIdx.y * Q8H9B_BM;
    const int wm = (wave % 2) * 64;
    const int wn = (wave / 2) * 64;

    const q8h9b_i32x4 rsrc_x = q8h9b_make_rsrc(x, (uint32_t) ((int64_t) M*row_bytes_x));
    const q8h9b_i32x4 rsrc_y = q8h9b_make_rsrc(y, (uint32_t) ((int64_t) (N - 1)*row_bytes_y + 4*32*nstage));

    // loader: 8 consecutive threads per row, rows lr0 + 32*i. A: qs word lj of the stage's block; B: floats 4lj..4lj+3.
    const int lj  = tid % 8;
    const int lr0 = tid / 8;
    int voa[4];
    int vob[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        voa[i] = min(m0 + lr0 + 32*i, M - 1)*row_bytes_x;
        vob[i] = min(n0 + lr0 + 32*i, N - 1)*row_bytes_y + 16*lj;
    }

    int32_t ra_lo[4], ra_hi[4], ra_d[4];
    q8h9b_f32x4 rb[4];
    int sh_q = 0; // byte misalignment of the qs of the staged block (0 or 2, uniform)
    int sh_d = 0; // 1: d is the high half of its word

    auto load_stage = [&](const int s) {
        // block s of each row starts at byte 34*s (row_bytes_x is a multiple of 68, so 4-aligned rows)
        const int qb = 34*s + 2 + 4*lj;
        const int oq = __builtin_amdgcn_readfirstlane((34*s + 2) & ~3);
        const int od = __builtin_amdgcn_readfirstlane((34*s) & ~3);
        const int ob = __builtin_amdgcn_readfirstlane(128*s);
        sh_q = (34*s + 2) & 3;
        sh_d = ((34*s) & 3) != 0;
        GGML_UNUSED(qb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ra_lo[i] = q8h9b_buffer_load_i32(rsrc_x, voa[i] + 4*lj,     oq, 0);
            ra_hi[i] = q8h9b_buffer_load_i32(rsrc_x, voa[i] + 4*lj + 4, oq, 0);
            ra_d[i]  = q8h9b_buffer_load_i32(rsrc_x, voa[i],            od, 0);
            rb[i]    = q8h9b_buffer_load_f32x4(rsrc_y, vob[i], ob, 0);
        }
    };

    auto store_stage = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = lr0 + 32*i;
            const uint32_t q  = __builtin_amdgcn_alignbyte((uint32_t) ra_hi[i], (uint32_t) ra_lo[i], sh_q);
            const uint32_t db = sh_d ? ((uint32_t) ra_d[i] >> 16) : ((uint32_t) ra_d[i] & 0xFFFFu);
            const float d = (float) __builtin_bit_cast(_Float16, (uint16_t) db);
            const float q0 = (float) (int8_t) (q >>  0);
            const float q1 = (float) (int8_t) (q >>  8);
            const float q2 = (float) (int8_t) (q >> 16);
            const float q3 = (float) (int8_t) (q >> 24);
            *(uint2 *) &s_a[buf][row*Q8H9B_RW + 2*lj] = make_uint2(q8h9b_pack(q0*d, q1*d), q8h9b_pack(q2*d, q3*d));
            *(uint2 *) &s_b[buf][row*Q8H9B_RW + 2*lj] = make_uint2(q8h9b_pack(rb[i].x, rb[i].y), q8h9b_pack(rb[i].z, rb[i].w));
        }
    };

    q8h9b_f32x16 acc[2][2];
#pragma unroll
    for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
            acc[tm][tn] = q8h9b_f32x16{0};
        }
    }

    load_stage(0);
    store_stage(0);
    load_stage(min(1, nstage - 1));
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;

        // lane half h holds words 8h..8h+7 of the row; MFMA j uses words 8h+2j, 8h+2j+1 (4 halves) for both operands,
        // so the 4 MFMAs x 2 halves cover the 32 k of the stage exactly once.
        q8h9b_i32x4 a[2][2];
        q8h9b_i32x4 b[2][2];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const uint32_t * pa = &s_a[buf][(wm + 32*t + l32)*Q8H9B_RW + 8*h];
            const uint32_t * pb = &s_b[buf][(wn + 32*t + l32)*Q8H9B_RW + 8*h];
            a[t][0] = *(const q8h9b_i32x4 *) (pa + 0);
            a[t][1] = *(const q8h9b_i32x4 *) (pa + 4);
            b[t][0] = *(const q8h9b_i32x4 *) (pb + 0);
            b[t][1] = *(const q8h9b_i32x4 *) (pb + 4);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
                for (int tn = 0; tn < 2; ++tn) {
                    const q8h9b_i32x4 & av = a[tm][j/2];
                    const q8h9b_i32x4 & bv = b[tn][j/2];
                    const int w = 2*(j%2);
                    const uint64_t a64 = (uint64_t) (uint32_t) av[w] | ((uint64_t) (uint32_t) av[w + 1] << 32);
                    const uint64_t b64 = (uint64_t) (uint32_t) bv[w] | ((uint64_t) (uint32_t) bv[w + 1] << 32);
                    acc[tm][tn] = __builtin_amdgcn_mfma_f32_32x32x8f16(
                        __builtin_bit_cast(q8h9b_f16x4, a64), __builtin_bit_cast(q8h9b_f16x4, b64), acc[tm][tn], 0, 0, 0);
                }
            }
        }
        // The dequant/pack of the next stage (registers loaded one stage ago) writes the idle buffer, so it is independent
        // of these MFMAs: interleave it between them (AITER style) instead of issuing it after the MFMA burst.
        store_stage(buf ^ 1);               // last stage: re-stores stage nstage-1 into the idle buffer (never read)
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA
            __builtin_amdgcn_sched_group_barrier(0x002, 10, 0); // VALU (dequant, f16 pack)
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);  // DS write
        }
        __builtin_amdgcn_sched_barrier(0);
        load_stage(min(s + 2, nstage - 1)); // next prefetch, in flight across the next stage's MFMAs
        __builtin_amdgcn_sched_barrier(0);

        __syncthreads();
    }

#pragma unroll
    for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
            const int n = n0 + wn + 32*tn + l32;
            if (n >= N) {
                continue;
            }
            float * dn = dst + (int64_t) n*stride_dst;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int m = m0 + wm + 32*tm + 8*(r/4) + 4*h + (r%4);
                if (m < M) {
                    dn[m] = acc[tm][tn][r];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, M, N, nstage, row_bytes_x, row_bytes_y, stride_dst);
    NO_DEVICE_CODE;
#endif // defined(Q8H9B_DEVICE)
}

bool ggml_cuda_q8_h9b_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!q8h9b_enabled() || cc != GGML_CUDA_CC_CDNA2) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (N < q8h9b_min_n() || K % 64 != 0 || M > INT32_MAX || N > INT32_MAX) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[1] != (size_t) (K/QK8_0)*sizeof(block_q8_0) || ((uintptr_t) src0->data) % 4 != 0) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || ((uintptr_t) src1->data) % 16 != 0) {
        return false;
    }
    if (dst->nb[0] != sizeof(float) || dst->ne[0] != M || dst->ne[1] != N) {
        return false;
    }
    // 32-bit buffer offsets
    if ((int64_t) ggml_nbytes(src0) >= INT32_MAX || (int64_t) src1->nb[1]*N >= INT32_MAX) {
        return false;
    }
    return true;
}

bool ggml_cuda_q8_h9b_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_q8_h9b_eligible(src0, src1, dst, cc)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const dim3 grid((N + Q8H9B_BN - 1)/Q8H9B_BN, (M + Q8H9B_BM - 1)/Q8H9B_BM, 1);
    mul_mat_q8_0_f16_h9b<<<grid, Q8H9B_NT, 0, ctx.stream()>>>(
        (const char *) src0->data, (const char *) src1->data, (float *) dst->data,
        (int) M, (int) N, (int) (K/32), (int) src0->nb[1], (int) src1->nb[1], (int64_t) (dst->nb[1]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    if (q8h9b_n_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
        GGML_LOG_INFO("ggml_cuda_q8_h9b: gfx90a AK-ASM-H9B fused Q8_0-dequant f16 MFMA GEMM ran (call 1)\n");
    }
    return true;
}
