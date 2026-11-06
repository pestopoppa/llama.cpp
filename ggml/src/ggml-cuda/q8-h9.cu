#include "q8-h9.cuh"
#include "quantize.cuh"

#include <atomic>
#include <cstdlib>

// ---------------------------------------------------------------------------------------------------------------------
// env switches + witness
// ---------------------------------------------------------------------------------------------------------------------

static bool q8h9_enabled() {
    static const bool v = [] {
        const char * s = getenv("GGML_HIP_Q8_H9");
        return !(s && s[0] == '0' && s[1] == '\0');
    }();
    return v;
}

static int64_t q8h9_min_n() {
    static const int64_t v = [] {
        const char * s = getenv("GGML_HIP_Q8_H9_MIN_N");
        return s && s[0] ? (int64_t) atoll(s) : (int64_t) 129;
    }();
    return v;
}

static std::atomic<int> q8h9_n_calls{0};

// ---------------------------------------------------------------------------------------------------------------------
// kernel: dst[n][m] = sum_kb d_w[m][kb]*d_a[n][kb] * sum_{k in kb} W[m][k]*X[n][k]
//   W: Q8_0 rows (34-byte blocks, row_bytes_x apart). X: plain block_q8_1 rows (36-byte blocks, row_bytes_y apart).
//   One stage = 64 k = 2 blocks: a Q8_0 pair is 17 words (qs of the even block at byte 2, i.e. misaligned by 2;
//   qs of the odd block at word 9), a q8_1 pair is 18 words (ds at words 0 and 9, qs at 1..8 and 10..17).
// ---------------------------------------------------------------------------------------------------------------------

#define Q8H9_BM 128
#define Q8H9_BN 128
#define Q8H9_NT 256
#define Q8H9_RW 20  // LDS words per tile row: 2 blocks x 8 qs words + 4 pad (16-byte aligned rows, staggered banks)

#if defined(__gfx90a__)
#define Q8H9_DEVICE 1
typedef int32_t q8h9_i32x16 __attribute__((ext_vector_type(16)));
typedef int32_t q8h9_i32x4  __attribute__((ext_vector_type(4)));

extern "C" __device__ int32_t q8h9_buffer_load_i32(q8h9_i32x4 rsrc, int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

static __device__ __forceinline__ q8h9_i32x4 q8h9_make_rsrc(const void * base, const uint32_t num_bytes) {
    const uint64_t a = (uint64_t) base;
    q8h9_i32x4 r;
    r.x = __builtin_amdgcn_readfirstlane((int32_t) (uint32_t) a);
    r.y = __builtin_amdgcn_readfirstlane((int32_t) (uint32_t) (a >> 32)); // stride 0
    r.z = __builtin_amdgcn_readfirstlane((int32_t) num_bytes);            // out-of-range reads return 0
    r.w = 0x00020000;                                                     // gfx9 raw buffer, 32-bit data format
    return r;
}
#endif

__launch_bounds__(Q8H9_NT, 1)
static __global__ void mul_mat_q8_0_q8_1_h9(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst,
        const int M, const int N, const int nstage,
        const int64_t row_bytes_x, const int64_t row_bytes_y, const int64_t stride_dst) {
#if defined(Q8H9_DEVICE)
    // quants: [buf][0 = weights, 1 = activations][row*RW + word], double buffered
    __shared__ __attribute__((aligned(16))) int32_t s_q[2][2][Q8H9_BM*Q8H9_RW];
    // scales: [slot][role][block of stage][row], 3-slot ring: block 2s+1 is rescaled during stage s+1, so its scales
    // must survive the refill at the end of stage s (which writes slot (s+1)%3).
    __shared__ __attribute__((aligned(16))) float   s_d[3][2][2][Q8H9_BM];

    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid % 64;
    const int l32  = lane % 32;
    const int h    = lane / 32;

    const int n0 = blockIdx.x * Q8H9_BN;
    const int m0 = blockIdx.y * Q8H9_BM;
    const int wm = (wave % 2) * 64;
    const int wn = (wave / 2) * 64;

    // Cooperative coalesced loader: 16 consecutive threads cover one row's 16 qs words of a stage, rows lr0 + 16*i.
    const int lj  = tid % 16;
    const int lr0 = tid / 16;
    // scale loader: one (row, block) per thread and role
    const int sr  = tid / 2;
    const int sb  = tid % 2;

    // Buffer loads (AITER style): SGPR resource + 32-bit VGPR row offset + SGPR stage offset, no 64-bit address math
    // per load. Host guarantees both operands are < 2 GiB. Rows past M/N are clamped to the last row.
    const q8h9_i32x4 rsrc_x = q8h9_make_rsrc(x, (uint32_t) ((int64_t) M*row_bytes_x));
    const q8h9_i32x4 rsrc_y = q8h9_make_rsrc(y, (uint32_t) ((int64_t) N*row_bytes_y));

    // word offsets inside a stage
    const int wb  = lj < 8 ? lj + 1 : lj + 2; // q8_1 qs word
    const int wsa = sb ? 8 : 0;               // Q8_0 d of block 1 is the high half of word 8, of block 0 the low half of word 0
    const int wsb = sb ? 9 : 0;               // q8_1 ds of block sb, d in the low half

    int voa[8]; // byte offset of word lj of the Q8_0 pair (even block: words lj, lj+1 -> alignbyte 2; odd: word lj+1)
    int vob[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        voa[i] = min(m0 + lr0 + 16*i, M - 1)*(int) row_bytes_x + 4*lj;
        vob[i] = min(n0 + lr0 + 16*i, N - 1)*(int) row_bytes_y + 4*wb;
    }
    const int vsa = min(m0 + sr, M - 1)*(int) row_bytes_x + 4*wsa;
    const int vsb = min(n0 + sr, N - 1)*(int) row_bytes_y + 4*wsb;

    int32_t ra_lo[8], ra_hi[8], rb_q[8];
    int32_t rsa, rsb;

    auto load_stage = [&](const int s) {
        const int oa = __builtin_amdgcn_readfirstlane(s*68);
        const int ob = __builtin_amdgcn_readfirstlane(s*72);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ra_lo[i] = q8h9_buffer_load_i32(rsrc_x, voa[i],     oa, 0);
            ra_hi[i] = q8h9_buffer_load_i32(rsrc_x, voa[i] + 4, oa, 0);
            rb_q[i]  = q8h9_buffer_load_i32(rsrc_y, vob[i],     ob, 0);
        }
        rsa = q8h9_buffer_load_i32(rsrc_x, vsa, oa, 0);
        rsb = q8h9_buffer_load_i32(rsrc_y, vsb, ob, 0);
    };

    auto store_stage = [&](const int buf, const int slot) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = lr0 + 16*i;
            const int32_t qa = lj < 8 ? (int32_t) __builtin_amdgcn_alignbyte((uint32_t) ra_hi[i], (uint32_t) ra_lo[i], 2)
                                      : ra_hi[i];
            s_q[buf][0][row*Q8H9_RW + lj] = qa;
            s_q[buf][1][row*Q8H9_RW + lj] = rb_q[i];
        }
        const uint32_t da = sb ? ((uint32_t) rsa >> 16) : ((uint32_t) rsa & 0xFFFFu);
        const uint32_t db = (uint32_t) rsb & 0xFFFFu;
        s_d[slot][0][sb][sr] = __half2float(__ushort_as_half((unsigned short) da));
        s_d[slot][1][sb][sr] = __half2float(__ushort_as_half((unsigned short) db));
    };

    float acc[2][2][16];
#pragma unroll
    for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                acc[tm][tn][r] = 0.0f;
            }
        }
    }

    // int32 partials of one 32-k block for the 4 32x32 tiles of this wave
    auto mfma_block = [&](q8h9_i32x16 (&c)[2][2], const int buf, const int qb) {
        q8h9_i32x4 a[2];
        q8h9_i32x4 b[2];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            // lane half h supplies word 4h+j to MFMA j for both operands: MFMA j sums words {j, 4+j}, the 4 MFMAs
            // cover the 8 words of the block exactly once.
            a[t] = *(const q8h9_i32x4 *) &s_q[buf][0][(wm + 32*t + l32)*Q8H9_RW + 8*qb + 4*h];
            b[t] = *(const q8h9_i32x4 *) &s_q[buf][1][(wn + 32*t + l32)*Q8H9_RW + 8*qb + 4*h];
        }
#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
            for (int tn = 0; tn < 2; ++tn) {
                q8h9_i32x16 cc = {0};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    cc = __builtin_amdgcn_mfma_i32_32x32x8i8(a[tm][j], b[tn][j], cc, 0, 0, 0);
                }
                c[tm][tn] = cc;
            }
        }
    };

    // acc += d_w*d_a * (float) partial; output register 4g+e of a 32x32x8 tile is row 8g+4h+e, column l32
    auto flush = [&](const q8h9_i32x16 (&c)[2][2], const int slot, const int qb) {
        float db[2];
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
            db[tn] = s_d[slot][1][qb][wn + 32*tn + l32];
        }
#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float4 da = *(const float4 *) &s_d[slot][0][qb][wm + 32*tm + 8*g + 4*h];
#pragma unroll
                for (int tn = 0; tn < 2; ++tn) {
                    acc[tm][tn][4*g + 0] = fmaf((float) c[tm][tn][4*g + 0]*db[tn], da.x, acc[tm][tn][4*g + 0]);
                    acc[tm][tn][4*g + 1] = fmaf((float) c[tm][tn][4*g + 1]*db[tn], da.y, acc[tm][tn][4*g + 1]);
                    acc[tm][tn][4*g + 2] = fmaf((float) c[tm][tn][4*g + 2]*db[tn], da.z, acc[tm][tn][4*g + 2]);
                    acc[tm][tn][4*g + 3] = fmaf((float) c[tm][tn][4*g + 3]*db[tn], da.w, acc[tm][tn][4*g + 3]);
                }
            }
        }
    };

    q8h9_i32x16 cA[2][2]; // even block of the stage
    q8h9_i32x16 cB[2][2]; // odd block, rescaled one half-stage later (during the next stage's even-block MFMAs)
#pragma unroll
    for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
            cB[tm][tn] = q8h9_i32x16{0};
        }
    }
    // Stage 0 flushes a zero "block -1" from slot 2: zero its scales so the loop body has no branch (a branch splits
    // the body into basic blocks and the IR passes then sink the whole rescale below the MFMA bursts).
    s_d[2][sb][0][sr] = 0.0f;
    s_d[2][sb][1][sr] = 0.0f;

    load_stage(0);
    store_stage(0, 0);
    load_stage(min(1, nstage - 1));
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf   = s & 1;
        const int slot  = s % 3;
        const int pslot = (s + 2) % 3; // slot of stage s-1

        // half 1: MFMAs of block 2s while the VALU rescales block 2s-1
        mfma_block(cA, buf, 0);
        flush(cB, pslot, 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);  // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);  // MFMA
            __builtin_amdgcn_sched_group_barrier(0x002, 12, 0); // VALU
        }
        __builtin_amdgcn_sched_barrier(0);

        // half 2: MFMAs of block 2s+1 while the VALU rescales block 2s
        mfma_block(cB, buf, 1);
        flush(cA, slot, 0);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
            __builtin_amdgcn_sched_group_barrier(0x002, 12, 0);
        }
        __builtin_amdgcn_sched_barrier(0);

        // refill the other buffer from the registers loaded one stage ago, then issue the next prefetch; its latency
        // hides under the next stage's MFMAs.
        // (last stage: re-stores stage nstage-1 into the idle buffer and slot nstage%3, which nothing reads again)
        store_stage(buf ^ 1, (s + 1) % 3);
        load_stage(min(s + 2, nstage - 1));
        __builtin_amdgcn_sched_barrier(0);

        __syncthreads();
    }
    flush(cB, (nstage - 1) % 3, 1);

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
#endif // defined(Q8H9_DEVICE)
}

// ---------------------------------------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_q8_h9_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!q8h9_enabled() || cc != GGML_CUDA_CC_CDNA2) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (N < q8h9_min_n() || K % 64 != 0 || M > INT32_MAX || N > INT32_MAX) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[1] != (size_t) (K/QK8_0)*sizeof(block_q8_0) || ((uintptr_t) src0->data) % 4 != 0) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || dst->ne[0] != M || dst->ne[1] != N) {
        return false;
    }
    // 32-bit buffer offsets: weights and the q8_1 activation copy must each be < 2 GiB
    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    if ((int64_t) ggml_nbytes(src0) >= INT32_MAX || N*(ne10_padded/QK8_1)*(int64_t) sizeof(block_q8_1) >= INT32_MAX) {
        return false;
    }
    return true;
}

bool ggml_cuda_q8_h9_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_q8_h9_eligible(src0, src1, dst, cc)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> q8_1(ctx.pool(), N*ne10_padded*sizeof(block_q8_1)/QK8_1);
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8_1.get(), src0->type, K, s11, s12, s13, ne10_padded, N, 1, 1, stream);

    const dim3 grid((N + Q8H9_BN - 1)/Q8H9_BN, (M + Q8H9_BM - 1)/Q8H9_BM, 1);
    mul_mat_q8_0_q8_1_h9<<<grid, Q8H9_NT, 0, stream>>>(
        (const char *) src0->data, q8_1.get(), (float *) dst->data,
        (int) M, (int) N, (int) (K/64),
        (int64_t) src0->nb[1], (int64_t) (ne10_padded/QK8_1*sizeof(block_q8_1)), (int64_t) (dst->nb[1]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());

    if (q8h9_n_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
        GGML_LOG_INFO("ggml_cuda_q8_h9: gfx90a AK-ASM-H9 per-32-scale int8 MFMA GEMM ran (call 1)\n");
    }
    return true;
}
