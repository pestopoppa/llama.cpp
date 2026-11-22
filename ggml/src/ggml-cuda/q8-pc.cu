#include "q8-pc.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

// ---------------------------------------------------------------------------------------------------------------------
// knobs + witness
// ---------------------------------------------------------------------------------------------------------------------

static bool q8pc_env_on(const char * name) {
    const char * s = getenv(name);
    return s && s[0] == '1' && s[1] == '\0';
}

bool ggml_cuda_q8_pc_enabled() {
    static const bool v = q8pc_env_on("GGML_HIP_Q8_PC");
    return v;
}

static bool q8pc_assume() {
    static const bool v = q8pc_env_on("GGML_HIP_Q8_PC_ASSUME");
    return v;
}

static int64_t q8pc_min_n() {
    static const int64_t v = [] {
        const char * s = getenv("GGML_HIP_Q8_PC_MIN_N");
        return s && s[0] ? (int64_t) atoll(s) : (int64_t) 129;
    }();
    return v;
}

static std::atomic<int> q8pc_n_calls{0};

// repacked weights: data pointer -> nbytes; upload progress for chunked writes
static std::mutex                    q8pc_mu;
static std::map<const void *, size_t> q8pc_done;
static std::map<const void *, size_t> q8pc_progress;

// ---------------------------------------------------------------------------------------------------------------------
// load-time in-place repack: Q8_0 row -> per-row int8 scale, stored as Q8_0 blocks that all carry s_r (PC8EMU layout)
// ---------------------------------------------------------------------------------------------------------------------

#define Q8PC_RT 256

static __device__ __forceinline__ float q8pc_block_max(const block_q8_0 & b) {
    int m = 0;
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        m = max(m, abs((int) b.qs[j]));
    }
    return fabsf(__half2float(b.d)) * (float) m;
}

static __device__ float q8pc_block_reduce_max(float v, float * s_red) {
    for (int off = 32; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor(v, off, 64));
    }
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    if (lane == 0) {
        s_red[wave] = v;
    }
    __syncthreads();
    float r = s_red[0];
    for (int w = 1; w < (int) (blockDim.x / 64); ++w) {
        r = fmaxf(r, s_red[w]);
    }
    __syncthreads();
    return r;
}

__launch_bounds__(Q8PC_RT, 1)
static __global__ void q8pc_repack_rows(block_q8_0 * __restrict__ x, const int nb, const int64_t row_blocks) {
    __shared__ float s_red[Q8PC_RT/64];
    block_q8_0 * row = x + (int64_t) blockIdx.x * row_blocks;

    float m = 0.0f;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        m = fmaxf(m, q8pc_block_max(row[b]));
    }
    m = q8pc_block_reduce_max(m, s_red);

    half s = __float2half(m / 127.0f);
    if (__half2float(s) == 0.0f) {
        s = __float2half(1.0f);
    }
    const float s32 = __half2float(s);

    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        block_q8_0 blk = row[b];
        const float d = __half2float(blk.d);
#pragma unroll
        for (int j = 0; j < QK8_0; ++j) {
            const float q = rintf(d * (float) blk.qs[j] / s32);
            blk.qs[j] = (int8_t) fminf(127.0f, fmaxf(-127.0f, q));
        }
        blk.d = s;
        row[b] = blk;
    }
}

static bool q8pc_shape_ok(const ggml_tensor * t) {
    return t->type == GGML_TYPE_Q8_0 && t->ne[2] == 1 && t->ne[3] == 1 && t->ne[0] % 64 == 0 && !t->view_src &&
        strncmp(t->name, "token_embd", 10) != 0;
}

void ggml_cuda_q8_pc_after_write(ggml_tensor * t, size_t offset, size_t size, bool weights_buffer, cudaStream_t stream) {
    if (!ggml_cuda_q8_pc_enabled() || !weights_buffer || !q8pc_shape_ok(t)) {
        return;
    }
    const size_t nbytes = ggml_nbytes(t);
    {
        std::lock_guard<std::mutex> lk(q8pc_mu);
        size_t & p = q8pc_progress[t->data];
        if (offset == 0) {
            p = 0;
        }
        if (offset != p) {   // out-of-order chunking: leave natural Q8_0 (GEMM then refuses it: always correct)
            q8pc_progress.erase(t->data);
            return;
        }
        p += size;
        if (p < nbytes) {
            return;
        }
        q8pc_progress.erase(t->data);
    }
    const int64_t K = t->ne[0];
    const int64_t M = t->ne[1];
    q8pc_repack_rows<<<(unsigned) M, Q8PC_RT, 0, stream>>>((block_q8_0 *) t->data, (int) (K/QK8_0), t->nb[1]/sizeof(block_q8_0));
    CUDA_CHECK(cudaGetLastError());

    std::lock_guard<std::mutex> lk(q8pc_mu);
    q8pc_done[t->data] = nbytes;
    const size_t n = q8pc_done.size();
    if (n == 1 || (n & (n - 1)) == 0) {
        GGML_LOG_INFO("ggml_cuda_q8_pc: re-quantized %zu Q8_0 weights to per-channel int8 in place (last %s %lldx%lld)\n",
            n, t->name, (long long) K, (long long) M);
    }
}

void ggml_cuda_q8_pc_forget_range(const void * base, size_t size) {
    std::lock_guard<std::mutex> lk(q8pc_mu);
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    for (auto it = q8pc_done.lower_bound(lo); it != q8pc_done.end() && (const char *) it->first < hi; ) {
        it = q8pc_done.erase(it);
    }
    for (auto it = q8pc_progress.lower_bound(lo); it != q8pc_progress.end() && (const char *) it->first < hi; ) {
        it = q8pc_progress.erase(it);
    }
}

static bool q8pc_is_repacked(const ggml_tensor * t) {
    if (q8pc_assume()) {
        return true;
    }
    std::lock_guard<std::mutex> lk(q8pc_mu);
    return q8pc_done.count(t->data) != 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// per-token activation quantize: xq[n][k] = rint(x[n][k]/s_a[n]), s_a[n] = max_k |x[n][k]| / 127 (f32)
// ---------------------------------------------------------------------------------------------------------------------

__launch_bounds__(Q8PC_RT, 1)
static __global__ void q8pc_quantize_tokens(const float * __restrict__ x, int8_t * __restrict__ xq, float * __restrict__ sa,
                                            const int K, const int64_t s1) {
    __shared__ float s_red[Q8PC_RT/64];
    const float * xr = x + (int64_t) blockIdx.x * s1;
    int8_t * qr = xq + (int64_t) blockIdx.x * K;

    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        m = fmaxf(m, fabsf(xr[k]));
    }
    m = q8pc_block_reduce_max(m, s_red);
    const float s  = m / 127.0f;
    const float is = s > 0.0f ? 1.0f / s : 0.0f;
    if (threadIdx.x == 0) {
        sa[blockIdx.x] = s;
    }
    for (int k = 4*threadIdx.x; k < K; k += 4*blockDim.x) {   // K % 64 == 0
        char4 q;
        q.x = (int8_t) rintf(xr[k + 0]*is);
        q.y = (int8_t) rintf(xr[k + 1]*is);
        q.z = (int8_t) rintf(xr[k + 2]*is);
        q.w = (int8_t) rintf(xr[k + 3]*is);
        *(char4 *) (qr + k) = q;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMM: dst[n][m] = s_w[m]*s_a[n] * sum_k W[m][k]*X[n][k]
//   W: per-channel Q8_0 rows (34-byte blocks, row_bytes_x apart, block 0's d = s_w[m]); X: int8 rows K bytes apart.
//   One stage = 64 k: a Q8_0 pair is 17 words (qs of the even block at byte 2, odd block at word 9); X is 16 words.
//   Tile 128x128 / 4 waves x 64x64 (2x2 v_mfma_i32_32x32x8i8 tiles), int32 accumulation across all of K (|acc| <=
//   127*127*K < 2^31 for K < 133k), LDS double buffer fed one stage ahead, buffer loads with an SGPR stage offset.
// ---------------------------------------------------------------------------------------------------------------------

#define Q8PC_BM 128
#define Q8PC_BN 128
#define Q8PC_NT 256
#define Q8PC_RW 20  // LDS words per tile row: 16 qs words + 4 pad

#if defined(__gfx90a__)
#define Q8PC_DEVICE 1
typedef int32_t q8pc_i32x16 __attribute__((ext_vector_type(16)));
typedef int32_t q8pc_i32x4  __attribute__((ext_vector_type(4)));

extern "C" __device__ int32_t q8pc_buffer_load_i32(q8pc_i32x4 rsrc, int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

static __device__ __forceinline__ q8pc_i32x4 q8pc_make_rsrc(const void * base, const uint32_t num_bytes) {
    const uint64_t a = (uint64_t) base;
    q8pc_i32x4 r;
    r.x = __builtin_amdgcn_readfirstlane((int32_t) (uint32_t) a);
    r.y = __builtin_amdgcn_readfirstlane((int32_t) (uint32_t) (a >> 32));
    r.z = __builtin_amdgcn_readfirstlane((int32_t) num_bytes);
    r.w = 0x00020000;
    return r;
}
#endif

__launch_bounds__(Q8PC_NT, 1)
static __global__ void mul_mat_q8_0_pc_i8(
        const char * __restrict__ x, const int8_t * __restrict__ y, const float * __restrict__ sa, float * __restrict__ dst,
        const int M, const int N, const int nstage, const int64_t row_bytes_x, const int64_t row_bytes_y, const int64_t stride_dst) {
#if defined(Q8PC_DEVICE)
    __shared__ __attribute__((aligned(16))) int32_t s_q[2][2][Q8PC_BM*Q8PC_RW];

    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid % 64;
    const int l32  = lane % 32;
    const int h    = lane / 32;

    const int n0 = blockIdx.x * Q8PC_BN;
    const int m0 = blockIdx.y * Q8PC_BM;
    const int wm = (wave % 2) * 64;
    const int wn = (wave / 2) * 64;

    const int lj  = tid % 16;
    const int lr0 = tid / 16;

    const q8pc_i32x4 rsrc_x = q8pc_make_rsrc(x, (uint32_t) ((int64_t) M*row_bytes_x));
    const q8pc_i32x4 rsrc_y = q8pc_make_rsrc(y, (uint32_t) ((int64_t) N*row_bytes_y));

    int voa[8];
    int vob[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        voa[i] = min(m0 + lr0 + 16*i, M - 1)*(int) row_bytes_x + 4*lj;
        vob[i] = min(n0 + lr0 + 16*i, N - 1)*(int) row_bytes_y + 4*lj;
    }

    int32_t ra_lo[8], ra_hi[8], rb_q[8];

    auto load_stage = [&](const int s) {
        const int oa = __builtin_amdgcn_readfirstlane(s*68);
        const int ob = __builtin_amdgcn_readfirstlane(s*64);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ra_lo[i] = q8pc_buffer_load_i32(rsrc_x, voa[i],     oa, 0);
            ra_hi[i] = q8pc_buffer_load_i32(rsrc_x, voa[i] + 4, oa, 0);
            rb_q[i]  = q8pc_buffer_load_i32(rsrc_y, vob[i],     ob, 0);
        }
    };

    auto store_stage = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = lr0 + 16*i;
            const int32_t qa = lj < 8 ? (int32_t) __builtin_amdgcn_alignbyte((uint32_t) ra_hi[i], (uint32_t) ra_lo[i], 2)
                                      : ra_hi[i];
            s_q[buf][0][row*Q8PC_RW + lj] = qa;
            s_q[buf][1][row*Q8PC_RW + lj] = rb_q[i];
        }
    };

    q8pc_i32x16 c[2][2];
#pragma unroll
    for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
        for (int tn = 0; tn < 2; ++tn) {
            c[tm][tn] = q8pc_i32x16{0};
        }
    }

    load_stage(0);
    store_stage(0);
    load_stage(min(1, nstage - 1));
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;
#pragma unroll
        for (int qb = 0; qb < 2; ++qb) {
            q8pc_i32x4 a[2];
            q8pc_i32x4 b[2];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                a[t] = *(const q8pc_i32x4 *) &s_q[buf][0][(wm + 32*t + l32)*Q8PC_RW + 8*qb + 4*h];
                b[t] = *(const q8pc_i32x4 *) &s_q[buf][1][(wn + 32*t + l32)*Q8PC_RW + 8*qb + 4*h];
            }
#pragma unroll
            for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
                for (int tn = 0; tn < 2; ++tn) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        c[tm][tn] = __builtin_amdgcn_mfma_i32_32x32x8i8(a[tm][j], b[tn][j], c[tm][tn], 0, 0, 0);
                    }
                }
            }
        }
        store_stage(buf ^ 1);
        load_stage(min(s + 2, nstage - 1));
        __syncthreads();
    }

    // epilogue: register 4g+e of a 32x32x8 tile is row 8g+4h+e, column l32
#pragma unroll
    for (int tn = 0; tn < 2; ++tn) {
        const int n = n0 + wn + 32*tn + l32;
        if (n >= N) {
            continue;
        }
        const float san = sa[n];
        float * dn = dst + (int64_t) n*stride_dst;
#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int m = m0 + wm + 32*tm + 8*(r/4) + 4*h + (r%4);
                if (m < M) {
                    const float sw = __half2float(*(const half *) (x + (int64_t) m*row_bytes_x));
                    dn[m] = (float) c[tm][tn][r] * (sw*san);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sa, dst, M, N, nstage, row_bytes_x, row_bytes_y, stride_dst);
    NO_DEVICE_CODE;
#endif // defined(Q8PC_DEVICE)
}

// ---------------------------------------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_q8_pc_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!ggml_cuda_q8_pc_enabled() || cc != GGML_CUDA_CC_CDNA2) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (N < q8pc_min_n() || K % 64 != 0 || M > INT32_MAX || N > INT32_MAX) {
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
    if ((int64_t) ggml_nbytes(src0) >= INT32_MAX || N*K >= INT32_MAX) {   // 32-bit buffer offsets
        return false;
    }
    return q8pc_is_repacked(src0->view_src ? src0->view_src : src0);
}

bool ggml_cuda_q8_pc_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_q8_pc_eligible(src0, src1, dst, cc)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(), N*K);
    ggml_cuda_pool_alloc<float>  sa(ctx.pool(), N);
    q8pc_quantize_tokens<<<(unsigned) N, Q8PC_RT, 0, stream>>>(
        (const float *) src1->data, xq.get(), sa.get(), (int) K, (int64_t) (src1->nb[1]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid((N + Q8PC_BN - 1)/Q8PC_BN, (M + Q8PC_BM - 1)/Q8PC_BM, 1);
    mul_mat_q8_0_pc_i8<<<grid, Q8PC_NT, 0, stream>>>(
        (const char *) src0->data, xq.get(), sa.get(), (float *) dst->data,
        (int) M, (int) N, (int) (K/64), (int64_t) src0->nb[1], K, (int64_t) (dst->nb[1]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());

    if (q8pc_n_calls.fetch_add(1, std::memory_order_relaxed) == 0) {
        GGML_LOG_INFO("ggml_cuda_q8_pc: gfx90a AK-ASM-I8GEMM-PERCHANNEL per-token x per-channel int8 MFMA GEMM ran (call 1)\n");
    }
    return true;
}
