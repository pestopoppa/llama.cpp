#include "q8-pipe.cuh"
#include "quantize.cuh"
#include "mmvq.cuh"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------------
// env switches
// ---------------------------------------------------------------------------------------------------------------------

static bool q8p_env_flag(const char * name, bool def) {
    const char * v = getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return def;
    }
    return !(v[0] == '0' && v[1] == '\0');
}

static bool q8p_h4_enabled() {
    static const bool v = q8p_env_flag("GGML_HIP_Q8_PIPE", true);
    return v;
}

static int64_t q8p_h4_min_n() {
    static const int64_t v = [] {
        const char * s = getenv("GGML_HIP_Q8_PIPE_MIN_N");
        return s && s[0] ? (int64_t) atoll(s) : (int64_t) 129;
    }();
    return v;
}

static bool q8p_h6_enabled() {
    static const bool v = q8p_env_flag("GGML_HIP_RMS_Q8", true);
    return v;
}

// Path witnesses: one log line per process on the first use of each path (gate rows require / forbid them).
static std::atomic<int> q8p_n_h4{0};
static std::atomic<int> q8p_n_h6{0};
static std::atomic<int> q8p_n_slot{0};

static void q8p_witness(std::atomic<int> & n, const char * line) {
    if (n.fetch_add(1, std::memory_order_relaxed) == 0) {
        GGML_LOG_INFO("%s\n", line);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// H4 kernel: dst[n][m] = sum_k W[m][k] * X[n][k], W Q8_0 (row-major blocks), X plain block_q8_1 rows of ne10_padded.
// ---------------------------------------------------------------------------------------------------------------------

#define Q8P_BM  128
#define Q8P_BN  128
#define Q8P_NT  256
#define Q8P_RW  20  // LDS words per tile row: 2 q-blocks x 8 words + 4 pad (keeps 16-byte alignment, staggers banks)

#if defined(__gfx90a__)
#define Q8P_DEVICE 1
typedef int32_t q8p_i32x16 __attribute__((ext_vector_type(16)));
typedef int32_t q8p_i32x4  __attribute__((ext_vector_type(4)));
#endif

__launch_bounds__(Q8P_NT, 1)
static __global__ void mul_mat_q8_0_q8_1_pipe(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst,
        const int M, const int N, const int nstage,
        const int64_t row_bytes_x, const int64_t row_bytes_y, const int64_t stride_dst) {
#if defined(Q8P_DEVICE)
    // [buf][role: 0 = weights, 1 = activations][row * Q8P_RW + word]
    __shared__ __attribute__((aligned(16))) int32_t s_q[2][2][128*Q8P_RW];
    // [buf][role][q-block][row]
    __shared__ __attribute__((aligned(16))) float   s_d[2][2][2][128];

    const int tid  = threadIdx.x;
    const int wave = tid / 64;
    const int lane = tid % 64;
    const int l32  = lane % 32;
    const int h    = lane / 32;

    const int n0 = blockIdx.x * Q8P_BN;
    const int m0 = blockIdx.y * Q8P_BM;
    const int wm = (wave % 2) * 64;
    const int wn = (wave / 2) * 64;

    // loader role is wave-uniform: waves 0,1 load 128 weight rows, waves 2,3 load 128 activation rows
    const int  role = __builtin_amdgcn_readfirstlane(tid / 128); // scalar: keeps the loop one basic block
    const bool isA  = role == 0;
    const int  lr   = tid % 128;
    const int  grow = isA ? min(m0 + lr, M - 1) : min(n0 + lr, N - 1);
    const int32_t * gp = (const int32_t *) (isA ? x + (int64_t) grow*row_bytes_x : y + (int64_t) grow*row_bytes_y);
    // one stage = 64 k = 2 blocks: Q8_0 pair is 68 bytes (17 words), q8_1 pair 72 bytes (18 words)
    const int stage_words = isA ? 17 : 18;
    const int last_word   = isA ? 16 : 17;

    int32_t w[18];

    auto load_stage = [&](const int s) {
        const int32_t * p = gp + (int64_t) s*stage_words;
#pragma unroll
        for (int k = 0; k < 17; ++k) {
            w[k] = p[k];
        }
        w[17] = p[last_word];
    };

    auto store_stage = [&](const int buf) {
        int32_t q[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // Q8_0: qs0 starts at byte 2 of the pair, qs1 at byte 36 (word 9). q8_1: qs0 at word 1, qs1 at word 10.
            const int32_t a0 = (int32_t) __builtin_amdgcn_alignbyte((uint32_t) w[j + 1], (uint32_t) w[j], 2);
            q[j]     = isA ? a0        : w[j + 1];
            q[8 + j] = isA ? w[9 + j]  : w[10 + j];
        }
        int32_t * dq = &s_q[buf][role][lr*Q8P_RW];
        *(q8p_i32x4 *) (dq +  0) = q8p_i32x4{q[ 0], q[ 1], q[ 2], q[ 3]};
        *(q8p_i32x4 *) (dq +  4) = q8p_i32x4{q[ 4], q[ 5], q[ 6], q[ 7]};
        *(q8p_i32x4 *) (dq +  8) = q8p_i32x4{q[ 8], q[ 9], q[10], q[11]};
        *(q8p_i32x4 *) (dq + 12) = q8p_i32x4{q[12], q[13], q[14], q[15]};

        const uint32_t d0b = (uint32_t) w[0] & 0xFFFFu;
        const uint32_t d1b = isA ? ((uint32_t) w[8] >> 16) : ((uint32_t) w[9] & 0xFFFFu);
        s_d[buf][role][0][lr] = __half2float(__ushort_as_half((unsigned short) d0b));
        s_d[buf][role][1][lr] = __half2float(__ushort_as_half((unsigned short) d1b));
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

    // Per stage (64 k): MFMA from s_q[buf] + scale into f32 accumulators, then refill the other buffer from the
    // registers loaded one stage earlier and issue the next global prefetch (in flight across the next stage).
    // (A one-stage-delayed scale pipeline was tried: on ROCm 6.2 clang it spills 656 B/lane at 256 AGPRs and the
    // scheduler still does not interleave across the MFMA burst, so the simple form is kept.)
    load_stage(0);
    store_stage(0);
    load_stage(min(1, nstage - 1)); // stage 1 in flight during stage 0 compute
    __syncthreads();

    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1;

#pragma unroll
        for (int qb = 0; qb < 2; ++qb) {
            q8p_i32x4 a[2];
            q8p_i32x4 b[2];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                // K permutation inside a block: lane half h supplies word 4h+j to MFMA j; identical for A and B,
                // so each MFMA sums words {j, 4+j} and the 4 MFMAs cover all 8 words of the block exactly once.
                a[t] = *(const q8p_i32x4 *) &s_q[buf][0][(wm + 32*t + l32)*Q8P_RW + 8*qb + 4*h];
                b[t] = *(const q8p_i32x4 *) &s_q[buf][1][(wn + 32*t + l32)*Q8P_RW + 8*qb + 4*h];
            }
#pragma unroll
            for (int tm = 0; tm < 2; ++tm) {
#pragma unroll
                for (int tn = 0; tn < 2; ++tn) {
                    q8p_i32x16 c = {0};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        c = __builtin_amdgcn_mfma_i32_32x32x8i8(a[tm][j], b[tn][j], c, 0, 0, 0);
                    }
                    const float db = s_d[buf][1][qb][wn + 32*tn + l32];
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        // accumulator row of register 4g+e: 8g + 4h + e (32x32x8 single-block layout), column l32
                        const float4 da = *(const float4 *) &s_d[buf][0][qb][wm + 32*tm + 8*g + 4*h];
                        acc[tm][tn][4*g + 0] = fmaf((float) c[4*g + 0], da.x*db, acc[tm][tn][4*g + 0]);
                        acc[tm][tn][4*g + 1] = fmaf((float) c[4*g + 1], da.y*db, acc[tm][tn][4*g + 1]);
                        acc[tm][tn][4*g + 2] = fmaf((float) c[4*g + 2], da.z*db, acc[tm][tn][4*g + 2]);
                        acc[tm][tn][4*g + 3] = fmaf((float) c[4*g + 3], da.w*db, acc[tm][tn][4*g + 3]);
                    }
                }
            }
        }

        // Interleave hint (AITER style): alternate LDS operand reads, MFMA groups and the VALU convert/scale work.
        // Pure scheduling hint, no semantic effect.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);  // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);  // MFMA
            __builtin_amdgcn_sched_group_barrier(0x002, 12, 0); // VALU
        }
        // Keep the compute above the LDS refill: the refill's global loads were issued one stage ago, so their
        // latency hides under the MFMAs and the vmcnt wait lands here, not before the compute.
        __builtin_amdgcn_sched_barrier(0);

        store_stage(buf ^ 1);
        load_stage(min(s + 2, nstage - 1)); // next prefetch (clamped: redundant reload on the last stages)
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
#endif // defined(Q8P_DEVICE)
}

// ---------------------------------------------------------------------------------------------------------------------
// q8_1 slot: one live H6-produced q8_1 copy per context, in a grow-only device buffer (never freed while graphs that
// captured its address may still launch; older buffers are kept until the context is destroyed).
// ---------------------------------------------------------------------------------------------------------------------

struct q8p_slot_state {
    int                 device = -1;
    std::vector<void *> retired;
    void *              buf      = nullptr;
    size_t              buf_size = 0;

    const ggml_tensor * tensor = nullptr;
    const void *        data   = nullptr;
    size_t              nbytes = 0;
    int64_t             ne[GGML_MAX_DIMS] = {};
    size_t              nb[GGML_MAX_DIMS] = {};
    int64_t             ne10_padded = 0;

    void invalidate() {
        tensor = nullptr;
        data   = nullptr;
        nbytes = 0;
    }
};

static std::mutex q8p_slot_mutex;
static std::unordered_map<const ggml_backend_cuda_context *, q8p_slot_state> q8p_slots;

static q8p_slot_state & q8p_slot_get(ggml_backend_cuda_context & ctx) {
    q8p_slot_state & st = q8p_slots[&ctx];
    st.device = ctx.device;
    return st;
}

static void * q8p_slot_reserve(q8p_slot_state & st, const size_t size) {
    if (st.buf_size < size) {
        if (st.buf != nullptr) {
            st.retired.push_back(st.buf);
        }
        const size_t alloc = GGML_PAD(size, (size_t) 1 << 20);
        ggml_cuda_set_device(st.device);
        CUDA_CHECK(cudaMalloc(&st.buf, alloc));
        st.buf_size = alloc;
    }
    return st.buf;
}

const void * ggml_cuda_q8_1_slot_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const int64_t ne10_padded) {
    if (ctx.curr_stream_no != 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(q8p_slot_mutex);
    auto it = q8p_slots.find(&ctx);
    if (it == q8p_slots.end()) {
        return nullptr;
    }
    const q8p_slot_state & st = it->second;
    if (st.tensor == nullptr || st.tensor != src1 || st.data != src1->data || st.ne10_padded != ne10_padded) {
        return nullptr;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (st.ne[i] != src1->ne[i] || st.nb[i] != src1->nb[i]) {
            return nullptr;
        }
    }
    q8p_witness(q8p_n_slot, "ggml_cuda_q8_1_slot: fused rms_norm q8_1 copy reused (call 1)");
    return st.buf;
}

void ggml_cuda_q8_1_slot_observe(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(q8p_slot_mutex);
    auto it = q8p_slots.find(&ctx);
    if (it == q8p_slots.end() || it->second.tensor == nullptr) {
        return;
    }
    q8p_slot_state & st = it->second;
    if (node == st.tensor || node->data == nullptr) {
        return;
    }
    const char * a0 = (const char *) node->data;
    const char * a1 = a0 + ggml_nbytes(node);
    const char * b0 = (const char *) st.data;
    const char * b1 = b0 + st.nbytes;
    if (a0 < b1 && b0 < a1) {
        st.invalidate();
    }
}

void ggml_cuda_q8_1_slot_reset(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(q8p_slot_mutex);
    auto it = q8p_slots.find(&ctx);
    if (it != q8p_slots.end()) {
        it->second.invalidate();
    }
}

void ggml_cuda_q8_1_slot_release(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(q8p_slot_mutex);
    auto it = q8p_slots.find(ctx);
    if (it == q8p_slots.end()) {
        return;
    }
    q8p_slot_state & st = it->second;
    if (st.buf != nullptr || !st.retired.empty()) {
        ggml_cuda_set_device(st.device);
        for (void * p : st.retired) {
            CUDA_CHECK(cudaFree(p));
        }
        if (st.buf != nullptr) {
            CUDA_CHECK(cudaFree(st.buf));
        }
    }
    q8p_slots.erase(it);
}

// ---------------------------------------------------------------------------------------------------------------------
// H4 host side
// ---------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_q8_pipe_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!q8p_h4_enabled() || cc != GGML_CUDA_CC_CDNA2) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (N < q8p_h4_min_n() || K % 64 != 0 || M > INT32_MAX || N > INT32_MAX) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[1] != (size_t) (K/QK8_0)*sizeof(block_q8_0) || src0->nb[1] % 4 != 0 || ((uintptr_t) src0->data) % 4 != 0) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || dst->ne[0] != M || dst->ne[1] != N) {
        return false;
    }
    return true;
}

bool ggml_cuda_q8_pipe_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_q8_pipe_eligible(src0, src1, dst, cc)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();

    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t  q8_1_size   = N*ne10_padded*sizeof(block_q8_1)/QK8_1;

    ggml_cuda_pool_alloc<char> q8_1_alloc(ctx.pool());
    const void * q8_1 = ggml_cuda_q8_1_slot_lookup(ctx, src1, ne10_padded);
    if (q8_1 == nullptr) {
        char * buf = q8_1_alloc.alloc(q8_1_size);
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, buf, src0->type, K, s11, s12, s13, ne10_padded, N, 1, 1, stream);
        q8_1 = buf;
    }

    const dim3 grid((N + Q8P_BN - 1)/Q8P_BN, (M + Q8P_BM - 1)/Q8P_BM, 1);
    mul_mat_q8_0_q8_1_pipe<<<grid, Q8P_NT, 0, stream>>>(
        (const char *) src0->data, (const char *) q8_1, (float *) dst->data,
        (int) M, (int) N, (int) (K/64),
        (int64_t) src0->nb[1], (int64_t) (ne10_padded/QK8_1*sizeof(block_q8_1)), (int64_t) (dst->nb[1]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    q8p_witness(q8p_n_h4, "ggml_cuda_q8_pipe: gfx90a pipelined Q8_0 int8 MFMA GEMM ran (call 1)");
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// H6: fused rms_norm * weight -> f32 dst + plain q8_1 copy (bit-identical to quantize_q8_1<false> of that dst)
// ---------------------------------------------------------------------------------------------------------------------

template <int block_size>
__launch_bounds__(block_size, 1)
static __global__ void rms_norm_mul_q8_1_f32(
        const float * __restrict__ x, const float * __restrict__ mul, float * __restrict__ dst, block_q8_1 * __restrict__ q,
        const int ncols, const int64_t stride_row, const int64_t stride_dst, const int64_t ncols_pad, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    x   += row*stride_row;
    dst += row*stride_dst;
    q   += row*(ncols_pad/QK8_1);

    // identical reduction to rms_norm_f32<block_size> so the scale (and hence dst) matches the unfused path
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    // block_size is a multiple of 32 and ncols_pad a multiple of 32: every 32-lane group covers whole q8_1 blocks
    for (int col = tid; col < ncols_pad; col += block_size) {
        float v = 0.0f;
        if (col < ncols) {
            v = scale * x[col] * mul[col];
            dst[col] = v;
        }
        float amax = fabsf(v);
        float sum  = v;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d  = amax / 127.0f;
        const int8_t qv = amax == 0.0f ? 0 : roundf(v / d);
        q[col/QK8_1].qs[col % QK8_1] = qv;
        if (col % QK8_1 == 0) {
            q[col/QK8_1].ds = make_half2(d, sum);
        }
    }
}

// Does some consumer of `t` (within the next few nodes) read its q8_1 copy through MMVQ or H4?
static bool q8p_has_q8_1_consumer(ggml_cgraph * cgraph, const int i_first, const ggml_tensor * t, const int cc) {
    const int lim = std::min(cgraph->n_nodes, i_first + 32);
    for (int j = i_first; j < lim; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op != GGML_OP_MUL_MAT || n->src[1] != t) {
            continue;
        }
        const ggml_tensor * w = n->src[0];
        if (!ggml_is_quantized(w->type) || w->type == GGML_TYPE_Q4_K) {
            continue;
        }
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            continue;
        }
        if (ggml_cuda_should_use_mmvq(w->type, cc, t->ne[1]) || ggml_cuda_q8_pipe_eligible(w, t, n, cc)) {
            return true;
        }
    }
    return false;
}

bool ggml_cuda_op_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i_rms) {
    if (!q8p_h6_enabled() || ctx.curr_stream_no != 0) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_CDNA2) {
        return false;
    }
    const ggml_tensor * rms = cgraph->nodes[i_rms];
    ggml_tensor * mul = cgraph->nodes[i_rms + 1];
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];

    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t ncols = x->ne[0];
    if (ncols % QK8_1 != 0 || x->nb[0] != sizeof(float) || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    if (!ggml_are_same_shape(mul, x) || !ggml_is_contiguous(mul)) {
        return false;
    }
    if (w->ne[0] != ncols || ggml_nrows(w) != 1 || !ggml_is_contiguous(w)) {
        return false;
    }
    if (!q8p_has_q8_1_consumer(cgraph, i_rms + 2, mul, cc)) {
        return false;
    }

    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));

    const int64_t nrows     = x->ne[1];
    const int64_t ncols_pad = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const size_t  q_size    = nrows*ncols_pad*sizeof(block_q8_1)/QK8_1;

    void * qbuf;
    {
        std::lock_guard<std::mutex> lock(q8p_slot_mutex);
        q8p_slot_state & st = q8p_slot_get(ctx);
        qbuf = q8p_slot_reserve(st, q_size);
        st.tensor      = mul;
        st.data        = mul->data;
        st.nbytes      = ggml_nbytes(mul);
        st.ne10_padded = ncols_pad;
        for (int k = 0; k < GGML_MAX_DIMS; ++k) {
            st.ne[k] = mul->ne[k];
            st.nb[k] = mul->nb[k];
        }
    }

    cudaStream_t stream = ctx.stream();
    const int64_t stride_row = x->nb[1] / sizeof(float);
    const int64_t stride_dst = mul->nb[1] / sizeof(float);
    if (ncols < 1024) {
        rms_norm_mul_q8_1_f32<256><<<nrows, 256, 0, stream>>>((const float *) x->data, (const float *) w->data,
            (float *) mul->data, (block_q8_1 *) qbuf, (int) ncols, stride_row, stride_dst, ncols_pad, eps);
    } else {
        rms_norm_mul_q8_1_f32<1024><<<nrows, 1024, 0, stream>>>((const float *) x->data, (const float *) w->data,
            (float *) mul->data, (block_q8_1 *) qbuf, (int) ncols, stride_row, stride_dst, ncols_pad, eps);
    }
    CUDA_CHECK(cudaGetLastError());
    q8p_witness(q8p_n_h6, "ggml_cuda_rms_q8: fused rms_norm+mul+q8_1 quantize ran (call 1)");
    return true;
}
