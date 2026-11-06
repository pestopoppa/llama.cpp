// AK-ASM-H5-PORT: FLASH_ATTN_EXT fast path on gfx90a through the retargeted AITER FA-v3 fwd hd128 bf16 asm kernels.
//
// What runs (all on the op's stream):
//   1. mask classification (only when there is a mask; cached per graph compute and mask tensor, see below):
//      two small kernels scan the first n_q rows x n_kv columns of the f16 KQ mask and decide whether it is exactly
//        - "causal": row r is 0 on [0, L - n_q + r] and -INF on (L - n_q + r, n_kv)   (bottom-right aligned), or
//        - "full":   every row is 0 on [0, L) and -INF on [L, n_kv),
//      for one effective KV length L (the -INF tail is llama.cpp's KV padding / unused cells). Any other value
//      (noise, ALiBi slopes, -65504, a hole, a sliding window, another sequence's cells) => fallback. The verdict is
//      read back to the host (one stream sync), because the kernel choice and its kernargs are host-side.
//   2. Q (f32|f16), K and V (f16|bf16|q8_0, first L cells only) -> bf16 BSHD scratch (pool), RNE rounding.
//   3. hipModuleLaunchKernel of fmha_fwd_hd128_bf16[_causal]_rtne with AITER's packed 512-byte kernarg blob
//      (fmha_fwd_v3_args, mha_fwd.h:274; values as init_fmha_fwd_v3_args, mha_fwd.cu:81-174, batch mode), block 512,
//      grid (ceil(ceil(n_q/256)/tg), H, 1), tg = 2 for causal. Exactly the configuration the AK-ASM-H5 harness ran.
//   4. O (bf16 BSHD) -> dst (f32). ggml's FA dst [D, H, n_q, 1] is BSHD already, so this is element-wise.
//
// Everything else returns false before anything is enqueued, and the caller runs the existing selection unchanged.
//
// Env (read once):
//   GGML_HIP_FA_ASM=0              disable every fast path (hd128 asm and hd256 CK) at run time
//   GGML_HIP_FA_ASM=1              also enable the hd128 bf16 asm path, which is OFF by default (bf16 PPL drift, R5)
//   GGML_HIP_FA_ASM_MIN_BATCH=n    smallest query-row count that takes the fast path (default 64, minimum 1)
//   GGML_HIP_FA_ASM_LOG=1          log every fast-path decision (default: one line at first use, one per new reject)
//
// AK-CK-HD256 (GGML_HIP_CK_FMHA_HD256): head_dim 256 takes a CK-tile fmha forward kernel instead
// (ggml/src/ggml-hip/ck-fmha-gfx90a/, composable_kernel rocm-6.2.4 headers, fp16, qr pipeline, compiled into this
// library by ROCm 6.2 hipcc). Same gates as above (mask classification, threshold, env), with these differences:
//   - K and V must be f16 and are read in place (no conversion, no copy); only the first L cells are attended.
//   - Q (f32|f16) -> f16 BSHD scratch (RNE; the existing f16 FA kernels make the same rounding), O f16 -> dst f32.
//     fp16 instead of bf16: 3 more mantissa bits on Q/P/O, and K/V need no conversion. The instance accumulates S and
//     O in fp32; P (softmax probabilities in [0, 1]) is rounded to fp16 for the second GEMM, as in the C6 kernels.
//   - GQA ratios {1, 2, 4, 6, 8, 12} (Qwen3.8-27B 24/4, Qwen3.6-35B-A3B 16/2, Q38FN 24/2).
//   GGML_HIP_FA_CK_HD256=0         disable only the hd256 CK path at run time (default: enabled when compiled in)

#include "fattn-asm-gfx90a.cuh"

#ifdef GGML_CUDA_FA_ASM_GFX90A_AVAILABLE

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

#ifdef GGML_HIP_CK_FMHA_HD256
#include "../ggml-hip/ck-fmha-gfx90a/ck_fmha_hd256.h"
#endif

extern "C" {
extern const unsigned char ggml_hip_fa_asm_fwd_hd128_bf16_rtne[];
extern const size_t        ggml_hip_fa_asm_fwd_hd128_bf16_rtne_size;
extern const char          ggml_hip_fa_asm_fwd_hd128_bf16_rtne_sha256[];
extern const unsigned char ggml_hip_fa_asm_fwd_hd128_bf16_causal_rtne[];
extern const size_t        ggml_hip_fa_asm_fwd_hd128_bf16_causal_rtne_size;
extern const char          ggml_hip_fa_asm_fwd_hd128_bf16_causal_rtne_sha256[];
}

namespace {

constexpr int     FA_ASM_D      = 128;  // the only head size these kernels implement
constexpr int     FA_CK_D       = 256;  // AK-CK-HD256: the CK-tile instances' head size
constexpr int     FA_ASM_TS_QO  = 256;  // fmha_fwd.csv ts_qo for hd128 bf16 (query rows per tile)
constexpr int     FA_ASM_BLOCK  = 512;  // mha_fwd.cu: bdx = 512
constexpr size_t  FA_ASM_KARGS  = 512;  // .kernarg_segment_size of both code objects

constexpr int FA_ASM_MASK_UNSUPPORTED = 0;
constexpr int FA_ASM_MASK_FULL        = 1;
constexpr int FA_ASM_MASK_CAUSAL      = 2;

const char * const k_fa_asm_kernel_name[2] = {
    "_ZN5aiter24fmha_fwd_hd128_bf16_rtneE",         // [0] full (no causal)
    "_ZN5aiter31fmha_fwd_hd128_bf16_causal_rtneE",  // [1] causal, AITER mask_type 2 (bottom-right)
};

bool fa_asm_env_flag(const char * name, bool def) {
    const char * e = getenv(name);
    return e ? atoi(e) != 0 : def;
}

bool fa_asm_enabled() {
    static const bool enabled = fa_asm_env_flag("GGML_HIP_FA_ASM", true);
    return enabled;
}

// AK-ASM-H5-PORT-R5: the hd128 asm kernels are bf16-only (AITER ships no fp16 FA-v3 fwd); Q, K, V and O are rounded
// to bf16 (8-bit mantissa vs f16's 11) and that drifts Qwen3-1.7B PPL +1.27% (20.2097 vs 19.9571, bound 0.5%).
// So hd128 is OFF by default and taken only when GGML_HIP_FA_ASM is set explicitly to a non-zero value.
// GGML_HIP_FA_ASM=0 still disables every fast path (hd128 asm and hd256 CK); unset leaves hd256 CK on.
bool fa_asm_hd128_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_HIP_FA_ASM");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
}

// AK-ASM-T0 (ROCm 6.2): hipModuleLaunchKernel captures and replays correctly, the kernarg buffer being copied
// into the graph node at capture. GGML_HIP_FA_ASM_GRAPH=0 restores the pre-T0 exclusion from CUDA-graph capture.
bool fa_asm_graph_enabled() {
    static const bool enabled = fa_asm_env_flag("GGML_HIP_FA_ASM_GRAPH", true);
    return enabled;
}

bool fa_asm_log_all() {
    static const bool v = fa_asm_env_flag("GGML_HIP_FA_ASM_LOG", false);
    return v;
}

// Default 64 [I, not measured]: the kernel computes whole 256-row query tiles, so at n_q rows only n_q/256 of its
// MFMA work is useful (at ~114 TFLOPS dense that is ~28 TFLOPS effective at 64 rows, about C6's large-shape FA
// rate), and every call also converts the first L cells of K and V to bf16 (a fixed cost that decode cannot
// amortize). Decode (1 row) and MTP verify (2-5 rows) therefore stay on the existing kernels. The crossover must
// be measured (test-backend-ops perf at n_q 16..256); GGML_HIP_FA_ASM_MIN_BATCH=1 forces the path for tests.
bool fa_ck_hd256_enabled() {
#ifdef GGML_HIP_CK_FMHA_HD256
    static const bool enabled = fa_asm_env_flag("GGML_HIP_FA_CK_HD256", true);
    return enabled;
#else
    return false;
#endif
}

int64_t fa_asm_min_batch() {
    static const int64_t v = [] {
        const char * e = getenv("GGML_HIP_FA_ASM_MIN_BATCH");
        const long long n = e ? atoll(e) : 64;
        return (int64_t) (n < 1 ? 1 : n);
    }();
    return v;
}

std::atomic<uint64_t> g_fa_asm_n_calls{0}; // every fast-path call (asm hd128 + CK hd256)
std::atomic<uint64_t> g_fa_ck_n_calls{0};  // AK-CK-HD256 calls only
std::atomic<uint64_t> g_fa_asm_n_captured{0}; // fast-path calls recorded into a CUDA graph (graph-mode witness)

void fa_asm_note_captured(const ggml_tensor * dst, const bool ck) {
    const uint64_t n = g_fa_asm_n_captured.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1 || fa_asm_log_all()) {
        GGML_LOG_INFO("ggml_cuda_fa_asm: fast path captured into a CUDA graph (call %llu) node=%s %s\n",
                      (unsigned long long) n, dst->name, ck ? "ck-hd256" : "asm-hd128");
    }
}

struct fa_asm_module {
    std::once_flag    once;
    std::atomic<bool> done{false}; // call_once finished (no module API call can follow; safe inside capture)
    bool              ok = false;
    hipModule_t    mod[2] = { nullptr, nullptr };
    hipFunction_t  fn[2]  = { nullptr, nullptr };
};

fa_asm_module g_fa_asm_modules[GGML_CUDA_MAX_DEVICES];

// Lazily load both code objects on the current device. Called outside stream capture only (the graph pre-pass loads
// them before capture, and _try() refuses while capturing if they were never loaded).
bool fa_asm_load(int device) {
    fa_asm_module & m = g_fa_asm_modules[device];
    std::call_once(m.once, [&] {
        const unsigned char * images[2] = { ggml_hip_fa_asm_fwd_hd128_bf16_rtne, ggml_hip_fa_asm_fwd_hd128_bf16_causal_rtne };
        const char * shas[2] = { ggml_hip_fa_asm_fwd_hd128_bf16_rtne_sha256, ggml_hip_fa_asm_fwd_hd128_bf16_causal_rtne_sha256 };
        bool ok = true;
        for (int i = 0; i < 2 && ok; ++i) {
            hipError_t err = hipModuleLoadData(&m.mod[i], images[i]);
            if (err != hipSuccess) {
                GGML_LOG_WARN("%s: device %d: hipModuleLoadData(%s) failed: %s; FLASH_ATTN_EXT keeps the existing kernels\n",
                              __func__, device, k_fa_asm_kernel_name[i], hipGetErrorString(err));
                (void) hipGetLastError();
                m.mod[i] = nullptr;
                ok = false;
                break;
            }
            err = hipModuleGetFunction(&m.fn[i], m.mod[i], k_fa_asm_kernel_name[i]);
            if (err != hipSuccess) {
                GGML_LOG_WARN("%s: device %d: hipModuleGetFunction(%s) failed: %s; FLASH_ATTN_EXT keeps the existing kernels\n",
                              __func__, device, k_fa_asm_kernel_name[i], hipGetErrorString(err));
                (void) hipGetLastError();
                ok = false;
                break;
            }
            GGML_LOG_INFO("%s: device %d: loaded %s (embedded code object sha256 %s)\n",
                          __func__, device, k_fa_asm_kernel_name[i], shas[i]);
        }
        if (!ok) {
            for (int i = 0; i < 2; ++i) {
                if (m.mod[i] != nullptr) {
                    (void) hipModuleUnload(m.mod[i]);
                    m.mod[i] = nullptr;
                }
                m.fn[i] = nullptr;
            }
        }
        m.ok = ok;
        m.done.store(true, std::memory_order_release);
    });
    return m.ok;
}

bool fa_asm_device_ok(int device) {
    return device >= 0 && device < GGML_CUDA_MAX_DEVICES && ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_CDNA2;
}

bool fa_asm_kv_type_ok(ggml_type t) {
    return t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_Q8_0;
}

// Shape/type/param predicate; *why names the first failed condition (for the log).
bool fa_asm_shape_ok(int device, const ggml_tensor * dst, const char ** why) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sink = dst->src[4];

#define FA_ASM_REQUIRE(cond, msg) do { if (!(cond)) { *why = msg; return false; } } while (0)
    FA_ASM_REQUIRE(fa_asm_enabled(), "disabled (GGML_HIP_FA_ASM=0)");
    FA_ASM_REQUIRE(fa_asm_device_ok(device), "device is not gfx90a");
    FA_ASM_REQUIRE(Q && K && V, "missing source");
    const int64_t D = Q->ne[0];
    FA_ASM_REQUIRE(K->ne[0] == D && V->ne[0] == D && dst->ne[0] == D, "head dims differ (K/V/dst vs Q)");
    FA_ASM_REQUIRE(D == FA_ASM_D || (D == FA_CK_D && fa_ck_hd256_enabled()),
                   "head dim not 128 (asm) or 256 (CK, compiled in and GGML_HIP_FA_CK_HD256 != 0)");
    FA_ASM_REQUIRE(D != FA_ASM_D || fa_asm_hd128_enabled(),
                   "hd128 bf16 asm off by default (PPL drift; set GGML_HIP_FA_ASM=1 to enable)");
    FA_ASM_REQUIRE(dst->type == GGML_TYPE_F32, "dst is not f32");
    FA_ASM_REQUIRE(Q->type == GGML_TYPE_F32 || Q->type == GGML_TYPE_F16, "Q type");
    if (D == FA_CK_D) {
        // the CK instances read K and V in place as fp16 with 16-byte vector loads along D
        FA_ASM_REQUIRE(K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16, "hd256: K/V not f16");
        FA_ASM_REQUIRE((uintptr_t) K->data % 16 == 0 && (uintptr_t) V->data % 16 == 0 &&
                       K->nb[1] % 16 == 0 && K->nb[2] % 16 == 0 && V->nb[1] % 16 == 0 && V->nb[2] % 16 == 0,
                       "hd256: K/V not 16-byte aligned");
    } else {
        FA_ASM_REQUIRE(fa_asm_kv_type_ok(K->type) && K->type == V->type, "K/V type");
    }
    FA_ASM_REQUIRE(Q->nb[0] == ggml_type_size(Q->type) && K->nb[0] == ggml_type_size(K->type) &&
                   V->nb[0] == ggml_type_size(V->type), "non-unit innermost stride");
    FA_ASM_REQUIRE(ggml_is_contiguous(dst), "dst not contiguous");
    FA_ASM_REQUIRE(Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1, "more than one sequence stream (ne[3] > 1)");
    FA_ASM_REQUIRE(K->ne[1] == V->ne[1] && K->ne[2] == V->ne[2], "K/V shape mismatch");
    FA_ASM_REQUIRE(K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0, "H % H_kv != 0");
    {
        const int64_t gqa = Q->ne[2] / K->ne[2];
        // ratios the AK-ASM-H5 correctness run and the fa_asm_gfx90a test set cover; widen only with test cases
        if (D == FA_CK_D) {
            FA_ASM_REQUIRE(gqa == 1 || gqa == 2 || gqa == 4 || gqa == 6 || gqa == 8 || gqa == 12,
                           "hd256: GQA ratio not in {1,2,4,6,8,12}");
        } else {
            FA_ASM_REQUIRE(gqa == 1 || gqa == 2 || gqa == 4 || gqa == 8, "GQA ratio not in {1,2,4,8}");
        }
    }
    FA_ASM_REQUIRE(Q->ne[1] >= fa_asm_min_batch(), "fewer query rows than GGML_HIP_FA_ASM_MIN_BATCH");
    FA_ASM_REQUIRE(sink == nullptr, "attention sinks");

    float max_bias = 0.0f;
    float softcap  = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    FA_ASM_REQUIRE(max_bias == 0.0f, "ALiBi (max_bias != 0)");
    FA_ASM_REQUIRE(softcap == 0.0f, "logit softcap");

    if (mask) {
        FA_ASM_REQUIRE(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask), "mask not contiguous f16");
        FA_ASM_REQUIRE(mask->ne[2] == 1 && mask->ne[3] == 1, "mask broadcast over heads/streams");
        FA_ASM_REQUIRE(mask->ne[0] >= K->ne[1] && mask->ne[1] >= Q->ne[1], "mask smaller than n_kv x n_q");
    }

    // the asm kernel addresses with 32-bit byte strides and buffer offsets: stay well inside 2^31. The CK instances use
    // 32-bit element offsets (ck_tile::index_t) over K/V read in place with the cache's own strides.
    const uint64_t q_bytes  = (uint64_t) Q->ne[1] * Q->ne[2] * D * 2;
    // CK: the byte extent actually addressed, (ne1-1)*nb1 + (ne2-1)*nb2 + one row. (The previous
    // ne1 * max(nb1, ne2*nb2) counted n_kv twice for a head-major [D, n_kv, H_kv] layout - nb2 = n_kv*nb1 - and
    // rejected every such case above ~2^31/(n_kv*H_kv*512): e.g. the test-file / default-permute 24/4 S=8192 cell.)
    const auto extent = [D](const ggml_tensor * t) {
        return (uint64_t) (t->ne[1] - 1) * t->nb[1] + (uint64_t) (t->ne[2] - 1) * t->nb[2] + (uint64_t) D * t->nb[0];
    };
    const uint64_t kv_bytes = D == FA_CK_D ? extent(K) : (uint64_t) K->ne[1] * K->ne[2] * D * 2;
    const uint64_t v_bytes  = D == FA_CK_D ? extent(V) : 0;
    FA_ASM_REQUIRE(q_bytes < (1ull << 31) && kv_bytes < (1ull << 31) && v_bytes < (1ull << 31),
                   "tensor too large for 32-bit kernel offsets");
#undef FA_ASM_REQUIRE

    *why = nullptr;
    return true;
}

void fa_asm_log_reject(const char * why, const ggml_tensor * dst) {
    // one line per distinct reason (the reason strings are literals), or every decision with GGML_HIP_FA_ASM_LOG=1
    static std::mutex mtx;
    static const char * seen[32] = {};
    static int n_seen = 0;
    bool first = false;
    {
        std::lock_guard<std::mutex> lock(mtx);
        bool found = false;
        for (int i = 0; i < n_seen; ++i) {
            found = found || seen[i] == why;
        }
        if (!found && n_seen < 32) {
            seen[n_seen++] = why;
            first = true;
        }
    }
    if (first || fa_asm_log_all()) {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        GGML_LOG_INFO("ggml_cuda_fa_asm: fallback (%s) node=%s Q=[%lld,%lld,%lld] K=[%lld,%lld,%lld] %s\n", why, dst->name,
                       (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2],
                       (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], ggml_type_name(K->type));
    }
}

// ---------------------------------------------------------------- device kernels

__device__ __forceinline__ uint16_t fa_asm_f32_to_bf16_rne(const float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((u >> 16) | 0x40u); // quiet NaN
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ float fa_asm_bf16_to_f32(const uint16_t b) {
    return __uint_as_float(((uint32_t) b) << 16);
}

// Per mask row r (one block per row): first[r] = index of the first column in [0, n_kv) that is not +-0, or n_kv.
// bad = 1 if the row holds any value other than +-0 / -INF, or a non -INF value at or after first[r].
static __global__ void fa_asm_mask_scan(const uint16_t * __restrict__ mask, const int64_t row_stride, const int n_kv,
                                        int * __restrict__ first, int * __restrict__ bad) {
    const int r = blockIdx.x;
    const uint16_t * row = mask + (int64_t) r * row_stride;

    __shared__ int s_first;
    if (threadIdx.x == 0) {
        s_first = n_kv;
    }
    __syncthreads();

    int my_first = n_kv;
    for (int c = threadIdx.x; c < n_kv; c += blockDim.x) {
        const uint16_t v = row[c];
        if ((v & 0x7fffu) != 0) {
            my_first = c;
            break;
        }
    }
    if (my_first < n_kv) {
        atomicMin(&s_first, my_first);
    }
    __syncthreads();

    const int f = s_first;
    bool ok = true;
    for (int c = f + threadIdx.x; c < n_kv; c += blockDim.x) {
        ok = ok && row[c] == 0xFC00u; // f16 -INF
    }
    if (!ok) {
        atomicOr(bad, 1);
    }
    if (threadIdx.x == 0) {
        first[r] = f;
    }
}

// Single block: res[0] = kind, res[1] = L.
//   causal: first[r] - r == c for all rows, c >= 1                       -> L = first[n_q-1]  (L >= n_q)
//   full:   first[r] == L for all rows, L >= 1 (also the n_q == 1 case)  -> L = first[0]
static __global__ void fa_asm_mask_reduce(const int * __restrict__ first, const int n_q, const int * __restrict__ bad,
                                          int * __restrict__ res) {
    __shared__ int s_not_full;
    __shared__ int s_not_causal;
    if (threadIdx.x == 0) {
        s_not_full   = 0;
        s_not_causal = 0;
    }
    __syncthreads();

    const int f0 = first[0];
    int nf = 0;
    int nc = 0;
    for (int r = threadIdx.x; r < n_q; r += blockDim.x) {
        nf |= first[r] != f0;
        nc |= first[r] - r != f0;
    }
    if (nf) { atomicOr(&s_not_full, 1); }
    if (nc) { atomicOr(&s_not_causal, 1); }
    __syncthreads();

    if (threadIdx.x == 0) {
        int kind = FA_ASM_MASK_UNSUPPORTED;
        int len  = 0;
        if (*bad == 0) {
            if (!s_not_full && f0 >= 1) {
                kind = FA_ASM_MASK_FULL;
                len  = f0;
            } else if (!s_not_causal && f0 >= 1) {
                kind = FA_ASM_MASK_CAUSAL;
                len  = first[n_q - 1];
            }
        }
        res[0] = kind;
        res[1] = len;
    }
}

// src [D=128, n_rows, n_heads] (byte strides nb1, nb2; f32, f16 or bf16) -> dst bf16 [n_rows][n_heads][128]
template <typename src_t>
static __global__ void fa_asm_to_bf16_bshd(const char * __restrict__ src, const int64_t nb1, const int64_t nb2,
                                           const int n_heads, const int64_t n, uint16_t * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int     d = i % FA_ASM_D;
    const int     h = (i / FA_ASM_D) % n_heads;
    const int64_t s = i / ((int64_t) FA_ASM_D * n_heads);
    const src_t * p = (const src_t *) (src + s*nb1 + h*nb2);
    if constexpr (std::is_same_v<src_t, float>) {
        dst[i] = fa_asm_f32_to_bf16_rne(p[d]);
    } else if constexpr (std::is_same_v<src_t, half>) {
        dst[i] = fa_asm_f32_to_bf16_rne(__half2float(p[d]));
    } else {
        dst[i] = ((const uint16_t *) p)[d]; // bf16 already
    }
}

// q8_0 [D=128 (4 blocks of 32), n_rows, n_heads] (byte strides nb1, nb2) -> dst bf16 [n_rows][n_heads][128]
static __global__ void fa_asm_q8_0_to_bf16_bshd(const char * __restrict__ src, const int64_t nb1, const int64_t nb2,
                                                const int n_heads, const int64_t n, uint16_t * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int     d = i % FA_ASM_D;
    const int     h = (i / FA_ASM_D) % n_heads;
    const int64_t s = i / ((int64_t) FA_ASM_D * n_heads);
    const block_q8_0 * b = (const block_q8_0 *) (src + s*nb1 + h*nb2) + d / QK8_0;
    const float scale = b->d;
    dst[i] = fa_asm_f32_to_bf16_rne(scale * (float) b->qs[d % QK8_0]);
}

static __global__ void fa_asm_o_to_f32(const uint16_t * __restrict__ src, const int64_t n, float * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = fa_asm_bf16_to_f32(src[i]);
    }
}

// AK-CK-HD256: Q [D=256, n_rows, n_heads] (byte strides nb1, nb2; f32 or f16) -> f16 [n_rows][n_heads][256] (RNE)
template <typename src_t>
static __global__ void fa_ck_q_to_f16_bshd(const char * __restrict__ src, const int64_t nb1, const int64_t nb2,
                                           const int n_heads, const int64_t n, half * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int     d = i % FA_CK_D;
    const int     h = (i / FA_CK_D) % n_heads;
    const int64_t s = i / ((int64_t) FA_CK_D * n_heads);
    const src_t * p = (const src_t *) (src + s*nb1 + h*nb2);
    if constexpr (std::is_same_v<src_t, float>) {
        dst[i] = __float2half_rn(p[d]);
    } else {
        dst[i] = p[d];
    }
}

static __global__ void fa_ck_o_f16_to_f32(const half * __restrict__ src, const int64_t n, float * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

constexpr int FA_ASM_CVT_BLOCK = 256;

int64_t fa_asm_cvt_blocks(const int64_t n) {
    return (n + FA_ASM_CVT_BLOCK - 1) / FA_ASM_CVT_BLOCK;
}

// first n_rows rows of t ([128, >= n_rows, n_heads]) -> bf16 BSHD
void fa_asm_convert(const ggml_tensor * t, const int64_t n_rows, uint16_t * dst, cudaStream_t stream) {
    const int     n_heads = (int) t->ne[2];
    const int64_t n       = n_rows * n_heads * FA_ASM_D;
    const dim3    grid((unsigned) fa_asm_cvt_blocks(n));
    const char *  src     = (const char *) t->data;
    switch (t->type) {
        case GGML_TYPE_F32:
            fa_asm_to_bf16_bshd<float><<<grid, FA_ASM_CVT_BLOCK, 0, stream>>>(src, t->nb[1], t->nb[2], n_heads, n, dst);
            break;
        case GGML_TYPE_F16:
            fa_asm_to_bf16_bshd<half><<<grid, FA_ASM_CVT_BLOCK, 0, stream>>>(src, t->nb[1], t->nb[2], n_heads, n, dst);
            break;
        case GGML_TYPE_BF16:
            fa_asm_to_bf16_bshd<uint16_t><<<grid, FA_ASM_CVT_BLOCK, 0, stream>>>(src, t->nb[1], t->nb[2], n_heads, n, dst);
            break;
        case GGML_TYPE_Q8_0:
            fa_asm_q8_0_to_bf16_bshd<<<grid, FA_ASM_CVT_BLOCK, 0, stream>>>(src, t->nb[1], t->nb[2], n_heads, n, dst);
            break;
        default:
            GGML_ABORT("fa_asm_convert: unsupported type %s", ggml_type_name(t->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

// Classify the mask (cached per graph compute and mask tensor). Returns kind; *len = effective KV length.
// allow_sync = false (inside stream capture): a cache miss returns FA_ASM_MASK_UNSUPPORTED (the caller falls back).
int fa_asm_classify_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * mask, const int64_t n_q, const int64_t n_kv,
                         int64_t * len, const bool allow_sync = true) {
    // Keyed on the tensor object as well as its data: a graph input is written once before the graph runs, while an
    // intermediate mask computed in-graph (e.g. a per-layer sparse mask) is a different tensor object per layer even
    // where the allocator reuses its memory.
    for (int i = 0; i < ggml_backend_cuda_context::FATTN_ASM_MASK_CACHE_N; ++i) {
        const ggml_cuda_fattn_asm_mask_cache & c = ctx.fattn_asm_mask_cache[i];
        if (c.epoch == ctx.graph_compute_epoch && c.tensor == mask && c.data == mask->data && c.n_q == n_q && c.n_kv == n_kv) {
            *len = c.len;
            return c.kind;
        }
    }
    if (!allow_sync) {
        *len = 0;
        return FA_ASM_MASK_UNSUPPORTED;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> first(ctx.pool(), n_q + 3); // [0, n_q) first, [n_q] bad, [n_q+1..n_q+2] result
    int * d_bad = first.ptr + n_q;
    int * d_res = first.ptr + n_q + 1;
    CUDA_CHECK(cudaMemsetAsync(d_bad, 0, sizeof(int), stream));

    const int64_t row_stride = mask->nb[1] / sizeof(uint16_t);
    fa_asm_mask_scan<<<(unsigned) n_q, 256, 0, stream>>>((const uint16_t *) mask->data, row_stride, (int) n_kv, first.ptr, d_bad);
    CUDA_CHECK(cudaGetLastError());
    fa_asm_mask_reduce<<<1, 256, 0, stream>>>(first.ptr, (int) n_q, d_bad, d_res);
    CUDA_CHECK(cudaGetLastError());

    int h_res[2] = { 0, 0 };
    CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, sizeof(h_res), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ggml_cuda_fattn_asm_mask_cache & c = ctx.fattn_asm_mask_cache[ctx.fattn_asm_mask_cache_next];
    ctx.fattn_asm_mask_cache_next = (ctx.fattn_asm_mask_cache_next + 1) % ggml_backend_cuda_context::FATTN_ASM_MASK_CACHE_N;
    c.epoch  = ctx.graph_compute_epoch;
    c.tensor = mask;
    c.data   = mask->data;
    c.n_q    = n_q;
    c.n_kv   = n_kv;
    c.kind   = h_res[0];
    c.len    = h_res[1];

    *len = c.len;
    return c.kind;
}

// fmha_fwd_v3_args (packed; p2 = 8 B pad after each pointer, p3 = 12 B pad after each u32/f32), batch mode.
struct fa_asm_kargs {
    unsigned char b[FA_ASM_KARGS];
    void u64(int off, uint64_t v) { memcpy(b + off, &v, 8); }
    void u32(int off, uint32_t v) { memcpy(b + off, &v, 4); }
    void f32(int off, float v)    { memcpy(b + off, &v, 4); }
};

} // namespace

bool ggml_cuda_fattn_asm_gfx90a_may_apply(int device, const ggml_tensor * dst) {
    const char * why = nullptr;
    if (!fa_asm_shape_ok(device, dst, &why)) {
        return false;
    }
    if (!fa_asm_graph_enabled()) {
        return true;
    }
    // a mask computed inside the graph cannot be classified before capture
    const ggml_tensor * mask = dst->src[3];
    return mask != nullptr && mask->op != GGML_OP_NONE;
}

uint64_t ggml_cuda_fattn_asm_gfx90a_graph_prepass(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    if (!fa_asm_graph_enabled()) {
        return 0;
    }
    uint64_t sig = 0;
    const auto mix = [&sig](uint64_t v) { sig = (sig ^ v) * 0x100000001b3ull; };
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        const char * why = nullptr;
        if (!fa_asm_shape_ok(ctx.device, node, &why)) {
            continue;
        }
        const ggml_tensor * mask = node->src[3];
        if (mask != nullptr && mask->op != GGML_OP_NONE) {
            continue; // graph not captured (may_apply)
        }
        if (sig == 0) {
            sig = 0xcbf29ce484222325ull;
        }
        const bool hd256  = node->src[0]->ne[0] == FA_CK_D;
        const bool loaded = hd256 || fa_asm_load(ctx.device);
        int     kind = FA_ASM_MASK_FULL;
        int64_t L    = node->src[1]->ne[1];
        if (mask != nullptr) {
            kind = fa_asm_classify_mask(ctx, mask, node->src[0]->ne[1], node->src[1]->ne[1], &L, true);
        }
        mix((uint64_t) i);
        mix((uint64_t) loaded);
        mix((uint64_t) kind);
        mix((uint64_t) L);
    }
    return sig;
}

uint64_t ggml_cuda_fattn_asm_gfx90a_n_calls() {
    return g_fa_asm_n_calls.load(std::memory_order_relaxed);
}

int64_t ggml_cuda_fattn_asm_gfx90a_min_batch() {
    return fa_asm_min_batch();
}

bool ggml_cuda_fattn_asm_gfx90a_usable(int device) {
    if (!fa_asm_enabled() || !fa_asm_hd128_enabled() || !fa_asm_device_ok(device)) {
        return false;
    }
    ggml_cuda_set_device(device);
    return fa_asm_load(device);
}

bool ggml_cuda_fattn_ck_hd256_usable(int device) {
    return fa_asm_enabled() && fa_ck_hd256_enabled() && fa_asm_device_ok(device);
}

uint64_t ggml_cuda_fattn_asm_gfx90a_n_captured() {
    return g_fa_asm_n_captured.load(std::memory_order_relaxed);
}

uint64_t ggml_cuda_fattn_ck_hd256_n_calls() {
    return g_fa_ck_n_calls.load(std::memory_order_relaxed);
}

#ifdef GGML_HIP_CK_FMHA_HD256
// AK-CK-HD256: Q -> f16 scratch, CK-tile fmha fwd (K/V in place), O f16 -> dst f32. Called after the mask verdict.
static bool fa_ck_hd256_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool causal, const int64_t L) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int64_t n_q  = Q->ne[1];
    const int64_t H    = Q->ne[2];
    const int64_t H_kv = K->ne[2];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<half> q_f16(pool, (size_t) n_q * H * FA_CK_D);
    ggml_cuda_pool_alloc<half> o_f16(pool, (size_t) n_q * H * FA_CK_D);

    {
        const int64_t n    = n_q * H * FA_CK_D;
        const dim3    grid((unsigned) fa_asm_cvt_blocks(n));
        const char *  src  = (const char *) Q->data;
        if (Q->type == GGML_TYPE_F32) {
            fa_ck_q_to_f16_bshd<float><<<grid, FA_ASM_CVT_BLOCK, 0, stream>>>(src, Q->nb[1], Q->nb[2], (int) H, n, q_f16.ptr);
        } else {
            fa_ck_q_to_f16_bshd<half><<<grid, FA_ASM_CVT_BLOCK, 0, stream>>>(src, Q->nb[1], Q->nb[2], (int) H, n, q_f16.ptr);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_ck_fmha_hd256_args a;
    a.q              = q_f16.ptr;
    a.k              = K->data;
    a.v              = V->data;
    a.o              = o_f16.ptr;
    a.seqlen_q       = (int) n_q;
    a.seqlen_k       = (int) L;
    a.nhead_q        = (int) H;
    a.nhead_k        = (int) H_kv;
    a.stride_q       = (int) (H * FA_CK_D);       // Q scratch BSHD
    a.nhead_stride_q = FA_CK_D;
    a.stride_k       = (int) (K->nb[1] / sizeof(half));
    a.nhead_stride_k = (int) (K->nb[2] / sizeof(half));
    a.stride_v       = (int) (V->nb[1] / sizeof(half));
    a.nhead_stride_v = (int) (V->nb[2] / sizeof(half));
    a.stride_o       = (int) (H * FA_CK_D);       // O scratch BSHD == dst element order [D, H, n_q]
    a.nhead_stride_o = FA_CK_D;
    a.scale          = scale;
    a.causal         = causal ? 1 : 0;

    const hipError_t err = ggml_ck_fmha_fwd_hd256_f16(a, stream);
    if (err != hipSuccess) {
        // only the Q scratch has been written: fall back
        GGML_LOG_WARN("ggml_cuda_fa_ck: CK fmha fwd hd256 launch failed: %s\n", hipGetErrorString(err));
        (void) hipGetLastError();
        fa_asm_log_reject("CK hd256 launch failed", dst);
        return false;
    }

    const int64_t n_out = n_q * H * FA_CK_D;
    fa_ck_o_f16_to_f32<<<(unsigned) fa_asm_cvt_blocks(n_out), FA_ASM_CVT_BLOCK, 0, stream>>>(o_f16.ptr, n_out, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());

    g_fa_asm_n_calls.fetch_add(1, std::memory_order_relaxed);
    const uint64_t n = g_fa_ck_n_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1 || fa_asm_log_all()) {
        GGML_LOG_INFO("ggml_cuda_fa_ck: gfx90a CK-tile hd256 fp16 path ran (call %llu) node=%s %s n_q=%lld H=%lld H_kv=%lld n_kv=%lld L=%lld [%s]\n",
                      (unsigned long long) n, dst->name, causal ? "causal" : "full", (long long) n_q, (long long) H,
                      (long long) H_kv, (long long) K->ne[1], (long long) L, ggml_ck_fmha_fwd_hd256_f16_name(a.causal));
    }
    return true;
}
#endif // GGML_HIP_CK_FMHA_HD256

bool ggml_cuda_fattn_asm_gfx90a_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int device = ctx.device;

    const char * why = nullptr;
    if (!fa_asm_shape_ok(device, dst, &why)) {
        // the common non-candidates (decode, other archs, other head sizes) are not worth a log line
        if (fa_asm_log_all()) {
            fa_asm_log_reject(why, dst);
        }
        return false;
    }

    cudaStream_t stream = ctx.stream();
    bool capturing = false;
    {
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &capture) != hipSuccess) {
            (void) hipGetLastError();
            fa_asm_log_reject("stream capture status unknown", dst);
            return false;
        }
        capturing = capture != hipStreamCaptureStatusNone;
        if (capturing && !fa_asm_graph_enabled()) {
            fa_asm_log_reject("stream is capturing (GGML_HIP_FA_ASM_GRAPH=0)", dst);
            return false;
        }
    }

    const bool is_hd256 = dst->src[0]->ne[0] == FA_CK_D;
    if (!is_hd256 && capturing && !g_fa_asm_modules[device].done.load(std::memory_order_acquire)) {
        // hipModuleLoadData is not legal inside capture; the pre-pass should have loaded them
        fa_asm_log_reject("capturing before the code objects were loaded", dst);
        return false;
    }
    if (!is_hd256 && !fa_asm_load(device)) {
        fa_asm_log_reject("code objects failed to load", dst);
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t n_q  = Q->ne[1];
    const int64_t H    = Q->ne[2];
    const int64_t H_kv = K->ne[2];
    const int64_t n_kv = K->ne[1];

    int     kind = FA_ASM_MASK_FULL;
    int64_t L    = n_kv;
    if (mask) {
        // inside capture only the pre-pass verdict may be used (no stream sync); a miss falls back to the
        // existing (capture-safe) kernels, which are correct for any mask
        kind = fa_asm_classify_mask(ctx, mask, n_q, n_kv, &L, !capturing);
    }
    if (kind == FA_ASM_MASK_UNSUPPORTED || L < 1 || L > n_kv || (kind == FA_ASM_MASK_CAUSAL && L < n_q)) {
        fa_asm_log_reject("mask is not a (padded) causal or full mask", dst);
        return false;
    }
    const bool causal = kind == FA_ASM_MASK_CAUSAL;

    if (is_hd256) {
#ifdef GGML_HIP_CK_FMHA_HD256
        const bool ran = fa_ck_hd256_run(ctx, dst, causal, L);
        if (ran && capturing) {
            fa_asm_note_captured(dst, true);
        }
        return ran;
#else
        GGML_ABORT("unreachable: hd256 admitted without GGML_HIP_CK_FMHA_HD256"); // fa_ck_hd256_enabled() is false
#endif
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // ---- scratch (stream-ordered pool), bf16 BSHD, B = 1
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<uint16_t> q_bf16(pool, (size_t) n_q * H * FA_ASM_D);
    ggml_cuda_pool_alloc<uint16_t> k_bf16(pool, (size_t) L * H_kv * FA_ASM_D);
    ggml_cuda_pool_alloc<uint16_t> v_bf16(pool, (size_t) L * H_kv * FA_ASM_D);
    ggml_cuda_pool_alloc<uint16_t> o_bf16(pool, (size_t) n_q * H * FA_ASM_D);
    ggml_cuda_pool_alloc<float>    lse(pool, (size_t) H * n_q);

    fa_asm_convert(Q, n_q, q_bf16.ptr, stream);
    fa_asm_convert(K, L,   k_bf16.ptr, stream);
    fa_asm_convert(V, L,   v_bf16.ptr, stream);

    // ---- kernargs (strides in bytes; bpe 2 for Q/K/V/O, 4 for LSE)
    const uint32_t q_seq = (uint32_t) (H    * FA_ASM_D * 2);   // BSHD: next query row
    const uint32_t q_hd  = (uint32_t) (FA_ASM_D * 2);          // next head
    const uint32_t q_bat = (uint32_t) (n_q  * H * FA_ASM_D * 2);
    const uint32_t k_seq = (uint32_t) (H_kv * FA_ASM_D * 2);
    const uint32_t k_hd  = (uint32_t) (FA_ASM_D * 2);
    const uint32_t k_bat = (uint32_t) (L    * H_kv * FA_ASM_D * 2);

    uint32_t tune_opt = 5;
    if (causal && ((H % 8 != 0) || (n_q > 16384))) {
        tune_opt -= 2; // init_fmha_fwd_v3_args: downgrade when nhead_q % 8 != 0 or seqlen_q > 16K
    }

    fa_asm_kargs a;
    memset(a.b, 0, sizeof(a.b));
    a.u64(  0, (uint64_t) (uintptr_t) o_bf16.ptr);  // ptr_o
    a.u64( 16, (uint64_t) (uintptr_t) q_bf16.ptr);  // ptr_q
    a.u64( 32, (uint64_t) (uintptr_t) k_bf16.ptr);  // ptr_k
    a.u64( 48, (uint64_t) (uintptr_t) v_bf16.ptr);  // ptr_v
    a.u64( 64, (uint64_t) (uintptr_t) lse.ptr);     // ptr_lse
    a.f32( 80, scale);                               // scalar (softmax scale)
    a.u32( 96, (uint32_t) n_q);                      // s_seq_len
    a.u32(112, q_seq);                               // s_Seqs
    a.u32(128, (uint32_t) FA_ASM_TS_QO * q_seq);     // s_Ts
    a.u32(144, q_hd);                                // s_Hs
    a.u32(160, q_bat);                               // s_Bs
    a.u32(176, (uint32_t) (H / H_kv));               // s_gqa
    a.u32(192, k_seq);                               // s_k_Seqs
    a.u32(208, k_hd);                                // s_k_Hs
    a.u32(224, k_bat);                               // s_k_Bs
    a.u32(240, tune_opt);                            // s_opt
    a.u32(256, 1);                                   // s_lse: write LSE (the configuration the harness verified)
    a.u32(272, (uint32_t) L);                        // s_kv_seq_len
    a.u32(288, FA_ASM_D);                            // s_qk_head_dim
    a.u32(304, FA_ASM_D);                            // s_v_head_dim
    a.u32(320, (uint32_t) H);                        // s_q_head_num
    a.u32(336, k_seq);                               // s_v_Seqs (V scratch has K's layout)
    a.u32(352, k_hd);                                // s_v_Hs
    a.u32(368, k_bat);                               // s_v_Bs
    a.u32(384, q_seq);                               // s_o_Seqs (O scratch has Q's layout)
    a.u32(400, q_hd);                                // s_o_Hs
    a.u32(416, q_bat);                               // s_o_Bs
    a.u64(432, 0);                                   // ptr_qseq (group mode only)
    a.u64(448, 0);                                   // ptr_kseq (group mode only)
    a.u32(464, (uint32_t) (n_q * 4));                // s_lse_Hs: LSE [B,H,Sq] fp32
    a.u64(480, 0);                                   // ptr_qseq_padding (group mode only)
    a.u64(496, 0);                                   // ptr_kseq_padding (group mode only)

    const int      tg_div = causal ? 2 : 1;          // causal: head and tail query tiles merged into one workgroup
    const unsigned gdx    = (unsigned) (((n_q + FA_ASM_TS_QO - 1) / FA_ASM_TS_QO + tg_div - 1) / tg_div);
    const unsigned gdy    = (unsigned) H;

    // Kernargs by value: HIP copies the HIP_LAUNCH_PARAM_BUFFER_POINTER blob at launch / at capture into the graph
    // node (AK-ASM-T0 verified replay x7 and hipGraphExecUpdate with a stack buffer that is gone by replay time), so
    // the stack-local blob need not outlive this call. Every value in it is fixed by the node's tensor properties plus
    // the mask verdict (kind, L), which the graph signature (graph_prepass) covers; the scratch pointers come from the
    // pool, whose addresses ggml's graph mode already relies on being stable across replays.
    size_t arg_size = FA_ASM_KARGS;
    void * config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, a.b,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END,
    };
    const hipFunction_t fn = g_fa_asm_modules[device].fn[causal ? 1 : 0];
    {
        const hipError_t err = hipModuleLaunchKernel(fn, gdx, gdy, 1, FA_ASM_BLOCK, 1, 1, 0, stream, nullptr, config);
        if (err != hipSuccess) {
            // nothing has touched dst yet (the conversions only wrote scratch): fall back
            GGML_LOG_WARN("ggml_cuda_fa_asm: hipModuleLaunchKernel failed: %s\n", hipGetErrorString(err));
            (void) hipGetLastError();
            fa_asm_log_reject("hipModuleLaunchKernel failed", dst);
            return false;
        }
    }

    // ---- O (bf16 BSHD) -> dst (f32 [D, H, n_q, 1], the same element order)
    const int64_t n_out = n_q * H * FA_ASM_D;
    fa_asm_o_to_f32<<<(unsigned) fa_asm_cvt_blocks(n_out), FA_ASM_CVT_BLOCK, 0, stream>>>(o_bf16.ptr, n_out, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());

    if (capturing) {
        fa_asm_note_captured(dst, false);
    }
    const uint64_t n = g_fa_asm_n_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1 || fa_asm_log_all()) {
        GGML_LOG_INFO("ggml_cuda_fa_asm: gfx90a FA-v3 asm path ran (call %llu) node=%s %s n_q=%lld H=%lld H_kv=%lld n_kv=%lld L=%lld K=%s\n",
                      (unsigned long long) n, dst->name, causal ? "causal" : "full", (long long) n_q, (long long) H,
                      (long long) H_kv, (long long) n_kv, (long long) L, ggml_type_name(K->type));
    }
    return true;
}

#endif // GGML_CUDA_FA_ASM_GFX90A_AVAILABLE
