// AK-ASM-H11: topk-moe fusion fast path on gfx90a via the retargeted AITER topksoftmax asm kernels.
//
// What the kernel computes (AITER csrc/py_itfs_cu/asm_topksoftmax.cu, hsa/gfx942/topksoftmax/topksoftmax.csv):
//   for every token row t of fp32 logits [T, E]: p = softmax(logits[t]); the k largest p by value -> ids[t, 0:k]
//   (int32) and weights[t, 0:k] (fp32), optionally renormalized to sum 1. Rows are written at t * out_stride bytes
//   for BOTH outputs. Kernarg blob: 8 x 16-byte slots (pointer + p2 / uint32 + p3), 128 bytes, matching the
//   code objects' .kernarg_segment_size. Block 256 (4 wave64), grid ceil(T / 4) ("4x" = 4 tokens per workgroup).
//
// ggml's fused topk-moe node (ggml-cuda.cu ggml_cuda_topk_moe_fusion, softmax branch) maps onto it when:
//   - no prob bias, no scale (or scale == 1), not sigmoid / sqrt-softplus / delayed softmax;
//   - with norm: ggml divides by clamp(sum, clamp_min). The top-k sum of a softmax is >= k / E, so the clamp is
//     inactive whenever clamp_min <= k / E (llama.cpp uses 6.1e-5) and the asm renormalize is the same op.
//   - ids is the argsort node viewed to k columns: row stride E * 4, while weights is contiguous (row stride k * 4).
//     One row (decode): the stride is never used, both are written in place. More rows: weights in place, ids into a
//     pool scratch [T, k] then one strided D2D copy into the argsort rows.
//
// Env:
//   GGML_HIP_TOPK_ASM=0       disable the fast path at run time (default: enabled when compiled in)
//   GGML_HIP_TOPK_ASM_LOG=1   log every fast-path call (default: one line at first use, one per new reject reason)
//
// Graph capture: hipModuleLaunchKernel is captured and replayed correctly on ROCm 6.2 (AK-ASM-T0). Code objects are
// loaded lazily outside capture (graph warmup runs the graph uncaptured first); while capturing with nothing loaded
// the call falls back.

#include "topk-moe.cuh"
#include "topk-moe-asm-gfx90a.cuh"

#ifdef GGML_CUDA_TOPK_ASM_GFX90A_AVAILABLE

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

extern "C" {
#define TOPK_ASM_BLOB(v)                                   \
    extern const unsigned char ggml_hip_topk_asm_##v[];   \
    extern const char          ggml_hip_topk_asm_##v##_sha256[];
TOPK_ASM_BLOB(4x128x4)
TOPK_ASM_BLOB(4x128x6)
TOPK_ASM_BLOB(4x128x8)
TOPK_ASM_BLOB(4x256x6)
TOPK_ASM_BLOB(4x256x8)
TOPK_ASM_BLOB(4x384x8)
TOPK_ASM_BLOB(4x640x8)
#undef TOPK_ASM_BLOB
}

namespace {

constexpr int    TOPK_ASM_SUBM  = 4;    // tokens per workgroup of the "4x" kernels
constexpr int    TOPK_ASM_BLOCK = 256;  // asm_topksoftmax.cu: bdx = 256 (4 wv64)
constexpr size_t TOPK_ASM_KARGS = 128;  // .kernarg_segment_size

struct topk_asm_variant {
    int                   n_expert;
    int                   k;
    const char *          sym;
    const unsigned char * image;
    const char *          sha;
};

#define TOPK_ASM_V(E, K) { E, K, "_ZN5aiter19topksoftmax_4x" #E "x" #K "E", \
                           ggml_hip_topk_asm_4x##E##x##K, ggml_hip_topk_asm_4x##E##x##K##_sha256 }
const topk_asm_variant k_variants[] = {
    TOPK_ASM_V(128, 4), TOPK_ASM_V(128, 6), TOPK_ASM_V(128, 8), TOPK_ASM_V(256, 6),
    TOPK_ASM_V(256, 8), TOPK_ASM_V(384, 8), TOPK_ASM_V(640, 8),
};
#undef TOPK_ASM_V
constexpr int N_VARIANTS = (int) (sizeof(k_variants) / sizeof(k_variants[0]));

struct __attribute__((packed)) topk_asm_args {
    void *   ptr_T;  uint64_t _p0;   // ids    int32 [T, k] at out_stride
    void *   ptr_W;  uint64_t _p1;   // weights fp32 [T, k] at out_stride
    const void * ptr_A; uint64_t _p2; // logits fp32 [T, E] contiguous
    uint32_t batch;       uint32_t _p3[3];
    uint32_t expert;      uint32_t _p4[3];
    uint32_t topk;        uint32_t _p5[3];
    uint32_t renormalize; uint32_t _p6[3];
    uint32_t out_stride;  uint32_t _p7[3];  // bytes
};
static_assert(sizeof(topk_asm_args) == TOPK_ASM_KARGS, "topksoftmax kernarg size");

bool topk_asm_env_flag(const char * name, bool def) {
    const char * e = getenv(name);
    return e ? atoi(e) != 0 : def;
}

bool topk_asm_enabled() {
    static const bool v = topk_asm_env_flag("GGML_HIP_TOPK_ASM", true);
    return v;
}

bool topk_asm_log_all() {
    static const bool v = topk_asm_env_flag("GGML_HIP_TOPK_ASM_LOG", false);
    return v;
}

std::atomic<uint64_t> g_topk_asm_n_calls{0};

struct topk_asm_module {
    std::once_flag    once;
    std::atomic<bool> done{false};
    bool              ok = false;
    hipModule_t       mod[N_VARIANTS] = {};
    hipFunction_t     fn[N_VARIANTS]  = {};
};

topk_asm_module g_modules[GGML_CUDA_MAX_DEVICES];

bool topk_asm_load(int device) {
    topk_asm_module & m = g_modules[device];
    std::call_once(m.once, [&] {
        bool ok = true;
        for (int i = 0; i < N_VARIANTS && ok; ++i) {
            hipError_t err = hipModuleLoadData(&m.mod[i], k_variants[i].image);
            if (err == hipSuccess) {
                err = hipModuleGetFunction(&m.fn[i], m.mod[i], k_variants[i].sym);
            }
            if (err != hipSuccess) {
                GGML_LOG_WARN("%s: device %d: loading %s failed: %s; topk-moe keeps the existing kernel\n",
                              __func__, device, k_variants[i].sym, hipGetErrorString(err));
                (void) hipGetLastError();
                ok = false;
            }
        }
        if (!ok) {
            for (int i = 0; i < N_VARIANTS; ++i) {
                if (m.mod[i] != nullptr) {
                    (void) hipModuleUnload(m.mod[i]);
                }
                m.mod[i] = nullptr;
                m.fn[i]  = nullptr;
            }
        } else {
            GGML_LOG_INFO("%s: device %d: loaded %d topksoftmax code objects (4x256x8 sha256 %s)\n",
                          __func__, device, N_VARIANTS, ggml_hip_topk_asm_4x256x8_sha256);
        }
        m.ok = ok;
        m.done.store(true, std::memory_order_release);
    });
    return m.ok;
}

void topk_asm_log_reject(const char * why, const ggml_tensor * logits, int k) {
    static std::mutex    mtx;
    static const char *  seen[16] = {};
    static int           n_seen   = 0;
    std::lock_guard<std::mutex> lock(mtx);
    if (!topk_asm_log_all()) {
        for (int i = 0; i < n_seen; ++i) {
            if (seen[i] == why) {
                return;
            }
        }
        if (n_seen < 16) {
            seen[n_seen++] = why;
        }
    }
    GGML_LOG_INFO("ggml_cuda_topk_asm: fallback (%s) logits=[%lld,%lld] k=%d\n", why, (long long) logits->ne[0],
                  (long long) logits->ne[1], k);
}

} // namespace

uint64_t ggml_cuda_topk_moe_asm_gfx90a_n_calls() {
    return g_topk_asm_n_calls.load(std::memory_order_relaxed);
}

bool ggml_cuda_topk_moe_asm_gfx90a_try(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor *         logits,
                                       ggml_tensor *               weights,
                                       ggml_tensor *               ids,
                                       const ggml_tensor *         clamp,
                                       const ggml_tensor *         scale,
                                       const ggml_tensor *         bias,
                                       const ggml_cuda_topk_moe_args & args) {
    if (!topk_asm_enabled()) {
        return false;
    }
    const int device = ctx.device;
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_CDNA2) {
        return false;
    }

    const int64_t n_expert = logits->ne[0];
    const int64_t n_rows   = logits->ne[1];
    const int     k        = (int) weights->ne[1];

#define TOPK_ASM_REQUIRE(cond, why) do { if (!(cond)) { topk_asm_log_reject(why, logits, k); return false; } } while (0)
    TOPK_ASM_REQUIRE(args.softmax && !args.sigmoid && !args.sqrt_softplus && !args.delayed_softmax, "gating not plain softmax");
    TOPK_ASM_REQUIRE(!args.prob_bias && bias == nullptr, "prob bias");
    TOPK_ASM_REQUIRE(scale == nullptr || ggml_get_op_params_f32(scale, 0) == 1.0f, "weight scale != 1");
    TOPK_ASM_REQUIRE(logits->type == GGML_TYPE_F32 && ggml_is_contiguous(logits), "logits not contiguous f32");
    TOPK_ASM_REQUIRE(ggml_nrows(logits) == n_rows, "logits not 2-D");
    TOPK_ASM_REQUIRE(weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights) &&
                     ggml_nelements(weights) == (int64_t) k * n_rows, "weights not contiguous f32 [k, rows]");
    TOPK_ASM_REQUIRE(ids->nb[0] == sizeof(int32_t) && ids->nb[1] == (size_t) n_expert * sizeof(int32_t), "ids layout");
    TOPK_ASM_REQUIRE(n_rows > 0 && n_rows < (1ll << 28), "row count");
    if (clamp != nullptr) {
        TOPK_ASM_REQUIRE(ggml_get_op_params_f32(clamp, 0) <= (float) k / (float) n_expert, "norm clamp may be active");
    }
    int vi = -1;
    for (int i = 0; i < N_VARIANTS; ++i) {
        if (k_variants[i].n_expert == n_expert && k_variants[i].k == k) {
            vi = i;
        }
    }
    TOPK_ASM_REQUIRE(vi >= 0, "no code object for (n_expert, k)");

    cudaStream_t stream = ctx.stream();
    topk_asm_module & m = g_modules[device];
    if (!m.done.load(std::memory_order_acquire)) {
        hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
        TOPK_ASM_REQUIRE(hipStreamIsCapturing(stream, &st) == hipSuccess && st == hipStreamCaptureStatusNone,
                         "code objects not loaded yet and the stream is capturing");
    }
    TOPK_ASM_REQUIRE(topk_asm_load(device), "code objects failed to load");
#undef TOPK_ASM_REQUIRE

    // one row: write ids in place (the stride is never applied); more rows: ids via scratch + strided copy
    ggml_cuda_pool_alloc<int32_t> ids_tmp(ctx.pool());
    int32_t * ids_out = (int32_t *) ids->data;
    if (n_rows > 1) {
        ids_out = ids_tmp.alloc(n_rows * k);
    }

    topk_asm_args a;
    memset(&a, 0, sizeof(a));
    a.ptr_T       = ids_out;
    a.ptr_W       = weights->data;
    a.ptr_A       = logits->data;
    a.batch       = (uint32_t) n_rows;
    a.expert      = (uint32_t) n_expert;
    a.topk        = (uint32_t) k;
    a.renormalize = clamp != nullptr ? 1u : 0u;
    a.out_stride  = (uint32_t) (k * sizeof(float));

    size_t arg_size = TOPK_ASM_KARGS;
    void * config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &a,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
        HIP_LAUNCH_PARAM_END,
    };
    const unsigned gdx = (unsigned) ((n_rows + TOPK_ASM_SUBM - 1) / TOPK_ASM_SUBM);
    const hipError_t err = hipModuleLaunchKernel(m.fn[vi], gdx, 1, 1, TOPK_ASM_BLOCK, 1, 1, 0, stream, nullptr, config);
    if (err != hipSuccess) {
        GGML_LOG_WARN("ggml_cuda_topk_asm: hipModuleLaunchKernel failed: %s\n", hipGetErrorString(err));
        (void) hipGetLastError();
        return false; // nothing written yet
    }
    if (n_rows > 1) {
        CUDA_CHECK(cudaMemcpy2DAsync(ids->data, ids->nb[1], ids_out, k * sizeof(int32_t), k * sizeof(int32_t),
                                     (size_t) n_rows, cudaMemcpyDeviceToDevice, stream));
    }

    const uint64_t n = g_topk_asm_n_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1 || topk_asm_log_all()) {
        GGML_LOG_INFO("ggml_cuda_topk_asm: gfx90a topksoftmax asm path ran (call %llu) %s rows=%lld norm=%d\n",
                      (unsigned long long) n, k_variants[vi].sym, (long long) n_rows, clamp != nullptr ? 1 : 0);
    }
    return true;
}

#endif // GGML_CUDA_TOPK_ASM_GFX90A_AVAILABLE
