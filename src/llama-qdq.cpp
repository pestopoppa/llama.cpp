#include "llama-qdq.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

// See llama-qdq.h for the normative formulas. Everything below is f32, as the reference kernels'
// compute_dtype.

static inline uint32_t qdq_f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

static inline float qdq_bits_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// kernel.py fast_log2_ceil: exponent field - 127, +1 if any mantissa bit is set. Reproduced from
// the bits on purpose, including its value for f32 subnormals (exponent field 0 -> -126/-127).
static inline int32_t qdq_fast_log2_ceil(float x) {
    const uint32_t b   = qdq_f32_bits(x);
    const int32_t  ex  = (int32_t) ((b >> 23) & 0xFF);
    const uint32_t man = b & ((1u << 23) - 1);
    return ex - 127 + (man != 0 ? 1 : 0);
}

// kernel.py fast_pow2: (k + 127) << 23 reinterpreted as f32
static inline float qdq_fast_pow2(int32_t k) {
    return qdq_bits_f32((uint32_t) (k + 127) << 23);
}

// floor(log2(|a|)) for a normal f32, read from the exponent field; f32 zero/subnormal -> -127
static inline int32_t qdq_floor_log2(float a) {
    return (int32_t) ((qdq_f32_bits(a) >> 23) & 0xFF) - 127;
}

// Round to nearest, ties to even, onto a float grid with `mbits` mantissa bits whose smallest
// normal exponent is `emin` (below it the quantum stays 2^(emin - mbits): the subnormals). The
// scale v/q and back are exact (q is a power of two inside the f32 normal range), so the only
// rounding is std::nearbyint, which is RNE in the default floating-point environment.
static inline float qdq_round_grid(float v, int32_t mbits, int32_t emin) {
    const int32_t e = std::max(qdq_floor_log2(std::fabs(v)), emin);
    const float   q = qdq_fast_pow2(e - mbits); // exact 2^(e - mbits), inside the f32 normal range
    return std::nearbyint(v / q) * q;
}

float llama_qdq_round_e4m3(float v) {
    return qdq_round_grid(v, 3, -6);
}

float llama_qdq_round_e2m1(float v) {
    return qdq_round_grid(v, 1, 0);
}

int64_t llama_qdq_block_size(enum llama_qdq_mode mode) {
    switch (mode) {
        case LLAMA_QDQ_FP8_E4M3_UE8M0_B32: return 32;
        case LLAMA_QDQ_FP4_E2M1_UE8M0_B32: return 32;
        case LLAMA_QDQ_FP4_E2M1_E4M3_B16:  return 16;
        case LLAMA_QDQ_NONE:               return 0;
    }
    return 0;
}

const char * llama_qdq_mode_name(enum llama_qdq_mode mode) {
    switch (mode) {
        case LLAMA_QDQ_FP8_E4M3_UE8M0_B32: return "fp8_e4m3_ue8m0_b32";
        case LLAMA_QDQ_FP4_E2M1_UE8M0_B32: return "fp4_e2m1_ue8m0_b32";
        case LLAMA_QDQ_FP4_E2M1_E4M3_B16:  return "fp4_e2m1_e4m3_b16";
        case LLAMA_QDQ_NONE:               return "none";
    }
    return "?";
}

static void qdq_block(enum llama_qdq_mode mode, const float * x, float * y, int64_t B) {
    float amax = 0.0f;
    for (int64_t i = 0; i < B; ++i) {
        amax = std::max(amax, std::fabs(x[i]));
    }

    switch (mode) {
        case LLAMA_QDQ_FP8_E4M3_UE8M0_B32: {
            // act_quant_kernel: fp8_max = 448, amax floor 1e-4, round_scale
            const float fp8_max     = 448.0f;
            const float fp8_max_inv = (float) (1.0 / 448.0);
            amax = std::max(amax, 1e-4f);
            const float s = qdq_fast_pow2(qdq_fast_log2_ceil(amax * fp8_max_inv));
            for (int64_t i = 0; i < B; ++i) {
                const float t = std::min(std::max(x[i] / s, -fp8_max), fp8_max);
                y[i] = llama_qdq_round_e4m3(t) * s;
            }
        } break;
        case LLAMA_QDQ_FP4_E2M1_UE8M0_B32: {
            // fp4_quant_kernel, E8M0 branch: amax floor 6 * 2^-126, fast_round_scale(amax, 1/6)
            const float fp4_max     = 6.0f;
            const float fp4_max_inv = (float) (1.0 / 6.0);
            amax = std::max(amax, std::ldexp(6.0f, -126));
            const float s = qdq_fast_pow2(qdq_fast_log2_ceil(amax * fp4_max_inv));
            for (int64_t i = 0; i < B; ++i) {
                const float t = std::min(std::max(x[i] / s, -fp4_max), fp4_max);
                y[i] = llama_qdq_round_e2m1(t) * s;
            }
        } break;
        case LLAMA_QDQ_FP4_E2M1_E4M3_B16: {
            // fp4_quant_kernel, FP8 branch: amax floor 6 * 2^-9, s = e4m3(amax / 6) (satfinite)
            const float fp4_max = 6.0f;
            amax = std::max(amax, std::ldexp(6.0f, -9));
            const float s = llama_qdq_round_e4m3(std::min(amax / fp4_max, 448.0f));
            for (int64_t i = 0; i < B; ++i) {
                const float t = std::min(std::max(x[i] / s, -fp4_max), fp4_max);
                y[i] = llama_qdq_round_e2m1(t) * s;
            }
        } break;
        case LLAMA_QDQ_NONE: {
            if (y != x) {
                std::memcpy(y, x, (size_t) B * sizeof(float));
            }
        } break;
    }
}

void llama_qdq_ref(enum llama_qdq_mode mode, const float * x, float * y, int64_t n) {
    const int64_t B = mode == LLAMA_QDQ_NONE ? n : llama_qdq_block_size(mode);
    GGML_ASSERT(B > 0 && n % B == 0);
    for (int64_t i = 0; i < n; i += B) {
        qdq_block(mode, x + i, y + i, B);
    }
}

// one userdata slot per mode, static storage so the pointer outlives every graph (never written)
static enum llama_qdq_mode qdq_mode_slots[] = {
    LLAMA_QDQ_NONE,
    LLAMA_QDQ_FP8_E4M3_UE8M0_B32,
    LLAMA_QDQ_FP4_E2M1_UE8M0_B32,
    LLAMA_QDQ_FP4_E2M1_E4M3_B16,
};

static void qdq_custom_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    const enum llama_qdq_mode mode = *(const enum llama_qdq_mode *) userdata;
    const int64_t B = llama_qdq_block_size(mode);

    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(dst));
    GGML_ASSERT(B > 0 && a->ne[0] % B == 0);
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(dst));

    // blocks are independent, so any split over threads gives the same bits
    const int64_t n_blocks = ggml_nelements(a) / B;
    const int64_t per      = (n_blocks + nth - 1) / nth;
    const int64_t b0       = std::min<int64_t>(n_blocks, (int64_t) ith * per);
    const int64_t b1       = std::min<int64_t>(n_blocks, b0 + per);

    const float * x = (const float *) a->data;
    float       * y = (float       *) dst->data;
    for (int64_t b = b0; b < b1; ++b) {
        qdq_block(mode, x + b*B, y + b*B, B);
    }
}

ggml_tensor * llama_build_qdq(ggml_context * ctx, ggml_tensor * x, enum llama_qdq_mode mode) {
    if (mode == LLAMA_QDQ_NONE) {
        return x;
    }

    const int64_t B = llama_qdq_block_size(mode);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && "llama_build_qdq: F32 activations only");
    GGML_ASSERT(B > 0 && x->ne[0] % B == 0 && "llama_build_qdq: ne[0] must be a multiple of the block");

    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }

    return ggml_map_custom1(ctx, x, qdq_custom_op, GGML_N_TASKS_MAX, (void *) &qdq_mode_slots[mode]);
}

//
// DS41 site switches
//

static std::string qdq_trim_lower(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) {
        --e;
    }
    std::string r = s.substr(b, e - b);
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return r;
}

bool llama_dsv41_qdq_parse(const char * spec, llama_dsv41_qdq_sites & out, std::string * bad_token) {
    out = llama_dsv41_qdq_sites();
    if (spec == nullptr) {
        return true;
    }

    const std::string s(spec);
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string tok = qdq_trim_lower(s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));

        if (tok.empty() || tok == "none" || tok == "0") {
            // nothing
        } else if (tok == "wo_a") {
            out.wo_a = true;
        } else if (tok == "kv_win") {
            out.kv_win = true;
        } else if (tok == "kv_comp") {
            out.kv_comp = true;
        } else if (tok == "idx_k") {
            out.idx_k = true;
        } else if (tok == "idx_q") {
            out.idx_q = true;
        } else if (tok == "kv") {
            out.kv_win = out.kv_comp = out.idx_k = out.idx_q = true;
        } else if (tok == "all") {
            out.wo_a = out.kv_win = out.kv_comp = out.idx_k = out.idx_q = true;
        } else {
            if (bad_token) {
                *bad_token = tok;
            }
            out = llama_dsv41_qdq_sites();
            return false;
        }

        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }

    return true;
}

const llama_dsv41_qdq_sites & llama_dsv41_qdq_get() {
    static const llama_dsv41_qdq_sites sites = []() {
        llama_dsv41_qdq_sites r;
        const char * env = std::getenv("LLAMA_DSV41_QDQ");
        std::string bad;
        if (!llama_dsv41_qdq_parse(env, r, &bad)) {
            GGML_ABORT("LLAMA_DSV41_QDQ: unknown site '%s' (valid: wo_a, kv_win, kv_comp, idx_k, idx_q, kv, all, none)",
                    bad.c_str());
        }
        if (env != nullptr) {
            LLAMA_LOG_INFO("llama_dsv41_qdq: sites = wo_a:%d kv_win:%d kv_comp:%d idx_k:%d idx_q:%d (LLAMA_DSV41_QDQ='%s')\n",
                    r.wo_a, r.kv_win, r.kv_comp, r.idx_k, r.idx_q, env);
        }
        return r;
    }();
    return sites;
}
