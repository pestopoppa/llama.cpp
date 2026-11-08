// AK-ASM-H10: GQA-packed multi-token verify attention for head_dim 256 (see fattn-gqa-verify.cuh).
//
// Technique after AITER pa_*_mtp asm (GQA-packed paged attention for MTP/spec verify), written as a HIP kernel with
// MFMA builtins because the AITER asm is head_size 128 only.
//
// Layout of one workgroup: (KV split, KV head, sequence). Rows R = gqa * n_q, row r = h_local * n_q + q. Wave w owns the
// 16-row tile r in [16w, 16w+16). Per 32-key chunk:
//   1. mask chunk -> LDS; chunk skipped if every mask value of the n_q rows is -inf (KV not yet filled / causal tail)
//   2. K chunk -> LDS as f16 row-major [key][d], V chunk -> LDS as f16 transposed [d][key]; q8_0 is dequantized in
//      registers between the global load and the LDS store
//   3. S = Q K^T (v_mfma_f32_16x16x16f16, Q fragments resident in VGPRs), online softmax in f32, P -> LDS (f16)
//   4. O += P V (MFMA, 16 accumulator tiles of 16x16 per wave)
// Partials (unnormalized O, running max m, running sum l) per split go to a pool buffer; a merge kernel combines the
// splits with the usual LSE rescaling and writes dst [D, n_head, n_q, n_seq].
//
// MFMA 16x16x16 f16 operand layout (lane l, kq = l / 16):
//   A[i][k]  : i = l % 16, k = 4*kq + 0..3
//   B[k][j]  : j = l % 16, k = 4*kq + 0..3
//   D[i][j]  : i = 4*kq + v (v = 0..3), j = l % 16

#include "fattn-gqa-verify.cuh"
#include "fattn-common.cuh"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cfloat>

#define GV_D        256
#define GV_KC       32                  // keys per chunk
#define GV_KSTR     (GV_D + 8)          // sK row stride (halves)
#define GV_VTSTR    (GV_KC + 8)         // sVt row stride (halves)
#define GV_PSTR     (GV_KC + 8)         // sP row stride (halves)
#define GV_MAX_NQ   16
#define GV_MAX_TILES 12
#define GV_MAX_ROWS (16 * GV_MAX_TILES)
#define GV_MAX_WAVES GV_MAX_TILES
#define GV_MIN_WAVES 4
#define GV_PIPE_MAX_WAVES 8   // pipelined variant: <= 8 tiles (gqa*n_q <= 128), 256-VGPR budget

static int gv_env_int(const char * name, int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

static bool gv_enabled() {
    static const bool v = gv_env_int("GGML_CUDA_FA_GQA_VERIFY", 1) != 0;
    return v;
}

// C8-FOLD1: default 1 routes n_q=1 decode through gv (27B Q8_0 tg64 1.08x @32k, 1.25x @128k vs 2);
// GGML_CUDA_FA_GQA_VERIFY_MIN_NQ=2 restores the old vec-kernel decode path. n_q=1 never groups
// (gv_group_rows returns 0 below n_q 2) and multi-seq n_q=1 decode is rejected by rows_are_seqs.
// C8-FOLD3: the n_q=1 win is q8_0-KV only (f16 KV: A3B tg@32k 0.928, 27B 0.978 in the fold2 gate), so the
// default is 1 when K and V are q8_0 and 2 otherwise; the env var, when set, wins for every K/V type.
static int gv_min_nq(const bool kv_q8_0) {
    static const int env = gv_env_int("GGML_CUDA_FA_GQA_VERIFY_MIN_NQ", -1);
    if (env >= 0) {
        return env;
    }
    return kv_q8_0 ? 1 : 2;
}

static int gv_splits_env() {
    static const int v = gv_env_int("GGML_CUDA_FA_GQA_VERIFY_SPLITS", 0);
    return v;
}

// GGML_CUDA_FA_GQA_VERIFY_PIPE: register-prefetch pipelined K/V load, default ON (H10 gate r3: best arm, 2.63x geomean vs C6
// over kv>=32k n_q 2..16, no cell < 0.95); =0 selects the single-buffered kernel.
static bool gv_pipe() {
    static const bool v = gv_env_int("GGML_CUDA_FA_GQA_VERIFY_PIPE", 1) != 0;
    return v;
}

// C8-1-LCATTN: multi-sequence verify (unified KV, n_seqs_unq > 1, e.g. DFlash2 np4: 4 seqs x 9 tokens = 36 rows).
// The query rows are split into row groups of G rows (G = rows per sequence when they divide evenly), each group a
// virtual sequence of the GQA-packed kernel (grid.z = n_seq * n_grp); each workgroup trims its KV split to the live
// chunk range of its own rows' mask before loading K/V, so a group only streams the KV cells of its own sequence.
// Default ON; GGML_HIP_GV_MULTISEQ=0 restores the C7r selection (n_q > 16 -> flash_attn_ext_f16).
static bool gv_multiseq() {
    static const bool v = gv_env_int("GGML_HIP_GV_MULTISEQ", 1) != 0;
    return v;
}

static int gv_multiseq_max_nq() {
    static const int v = gv_env_int("GGML_HIP_GV_MULTISEQ_MAX_NQ", 64);
    return v;
}

// row-group size for a call with n_q query rows, gqa q heads per KV head and n_su distinct sequences; 0 = no grouping
static int gv_group_rows(const int n_q, const int gqa, const int n_su) {
    if (!gv_multiseq() || n_q < 2) {
        return 0;
    }
    if (n_su > 1 && n_q % n_su == 0) {
        const int g = n_q / n_su;
        if (g <= GV_MAX_NQ && gqa * g <= 16 * GV_PIPE_MAX_WAVES) {
            return g;
        }
    }
    if (n_q <= GV_MAX_NQ && gqa * n_q <= GV_MAX_ROWS) {
        return 0; // single group: unchanged H10 path
    }
    int g = std::min(GV_MAX_NQ, (16 * GV_PIPE_MAX_WAVES) / gqa);
    return g >= 1 ? g : 0;
}

static bool gv_debug() {
    static const bool v = gv_env_int("GGML_CUDA_FA_GQA_VERIFY_DEBUG", 0) != 0;
    return v;
}

static size_t gv_smem_bytes(const int nwaves) {
    return sizeof(_Float16) * (size_t) (GV_KC * GV_KSTR + GV_D * GV_VTSTR + nwaves * 16 * GV_PSTR) +
           sizeof(float) * (size_t) (GV_MAX_NQ * GV_KC);
}

#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
typedef _Float16 gv_halfx4 __attribute__((ext_vector_type(4)));
typedef float    gv_floatx4 __attribute__((ext_vector_type(4)));

// steps 3+4 of one 32-key chunk (sK / sVt / smask filled, block-synchronized); contains one __syncthreads()
static __device__ __forceinline__ void gv_compute_chunk(
        const bool tile_active, const _Float16 * sK, const _Float16 * sVt, _Float16 * sP, const float * smask,
        const gv_halfx4 (&qa)[GV_D / 16], gv_floatx4 (&o)[GV_D / 16], float (&m_run)[4], float (&l_run)[4],
        const bool (&row_ok)[4], const int (&row_q)[4], const float scale, const int li, const int kq) {
    // 3. S = Q K^T, online softmax, P -> sP
    float alpha[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (tile_active) {
        gv_floatx4 S[GV_KC / 16];
#pragma unroll
        for (int nb = 0; nb < GV_KC / 16; ++nb) {
            S[nb] = (gv_floatx4) {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int s = 0; s < GV_D / 16; ++s) {
                const gv_halfx4 kb = *(const gv_halfx4 *) (sK + (nb * 16 + li) * GV_KSTR + 16 * s + 4 * kq);
                S[nb] = __builtin_amdgcn_mfma_f32_16x16x16f16(qa[s], kb, S[nb], 0, 0, 0);
            }
        }

        float rmax[4];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            rmax[v] = -FLT_MAX / 2.0f;
#pragma unroll
            for (int nb = 0; nb < GV_KC / 16; ++nb) {
                const float mv = row_ok[v] ? smask[row_q[v] * GV_KC + nb * 16 + li] : -INFINITY;
                S[nb][v] = S[nb][v] * scale + mv;
                rmax[v] = fmaxf(rmax[v], S[nb][v]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                rmax[v] = fmaxf(rmax[v], __shfl_xor(rmax[v], off, 64));
            }
        }

#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const float m_new = fmaxf(m_run[v], rmax[v]);
            alpha[v] = __expf(m_run[v] - m_new);
            float rsum = 0.0f;
#pragma unroll
            for (int nb = 0; nb < GV_KC / 16; ++nb) {
                const float p = __expf(S[nb][v] - m_new);
                rsum += p;
                sP[(4 * kq + v) * GV_PSTR + nb * 16 + li] = (_Float16) p;
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                rsum += __shfl_xor(rsum, off, 64);
            }
            l_run[v] = l_run[v] * alpha[v] + rsum;
            m_run[v] = m_new;
        }
    }
    __syncthreads();

    // 4. O = O*alpha + P V
    if (tile_active) {
#pragma unroll
        for (int nb = 0; nb < GV_D / 16; ++nb) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                o[nb][v] *= alpha[v];
            }
        }
#pragma unroll
        for (int t = 0; t < GV_KC / 16; ++t) {
            const gv_halfx4 pa = *(const gv_halfx4 *) (sP + li * GV_PSTR + 16 * t + 4 * kq);
#pragma unroll
            for (int nb = 0; nb < GV_D / 16; ++nb) {
                const gv_halfx4 vb = *(const gv_halfx4 *) (sVt + (nb * 16 + li) * GV_VTSTR + 16 * t + 4 * kq);
                o[nb] = __builtin_amdgcn_mfma_f32_16x16x16f16(pa, vb, o[nb], 0, 0, 0);
            }
        }
    }
}
#endif

template <bool KV_Q8, bool PIPE>
static __global__ void __launch_bounds__(PIPE ? 64 * GV_PIPE_MAX_WAVES : 64 * GV_MAX_WAVES, 1) gv_fattn_kernel(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, float * __restrict__ Opart, float2 * __restrict__ MLpart,
        const float scale, const int n_q_tot, const int gqa, const int n_head_kv, const int n_grp, const int G,
        const int chunks_per_split, const int n_chunks,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int64_t nb31, const int64_t nb33, const int ne33) {
#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
    extern __shared__ char gv_smem[];
    _Float16 * sK   = (_Float16 *) gv_smem;
    _Float16 * sVt  = sK + GV_KC * GV_KSTR;
    _Float16 * sP0  = sVt + GV_D * GV_VTSTR;
    const int nthreads = blockDim.x;
    const int nwaves   = nthreads / 64;
    float    * smask = (float *) (sP0 + nwaves * 16 * GV_PSTR);

    const int split = blockIdx.x;
    const int hkv   = blockIdx.y;
    const int vseq  = blockIdx.z;          // virtual sequence = (sequence, row group)
    const int n_vseq = gridDim.z;
    const int seq   = vseq / n_grp;
    const int q0    = (vseq % n_grp) * G;  // first query row of this group
    const int n_q   = min(G, n_q_tot - q0);
    const int tid   = threadIdx.x;
    const int w     = tid / 64;
    const int l     = tid % 64;
    const int li    = l % 16;
    const int kq    = l / 16;

    const int R      = gqa * n_q;
    const int Rs     = gqa * G;            // partial-buffer row stride per (vseq, hkv)
    const int ntiles = (R + 15) / 16;
    const bool tile_active = w < ntiles;
    _Float16 * sP = sP0 + w * 16 * GV_PSTR;

    const char * Kb = K + seq * nb13 + hkv * nb12;
    const char * Vb = V + seq * nb23 + hkv * nb22;
    const char * Mb = mask + (seq % ne33) * nb33 + (int64_t) q0 * nb31;

    // Q fragments of this wave's tile (A operand), f32 -> f16
    gv_halfx4 qa[GV_D / 16];
    {
        const int r = 16 * w + li;
        const bool valid = tile_active && r < R;
        const int h = valid ? r / n_q : 0;
        const int q = valid ? r % n_q : 0;
        const float * Qr = (const float *) (Q + seq * nb03 + (int64_t) (hkv * gqa + h) * nb02 + (int64_t) (q0 + q) * nb01);
#pragma unroll
        for (int s = 0; s < GV_D / 16; ++s) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                qa[s][t] = valid ? (_Float16) Qr[16 * s + 4 * kq + t] : (_Float16) 0.0f;
            }
        }
    }

    gv_floatx4 o[GV_D / 16];
#pragma unroll
    for (int nb = 0; nb < GV_D / 16; ++nb) {
        o[nb] = (gv_floatx4) {0.0f, 0.0f, 0.0f, 0.0f};
    }
    float m_run[4];
    float l_run[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        m_run[v] = -FLT_MAX / 2.0f;
        l_run[v] = 0.0f;
    }

    // row -> query index for the mask, per accumulator row v
    int row_q[4];
    bool row_ok[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int r = 16 * w + 4 * kq + v;
        row_ok[v] = tile_active && r < R;
        row_q[v]  = row_ok[v] ? r % n_q : 0;
    }

    int c_begin = split * chunks_per_split;
    int c_end   = min(c_begin + chunks_per_split, n_chunks);

    if (n_grp > 1) {
        // live-range trim: [first, last] chunk of this split with a non -inf mask value in this group's rows
        __shared__ int s_lo;
        __shared__ int s_hi;
        if (tid == 0) {
            s_lo = c_end;
            s_hi = c_begin - 1;
        }
        __syncthreads();
        int my_lo = c_end;
        int my_hi = c_begin - 1;
        const int nk = (c_end - c_begin) * GV_KC;
        for (int i = tid; i < n_q * nk; i += nthreads) {
            const int q   = i / nk;
            const int key = c_begin * GV_KC + i % nk;
            const uint16_t u = *(const uint16_t *) (Mb + q * nb31 + (int64_t) key * sizeof(half));
            if (u != 0xFC00) {
                my_lo = min(my_lo, key / GV_KC);
                my_hi = max(my_hi, key / GV_KC);
            }
        }
        if (my_lo < c_end) {
            atomicMin(&s_lo, my_lo);
        }
        if (my_hi >= c_begin) {
            atomicMax(&s_hi, my_hi);
        }
        __syncthreads();
        c_begin = s_lo;
        c_end   = s_hi + 1;
    }

    if constexpr (PIPE) {
        // Software pipeline: the raw global data of chunk c+1 (mask halves, q8_0 blocks or f16 16-byte rows) is loaded into
        // registers while chunk c is computed, then dequantized/stored to LDS after the end-of-chunk barrier.
        // nthreads in [256, 512]: q8_0 = 1 block of K + 1 of V per thread, f16 = up to 4 x 16 B of K and of V, mask = up to 2.
        constexpr int NK = KV_Q8 ? 1 : (GV_KC * (GV_D / 8) + 255) / 256;
        uint16_t rk[KV_Q8 ? 1 + QK8_0 / 2 : 1];
        uint16_t rv[KV_Q8 ? 1 + QK8_0 / 2 : 1];
        uint32_t fk[4 * NK]; // plain words: an array of HIP_vector_type uint4 is not promoted to registers (scratch)
        uint32_t fv[4 * NK];
        uint16_t rm[2];

        auto load_raw = [&](const int c) __attribute__((always_inline)) {
            const int kbase = c * GV_KC;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int i = tid + s * nthreads;
                rm[s] = i < n_q * GV_KC ?
                    *(const uint16_t *) (Mb + (i / GV_KC) * nb31 + (int64_t) (kbase + i % GV_KC) * sizeof(half)) : (uint16_t) 0xFC00;
            }
            if constexpr (KV_Q8) {
                if (tid < GV_KC * (GV_D / QK8_0)) {
                    const int b = tid / GV_KC, key = tid % GV_KC;
                    const uint16_t * pk = (const uint16_t *) (Kb + (int64_t) (kbase + key) * nb11 + b * (int64_t) sizeof(block_q8_0));
                    const uint16_t * pv = (const uint16_t *) (Vb + (int64_t) (kbase + key) * nb21 + b * (int64_t) sizeof(block_q8_0));
#pragma unroll
                    for (int t = 0; t < 1 + QK8_0 / 2; ++t) {
                        rk[t] = pk[t];
                        rv[t] = pv[t];
                    }
                }
            } else {
#pragma unroll
                for (int s = 0; s < NK; ++s) {
                    const int i = tid + s * nthreads;
                    if (i < GV_KC * (GV_D / 8)) {
                        const int cc = i / GV_KC, key = i % GV_KC;
                        const uint4 tk = *(const uint4 *) (Kb + (int64_t) (kbase + key) * nb11 + cc * 16);
                        const uint4 tv = *(const uint4 *) (Vb + (int64_t) (kbase + key) * nb21 + cc * 16);
                        fk[4 * s + 0] = tk.x; fk[4 * s + 1] = tk.y; fk[4 * s + 2] = tk.z; fk[4 * s + 3] = tk.w;
                        fv[4 * s + 0] = tv.x; fv[4 * s + 1] = tv.y; fv[4 * s + 2] = tv.z; fv[4 * s + 3] = tv.w;
                    }
                }
            }
        };

        auto store_raw = [&]() __attribute__((always_inline)) -> int {
            int any = 0;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int i = tid + s * nthreads;
                if (i < n_q * GV_KC) {
                    const float mv = __half2float(__ushort_as_half(rm[s]));
                    smask[i] = mv;
                    any |= mv != -INFINITY;
                }
            }
            if constexpr (KV_Q8) {
                if (tid < GV_KC * (GV_D / QK8_0)) {
                    const int b = tid / GV_KC, key = tid % GV_KC;
                    const float dk = __half2float(__ushort_as_half(rk[0]));
                    const float dv = __half2float(__ushort_as_half(rv[0]));
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        _Float16 tmp[8];
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const uint16_t u = rk[1 + 4 * g + t];
                            tmp[2 * t + 0] = (_Float16) (dk * (float) (int8_t) (u & 0xFF));
                            tmp[2 * t + 1] = (_Float16) (dk * (float) (int8_t) (u >> 8));
                        }
                        *(uint4 *) (sK + key * GV_KSTR + b * QK8_0 + 8 * g) = *(const uint4 *) tmp;
                    }
#pragma unroll
                    for (int t = 0; t < QK8_0 / 2; ++t) {
                        const uint16_t u = rv[1 + t];
                        sVt[(b * QK8_0 + 2 * t + 0) * GV_VTSTR + key] = (_Float16) (dv * (float) (int8_t) (u & 0xFF));
                        sVt[(b * QK8_0 + 2 * t + 1) * GV_VTSTR + key] = (_Float16) (dv * (float) (int8_t) (u >> 8));
                    }
                }
            } else {
#pragma unroll
                for (int s = 0; s < NK; ++s) {
                    const int i = tid + s * nthreads;
                    if (i < GV_KC * (GV_D / 8)) {
                        const int cc = i / GV_KC, key = i % GV_KC;
                        *(uint4 *) (sK + key * GV_KSTR + cc * 8) = make_uint4(fk[4 * s + 0], fk[4 * s + 1], fk[4 * s + 2], fk[4 * s + 3]);
#pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            const uint16_t hb = (uint16_t) (fv[4 * s + t / 2] >> (16 * (t % 2)));
                            *(uint16_t *) (sVt + (cc * 8 + t) * GV_VTSTR + key) = hb;
                        }
                    }
                }
            }
            return any;
        };

        if (c_begin < c_end) {
            load_raw(c_begin);
        }
        for (int c = c_begin; c < c_end; ++c) {
            const int live = __syncthreads_or(store_raw());
            if (c + 1 < c_end) {
                load_raw(c + 1); // in flight during the MFMA work below
            }
            if (live) {
                gv_compute_chunk(tile_active, sK, sVt, sP, smask, qa, o, m_run, l_run, row_ok, row_q, scale, li, kq);
            }
            __syncthreads();
        }
    } else
    for (int c = c_begin; c < c_end; ++c) {
        const int kbase = c * GV_KC;

        // 1. mask chunk + skip test
        int any_live = 0;
        for (int i = tid; i < n_q * GV_KC; i += nthreads) {
            const int q = i / GV_KC;
            const int j = i % GV_KC;
            const float mv = __half2float(*(const half *) (Mb + q * nb31 + (int64_t) (kbase + j) * sizeof(half)));
            smask[q * GV_KC + j] = mv;
            any_live |= mv != -INFINITY;
        }
        if (!__syncthreads_or(any_live)) {
            continue;
        }

        // 2. K -> sK [key][d], V -> sVt [d][key]
        if constexpr (KV_Q8) {
            for (int i = tid; i < GV_KC * (GV_D / QK8_0); i += nthreads) {
                const int b   = i / GV_KC;
                const int key = i % GV_KC;
                {
                    const char * blk = Kb + (int64_t) (kbase + key) * nb11 + b * (int64_t) sizeof(block_q8_0);
                    const float d = __half2float(*(const half *) blk);
                    const uint16_t * qs = (const uint16_t *) (blk + sizeof(half));
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        _Float16 tmp[8];
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const uint16_t u = qs[4 * g + t];
                            tmp[2 * t + 0] = (_Float16) (d * (float) (int8_t) (u & 0xFF));
                            tmp[2 * t + 1] = (_Float16) (d * (float) (int8_t) (u >> 8));
                        }
                        *(uint4 *) (sK + key * GV_KSTR + b * QK8_0 + 8 * g) = *(const uint4 *) tmp;
                    }
                }
                {
                    const char * blk = Vb + (int64_t) (kbase + key) * nb21 + b * (int64_t) sizeof(block_q8_0);
                    const float d = __half2float(*(const half *) blk);
                    const uint16_t * qs = (const uint16_t *) (blk + sizeof(half));
#pragma unroll
                    for (int t = 0; t < QK8_0 / 2; ++t) {
                        const uint16_t u = qs[t];
                        sVt[(b * QK8_0 + 2 * t + 0) * GV_VTSTR + key] = (_Float16) (d * (float) (int8_t) (u & 0xFF));
                        sVt[(b * QK8_0 + 2 * t + 1) * GV_VTSTR + key] = (_Float16) (d * (float) (int8_t) (u >> 8));
                    }
                }
            }
        } else {
            for (int i = tid; i < GV_KC * (GV_D / 8); i += nthreads) {
                const int cc  = i / GV_KC;
                const int key = i % GV_KC;
                const uint4 kv = *(const uint4 *) (Kb + (int64_t) (kbase + key) * nb11 + cc * 16);
                *(uint4 *) (sK + key * GV_KSTR + cc * 8) = kv;
                const uint4 vv = *(const uint4 *) (Vb + (int64_t) (kbase + key) * nb21 + cc * 16);
                const _Float16 * vh = (const _Float16 *) &vv;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    sVt[(cc * 8 + t) * GV_VTSTR + key] = vh[t];
                }
            }
        }
        __syncthreads();

        gv_compute_chunk(tile_active, sK, sVt, sP, smask, qa, o, m_run, l_run, row_ok, row_q, scale, li, kq);
        __syncthreads();
    }

    // partials: Opart[(((split*n_seq + seq)*n_head_kv + hkv)*R + r)*D + d], MLpart[... r]
    if (tile_active) {
        const int64_t base_row = ((int64_t) (split * n_vseq + vseq) * n_head_kv + hkv) * Rs;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            if (!row_ok[v]) {
                continue;
            }
            const int r = 16 * w + 4 * kq + v;
            float * Or = Opart + (base_row + r) * GV_D;
#pragma unroll
            for (int nb = 0; nb < GV_D / 16; ++nb) {
                Or[nb * 16 + li] = o[nb][v];
            }
            if (li == 0) {
                MLpart[base_row + r] = make_float2(m_run[v], l_run[v]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, Opart, MLpart, scale, n_q_tot, gqa, n_head_kv, n_grp, G, chunks_per_split, n_chunks,
        nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33, ne33);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
}

// one block per (row r, kv head, seq), one thread per d
static __global__ void gv_merge_kernel(
        const float * __restrict__ Opart, const float2 * __restrict__ MLpart, float * __restrict__ dst,
        const int n_splits, const int n_q_tot, const int gqa, const int n_head_kv, const int n_grp, const int G) {
    const int r    = blockIdx.x;
    const int hkv  = blockIdx.y;
    const int vseq = blockIdx.z;
    const int n_seq = gridDim.z;           // virtual sequences
    const int d    = threadIdx.x;
    const int R    = gqa * G;              // row stride
    const int q0   = (vseq % n_grp) * G;
    const int n_q  = min(G, n_q_tot - q0);
    if (r >= gqa * n_q) {
        return;
    }

    float M = -FLT_MAX;
    for (int s = 0; s < n_splits; ++s) {
        const float2 ml = MLpart[((int64_t) (s * n_seq + vseq) * n_head_kv + hkv) * R + r];
        if (ml.y > 0.0f) {
            M = fmaxf(M, ml.x);
        }
    }
    float acc = 0.0f;
    float L   = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const int64_t row = ((int64_t) (s * n_seq + vseq) * n_head_kv + hkv) * R + r;
        const float2 ml = MLpart[row];
        if (ml.y > 0.0f) {
            const float f = __expf(ml.x - M);
            L   += f * ml.y;
            acc += f * Opart[row * GV_D + d];
        }
    }
    const int h = r / n_q;
    const int q = q0 + r % n_q;
    const int seq = vseq / n_grp;
    const int n_head = n_head_kv * gqa;
    dst[(((int64_t) seq * n_q_tot + q) * n_head + hkv * gqa + h) * GV_D + d] = L > 0.0f ? acc / L : 0.0f;
}

// GGML_CUDA_FA_GQA_VERIFY_DEBUG=1: the first 32 rejections of hd256 FA nodes are logged with their reason (WARN level, so that
// llama-server's default verbosity shows them; ggml INFO lines are filtered there).
static bool gv_reject(const ggml_tensor * dst, const char * why) {
    // expected rejections (kill switch, prefill ubatches, single-token decode) are not logged: they would fill the 32 slots
    const bool expected = strncmp(why, "disabled", 8) == 0 || strncmp(why, "n_q outside", 11) == 0;
    if (gv_debug() && !expected && dst->src[0]->ne[0] == GV_D) {
        static int n_rej = 0;
        if (n_rej++ < 32) {
            const ggml_tensor * Q = dst->src[0];
            const ggml_tensor * K = dst->src[1];
            const ggml_tensor * M = dst->src[3];
            GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: reject %s: %s (Q %lld,%lld,%lld,%lld K %lld,%lld,%lld,%lld %s nb %zu,%zu,%zu mask %lld,%lld,%lld,%lld)\n",
                dst->name, why, (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], (long long) Q->ne[3],
                (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], (long long) K->ne[3], ggml_type_name(K->type),
                K->nb[1], K->nb[2], K->nb[3],
                M ? (long long) M->ne[0] : -1LL, M ? (long long) M->ne[1] : -1LL, M ? (long long) M->ne[2] : -1LL, M ? (long long) M->ne[3] : -1LL);
        }
    }
    return false;
}

bool ggml_cuda_fattn_gqa_verify_applicable(const int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    if (!gv_enabled()) {
        return gv_reject(dst, "disabled (GGML_CUDA_FA_GQA_VERIFY=0)");
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!amd_mfma_available(cc)) {
        return gv_reject(dst, "no MFMA");
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return gv_reject(dst, "Q/dst not f32 or dst not contiguous");
    }
    if (Q->ne[0] != GV_D || K->ne[0] != GV_D || V->ne[0] != GV_D) {
        return gv_reject(dst, "head dim != 256");
    }
    if (K->type != V->type || (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0)) {
        return gv_reject(dst, "K/V type not equal f16/q8_0");
    }
    if (mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || sinks != nullptr) {
        return gv_reject(dst, "mask missing/not f16/ne2!=1 or sinks");
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return gv_reject(dst, "ALiBi or softcap");
    }
    const int64_t n_q = Q->ne[1];
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0 || V->ne[2] != K->ne[2]) {
        return gv_reject(dst, "GQA layout");
    }
    const int64_t gqa = Q->ne[2] / K->ne[2];
    // C8-1-LCATTN: multi-sequence verify batches (n_q > 16 or gqa*n_q > 192) run as row groups
    const int n_grp_rows = n_q <= gv_multiseq_max_nq() ?
        gv_group_rows((int) n_q, (int) gqa, ggml_flash_attn_ext_get_n_seq(dst)) : 0;
    if (n_q < gv_min_nq(K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0) || n_q < 1 || (n_q > GV_MAX_NQ && n_grp_rows == 0)) {
        return gv_reject(dst, "n_q outside [MIN_NQ,16] (multiseq off or > MAX_NQ)");
    }
    if (n_grp_rows == 0 && gqa * n_q > GV_MAX_ROWS) {
        return gv_reject(dst, "gqa*n_q > 192");
    }
    if (K->ne[1] % FATTN_KQ_STRIDE != 0 || V->ne[1] != K->ne[1] || mask->ne[0] < K->ne[1] || mask->ne[1] < n_q) {
        return gv_reject(dst, "n_kv not multiple of 256 or mask too small");
    }
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return gv_reject(dst, "K/V/Q ne3 mismatch");
    }
    if (Q->nb[0] != sizeof(float) || K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return gv_reject(dst, "non-unit element stride");
    }
    // f16: 16-byte loads per key row
    if (K->type == GGML_TYPE_F16) {
        if (K->nb[1] % 16 != 0 || V->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || V->nb[2] % 16 != 0 ||
            K->nb[3] % 16 != 0 || V->nb[3] % 16 != 0 ||
            (uintptr_t) K->data % 16 != 0 || (uintptr_t) V->data % 16 != 0) {
            return gv_reject(dst, "f16 K/V stride/alignment not 16 B");
        }
    } else {
        if (K->nb[1] % 2 != 0 || V->nb[1] % 2 != 0 || (uintptr_t) K->data % 2 != 0 || (uintptr_t) V->data % 2 != 0) {
            return gv_reject(dst, "q8_0 K/V stride/alignment");
        }
    }
    // batched decode of several sequences on a unified KV stays on the per-sequence vec layout
    if (ggml_cuda_fattn_rows_are_seqs(dst)) {
        return gv_reject(dst, "rows_are_seqs (multi-sequence decode)");
    }
    if (gv_debug()) {
        static int n_ok = 0;
        if (n_ok++ < 4) {
            GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: accept %s n_q=%lld n_kv=%lld type=%s\n", dst->name, (long long) Q->ne[1], (long long) K->ne[1], ggml_type_name(K->type));
        }
    }
    return true;
#else
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#endif // defined(GGML_USE_HIP)
}

void ggml_cuda_fattn_gqa_verify(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int n_q       = (int) Q->ne[1];
    const int n_head_kv = (int) K->ne[2];
    const int gqa       = (int) (Q->ne[2] / K->ne[2]);
    const int n_seq     = (int) Q->ne[3];
    const int n_kv      = (int) K->ne[1];
    const int G0        = gv_group_rows(n_q, gqa, ggml_flash_attn_ext_get_n_seq(dst));
    const int G         = G0 > 0 ? G0 : n_q;
    const int n_grp     = (n_q + G - 1) / G;
    const int n_vseq    = n_seq * n_grp;
    const int R         = gqa * G;
    const int ntiles    = (R + 15) / 16;
    const int nwaves    = std::max(ntiles, GV_MIN_WAVES);
    GGML_ASSERT(ntiles <= GV_MAX_TILES);
    const bool pipe     = gv_pipe() && ntiles <= GV_PIPE_MAX_WAVES;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_chunks = n_kv / GV_KC;
    const int nsm      = ggml_cuda_info().devices[ctx.device].nsm;
    int n_splits = gv_splits_env();
    if (n_splits <= 0) {
        // grouped: each group streams ~1/n_grp of the KV (live-range trim), so size the splits per real sequence
        const int base = n_head_kv * n_seq;
        n_splits = (2 * nsm + base - 1) / base;
        n_splits = std::min(n_splits, std::max(1, n_kv / 256));
        n_splits = std::min(n_splits, 256);
    }
    n_splits = std::max(1, std::min(n_splits, n_chunks));
    const int chunks_per_split = (n_chunks + n_splits - 1) / n_splits;
    n_splits = (n_chunks + chunks_per_split - 1) / chunks_per_split;

    const size_t n_rows_part = (size_t) n_splits * n_vseq * n_head_kv * R;
    ggml_cuda_pool_alloc<float>  Opart(ctx.pool(), n_rows_part * GV_D);
    ggml_cuda_pool_alloc<float2> MLpart(ctx.pool(), n_rows_part);

    cudaStream_t stream = ctx.stream();
    const dim3 grid(n_splits, n_head_kv, n_vseq);
    const dim3 block(64 * nwaves);
    const size_t smem = gv_smem_bytes(nwaves);

    const int ne33 = (int) mask->ne[3];
    const bool q8 = K->type == GGML_TYPE_Q8_0;
    if (pipe) {
        if (q8) {
            gv_fattn_kernel<true, true><<<grid, block, smem, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, n_grp, G, chunks_per_split, n_chunks,
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
            mask->nb[1], mask->nb[3], ne33);
        } else {
            gv_fattn_kernel<false, true><<<grid, block, smem, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, n_grp, G, chunks_per_split, n_chunks,
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
            mask->nb[1], mask->nb[3], ne33);
        }
    } else {
        if (q8) {
            gv_fattn_kernel<true, false><<<grid, block, smem, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, n_grp, G, chunks_per_split, n_chunks,
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
            mask->nb[1], mask->nb[3], ne33);
        } else {
            gv_fattn_kernel<false, false><<<grid, block, smem, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, n_grp, G, chunks_per_split, n_chunks,
            Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
            mask->nb[1], mask->nb[3], ne33);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    gv_merge_kernel<<<dim3(R, n_head_kv, n_vseq), dim3(GV_D), 0, stream>>>(
        Opart.get(), MLpart.get(), (float *) dst->data, n_splits, n_q, gqa, n_head_kv, n_grp, G);
    CUDA_CHECK(cudaGetLastError());

    // path witness for the gate: emitted once per process, on the first call
    static int n_calls = 0;
    if (n_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GQA-packed verify attention ran (call 1)\n");
    }
    static int n_ms_calls = 0;
    if (n_grp > 1 && n_ms_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GGML_HIP_GV_MULTISEQ row-group variant ran (n_q=%d G=%d n_grp=%d)\n", n_q, G, n_grp);
    }
    static int n_pipe_calls = 0;
    if (pipe && n_pipe_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: pipelined K/V load variant ran (call 1)\n");
    }
    if (gv_debug() && n_calls <= 8) {
        GGML_LOG_WARN("%s: %s n_q=%d gqa=%d n_head_kv=%d n_kv=%d n_seq=%d type=%s splits=%d waves=%d smem=%zu pipe=%d\n",
            __func__, dst->name, n_q, gqa, n_head_kv, n_kv, n_seq, ggml_type_name(K->type), n_splits, nwaves, smem, (int) pipe);
    }
}
