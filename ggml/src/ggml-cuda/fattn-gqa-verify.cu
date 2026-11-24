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

// GGML_HIP_GV_PREFETCH_LATE: C9-LONGCTX-FA-BANDWIDTH prefetch-overlap variant of the q8_0 pipelined kernel, default ON (=0 disables). Holds the
// raw blocks in the dwords the global loads produce and defers all 16-bit extraction to store_raw, so that the wave does not wait on
// vmcnt before the chunk's MFMA work. The bytes read and the arithmetic of store_raw are unchanged, so results are identical.
static bool gv_prefetch_late() {
    static const bool v = gv_env_int("GGML_HIP_GV_PREFETCH_LATE", 1) != 0;
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

// GGML_HIP_VATTN_RF=1 (C22 VATTN-ROOFLINE-1, default OFF): occupancy-first launch regime for the single-sequence q8_0
// verify shapes (n_seq 1, no row grouping, R = gqa*n_q <= 64). C21 launches 4-wave workgroups with 43.5 KB LDS -> 1 WG/CU,
// and sizes splits as min(ceil(2*nsm/n_head_kv), n_kv/256): 128 WGs at kv 8k (512 waves, one CU in four runs a second
// WG after the first finishes, so the kernel takes two WG lifetimes) and 208 at kv 32k. RF instead (a) runs the 16-key
// LDS-cut kernel (24.8 KB, __launch_bounds__(256, 2): 2 WG = 8 waves per CU resident), (b) sizes splits to fill exactly
// one resident round, splits = floor(2*nsm/n_head_kv) (52 on MI210 at 4 KV heads -> 208 WGs = 832 waves = 8 per CU,
// no tail), (c) caps splits so the f32 partial-O write stays <= 1/2 of the KV bytes read (the merge traffic is otherwise
// what the nq 8 shapes pay for a wide split) and so every split streams >= 8 chunks (128 keys). The merge is unchanged
// and sums splits in index order, so the result is deterministic run to run.
// GGML_HIP_VATTN_RF=2 adds lever 2 (double-buffered LDS, one barrier per chunk, see gv_rf_kernel); =4 adds lever 4 (K
// register-direct, 4 WG/CU). Any other nonzero value is lever 1 only.
static int gv_vattn_rf_level() {
    static const int v = gv_env_int("GGML_HIP_VATTN_RF", 0);
    return v;
}
static bool gv_vattn_rf() {
    return gv_vattn_rf_level() != 0;
}

static bool gv_debug() {
    static const bool v = gv_env_int("GGML_CUDA_FA_GQA_VERIFY_DEBUG", 0) != 0;
    return v;
}

static size_t gv_smem_bytes(const int nwaves, const int kc = GV_KC) {
    return sizeof(_Float16) * (size_t) (kc * GV_KSTR + GV_D * (kc + 8) + nwaves * 16 * (kc + 8)) +
           sizeof(float) * (size_t) (GV_MAX_NQ * kc);
}

#define GV_KC_CUT 16

// GGML_CUDA_FA_GQA_LDS_CUT=1: 16-key chunk variant (24,832 B LDS at 4 waves -> 2 WG/CU), default OFF.
static bool gv_lds_cut() {
    static const bool v = gv_env_int("GGML_CUDA_FA_GQA_LDS_CUT", 0) != 0;
    return v;
}

#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
typedef _Float16 gv_halfx4 __attribute__((ext_vector_type(4)));
typedef float    gv_floatx4 __attribute__((ext_vector_type(4)));

// steps 3+4 of one KC-key chunk (sK / sVt / smask filled, block-synchronized); contains one __syncthreads()
template <int KC>
static __device__ __forceinline__ void gv_compute_chunk(
        const bool tile_active, const _Float16 * sK, const _Float16 * sVt, _Float16 * sP, const float * smask,
        const gv_halfx4 (&qa)[GV_D / 16], gv_floatx4 (&o)[GV_D / 16], float (&m_run)[4], float (&l_run)[4],
        const bool (&row_ok)[4], const int (&row_q)[4], const float scale, const int li, const int kq) {
    constexpr int VTSTR = KC + 8;
    constexpr int PSTR  = KC + 8;
    // 3. S = Q K^T, online softmax, P -> sP
    float alpha[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (tile_active) {
        gv_floatx4 S[KC / 16];
#pragma unroll
        for (int nb = 0; nb < KC / 16; ++nb) {
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
            for (int nb = 0; nb < KC / 16; ++nb) {
                const float mv = row_ok[v] ? smask[row_q[v] * KC + nb * 16 + li] : -INFINITY;
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
            for (int nb = 0; nb < KC / 16; ++nb) {
                const float p = __expf(S[nb][v] - m_new);
                rsum += p;
                sP[(4 * kq + v) * PSTR + nb * 16 + li] = (_Float16) p;
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
        for (int t = 0; t < KC / 16; ++t) {
            const gv_halfx4 pa = *(const gv_halfx4 *) (sP + li * PSTR + 16 * t + 4 * kq);
#pragma unroll
            for (int nb = 0; nb < GV_D / 16; ++nb) {
                const gv_halfx4 vb = *(const gv_halfx4 *) (sVt + (nb * 16 + li) * VTSTR + 16 * t + 4 * kq);
                o[nb] = __builtin_amdgcn_mfma_f32_16x16x16f16(pa, vb, o[nb], 0, 0, 0);
            }
        }
    }
}
#endif

// KPF-41 FAROWS-INPLACE: with ROWS the kernel reads the K/V rows of slice `seq` in place from the unified KV cache through
// the per-slice row list (rows_s = rows + seq*rows_s1, entry i = KV cell of gathered row i; < 0 = padding, read as cell 0
// and hidden by the gathered -inf mask column, exactly like the zero row of the gather path). Without ROWS it is the
// identity, so the non-ROWS instantiations compile to the unchanged code.
template <bool ROWS>
static __device__ __forceinline__ int64_t gv_kv_row(const int32_t * __restrict__ rows_s, const int i) {
    if constexpr (ROWS) {
        const int32_t r = rows_s[i];
        return (int64_t) (r < 0 ? 0 : r);
    } else {
        GGML_UNUSED(rows_s);
        return (int64_t) i;
    }
}

// GGML_CUDA_FA_GQA_LDS_CUT (AK-GVFATTN-LONGCTX step 2): KC = 16-key chunks instead of 32 halve sK, sVt, sP and the mask
// tile: 24,832 B LDS at 4 waves (vs 43,520 B), so two workgroups fit in the 64 KiB LDS of a CU. The cut variant is only
// launched with exactly GV_MIN_WAVES waves (R <= 64) and compiled with __launch_bounds__(256, 2) so that the register
// budget also admits two workgroups per CU. Per-key arithmetic is unchanged (online softmax is chunk-size invariant up
// to f32 rounding of the running max/sum).
template <bool KV_Q8, bool PIPE, bool LATE = false, bool ROWS = false, int KC = GV_KC>
static __global__ void __launch_bounds__(KC != GV_KC ? 64 * GV_MIN_WAVES : (PIPE ? 64 * GV_PIPE_MAX_WAVES : 64 * GV_MAX_WAVES),
                                         KC != GV_KC ? 2 : 1) gv_fattn_kernel(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, float * __restrict__ Opart, float2 * __restrict__ MLpart,
        const float scale, const int n_q_tot, const int gqa, const int n_head_kv, const int n_grp, const int G,
        const int chunks_per_split, const int n_chunks,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int64_t nb31, const int64_t nb33, const int ne33,
        const int32_t * __restrict__ rows = nullptr, const int64_t rows_s1 = 0) {
#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
    constexpr int VTSTR = KC + 8;
    constexpr int PSTR  = KC + 8;
    extern __shared__ char gv_smem[];
    _Float16 * sK   = (_Float16 *) gv_smem;
    _Float16 * sVt  = sK + KC * GV_KSTR;
    _Float16 * sP0  = sVt + GV_D * VTSTR;
    const int nthreads = blockDim.x;
    const int nwaves   = nthreads / 64;
    float    * smask = (float *) (sP0 + nwaves * 16 * PSTR);

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
    _Float16 * sP = sP0 + w * 16 * PSTR;

    const char * Kb = K + seq * nb13 + hkv * nb12;
    const char * Vb = V + seq * nb23 + hkv * nb22;
    const int32_t * rows_s = ROWS ? rows + (int64_t) seq * rows_s1 : nullptr; // ROWS: host passes nb13 = nb23 = 0
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
        const int nk = (c_end - c_begin) * KC;
        for (int i = tid; i < n_q * nk; i += nthreads) {
            const int q   = i / nk;
            const int key = c_begin * KC + i % nk;
            const uint16_t u = *(const uint16_t *) (Mb + q * nb31 + (int64_t) key * sizeof(half));
            if (u != 0xFC00) {
                my_lo = min(my_lo, key / KC);
                my_hi = max(my_hi, key / KC);
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
        constexpr int NK = KV_Q8 ? 1 : (KC * (GV_D / 8) + 255) / 256;
        uint16_t rk[KV_Q8 ? 1 + QK8_0 / 2 : 1];
        uint16_t rv[KV_Q8 ? 1 + QK8_0 / 2 : 1];
        uint32_t fk[4 * NK]; // plain words: an array of HIP_vector_type uint4 is not promoted to registers (scratch)
        uint32_t fv[4 * NK];
        uint16_t rm[2];
        // LATE: block_q8_0 (34 B, 2-aligned) as 8 dwords + the last halfword. No 16-bit value exists until store_raw.
        struct __attribute__((packed, aligned(2))) gv_q8_raw { uint32_t w[8]; uint16_t tail; };
        uint32_t kw[LATE ? 8 : 1];
        uint32_t vw[LATE ? 8 : 1];
        uint16_t kt = 0;
        uint16_t vt = 0;

        auto load_raw = [&](const int c) __attribute__((always_inline)) {
            const int kbase = c * KC;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int i = tid + s * nthreads;
                if constexpr (LATE) {
                    // no select against a default: store_raw reads rm[s] only under this same condition
                    if (i < n_q * KC) {
                        rm[s] = *(const uint16_t *) (Mb + (i / KC) * nb31 + (int64_t) (kbase + i % KC) * sizeof(half));
                    }
                } else {
                rm[s] = i < n_q * KC ?
                    *(const uint16_t *) (Mb + (i / KC) * nb31 + (int64_t) (kbase + i % KC) * sizeof(half)) : (uint16_t) 0xFC00;
                }
            }
            if constexpr (KV_Q8) {
                if (tid < KC * (GV_D / QK8_0)) {
                    const int b = tid / KC, key = tid % KC;
                    const uint16_t * pk = (const uint16_t *) (Kb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb11 + b * (int64_t) sizeof(block_q8_0));
                    const uint16_t * pv = (const uint16_t *) (Vb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb21 + b * (int64_t) sizeof(block_q8_0));
                    if constexpr (LATE) {
                        const gv_q8_raw * qk = (const gv_q8_raw *) pk;
                        const gv_q8_raw * qv = (const gv_q8_raw *) pv;
#pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            kw[t] = qk->w[t];
                            vw[t] = qv->w[t];
                        }
                        kt = qk->tail;
                        vt = qv->tail;
                    } else {
#pragma unroll
                    for (int t = 0; t < 1 + QK8_0 / 2; ++t) {
                        rk[t] = pk[t];
                        rv[t] = pv[t];
                    }
                    }
                }
            } else {
#pragma unroll
                for (int s = 0; s < NK; ++s) {
                    const int i = tid + s * nthreads;
                    if (i < KC * (GV_D / 8)) {
                        const int cc = i / KC, key = i % KC;
                        const uint4 tk = *(const uint4 *) (Kb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb11 + cc * 16);
                        const uint4 tv = *(const uint4 *) (Vb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb21 + cc * 16);
                        fk[4 * s + 0] = tk.x; fk[4 * s + 1] = tk.y; fk[4 * s + 2] = tk.z; fk[4 * s + 3] = tk.w;
                        fv[4 * s + 0] = tv.x; fv[4 * s + 1] = tv.y; fv[4 * s + 2] = tv.z; fv[4 * s + 3] = tv.w;
                    }
                }
            }
        };

        // halfword k of the raw block: same bytes as pk[k] (little-endian) in both representations
        auto raw_k = [&](const int k) __attribute__((always_inline)) -> uint16_t {
            if constexpr (LATE) {
                return k == QK8_0 / 2 ? kt : (uint16_t) (kw[k / 2] >> (16 * (k % 2)));
            } else {
                return rk[k];
            }
        };
        auto raw_v = [&](const int k) __attribute__((always_inline)) -> uint16_t {
            if constexpr (LATE) {
                return k == QK8_0 / 2 ? vt : (uint16_t) (vw[k / 2] >> (16 * (k % 2)));
            } else {
                return rv[k];
            }
        };

        auto store_raw = [&]() __attribute__((always_inline)) -> int {
            int any = 0;
            if constexpr (LATE) {
                // opaque point of first use of the raw halfwords: without it the scheduler hoists `tail >> 8` beside the load.
                uint32_t x0 = kt, x1 = vt, x2 = rm[0], x3 = rm[1];
                asm volatile("" : "+v"(x0), "+v"(x1), "+v"(x2), "+v"(x3));
                kt = (uint16_t) x0; vt = (uint16_t) x1; rm[0] = (uint16_t) x2; rm[1] = (uint16_t) x3;
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int i = tid + s * nthreads;
                if (i < n_q * KC) {
                    const float mv = __half2float(__ushort_as_half(rm[s]));
                    smask[i] = mv;
                    any |= mv != -INFINITY;
                }
            }
            if constexpr (KV_Q8) {
                if (tid < KC * (GV_D / QK8_0)) {
                    const int b = tid / KC, key = tid % KC;
                    const float dk = __half2float(__ushort_as_half(raw_k(0)));
                    const float dv = __half2float(__ushort_as_half(raw_v(0)));
#pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        _Float16 tmp[8];
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const uint16_t u = raw_k(1 + 4 * g + t);
                            tmp[2 * t + 0] = (_Float16) (dk * (float) (int8_t) (u & 0xFF));
                            tmp[2 * t + 1] = (_Float16) (dk * (float) (int8_t) (u >> 8));
                        }
                        *(uint4 *) (sK + key * GV_KSTR + b * QK8_0 + 8 * g) = *(const uint4 *) tmp;
                    }
#pragma unroll
                    for (int t = 0; t < QK8_0 / 2; ++t) {
                        const uint16_t u = raw_v(1 + t);
                        sVt[(b * QK8_0 + 2 * t + 0) * VTSTR + key] = (_Float16) (dv * (float) (int8_t) (u & 0xFF));
                        sVt[(b * QK8_0 + 2 * t + 1) * VTSTR + key] = (_Float16) (dv * (float) (int8_t) (u >> 8));
                    }
                }
            } else {
#pragma unroll
                for (int s = 0; s < NK; ++s) {
                    const int i = tid + s * nthreads;
                    if (i < KC * (GV_D / 8)) {
                        const int cc = i / KC, key = i % KC;
                        *(uint4 *) (sK + key * GV_KSTR + cc * 8) = make_uint4(fk[4 * s + 0], fk[4 * s + 1], fk[4 * s + 2], fk[4 * s + 3]);
#pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            const uint16_t hb = (uint16_t) (fv[4 * s + t / 2] >> (16 * (t % 2)));
                            *(uint16_t *) (sVt + (cc * 8 + t) * VTSTR + key) = hb;
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
                gv_compute_chunk<KC>(tile_active, sK, sVt, sP, smask, qa, o, m_run, l_run, row_ok, row_q, scale, li, kq);
            }
            __syncthreads();
        }
    } else
    for (int c = c_begin; c < c_end; ++c) {
        const int kbase = c * KC;

        // 1. mask chunk + skip test
        int any_live = 0;
        for (int i = tid; i < n_q * KC; i += nthreads) {
            const int q = i / KC;
            const int j = i % KC;
            const float mv = __half2float(*(const half *) (Mb + q * nb31 + (int64_t) (kbase + j) * sizeof(half)));
            smask[q * KC + j] = mv;
            any_live |= mv != -INFINITY;
        }
        if (!__syncthreads_or(any_live)) {
            continue;
        }

        // 2. K -> sK [key][d], V -> sVt [d][key]
        if constexpr (KV_Q8) {
            for (int i = tid; i < KC * (GV_D / QK8_0); i += nthreads) {
                const int b   = i / KC;
                const int key = i % KC;
                {
                    const char * blk = Kb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb11 + b * (int64_t) sizeof(block_q8_0);
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
                    const char * blk = Vb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb21 + b * (int64_t) sizeof(block_q8_0);
                    const float d = __half2float(*(const half *) blk);
                    const uint16_t * qs = (const uint16_t *) (blk + sizeof(half));
#pragma unroll
                    for (int t = 0; t < QK8_0 / 2; ++t) {
                        const uint16_t u = qs[t];
                        sVt[(b * QK8_0 + 2 * t + 0) * VTSTR + key] = (_Float16) (d * (float) (int8_t) (u & 0xFF));
                        sVt[(b * QK8_0 + 2 * t + 1) * VTSTR + key] = (_Float16) (d * (float) (int8_t) (u >> 8));
                    }
                }
            }
        } else {
            for (int i = tid; i < KC * (GV_D / 8); i += nthreads) {
                const int cc  = i / KC;
                const int key = i % KC;
                const uint4 kv = *(const uint4 *) (Kb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb11 + cc * 16);
                *(uint4 *) (sK + key * GV_KSTR + cc * 8) = kv;
                const uint4 vv = *(const uint4 *) (Vb + gv_kv_row<ROWS>(rows_s, kbase + key) * nb21 + cc * 16);
                const _Float16 * vh = (const _Float16 *) &vv;
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    sVt[(cc * 8 + t) * VTSTR + key] = vh[t];
                }
            }
        }
        __syncthreads();

        gv_compute_chunk<KC>(tile_active, sK, sVt, sP, smask, qa, o, m_run, l_run, row_ok, row_q, scale, li, kq);
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
        nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, nb31, nb33, ne33, rows, rows_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
}

// GGML_HIP_VATTN_RF=2 (C22 VATTN-ROOFLINE-2, lever 2): double-buffered LDS q8_0 verify kernel for the RF shapes (n_vseq 1,
// no row list, R <= 64 -> 4 waves, q8_0 K/V). KC = 16. sK / sVt / smask are ping-pong buffers; the raw q8_0 blocks are
// register-prefetched two chunks ahead of the compute: in iteration c the block-wide barrier publishes buffer c&1, then
// the raw registers (chunk c+1) are dequantized into buffer (c+1)&1, the global loads of chunk c+2 are issued, and chunk
// c is computed from buffer c&1. Buffer (c+1)&1 was last read by chunk c-1, which every wave finished before the barrier,
// so ONE barrier per chunk suffices (C21 / RF1: two). sP is per wave, so the P store -> P load hazard needs only a wave
// barrier. Thread roles: tid < 128 own one K block (key, b), tid >= 128 one V block (RF1: tid < 128 own both, 128 idle).
// LDS = 2 x (8,448 + 12,288 + 1,024) + 3,072 = 46,592 B -> 1 WG (4 waves) per CU; splits are sized for that round.
// Per-key arithmetic, dequant rounding and the merge are unchanged (deterministic split order).
//
// GGML_HIP_VATTN_RF=4 (lever 4, KDIRECT): no K in LDS. The MFMA k order is free, so d is permuted consistently in Q and
// K (logical k = 16s + 4kq + t <-> d = 64kq + 4s + t): lane (key li, kq) loads q8_0 blocks 2kq and 2kq+1 of its key (68 B,
// 17 dwords) straight into registers and dequantizes them into the B fragment of each of the 16 QK^T MFMAs. The K load of
// chunk c+1 is issued right after chunk c's QK^T, so it overlaps softmax + PV. V stays in LDS, transposed, single-buffered:
// LDS = 12,288 + 1,024 + 3,072 = 16,384 B -> 4 WG (16 waves) per CU, and splits are sized for one 4-WG/CU round. With a
// single V buffer the chunk keeps two block barriers (latency is hidden by the 4 resident WGs instead); the V/mask raw
// registers are still loaded one chunk ahead, before the publishing barrier. Every active wave reads the K chunk (L2/L1
// reuse, 4x K requests per WG, unchanged HBM bytes). Requires 4-byte-aligned K rows (host-checked).
#define GV_RF_KC    16
#define GV_RF_VTSTR (GV_RF_KC + 8)
#define GV_RF_PSTR  (GV_RF_KC + 8)
#define GV_RF_NT    (64 * GV_MIN_WAVES)

static size_t gv_rf_smem_bytes(const bool kdirect, const bool db) {
    const size_t buf = sizeof(_Float16) * (size_t) ((kdirect ? 0 : GV_RF_KC * GV_KSTR) + GV_D * GV_RF_VTSTR) +
                       sizeof(float) * (size_t) (GV_MAX_NQ * GV_RF_KC);
    return (db ? 2 : 1) * buf + sizeof(_Float16) * (size_t) (GV_MIN_WAVES * 16 * GV_RF_PSTR);
}

#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
// LDS written and read back by the same wave: in-order per wave, only the compiler must not reorder across this point
static __device__ __forceinline__ void gv_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#endif

template <bool KDIRECT, bool DB, int MINB>
static __global__ void __launch_bounds__(GV_RF_NT, MINB) gv_rf_kernel(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, float * __restrict__ Opart, float2 * __restrict__ MLpart,
        const float scale, const int n_q, const int gqa, const int n_head_kv,
        const int chunks_per_split, const int n_chunks,
        const int64_t nb01, const int64_t nb02, const int64_t nb11, const int64_t nb12,
        const int64_t nb21, const int64_t nb22, const int64_t nb31) {
#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
    constexpr int KC    = GV_RF_KC;
    constexpr int VTSTR = GV_RF_VTSTR;
    constexpr int PSTR  = GV_RF_PSTR;
    constexpr int SK_H  = KDIRECT ? 0 : KC * GV_KSTR;            // halves
    constexpr int BUF_B = 2 * (SK_H + GV_D * VTSTR) + 4 * GV_MAX_NQ * KC; // bytes per buffer
    extern __shared__ char gv_smem[];
    auto sK_of   = [&](const int b) __attribute__((always_inline)) { return (_Float16 *) (gv_smem + b * BUF_B); };
    auto sVt_of  = [&](const int b) __attribute__((always_inline)) { return (_Float16 *) (gv_smem + b * BUF_B) + SK_H; };
    auto smk_of  = [&](const int b) __attribute__((always_inline)) { return (float *) ((_Float16 *) (gv_smem + b * BUF_B) + SK_H + GV_D * VTSTR); };

    const int split = blockIdx.x;
    const int hkv   = blockIdx.y;
    const int tid   = threadIdx.x;
    const int w     = tid / 64;
    const int l     = tid % 64;
    const int li    = l % 16;
    const int kq    = l / 16;

    const int R      = gqa * n_q;
    const int ntiles = (R + 15) / 16;
    const bool tile_active = w < ntiles;
    _Float16 * sP = (_Float16 *) (gv_smem + (DB ? 2 : 1) * BUF_B) + w * 16 * PSTR;

    const char * Kb = K + hkv * nb12;
    const char * Vb = V + hkv * nb22;
    const char * Mb = mask;

    gv_halfx4 qa[GV_D / 16];
    {
        const int r = 16 * w + li;
        const bool valid = tile_active && r < R;
        const int h = valid ? r / n_q : 0;
        const int q = valid ? r % n_q : 0;
        const float * Qr = (const float *) (Q + (int64_t) (hkv * gqa + h) * nb02 + (int64_t) q * nb01);
#pragma unroll
        for (int s = 0; s < GV_D / 16; ++s) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                // KDIRECT: MFMA k index permuted consistently in Q and K: logical k = 16s+4kq+t <-> d = 64kq+4s+t
                const int d = KDIRECT ? 64 * kq + 4 * s + t : 16 * s + 4 * kq + t;
                qa[s][t] = valid ? (_Float16) Qr[d] : (_Float16) 0.0f;
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
    int   row_q[4];
    bool  row_ok[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        m_run[v] = -FLT_MAX / 2.0f;
        l_run[v] = 0.0f;
        const int r = 16 * w + 4 * kq + v;
        row_ok[v] = tile_active && r < R;
        row_q[v]  = row_ok[v] ? r % n_q : 0;
    }

    const int c_begin = split * chunks_per_split;
    const int c_end   = min(c_begin + chunks_per_split, n_chunks);

    // raw q8_0 block of this thread's role (34 B as 8 dwords + tail), mask halfword
    struct __attribute__((packed, aligned(2))) gv_q8_raw { uint32_t w[8]; uint16_t tail; };
    const bool is_v    = KDIRECT ? true : tid >= 128;
    const bool has_blk = KDIRECT ? tid < 128 : true;
    const int  bi      = tid & 127;
    const int  bb      = bi / KC;  // q8_0 block index along d
    const int  bkey    = bi % KC;
    uint32_t qw[8];
    uint16_t qt = 0;
    uint16_t rm = 0;

    auto load_raw = [&](const int c) __attribute__((always_inline)) {
        const int kbase = c * KC;
        if (tid < n_q * KC) {
            rm = *(const uint16_t *) (Mb + (tid / KC) * nb31 + (int64_t) (kbase + tid % KC) * sizeof(half));
        }
        if (has_blk) {
            const char * p = is_v ? Vb + (int64_t) (kbase + bkey) * nb21 : Kb + (int64_t) (kbase + bkey) * nb11;
            const gv_q8_raw * qb = (const gv_q8_raw *) (p + bb * (int64_t) sizeof(block_q8_0));
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                qw[t] = qb->w[t];
            }
            qt = qb->tail;
        }
    };
    // KDIRECT: this lane's 2 q8_0 blocks of key li (blocks 2kq, 2kq+1), raw
    uint32_t kr[KDIRECT ? 17 : 1];
    auto kload = [&](const int c) __attribute__((always_inline)) {
        if constexpr (KDIRECT) {
            const uint32_t * pk = (const uint32_t *) (Kb + (int64_t) (c * KC + li) * nb11 + kq * 2 * (int64_t) sizeof(block_q8_0));
#pragma unroll
            for (int t = 0; t < 17; ++t) {
                kr[t] = pk[t];
            }
        } else {
            GGML_UNUSED(c);
        }
    };
    auto raw_h = [&](const int k) __attribute__((always_inline)) -> uint16_t {
        return k == QK8_0 / 2 ? qt : (uint16_t) (qw[k / 2] >> (16 * (k % 2)));
    };
    auto store_raw = [&](const int b) __attribute__((always_inline)) -> int {
        int any = 0;
        uint32_t x0 = qt, x1 = rm;
        asm volatile("" : "+v"(x0), "+v"(x1));
        qt = (uint16_t) x0; rm = (uint16_t) x1;
        if (tid < n_q * KC) {
            const float mv = __half2float(__ushort_as_half(rm));
            smk_of(b)[tid] = mv;
            any = mv != -INFINITY;
        }
        if (has_blk) {
            const float dq = __half2float(__ushort_as_half(raw_h(0)));
            if (!is_v) {
                _Float16 * sK = sK_of(b);
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    _Float16 tmp[8];
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uint16_t u = raw_h(1 + 4 * g + t);
                        tmp[2 * t + 0] = (_Float16) (dq * (float) (int8_t) (u & 0xFF));
                        tmp[2 * t + 1] = (_Float16) (dq * (float) (int8_t) (u >> 8));
                    }
                    *(uint4 *) (sK + bkey * GV_KSTR + bb * QK8_0 + 8 * g) = *(const uint4 *) tmp;
                }
            } else {
                _Float16 * sVt = sVt_of(b);
#pragma unroll
                for (int t = 0; t < QK8_0 / 2; ++t) {
                    const uint16_t u = raw_h(1 + t);
                    sVt[(bb * QK8_0 + 2 * t + 0) * VTSTR + bkey] = (_Float16) (dq * (float) (int8_t) (u & 0xFF));
                    sVt[(bb * QK8_0 + 2 * t + 1) * VTSTR + bkey] = (_Float16) (dq * (float) (int8_t) (u >> 8));
                }
            }
        }
        return any;
    };

    // S = Q K^T, online softmax, P -> sP (wave-local), O = O*alpha + P V, for one 16-key chunk from buffer b
    auto compute = [&](const int b, const int c) __attribute__((always_inline)) {
        float alpha[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const float * smask = smk_of(b);
        if (tile_active) {
            gv_floatx4 S = (gv_floatx4) {0.0f, 0.0f, 0.0f, 0.0f};
            if constexpr (KDIRECT) {
                // block 0: d at bytes 0-1, qs at 2..33; block 1: d at 34-35, qs at 36..67
                const float d0 = __half2float(__ushort_as_half((uint16_t) kr[0]));
                const float d1 = __half2float(__ushort_as_half((uint16_t) (kr[8] >> 16)));
#pragma unroll
                for (int s = 0; s < GV_D / 16; ++s) {
                    gv_halfx4 kb;
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const int e   = 4 * s + t;
                        const int off = e < QK8_0 ? 2 + e : 36 + (e - QK8_0);
                        const int8_t q = (int8_t) (kr[off >> 2] >> (8 * (off & 3)));
                        kb[t] = (_Float16) ((e < QK8_0 ? d0 : d1) * (float) q);
                    }
                    S = __builtin_amdgcn_mfma_f32_16x16x16f16(qa[s], kb, S, 0, 0, 0);
                }
                if (c + 1 < c_end) {
                    kload(c + 1); // in flight during softmax + PV of this chunk
                }
            } else {
                const _Float16 * sK = sK_of(b);
#pragma unroll
                for (int s = 0; s < GV_D / 16; ++s) {
                    const gv_halfx4 kb = *(const gv_halfx4 *) (sK + li * GV_KSTR + 16 * s + 4 * kq);
                    S = __builtin_amdgcn_mfma_f32_16x16x16f16(qa[s], kb, S, 0, 0, 0);
                }
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float mv = row_ok[v] ? smask[row_q[v] * KC + li] : -INFINITY;
                S[v] = S[v] * scale + mv;
                float rmax = S[v];
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) {
                    rmax = fmaxf(rmax, __shfl_xor(rmax, off, 64));
                }
                rmax = fmaxf(rmax, -FLT_MAX / 2.0f);
                const float m_new = fmaxf(m_run[v], rmax);
                alpha[v] = __expf(m_run[v] - m_new);
                const float p = __expf(S[v] - m_new);
                sP[(4 * kq + v) * PSTR + li] = (_Float16) p;
                float rsum = p;
#pragma unroll
                for (int off = 1; off < 16; off <<= 1) {
                    rsum += __shfl_xor(rsum, off, 64);
                }
                l_run[v] = l_run[v] * alpha[v] + rsum;
                m_run[v] = m_new;
            }
        }
        gv_wave_sync();
        if (tile_active) {
            const _Float16 * sVt = sVt_of(b);
            const gv_halfx4 pa = *(const gv_halfx4 *) (sP + li * PSTR + 4 * kq);
#pragma unroll
            for (int nb = 0; nb < GV_D / 16; ++nb) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    o[nb][v] *= alpha[v];
                }
                const gv_halfx4 vb = *(const gv_halfx4 *) (sVt + (nb * 16 + li) * VTSTR + 4 * kq);
                o[nb] = __builtin_amdgcn_mfma_f32_16x16x16f16(pa, vb, o[nb], 0, 0, 0);
            }
        }
        gv_wave_sync(); // sP is rewritten by the next chunk of this wave
    };

    if constexpr (DB) {
        int any = 0;
        if (c_begin < c_end) {
            load_raw(c_begin);
            any = store_raw(0);
        }
        if (c_begin + 1 < c_end) {
            load_raw(c_begin + 1);
        }
        if (KDIRECT && tile_active && c_begin < c_end) {
            kload(c_begin);
        }
        for (int c = c_begin; c < c_end; ++c) {
            const int live = __syncthreads_or(any); // the one barrier of the chunk: publishes buffer c&1
            if (c + 1 < c_end) {
                any = store_raw((c + 1 - c_begin) & 1);
            }
            if (c + 2 < c_end) {
                load_raw(c + 2);
            }
            if (live) {
                compute((c - c_begin) & 1, c);
            } else if (KDIRECT && tile_active && c + 1 < c_end) {
                kload(c + 1);
            }
        }
    } else {
        if (c_begin < c_end) {
            load_raw(c_begin);
            if (KDIRECT && tile_active) {
                kload(c_begin);
            }
        }
        for (int c = c_begin; c < c_end; ++c) {
            const int any = store_raw(0);
            if (c + 1 < c_end) {
                load_raw(c + 1);
            }
            const int live = __syncthreads_or(any);
            if (live) {
                compute(0, c);
            } else if (KDIRECT && tile_active && c + 1 < c_end) {
                kload(c + 1);
            }
            __syncthreads();
        }
    }

    if (tile_active) {
        const int64_t base_row = ((int64_t) split * n_head_kv + hkv) * R;
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
    GGML_UNUSED_VARS(Q, K, V, mask, Opart, MLpart, scale, n_q, gqa, n_head_kv, chunks_per_split, n_chunks,
        nb01, nb02, nb11, nb12, nb21, nb22, nb31);
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

static void gv_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int32_t * rows, const int64_t rows_s1) {
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
    const bool rf       = gv_vattn_rf() && pipe && nwaves == GV_MIN_WAVES && n_vseq == 1 && rows == nullptr &&
                          K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
    const bool cut      = (gv_lds_cut() || rf) && pipe && nwaves == GV_MIN_WAVES;
    // lever 2 (RF=2): dedicated double-buffered kernel, 1 WG/CU; lever 1 (RF=1): LDS-cut kernel, 2 WG/CU
    const bool rf2      = rf && gv_vattn_rf_level() == 2;
    // lever 4 (RF=4): K register-direct kernel, 4 WG/CU; needs 4-byte-aligned K rows for the dword loads
    const bool rf4      = rf && gv_vattn_rf_level() == 4 && K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 &&
                          (uintptr_t) K->data % 4 == 0;
    const int  rf_wgcu  = rf4 ? 4 : rf2 ? 1 : 2;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int kc       = cut ? GV_KC_CUT : GV_KC;
    const int n_chunks = n_kv / kc;
    const int nsm      = ggml_cuda_info().devices[ctx.device].nsm;
    int n_splits = gv_splits_env();
    if (n_splits <= 0) {
        // grouped: each group streams ~1/n_grp of the KV (live-range trim), so size the splits per real sequence
        const int base = n_head_kv * n_seq;
        n_splits = (2 * nsm + base - 1) / base;
        n_splits = std::min(n_splits, std::max(1, n_kv / 256));
        n_splits = std::min(n_splits, 256);
    }
    if (rf && gv_splits_env() <= 0) {
        // one resident round at 2 WG/CU (see gv_vattn_rf)
        int s_rf = std::max(1, (rf_wgcu * nsm) / n_head_kv);
        // partial-O traffic cap: splits * R * (D*4 + 8) <= 0.5 * n_kv * 2 * (K row bytes per head)
        const int64_t kv_bytes_head = (int64_t) n_kv * (int64_t) (ggml_row_size(K->type, GV_D) + ggml_row_size(V->type, GV_D));
        const int64_t cap_traffic   = std::max<int64_t>(1, kv_bytes_head / (2 * (int64_t) R * (GV_D * 4 + 8)));
        s_rf = (int) std::min<int64_t>(s_rf, cap_traffic);
        s_rf = std::min(s_rf, std::max(1, n_chunks / 8));
        n_splits = s_rf;
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
    const size_t smem = rf4 ? gv_rf_smem_bytes(true, false) : rf2 ? gv_rf_smem_bytes(false, true) : gv_smem_bytes(nwaves, kc);

    const int ne33 = (int) mask->ne[3];
    const bool q8 = K->type == GGML_TYPE_Q8_0;
    const bool late = pipe && q8 && gv_prefetch_late();
    // ROWS (KPF-41 in-place): K/V are the unified cache itself, one slice per ne[3] -> no per-slice K/V offset
    const int64_t nb13 = rows ? 0 : K->nb[3];
    const int64_t nb23 = rows ? 0 : V->nb[3];
#define GV_LAUNCH_K(Q8, PIPE_, LATE_, ROWS_, KC_) \
    gv_fattn_kernel<Q8, PIPE_, LATE_, ROWS_, KC_><<<grid, block, smem, stream>>>( \
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data, \
        Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, n_grp, G, chunks_per_split, n_chunks, \
        Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], nb13, V->nb[1], V->nb[2], nb23, \
        mask->nb[1], mask->nb[3], ne33, rows, rows_s1)
#define GV_LAUNCH(Q8, PIPE_, LATE_, ROWS_) do { if (cut) { GV_LAUNCH_K(Q8, PIPE_, LATE_, ROWS_, GV_KC_CUT); } else { GV_LAUNCH_K(Q8, PIPE_, LATE_, ROWS_, GV_KC); } } while (0)
#define GV_LAUNCH_R(Q8, PIPE_, LATE_) do { if (rows) { GV_LAUNCH(Q8, PIPE_, LATE_, true); } else { GV_LAUNCH(Q8, PIPE_, LATE_, false); } } while (0)
    if (rf2) {
        gv_rf_kernel<false, true, 1><<<grid, block, smem, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, chunks_per_split, n_chunks,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1]);
    } else if (rf4) {
        gv_rf_kernel<true, false, 4><<<grid, block, smem, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, chunks_per_split, n_chunks,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1]);
    } else if (late) {
        GV_LAUNCH_R(true, true, true);
    } else if (pipe) {
        if (q8) {
            GV_LAUNCH_R(true, true, false);
        } else {
            GV_LAUNCH_R(false, true, false);
        }
    } else {
        if (q8) {
            GV_LAUNCH_R(true, false, false);
        } else {
            GV_LAUNCH_R(false, false, false);
        }
    }
#undef GV_LAUNCH_R
#undef GV_LAUNCH
#undef GV_LAUNCH_K
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
    static int n_rows_calls = 0;
    if (rows && n_rows_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GGML_CUDA_FA_ROWS_INPLACE in-place row-list K/V read ran (call 1)\n");
    }
    static int n_rf_calls = 0;
    if (rf && n_rf_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GGML_HIP_VATTN_RF roofline launch ran (splits=%d R=%d smem=%zu)\n", n_splits, R, smem);
    }
    static int n_rf4_calls = 0;
    if (rf4 && n_rf4_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GGML_HIP_VATTN_RF=4 K-register-direct launch ran (splits=%d R=%d smem=%zu)\n", n_splits, R, smem);
    }
    static int n_rf2_calls = 0;
    if (rf2 && n_rf2_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GGML_HIP_VATTN_RF=2 double-buffered LDS launch ran (splits=%d R=%d smem=%zu)\n", n_splits, R, smem);
    }
    static int n_cut_calls = 0;
    if (cut && n_cut_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GGML_CUDA_FA_GQA_LDS_CUT 16-key chunk variant ran (smem=%zu)\n", smem);
    }
    static int n_late_calls = 0;
    if (late && n_late_calls++ == 0) {
        GGML_LOG_WARN("ggml_cuda_fattn_gqa_verify: GGML_HIP_GV_PREFETCH_LATE prefetch-overlap path ran (call 1)\n");
    }
    if (gv_debug() && n_calls <= 8) {
        GGML_LOG_WARN("%s: %s n_q=%d gqa=%d n_head_kv=%d n_kv=%d n_seq=%d type=%s splits=%d waves=%d smem=%zu pipe=%d cut=%d kc=%d\n",
            __func__, dst->name, n_q, gqa, n_head_kv, n_kv, n_seq, ggml_type_name(K->type), n_splits, nwaves, smem, (int) pipe, (int) cut, kc);
    }
}

void ggml_cuda_fattn_gqa_verify(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    gv_run(ctx, dst, nullptr, 0);
}

// KPF-41 FAROWS-INPLACE: dst is the gathered [D, n_rows, H_kv, n_slices] view of the UP-FAROWS-HIP plan, except that
// src[1]/src[2] point at the unified K/V cache itself (ne[1] = n_rows, nb[1]/nb[2] of the cache) and `rows` is the
// device row list [n_rows, n_slices] (row stride rows_s1). The caller gathers only the mask.
void ggml_cuda_fattn_gqa_verify_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int32_t * rows, const int64_t rows_s1) {
    GGML_ASSERT(rows != nullptr);
    gv_run(ctx, dst, rows, rows_s1);
}
