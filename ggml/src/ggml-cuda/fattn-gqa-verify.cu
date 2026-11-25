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
static bool gv_vattn_rf() {
    static const int v = gv_env_int("GGML_HIP_VATTN_RF", 0);
    return v != 0 && v != 7;
}

// GGML_HIP_VATTN_RF=7 (C22 VATTN-ROOFLINE-5, default OFF): wave-independent split-K streaming kernel (gv7_kernel below).
// Selected only for the single-sequence q8_0 shapes (n_vseq 1, no row list, R = gqa*n_q <= 64, aligned views); every
// other call keeps the C21 path. RF=7 does not enable RF1.
static bool gv_vattn_rf7() {
    static const bool v = gv_env_int("GGML_HIP_VATTN_RF", 0) == 7;
    return v;
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

// ---------------------------------------------------------------------------------------------------------------------
// GGML_HIP_VATTN_RF=7: wave-independent split-K streaming verify attention for q8_0 K/V (design:
// /mnt/raid0/llm/tmp/vattn-rf7-design-20261125.md). No LDS and no barrier inside the key loop: each wave streams its own
// 16-key chunks global -> VGPR straight into the MFMA operands, with the next chunk's loads in flight while it computes.
//   S^T[key][row] = K Q^T : A = K  (lane l%16 = key, kq = l/16 owns d in [64kq, 64kq+64) = two whole q8_0 blocks, 68 B,
//                                   4-byte aligned -> 4 dwordx4 + 1 dword), B = Q^T (Q fragments resident).
//   The S^T D layout (lane l%16 = row, 4 consecutive keys per lane) is exactly the B operand of
//   O^T[d][row] = V^T P^T, so P never leaves the registers. A = V^T with the d permutation d = 16*(l%16) + n for output
//   tile n: lane l%16 = i reads 16 contiguous bytes (one half of q8_0 block i/2) of each of its 4 keys.
//   Every O^T value a lane holds belongs to its row l%16, so the online-softmax rescale is a per-lane scalar.
// A workgroup is 4 waves on one (row tile, split, KV head) sharing the Q tile in LDS (read-only after one barrier); wave
// w takes chunks cb + w, cb + w + 4, ...; per wave one K set (issued right after the previous QK^T) and one V set
// (issued right after the previous PV) are in flight. The 4 wave partials are combined once per split in LDS in a fixed
// order, then a parallel merge combines the splits in index order: deterministic run to run.
#define GV7_WAVES 4
#define GV7_KC    16

#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
typedef uint32_t gv7_u32x4 __attribute__((ext_vector_type(4)));
typedef gv7_u32x4 gv7_u32x4_a4 __attribute__((aligned(4)));
typedef uint32_t gv7_u32x2 __attribute__((ext_vector_type(2)));
typedef _Float16 gv7_halfx2 __attribute__((ext_vector_type(2)));

// K group of one chunk (issued right after the previous chunk's QK^T) and V group (issued right after its PV): one
// register set each, so a wave holds 17 + 2 + 24 raw dwords instead of two full chunks.
struct gv7_kraw {
    gv7_u32x4 k[4];   // K bytes [68kq, 68kq+64) of key c0 + l%16
    uint32_t  k16;    // K bytes [68kq+64, 68kq+68)
    gv7_u32x2 m;      // 4 f16 mask values (keys c0 + 4kq .. +3, query row of lane l%16)
};
struct gv7_vraw {
    uint32_t  vs[4];  // dword holding the V block scale, keys c0 + 4kq + t
    gv7_u32x4 v[4];   // V 16-byte window (dword aligned) of the lane's half block
    uint32_t  v4[4];  // 5th window dword (alignment spill)
};

// uniform (SGPR) chunk bases + 32-bit lane offsets, so the loads use the saddr form and hold no 64-bit VGPR pointers
static __device__ __forceinline__ void gv7_load_k(gv7_kraw & r, const char * __restrict__ kc, const uint32_t koff,
        const char * __restrict__ mc, const uint32_t moff) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r.k[i] = *(const gv7_u32x4_a4 *) (kc + koff + 16 * i);
    }
    r.k16 = *(const uint32_t *) (kc + koff + 64);
    r.m   = *(const gv7_u32x2 *) (mc + moff);
}

static __device__ __forceinline__ void gv7_load_v(gv7_vraw & r, const char * __restrict__ vc, const int64_t nb21,
        const uint32_t vsoff, const uint32_t vwoff, const uint32_t v4off) {
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const char * p = vc + t * nb21;
        r.vs[t] = *(const uint32_t *) (p + vsoff);
        r.v[t]  = *(const gv7_u32x4_a4 *) (p + vwoff);
        r.v4[t] = *(const uint32_t *) (p + v4off);
    }
}

// 4 signed bytes of x -> 4 halves * s2 (exact integer via the 0x6400 magic, one f16 rounding in the scale product)
static __device__ __forceinline__ gv_halfx4 gv7_deq4(uint32_t x, const gv7_halfx2 s2) {
    x ^= 0x80808080u;
    const uint32_t lo = __builtin_amdgcn_perm(0x64646464u, x, 0x04010400u);
    const uint32_t hi = __builtin_amdgcn_perm(0x64646464u, x, 0x04030402u);
    const gv7_halfx2 off = {(_Float16) 1152.0f, (_Float16) 1152.0f};
    const gv7_halfx2 a = (__builtin_bit_cast(gv7_halfx2, lo) - off) * s2;
    const gv7_halfx2 b = (__builtin_bit_cast(gv7_halfx2, hi) - off) * s2;
    return (gv_halfx4) {a.x, a.y, b.x, b.y};
}

// S^T = K Q^T over d (16 MFMA steps, d = 64kq + 4s + t); Q^T fragments read from LDS (row stride GV7_QSTR halves)
static __device__ __forceinline__ gv_floatx4 gv7_qk(const gv7_kraw & r, const _Float16 * __restrict__ sq) {
    uint32_t kd[17];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        kd[i] = r.k[i / 4][i % 4];
    }
    kd[16] = r.k16;
    const _Float16 ks0 = __builtin_bit_cast(_Float16, (uint16_t) (kd[0] & 0xffffu));
    const _Float16 ks1 = __builtin_bit_cast(_Float16, (uint16_t) (kd[8] >> 16));
    const gv7_halfx2 ks0x2 = {ks0, ks0};
    const gv7_halfx2 ks1x2 = {ks1, ks1};
    gv_floatx4 S = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const gv_halfx4 qb = *(const gv_halfx4 *) (sq + 4 * s);
        const uint32_t x = s < 8 ? __builtin_amdgcn_alignbyte(kd[s + 1], kd[s], 2) : kd[s + 1];
        const gv_halfx4 ka = gv7_deq4(x, s < 8 ? ks0x2 : ks1x2);
        S = __builtin_amdgcn_mfma_f32_16x16x16f16(ka, qb, S, 0, 0, 0);
    }
    return S;
}

// online softmax over the lane's 4 keys of row l%16; returns P^T (the PV B operand) and the O rescale factor
static __device__ __forceinline__ gv_halfx4 gv7_softmax(const gv_floatx4 S, const gv7_u32x2 mraw, float & m_run,
        float & l_run, const float scale, float & alpha) {
    const gv7_halfx2 mk01 = __builtin_bit_cast(gv7_halfx2, mraw.x);
    const gv7_halfx2 mk23 = __builtin_bit_cast(gv7_halfx2, mraw.y);
    float sv[4];
    sv[0] = S[0] * scale + (float) mk01.x;
    sv[1] = S[1] * scale + (float) mk01.y;
    sv[2] = S[2] * scale + (float) mk23.x;
    sv[3] = S[3] * scale + (float) mk23.y;
    float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
    mx = fmaxf(mx, __shfl_xor(mx, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 32));
    const float m_new = fmaxf(m_run, mx);          // m_run starts at -FLT_MAX: finite even for an all -inf chunk
    alpha = __expf(m_run - m_new);
    m_run = m_new;
    float p[4];
    float psum = 0.0f;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        p[v] = __expf(sv[v] - m_new);
        psum += p[v];
    }
    l_run = l_run * alpha + psum;
    return (gv_halfx4) {(_Float16) p[0], (_Float16) p[1], (_Float16) p[2], (_Float16) p[3]};
}

// O^T = V^T P^T, d = 16*(l%16) + n
static __device__ __forceinline__ void gv7_pv(const gv7_vraw & r, const gv_halfx4 pb, const float alpha,
        gv_floatx4 (&o)[GV_D / 16], const int vshift, const int sshift) {
    uint32_t vq[4][4];
    uint16_t sc[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        vq[t][0] = __builtin_amdgcn_alignbyte(r.v[t][1], r.v[t][0], vshift) ^ 0x80808080u;
        vq[t][1] = __builtin_amdgcn_alignbyte(r.v[t][2], r.v[t][1], vshift) ^ 0x80808080u;
        vq[t][2] = __builtin_amdgcn_alignbyte(r.v[t][3], r.v[t][2], vshift) ^ 0x80808080u;
        vq[t][3] = __builtin_amdgcn_alignbyte(r.v4[t],   r.v[t][3], vshift) ^ 0x80808080u;
        sc[t] = (uint16_t) ((r.vs[t] >> sshift) & 0xffffu);
    }
    const gv7_halfx2 vs01 = __builtin_bit_cast(gv7_halfx2, (uint32_t) sc[0] | ((uint32_t) sc[1] << 16));
    const gv7_halfx2 vs23 = __builtin_bit_cast(gv7_halfx2, (uint32_t) sc[2] | ((uint32_t) sc[3] << 16));
    const gv7_halfx2 off = {(_Float16) 1152.0f, (_Float16) 1152.0f};
#pragma unroll
    for (int n = 0; n < GV_D / 16; ++n) {
        const int w = n / 4;
        const int b = n % 4;
        const uint32_t sel = (uint32_t) b | 0x0c00u | ((uint32_t) (4 + b) << 16) | 0x0c000000u;
        const uint32_t x01 = __builtin_amdgcn_perm(vq[1][w], vq[0][w], sel) | 0x64006400u;
        const uint32_t x23 = __builtin_amdgcn_perm(vq[3][w], vq[2][w], sel) | 0x64006400u;
        const gv7_halfx2 a01 = (__builtin_bit_cast(gv7_halfx2, x01) - off) * vs01;
        const gv7_halfx2 a23 = (__builtin_bit_cast(gv7_halfx2, x23) - off) * vs23;
        const gv_halfx4 va = {a01.x, a01.y, a23.x, a23.y};
        o[n] *= alpha;
        o[n] = __builtin_amdgcn_mfma_f32_16x16x16f16(va, pb, o[n], 0, 0, 0);
    }
}
#endif // defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)

#define GV7_QSTR  (GV_D + 8)     // sQ row stride (halves): 528 B, 16-byte aligned rows, 2-way banked fragment reads
#define GV7_SLOT  ((GV_D / 16) * 4 * 64)   // floats: one wave's O^T partial (16 KB)

// grid (row tile, split, KV head), 4 waves, 2 WG per CU (8 independent waves per CU; at the 3 WG/CU budget of 168
// VGPRs the compiler spilled 44). Opart rows: ((split * n_head_kv + hkv) * R + r), MLpart = (max, sum).
static __global__ void __launch_bounds__(64 * GV7_WAVES, 2) gv7_kernel(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, float * __restrict__ Opart, float2 * __restrict__ MLpart,
        const float scale, const int n_q, const int gqa, const int n_head_kv, const int chunks_per_split, const int n_chunks,
        const int64_t nb01, const int64_t nb02, const int64_t nb11, const int64_t nb12,
        const int64_t nb21, const int64_t nb22, const int64_t nb31) {
#if defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
    // sQ (16 x GV7_QSTR f16, 8.4 KB) during the key loop; one O slot (16 KB) for the end-of-split combine; sml apart
    __shared__ float sbuf[GV7_SLOT];
    __shared__ float sml[GV7_WAVES * 32];
    _Float16 * sQ = (_Float16 *) sbuf;

    const int tile  = blockIdx.x;
    const int split = blockIdx.y;
    const int hkv   = blockIdx.z;
    const int tid   = threadIdx.x;
    const int w     = __builtin_amdgcn_readfirstlane(tid / 64);
    const int l     = tid % 64;
    const int li    = l % 16;
    const int kq    = l / 16;
    const int R     = gqa * n_q;

    // Q tile -> LDS as f16 [row][d]: thread -> row tid/16, 16 d at 16*(tid%16)
    {
        const int rr = tid / 16;
        const int r  = 16 * tile + rr;
        const bool ok = r < R;
        const int h = ok ? r / n_q : 0;
        const int q = ok ? r % n_q : 0;
        const float4 * Qr = (const float4 *) (Q + (int64_t) (hkv * gqa + h) * nb02 + (int64_t) q * nb01) + 4 * (tid % 16);
        float4 f[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            f[i] = ok ? Qr[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        gv_halfx4 hq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            hq[i] = (gv_halfx4) {(_Float16) f[i].x, (_Float16) f[i].y, (_Float16) f[i].z, (_Float16) f[i].w};
        }
        gv_halfx4 * dq = (gv_halfx4 *) (sQ + rr * GV7_QSTR + 16 * (tid % 16));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dq[i] = hq[i];
        }
    }
    __syncthreads();
    const _Float16 * sq = sQ + li * GV7_QSTR + 64 * kq;   // B operand of lane (row li, kq): + 4s per MFMA step

    gv_floatx4 o[GV_D / 16];
#pragma unroll
    for (int n = 0; n < GV_D / 16; ++n) {
        o[n] = (gv_floatx4) {0.0f, 0.0f, 0.0f, 0.0f};
    }
    float m_run = -FLT_MAX;
    float l_run = 0.0f;

    const int  r     = 16 * tile + li;
    const bool valid = r < R;
    const int  qm    = valid ? r % n_q : 0;
    // per-lane 32-bit byte offsets (the chunk / head part of every address is uniform)
    const int vb     = li / 2;                       // q8_0 block of this lane's 16 d values
    const int vh     = li % 2;
    const int va     = 34 * vb + 2 + 16 * vh;        // first qs byte
    const int vshift = va & 3;                       // 0 or 2
    const int sshift = 8 * ((34 * vb) & 3);          // 0 or 16
    const uint32_t vrow  = (uint32_t) (4 * kq) * (uint32_t) nb21;
    const uint32_t vsoff = vrow + (uint32_t) ((34 * vb) & ~3);
    const uint32_t vwoff = vrow + (uint32_t) (va & ~3);
    const uint32_t v4off = vwoff + (vshift ? 16u : 12u); // never past the 272 B row
    const uint32_t koff  = (uint32_t) li * (uint32_t) nb11 + 68u * (uint32_t) kq;
    const uint32_t moff  = (uint32_t) qm * (uint32_t) nb31 + 8u * (uint32_t) kq;

    const char * Kh = K + (int64_t) hkv * nb12;
    const char * Vh = V + (int64_t) hkv * nb22;
    const int cb = split * chunks_per_split;
    const int ce = min(cb + chunks_per_split, n_chunks);

#define GV7_KC_PTR(C_) (Kh + (int64_t) (C_) * GV7_KC * nb11)
#define GV7_VC_PTR(C_) (Vh + (int64_t) (C_) * GV7_KC * nb21)
#define GV7_MC_PTR(C_) (mask + (int64_t) (C_) * GV7_KC * 2)
    gv7_kraw rk;
    gv7_vraw rv;
    int c = cb + w;
    if (c < ce) {
        gv7_load_k(rk, GV7_KC_PTR(c), koff, GV7_MC_PTR(c), moff);
        gv7_load_v(rv, GV7_VC_PTR(c), nb21, vsoff, vwoff, v4off);
    }
    for (; c < ce; c += GV7_WAVES) {
        const int cn = c + GV7_WAVES;
        const gv_floatx4 S = gv7_qk(rk, sq);
        const gv7_u32x2 mraw = rk.m;
        if (cn < ce) {
            gv7_load_k(rk, GV7_KC_PTR(cn), koff, GV7_MC_PTR(cn), moff);
        }
        float alpha;
        const gv_halfx4 pb = gv7_softmax(S, mraw, m_run, l_run, scale, alpha);
        gv7_pv(rv, pb, alpha, o, vshift, sshift);
        if (cn < ce) {
            gv7_load_v(rv, GV7_VC_PTR(cn), nb21, vsoff, vwoff, v4off);
        }
    }
#undef GV7_KC_PTR
#undef GV7_VC_PTR
#undef GV7_MC_PTR

    // row sum over the 4 kq lane groups (m_run is already uniform across them)
    l_run += __shfl_xor(l_run, 16);
    l_run += __shfl_xor(l_run, 32);

    // ---- combine the 4 wave partials of this workgroup in a fixed order ((O3 + O2) + O1) + O0 -> deterministic
    if (kq == 0) {
        sml[w * 32 + li]      = m_run;
        sml[w * 32 + 16 + li] = l_run;
    }
    __syncthreads();   // also: every wave is past its last sQ read, so sbuf may now hold O
    float M = -FLT_MAX;
#pragma unroll
    for (int u = 0; u < GV7_WAVES; ++u) {
        if (sml[u * 32 + 16 + li] > 0.0f) {
            M = fmaxf(M, sml[u * 32 + li]);
        }
    }
    float Lt = 0.0f;
#pragma unroll
    for (int u = GV7_WAVES - 1; u >= 0; --u) {
        const float lu = sml[u * 32 + 16 + li];
        Lt += lu > 0.0f ? __expf(sml[u * 32 + li] - M) * lu : 0.0f;
    }
    const float f_self = l_run > 0.0f ? __expf(m_run - M) : 0.0f;
#pragma unroll
    for (int n = 0; n < GV_D / 16; ++n) {
        o[n] *= f_self;
    }
#pragma unroll
    for (int u = GV7_WAVES - 1; u >= 0; --u) {
        if (w == u) {
            if (u < GV7_WAVES - 1) {
#pragma unroll
                for (int n = 0; n < GV_D / 16; ++n) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        o[n][v] += sbuf[(n * 4 + v) * 64 + l];
                    }
                }
            }
            if (u > 0) {
#pragma unroll
                for (int n = 0; n < GV_D / 16; ++n) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        sbuf[(n * 4 + v) * 64 + l] = o[n][v];
                    }
                }
            }
        }
        if (u > 0) {
            __syncthreads();
        }
    }
    if (w == 0 && valid) {
        const int64_t row = ((int64_t) split * n_head_kv + hkv) * R + r;
        float * op = Opart + row * GV_D;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
#pragma unroll
            for (int n4 = 0; n4 < 4; ++n4) {
                float4 f;
                f.x = o[4 * n4 + 0][v];
                f.y = o[4 * n4 + 1][v];
                f.z = o[4 * n4 + 2][v];
                f.w = o[4 * n4 + 3][v];
                *(float4 *) (op + 16 * (4 * kq + v) + 4 * n4) = f;
            }
        }
        if (kq == 0) {
            MLpart[row] = make_float2(M, Lt);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, Opart, MLpart, scale, n_q, gqa, n_head_kv, chunks_per_split, n_chunks,
                     nb01, nb02, nb11, nb12, nb21, nb22, nb31);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(AMD_MFMA_AVAILABLE)
}

// parallel deterministic split merge: one block per (row r, KV head), one thread per d; splits <= 256
static __global__ void __launch_bounds__(GV_D) gv7_merge_kernel(
        const float * __restrict__ Opart, const float2 * __restrict__ MLpart, float * __restrict__ dst,
        const int n_splits, const int n_q, const int gqa, const int n_head_kv) {
    __shared__ float sf[GV_D];
    __shared__ float sl[GV_D];
    __shared__ float swm[GV_D / 64];
    const int r   = blockIdx.x;
    const int hkv = blockIdx.y;
    const int t   = threadIdx.x;
    const int R   = gqa * n_q;

    float m = -FLT_MAX;
    float lv = 0.0f;
    if (t < n_splits) {
        const float2 ml = MLpart[((int64_t) t * n_head_kv + hkv) * R + r];
        lv = ml.y;
        m  = lv > 0.0f ? ml.x : -FLT_MAX;
    }
    float mx = m;
#pragma unroll
    for (int off = 32; off >= 1; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor(mx, off));
    }
    if (t % 64 == 0) {
        swm[t / 64] = mx;
    }
    __syncthreads();
    float M = swm[0];
#pragma unroll
    for (int u = 1; u < GV_D / 64; ++u) {
        M = fmaxf(M, swm[u]);
    }
    const float f = (t < n_splits && lv > 0.0f) ? __expf(m - M) : 0.0f;
    sf[t] = f;
    sl[t] = f * lv;
    __syncthreads();

    float L = 0.0f;
    float acc = 0.0f;
    const float * op = Opart + ((int64_t) hkv * R + r) * GV_D + t;
    const int64_t sstride = (int64_t) n_head_kv * R * GV_D;
    int s = 0;
    for (; s + 4 <= n_splits; s += 4) {
        const float o0 = op[(s + 0) * sstride];
        const float o1 = op[(s + 1) * sstride];
        const float o2 = op[(s + 2) * sstride];
        const float o3 = op[(s + 3) * sstride];
        acc += sf[s + 0] * o0;
        acc += sf[s + 1] * o1;
        acc += sf[s + 2] * o2;
        acc += sf[s + 3] * o3;
        L += sl[s + 0];
        L += sl[s + 1];
        L += sl[s + 2];
        L += sl[s + 3];
    }
    for (; s < n_splits; ++s) {
        acc += sf[s] * op[s * sstride];
        L   += sl[s];
    }
    const int h = r / n_q;
    const int q = r % n_q;
    const int n_head = n_head_kv * gqa;
    dst[((int64_t) q * n_head + hkv * gqa + h) * GV_D + t] = L > 0.0f ? acc / L : 0.0f;
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

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const bool rf7 = gv_vattn_rf7() && n_vseq == 1 && rows == nullptr && R <= 64 &&
                     K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0 &&
                     K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 && V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 &&
                     (uintptr_t) K->data % 4 == 0 && (uintptr_t) V->data % 4 == 0 &&
                     mask->nb[1] % 8 == 0 && (uintptr_t) mask->data % 8 == 0 &&
                     Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && (uintptr_t) Q->data % 16 == 0;
    if (rf7) {
        const int nrt      = (R + 15) / 16;
        const int n_chunks = n_kv / GV7_KC;
        const int nsm      = ggml_cuda_info().devices[ctx.device].nsm;
        // one resident round at 2 WG/CU (8 waves per CU), >= 4 chunks per wave
        int n_splits = gv_splits_env() > 0 ? gv_splits_env() :
            std::min(std::max(1, (2 * nsm) / (n_head_kv * nrt)), std::max(1, n_chunks / (4 * GV7_WAVES)));
        n_splits = std::max(1, std::min(n_splits, std::min(n_chunks, GV_D)));
        const int chunks_per_split = (n_chunks + n_splits - 1) / n_splits;
        n_splits = (n_chunks + chunks_per_split - 1) / chunks_per_split;

        const size_t n_rows_part = (size_t) n_splits * n_head_kv * R;
        ggml_cuda_pool_alloc<float>  Opart(ctx.pool(), n_rows_part * GV_D);
        ggml_cuda_pool_alloc<float2> MLpart(ctx.pool(), n_rows_part);
        cudaStream_t stream = ctx.stream();
        gv7_kernel<<<dim3(nrt, n_splits, n_head_kv), dim3(64 * GV7_WAVES), 0, stream>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            Opart.get(), MLpart.get(), scale, n_q, gqa, n_head_kv, chunks_per_split, n_chunks,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1]);
        CUDA_CHECK(cudaGetLastError());
        gv7_merge_kernel<<<dim3(R, n_head_kv), dim3(GV_D), 0, stream>>>(
            Opart.get(), MLpart.get(), (float *) dst->data, n_splits, n_q, gqa, n_head_kv);
        CUDA_CHECK(cudaGetLastError());
        static int n_rf7_calls = 0;
        if (n_rf7_calls++ == 0) {
            GGML_LOG_INFO("ggml_cuda_fattn_gqa_verify: GGML_HIP_VATTN_RF=7 wave-stream launch ran (splits=%d tiles=%d R=%d chunks/split=%d)\n",
                n_splits, nrt, R, chunks_per_split);
        }
        return;
    }

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
        int s_rf = std::max(1, (2 * nsm) / n_head_kv);
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
    const size_t smem = gv_smem_bytes(nwaves, kc);

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
    if (late) {
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
