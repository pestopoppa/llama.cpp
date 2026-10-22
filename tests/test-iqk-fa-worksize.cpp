// test-iqk-fa-worksize.cpp: [AKX-ALL-LB4.6] the iqk CPU FA work-buffer sizer vs the split-KV consumers.
//
// iqk_fa_work_buffer_size (ggml/src/ggml-cpu/iqk/iqk_flash_attn.cpp) is what the CPU graph plan reserves
// for the iqk FLASH_ATTN_EXT kernel (ggml-cpu.c: MAX(mainline prefill, mainline decode, iqk)). At decode
// (neq1 == 1, neq3 == 1, rk2 = rv2 > 1) iqk_flash_attn_noalibi writes its split-KV partials into that
// buffer from offset 0, on the nek1 the runtime -inf range trim leaves, not on K->ne[1]. A sanitizer
// cannot gate the sizer: the mainline prefill tile term dominates the MAX, so an undersized iqk term is
// covered and an ASan run passes vacuously. This test checks the iqk term alone, by enumeration.
//
// Consumer (transcribed from iqk_flash_attn_noalibi, decode, nek1 = the trimmed length n1):
//   P1  nek2 == 1, n1/32 > 1, nstep_k = n1/32 >= 4*nth: thread ith writes its rk2 rows (+ M, S) at
//       ith*(Dv+16)*rk2 floats                                    -> extent nth slots of (Dv+16)*rk2
//   P2  nek2 == 1, n1/32 > 1, else nq_per_thread = ceil(rk2/(nth/gcd(nstep_k, nth))) > 1: thread ith
//       writes at ith*(Dv+16)*nq_per_thread floats                 -> extent nth slots of (Dv+16)*nq
//   P3  otherwise if nek2*n1 >= 32*nth: chunk istep_k (0 <= istep_k < nstep_k = nek2*nkk, nk and nkk
//       from the gcd/npt arithmetic below) writes at istep_k*(Dv+16)*rk2 floats; a thread stops at its
//       first empty chunk (this_nk <= 0)                           -> extent = last written chunk + 1
//   else no work buffer at decode (the generic path; the K repack needs neq1 >= 8)
// The extent is in whole slots (the consumer's own stride), never the bytes actually touched inside the
// last slot, so the check is the stricter one.
//
// Trim (iqk_flash_attn_noalibi, the "quick hack for SWA models"): only when rk2 > 1, neq1 == 1, a mask
// and nek1 > 256; it narrows nek1 to a 32-multiple n1 <= 3*nek1/4 (0 for an all -inf row). The ops.cpp
// hook requires K->ne[1] % 32 == 0 and passes n_swa = 0, so for each n_kv the consumer can see
// n1 = n_kv, plus (n_kv > 256) every 32-multiple n1 in [0, 3*n_kv/4].
//
// Domain, per shape (rk2, nek2, Dk, Dv): DS41 (64, 1, 576, 512), MQA2 (2, 1, 128, 128),
// GQA12x2 (12, 2, 256, 256):
//   A  nth 1..128,                                       n_kv 96..16384 step 32,    every trim
//   B  nth 1 2 3 4 6 8 12 16 24 32 48 64 96 128 192,     n_kv 512..131072 step 256, every trim
//   = 24544392 (nth, n_kv, n1) checks per shape (kExpectedChecked; a different count is a FAIL, so an
//   empty or truncated enumeration cannot pass).
//
// Sizers checked against the same consumer:
//   real    iqk_fa_work_buffer_size itself (the exported symbol of the ON build), on ggml_tensor
//           headers with the decode shape; must have 0 shortfalls
//   prefix  MUTANT: the port's sizer before AKX-ALL-LB4.6 (9a338d4f1 L50-L108, verbatim): sized the
//           partials from the untrimmed K->ne[1] and with nth slots; must be DETECTED (shortfalls > 0)
//   minus1  MUTANT: real minus one (Dv+16)*rk2 float slot; must be DETECTED (the check is tight)
// Expected shortfall counts (a pure-arithmetic mirror of this file, 2026-10-22): real 0 / 0 / 0;
// prefix DS41 2909736, MQA2 6521534, GQA12x2 10257055; minus1 DS41 11494638, MQA2 911986,
// GQA12x2 12254. Only zero / non-zero is gated.
//
// Output: exactly 10 RESULT lines: per shape (DS41, MQA2, GQA12x2) one line each for sizer=real,
// sizer=prefix, sizer=minus1 (9), then one final "RESULT verdict=PASS|FAIL" line.
// Exit 0 iff verdict=PASS (every real line PASS, every mutant line DETECTED, every checked count ==
// kExpectedChecked), 1 otherwise, 2 if the linked sizer is the no-IQK stub (returns 0 everywhere).
//
// usage: test-iqk-fa-worksize [--dump]
//   --dump  also print one "SLOTS shape=.. nth=.. n_kv=.. need_max=.. real=.." line per (shape, nth,
//           n_kv) (the worst consumer extent over n_kv and its trims, in bytes); not RESULT lines
// Pure arithmetic: no ggml context, no graph, no threads. Built only with -DGGML_IQK_FLASH_ATTENTION=ON.

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// exported (extern "C", IQK_API) by libggml-cpu in -DGGML_IQK_FLASH_ATTENTION=ON builds (iqk/iqk_mul_mat.h)
extern "C" size_t iqk_fa_work_buffer_size(const struct ggml_tensor * dst, int nthread);

static constexpr long long kExpectedChecked = 24544392;

struct shape_t {
    const char * name;
    int rk2, nek2, Dk, Dv;
};

// ---------------------------------------------------------------- consumer (iqk_flash_attn_noalibi)
// bytes of work buffer the decode split-KV paths address for K length nek1 (whole slots); *path = 1, 2, 3, 0
static size_t consumer_extent(int rk2, int nek2, int Dv, int nth, int nek1, int * path) {
    *path = 0;
    if (nek2 == 1 && nek1/32 > 1) {
        const int nstep_k = nek1/32;
        if (nstep_k >= 4*nth) {
            // P1: work_this_thread = result_buffer + ith*size_thread, ith in [0, nth)
            const size_t size_thread = (size_t) (Dv + 16)*rk2*sizeof(float);
            *path = 1;
            return size_thread*nth;
        }
        const int gcd_k = std::gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            const int nth_k = nth/gcd_k;
            const int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                // P2: work_this_thread = result_buffer + ith*size_thread, ith in [0, nth)
                const size_t size_thread = (size_t) (Dv + 16)*nq_per_thread*sizeof(float);
                *path = 2;
                return size_thread*nth;
            }
        }
    }
    if (nek2*nek1 >= 32*nth) {
        // P3 (the port requires rk2 == rv2 here; every shape below has V heads == K heads)
        const size_t result_size = (size_t) (Dv + 16)*rk2*sizeof(float);
        const int gcd    = std::gcd(nek2, nth);
        const int nth_k  = nth/gcd;
        const int nek2_k = nek2/gcd;
        const int nchunk = nek2_k*nek1/32;
        const int npt    = (nchunk + nth_k - 1)/nth_k;
        int nk;
        if (npt*nth_k == nchunk) {
            nk = 32 * (nek2*nek1/(32*nth));
        } else {
            int nm = 1;
            while (true) {
                if (nm*4 >= npt) break;
                nm *= 2;
            }
            nk = 32*nm;
        }
        if (nk <= 0) {
            // the kernel would divide by zero computing nkk: report it as an unbounded need
            *path = 3;
            return SIZE_MAX;
        }
        const int nkk     = (nek1 + nk - 1)/nk;
        const int nstep_k = nek2*nkk;
        *path = 3;
        // istep_k = ith, ith + nth, ...; ik01 = nk*(istep_k % nkk) grows with istep_k inside a head, so a
        // head's chunks are all non-empty iff its last one is (this_nk = nek1 - nk*(nkk - 1) > 0); then no
        // thread breaks early and every chunk < nstep_k is written
        if (nek1 - nk*(nkk - 1) > 0) {
            return (size_t) nstep_k*result_size;
        }
        size_t ext = 0; // not reachable with nkk = ceil(nek1/nk); kept as the literal loop
        for (int ith = 0; ith < nth; ++ith) {
            for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
                const int ik02 = istep_k/nkk;
                const int ik01 = nk*(istep_k - ik02*nkk);
                const int this_nk = ik01 + nk <= nek1 ? nk : nek1 - ik01;
                if (this_nk <= 0) break;
                ext = std::max(ext, (size_t) (istep_k + 1)*result_size);
            }
        }
        return ext;
    }
    return 0;
}

// ---------------------------------------------------------------- sizers
static ggml_tensor g_q, g_k, g_v, g_dst;

static const ggml_tensor * decode_dst(const shape_t & s, int n_kv) {
    g_q = ggml_tensor{}; g_k = ggml_tensor{}; g_v = ggml_tensor{}; g_dst = ggml_tensor{};
    g_q.type = GGML_TYPE_F32; g_k.type = GGML_TYPE_F16; g_v.type = GGML_TYPE_F16; g_dst.type = GGML_TYPE_F32;
    const int64_t qe[4] = { s.Dk, 1,    (int64_t) s.rk2*s.nek2, 1 };
    const int64_t ke[4] = { s.Dk, n_kv, s.nek2,                 1 };
    const int64_t ve[4] = { s.Dv, n_kv, s.nek2,                 1 };
    const int64_t de[4] = { s.Dv, (int64_t) s.rk2*s.nek2, 1,    1 };
    for (int i = 0; i < 4; ++i) { g_q.ne[i] = qe[i]; g_k.ne[i] = ke[i]; g_v.ne[i] = ve[i]; g_dst.ne[i] = de[i]; }
    g_dst.op     = GGML_OP_FLASH_ATTN_EXT;
    g_dst.src[0] = &g_q;
    g_dst.src[1] = &g_k;
    g_dst.src[2] = &g_v;
    return &g_dst;
}

static inline uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) a -= b;
        else b -= a;
    }
    return a;
}

// MUTANT: the port's iqk_fa_work_buffer_size at 9a338d4f1 (before AKX-ALL-LB4.6), verbatim
static size_t sizer_prefix(const struct ggml_tensor * dst, int nth) {
    auto Q = dst->src[0];
    auto K = dst->src[1];
    auto V = dst->src[2];
    int rk2 = Q->ne[2]/K->ne[2];
    size_t size = 0;
    if (Q->ne[1] >= 8 && K->type == GGML_TYPE_Q8_0) {
        size = ggml_row_size(GGML_TYPE_Q8_0, K->ne[0]) * K->ne[1]*K->ne[2]*K->ne[3];
    }
    if (Q->ne[1] == 1 && Q->ne[3] == 1 && Q->ne[2]/K->ne[2] > 1 && nth >= 1 && K->ne[1]/32 > 1) {
        if (K->ne[2] > 1) {
            int gcd = simple_gcd(K->ne[2], nth);
            int nth_k  = nth/gcd;
            int nek2_k = K->ne[2]/gcd;
            int nchunk = nek2_k*K->ne[1]/32;
            int npt = (nchunk + nth_k - 1)/nth_k;
            int nk;
            if (npt*nth_k == nchunk) {
                nk = 32 * (K->ne[1]*K->ne[2]/(32*nth));
            } else {
                //int nm = std::max(1, npt/8);
                int nm = 1;
                while (true) {
                    if (nm*4 >= npt) break;
                    nm *= 2;
                }
                nk = 32*nm;
            }
            int nkk = (K->ne[1] + nk - 1)/nk;
            int nstep_k = K->ne[2]*nkk;
            size_t result_size = (V->ne[0] + 16)*Q->ne[2]/K->ne[2]*sizeof(float);
            size += nstep_k*result_size;
            return size;
        }
        int nstep_k = K->ne[1]/32;
        if (nstep_k >= 4*nth) {
            auto size_thread = (V->ne[0] + 16)*rk2*sizeof(float);
            size += size_thread*nth;
            return size;
        }
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                auto size_thread = (V->ne[0] + 16)*nq_per_thread*sizeof(float);
                size += size_thread*nth;
                return size;
            }
        }
        int rv2 = Q->ne[2] / V->ne[2];
        if (Q->ne[1] == 1 && Q->ne[3] == 1 && rk2 > 1 && rk2 == rv2 && K->ne[1]*K->ne[2] >= 32*nth) {
            auto result_size = (V->ne[0] + 16)*rk2*sizeof(float);
            size += result_size*nth;
        }
        return size;
    }
    return size;
}

// ---------------------------------------------------------------- enumeration
enum { SZ_REAL = 0, SZ_PREFIX = 1, SZ_MINUS1 = 2, SZ_N = 3 };
static const char * kSizerName[SZ_N] = { "real", "prefix", "minus1" };

struct tally_t {
    long long checked    = 0;
    long long shortfalls = 0;
    double    worst      = 0.0; // max need/alloc over shortfalls
    char      first[192] = "-";
};

static void record(tally_t & t, size_t need, size_t alloc, int nth, int n_kv, int n1, int path) {
    if (need <= alloc) { return; }
    t.shortfalls++;
    const double ratio = alloc > 0 ? (need == SIZE_MAX ? 1e30 : (double) need / (double) alloc) : 1e30;
    if (t.shortfalls == 1) {
        snprintf(t.first, sizeof(t.first), "nth=%d,n_kv=%d,nek1=%d,path=P%d,need=%zu,alloc=%zu", nth, n_kv, n1, path, need, alloc);
    }
    t.worst = std::max(t.worst, ratio);
}

static void run_shape(const shape_t & s, const std::vector<int> & nths, int n_kv_lo, int n_kv_hi, int n_kv_step,
                      tally_t * tally, bool dump) {
    const size_t slot = (size_t) (s.Dv + 16)*s.rk2*sizeof(float);
    std::vector<size_t> need((size_t) n_kv_hi/32 + 1);
    std::vector<int>    path((size_t) n_kv_hi/32 + 1);
    for (int nth : nths) {
        // the consumer depends on (nth, n1) only: tabulate it once per nth
        for (size_t i = 0; i < need.size(); ++i) {
            need[i] = consumer_extent(s.rk2, s.nek2, s.Dv, nth, (int) (32*i), &path[i]);
        }
        for (int n_kv = n_kv_lo; n_kv <= n_kv_hi; n_kv += n_kv_step) {
            const ggml_tensor * dst = decode_dst(s, n_kv);
            size_t alloc[SZ_N];
            alloc[SZ_REAL]   = iqk_fa_work_buffer_size(dst, nth);
            alloc[SZ_PREFIX] = sizer_prefix(dst, nth);
            alloc[SZ_MINUS1] = alloc[SZ_REAL] >= slot ? alloc[SZ_REAL] - slot : 0;
            size_t worst_need = 0;
            auto check = [&](int n1) {
                const size_t nd = need[n1/32];
                worst_need = std::max(worst_need, nd);
                for (int z = 0; z < SZ_N; ++z) {
                    tally[z].checked++;
                    record(tally[z], nd, alloc[z], nth, n_kv, n1, path[n1/32]);
                }
            };
            check(n_kv);                                    // untrimmed
            if (n_kv > 256) {
                for (int n1 = 0; n1 <= 3*n_kv/4; n1 += 32) { // every -inf prefix/suffix trim
                    check(n1);
                }
            }
            if (dump) {
                printf("SLOTS shape=%s nth=%d n_kv=%d need_max=%zu real=%zu prefix=%zu\n",
                       s.name, nth, n_kv, worst_need, alloc[SZ_REAL], alloc[SZ_PREFIX]);
            }
        }
    }
}

int main(int argc, char ** argv) {
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump") == 0) { dump = true; }
        else { fprintf(stderr, "usage: %s [--dump]\n", argv[0]); return 2; }
    }

    const shape_t shapes[] = {
        { "DS41",    64, 1, 576, 512 },
        { "MQA2",     2, 1, 128, 128 },
        { "GQA12x2", 12, 2, 256, 256 },
    };

    // the no-IQK build links a stub that reserves nothing: there is no sizer to test
    if (iqk_fa_work_buffer_size(decode_dst(shapes[0], 4096), 8) == 0) {
        fprintf(stderr, "iqk_fa_work_buffer_size returned 0 for a DS41 decode: stub sizer (no IQK_IMPLEMENT)\n");
        return 2;
    }

    std::vector<int> nth_a(128);
    std::iota(nth_a.begin(), nth_a.end(), 1);
    const std::vector<int> nth_b = { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192 };

    printf("test-iqk-fa-worksize: A nth=1..128 n_kv=96..16384/32; B nth=1,2,3,4,6,8,12,16,24,32,48,64,96,128,192 "
           "n_kv=512..131072/256; trims n1=0..3*n_kv/4 step 32 when n_kv > 256; expected_checked=%lld per shape\n",
           kExpectedChecked);

    bool real_ok = true, checked_ok = true;
    int  n_detected = 0, n_mutants = 0;
    for (const auto & s : shapes) {
        tally_t tally[SZ_N];
        run_shape(s, nth_a, 96,  16384,  32,  tally, dump);
        run_shape(s, nth_b, 512, 131072, 256, tally, dump);
        for (int z = 0; z < SZ_N; ++z) {
            const tally_t & t = tally[z];
            const bool count_ok = t.checked == kExpectedChecked;
            checked_ok = checked_ok && count_ok;
            const char * verdict;
            if (z == SZ_REAL) {
                const bool ok = count_ok && t.shortfalls == 0;
                real_ok = real_ok && ok;
                verdict = ok ? "PASS" : "FAIL";
            } else {
                n_mutants++;
                const bool det = count_ok && t.shortfalls > 0;
                if (det) { n_detected++; }
                verdict = det ? "DETECTED" : "MISSED";
            }
            printf("RESULT sizer=%s shape=%s rk2=%d nek2=%d Dk=%d Dv=%d checked=%lld checked_ok=%d shortfalls=%lld "
                   "worst_ratio=%.3f first=%s verdict=%s\n",
                   kSizerName[z], s.name, s.rk2, s.nek2, s.Dk, s.Dv, t.checked, (int) count_ok, t.shortfalls,
                   t.worst, t.first, verdict);
        }
    }
    const bool pass = real_ok && checked_ok && n_detected == n_mutants;
    printf("RESULT verdict=%s real_ok=%d checked_ok=%d mutants_detected=%d/%d\n",
           pass ? "PASS" : "FAIL", (int) real_ok, (int) checked_ok, n_detected, n_mutants);
    fflush(stdout);
    return pass ? 0 : 1;
}
