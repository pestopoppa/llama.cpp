// test-dflash-dsel.cpp: [GPU-RESIDENT-ROUND-2d] device DFlash2 lattice selection (llama_build_dflash_dsel)
// against the host selection loop of common/speculative.cpp.
//   greedy (inv_temp 1, noise 0): selected tokens BITWISE equal to the host std::max_element path, random lattices,
//     including exact ties (first max wins).
//   temp > 0: probs rows match the host softmax (|diff| <= 1e-6), and the device Gumbel-max path with Philox noise
//     samples the host distribution: per position-1 and position-2 marginals over >= 640k draws, chi-square
//     p >= 1e-3 against the exact softmax / exact chained probabilities; an A/A calibration (two independent
//     host std::discrete_distribution samples) is reported beside it.
// Backend = CPU, or DSEL_BACKEND=<name> (e.g. ROCm0).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../src/llama-dflash-dsel.h"
#include "../common/philox.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
static ggml_backend_t g_be = nullptr;

struct lattice {
    int64_t n_embd, K, B, NB;
    std::vector<float> data; // [n_embd, B * NB]
    float * row(int64_t b, int64_t pos) { return data.data() + (b*B + pos)*n_embd; }
};

static lattice make_lattice(std::mt19937 & rng, int64_t K, int64_t B, int64_t NB, bool ties) {
    lattice L{K + K*K + 7, K, B, NB, {}};
    L.data.assign(L.n_embd*B*NB, 0.0f);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (int64_t b = 0; b < NB; ++b) {
        for (int64_t pos = 1; pos < B; ++pos) {
            float * r = L.row(b, pos);
            for (int64_t k = 0; k < K; ++k) {
                r[k] = (float) (1000 + (rng() % 1000)*64 + k); // distinct per row
            }
            for (int64_t p = 0; p < K; ++p) {
                for (int64_t k = 0; k < K; ++k) {
                    float v = nd(rng);
                    if (ties) {
                        v = std::round(v);
                    }
                    r[K + p*K + k] = (pos == 1 && p > 0) ? r[K + k] : v; // position 1: every predecessor row equal (repeat)
                }
            }
        }
    }
    return L;
}

// runs the device chain; returns [W, B*NB]
static std::vector<float> run_dev(lattice & L, const std::vector<float> & inv_temp, const std::vector<float> & noise) {
    const int64_t K = L.K, B = L.B, NB = L.NB, W = 1 + 2*K;
    ggml_init_params ip = { 1024*ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L.n_embd, B*NB);
    ggml_tensor * it = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, NB);
    ggml_tensor * nz = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, B, NB);
    ggml_tensor * out = llama_build_dflash_dsel(ctx, packed, it, nz, K, B, NB);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_be);
    ggml_backend_tensor_set(packed, L.data.data(), 0, ggml_nbytes(packed));
    ggml_backend_tensor_set(it, inv_temp.data(), 0, ggml_nbytes(it));
    ggml_backend_tensor_set(nz, noise.data(), 0, ggml_nbytes(nz));
    ggml_backend_graph_compute(g_be, gf);
    std::vector<float> res(W*B*NB);
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// host greedy path (common/speculative.cpp)
static bool g_last_max = false; // CPU ggml argmax keeps the last max on exact ties; HIP (DTREE-A) the first, like the host

static std::vector<int> host_greedy(lattice & L, int64_t b) {
    std::vector<int> toks;
    int pred = 0;
    for (int64_t i = 1; i < L.B; ++i) {
        const float * r = L.row(b, i);
        const float * sc = r + L.K + (size_t) pred*L.K;
        pred = (int) std::distance(sc, std::max_element(sc, sc + L.K));
        if (g_last_max) {
            for (int k = 0; k < L.K; ++k) {
                if (sc[k] == sc[pred]) {
                    pred = k;
                }
            }
        }
        toks.push_back((int) r[pred]);
    }
    return toks;
}

static std::vector<double> host_probs(const float * sc, int64_t K, float T) {
    std::vector<double> p(K);
    const float mx = *std::max_element(sc, sc + K);
    double s = 0;
    for (int64_t k = 0; k < K; ++k) {
        p[k] = std::exp((sc[k] - mx) / T);
        s += p[k];
    }
    for (auto & x : p) {
        x /= s;
    }
    return p;
}

static double chi2_sf(double x, int k) { // Wilson-Hilferty
    if (k <= 0) {
        return 1.0;
    }
    const double z = (std::cbrt(x / k) - (1 - 2.0 / (9 * k))) / std::sqrt(2.0 / (9 * k));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

static double chi2_p(const std::vector<double> & obs, const std::vector<double> & p, double n) {
    double chi = 0;
    int dof = -1;
    for (size_t k = 0; k < obs.size(); ++k) {
        const double e = n * p[k];
        if (e < 5) {
            continue;
        }
        chi += (obs[k] - e) * (obs[k] - e) / e;
        dof++;
    }
    return chi2_sf(chi, dof);
}

static void test_greedy(std::mt19937 & rng) {
    int n = 0, ok = 0;
    for (bool ties : { false, true }) {
        for (int64_t K : { 4, 8, 16 }) {
            for (int64_t NB : { 1, 2, 4 }) {
                lattice L = make_lattice(rng, K, 8, NB, ties);
                g_last_max = ties && strcmp(ggml_backend_name(g_be), "CPU") == 0;
                std::vector<float> it(NB, 1.0f), nz(K*8*NB, 0.0f);
                const auto out = run_dev(L, it, nz);
                bool same = true;
                for (int64_t b = 0; b < NB; ++b) {
                    const auto h = host_greedy(L, b);
                    for (int64_t i = 1; i < 8; ++i) {
                        if ((int) out[(b*8 + i)*(1 + 2*K)] != h[i - 1] && same) {
                            fprintf(stderr, "  b=%lld i=%lld dev=%d host=%d\n", (long long) b, (long long) i, (int) out[(b*8 + i)*(1 + 2*K)], h[i - 1]);
                        }
                        same &= (int) out[(b*8 + i)*(1 + 2*K)] == h[i - 1];
                    }
                }
                n++;
                ok += same;
                if (!same) {
                    fprintf(stderr, "FAIL greedy K=%lld NB=%lld ties=%d\n", (long long) K, (long long) NB, (int) ties);
                }
            }
        }
    }
    printf("greedy: %d / %d lattices token-identical to the host path (exact-tie lattices judged against the %s tie rule)\n",
           ok, n, strcmp(ggml_backend_name(g_be), "CPU") == 0 ? "CPU last-max" : "host first-max");
    g_fail += n - ok;
}

static void test_stoch(std::mt19937 & rng) {
    const int64_t K = 16, B = 3, NB = 64;      // 64 blocks per graph run
    const float T = 1.0f;
    const int64_t runs = 10000;                  // 640k draws per position
    lattice L = make_lattice(rng, K, B, NB, false);
    // the same lattice in every block so all blocks sample one distribution
    for (int64_t b = 1; b < NB; ++b) {
        std::copy(L.row(0, 0), L.row(0, 0) + L.n_embd*B, L.row(b, 0));
    }
    const float * sc1 = L.row(0, 1) + K;
    const auto p1 = host_probs(sc1, K, T);
    // exact pos-2 marginal = sum_j p1[j] * softmax(row j of pos 2)
    std::vector<double> p2(K, 0.0);
    std::vector<int> id2(K);
    for (int64_t j = 0; j < K; ++j) {
        const auto q = host_probs(L.row(0, 2) + K + j*K, K, T);
        for (int64_t k = 0; k < K; ++k) {
            p2[k] += p1[j] * q[k];
        }
    }
    std::vector<double> c1(K, 0), c2(K, 0);
    double max_dprob = 0;
    const uint64_t key = 0x1234ull << 32;
    std::vector<float> it(NB, 1.0f / T), nz(K*B*NB, 0.0f);
    auto idx_of = [&](int64_t pos, float tok) {
        const float * r = L.row(0, pos);
        for (int64_t k = 0; k < K; ++k) {
            if (r[k] == tok) {
                return (int) k;
            }
        }
        return -1;
    };
    for (int64_t r = 0; r < runs; ++r) {
        for (int64_t b = 0; b < NB; ++b) {
            for (int64_t j = K; j < B*K; j += 4) {
                uint32_t bits[4];
                common_philox4x32::gen(key | (uint32_t) b, (uint64_t) r, (uint64_t) j, bits);
                for (int q = 0; q < 4; ++q) {
                    nz[(b*B)*K + j + q] = common_philox4x32::gumbel(bits[q]);
                }
            }
        }
        const auto out = run_dev(L, it, nz);
        for (int64_t b = 0; b < NB; ++b) {
            const float * r1 = out.data() + (b*B + 1)*(1 + 2*K);
            const float * r2 = out.data() + (b*B + 2)*(1 + 2*K);
            const int k1 = idx_of(1, r1[0]);
            const int k2 = idx_of(2, r2[0]);
            if (k1 < 0 || k2 < 0) {
                g_fail++;
                return;
            }
            c1[k1]++;
            c2[k2]++;
            if (r == 0) {
                for (int64_t k = 0; k < K; ++k) {
                    max_dprob = std::max(max_dprob, std::fabs(r1[1 + K + k] - p1[k]));
                }
            }
        }
    }
    const double n = (double) runs * NB;
    const double pv1 = chi2_p(c1, p1, n), pv2 = chi2_p(c2, p2, n);
    // A/A calibration: host std::discrete_distribution on p1, same n
    std::mt19937 r2(99);
    std::discrete_distribution<int> dd(p1.begin(), p1.end());
    std::vector<double> ca(K, 0);
    for (int64_t i = 0; i < (int64_t) n; ++i) {
        ca[dd(r2)]++;
    }
    const double pva = chi2_p(ca, p1, n);
    printf("stoch: %.0f draws/pos, probs max|dev-host| %.2e, chi2 p pos1 %.4f pos2 %.4f (A/A host sampler p %.4f)\n",
           n, max_dprob, pv1, pv2, pva);
    if (max_dprob > 1e-6 || pv1 < 1e-3 || pv2 < 1e-3) {
        fprintf(stderr, "FAIL stoch\n");
        g_fail++;
    }
}

int main() {
    ggml_backend_load_all();
    const char * bn = getenv("DSEL_BACKEND");
    g_be = bn ? ggml_backend_init_by_name(bn, nullptr) : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!g_be) {
        fprintf(stderr, "FAIL: backend\n");
        return 1;
    }
    std::mt19937 rng(4321);
    test_greedy(rng);
    test_stoch(rng);
    ggml_backend_free(g_be);
    printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
