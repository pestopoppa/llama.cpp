// DF2-PM: point-mass DFlash2 proposals at T > 0 (LLAMA_DF2_POINT_MASS=1, default off). Checks
// (a) the knob is default off, (b) the point-mass proposal is the argmax with one-hot q,
// (c) the stock proposal is softmax(scores/T) and the sampler follows it,
// (d) rejection verify (accept if u*q(x) <= p(x), else draw from max(p-q,0)) with the one-hot q is lossless:
//     the emitted token's distribution equals the target p.
#include "speculative-tree.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    unsetenv("LLAMA_DF2_POINT_MASS");
    CHECK(!common_df2_point_mass_enabled());

    const std::vector<float> scores = { 0.3f, 2.0f, -1.0f, 1.5f, 0.0f, 1.9f };
    const int32_t K = (int32_t) scores.size();
    std::vector<float> q;

    // (b)
    for (double u : { 0.0, 0.37, 0.999 }) {
        const int32_t i = common_df2_propose(scores.data(), K, 0.6f, true, u, q);
        CHECK(i == 1);
        CHECK(q.size() == (size_t) K);
        for (int32_t k = 0; k < K; ++k) { CHECK(q[k] == (k == 1 ? 1.0f : 0.0f)); }
    }

    // (c)
    {
        const float T = 0.6f;
        std::vector<double> ref(K); double s = 0;
        for (int32_t k = 0; k < K; ++k) { ref[k] = std::exp((scores[k] - 2.0f) / T); s += ref[k]; }
        common_df2_propose(scores.data(), K, T, false, 0.5, q);
        for (int32_t k = 0; k < K; ++k) { CHECK(std::fabs(q[k] - ref[k] / s) < 1e-6); }
        std::vector<int> hits(K, 0); const int N = 200000;
        for (int n = 0; n < N; ++n) { hits[common_df2_propose(scores.data(), K, T, false, (n + 0.5) / N, q)]++; }
        for (int32_t k = 0; k < K; ++k) { CHECK(std::fabs(hits[k] / (double) N - ref[k] / s) < 1e-3); }
    }

    // (d) losslessness of the residual verify with one-hot q against an arbitrary target p
    {
        const std::vector<float> p = { 0.10f, 0.25f, 0.05f, 0.30f, 0.20f, 0.10f };
        const int32_t x = common_df2_propose(scores.data(), K, 0.6f, true, 0.0, q);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> res(K); double rs = 0;
        for (int32_t k = 0; k < K; ++k) { res[k] = std::max(0.0f, p[k] - q[k]); rs += res[k]; }
        std::discrete_distribution<int> R(res.begin(), res.end());
        std::vector<int> hits(K, 0); const int N = 400000;
        for (int n = 0; n < N; ++n) {
            const int out = (U(rng) * q[x] <= p[x]) ? x : R(rng);
            hits[out]++;
        }
        CHECK(rs > 0.0);
        for (int32_t k = 0; k < K; ++k) { CHECK(std::fabs(hits[k] / (double) N - p[k]) < 4e-3); }
    }

    printf("%s (%d failures)\n", n_fail ? "FAIL" : "OK", n_fail);
    return n_fail ? 1 : 0;
}
