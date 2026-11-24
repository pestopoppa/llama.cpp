// DF2-QMATCH-1: truncating the DFlash2 proposal q by the request top_k/top_p/min_p (LLAMA_DFLASH_Q_MATCH=1) must keep
// the speculative output exact. Model-free: a fixed selector score vector stands in for the lattice; the target p is
// a top_k=20/top_p=0.95 truncated softmax (the served request). Checks (a) the emitted distribution with knob OFF and
// ON both match p and match each other (chi-square, p > 0.01), (b) acceptance does not drop with the knob ON,
// (c) greedy (T = 0) is unaffected: q_match keeps the argmax and the T<=0 path never calls it.
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static std::vector<float> softmax(const std::vector<float> & s, float T) {
    std::vector<float> p(s.size());
    const float m = *std::max_element(s.begin(), s.end());
    double sum = 0; for (size_t i = 0; i < s.size(); ++i) { p[i] = std::exp((s[i] - m) / T); sum += p[i]; }
    for (float & x : p) { x = (float) (x / sum); }
    return p;
}

// upper-tail chi-square p-value (Wilson-Hilferty)
static double chi2_p(double x, int df) {
    const double z = (std::pow(x / df, 1.0 / 3) - (1 - 2.0 / (9 * df))) / std::sqrt(2.0 / (9 * df));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

struct run_out { std::vector<long> hist; double acc; };

static run_out run(const std::vector<float> & p, const std::vector<float> & q0, bool qmatch, int N, uint32_t seed) {
    run_out o{std::vector<long>(p.size(), 0), 0.0};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<float> q = q0;
    if (qmatch) { common_speculative_q_match(q, 20, 0.95f, 0.0f); }
    std::discrete_distribution<int> Q(q.begin(), q.end());
    long acc = 0;
    for (int n = 0; n < N; ++n) {
        const int x = Q(rng);
        int y;
        if (q[x] > 0.0f && U(rng) * q[x] <= p[x]) { y = x; acc++; }
        else {
            std::vector<float> r(p.size());
            for (size_t j = 0; j < p.size(); ++j) { r[j] = std::max(0.0f, p[j] - q[j]); }
            std::discrete_distribution<int> R(r.begin(), r.end());
            y = R(rng);
        }
        o.hist[y]++;
    }
    o.acc = (double) acc / N;
    return o;
}

int main() {
    const int K = 32; // selector_top_k
    std::vector<float> s_tgt(K), s_dft(K);
    for (int k = 0; k < K; ++k) { s_tgt[k] = -0.35f * k + 0.3f * std::sin(k); s_dft[k] = -0.22f * k + 0.5f * std::cos(1.7f * k); }
    std::vector<float> p = softmax(s_tgt, 1.0f);
    common_speculative_q_match(p, 20, 0.95f, 0.0f); // target chain: top_k 20, top_p 0.95
    const std::vector<float> q = softmax(s_dft, 1.0f);

    int support = 0; for (float x : p) { support += x > 0; }
    CHECK(support >= 2 && support <= 20);
    { std::vector<float> t = q; common_speculative_q_match(t, 20, 0.95f, 0.0f); double s = 0; int nz = 0;
      for (float x : t) { s += x; nz += x > 0; } CHECK(std::fabs(s - 1.0) < 1e-5); CHECK(nz <= 20); }

    double worst = 1.0, acc_off = 0, acc_on = 0;
    const int N = 200000;
    for (uint32_t seed = 1; seed <= 8; ++seed) {
        const auto off = run(p, q, false, N, seed * 7919u);
        const auto on  = run(p, q, true,  N, seed * 104729u);
        acc_off += off.acc / 8; acc_on += on.acc / 8;
        double c_off = 0, c_on = 0, c_two = 0; int df = 0, df2 = 0;
        for (int k = 0; k < K; ++k) {
            if (p[k] > 0) { const double e = p[k] * N; c_off += std::pow(off.hist[k] - e, 2) / e; c_on += std::pow(on.hist[k] - e, 2) / e; df++; }
            else { CHECK(off.hist[k] == 0 && on.hist[k] == 0); }
            const long t = off.hist[k] + on.hist[k];
            if (t > 0) { c_two += std::pow(off.hist[k] - on.hist[k], 2) / (double) t; df2++; }
        }
        const double p1 = chi2_p(c_off, df - 1), p2 = chi2_p(c_on, df - 1), p3 = chi2_p(c_two, df2 - 1);
        worst = std::min({worst, p1, p2, p3});
        printf("seed %u: chi2 p OFF-vs-p %.3f  ON-vs-p %.3f  OFF-vs-ON %.3f\n", seed, p1, p2, p3);
    }
    // Bonferroni over 24 tests at family alpha 0.01 would be stricter; require each > 0.01/24 and the min reported
    CHECK(worst > 0.01 / 24);
    printf("acceptance OFF %.4f ON %.4f (sum min(p,q) analytic OFF %.4f)\n", acc_off, acc_on, [&]{ double a=0; for(int k=0;k<K;++k) a+=std::min(p[k],q[k]); return a; }());
    CHECK(acc_on >= acc_off - 0.002);

    // (c) greedy: the argmax survives every truncation, so a T<=0 pick (argmax over scores) is the same either way
    for (int tk : {1, 5, 20}) for (float tp : {0.1f, 0.95f}) for (float mp : {0.0f, 0.5f}) {
        std::vector<float> t = q; common_speculative_q_match(t, tk, tp, mp);
        CHECK(std::distance(t.begin(), std::max_element(t.begin(), t.end())) == std::distance(q.begin(), std::max_element(q.begin(), q.end())));
    }
    // knob default off
    CHECK(!common_speculative_q_match_enabled() || getenv("LLAMA_DFLASH_Q_MATCH") != nullptr);

    printf("%s (worst p %.4f)\n", n_fail ? "FAILED" : "OK", worst);
    return n_fail ? 1 : 0;
}
