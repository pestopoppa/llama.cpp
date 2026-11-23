// CAFE-3: online chain-width controller (common/speculative-cafe3.h).
//  (a) convergence: on synthetic acceptance/cost traces the controller settles on the best static cap
//  (b) zero is not absorbing: after a long zero-acceptance phase it recovers a positive cap within the back-off bound
//  (c) law: speculative rejection sampling whose per-round draft length is chosen by the controller emits exactly the
//      target distribution (chi-square over a Markov target, conditioned on the previous token), as does a static arm
//      (the harness is the reject_step of test-spec-dists-trunc / common_sampler_sample_and_accept_n)
//  (d) off: LLAMA_CAFE3_ADAPTIVE unset or != "1" means disabled
#include "speculative-cafe3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

// a synthetic workload: position i (given i drafted) accepted with prob a_i; round cost = base + per*w (+ noise)
struct trace {
    std::vector<double> a;   // conditional acceptance per position
    double base, per;
    double pmin_stop = 0.0;  // probability the drafter stops early at each position (p-min floor)
};

static double true_rate(const trace & t, int w) {
    // expected committed tokens / expected round cost for cap w. The drafter keeps drafting until the cap or a p-min
    // stop (it does not know which tokens the target will reject); verify cost ~ drafted tokens.
    double tok = 1.0, surv = 1.0, ecost = t.base, reach = 1.0;
    for (int i = 0; i < w; ++i) {
        reach *= (1.0 - t.pmin_stop); // position i drafted
        ecost += t.per * reach;
        surv  *= (1.0 - t.pmin_stop) * t.a[i];
        tok   += surv;                // committed only on an accepted, drafted prefix
    }
    return tok / ecost;
}

static void run_round(const trace & t, int sel, std::mt19937 & rng, int & drafted, int & accepted, double & us) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    drafted = 0; accepted = 0;
    bool alive = true;
    for (int i = 0; i < sel; ++i) {
        if (U(rng) < t.pmin_stop) break;
        drafted++;
        if (alive && U(rng) < t.a[i]) accepted++; else alive = false;
    }
    std::normal_distribution<double> N(1.0, 0.05);
    us = (t.base + t.per * drafted) * std::max(0.5, N(rng));
}

static int converge(const trace & t, int max_w, int rounds, uint32_t seed, std::vector<int> & hist_tail) {
    common_cafe3_ctrl c(max_w);
    std::mt19937 rng(seed);
    hist_tail.assign(max_w + 1, 0);
    for (int r = 0; r < rounds; ++r) {
        const int sel = c.select(max_w);
        int d, a; double us;
        run_round(t, sel, rng, d, a, us);
        c.observe(sel, d, a, us);
        if (r >= rounds - rounds / 4) hist_tail[sel]++;
    }
    int mode = 0;
    for (int w = 1; w <= max_w; ++w) if (hist_tail[w] > hist_tail[mode]) mode = w;
    return mode;
}

static int argmax_rate(const trace & t, int max_w) {
    int best = 0;
    for (int w = 1; w <= max_w; ++w) if (true_rate(t, w) > true_rate(t, best)) best = w;
    return best;
}

int main() {
    const int W = 6;
    // (a) convergence on four regimes: high acceptance/cheap verify (wide), decaying, expensive verify (narrow), bad
    // drafter (0); plus a p-min-floor regime
    std::vector<trace> T = {
        { {0.92, 0.90, 0.88, 0.86, 0.85, 0.84}, 1000, 60, 0.0 },
        { {0.85, 0.70, 0.50, 0.30, 0.20, 0.10}, 1000, 150, 0.0 },
        { {0.70, 0.60, 0.50, 0.40, 0.30, 0.20}, 1000, 450, 0.0 },
        { {0.05, 0.05, 0.05, 0.05, 0.05, 0.05}, 1000, 300, 0.0 },
        { {0.90, 0.85, 0.80, 0.70, 0.60, 0.50}, 1000, 120, 0.25 },
    };
    for (size_t k = 0; k < T.size(); ++k) {
        const int best = argmax_rate(T[k], W);
        int ok = 0;
        for (uint32_t s = 0; s < 5; ++s) {
            std::vector<int> h;
            const int mode = converge(T[k], W, 4000, 1234 + s + 17 * (uint32_t) k, h);
            // settle on the optimum or a cap within 3% of its true rate (EMA noise between near-tied caps)
            if (mode == best || true_rate(T[k], mode) >= 0.97 * true_rate(T[k], best)) ok++;
            printf("regime %zu seed %u: best %d mode %d (rate %.5f vs %.5f) tail[", k, s, best, mode,
                   true_rate(T[k], mode), true_rate(T[k], best));
            for (int w = 0; w <= W; ++w) printf("%s%d", w ? "," : "", h[w]);
            printf("]\n");
        }
        CHECK(ok >= 4);
    }

    // (b) zero is not absorbing: 600 rounds of a useless drafter, then a good one; a positive cap must win again
    {
        common_cafe3_ctrl c(W, 0.2, 64);
        std::mt19937 rng(99);
        trace bad  = { {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1000, 300, 0.0 };
        trace good = { {0.95, 0.93, 0.92, 0.90, 0.90, 0.90}, 1000, 80, 0.0 };
        int last_zero_streak = 0, streak = 0;
        for (int r = 0; r < 600; ++r) {
            const int sel = c.select(W); int d, a; double us;
            run_round(bad, sel, rng, d, a, us); c.observe(sel, d, a, us);
            streak = sel == 0 ? streak + 1 : 0;
            last_zero_streak = std::max(last_zero_streak, streak);
        }
        CHECK(last_zero_streak <= 64 + W + 1); // bounded back-off: a positive cap is always re-probed
        int first_pos_win = -1, n_pos = 0;
        for (int r = 0; r < 600; ++r) {
            const int sel = c.select(W); int d, a; double us;
            run_round(good, sel, rng, d, a, us); c.observe(sel, d, a, us);
            if (r >= 300 && sel > 0) n_pos++;
            if (first_pos_win < 0 && sel > 0 && !c.zero_probe && r > 0) first_pos_win = r;
        }
        printf("absorbing: max zero streak %d, recovered by round %d, positive share in tail %d/300\n",
               last_zero_streak, first_pos_win, n_pos);
        CHECK(first_pos_win >= 0 && first_pos_win <= 2 * (64 + W + 1));
        CHECK(n_pos >= 270);
        // cap 0 (no room in the context) is legal and never selects a positive width
        for (int r = 0; r < 10; ++r) { CHECK(c.select(0) == 0); c.observe(0, 0, 0, 1000.0); }
    }

    // (c) law: Markov target p(.|prev) over V tokens, draft q(.|prev); controller-chosen width per round; every emitted
    // token is produced by the rejection step; chi-square of emitted transitions against p for both arms
    {
        const int V = 8;
        auto P = [&](int prev, std::vector<double> & p) {
            p.assign(V, 0.0); double s = 0;
            for (int k = 0; k < V; ++k) { p[k] = 0.3 + (double) ((k * 5 + prev * 3) % 7); s += p[k]; }
            for (auto & x : p) x /= s;
        };
        auto Q = [&](int prev, std::vector<double> & q) {
            q.assign(V, 0.0); double s = 0;
            for (int k = 0; k < V; ++k) { q[k] = 0.3 + (double) ((k * 5 + prev * 3 + (k % 3)) % 7); s += q[k]; }
            for (auto & x : q) x /= s;
        };
        for (int arm = 0; arm < 2; ++arm) {
            std::mt19937 rng(777 + arm);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            common_cafe3_ctrl c(W, 0.2, 64);
            std::vector<std::vector<long>> cnt(V, std::vector<long>(V, 0));
            int prev = 0;
            long n_emit = 0;
            std::vector<long> wh(W + 1, 0);
            while (n_emit < 400000) {
                const int sel = arm == 0 ? 3 : c.select(W);
                wh[sel]++;
                // draft chain from q
                std::vector<int> dr; std::vector<std::vector<double>> qs;
                int x = prev;
                for (int i = 0; i < sel; ++i) {
                    std::vector<double> q; Q(x, q);
                    std::discrete_distribution<int> D(q.begin(), q.end());
                    const int t = D(rng); dr.push_back(t); qs.push_back(q); x = t;
                }
                // verify
                int acc = 0; int ctx = prev; bool rejected = false;
                for (int i = 0; i < sel; ++i) {
                    std::vector<double> p; P(ctx, p);
                    const int t = dr[i];
                    if (U(rng) * qs[i][t] <= p[t]) { cnt[ctx][t]++; n_emit++; ctx = t; acc++; continue; }
                    std::vector<double> r(V); for (int k = 0; k < V; ++k) r[k] = std::max(0.0, p[k] - qs[i][k]);
                    std::discrete_distribution<int> R(r.begin(), r.end());
                    const int y = R(rng); cnt[ctx][y]++; n_emit++; ctx = y; rejected = true; break;
                }
                if (!rejected) { // bonus token from the target
                    std::vector<double> p; P(ctx, p);
                    std::discrete_distribution<int> D(p.begin(), p.end());
                    const int y = D(rng); cnt[ctx][y]++; n_emit++; ctx = y;
                }
                prev = ctx;
                // synthetic cost: verify ~ width; controller sees the real acceptance of this law
                if (arm == 1) c.observe(sel, sel, acc, 1000.0 + 200.0 * sel);
            }
            double chi2 = 0.0; int df = 0;
            for (int a = 0; a < V; ++a) {
                long na = 0; for (int b = 0; b < V; ++b) na += cnt[a][b];
                if (na == 0) continue;
                std::vector<double> p; P(a, p);
                for (int b = 0; b < V; ++b) { const double e = na * p[b]; chi2 += (cnt[a][b] - e) * (cnt[a][b] - e) / e; }
                df += V - 1;
            }
            printf("law arm %s: emitted %ld, chi2 %.2f df %d, widths [", arm ? "controller" : "static3", n_emit, chi2, df);
            for (int w = 0; w <= W; ++w) printf("%s%ld", w ? "," : "", wh[w]);
            printf("]\n");
            // df = 56: p = 0.001 critical value 93.17
            CHECK(df == 56 && chi2 < 93.17);
            if (arm == 1) { int used = 0; for (int w = 0; w <= W; ++w) used += wh[w] > 0; CHECK(used >= 3); }
        }
    }

    // (d) default off
    {
        unsetenv("LLAMA_CAFE3_ADAPTIVE");      CHECK(!common_cafe3_ctrl::env_enabled());
        setenv("LLAMA_CAFE3_ADAPTIVE", "0", 1); CHECK(!common_cafe3_ctrl::env_enabled());
        setenv("LLAMA_CAFE3_ADAPTIVE", "true", 1); CHECK(!common_cafe3_ctrl::env_enabled());
        setenv("LLAMA_CAFE3_ADAPTIVE", "1", 1); CHECK(common_cafe3_ctrl::env_enabled());
        unsetenv("LLAMA_CAFE3_ADAPTIVE");
    }

    printf("%s (%d failures)\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 1 : 0;
}
