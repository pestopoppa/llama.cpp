// SPEC-DTREE-DICE V2 (LLAMA_DTREE_SAMPLED): unit test of common_dtree_sampled_walk with synthetic logits. No model.
//
// 1. lossless: for W in {1, 4, 8, 12} tree paths, the emitted tokens of a round (accepted + bonus), extended to
//    2 tokens by plain target sampling, follow the exact target joint (chi-square, df = V^2 - 1, p = 0.001), and
//    are bit-identical to plain autoregressive sampling with the same (seed, position) keyed draws.
// 2. W = 1 reduces to standard single-draft sampling (sample-and-compare chain accept, common_sampler_sample_and_accept_n).
// 3. T = 0: with an argmax "sample" the walk length equals the greedy tree accept (max over paths of
//    common_dtree_accept_walk on argmax picks), and the main chain wins ties (bit-identical greedy path).

#include "speculative-tree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static constexpr int V = 6;
static constexpr int D = 3;

static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// keyed draw u in [0, 1) from (seed, position)
static double u01(uint64_t seed, int pos) {
    return (double) (mix(mix(seed) ^ mix(0xa5a5ULL + (uint64_t) pos)) >> 11) * (1.0 / 9007199254740992.0);
}

// target distribution after a context (prefix of tokens): deterministic in the context
static std::vector<double> target_p(const std::vector<int> & ctx) {
    uint64_t h = 1469598103934665603ULL;
    for (int t : ctx) { h = mix(h ^ (uint64_t) (t + 1)); }
    std::vector<double> p(V);
    double z = 0;
    for (int v = 0; v < V; ++v) {
        p[v] = std::exp(2.0 * ((double) (mix(h + v) % 1000) / 1000.0 - 0.5) * 2.0);
        z += p[v];
    }
    for (auto & x : p) { x /= z; }
    return p;
}

// draft distribution: a perturbed target (so the tree is useful but imperfect)
static std::vector<double> draft_p(const std::vector<int> & ctx) {
    auto p = target_p(ctx);
    uint64_t h = 7;
    for (int t : ctx) { h = mix(h ^ (uint64_t) (t + 11)); }
    double z = 0;
    for (int v = 0; v < V; ++v) { p[v] *= 0.5 + (double) (mix(h + v) % 1000) / 1000.0; z += p[v]; }
    for (auto & x : p) { x /= z; }
    return p;
}

static int inv_cdf(const std::vector<double> & p, double u) {
    double c = 0;
    for (int v = 0; v < V; ++v) { c += p[v]; if (u < c) { return v; } }
    return V - 1;
}

static int argmax(const std::vector<double> & p) {
    return (int) (std::max_element(p.begin(), p.end()) - p.begin());
}

// fixed top-q tree: W root-to-leaf paths of depth D, the W most probable draft sequences (beam over the draft)
static std::vector<std::vector<llama_token>> top_paths(const std::vector<int> & ctx, int W) {
    std::vector<std::pair<double, std::vector<int>>> beam { {1.0, {}} };
    for (int d = 0; d < D; ++d) {
        std::vector<std::pair<double, std::vector<int>>> nb;
        for (auto & [pr, seq] : beam) {
            auto c = ctx; c.insert(c.end(), seq.begin(), seq.end());
            auto q = draft_p(c);
            for (int v = 0; v < V; ++v) { auto s = seq; s.push_back(v); nb.push_back({pr * q[v], s}); }
        }
        std::stable_sort(nb.begin(), nb.end(), [](auto & a, auto & b) { return a.first > b.first; });
        nb.resize(std::min<size_t>(nb.size(), (size_t) W));
        beam = nb;
    }
    std::vector<std::vector<llama_token>> out;
    for (auto & b : beam) { out.emplace_back(b.second.begin(), b.second.end()); }
    return out;
}

struct round_result { int n_acc; int best; std::vector<int> emitted; };

// one verified round from context ctx: rows are (path, depth) -> encoded as path * (D + 1) + depth
static round_result run_round(const std::vector<int> & ctx, const std::vector<std::vector<llama_token>> & paths,
                              uint64_t seed, bool greedy) {
    std::vector<std::vector<int32_t>> rows(paths.size());
    for (size_t k = 0; k < paths.size(); ++k) {
        for (int d = 0; d <= (int) paths[k].size(); ++d) { rows[k].push_back((int32_t) (k * (D + 1) + d)); }
    }
    std::vector<const std::vector<int32_t> *>     pr;
    std::vector<const std::vector<llama_token> *> pt;
    for (size_t k = 0; k < paths.size(); ++k) { pr.push_back(&rows[k]); pt.push_back(&paths[k]); }
    round_result r;
    int32_t best = -1;
    r.n_acc = common_dtree_sampled_walk(pr, pt,
        [&](int32_t row) {
            const int k = row / (D + 1), d = row % (D + 1);
            auto c = ctx; c.insert(c.end(), paths[k].begin(), paths[k].begin() + d);
            auto p = target_p(c);
            const int tok = greedy ? argmax(p) : inv_cdf(p, u01(seed, (int) c.size()));
            r.emitted.push_back(tok); // last one is the bonus
            return (llama_token) tok;
        },
        [&](llama_token) {}, &best);
    r.best = best;
    CHECK((int) r.emitted.size() == r.n_acc + 1);
    return r;
}

static std::vector<int> autoregressive(const std::vector<int> & ctx, uint64_t seed, int n) {
    auto c = ctx;
    std::vector<int> out;
    for (int i = 0; i < n; ++i) {
        const int t = inv_cdf(target_p(c), u01(seed, (int) c.size()));
        out.push_back(t); c.push_back(t);
    }
    return out;
}

int main() {
    const std::vector<int> ctx0 = {3, 1};
    const int N = 200000;
    const double crit = 66.62; // chi-square df = 35, p = 0.001

    // exact target joint over the first two emitted tokens
    std::vector<double> joint(V * V);
    {
        auto p0 = target_p(ctx0);
        for (int a = 0; a < V; ++a) {
            auto c = ctx0; c.push_back(a);
            auto p1 = target_p(c);
            for (int b = 0; b < V; ++b) { joint[a * V + b] = p0[a] * p1[b]; }
        }
    }

    for (int W : {1, 4, 8, 12}) {
        const auto paths = top_paths(ctx0, W);
        CHECK((int) paths.size() == W);
        std::vector<double> cnt(V * V, 0.0);
        double mean_acc = 0;
        for (int s = 0; s < N; ++s) {
            const uint64_t seed = 1000003ULL * (uint64_t) W + (uint64_t) s;
            auto r = run_round(ctx0, paths, seed, false);
            std::vector<int> em = r.emitted;
            if (em.size() < 2) {
                auto c = ctx0; c.insert(c.end(), em.begin(), em.end());
                em.push_back(inv_cdf(target_p(c), u01(seed, (int) c.size())));
            }
            // bit-identical to plain autoregressive sampling with the same keyed draws
            const auto ar = autoregressive(ctx0, seed, (int) r.emitted.size());
            CHECK(std::equal(ar.begin(), ar.end(), r.emitted.begin()));
            cnt[em[0] * V + em[1]] += 1;
            mean_acc += r.n_acc;

            if (W == 1) {
                // standard single-draft sampling: sample each row, accept while equal to the draft (accept_n)
                std::vector<int> c = ctx0;
                int n = 0;
                for (; n < D; ++n) {
                    const int t = inv_cdf(target_p(c), u01(seed, (int) c.size()));
                    if (t != paths[0][n]) { break; }
                    c.push_back(t);
                }
                CHECK(n == r.n_acc && r.best == 0);
            }
        }
        double chi = 0;
        for (int i = 0; i < V * V; ++i) {
            const double e = joint[i] * N;
            if (e > 0) { chi += (cnt[i] - e) * (cnt[i] - e) / e; }
        }
        printf("W=%2d  mean accepted/round=%.4f  chi2(df=35)=%.2f  crit(p=0.001)=%.2f  %s\n",
               W, mean_acc / N, chi, crit, chi < crit ? "PASS" : "FAIL");
        CHECK(chi < crit);
    }

    // T = 0: argmax walk == greedy tree accept, main chain wins ties
    for (int W : {1, 4, 8, 12}) {
        for (int t = 0; t < 200; ++t) {
            std::vector<int> ctx = {t % V, (t / V) % V, t};
            const auto paths = top_paths(ctx, W);
            auto r = run_round(ctx, paths, 0, true);
            int best_n = -1, best_k = -1;
            for (size_t k = 0; k < paths.size(); ++k) {
                std::vector<llama_token> picks;
                auto c = ctx;
                for (int d = 0; d < D; ++d) {
                    picks.push_back(argmax(target_p(c)));
                    c.push_back(paths[k][d]);
                }
                const int n = common_dtree_accept_walk(picks, paths[k]);
                if (n > best_n) { best_n = n; best_k = (int) k; }
            }
            CHECK(r.n_acc == best_n);
            CHECK(r.best == best_k);
        }
    }
    printf("T=0 greedy equivalence: PASS\n");
    printf("OK\n");
    return 0;
}
