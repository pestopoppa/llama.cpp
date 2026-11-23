// SPEC-DTREE-SAMPLED-V2 (LLAMA_DTREE_SAMPLED): unit test of the recursive rejection-sampling tree walk
// (common_dtree_rrs_walk) and the T > 0 sibling builder (common_dtree_build_sampled), synthetic distributions, no model.
//
// The draft is a Markov lattice q(. | prev) (as the DFlash2 selector: scores conditioned on the predecessor only);
// the main chain is SAMPLED from q with keyed draws, siblings are the top-q candidates per depth with greedy
// continuation. The served single-draft rejection step (accept x0 iff u * q(x0) <= p(x0), else inverse-CDF residual
// norm(max(p - q, 0)), both on (seed, position) keyed uniforms) is reimplemented below exactly as
// common_sampler_spec_reject_step does it.
//
// 1. lossless: for W in {1, 4, 8, 12}, with and without the main-chain q (DFlash2 / MTP), the first two emitted
//    tokens follow the exact target joint (chi-square, df = 35, p = 0.001).
// 2. W = 1 == the served single-draft rejection loop: same keyed draws -> same tokens, same stop.
// 3. W > 1: siblings are actually drafted (count > 0) and "alternative accepted" > 0; W = 1: alt accepted == 0.
// 4. the sibling walk never changes a token the served path emits at the same node (prefix of the served chain
//    up to the first sibling acceptance is identical).

#include "speculative-tree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

enum { ROLE_SMP = 1, ROLE_ACC = 2, ROLE_RES = 3, ROLE_DRAFT = 4 };

static double u01(uint64_t seed, int pos, int role) {
    return (double) (mix(mix(seed) ^ mix(0xa5a5ULL + (uint64_t) pos * 8 + (uint64_t) role)) >> 11) * (1.0 / 9007199254740992.0);
}

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

// Markov draft: depends on the previous token only (an imperfect, lattice-shaped proposal)
static std::vector<float> draft_q(int prev) {
    std::vector<float> q(V);
    float z = 0;
    for (int v = 0; v < V; ++v) {
        q[v] = (float) std::exp(1.5 * ((double) (mix(977 * (uint64_t) prev + v) % 1000) / 1000.0));
        z += q[v];
    }
    for (auto & x : q) { x /= z; }
    return q;
}

static int inv_cdf(const std::vector<double> & p, double u) {
    double c = 0;
    for (int v = 0; v < V; ++v) { c += p[v]; if (u < c) { return v; } }
    return V - 1;
}

// the served rejection step (common_sampler_spec_reject_step) on keyed uniforms at absolute position pos
static int reject_step(const std::vector<double> & p, const std::vector<float> & q, int x0, uint64_t seed, int pos, bool * acc) {
    const double qx = q[x0], px = p[x0];
    if (qx > 0 && u01(seed, pos, ROLE_ACC) * qx <= px) { *acc = true; return x0; }
    *acc = false;
    std::vector<double> r(V);
    double rs = 0;
    for (int v = 0; v < V; ++v) { r[v] = std::max(0.0, p[v] - (double) q[v]); rs += r[v]; }
    if (rs <= 0) { return inv_cdf(p, u01(seed, pos, ROLE_SMP)); }
    const double tgt = u01(seed, pos, ROLE_RES) * rs;
    double run = 0;
    for (int v = 0; v < V; ++v) { run += r[v]; if (r[v] > 0 && run > tgt) { return v; } }
    return V - 1;
}

struct tree_t {
    std::vector<std::vector<llama_token>> paths; // path 0 = sampled main chain
    std::vector<std::vector<float>>       q;     // main chain's draft dists
};

static tree_t draft_tree(const std::vector<int> & ctx, uint64_t seed, int W) {
    std::vector<std::vector<float>> lat(V); // scores(j, pred) = log q(. | pred); j = 0 uses ctx.back()
    std::vector<std::vector<float>> lq(V);
    for (int v = 0; v < V; ++v) {
        lat[v] = draft_q(v);
        lq[v].resize(V);
        for (int k = 0; k < V; ++k) { lq[v][k] = std::log(lat[v][k]); }
    }
    tree_t t;
    std::vector<int32_t> main;
    int prev = ctx.back();
    for (int j = 0; j < D; ++j) {
        const auto & q = lat[prev];
        std::vector<double> qd(q.begin(), q.end());
        const int x = inv_cdf(qd, u01(seed, 1000 + j, ROLE_DRAFT));
        main.push_back(x);
        t.q.push_back(q);
        prev = x;
    }
    const common_dtree_scores_fn scores = [&](int32_t j, int32_t pred) {
        return lq[j == 0 ? ctx.back() : pred].data();
    };
    const auto alts = common_dtree_build_sampled(D, V, scores, main, t.q, 1 << 20, W);
    CHECK((int) alts.size() <= W - 1);
    t.paths.emplace_back(main.begin(), main.end());
    for (const auto & a : alts) {
        CHECK((int) a.size() == D);
        t.paths.emplace_back(a.begin(), a.end());
    }
    return t;
}

struct round_result { std::vector<int> emitted; int n_acc; int best; int64_t n_alt; };

static round_result run_round(const std::vector<int> & ctx, const tree_t & t, uint64_t seed, bool use_q) {
    const auto & paths = t.paths;
    std::vector<std::vector<int32_t>> rows(paths.size());
    for (size_t k = 0; k < paths.size(); ++k) {
        for (int d = 0; d <= (int) paths[k].size(); ++d) { rows[k].push_back((int32_t) (k * (D + 1) + d)); }
    }
    std::vector<const std::vector<int32_t> *>     pr;
    std::vector<const std::vector<llama_token> *> pt;
    for (size_t k = 0; k < paths.size(); ++k) { pr.push_back(&rows[k]); pt.push_back(&paths[k]); }
    auto ctx_of = [&](int32_t row) {
        const int k = row / (D + 1), d = row % (D + 1);
        auto c = ctx; c.insert(c.end(), paths[k].begin(), paths[k].begin() + d);
        return c;
    };
    round_result r { {}, 0, 0, 0 };
    llama_token bonus = -1;
    int32_t best = -1;
    r.n_acc = common_dtree_rrs_walk(pr, pt,
        [&](int32_t row) {
            const auto c = ctx_of(row);
            return (llama_token) inv_cdf(target_p(c), u01(seed, (int) c.size(), ROLE_SMP));
        },
        use_q ? std::function<bool(int32_t, int32_t, llama_token *)>([&](int32_t row, int32_t d, llama_token * y) {
            const auto c = ctx_of(row);
            bool acc = false;
            *y = reject_step(target_p(c), t.q[d], paths[0][d], seed, (int) c.size(), &acc);
            return acc;
        }) : std::function<bool(int32_t, int32_t, llama_token *)>(),
        [&](llama_token tok) { r.emitted.push_back(tok); }, &bonus, &best, &r.n_alt);
    CHECK(bonus >= 0);
    r.emitted.push_back(bonus);
    r.best = best;
    CHECK((int) r.emitted.size() == r.n_acc + 1);
    return r;
}

// the served single-draft verify (common_sampler_sample_and_accept_n with / without dists) on the main chain only
static std::vector<int> served(const std::vector<int> & ctx, const tree_t & t, uint64_t seed, bool use_q) {
    std::vector<int> out;
    auto c = ctx;
    for (int i = 0; i < D; ++i) {
        const auto p = target_p(c);
        int y; bool acc;
        if (use_q) {
            y = reject_step(p, t.q[i], t.paths[0][i], seed, (int) c.size(), &acc);
        } else {
            y = inv_cdf(p, u01(seed, (int) c.size(), ROLE_SMP));
            acc = y == t.paths[0][i];
        }
        out.push_back(y);
        if (!acc) { return out; }
        c.push_back(y);
    }
    out.push_back(inv_cdf(target_p(c), u01(seed, (int) c.size(), ROLE_SMP)));
    return out;
}

int main() {
    // the ENABLED witness names the V2 recursive rejection verifier (not the V1 target-sample walk)
    {
        const std::string msg = COMMON_DTREE_SAMPLED_ENABLED_MSG;
        CHECK(msg.rfind("llama_dtree_sampled: ENABLED", 0) == 0);
        CHECK(msg.find("recursive rejection (RRS)") != std::string::npos);
        CHECK(msg.find("target-sample walk") == std::string::npos);
    }
    // rejection accept is used only when the dists cover the main chain and the target can roll it back
    {
        CHECK(common_dtree_rrs_q_ok(7, 7, true, false, 0));
        CHECK(common_dtree_rrs_q_ok(7, 7, false, true, 7));
        CHECK(!common_dtree_rrs_q_ok(7, 7, false, true, 6));   // RS bound too small -> downgrade
        CHECK(!common_dtree_rrs_q_ok(0, 7, true, false, 0));   // dists cleared (the V1 defect) -> downgrade
        CHECK(!common_dtree_rrs_q_ok(6, 7, true, false, 0));
        CHECK(!common_dtree_rrs_q_ok(0, 0, true, false, 0));
        CHECK(common_dtree_new_cols({1, 2, 3, 4}, {1, 2, 9, 5}) == 2);
        CHECK(common_dtree_new_cols({1, 2, 3, 4}, {7, 2, 3, 4}) == 4);
        CHECK(common_dtree_new_cols({1, 2, 3, 4}, {1, 2, 3, 4}) == 0);
        CHECK(common_dtree_new_cols({1, 2}, {1, 2, 3}) == 1);
    }
    const std::vector<int> ctx0 = {3, 1};
    const int N = 200000;
    const double crit = 66.62; // chi-square df = 35, p = 0.001

    std::vector<double> joint(V * V);
    {
        auto p0 = target_p(ctx0);
        for (int a = 0; a < V; ++a) {
            auto c = ctx0; c.push_back(a);
            auto p1 = target_p(c);
            for (int b = 0; b < V; ++b) { joint[a * V + b] = p0[a] * p1[b]; }
        }
    }

    for (int use_q = 1; use_q >= 0; --use_q) {
        for (int W : {1, 4, 8, 12}) {
            std::vector<double> cnt(V * V, 0.0);
            double mean_acc = 0, mean_served = 0;
            int64_t n_sib = 0, n_alt = 0;
            for (int s = 0; s < N; ++s) {
                const uint64_t seed = 1000003ULL * (uint64_t) W + 7919ULL * (uint64_t) use_q + (uint64_t) s;
                const auto t = draft_tree(ctx0, seed, W);
                n_sib += (int64_t) t.paths.size() - 1;
                const auto r = run_round(ctx0, t, seed, use_q);
                n_alt += r.n_alt;
                const auto sv = served(ctx0, t, seed, use_q);
                mean_served += (double) sv.size() - 1;
                if (W == 12 && s < 20000) {
                    // DTREE-B coupling: the walk on the first m alternatives emits exactly 1 + arm_cover of the full round
                    std::vector<int32_t> acc(r.emitted.begin(), r.emitted.end() - 1);
                    std::vector<int32_t> mn(t.paths[0].begin(), t.paths[0].end());
                    std::vector<std::vector<int32_t>> al;
                    for (size_t k = 1; k < t.paths.size(); ++k) { al.emplace_back(t.paths[k].begin(), t.paths[k].end()); }
                    int32_t bud = 0;
                    for (size_t m = 0; m <= al.size(); ++m) {
                        if (m > 0) { bud += common_dtree_new_cols(mn, al[m - 1]); }
                        tree_t sub; sub.q = t.q; sub.paths.assign(t.paths.begin(), t.paths.begin() + 1 + m);
                        const auto rs = run_round(ctx0, sub, seed, use_q);
                        CHECK((int) rs.emitted.size() == 1 + common_dtree_arm_cover(mn, al, acc, bud));
                        CHECK(std::equal(rs.emitted.begin(), rs.emitted.end(), r.emitted.begin()));
                    }
                }
                if (W == 1) {
                    // W = 1: bit-identical to the served single-draft verify
                    CHECK(r.emitted == sv);
                    CHECK(r.best == 0 && r.n_alt == 0);
                } else {
                    // the tree only ever extends the served chain: identical up to where the served round stopped
                    // (the served stop token is either the same bonus or the sibling the walk accepted)
                    CHECK(r.emitted.size() >= sv.size());
                    CHECK(std::equal(sv.begin(), sv.end(), r.emitted.begin()));
                }
                std::vector<int> em = r.emitted;
                if (em.size() < 2) {
                    auto c = ctx0; c.insert(c.end(), em.begin(), em.end());
                    em.push_back(inv_cdf(target_p(c), u01(seed ^ 0x5555, (int) c.size(), ROLE_SMP)));
                }
                cnt[em[0] * V + em[1]] += 1;
                mean_acc += r.n_acc;
            }
            double chi = 0;
            for (int i = 0; i < V * V; ++i) {
                const double e = joint[i] * N;
                if (e > 0) { chi += (cnt[i] - e) * (cnt[i] - e) / e; }
            }
            printf("%s W=%2d  siblings drafted=%lld  alternative accepted=%lld  mean accepted/round=%.4f (served chain %.4f)  "
                   "chi2(df=35)=%.2f crit=%.2f %s\n",
                   use_q ? "q-RRS " : "s&c   ", W, (long long) n_sib, (long long) n_alt, mean_acc / N, mean_served / N,
                   chi, crit, chi < crit ? "PASS" : "FAIL");
            CHECK(chi < crit);
            if (W > 1) {
                CHECK(n_sib > 0);
                CHECK(n_alt > 0);
                CHECK(mean_acc > mean_served);
            } else {
                CHECK(n_sib == 0 && n_alt == 0);
            }
        }
    }
    printf("OK\n");
    return 0;
}
