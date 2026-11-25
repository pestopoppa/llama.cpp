// C22 DTREE-PANDO (LLAMA_DTREE_BUILDER=pando): unit test of common_dtree_build_pando on fixed synthetic lattices, no model.
//
// The reference below is an independent best-first builder written as an exhaustive frontier scan (no heap): at each
// step every tree node's top-k children not yet in the tree are scored by their cumulative draft probability
// (recomputed from scratch along the path) and the best one joins (ties: lexicographically smaller index path).
// Checks, over many seeded lattices (D, K, k, N, temperature, sampled or greedy main):
//  1. the builder's off-main node set equals the reference's, and its leaves (in insertion order) equal the reference's;
//  2. the main chain is whole: tree columns (anchor + main + alternatives' new columns) == N when the frontier allows;
//  3. no node has more than k off-main children (+ the main child), every alternative branches off the tree.

#include "speculative-tree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s (case %d)\n", __FILE__, __LINE__, #cond, g_case); exit(1); } } while (0)

static int g_case = 0;

static uint64_t g_rng = 0;
static uint64_t rnd() {
    g_rng += 0x9e3779b97f4a7c15ULL;
    uint64_t x = g_rng;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static double unif() { return (double) (rnd() >> 11) * (1.0 / 9007199254740992.0); }

using path_t = std::vector<int32_t>;

// softmax of s / tau with the same arithmetic as the builder (float-rounded s / tau, double exp, Neumaier sum), so exact
// ties in the forced-tie lattices compare equal on both sides; the tree SELECTION logic is what is under test
static std::vector<double> ref_dist(const float * s, int32_t K, double tau) {
    std::vector<double> p(K);
    std::vector<float> x(K);
    for (int32_t k = 0; k < K; ++k) x[k] = (float) ((double) s[k] / tau);
    double mx = (double) x[0];
    for (int32_t k = 1; k < K; ++k) mx = std::max(mx, (double) x[k]);
    double sum = 0.0, c = 0.0;
    for (int32_t k = 0; k < K; ++k) {
        p[k] = std::exp((double) x[k] - mx);
        const double t = sum + p[k];
        c += std::fabs(sum) >= std::fabs(p[k]) ? (sum - t) + p[k] : (p[k] - t) + sum;
        sum = t;
    }
    if (c != 0.0 && std::isfinite(c)) sum += c;
    for (auto & v : p) v = v / sum;
    return p;
}

struct lattice {
    int32_t D, K;
    std::vector<float> sc; // [D][K pred][K]
    const float * at(int32_t j, int32_t pred) const { return sc.data() + ((size_t) j * K + pred) * K; }
};

static std::vector<path_t> ref_pando(const lattice & L, const path_t & main, double tau, int32_t N, int32_t kc,
                                     std::vector<path_t> * order_out) {
    const int32_t D = L.D, K = L.K;
    std::set<path_t> in;
    std::vector<path_t> nodes; // anchor-rooted tree nodes (main prefixes first)
    nodes.push_back({});
    for (int32_t j = 1; j <= D; ++j) { path_t p(main.begin(), main.begin() + j); in.insert(p); nodes.push_back(p); }
    auto cum = [&](const path_t & p) {
        double v = 1.0;
        for (size_t j = 0; j < p.size(); ++j) v *= ref_dist(L.at((int32_t) j, j == 0 ? 0 : p[j - 1]), K, tau)[p[j]];
        return v;
    };
    std::vector<path_t> order;
    while ((int32_t) order.size() < N - 1 - D) {
        bool have = false; double bv = -1.0; path_t best;
        for (const auto & u : nodes) {
            const int32_t j = (int32_t) u.size();
            if (j >= D) continue;
            const auto p = ref_dist(L.at(j, j == 0 ? 0 : u.back()), K, tau);
            std::vector<int32_t> idx(K);
            for (int32_t k = 0; k < K; ++k) idx[k] = k;
            std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return p[a] > p[b]; });
            for (int32_t r = 0; r < kc && r < K; ++r) {
                path_t c = u; c.push_back(idx[r]);
                if (in.count(c) || !(p[idx[r]] > 0.0)) continue;
                const double v = cum(c);
                if (!(v > 0.0)) continue;
                if (!have || v > bv || (v == bv && c < best)) { have = true; bv = v; best = c; }
            }
        }
        if (!have) break;
        in.insert(best); nodes.push_back(best); order.push_back(best);
    }
    std::vector<path_t> leaves;
    for (const auto & a : order) {
        bool leaf = true;
        for (const auto & b : order) if (b.size() == a.size() + 1 && std::equal(a.begin(), a.end(), b.begin())) { leaf = false; break; }
        if (leaf) leaves.push_back(a);
    }
    if (order_out) *order_out = order;
    return leaves;
}

int main() {
    int n_cases = 0, n_full = 0, n_tie = 0;
    for (int it = 0; it < 4000; ++it) {
        g_case = it;
        g_rng = 0x5eed0000ULL + (uint64_t) it * 7919ULL;
        lattice L;
        L.D = 2 + (int32_t) (rnd() % 8);   // 2..9
        L.K = 2 + (int32_t) (rnd() % 7);   // 2..8
        const int32_t kc = 1 + (int32_t) (rnd() % 4);
        const int32_t N = L.D + 2 + (int32_t) (rnd() % 16);
        const bool tie = it % 10 == 0;     // quantized scores: many exact ties
        const float T = (it % 3 == 0) ? 0.0f : (it % 3 == 1 ? 0.3f : 1.0f);
        const double tau = T > 0.0f ? (double) T : 1.0;
        L.sc.resize((size_t) L.D * L.K * L.K);
        for (auto & s : L.sc) s = tie ? (float) (rnd() % 3) : (float) (4.0 * unif() - 2.0);
        n_tie += tie;
        const common_dtree_scores_fn scores = [&](int32_t j, int32_t pred) { return L.at(j, pred); };

        // main: greedy (empty -> builder computes argmax) at T = 0, else sampled from the drafter's own law
        path_t main;
        if (T > 0.0f) {
            int32_t pred = 0;
            for (int32_t j = 0; j < L.D; ++j) {
                const auto p = ref_dist(L.at(j, pred), L.K, tau);
                double u = unif(), c = 0.0; int32_t k = 0;
                for (; k < L.K - 1; ++k) { c += p[k]; if (u < c) break; }
                main.push_back(k); pred = k;
            }
        }
        const auto got = common_dtree_build_pando(L.D, L.K, scores, main, T, N, kc);
        path_t mref = main;
        if (mref.empty()) {
            int32_t pred = 0;
            for (int32_t j = 0; j < L.D; ++j) {
                const float * s = L.at(j, pred);
                pred = (int32_t) std::distance(s, std::max_element(s, s + L.K));
                mref.push_back(pred);
            }
        }
        std::vector<path_t> order;
        const auto want = ref_pando(L, mref, tau, N, kc, &order);

        // 1. same leaves in the same order (hence the same tree: a tree is the prefix closure of its leaves + main)
        if (got != want) {
            fprintf(stderr, "case %d D=%d K=%d k=%d N=%d T=%.1f: got %zu leaves, want %zu\n", it, L.D, L.K, kc, N, T,
                    got.size(), want.size());
        }
        CHECK(got == want);

        // 2. columns: anchor + main D + off-main nodes
        std::vector<path_t> all = { mref };
        all.insert(all.end(), got.begin(), got.end());
        CHECK(common_dtree_n_columns(all) == 1 + L.D + (int32_t) order.size());
        if ((int32_t) order.size() == N - 1 - L.D) {
            CHECK(common_dtree_n_columns(all) == N);
            ++n_full;
        }
        // 3. child cap and branching structure
        std::map<path_t, std::set<int32_t>> kids;
        for (const auto & a : order) {
            CHECK(!a.empty() && (int32_t) a.size() <= L.D);
            kids[path_t(a.begin(), a.end() - 1)].insert(a.back());
            bool on_main = std::equal(a.begin(), a.end(), mref.begin());
            CHECK(!on_main);
        }
        for (const auto & kv : kids) CHECK((int32_t) kv.second.size() <= kc);
        ++n_cases;
    }
    // degenerate budgets: chain only
    {
        lattice L; L.D = 4; L.K = 3; L.sc.assign((size_t) 4 * 3 * 3, 0.5f);
        const common_dtree_scores_fn scores = [&](int32_t j, int32_t pred) { return L.at(j, pred); };
        CHECK(common_dtree_build_pando(4, 3, scores, {}, 0.0f, 5, 3).empty());
        CHECK(common_dtree_build_pando(4, 3, scores, {}, 0.0f, 6, 3).size() == 1);
    }
    printf("test-dtree-pando: OK (%d lattices, %d with full budget, %d forced-tie)\n", n_cases, n_full, n_tie);
    return 0;
}
