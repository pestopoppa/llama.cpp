// SPEC-DTREE-DICE step 2a: unit test of the capture's forced superset tree (no model).
//   - every node's parent precedes it, the node count respects the column budget, the chain is fully present;
//   - the probability-greedy and DICE-style best-first orders equal a brute-force sort of all lattice paths by their
//     cumulative score (both scores are non-increasing along a path, so best-first = global top-n);
//   - the tree's present nodes of each order form a rank prefix 1..m (the replay's coverage rule);
//   - common_dice_tree_alts + the chain cover exactly the tree's nodes (the batch builder's column count);
//   - the keyed Gumbel draw is deterministic and has mean ~ Euler-Mascheroni.

#include "dice-capture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++n_fail; } } while (0)

struct lattice {
    int D, K;
    std::vector<float> sc;  // [j][pred][k]
    std::vector<int>   tok; // [j][k]
    const float * scores(int j, int pred) const { return sc.data() + ((size_t) j * K + pred) * K; }
};

static lattice make_lattice(int D, int K, uint32_t seed) {
    lattice L{D, K, {}, {}};
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    L.sc.resize((size_t) D * K * K);
    for (auto & x : L.sc) {
        x = nd(rng);
    }
    L.tok.resize((size_t) D * K);
    for (int j = 0; j < D; ++j) {
        for (int k = 0; k < K; ++k) {
            L.tok[(size_t) j * K + k] = 1000 * (j + 1) + 7 * k + 3; // distinct per depth
        }
    }
    return L;
}

static std::vector<double> lsm(const float * s, int K, float T) {
    std::vector<double> o(K);
    double mx = -INFINITY;
    for (int k = 0; k < K; ++k) mx = std::max(mx, (double) s[k] / T);
    double sum = 0;
    for (int k = 0; k < K; ++k) sum += std::exp((double) s[k] / T - mx);
    for (int k = 0; k < K; ++k) o[k] = (double) s[k] / T - mx - std::log(sum);
    return o;
}

// all paths up to depth D with their cumulative score under edge(j, pred) increments
static void all_paths(const lattice & L, const std::function<std::vector<double>(int, int)> & edge,
                      std::vector<std::pair<double, std::vector<int>>> & out) {
    std::function<void(std::vector<int> &, double)> rec = [&](std::vector<int> & p, double sc) {
        const int j = (int) p.size();
        if (j >= L.D) return;
        const auto inc = edge(j, j == 0 ? 0 : p.back());
        for (int k = 0; k < L.K; ++k) {
            p.push_back(k);
            out.push_back({sc + inc[k], p});
            rec(p, sc + inc[k]);
            p.pop_back();
        }
    };
    std::vector<int> p;
    rec(p, 0.0);
}

static std::vector<int> path_of(const std::vector<common_dice_node> & nodes, int i) {
    std::vector<int> p;
    for (int c = i; c > 0; c = nodes[c].par) p.push_back(nodes[c].k);
    std::reverse(p.begin(), p.end());
    return p;
}

int main() {
    // Gumbel: deterministic, mean ~ 0.5772
    {
        CHECK(common_dice_gumbel(42, 100, 7) == common_dice_gumbel(42, 100, 7));
        CHECK(common_dice_gumbel(42, 100, 7) != common_dice_gumbel(42, 101, 7));
        CHECK(common_dice_gumbel(42, 100, 7) != common_dice_gumbel(42, 100, 8));
        double m = 0;
        const int n = 200000;
        for (int i = 0; i < n; ++i) {
            const double g = common_dice_gumbel(1234, (uint64_t) i, i % 997);
            CHECK(std::isfinite(g));
            m += g;
        }
        m /= n;
        CHECK(std::fabs(m - 0.5772156649) < 0.02);
    }

    for (int trial = 0; trial < 40; ++trial) {
        const int D = trial % 2 ? 3 : 4;
        const int K = trial % 3 ? 4 : 3;
        const float T = trial % 4 == 0 ? 1.0f : (trial % 4 == 1 ? 0.3f : (trial % 4 == 2 ? 0.1f : 0.7f));
        const uint64_t seed = 1000 + trial;
        const uint64_t n0 = 500 + 17 * trial;
        const lattice L = make_lattice(D, K, 77 + trial);
        const common_dice_scores_fn sfn = [&](int32_t j, int32_t pred) { return L.scores(j, pred); };
        const common_dice_tok_fn tfn = [&](int32_t j, int32_t k) { return (llama_token) L.tok[(size_t) j * K + k]; };
        std::mt19937 rng(trial);
        std::vector<int32_t> chain_k;
        for (int j = 0; j < D; ++j) chain_k.push_back((int32_t) (rng() % K));

        const int n_cols = 6 + trial % 10;
        const auto nodes = common_dice_build_tree(D, K, sfn, tfn, chain_k, T, seed, n0, n_cols);

        CHECK((int) nodes.size() <= n_cols);
        CHECK(nodes[0].par == -1);
        std::set<std::vector<int>> present;
        for (size_t i = 1; i < nodes.size(); ++i) {
            CHECK(nodes[i].par >= 0 && nodes[i].par < (int) i);
            CHECK(nodes[i].depth == (nodes[i].par == 0 ? 0 : nodes[nodes[i].par].depth + 1));
            CHECK(nodes[i].tok == tfn(nodes[i].depth, nodes[i].k));
            CHECK(present.insert(path_of(nodes, (int) i)).second); // no duplicate node
        }
        for (int j = 0; j < D; ++j) { // chain fully present (it is added first and fits: n_cols >= D + 1 here)
            if (n_cols >= D + 1) {
                CHECK(present.count(std::vector<int>(chain_k.begin(), chain_k.begin() + j + 1)) == 1);
            }
        }

        // brute-force orders
        const auto prob_edge = [&](int j, int pred) { return lsm(L.scores(j, pred), K, T); };
        const auto dice_edge = [&](int j, int pred) {
            auto o = lsm(L.scores(j, pred), K, T);
            double mx = -INFINITY;
            for (int k = 0; k < K; ++k) { o[k] += common_dice_gumbel(seed, n0 + j, tfn(j, k)); mx = std::max(mx, o[k]); }
            for (int k = 0; k < K; ++k) o[k] -= mx;
            return o;
        };
        for (int which = 0; which < 2; ++which) {
            std::vector<std::pair<double, std::vector<int>>> all;
            all_paths(L, which == 0 ? std::function<std::vector<double>(int, int)>(prob_edge)
                                    : std::function<std::vector<double>(int, int)>(dice_edge), all);
            std::stable_sort(all.begin(), all.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            // ranks of present nodes must equal their brute-force rank, and form a prefix 1..m
            std::map<std::vector<int>, int> rank;
            for (size_t r = 0; r < all.size(); ++r) rank[all[r].second] = (int) r + 1;
            std::set<int> got;
            for (size_t i = 1; i < nodes.size(); ++i) {
                const int r = which == 0 ? nodes[i].pr : nodes[i].dr;
                if (r > 0) {
                    CHECK(rank[path_of(nodes, (int) i)] == r);
                    got.insert(r);
                }
            }
            int m = 0;
            while (got.count(m + 1)) ++m;
            // interleaving guarantees roughly half the non-chain budget to each order
            CHECK(m >= std::min<int>((n_cols - 1 - D) / 2, (int) all.size()));
        }

        // alts + chain cover exactly the nodes
        const auto alts = common_dice_tree_alts(nodes);
        std::set<std::vector<llama_token>> cols;
        std::vector<llama_token> chain_tok;
        for (int j = 0; j < D && n_cols >= D + 1; ++j) chain_tok.push_back(tfn(j, chain_k[j]));
        for (size_t j = 1; j <= chain_tok.size(); ++j) cols.insert(std::vector<llama_token>(chain_tok.begin(), chain_tok.begin() + j));
        for (const auto & a : alts) {
            for (size_t j = 1; j <= a.size(); ++j) cols.insert(std::vector<llama_token>(a.begin(), a.begin() + j));
        }
        if (n_cols >= D + 1) {
            CHECK(cols.size() == nodes.size() - 1);
        }
    }

    if (n_fail) {
        std::fprintf(stderr, "test-dice-capture: %d failure(s)\n", n_fail);
        return 1;
    }
    std::printf("test-dice-capture: OK\n");
    return 0;
}
