// AK-DTREE-1: unit test of the path decomposition of the path-row tree verify (common/speculative-tree.*).
// No model needed.

#include <cmath>
#include "speculative-tree.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

struct lattice {
    int32_t D, K;
    std::vector<float> s; // [D][K pred][K]
    const float * at(int32_t j, int32_t pred) const { return s.data() + ((size_t) j * K + pred) * K; }
};

static lattice random_lattice(int32_t D, int32_t K, uint32_t seed) {
    lattice L { D, K, {} };
    L.s.resize((size_t) D * K * K);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    for (auto & v : L.s) {
        v = nd(rng);
    }
    return L;
}

static std::vector<int32_t> reference_greedy(const lattice & L) {
    // the linear DFlash2 greedy reader (common/speculative.cpp, is_dflash2 branch)
    std::vector<int32_t> out;
    int32_t pred = 0;
    for (int32_t j = 0; j < L.D; ++j) {
        const float * s = L.at(j, pred);
        pred = (int32_t) std::distance(s, std::max_element(s, s + L.K));
        out.push_back(pred);
    }
    return out;
}

int main() {
    int n_cases = 0;
    for (uint32_t seed = 1; seed <= 200; ++seed) {
        const int32_t D = 1 + (int32_t) (seed % 12);
        const int32_t K = 2 + (int32_t) (seed % 7);
        const lattice L = random_lattice(D, K, seed);
        const common_dtree_scores_fn fn = [&](int32_t j, int32_t pred) { return L.at(j, pred); };
        const auto ref = reference_greedy(L);

        for (int32_t n_nodes : { 0, D + 1, D + 2, D + 5, D + 9, 32 }) {
            for (int32_t max_paths : { 1, 2, 3, 8 }) {
                const auto paths = common_dtree_build_paths(D, K, fn, n_nodes, max_paths);
                ++n_cases;

                // path 0 is exactly the linear greedy draft
                CHECK(!paths.empty());
                CHECK(paths[0] == ref);

                // never more than min(max_paths, 3) paths; chain when the budget or width forbids a branch
                CHECK((int32_t) paths.size() <= std::min(max_paths, COMMON_DTREE_MAX_PATHS));
                if (n_nodes <= D + 1 || max_paths <= 1) {
                    CHECK(paths.size() == 1);
                }

                // equal lengths (one ubatch for the recurrent split), column budget respected
                for (const auto & p : paths) {
                    CHECK((int32_t) p.size() == D);
                    for (int32_t c : p) {
                        CHECK(c >= 0 && c < K);
                    }
                }
                if (paths.size() > 1) {
                    CHECK(common_dtree_n_columns(paths) <= n_nodes);
                }

                // every alternative is a root-to-leaf path: shares a prefix with path 0, differs at the branch,
                // then continues greedily from the branch candidate
                for (size_t a = 1; a < paths.size(); ++a) {
                    const auto & p = paths[a];
                    int32_t j = 0;
                    while (j < D && p[j] == paths[0][j]) {
                        ++j;
                    }
                    CHECK(j < D);
                    int32_t pred = p[j];
                    for (int32_t i = j + 1; i < D; ++i) {
                        const float * s = L.at(i, pred);
                        pred = (int32_t) std::distance(s, std::max_element(s, s + K));
                        CHECK(p[i] == pred);
                    }
                    // distinct paths
                    for (size_t b = 0; b < a; ++b) {
                        CHECK(paths[b] != p);
                    }
                }
            }
        }
    }

    // hand-made lattice: the branch with the most mass that fits is chosen
    {
        // D = 3, K = 3, scores independent of pred. pos0: [5, 4.9, -9]; pos1: [5, -9, -9]; pos2: [5, 4.5, -9]
        lattice L { 3, 3, {} };
        L.s.assign(3 * 3 * 3, -9.0f);
        for (int32_t pred = 0; pred < 3; ++pred) {
            float * p0 = L.s.data() + ((size_t) 0 * 3 + pred) * 3; p0[0] = 5; p0[1] = 4.9f;
            float * p1 = L.s.data() + ((size_t) 1 * 3 + pred) * 3; p1[0] = 5;
            float * p2 = L.s.data() + ((size_t) 2 * 3 + pred) * 3; p2[0] = 5; p2[1] = 4.5f;
        }
        const common_dtree_scores_fn fn = [&](int32_t j, int32_t pred) { return L.at(j, pred); };
        // budget of 1 extra column: only a branch at j = 2 (cost 1) fits
        auto paths = common_dtree_build_paths(3, 3, fn, 5, 3);
        CHECK(paths.size() == 2);
        CHECK((paths[1] == std::vector<int32_t>{ 0, 0, 1 }));
        // budget of 3 extra columns: the j = 0 branch (most mass, cost 3) is taken first
        paths = common_dtree_build_paths(3, 3, fn, 7, 3);
        CHECK(paths.size() == 2);
        CHECK((paths[1] == std::vector<int32_t>{ 1, 0, 0 }));
        // budget of 4 extra columns: j = 0 then j = 2
        paths = common_dtree_build_paths(3, 3, fn, 8, 3);
        CHECK(paths.size() == 3);
        CHECK((paths[1] == std::vector<int32_t>{ 1, 0, 0 }));
        CHECK((paths[2] == std::vector<int32_t>{ 0, 0, 1 }));
        CHECK(common_dtree_n_columns(paths) == 8);
    }

    // DTREE-B auto controller (Infernix TreeWidthController): simulated engine. Each arm a has a true mean
    // emission g[a] (narrower trees are prefixes: a round of arm u yields arms <= u) and a round time T[a].
    {
        auto sim = [&](const double g[3], const double T[3], int rounds, int spike_at, int & n_best_tail, int best) {
            common_dtree_auto c;
            uint32_t rng = 12345u;
            auto u01 = [&]() { rng = rng * 1664525u + 1013904223u; return (double) (rng >> 8) / 16777216.0; };
            n_best_tail = 0;
            for (int r = 0; r < rounds; ++r) {
                const int32_t arm = c.choose(1, 1000);
                int32_t e[3] = { -1, -1, -1 };
                // emissions: nested, each arm adds a chance of a longer path
                const double x = u01();
                for (int a = 0; a <= arm; ++a) {
                    e[a] = 1 + (int32_t) std::floor(g[a] * 2.0 * x + 0.5);
                    if (a > 0) {
                        e[a] = std::max(e[a], e[a - 1]);
                    }
                }
                double t = T[arm] * (0.95 + 0.1 * u01());
                if (r == spike_at) {
                    t *= 20.0;
                }
                c.observe(arm, e, (int64_t) t, 1, 1000);
                if (r >= rounds - 500) {
                    n_best_tail += arm == best;
                }
            }
            return c;
        };
        // arm 1 has the best tokens/s: mean emission 3/4/4.3 per round, times 50/58/75 ms
        const double g1[3] = { 2.0, 3.0, 3.3 };
        const double T1[3] = { 50000, 58000, 75000 };
        int tail = 0;
        sim(g1, T1, 3000, -1, tail, 1);
        CHECK(tail >= 440);   // the widest arm is re-timed 1 round in 32 (~16 of 500)
        // chain best when trees cost too much
        const double T2[3] = { 50000, 70000, 90000 };
        sim(g1, T2, 3000, -1, tail, 0);
        CHECK(tail >= 440);
        // one 20x spike on the chosen arm after settling does not flip the choice
        {
            common_dtree_auto c;
            for (int r = 0; r < 2000; ++r) {
                const int32_t arm = c.choose(1, 1000);
                int32_t e[3] = { 3, 4, 4 };
                c.observe(arm, e, (int64_t) T1[arm], 1, 1000);
            }
            CHECK(c.choose(1, 1000) == 1);
            int32_t e[3] = { 3, 4, -1 };
            c.observe(1, e, (int64_t) T1[1] * 20, 1, 1000);
            int n1 = 0;
            for (int r = 0; r < 31; ++r) {
                const int32_t arm = c.choose(1, 1000);
                n1 += arm == 1;
                int32_t e2[3] = { 3, 4, 4 };
                c.observe(arm, e2, (int64_t) T1[arm], 1, 1000);
            }
            CHECK(n1 >= 29);
        }
        // batches above 4 run the chain; context buckets time separately
        {
            common_dtree_auto c;
            CHECK(c.choose(5, 100) == 0);
            CHECK(common_dtree_auto::bucket(1, 100) != common_dtree_auto::bucket(1, 40000));
            CHECK(common_dtree_auto::bucket(1, 100) != common_dtree_auto::bucket(2, 100));
        }
        CHECK(common_dtree_auto::arm_nodes(0, 7) == 0);
        CHECK(common_dtree_auto::arm_nodes(1, 7) == 12);
        CHECK(common_dtree_auto::arm_nodes(2, 7) == 16);
    }

    // flatten (the BATCH-mode view of the same paths): columns == common_dtree_n_columns, parents precede children
    {
        const std::vector<std::vector<llama_token>> p = { { 10, 11, 12 }, { 20, 21, 22 }, { 10, 11, 30 } };
        const auto t = common_dtree_flatten(5, p);
        CHECK(t.tokens.size() == 8);
        CHECK((t.leaf_col == std::vector<int32_t>{ 3, 6, 7 }));
        CHECK(t.parent[7] == 2 && t.depth[7] == 3 && t.parent[4] == 0);
        for (size_t c = 1; c < t.tokens.size(); ++c) {
            CHECK(t.parent[c] >= 0 && t.parent[c] < (int32_t) c && t.depth[c] == t.depth[t.parent[c]] + 1);
        }
        const std::vector<std::vector<int32_t>> pi = { { 0, 0, 0 }, { 1, 1, 1 }, { 0, 0, 2 } };
        CHECK(common_dtree_n_columns(pi) == (int32_t) t.tokens.size());
    }

    printf("test-spec-dtree-paths: OK (%d cases)\n", n_cases);
    return 0;
}
