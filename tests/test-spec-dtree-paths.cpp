// AK-DTREE-1: unit test of the path decomposition of the path-row tree verify (common/speculative-tree.*).
// No model needed.

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

    // auto chooser: initial timing widest first, then the fastest arm
    {
        common_dtree_auto a;
        for (int i = 0; i < 9; ++i) {
            const int32_t arm = a.choose();
            CHECK(arm == 2 - i / 3);
            a.observe(arm, arm == 1 ? 6 : 4, 100000); // arm 1 is fastest
        }
        int n1 = 0;
        for (int i = 0; i < 64; ++i) {
            const int32_t arm = a.choose();
            n1 += arm == 1;
            a.observe(arm, arm == 1 ? 6 : 4, 100000);
        }
        CHECK(n1 >= 60);
        CHECK(common_dtree_auto::arm_nodes(0, 7) == 0);
        CHECK(common_dtree_auto::arm_nodes(1, 7) == 12);
        CHECK(common_dtree_auto::arm_nodes(2, 7) == 16);
    }

    printf("test-spec-dtree-paths: OK (%d cases)\n", n_cases);
    return 0;
}
