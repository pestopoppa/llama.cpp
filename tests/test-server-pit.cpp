// SPEC-PIT.1: unit test for server_pit_masses (tools/server/server-pit.h).
#include "../tools/server/server-pit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++n_fail; } } while (0)

static bool near(float a, double b) { return std::fabs(a - b) < 1e-6; }

int main() {
    // sorted descending, as the sampler leaves them
    std::vector<llama_token_data> c = { {7, 0.f, 0.5f}, {3, 0.f, 0.25f}, {9, 0.f, 0.125f}, {1, 0.f, 0.125f}, {4, 0.f, 0.0f} };
    float less, tie;
    CHECK(server_pit_masses(c.data(), c.size(), 7, less, tie) && near(less, 0.5) && near(tie, 0.5));
    CHECK(server_pit_masses(c.data(), c.size(), 3, less, tie) && near(less, 0.25) && near(tie, 0.25));
    CHECK(server_pit_masses(c.data(), c.size(), 9, less, tie) && near(less, 0.0) && near(tie, 0.25)); // tie group 9+1
    CHECK(!server_pit_masses(c.data(), c.size(), 4, less, tie) && less == -1.0f);  // p == 0: out of support
    CHECK(!server_pit_masses(c.data(), c.size(), 42, less, tie));                   // not a candidate
    // unnormalised input: normalised by total mass
    std::vector<llama_token_data> u = { {0, 0.f, 2.f}, {1, 0.f, 1.f}, {2, 0.f, 1.f} };
    CHECK(server_pit_masses(u.data(), u.size(), 0, less, tie) && near(less, 0.5) && near(tie, 0.5));

    // statistical: tokens drawn from the distribution give a uniform randomized PIT (KS D small)
    std::mt19937_64 rng(123);
    std::vector<llama_token_data> d;
    double z = 0;
    for (int i = 0; i < 200; ++i) { float p = std::exp(-0.05f * i); d.push_back({i, 0.f, p}); z += p; }
    std::vector<double> w;
    for (auto & x : d) w.push_back(x.p);
    std::discrete_distribution<int> pick(w.begin(), w.end());
    std::uniform_real_distribution<double> U(0, 1);
    const int N = 20000;
    std::vector<double> pit;
    for (int k = 0; k < N; ++k) {
        int t = pick(rng);
        CHECK(server_pit_masses(d.data(), d.size(), t, less, tie));
        pit.push_back(less + U(rng) * tie);
    }
    std::sort(pit.begin(), pit.end());
    double D = 0;
    for (int i = 0; i < N; ++i) D = std::max(D, std::max((i + 1.0) / N - pit[i], pit[i] - (double) i / N));
    CHECK(D < 1.63 / std::sqrt((double) N)); // KS 1% critical value
    // a biased draw (always the top token) is far from uniform
    CHECK(server_pit_masses(d.data(), d.size(), 0, less, tie) && less > 0.9f);

    if (n_fail) { fprintf(stderr, "%d failure(s)\n", n_fail); return 1; }
    printf("test-server-pit: OK (KS D = %.4f)\n", D);
    return 0;
}
