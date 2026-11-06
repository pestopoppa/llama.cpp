// decode-aware prefill budget controller (tools/server/server-prefill-budget.h). No model needed.
//
//   test-server-prefill-budget                 synthetic checks + closed-loop replay on the KVU-16g-3 plant
//   test-server-prefill-budget <cell.csv> ...  also replay recorded server logs (rows "decode_rows,prompt,t_iter_ms,
//                                              logged_ms_per_tok", one per `prefill-budget: iter` line) through the
//                                              legacy estimator and require the logged ms/tok to be reproduced

#include "server-prefill-budget.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int n_fail  = 0;
static int n_check = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        n_check++;                                                               \
        if (!(cond)) {                                                           \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            n_fail++;                                                            \
        }                                                                        \
    } while (0)

static server_prefill_budget_ctl make(int32_t cap, int32_t target, int32_t align, double k) {
    server_prefill_budget_ctl c;
    c.cap = cap; c.target_ms = target; c.align = align; c.tail_k = k;
    return c;
}

// the pre-fix server-context.cpp prefill_budget_get(), verbatim arithmetic
static int32_t legacy_get(int32_t n_batch, int32_t n_generating, int32_t cap, int32_t target, double ms_per_tok) {
    if (n_generating == 0) return 0;
    int32_t budget = n_batch;
    if (cap > 0) budget = std::min(budget, cap);
    if (target > 0 && ms_per_tok > 0.0) {
        const int32_t adaptive = (int32_t) std::min<double>((double) target / ms_per_tok, (double) n_batch);
        budget = std::min(budget, std::max(64, adaptive));
    }
    return budget < n_batch ? budget : 0;
}

static void test_legacy_identity() {
    const double mpt[] = {0.0, 0.9, 1.3, 2.4, 4.4, 6.4, 12.0, 30.0};
    for (int32_t cap : {0, 256, 512}) {
        for (int32_t target : {0, 460, 670, 900}) {
            for (double m : mpt) {
                for (int32_t dr : {0, 8, 24}) {
                    auto c = make(cap, target, 0, 0.0);
                    c.ms_per_tok = m;
                    CHECK(c.get(512, dr, 3) == legacy_get(512, 3, cap, target, m));
                    CHECK(c.get(512, dr, 0) == 0);
                }
            }
        }
    }
    // legacy update rule: first sample sets, then 0.5/0.5; chunks < 64 do not update
    auto c = make(256, 670, 0, 0.0);
    c.observe(600.0, 32, 24);  CHECK(c.ms_per_tok == 0.0);
    c.observe(600.0, 200, 24); CHECK(std::fabs(c.ms_per_tok - 3.0) < 1e-12);
    c.observe(700.0, 100, 24); CHECK(std::fabs(c.ms_per_tok - 5.0) < 1e-12);
    CHECK(c.target_eff() == 670.0); // tail_k = 0: never a margin
}

static void test_align() {
    auto c = make(512, 670, 32, 0.0);
    c.ms_per_tok = 670.0 / 152.0;          // wants 152 prompt tokens
    CHECK(c.get(512, 24, 3) == 136);       // 24 + ~152 = ~176 rows (5.5 tiles) -> 160 rows -> 136 tokens
    c.ms_per_tok = 670.0 / 154.5;          // wants 154 -> 178 rows -> 160 rows -> 136 tokens
    CHECK(c.get(512, 24, 3) == 136);
    c.ms_per_tok = 670.0 / 168.5;          // wants 168 -> 192 rows (aligned) -> 168
    CHECK(c.get(512, 24, 3) == 168);
    c.ms_per_tok = 670.0 / 70.5;           // wants 70 -> 94 rows -> 64 rows -> 40 < MIN: keep 70 unaligned
    CHECK(c.get(512, 24, 3) == 70);
    c.ms_per_tok = 670.0 / 30.0;           // below MIN: MIN
    CHECK(c.get(512, 24, 3) == 64);
    c.ms_per_tok = 670.0 / 300.0;          // the decoding cap still bounds it
    auto d = make(256, 670, 32, 0.0); d.ms_per_tok = c.ms_per_tok;
    CHECK(d.get(512, 24, 3) == 256);
    c.ms_per_tok = 0.5;                    // adaptive >= n_batch: no budget
    CHECK(c.get(512, 24, 3) == 0);
    // every aligned answer puts the batch rows on a tile boundary
    for (int32_t dr : {8, 16, 24}) {
        for (int n = 100; n < 480; n++) {
            auto e = make(0, 670, 32, 0.0);
            e.ms_per_tok = 670.0 / (n + 0.5);
            const int32_t b = e.get(512, dr, 3);
            CHECK(b > 0 && b <= n && (dr + b) % 32 == 0 && n - b < 32);
        }
    }
}

static void test_tail() {
    auto c = make(0, 670, 0, 1.0);
    c.observe(600.0, 150, 24);             // first: no residual yet
    CHECK(c.n_resid == 0 && c.target_eff() == 670.0);
    for (int i = 0; i < 3; i++) c.observe(c.ms_per_tok * 150 + 30.0, 150, 24);
    CHECK(c.n_resid == 3 && c.target_eff() == 670.0); // warm-up
    c.observe(c.ms_per_tok * 150 + 30.0, 150, 24);
    CHECK(c.n_resid == 4 && std::fabs(c.target_eff() - 640.0) < 1e-9);
    // no decode rows (solo prefill): residual not collected
    c.observe(800.0, 512, 0);
    CHECK(c.n_resid == 4);
    // floor
    auto f = make(0, 600, 0, 50.0);
    for (int i = 0; i < 8; i++) f.observe(500.0 + (i % 2 ? 200.0 : -200.0), 100, 24);
    CHECK(f.target_eff() == 300.0);
}

// ---------------------------------------------------------------------------------------------------------------
// closed-loop replay on the KVU-16g-3 plant: iteration ms as a function of total KV positions and batch rows,
// least-squares fit to the 4016 decode_rows = 24 iterations of all 14 cells of
// /mnt/raid0/llm/tmp/kvu16g3-retune-20261105/out/cells (Qwen3.8-27B Q8_0 + DFlash2, C6 729599d69, MI210, -ub 512):
//   t = b0 + b1 x + b2 x^2 + B (b3 + b4 x + b5 x^2) + s[B mod 32],  x = kv_pos / 1e5,  s[0] = 0
// The step table s is the measured staircase (cost flat inside a 32-row tile, ~60 ms jump at each boundary).
// Open-loop noise at a fixed chunk was sd 7.6-8.7 ms (cells a1-1-ON / a3-1-ON at budget 256); modelled as N(0, 8).
static const double PB[6] = {-86.7096, 536.9204, -197.6843, 0.5656, 0.7953, 0.4688};
static const double PS[32] = {0.0,
    66.0282, 62.4488, 63.2665, 57.7929, 54.9328, 51.8509, 51.5020, 46.2450, 38.4528, 39.3676, 35.3302, 34.0784,
    28.3399, 27.8195, 26.2114, 22.6501, 49.2349, 54.3593, 50.6433, 33.7470, 34.3349, 35.0974, 21.7966, 20.3170,
    16.2151, 6.2475, 4.2306, 7.9164, 4.7037, 4.4638, 6.0499};

static double plant(double kv_pos, int32_t rows) {
    const double x = kv_pos / 1e5, B = rows;
    return PB[0] + PB[1] * x + PB[2] * x * x + B * (PB[3] + PB[4] * x + PB[5] * x * x) + PS[rows % 32];
}

struct rng_t {
    uint64_t s;
    double u() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 11) + 0.5) / 9007199254740992.0; }
    double n() { return std::sqrt(-2.0 * std::log(u())) * std::cos(6.283185307179586 * u()); }
};

struct replay_out { double p99, mean, ttft_s; };

// KVU-16g-3 collision: 3 decoders (24 decode rows) at ~63k total KV positions, an 80k neighbour arriving
static replay_out replay(server_prefill_budget_ctl c, uint64_t seed) {
    rng_t r{seed};
    double kv = 63000.0; int32_t left = 80000; std::vector<double> ts;
    const int32_t n_batch = 512, dr = 24;
    while (left > 0) {
        const int32_t b = c.get(n_batch, dr, 3);
        int32_t n = std::min(b > 0 ? b : n_batch - dr, left);
        const double t = plant(kv, dr + n) + 8.0 * r.n();
        ts.push_back(t);
        c.observe(t, n, dr);
        kv += n + 8; left -= n;
    }
    std::vector<double> s = ts; std::sort(s.begin(), s.end());
    double sum = 0; for (double t : ts) sum += t;
    return {s[(size_t) (0.99 * (s.size() - 1))], sum / ts.size(), sum / 1e3};
}

static replay_out replay_avg(const server_prefill_budget_ctl & c) {
    replay_out a{0, 0, 0};
    const int N = 16;
    for (int i = 0; i < N; i++) { auto o = replay(c, 1000 + i); a.p99 += o.p99 / N; a.mean += o.mean / N; a.ttft_s += o.ttft_s / N; }
    return a;
}

static void test_replay() {
    const auto off = replay_avg(make(0, 0, 0, 0.0));
    for (int32_t target : {670, 900}) {
        const auto legacy = replay_avg(make(256, target, 0, 0.0));
        const auto fixed  = replay_avg(make(256, target, 32, 1.0));
        printf("  plant target %d: OFF ttft %.1f s | legacy p99 %.0f mean %.0f ttft x%.2f | fixed p99 %.0f mean %.0f ttft x%.2f\n",
               target, off.ttft_s, legacy.p99, legacy.mean, legacy.ttft_s / off.ttft_s, fixed.p99, fixed.mean, fixed.ttft_s / off.ttft_s);
        // the plant reproduces the defect (observed: p99 +50..+80 ms over target at 670 and 900)
        CHECK(legacy.p99 > target + 30.0);
        // the fix holds the tail at the target
        CHECK(fixed.p99 <= target + 5.0);
        // at a TTFT cost no worse than ~5 % over the legacy controller
        CHECK(fixed.ttft_s <= 1.05 * legacy.ttft_s);
    }
}

static int replay_cell(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path); n_fail++; return 0; }
    auto c = make(0, 0, 0, 0.0);
    int dr, np, rows = 0, bad = 0; double t, logged;
    while (fscanf(f, "%d,%d,%lf,%lf", &dr, &np, &t, &logged) == 4) {
        // the server logs ms/tok rounded to 3 decimals, and t_iter to 0.1 ms
        c.observe(t, np, dr);
        if (std::fabs(c.ms_per_tok - logged) > 0.002 + 1e-3 * logged) bad++;
        // keep the replay on the logged state so one rounding error does not propagate
        if (logged > 0) c.ms_per_tok = logged;
        rows++;
    }
    fclose(f);
    printf("  %s: %d iterations, %d estimator mismatches\n", path, rows, bad);
    CHECK(rows > 0 && bad == 0);
    return rows;
}

int main(int argc, char ** argv) {
    test_legacy_identity();
    test_align();
    test_tail();
    test_replay();
    for (int i = 1; i < argc; i++) replay_cell(argv[i]);
    printf("test-server-prefill-budget: %d checks, %d failures\n", n_check, n_fail);
    return n_fail == 0 ? 0 : 1;
}
