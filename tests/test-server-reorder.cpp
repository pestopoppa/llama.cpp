// Unit tests for the llama-server reorder planner (tools/server/server-reorder.h), chunk-KV reuse phase P1a.
#undef NDEBUG
#include "server-reorder.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>

using namespace server_reorder;

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

// a run of n distinct tokens starting at base
static tokens doc(int base, int n) {
    tokens t(n);
    for (int i = 0; i < n; i++) t[i] = base + i;
    return t;
}
static tokens cat(std::initializer_list<tokens> parts) {
    tokens r;
    for (const auto & p : parts) r.insert(r.end(), p.begin(), p.end());
    return r;
}
static tokens reordered_by(const tokens & P, const std::vector<int32_t> & perm) {
    tokens r(P.size());
    for (size_t i = 0; i < P.size(); i++) r[i] = P[perm[i]];
    return r;
}
static bool is_perm(const std::vector<int32_t> & perm, size_t n) {
    if (perm.size() != n) return false;
    std::set<int32_t> s(perm.begin(), perm.end());
    return s.size() == n && *s.begin() == 0 && *s.rbegin() == (int32_t) n - 1;
}
static std::vector<bool> region(size_t n, int b, int e) { // movable = [b, e)
    std::vector<bool> pin(n, true);
    for (int i = b; i < e; i++) pin[i] = false;
    return pin;
}
static bool run(const tokens & P, const std::vector<const tokens *> & ct, const std::vector<bool> & pin, const config & cfg, result & r) {
    std::vector<candidate> c;
    for (const tokens * t : ct) c.push_back({t, true, {}});
    return plan(P, c, pin, cfg, r);
}

// the headline case: cached prompt has docs A B C, new prompt carries them as C A B -> exact prefix hit
static void test_reorder_docs() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40), C = doc(300, 40);
    const tokens H = cat({sys, A, B, C, q});
    const tokens P = cat({sys, C, A, B, q});
    const auto pin = region(P.size(), 20, 20 + 120);
    config cfg;
    result r;
    CHECK(run(P, {&H}, pin, cfg, r));
    CHECK(is_perm(r.perm, P.size()));
    const tokens out = reordered_by(P, r.perm);
    CHECK(out == H);                          // docs A B C restored, tail untouched
    CHECK(r.baseline_prefix == 20);
    CHECK(r.reordered_prefix == (int32_t) H.size());
    CHECK(r.candidate == 0);
    CHECK(r.n_pieces == 3);                   // head, then A+B (already contiguous in P: one run), then C; junctions = piece starts
    CHECK((r.junctions == std::vector<int32_t>{20, 100}));
    // non-movable tokens kept their positions
    for (int i = 0; i < 20; i++) CHECK(r.perm[i] == i);
    for (size_t i = 140; i < P.size(); i++) CHECK(r.perm[i] == (int32_t) i);
}

// the new prompt has docs the cache does not know: the copy stops at the first missing doc, the rest keeps its order
static void test_partial_set() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40), C = doc(300, 40), D = doc(400, 40);
    const tokens H = cat({sys, A, B, C, q});
    const tokens P = cat({sys, D, B, A, q});
    const auto pin = region(P.size(), 20, 140);
    config cfg;
    result r;
    CHECK(run(P, {&H}, pin, cfg, r));
    const tokens out = reordered_by(P, r.perm);
    CHECK(r.reordered_prefix == 20 + 40 + 40);          // sys + A + B, then C is missing
    CHECK(tokens(out.begin(), out.begin() + 100) == tokens(H.begin(), H.begin() + 100));
    CHECK(is_perm(r.perm, P.size()));
    // leftovers (D) follow in original order, before the pinned tail
    CHECK(tokens(out.begin() + 100, out.begin() + 140) == D);
}

// min-run rule: a doc shorter than lmin cannot be copied, so the copy stops before it
static void test_min_run() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), S = doc(200, 10), C = doc(300, 40);   // S is shorter than lmin=16
    const tokens H = cat({sys, A, S, C, q});
    const tokens P = cat({sys, C, S, A, q});
    const auto pin = region(P.size(), 20, 110);
    config cfg;                                      // lmin = 16
    result r;
    CHECK(run(P, {&H}, pin, cfg, r));
    CHECK(r.reordered_prefix == 20 + 40 + 0 || r.reordered_prefix >= 60);
    const tokens out = reordered_by(P, r.perm);
    // S (10 tokens) is a single run whose neighbours differ in P: it is only reachable as glue (<= 3 tokens) -> not copied
    CHECK(r.reordered_prefix == 60);
    CHECK(tokens(out.begin(), out.begin() + 60) == tokens(H.begin(), H.begin() + 60));
    // with lmin lowered to 8 the same prompt reorders fully
    cfg.lmin = 8;
    result r2;
    CHECK(run(P, {&H}, pin, cfg, r2));
    CHECK(r2.reordered_prefix == (int32_t) H.size());
    CHECK(reordered_by(P, r2.perm) == H);
}

// glue rule: a short separator (<= glue tokens) that sits elsewhere in P is pulled between two pieces
static void test_glue() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40), sep = {9001, 9002};
    const tokens H = cat({sys, A, sep, B, q});
    const tokens P = cat({sys, B, sep, A, q});       // sep is next to B here, A needs it after itself
    const auto pin = region(P.size(), 20, 20 + 40 + 2 + 40);
    config cfg;
    result r;
    CHECK(run(P, {&H}, pin, cfg, r));
    CHECK(reordered_by(P, r.perm) == H);

    // the separator sits before A in P, so no run of P reads "sep B": it is pulled in as glue between A and B
    const tokens P2 = cat({sys, sep, A, B, q});
    const auto pin2 = region(P2.size(), 20, 20 + 82);
    result r2;
    CHECK(run(P2, {&H}, pin2, cfg, r2));
    CHECK(reordered_by(P2, r2.perm) == H);
    CHECK(r2.n_pieces == 4);                  // head, A, glue, B
    CHECK(is_perm(r2.perm, P2.size()));
}

// keep-tail rule: the last lmin tokens never move, even when they match the cached prompt
static void test_keep_tail() {
    const tokens sys = doc(1000, 20);
    const tokens A = doc(100, 40), B = doc(200, 40);
    // the cached prompt continues with B's last 16 tokens right after A; in P those tokens end the prompt
    const tokens H = cat({sys, A, B});
    const tokens P = cat({sys, B, A});               // the whole body is movable, but the last 16 (tail of A) are pinned
    const auto pin = region(P.size(), 20, 100);
    config cfg;
    result r;
    const bool ok = run(P, {&H}, pin, cfg, r);
    // the last 16 positions must be untouched whatever the plan does
    if (ok) {
        for (size_t i = P.size() - 16; i < P.size(); i++) CHECK(r.perm[i] == (int32_t) i);
    }
    // and the copy can never contain P's last 16 tokens
    CHECK(!ok || r.reordered_prefix < (int32_t) H.size());
    CHECK(!ok || is_perm(r.perm, P.size()));
}

// pinned tokens (outside segments) never move; a prompt with no marked segment is never reordered
static void test_pinned_and_unmarked() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40);
    const tokens H = cat({sys, A, B, q});
    const tokens P = cat({sys, B, A, q});
    config cfg;
    result r;
    CHECK(!run(P, {&H}, std::vector<bool>(P.size(), true), cfg, r));          // everything pinned
    std::vector<bool> pin;
    CHECK(pinned_from_segments(P.size(), {{20, 80}}, pin));
    CHECK(run(P, {&H}, pin, cfg, r));
    CHECK(reordered_by(P, r.perm) == H);

    // a movable window that excludes B: B is pinned in place, so nothing can be moved in front of it
    result r2;
    const auto pinB = region(P.size(), 60, 100);  // only A movable
    CHECK(!run(P, {&H}, pinB, cfg, r2));
    // invalid segment lists are rejected (and degrade to "all pinned")
    CHECK(!pinned_from_segments(P.size(), {{20, 80}, {50, 10}}, pin));        // overlap
    CHECK(!pinned_from_segments(P.size(), {{100, 100}}, pin));                // out of range
    CHECK(!pinned_from_segments(P.size(), {{20, 0}}, pin));                   // empty
    CHECK(!pinned_from_segments(P.size(), {{-1, 5}}, pin));                   // negative
    for (bool b : pin) CHECK(b);
}

// no gain: an already-exact prefix, an identical order, or a candidate with a different head is left alone
static void test_no_gain() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40);
    config cfg;
    result r;
    const tokens H = cat({sys, A, B, q});
    const auto pin = region(H.size(), 20, 100);
    CHECK(!run(H, {&H}, pin, cfg, r));                                        // identical
    const tokens other = cat({doc(7000, 20), A, B, q});                       // different pinned head
    const tokens P = cat({sys, B, A, q});
    CHECK(!run(P, {&other}, pin, cfg, r));
    CHECK(!run(P, {}, pin, cfg, r));                                          // no candidates
    cfg.min_gain = 1000;
    const tokens H2 = cat({sys, A, B, q});
    CHECK(!run(P, {&H2}, pin, cfg, r));                                       // gain below the floor
}

// several candidates: the one with the longest achieved prefix wins; ties keep the newest
static void test_candidate_choice() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40), C = doc(300, 40);
    const tokens H1 = cat({sys, A, q});
    const tokens H2 = cat({sys, A, B, C, q});
    const tokens H3 = cat({sys, A, B, C, q});                                 // duplicate of H2 (older)
    const tokens P = cat({sys, C, B, A, q});
    const auto pin = region(P.size(), 20, 140);
    config cfg;
    result r;
    CHECK(run(P, {&H1, &H2, &H3}, pin, cfg, r));
    CHECK(r.candidate == 1);
    CHECK(reordered_by(P, r.perm) == H2);
}

// hybrid memory: only a prefix ending at a checkpoint (or the whole cached prompt) is reusable
static void test_hybrid_checkpoints() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24), q2 = doc(6000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40), C = doc(300, 40);
    const tokens H = cat({sys, A, B, C, q});
    const tokens P = cat({sys, C, B, A, q2});            // different question: the whole cached prompt can never match
    const auto pin = region(P.size(), 20, 140);
    config cfg;
    result r;
    // no checkpoints: nothing short of the whole prompt is reusable
    CHECK(!plan(P, {{&H, false, {}}}, pin, cfg, r));
    // checkpoints at the A|B (60) and B|C (100) junctions: the best reusable prefix is 100
    CHECK(plan(P, {{&H, false, {60, 100}}}, pin, cfg, r));
    CHECK(r.common_prefix == 140);
    CHECK(r.reordered_prefix == 100);
    CHECK(r.baseline_prefix == 0);
    // only the pinned-head checkpoint (20): already reusable without reordering by 20, the plan gains 120-20 -> still a win only via 140
    CHECK(!plan(P, {{&H, false, {20}}}, pin, cfg, r));   // reusable stays 20 (gain 0)
    // a candidate with the junction checkpoints beats an otherwise identical one without (same tokens, kept distinct)
    CHECK(plan(P, {{&H, false, {}}, {&H, false, {60, 100, 140}}}, pin, cfg, r));
    CHECK(r.candidate == 1);
    CHECK(r.reordered_prefix == 140);
    // attention-only memory: every common prefix is reusable
    CHECK(plan(P, {{&H, true, {}}}, pin, cfg, r));
    CHECK(r.reordered_prefix == 140);
}

// time budget: a zero budget aborts planning and the caller keeps P
static void test_budget() {
    const tokens sys = doc(1000, 20), q = doc(5000, 24);
    const tokens A = doc(100, 40), B = doc(200, 40);
    const tokens H = cat({sys, A, B, q}), P = cat({sys, B, A, q});
    config cfg;
    cfg.budget_us = -1000000;                                                 // deadline already in the past
    result r;
    CHECK(!run(P, {&H}, region(P.size(), 20, 100), cfg, r));
}

// fuzz: invariants hold on random segment layouts
static void test_fuzz() {
    std::mt19937 rng(12345);
    for (int it = 0; it < 400; it++) {
        const int nd = 2 + rng() % 5;
        std::vector<tokens> docs;
        for (int d = 0; d < nd; d++) docs.push_back(doc(100 * (d + 1), 8 + rng() % 40));
        const tokens sys = doc(9000, rng() % 10), q = doc(8000, 16 + rng() % 8);
        std::vector<int> o1(nd), o2(nd);
        std::iota(o1.begin(), o1.end(), 0); o2 = o1;
        std::shuffle(o1.begin(), o1.end(), rng);
        std::shuffle(o2.begin(), o2.end(), rng);
        tokens H = sys, P = sys;
        for (int i : o1) H.insert(H.end(), docs[i].begin(), docs[i].end());
        for (int i : o2) P.insert(P.end(), docs[i].begin(), docs[i].end());
        H.insert(H.end(), q.begin(), q.end());
        P.insert(P.end(), q.begin(), q.end());
        const auto pin = region(P.size(), (int) sys.size(), (int) (P.size() - q.size()));
        config cfg;
        result r;
        if (run(P, {&H}, pin, cfg, r)) {
            CHECK(is_perm(r.perm, P.size()));
            const tokens out = reordered_by(P, r.perm);
            for (size_t i = 0; i < P.size(); i++) if (pin[i]) CHECK(r.perm[i] == (int32_t) i);
            CHECK(r.reordered_prefix > r.baseline_prefix);
            int32_t lcp = 0;
            while ((size_t) lcp < out.size() && out[lcp] == H[lcp]) lcp++;
            CHECK(lcp == r.reordered_prefix);
            // multiset preserved
            tokens a = P, b = out;
            std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
            CHECK(a == b);
            for (int32_t j : r.junctions) CHECK(j > 0 && j < r.reordered_prefix);
        }
    }
}

int main() {
    test_reorder_docs();
    test_partial_set();
    test_min_run();
    test_glue();
    test_keep_tail();
    test_pinned_and_unmarked();
    test_no_gain();
    test_candidate_choice();
    test_hybrid_checkpoints();
    test_budget();
    test_fuzz();
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-server-reorder: OK\n");
    return 0;
}
