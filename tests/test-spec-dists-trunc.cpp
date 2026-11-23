// SPEC-DISTS-TRUNC: a draft capped below its drafted length (n_predict tail, any n_max limit) must keep its per-token
// draft distributions aligned, so the verifier still takes the rejection path (dists.size() == draft.size(),
// server-context.cpp) instead of falling back to sample-compare. Checks (a) the cap keeps the pair aligned and the
// kept prefix bit-identical, (b) misaligned / short inputs are left alone, (c) the rejection step driven by the
// TRUNCATED dists still emits exactly the target distribution (chi-square), i.e. truncation changes nothing it emits.
#include "speculative.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static common_speculative_token_dist mk_dist(int depth) {
    common_speculative_token_dist d;
    float sum = 0.0f;
    for (int k = 0; k < 8; ++k) {
        d.ids.push_back(k);
        d.probs.push_back(1.0f + (float) ((k * 3 + depth) % 5));
        sum += d.probs.back();
    }
    for (float & p : d.probs) { p /= sum; }
    return d;
}

// the per-position rejection rule of common_sampler_sample_and_accept_n (dists overload): accept the draft token with
// prob min(1, p/q), else draw from norm(max(0, p - q))
static int reject_step(const std::vector<float> & p, const common_speculative_token_dist & q, int draft_tok, std::mt19937 & rng) {
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<float> qv(p.size(), 0.0f);
    for (size_t j = 0; j < q.ids.size(); ++j) { qv[q.ids[j]] += q.probs[j]; }
    if (qv[draft_tok] > 0.0f && U(rng) * qv[draft_tok] <= p[draft_tok]) { return draft_tok; }
    std::vector<float> r(p.size());
    for (size_t j = 0; j < p.size(); ++j) { r[j] = std::max(0.0f, p[j] - qv[j]); }
    std::discrete_distribution<int> R(r.begin(), r.end());
    return R(rng);
}

int main() {
    const int D = 7;
    llama_tokens draft;
    std::vector<common_speculative_token_dist> dists;
    for (int i = 0; i < D; ++i) { draft.push_back(100 + i); dists.push_back(mk_dist(i)); }
    const auto dists0 = dists;

    // (a) cap 4 (the captured tail rounds had n_draft 4 and 2 with D = 7)
    for (int cap : {4, 2, 1}) {
        llama_tokens d = draft; auto q = dists;
        common_speculative_cap_draft(d, &q, cap);
        CHECK((int) d.size() == cap);
        CHECK(q.size() == d.size()); // the rejection-path gate
        for (int i = 0; i < cap; ++i) { CHECK(d[i] == draft[i]); CHECK(q[i].ids == dists0[i].ids && q[i].probs == dists0[i].probs); }
    }
    // (b) no-op when not longer than cap; misaligned dists (greedy, T = 0: empty) untouched; null dists ok
    { llama_tokens d = draft; auto q = dists; common_speculative_cap_draft(d, &q, 7); CHECK(d.size() == 7 && q.size() == 7); }
    { llama_tokens d = draft; std::vector<common_speculative_token_dist> q; common_speculative_cap_draft(d, &q, 3); CHECK(d.size() == 3 && q.empty()); }
    { llama_tokens d = draft; common_speculative_cap_draft(d, nullptr, 3); CHECK(d.size() == 3); }

    // (c) chi-square: rejection with the truncated dists emits the target p at every kept position
    {
        llama_tokens d = draft; auto q = dists;
        common_speculative_cap_draft(d, &q, 4);
        std::mt19937 rng(4242);
        const int N = 200000;
        for (int i = 0; i < 4; ++i) {
            std::vector<float> p(8);
            float s = 0.0f;
            for (int k = 0; k < 8; ++k) { p[k] = 0.5f + (float) ((k * 5 + 2 * i) % 7); s += p[k]; }
            for (float & x : p) { x /= s; }
            std::discrete_distribution<int> Q(q[i].probs.begin(), q[i].probs.end());
            std::vector<int> cnt(8, 0);
            for (int n = 0; n < N; ++n) {
                const int tok = q[i].ids[Q(rng)]; // the drafter samples its token from its own dist
                cnt[reject_step(p, q[i], tok, rng)]++;
            }
            double chi2 = 0.0;
            for (int k = 0; k < 8; ++k) { const double e = (double) N * p[k]; chi2 += (cnt[k] - e) * (cnt[k] - e) / e; }
            // df = 7, p = 0.001 critical value 24.32
            printf("pos %d chi2 %.3f\n", i, chi2);
            CHECK(chi2 < 24.32);
        }
    }

    printf(n_fail ? "test-spec-dists-trunc: %d FAIL\n" : "test-spec-dists-trunc: OK\n", n_fail);
    return n_fail ? 1 : 0;
}
