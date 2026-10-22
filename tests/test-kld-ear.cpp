// Unit test for tools/perplexity/kld-ear.h (EAR column of llama-perplexity --kl-divergence).
//
// Checks: EAR(base, base) == 1 within one base-file quantization step; 1 - sqrt(KLD/2) <= EAR <= 1
// (Pinsker) on perturbed distributions; the full-vocab and top-K values against a naive double
// reference; SLQ top-K selection (ties, n_vocab < K); a near-disjoint pair.

#include "kld-ear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { ++n_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// Same encoding as tools/perplexity/perplexity.cpp log_softmax(..., uint16_t * log_prob, ...),
// which writes the --kl-divergence-base file.
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

struct encoded_base {
    float scale;
    float min_log_prob;
    std::vector<uint16_t> codes;
};

static encoded_base encode_base(const std::vector<float> & logits) {
    const int n_vocab = (int) logits.size();
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    encoded_base e;
    e.min_log_prob = min_logit - max_logit - log_sum_exp;
    e.scale = (max_logit - min_logit)/65535.f;
    e.codes.assign(n_vocab, 0);
    if (e.scale) {
        const float inv_scale = 1/e.scale;
        for (int i = 0; i < n_vocab; ++i) {
            e.codes[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
        }
    }
    return e;
}

static float log_norm_of(const std::vector<float> & logits) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (float l : logits) {
        sum_exp += expf(l - max_logit);
    }
    return max_logit + (float) log(sum_exp);
}

struct reference {
    double full;
    double topk;
    double kld; // exact KL(p_base || q) over the full vocabulary
};

// Naive double-precision reference over the reconstructed base distribution.
static reference reference_ear(const std::vector<float> & logits, const encoded_base & e, int top_k) {
    const int n_vocab = (int) logits.size();
    const double ln = log_norm_of(logits);
    std::vector<double> p(n_vocab), q(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        p[i] = exp(double(e.scale*e.codes[i] + e.min_log_prob));
        q[i] = exp(double(logits[i]) - ln);
    }
    reference r = {0.0, 0.0, 0.0};
    for (int i = 0; i < n_vocab; ++i) {
        r.full += std::min(p[i], q[i]);
        r.kld  += p[i]*(log(p[i]) - log(q[i]));
    }
    std::vector<int> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return e.codes[a] > e.codes[b]; });
    const int k = std::min(top_k, n_vocab);
    double zp = 0.0, zq = 0.0;
    for (int j = 0; j < k; ++j) { zp += p[idx[j]]; zq += q[idx[j]]; }
    for (int j = 0; j < k; ++j) { r.topk += std::min(p[idx[j]]/zp, q[idx[j]]/zq); }
    return r;
}

static kld_ear_values run(const std::vector<float> & logits, const encoded_base & e, int top_k = KLD_EAR_TOP_K) {
    return kld_ear_compute((int) logits.size(), logits.data(), log_norm_of(logits), e.codes.data(), e.scale, e.min_log_prob, top_k);
}

static std::vector<float> random_logits(std::mt19937 & rng, int n_vocab, float spread) {
    std::normal_distribution<float> nd(0.0f, spread);
    std::vector<float> l(n_vocab);
    for (auto & x : l) x = nd(rng);
    return l;
}

int main() {
    std::mt19937 rng(20261022);
    const double tol = KLD_EAR_BASE_QUANT_TOL;

    // 1. EAR(base, base) == 1 within the base quantization step, flat and peaked distributions
    for (float spread : {0.5f, 2.0f, 6.0f}) {
        for (int n_vocab : {32, 4096, 151936}) {
            const auto logits = random_logits(rng, n_vocab, spread);
            const auto e = encode_base(logits);
            const auto v = run(logits, e);
            CHECK(v.full >= 1.0 - tol && v.full <= 1.0, "EAR(base,base) full=%.9f n_vocab=%d spread=%.1f", v.full, n_vocab, spread);
            CHECK(v.topk >= 1.0 - tol && v.topk <= 1.0, "EAR(base,base) top=%.9f n_vocab=%d spread=%.1f", v.topk, n_vocab, spread);
        }
    }

    // 2. Pinsker 1 - sqrt(KL/2) <= EAR <= 1, and agreement with the naive reference, on perturbed pairs
    for (float noise : {0.01f, 0.1f, 0.5f, 2.0f}) {
        for (int trial = 0; trial < 4; ++trial) {
            const int n_vocab = 2048;
            const auto base_logits = random_logits(rng, n_vocab, 3.0f);
            const auto e = encode_base(base_logits);
            std::normal_distribution<float> nd(0.0f, noise);
            auto q_logits = base_logits;
            for (auto & x : q_logits) x += nd(rng);
            const auto v = run(q_logits, e);
            const auto r = reference_ear(q_logits, e, KLD_EAR_TOP_K);
            CHECK(v.full <= 1.0, "EAR full > 1: %.9f", v.full);
            CHECK(v.full + tol >= 1.0 - sqrt(std::max(r.kld, 0.0)/2.0), "Pinsker violated: ear=%.9f kld=%.9f", v.full, r.kld);
            CHECK(fabs(v.full - r.full) < 1e-5, "full vs reference: %.9f vs %.9f (noise %.2f)", v.full, r.full, noise);
            CHECK(fabs(v.topk - r.topk) < 1e-6, "top-K vs reference: %.9f vs %.9f (noise %.2f)", v.topk, r.topk, noise);
        }
    }

    // 3. hand-checked 3-token case: p ~ (0.7, 0.2, 0.1), q ~ (0.2, 0.7, 0.1) -> EAR = 0.2 + 0.2 + 0.1 = 0.5
    {
        const std::vector<float> base_logits = { logf(0.7f), logf(0.2f), logf(0.1f) };
        const std::vector<float> q_logits    = { logf(0.2f), logf(0.7f), logf(0.1f) };
        const auto e = encode_base(base_logits);
        const auto v = run(q_logits, e);
        CHECK(fabs(v.full - 0.5) < 1e-3, "3-token full EAR %.6f != 0.5", v.full);
        CHECK(fabs(v.topk - 0.5) < 1e-3, "3-token top-K EAR %.6f != 0.5 (n_vocab < K uses every token)", v.topk);
    }

    // 4. top-K is renormalized over the base top-K only: q's mass outside it does not count
    {
        const int n_vocab = 64;
        std::vector<float> base_logits(n_vocab, -10.0f);
        for (int i = 0; i < KLD_EAR_TOP_K; ++i) base_logits[i] = 0.0f; // uniform top-10 under p
        std::vector<float> q_logits = base_logits;
        q_logits[n_vocab - 1] = 8.0f;                                 // q puts most mass on a non-top-10 token
        const auto e = encode_base(base_logits);
        const auto v = run(q_logits, e);
        const auto r = reference_ear(q_logits, e, KLD_EAR_TOP_K);
        CHECK(v.topk > 0.999, "top-K should ignore mass outside the base top-K: %.6f", v.topk);
        CHECK(v.full < 0.01, "full EAR must see the mass q moved away: %.6f", v.full);
        CHECK(fabs(v.full - r.full) < 1e-5 && fabs(v.topk - r.topk) < 1e-6, "case 4 vs reference");
    }

    // 5. ties in the base codes: lower index wins, matching a stable sort
    {
        const int n_vocab = 40;
        std::vector<float> base_logits(n_vocab, 0.0f);                // all tied
        std::vector<float> q_logits(n_vocab);
        for (int i = 0; i < n_vocab; ++i) q_logits[i] = 0.05f*i;
        const auto e = encode_base(base_logits);
        const auto v = run(q_logits, e);
        const auto r = reference_ear(q_logits, e, KLD_EAR_TOP_K);
        CHECK(fabs(v.topk - r.topk) < 1e-6, "tie handling: %.9f vs %.9f", v.topk, r.topk);
    }

    // 6. near-disjoint pair: EAR ~ 0, Pinsker floor goes negative and is trivially met
    {
        const int n_vocab = 1000;
        std::vector<float> base_logits(n_vocab, -20.0f);
        std::vector<float> q_logits(n_vocab, -20.0f);
        base_logits[3] = 10.0f;
        q_logits[700]  = 10.0f;
        const auto e = encode_base(base_logits);
        const auto v = run(q_logits, e);
        const auto r = reference_ear(q_logits, e, KLD_EAR_TOP_K);
        CHECK(v.full < 1e-3, "disjoint full EAR %.6f", v.full);
        CHECK(fabs(v.full - r.full) < 1e-6, "disjoint vs reference");
    }

    // 7. top_k argument smaller than the default is honoured; K = 1 is a same-top indicator after renorm
    {
        const auto logits = random_logits(rng, 512, 2.0f);
        const auto e = encode_base(logits);
        auto q = logits;
        for (auto & x : q) x += 0.3f;                                 // shift: same distribution
        const auto v = run(q, e, 1);
        CHECK(fabs(v.topk - 1.0) < 1e-12, "K=1 renormalized EAR must be 1: %.12f", v.topk);
    }

    if (n_fail) {
        fprintf(stderr, "test-kld-ear: %d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-kld-ear: all checks passed\n");
    return 0;
}
