#pragma once

// EAR ("expected acceptance rate") for llama-perplexity --kl-divergence.
//
// EAR(p, q) = sum_v min(p(v), q(v)) = 1 - TV(p, q), the per-position probability that a token drawn
// from q is accepted by a speculative-decoding verifier with target p (Leviathan et al.). Definition
// and the top-K variant follow SLQ (src/metrics/evaluation.py, _nll_kl_ear): p and q are renormalized
// over the top-K tokens under the base p, and EAR_topK = sum_{k in topK(p)} min(p~_k, q~_k).
//
// Two variants are reported because they are NOT interchangeable:
//   * full-vocab EAR is a strict 1 - TV and satisfies Pinsker, EAR >= 1 - sqrt(KLD/2), per position
//     and (by Jensen) for the means;
//   * top-K renormalized EAR is what SLQ's paper and model cards quote; it is not a strict TV bound.
//
// The base distribution comes from the --kl-divergence-base file, where each log-prob is stored as
// uint16 code c with log p = scale*c + min_log_prob (codes for tokens more than 16 nats below the
// mode are clamped to 0). This reconstruction is used verbatim for every vocab entry, so a
// recomputation from the base file reproduces these numbers exactly.

#include <algorithm>
#include <cmath>
#include <cstdint>

static constexpr int KLD_EAR_TOP_K = 10;

// One uint16 step of the base file's log-prob encoding (a 16-nat range over 65535 codes): the bound on the
// reconstruction error of log p_base, hence on how far EAR(base, base) may fall below 1.
static constexpr double KLD_EAR_BASE_QUANT_TOL = 16.0/65535.0;

struct kld_ear_values {
    double full = 0.0; // sum_v min(p_base, q) over the whole vocabulary
    double topk = 0.0; // SLQ top-K renormalized variant, K = min(top_k, n_vocab)
};

// logits:   the evaluated model's raw logits for one position (n_vocab entries)
// log_norm: max_logit + log(sum exp(logits - max_logit)), so log q(v) = logits[v] - log_norm
// base:     the base file's uint16 codes for the same position (n_vocab entries, after the 4-word header)
static inline kld_ear_values kld_ear_compute(int n_vocab, const float * logits, float log_norm,
        const uint16_t * base, float scale, float min_log_prob, int top_k = KLD_EAR_TOP_K) {
    kld_ear_values res;
    if (n_vocab <= 0) {
        return res;
    }
    const int k_max = std::max(1, std::min(std::min(top_k, n_vocab), KLD_EAR_TOP_K));

    // top-K indices under the base distribution, sorted by code descending; ties keep the lower index
    int      top_idx [KLD_EAR_TOP_K];
    uint16_t top_code[KLD_EAR_TOP_K];
    int      n_top = 0;

    double ear = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const float lp = scale*base[i] + min_log_prob;
        const float lq = logits[i] - log_norm;
        ear += expf(std::min(lp, lq));

        const uint16_t c = base[i];
        if (n_top < k_max || c > top_code[n_top - 1]) {
            int j = n_top < k_max ? n_top++ : k_max - 1;
            while (j > 0 && top_code[j - 1] < c) {
                top_code[j] = top_code[j - 1];
                top_idx [j] = top_idx [j - 1];
                --j;
            }
            top_code[j] = c;
            top_idx [j] = i;
        }
    }
    res.full = std::min(ear, 1.0);

    // renormalize both distributions over the base top-K (log-sum-exp for stability)
    float lp_max = -INFINITY;
    float lq_max = -INFINITY;
    for (int k = 0; k < n_top; ++k) {
        lp_max = std::max(lp_max, scale*top_code[k] + min_log_prob);
        lq_max = std::max(lq_max, logits[top_idx[k]]);
    }
    double zp = 0.0;
    double zq = 0.0;
    for (int k = 0; k < n_top; ++k) {
        zp += exp(double(scale*top_code[k] + min_log_prob) - lp_max);
        zq += exp(double(logits[top_idx[k]]) - lq_max);
    }
    double ear_top = 0.0;
    for (int k = 0; k < n_top; ++k) {
        const double pk = exp(double(scale*top_code[k] + min_log_prob) - lp_max)/zp;
        const double qk = exp(double(logits[top_idx[k]]) - lq_max)/zq;
        ear_top += std::min(pk, qk);
    }
    res.topk = std::min(ear_top, 1.0);
    return res;
}
