#pragma once

// decode-aware prefill budget controller (--prefill-budget-decoding / --prefill-budget-target-ms /
// --prefill-budget-align / --prefill-budget-tail-k). pure state + arithmetic, no llama state, so it is
// unit-tested without a model (tests/test-server-prefill-budget.cpp).
//
// while any slot is generating, every server iteration carries the decode/verify rows plus a chunk of a
// neighbour's prompt. the adaptive budget sizes that chunk from a smoothed "iteration ms per prompt token"
// so an iteration takes about target_ms.
//
// KVU-16g-3 overshoot fix (2026-11-06). the legacy estimator (align = 0, tail_k = 0) is a MEAN controller
// on a linear cost model. the recorded KVU-16g-3 cells (Qwen3.8-27B Q8_0 + DFlash2, MI210, -ub 512) show
//   (1) the iteration cost is a staircase in the batch rows (decode rows + prompt tokens), with a step of
//       ~60 ms at every multiple of 32 rows (sub-step at 16): the linear model parks the chunk on a step
//       edge and the loop limit-cycles across it (t alternating 652 / 685 ms at 176 / 178 rows), which
//       triples the closed-loop jitter (sd 25-33 ms, lag-1 autocorrelation -0.33) over the open-loop
//       noise at a fixed chunk (sd 8 ms);
//   (2) a mean controller puts the median on the target, so p99 lands ~2.5-3 sd above it (+55..+80 ms).
// the fix: align the batch rows DOWN to a multiple of `align` (cost is flat inside a tile, so the
// aligned size is the cheapest point of its step and the limit cycle disappears), and aim at
// target - tail_k * RMS(prediction residual) so the tail, not the mean, is held near the target.
// align = 0 and tail_k = 0 reproduce the legacy controller exactly.

#include <algorithm>
#include <cmath>
#include <cstdint>

struct server_prefill_budget_ctl {
    static constexpr int32_t MIN_TOKENS  = 64;   // smallest adaptive budget; also the min chunk that updates the estimate
    static constexpr double  ALPHA       = 0.5;  // EWMA weight of the newest ms/tok sample (legacy value)
    static constexpr double  BETA        = 0.1;  // EWMA weight of the newest squared prediction residual
    static constexpr int32_t RESID_WARM  = 4;    // residual samples before the tail margin applies
    static constexpr double  TEFF_FLOOR  = 0.5;  // the tail margin never lowers the target below this fraction

    // configuration
    int32_t cap       = 0;   // --prefill-budget-decoding (0 = no cap)
    int32_t target_ms = 0;   // --prefill-budget-target-ms (0 = adaptive off)
    int32_t align     = 0;   // --prefill-budget-align: batch-row tile for the adaptive budget (0 = off)
    double  tail_k    = 0.0; // --prefill-budget-tail-k: margin in residual RMS units (0 = mean controller)

    // state
    double  ms_per_tok = 0.0; // smoothed iteration ms per prompt token
    double  ew_r2      = 0.0; // smoothed squared residual t - ms_per_tok * n_prompt (ms^2)
    int32_t n_resid    = 0;   // residual samples seen

    // the target the adaptive budget aims at this iteration (ms)
    double target_eff() const {
        if (target_ms <= 0) {
            return 0.0;
        }
        const double t = (double) target_ms;
        if (tail_k <= 0.0 || n_resid < RESID_WARM) {
            return t;
        }
        return std::max(TEFF_FLOOR * t, t - tail_k * std::sqrt(ew_r2));
    }

    // prompt-token budget for an iteration that already holds n_decode_rows rows, or 0 when it does not apply
    // (no generating slot, or a budget >= n_batch)
    int32_t get(int32_t n_batch, int32_t n_decode_rows, int32_t n_generating) const {
        if (n_generating == 0) {
            return 0;
        }

        int32_t budget = n_batch;

        if (cap > 0) {
            budget = std::min(budget, cap);
        }

        if (target_ms > 0 && ms_per_tok > 0.0) {
            const double n = target_eff() / ms_per_tok;
            int32_t adaptive = (int32_t) std::min<double>(n, (double) n_batch);
            if (align > 0 && adaptive < n_batch) {
                // whole tiles of batch rows: round the batch size down to a multiple of align, as long as
                // that still leaves MIN_TOKENS prompt tokens
                const int32_t rows    = n_decode_rows + adaptive;
                const int32_t aligned = (rows / align) * align - n_decode_rows;
                if (aligned >= MIN_TOKENS) {
                    adaptive = aligned;
                }
            }
            budget = std::min(budget, std::max(MIN_TOKENS, adaptive));
        }

        return budget < n_batch ? budget : 0;
    }

    // update the cost estimate from an iteration that took t_ms and carried n_prompt prompt tokens next to
    // n_decode_rows decode rows
    void observe(double t_ms, int32_t n_prompt, int32_t n_decode_rows) {
        if (n_prompt < MIN_TOKENS) {
            return;
        }
        if (ms_per_tok > 0.0 && n_decode_rows > 0) {
            const double r = t_ms - ms_per_tok * n_prompt;
            ew_r2 = n_resid == 0 ? r * r : (1.0 - BETA) * ew_r2 + BETA * r * r;
            n_resid++;
        }
        const double c = t_ms / n_prompt;
        ms_per_tok = ms_per_tok > 0.0 ? (1.0 - ALPHA) * ms_per_tok + ALPHA * c : c;
    }
};
