#pragma once

// CAFE-3: online per-request controller for chain draft width (MTP, DFlash/DFlash2/DSpark).
//
// The controller only chooses a per-round CAP on the chain draft length (0 = target-only round). It never touches
// tokens, sampler state or the verify rule, so every emitted token is still produced by the unchanged target
// verification (greedy compare or rejection sampling) -- the emitted law is the static law. A static --draft-p-min
// keeps acting INSIDE the drafter as the floor: the cap bounds the width from above, p-min may stop earlier.
//
// Design (handoffs/active/speculative-decoding.md, CAFE-3; scripts/diagnostics/cafe3-adaptive/controller.h):
//   - per-position acceptance EMA s[i] = P(round commits draft position i | cap allowed position i). Rounds whose
//     cap did not reach i are censored for s[i]; a p-min early stop under a cap that allowed i counts as 0, so the
//     estimate is the marginal gain of allowing position i under the active p-min floor.
//   - per-cap round-time EMA c[w] (draft + verify + accept/rollback; measured draft-call to draft-call).
//   - score(w) = (1 + sum_{i<w} s[i]) / c[w]  (committed tokens per microsecond, higher = better).
//   - probe-before-veto: every eligible cap (incl. 0) is measured before a cost estimate can veto it.
//   - stale refresh: any cap not chosen for `gap` rounds is re-measured (oldest first).
//   - zero re-probe with exponential back-off (Cascade-style): when 0 wins, a positive cap is still re-probed after
//     1, 2, 4, ... gap rounds; a probe that beats 0 resets the back-off. Zero can never be absorbing.
//   - position-0 draft probability: not consumed directly; the static p-min already gates position 0 on it inside
//     the drafter, and s[0] measures the resulting position-0 commit rate.
//
// Default OFF: enabled only by LLAMA_CAFE3_ADAPTIVE=1 (exact). Optional LLAMA_CAFE3_ALPHA (round-time EMA, default 0.2;
// the acceptance EMA uses alpha/4) and
// LLAMA_CAFE3_GAP (stale/back-off ceiling in rounds, default 64).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

struct common_cafe3_ctrl {
    int      max_w = 0;
    double   alpha = 0.2;  // round-time EMA
    double   alpha_acc = 0.05; // acceptance EMA (Bernoulli samples: a slower EMA, sd ~ sqrt(a/(2-a)) * 0.5)
    uint32_t gap   = 64;

    std::vector<double>   cost; // per cap, us/round
    std::vector<bool>     seen; // per cap
    std::vector<uint64_t> last; // per cap, round index of last observation
    std::vector<double>   acc;  // per position, commit probability given the cap allowed it

    uint64_t round     = 0;
    uint64_t next_zero = 0;
    uint32_t backoff   = 1;
    uint32_t cursor    = 0;
    bool     zero_probe = false;

    // stats
    uint64_t n_probe = 0;

    common_cafe3_ctrl() = default;
    common_cafe3_ctrl(int max_width, double a = 0.2, uint32_t g = 64) { init(max_width, a, g); }

    void init(int max_width, double a = 0.2, uint32_t g = 64) {
        max_w = std::max(0, std::min(64, max_width));
        alpha = (std::isfinite(a) && a > 0.0 && a <= 1.0) ? a : 0.2;
        alpha_acc = alpha / 4.0;
        gap   = std::max<uint32_t>(g, (uint32_t) max_w + 1);
        cost.assign(max_w + 1, 0.0);
        seen.assign(max_w + 1, false);
        last.assign(max_w + 1, 0);
        acc.assign(max_w, 0.0);
        for (int i = 0; i < max_w; ++i) {
            acc[i] = std::pow(0.5, i + 1);
        }
        round = next_zero = 0;
        backoff = 1;
        cursor = 0;
        zero_probe = false;
        n_probe = 0;
    }

    double tokens(int w) const {
        double t = 1.0;   // the target's correction/bonus token
        double prev = 1.0;
        for (int i = 0; i < w; ++i) {
            prev = std::min(prev, acc[i]); // prefix survival is non-increasing
            t += prev;
        }
        return t;
    }

    double score(int w) const {
        return cost[w] > 0.0 ? tokens(w) / cost[w] : 0.0;
    }

    // cap: the legal width this round (context limit / --draft-max). Returns the selected cap in [0, min(cap, max_w)].
    int select(int cap) {
        const int limit = std::max(0, std::min(cap, max_w));
        int  sel   = -1;
        zero_probe = false;
        for (int w = 0; w <= limit; ++w) {
            if (!seen[w]) { sel = w; break; }
        }
        if (sel < 0) {
            for (int w = 0; w <= limit; ++w) {
                if (round - last[w] >= gap && (sel < 0 || last[w] < last[sel])) {
                    sel = w;
                }
            }
        }
        if (sel >= 0) {
            n_probe++;
            return sel;
        }
        sel = 0;
        for (int w = 1; w <= limit; ++w) {
            if (score(w) > score(sel)) {
                sel = w;
            }
        }
        if (sel == 0 && limit > 0 && round >= next_zero) {
            sel        = 1 + (int) (cursor++ % (uint32_t) limit);
            zero_probe = true;
            n_probe++;
        }
        return sel;
    }

    // sel: cap chosen by select(); n_drafted: tokens actually drafted (<= sel); n_accepted: draft tokens committed;
    // round_us: wall time of the whole round
    void observe(int sel, int n_drafted, int n_accepted, double round_us) {
        if (sel < 0 || sel > max_w || !std::isfinite(round_us) || round_us <= 0.0) {
            zero_probe = false;
            return;
        }
        n_drafted  = std::max(0, std::min(n_drafted, sel));
        n_accepted = std::max(0, std::min(n_accepted, n_drafted));
        cost[sel] = seen[sel] ? cost[sel] + alpha * (round_us - cost[sel]) : round_us;
        seen[sel] = true;
        last[sel] = ++round;
        for (int i = 0; i < sel; ++i) {
            acc[i] += alpha_acc * ((i < n_accepted ? 1.0 : 0.0) - acc[i]);
        }
        if (zero_probe) {
            backoff   = score(sel) > score(0) ? 1u : std::min(gap, backoff * 2u);
            next_zero = round + backoff;
        }
        zero_probe = false;
    }

    static bool env_enabled() {
        const char * v = std::getenv("LLAMA_CAFE3_ADAPTIVE");
        return v && std::string(v) == "1";
    }
    static double env_alpha() {
        const char * v = std::getenv("LLAMA_CAFE3_ALPHA");
        return v ? std::atof(v) : 0.2;
    }
    static uint32_t env_gap() {
        const char * v = std::getenv("LLAMA_CAFE3_GAP");
        const long g = v ? std::atol(v) : 64;
        return g > 0 ? (uint32_t) g : 64u;
    }
};
