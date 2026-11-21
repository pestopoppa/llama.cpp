#pragma once

// AK-H-MTP-DEGEN-1: per-slot suspend/recover guard for speculative drafting.
//
// Default OFF. Knob: LLAMA_SPEC_DEGEN_SUSPEND=1 (optional LLAMA_SPEC_DEGEN_THRESH, default 0.2;
// LLAMA_SPEC_DEGEN_TOKENS, default 512; LLAMA_SPEC_DEGEN_PROBE, default 64).
//
// When a slot's draft acceptance over a window of >= TOKENS emitted tokens stays below THRESH, drafting is suspended
// (the slot decodes target-only) and re-probed with Raptor's non-absorbing recovery rule: every PROBE target-only
// tokens a probe round runs with depth cap 2; a shallow rejection (accepted < drafted) keeps the slot suspended and
// the cap at 2; every fully accepted round raises the cap by 1; when the cap reaches the configured maximum the slot
// returns to normal drafting with a fresh window. Verification is untouched, so output tokens cannot change on a
// deterministic model; only the amount of drafting (and rejected verify rows) changes.

#include <algorithm>
#include <cstdint>
#include <cstdlib>

struct common_spec_degen {
    enum state_t { ACTIVE = 0, SUSPENDED = 1, RECOVERING = 2 };

    // configuration (copied from the environment once per process by common_spec_degen_cfg())
    struct cfg_t {
        bool    enabled = false;
        float   thresh  = 0.2f;
        int32_t tokens  = 512;
        int32_t probe   = 64;
    };

    static cfg_t cfg_from_env() {
        cfg_t c;
        const char * e = getenv("LLAMA_SPEC_DEGEN_SUSPEND");
        c.enabled = e != nullptr && e[0] != 0 && e[0] != '0';
        if (const char * t = getenv("LLAMA_SPEC_DEGEN_THRESH"); t && *t) { c.thresh = (float) atof(t); }
        if (const char * t = getenv("LLAMA_SPEC_DEGEN_TOKENS"); t && *t) { c.tokens = std::max(1, atoi(t)); }
        if (const char * t = getenv("LLAMA_SPEC_DEGEN_PROBE");  t && *t) { c.probe  = std::max(1, atoi(t)); }
        return c;
    }

    static const cfg_t & cfg() {
        static const cfg_t c = cfg_from_env();
        return c;
    }

    cfg_t   c   = cfg();
    state_t st  = ACTIVE;
    int32_t win_emitted  = 0;   // tokens emitted by draft rounds in the current window
    int32_t win_drafted  = 0;
    int32_t win_accepted = 0;
    int32_t countdown    = 0;   // target-only tokens left before the next probe (SUSPENDED)
    int32_t probe_cap    = 2;   // depth cap while probing / recovering
    int32_t last_n_decoded = 0;
    int32_t n_suspends   = 0;   // statistics
    int32_t n_resumes    = 0;

    void reset() {
        const cfg_t keep = c;
        *this = common_spec_degen();
        c = keep;
    }

    // depth limit for the next round; n_max when the guard is off or ACTIVE
    int32_t cap(int32_t n_max) const {
        if (!c.enabled || st == ACTIVE) {
            return n_max;
        }
        if (st == SUSPENDED && countdown > 0) {
            return 0;
        }
        return std::min(n_max, probe_cap);
    }

    // called once per scheduling pass with the slot's decoded-token count; counts target-only tokens down
    void advance(int32_t n_decoded) {
        const int32_t delta = std::max(0, n_decoded - last_n_decoded);
        last_n_decoded = n_decoded;
        if (c.enabled && st == SUSPENDED && countdown > 0) {
            countdown = std::max(0, countdown - delta);
        }
    }

    // called after each drafted round was verified: drafted draft tokens, accepted of them, emitted = accepted + 1
    // returns +1 on a transition into SUSPENDED, -1 on a return to ACTIVE, 0 otherwise
    int32_t on_round(int32_t drafted, int32_t accepted, int32_t n_max) {
        if (!c.enabled || drafted <= 0) {
            return 0;
        }
        if (st == ACTIVE) {
            win_drafted  += drafted;
            win_accepted += accepted;
            win_emitted  += accepted + 1;
            if (win_emitted >= c.tokens) {
                const float acc = (float) win_accepted / (float) std::max(1, win_drafted);
                win_emitted = win_drafted = win_accepted = 0;
                if (acc < c.thresh) {
                    st = SUSPENDED;
                    countdown = c.probe;
                    probe_cap = 2;
                    n_suspends++;
                    return +1;
                }
            }
            return 0;
        }
        // SUSPENDED (probe round) or RECOVERING
        if (accepted < drafted) {            // shallow rejection: back to suspended, cap 2
            st = SUSPENDED;
            countdown = c.probe;
            probe_cap = 2;
            return 0;
        }
        probe_cap = std::min(n_max, probe_cap + 1);   // fully accepted round: +1
        st = RECOVERING;
        if (probe_cap >= n_max) {
            st = ACTIVE;
            n_resumes++;
            win_emitted = win_drafted = win_accepted = 0;
            return -1;
        }
        return 0;
    }
};
