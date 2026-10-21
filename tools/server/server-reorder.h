#pragma once

// Reorder planner for order-insensitive prompt segments (chunk-KV reuse, phase P1a).
//
// Port of the LMCache CacheBlend reorder planner (LMCache blend/reorder.py, Apache-2.0, pinned 032e23d):
// given a new prompt P and the token streams H of prompts that are already cached in slots, rebuild the
// longest possible BEGINNING of one cached prompt H out of P's own tokens, then serve the rest of P in its
// original order. The copied part is an exact-prefix hit, so it is valid for recurrent/hybrid state too:
// nothing is blended, shifted or approximated.
//
// EPYC differences from the LMCache original (all deliberate):
//   * Only tokens inside request-marked segments may move. Every other token is PINNED: it keeps its
//     position and is never part of a copied piece. The planner never runs on an unmarked prompt.
//   * Candidates are scored by the prefix length the reordered prompt ACTUALLY shares with H (re-measured
//     after pinned tokens are put back), not by the copied-token count. The caller therefore never accepts a
//     plan that does not beat the baseline common prefix of the unmodified prompt.
//   * Run length (lmin), glue and the time budget are parameters (LMCache hard-codes 16 / 3); the defaults are
//     the LMCache values, and the min-run and keep-tail rules are unchanged.
//
// Header only and free of llama.cpp dependencies except the token typedef, so it is unit-testable in isolation.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace server_reorder {

using token = int32_t;
using tokens = std::vector<token>;

struct config {
    int     lmin    = 16;      // shortest run of P copied as one piece; also the number of trailing tokens pinned last
    int     glue    = 3;       // longest short run (e.g. a separator) copied between two pieces
    int64_t budget_us = 50000; // planning time budget; a finished candidate still counts when it runs out
    int     chunk   = 1;       // candidates are first ranked by whole `chunk`s of achieved prefix, then by tokens
    int     min_gain = 16;     // the plan must extend the baseline common prefix by at least this many tokens
};

struct segment {
    int32_t offset   = 0; // token position in the prompt
    int32_t n_tokens = 0;
};

// A cached prompt the planner may copy from. `any_prefix` is true when the memory can drop an arbitrary tail
// (plain attention): every common prefix is then reusable. On hybrid/recurrent/SWA memory only a prefix that ends
// exactly at a saved checkpoint (`ckpts`, in tokens) or covers the whole cached prompt is reusable, so the planner
// scores the REUSABLE prefix, not the raw common prefix.
struct candidate {
    const tokens *       toks = nullptr;
    bool                 any_prefix = true;
    std::vector<int32_t> ckpts;

    int32_t reusable(int32_t n) const {
        if (any_prefix || n >= (int32_t) toks->size()) {
            return n;
        }
        int32_t best = 0;
        for (int32_t c : ckpts) {
            if (c <= n && c > best) {
                best = c;
            }
        }
        return best;
    }
};

struct result {
    std::vector<int32_t> perm;      // reordered[k] = P[perm[k]], a permutation of 0..|P|-1; pinned positions map to themselves
    std::vector<int32_t> junctions; // positions in the reordered prompt where a new copied piece starts (excluding 0)
    int32_t baseline_prefix = 0;    // reusable prefix of the unmodified P with the chosen candidate
    int32_t reordered_prefix = 0;   // reusable prefix of the reordered P with the chosen candidate
    int32_t common_prefix = 0;      // raw common prefix of the reordered P with the chosen candidate (>= reordered_prefix)
    int32_t n_pieces   = 0;
    int32_t candidate  = -1;        // index into the candidate list
    bool    timed_out  = false;
};

// Build the pinned mask from segments: a token is movable iff it lies inside a (validated) segment.
// Segments that are out of range, empty or overlapping an earlier one are rejected (returns false, mask untouched
// semantics are "everything pinned") so a malformed request degrades to the normal path.
inline bool pinned_from_segments(size_t n, const std::vector<segment> & segs, std::vector<bool> & pinned) {
    pinned.assign(n, true);
    for (const auto & s : segs) {
        if (s.offset < 0 || s.n_tokens <= 0 || (size_t) s.offset + (size_t) s.n_tokens > n) {
            pinned.assign(n, true);
            return false;
        }
        for (int32_t i = s.offset; i < s.offset + s.n_tokens; i++) {
            if (!pinned[i]) { // overlap
                pinned.assign(n, true);
                return false;
            }
            pinned[i] = false;
        }
    }
    return true;
}

namespace detail {

struct budget_exceeded {};

using clock = std::chrono::steady_clock;

struct piece {
    int32_t b, e; // [b, e) of P
};

class copier {
public:
    copier(const tokens & P, const config & cfg, clock::time_point deadline)
        : P(P), cfg(cfg), deadline(deadline) {
        const int L = cfg.lmin;
        const int64_t m = std::max<int64_t>((int64_t) P.size() - L + 1, 0);
        for (int64_t i = 0; i < m; i++) {
            index[hash(P.data() + i)].push_back((int32_t) i); // ascending by construction
        }
        check();
    }

    // Window hash: sum of x[j] * MUL^(lmin-1-j), mod 2^64 (as in LMCache). Collisions are harmless:
    // every run is verified token by token.
    uint64_t hash(const token * x) const {
        uint64_t h = 0;
        for (int j = 0; j < cfg.lmin; j++) {
            h = h * MUL + (uint64_t) (int64_t) x[j];
        }
        return h;
    }

    void check() const {
        if (clock::now() > deadline) {
            throw budget_exceeded{};
        }
    }

    // ascending positions of P whose lmin-token window has the hash of H[k : k + lmin]
    const std::vector<int32_t> & starts(const tokens & H, int64_t k) const {
        static const std::vector<int32_t> none;
        if (k + cfg.lmin > (int64_t) H.size()) {
            return none;
        }
        auto it = index.find(hash(H.data() + k));
        return it == index.end() ? none : it->second;
    }

    // (start, length) of the longest unused run of P equal to H[k:], at least lmin tokens; ties keep the leftmost
    bool longest(const tokens & H, int64_t k, const std::vector<char> & used, int32_t & bs, int32_t & bl) const {
        bool found = false;
        for (int32_t i : starts(H, k)) {
            if (used[i]) {
                continue;
            }
            const int64_t n = std::min<int64_t>((int64_t) P.size() - i, (int64_t) H.size() - k); // never grows with i
            if (found && n <= bl) {
                break; // later windows can only tie
            }
            check();
            int64_t m = 0;
            while (m < n && P[i + m] == H[k + m] && !used[i + m]) {
                m++;
            }
            if (m >= cfg.lmin && (!found || m > bl)) {
                found = true;
                bs = i;
                bl = (int32_t) m;
            }
        }
        return found;
    }

    // pieces of P that, laid end to end, equal the beginning of H. The first piece is P's first keep_prefix tokens.
    std::vector<piece> copy(const tokens & H, int32_t keep_prefix, const std::vector<bool> & pinned) const {
        const int64_t Ln = (int64_t) P.size();
        std::vector<char> used(P.size(), 0);
        for (int64_t i = 0; i < Ln; i++) {
            used[i] = pinned[i];
        }
        for (int32_t i = 0; i < keep_prefix; i++) {
            used[i] = 1;
        }
        for (int64_t i = std::max<int64_t>(Ln - cfg.lmin, 0); i < Ln; i++) {
            used[i] = 1; // the prompt's end stays last
        }
        std::vector<piece> pieces;
        if (keep_prefix) {
            pieces.push_back({0, keep_prefix});
        }
        int64_t k = keep_prefix;
        while (k < (int64_t) H.size()) {
            int32_t bs = 0, bl = 0;
            if (!longest(H, k, used, bs, bl)) {
                piece g, nx;
                if (!glue(H, k, used, pieces, g, nx)) {
                    break;
                }
                pieces.push_back(g);
                std::fill(used.begin() + g.b, used.begin() + g.e, 1);
                k += g.e - g.b;
                bs = nx.b;
                bl = nx.e - nx.b;
            }
            pieces.push_back({bs, bs + bl});
            std::fill(used.begin() + bs, used.begin() + bs + bl, 1);
            k += bl;
        }
        return pieces;
    }

private:
    // up to `glue` tokens of P equal to H[k:], then a piece for what follows. Glue next to copied text comes first,
    // then the leftmost; it is never taken from beyond the copy (P's final run, where the question is).
    bool glue(const tokens & H, int64_t k, std::vector<char> & used, const std::vector<piece> & pieces,
              piece & g, piece & nx) const {
        const int64_t L = (int64_t) P.size();
        for (int gl = 1; gl <= cfg.glue; gl++) {
            if (k + gl >= (int64_t) H.size()) {
                break;
            }
            if (starts(H, k + gl).empty()) {
                continue; // no piece can follow
            }
            std::vector<int32_t> cands;
            for (int64_t i = 0; i < L; i++) {
                if (P[i] != H[k] || used[i] || i + gl > L) {
                    continue;
                }
                bool ok = true;
                for (int t = 0; t < gl && ok; t++) {
                    ok = !used[i + t] && P[i + t] == H[k + t];
                }
                if (ok) {
                    cands.push_back((int32_t) i);
                }
            }
            auto next_to_copied = [&](int32_t i) {
                return (i > 0 && used[i - 1]) || (i + gl < L && used[i + gl]);
            };
            std::stable_sort(cands.begin(), cands.end(), [&](int32_t a, int32_t b) {
                const bool na = !next_to_copied(a), nb = !next_to_copied(b);
                return na != nb ? na < nb : a < b;
            });
            for (int32_t i : cands) {
                check();
                std::fill(used.begin() + i, used.begin() + i + gl, 1); // the next piece must not reuse the glue
                int32_t bs = 0, bl = 0;
                const bool f = longest(H, k + gl, used, bs, bl);
                std::fill(used.begin() + i, used.begin() + i + gl, 0);
                if (f) {
                    int32_t mx = bs + bl;
                    for (const auto & p : pieces) {
                        mx = std::max(mx, p.e);
                    }
                    if (i + gl <= mx) {
                        g  = {i, i + gl};
                        nx = {bs, bs + bl};
                        return true;
                    }
                }
            }
        }
        return false;
    }

    static constexpr uint64_t MUL = 0x9E3779B97F4A7C15ULL;

    const tokens & P;
    const config & cfg;
    clock::time_point deadline;
    std::unordered_map<uint64_t, std::vector<int32_t>> index;
};

inline int32_t common_prefix(const tokens & a, const tokens & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        i++;
    }
    return (int32_t) i;
}

} // namespace detail

// Plan a reordering of P so that it starts with the longest copy of one candidate's beginning.
// `candidates` are newest first (ties keep the newest). `pinned[i]` marks tokens that must not move
// (size must equal P.size()). Returns false when no reordering beats the baseline by cfg.min_gain tokens,
// i.e. the caller keeps P as is.
inline bool plan(const tokens & P, const std::vector<candidate> & candidates, const std::vector<bool> & pinned,
                 const config & cfg, result & out) {
    if (pinned.size() != P.size() || cfg.lmin < 1 || P.size() < (size_t) cfg.lmin * 2) {
        return false;
    }
    // the pinned head: P's leading run of pinned tokens stays in place and every copied candidate must share it
    int32_t keep_prefix = 0;
    while ((size_t) keep_prefix < P.size() && pinned[keep_prefix]) {
        keep_prefix++;
    }
    if ((size_t) keep_prefix == P.size()) {
        return false; // nothing is movable
    }

    const auto deadline = detail::clock::now() + std::chrono::microseconds(cfg.budget_us);

    bool have = false;
    struct best_t { int64_t chunks, tokens; int64_t neg_pieces; } best{};
    result best_res;

    try {
        detail::copier cp(P, cfg, deadline);
        std::vector<const candidate *> seen;
        for (size_t ci = 0; ci < candidates.size(); ci++) {
            const tokens & H = *candidates[ci].toks;
            if (H.size() < (size_t) keep_prefix || !std::equal(H.begin(), H.begin() + keep_prefix, P.begin())) {
                continue;
            }
            bool dup = false;
            for (const candidate * s : seen) {
                if (*s->toks == H && s->any_prefix == candidates[ci].any_prefix && s->ckpts == candidates[ci].ckpts) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            seen.push_back(&candidates[ci]);

            const auto pieces = cp.copy(H, keep_prefix, pinned);

            // permutation: pieces end to end, then (LMCache) H's next token if P has it beside a cut before the
            // copy's end, then everything else in original order
            std::vector<char> usedp(P.size(), 0);
            std::vector<int32_t> perm;
            perm.reserve(P.size());
            int64_t kc = 0;
            int32_t mx = 0;
            for (const auto & pc : pieces) {
                for (int32_t i = pc.b; i < pc.e; i++) {
                    perm.push_back(i);
                }
                std::fill(usedp.begin() + pc.b, usedp.begin() + pc.e, 1);
                kc += pc.e - pc.b;
                mx = std::max(mx, pc.e);
            }
            if (kc < (int64_t) H.size()) {
                for (int32_t i = 0; i < mx; i++) {
                    if (!usedp[i] && !pinned[i] && P[i] == H[kc] &&
                        ((i > 0 && usedp[i - 1]) || ((size_t) i + 1 < P.size() && usedp[i + 1]))) {
                        perm.push_back(i);
                        usedp[i] = 1;
                        break;
                    }
                }
            }
            for (size_t i = 0; i < P.size(); i++) {
                if (!usedp[i]) {
                    perm.push_back((int32_t) i);
                }
            }

            // put pinned tokens back: they keep their positions, movable slots are filled in ascending position
            // order with the movable part of `perm`
            std::vector<int32_t> final_perm(P.size());
            {
                size_t m = 0;
                std::vector<int32_t> mov;
                mov.reserve(P.size());
                for (int32_t p : perm) {
                    if (!pinned[p]) {
                        mov.push_back(p);
                    }
                }
                for (size_t pos = 0; pos < P.size(); pos++) {
                    final_perm[pos] = pinned[pos] ? (int32_t) pos : mov[m++];
                }
            }

            tokens reord(P.size());
            for (size_t i = 0; i < P.size(); i++) {
                reord[i] = P[final_perm[i]];
            }
            const int32_t common   = detail::common_prefix(reord, H);
            const int32_t achieved = candidates[ci].reusable(common);
            const int32_t baseline = candidates[ci].reusable(detail::common_prefix(P, H));

            const best_t cur{ achieved / std::max(cfg.chunk, 1), achieved, -(int64_t) pieces.size() };
            if (!have || std::make_tuple(cur.chunks, cur.tokens, cur.neg_pieces) >
                         std::make_tuple(best.chunks, best.tokens, best.neg_pieces)) {
                have = true;
                best = cur;
                best_res = result{};
                best_res.perm = std::move(final_perm);
                best_res.baseline_prefix  = baseline;
                best_res.reordered_prefix = achieved;
                best_res.common_prefix    = common;
                best_res.n_pieces  = (int32_t) pieces.size();
                best_res.candidate = (int32_t) ci;
                // junctions: where each copied piece starts in the reordered prompt (movable-stream cumulative
                // lengths mapped to positions), only those inside the achieved prefix are useful checkpoints
                int32_t acc = 0;
                for (size_t pi = 0; pi + 1 < pieces.size(); pi++) {
                    acc += pieces[pi].e - pieces[pi].b;
                    if (acc > 0 && acc < common) {
                        best_res.junctions.push_back(acc);
                    }
                }
            }
        }
    } catch (const detail::budget_exceeded &) {
        if (have) {
            best_res.timed_out = true;
        }
        // out of time: a finished candidate still counts
    }

    if (!have || best_res.reordered_prefix < best_res.baseline_prefix + cfg.min_gain) {
        return false;
    }
    bool identity = true;
    for (size_t i = 0; i < best_res.perm.size() && identity; i++) {
        identity = best_res.perm[i] == (int32_t) i;
    }
    if (identity) {
        return false;
    }
    out = std::move(best_res);
    return true;
}

} // namespace server_reorder
