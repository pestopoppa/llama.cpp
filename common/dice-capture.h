#pragma once

// SPEC-DTREE-DICE step 2a: capture-only instrumentation for an OFFLINE replay of sampled-tree verification at served
// temperatures (DICE / Gumbel-max / recursive rejection arms). Default OFF: every entry point is a no-op unless
// LLAMA_DICE_CAPTURE=<path> is set when the process starts, and the server then appends one JSON line per verify round
// (and one per finished task) to <path>. With the knob unset the binary behaves exactly like its base.
//
// What is captured per round (schema "dice.v1", see the row's manifest for the field list):
//   drafter side  DFlash2: the full D x K lattice (candidate ids + predecessor-conditioned selector scores), the chain's
//                 candidate indices and, at T > 0, a forced superset tree (chain U probability-greedy best-first U
//                 DICE-style best-first with keyed Gumbel noise) that is verified in the same target batch while the
//                 round COMMITS through the unchanged chain accept path; MTP: top-10 (id, logit, p) per drafted
//                 position of the head's own sampler.
//   target side   at every verified row (anchor + chain, plus every tree node for DFlash2): the raw top-64 logits and
//                 the post-sampler candidate list (id, p) of the request's own sampler chain.
//   outcome       the accepted tokens of the round, the RNG keys (dist seed, speculative seed, absolute position).

#include "llama.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct common_sampler;

// true iff LLAMA_DICE_CAPTURE=<non-empty path> was set at first use and the file could be opened for append
bool common_dice_capture_enabled();

// appends one line (a '\n' is added) under a process-wide mutex; no-op when disabled
void common_dice_capture_line(const std::string & line);

// keyed Gumbel noise for the DICE-style tree: u = llama_rng_poskey_u01(seed, pos, COMMON_DICE_ROLE_GUMBEL + tok),
// g = -log(-log(u)) with u clamped to [2^-53, 1 - 2^-53]. pos is the sampler's absolute position (n_accepted) of the
// token being decided, i.e. n_acc0 + depth for a node at tree depth `depth` (0-based) of a round with n_acc0.
constexpr uint64_t COMMON_DICE_ROLE_GUMBEL = 0x100000000ULL;
double common_dice_gumbel(uint64_t seed, uint64_t pos, int32_t tok);

// one node of the forced superset tree (node 0 is the anchor, par = -1)
struct common_dice_node {
    int32_t     par   = -1;  // parent node index
    int32_t     depth = -1;  // 0-based draft depth (the anchor is -1)
    int32_t     k     = -1;  // lattice candidate index at that depth
    llama_token tok   = -1;
    int32_t     src   = 0;   // bit 1 chain, bit 2 probability-greedy, bit 4 DICE-style
    int32_t     pr    = 0;   // 1-based rank in the probability-greedy best-first order (0: not within the computed order)
    int32_t     dr    = 0;   // 1-based rank in the DICE-style best-first order (0: not within the computed order)
};

// scores(j, pred) -> K selector scores at draft depth j given the predecessor candidate index at j - 1 (pred 0 at j 0);
// tok(j, k) -> the candidate id at depth j, index k.
using common_dice_scores_fn = std::function<const float *(int32_t j, int32_t pred)>;
using common_dice_tok_fn    = std::function<llama_token(int32_t j, int32_t k)>;

// Builds the forced superset tree within n_cols columns INCLUDING the anchor: first the chain (chain_k = the chain's
// candidate indices), then the probability-greedy (cumulative log softmax(score / T)) and the DICE-style (cumulative
// [log q + g - max over siblings of (log q + g)], g keyed on (seed, n_acc0 + depth, tok)) best-first orders,
// interleaved one node each, until n_cols is full or both orders are exhausted. Parents always precede children.
std::vector<common_dice_node> common_dice_build_tree(
        int32_t D, int32_t K, const common_dice_scores_fn & scores, const common_dice_tok_fn & tok,
        const std::vector<int32_t> & chain_k, float temperature, uint64_t seed, uint64_t n_acc0, int32_t n_cols);

// root-to-leaf token paths of `nodes` that are not a prefix of the chain, in node order (for spec_tree_alts)
std::vector<std::vector<llama_token>> common_dice_tree_alts(const std::vector<common_dice_node> & nodes);

// JSON array of the target rows `rows` (batch indices into ctx's last decode; -1 = not verified, emitted as null):
// [[[id,logit] x top_n raw logits], [[id,p] post-sampler candidates]] per row. The post-sampler list comes from a CLONE
// of `smpl` (the live sampler is never touched); the served chain carries no history-dependent sampler (no penalties,
// no grammar), so the clone's state at the round start is the state every row is sampled with.
std::string common_dice_rows_json(common_sampler * smpl, llama_context * ctx, const std::vector<int32_t> & rows, int32_t top_n);
