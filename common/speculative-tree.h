#pragma once

// AK-DTREE-1: tree verification for block drafters (DFlash2), path-row form.
//
// Port of the `--draft-tree-nodes` idea from Infernix (github.com/Wallawalla47/Infernix, Apache-2.0,
// @397f8cd1cc64, docs/maintainer/tree-verification.md). Infernix verifies a real tree (tree-mask attention
// plus a parent-indexed recurrent record kernel). This is step 1 of the port and needs no new kernel: the
// tree is decomposed into <= 3 equal-length root-to-leaf paths, each path is verified as its own sequence
// in the same target decode (KV + recurrent state seq_cp'd from the slot's sequence), the path with the
// longest greedy match wins and its state is promoted back to the slot's sequence. Each path is a plain
// chain, so this is exact on hybrid (GDN) targets; the cost is the duplicated shared-prefix columns.

#include "llama.h"

#include <cstdint>
#include <functional>
#include <vector>

// Node budget convention (Infernix): `n_nodes` counts tree columns INCLUDING the anchor, so a chain of D draft
// tokens is D + 1 columns and a tree adds n_nodes - (D + 1) extra columns. An alternative path that branches off
// the main (greedy) path at draft position j shares positions [0, j) and adds D - j new columns.
//
// scores(j, pred) returns the K selector scores of draft position j given the predecessor candidate index pred
// at position j - 1 (pred is 0 at j = 0, exactly as the DFlash2 greedy reader starts).
//
// Returns candidate-index paths, each of length D. Path 0 is the greedy path (std::max_element tie rule, i.e.
// identical to the linear DFlash2 draft); paths 1.. branch off path 0 at some position with a rank-1 or rank-2
// candidate and continue greedily. Branches are chosen best-first by the log-probability that the greedy path
// fails exactly there and the branch is right (sum of path-0 log-softmax over [0, j) + the branch's own), while
// their column cost fits the budget. Fewer than max_paths paths are returned when the budget does not allow more;
// a budget <= D + 1 returns the chain alone.
using common_dtree_scores_fn = std::function<const float *(int32_t j, int32_t pred)>;

std::vector<std::vector<int32_t>> common_dtree_build_paths(
        int32_t D, int32_t K, const common_dtree_scores_fn & scores, int32_t n_nodes, int32_t max_paths);

// DTREE-TREE-QUALITY (LLAMA_DTREE_POLICY=greedy): probability-greedy best-first tree (SpecInfer/Sequoia-style).
// Starting from the anchor, the frontier node with the highest cumulative selector probability (product of the
// per-position softmax of the scores along its path) joins the tree until n_nodes - 1 draft nodes are used; any
// node may take children, there is no path cap and no depth below D is forced. Ties go to the lexicographically
// smaller index path. branch_follows_main (MTP): a node off the greedy chain has one child only, candidate 0 of the
// next position (the head never saw the branch, so the continuation is the main chain's suffix).
// Returns index paths: path 0 is the greedy chain's prefix inside the tree (length >= 1 when n_nodes >= 2, may be
// shorter than D), paths 1.. are the other leaves' root paths (any length <= D), in insertion order of the leaves.
// The tree is the prefix closure of the returned paths; it exactly matches tree_sim.py grow(r, W) (raw probs).
std::vector<std::vector<int32_t>> common_dtree_build_greedy(
        int32_t D, int32_t K, const common_dtree_scores_fn & scores, int32_t n_nodes, bool branch_follows_main);

// LLAMA_DTREE_POLICY: 0 = static paths (default, common_dtree_build_paths), 1 = greedy (common_dtree_build_greedy)
int32_t common_dtree_policy();

// A tree round appends the anchor and the MAIN draft to the slot's prompt tokens before the verify. When an
// alternative wins, the accept path rolls the prompt back by the WINNER's draft length (keep_first(n - n_draft)),
// so the main draft in the prompt must first be replaced by the winner: with the static paths every draft had the
// same length D and the mismatch was invisible, but a greedy tree's alternatives are shorter or longer than its
// main chain, and the prompt (hence pos_next) then drifted from the target memory by the length difference
// (r3 rows: "root token ... has pos 144, expected 143", "inconsistent sequence positions"). T is any token
// container with size() / keep_first(n) / insert(tokens) (server_tokens; a vector wrapper in the unit test).
template <typename T, typename D>
void common_dtree_swap_winner_draft(T & prompt_tokens, size_t n_main_draft, const D & winner) {
    prompt_tokens.keep_first(prompt_tokens.size() - n_main_draft);
    prompt_tokens.insert(winner);
}

// number of distinct tree columns (anchor included) of a path set returned by common_dtree_build_paths
int32_t common_dtree_n_columns(const std::vector<std::vector<int32_t>> & paths);

// How a drafted tree is verified (--draft-tree-mode). The drafters produce root-to-leaf paths in both modes;
// only the verify plumbing differs, so a mode swap needs no drafter change.
//   PATHS : step 1 (this file): each path is its own target sequence (fork / promote / drop), no new kernel.
//   BATCH : AK-DTREE-2 (separate branch): ONE sequence, the tree's unique columns in one batch with a parent-index
//           tensor and a tree attention mask (Infernix design). Not implemented here; the server refuses it.
enum common_dtree_mode {
    COMMON_DTREE_MODE_PATHS = 0,
    COMMON_DTREE_MODE_BATCH = 1,
};

// Flattened tree for the BATCH mode: unique columns in DFS order. Column 0 is the anchor (parent -1); every other
// column has parent < its own index. leaf_col[p] is the last column of path p (path 0 = the main draft).
struct common_dtree_flat {
    std::vector<llama_token> tokens;   // tokens[0] = anchor
    std::vector<int32_t>     parent;
    std::vector<int32_t>     depth;    // anchor depth 0
    std::vector<int32_t>     leaf_col;
};

common_dtree_flat common_dtree_flatten(llama_token anchor, const std::vector<std::vector<llama_token>> & paths);

// maximum number of paths (main + alternatives) verified per slot
constexpr int32_t COMMON_DTREE_MAX_PATHS = 3;

// greedy acceptance length of `draft` given the target logits rows `rows` (rows.size() == draft.size() + 1,
// rows[0] is the anchor's row): the number of draft tokens d such that argmax(rows[i]) == draft[i] for all i < d.
// Pure argmax over the raw logits; the caller's sampler remains the authority on what is accepted.
//
// DTREE-DEVICE-TREE-ACCEPT (LLAMA_DTREE_DEVICE_ACCEPT=1, default off): when `allow_device` is true (the caller has
// proven its sampler greedy: no grammar / reasoning budget, temp <= 0, ban-only logit bias) each row's pick is the
// device argmax (result_argmax, first maximum on ties == std::max_element), so only T int32 ids cross D2H and the
// lazy full-vocab logits download is never flushed by the accept walk. A row without a device pick falls back to
// the host scan for that row. With the knob off or allow_device false the function is the host path, unchanged.
int32_t common_dtree_greedy_accept_len(
        llama_context * ctx, const std::vector<int32_t> & rows, const std::vector<llama_token> & draft,
        bool allow_device = false);

// host greedy pick over one raw logits row: first maximum (lowest index on ties), exactly the host accept rule
llama_token common_dtree_row_argmax(const float * logits, int32_t n_vocab);

// accept walk over per-row picks (picks[i] = argmax of row i): number of leading i with picks[i] == draft[i]
int32_t common_dtree_accept_walk(const std::vector<llama_token> & picks, const std::vector<llama_token> & draft);

// SPEC-DTREE-DICE V2 (LLAMA_DTREE_SAMPLED=1, default off): sampled (T > 0) tree accept walk.
// paths[k] are the verified root-to-leaf paths (k = 0 is the main chain), rows[k][d] is the batch row whose logits
// give the target distribution after the anchor + paths[k][0..d) (rows[k].size() == paths[k].size() + 1). Paths
// that agree on their first d tokens share the same context at depth d, so any of their rows[k][d] serves.
// The walk (V2 of scripts/kernel_rnd/dice_replay/dice_replay.py): at depth d draw the TARGET's own sample
// `sample(row)` at the current node; if it equals the depth-d token of a still-live path, `accept(tok)` and
// descend, else stop (the sampled token is the bonus the caller's accept pass emits). Every emitted token is a
// target sample, so the emitted distribution equals the target's (lossless). Returns the accepted length;
// *best = the lowest live path index at the stop (0 = the main chain when it is still live).
int32_t common_dtree_sampled_walk(
        const std::vector<const std::vector<int32_t> *>     & rows,
        const std::vector<const std::vector<llama_token> *> & paths,
        const std::function<llama_token(int32_t row)>        & sample,
        const std::function<void(llama_token tok)>           & accept,
        int32_t * best);

// SPEC-DTREE-SAMPLED-V2: recursive rejection-sampling tree walk (multi-draft, replay V1; intake-2846#02).
// Same rows/paths contract as common_dtree_sampled_walk. At each node (depth d, live paths L):
//  - main-chain node (path 0 live, d < |path 0|) and `reject` set: the SERVED single-draft rejection step decides
//    the main child against the draft distribution q_d (reject(row, d, &y) returns true = main child accepted,
//    y = main token); on rejection y is the residual sample norm(max(p - q, 0)) the served path would emit.
//  - any other node (or reject unset): y = sample(row), the served sample-and-compare draw (p, or the current
//    residual implicitly: the sibling proposals are point masses, so "accept sibling x w.p. r(x), else renormalise
//    r without x" is the same law as "draw y ~ r, accept the sibling equal to y").
//  Then: if y is the depth-d token of a live path (main excluded after a main rejection), accept(y) and descend
//  into those paths; else y is the bonus. Every emitted token has exactly the served path's law at that node, so
//  the round is lossless; with no siblings (W = 1) the draws, tokens and stop are the served path's, bit for bit.
// Returns the accepted length; *bonus = the final token (not passed to accept); *best = lowest live path at the stop;
// *n_alt += 1 when the walk left the main chain through a sibling ("alternative accepted").
int32_t common_dtree_rrs_walk(
        const std::vector<const std::vector<int32_t> *>     & rows,
        const std::vector<const std::vector<llama_token> *> & paths,
        const std::function<llama_token(int32_t row)>        & sample,
        const std::function<bool(int32_t row, int32_t d, llama_token * y)> & reject,
        const std::function<void(llama_token tok)>           & accept,
        llama_token * bonus,
        int32_t * best,
        int64_t * n_alt);

// SPEC-DTREE-SAMPLED-V2: sibling paths around a SAMPLED main chain (T > 0). main[j] is the candidate index the
// drafter sampled at depth j, q[j][k] its draft probability of candidate k at depth j (given main[0..j)), scores
// the lattice (continuations are greedy from the sibling, as common_dtree_build_paths). Candidates are siblings
// (k != main[j], q > 0) ranked by log q(main[0..j)) + log q[j][k] (top-q per depth, best-first across depths), each
// path = main[0..j) + k + greedy continuation to D, taken while its D - j new columns fit n_nodes - (D + 1) and
// fewer than max_paths paths exist. Returns the alternatives only (path 0 = main is implied).
std::vector<std::vector<int32_t>> common_dtree_build_sampled(
        int32_t D, int32_t K, const common_dtree_scores_fn & scores, const std::vector<int32_t> & main,
        const std::vector<std::vector<float>> & q, int32_t n_nodes, int32_t max_paths);

// SPEC-DTREE-SAMPLED-V2: whether the tree accept may run the served REJECTION step on the main chain (the draft
// dists cover the whole main chain and the target can roll back a partial main chain). false = every node falls back
// to sample-and-compare, the downgrade the V1 build suffered; the server counts such rounds (tree_rrs_noq).
bool common_dtree_rrs_q_ok(size_t n_dists, size_t n_main, bool seq_rm_part, bool seq_rm_rs, int32_t n_rs_seq);

// SPEC-DTREE-SAMPLED-V2: verify columns a sibling path adds beyond the main chain (its length past the shared prefix)
int32_t common_dtree_new_cols(const std::vector<int32_t> & main, const std::vector<int32_t> & alt);

// DTREE-B at T > 0: emission of a NARROWER tree arm in an RRS round, by coupling. The narrower arm's tree is the main
// path plus the leading (best-first) alternatives whose cumulative new columns fit extra_budget; the RRS walk on it
// draws the same keyed samples as the full walk until the full walk takes a node outside it, where it emits that same y
// as its bonus. Returns the arm's accepted length = the longest prefix of acc (the full round's accepted tokens, bonus
// excluded) covered by one of its paths. Emission = return + 1.
int32_t common_dtree_arm_cover(const std::vector<int32_t> & main, const std::vector<std::vector<int32_t>> & alts,
                               const std::vector<int32_t> & acc, int32_t extra_budget);

// LLAMA_DTREE_SAMPLED_W (paths incl. main, default COMMON_DTREE_MAX_PATHS) for the T > 0 sibling builders
int32_t common_dtree_sampled_max_paths();

// Witness line printed once when LLAMA_DTREE_SAMPLED=1 is read: names the verifier actually compiled in (the V2
// recursive rejection tree accept, common_dtree_rrs_walk), so a run log proves which sampled verifier served it.
#define COMMON_DTREE_SAMPLED_ENABLED_MSG \
    "llama_dtree_sampled: ENABLED (LLAMA_DTREE_SAMPLED=1; T>0 tree verify, V2 recursive rejection (RRS) tree accept, siblings at T>0)"

// LLAMA_DTREE_SAMPLED=1 read once (lifts the T > 0 guard of the tree verify; the accept walk is the V2 walk above)
bool common_dtree_sampled_enabled();

// LLAMA_DTREE_DEVICE_ACCEPT=1 read once; witness counters (rows decided by the device pick / by the host scan)
bool    common_dtree_device_accept_enabled();
void    common_dtree_device_accept_counts(int64_t * n_dev_rows, int64_t * n_host_rows);

// fork: make seq `dst` an exact copy of seq `src` (attention KV + recurrent state, including a pending recurrent
// rollback of src) so that `dst` can be decoded from src's next position. Requires a unified KV cache.
void common_dtree_fork(llama_context * ctx, llama_seq_id src, llama_seq_id dst);

// promote: replace seq `dst` by seq `src` and drop `src` (the winning path becomes the slot's sequence).
// After this, `dst` owns its recurrent cell alone, so a partial rollback of `dst` is possible.
void common_dtree_promote(llama_context * ctx, llama_seq_id src, llama_seq_id dst);

// drop an alternative path sequence entirely
void common_dtree_drop(llama_context * ctx, llama_seq_id seq);

// --draft-tree-nodes auto (DTREE-B): Infernix's TreeWidthController (TV:L73-108) adapted to path rows.
// Arms: 0 = chain, 1 = D + 5 columns, 2 = D + 9 columns.
//  * Gain is measured on the generated text: a narrower tree is a prefix of a wider one (paths are added best
//    first), so a round of arm u also yields what every arm a <= u would have emitted (observe's emit_by_arm).
//    Per arm: decayed sum (decay 1/512) of emitted tokens per round; an arm's gain is trusted after 32 rounds.
//  * Cost is the wall time per round, per (batch bucket, context bucket, arm): the minimum of the first 3 rounds,
//    then a 1/8 EWMA whose sample is clamped to [t/2, 2t] (one spike moves it by at most 1/8 of t).
//  * Choice: the arm with the most expected tokens per second. Arms lacking gain data are reached by running the
//    widest arm (its rounds give all narrower gains); arms lacking timing in this bucket are timed 3 times,
//    widest first. The widest arm is re-timed every 32 rounds and any arm whose timing is 512 rounds old.
//  * Batches above 4 always run the chain.
struct common_dtree_auto {
    static constexpr int32_t N_ARMS      = 3;
    static constexpr int32_t N_BATCH_B   = 3;   // 1, 2, 3-4 active slots
    static constexpr int32_t N_CTX_B     = 4;   // <= 8K, <= 32K, <= 64K, > 64K
    static constexpr int32_t N_BUCKETS   = N_BATCH_B * N_CTX_B;
    static constexpr int32_t GAIN_MIN    = 32;
    static constexpr int32_t RETIME_WIDE = 32;
    static constexpr int32_t STALE       = 512;

    double  gsum[N_ARMS]  = { 0.0, 0.0, 0.0 };  // decayed sum of emitted tokens per round
    double  gw[N_ARMS]    = { 0.0, 0.0, 0.0 };  // decayed weight
    int32_t n_gain[N_ARMS] = { 0, 0, 0 };       // rounds that measured this arm's gain

    double  t_us[N_BUCKETS][N_ARMS]   = {};     // round wall time estimate
    int32_t n_t[N_BUCKETS][N_ARMS]    = {};     // timed rounds
    int64_t t_last[N_BUCKETS][N_ARMS] = {};     // round index of the last timing

    int64_t n_round = 0;

    // kept for the server's stats line: tokens/s of each arm in the last bucket chosen (0 = unknown)
    double  rate[N_ARMS]  = { 0.0, 0.0, 0.0 };
    int32_t n_obs[N_ARMS] = { 0, 0, 0 };

    static int32_t bucket(int32_t n_batch, int32_t n_ctx);
    // pick the arm of the next round (n_batch = active generating slots, n_ctx = this slot's context length)
    int32_t choose(int32_t n_batch = 1, int32_t n_ctx = 0);
    // record a finished round of arm `arm`: emit_by_arm[a] = tokens arm a would have emitted (a <= arm; < 0 =
    // unknown), t_us_round = the round's wall time
    void observe(int32_t arm, const int32_t * emit_by_arm, int64_t t_us_round, int32_t n_batch = 1, int32_t n_ctx = 0);
    // compatibility: only the used arm's emitted count is known
    void observe(int32_t arm, int32_t n_emit, int64_t t_us_round);
    double gain(int32_t arm) const { return gw[arm] > 0.0 ? gsum[arm] / gw[arm] : 0.0; }
    // tree columns for an arm given D draft tokens (0 for the chain)
    static int32_t arm_nodes(int32_t arm, int32_t D);
};
