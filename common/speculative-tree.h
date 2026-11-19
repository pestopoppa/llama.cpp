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
int32_t common_dtree_greedy_accept_len(
        llama_context * ctx, const std::vector<int32_t> & rows, const std::vector<llama_token> & draft);

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
