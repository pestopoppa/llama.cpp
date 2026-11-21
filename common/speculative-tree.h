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

// --draft-tree-nodes auto: per-slot online chooser between the chain and tree widths.
// Arms: 0 = chain, 1 = D + 5 columns, 2 = D + 9 columns (Infernix's K+5 / K+9 auto widths).
// The rate of an arm is an EMA (alpha 1/8) of emitted tokens per second of round wall time. Each arm is timed
// 3 times first (widest first); afterwards the best arm runs, and every other arm is re-timed once per 32 rounds.
struct common_dtree_auto {
    static constexpr int32_t N_ARMS = 3;

    double  rate[N_ARMS]  = { 0.0, 0.0, 0.0 };
    int32_t n_obs[N_ARMS] = { 0, 0, 0 };
    int64_t n_round = 0;

    // pick the arm of the next round
    int32_t choose() const;
    // record a finished round of arm `arm`: `n_emit` tokens emitted in `t_us` microseconds
    void observe(int32_t arm, int32_t n_emit, int64_t t_us);
    // tree columns for an arm given D draft tokens (0 for the chain)
    static int32_t arm_nodes(int32_t arm, int32_t D);
};
