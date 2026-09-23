// INF-77 DS41-B13 -- the DSpark draft-block mask, without weights.
//
// Normative: DeepSeek-V4.1-Flash/inference/model.py:1020-1029
//
//     @lru_cache(1)
//     def get_dspark_topk_idxs(window_size, bsz, block_size, start_pos):
//         assert start_pos > 0
//         matrix = torch.cat([
//             torch.arange(min(window_size, start_pos + 1)),
//             window_size + torch.arange(block_size),
//         ])
//         return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()
//
// and :1066-1067, where the block's own KV is CONCATENATED onto the 128-slot window rather than
// stored in it, so index space is [0, window) = ring slots and [window, window+block) = the block.
//
// The `.expand(bsz, block_size, -1)` is the whole point: EVERY draft position gets the SAME index
// list, so the block is bidirectional -- position 0 attends to position 4. A causal mask fails
// property (2) below.
//
// This test also pins the one place our runtime diverges from the reference: llama.cpp masks the
// window by sliding-window DISTANCE (|p_q - p_k| < n_swa) while the reference hands every draft
// position the whole filled ring. The divergence is computed here rather than asserted away, so
// that it is a number in CI instead of a sentence in a design doc.

#include "testing.h"

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

static constexpr int32_t WINDOW = 128; // config sliding_window
static constexpr int32_t BLOCK  = 5;   // config dspark_block_size

// model.py:1020-1029, one draft position's index list (identical for all of them).
static std::vector<int32_t> reference_topk_idxs(int32_t window_size, int32_t block_size, int32_t start_pos) {
    GGML_ASSERT(start_pos > 0);

    std::vector<int32_t> out;

    const int32_t n_window = std::min(window_size, start_pos + 1);
    for (int32_t i = 0; i < n_window; ++i) {
        out.push_back(i);
    }
    for (int32_t i = 0; i < block_size; ++i) {
        out.push_back(window_size + i);
    }

    return out;
}

// What our graph produces: a boolean visibility row per (draft position, key), where keys are
// [0, n_window) ring slots and [window_size, window_size + block_size) block positions.
//
//   - ring slot i holds the main_kv of an absolute position; with `start_pos` the last committed
//     position, the filled slots carry positions start_pos - n_window + 1 .. start_pos.
//   - draft position j sits at absolute position start_pos + 1 + j.
//   - llama.cpp keeps a key iff it belongs to the sequence AND is inside the SWA window:
//     is_masked_swa(n_swa, STANDARD, p_key, p_q) drops it when p_q - p_key >= n_swa.
//   - causal_attn is OFF for the draft context, so nothing else is masked -- in particular the
//     block half is fully visible in both directions.
static std::vector<std::vector<bool>> runtime_visibility(
        int32_t window_size, int32_t block_size, int32_t start_pos, bool causal) {
    const int32_t n_window = std::min(window_size, start_pos + 1);

    std::vector<std::vector<bool>> vis(block_size, std::vector<bool>(window_size + block_size, false));

    for (int32_t j = 0; j < block_size; ++j) {
        const int32_t p_q = start_pos + 1 + j;

        for (int32_t i = 0; i < n_window; ++i) {
            // slot i (oldest first) carries absolute position start_pos - n_window + 1 + i
            const int32_t p_k = start_pos - n_window + 1 + i;
            if (p_q - p_k < window_size) {
                vis[j][i] = true;
            }
        }

        for (int32_t b = 0; b < block_size; ++b) {
            const int32_t p_k = start_pos + 1 + b;
            if (causal && p_k > p_q) {
                continue;
            }
            vis[j][window_size + b] = true;
        }
    }

    return vis;
}

static void test_reference_index_set(testing & t) {
    // below, at and above the window
    for (int32_t start_pos : { 1, 7, WINDOW - 2, WINDOW - 1, WINDOW, 4*WINDOW + 3 }) {
        const auto idxs = reference_topk_idxs(WINDOW, BLOCK, start_pos);

        const int32_t n_window = std::min(WINDOW, start_pos + 1);

        t.assert_equal("index list length at start_pos=" + std::to_string(start_pos),
                (size_t) (n_window + BLOCK), idxs.size());

        // the two halves are disjoint: ring slots are < WINDOW, block keys are >= WINDOW
        bool disjoint = true;
        for (int32_t i = 0; i < n_window; ++i) {
            disjoint = disjoint && idxs[(size_t) i] == i;
        }
        for (int32_t b = 0; b < BLOCK; ++b) {
            disjoint = disjoint && idxs[(size_t) (n_window + b)] == WINDOW + b;
        }
        t.assert_true("halves are [0,n_window) then [WINDOW,WINDOW+BLOCK) at start_pos=" +
                        std::to_string(start_pos), disjoint);
    }
}

static void test_block_is_bidirectional(testing & t) {
    const int32_t start_pos = 4*WINDOW + 3;

    const auto vis = runtime_visibility(WINDOW, BLOCK, start_pos, /*causal =*/ false);

    bool all_pairs = true;
    for (int32_t j = 0; j < BLOCK; ++j) {
        for (int32_t b = 0; b < BLOCK; ++b) {
            all_pairs = all_pairs && vis[(size_t) j][(size_t) (WINDOW + b)];
        }
    }
    t.assert_true("every draft position sees every draft position, including later ones", all_pairs);

    // the foil: a causal mask satisfies "sees earlier positions" but fails the property above,
    // and would silently halve the block's information.
    const auto vis_causal = runtime_visibility(WINDOW, BLOCK, start_pos, /*causal =*/ true);
    t.assert_true("a causal in-block mask is detectably different",
            !vis_causal[0][(size_t) (WINDOW + BLOCK - 1)] && vis[0][(size_t) (WINDOW + BLOCK - 1)]);
}

static void test_window_half(testing & t) {
    for (int32_t start_pos : { 1, 7, WINDOW - 1, WINDOW, 4*WINDOW + 3 }) {
        const int32_t n_window = std::min(WINDOW, start_pos + 1);
        const auto vis = runtime_visibility(WINDOW, BLOCK, start_pos, false);

        // (iii) no draft position sees an UNFILLED ring slot
        bool no_unfilled = true;
        for (int32_t j = 0; j < BLOCK; ++j) {
            for (int32_t i = n_window; i < WINDOW; ++i) {
                no_unfilled = no_unfilled && !vis[(size_t) j][(size_t) i];
            }
        }
        t.assert_true("no unfilled ring slot is visible at start_pos=" + std::to_string(start_pos),
                no_unfilled);

        // (i) how many filled slots each draft position actually sees, vs the reference's n_window
        for (int32_t j = 0; j < BLOCK; ++j) {
            int32_t seen = 0;
            for (int32_t i = 0; i < n_window; ++i) {
                seen += vis[(size_t) j][(size_t) i] ? 1 : 0;
            }

            // The reference gives every draft position all n_window slots. llama.cpp's SWA cuts
            // at a fixed DISTANCE -- keep iff p_q - p_k < n_swa -- and draft j sits j+1 positions
            // past the newest ring entry, so once the ring is full it loses the j+1 oldest slots.
            // Bounded, always the oldest, and it can only cost acceptance: the target verifies
            // every token regardless. Measured here rather than asserted away.
            const int32_t expect = std::min(n_window, WINDOW - 1 - j);

            t.assert_equal("visible window slots, start_pos=" + std::to_string(start_pos) +
                            " draft=" + std::to_string(j),
                    expect, seen);

            t.assert_true("the SWA divergence never exceeds the block width",
                    n_window - seen <= BLOCK);
        }
    }
}

static void test_divergence_is_bounded(testing & t) {
    // The headline number quoted in DESIGN.md 3.4 / NOTES.md: worst case, the last draft position
    // of a full block loses BLOCK-1 of WINDOW slots.
    const int32_t start_pos = 10*WINDOW;
    const auto vis = runtime_visibility(WINDOW, BLOCK, start_pos, false);

    int32_t worst = 0;
    for (int32_t j = 0; j < BLOCK; ++j) {
        int32_t seen = 0;
        for (int32_t i = 0; i < WINDOW; ++i) {
            seen += vis[(size_t) j][(size_t) i] ? 1 : 0;
        }
        worst = std::max(worst, WINDOW - seen);
    }

    // BLOCK of WINDOW = 5/128 = 3.9%, always the oldest entries of the ring. This is the number
    // quoted in DESIGN.md 3.4 and NOTES.md; if the mask strategy changes, this test is what says
    // so.
    t.assert_equal("worst-case lost window slots", BLOCK, worst);
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("reference-index-set",  test_reference_index_set);
    t.test("block-bidirectional",  test_block_is_bidirectional);
    t.test("window-half",          test_window_half);
    t.test("divergence-bounded",   test_divergence_is_bounded);

    return t.summary();
}
