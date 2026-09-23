// INF-77 DS41-B13 / DS41-B14 -- the DSpark speculative rollback contract, without weights.
//
// What is under test is NOT "does seq_rm compile". It is the one property that makes a DeepSeek
// V4.1 rollback different from a position rewind:
//
//   the compressor's kv_state / score_state are FIXED-SIZE ACCUMULATORS indexed by pos % ratio.
//   A slot is written more than once inside a speculative window, so the pre-step value cannot be
//   recomputed by rewinding a counter -- it has to have been SAVED.
//
// The test therefore does snapshot -> mutate -> restore -> byte-compare on a buffer laid out
// exactly like llama_dsv4_comp_state (src/llama-kv-cache-dsv4.cpp:944-957), reproducing the plane
// arithmetic of dsv4_build_comp_plan (:655-709) independently, and then runs the counter-rewind
// FOIL and asserts it does NOT reproduce the byte image.
//
// The Engram half uses the real llama_dsv41_engram_state, which is host-side and needs no model.

#include "testing.h"

#include "llama.h"

#include "../src/llama-dsv41-engram.h"

#include <cstdint>
#include <cstring>
#include <vector>

//
// A faithful stand-in for one layer of llama_dsv4_comp_state.
//
// Layout (src/llama-kv-cache-dsv4.cpp:944): ggml_new_tensor_3d(F32, n_embd_state, state_size,
// n_stream*(1 + n_rs_seq)). Row index arithmetic (:513-514, :662-668, :685-705):
//
//   row(plane d, seq s, slot r) = d*state_rows + s*state_size + r,   state_rows = state_size*n_stream
//
struct comp_state_model {
    uint32_t n_embd_state;
    uint32_t state_size;   // == ratio for the non-overlapping V4.1 geometry
    uint32_t n_stream;
    uint32_t n_rs_seq;

    std::vector<float> kv;

    comp_state_model(uint32_t n_embd_state, uint32_t state_size, uint32_t n_stream, uint32_t n_rs_seq)
        : n_embd_state(n_embd_state), state_size(state_size), n_stream(n_stream), n_rs_seq(n_rs_seq),
          kv((size_t) n_embd_state*state_size*n_stream*(1 + n_rs_seq), 0.0f) {
    }

    uint32_t state_rows() const { return state_size*n_stream; }

    size_t row(uint32_t plane, uint32_t seq, uint32_t slot) const {
        return (size_t) (plane*state_rows() + seq*state_size + slot)*n_embd_state;
    }

    float * at(uint32_t plane, uint32_t seq, uint32_t slot) {
        return kv.data() + row(plane, seq, slot);
    }

    const float * at(uint32_t plane, uint32_t seq, uint32_t slot) const {
        return kv.data() + row(plane, seq, slot);
    }

    // What deepseek41.cpp:1030-1036 does per token: the compressor carries its partial group
    // forward by ACCUMULATING into the slot for this position. Destructive: the previous value is
    // gone, and there is no per-position log anywhere in the cache that could reconstruct it.
    void step(uint32_t seq, llama_pos pos, float delta) {
        const uint32_t slot = (uint32_t) (pos % (llama_pos) state_size);
        float * dst = at(0, seq, slot);
        for (uint32_t j = 0; j < n_embd_state; ++j) {
            dst[j] = dst[j]*0.5f + delta*(float) (j + 1);
        }
    }

    // dsv4_build_comp_plan, :683-705. For depth d, plane d receives "the state as of d tokens
    // ago". The real plan reads either the live plane or the ubatch scratch region; the observable
    // contract is the same, so the model snapshots by shifting planes before each step.
    void snapshot_before_step(uint32_t seq) {
        for (uint32_t d = n_rs_seq; d >= 1; --d) {
            const uint32_t src = d - 1; // plane 0 is live
            for (uint32_t r = 0; r < state_size; ++r) {
                std::memcpy(at(d, seq, r), at(src, seq, r), (size_t) n_embd_state*sizeof(float));
            }
        }
    }

    // dsv41_build_state_restore (src/models/deepseek41.cpp:576-597) + plan :662-669.
    void restore(uint32_t seq, uint32_t rollback) {
        GGML_ASSERT(rollback >= 1 && rollback <= n_rs_seq);
        for (uint32_t r = 0; r < state_size; ++r) {
            std::memcpy(at(0, seq, r), at(rollback, seq, r), (size_t) n_embd_state*sizeof(float));
        }
    }

    std::vector<float> plane0(uint32_t seq) const {
        std::vector<float> out((size_t) n_embd_state*state_size);
        for (uint32_t r = 0; r < state_size; ++r) {
            std::memcpy(out.data() + (size_t) r*n_embd_state, at(0, seq, r),
                    (size_t) n_embd_state*sizeof(float));
        }
        return out;
    }
};

static bool bytes_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

//
// 1. snapshot -> mutate -> restore -> byte-compare, for every rollback depth a 5-wide DSpark
//    block can produce.
//
static void test_restore_is_byte_exact(testing & t) {
    // V4.1 csa geometry: ratio 2, n_embd_state = n_embd_head_k. Trimmed width; the arithmetic is
    // width-independent and a narrow buffer keeps the test cheap.
    const uint32_t n_embd_state = 8;
    const uint32_t state_size   = 2;
    const uint32_t n_stream     = 2;
    const uint32_t n_rs_seq     = 5; // == the DSpark block size

    for (uint32_t rollback = 1; rollback <= n_rs_seq; ++rollback) {
        comp_state_model s(n_embd_state, state_size, n_stream, n_rs_seq);

        const uint32_t seq   = 1;
        const uint32_t other = 0;

        // warm the accumulator so that the slots hold something a zero-init could not fake
        llama_pos pos = 0;
        for (; pos < 7; ++pos) {
            s.snapshot_before_step(seq);
            s.step(seq, pos, 0.25f*(float) (pos + 1));
            s.step(other, pos, -1.0f);
        }

        // the verify step: n_rs_seq speculative tokens, each one a destructive accumulate
        const std::vector<float> want = s.plane0(seq);
        const std::vector<float> want_other = s.plane0(other);

        const llama_pos pos_commit = pos - 1;

        for (uint32_t k = 0; k < rollback; ++k) {
            s.snapshot_before_step(seq);
            s.step(seq, pos_commit + 1 + (llama_pos) k, 3.5f + (float) k);
        }

        s.restore(seq, rollback);

        t.assert_true("plane 0 is byte-identical to the pre-verify state (rollback=" +
                        std::to_string(rollback) + ")",
                bytes_equal(want, s.plane0(seq)));

        t.assert_true("restoring one sequence does not touch another (rollback=" +
                        std::to_string(rollback) + ")",
                bytes_equal(want_other, s.plane0(other)));
    }
}

//
// 2. THE FOILS. Two things a "rollback" could plausibly mean that are NOT a value snapshot, and
//    that a naive llama_kv_cache_seq_rm port amounts to. Both must fail to reproduce the bytes.
//
//    (a) a pure POSITION rewind: put the cell metadata back and leave the accumulator alone. This
//        is literally what seq_rm does to a KV cache, and it is why DS41-B14 says a naive port
//        corrupts state silently -- nothing errors, the numbers are just wrong from then on.
//    (b) a RECOMPUTE from the counter: reinitialise the slot from the last committed position's
//        contribution, as if the slot held only the current group. The accumulator is
//        exponential, so its content is a function of the whole history, not of the group.
//
//    Note what is deliberately NOT tested as a foil: replaying every rejected delta in reverse.
//    That IS exact -- but it needs a per-step value log, which is a snapshot by another name and
//    is not something the cache keeps. (This test asserted the opposite on its first run and was
//    wrong; the correction is the reason the foils are spelled out rather than hand-waved.)
//
static void test_counter_rewind_is_not_enough(testing & t) {
    const uint32_t n_embd_state = 8;
    const uint32_t state_size   = 2;   // < block, so slots are revisited inside the window
    const uint32_t n_stream     = 1;
    const uint32_t n_rs_seq     = 5;
    const uint32_t seq          = 0;

    comp_state_model s(n_embd_state, state_size, n_stream, n_rs_seq);

    llama_pos pos = 0;
    for (; pos < 7; ++pos) {
        s.snapshot_before_step(seq);
        s.step(seq, pos, 0.25f*(float) (pos + 1));
    }

    const std::vector<float> want = s.plane0(seq);
    const llama_pos pos_commit = pos - 1;

    for (uint32_t k = 0; k < n_rs_seq; ++k) {
        s.snapshot_before_step(seq);
        s.step(seq, pos_commit + 1 + (llama_pos) k, 3.5f + (float) k);
    }

    // foil (a): do nothing to the accumulator
    {
        const comp_state_model foil = s;
        t.assert_true("a pure position rewind does NOT reproduce the pre-verify bytes",
                !bytes_equal(want, foil.plane0(seq)));
    }

    // foil (b): recompute each slot from the last committed position alone
    {
        comp_state_model foil = s;
        for (uint32_t r = 0; r < state_size; ++r) {
            float * dst = foil.at(0, seq, r);
            for (uint32_t j = 0; j < n_embd_state; ++j) {
                dst[j] = 0.25f*(float) (pos_commit + 1)*(float) (j + 1);
            }
        }
        t.assert_true("recomputing the slot from the counter does NOT reproduce it either",
                !bytes_equal(want, foil.plane0(seq)));
    }

    // ...while the value snapshot does, exactly.
    s.restore(seq, n_rs_seq);
    t.assert_true("the value snapshot does", bytes_equal(want, s.plane0(seq)));
}

//
// 3. The Engram n-gram state, the fourth rollback surface (DS41-B14 (d)). Real class.
//
static void test_engram_state_rollback(testing & t) {
    llama_dsv41_engram_state st(2);

    const llama_seq_id seq = 1;

    for (llama_pos p = 0; p < 12; ++p) {
        st.set(seq, p, 100 + (int32_t) p);
    }

    std::vector<int32_t> want;
    for (llama_pos p = 0; p < 12; ++p) {
        want.push_back(st.get(seq, p));
    }

    // a 5-wide speculative block, then a rejection that keeps only 2 of it
    for (llama_pos p = 12; p < 17; ++p) {
        st.set(seq, p, 900 + (int32_t) p);
    }

    st.seq_rm(seq, 14, -1);

    t.assert_true("accepted speculative positions survive",
            st.get(seq, 12) == 912 && st.get(seq, 13) == 913);

    for (llama_pos p = 14; p < 17; ++p) {
        t.assert_true("rejected position " + std::to_string((int) p) + " is DEAD",
                st.get(seq, p) == llama_dsv41_engram_spec::DEAD);
    }

    bool prefix_intact = true;
    for (llama_pos p = 0; p < 12; ++p) {
        prefix_intact = prefix_intact && st.get(seq, p) == want[(size_t) p];
    }
    t.assert_true("the committed prefix is untouched", prefix_intact);

    t.assert_true("the other sequence is untouched",
            st.get(0, 3) == llama_dsv41_engram_spec::DEAD);
}

//
// 4. The bound. seq_rm refuses rather than corrupting when the block is wider than the planes --
//    llama-kv-cache-dsv4.cpp:1507-1509. Modelled here so the relation is pinned in a test rather
//    than only in a comment: n_rollback = n_draft + 1 - n_accepted, n_accepted >= 1.
//
static void test_rollback_depth_bound(testing & t) {
    const int n_draft = 5; // the DSpark block

    int worst = 0;
    for (int n_accepted = 1; n_accepted <= n_draft + 1; ++n_accepted) {
        const int n_rollback = n_draft + 1 - n_accepted;
        worst = std::max(worst, n_rollback);
    }

    t.assert_equal("the deepest rollback a block of 5 can need", n_draft, worst);
    t.assert_true("n_rs_seq = n_max covers it", worst <= n_draft);
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

    t.test("restore-byte-exact",   test_restore_is_byte_exact);
    t.test("counter-rewind-foil",  test_counter_rewind_is_not_enough);
    t.test("engram-state",         test_engram_state_rollback);
    t.test("rollback-depth-bound", test_rollback_depth_bound);

    return t.summary();
}
