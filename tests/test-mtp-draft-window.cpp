// [TAG_MTP_DRAFT_WINDOW] the MTP draft head's sink+window mask predicate (Q38FN-DRAFT-FA-WINDOW).
//
// The draft-only window keeps the first `sink` positions and the last `window` positions of the
// draft context's KV; window == 0 is the dense default. The mask must only ever ADD drops to the
// causal mask (it never un-masks a future cell) and must keep the diagonal.

#include "testing.h"

#include "llama.h"

#include "../src/llama-hparams.h"

#include <string>

static void test_off_is_dense(testing & t) {
    for (int p1 = 0; p1 < 70000; p1 += 997) {
        for (int p0 = 0; p0 <= p1; p0 += 101) {
            if (llama_hparams::is_masked_draft_window(0, 0, p0, p1) ||
                llama_hparams::is_masked_draft_window(0, 64, p0, p1)) {
                t.assert_true("window 0 never masks", false);
                return;
            }
        }
    }
    t.assert_true("window 0 never masks", true);
}

static void test_window_edges(testing & t) {
    const uint32_t W = 4096;
    t.assert_true("diagonal kept",                 !llama_hparams::is_masked_draft_window(W, 0, 32000, 32000));
    t.assert_true("p1-p0 = W-1 kept",              !llama_hparams::is_masked_draft_window(W, 0, 32000 - (W - 1), 32000));
    t.assert_true("p1-p0 = W dropped",              llama_hparams::is_masked_draft_window(W, 0, 32000 - W, 32000));
    t.assert_true("far past dropped",               llama_hparams::is_masked_draft_window(W, 0, 10, 32000));
    t.assert_true("short context: nothing dropped",!llama_hparams::is_masked_draft_window(W, 0, 0, W - 1));
}

static void test_sink(testing & t) {
    const uint32_t W = 2048, S = 64;
    t.assert_true("sink cell 0 kept",              !llama_hparams::is_masked_draft_window(W, S, 0, 64000));
    t.assert_true("sink cell S-1 kept",            !llama_hparams::is_masked_draft_window(W, S, S - 1, 64000));
    t.assert_true("cell S dropped",                 llama_hparams::is_masked_draft_window(W, S, S, 64000));
    int kept = 0;
    for (int p0 = 0; p0 <= 64000; ++p0) {
        kept += !llama_hparams::is_masked_draft_window(W, S, p0, 64000);
    }
    t.assert_equal("kept = sink + window", (int) (S + W), kept);
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

    t.test("off-is-dense", test_off_is_dense);
    t.test("window-edges", test_window_edges);
    t.test("sink",         test_sink);

    return t.summary();
}
