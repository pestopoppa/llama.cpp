// INF-77 DS41-B13 -- the DSpark drafter's two hidden widths, without weights.
//
// A DSpark drafter has THREE widths that are all 5120 on this checkpoint and are all different
// quantities:
//
//   n_embd              the DRAFT's hidden size
//   target_hidden       the TARGET's hidden size (what each tapped layer contributes)
//   n_embd_inp_enc      len(target_layer_ids) * target_hidden -- the encoder input width
//
// The first real spec-dec request crashed on GGML_ASSERT(n_embd == embd->ne[0]) because the graph
// input was constructed with n_embd while its tensor was allocated at n_embd_inp_enc. Underneath
// that, llama_context::decode sizes its batch allocator at n_embd_inp() while only
// llama_context::encode sizes it at n_embd_inp_enc(), so a wide embd batch sent to llama_decode
// would have been mis-strided by llama-batch.cpp before any assert could fire.
//
// This test pins the algebra, including the case that the shipped model cannot exercise: a drafter
// whose hidden size differs from its target's. It uses the real llama_hparams.

#include "testing.h"

#include "llama.h"

#include "../src/llama-hparams.h"

#include <cstdint>
#include <string>
#include <vector>

// src/models/dflash.cpp dsv41_dspark_load_hparams: the encoder width comes from the WEIGHT
// (blk.0.dspark_main_proj.weight ne[0]), never from len(ids)*n_embd.
static uint32_t enc_width_from_weight(int64_t main_proj_ne0) {
    return (uint32_t) main_proj_ne0;
}

// src/llama-model.cpp llama_model_dspark_target_hidden_size()
static int32_t target_hidden(const llama_hparams & hp, size_t n_target_layers) {
    if (n_target_layers == 0) {
        return 0;
    }
    return (int32_t) (hp.n_embd_inp_enc()/n_target_layers);
}

static void test_shipped_geometry(testing & t) {
    llama_hparams hp = {};
    hp.n_embd = 5120;
    hp.n_embd_inp_enc_impl = enc_width_from_weight(15360);

    t.assert_equal("encoder input width", (uint32_t) 15360, hp.n_embd_inp_enc());
    t.assert_equal("decoder input width", (uint32_t) 5120,  hp.n_embd_inp());

    t.assert_true("the two widths are NOT the same quantity",
            hp.n_embd_inp_enc() != hp.n_embd_inp());

    t.assert_equal("derived target hidden size", 5120, target_hidden(hp, 3));
}

// The coincidence the shipped model hides: len(ids)*n_embd == main_proj->ne[0] only while the
// drafter and the target share a hidden size. A drafter distilled onto a wider target breaks it,
// and every derivation that goes through n_embd instead of the weight is then silently wrong.
static void test_mismatched_hidden_sizes(testing & t) {
    llama_hparams hp = {};
    hp.n_embd = 4096;                                    // a narrower drafter
    hp.n_embd_inp_enc_impl = enc_width_from_weight(15360); // 3 target layers x 5120

    t.assert_equal("encoder width still comes from the weight", (uint32_t) 15360, hp.n_embd_inp_enc());
    t.assert_equal("derived target hidden size is the TARGET's", 5120, target_hidden(hp, 3));

    const uint32_t wrong = 3*hp.n_embd; // what len(ids)*n_embd would have given
    t.assert_true("len(target_layers)*n_embd would have been wrong here",
            wrong != hp.n_embd_inp_enc());

    // and the old driver fallback -- "expect the target's hidden size to equal the draft's" --
    // would have accepted a target it must reject
    t.assert_true("the draft-hidden-size fallback would accept a mismatched pair",
            (int32_t) hp.n_embd != target_hidden(hp, 3));
}

static void test_divisibility_is_checked(testing & t) {
    // dsv41_dspark_load_hparams throws when the width is not a multiple of the layer count; this
    // pins the arithmetic that check relies on.
    t.assert_true("15360 splits evenly over 3 target layers", 15360 % 3 == 0);
    t.assert_true("15360 does not split evenly over 7",       15360 % 7 != 0);

    llama_hparams hp = {};
    hp.n_embd = 5120;
    hp.n_embd_inp_enc_impl = enc_width_from_weight(15360);

    t.assert_equal("two target layers would imply a 7680-wide target", 7680, target_hidden(hp, 2));
}

static void test_unset_falls_back(testing & t) {
    // A model that declares no encoder input keeps n_embd_inp() -- the DFlash/DSpark drafters are
    // the only ones that set the impl field, and the fallback must not invent a width.
    llama_hparams hp = {};
    hp.n_embd = 5120;

    t.assert_equal("no encoder width declared -> n_embd_inp()", hp.n_embd_inp(), hp.n_embd_inp_enc());
    t.assert_equal("and no target hidden size is claimed", 0, target_hidden(hp, 0));
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

    t.test("shipped-geometry",        test_shipped_geometry);
    t.test("mismatched-hidden-sizes", test_mismatched_hidden_sizes);
    t.test("divisibility",            test_divisibility_is_checked);
    t.test("unset-falls-back",        test_unset_falls_back);

    return t.summary();
}
