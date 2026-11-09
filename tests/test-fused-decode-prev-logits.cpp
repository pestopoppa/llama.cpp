// FUSED-DECODE-ZERO-ROW-LOGITS: the fused path may only reuse a previous t_logits
// that has a row to write. A zero-output ubatch leaves ne[1] == 0, which aborted
// production v10 with "tensor read out of bounds" in llama_context::decode.
#include "../src/llama-fused-guard.h"

#include <cstdio>

static int check(const char * name, bool got, bool want) {
    if (got != want) {
        fprintf(stderr, "FAIL %s: got %d want %d\n", name, got, want);
        return 1;
    }
    return 0;
}

int main() {
    const int64_t n_vocab = 1000;
    char storage[16];
    int fails = 0;

    ggml_tensor t = {};
    t.data = storage;
    t.ne[0] = n_vocab;

    t.ne[1] = 0;
    fails += check("zero rows", llama_fused_decode_prev_logits_ok(&t, n_vocab), false);
    t.ne[1] = 1;
    fails += check("one row", llama_fused_decode_prev_logits_ok(&t, n_vocab), true);
    t.ne[1] = 4;
    fails += check("many rows", llama_fused_decode_prev_logits_ok(&t, n_vocab), true);
    fails += check("wrong width", llama_fused_decode_prev_logits_ok(&t, n_vocab + 1), false);
    t.data = nullptr;
    fails += check("unallocated", llama_fused_decode_prev_logits_ok(&t, n_vocab), false);
    fails += check("null", llama_fused_decode_prev_logits_ok(nullptr, n_vocab), false);

    if (fails == 0) {
        printf("PASS\n");
    }
    return fails == 0 ? 0 : 1;
}
