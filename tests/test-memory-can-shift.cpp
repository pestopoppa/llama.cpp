// Regression test: a memory that carries recurrent state (pure recurrent or hybrid attention+recurrent)
// must report llama_memory_can_shift() == false. KV-shift chunk reuse (--cache-reuse) and context shift
// move attention KV with llama_memory_seq_add(), which for recurrent state only rewrites cell.pos and
// leaves the state stale -> silently wrong output. Non-recurrent models must still report true
// (unless they have a model-specific reason not to), so the guard is not over-broad.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cstdio>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init model/context\n", __func__);
        return 1;
    }

    const bool has_recurrent = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    const bool can_shift     = llama_memory_can_shift(llama_get_memory(ctx));

    fprintf(stderr, "%s : recurrent_or_hybrid=%d can_shift=%d\n", __func__, has_recurrent, can_shift);

    if (has_recurrent && can_shift) {
        fprintf(stderr, "%s : FAIL: memory with recurrent state reports can_shift=true\n", __func__);
        return 1;
    }
    if (!has_recurrent && !can_shift) {
        fprintf(stderr, "%s : FAIL: plain attention model reports can_shift=false (guard too broad)\n", __func__);
        return 1;
    }
    return 0;
}
