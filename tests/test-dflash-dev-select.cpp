// Q8-SERVED-DEV-SELECT-1: the device DFlash2 selector chain (src/llama-dflash-select.h) must give the same draft ids as
// the host chain in common/speculative.cpp draft() on every backend, including heavy-tie lattices.
#include "ggml.h"
#include "ggml-backend.h"
#include "../src/llama-dflash-select.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

// verbatim copy of the host greedy chain (speculative.cpp, DFlash2 branch, temperature <= 0)
static std::vector<int> host_chain(const std::vector<float> & lat, int n_embd, int top_k, int bs, int blk) {
    std::vector<int> out;
    int predecessor = 0;
    for (int i = 1; i < bs; ++i) {
        const float * row = lat.data() + (size_t) (blk * bs + i) * n_embd;
        const float * scores = row + top_k + (size_t) predecessor * top_k;
        predecessor = (int) std::distance(scores, std::max_element(scores, scores + top_k));
        out.push_back((int) row[predecessor]);
    }
    return out;
}

static int run_case(ggml_backend_t be, int n_embd, int top_k, int bs, int nb, int levels, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> lv(0, levels - 1);
    std::uniform_int_distribution<int> tid(0, 248319);
    std::vector<float> lat((size_t) n_embd * bs * nb, 0.0f);
    for (int r = 0; r < bs * nb; ++r) {
        if (r % bs == 0) continue; // padding row
        float * row = lat.data() + (size_t) r * n_embd;
        for (int k = 0; k < top_k; ++k) row[k] = (float) tid(rng);
        for (int k = 0; k < top_k * top_k; ++k) row[top_k + k] = levels > 0 ? (float) lv(rng) * 0.5f - 3.0f
                                                                         : std::normal_distribution<float>(0, 2)(rng);
        if (levels == 1) { /* all-equal scores: every argmax is a full tie -> index 0 */ }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, bs * nb);
    ggml_tensor * out = llama_dflash2_dev_select(ctx, t, top_k, bs, nb);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(t, lat.data(), 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-8s top_k=%d bs=%d nb=%d levels=%d seed=%u : SKIP (compute failed)\n", ggml_backend_name(be), top_k, bs, nb, levels, seed);
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 2;
    }
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    int bad = 0;
    for (int b = 0; b < nb; ++b) {
        auto ref = host_chain(lat, n_embd, top_k, bs, b);
        for (int i = 0; i < bs - 1; ++i) {
            if ((int) got[(size_t) b * (bs - 1) + i] != ref[i]) ++bad;
        }
    }
    printf("  %-8s top_k=%d bs=%d nb=%d levels=%d seed=%u : %s (%d mismatches)\n", ggml_backend_name(be), top_k, bs,
           nb, levels, seed, bad ? "FAIL" : "ok", bad);
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bad ? 1 : 0;
}

int main() {
    ggml_backend_load_all();
    int fails = 0, n_be = 0;
    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) continue;
        ++n_be;
        // served shape (n_embd 5120 >= 8*9, block 8 -> 7 drafts), plus wide top_k, multi-block, and tie lattices
        const int cfg[][5] = { {5120, 8, 8, 1, 0}, {5120, 8, 8, 1, 3}, {5120, 8, 8, 1, 1}, {5120, 8, 8, 4, 2},
                               {1024, 16, 16, 3, 0}, {1024, 16, 16, 3, 2}, {5120, 32, 8, 2, 0}, {5120, 32, 8, 2, 4} };
        int n = 0, ok = 0, bad = 0, skip = 0;
        for (auto & c : cfg) for (uint32_t s = 1; s <= 25; ++s) {
            const int r = run_case(be, c[0], c[1], c[2], c[3], c[4], s);
            ++n; ok += r == 0; bad += r == 1; skip += r == 2;
        }
        fails += bad + skip;
        // machine-readable per-backend tally: gates read THIS line, never count case lines (backend stderr can
        // interleave with stdout and split a case line)
        fflush(stderr);
        printf("TALLY backend=%s cases=%d ok=%d fail=%d skip=%d\n", ggml_backend_name(be), n, ok, bad, skip);
        fflush(stdout);
        ggml_backend_free(be);
    }
    printf("backends=%d failing_cases=%d -> %s\n", n_be, fails, (fails == 0 && n_be > 0) ? "PASS" : "FAIL");
    return (fails == 0 && n_be > 0) ? 0 : 1;
}
