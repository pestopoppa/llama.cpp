// SPEC-BSAMP: backend sampling for speculative slots (LLAMA_SPEC_BACKEND_SAMPLING=1). Model-free checks of the
// chain-prefix rule that makes per-row device sampling exact and keeps the position-keyed draw on the host:
// (a) neutral penalties + top-k + top-p + temp go to the device prefix; the host apply then runs only the dist,
// (b) dist / dist_poskey never enter the prefix, so the host draw for a row equals the reference chain's draw on the
//     same device-filtered row (same seed, same absolute position -> same token),
// (c) a non-neutral penalties sampler closes the prefix: nothing after it is offloaded and the host runs it all,
// (d) a non-empty logit bias (per-call device inputs) closes the prefix.
#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static std::vector<llama_token_data> row(int n) {
    std::vector<llama_token_data> d(n);
    for (int i = 0; i < n; ++i) { d[i] = { i, 0.11f * (float) ((i * 37) % 53), 0.0f }; }
    return d;
}

static llama_sampler * chain_of(bool penalties_active, bool bias, bool poskey) {
    llama_sampler * c = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (bias) {
        const llama_logit_bias lb = { 3, -5.0f };
        llama_sampler_chain_add(c, llama_sampler_init_logit_bias(64, 1, &lb));
    }
    llama_sampler_chain_add(c, llama_sampler_init_penalties(64, 1.0f, 0.0f, penalties_active ? 1.5f : 0.0f));
    llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
    llama_sampler_chain_add(c, poskey ? llama_sampler_init_dist_poskey(7) : llama_sampler_init_dist(7));
    return c;
}

static void init_backend(llama_sampler * c) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(dev != nullptr);
    c->iface->backend_init(c, ggml_backend_dev_buffer_type(dev));
}

int main() {
    setenv("LLAMA_SPEC_BACKEND_SAMPLING", "1", 1);
    ggml_backend_load_all();
    CHECK(llama_spec_backend_sampling_enabled());

    // (a) + (b): host apply after init skips the prefix -> the row keeps its 64 candidates (no top-k on the host)
    for (bool poskey : { false, true }) {
        llama_sampler * dev = chain_of(false, false, poskey);
        init_backend(dev);
        auto d = row(64);
        llama_token_data_array a = { d.data(), d.size(), -1, false };
        llama_sampler_apply(dev, &a);
        CHECK(a.size == 64);
        CHECK(a.selected >= 0);

        // reference: the full chain on the host, fed the row the device prefix would hand over
        // (host-side top-k/top-p/temp produce that row; the reference dist then draws on it with the same key)
        llama_sampler * ref  = chain_of(false, false, poskey);
        llama_sampler * pre  = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(pre, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(pre, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(pre, llama_sampler_init_temp(0.6f));
        llama_sampler * dev2 = chain_of(false, false, poskey);
        init_backend(dev2);
        for (int pos = 0; pos < 50; ++pos) {
            auto r1 = row(64);
            llama_token_data_array a1 = { r1.data(), r1.size(), -1, false };
            llama_sampler_apply(ref, &a1);
            const llama_token t_ref = a1.data[a1.selected].id;

            auto r2 = row(64);
            llama_token_data_array a2 = { r2.data(), r2.size(), -1, false };
            llama_sampler_apply(pre, &a2);         // stands in for the device prefix
            llama_sampler_apply(dev2, &a2);        // host: dist only
            const llama_token t_dev = a2.data[a2.selected].id;
            CHECK(t_ref == t_dev);

            llama_sampler_accept(ref, t_ref);
            llama_sampler_accept(dev2, t_dev);
        }
        llama_sampler_free(ref); llama_sampler_free(pre); llama_sampler_free(dev2); llama_sampler_free(dev);
    }

    // (c) active penalties close the prefix: the host runs top-k again -> at most 20 candidates remain
    {
        llama_sampler * c = chain_of(true, false, true);
        init_backend(c);
        auto d = row(64);
        llama_token_data_array a = { d.data(), d.size(), -1, false };
        llama_sampler_apply(c, &a);
        CHECK(a.size <= 20);
        llama_sampler_free(c);
    }

    // (d) a non-empty logit bias closes the prefix as well
    {
        llama_sampler * c = chain_of(false, true, true);
        init_backend(c);
        auto d = row(64);
        llama_token_data_array a = { d.data(), d.size(), -1, false };
        llama_sampler_apply(c, &a);
        CHECK(a.size <= 20);
        llama_sampler_free(c);
    }

    printf("%s (%d failures)\n", n_fail ? "FAIL" : "OK", n_fail);
    return n_fail ? 1 : 0;
}
