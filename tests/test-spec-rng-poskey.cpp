// SPEC-RNG.1: position-keyed counter RNG. Checks (a) the RNG is stateless/order-free,
// (b) the dist sampler's output depends only on (seed, absolute position), so interleaving
// (np 1 vs 4) and extra speculative re-draws at the same position (depth 0 vs 3) do not change it,
// (c) the switch is the constructor (llama_sampler_init_dist_poskey), NOT the environment: the default dist sampler is untouched
//     by any LLAMA_RNG_POSKEY value, and LLAMA_RNG_POSKEY=0 is only a kill switch that turns the poskey constructor into the stateful dist.
#include "rng-poskey.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static llama_token draw(llama_sampler * s, int seed_variant) {
    const int n = 64;
    std::vector<llama_token_data> d(n);
    for (int i = 0; i < n; ++i) { d[i] = { i, 0.05f * (float) ((i * 7 + seed_variant) % 13), 0.0f }; }
    llama_token_data_array arr = { d.data(), d.size(), -1, false };
    llama_sampler_apply(s, &arr);
    return arr.data[arr.selected].id;
}

int main() {
    // the env var can never ENABLE the feature: =1 changes nothing, =0 is the kill switch (checked in (c))
    unsetenv("LLAMA_RNG_POSKEY");
    CHECK(!llama_rng_poskey_killed());

    // (a) statelessness + range + rough uniformity + role/pos/seed sensitivity
    double sum = 0; const int N = 100000;
    for (int i = 0; i < N; ++i) { double u = llama_rng_poskey_u01(42, i, 1); CHECK(u >= 0.0 && u < 1.0); sum += u; }
    CHECK(sum / N > 0.49 && sum / N < 0.51);
    CHECK(llama_rng_poskey_u01(42, 7, 1) == llama_rng_poskey_u01(42, 7, 1));
    CHECK(llama_rng_poskey_u01(42, 7, 1) != llama_rng_poskey_u01(42, 7, 2));
    CHECK(llama_rng_poskey_u01(42, 7, 1) != llama_rng_poskey_u01(42, 8, 1));
    CHECK(llama_rng_poskey_u01(42, 7, 1) != llama_rng_poskey_u01(43, 7, 1));

    // (b) reference sequence: depth-0, single stream
    const int L = 200;
    std::vector<llama_token> ref(L);
    {
        llama_sampler * s = llama_sampler_init_dist_poskey(1234);
        for (int p = 0; p < L; ++p) { ref[p] = draw(s, p); llama_sampler_accept(s, ref[p]); }
        llama_sampler_free(s);
    }
    // np=4: four interleaved streams with the same seed; every stream must equal ref
    {
        llama_sampler * s[4];
        for (auto & x : s) { x = llama_sampler_init_dist_poskey(1234); }
        std::vector<llama_token> out[4];
        for (int p = 0; p < L; ++p) {
            for (int k = 3; k >= 0; --k) { llama_token t = draw(s[k], p); out[k].push_back(t); llama_sampler_accept(s[k], t); }
        }
        for (int k = 0; k < 4; ++k) { CHECK(out[k] == ref); }
        for (auto & x : s) { llama_sampler_free(x); }
    }
    // depth 3: before each accepted token, 3 speculative extra draws at the same position
    // (rejected drafts re-sample without accepting) must not perturb the stream.
    {
        llama_sampler * s = llama_sampler_init_dist_poskey(1234);
        std::vector<llama_token> out;
        for (int p = 0; p < L; ++p) {
            for (int r = 0; r < 3; ++r) { (void) draw(s, p); }
            llama_token t = draw(s, p); out.push_back(t); llama_sampler_accept(s, t);
        }
        CHECK(out == ref);
        // clone mid-stream continues identically
        llama_sampler * c = llama_sampler_clone(s);
        CHECK(draw(c, L) == draw(s, L));
        llama_sampler_free(c); llama_sampler_free(s);
    }
    // (c) env is not the switch. The default dist sampler is stateful (mt19937): two draws at the SAME position differ.
    {
        setenv("LLAMA_RNG_POSKEY", "1", 1);                    // =1 must change nothing
        llama_sampler * d = llama_sampler_init_dist(1234);      // default constructor: stateful rng
        bool all_same = true;
        llama_token first = draw(d, 0);
        for (int r = 0; r < 50; ++r) { if (draw(d, 0) != first) { all_same = false; } }
        CHECK(!all_same);                                       // repeated draws at one position vary: not position-keyed
        llama_sampler_free(d);
        // the poskey constructor is position-keyed with env =1 or unset
        llama_sampler * k = llama_sampler_init_dist_poskey(1234);
        bool k_same = true;
        llama_token k0 = draw(k, 0);
        for (int r = 0; r < 50; ++r) { if (draw(k, 0) != k0) { k_same = false; } }
        CHECK(k_same);
        llama_sampler_free(k);
        // kill switch: LLAMA_RNG_POSKEY=0 makes the poskey constructor behave as the stateful dist
        setenv("LLAMA_RNG_POSKEY", "0", 1);
        CHECK(llama_rng_poskey_killed());
        llama_sampler * z = llama_sampler_init_dist_poskey(1234);
        bool z_same = true;
        llama_token z0 = draw(z, 0);
        for (int r = 0; r < 50; ++r) { if (draw(z, 0) != z0) { z_same = false; } }
        CHECK(!z_same);
        llama_sampler_free(z);
        unsetenv("LLAMA_RNG_POSKEY");
    }
    printf("RESULT test-spec-rng-poskey fail=%d\n", n_fail);
    return n_fail ? 1 : 0;
}
