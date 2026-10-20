#pragma once
// SPEC-RNG.1 (experimental): position-keyed counter RNG.
// u = SplitMix64-finalizer(seed, absolute position, role) -> uniform double in [0,1).
// Stateless: the draw for position P is identical regardless of how many other draws happened
// before it (np 1 vs 4, speculative depth 0 vs 3). Enabled only when LLAMA_RNG_POSKEY=1.
#include <cstdint>
#include <cstdlib>

enum llama_rng_role : uint64_t {
    LLAMA_RNG_ROLE_DIST     = 1, // target dist sampler draw
    LLAMA_RNG_ROLE_SPEC_ACC = 2, // speculative accept test
    LLAMA_RNG_ROLE_SPEC_RES = 3, // speculative residual draw
};

static inline uint64_t llama_rng_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double llama_rng_poskey_u01(uint64_t seed, uint64_t pos, uint64_t role) {
    uint64_t h = llama_rng_splitmix64(seed);
    h = llama_rng_splitmix64(h ^ pos);
    h = llama_rng_splitmix64(h ^ (role * 0xd6e8feb86659fd93ULL));
    return (double) (h >> 11) * (1.0 / 9007199254740992.0); // 53 bits -> [0,1)
}

static inline bool llama_rng_poskey_enabled() {
    static const bool v = [] { const char * e = std::getenv("LLAMA_RNG_POSKEY"); return e && e[0] == '1'; }();
    return v;
}
