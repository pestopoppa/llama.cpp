#pragma once

#include <cstdint>
#include <array>

namespace math_tools {

// Xoshiro256++ - fast, high-quality PRNG
// Reference: https://prng.di.unimi.it/
class Xoshiro256pp {
public:
    using result_type = uint64_t;

    explicit Xoshiro256pp(uint64_t seed = 12345) {
        // SplitMix64 for seeding
        for (int i = 0; i < 4; ++i) {
            seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
            seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
            state_[i] = seed ^ (seed >> 31);
        }
    }

    uint64_t operator()() {
        const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const uint64_t t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];

        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);

        return result;
    }

    // Uniform double in [0, 1)
    double uniform() {
        return ((*this)() >> 11) * 0x1.0p-53;
    }

    // Uniform double in [a, b)
    double uniform(double a, double b) {
        return a + (b - a) * uniform();
    }

    // Standard normal via Box-Muller
    double normal() {
        double u1 = uniform();
        double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

    double normal(double mean, double stddev) {
        return mean + stddev * normal();
    }

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }

private:
    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    std::array<uint64_t, 4> state_;
};

} // namespace math_tools
