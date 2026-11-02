#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

// A diagnostic classification of one cache cell relative to the query row.
enum class llama_kv_block_owner { empty, unknown, current, foreign, shared };
enum class llama_kv_block_visibility { masked, visible, invalid };

struct llama_kv_block_counts {
    uint64_t blocks = 0;
    uint64_t live = 0;
    uint64_t mixed = 0;
    uint64_t live_unknown = 0;
    uint64_t unknown = 0;
    uint64_t invalid_mask = 0;
};

// Count per-row 256-cell blocks. A mixed block contains a visible cell and a
// foreign single-owner cell; multi-owner shared cells never count as foreign.
// Unknown ownership is recorded independently, including in dead blocks.
template<typename Owner, typename Visibility>
llama_kv_block_counts llama_kv_block_observe(size_t n, Owner owner, Visibility visibility) {
    constexpr size_t block_size = 256;
    llama_kv_block_counts out;
    for (size_t base = 0; base < n; base += block_size) {
        bool live = false;
        bool foreign = false;
        bool unknown = false;
        for (size_t j = base; j < std::min(n, base + block_size); ++j) {
            const auto v = visibility(j);
            live |= v == llama_kv_block_visibility::visible;
            out.invalid_mask += v == llama_kv_block_visibility::invalid;
            const auto o = owner(j);
            foreign |= o == llama_kv_block_owner::foreign;
            unknown |= o == llama_kv_block_owner::unknown;
        }
        ++out.blocks;
        out.live += live;
        out.mixed += live && foreign;
        out.unknown += unknown;
        out.live_unknown += live && unknown;
    }
    return out;
}
