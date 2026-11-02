#include "../src/llama-kv-block-observer.h"

#include <cstdio>
#include <vector>

using O = llama_kv_block_owner;
using V = llama_kv_block_visibility;

static bool check(const llama_kv_block_counts & c, uint64_t blocks, uint64_t live,
                  uint64_t mixed, uint64_t unknown, uint64_t live_unknown, uint64_t invalid) {
    return c.blocks == blocks && c.live == live && c.mixed == mixed &&
           c.unknown == unknown && c.live_unknown == live_unknown && c.invalid_mask == invalid;
}

int main() {
    std::vector<O> owners(4*256 + 1, O::empty);
    std::vector<V> mask(owners.size(), V::masked);
    // Live private, mixed private, shared-trunk-only foreign, unknown, partial dead block.
    owners[0] = O::current; mask[0] = V::visible;
    owners[256] = O::current; mask[256] = V::visible; owners[257] = O::foreign;
    owners[512] = O::shared; mask[512] = V::visible; owners[513] = O::shared;
    owners[768] = O::current; mask[768] = V::visible; owners[769] = O::unknown;
    owners[1024] = O::unknown;
    auto count = [&] {
        return llama_kv_block_observe(owners.size(), [&](size_t i) { return owners[i]; },
                                      [&](size_t i) { return mask[i]; });
    };
    if (!check(count(), 5, 4, 1, 2, 1, 0)) return 1;
    // A foreign cell in a dead block must not turn it into a live mixed block.
    owners[1024] = O::foreign;
    if (!check(count(), 5, 4, 1, 1, 1, 0)) return 2;
    // Invalid masks are never treated as visible; report them separately.
    mask[1024] = V::invalid;
    if (!check(count(), 5, 4, 1, 1, 1, 1)) return 3;
    if (!check(llama_kv_block_observe(0, [&](size_t i) { return owners[i]; },
                                      [&](size_t i) { return mask[i]; }), 0, 0, 0, 0, 0, 0)) return 4;
    std::puts("KPF-30 block observer fixtures passed");
    return 0;
}
