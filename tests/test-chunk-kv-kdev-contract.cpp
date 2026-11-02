#include "../tools/server/chunk-kv-selector.h"
#include "../tools/server/server-chunk-recompute.h"
#include <cassert>
#include <vector>

int main() {
    using namespace server_chunk;
    const span sp{17, 200};
    std::vector<float> fresh(400, 0), cached(400, 0);
    for (int i = 128; i < 192; ++i) fresh[i*2] = 2;
    const auto selected = chunk_kv_selector::select(sp.offset, sp.n_tokens, 2, .4, fresh, cached);
    std::vector<run> runs;
    for (const auto & r : selected.runs) runs.push_back({r.begin, r.end});
    assert(valid_kdev_runs(sp, runs));
    assert(runs_tokens(runs) == selected.selected_tokens && selected.selected_tokens == 128);
    assert(!valid_kdev_runs(sp, {}));
    assert(!valid_kdev_runs(sp, {{17, 80}})); // less than mandatory 64
    assert(!valid_kdev_runs(sp, {{17, 82}})); // non-aligned group
    assert(!valid_kdev_runs(sp, {{17, 81}, {80, 100}})); // overlap
    assert(!valid_kdev_runs(sp, {{17, 81}, {145, 218}})); // beyond chunk
    assert(!valid_kdev_runs(sp, {{18, 82}})); // missing head
    assert(!valid_kdev_runs({INT32_MAX, 1}, {{0, 1}})); // overflow
    // A missing oracle must refuse before accessing or mutating a context.
    assemble_stats st; int32_t done = -1;
    auto no_blob = [](int32_t) { return blob_ref{}; };
    assert(assemble(nullptr, nullptr, {}, 0, 0, 0, 1, SELECT_KDEV, 0, no_blob, st, &done) == -1);
    assert(done == 0);
    return 0;
}
