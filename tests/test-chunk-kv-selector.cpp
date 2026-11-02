#include "../tools/server/chunk-kv-selector.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using chunk_kv_selector::select;
int main() {
    // Mandatory head, a high-score later whole block, a short final block and
    // absolute non-aligned offset: groups are chunk-relative, never global.
    constexpr int n = 200;
    std::vector<float> fresh(n * 2, 0.0f), cached(n * 2, 0.0f);
    for (int t = 128; t < 192; ++t) fresh[t*2] = 2.0f;
    auto a = select(17, n, 2, 0.4, fresh, cached);
    assert(a.target_tokens == 80 && a.selected_tokens == 128 && a.runs.size() == 2);
    assert(a.runs[0].begin == 17 && a.runs[0].end == 81);
    assert(a.runs[1].begin == 145 && a.runs[1].end == 209);
    assert(a.target_r == 0.4 && a.realized_r == 0.64);
    // Exact-score ties choose lower chunk start; adjacent head/block merge.
    std::fill(fresh.begin(), fresh.end(), 0.0f);
    a = select(17, n, 2, 0.4, fresh, cached);
    assert(a.runs.size() == 1 && a.runs[0].begin == 17 && a.runs[0].end == 145);
    // Global alignment would end the first post-head block at 128, not 145.
    assert(a.runs[0].end != 128);
    // Sum aggregation favors 64 entries of score 1 over an 8-token outlier
    // block of per-token score 4. Mean and max alternatives would select it.
    for (int t = 64; t < 128; ++t) fresh[t*2] = 1;
    for (int t = 192; t < 200; ++t) fresh[t*2] = 2;
    a = select(0, n, 2, 0.4, fresh, cached);
    assert(a.runs.size() == 1 && a.runs[0].end == 128);
    auto control_score = [&](int begin, int end, bool use_max) {
        double value = 0;
        for (int t = begin; t < end; ++t) {
            double token = 0;
            for (int d = 0; d < 2; ++d) {
                const double delta = double(fresh[t*2+d]) - double(cached[t*2+d]);
                token += delta*delta;
            }
            value = use_max ? std::max(value, token) : value + token;
        }
        return use_max ? value : value / (end - begin);
    };
    assert(control_score(64, 128, false) < control_score(192, 200, false)); // mean
    assert(control_score(64, 128, true) < control_score(192, 200, true));   // max
    // r=0 still pays mandatory head; r=1 selects every token, including tail.
    a = select(0, n, 2, 0, fresh, cached); assert(a.selected_tokens == 64);
    a = select(0, n, 2, 1, fresh, cached); assert(a.selected_tokens == n && a.runs.size() == 1 && a.runs[0].end == n);
    std::vector<float> short_k(63, 0);
    a = select(0, 63, 1, 0.5, short_k, short_k); assert(a.selected_tokens == 63 && a.target_tokens == 32);
    auto rejects = [](auto f) { bool thrown = false; try { f(); } catch (const std::invalid_argument &) { thrown = true; } assert(thrown); };
    rejects([&] { select(0, n, 2, NAN, fresh, cached); });
    rejects([&] { select(0, n, 2, 1.1, fresh, cached); });
    rejects([&] { select(INT32_MAX, n, 2, 0.5, fresh, cached); });
    rejects([&] { select(0, n, 1, 0.5, fresh, cached); });
    fresh[0] = std::numeric_limits<float>::infinity();
    rejects([&] { select(0, n, 2, 0.5, fresh, cached); });
    return 0;
}
