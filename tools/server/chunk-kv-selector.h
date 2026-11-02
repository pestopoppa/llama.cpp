#pragma once

// KPF-62.3 Stage A selector: input K rows must be in the same positional frame.
// This is a quality instrument, not a partial-layer inference or speed optimization.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace chunk_kv_selector {

struct run { int32_t begin; int32_t end; }; // absolute, half-open
struct result {
    std::vector<run> runs;
    int32_t target_tokens;
    int32_t selected_tokens;
    double target_r;
    double realized_r;
};

// Each row is one token and `dim` check-layer K values. Always recompute the
// first min(64,n) tokens. Following groups are 64-token, chunk-relative blocks;
// score is the SUM of token-wise squared K deviations. Highest score wins;
// equal scores prefer lower start position. Whole blocks are added until the
// target token count is reached; this can overshoot the target in the final block.
inline result select(int32_t offset, int32_t n, int32_t dim, double target_r,
                     const std::vector<float> & fresh, const std::vector<float> & cached) {
    constexpr int32_t width = 64;
    if (offset < 0 || n <= 0 || dim <= 0 || !std::isfinite(target_r) || target_r < 0 || target_r > 1 ||
        int64_t(offset) + n > std::numeric_limits<int32_t>::max() ||
        size_t(n) > std::numeric_limits<size_t>::max()/size_t(dim) ||
        fresh.size() != size_t(n)*size_t(dim) || cached.size() != fresh.size()) {
        throw std::invalid_argument("invalid chunk selector input");
    }
    struct block { int32_t begin; int32_t end; double score; };
    const int32_t head = std::min(width, n);
    std::vector<double> token_score(static_cast<size_t>(n), 0.0);
    for (int32_t t = 0; t < n; ++t) {
        for (int32_t d = 0; d < dim; ++d) {
            const size_t i = size_t(t)*size_t(dim) + size_t(d);
            if (!std::isfinite(fresh[i]) || !std::isfinite(cached[i]))
                throw std::invalid_argument("nonfinite K value");
            const double delta = double(fresh[i]) - double(cached[i]);
            token_score[size_t(t)] += delta*delta;
        }
        if (!std::isfinite(token_score[size_t(t)]))
            throw std::invalid_argument("nonfinite token score");
    }
    std::vector<block> candidates;
    for (int32_t begin = head; begin < n;) {
        const int32_t end = int32_t(std::min<int64_t>(n, int64_t(begin) + width));
        double score = 0;
        for (int32_t t = begin; t < end; ++t) score += token_score[size_t(t)];
        if (!std::isfinite(score)) throw std::invalid_argument("nonfinite block score");
        candidates.push_back({begin, end, score});
        begin = end;
    }
    std::sort(candidates.begin(), candidates.end(), [](const block & a, const block & b) {
        return a.score != b.score ? a.score > b.score : a.begin < b.begin;
    });
    const int32_t target = int32_t(std::ceil(target_r * double(n)));
    int32_t selected = head;
    std::vector<run> runs = {{offset, offset + head}};
    for (const auto & c : candidates) {
        if (selected >= target) break;
        runs.push_back({offset + c.begin, offset + c.end});
        selected += c.end - c.begin;
    }
    std::sort(runs.begin(), runs.end(), [](const run & a, const run & b) { return a.begin < b.begin; });
    std::vector<run> merged;
    for (const auto & r : runs) {
        if (!merged.empty() && merged.back().end == r.begin) merged.back().end = r.end;
        else merged.push_back(r);
    }
    return {std::move(merged), target, selected, target_r, double(selected)/n};
}
}
