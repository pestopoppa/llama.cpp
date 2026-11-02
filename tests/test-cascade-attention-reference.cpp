#include "cascade-attention-reference.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static bool near(double a, long double b, double tolerance = 1e-11) {
    return std::abs(static_cast<long double>(a) - b) <= tolerance * (1 + std::abs(b));
}

// Independent, high-precision full-attention softmax, not the partial merge.
static bool check_case(const std::vector<double> & scores, const std::vector<double> & values,
                       size_t dim, size_t split) {
    if (split > scores.size()) return false;
    std::vector<double> ts(scores.begin(), scores.begin() + split), ss(scores.begin() + split, scores.end());
    std::vector<double> tv(values.begin(), values.begin() + split * dim), sv(values.begin() + split * dim, values.end());
    cascade_reference_partial trunk, suffix, merged;
    if (!cascade_reference_partition(ts, tv, dim, trunk) || !cascade_reference_partition(ss, sv, dim, suffix) ||
        !cascade_reference_merge(trunk, suffix, merged)) return false;
    long double m = -std::numeric_limits<long double>::infinity();
    for (auto x : scores) m = std::max(m, static_cast<long double>(x));
    if (!std::isfinite(m)) return !merged.has_value && merged.log_sum_exp == -std::numeric_limits<double>::infinity();
    long double denom = 0;
    std::vector<long double> numerator(dim, 0);
    for (size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) continue;
        const long double weight = std::exp(static_cast<long double>(scores[i]) - m);
        denom += weight;
        for (size_t d = 0; d < dim; ++d) numerator[d] += weight * values[i * dim + d];
    }
    if (!merged.has_value || !near(merged.log_sum_exp, m + std::log(denom))) return false;
    for (size_t d = 0; d < dim; ++d) if (!near(merged.output[d], numerator[d] / denom)) return false;
    // Swapping the two parts must preserve the result.
    cascade_reference_partial reverse;
    if (!cascade_reference_merge(suffix, trunk, reverse)) return false;
    for (size_t d = 0; d < dim; ++d) if (!near(reverse.output[d], merged.output[d])) return false;
    return true;
}

int main() {
    const double ni = -std::numeric_limits<double>::infinity();
    // Empty trunk/suffix, uneven partitions and extreme logits across partitions.
    const std::vector<double> scores{10000, 9999, ni, -10000, 9998, 10001, -1};
    const std::vector<double> vals{1,2,3, 4,5,6, 7,8,9, -3,1,2, 11,12,13, 17,19,23, -4,7,1};
    for (size_t split = 0; split <= scores.size(); ++split) if (!check_case(scores, vals, 3, split)) return 1;
    // All masked and empty inputs are valid but do not claim a defined output.
    if (!check_case({ni, ni}, {1, 2, 3, 4}, 2, 1) || !check_case({}, {}, 2, 0)) return 2;
    // Mixed shared/private rows and unequal lengths, with all possible trunk sizes.
    for (size_t n = 1; n <= 33; ++n) {
        std::vector<double> s(n), v(n * 5);
        for (size_t i = 0; i < n; ++i) {
            s[i] = i % 5 == 0 ? ni : static_cast<double>(int(i % 11) - 5) * 7;
            for (size_t d = 0; d < 5; ++d) v[i * 5 + d] = static_cast<double>(int(i * (d + 1)) - 9);
        }
        for (size_t split = 0; split <= n; ++split) if (!check_case(s, v, 5, split)) return 3;
    }
    cascade_reference_partial part, empty, merged;
    if (cascade_reference_partition({0}, {1, 2}, 3, part) || cascade_reference_partition({0}, {1}, 0, part) ||
        cascade_reference_partition({std::numeric_limits<double>::quiet_NaN()}, {1}, 1, part) ||
        cascade_reference_partition({std::numeric_limits<double>::infinity()}, {1}, 1, part) ||
        cascade_reference_partition({0}, {std::numeric_limits<double>::quiet_NaN()}, 1, part)) return 4;
    if (!cascade_reference_partition({}, {}, 1, part) || !cascade_reference_partition({}, {}, 2, empty) ||
        cascade_reference_merge(part, empty, merged)) return 5;
    part.has_value = true; part.log_sum_exp = ni;
    if (cascade_reference_merge(part, part, merged)) return 6;
    // Finite weighted means must not overflow and in-place merges are allowed.
    const double huge = std::numeric_limits<double>::max() / 1.5;
    if (!cascade_reference_partition({0, 0}, {huge, huge}, 1, part) || !near(part.output[0], huge)) return 7;
    if (!cascade_reference_partition({0}, {huge}, 1, empty) ||
        !cascade_reference_merge(part, empty, part) || !near(part.output[0], huge)) return 8;
    std::puts("cascade attention reference parity passed");
    return 0;
}
