#pragma once

// Test-only oracle for the P5 shared-trunk/private-suffix decomposition.
// No serving graph or backend dispatch calls this code.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

struct cascade_reference_partial {
    std::vector<double> output;
    double log_sum_exp = -std::numeric_limits<double>::infinity();
    bool has_value = false;
};

// scores: one logit per key; values: row-major [key, value dimension].
// -infinity means masked. Empty and all-masked partitions are valid and
// produce no value. NaN, +infinity, malformed shapes and non-finite visible
// values refuse rather than manufacturing an answer.
inline bool cascade_reference_partition(const std::vector<double> & scores,
                                        const std::vector<double> & values,
                                        size_t dimension,
                                        cascade_reference_partial & result) {
    result = {};
    if (dimension == 0 || scores.size() > values.max_size() / dimension ||
        values.size() != scores.size() * dimension) {
        return false;
    }
    result.output.assign(dimension, 0.0);
    double max_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < scores.size(); ++i) {
        const double score = scores[i];
        if (std::isnan(score) || score == std::numeric_limits<double>::infinity()) return false;
        if (score == -std::numeric_limits<double>::infinity()) continue;
        if (!std::isfinite(score)) return false;
        for (size_t d = 0; d < dimension; ++d) {
            if (!std::isfinite(values[i * dimension + d])) return false;
        }
        max_score = std::max(max_score, score);
    }
    if (max_score == -std::numeric_limits<double>::infinity()) return true;
    double denominator = 0;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] == -std::numeric_limits<double>::infinity()) continue;
        const double weight = std::exp(scores[i] - max_score);
        denominator += weight;
    }
    if (!(denominator > 0) || !std::isfinite(denominator)) return false;
    // Normalize before accumulating values to avoid overflowing an otherwise
    // finite weighted mean when individual finite values are very large.
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] == -std::numeric_limits<double>::infinity()) continue;
        const double weight = std::exp(scores[i] - max_score) / denominator;
        for (size_t d = 0; d < dimension; ++d) result.output[d] += weight * values[i * dimension + d];
    }
    for (double v : result.output) if (!std::isfinite(v)) return false;
    result.log_sum_exp = max_score + std::log(denominator);
    result.has_value = true;
    return std::isfinite(result.log_sum_exp);
}

// Merge independently normalized attention outputs using their log-sum-exp.
// An all-masked total has no defined attention output: has_value=false and a
// zero-filled vector are explicit, rather than implying softmax(−inf,...).
inline bool cascade_reference_merge(const cascade_reference_partial & trunk,
                                    const cascade_reference_partial & suffix,
                                    cascade_reference_partial & result) {
    // Support in-place accumulation when either input is also the destination.
    if (&result == &trunk || &result == &suffix) {
        const auto trunk_copy = trunk;
        const auto suffix_copy = suffix;
        return cascade_reference_merge(trunk_copy, suffix_copy, result);
    }
    result = {};
    if (trunk.output.empty() || trunk.output.size() != suffix.output.size()) return false;
    const auto valid = [](const cascade_reference_partial & part) {
        if (part.has_value ? !std::isfinite(part.log_sum_exp) :
            part.log_sum_exp != -std::numeric_limits<double>::infinity()) return false;
        return std::all_of(part.output.begin(), part.output.end(), [](double x) { return std::isfinite(x); });
    };
    if (!valid(trunk) || !valid(suffix)) return false;
    result.output.assign(trunk.output.size(), 0.0);
    if (!trunk.has_value && !suffix.has_value) return true;
    if (!trunk.has_value) { result = suffix; return true; }
    if (!suffix.has_value) { result = trunk; return true; }
    const double m = std::max(trunk.log_sum_exp, suffix.log_sum_exp);
    const double a = std::exp(trunk.log_sum_exp - m);
    const double b = std::exp(suffix.log_sum_exp - m);
    const double sum = a + b;
    for (size_t d = 0; d < result.output.size(); ++d) {
        result.output[d] = (a / sum) * trunk.output[d] + (b / sum) * suffix.output[d];
        if (!std::isfinite(result.output[d])) return false;
    }
    result.log_sum_exp = m + std::log(sum);
    result.has_value = true;
    return std::isfinite(result.log_sum_exp);
}
