#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t k_dim = 256;
constexpr int64_t m_dim = 32;
constexpr int64_t n_tokens = 2;
constexpr int64_t n_mats = 4;
constexpr int64_t n_used = 4;

const std::vector<int32_t> k_valid_ids = {0, 1, 2, 3, 3, 2, 1, 0};
const std::vector<int32_t> k_invalid_ids = {-1, (int32_t) n_mats, 0, 1, 2, 3, -1, (int32_t) n_mats};

bool check_rows(const std::vector<float> & output, const std::vector<int32_t> & ids) {
    bool saw_valid_nonzero = false;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id = 0; id < n_used; ++id) {
            const int32_t expert = ids[token*n_used + id];
            const float * row = output.data() + (token*n_used + id)*m_dim;
            if (expert < 0 || expert >= n_mats) {
                for (int64_t i = 0; i < m_dim; ++i) {
                    if (row[i] != 0.0f) {
                        fprintf(stderr, "invalid row (%lld, %lld) was not zero at %lld: %g\n",
                                (long long) token, (long long) id, (long long) i, row[i]);
                        return false;
                    }
                }
                continue;
            }
            for (int64_t i = 0; i < m_dim; ++i) {
                if (!std::isfinite(row[i])) {
                    fprintf(stderr, "valid row (%lld, %lld) is non-finite at %lld\n",
                            (long long) token, (long long) id, (long long) i);
                    return false;
                }
                saw_valid_nonzero = saw_valid_nonzero || std::fabs(row[i]) > 1e-8f;
            }
        }
    }
    if (!saw_valid_nonzero) {
        fprintf(stderr, "all valid expert rows were zero\n");
        return false;
    }
    return true;
}

bool run_case(const std::vector<int32_t> & expert_ids, bool use_ref, std::vector<float> * output_data) {
    ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) return false;

    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_IQ2_XXS, k_dim, m_dim, n_mats);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_dim, n_used, n_tokens);
    ggml_tensor * output = ggml_mul_mat_id(ctx.get(), weights, activations, ids);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    if (!backend) return false;
    ggml_backend_cpu_set_n_threads(backend.get(), 96);
    ggml_backend_cpu_set_use_ref(backend.get(), use_ref);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) return false;

    std::vector<float> weight_f32((size_t) k_dim*m_dim*n_mats);
    for (size_t i = 0; i < weight_f32.size(); ++i) {
        weight_f32[i] = 0.4f*std::sin((float) i*0.013f) + 0.1f*std::cos((float) i*0.007f);
    }
    std::vector<uint8_t> weight_q(ggml_nbytes(weights));
    std::vector<float> imatrix(k_dim, 1.0f);
    const size_t quantized = ggml_quantize_chunk(
        GGML_TYPE_IQ2_XXS, weight_f32.data(), weight_q.data(), 0, m_dim*n_mats, k_dim, imatrix.data());
    if (quantized != weight_q.size()) return false;
    ggml_backend_tensor_set(weights, weight_q.data(), 0, weight_q.size());

    std::vector<float> activation_data((size_t) k_dim*n_used*n_tokens);
    for (size_t i = 0; i < activation_data.size(); ++i) {
        activation_data[i] = 0.25f*std::sin((float) i*0.017f) - 0.2f*std::cos((float) i*0.011f);
    }
    ggml_backend_tensor_set(activations, activation_data.data(), 0, activation_data.size()*sizeof(float));
    ggml_backend_tensor_set(ids, expert_ids.data(), 0, expert_ids.size()*sizeof(int32_t));

    output_data->assign((size_t) m_dim*n_used*n_tokens, 7.0f);
    ggml_backend_tensor_set(output, output_data->data(), 0, output_data->size()*sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_tensor_get(output, output_data->data(), 0, output_data->size()*sizeof(float));
    return check_rows(*output_data, expert_ids);
}

bool compare_outputs(const std::vector<float> & fast, const std::vector<float> & reference) {
    constexpr float abs_tolerance = 1e-4f;
    constexpr float rel_tolerance = 1e-3f;
    constexpr double normalized_squared_error_tolerance = 1e-5;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;
    for (size_t i = 0; i < fast.size(); ++i) {
        if (!std::isfinite(fast[i]) || !std::isfinite(reference[i])) return false;
        const float diff = std::fabs(fast[i] - reference[i]);
        const float rel = diff / std::max(std::fabs(reference[i]), 1e-6f);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        sum_diff_sq += (double) diff*diff;
        sum_ref_sq += (double) reference[i]*reference[i];
        const float limit = abs_tolerance + rel_tolerance*std::fabs(reference[i]);
        if (diff > limit) {
            fprintf(stderr, "IQK/reference mismatch at %zu: fast=%g ref=%g diff=%g limit=%g\n",
                    i, fast[i], reference[i], diff, limit);
            return false;
        }
    }
    const double normalized_squared_error = sum_diff_sq / std::max(sum_ref_sq, 1e-30);
    fprintf(stderr, "IQK/reference telemetry: max_abs=%g max_rel=%g normalized_squared_error=%g "
            "elementwise_limit=%g+%g*abs(ref) normalized_squared_error_limit=%g\n",
            max_abs, max_rel, normalized_squared_error,
            abs_tolerance, rel_tolerance, normalized_squared_error_tolerance);
    if (normalized_squared_error > normalized_squared_error_tolerance) {
        fprintf(stderr, "IQK/reference normalized squared error exceeds limit\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const bool invalid_native = argc == 2 && strcmp(argv[1], "--invalid-native") == 0;
    const bool invalid_reference = argc == 2 && strcmp(argv[1], "--invalid-reference") == 0;
    const bool invalid_iqk = argc == 2 && strcmp(argv[1], "--invalid-iqk") == 0;
    if (argc > 2 || (argc == 2 && !invalid_native && !invalid_reference && !invalid_iqk)) {
        fprintf(stderr, "usage: %s [--invalid-iqk|--invalid-native|--invalid-reference]\n", argv[0]);
        return 2;
    }

    if (setenv("GGML_IQK", invalid_native ? "0" : "1", 1) != 0) return 2;
    std::vector<float> output;
    if (invalid_iqk || invalid_native || invalid_reference) {
        if (!run_case(k_invalid_ids, invalid_reference, &output)) return 1;
        fprintf(stderr, "MUL_MAT_ID invalid-ID %s passed\n",
                invalid_iqk ? "IQK" : invalid_reference ? "reference fallback" : "native fallback");
        return 0;
    }

    std::vector<float> fast;
    std::vector<float> reference;
    if (!run_case(k_valid_ids, false, &fast) || !run_case(k_valid_ids, true, &reference)) return 1;
    if (!compare_outputs(fast, reference)) return 1;
    fprintf(stderr, "IQK MUL_MAT_ID numerical comparison passed\n");
    return 0;
}
