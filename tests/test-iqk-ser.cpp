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

bool check_rows(const std::vector<float> & output, const std::vector<int32_t> & ids, bool expect_invalid) {
    bool saw_valid_nonzero = false;

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id = 0; id < n_used; ++id) {
            const int32_t expert = ids[token*n_used + id];
            const float * row = output.data() + (token*n_used + id)*m_dim;

            if (expert < 0 || expert >= n_mats) {
                if (!expect_invalid) {
                    fprintf(stderr, "unexpected invalid expert id %d\n", expert);
                    return false;
                }
                for (int64_t i = 0; i < m_dim; ++i) {
                    if (row[i] != 0.0f) {
                        fprintf(stderr, "invalid expert row (%lld, %lld) was not zero at %lld: %g\n",
                                (long long) token, (long long) id, (long long) i, row[i]);
                        return false;
                    }
                }
                continue;
            }

            for (int64_t i = 0; i < m_dim; ++i) {
                if (!std::isfinite(row[i])) {
                    fprintf(stderr, "valid expert row (%lld, %lld) is non-finite at %lld\n",
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

} // namespace

int main(int argc, char ** argv) {
    const bool control = argc == 2 && strcmp(argv[1], "--control") == 0;
    if (argc > 2 || (argc == 2 && !control)) {
        fprintf(stderr, "usage: %s [--control]\n", argv[0]);
        return 2;
    }

    if (setenv("GGML_IQK", control ? "0" : "1", 1) != 0) {
        perror("setenv");
        return 2;
    }

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        fprintf(stderr, "failed to create ggml context\n");
        return 2;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_IQ2_XXS, k_dim, m_dim, n_mats);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k_dim, n_used, n_tokens);
    ggml_tensor * output = ggml_mul_mat_id(ctx.get(), weights, activations, ids);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    if (!backend) {
        fprintf(stderr, "failed to create CPU backend\n");
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend.get(), 96);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) {
        fprintf(stderr, "failed to allocate backend tensors\n");
        return 2;
    }

    std::vector<float> weight_f32((size_t) k_dim*m_dim*n_mats);
    for (size_t i = 0; i < weight_f32.size(); ++i) {
        weight_f32[i] = 0.4f*std::sin((float) i*0.013f) + 0.1f*std::cos((float) i*0.007f);
    }
    std::vector<uint8_t> weight_q(ggml_nbytes(weights));
    std::vector<float> imatrix(k_dim, 1.0f);
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_IQ2_XXS, weight_f32.data(), weight_q.data(), 0, m_dim*n_mats, k_dim, imatrix.data());
    if (quantized != weight_q.size()) {
        fprintf(stderr, "quantized weight size mismatch: %zu != %zu\n", quantized, weight_q.size());
        return 2;
    }
    ggml_backend_tensor_set(weights, weight_q.data(), 0, weight_q.size());

    std::vector<float> activation_data((size_t) k_dim*n_used*n_tokens);
    for (size_t i = 0; i < activation_data.size(); ++i) {
        activation_data[i] = 0.25f*std::sin((float) i*0.017f) - 0.2f*std::cos((float) i*0.011f);
    }
    ggml_backend_tensor_set(activations, activation_data.data(), 0, activation_data.size()*sizeof(float));

    const std::vector<int32_t> expert_ids = control
        ? std::vector<int32_t>{0, 1, 2, 3, 3, 2, 1, 0}
        : std::vector<int32_t>{-1, (int32_t) n_mats, 0, 1, 2, 3, -1, (int32_t) n_mats};
    ggml_backend_tensor_set(ids, expert_ids.data(), 0, expert_ids.size()*sizeof(int32_t));

    std::vector<float> output_data((size_t) m_dim*n_used*n_tokens, 7.0f);
    ggml_backend_tensor_set(output, output_data.data(), 0, output_data.size()*sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        return 1;
    }

    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size()*sizeof(float));
    if (!check_rows(output_data, expert_ids, !control)) {
        return 1;
    }

    fprintf(stderr, "IQK MUL_MAT_ID %s regression passed\n", control ? "valid-ID control" : "SER invalid-ID");
    return 0;
}
