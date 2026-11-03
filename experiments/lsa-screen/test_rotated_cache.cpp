// Mode-free ggml fixture for the C5 quantized KV reference path.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr int D = 1024; // four 256-wide KV heads; mixes rotary and free coordinates
static constexpr int N = 2;

static void hadamard(std::vector<float> & x) {
    for (int width = 1; width < D; width *= 2) {
        for (int base = 0; base < D; base += 2*width) {
            for (int j = 0; j < width; ++j) {
                const float a = x[base+j], b = x[base+j+width];
                x[base+j] = (a+b)*float(M_SQRT1_2);
                x[base+j+width] = (a-b)*float(M_SQRT1_2);
            }
        }
    }
}

int main() {
    ggml_context * ctx = ggml_init({64*1024*1024, nullptr, false});
    auto * rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, D);
    auto * query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    auto * key_q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, D, N);
    auto * value_q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, D, N);
    // Check ggml's row/column convention with an orthogonal, non-symmetric R.
    // Forward R and inverse R^T must differ on this input.
    auto * orientation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    auto * basis = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * orientation_forward = ggml_mul_mat(ctx, orientation, basis);
    auto * orientation_inverse = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, orientation)), basis);
    auto * matrix = static_cast<float *>(orientation->data);
    std::fill_n(matrix, 16, 0.0f);
    matrix[0*4+1] = 1; matrix[1*4+2] = 1; matrix[2*4+3] = 1; matrix[3*4+0] = -1;
    auto * unit = static_cast<float *>(basis->data);
    unit[0] = 1; unit[1] = 2; unit[2] = 3; unit[3] = 4;
    auto * r = static_cast<float *>(rot->data);
    for (int col = 0; col < D; ++col) for (int row = 0; row < D; ++row) {
        const int bits = __builtin_popcount(unsigned(col & row));
        r[col*D+row] = (bits & 1 ? -1.0f : 1.0f)/std::sqrt(float(D));
    }
    std::vector<float> logical_k(D*N), physical_k(D*N), physical_v(D*N);
    for (int t = 0; t < N; ++t) {
        auto * q = static_cast<float *>(query->data) + t*D;
        for (int d = 0; d < D; ++d) {
            // RoPE has already mixed the first 64 dimensions of each head.
            q[d] = 0.19f*std::sin(0.017f*d + 0.3f*t) + 0.013f*(d%19);
            logical_k[t*D+d] = 0.17f*std::cos(0.013f*d + 0.5f*t) + 0.009f*(d%23);
            physical_v[t*D+d] = 0.11f*std::sin(0.011f*d + 0.7f*t);
        }
        std::copy_n(logical_k.data()+t*D, D, physical_k.data()+t*D);
        std::vector<float> k(physical_k.data()+t*D, physical_k.data()+(t+1)*D);
        std::vector<float> v(physical_v.data()+t*D, physical_v.data()+(t+1)*D);
        hadamard(k); hadamard(v);
        std::copy(k.begin(), k.end(), physical_k.begin()+t*D);
        std::copy(v.begin(), v.end(), physical_v.begin()+t*D);
    }
    ggml_quantize_chunk(GGML_TYPE_Q8_0, physical_k.data(), key_q8->data, 0, N, D, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, physical_v.data(), value_q8->data, 0, N, D, nullptr);
    auto * q_phys = ggml_mul_mat(ctx, rot, query);
    auto * k_deq = ggml_cast(ctx, key_q8, GGML_TYPE_F32);
    auto * v_deq = ggml_cast(ctx, value_q8, GGML_TYPE_F32);
    auto * k_logical = ggml_mul_mat(ctx, rot, k_deq); // H^-1 = H
    auto * v_logical = ggml_mul_mat(ctx, rot, v_deq);
    auto * scores = ggml_mul_mat(ctx, k_deq, q_phys); // all keys in physical coordinates
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, scores);
    ggml_build_forward_expand(graph, k_logical);
    ggml_build_forward_expand(graph, v_logical);
    ggml_build_forward_expand(graph, orientation_forward);
    ggml_build_forward_expand(graph, orientation_inverse);
    if (ggml_graph_compute_with_ctx(ctx, graph, 2) != GGML_STATUS_SUCCESS) return 2;
    const auto * kp = static_cast<const float *>(k_deq->data);
    const auto * vp = static_cast<const float *>(v_deq->data);
    const auto * qp = static_cast<const float *>(q_phys->data);
    const auto * got_score = static_cast<const float *>(scores->data);
    const auto * got_logical = static_cast<const float *>(k_logical->data);
    const auto * got_value = static_cast<const float *>(v_logical->data);
    const auto * forward = static_cast<const float *>(orientation_forward->data);
    const auto * inverse = static_cast<const float *>(orientation_inverse->data);
    const bool orientation_ok = forward[0] == 2 && forward[1] == 3 && forward[2] == 4 && forward[3] == -1 &&
                                inverse[0] == -4 && inverse[1] == 1 && inverse[2] == 2 && inverse[3] == 3;
    float max_score = 0, max_key = 0, max_value = 0, max_project = 0, physical_leak = 0;
    for (int t = 0; t < N; ++t) {
        std::vector<float> key(kp+t*D, kp+(t+1)*D);
        std::vector<float> value(vp+t*D, vp+(t+1)*D);
        hadamard(key); hadamard(value);
        for (int d = 0; d < D; ++d) {
            max_key = std::max(max_key, std::abs(key[d]-got_logical[t*D+d]));
            max_value = std::max(max_value, std::abs(value[d]-got_value[t*D+d]));
            if (d%256 >= 64) physical_leak = std::max(physical_leak, std::abs(kp[t*D+d]-logical_k[t*D+d]));
        }
        // Project only the logical free coordinates, as the PCA/distilled maps do.
        // The independent oracle starts before rotation and quantization.
        for (int out = 0; out < 64; ++out) {
            double oracle = 0, recovered = 0;
            for (int head = 0; head < 4; ++head) for (int free = 64; free < 256; ++free) {
                const int d = head*256 + free;
                const double weight = std::sin(0.003*(out+1)*(d+1))/std::sqrt(768.0);
                oracle += logical_k[t*D+d]*weight;
                recovered += got_logical[t*D+d]*weight;
            }
            max_project = std::max(max_project, float(std::abs(oracle-recovered)));
        }
        for (int q = 0; q < N; ++q) {
            float expected = 0;
            for (int d = 0; d < D; ++d) expected += kp[t*D+d]*qp[q*D+d];
            max_score = std::max(max_score, std::abs(expected-got_score[q*N+t]));
        }
    }
    std::printf("all-key score error %.6g; inverse K %.6g; inverse V %.6g; logical projection vs prequant oracle %.6g; physical/logical free-coordinate gap %.6g\n",
                max_score, max_key, max_value, max_project, physical_leak);
    const bool ok = orientation_ok && max_score < 2e-4f && max_key < 2e-5f && max_value < 2e-5f && max_project < 0.01f && physical_leak > 0.1f;
    ggml_free(ctx);
    return ok ? 0 : 3;
}
