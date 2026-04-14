#include "llama-kv-compress.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// RoPE helpers (inverse + averaged forward)
// ---------------------------------------------------------------------------

static void rope_inverse(float * out, const float * x, int d_head, float theta_base, int pos) {
    const int half = d_head / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)d_head);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        out[i]        =  x[i] * cos_a + x[i + half] * sin_a;
        out[i + half] = -x[i] * sin_a + x[i + half] * cos_a;
    }
}

static void rope_avg_cos_sin(float * avg_cos, float * avg_sin, int d_head,
                             float theta_base, int pos_start, int n_future) {
    const int half = d_head / 2;
    for (int i = 0; i < half; i++) {
        float sum_cos = 0.0f, sum_sin = 0.0f;
        float freq = 1.0f / powf(theta_base, (float)(2 * i) / (float)d_head);
        for (int p = pos_start; p < pos_start + n_future; p++) {
            float angle = (float)p * freq;
            sum_cos += cosf(angle);
            sum_sin += sinf(angle);
        }
        avg_cos[i] = sum_cos / (float)n_future;
        avg_sin[i] = sum_sin / (float)n_future;
    }
}

static void rope_apply_avg(float * out, const float * x,
                           const float * avg_cos, const float * avg_sin, int d_head) {
    const int half = d_head / 2;
    for (int i = 0; i < half; i++) {
        out[i]        = x[i] * avg_cos[i] - x[i + half] * avg_sin[i];
        out[i + half] = x[i] * avg_sin[i] + x[i + half] * avg_cos[i];
    }
}

// ---------------------------------------------------------------------------
// KV data extraction (any quantized type → f32)
// ---------------------------------------------------------------------------

static void extract_layer_kv_f32(
        llama_kv_cache * kvc,
        int il, int n_kv, int n_kv_heads, int d_head,
        std::vector<float> & k_out,
        std::vector<float> & v_out) {

    const int total = n_kv * n_kv_heads * d_head;
    k_out.resize(total);
    v_out.resize(total);

    ggml_tensor * k_tensor = kvc->get_k_layer_raw(il);
    ggml_tensor * v_tensor = kvc->get_v_layer_raw(il);

    const size_t row_size = n_kv_heads * d_head;
    const size_t row_bytes_k = ggml_row_size(k_tensor->type, row_size);
    const size_t row_bytes_v = ggml_row_size(v_tensor->type, row_size);

    std::vector<uint8_t> row_buf(std::max(row_bytes_k, row_bytes_v));

    for (int p = 0; p < n_kv; p++) {
        // K
        ggml_backend_tensor_get(k_tensor, row_buf.data(), p * row_bytes_k, row_bytes_k);
        if (k_tensor->type == GGML_TYPE_F32) {
            memcpy(k_out.data() + p * row_size, row_buf.data(), row_size * sizeof(float));
        } else if (k_tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *)row_buf.data();
            for (size_t i = 0; i < row_size; i++) {
                k_out[p * row_size + i] = ggml_fp16_to_fp32(src[i]);
            }
        } else {
            ggml_get_type_traits(k_tensor->type)->to_float(
                row_buf.data(), k_out.data() + p * row_size, row_size);
        }

        // V
        ggml_backend_tensor_get(v_tensor, row_buf.data(), p * row_bytes_v, row_bytes_v);
        if (v_tensor->type == GGML_TYPE_F32) {
            memcpy(v_out.data() + p * row_size, row_buf.data(), row_size * sizeof(float));
        } else if (v_tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *)row_buf.data();
            for (size_t i = 0; i < row_size; i++) {
                v_out[p * row_size + i] = ggml_fp16_to_fp32(src[i]);
            }
        } else {
            ggml_get_type_traits(v_tensor->type)->to_float(
                row_buf.data(), v_out.data() + p * row_size, row_size);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer Expected Attention scoring
// ---------------------------------------------------------------------------

static std::vector<float> score_layer(
        const float * k_data, const float * v_data, const int * positions,
        int n_kv, int n_kv_heads, int d_head, float theta_base,
        int n_q_heads, const llama_kv_compress_params & cfg) {

    const int n_active = n_kv - cfg.n_sink;
    if (n_active <= 0) return std::vector<float>(n_kv, 1.0f);

    const int stride_pos  = n_kv_heads * d_head;
    const int stride_head = d_head;

    // Step 1: Reverse-RoPE K → pre-RoPE K, accumulate mean per head
    std::vector<float> mu(n_kv_heads * d_head, 0.0f);
    std::vector<float> k_prerope(n_active * stride_pos);

    for (int p = cfg.n_sink; p < n_kv; p++) {
        int idx = p - cfg.n_sink;
        for (int h = 0; h < n_kv_heads; h++) {
            const float * k_pos = k_data + p * stride_pos + h * stride_head;
            float * k_out = k_prerope.data() + idx * stride_pos + h * stride_head;
            rope_inverse(k_out, k_pos, d_head, theta_base, positions[p]);
            for (int d = 0; d < d_head; d++) {
                mu[h * d_head + d] += k_out[d];
            }
        }
    }
    for (auto & v : mu) v /= (float)n_active;

    // Step 2: Apply averaged forward RoPE to mean
    std::vector<float> avg_cos(d_head / 2), avg_sin(d_head / 2);
    rope_avg_cos_sin(avg_cos.data(), avg_sin.data(), d_head, theta_base, n_kv, cfg.n_future);

    std::vector<float> mu_rotated(n_kv_heads * d_head);
    for (int h = 0; h < n_kv_heads; h++) {
        rope_apply_avg(mu_rotated.data() + h * d_head, mu.data() + h * d_head,
                       avg_cos.data(), avg_sin.data(), d_head);
    }

    // Step 2b: Covariance (optional)
    std::vector<float> cov;
    if (cfg.use_covariance) {
        cov.resize(n_kv_heads * d_head * d_head, 0.0f);
        for (int p = cfg.n_sink; p < n_kv; p++) {
            int idx = p - cfg.n_sink;
            for (int h = 0; h < n_kv_heads; h++) {
                const float * k_pre = k_prerope.data() + idx * stride_pos + h * stride_head;
                const float * mu_h = mu.data() + h * d_head;
                float * cov_h = cov.data() + h * d_head * d_head;
                for (int i = 0; i < d_head; i++) {
                    float di = k_pre[i] - mu_h[i];
                    for (int j = i; j < d_head; j++) {
                        float dj = k_pre[j] - mu_h[j];
                        float v = di * dj;
                        cov_h[i * d_head + j] += v;
                        if (i != j) cov_h[j * d_head + i] += v;
                    }
                }
            }
        }
        for (auto & v : cov) v /= (float)n_active;
    }

    // Step 3: Score each KV position
    const float scale = 1.0f / sqrtf((float)d_head);
    const int n_groups = n_q_heads / n_kv_heads;

    std::vector<float> scores(n_kv, 0.0f);

    for (int p = cfg.n_sink; p < n_kv; p++) {
        float score = 0.0f;
        for (int h = 0; h < n_kv_heads; h++) {
            const float * k_pos = k_data + p * stride_pos + h * stride_head;
            const float * mu_h = mu_rotated.data() + h * d_head;

            float dot = 0.0f;
            for (int d = 0; d < d_head; d++) dot += k_pos[d] * mu_h[d];

            float cov_term = 0.0f;
            if (cfg.use_covariance) {
                const float * cov_h = cov.data() + h * d_head * d_head;
                for (int i = 0; i < d_head; i++) {
                    float ki = k_pos[i];
                    cov_term += ki * ki * cov_h[i * d_head + i];
                    for (int j = i + 1; j < d_head; j++) {
                        cov_term += 2.0f * ki * k_pos[j] * cov_h[i * d_head + j];
                    }
                }
            }

            score += (dot * scale + 0.5f * cov_term / (float)d_head) * (float)n_groups;
        }

        // V-norm weighting
        float v_norm = 0.0f;
        for (int h = 0; h < n_kv_heads; h++) {
            const float * v_pos = v_data + p * stride_pos + h * stride_head;
            for (int d = 0; d < d_head; d++) v_norm += v_pos[d] * v_pos[d];
        }
        scores[p] = score * sqrtf(v_norm + 1e-8f);
    }

    // Protect sink tokens
    float max_score = *std::max_element(scores.begin() + cfg.n_sink, scores.end());
    for (int p = 0; p < cfg.n_sink && p < n_kv; p++) {
        scores[p] = max_score + 1.0f;
    }

    return scores;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

static llama_kv_cache * get_kv_cache(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    auto * kvc = dynamic_cast<llama_kv_cache *>(mem);
    if (!kvc) {
        auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
        if (hybrid) kvc = hybrid->get_mem_attn();
    }
    return kvc;
}

static float get_theta_base(const llama_model * model) {
    char buf[64] = {};
    const char * keys[] = {
        "qwen3.rope.freq_base", "qwen35moe.rope.freq_base",
        "qwen2.rope.freq_base", "llama.rope.freq_base",
        "phi3.rope.freq_base", "gemma.rope.freq_base",
        nullptr
    };
    for (const char ** k = keys; *k; k++) {
        if (llama_model_meta_val_str(model, *k, buf, sizeof(buf)) > 0) {
            return strtof(buf, nullptr);
        }
    }
    return 1000000.0f;  // fallback
}

std::vector<float> llama_kv_compress_score(
        llama_context * ctx,
        llama_seq_id    seq_id,
        const llama_kv_compress_params & params) {

    llama_memory_t mem = llama_get_memory(ctx);
    llama_kv_cache * kvc = get_kv_cache(ctx);
    if (!kvc) return {};

    const llama_model * model = llama_get_model(ctx);
    const int n_layers   = llama_model_n_layer(model);
    const int n_kv_heads = llama_model_n_head_kv(model);
    const int n_q_heads  = llama_model_n_head(model);
    const int d_head     = llama_model_n_embd(model) / n_q_heads;
    const float theta    = get_theta_base(model);

    llama_pos pos_max = llama_memory_seq_pos_max(mem, seq_id);
    if (pos_max < 0) return {};
    const int n_kv = pos_max + 1;

    std::vector<int> positions(n_kv);
    std::iota(positions.begin(), positions.end(), 0);

    // Find attention layers
    std::vector<int> attn_layers;
    for (int il = 0; il < n_layers; il++) {
        if (kvc->has_layer(il)) attn_layers.push_back(il);
    }
    if (attn_layers.empty()) return {};

    // Multi-layer aggregated scoring with optional layer weights
    // If layer_weights is provided, each layer's scores are multiplied by its weight.
    // If empty, uniform weighting (each layer contributes equally).
    const bool has_weights = !params.layer_weights.empty();
    float weight_sum = 0.0f;

    std::vector<float> scores(n_kv, 0.0f);

    for (size_t li = 0; li < attn_layers.size(); li++) {
        int il = attn_layers[li];
        float w = has_weights ? params.layer_weights[std::min(li, params.layer_weights.size() - 1)] : 1.0f;
        weight_sum += w;

        std::vector<float> k_data, v_data;
        extract_layer_kv_f32(kvc, il, n_kv, n_kv_heads, d_head, k_data, v_data);

        auto layer_scores = score_layer(
            k_data.data(), v_data.data(), positions.data(),
            n_kv, n_kv_heads, d_head, theta, n_q_heads, params);

        for (int p = 0; p < n_kv; p++) scores[p] += layer_scores[p] * w;
    }

    if (weight_sum > 0.0f) {
        for (auto & s : scores) s /= weight_sum;
    }

    return scores;
}

int llama_kv_compress_evict(
        llama_context * ctx,
        llama_seq_id    seq_id,
        const llama_kv_compress_params & params) {

    auto scores = llama_kv_compress_score(ctx, seq_id, params);
    if (scores.empty()) return -1;

    llama_memory_t mem = llama_get_memory(ctx);
    const int n_kv = (int)scores.size();

    // Find eviction threshold
    std::vector<float> sorted(scores.begin() + params.n_sink, scores.end());
    std::sort(sorted.begin(), sorted.end());
    int n_evict_target = (int)(sorted.size() * params.compression_ratio);
    if (n_evict_target <= 0) return 0;
    float threshold = sorted[std::min(n_evict_target, (int)sorted.size() - 1)];

    // Mark positions for eviction
    std::vector<bool> evict(n_kv, false);
    for (int p = params.n_sink; p < n_kv; p++) {
        if (scores[p] < threshold) {
            evict[p] = true;
        }
    }

    // Bulk eviction: merge consecutive positions into ranges to minimize seq_rm calls.
    // At 50% eviction on 32K context, this reduces ~16K calls to ~8K (alternating keep/evict)
    // and far fewer when eviction is spatially clustered (common — low-importance tokens
    // are often contiguous filler/padding regions).
    int evicted = 0;
    int range_start = -1;

    for (int p = 0; p <= n_kv; p++) {
        bool should_evict = (p < n_kv) && evict[p];
        if (should_evict && range_start < 0) {
            range_start = p;
        } else if (!should_evict && range_start >= 0) {
            // Evict range [range_start, p)
            llama_memory_seq_rm(mem, seq_id, range_start, p);
            evicted += (p - range_start);
            range_start = -1;
        }
    }

    // Position gap compaction (opt-in): shift remaining positions to close gaps.
    // Without this, gaps consume context window — new tokens at pos_max+1
    // waste positions that have no KV entries. The shift preserves relative
    // ordering but changes absolute RoPE positions.
    //
    // WARNING: Only safe when the caller manages token positions directly.
    // llama-server tracks positions in its prompt cache independently — using
    // seq_add here would desync the server's state. Default is OFF.
    if (evicted > 0 && params.compact_positions) {
        int gap = 0;
        bool in_gap = false;
        int gap_start = -1;

        for (int p = params.n_sink; p < n_kv; p++) {
            if (evict[p]) {
                if (!in_gap) {
                    gap_start = p;
                    in_gap = true;
                }
            } else {
                if (in_gap) {
                    gap += (p - gap_start);
                    in_gap = false;
                }
                // Shift this position left by accumulated gap
                // (handled by a single seq_add call after the scan)
            }
        }
        // Final gap at end
        if (in_gap) {
            gap += (n_kv - gap_start);
        }

        // Apply shift: for each contiguous kept range after a gap,
        // shift it left by the gap size at that point.
        // We do this in reverse order of gaps to avoid shifting already-shifted ranges.
        // Simpler approach: rebuild position sequence via seq_rm of the whole range
        // then seq_add. But seq_add shifts ALL positions >= p0, which handles it in one call
        // per gap.
        gap = 0;
        for (int p = params.n_sink; p < n_kv; p++) {
            if (evict[p]) {
                gap++;
            } else if (gap > 0) {
                // Found a kept position after a gap — shift everything from here onward
                llama_memory_seq_add(mem, seq_id, p, -1, -gap);
                // After shifting, positions [p, ...) become [p-gap, ...).
                // Adjust our scan to account for the shift.
                // The remaining positions are now at [p-gap, ...) so we continue
                // scanning but the evict[] array still uses original indices.
                // Since seq_add shifted ALL positions >= p, subsequent kept positions
                // are already shifted. We only need one seq_add call per contiguous gap.
                gap = 0;
            }
        }
    }

    return evicted;
}
