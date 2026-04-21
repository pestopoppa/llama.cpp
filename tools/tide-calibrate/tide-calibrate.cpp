// TIDE Calibration Tool — Extract per-layer hidden states for router training
//
// Two modes:
//   --full-states: Dump raw hidden states per checkpoint layer (large files, ~84GB per model)
//                  Uses N separate forward passes per sample (one per checkpoint)
//   (default):     On-the-fly cosine similarity via cb_eval callback (compact, ~MB)
//                  Uses ONE forward pass per sample — captures "l_out" at checkpoints
//
// Usage:
//   llama-tide-calibrate -m model.gguf --output-dir ./tide_data \
//       --n-samples 1000 --seq-len 512 --checkpoint-interval 4 -t 96
//
//   llama-tide-calibrate -m model.gguf --output-dir ./tide_data --full-states \
//       --n-samples 2000 --seq-len 512 --checkpoint-interval 4 -t 96

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <filesystem>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>

namespace fs = std::filesystem;

// --- Callback state for on-the-fly hidden state capture ---
struct capture_state {
    std::set<int> checkpoint_layers;  // which layers to capture
    int n_embd;
    // Captured hidden states: checkpoint_idx -> flat float vector (n_tokens * n_embd)
    std::vector<std::vector<float>> captured;
    int n_checkpoints;
    bool active = false;
};

static capture_state g_capture;

// cb_eval callback: captures "l_out" tensors at checkpoint layers
static bool tide_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;

    if (!g_capture.active) return false;

    const std::string name(t->name);

    // Debug: print first few unique tensor names to find the right pattern
    static std::set<std::string> seen_names;
    static int debug_count = 0;
    if (ask && debug_count < 100 && seen_names.insert(name).second) {
        if (name.find("l_out") != std::string::npos ||
            name.find("post_moe") != std::string::npos ||
            name.find("ffn_out") != std::string::npos ||
            name.find("attn_res") != std::string::npos) {
            fprintf(stderr, "  [cb_eval] tensor: '%s' ne=(%lld,%lld,%lld,%lld)\n",
                    name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                    (long long)t->ne[2], (long long)t->ne[3]);
            debug_count++;
        }
    }

    // Try multiple name patterns — model builders use different conventions
    int layer = -1;
    if (name.rfind("l_out-", 0) == 0) {
        layer = std::atoi(name.c_str() + 6) + 1;
    } else if (name.rfind("post_moe-", 0) == 0) {
        layer = std::atoi(name.c_str() + 9) + 1;
    }

    if (layer < 0 || g_capture.checkpoint_layers.count(layer) == 0) return false;

    if (ask) {
        return true; // yes, copy this tensor's data
    }

    // Not asking — data is available, capture it
    int n_elements = (int)ggml_nelements(t);
    const float * data = (const float *)t->data;

    // Find checkpoint index
    int ckpt_idx = 0;
    for (int cl : g_capture.checkpoint_layers) {
        if (cl == layer) break;
        ckpt_idx++;
    }

    if (ckpt_idx < (int)g_capture.captured.size()) {
        g_capture.captured[ckpt_idx].assign(data, data + n_elements);
    }

    return false;
}

// Compute cosine similarity between two float vectors
static float cosine_similarity(const float * a, const float * b, int n) {
    double dot = 0, norm_a = 0, norm_b = 0;
    for (int i = 0; i < n; i++) {
        dot    += (double)a[i] * b[i];
        norm_a += (double)a[i] * a[i];
        norm_b += (double)b[i] * b[i];
    }
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 1e-12 ? (float)(dot / denom) : 0.0f;
}

// Generate diverse calibration text
static std::vector<std::string> generate_calibration_text(int n_samples) {
    std::vector<std::string> base_texts = {
        "The transformer architecture revolutionized natural language processing through self-attention mechanisms "
        "that allow models to weigh the importance of different input positions. Multi-head attention provides "
        "multiple representation subspaces enabling simultaneous attention to different relationship types. "
        "Feed-forward networks in each layer provide nonlinear transformations increasing model capacity. "
        "Layer normalization stabilizes training by normalizing activations across the feature dimension. "
        "Residual connections enable gradient flow through deep networks preventing vanishing gradients. ",

        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n"
        "        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n"
        "        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n"
        "    return -1\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n"
        "        self.cache = {}\n        self.order = []\n\n    def get(self, key: int) -> int:\n"
        "        if key not in self.cache:\n            return -1\n        self.order.remove(key)\n",

        "In quantum field theory the path integral formulation provides a framework for computing transition "
        "amplitudes between quantum states. The generating functional Z[J] encodes all correlation functions "
        "of the theory through functional derivatives with respect to the source J. Renormalization removes "
        "ultraviolet divergences order by order in perturbation theory by absorbing infinities into physical "
        "parameters. The renormalization group describes how coupling constants flow with energy scale. ",

        "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent\n"
        "FROM users u LEFT JOIN orders o ON u.id = o.user_id\n"
        "WHERE o.created_at > NOW() - INTERVAL '30 days'\nGROUP BY u.name\n"
        "HAVING COUNT(o.id) > 5\nORDER BY total_spent DESC LIMIT 100;\n"
        "CREATE INDEX CONCURRENTLY idx_orders_user_date ON orders (user_id, created_at DESC);\n",
    };

    std::vector<std::string> samples;
    std::mt19937 rng(42);
    for (int i = 0; i < n_samples; i++) {
        std::string text;
        while (text.size() < 4096) {
            text += base_texts[rng() % base_texts.size()];
        }
        samples.push_back(text);
    }
    return samples;
}

int main(int argc, char ** argv) {
    common_params params;

    std::string output_dir = "./tide_calibration";
    int n_samples = 1000;
    int seq_len = 512;
    int checkpoint_interval = 4;
    bool full_states = false;
    float cos_threshold = 0.98f;

    // Pre-parse custom args
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--n-samples" && i + 1 < argc) {
            n_samples = std::atoi(argv[++i]);
        } else if (arg == "--seq-len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (arg == "--checkpoint-interval" && i + 1 < argc) {
            checkpoint_interval = std::atoi(argv[++i]);
        } else if (arg == "--full-states") {
            full_states = true;
        } else if (arg == "--cos-threshold" && i + 1 < argc) {
            cos_threshold = std::atof(argv[++i]);
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int)filtered_argv.size();

    params.embedding = true;

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    fprintf(stderr, "TIDE Calibration%s\n", full_states ? " (full-states mode)" : " (on-the-fly cosine mode)");
    fprintf(stderr, "  Model: %s\n", params.model.path.c_str());
    fprintf(stderr, "  Output: %s\n", output_dir.c_str());
    fprintf(stderr, "  Samples: %d x %d tokens\n", n_samples, seq_len);
    fprintf(stderr, "  Checkpoint interval: every %d layers\n", checkpoint_interval);
    if (!full_states) {
        fprintf(stderr, "  Cosine threshold: %.2f\n", cos_threshold);
    }

    // Set up eval callback for on-the-fly mode BEFORE context creation
    if (!full_states) {
        params.cb_eval = tide_eval_callback;
        params.cb_eval_user_data = nullptr;
    }

    llama_backend_init();

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const int n_layer = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);

    fprintf(stderr, "  Model: %d layers, %d hidden dim\n", n_layer, n_embd);

    // Compute checkpoint layers
    std::vector<int> checkpoint_layers;
    for (int l = checkpoint_interval; l <= n_layer; l += checkpoint_interval) {
        checkpoint_layers.push_back(l);
    }
    if (checkpoint_layers.back() != n_layer) {
        checkpoint_layers.push_back(n_layer);
    }
    int n_checkpoints = (int)checkpoint_layers.size();

    fprintf(stderr, "  Checkpoints: ");
    for (int l : checkpoint_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "(%d total)\n", n_checkpoints);

    if (full_states) {
        size_t total_bytes = (size_t)n_samples * seq_len * n_checkpoints * n_embd * sizeof(float);
        fprintf(stderr, "  Storage needed: %.1f GB (full-states)\n", total_bytes / 1e9);
    } else {
        // On-the-fly: only store cosine similarities per checkpoint pair
        size_t total_bytes = (size_t)n_samples * seq_len * (n_checkpoints - 1) * sizeof(float);
        fprintf(stderr, "  Storage needed: %.1f MB (on-the-fly cosine)\n", total_bytes / 1e6);
    }

    fs::create_directories(output_dir);

    auto texts = generate_calibration_text(n_samples);

    if (!full_states) {
        // =====================================================
        // ON-THE-FLY MODE: Pairwise forward passes with embeddings API
        // For each checkpoint pair (L, L+interval), run two forward passes
        // at n_layer_exit=L and n_layer_exit=L+interval, compute cosine
        // similarity between the embeddings, write compact labels.
        // Total: 2*(N-1) forward passes per sample.
        // =====================================================
        struct pair_file {
            int layer_a, layer_b;
            FILE * fp_cos;
            FILE * fp_label;
            size_t n_converged;
            size_t n_total;
        };
        std::vector<pair_file> pairs;
        for (int i = 0; i < n_checkpoints - 1; i++) {
            char fname_cos[256], fname_label[256];
            snprintf(fname_cos, sizeof(fname_cos), "%s/cosine_%03d_%03d.bin",
                     output_dir.c_str(), checkpoint_layers[i], checkpoint_layers[i+1]);
            snprintf(fname_label, sizeof(fname_label), "%s/labels_%03d_%03d.bin",
                     output_dir.c_str(), checkpoint_layers[i], checkpoint_layers[i+1]);
            pairs.push_back({
                checkpoint_layers[i], checkpoint_layers[i+1],
                fopen(fname_cos, "wb"), fopen(fname_label, "wb"),
                0, 0
            });
        }

        fprintf(stderr, "  Forward passes per sample: %d (2 per checkpoint pair)\n", 2 * (n_checkpoints - 1));

        auto t_start = std::chrono::high_resolution_clock::now();

        // Temp buffers for embeddings at two consecutive checkpoints
        std::vector<float> emb_a(seq_len * n_embd, 0.0f);
        std::vector<float> emb_b(seq_len * n_embd, 0.0f);

        for (int s = 0; s < n_samples; s++) {
            if (s % 10 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                double rate = s > 0 ? s / elapsed : 0;
                double eta = rate > 0 ? (n_samples - s) / rate : 0;
                fprintf(stderr, "  [%d/%d] %.3f samples/s, ETA %.0fs\n", s, n_samples, rate, eta);
            }

            std::vector<llama_token> tokens = common_tokenize(ctx, texts[s], true);
            if ((int)tokens.size() > seq_len) tokens.resize(seq_len);
            int n_tokens = (int)tokens.size();

            for (int p = 0; p < (int)pairs.size(); p++) {
                auto & pf = pairs[p];

                // Forward pass at layer A
                llama_set_n_layer_exit(ctx, pf.layer_a == n_layer ? 0 : pf.layer_a);
                llama_memory_clear(llama_get_memory(ctx), false);
                {
                    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
                    for (int i = 0; i < n_tokens; i++) common_batch_add(batch, tokens[i], i, {0}, true);
                    llama_decode(ctx, batch);
                    llama_batch_free(batch);
                }
                for (int i = 0; i < n_tokens && i < seq_len; i++) {
                    float * e = llama_get_embeddings_ith(ctx, i);
                    if (e) memcpy(emb_a.data() + i * n_embd, e, n_embd * sizeof(float));
                    else   memset(emb_a.data() + i * n_embd, 0, n_embd * sizeof(float));
                }

                // Forward pass at layer B
                llama_set_n_layer_exit(ctx, pf.layer_b == n_layer ? 0 : pf.layer_b);
                llama_memory_clear(llama_get_memory(ctx), false);
                {
                    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
                    for (int i = 0; i < n_tokens; i++) common_batch_add(batch, tokens[i], i, {0}, true);
                    llama_decode(ctx, batch);
                    llama_batch_free(batch);
                }
                for (int i = 0; i < n_tokens && i < seq_len; i++) {
                    float * e = llama_get_embeddings_ith(ctx, i);
                    if (e) memcpy(emb_b.data() + i * n_embd, e, n_embd * sizeof(float));
                    else   memset(emb_b.data() + i * n_embd, 0, n_embd * sizeof(float));
                }

                // Compute cosine similarity per token
                for (int t = 0; t < seq_len; t++) {
                    float cos = 0.0f;
                    uint8_t label = 0;
                    if (t < n_tokens) {
                        cos = cosine_similarity(
                            emb_a.data() + t * n_embd,
                            emb_b.data() + t * n_embd,
                            n_embd);
                        label = (cos > cos_threshold) ? 1 : 0;
                    }
                    fwrite(&cos, sizeof(float), 1, pf.fp_cos);
                    fwrite(&label, sizeof(uint8_t), 1, pf.fp_label);
                    pf.n_total++;
                    if (label) pf.n_converged++;
                }
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(t_end - t_start).count();

        fprintf(stderr, "\nCalibration complete: %d samples in %.0fs (%.3f samples/s)\n",
                n_samples, total_time, n_samples / total_time);

        fprintf(stderr, "\nConvergence rates (threshold=%.2f):\n", cos_threshold);
        for (auto & pf : pairs) {
            double rate = pf.n_total > 0 ? (double)pf.n_converged / pf.n_total : 0;
            fprintf(stderr, "  Layer %d→%d: %.1f%% converged\n", pf.layer_a, pf.layer_b, rate * 100);
            fclose(pf.fp_cos);
            fclose(pf.fp_label);
        }

        {
            char meta_path[256];
            snprintf(meta_path, sizeof(meta_path), "%s/meta.json", output_dir.c_str());
            FILE * fp = fopen(meta_path, "w");
            fprintf(fp, "{\n  \"mode\": \"on-the-fly\",\n");
            fprintf(fp, "  \"model_path\": \"%s\",\n", params.model.path.c_str());
            fprintf(fp, "  \"n_samples\": %d,\n  \"seq_len\": %d,\n", n_samples, seq_len);
            fprintf(fp, "  \"n_layers\": %d,\n  \"n_embd\": %d,\n", n_layer, n_embd);
            fprintf(fp, "  \"checkpoint_interval\": %d,\n  \"cos_threshold\": %.2f,\n", checkpoint_interval, cos_threshold);
            fprintf(fp, "  \"checkpoint_layers\": [");
            for (int i = 0; i < n_checkpoints; i++)
                fprintf(fp, "%d%s", checkpoint_layers[i], i < n_checkpoints-1 ? ", " : "");
            fprintf(fp, "],\n  \"n_checkpoints\": %d,\n", n_checkpoints);
            fprintf(fp, "  \"calibration_time_s\": %.1f,\n", total_time);
            fprintf(fp, "  \"convergence_rates\": {");
            for (int p = 0; p < (int)pairs.size(); p++) {
                double rate = pairs[p].n_total > 0 ? (double)pairs[p].n_converged / pairs[p].n_total : 0;
                fprintf(fp, "\"%d_%d\": %.4f%s",
                        pairs[p].layer_a, pairs[p].layer_b, rate,
                        p < (int)pairs.size()-1 ? ", " : "");
            }
            fprintf(fp, "}\n}\n");
            fclose(fp);
            fprintf(stderr, "Metadata: %s\n", meta_path);
        }

    } else {
        // =====================================================
        // FULL-STATES MODE: N forward passes per sample
        // =====================================================
        struct layer_file { int layer; FILE * fp; };
        std::vector<layer_file> files;
        for (int l : checkpoint_layers) {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/hidden_layer_%03d.bin", output_dir.c_str(), l);
            FILE * fp = fopen(fname, "wb");
            if (!fp) { fprintf(stderr, "Failed to open %s\n", fname); return 1; }
            files.push_back({l, fp});
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int s = 0; s < n_samples; s++) {
            if (s % 50 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                double rate = s > 0 ? s / elapsed : 0;
                double eta = rate > 0 ? (n_samples - s) / rate : 0;
                fprintf(stderr, "  [%d/%d] %.1f samples/s, ETA %.0fs\n", s, n_samples, rate, eta);
            }

            std::vector<llama_token> tokens = common_tokenize(ctx, texts[s], true);
            if ((int)tokens.size() > seq_len) tokens.resize(seq_len);
            int n_tokens = (int)tokens.size();

            for (size_t ci = 0; ci < checkpoint_layers.size(); ci++) {
                int exit_layer = checkpoint_layers[ci];
                llama_set_n_layer_exit(ctx, exit_layer == n_layer ? 0 : exit_layer);
                llama_memory_clear(llama_get_memory(ctx), false);

                llama_batch batch = llama_batch_init(n_tokens, 0, 1);
                for (int i = 0; i < n_tokens; i++) {
                    common_batch_add(batch, tokens[i], i, {0}, true);
                }

                if (llama_decode(ctx, batch) != 0) {
                    std::vector<float> zeros(seq_len * n_embd, 0.0f);
                    fwrite(zeros.data(), sizeof(float), zeros.size(), files[ci].fp);
                    llama_batch_free(batch);
                    continue;
                }

                for (int i = 0; i < seq_len; i++) {
                    if (i < n_tokens) {
                        float * emb = llama_get_embeddings_ith(ctx, i);
                        if (emb) {
                            fwrite(emb, sizeof(float), n_embd, files[ci].fp);
                        } else {
                            std::vector<float> zeros(n_embd, 0.0f);
                            fwrite(zeros.data(), sizeof(float), n_embd, files[ci].fp);
                        }
                    } else {
                        std::vector<float> zeros(n_embd, 0.0f);
                        fwrite(zeros.data(), sizeof(float), n_embd, files[ci].fp);
                    }
                }
                llama_batch_free(batch);
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(stderr, "\nCalibration complete: %d samples in %.0fs (%.1f samples/s)\n",
                n_samples, total_time, n_samples / total_time);

        for (auto & lf : files) fclose(lf.fp);

        {
            size_t total_bytes = (size_t)n_samples * seq_len * n_checkpoints * n_embd * sizeof(float);
            char meta_path[256];
            snprintf(meta_path, sizeof(meta_path), "%s/meta.json", output_dir.c_str());
            FILE * fp = fopen(meta_path, "w");
            fprintf(fp, "{\n  \"mode\": \"full-states\",\n");
            fprintf(fp, "  \"model_path\": \"%s\",\n", params.model.path.c_str());
            fprintf(fp, "  \"n_samples\": %d,\n  \"seq_len\": %d,\n", n_samples, seq_len);
            fprintf(fp, "  \"n_layers\": %d,\n  \"n_embd\": %d,\n", n_layer, n_embd);
            fprintf(fp, "  \"checkpoint_interval\": %d,\n", checkpoint_interval);
            fprintf(fp, "  \"checkpoint_layers\": [");
            for (int i = 0; i < n_checkpoints; i++)
                fprintf(fp, "%d%s", checkpoint_layers[i], i < n_checkpoints-1 ? ", " : "");
            fprintf(fp, "],\n  \"n_checkpoints\": %d,\n", n_checkpoints);
            fprintf(fp, "  \"calibration_time_s\": %.1f,\n", total_time);
            fprintf(fp, "  \"storage_gb\": %.1f,\n", total_bytes / 1e9);
            fprintf(fp, "  \"file_format\": \"raw float32, shape (n_samples, seq_len, n_embd) per layer file\"\n}\n");
            fclose(fp);
            fprintf(stderr, "Metadata: %s\n", meta_path);
        }
    }

    llama_set_n_layer_exit(ctx, 0);
    llama_backend_free();
    return 0;
}
