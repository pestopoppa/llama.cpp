// TIDE Calibration Tool — Extract per-layer hidden states for router training
//
// For each calibration sample, runs the model at each checkpoint layer
// (using n_layer_exit) and extracts the embedding (hidden state at exit point).
// Writes results as raw float32 binary files compatible with numpy.
//
// Usage:
//   llama-tide-calibrate -m model.gguf --output-dir ./tide_data \
//       --n-samples 2000 --seq-len 512 --checkpoint-interval 4 -t 96
//
// Output files:
//   ./tide_data/hidden_layer_004.bin  (n_samples * seq_len * n_embd float32)
//   ./tide_data/hidden_layer_008.bin
//   ...
//   ./tide_data/meta.json

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

// Generate diverse calibration text (mix of prose, code, technical)
static std::vector<std::string> generate_calibration_text(int n_samples) {
    std::vector<std::string> base_texts = {
        "The transformer architecture revolutionized natural language processing through self-attention mechanisms "
        "that allow models to weigh the importance of different input positions. Multi-head attention provides "
        "multiple representation subspaces enabling simultaneous attention to different relationship types. "
        "Feed-forward networks in each layer provide nonlinear transformations increasing model capacity. "
        "Layer normalization stabilizes training by normalizing activations across the feature dimension. "
        "Residual connections enable gradient flow through deep networks preventing vanishing gradients. "
        "Position embeddings encode sequential order since attention is permutation invariant. ",

        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n"
        "        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n"
        "        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n"
        "    return -1\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n"
        "        self.cache = {}\n        self.order = []\n\n    def get(self, key: int) -> int:\n"
        "        if key not in self.cache:\n            return -1\n        self.order.remove(key)\n"
        "        self.order.append(key)\n        return self.cache[key]\n",

        "In quantum field theory the path integral formulation provides a framework for computing transition "
        "amplitudes between quantum states. The generating functional Z[J] encodes all correlation functions "
        "of the theory through functional derivatives with respect to the source J. Renormalization removes "
        "ultraviolet divergences order by order in perturbation theory by absorbing infinities into physical "
        "parameters. The renormalization group describes how coupling constants flow with energy scale revealing "
        "fixed points that characterize universality classes of critical phenomena. ",

        "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent\n"
        "FROM users u\nLEFT JOIN orders o ON u.id = o.user_id\n"
        "WHERE o.created_at > NOW() - INTERVAL '30 days'\nGROUP BY u.name\n"
        "HAVING COUNT(o.id) > 5\nORDER BY total_spent DESC\nLIMIT 100;\n\n"
        "CREATE INDEX CONCURRENTLY idx_orders_user_date ON orders (user_id, created_at DESC);\n"
        "ANALYZE orders;\n\nEXPLAIN (ANALYZE, BUFFERS) SELECT * FROM orders WHERE user_id = 42;\n",
    };

    std::vector<std::string> samples;
    std::mt19937 rng(42);
    for (int i = 0; i < n_samples; i++) {
        // Concatenate random base texts to fill seq_len
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

    // Custom args
    std::string output_dir = "./tide_calibration";
    int n_samples = 2000;
    int seq_len = 512;
    int checkpoint_interval = 4;

    // Parse standard llama args first
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Parse our custom args (after standard parsing)
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::string(argv[i]) == "--n-samples" && i + 1 < argc) {
            n_samples = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--seq-len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--checkpoint-interval" && i + 1 < argc) {
            checkpoint_interval = std::atoi(argv[++i]);
        }
    }

    // Force embeddings mode
    params.embedding = true;

    fprintf(stderr, "TIDE Calibration\n");
    fprintf(stderr, "  Model: %s\n", params.model.path.c_str());
    fprintf(stderr, "  Output: %s\n", output_dir.c_str());
    fprintf(stderr, "  Samples: %d x %d tokens\n", n_samples, seq_len);
    fprintf(stderr, "  Checkpoint interval: every %d layers\n", checkpoint_interval);

    // Init model
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
        checkpoint_layers.push_back(n_layer); // always include full model
    }

    fprintf(stderr, "  Checkpoints: ");
    for (int l : checkpoint_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "(%zu total)\n", checkpoint_layers.size());

    // Calculate storage
    size_t total_bytes = (size_t)n_samples * seq_len * checkpoint_layers.size() * n_embd * sizeof(float);
    fprintf(stderr, "  Storage needed: %.1f GB\n", total_bytes / 1e9);

    // Create output directory
    fs::create_directories(output_dir);

    // Open output files (one per checkpoint layer)
    struct layer_file {
        int layer;
        FILE * fp;
    };
    std::vector<layer_file> files;
    for (int l : checkpoint_layers) {
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/hidden_layer_%03d.bin", output_dir.c_str(), l);
        FILE * fp = fopen(fname, "wb");
        if (!fp) {
            fprintf(stderr, "Failed to open %s\n", fname);
            return 1;
        }
        files.push_back({l, fp});
    }

    // Generate calibration text
    fprintf(stderr, "Generating calibration text...\n");
    auto texts = generate_calibration_text(n_samples);

    // Process samples
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < n_samples; s++) {
        if (s % 50 == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double rate = s > 0 ? s / elapsed : 0;
            double eta = rate > 0 ? (n_samples - s) / rate : 0;
            fprintf(stderr, "  [%d/%d] %.1f samples/s, ETA %.0fs\n", s, n_samples, rate, eta);
        }

        // Tokenize
        std::vector<llama_token> tokens = common_tokenize(ctx, texts[s], true);
        if ((int)tokens.size() > seq_len) {
            tokens.resize(seq_len);
        }
        int n_tokens = (int)tokens.size();

        // For each checkpoint layer: set exit, decode, extract embeddings
        for (size_t ci = 0; ci < checkpoint_layers.size(); ci++) {
            int exit_layer = checkpoint_layers[ci];

            // Set n_layer_exit
            llama_set_n_layer_exit(ctx, exit_layer == n_layer ? 0 : exit_layer);

            // Clear KV cache for fresh eval
            llama_memory_clear(llama_get_memory(ctx), false);

            // Prepare batch
            llama_batch batch = llama_batch_init(n_tokens, 0, 1);
            for (int i = 0; i < n_tokens; i++) {
                common_batch_add(batch, tokens[i], i, {0}, true); // all positions output embeddings
            }

            // Decode
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "  [WARN] Decode failed at sample %d, layer %d\n", s, exit_layer);
                // Write zeros for this sample/layer
                std::vector<float> zeros(n_tokens * n_embd, 0.0f);
                fwrite(zeros.data(), sizeof(float), zeros.size(), files[ci].fp);
                llama_batch_free(batch);
                continue;
            }

            // Extract embeddings for all tokens
            // With embeddings=true, llama_get_embeddings_ith(ctx, i) gives the hidden state for token i
            for (int i = 0; i < seq_len; i++) {
                if (i < n_tokens) {
                    float * emb = llama_get_embeddings_ith(ctx, i);
                    if (emb) {
                        fwrite(emb, sizeof(float), n_embd, files[ci].fp);
                    } else {
                        // fallback: write zeros
                        std::vector<float> zeros(n_embd, 0.0f);
                        fwrite(zeros.data(), sizeof(float), n_embd, files[ci].fp);
                    }
                } else {
                    // Pad with zeros if tokens < seq_len
                    std::vector<float> zeros(n_embd, 0.0f);
                    fwrite(zeros.data(), sizeof(float), n_embd, files[ci].fp);
                }
            }

            llama_batch_free(batch);
        }
    }

    // Reset n_layer_exit
    llama_set_n_layer_exit(ctx, 0);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    fprintf(stderr, "\nCalibration complete: %d samples in %.0fs (%.1f samples/s)\n",
            n_samples, total_time, n_samples / total_time);

    // Close files
    for (auto & lf : files) {
        fclose(lf.fp);
    }

    // Write metadata JSON
    {
        char meta_path[256];
        snprintf(meta_path, sizeof(meta_path), "%s/meta.json", output_dir.c_str());
        FILE * fp = fopen(meta_path, "w");
        fprintf(fp, "{\n");
        fprintf(fp, "  \"model_path\": \"%s\",\n", params.model.path.c_str());
        fprintf(fp, "  \"n_samples\": %d,\n", n_samples);
        fprintf(fp, "  \"seq_len\": %d,\n", seq_len);
        fprintf(fp, "  \"n_layers\": %d,\n", n_layer);
        fprintf(fp, "  \"n_embd\": %d,\n", n_embd);
        fprintf(fp, "  \"checkpoint_interval\": %d,\n", checkpoint_interval);
        fprintf(fp, "  \"checkpoint_layers\": [");
        for (size_t i = 0; i < checkpoint_layers.size(); i++) {
            fprintf(fp, "%d%s", checkpoint_layers[i], i < checkpoint_layers.size()-1 ? ", " : "");
        }
        fprintf(fp, "],\n");
        fprintf(fp, "  \"n_checkpoints\": %zu,\n", checkpoint_layers.size());
        fprintf(fp, "  \"calibration_time_s\": %.1f,\n", total_time);
        fprintf(fp, "  \"storage_gb\": %.1f,\n", total_bytes / 1e9);
        fprintf(fp, "  \"file_format\": \"raw float32, shape (n_samples, seq_len, n_embd) per layer file\"\n");
        fprintf(fp, "}\n");
        fclose(fp);
        fprintf(stderr, "Metadata: %s\n", meta_path);
    }

    llama_backend_free();
    return 0;
}
