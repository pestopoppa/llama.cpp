/**
 * llama-tts-qwen3 — Qwen3-TTS codec token generator using llama.cpp
 *
 * Loads:
 *   1. Talker GGUF    (28-layer Qwen3, generates 1st codebook token per frame)
 *   2. Code Predictor GGUF (5-layer Qwen3, generates remaining 15 codebook tokens)
 *   3. Sidecar weights (text_embedding, text_projection, multi-head embeddings/lm_heads)
 *
 * Outputs raw codec tokens (16 per frame, one frame per line) to stdout.
 * A Python wrapper converts these to waveform via the Tokenizer Decoder.
 *
 * Usage:
 *   llama-tts-qwen3 \
 *     --model-talker talker.gguf \
 *     --model-cp code_predictor.gguf \
 *     --sidecar sidecar.bin \
 *     -p "Hello, world!" \
 *     -t 48
 */

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// Qwen3-TTS constants
static constexpr int N_CODE_GROUPS  = 16;   // codebook entries per audio frame
static constexpr int CODEC_VOCAB    = 2048; // codec vocabulary size per codebook (actual in model: 3072 with specials)
static constexpr int N_EMBD_TALKER  = 1024; // Talker hidden dimension
static constexpr int N_EMBD_TEXT    = 2048; // Text embedding dimension
static constexpr int TEXT_VOCAB     = 151936; // Qwen3 text vocabulary
static constexpr int CODEC_BOS     = 2149;  // codec BOS token
static constexpr int CODEC_EOS     = 2150;  // codec EOS token
static constexpr int CODEC_PAD     = 2148;  // codec PAD token

// Sidecar weight file header (32 bytes, matches Python create_tts_sidecar.py)
struct sidecar_header {
    char     magic[8];      // "QWTTS02\0"
    int32_t  n_embd_text;   // 2048
    int32_t  n_embd;        // 1024
    int32_t  codec_vocab;   // 3072 (Talker codec_embedding vocab, with specials)
    int32_t  cp_vocab;      // 2048 (Code Predictor vocab size)
    int32_t  n_code_groups; // 16
    int32_t  reserved;
};

// Simple sidecar weight container
struct sidecar_weights {
    // Text embedding: [TEXT_VOCAB, N_EMBD_TEXT]
    std::vector<float> text_embedding;

    // Text projection MLP: fc1(2048→2048) + SiLU + fc2(2048→1024)
    std::vector<float> text_proj_fc1_w;  // [2048, 2048]
    std::vector<float> text_proj_fc1_b;  // [2048]
    std::vector<float> text_proj_fc2_w;  // [1024, 2048]
    std::vector<float> text_proj_fc2_b;  // [1024]

    // Codec embedding: [codec_vocab, N_EMBD_TALKER]
    std::vector<float> codec_embedding;
    int codec_vocab_actual = 0;
    int cp_vocab_actual = 0;

    // Code Predictor multi-head embeddings: [N_CODE_GROUPS-1][CODEC_VOCAB, N_EMBD]
    std::vector<std::vector<float>> cp_embeddings;

    // Code Predictor multi-head lm_heads: [N_CODE_GROUPS-1][CODEC_VOCAB, N_EMBD]
    std::vector<std::vector<float>> cp_lm_heads;
};

// ---- Sidecar loading (from safetensors via pre-converted binary) ----

static bool load_sidecar_safetensors(const std::string & path, sidecar_weights & sw) {
    // We'll load via a Python-generated binary dump for simplicity.
    // Format: header + flat float32 arrays in known order.
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERR("Failed to open sidecar: %s\n", path.c_str());
        return false;
    }

    sidecar_header hdr;
    fprintf(stderr, "Reading sidecar header (%zu bytes)...\n", sizeof(hdr));
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "ERROR: Failed to read sidecar header (expected %zu bytes)\n", sizeof(hdr));
        fclose(f);
        return false;
    }
    if (memcmp(hdr.magic, "QWTTS02", 7) != 0) {
        fprintf(stderr, "ERROR: Invalid sidecar magic: %.8s (expected QWTTS02)\n", hdr.magic);
        fclose(f);
        return false;
    }

    const int ne  = hdr.n_embd;        // 1024
    const int nt  = hdr.n_embd_text;   // 2048
    const int cv  = hdr.codec_vocab;   // 3072 (Talker)
    const int cpv = hdr.cp_vocab;      // 2048 (Code Predictor)
    const int ng  = hdr.n_code_groups; // 16
    fprintf(stderr, "Sidecar header: n_embd_text=%d n_embd=%d codec_vocab=%d cp_vocab=%d n_code_groups=%d\n",
            nt, ne, cv, cpv, ng);
    sw.codec_vocab_actual = cv;
    sw.cp_vocab_actual = cpv;

    auto read_vec = [&](const char * name, std::vector<float> & v, size_t n) -> bool {
        v.resize(n);
        size_t got = fread(v.data(), sizeof(float), n, f);
        if (got != n) {
            fprintf(stderr, "ERROR: read_vec '%s': expected %zu floats, got %zu (at file pos %ld)\n",
                    name, n, got, ftell(f));
            return false;
        }
        fprintf(stderr, "  loaded: %s [%zu floats, %.1f MB]\n", name, n, n * 4.0 / 1048576.0);
        return true;
    };

    // Read in order: text_embedding, text_proj fc1_w, fc1_b, fc2_w, fc2_b, codec_embedding
    if (!read_vec("text_embedding",  sw.text_embedding,  (size_t)TEXT_VOCAB * nt)) { fclose(f); return false; }
    if (!read_vec("text_proj_fc1_w", sw.text_proj_fc1_w, (size_t)nt * nt))        { fclose(f); return false; }
    if (!read_vec("text_proj_fc1_b", sw.text_proj_fc1_b, nt))                      { fclose(f); return false; }
    if (!read_vec("text_proj_fc2_w", sw.text_proj_fc2_w, (size_t)ne * nt))         { fclose(f); return false; }
    if (!read_vec("text_proj_fc2_b", sw.text_proj_fc2_b, ne))                      { fclose(f); return false; }
    if (!read_vec("codec_embedding", sw.codec_embedding, (size_t)cv * ne))         { fclose(f); return false; }

    // CP embeddings: 15 tables of [cp_vocab, N_EMBD]
    sw.cp_embeddings.resize(ng - 1);
    for (int i = 0; i < ng - 1; i++) {
        char name[64]; snprintf(name, sizeof(name), "cp_embd.%d", i);
        if (!read_vec(name, sw.cp_embeddings[i], (size_t)cpv * ne)) { fclose(f); return false; }
    }

    // CP lm_heads: 15 tables of [cp_vocab, N_EMBD]
    sw.cp_lm_heads.resize(ng - 1);
    for (int i = 0; i < ng - 1; i++) {
        char name[64]; snprintf(name, sizeof(name), "cp_head.%d", i);
        if (!read_vec(name, sw.cp_lm_heads[i], (size_t)cpv * ne)) { fclose(f); return false; }
    }

    fclose(f);
    LOG_INF("Sidecar loaded: text_embd=%dx%d, codec=%dx%d, cp_heads=%d\n",
            TEXT_VOCAB, nt, cv, ne, ng - 1);
    return true;
}

// ---- Linear algebra helpers (CPU, float32) ----

// out[row] = embedding_table[token_id * n_embd ... (token_id+1)*n_embd]
static void embed_lookup(float * out, const float * table, int token_id, int n_embd) {
    memcpy(out, table + (size_t)token_id * n_embd, n_embd * sizeof(float));
}

// out = W @ x + bias  (W is [out_dim, in_dim], row-major)
static void linear_forward(float * out, const float * W, const float * x, const float * bias,
                           int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float sum = bias ? bias[i] : 0.0f;
        const float * row = W + (size_t)i * in_dim;
        for (int j = 0; j < in_dim; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

// SiLU activation in-place
static void silu_inplace(float * x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// Vector add: out += src
static void vec_add(float * out, const float * src, int n) {
    for (int i = 0; i < n; i++) {
        out[i] += src[i];
    }
}

// Vector zero
static void vec_zero(float * out, int n) {
    memset(out, 0, n * sizeof(float));
}

// Argmax over logits
static int argmax(const float * logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// Sample from logits with temperature and top-k
static int sample_top_k(const float * logits, int n, float temperature, int top_k, std::mt19937 & rng) {
    if (temperature <= 0.0f) {
        return argmax(logits, n);
    }

    // Build (index, logit) pairs
    std::vector<std::pair<float, int>> candidates(n);
    for (int i = 0; i < n; i++) {
        candidates[i] = {logits[i] / temperature, i};
    }

    // Partial sort for top-k
    int k = std::min(top_k, n);
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });
    candidates.resize(k);

    // Softmax over top-k
    float max_val = candidates[0].first;
    float sum = 0.0f;
    for (auto & c : candidates) {
        c.first = expf(c.first - max_val);
        sum += c.first;
    }

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (const auto & c : candidates) {
        cumsum += c.first;
        if (cumsum >= r) {
            return c.second;
        }
    }
    return candidates.back().second;
}

// Apply repetition penalty to logits in-place
static void apply_repetition_penalty(float * logits, int n, const std::vector<int> & history,
                                      float penalty) {
    if (penalty <= 1.0f) return;
    for (int tok : history) {
        if (tok >= 0 && tok < n) {
            if (logits[tok] > 0) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

// ---- Text projection: text_embedding -> text_projection MLP -> talker space ----

static void text_to_talker_embeds(
        const sidecar_weights & sw,
        const std::vector<llama_token> & text_tokens,
        std::vector<float> & out_embeds, // [n_tokens * N_EMBD_TALKER]
        int n_embd_text,
        int n_embd) {

    const int n_tokens = (int)text_tokens.size();
    out_embeds.resize((size_t)n_tokens * n_embd);

    std::vector<float> tmp_text(n_embd_text);
    std::vector<float> tmp_fc1(n_embd_text);

    for (int t = 0; t < n_tokens; t++) {
        // 1. Text embedding lookup
        embed_lookup(tmp_text.data(), sw.text_embedding.data(), text_tokens[t], n_embd_text);

        // 2. text_projection_fc1: Linear(2048 -> 2048) + SiLU
        linear_forward(tmp_fc1.data(), sw.text_proj_fc1_w.data(), tmp_text.data(),
                       sw.text_proj_fc1_b.data(), n_embd_text, n_embd_text);
        silu_inplace(tmp_fc1.data(), n_embd_text);

        // 3. text_projection_fc2: Linear(2048 -> 1024)
        linear_forward(out_embeds.data() + (size_t)t * n_embd,
                       sw.text_proj_fc2_w.data(), tmp_fc1.data(),
                       sw.text_proj_fc2_b.data(), n_embd, n_embd_text);
    }
}

// ---- Main ----

static void print_usage(int, char ** argv) {
    LOG("\nUsage: %s [options]\n", argv[0]);
    LOG("\n  --model-talker PATH   Talker GGUF model");
    LOG("\n  --model-cp PATH       Code Predictor GGUF model");
    LOG("\n  --sidecar PATH        Sidecar weights binary");
    LOG("\n  -p TEXT               Text to synthesize");
    LOG("\n  --max-frames N        Maximum audio frames (default: 512)");
    LOG("\n  -t N                  Number of threads (default: 48)");
    LOG("\n");
}

int main(int argc, char ** argv) {
    // Parse arguments manually (simpler than common_params for this tool)
    std::string model_talker_path;
    std::string model_cp_path;
    std::string sidecar_path;
    std::string text;
    int max_frames = 512;
    int n_threads = 48;
    float temperature = 0.9f;
    int top_k = 50;
    float rep_penalty = 1.05f;
    uint32_t seed = 42;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--model-talker" && i + 1 < argc) {
            model_talker_path = argv[++i];
        } else if (arg == "--model-cp" && i + 1 < argc) {
            model_cp_path = argv[++i];
        } else if (arg == "--sidecar" && i + 1 < argc) {
            sidecar_path = argv[++i];
        } else if (arg == "-p" && i + 1 < argc) {
            text = argv[++i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--rep-penalty" && i + 1 < argc) {
            rep_penalty = std::atof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(0, argv);
            return 0;
        }
    }

    if (model_talker_path.empty() || sidecar_path.empty() || text.empty()) {
        fprintf(stderr, "ERROR: Missing required arguments: --model-talker, --sidecar, -p\n");
        print_usage(0, argv);
        return 1;
    }

    // Initialize llama backend
    llama_backend_init();

    fprintf(stderr, "tts-qwen3: talker=%s cp=%s sidecar=%s text=\"%s\" threads=%d\n",
            model_talker_path.c_str(), model_cp_path.c_str(), sidecar_path.c_str(),
            text.c_str(), n_threads);

    // ---- Load sidecar weights ----
    sidecar_weights sw;
    if (!load_sidecar_safetensors(sidecar_path, sw)) {
        fprintf(stderr, "ERROR: Failed to load sidecar from %s\n", sidecar_path.c_str());
        return 1;
    }
    fprintf(stderr, "Sidecar loaded OK (sizeof header=%zu)\n", sizeof(sidecar_header));

    // ---- Load Talker model ----
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    LOG_INF("Loading Talker model: %s\n", model_talker_path.c_str());
    llama_model * model_talker = llama_model_load_from_file(model_talker_path.c_str(), model_params);
    if (!model_talker) {
        LOG_ERR("Failed to load Talker model\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_threads     = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.no_perf = false;

    llama_context * ctx_talker = llama_init_from_model(model_talker, ctx_params);
    if (!ctx_talker) {
        LOG_ERR("Failed to create Talker context\n");
        return 1;
    }

    // Enable embeddings to extract hidden states for Code Predictor's past_hidden input
    llama_set_embeddings(ctx_talker, true);

    const int n_embd = llama_model_n_embd(model_talker);
    LOG_INF("Talker: n_embd=%d, n_ctx=%d (embeddings enabled)\n", n_embd, (int)llama_n_ctx(ctx_talker));

    // ---- Load Code Predictor model (optional — if not provided, use greedy from logits) ----
    llama_model   * model_cp = nullptr;
    llama_context * ctx_cp   = nullptr;

    if (!model_cp_path.empty()) {
        LOG_INF("Loading Code Predictor model: %s\n", model_cp_path.c_str());
        model_cp = llama_model_load_from_file(model_cp_path.c_str(), model_params);
        if (!model_cp) {
            LOG_ERR("Failed to load Code Predictor model\n");
            return 1;
        }

        llama_context_params cp_ctx_params = llama_context_default_params();
        cp_ctx_params.n_ctx   = 64;  // CP only needs short sequences (max 17 tokens)
        cp_ctx_params.n_batch = 32;
        cp_ctx_params.n_threads     = n_threads;
        cp_ctx_params.n_threads_batch = n_threads;

        ctx_cp = llama_init_from_model(model_cp, cp_ctx_params);
        if (!ctx_cp) {
            LOG_ERR("Failed to create Code Predictor context\n");
            return 1;
        }

        // Enable embeddings so we can extract hidden states for multi-head lm_head
        llama_set_embeddings(ctx_cp, true);
        LOG_INF("Code Predictor loaded: n_embd=%d (embeddings enabled)\n", llama_model_n_embd(model_cp));
    }

    // ---- Tokenize text ----
    // Use Talker's tokenizer (Qwen3 BPE)
    const llama_vocab * vocab = llama_model_get_vocab(model_talker);

    // Build chat-formatted text: "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    std::string formatted = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> text_tokens(formatted.size() + 32);
    int n_text = llama_tokenize(vocab, formatted.c_str(), formatted.size(),
                                text_tokens.data(), (int)text_tokens.size(), false, true);
    if (n_text < 0) {
        text_tokens.resize(-n_text);
        n_text = llama_tokenize(vocab, formatted.c_str(), formatted.size(),
                                text_tokens.data(), (int)text_tokens.size(), false, true);
    }
    text_tokens.resize(n_text);
    LOG_INF("Text tokens: %d\n", n_text);

    // ---- Compute text embeddings via text_projection ----
    std::vector<float> text_embeds;
    text_to_talker_embeds(sw, text_tokens, text_embeds, N_EMBD_TEXT, n_embd);
    LOG_INF("Text embeddings computed: %d x %d\n", n_text, n_embd);

    // ---- Talker Prefill ----
    // Feed text embeddings to Talker via embedding injection
    {
        llama_batch batch = llama_batch_init(n_text, n_embd, 1);
        batch.n_tokens = n_text;

        for (int i = 0; i < n_text; i++) {
            memcpy(batch.embd + (size_t)i * n_embd,
                   text_embeds.data() + (size_t)i * n_embd,
                   n_embd * sizeof(float));
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1; // all tokens produce output (needed for embeddings extraction)
        }

        const auto t_prefill_start = ggml_time_us();
        if (llama_decode(ctx_talker, batch) != 0) {
            LOG_ERR("Talker prefill failed\n");
            llama_batch_free(batch);
            return 1;
        }
        const float t_prefill_ms = (ggml_time_us() - t_prefill_start) / 1000.0f;
        LOG_INF("Talker prefill: %d tokens in %.1f ms (%.1f t/s)\n",
                n_text, t_prefill_ms, n_text / (t_prefill_ms / 1000.0f));

        llama_batch_free(batch);
    }

    // Set up sampling
    std::mt19937 rng(seed);
    std::vector<int> talker_history; // for repetition penalty

    // Get logits from prefill to sample first codec token
    // Copy logits so we can apply penalties
    std::vector<float> logits_buf(sw.codec_vocab_actual);
    memcpy(logits_buf.data(), llama_get_logits(ctx_talker), sw.codec_vocab_actual * sizeof(float));
    int first_token = sample_top_k(logits_buf.data(), sw.codec_vocab_actual, temperature, top_k, rng);
    talker_history.push_back(first_token);
    LOG_INF("First codec token: %d\n", first_token);

    // ---- Generation loop ----
    // For each audio frame:
    //   1. We have the Talker's logits → sample 1st codebook token
    //   2. Code Predictor generates remaining 15 tokens (using correct per-step lm_heads)
    //   3. Compute sum of all 16 code embeddings → input for next Talker step
    //   4. Add trailing text hidden state from Talker's actual hidden states

    const auto t_gen_start = ggml_time_us();
    int n_frames = 0;
    int pos = n_text; // current position in Talker's KV cache

    // Embedding buffer for next Talker input
    std::vector<float> next_embd(n_embd);
    std::vector<float> tmp_embd(n_embd);

    // CP logits buffer for manual lm_head application
    std::vector<float> cp_logits_buf(sw.cp_vocab_actual);

    // EOS detection: track consecutive identical first_tokens
    int stale_count = 0;
    int prev_first_token = -1;
    const int MAX_STALE_FRAMES = 5; // stop if same token repeats this many times

    while (n_frames < max_frames && first_token != CODEC_EOS) {
        // ---- Stale frame detection (EOS heuristic) ----
        if (first_token == prev_first_token) {
            stale_count++;
            if (stale_count >= MAX_STALE_FRAMES) {
                fprintf(stderr, "Stale frame detected (token %d repeated %d times), stopping\n",
                        first_token, stale_count);
                break;
            }
        } else {
            stale_count = 0;
        }
        prev_first_token = first_token;

        // Collect all 16 codes for this frame
        std::vector<int> frame_codes(N_CODE_GROUPS);
        frame_codes[0] = first_token;

        // Get Talker's hidden state for Code Predictor's past_hidden input
        const float * talker_hidden = llama_get_embeddings_ith(ctx_talker, -1);
        if (!talker_hidden) {
            fprintf(stderr, "WARNING: Could not get Talker embeddings at frame %d, using codec_embedding fallback\n", n_frames);
        }

        // ---- Code Predictor: generate remaining 15 tokens with proper multi-head ----
        if (ctx_cp) {
            // Clear CP KV cache for this frame
            llama_memory_clear(llama_get_memory(ctx_cp), true);

            // Prefill CP with: [past_hidden, first_code_embedding]
            // past_hidden = Talker's last hidden state (or codec_embedding fallback)
            // first_code_embedding = codec_embedding[first_token]
            std::vector<float> cp_prefill(2 * n_embd);

            // Position 0: Talker's hidden state
            if (talker_hidden) {
                memcpy(cp_prefill.data(), talker_hidden, n_embd * sizeof(float));
            } else {
                embed_lookup(cp_prefill.data(), sw.codec_embedding.data(), first_token, n_embd);
            }
            // Position 1: codec embedding of first token
            embed_lookup(cp_prefill.data() + n_embd, sw.codec_embedding.data(), first_token, n_embd);

            llama_batch cp_batch = llama_batch_init(2, n_embd, 1);
            cp_batch.n_tokens = 2;
            for (int i = 0; i < 2; i++) {
                memcpy(cp_batch.embd + (size_t)i * n_embd,
                       cp_prefill.data() + (size_t)i * n_embd,
                       n_embd * sizeof(float));
                cp_batch.pos[i]       = i;
                cp_batch.n_seq_id[i]  = 1;
                cp_batch.seq_id[i][0] = 0;
                cp_batch.logits[i]    = 1;
            }

            if (llama_decode(ctx_cp, cp_batch) != 0) {
                LOG_ERR("CP prefill failed at frame %d\n", n_frames);
                llama_batch_free(cp_batch);
                break;
            }
            llama_batch_free(cp_batch);

            // Step 0: get hidden states from CP and apply lm_head[0] from sidecar
            {
                const float * cp_hidden = llama_get_embeddings_ith(ctx_cp, -1);
                if (cp_hidden) {
                    // Apply lm_head[0]: logits = cp_lm_heads[0] @ hidden (no bias)
                    linear_forward(cp_logits_buf.data(), sw.cp_lm_heads[0].data(),
                                   cp_hidden, nullptr, sw.cp_vocab_actual, n_embd);
                    frame_codes[1] = argmax(cp_logits_buf.data(), sw.cp_vocab_actual);
                } else {
                    // Fallback to GGUF's built-in lm_head (which is lm_head[0])
                    const float * cp_logits = llama_get_logits(ctx_cp);
                    frame_codes[1] = argmax(cp_logits, sw.cp_vocab_actual);
                }
            }

            // Steps 1-14: generate codes 2-15 with per-step embedding and lm_head
            int cp_pos = 2;
            for (int step = 1; step < N_CODE_GROUPS - 1; step++) {
                // Embed previous code with cp_embeddings[step-1]
                std::vector<float> step_embd(n_embd);
                embed_lookup(step_embd.data(), sw.cp_embeddings[step - 1].data(),
                             frame_codes[step], n_embd);

                llama_batch step_batch = llama_batch_init(1, n_embd, 1);
                step_batch.n_tokens = 1;
                memcpy(step_batch.embd, step_embd.data(), n_embd * sizeof(float));
                step_batch.pos[0]       = cp_pos++;
                step_batch.n_seq_id[0]  = 1;
                step_batch.seq_id[0][0] = 0;
                step_batch.logits[0]    = 1;

                if (llama_decode(ctx_cp, step_batch) != 0) {
                    LOG_ERR("CP decode failed at frame %d step %d\n", n_frames, step);
                    llama_batch_free(step_batch);
                    break;
                }
                llama_batch_free(step_batch);

                // Get hidden states and apply the correct lm_head[step]
                const float * cp_hidden = llama_get_embeddings_ith(ctx_cp, -1);
                if (cp_hidden) {
                    linear_forward(cp_logits_buf.data(), sw.cp_lm_heads[step].data(),
                                   cp_hidden, nullptr, sw.cp_vocab_actual, n_embd);
                    frame_codes[step + 1] = argmax(cp_logits_buf.data(), sw.cp_vocab_actual);
                } else {
                    // Fallback: use GGUF logits (approximate, uses lm_head[0])
                    const float * step_logits = llama_get_logits(ctx_cp);
                    frame_codes[step + 1] = argmax(step_logits, sw.cp_vocab_actual);
                }
            }
        } else {
            // No Code Predictor: fill remaining codes with zeros (silence)
            for (int i = 1; i < N_CODE_GROUPS; i++) {
                frame_codes[i] = 0;
            }
        }

        // Output frame codes
        for (int i = 0; i < N_CODE_GROUPS; i++) {
            printf("%d%s", frame_codes[i], (i < N_CODE_GROUPS - 1) ? " " : "\n");
        }
        fflush(stdout);
        n_frames++;

        // ---- Prepare next Talker input ----
        // Sum of all 16 code embeddings
        vec_zero(next_embd.data(), n_embd);

        // First code uses the Talker's codec_embedding
        embed_lookup(tmp_embd.data(), sw.codec_embedding.data(), frame_codes[0], n_embd);
        vec_add(next_embd.data(), tmp_embd.data(), n_embd);

        // Remaining 15 codes use CP embeddings
        for (int i = 0; i < N_CODE_GROUPS - 1; i++) {
            embed_lookup(tmp_embd.data(), sw.cp_embeddings[i].data(), frame_codes[i + 1], n_embd);
            vec_add(next_embd.data(), tmp_embd.data(), n_embd);
        }

        // Add trailing text hidden state
        // Use projected text embeddings (approximation of Talker's text hidden states)
        if (n_frames - 1 < n_text) {
            vec_add(next_embd.data(), text_embeds.data() + (size_t)(n_frames - 1) * n_embd, n_embd);
        }

        // ---- Talker decode step ----
        {
            llama_batch batch = llama_batch_init(1, n_embd, 1);
            batch.n_tokens = 1;
            memcpy(batch.embd, next_embd.data(), n_embd * sizeof(float));
            batch.pos[0]       = pos++;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;

            if (llama_decode(ctx_talker, batch) != 0) {
                LOG_ERR("Talker decode failed at frame %d\n", n_frames);
                llama_batch_free(batch);
                break;
            }
            llama_batch_free(batch);
        }

        // Sample next first codec token with temperature and repetition penalty
        memcpy(logits_buf.data(), llama_get_logits(ctx_talker), sw.codec_vocab_actual * sizeof(float));
        apply_repetition_penalty(logits_buf.data(), sw.codec_vocab_actual, talker_history, rep_penalty);
        first_token = sample_top_k(logits_buf.data(), sw.codec_vocab_actual, temperature, top_k, rng);
        talker_history.push_back(first_token);

        if (first_token == CODEC_EOS) {
            fprintf(stderr, "EOS token at frame %d\n", n_frames);
        }
    }

    const float t_gen_ms = (ggml_time_us() - t_gen_start) / 1000.0f;
    const float audio_sec = n_frames / 12.5f;
    LOG_INF("Generated %d frames (%.2f sec audio) in %.1f ms (%.1fx real-time)\n",
            n_frames, audio_sec, t_gen_ms, audio_sec / (t_gen_ms / 1000.0f));

    // Cleanup
    if (ctx_cp)    llama_free(ctx_cp);
    if (model_cp)  llama_model_free(model_cp);
    llama_free(ctx_talker);
    llama_model_free(model_talker);
    llama_backend_free();

    return 0;
}
