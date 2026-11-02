// Tokenize prospective corpus captures and align chunk interiors to exact token boundaries.
// No model weights, context, evaluation, generation or inference are loaded.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &s) {
    if (s.size() > INT32_MAX) throw std::runtime_error("text too large");
    int32_t n = llama_tokenize(vocab, s.data(), int32_t(s.size()), nullptr, 0, false, false);
    if (n == INT32_MIN || n > 0) throw std::runtime_error("tokenizer size failure");
    std::vector<llama_token> out(size_t(-n), 0);
    int32_t actual = llama_tokenize(vocab, s.data(), int32_t(s.size()), out.data(), int32_t(out.size()), false, false);
    if (actual < 0 || actual != int32_t(out.size())) throw std::runtime_error("tokenizer output failure");
    return out;
}

int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "usage: g3_vocab_tokenize MODEL.gguf\n"; return 2; }
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    auto *model = llama_model_load_from_file(argv[1], params);
    if (!model) { std::cerr << "model vocabulary load failed\n"; return 2; }
    auto *vocab = llama_model_get_vocab(model);
    std::string line;
    try {
        while (std::getline(std::cin, line)) {
            auto in = json::parse(line);
            const auto prompt = in.at("prompt").get<std::string>();
            auto full = tokenize(vocab, prompt);
            auto target = tokenize(vocab, in.at("target").get<std::string>());
            if (target.empty()) throw std::runtime_error("empty target");
            json out = {{"id", in.at("id")}, {"tokens", full}, {"target_tokens", target},
                        {"chunks", json::array()}, {"rejected_chunks", 0}};
            // Reconstruct each token's original bytes. Refuse tokenizers whose
            // pieces do not exactly reconstruct the given unmodified prompt.
            std::vector<size_t> boundaries{0};
            std::string rebuilt;
            for (const auto token : full) {
                std::vector<char> bytes(128);
                int n = llama_token_to_piece(vocab, token, bytes.data(), int32_t(bytes.size()), 0, false);
                if (n < 0) {
                    bytes.resize(size_t(-n));
                    n = llama_token_to_piece(vocab, token, bytes.data(), int32_t(bytes.size()), 0, false);
                }
                if (n < 0) throw std::runtime_error("token piece decode failure");
                rebuilt.append(bytes.data(), size_t(n));
                boundaries.push_back(rebuilt.size());
            }
            if (rebuilt != prompt) throw std::runtime_error("token pieces do not reproduce source prompt");
            for (const auto &span : in.at("spans")) {
                const auto a = span.at("start_byte").get<size_t>();
                const auto b = span.at("end_byte").get<size_t>();
                if (a >= b || b > prompt.size()) throw std::runtime_error("invalid byte span");
                // Trim only partial boundary tokens from the candidate cached
                // chunk; never change prompt text, gold, or token positions.
                const auto left = std::lower_bound(boundaries.begin(), boundaries.end(), a);
                const auto right_after = std::upper_bound(boundaries.begin(), boundaries.end(), b);
                if (left == boundaries.end() || right_after == boundaries.begin()) {
                    out["rejected_chunks"] = out["rejected_chunks"].get<int>() + 1;
                    continue;
                }
                const auto right = right_after - 1;
                if (left >= right || right == boundaries.end() - 1) {
                    out["rejected_chunks"] = out["rejected_chunks"].get<int>() + 1;
                    continue;
                }
                out["chunks"].push_back({{"offset", std::distance(boundaries.begin(), left)},
                                          {"n_tokens", std::distance(left, right)},
                                          {"trim_start_bytes", *left - a}, {"trim_end_bytes", b - *right}});
            }
            std::cout << out.dump() << '\n';
        }
    } catch (const std::exception &e) {
        std::cerr << "tokenization refused: " << e.what() << '\n';
        llama_model_free(model); llama_backend_free(); return 2;
    }
    llama_model_free(model); llama_backend_free(); return 0;
}
