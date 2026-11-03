// One CPU forward over exact token IDs; emits raw final-row logits for all-key parity.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <cstring>
#include <sched.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s model token-ids.txt output.bin\n", argv[0]); return 2; }
    std::ifstream file(argv[2], std::ios::binary);
    const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const std::string canonical = "248045\n846\n198\n41089\n5020\n25\n5606\n2639\n2080\n887\n682\n81049\n69234\n12\n22\n19\n17\n7428\n13\n3437\n369\n279\n4581\n81049\n364\n5606\n2639\n2080\n887\n30\n17308\n1132\n440\n279\n81049\n13\n593\n2083\n5673\n741\n248046\n198\n248045\n74455\n198\n";
    if (!file || content != canonical) { std::fprintf(stderr, "invalid frozen token fixture\n"); return 2; }
    std::istringstream in(content);
    std::vector<llama_token> tokens;
    int tok;
    while (in >> tok) tokens.push_back(tok);
    if (!in.eof() || tokens.size() != 45) return 2;
    const char * mode = std::getenv("LLAMA_LSA_SCREEN");
    if (mode && std::strcmp(mode, "all")) return 2;
    const char * thread_env = std::getenv("LSA_PROBE_THREADS");
    if (!thread_env) return 2;
    char * end = nullptr;
    const long threads = std::strtol(thread_env, &end, 10);
    if (*end || threads < 1 || threads > 96) return 2;
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity)) return 2;
    std::string cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &affinity)) {
        if (!cpus.empty()) cpus += ",";
        cpus += std::to_string(i);
    }
    if (cpus.empty()) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_BF16; cp.type_v = GGML_TYPE_BF16;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) return 5;
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (!logits || n_vocab < 1) return 6;
    FILE * out = std::fopen(argv[3], "wb");
    if (!out) return 7;
    const bool ok = std::fwrite(logits, sizeof(float), n_vocab, out) == size_t(n_vocab) && std::fclose(out) == 0;
    std::fprintf(stderr, "LSA_LOGITS model=%s arm=%s input_sha256=209249faab8ffb4bfe2e966b85b9caaa6628a427da237e0e158a76f31fb495ef tokens=%zu vocab=%d configured_type_k=bf16 configured_type_v=bf16 ctx=%u threads=%d threads_batch=%d affinity=%s configured_gpu_layers=0\n", argv[1], mode ? "all" : "dense", tokens.size(), n_vocab, llama_n_ctx(ctx), llama_n_threads(ctx), llama_n_threads_batch(ctx), cpus.c_str());
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return ok ? 0 : 8;
}
