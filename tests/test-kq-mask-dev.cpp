// test-kq-mask-dev.cpp: [GPU-RESIDENT-ROUND-2a] identity of the device-built KQ mask (LLAMA_DEVICE_KQ_MASK=1).
//
// Part 1 (always): the ggml op chain llama_build_kq_mask_dev against an independent reference of the host mask rule
// (causal, empty cells and other sequences dropped), BITWISE, F32 and F16, unified (1 stream) and per-sequence
// streams, random cell layouts. Backend = CPU, or KQMD_BACKEND=<name> (e.g. ROCm0) for the device kernels.
// Part 2 (KQMD_MODEL=<gguf>): end-to-end on the CPU backend. A 4-sequence speculative-like decode (prompt, then
// rounds of 1 + draft tokens per sequence with partial rollback when the memory allows it) runs with the knob off
// and on; every output logit row must be bitwise identical, for unified and non-unified KV.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-kq-mask-dev.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_witness = 0;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (strstr(text, "LLAMA_DEVICE_KQ_MASK: device-built KQ mask ENABLED")) {
        g_witness++;
        fputs(text, stderr);
    }
    (void) level;
}

// ---------------------------------------------------------------- part 1

static bool op_case(ggml_backend_t be, std::mt19937 & rng, int64_t n_kv, int64_t n_seq, int64_t n_stream, int64_t n_tps, ggml_type type) {
    const int64_t n_tokens = n_tps*n_stream;

    // cells: per stream a pos and a seq set; empty with prob 0.2
    std::vector<int32_t>  cpos_i(n_stream*n_kv);
    std::vector<uint32_t> cseq(n_stream*n_kv);
    std::uniform_int_distribution<int> pd(0, 300), ud(0, 99);
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t j = 0; j < n_kv; ++j) {
            cpos_i[s*n_kv + j] = pd(rng);
            cseq[s*n_kv + j]   = ud(rng) < 20 ? 0u : (uint32_t) (rng() & ((1u << n_seq) - 1));
        }
    }
    // tokens: stream s holds tokens of seq s (n_stream > 1) or any seq (unified)
    std::vector<int32_t> tseq(n_tokens), tpos_i(n_tokens);
    std::vector<int32_t> seq_to_stream(n_seq);
    for (int64_t q = 0; q < n_seq; ++q) {
        seq_to_stream[q] = n_stream > 1 ? (int32_t) (q % n_stream) : 0;
    }
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;
            tseq[i]   = n_stream > 1 ? (int32_t) s : (int32_t) (rng() % n_seq);
            tpos_i[i] = pd(rng);
        }
    }

    // reference: host rule
    std::vector<float> ref(n_kv*n_tokens);
    for (int64_t i = 0; i < n_tokens; ++i) {
        const int32_t q = tseq[i];
        const int64_t st = seq_to_stream[q];
        for (int64_t j = 0; j < n_kv; ++j) {
            const uint32_t m = cseq[st*n_kv + j];
            const bool keep = m != 0 && (m >> q & 1u) && cpos_i[st*n_kv + j] <= tpos_i[i];
            ref[i*n_kv + j] = keep ? 0.0f : -INFINITY;
        }
    }

    // inputs as set_input_kq_mask_dev writes them
    std::vector<float> cpos(n_seq*n_kv), tpos(n_tokens);
    for (int64_t q = 0; q < n_seq; ++q) {
        const int64_t st = seq_to_stream[q];
        for (int64_t j = 0; j < n_kv; ++j) {
            const uint32_t m = cseq[st*n_kv + j];
            cpos[q*n_kv + j] = (m != 0 && (m >> q & 1u)) ? (float) cpos_i[st*n_kv + j] : LLAMA_KQ_MASK_DEV_FAR;
        }
    }
    for (int64_t i = 0; i < n_tokens; ++i) {
        tpos[i] = (float) tpos_i[i];
    }

    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_seq);
    ggml_tensor * ts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * tt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, n_tps, 1, n_stream);
    ggml_tensor * out = llama_build_kq_mask_dev(ctx, tc, ts, tt, type);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(tc, cpos.data(), 0, ggml_nbytes(tc));
    ggml_backend_tensor_set(ts, tseq.data(), 0, ggml_nbytes(ts));
    ggml_backend_tensor_set(tt, tpos.data(), 0, ggml_nbytes(tt));
    ggml_backend_graph_compute(be, gf);

    std::vector<char> got(ggml_nbytes(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size());

    std::vector<char> want(got.size());
    if (type == GGML_TYPE_F16) {
        ggml_fp16_t * w = (ggml_fp16_t *) want.data();
        for (size_t k = 0; k < ref.size(); ++k) {
            w[k] = ggml_fp32_to_fp16(ref[k]);
        }
    } else {
        memcpy(want.data(), ref.data(), want.size());
    }
    const bool ok = got == want;
    if (!ok) {
        fprintf(stderr, "FAIL op n_kv=%lld n_seq=%lld n_stream=%lld n_tps=%lld type=%s\n",
                (long long) n_kv, (long long) n_seq, (long long) n_stream, (long long) n_tps, ggml_type_name(type));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static void part1() {
    const char * bn = getenv("KQMD_BACKEND");
    ggml_backend_t be = bn ? ggml_backend_init_by_name(bn, nullptr) : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) {
        fprintf(stderr, "FAIL: backend %s not available\n", bn ? bn : "CPU");
        g_fail++;
        return;
    }
    std::mt19937 rng(1234);
    int n = 0, nok = 0;
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (int64_t n_kv : { 1, 7, 256, 1000, 4096 }) {
            for (int64_t n_seq : { 1, 2, 4 }) {
                for (bool unified : { true, false }) {
                    const int64_t n_stream = unified ? 1 : n_seq;
                    for (int64_t n_tps : { 1, 8, 33 }) {
                        n++;
                        nok += op_case(be, rng, n_kv, n_seq, n_stream, n_tps, type) ? 1 : 0;
                    }
                }
            }
        }
    }
    printf("part1 backend=%s: %d / %d op cases bitwise identical\n", ggml_backend_name(be), nok, n);
    g_fail += n - nok;
    ggml_backend_free(be);
}

// ---------------------------------------------------------------- part 2

static std::vector<float> run_e2e(llama_model * model, bool knob, bool unified) {
    if (knob) {
        setenv("LLAMA_DEVICE_KQ_MASK", "1", 1);
    } else {
        unsetenv("LLAMA_DEVICE_KQ_MASK");
    }
    const int n_seq = 4, n_draft = 7, n_rounds = 8;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 2048;
    cp.n_batch    = 512;
    cp.n_ubatch   = 512;
    cp.n_seq_max  = n_seq;
    cp.kv_unified = unified;
    cp.n_threads  = 8;
    cp.n_threads_batch = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "FAIL: context\n");
        g_fail++;
        return {};
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<float> all;
    llama_batch b = llama_batch_init(512, 0, 1);
    std::vector<int> pos(n_seq, 0);
    std::vector<llama_token> last(n_seq);

    // prompts of different lengths
    b.n_tokens = 0;
    for (int q = 0; q < n_seq; ++q) {
        const int len = 5 + 9*q;
        for (int k = 0; k < len; ++k) {
            const int i = b.n_tokens++;
            b.token[i] = (llama_token) ((101 + 37*q + 13*k) % n_vocab);
            b.pos[i] = pos[q]++;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = q;
            b.logits[i] = k == len - 1;
        }
    }
    auto grab = [&](int n_out) {
        for (int k = 0; k < n_out; ++k) {
            const float * l = llama_get_logits_ith(ctx, k);
            all.insert(all.end(), l, l + n_vocab);
        }
    };
    auto argmax = [&](int k) {
        const float * l = llama_get_logits_ith(ctx, k);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (l[v] > l[best]) {
                best = v;
            }
        }
        return (llama_token) best;
    };
    if (llama_decode(ctx, b) != 0) {
        fprintf(stderr, "FAIL: prompt decode\n");
        g_fail++;
    }
    grab(n_seq);
    for (int q = 0; q < n_seq; ++q) {
        last[q] = argmax(q);
    }

    for (int r = 0; r < n_rounds; ++r) {
        b.n_tokens = 0;
        for (int q = 0; q < n_seq; ++q) {
            for (int k = 0; k <= n_draft; ++k) {
                const int i = b.n_tokens++;
                b.token[i] = k == 0 ? last[q] : (llama_token) ((7 + 31*r + 17*q + 5*k) % n_vocab);
                b.pos[i] = pos[q] + k;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = q;
                b.logits[i] = true;
            }
        }
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "FAIL: verify decode r=%d\n", r);
            g_fail++;
            break;
        }
        grab(b.n_tokens);
        for (int q = 0; q < n_seq; ++q) {
            // accept (r + q) % (n_draft + 1) draft tokens, roll back the rest when the memory allows it
            const int acc = (r + q) % (n_draft + 1);
            int keep = n_draft + 1;
            if (llama_memory_seq_rm(mem, q, pos[q] + 1 + acc, -1)) {
                keep = 1 + acc;
            }
            last[q] = argmax(q*(n_draft + 1) + keep - 1);
            pos[q] += keep;
        }
    }
    llama_batch_free(b);
    llama_free(ctx);
    return all;
}

static void part2(const char * path) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        fprintf(stderr, "FAIL: model %s\n", path);
        g_fail++;
        return;
    }
    for (bool unified : { true, false }) {
        const std::vector<float> off = run_e2e(model, false, unified);
        const std::vector<float> on  = run_e2e(model, true,  unified);
        if (g_witness == 0) {
            fprintf(stderr, "FAIL: knob never engaged (no ENABLED witness)\n");
            g_fail++;
        }
        const bool ok = !off.empty() && off.size() == on.size() && memcmp(off.data(), on.data(), off.size()*sizeof(float)) == 0;
        printf("part2 %s kv_unified=%d: %zu logits, ON == OFF bitwise: %s\n", path, (int) unified, off.size(), ok ? "yes" : "NO");
        g_fail += ok ? 0 : 1;
    }
    llama_model_free(model);
}

int main() {
    ggml_backend_load_all();
    part1();
    if (const char * m = getenv("KQMD_MODEL")) {
        llama_backend_init();
        llama_log_set(log_cb, nullptr);
        part2(m);
        llama_backend_free();
    }
    printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
