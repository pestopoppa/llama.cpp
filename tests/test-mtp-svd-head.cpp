// [TAG_MTP_SVD_HEAD] DRAFT-HEAD-SVD-CPU (src/llama-mtp-svd-ops.h): the two-stage truncated-SVD MTP draft head.
//  1. C = V: the SVD head reproduces the full head (ggml_mul_mat of the same quantized W, per column) BIT-FOR-BIT, n = 1 and
//     n = 3 columns, Q6_K and Q8_0 heads, random factors (the pre-score is irrelevant when every row is rescored). The batched
//     (n = 3) full head is reported for information: Q6_K matches it too, Q8_0 uses batch-shape-dependent activation arithmetic.
//  2. C < V with exact factors (v = I, a = dequant W): exactly C finite logits, each bit-equal to the full head's logit for that
//     row, everything else -inf, and the same argmax as the full head.
//  3. Saved dump (optional): --dump <act_dump --head dir> --head-gguf <gguf holding output.weight> --svd <factor gguf>
//     [--rows N] [--min-agree F]: per dump row, top-1 of the SVD head (factor file's C) vs the full head computed by the same
//     kernels (n = 1, as in drafting), plus agreement of each with the model's own argmax from the dump. Passes iff SVD/full
//     top-1 agreement >= min-agree (default 0.99) and every agreeing row's top-1 logit is bit-equal.
// Usage: test-mtp-svd-head [n_threads] [dump options]

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "../src/llama-mtp-svd-ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_nth = 4;

static ggml_tensor * quant_tensor(ggml_context * ctx, ggml_type type, const std::vector<float> & w, int64_t d, int64_t V) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, d, V);
    ggml_quantize_chunk(type, w.data(), t->data, 0, V, d, nullptr);
    return t;
}

static bool compute(ggml_context * ctx, ggml_cgraph * gf) {
    return ggml_graph_compute_with_ctx(ctx, gf, g_nth) == GGML_STATUS_SUCCESS;
}

static int64_t argmax(const float * x, int64_t n) {
    int64_t b = 0;
    for (int64_t i = 1; i < n; ++i) if (x[i] > x[b]) b = i;
    return b;
}

static int test_synthetic(ggml_type type, int64_t n, bool exact_factors, int64_t C_req) {
    const int64_t d = 512, V = 3001;
    std::mt19937 rng(1234 + (int) type + (int) n);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w(d * V), hx(d * n);
    for (auto & x : w) x = nd(rng) * 0.05f;
    for (auto & x : hx) x = nd(rng);
    const int64_t r = exact_factors ? d : 64;
    ggml_init_params ip = { (size_t) 512 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = quant_tensor(ctx, type, w, d, V);
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
    memcpy(h->data, hx.data(), hx.size() * 4);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, r);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, exact_factors ? GGML_TYPE_F32 : GGML_TYPE_F16, r, V);
    if (exact_factors) {
        float * vv = (float *) v->data;
        for (int64_t i = 0; i < r * d; ++i) vv[i] = 0.f;
        for (int64_t i = 0; i < d; ++i) vv[i * d + i] = 1.f;
        std::vector<float> deq(d * V);
        ggml_get_type_traits(type)->to_float(W->data, deq.data(), d * V);
        memcpy(a->data, deq.data(), deq.size() * 4);
    } else {
        float * vv = (float *) v->data;
        for (int64_t i = 0; i < r * d; ++i) vv[i] = nd(rng);
        ggml_fp16_t * aa = (ggml_fp16_t *) a->data;
        for (int64_t i = 0; i < r * V; ++i) aa[i] = ggml_fp32_to_fp16(nd(rng));
    }
    const int64_t C = C_req > 0 ? C_req : V;
    // reference = the full head evaluated per column (n = 1 mul_mat, as the draft head runs it). For n > 1 the batched full
    // head is also compared, for information only: some quant types (Q8_0) use batch-shape-dependent activation arithmetic.
    std::vector<ggml_tensor *> fullc;
    for (int64_t j = 0; j < n; ++j) {
        ggml_tensor * hj = ggml_cont(ctx, ggml_view_2d(ctx, h, d, 1, h->nb[1], j * h->nb[1]));
        fullc.push_back(ggml_mul_mat(ctx, W, hj));
    }
    ggml_tensor * fullb = ggml_mul_mat(ctx, W, h);
    ggml_tensor * svd = mtp_svd::build_head(ctx, W, h, v, a, C);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (auto * t : fullc) ggml_build_forward_expand(gf, t);
    ggml_build_forward_expand(gf, fullb);
    ggml_build_forward_expand(gf, svd);
    if (!compute(ctx, gf)) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }
    int bad = 0;
    for (int64_t j = 0; j < n; ++j) {
        const float * F = (const float *) fullc[j]->data;
        const float * Fb = (const float *) fullb->data + j * V;
        const float * S = (const float *) ((const char *) svd->data + j * svd->nb[1]);
        int64_t finite = 0, mism = 0, mism_b = 0;
        for (int64_t i = 0; i < V; ++i) {
            if (std::isinf(S[i]) && S[i] < 0) continue;
            ++finite;
            if (memcmp(&S[i], &F[i], 4) != 0) ++mism;
            if (memcmp(&S[i], &Fb[i], 4) != 0) ++mism_b;
        }
        const bool am = argmax(F, V) == argmax(S, V);
        const bool ok = mism == 0 && finite == C && am;
        printf("  %-5s n=%lld col %lld C=%lld %s: finite %lld, bit-mismatch %lld, argmax %s (info: vs batched full head %lld) -> %s\n",
               ggml_type_name(type), (long long) n, (long long) j, (long long) C, exact_factors ? "exact-factors" : "random-factors",
               (long long) finite, (long long) mism, am ? "equal" : "DIFFERENT", (long long) mism_b, ok ? "ok" : "FAIL");
        bad += !ok;
    }
    ggml_free(ctx);
    return bad;
}

static bool read_tensor_from_gguf(const char * path, const char * name, ggml_context * dst_ctx, ggml_tensor ** out) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;
    const int64_t id = gguf_find_tensor(g, name);
    bool ok = false;
    if (id >= 0) {
        ggml_tensor * m = ggml_get_tensor(meta, name);
        ggml_tensor * t = ggml_new_tensor(dst_ctx, m->type, GGML_MAX_DIMS, m->ne);
        FILE * fp = fopen(path, "rb");
        if (fp) {
            const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
            ok = fseeko(fp, (off_t) off, SEEK_SET) == 0 && fread(t->data, 1, ggml_nbytes(t), fp) == ggml_nbytes(t);
            fclose(fp);
        }
        *out = t;
    }
    gguf_free(g);
    ggml_free(meta);
    return ok;
}

static std::vector<char> read_file(const std::string & p) {
    std::vector<char> b;
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return b;
    fseeko(f, 0, SEEK_END);
    b.resize((size_t) ftello(f));
    fseeko(f, 0, SEEK_SET);
    if (fread(b.data(), 1, b.size(), f) != b.size()) b.clear();
    fclose(f);
    return b;
}

static int test_dump(const std::string & dir, const char * head_gguf, const char * svd_gguf, int64_t rows, double min_agree) {
    ggml_init_params ip = { (size_t) 2048 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = nullptr, * v = nullptr, * a = nullptr;
    if (!read_tensor_from_gguf(head_gguf, "output.weight", ctx, &W) || !read_tensor_from_gguf(svd_gguf, "mtp_svd.v", ctx, &v) ||
        !read_tensor_from_gguf(svd_gguf, "mtp_svd.a", ctx, &a)) {
        fprintf(stderr, "dump: cannot read output.weight / mtp_svd.v / mtp_svd.a\n");
        return 1;
    }
    int64_t C = 16384;
    {
        gguf_init_params gp = { true, nullptr };
        gguf_context * g = gguf_init_from_file(svd_gguf, gp);
        const int64_t k = g ? gguf_find_key(g, "mtp_svd.c") : -1;
        if (k >= 0) C = gguf_get_val_u32(g, k);
        if (g) gguf_free(g);
    }
    const int64_t d = W->ne[0], V = W->ne[1];
    std::vector<char> hb = read_file(dir + "/head_hidden.bin"), ib = read_file(dir + "/head_topk_ids.bin");
    const int64_t N = (int64_t) (hb.size() / (4 * d));
    const int64_t K = N ? (int64_t) (ib.size() / (4 * N)) : 0;
    if (!N || !K || v->ne[0] != d || a->ne[1] != V) {
        fprintf(stderr, "dump: shape mismatch (N %lld K %lld d %lld V %lld)\n", (long long) N, (long long) K, (long long) d, (long long) V);
        return 1;
    }
    if (rows <= 0 || rows > N) rows = N;
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 1);
    ggml_tensor * full = ggml_mul_mat(ctx, W, h);
    ggml_tensor * svd = mtp_svd::build_head(ctx, W, h, v, a, C);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, svd);
    const float * H = (const float *) hb.data();
    const int32_t * ids = (const int32_t *) ib.data();
    int64_t agree = 0, bitok = 0, full_model = 0, svd_model = 0;
    const int64_t stride = N / rows;
    for (int64_t q = 0; q < rows; ++q) {
        const int64_t row = q * stride;
        memcpy(h->data, H + row * d, d * 4);
        if (!compute(ctx, gf)) return 1;
        const float * F = (const float *) full->data, * S = (const float *) svd->data;
        const int64_t tf = argmax(F, V), ts = argmax(S, V);
        agree += tf == ts;
        bitok += tf == ts && memcmp(&F[tf], &S[ts], 4) == 0;
        full_model += tf == ids[row * K];
        svd_model += ts == ids[row * K];
    }
    const double ag = (double) agree / rows;
    const bool ok = ag >= min_agree && bitok == agree;
    printf("  dump %s: rows %lld (stride %lld), d %lld, V %lld, A %s, rank %lld, C %lld\n", dir.c_str(), (long long) rows,
           (long long) stride, (long long) d, (long long) V, ggml_type_name(a->type), (long long) a->ne[0], (long long) C);
    printf("  dump: svd/full top-1 agreement %.5f (%lld/%lld), agreeing rows with bit-equal top-1 logit %lld; full vs model argmax %.5f, "
           "svd vs model argmax %.5f -> %s (min %.3f)\n", ag, (long long) agree, (long long) rows, (long long) bitok,
           (double) full_model / rows, (double) svd_model / rows, ok ? "ok" : "FAIL", min_agree);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string dump;
    const char * head_gguf = nullptr, * svd_gguf = nullptr;
    int64_t rows = 512;
    double min_agree = 0.99;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (s == "--head-gguf" && i + 1 < argc) head_gguf = argv[++i];
        else if (s == "--svd" && i + 1 < argc) svd_gguf = argv[++i];
        else if (s == "--rows" && i + 1 < argc) rows = atoll(argv[++i]);
        else if (s == "--min-agree" && i + 1 < argc) min_agree = atof(argv[++i]);
        else g_nth = atoi(argv[i]);
    }
    if (g_nth < 1) g_nth = 1;
    ggml_cpu_init();
    int bad = 0;
    printf("test-mtp-svd-head: %d threads\n", g_nth);
    for (ggml_type t : { GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
        for (int64_t n : { 1, 3 }) {
            bad += test_synthetic(t, n, false, 0);      // C = V, random factors: bit-equal to the full head
            bad += test_synthetic(t, n, true, 256);     // C < V, exact factors: C exact logits, -inf elsewhere, same argmax
        }
    }
    if (!dump.empty()) {
        if (!head_gguf || !svd_gguf) {
            fprintf(stderr, "--dump needs --head-gguf and --svd\n");
            return 2;
        }
        bad += test_dump(dump, head_gguf, svd_gguf, rows, min_agree);
    }
    printf("%s\n", bad ? "FAIL" : "ALL OK");
    return bad ? 1 : 0;
}
