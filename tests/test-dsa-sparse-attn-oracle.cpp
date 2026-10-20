// DSA-IDX.1: CPU oracle for the indexed sparse-attention op contract (src/llama-dsa-sparse-attn.h).
// The sparse reference must equal dense masked attention over the same rows BIT-FOR-BIT.
#include "../src/llama-dsa-sparse-attn.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

static int n_pass = 0, n_fail = 0;
static void check(bool ok, const char * name) { printf("%-44s %s\n", name, ok ? "ok" : "FAIL"); ok ? ++n_pass : ++n_fail; }
static bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Fix {
    int64_t n_q = 3, n_head = 4, d_qk = 16, d_v = 8, page_size = 16, n_pages = 16; // 256 rows
    std::vector<float> q, k, v, sink;
    std::mt19937 rng{7};
    Fix() {
        std::normal_distribution<float> nd;
        q.resize(n_q * n_head * d_qk); k.resize(page_size * n_pages * d_qk); v.resize(page_size * n_pages * d_v); sink.resize(n_head);
        for (auto * vec : { &q, &k, &v }) { for (auto & x : *vec) { x = nd(rng); } }
        for (auto & x : sink) { x = nd(rng); }
    }
    // build ascending unique rows via page*page_size+offset
    std::vector<int32_t> rows(int64_t n, bool ascending, int64_t lo = 0, int64_t hi = -1) {
        if (hi < 0) { hi = page_size * n_pages; }
        std::vector<int32_t> all; for (int64_t r = lo; r < hi; ++r) { all.push_back((int32_t) r); }
        std::shuffle(all.begin(), all.end(), rng);
        all.resize(n);
        if (ascending) { std::sort(all.begin(), all.end()); }
        return all;
    }
    llama_dsa_sparse_attn_args base() const {
        llama_dsa_sparse_attn_args a; a.n_q = n_q; a.n_head = n_head; a.d_qk = d_qk; a.d_v = d_v;
        a.page_size = page_size; a.n_pages = n_pages; a.q = q.data(); a.k = k.data(); a.v = v.data(); a.scale = 0.25f;
        return a;
    }
};

int main() {
    Fix f;
    const size_t n_out = f.n_q * f.n_head * f.d_v;
    std::vector<float> o_ref(n_out), o_a(n_out), o_b(n_out);

    // 1. gathered oracle: arbitrary order, -1 holes, ragged topk_length, sink, extra set
    {
        const int64_t topk = 24, extra = 8;
        std::vector<int32_t> idx, eidx, len = { 24, 10, 0 }, elen = { 8, 3, 8 };
        for (int64_t iq = 0; iq < f.n_q; ++iq) {
            auto r = f.rows(topk, false); r[2] = -1; r[11] = -1; idx.insert(idx.end(), r.begin(), r.end());
            auto e = f.rows(extra, false); e[5] = -1; eidx.insert(eidx.end(), e.begin(), e.end());
        }
        auto a = f.base(); a.topk = topk; a.indices = idx.data(); a.topk_length = len.data();
        a.extra_topk = extra; a.extra_indices = eidx.data(); a.extra_topk_length = elen.data(); a.attn_sink = f.sink.data();
        a.out = o_ref.data(); check(llama_dsa_sparse_attn_ref(a), "ref ran (full contract)");
        a.out = o_a.data();   check(llama_dsa_dense_masked_oracle_gathered(a), "oracle-A ran");
        check(bit_equal(o_ref, o_a), "A bit-equal: -1 holes+len+extra+sink");
        a.attn_sink = nullptr; a.out = o_ref.data(); llama_dsa_sparse_attn_ref(a);
        a.out = o_a.data(); llama_dsa_dense_masked_oracle_gathered(a);
        check(bit_equal(o_ref, o_a), "A bit-equal: no sink");
        a.extra_indices = nullptr; a.out = o_ref.data(); llama_dsa_sparse_attn_ref(a);
        a.out = o_a.data(); llama_dsa_dense_masked_oracle_gathered(a);
        check(bit_equal(o_ref, o_a), "A bit-equal: no extra, no sink");
    }
    // 2. full-KV oracle: strictly ascending rows (main < extra)
    {
        const int64_t topk = 20, extra = 6;
        std::vector<int32_t> idx, eidx;
        for (int64_t iq = 0; iq < f.n_q; ++iq) {
            auto r = f.rows(topk, true, 0, 200); r[4] = -1; idx.insert(idx.end(), r.begin(), r.end());
            auto e = f.rows(extra, true, 200, 256); eidx.insert(eidx.end(), e.begin(), e.end());
        }
        auto a = f.base(); a.topk = topk; a.indices = idx.data(); a.extra_topk = extra; a.extra_indices = eidx.data(); a.attn_sink = f.sink.data();
        a.out = o_ref.data(); llama_dsa_sparse_attn_ref(a);
        a.out = o_b.data();   check(llama_dsa_dense_masked_oracle_full_kv(a), "oracle-B ran");
        check(bit_equal(o_ref, o_b), "B bit-equal vs full-KV masked dense");
    }
    // 3. degenerate rows: all invalid => zeros (with and without sink); selected value check
    {
        std::vector<int32_t> idx(f.n_q * 4, -1);
        auto a = f.base(); a.topk = 4; a.indices = idx.data(); a.attn_sink = f.sink.data();
        a.out = o_ref.data(); llama_dsa_sparse_attn_ref(a);
        check(std::all_of(o_ref.begin(), o_ref.end(), [](float x) { return x == 0.0f; }), "all-invalid => zeros");
    }
    // 4. single valid slot, no sink => output == that V row exactly
    {
        std::vector<int32_t> idx(f.n_q * 2, -1); for (int64_t iq = 0; iq < f.n_q; ++iq) { idx[iq * 2 + 1] = (int32_t) (3 * f.page_size + 5); }
        auto a = f.base(); a.topk = 2; a.indices = idx.data(); a.out = o_ref.data(); llama_dsa_sparse_attn_ref(a);
        bool ok = true; const float * vr = f.v.data() + (3 * f.page_size + 5) * f.d_v;
        for (int64_t i = 0; i < f.n_q * f.n_head; ++i) { for (int64_t d = 0; d < f.d_v; ++d) { ok &= o_ref[i * f.d_v + d] == vr[d]; } }
        check(ok, "single slot => exactly V row");
    }
    // 5. contract violations rejected
    {
        std::vector<int32_t> idx(f.n_q * 2, 0); idx[1] = 256;
        auto a = f.base(); a.topk = 2; a.indices = idx.data(); a.out = o_ref.data();
        check(!llama_dsa_sparse_attn_ref(a), "index >= rows rejected");
        idx[1] = -2; check(!llama_dsa_sparse_attn_ref(a), "index < -1 rejected");
    }
    printf("RESULT test-dsa-sparse-attn-oracle pass=%d fail=%d\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
