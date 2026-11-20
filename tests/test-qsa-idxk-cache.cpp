// [TAG_QSA_IDXK_CACHE] LLAMA_QSA_IDXK_CACHE=1: the qwen4exp QSA indexer keeps its pooled, normed, roped block
// keys across decode steps (src/llama-qsa-idxk-cache.h). Two things must hold for greedy output to stay
// identical, and this test proves both:
//   1. validity: a cached row is reused only while its inputs are unchanged. Named scenarios (append,
//      MTP rollback rewriting the same cell at the same position, prompt-cache suffix seq_rm, incomplete
//      tail) plus a randomized simulation where every reused row is checked against a content hash of
//      its block. A negative control with the write watermark disabled must catch a stale reuse.
//   2. bit identity: the graph the cache path builds (recompute blocks [c, n_blocks), write them into a
//      persistent store, read [0, n_blocks) back) produces rows and indexer scores bitwise equal to the
//      full recompute, across thread counts and after the cache content was produced by an earlier step.

#include "../src/llama-qsa-idxk-cache.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

//
// part 1: validity
//

// one stream of cells. a cell holds a token at a position, or is empty (pos < 0).
struct sim_stream {
    uint32_t r;
    std::vector<llama_pos> pos;
    std::vector<int32_t>   tok;

    sim_stream(uint32_t r, uint32_t n_kv) : r(r), pos(n_kv, -1), tok(n_kv, 0) {}

    // lowest empty cell first, so a rollback hands the SAME cell back to the re-decode
    void write(llama_pos p, int32_t t) {
        for (size_t j = 0; j < pos.size(); ++j) {
            if (pos[j] < 0) {
                pos[j] = p;
                tok[j] = t;
                return;
            }
        }
        abort();
    }

    void rm(llama_pos p0) {
        for (size_t j = 0; j < pos.size(); ++j) {
            if (pos[j] >= p0) {
                pos[j] = -1;
            }
        }
    }

    llama_pos n_past() const {
        llama_pos m = -1;
        for (llama_pos p : pos) m = std::max(m, p);
        return m + 1;
    }
};

// stands in for the pooled+normed+roped row: a function of exactly the inputs the real row depends on
// (the raw keys, i.e. tokens, of the block's cells in slot order, and the block index)
static uint64_t row_of(const sim_stream & st, const int32_t * blk_cells, const int32_t * filled, uint32_t b) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t) b;
    for (uint32_t i = 0; i < st.r; ++i) {
        const int32_t j = blk_cells[(size_t) b*st.r + i];
        // an incomplete block gathers cell 0 for its holes, exactly like set_input_qsa
        const uint64_t v = (uint64_t) (uint32_t) st.tok[j] * 1000003ull + (uint64_t) (uint32_t) st.pos[j < 0 ? 0 : j];
        h = (h ^ v) * 1099511628211ull;
    }
    return h ^ ((uint64_t) filled[b] << 56);
}

struct sim_cache {
    llama_qsa_idxk_validity v;
    std::vector<uint64_t>   store;   // the persistent rows
    uint32_t gran;
    bool     watermark = true;       // false = negative control
    int64_t  reused = 0, stale = 0, steps = 0;

    sim_cache(uint32_t r, uint32_t n_blk_max, uint32_t gran) : store(n_blk_max, 0), gran(gran) { v.r = r; }

    // one ubatch: apply (watermark), write cells, plan, compute, optionally commit
    void ubatch(sim_stream & st, const std::vector<std::pair<llama_pos, int32_t>> & toks, bool succeed = true) {
        llama_pos min_pos = toks[0].first;
        for (auto & t : toks) min_pos = std::min(min_pos, t.first);
        if (watermark) {
            v.on_write(min_pos);
        }
        for (auto & t : toks) st.write(t.first, t.second);

        const uint32_t n_kv     = (uint32_t) st.pos.size();
        const uint32_t n_blocks = (n_kv + v.r - 1)/v.r;
        std::vector<int32_t> blk_cells((size_t) v.r*n_blocks), filled(n_blocks);
        CHECK(llama_qsa_idxk_map(v.r, n_blocks, st.pos.data(), n_kv, blk_cells.data(), filled.data()));

        const uint32_t c = llama_qsa_idxk_split(v.n_reusable(blk_cells.data(), filled.data(), n_blocks), n_blocks, gran);
        for (uint32_t b = 0; b < c; ++b) {
            reused++;
            if (store[b] != row_of(st, blk_cells.data(), filled.data(), b)) {
                stale++;
            }
        }
        if (!succeed) {
            return;   // compute failed: no write-back, no commit
        }
        for (uint32_t b = c; b < n_blocks; ++b) {
            store[b] = row_of(st, blk_cells.data(), filled.data(), b);
        }
        v.commit(blk_cells.data(), filled.data(), n_blocks, min_pos);
        steps++;
    }

    void seq_rm(sim_stream & st, llama_pos p0) {
        if (watermark) {
            v.on_write(p0);
        }
        st.rm(p0);
    }
};

static std::vector<std::pair<llama_pos, int32_t>> run(llama_pos p0, int n, int32_t t0) {
    std::vector<std::pair<llama_pos, int32_t>> v;
    for (int i = 0; i < n; ++i) v.push_back({ p0 + i, t0 + i });
    return v;
}

static void test_named() {
    const uint32_t r = 4, n_kv = 64;

    // append: blocks become valid only after a LATER ubatch starts past them
    {
        // a block is cacheable only when all of its positions are below the ubatch's min position
        CHECK(llama_qsa_idxk_validity::n_cacheable(r, std::vector<int32_t>{4, 4, 2}.data(), 3, 0) == 0);
        CHECK(llama_qsa_idxk_validity::n_cacheable(r, std::vector<int32_t>{4, 4, 2}.data(), 3, 8) == 2);
        CHECK(llama_qsa_idxk_validity::n_cacheable(r, std::vector<int32_t>{4, 4, 2}.data(), 3, 7) == 1);
        CHECK(llama_qsa_idxk_validity::n_cacheable(r, std::vector<int32_t>{4, 4, 4}.data(), 3, 100) == 3);
        CHECK(llama_qsa_idxk_validity::n_cacheable(r, std::vector<int32_t>{4, 3, 4}.data(), 3, 100) == 1);   // hole stops the prefix
    }
    {
        sim_stream st(r, n_kv);
        sim_cache  sc(r, n_kv/r, 1);
        sc.ubatch(st, run(0, 10, 100));        // prefill
        CHECK(sc.v.n_valid == 0);              // min pos 0
        sc.ubatch(st, run(10, 1, 200));        // decode pos 10: blocks 0,1 (pos 0..7) are below it
        CHECK(sc.v.n_valid == 2);
        sc.ubatch(st, run(11, 1, 201));
        CHECK(sc.v.n_valid == 2);              // block 2 (8..11) was just completed by this ubatch: not yet
        sc.ubatch(st, run(12, 1, 202));
        CHECK(sc.v.n_valid == 3);              // now block 2 lies wholly below pos 12
        CHECK(sc.stale == 0 && sc.reused > 0);
    }
    // incomplete tail: never cacheable, however old
    {
        sim_stream st(r, n_kv);
        sim_cache  sc(r, n_kv/r, 1);
        sc.ubatch(st, run(0, 6, 100));         // block 1 holds 4,5 only
        sc.ubatch(st, run(40, 1, 1));          // a far position: block 1 still incomplete
        CHECK(sc.v.n_valid == 1);
        CHECK(sc.stale == 0);
    }
    // MTP rollback: drafts at 13..15 rejected, positions re-decoded into the SAME cells with other tokens
    {
        sim_stream st(r, n_kv);
        sim_cache  sc(r, n_kv/r, 1);
        sc.ubatch(st, run(0, 12, 100));
        sc.ubatch(st, run(12, 4, 300));        // target token 12 + drafts 13,14,15 (block 3 complete)
        CHECK(sc.v.n_valid == 3);
        sc.ubatch(st, run(16, 1, 400));        // block 3 (12..15) now below 16: cached
        CHECK(sc.v.n_valid == 4);
        const auto cells_before = st.pos;
        sc.seq_rm(st, 13);                     // reject 13..16
        CHECK(sc.v.n_valid == 3);              // block 3 holds 13: dropped
        sc.ubatch(st, run(13, 3, 900));        // same positions, same cells, DIFFERENT tokens
        CHECK(st.pos[13] == cells_before[13]); // the cell index and position really did come back unchanged
        sc.ubatch(st, run(16, 1, 901));
        CHECK(sc.stale == 0);
    }
    // the watermark alone catches a same-cell rewrite even if the seq_rm hook were bypassed: the re-decode
    // ubatch's own min position truncates validity in apply()
    {
        sim_stream st(r, n_kv);
        sim_cache  sc(r, n_kv/r, 1);
        sc.ubatch(st, run(0, 16, 100));
        sc.ubatch(st, run(16, 1, 1));
        CHECK(sc.v.n_valid == 4);
        st.rm(13);                             // removal without telling the cache
        sc.ubatch(st, run(13, 4, 777));
        CHECK(sc.stale == 0);
        CHECK(sc.v.n_valid == 3);
    }
    // prompt-cache reuse: seq_rm a suffix, re-decode the suffix with a different continuation
    {
        sim_stream st(r, n_kv);
        sim_cache  sc(r, n_kv/r, 1);
        sc.ubatch(st, run(0, 40, 100));
        sc.ubatch(st, run(40, 1, 1));
        CHECK(sc.v.n_valid == 10);
        sc.seq_rm(st, 21);
        CHECK(sc.v.n_valid == 5);              // blocks 0..4 end at 19 < 21
        sc.ubatch(st, run(21, 19, 5000));
        sc.ubatch(st, run(40, 1, 2));
        CHECK(sc.stale == 0);
        CHECK(sc.v.n_valid == 10);
    }
    // a failed compute commits nothing
    {
        sim_stream st(r, n_kv);
        sim_cache  sc(r, n_kv/r, 1);
        sc.ubatch(st, run(0, 16, 100));
        sc.ubatch(st, run(16, 1, 1), /*succeed=*/false);
        CHECK(sc.v.n_valid == 0);
    }
    // split: never the last block, rounded down to the granularity
    CHECK(llama_qsa_idxk_split(10, 10, 1) == 9);
    CHECK(llama_qsa_idxk_split(10, 20, 4) == 8);
    CHECK(llama_qsa_idxk_split(3, 20, 4) == 0);
    CHECK(llama_qsa_idxk_split(0, 0, 4) == 0);
    // mapping: a position past the window is refused
    {
        std::vector<llama_pos> pos = { 0, 1, 2, 3, 9 };
        std::vector<int32_t> bc(8), f(2);
        CHECK(!llama_qsa_idxk_map(4, 2, pos.data(), 5, bc.data(), f.data()));
    }
}

// random decode traffic: plain decode, MTP draft+rollback, prompt-cache suffix reuse, failed computes
static void test_random(bool watermark, int64_t & reused, int64_t & stale) {
    std::mt19937 rng(1234);
    reused = stale = 0;
    for (uint32_t r : { 1u, 2u, 4u, 8u }) {
        for (uint32_t gran : { 1u, 4u }) {
            const uint32_t n_kv = 512;
            sim_stream st(r, n_kv);
            sim_cache  sc(r, (n_kv + r - 1)/r, gran);
            sc.watermark = watermark;
            int32_t tok = 1;
            sc.ubatch(st, run(0, 1 + rng() % 40, tok)); tok += 64;
            for (int it = 0; it < 3000; ++it) {
                const llama_pos np = st.n_past();
                if (np > (llama_pos) n_kv - 24) {
                    sc.seq_rm(st, rng() % (np/2));        // prompt-cache style truncation
                    continue;
                }
                switch (rng() % 6) {
                    case 0: case 1: sc.ubatch(st, run(np, 1, tok++)); break;
                    case 2: {                             // MTP: target + k drafts, reject a suffix, re-decode
                        const int k = 1 + rng() % 4;
                        sc.ubatch(st, run(np, 1 + k, tok)); tok += 8;
                        const int acc = rng() % (k + 1);
                        sc.seq_rm(st, np + 1 + acc);
                        sc.ubatch(st, run(np + 1 + acc, 1, tok++));
                        break;
                    }
                    case 3: {                             // suffix seq_rm, re-decode with another continuation
                        const llama_pos p0 = np - std::min<llama_pos>(np, 1 + rng() % 20);
                        sc.seq_rm(st, p0);
                        sc.ubatch(st, run(p0, 1 + rng() % 24, tok)); tok += 32;
                        break;
                    }
                    case 4: sc.ubatch(st, run(np, 1, tok++), rng() % 2 == 0); break;
                    case 5: sc.ubatch(st, run(np, 1 + rng() % 16, tok)); tok += 16; break;
                }
            }
            reused += sc.reused;
            stale  += sc.stale;
        }
    }
}

//
// part 2: bit identity of the cache graph against the full recompute (CPU backend)
//

struct ident_cfg {
    int64_t idx_dim = 128, r = 4, n_kv = 1024, n_idx_h = 8, n_tps = 1;
    int     sections[4] = { 16, 24, 24, 0 };
    float   eps = 1e-6f;
};

// the pooled -> norm -> rope chain of build_qsa_top_k, on blocks [c, c + n_rc) of the given cells
static ggml_tensor * build_chain(ggml_context * ctx, const ident_cfg & cfg, ggml_tensor * k_all, ggml_tensor * blk_cells,
                                 ggml_tensor * blk_pos, ggml_tensor * w_norm, int64_t c, int64_t n_rc) {
    const int64_t r = cfg.r, idx_dim = cfg.idx_dim;
    ggml_tensor * bc = c > 0 || blk_cells->ne[0] != r*n_rc ?
        ggml_view_2d(ctx, blk_cells, r*n_rc, 1, blk_cells->nb[1], r*c*ggml_element_size(blk_cells)) : blk_cells;
    ggml_tensor * members = ggml_get_rows(ctx, k_all, bc);
    members = ggml_reshape_4d(ctx, members, idx_dim, r, n_rc, 1);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx, ggml_view_3d(ctx, members, idx_dim, n_rc, 1, members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx, pooled, 1.0f/(float) r);
    pooled = ggml_reshape_3d(ctx, pooled, idx_dim, 1, n_rc);
    pooled = ggml_rms_norm(ctx, pooled, cfg.eps);
    pooled = ggml_mul(ctx, pooled, w_norm);
    int sec[4] = { cfg.sections[0], cfg.sections[1], cfg.sections[2], cfg.sections[3] };
    pooled = ggml_rope_multi(ctx, pooled, blk_pos, nullptr, (int) idx_dim, sec, GGML_ROPE_TYPE_MROPE, 262144,
            10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    return ggml_reshape_3d(ctx, pooled, idx_dim, n_rc, 1);
}

static ggml_tensor * build_score(ggml_context * ctx, const ident_cfg & cfg, ggml_tensor * keys, ggml_tensor * q, int64_t n_blocks) {
    ggml_tensor * score = ggml_mul_mat(ctx, keys, q);
    score = ggml_reshape_4d(ctx, score, n_blocks, cfg.n_idx_h, cfg.n_tps, 1);
    score = ggml_relu(ctx, score);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
    return ggml_sum_rows(ctx, score);
}

struct ident_inputs {
    std::vector<uint16_t> k_raw;      // F16 [idx_dim, n_kv]
    std::vector<int32_t>  cell_of;    // position -> cell (a shuffled layout, so blocks are not contiguous)
    std::vector<float>    w, q;
};

// n_valid_pos tokens at positions 0..n_valid_pos-1; returns {rows, scores} for blocks [0, n_blocks)
// cache mode: recompute [c, n_blocks) into `store` and read [0, n_blocks) back
static void run_ident(const ident_cfg & cfg, const ident_inputs & in, int64_t n_pos, bool cache, int64_t c,
                      std::vector<float> & store, int nth, std::vector<float> & rows, std::vector<float> & scores) {
    const int64_t r = cfg.r, idx_dim = cfg.idx_dim, n_kv = cfg.n_kv, n_blocks = (n_kv + r - 1)/r;
    const int64_t n_rc = cache ? n_blocks - c : n_blocks;

    ggml_init_params ip = { (size_t) 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * k_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, idx_dim, n_kv, 1);
    memcpy(k_all->data, in.k_raw.data(), ggml_nbytes(k_all));

    ggml_tensor * blk_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r*n_blocks, 1);
    std::vector<llama_pos> cell_pos(n_kv, -1);
    for (int64_t p = 0; p < n_pos; ++p) cell_pos[in.cell_of[p]] = (llama_pos) p;
    std::vector<int32_t> filled(n_blocks);
    CHECK(llama_qsa_idxk_map((uint32_t) r, (uint32_t) n_blocks, cell_pos.data(), (uint32_t) n_kv, (int32_t *) blk_cells->data, filled.data()));

    ggml_tensor * blk_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4*n_rc);
    const int64_t blk0 = cache ? c : 0;
    for (int64_t sec = 0; sec < 4; ++sec)
        for (int64_t b = 0; b < n_rc; ++b) ((int32_t *) blk_pos->data)[sec*n_rc + b] = (int32_t) ((blk0 + b)*r);

    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, idx_dim);
    memcpy(w->data, in.w.data(), ggml_nbytes(w));
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, idx_dim, cfg.n_idx_h*cfg.n_tps, 1);
    memcpy(q->data, in.q.data(), ggml_nbytes(q));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * keys = build_chain(ctx, cfg, k_all, blk_cells, blk_pos, w, blk0, n_rc);
    if (cache) {
        ggml_tensor * st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, idx_dim, n_blocks + 7, 1);   // spare rows: a strided store
        memcpy(st->data, store.data(), ggml_nbytes(st));
        ggml_tensor * dst = ggml_view_3d(ctx, st, idx_dim, n_rc, 1, st->nb[1], st->nb[2], c*st->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, keys, dst));
        keys = ggml_view_3d(ctx, st, idx_dim, n_blocks, 1, st->nb[1], st->nb[2], 0);
        if (!ggml_is_contiguous(keys)) keys = ggml_cont(ctx, keys);
        ggml_tensor * sc = build_score(ctx, cfg, keys, q, n_blocks);
        ggml_build_forward_expand(gf, sc);
        CHECK(ggml_graph_compute_with_ctx(ctx, gf, nth) == GGML_STATUS_SUCCESS);
        memcpy(store.data(), st->data, ggml_nbytes(st));
        rows.assign((float *) st->data, (float *) st->data + idx_dim*n_blocks);
        scores.assign((float *) sc->data, (float *) sc->data + ggml_nelements(sc));
    } else {
        keys = ggml_cont(ctx, keys);
        ggml_tensor * sc = build_score(ctx, cfg, keys, q, n_blocks);
        ggml_build_forward_expand(gf, sc);
        ggml_build_forward_expand(gf, keys);
        CHECK(ggml_graph_compute_with_ctx(ctx, gf, nth) == GGML_STATUS_SUCCESS);
        rows.assign((float *) keys->data, (float *) keys->data + idx_dim*n_blocks);
        scores.assign((float *) sc->data, (float *) sc->data + ggml_nelements(sc));
    }
    ggml_free(ctx);
}

static bool bit_equal(const std::vector<float> & a, const std::vector<float> & b, int64_t n) {
    return (int64_t) a.size() >= n && (int64_t) b.size() >= n && memcmp(a.data(), b.data(), n*sizeof(float)) == 0;
}

static void test_identity() {
    ident_cfg cfg;
    const int64_t n_blocks = cfg.n_kv/cfg.r;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    ident_inputs in;
    in.k_raw.resize(cfg.idx_dim*cfg.n_kv);
    for (auto & x : in.k_raw) x = ggml_fp32_to_fp16(nd(rng) * 3.0f);
    in.cell_of.resize(cfg.n_kv);
    for (int64_t i = 0; i < cfg.n_kv; ++i) in.cell_of[i] = (int32_t) i;
    std::shuffle(in.cell_of.begin(), in.cell_of.end(), rng);
    in.w.resize(cfg.idx_dim);
    for (auto & x : in.w) x = 1.0f + 0.1f*nd(rng);
    in.q.resize(cfg.idx_dim*cfg.n_idx_h*cfg.n_tps);
    for (auto & x : in.q) x = nd(rng);

    for (int64_t n_tps : { 1, 4 }) {
        cfg.n_tps = n_tps;
        in.q.resize(cfg.idx_dim*cfg.n_idx_h*cfg.n_tps);
        for (auto & x : in.q) x = nd(rng);

        std::vector<float> store(cfg.idx_dim*(n_blocks + 7), 0.0f);
        std::vector<float> ra, sa, rb, sb, rc, scc;

        // step 1: 600 tokens (150 complete blocks, the rest incomplete/empty)
        run_ident(cfg, in, 600, false, 0, store, 1, ra, sa);
        run_ident(cfg, in, 600, false, 0, store, 8, rb, sb);
        CHECK(bit_equal(ra, rb, cfg.idx_dim*n_blocks) && bit_equal(sa, sb, (int64_t) sa.size()));   // threads do not matter
        run_ident(cfg, in, 600, true, 0, store, 4, rc, scc);     // the cache's first step: c = 0, fills the store
        CHECK(bit_equal(ra, rc, cfg.idx_dim*n_blocks));
        CHECK(bit_equal(sa, scc, (int64_t) sa.size()));

        // steps 2..: more tokens; reuse store rows [0, c) produced by an EARLIER graph, recompute the rest
        int nth = 3;
        for (int64_t n_pos : { 637, 640, 700 }) {
            for (int64_t c : { 64, 128, 149 }) {
                std::vector<float> st = store;
                run_ident(cfg, in, n_pos, false, 0, store, 8, ra, sa);
                run_ident(cfg, in, n_pos, true, c, st, nth, rc, scc);
                nth = nth % 7 + 1;
                const bool ok_rows = bit_equal(ra, rc, cfg.idx_dim*n_blocks);
                const bool ok_scr  = bit_equal(sa, scc, (int64_t) sa.size());
                CHECK(ok_rows);
                CHECK(ok_scr);
                printf("identity n_tps=%" PRId64 " n_pos=%" PRId64 " c=%3" PRId64 ": rows %s, scores %s\n",
                        n_tps, n_pos, c, ok_rows ? "bit-identical" : "DIFFER", ok_scr ? "bit-identical" : "DIFFER");
            }
        }

        // negative control: a store row left over from different keys must be detected as different
        {
            std::vector<float> st(store.size(), 0.0f);
            run_ident(cfg, in, 700, false, 0, store, 8, ra, sa);
            run_ident(cfg, in, 700, true, 64, st, 4, rc, scc);
            CHECK(!bit_equal(ra, rc, cfg.idx_dim*n_blocks));
        }
    }
}

int main() {
    test_named();

    int64_t reused = 0, stale = 0;
    test_random(true, reused, stale);
    printf("random traffic, watermark on : reused rows %" PRId64 ", stale reuses %" PRId64 "\n", reused, stale);
    CHECK(stale == 0);
    CHECK(reused > 10000);

    test_random(false, reused, stale);
    printf("negative control, watermark off: reused rows %" PRId64 ", stale reuses %" PRId64 " (must be > 0)\n", reused, stale);
    CHECK(stale > 0);

    test_identity();

    if (g_fail) {
        printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
