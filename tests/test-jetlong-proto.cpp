// CTX-JETLONG-PROTO synthetic tests (no model, CPU backend only).
//
// The harness mirrors llm_graph_context::build_attn (kv-cache path) for the qwen35moe full-attention layers:
//   IMRoPE (n_rot 64 of head 256, sections [11,11,10,0], base 1e7) -> optional K Hadamard (quantized K) ->
//   K/V store with set_rows -> stock flash_attn_ext -> [Jet-Long splice] -> optional V Hadamard
// and drives the SAME planner / side-cache bookkeeping / graph builder that libllama uses (src/llama-jetlong.*).
//
//   T1  in-window identity: flag on vs off, max-abs 0 (np 1, np 4 mixed, prefill + decode; plus in-window
//       sequences co-batched with an above-native one)
//   T2  above native vs a double-precision reference of Jet-Long Eq. 1 with the semantics of the repo's
//       `_reference_jetlong_decode` (near base branch + grouped distant branch, LSE merge)
//   T3  epoch crossing (G changes mid-sequence): cached-per-epoch == uncached (max-abs 0), both vs reference
//
// usage: test-jetlong-proto [--long]   (--long adds the real-geometry cases at L = 300000 and 600000)

#include "llama-jetlong.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static int g_threads = 24;

// ---------------------------------------------------------------------------------------------
// geometry and deterministic data
// ---------------------------------------------------------------------------------------------

struct geom {
    int   D      = 256;
    int   H      = 16;
    int   Hkv    = 2;
    int   n_rot  = 64;
    int   sections[4] = {11, 11, 10, 0};
    float base   = 1e7f;
    bool  neox   = false; // false: IMRoPE (qwen35/qwen35moe), true: plain NeoX (qwen3)
};

static uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// raw (pre-RoPE) vectors of a token, a deterministic function of (what, seq, pos)
static void gen_vec(float * dst, int n, int what, int seq, int pos, float scale) {
    std::mt19937_64 rng(splitmix(((uint64_t) what << 56) ^ ((uint64_t) seq << 40) ^ (uint64_t) pos));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        dst[i] = scale*nd(rng);
    }
}

static void gen_hadamard(std::vector<float> & h, int n) {
    h.assign((size_t) n*n, 0.0f);
    h[0] = 1.0f/sqrtf((float) n);
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = h[i*n + j];
                h[(i + s)*n + j]     =  val;
                h[i*n + (j + s)]     =  val;
                h[(i + s)*n + j + s] = -val;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// reference rotations (double-exact and fp32-mirror of ggml's rope cache)
// ---------------------------------------------------------------------------------------------

// rotate the n_rot NeoX-paired dims of x (double) by position p
static void rot_exact(double * x, const geom & g, double p) {
    const int half = g.n_rot/2;
    for (int i = 0; i < half; ++i) {
        const double th = p*pow((double) g.base, -2.0*i/g.n_rot);
        const double c = cos(th), s = sin(th);
        const double x0 = x[i], x1 = x[i + half];
        x[i]        = x0*c - x1*s;
        x[i + half] = x0*s + x1*c;
    }
}

// exactly what ggml-cpu computes (theta *= theta_scale in fp32, cosf/sinf), applied in double
static void rot_f32(double * x, const geom & g, int32_t p) {
    const int half = g.n_rot/2;
    const float theta_scale = powf(g.base, -2.0f/g.n_rot);
    float theta = (float) p;
    for (int i = 0; i < half; ++i) {
        const double c = cosf(theta), s = sinf(theta);
        const double x0 = x[i], x1 = x[i + half];
        x[i]        = x0*c - x1*s;
        x[i + half] = x0*s + x1*c;
        theta *= theta_scale;
    }
}

// fp32 mirror of the forward rope (float arithmetic, to reproduce the stock q_rope)
static void rope_fwd_f32(float * x, const geom & g, int32_t p) {
    const int half = g.n_rot/2;
    const float theta_scale = powf(g.base, -2.0f/g.n_rot);
    float theta = (float) p;
    for (int i = 0; i < half; ++i) {
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = x[i], x1 = x[i + half];
        x[i]        = x0*c - x1*s;
        x[i + half] = x0*s + x1*c;
        theta *= theta_scale;
    }
}

// ---------------------------------------------------------------------------------------------
// runner: one "model layer" with its own K/V cache and Jet-Long side cache
// ---------------------------------------------------------------------------------------------

struct tok { int seq; int pos; int cell; };

struct run_cfg {
    std::string       name;
    ggml_type         tk = GGML_TYPE_F16;
    ggml_type         tv = GGML_TYPE_F16;
    bool              rot = false;   // K Hadamard (256) + V Hadamard (64), as llama.cpp does for quantized K/V
    llama_jetlong_cfg jl;            // w0 == 0 -> Jet-Long off
};

struct step_out {
    std::vector<float> out;          // [Dv*H, T]
    bool    active = false;
    int32_t G      = 1;
    size_t  n_upd  = 0;
    std::vector<uint8_t> tok_ext;
};

struct runner {
    geom    g;
    run_cfg rc;
    int     kv_size;

    ggml_backend_t        be  = nullptr;
    ggml_context        * cp  = nullptr;
    ggml_backend_buffer_t bp  = nullptr;
    ggml_tensor * kc = nullptr, * vc = nullptr, * kgrp = nullptr, * hk = nullptr, * hv = nullptr;

    std::vector<int32_t> cpos, cseq;
    llama_jetlong_cell_state st;
    llama_jetlong_rope rope;

    runner(const geom & g_, const run_cfg & rc_, int kv_size_) : g(g_), rc(rc_), kv_size(kv_size_) {
        be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, g_threads);

        ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
        cp = ggml_init(ip);
        kc   = ggml_new_tensor_2d(cp, rc.tk, (int64_t) g.D*g.Hkv, kv_size);
        vc   = ggml_new_tensor_2d(cp, rc.tv, (int64_t) g.D*g.Hkv, kv_size);
        kgrp = ggml_new_tensor_3d(cp, GGML_TYPE_F32, (int64_t) g.n_rot*g.Hkv, kv_size, 1);
        hk   = ggml_new_tensor_2d(cp, GGML_TYPE_F32, g.D, g.D);
        hv   = ggml_new_tensor_2d(cp, GGML_TYPE_F32, 64, 64);
        bp = ggml_backend_alloc_ctx_tensors(cp, be);
        ggml_backend_buffer_clear(bp, 0);

        std::vector<float> h;
        gen_hadamard(h, g.D); ggml_backend_tensor_set(hk, h.data(), 0, ggml_nbytes(hk));
        gen_hadamard(h, 64);  ggml_backend_tensor_set(hv, h.data(), 0, ggml_nbytes(hv));

        cpos.assign(kv_size, -1);
        cseq.assign(kv_size, -1);
        st.resize(kv_size);

        rope.n_rot = g.n_rot;
        for (int i = 0; i < 4; ++i) rope.sections[i] = g.sections[i];
        rope.mode      = g.neox ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_IMROPE;
        rope.freq_base = g.base;
    }

    ~runner() {
        ggml_backend_buffer_free(bp);
        ggml_free(cp);
        ggml_backend_free(be);
    }

    step_out step(const std::vector<tok> & toks) {
        const int T  = (int) toks.size();
        const int D  = g.D, H = g.H, Hkv = g.Hkv;
        const int n_kv = kv_size;

        // apply_ubatch: the cells hold the new tokens before the graph is built
        for (const auto & t : toks) {
            cpos[t.cell] = t.pos;
            cseq[t.cell] = t.seq;
        }

        std::vector<int32_t> pos(T), seq(T);
        for (int i = 0; i < T; ++i) { pos[i] = toks[i].pos; seq[i] = toks[i].seq; }

        step_out so;
        llama_jetlong_ubatch_plan plan;
        std::vector<int64_t> upd_idx;
        std::vector<int32_t> upd_delta;
        if (rc.jl.enabled()) {
            plan = llama_jetlong_plan_ubatch(rc.jl, T, pos.data(), seq.data());
            if (plan.active) {
                std::vector<int64_t> cand_idx, forced;
                std::vector<int32_t> cand_pos;
                for (int j = 0; j < n_kv; ++j) {
                    if (cpos[j] >= 0) { cand_idx.push_back(j); cand_pos.push_back(cpos[j]); }
                }
                for (const auto & t : toks) forced.push_back(t.cell);
                st.collect(plan.G, rc.jl.uncached, cand_idx, cand_pos, forced, upd_idx, upd_delta);
            }
        }
        so.active = plan.active; so.G = plan.G; so.n_upd = upd_idx.size(); so.tok_ext = plan.tok_ext;

        ggml_init_params ip = { 512*ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

        ggml_tensor * q_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H,   T); ggml_set_input(q_raw);
        ggml_tensor * k_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, T); ggml_set_input(k_raw);
        ggml_tensor * v_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, T); ggml_set_input(v_raw);
        ggml_tensor * ipos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4*T);       ggml_set_input(ipos);
        ggml_tensor * kidx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);         ggml_set_input(kidx);
        ggml_tensor * mask  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, T, 1, 1); ggml_set_input(mask);

        int sections[4] = { g.sections[0], g.sections[1], g.sections[2], g.sections[3] };
        ggml_tensor * q_rope, * k_rope;
        if (g.neox) {
            ggml_tensor * p1 = ggml_view_1d(ctx, ipos, T, 0);
            q_rope = ggml_rope_ext(ctx, q_raw, p1, nullptr, g.n_rot, GGML_ROPE_TYPE_NEOX, 262144, g.base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k_rope = ggml_rope_ext(ctx, k_raw, p1, nullptr, g.n_rot, GGML_ROPE_TYPE_NEOX, 262144, g.base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        } else {
            q_rope = ggml_rope_multi(ctx, q_raw, ipos, nullptr, g.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                    262144, g.base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k_rope = ggml_rope_multi(ctx, k_raw, ipos, nullptr, g.n_rot, sections, GGML_ROPE_TYPE_IMROPE,
                    262144, g.base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        }

        auto hadamard = [&](ggml_tensor * x, ggml_tensor * rot) {
            const int64_t n = rot->ne[0];
            ggml_tensor * r = ggml_is_contiguous(x) ? ggml_reshape_2d(ctx, x, n, ggml_nelements(x)/n)
                                                    : ggml_cont_2d(ctx, x, n, ggml_nelements(x)/n);
            r = ggml_mul_mat(ctx, rot, r);
            return ggml_reshape_4d(ctx, r, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        };

        ggml_tensor * q = q_rope, * k = k_rope, * v = v_raw;
        if (rc.rot) {
            q = hadamard(q, hk);
            k = hadamard(k, hk);
            v = hadamard(v, hv);
        }
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, v);
        ggml_build_forward_expand(gf, k);

        ggml_tensor * k_store = ggml_set_rows(ctx, kc, ggml_view_2d(ctx, k, D*Hkv, T, k->nb[2], 0), kidx);
        ggml_build_forward_expand(gf, k_store);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, vc, ggml_view_2d(ctx, v, D*Hkv, T, v->nb[2], 0), kidx));

        ggml_tensor * kv4 = ggml_view_4d(ctx, kc, D, Hkv, n_kv, 1, ggml_row_size(kc->type, D), kc->nb[1], kc->nb[1]*kv_size, 0);
        ggml_tensor * vv4 = ggml_view_4d(ctx, vc, D, Hkv, n_kv, 1, ggml_row_size(vc->type, D), vc->nb[1], vc->nb[1]*kv_size, 0);

        const float kq_scale = 1.0f/sqrtf((float) D);

        // stock path (build_attn_mha, flash attention branch)
        ggml_tensor * out;
        {
            ggml_tensor * qp = ggml_permute(ctx, ggml_view_4d(ctx, q, D, H, T, 1, q->nb[1], q->nb[2], q->nb[3], 0), 0, 2, 1, 3);
            ggml_tensor * kp = ggml_permute(ctx, kv4, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(ctx, vv4, 0, 2, 1, 3);
            if (kp->type == GGML_TYPE_F32) kp = ggml_cast(ctx, kp, GGML_TYPE_F16);
            if (vp->type == GGML_TYPE_F32) vp = ggml_cast(ctx, vp, GGML_TYPE_F16);
            out = ggml_flash_attn_ext(ctx, qp, kp, vp, mask, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            out = ggml_reshape_2d(ctx, out, out->ne[0]*out->ne[1], out->ne[2]*out->ne[3]);
        }

        llama_jetlong_graph_inp jin;
        if (plan.active) {
            jin.create(ctx, rope, T, (int64_t) upd_idx.size(), n_kv, 1, D);
            out = llama_jetlong_build_attn(ctx, gf, rope, jin, q, q_rope, rc.rot ? hk : nullptr,
                    k_store, kv4, vv4, false, kgrp, 0, mask, kq_scale, out);
        }

        if (rc.rot) {
            out = hadamard(out, hv);
        }
        ggml_build_forward_expand(gf, out);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            fprintf(stderr, "alloc failed\n");
            exit(1);
        }

        // inputs
        {
            std::vector<float> qb((size_t) D*H*T), kb((size_t) D*Hkv*T), vb((size_t) D*Hkv*T);
            for (int i = 0; i < T; ++i) {
                gen_vec(qb.data() + (size_t) i*D*H,   D*H,   0, toks[i].seq, toks[i].pos, 0.5f);
                gen_vec(kb.data() + (size_t) i*D*Hkv, D*Hkv, 1, toks[i].seq, toks[i].pos, 0.5f);
                gen_vec(vb.data() + (size_t) i*D*Hkv, D*Hkv, 2, toks[i].seq, toks[i].pos, 1.0f);
            }
            ggml_backend_tensor_set(q_raw, qb.data(), 0, ggml_nbytes(q_raw));
            ggml_backend_tensor_set(k_raw, kb.data(), 0, ggml_nbytes(k_raw));
            ggml_backend_tensor_set(v_raw, vb.data(), 0, ggml_nbytes(v_raw));

            std::vector<int32_t> pb(4*T);
            for (int i = 0; i < T; ++i) { pb[i] = pb[T + i] = pb[2*T + i] = toks[i].pos; pb[3*T + i] = 0; }
            ggml_backend_tensor_set(ipos, pb.data(), 0, ggml_nbytes(ipos));

            std::vector<int64_t> ib(T);
            for (int i = 0; i < T; ++i) ib[i] = toks[i].cell;
            ggml_backend_tensor_set(kidx, ib.data(), 0, ggml_nbytes(kidx));

            std::vector<ggml_fp16_t> mb((size_t) n_kv*T);
            const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < n_kv; ++j) {
                    const bool ok = cpos[j] >= 0 && cseq[j] == toks[i].seq && cpos[j] <= toks[i].pos;
                    mb[(size_t) i*n_kv + j] = ok ? keep : drop;
                }
            }
            ggml_backend_tensor_set(mask, mb.data(), 0, ggml_nbytes(mask));

            if (plan.active) {
                // jin tensors are in a CPU buffer -> fill through host data pointers
                llama_jetlong_fill_inputs(jin, rc.jl, rope, plan, pos.data(), upd_idx, upd_delta,
                        [&](int64_t, int64_t j) { return cpos[j]; });
            }
        }

        ggml_backend_graph_compute(be, gf);

        so.out.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, so.out.data(), 0, ggml_nbytes(out));

        ggml_gallocr_free(ga);
        ggml_free(ctx);
        return so;
    }

    // host-side fill of cells [0, n) of sequence `s` with synthetic (already base-roped) K and V
    void host_fill(int n, int s) {
        const int64_t row = (int64_t) g.D*g.Hkv;
        std::vector<float> kf((size_t) row), vf((size_t) row);
        std::vector<uint8_t> kq(ggml_row_size(rc.tk, row)), vq(ggml_row_size(rc.tv, row));
        const size_t chunk = 4096;
        std::vector<uint8_t> kbuf(kq.size()*chunk), vbuf(vq.size()*chunk);
        for (int j0 = 0; j0 < n; j0 += (int) chunk) {
            const int nj = std::min<int>((int) chunk, n - j0);
            for (int jj = 0; jj < nj; ++jj) {
                const int j = j0 + jj;
                gen_vec(kf.data(), (int) row, 3, s, j, 0.5f);
                gen_vec(vf.data(), (int) row, 4, s, j, 1.0f);
                ggml_quantize_chunk(rc.tk, kf.data(), kbuf.data() + kq.size()*jj, 0, 1, row, nullptr);
                ggml_quantize_chunk(rc.tv, vf.data(), vbuf.data() + vq.size()*jj, 0, 1, row, nullptr);
                cpos[j] = j;
                cseq[j] = s;
            }
            ggml_backend_tensor_set(kc, kbuf.data(), kc->nb[1]*j0, kq.size()*nj);
            ggml_backend_tensor_set(vc, vbuf.data(), vc->nb[1]*j0, vq.size()*nj);
        }
    }

    // dequantized cache rows back in the RoPE domain (undo the Hadamard), double
    void read_cache(std::vector<double> & kd, std::vector<double> & vd, std::vector<double> * kh = nullptr) const {
        const int64_t row = (int64_t) g.D*g.Hkv;
        std::vector<uint8_t> raw(ggml_nbytes(kc));
        std::vector<float> f((size_t) row*kv_size);
        auto to_f32 = [&](ggml_type t) {
            if (t == GGML_TYPE_F32) memcpy(f.data(), raw.data(), f.size()*sizeof(float));
            else ggml_get_type_traits(t)->to_float(raw.data(), f.data(), row*kv_size);
        };
        ggml_backend_tensor_get(kc, raw.data(), 0, raw.size());
        to_f32(rc.tk);
        kd.assign(f.begin(), f.end());
        raw.resize(ggml_nbytes(vc));
        ggml_backend_tensor_get(vc, raw.data(), 0, raw.size());
        to_f32(rc.tv);
        vd.assign(f.begin(), f.end());
        if (rc.rot && kh) {
            *kh = kd; // the cache content in the Hadamard domain (what the q8_0 dot products see)
        }
        if (rc.rot) {
            std::vector<float> h; gen_hadamard(h, g.D);
            std::vector<float> h64; gen_hadamard(h64, 64);
            std::vector<double> t(g.D);
            for (int64_t r = 0; r < (int64_t) kv_size*g.Hkv; ++r) {
                double * x = kd.data() + r*g.D;
                for (int a = 0; a < g.D; ++a) { double acc = 0; for (int b = 0; b < g.D; ++b) acc += (double) h[a*g.D + b]*x[b]; t[a] = acc; }
                std::copy(t.begin(), t.end(), x);
                double * y = vd.data() + r*g.D;
                for (int c = 0; c < g.D; c += 64) {
                    for (int a = 0; a < 64; ++a) { double acc = 0; for (int b = 0; b < 64; ++b) acc += (double) h64[a*64 + b]*y[c + b]; t[a] = acc; }
                    for (int a = 0; a < 64; ++a) y[c + a] = t[a];
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// reference: Jet-Long Eq. 1 with `_reference_jetlong_decode` semantics (near base + grouped distant, LSE merge)
// ---------------------------------------------------------------------------------------------

enum ref_mode { REF_EXACT, REF_F32ANGLE, REF_Q8Q };

// q8_0 round trip of a 256-vector (what ggml's q8_0 x q8_0 vec_dot does to the f32 query)
static void q8_roundtrip(std::vector<double> & x) {
    const int n = (int) x.size();
    std::vector<float> f(x.begin(), x.end()), o(n);
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, n));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, 1, n, nullptr);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), o.data(), n);
    for (int i = 0; i < n; ++i) x[i] = o[i];
}

// returns [D*H] for one query token
// REF_Q8Q (attribution only, quantized-K configs): same Jet-Long math, but the two q8_0 dot products of the
// implementation (near: H*q_base, distant pass-through: H*[0, q_pass]) see the q8_0-rounded query, as ggml does
static void reference_row(const runner & r, const std::vector<double> & kd, const std::vector<double> & vd,
        const tok & t, bool ext, int32_t G, int32_t w0, ref_mode mode, std::vector<double> & out,
        const std::vector<double> * kh = nullptr) {
    const geom & g = r.g;
    const int D = g.D, H = g.H, Hkv = g.Hkv;
    out.assign((size_t) D*H, 0.0);

    std::vector<float> qraw((size_t) D*H);
    gen_vec(qraw.data(), D*H, 0, t.seq, t.pos, 0.5f);

    // keys of the sequence visible to the query
    std::vector<int> keys;
    for (int j = 0; j < r.kv_size; ++j) {
        if (r.cpos[j] >= 0 && r.cseq[j] == t.seq && r.cpos[j] <= t.pos) keys.push_back(j);
    }

    const double scale = 1.0/sqrt((double) D);
    std::vector<double> qb(D), qg(D), kg(D), qn(D), qp(D);
    std::vector<float> hm;
    if (mode == REF_Q8Q) gen_hadamard(hm, D);
    auto had = [&](const std::vector<double> & x, std::vector<double> & y) {
        for (int a = 0; a < D; ++a) { double acc = 0; for (int b = 0; b < D; ++b) acc += (double) hm[a*D + b]*x[b]; y[a] = acc; }
    };

    for (int h = 0; h < H; ++h) {
        const int hk = h/(H/Hkv);

        // query base / grouped
        if (mode == REF_EXACT) {
            for (int d = 0; d < D; ++d) qb[d] = qraw[(size_t) h*D + d];
            qg = qb;
            rot_exact(qb.data(), g, t.pos);
            rot_exact(qg.data(), g, ext ? (double) (t.pos/G) : (double) t.pos);  // direct grouped position
        } else {
            std::vector<float> qf(qraw.begin() + (size_t) h*D, qraw.begin() + (size_t) (h + 1)*D);
            rope_fwd_f32(qf.data(), g, t.pos);                                     // == the stock q_rope
            for (int d = 0; d < D; ++d) qb[d] = qf[d];
            qg = qb;
            if (ext) rot_f32(qg.data(), g, llama_jetlong_delta(t.pos, G));         // correction rotation
            if (mode == REF_Q8Q) {
                had(qb, qn); q8_roundtrip(qn);
                std::vector<double> tp = qb; for (int d = 0; d < g.n_rot; ++d) tp[d] = 0;
                had(tp, qp); q8_roundtrip(qp);
            }
        }

        // two regions, each with its own (max, sum, acc) -> LSE merge (== cute_jetlong_backend.py:93)
        double mx[2] = {-INFINITY, -INFINITY}, sm[2] = {0, 0};
        std::vector<double> acc[2] = { std::vector<double>(D, 0.0), std::vector<double>(D, 0.0) };
        std::vector<double> sc(keys.size());
        std::vector<int>    rg(keys.size());

        for (size_t n = 0; n < keys.size(); ++n) {
            const int j = keys[n];
            const double * kb = kd.data() + ((size_t) j*Hkv + hk)*D;
            const bool near = !ext || (int64_t) t.pos - r.cpos[j] <= w0;
            double s = 0;
            if (mode == REF_Q8Q) {
                const double * kq = kh->data() + ((size_t) j*Hkv + hk)*D;
                if (near) {
                    for (int d = 0; d < D; ++d) s += qn[d]*kq[d];
                } else {
                    for (int d = 0; d < D; ++d) s += qp[d]*kq[d];
                    std::copy(kb, kb + D, kg.begin());
                    rot_f32(kg.data(), g, llama_jetlong_delta(r.cpos[j], G));
                    for (int d = 0; d < g.n_rot; ++d) s += qg[d]*kg[d];
                }
            } else if (near) {
                for (int d = 0; d < D; ++d) s += qb[d]*kb[d];
            } else {
                std::copy(kb, kb + D, kg.begin());
                const int32_t dk = llama_jetlong_delta(r.cpos[j], G);
                if (mode == REF_EXACT) rot_exact(kg.data(), g, dk); else rot_f32(kg.data(), g, dk);
                for (int d = 0; d < D; ++d) s += qg[d]*kg[d];
            }
            sc[n] = s*scale;
            rg[n] = near ? 0 : 1;
            mx[rg[n]] = std::max(mx[rg[n]], sc[n]);
        }
        for (size_t n = 0; n < keys.size(); ++n) {
            const double e = exp(sc[n] - mx[rg[n]]);
            sm[rg[n]] += e;
            const double * vb = vd.data() + ((size_t) keys[n]*Hkv + hk)*D;
            for (int d = 0; d < D; ++d) acc[rg[n]][d] += e*vb[d];
        }
        double lse[2];
        for (int b = 0; b < 2; ++b) lse[b] = sm[b] > 0 ? mx[b] + log(sm[b]) : -INFINITY;
        const double m = std::max(lse[0], lse[1]);
        const double w[2] = { exp(lse[0] - m), exp(lse[1] - m) };
        const double den = w[0] + w[1];
        for (int d = 0; d < D; ++d) {
            double o = 0;
            for (int b = 0; b < 2; ++b) if (sm[b] > 0) o += (w[b]/den)*(acc[b][d]/sm[b]);
            out[(size_t) h*D + d] = o;
        }
    }
}

struct err_acc {
    double max_abs = 0, max_ref = 0;
    void add(const std::vector<double> & ref, const float * got) {
        for (size_t i = 0; i < ref.size(); ++i) {
            max_abs = std::max(max_abs, fabs(ref[i] - (double) got[i]));
            max_ref = std::max(max_ref, fabs(ref[i]));
        }
    }
    double rel() const { return max_ref > 0 ? max_abs/max_ref : 0; }
};

static double max_abs_rows(const std::vector<float> & a, const std::vector<float> & b, int rowlen, const std::vector<int> & rows) {
    double m = 0;
    for (int r : rows) {
        for (int i = 0; i < rowlen; ++i) {
            const double d = fabs((double) a[(size_t) r*rowlen + i] - (double) b[(size_t) r*rowlen + i]);
            if (std::isnan(d)) return INFINITY;
            m = std::max(m, d);
        }
    }
    return m;
}

static int g_fail = 0;
static void verdict(const char * name, bool ok, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("%-58s %s  ", name, ok ? "PASS" : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    fflush(stdout);
    if (!ok) g_fail++;
}

// ---------------------------------------------------------------------------------------------
// schedules
// ---------------------------------------------------------------------------------------------

struct sched_builder {
    std::vector<std::vector<tok>> steps;
    std::map<int, int> next_pos;
    int next_cell = 0;
    // prefill seq s up to (excluding) pos_end in ubatches of n_ub
    void prefill(int s, int pos_end, int n_ub) {
        int & p = next_pos[s];
        while (p < pos_end) {
            std::vector<tok> ub;
            for (int i = 0; i < n_ub && p < pos_end; ++i) ub.push_back({s, p++, next_cell++});
            steps.push_back(ub);
        }
    }
    // interleaved multi-seq prefill: each ubatch carries n_each tokens of every listed seq
    void prefill_mixed(const std::vector<int> & seqs, const std::vector<int> & ends, int n_each) {
        for (;;) {
            std::vector<tok> ub;
            for (size_t k = 0; k < seqs.size(); ++k) {
                int & p = next_pos[seqs[k]];
                for (int i = 0; i < n_each && p < ends[k]; ++i) ub.push_back({seqs[k], p++, next_cell++});
            }
            if (ub.empty()) break;
            steps.push_back(ub);
        }
    }
    // decode n steps; each step one token per listed seq
    void decode(const std::vector<int> & seqs, int n) {
        for (int i = 0; i < n; ++i) {
            std::vector<tok> ub;
            for (int s : seqs) ub.push_back({s, next_pos[s]++, next_cell++});
            steps.push_back(ub);
        }
    }
};

// ---------------------------------------------------------------------------------------------
// T1
// ---------------------------------------------------------------------------------------------

static void t1_case(const char * name, const geom & g, const run_cfg & base, const sched_builder & sb, int kv_size,
        bool expect_active) {
    run_cfg off = base; off.jl = llama_jetlong_cfg{};
    runner ra(g, base, kv_size), rb(g, off, kv_size);
    double worst = 0;
    int n_rows = 0, n_active = 0, n_prefill = 0, n_decode = 0;
    for (const auto & ub : sb.steps) {
        step_out a = ra.step(ub), b = rb.step(ub);
        std::vector<int> rows;
        for (int i = 0; i < (int) ub.size(); ++i) {
            const bool ext = a.active && a.tok_ext[i];
            if (!ext) rows.push_back(i);
        }
        n_active += a.active;
        // prefill-type: some sequence contributes more than one token to the ubatch
        std::map<int, int> per_seq;
        int mx = 0;
        for (const auto & t : ub) mx = std::max(mx, ++per_seq[t.seq]);
        if (mx > 1) n_prefill++; else n_decode++;
        worst = std::max(worst, max_abs_rows(a.out, b.out, g.D*g.H, rows));
        n_rows += (int) rows.size();
    }
    const bool ok = worst == 0.0 && (expect_active ? n_active > 0 : n_active == 0);
    verdict(name, ok, "max-abs %.3g over %d in-window rows, %zu ubatches (%d prefill-type, %d decode-type), active ubatches %d",
            worst, n_rows, sb.steps.size(), n_prefill, n_decode, n_active);
}

// ---------------------------------------------------------------------------------------------
// T2 / T3 helpers
// ---------------------------------------------------------------------------------------------

struct ref_stats { err_acc ext_exact, ext_f32, inw_exact, ext_q8q, inw_q8q; int n_ext = 0, n_inw = 0; };

static void check_vs_ref(const runner & r, const std::vector<tok> & ub, const step_out & so, ref_stats & rs, bool do_f32 = true) {
    std::vector<double> kd, vd, kh, ref;
    r.read_cache(kd, vd, &kh);
    const int row = r.g.D*r.g.H;
    for (int i = 0; i < (int) ub.size(); ++i) {
        const bool ext = so.active && so.tok_ext[i];
        reference_row(r, kd, vd, ub[i], ext, so.G, r.rc.jl.w0, REF_EXACT, ref);
        if (ext) { rs.ext_exact.add(ref, so.out.data() + (size_t) i*row); rs.n_ext++; }
        else     { rs.inw_exact.add(ref, so.out.data() + (size_t) i*row); rs.n_inw++; }
        if (ext && do_f32) {
            reference_row(r, kd, vd, ub[i], ext, so.G, r.rc.jl.w0, REF_F32ANGLE, ref);
            rs.ext_f32.add(ref, so.out.data() + (size_t) i*row);
        }
        if (r.rc.rot) {
            reference_row(r, kd, vd, ub[i], ext, so.G, r.rc.jl.w0, REF_Q8Q, ref, &kh);
            (ext ? rs.ext_q8q : rs.inw_q8q).add(ref, so.out.data() + (size_t) i*row);
        }
    }
}

int main(int argc, char ** argv) {
    bool long_cases = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--long")) long_cases = true;
        if (!strcmp(argv[i], "-t") && i + 1 < argc) g_threads = atoi(argv[++i]);
    }
    ggml_cpu_init();

    const geom g;
    printf("geometry: head %d, n_head %d, n_head_kv %d, n_rot %d, IMRoPE sections [%d,%d,%d,%d], base %.0e, threads %d\n",
            g.D, g.H, g.Hkv, g.n_rot, g.sections[0], g.sections[1], g.sections[2], g.sections[3], g.base, g_threads);

    // small-scale window so that L crosses w, 2w, 3w ... cheaply; the code path is identical for w = 262144
    llama_jetlong_cfg jl;
    jl.w0 = 32;
    jl.w_native = 256;

    const run_cfg c_f16 { "f16",        GGML_TYPE_F16,  GGML_TYPE_F16,  false, jl };
    const run_cfg c_q8  { "q8_0+hadamard", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true,  jl };
    const run_cfg c_f32 { "f32",        GGML_TYPE_F32,  GGML_TYPE_F32,  false, jl };

    // ------------------------------------------------------------------ T1
    printf("\n== T1 in-window identity (flag on vs off, w = %d, w0 = %d) ==\n", jl.w_native, jl.w0);
    for (const run_cfg * c : { &c_q8, &c_f16 }) {
        {   // np 1: prefill 0..191 in ubatches of 64, decode 192..255 (L = 256 = w, the last in-window length)
            sched_builder sb; sb.prefill(0, 192, 64); sb.decode({0}, 64);
            t1_case((std::string("T1a np1 prefill+decode L<=w [") + c->name + "]").c_str(), g, *c, sb, 512, false);
        }
        {   // np 4 mixed: interleaved multi-sequence prefill ubatches, then 4-token decode ubatches
            sched_builder sb; sb.prefill_mixed({0, 1, 2, 3}, {120, 200, 64, 180}, 16); sb.decode({0, 1, 2, 3}, 40);
            t1_case((std::string("T1b np4 mixed prefill+decode L<=w [") + c->name + "]").c_str(), g, *c, sb, 1024, false);
        }
        {   // np 4 mixed with seq 0 ABOVE native: the in-window seqs 1..3 must still be bit-identical (splice)
            sched_builder sb; sb.prefill(0, 300, 64); sb.prefill_mixed({0, 1, 2, 3}, {340, 100, 150, 60}, 8); sb.decode({0, 1, 2, 3}, 20);
            t1_case((std::string("T1c np4, seq0 above native, seqs1-3 in-window [") + c->name + "]").c_str(), g, *c, sb, 1024, true);
        }
    }

    // ------------------------------------------------------------------ T2
    printf("\n== T2 above native vs reference (Eq. 1, _reference_jetlong_decode semantics) ==\n");
    for (const run_cfg * c : { &c_f32, &c_f16, &c_q8 }) {
        // prefill to L = 1100 (G = 5 at the end) in ubatches of 64, then 24 decode steps; every row checked
        sched_builder sb; sb.prefill(0, 1100, 64); sb.decode({0}, 24);
        run_cfg co = *c; co.jl = llama_jetlong_cfg{};
        runner r(g, *c, 1280), ro(g, co, 1280);
        ref_stats rs, rso;
        for (const auto & ub : sb.steps) {
            step_out so = r.step(ub);
            check_vs_ref(r, ub, so, rs);
            // like-for-like floor: the STOCK path (flag off) on the same rows vs plain base-RoPE attention in double
            step_out sf = ro.step(ub);
            check_vs_ref(ro, ub, sf, rso, false);
        }
        const double tol_floor = rso.inw_exact.rel();
        const double tol = std::max(1e-3, 2.0*tol_floor);
        if (c->rot) {
            verdict((std::string("T2 vs Q-quantization-aware reference [") + c->name + "]").c_str(),
                    rs.ext_q8q.rel() <= 1e-3,
                    "above-native rel max-abs %.3e (%d rows) | stock path, same rows, same reference %.3e",
                    rs.ext_q8q.rel(), rs.n_ext, rso.inw_q8q.rel());
        }
        verdict((std::string("T2 prefill+decode to L=1124 (G<=5) [") + c->name + "]").c_str(),
                rs.ext_exact.rel() <= tol,
                "above-native rel max-abs %.3e (abs %.3e, %d rows; fp32-angle ref %.3e) | stock path on the same %d rows vs its own reference %.3e | tol %.1e",
                rs.ext_exact.rel(), rs.ext_exact.max_abs, rs.n_ext, rs.ext_f32.rel(), rso.n_inw, tol_floor, tol);
    }

    // ------------------------------------------------------------------ T3
    printf("\n== T3 epoch crossing: cached per G-epoch vs uncached ==\n");
    for (const run_cfg * c : { &c_q8, &c_f16 }) {
        run_cfg cu = *c; cu.jl.uncached = true;
        // prefill to 500 (G=2), decode across 512 -> 513 (G 2 -> 3), then a prefill ubatch straddling 768 (G -> 4),
        // then decode across 1024 (G 4 -> 5)
        sched_builder sb; sb.prefill(0, 500, 64); sb.decode({0}, 30); sb.prefill(0, 800, 64); sb.decode({0}, 20);
        sb.prefill(0, 1010, 64); sb.decode({0}, 30);
        runner rc_(g, *c, 1280), ru(g, cu, 1280);
        double worst = 0;
        ref_stats rs;
        int crossings = 0; int32_t lastG = 1;
        size_t upd_steady_max = 0, upd_epoch = 0;
        for (const auto & ub : sb.steps) {
            step_out a = rc_.step(ub), b = ru.step(ub);
            std::vector<int> rows(ub.size()); for (int i = 0; i < (int) ub.size(); ++i) rows[i] = i;
            worst = std::max(worst, max_abs_rows(a.out, b.out, g.D*g.H, rows));
            if (a.active && a.G != lastG) { crossings++; upd_epoch = std::max(upd_epoch, a.n_upd); }
            else if (a.active) { upd_steady_max = std::max(upd_steady_max, a.n_upd); }
            if (a.active) lastG = a.G;
            check_vs_ref(rc_, ub, a, rs, false);
        }
        verdict((std::string("T3 cached == uncached across G 2->3->4->5 [") + c->name + "]").c_str(),
                worst == 0.0 && crossings >= 3,
                "max-abs %.3g, %d epoch changes, rows recomputed: <=%zu per steady step vs %zu at an epoch change",
                worst, crossings, upd_steady_max, upd_epoch);
        const double tol = std::max(1e-3, 2.0*rs.inw_exact.rel());
        verdict((std::string("T3 cached vs reference [") + c->name + "]").c_str(), rs.ext_exact.rel() <= tol,
                "rel max-abs %.3e over %d above-native rows (tol %.1e)", rs.ext_exact.rel(), rs.n_ext, tol);
    }

    // ------------------------------------------------------------------ T4: ubatch size sweep x rope type
    printf("\n== T4 ubatch sizes across native (ub 1, 4, 16 < w0 = 32, 64 > w0), decode past native, IMRoPE and NeoX ==\n");
    for (int neox = 0; neox < 2; ++neox) {
        geom g4 = g; g4.neox = neox != 0;
        for (const run_cfg * c : { &c_q8, &c_f16 }) {
            for (int ub_n : { 1, 4, 16, 64 }) {
                run_cfg cu = *c; cu.jl.uncached = true;
                sched_builder sb;
                sb.prefill(0, 224, 64);             // in-window bulk
                sb.prefill(0, 556, ub_n);           // ub_n-token ubatches straddling w = 256 and 2w = 512 (G 1 -> 2 -> 3)
                sb.prefill(0, 560, 4);              // a short tail ubatch beyond native (refined coordinator repro)
                sb.decode({0}, 8);                  // decode past native
                runner ra(g4, *c, 1024), rb(g4, cu, 1024);
                double worst = 0;
                ref_stats rs;
                int n_active = 0;
                for (const auto & ub : sb.steps) {
                    step_out a2 = ra.step(ub), b2 = rb.step(ub);
                    std::vector<int> rows(ub.size()); for (int i = 0; i < (int) ub.size(); ++i) rows[i] = i;
                    worst = std::max(worst, max_abs_rows(a2.out, b2.out, g4.D*g4.H, rows));
                    n_active += a2.active;
                    check_vs_ref(ra, ub, a2, rs, false);
                }
                const double ref_err = c->rot ? rs.ext_q8q.rel() : rs.ext_exact.rel();
                char nm[128]; snprintf(nm, sizeof nm, "T4 %s ub=%d +tail4 +decode8 to L=568 [%s]", neox ? "NeoX" : "IMRoPE", ub_n, c->name.c_str());
                verdict(nm, worst == 0.0 && ref_err <= 1e-3 && n_active > 0,
                        "cached==uncached %.3g | vs %s ref %.3e (%d rows) | %zu ubatches, %d active",
                        worst, c->rot ? "Q-q8-aware" : "exact", ref_err, rs.n_ext, sb.steps.size(), n_active);
            }
        }
    }

    // ------------------------------------------------------------------ real geometry (optional)
    if (long_cases) {
        printf("\n== T2-long real geometry (w = 262144, w0 = 2048), decode at depth ==\n");
        llama_jetlong_cfg jr; jr.w0 = 2048; jr.w_native = 262144;
        struct lc { int L; ggml_type t; bool rot; const char * nm; };
        for (const lc & x : { lc{300000, GGML_TYPE_F16, false, "f16"}, lc{300000, GGML_TYPE_Q8_0, true, "q8_0+hadamard"},
                              lc{600000, GGML_TYPE_F16, false, "f16"} }) {
            run_cfg c { x.nm, x.t, x.t, x.rot, jr };
            runner r(g, c, x.L + 256);
            r.host_fill(x.L - 1, 0);
            sched_builder sb; sb.next_pos[0] = x.L - 1; sb.next_cell = x.L - 1; sb.decode({0}, 2);
            ref_stats rs;
            size_t upd0 = 0, upd1 = 0;
            for (size_t k = 0; k < sb.steps.size(); ++k) {
                step_out so = r.step(sb.steps[k]);
                (k == 0 ? upd0 : upd1) = so.n_upd;
                check_vs_ref(r, sb.steps[k], so, rs);
            }
            char nm[128]; snprintf(nm, sizeof nm, "T2-long decode L=%d (G=%d) [%s]", x.L + 1, llama_jetlong_group(x.L + 1, jr.w_native), x.nm);
            const double gate = x.rot ? rs.ext_q8q.rel() : rs.ext_f32.rel();
            verdict(nm, gate <= 1e-3,
                    "rel max-abs vs fp32-angle ref %.3e; vs exact-angle ref %.3e; vs Q-q8-aware ref %s%.3e | rows recomputed: %zu (epoch build), %zu (next step)",
                    rs.ext_f32.rel(), rs.ext_exact.rel(), x.rot ? "" : "(n/a) ", x.rot ? rs.ext_q8q.rel() : 0.0, upd0, upd1);
        }
    }

    printf("\n%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
