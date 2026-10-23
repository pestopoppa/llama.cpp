// Model-free test of the eval callback (moe-trace-tap.h) against REAL ggml tensors (INF-72 RT-3).
//
// Builds exactly what src/llama-graph.cpp builds - ggml_argsort_top_k(), a ggml_view_4d of the
// [n_expert, n_tokens] argsort result, cb-named "ffn_moe_topk-<il>" with the argsort parent named
// "ffn_moe_argsort-<il>" - allocates it in a CPU buffer, writes a known argsort matrix into the parent and
// calls moe_trace_cb() on the view. No graph is computed and no GGUF is loaded. What this proves, on the real
// view layout (ne[0]=k, nb[1]=n_expert*4), is that the callback reads the SELECTED experts and not whole
// argsort rows, takes n_experts from the graph, and refuses what it cannot read.
//
// Links ggml-base only (ggml_backend_cpu_buffer_type, ggml_backend_alloc_ctx_tensors_from_buft,
// ggml_backend_tensor_get/set live there).

#include "../moe-trace-tap.h"

#include <cstdio>
#include <vector>

using namespace moe_trace;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d  %s  [%s]\n", __FILE__, __LINE__, msg, #cond); g_fail++; } } while (0)

struct fixture {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         probs = nullptr;
    ggml_tensor *         topk = nullptr;       // the view
    ggml_tensor *         argsort = nullptr;    // its parent
    std::vector<int32_t>  matrix;               // [n_tokens][n_experts] descending order

    fixture(int n_experts, int k, int n_tokens, int il, uint64_t seed) {
        ggml_init_params ip = { /*mem_size*/ 16u << 20, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        ctx = ggml_init(ip);
        probs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, n_tokens);
        topk = ggml_argsort_top_k(ctx, probs, k);                  // [k, n_tokens] VIEW of the [n_experts, n_tokens] argsort
        argsort = topk->view_src;
        char nm[64];
        std::snprintf(nm, sizeof nm, "ffn_moe_argsort-%d", il); ggml_set_name(argsort, nm);
        std::snprintf(nm, sizeof nm, "ffn_moe_topk-%d", il);    ggml_set_name(topk, nm);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        matrix = detail::synth_argsort(n_experts, n_tokens, seed);
        ggml_backend_tensor_set(argsort, matrix.data(), 0, matrix.size() * sizeof(int32_t));
    }
    ~fixture() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
};

int main() {
    // 1. the real layout: a strided view, hazard live
    {
        fixture f(256, 8, 512, 5, 11);
        CHECK(f.topk->op == GGML_OP_VIEW && f.topk->view_src == f.argsort && f.argsort->op == GGML_OP_ARGSORT, "fixture mirrors ggml_argsort_top_k");
        CHECK(f.topk->ne[0] == 8 && f.topk->ne[1] == 512, "view is [k, n_tokens]");
        CHECK(f.topk->nb[1] == 256 * sizeof(int32_t) && f.topk->nb[1] != 8 * sizeof(int32_t), "the view is NOT contiguous: nb[1] is the parent's row");

        tap_state st;
        CHECK(moe_trace_cb(f.topk, true, &st), "ask: wants ffn_moe_topk-5");
        CHECK(!moe_trace_cb(f.argsort, true, &st), "ask: SRP off, ffn_moe_argsort is not wanted");
        CHECK(moe_trace_cb(f.topk, false, &st), "callback returns true");
        CHECK(st.fatal.empty(), st.fatal.c_str());
        CHECK(st.cur.count(5) == 1, "layer 5 recorded from the name");
        const layer_tap & lt = st.cur[5];
        CHECK(lt.have_topk && lt.k == 8 && lt.n_rows == 512 && lt.n_experts == 256, "k, rows and n_experts (from the argsort parent) read from the graph");
        bool exact = lt.ids.size() == 512u * 8;
        for (int t = 0; exact && t < 512; t++) for (int j = 0; j < 8; j++) exact &= lt.ids[(size_t) t * 8 + j] == f.matrix[(size_t) t * 256 + j];
        CHECK(exact, "ids are the first k of EVERY argsort row (row by row at i*nb[1])");
        CHECK(st.stride_note.find("LIVE") != std::string::npos, "the stride hazard is reported as live");

        // the same tensor read flat would have been caught: feed it to the guard
        std::vector<int16_t> flat(512 * 8);
        for (size_t i = 0; i < flat.size(); i++) flat[i] = (int16_t) f.matrix[i];
        CHECK(hazard_check(flat.data(), 512, 8, 256).fatal, "a flat read of this very tensor trips the guard");
        CHECK(!hazard_check(lt.ids.data(), 512, 8, 256).fatal, "the callback's read does not");

        // and assemble_forward accepts the whole forward of a 1-layer model
        std::vector<layer_info> layers; bool known = false; assembled_forward af; guard_stats gs; std::string msg;
        CHECK(assemble_forward(st.cur, 512, true, false, layers, known, af, gs, msg) == RC_DONE, msg.c_str());
        CHECK(layers.size() == 1 && layers[0].il == 5 && layers[0].n_experts == 256 && layers[0].top_k == 8, "layer table from the graph");
        CHECK(gs.checks == 1, "guard evaluated the prefill forward");
    }
    // 2. 512 experts / top-10 (the Q38FN qwen4exp geometry) and ids above 255
    {
        fixture f(512, 10, 512, 47, 12);
        tap_state st;
        moe_trace_cb(f.topk, false, &st);
        CHECK(st.fatal.empty() && st.cur[47].n_experts == 512 && st.cur[47].k == 10, "qwen4exp geometry");
        int16_t mx = 0; for (int16_t v : st.cur[47].ids) mx = std::max(mx, v);
        CHECK(mx > 255, "ids above 255 survive (int8 and uint8 would not)");
    }
    // 3. SRP: the argsort parent is tapped too, in full
    {
        fixture f(128, 6, 64, 3, 13);
        tap_state st; st.srp = true;
        CHECK(moe_trace_cb(f.argsort, true, &st), "ask: SRP on wants ffn_moe_argsort-3");
        moe_trace_cb(f.argsort, false, &st); moe_trace_cb(f.topk, false, &st);
        const layer_tap & lt = st.cur[3];
        CHECK(lt.have_order && lt.order.size() == 64u * 128, "full order captured");
        bool exact = true; for (size_t i = 0; i < lt.order.size(); i++) exact &= lt.order[i] == f.matrix[i];
        CHECK(exact, "argsort order is exact");
        std::vector<layer_info> layers; bool known = false; assembled_forward af; guard_stats gs; std::string msg;
        CHECK(assemble_forward(st.cur, 64, true, true, layers, known, af, gs, msg) == RC_DONE, msg.c_str());
        CHECK(af.order.size() == 64u * 1 * 128, "SRP order packed [n][layers][max_e]");
    }
    // 4. refusals
    {
        fixture f(64, 4, 16, 2, 14);
        tap_state st;
        moe_trace_cb(f.topk, false, &st); moe_trace_cb(f.topk, false, &st);
        CHECK(!st.fatal.empty() && st.fatal.find("fired twice") != std::string::npos, "second firing in one forward is fatal");
    }
    {
        fixture f(64, 4, 16, 2, 15);
        ggml_set_name(f.topk, "ffn_moe_topk-2 (copy)");
        tap_state st;
        CHECK(!moe_trace_cb(f.topk, true, &st), "a scheduler copy is not tapped");
        ggml_set_name(f.topk, "ffn_moe_topk-");  CHECK(!moe_trace_cb(f.topk, true, &st), "empty layer number is not tapped");
        ggml_set_name(f.topk, "ffn_moe_topk-x");  CHECK(!moe_trace_cb(f.topk, true, &st), "non-numeric layer is not tapped");
    }
    {
        ggml_init_params ip = { 1u << 20, nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * bad = ggml_new_tensor_2d(c, GGML_TYPE_F32, 8, 8);
        ggml_set_name(bad, "ffn_moe_topk-0");
        // an I32 tensor with no argsort parent (selected_experts_in path) needs the model fallback
        ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 4, 8);
        ggml_set_name(ids, "ffn_moe_topk-1");
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, ggml_backend_cpu_buffer_type());
        tap_state st;
        moe_trace_cb(bad, false, &st);
        CHECK(!st.fatal.empty() && st.fatal.find("expected I32") != std::string::npos, "non-I32 tensor is refused");
        std::vector<int32_t> v(32); for (int i = 0; i < 32; i++) v[i] = i % 16;
        ggml_backend_tensor_set(ids, v.data(), 0, v.size() * 4);
        tap_state st2;
        moe_trace_cb(ids, false, &st2);
        CHECK(!st2.fatal.empty() && st2.fatal.find("cannot determine n_experts") != std::string::npos, "no argsort parent and no fallback: refused");
        tap_state st3; st3.model_n_expert_fallback = 16;
        moe_trace_cb(ids, false, &st3);
        CHECK(st3.fatal.empty() && st3.cur[1].n_experts == 16, "model metadata fallback accepted");
        ggml_backend_buffer_free(b); ggml_free(c);
    }
    // 5. a forward with a missing layer is refused, never written short
    {
        std::map<int, layer_tap> taps;
        std::vector<layer_info> layers = {{0, 256, 8}, {1, 256, 8}}; bool known = true;
        layer_tap a; a.k = 8; a.n_rows = 4; a.n_experts = 256; a.ids.assign(32, 1); a.have_topk = true;
        taps[0] = a;
        assembled_forward af; guard_stats gs; std::string msg;
        CHECK(assemble_forward(taps, 4, true, false, layers, known, af, gs, msg) == RC_INCOMPLETE && msg.find("short trace") != std::string::npos, "missing layer: RC_INCOMPLETE");
        taps[1] = a; taps[1].n_rows = 3;
        CHECK(assemble_forward(taps, 4, true, false, layers, known, af, gs, msg) == RC_INCOMPLETE && msg.find("expected 4 rows") != std::string::npos, "short row count (pruned last layer): RC_INCOMPLETE");
        std::map<int, layer_tap> none;
        CHECK(assemble_forward(none, 1, false, false, layers, known, af, gs, msg) == RC_INCOMPLETE, "no taps at all (fused decode bypass): RC_INCOMPLETE");
    }
    // 6. the hazard produced by the DRIVER'S OWN data path: a flat-filled tap on a prefill forward is RC_HAZARD
    {
        const int ne = 256, k = 8, n = 512;
        std::vector<int32_t> mem = detail::synth_argsort(ne, n, 21);
        layer_tap lt; lt.k = k; lt.n_rows = n; lt.n_experts = ne; lt.have_topk = true; lt.ids.resize((size_t) n * k);
        for (size_t i = 0; i < lt.ids.size(); i++) lt.ids[i] = (int16_t) mem[i];     // the forbidden flat read
        std::map<int, layer_tap> taps; taps[0] = lt;
        std::vector<layer_info> layers; bool known = false; assembled_forward af; guard_stats gs; std::string msg;
        CHECK(assemble_forward(taps, n, true, false, layers, known, af, gs, msg) == RC_HAZARD && msg.find("STRIDE-HAZARD") != std::string::npos, "flat-filled tap on prefill: RC_HAZARD");
        known = false; layers.clear();
        CHECK(assemble_forward(taps, n, false, false, layers, known, af, gs, msg) == RC_DONE, "the guard judges prefill forwards only");
    }
    if (g_fail) { std::fprintf(stderr, "%d check(s) FAILED\n", g_fail); return 1; }
    std::printf("test-moe-trace-tap: all checks passed\n");
    return 0;
}
