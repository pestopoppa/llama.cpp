// moe-trace-tap.h - the ggml-facing half of llama-moe-trace: the backend-scheduler eval callback.
// Depends on ggml only (not llama, not common), so tests/test-moe-trace-tap.cpp can drive it with real ggml
// view tensors built by ggml_argsort_top_k() and no model.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "moe-trace-core.h"

#include <cstdlib>
#include <map>
#include <string>

namespace moe_trace {

struct tap_state {
    bool srp = false;
    std::map<int, layer_tap> cur;        // il -> taps of the forward in flight
    std::string fatal;                   // first fatal condition seen in the callback
    int  fatal_rc = RC_INCOMPLETE;
    bool stride_logged = false;
    std::string stride_note;             // what the first top-k tensor looked like (the driver logs it)
    int  model_n_expert_fallback = 0;    // <arch>.expert_count, used only when the graph gives no argsort parent
};

// "ffn_moe_topk-12" -> 12; rejects "ffn_moe_topk-12 (copy)", "ffn_moe_topk-", "ffn_moe_topk-x"
inline bool parse_name(const char * name, const char * prefix, int & il) {
    const size_t pl = std::strlen(prefix);
    if (std::strncmp(name, prefix, pl) != 0) return false;
    const char * d = name + pl;
    if (!*d) return false;
    for (const char * c = d; *c; c++) if (*c < '0' || *c > '9') return false;
    il = std::atoi(d);
    return true;
}

// ggml_backend_sched_eval_callback. ask=true: "do you want this node's data?"; ask=false: the node has been computed.
inline bool moe_trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    int il = -1;
    auto * st = (tap_state *) user_data;
    const bool is_topk = parse_name(t->name, "ffn_moe_topk-", il);
    const bool is_order = !is_topk && st->srp && parse_name(t->name, "ffn_moe_argsort-", il);
    if (ask) return is_topk || is_order;
    if (!is_topk && !is_order) return true;
    if (!st->fatal.empty()) return true;   // already failing: let the forward finish, the driver stops after llama_decode

    auto fail = [&](const std::string & why, int rc) { st->fatal = why; st->fatal_rc = rc; return true; };
    const std::string nm = t->name;

    if (t->type != GGML_TYPE_I32)       return fail(nm + ": expected I32, got " + ggml_type_name(t->type), RC_INCOMPLETE);
    if (t->nb[0] != sizeof(int32_t))    return fail(nm + ": element stride nb[0]=" + std::to_string(t->nb[0]) + " is not 4", RC_INCOMPLETE);
    if (t->ne[2] != 1 || t->ne[3] != 1) return fail(nm + ": unexpected batch dims ne[2]/ne[3]", RC_INCOMPLETE);

    strided_i32 v;
    v.k = t->ne[0]; v.n_rows = t->ne[1]; v.nb1 = t->nb[1];
    layer_tap & lt = st->cur[il];
    if ((is_topk && lt.have_topk) || (is_order && lt.have_order))
        return fail(nm + " fired twice in one forward pass (a second graph/ffn per layer?)", RC_INCOMPLETE);

    // n_experts per layer from the graph: the top-k tensor is a view of the [n_expert, n_tokens] argsort
    int64_t n_experts = 0;
    if (is_order) {
        n_experts = t->ne[0];
    } else if (t->view_src && t->view_src->op == GGML_OP_ARGSORT) {
        n_experts = t->view_src->ne[0];
        if (t->view_offs != 0) return fail(nm + ": view offset != 0, layout unknown", RC_INCOMPLETE);
    } else {
        n_experts = st->model_n_expert_fallback;   // selected_experts_in path: no argsort parent
    }
    if (n_experts <= 0 || v.k > n_experts) return fail(nm + ": cannot determine n_experts for this layer", RC_INCOMPLETE);

    if (is_topk && !st->stride_logged) {
        st->stride_note = nm + " ne=[" + std::to_string(v.k) + "," + std::to_string(v.n_rows) + "] nb[1]=" + std::to_string(v.nb1)
            + " bytes, row payload " + std::to_string((size_t) v.k * 4) + " bytes -> stride hazard "
            + (v.nb1 != (size_t) v.k * 4 ? "LIVE (view of the argsort rows)" : "absent (contiguous)") + "; reading row by row at i*nb[1]";
        st->stride_logged = true;
    }

    read_fn rd = [t](void * dst, size_t off, size_t sz) { ggml_backend_tensor_get(t, dst, off, sz); };
    std::vector<int32_t> raw;
    read_rows_strided(v, rd, raw);       // RT-3: row by row, never a flat read
    std::string err;
    std::vector<int16_t> n16;
    if (!narrow_ids(raw, (int) n_experts, n16, err)) return fail(nm + ": " + err, RC_INCOMPLETE);

    lt.k = v.k; lt.n_rows = v.n_rows; lt.n_experts = n_experts;
    if (is_topk) { lt.ids = std::move(n16); lt.have_topk = true; }
    else         { lt.order = std::move(n16); lt.have_order = true; }
    return true;
}

} // namespace moe_trace
