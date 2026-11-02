#include "llama-kv-rows-backend.h"
#include <algorithm>
#include <functional>
#include <unordered_set>

bool llama_kv_rows_backend_compute(ggml_backend_t backend, ggml_context * host_ctx, ggml_cgraph * graph,
        const std::vector<ggml_tensor *> & outputs, std::vector<std::vector<float>> & values) {
    values.clear();
    if (!backend || !host_ctx || !graph || outputs.empty()) return false;
    std::vector<int> indexes;
    for (auto * out : outputs) {
        if (!out || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out)) return false;
        int found = -1;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (ggml_graph_node(graph, i) == out) { found = i; break; }
        }
        if (found < 0 || std::find(indexes.begin(), indexes.end(), found) != indexes.end()) return false;
        indexes.push_back(found);
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_NONE && !ggml_backend_supports_op(backend, node)) return false;
    }
    // ggml_graph_copy requires host tensors to carry a backend buffer even
    // when their bytes are owned by a normal, allocated ggml context. Attach
    // a temporary non-owning buffer only to tensors in that context, then
    // restore every attachment as soon as copying is complete.
    auto * base = static_cast<char *>(ggml_get_mem_buffer(host_ctx));
    const size_t size = ggml_get_mem_size(host_ctx);
    if (!base || !size) return false;
    auto host_buffer = ggml_backend_cpu_buffer_from_ptr(base, size);
    if (!host_buffer) return false;
    std::vector<ggml_tensor *> attached;
    for (auto * t = ggml_get_first_tensor(host_ctx); t; t = ggml_get_next_tensor(host_ctx, t)) {
        if (!t->buffer && !t->view_src && t->data) {
            const auto addr = reinterpret_cast<uintptr_t>(t->data);
            const auto start = reinterpret_cast<uintptr_t>(base);
            if (addr < start || addr - start > size || ggml_nbytes(t) > size - (addr - start)) {
                for (auto * a : attached) a->buffer = nullptr;
                ggml_backend_buffer_free(host_buffer);
                return false;
            }
            t->buffer = host_buffer;
            attached.push_back(t);
        }
    }
    std::unordered_set<const ggml_tensor *> checked;
    std::function<bool(const ggml_tensor *)> bound = [&](const ggml_tensor * t) {
        if (!t || !checked.insert(t).second) return true;
        if (t->view_src) {
            if (!bound(t->view_src)) return false;
        } else if (!t->buffer || !t->data) return false;
        for (const auto * src : t->src) if (!bound(src)) return false;
        return true;
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!bound(ggml_graph_node(graph, i))) {
            for (auto * t : attached) t->buffer = nullptr;
            ggml_backend_buffer_free(host_buffer);
            return false;
        }
    }
    auto copy = ggml_backend_graph_copy(backend, graph);
    for (auto * t : attached) t->buffer = nullptr;
    ggml_backend_buffer_free(host_buffer);
    if (!copy.graph || !copy.buffer) { ggml_backend_graph_copy_free(copy); return false; }
    if (ggml_backend_graph_compute(backend, copy.graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_graph_copy_free(copy);
        return false;
    }
    std::vector<std::vector<float>> result;
    result.reserve(indexes.size());
    for (const auto i : indexes) {
        auto * dst = ggml_graph_node(copy.graph, i);
        if (!dst || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
            ggml_backend_graph_copy_free(copy);
            return false;
        }
        result.emplace_back(ggml_nelements(dst));
        ggml_backend_tensor_get(dst, result.back().data(), 0, ggml_nbytes(dst));
    }
    ggml_backend_synchronize(backend);
    ggml_backend_graph_copy_free(copy);
    values.swap(result);
    return true;
}
