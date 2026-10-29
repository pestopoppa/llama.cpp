#include "llama-mmap-evict.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "mmap eviction test failure: %s\n", message);
        std::abort();
    }
}

struct graph_eviction_state {
    llama_mmap_evict::plan * page_plan = nullptr;
    std::unordered_map<const ggml_tensor *, size_t> boundary;
    std::vector<size_t> evicted_nodes;
    bool invalid_callback_order = false;
};

static bool graph_eviction_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<graph_eviction_state *>(user_data);
    const auto found = state.boundary.find(tensor);
    if (ask) return found != state.boundary.end();
    if (found == state.boundary.end() || !state.page_plan->drop_after(found->second)) {
        state.invalid_callback_order = true;
        return false;
    }
    state.evicted_nodes.push_back(found->second);
    return true;
}

int main() {
#ifndef __linux__
    return 0;
#else
    require(llama_mmap_evict::can_enable(false), "opt-in accepted without existing eval callback");
    require(!llama_mmap_evict::can_enable(true), "opt-in refuses to replace existing eval callback");
    const long page_size_raw = sysconf(_SC_PAGESIZE);
    require(page_size_raw > 0, "page size available");
    const size_t page = static_cast<size_t>(page_size_raw);
    const size_t bytes = page * 6;
    const int fd = static_cast<int>(syscall(SYS_memfd_create, "llama-mmap-evict-test", 0));
    require(fd >= 0, "memfd created");
    require(ftruncate(fd, static_cast<off_t>(bytes)) == 0, "memfd sized");
    auto * writable = static_cast<uint8_t *>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                                   MAP_SHARED, fd, 0));
    require(writable != MAP_FAILED, "memfd mapping created");
    for (size_t i = 0; i < bytes; ++i) {
        writable[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    }
    auto * weight_data = reinterpret_cast<float *>(writable);
    for (size_t i = 0; i < 16; ++i) weight_data[i] = float(i + 1) / 8.0f;
    require(msync(writable, bytes, MS_SYNC) == 0, "synthetic mapping flushed");
    require(mprotect(writable, bytes, PROT_READ) == 0, "mapping made read-only");

    llama_mmap_evict::plan plan({{reinterpret_cast<uintptr_t>(writable), bytes}});
    // A tensor that begins outside a claimed backing interval but extends into
    // it must remain fail-closed. The error-only process map capture should
    // identify the real VMA and its file offset for diagnosis.
    bool overlap_refused = false;
    try {
        llama_mmap_evict::plan partial({{reinterpret_cast<uintptr_t>(writable) + page, page}});
        partial.add_tensor_use(0, writable + page - 64, 128);
    } catch (const std::invalid_argument & error) {
        const std::string message = error.what();
        overlap_refused = message.find("tensor range partially overlaps") != std::string::npos &&
            message.find("all_mappings=") != std::string::npos &&
            message.find("vmas=") != std::string::npos &&
            message.find("file_offset=0x") != std::string::npos;
    }
    require(overlap_refused, "partial crossing refuses and reports actual VMA/file offset metadata");

    // Two aliases share pages 0-1; a third tensor overlaps page 1. Their
    // pages must remain resident until the latest graph use, not the first.
    require(plan.add_tensor_use(1, writable + 100, page * 2 - 100), "first alias use mapped");
    require(plan.add_tensor_use(4, writable + 200, page - 200), "later alias use mapped");
    require(plan.add_tensor_use(2, writable + page + 10, page - 10), "overlapping shared use mapped");
    // Recurrent/shared input use: the same mapped page appears at nodes 0 and 5.
    require(plan.add_tensor_use(0, writable + page * 4, page), "shared input first use mapped");
    require(plan.add_tensor_use(5, writable + page * 4 + 64, page - 64), "shared input last use mapped");
    require(!plan.add_tensor_use(3, reinterpret_cast<void *>(uintptr_t{1}), 8), "foreign pointer ignored");
    plan.finalize(6);

    auto pages_at = [&](size_t node) {
        std::vector<std::pair<size_t, size_t>> pages;
        for (const auto & range : plan.after(node)) {
            require(range.mapping_index == 0, "range belongs to expected mapping");
            require(range.offset % page == 0, "range is page aligned");
            pages.emplace_back(range.offset / page, range.size / page);
        }
        return pages;
    };
    require(pages_at(1).empty(), "page retained past first alias use");
    require((pages_at(2) == std::vector<std::pair<size_t,size_t>>{{1,1}}), "shared page waits for later source use");
    require((pages_at(4) == std::vector<std::pair<size_t,size_t>>{{0,1}}), "alias page released after last use");
    require((pages_at(5) == std::vector<std::pair<size_t,size_t>>{{4,1}}), "shared input page released after last use");

    // A process-local DONTNEED may discard clean mapped pages, but refaults
    // must return byte-identical source data. The backing memfd is never edited
    // after the mapping becomes read-only.
    std::vector<uint8_t> before(writable, writable + bytes);
    require(plan.drop_after(1), "first page advice succeeds");
    require(plan.drop_after(2), "shared page advice succeeds");
    require(plan.drop_after(4), "alias page advice succeeds");
    require(plan.drop_after(5), "shared input page advice succeeds");
    require(std::equal(before.begin(), before.end(), writable), "mapped output is byte-identical after refault");

    // Exercise the same last-use plan through the real ggml CPU scheduler.
    // Two graph leaves alias the same read-only mapped F32 matrix. The mapping
    // must stay live through both matmuls, then the callback may discard it at
    // the later source's graph boundary. The resulting graph output must match
    // an uninstrumented graph bit-for-bit after the page refaults.
    const float input_values[4] = { 0.25f, -0.5f, 1.0f, 2.0f };

    auto make_graph = [&](ggml_context ** ctx_out, ggml_cgraph ** graph_out,
                          ggml_tensor ** output_out, ggml_tensor ** first_weight_out,
                          ggml_tensor ** second_weight_out, ggml_tensor ** input_out) {
        ggml_init_params params = { 1u << 20, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        require(ctx != nullptr, "ggml graph context created");
        ggml_tensor * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
        require(w1 && w2 && x, "ggml graph inputs allocated");
        w1->data = writable;
        w2->data = writable; // a second tensor alias of the same mapped pages
        x->data = const_cast<float *>(input_values);
        ggml_tensor * y1 = ggml_mul_mat(ctx, w1, x);
        ggml_tensor * y2 = ggml_mul_mat(ctx, w2, x);
        ggml_tensor * out = ggml_add(ctx, y1, y2);
        require(out != nullptr, "ggml graph output built");
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, out);
        *ctx_out = ctx;
        *graph_out = graph;
        *output_out = out;
        *first_weight_out = w1;
        *second_weight_out = w2;
        *input_out = x;
    };

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr, "CPU backend initialized");
    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(backends, nullptr, 1, 16, false, false);
    require(scheduler != nullptr, "CPU graph scheduler created");

    ggml_context * baseline_ctx = nullptr;
    ggml_cgraph * baseline_graph = nullptr;
    ggml_tensor * baseline_out = nullptr;
    ggml_tensor * baseline_w1 = nullptr;
    ggml_tensor * baseline_w2 = nullptr;
    ggml_tensor * baseline_x = nullptr;
    make_graph(&baseline_ctx, &baseline_graph, &baseline_out, &baseline_w1, &baseline_w2, &baseline_x);
    require(ggml_backend_sched_alloc_graph(scheduler, baseline_graph), "baseline graph allocated");
    require(ggml_backend_sched_graph_compute(scheduler, baseline_graph) == GGML_STATUS_SUCCESS,
            "baseline graph computed");
    float baseline_values[4] = {};
    ggml_backend_tensor_get(baseline_out, baseline_values, 0, sizeof(baseline_values));
    ggml_backend_sched_reset(scheduler);
    ggml_free(baseline_ctx);

    ggml_context * evict_ctx = nullptr;
    ggml_cgraph * evict_graph = nullptr;
    ggml_tensor * evict_out = nullptr;
    ggml_tensor * evict_w1 = nullptr;
    ggml_tensor * evict_w2 = nullptr;
    ggml_tensor * evict_x = nullptr;
    make_graph(&evict_ctx, &evict_graph, &evict_out, &evict_w1, &evict_w2, &evict_x);
    llama_mmap_evict::plan graph_plan({{reinterpret_cast<uintptr_t>(writable), bytes}});
    const int graph_nodes = ggml_graph_n_nodes(evict_graph);
    std::vector<size_t> matmul_nodes;
    for (int i = 0; i < graph_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(evict_graph, i);
        if (node->op == GGML_OP_MUL_MAT) matmul_nodes.push_back(static_cast<size_t>(i));
        for (size_t source = 0; source < GGML_MAX_SRC; ++source) {
            const ggml_tensor * src = node->src[source];
            if (src && src->data) graph_plan.add_tensor_use(static_cast<size_t>(i), src->data, ggml_nbytes(src));
        }
    }
    require(matmul_nodes.size() == 2, "graph contains two aliased matmul uses");
    graph_plan.finalize(static_cast<size_t>(graph_nodes));
    require(graph_plan.after(matmul_nodes[0]).empty(), "first aliased use does not evict shared page");
    require(!graph_plan.after(matmul_nodes[1]).empty(), "second aliased use is the shared page last-use boundary");
    graph_eviction_state callback_state;
    callback_state.page_plan = &graph_plan;
    for (int i = 0; i < graph_nodes; ++i) {
        if (!graph_plan.after(static_cast<size_t>(i)).empty()) {
            callback_state.boundary.emplace(ggml_graph_node(evict_graph, i), static_cast<size_t>(i));
        }
    }
    require(!callback_state.boundary.empty(), "graph has mapped-page last-use boundary");
    require(ggml_backend_sched_alloc_graph(scheduler, evict_graph), "eviction graph allocated");
    ggml_backend_sched_set_eval_callback(scheduler, graph_eviction_callback, &callback_state);
    require(ggml_backend_sched_graph_compute(scheduler, evict_graph) == GGML_STATUS_SUCCESS,
            "eviction graph computed");
    require(!callback_state.invalid_callback_order, "eviction callback ran at a planned boundary");
    require(callback_state.evicted_nodes.size() == callback_state.boundary.size(),
            "all graph eviction boundaries were observed");
    require(callback_state.evicted_nodes.size() == 1 && callback_state.evicted_nodes[0] == matmul_nodes[1],
            "callback evicts only after the second aliased source use");
    float evicted_values[4] = {};
    ggml_backend_tensor_get(evict_out, evicted_values, 0, sizeof(evicted_values));
    for (size_t i = 0; i < 4; ++i) {
        require(evicted_values[i] == baseline_values[i], "ggml output is bit-identical after last-use page eviction");
    }
    ggml_backend_sched_free(scheduler);
    ggml_backend_free(backend);
    ggml_free(evict_ctx);
    require(munmap(writable, bytes) == 0, "mapping released");
    require(close(fd) == 0, "memfd closed");
    std::puts("mmap eviction last-use alias/shared/output test passed");
    return 0;
#endif
}
