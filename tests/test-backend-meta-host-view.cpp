// Meta host-view regression test for 9422960da ("meta: handle bounded host-backed views in mixed subgraphs").
//
// The test compiles the Meta backend source into itself so it can reach the private helper
// ggml_backend_meta_is_foreign_host_view() and the private Meta split-state API; it links against
// the ggml libraries of the image under test (same source, same ABI).
//
// Modes (exactly one of the device names must enumerate in the linked ggml):
//   test-backend-meta-host-view [--device NAME] [--unit] [--reject]
//     default      mixed Meta subgraph: partial two-way MUL_MAT on a Meta device built from NAME twice,
//                  followed by a trailing host-backed RESHAPE of a CPU MEAN_D1 result. Must PASS:
//                  the reduction is correct on both shards and the host view is untouched.
//     --reject     same graph, but the trailing node is the host MEAN_D1 itself (foreign compute).
//                  Must ABORT (the Meta backend may not project foreign compute); a return is a FAIL.
//     --unit       bounds checks of ggml_backend_meta_is_foreign_host_view() on CPU buffers.
//
// Build with -DMETA_SOURCE="<path>" to compile a different Meta source (negative control with the
// pre-fix source; --unit is unavailable there, define META_BASELINE).
#ifndef META_SOURCE
#define META_SOURCE "../ggml/src/ggml-backend-meta.cpp"
#endif
#include META_SOURCE
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static ggml_backend_meta_split_state split_axis0(const ggml_tensor *, void *) {
    ggml_backend_meta_split_state s{};
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.ne[0] = 2; s.ne[1] = 2; s.nr[0] = 1; s.n_segments = 1;
    return s;
}

static void report_abort(const char * message) { std::fprintf(stderr, "META_TEST_ABORT:%s\n", message); }

#ifndef META_BASELINE
static int run_unit() {
    ggml_backend_buffer_t buf   = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096);
    ggml_backend_buffer_t other = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096);
    require(buf && other, "CPU buffers missing");
    ggml_tensor src{};
    src.op = GGML_OP_MEAN_D1;
    src.type = GGML_TYPE_F32;
    src.ne[0] = 16; src.ne[1] = src.ne[2] = src.ne[3] = 1;
    src.nb[0] = 4; src.nb[1] = src.nb[2] = src.nb[3] = 64;
    src.buffer = buf;
    src.data = ggml_backend_buffer_get_base(buf);
    ggml_tensor view{};
    view.view_src = &src;
    view.type = GGML_TYPE_F32;
    view.ne[0] = 8; view.ne[1] = view.ne[2] = view.ne[3] = 1;
    view.nb[0] = 4; view.nb[1] = view.nb[2] = view.nb[3] = 32;
    view.buffer = buf;
    view.data = src.data;
    for (ggml_op op : { GGML_OP_VIEW, GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_TRANSPOSE }) {
        view.op = op;
        require(ggml_backend_meta_is_foreign_host_view(&view), "valid host view rejected");
    }
    view.op = GGML_OP_ADD;
    require(!ggml_backend_meta_is_foreign_host_view(&view), "host compute accepted as a view");
    view.op = GGML_OP_RESHAPE;
    view.buffer = other;
    require(!ggml_backend_meta_is_foreign_host_view(&view), "view in a different buffer accepted");
    view.buffer = buf;
    view.view_offs = 4;
    require(!ggml_backend_meta_is_foreign_host_view(&view), "data/offset mismatch accepted");
    view.data = (char *) src.data + view.view_offs;
    require(ggml_backend_meta_is_foreign_host_view(&view), "valid offset view rejected");
    view.view_offs = 60;
    view.data = (char *) src.data + view.view_offs;
    require(!ggml_backend_meta_is_foreign_host_view(&view), "out-of-range view accepted");
    view.view_src = nullptr;
    require(!ggml_backend_meta_is_foreign_host_view(&view), "view without a source accepted");
    ggml_backend_buffer_free(other);
    ggml_backend_buffer_free(buf);
    std::puts("PASS unit: foreign host-view bounds (4 view ops, compute, foreign buffer, offset, range, no source)");
    return 0;
}
#endif

static int run_mixed(const char * device_name, bool reject) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device_name);
    require(dev != nullptr, "requested device does not enumerate in the linked ggml");
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu_dev != nullptr, "CPU device missing");
    std::fprintf(stderr, "META_TEST_DEVICE name=%s description=%s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    ggml_backend_dev_t devices[] = { dev, dev };
    ggml_backend_t meta = ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, split_axis0, nullptr), nullptr);
    ggml_backend_t cpu  = ggml_backend_dev_init(cpu_dev, nullptr);
    require(meta && cpu, "backend init failed");

    ggml_init_params p = { 8 * 1024 * 1024, nullptr, true };
    ggml_context * weights = ggml_init(p);
    ggml_context * work    = ggml_init(p);
    ggml_context * host    = ggml_init(p);
    require(weights && work && host, "context init failed");

    ggml_tensor * a = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 4, 1);
    ggml_tensor * b = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 4, 1);
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(weights, meta);
    require(wb != nullptr, "weight allocation failed");
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const float av[] = { 1, 2, 3, 4 }, bv[] = { 5, 6, 7, 8 };
    ggml_backend_tensor_set(a, av, 0, sizeof av);
    ggml_backend_tensor_set(b, bv, 0, sizeof bv);

    ggml_tensor * dot = ggml_mul_mat(work, a, b);
    ggml_cgraph * graph = ggml_new_graph_custom(work, 32, false);
    ggml_build_forward_expand(graph, dot);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(meta));
    require(alloc && ggml_gallocr_alloc_graph(alloc, graph), "meta graph allocation failed");
    require(ggml_backend_meta_get_split_state(dot, false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL,
            "graph did not produce a partial matmul");

    ggml_tensor * x    = ggml_new_tensor_2d(host, GGML_TYPE_F32, 2, 2);
    ggml_tensor * mean = ggml_mean_d1(host, x);
    ggml_tensor * view = ggml_reshape_1d(host, mean, 2);
    ggml_cgraph * hg = ggml_new_graph_custom(host, 32, false);
    ggml_build_forward_expand(hg, view);
    ggml_backend_buffer_t hb = ggml_backend_alloc_ctx_tensors(host, cpu);
    require(hb != nullptr, "host allocation failed");
    const float hv[] = { 1, 2, 3, 4 };
    ggml_backend_tensor_set(x, hv, 0, sizeof hv);
    require(ggml_backend_graph_compute(cpu, hg) == GGML_STATUS_SUCCESS, "host mean failed");
    float expected_host[2];
    ggml_backend_tensor_get(view, expected_host, 0, sizeof expected_host);
    require(expected_host[0] == 2 && expected_host[1] == 3, "host mean wrong");

    // The scheduler does not split at view ops, so a host view can trail a Meta subgraph.
    ggml_tensor * tail = reject ? mean : view;
    ggml_graph_add_node(graph, tail);
    const size_t pos = ggml_hash_insert(&graph->visited_hash_set, tail);
    graph->use_counts[pos] = 0;
    ggml_set_abort_callback(report_abort);
    std::fprintf(stderr, "META_TEST_READY device=%s partial=1 host_mean=2,3 tail=%s\n", device_name,
                 reject ? "foreign-compute" : "host-view");
    require(ggml_backend_graph_compute(meta, graph) == GGML_STATUS_SUCCESS, "Meta graph compute failed");
    require(!reject, "foreign host compute was accepted by the Meta backend");

    for (size_t j = 0; j < 2; ++j) {
        const ggml_tensor * simple = ggml_backend_meta_buffer_simple_tensor(dot, j);
        require(simple != nullptr, "missing per-shard result tensor");
        float actual = 0;
        ggml_backend_tensor_get(simple, &actual, 0, sizeof actual);
        std::fprintf(stderr, "META_TEST_SHARD j=%zu dot=%.6f\n", j, actual);
        require(std::fabs(actual - 70.0f) < 1e-5f, "partial reduction produced a wrong dot product");
    }
    float observed[2];
    ggml_backend_tensor_get(view, observed, 0, sizeof observed);
    require(observed[0] == expected_host[0] && observed[1] == expected_host[1], "host view changed");
    std::printf("PASS mixed Meta(%s x2)/host trailing view and two-shard partial reduction\n", device_name);

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(hb);
    ggml_backend_buffer_free(wb);
    ggml_free(host); ggml_free(work); ggml_free(weights);
    ggml_backend_free(cpu); ggml_backend_free(meta);
    return 0;
}

int main(int argc, char ** argv) {
    std::string device = "CPU";
    bool unit = false, reject = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = argv[++i];
        } else if (std::strcmp(argv[i], "--unit") == 0) {
            unit = true;
        } else if (std::strcmp(argv[i], "--reject") == 0) {
            reject = true;
        } else {
            std::fprintf(stderr, "usage: %s [--device NAME] [--unit] [--reject]\n", argv[0]);
            return 2;
        }
    }
    if (unit) {
#ifdef META_BASELINE
        std::fprintf(stderr, "--unit is unavailable in a META_BASELINE build\n");
        return 2;
#else
        return run_unit();
#endif
    }
    return run_mixed(device.c_str(), reject);
}
