#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct shape4 {
    int64_t ne0, ne1, ne2, ne3;
    size_t nb1, nb2, nb3;
};

size_t span_bytes(ggml_type type, const shape4 & s, size_t offset = 0) {
    return offset + ggml_row_size(type, s.ne0) +
        (size_t) (s.ne1 - 1) * s.nb1 +
        (size_t) (s.ne2 - 1) * s.nb2 +
        (size_t) (s.ne3 - 1) * s.nb3;
}

size_t row_offset(const shape4 & s, int64_t i1, int64_t i2, int64_t i3, size_t base = 0) {
    return base + (size_t) i1*s.nb1 + (size_t) i2*s.nb2 + (size_t) i3*s.nb3;
}

ggml_tensor * backing(ggml_context * ctx, ggml_type type, int64_t ne0, size_t bytes) {
    const size_t rs = ggml_row_size(type, ne0);
    return ggml_new_tensor_2d(ctx, type, ne0, (int64_t) ((bytes + rs - 1) / rs));
}

void fill_pattern(std::vector<uint8_t> & data, uint32_t state) {
    for (auto & value : data) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = (uint8_t) (state >> 16);
    }
}

bool run_separate_case(const char * name, ggml_type type, const shape4 & src_shape, const shape4 & dst_shape, int threads) {
    const size_t src_logical = ggml_row_size(type, src_shape.ne0) *
        (size_t) src_shape.ne1 * src_shape.ne2 * src_shape.ne3;
    const size_t dst_logical = ggml_row_size(type, dst_shape.ne0) *
        (size_t) dst_shape.ne1 * dst_shape.ne2 * dst_shape.ne3;
    const size_t src_bytes = std::max(span_bytes(type, src_shape), src_logical) + 64;
    const size_t dst_bytes = std::max(span_bytes(type, dst_shape), dst_logical) + 64;
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * src_base = backing(ctx.get(), type, src_shape.ne0, src_bytes);
    ggml_tensor * dst_base = backing(ctx.get(), type, dst_shape.ne0, dst_bytes);
    ggml_tensor * src = ggml_view_4d(ctx.get(), src_base, src_shape.ne0, src_shape.ne1, src_shape.ne2, src_shape.ne3,
                                    src_shape.nb1, src_shape.nb2, src_shape.nb3, 0);
    ggml_tensor * dst = ggml_view_4d(ctx.get(), dst_base, dst_shape.ne0, dst_shape.ne1, dst_shape.ne2, dst_shape.ne3,
                                    dst_shape.nb1, dst_shape.nb2, dst_shape.nb3, 0);
    ggml_tensor * out = ggml_cpy(ctx.get(), src, dst);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) return false;

    std::vector<uint8_t> src_data(ggml_nbytes(src_base));
    std::vector<uint8_t> got(ggml_nbytes(dst_base), 0xa5);
    std::vector<uint8_t> expected = got;
    fill_pattern(src_data, 0x9e3779b9u);
    const size_t rs = ggml_row_size(type, src_shape.ne0);
    for (int64_t i3 = 0; i3 < src_shape.ne3; ++i3)
        for (int64_t i2 = 0; i2 < src_shape.ne2; ++i2)
            for (int64_t i1 = 0; i1 < src_shape.ne1; ++i1)
                memcpy(expected.data() + row_offset(dst_shape, i1, i2, i3),
                       src_data.data() + row_offset(src_shape, i1, i2, i3), rs);

    ggml_backend_tensor_set(src_base, src_data.data(), 0, src_data.size());
    ggml_backend_tensor_set(dst_base, got.data(), 0, got.size());
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_tensor_get(dst_base, got.data(), 0, got.size());
    const bool ok = got == expected;
    std::printf("case=%s type=%s threads=%d bytes=%zu exact=%d\n", name, ggml_type_name(type), threads, rs, ok);
    return ok;
}

bool run_alias_case() {
    constexpr int64_t ne0 = 256;
    const size_t rs = ggml_row_size(GGML_TYPE_F32, ne0);
    const size_t plane_stride = rs + 64;
    const shape4 shape = { ne0, 1, 4, 1, rs, plane_stride, 4*plane_stride };
    const size_t dst_offset = plane_stride;
    const size_t bytes = span_bytes(GGML_TYPE_F32, shape, dst_offset) + 64;
    ggml_init_params ip = { ggml_tensor_overhead()*7 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * base = backing(ctx.get(), GGML_TYPE_F32, ne0, bytes);
    ggml_tensor * src = ggml_view_4d(ctx.get(), base, ne0, 1, 4, 1, rs, plane_stride, 4*plane_stride, 0);
    ggml_tensor * dst = ggml_view_4d(ctx.get(), base, ne0, 1, 4, 1, rs, plane_stride, 4*plane_stride, dst_offset);
    ggml_tensor * out = ggml_cpy(ctx.get(), src, dst);
    ggml_cgraph * graph = ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph, out);
    ggml_backend_ptr backend(ggml_backend_cpu_init()); ggml_backend_cpu_set_n_threads(backend.get(), 48);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) return false;
    std::vector<uint8_t> initial(ggml_nbytes(base)), expected, got;
    fill_pattern(initial, 0x243f6a88u);
    expected=initial; got=initial;
    for (int64_t i2=0;i2<4;++i2) memcpy(expected.data()+dst_offset+(size_t)i2*plane_stride, expected.data()+(size_t)i2*plane_stride, rs);
    ggml_backend_tensor_set(base,initial.data(),0,initial.size());
    if (ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS) return false;
    ggml_backend_tensor_get(base,got.data(),0,got.size());
    const bool ok=got==expected; std::printf("case=alias-forward exact=%d\n",ok); return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    const size_t f32_rs = ggml_row_size(GGML_TYPE_F32, 1048576);
    const shape4 f32_1 = {1048576,1,1,1,f32_rs,f32_rs+64,f32_rs+128};
    const shape4 f32_2 = {1048576,1,2,1,f32_rs,f32_rs+64,2*(f32_rs+64)+128};
    const shape4 f32_4 = {1048576,1,4,1,f32_rs,f32_rs+64,4*(f32_rs+64)+128};
    const size_t q8_rs = ggml_row_size(GGML_TYPE_Q8_0,512);
    const shape4 q8_4 = {512,1,4,1,q8_rs,q8_rs+64,4*(q8_rs+64)+128};
    const size_t small_rs = ggml_row_size(GGML_TYPE_F32,257);
    const shape4 perm_src = {257,2,3,2,3*(small_rs+16),small_rs+16,2*3*(small_rs+16)+32};
    const shape4 perm_dst = {257,2,3,2,small_rs+32,2*(small_rs+32)+32,3*2*(small_rs+32)+64};

    if (mode == "alias") {
        const shape4 source_rows = {257,1,4,1,small_rs,small_rs+32,4*(small_rs+32)+64};
        const shape4 overlap_rows = {257,1,4,1,small_rs,0,small_rs+64};
        bool ok = run_alias_case();
        ok &= run_separate_case("dst-self-overlap", GGML_TYPE_F32, source_rows, overlap_rows, 48);
        ok &= run_separate_case("src-self-overlap", GGML_TYPE_F32, overlap_rows, source_rows, 48);
        return ok ? 0 : 3;
    }
    if (mode != "separate") return 2;

    bool ok = true;
    ok &= run_separate_case("f32-plane1", GGML_TYPE_F32, f32_1, f32_1, 48);
    ok &= run_separate_case("f32-plane2", GGML_TYPE_F32, f32_2, f32_2, 48);
    ok &= run_separate_case("f32-plane4-real", GGML_TYPE_F32, f32_4, f32_4, 48);
    ok &= run_separate_case("f32-plane4-nth1", GGML_TYPE_F32, f32_4, f32_4, 1);
    ok &= run_separate_case("f32-permuted-odd", GGML_TYPE_F32, perm_src, perm_dst, 48);
    ok &= run_separate_case("q8-plane4", GGML_TYPE_Q8_0, q8_4, q8_4, 48);
    return ok ? 0 : 3;
}
