// [GDN-COLSPLIT] bit-exactness of GGML_GDN_COLSPLIT=1 vs unset for the CPU GATED_DELTA_NET kernel:
// every output float (attention scores + all K snapshot slots) must be identical (memcmp), across
// threads, K, KDA, n_tokens, n_seqs, heads and head sizes. Exit 0 iff all cases are bit-identical.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct cfg { int64_t H, S, nt, ns; int vr; bool kda; int64_t K; };

static std::vector<float> run(const cfg & c, int nth, bool knob, uint32_t seed) {
    if (knob) { setenv("GGML_GDN_COLSPLIT", "1", 1); } else { unsetenv("GGML_GDN_COLSPLIT"); }
    ggml_init_params ip = { 256u*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const int64_t HV = c.H * c.vr;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, c.H,  c.nt, c.ns);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, c.H,  c.nt, c.ns);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, HV,   c.nt, c.ns);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.kda ? c.S : 1, HV, c.nt, c.ns);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, c.nt, c.ns);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S, c.S, HV, c.ns);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (ggml_tensor * t : { q, k, v, g, b, s }) {
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++) { d[i] = u(rng); }
    }
    for (int64_t i = 0; i < ggml_nelements(g); i++) { ((float *) g->data)[i] = -0.5f * (u(rng) + 1.f); } // decay <= 0
    ggml_tensor * o = ggml_gated_delta_net(ctx, q, k, v, g, b, s, c.K);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_graph_compute_with_ctx(ctx, gf, nth);
    std::vector<float> out((float *) o->data, (float *) o->data + ggml_nelements(o));
    ggml_free(ctx);
    return out;
}

int main() {
    const std::vector<cfg> cases = {
        {16, 128, 1, 1, 3, false, 1}, {16, 128, 1, 1, 2, false, 1}, {16, 128, 2, 1, 3, false, 1},
        {16, 128, 5, 1, 3, false, 1}, {16, 128, 5, 1, 3, false, 5}, {16, 128, 9, 1, 2, false, 4},
        {16, 128, 1, 1, 3, true,  1}, {16, 128, 5, 2, 3, true,  5}, {4, 192, 3, 2, 2, false, 3},
        {16, 64, 5, 1, 3, false, 5},  {4, 128, 64, 1, 1, false, 4}, {2, 130, 3, 1, 1, false, 2}, // S=130: split must stay off
    };
    int bad = 0, n = 0;
    for (const cfg & c : cases) {
        for (int nth : { 1, 3, 8, 24 }) {
            auto a = run(c, nth, false, 1234 + (uint32_t) n);
            auto b = run(c, nth, true,  1234 + (uint32_t) n);
            const bool same = a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
            printf("H=%ld S=%ld nt=%ld ns=%ld vr=%d kda=%d K=%ld nth=%d : %s\n", (long) c.H, (long) c.S, (long) c.nt, (long) c.ns,
                   c.vr, (int) c.kda, (long) c.K, nth, same ? "BIT-EXACT" : "MISMATCH");
            bad += !same; n++;
        }
    }
    printf("%d/%d bit-exact\n", n - bad, n);
    return bad ? 1 : 0;
}
