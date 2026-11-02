#include "../src/llama-kv-rows-reference.h"
#include "../src/llama-kv-rows-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <unistd.h>

int main() {
    constexpr int D = 64, NK = 256, NT = 4;
    auto * ctx = ggml_init({64u * 1024u * 1024u, nullptr, false});
    auto * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NT);
    auto * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NK);
    auto * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, NK);
    auto * qd = static_cast<float *>(q->data);
    auto * kd = static_cast<float *>(k->data);
    auto * vd = static_cast<float *>(v->data);
    for (int i=0;i<D*NT;++i) qd[i] = float((i%13)-6)/17;
    for (int i=0;i<D*NK;++i) { kd[i] = float((i%11)-5)/13; vd[i] = float((i%17)-8)/9; }
    llama_kv_rows_plan plan{NK, 2, NK, std::vector<int32_t>(2*NK, -1)};
    plan.rows[0]=0; plan.rows[1]=2; plan.rows[NK]=1; plan.rows[NK+1]=3;
    const auto ni = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    const auto z = ggml_fp32_to_fp16(0);
    std::vector<ggml_fp16_t> mask(NT*NK, ni);
    for (int t=0;t<NT;++t) { mask[t*NK+(t<2?0:1)]=z; mask[t*NK+(t<2?2:3)]=z; }
    std::vector<ggml_tensor *> outputs;
    if (!llama_kv_rows_reference_graph(ctx,plan,q,k,v,mask,0.125f,outputs)) return 1;
    auto * graph=ggml_new_graph(ctx);
    for (auto * out : outputs) ggml_build_forward_expand(graph,out);
    auto * backend=ggml_backend_cuda_init(0);
    if (!backend || !ggml_backend_is_cuda(backend)) return 2;
    std::fprintf(stderr,"KPF41 GPU PID %d backend %s\n", int(getpid()), ggml_backend_name(backend));
    std::vector<std::vector<float>> actual;
    if (!llama_kv_rows_backend_compute(backend,ctx,graph,outputs,actual) || actual.size()!=2) return 3;
    for (int t=0;t<NT;++t) {
        const int a=t<2?0:1, b=t<2?2:3;
        float sa=0,sb=0;
        for(int d=0;d<D;++d) { sa+=qd[t*D+d]*kd[a*D+d]*0.125f; sb+=qd[t*D+d]*kd[b*D+d]*0.125f; }
        const float wa=std::exp(sa), wb=std::exp(sb);
        for(int d=0;d<D;++d) {
            const float expected=(wa*vd[a*D+d]+wb*vd[b*D+d])/(wa+wb);
            const float got=actual[t/2][(t%2)*D+d];
            if(!std::isfinite(got)||std::fabs(got-expected)>0.003f) {
                std::fprintf(stderr,"mismatch t=%d d=%d got=%g expected=%g\n",t,d,got,expected); return 4;
            }
        }
    }
    std::vector<std::vector<float>> bad{{1}};
    if (llama_kv_rows_backend_compute(backend,ctx,graph,{nullptr},bad) || !bad.empty()) return 5;
    // A graph borrowing unbound data from a second context must refuse rather
    // than entering ggml_backend_graph_copy with an absent source buffer.
    auto * other = ggml_init({1024u*1024u, nullptr, false});
    auto * foreign = ggml_new_tensor_2d(other, GGML_TYPE_F32, D, NT);
    auto * mixed = ggml_add(ctx, q, foreign);
    auto * foreign_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(foreign_graph, mixed);
    if (llama_kv_rows_backend_compute(backend,ctx,foreign_graph,{mixed},bad) || !bad.empty()) return 6;
    ggml_free(other);
    // Give the caller time to sample this exact PID while the HIP backend remains resident.
    sleep(2);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::puts("KPF-41 GPU gather + attention graph parity passed");
    return 0;
}
