#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int64_t DK = 4096, DM = 288, K = 256, M = 64, NT = 4, NE = 32, NU = 8;
std::vector<uint8_t> weights(ggml_type type, int64_t k, int64_t rows) {
    std::vector<float> f((size_t) k*rows);
    for (size_t i=0;i<f.size();++i) f[i]=0.35f*std::sin((float)i*.013f)+.09f*std::cos((float)i*.007f);
    std::vector<uint8_t> q(ggml_row_size(type,k)*(size_t)rows);
    if (ggml_quantize_chunk(type,f.data(),q.data(),0,rows,k,nullptr)!=q.size()) std::abort();
    return q;
}
float act(size_t i) { return .27f*std::sin((float)i*.017f)-.19f*std::cos((float)i*.011f); }
std::vector<float> compute(ggml_backend_t be, ggml_backend_graph_plan_t plan, ggml_tensor * y) {
    if (ggml_backend_graph_plan_compute(be,plan)!=GGML_STATUS_SUCCESS) std::abort();
    std::vector<float> out((size_t)ggml_nelements(y));
    ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float)); return out;
}
bool equal(const std::vector<float>&a,const std::vector<float>&b) { return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0; }
int run(bool mmid, ggml_type dense_type) {
    ggml_init_params ip={ggml_tensor_overhead()*12+ggml_graph_overhead(),nullptr,true};
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * w=mmid?ggml_new_tensor_3d(ctx.get(),GGML_TYPE_Q4_K,K,M,NE):ggml_new_tensor_2d(ctx.get(),dense_type,DK,DM);
    ggml_tensor * x=nullptr,*ids=nullptr,*y=nullptr;
    if (mmid) { ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,NU,NT); x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,K,NU,NT); y=ggml_mul_mat_id(ctx.get(),w,x,ids); }
    else { x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,DK,NT); y=ggml_mul_mat(ctx.get(),w,x); }
    ggml_cgraph * gf=ggml_new_graph(ctx.get()); ggml_build_forward_expand(gf,y);
    ggml_backend_ptr be(ggml_backend_cpu_init()); ggml_backend_cpu_set_n_threads(be.get(),48);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(),be.get()));
    if (w->type == GGML_TYPE_F32) {
        std::vector<float> fw((size_t)DK*DM); for(size_t i=0;i<fw.size();++i)fw[i]=.35f*std::sin((float)i*.013f)+.09f*std::cos((float)i*.007f);
        ggml_backend_tensor_set(w,fw.data(),0,fw.size()*sizeof(float));
    } else {
        auto qw=weights(w->type,mmid?K:DK,mmid?M*NE:DM); ggml_backend_tensor_set(w,qw.data(),0,qw.size());
    }
    std::vector<float> xv((size_t)ggml_nelements(x)); for(size_t i=0;i<xv.size();++i)xv[i]=act(i); ggml_backend_tensor_set(x,xv.data(),0,xv.size()*sizeof(float));
    if (mmid) { std::vector<int32_t> iv((size_t)NU*NT); for(int t=0;t<NT;++t)for(int i=0;i<NU;++i)iv[(size_t)t*NU+i]=i; ggml_backend_tensor_set(ids,iv.data(),0,iv.size()*sizeof(int32_t)); }
    ggml_backend_graph_plan_t plan=ggml_backend_graph_plan_create(be.get(),gf); if(!plan)std::abort();
    ggml_backend_cpu_set_rowexact(be.get(),false); auto off1=compute(be.get(),plan,y);
    ggml_backend_cpu_set_rowexact(be.get(),true);  auto on  =compute(be.get(),plan,y);
    ggml_backend_cpu_set_rowexact(be.get(),false); auto off2=compute(be.get(),plan,y);
    ggml_backend_graph_plan_free(be.get(),plan);
    const bool restored=equal(off1,off2), witness=!equal(off1,on);
    std::printf("mode=%s restored=%d difference_witness=%d elements=%zu\n",mmid?"mmid-q4":dense_type==GGML_TYPE_F32?"dense-f32":"dense-q8",restored,witness,off1.size());
    return restored&&witness?0:3;
}
}
int main(int argc,char**argv){if(argc!=2)return 2;std::string m=argv[1];if(m=="dense")return run(false,GGML_TYPE_Q8_0);if(m=="densef32")return run(false,GGML_TYPE_F32);if(m=="mmid")return run(true,GGML_TYPE_Q4_K);return 2;}
