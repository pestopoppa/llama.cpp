// AKX-ALL-LB4.9 FLASH_ATTN_EXT per-head-sinks op cases vs the ggml reference (nth 1/16/32/64/96).
// Build ggml-cpu with -DGGML_IQK_FLASH_ATTENTION=ON and -DLB49_TEST_ROUTE_COUNTERS; run with GGML_IQK_FA_SINKS=1.
// Not registered in CMake: the route witness is a test-only compile definition.
// No result is evidence until built and executed against the reviewed candidate.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

static bool run(ggml_context * ctx, ggml_tensor * t, int nth, bool ref) {
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, t);
    ggml_cplan p = ggml_graph_plan(g, nth, nullptr);
    std::vector<uint8_t> buf(p.work_size + 65536, 0xa5);
    p.work_data = buf.data(); p.use_ref = ref;
    bool ok = ggml_graph_compute(g, &p) == GGML_STATUS_SUCCESS;
    return ok && std::all_of(buf.begin() + p.work_size, buf.end(), [](uint8_t x) { return x == 0xa5; });
}
extern "C" unsigned long long lb49_route_count(int);
static int gcd(int a,int b) { while (b) { int t=a%b; a=b; b=t; } return a; }
static bool smoke() { static const bool s = std::getenv("LB49_SMOKE") != nullptr; return s; }
static int test(const char * name, int nk, int mode, int kvh=1, bool zero_sinks=false) {
    const bool dead = mode == 1;
    constexpr int D=512, H=64;
    ggml_init_params ip={size_t(128)*1024*1024, nullptr, false};
    ggml_context * c=ggml_init(ip); if (!c) return 2;
    auto * q=ggml_new_tensor_4d(c,GGML_TYPE_F32,D,1,H,1);
    auto * k=ggml_new_tensor_4d(c,GGML_TYPE_F16,D,nk,kvh,1);
    auto * v=k; // DS41 K-only cache: K and V are the same tensor.
    auto * mask=ggml_new_tensor_4d(c,GGML_TYPE_F16,nk,1,1,1);
    auto * sinks=ggml_new_tensor_1d(c,GGML_TYPE_F32,H);
    for (int i=0;i<D*H;i++) ((float*)q->data)[i]=float((i*17)%193-96)/96;
    for (int i=0;i<D*nk*kvh;i++) {
        ((ggml_fp16_t*)k->data)[i]=ggml_fp32_to_fp16(float((i*11)%197-98)/98);
    }
    for (int j=0;j<nk;j++) ((ggml_fp16_t*)mask->data)[j]=ggml_fp32_to_fp16(dead || (mode == 2 && j>0 && j<nk-1 && j%7==0) ? -INFINITY : 0.0f);
    for (int h=0;h<H;h++) ((float*)sinks->data)[h]=zero_sinks ? 0.0f : float(h-31)/16;
    auto make=[&]() {
        auto * o=ggml_flash_attn_ext(c,q,k,v,mask,1.0f/std::sqrt(float(D)),0,0);
        ggml_flash_attn_ext_set_prec(o,GGML_PREC_F32); if (mode!=3) ggml_flash_attn_ext_add_sinks(o,sinks); return o;
    };
    auto reg=ggml_backend_cpu_reg();
    auto set=(ggml_backend_cpu_set_iqk_fa_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_cpu_set_iqk_fa");
    auto count=(ggml_backend_cpu_iqk_fa_n_calls_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_cpu_iqk_fa_n_calls");
    if (!set || !count) {ggml_free(c);return 2;}
    set(true);auto * reference=make();auto before=count();
    if (!run(c,reference,1,true) || count()!=before) {ggml_free(c);return 2;}
    std::vector<double> oracle(D*H, 0.0), score(nk);
    for (int h=0; h<H; ++h) {
        double maximum=-INFINITY;
        for (int j=0; j<nk; ++j) {
            double m=ggml_fp16_to_fp32(((ggml_fp16_t*)mask->data)[j]);
            if (std::isinf(m) && m<0) { score[j]=-INFINITY; continue; }
            double dot=0; for (int d=0; d<D; ++d)
                dot+=double(((float*)q->data)[h*D+d])*ggml_fp16_to_fp32(((ggml_fp16_t*)k->data)[((h/(H/kvh))*nk+j)*D+d]);
            score[j]=dot/std::sqrt(double(D))+m; maximum=std::max(maximum,score[j]);
        }
        if (maximum==-INFINITY) continue;
        double sink=((float*)sinks->data)[h]; if (mode!=3) maximum=std::max(maximum,sink);
        double denominator=mode==3 ? 0.0 : std::exp(sink-maximum);
        for (int j=0; j<nk; ++j) {
            if (score[j]==-INFINITY) continue;
            double w=std::exp(score[j]-maximum);denominator+=w;
            for (int d=0;d<D;++d) oracle[h*D+d]+=w*ggml_fp16_to_fp32(((ggml_fp16_t*)v->data)[((h/(H/kvh))*nk+j)*D+d]);
        }
        for (int d=0;d<D;++d) oracle[h*D+d]/=denominator;
    }
    // OFF control must route to ggml with no IQK invocation or test route count.
    set(false); auto * off=make(); auto off_before=count();
    unsigned long long off_routes[4]; for (int z=0;z<4;z++) off_routes[z]=lb49_route_count(z);
    bool off_ok=run(c,off,1,false) && count()==off_before;
    for (int z=0;z<4;z++) off_ok &= lb49_route_count(z)==off_routes[z];
    const float * rr=(const float*)reference->data; const float * oo=(const float*)off->data;
    double off_err=0,off_den=0;for(int i=0;i<D*H;i++) { if (!std::isfinite(oo[i])) off_ok=false; double e=oo[i]-rr[i];off_err+=e*e;off_den+=double(rr[i])*rr[i]; }
    off_ok &= (off_den ? off_err/off_den : (off_err ? INFINITY : 0)) <= 5e-4;
    printf("RESULT case=%s control=off verdict=%s\n",name,off_ok ? "PASS" : "FAIL");
    // Operator 2026-11-09: bitwise equality applies to repeats at one route and one nth. Different nth values are different
    // partitions and are judged by NMSE and the independent oracle. A thread-split-dependent bug below 5e-4 NMSE passes this gate.
    struct Prior { int nth; int route; std::vector<float> out; };
    constexpr int REPEATS=3;
    set(true);std::vector<Prior> prior; bool all=off_ok;
    std::vector<int> nths={1,16,32,64,96}; if (smoke()) nths={1,16};
    for (int nth : nths) {
        int blocks=nk/32, g=gcd(blocks,nth), nth_k=nth/g;
        int route = kvh==1 ? (blocks>=4*nth ? 0 : ((H+nth_k-1)/nth_k>1 ? 1 : (nk>=32*nth ? 2 : 3))) : (kvh*nk>=32*nth ? 2 : 3);
        std::vector<float> first; bool ok=true, repeat_bits=true; unsigned long long delta=~0ULL;
        double nmse=0,oracle_nmse=0,max_abs=0,mean_abs=0;const float * r=(const float*)reference->data;
        for (int rep=0; rep<REPEATS; ++rep) {
            auto * out=make();float * x=(float*)out->data;
            for (int i=0;i<D*H;i++) x[i]=std::numeric_limits<float>::quiet_NaN();
            unsigned long long routes[4];for(int z=0;z<4;z++) routes[z]=lb49_route_count(z);
            before=count();ok &= run(c,out,nth,false);delta=std::min<unsigned long long>(delta,count()-before);
            for(int z=0;z<4;z++) ok &= (lb49_route_count(z)-routes[z]) == (z==route ? 1u : 0u);
            double err=0,den=0,oracle_err=0,oracle_den=0,rep_max=0,rep_mean=0;
            for(int i=0;i<D*H;i++){if(!std::isfinite(x[i]) || !std::isfinite(r[i])) ok=false;double diff=x[i]-r[i];rep_max=std::max(rep_max,std::abs(diff));rep_mean+=std::abs(diff);err+=diff*diff;den+=double(r[i])*r[i];double od=x[i]-oracle[i];oracle_err+=od*od;oracle_den+=oracle[i]*oracle[i];}
            // Existing test-fa-rowexact/test-backend-ops FLASH_ATTN_EXT threshold is 5e-4 NMSE; the worst repeat is what counts.
            nmse=std::max(nmse,den ? err/den : (err ? INFINITY : 0));
            oracle_nmse=std::max(oracle_nmse,oracle_den ? oracle_err/oracle_den : (oracle_err ? INFINITY : 0));
            max_abs=std::max(max_abs,rep_max);mean_abs=std::max(mean_abs,rep_mean/(D*H));
            if (rep==0) first.assign(x,x+D*H); else repeat_bits &= std::memcmp(first.data(),x,D*H*sizeof(float))==0;
        }
        // Direct output-to-output NMSE against every earlier nth, including across routes; each pair is printed.
        double cross=0; int cross_ref=-1;
        for (const Prior & p : prior) {
            double err=0,den=0;
            for(int i=0;i<D*H;i++){double diff=first[i]-p.out[i];err+=diff*diff;den+=double(p.out[i])*p.out[i];}
            double n=den ? err/den : (err ? INFINITY : 0);
            printf("CROSS case=%s nth_a=%d route_a=%d nth_b=%d route_b=%d nmse=%.9g\n",name,p.nth,p.route,nth,route,n);
            if (cross_ref<0 || n>=cross) { cross=n; cross_ref=p.nth; }
        }
        prior.push_back({nth,route,first});
        bool pass=ok && delta>0 && repeat_bits && nmse<=5e-4 && oracle_nmse<=5e-4 && cross<=5e-4;
        printf("RESULT case=%s nk=%d mode=%d nth=%d route=%d iqk_delta=%llu repeats=%d bitwise_repeats=%d cross_nth_nmse=%.9g cross_ref_nth=%d nmse=%.9g max_abs=%.9g mean_abs=%.9g oracle_nmse=%.9g ran=%d verdict=%s\n",name,nk,mode,nth,route,delta,REPEATS,repeat_bits,cross,cross_ref,nmse,max_abs,mean_abs,oracle_nmse,ok,pass ? "PASS" : "FAIL");
        all &= pass;
    }
    ggml_free(c); return all ? 0 : 1;
}
int main() {
    setenv("GGML_FA_VKQ_F32", "1", 1);
    int rc=0;
    if (smoke()) { rc |= test("gathered-640",640,2); return rc; }
    rc |= test("ds41-swa-layer0",128,0); rc |= test("ds41-swa-layer1",128,0);
    rc |= test("ds41-stock-mask",8192,0); rc |= test("gathered-640",640,2);
    rc |= test("gathered-768",768,2); rc |= test("fully-masked",128,1);
    rc |= test("generic-fallback",608,0); rc |= test("null-sinks",640,3);

    rc |= test("ds41-sinks-zero-swa0",128,0,1,true);
    rc |= test("ds41-sinks-zero-stock",8192,0,1,true);
    rc |= test("gathered-640-sinks-zero",640,2,1,true);
    rc |= test("gathered-768-sinks-zero",768,2,1,true);
    return rc;
}
