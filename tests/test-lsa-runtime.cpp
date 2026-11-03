#include "llama-lsa-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
using namespace llama_lsa;
static int failures=0;
static void check(bool ok,const char*name){std::printf("%s %s\n",ok?"PASS":"FAIL",name);failures+=!ok;}
static map_bundle maps(const char*id="maps-a") {
    map_bundle b;b.shape={24,4,256,64};b.model_identity="runtime-analytic-fixture";
    projection_maps m;m.shape=b.shape;m.identity=id;m.q.resize(768*64);m.k.resize(768*64);
    for(int d=0;d<64;++d)m.q[d*64+d]=m.k[d*64+d]=ggml_fp32_to_bf16(1);
    b.layers.emplace(3,m);b.layers.emplace(7,m);return b;
}
static runtime_options options() {
    runtime_options o;o.enabled=true;o.max_keys=2048;o.max_new_tokens=600;
    o.max_host_bytes=256*1024*1024;o.max_device_bytes=256*1024*1024;return o;
}
static tensor_view view(const void*p,int64_t rows,int heads) {
    return {p,GGML_TYPE_F32,rows,heads,256,size_t(heads)*256*4,256*4};
}
struct fixture {
    int64_t total;
    std::vector<float> q,k,pq,pk,v;
    std::vector<int32_t> positions,slots;
    fixture(int64_t n):total(n),q(n*24*256),k(n*4*256),pq(q.size()),pk(k.size()),v(k.size()),positions(n),slots(n) {
        for(int64_t i=0;i<n;++i){positions[i]=slots[i]=int32_t(i);q[i*24*256+64]=1;k[i*4*256+64]=1;
            std::fill(v.begin()+i*4*256,v.begin()+(i+1)*4*256,float(i));}
    }
    runtime_frame frame(int64_t first,int64_t count,uint64_t slot=1,uint64_t epoch=1,int layer=3) {
        runtime_frame f;f.slot=slot;f.epoch=epoch;f.layer=layer;f.first=first;f.map_identity="maps-a";
        f.positions=positions.data()+first;f.slots=slots.data()+first;
        f.logical_q=view(q.data()+first*24*256,count,24);f.logical_k=view(k.data()+first*4*256,count,4);
        f.physical_q=view(pq.data()+first*24*256,count,24);f.full_k=view(pk.data(),total,4);f.full_v=view(v.data(),total,4);f.scale=1/std::sqrt(256.f);return f;
    }
};
struct hook_state{int calls=0,fail_call=0;std::vector<int64_t> chunks;};
static bool hook(const attention_args&a,attention_stats*s,std::string*e,void*user) {
    auto &h=*static_cast<hook_state*>(user);++h.calls;h.chunks.push_back(a.q.rows);
    if(h.fail_call==h.calls){std::fill(a.output,a.output+a.q.rows*a.q.heads*a.v.dim,999.f);if(e)*e="injected late attention failure";return false;}
    return selected_attention_cpu(a,s,e);
}
int main() {
    auto o=options();fixture x(700);runtime r(maps(),o);
    check(r.valid() && r.begin_request(1,1,runtime_backend::cpu),"validated maps/options create isolated CPU request");
    hook_state h;r.set_cpu_attention_for_test(hook,&h);
    auto result=r.execute(x.frame(0,65));
    check(result.sparse && h.chunks==std::vector<int64_t>({32,32,1}),"real native attention executes bounded32-query chunks");
    check(r.indexed_rows(1,3)==65,"all new logical keys append once across query chunks");
    std::vector<float> destination(65*24*256,-999);
    check(r.publish_cpu(result,destination.data(),destination.size()),"successful host lease publishes after complete execution");
    bool oracle=true;for(int i=0;i<65;++i)oracle &= std::fabs(destination[i*24*256]-float(i)/2)<1e-5f;
    check(oracle,"physical zero QK produces independent causal uniform-V mean");
    auto forged=result;forged.rows=66;
    check(!r.publish_cpu(forged,destination.data(),destination.size()),"forged lease shape rejected before publication");
    check(!r.publish_cpu(result,destination.data(),destination.size()-1),"publication element count checked");
    auto next=r.execute(x.frame(65,1));check(next.sparse && r.indexed_rows(1,3)==66,"decode appends one new key without old logical K input");
    check(!r.publish_cpu(result,destination.data(),destination.size()),"next execute invalidates prior output lease");
    runtime other(maps(),o);other.begin_request(1,1,runtime_backend::cpu);
    check(!other.publish_cpu(next,destination.data(),destination.size()),"different runtime cannot publish another instance's lease");
    check(r.begin_request(2,1,runtime_backend::cpu),"different slot has independent request ownership");
    std::vector<float> one(24*256,-42);
    check(!r.publish_cpu(next,one.data(),one.size()) && one[0]==-42,"cross-slot mutation invalidates old lease without destination writes");
    check(r.execute(x.frame(0,2,2)).sparse && r.indexed_rows(2,3)==2 && r.indexed_rows(1,3)==66,"slot indexes remain isolated");
    check(r.execute(x.frame(0,3,1,1,7)).sparse && r.indexed_rows(1,7)==3,"layers keep separate index histories");
    check(r.truncate(1,1,2) && r.indexed_rows(1,3)==2 && r.indexed_rows(1,7)==2,"suffix rollback truncates every existing layer");
    check(r.execute(x.frame(2,1)).sparse,"replacement suffix appends after explicit truncation");
    check(!r.begin_request(1,1,runtime_backend::cpu),"request epoch must increase on slot reuse");
    auto stale=r.execute(x.frame(3,1,1,0));check(!stale.sparse && stale.reason==dense_reason::stale_epoch,"stale request epoch returns original-dense reason");
    for(auto cause:{invalidation::cache_clear,invalidation::sequence_copy,invalidation::state_restore,invalidation::shift,invalidation::slot_reuse}) {
        static uint64_t epoch=2;r.begin_request(1,epoch,runtime_backend::cpu);r.invalidate(1,cause);
        check(!r.execute(x.frame(0,1,1,epoch)).sparse,"clear/copy/restore/shift/reuse requires explicit fresh epoch");
        ++epoch;r.begin_request(1,epoch,runtime_backend::cpu);
        auto lost=r.execute(x.frame(5,1,1,epoch));check(!lost.sparse && lost.reason==dense_reason::missing_history,"restored nonzero history never reconstructed from physical quant KV");
        check(!r.execute(x.frame(0,1,1,epoch)).sparse,"failed nonzero restart remains dense even if same epoch later sends zero");++epoch;
    }
    runtime late(maps(),o);late.begin_request(1,1,runtime_backend::cpu);hook_state fail;fail.fail_call=2;late.set_cpu_attention_for_test(hook,&fail);
    std::vector<float> sentinel(65*24*256,-123);
    auto failed=late.execute(x.frame(0,65));
    check(!failed.sparse && failed.reason==dense_reason::attention && !failed.staged_output,"late chunk failure exposes no partial output");
    check(!late.publish_cpu(failed,sentinel.data(),sentinel.size()) && sentinel[0]==-123 && sentinel.back()==-123,"failed publication leaves graph destination untouched");
    int original_dense_calls=0;if(!failed.sparse){++original_dense_calls;std::fill(sentinel.begin(),sentinel.end(),7);}
    check(original_dense_calls==1 && sentinel[0]==7,"caller chooses original dense callback explicitly on fallback");
    check(late.indexed_rows(1,3)==-1 && !late.execute(x.frame(65,1)).sparse,"failure poisons appended layer and prevents double append/retry");
    check(late.begin_request(1,2,runtime_backend::cpu),"explicit newer epoch permits fresh retry");
    late.set_cpu_attention_for_test(nullptr,nullptr);check(late.execute(x.frame(0,2,1,2)).sparse,"fresh zero epoch recovers after failed transaction");
    auto changed=maps("maps-b");check(late.replace_maps(changed),"validated map replacement accepted");
    check(!late.execute(x.frame(2,1,1,2)).sparse,"map replacement invalidates existing request history");
    late.begin_request(1,3,runtime_backend::cpu);auto rebound=x.frame(0,1,1,3);
    check(!late.execute(rebound).sparse,"stale graph map identity rejected");rebound.map_identity="maps-b";
    check(late.execute(rebound).sparse,"fresh zero epoch can bind replacement map identity");
    runtime guard(maps(),o);guard.begin_request(1,1,runtime_backend::cpu);
    auto f=x.frame(0,1);
    for(int which=0;which<6;++which){auto bad=f;if(which==0)bad.single_sequence=false;if(which==1)bad.causal_text=false;if(which==2)bad.dca=true;if(which==3)bad.speculative=true;if(which==4)bad.mtp=true;if(which==5)bad.contiguous_prefix=false;
        check(!guard.execute(bad).sparse,"unsupported request/mask/physical-prefix guard returns dense");}
    f=x.frame(0,2);auto prior=x.slots[1];x.slots[1]=3;check(!guard.execute(f).sparse,"noncontiguous physical new slot rejected");x.slots[1]=prior;
    x.positions[1]=0;check(!guard.execute(f).sparse,"noncontiguous causal position rejected");x.positions[1]=1;
    auto bad=f;bad.logical_k.rows=1;check(!guard.execute(bad).sparse,"logical new K count differs from Q rejected");
    bad=f;bad.full_k.row_stride=1;check(!guard.execute(bad).sparse,"overlapping physical KV stride rejected");
    bad=f;bad.first=std::numeric_limits<int64_t>::max();check(!guard.execute(bad).sparse,"overflowing append position refused before allocation");
    bad=f;bad.layer=99;check(!guard.execute(bad).sparse,"uncovered global layer cannot select sparse attention");
    auto small=o;small.max_host_bytes=1;runtime invalid(maps(),small);check(!invalid.valid(),"undersized host admission rejected before execution");
    small=o;small.query_chunk=33;runtime wide(maps(),small);check(!wide.valid(),"query chunk above32 rejected");
    small=o;small.enabled=false;runtime disabled(maps(),small);disabled.begin_request(1,1,runtime_backend::cpu);
    check(disabled.execute(x.frame(0,1)).reason==dense_reason::feature_disabled,"feature is explicitly disabled by default");
    auto broken=maps();broken.layers[3].shape.kv_heads=2;runtime wrong(std::move(broken),o);check(!wrong.valid(),"bundle per-layer geometry mismatch rejected");
    auto cap=o;cap.max_new_tokens=1;cap.max_host_bytes=4*1024*1024;cap.max_requests=2;
    runtime aggregate(maps(),cap);aggregate.begin_request(1,1,runtime_backend::cpu);aggregate.begin_request(2,1,runtime_backend::cpu);
    check(aggregate.execute(x.frame(0,1)).sparse,"first CPU layer fits aggregate admission");
    auto denied=aggregate.execute(x.frame(0,1,2));check(!denied.sparse && denied.reason==dense_reason::resource,"second live CPU cache denied by aggregate cap before allocation");
    check(!aggregate.begin_request(3,1,runtime_backend::cpu),"configured request limit bounds metadata ownership");
    auto different=maps();different.model_identity="wrong-model";
    check(!aggregate.replace_maps(different),"map replacement cannot change bound model identity");
    fixture ties(513);for(int i=0;i<513;++i){std::fill(ties.k.begin()+i*4*256,ties.k.begin()+(i+1)*4*256,0);const int pool=(i%512)/4;const int axis=pool>=124?pool-124:pool/2;ties.k[i*4*256+64+axis]=1;}
    runtime ambiguous(maps(),o);ambiguous.begin_request(1,1,runtime_backend::cpu);
    auto tie=ambiguous.execute(ties.frame(0,513));
    check(!tie.sparse && tie.reason==dense_reason::ambiguous_selection && !tie.staged_output,"visible ambiguous Ward cluster returns original dense without publishing earlier chunks");
    std::printf("RESULT runtime failures=%d\n",failures);return failures?1:0;
}
