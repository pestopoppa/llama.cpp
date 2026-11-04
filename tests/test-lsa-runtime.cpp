#include "llama-lsa-runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
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
    check(result.key_append_host_us==0 && result.query_select_host_us==0 && result.attention_host_us==0,
          "host phase timers are disabled by default");
    auto profiled_options=o;profiled_options.collect_phase_times=true;
    runtime profiled(maps(),profiled_options);profiled.begin_request(1,1,runtime_backend::cpu);
    auto profiled_result=profiled.execute(x.frame(0,65));
    check(profiled_result.sparse && profiled_result.key_append_host_us>0 &&
          profiled_result.query_select_host_us>0 && profiled_result.attention_host_us>0,
          "opt-in host intervals cover key append, query selection and attention");
    runtime profiled_failure(maps(),profiled_options);profiled_failure.begin_request(1,1,runtime_backend::cpu);
    hook_state phase_fail;phase_fail.fail_call=2;profiled_failure.set_cpu_attention_for_test(hook,&phase_fail);
    auto failed_phase=profiled_failure.execute(x.frame(0,65));
    check(!failed_phase.sparse && !failed_phase.staged_output &&
          failed_phase.key_append_host_us==0 && failed_phase.query_select_host_us==0 &&
          failed_phase.attention_host_us==0,
          "failed opt-in request exposes no partial phase or output lease");
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

    // Keep two requests live at the same positions and alternate their work.
    // Each result must match an isolated runtime using that request's own KV.
    fixture request_a(700), request_b(700);
    for (int row = 0; row < 8; ++row) {
        std::fill(request_b.k.begin()+row*4*256, request_b.k.begin()+(row+1)*4*256, float(row+3));
        std::fill(request_b.v.begin()+row*4*256, request_b.v.begin()+(row+1)*4*256, float(100+row));
    }
    runtime interleaved(maps(),o);interleaved.begin_request(1,1,runtime_backend::cpu);
    interleaved.begin_request(2,1,runtime_backend::cpu);
    auto publish_rows = [&](runtime & owner, const runtime_result & lease, int rows) {
        std::vector<float> output(size_t(rows)*24*256,-777);
        const bool published=owner.publish_cpu(lease,output.data(),output.size());
        return std::pair<bool,std::vector<float>>(published,std::move(output));
    };
    auto a_prefix=interleaved.execute(request_a.frame(0,2,1));
    auto a_prefix_out=publish_rows(interleaved,a_prefix,2);
    auto b_prefix=interleaved.execute(request_b.frame(0,2,2));
    auto b_prefix_out=publish_rows(interleaved,b_prefix,2);
    auto a_decode=interleaved.execute(request_a.frame(2,1,1));
    auto a_decode_out=publish_rows(interleaved,a_decode,1);
    runtime isolated_a(maps(),o);isolated_a.begin_request(1,1,runtime_backend::cpu);
    auto isolated_a_prefix=isolated_a.execute(request_a.frame(0,2,1));
    auto isolated_a_prefix_out=publish_rows(isolated_a,isolated_a_prefix,2);
    auto isolated_a_decode=isolated_a.execute(request_a.frame(2,1,1));
    auto isolated_a_decode_out=publish_rows(isolated_a,isolated_a_decode,1);
    runtime isolated_b(maps(),o);isolated_b.begin_request(2,1,runtime_backend::cpu);
    auto isolated_b_prefix=isolated_b.execute(request_b.frame(0,2,2));
    auto isolated_b_prefix_out=publish_rows(isolated_b,isolated_b_prefix,2);
    check(a_prefix.sparse && b_prefix.sparse && a_decode.sparse &&
          isolated_a_prefix.sparse && isolated_a_decode.sparse && isolated_b_prefix.sparse &&
          a_prefix_out.first && b_prefix_out.first && a_decode_out.first &&
          isolated_a_prefix_out.first && isolated_a_decode_out.first && isolated_b_prefix_out.first &&
          a_prefix_out.second==isolated_a_prefix_out.second &&
          a_decode_out.second==isolated_a_decode_out.second &&
          b_prefix_out.second==isolated_b_prefix_out.second,
          "interleaved live requests match isolated A/B/A KV and output oracles");

    // A rejected speculative suffix is removed from the selector history before
    // different tokens are appended at the same positions. Compare replay with
    // a fresh runtime that has seen only the retained prefix and replacement.
    fixture replay_candidate(700), replay_oracle(700);
    runtime replay(maps(),o);replay.begin_request(1,1,runtime_backend::cpu);
    runtime fresh_replay(maps(),o);fresh_replay.begin_request(1,1,runtime_backend::cpu);
    auto candidate_prefix=replay.execute(replay_candidate.frame(0,4));
    auto oracle_prefix=fresh_replay.execute(replay_oracle.frame(0,4));
    auto discarded=replay.execute(replay_candidate.frame(4,3));
    const bool prefix_ok=candidate_prefix.sparse && oracle_prefix.sparse && discarded.sparse &&
        replay.indexed_rows(1,3)==7 && replay.truncate(1,1,4) && replay.indexed_rows(1,3)==4;
    auto replace_suffix=[](fixture & f) {
        for(int row=4;row<7;++row) {
            std::fill(f.q.begin()+row*24*256,f.q.begin()+(row+1)*24*256,float(row+9)/13.0f);
            std::fill(f.k.begin()+row*4*256,f.k.begin()+(row+1)*4*256,float(row+11)/17.0f);
            std::fill(f.v.begin()+row*4*256,f.v.begin()+(row+1)*4*256,float(200+row));
        }
    };
    replace_suffix(replay_candidate);replace_suffix(replay_oracle);
    auto replayed=replay.execute(replay_candidate.frame(4,3));
    auto fresh_suffix=fresh_replay.execute(replay_oracle.frame(4,3));
    auto replayed_out=publish_rows(replay,replayed,3);
    auto fresh_out=publish_rows(fresh_replay,fresh_suffix,3);
    check(prefix_ok && replayed.sparse && fresh_suffix.sparse &&
          replay.indexed_rows(1,3)==7 && fresh_replay.indexed_rows(1,3)==7 &&
          replayed_out.first && fresh_out.first && replayed_out.second==fresh_out.second,
          "truncate then replayed speculative suffix matches fresh-runtime output oracle");

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
    // A serving graph can evaluate one native layer before a later layer fails.
    // Exercise the shared request and both layer indexes, then mirror the graph
    // boundary's invalidate-on-any-native-failure rule.
    runtime composed(maps(),o);check(composed.begin_request(1,1,runtime_backend::cpu),"composed request starts at epoch one");
    hook_state later;later.fail_call=3;composed.set_cpu_attention_for_test(hook,&later);
    auto first_layer=composed.execute(x.frame(0,2,1,1,3));
    std::vector<float> first_output(2*24*256,-41);
    check(first_layer.sparse && composed.publish_cpu(first_layer,first_output.data(),first_output.size()) &&
          composed.indexed_rows(1,3)==2,"earlier native layer publishes complete output and owns its index");
    auto later_layer=composed.execute(x.frame(0,65,1,1,7));
    std::vector<float> later_output(65*24*256,-43);
    check(later.chunks==std::vector<int64_t>({2,32,32}) && !later_layer.sparse &&
          later_layer.reason==dense_reason::attention && !later_layer.staged_output,
          "later native layer fails after one successful internal chunk without a staged lease");
    check(!composed.publish_cpu(later_layer,later_output.data(),later_output.size()) &&
          std::all_of(later_output.begin(),later_output.end(),[](float value){return value==-43;}) &&
          !composed.publish_cpu(first_layer,first_output.data(),first_output.size()),
          "failed layer publishes nothing; subsequent execute invalidates the earlier lease");
    composed.invalidate(1,invalidation::shift);
    check(composed.indexed_rows(1,3)==-1 && composed.indexed_rows(1,7)==-1 &&
          !composed.execute(x.frame(2,1,1,1,3)).sparse,
          "graph failure invalidation clears every layer and bars continuation in the old epoch");
    check(composed.begin_request(1,2,runtime_backend::cpu),"fresh epoch reopens composed request");
    composed.set_cpu_attention_for_test(nullptr,nullptr);
    auto fresh3=composed.execute(x.frame(0,2,1,2,3));
    auto fresh7=composed.execute(x.frame(0,2,1,2,7));
    check(fresh3.sparse && fresh7.sparse && composed.indexed_rows(1,3)==2 &&
          composed.indexed_rows(1,7)==2,"fresh epoch rebuilds both native layer histories from zero");
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
    check(tie.sparse && tie.reason==dense_reason::none && ambiguous.indexed_rows(1,3)==513,
          "author-rank tie resolution retains the visible 512-key cluster");
    std::vector<float> tied_output(513*24*256,-44);
    check(ambiguous.publish_cpu(tie,tied_output.data(),tied_output.size()),
          "resolved tie publishes a complete CPU attention lease");
    bool tied_oracle=true;
    for(int i=0;i<513;++i) tied_oracle &= std::fabs(tied_output[i*24*256]-float(i)/2)<1e-6f;
    check(tied_oracle,"all visible keys with zero physical QK return causal uniform-V mean through tied boundary");
    std::printf("RESULT runtime failures=%d\n",failures);return failures?1:0;
}
