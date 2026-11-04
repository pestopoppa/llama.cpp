#include "llama-lsa-runtime-impl.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

namespace llama_lsa {
using phase_clock = std::chrono::steady_clock;
static uint64_t phase_us(phase_clock::time_point begin) {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(phase_clock::now() - begin).count());
}
bool runtime_add(size_t a,size_t b,size_t &o) { if(b>SIZE_MAX-a)return false;o=a+b;return true; }
bool runtime_mul(size_t a,size_t b,size_t &o) { if(b && a>SIZE_MAX/b)return false;o=a*b;return true; }
tensor_view runtime_slice(tensor_view v,int64_t first,int64_t count) {
    v.data=static_cast<const char*>(v.data)+size_t(first)*v.row_stride;v.rows=count;return v;
}
bool runtime_view(const tensor_view &v,int64_t rows,int heads,int dim,bool floating) {
    if(v.rows!=rows || rows<=0 || !v.data || v.heads!=heads || v.dim!=dim ||
       v.type<0 || v.type>=GGML_TYPE_COUNT)return false;
    if(floating && v.type!=GGML_TYPE_F32 && v.type!=GGML_TYPE_F16 && v.type!=GGML_TYPE_BF16)return false;
    const auto block=ggml_blck_size(v.type);if(block<=0 || dim%block)return false;
    const size_t bytes=ggml_row_size(v.type,dim);size_t last,end;
    if(v.row_stride<bytes || v.head_stride<bytes ||
       !runtime_mul(size_t(heads-1),v.head_stride,last) || !runtime_add(last,bytes,end))return false;
    if(heads>1 && v.row_stride<end && uint64_t(rows)>v.head_stride/v.row_stride)return false;
    return runtime_mul(size_t(rows-1),v.row_stride,last) && runtime_add(last,end,last);
}
const char * dense_reason_name(dense_reason r) {
    switch(r) {
#define R(x) case dense_reason::x:return #x
    R(none);R(feature_disabled);R(invalid_configuration);R(unknown_request);R(stale_epoch);
    R(unsupported_request);R(unsupported_layout);R(missing_layer);R(missing_history);
    R(noncontiguous_slots);R(capacity);R(resource);R(projection);R(ambiguous_selection);
    R(attention);R(device);R(invalidated);
#undef R
    }return "invalid";
}
static bool same_shape(const geometry&a,const geometry&b) {
    return a.q_heads==b.q_heads && a.kv_heads==b.kv_heads && a.head_dim==b.head_dim && a.rope_dim==b.rope_dim;
}
bool runtime::impl::configure() {
    const auto &s=maps.shape;const auto &o=options;
    if(!s.valid() || !((s.q_heads==24 && s.kv_heads==4)||(s.q_heads==16 && s.kv_heads==2)) ||
       s.head_dim!=256 || s.rope_dim!=64 || maps.model_identity.empty() || maps.layers.empty() || maps.layers.size()>1024 ||
       o.max_keys<=0 || o.max_keys>=INT32_MAX || o.max_new_tokens<=0 || o.max_new_tokens>o.max_keys ||
       !o.max_requests || o.max_requests>1024 || o.query_chunk<=0 || o.query_chunk>32 || !o.selection.valid() || !o.max_host_bytes)return false;
    const int64_t capacity=int64_t(o.selection.top_k)+o.selection.window+o.selection.sink;
    if(capacity<=0 || capacity>INT32_MAX)return false;
    map_bytes=0;
    for(const auto &entry:maps.layers) {
        const auto &m=entry.second;size_t bytes;
        if(entry.first<0 || !same_shape(m.shape,s) || !m.valid() ||
           !runtime_mul(m.q.size(),2*sizeof(ggml_bf16_t),bytes) || !runtime_add(map_bytes,bytes,map_bytes))return false;
    }
    // Conservative host admission includes transaction output, new-key projection,
    // native attention staging, selection buffers and the largest scoring pass.
    size_t transaction,slots,keys,score,other,metadata;
    if(!runtime_mul(maps.layers.size(),sizeof(runtime_layer)+64,metadata) ||
       !runtime_add(metadata,sizeof(runtime_request)+64,metadata) ||
       !runtime_mul(metadata,o.max_requests,metadata))return false;
    if(!runtime_mul(size_t(o.max_new_tokens),size_t(s.q_heads)*s.head_dim*sizeof(float),transaction) ||
       !runtime_mul(transaction,2,transaction) ||
       !runtime_mul(size_t(o.query_chunk),size_t(capacity)*sizeof(int32_t),slots) ||
       !runtime_mul(size_t(o.max_new_tokens)+o.query_chunk,index_dimension*sizeof(float),keys) ||
       !runtime_mul(size_t(o.max_keys/cluster_size)*buckets_per_cluster,16,score) ||
       !runtime_add(transaction,slots,other) || !runtime_add(other,keys,other) ||
       !runtime_add(other,score,other) || !runtime_add(other,metadata,other) ||
       !runtime_add(other,1024*1024,host_scratch_admission))return false;
    return map_bytes<=o.max_host_bytes && host_scratch_admission<=o.max_host_bytes-map_bytes;
}
size_t runtime::impl::host_usage() const {
    size_t n=map_bytes+host_scratch_admission;
    for(const auto&r:requests)for(const auto&l:r.second.layers) {
        if(!runtime_add(n,l.second.host_admission,n))return SIZE_MAX;
    }return n;
}
size_t runtime::impl::device_usage() const {
    size_t n=0;
    for(const auto&r:requests) {
        if(!runtime_add(n,r.second.scratch_admission,n))return SIZE_MAX;
        for(const auto&l:r.second.layers)if(!runtime_add(n,l.second.device_admission,n))return SIZE_MAX;
    }return n;
}
runtime_result runtime::impl::dense(dense_reason why,const std::string&detail) {
    runtime_result r;r.reason=why;r.detail=detail;r.generation=generation;r.owner=this;last=r;return r;
}
runtime::runtime(map_bundle maps,runtime_options options):p(new impl) {
    p->maps=std::move(maps);p->options=options;p->configured=p->configure();
}
runtime::~runtime()=default;
bool runtime::valid() const {std::lock_guard<std::mutex>g(p->mutex);return p->configured;}
bool runtime::begin_request(uint64_t slot,uint64_t epoch,runtime_backend backend) {
    std::lock_guard<std::mutex>g(p->mutex);++p->generation;p->last={};
    auto found=p->requests.find(slot);
    if(!p->configured || (backend!=runtime_backend::cpu && backend!=runtime_backend::hip) ||
       (found==p->requests.end() && p->requests.size()>=p->options.max_requests) ||
       (found!=p->requests.end() && epoch<=found->second.epoch))return false;
    try {auto &r=p->requests[slot];r=runtime_request{};r.epoch=epoch;r.backend=backend;return true;}
    catch(const std::bad_alloc&){return false;}
}
void runtime::invalidate(uint64_t slot,invalidation) {
    std::lock_guard<std::mutex>g(p->mutex);++p->generation;p->last={};
    auto i=p->requests.find(slot);if(i!=p->requests.end()){i->second.layers.clear();i->second.hip_scratch.reset();i->second.scratch_admission=0;i->second.blocked=true;}
}
bool runtime::replace_maps(map_bundle maps) {
    std::lock_guard<std::mutex>g(p->mutex);++p->generation;p->last={};
    impl test;test.maps=std::move(maps);test.options=p->options;
    if(!test.configure() || test.maps.model_identity!=p->maps.model_identity || !same_shape(test.maps.shape,p->maps.shape))return false;
    p->maps=std::move(test.maps);p->map_bytes=test.map_bytes;p->host_scratch_admission=test.host_scratch_admission;
    for(auto&r:p->requests){r.second.layers.clear();r.second.hip_scratch.reset();r.second.scratch_admission=0;r.second.blocked=true;}return true;
}
bool runtime::truncate(uint64_t slot,uint64_t epoch,int64_t count) {
    std::lock_guard<std::mutex>g(p->mutex);++p->generation;p->last={};
    auto i=p->requests.find(slot);if(i==p->requests.end() || i->second.epoch!=epoch || count<0 || i->second.blocked)return false;
    bool ok=true;
    for(auto &entry:i->second.layers){auto &l=entry.second;
        const bool cut=!l.poisoned && (i->second.backend==runtime_backend::cpu ? l.cpu.truncate(epoch,count) : p->hip_truncate_layer(l,epoch,count));
        if(!cut){l.poisoned=true;ok=false;}
    }return ok;
}
runtime_result runtime::execute(const runtime_frame &f) {
    std::lock_guard<std::mutex>g(p->mutex);++p->generation;p->last={};
    if(!p->options.enabled)return p->dense(dense_reason::feature_disabled);
    if(!p->configured)return p->dense(dense_reason::invalid_configuration);
    auto ri=p->requests.find(f.slot);if(ri==p->requests.end())return p->dense(dense_reason::unknown_request);
    auto &r=ri->second;if(r.epoch!=f.epoch)return p->dense(dense_reason::stale_epoch);
    if(r.blocked)return p->dense(dense_reason::invalidated);
    if(!f.single_sequence || !f.causal_text || f.dca || f.speculative || f.mtp)return p->dense(dense_reason::unsupported_request);
    if(!f.contiguous_prefix)return p->dense(dense_reason::noncontiguous_slots);
    auto mi=p->maps.layers.find(f.layer);if(mi==p->maps.layers.end())return p->dense(dense_reason::missing_layer);
    if(f.map_identity!=mi->second.identity)return p->dense(dense_reason::missing_history,"map identity differs from graph binding");
    const auto &s=p->maps.shape;const auto n=f.logical_q.rows;
    if(n<=0 || n>p->options.max_new_tokens || f.first<0 || f.first>p->options.max_keys-n)return p->dense(dense_reason::capacity);
    if(!f.positions || !f.slots)return p->dense(dense_reason::noncontiguous_slots);
    for(int64_t j=0;j<n;++j)if(f.positions[j]!=f.first+j || f.slots[j]!=f.first+j)return p->dense(dense_reason::noncontiguous_slots);
    if(!runtime_view(f.logical_q,n,s.q_heads,s.head_dim,true) || !runtime_view(f.logical_k,n,s.kv_heads,s.head_dim,true) ||
       !runtime_view(f.physical_q,n,s.q_heads,s.head_dim,true) || f.physical_q.type!=GGML_TYPE_F32 ||
       f.full_k.rows<f.first+n || f.full_v.rows!=f.full_k.rows ||
       !runtime_view(f.full_k,f.full_k.rows,s.kv_heads,s.head_dim,false) ||
       !runtime_view(f.full_v,f.full_v.rows,s.kv_heads,s.head_dim,false) ||
       !std::isfinite(f.scale) || !std::isfinite(f.softcap) || f.softcap<0)return p->dense(dense_reason::unsupported_layout);
    if(r.backend==runtime_backend::cpu && !cpu_projection_profile_available(s))
        return p->dense(dense_reason::projection,"pinned CPU projection profile unavailable; use dense path");
    try {
        auto &l=r.layers[f.layer];if(l.poisoned)return p->dense(dense_reason::missing_history);
        auto result=r.backend==runtime_backend::cpu ? p->execute_cpu(f,r,l,mi->second) : p->execute_hip(f,r,l,mi->second);
        if(!result.sparse)l.poisoned=true;return result;
    }catch(const std::bad_alloc&){r.layers.clear();r.blocked=true;return p->dense(dense_reason::resource,"host allocation failed");}
}
runtime_result runtime::impl::execute_cpu(const runtime_frame&f,runtime_request&r,runtime_layer&l,const projection_maps&m) {
    const auto n=f.logical_q.rows;const int64_t capacity=int64_t(options.selection.top_k)+options.selection.window+options.selection.sink;
    if(!l.host_admission){
        if(f.first!=0)return dense(dense_reason::missing_history,"fresh logical history must start at zero");
        size_t admission,keys,buckets;
        if(!runtime_mul(size_t(options.max_keys),index_dimension*sizeof(float)*3,keys) ||
           !runtime_mul(size_t(options.max_keys/cluster_size)*buckets_per_cluster,sizeof(bucket)*3,buckets) ||
           !runtime_add(keys,buckets,admission) || !runtime_add(admission,size_t(options.max_keys/cluster_size)*3,admission) ||
           host_usage()>options.max_host_bytes || admission>options.max_host_bytes-host_usage())return dense(dense_reason::resource,"host cache admission cap");
        l.host_admission=admission;l.cpu.reset(r.epoch,m.identity);
    }
    if(!l.cpu.matches(r.epoch,m.identity) || l.cpu.size()!=f.first)return dense(dense_reason::missing_history,"append position/epoch mismatch");
    std::string error;std::vector<float> keys(size_t(n)*index_dimension);
    const auto key_begin=options.collect_phase_times ? phase_clock::now() : phase_clock::time_point{};
    if(!project_keys(f.logical_k,m,keys.data(),&error))return dense(dense_reason::projection,error);
    if(!l.cpu.append(r.epoch,f.first,keys.data(),n,&error))return dense(dense_reason::projection,error);
    const uint64_t key_host_us=options.collect_phase_times ? phase_us(key_begin) : 0;
    output.resize(size_t(n)*m.shape.q_heads*m.shape.head_dim);
    std::vector<float> queries(size_t(options.query_chunk)*index_dimension);
    std::vector<int32_t> indices(size_t(options.query_chunk)*capacity,-1),lengths(options.query_chunk),rows;
    attention_stats total_stats{};
    uint64_t query_host_us=0, attention_host_us=0;
    for(int64_t start=0;start<n;start+=options.query_chunk){
        const int64_t count=std::min<int64_t>(options.query_chunk,n-start);
        const auto query_begin=options.collect_phase_times ? phase_clock::now() : phase_clock::time_point{};
        if(!project_queries(runtime_slice(f.logical_q,start,count),m,queries.data(),&error))return dense(dense_reason::projection,error);
        std::fill(indices.begin(),indices.end(),-1);
        for(int64_t j=0;j<count;++j){
            if(!l.cpu.select(r.epoch,queries.data()+j*index_dimension,f.positions[start+j],options.selection,rows,&error))return dense(dense_reason::ambiguous_selection,error);
            if(rows.size()>size_t(capacity))return dense(dense_reason::capacity,"selection exceeds admitted capacity");
            lengths[j]=int32_t(rows.size());std::copy(rows.begin(),rows.end(),indices.begin()+j*capacity);
        }
        if(options.collect_phase_times)query_host_us+=phase_us(query_begin);
        attention_args a;a.q=runtime_slice(f.physical_q,start,count);a.k=f.full_k;a.v=f.full_v;
        a.main={indices.data(),lengths.data(),capacity};a.positions=f.positions+start;a.scale=f.scale;a.softcap=f.softcap;
        a.output=output.data()+size_t(start)*m.shape.q_heads*m.shape.head_dim;
        attention_stats chunk_stats{};
        attention_stats * stats=options.collect_stats ? &chunk_stats : nullptr;
        const auto attention_begin=options.collect_phase_times ? phase_clock::now() : phase_clock::time_point{};
        bool ok=attention ? attention(a,stats,&error,attention_userdata) : selected_attention_cpu(a,stats,&error);
        if(!ok)return dense(dense_reason::attention,error);
        if(options.collect_phase_times)attention_host_us+=phase_us(attention_begin);
        total_stats.key_rows += chunk_stats.key_rows;
        total_stats.value_rows += chunk_stats.value_rows;
    }
    runtime_result result;result.sparse=true;result.reason=dense_reason::none;result.backend=runtime_backend::cpu;
    result.staged_output=output.data();result.rows=n;result.heads=m.shape.q_heads;result.dim=m.shape.head_dim;
    result.key_rows=total_stats.key_rows;result.value_rows=total_stats.value_rows;
    result.key_append_host_us=key_host_us;
    result.query_select_host_us=query_host_us;
    result.attention_host_us=attention_host_us;
    result.generation=generation;result.owner=this;last=result;return result;
}
bool runtime::lease_valid(const runtime_result&r) const {
    std::lock_guard<std::mutex>g(p->mutex);
    return r.sparse && r.owner==p.get() && r.generation==p->generation && p->last.sparse &&
        r.staged_output==p->last.staged_output && r.rows==p->last.rows && r.heads==p->last.heads &&
        r.dim==p->last.dim && r.backend==p->last.backend && r.device==p->last.device && r.stream==p->last.stream;
}
bool runtime::publish_cpu(const runtime_result&r,float*dst,size_t count) {
    std::lock_guard<std::mutex>g(p->mutex);size_t n;
    if(!r.sparse || r.backend!=runtime_backend::cpu || r.owner!=p.get() || r.generation!=p->generation ||
       !p->last.sparse || r.staged_output!=p->output.data() || r.staged_output!=p->last.staged_output ||
       r.rows!=p->last.rows || r.heads!=p->last.heads || r.dim!=p->last.dim || !dst ||
       !runtime_mul(size_t(r.rows),size_t(r.heads)*r.dim,n) || count!=n)return false;
    std::memmove(dst,r.staged_output,n*sizeof(float));return true;
}
int64_t runtime::indexed_rows(uint64_t slot,int layer) const {
    std::lock_guard<std::mutex>g(p->mutex);auto r=p->requests.find(slot);if(r==p->requests.end())return -1;
    auto l=r->second.layers.find(layer);if(l==r->second.layers.end() || l->second.poisoned)return -1;
    return r->second.backend==runtime_backend::cpu ? l->second.cpu.size() : l->second.hip_rows;
}
size_t runtime::admitted_device_bytes() const {std::lock_guard<std::mutex>g(p->mutex);return p->device_usage();}
void runtime::set_cpu_attention_for_test(cpu_attention_fn fn,void*user) {std::lock_guard<std::mutex>g(p->mutex);++p->generation;p->last={};p->attention=fn;p->attention_userdata=user;}
#ifndef LLAMA_LSA_RUNTIME_HIP
void runtime_destroy_hip(hip_context*){}
void runtime_destroy_scratch(runtime_hip_scratch*){}
bool runtime::impl::hip_truncate_layer(runtime_layer&,uint64_t,int64_t){return false;}
runtime_result runtime::impl::execute_hip(const runtime_frame&,runtime_request&,runtime_layer&,const projection_maps&){return dense(dense_reason::unsupported_layout,"HIP module not built");}
#endif
} // namespace llama_lsa
