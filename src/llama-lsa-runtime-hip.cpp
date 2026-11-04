#include "llama-lsa-runtime-impl.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>

namespace llama_lsa {
struct runtime_hip_scratch {
    int device = -1;
    hipStream_t stream = nullptr;
    float * output = nullptr, * queries = nullptr;
    int32_t * indices = nullptr, * lengths = nullptr, * positions = nullptr;
    ~runtime_hip_scratch() {
        if(device<0)return;
        int old=-1;const bool restore=hipGetDevice(&old)==hipSuccess && old!=device;
        if(restore)(void)hipSetDevice(device);
        (void)hipStreamSynchronize(stream);
        (void)hipFree(output);(void)hipFree(queries);(void)hipFree(indices);
        (void)hipFree(lengths);(void)hipFree(positions);
        if(restore)(void)hipSetDevice(old);
    }
};
void runtime_destroy_hip(hip_context*c){hip_destroy(c);}
void runtime_destroy_scratch(runtime_hip_scratch*s){delete s;}
bool runtime::impl::hip_truncate_layer(runtime_layer&l,uint64_t epoch,int64_t count) {
    if(!l.hip || hip_truncate(l.hip.get(),epoch,count)!=hip_result::success)return false;
    l.hip_rows=count;return true;
}
static dense_reason classify(hip_result r,dense_reason phase) {
    if(r==hip_result::device_error)return dense_reason::device;
    if(r==hip_result::resource_error)return dense_reason::resource;
    if(r==hip_result::unsupported)return dense_reason::unsupported_layout;
    return phase;
}
static bool device_view(const tensor_view&v,int device) {
    hipPointerAttribute_t attrs{};
    if(hipPointerGetAttributes(&attrs,v.data)!=hipSuccess) {
        (void)hipGetLastError(); // consume expected pointer-validation error only
        return false;
    }
    if(attrs.type!=hipMemoryTypeDevice || attrs.device!=device || attrs.isManaged)return false;
    hipDeviceptr_t base=nullptr;size_t allocation=0,last,end;
    if(hipMemGetAddressRange(&base,&allocation,const_cast<void*>(v.data))!=hipSuccess) {
        (void)hipGetLastError();return false;
    }
    if(!runtime_mul(size_t(v.rows-1),v.row_stride,last) ||
       !runtime_add(last,size_t(v.heads-1)*v.head_stride,end) ||
       !runtime_add(end,ggml_row_size(v.type,v.dim),end))return false;
    const uintptr_t offset=reinterpret_cast<uintptr_t>(v.data)-reinterpret_cast<uintptr_t>(base);
    return offset<=allocation && end<=allocation-offset;
}
runtime_result runtime::impl::execute_hip(const runtime_frame&f,runtime_request&r,runtime_layer&l,const projection_maps&m) {
    const int64_t capacity=int64_t(options.selection.top_k)+options.selection.window+options.selection.sink;
    if(!options.max_device_bytes)return dense(dense_reason::resource,"HIP aggregate admission cap required");
    // All count arithmetic and aggregate admission happen before any allocation.
    hip_limits limits{options.max_keys,options.query_chunk,capacity,0};hip_memory memory;std::string error;
    if(hip_memory_requirements(m.shape,limits,&memory,&error)!=hip_result::success)return dense(dense_reason::resource,error);
    size_t output_bytes,query_bytes,index_bytes,position_bytes,scratch_bytes,next;
    if(!runtime_mul(size_t(options.max_new_tokens),size_t(m.shape.q_heads)*m.shape.head_dim*sizeof(float),output_bytes) ||
       !runtime_mul(size_t(options.query_chunk),index_dimension*sizeof(float),query_bytes) ||
       !runtime_mul(size_t(options.query_chunk),size_t(capacity)*sizeof(int32_t),index_bytes) ||
       !runtime_mul(size_t(options.query_chunk),sizeof(int32_t),position_bytes) ||
       !runtime_add(output_bytes,query_bytes,scratch_bytes) || !runtime_add(scratch_bytes,index_bytes,scratch_bytes) ||
       !runtime_add(scratch_bytes,position_bytes*2,scratch_bytes))return dense(dense_reason::capacity,"HIP scratch overflow");
    next=device_usage();
    if((!l.hip && !runtime_add(next,memory.total_device_bytes,next)) ||
       (!r.hip_scratch && !runtime_add(next,scratch_bytes,next)) || next>options.max_device_bytes)
        return dense(dense_reason::resource,"aggregate live HIP contexts/scratch exceed admission cap");
    int device=-1;
    if(hipGetDevice(&device)!=hipSuccess)return dense(dense_reason::device,"cannot resolve current HIP device");
    for(const auto*v:{&f.logical_q,&f.logical_k,&f.physical_q,&f.full_k,&f.full_v})
        if(!device_view(*v,device))return dense(dense_reason::unsupported_layout,"HIP tensor is not a bounded same-device allocation");
    for(const auto*v:{&f.full_k,&f.full_v})
        if(v->type!=GGML_TYPE_F32 && v->type!=GGML_TYPE_F16 && v->type!=GGML_TYPE_BF16 && v->type!=GGML_TYPE_Q8_0)
            return dense(dense_reason::unsupported_layout,"HIP KV type unsupported");
    if(r.hip_scratch && (r.hip_scratch->device!=device || r.hip_scratch->stream!=reinterpret_cast<hipStream_t>(f.stream)))
        return dense(dense_reason::device,"request backend device/stream changed");
    if(!l.hip) {
        if(f.first!=0)return dense(dense_reason::missing_history,"fresh logical HIP history must start at zero");
        limits.max_device_bytes=memory.total_device_bytes;
        hip_context*context=nullptr;
        if(hip_create(m,limits,f.stream,&context,&error)!=hip_result::success)return dense(dense_reason::resource,error);
        l.hip.reset(context);l.device_admission=memory.total_device_bytes;
        if(hip_reset(context,r.epoch,m.identity,&error)!=hip_result::success)return dense(dense_reason::device,error);
    }
    if(l.hip_rows!=f.first)return dense(dense_reason::missing_history,"HIP append position mismatch");
    if(!r.hip_scratch) {
        std::unique_ptr<runtime_hip_scratch,runtime_scratch_deleter>s(new runtime_hip_scratch);
        s->device=device;s->stream=reinterpret_cast<hipStream_t>(f.stream);
        hipError_t rc=hipSuccess;
        auto alloc=[&](auto**dst,size_t bytes){if(rc==hipSuccess)rc=hipMalloc(reinterpret_cast<void**>(dst),bytes);};
        alloc(&s->output,output_bytes);alloc(&s->queries,query_bytes);alloc(&s->indices,index_bytes);
        alloc(&s->positions,position_bytes);alloc(&s->lengths,position_bytes);
        if(rc!=hipSuccess)return dense(dense_reason::resource,hipGetErrorString(rc));
        r.hip_scratch=std::move(s);r.scratch_admission=scratch_bytes;
    }
    auto &s=*r.hip_scratch;
    auto rc=hip_append(l.hip.get(),r.epoch,f.first,f.logical_k,&error);
    if(rc!=hip_result::success)return dense(classify(rc,dense_reason::projection),error);
    l.hip_rows+=f.logical_k.rows;
    attention_stats total_stats{};
    for(int64_t start=0;start<f.logical_q.rows;start+=options.query_chunk) {
        const int64_t count=std::min<int64_t>(options.query_chunk,f.logical_q.rows-start);
        if(hipMemcpyAsync(s.positions,f.positions+start,size_t(count)*sizeof(int32_t),hipMemcpyHostToDevice,s.stream)!=hipSuccess)
            return dense(dense_reason::device,"position upload failed");
        rc=hip_project_queries(l.hip.get(),runtime_slice(f.logical_q,start,count),s.queries,&error);
        if(rc!=hip_result::success)return dense(classify(rc,dense_reason::projection),error);
        rc=hip_select(l.hip.get(),r.epoch,s.queries,count,s.positions,options.selection,s.indices,s.lengths,capacity,&error);
        if(rc!=hip_result::success)return dense(classify(rc,dense_reason::ambiguous_selection),error);
        attention_args a;a.q=runtime_slice(f.physical_q,start,count);a.k=f.full_k;a.v=f.full_v;
        a.main={s.indices,s.lengths,capacity};a.positions=s.positions;a.scale=f.scale;a.softcap=f.softcap;
        a.output=s.output+size_t(start)*m.shape.q_heads*m.shape.head_dim;
        attention_stats chunk_stats{};
        rc=hip_selected_attention(l.hip.get(),a,options.collect_stats ? &chunk_stats : nullptr,&error);
        if(rc!=hip_result::success)return dense(classify(rc,dense_reason::attention),error);
        total_stats.key_rows+=chunk_stats.key_rows;
        total_stats.value_rows+=chunk_stats.value_rows;
    }
    runtime_result result;result.sparse=true;result.reason=dense_reason::none;result.backend=runtime_backend::hip;
    result.staged_output=s.output;result.rows=f.logical_q.rows;result.heads=m.shape.q_heads;result.dim=m.shape.head_dim;
    result.key_rows=total_stats.key_rows;result.value_rows=total_stats.value_rows;
    result.generation=generation;result.owner=this;result.device=device;result.stream=f.stream;last=result;return result;
}
} // namespace llama_lsa
