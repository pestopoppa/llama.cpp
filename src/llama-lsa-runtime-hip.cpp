#include "llama-lsa-runtime-impl.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace llama_lsa {
using hip_phase_clock = std::chrono::steady_clock;
static uint64_t hip_phase_us(hip_phase_clock::time_point begin) {
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(hip_phase_clock::now() - begin).count());
}
// Diagnostic wire format: magic[8], nine little-endian u32 values (key count,
// capacity, selected length, layer, position, top_k, top_p, window, sink),
// slot/epoch as two little-endian u64, then model/map/map-source SHA256 as 64 ASCII
// hex bytes each, then F32 query[64], F32 keys[key_count][64], I32 indices[length].
// Native runs are Linux/x86-64 little-endian. The offline reader rejects all
// other sizes and provenance; this record is not timing or prompt-identity proof.
static bool hex_digest(const std::string & s) {
    if (s.size()!=64) return false;
    for (char c:s) if (!((c>='0'&&c<='9')||(c>='a'&&c<='f'))) return false;
    return true;
}
static bool write_all(int fd,const void * data,size_t bytes) {
    const char * p=static_cast<const char*>(data);
    while(bytes) {const ssize_t n=::write(fd,p,bytes);if(n<0&&errno==EINTR)continue;if(n<=0)return false;p+=n;bytes-=size_t(n);}
    return true;
}
static bool write_witness_file(const runtime_options & o,const map_bundle & maps,
                               const runtime_frame & f,int32_t length,int64_t capacity,
                               const std::vector<float> & query,const std::vector<float> & keys,
                               const std::vector<int32_t> & indices,std::string & error) {
    if (o.index_witness_path.empty() || o.index_witness_path[0]!='/' ||
        o.index_witness_path.size()>4096 || !hex_digest(o.witness_model_sha256) ||
        !hex_digest(o.witness_map_sha256) || !hex_digest(maps.source_sha256)) {
        error="invalid witness path or digest provenance";return false;
    }
    std::string temp=o.index_witness_path+".tmp.XXXXXX";
    std::vector<char> path(temp.begin(),temp.end());path.push_back('\0');
    const int fd=::mkstemp(path.data());
    if(fd<0){error="cannot create witness temporary file";return false;}
    bool permissions_ok=::fchmod(fd,0600)==0;
    const char magic[8]={'L','S','A','W','I','T','1','\0'};
    const std::array<uint32_t,9> fields={5132U,uint32_t(capacity),uint32_t(length),
        uint32_t(f.layer),5131U,uint32_t(o.selection.top_k),uint32_t(o.selection.top_p),
        uint32_t(o.selection.window),uint32_t(o.selection.sink)};
    const std::array<uint64_t,2> request={f.slot,f.epoch};
    bool ok=permissions_ok && write_all(fd,magic,sizeof(magic)) && write_all(fd,fields.data(),sizeof(fields)) &&
        write_all(fd,request.data(),sizeof(request)) &&
        write_all(fd,o.witness_model_sha256.data(),64) && write_all(fd,o.witness_map_sha256.data(),64) &&
        write_all(fd,maps.source_sha256.data(),64) &&
        write_all(fd,query.data(),query.size()*sizeof(float)) &&
        write_all(fd,keys.data(),keys.size()*sizeof(float)) &&
        write_all(fd,indices.data(),indices.size()*sizeof(int32_t));
    if(ok)ok=::fsync(fd)==0;
    if(::close(fd)!=0)ok=false;
    if(ok)ok=::link(path.data(),o.index_witness_path.c_str())==0; // atomic no-clobber publication
    (void)::unlink(path.data());
    if(!ok){error="witness write/sync/publish failed or destination already exists";return false;}
    return true;
}
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
static bool capture_index_witness(const runtime_options & o,const map_bundle & maps,
                                  const runtime_frame & f,hip_context * context,
                                  const float * device_queries,const int32_t * device_indices,
                                  const int32_t * device_lengths,hipStream_t stream,
                                  int64_t start,int64_t row,int64_t capacity,
                                  std::string & error) {
    constexpr int64_t nkeys=5132;
    // The pinned author oracle's default selector uses this exact tuple; the
    // replay calls its unchanged select(query, keys, positions) entry point.
    if(o.selection.top_k!=4096 || o.selection.top_p!=1024 ||
       o.selection.window!=512 || o.selection.sink!=4 ||
       f.layer!=3 || start+row<0 || start+row>=f.logical_q.rows ||
       f.positions[start+row]!=5131 || hip_size(context)<nkeys ||
       capacity<=0 || capacity>4612 || row<0) {
        error="witness target geometry or position mismatch";return false;
    }
    std::vector<float> query(index_dimension),keys(size_t(nkeys)*index_dimension);
    std::vector<int32_t> indices(static_cast<size_t>(capacity));
    std::vector<int32_t> length(1);
    if(!device_queries || !device_indices || !device_lengths || !hip_index_keys(context)) {
        error="witness device buffer missing";return false;
    }
    hipError_t rc=hipMemcpyAsync(length.data(),device_lengths+row,sizeof(int32_t),hipMemcpyDeviceToHost,stream);
    if(rc==hipSuccess)rc=hipMemcpyAsync(query.data(),device_queries+row*index_dimension,
                                        query.size()*sizeof(float),hipMemcpyDeviceToHost,stream);
    if(rc==hipSuccess)rc=hipMemcpyAsync(keys.data(),hip_index_keys(context),
                                        keys.size()*sizeof(float),hipMemcpyDeviceToHost,stream);
    if(rc==hipSuccess)rc=hipMemcpyAsync(indices.data(),device_indices+row*capacity,
                                        indices.size()*sizeof(int32_t),hipMemcpyDeviceToHost,stream);
    if(rc==hipSuccess)rc=hipStreamSynchronize(stream); // one extra sync, diagnostic only
    if(rc!=hipSuccess){error="witness HIP copy/sync failed";return false;}
    if(length[0]<0 || length[0]>capacity){error="witness selection length out of bounds";return false;}
    indices.resize(size_t(length[0]));
    for(int32_t i:indices)if(i<0 || i>=nkeys){error="witness selected index out of bounds";return false;}
    return write_witness_file(o,maps,f,length[0],capacity,query,keys,indices,error);
}
bool hip_capture_index_witness_for_test(const runtime_options & o,const map_bundle & maps,
                                        const runtime_frame & f,hip_context * context,
                                        const float * queries,const int32_t * indices,
                                        const int32_t * lengths,void * stream,
                                        int64_t capacity,std::string & error) {
    if(f.logical_q.rows!=1 || !f.positions || f.positions[0]!=5131 || f.first!=5131) {
        error="test witness frame mismatch";return false;
    }
    return capture_index_witness(o,maps,f,context,queries,indices,lengths,
                                 reinterpret_cast<hipStream_t>(stream),0,0,capacity,error);
}
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
    const auto key_begin=options.collect_phase_times ? hip_phase_clock::now() : hip_phase_clock::time_point{};
    auto rc=hip_append(l.hip.get(),r.epoch,f.first,f.logical_k,&error);
    if(rc!=hip_result::success)return dense(classify(rc,dense_reason::projection),error);
    l.hip_rows+=f.logical_k.rows;
    const uint64_t key_host_us=options.collect_phase_times ? hip_phase_us(key_begin) : 0;
    attention_stats total_stats{};
    uint64_t query_host_us=0, attention_host_us=0;
    for(int64_t start=0;start<f.logical_q.rows;start+=options.query_chunk) {
        const int64_t count=std::min<int64_t>(options.query_chunk,f.logical_q.rows-start);
        const auto query_begin=options.collect_phase_times ? hip_phase_clock::now() : hip_phase_clock::time_point{};
        if(hipMemcpyAsync(s.positions,f.positions+start,size_t(count)*sizeof(int32_t),hipMemcpyHostToDevice,s.stream)!=hipSuccess)
            return dense(dense_reason::device,"position upload failed");
        rc=hip_project_queries(l.hip.get(),runtime_slice(f.logical_q,start,count),s.queries,&error);
        if(rc!=hip_result::success)return dense(classify(rc,dense_reason::projection),error);
        rc=hip_select(l.hip.get(),r.epoch,s.queries,count,s.positions,options.selection,s.indices,s.lengths,capacity,&error);
        if(rc!=hip_result::success)return dense(classify(rc,dense_reason::ambiguous_selection),error);
        if(!options.index_witness_path.empty() && f.layer==3 &&
           f.first<=5131 && f.first+f.logical_q.rows>5131) {
            const int64_t row=5131-f.first-start;
            if(row>=0 && row<count &&
               !capture_index_witness(options,maps,f,l.hip.get(),s.queries,s.indices,s.lengths,
                                      s.stream,start,row,capacity,error))
                return dense(dense_reason::device,error);
        }
        if(options.collect_phase_times)query_host_us+=hip_phase_us(query_begin);
        attention_args a;a.q=runtime_slice(f.physical_q,start,count);a.k=f.full_k;a.v=f.full_v;
        a.main={s.indices,s.lengths,capacity};a.positions=s.positions;a.scale=f.scale;a.softcap=f.softcap;
        a.output=s.output+size_t(start)*m.shape.q_heads*m.shape.head_dim;
        attention_stats chunk_stats{};
        const auto attention_begin=options.collect_phase_times ? hip_phase_clock::now() : hip_phase_clock::time_point{};
        rc=hip_selected_attention(l.hip.get(),a,options.collect_stats ? &chunk_stats : nullptr,&error);
        if(rc!=hip_result::success)return dense(classify(rc,dense_reason::attention),error);
        if(options.collect_phase_times)attention_host_us+=hip_phase_us(attention_begin);
        total_stats.key_rows+=chunk_stats.key_rows;
        total_stats.value_rows+=chunk_stats.value_rows;
    }
    runtime_result result;result.sparse=true;result.reason=dense_reason::none;result.backend=runtime_backend::hip;
    result.staged_output=s.output;result.rows=f.logical_q.rows;result.heads=m.shape.q_heads;result.dim=m.shape.head_dim;
    result.key_rows=total_stats.key_rows;result.value_rows=total_stats.value_rows;
    result.key_append_host_us=key_host_us;
    result.query_select_host_us=query_host_us;
    result.attention_host_us=attention_host_us;
    result.generation=generation;result.owner=this;result.device=device;result.stream=f.stream;last=result;return result;
}
} // namespace llama_lsa
