#include "llama-lsa-runtime.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <unistd.h>
using namespace llama_lsa;
static void need(bool ok,const char*s){if(!ok){std::fprintf(stderr,"FAIL %s\n",s);std::exit(1);}std::printf("PASS %s\n",s);}
static void checked(hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"HIP %s\n",hipGetErrorString(e));std::exit(1);}}
static void stamp(const char*s){if(const char*p=std::getenv("LSA_HIP_TIMELINE")){std::ofstream out(p,std::ios::app);out<<s<<' '<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()<<'\n';}}
template<class T>struct array {
    T*p=nullptr;size_t n;
    array(size_t n):n(n){checked(hipMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~array(){(void)hipFree(p);}
    void put(const std::vector<T>&v){need(v.size()==n,"bounded fixture upload");checked(hipMemcpy(p,v.data(),n*sizeof(T),hipMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T>v(n);checked(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
static tensor_view view(const void*p,int64_t n,int heads){return {p,GGML_TYPE_F32,n,heads,256,size_t(heads)*256*4,256*4};}
static void witness(){const char*ready=std::getenv("LSA_HIP_WITNESS_READY"),*release=std::getenv("LSA_HIP_WITNESS_RELEASE");need(ready && release,"external in-window observer configured");{std::ofstream f(ready);f<<getpid()<<'\n';}
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);while(!std::ifstream(release).good()){if(std::chrono::steady_clock::now()>=deadline)need(false,"observer release within deadline");std::this_thread::sleep_for(std::chrono::milliseconds(2));}}
int main(int argc,char**argv){
    if(argc!=2 || std::strcmp(argv[1],"--run-runtime-hip")){std::puts("SKIP: explicit --run-runtime-hip plus held CPU/GPU/device claims required");return 77;}
    checked(hipSetDevice(0));hipDeviceProp_t prop;checked(hipGetDeviceProperties(&prop,0));need(std::strstr(prop.gcnArchName,"gfx90a"),"device0 is gfx90a");
    hipStream_t stream;checked(hipStreamCreate(&stream));
    {
        map_bundle maps;maps.shape={24,4,256,64};maps.model_identity="analytic-runtime-device-unit";
        projection_maps m;m.shape=maps.shape;m.identity="analytic-native";m.q.resize(768*64);m.k.resize(768*64);
        for(int i=0;i<64;++i)m.q[i*64+i]=m.k[i*64+i]=ggml_fp32_to_bf16(1);maps.layers.emplace(3,m);
        runtime_options o;o.enabled=true;o.max_keys=128;o.max_new_tokens=65;o.query_chunk=32;o.selection.top_k=16;
        o.max_host_bytes=64*1024*1024;o.max_device_bytes=8*1024*1024;
        runtime r(maps,o);need(r.valid() && r.begin_request(1,1,runtime_backend::hip),"valid HIP runtime request");
        std::vector<float>q(66*24*256),k(66*4*256),zeroq(q.size()),zerok(k.size()),values(k.size()),sentinel(65*24*256,-999);
        std::vector<int32_t>positions(66),slots(66);
        for(int i=0;i<66;++i){q[i*24*256+64]=k[i*4*256+64]=1;positions[i]=slots[i]=i;std::fill(values.begin()+i*4*256,values.begin()+(i+1)*4*256,float(i));}
        array<float>dq(q.size()),dk(k.size()),pq(q.size()),pk(k.size()),v(values.size()),destination(sentinel.size());
        dq.put(q);dk.put(k);pq.put(zeroq);pk.put(zerok);v.put(values);destination.put(sentinel);
        auto frame=[&](int first,int count,uint64_t epoch){runtime_frame f;f.slot=1;f.epoch=epoch;f.layer=3;f.map_identity=m.identity;f.first=first;
            f.positions=positions.data()+first;f.slots=slots.data()+first;f.logical_q=view(dq.p+first*24*256,count,24);f.logical_k=view(dk.p+first*4*256,count,4);
            f.physical_q=view(pq.p+first*24*256,count,24);f.full_k=view(pk.p,66,4);f.full_v=view(v.p,66,4);f.scale=1.f/16;f.stream=reinterpret_cast<void*>(stream);return f;};
        witness();stamp("first_launch");
        auto result=r.execute(frame(0,65,1));need(result.sparse && r.lease_valid(result),"65 new keys and32/32/1 device query chunks succeed");
        need(result.backend==runtime_backend::hip && result.device==0 && result.stream==reinterpret_cast<void*>(stream),"lease retains backend device and stream");
        checked(hipMemcpyAsync(destination.p,result.staged_output,sentinel.size()*4,hipMemcpyDeviceToDevice,stream));checked(hipStreamSynchronize(stream));
        auto output=destination.get();bool equal=true;for(int i=0;i<65;++i)for(int j=0;j<24*256;++j)equal &= std::fabs(output[i*24*256+j]-float(i)/2)<1e-5;
        need(equal,"device staged output equals independent causal-V mean");
        auto decode=r.execute(frame(65,1,1));need(decode.sparse && r.indexed_rows(1,3)==66,"HIP cache append touches one new logical key");
        need(!r.lease_valid(result),"later append invalidates old device output lease");
        need(r.truncate(1,1,64) && r.indexed_rows(1,3)==64 && r.execute(frame(64,2,1)).sparse,"HIP suffix truncate and replacement work");
        r.invalidate(1,invalidation::state_restore);need(!r.execute(frame(0,1,1)).sparse,"restore invalidates HIP request");
        need(r.begin_request(1,2,runtime_backend::hip),"fresh epoch for nonzero-history guard");
        need(!r.execute(frame(10,1,2)).sparse && !r.execute(frame(0,1,2)).sparse,"nonzero restored history stays dense until explicit fresh epoch");
        need(r.begin_request(1,3,runtime_backend::hip),"fresh epoch for host pointer rejection");auto host=frame(0,1,3);host.logical_q.data=q.data();
        auto rejected=r.execute(host);need(!rejected.sparse && rejected.reason==dense_reason::unsupported_layout,"host tensor pointer rejected before device launch");
        need(r.begin_request(1,4,runtime_backend::hip) && r.execute(frame(0,1,4)).sparse,"expected pointer-validation error does not poison subsequent valid launch");
        r.begin_request(1,5,runtime_backend::hip);values[40*4*256]=std::numeric_limits<float>::quiet_NaN();v.put(values);destination.put(sentinel);
        auto failed=r.execute(frame(0,65,5));need(!failed.sparse && failed.reason==dense_reason::attention && !failed.staged_output,"NaN in later chunk returns dense with no partial lease");
        need(destination.get()==sentinel,"late failure leaves unpublished destination sentinel untouched");
        need(!r.execute(frame(65,1,5)).sparse && r.indexed_rows(1,3)==-1,"failed HIP layer cannot be reused or appended twice");
        values[40*4*256]=40;v.put(values);r.begin_request(1,6,runtime_backend::hip);need(r.execute(frame(0,2,6)).sparse,"fresh zero epoch recovers after failed HIP transaction");
        auto small=o;small.max_device_bytes=1;runtime limited(maps,small);limited.begin_request(1,1,runtime_backend::hip);
        auto denied=limited.execute(frame(0,1,1));need(!denied.sparse && denied.reason==dense_reason::resource && limited.admitted_device_bytes()==0,"aggregate device admission denies before context allocation");
        stamp("last_finish");
    }
    checked(hipStreamDestroy(stream));std::puts("RESULT runtime HIP integration passed; synthetic only");return 0;
}
