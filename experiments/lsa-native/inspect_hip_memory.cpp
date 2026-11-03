#include "llama-lsa-native-hip.h"
#include <cstdio>
#include <limits>

// Only hip_memory_requirements is called: pure host arithmetic, no device init.
int main() {
    using namespace llama_lsa;
    for (geometry shape : {geometry{24,4,256,64}, geometry{16,2,256,64}}) {
        for (int64_t keys : {int64_t(131072),int64_t(1048576)}) {
            for (int64_t queries : {int64_t(32),int64_t(2048)}) {
                hip_limits limits{keys, queries, 4612, 0};
                hip_memory bytes; std::string error;
                if (hip_memory_requirements(shape, limits, &bytes, &error) != hip_result::success) {
                    std::fprintf(stderr,"memory query rejected: %s\n",error.c_str()); return 1;
                }
                std::printf("{\"q_heads\":%d,\"kv_heads\":%d,\"max_keys\":%lld,\"max_queries\":%lld,\"max_slots\":4612,\"persistent_bytes\":%zu,\"workspace_bytes\":%zu,\"total_device_bytes\":%zu,\"sixteen_layer_bytes\":%zu}\n",
                    shape.q_heads,shape.kv_heads,(long long)keys,(long long)queries,
                    bytes.persistent_bytes,bytes.workspace_bytes,bytes.total_device_bytes,
                    bytes.total_device_bytes*16);
                limits.max_device_bytes=bytes.total_device_bytes-1;
                if (hip_memory_requirements(shape,limits,&bytes,&error)!=hip_result::resource_error) {
                    std::fprintf(stderr,"undersized admission cap accepted\n"); return 1;
                }
            }
        }
    }
    hip_limits invalid{std::numeric_limits<int64_t>::max(),32,4612,0};
    hip_memory bytes;std::string error;
    if (hip_memory_requirements(geometry{24,4,256,64},invalid,&bytes,&error)!=hip_result::invalid_argument) {
        std::fprintf(stderr,"oversized index accepted\n");return 1;
    }
    return 0;
}
