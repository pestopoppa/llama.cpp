#include "llama-lsa-map.h"
#include <cstdio>
#include <sstream>
#include <fstream>
int main(int argc,char **argv) {
    if(argc!=6 && argc!=7) {std::fprintf(stderr,"usage: lsa-inspect-map PATH MODEL_ID SOURCE_SHA256 27b|35b LAYER_IDS [DUMP_BF16]\n");return 2;}
    std::string kind=argv[4],piece;llama_lsa::geometry shape;
    if(kind=="27b")shape={24,4,256,64};else if(kind=="35b")shape={16,2,256,64};else return 2;
    std::vector<int> layers;std::istringstream ids(argv[5]);
    try{while(std::getline(ids,piece,','))layers.push_back(std::stoi(piece));}catch(...){return 2;}
    llama_lsa::map_bundle maps;std::string error;
    if(!llama_lsa::load_map_bundle(argv[1],argv[2],shape,layers,argv[3],maps,&error)) {std::fprintf(stderr,"rejected: %s\n",error.c_str());return 1;}
    if(argc==7) {
        std::ofstream out(argv[6],std::ios::binary);
        for(const auto & entry:maps.layers) { uint32_t layer=entry.first;for(int i=0;i<4;++i)out.put(char(layer>>(8*i)));for(const auto * b:{&entry.second.q,&entry.second.k})for(auto x:*b){out.put(char(x.bits));out.put(char(x.bits>>8));} }
        if(!out){std::fprintf(stderr,"dump failed\n");return 1;}
    }
    std::printf("validated layers=%zu rows=%d dimensions=64 model=%s source=%s payload=%s\n",maps.layers.size(),shape.map_rows(),maps.model_identity.c_str(),maps.source_sha256.c_str(),maps.payload_sha256.c_str());return 0;
}
