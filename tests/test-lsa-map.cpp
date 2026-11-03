#include "llama-lsa-map.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
using namespace llama_lsa;
static int failures=0;
static void check(bool b,const char * s){std::printf("%s %s\n",b?"PASS":"FAIL",s);failures+=!b;}
static void u32(std::vector<uint8_t>&b,uint32_t x){for(int i=0;i<4;++i)b.push_back(uint8_t(x>>(8*i)));}
static void digest(std::vector<uint8_t>&b,const std::string&s){for(int i=0;i<32;++i)b.push_back(uint8_t(std::stoul(s.substr(i*2,2),nullptr,16)));}
static std::vector<uint8_t> fixture(geometry shape,const std::vector<int>&layers){
    const std::string model="test-model@sha256:fixture",id="analytic";
    std::vector<uint8_t> p(model.begin(),model.end());p.insert(p.end(),id.begin(),id.end());
    for(int layer:layers){u32(p,layer);for(int buf=0;buf<2;++buf)for(int i=0;i<shape.map_rows()*index_dimension;++i){p.push_back(0x80);p.push_back(0x3f);}}
    std::vector<uint8_t>b{'L','S','A','M','A','P','1',0};
    for(uint32_t x:{1u,uint32_t(shape.q_heads),uint32_t(shape.kv_heads),uint32_t(shape.head_dim),uint32_t(shape.rope_dim),64u,uint32_t(layers.size()),uint32_t(model.size()),uint32_t(id.size())})u32(b,x);
    digest(b,std::string(64,'a'));digest(b,map_sha256(p.data(),p.size()));b.insert(b.end(),p.begin(),p.end());return b;
}
static void rehash(std::vector<uint8_t>&b){std::vector<uint8_t>d;digest(d,map_sha256(b.data()+108,b.size()-108));std::copy(d.begin(),d.end(),b.begin()+76);}
int main(int argc,char**argv){
    if(argc!=2){std::fprintf(stderr,"usage: test-lsa-map SCRATCH_FILE\n");return 2;}
    check(map_sha256(nullptr,0)=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256 empty known vector");
    check(map_sha256("abc",3)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 abc known vector");
    const std::string path=argv[1], model="test-model@sha256:fixture",source(64,'a');
    std::string error;map_bundle bundle;
    auto load=[&](const std::vector<uint8_t>&b,const geometry &shape,const std::vector<int>&layers,const std::string&m,const std::string&src){std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(b.data()),b.size());out.close();return load_map_bundle(path,m,shape,layers,src,bundle,&error);};
    for(auto shape:{geometry{24,4,256,64},geometry{16,2,256,64}}){
        const auto good=fixture(shape,{3,7});
        check(load(good,shape,{3,7},model,source),"valid map geometry, BF16 buffers and exact layer coverage load");
        check(bundle.layers.size()==2 && bundle.layers.at(3).q.size()==size_t(shape.map_rows())*64 && bundle.layers.at(3).q[0].bits==0x3f80,"Q/K BF16 buffers and per-layer identity retained");
        auto bad=good;bad[0]='X';check(!load(bad,shape,{3,7},model,source),"bad magic rejected");
        bad=good;bad[8]=2;check(!load(bad,shape,{3,7},model,source),"unknown format version rejected");
        bad=good;bad[12]=1;check(!load(bad,shape,{3,7},model,source),"wrong geometry rejected");
        bad=good;bad.pop_back();check(!load(bad,shape,{3,7},model,source),"truncated map rejected");
        bad=good;bad.back()^=1;check(!load(bad,shape,{3,7},model,source),"payload digest mismatch rejected");
        check(!load(good,shape,{3,7},model,std::string(64,'b')),"unapproved source digest rejected");
        check(!load(good,shape,{3,7},"other-model",source),"model binding mismatch rejected");
        check(!load(good,shape,{3,11},model,source),"wrong global layer coverage rejected");
        bad=good;bad.push_back(0);rehash(bad);check(!load(bad,shape,{3,7},model,source),"hash-valid trailing data rejected");
        bad=good;bad[108+model.size()+8+4]=0x80;bad[108+model.size()+8+5]=0x7f;rehash(bad);check(!load(bad,shape,{3,7},model,source),"hash-valid BF16 infinity rejected");
        bad=fixture(shape,{3,3});check(!load(bad,shape,{3,7},model,source),"hash-valid duplicate layers rejected");
        bad=good;bad[36]=0xff;bad[37]=0xff;check(!load(bad,shape,{3,7},model,source),"oversized identity length rejected without allocation");
        check(bundle.model_identity==model && bundle.layers.size()==2,"rejected map leaves previously loaded bundle unchanged");
    }
    std::remove(path.c_str());return failures?1:0;
}
