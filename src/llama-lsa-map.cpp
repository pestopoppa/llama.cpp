#include "llama-lsa-map.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <iomanip>

namespace llama_lsa {
namespace {
uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32-n)); }
const uint32_t constants[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
struct reader {
    const std::vector<uint8_t> & bytes; size_t at = 0;
    bool read(void * dst, size_t n) { if (at > bytes.size() || n > bytes.size()-at) return false; std::memcpy(dst, bytes.data()+at, n); at += n; return true; }
    bool u32(uint32_t & n) { uint8_t b[4]; if (!read(b,4)) return false; n=uint32_t(b[0])|uint32_t(b[1])<<8|uint32_t(b[2])<<16|uint32_t(b[3])<<24; return true; }
    bool str(std::string & s, uint32_t n) { if (n>4096 || n>bytes.size()-at) return false; s.assign(reinterpret_cast<const char *>(bytes.data()+at),n); at+=n; return s.find('\0')==std::string::npos; }
};
std::string hex(const uint8_t * p, size_t n) { std::ostringstream out; out << std::hex << std::setfill('0'); for(size_t i=0;i<n;++i) out << std::setw(2) << int(p[i]); return out.str(); }
}
std::string map_sha256(const void * data, size_t size) {
    const auto * input = static_cast<const uint8_t *>(data);
    std::vector<uint8_t> bytes;
    if(size) bytes.assign(input,input+size);
    const uint64_t bits=uint64_t(size)*8; bytes.push_back(0x80);
    while(bytes.size()%64!=56) bytes.push_back(0);
    for(int i=7;i>=0;--i) bytes.push_back(uint8_t(bits>>(8*i)));
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for(size_t off=0;off<bytes.size();off+=64) {
        uint32_t w[64]; for(int i=0;i<16;++i) { const auto *p=bytes.data()+off+4*i; w[i]=uint32_t(p[0])<<24|uint32_t(p[1])<<16|uint32_t(p[2])<<8|p[3]; }
        for(int i=16;i<64;++i) w[i]=w[i-16]+(ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10));
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for(int i=0;i<64;++i) { uint32_t t1=v+(ror(e,6)^ror(e,11)^ror(e,25))+((e&f)^(~e&g))+constants[i]+w[i]; uint32_t t2=(ror(a,2)^ror(a,13)^ror(a,22))+((a&b)^(a&c)^(b&c)); v=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    }
    uint8_t digest[32]; for(int i=0;i<8;++i) for(int j=0;j<4;++j) digest[4*i+j]=uint8_t(h[i]>>(24-8*j));
    return hex(digest,32);
}
bool load_map_bundle(const std::string & path, const std::string & model_identity,
                     const geometry & shape, const std::vector<int> & global_layers,
                     const std::string & expected_source_sha256, map_bundle & output, std::string * error) {
    auto fail=[&](const char * msg){ if(error) *error=msg; return false; };
    const bool supported=(shape.q_heads==24 && shape.kv_heads==4 && shape.head_dim==256 && shape.rope_dim==64) || (shape.q_heads==16 && shape.kv_heads==2 && shape.head_dim==256 && shape.rope_dim==64);
    if(!supported || global_layers.size()>1024 || !shape.valid() || model_identity.empty() || global_layers.empty() || expected_source_sha256.size()!=64 ||
       expected_source_sha256.find_first_not_of("0123456789abcdef")!=std::string::npos) return fail("invalid expected map binding");
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in) return fail("cannot open map");
    auto size=in.tellg(); if(size<112 || size>128*1024*1024) return fail("map size outside bounds");
    std::vector<uint8_t> bytes(static_cast<size_t>(size)); in.seekg(0); if(!in.read(reinterpret_cast<char *>(bytes.data()),size)) return fail("cannot read map");
    reader r{bytes}; char magic[8]; uint32_t v,q,kv,d,rope,index,nlayers,mlen,ilen;
    uint8_t source[32],payload[32];
    if(!r.read(magic,8)||std::memcmp(magic,"LSAMAP1\0",8)||!r.u32(v)||!r.u32(q)||!r.u32(kv)||!r.u32(d)||!r.u32(rope)||!r.u32(index)||!r.u32(nlayers)||!r.u32(mlen)||!r.u32(ilen)||!r.read(source,32)||!r.read(payload,32)) return fail("malformed map header");
    if(v!=1 || q!=uint32_t(shape.q_heads)||kv!=uint32_t(shape.kv_heads)||d!=uint32_t(shape.head_dim)||rope!=uint32_t(shape.rope_dim)||index!=index_dimension||nlayers!=global_layers.size()) return fail("map geometry or coverage mismatch");
    const auto payload_hash=map_sha256(bytes.data()+r.at,bytes.size()-r.at);
    if(payload_hash!=hex(payload,32)||hex(source,32)!=expected_source_sha256) return fail("map source or payload hash mismatch");
    const uint64_t needed=uint64_t(mlen)+ilen+uint64_t(nlayers)*(4+uint64_t(shape.map_rows())*index_dimension*4);
    if(needed!=bytes.size()-r.at) return fail("map payload size mismatch");
    map_bundle next; next.shape=shape;next.source_sha256=hex(source,32);next.payload_sha256=payload_hash;
    std::string identity;
    if(!r.str(next.model_identity,mlen)||!r.str(identity,ilen)||next.model_identity!=model_identity||identity.empty()) return fail("map model identity mismatch");
    std::vector<int> actual;
    const size_t count=size_t(shape.map_rows())*index_dimension;
    for(uint32_t i=0;i<nlayers;++i) {
        uint32_t layer; if(!r.u32(layer)||layer>4095 || next.layers.count(int(layer))) return fail("duplicate or malformed layer");
        projection_maps map;map.shape=shape;map.identity=identity+":"+payload_hash+":"+std::to_string(layer);map.q.resize(count);map.k.resize(count);
        for(auto * buffer : {&map.q,&map.k}) for(auto & bf:*buffer) { uint8_t value[2]; if(!r.read(value,2)) return fail("truncated map buffer"); bf.bits=uint16_t(value[0])|uint16_t(value[1])<<8; }
        if(!map.valid()) return fail("invalid map buffer");
        actual.push_back(int(layer));next.layers.emplace(int(layer),std::move(map));
    }
    auto expected=global_layers;std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());
    if(actual!=expected||r.at!=bytes.size()) return fail("map coverage or trailing byte mismatch");
    output=std::move(next);return true;
}
}
