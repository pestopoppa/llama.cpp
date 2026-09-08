#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

struct ctx_del { void operator()(llama_context * p) const { llama_free(p); } };
using ctx_ptr = std::unique_ptr<llama_context, ctx_del>;

struct logits_copy { std::vector<float> v; };

static double env_real(const char * name, double fallback) {
    const char * s=std::getenv(name); if (!s || !*s) return fallback;
    char * e=nullptr; errno=0; const double v=std::strtod(s,&e);
    if (errno || !e || *e || !std::isfinite(v) || v<0) { std::fprintf(stderr,"invalid %s\n",name); std::exit(2); }
    return v;
}
static int env_int(const char * name, int fallback, int n_vocab) {
    const char * s=std::getenv(name); if (!s || !*s) return fallback;
    char * e=nullptr; errno=0; const long v=std::strtol(s,&e,10);
    if (errno || !e || *e || v<0 || v>=n_vocab) { std::fprintf(stderr,"invalid %s\n",name); std::exit(2); }
    return (int)v;
}
static ctx_ptr make_ctx(const common_params & p, llama_model * m) {
    auto cp=common_context_params_to_llama(p); cp.n_seq_max=1; cp.n_rs_seq=8;
    cp.n_batch=std::max<uint32_t>(cp.n_batch,32); cp.n_ubatch=std::max<uint32_t>(cp.n_ubatch,8);
    return ctx_ptr(llama_init_from_model(m,cp));
}
static std::vector<logits_copy> decode(llama_context * c, const std::vector<llama_token> & ids,
                                       llama_pos pos, int n_vocab, bool all_logits) {
    llama_batch b=llama_batch_init((int)ids.size(),0,1);
    for (size_t i=0;i<ids.size();++i) common_batch_add(b,ids[i],pos+(llama_pos)i,{0},all_logits || i+1==ids.size());
    if (llama_decode(c,b)!=0) { llama_batch_free(b); throw std::runtime_error("decode failed"); }
    std::vector<logits_copy> out;
    const size_t nout=all_logits?ids.size():1;
    for (size_t i=0;i<nout;++i) {
        const float * x=llama_get_logits_ith(c,all_logits?(int)i:-1);
        if (!x) { llama_batch_free(b); throw std::runtime_error("missing logits row"); }
        out.push_back({std::vector<float>(x,x+n_vocab)});
    }
    llama_batch_free(b); return out;
}
static void decode_prompt_split(llama_context * c, const std::vector<llama_token> & ids, int n_vocab) {
    if (ids.size() != 26) throw std::runtime_error("split diagnostic requires 26 prompt tokens");
    decode(c, std::vector<llama_token>(ids.begin(), ids.begin() + 22), 0, n_vocab, false);
    decode(c, std::vector<llama_token>(ids.begin() + 22, ids.end()), 22, n_vocab, false);
}
static std::vector<float> decode_sequential(llama_context * c, const std::vector<llama_token> & ids,
                                            llama_pos pos, int n_vocab) {
    std::vector<float> last;
    for (size_t i = 0; i < ids.size(); ++i) {
        last = decode(c, {ids[i]}, pos + (llama_pos) i, n_vocab, false).back().v;
    }
    return last;
}
static double diff(const std::vector<float> & a,const std::vector<float> & b) {
    if (a.size()!=b.size() || a.empty()) throw std::runtime_error("logit shape mismatch");
    double d=0; for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i])||!std::isfinite(b[i])) throw std::runtime_error("nonfinite logits");
        d=std::max(d,std::fabs((double)a[i]-b[i]));
    } return d;
}
static int argmax(const std::vector<float> & x) { return (int)(std::max_element(x.begin(),x.end())-x.begin()); }
static double margin(const std::vector<float> & x) {
    float a=-INFINITY,b=-INFINITY; for(float v:x) { if(v>a){b=a;a=v;} else if(v>b)b=v; } return (double)a-b;
}
static void emit_cmp(std::ostream & o,const char * name,const std::vector<float> & x,
                     const std::vector<float> & ref,double tol,bool & pass) {
    const double d=diff(x,ref); const int ax=argmax(x), ar=argmax(ref); const bool ok=d<=tol && ax==ar; pass &= ok;
    o << "\"" << name << "\":{\"max_abs_diff\":" << d << ",\"argmax\":" << ax
      << ",\"reference_argmax\":" << ar << ",\"argmax_equal\":" << (ax==ar?"true":"false")
      << ",\"argmax_margin\":" << margin(x) << ",\"pass\":" << (ok?"true":"false") << "}";
}
int main(int argc,char ** argv) {
    common_params p;
    p.prompt="[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Explain how a CPU cache hierarchy works, with a concrete example.<|assistant|><think>";
    common_init(); if(!common_params_parse(argc,argv,p,LLAMA_EXAMPLE_COMMON)) return 1;
    ggml_backend_load_all(); auto init=common_init_from_params(p); llama_model * m=init?init->model():nullptr; if(!m)return 1;
    char arch[64]={}; llama_model_meta_val_str(m,"general.architecture",arch,sizeof(arch));
    if(std::string(arch)!="glm5next"&&std::string(arch)!="glm5-next") { std::fprintf(stderr,"REFUSE: not GLM5Next\n"); return 2; }
    const int nv=llama_vocab_n_tokens(llama_model_get_vocab(m)); if(nv<=0)return 2;
    const double tol=env_real("GLM53_DIAG_LOGIT_TOLERANCE",1e-5);
    const int reject1=env_int("GLM53_DIAG_REJECT1",459,nv), reject2=env_int("GLM53_DIAG_REJECT2",1247,nv);
    const char * path=std::getenv("GLM53_DIAG_JSON"); if(!path||!*path){std::fprintf(stderr,"GLM53_DIAG_JSON required\n");return 2;}
    try {
        auto prompt=common_tokenize(init->context(),p.prompt,true,true);
        if(prompt.size()!=26) throw std::runtime_error("rendered prompt did not tokenize to observed 26 tokens");
        const std::vector<llama_token> accepted={785,1196,6801,458,16134,315};
        const std::vector<llama_token> b1={785,1196,6801,(llama_token)reject1};
        const std::vector<llama_token> b2={458,16134,315,(llama_token)reject2};
        const std::vector<llama_token> n1(accepted.begin(),accepted.begin()+3), n2(accepted.begin()+3,accepted.end());
        auto serial=make_ctx(p,m), batched=make_ctx(p,m), toggle=make_ctx(p,m), rollback=make_ctx(p,m);
        llama_set_rowexact(serial.get(), false); llama_set_rowexact(batched.get(), false);
        llama_set_rowexact(toggle.get(), false); llama_set_rowexact(rollback.get(), false);
        decode_prompt_split(serial.get(),prompt,nv); decode_prompt_split(batched.get(),prompt,nv);
        decode_prompt_split(toggle.get(),prompt,nv); decode_prompt_split(rollback.get(),prompt,nv);
        const int first=argmax(decode_sequential(serial.get(),{accepted[0]},(llama_pos)prompt.size(),nv));
        if(first!=1196) throw std::runtime_error("exact-prefix precondition failed after token 785");
        auto s1=decode_sequential(serial.get(),{accepted[1],accepted[2]},(llama_pos)prompt.size()+1,nv);
        if(argmax(s1)!=458) throw std::runtime_error("exact-prefix precondition failed after token 6801");
        auto s2=decode_sequential(serial.get(),n2,(llama_pos)prompt.size()+3,nv);
        if(argmax(s2)!=1246) throw std::runtime_error("exact-prefix precondition failed after token 315");
        llama_set_rowexact(batched.get(), true);
        auto nb1=decode(batched.get(),n1,(llama_pos)prompt.size(),nv,false).back().v;
        auto nb2=decode(batched.get(),n2,(llama_pos)prompt.size()+3,nv,false).back().v;
        llama_set_rowexact(batched.get(), false);
        llama_set_rowexact(toggle.get(), true);
        auto toggle1=decode(toggle.get(),n1,(llama_pos)prompt.size(),nv,false).back().v;
        llama_set_rowexact(toggle.get(), false);
        auto toggle2=decode_sequential(toggle.get(),n2,(llama_pos)prompt.size()+3,nv);
        llama_set_rowexact(rollback.get(), true);
        auto rb1rows=decode(rollback.get(),b1,(llama_pos)prompt.size(),nv,true);
        const auto rb1=rb1rows[2].v; // logits after accepted token 6801, before rejected input row
        if(!llama_memory_seq_rm(llama_get_memory(rollback.get()),0,(llama_pos)prompt.size()+3,-1))
            throw std::runtime_error("first rollback refused");
        auto rb2rows=decode(rollback.get(),b2,(llama_pos)prompt.size()+3,nv,true);
        const auto rb2=rb2rows[2].v; // exact server divergence row: logits after token 315
        llama_set_rowexact(rollback.get(), false);
        if(!llama_memory_seq_rm(llama_get_memory(rollback.get()),0,(llama_pos)prompt.size()+6,-1))
            throw std::runtime_error("second rollback refused");
        bool pass=true; std::ofstream o(path); if(!o)throw std::runtime_error("cannot open output"); o<<std::setprecision(17);
        o << "{\"schema\":\"epyc.glm53.rowexact_phase_gate.v1\",\"prompt_tokens\":" << prompt.size() << ",\"prefill_chunks\":[22,4]"
          << ",\"accepted_ids\":[785,1196,6801,458,16134,315],\"synthetic_rejected_ids\":[" << reject1 << ',' << reject2 << ']'
          << ",\"observed_plain_next\":1246,\"observed_mtp_next\":13931,\"declared_logit_tolerance\":" << tol << ",\"comparisons\":{";
        emit_cmp(o,"batch1_no_reject_vs_serial",nb1,s1,tol,pass); o<<',';
        emit_cmp(o,"batch1_reject_row_vs_serial",rb1,s1,tol,pass); o<<',';
        emit_cmp(o,"batch2_no_reject_vs_serial",nb2,s2,tol,pass); o<<',';
        emit_cmp(o,"batch1_exact_then_off_vs_serial",toggle1,s1,tol,pass); o<<',';
        emit_cmp(o,"off_after_exact_reuse_vs_serial",toggle2,s2,tol,pass); o<<',';
        emit_cmp(o,"batch2_after_prior_rollback_vs_serial",rb2,s2,tol,pass);
        o << "},\"diagnostic_scope\":\"target-only phase-scoped rowexact; prompt and sequential replay disabled, verification batches enabled; includes off-on-off reused context\",\"verdict\":\"" << (pass?"PASS":"FAIL") << "\"}\n";
        return pass?0:3;
    } catch(const std::exception & e) { std::fprintf(stderr,"REFUSE: %s\n",e.what()); return 2; }
}
