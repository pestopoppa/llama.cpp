from pathlib import Path
import os, subprocess, tempfile
p=Path(os.environ.get('R232_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
s=(p/'common/speculative.cpp').read_text()
start=s.index('        int32_t n_embd_tgt_expected = n_embd_dec;')
end=s.index('        if (target_layer_ids_n > ', start)
width=s[start:end]
t=(p/'tools/server/server-context.cpp').read_text()
start=t.index('                } catch (const std::exception & e) {',t.index('bool measure_model_bytes = has_draft;'))
end=t.index('\n                }',start)+len('\n                }')
catch=t[start:end]
cpp=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdarg>
#include <iostream>
std::string logline, metadata; int dspark=0;
void logit(const char *fmt, ...) { char buf[2048]; va_list args; va_start(args,fmt); vsnprintf(buf,sizeof buf,fmt,args); va_end(args); logline += buf; }
#define LOG_INF(...) logit(__VA_ARGS__)
#define SRV_WRN(...) logit(__VA_ARGS__)
int llama_model_meta_val_str(void*,const char*,char *out,size_t n) { if (metadata.empty()) return -1; snprintf(out,n,"%s",metadata.c_str()); return 0; }
int32_t llama_model_dspark_target_hidden_size(void*) { return dspark; }
void widths(int n_embd_dec,int n_embd_tgt) { void *model_dft=nullptr;
''' + width + r'''
}
void memory_error(bool has_draft) { try { throw std::runtime_error("fixture device failure");
''' + catch + r'''
}
int main() {
 logline.clear(); widths(5120,5120); assert(logline.find("legacy draft hidden size 5120 matches target")!=std::string::npos);
 logline.clear(); try { widths(5120,4096); assert(false); } catch (const std::runtime_error &e) { assert(std::string(e.what()).find("does not match")!=std::string::npos); } assert(logline.find("matches target")==std::string::npos);
 metadata="bogus"; logline.clear(); try { widths(5120,5120); assert(false); } catch (const std::runtime_error &e) { assert(std::string(e.what()).find("invalid")!=std::string::npos); } assert(logline.empty());
 metadata="5120"; logline.clear(); widths(4096,5120); assert(logline.find("legacy")==std::string::npos);
 metadata.clear(); dspark=4096; logline.clear(); widths(5120,4096); assert(logline.find("derived")!=std::string::npos); assert(logline.find("legacy")==std::string::npos);
 logline.clear(); memory_error(true); assert(logline.find("failed to measure draft model memory")!=std::string::npos); assert(logline.find("fit reservation may be incomplete")!=std::string::npos); assert(logline.find("fixture device failure")!=std::string::npos);
 logline.clear(); memory_error(false); assert(logline.find("MTP context")!=std::string::npos); assert(logline.find("fixture device failure")!=std::string::npos);
 std::cout << "PASS: legacy match, mismatch, invalid metadata, explicit metadata, DSpark, draft/MTP estimate errors\n";
}
'''
with tempfile.TemporaryDirectory(prefix='r232-controls-') as tmp:
    source=Path(tmp)/'controls.cpp'; binary=Path(tmp)/'controls'
    source.write_text(cpp)
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
