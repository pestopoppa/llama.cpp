#include "llama-profile.h"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

bool llama_profile_enabled() {
    static const bool enabled = [] { const char * p = std::getenv("LLAMA_NATIVE_PROFILE"); return p && *p; }();
    return enabled;
}
namespace {
[[noreturn]] void fail(const char * reason) {
    std::fprintf(stderr, "native profile capture failed: %s (%s)\n", reason, std::strerror(errno));
    std::abort();
}
uint64_t now(clockid_t clock) {
    timespec t{};
    if (clock_gettime(clock, &t)) fail("clock_gettime");
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
struct capture {
    int fd;
    uint64_t birth;
    int (*push)(const char *);
    int (*pop)();
    std::mutex mutex;
    std::atomic<uint64_t> next{1};
    capture() {
        std::ifstream stat("/proc/self/stat");
        std::string line;
        if (!std::getline(stat, line)) fail("read process birth");
        const auto end = line.rfind(')');
        if (end == std::string::npos) fail("parse process birth");
        std::istringstream fields(line.substr(end + 2));
        std::string value;
        for (int field = 3; field <= 22; ++field) if (!(fields >> value)) fail("missing process birth");
        try { birth = std::stoull(value); } catch (...) { fail("invalid process birth"); }
        const char * lib = std::getenv("LLAMA_NATIVE_ROCTX_LIBRARY");
        if (!lib || !*lib) fail("missing exact ROCTX library path");
        void * handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
        if (!handle) fail(dlerror());
        push = reinterpret_cast<int (*)(const char *)>(dlsym(handle, "roctxRangePushA"));
        pop = reinterpret_cast<int (*)()>(dlsym(handle, "roctxRangePop"));
        if (!push || !pop) fail("missing ROCTX entrypoints");
        // Exclusive create prevents mixing process incarnations or overwriting evidence.
        fd = open(std::getenv("LLAMA_NATIVE_PROFILE"), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) fail("exclusive output create");
    }
};
capture & state() { static capture c; return c; }
thread_local std::string slots = "[]";
void emit(const std::string & fields) {
    auto & c = state();
    std::ostringstream out;
    out << "{\"schema\":\"epyc.native_profile.v1\",\"pid\":" << getpid()
        << ",\"process_start_ticks\":" << c.birth
        << ",\"tid\":" << syscall(SYS_gettid)
        << ",\"monotonic_ns\":" << now(CLOCK_MONOTONIC)
        << ",\"monotonic_raw_ns\":" << now(CLOCK_MONOTONIC_RAW) << ',' << fields << "}\n";
    const std::string line = out.str();
    std::lock_guard<std::mutex> guard(c.mutex);
    size_t off = 0;
    while (off < line.size()) {
        ssize_t n = write(c.fd, line.data() + off, line.size() - off);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) fail("output write");
        off += size_t(n);
    }
}
}
void llama_profile_slots(const char * json) {
    if (llama_profile_enabled()) slots = json;
}
void llama_profile_record(const char * json_fields) {
    if (llama_profile_enabled()) emit(json_fields);
}
uint64_t llama_profile_decode_begin(const llama_context * ctx, const llama_batch & batch) {
    if (!llama_profile_enabled()) return 0;
    auto & c = state();
    const uint64_t call = c.next++;
    std::string label = "epyc.decode." + std::to_string(call);
    if (c.push(label.c_str()) < 0) fail("ROCTX push");
    std::ostringstream out;
    out << "\"event\":\"decode_begin\",\"call_id\":" << call
        << ",\"ctx\":\"" << ctx << "\",\"model\":\"" << llama_get_model(ctx)
        << "\",\"slots\":" << slots << ",\"n_tokens\":" << batch.n_tokens << ",\"rows\":[";
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (i) out << ',';
        out << "{\"index\":" << i << ",\"pos\":";
        if (batch.pos) out << batch.pos[i]; else out << "null";
        out << ",\"token\":";
        if (batch.token) out << batch.token[i]; else out << "null";
        out << ",\"seq_ids\":";
        if (!batch.n_seq_id || !batch.seq_id || !batch.seq_id[i]) out << "null";
        else {
            out << '[';
            for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) { if (j) out << ','; out << batch.seq_id[i][j]; }
            out << ']';
        }
        out << '}';
    }
    out << ']';
    emit(out.str());
    return call;
}
void llama_profile_decode_end(uint64_t call, int32_t result) {
    if (!call) return;
    emit("\"event\":\"decode_end\",\"call_id\":" + std::to_string(call) + ",\"rc\":" + std::to_string(result));
    if (state().pop() < 0) fail("ROCTX pop");
}
