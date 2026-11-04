// Experimental correctness reference: a persistent CPU worker, serialized per process.
#pragma once
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char ** environ;

// Development-only recall state is keyed by the selector's output tensor. It is
// consumed by the immediately following physical-Q/K callback, never serialized.
struct lsa_recall_row { uint32_t row, position; };
static std::mutex lsa_recall_lock;
static std::map<const ggml_tensor *, lsa_recall_row> lsa_recall_rows;
static std::atomic<float> lsa_recall_scale{0.0f}, lsa_recall_softcap{0.0f};

static bool lsa_recall_enabled(float scale, float softcap) {
    const char * path = std::getenv("LLAMA_LSA_RECALL_OUT");
    if (!path) return false;
    const char * item = std::getenv("LLAMA_LSA_RECALL_ITEM");
    const char * split = std::getenv("LLAMA_LSA_RECALL_SPLIT");
    const char * nominal = std::getenv("LLAMA_LSA_RECALL_NOMINAL");
    const char * positions = std::getenv("LLAMA_LSA_RECALL_POSITIONS");
    const char * digest = std::getenv("LLAMA_LSA_RECALL_TOKEN_SHA256");
    const char * map_digest = std::getenv("LLAMA_LSA_RECALL_MAP_SHA256");
    const char * mode = std::getenv("LLAMA_LSA_SCREEN");
    if (!*path || !item || std::strcmp(item, "dev-code-00") || !split || std::strcmp(split, "dev") ||
        !nominal || std::strcmp(nominal, "32768") || !positions || std::strcmp(positions, "32767") ||
        !digest || std::strcmp(digest, "9a6948657870e895f48aabc0f3acc15fdf273ccf66f72345f0a482191dc6a480") ||
        !map_digest || std::strcmp(map_digest, "baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65") ||
        !mode || std::strcmp(mode, "pca") ||
        !std::isfinite(scale) || scale <= 0 || !std::isfinite(softcap) || softcap < 0) {
        GGML_ABORT("LSA recall: only frozen dev-code-00/32768 query 32767 is permitted");
    }
    lsa_recall_scale.store(scale);
    lsa_recall_softcap.store(softcap);
    return true;
}

static void lsa_recall_measure(ggml_tensor * dst, const ggml_tensor * selected, const ggml_tensor * q,
                               const ggml_tensor * k, int ith, int nth, void * user) {
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    if (ith != 0) return;
    GGML_ASSERT(selected->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32);
    GGML_ASSERT(q->ne[0] == 256 && q->ne[1] == 24 && k->ne[0] == 256 && k->ne[1] == 4);
    const uint32_t nq = q->ne[2], nk = k->ne[2], layer = reinterpret_cast<intptr_t>(user);
    GGML_ASSERT(selected->ne[0] == nk && selected->ne[1] >= nq);
    for (int64_t i = 0; i < selected->ne[1]; ++i) for (uint32_t j = 0; j < nk; ++j) {
        const float v = *reinterpret_cast<const float *>(static_cast<const char *>(selected->data) + i*selected->nb[1] + j*selected->nb[0]);
        *reinterpret_cast<float *>(static_cast<char *>(dst->data) + i*dst->nb[1] + j*dst->nb[0]) = v;
    }
    lsa_recall_row target{};
    {
        std::lock_guard<std::mutex> guard(lsa_recall_lock);
        auto it = lsa_recall_rows.find(selected);
        if (it == lsa_recall_rows.end()) return;
        target = it->second;
        lsa_recall_rows.erase(it);
    }
    GGML_ASSERT(target.position == 32767 && target.row < nq && nk >= 32768);
    const uint32_t end = target.position + 1;
    const float scale = lsa_recall_scale.load(), softcap = lsa_recall_softcap.load();
    std::vector<double> mass(end, 0.0);
    for (uint32_t h = 0; h < 24; ++h) {
        std::vector<double> logits(end);
        double maximum = -INFINITY;
        for (uint32_t j = 0; j < end; ++j) {
            double dot = 0.0;
            const char * qp = static_cast<const char *>(q->data) + target.row*q->nb[2] + h*q->nb[1];
            const char * kp = static_cast<const char *>(k->data) + j*k->nb[2] + (h/6)*k->nb[1];
            for (uint32_t d = 0; d < 256; ++d) {
                dot += double(*reinterpret_cast<const float *>(qp + d*q->nb[0])) *
                       double(*reinterpret_cast<const float *>(kp + d*k->nb[0]));
            }
            double logit = dot * scale;
            if (softcap > 0) logit = softcap * std::tanh(logit / softcap);
            if (!std::isfinite(logit)) GGML_ABORT("LSA recall: nonfinite dense logit");
            logits[j] = logit;
            maximum = std::max(maximum, logit);
        }
        double denominator = 0.0;
        for (double & v : logits) { v = std::exp(v - maximum); denominator += v; }
        for (uint32_t j = 0; j < end; ++j) mass[j] += logits[j] / denominator / 24.0;
    }
    std::vector<uint32_t> ranked(end);
    for (uint32_t j = 0; j < end; ++j) ranked[j] = j;
    std::stable_sort(ranked.begin(), ranked.end(), [&mass](uint32_t a, uint32_t b) { return mass[a] > mass[b]; });
    double selected_mass = 0.0;
    uint32_t selected_count = 0, top_hits = 0;
    for (uint32_t j = 0; j < nk; ++j) {
        const float v = *reinterpret_cast<const float *>(static_cast<const char *>(selected->data) + target.row*selected->nb[1] + j*selected->nb[0]);
        if (j >= end && std::isfinite(v)) GGML_ABORT("LSA recall: selected future key");
        if (j < end && std::isfinite(v)) {
            if (v != 0.0f) GGML_ABORT("LSA recall: selected mask changed causal value");
            selected_mass += mass[j]; ++selected_count;
        }
    }
    if (selected_count == 0 || selected_count > 4096 + 512 + 4) GGML_ABORT("LSA recall: invalid selected budget");
    for (uint32_t t = 0; t < 4096; ++t) {
        const uint32_t j = ranked[t];
        const float v = *reinterpret_cast<const float *>(static_cast<const char *>(selected->data) + target.row*selected->nb[1] + j*selected->nb[0]);
        top_hits += std::isfinite(v);
    }
    static std::mutex output_lock;
    std::lock_guard<std::mutex> guard(output_lock);
    static std::set<uint32_t> emitted_layers;
    if (!emitted_layers.insert(layer).second) GGML_ABORT("LSA recall: duplicate layer aggregate");
    std::ofstream out(std::getenv("LLAMA_LSA_RECALL_OUT"), std::ios::app);
    out << std::setprecision(17) << "{\"item\":\"dev-code-00\",\"split\":\"dev\",\"nominal\":32768,\"token_sha256\":\"9a6948657870e895f48aabc0f3acc15fdf273ccf66f72345f0a482191dc6a480\",\"query\":32767,\"layer\":" << layer
        << ",\"mode\":\"pca\",\"released_pt_map_sha256\":\"baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65\",\"budget\":" << selected_count
        << ",\"top_k\":4096,\"top_hits\":" << top_hits << ",\"top_recall\":" << (double(top_hits) / 4096.0)
        << ",\"selected_mass\":" << selected_mass
        << ",\"scale\":" << scale << ",\"softcap\":" << softcap
        << ",\"method\":\"reconstructed_dense_dequantized_k_double_dot\"}\n";
    if (!out) GGML_ABORT("LSA recall: aggregate output failed");
}

class lsa_screen_worker {
public:
    lsa_screen_worker() {
        const char * mode = std::getenv("LLAMA_LSA_SCREEN");
        const char * worker = std::getenv("LLAMA_LSA_WORKER");
        const char * python = std::getenv("LLAMA_LSA_PYTHON");
        const char * map = std::getenv("LLAMA_LSA_MAP");
        if (!mode || !worker || !python || (std::strcmp(mode, "all") && std::strcmp(mode, "positional") &&
                                         std::strcmp(mode, "oracle_topk") && !map))
            GGML_ABORT("LSA screen: missing configuration");
        int pair[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) GGML_ABORT("LSA screen: socketpair failed");
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addclose(&actions, pair[0]);
        posix_spawn_file_actions_adddup2(&actions, pair[1], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, pair[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, pair[1]);
        const char * args[] = {python, worker, "--serve", mode, map ? map : "-", nullptr};
        int err = posix_spawn(&pid, python, &actions, nullptr, const_cast<char * const *>(args), environ);
        posix_spawn_file_actions_destroy(&actions);
        close(pair[1]);
        if (err) GGML_ABORT("LSA screen: spawn failed: %s", std::strerror(err));
        fd = pair[0];
    }
    ~lsa_screen_worker() {
        if (fd >= 0) close(fd);
        if (pid > 0) { int status; while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {} }
    }
    void exchange(const std::vector<char> & input, std::vector<float> & output) {
        std::lock_guard<std::mutex> guard(lock);
        size_t offset = 0;
        while (offset < input.size()) {
            ssize_t n = send(fd, input.data() + offset, input.size() - offset, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) GGML_ABORT("LSA screen: worker write failed");
            offset += n;
        }
        int32_t status = -1;
        receive(&status, sizeof(status));
        if (status) GGML_ABORT("LSA screen: worker rejected input");
        receive(output.data(), output.size() * sizeof(float));
    }
private:
    void receive(void * data, size_t size) {
        size_t offset = 0;
        while (offset < size) {
            ssize_t n = recv(fd, static_cast<char *>(data) + offset, size - offset, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) GGML_ABORT("LSA screen: worker read failed");
            offset += n;
        }
    }
    int fd = -1;
    pid_t pid = -1;
    std::mutex lock;
};

static void lsa_screen_mask(ggml_tensor * dst, const ggml_tensor * mask, const ggml_tensor * q,
                            const ggml_tensor * k, int ith, int nth, void * user) {
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    if (ith != 0) return;
    GGML_ASSERT(mask->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32);
    const uint32_t nq = q->ne[2], nk = k->ne[2], layer = reinterpret_cast<intptr_t>(user);
    GGML_ASSERT(mask->ne[0] == nk && mask->ne[1] >= nq && mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(q->ne[0] == 256 && q->ne[1] == 24 && q->ne[3] == 1 && k->ne[0] == 256 && k->ne[1] == 4 && k->ne[3] == 1);
    std::vector<char> input;
    input.reserve(12 + 4*(size_t(nq)*24*256 + size_t(nk)*4*256 + size_t(nq)*nk));
    auto append = [&input](const void * p, size_t n) { const char * c = static_cast<const char *>(p); input.insert(input.end(), c, c+n); };
    append(&nq, 4); append(&nk, 4); append(&layer, 4);
    auto write3 = [&append](const ggml_tensor * t, uint32_t n, uint32_t heads) {
        for (uint32_t i = 0; i < n; ++i) for (uint32_t h = 0; h < heads; ++h) {
            const char * p = static_cast<const char *>(t->data) + i*t->nb[2] + h*t->nb[1];
            for (uint32_t d = 0; d < 256; ++d) append(p + d*t->nb[0], sizeof(float));
        }
    };
    write3(q, nq, 24); write3(k, nk, 4);
    for (uint32_t i = 0; i < nq; ++i) for (uint32_t j = 0; j < nk; ++j)
        append(static_cast<const char *>(mask->data) + i*mask->nb[1] + j*mask->nb[0], sizeof(float));
    std::vector<float> result(size_t(nq) * nk);
    static lsa_screen_worker worker;
    worker.exchange(input, result);
    if (std::getenv("LLAMA_LSA_RECALL_OUT")) {
        uint32_t matches = 0;
        for (uint32_t i = 0; i < nq; ++i) {
            uint32_t allowed = 0;
            for (uint32_t j = 0; j < nk; ++j) {
                const float prior = *reinterpret_cast<const float *>(static_cast<const char *>(mask->data) + i*mask->nb[1] + j*mask->nb[0]);
                allowed += std::isfinite(prior);
            }
            if (allowed == 32768) {
                if (++matches != 1) GGML_ABORT("LSA recall: duplicate target query");
                std::lock_guard<std::mutex> guard(lsa_recall_lock);
                if (lsa_recall_rows.size() >= 16) GGML_ABORT("LSA recall: too many pending rows");
                lsa_recall_rows[dst] = {i, 32767};
            }
        }
    }
    uint64_t before = 0, after = 0, changed = 0;
    for (uint32_t i = 0; i < nq; ++i) for (uint32_t j = 0; j < nk; ++j) {
        const float prior = *reinterpret_cast<const float *>(static_cast<const char *>(mask->data) + i*mask->nb[1] + j*mask->nb[0]);
        const float next = result[size_t(i)*nk+j];
        before += std::isfinite(prior); after += std::isfinite(next); changed += prior != next;
    }
    if (const char * trace = std::getenv("LLAMA_LSA_TRACE")) {
        static std::mutex trace_lock;
        std::lock_guard<std::mutex> guard(trace_lock);
        std::ofstream log(trace, std::ios::app);
        log << "{\"layer\":" << layer << ",\"queries\":" << nq << ",\"slots\":" << nk
            << ",\"allowed_before\":" << before << ",\"allowed_after\":" << after
            << ",\"changed\":" << changed << "}\n";
        if (!log) GGML_ABORT("LSA screen: cannot write callback trace");
    }
    for (int64_t i = 0; i < mask->ne[1]; ++i) for (uint32_t j = 0; j < nk; ++j) {
        const float value = i < nq ? result[size_t(i)*nk+j] : *reinterpret_cast<const float *>(static_cast<const char *>(mask->data) + i*mask->nb[1] + j*mask->nb[0]);
        *reinterpret_cast<float *>(static_cast<char *>(dst->data) + i*dst->nb[1] + j*dst->nb[0]) = value;
    }
}
