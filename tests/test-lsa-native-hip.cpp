#include "llama-lsa-native-hip.h"
#include "ggml-quants.h"
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// Never initialize HIP from a normal CPU-only ctest run. Execution needs BOTH
// an explicit argument and the operator's externally held GPU/CPU claims.
// CTest should register the default invocation with SKIP_RETURN_CODE 77.
namespace {
using namespace llama_lsa;

void need(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
void checked(hipError_t rc) { need(rc == hipSuccess, hipGetErrorString(rc)); }
void success(hip_result rc, const std::string & error) { need(rc == hip_result::success, error.c_str()); }

void timestamp(const char * phase) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (const char * path = std::getenv("LSA_HIP_TIMELINE")) {
        std::ofstream file(path, std::ios::app);
        file << phase << ' ' << ns << '\n';
        need(bool(file), "write kernel timeline");
    }
}

template<class T> struct device_array {
    T * data = nullptr;
    size_t count;
    explicit device_array(size_t n) : count(n) {
        if (n) { checked(hipMalloc(reinterpret_cast<void **>(&data), n * sizeof(T))); }
    }
    explicit device_array(const std::vector<T> & v) : device_array(v.size()) { put(v); }
    ~device_array() { (void) hipFree(data); }
    device_array(const device_array &) = delete;
    device_array & operator=(const device_array &) = delete;
    void put(const std::vector<T> & v) {
        need(v.size() == count, "device fixture size");
        if (count) { checked(hipMemcpy(data, v.data(), count * sizeof(T), hipMemcpyHostToDevice)); }
    }
    std::vector<T> get() const {
        std::vector<T> out(count);
        if (count) { checked(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost)); }
        return out;
    }
};

tensor_view view(const void * p, ggml_type type, int64_t rows, int heads, int dim) {
    const size_t bytes = ggml_row_size(type, dim);
    return { p, type, rows, heads, dim, bytes * heads, bytes };
}

std::vector<float> samples(size_t n, uint32_t state) {
    std::vector<float> out(n);
    for (float & x : out) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        x = (int32_t(state % 65537) - 32768) / 32768.0f;
    }
    return out;
}

projection_maps maps_for(int qheads, int kvheads, bool tied = false) {
    projection_maps maps;
    maps.shape = { qheads, kvheads, 256, 64 };
    maps.identity = tied ? "hip-tied-fixture" : "hip-seeded-fixture-" + std::to_string(qheads);
    const auto q = samples(size_t(maps.shape.map_rows()) * index_dimension, 771);
    const auto k = samples(q.size(), 992);
    for (size_t i = 0; i < q.size(); ++i) {
        maps.q.push_back(ggml_fp32_to_bf16(tied ? (i % 64 == 0 ? 1.0f : 0.0f) : q[i]));
        maps.k.push_back(ggml_fp32_to_bf16(tied ? (i % 64 == 0 ? 1.0f : 0.0f) : k[i]));
    }
    return maps;
}

void close(const std::vector<float> & actual, const std::vector<float> & expected, float tolerance, const char * what) {
    need(actual.size() == expected.size(), what);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance * (1.0f + std::fabs(expected[i]))) {
            std::fprintf(stderr, "%s mismatch[%zu]: %.9g != %.9g (tol %.3g)\n", what, i, actual[i], expected[i], tolerance);
            std::exit(1);
        }
    }
}

std::vector<uint8_t> quantize(const std::vector<float> & source, int heads, int64_t rows) {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, 256);
    std::vector<uint8_t> out(size_t(rows) * heads * row_bytes);
    for (int64_t row = 0; row < rows * heads; ++row) {
        quantize_row_q8_0_ref(source.data() + row * 256, reinterpret_cast<block_q8_0 *>(out.data() + row * row_bytes), 256);
    }
    return out;
}

void witness_gate() {
    static bool done = false;
    if (done) { return; }
    done = true;
    const char * ready = std::getenv("LSA_HIP_WITNESS_READY");
    const char * release = std::getenv("LSA_HIP_WITNESS_RELEASE");
    if (!ready && !release) { return; }
    need(ready && release, "both witness gate paths supplied");
    { std::ofstream f(ready); f << getpid() << '\n'; need(bool(f), "write witness ready"); }
    // Bound the observer handshake; no additional GPU work or synthetic repeat.
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!std::ifstream(release).good()) {
        need(std::chrono::steady_clock::now() < until, "witness release arrived within10seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void exact_attention(hip_context * context, const projection_maps & maps, ggml_type kvtype) {
    constexpr int64_t rows = 1537, nq = 3, main_capacity = 9, extra_capacity = 3;
    const int qh = maps.shape.q_heads, kh = maps.shape.kv_heads;
    const auto q = samples(nq * qh * 256, 997);
    const auto k = samples(rows * kh * 256, 298);
    const auto v = samples(k.size(), 299);
    auto main = std::vector<int32_t>{0,-1,2,2,3, -999,-999,-999,-999,
                                   0,9,9,100,200,300,-1,1000,-999,
                                   1000,2,1536,1500,1200,200,600,-1,0};
    auto extra = std::vector<int32_t>{3,0,-999, 9,-1,999, 1536,0,0};
    auto lengths = std::vector<int32_t>{5,8,9};
    const auto extra_lengths = std::vector<int32_t>{2,3,3};
    const auto positions = std::vector<int32_t>{3,1000,1536};
    auto sink = samples(qh, 552);
    sink[0] = -INFINITY;
    device_array<float> dq(q), dk(k), dv(v), dsink(sink), output(nq * qh * 256);
    device_array<int32_t> dmain(main), dextra(extra), dlen(lengths), dextralen(extra_lengths), dpos(positions);
    const auto kq8 = quantize(k, kh, rows), vq8 = quantize(v, kh, rows);
    device_array<uint8_t> dkq8(kq8), dvq8(vq8);
    std::vector<ggml_bf16_t> kb(k.size()), vb(v.size());
    for (size_t i = 0; i < k.size(); ++i) { kb[i] = ggml_fp32_to_bf16(k[i]); vb[i] = ggml_fp32_to_bf16(v[i]); }
    device_array<ggml_bf16_t> dkb(kb), dvb(vb);
    const void * hk = kvtype == GGML_TYPE_Q8_0 ? static_cast<const void *>(kq8.data()) :
                     kvtype == GGML_TYPE_BF16 ? static_cast<const void *>(kb.data()) : k.data();
    const void * hv = kvtype == GGML_TYPE_Q8_0 ? static_cast<const void *>(vq8.data()) :
                     kvtype == GGML_TYPE_BF16 ? static_cast<const void *>(vb.data()) : v.data();
    const void * gk = kvtype == GGML_TYPE_Q8_0 ? static_cast<const void *>(dkq8.data) :
                     kvtype == GGML_TYPE_BF16 ? static_cast<const void *>(dkb.data) : dk.data;
    const void * gv = kvtype == GGML_TYPE_Q8_0 ? static_cast<const void *>(dvq8.data) :
                     kvtype == GGML_TYPE_BF16 ? static_cast<const void *>(dvb.data) : dv.data;
    // The exact physical oracle dequantizes K/V and retains the F32 query.
    // Native CPU dense arithmetic separately converts Q to K's vec_dot_type;
    // that backend approximation is deliberately not a GPU bit-parity oracle.
    std::vector<float> oracle_k(k.size()), oracle_v(v.size());
    for (int64_t row = 0; row < rows * kh; ++row) {
        if (kvtype == GGML_TYPE_Q8_0) {
            const size_t stride = ggml_row_size(kvtype, 256);
            dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(static_cast<const uint8_t *>(hk) + row * stride), oracle_k.data() + row * 256, 256);
            dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(static_cast<const uint8_t *>(hv) + row * stride), oracle_v.data() + row * 256, 256);
        } else {
            for (int d = 0; d < 256; ++d) {
                oracle_k[row * 256 + d] = kvtype == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(kb[row * 256 + d]) : k[row * 256 + d];
                oracle_v[row * 256 + d] = kvtype == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(vb[row * 256 + d]) : v[row * 256 + d];
            }
        }
    }
    std::vector<float> expected(nq * qh * 256);
    attention_args cpu;
    cpu.q = view(q.data(), GGML_TYPE_F32, nq, qh, 256);
    cpu.k = view(oracle_k.data(), GGML_TYPE_F32, rows, kh, 256); cpu.v = view(oracle_v.data(), GGML_TYPE_F32, rows, kh, 256);
    cpu.main = { main.data(), lengths.data(), main_capacity };
    cpu.extra = { extra.data(), extra_lengths.data(), extra_capacity };
    cpu.positions = positions.data(); cpu.attn_sink = sink.data(); cpu.scale = 0.0625f; cpu.softcap = 3.0f;
    cpu.output = expected.data();
    attention_stats cs, gs;
    std::string error;
    need(selected_attention_cpu(cpu, &cs, &error), error.c_str());
    attention_args gpu = cpu;
    gpu.q.data = dq.data; gpu.k = view(gk, kvtype, rows, kh, 256); gpu.v = view(gv, kvtype, rows, kh, 256);
    gpu.main = { dmain.data, dlen.data, main_capacity };
    gpu.extra = { dextra.data, dextralen.data, extra_capacity };
    gpu.positions = dpos.data; gpu.attn_sink = dsink.data; gpu.output = output.data;
    success(hip_selected_attention(context, gpu, &gs, &error), error);
    close(output.get(), expected, 3e-5f, "selected exact attention");
    need(gs.key_rows == cs.key_rows && gs.value_rows == cs.value_rows, "selected row visits equal CPU");
    need(gs.key_rows < uint64_t(nq * qh * rows / 50), "selected attention avoids dense KV visits");

    // Invalid live slots and lengths must fail before output or KV access. An
    // invalid *non-live* slot above is deliberately accepted by the success case.
    const std::vector<float> sentinel(expected.size(), 97.0f);
    output.put(sentinel);
    main[0] = 4; dmain.put(main);
    need(hip_selected_attention(context, gpu, nullptr, &error) == hip_result::invalid_argument, "future row rejected");
    need(output.get() == sentinel, "invalid causal input leaves output untouched");
    main[0] = -2; dmain.put(main);
    need(hip_selected_attention(context, gpu, nullptr, &error) == hip_result::invalid_argument, "index below -1 rejected");
    main[0] = 0; dmain.put(main); lengths[0] = 10; dlen.put(lengths);
    need(hip_selected_attention(context, gpu, nullptr, &error) == hip_result::invalid_argument, "length over capacity rejected");
    lengths.assign(3, 0); dlen.put(lengths);
    gpu.extra = {}; // no valid slots; denominator sink alone still returns zeros
    success(hip_selected_attention(context, gpu, &gs, &error), error);
    need(output.get() == std::vector<float>(expected.size(), 0.0f), "sink-only output zero");
    need(gs.key_rows == 0 && gs.value_rows == 0, "sink-only visits no KV");
}

void geometry_fixture(int qh, int kh) {
    constexpr int64_t rows = 1537, nq = 3;
    const auto maps = maps_for(qh, kh);
    auto q = samples(nq * qh * 256, 71), k = samples(rows * kh * 256, 91);
    device_array<float> dq(q), dk(k), projected(nq * index_dimension);
    hip_context * context = nullptr;
    std::string error;
    success(hip_create(maps, {rows, nq, 1024}, nullptr, &context, &error), error);
    witness_gate();
    success(hip_reset(context, 55, maps.identity, &error), error);
    need(hip_reset(context, 55, "other-map", &error) == hip_result::invalid_argument, "map identity fail closed");
    timestamp("projection_launch");
    success(hip_project_queries(context, view(dq.data, GGML_TYPE_F32, nq, qh, 256), projected.data, &error), error);
    timestamp("projection_finish");
    std::vector<float> expected_q(nq * index_dimension), expected_k(rows * index_dimension);
    need(project_queries(view(q.data(), GGML_TYPE_F32, nq, qh, 256), maps, expected_q.data(), &error), error.c_str());
    need(project_keys(view(k.data(), GGML_TYPE_F32, rows, kh, 256), maps, expected_k.data(), &error), error.c_str());
    close(projected.get(), expected_q, 2e-6f, "logical query projection");

    // Close an immutable block by appending across its boundary. Every trailing
    // incomplete block is left out of Ward and must be covered by the window.
    timestamp("append_launch");
    success(hip_append(context, 55, 0, view(dk.data, GGML_TYPE_F32, 511, kh, 256), &error), error);
    need(hip_size(context) == 511, "partial append count");
    success(hip_append(context, 55, 511, view(dk.data + 511 * kh * 256, GGML_TYPE_F32, rows - 511, kh, 256), &error), error);
    timestamp("append_finish");
    std::vector<float> actual_k(expected_k.size());
    checked(hipMemcpy(actual_k.data(), hip_index_keys(context), actual_k.size() * sizeof(float), hipMemcpyDeviceToHost));
    close(actual_k, expected_k, 0.0f, "BF16 logical key projection");
    index_cache cpu;
    cpu.reset(55, maps.identity);
    need(cpu.append(55, 0, expected_k.data(), rows, &error), error.c_str());
    need(hip_ambiguous_tie_clusters(context) == cpu.ambiguous_tie_clusters(), "Ward cutoff ambiguity flags");
    std::vector<bucket> actual_b(cpu.buckets().size());
    checked(hipMemcpy(actual_b.data(), hip_index_buckets(context), actual_b.size() * sizeof(bucket), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < actual_b.size(); ++i) {
        need(std::equal(actual_b[i].members, actual_b[i].members + bucket_capacity, cpu.buckets()[i].members), "Ward membership matches CPU");
        close({actual_b[i].centroid, actual_b[i].centroid + index_dimension},
              {cpu.buckets()[i].centroid, cpu.buckets()[i].centroid + index_dimension}, 2e-6f, "Ward centroids");
    }
    const std::vector<int32_t> positions{511, 1024, 1536};
    device_array<int32_t> dp(positions), selected(nq * 1024), lengths(nq);
    const budgets budget{128, 32, 512, 4};
    const hip_result selection = hip_select(context, 55, projected.data, nq, dp.data, budget, selected.data, lengths.data, 1024, &error);
    std::vector<std::vector<int32_t>> expected_rows(nq);
    bool cpu_success = true;
    for (int64_t qrow = 0; qrow < nq; ++qrow) {
        cpu_success &= cpu.select(55, expected_q.data() + qrow * index_dimension, positions[qrow], budget, expected_rows[qrow], &error);
    }
    need((selection == hip_result::success) == cpu_success, "selection failure agrees with CPU ambiguity policy");
    if (cpu_success) {
        const auto ids = selected.get(), lens = lengths.get();
        for (int64_t qrow = 0; qrow < nq; ++qrow) {
            need(lens[qrow] == int32_t(expected_rows[qrow].size()), "selected length");
            need(std::equal(expected_rows[qrow].begin(), expected_rows[qrow].end(), ids.begin() + qrow * 1024), "stable flat selection");
            for (int64_t j = lens[qrow]; j < 1024; ++j) { need(ids[qrow * 1024 + j] == -1, "selection -1 padding"); }
        }
    }
    need(hip_select(context, 56, projected.data, nq, dp.data, budget, selected.data, lengths.data, 1024, &error) == hip_result::invalid_argument, "wrong epoch rejected");
    success(hip_truncate(context, 55, 513, &error), error);
    success(hip_append(context, 55, 513, view(dk.data + 513 * kh * 256, GGML_TYPE_F32, rows - 513, kh, 256), &error), error);
    need(hip_size(context) == rows, "rollback then append");
    exact_attention(context, maps, GGML_TYPE_F32);
    exact_attention(context, maps, GGML_TYPE_BF16);
    exact_attention(context, maps, GGML_TYPE_Q8_0);
    success(hip_reset(context, 77, maps.identity, &error), error);
    need(hip_size(context) == 0 && hip_ambiguous_tie_clusters(context) == 0, "slot reuse clears cache");
    hip_destroy(context);
}

void stable_tie_fixture() {
    const auto maps = maps_for(24, 4, true);
    const std::vector<float> k(1025 * 4 * 256, 1.0f), q(24 * 256, 1.0f);
    device_array<float> dk(k), dq(q), iq(index_dimension);
    device_array<int32_t> position(std::vector<int32_t>{1024}), selected(532), lengths(1);
    hip_context * c = nullptr;
    std::string error;
    success(hip_create(maps, {1025, 1, 532}, nullptr, &c, &error), error);
    success(hip_reset(c, 1, maps.identity, &error), error);
    timestamp("tie_append_launch");
    success(hip_append(c, 1, 0, view(dk.data, GGML_TYPE_F32, 1025, 4, 256), &error), error);
    timestamp("tie_append_finish");
    success(hip_project_queries(c, view(dq.data, GGML_TYPE_F32, 1, 24, 256), iq.data, &error), error);
    success(hip_select(c, 1, iq.data, 1, position.data, {16, 2, 512, 4}, selected.data, lengths.data, 532, &error), error);
    const auto ids = selected.get();
    need(lengths.get()[0] == 532, "tied bucket selection length");
    for (int i = 0; i < 16; ++i) { need(ids[i] == i + 4, "bucket-score ties choose lower bucket and ascending members"); }
    for (int i = 0; i < 512; ++i) { need(ids[i + 16] == i + 513, "tied selection appends window"); }
    for (int i = 0; i < 4; ++i) { need(ids[i + 528] == i, "tied selection appends sinks"); }
    hip_destroy(c);
}

void numeric_failure_fixture() {
    const auto maps = maps_for(24, 4);
    hip_context * c = nullptr;
    std::string error;
    hip_memory memory;
    success(hip_memory_requirements(maps.shape, {1,1,2}, &memory, &error), error);
    need(memory.total_device_bytes == memory.persistent_bytes + memory.workspace_bytes, "memory estimate sum");
    need(hip_create(maps, {1,1,2,1}, nullptr, &c, &error) == hip_result::resource_error && !c, "caller memory admission cap");
    success(hip_create(maps, {1,1,2}, nullptr, &c, &error), error);
    witness_gate();
    device_array<float> q(std::vector<float>(24*256, std::numeric_limits<float>::max()));
    device_array<float> k(std::vector<float>(4*256, 2.0f)), v(std::vector<float>(4*256, 1.0f));
    device_array<float> out(24*256), projected(index_dimension);
    device_array<int32_t> ids(std::vector<int32_t>{0,0}), position(std::vector<int32_t>{0});
    attention_args a;
    a.q = view(q.data, GGML_TYPE_F32, 1,24,256); a.k = view(k.data, GGML_TYPE_F32,1,4,256);
    a.v = view(v.data, GGML_TYPE_F32,1,4,256); a.positions=position.data;
    a.main={ids.data,nullptr,2}; a.output=out.data; a.softcap=3.0f;
    // Independent scalar physical-F32 oracle: validation occurs before cap.
    // A finite tanh(infinity) does not make the original dot a finite logit.
    volatile float maximum = std::numeric_limits<float>::max();
    float raw_dot = 0.0f;
    for (int d = 0; d < 256; ++d) { raw_dot += maximum * 2.0f; }
    need(!std::isfinite(raw_dot) && std::isfinite(3.0f * std::tanh(raw_dot / 3.0f)), "independent precap overflow oracle");
    timestamp("numeric_launch");
    need(hip_selected_attention(c,a,nullptr,&error) == hip_result::invalid_argument, "softcap cannot hide nonfinite raw dot");
    need(hip_project_queries(c,a.q,projected.data,&error) == hip_result::invalid_argument, "projection cannot hide nonfinite accumulator");
    q.put(std::vector<float>(24*256,0.0f)); v.put(std::vector<float>(4*256,std::numeric_limits<float>::max()));
    need(hip_selected_attention(c,a,nullptr,&error) == hip_result::invalid_argument, "nonfinite selected output rejected");
    timestamp("numeric_finish");
    hip_destroy(c);
}

void optional_stats_fixture() {
    const auto maps = maps_for(24, 4);
    hip_context * c = nullptr;
    std::string error;
    success(hip_create(maps, {8,1,7}, nullptr, &c, &error), error);
    witness_gate();
    const auto q = samples(24*256, 911), k = samples(8*4*256, 912), v = samples(k.size(), 913);
    device_array<float> dq(q), dk(k), dv(v), output(24*256);
    device_array<int32_t> main(std::vector<int32_t>{0,-1,3,3,7}), extra(std::vector<int32_t>{1,-1});
    device_array<int32_t> position(std::vector<int32_t>{7});
    attention_args a;
    a.q=view(dq.data,GGML_TYPE_F32,1,24,256); a.k=view(dk.data,GGML_TYPE_F32,8,4,256);
    a.v=view(dv.data,GGML_TYPE_F32,8,4,256); a.main={main.data,nullptr,5}; a.extra={extra.data,nullptr,2};
    a.positions=position.data; a.output=output.data; a.scale=0.0625f;
    attention_stats stats;
    timestamp("stats_launch");
    success(hip_selected_attention(c,a,&stats,&error), error);
    const auto counted = output.get();
    need(stats.key_rows == 5*24 && stats.value_rows == 5*24, "optional stats count actual selected slots including duplicates");
    success(hip_selected_attention(c,a,nullptr,&error), error);
    need(output.get() == counted, "stats-disabled output exactly matches stats-enabled output");
    timestamp("stats_finish");
    hip_destroy(c);
}

void author_tied_fixture(const char * path) {
    std::ifstream input(path, std::ios::binary);
    auto read = [&](void * dst, size_t bytes) { return bool(input.read(static_cast<char *>(dst), bytes)); };
    char magic[8]; uint32_t nk = 0, nq = 0;
    need(read(magic, 8) && std::memcmp(magic, "LSAFIX1\0", 8) == 0 &&
         read(&nk, 4) && read(&nq, 4) && nk == 513 && nq == 1, "bounded author tie fixture header");
    std::vector<float> keys(size_t(nk) * index_dimension), query(index_dimension);
    int32_t position = -1;
    std::vector<int32_t> members(buckets_per_cluster * bucket_capacity);
    std::vector<float> centroids(buckets_per_cluster * index_dimension);
    need(read(keys.data(), keys.size() * sizeof(float)) && read(query.data(), query.size() * sizeof(float)) &&
         read(&position, sizeof(position)) && read(members.data(), members.size() * sizeof(int32_t)) &&
         read(centroids.data(), centroids.size() * sizeof(float)) && position == 512,
         "complete bounded author tie fixture");
    int32_t expected_count = -1;
    need(read(&expected_count, 4) && expected_count >= 0 && expected_count <= 4612,
         "bounded author tie selected count");
    std::vector<int32_t> expected(expected_count);
    need(read(expected.data(), expected.size() * sizeof(int32_t)) &&
         input.peek() == std::char_traits<char>::eof(), "complete author tie selected rows");

    projection_maps maps;
    maps.shape = {24, 4, 256, 64}; maps.identity = "pinned-author-ward-tie";
    maps.q.resize(maps.shape.map_rows() * index_dimension);
    maps.k.resize(maps.q.size());
    for (int d = 0; d < index_dimension; ++d) {
        maps.q[d * index_dimension + d] = ggml_fp32_to_bf16(1.0f);
        maps.k[d * index_dimension + d] = ggml_fp32_to_bf16(1.0f);
    }
    std::vector<float> physical_k(size_t(nk) * 4 * 256, 0.0f), physical_q(24 * 256, 0.0f);
    for (uint32_t row = 0; row < nk; ++row)
        for (int d = 0; d < index_dimension; ++d)
            physical_k[(size_t(row) * 4 * 256) + 64 + d] = keys[size_t(row) * index_dimension + d];
    for (int d = 0; d < index_dimension; ++d) physical_q[64 + d] = query[d];
    device_array<float> dk(physical_k), dq(physical_q), projected(index_dimension);
    device_array<int32_t> dp(std::vector<int32_t>{position}), selected(4612), lengths(1);
    hip_context * c = nullptr;
    std::string error;
    success(hip_create(maps, {nk, 1, 4612}, nullptr, &c, &error), error);
    witness_gate();
    success(hip_reset(c, 1, maps.identity, &error), error);
    timestamp("author_tie_launch");
    success(hip_append(c, 1, 0, view(dk.data, GGML_TYPE_F32, nk, 4, 256), &error), error);
    success(hip_project_queries(c, view(dq.data, GGML_TYPE_F32, 1, 24, 256), projected.data, &error), error);
    std::vector<float> actual_keys(keys.size());
    checked(hipMemcpy(actual_keys.data(), hip_index_keys(c), keys.size() * sizeof(float), hipMemcpyDeviceToHost));
    need(actual_keys == keys, "author tie projected keys exact");
    std::vector<bucket> actual_b(buckets_per_cluster);
    checked(hipMemcpy(actual_b.data(), hip_index_buckets(c), actual_b.size() * sizeof(bucket), hipMemcpyDeviceToHost));
    for (int b = 0; b < buckets_per_cluster; ++b) {
        need(std::equal(actual_b[b].members, actual_b[b].members + bucket_capacity,
                        members.data() + b * bucket_capacity), "author tie HIP bucket members exact");
        close({actual_b[b].centroid, actual_b[b].centroid + index_dimension},
              {centroids.data() + b * index_dimension, centroids.data() + (b + 1) * index_dimension},
              3e-6f, "author tie HIP centroid tolerance");
    }
    success(hip_select(c, 1, projected.data, 1, dp.data, {}, selected.data, lengths.data, 4612, &error), error);
    need(lengths.get()[0] == expected_count, "author tie HIP selected length");
    const auto actual = selected.get();
    need(std::equal(expected.begin(), expected.end(), actual.begin()), "author tie HIP ordered indices exact");
    timestamp("author_tie_finish");
    hip_destroy(c);
}
} // namespace

int main(int argc, char ** argv) {
    const bool author_tie = argc == 3 && std::strcmp(argv[1], "--run-hip-ward-fixture") == 0;
    const bool target_27b = argc == 2 && std::strcmp(argv[1], "--run-hip-27b") == 0;
    const bool numeric_only = argc == 2 && std::strcmp(argv[1], "--run-hip-numeric") == 0;
    const bool stats_only = argc == 2 && std::strcmp(argv[1], "--run-hip-stats") == 0;
    if (!author_tie && (argc != 2 || (std::strcmp(argv[1], "--run-hip") != 0 && !target_27b && !numeric_only && !stats_only))) {
        std::puts("SKIP: LSA HIP device parity requires explicit --run-hip and externally held resource claims");
        return 77;
    }
    checked(hipSetDevice(0));
    hipDeviceProp_t properties;
    checked(hipGetDeviceProperties(&properties, 0));
    need(std::strstr(properties.gcnArchName, "gfx90a") != nullptr, "device0 is gfx90a");
    std::printf("device=0 arch=%s pid=%d\n", properties.gcnArchName, getpid());
    std::fflush(stdout);
    if (author_tie) {
        author_tied_fixture(argv[2]);
        timestamp("last_finish");
        std::puts("LSA HIP pinned author Ward tie fixture passed");
        return 0;
    }
    if (stats_only) {
        optional_stats_fixture();
        std::puts("LSA HIP optional stats output/counter regression passed");
        return 0;
    }
    if (numeric_only) {
        numeric_failure_fixture();
        std::puts("LSA HIP numeric failure/admission guards passed");
        return 0;
    }
    geometry_fixture(24, 4);
    if (!target_27b) geometry_fixture(16, 2);
    stable_tie_fixture();
    numeric_failure_fixture();
    optional_stats_fixture();
    timestamp("last_finish");
    std::puts(target_27b ? "LSA HIP 27B-shape synthetic passed" :
        "LSA HIP device parity passed (projection, Ward, selection, selected-only F32/BF16/Q8 attention)");
    return 0;
}
