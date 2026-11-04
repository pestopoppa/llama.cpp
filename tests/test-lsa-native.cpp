#include "../src/llama-lsa-native.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

using namespace llama_lsa;

static int failures = 0;
static void check(bool value, const char * name) {
    std::printf("%s %s\n", value ? "PASS" : "FAIL", name);
    failures += !value;
}
static bool near(float a, float b, float tolerance = 1e-6f) { return std::fabs(a - b) <= tolerance; }
static tensor_view view(const void * data, ggml_type type, int64_t rows, int heads, int dim) {
    tensor_view v;
    v.data = data; v.type = type; v.rows = rows; v.heads = heads; v.dim = dim;
    v.head_stride = ggml_row_size(type, dim); v.row_stride = heads * v.head_stride;
    return v;
}

static void projection_tests() {
    for (const auto shape : { geometry{24, 4, 256, 64}, geometry{16, 2, 256, 64} }) {
        projection_maps maps;
        maps.shape = shape; maps.identity = "analytic-fixture";
        maps.q.resize(shape.map_rows() * index_dimension);
        maps.k.resize(maps.q.size());
        // Each index channel copies one RoPE-free coordinate of the first KV group.
        for (int d = 0; d < index_dimension; ++d) {
            maps.q[d * index_dimension + d] = ggml_fp32_to_bf16(1.0f);
            maps.k[d * index_dimension + d] = ggml_fp32_to_bf16(1.0f);
        }
        std::vector<float> q(shape.q_heads * shape.head_dim, 0.0f), k(shape.kv_heads * shape.head_dim, 0.0f);
        const int group = shape.q_heads / shape.kv_heads;
        for (int h = 0; h < shape.q_heads; ++h) {
            // Rotary values must have no effect, even at huge magnitude.
            std::fill(q.begin() + h * shape.head_dim, q.begin() + h * shape.head_dim + shape.rope_dim, 1e20f);
            q[h * shape.head_dim + shape.rope_dim] = 1.0f / group;
        }
        k[shape.rope_dim] = 1.0f;
        float iq[index_dimension], ik[index_dimension];
        check(project_queries(view(q.data(), GGML_TYPE_F32, 1, shape.q_heads, shape.head_dim), maps, iq), "logical Q projection supports GQA geometry");
        check(project_keys(view(k.data(), GGML_TYPE_F32, 1, shape.kv_heads, shape.head_dim), maps, ik), "logical K projection supports geometry");
        check(iq[0] == 1.0f && ik[0] == 1.0f && iq[1] == 0.0f && ik[1] == 0.0f, "GQA sum excludes all rotary coordinates");
        // BF16 midpoint: 1+1/256 rounds to even 1; next representable BF16 is1+1/128.
        std::fill(q.begin(), q.end(), 0.0f);
        q[shape.rope_dim] = 1.0f + 1.0f / 256.0f;
        q[shape.rope_dim + 1] = 1.0f;
        project_queries(view(q.data(), GGML_TYPE_F32, 1, shape.q_heads, shape.head_dim), maps, iq);
        const float half = 1.0f / std::sqrt(2.0f);
        check(near(iq[0], half) && near(iq[1], half), "BF16 midpoint input rounds ties to even before projection");
        q[shape.rope_dim] = std::nextafter(1.0f + 1.0f / 256.0f, 2.0f);
        project_queries(view(q.data(), GGML_TYPE_F32, 1, shape.q_heads, shape.head_dim), maps, iq);
        check(iq[0] > iq[1], "BF16 midpoint next float retains rounding direction");
        std::fill(q.begin(), q.end(), 0.0f);
        q[shape.rope_dim] = q[shape.rope_dim + 1] = q[shape.rope_dim + 2] = 1.0f;
        maps.q[2 * index_dimension] = ggml_fp32_to_bf16(1.0f / 256.0f);
        project_queries(view(q.data(), GGML_TYPE_F32, 1, shape.q_heads, shape.head_dim), maps, iq);
        // Ignore index2 when checking the ratio: it also carries copied input2.
        check(iq[0] == iq[1], "BF16 matmul output midpoint rounds before normalization");
        maps.identity.clear();
        check(!project_queries(view(q.data(), GGML_TYPE_F32, 1, shape.q_heads, shape.head_dim), maps, iq), "unidentified projection maps fail closed");
    }
}

static bool equal_buckets(const index_cache & a, const index_cache & b) {
    if (a.buckets().size() != b.buckets().size()) { return false; }
    for (size_t i = 0; i < a.buckets().size(); ++i) {
        if (std::memcmp(a.buckets()[i].members, b.buckets()[i].members, sizeof(bucket::members)) ||
            std::memcmp(a.buckets()[i].centroid, b.buckets()[i].centroid, sizeof(bucket::centroid))) { return false; }
    }
    return true;
}

static void cache_tests() {
    std::vector<float> keys(1537 * index_dimension, 0.0f);
    // Identical keys exercise partner ties, final-round ties and bucket-score ties.
    for (int i = 0; i < 1537; ++i) { keys[i * index_dimension] = 1.0f; }
    index_cache cache;
    cache.reset(7, "maps-A");
    check(cache.append(7, 0, keys.data(), 511) && cache.buckets().empty(), "partial cluster never publishes a bucket");
    check(cache.append(7, 511, keys.data() + 511 * index_dimension, 1) && cache.buckets().size() == 64, "cluster publishes exactly at512");
    check(cache.append(7, 512, keys.data() + 512 * index_dimension, 1025) && cache.buckets().size() == 192, "append completes clusters incrementally");
    std::vector<int32_t> rows;
    budgets b; b.top_k = 16; b.top_p = 2;
    check(cache.select(7, keys.data(), 1536, b, rows) && rows.size() == 532, "selection count equals budget plus window plus sinks");
    bool expected = rows.size() == 532;
    if (expected) {
        for (int i = 0; i < 16; ++i) { expected &= rows[i] == i + 4; }
        for (int i = 0; i < 512; ++i) { expected &= rows[16 + i] == 1025 + i; }
        for (int i = 0; i < 4; ++i) { expected &= rows[528 + i] == i; }
    }
    check(expected, "stable score ties expand ascending bucket members then window and sinks");
    b.top_k = 0;
    check(cache.select(7, keys.data(), 0, b, rows) && rows == std::vector<int32_t>{0}, "first position contains itself once");
    check(!cache.select(8, keys.data(), 0, b, rows) && !cache.append(8, cache.size(), nullptr, 0), "request epoch mismatch rejected");
    check(!cache.append(7, 0, keys.data(), 1), "nonappend cache overwrite rejected");
    check(cache.truncate(7, 700) && cache.buckets().size() == 64, "rollback invalidates affected cluster and later clusters");
    check(cache.append(7, 700, keys.data() + 700 * index_dimension, 837), "rollback can append accepted replacement suffix");
    index_cache fresh; fresh.reset(7, "maps-A"); fresh.append(7, 0, keys.data(), 1537);
    check(equal_buckets(cache, fresh), "rollback reconstruction matches fresh clustering exactly");
    cache.reset(8, "maps-B");
    check(cache.size() == 0 && cache.buckets().empty() && !cache.matches(7, "maps-A") && cache.matches(8, "maps-B"), "slot reuse and map change clear all native cache state");

    // Paired duplicates and BF16-near-ties exercise simultaneous mutual merges.
    // These are structural tests; author parity requires --reference-fixture.
    for (int profile = 0; profile < 3; ++profile) {
        for (int token = 0; token < 1537; ++token) {
            std::fill(keys.begin() + token * index_dimension, keys.begin() + (token + 1) * index_dimension, 0.0f);
            const int pool = (token % cluster_size) / 4;
            const int axis = profile == 2 && pool >= 124 ? pool - 124 : pool / 2;
            keys[token * index_dimension + axis] = 1.0f;
            if (profile == 1) { keys[token * index_dimension + (pool / 2 + 1) % index_dimension] = (pool % 2 ? 1 : -1) / 256.0f; }
        }
        cache.reset(10, "ties");
        check(cache.append(10, 0, keys.data(), 1537), "paired duplicate or BF16-near-tie clustering completes");
        std::vector<int> seen(1536, 0);
        bool partition = true;
        for (const auto & bucket : cache.buckets()) {
            int previous = -1;
            for (int32_t member : bucket.members) {
                if (member < 0) { continue; }
                partition &= member > previous && member < 1536;
                previous = member;
                if (member < 1536) { ++seen[member]; }
            }
        }
        partition &= std::all_of(seen.begin(), seen.end(), [](int count) { return count == 1; });
        check(partition, "tie fixtures preserve a capped complete partition in member order");
        if (profile == 2) {
            check(cache.ambiguous_tie_clusters() == 0, "pinned author full-rank tie ordering resolves all cutoffs");
            check(cache.select(10, keys.data(), 511, {}, rows), "causal window selection before first completed cluster");
            check(cache.select(10, keys.data(), 512, {}, rows), "visible tied cluster selects after author-rank resolution");
            check(cache.truncate(10, 511) && cache.ambiguous_tie_clusters() == 0, "rollback clears tied clusters");
        }
    }
}

// Little-endian fixture generated from the unchanged author Python oracle:
// magic[8]="LSAFIX1\0", uint32 nkeys,nqueries; FP32 keys[nkeys,64],
// queries[nqueries,64]; int32 positions[nqueries]; int32 members[nkeys/512*64,16];
// FP32 centroids[nkeys/512*64,64]; for each query: int32 count, rows[count].
// Include identical, paired-duplicate and BF16-near-tie/final-truncation cases.
static void reference_fixture(const char * path) {
    std::ifstream input(path, std::ios::binary);
    auto read = [&](void * out, size_t bytes) { return bool(input.read(static_cast<char *>(out), bytes)); };
    char magic[8]; uint32_t nk = 0, nq = 0;
    if (!read(magic, 8) || std::memcmp(magic, "LSAFIX1\0", 8) || !read(&nk, 4) || !read(&nq, 4) ||
        nk > 16384 || nq > 32 || !nk || !nq) { check(false, "valid bounded author fixture header"); return; }
    std::vector<float> keys((size_t) nk * index_dimension), queries((size_t) nq * index_dimension);
    std::vector<int32_t> positions(nq), members((nk / cluster_size) * buckets_per_cluster * bucket_capacity);
    std::vector<float> centroids((nk / cluster_size) * buckets_per_cluster * index_dimension);
    if (!read(keys.data(), keys.size() * sizeof(float)) || !read(queries.data(), queries.size() * sizeof(float)) ||
        !read(positions.data(), positions.size() * sizeof(int32_t)) || !read(members.data(), members.size() * sizeof(int32_t)) ||
        !read(centroids.data(), centroids.size() * sizeof(float))) { check(false, "complete author fixture payload"); return; }
    index_cache cache; cache.reset(1, "author-fixture");
    if (!cache.append(1, 0, keys.data(), nk)) { check(false, "author fixture keys cluster"); return; }
    bool members_equal = true, centroids_equal = true;
    for (size_t b = 0; b < cache.buckets().size(); ++b) {
        members_equal &= !std::memcmp(cache.buckets()[b].members, members.data() + b * bucket_capacity, bucket_capacity * sizeof(int32_t));
        for (int d = 0; d < index_dimension; ++d) {
            centroids_equal &= near(cache.buckets()[b].centroid[d], centroids[b * index_dimension + d], 3e-6f);
        }
    }
    std::printf("AUTHOR_FIXTURE members_exact=%d centroids_tol=%d ambiguous_clusters=%zu\n", members_equal, centroids_equal, cache.ambiguous_tie_clusters());
    check(cache.ambiguous_tie_clusters() == 0, "author fixture all tie cutoffs resolved");
    check(members_equal, "author fixture bucket members exact, including ties");
    check(centroids_equal, "author fixture centroid error <=3e-6");
    for (uint32_t i = 0; i < nq; ++i) {
        int32_t count;
        if (!read(&count, 4) || count < 0 || count > 4612) { check(false, "bounded author selected count"); return; }
        std::vector<int32_t> expected(count), actual;
        if (!read(expected.data(), count * sizeof(int32_t))) { check(false, "complete author selected rows"); return; }
        std::string error;
        const bool selected = cache.select(1, queries.data() + i * index_dimension, positions[i], {}, actual, &error);
        check(selected && actual == expected, "author fixture selected indices exact, including ties");
    }
    check(input.peek() == std::char_traits<char>::eof(), "author fixture has no trailing bytes");
}

// LSAPRO2 fixture: uint32 nq,nk; BF16 WQ/WK[768,64], F32 Q[nq,24,256],
// K[nk,4,256], expected IQ[nq,64], BF16-rounded IK[nk,64] stored asF32;
// positions[nq], then each query's int32 count and expected selected rows.
static void projection_fixture(const char * path) {
    std::ifstream input(path, std::ios::binary);
    auto read = [&](void * out, size_t bytes) { return bool(input.read(static_cast<char *>(out), bytes)); };
    char magic[8]; uint32_t nq = 0, nk = 0;
    if (!read(magic, 8) || std::memcmp(magic, "LSAPRO2\0", 8) || !read(&nq, 4) || !read(&nk, 4) ||
        nq > 32 || nk > 16384 || !nk || !nq) { check(false, "valid bounded projection fixture header"); return; }
    projection_maps maps; maps.identity = "author-projection-fixture";
    maps.q.resize(768 * index_dimension); maps.k.resize(maps.q.size());
    std::vector<float> q((size_t) nq * 24 * 256), k((size_t) nk * 4 * 256),
        expected_q((size_t) nq * index_dimension), expected_k((size_t) nk * index_dimension),
        actual_q(expected_q.size()), actual_k(expected_k.size());
    if (!read(maps.q.data(), maps.q.size() * sizeof(ggml_bf16_t)) || !read(maps.k.data(), maps.k.size() * sizeof(ggml_bf16_t)) ||
        !read(q.data(), q.size() * sizeof(float)) || !read(k.data(), k.size() * sizeof(float)) ||
        !read(expected_q.data(), expected_q.size() * sizeof(float)) || !read(expected_k.data(), expected_k.size() * sizeof(float))) {
        check(false, "complete author projection fixture"); return;
    }
    check(project_queries(view(q.data(), GGML_TYPE_F32, nq, 24, 256), maps, actual_q.data()), "author query projection fixture executes");
    check(project_keys(view(k.data(), GGML_TYPE_F32, nk, 4, 256), maps, actual_k.data()), "author key projection fixture executes");
    float q_error = 0.0f, k_error = 0.0f;
    for (size_t i = 0; i < actual_q.size(); ++i) { q_error = std::max(q_error, std::fabs(actual_q[i] - expected_q[i])); }
    int mismatches = 0, maximum_bf16_steps = 0;
    for (size_t i = 0; i < actual_k.size(); ++i) {
        k_error = std::max(k_error, std::fabs(actual_k[i] - expected_k[i]));
        if (actual_k[i] != expected_k[i]) {
            const int steps = std::abs(int(ggml_fp32_to_bf16(actual_k[i]).bits) - int(ggml_fp32_to_bf16(expected_k[i]).bits));
            maximum_bf16_steps = std::max(maximum_bf16_steps, steps);
            if (mismatches++ < 8) {
                std::printf("KEY_MISMATCH row=%zu dim=%zu actual=%g expected=%g bf16_steps=%d\n", i / index_dimension, i % index_dimension, actual_k[i], expected_k[i], steps);
            }
        }
    }
    std::printf("KEY_MISMATCH count=%d max_bf16_steps=%d\n", mismatches, maximum_bf16_steps);
    std::printf("AUTHOR_PROJECTION q_max_abs=%g k_max_abs=%g\n", q_error, k_error);
    check(q_error <= 3e-6f, "author query projections within3e-6");
    // FP32 GEMM and L2 reduction orders can flip a BF16 midpoint. The producer
    // records raw BF16, FP32 norms and an independent FP64 diagnostic. Require
    // one BF16 step here AND exact coupled selected rows below, not tensor parity.
    check(maximum_bf16_steps <= 1, "author BF16 index keys differ by at most one BF16 step");
    std::vector<int32_t> positions(nq);
    if (!read(positions.data(), positions.size() * sizeof(int32_t))) { check(false, "complete coupled query positions"); return; }
    index_cache cache; cache.reset(1, "coupled-author-fixture");
    check(cache.append(1, 0, actual_k.data(), nk), "native projected keys build native Ward cache");
    index_cache expected_cache; expected_cache.reset(1, "coupled-author-fixture");
    expected_cache.append(1, 0, expected_k.data(), nk);
    bool same_partition = cache.buckets().size() == expected_cache.buckets().size();
    for (size_t b = 0; same_partition && b < cache.buckets().size(); ++b) {
        same_partition &= !std::memcmp(cache.buckets()[b].members, expected_cache.buckets()[b].members, bucket_capacity * sizeof(int32_t));
    }
    std::printf("COUPLED_PARTITION native_vs_author_keys_members_exact=%d\n", same_partition);
    for (uint32_t iq = 0; iq < nq; ++iq) {
        int32_t count;
        if (!read(&count, 4) || count < 0 || count > 4612) { check(false, "bounded coupled author count"); return; }
        std::vector<int32_t> expected(count), actual;
        if (!read(expected.data(), count * sizeof(int32_t))) { check(false, "complete coupled author rows"); return; }
        std::string error;
        const bool selected = cache.select(1, actual_q.data() + iq * index_dimension, positions[iq], {}, actual, &error);
        if (selected) {
            auto sorted_actual = actual, sorted_expected = expected;
            std::sort(sorted_actual.begin(), sorted_actual.end()); std::sort(sorted_expected.begin(), sorted_expected.end());
            std::printf("COUPLED_INDICES position=%d order_exact=%d set_exact=%d actual=%zu expected=%zu\n",
                    positions[iq], actual == expected, sorted_actual == sorted_expected, actual.size(), expected.size());
            check(sorted_actual == sorted_expected, "coupled native projection selected attention support exact");
            // Independent consequence check: same physical Q/K/V and selected
            // support, with each backend's slot order, must agree numerically.
            std::vector<float> values((size_t) nk * 4 * 32), actual_output(24 * 32), expected_output(24 * 32);
            for (size_t j = 0; j < values.size(); ++j) { values[j] = (int(j % 31) - 15) / 16.0f; }
            attention_args attention;
            attention.q = view(q.data() + iq * 24 * 256, GGML_TYPE_F32, 1, 24, 256);
            attention.k = view(k.data(), GGML_TYPE_F32, nk, 4, 256);
            attention.v = view(values.data(), GGML_TYPE_F32, nk, 4, 32);
            attention.positions = positions.data() + iq; attention.scale = 1.0f / 16;
            attention.main = {actual.data(), nullptr, (int64_t) actual.size()}; attention.output = actual_output.data();
            const bool actual_ok = selected_attention_cpu(attention);
            attention.main = {expected.data(), nullptr, (int64_t) expected.size()}; attention.output = expected_output.data();
            const bool expected_ok = selected_attention_cpu(attention);
            float output_error = 0.0f;
            for (size_t j = 0; j < actual_output.size(); ++j) { output_error = std::max(output_error, std::fabs(actual_output[j] - expected_output[j])); }
            std::printf("COUPLED_ATTENTION position=%d output_max_abs=%g\n", positions[iq], output_error);
            check(actual_ok && expected_ok && output_error <= 1e-5f, "coupled selected attention output within1e-5 despite possible slot order rounding");
            if (actual != expected) {
                size_t first = 0;
                while (first < actual.size() && first < expected.size() && actual[first] == expected[first]) { ++first; }
                if (first < actual.size() && first < expected.size()) {
                    std::printf("COUPLED_FIRST_DIFFERENCE slot=%zu native_row=%d author_row=%d\n", first, actual[first], expected[first]);
                }
            }
            check(actual == expected, "coupled native projection Ward and selection author indices exact");
        }
        else { check(false, "coupled author Ward selection must not decline a resolved tie"); }
    }
    check(input.peek() == std::char_traits<char>::eof(), "projection fixture has no trailing bytes");
}

static void attention_tests(ggml_type type) {
    constexpr int dim = 32, heads = 4, kv_heads = 2, nrows = 7;
    std::vector<float> q(heads * dim, 0.0f), k(nrows * kv_heads * dim), v(k.size());
    for (int h = 0; h < heads; ++h) { q[h * dim] = 0.5f; }
    for (int row = 0; row < nrows; ++row) {
        for (int head = 0; head < kv_heads; ++head) {
            for (int d = 0; d < dim; ++d) {
                k[(row * kv_heads + head) * dim + d] = (row - 3) * 0.125f;
                v[(row * kv_heads + head) * dim + d] = (row + 2 * head + d) * 0.25f;
            }
        }
    }
    const size_t row_bytes = ggml_row_size(type, dim);
    std::vector<uint64_t> kb((nrows * kv_heads * row_bytes + 7) / 8), vb(kb.size());
    auto encode = [&](const std::vector<float> & src, std::vector<uint64_t> & dest) {
        for (int i = 0; i < nrows * kv_heads; ++i) {
            void * p = reinterpret_cast<char *>(dest.data()) + i * row_bytes;
            if (type == GGML_TYPE_F32) { std::memcpy(p, src.data() + i * dim, row_bytes); }
            else { ggml_get_type_traits(type)->from_float_ref(src.data() + i * dim, p, dim); }
        }
    };
    encode(k, kb); encode(v, vb);
    const int32_t ids[] = { 5, -1, 1, 6 }, extra[] = { 1, -1 };
    int32_t length = 3, extra_length = 2, position = 5;
    float sink[heads] = { -0.5f, -0.5f, -0.5f, -0.5f };
    std::vector<float> output(heads * dim, -999.0f), gathered_out(output.size());
    attention_args a;
    a.q = view(q.data(), GGML_TYPE_F32, 1, heads, dim);
    a.k = view(kb.data(), type, nrows, kv_heads, dim);
    a.v = view(vb.data(), type, nrows, kv_heads, dim);
    a.main = {ids, &length, 4}; a.extra = {extra, &extra_length, 2};
    a.positions = &position; a.attn_sink = sink; a.output = output.data();
    attention_stats stats;
    check(selected_attention_cpu(a, &stats), "selected attention accepts ragged indices, holes, extras and sinks");
    check(stats.key_rows == 12 && stats.value_rows == 12, "only three selected slots per head touch KV including duplicate extra");

    // Independent gather oracle: construct the same ordered slots as a compact KV.
    const int32_t selected[] = {5, 1, 1}, compact_ids[] = {0, 1, 2};
    std::vector<uint64_t> compact_k((3 * kv_heads * row_bytes + 7) / 8), compact_v(compact_k.size());
    for (int j = 0; j < 3; ++j) {
        std::memcpy(reinterpret_cast<char *>(compact_k.data()) + j * kv_heads * row_bytes,
                    reinterpret_cast<char *>(kb.data()) + selected[j] * kv_heads * row_bytes, kv_heads * row_bytes);
        std::memcpy(reinterpret_cast<char *>(compact_v.data()) + j * kv_heads * row_bytes,
                    reinterpret_cast<char *>(vb.data()) + selected[j] * kv_heads * row_bytes, kv_heads * row_bytes);
    }
    auto gathered = a;
    gathered.k = view(compact_k.data(), type, 3, kv_heads, dim);
    gathered.v = view(compact_v.data(), type, 3, kv_heads, dim);
    int32_t compact_pos = 2;
    gathered.main = { compact_ids, nullptr, 3 }; gathered.extra = {};
    gathered.positions = &compact_pos; gathered.output = gathered_out.data();
    check(selected_attention_cpu(gathered) && !std::memcmp(output.data(), gathered_out.data(), output.size() * sizeof(float)), "sparse full-KV execution equals compact gathered execution bit for bit");
    // Analytic GQA check: constant channel K means exact repeated scores even with Q8.
    const float e5 = std::exp(0.125f), e1 = std::exp(-0.125f), es = std::exp(-0.5f);
    bool analytic = true;
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
            const float expected = (e5 * v[(5 * kv_heads + h / 2) * dim + d] +
                                    2 * e1 * v[(kv_heads + h / 2) * dim + d]) / (e5 + 2 * e1 + es);
            analytic &= near(output[h * dim + d], expected, type == GGML_TYPE_Q8_0 ? 0.04f : 2e-5f);
        }
    }
    check(analytic, "attention matches independent analytic GQA and sink result");
    if (type == GGML_TYPE_F32) {
        float * poisoned_k = reinterpret_cast<float *>(kb.data());
        float * poisoned_v = reinterpret_cast<float *>(vb.data());
        poisoned_k[0] = poisoned_v[0] = std::numeric_limits<float>::quiet_NaN();
        check(selected_attention_cpu(a) && !std::memcmp(output.data(), gathered_out.data(), output.size() * sizeof(float)), "unselected NaN KV is never read");
        poisoned_v[5 * kv_heads * dim] = std::numeric_limits<float>::quiet_NaN();
        std::fill(output.begin(), output.end(), -999.0f);
        check(!selected_attention_cpu(a) && output[0] == -999.0f && output.back() == -999.0f, "nonfinite selected V fails without publishing partial output");
        encode(k, kb); encode(v, vb);
    }
    length = 4;
    std::fill(output.begin(), output.end(), -999.0f);
    check(!selected_attention_cpu(a, &stats) && stats.key_rows == 0 && output[0] == -999.0f, "future row rejected before KV reads or output writes");
    length = -1;
    check(!selected_attention_cpu(a), "negative per-query length rejected");
    length = 0; extra_length = 0;
    check(selected_attention_cpu(a, &stats) && stats.key_rows == 0 &&
          std::all_of(output.begin(), output.end(), [](float x) { return x == 0.0f; }), "zero selected slots with sink produce zero output");
    length = 1; extra_length = 0; a.attn_sink = nullptr;
    selected_attention_cpu(a);
    std::vector<float> decoded(dim);
    bool single = true;
    for (int h = 0; h < heads; ++h) {
        const char * p = reinterpret_cast<const char *>(vb.data()) + (5 * kv_heads + h / 2) * row_bytes;
        if (type == GGML_TYPE_F32) { std::memcpy(decoded.data(), p, dim * sizeof(float)); }
        else { ggml_get_type_traits(type)->to_float(p, decoded.data(), dim); }
        single &= !std::memcmp(output.data() + h * dim, decoded.data(), dim * sizeof(float));
    }
    check(single, "single slot returns stored V exactly for each GQA head");
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (argc == 3 && !std::strcmp(argv[1], "--reference-fixture")) {
        reference_fixture(argv[2]);
        return failures ? 1 : 0;
    }
    if (argc == 3 && !std::strcmp(argv[1], "--projection-fixture")) {
        projection_fixture(argv[2]);
        return failures ? 1 : 0;
    }
    if (argc != 1) { return 2; }
    projection_tests();
    cache_tests();
    for (auto type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0 }) { attention_tests(type); }
    std::printf("RESULT test-lsa-native failures=%d avx512=%d\n", failures, cpu_has_avx512());
    return failures ? 1 : 0;
}
