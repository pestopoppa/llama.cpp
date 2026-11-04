#include "llama-lsa-native.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define LLAMA_LSA_AVX512 1
#endif

namespace llama_lsa {
namespace {

bool fail(std::string * error, const char * message) {
    if (error) { *error = message; }
    return false;
}

float rounded(float value) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)); }

const char * row_data(const tensor_view & view, int64_t row, int head) {
    return static_cast<const char *>(view.data) + row * view.row_stride + head * view.head_stride;
}

bool valid_view(const tensor_view & view) {
    if (view.rows < 0 || view.heads <= 0 || view.dim <= 0 ||
        view.type < 0 || view.type >= GGML_TYPE_COUNT ||
        (view.rows && !view.data)) { return false; }
    const int64_t block = ggml_blck_size(view.type);
    if (block <= 0 || view.dim % block) { return false; }
    const size_t bytes = ggml_row_size(view.type, view.dim);
    // Channels must be contiguous; overlapping heads/rows are unsupported.
    if (view.head_stride < bytes || view.row_stride < bytes ||
        (view.heads > 1 && view.head_stride > (SIZE_MAX - bytes) / (view.heads - 1))) { return false; }
    const size_t last_head = (view.heads - 1) * view.head_stride;
    if (view.rows > 1 && (uint64_t) (view.rows - 1) > (SIZE_MAX - last_head - bytes) / view.row_stride) { return false; }
    // Both token-major and head-major strides are valid, but aliasing rows is not.
    if (view.heads > 1 && view.row_stride < last_head + bytes &&
        (uint64_t) view.rows > view.head_stride / view.row_stride) { return false; }
    return true;
}

bool read_float_row(const tensor_view & view, int64_t row, int head, float * out) {
    const auto * traits = ggml_get_type_traits(view.type);
    if (view.type == GGML_TYPE_F32) {
        std::memcpy(out, row_data(view, row, head), view.dim * sizeof(float));
    } else if (traits->to_float) {
        traits->to_float(row_data(view, row, head), out, view.dim);
    } else { return false; }
    return true;
}

float dot(const float * a, const float * b, int n) {
    float total = 0.0f;
    for (int d = 0; d < n; ++d) { total += a[d] * b[d]; }
    return total;
}

void project_scalar(const float * input, const ggml_bf16_t * map, int n, float * out) {
    std::fill(out, out + index_dimension, 0.0f);
    for (int d = 0; d < n; ++d) {
        for (int j = 0; j < index_dimension; ++j) {
            out[j] += input[d] * ggml_bf16_to_fp32(map[d * index_dimension + j]);
        }
    }
}

#if LLAMA_LSA_AVX512
__attribute__((target("avx512f,avx512bw")))
void project_avx512(const float * input, const ggml_bf16_t * map, int n, float * out) {
    __m512 sum[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
    for (int d = 0; d < n; ++d) {
        const __m512 x = _mm512_set1_ps(input[d]);
        for (int j = 0; j < 4; ++j) {
            const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(map + d * index_dimension + j * 16));
            const __m512 w = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bits), 16));
            const __m512 product = _mm512_mul_round_ps(x, w, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            sum[j] = _mm512_add_round_ps(sum[j], product, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
    }
    for (int j = 0; j < 4; ++j) { _mm512_storeu_ps(out + j * 16, sum[j]); }
}

__attribute__((target("avx512f")))
void score_avx512(const float * query, const bucket * buckets, size_t count, float * scores) {
    const int stride = sizeof(bucket) / sizeof(float);
    const __m512i offsets = _mm512_setr_epi32(0, stride, 2 * stride, 3 * stride, 4 * stride, 5 * stride,
            6 * stride, 7 * stride, 8 * stride, 9 * stride, 10 * stride, 11 * stride, 12 * stride,
            13 * stride, 14 * stride, 15 * stride);
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (int d = 0; d < index_dimension; ++d) {
            const __m512 c = _mm512_i32gather_ps(offsets, buckets[i].centroid + d, sizeof(float));
            const __m512 product = _mm512_mul_round_ps(_mm512_set1_ps(query[d]), c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            sum = _mm512_add_round_ps(sum, product, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        _mm512_storeu_ps(scores + i, sum);
    }
    for (; i < count; ++i) { scores[i] = dot(query, buckets[i].centroid, index_dimension); }
}
#endif

// Experimental pinned PyTorch CPU profile, requires no contraction.
float pinned_norm64(const float * x) {
    volatile float lanes[8] = {};
    for (int i = 0; i < index_dimension; ++i) {
        volatile float product = x[i] * x[i];
        lanes[i % 8] = lanes[i % 8] + product;
    }
    volatile float total = lanes[0];
    for (int i = 1; i < 8; ++i) { total = total + lanes[i]; }
    return std::sqrt(total);
}

#if LLAMA_LSA_AVX512
__attribute__((target("avx512f,avx512bw,avx512bf16")))
static void dpbf16_dot64(const uint16_t *x, const uint16_t *w, int n, float *out) {
    __m512 sums[4] = {_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps()};
    for (int base=0; base<n; base+=512) {
    __m512 acc[4] = {_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps()};
    alignas(64) uint16_t packed[32];
    for (int k=base;k<std::min(base + 512, n);k+=2) {
        uint32_t pair; std::memcpy(&pair,x+k,4);
        auto a = (__m512bh) _mm512_set1_epi32(static_cast<int>(pair));
        for (int block=0;block<4;++block) {
            for (int j=0;j<16;++j) { packed[2*j]=w[k*64+block*16+j]; packed[2*j+1]=w[(k+1)*64+block*16+j]; }
            auto b = (__m512bh) _mm512_load_si512(packed);
            acc[block] = _mm512_dpbf16_ps(acc[block],a,b);
        }
    }
    for(int i=0;i<4;++i) sums[i] = _mm512_add_ps(sums[i],acc[i]);
    }
    for(int i=0;i<4;++i) _mm512_storeu_ps(out+i*16,sums[i]);
}
#endif

bool project(const tensor_view & view, const projection_maps & maps, bool query, float * output, std::string * error) {
    if (!maps.valid() || !valid_view(view) || view.dim != maps.shape.head_dim ||
        view.heads != (query ? maps.shape.q_heads : maps.shape.kv_heads) ||
        (view.rows && !output)) { return fail(error, "unsupported logical projection geometry or map"); }
    const int n = maps.shape.map_rows();
    const int width = maps.shape.head_dim - maps.shape.rope_dim;
    const int group = maps.shape.q_heads / maps.shape.kv_heads;
    std::vector<float> input(n), row(view.dim);
    const auto & map = query ? maps.q : maps.k;
#if LLAMA_LSA_AVX512
    if (!cpu_projection_profile_available(maps.shape)) {
        return fail(error, "pinned CPU BF16 profile unavailable");
    }
    std::vector<uint16_t> bf_map(n * index_dimension);
    std::memcpy(bf_map.data(), map.data(), bf_map.size() * sizeof(uint16_t));
#else
    return fail(error, "pinned CPU BF16 profile unavailable");
#endif
    for (int64_t token = 0; token < view.rows; ++token) {
        std::fill(input.begin(), input.end(), 0.0f);
        for (int head = 0; head < view.heads; ++head) {
            if (!read_float_row(view, token, head, row.data())) { return fail(error, "projection input dtype unsupported"); }
            const int kv_head = query ? head / group : head;
            for (int d = 0; d < width; ++d) {
                const float value = row[maps.shape.rope_dim + d];
                if (!std::isfinite(value)) { return fail(error, "nonfinite logical projection input"); }
                input[kv_head * width + d] += value;
            }
        }
        for (float & x : input) { x = rounded(x); }
        float * out = output + token * index_dimension;
#if LLAMA_LSA_AVX512
        {
            std::vector<uint16_t> bf_input(n);
            for (int d = 0; d < n; ++d) {
                const auto val = ggml_fp32_to_bf16(input[d]);
                std::memcpy(&bf_input[d], &val, sizeof(uint16_t));
            }
            dpbf16_dot64(bf_input.data(), bf_map.data(), n, out);
        }
#endif
        // torch BF16 matmul rounds the result before FP32 normalization.
        for (int d = 0; d < index_dimension; ++d) { out[d] = rounded(out[d]); }
        const float norm = std::max(pinned_norm64(out), 1e-12f);
        for (int d = 0; d < index_dimension; ++d) { out[d] = query ? out[d] / norm : rounded(out[d] / norm); }
    }
    return true;
}

bool cluster(const float * keys, int32_t offset, std::vector<bucket> & result, bool & ambiguous) {
    ambiguous = false;
    constexpr int pools = cluster_size / 4;
    float sums[pools][index_dimension] = {};
    int sizes[pools], parent[pools];
    bool alive[pools];
    for (int i = 0; i < pools; ++i) {
        sizes[i] = 4; parent[i] = i; alive[i] = true;
        for (int k = 0; k < 4; ++k) {
            for (int d = 0; d < index_dimension; ++d) { sums[i][d] += keys[(i * 4 + k) * index_dimension + d]; }
        }
    }
    int count = pools;
    for (int round = 0; round < 256 && count > buckets_per_cluster; ++round) {
        int best[pools];
        float best_cost[pools], squares[pools], norms[pools];
        for (int i = 0; i < pools; ++i) {
            squares[i] = dot(sums[i], sums[i], index_dimension);
            norms[i] = std::sqrt(std::max(0.0f, squares[i]));
            best[i] = -1; best_cost[i] = INFINITY;
        }
        for (int i = 0; i < pools; ++i) {
            if (!alive[i]) { continue; }
            for (int j = 0; j < pools; ++j) {
                if (i == j || !alive[j] || sizes[i] + sizes[j] > bucket_capacity) { continue; }
                const float gram = dot(sums[i], sums[j], index_dimension);
                const float cost = norms[i] + norms[j] - std::sqrt(std::max(0.0f, squares[i] + squares[j] + 2.0f * gram));
                if (cost < best_cost[i]) { best_cost[i] = cost; best[i] = j; }
            }
        }
        // The pinned author calls argsort over all 128 masked costs, including
        // non-mutual +inf entries. Its CPU unstable argsort uses std::sort;
        // sorting only the mutual candidates changes ties at the final cutoff.
        std::array<std::pair<float, int>, pools> ranked;
        for (int i = 0; i < pools; ++i) {
            ranked[i] = {best[i] > i && best[best[i]] == i ? best_cost[i] : INFINITY, i};
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        std::vector<int> merges;
        for (int rank = 0; rank < count - buckets_per_cluster && std::isfinite(ranked[rank].first); ++rank) {
            merges.push_back(ranked[rank].second);
        }
        if (merges.empty()) { return false; }
        for (int i : merges) {
            const int j = best[i];
            for (int d = 0; d < index_dimension; ++d) { sums[i][d] += sums[j][d]; }
            sizes[i] += sizes[j]; alive[j] = false; parent[j] = i; --count;
        }
    }
    if (count != buckets_per_cluster) { return false; }
    const size_t base = result.size();
    result.resize(base + buckets_per_cluster);
    int numbers[pools], next = 0, lengths[buckets_per_cluster] = {};
    for (int i = 0; i < pools; ++i) { numbers[i] = alive[i] ? next++ : -1; }
    for (int token = 0; token < cluster_size; ++token) {
        int root = token / 4;
        while (parent[root] != root) { root = parent[root]; }
        const int label = numbers[root];
        bucket & target = result[base + label];
        if (lengths[label] == bucket_capacity) { return false; }
        target.members[lengths[label]++] = offset + token;
        for (int d = 0; d < index_dimension; ++d) { target.centroid[d] += keys[token * index_dimension + d]; }
    }
    for (int i = 0; i < buckets_per_cluster; ++i) {
        float * c = result[base + i].centroid;
        const float norm = std::max(std::sqrt(dot(c, c, index_dimension)), 1e-30f);
        for (int d = 0; d < index_dimension; ++d) { c[d] /= norm; }
    }
    return true;
}

bool collect(const attention_args & a, int64_t query, std::vector<int32_t> & rows) {
    rows.clear();
    for (const auto * set : { &a.main, &a.extra }) {
        if (!set->indices) { continue; }
        const int64_t count = set->lengths ? set->lengths[query] : set->capacity;
        if (count < 0 || count > set->capacity) { return false; }
        for (int64_t j = 0; j < count; ++j) {
            const int32_t row = set->indices[query * set->capacity + j];
            if (row == -1) { continue; }
            if (row < -1 || row >= a.k.rows || row > a.positions[query]) { return false; }
            rows.push_back(row);
        }
    }
    return true;
}

} // namespace

bool geometry::valid() const {
    return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
           head_dim > 0 && rope_dim >= 0 && rope_dim < head_dim &&
           (int64_t) kv_heads * (head_dim - rope_dim) <= INT32_MAX;
}

bool projection_maps::valid() const {
    if (!shape.valid() || identity.empty() || q.size() != (size_t) shape.map_rows() * index_dimension || k.size() != q.size()) { return false; }
    for (const auto * map : { &q, &k }) {
        for (const auto x : *map) { if (!std::isfinite(ggml_bf16_to_fp32(x))) { return false; } }
    }
    return true;
}

bool budgets::valid() const { return top_k >= 0 && top_p >= 0 && window >= cluster_size && sink >= 0; }

bucket::bucket() { std::fill(members, members + bucket_capacity, -1); }

const char * cpu_projection_profile() { return "torch-2.13-cpu-bf16-k512-norm8"; }
bool cpu_projection_profile_available(const geometry & shape) {
#if LLAMA_LSA_AVX512
    __builtin_cpu_init();
    return shape.valid() && (shape.map_rows() % 2) == 0 && cpu_has_avx512() && __builtin_cpu_supports("avx512bf16");
#else
    (void) shape;
    return false;
#endif
}

bool cpu_has_avx512() {
#if LLAMA_LSA_AVX512
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    return false;
#endif
}

bool project_queries(const tensor_view & q, const projection_maps & maps, float * output, std::string * error) {
    return project(q, maps, true, output, error);
}
bool project_keys(const tensor_view & k, const projection_maps & maps, float * output, std::string * error) {
    return project(k, maps, false, output, error);
}

void index_cache::reset(uint64_t epoch, const std::string & identity) {
    initialized_ = true; epoch_ = epoch; map_identity_ = identity; keys_.clear(); buckets_.clear(); ambiguous_.clear();
}
size_t index_cache::ambiguous_tie_clusters() const {
    return std::count(ambiguous_.begin(), ambiguous_.end(), uint8_t(1));
}
bool index_cache::matches(uint64_t epoch, const std::string & identity) const {
    return initialized_ && epoch == epoch_ && identity == map_identity_;
}
bool index_cache::append(uint64_t epoch, int64_t first, const float * keys, int64_t count, std::string * error) {
    if (!initialized_ || epoch != epoch_ || first != size() || count < 0 || count > INT32_MAX - first || (count && !keys)) {
        return fail(error, "cache append epoch, position or range mismatch");
    }
    for (int64_t i = 0; i < count * index_dimension; ++i) {
        if (!std::isfinite(keys[i]) || rounded(keys[i]) != keys[i]) { return fail(error, "index key must be finite and BF16-rounded"); }
    }
    const size_t old_keys = keys_.size(), old_buckets = buckets_.size(), old_clusters = ambiguous_.size();
    if (count) { keys_.insert(keys_.end(), keys, keys + count * index_dimension); }
    for (int64_t offset = old_buckets / buckets_per_cluster * cluster_size; offset + cluster_size <= size(); offset += cluster_size) {
        bool ambiguous;
        if (!cluster(keys_.data() + offset * index_dimension, (int32_t) offset, buckets_, ambiguous)) {
            keys_.resize(old_keys); buckets_.resize(old_buckets); ambiguous_.resize(old_clusters); return fail(error, "Ward clustering failed");
        }
        ambiguous_.push_back(ambiguous);
    }
    return true;
}
bool index_cache::truncate(uint64_t epoch, int64_t count, std::string * error) {
    if (!initialized_ || epoch != epoch_ || count < 0 || count > size()) { return fail(error, "cache rollback epoch or range mismatch"); }
    keys_.resize(count * index_dimension);
    buckets_.resize(count / cluster_size * buckets_per_cluster);
    ambiguous_.resize(count / cluster_size);
    return true;
}
bool index_cache::select(uint64_t epoch, const float * query, int32_t position, const budgets & b,
                         std::vector<int32_t> & rows, std::string * error) const {
    if (!initialized_ || epoch != epoch_ || !query || position < 0 || position >= size() || !b.valid()) {
        return fail(error, "selector epoch, position or budgets invalid");
    }
    for (int d = 0; d < index_dimension; ++d) { if (!std::isfinite(query[d])) { return fail(error, "nonfinite index query"); } }
    const size_t visible = std::min((size_t) (position / cluster_size * buckets_per_cluster), buckets_.size());
    if (std::find(ambiguous_.begin(), ambiguous_.begin() + visible / buckets_per_cluster, uint8_t(1)) !=
        ambiguous_.begin() + visible / buckets_per_cluster) { return fail(error, "author Ward final-round tie order ambiguous; use dense fallback"); }
    std::vector<std::pair<float, int32_t>> scores;
    scores.reserve(visible);
    std::vector<float> score_values(visible);
#if LLAMA_LSA_AVX512
    if (cpu_has_avx512()) { score_avx512(query, buckets_.data(), visible, score_values.data()); } else
#endif
    { for (size_t i = 0; i < visible; ++i) { score_values[i] = dot(query, buckets_[i].centroid, index_dimension); } }
    for (size_t i = 0; i < visible; ++i) { scores.emplace_back(score_values[i], (int32_t) i); }
    const auto cmp = [](const std::pair<float, int32_t> & x, const std::pair<float, int32_t> & y) {
        return x.first != y.first ? x.first > y.first : x.second < y.second;
    };
    const size_t picked = std::min(visible, (size_t) b.top_p);
    std::partial_sort(scores.begin(), scores.begin() + picked, scores.end(), cmp);
    const int32_t start = std::max(0, position + 1 - b.window);
    rows.clear();
    for (size_t i = 0; i < picked && rows.size() < (size_t) b.top_k; ++i) {
        for (int32_t member : buckets_[scores[i].second].members) {
            if (member >= b.sink && member < start) {
                rows.push_back(member);
                if (rows.size() == (size_t) b.top_k) { break; }
            }
        }
    }
    for (int32_t i = start; i <= position; ++i) { rows.push_back(i); }
    for (int32_t i = 0; i < std::min(b.sink, start); ++i) { rows.push_back(i); }
    return true;
}

bool selected_attention_cpu(const attention_args & a, attention_stats * stats, std::string * error) {
    if (stats) { *stats = {}; }
    if (!valid_view(a.q) || !valid_view(a.k) || !valid_view(a.v) || a.q.type != GGML_TYPE_F32 ||
        a.q.dim != a.k.dim || a.k.rows != a.v.rows || a.k.heads != a.v.heads || a.q.heads % a.k.heads ||
        !a.positions || (a.q.rows && !a.output) || !std::isfinite(a.scale) || !std::isfinite(a.softcap) || a.softcap < 0) {
        return fail(error, "unsupported attention layout or geometry");
    }
    if ((uint64_t) a.q.rows > SIZE_MAX / sizeof(float) / a.q.heads / a.v.dim) { return fail(error, "attention output size overflow"); }
    for (const auto * set : { &a.main, &a.extra }) {
        if (set->capacity < 0 || (!set->indices && set->capacity) ||
            (set->capacity && a.q.rows > INT64_MAX / set->capacity)) { return fail(error, "invalid index-set capacity"); }
    }
    const auto * kt = ggml_get_type_traits_cpu(a.k.type);
    const auto * vt = ggml_get_type_traits(a.v.type);
    if (!kt || !kt->vec_dot || (a.v.type != GGML_TYPE_F32 && !vt->to_float)) { return fail(error, "unsupported attention KV dtype"); }
    const auto * qt = ggml_get_type_traits_cpu(kt->vec_dot_type);
    if (!qt || !qt->from_float || a.q.dim % ggml_blck_size(kt->vec_dot_type)) { return fail(error, "unsupported query dot dtype"); }
    std::vector<int32_t> rows;
    // Validate every row before accessing any KV data or writing output.
    for (int64_t iq = 0; iq < a.q.rows; ++iq) {
        if (a.positions[iq] < 0 || a.positions[iq] >= a.k.rows || !collect(a, iq, rows)) { return fail(error, "invalid or noncausal selected row"); }
    }
    if (a.attn_sink) {
        for (int h = 0; h < a.q.heads; ++h) {
            if (!std::isfinite(a.attn_sink[h]) && a.attn_sink[h] != -INFINITY) { return fail(error, "invalid attention sink"); }
        }
    }
    for (int64_t iq = 0; iq < a.q.rows; ++iq) {
        for (int h = 0; h < a.q.heads; ++h) {
            const float * q = reinterpret_cast<const float *>(row_data(a.q, iq, h));
            for (int d = 0; d < a.q.dim; ++d) {
                if (!std::isfinite(q[d])) { return fail(error, "nonfinite attention query"); }
            }
        }
    }
    const size_t qbytes = ggml_row_size(kt->vec_dot_type, a.q.dim);
    std::vector<uint64_t> converted((qbytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    std::vector<float> logits, value(a.v.dim);
    std::vector<float> result(a.q.rows * a.q.heads * (int64_t) a.v.dim);
    const int group = a.q.heads / a.k.heads;
    for (int64_t iq = 0; iq < a.q.rows; ++iq) {
        collect(a, iq, rows);
        logits.resize(rows.size());
        for (int head = 0; head < a.q.heads; ++head) {
            float * out = result.data() + (iq * a.q.heads + head) * a.v.dim;
            std::fill(out, out + a.v.dim, 0.0f);
            if (rows.empty()) { continue; }
            const float * q = reinterpret_cast<const float *>(row_data(a.q, iq, head));
            qt->from_float(q, converted.data(), a.q.dim);
            float maximum = -INFINITY;
            for (size_t j = 0; j < rows.size(); ++j) {
                float score;
                kt->vec_dot(a.q.dim, &score, 0, row_data(a.k, rows[j], head / group), 0, converted.data(), 0, 1);
                if (stats) { ++stats->key_rows; }
                if (!std::isfinite(score)) { return fail(error, "nonfinite selected key dot product"); }
                score *= a.scale;
                if (a.softcap) { score = a.softcap * std::tanh(score / a.softcap); }
                if (!std::isfinite(score)) { return fail(error, "nonfinite attention score"); }
                logits[j] = score; maximum = std::fmax(maximum, score);
            }
            if (a.attn_sink) { maximum = std::fmax(maximum, a.attn_sink[head]); }
            float denominator = 0.0f;
            for (size_t j = 0; j < rows.size(); ++j) {
                const float probability = std::exp(logits[j] - maximum);
                denominator += probability;
                read_float_row(a.v, rows[j], head / group, value.data());
                if (stats) { ++stats->value_rows; }
                for (float x : value) { if (!std::isfinite(x)) { return fail(error, "nonfinite selected value"); } }
                for (int d = 0; d < a.v.dim; ++d) { out[d] += probability * value[d]; }
            }
            if (a.attn_sink) { denominator += std::exp(a.attn_sink[head] - maximum); }
            for (int d = 0; d < a.v.dim; ++d) {
                out[d] /= denominator;
                if (!std::isfinite(out[d])) { return fail(error, "nonfinite attention output"); }
            }
        }
    }
    if (!result.empty()) { std::memcpy(a.output, result.data(), result.size() * sizeof(float)); }
    return true;
}

} // namespace llama_lsa
