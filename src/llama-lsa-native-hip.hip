#include "llama-lsa-native-hip.h"

#if !defined(GGML_USE_HIP)
#error "llama-lsa-native-hip.hip must be built only by the HIP backend"
#endif

// Reuse ggml's HIP-compatible Q8_0 layout/dequantizer; no private quant format.
#include "ggml-cuda/dequantize.cuh"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace llama_lsa {

struct hip_context {
    geometry shape;
    hip_limits limits;
    std::string identity;
    hipStream_t stream = nullptr;
    int device = -1;
    bool initialized = false;
    bool healthy = true;
    uint64_t epoch = 0;
    int64_t count = 0;
    ggml_bf16_t * qmap = nullptr;
    ggml_bf16_t * kmap = nullptr;
    float * keys = nullptr;
    bucket * buckets = nullptr;
    uint8_t * ambiguous = nullptr;
    std::vector<uint8_t> host_ambiguous;
    float * bucket_scores = nullptr;
    float * attention_scores = nullptr;
    int * fault = nullptr;
    attention_stats * stats = nullptr;
};

namespace {
constexpr int pools = cluster_size / 4;
enum fault_code { no_fault, bad_position, bad_length, bad_index, bad_number, ward_failed, ambiguous_ward };

// Fixed-128 device spelling of the pinned CPU std::sort (libstdc++ introsort).
// Values remain in masked_cost; only their associated original indices move.
__device__ bool rank_less(const float * masked_cost, int a, int b) {
    return masked_cost[a] < masked_cost[b];
}
__device__ void rank_swap(int & a, int & b) { const int value = a; a = b; b = value; }
__device__ void rank_heap_push(int * order, const float * cost, int hole, int top, int value) {
    int parent = (hole - 1) / 2;
    while (hole > top && rank_less(cost, order[parent], value)) {
        order[hole] = order[parent]; hole = parent; parent = (hole - 1) / 2;
    }
    order[hole] = value;
}
__device__ void rank_heap_adjust(int * order, const float * cost, int hole, int length, int value) {
    const int top = hole;
    int child = hole;
    while (child < (length - 1) / 2) {
        child = 2 * (child + 1);
        if (rank_less(cost, order[child], order[child - 1])) --child;
        order[hole] = order[child]; hole = child;
    }
    if ((length & 1) == 0 && child == (length - 2) / 2) {
        child = 2 * (child + 1);
        order[hole] = order[child - 1]; hole = child - 1;
    }
    rank_heap_push(order, cost, hole, top, value);
}
__device__ void rank_heap_sort(int * order, const float * cost, int lo, int hi) {
    int * data = order + lo;
    const int length = hi - lo;
    for (int parent = (length - 2) / 2; parent >= 0; --parent) {
        const int value = data[parent];
        rank_heap_adjust(data, cost, parent, length, value);
    }
    for (int end = length; end > 1; --end) {
        const int value = data[end - 1];
        data[end - 1] = data[0];
        rank_heap_adjust(data, cost, 0, end - 1, value);
    }
}
__device__ int rank_partition(int * order, const float * cost, int lo, int hi) {
    const int a = lo + 1, b = lo + (hi - lo) / 2, c = hi - 1;
    if (rank_less(cost, order[a], order[b])) {
        if (rank_less(cost, order[b], order[c])) rank_swap(order[lo], order[b]);
        else if (rank_less(cost, order[a], order[c])) rank_swap(order[lo], order[c]);
        else rank_swap(order[lo], order[a]);
    } else if (rank_less(cost, order[a], order[c])) rank_swap(order[lo], order[a]);
    else if (rank_less(cost, order[b], order[c])) rank_swap(order[lo], order[c]);
    else rank_swap(order[lo], order[b]);
    int first = lo + 1, last = hi;
    while (true) {
        while (rank_less(cost, order[first], order[lo])) ++first;
        --last;
        while (rank_less(cost, order[lo], order[last])) --last;
        if (first >= last) return first;
        rank_swap(order[first], order[last]); ++first;
    }
}
__device__ void rank_insert(int * order, const float * cost, int position) {
    const int value = order[position];
    int at = position;
    do {
        --at;
        if (!rank_less(cost, value, order[at])) break;
        order[at + 1] = order[at];
    } while (at > 0);
    order[at + 1] = value;
}
__device__ void rank_sort_128(int * order, const float * cost) {
    int stack_lo[16], stack_hi[16], stack_depth[16], pending = 1;
    stack_lo[0] = 0; stack_hi[0] = pools; stack_depth[0] = 14;
    while (pending) {
        --pending;
        int lo = stack_lo[pending], hi = stack_hi[pending], depth = stack_depth[pending];
        while (hi - lo > 16) {
            if (depth == 0) { rank_heap_sort(order, cost, lo, hi); break; }
            --depth;
            const int cut = rank_partition(order, cost, lo, hi);
            stack_lo[pending] = cut; stack_hi[pending] = hi; stack_depth[pending] = depth;
            ++pending;
            hi = cut;
        }
    }
    // libstdc++ first sorts 16 with a guarded insertion, then inserts the rest
    // against its smallest-element sentinel.
    for (int i = 1; i < 16; ++i) {
        if (rank_less(cost, order[i], order[0])) {
            const int value = order[i];
            for (int at = i; at > 0; --at) order[at] = order[at - 1];
            order[0] = value;
        } else rank_insert(order, cost, i);
    }
    for (int i = 16; i < pools; ++i) rank_insert(order, cost, i);
}

hip_result fail(hip_result result, std::string * error, const char * message) {
    if (error) { *error = message; }
    return result;
}

bool product(size_t a, size_t b, size_t & result) {
    if (b && a > SIZE_MAX / b) { return false; }
    result = a * b; return true;
}

bool add(size_t a, size_t b, size_t & result) {
    if (b > SIZE_MAX - a) { return false; }
    result = a + b; return true;
}

bool floating(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

bool valid_view(const tensor_view & v, bool quantized) {
    if ((!floating(v.type) && !(quantized && v.type == GGML_TYPE_Q8_0)) ||
        v.rows < 0 || v.rows > INT32_MAX || v.heads <= 0 || v.dim <= 0 ||
        (v.rows && !v.data) || (v.type == GGML_TYPE_Q8_0 && v.dim % QK8_0)) { return false; }
    const size_t bytes = ggml_row_size(v.type, v.dim);
    if (v.head_stride < bytes || v.row_stride < bytes ||
        (v.heads > 1 && v.head_stride > (SIZE_MAX - bytes) / (v.heads - 1))) { return false; }
    const size_t end = (v.heads - 1) * v.head_stride + bytes;
    if (v.rows > 1 && uint64_t(v.rows - 1) > (SIZE_MAX - end) / v.row_stride) { return false; }
    if (v.heads > 1 && v.row_stride < end && uint64_t(v.rows) > v.head_stride / v.row_stride) { return false; }
    const size_t alignment = v.type == GGML_TYPE_F32 ? 4 : 2;
    return reinterpret_cast<uintptr_t>(v.data) % alignment == 0 &&
           v.row_stride % alignment == 0 && v.head_stride % alignment == 0;
}

hip_result ready(hip_context * c, std::string * error, bool epoch_required = false, uint64_t epoch = 0) {
    if (!c) { return fail(hip_result::invalid_argument, error, "null LSA HIP context"); }
    int device = -1;
    if (!c->healthy || hipGetDevice(&device) != hipSuccess || device != c->device) {
        return fail(hip_result::device_error, error, "LSA HIP device changed or context failed; recreate context");
    }
    if (epoch_required && (!c->initialized || c->epoch != epoch)) {
        return fail(hip_result::invalid_argument, error, "LSA HIP epoch mismatch");
    }
    if (error) { error->clear(); }
    return hip_result::success;
}

hip_result device_failure(hip_context * c, hipError_t status, std::string * error) {
    c->healthy = false;
    if (error) { *error = std::string("LSA HIP: ") + hipGetErrorString(status); }
    return hip_result::device_error;
}

hip_result begin(hip_context * c, std::string * error) {
    const hipError_t rc = hipMemsetAsync(c->fault, 0, sizeof(int), c->stream);
    return rc == hipSuccess ? hip_result::success : device_failure(c, rc, error);
}

hip_result finish(hip_context * c, std::string * error) {
    hipError_t rc = hipGetLastError();
    if (rc != hipSuccess) { return device_failure(c, rc, error); }
    int fault = no_fault;
    rc = hipMemcpyAsync(&fault, c->fault, sizeof(fault), hipMemcpyDeviceToHost, c->stream);
    if (rc == hipSuccess) { rc = hipStreamSynchronize(c->stream); }
    if (rc != hipSuccess) { return device_failure(c, rc, error); }
    switch (fault) {
        case no_fault: return hip_result::success;
        case bad_position: return fail(hip_result::invalid_argument, error, "LSA HIP invalid causal position");
        case bad_length: return fail(hip_result::invalid_argument, error, "LSA HIP index length outside capacity");
        case bad_index: return fail(hip_result::invalid_argument, error, "LSA HIP selected index outside KV or causal bound");
        case bad_number: return fail(hip_result::invalid_argument, error, "LSA HIP nonfinite input or intermediate");
        case ambiguous_ward: return fail(hip_result::index_error, error, "author Ward final-round tie order ambiguous; use dense fallback");
        default: return fail(hip_result::index_error, error, "LSA HIP Ward clustering failed; ignore index output");
    }
}

__device__ float bf16_float(ggml_bf16_t x) {
    union { uint32_t u; float f; } bits;
    bits.u = uint32_t(x.bits) << 16;
    return bits.f;
}

__device__ float rounded(float x) {
    union { uint32_t u; float f; } bits;
    bits.f = x;
    // RNE, including NaN preservation rather than rounding a NaN into infinity.
    if ((bits.u & 0x7fffffffU) > 0x7f800000U) { bits.u |= 0x00400000U; }
    else { bits.u += 0x7fffU + ((bits.u >> 16) & 1U); }
    bits.u &= 0xffff0000U;
    return bits.f;
}

__device__ const char * row_data(tensor_view v, int64_t row, int head) {
    return static_cast<const char *>(v.data) + row * v.row_stride + head * v.head_stride;
}

__device__ float element(tensor_view v, int64_t row, int head, int d) {
    const char * p = row_data(v, row, head);
    switch (v.type) {
        case GGML_TYPE_F32: return reinterpret_cast<const float *>(p)[d];
        case GGML_TYPE_F16: return __half2float(reinterpret_cast<const half *>(p)[d]);
        case GGML_TYPE_BF16: return bf16_float(reinterpret_cast<const ggml_bf16_t *>(p)[d]);
        case GGML_TYPE_Q8_0: {
            float2 pair;
            dequantize_q8_0(p, d / QK8_0, (d % QK8_0) & ~1, pair);
            return d & 1 ? pair.y : pair.x;
        }
        default: return NAN; // host metadata validation rejects this
    }
}

__device__ void report(int * fault, int code) { atomicCAS(fault, no_fault, code); }

// Validate logical data separately so an invalid input cannot partially publish
// projected keys or queries. RoPE channels are deliberately never consulted.
__global__ void validate_logical(tensor_view v, int rope, int * fault) {
    const int64_t total = v.rows * int64_t(v.heads) * (v.dim - rope);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
        const int d = i % (v.dim - rope) + rope;
        const int64_t h = i / (v.dim - rope);
        if (!isfinite(element(v, h / v.heads, h % v.heads, d))) { report(fault, bad_number); }
    }
}

__global__ void project(tensor_view v, geometry g, const ggml_bf16_t * map,
                        bool query, float * output, int * fault) {
    const int64_t row = blockIdx.x;
    const int j = threadIdx.x;
    __shared__ float values[index_dimension];
    __shared__ float norm;
    const int width = g.head_dim - g.rope_dim;
    const int group = g.q_heads / g.kv_heads;
    const int map_rows = g.kv_heads * width;
    float sum = 0.0f;
    // Match the pinned AVX512 BF16 dot: each 512-K block has a fresh FP32
    // accumulator, and VDPBF16PS applies the high BF16 pair before the low.
    for (int base = 0; base < map_rows; base += 512) {
        float acc = 0.0f;
        for (int k = base; k < min(base + 512, map_rows); k += 2) {
            const int kv = k / width;
            const int d = k % width;
            const int first = query ? kv * group : kv;
            const int end = query ? first + group : first + 1;
            float low = 0.0f;
            float high = 0.0f;
            for (int h = first; h < end; ++h) {
                low = __ocml_add_rte_f32(low, element(v, row, h, g.rope_dim + d));
                high = __ocml_add_rte_f32(high, element(v, row, h, g.rope_dim + d + 1));
            }
            const float x_high = rounded(high);
            const float x_low = rounded(low);
            const float w_high = bf16_float(map[(k + 1) * index_dimension + j]);
            const float w_low = bf16_float(map[k * index_dimension + j]);
            acc = __ocml_fma_rte_f32(x_high, w_high, acc);
            acc = __ocml_fma_rte_f32(x_low, w_low, acc);
        }
        sum = __ocml_add_rte_f32(sum, acc);
    }
    values[j] = rounded(sum);
    if (!isfinite(values[j])) { report(fault, bad_number); }
    __syncthreads();
    if (j == 0) {
        volatile float lanes[8] = {};
        for (int d = 0; d < index_dimension; ++d) {
            const float product = __ocml_mul_rte_f32(values[d], values[d]);
            lanes[d % 8] = __ocml_add_rte_f32(lanes[d % 8], product);
        }
        float sq = lanes[0];
        for (int lane = 1; lane < 8; ++lane) sq = __ocml_add_rte_f32(sq, lanes[lane]);
        if (!isfinite(sq)) { report(fault, bad_number); }
        norm = fmaxf(__ocml_sqrt_f32(sq), 1e-12f);
        if (!isfinite(norm)) { report(fault, bad_number); }
    }
    __syncthreads();
    const float normalized = values[j] / norm;
    output[row * index_dimension + j] = query ? normalized : rounded(normalized);
}

// One cooperative CTA per immutable complete block. Shared state is under 40KiB
// on gfx90a: 128 pools of four keys, no 512x512 matrix or all-KV attention mask.
// Each lane searches a pool's partner. Thread zero chooses only the mutually
// cheapest disjoint pairs; equal costs retain the lower kept-pool index.
__global__ void build_buckets(const float * keys, int64_t first_cluster, bucket * out, uint8_t * ambiguous, int * fault) {
    __shared__ float sums[pools][index_dimension], squares[pools], norms[pools], costs[pools];
    __shared__ int size[pools], parent[pools], alive[pools], best[pools], merge[pools], number[pools];
    __shared__ int count, nmerge;
    const int t = threadIdx.x;
    const int64_t cluster = first_cluster + blockIdx.x;
    const int32_t offset = int32_t(cluster * cluster_size);
    const float * k = keys + int64_t(offset) * index_dimension;
    size[t] = 4; parent[t] = t; alive[t] = 1;
    for (int d = 0; d < index_dimension; ++d) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) { sum += k[(t * 4 + i) * index_dimension + d]; }
        sums[t][d] = sum;
    }
    if (t == 0) { count = pools; ambiguous[cluster] = 0; }
    __syncthreads();
    for (int round = 0; round < 256 && count > buckets_per_cluster; ++round) {
        float sq = 0.0f;
        for (int d = 0; d < index_dimension; ++d) { sq += sums[t][d] * sums[t][d]; }
        squares[t] = sq; norms[t] = sqrtf(fmaxf(0.0f, sq));
        __syncthreads();
        float cost = INFINITY;
        int partner = -1;
        if (alive[t]) {
            for (int j = 0; j < pools; ++j) {
                if (t == j || !alive[j] || size[t] + size[j] > bucket_capacity) { continue; }
                float gram = 0.0f;
                for (int d = 0; d < index_dimension; ++d) { gram += sums[t][d] * sums[j][d]; }
                const float candidate = norms[t] + norms[j] - sqrtf(fmaxf(0.0f, squares[t] + squares[j] + 2.0f * gram));
                if (candidate < cost) { cost = candidate; partner = j; }
            }
        }
        costs[t] = cost; best[t] = partner;
        __syncthreads();
        if (t == 0) {
            // The pinned author ranks all 128 masked costs, including +inf
            // non-mutual entries, using the CPU unstable argsort profile.
            for (int i = 0; i < pools; ++i) {
                if (best[i] <= i || best[best[i]] != i) costs[i] = INFINITY;
                merge[i] = i;
            }
            const int cutoff = count - buckets_per_cluster;
            rank_sort_128(merge, costs);
            nmerge = 0;
            while (nmerge < cutoff && isfinite(costs[merge[nmerge]])) ++nmerge;
            if (!nmerge) { report(fault, ward_failed); }
        }
        __syncthreads();
        if (!nmerge) { return; }
        if (t < nmerge) {
            const int i = merge[t], j = best[i];
            for (int d = 0; d < index_dimension; ++d) { sums[i][d] += sums[j][d]; }
            size[i] += size[j]; alive[j] = 0; parent[j] = i;
        }
        __syncthreads();
        if (t == 0) { count -= nmerge; }
        __syncthreads();
    }
    if (count != buckets_per_cluster) { if (t == 0) { report(fault, ward_failed); } return; }
    if (t == 0) {
        int n = 0;
        for (int i = 0; i < pools; ++i) { number[i] = alive[i] ? n++ : -1; }
    }
    __syncthreads();
    int root = t;
    while (parent[root] != root) { root = parent[root]; }
    best[t] = number[root];
    __syncthreads();
    if (t < buckets_per_cluster) {
        bucket & b = out[cluster * buckets_per_cluster + t];
        for (int d = 0; d < index_dimension; ++d) { b.centroid[d] = 0.0f; }
        for (int i = 0; i < bucket_capacity; ++i) { b.members[i] = -1; }
        int length = 0;
        // Recompute from original rows, ordered by token, not merge history.
        for (int token = 0; token < cluster_size; ++token) {
            if (best[token / 4] != t) { continue; }
            if (length == bucket_capacity) { report(fault, ward_failed); return; }
            b.members[length++] = offset + token;
            for (int d = 0; d < index_dimension; ++d) { b.centroid[d] += k[token * index_dimension + d]; }
        }
        float sq = 0.0f;
        for (int d = 0; d < index_dimension; ++d) { sq += b.centroid[d] * b.centroid[d]; }
        const float norm = fmaxf(sqrtf(sq), 1e-30f);
        for (int d = 0; d < index_dimension; ++d) { b.centroid[d] /= norm; }
    }
}

__global__ void validate_queries(const float * queries, int64_t nq, const int32_t * positions,
                                 int64_t keys, const uint8_t * ambiguous, int * fault) {
    const int64_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) { return; }
    if (positions[q] < 0 || positions[q] >= keys) { report(fault, bad_position); }
    for (int64_t block = 0; block < min(int64_t(positions[q] / cluster_size), keys / cluster_size); ++block) {
        if (ambiguous[block]) { report(fault, ambiguous_ward); }
    }
    for (int d = 0; d < index_dimension; ++d) {
        if (!isfinite(queries[q * index_dimension + d])) { report(fault, bad_number); }
    }
}

__global__ void score_buckets(const float * queries, const int32_t * positions, const bucket * buckets,
                              int64_t nb, int64_t nq, float * scores, int * fault) {
    const int64_t total = nq * nb;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t q = i / nb, b = i % nb;
        float score = -INFINITY;
        if (b < int64_t(positions[q] / cluster_size) * buckets_per_cluster) {
            score = 0.0f;
            for (int d = 0; d < index_dimension; ++d) { score += queries[q * index_dimension + d] * buckets[b].centroid[d]; }
            if (!isfinite(score)) { report(fault, bad_number); }
        }
        scores[i] = score;
    }
}

__global__ void select_rows(float * scores, const bucket * buckets, int64_t nb,
                            const int32_t * positions, budgets budget, int64_t capacity,
                            int32_t * indices, int32_t * lengths) {
    __shared__ float lane_score[256];
    __shared__ int lane_bucket[256];
    __shared__ int count, chosen;
    const int t = threadIdx.x;
    const int64_t q = blockIdx.x;
    const int position = positions[q];
    const int start = max(0, position + 1 - budget.window);
    const int64_t visible = min(nb, int64_t(position / cluster_size) * buckets_per_cluster);
    int32_t * out = indices + q * capacity;
    for (int64_t i = t; i < capacity; i += blockDim.x) { out[i] = -1; }
    if (!t) { count = 0; }
    __syncthreads();
    for (int p = 0; p < min(int64_t(budget.top_p), visible) && count < budget.top_k; ++p) {
        float best = -INFINITY;
        int id = INT32_MAX;
        for (int64_t b = t; b < visible; b += blockDim.x) {
            const float s = scores[q * nb + b];
            if (s > best || (s == best && b < id)) { best = s; id = int(b); }
        }
        lane_score[t] = best; lane_bucket[t] = id;
        __syncthreads();
        for (int step = 128; step; step >>= 1) {
            if (t < step) {
                const float s = lane_score[t + step];
                const int id2 = lane_bucket[t + step];
                if (s > lane_score[t] || (s == lane_score[t] && id2 < lane_bucket[t])) {
                    lane_score[t] = s; lane_bucket[t] = id2;
                }
            }
            __syncthreads();
        }
        if (!t) {
            chosen = lane_bucket[0];
            if (chosen < visible) {
                scores[q * nb + chosen] = -INFINITY;
                // Mark picked separately from -infinity scores: inputs are
                // finite, so a real score cannot equal -infinity.
                for (int m = 0; m < bucket_capacity && count < budget.top_k; ++m) {
                    const int row = buckets[chosen].members[m];
                    if (row >= budget.sink && row < start) { out[count++] = row; }
                }
            }
        }
        __syncthreads();
    }
    if (!t) {
        for (int row = start; row <= position; ++row) { out[count++] = row; }
        for (int row = 0; row < min(budget.sink, start); ++row) { out[count++] = row; }
        lengths[q] = count;
    }
}

__device__ int64_t live(index_set s, int64_t q) { return s.indices ? (s.lengths ? s.lengths[q] : s.capacity) : 0; }

__device__ int slot_row(attention_args a, int64_t q, int64_t slot) {
    const index_set s = slot < a.main.capacity ? a.main : a.extra;
    const int64_t j = slot < a.main.capacity ? slot : slot - a.main.capacity;
    return s.indices && j < live(s, q) ? s.indices[q * s.capacity + j] : -1;
}

__global__ void validate_attention(attention_args a, int * fault) {
    const int64_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= a.q.rows) { return; }
    if (a.positions[q] < 0 || a.positions[q] >= a.k.rows) { report(fault, bad_position); return; }
    for (int set = 0; set < 2; ++set) {
        const index_set s = set ? a.extra : a.main;
        const int64_t n = live(s, q);
        if (n < 0 || n > s.capacity) { report(fault, bad_length); return; }
        for (int64_t j = 0; j < n; ++j) {
            const int row = s.indices[q * s.capacity + j];
            if (row < -1 || row >= a.k.rows || row > a.positions[q]) { report(fault, bad_index); }
        }
    }
    for (int h = 0; h < a.q.heads; ++h) {
        if (a.attn_sink && !isfinite(a.attn_sink[h]) && a.attn_sink[h] != -INFINITY) { report(fault, bad_number); }
        for (int d = 0; d < a.q.dim; ++d) {
            if (!isfinite(element(a.q, q, h, d))) { report(fault, bad_number); }
        }
    }
}

// Dot products keep dimension order and never touch an unselected K row.
__global__ void attention_logits(attention_args a, int64_t slots, float * scores,
                                 attention_stats * stats, int * fault) {
    const int64_t qh = blockIdx.x;
    const int64_t q = qh / a.q.heads;
    const int h = qh % a.q.heads;
    const int kh = h / (a.q.heads / a.k.heads);
    for (int64_t j = threadIdx.x; j < slots; j += blockDim.x) {
        const int row = slot_row(a, q, j);
        float logit = -INFINITY;
        if (row >= 0) {
            float dot = 0.0f;
            for (int d = 0; d < a.q.dim; ++d) { dot += element(a.q, q, h, d) * element(a.k, row, kh, d); }
            if (!isfinite(dot)) { report(fault, bad_number); }
            logit = dot * a.scale;
            if (!isfinite(logit)) { report(fault, bad_number); }
            if (a.softcap > 0.0f) { logit = a.softcap * tanhf(logit / a.softcap); }
            if (!isfinite(logit)) { report(fault, bad_number); }
        }
        scores[qh * slots + j] = logit;
    }
    if (stats && !threadIdx.x) {
        unsigned long long count = 0;
        for (int64_t j = 0; j < slots; ++j) { count += slot_row(a, q, j) >= 0; }
        atomicAdd(reinterpret_cast<unsigned long long *>(&stats->key_rows), count);
    }
}

// One lane per output channel; thread zero computes m and denominator in the
// normative main-then-extra slot order. The second pass reads only selected V.
__global__ void attention_values(attention_args a, int64_t slots, float * scores,
                                 attention_stats * stats, int * fault) {
    const int64_t qh = blockIdx.x, q = qh / a.q.heads;
    const int h = qh % a.q.heads, kh = h / (a.q.heads / a.k.heads);
    __shared__ float denom;
    __shared__ int any;
    if (!threadIdx.x) {
        float m = -INFINITY;
        any = 0;
        for (int64_t j = 0; j < slots; ++j) {
            if (slot_row(a, q, j) >= 0) { m = fmaxf(m, scores[qh * slots + j]); ++any; }
        }
        if (a.attn_sink) { m = fmaxf(m, a.attn_sink[h]); }
        denom = 0.0f;
        for (int64_t j = 0; j < slots; ++j) {
            const float p = slot_row(a, q, j) >= 0 ? expf(scores[qh * slots + j] - m) : 0.0f;
            scores[qh * slots + j] = p;
            denom += p;
        }
        if (a.attn_sink && any) { denom += expf(a.attn_sink[h] - m); }
        if (any && (!isfinite(denom) || denom <= 0.0f)) { report(fault, bad_number); }
        if (stats) { atomicAdd(reinterpret_cast<unsigned long long *>(&stats->value_rows), static_cast<unsigned long long>(any)); }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < a.v.dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int64_t j = 0; j < slots; ++j) {
            const int row = slot_row(a, q, j);
            if (row < 0) { continue; }
            const float value = element(a.v, row, kh, d);
            if (!isfinite(value)) { report(fault, bad_number); }
            sum += scores[qh * slots + j] * value;
        }
        if (!isfinite(sum)) { report(fault, bad_number); }
        const float result = any ? sum / denom : 0.0f;
        if (!isfinite(result)) { report(fault, bad_number); }
        a.output[qh * a.v.dim + d] = result;
    }
}

unsigned grid(int64_t count) { return unsigned(std::min<int64_t>((count + 255) / 256, 65535)); }

hip_result validate_projection(hip_context * c, const tensor_view & v, bool query, std::string * error) {
    const hip_result rc = ready(c, error);
    if (rc != hip_result::success) { return rc; }
    if (!valid_view(v, false) || v.dim != c->shape.head_dim ||
        v.heads != (query ? c->shape.q_heads : c->shape.kv_heads) ||
        v.rows > (query ? c->limits.max_queries : c->limits.max_keys)) {
        return fail(hip_result::unsupported, error, "LSA HIP unsupported logical projection view");
    }
    return hip_result::success;
}

hip_result project_into(hip_context * c, const tensor_view & v, bool query, float * output, std::string * error) {
    hip_result rc = begin(c, error);
    if (rc != hip_result::success || !v.rows) { return rc; }
    validate_logical<<<grid(v.rows * v.heads * (v.dim - c->shape.rope_dim)), 256, 0, c->stream>>>(v, c->shape.rope_dim, c->fault);
    rc = finish(c, error);
    if (rc != hip_result::success) { return rc; }
    project<<<unsigned(v.rows), index_dimension, 0, c->stream>>>(v, c->shape, query ? c->qmap : c->kmap, query, output, c->fault);
    return finish(c, error);
}
} // namespace

hip_result hip_memory_requirements(const geometry & shape, const hip_limits & limits, hip_memory * out, std::string * error) {
    if (!out || !shape.valid() || limits.max_keys <= 0 || limits.max_keys > INT32_MAX - 1 ||
        limits.max_queries <= 0 || limits.max_queries > UINT32_MAX / shape.q_heads ||
        limits.max_slots < 0 || limits.max_slots > INT32_MAX) {
        return fail(hip_result::invalid_argument, error, "LSA HIP invalid memory request");
    }
    hip_memory bytes;
    size_t keys, buckets, maps, bucket_scores, attention_scores, temporary;
    const size_t clusters = size_t(limits.max_keys / cluster_size);
    const size_t nb = clusters * buckets_per_cluster;
    if (!product(size_t(limits.max_keys), index_dimension * sizeof(float), keys) ||
        !product(nb, sizeof(bucket), buckets) ||
        !product(size_t(shape.map_rows()), 2 * index_dimension * sizeof(ggml_bf16_t), maps) ||
        !product(size_t(limits.max_queries), nb, temporary) || !product(temporary, sizeof(float), bucket_scores) ||
        !product(size_t(limits.max_queries), shape.q_heads, temporary) ||
        !product(temporary, size_t(limits.max_slots), attention_scores) ||
        !product(attention_scores, sizeof(float), attention_scores) ||
        !add(keys, buckets, bytes.persistent_bytes) || !add(bytes.persistent_bytes, maps, bytes.persistent_bytes) ||
        !add(bytes.persistent_bytes, clusters, bytes.persistent_bytes) ||
        !add(bucket_scores, attention_scores, bytes.workspace_bytes) ||
        !add(bytes.workspace_bytes, sizeof(int) + sizeof(attention_stats), bytes.workspace_bytes) ||
        !add(bytes.persistent_bytes, bytes.workspace_bytes, bytes.total_device_bytes)) {
        return fail(hip_result::invalid_argument, error, "LSA HIP memory requirement overflow");
    }
    *out = bytes;
    if (limits.max_device_bytes && bytes.total_device_bytes > limits.max_device_bytes) {
        return fail(hip_result::resource_error, error, "LSA HIP memory request exceeds caller admission cap");
    }
    if (error) { error->clear(); }
    return hip_result::success;
}

hip_result hip_create(const projection_maps & maps, const hip_limits & limits, void * stream,
                      hip_context ** context, std::string * error) {
    if (!context) { return fail(hip_result::invalid_argument, error, "null LSA HIP context output"); }
    *context = nullptr;
    hip_memory memory;
    hip_result estimate = hip_memory_requirements(maps.shape, limits, &memory, error);
    if (estimate != hip_result::success) { return estimate; }
    if (!maps.valid() || limits.max_keys <= 0 || limits.max_keys > INT32_MAX - 1 ||
        limits.max_queries <= 0 || limits.max_queries > UINT32_MAX / maps.shape.q_heads || limits.max_slots < 0 ||
        limits.max_slots > INT32_MAX || maps.shape.head_dim != 256 || maps.shape.rope_dim != 64 ||
        !((maps.shape.q_heads == 24 && maps.shape.kv_heads == 4) ||
          (maps.shape.q_heads == 16 && maps.shape.kv_heads == 2))) {
        return fail(hip_result::unsupported, error, "LSA HIP requires supported GQA geometry and bounded workspace");
    }
    size_t key_bytes, bucket_bytes, score_bytes, attention_bytes, intermediate;
    const size_t nb = size_t(limits.max_keys / cluster_size) * buckets_per_cluster;
    if (!product(size_t(limits.max_keys), index_dimension * sizeof(float), key_bytes) ||
        !product(nb, sizeof(bucket), bucket_bytes) ||
        !product(size_t(limits.max_queries), nb, intermediate) || !product(intermediate, sizeof(float), score_bytes) ||
        !product(size_t(limits.max_queries), maps.shape.q_heads, intermediate) ||
        !product(intermediate, size_t(limits.max_slots), attention_bytes) ||
        !product(attention_bytes, sizeof(float), attention_bytes)) {
        return fail(hip_result::invalid_argument, error, "LSA HIP workspace size overflow");
    }
    auto * c = new (std::nothrow) hip_context;
    if (!c) { return fail(hip_result::resource_error, error, "LSA HIP host allocation failed"); }
    c->shape = maps.shape; c->limits = limits; c->identity = maps.identity;
    c->stream = reinterpret_cast<hipStream_t>(stream);
    hipError_t rc = hipGetDevice(&c->device);
    auto alloc = [&](auto ** target, size_t bytes) {
        if (rc == hipSuccess && bytes) { rc = hipMalloc(reinterpret_cast<void **>(target), bytes); }
    };
    const size_t map_bytes = maps.q.size() * sizeof(ggml_bf16_t);
    alloc(&c->qmap, map_bytes); alloc(&c->kmap, map_bytes);
    alloc(&c->ambiguous, size_t(limits.max_keys / cluster_size));
    alloc(&c->keys, key_bytes); alloc(&c->buckets, bucket_bytes); alloc(&c->bucket_scores, score_bytes);
    alloc(&c->attention_scores, attention_bytes); alloc(&c->fault, sizeof(int)); alloc(&c->stats, sizeof(attention_stats));
    if (rc == hipSuccess) { rc = hipMemcpyAsync(c->qmap, maps.q.data(), map_bytes, hipMemcpyHostToDevice, c->stream); }
    if (rc == hipSuccess) { rc = hipMemcpyAsync(c->kmap, maps.k.data(), map_bytes, hipMemcpyHostToDevice, c->stream); }
    if (rc == hipSuccess) { rc = hipStreamSynchronize(c->stream); }
    if (rc != hipSuccess) {
        if (error) { *error = std::string("LSA HIP allocation/upload: ") + hipGetErrorString(rc); }
        hip_destroy(c); return hip_result::resource_error;
    }
    *context = c;
    if (error) { error->clear(); }
    return hip_result::success;
}

void hip_destroy(hip_context * c) {
    if (!c) { return; }
    if (c->device < 0) { delete c; return; }
    int old = -1;
    const bool restore = hipGetDevice(&old) == hipSuccess && old != c->device;
    if (restore) { (void) hipSetDevice(c->device); }
    (void) hipStreamSynchronize(c->stream);
    (void) hipFree(c->qmap); (void) hipFree(c->kmap); (void) hipFree(c->keys);
    (void) hipFree(c->buckets); (void) hipFree(c->ambiguous);
    (void) hipFree(c->bucket_scores); (void) hipFree(c->attention_scores);
    (void) hipFree(c->fault); (void) hipFree(c->stats);
    if (restore) { (void) hipSetDevice(old); }
    delete c;
}

hip_result hip_reset(hip_context * c, uint64_t epoch, const std::string & identity, std::string * error) {
    hip_result rc = ready(c, error);
    if (rc != hip_result::success) { return rc; }
    if (identity != c->identity) { return fail(hip_result::invalid_argument, error, "LSA HIP map changed; recreate context"); }
    c->count = 0; c->epoch = epoch; c->initialized = true; c->host_ambiguous.clear();
    return hip_result::success;
}

hip_result hip_append(hip_context * c, uint64_t epoch, int64_t first, const tensor_view & k, std::string * error) {
    hip_result rc = ready(c, error, true, epoch);
    if (rc != hip_result::success) { return rc; }
    rc = validate_projection(c, k, false, error);
    if (rc != hip_result::success) { return rc; }
    if (first != c->count || k.rows > c->limits.max_keys - first) {
        return fail(hip_result::invalid_argument, error, "LSA HIP append is not contiguous or exceeds capacity");
    }
    rc = project_into(c, k, false, c->keys + first * index_dimension, error);
    if (rc != hip_result::success) { return rc; }
    const int64_t old_clusters = c->count / cluster_size, new_count = first + k.rows;
    const int64_t new_clusters = new_count / cluster_size;
    if (new_clusters > old_clusters) {
        rc = begin(c, error);
        if (rc != hip_result::success) { return rc; }
        build_buckets<<<unsigned(new_clusters - old_clusters), pools, 0, c->stream>>>(c->keys, old_clusters, c->buckets, c->ambiguous, c->fault);
        rc = finish(c, error);
        if (rc != hip_result::success) { return rc; }
        c->host_ambiguous.resize(size_t(new_clusters));
        hipError_t hr = hipMemcpyAsync(c->host_ambiguous.data() + old_clusters, c->ambiguous + old_clusters,
                                      size_t(new_clusters - old_clusters), hipMemcpyDeviceToHost, c->stream);
        if (hr == hipSuccess) { hr = hipStreamSynchronize(c->stream); }
        if (hr != hipSuccess) { return device_failure(c, hr, error); }
    }
    c->count = new_count;
    return hip_result::success;
}

hip_result hip_truncate(hip_context * c, uint64_t epoch, int64_t count, std::string * error) {
    const hip_result rc = ready(c, error, true, epoch);
    if (rc != hip_result::success) { return rc; }
    if (count < 0 || count > c->count) { return fail(hip_result::invalid_argument, error, "LSA HIP invalid rollback"); }
    c->count = count; // suffix clusters are hidden; append rebuilds any touching block
    c->host_ambiguous.resize(size_t(count / cluster_size));
    return hip_result::success;
}

hip_result hip_project_queries(hip_context * c, const tensor_view & q, float * output, std::string * error) {
    const hip_result rc = validate_projection(c, q, true, error);
    if (rc != hip_result::success) { return rc; }
    if (q.rows && !output) { return fail(hip_result::invalid_argument, error, "LSA HIP null query output"); }
    return project_into(c, q, true, output, error);
}

hip_result hip_select(hip_context * c, uint64_t epoch, const float * queries, int64_t nq,
                     const int32_t * positions, const budgets & budget, int32_t * indices,
                     int32_t * lengths, int64_t capacity, std::string * error) {
    hip_result rc = ready(c, error, true, epoch);
    if (rc != hip_result::success) { return rc; }
    const int64_t width = int64_t(budget.top_k) + budget.window + budget.sink;
    if (!budget.valid() || nq < 0 || nq > c->limits.max_queries || capacity < width || capacity > c->limits.max_slots ||
        (nq && (!queries || !positions || !indices || !lengths))) {
        return fail(hip_result::invalid_argument, error, "LSA HIP invalid selection buffers or budget");
    }
    if (!nq) { return hip_result::success; }
    rc = begin(c, error); if (rc != hip_result::success) { return rc; }
    validate_queries<<<grid(nq), 256, 0, c->stream>>>(queries, nq, positions, c->count, c->ambiguous, c->fault);
    rc = finish(c, error); if (rc != hip_result::success) { return rc; }
    const int64_t nb = c->count / cluster_size * buckets_per_cluster;
    if (nb) {
        score_buckets<<<grid(nq * nb), 256, 0, c->stream>>>(queries, positions, c->buckets, nb, nq, c->bucket_scores, c->fault);
        rc = finish(c, error); if (rc != hip_result::success) { return rc; }
    }
    select_rows<<<unsigned(nq), 256, 0, c->stream>>>(c->bucket_scores, c->buckets, nb, positions, budget, capacity, indices, lengths);
    return finish(c, error);
}

hip_result hip_selected_attention(hip_context * c, const attention_args & a,
                                 attention_stats * host_stats, std::string * error) {
    hip_result rc = ready(c, error);
    if (rc != hip_result::success) { return rc; }
    if (!valid_view(a.q, false) || !valid_view(a.k, true) || !valid_view(a.v, true) ||
        a.q.heads != c->shape.q_heads || a.k.heads != c->shape.kv_heads || a.v.heads != a.k.heads ||
        a.q.dim != c->shape.head_dim || a.k.dim != a.q.dim || a.v.dim != c->shape.head_dim ||
        a.k.rows != a.v.rows || a.k.rows > c->limits.max_keys || a.q.rows > c->limits.max_queries ||
        a.main.capacity < 0 || a.extra.capacity < 0 || a.main.capacity > c->limits.max_slots ||
        a.extra.capacity > c->limits.max_slots - a.main.capacity ||
        (!a.main.indices && (a.main.capacity || a.main.lengths)) ||
        (!a.extra.indices && (a.extra.capacity || a.extra.lengths)) ||
        (a.q.rows && (!a.positions || !a.output)) || !std::isfinite(a.scale) ||
        !std::isfinite(a.softcap) || a.softcap < 0.0f) {
        return fail(hip_result::invalid_argument, error, "LSA HIP invalid physical attention geometry or buffers");
    }
    const int64_t slots = a.main.capacity + a.extra.capacity;
    if (host_stats) { *host_stats = {}; }
    if (!a.q.rows) { return hip_result::success; }
    rc = begin(c, error); if (rc != hip_result::success) { return rc; }
    validate_attention<<<grid(a.q.rows), 256, 0, c->stream>>>(a, c->fault);
    rc = finish(c, error); if (rc != hip_result::success) { return rc; }
    attention_stats * device_stats = host_stats ? c->stats : nullptr;
    hipError_t hr = device_stats ? hipMemsetAsync(device_stats, 0, sizeof(attention_stats), c->stream) : hipSuccess;
    if (hr != hipSuccess) { return device_failure(c, hr, error); }
    const unsigned blocks = unsigned(a.q.rows * a.q.heads);
    attention_logits<<<blocks, 256, 0, c->stream>>>(a, slots, c->attention_scores, device_stats, c->fault);
    rc = finish(c, error); if (rc != hip_result::success) { return rc; }
    attention_values<<<blocks, 256, 0, c->stream>>>(a, slots, c->attention_scores, device_stats, c->fault);
    rc = finish(c, error); if (rc != hip_result::success) { return rc; }
    if (host_stats) {
        hr = hipMemcpyAsync(host_stats, c->stats, sizeof(*host_stats), hipMemcpyDeviceToHost, c->stream);
        if (hr == hipSuccess) { hr = hipStreamSynchronize(c->stream); }
        if (hr != hipSuccess) { return device_failure(c, hr, error); }
    }
    return hip_result::success;
}

int64_t hip_size(const hip_context * c) { return c ? c->count : 0; }
const float * hip_index_keys(const hip_context * c) { return c ? c->keys : nullptr; }
const bucket * hip_index_buckets(const hip_context * c) { return c ? c->buckets : nullptr; }
int64_t hip_ambiguous_tie_clusters(const hip_context * c) {
    return c ? std::count(c->host_ambiguous.begin(), c->host_ambiguous.end(), uint8_t(1)) : 0;
}

} // namespace llama_lsa
