#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

struct oracle_input {
    int n_embd;
    int n_head;
    int n_kv;
    int n_batch;
    int n_stream;
    int n_mask_stream;
    int indexer_top_k;

    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> weights;
    std::vector<float> mask;
};

static constexpr float neg_inf = -std::numeric_limits<float>::infinity();

static size_t idx_q(const oracle_input & in, int e, int h, int t, int s) {
    return (((size_t) s*in.n_batch + t)*in.n_head + h)*in.n_embd + e;
}

static size_t idx_k(const oracle_input & in, int e, int kv, int s) {
    return (((size_t) s*in.n_kv + kv)*in.n_embd) + e;
}

static size_t idx_w(const oracle_input & in, int h, int t, int s) {
    return (((size_t) s*in.n_batch + t)*in.n_head) + h;
}

static size_t idx_mask(const oracle_input & in, int kv, int t, int ms) {
    return (((size_t) ms*in.n_batch + t)*in.n_kv) + kv;
}

static size_t idx_out(const oracle_input & in, int kv, int t, int s) {
    return (((size_t) s*in.n_batch + t)*in.n_kv) + kv;
}

static oracle_input make_input(
        int n_embd,
        int n_head,
        int n_kv,
        int n_batch,
        int n_stream,
        int n_mask_stream,
        int indexer_top_k) {
    oracle_input in {
        n_embd,
        n_head,
        n_kv,
        n_batch,
        n_stream,
        n_mask_stream,
        indexer_top_k,
        std::vector<float>((size_t) n_embd*n_head*n_batch*n_stream, 0.0f),
        std::vector<float>((size_t) n_embd*n_kv*n_stream, 0.0f),
        std::vector<float>((size_t) n_head*n_batch*n_stream, 1.0f),
        std::vector<float>((size_t) n_kv*n_batch*n_mask_stream, 0.0f),
    };

    assert(n_embd > 0);
    assert(n_head > 0);
    assert(n_kv > 0);
    assert(n_batch > 0);
    assert(n_stream > 0);
    assert(n_mask_stream > 0);
    assert(n_stream % n_mask_stream == 0);
    assert(indexer_top_k >= 0);

    return in;
}

static std::vector<float> dense_scores(const oracle_input & in) {
    std::vector<float> scores((size_t) in.n_kv*in.n_batch*in.n_stream, 0.0f);

    for (int s = 0; s < in.n_stream; ++s) {
        const int ms = s % in.n_mask_stream;
        for (int t = 0; t < in.n_batch; ++t) {
            for (int kv = 0; kv < in.n_kv; ++kv) {
                float score = 0.0f;
                for (int h = 0; h < in.n_head; ++h) {
                    float qk = 0.0f;
                    for (int e = 0; e < in.n_embd; ++e) {
                        qk += in.q[idx_q(in, e, h, t, s)]*in.k[idx_k(in, e, kv, s)];
                    }
                    score += std::max(qk, 0.0f)*in.weights[idx_w(in, h, t, s)];
                }
                scores[idx_out(in, kv, t, s)] = score + in.mask[idx_mask(in, kv, t, ms)];
            }
        }
    }

    return scores;
}

static std::vector<int32_t> selected_top_k(const oracle_input & in, const std::vector<float> & scores) {
    const int n_top_k = std::min(in.n_kv, in.indexer_top_k);
    std::vector<int32_t> top_k((size_t) n_top_k*in.n_batch*in.n_stream);

    for (int s = 0; s < in.n_stream; ++s) {
        for (int t = 0; t < in.n_batch; ++t) {
            std::vector<int32_t> order(in.n_kv);
            std::iota(order.begin(), order.end(), 0);
            const float * row = scores.data() + idx_out(in, 0, t, s);

            std::partial_sort(order.begin(), order.begin() + n_top_k, order.end(),
                    [row](int32_t a, int32_t b) {
                        if (row[a] == row[b]) {
                            return a < b;
                        }
                        return row[a] > row[b];
                    });

            for (int i = 0; i < n_top_k; ++i) {
                top_k[((size_t) s*in.n_batch + t)*n_top_k + i] = order[i];
            }
        }
    }

    return top_k;
}

static std::vector<float> selected_dense_mask(const oracle_input & in) {
    const std::vector<float> scores = dense_scores(in);
    const std::vector<int32_t> top_k = selected_top_k(in, scores);
    const int n_top_k = std::min(in.n_kv, in.indexer_top_k);
    std::vector<float> result((size_t) in.n_kv*in.n_batch*in.n_stream, neg_inf);

    for (int s = 0; s < in.n_stream; ++s) {
        const int ms = s % in.n_mask_stream;
        for (int t = 0; t < in.n_batch; ++t) {
            for (int i = 0; i < n_top_k; ++i) {
                const int32_t kv = top_k[((size_t) s*in.n_batch + t)*n_top_k + i];
                assert(kv >= 0);
                assert(kv < in.n_kv);
                result[idx_out(in, kv, t, s)] = 0.0f;
            }
            for (int kv = 0; kv < in.n_kv; ++kv) {
                result[idx_out(in, kv, t, s)] += in.mask[idx_mask(in, kv, t, ms)];
            }
        }
    }

    return result;
}

static std::vector<int32_t> visible_keys(const oracle_input & in, const std::vector<float> & mask, int t, int s) {
    std::vector<int32_t> keys;
    for (int kv = 0; kv < in.n_kv; ++kv) {
        if (std::isfinite(mask[idx_out(in, kv, t, s)])) {
            keys.push_back(kv);
        }
    }
    return keys;
}

static void require_visible(const oracle_input & in, const std::vector<float> & mask, int t, int s, std::vector<int32_t> expected) {
    std::vector<int32_t> actual = visible_keys(in, mask, t, s);
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    assert(actual == expected);
}

static void test_dense_mask_applies_before_top_k() {
    oracle_input in = make_input(1, 1, 5, 1, 1, 1, 2);

    in.q[idx_q(in, 0, 0, 0, 0)] = 1.0f;
    for (int kv = 0; kv < in.n_kv; ++kv) {
        in.k[idx_k(in, 0, kv, 0)] = float(kv + 1);
    }
    in.k[idx_k(in, 0, 1, 0)] = 100.0f;
    in.mask[idx_mask(in, 1, 0, 0)] = neg_inf;

    const std::vector<float> mask = selected_dense_mask(in);
    require_visible(in, mask, 0, 0, { 3, 4 });
}

static void test_original_mask_survives_top_k_overflow() {
    oracle_input in = make_input(1, 1, 5, 1, 1, 1, 4);

    in.q[idx_q(in, 0, 0, 0, 0)] = 1.0f;
    for (int kv = 0; kv < in.n_kv; ++kv) {
        in.k[idx_k(in, 0, kv, 0)] = float(kv + 1);
        in.mask[idx_mask(in, kv, 0, 0)] = neg_inf;
    }
    in.mask[idx_mask(in, 0, 0, 0)] = 0.0f;
    in.mask[idx_mask(in, 3, 0, 0)] = 0.0f;

    const std::vector<float> mask = selected_dense_mask(in);
    require_visible(in, mask, 0, 0, { 0, 3 });
}

static void test_broadcast_mask_keeps_stream_specific_scores() {
    oracle_input in = make_input(1, 1, 4, 2, 2, 1, 1);

    for (int s = 0; s < in.n_stream; ++s) {
        for (int t = 0; t < in.n_batch; ++t) {
            in.q[idx_q(in, 0, 0, t, s)] = 1.0f;
        }
    }

    const float stream0[] = { 1.0f, 2.0f, 3.0f, 100.0f };
    const float stream1[] = { 100.0f, 3.0f, 2.0f, 1.0f };
    for (int kv = 0; kv < in.n_kv; ++kv) {
        in.k[idx_k(in, 0, kv, 0)] = stream0[kv];
        in.k[idx_k(in, 0, kv, 1)] = stream1[kv];
    }

    in.mask[idx_mask(in, 3, 0, 0)] = neg_inf;
    in.mask[idx_mask(in, 0, 1, 0)] = neg_inf;

    const std::vector<float> mask = selected_dense_mask(in);
    require_visible(in, mask, 0, 0, { 2 });
    require_visible(in, mask, 0, 1, { 0 });
    require_visible(in, mask, 1, 0, { 3 });
    require_visible(in, mask, 1, 1, { 1 });
}

static void test_relu_weighted_scores_feed_selection() {
    oracle_input in = make_input(2, 2, 3, 1, 1, 1, 1);

    in.q[idx_q(in, 0, 0, 0, 0)] =  1.0f;
    in.q[idx_q(in, 1, 0, 0, 0)] =  0.0f;
    in.q[idx_q(in, 0, 1, 0, 0)] = -1.0f;
    in.q[idx_q(in, 1, 1, 0, 0)] =  0.0f;
    in.weights[idx_w(in, 0, 0, 0)] =  1.0f;
    in.weights[idx_w(in, 1, 0, 0)] = 10.0f;

    in.k[idx_k(in, 0, 0, 0)] =  2.0f;
    in.k[idx_k(in, 1, 0, 0)] =  0.0f;
    in.k[idx_k(in, 0, 1, 0)] =  0.0f;
    in.k[idx_k(in, 1, 1, 0)] =  1.0f;
    in.k[idx_k(in, 0, 2, 0)] = -1.0f;
    in.k[idx_k(in, 1, 2, 0)] =  0.0f;

    const std::vector<float> scores = dense_scores(in);
    assert(scores[idx_out(in, 0, 0, 0)] ==  2.0f);
    assert(scores[idx_out(in, 1, 0, 0)] ==  0.0f);
    assert(scores[idx_out(in, 2, 0, 0)] == 10.0f);

    const std::vector<float> mask = selected_dense_mask(in);
    require_visible(in, mask, 0, 0, { 2 });
}

int main() {
    test_dense_mask_applies_before_top_k();
    test_original_mask_survives_top_k_overflow();
    test_broadcast_mask_keeps_stream_specific_scores();
    test_relu_weighted_scores_feed_selection();

    return 0;
}
