// AK-DTREE-1 path-row tree verification helpers. Port of the --draft-tree-nodes design of Infernix
// (github.com/Wallawalla47/Infernix, Apache-2.0, @397f8cd1cc64); see speculative-tree.h.

#include "speculative-tree.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>

namespace {

struct branch_cand {
    int32_t j;     // branch position on path 0
    int32_t c;     // candidate index taken at j
    double  score; // log P(path 0 right on [0, j)) + log P(c at j)
};

void log_softmax(const float * s, int32_t K, std::vector<double> & out) {
    out.resize(K);
    const float mx = *std::max_element(s, s + K);
    double sum = 0.0;
    for (int32_t k = 0; k < K; ++k) {
        sum += std::exp((double) s[k] - mx);
    }
    const double lse = mx + std::log(sum);
    for (int32_t k = 0; k < K; ++k) {
        out[k] = (double) s[k] - lse;
    }
}

// greedy continuation from position j (inclusive) given predecessor pred
void greedy_from(int32_t D, int32_t K, const common_dtree_scores_fn & scores, int32_t j, int32_t pred,
                 std::vector<int32_t> & path) {
    for (int32_t i = j; i < D; ++i) {
        const float * s = scores(i, pred);
        pred = (int32_t) std::distance(s, std::max_element(s, s + K));
        path.push_back(pred);
    }
}

} // namespace

std::vector<std::vector<int32_t>> common_dtree_build_paths(
        int32_t D, int32_t K, const common_dtree_scores_fn & scores, int32_t n_nodes, int32_t max_paths) {
    std::vector<std::vector<int32_t>> paths;
    if (D <= 0 || K <= 0) {
        return paths;
    }

    paths.emplace_back();
    paths[0].reserve(D);
    greedy_from(D, K, scores, 0, 0, paths[0]);

    max_paths = std::min(max_paths, COMMON_DTREE_MAX_PATHS);
    int32_t budget = n_nodes - (D + 1);
    if (max_paths <= 1 || budget <= 0 || K < 2) {
        return paths;
    }

    // enumerate rank-1 and rank-2 alternatives along path 0
    std::vector<branch_cand> cands;
    std::vector<double> lp;
    double cum = 0.0;
    for (int32_t j = 0; j < D; ++j) {
        const int32_t pred = j == 0 ? 0 : paths[0][j - 1];
        const float * s = scores(j, pred);
        log_softmax(s, K, lp);

        std::vector<int32_t> order(K);
        for (int32_t k = 0; k < K; ++k) {
            order[k] = k;
        }
        // stable: ties keep the lower index first, matching std::max_element for rank 0
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return s[a] > s[b]; });
        for (int32_t r = 1; r <= 2 && r < K; ++r) {
            const int32_t c = order[r];
            if (c == paths[0][j]) {
                continue;
            }
            cands.push_back({ j, c, cum + lp[c] });
        }
        cum += lp[paths[0][j]];
    }

    std::stable_sort(cands.begin(), cands.end(), [](const branch_cand & a, const branch_cand & b) {
        return a.score > b.score;
    });

    for (const auto & bc : cands) {
        if ((int32_t) paths.size() >= max_paths) {
            break;
        }
        const int32_t cost = D - bc.j;
        if (cost > budget) {
            continue;
        }
        std::vector<int32_t> p(paths[0].begin(), paths[0].begin() + bc.j);
        p.push_back(bc.c);
        greedy_from(D, K, scores, bc.j + 1, bc.c, p);
        paths.push_back(std::move(p));
        budget -= cost;
    }

    return paths;
}

namespace {

struct greedy_node {
    double               v;     // cumulative probability
    std::vector<int32_t> path;  // candidate-index path from the anchor
};

// max-heap order: higher v first, ties to the lexicographically smaller path (tree_sim.py heapq on (-v, path))
struct greedy_worse {
    bool operator()(const greedy_node & a, const greedy_node & b) const {
        if (a.v != b.v) {
            return a.v < b.v;
        }
        return b.path < a.path;
    }
};

// softmax exactly as tree_sim.py: exp(x - max) in double, Neumaier-compensated sum (CPython >= 3.12 sum() of floats),
// then divide. Bit-exact agreement keeps near-tie expansion order identical to the simulator.
void softmax_d(const float * s, int32_t K, std::vector<double> & out) {
    out.resize(K);
    double mx = (double) s[0];
    for (int32_t k = 1; k < K; ++k) {
        mx = std::max(mx, (double) s[k]);
    }
    double sum = 0.0;
    double c   = 0.0;
    for (int32_t k = 0; k < K; ++k) {
        out[k] = std::exp((double) s[k] - mx);
        const double x = out[k];
        const double t = sum + x;
        c += std::fabs(sum) >= std::fabs(x) ? (sum - t) + x : (x - t) + sum;
        sum = t;
    }
    if (c != 0.0 && std::isfinite(c)) {
        sum += c;
    }
    for (int32_t k = 0; k < K; ++k) {
        out[k] = out[k] / sum;
    }
}

} // namespace

std::vector<std::vector<int32_t>> common_dtree_build_greedy(
        int32_t D, int32_t K, const common_dtree_scores_fn & scores, int32_t n_nodes, bool branch_follows_main) {
    std::vector<std::vector<int32_t>> paths;
    if (D <= 0 || K <= 0 || n_nodes < 2) {
        return paths;
    }

    std::priority_queue<greedy_node, std::vector<greedy_node>, greedy_worse> heap;
    std::vector<double> pr;

    auto on_chain = [](const std::vector<int32_t> & p) {
        return std::all_of(p.begin(), p.end(), [](int32_t k) { return k == 0; });
    };
    // greedy chain index at each depth (argmax, lowest index on ties == std::max_element)
    std::vector<int32_t> chain;
    {
        int32_t pred = 0;
        for (int32_t j = 0; j < D; ++j) {
            const float * s = scores(j, pred);
            pred = (int32_t) std::distance(s, std::max_element(s, s + K));
            chain.push_back(pred);
        }
    }
    auto is_chain_prefix = [&](const std::vector<int32_t> & p) {
        return std::equal(p.begin(), p.end(), chain.begin());
    };

    auto push_children = [&](const std::vector<int32_t> & path, double v) {
        const int32_t j = (int32_t) path.size();
        if (j >= D) {
            return;
        }
        if (branch_follows_main && !on_chain(path)) {
            const float * s = scores(j, 0);
            softmax_d(s, K, pr);
            greedy_node n { v * pr[0], path };
            n.path.push_back(0);
            heap.push(std::move(n));
            return;
        }
        const float * s = scores(j, path.empty() ? 0 : path.back());
        softmax_d(s, K, pr);
        for (int32_t k = 0; k < K; ++k) {
            greedy_node n { v * pr[k], path };
            n.path.push_back(k);
            heap.push(std::move(n));
        }
    };

    std::vector<std::vector<int32_t>> tree; // insertion order
    push_children({}, 1.0);
    while ((int32_t) tree.size() < n_nodes - 1 && !heap.empty()) {
        greedy_node n = heap.top();
        heap.pop();
        tree.push_back(n.path);
        push_children(n.path, n.v);
    }

    // leaves: nodes that are no other node's parent
    std::vector<bool> leaf(tree.size(), true);
    for (const auto & p : tree) {
        if (p.size() < 2) {
            continue;
        }
        const std::vector<int32_t> par(p.begin(), p.end() - 1);
        for (size_t i = 0; i < tree.size(); ++i) {
            if (tree[i] == par) {
                leaf[i] = false;
                break;
            }
        }
    }
    // path 0: the deepest greedy-chain node in the tree
    std::vector<int32_t> main;
    for (const auto & p : tree) {
        if (is_chain_prefix(p) && p.size() > main.size()) {
            main = p;
        }
    }
    paths.push_back(main);
    for (size_t i = 0; i < tree.size(); ++i) {
        if (leaf[i] && !is_chain_prefix(tree[i])) {
            paths.push_back(tree[i]);
        }
    }
    return paths;
}

std::vector<std::vector<int32_t>> common_dtree_build_pando(
        int32_t D, int32_t K, const common_dtree_scores_fn & scores, const std::vector<int32_t> & main_in,
        float temp, int32_t n_nodes, int32_t k_children) {
    std::vector<std::vector<int32_t>> out;
    if (D <= 0 || K < 2 || k_children < 1 || n_nodes - 1 - D <= 0) {
        return out;
    }
    const double tau = temp > 0.0f ? (double) temp : 1.0;
    std::vector<float>  sbuf(K);
    std::vector<double> pr;
    auto dist = [&](int32_t j, int32_t pred) {
        const float * s = scores(j, pred);
        for (int32_t k = 0; k < K; ++k) {
            sbuf[k] = (float) ((double) s[k] / tau);
        }
        softmax_d(sbuf.data(), K, pr);
    };

    // main chain: the caller's (sampled) indices, or the greedy argmax chain (std::max_element tie rule)
    std::vector<int32_t> main = main_in;
    if ((int32_t) main.size() < D) {
        main.clear();
        int32_t pred = 0;
        for (int32_t j = 0; j < D; ++j) {
            const float * s = scores(j, pred);
            pred = (int32_t) std::distance(s, std::max_element(s, s + K));
            main.push_back(pred);
        }
    }
    main.resize(D);

    std::priority_queue<greedy_node, std::vector<greedy_node>, greedy_worse> heap;
    // children of `path` (cumulative probability v) except `skip` (the main child, already in the tree): the top
    // k_children candidates by the drafter's distribution at that depth (stable: lowest index first on ties)
    std::vector<int32_t> order(K);
    auto push_children = [&](const std::vector<int32_t> & path, double v, int32_t skip) {
        const int32_t j = (int32_t) path.size();
        if (j >= D || !(v > 0.0)) {
            return;
        }
        dist(j, path.empty() ? 0 : path.back());
        for (int32_t k = 0; k < K; ++k) {
            order[k] = k;
        }
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return pr[a] > pr[b]; });
        for (int32_t r = 0; r < k_children && r < K; ++r) {
            const int32_t k = order[r];
            if (k == skip || !(pr[k] > 0.0)) {
                continue;
            }
            greedy_node n { v * pr[k], path };
            n.path.push_back(k);
            heap.push(std::move(n));
        }
    };

    // the whole main chain is in the tree (the served rejection step needs it); its nodes seed the frontier
    {
        std::vector<int32_t> p;
        double v = 1.0;
        for (int32_t j = 0; j < D; ++j) {
            push_children(p, v, main[j]);
            dist(j, j == 0 ? 0 : main[j - 1]);
            v *= pr[main[j]];
            p.push_back(main[j]);
        }
    }

    std::vector<std::vector<int32_t>> tree; // off-main nodes, insertion order
    const int32_t extra = n_nodes - 1 - D;
    while ((int32_t) tree.size() < extra && !heap.empty()) {
        greedy_node n = heap.top();
        heap.pop();
        tree.push_back(n.path);
        push_children(n.path, n.v, -1);
    }

    std::vector<bool> leaf(tree.size(), true);
    for (const auto & p : tree) {
        const std::vector<int32_t> par(p.begin(), p.end() - 1);
        for (size_t i = 0; i < tree.size(); ++i) {
            if (tree[i] == par) {
                leaf[i] = false;
                break;
            }
        }
    }
    for (size_t i = 0; i < tree.size(); ++i) {
        if (leaf[i]) {
            out.push_back(tree[i]);
        }
    }
    return out;
}

int32_t common_dtree_pando_k() {
    static const int32_t k = [] {
        const char * e = std::getenv("LLAMA_DTREE_PANDO_K");
        const int32_t v = e ? std::atoi(e) : 0;
        return v >= 1 ? v : 3;
    }();
    return k;
}

int32_t common_dtree_builder() {
    static const int32_t b = [] {
        const char * e = std::getenv("LLAMA_DTREE_BUILDER");
        const int32_t v = e != nullptr && std::strcmp(e, "pando") == 0 ? 1 : 0;
        if (v == 1) {
            fprintf(stderr, COMMON_DTREE_PANDO_ENABLED_MSG "\n", common_dtree_pando_k());
        }
        return v;
    }();
    return b;
}

bool common_dtree_varlen() {
    return common_dtree_policy() == 1 || common_dtree_builder() == 1;
}

int32_t common_dtree_policy() {
    static const int32_t policy = [] {
        const char * e = std::getenv("LLAMA_DTREE_POLICY");
        return e != nullptr && std::strcmp(e, "greedy") == 0 ? 1 : 0;
    }();
    return policy;
}

int32_t common_dtree_n_columns(const std::vector<std::vector<int32_t>> & paths) {
    if (paths.empty()) {
        return 0;
    }
    int32_t n = 1 + (int32_t) paths[0].size();
    for (size_t p = 1; p < paths.size(); ++p) {
        // longest common prefix with any earlier path
        size_t best = 0;
        for (size_t q = 0; q < p; ++q) {
            size_t l = 0;
            while (l < paths[p].size() && l < paths[q].size() && paths[p][l] == paths[q][l]) {
                ++l;
            }
            best = std::max(best, l);
        }
        n += (int32_t) (paths[p].size() - best);
    }
    return n;
}

common_dtree_flat common_dtree_flatten(llama_token anchor, const std::vector<std::vector<llama_token>> & paths) {
    common_dtree_flat t;
    t.tokens.push_back(anchor);
    t.parent.push_back(-1);
    t.depth.push_back(0);
    for (const auto & p : paths) {
        int32_t cur = 0;
        for (size_t d = 0; d < p.size(); ++d) {
            int32_t next = -1;
            for (size_t c = 1; c < t.tokens.size(); ++c) {
                if (t.parent[c] == cur && t.tokens[c] == p[d]) {
                    next = (int32_t) c;
                    break;
                }
            }
            if (next < 0) {
                next = (int32_t) t.tokens.size();
                t.tokens.push_back(p[d]);
                t.parent.push_back(cur);
                t.depth.push_back((int32_t) d + 1);
            }
            cur = next;
        }
        t.leaf_col.push_back(cur);
    }
    return t;
}

llama_token common_dtree_row_argmax(const float * logits, int32_t n_vocab) {
    return (llama_token) std::distance(logits, std::max_element(logits, logits + n_vocab));
}

int32_t common_dtree_accept_walk(const std::vector<llama_token> & picks, const std::vector<llama_token> & draft) {
    int32_t n = 0;
    for (size_t i = 0; i < draft.size() && i < picks.size(); ++i) {
        if (picks[i] == LLAMA_TOKEN_NULL || picks[i] != draft[i]) {
            break;
        }
        ++n;
    }
    return n;
}

static std::atomic<int64_t> g_dtree_dev_rows  {0};
static std::atomic<int64_t> g_dtree_host_rows {0};

bool common_dtree_device_accept_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_DTREE_DEVICE_ACCEPT");
        const bool v = e != nullptr && e[0] == '1';
        if (v) {
            fprintf(stderr, "llama_dtree_device_accept: ENABLED (LLAMA_DTREE_DEVICE_ACCEPT=1; greedy tree accept reads device argmax rows, host fallback per row)\n");
        }
        return v;
    }();
    return on;
}

void common_dtree_device_accept_counts(int64_t * n_dev_rows, int64_t * n_host_rows) {
    if (n_dev_rows)  { *n_dev_rows  = g_dtree_dev_rows.load(); }
    if (n_host_rows) { *n_host_rows = g_dtree_host_rows.load(); }
}

int32_t common_dtree_greedy_accept_len(
        llama_context * ctx, const std::vector<int32_t> & rows, const std::vector<llama_token> & draft, bool allow_device) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const bool dev = allow_device && common_dtree_device_accept_enabled();

    int32_t n = 0;
    for (size_t i = 0; i < draft.size() && i < rows.size(); ++i) {
        llama_token am = LLAMA_TOKEN_NULL;
        if (dev) {
            // ggml_argmax ties to the lowest index (argmax.cu DTREE-A rule) == common_dtree_row_argmax; -1 (all -inf
            // or NaN row) comes back as LLAMA_TOKEN_NULL and takes the host scan, so edge rows match the host exactly
            am = llama_get_argmax_ith(ctx, rows[i]);
            if (am != LLAMA_TOKEN_NULL && (am < 0 || am >= n_vocab)) {
                am = LLAMA_TOKEN_NULL;
            }
        }
        if (am != LLAMA_TOKEN_NULL) {
            g_dtree_dev_rows.fetch_add(1, std::memory_order_relaxed);
        } else {
            const float * l = llama_get_logits_ith(ctx, rows[i]);
            if (l == nullptr) {
                break;
            }
            am = common_dtree_row_argmax(l, n_vocab);
            if (dev) {
                g_dtree_host_rows.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (am != draft[i]) {
            break;
        }
        ++n;
    }
    return n;
}

int32_t common_dtree_sampled_walk(
        const std::vector<const std::vector<int32_t> *>     & rows,
        const std::vector<const std::vector<llama_token> *> & paths,
        const std::function<llama_token(int32_t row)>        & sample,
        const std::function<void(llama_token tok)>           & accept,
        int32_t * best) {
    std::vector<int32_t> live;
    for (size_t k = 0; k < paths.size(); ++k) {
        if (rows[k]->size() == paths[k]->size() + 1) {
            live.push_back((int32_t) k);
        }
    }
    if (best) {
        *best = live.empty() ? 0 : live[0];
    }
    if (live.empty()) {
        return 0;
    }
    int32_t d = 0;
    std::vector<int32_t> next;
    for (;;) {
        const llama_token tok = sample((*rows[live[0]])[d]);
        next.clear();
        for (const int32_t k : live) {
            if ((size_t) d < paths[k]->size() && (*paths[k])[d] == tok) {
                next.push_back(k);
            }
        }
        if (next.empty()) {
            break; // tok is the bonus: not a child of the current node
        }
        accept(tok);
        live.swap(next);
        ++d;
        if (best) {
            *best = live[0];
        }
    }
    return d;
}

int32_t common_dtree_rrs_walk(
        const std::vector<const std::vector<int32_t> *>     & rows,
        const std::vector<const std::vector<llama_token> *> & paths,
        const std::function<llama_token(int32_t row)>        & sample,
        const std::function<bool(int32_t row, int32_t d, llama_token * y)> & reject,
        const std::function<void(llama_token tok)>           & accept,
        llama_token * bonus,
        int32_t * best,
        int64_t * n_alt) {
    std::vector<int32_t> live;
    for (size_t k = 0; k < paths.size(); ++k) {
        if (rows[k]->size() == paths[k]->size() + 1) {
            live.push_back((int32_t) k);
        }
    }
    if (best) {
        *best = live.empty() ? 0 : live[0];
    }
    if (live.empty()) {
        if (bonus) {
            *bonus = LLAMA_TOKEN_NULL;
        }
        return 0;
    }
    int32_t d = 0;
    std::vector<int32_t> next;
    for (;;) {
        const int32_t row = (*rows[live[0]])[d];
        bool has_child = false;
        for (const int32_t k : live) {
            has_child = has_child || (size_t) d < paths[k]->size();
        }
        const bool main_node = live[0] == 0 && (size_t) d < paths[0]->size();
        llama_token y;
        bool main_rejected = false;
        if (has_child && main_node && reject) {
            main_rejected = !reject(row, d, &y);
        } else {
            y = sample(row);
        }
        next.clear();
        const bool y_is_main = main_node && (*paths[0])[d] == y;
        if (!(main_rejected && y_is_main)) { // served: a rejected main stops even if the residual re-draws it
            for (const int32_t k : live) {
                if ((size_t) d < paths[k]->size() && (*paths[k])[d] == y) {
                    next.push_back(k);
                }
            }
        }
        if (next.empty()) {
            if (bonus) {
                *bonus = y;
            }
            break;
        }
        if (live[0] == 0 && next[0] != 0 && n_alt) {
            (*n_alt)++;
        }
        accept(y);
        live.swap(next);
        ++d;
        if (best) {
            *best = live[0];
        }
    }
    return d;
}

std::vector<std::vector<int32_t>> common_dtree_build_sampled(
        int32_t D, int32_t K, const common_dtree_scores_fn & scores, const std::vector<int32_t> & main,
        const std::vector<std::vector<float>> & q, int32_t n_nodes, int32_t max_paths) {
    std::vector<std::vector<int32_t>> out;
    int32_t budget = n_nodes - (D + 1);
    if (D <= 0 || K < 2 || max_paths <= 1 || budget <= 0 || (int32_t) main.size() < D || (int32_t) q.size() < D) {
        return out;
    }
    std::vector<branch_cand> cands;
    double cum = 0.0;
    for (int32_t j = 0; j < D; ++j) {
        const auto & qj = q[j];
        for (int32_t k = 0; k < K && k < (int32_t) qj.size(); ++k) {
            if (k != main[j] && qj[k] > 0.0f) {
                cands.push_back({ j, k, cum + std::log((double) qj[k]) });
            }
        }
        const float qm = main[j] < (int32_t) qj.size() ? qj[main[j]] : 0.0f;
        if (qm <= 0.0f) {
            break; // the main chain has no draft mass past here: no sibling below it is reachable by the walk
        }
        cum += std::log((double) qm);
    }
    std::stable_sort(cands.begin(), cands.end(), [](const branch_cand & a, const branch_cand & b) {
        return a.score > b.score;
    });
    for (const auto & c : cands) {
        if ((int32_t) out.size() + 1 >= max_paths) {
            break;
        }
        const int32_t cost = D - c.j;
        if (cost > budget) {
            continue;
        }
        std::vector<int32_t> path(main.begin(), main.begin() + c.j);
        path.push_back(c.c);
        greedy_from(D, K, scores, c.j + 1, c.c, path);
        out.push_back(std::move(path));
        budget -= cost;
    }
    return out;
}

int32_t common_dtree_sampled_max_paths() {
    static const int32_t w = [] {
        const char * e = getenv("LLAMA_DTREE_SAMPLED_W");
        const int32_t v = e ? atoi(e) : 0;
        return v >= 1 ? v : COMMON_DTREE_MAX_PATHS;
    }();
    return w;
}

bool common_dtree_rrs_q_ok(size_t n_dists, size_t n_main, bool seq_rm_part, bool seq_rm_rs, int32_t n_rs_seq) {
    return n_main > 0 && n_dists == n_main && (seq_rm_part || (seq_rm_rs && (int64_t) n_main <= (int64_t) n_rs_seq));
}

int32_t common_dtree_new_cols(const std::vector<int32_t> & main, const std::vector<int32_t> & alt) {
    size_t j = 0;
    while (j < main.size() && j < alt.size() && main[j] == alt[j]) {
        ++j;
    }
    return (int32_t) (alt.size() - j);
}

int32_t common_dtree_arm_cover(const std::vector<int32_t> & main, const std::vector<std::vector<int32_t>> & alts,
                               const std::vector<int32_t> & acc, int32_t extra_budget) {
    auto cover = [&](const std::vector<int32_t> & p) {
        size_t j = 0;
        while (j < acc.size() && j < p.size() && p[j] == acc[j]) {
            ++j;
        }
        return (int32_t) j;
    };
    int32_t best = cover(main);
    int32_t used = 0;
    for (const auto & a : alts) {
        used += common_dtree_new_cols(main, a);
        if (used > extra_budget) {
            break;
        }
        best = std::max(best, cover(a));
    }
    return best;
}

bool common_dtree_sampled_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_DTREE_SAMPLED");
        const bool v = e != nullptr && e[0] == '1';
        if (v) {
            fprintf(stderr, "%s\n", COMMON_DTREE_SAMPLED_ENABLED_MSG);
        }
        return v;
    }();
    return on;
}

void common_dtree_fork(llama_context * ctx, llama_seq_id src, llama_seq_id dst) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, dst, -1, -1);
    llama_memory_seq_cp(mem, src, dst, -1, -1);
}

void common_dtree_promote(llama_context * ctx, llama_seq_id src, llama_seq_id dst) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, dst, -1, -1);
    llama_memory_seq_cp(mem, src, dst, -1, -1);
    llama_memory_seq_rm(mem, src, -1, -1);
}

void common_dtree_drop(llama_context * ctx, llama_seq_id seq) {
    llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
}

int32_t common_dtree_auto::arm_nodes(int32_t arm, int32_t D) {
    switch (arm) {
        case 1:  return D + 5;
        case 2:  return D + 9;
        default: return 0;
    }
}

int32_t common_dtree_auto::bucket(int32_t n_batch, int32_t n_ctx) {
    const int32_t b = n_batch <= 1 ? 0 : (n_batch == 2 ? 1 : 2);
    const int32_t c = n_ctx <= 8192 ? 0 : (n_ctx <= 32768 ? 1 : (n_ctx <= 65536 ? 2 : 3));
    return b * N_CTX_B + c;
}

int32_t common_dtree_auto::choose(int32_t n_batch, int32_t n_ctx) {
    if (n_batch > 4) {
        return 0;
    }
    const int32_t b = bucket(n_batch, n_ctx);
    // timing first: 3 rounds per arm in this bucket, widest first
    for (int32_t a = N_ARMS - 1; a >= 0; --a) {
        if (n_t[b][a] < 3) {
            return a;
        }
    }
    // gain data: the widest arm's rounds measure every arm
    for (int32_t a = 0; a < N_ARMS; ++a) {
        if (n_gain[a] < GAIN_MIN) {
            return N_ARMS - 1;
        }
    }
    int32_t best = 0;
    double  best_r = -1.0;
    for (int32_t a = 0; a < N_ARMS; ++a) {
        const double r = gain(a) * 1e6 / t_us[b][a];
        rate[a]  = r;
        n_obs[a] = n_t[b][a];
        if (r > best_r) {
            best_r = r;
            best   = a;
        }
    }
    // re-timing: the widest arm every RETIME_WIDE rounds (it also refreshes every gain), stale arms
    if (best != N_ARMS - 1 && n_round - t_last[b][N_ARMS - 1] >= RETIME_WIDE) {
        return N_ARMS - 1;
    }
    for (int32_t a = 0; a < N_ARMS; ++a) {
        if (a != best && n_round - t_last[b][a] >= STALE) {
            return a;
        }
    }
    return best;
}

void common_dtree_auto::observe(int32_t arm, const int32_t * emit_by_arm, int64_t t_us_round, int32_t n_batch, int32_t n_ctx) {
    ++n_round;
    if (arm < 0 || arm >= N_ARMS || t_us_round <= 0 || n_batch > 4) {
        return;
    }
    for (int32_t a = 0; a <= arm; ++a) {
        if (emit_by_arm == nullptr || emit_by_arm[a] < 0) {
            continue;
        }
        gsum[a] = gsum[a] * (1.0 - 1.0 / 512.0) + (double) emit_by_arm[a];
        gw[a]   = gw[a]   * (1.0 - 1.0 / 512.0) + 1.0;
        ++n_gain[a];
    }
    const int32_t b = bucket(n_batch, n_ctx);
    const double  x = (double) t_us_round;
    double &      t = t_us[b][arm];
    if (n_t[b][arm] < 3) {
        t = n_t[b][arm] == 0 ? x : std::min(t, x);
    } else {
        t += (std::min(std::max(x, t * 0.5), t * 2.0) - t) / 8.0;
    }
    ++n_t[b][arm];
    t_last[b][arm] = n_round;
}

void common_dtree_auto::observe(int32_t arm, int32_t n_emit, int64_t t_us_round) {
    int32_t e[N_ARMS] = { -1, -1, -1 };
    if (arm >= 0 && arm < N_ARMS) {
        e[arm] = n_emit;
    }
    observe(arm, e, t_us_round, 1, 0);
}
