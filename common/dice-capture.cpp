// SPEC-DTREE-DICE step 2a capture (default off, LLAMA_DICE_CAPTURE=<path>); see dice-capture.h.

#include "dice-capture.h"

#include "common.h"
#include "rng-poskey.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <queue>

static FILE * common_dice_capture_file() {
    static FILE * f = [] {
        const char * p = std::getenv("LLAMA_DICE_CAPTURE");
        return (p && *p) ? std::fopen(p, "a") : (FILE *) nullptr;
    }();
    return f;
}

bool common_dice_capture_enabled() {
    return common_dice_capture_file() != nullptr;
}

void common_dice_capture_line(const std::string & line) {
    static std::mutex mtx;
    FILE * f = common_dice_capture_file();
    if (!f) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    std::fwrite(line.data(), 1, line.size(), f);
    std::fputc('\n', f);
    std::fflush(f);
}

double common_dice_gumbel(uint64_t seed, uint64_t pos, int32_t tok) {
    double u = llama_rng_poskey_u01(seed, pos, COMMON_DICE_ROLE_GUMBEL + (uint64_t) (uint32_t) tok);
    const double eps = 1.0 / 9007199254740992.0;
    u = std::min(std::max(u, eps), 1.0 - eps);
    return -std::log(-std::log(u));
}

namespace {

// log softmax(scores / T) over K entries
void dice_log_softmax(const float * s, int32_t K, float T, std::vector<double> & out) {
    out.resize(K);
    const double t = T > 0.0f ? (double) T : 1.0;
    double mx = -INFINITY;
    for (int32_t k = 0; k < K; ++k) {
        mx = std::max(mx, (double) s[k] / t);
    }
    double sum = 0.0;
    for (int32_t k = 0; k < K; ++k) {
        sum += std::exp((double) s[k] / t - mx);
    }
    const double lse = mx + std::log(sum);
    for (int32_t k = 0; k < K; ++k) {
        out[k] = (double) s[k] / t - lse;
    }
}

struct bf_item {
    double               score;
    std::vector<int32_t> path; // candidate indices, depth 0..
    bool operator<(const bf_item & o) const {
        if (score != o.score) {
            return score < o.score;          // max-heap on score
        }
        return o.path < path;                // ties: lexicographically smaller path first
    }
};

// best-first order of up to n nodes; edge(j, pred, out) fills the K per-child increments at depth j
std::vector<std::vector<int32_t>> dice_best_first(int32_t D, int32_t K, int32_t n,
        const std::function<void(int32_t, int32_t, std::vector<double> &)> & edge) {
    std::vector<std::vector<int32_t>> order;
    std::priority_queue<bf_item> pq;
    std::vector<double> inc;
    edge(0, 0, inc);
    for (int32_t k = 0; k < K; ++k) {
        pq.push({inc[k], {k}});
    }
    while (!pq.empty() && (int32_t) order.size() < n) {
        bf_item it = pq.top();
        pq.pop();
        order.push_back(it.path);
        const int32_t j = (int32_t) it.path.size();
        if (j < D) {
            edge(j, it.path.back(), inc);
            for (int32_t k = 0; k < K; ++k) {
                bf_item c{it.score + inc[k], it.path};
                c.path.push_back(k);
                pq.push(std::move(c));
            }
        }
    }
    return order;
}

} // namespace

std::vector<common_dice_node> common_dice_build_tree(
        int32_t D, int32_t K, const common_dice_scores_fn & scores, const common_dice_tok_fn & tok,
        const std::vector<int32_t> & chain_k, float temperature, uint64_t seed, uint64_t n_acc0, int32_t n_cols) {
    std::vector<common_dice_node> nodes(1); // node 0 = anchor
    if (D <= 0 || K <= 0 || n_cols < 2) {
        return nodes;
    }

    const int32_t n_order = std::max<int32_t>(n_cols - 1, 1);

    std::vector<double> ls;
    const auto prob_edge = [&](int32_t j, int32_t pred, std::vector<double> & out) {
        dice_log_softmax(scores(j, pred), K, temperature, out);
    };
    const auto dice_edge = [&](int32_t j, int32_t pred, std::vector<double> & out) {
        dice_log_softmax(scores(j, pred), K, temperature, ls);
        out.resize(K);
        double mx = -INFINITY;
        for (int32_t k = 0; k < K; ++k) {
            out[k] = ls[k] + common_dice_gumbel(seed, n_acc0 + (uint64_t) j, tok(j, k));
            mx = std::max(mx, out[k]);
        }
        for (int32_t k = 0; k < K; ++k) {
            out[k] -= mx;
        }
    };

    const auto prob_order = dice_best_first(D, K, n_order, prob_edge);
    const auto dice_order = dice_best_first(D, K, n_order, dice_edge);

    std::map<std::vector<int32_t>, int32_t> idx; // path -> node index
    const auto add = [&](const std::vector<int32_t> & path, int32_t bit) -> bool {
        auto it = idx.find(path);
        if (it != idx.end()) {
            nodes[it->second].src |= bit;
            return true;
        }
        if ((int32_t) nodes.size() >= n_cols) {
            return false;
        }
        int32_t par = 0;
        if (path.size() > 1) {
            auto ip = idx.find(std::vector<int32_t>(path.begin(), path.end() - 1));
            if (ip == idx.end()) {
                return false; // parent not in the tree (cannot happen for best-first orders; defensive)
            }
            par = ip->second;
        }
        common_dice_node nd;
        nd.par   = par;
        nd.depth = (int32_t) path.size() - 1;
        nd.k     = path.back();
        nd.tok   = tok(nd.depth, nd.k);
        nd.src   = bit;
        idx[path] = (int32_t) nodes.size();
        nodes.push_back(nd);
        return true;
    };

    for (size_t j = 0; j < chain_k.size() && (int32_t) j < D; ++j) {
        add(std::vector<int32_t>(chain_k.begin(), chain_k.begin() + j + 1), 1);
    }
    for (size_t i = 0; i < prob_order.size() || i < dice_order.size(); ++i) {
        if (i < prob_order.size()) {
            add(prob_order[i], 2);
        }
        if (i < dice_order.size()) {
            add(dice_order[i], 4);
        }
        if ((int32_t) nodes.size() >= n_cols) {
            break;
        }
    }
    // ranks of the present nodes in each best-first order (lets the replay find the fully captured prefix of each arm)
    for (size_t i = 0; i < prob_order.size(); ++i) {
        auto it = idx.find(prob_order[i]);
        if (it != idx.end()) {
            nodes[it->second].pr = (int32_t) i + 1;
        }
    }
    for (size_t i = 0; i < dice_order.size(); ++i) {
        auto it = idx.find(dice_order[i]);
        if (it != idx.end()) {
            nodes[it->second].dr = (int32_t) i + 1;
        }
    }
    // src bits 2/4 = "within the first n_cols - 1 nodes of that order" (the ranks are the authority for coverage)
    for (size_t i = 1; i < nodes.size(); ++i) {
        nodes[i].src = (nodes[i].src & 1) | (nodes[i].pr > 0 ? 2 : 0) | (nodes[i].dr > 0 ? 4 : 0);
    }
    return nodes;
}

std::vector<std::vector<llama_token>> common_dice_tree_alts(const std::vector<common_dice_node> & nodes) {
    std::vector<std::vector<llama_token>> alts;
    const int32_t n = (int32_t) nodes.size();
    std::vector<bool> has_child(n, false);
    for (int32_t i = 1; i < n; ++i) {
        if (nodes[i].par >= 0) {
            has_child[nodes[i].par] = true;
        }
    }
    for (int32_t i = 1; i < n; ++i) {
        if (has_child[i]) {
            continue;
        }
        // a chain-only path is the main draft (a leaf on the chain is the chain's last node or a prefix of it)
        bool all_chain = true;
        std::vector<llama_token> p;
        for (int32_t c = i; c > 0; c = nodes[c].par) {
            p.push_back(nodes[c].tok);
            all_chain = all_chain && (nodes[c].src & 1);
        }
        if (all_chain) {
            continue;
        }
        std::reverse(p.begin(), p.end());
        alts.push_back(std::move(p));
    }
    return alts;
}

std::string common_dice_rows_json(common_sampler * smpl, llama_context * ctx, const std::vector<int32_t> & rows, int32_t top_n) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    common_sampler * cl = common_sampler_clone(smpl);

    std::string js = "[";
    std::vector<std::pair<float, int32_t>> top;
    for (size_t r = 0; r < rows.size(); ++r) {
        if (r) {
            js += ",";
        }
        const int32_t idx = rows[r];
        const float * lg = idx >= 0 ? llama_get_logits_ith(ctx, idx) : nullptr;
        if (!lg) {
            js += "null";
            continue;
        }
        // raw top-n logits (ties: lower id first)
        const int32_t n = std::min(top_n, n_vocab);
        // min-heap of the n best (logit desc, id asc): heap top = the current worst kept entry
        const auto better = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        };
        top.clear();
        top.reserve(n + 1);
        for (int32_t t = 0; t < n_vocab; ++t) {
            const std::pair<float, int32_t> e(lg[t], t);
            if ((int32_t) top.size() < n) {
                top.push_back(e);
                std::push_heap(top.begin(), top.end(), better);
            } else if (better(e, top.front())) {
                std::pop_heap(top.begin(), top.end(), better);
                top.back() = e;
                std::push_heap(top.begin(), top.end(), better);
            }
        }
        std::sort(top.begin(), top.end(), better);
        js += "[[";
        for (int32_t i = 0; i < n; ++i) {
            js += string_format(i ? ",[%d,%.6g]" : "[%d,%.6g]", top[i].second, (double) top[i].first);
        }
        js += "],[";
        // post-sampler candidates of the request's own chain (clone; never accepted into)
        common_sampler_sample(cl, ctx, idx, false);
        const llama_token_data_array * cur = common_sampler_get_candidates(cl, true);
        bool first = true;
        for (size_t i = 0; cur && i < cur->size; ++i) {
            if (!(cur->data[i].p > 0.0f)) {
                continue;
            }
            js += string_format(first ? "[%d,%.6g]" : ",[%d,%.6g]", cur->data[i].id, (double) cur->data[i].p);
            first = false;
        }
        js += "]]";
    }
    js += "]";

    common_sampler_free(cl);
    return js;
}
