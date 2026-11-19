// AK-DTREE-1 path-row tree verification helpers. Port of the --draft-tree-nodes design of Infernix
// (github.com/Wallawalla47/Infernix, Apache-2.0, @397f8cd1cc64); see speculative-tree.h.

#include "speculative-tree.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

int32_t common_dtree_greedy_accept_len(
        llama_context * ctx, const std::vector<int32_t> & rows, const std::vector<llama_token> & draft) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    int32_t n = 0;
    for (size_t i = 0; i < draft.size() && i < rows.size(); ++i) {
        // DTREE-A: the device argmax when the graph computed it (no n_vocab scan on the host)
        const llama_token dam = llama_get_argmax_ith(ctx, rows[i]);
        if (dam != LLAMA_TOKEN_NULL) {
            if (dam != draft[i]) {
                break;
            }
            ++n;
            continue;
        }
        const float * l = llama_get_logits_ith(ctx, rows[i]);
        if (l == nullptr) {
            break;
        }
        const llama_token am = (llama_token) std::distance(l, std::max_element(l, l + n_vocab));
        if (am != draft[i]) {
            break;
        }
        ++n;
    }
    return n;
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
