// Q38FN-CPU-HOST-DEPTH-1: LLAMA_KV_PREV_FAST must not change get_prev_tokens() output.
// Builds random multi-stream / multi-seq KV cell layouts (holes, seq_rm-like gaps, shared cells, duplicate positions,
// out-of-order cell allocation), then compares every lookup the reference performs against the fast collection.
#include "../src/llama-kv-prev-tokens.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

static llama_token lookup(const std::unordered_map<uint64_t, llama_token> & hist,
                          const std::array<std::pair<llama_pos, llama_token>, LLAMA_MAX_SEQ> & below,
                          llama_seq_id s, llama_pos p, llama_pos w0) {
    for (llama_pos q = p; q >= w0; --q) {
        const auto it = hist.find(llama_kv_prev_key(s, q));
        if (it != hist.end()) {
            return it->second;
        }
    }
    return below[s].second;
}

int main() {
    std::mt19937 rng(1234);
    int n_cases = 0, n_lookups = 0;
    for (int iter = 0; iter < 3000; ++iter) {
        const uint32_t n_stream = 1 + rng() % 3;
        const uint32_t size     = 8 + rng() % 400;
        const int      n_seq    = 1 + rng() % 4;
        std::vector<llama_kv_cells> cells(n_stream);
        std::vector<llama_pos> next(n_seq, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            cells[s].resize(size);
            std::vector<uint32_t> order(size);
            for (uint32_t i = 0; i < size; ++i) order[i] = i;
            if (rng() % 3 == 0) std::shuffle(order.begin(), order.end(), rng);      // out-of-order allocation
            const uint32_t fill = rng() % size;
            for (uint32_t k = 0; k < fill; ++k) {
                const uint32_t i = order[k];
                const llama_seq_id sq = rng() % n_seq;
                llama_pos p = next[sq]++;
                if (rng() % 10 == 0) next[sq] += 1 + rng() % 3;                      // gap (seq_rm)
                if (rng() % 20 == 0 && p > 0) p -= 1;                                // duplicate position
                cells[s].pos_set(i, p);
                llama_kv_cell_ext e;
                e.tok = (llama_token) (rng() % 50000);
                cells[s].ext_set(i, e);
                cells[s].seq_add(i, sq);
                if (rng() % 15 == 0) {
                    const llama_seq_id o = rng() % n_seq;
                    if (!cells[s].seq_has(i, o)) cells[s].seq_add(i, o);            // shared cell
                }
            }
        }
        // a "ubatch": 1..5 tokens of 1..2 seqs at the tail of their positions
        const uint32_t n = 1 + rng() % 4;
        std::bitset<LLAMA_MAX_SEQ> seqs;
        std::vector<std::pair<llama_seq_id, llama_pos>> toks;
        const uint32_t nt = 1 + rng() % 5;
        for (uint32_t t = 0; t < nt; ++t) {
            const llama_seq_id sq = rng() % n_seq;
            const llama_pos p = std::max<llama_pos>(0, next[sq] - 1 - (llama_pos) (rng() % 3));
            toks.push_back({ sq, p });
            seqs.set(sq);
        }
        llama_pos p_min = toks[0].second, p_max = toks[0].second;
        for (auto & t : toks) { p_min = std::min(p_min, t.second); p_max = std::max(p_max, t.second); }
        const llama_pos w0 = p_min - (llama_pos) n;

        std::unordered_map<uint64_t, llama_token> h_ref, h_fast;
        std::array<std::pair<llama_pos, llama_token>, LLAMA_MAX_SEQ> b_ref, b_fast;
        llama_kv_prev_collect(cells, n_stream, seqs, w0, p_max, nullptr, h_ref, b_ref);
        std::unordered_set<uint64_t> needed;
        for (auto & t : toks) for (uint32_t d = 1; d <= n; ++d) if (t.second - (llama_pos) d >= 0) needed.insert(llama_kv_prev_key(t.first, t.second - d));
        llama_kv_prev_collect(cells, n_stream, seqs, w0, p_max, &needed, h_fast, b_fast);

        for (auto & t : toks) {
            for (uint32_t d = 1; d <= n; ++d) {
                const llama_pos p = t.second - (llama_pos) d;
                if (p < 0) continue;
                const llama_token a = lookup(h_ref, b_ref, t.first, p, w0);
                const llama_token b = lookup(h_fast, b_fast, t.first, p, w0);
                ++n_lookups;
                if (a != b) {
                    fprintf(stderr, "MISMATCH iter %d seq %d pos %d: ref %d fast %d\n", iter, t.first, p, a, b);
                    return 1;
                }
            }
        }
        ++n_cases;
    }
    printf("test-kv-prev-tokens: %d cases, %d lookups identical\n", n_cases, n_lookups);
    return 0;
}
