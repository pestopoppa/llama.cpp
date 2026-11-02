#include "../src/llama-kv-rows-plan.h"
#include "../src/llama-kv-cells.h"

#include <cstdint>
#include <cstdio>
#include <vector>

static uint32_t evaluate(std::vector<llama_seq_id> ids, uint32_t n_seqs,
                         uint32_t n_unique, bool equal, bool backend = true,
                         uint32_t streams = 1,
                         llama_kv_rows_sharing sharing = llama_kv_rows_sharing::absent) {
    std::vector<int32_t> owner_counts(ids.size(), 1);
    std::vector<llama_seq_id *> owners;
    for (auto & id : ids) owners.push_back(&id);
    llama_ubatch batch{};
    batch.b_equal_seqs = equal;
    batch.n_tokens = ids.size();
    batch.n_seqs = n_seqs;
    batch.n_seqs_unq = n_unique;
    batch.n_seq_id = owner_counts.data();
    batch.seq_id = owners.data();
    return llama_kv_rows_slices(batch, backend, streams, sharing);
}

int main() {
    // Independent single sequence, equal-length multi-token verify and one-token decode.
    if (evaluate({4, 4, 4}, 1, 1, true) != 1) return 1;
    if (evaluate({1, 1, 2, 2}, 2, 2, true) != 2) return 2;
    if (evaluate({1, 2, 3}, 3, 3, false) != 3) return 3;
    // Unsupported graph/backend, multi-stream and shared-cache all fall back.
    if (evaluate({1, 2}, 2, 2, false, false) != 0) return 4;
    if (evaluate({1, 2}, 2, 2, false, true, 2) != 0) return 5;
    if (evaluate({1, 2}, 2, 2, false, true, 1, llama_kv_rows_sharing::present) != 0) return 6;
    if (evaluate({1, 2}, 2, 2, false, true, 1, llama_kv_rows_sharing::unknown) != 0) return 12;
    // Unequal lengths, interleaving, duplicate slices and empty batches fall back.
    if (evaluate({1, 1, 2}, 2, 2, false) != 0) return 7;
    if (evaluate({1, 2, 1, 2}, 2, 2, true) != 0) return 8;
    if (evaluate({1, 1}, 2, 2, false) != 0) return 9;
    if (evaluate({}, 0, 0, false) != 0) return 10;
    llama_seq_id a[] = {1, 2};
    llama_seq_id * ptrs[] = {a};
    int32_t counts[] = {2};
    llama_ubatch coupled{};
    coupled.n_tokens = coupled.n_seqs = coupled.n_seqs_unq = 1;
    coupled.n_seq_id = counts; coupled.seq_id = ptrs;
    if (llama_kv_rows_slices(coupled, true, 1, llama_kv_rows_sharing::absent) != 0) return 11;
    // Actual cache ownership -> physical row map; a different map with the
    // same shape may reuse a graph only after its row tensor is refreshed.
    llama_seq_id x = 1, y = 2;
    llama_seq_id * seqs[] = {&x, &y};
    int32_t one[] = {1, 1};
    llama_ubatch b{};
    b.n_tokens = b.n_seqs = b.n_seqs_unq = 2;
    b.n_seq_id = one; b.seq_id = seqs;
    llama_kv_cells cells; cells.resize(8);
    cells.pos_set(1, 0); cells.seq_add(1, x);
    cells.pos_set(3, 0); cells.seq_add(3, y);
    auto plan = llama_kv_rows_make_plan(b, cells, 8, true, 1);
    if (!plan.usable() || plan.n_slices != 2 || plan.padded_rows != 8 ||
        plan.rows.size() != 16 || plan.rows[0] != 1 || plan.rows[8] != 3 || plan.rows[1] != -1) return 13;
    cells.pos_set(4, 1); cells.seq_add(4, x);
    auto refreshed = llama_kv_rows_make_plan(b, cells, 8, true, 1);
    if (!refreshed.same_graph_shape(plan) || refreshed.rows[1] != 4 || plan.rows[1] != -1) return 14;
    if (llama_kv_rows_make_plan(b, cells, 8, false, 1).usable()) return 15;
    if (llama_kv_rows_make_plan(b, cells, 9, true, 1).usable()) return 16;
    std::vector<int> full(16), compact;
    for (size_t i = 0; i < full.size(); ++i) full[i] = int(i);
    if (!llama_kv_rows_compact_mask(refreshed, 2, full, -99, compact) ||
        compact.size() != 16 || compact[0] != 1 || compact[1] != 4 || compact[2] != -99 ||
        compact[8] != 11 || compact[9] != -99) return 19;
    auto corrupted = refreshed; corrupted.rows[0] = 8;
    if (llama_kv_rows_compact_mask(corrupted, 2, full, -99, compact) || !compact.empty()) return 20;
    if (llama_kv_rows_compact_mask(refreshed, 3, full, -99, compact) || !compact.empty()) return 21;
    corrupted = refreshed; corrupted.n_kv = 0;
    if (llama_kv_rows_compact_mask(corrupted, 2, full, -99, compact) || !compact.empty()) return 22;
    cells.seq_add(1, y); // Shared prefix: always fall back.
    if (llama_kv_rows_make_plan(b, cells, 8, true, 1).usable()) return 17;
    cells.seq_rm(1, y);
    cells.pos_set(7, 0); // occupied but unknown owner: never infer safe sharing
    if (llama_kv_rows_make_plan(b, cells, 8, true, 1).usable()) return 18;
    std::puts("kv_rows shape and cache row planner passed");
    return 0;
}
