#include "../src/llama-kv-rows-plan.h"

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
    std::puts("kv_rows shape preflight passed");
    return 0;
}
