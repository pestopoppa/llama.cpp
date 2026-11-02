// Model-free native cell accounting regression. No inference.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/llama-kv-cells.h"
#include "../src/llama-memory.h"
#include "llama.h"

#include <cassert>
#include <cstdint>

struct unsupported_memory : llama_memory_i {
    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override { return {}; }
    llama_memory_context_ptr init_full() override { return {}; }
    llama_memory_context_ptr init_update(llama_context *, bool) override { return {}; }
    bool get_can_shift() const override { return false; }
    void clear(bool) override {}
    bool seq_rm(llama_seq_id, llama_pos, llama_pos) override { return false; }
    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}
    void seq_keep(llama_seq_id) override {}
    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override {}
    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override {}
    llama_pos seq_pos_min(llama_seq_id) const override { return -1; }
    llama_pos seq_pos_max(llama_seq_id) const override { return -1; }
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }
    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override {}
    void state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) override {}
};

int main() {
    uint32_t untouched = 999;
    assert(!llama_memory_get_unique_kv_used_cells(nullptr, &untouched));
    assert(untouched == 999);
    assert(!llama_memory_get_unique_kv_used_cells(nullptr, nullptr));
    unsupported_memory unsupported;
    assert(!llama_memory_get_unique_kv_used_cells(&unsupported, &untouched));
    assert(untouched == 999);

    llama_kv_cells cells;
    cells.resize(8);
    assert(cells.get_used() == 0);
    cells.pos_set(0, 0);
    cells.seq_add(0, 0);
    cells.seq_add(0, 1); // shared by two sequences: still one physical cell
    cells.pos_set(3, 1);
    cells.seq_add(3, 0);
    assert(cells.get_used() == 2);
    assert(!cells.seq_rm(0, 0)); // sequence 1 still owns it
    assert(cells.get_used() == 2);
    assert(cells.seq_rm(0, 1));
    assert(cells.get_used() == 1);
    cells.reset();
    assert(cells.get_used() == 0);
}
