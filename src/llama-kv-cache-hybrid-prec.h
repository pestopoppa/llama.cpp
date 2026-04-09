#pragma once

#include "llama-kv-cache.h"

#include <vector>

//
// llama_kv_cache_hybrid_prec
//
// Hybrid precision KV cache for QJL/TurboQuant.
// Uses two llama_kv_cache instances:
//   - kv_recent: f16, small (n_kv_recent cells), for recent tokens (exact attention)
//   - kv_old:    turbo_q3 (or other compressed type), large, for older tokens
//
// New tokens always go to kv_recent. When it fills, oldest cells are evicted
// (compressed) to kv_old. At attention time, K/V from both caches are concatenated.

class llama_kv_cache_hybrid_prec : public llama_memory_i {
public:
    llama_kv_cache_hybrid_prec(
            const llama_model & model,
                    ggml_type   type_k,      // type for old K cache (e.g. TURBO_Q3)
                    ggml_type   type_v,      // type for V cache (both recent and old)
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,     // total context size
                     uint32_t   n_kv_recent, // recent buffer size (default 128)
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad);

    ~llama_kv_cache_hybrid_prec() = default;

    // llama_memory_i interface
    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // Hybrid-prec specific API
    llama_kv_cache * get_recent() const;
    llama_kv_cache * get_old()    const;
    bool has_old_data() const { return n_evicted > 0; }
    uint32_t get_n_evicted() const { return n_evicted; }

private:
    const llama_hparams & hparams;
    const bool unified;
    const uint32_t n_kv_recent;

    std::unique_ptr<llama_kv_cache> kv_recent; // f16, working cache
    std::unique_ptr<llama_kv_cache> kv_old;    // compressed, receives evicted tokens
    bool kv_old_has_data = false;
    uint32_t n_evicted = 0;                     // number of cells written to kv_old
    bool in_prefill = false;                    // suppress eviction during multi-token prefill

    // Evict oldest cells from kv_recent to kv_old (CPU-side tensor copy + quantize)
    void evict_oldest(uint32_t n_to_evict);
};
