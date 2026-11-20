#pragma once

#include "llama-memory-hybrid.h"
#include "llama-qsa-idxk-cache.h"

#include "ggml-cpp.h"

#include <map>
#include <memory>
#include <vector>

// [TAG_QSA_IDXK_CACHE] persistent pooled indexer keys (LLAMA_QSA_IDXK_CACHE=1), see llama-qsa-idxk-cache.h
struct llama_qsa_idxk_store {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;

    std::map<int32_t, ggml_tensor *> rows;   // il -> F32 [idx_dim, ceil(kv_size/ratio), n_stream], post-rope rows
    std::map<uint32_t, int32_t>      n_layers;   // ratio -> layers holding rows for it
    std::map<uint32_t, std::vector<llama_qsa_idxk_validity>> state;   // ratio -> per absolute stream

    void reset_all() {
        for (auto & it : state) {
            for (auto & v : it.second) {
                v.reset();
            }
        }
    }

    // positions >= p_min of stream strm (every stream if strm < 0) were written or removed
    void on_write(int32_t strm, llama_pos p_min) {
        for (auto & it : state) {
            for (size_t s = 0; s < it.second.size(); ++s) {
                if (strm < 0 || (size_t) strm == s) {
                    it.second[s].on_write(p_min);
                }
            }
        }
    }
};

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // [TAG_QSA_IDXK_CACHE] nullptr unless LLAMA_QSA_IDXK_CACHE=1 and every precondition held at construction
    llama_qsa_idxk_store * get_idxk() const { return idxk.get(); }

private:
    // [TAG_QSA_IDXK_CACHE]
    void idxk_init(const llama_model & model, bool unified, uint32_t n_seq_max);
    void idxk_reset();

    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    std::unique_ptr<llama_qsa_idxk_store> idxk;   // [TAG_QSA_IDXK_CACHE]
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer, and for the update context, which builds no sparse graph
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // [TAG_QSA_IDXK_CACHE] blk_pos may hold only blocks [blk0, blk0 + blk_pos->ne[0]/(4*ns)) (default: all)
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, uint32_t blk0 = 0) const;

    // [TAG_QSA_IDXK_CACHE] the pooled-key split for the current ubatch and ratio: blocks [0, c) come from
    // the persistent rows, blocks [c, n_blocks) are recomputed and written back. on == false: full recompute.
    struct idxk_plan {
        bool     on       = false;
        bool     batch    = false;   // a real ubatch (commit allowed), not the reservation context
        uint32_t c        = 0;
        uint32_t n_blocks = 0;
        uint32_t s0       = 0;
        int32_t  n_written = 0;      // layers whose rows the executed graph writes (set_input)
        std::vector<std::vector<int32_t>> blk_cells;   // [ns][r*n_blocks]
        std::vector<std::vector<int32_t>> filled;      // [ns][n_blocks]
        std::vector<llama_pos>            min_pos;     // [ns] lowest position the ubatch writes per stream
    };

    const idxk_plan & get_idxk_plan(uint32_t ratio) const;

    // persistent rows of layer il, nullptr if the cache is off
    ggml_tensor * get_idxk_rows(int32_t il) const;

    // the executed graph (plan c) writes the rows of n_layers layers of this ratio
    void idxk_mark_written(uint32_t ratio, uint32_t c, int32_t n_layers) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer and this is a batch or full context
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // [TAG_QSA_IDXK_CACHE]
    const bool idxk_batch = false;
    mutable std::map<uint32_t, idxk_plan> idxk_plans;   // ratio -> plan of the current ubatch
};
