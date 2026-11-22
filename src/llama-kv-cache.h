#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <functional>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "");

    ~llama_kv_cache();

    //
    // llama_memory_i
    //

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
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // raw tensor access for KV compression scorers (Expected Attention, etc.)
    ggml_tensor * get_k_layer_raw(int32_t il) const;
    ggml_tensor * get_v_layer_raw(int32_t il) const;
    bool has_layer(int32_t il) const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // HSF-5.P host KV mirror (experimental). With no observer set (the default) nothing below changes behaviour.
    struct write_observer {
        virtual ~write_observer() = default;
        // cells `idxs` of stream `strm` were (re)written in every K/V layer
        virtual void on_cells_written(uint32_t strm, const uint32_t * idxs, size_t n) = 0;
        // the whole cache content changed in place (K-shift, stream copy, clear with data)
        virtual void on_invalidate_all() = 0;
    };

    void set_write_observer(write_observer * obs) { observer = obs; }

    // the per-layer base K and V tensors ([n_embd, kv_size, n_stream], null layers skipped)
    void get_mirror_tensors(std::vector<ggml_tensor *> & out) const;

    bool get_v_trans()      const { return v_trans; }
    bool get_shares_cells() const { return other != nullptr; }

    // state_write() keeps only cells with pos <= cap when cap >= 0 (a lag-mode mirror save); -1 = all cells
    mutable llama_pos state_write_pos_cap = -1;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
    const slot_info_vec_t *   sinfos_in);

    //
    // chunk-KV reuse (experimental; nothing here runs unless a caller invokes it)
    //

    // true when this cache can take part in chunk export/inject: a plain (non-SWA, non-mirrored) attention cache whose
    // K can be re-rotated. On false, *why (if given) says which condition failed.
    bool chunk_supported(std::string * why = nullptr) const;

    // write the cells of seq_id with pos in [p0, p1) (p1 < 0: up to the sequence end). The positions must be exactly
    // p0..p1-1, each once. Returns false and writes NOTHING when unsupported or the range has holes.
    bool chunk_export(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;

    // read a blob written by chunk_export into seq_id, placing its positions at pos + delta. Existing cells of the
    // sequence are kept; a clash with positions already present is refused. K is re-rotated by delta by the ordinary
    // K-shift graph at the next update (delta == 0: no rotation). Returns false and leaves the cache unchanged on refusal.
    bool chunk_inject(llama_io_read_i & io, llama_seq_id seq_id, llama_pos delta);

    // test/diagnostic read-back of the cells of seq_id at positions [p0, p1), converted to f32: layer-major,
    // [layer][pos][n_embd_{k,v}_gqa]. Returns the number of K floats (0 on refusal or a missing position); with
    // k_out/v_out null it only counts. Does NOT apply a pending K-shift: call the update first.
    size_t chunk_read_f32(llama_seq_id seq_id, llama_pos p0, llama_pos p1, float * k_out, float * v_out, size_t * n_v_out = nullptr) const;

    // startup self-check of the K re-rotation: rotate random K by +d then -d through the same graph the shift uses, for
    // every distinct rope configuration of the model. nmse_out = worst NMSE over configurations (f64 accumulation).
    bool chunk_rope_selfcheck(llama_context * lctx, llama_pos delta, double * nmse_out);

    // true when seq_id holds no cell with pos in [p0, p1) (sparse recompute fills such a hole)
    bool chunk_range_free(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // [GPU-RESIDENT-ROUND-2a] device-built KQ mask inputs (see llama-kq-mask-dev.h)
    bool kq_mask_dev_eligible(const llama_ubatch & ubatch, bool causal_attn) const;
    void set_input_kq_mask_dev(ggml_tensor * cpos, ggml_tensor * csel, ggml_tensor * tpos, const llama_ubatch * ubatch) const;

    // UP-FAROWS-HIP: per-sequence K/V row lists for FLASH_ATTN_EXT (ggml_flash_attn_ext_set_rows_key) over the first
    // n_kv cells of a unified cache. The ubatch is split into n_slices equal runs of single-sequence tokens when it is
    // made of such runs (np slots, draft verify), else one slice over the union of the cells of its sequences. Each
    // list holds the slice's cells in ascending cell order, padded with -1 to n_rows (a multiple of 256). Returns
    // false (rows untouched or empty) for a non-unified cache, or when the lists would not be shorter than n_kv.
    bool get_kv_rows(const llama_ubatch & ubatch, uint32_t n_kv, std::vector<int32_t> & rows, int64_t & n_rows, int64_t & n_slices) const;

    // UP-FAROWS-HIP C13: process-unique, stable non-zero key of this cache's row list in the backend registry
    int32_t get_fa_rows_key() const;
    // [AK-DTREE-2] tree ubatch: re-mask the cells written by this ubatch so token i sees only its ancestors
    void set_input_kq_mask_tree(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    // [AK-DTREE-2] keep only the accepted root path of the last tree ubatch of seq_id
    bool tree_accept(llama_seq_id seq_id, const int32_t * cols, int32_t n_cols) override;

    // Dual Chunk Attention: dst is F32 [n_kv, n_tps, n_stream, 3]; slice 0/1/2 is 1 where the
    // key cell lies in the query's chunk / the previous chunk / an earlier chunk, else 0
    // (chunk = pos / chunk_len). Cells the KQ mask drops get 0 in all three slices.
    void set_input_dca_sel(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t chunk_len) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    mutable int32_t fa_rows_key = 0; // UP-FAROWS-HIP C13: see get_fa_rows_key
    // [AK-DTREE-2] cells (stream, idx) of the last tree ubatch per seq, in tree column order
    std::unordered_map<llama_seq_id, std::vector<std::pair<uint32_t, uint32_t>>> tree_cells;

    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // HSF-5.P: null unless a host KV mirror is enabled
    write_observer * observer = nullptr;

public:
    // [GPU-RESIDENT-ROUND-2f phase 2.2] LLAMA_DEVICE_CTL=1: per-stream device control block F32 [2, n_stream]
    // (row 0 = next first position n_past the device expects, row 1 = stream base cell). The graph reads its K/V
    // write indices from it and writes back p0 + n_tokens; the host mirror tracks the device value and the host
    // writes an element only when the ubatch disagrees with it (rollback / new sequence). nullptr = disabled
    ggml_tensor * ctl = nullptr;
    mutable std::vector<float> ctl_host;
    mutable int64_t ctl_host_writes = 0;
    mutable int64_t ctl_device_hits = 0;
    mutable int64_t ctl_reads = 0;
    // [GPU-RESIDENT-ROUND-2f phase 3.1] LLAMA_DEVICE_CTL_POS=1: after a device accept advance the host predicts the
    // new n_past from the argmax rows it downloads anyway (same leading-match count as the device), instead of a
    // 4-byte read. ctl_pend[s]: the verify tokens and the pre-reorder output row of the first one
    struct ctl_pend_t {
        bool                 on   = false;
        float                p0   = 0.0f;
        int64_t              row0 = 0;
        std::vector<int32_t> tok;
    };
    mutable std::vector<ctl_pend_t> ctl_pend;
    mutable int64_t ctl_out_base = 0; // pre-reorder output row of the first token of the ubatch being built
    mutable int64_t ctl_predicts = 0;
    // [GPU-RESIDENT-ROUND-2f phase 3.2] LLAMA_DEVICE_VTOK=<T>: verify-token buffer F32 [T + 1, n_stream] the target's
    // verify graph reads its token input from (the drafter writes d1..d(T-1), the verify graph the next bonus token);
    // vtok_host mirrors it (NaN = unknown), the host writes a stream's T tokens only when the ubatch disagrees
    ggml_tensor * vtok = nullptr;
    int32_t       vtok_T = 0;
    mutable std::vector<float> vtok_host;
    mutable int64_t vtok_hits = 0, vtok_writes = 0, vtok_bonus_predicts = 0, vtok_mirror_sets = 0;
    // the drafter graph wrote token tok to col i (1..T-1) of stream s
    void vtok_mirror_set(int32_t s, int32_t i, int32_t tok) const;
    // [GPU-RESIDENT-ROUND-2f phase 3.3] LLAMA_DEVICE_RS_REP=1: control-block row 2 = the recurrent replay count of the
    // next ubatch (n_accepted + 1 after a device accept advance). rsrep_host mirrors it (NaN = unknown); the host
    // writes a stream's element only when the memory's replay count disagrees
    int32_t ctl_rows = 2;
    mutable std::vector<float> rsrep_host;
    mutable int64_t rsrep_hits = 0, rsrep_writes = 0, rsrep_predicts = 0;
    // rep[s] = replay count of stream s0 + s for this ubatch
    void rsrep_sync(int64_t s0, const std::vector<int32_t> & rep) const;
    // [GPU-RESIDENT-ROUND-2f phase 3.4a] LLAMA_DEVICE_CPOS=1: control-block row ctl_hw_row = high-water position of
    // the stream as the device last advanced it; the device-mask cell positions follow the ring formula
    // cpos_j = (E-1) - ((E-1-j) mod size), valid iff cpos_j > max(hw, E-1) - size and >= 0. hw_host mirrors the device
    // row, hw_true is the host's max position written since the stream was last emptied
    int32_t ctl_hw_row = -1;
    mutable std::vector<float>     hw_host;
    std::vector<llama_pos>         hw_true;
    mutable int64_t cpos_dev_hits = 0, cpos_hw_writes = 0, cpos_host_rounds = 0;
    bool cpos_check_streams(const llama_ubatch & ubatch, int64_t s0, int64_t n, int64_t n_tps, int64_t n_kv, int64_t size,
            const std::vector<float> & p0, std::vector<float> * hw_c) const;
    bool ctl_any_pending() const;
    // am(row) = device argmax of a pre-reorder output row of the previous decode (< 0 = unavailable)
    void ctl_resolve(const std::function<int32_t(int64_t)> & am) const;
private:

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il,
                           bool   inplace = true) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    bool kq_mask_dev_eligible (const llama_ubatch & ubatch, bool causal_attn) const;
    void set_input_kq_mask_dev(ggml_tensor * cpos, ggml_tensor * csel, ggml_tensor * tpos, const llama_ubatch * ubatch) const;
    // [GPU-RESIDENT-ROUND-2f phase 2] K/V write indices derivable on the device: per-sequence streams whose slots follow
    // the ring (cell = pos % size) for position-contiguous tokens; fills per stream the first position and stream base
    bool kv_idx_dev_info(const llama_ubatch & ubatch, std::vector<float> & p0, std::vector<float> & base, int64_t & size) const;
    // [GPU-RESIDENT-ROUND-2f phase 2.2] the ubatch's streams as a contiguous ascending range [s0, s0+n) of the
    // device control block (nullptr when disabled or not contiguous)
    ggml_tensor * ctl_range(const llama_ubatch & ubatch, int64_t & s0, int64_t & n) const;
    // [GPU-RESIDENT-ROUND-2f phase 3.2] the verify-token buffer when this ubatch is n streams x T tokens of seq ids
    // s0..s0+n-1 on the control-block path (nullptr otherwise); vtok_sync writes the streams whose mirror disagrees
    ggml_tensor * vtok_range(const llama_ubatch & ubatch, int64_t & s0, int64_t & n) const;
    int64_t       vtok_T() const;
    void          vtok_sync(const llama_ubatch & ubatch, int64_t s0, bool bonus) const;
    // writes the control-block elements whose mirror disagrees with the ubatch, then advances the mirror by n_tps
    // accept = the graph advances by the device accept count: the mirror becomes pending (resolved by a 4-byte read)
    // mode: 0 = graph has no write-back (build_sampling not run), 1 = advances by n_tps, 2 = by the device accept count
    void ctl_sync(const llama_ubatch & ubatch, int mode, const llama_kv_cache * tgt = nullptr) const;
    // [phase 3.3] writes control-block row 2 (recurrent replay count) of the streams whose mirror disagrees
    void rsrep_sync(int64_t s0, const std::vector<int32_t> & rep) const;
    // [phase 3.4a] true when the device ring formula reproduces the host cell positions of every stream of this
    // ubatch exactly (seq id == stream, ring layout); *hw_c = per-stream high-water the device must hold (mirror or
    // the host's true value). cpos_sync writes the elements whose mirror differs and advances the mirror
    bool cpos_dev_check(const llama_ubatch & ubatch, std::vector<float> * hw_c = nullptr) const;
    void cpos_sync(const llama_ubatch & ubatch, bool advance) const;
    int32_t ctl_hw_row() const;
    void cpos_host_round() const;
    void set_input_dca_sel   (ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t chunk_len) const;

    // UP-FAROWS-HIP: see llama_kv_cache::get_kv_rows
    bool get_kv_rows(const llama_ubatch & ubatch, std::vector<int32_t> & rows, int64_t & n_rows, int64_t & n_slices) const;
    int32_t get_fa_rows_key() const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

    // the used cells of a sequence (the fused decode fast path reads the cell
    // positions to compute the visible set and the current write cell)
    const llama_kv_cells & get_cells(llama_seq_id seq_id) const {
        return kv->get_cells(seq_id);
    }

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
