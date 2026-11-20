#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // [TAG_QSA_IDXK_CACHE] default off: unset builds nothing, so every graph stays as it was
    const char * e = getenv("LLAMA_QSA_IDXK_CACHE");
    if (mem_idx && e != nullptr && atoi(e) != 0) {
        idxk_init(model, unified, n_seq_max);
    }
}

// [TAG_QSA_IDXK_CACHE] the cache is enabled only where bit identity with the full recompute is provable:
// a CPU-only model (no device that op offload could move a short recompute to), the indexer cache and every
// index_k_norm weight in host memory, and one sequence per stream. Anything else logs and stays off.
void llama_memory_hybrid_idx::idxk_init(const llama_model & model, bool unified, uint32_t n_seq_max) {
    const auto & hp = model.hparams;

    auto fail = [](const char * why) {
        LLAMA_LOG_WARN("%s: LLAMA_QSA_IDXK_CACHE=1 ignored (%s); full indexer-key recompute\n", "idxk_init", why);
    };

    if (model.n_devices() > 0) {
        return fail("model has GPU devices");
    }
    if (unified && n_seq_max > 1) {
        return fail("unified KV with several sequences per stream");
    }
    for (const auto & bs : mem_idx->memory_breakdown()) {
        if (!ggml_backend_buft_is_host(bs.first)) {
            return fail("indexer cache is not in host memory");
        }
    }

    std::vector<int32_t> layers;
    for (int32_t il = 0; il < (int32_t) hp.n_layer(); ++il) {
        if (!mem_idx->has_layer(il) || hp.dsv4_compress_ratios[il] == 0) {
            continue;
        }
        const ggml_tensor * w = model.layers[il].index_k_norm;
        if (w == nullptr || w->buffer == nullptr || !ggml_backend_buffer_is_host(w->buffer)) {
            return fail("index_k_norm weight is not in host memory");
        }
        layers.push_back(il);
    }
    if (layers.empty()) {
        return fail("no QSA layer");
    }

    auto st = std::make_unique<llama_qsa_idxk_store>();

    ggml_init_params ip = { layers.size()*ggml_tensor_overhead(), nullptr, true };
    st->ctx.reset(ggml_init(ip));

    const uint32_t kv_size  = mem_idx->get_size();
    const uint32_t n_stream = mem_idx->get_n_stream();

    for (int32_t il : layers) {
        const uint32_t r = hp.dsv4_compress_ratios[il];
        ggml_tensor * t = ggml_new_tensor_3d(st->ctx.get(), GGML_TYPE_F32, hp.indexer_head_size, (kv_size + r - 1)/r, n_stream);
        ggml_format_name(t, "idxk_cache_l%d", il);
        st->rows[il] = t;
        st->n_layers[r]++;
        if (st->state.find(r) == st->state.end()) {
            llama_qsa_idxk_validity v;
            v.r = r;
            st->state[r] = std::vector<llama_qsa_idxk_validity>(n_stream, v);
        }
    }

    st->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(st->ctx.get(), ggml_backend_cpu_buffer_type()));
    if (!st->buf) {
        return fail("buffer allocation failed");
    }
    ggml_backend_buffer_clear(st->buf.get(), 0);

    LLAMA_LOG_INFO("%s: LLAMA_QSA_IDXK_CACHE=1: caching post-rope pooled indexer keys, %zu layers, %u streams, %.2f MiB\n",
            __func__, layers.size(), n_stream, ggml_backend_buffer_get_size(st->buf.get())/1024.0/1024.0);

    idxk = std::move(st);
}

void llama_memory_hybrid_idx::idxk_reset() {
    if (idxk) {
        idxk->reset_all();
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.// rollback snapshots are per-row: a non-sequential split reorders cells and breaks them (C8-NP4-GARBAGE)
                split_equal(n_ubatch, !unified || n_rs_seq > 0, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    idxk_reset();   // [TAG_QSA_IDXK_CACHE]

    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // [TAG_QSA_IDXK_CACHE] before the removal, and even if it is refused: a rollback then re-decodes
    // these positions, possibly into the same cells with other tokens
    if (idxk) {
        idxk->on_write(seq_id < 0 ? -1 : (int32_t) mem_idx->get_seq_stream(seq_id), p0);
    }

    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    idxk_reset();   // [TAG_QSA_IDXK_CACHE] rare: forget everything

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    idxk_reset();   // [TAG_QSA_IDXK_CACHE] rare: forget everything

    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    idxk_reset();   // [TAG_QSA_IDXK_CACHE] rare: forget everything

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    idxk_reset();   // [TAG_QSA_IDXK_CACHE] rare: forget everything

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    if (idxk && idxk->buf) {   // [TAG_QSA_IDXK_CACHE]
        mb[ggml_backend_buffer_get_type(idxk->buf.get())] += ggml_backend_buffer_get_size(idxk->buf.get());
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    idxk_reset();   // [TAG_QSA_IDXK_CACHE]

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    idxk_reset();   // [TAG_QSA_IDXK_CACHE]

    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)),
    idxk_batch(true) {}

bool llama_memory_hybrid_idx_context::next() {
    // [TAG_QSA_IDXK_CACHE] next() is reached only after the ubatch computed successfully, so this is
    // where the rows its graph wrote become valid. A plan whose graph did not write every layer's rows
    // (fallback graph, fused decode, a partial graph) commits nothing.
    if (auto * st = mem ? mem->get_idxk() : nullptr) {
        for (const auto & it : idxk_plans) {
            const idxk_plan & pl = it.second;
            const auto nl = st->n_layers.find(it.first);
            if (!pl.on || !pl.batch || nl == st->n_layers.end() || pl.n_written != nl->second) {
                continue;
            }
            auto & vs = st->state.at(it.first);
            for (size_t s = 0; s < pl.blk_cells.size(); ++s) {
                vs.at(pl.s0 + s).commit(pl.blk_cells[s].data(), pl.filled[s].data(), pl.n_blocks, pl.min_pos[s]);
            }
        }
    }
    idxk_plans.clear();

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // [TAG_QSA_IDXK_CACHE] this ubatch (re)writes its positions in every stream it touches: every cached
    // block reaching the lowest of them is stale from here on, whatever happens to the compute
    idxk_plans.clear();
    if (auto * st = mem ? mem->get_idxk() : nullptr) {
        if (idxk_batch && ctx_idx) {
            const llama_ubatch & ub = get_ubatch();
            std::map<uint32_t, llama_pos> min_pos;
            for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                for (int32_t k = 0; k < ub.n_seq_id[i]; ++k) {
                    const uint32_t strm = mem->get_mem_idx()->get_seq_stream(ub.seq_id[i][k]);
                    auto mp = min_pos.find(strm);
                    if (mp == min_pos.end() || ub.pos[i] < mp->second) {
                        min_pos[strm] = ub.pos[i];
                    }
                }
            }
            for (const auto & mp : min_pos) {
                st->on_write((int32_t) mp.first, mp.second);
            }
        }
    }

    return res;
}

// [TAG_QSA_IDXK_CACHE]
const llama_memory_hybrid_idx_context::idxk_plan & llama_memory_hybrid_idx_context::get_idxk_plan(uint32_t ratio) const {
    auto found = idxk_plans.find(ratio);
    if (found != idxk_plans.end()) {
        return found->second;
    }

    idxk_plan & pl = idxk_plans[ratio];   // starts off: every early return below is the full recompute

    auto * st = mem ? mem->get_idxk() : nullptr;
    const auto * kvc = get_idx();
    if (st == nullptr || kvc == nullptr || st->state.find(ratio) == st->state.end()) {
        return pl;
    }

    const uint32_t r        = ratio;
    const uint32_t n_kv     = kvc->get_n_kv();
    const uint32_t n_blocks = (n_kv + r - 1)/r;
    const uint32_t s0       = kvc->get_s0();
    const uint32_t ns       = get_n_stream();

    const auto & vs = st->state.at(ratio);
    if (n_blocks == 0 || s0 + ns > vs.size()) {
        return pl;
    }

    pl.n_blocks = n_blocks;
    pl.s0       = s0;

    if (!idxk_batch) {
        // the reservation graph: recompute everything, the worst case, and commit nothing
        pl.on = true;
        return pl;
    }

    const llama_ubatch & ub = get_ubatch();
    if (ub.n_tokens == 0 || ub.n_tokens % ns != 0) {
        return pl;
    }
    const uint32_t n_tps = ub.n_tokens/ns;

    pl.blk_cells.assign(ns, std::vector<int32_t>((size_t) r*n_blocks));
    pl.filled   .assign(ns, std::vector<int32_t>(n_blocks));
    pl.min_pos  .assign(ns, INT_MAX);

    std::vector<llama_pos> cell_pos(n_kv);
    uint32_t n_reuse = UINT32_MAX;

    for (uint32_t s = 0; s < ns; ++s) {
        // the same stream -> cells resolution set_input_qsa uses
        const llama_seq_id seq = ub.seq_id[s*n_tps][0];
        if (kvc->get_s0() + s != mem->get_mem_idx()->get_seq_stream(seq)) {
            return pl;
        }
        const auto & cells = kvc->get_cells(seq);
        for (uint32_t j = 0; j < n_kv; ++j) {
            cell_pos[j] = cells.is_empty(j) ? -1 : cells.pos_get(j);
        }
        if (!llama_qsa_idxk_map(r, n_blocks, cell_pos.data(), n_kv, pl.blk_cells[s].data(), pl.filled[s].data())) {
            return pl;   // a position past the block window
        }
        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            pl.min_pos[s] = std::min(pl.min_pos[s], ub.pos[s*n_tps + ii]);
        }
        n_reuse = std::min(n_reuse, vs[s0 + s].n_reusable(pl.blk_cells[s].data(), pl.filled[s].data(), n_blocks));
    }

    // 64 blocks of slack: the split moves once every 64*ratio tokens, so most decode steps reuse the graph
    pl.c     = llama_qsa_idxk_split(n_reuse, n_blocks, 64);
    pl.batch = true;
    pl.on    = true;

    return pl;
}

ggml_tensor * llama_memory_hybrid_idx_context::get_idxk_rows(int32_t il) const {
    auto * st = mem ? mem->get_idxk() : nullptr;
    if (st == nullptr) {
        return nullptr;
    }
    const auto it = st->rows.find(il);
    return it == st->rows.end() ? nullptr : it->second;
}

void llama_memory_hybrid_idx_context::idxk_mark_written(uint32_t ratio, uint32_t c, int32_t n_layers) const {
    auto it = idxk_plans.find(ratio);
    // a graph built for another split must never run against this plan
    GGML_ASSERT(it != idxk_plans.end() && it->second.on && it->second.c == c && "qsa idxk: graph/plan split mismatch");
    it->second.n_written = n_layers;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        uint32_t blk0) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_cells->ne[0]/ratio;
    const int64_t n_pb     = blk_pos->ne[0]/(4*n_ns);   // [TAG_QSA_IDXK_CACHE] == n_blocks unless the cache splits
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // block b covers [b*ratio, (b+1)*ratio), so its first token is at b*ratio
    // all mrope sections carry it: exact for text, approximate for images
    for (int64_t sec = 0; sec < 4; ++sec) {
        for (int64_t s = 0; s < n_ns; ++s) {
            for (int64_t b = 0; b < n_pb; ++b) {
                dst_blk_pos[sec*(n_pb*n_ns) + s*n_pb + b] = (int32_t) ((blk0 + b)*r);
            }
        }
    }

    // one pass per stream: cell j is a different token in each, so no mapping is shared
    std::vector<int32_t> blk_of(n_kv);
    std::vector<int32_t> filled(n_blocks);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        // an incomplete block cannot be pooled; the bias below forces those tail cells in
        // -1 means no usable block, and block 0 only keeps the gather in range
        std::fill(blk_of.begin(),  blk_of.end(),  -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t   b = p/r;

            if (b >= n_blocks) {
                oor = true;
                continue;
            }

            blk_of[j] = (int32_t) b;
            cur_blk_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        // per-block mode keeps an unpooled cell's real block, so the block's own -inf reaches it
        // per-cell mode carries that -inf itself and only needs the gather in range
        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 && filled[blk_of[j]] < r && !blk_bias) {
                blk_of[j] = -1;
            }
            cur_cell_blk[j] = blk_of[j] < 0 ? 0 : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos    q      = ubatch->pos[i];

            // the tail is an incomplete block and is always visible, as in the reference
            const llama_pos tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = b*r >= tail_start ? 1e9f : (filled[b] < r ? -INFINITY : 0.0f);
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    v = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                }

                cur_bias[j] = v;
            }
        }
    }
}
