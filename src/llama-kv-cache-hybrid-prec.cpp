#include "llama-kv-cache-hybrid-prec.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_hybrid_prec
//

llama_kv_cache_hybrid_prec::llama_kv_cache_hybrid_prec(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_kv_recent,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad)
    : hparams(model.hparams), unified(unified), n_kv_recent(n_kv_recent) {

    uint32_t size_recent = kv_size;
    uint32_t size_old    = kv_size;

    LLAMA_LOG_INFO("%s: creating hybrid-prec KV cache: recent=%u cells (f16, evict threshold=%u), old=%u cells (%s)\n",
            __func__, size_recent, n_kv_recent, size_old, ggml_type_name(type_k));

    layer_filter_cb no_filter = nullptr;
    layer_reuse_cb  no_reuse  = nullptr;

    kv_recent = std::make_unique<llama_kv_cache>(
            model, GGML_TYPE_F16, type_v,
            v_trans, offload, unified, size_recent, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, no_filter, no_reuse);

    // Use q4_0 for old K cache instead of turbo_q3 — PolarQuant dequant has too much error
    // for direct K reconstruction. q4_0 with Hadamard smoothing is quality-neutral.
    ggml_type old_type_k = (type_k == GGML_TYPE_TURBO_Q3) ? GGML_TYPE_Q4_0 : type_k;
    LLAMA_LOG_INFO("%s: old cache K type: %s (requested: %s)\n",
            __func__, ggml_type_name(old_type_k), ggml_type_name(type_k));

    kv_old = std::make_unique<llama_kv_cache>(
            model, old_type_k, type_v,
            v_trans, offload, unified, size_old, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, no_filter, no_reuse);
}

void llama_kv_cache_hybrid_prec::clear(bool data) {
    LLAMA_LOG_DEBUG("%s: clearing hybrid cache (data=%d, n_evicted=%u)\n", __func__, data, n_evicted);
    kv_recent->clear(data);
    if (kv_old) {
        kv_old->clear(data);
        kv_old_has_data = false;
        n_evicted = 0;
    }
}

bool llama_kv_cache_hybrid_prec::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = kv_recent->seq_rm(seq_id, p0, p1);
    if (kv_old && kv_old_has_data) {
        res = res & kv_old->seq_rm(seq_id, p0, p1);
        // If old cache is now empty after removal, reset eviction counter
        if (kv_old->get_used() == 0) {
            kv_old_has_data = false;
            n_evicted = 0;
        }
    }
    return res;
}

void llama_kv_cache_hybrid_prec::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_recent->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (kv_old && kv_old_has_data) kv_old->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_hybrid_prec::seq_keep(llama_seq_id seq_id) {
    kv_recent->seq_keep(seq_id);
    if (kv_old && kv_old_has_data) kv_old->seq_keep(seq_id);
}

void llama_kv_cache_hybrid_prec::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_recent->seq_add(seq_id, p0, p1, shift);
    if (kv_old && kv_old_has_data) kv_old->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_hybrid_prec::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_recent->seq_div(seq_id, p0, p1, d);
    if (kv_old && kv_old_has_data) kv_old->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_hybrid_prec::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos p = kv_recent->seq_pos_min(seq_id);
    if (kv_old && kv_old_has_data) {
        llama_pos p2 = kv_old->seq_pos_min(seq_id);
        if (p2 >= 0 && (p < 0 || p2 < p)) p = p2;
    }
    return p;
}

llama_pos llama_kv_cache_hybrid_prec::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos p = kv_recent->seq_pos_max(seq_id);
    if (kv_old && kv_old_has_data) {
        llama_pos p2 = kv_old->seq_pos_max(seq_id);
        if (p2 > p) p = p2;
    }
    return p;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_hybrid_prec::memory_breakdown() const {
    auto mb = kv_recent->memory_breakdown();
    if (kv_old) {
        for (const auto & buft_size : kv_old->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    return mb;
}

bool llama_kv_cache_hybrid_prec::get_can_shift() const {
    return false; // hybrid cache does not support shifting
}

llama_memory_context_ptr llama_kv_cache_hybrid_prec::init_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // Detect prefill: if n_ubatch > 1, we're processing multiple tokens per batch.
    // Suppress eviction during prefill — all tokens need f16 precision for correct attention.
    // Eviction should only happen during single-token decode.
    in_prefill = (n_ubatch > 1);
    return kv_recent->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_kv_cache_hybrid_prec::init_full() {
    // For init_full, we return the recent cache's full context
    return kv_recent->init_full();
}

llama_memory_context_ptr llama_kv_cache_hybrid_prec::init_update(llama_context * lctx, bool optimize) {
    // Eviction: if kv_recent has more used cells than threshold, evict oldest to kv_old
    // Skip during prefill — all tokens need f16 precision during multi-token batch processing
    if (kv_old && !in_prefill) {
        uint32_t used = kv_recent->get_used();

        // Batch eviction: only evict when 64+ cells over threshold, amortizing CPU quantize cost
        if (used > n_kv_recent + 64) {
            uint32_t excess = used - n_kv_recent;
            // Cap eviction to avoid overwhelming kv_old capacity
            uint32_t kv_old_capacity = kv_old->get_size();
            if (n_evicted + excess > kv_old_capacity) {
                excess = kv_old_capacity > n_evicted ? kv_old_capacity - n_evicted : 0;
            }
            if (excess > 0) {
                evict_oldest(excess);
            }
        }
    }

    return kv_recent->init_update(lctx, optimize);
}

void llama_kv_cache_hybrid_prec::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_recent->state_write(io, seq_id, flags);
    if (kv_old) kv_old->state_write(io, seq_id, flags);
}

void llama_kv_cache_hybrid_prec::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kv_recent->state_read(io, seq_id, flags);
    if (kv_old) kv_old->state_read(io, seq_id, flags);
}

void llama_kv_cache_hybrid_prec::evict_oldest(uint32_t n_to_evict) {
    if (!kv_old || n_to_evict == 0) return;

    const uint32_t n_streams = kv_recent->get_n_stream();
    const uint32_t kv_size_old = kv_old->get_size();

    // Process each stream independently
    // Divide eviction count across streams (each stream contributes proportionally)
    uint32_t total_evicted_this_call = 0;

    for (uint32_t s = 0; s < n_streams; s++) {
        const auto & cells = kv_recent->get_cells(s);
        uint32_t stream_used = cells.get_used();
        uint32_t stream_threshold = n_kv_recent / n_streams;

        if (stream_used <= stream_threshold) continue;

        uint32_t stream_excess = stream_used - stream_threshold;

        // Cap by remaining kv_old capacity
        if (n_evicted + stream_excess > kv_size_old) {
            stream_excess = kv_size_old > n_evicted ? kv_size_old - n_evicted : 0;
        }
        if (stream_excess == 0) continue;

        // Find the oldest cells in this stream
        std::vector<std::pair<llama_pos, uint32_t>> pos_idx;
        for (uint32_t i = 0; i < cells.size(); i++) {
            if (cells.is_empty(i)) continue;
            pos_idx.push_back({cells.pos_get(i), i});
        }
        std::sort(pos_idx.begin(), pos_idx.end());

        if (pos_idx.size() < stream_excess) {
            stream_excess = (uint32_t)pos_idx.size();
        }

        // Copy K and V data for each layer
        for (uint32_t li = 0; li < hparams.n_layer; li++) {
        ggml_tensor * k_src = kv_recent->get_k_tensor(li);
        ggml_tensor * k_dst = kv_old->get_k_tensor(li);
        ggml_tensor * v_src = kv_recent->get_v_tensor(li);
        ggml_tensor * v_dst = kv_old->get_v_tensor(li);

        if (!k_src || !k_dst || !v_src || !v_dst) continue;

        const uint64_t n_embd_k = k_src->ne[0]; // n_embd_k_gqa
        const uint64_t n_embd_v = v_src->ne[0]; // n_embd_v_gqa

            for (uint32_t e = 0; e < stream_excess; e++) {
                uint32_t src_cell = pos_idx[e].second;
                uint32_t dst_cell = n_evicted + total_evicted_this_call + e;

                // K: f16 → f32 → turbo_q3
                {
                    const char * src_data = (const char *)k_src->data
                        + src_cell * ggml_row_size(k_src->type, n_embd_k);
                    char * dst_data = (char *)k_dst->data
                        + dst_cell * ggml_row_size(k_dst->type, n_embd_k);

                    float f32_buf[1024];
                    GGML_ASSERT(n_embd_k <= 1024);
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *)src_data, f32_buf, n_embd_k);

                    const auto * traits = ggml_get_type_traits(k_dst->type);
                    if (traits->from_float_ref) {
                        traits->from_float_ref(f32_buf, dst_data, n_embd_k);
                    }
                }

                // V: direct copy (same type in both caches)
                {
                    const size_t row_size = ggml_row_size(v_src->type, n_embd_v);
                    const char * src_data = (const char *)v_src->data + src_cell * row_size;
                    char * dst_data = (char *)v_dst->data + dst_cell * row_size;
                    memcpy(dst_data, src_data, row_size);
                }
            }
        }

        // Free evicted cells from kv_recent (this stream)
        {
            auto & cells_mut = kv_recent->get_cells(s);
            for (uint32_t e = 0; e < stream_excess; e++) {
                uint32_t cell_idx = pos_idx[e].second;
                if (!cells_mut.is_empty(cell_idx)) {
                    cells_mut.rm(cell_idx);
                }
            }
        }

        total_evicted_this_call += stream_excess;
    } // end stream loop

    n_evicted += total_evicted_this_call;
    if (total_evicted_this_call > 0) {
        kv_old_has_data = true;
        LLAMA_LOG_INFO("%s: evicted %u cells to kv_old (total: %u)\n",
                __func__, total_evicted_this_call, n_evicted);
    }
}

llama_kv_cache * llama_kv_cache_hybrid_prec::get_recent() const {
    return kv_recent.get();
}

llama_kv_cache * llama_kv_cache_hybrid_prec::get_old() const {
    return kv_old.get();
}
