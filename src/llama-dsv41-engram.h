#pragma once

#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <vector>

class llama_model_loader;
class llama_io_write_i;
class llama_io_read_i;

//
// DeepSeek-V4.1-Flash "Engram" conditional memory -- the host side.
//
// Engram writes an n-gram lookup into the residual stream at a few layers. Each position is
// hashed as (max_ngram_size - 1) n-grams (2-gram .. max_ngram_size-gram), each split over
// n_heads heads, so a position owns n_col = (max_ngram_size - 1) * n_heads table rows per
// engram layer. For the shipped checkpoint that is 3 * 8 = 24 rows per token per layer, 48
// rows per token over both layers; each row is 264 packed bytes (256 E4M3 codes + 8 E8M0
// scales) and dequantizes to 256 floats.
//
// Normative source: DeepSeek-V4.1-Flash/inference/engram.py -- build_compressed_token_map(),
// compute_hash_multipliers(), EngramLayout.from_args() and NgramHashState.forward() (lines
// 17-184). Working cross-check: antirez/ds4 @ 0aaea5a, ds4_engram.c:46-87.
//
// Nothing in the HF text-normalization chain has to be reimplemented: the GGUF ships the whole
// hash spec (deepseek41.engram.{token_map,primes,multipliers,pad_id,compressed_vocab_size}),
// and the multipliers it carries reproduce compute_hash_multipliers() exactly.
//

enum { LLAMA_DSV41_ENGRAM_MAX_NGRAM = 8 };

struct llama_dsv41_engram_spec {
    // a compressed id of DEAD takes no part in an n-gram and poisons every n-gram that would
    // span it: engram.py:138 (NgramHashState.DEAD), ds4_engram.h:16.
    static const int32_t DEAD = -1;

    uint32_t n_engram = 0; // engram layers                              (2)
    uint32_t n_ngram  = 0; // max_ngram_size, i.e. the lookback depth    (4)
    uint32_t n_head   = 0; // engram_n_heads                             (8)
    uint32_t n_col    = 0; // (n_ngram - 1) * n_head                     (24)

    uint32_t pad_id                = 0; // already a COMPRESSED id (engram.py:147)
    uint32_t compressed_vocab_size = 0;

    std::vector<uint32_t> layer_ids;   // [n_engram]           engram.layer_ids
    std::vector<uint64_t> rows;        // [n_engram]           engram.rows
    std::vector<uint32_t> token_map;   // [n_vocab]            engram.token_map
    std::vector<uint64_t> multipliers; // [n_engram][n_ngram]  engram.multipliers
    std::vector<uint32_t> primes;      // [n_engram][n_col]    engram.primes
    std::vector<uint64_t> offsets;     // [n_engram][n_col]    derived: exclusive cumsum of primes

    bool empty() const { return n_engram == 0; }

    // position of `il` inside layer_ids, or -1 when `il` carries no engram table
    int index_of_layer(uint32_t il) const;

    // token_map lookup; DEAD for an out-of-range id rather than an out-of-bounds read
    int32_t compress(llama_token tok) const;

    // Read deepseek41.engram.* out of the GGUF and derive n_ngram / n_head / n_col / offsets.
    // Returns false (leaving the spec empty) when engram.layer_ids is absent.
    bool load(llama_model_loader & ml, uint32_t n_vocab);

    // Every invariant the reference relies on. Throws on the first violation.
    void validate() const;

    // ngram[0] is the current token's compressed id (or DEAD); ngram[j] is j positions back
    // (or DEAD when that position is before the sequence start or itself dead). out receives
    // n_engram * n_col row ids, already offset into each layer's own table.
    void hash(const int32_t * ngram, int32_t * out) const;
};

//
// Per-sequence hash state.
//
// engram.py keeps a [batch, max_seq_len] cache of compressed ids and gathers the lookback out
// of it (engram.py:155-157, 167-174), which is what carries the hash across the prefill/decode
// split. This is the same thing, indexed by absolute position so that an MTP rollback -- which
// re-decodes positions it already wrote -- simply overwrites them. antirez/ds4's flat
// three-entry tail (ds4_engram.h:27-29) is the single-stream degenerate case of this.
//
class llama_dsv41_engram_state {
public:
    explicit llama_dsv41_engram_state(uint32_t n_seq_max) : seqs(n_seq_max) {}

    void    set(llama_seq_id seq_id, llama_pos pos, int32_t cid);
    int32_t get(llama_seq_id seq_id, llama_pos pos) const; // DEAD when never written

    void clear();
    void seq_rm  (llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void seq_keep(llama_seq_id seq_id);
    void seq_add (llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift);

    // Not yet wired into llama_kv_cache_dsv4::state_{write,read} -- see NOTES.md. Provided so
    // that closing that gap is a two-line change rather than a new design.
    void state_write(llama_io_write_i & io, llama_seq_id seq_id) const;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id);

private:
    bool valid(llama_seq_id seq_id) const {
        return seq_id >= 0 && (size_t) seq_id < seqs.size();
    }

    // [seq][pos] -> compressed id, or DEAD. 4 bytes per position per sequence.
    std::vector<std::vector<int32_t>> seqs;
};

//
// Graph input: the per-batch row ids, plus the per-token gate mask.
//
class llm_graph_input_dsv41_engram : public llm_graph_input_i {
public:
    llm_graph_input_dsv41_engram(
            const llama_dsv41_engram_spec * spec,
            llama_dsv41_engram_state      * state,
            uint32_t                        n_tokens) :
        spec(spec), state(state), n_tokens(n_tokens) {
    }

    ~llm_graph_input_dsv41_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // one per engram layer, in engram.layer_ids order: I32 [n_col, n_tokens]
    std::vector<ggml_tensor *> ids;

    // F32 [1, 1, n_tokens]: 1 where the Engram contribution applies, 0 at dead/image positions.
    // engram.py:363-364 zeroes the gate there; ds4 skips the whole block (ds4.c:40838).
    ggml_tensor * gate_mask = nullptr;

    const llama_dsv41_engram_spec * spec  = nullptr;
    llama_dsv41_engram_state      * state = nullptr;

    uint32_t n_tokens = 0;
};
