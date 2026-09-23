#pragma once

#include "llama.h"
#include "llama-graph.h"

#include <cstdint>
#include <string>
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
// Engram profiling -- the port-specific half.
//
// The model-agnostic half lives with the op (ggml-cpu.h, gather_rows_e4m3_e8m0 profiling): it
// answers "what did the gather cost", keyed by table address, and knows nothing about engrams.
// This half answers the questions that only make sense here -- per engram LAYER, which n-gram
// column produced which row, how often a row would have been found in a cache, and how much of
// a decode step the gather accounts for -- and folds the op's counters into one artifact.
//
// COMPILE-TIME GATE: -DGGML_CPU_PROF, the same cmake option (OFF by default) that gates the
// INF-70 per-node profiler in ggml-cpu.c, the host-phase profiler in llama-graph.cpp, and the
// op half in ops.cpp.  With the option off none of this exists: no struct, no call site in
// set_input, no environment-variable string.  The measured build is therefore provably
// uninstrumented, and a profiling build is a SEPARATE arm.
//
// Inside a profiling build everything is still off at run time unless
// LLAMA_ENGRAM_PROF_JSON_FILE names an output path.  No state is allocated and no counter is
// touched until then, so an un-enabled profiling build pays one pointer test per ubatch.
//
//   LLAMA_ENGRAM_PROF_JSON_FILE output path for the JSON artifact; naming it also ENABLES the
//                               profiler.  Written once per process, at exit (and on an
//                               explicit flush).  Same shape of switch as
//                               GGML_CPU_PROF_JSON_FILE and LLAMA_HOST_PROF_JSON_FILE.
//   LLAMA_ENGRAM_PROFILE        legacy alias for the path, honoured only if the above is unset
//   LLAMA_ENGRAM_PROFILE_LEVEL  1 (default) counters only
//                               2 also sets the op profiler to level 2, i.e. per-thread
//                                 getrusage fault attribution -- PERTURBING, diagnostic only
//   GGML_GATHER_PROF            the op profiler's own level, if it is to be set independently
//
#ifdef GGML_CPU_PROF
struct llama_dsv41_engram_prof {
    // Row-cache simulation capacities, in ROWS. Direct-mapped on the low bits of the row id,
    // which is a LOWER bound on an LRU cache of the same capacity: a reported hit rate that is
    // already too low to justify a cache settles the question, a high one does not by itself
    // prove an LRU cache would do as well or better.
    static const int N_CAP = 4;

    static const int N_HIST = 1024; // row-index histogram buckets, spanning [0, rows[e])

    enum { PHASE_PREFILL = 0, PHASE_DECODE = 1, N_PHASE = 2 };

    // Per (phase, engram layer). Counts are over rows EMITTED by the hash, which is also the
    // number of rows the op is asked to gather.
    struct layer_phase_stats {
        uint64_t n_calls   = 0;   // ubatches
        uint64_t n_tokens  = 0;
        uint64_t n_gated   = 0;   // tokens whose gate mask is 0: rows read, contribution zero
        uint64_t rows      = 0;   // = n_tokens * n_col
        uint64_t rows_uniq_in_token  = 0; // distinct row ids within one token's n_col ids
        uint64_t rows_uniq_in_ubatch = 0; // distinct row ids within the whole ubatch
        uint64_t rows_same_as_prev   = 0; // same row as the previous token in the same column
        uint64_t cache_hit[N_CAP] = { 0, 0, 0, 0 };
    };

    // Per engram layer, phase-independent: the simulated cache and the index histogram.
    struct layer_state {
        std::vector<int32_t>  cache_tag[N_CAP]; // -1 = empty
        std::vector<int32_t>  prev_row;         // [n_col]
        std::vector<uint64_t> index_hist;       // [N_HIST]
        std::vector<uint64_t> col_rows;         // [n_col]
        std::vector<uint64_t> col_same_as_prev; // [n_col]
    };

    // The op's counters, differenced per ubatch and attributed to the phase that ran them.
    struct op_phase_stats {
        uint64_t n_calls        = 0;
        uint64_t n_rows         = 0;
        uint64_t n_bytes_src    = 0;
        uint64_t us_span_ith0   = 0;
        uint64_t us_cpu         = 0;
        uint64_t n_thread_spans = 0;
        uint64_t minflt         = 0;
        uint64_t majflt         = 0;
    };

    static llama_dsv41_engram_prof & get();

    bool enabled() const { return level > 0; }

    // idempotent; safe to call on every ubatch
    void configure(const llama_dsv41_engram_spec & spec);

    // Called once per ubatch from set_input, AFTER the row ids for it have been computed.
    // `rows` is [n_tokens][n_engram][n_col], exactly the buffer set_input builds.
    void observe(const llama_dsv41_engram_spec & spec,
                 uint32_t n_tokens, uint32_t n_gated,
                 const int32_t * rows, int64_t us_hash);

    // Write the artifact. Called explicitly and, failing that, at process exit.
    void flush();

    int         level = 0;
    std::string path;

    uint32_t n_engram = 0;
    uint32_t n_col    = 0;

    std::vector<layer_state>       lstate;  // [n_engram]
    std::vector<layer_phase_stats> lstats;  // [N_PHASE][n_engram], flattened
    std::vector<op_phase_stats>    opstats; // [N_PHASE][n_op_slot], flattened

    // op slot -> engram layer, matched on the table's row count (spec.rows[e]); -1 if unknown
    std::vector<int>      op_slot_layer;
    std::vector<uint64_t> op_rows;   // spec.rows, the row count of each layer's table

    uint64_t us_hash      = 0;  // host-side hashing time in set_input
    uint64_t us_observe   = 0;  // the profiler's own cost, so it can be subtracted
    uint64_t us_step_span = 0;  // summed decode-step wall: set_input(t) -> set_input(t+1)
    uint64_t n_step_span  = 0;

    int64_t  t_last_set_input = 0;
    int      phase_last       = -1;

    bool configured = false;
    bool flushed    = false;
};
#endif // GGML_CPU_PROF

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
