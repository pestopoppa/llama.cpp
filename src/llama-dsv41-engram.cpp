#include "llama-dsv41-engram.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model-loader.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <stdexcept>

const int32_t llama_dsv41_engram_spec::DEAD;

//
// llama_dsv41_engram_spec
//

int llama_dsv41_engram_spec::index_of_layer(uint32_t il) const {
    for (size_t e = 0; e < layer_ids.size(); ++e) {
        if (layer_ids[e] == il) {
            return (int) e;
        }
    }

    return -1;
}

int32_t llama_dsv41_engram_spec::compress(llama_token tok) const {
    if (tok < 0 || (size_t) tok >= token_map.size()) {
        return DEAD;
    }

    return (int32_t) token_map[tok];
}

bool llama_dsv41_engram_spec::load(llama_model_loader & ml, uint32_t n_vocab) {
    std::vector<uint32_t> engram_layer_ids;

    if (!ml.get_arr(LLM_KV_DSV41_ENGRAM_LAYERS, engram_layer_ids, false) || engram_layer_ids.empty()) {
        *this = llama_dsv41_engram_spec();
        return false;
    }

    std::vector<uint32_t> engram_rows;

    ml.get_arr(LLM_KV_DSV41_ENGRAM_ROWS,        engram_rows);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_TOKEN_MAP,   token_map);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_PRIMES,      primes);
    ml.get_arr(LLM_KV_DSV41_ENGRAM_MULTIPLIERS, multipliers);

    ml.get_key(LLM_KV_DSV41_ENGRAM_PAD_ID,                pad_id);
    ml.get_key(LLM_KV_DSV41_ENGRAM_COMPRESSED_VOCAB_SIZE, compressed_vocab_size);

    layer_ids = engram_layer_ids;

    rows.assign(engram_rows.begin(), engram_rows.end());

    n_engram = (uint32_t) layer_ids.size();

    if (rows.size() != n_engram) {
        throw std::runtime_error("deepseek41: engram.rows and engram.layer_ids differ in length");
    }

    if (multipliers.size() % n_engram != 0 || primes.size() % n_engram != 0) {
        throw std::runtime_error("deepseek41: engram.primes/engram.multipliers are not a whole "
                                 "number of rows per engram layer");
    }

    // Everything the reference calls max_ngram_size / n_heads is recoverable from the two array
    // lengths: engram.py:344 fixes n_hash_cols = (max_ngram_size - 1) * n_heads, and
    // compute_hash_multipliers() emits exactly max_ngram_size multipliers per layer. The GGUF
    // publishes neither scalar (they live only inside the deepseek41.config blob).
    n_ngram = (uint32_t) (multipliers.size() / n_engram);
    n_col   = (uint32_t) (primes.size()      / n_engram);

    if (n_ngram < 2 || n_col == 0 || n_col % (n_ngram - 1) != 0) {
        throw std::runtime_error("deepseek41: engram.primes length is not a multiple of "
                                 "(max_ngram_size - 1)");
    }

    n_head = n_col / (n_ngram - 1);

    // engram.py:149 -- offsets = cumsum([0, *sizes[:-1]]) over the layer's flattened primes, so
    // every (n-gram order, head) bucket range is disjoint and they tile [0, rows).
    offsets.assign(primes.size(), 0);

    for (uint32_t e = 0; e < n_engram; ++e) {
        uint64_t acc = 0;

        for (uint32_t c = 0; c < n_col; ++c) {
            offsets[e*n_col + c] = acc;
            acc += primes[e*n_col + c];
        }
    }

    if (token_map.size() != n_vocab) {
        throw std::runtime_error(format("deepseek41: engram.token_map has %zu entries but the "
                                        "vocabulary has %u", token_map.size(), n_vocab));
    }

    validate();

    LLAMA_LOG_INFO("%s: engram: %u layers, %u-gram x %u heads = %u rows/token/layer, "
                   "compressed vocab %u, pad %u\n",
            __func__, n_engram, n_ngram, n_head, n_col, compressed_vocab_size, pad_id);

    return true;
}

void llama_dsv41_engram_spec::validate() const {
    if (compressed_vocab_size == 0 || compressed_vocab_size > (uint32_t) INT32_MAX) {
        throw std::runtime_error("deepseek41: engram.compressed_vocab_size out of range");
    }

    if (pad_id >= compressed_vocab_size) {
        throw std::runtime_error("deepseek41: engram.pad_id is not a valid compressed id");
    }

    if (n_ngram > LLAMA_DSV41_ENGRAM_MAX_NGRAM) {
        throw std::runtime_error("deepseek41: engram max_ngram_size exceeds the compiled maximum");
    }

    for (uint32_t tok = 0; tok < token_map.size(); ++tok) {
        if (token_map[tok] >= compressed_vocab_size) {
            throw std::runtime_error("deepseek41: engram.token_map entry out of range");
        }
    }

    for (uint32_t e = 0; e < n_engram; ++e) {
        // engram.py:70-72 bounds every multiplier so that id * multiplier cannot overflow int64,
        // and forces it odd (values * 2 + 1). ds4_engram.c:27-31 re-checks both at load.
        for (uint32_t j = 0; j < n_ngram; ++j) {
            const uint64_t m = multipliers[e*n_ngram + j];

            if ((m & 1) == 0 || m > (uint64_t) INT64_MAX / compressed_vocab_size) {
                throw std::runtime_error("deepseek41: engram.multipliers entry is even or "
                                         "large enough to overflow the hash");
            }
        }

        uint64_t total = 0;

        for (uint32_t c = 0; c < n_col; ++c) {
            if (primes[e*n_col + c] < 2) {
                throw std::runtime_error("deepseek41: engram.primes entry below 2");
            }

            total += primes[e*n_col + c];
        }

        // the bucket ranges must tile the table exactly (engram.py:92-93, ds4_engram.c:32-37)
        if (total != rows[e]) {
            throw std::runtime_error(format("deepseek41: engram.primes for layer %u sum to "
                                            "%" PRIu64 " but engram.rows says %" PRIu64,
                    layer_ids[e], total, rows[e]));
        }

        // row ids ride in an I32 ids tensor
        if (rows[e] > (uint64_t) INT32_MAX) {
            throw std::runtime_error("deepseek41: engram table has more rows than an I32 id "
                                     "can address");
        }
    }
}

void llama_dsv41_engram_spec::hash(const int32_t * ngram, int32_t * out) const {
    // engram.py:170-175: the blocked flag is sticky over the lookback, so pad_id fills the slot
    // at and past the first dead id (or the sequence start) and an n-gram never spans one.
    uint32_t ids[LLAMA_DSV41_ENGRAM_MAX_NGRAM];

    bool blocked = false;

    for (uint32_t j = 0; j < n_ngram; ++j) {
        blocked = blocked || ngram[j] == DEAD;
        ids[j]  = blocked ? pad_id : (uint32_t) ngram[j];
    }

    for (uint32_t e = 0; e < n_engram; ++e) {
        const uint64_t * mul = multipliers.data() + (size_t) e*n_ngram;
        const uint32_t * pri = primes.data()      + (size_t) e*n_col;
        const uint64_t * off = offsets.data()     + (size_t) e*n_col;

        int32_t * dst = out + (size_t) e*n_col;

        // engram.py:179-184: the running XOR after step j is the hash of the (j+1)-gram, and
        // each n-gram order lands in its own prime-sized bucket range.
        uint64_t h = (uint64_t) ids[0] * mul[0];

        for (uint32_t j = 1; j < n_ngram; ++j) {
            h ^= (uint64_t) ids[j] * mul[j];

            for (uint32_t head = 0; head < n_head; ++head) {
                const uint32_t c = (j - 1)*n_head + head;

                dst[c] = (int32_t) (h % pri[c] + off[c]);
            }
        }
    }
}

//
// llama_dsv41_engram_state
//

void llama_dsv41_engram_state::set(llama_seq_id seq_id, llama_pos pos, int32_t cid) {
    if (!valid(seq_id) || pos < 0) {
        return;
    }

    auto & cache = seqs[seq_id];

    if ((size_t) pos >= cache.size()) {
        cache.resize((size_t) pos + 1, llama_dsv41_engram_spec::DEAD);
    }

    cache[pos] = cid;
}

int32_t llama_dsv41_engram_state::get(llama_seq_id seq_id, llama_pos pos) const {
    if (!valid(seq_id) || pos < 0) {
        return llama_dsv41_engram_spec::DEAD;
    }

    const auto & cache = seqs[seq_id];

    if ((size_t) pos >= cache.size()) {
        return llama_dsv41_engram_spec::DEAD;
    }

    return cache[pos];
}

void llama_dsv41_engram_state::clear() {
    for (auto & cache : seqs) {
        cache.clear();
    }
}

void llama_dsv41_engram_state::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id < 0) {
        for (llama_seq_id s = 0; s < (llama_seq_id) seqs.size(); ++s) {
            seq_rm(s, p0, p1);
        }

        return;
    }

    if (!valid(seq_id)) {
        return;
    }

    auto & cache = seqs[seq_id];

    const llama_pos lo = p0 < 0 ? 0 : p0;
    const llama_pos hi = p1 < 0 ? (llama_pos) cache.size() : std::min<llama_pos>(p1, (llama_pos) cache.size());

    for (llama_pos p = lo; p < hi; ++p) {
        cache[p] = llama_dsv41_engram_spec::DEAD;
    }
}

void llama_dsv41_engram_state::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (!valid(seq_id_src) || !valid(seq_id_dst) || seq_id_src == seq_id_dst) {
        return;
    }

    const auto & src = seqs[seq_id_src];

    const llama_pos lo = p0 < 0 ? 0 : p0;
    const llama_pos hi = p1 < 0 ? (llama_pos) src.size() : std::min<llama_pos>(p1, (llama_pos) src.size());

    for (llama_pos p = lo; p < hi; ++p) {
        set(seq_id_dst, p, src[p]);
    }
}

void llama_dsv41_engram_state::seq_keep(llama_seq_id seq_id) {
    for (llama_seq_id s = 0; s < (llama_seq_id) seqs.size(); ++s) {
        if (s != seq_id) {
            seqs[s].clear();
        }
    }
}

void llama_dsv41_engram_state::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (seq_id < 0) {
        for (llama_seq_id s = 0; s < (llama_seq_id) seqs.size(); ++s) {
            seq_add(s, p0, p1, shift);
        }

        return;
    }

    if (!valid(seq_id) || shift == 0) {
        return;
    }

    auto & cache = seqs[seq_id];

    const llama_pos lo = p0 < 0 ? 0 : p0;
    const llama_pos hi = p1 < 0 ? (llama_pos) cache.size() : std::min<llama_pos>(p1, (llama_pos) cache.size());

    std::vector<int32_t> moved;

    moved.reserve(hi > lo ? (size_t) (hi - lo) : 0);

    for (llama_pos p = lo; p < hi; ++p) {
        moved.push_back(cache[p]);
        cache[p] = llama_dsv41_engram_spec::DEAD;
    }

    for (size_t i = 0; i < moved.size(); ++i) {
        set(seq_id, lo + (llama_pos) i + shift, moved[i]);
    }
}

void llama_dsv41_engram_state::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    const llama_seq_id lo = seq_id < 0 ? 0                          : seq_id;
    const llama_seq_id hi = seq_id < 0 ? (llama_seq_id) seqs.size() : seq_id + 1;

    for (llama_seq_id s = lo; s < hi; ++s) {
        const uint32_t n = valid(s) ? (uint32_t) seqs[s].size() : 0;

        io.write(&n, sizeof(n));

        if (n > 0) {
            io.write(seqs[s].data(), n*sizeof(int32_t));
        }
    }
}

void llama_dsv41_engram_state::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    const llama_seq_id lo = seq_id < 0 ? 0                          : seq_id;
    const llama_seq_id hi = seq_id < 0 ? (llama_seq_id) seqs.size() : seq_id + 1;

    for (llama_seq_id s = lo; s < hi; ++s) {
        uint32_t n = 0;

        io.read(&n, sizeof(n));

        std::vector<int32_t> cache(n, llama_dsv41_engram_spec::DEAD);

        if (n > 0) {
            io.read(cache.data(), n*sizeof(int32_t));
        }

        if (valid(s)) {
            seqs[s] = std::move(cache);
        }
    }
}

//
// llm_graph_input_dsv41_engram
//

void llm_graph_input_dsv41_engram::set_input(const llama_ubatch * ubatch) {
    if (spec == nullptr || spec->empty()) {
        return;
    }

    GGML_ASSERT(state != nullptr);
    GGML_ASSERT(ids.size() == spec->n_engram);
    GGML_ASSERT(gate_mask != nullptr);

    // Engram hashes token ids. An embedding batch carries no ids, so there is nothing to hash
    // and no honest fallback -- engram.py takes input_ids, not h (model.py:1252).
    GGML_ASSERT(ubatch->token != nullptr && "deepseek41: Engram requires a token batch");

    const uint32_t n_tok   = ubatch->n_tokens;
    const uint32_t n_col   = spec->n_col;
    const uint32_t n_ngram = spec->n_ngram;
    const uint32_t n_eng   = spec->n_engram;

    GGML_ASSERT(n_tok == n_tokens);

    // (1) Commit this ubatch's compressed ids first, exactly as engram.py:167 fills the cache
    //     over [start_pos, start_pos + seqlen) before it gathers any lookback. Tokens inside one
    //     ubatch may then see each other, whatever order the split produced.
    for (uint32_t i = 0; i < n_tok; ++i) {
        const int32_t cid = spec->compress(ubatch->token[i]);

        for (int32_t s = 0; s < ubatch->n_seq_id[i]; ++s) {
            state->set(ubatch->seq_id[i][s], ubatch->pos[i], cid);
        }
    }

    // (2) Hash every token. 48 row ids per token for the shipped checkpoint.
    std::vector<int32_t> rows((size_t) n_tok * n_eng * n_col);
    std::vector<float>   mask(n_tok, 1.0f);

    int32_t ngram[LLAMA_DSV41_ENGRAM_MAX_NGRAM];

    for (uint32_t i = 0; i < n_tok; ++i) {
        const llama_seq_id seq = ubatch->n_seq_id[i] > 0 ? ubatch->seq_id[i][0] : 0;
        const llama_pos    pos = ubatch->pos[i];

        for (uint32_t j = 0; j < n_ngram; ++j) {
            // engram.py:172-173: (positions - shift).clamp_min(0) combined with
            // `blocked | (positions < shift)`, i.e. a lookback before position 0 is blocked.
            ngram[j] = pos >= (llama_pos) j
                ? state->get(seq, pos - (llama_pos) j)
                : llama_dsv41_engram_spec::DEAD;
        }

        // engram.py:363-364 / model.py:1251: a position that takes no part in an n-gram also
        // gets no Engram contribution. Its rows are still computed (and still read), matching
        // the reference, which masks the gate rather than the lookup.
        mask[i] = ngram[0] == llama_dsv41_engram_spec::DEAD ? 0.0f : 1.0f;

        spec->hash(ngram, rows.data() + (size_t) i*n_eng*n_col);
    }

    // (3) Scatter into one I32 [n_col, n_tokens] tensor per engram layer.
    std::vector<int32_t> plane((size_t) n_tok * n_col);

    for (uint32_t e = 0; e < n_eng; ++e) {
        for (uint32_t i = 0; i < n_tok; ++i) {
            const int32_t * src = rows.data() + (size_t) i*n_eng*n_col + (size_t) e*n_col;

            std::copy(src, src + n_col, plane.begin() + (size_t) i*n_col);
        }

        ggml_backend_tensor_set(ids[e], plane.data(), 0, plane.size()*sizeof(int32_t));
    }

    ggml_backend_tensor_set(gate_mask, mask.data(), 0, mask.size()*sizeof(float));
}

bool llm_graph_input_dsv41_engram::can_reuse(const llm_graph_params & params) {
    // The tensors are shaped by n_tokens alone; their contents are refreshed by set_input on
    // every ubatch, so an identical token count can reuse the graph.
    return params.ubatch.n_tokens == n_tokens;
}
