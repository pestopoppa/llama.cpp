#include "llama-dsv41-engram.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model-loader.h"

#include "ggml.h"
#include "ggml-backend.h"

// Profiling build only: ggml-cpu.h carries the op-half counter types, and they only exist
// under the same -DGGML_CPU_PROF gate.  libllama takes no link dependency on the CPU backend
// either way -- the op half is reached through dlsym(RTLD_DEFAULT, ...), never a direct call.
#ifdef GGML_CPU_PROF
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// llama_dsv41_engram_prof
//
// Profiling build only (cmake -DGGML_CPU_PROF=ON).  Nothing below, and no string it holds,
// exists in the measured build.
//
#ifdef GGML_CPU_PROF

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#define LLAMA_ENGRAM_PROF_DLSYM 1
#endif

namespace {

// Simulated row-cache capacities, in rows. 4 M rows of the shipped 264-byte layout is a 1 GiB
// cache of packed rows (4 GiB if the cache holds dequantized f32), which is about as large as
// a row cache can get before it is competing with the page cache it would be sitting on.
const int64_t ENGRAM_PROF_CAP_ROWS[llama_dsv41_engram_prof::N_CAP] = {
    1 << 12, 1 << 16, 1 << 20, 1 << 22,
};

// The op profiler lives in the CPU backend, which may be a separately loaded module, so it is
// resolved at run time rather than linked. If it is not there the artifact still carries every
// host-side counter and reports op_profiler = "unavailable" -- it never fails the run.
typedef void (*engram_prof_read_fn)     (struct ggml_gather_e4m3_prof *);
typedef void (*engram_prof_set_level_fn)(int);

struct engram_prof_api {
    engram_prof_read_fn      read      = nullptr;
    engram_prof_set_level_fn set_level = nullptr;
};

const engram_prof_api & engram_api() {
    static const engram_prof_api api = [] {
        engram_prof_api a;
#ifdef LLAMA_ENGRAM_PROF_DLSYM
        a.read = (engram_prof_read_fn)
            dlsym(RTLD_DEFAULT, "ggml_gather_rows_e4m3_e8m0_prof_read");
        a.set_level = (engram_prof_set_level_fn)
            dlsym(RTLD_DEFAULT, "ggml_gather_rows_e4m3_e8m0_prof_set_level");
#endif
        return a;
    }();

    return api;
}

// last op-counter snapshot, differenced on every ubatch
ggml_gather_e4m3_prof g_engram_op_last;
bool                  g_engram_op_last_valid = false;

struct engram_prof_at_exit {
    ~engram_prof_at_exit();
};

} // namespace

llama_dsv41_engram_prof & llama_dsv41_engram_prof::get() {
    static llama_dsv41_engram_prof inst;

    // constructed after inst, therefore destroyed BEFORE it: the exit flush never touches a
    // destroyed singleton
    static engram_prof_at_exit guard;

    (void) guard;

    return inst;
}

namespace {

engram_prof_at_exit::~engram_prof_at_exit() {
    llama_dsv41_engram_prof::get().flush();
}

// Fold the op counters accumulated since the previous ubatch into `phase`. The gathers for
// ubatch N run between set_input(N) and set_input(N+1), so this delta belongs to the phase of
// the PREVIOUS ubatch, not the current one.
void engram_prof_drain_op(llama_dsv41_engram_prof & p, int phase) {
    if (engram_api().read == nullptr) {
        return;
    }

    ggml_gather_e4m3_prof cur;

    engram_api().read(&cur);

    for (int i = 0; i < cur.n_tables && i < GGML_GATHER_E4M3_PROF_MAX_TABLES; ++i) {
        if (p.op_slot_layer[i] < 0) {
            // Map the op's slot to an engram layer by the table's row count. The two shipped
            // tables differ (384,006,168 vs 384,016,682), and the row count is the only thing
            // the op exposes that the port can match on without the op learning what a layer
            // is. A tie would leave both slots reported separately, never merged.
            for (uint32_t e = 0; e < p.n_engram; ++e) {
                if ((uint64_t) cur.tables[i].n_table_rows == p.op_rows[e]) {
                    p.op_slot_layer[i] = (int) e;
                    break;
                }
            }
        }

        if (phase < 0 || !g_engram_op_last_valid) {
            continue;
        }

        const ggml_gather_e4m3_prof_table & a = g_engram_op_last.tables[i];
        const ggml_gather_e4m3_prof_table & b = cur.tables[i];

        if (i < g_engram_op_last.n_tables && a.table != b.table) {
            continue; // slots are append-only, so this should not happen; skip rather than lie
        }

        const ggml_gather_e4m3_prof_table zero = {};
        const ggml_gather_e4m3_prof_table & base = i < g_engram_op_last.n_tables ? a : zero;

        llama_dsv41_engram_prof::op_phase_stats & d =
            p.opstats[(size_t) phase*GGML_GATHER_E4M3_PROF_MAX_TABLES + i];

        d.n_calls        += (uint64_t) (b.n_calls        - base.n_calls);
        d.n_rows         += (uint64_t) (b.n_rows         - base.n_rows);
        d.n_bytes_src    += (uint64_t) (b.n_bytes_src    - base.n_bytes_src);
        d.us_span_ith0   += (uint64_t) (b.us_span_ith0   - base.us_span_ith0);
        d.us_cpu         += (uint64_t) (b.us_cpu         - base.us_cpu);
        d.n_thread_spans += (uint64_t) (b.n_thread_spans - base.n_thread_spans);
        d.minflt         += (uint64_t) (b.minflt         - base.minflt);
        d.majflt         += (uint64_t) (b.majflt         - base.majflt);
    }

    g_engram_op_last       = cur;
    g_engram_op_last_valid = true;
}

} // namespace

void llama_dsv41_engram_prof::configure(const llama_dsv41_engram_spec & spec) {
    if (configured) {
        return;
    }

    configured = true;

    // The artifact is an env-var-named JSON file written once per process at exit, exactly as
    // GGML_CPU_PROF_JSON_FILE (ggml-cpu.c) and LLAMA_HOST_PROF_JSON_FILE (llama-graph.cpp) are.
    // LLAMA_ENGRAM_PROFILE is kept as a legacy alias for the path SPEC.md documents.
    const char * p = getenv("LLAMA_ENGRAM_PROF_JSON_FILE");

    if (p == nullptr || *p == '\0') {
        p = getenv("LLAMA_ENGRAM_PROFILE");
    }

    if (p == nullptr || *p == '\0') {
        level = 0;
        return;
    }

    path = p;

    const char * l = getenv("LLAMA_ENGRAM_PROFILE_LEVEL");

    level = l ? atoi(l) : 1;

    if (level < 1) {
        level = 1;
    }

    // Level 2 turns on the op's fault attribution. An explicit GGML_GATHER_PROF wins, so that
    // the op's level can still be pinned independently of this one.
    if (engram_api().set_level && getenv("GGML_GATHER_PROF") == nullptr) {
        engram_api().set_level(level >= 2 ? 2 : 1);
    }

    n_engram = spec.n_engram;
    n_col    = spec.n_col;

    op_rows.assign(spec.rows.begin(), spec.rows.end());

    lstate .resize(n_engram);
    lstats .resize((size_t) N_PHASE*n_engram);
    opstats.resize((size_t) N_PHASE*GGML_GATHER_E4M3_PROF_MAX_TABLES);

    op_slot_layer.assign(GGML_GATHER_E4M3_PROF_MAX_TABLES, -1);

    for (uint32_t e = 0; e < n_engram; ++e) {
        layer_state & st = lstate[e];

        st.prev_row        .assign(n_col,  -1);
        st.index_hist      .assign(N_HIST,  0);
        st.col_rows        .assign(n_col,   0);
        st.col_same_as_prev.assign(n_col,   0);

        for (int k = 0; k < N_CAP; ++k) {
            st.cache_tag[k].assign((size_t) ENGRAM_PROF_CAP_ROWS[k], -1);
        }
    }

    LLAMA_LOG_INFO("%s: engram profile: level %d -> %s (op profiler %s)\n",
            __func__, level, path.c_str(),
            engram_api().read ? "resolved" : "UNAVAILABLE");
}

void llama_dsv41_engram_prof::observe(
        const llama_dsv41_engram_spec & spec,
        uint32_t n_tokens, uint32_t n_gated,
        const int32_t * rows, int64_t us_hash_) {
    if (!enabled() || n_tokens == 0) {
        return;
    }

    const int64_t t_begin = ggml_time_us();

    // A one-token ubatch is a decode step. Anything wider is prefill (or a draft batch), and
    // the two have completely different gather shapes, so they are never pooled.
    const int phase = n_tokens == 1 ? PHASE_DECODE : PHASE_PREFILL;

    engram_prof_drain_op(*this, phase_last);

    if (phase == PHASE_DECODE && phase_last == PHASE_DECODE && t_last_set_input > 0) {
        // set_input(t) -> set_input(t+1) is one whole decode step, sampling included. It is
        // the denominator for "how much of a token is the gather", and it is wall time on the
        // caller's thread, not a model of the step.
        us_step_span += (uint64_t) (t_begin - t_last_set_input);
        n_step_span  += 1;
    }

    t_last_set_input = t_begin;
    phase_last       = phase;

    us_hash += (uint64_t) us_hash_;

    std::vector<int32_t> tok_ids(n_col);
    std::vector<int32_t> ub_ids;

    ub_ids.reserve((size_t) n_tokens*n_col);

    for (uint32_t e = 0; e < n_engram; ++e) {
        layer_state       & st = lstate[e];
        layer_phase_stats & ls = lstats[(size_t) phase*n_engram + e];

        ls.n_calls  += 1;
        ls.n_tokens += n_tokens;
        ls.n_gated  += n_gated;
        ls.rows     += (uint64_t) n_tokens*n_col;

        const uint64_t n_table_rows = spec.rows[e];

        ub_ids.clear();

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const int32_t * r = rows + ((size_t) i*n_engram + e)*n_col;

            for (uint32_t c = 0; c < n_col; ++c) {
                const int32_t row = r[c];

                st.col_rows[c] += 1;

                if (st.prev_row[c] == row) {
                    ls.rows_same_as_prev  += 1;
                    st.col_same_as_prev[c] += 1;
                }

                st.prev_row[c] = row;

                // uniformity over the whole table, not over a column's own prime bucket: the
                // question is whether the 48 addresses of a token land anywhere near each
                // other, which is what decides whether a locality lever can exist at all
                const uint64_t b = n_table_rows > 0
                    ? (uint64_t) row*N_HIST/n_table_rows
                    : 0;

                st.index_hist[b < N_HIST ? b : N_HIST - 1] += 1;

                for (int k = 0; k < N_CAP; ++k) {
                    const size_t idx = (size_t) row & (size_t) (ENGRAM_PROF_CAP_ROWS[k] - 1);

                    if (st.cache_tag[k][idx] == row) {
                        ls.cache_hit[k] += 1;
                    } else {
                        st.cache_tag[k][idx] = row;
                    }
                }

                ub_ids.push_back(row);
            }

            tok_ids.assign(r, r + n_col);

            std::sort(tok_ids.begin(), tok_ids.end());

            ls.rows_uniq_in_token +=
                (uint64_t) (std::unique(tok_ids.begin(), tok_ids.end()) - tok_ids.begin());
        }

        std::sort(ub_ids.begin(), ub_ids.end());

        ls.rows_uniq_in_ubatch +=
            (uint64_t) (std::unique(ub_ids.begin(), ub_ids.end()) - ub_ids.begin());
    }

    us_observe += (uint64_t) (ggml_time_us() - t_begin);
}

void llama_dsv41_engram_prof::flush() {
    if (!enabled() || flushed) {
        return;
    }

    flushed = true;

    engram_prof_drain_op(*this, phase_last);

    FILE * f = fopen(path.c_str(), "w");

    if (f == nullptr) {
        LLAMA_LOG_WARN("%s: engram profile: cannot write %s\n", __func__, path.c_str());
        return;
    }

    ggml_gather_e4m3_prof op;

    memset(&op, 0, sizeof(op));

    if (engram_api().read) {
        engram_api().read(&op);
    }

    const char * phase_name[N_PHASE] = { "prefill", "decode" };

    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"dsv41-engram-profile/1\",\n");
    fprintf(f, "  \"kind\": \"observation\",\n");
    fprintf(f, "  \"level\": %d,\n", level);
    fprintf(f, "  \"op_profiler\": \"%s\",\n", engram_api().read ? "resolved" : "unavailable");
    fprintf(f, "  \"op_level\": %d,\n", op.level);
    fprintf(f, "  \"fault_source\": \"%s\",\n", op.fault_source == 1 ? "getrusage_thread" : "none");
    fprintf(f, "  \"n_engram\": %u,\n", n_engram);
    fprintf(f, "  \"n_col\": %u,\n", n_col);
    fprintf(f, "  \"host\": {\n");
    fprintf(f, "    \"us_hash\": %" PRIu64 ",\n", us_hash);
    fprintf(f, "    \"us_observe\": %" PRIu64 ",\n", us_observe);
    fprintf(f, "    \"us_decode_step_span\": %" PRIu64 ",\n", us_step_span);
    fprintf(f, "    \"n_decode_step_span\": %" PRIu64 "\n", n_step_span);
    fprintf(f, "  },\n");

    fprintf(f, "  \"layers\": [\n");

    for (uint32_t e = 0; e < n_engram; ++e) {
        const layer_state & st = lstate[e];

        fprintf(f, "    {\n");
        fprintf(f, "      \"index\": %u,\n", e);
        fprintf(f, "      \"table_rows\": %" PRIu64 ",\n", e < op_rows.size() ? op_rows[e] : 0);

        for (int ph = 0; ph < N_PHASE; ++ph) {
            const layer_phase_stats & ls = lstats[(size_t) ph*n_engram + e];

            fprintf(f, "      \"%s\": {\n", phase_name[ph]);
            fprintf(f, "        \"n_calls\": %" PRIu64 ",\n",            ls.n_calls);
            fprintf(f, "        \"n_tokens\": %" PRIu64 ",\n",           ls.n_tokens);
            fprintf(f, "        \"n_gated\": %" PRIu64 ",\n",            ls.n_gated);
            fprintf(f, "        \"rows\": %" PRIu64 ",\n",               ls.rows);
            fprintf(f, "        \"rows_uniq_in_token\": %" PRIu64 ",\n", ls.rows_uniq_in_token);
            fprintf(f, "        \"rows_uniq_in_ubatch\": %" PRIu64 ",\n",ls.rows_uniq_in_ubatch);
            fprintf(f, "        \"rows_same_as_prev\": %" PRIu64 ",\n",  ls.rows_same_as_prev);
            fprintf(f, "        \"cache_hit\": [");

            for (int k = 0; k < N_CAP; ++k) {
                fprintf(f, "%s%" PRIu64, k ? ", " : "", ls.cache_hit[k]);
            }

            fprintf(f, "]\n      },\n");
        }

        fprintf(f, "      \"cache_capacity_rows\": [");

        for (int k = 0; k < N_CAP; ++k) {
            fprintf(f, "%s%" PRId64, k ? ", " : "", ENGRAM_PROF_CAP_ROWS[k]);
        }

        fprintf(f, "],\n");

        fprintf(f, "      \"col_rows\": [");

        for (uint32_t c = 0; c < n_col; ++c) {
            fprintf(f, "%s%" PRIu64, c ? ", " : "", st.col_rows[c]);
        }

        fprintf(f, "],\n");

        fprintf(f, "      \"col_same_as_prev\": [");

        for (uint32_t c = 0; c < n_col; ++c) {
            fprintf(f, "%s%" PRIu64, c ? ", " : "", st.col_same_as_prev[c]);
        }

        fprintf(f, "],\n");

        fprintf(f, "      \"index_hist\": [");

        for (int b = 0; b < N_HIST; ++b) {
            fprintf(f, "%s%" PRIu64, b ? ", " : "", st.index_hist[b]);
        }

        fprintf(f, "]\n    }%s\n", e + 1 < n_engram ? "," : "");
    }

    fprintf(f, "  ],\n");

    fprintf(f, "  \"op\": {\n");
    fprintf(f, "    \"n_calls_unattributed\": %" PRId64 ",\n", op.n_calls_unattributed);
    fprintf(f, "    \"tables\": [\n");

    for (int i = 0; i < op.n_tables && i < GGML_GATHER_E4M3_PROF_MAX_TABLES; ++i) {
        fprintf(f, "      {\n");
        fprintf(f, "        \"slot\": %d,\n", i);
        fprintf(f, "        \"engram_layer\": %d,\n", op_slot_layer[i]);
        fprintf(f, "        \"table_rows\": %" PRId64 ",\n", op.tables[i].n_table_rows);
        fprintf(f, "        \"row_bytes\": %" PRId64 ",\n",  op.tables[i].row_bytes);

        for (int ph = 0; ph < N_PHASE; ++ph) {
            const op_phase_stats & d =
                opstats[(size_t) ph*GGML_GATHER_E4M3_PROF_MAX_TABLES + i];

            fprintf(f, "        \"%s\": {", phase_name[ph]);
            fprintf(f, "\"n_calls\": %" PRIu64 ", ",        d.n_calls);
            fprintf(f, "\"n_rows\": %" PRIu64 ", ",         d.n_rows);
            fprintf(f, "\"n_bytes_src\": %" PRIu64 ", ",    d.n_bytes_src);
            fprintf(f, "\"us_span_ith0\": %" PRIu64 ", ",   d.us_span_ith0);
            fprintf(f, "\"us_cpu\": %" PRIu64 ", ",         d.us_cpu);
            fprintf(f, "\"n_thread_spans\": %" PRIu64 ", ", d.n_thread_spans);
            fprintf(f, "\"minflt\": %" PRIu64 ", ",         d.minflt);
            fprintf(f, "\"majflt\": %" PRIu64 "},\n",       d.majflt);
        }

        fprintf(f, "        \"us_hist\": [");

        for (int b = 0; b < GGML_GATHER_E4M3_PROF_NBUCKET; ++b) {
            fprintf(f, "%s%" PRId64, b ? ", " : "", op.tables[i].us_hist[b]);
        }

        fprintf(f, "]\n      }%s\n", i + 1 < op.n_tables ? "," : "");
    }

    fprintf(f, "    ]\n  }\n}\n");

    fclose(f);

    LLAMA_LOG_INFO("%s: engram profile written to %s\n", __func__, path.c_str());
}

#endif // GGML_CPU_PROF

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

#ifdef GGML_CPU_PROF
    // Profiling build only: in the measured build there is no profiler object, no clock read
    // here and no gated-token counter.
    auto & prof = llama_dsv41_engram_prof::get();

    prof.configure(*spec);

    const int64_t t_hash0 = prof.enabled() ? ggml_time_us() : 0;

    uint32_t n_gated = 0;
#endif

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

#ifdef GGML_CPU_PROF
        n_gated += mask[i] == 0.0f ? 1 : 0;
#endif

        spec->hash(ngram, rows.data() + (size_t) i*n_eng*n_col);
    }

#ifdef GGML_CPU_PROF
    // The row ids for this ubatch are final here, and the gathers that consume them have not
    // run yet -- which is what makes this the point where the op's counters can be split into
    // per-ubatch, per-phase deltas.
    prof.observe(*spec, n_tok, n_gated, rows.data(),
            prof.enabled() ? ggml_time_us() - t_hash0 : 0);
#endif

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
