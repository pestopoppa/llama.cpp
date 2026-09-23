// DS41-B8 -- DeepSeek-V4.1-Flash Engram n-gram hash.
//
// The expected row ids in test-dsv41-engram-hash-vectors.inc were produced by an INDEPENDENT
// reimplementation of inference/engram.py::NgramHashState.forward (see
// /mnt/raid0/llm/tmp/ds41-engram-graph/gen_test_vectors.py), not by this code, and were then
// cross-checked against antirez/ds4's ds4_engram_hash() @ 0aaea5a compiled standalone. All
// three agree bit-for-bit.
//
// The primes, multipliers, pad_id and compressed_vocab_size below are the ones
// antirez/deepseek-v4.1-flash-gguf actually ships under deepseek41.engram.*; the multipliers
// additionally reproduce engram.py::compute_hash_multipliers(layer_ids=(1,14),
// max_ngram_size=4, tokenizer_vocab_size=99092) exactly. Only the 10-entry token_map is
// synthetic, so it fits in a literal -- the hash only ever sees compressed ids.

#include "testing.h"

#include "llama.h"

#include "../src/llama-dsv41-engram.h"

#include <cstdint>
#include <vector>

#include "test-dsv41-engram-hash-vectors.inc"

static const uint32_t k_layer_ids[2] = { 1, 14 };
static const uint64_t k_rows[2]      = { 384006168ull, 384016682ull };

static const uint64_t k_multipliers[8] = {
    76632096046245ull,  4839876093313ull, 35959672319349ull, 73987337458391ull,
    67716810739261ull, 51510806800915ull, 30921347202721ull, 82619226485591ull,
};

static const uint32_t k_primes[48] = {
    16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183,
    16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321,
    16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463,
    16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627,
    16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769,
    16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
};

static llama_dsv41_engram_spec make_spec() {
    llama_dsv41_engram_spec spec;

    spec.n_engram = 2;
    spec.n_ngram  = 4;
    spec.n_head   = 8;
    spec.n_col    = 24;

    spec.pad_id                = 2;
    spec.compressed_vocab_size = 99092;

    spec.layer_ids.assign(k_layer_ids, k_layer_ids + 2);
    spec.rows.assign(k_rows, k_rows + 2);
    spec.multipliers.assign(k_multipliers, k_multipliers + 8);
    spec.primes.assign(k_primes, k_primes + 48);

    spec.token_map.assign(k_token_map, k_token_map + 10);

    // engram.py:149 -- exclusive cumsum of the layer's flattened primes
    spec.offsets.assign(spec.primes.size(), 0);
    for (uint32_t e = 0; e < spec.n_engram; ++e) {
        uint64_t acc = 0;
        for (uint32_t c = 0; c < spec.n_col; ++c) {
            spec.offsets[e*spec.n_col + c] = acc;
            acc += spec.primes[e*spec.n_col + c];
        }
    }

    return spec;
}

// Hash a whole sequence the way llm_graph_input_dsv41_engram::set_input does: commit every
// compressed id into the position cache first, then gather each token's lookback out of it.
static std::vector<int32_t> hash_sequence(
        const llama_dsv41_engram_spec & spec,
        llama_dsv41_engram_state      & state,
        llama_seq_id                    seq_id,
        llama_pos                       pos0,
        const int32_t                 * tokens,
        const int8_t                  * mask,
        size_t                          count) {
    for (size_t i = 0; i < count; ++i) {
        const int32_t cid = (mask && !mask[i])
            ? llama_dsv41_engram_spec::DEAD
            : spec.compress(tokens[i]);

        state.set(seq_id, pos0 + (llama_pos) i, cid);
    }

    std::vector<int32_t> out(count * spec.n_engram * spec.n_col);

    for (size_t i = 0; i < count; ++i) {
        const llama_pos pos = pos0 + (llama_pos) i;

        int32_t ngram[LLAMA_DSV41_ENGRAM_MAX_NGRAM];

        for (uint32_t j = 0; j < spec.n_ngram; ++j) {
            ngram[j] = pos >= (llama_pos) j
                ? state.get(seq_id, pos - (llama_pos) j)
                : llama_dsv41_engram_spec::DEAD;
        }

        spec.hash(ngram, out.data() + i*spec.n_engram*spec.n_col);
    }

    return out;
}

static void test_validate(testing & t) {
    llama_dsv41_engram_spec spec = make_spec();

    bool ok = true;
    try {
        spec.validate();
    } catch (const std::exception &) {
        ok = false;
    }
    t.assert_true("the shipped layout validates", ok);

    auto rejects = [](llama_dsv41_engram_spec s) {
        try {
            s.validate();
        } catch (const std::exception &) {
            return true;
        }
        return false;
    };

    // sum(primes[e]) must tile the table exactly (engram.py:92-93, ds4_engram.c:32-37)
    {
        llama_dsv41_engram_spec s = spec;
        s.rows[1] += 1;
        t.assert_true("a table whose primes do not sum to its row count is rejected", rejects(s));
    }

    // engram.py:82 -- values*2+1, so every multiplier is odd
    {
        llama_dsv41_engram_spec s = spec;
        s.multipliers[3] -= 1;
        t.assert_true("an even multiplier is rejected", rejects(s));
    }

    // engram.py:70-72 -- id*multiplier must not overflow int64
    {
        llama_dsv41_engram_spec s = spec;
        s.multipliers[0] = (uint64_t) INT64_MAX / s.compressed_vocab_size + 2;
        s.multipliers[0] |= 1;
        t.assert_true("a multiplier that could overflow the hash is rejected", rejects(s));
    }

    {
        llama_dsv41_engram_spec s = spec;
        s.token_map[4] = s.compressed_vocab_size;
        t.assert_true("a token_map entry outside the compressed vocab is rejected", rejects(s));
    }

    {
        llama_dsv41_engram_spec s = spec;
        s.pad_id = s.compressed_vocab_size;
        t.assert_true("a pad_id outside the compressed vocab is rejected", rejects(s));
    }

    t.assert_true("layer 1 is engram index 0",  spec.index_of_layer(1)  == 0);
    t.assert_true("layer 14 is engram index 1", spec.index_of_layer(14) == 1);
    t.assert_true("layer 0 carries no engram",  spec.index_of_layer(0)  == -1);
    t.assert_true("an out-of-range token id is DEAD, not an out-of-bounds read",
            spec.compress(10) == llama_dsv41_engram_spec::DEAD);
}

static void test_vectors(testing & t) {
    const llama_dsv41_engram_spec spec = make_spec();

    const size_t per_token = spec.n_engram * spec.n_col;

    // F1: a plain 8-token prefill. Positions 0,1,2 must block the deeper lookbacks.
    {
        llama_dsv41_engram_state state(4);
        const std::vector<int32_t> got = hash_sequence(spec, state, 0, 0, k_f1_tokens, nullptr, 8);

        bool eq = got.size() == 8*per_token;
        for (size_t i = 0; eq && i < got.size(); ++i) {
            eq = got[i] == k_f1_rows[i];
        }
        t.assert_true("F1: prefill matches the engram.py reference", eq);
    }

    // F2: position 3 is dead. The sticky blocked flag must pad every lookback at and past that
    // shift, for positions 3..6 (engram.py:173).
    {
        llama_dsv41_engram_state state(4);
        const std::vector<int32_t> got = hash_sequence(spec, state, 0, 0, k_f1_tokens, k_f2_mask, 8);

        bool eq = got.size() == 8*per_token;
        for (size_t i = 0; eq && i < got.size(); ++i) {
            eq = got[i] == k_f2_rows[i];
        }
        t.assert_true("F2: a dead token breaks every n-gram that spans it", eq);

        bool differs = false;
        for (size_t i = 3*per_token; i < 7*per_token; ++i) {
            differs = differs || k_f2_rows[i] != k_f1_rows[i];
        }
        t.assert_true("F2: positions 3..6 really do differ from the unmasked run", differs);

        bool same_before = true;
        for (size_t i = 0; i < 3*per_token; ++i) {
            same_before = same_before && k_f2_rows[i] == k_f1_rows[i];
        }
        t.assert_true("F2: positions before the dead token are untouched", same_before);

        bool same_after = true;
        for (size_t i = 7*per_token; i < 8*per_token; ++i) {
            same_after = same_after && k_f2_rows[i] == k_f1_rows[i];
        }
        t.assert_true("F2: the block expires after max_ngram_size-1 positions", same_after);
    }

    // F3: two token ids that normalize alike collapse to one compressed id and hash the same.
    {
        llama_dsv41_engram_state sa(4), sb(4);
        const std::vector<int32_t> a = hash_sequence(spec, sa, 0, 0, k_f3a_tokens, nullptr, 4);
        const std::vector<int32_t> b = hash_sequence(spec, sb, 0, 0, k_f3b_tokens, nullptr, 4);

        bool eq = a.size() == 4*per_token && a == b;
        for (size_t i = 0; eq && i < a.size(); ++i) {
            eq = a[i] == k_f3a_rows[i];
        }
        t.assert_true("F3: a token_map collision hashes identically", eq);
    }

    // Every row id must land inside its own table.
    {
        llama_dsv41_engram_state state(4);
        const std::vector<int32_t> got = hash_sequence(spec, state, 0, 0, k_f1_tokens, nullptr, 8);

        bool in_range = true;
        for (size_t i = 0; i < 8; ++i) {
            for (uint32_t e = 0; e < spec.n_engram; ++e) {
                for (uint32_t c = 0; c < spec.n_col; ++c) {
                    const int32_t r = got[i*per_token + e*spec.n_col + c];
                    in_range = in_range && r >= 0 && (uint64_t) r < spec.rows[e];
                }
            }
        }
        t.assert_true("every row id lands inside its own table", in_range);
    }
}

static void test_state(testing & t) {
    const llama_dsv41_engram_spec spec = make_spec();

    const size_t per_token = spec.n_engram * spec.n_col;

    // The prefill/decode split must not change a single row id: this is what engram.py's
    // [batch, max_seq_len] cache buys (engram.py:155-157) and what a per-seq tail would give up.
    {
        llama_dsv41_engram_state prefill(4);
        const std::vector<int32_t> whole = hash_sequence(spec, prefill, 0, 0, k_f1_tokens, nullptr, 8);

        llama_dsv41_engram_state stream(4);
        std::vector<int32_t> one_at_a_time;
        for (size_t i = 0; i < 8; ++i) {
            const std::vector<int32_t> step =
                hash_sequence(spec, stream, 0, (llama_pos) i, k_f1_tokens + i, nullptr, 1);
            one_at_a_time.insert(one_at_a_time.end(), step.begin(), step.end());
        }

        t.assert_true("decoding one token at a time reproduces the prefill exactly",
                whole == one_at_a_time);
    }

    // An MTP rollback re-decodes positions it already wrote. seq_rm drops them; the replay must
    // land on the same rows as the original pass.
    {
        llama_dsv41_engram_state state(4);
        const std::vector<int32_t> whole = hash_sequence(spec, state, 0, 0, k_f1_tokens, nullptr, 8);

        state.seq_rm(0, 5, -1);

        const std::vector<int32_t> replay =
            hash_sequence(spec, state, 0, 5, k_f1_tokens + 5, nullptr, 3);

        bool eq = replay.size() == 3*per_token;
        for (size_t i = 0; eq && i < replay.size(); ++i) {
            eq = replay[i] == whole[5*per_token + i];
        }
        t.assert_true("rollback then replay reproduces the original rows", eq);
    }

    // seq_rm must actually drop what it removed: a token decoded at a rolled-back position with
    // a different id must hash differently.
    {
        llama_dsv41_engram_state state(4);
        hash_sequence(spec, state, 0, 0, k_f1_tokens, nullptr, 8);
        state.seq_rm(0, 5, -1);

        t.assert_true("removed positions read back as DEAD",
                state.get(0, 5) == llama_dsv41_engram_spec::DEAD &&
                state.get(0, 7) == llama_dsv41_engram_spec::DEAD);
        t.assert_true("positions below the rollback point survive",
                state.get(0, 4) == spec.compress(k_f1_tokens[4]));
    }

    // A server slot that forks a sequence must carry the history with it, or the first three
    // tokens of the fork silently lose their n-grams.
    {
        llama_dsv41_engram_state a(4);
        const std::vector<int32_t> src = hash_sequence(spec, a, 0, 0, k_f1_tokens, nullptr, 8);

        a.seq_cp(0, 1, -1, -1);

        const std::vector<int32_t> fork =
            hash_sequence(spec, a, 1, 5, k_f1_tokens + 5, nullptr, 3);

        bool eq = fork.size() == 3*per_token;
        for (size_t i = 0; eq && i < fork.size(); ++i) {
            eq = fork[i] == src[5*per_token + i];
        }
        t.assert_true("seq_cp carries the n-gram history to the destination sequence", eq);
    }

    // Without the copy the fork degrades to pad-blocked n-grams -- bounded, but it must not be
    // mistaken for equality.
    {
        llama_dsv41_engram_state a(4);
        const std::vector<int32_t> src = hash_sequence(spec, a, 0, 0, k_f1_tokens, nullptr, 8);

        const std::vector<int32_t> fork =
            hash_sequence(spec, a, 1, 5, k_f1_tokens + 5, nullptr, 3);

        bool differs = false;
        for (size_t i = 0; i < per_token; ++i) {
            differs = differs || fork[i] != src[5*per_token + i];
        }
        t.assert_true("an uncopied fork does NOT silently match", differs);
    }

    // seq_keep drops every other sequence.
    {
        llama_dsv41_engram_state a(4);
        hash_sequence(spec, a, 0, 0, k_f1_tokens, nullptr, 8);
        hash_sequence(spec, a, 2, 0, k_f1_tokens, nullptr, 8);

        a.seq_keep(2);

        t.assert_true("seq_keep clears the other sequences",
                a.get(0, 4) == llama_dsv41_engram_spec::DEAD &&
                a.get(2, 4) == spec.compress(k_f1_tokens[4]));
    }

    // seq_add shifts the cache with the positions it renumbers.
    {
        llama_dsv41_engram_state a(4);
        hash_sequence(spec, a, 0, 0, k_f1_tokens, nullptr, 8);

        a.seq_add(0, 4, 8, 3);

        t.assert_true("seq_add moves the ids with their positions",
                a.get(0, 7) == spec.compress(k_f1_tokens[4]) &&
                a.get(0, 4) == llama_dsv41_engram_spec::DEAD &&
                a.get(0, 3) == spec.compress(k_f1_tokens[3]));
    }
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("validate", test_validate);
    t.test("vectors",  test_vectors);
    t.test("state",    test_state);

    return t.summary();
}
