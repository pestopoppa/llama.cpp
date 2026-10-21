#pragma once

// Chunk store for chunk-KV reuse (phase P1): exported K/V blobs keyed by CONTENT only.
//
//   key = H(model fingerprint, token ids)           -- no predecessor, no position: a chunk is found wherever it is
//                                                      placed next time (LMCache SegmentTokenDatabase semantics,
//                                                      token_database.py:452-560)
//   chain_key(parent, key)                           -- the separate chained key for exact-prefix reuse, kept apart
//
// The model fingerprint is an opaque string supplied by the caller. It MUST cover everything that changes the bytes of
// a blob: GGUF path + size + mtime (or sha), the kernel version string, K/V cache types, rope parameters / n_ctx_orig
// / YaRN, and the chunk blob format version. A stack reload then cannot serve a stale blob (CLAUDE.md stale-artifact rule).
//
// A 128-bit hash is not trusted on its own: a hit is only returned after the stored token ids compare EQUAL to the
// request's, so a collision degrades to a miss, never to wrong KV.
//
// Eviction is cost-aware LRU: score = (hits + 1) * n_tokens / bytes, lowest score goes first, ties by oldest use;
// pinned entries are never evicted. Budget 0 = feature off: put() stores nothing and get() always misses.
//
// Header only and free of llama.cpp dependencies, so it is unit-testable in isolation (tests/test-server-chunk-store.cpp).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace server_chunk {

using token  = int32_t;
using tokens = std::vector<token>;

struct key {
    uint64_t hi = 0;
    uint64_t lo = 0;

    bool operator==(const key & o) const { return hi == o.hi && lo == o.lo; }
    bool operator!=(const key & o) const { return !(*this == o); }
};

struct key_hash {
    size_t operator()(const key & k) const { return (size_t) (k.hi ^ (k.lo * 0x9E3779B97F4A7C15ull)); }
};

namespace detail {

inline uint64_t mix(uint64_t x) { // splitmix64 finalizer
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// two independent 64-bit streams over the same words (different seeds and multipliers)
struct hasher {
    uint64_t a = 0x243F6A8885A308D3ull;
    uint64_t b = 0x13198A2E03707344ull;
    uint64_t n = 0;

    void word(uint64_t w) {
        a = mix(a ^ w) * 0xFF51AFD7ED558CCDull;
        b = mix(b + w * 0xC4CEB9FE1A85EC53ull) ^ (b >> 17);
        n++;
    }

    key done() const { return key{ mix(a ^ n), mix(b + n * 0x9E3779B97F4A7C15ull) }; }
};

} // namespace detail

inline key make_key(const std::string & model_fp, const token * t, size_t n) {
    detail::hasher h;
    h.word(model_fp.size());
    for (size_t i = 0; i < model_fp.size(); i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, model_fp.data() + i, std::min<size_t>(8, model_fp.size() - i));
        h.word(w);
    }
    h.word(n);
    for (size_t i = 0; i < n; i++) {
        h.word((uint64_t) (uint32_t) t[i]);
    }
    return h.done();
}

inline key make_key(const std::string & model_fp, const tokens & t) {
    return make_key(model_fp, t.data(), t.size());
}

// chained key: the identity of "this chunk after that history" (exact-prefix reuse / KPF), distinct from the content key
inline key chain_key(const key & parent, const key & content) {
    detail::hasher h;
    h.word(0xC4A1ull);
    h.word(parent.hi);
    h.word(parent.lo);
    h.word(content.hi);
    h.word(content.lo);
    return h.done();
}

struct entry {
    key                  k;
    std::string          fp;      // model fingerprint the blob was produced under
    tokens               toks;    // the chunk's token ids, compared on every hit
    std::vector<uint8_t> blob;    // llama_chunk_kv_export output (positions 0..n-1 of an isolated encode)
    bool                 pinned = false;
    uint64_t             hits = 0;
    uint64_t             last_use = 0;

    size_t bytes() const { return blob.size() + toks.size() * sizeof(token) + fp.size(); }
};

struct stats {
    uint64_t hits       = 0;
    uint64_t misses     = 0;
    uint64_t collisions = 0; // key matched but tokens differed (served as a miss)
    uint64_t puts       = 0;
    uint64_t rejected   = 0; // put() refused: store off, oversize, or only pinned entries left to evict
    uint64_t evictions  = 0;
};

class store {
public:
    explicit store(size_t budget_bytes = 0) : budget(budget_bytes) {}

    bool enabled() const { return budget > 0; }

    size_t budget_bytes() const { return budget; }
    size_t used_bytes()   const { return used; }
    size_t size()         const { return map.size(); }
    const stats & get_stats() const { return st; }

    void set_budget(size_t b) {
        budget = b;
        make_room(0);
    }

    // returns the stored entry or nullptr. The token ids are compared, not just the hash.
    entry * get(const std::string & model_fp, const tokens & toks) {
        if (!enabled()) {
            return nullptr;
        }
        return lookup(make_key(model_fp, toks), model_fp, toks);
    }

    // same, with a caller-supplied key (the client-provided "key" of a request, or a test forcing a collision)
    entry * lookup(const key & k, const std::string & model_fp, const tokens & toks) {
        if (!enabled()) {
            return nullptr;
        }
        auto it = map.find(k);
        if (it == map.end()) {
            st.misses++;
            return nullptr;
        }
        entry & e = *it->second;
        if (e.fp != model_fp || e.toks != toks) {
            st.collisions++;
            st.misses++;
            return nullptr;
        }
        e.hits++;
        e.last_use = ++clock;
        st.hits++;
        return &e;
    }

    // insert (or replace) a chunk. Returns false when it cannot be stored.
    bool put(const std::string & model_fp, const tokens & toks, std::vector<uint8_t> blob, bool pin = false) {
        return put_key(make_key(model_fp, toks), model_fp, toks, std::move(blob), pin);
    }

    bool put_key(const key & k, const std::string & model_fp, const tokens & toks, std::vector<uint8_t> blob, bool pin = false) {
        if (!enabled() || blob.empty() || toks.empty()) {
            st.rejected++;
            return false;
        }

        entry e;
        e.k      = k;
        e.fp     = model_fp;
        e.toks   = toks;
        e.blob   = std::move(blob);
        e.pinned = pin;

        const size_t need = e.bytes();
        if (need > budget) {
            st.rejected++;
            return false;
        }

        // replacing an existing key frees its bytes first
        auto old = map.find(k);
        if (old != map.end()) {
            e.hits = old->second->hits;
            erase(old);
        }

        if (!make_room(need)) {
            st.rejected++;
            return false;
        }

        e.last_use = ++clock;
        used += need;
        lru.push_back(std::move(e));
        auto last = std::prev(lru.end());
        map[k] = last;
        st.puts++;
        return true;
    }

    bool pin(const key & k, bool value) {
        auto it = map.find(k);
        if (it == map.end()) {
            return false;
        }
        it->second->pinned = value;
        return true;
    }

    bool erase_key(const key & k) {
        auto it = map.find(k);
        if (it == map.end()) {
            return false;
        }
        erase(it);
        return true;
    }

    void clear() {
        map.clear();
        lru.clear();
        used = 0;
    }

private:
    using list_t = std::list<entry>;
    using map_t  = std::unordered_map<key, list_t::iterator, key_hash>;

    static double score(const entry & e) {
        const size_t b = e.bytes();
        return b ? (double) (e.hits + 1) * (double) e.toks.size() / (double) b : 0.0;
    }

    void erase(map_t::iterator it) {
        used -= it->second->bytes();
        lru.erase(it->second);
        map.erase(it);
    }

    // evict lowest-score unpinned entries until `need` more bytes fit; false if pinned entries block it
    bool make_room(size_t need) {
        while (used + need > budget) {
            list_t::iterator victim = lru.end();
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                if (it->pinned) {
                    continue;
                }
                if (victim == lru.end() || score(*it) < score(*victim) ||
                    (score(*it) == score(*victim) && it->last_use < victim->last_use)) {
                    victim = it;
                }
            }
            if (victim == lru.end()) {
                return false;
            }
            map.erase(victim->k);
            used -= victim->bytes();
            lru.erase(victim);
            st.evictions++;
        }
        return true;
    }

    size_t   budget = 0;
    size_t   used   = 0;
    uint64_t clock  = 0;
    list_t   lru;
    map_t    map;
    stats    st;
};

} // namespace server_chunk
