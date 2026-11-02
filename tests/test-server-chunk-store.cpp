// Unit tests for the llama-server chunk store (tools/server/server-chunk-store.h), chunk-KV reuse phase P1.
#undef NDEBUG
#include "server-chunk-store.h"

#include <cstdio>
#include <cstdlib>
#include <set>

using namespace server_chunk;

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static tokens run(int base, int n) {
    tokens t(n);
    for (int i = 0; i < n; i++) t[i] = base + i;
    return t;
}
static std::vector<uint8_t> blob_of(size_t n, uint8_t v = 1) { return std::vector<uint8_t>(n, v); }

int main() {
    const std::string fp = "model-A|gguf=123|k=f16|v=f16|rope=10000/1.0";

    // key: content only
    {
        CHECK(make_key(fp, run(0, 64)) == make_key(fp, run(0, 64)));
        CHECK(make_key(fp, run(0, 64)) != make_key(fp, run(1, 64)));
        CHECK(make_key(fp, run(0, 64)) != make_key(fp, run(0, 63)));
        CHECK(make_key(fp, run(0, 64)) != make_key("model-B", run(0, 64)));
        // the fingerprint's last bytes matter (not only its first 8-byte words)
        CHECK(make_key("abcdefghijk", run(0, 8)) != make_key("abcdefghijl", run(0, 8)));
        // a content key has no parent; the chained key does
        const key a = make_key(fp, run(0, 64));
        const key b = make_key(fp, run(100, 64));
        CHECK(chain_key(a, b) != chain_key(b, a));
        CHECK(chain_key(a, b) != b);
        // no collisions over a modest sweep
        std::set<std::pair<uint64_t, uint64_t>> seen;
        for (int i = 0; i < 20000; i++) {
            const key k = make_key(fp, run(i, 16 + i % 7));
            CHECK(seen.insert({ k.hi, k.lo }).second);
        }
    }

    // budget 0 = off
    {
        store s(0);
        CHECK(!s.enabled());
        CHECK(!s.put(fp, run(0, 64), blob_of(1000)));
        CHECK(s.get(fp, run(0, 64)) == nullptr);
        CHECK(s.size() == 0 && s.used_bytes() == 0);
    }

    // put / get, exact tokens, fingerprint isolation
    {
        store s(1 << 20);
        CHECK(s.put(fp, run(0, 64), blob_of(1000, 7)));
        entry * e = s.get(fp, run(0, 64));
        CHECK(e != nullptr && e->blob.size() == 1000 && e->blob[0] == 7 && e->hits == 1);
        CHECK(s.get("model-B", run(0, 64)) == nullptr);
        CHECK(s.get(fp, run(0, 65)) == nullptr);
        CHECK(s.get_stats().hits == 1);

        // forced key collision: same key, different tokens -> a miss, never wrong KV
        const key k = make_key(fp, run(0, 64));
        CHECK(s.lookup(k, fp, run(5, 64)) == nullptr);
        CHECK(s.get_stats().collisions == 1);

        // re-put replaces in place and keeps the hit count
        CHECK(s.put(fp, run(0, 64), blob_of(500, 9)));
        CHECK(s.size() == 1);
        e = s.get(fp, run(0, 64));
        CHECK(e && e->blob.size() == 500 && e->blob[0] == 9 && e->hits == 2);
        CHECK(s.used_bytes() == e->bytes());
    }

    // eviction: budget respected, cost-aware, pinned survive
    {
        // each entry: 1000 B blob + 64 tokens*4 + fp
        store s(5000);
        const size_t one = blob_of(1000).size() + 64*sizeof(token) + fp.size();
        const int fit = (int) (5000 / one);
        CHECK(fit >= 2);

        for (int i = 0; i < fit; i++) CHECK(s.put(fp, run(i*1000, 64), blob_of(1000)));
        CHECK(s.size() == (size_t) fit);
        CHECK(s.used_bytes() <= 5000);

        // make entry 0 hot, then overflow: a cold entry is evicted, the hot one stays
        for (int h = 0; h < 5; h++) CHECK(s.get(fp, run(0, 64)) != nullptr);
        CHECK(s.put(fp, run(900000, 64), blob_of(1000)));
        CHECK(s.get(fp, run(0, 64)) != nullptr);
        CHECK(s.used_bytes() <= 5000);
        CHECK(s.get_stats().evictions >= 1);

        // pin entry 0, fill the rest with churn: it never goes
        CHECK(s.pin(make_key(fp, run(0, 64)), true));
        for (int i = 0; i < 50; i++) s.put(fp, run(2000000 + i*100, 64), blob_of(1000));
        CHECK(s.get(fp, run(0, 64)) != nullptr);
        CHECK(s.used_bytes() <= 5000);

        // an oversize blob is rejected without evicting anything
        const size_t before = s.size();
        CHECK(!s.put(fp, run(7, 64), blob_of(6000)));
        CHECK(s.size() == before);

        // everything pinned: nothing can be evicted, so the put is refused
        store p(2 * one);
        CHECK(p.put(fp, run(1, 64), blob_of(1000), true));
        CHECK(p.put(fp, run(2, 64), blob_of(1000), true));
        CHECK(!p.put(fp, run(3, 64), blob_of(1000)));
        CHECK(p.size() == 2);
        CHECK(p.get_stats().rejected >= 1);

        // shrinking the budget evicts down to it
        store q(10 * one);
        for (int i = 0; i < 8; i++) CHECK(q.put(fp, run(i*1000, 64), blob_of(1000)));
        q.set_budget(3 * one);
        CHECK(q.used_bytes() <= 3 * one);
        CHECK(q.size() <= 3);

        // score = hits*tokens/bytes: a long cheap chunk outlives a short big one at equal hits
        store r(5000);
        CHECK(r.put(fp, run(0, 512), blob_of(1000)));      // 512 tok / ~3.1 KB ... high value per byte
        CHECK(r.put(fp, run(1000, 8), blob_of(1000)));     // 8 tok / ~1 KB ... low value per byte
        CHECK(r.put(fp, run(2000, 8), blob_of(1000)));     // forces an eviction
        CHECK(r.get(fp, run(0, 512)) != nullptr);
    }

    // clear
    {
        store s(1 << 16);
        s.put(fp, run(0, 8), blob_of(100));
        s.clear();
        CHECK(s.size() == 0 && s.used_bytes() == 0);
    }

    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
