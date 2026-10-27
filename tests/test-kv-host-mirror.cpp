// HSF-5.P host KV mirror: CPU unit tests for the host-side logic (no device, no model).
//
//   - spec parsing
//   - the k-step recurrent checkpoint schedule (staggering, at most one checkpoint per step for n_seq <= k)
//   - the replay bound (a lag save replays at most k decode steps)
//   - unit coalescing and fragment construction (piece splitting, contiguous-fragment coalescing)
//   - per-cell validity under in-flight copies (a cell rewritten during its copy is never served)
//   - a fake-device round trip: decode-step writes -> async tail copies -> save from the mirror ->
//     restore through the pinned-staging reader, reproducing the device bytes exactly
//   - checkpoint capture: a captured stream replays byte-identically to a direct write

#include "../src/llama-kv-host-mirror.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llama_kvm;

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

//
// a deferred fake device: copies happen only when pumped, so tests can interleave "device" writes with
// in-flight copies the way a side stream does
//

struct deferred_dev {
    struct job { uint64_t t; std::vector<ggml_kv_mirror_frag> f; ggml_kv_mirror_dir dir; };
    std::deque<job> q;
    uint64_t next = 1;
    uint64_t done_upto = 0;

    void pump_one() {
        if (q.empty()) return;
        auto j = q.front(); q.pop_front();
        for (const auto & f : j.f) {
            if (j.dir == GGML_KV_MIRROR_D2H) memcpy(f.host, f.dev, f.bytes);
            else                             memcpy(f.dev, f.host, f.bytes);
        }
        done_upto = j.t;
    }
    void pump_all() { while (!q.empty()) pump_one(); }
};

static ggml_kv_mirror_dev_t dd_init(ggml_backend_t) { return (ggml_kv_mirror_dev_t) new deferred_dev; }
static void dd_free(ggml_kv_mirror_dev_t d) { delete (deferred_dev *) d; }
static bool dd_reg(ggml_kv_mirror_dev_t, void * h, size_t, void ** a) { *a = h; return true; }
static void dd_unreg(ggml_kv_mirror_dev_t, void *) {}
static uint64_t dd_copy(ggml_kv_mirror_dev_t d, const ggml_kv_mirror_frag * f, size_t n, ggml_kv_mirror_dir dir, ggml_kv_mirror_path) {
    auto * s = (deferred_dev *) d;
    if (n == 0) return 0;
    deferred_dev::job j{ s->next++, std::vector<ggml_kv_mirror_frag>(f, f + n), dir };
    s->q.push_back(j);
    return j.t;
}
static bool dd_done(ggml_kv_mirror_dev_t d, uint64_t t) { return ((deferred_dev *) d)->done_upto >= t; }
static void dd_wait(ggml_kv_mirror_dev_t d, uint64_t t) { auto * s = (deferred_dev *) d; while (s->done_upto < t && !s->q.empty()) s->pump_one(); }
static void dd_bwait(ggml_kv_mirror_dev_t, uint64_t) {}

static const ggml_kv_mirror_ops dd_ops = { GGML_KV_MIRROR_OPS_VERSION, dd_init, dd_free, dd_reg, dd_unreg, dd_copy, dd_done, dd_wait, dd_bwait };

//
// tests
//

static void test_parse() {
    params p;
    std::string err;
    CHECK(parse_params("", p, err) && p.k == 6 && p.numa_node == 3 && p.mode == save_mode::lag);
    CHECK(parse_params("on", p, err) && p.k == 6);
    CHECK(parse_params("k=13,node=-1,mode=fresh,piece_mib=64,path=kernel", p, err));
    CHECK(p.k == 13 && p.numa_node == -1 && p.mode == save_mode::fresh && p.piece_bytes == (size_t) 64 << 20 && p.path == GGML_KV_MIRROR_PATH_KERNEL);
    CHECK(!parse_params("k=0", p, err));
    CHECK(!parse_params("bogus=1", p, err));
    CHECK(!parse_params("mode=sometimes", p, err));
    CHECK(!parse_params("k", p, err));
    params q;
    CHECK(parse_params(to_string(p), q, err) && q.k == p.k && q.numa_node == p.numa_node && q.mode == p.mode && q.path == p.path);
}

// every sequence decodes every step; checkpoints issued at the boundary as the bind does
static void test_schedule(uint32_t k, uint32_t n_seq) {
    ckpt_schedule s;
    s.init(k, n_seq);
    const size_t max_n = (n_seq + k - 1) / k;

    std::vector<int32_t> all(n_seq);
    for (uint32_t i = 0; i < n_seq; ++i) all[i] = (int32_t) i;

    std::vector<int> last(n_seq, -1);
    std::vector<int> n_ckpt(n_seq, 0);
    for (int step = 0; step < 50*(int) k; ++step) {
        const auto due = s.due(max_n);
        CHECK(due.size() <= max_n);
        for (int32_t q : due) {
            if (last[q] >= 0) {
                CHECK(step - last[q] == (int) k); // period exactly k once in steady state
            }
            last[q] = step;
            n_ckpt[q]++;
            s.on_issued(q);
        }
        for (uint32_t q = 0; q < n_seq; ++q) {
            if (s.has_ckpt(q)) {
                CHECK(s.steps_since(q) <= k); // replay bound holds at every boundary
            }
        }
        s.on_step(all);
    }
    for (uint32_t q = 0; q < n_seq; ++q) {
        CHECK(n_ckpt[q] >= 49);
    }
    if (n_seq <= k) {
        // staggered: no two sequences share a residue
        for (uint32_t a = 0; a < n_seq; ++a) {
            for (uint32_t b = a + 1; b < n_seq; ++b) {
                CHECK(last[a] % (int) k != last[b] % (int) k);
            }
        }
    }

    // reset: no checkpoint, nothing due until it decodes again
    s.on_reset(0);
    CHECK(!s.has_ckpt(0));
    CHECK(s.steps_since(0) == 0);
}

static void test_schedule_idle_seq() {
    // a sequence that does not decode is never due (nothing new to checkpoint)
    ckpt_schedule s;
    s.init(6, 4);
    for (int i = 0; i < 100; ++i) {
        for (int32_t q : s.due(4)) s.on_issued(q);
        s.on_step({ 1, 2 });
    }
    CHECK(!s.has_ckpt(0) && !s.has_ckpt(3));
    CHECK(s.has_ckpt(1) && s.has_ckpt(2));
}

static void test_replay() {
    auto r = compute_replay(100, 82, 6, 6);
    CHECK(r.ok && r.n_tokens == 18 && r.n_steps == 6);
    r = compute_replay(100, 82, 7, 6);
    CHECK(!r.ok);
    r = compute_replay(100, -1, 0, 6);
    CHECK(!r.ok);
    r = compute_replay(50, 82, 0, 6); // truncated below the checkpoint
    CHECK(!r.ok);

    // simulation: variable accepted tokens per step (spec decode 1..8), checkpoints every k
    std::mt19937 rng(7);
    const uint32_t k = 6;
    ckpt_schedule s;
    s.init(k, 4);
    std::vector<int32_t> pos(4, 99);     // prompt of 100 tokens, pos_max = 99
    std::vector<int32_t> ckpt_pos(4, -1);
    std::vector<std::deque<int>> hist(4);
    for (int step = 0; step < 600; ++step) {
        for (int32_t q : s.due(1)) {
            ckpt_pos[q] = pos[q];
            s.on_issued(q);
            hist[q].clear();
        }
        // a lag save at this boundary replays at most k steps and at most the tokens of those steps
        for (int q = 0; q < 4; ++q) {
            if (ckpt_pos[q] < 0) continue;
            const auto rb = compute_replay(pos[q], ckpt_pos[q], s.steps_since(q), k);
            CHECK(rb.ok);
            int sum = 0;
            for (int t : hist[q]) sum += t;
            CHECK(rb.n_tokens == sum);
            CHECK((int) hist[q].size() <= (int) k);
        }
        for (int q = 0; q < 4; ++q) {
            const int acc = 1 + (int) (rng() % 8);
            pos[q] += acc;
            hist[q].push_back(acc);
        }
        s.on_step({ 0, 1, 2, 3 });
    }
}

static void test_coalesce_units() {
    auto r = coalesce_units({ 5, 3, 4, 10, 11, 3, 4 });
    CHECK(r.size() == 2);
    CHECK(r[0] == run_t(3, 6));
    CHECK(r[1] == run_t(10, 12));
    CHECK(coalesce_units({}).empty());
    r = coalesce_units({ 0 });
    CHECK(r.size() == 1 && r[0] == run_t(0, 1));
}

static void test_frags() {
    const size_t piece = 4096;
    std::vector<uint8_t> host(4*piece);
    std::vector<uint8_t> aliases_store(4*piece);
    std::vector<void *> aliases = { aliases_store.data(), aliases_store.data() + piece, nullptr, aliases_store.data() + 3*piece };
    host_tier tier;
    tier.adopt(host.data(), host.size(), piece, aliases);

    std::vector<uint8_t> dev(4*piece);
    std::vector<ggml_kv_mirror_frag> f;

    // crosses the 0|1 piece boundary: split in two with each piece's alias
    append_frag(tier, dev.data() + 4000, host.data() + 4000, 200, f);
    CHECK(f.size() == 2);
    CHECK(f[0].bytes == 96 && f[1].bytes == 104);
    CHECK(f[0].host_alias == aliases_store.data() + 4000);
    CHECK(f[1].host_alias == aliases_store.data() + piece);

    // piece 2 is unmapped: no alias (copy-engine path only)
    f.clear();
    append_frag(tier, dev.data() + 2*piece + 8, host.data() + 2*piece + 8, 16, f);
    CHECK(f.size() == 1 && f[0].host_alias == nullptr);

    // contiguous on both sides inside one piece -> merged; across a piece boundary -> kept apart
    f.clear();
    append_frag(tier, dev.data() + 0,   host.data() + 0,   64, f);
    append_frag(tier, dev.data() + 64,  host.data() + 64,  64, f);
    append_frag(tier, dev.data() + 512, host.data() + 512, 64, f); // gap
    append_frag(tier, dev.data() + 4032, host.data() + 4032, 64, f);
    append_frag(tier, dev.data() + 4096, host.data() + 4096, 64, f);
    coalesce_frags(tier, f);
    CHECK(f.size() == 4);
    CHECK(f[0].bytes == 128);
    CHECK(f[1].bytes == 64 && f[2].bytes == 64 && f[3].bytes == 64);

    // tail fragments over two regions, runs coalesced per region
    std::vector<kv_region> regions(2);
    regions[0] = { dev.data(),           (size_t) 16*64, 64, 0 };
    regions[1] = { dev.data() + 2*piece, (size_t) 16*64, 64, piece };
    f.clear();
    build_tail_frags(regions, { { 2, 5 }, { 9, 10 } }, tier, f);
    CHECK(f.size() == 4);
    CHECK(f[0].dev == dev.data() + 2*64 && f[0].host == host.data() + 2*64 && f[0].bytes == 3*64);
    CHECK(f[1].dev == dev.data() + 9*64 && f[1].bytes == 64);
    CHECK(f[2].dev == dev.data() + 2*piece + 2*64 && f[2].host == host.data() + piece + 2*64);
    // runs past a region's rows are clipped
    f.clear();
    build_tail_frags(regions, { { 15, 40 } }, tier, f);
    CHECK(f.size() == 2 && f[0].bytes == 64 && f[1].bytes == 64);
}

static void test_units() {
    unit_tracker u;
    u.init(8);
    CHECK(u.valid(0, 8)); // never written

    u.mark_written(3);
    CHECK(!u.valid(3, 4));
    uint32_t e1 = 0;
    auto runs1 = u.take_pending(&e1);
    CHECK(runs1.size() == 1 && runs1[0] == run_t(3, 4));
    CHECK(!u.has_pending());

    // rewritten while the copy of e1 is in flight: completing e1 must not validate it
    u.mark_written(3);
    u.on_complete(runs1, e1);
    CHECK(!u.valid(3, 4));

    uint32_t e2 = 0;
    auto runs2 = u.take_pending(&e2);
    CHECK(e2 > e1);
    u.on_complete(runs2, e2);
    CHECK(u.valid(3, 4));

    // out-of-order completion of an older copy does not regress a newer one
    u.on_complete(runs1, e1);
    CHECK(u.valid(3, 4));

    u.mark_all_written();
    CHECK(!u.valid(0, 1) && u.has_pending());
    uint32_t e3 = 0;
    auto runs3 = u.take_pending(&e3);
    CHECK(runs3.size() == 1 && runs3[0] == run_t(0, 8));
    u.on_complete(runs3, e3);
    CHECK(u.valid(0, 8));
}

static ggml_tensor * fake_tensor(ggml_context * ctx, uint8_t * data, size_t n) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) n);
    t->data = data;
    return t;
}

// decode-step writes -> async tail copies (deferred) -> save from the mirror -> restore via io_to_device
static void test_round_trip() {
    const uint32_t n_units = 512;
    const size_t   row_k = 96, row_v = 80; // like two q8_0 rows of different widths
    std::vector<uint8_t> dev_k(n_units*row_k), dev_v(n_units*row_v);

    host_tier tier;
    std::vector<uint8_t> host(dev_k.size() + dev_v.size() + 4096);
    tier.adopt(host.data(), host.size(), 1 << 12, {}); // all pieces unmapped -> copy-engine path in fragments
    {
        // give it identity aliases like the memcpy backend does
        std::vector<void *> al;
        for (size_t o = 0; o < host.size(); o += 1 << 12) al.push_back(host.data() + o);
        tier.adopt(host.data(), host.size(), 1 << 12, al);
    }

    std::vector<kv_region> regions = {
        { dev_k.data(), dev_k.size(), row_k, 0 },
        { dev_v.data(), dev_v.size(), row_v, dev_k.size() },
    };

    unit_tracker units;
    units.init(n_units);

    auto * dev = (deferred_dev *) dd_ops.dev_init(nullptr);
    struct infl { uint64_t t; runs_t runs; uint32_t e; };
    std::vector<infl> tail;

    std::mt19937 rng(42);
    auto write_cell = [&](uint32_t u, uint8_t salt) {
        for (size_t i = 0; i < row_k; ++i) dev_k[u*row_k + i] = (uint8_t) (u*31 + i + salt);
        for (size_t i = 0; i < row_v; ++i) dev_v[u*row_v + i] = (uint8_t) (u*17 + i*3 + salt);
        units.mark_written(u);
    };

    // 4 "slots" interleaved in a unified cache: slot s owns cells s, s+4, s+8, ... (fragmented layout)
    std::vector<uint32_t> next(4);
    for (uint32_t s = 0; s < 4; ++s) next[s] = s;
    for (int step = 0; step < 40; ++step) {
        for (uint32_t s = 0; s < 4; ++s) {
            const int n_tok = 1 + (int) (rng() % 3);
            for (int t = 0; t < n_tok && next[s] < n_units; ++t) {
                write_cell(next[s], (uint8_t) step);
                next[s] += 4;
            }
        }
        // spec rejection: rewrite the last cell of slot 1 with different content during the copy window
        uint32_t e = 0;
        runs_t runs = units.take_pending(&e);
        std::vector<ggml_kv_mirror_frag> f;
        build_tail_frags(regions, runs, tier, f);
        coalesce_frags(tier, f);
        const uint64_t t = dd_ops.copy_async((ggml_kv_mirror_dev_t) dev, f.data(), f.size(), GGML_KV_MIRROR_D2H, GGML_KV_MIRROR_PATH_AUTO);
        CHECK(t != 0);
        tail.push_back({ t, runs, e });
        if (step % 3 == 0 && next[1] >= 4) {
            write_cell(next[1] - 4, (uint8_t) (step + 100));
        }
        // the side stream progresses lazily: complete roughly every other copy
        if (step % 2 == 1) dev->pump_one();
        // poll as on_decode_begin does
        size_t w = 0;
        for (auto & x : tail) {
            if (dd_ops.ticket_done((ggml_kv_mirror_dev_t) dev, x.t)) units.on_complete(x.runs, x.e);
            else tail[w++] = x;
        }
        tail.resize(w);
    }

    // a save BEFORE draining: some cells are still in flight or dirty -> served by fallback, bytes still exact
    ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tk = fake_tensor(ctx, dev_k.data(), dev_k.size());
    ggml_tensor * tv = fake_tensor(ctx, dev_v.data(), dev_v.size());

    auto device_get = [](ggml_tensor * t, void * dst, size_t off, size_t size) { memcpy(dst, (uint8_t *) t->data + off, size); };

    // slot 2's cells as ranges (one row each: the fragmented case)
    std::vector<uint32_t> cells2;
    for (uint32_t u = 2; u < next[2]; u += 4) cells2.push_back(u);

    auto save = [&](io_from_mirror & io) {
        const uint32_t n = (uint32_t) cells2.size();
        io.write(&n, sizeof(n));
        for (uint32_t u : cells2) io.write_tensor(tk, u*row_k, row_k);
        for (uint32_t u : cells2) io.write_tensor(tv, u*row_v, row_v);
    };

    io_sink_buffer direct;
    {
        // reference: what a device save produces
        const uint32_t n = (uint32_t) cells2.size();
        direct.write(&n, sizeof(n));
        for (uint32_t u : cells2) direct.write(dev_k.data() + u*row_k, row_k);
        for (uint32_t u : cells2) direct.write(dev_v.data() + u*row_v, row_v);
    }

    io_sink_buffer early;
    {
        io_from_mirror io(early, regions, units, tier, device_get);
        save(io);
        CHECK(io.bytes_fallback + io.bytes_from_mirror == cells2.size()*(row_k + row_v));
        CHECK(!tail.empty()); // copies were still in flight, so part of this save came from the device
        CHECK(io.bytes_fallback > 0);
    }
    CHECK(early.buf == direct.buf);

    // drain the side stream (save-time wait), then every cell is served from the mirror
    dev->pump_all();
    for (auto & x : tail) units.on_complete(x.runs, x.e);
    tail.clear();
    // the rewritten cells of slot 1 were written after their copy was issued: a final flush, as fresh mode does
    {
        uint32_t e = 0;
        runs_t runs = units.take_pending(&e);
        std::vector<ggml_kv_mirror_frag> f;
        build_tail_frags(regions, runs, tier, f);
        const uint64_t t = dd_ops.copy_async((ggml_kv_mirror_dev_t) dev, f.data(), f.size(), GGML_KV_MIRROR_D2H, GGML_KV_MIRROR_PATH_AUTO);
        dd_ops.ticket_wait((ggml_kv_mirror_dev_t) dev, t);
        units.on_complete(runs, e);
    }
    CHECK(units.valid(0, n_units));
    CHECK(memcmp(host.data(), dev_k.data(), dev_k.size()) == 0);
    CHECK(memcmp(host.data() + dev_k.size(), dev_v.data(), dev_v.size()) == 0);

    io_sink_buffer late;
    {
        io_from_mirror io(late, regions, units, tier, device_get);
        save(io);
        CHECK(io.bytes_fallback == 0);
        CHECK(io.bytes_from_mirror == cells2.size()*(row_k + row_v));
    }
    CHECK(late.buf == direct.buf);

    // restore into an empty "device" at DIFFERENT cells (another slot's free cells: 3, 7, 11, ...)
    std::vector<uint8_t> dev2_k(dev_k.size(), 0), dev2_v(dev_v.size(), 0);
    ggml_tensor * tk2 = fake_tensor(ctx, dev2_k.data(), dev2_k.size());
    ggml_tensor * tv2 = fake_tensor(ctx, dev2_v.data(), dev2_v.size());

    // the stream is staged in "pinned" memory: adopt the buffer as a registered tier
    host_tier stage;
    {
        stage.adopt(late.buf.data(), late.buf.size(), 1 << 20, { late.buf.data() });
    }
    size_t n_dev_frags = 0;
    {
        io_to_device io(late.buf.data(), late.buf.size(), &stage, &dd_ops, (ggml_kv_mirror_dev_t) dev, GGML_KV_MIRROR_PATH_AUTO,
                        [](const ggml_tensor *) { return true; },
                        [](ggml_tensor * t, const void * src, size_t off, size_t size) { memcpy((uint8_t *) t->data + off, src, size); });
        uint32_t n = 0;
        io.read(&n, sizeof(n));
        CHECK(n == cells2.size());
        for (uint32_t i = 0; i < n; ++i) io.read_tensor(tk2, (3 + 4*i)*row_k, row_k);
        for (uint32_t i = 0; i < n; ++i) io.read_tensor(tv2, (3 + 4*i)*row_v, row_v);
        CHECK(io.flush());
        n_dev_frags = io.n_frags_device;
        CHECK(io.bytes_fallback == 0);
        CHECK(io.bytes_device == cells2.size()*(row_k + row_v));
    }
    CHECK(n_dev_frags == 2*cells2.size()); // fragmented destination: one fragment per cell per tensor
    for (uint32_t i = 0; i < cells2.size(); ++i) {
        CHECK(memcmp(dev2_k.data() + (3 + 4*i)*row_k, dev_k.data() + cells2[i]*row_k, row_k) == 0);
        CHECK(memcmp(dev2_v.data() + (3 + 4*i)*row_v, dev_v.data() + cells2[i]*row_v, row_v) == 0);
    }

    // a contiguous destination coalesces into one bulk fragment per tensor
    {
        std::vector<uint8_t> dev3(dev_k.size(), 0);
        ggml_tensor * tk3 = fake_tensor(ctx, dev3.data(), dev3.size());
        io_to_device io(late.buf.data(), late.buf.size(), &stage, &dd_ops, (ggml_kv_mirror_dev_t) dev, GGML_KV_MIRROR_PATH_AUTO,
                        [](const ggml_tensor *) { return true; },
                        [](ggml_tensor * t, const void * src, size_t off, size_t size) { memcpy((uint8_t *) t->data + off, src, size); });
        uint32_t n = 0;
        io.read(&n, sizeof(n));
        for (uint32_t i = 0; i < n; ++i) io.read_tensor(tk3, i*row_k, row_k);
        CHECK(io.flush());
        CHECK(io.n_frags_device == 1);
        for (uint32_t i = 0; i < n; ++i) {
            CHECK(memcmp(dev3.data() + i*row_k, dev_k.data() + cells2[i]*row_k, row_k) == 0);
        }
    }

    // a tensor not on the mirror device takes the fallback, same bytes
    {
        std::vector<uint8_t> dev4(dev_k.size(), 0);
        ggml_tensor * tk4 = fake_tensor(ctx, dev4.data(), dev4.size());
        io_to_device io(late.buf.data(), late.buf.size(), &stage, &dd_ops, (ggml_kv_mirror_dev_t) dev, GGML_KV_MIRROR_PATH_AUTO,
                        [](const ggml_tensor *) { return false; },
                        [](ggml_tensor * t, const void * src, size_t off, size_t size) { memcpy((uint8_t *) t->data + off, src, size); });
        uint32_t n = 0;
        io.read(&n, sizeof(n));
        for (uint32_t i = 0; i < n; ++i) io.read_tensor(tk4, i*row_k, row_k);
        CHECK(io.flush());
        CHECK(io.n_frags_device == 0 && io.bytes_fallback == n*row_k);
        for (uint32_t i = 0; i < n; ++i) {
            CHECK(memcmp(dev4.data() + i*row_k, dev_k.data() + cells2[i]*row_k, row_k) == 0);
        }
    }

    ggml_free(ctx);
    dd_ops.dev_free((ggml_kv_mirror_dev_t) dev);
}

// checkpoint capture: literal metadata + tensor ranges into a slot, replayed after the copy completes
static void test_capture() {
    std::vector<uint8_t> state(3*1000);
    for (size_t i = 0; i < state.size(); ++i) state[i] = (uint8_t) (i*7 + 1);

    std::vector<uint8_t> host(64*1024);
    host_tier tier;
    {
        std::vector<void *> al;
        for (size_t o = 0; o < host.size(); o += 4096) al.push_back(host.data() + o);
        tier.adopt(host.data(), host.size(), 4096, al);
    }

    ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * r = fake_tensor(ctx, state.data(), state.size());

    auto emit = [&](llama_io_write_i & io) {
        const uint32_t cell_count = 1;
        const int32_t pos = 1234;
        io.write(&cell_count, sizeof(cell_count));
        io.write(&pos, sizeof(pos));
        for (int l = 0; l < 3; ++l) {
            const uint64_t row = 1000;
            io.write(&row, sizeof(row));
            io.write_tensor(r, l*1000, 1000);
        }
    };

    // reference stream (what a device save writes)
    io_sink_buffer ref;
    {
        struct ref_io : public llama_io_write_i {
            io_sink_buffer & s;
            explicit ref_io(io_sink_buffer & s) : s(s) {}
            void write(const void * src, size_t n) override { s.write(src, n); }
            void write_tensor(ggml_tensor * t, size_t off, size_t n) override { s.write((uint8_t *) t->data + off, n); }
            size_t n_bytes() override { return s.n_bytes(); }
        } io(ref);
        emit(io);
    }

    captured_stream cs;
    std::vector<ggml_kv_mirror_frag> frags;
    const size_t slot_off = 8192, slot_cap = 4096;
    {
        io_capture io(cs, tier, slot_off, slot_cap, frags);
        emit(io);
        CHECK(io.n_bytes() == ref.buf.size());
    }
    CHECK(cs.data_used == 3000);
    CHECK(!frags.empty());
    coalesce_frags(tier, frags);

    const auto * mc = memcpy_ops();
    auto d = mc->dev_init(nullptr);
    const uint64_t t = mc->copy_async(d, frags.data(), frags.size(), GGML_KV_MIRROR_D2H, GGML_KV_MIRROR_PATH_AUTO);
    CHECK(t != 0 && mc->ticket_done(d, t));

    // the device state moves on after the snapshot; the checkpoint keeps the snapshot
    std::vector<uint8_t> snapshot = state;
    for (auto & b : state) b ^= 0x5a;

    io_sink_buffer out;
    cs.write_to(out, tier.data());
    std::vector<uint8_t> want = ref.buf; // ref was produced from the pre-mutation state
    CHECK(out.buf == want);

    // capacity is enforced
    captured_stream small;
    std::vector<ggml_kv_mirror_frag> f2;
    bool threw = false;
    try {
        io_capture io(small, tier, slot_off, 1500, f2);
        emit(io);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);

    mc->dev_free(d);
    ggml_free(ctx);
    (void) snapshot;
}

static void test_host_tier_reserve() {
    host_tier t;
    std::string err;
    CHECK(t.reserve(8 << 20, -1, 2 << 20, err));
    CHECK(t.data() != nullptr && t.bytes() == (size_t) 8 << 20);
    const auto * mc = memcpy_ops();
    auto d = mc->dev_init(nullptr);
    CHECK(t.register_with(mc, d));
    CHECK(t.n_registered_pieces() == 4);
    CHECK(t.registered(t.data() + 100, 1000));
    CHECK(t.alias_of(t.data() + (3 << 20)) == t.data() + (3 << 20));
    t.data()[0] = 1; // writable
    t.release();
    CHECK(t.data() == nullptr);
    mc->dev_free(d);
}

int main() {
    test_parse();
    test_schedule(6, 4);
    test_schedule(13, 4);
    test_schedule(6, 6);
    test_schedule(6, 8);
    test_schedule(1, 4);
    test_schedule_idle_seq();
    test_replay();
    test_coalesce_units();
    test_frags();
    test_units();
    test_round_trip();
    test_capture();
    test_host_tier_reserve();

    if (g_fail) {
        fprintf(stderr, "test-kv-host-mirror: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-kv-host-mirror: all checks passed\n");
    return 0;
}
