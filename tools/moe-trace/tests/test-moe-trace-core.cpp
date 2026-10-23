// Model-free unit tests for moe-trace-core.h (INF-72 RT-1 / RT-3). No llama, no ggml, no GGUF.
//
//   test-moe-trace-core [OUT_DIR]
//
// With OUT_DIR the test also writes a synthetic run (two layers: 256 experts / top-6 and 128 experts /
// top-4, ids above 127, decode steps, three samples) through the SAME run_writer the tool uses, so the
// python side (tests/test_moe_trace_convert.py, and trace_shape.load_run in epyc-inference-research) can
// read what C++ wrote. That is the cross-language contract test.

#include "../moe-trace-core.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

using namespace moe_trace;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d  %s  [%s]\n", __FILE__, __LINE__, msg, #cond); g_fail++; } } while (0)

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

static void test_sha256() {
    CHECK(sha256::of("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
    CHECK(sha256::of("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
    const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";   // FIPS 180-4 two-block vector
    CHECK(sha256::of(two.data(), two.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "sha256 two-block");
    std::string big(1000003, 'x');
    for (size_t i = 0; i < big.size(); i++) big[i] = (char) (i * 31 + 7);
    sha256 inc; size_t off = 0, step = 1;
    while (off < big.size()) { const size_t n = std::min(step, big.size() - off); inc.update(big.data() + off, n); off += n; step = step * 3 + 1; }
    CHECK(inc.hex() == sha256::of(big.data(), big.size()), "incremental == one-shot");
    CHECK(inc.hex() == inc.hex(), "hex() is non-destructive");
}

static void test_selftest() {
    const std::string e = selftest();
    CHECK(e.empty(), e.c_str());
}

// RT-3: the strided read returns the selected experts; a flat read of the same tensor is the broken one
static void test_stride_read_and_flat_hazard() {
    for (int n_experts : {64, 256, 512}) {
        const int k = n_experts == 512 ? 10 : 8, n_tokens = 512;
        std::vector<int32_t> mem = detail::synth_argsort(n_experts, n_tokens, 99 + n_experts);
        strided_i32 v; v.k = k; v.n_rows = n_tokens; v.nb1 = (size_t) n_experts * 4;
        const uint8_t * base = reinterpret_cast<const uint8_t *>(mem.data());
        int reads = 0;
        read_fn rd = [&](void * dst, size_t off, size_t sz) { reads++; CHECK(off % v.nb1 == 0 && sz == (size_t) k * 4, "one row per read, at i*nb1"); std::memcpy(dst, base + off, sz); };
        std::vector<int32_t> good;
        read_rows_strided(v, rd, good);
        CHECK(reads == n_tokens, "exactly one read per token row");
        bool exact = true;
        for (int t = 0; t < n_tokens; t++) for (int j = 0; j < k; j++) exact &= good[(size_t) t * k + j] == mem[(size_t) t * n_experts + j];
        CHECK(exact, "strided read matches the first k of every argsort row");

        std::vector<int16_t> g16, b16; std::string err;
        CHECK(narrow_ids(good, n_experts, g16, err), err.c_str());
        const hazard_result hg = hazard_check(g16.data(), n_tokens, k, n_experts);
        CHECK(hg.evaluated && !hg.fatal && !hg.uniform, "correct read passes the guard");
        CHECK(hg.ir_mean > hg.chance, "synthetic routing reuse is above chance");

        const std::vector<int32_t> bad = detail::flat_read_WRONG(mem, v);
        CHECK(narrow_ids(bad, n_experts, b16, err), err.c_str());
        const hazard_result hb = hazard_check(b16.data(), n_tokens, k, n_experts);
        CHECK(hb.evaluated && hb.fatal, "FLAT read is detected");
        if (n_experts >= 256 && (n_tokens * k) % n_experts == 0) {
            const int expect = n_tokens * k / n_experts;
            CHECK(hb.uniform && hb.count_min == expect && hb.count_max == expect, "every expert appears exactly the same number of times");
        }
        CHECK(hb.sub_chance, "flat read shows sub-chance adjacent reuse");
        CHECK(hb.why.find("STRIDE-HAZARD") != std::string::npos, "message names the hazard");
    }
    // the literal source-tool signature: n_experts == n_tokens, each expert exactly k times
    {
        const int n_experts = 128, k = 8, n_tokens = 128;
        std::vector<int32_t> mem = detail::synth_argsort(n_experts, n_tokens, 7);
        strided_i32 v; v.k = k; v.n_rows = n_tokens; v.nb1 = n_experts * 4;
        std::vector<int16_t> b16; std::string err;
        narrow_ids(detail::flat_read_WRONG(mem, v), n_experts, b16, err);
        const hazard_result hb = hazard_check(b16.data(), n_tokens, k, n_experts);
        CHECK(hb.fatal && hb.count_min == k && hb.count_max == k, "every expert appearing exactly k times");
    }
}

static void test_guard_edges() {
    // sub-chance reuse ALONE is a result, not a hazard
    {
        const int n_experts = 64, k = 4, n_tokens = 128;
        std::vector<int16_t> ids((size_t) n_tokens * k);
        // skewed (hot experts 0..7 get most picks) but adjacent tokens never share: tokens alternate between two disjoint sets
        for (int t = 0; t < n_tokens; t++) for (int j = 0; j < k; j++) ids[(size_t) t * k + j] = (int16_t) ((t % 2 ? 4 : 0) + j + (t % 5 == 0 ? 8 : 0) * (j % 2));
        // keep rows distinct
        for (int t = 0; t < n_tokens; t++) { std::set<int> s(ids.begin() + (size_t) t * k, ids.begin() + (size_t) (t + 1) * k); if ((int) s.size() != k) { for (int j = 0; j < k; j++) ids[(size_t) t * k + j] = (int16_t) (t % 2 ? 4 + j : j); } }
        const hazard_result h = hazard_check(ids.data(), n_tokens, k, n_experts);
        CHECK(h.evaluated && !h.uniform, "skewed counts are not uniform");
        CHECK(h.sub_chance, "this fixture has sub-chance reuse");
        CHECK(!h.fatal, "sub-chance reuse alone must not be fatal");
    }
    // near-uniform AND sub-chance is fatal even when not exactly +-1
    {
        const int n_experts = 64, k = 4, n_tokens = 64;
        std::vector<int32_t> perm(n_experts); std::vector<int16_t> ids; std::string err;
        std::vector<int32_t> flat;
        for (int r = 0; r < 4; r++) for (int e = 0; e < n_experts; e++) flat.push_back((e * 7 + r * 3) % n_experts);   // 4 permutations back to back
        narrow_ids(flat, n_experts, ids, err);
        ids[0] = 0; ids[1] = 1; ids[2] = 2; ids[3] = 5;    // perturb one token: counts now differ by 2, cv stays tiny
        const hazard_result h = hazard_check(ids.data(), n_tokens, k, n_experts);
        CHECK(h.evaluated && !h.uniform && h.near_uniform && h.sub_chance && h.fatal, "near-uniform + sub-chance is fatal");
    }
    // not enough mass to judge: decode step, tiny experts count
    {
        std::vector<int16_t> one = {1, 2, 3, 4};
        CHECK(!hazard_check(one.data(), 1, 4, 256).evaluated, "single-token forward is not evaluated");
        std::vector<int16_t> few((size_t) 32 * 2, 0);
        CHECK(!hazard_check(few.data(), 32, 2, 8).evaluated, "n_experts < 16 is not evaluated");
        std::vector<int16_t> thin((size_t) 8 * 4, 0);
        CHECK(!hazard_check(thin.data(), 8, 4, 256).evaluated, "selection mass < 2*n_experts is not evaluated");
    }
    // k a large share of the pool: uniform by construction, not judgeable
    {
        std::vector<int16_t> all((size_t) 64 * 8);
        for (int t = 0; t < 64; t++) for (int j = 0; j < 8; j++) all[(size_t) t * 8 + j] = (int16_t) j;
        CHECK(!hazard_check(all.data(), 64, 8, 16).evaluated, "k*4 > n_experts is not evaluated");
        std::vector<int16_t> edge((size_t) 64 * 4);
        for (int t = 0; t < 64; t++) for (int j = 0; j < 4; j++) edge[(size_t) t * 4 + j] = (int16_t) ((t * 3 + j) % 16);
        CHECK(hazard_check(edge.data(), 64, 4, 16).evaluated, "k*4 == n_experts is evaluated (dbrx 4/16)");
    }
    // out-of-range id is fatal
    {
        std::vector<int16_t> ids((size_t) 64 * 4, 1); ids[17] = 70;
        const hazard_result h = hazard_check(ids.data(), 64, 4, 64);
        CHECK(h.fatal, "id >= n_experts is fatal");
    }
    // int8 would wrap: ids above 127 survive narrowing
    {
        std::vector<int32_t> wide = {0, 127, 128, 255, 256, 511}; std::vector<int16_t> o; std::string e;
        CHECK(narrow_ids(wide, 512, o, e) && o[2] == 128 && o[5] == 511, "ids above 127 are preserved");
        CHECK(!narrow_ids(wide, 256, o, e), "ids >= n_experts are refused");
        std::vector<int32_t> neg = {-1}; CHECK(!narrow_ids(neg, 8, o, e), "negative ids are refused");
    }
}

static void test_eor() {
    std::vector<layer_info> L = {{0, 8, 4}};
    eor_tracker tr; tr.init(L);
    auto fwd = [&](int phase, std::vector<std::vector<int16_t>> rows) {
        std::vector<int16_t> ids; for (auto & r : rows) ids.insert(ids.end(), r.begin(), r.end());
        tr.add_forward(phase, ids.data(), (int) rows.size());
    };
    fwd(PHASE_PREFILL, {{0, 1, 2, 3}, {2, 3, 4, 5}, {2, 3, 4, 5}});   // pairs: 2/4, 4/4
    fwd(PHASE_DECODE,  {{4, 5, 6, 7}});                                  // pair vs {2,3,4,5}: 2/4, counted as decode
    tr.reset_sample();
    fwd(PHASE_PREFILL, {{0, 1, 2, 3}});                                  // new sample: no pair across the boundary
    CHECK(tr.pairs(0, PHASE_PREFILL) == 2 && std::abs(tr.ir_mean(0, PHASE_PREFILL) - 0.75) < 1e-12, "prefill IR_t = (0.5 + 1.0) / 2");
    CHECK(tr.pairs(0, PHASE_DECODE) == 1 && std::abs(tr.ir_mean(0, PHASE_DECODE) - 0.5) < 1e-12, "decode IR_t = 0.5");
    const std::string j = tr.to_json();
    CHECK(j.find("\"chance\": 0.500000") != std::string::npos, "chance baseline = K / n_experts");
    CHECK(j.find("epyc.moe.routing_trace.eor.v1") != std::string::npos, "eor schema id");
    // the fleet baselines named in the handoff
    CHECK(std::abs(8.0 / 256 - 0.03125) < 1e-12 && std::abs(10.0 / 512 - 0.01953125) < 1e-12, "3.13% and 1.95% baselines");
}

// the synthetic run: also what the python tests read back
static int write_synthetic_run(const std::string & dir) {
    run_meta m;
    m.model = "synthetic-fixture"; m.arch = "fixture-moe"; m.domain = "synthetic"; m.batch_size = 512;
    m.layers = {{0, 256, 6}, {1, 128, 4}};    // different n_experts AND top_k per layer (DSpark-like), il = ordinal
    run_writer w; std::string err;
    if (!w.open(dir, m, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    const int nl = 2, mk = 6;
    auto make_ids = [&](int row, std::vector<int16_t> & out) {
        for (int l = 0; l < nl; l++) for (int j = 0; j < mk; j++) {
            int16_t v = -1;
            if (l == 0 && j < 6) v = (int16_t) ((200 + row * 7 + j * 13) % 256);
            if (l == 1 && j < 4) v = (int16_t) ((100 + row * 3 + j * 5) % 128);
            out[(size_t) l * mk + j] = v;
        }
    };
    int row = 0;
    auto forward = [&](uint32_t sid, uint32_t pos0, int phase, uint32_t step, int n) -> bool {
        std::vector<int16_t> ids((size_t) n * nl * mk);
        for (int t = 0; t < n; t++) { std::vector<int16_t> one((size_t) nl * mk); make_ids(row++, one); std::copy(one.begin(), one.end(), ids.begin() + (size_t) t * nl * mk); }
        return w.append_forward(sid, "synthetic", pos0, phase, step, n, ids.data(), nullptr, err);
    };
    bool ok = forward(0, 0, PHASE_PREFILL, 0, 5) && forward(0, 5, PHASE_DECODE, 1, 1) && forward(0, 6, PHASE_DECODE, 2, 1) && forward(0, 7, PHASE_DECODE, 3, 1)
           && forward(1, 0, PHASE_PREFILL, 0, 4)
           && forward(2, 0, PHASE_PREFILL, 0, 3) && forward(2, 3, PHASE_DECODE, 1, 1) && forward(2, 4, PHASE_DECODE, 2, 1);
    if (!ok) { std::fprintf(stderr, "append: %s\n", err.c_str()); return 1; }
    std::string manifest;
    if (!w.close(manifest, err)) { std::fprintf(stderr, "close: %s\n", err.c_str()); return 1; }
    CHECK(w.rows() == 8 + 4 + 5 && row == 17, "17 rows written");

    // what is on disk must agree with the manifest, byte for byte
    const std::string rows = slurp(dir + "/routing_trace.rows.bin"), ids = slurp(dir + "/routing_trace.ids.i16");
    CHECK(rows.size() == 17 * ROW_BYTES, "rows.bin is 16 bytes per row");
    CHECK(ids.size() == (size_t) 17 * nl * mk * 2, "ids.i16 is [rows, n_layers, max_top_k] int16");
    CHECK(manifest.find(sha256::of(rows.data(), rows.size())) != std::string::npos, "manifest carries the rows sha256");
    CHECK(manifest.find(sha256::of(ids.data(), ids.size())) != std::string::npos, "manifest carries the ids sha256");
    CHECK(manifest.find("\"id_dtype\": \"int16\"") != std::string::npos && manifest.find("int8") == std::string::npos, "int16 only");
    int16_t first_l0 = 0; std::memcpy(&first_l0, ids.data(), 2);
    CHECK(first_l0 == 200, "first id of row 0 layer 0 is 200 (> 127 survives)");
    int16_t pad = 0; std::memcpy(&pad, ids.data() + (size_t) (mk + 4) * 2, 2);
    CHECK(pad == -1, "layer 1 pads its 5th slot with -1");
    uint8_t acc = 0; std::memcpy(&acc, rows.data() + 15, 1);
    CHECK(acc == ACC_NULL, "spec_accepted is null (this tool never speculates)");
    return 0;
}

int main(int argc, char ** argv) {
    test_sha256();
    test_selftest();
    test_stride_read_and_flat_hazard();
    test_guard_edges();
    test_eor();
    if (argc > 1) {
        std::error_code ec; std::filesystem::remove_all(argv[1], ec);
        if (write_synthetic_run(argv[1]) != 0) g_fail++;
    }
    if (g_fail) { std::fprintf(stderr, "%d check(s) FAILED\n", g_fail); return 1; }
    std::printf("test-moe-trace-core: all checks passed\n");
    return 0;
}
