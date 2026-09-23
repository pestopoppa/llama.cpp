// test-gather-rows-e4m3-e8m0.cpp: validate GGML_OP_GATHER_ROWS_E4M3_E8M0 against
// hand-computed values.
//
// The op gathers rows of a packed FP8 table carried as GGML_TYPE_I8 bytes,
// ne = [33*k, n_rows]: 32*k E4M3 code bytes followed by k E8M0 scale bytes, one
// scale per 32 consecutive codes, dequantized as
//
//   value[j] = ldexpf(e4m3_to_f32(code[j]), scale[j/32] - 127)
//
// with 0.0f substituted whenever the code is one of E4M3's two NaN patterns
// (0x7F, 0xFF) or the scale byte is 0xFF.
//
// Nothing here shares code with the implementation: the sixteen E4M3 magnitudes
// and the two scale factors below are written out by hand, the expectation is
// their product, and the comparison is bit-exact (every product is a value times
// an exact power of two, well inside the normal range, so it is exact).
//
// Scale bytes 119 and 120 are the values observed in the shipped tables; 0xFF
// exercises the scale-side NaN guard. Two geometries are checked, k = 8 (the
// 264-byte row the GGUF tags `e4m3_e8m0_32_row264`) and k = 2, so the op is
// proved generic over the row width rather than hardcoded to 264.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The E4M3 codes used to fill the table, and their hand-computed magnitudes.
//
//   0x00 exp= 0 man=0  +0.0                       0x38 exp= 7 man=0  +1.0
//   0xB8 sign,exp= 7   -1.0                       0x3C exp= 7 man=4  +1.5
//   0x40 exp= 8 man=0  +2.0                       0x30 exp= 6 man=0  +0.5
//   0x07 exp= 0 man=7  +7*2^-9  = +0.013671875    0x01 exp= 0 man=1  +2^-9 = +0.001953125
//   0x7E exp=15 man=6  +1.75*2^8 = +448.0         0xFE sign          -448.0
//   0x7F NaN           guarded                    0xFF NaN           guarded
//   0x80 sign,zero     -0.0                       0x08 exp= 1 man=0  +2^-6 = +0.015625
//   0x78 exp=15 man=0  +2^8 = +256.0              0xBC sign,exp=7    -1.5
static const uint8_t CODES[16] = {
    0x00, 0x38, 0xB8, 0x3C, 0x40, 0x30, 0x07, 0x01,
    0x7E, 0xFE, 0x7F, 0xFF, 0x80, 0x08, 0x78, 0xBC,
};

static const float MAGS[16] = {
    0.0f,   1.0f,  -1.0f,   1.5f,   2.0f,   0.5f,  0.013671875f, 0.001953125f,
  448.0f, -448.0f, 0.0f,    0.0f,  -0.0f,   0.015625f, 256.0f,  -1.5f,
};

// true for the codes that must be replaced by +0.0f by the NaN guard
static bool code_is_nan(int i) {
    return CODES[i] == 0x7F || CODES[i] == 0xFF;
}

// scale byte -> factor, hand-computed: 2^(119-127) = 2^-8, 2^(120-127) = 2^-7
static const uint8_t SCALE_A = 119;
static const uint8_t SCALE_B = 120;
static const float   FACTOR_A = 0.00390625f;  // 2^-8
static const float   FACTOR_B = 0.0078125f;   // 2^-7

// the scale byte group g of row r carries
static uint8_t scale_byte(int64_t g) {
    if (g == 5) {
        return 0xFF;       // NaN-guarded group: the whole 32 values go to zero
    }
    return (g % 2) == 0 ? SCALE_A : SCALE_B;
}

// the code byte at column c of row r
static uint8_t code_byte(int64_t r, int64_t c) {
    return CODES[(c % 32 + r) % 16];
}

// the expected f32 at column c of row r
static float expected(int64_t r, int64_t c) {
    const uint8_t s = scale_byte(c/32);
    if (s == 0xFF) {
        return 0.0f;
    }
    const int i = (int) ((c % 32 + r) % 16);
    if (code_is_nan(i)) {
        return 0.0f;
    }
    return MAGS[i] * (s == SCALE_A ? FACTOR_A : FACTOR_B);
}

static bool bit_equal(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

// One geometry: k groups of 32 per row, n_rows in the table, ids shaped [n_ids, n_seq]
struct gather_case {
    int64_t k;
    int64_t n_rows;
    int64_t n_ids;
    int64_t n_seq;
};

static const gather_case CASES[] = {
    { 8, 4, 5, 2 },   // 264-byte rows, ids wider than the table (indices repeat)
    { 8, 3, 1, 1 },   // single row gathered
    { 2, 6, 7, 1 },   // 66-byte rows
};

static int run_case(const gather_case & c, int n_threads, std::vector<float> & out) {
    const int64_t nb_row = 33*c.k;   // packed bytes per row
    const int64_t nc     = 32*c.k;   // dequantized values per row
    const int64_t nr     = c.n_ids*c.n_seq;

    struct ggml_init_params params = {
        /* .mem_size   = */ (size_t) 64 << 20,
        /* .mem_base   = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  nb_row, c.n_rows);
    struct ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_ids, c.n_seq);

    uint8_t * tb = (uint8_t *) table->data;
    for (int64_t r = 0; r < c.n_rows; r++) {
        uint8_t * row = tb + r*nb_row;
        for (int64_t col = 0; col < nc; col++) {
            row[col] = code_byte(r, col);
        }
        for (int64_t g = 0; g < c.k; g++) {
            row[nc + g] = scale_byte(g);
        }
    }

    int32_t * ib = (int32_t *) ids->data;
    for (int64_t i = 0; i < nr; i++) {
        ib[i] = (int32_t) ((i*3 + 1) % c.n_rows);
    }

    struct ggml_tensor * y = ggml_gather_rows_e4m3_e8m0(ctx, table, ids);

    GGML_ASSERT(y->type  == GGML_TYPE_F32);
    GGML_ASSERT(y->ne[0] == nc);
    GGML_ASSERT(y->ne[1] == c.n_ids);
    GGML_ASSERT(y->ne[2] == c.n_seq);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    int fails = 0;

    out.assign((size_t) nc*nr, 0.0f);
    const float * yd = (const float *) y->data;

    for (int64_t i = 0; i < nr; i++) {
        const int64_t r = ib[i];
        for (int64_t col = 0; col < nc; col++) {
            const float got  = yd[i*nc + col];
            const float want = expected(r, col);
            out[(size_t) (i*nc + col)] = got;
            if (!bit_equal(got, want) && fails < 8) {
                printf("  MISMATCH row=%lld idx=%lld col=%lld: got %.9g, expected %.9g\n",
                    (long long) r, (long long) i, (long long) col, (double) got, (double) want);
                fails++;
            } else if (!bit_equal(got, want)) {
                fails++;
            }
        }
    }

    ggml_free(ctx);
    return fails;
}

int main(void) {
    int fails = 0;

    // A hand-written spot check of the whole pipeline, independent of the loops above:
    // row 0 column 0 is code 0x00 under scale 119 -> +0.0; row 0 column 1 is code
    // 0x38 (+1.0) under scale 119 -> 1.0 * 2^-8 = 0.00390625; row 0 column 34 is
    // code 0xB8 (-1.0) under scale 120 -> -1.0 * 2^-7 = -0.0078125 (column 33 is
    // code index 1, i.e. 0x38 again, under that same scale -> +0.0078125); row 0
    // column 160 falls in group 5, whose scale byte is 0xFF -> +0.0.
    if (!bit_equal(expected(0,   0),  0.0f)       ||
        !bit_equal(expected(0,   1),  0.00390625f) ||
        !bit_equal(expected(0,  33),  0.0078125f)  ||
        !bit_equal(expected(0,  34), -0.0078125f)  ||
        !bit_equal(expected(0, 160),  0.0f)) {
        printf("FAIL: the test's own expectation table is wrong\n");
        return 1;
    }

    for (const gather_case & c : CASES) {
        std::vector<float> ref;
        std::vector<float> alt;

        const int f1 = run_case(c, 1, ref);

        // thread counts above the gathered-row count force the column-chunk split
        static const int NTH[2] = { 3, 8 };
        int fmt = 0;
        for (int nth : NTH) {
            fmt += run_case(c, nth, alt);
            if (alt != ref) {
                printf("  MISMATCH: nth=%d differs from nth=1\n", nth);
                fmt++;
            }
        }

        const int f = f1 + fmt;
        fails += f;
        printf("gather_rows_e4m3_e8m0 k=%lld n_rows=%lld ids=[%lld,%lld]: %s\n",
            (long long) c.k, (long long) c.n_rows, (long long) c.n_ids, (long long) c.n_seq,
            f == 0 ? "OK" : "FAIL");
    }

    printf(fails == 0 ? "all gather_rows_e4m3_e8m0 checks passed\n"
                      : "%d gather_rows_e4m3_e8m0 checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
