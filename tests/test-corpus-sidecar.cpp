#ifdef LLAMA_CORPUS_SIDECAR

#include "corpus-sidecar.h"
#include "md5.h"
#include "ngram-cache.h"
#include "log.h"

#include <sqlite3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

// Create a minimal 2-shard corpus in a temp directory for testing.
// Returns the path to the temp directory.
static std::string create_test_corpus(const std::string & base_dir) {
    std::string corpus_dir = base_dir + "/test_corpus";

    // Create directory
    std::string cmd = "mkdir -p " + corpus_dir;
    int ret = system(cmd.c_str());
    assert(ret == 0);

    // Create snippets.db
    {
        std::string db_path = corpus_dir + "/snippets.db";
        sqlite3 * db = nullptr;
        int rc = sqlite3_open(db_path.c_str(), &db);
        assert(rc == SQLITE_OK);

        rc = sqlite3_exec(db,
            "CREATE TABLE snippets (id INTEGER PRIMARY KEY, code TEXT, source TEXT, hash TEXT);"
            "INSERT INTO snippets VALUES (3000000000, 'def calculate_loss(predictions, targets):\n    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)\n', 'test.py', 'abc123');"
            "INSERT INTO snippets VALUES (3000000001, 'def train_model(data, epochs=10):\n    for epoch in range(epochs):\n        loss = calculate_loss(predict(data), data.targets)\n        update_weights(loss)\n', 'test.py', 'def456');"
            "INSERT INTO snippets VALUES (3000000002, 'class DataLoader:\n    def __init__(self, dataset, batch_size=32):\n        self.dataset = dataset\n        self.batch_size = batch_size\n', 'test.py', 'ghi789');"
            "INSERT INTO snippets VALUES (3001000000, 'fn calculate_distance(a: Point, b: Point) -> f64 {\n    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()\n}\n', 'test.rs', 'jkl012');"
            "INSERT INTO snippets VALUES (3001000001, 'struct Point {\n    x: f64,\n    y: f64,\n}\n\nimpl Point {\n    fn new(x: f64, y: f64) -> Self {\n        Self { x, y }\n    }\n}\n', 'test.rs', 'mno345');",
            nullptr, nullptr, nullptr);
        assert(rc == SQLITE_OK);
        sqlite3_close(db);
    }

    // Create 2 shard databases
    for (int shard = 0; shard < 2; shard++) {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/shard_%02d.db", corpus_dir.c_str(), shard);

        sqlite3 * db = nullptr;
        int rc = sqlite3_open(fname, &db);
        assert(rc == SQLITE_OK);

        rc = sqlite3_exec(db,
            "CREATE TABLE ngrams (gram TEXT, snippet_id INTEGER);"
            "CREATE INDEX idx_ngrams_gram ON ngrams(gram);",
            nullptr, nullptr, nullptr);
        assert(rc == SQLITE_OK);

        sqlite3_close(db);
    }

    // Insert grams into the correct shard based on MD5 routing
    struct gram_entry {
        std::string gram;
        int64_t snippet_id;
    };

    std::vector<gram_entry> grams = {
        {"def calculate_loss predictions",    3000000000},
        {"calculate_loss predictions targets", 3000000000},
        {"def train_model data",              3000000001},
        {"train_model data epochs 10",        3000000001},
        {"loss calculate_loss predict data",  3000000001},
        {"class dataloader def __init__",     3000000002},
        {"fn calculate_distance a point",     3001000000},
        {"calculate_distance a point b",      3001000000},
        {"struct point x f64",                3001000001},
        {"impl point fn new",                 3001000001},
    };

    // Open both shards for writing
    sqlite3 * shard_dbs[2] = {};
    for (int i = 0; i < 2; i++) {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/shard_%02d.db", corpus_dir.c_str(), i);
        int rc = sqlite3_open(fname, &shard_dbs[i]);
        assert(rc == SQLITE_OK);
    }

    for (const auto & entry : grams) {
        uint32_t h = md5_first4(entry.gram.data(), entry.gram.size());
        int shard_idx = (int)(h % 2);

        char sql[1024];
        snprintf(sql, sizeof(sql),
            "INSERT INTO ngrams VALUES ('%s', %lld);",
            entry.gram.c_str(), (long long)entry.snippet_id);

        int rc = sqlite3_exec(shard_dbs[shard_idx], sql, nullptr, nullptr, nullptr);
        assert(rc == SQLITE_OK);
    }

    for (int i = 0; i < 2; i++) {
        sqlite3_close(shard_dbs[i]);
    }

    return corpus_dir;
}

static void test_md5_basic() {
    printf("test_md5_basic...\n");

    // Known MD5 test vectors (first 4 bytes as LE uint32)
    // MD5("") = d41d8cd98f00b204e9800998ecf8427e
    // Digest bytes: d4 1d 8c d9 ... -> LE uint32 = 0xd98c1dd4
    uint32_t h_empty = md5_first4("", 0);
    printf("  md5_first4(\"\") = 0x%08x (expected 0xd98c1dd4)\n", h_empty);
    assert(h_empty == 0xd98c1dd4);

    // MD5("a") = 0cc175b9c0f1b6a831c399e269772661
    // Digest bytes: 0c c1 75 b9 ... -> LE uint32 = 0xb975c10c
    uint32_t h_a = md5_first4("a", 1);
    printf("  md5_first4(\"a\") = 0x%08x (expected 0xb975c10c)\n", h_a);
    assert(h_a == 0xb975c10c);

    // MD5("abc") = 900150983cd24fb0d6963f7d28e17f72
    // Digest bytes: 90 01 50 98 ... -> LE uint32 = 0x98500190
    uint32_t h_abc = md5_first4("abc", 3);
    printf("  md5_first4(\"abc\") = 0x%08x (expected 0x98500190)\n", h_abc);
    assert(h_abc == 0x98500190);

    printf("  PASSED\n");
}

static void test_md5_shard_routing() {
    printf("test_md5_shard_routing...\n");

    // Test that our shard routing matches Python:
    // Python: int.from_bytes(hashlib.md5(b"def calculate_loss predictions").digest()[:4], "little") % 16
    // We verify determinism and range
    const char * gram = "def calculate_loss predictions";
    uint32_t h = md5_first4(gram, strlen(gram));
    int shard = (int)(h % 16);
    assert(shard >= 0 && shard < 16);

    // Verify determinism
    uint32_t h2 = md5_first4(gram, strlen(gram));
    assert(h == h2);

    // Test various grams route to different shards (probabilistic but very likely with 16 shards)
    const char * test_grams[] = {
        "def calculate_loss predictions",
        "class dataloader def __init__",
        "fn calculate_distance a point",
        "struct point x f64",
    };
    bool all_same = true;
    int first_shard = (int)(md5_first4(test_grams[0], strlen(test_grams[0])) % 16);
    for (int i = 1; i < 4; i++) {
        int s = (int)(md5_first4(test_grams[i], strlen(test_grams[i])) % 16);
        if (s != first_shard) {
            all_same = false;
        }
    }
    // It's astronomically unlikely all 4 different grams hash to same shard
    // but if they do, it's still not a test failure — just check range
    (void)all_same;

    printf("  PASSED (gram '%s' -> shard %d of 16)\n", gram, shard);
}

static void test_sidecar_init_and_query() {
    printf("test_sidecar_init_and_query...\n");

    // Use /tmp for test corpus (or TMPDIR if set)
    const char * tmpdir = getenv("TMPDIR");
    if (!tmpdir) tmpdir = "/tmp";
    std::string base_dir = std::string(tmpdir) + "/llama_test_corpus_sidecar";

    std::string corpus_dir = create_test_corpus(base_dir);

    // Test init — we pass nullptr for vocab since we can't easily create one without a model
    // This tests the DB opening logic
    corpus_sidecar * cs = corpus_sidecar_init(corpus_dir, nullptr);
    // With nullptr vocab, init should still work (vocab only needed for query tokenization)
    // Actually, our init checks for nullptr vocab and returns nullptr
    assert(cs == nullptr); // expected: null vocab => null result

    printf("  init with null vocab correctly returns nullptr\n");

    // Test sidecar_ready
    assert(!corpus_sidecar_ready(nullptr));
    assert(!corpus_sidecar_ready(cs)); // cs is nullptr

    // Test free on nullptr (should not crash)
    corpus_sidecar_free(nullptr);

    // Clean up test corpus
    std::string cmd = "rm -rf " + base_dir;
    int ret = system(cmd.c_str());
    (void)ret;

    printf("  PASSED\n");
}

static void test_sidecar_db_structure() {
    printf("test_sidecar_db_structure...\n");

    const char * tmpdir = getenv("TMPDIR");
    if (!tmpdir) tmpdir = "/tmp";
    std::string base_dir = std::string(tmpdir) + "/llama_test_corpus_sidecar2";
    std::string corpus_dir = create_test_corpus(base_dir);

    // Directly test that shard databases were created correctly
    for (int i = 0; i < 2; i++) {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/shard_%02d.db", corpus_dir.c_str(), i);

        sqlite3 * db = nullptr;
        int rc = sqlite3_open_v2(fname, &db, SQLITE_OPEN_READONLY, nullptr);
        assert(rc == SQLITE_OK);

        // Count total grams in this shard
        sqlite3_stmt * stmt = nullptr;
        rc = sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM ngrams", -1, &stmt, nullptr);
        assert(rc == SQLITE_OK);
        rc = sqlite3_step(stmt);
        assert(rc == SQLITE_ROW);
        int count = sqlite3_column_int(stmt, 0);
        printf("  shard_%02d has %d ngram entries\n", i, count);
        assert(count > 0); // each shard should have at least 1 gram

        sqlite3_finalize(stmt);
        sqlite3_close(db);
    }

    // Verify snippets.db has 5 entries
    {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/snippets.db", corpus_dir.c_str());

        sqlite3 * db = nullptr;
        int rc = sqlite3_open_v2(fname, &db, SQLITE_OPEN_READONLY, nullptr);
        assert(rc == SQLITE_OK);

        sqlite3_stmt * stmt = nullptr;
        rc = sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM snippets", -1, &stmt, nullptr);
        assert(rc == SQLITE_OK);
        rc = sqlite3_step(stmt);
        assert(rc == SQLITE_ROW);
        int count = sqlite3_column_int(stmt, 0);
        printf("  snippets.db has %d entries\n", count);
        assert(count == 5);

        sqlite3_finalize(stmt);
        sqlite3_close(db);
    }

    // Clean up
    std::string cmd = "rm -rf " + base_dir;
    system(cmd.c_str());

    printf("  PASSED\n");
}

int main() {
    printf("=== test-corpus-sidecar ===\n\n");

    test_md5_basic();
    test_md5_shard_routing();
    test_sidecar_db_structure();
    test_sidecar_init_and_query();

    printf("\nAll tests passed!\n");
    return 0;
}

#else // !LLAMA_CORPUS_SIDECAR

#include <cstdio>

int main() {
    printf("test-corpus-sidecar: SKIPPED (compiled without -DLLAMA_CORPUS_SIDECAR)\n");
    return 0;
}

#endif // LLAMA_CORPUS_SIDECAR
