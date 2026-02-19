#ifdef LLAMA_CORPUS_SIDECAR

#include "corpus-sidecar.h"
#include "md5.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

// Detokenize recent tokens to text using the vocab
static std::string detokenize_tokens(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string result;
    for (const auto & token : tokens) {
        char buf[256];
        int n = llama_detokenize(vocab, &token, 1, buf, sizeof(buf), false, false);
        if (n > 0) {
            result.append(buf, n);
        }
    }
    return result;
}

// Normalize text: lowercase, keep [a-z0-9_], spaces between words
static std::string normalize_text(const std::string & text) {
    std::string result;
    result.reserve(text.size());
    bool last_was_sep = true;
    for (char c : text) {
        char lc = (char)std::tolower((unsigned char)c);
        if ((lc >= 'a' && lc <= 'z') || (lc >= '0' && lc <= '9') || lc == '_') {
            result.push_back(lc);
            last_was_sep = false;
        } else if (!last_was_sep) {
            result.push_back(' ');
            last_was_sep = true;
        }
    }
    // trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

// Extract word-level 4-grams from normalized text
static std::vector<std::string> extract_4grams(const std::string & normalized) {
    std::vector<std::string> words;
    size_t start = 0;
    while (start < normalized.size()) {
        size_t end = normalized.find(' ', start);
        if (end == std::string::npos) end = normalized.size();
        if (end > start) {
            words.push_back(normalized.substr(start, end - start));
        }
        start = end + 1;
    }

    std::vector<std::string> grams;
    if (words.size() < 4) {
        return grams;
    }
    for (size_t i = 0; i <= words.size() - 4; i++) {
        std::string gram = words[i] + " " + words[i+1] + " " + words[i+2] + " " + words[i+3];
        grams.push_back(gram);
    }
    return grams;
}

// Route a gram to a shard index using MD5
static int gram_to_shard(const std::string & gram, int num_shards) {
    uint32_t h = md5_first4(gram.data(), gram.size());
    return (int)(h % (uint32_t)num_shards);
}

corpus_sidecar * corpus_sidecar_init(const std::string & corpus_path, const llama_vocab * vocab) {
    if (corpus_path.empty() || vocab == nullptr) {
        return nullptr;
    }

    auto * cs = new corpus_sidecar();
    memset(cs, 0, sizeof(corpus_sidecar));
    cs->vocab = vocab;

    // Detect number of shards
    cs->num_shards = 0;
    for (int i = 0; i < CORPUS_SIDECAR_MAX_SHARDS; i++) {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/shard_%02d.db", corpus_path.c_str(), i);
        FILE * f = fopen(fname, "r");
        if (f) {
            fclose(f);
            cs->num_shards = i + 1;
        } else {
            break;
        }
    }

    if (cs->num_shards == 0) {
        LOG_ERR("%s: no shard databases found in %s\n", __func__, corpus_path.c_str());
        delete cs;
        return nullptr;
    }

    // Open shard databases
    for (int i = 0; i < cs->num_shards; i++) {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/shard_%02d.db", corpus_path.c_str(), i);

        int rc = sqlite3_open_v2(fname, &cs->shard_dbs[i], SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
        if (rc != SQLITE_OK) {
            LOG_ERR("%s: failed to open shard %d: %s\n", __func__, i, sqlite3_errmsg(cs->shard_dbs[i]));
            corpus_sidecar_free(cs);
            return nullptr;
        }

        // Performance pragmas
        sqlite3_exec(cs->shard_dbs[i], "PRAGMA mmap_size=1073741824;", nullptr, nullptr, nullptr);
        sqlite3_exec(cs->shard_dbs[i], "PRAGMA journal_mode=OFF;", nullptr, nullptr, nullptr);
        sqlite3_exec(cs->shard_dbs[i], "PRAGMA query_only=ON;", nullptr, nullptr, nullptr);

        // Prepare statement: find snippet_ids matching a gram, grouped by count
        const char * sql = "SELECT snippet_id, COUNT(*) as cnt FROM ngrams WHERE gram = ? GROUP BY snippet_id ORDER BY cnt DESC LIMIT 32";
        rc = sqlite3_prepare_v2(cs->shard_dbs[i], sql, -1, &cs->shard_stmts[i], nullptr);
        if (rc != SQLITE_OK) {
            LOG_ERR("%s: failed to prepare shard %d statement: %s\n", __func__, i, sqlite3_errmsg(cs->shard_dbs[i]));
            corpus_sidecar_free(cs);
            return nullptr;
        }
    }

    // Open snippets database
    {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/snippets.db", corpus_path.c_str());

        int rc = sqlite3_open_v2(fname, &cs->snippets_db, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
        if (rc != SQLITE_OK) {
            LOG_ERR("%s: failed to open snippets.db: %s\n", __func__, sqlite3_errmsg(cs->snippets_db));
            corpus_sidecar_free(cs);
            return nullptr;
        }

        sqlite3_exec(cs->snippets_db, "PRAGMA mmap_size=1073741824;", nullptr, nullptr, nullptr);
        sqlite3_exec(cs->snippets_db, "PRAGMA journal_mode=OFF;", nullptr, nullptr, nullptr);
        sqlite3_exec(cs->snippets_db, "PRAGMA query_only=ON;", nullptr, nullptr, nullptr);

        const char * sql = "SELECT code FROM snippets WHERE id = ?";
        rc = sqlite3_prepare_v2(cs->snippets_db, sql, -1, &cs->snippet_stmt, nullptr);
        if (rc != SQLITE_OK) {
            LOG_ERR("%s: failed to prepare snippet statement: %s\n", __func__, sqlite3_errmsg(cs->snippets_db));
            corpus_sidecar_free(cs);
            return nullptr;
        }
    }

    LOG_INF("%s: corpus sidecar initialized with %d shards from %s\n", __func__, cs->num_shards, corpus_path.c_str());
    return cs;
}

void corpus_sidecar_query(
        corpus_sidecar * cs,
        const std::vector<llama_token> & recent_tokens,
        common_ngram_cache & nc_static,
        int max_snippets) {
    if (!corpus_sidecar_ready(cs) || recent_tokens.empty()) {
        return;
    }

    // Step 1: Detokenize recent tokens to text
    std::string text = detokenize_tokens(cs->vocab, recent_tokens);
    if (text.empty()) {
        return;
    }

    // Step 2: Normalize and extract 4-grams
    std::string normalized = normalize_text(text);
    std::vector<std::string> grams = extract_4grams(normalized);
    if (grams.empty()) {
        return;
    }

    // Step 3: Query shards and aggregate snippet scores
    std::unordered_map<int64_t, int> snippet_scores;

    for (const auto & gram : grams) {
        int shard_idx = gram_to_shard(gram, cs->num_shards);
        sqlite3_stmt * stmt = cs->shard_stmts[shard_idx];

        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, gram.c_str(), (int)gram.size(), SQLITE_STATIC);

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int64_t snippet_id = sqlite3_column_int64(stmt, 0);
            int     count      = sqlite3_column_int(stmt, 1);
            snippet_scores[snippet_id] += count;
        }
    }

    if (snippet_scores.empty()) {
        return;
    }

    // Step 4: Sort by score descending, take top N
    std::vector<std::pair<int64_t, int>> ranked(snippet_scores.begin(), snippet_scores.end());
    std::sort(ranked.begin(), ranked.end(),
        [](const std::pair<int64_t, int> & a, const std::pair<int64_t, int> & b) {
            return a.second > b.second;
        });

    int n_fetch = std::min(max_snippets, (int)ranked.size());

    // Step 5: Fetch snippet code and tokenize, then update nc_static
    for (int i = 0; i < n_fetch; i++) {
        int64_t snippet_id = ranked[i].first;

        sqlite3_reset(cs->snippet_stmt);
        sqlite3_bind_int64(cs->snippet_stmt, 1, snippet_id);

        if (sqlite3_step(cs->snippet_stmt) != SQLITE_ROW) {
            continue;
        }

        const char * code = (const char *)sqlite3_column_text(cs->snippet_stmt, 0);
        int code_len = sqlite3_column_bytes(cs->snippet_stmt, 0);
        if (!code || code_len == 0) {
            continue;
        }

        // Tokenize the snippet
        std::string code_str(code, code_len);
        int n_tokens_max = code_len + 16; // rough upper bound
        std::vector<llama_token> tokens(n_tokens_max);
        int n_tokens = llama_tokenize(cs->vocab, code_str.c_str(), (int)code_str.size(),
                                      tokens.data(), n_tokens_max, false, true);
        if (n_tokens <= 0) {
            continue;
        }
        tokens.resize(n_tokens);

        // Update ngram cache with the tokenized snippet
        common_ngram_cache_update(nc_static, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                                  tokens, n_tokens, false);
    }
}

void corpus_sidecar_free(corpus_sidecar * cs) {
    if (!cs) {
        return;
    }

    for (int i = 0; i < cs->num_shards; i++) {
        if (cs->shard_stmts[i]) {
            sqlite3_finalize(cs->shard_stmts[i]);
        }
        if (cs->shard_dbs[i]) {
            sqlite3_close(cs->shard_dbs[i]);
        }
    }

    if (cs->snippet_stmt) {
        sqlite3_finalize(cs->snippet_stmt);
    }
    if (cs->snippets_db) {
        sqlite3_close(cs->snippets_db);
    }

    delete cs;
}

bool corpus_sidecar_ready(const corpus_sidecar * cs) {
    return cs != nullptr && cs->snippets_db != nullptr && cs->num_shards > 0;
}

#endif // LLAMA_CORPUS_SIDECAR
