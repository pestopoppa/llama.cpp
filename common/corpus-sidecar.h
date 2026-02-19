#pragma once

#ifdef LLAMA_CORPUS_SIDECAR

#include "llama.h"
#include "ngram-cache.h"

#include <sqlite3.h>

#define CORPUS_SIDECAR_MAX_SHARDS 16

struct corpus_sidecar {
    sqlite3      * shard_dbs[CORPUS_SIDECAR_MAX_SHARDS];
    sqlite3      * snippets_db;
    sqlite3_stmt * shard_stmts[CORPUS_SIDECAR_MAX_SHARDS]; // SELECT snippet_id FROM ngrams WHERE gram = ?
    sqlite3_stmt * snippet_stmt;                            // SELECT code FROM snippets WHERE id = ?
    const llama_vocab * vocab;
    int num_shards;
};

// Initialize the corpus sidecar from a directory containing shard_XX.db and snippets.db.
// Returns nullptr on failure. The vocab is needed for tokenization of retrieved snippets.
corpus_sidecar * corpus_sidecar_init(const std::string & corpus_path, const llama_vocab * vocab);

// Query the corpus for snippets matching recent token context, and populate nc_static
// with n-gram entries extracted from the retrieved code snippets.
//   cs:           initialized corpus sidecar
//   recent_tokens: last N tokens of generation context (typically 32)
//   nc_static:    output ngram cache to populate
//   max_snippets: maximum number of snippets to retrieve (default: 8)
void corpus_sidecar_query(
    corpus_sidecar * cs,
    const std::vector<llama_token> & recent_tokens,
    common_ngram_cache & nc_static,
    int max_snippets = 8);

// Free the corpus sidecar and close all database connections.
void corpus_sidecar_free(corpus_sidecar * cs);

// Check if the corpus sidecar is initialized and ready.
bool corpus_sidecar_ready(const corpus_sidecar * cs);

#endif // LLAMA_CORPUS_SIDECAR
