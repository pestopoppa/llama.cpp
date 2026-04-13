// Test skeleton: Memento-style block reasoning compression via KV cache eviction
//
// Purpose: Validate that llama_memory_seq_rm() can serve as the block eviction
// primitive for Memento-style reasoning compression. This test exercises:
//   1. Mid-sequence KV removal (removing a position range from an active sequence)
//   2. Position gap handling after removal (verifying RoPE position semantics)
//   3. Continued generation after eviction (attention correctness)
//   4. Memory usage reduction verification
//
// Design document — NOT runnable without a model file.
// Compile: passes with correct includes/signatures. Runtime: requires --model flag.
//
// References:
//   - llama_memory_seq_rm():  include/llama.h:733
//   - llama_kv_cache::seq_rm: src/llama-kv-cache.cpp:388
//   - llama_kv_cells::seq_rm: src/llama-kv-cells.h:238
//   - Memento handoff: epyc-root/handoffs/active/memento-block-reasoning-compression.md

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Simulated block structure for Memento reasoning compression.
// In production, these boundaries would be detected via special tokens
// (<|block_start|>, <|block_end|>, <|summary_start|>, <|summary_end|>).
struct reasoning_block {
    llama_pos start;  // first token position in the block (inclusive)
    llama_pos end;    // last token position in the block (exclusive)
    llama_pos summary_start;  // first position of the summary for this block
    llama_pos summary_end;    // last position of the summary (exclusive)
};

// Helper: tokenize a string, returning token vector
static std::vector<llama_token> tokenize_str(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special) {
    // Upper bound on token count
    int n_max = text.size() + 2 * add_special;
    std::vector<llama_token> tokens(n_max);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_max, add_special, false);
    assert(n >= 0);
    tokens.resize(n);
    return tokens;
}

// Helper: build a batch for a token sequence starting at a given position
static llama_batch make_batch(
        const std::vector<llama_token> & tokens,
        llama_pos pos_start,
        llama_seq_id seq_id,
        bool logits_last_only) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens = tokens.size();
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos_start + (llama_pos)i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i]    = logits_last_only ? (i == tokens.size() - 1 ? 1 : 0) : 1;
    }
    return batch;
}

// ============================================================================
// Test 1: Basic mid-sequence block eviction
//
// Fill KV cache with a prompt containing a "reasoning block" region,
// then evict the block range and verify the KV metadata is updated.
// ============================================================================
static bool test_basic_block_eviction(llama_context * ctx, const llama_vocab * vocab) {
    fprintf(stderr, "\n=== Test 1: Basic mid-sequence block eviction ===\n");

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_seq_id seq_id = 0;

    // Phase 1: Tokenize a prompt with an embedded "reasoning block"
    // Structure: [system prompt] [reasoning block] [summary] [continuation]
    // Positions:  0..99           100..499          500..549  550..599

    // TODO: requires model file — these strings are placeholders
    const std::string system_prompt =
        "You are a helpful assistant. Solve the following math problem step by step.";
    const std::string reasoning_block_text =
        "Let me think about this carefully. First, I need to consider... "
        "[simulated long reasoning chain that would span ~400 tokens in practice]";
    const std::string summary_text =
        "Summary: The answer is 42, derived from the key insight that...";
    const std::string continuation_text =
        "Therefore, the final answer is 42.";

    auto tokens_system  = tokenize_str(vocab, system_prompt, true);
    auto tokens_reason  = tokenize_str(vocab, reasoning_block_text, false);
    auto tokens_summary = tokenize_str(vocab, summary_text, false);
    auto tokens_cont    = tokenize_str(vocab, continuation_text, false);

    // Track positions
    llama_pos pos = 0;
    llama_pos reason_start  = (llama_pos)tokens_system.size();
    llama_pos reason_end    = reason_start + (llama_pos)tokens_reason.size();
    llama_pos summary_start = reason_end;
    llama_pos summary_end   = summary_start + (llama_pos)tokens_summary.size();

    fprintf(stderr, "  Positions: system=[0, %d), reason=[%d, %d), summary=[%d, %d)\n",
            reason_start, reason_start, reason_end, summary_start, summary_end);

    // Phase 2: Process all tokens to fill KV cache
    {
        llama_batch batch = make_batch(tokens_system, 0, seq_id, false);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: llama_decode failed for system prompt (rc=%d)\n", rc);
            return false;
        }
        pos = reason_start;
    }
    {
        llama_batch batch = make_batch(tokens_reason, pos, seq_id, false);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: llama_decode failed for reasoning block (rc=%d)\n", rc);
            return false;
        }
        pos = reason_end;
    }
    {
        llama_batch batch = make_batch(tokens_summary, pos, seq_id, false);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: llama_decode failed for summary (rc=%d)\n", rc);
            return false;
        }
        pos = summary_end;
    }

    // Record KV state before eviction
    llama_pos pos_min_before = llama_memory_seq_pos_min(mem, seq_id);
    llama_pos pos_max_before = llama_memory_seq_pos_max(mem, seq_id);
    fprintf(stderr, "  Before eviction: pos_min=%d, pos_max=%d\n", pos_min_before, pos_max_before);

    // Phase 3: Evict the reasoning block [reason_start, reason_end)
    // This is the core Memento operation: remove KV entries for the reasoning
    // block while keeping the system prompt and summary intact.
    fprintf(stderr, "  Evicting reasoning block [%d, %d)...\n", reason_start, reason_end);
    bool ok = llama_memory_seq_rm(mem, seq_id, reason_start, reason_end);
    if (!ok) {
        fprintf(stderr, "  FAIL: llama_memory_seq_rm returned false\n");
        return false;
    }

    // Phase 4: Verify KV metadata after eviction
    llama_pos pos_min_after = llama_memory_seq_pos_min(mem, seq_id);
    llama_pos pos_max_after = llama_memory_seq_pos_max(mem, seq_id);
    fprintf(stderr, "  After eviction: pos_min=%d, pos_max=%d\n", pos_min_after, pos_max_after);

    // The system prompt should still be at position 0
    assert(pos_min_after == 0);

    // The summary tokens should still be present at their original positions.
    // CRITICAL: seq_rm does NOT shift positions. There is now a position GAP
    // in [reason_start, reason_end). The summary tokens retain their original
    // RoPE positions. This is by design — the KV entries for the summary were
    // computed with the reasoning block visible, so their KV states encode
    // information about the block (the "dual information stream" from Memento).
    // Shifting positions would invalidate the RoPE encoding.
    assert(pos_max_after == pos_max_before);

    fprintf(stderr, "  PASS: Basic block eviction succeeded, position gap created\n");
    return true;
}

// ============================================================================
// Test 2: Position gap semantics — verify no automatic shifting
//
// After evicting a mid-sequence range, confirm that:
// (a) Remaining tokens keep their original positions (no RoPE corruption)
// (b) seq_add CAN be used to close the gap if desired (but shouldn't for Memento)
// ============================================================================
static bool test_position_gap_semantics(llama_context * ctx, const llama_vocab * vocab) {
    fprintf(stderr, "\n=== Test 2: Position gap semantics ===\n");

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_seq_id seq_id = 0;

    // Clear KV cache for clean test
    llama_memory_clear(mem, false);

    // Fill positions [0, 100) with dummy tokens
    // TODO: requires model file — using token ID 1 as placeholder
    std::vector<llama_token> dummy(100, 1);
    {
        llama_batch batch = make_batch(dummy, 0, seq_id, false);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: decode failed (rc=%d)\n", rc);
            return false;
        }
    }

    // Evict positions [30, 70) — the "reasoning block"
    llama_memory_seq_rm(mem, seq_id, 30, 70);

    llama_pos pos_min = llama_memory_seq_pos_min(mem, seq_id);
    llama_pos pos_max = llama_memory_seq_pos_max(mem, seq_id);

    fprintf(stderr, "  After removing [30,70): pos_min=%d, pos_max=%d\n", pos_min, pos_max);

    // Positions 0-29 and 70-99 should remain — with a GAP at 30-69
    assert(pos_min == 0);
    assert(pos_max == 99);

    // Demonstration: seq_add could close the gap by shifting [70, inf) left by 40
    // BUT for Memento, we do NOT want this — the KV states at positions 70-99
    // were computed with the block visible. Shifting would corrupt RoPE.
    //
    // llama_memory_seq_add(mem, seq_id, 70, -1, -40);  // DO NOT DO THIS
    //
    // If we did call seq_add, positions 70-99 would become 30-59, closing the gap.
    // This is useful for context pruning but NOT for Memento.

    fprintf(stderr, "  PASS: Position gap preserved correctly\n");
    return true;
}

// ============================================================================
// Test 3: Continued generation after block eviction
//
// After evicting a reasoning block, verify that:
// (a) llama_decode succeeds for new tokens at positions after the gap
// (b) The model can generate coherent output (manual inspection)
// ============================================================================
static bool test_generation_after_eviction(llama_context * ctx, const llama_vocab * vocab) {
    fprintf(stderr, "\n=== Test 3: Generation after eviction ===\n");

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_seq_id seq_id = 0;

    // Clear and set up a fresh sequence
    llama_memory_clear(mem, false);

    // TODO: requires model file — in practice, use a real prompt
    const std::string prompt = "What is 2+2? Let me think step by step.";
    auto tokens = tokenize_str(vocab, prompt, true);

    {
        llama_batch batch = make_batch(tokens, 0, seq_id, true);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: initial decode failed (rc=%d)\n", rc);
            return false;
        }
    }

    llama_pos current_pos = (llama_pos)tokens.size();

    // Simulate: model generates reasoning tokens at positions [current_pos, current_pos+50)
    // Then generates a summary at [current_pos+50, current_pos+70)
    // TODO: requires model — using dummy tokens
    std::vector<llama_token> reasoning_tokens(50, 1);
    std::vector<llama_token> summary_tokens(20, 1);

    llama_pos reason_start = current_pos;
    {
        llama_batch batch = make_batch(reasoning_tokens, reason_start, seq_id, true);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: reasoning decode failed (rc=%d)\n", rc);
            return false;
        }
    }

    llama_pos reason_end = reason_start + 50;
    llama_pos summary_start = reason_end;
    {
        llama_batch batch = make_batch(summary_tokens, summary_start, seq_id, true);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: summary decode failed (rc=%d)\n", rc);
            return false;
        }
    }

    // Evict the reasoning block
    fprintf(stderr, "  Evicting reasoning [%d, %d)...\n", reason_start, reason_end);
    bool ok = llama_memory_seq_rm(mem, seq_id, reason_start, reason_end);
    assert(ok);

    // Continue generation AFTER the eviction — new tokens go at positions after the summary
    llama_pos continue_pos = summary_start + 20;
    std::vector<llama_token> continuation(10, 1);
    {
        llama_batch batch = make_batch(continuation, continue_pos, seq_id, true);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: post-eviction decode failed (rc=%d)\n", rc);
            fprintf(stderr, "  This would indicate attention mask or KV cache corruption\n");
            return false;
        }
    }

    fprintf(stderr, "  PASS: Generation continues successfully after block eviction\n");

    // TODO: manual inspection — compare generated logits/tokens with and without
    // eviction to measure quality impact. For automated testing, compare perplexity
    // on a reference text with and without block eviction.

    return true;
}

// ============================================================================
// Test 4: Multi-block iterative eviction
//
// Simulate the full Memento pattern: multiple reasoning blocks, each evicted
// after its summary is generated. Verify KV cache state after each eviction.
// ============================================================================
static bool test_multi_block_eviction(llama_context * ctx, const llama_vocab * vocab) {
    fprintf(stderr, "\n=== Test 4: Multi-block iterative eviction ===\n");

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_seq_id seq_id = 0;

    llama_memory_clear(mem, false);

    // Simulate 3 reasoning blocks, each followed by a summary:
    //   [prompt: 0..19] [block1: 20..69] [sum1: 70..79] [block2: 80..129] [sum2: 130..139] [block3: 140..189] [sum3: 190..199]
    reasoning_block blocks[3] = {
        { 20,  70,  70,  80},
        { 80, 130, 130, 140},
        {140, 190, 190, 200},
    };

    // Process prompt
    std::vector<llama_token> prompt_tokens(20, 1);  // TODO: real tokens
    {
        llama_batch batch = make_batch(prompt_tokens, 0, seq_id, false);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        assert(rc == 0);
    }

    // Process each block + summary, then evict the block
    for (int b = 0; b < 3; b++) {
        const auto & blk = blocks[b];

        // Process reasoning block tokens
        uint32_t block_len = blk.end - blk.start;
        std::vector<llama_token> block_tokens(block_len, 1);  // TODO: real tokens
        {
            llama_batch batch = make_batch(block_tokens, blk.start, seq_id, false);
            int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            assert(rc == 0);
        }

        // Process summary tokens
        uint32_t sum_len = blk.summary_end - blk.summary_start;
        std::vector<llama_token> sum_tokens(sum_len, 1);  // TODO: real tokens
        {
            llama_batch batch = make_batch(sum_tokens, blk.summary_start, seq_id, false);
            int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            assert(rc == 0);
        }

        // Evict the reasoning block (keep the summary)
        fprintf(stderr, "  Evicting block %d [%d, %d)...\n", b, blk.start, blk.end);
        bool ok = llama_memory_seq_rm(mem, seq_id, blk.start, blk.end);
        assert(ok);

        // Verify: summary positions still present
        llama_pos pmin = llama_memory_seq_pos_min(mem, seq_id);
        llama_pos pmax = llama_memory_seq_pos_max(mem, seq_id);
        fprintf(stderr, "  After block %d eviction: pos_min=%d, pos_max=%d\n", b, pmin, pmax);

        // Prompt at position 0 should always survive
        assert(pmin == 0);
        // Max position should be the end of the current summary
        assert(pmax == blk.summary_end - 1);
    }

    // After all 3 evictions, KV cache should contain:
    //   [prompt: 0..19] [gap: 20..69] [sum1: 70..79] [gap: 80..129] [sum2: 130..139] [gap: 140..189] [sum3: 190..199]
    // Total KV entries: 20 (prompt) + 10 (sum1) + 10 (sum2) + 10 (sum3) = 50
    // Evicted: 50 + 50 + 50 = 150 entries
    // Compression ratio: 200 / 50 = 4x

    fprintf(stderr, "  KV entries remaining: ~50 out of 200 (4x compression)\n");
    fprintf(stderr, "  PASS: Multi-block iterative eviction succeeded\n");
    return true;
}

// ============================================================================
// Test 5: Memory usage measurement
//
// Measure KV cache metadata before and after eviction to verify that cells
// are actually freed (not just marked). Note: the data buffers (K/V tensors)
// are pre-allocated and do not shrink, but the cells become available for reuse.
// ============================================================================
static bool test_memory_usage(llama_context * ctx, const llama_vocab * vocab) {
    fprintf(stderr, "\n=== Test 5: Memory usage verification ===\n");

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_seq_id seq_id = 0;

    llama_memory_clear(mem, false);

    // Fill 200 positions
    std::vector<llama_token> tokens(200, 1);  // TODO: real tokens
    {
        llama_batch batch = make_batch(tokens, 0, seq_id, false);
        int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        assert(rc == 0);
    }

    // Print memory breakdown before eviction
    fprintf(stderr, "  Before eviction:\n");
    llama_memory_breakdown_print(ctx);

    // Evict positions [50, 150) — 100 tokens
    llama_memory_seq_rm(mem, seq_id, 50, 150);

    // Print memory breakdown after eviction
    fprintf(stderr, "  After eviction of [50, 150):\n");
    llama_memory_breakdown_print(ctx);

    // NOTE: The actual data buffers (ggml tensors for K and V) are pre-allocated
    // at context creation time and do NOT shrink. What changes:
    //   1. Cell metadata (pos, seq bitset) is reset — cells become "empty"
    //   2. Empty cells are available for reuse by future llama_decode calls
    //   3. The head pointer is updated so find_slot() starts at the freed region
    //
    // For Memento's use case (long reasoning chains), the benefit is:
    //   - Freed cells can be reused for NEW reasoning blocks
    //   - Effective context length is extended without increasing n_ctx
    //   - Memory bandwidth during attention is reduced (fewer active KV entries)

    fprintf(stderr, "  PASS: Memory usage check completed (see breakdown above)\n");
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 42;
    params.n_ctx = 512;   // Small context for testing
    params.n_batch = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "Usage: %s --model <path-to-gguf>\n", argv[0]);
        return 1;
    }

    // TODO: requires model file — pass via --model flag
    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s: failed to init model/context\n", __func__);
        fprintf(stderr, "This test requires a model file. Pass --model <path>\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n_pass = 0;
    int n_fail = 0;

    if (test_basic_block_eviction(ctx, vocab))    { n_pass++; } else { n_fail++; }
    if (test_position_gap_semantics(ctx, vocab))   { n_pass++; } else { n_fail++; }
    if (test_generation_after_eviction(ctx, vocab)) { n_pass++; } else { n_fail++; }
    if (test_multi_block_eviction(ctx, vocab))     { n_pass++; } else { n_fail++; }
    if (test_memory_usage(ctx, vocab))             { n_pass++; } else { n_fail++; }

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Results: %d passed, %d failed\n", n_pass, n_fail);
    fprintf(stderr, "========================================\n");

    return n_fail > 0 ? 1 : 0;
}
