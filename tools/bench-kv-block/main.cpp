// Benchmark for KV cache block tracking
// Tests the PagedAttention Phase 1 implementation:
// - Block tracking overhead
// - Fragmentation measurement
// - Block-aligned allocation

#include "llama.h"
#include "llama-kv-block.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

static void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -h, --help           Show this help\n");
    printf("  --block-size N       Tokens per block (default: 64)\n");
    printf("  --cache-size N       KV cache size in tokens (default: 4096)\n");
    printf("  --n-seqs N           Number of sequences to simulate (default: 8)\n");
    printf("  --n-iters N          Number of iterations (default: 1000)\n");
}

int main(int argc, char ** argv) {
    uint32_t block_size = 64;
    uint32_t cache_size = 4096;
    uint32_t n_seqs = 8;
    uint32_t n_iters = 1000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            block_size = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cache-size") == 0 && i + 1 < argc) {
            cache_size = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-seqs") == 0 && i + 1 < argc) {
            n_seqs = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-iters") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        }
    }

    printf("KV Block Tracking Benchmark\n");
    printf("===========================\n");
    printf("Block size:  %u tokens\n", block_size);
    printf("Cache size:  %u tokens\n", cache_size);
    printf("Sequences:   %u\n", n_seqs);
    printf("Iterations:  %u\n", n_iters);
    printf("\n");

    //
    // Test 1: Block pool allocation overhead
    //
    printf("Test 1: Block pool allocation overhead\n");
    {
        const uint32_t n_blocks = tokens_to_blocks(cache_size, block_size);

        llama_kv_block_pool pool;
        pool.init(n_blocks);

        auto start = std::chrono::high_resolution_clock::now();

        for (uint32_t iter = 0; iter < n_iters; iter++) {
            // Allocate all blocks
            std::vector<int32_t> allocated;
            allocated.reserve(n_blocks);
            for (uint32_t i = 0; i < n_blocks; i++) {
                int32_t idx = pool.allocate();
                if (idx >= 0) {
                    allocated.push_back(idx);
                    pool.get(idx).n_tokens = block_size;
                }
            }

            // Deallocate all
            for (int32_t idx : allocated) {
                pool.deallocate(idx);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        printf("  %u alloc+dealloc cycles: %.2f ms (%.2f us/cycle)\n",
               n_iters, duration.count() / 1000.0, (double)duration.count() / n_iters);
    }

    //
    // Test 2: Block table operations
    //
    printf("\nTest 2: Block table operations\n");
    {
        llama_kv_block_table table;

        auto start = std::chrono::high_resolution_clock::now();

        for (uint32_t iter = 0; iter < n_iters; iter++) {
            // Simulate sequence creation
            for (uint32_t seq = 0; seq < n_seqs; seq++) {
                uint32_t n_blocks_per_seq = cache_size / block_size / n_seqs;
                for (uint32_t b = 0; b < n_blocks_per_seq; b++) {
                    table.append_block(seq, seq * n_blocks_per_seq + b);
                }
            }

            // Lookup operations
            for (uint32_t seq = 0; seq < n_seqs; seq++) {
                uint32_t n_blocks_per_seq = cache_size / block_size / n_seqs;
                for (uint32_t b = 0; b < n_blocks_per_seq; b++) {
                    (void)table.get_physical(seq, b);
                }
            }

            // Clear
            table.clear();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        printf("  %u table cycles: %.2f ms (%.2f us/cycle)\n",
               n_iters, duration.count() / 1000.0, (double)duration.count() / n_iters);
    }

    //
    // Test 3: Statistics computation
    //
    printf("\nTest 3: Statistics computation overhead\n");
    {
        const uint32_t n_blocks = tokens_to_blocks(cache_size, block_size);

        llama_kv_block_pool pool;
        pool.init(n_blocks);

        // Fill half the blocks with varying token counts
        for (uint32_t i = 0; i < n_blocks / 2; i++) {
            int32_t idx = pool.allocate();
            if (idx >= 0) {
                // Simulate varying fill levels
                pool.get(idx).n_tokens = (i % block_size) + 1;
            }
        }

        auto start = std::chrono::high_resolution_clock::now();

        llama_kv_block_stats stats;
        for (uint32_t iter = 0; iter < n_iters * 10; iter++) {
            stats = pool.compute_stats(block_size);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        printf("  %u stat computations: %.2f ms (%.2f us/call)\n",
               n_iters * 10, duration.count() / 1000.0, (double)duration.count() / (n_iters * 10));
        printf("  Final stats: blocks=%u/%u, tokens=%u/%u, frag=%.1f%%, util=%.1f%%\n",
               stats.n_blocks_used, stats.n_blocks_total,
               stats.n_tokens_used, stats.n_tokens_total,
               stats.fragmentation * 100, stats.utilization * 100);
    }

    //
    // Test 4: Fragmentation simulation
    //
    printf("\nTest 4: Fragmentation patterns\n");
    {
        const uint32_t n_blocks = tokens_to_blocks(cache_size, block_size);

        // Pattern A: Sequential fill (low fragmentation)
        {
            llama_kv_block_pool pool;
            pool.init(n_blocks);

            for (uint32_t i = 0; i < n_blocks * 3 / 4; i++) {
                int32_t idx = pool.allocate();
                if (idx >= 0) {
                    pool.get(idx).n_tokens = block_size; // Full blocks
                }
            }

            auto stats = pool.compute_stats(block_size);
            printf("  Sequential full blocks: frag=%.1f%%, util=%.1f%%\n",
                   stats.fragmentation * 100, stats.utilization * 100);
        }

        // Pattern B: Half-filled blocks (high internal fragmentation)
        {
            llama_kv_block_pool pool;
            pool.init(n_blocks);

            for (uint32_t i = 0; i < n_blocks * 3 / 4; i++) {
                int32_t idx = pool.allocate();
                if (idx >= 0) {
                    pool.get(idx).n_tokens = block_size / 2; // Half-full blocks
                }
            }

            auto stats = pool.compute_stats(block_size);
            printf("  Half-filled blocks:     frag=%.1f%%, util=%.1f%%\n",
                   stats.fragmentation * 100, stats.utilization * 100);
        }

        // Pattern C: Varying fill (realistic)
        {
            llama_kv_block_pool pool;
            pool.init(n_blocks);

            for (uint32_t i = 0; i < n_blocks * 3 / 4; i++) {
                int32_t idx = pool.allocate();
                if (idx >= 0) {
                    // Simulate realistic pattern: most blocks full, some partial
                    uint32_t tokens = (i % 4 == 0) ? (block_size * 2 / 3) : block_size;
                    pool.get(idx).n_tokens = tokens;
                }
            }

            auto stats = pool.compute_stats(block_size);
            printf("  Realistic pattern:      frag=%.1f%%, util=%.1f%%\n",
                   stats.fragmentation * 100, stats.utilization * 100);
        }
    }

    //
    // Test 5: Helper function performance
    //
    printf("\nTest 5: Helper functions\n");
    {
        auto start = std::chrono::high_resolution_clock::now();

        volatile uint32_t sum = 0;
        for (uint32_t iter = 0; iter < n_iters * 1000; iter++) {
            sum += cell_to_block(iter, block_size);
            sum += block_to_cell(iter % 100, block_size);
            sum += cell_offset_in_block(iter, block_size);
            sum += tokens_to_blocks(iter % cache_size, block_size);
        }
        (void)sum;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        printf("  %u helper calls: %.2f ms (%.2f ns/call)\n",
               n_iters * 1000 * 4, duration.count() / 1e6, (double)duration.count() / (n_iters * 1000 * 4));
    }

    printf("\n===========================\n");
    printf("Benchmark complete.\n");

    return 0;
}
