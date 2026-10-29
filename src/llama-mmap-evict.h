#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_mmap_evict {

// The prototype owns ggml's single evaluation callback slot. It must refuse
// activation when a caller has already installed an observer.
bool can_enable(bool existing_eval_callback) noexcept;

struct mapping {
    uintptr_t address;
    size_t size;
};

struct range {
    size_t mapping_index;
    size_t offset;
    size_t size;
};

// Build a conservative, page-granular last-use plan from graph tensor uses.
// Overlapping and aliased tensor pages are retained until their latest use.
class plan {
public:
    explicit plan(std::vector<mapping> mappings);

    // Returns false when the tensor is not backed by any supplied mapping.
    // Throws if it partially overlaps a mapping or has invalid bounds.
    bool add_tensor_use(size_t node_index, const void * data, size_t bytes);
    void finalize(size_t node_count);

    const std::vector<range> & after(size_t node_index) const;
    // Drops only read-only mapped pages in the finalized ranges. File bytes are
    // untouched; pages are faulted back from the original backing file later.
    bool drop_after(size_t node_index) const;

private:
    struct use {
        size_t mapping_index;
        size_t begin;
        size_t end;
        size_t node_index;
    };

    std::vector<mapping> mappings_;
    std::vector<use> uses_;
    std::vector<std::vector<range>> after_;
    size_t page_size_;
    bool finalized_ = false;
};

} // namespace llama_mmap_evict
