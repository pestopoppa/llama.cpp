#pragma once

// prompt fill order for update_slots() (--prefill-order srpf, --prefill-contend-cap)
//
// with the default slot order, the pending-prompt loop walks the slots by index and stops once the batch is
// full, so a short prompt in a higher-index slot waits behind every iteration of a long prefill in a
// lower-index slot (head-of-line blocking). shortest-remaining-prompt-first fills the batch from the slot
// with the fewest prompt tokens left; the contend cap bounds the prompt tokens per iteration while two or
// more prompts are pending, so the iteration that carries the short prompt is not a full n_batch chunk of a
// long one. pure functions, no llama state, so they are unit-tested without a model
// (tests/test-server-prefill-order.cpp).

#include <algorithm>
#include <cstdint>
#include <vector>

struct server_prefill_order_entry {
    int32_t idx         = 0;     // slot index
    bool    pending     = false; // the slot has prompt tokens to add (SLOT_STATE_STARTED or SLOT_STATE_PROCESSING_PROMPT)
    bool    can_split   = true;  // the prompt may be split across iterations
    int32_t n_remaining = 0;     // prompt tokens still to process (for SLOT_STATE_STARTED: the full prompt, an upper bound)
};

// visit order: the pending slots by remaining prompt tokens ascending (ties keep slot-index order), then every
// other slot in slot-index order
static inline std::vector<int32_t> server_prefill_order_srpf(const std::vector<server_prefill_order_entry> & entries) {
    std::vector<server_prefill_order_entry> pending;
    std::vector<int32_t> order;
    order.reserve(entries.size());

    for (const auto & e : entries) {
        if (e.pending) {
            pending.push_back(e);
        }
    }

    std::stable_sort(pending.begin(), pending.end(), [](const auto & a, const auto & b) {
        return a.n_remaining < b.n_remaining;
    });

    for (const auto & e : pending) {
        order.push_back(e.idx);
    }
    for (const auto & e : entries) {
        if (!e.pending) {
            order.push_back(e.idx);
        }
    }

    return order;
}

// the prompt-token budget for this iteration after the prefill-vs-prefill cap
//   budget: the decode-aware budget already chosen (0 = none, prompts fill n_batch)
//   cap   : --prefill-contend-cap (0 = off)
// the cap applies only while >= 2 slots have pending prompt tokens and every one of them can split (a prompt
// that cannot be split needs the whole n_batch); the result follows the decode-aware budget's convention,
// 0 when no limit below n_batch applies
static inline int32_t server_prefill_contend_budget(
        const std::vector<server_prefill_order_entry> & entries, int32_t budget, int32_t n_batch, int32_t cap) {
    if (cap <= 0) {
        return budget;
    }

    int32_t n_pending = 0;
    for (const auto & e : entries) {
        if (!e.pending) {
            continue;
        }
        if (!e.can_split) {
            return budget;
        }
        n_pending++;
    }

    if (n_pending < 2) {
        return budget;
    }

    const int32_t capped = budget > 0 ? std::min(budget, cap) : cap;

    return capped < n_batch ? capped : budget;
}
