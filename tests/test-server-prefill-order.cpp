// --prefill-order srpf / --prefill-contend-cap: the pure ordering and cap functions of update_slots()
// (tools/server/server-prefill-order.h). No model needed.

#include "server-prefill-order.h"

#include <cstdio>
#include <vector>

static int n_fail  = 0;
static int n_check = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        n_check++;                                                               \
        if (!(cond)) {                                                           \
            fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            n_fail++;                                                            \
        }                                                                        \
    } while (0)

static server_prefill_order_entry pend(int32_t idx, int32_t n_remaining, bool can_split = true) {
    server_prefill_order_entry e;
    e.idx         = idx;
    e.pending     = true;
    e.can_split   = can_split;
    e.n_remaining = n_remaining;
    return e;
}

static server_prefill_order_entry other(int32_t idx) {
    server_prefill_order_entry e;
    e.idx = idx;
    return e;
}

using order_t = std::vector<int32_t>;

static void test_order() {
    // PB-HOL-1 B arm: long prefill in slot 0, short prompt in slot 1, slot 2 decoding -> short first
    CHECK(server_prefill_order_srpf({pend(0, 32768), pend(1, 128), other(2)}) == (order_t{1, 0, 2}));

    // A arm: the short prompt is already in the lower slot -> unchanged
    CHECK(server_prefill_order_srpf({pend(0, 128), pend(1, 32768), other(2)}) == (order_t{0, 1, 2}));

    // ties keep slot-index order (stable)
    CHECK(server_prefill_order_srpf({pend(0, 500), pend(1, 500), pend(2, 100), pend(3, 500)}) == (order_t{2, 0, 1, 3}));

    // non-pending slots (idle, generating, waiting on a parent) follow in slot-index order
    CHECK(server_prefill_order_srpf({other(0), pend(1, 9000), other(2), pend(3, 10)}) == (order_t{3, 1, 0, 2}));

    // nothing pending -> the plain slot-index walk
    CHECK(server_prefill_order_srpf({other(0), other(1), other(2)}) == (order_t{0, 1, 2}));

    // every slot appears exactly once
    CHECK(server_prefill_order_srpf({pend(0, 3), other(1), pend(2, 1), other(3), pend(4, 2)}) == (order_t{2, 4, 0, 1, 3}));

    // empty
    CHECK(server_prefill_order_srpf({}).empty());
}

static void test_contend_budget() {
    const int32_t n_batch = 2048;

    // one pending prompt: no cap (solo prefill keeps the full n_batch, or the decode-aware budget)
    CHECK(server_prefill_contend_budget({pend(0, 65536), other(1)}, 0,   n_batch, 512) == 0);
    CHECK(server_prefill_contend_budget({pend(0, 65536), other(1)}, 256, n_batch, 512) == 256);

    // two pending, no decode-aware budget -> the cap
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128)}, 0, n_batch, 512) == 512);

    // the lower of the decode-aware budget and the cap
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128)}, 256, n_batch, 512) == 256);
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128)}, 1024, n_batch, 512) == 512);

    // cap off
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128)}, 0,   n_batch, 0) == 0);
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128)}, 300, n_batch, 0) == 300);

    // cap >= n_batch is no limit (0 by the decode-aware budget's convention)
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128)}, 0, 512,  512) == 0);
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128)}, 0, 512, 4096) == 0);

    // a pending prompt that cannot be split (embedding) needs the whole batch -> no cap
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128, false)}, 0,   n_batch, 512) == 0);
    CHECK(server_prefill_contend_budget({pend(0, 32768), pend(1, 128, false)}, 256, n_batch, 512) == 256);

    // non-pending slots do not count towards contention
    CHECK(server_prefill_contend_budget({pend(0, 32768), other(1), other(2)}, 0, n_batch, 512) == 0);
}

int main() {
    test_order();
    test_contend_budget();

    printf("test-server-prefill-order: %d checks, %d failures\n", n_check, n_fail);

    return n_fail == 0 ? 0 : 1;
}
