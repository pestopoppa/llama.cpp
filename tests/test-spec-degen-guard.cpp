// AK-H-MTP-DEGEN-1: unit test of common_spec_degen (no model). Run with the knob forced on via the struct config.
#include "../common/speculative-degen.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

int main() {
    // default OFF: env unset -> never restricts
    {
        common_spec_degen g;
        CHECK(!g.c.enabled);
        for (int i = 0; i < 1000; i++) { g.on_round(3, 0, 3); }
        CHECK(g.cap(3) == 3);
        CHECK(g.st == common_spec_degen::ACTIVE);
    }
    common_spec_degen g;
    g.c.enabled = true;
    // healthy slot (acceptance 0.8) never trips
    for (int i = 0; i < 2000; i++) { CHECK(g.on_round(3, 2, 3) == 0); }
    CHECK(g.st == common_spec_degen::ACTIVE && g.cap(3) == 3);
    g.reset(); g.c.enabled = true;
    // degenerate: 0 accepted, 3 drafted per round -> 1 token per round; window fills at 512 rounds
    int tripped = 0;
    for (int i = 0; i < 511; i++) { CHECK(g.on_round(3, 0, 3) == 0); }
    tripped = g.on_round(3, 0, 3);
    CHECK(tripped == 1 && g.st == common_spec_degen::SUSPENDED);
    CHECK(g.cap(3) == 0);                      // target-only
    g.advance(10);                             // first call baselines last_n_decoded (delta 10)
    CHECK(g.countdown == g.c.probe - 10 && g.cap(3) == 0);
    g.advance(10 + g.c.probe);
    CHECK(g.countdown == 0 && g.cap(3) == 2);  // probe round, depth cap 2
    // shallow rejection -> suspended again, cap 2, countdown re-armed
    CHECK(g.on_round(2, 1, 3) == 0 && g.st == common_spec_degen::SUSPENDED && g.countdown == g.c.probe && g.probe_cap == 2);
    g.advance(1000);
    CHECK(g.cap(3) == 2);
    // fully accepted probe -> cap 3 == n_max -> back to ACTIVE
    CHECK(g.on_round(2, 2, 3) == -1 && g.st == common_spec_degen::ACTIVE && g.cap(3) == 3 && g.n_resumes == 1);
    // n_max = 5: +1 per fully accepted round, shallow rejection resets to 2
    common_spec_degen h;
    h.c.enabled = true;
    h.st = common_spec_degen::SUSPENDED; h.countdown = 0; h.probe_cap = 2;
    CHECK(h.on_round(2, 2, 5) == 0 && h.st == common_spec_degen::RECOVERING && h.cap(5) == 3);
    CHECK(h.on_round(3, 3, 5) == 0 && h.cap(5) == 4);
    CHECK(h.on_round(4, 1, 5) == 0 && h.st == common_spec_degen::SUSPENDED && h.probe_cap == 2);
    // env parse
    setenv("LLAMA_SPEC_DEGEN_SUSPEND", "1", 1);
    CHECK(common_spec_degen::cfg_from_env().enabled);
    setenv("LLAMA_SPEC_DEGEN_SUSPEND", "0", 1);
    CHECK(!common_spec_degen::cfg_from_env().enabled);
    printf("test-spec-degen-guard OK\n");
    return 0;
}
