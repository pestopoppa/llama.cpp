#pragma once
// INF-70 HARNESS-1: runtime-switchable compute-path knobs.
//
// The champion levers (GGML_ROWCOL_SPLIT, GGML_TINY_SOLO, GGML_EMPTY_SKIP, GGML_VEC_Q8K,
// GGML_QSPLIT and their *_MIN/*_MAX companions) all select a code path DURING graph
// execution -- none of them changes how a buffer is allocated. They were previously latched
// on first use (function-local `static const` / a -1 sentinel), which forced one process per
// arm and therefore paired every A/B across two separate 92 GB model loads.
//
// Here they live in ONE snapshot struct that is refreshed exactly once per ggml_graph_compute,
// by the calling thread, BEFORE any worker thread is dispatched. Consequences that matter:
//
//   * every one of the `nth` threads sees the same value for the whole graph, so barrier
//     pairing and the tiny-solo run detection stay consistent (SYNC-12: a matcher must be a
//     pure function of the graph);
//   * a worker-side read is a plain load of a global -- cheaper than the C++ guard-variable
//     check it replaces, and NOT a getenv;
//   * with no control file present the values are exactly the getenv-derived defaults, so the
//     shipped behaviour is bit-identical to CHAMPION-3.
//
// Switching is done through a control page, not the environment (a live process's environ
// cannot be rewritten from outside). Set GGML_KNOB_FILE=<path> to a 4096-byte file:
//
//   [0..8)      uint64 seq, little endian. Slot in use = seq & 1. Bumped LAST by the writer.
//   [512..1536) slot 0: NUL-terminated "KEY=VALUE\n" ASCII
//   [1536..2560) slot 1: same
//
// Refresh cost when nothing changed: one relaxed atomic load of the seq word (the page is
// mapped MAP_SHARED once at init). Unset GGML_KNOB_FILE => no map, no load, no cost.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cpu_knobs {
    int32_t rowcol_split;       // GGML_ROWCOL_SPLIT      default 1
    int32_t tiny_solo;          // GGML_TINY_SOLO         default 1
    int32_t empty_skip;         // GGML_EMPTY_SKIP        default 1
    int32_t vec_sigmoid;        // GGML_VEC_SIGMOID       default 0
    int32_t vec_q8k;            // GGML_VEC_Q8K           default 1
    int32_t qsplit;             // GGML_QSPLIT            default 1
    int64_t rowcol_min_elems;   // GGML_ROWCOL_MIN_ELEMS  default 512
    int64_t tiny_solo_max;      // GGML_TINY_SOLO_MAX     default 4096
    int64_t qsplit_min;         // GGML_QSPLIT_MIN        default INT64_MAX
    int64_t tiny_solo_rows;     // GGML_TINY_SOLO_ROWS    default 1
    int64_t tiny_solo_rows_max; // GGML_TINY_SOLO_ROWS_MAX default 4096
};

// Definition lives in ggml-cpu.c. Read-only for worker threads.
extern struct ggml_cpu_knobs ggml_cpu_knobs_cur;

// Seed from the environment and map GGML_KNOB_FILE if set. Called once from ggml_cpu_init().
void ggml_cpu_knobs_init(void);

// Re-read the control page if its seq changed. Called from ggml_graph_compute() on the
// calling thread only, before any worker is dispatched. Returns 1 if the snapshot changed.
int  ggml_cpu_knobs_refresh(void);

#ifdef __cplusplus
}
#endif
