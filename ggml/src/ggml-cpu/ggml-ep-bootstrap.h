// ggml-ep-bootstrap — env-var-driven inter-process Expert Parallelism setup.
//
// Read by ggml_cpu_init() at first call. If GGML_EP_ROLE=master is set,
// forks N-1 worker processes via ep_dispatcher (master is instance 0).
// Worker children never return to their caller's main() — they enter a
// passive wait loop and _exit() when master signals EXIT (or dies, via
// PR_SET_PDEATHSIG=SIGTERM set by ep_dispatcher).
//
// The session pointer is stored in a process-global so the graph executor
// can later find it without plumbing through llama.cpp's parameter structs.
//
// Env vars (read once at init):
//   GGML_EP_ROLE=master|none      (default: none — disables EP entirely)
//   GGML_EP_N_INSTANCES=N         (1..EP_MAX_WORKERS+1; 1 = no EP; default: 4)
//
// Subsequent steps (b+) will add CLI args and the graph-executor hook so
// workers actually receive and process expert-shard work.

#ifndef GGML_EP_BOOTSTRAP_H
#define GGML_EP_BOOTSTRAP_H

#ifdef __cplusplus
extern "C" {
#endif

struct ep_session;

// Idempotent. Reads env vars and forks workers if requested.
// Returns 1 if EP is enabled (master only — workers never return to caller),
// 0 otherwise. Safe to call from C; safe to call multiple times.
int ggml_ep_bootstrap_if_requested(void);

// Returns the global session pointer, or NULL if EP not enabled.
struct ep_session * ggml_ep_get_session(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_EP_BOOTSTRAP_H
