// ep_dispatcher — inter-process Expert Parallelism IPC primitive.
//
// Master + N worker process pool with NUMA-pinned busy-spin synchronization
// over MAP_SHARED|MAP_ANONYMOUS shared memory. Each iteration consists of:
//
//   master:  ep_broadcast(src, bytes) → ep_wait_workers() → ep_gather(out_ptrs)
//   worker:  ep_worker_recv(dst, bytes) → ...local compute... →
//            ep_worker_send_done(src, bytes)
//
// The synchronization primitive is sub-microsecond on EPYC NPS4 (measured
// 0.86 μs RTT for 4 workers in the original prototype). Workers busy-spin
// on a per-worker cacheline-separated atomic state, so they hold a CPU each
// — this is the production deployment pattern (workers are llama.cpp
// instances doing real compute between sync points).
//
// API is C-callable so this library can be linked into llama.cpp without
// pulling in C++ runtime considerations beyond the implementation .cpp.
//
// Build: see Makefile in this directory. Outputs libep_dispatcher.a.

#ifndef EP_DISPATCHER_H
#define EP_DISPATCHER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum supported worker count. Sized to fit comfortably for our 4-NUMA
// EPYC target plus headroom for experimentation up to 16-CCD configurations.
#define EP_MAX_WORKERS 16

// Opaque handle. The full struct lives in the implementation; callers only
// pass the pointer through the API.
struct ep_session;

// Configuration for ep_session_create_master(). All fields must be
// populated by the caller.
struct ep_config {
    int    n_workers;                       // 1..EP_MAX_WORKERS

    // Sizes of the shared regions, in bytes. Sized to comfortably hold the
    // largest expected payload (one round-trip's hidden state for broadcast,
    // one worker's partial output for gather). Caller responsible for
    // matching producer and consumer expectations.
    size_t broadcast_bytes;                 // master → workers (single buffer)
    size_t gather_bytes_per_worker;         // worker → master (one per worker)

    // Optional NUMA / CPU pinning. If `worker_cpus[w] >= 0`, worker w is
    // pinned to that CPU via sched_setaffinity. If `worker_numa_nodes[w]
    // >= 0`, the worker's gather region is mbind'd MPOL_BIND to that node
    // (so first-touch by the worker faults onto the local node). Pass -1
    // to opt out per-slot.
    //
    // Pointers may be NULL to opt out entirely (no pinning, no mbind).
    const int * worker_cpus;
    const int * worker_numa_nodes;
    int         master_cpu;                 // -1 to opt out
    int         master_numa_node;           // -1 to opt out (mbinds broadcast region)
};

// Master entry point. Forks n_workers child processes; each child enters
// the worker code path automatically and is identified by ep_session_role()
// returning EP_ROLE_WORKER plus ep_session_instance_id() returning the
// child's worker index (0..n_workers-1).
//
// On success returns 0 and *out is the new session. Master continues
// executing after the call. On any failure (mmap, fork, mbind), returns
// non-zero errno-style value and *out is NULL.
//
// IMPORTANT: the master process is responsible for orchestrating the
// session and calling ep_session_destroy() at shutdown. Worker children
// detect master death (via PR_SET_PDEATHSIG=SIGTERM) and exit cleanly.
int ep_session_create_master(const struct ep_config * cfg, struct ep_session ** out);

// Returned role: master is always 0; worker indices are 1..n_workers.
enum ep_role {
    EP_ROLE_MASTER = 0,
    EP_ROLE_WORKER = 1,
};

// Query the current process's role and instance id. Valid after
// ep_session_create_master() returns (in the calling process — master) or
// after fork (in child processes — workers; their ep_session pointer is
// inherited via copy-on-write of the parent's address space).
enum ep_role ep_session_role(const struct ep_session * sess);
int          ep_session_instance_id(const struct ep_session * sess);
int          ep_session_n_workers(const struct ep_session * sess);

// Tear down. Master sends EXIT to all workers; workers exit; master
// waitpid()'s them and frees shm. Idempotent.
void ep_session_destroy(struct ep_session * sess);

// ----- Master-side ops -----

// Copy `src` (size `bytes`) into the broadcast region and signal all
// workers to enter their compute round. `bytes` must be ≤ cfg.broadcast_bytes.
// Non-blocking; returns 0 on success.
int ep_broadcast(struct ep_session * sess, const void * src, size_t bytes);

// Block (busy-spin) until all workers have signaled DONE. Resets state
// machine to IDLE for the next round. Returns 0 on success; non-zero if
// a worker died while we were waiting.
int ep_wait_workers(struct ep_session * sess);

// Populate `out_ptrs[w]` with a read-only pointer to worker w's gather
// region. Pointers are stable for the lifetime of the session; valid only
// AFTER ep_wait_workers() returns. Caller can read directly from these
// pointers (no copy needed).
int ep_gather(struct ep_session * sess, const void * out_ptrs[]);

// Returns a writable pointer to the broadcast region. Master can write
// directly without going through ep_broadcast()'s memcpy; then call
// `ep_broadcast(sess, NULL, 0)` to just signal workers GO. Useful in
// hot paths where the master computes the broadcast value in place
// (e.g. ggml graph already wrote the hidden state to a buffer that we
// want to make the broadcast region).
void * ep_master_broadcast_buffer(struct ep_session * sess);

// ----- Worker-side ops -----

// Block (busy-spin) until master signals GO, then copy the broadcast
// region into `dst` (size `bytes` must be ≤ cfg.broadcast_bytes). Returns
// 0 on success, non-zero if EXIT signal received (worker should return
// from its loop).
int ep_worker_recv(struct ep_session * sess, void * dst, size_t bytes);

// Copy `src` (size `bytes` ≤ cfg.gather_bytes_per_worker) into this
// worker's gather region and signal DONE. Returns 0 on success.
int ep_worker_send_done(struct ep_session * sess, const void * src, size_t bytes);

// Convenience: as a worker, returns a writable pointer to this worker's
// gather region. Lets the worker write its output directly without an
// intermediate buffer + ep_worker_send_done copy. After writing, call
// ep_worker_signal_done() to publish.
void * ep_worker_gather_buffer(struct ep_session * sess);

// As above: returns a read-only pointer to the broadcast region for this
// worker. Lets the worker read directly without copying through dst.
// Valid only between ep_worker_wait_go() and ep_worker_signal_done().
const void * ep_worker_broadcast_buffer(struct ep_session * sess);

// Lower-level worker primitives (used by ep_worker_recv internally) for
// callers that want direct access without the implicit copy.
int ep_worker_wait_go(struct ep_session * sess);   // returns 0=GO, 1=EXIT
int ep_worker_signal_done(struct ep_session * sess);

// ----- Health-check (master-only) -----

// Reaps any worker child processes that have exited without signaling DONE
// and marks their state as DEAD so subsequent ep_wait_workers() returns
// ESRCH instead of spinning forever. Master should call this either
// periodically from a watchdog thread (preferred for long-running
// production servers) or before each ep_wait_workers() (acceptable for
// short benchmarks; adds ~2 μs of waitpid overhead per round).
//
// Returns the number of dead workers detected this call (0 if all alive).
int ep_master_reap_dead(struct ep_session * sess);

#ifdef __cplusplus
}
#endif

#endif // EP_DISPATCHER_H
