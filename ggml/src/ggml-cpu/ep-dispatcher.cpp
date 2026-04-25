// ep_dispatcher implementation. See ep_dispatcher.h for the API contract.

#include "ggml-ep-dispatcher.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <linux/mempolicy.h>

// Forward declaration so ep_wait_workers can call it (defined below).
extern "C" int ep_master_reap_dead(struct ep_session * sess);

// State machine values (32-bit so they fit in atomic<uint32_t>).
//   IDLE: worker waiting; master may post GO.
//   GO  : master has posted; worker should run.
//   DONE: worker has finished; master should gather + reset to IDLE.
//   EXIT: terminate (from either side; workers exit on receipt).
//   DEAD: a worker has died (set by master if waitpid detects it).
static constexpr uint32_t STATE_IDLE = 0;
static constexpr uint32_t STATE_GO   = 1;
static constexpr uint32_t STATE_DONE = 2;
static constexpr uint32_t STATE_EXIT = 0xFFFFFFFFu;
static constexpr uint32_t STATE_DEAD = 0xFFFFFFFEu;

// Cacheline padding. Pessimistic for cross-platform; on x86_64 this is 64.
static constexpr size_t EP_CACHE_LINE = 128;

// One cacheline per worker state to eliminate false sharing. With 4 workers,
// the master writes state[0]=GO, state[1]=GO, ... state[3]=GO in succession.
// If those atomics shared a cacheline, each store would invalidate all
// readers' cached copies (workers spinning on their state[w]). Per-line
// isolation lets each worker spin on its own cacheline; only its master's
// store invalidates that line.
struct ep_state_slot {
    alignas(64) std::atomic<uint32_t> state;
    char _pad[64 - sizeof(std::atomic<uint32_t>)];
};
static_assert(sizeof(ep_state_slot) == 64, "ep_state_slot must be one cacheline");

struct ep_session_shm {
    ep_state_slot slots[EP_MAX_WORKERS];
    // Note: broadcast and gather regions are SEPARATE mmaps, not embedded
    // here, so we can size them independently per session and place them
    // on different NUMA nodes.
};

// IMPORTANT: ep_session is allocated via malloc/calloc, NOT MAP_SHARED.
// After fork, each process gets a copy-on-write copy of this struct, so
// per-process fields like `role` and `instance_id` can differ between
// master and workers without the writes propagating across processes.
//
// The pointers inside (ctl, broadcast, gather, worker_pids array) point
// into MAP_SHARED regions and are correctly shared. The MAP_SHARED regions
// hold the actual cross-process state (atomic worker_state, broadcast bytes,
// gather bytes); the ep_session struct is just a per-process handle.
struct ep_session {
    // Mode of THIS process (master in parent, worker in child after fork).
    enum ep_role role;
    int          instance_id;     // 0 for master, 1..n for workers
    int          n_workers;

    // Configuration snapshot
    size_t       broadcast_bytes;
    size_t       gather_bytes_per_worker;

    // Shared memory regions (visible to all processes via MAP_SHARED)
    ep_session_shm * ctl;
    void *           broadcast;                       // master writes; workers read
    void *           gather[EP_MAX_WORKERS];          // worker w writes; master reads

    // Worker child PIDs (master-only; valid in master after create)
    pid_t        worker_pids[EP_MAX_WORKERS];

    bool         destroyed;
};

// ---- helpers ----

static long sys_mbind(void * addr, size_t len, int mode,
                      const unsigned long * nodemask, unsigned long maxnode, unsigned flags) {
    return syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
}

static void pin_to_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

static void mbind_to_node(void * addr, size_t bytes, int node) {
    if (!addr || bytes == 0 || node < 0) return;
    unsigned long mask = 1UL << node;
    sys_mbind(addr, bytes, MPOL_BIND, &mask, 64UL, 0);
}

static void * mmap_shared_anon(size_t bytes) {
    if (bytes == 0) return nullptr;
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? nullptr : p;
}

// ---- create / attach / destroy ----

extern "C" int ep_session_create_master(const struct ep_config * cfg,
                                          struct ep_session ** out) {
    if (!cfg || !out) return EINVAL;
    if (cfg->n_workers <= 0 || cfg->n_workers > EP_MAX_WORKERS) return EINVAL;

    ep_session * s = (ep_session *) calloc(1, sizeof(ep_session));
    if (!s) return ENOMEM;
    s->role        = EP_ROLE_MASTER;
    s->instance_id = 0;
    s->n_workers   = cfg->n_workers;
    s->broadcast_bytes        = cfg->broadcast_bytes;
    s->gather_bytes_per_worker = cfg->gather_bytes_per_worker;

    // Control region (per-worker state). Use a separate shared mmap so it
    // can be mbind'd to master's node (workers cross-NUMA-read this region
    // on every iteration; placing it locally to master is fine since the
    // hot path is master signaling and master gets local-cache benefit).
    s->ctl = (ep_session_shm *) mmap_shared_anon(sizeof(ep_session_shm));
    if (!s->ctl) { free(s); return ENOMEM; }
    for (int w = 0; w < cfg->n_workers; ++w) {
        s->ctl->slots[w].state.store(STATE_IDLE, std::memory_order_relaxed);
    }

    // Broadcast region (master writes, workers read)
    s->broadcast = mmap_shared_anon(cfg->broadcast_bytes);
    if (!s->broadcast && cfg->broadcast_bytes > 0) {
        munmap(s->ctl, sizeof(*s->ctl));
        free(s);
        return ENOMEM;
    }
    if (cfg->master_numa_node >= 0 && s->broadcast) {
        mbind_to_node(s->broadcast, cfg->broadcast_bytes, cfg->master_numa_node);
    }

    // Gather regions (one per worker; mbind to worker's node)
    for (int w = 0; w < cfg->n_workers; ++w) {
        s->gather[w] = mmap_shared_anon(cfg->gather_bytes_per_worker);
        if (!s->gather[w] && cfg->gather_bytes_per_worker > 0) {
            for (int u = 0; u < w; ++u) munmap(s->gather[u], cfg->gather_bytes_per_worker);
            if (s->broadcast) munmap(s->broadcast, cfg->broadcast_bytes);
            munmap(s->ctl, sizeof(*s->ctl));
            free(s);
            return ENOMEM;
        }
        if (cfg->worker_numa_nodes && cfg->worker_numa_nodes[w] >= 0 && s->gather[w]) {
            mbind_to_node(s->gather[w], cfg->gather_bytes_per_worker, cfg->worker_numa_nodes[w]);
        }
    }

    // Snapshot pinning configuration for use by worker children. Workers
    // inherit ‘s‘ via fork (it's MAP_SHARED so points to the same region);
    // they read s->n_workers etc. directly. We need the per-worker CPU and
    // node info for the child to pin itself; pass via temporary copies on
    // the stack of each child (set before fork).

    // Fork workers
    for (int w = 0; w < cfg->n_workers; ++w) {
        const int worker_cpu = (cfg->worker_cpus       ? cfg->worker_cpus[w]       : -1);
        // worker_numa_node was already used at gather-region mbind above;
        // we don't need it again per child since the gather region is
        // shared (each worker's own mbind'd slot is already on the right
        // node before fork).
        pid_t pid = fork();
        if (pid < 0) {
            // Fork failed; tear down already-spawned workers
            int err = errno;
            for (int u = 0; u < w; ++u) {
                kill(s->worker_pids[u], SIGTERM);
                waitpid(s->worker_pids[u], nullptr, 0);
            }
            for (int u = 0; u < cfg->n_workers; ++u) {
                if (s->gather[u]) munmap(s->gather[u], cfg->gather_bytes_per_worker);
            }
            if (s->broadcast) munmap(s->broadcast, cfg->broadcast_bytes);
            munmap(s->ctl, sizeof(*s->ctl));
            free(s);
            *out = nullptr;
            return err;
        }
        if (pid == 0) {
            // CHILD: become worker w
            s->role = EP_ROLE_WORKER;
            s->instance_id = w + 1;     // workers are 1-indexed; master is 0
            // Die when master dies
            prctl(PR_SET_PDEATHSIG, SIGTERM);
            // Pin
            pin_to_cpu(worker_cpu);
            // Worker just returns out of this call; the caller's main loop
            // takes over (calls ep_worker_recv / send_done in a loop). Note
            // child inherits *out=nullptr from before the master populated
            // it; we don't write *out here.
            *out = s;
            return 0;
        }
        // MASTER: record child pid
        s->worker_pids[w] = pid;
    }

    // Master: pin self, finalize
    pin_to_cpu(cfg->master_cpu);

    *out = s;
    return 0;
}

extern "C" enum ep_role ep_session_role(const struct ep_session * sess) {
    return sess ? sess->role : EP_ROLE_MASTER;
}

extern "C" int ep_session_instance_id(const struct ep_session * sess) {
    return sess ? sess->instance_id : 0;
}

extern "C" int ep_session_n_workers(const struct ep_session * sess) {
    return sess ? sess->n_workers : 0;
}

extern "C" void ep_session_destroy(struct ep_session * sess) {
    if (!sess || sess->destroyed) return;
    if (sess->role == EP_ROLE_MASTER) {
        // Send EXIT to all workers; wait for them to terminate
        for (int w = 0; w < sess->n_workers; ++w) {
            sess->ctl->slots[w].state.store(STATE_EXIT, std::memory_order_release);
        }
        for (int w = 0; w < sess->n_workers; ++w) {
            if (sess->worker_pids[w] > 0) {
                waitpid(sess->worker_pids[w], nullptr, 0);
            }
        }
    }
    // Free shared memory regions (last process to munmap actually unmaps)
    for (int w = 0; w < sess->n_workers; ++w) {
        if (sess->gather[w]) munmap(sess->gather[w], sess->gather_bytes_per_worker);
    }
    if (sess->broadcast) munmap(sess->broadcast, sess->broadcast_bytes);
    if (sess->ctl)       munmap(sess->ctl, sizeof(*sess->ctl));
    sess->destroyed = true;
    free(sess);
}

// ---- master ops ----

extern "C" int ep_broadcast(struct ep_session * sess, const void * src, size_t bytes) {
    if (!sess || sess->role != EP_ROLE_MASTER) return EINVAL;
    if (bytes > sess->broadcast_bytes)         return EINVAL;
    if (bytes > 0 && src && sess->broadcast)    memcpy(sess->broadcast, src, bytes);
    for (int w = 0; w < sess->n_workers; ++w) {
        sess->ctl->slots[w].state.store(STATE_GO, std::memory_order_release);
    }
    return 0;
}

extern "C" int ep_wait_workers(struct ep_session * sess) {
    if (!sess || sess->role != EP_ROLE_MASTER) return EINVAL;
    // Spin counter triggers a periodic non-blocking waitpid() to detect
    // dead workers without a separate watchdog thread. The check fires every
    // N spins (~few μs) so detection latency is bounded but the hot path
    // overhead is negligible (the check itself is one syscall every ~10 μs).
    constexpr int kReapEvery = 4096;
    int spin_since_check = 0;
    for (int w = 0; w < sess->n_workers; ++w) {
        for (;;) {
            uint32_t s = sess->ctl->slots[w].state.load(std::memory_order_acquire);
            if (s == STATE_DONE) break;
            if (s == STATE_DEAD) return ESRCH;
            __builtin_ia32_pause();
            if (++spin_since_check >= kReapEvery) {
                spin_since_check = 0;
                // Periodic dead-worker reap. Returns count; if any died, the
                // affected slot's state is now STATE_DEAD and the next load
                // above will pick it up and return ESRCH.
                if (ep_master_reap_dead(sess) > 0) {
                    // Re-check this worker's state immediately
                    if (sess->ctl->slots[w].state.load(std::memory_order_acquire) == STATE_DEAD) {
                        return ESRCH;
                    }
                }
            }
        }
        sess->ctl->slots[w].state.store(STATE_IDLE, std::memory_order_release);
    }
    return 0;
}

extern "C" int ep_gather(struct ep_session * sess, const void * out_ptrs[]) {
    if (!sess || sess->role != EP_ROLE_MASTER || !out_ptrs) return EINVAL;
    for (int w = 0; w < sess->n_workers; ++w) {
        out_ptrs[w] = sess->gather[w];
    }
    return 0;
}

extern "C" void * ep_master_broadcast_buffer(struct ep_session * sess) {
    if (!sess || sess->role != EP_ROLE_MASTER) return nullptr;
    return sess->broadcast;
}

// ---- worker ops ----

extern "C" int ep_worker_wait_go(struct ep_session * sess) {
    if (!sess || sess->role != EP_ROLE_WORKER) return EINVAL;
    const int idx = sess->instance_id - 1;
    for (;;) {
        uint32_t s = sess->ctl->slots[idx].state.load(std::memory_order_acquire);
        if (s == STATE_GO)   return 0;
        if (s == STATE_EXIT) return 1;
        __builtin_ia32_pause();
    }
}

extern "C" int ep_worker_signal_done(struct ep_session * sess) {
    if (!sess || sess->role != EP_ROLE_WORKER) return EINVAL;
    const int idx = sess->instance_id - 1;
    sess->ctl->slots[idx].state.store(STATE_DONE, std::memory_order_release);
    return 0;
}

extern "C" int ep_worker_recv(struct ep_session * sess, void * dst, size_t bytes) {
    int rc = ep_worker_wait_go(sess);
    if (rc) return rc;
    if (bytes > sess->broadcast_bytes) return EINVAL;
    if (dst && sess->broadcast && bytes > 0) {
        memcpy(dst, sess->broadcast, bytes);
    }
    return 0;
}

extern "C" int ep_worker_send_done(struct ep_session * sess, const void * src, size_t bytes) {
    if (!sess || sess->role != EP_ROLE_WORKER) return EINVAL;
    if (bytes > sess->gather_bytes_per_worker) return EINVAL;
    const int idx = sess->instance_id - 1;
    if (src && sess->gather[idx] && bytes > 0) {
        memcpy(sess->gather[idx], src, bytes);
    }
    sess->ctl->slots[idx].state.store(STATE_DONE, std::memory_order_release);
    return 0;
}

extern "C" void * ep_worker_gather_buffer(struct ep_session * sess) {
    if (!sess || sess->role != EP_ROLE_WORKER) return nullptr;
    return sess->gather[sess->instance_id - 1];
}

extern "C" const void * ep_worker_broadcast_buffer(struct ep_session * sess) {
    if (!sess || sess->role != EP_ROLE_WORKER) return nullptr;
    return sess->broadcast;
}

extern "C" int ep_master_reap_dead(struct ep_session * sess) {
    if (!sess || sess->role != EP_ROLE_MASTER) return 0;
    int n_dead = 0;
    for (int w = 0; w < sess->n_workers; ++w) {
        if (sess->worker_pids[w] <= 0) continue;
        int status;
        pid_t r = waitpid(sess->worker_pids[w], &status, WNOHANG);
        if (r > 0) {
            // Worker exited. Mark its state DEAD so spinning waiters break out.
            sess->ctl->slots[w].state.store(STATE_DEAD, std::memory_order_release);
            sess->worker_pids[w] = -1;
            n_dead++;
        }
    }
    return n_dead;
}
