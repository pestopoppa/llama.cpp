// See ggml-ep-bootstrap.h for the API contract.

#include "ggml-ep-bootstrap.h"
#include "ggml-ep-dispatcher.h"

#define _GNU_SOURCE 1

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

static struct ep_session * g_ep_session = nullptr;
static std::atomic<bool>   g_ep_bootstrapped{false};

// Add all CPUs in `node_id` (parsed from /sys/devices/system/node/nodeN/cpulist)
// to `cpuset`. Returns the number of CPUs added, or -1 on parse failure.
static int ggml_ep_add_node_cpus(int node_id, cpu_set_t * cpuset) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node_id);
    FILE * f = fopen(path, "r");
    if (!f) return -1;
    char buf[1024] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) return -1;

    int n_cpus = 0;
    char * tok = strtok(buf, ",\n");
    while (tok) {
        int s = -1, e = -1;
        if (sscanf(tok, "%d-%d", &s, &e) == 2) {
            for (int c = s; c <= e; ++c) { CPU_SET(c, cpuset); ++n_cpus; }
        } else if (sscanf(tok, "%d", &s) == 1) {
            CPU_SET(s, cpuset); ++n_cpus;
        }
        tok = strtok(nullptr, ",\n");
    }
    return n_cpus;
}

// Pin the calling process to all CPUs in the listed `nodes[0..n)` and set
// the default memory allocation policy. With a single node, MPOL_PREFERRED
// gives strict locality; with multiple nodes, MPOL_INTERLEAVE round-robins
// pages across them so memory bandwidth is averaged. Returns 0 on success.
static int ggml_ep_pin_to_numa_nodes(const int * nodes, int n) {
    if (n <= 0) return -1;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int total_cpus = 0;
    for (int i = 0; i < n; ++i) {
        int added = ggml_ep_add_node_cpus(nodes[i], &cpuset);
        if (added < 0) {
            fprintf(stderr, "ggml-ep: cpulist for node %d unparseable; skipping pin\n", nodes[i]);
            return -1;
        }
        total_cpus += added;
    }
    if (total_cpus == 0) return -1;
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        fprintf(stderr, "ggml-ep: sched_setaffinity failed (%s)\n", strerror(errno));
        return -1;
    }

    unsigned long nodemask = 0;
    for (int i = 0; i < n; ++i) nodemask |= (1UL << nodes[i]);
    int policy = (n == 1) ? MPOL_PREFERRED : MPOL_INTERLEAVE;
    long mp_rc = syscall(SYS_set_mempolicy, policy, &nodemask, sizeof(nodemask) * 8);
    if (mp_rc != 0) {
        fprintf(stderr, "ggml-ep: set_mempolicy failed (rc=%ld, %s)\n", mp_rc, strerror(errno));
    }

    char nodes_str[64] = {0};
    int  off = 0;
    for (int i = 0; i < n; ++i) {
        off += snprintf(nodes_str + off, sizeof(nodes_str) - off,
                        i == 0 ? "%d" : ",%d", nodes[i]);
    }
    fprintf(stderr, "ggml-ep: pinned (pid=%d) to NUMA nodes [%s] (%d cpus, %s)\n",
            (int) getpid(), nodes_str, total_cpus,
            (n == 1) ? "MPOL_PREFERRED" : "MPOL_INTERLEAVE");
    return 0;
}

static int ggml_ep_pin_to_numa_node(int node_id) {
    return ggml_ep_pin_to_numa_nodes(&node_id, 1);
}

extern "C" struct ep_session * ggml_ep_get_session(void) {
    return g_ep_session;
}

static void ggml_ep_master_atexit(void) {
    if (g_ep_session) {
        struct ep_session * s = g_ep_session;
        g_ep_session = nullptr;  // null first so any concurrent getter sees end-of-life
        ep_session_destroy(s);
    }
}

extern "C" int ggml_ep_bootstrap_if_requested(void) {
    bool expected = false;
    if (!g_ep_bootstrapped.compare_exchange_strong(expected, true)) {
        return g_ep_session != nullptr ? 1 : 0;
    }

    const char * role_env = getenv("GGML_EP_ROLE");
    if (!role_env || role_env[0] == 0 || strcmp(role_env, "none") == 0) {
        return 0;
    }
    if (strcmp(role_env, "master") != 0) {
        fprintf(stderr, "ggml-ep: GGML_EP_ROLE='%s' not supported (only 'master' for now)\n", role_env);
        return 0;
    }

    int n_instances = 4;
    const char * n_env = getenv("GGML_EP_N_INSTANCES");
    if (n_env && n_env[0]) {
        n_instances = atoi(n_env);
    }
    if (n_instances < 1 || n_instances > EP_MAX_WORKERS + 1) {
        fprintf(stderr, "ggml-ep: invalid GGML_EP_N_INSTANCES=%d (must be 1..%d)\n",
                n_instances, EP_MAX_WORKERS + 1);
        return 0;
    }
    if (n_instances == 1) {
        // EP requested but only one instance — nothing to do.
        return 0;
    }

    const int n_workers = n_instances - 1;  // master is instance 0; spawn N-1 children

    struct ep_config cfg = {};
    cfg.n_workers              = n_workers;
    // 32 MiB sized to comfortably hold a `mul_mat_id` dst tensor at typical
    // configurations: decode (batch=1) writes ~hidden_dim × n_ids × 4B ≈ 72 KiB
    // for gemma-26B-A4B, ≈ 160 KiB for REAP-246B; prompt processing at
    // batch=512 is ~32-50 MiB depending on hidden size. 32 MiB shared mem
    // is trivial on a 1 TB system. Step (e) GGUF shard loading will let
    // us right-size this to the actual graph need.
    cfg.broadcast_bytes        = 32ULL * 1024 * 1024;
    cfg.gather_bytes_per_worker = 32ULL * 1024 * 1024;
    cfg.master_cpu             = -1;
    cfg.master_numa_node       = -1;
    cfg.worker_cpus            = nullptr;
    cfg.worker_numa_nodes      = nullptr;

    int rc = ep_session_create_master(&cfg, &g_ep_session);
    if (rc != 0) {
        fprintf(stderr, "ggml-ep: ep_session_create_master failed rc=%d\n", rc);
        g_ep_session = nullptr;
        return 0;
    }

    // Optional NUMA pinning: GGML_EP_NUMA_PIN=1 spreads instances across the
    // available NUMA nodes. With n_inst instances on n_nodes total, each gets
    // a CONSECUTIVE BLOCK of `n_nodes / n_inst` nodes (instance i takes
    // nodes [i*nodes_per_inst, (i+1)*nodes_per_inst)). MPOL_INTERLEAVE
    // round-robins pages within each instance's node block.
    //
    // On EPYC NPS4 (4 nodes):
    //   N=4: each instance one node          (master→0, w1→1, w2→2, w3→3)
    //   N=2: each instance spans two nodes   (master→0,1 ; w1→2,3)
    //   N=1: degenerate (no fork happens above)
    // This recovers the cross-node bandwidth that single-node pinning sacrifices
    // for non-MoE ops, which only master executes in drone mode.
    //
    // If n_inst doesn't evenly divide n_nodes, we fall back to single-node
    // pinning (worker w → node (w+1) mod n_nodes) — same as the prior behaviour.
    const char * pin_env = getenv("GGML_EP_NUMA_PIN");
    const bool pin_numa = (pin_env && pin_env[0] && pin_env[0] != '0');
    if (pin_numa) {
        int n_nodes = 4;  // EPYC NPS4 default; extend later if other topologies appear
        const int my_id = ep_session_instance_id(g_ep_session);  // master=0, workers=1..N-1
        if (n_nodes % n_instances == 0) {
            const int nodes_per_inst = n_nodes / n_instances;
            int nodes_buf[16];
            for (int i = 0; i < nodes_per_inst && i < 16; ++i) {
                nodes_buf[i] = my_id * nodes_per_inst + i;
            }
            ggml_ep_pin_to_numa_nodes(nodes_buf, nodes_per_inst);
        } else {
            const int my_node = my_id % n_nodes;
            ggml_ep_pin_to_numa_node(my_node);
        }
    }

    if (ep_session_role(g_ep_session) == EP_ROLE_WORKER) {
        // Worker child path. We forked out of master's address space; the
        // caller (llama.cpp main()) is sitting just above ggml_cpu_init().
        //
        // Phase 3.2(d.1) design: workers RETURN from bootstrap and run
        // llama.cpp normally. They independently mmap the same GGUF (kernel
        // page-cache de-dup means one set of physical pages, four VAs) and
        // execute the same forward pass deterministically. Synchronization
        // happens at MoE op boundaries via the EP path inside
        // ggml_compute_forward_mul_mat_id.
        //
        // To prevent multiple processes fighting over a TTY (or workers
        // EOF'ing on /dev/null'd stdin), redirect stdin/stdout to /dev/null.
        // stderr stays attached so worker errors are visible.
        const int wid = ep_session_instance_id(g_ep_session) - 1;
        fprintf(stderr, "ggml-ep: worker %d (pid=%d) returning to caller (will run llama.cpp normally)\n",
                wid, (int) getpid());
        fflush(stderr);

        // Disconnect stdin/stdout. We open /dev/null fresh and dup2 over the
        // existing fds so the inherited TTY mappings are replaced; freopen
        // would also work but dup2 is more explicit about which fds we
        // touch (we deliberately leave stderr alone for diagnostics).
        int devnull_r = open("/dev/null", O_RDONLY);
        int devnull_w = open("/dev/null", O_WRONLY);
        if (devnull_r >= 0) { dup2(devnull_r, 0); close(devnull_r); }
        if (devnull_w >= 0) { dup2(devnull_w, 1); close(devnull_w); }

        // Caller continues — for llama-cli/llama-bench/etc this means the
        // worker proceeds to parse args (inherited via fork), load model,
        // run inference, participate in EP sync at mul_mat_id ops, and exit.
        return 1;
    }

    // Master path. Register cleanup so workers are reaped at normal exit.
    atexit(ggml_ep_master_atexit);

    fprintf(stderr, "ggml-ep: master (pid=%d) spawned %d worker process(es), n_instances=%d\n",
            (int) getpid(), n_workers, n_instances);
    return 1;
}
