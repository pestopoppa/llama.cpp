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

// Pin the calling process to all CPUs in `node_id` (parsed from
// /sys/devices/system/node/nodeN/cpulist) and set the default memory
// allocation policy to prefer that node. Returns 0 on success, -1 if the
// node sysfs path is missing or unparseable. Best-effort: failure is
// logged but doesn't abort EP — the process just runs unpinned.
static int ggml_ep_pin_to_numa_node(int node_id) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node_id);
    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ggml-ep: cpulist for node %d missing (%s); skipping pin\n",
                node_id, path);
        return -1;
    }
    char buf[1024] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) {
        fprintf(stderr, "ggml-ep: cpulist for node %d empty; skipping pin\n", node_id);
        return -1;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int n_cpus = 0;
    char * tok = strtok(buf, ",\n");
    while (tok) {
        int s = -1, e = -1;
        if (sscanf(tok, "%d-%d", &s, &e) == 2) {
            for (int c = s; c <= e; ++c) { CPU_SET(c, &cpuset); ++n_cpus; }
        } else if (sscanf(tok, "%d", &s) == 1) {
            CPU_SET(s, &cpuset); ++n_cpus;
        }
        tok = strtok(nullptr, ",\n");
    }
    if (n_cpus == 0) {
        fprintf(stderr, "ggml-ep: parsed 0 cpus for node %d; skipping pin\n", node_id);
        return -1;
    }
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        fprintf(stderr, "ggml-ep: sched_setaffinity for node %d failed (%s)\n",
                node_id, strerror(errno));
        return -1;
    }

    // MPOL_PREFERRED: future allocations prefer this node, fall back to
    // others on pressure. This affects heap (KV cache, scratch buffers)
    // and anonymous mmaps; it does NOT relocate the GGUF file mmap (those
    // pages are governed by the kernel page cache + first-toucher).
    unsigned long nodemask = 1UL << node_id;
    long mp_rc = syscall(SYS_set_mempolicy, MPOL_PREFERRED, &nodemask, sizeof(nodemask) * 8);
    if (mp_rc != 0) {
        fprintf(stderr, "ggml-ep: set_mempolicy(node %d) failed (rc=%ld, %s)\n",
                node_id, mp_rc, strerror(errno));
        // not fatal — affinity is the bigger lever
    }

    fprintf(stderr, "ggml-ep: pinned (pid=%d) to NUMA node %d (%d cpus, MPOL_PREFERRED)\n",
            (int) getpid(), node_id, n_cpus);
    return 0;
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

    // Optional NUMA pinning: GGML_EP_NUMA_PIN=1 spreads instances one-per-node
    // (master → node 0, worker w → node (w+1) mod n_nodes). On EPYC NPS4 this
    // gives each instance a dedicated 24-core CCD-quad and dedicated DDR
    // channels — escapes the cross-instance memory-bandwidth contention that
    // currently throttles the unpinned multi-process configuration.
    const char * pin_env = getenv("GGML_EP_NUMA_PIN");
    const bool pin_numa = (pin_env && pin_env[0] && pin_env[0] != '0');
    if (pin_numa) {
        int n_nodes = 4;  // EPYC NPS4 default; extend later if other topologies appear
        int my_node;
        if (ep_session_role(g_ep_session) == EP_ROLE_MASTER) {
            my_node = 0;
        } else {
            // worker instance_id is 1..n_workers; assign to node (1..N-1) % n_nodes
            my_node = ep_session_instance_id(g_ep_session) % n_nodes;
        }
        ggml_ep_pin_to_numa_node(my_node);
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
