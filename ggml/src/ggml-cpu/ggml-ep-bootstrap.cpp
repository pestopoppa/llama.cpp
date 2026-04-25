// See ggml-ep-bootstrap.h for the API contract.

#include "ggml-ep-bootstrap.h"
#include "ggml-ep-dispatcher.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static struct ep_session * g_ep_session = nullptr;
static std::atomic<bool>   g_ep_bootstrapped{false};

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
    // 1 MiB scratch is comfortably above the largest expected hidden state
    // (e.g. 5120 floats = 20 KiB for Qwen3-Coder-REAP-246B). Step (d)+ will
    // size this to the model's hidden dim once the integration knows it.
    cfg.broadcast_bytes        = 1024 * 1024;
    cfg.gather_bytes_per_worker = 1024 * 1024;
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

    if (ep_session_role(g_ep_session) == EP_ROLE_WORKER) {
        // Worker child path. We forked out of master's address space; the
        // caller (llama.cpp main()) is sitting just above ggml_cpu_init().
        // Returning would cause the worker to load a model and run inference
        // independently — not what we want. Instead, sit in a wait loop
        // that ack's every GO immediately with no compute. Master will
        // teardown via EXIT signal when it shuts down (or PR_SET_PDEATHSIG
        // will SIGTERM us if master crashes).
        //
        // This is intentionally a no-op for step (b): we are validating the
        // fork+lifecycle harness only. Step (d)+ will replace this body
        // with the actual MoE expert compute path.
        const int wid = ep_session_instance_id(g_ep_session) - 1;
        fprintf(stderr, "ggml-ep: worker %d (pid=%d) entering passive wait loop\n",
                wid, (int) getpid());
        while (true) {
            int r = ep_worker_wait_go(g_ep_session);
            if (r != 0) break;  // EXIT received
            ep_worker_signal_done(g_ep_session);  // immediate ack, no compute
        }
        // Worker only: do not run atexit handlers (they belong to master's
        // pre-fork state and would corrupt master-only globals).
        ep_session_destroy(g_ep_session);
        g_ep_session = nullptr;
        _exit(0);
    }

    // Master path. Register cleanup so workers are reaped at normal exit.
    atexit(ggml_ep_master_atexit);

    fprintf(stderr, "ggml-ep: master (pid=%d) spawned %d worker process(es), n_instances=%d\n",
            (int) getpid(), n_workers, n_instances);
    return 1;
}
