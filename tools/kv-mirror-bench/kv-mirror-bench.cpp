// HSF-5.P host KV mirror microbench, steps 1-2 of the acceptance microbench in epyc-root
// docs/design/hsf-5-host-kv-mirror-costing-20261024.md ("Unmeasured" section). No model is loaded.
//
//   step 1: D2H from a device buffer into the hipHostRegister(Mapped) host tier at 4 KiB, 64 KiB, 400 KiB, 1 MiB
//           and 157 MB (one recurrent checkpoint), copy engine and kernel path; plus the per-step KV tail shapes
//           (128 separate fragments vs one staged fragment of the same 407,630 bytes)
//   step 2: the gather kernel (source-copied UMBP GatherFragmentsKernel) reading 2,176-byte KV rows in runs of
//           1, 16 and 512 contiguous tokens from the mapped tier into device memory, GB/s vs the 28.89 H2D plateau
//
// Every case verifies its bytes (ok=true/false). Output: one JSON object per line on stdout.
//
// usage: llama-kv-mirror-bench [--node N] [--reps N] [--device-index N]

#include "llama-kv-host-mirror.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-kv-mirror.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    int node = 3;
    int reps = 3;
    int dev_index = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--node" && i + 1 < argc) {
            node = atoi(argv[++i]);
        } else if (a == "--reps" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (a == "--device-index" && i + 1 < argc) {
            dev_index = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [--node N] [--reps N] [--device-index N]\n", argv[0]);
            return 2;
        }
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0, k = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            if ((int) k++ == dev_index) {
                dev = d;
                break;
            }
        }
    }
    if (!dev) {
        fprintf(stderr, "no GPU device\n");
        return 3;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    auto get_ops = (ggml_backend_kv_mirror_get_ops_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_mirror_get_ops");
    if (!backend || !get_ops) {
        fprintf(stderr, "backend %s has no kv mirror ops\n", ggml_backend_dev_name(dev));
        return 4;
    }
    const ggml_kv_mirror_ops * ops = get_ops();
    ggml_kv_mirror_dev_t mdev = ops->dev_init(backend);
    if (!mdev) {
        fprintf(stderr, "kv mirror dev_init failed\n");
        return 5;
    }

    const size_t big = (size_t) 156900000 / 4096 * 4096; // one recurrent checkpoint (156.9 MB)
    const size_t gather_rows = 32768;                     // 2,176 B x 32768 = 71.3 MB per gather case
    const size_t row = 2176;
    const size_t dev_bytes = std::max(big, 2*gather_rows*row) + (1 << 20);

    // device buffer as a tensor so the bytes can be checked with tensor get/set
    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) dev_bytes);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "device allocation of %zu bytes failed\n", dev_bytes);
        return 6;
    }

    std::vector<uint8_t> pattern(dev_bytes);
    for (size_t i = 0; i < dev_bytes; ++i) {
        pattern[i] = (uint8_t) ((i * 2654435761u) >> 13);
    }
    ggml_backend_tensor_set(t, pattern.data(), 0, dev_bytes);

    llama_kvm::host_tier tier;
    std::string err;
    if (!tier.reserve(dev_bytes, node, (size_t) 1 << 30, err)) {
        fprintf(stderr, "host tier: %s\n", err.c_str());
        return 7;
    }
    const double t_reg0 = now_us();
    const bool all_reg = tier.register_with(ops, mdev);
    printf("{\"event\":\"host_tier\",\"bytes\":%zu,\"node\":%d,\"pieces\":%zu,\"all_registered\":%s,\"register_ms\":%.1f}\n",
           tier.bytes(), node, tier.n_registered_pieces(), all_reg ? "true" : "false", (now_us() - t_reg0)/1000.0);

    uint8_t * d0 = (uint8_t *) t->data;

    auto run = [&](const char * step, const char * name, std::vector<ggml_kv_mirror_frag> & f, ggml_kv_mirror_dir dir, ggml_kv_mirror_path path) {
        size_t bytes = 0;
        for (const auto & x : f) bytes += x.bytes;
        for (int r = -1; r < reps; ++r) { // r = -1: warmup, not reported
            if (dir == GGML_KV_MIRROR_D2H) {
                for (const auto & x : f) memset(x.host, 0, x.bytes);
            } else {
                ggml_backend_tensor_set(t, std::vector<uint8_t>(dev_bytes, 0).data(), 0, dev_bytes);
            }
            const double t0 = now_us();
            const uint64_t tk = ops->copy_async(mdev, f.data(), f.size(), dir, path);
            ops->ticket_wait(mdev, tk);
            const double us = now_us() - t0;
            bool ok = tk != 0;
            if (dir == GGML_KV_MIRROR_D2H) {
                for (const auto & x : f) ok = ok && memcmp(x.host, pattern.data() + ((uint8_t *) x.dev - d0), x.bytes) == 0;
            } else {
                std::vector<uint8_t> back(dev_bytes);
                ggml_backend_tensor_get(t, back.data(), 0, dev_bytes);
                for (const auto & x : f) ok = ok && memcmp(back.data() + ((uint8_t *) x.dev - d0), x.host, x.bytes) == 0;
            }
            if (r >= 0) {
                printf("{\"step\":\"%s\",\"case\":\"%s\",\"dir\":\"%s\",\"path\":\"%s\",\"rep\":%d,\"n_frags\":%zu,\"bytes\":%zu,\"us\":%.2f,\"gbps\":%.3f,\"ok\":%s}\n",
                       step, name, dir == GGML_KV_MIRROR_D2H ? "d2h" : "h2d",
                       path == GGML_KV_MIRROR_PATH_KERNEL ? "kernel" : path == GGML_KV_MIRROR_PATH_COPY_ENGINE ? "ce" : "auto",
                       r, f.size(), bytes, us, bytes / us / 1000.0, ok ? "true" : "false");
                fflush(stdout);
            }
        }
    };

    // step 1: D2H latency curve
    for (size_t sz : { (size_t) 4096, (size_t) 65536, (size_t) 409600, (size_t) 1 << 20, big }) {
        for (auto path : { GGML_KV_MIRROR_PATH_COPY_ENGINE, GGML_KV_MIRROR_PATH_KERNEL }) {
            std::vector<ggml_kv_mirror_frag> f;
            llama_kvm::append_frag(tier, d0, tier.data(), sz, f);
            run("1", ("single_" + std::to_string(sz)).c_str(), f, GGML_KV_MIRROR_D2H, path);
        }
    }

    // step 1b: the per-step KV tail, spec ON (407,630 B): 128 fragments vs one staged fragment
    {
        const size_t tail = 407630;
        const size_t per = tail / 128;
        std::vector<ggml_kv_mirror_frag> f128;
        for (size_t i = 0; i < 128; ++i) {
            // spread like 16 layers x K,V x 4 slots: every fragment in a different 1 MiB stripe
            const size_t off = i * ((size_t) 1 << 20);
            llama_kvm::append_frag(tier, d0 + off, tier.data() + off, per, f128);
        }
        for (auto path : { GGML_KV_MIRROR_PATH_COPY_ENGINE, GGML_KV_MIRROR_PATH_KERNEL, GGML_KV_MIRROR_PATH_AUTO }) {
            run("1b", "tail_128frags", f128, GGML_KV_MIRROR_D2H, path);
        }
        std::vector<ggml_kv_mirror_frag> f1;
        llama_kvm::append_frag(tier, d0, tier.data(), tail, f1);
        run("1b", "tail_staged_1frag", f1, GGML_KV_MIRROR_D2H, GGML_KV_MIRROR_PATH_COPY_ENGINE);
    }

    // step 2: gather from the mapped tier into device memory, runs of 1 / 16 / 512 tokens of 2,176 B
    for (size_t run_tok : { (size_t) 1, (size_t) 16, (size_t) 512 }) {
        std::vector<ggml_kv_mirror_frag> f;
        const size_t fb = run_tok * row;
        const size_t n = gather_rows / run_tok;
        for (size_t i = 0; i < n; ++i) {
            // destination cells interleaved (every other run), as in a fragmented --kv-unified restore
            const size_t src = i * fb;
            const size_t dst = 2 * i * fb;
            f.push_back({ d0 + dst, tier.data() + src, tier.alias_of(tier.data() + src), fb });
            memcpy(tier.data() + src, pattern.data() + dst, fb);
        }
        // the H2D check compares device bytes with the host source, which run() does not overwrite for H2D
        run("2", ("gather_run" + std::to_string(run_tok)).c_str(), f, GGML_KV_MIRROR_H2D, GGML_KV_MIRROR_PATH_KERNEL);
        run("2", ("gather_run" + std::to_string(run_tok)).c_str(), f, GGML_KV_MIRROR_H2D, GGML_KV_MIRROR_PATH_COPY_ENGINE);
    }

    tier.release();
    ops->dev_free(mdev);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
