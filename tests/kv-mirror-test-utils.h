#pragma once

#include "../src/llama-kv-host-mirror.h"
#include "ggml-backend.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// Round-trip the initialized inputs through the same ops and pinned host tier
// used by the context binding. A missing device implementation is a failure.
inline size_t kv_mirror_test_roundtrip(ggml_backend_t backend, ggml_context * ctx) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    const ggml_kv_mirror_ops * ops = nullptr;
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ops = llama_kvm::memcpy_ops();
    } else {
        auto get_ops = (ggml_backend_kv_mirror_get_ops_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_kv_mirror_get_ops");
        if (get_ops) {
            ops = get_ops();
        }
    }
    if (!ops || ops->version != GGML_KV_MIRROR_OPS_VERSION) {
        throw std::runtime_error("backend has no KV mirror ops");
    }
    auto destroy = [ops](ggml_kv_mirror_dev * dev) { ops->dev_free(dev); };
    std::unique_ptr<ggml_kv_mirror_dev, decltype(destroy)> dev(ops->dev_init(backend), destroy);
    if (!dev) {
        throw std::runtime_error("KV mirror device init failed");
    }
    std::vector<ggml_tensor *> tensors;
    size_t bytes = 0;
    for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src || !t->data || !t->buffer || ggml_nbytes(t) == 0) {
            continue;
        }
        tensors.push_back(t);
        bytes += ggml_nbytes(t);
    }
    if (!bytes) {
        throw std::runtime_error("KV mirror probe has no allocated tensors");
    }
    llama_kvm::host_tier tier;
    std::string err;
    if (!tier.reserve(bytes, -1, 2u << 20, err) || !tier.register_with(ops, dev.get())) {
        throw std::runtime_error("KV mirror host registration failed: " + err);
    }
    ggml_backend_synchronize(backend);
    std::vector<ggml_kv_mirror_frag> frags;
    std::vector<uint8_t> reference(bytes);
    size_t offset = 0;
    for (auto * t : tensors) {
        const size_t n = ggml_nbytes(t);
        ggml_backend_tensor_get(t, reference.data() + offset, 0, n);
        llama_kvm::append_frag(tier, (uint8_t *) t->data, tier.data() + offset, n, frags);
        offset += n;
    }
    const uint64_t down = ops->copy_async(dev.get(), frags.data(), frags.size(),
            GGML_KV_MIRROR_D2H, GGML_KV_MIRROR_PATH_AUTO);
    if (!down) {
        throw std::runtime_error("KV mirror D2H was not issued");
    }
    ops->ticket_wait(dev.get(), down);
    if (memcmp(reference.data(), tier.data(), bytes) != 0) {
        throw std::runtime_error("KV mirror D2H bytes differ");
    }
    // Poison the device copies so an inert H2D path cannot pass.
    offset = 0;
    for (auto * t : tensors) {
        std::vector<uint8_t> poison(ggml_nbytes(t), 0xa5);
        ggml_backend_tensor_set(t, poison.data(), 0, poison.size());
    }
    const uint64_t up = ops->copy_async(dev.get(), frags.data(), frags.size(),
            GGML_KV_MIRROR_H2D, GGML_KV_MIRROR_PATH_AUTO);
    if (!up) {
        throw std::runtime_error("KV mirror H2D was not issued");
    }
    ops->ticket_wait(dev.get(), up);
    for (auto * t : tensors) {
        std::vector<uint8_t> restored(ggml_nbytes(t));
        ggml_backend_tensor_get(t, restored.data(), 0, restored.size());
        if (memcmp(reference.data() + offset, restored.data(), restored.size()) != 0) {
            throw std::runtime_error("KV mirror H2D bytes differ");
        }
        offset += restored.size();
    }
    return bytes;
}
