// HSF-5.P (experimental): host-pinned KV mirror, device side.
//
// Pins and maps host memory (cudaHostRegister / hipHostRegister with Mapped|Portable), and moves fragment
// lists between device memory and that host memory on a SIDE stream that is ordered after the backend's own
// stream with an event. Nothing here runs inside a graph capture, and nothing blocks the host except
// ticket_wait (save/restore paths only) and the staging-ring back-pressure described in copy_async.
//
// Host-side logic (which cells, when, the checkpoint schedule, fragment construction) lives in libllama
// (src/llama-kv-host-mirror.{h,cpp}); see docs/design/hsf-5-host-kv-mirror-costing-20261024.md in epyc-root.

#include "kv-host-mirror.cuh"

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(GGML_USE_HIP)
#ifndef cudaHostRegisterMapped
#define cudaHostRegisterMapped hipHostRegisterMapped
#endif
#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#endif
#ifndef cudaEventQuery
#define cudaEventQuery hipEventQuery
#endif
#ifndef cudaErrorNotReady
#define cudaErrorNotReady hipErrorNotReady
#endif
#ifndef cudaMemcpyKind
#define cudaMemcpyKind hipMemcpyKind
#endif
#endif // defined(GGML_USE_HIP)

// ---------------------------------------------------------------------------------------------------------------
// Source-copied from ROCm/mori, src/umbp/local/tiers/device_gather.hip @ 37cdd8e9f8a3b55d0e133ac8e0e15ebeae3ca62b
// (https://github.com/ROCm/mori, MIT): GatherFragmentsKernel and its launch geometry (kThreadsPerBlock,
// kMaxBlocksPerFragment, uint4 vector path with a byte-loop fallback). Adapted for HSF-5.P: renamed namespace,
// CUDA/HIP-portable spelling, descriptor staging moved to a per-ticket ring (below). The kernel body is unchanged.
//
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
namespace kvm_umbp {

struct DeviceGatherFragment {
    const void * src;
    void *       dst;
    size_t       bytes;
};

constexpr unsigned kThreadsPerBlock = 256;

// Enough parallelism to keep the link busy on the largest fragment without
// launching blocks that immediately exit on the smallest. Fragments below one
// block's worth of work still get one block each.
constexpr unsigned kMaxBlocksPerFragment = 8;

constexpr size_t kVectorBytes = 16;

// A block copies its slice of one fragment. Fragments are independent, so the
// grid is flat: blockIdx.x / blocks_per_fragment picks the fragment and the
// remainder picks the slice.
__global__ void GatherFragmentsKernel(const DeviceGatherFragment * __restrict__ fragments,
                                      unsigned blocks_per_fragment) {
    const DeviceGatherFragment fragment = fragments[blockIdx.x / blocks_per_fragment];
    const char * __restrict__ src = static_cast<const char *>(fragment.src);
    char * __restrict__       dst = static_cast<char *>(fragment.dst);

    const size_t slice  = blockIdx.x % blocks_per_fragment;
    const size_t index  = slice * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blocks_per_fragment) * blockDim.x;

    // KV fragments are page-aligned in practice; the byte loop is correctness
    // insurance for pools whose element size makes them odd.
    const uintptr_t mask = reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst) |
                           static_cast<uintptr_t>(fragment.bytes);
    if ((mask & (kVectorBytes - 1)) == 0) {
        const uint4 * __restrict__ src_vector = reinterpret_cast<const uint4 *>(src);
        uint4 * __restrict__       dst_vector = reinterpret_cast<uint4 *>(dst);
        const size_t count = fragment.bytes / kVectorBytes;
        for (size_t i = index; i < count; i += stride) dst_vector[i] = src_vector[i];
        return;
    }
    for (size_t i = index; i < fragment.bytes; i += stride) dst[i] = src[i];
}

// launch geometry, as in mori's LaunchDeviceGather
static unsigned blocks_per_fragment_for(size_t max_bytes) {
    const size_t chunks = (max_bytes + kVectorBytes - 1) / kVectorBytes;
    return static_cast<unsigned>(std::clamp<size_t>((chunks + kThreadsPerBlock - 1) / kThreadsPerBlock, 1, kMaxBlocksPerFragment));
}

} // namespace kvm_umbp
// End of the source-copied section.
// ---------------------------------------------------------------------------------------------------------------

// Fragments at or above this size go to the copy engine on PATH_AUTO: one large async memcpy already amortizes
// its submission (mori device_gather.h: "Only worth it below roughly 128 KiB per fragment").
static constexpr size_t KVM_KERNEL_MAX_FRAG_BYTES = 128u * 1024u;

// tickets in flight before copy_async waits for the oldest one (back-pressure, never reached at one copy per
// decode step plus one checkpoint every k steps)
static constexpr int KVM_RING = 32;

struct ggml_kv_mirror_dev {
    int device = 0;
    ggml_backend_cuda_context * bctx = nullptr;

    cudaStream_t side = nullptr;

    cudaEvent_t ev_main[KVM_RING] = {};
    cudaEvent_t ev_done[KVM_RING] = {};
    uint64_t    ticket [KVM_RING] = {};

    // per-ring-slot descriptor staging (the slot's previous upload is complete once its ticket is)
    kvm_umbp::DeviceGatherFragment * desc_host[KVM_RING] = {};
    kvm_umbp::DeviceGatherFragment * desc_dev [KVM_RING] = {};
    size_t                           desc_cap [KVM_RING] = {};

    uint64_t next_ticket = 1;

    std::vector<kvm_umbp::DeviceGatherFragment> scratch;
};

static cudaStream_t kvm_main_stream(ggml_kv_mirror_dev * dev) {
    // stream 0 of the backend's device: the stream graphs are captured on and launched from
    return dev->bctx->stream(dev->device, 0);
}

static ggml_kv_mirror_dev_t kvm_dev_init(ggml_backend_t backend) {
    if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
        return nullptr;
    }

    auto * bctx = (ggml_backend_cuda_context *) backend->context;

    auto * dev = new ggml_kv_mirror_dev;
    dev->device = bctx->device;
    dev->bctx   = bctx;

    ggml_cuda_set_device(dev->device);
    CUDA_CHECK(cudaStreamCreateWithFlags(&dev->side, cudaStreamNonBlocking));
    for (int i = 0; i < KVM_RING; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_main[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&dev->ev_done[i], cudaEventDisableTiming));
    }

    return dev;
}

static void kvm_dev_free(ggml_kv_mirror_dev_t dev) {
    if (dev == nullptr) {
        return;
    }
    ggml_cuda_set_device(dev->device);
    (void) cudaStreamSynchronize(dev->side);
    for (int i = 0; i < KVM_RING; ++i) {
        (void) cudaEventDestroy(dev->ev_main[i]);
        (void) cudaEventDestroy(dev->ev_done[i]);
        if (dev->desc_host[i]) { (void) cudaFreeHost(dev->desc_host[i]); }
        if (dev->desc_dev[i])  { (void) cudaFree(dev->desc_dev[i]); }
    }
    (void) cudaStreamDestroy(dev->side);
    delete dev;
}

static bool kvm_host_register(ggml_kv_mirror_dev_t dev, void * host, size_t bytes, void ** alias_out) {
    ggml_cuda_set_device(dev->device);
    *alias_out = nullptr;

    if (cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: host register of %.1f MiB failed\n", __func__, bytes/1024.0/1024.0);
        return false;
    }

    void * alias = nullptr;
    if (cudaHostGetDevicePointer(&alias, host, 0) != cudaSuccess || alias == nullptr) {
        (void) cudaGetLastError();
        // registered (pinned) but not mapped: the copy-engine path still works, the kernel path does not
        GGML_LOG_WARN("%s: no device alias for a registered host piece; kernel path disabled for it\n", __func__);
        return true;
    }

    *alias_out = alias;
    return true;
}

static void kvm_host_unregister(ggml_kv_mirror_dev_t dev, void * host) {
    ggml_cuda_set_device(dev->device);
    if (cudaHostUnregister(host) != cudaSuccess) {
        (void) cudaGetLastError();
    }
}

static int kvm_slot(uint64_t ticket) {
    return (int) (ticket % KVM_RING);
}

static bool kvm_stage_descriptors(ggml_kv_mirror_dev * dev, int slot, size_t count) {
    if (dev->desc_cap[slot] >= count) {
        return true;
    }
    const size_t cap   = std::max<size_t>(count*2, 256);
    const size_t bytes = cap*sizeof(kvm_umbp::DeviceGatherFragment);

    kvm_umbp::DeviceGatherFragment * h = nullptr;
    kvm_umbp::DeviceGatherFragment * d = nullptr;
    if (cudaMallocHost((void **) &h, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaMalloc((void **) &d, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(h);
        return false;
    }
    if (dev->desc_host[slot]) { (void) cudaFreeHost(dev->desc_host[slot]); }
    if (dev->desc_dev[slot])  { (void) cudaFree(dev->desc_dev[slot]); }
    dev->desc_host[slot] = h;
    dev->desc_dev [slot] = d;
    dev->desc_cap [slot] = cap;
    return true;
}

static uint64_t kvm_copy_async(ggml_kv_mirror_dev_t dev, const ggml_kv_mirror_frag * frags, size_t n,
                               ggml_kv_mirror_dir dir, ggml_kv_mirror_path path) {
    if (n == 0) {
        return 0;
    }

    ggml_cuda_set_device(dev->device);

    const uint64_t t    = dev->next_ticket;
    const int      slot = kvm_slot(t);

    // back-pressure: the ring slot (its events and descriptor staging) is reused only after its previous ticket
    // completed. With one ticket per decode step this waits only if KVM_RING copies are still in flight.
    if (dev->ticket[slot] != 0) {
        CUDA_CHECK(cudaEventSynchronize(dev->ev_done[slot]));
        dev->ticket[slot] = 0;
    }

    // split: small fragments with a device alias -> one kernel launch; the rest -> copy engine
    auto & kfr = dev->scratch;
    kfr.clear();
    size_t kmax = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto & f = frags[i];
        if (f.bytes == 0) {
            continue;
        }
        const bool aliased = f.host_alias != nullptr;
        const bool kernel  = aliased && (path == GGML_KV_MIRROR_PATH_KERNEL ||
                                        (path == GGML_KV_MIRROR_PATH_AUTO && f.bytes < KVM_KERNEL_MAX_FRAG_BYTES));
        if (kernel) {
            if (dir == GGML_KV_MIRROR_D2H) {
                kfr.push_back({ f.dev, f.host_alias, f.bytes });
            } else {
                kfr.push_back({ f.host_alias, f.dev, f.bytes });
            }
            kmax = std::max(kmax, f.bytes);
        }
    }

    // order the side stream after everything already queued on the backend stream (the step's graph launch and
    // its output copies). This is issued from the host after the launch returned, never inside a capture.
    CUDA_CHECK(cudaEventRecord(dev->ev_main[slot], kvm_main_stream(dev)));
    CUDA_CHECK(cudaStreamWaitEvent(dev->side, dev->ev_main[slot], 0));

    if (!kfr.empty()) {
        const unsigned bpf = kvm_umbp::blocks_per_fragment_for(kmax);
        if (kfr.size() > std::numeric_limits<unsigned>::max()/bpf || !kvm_stage_descriptors(dev, slot, kfr.size())) {
            // fall back to the copy engine for these fragments
            for (const auto & k : kfr) {
                CUDA_CHECK(cudaMemcpyAsync(k.dst, k.src, k.bytes,
                            dir == GGML_KV_MIRROR_D2H ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, dev->side));
            }
        } else {
            memcpy(dev->desc_host[slot], kfr.data(), kfr.size()*sizeof(kfr[0]));
            CUDA_CHECK(cudaMemcpyAsync(dev->desc_dev[slot], dev->desc_host[slot], kfr.size()*sizeof(kfr[0]),
                        cudaMemcpyHostToDevice, dev->side));
            kvm_umbp::GatherFragmentsKernel<<<dim3((unsigned) kfr.size()*bpf), dim3(kvm_umbp::kThreadsPerBlock), 0, dev->side>>>(
                    dev->desc_dev[slot], bpf);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const auto & f = frags[i];
        if (f.bytes == 0) {
            continue;
        }
        const bool aliased = f.host_alias != nullptr;
        const bool kernel  = aliased && (path == GGML_KV_MIRROR_PATH_KERNEL ||
                                        (path == GGML_KV_MIRROR_PATH_AUTO && f.bytes < KVM_KERNEL_MAX_FRAG_BYTES));
        if (kernel) {
            continue;
        }
        if (dir == GGML_KV_MIRROR_D2H) {
            CUDA_CHECK(cudaMemcpyAsync(f.host, f.dev, f.bytes, cudaMemcpyDeviceToHost, dev->side));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(f.dev, f.host, f.bytes, cudaMemcpyHostToDevice, dev->side));
        }
    }

    CUDA_CHECK(cudaEventRecord(dev->ev_done[slot], dev->side));

    dev->ticket[slot] = t;
    dev->next_ticket++;

    return t;
}

static bool kvm_ticket_done(ggml_kv_mirror_dev_t dev, uint64_t t) {
    if (t == 0) {
        return true;
    }
    const int slot = kvm_slot(t);
    if (dev->ticket[slot] != t) {
        return true; // the slot was recycled, which happens only after this ticket completed
    }
    ggml_cuda_set_device(dev->device);
    const cudaError_t err = cudaEventQuery(dev->ev_done[slot]);
    if (err == cudaSuccess) {
        return true;
    }
    if (err != cudaErrorNotReady) {
        CUDA_CHECK(err);
    }
    (void) cudaGetLastError();
    return false;
}

static void kvm_ticket_wait(ggml_kv_mirror_dev_t dev, uint64_t t) {
    if (t == 0) {
        return;
    }
    const int slot = kvm_slot(t);
    if (dev->ticket[slot] != t) {
        return;
    }
    ggml_cuda_set_device(dev->device);
    CUDA_CHECK(cudaEventSynchronize(dev->ev_done[slot]));
}

static void kvm_backend_wait(ggml_kv_mirror_dev_t dev, uint64_t t) {
    if (t == 0) {
        return;
    }
    const int slot = kvm_slot(t);
    if (dev->ticket[slot] != t) {
        return;
    }
    ggml_cuda_set_device(dev->device);
    // issued from the host before the next graph launch, so it is not part of any capture
    CUDA_CHECK(cudaStreamWaitEvent(kvm_main_stream(dev), dev->ev_done[slot], 0));
}

static const ggml_kv_mirror_ops kvm_ops = {
    /* .version         = */ GGML_KV_MIRROR_OPS_VERSION,
    /* .dev_init        = */ kvm_dev_init,
    /* .dev_free        = */ kvm_dev_free,
    /* .host_register   = */ kvm_host_register,
    /* .host_unregister = */ kvm_host_unregister,
    /* .copy_async      = */ kvm_copy_async,
    /* .ticket_done     = */ kvm_ticket_done,
    /* .ticket_wait     = */ kvm_ticket_wait,
    /* .backend_wait    = */ kvm_backend_wait,
};

const ggml_kv_mirror_ops * ggml_backend_cuda_kv_mirror_get_ops(void) {
    return &kvm_ops;
}
