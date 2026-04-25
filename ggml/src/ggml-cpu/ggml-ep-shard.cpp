// See ggml-ep-shard.h for the API contract.

#include "ggml-ep-shard.h"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

// File-scope cache. mul_mat_id ops execute sequentially in the graph
// executor, so contention is rare; a single mutex around insertion is
// fine. Lookups are read-only after insertion and don't take the mutex.
namespace {
struct ShardEntry {
    void * buf;          // anon-mmap'd compact expert region
    size_t buf_bytes;
    size_t per_expert_bytes;
    int    n_kept;
    int    my_instance_id;
    int    n_instances;
};

std::unordered_map<const ggml_tensor *, ShardEntry> g_shards;
std::mutex                                          g_shards_mutex;

// Match the pinned-NUMA-node mask the bootstrap installed for this
// process. We re-derive it via get_mempolicy so the shard manager
// doesn't need to know about the bootstrap's node assignment.
unsigned long current_nodemask() {
    unsigned long mask = 0;
    int           policy = 0;
    long rc = syscall(SYS_get_mempolicy, &policy, &mask, sizeof(mask) * 8, (void *) 0, 0UL);
    if (rc < 0) {
        return 0;
    }
    return mask;
}

// Try to bind `buf` (length `bytes`) to the same NUMA nodes the calling
// process is pinned to. Best-effort: failure is logged but doesn't abort.
void mbind_to_current(void * buf, size_t bytes) {
    unsigned long mask = current_nodemask();
    if (mask == 0) {
        return;
    }
    int policy = MPOL_INTERLEAVE;
    // Single-bit mask → MPOL_BIND for stricter locality.
    if ((mask & (mask - 1)) == 0) {
        policy = MPOL_BIND;
    }
    long rc = syscall(SYS_mbind, buf, bytes, policy, &mask, sizeof(mask) * 8, 0UL);
    if (rc != 0) {
        fprintf(stderr, "ggml-ep-shard: mbind failed (rc=%ld)\n", rc);
    }
}
} // namespace

extern "C" int ggml_ep_shard_enabled(void) {
    static int s = -1;
    if (s < 0) {
        const char * env = getenv("GGML_EP_SHARD");
        s = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return s;
}

extern "C" void * ggml_ep_shard_lookup(const ggml_tensor * src0,
                                      int my_id,
                                      int n_inst) {
    if (n_inst <= 1 || src0 == nullptr) {
        return nullptr;
    }

    // Fast path: already cached.
    {
        std::lock_guard<std::mutex> lock(g_shards_mutex);
        auto it = g_shards.find(src0);
        if (it != g_shards.end()) {
            // Sanity: if the EP partition changed (re-bootstrap), bail.
            if (it->second.my_instance_id == my_id &&
                it->second.n_instances    == n_inst) {
                return it->second.buf;
            }
            return nullptr;
        }
    }

    // Slow path: allocate + copy under the lock so two threads don't
    // each allocate. The mutex blocks at most for the duration of one
    // memcpy of n_kept * per_expert_bytes.
    std::lock_guard<std::mutex> lock(g_shards_mutex);
    auto it = g_shards.find(src0);
    if (it != g_shards.end()) {
        return it->second.buf;
    }

    // Inspect tensor shape. Expert tensors have ne[2] = n_experts; we
    // shard along that axis. Tensors with ne[2] < n_inst aren't worth
    // sharding (some instances would have nothing to do).
    const int64_t n_experts = src0->ne[2];
    if (n_experts < n_inst) {
        return nullptr;
    }
    const size_t total_bytes      = ggml_nbytes(src0);
    const size_t per_expert_bytes = total_bytes / (size_t) n_experts;
    if (per_expert_bytes * (size_t) n_experts != total_bytes) {
        // Shape isn't expert-major or has padding we don't understand.
        return nullptr;
    }

    // Compute kept-expert indices: ceil((n_experts - my_id) / n_inst).
    int n_kept = 0;
    for (int e = my_id; e < (int) n_experts; e += n_inst) {
        ++n_kept;
    }
    const size_t buf_bytes = (size_t) n_kept * per_expert_bytes;

    void * buf = mmap(nullptr, buf_bytes, PROT_READ | PROT_WRITE,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        fprintf(stderr, "ggml-ep-shard: mmap %zu bytes failed for tensor %s\n",
                buf_bytes, src0->name);
        return nullptr;
    }
    mbind_to_current(buf, buf_bytes);

    // Copy kept experts into the compact buffer. The source is the
    // mmap'd GGUF region (or whatever ggml has loaded). Local index is
    // (e - my_id) / n_inst, which is contiguous 0..n_kept-1.
    for (int e = my_id, k = 0; e < (int) n_experts; e += n_inst, ++k) {
        const char * src = (const char *) src0->data + (size_t) e * per_expert_bytes;
        char *       dst = (char *) buf             + (size_t) k * per_expert_bytes;
        memcpy(dst, src, per_expert_bytes);
    }

    ShardEntry entry = {};
    entry.buf              = buf;
    entry.buf_bytes        = buf_bytes;
    entry.per_expert_bytes = per_expert_bytes;
    entry.n_kept           = n_kept;
    entry.my_instance_id   = my_id;
    entry.n_instances      = n_inst;
    g_shards.emplace(src0, entry);

    fprintf(stderr, "ggml-ep-shard: %s sharded (%d/%lld experts, %zu MiB local)\n",
            src0->name ? src0->name : "(anon)",
            n_kept, (long long) n_experts, buf_bytes >> 20);
    return buf;
}
