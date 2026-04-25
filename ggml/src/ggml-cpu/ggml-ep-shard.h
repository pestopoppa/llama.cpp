// ggml-ep-shard — lazy per-tensor expert sharding for inter-process EP.
//
// When `GGML_EP_SHARD=1` and an inter-process EP session is active, the
// FIRST mul_mat_id call that sees a given `src0` expert tensor allocates
// a node-local anon-mmap buffer sized for only this instance's 1/N of
// experts, copies the kept experts in, and stores the buffer pointer in
// a process-local cache. Subsequent calls reuse the buffer.
//
// The original mmap'd file pages stay mapped; once they aren't being
// read by the hot path the kernel evicts them via LRU, leaving each
// instance with only its 1/N of expert weights resident in node-local
// RAM. This converts cross-NUMA expert reads (the dominant cost on
// large MoE models) into 100% node-local reads.

#ifndef GGML_EP_SHARD_H
#define GGML_EP_SHARD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

// Returns 1 if expert sharding is enabled (GGML_EP_SHARD=1 in env). Cached.
int ggml_ep_shard_enabled(void);

// Look up (or lazily allocate) the shard buffer for `src0`. Returns a
// pointer to the start of the compact expert-weight region. Index into
// it as `(char*)buf + (cur_a / n_inst) * per_expert_bytes`.
//
// Thread-safe: one allocation per (tensor) across all calling threads;
// concurrent callers spin on a per-tensor seqlock until the first
// allocator publishes the result.
//
// Returns NULL if sharding is not active for this call (e.g. the tensor
// has fewer experts than `n_inst`, or allocation failed). Caller must
// fall back to the un-sharded `src0->data + cur_a * nb02` path.
void * ggml_ep_shard_lookup(const struct ggml_tensor * src0,
                            int my_instance_id,
                            int n_instances);

#ifdef __cplusplus
}
#endif

#endif // GGML_EP_SHARD_H
