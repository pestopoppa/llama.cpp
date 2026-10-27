#pragma once

// HSF-5.P (experimental): device-side operations for the host-pinned KV mirror.
//
// A backend that supports the mirror exposes
//
//     ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_mirror_get_ops")
//
// returning a ggml_backend_kv_mirror_get_ops_t. libllama owns all host-side logic (which cells to
// mirror, the checkpoint schedule, the fragment lists); a backend only pins/maps host memory and moves
// fragment lists on a side stream that is ordered after the backend's own stream.
//
// Contract (the reason this interface exists, see docs/design/hsf-5-host-kv-mirror-costing-20261024.md §4):
//   - copy_async is never issued inside a graph capture and never blocks the host;
//   - the backend's compute stream is never made to wait unless backend_wait is called;
//   - nothing here is called unless the mirror was enabled, so with the mirror OFF no allocation,
//     stream, event or kernel launch happens.

#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_KV_MIRROR_OPS_VERSION 1

enum ggml_kv_mirror_dir {
    GGML_KV_MIRROR_D2H = 0, // device -> registered host memory (mirror write side)
    GGML_KV_MIRROR_H2D = 1, // registered host memory -> device (restore side)
};

enum ggml_kv_mirror_path {
    GGML_KV_MIRROR_PATH_AUTO        = 0, // small fragments -> one gather/scatter kernel, large -> copy engine
    GGML_KV_MIRROR_PATH_COPY_ENGINE = 1, // one async memcpy per fragment
    GGML_KV_MIRROR_PATH_KERNEL      = 2, // all fragments in one gather/scatter kernel launch
};

// one contiguous byte range moved between device memory and registered host memory
struct ggml_kv_mirror_frag {
    void * dev;        // device address (a backend buffer address, e.g. tensor->data + offset)
    void * host;       // host address inside a range passed to host_register
    void * host_alias; // device-side alias of `host` (from host_register), used by the kernel path
    size_t bytes;
};

typedef struct ggml_kv_mirror_dev * ggml_kv_mirror_dev_t;

struct ggml_kv_mirror_ops {
    int version; // GGML_KV_MIRROR_OPS_VERSION

    // create the side stream / event ring for the backend instance whose stream the mirror orders after;
    // returns NULL if the backend is not supported
    ggml_kv_mirror_dev_t (*dev_init)(ggml_backend_t backend);
    void                 (*dev_free)(ggml_kv_mirror_dev_t dev);

    // pin + map one piece of host memory (mapped|portable); returns false if it could not be pinned.
    // *alias_out = the device alias of `host`, or NULL if pinned but not mapped (copy-engine path only)
    bool   (*host_register)  (ggml_kv_mirror_dev_t dev, void * host, size_t bytes, void ** alias_out);
    void   (*host_unregister)(ggml_kv_mirror_dev_t dev, void * host);

    // enqueue `n` fragments on the side stream, ordered after all work already queued on the backend stream;
    // returns a ticket > 0, or 0 if nothing was enqueued (the caller then falls back to a synchronous copy)
    uint64_t (*copy_async)(ggml_kv_mirror_dev_t dev, const struct ggml_kv_mirror_frag * frags, size_t n,
                           enum ggml_kv_mirror_dir dir, enum ggml_kv_mirror_path path);

    bool (*ticket_done) (ggml_kv_mirror_dev_t dev, uint64_t ticket); // non-blocking event query
    void (*ticket_wait) (ggml_kv_mirror_dev_t dev, uint64_t ticket); // host wait: save/restore paths only
    void (*backend_wait)(ggml_kv_mirror_dev_t dev, uint64_t ticket); // device-side: backend stream waits on ticket
};

typedef const struct ggml_kv_mirror_ops * (*ggml_backend_kv_mirror_get_ops_t)(void);

#ifdef __cplusplus
}
#endif
