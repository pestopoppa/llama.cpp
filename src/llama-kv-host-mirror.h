#pragma once

// HSF-5.P (experimental): host-pinned KV mirror, host-side logic.
//
// Everything in this header is plain C++ over byte ranges and a device-ops table (ggml-kv-mirror.h): it does not
// touch libllama internals, so tests/test-kv-host-mirror.cpp exercises it on the CPU with a memcpy "device".
// The libllama binding (llama-kv-host-mirror-bind.cpp) feeds it from the KV cache / recurrent memory.
//
// Design (docs/design/hsf-5-host-kv-mirror-costing-20261024.md in epyc-root, "Proposed prototype task text"):
//   (a) host tier: one anonymous mapping, NUMA-bound (mbind) before first touch, pinned + mapped in pieces;
//       the KV part is a byte image of every attention K/V tensor (same offsets), the recurrent part holds
//       two checkpoint slots per sequence (ping-pong, so a completed checkpoint is never overwritten in flight);
//   (b) KV tail write-through: cells written by a decode step are coalesced into fragments and moved by ONE
//       copy_async after the step's launch (side stream, ordered by an event; never inside a capture, no host
//       wait). Validity is tracked per cell with generations, so a cell rewritten while its copy was in flight
//       is never served from the mirror;
//   (c) recurrent state: a checkpoint of each sequence every k decode steps, staggered across sequences; a
//       saves advance the checkpoint and flush KV to the live position;
//   (d) restore: a standard slot stream staged in pinned memory and moved by bulk H2D / the gather kernel
//       (fragmented cells) instead of one blocking tensor_set per range.

#include "ggml-kv-mirror.h"
#include "llama-io.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct ggml_tensor;

namespace llama_kvm {

using run_t  = std::pair<uint32_t, uint32_t>; // [first, last) unit range
using runs_t = std::vector<run_t>;

enum class save_mode {
    lag,   // legacy spelling; saves now advance to the live boundary too
    fresh, // flush the tail and take a live checkpoint at save
};

struct params {
    uint32_t            k           = 6;            // recurrent checkpoint period in decode steps per sequence
    int                 numa_node   = 3;            // membind node for the host tier, -1 = no binding
    save_mode           mode        = save_mode::lag;
    size_t              piece_bytes = size_t(1) << 30; // host registration piece (mori UMBP uses 1 GiB)
    ggml_kv_mirror_path path        = GGML_KV_MIRROR_PATH_AUTO;
};

// spec grammar: comma-separated key=value, keys k, node, mode (lag|fresh), piece_mib, path (auto|ce|kernel);
// "on" or "" = all defaults
bool parse_params(const std::string & spec, params & out, std::string & err);

std::string to_string(const params & p);

//
// (c) checkpoint schedule
//

class ckpt_schedule {
public:
    void init(uint32_t k, uint32_t n_seq);

    // one decode step happened that included these sequences
    void on_step(const std::vector<int32_t> & seqs);

    // sequences due a checkpoint at this boundary, most overdue first, at most `max_n`
    std::vector<int32_t> due(size_t max_n) const;

    void on_issued(int32_t seq);  // the checkpoint snapshot is taken at issue time
    void on_reset (int32_t seq);  // the sequence was truncated below its checkpoint or replaced

    bool     has_ckpt   (int32_t seq) const { return has[seq]; }
    uint32_t steps_since(int32_t seq) const { return since[seq]; }
    uint32_t period() const { return k; }

    // first checkpoint of seq s is due after k - phase(s) steps, so n_seq <= k sequences never share a step
    uint32_t phase(int32_t seq) const { return ph[seq]; }

private:
    uint32_t k = 6;
    std::vector<uint32_t> since;
    std::vector<uint32_t> ph;
    std::vector<bool>     has;
};

// replay needed at restore after a lag save: tokens and decode steps since the checkpoint
struct replay_bound {
    int32_t  n_tokens = 0;
    uint32_t n_steps  = 0;
    bool     ok       = false; // n_steps <= k
};

replay_bound compute_replay(int32_t pos_max, int32_t ckpt_pos, uint32_t steps_since, uint32_t k);

//
// (b) per-unit validity (a unit = one KV cell index across all streams, i.e. one row in every K/V tensor)
//

class unit_tracker {
public:
    void init(uint32_t n_units);

    uint32_t size()  const { return (uint32_t) written.size(); }
    uint32_t epoch() const { return cur; }

    void mark_written(uint32_t u);
    void mark_all_written();

    // pending units as sorted, coalesced runs; clears the pending set. The copy that carries them reflects
    // device content as of the end of the current epoch, which is returned in *epoch_out; the epoch advances.
    runs_t take_pending(uint32_t * epoch_out);

    bool has_pending() const { return full || !pending.empty(); }

    // the copy of epoch e for these runs completed
    void on_complete(const runs_t & runs, uint32_t e);

    bool valid(uint32_t u0, uint32_t u1) const;

private:
    uint32_t cur = 1;
    bool full = false;
    std::vector<uint32_t> written;
    std::vector<uint32_t> mirrored;
    std::vector<uint32_t> pending;
};

// sort + unique + merge adjacent unit indices into runs
runs_t coalesce_units(std::vector<uint32_t> units);

//
// (a) host tier
//

class host_tier {
public:
    host_tier() = default;
    ~host_tier();
    host_tier(const host_tier &) = delete;
    host_tier & operator=(const host_tier &) = delete;

    // reserve `bytes` of anonymous memory, mbind it to `numa_node` (if >= 0) before first touch; no pinning yet
    bool reserve(size_t bytes, int numa_node, size_t piece_bytes, std::string & err);

    // pin + map in pieces with the device ops; a failed piece leaves the rest pageable (copies fall back)
    bool register_with(const ggml_kv_mirror_ops * ops, ggml_kv_mirror_dev_t dev);

    void release();

    uint8_t * data()        { return base; }
    const uint8_t * data() const { return base; }
    size_t    bytes() const { return n_bytes; }
    size_t    piece() const { return piece_bytes; }
    size_t    n_registered_pieces() const { return n_reg; }

    bool  contains  (const void * p, size_t n) const;
    bool  registered(const void * p, size_t n) const; // whole range inside registered pieces
    void * alias_of (const void * p) const;            // device alias, nullptr if unmapped

    // for tests: adopt an external buffer as the tier (not owned), with an alias table
    void adopt(uint8_t * p, size_t n, size_t piece, const std::vector<void *> & aliases);

private:
    uint8_t * base = nullptr;
    size_t    n_bytes = 0;
    size_t    piece_bytes = size_t(1) << 30;
    bool      owned = false;
    size_t    n_reg = 0;
    std::vector<void *> alias;

    const ggml_kv_mirror_ops * reg_ops = nullptr;
    ggml_kv_mirror_dev_t       reg_dev = nullptr;
};

// append a fragment for [dev, dev+bytes) <-> [host, host+bytes), split at host piece boundaries
void append_frag(const host_tier & tier, uint8_t * dev, uint8_t * host, size_t bytes, std::vector<ggml_kv_mirror_frag> & out);

// merge fragments that are contiguous on both sides and share a mapping piece (in place, order preserving)
void coalesce_frags(const host_tier & tier, std::vector<ggml_kv_mirror_frag> & frags);

// a mirrored KV tensor: rows are indexed by unit; host image at tier offset `host_off`
struct kv_region {
    uint8_t * dev       = nullptr;
    size_t    bytes     = 0;
    size_t    row_bytes = 0;
    size_t    host_off  = 0;
};

// fragments for the given unit runs across all regions
void build_tail_frags(const std::vector<kv_region> & regions, const runs_t & runs, host_tier & tier,
                      std::vector<ggml_kv_mirror_frag> & out);

//
// state-stream capture: a slot stream as literal bytes + tensor byte ranges held in the host tier
//

struct seg {
    bool   lit;  // true: bytes in `lit`, false: bytes in the host tier
    size_t off;
    size_t len;
};

struct captured_stream {
    std::vector<uint8_t> lit;
    std::vector<seg>     segs;
    size_t               n_bytes   = 0;
    size_t               data_used = 0; // bytes used in the slot

    void clear() { lit.clear(); segs.clear(); n_bytes = 0; data_used = 0; }

    // replay the stream into a writer, reading tensor bytes from the tier
    void write_to(llama_io_write_i & io, const uint8_t * tier_base) const;
};

// records metadata literally and turns write_tensor into fragments targeting [slot, slot + slot_cap)
class io_capture : public llama_io_write_i {
public:
    io_capture(captured_stream & out, host_tier & tier, size_t slot_off, size_t slot_cap,
               std::vector<ggml_kv_mirror_frag> & frags);

    void   write(const void * src, size_t size) override;
    void   write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;
    size_t n_bytes() override { return out.n_bytes; }

private:
    captured_stream & out;
    host_tier & tier;
    size_t slot_off;
    size_t slot_cap;
    std::vector<ggml_kv_mirror_frag> & frags;
};

using tensor_get_fn = std::function<void(ggml_tensor * t, void * dst, size_t offset, size_t size)>;
using tensor_set_fn = std::function<void(ggml_tensor * t, const void * src, size_t offset, size_t size)>;

// serves write_tensor for mirrored KV tensors from the host image when the cells are valid, else falls back
class io_from_mirror : public llama_io_write_i {
public:
    io_from_mirror(llama_io_write_i & sink, const std::vector<kv_region> & regions, const unit_tracker & units,
                   const host_tier & tier, tensor_get_fn fallback);

    void   write(const void * src, size_t size) override;
    void   write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;
    size_t n_bytes() override { return sink.n_bytes(); }

    size_t bytes_from_mirror = 0;
    size_t bytes_fallback    = 0;

private:
    llama_io_write_i & sink;
    const std::vector<kv_region> & regions;
    const unit_tracker & units;
    const host_tier & tier;
    tensor_get_fn fallback;
    std::vector<uint8_t> tmp;
};

// reads a slot stream from (pinned) host memory; tensor reads become fragments flushed by one copy_async
class io_to_device : public llama_io_read_i {
public:
    io_to_device(const uint8_t * src, size_t size, const host_tier * tier,
                 const ggml_kv_mirror_ops * ops, ggml_kv_mirror_dev_t dev, ggml_kv_mirror_path path,
                 std::function<bool(const ggml_tensor *)> on_device, tensor_set_fn fallback);
    ~io_to_device() override;

    void   read(void * dst, size_t size) override;
    void   read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;
    size_t n_bytes() override { return n_read; }

    // move every recorded tensor range now (also done by the destructor); returns false on a failed copy
    bool flush();

    size_t n_frags_device   = 0;
    size_t bytes_device     = 0;
    size_t bytes_fallback   = 0;

private:
    const uint8_t * ptr;
    size_t left;
    size_t n_read = 0;
    const host_tier * tier;
    const ggml_kv_mirror_ops * ops;
    ggml_kv_mirror_dev_t dev;
    ggml_kv_mirror_path path;
    std::function<bool(const ggml_tensor *)> on_device;
    tensor_set_fn fallback;

    struct pend { ggml_tensor * t; const uint8_t * src; size_t off; size_t size; };
    std::vector<pend> pending;
};

// simple sinks
class io_sink_buffer : public llama_io_write_i {
public:
    std::vector<uint8_t> buf;
    void   write(const void * src, size_t size) override;
    void   write_tensor(ggml_tensor *, size_t, size_t) override; // not supported: throws
    size_t n_bytes() override { return buf.size(); }
};

class io_sink_fd : public llama_io_write_i {
public:
    explicit io_sink_fd(int fd) : fd(fd) {}
    void   write(const void * src, size_t size) override;
    void   write_tensor(ggml_tensor *, size_t, size_t) override; // not supported: throws
    size_t n_bytes() override { return n; }
private:
    int fd;
    size_t n = 0;
};

// synchronous host "device": memcpy, every ticket complete on return. Used for host-resident KV buffers
// (CPU backend) and by the unit tests.
const ggml_kv_mirror_ops * memcpy_ops();

} // namespace llama_kvm
