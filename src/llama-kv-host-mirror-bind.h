#pragma once

// HSF-5.P (experimental): binds the host KV mirror (llama-kv-host-mirror.h) to a context's memory.
// A llama_context holds one of these only after llama_kv_mirror_enable(); otherwise the pointer is null and
// every hook is a skipped branch.

#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-kv-host-mirror.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct llama_memory_i;
class llama_memory_recurrent;

struct llama_kv_mirror : public llama_kv_cache::write_observer {
    static std::unique_ptr<llama_kv_mirror> create(
            llama_memory_i * mem,
            const std::vector<ggml_backend_t> & backends,
            uint32_t n_seq_max,
            const std::string & spec,
            std::string & err);

    ~llama_kv_mirror() override;

    // llama_kv_cache::write_observer
    void on_cells_written(uint32_t strm, const uint32_t * idxs, size_t n) override;
    void on_invalidate_all() override;

    // llama_context hooks
    void on_decode_begin();                       // poll, then issue due recurrent checkpoints
    void on_decode_end(const llama_batch & batch); // one async KV tail copy after the step's launch
    void on_state_read(llama_seq_id seq_id);      // a state restore replaced seq_id (-1 = all): drop checkpoints

    // a standard llama_state_seq file written from the host mirror (see llama.h)
    // `sync` synchronizes the context; it is called only if some range must be read from the device
    size_t seq_save_file(const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count, size_t * n_token_saved,
                         const std::function<void()> & sync);

    // a standard llama_state_seq file restored through pinned staging + bulk H2D / gather;
    // read_state runs the context's state_seq_read_data on the given reader
    size_t seq_load_file(const char * filepath, llama_seq_id seq_id, llama_token * tokens_out, size_t n_token_capacity,
                         size_t * n_token_count_out, const std::function<size_t(llama_io_read_i &)> & read_state);

    void get_stats(llama_kv_mirror_stats * out) const;

private:
    llama_kv_mirror() = default;

    struct ckpt {
        llama_kvm::captured_stream stream;
        llama_pos pos     = -1;
        uint64_t  ticket  = 0;
        bool      valid   = false; // complete and usable
        bool      pending = false; // issued, copy in flight
    };

    struct seq_ckpts {
        ckpt c[2];
        int  cur = -1; // index of the latest complete checkpoint
    };

    struct inflight {
        uint64_t          ticket;
        llama_kvm::runs_t runs;
        uint32_t          epoch;
    };

    void poll(bool wait);
    void issue_tail();
    bool issue_ckpt(llama_seq_id seq, bool device_wait);
    void finish_ckpts(bool wait, llama_seq_id only = -1);

    llama_kvm::params p;

    llama_memory_i *         mem  = nullptr;
    llama_kv_cache *         kv   = nullptr;
    llama_memory_recurrent * recr = nullptr;

    const ggml_kv_mirror_ops * ops = nullptr;
    ggml_kv_mirror_dev_t       dev = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;

    uint32_t kv_size  = 0;
    uint32_t n_stream = 0;
    uint32_t n_seq    = 0;

    std::vector<llama_kvm::kv_region> regions;
    size_t ckpt_off = 0; // tier offset of the checkpoint slots
    size_t ckpt_cap = 0; // bytes per slot

    llama_kvm::host_tier     tier;
    llama_kvm::host_tier     staging; // restore staging, grown on demand
    llama_kvm::unit_tracker  units;
    llama_kvm::ckpt_schedule sched;

    std::vector<seq_ckpts> ckpts;
    std::vector<inflight>  tail;

    std::vector<ggml_kv_mirror_frag> frags;
    std::vector<int32_t> step_seqs;

    llama_kv_mirror_stats st = {};
};
