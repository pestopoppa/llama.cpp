#include "llama-kv-host-mirror-bind.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static constexpr size_t KVM_ALIGN = 4096;

static size_t kvm_align(size_t x) {
    return (x + KVM_ALIGN - 1) / KVM_ALIGN * KVM_ALIGN;
}

std::unique_ptr<llama_kv_mirror> llama_kv_mirror::create(
        llama_memory_i * mem,
        const std::vector<ggml_backend_t> & backends,
        uint32_t n_seq_max,
        const std::string & spec,
        std::string & err) {
    std::unique_ptr<llama_kv_mirror> m(new llama_kv_mirror());

    if (!llama_kvm::parse_params(spec, m->p, err)) {
        return nullptr;
    }

    if (mem == nullptr) {
        err = "context has no memory module";
        return nullptr;
    }

    m->mem = mem;
    if (auto * hy = dynamic_cast<llama_memory_hybrid *>(mem)) {
        m->kv   = hy->get_mem_attn();
        m->recr = hy->get_mem_recr();
    } else if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        m->kv = kv;
    } else if (auto * rc = dynamic_cast<llama_memory_recurrent *>(mem)) {
        m->recr = rc;
    } else {
        err = "unsupported memory type (supported: kv cache, recurrent, hybrid)";
        return nullptr;
    }

    std::vector<ggml_tensor *> kv_tensors;
    if (m->kv) {
        if (m->kv->get_shares_cells()) {
            err = "a KV cache that shares another cache's cells is not supported";
            return nullptr;
        }
        if (m->kv->get_v_trans()) {
            err = "transposed V (no flash attention) is not supported: the mirror needs one row per cell";
            return nullptr;
        }
        m->kv->get_mirror_tensors(kv_tensors);
        m->kv_size  = m->kv->get_size();
        m->n_stream = m->kv->get_n_stream();
    }

    std::vector<ggml_tensor *> rc_tensors;
    size_t ckpt_bytes = 0;
    if (m->recr) {
        const size_t n_rows = (size_t) m->recr->size * (1 + m->recr->n_rs_seq);
        for (const auto * v : { &m->recr->r_l, &m->recr->s_l, &m->recr->p_l }) {
            for (ggml_tensor * t : *v) {
                if (t) {
                    rc_tensors.push_back(t);
                    ckpt_bytes += ggml_nbytes(t) / std::max<size_t>(1, n_rows);
                }
            }
        }
        // one cell per sequence plus alignment slack per tensor range
        ckpt_bytes += rc_tensors.size() * 64 + KVM_ALIGN;
    }

    if (kv_tensors.empty() && rc_tensors.empty()) {
        err = "nothing to mirror";
        return nullptr;
    }

    // every mirrored tensor must live in one buffer type: one device, one side stream
    {
        ggml_tensor * t0 = !kv_tensors.empty() ? kv_tensors[0] : rc_tensors[0];
        if (t0->buffer == nullptr) {
            err = "memory tensors are not allocated";
            return nullptr;
        }
        m->buft = ggml_backend_buffer_get_type(t0->buffer);
        for (const auto * v : { &kv_tensors, &rc_tensors }) {
            for (ggml_tensor * t : *v) {
                if (t->buffer == nullptr || ggml_backend_buffer_get_type(t->buffer) != m->buft) {
                    err = "memory tensors span several buffer types (split KV across devices is not supported)";
                    return nullptr;
                }
            }
        }

        if (ggml_backend_buft_is_host(m->buft)) {
            m->ops = llama_kvm::memcpy_ops();
            m->dev = m->ops->dev_init(nullptr);
        } else {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(m->buft);
            ggml_backend_t backend = nullptr;
            for (ggml_backend_t b : backends) {
                if (ggml_backend_get_device(b) == dev) {
                    backend = b;
                    break;
                }
            }
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            auto get_ops = reg ? (ggml_backend_kv_mirror_get_ops_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_mirror_get_ops") : nullptr;
            if (backend == nullptr || get_ops == nullptr) {
                err = std::string("backend ") + (dev ? ggml_backend_dev_name(dev) : "?") + " has no kv mirror support";
                return nullptr;
            }
            m->ops = get_ops();
            if (m->ops == nullptr || m->ops->version != GGML_KV_MIRROR_OPS_VERSION) {
                err = "kv mirror ops version mismatch";
                return nullptr;
            }
            m->dev = m->ops->dev_init(backend);
            if (m->dev == nullptr) {
                err = "kv mirror device init failed";
                return nullptr;
            }
        }
    }

    // host tier layout: [K/V tensor images ...][checkpoint slots: n_seq x 2]
    size_t off = 0;
    for (ggml_tensor * t : kv_tensors) {
        if (!ggml_is_contiguous(t) || t->ne[1] != (int64_t) m->kv_size || t->ne[2] != (int64_t) m->n_stream) {
            err = "unexpected KV tensor layout";
            return nullptr;
        }
        llama_kvm::kv_region r;
        r.dev       = (uint8_t *) t->data;
        r.bytes     = ggml_nbytes(t);
        r.row_bytes = t->nb[1];
        r.host_off  = off;
        m->regions.push_back(r);
        off = kvm_align(off + r.bytes);
    }

    m->n_seq = std::max<uint32_t>(1, n_seq_max);
    m->ckpt_off = off;
    m->ckpt_cap = m->recr ? kvm_align(ckpt_bytes) : 0;
    off += (size_t) m->n_seq * 2 * m->ckpt_cap;

    if (!m->tier.reserve(off, m->p.numa_node, m->p.piece_bytes, err)) {
        return nullptr;
    }
    const int64_t t0 = ggml_time_us();
    const bool all_reg = m->tier.register_with(m->ops, m->dev);
    LLAMA_LOG_INFO("%s: host tier %.1f MiB (KV %.1f MiB, checkpoints %u x 2 x %.1f MiB) on node %d, %zu pieces pinned+mapped in %.1f s%s\n",
            __func__, off/1024.0/1024.0, m->ckpt_off/1024.0/1024.0, m->n_seq, m->ckpt_cap/1024.0/1024.0, m->p.numa_node,
            m->tier.n_registered_pieces(), (ggml_time_us() - t0)/1e6, all_reg ? "" : " (PARTIAL: the rest falls back to device reads)");

    m->units.init(m->n_stream * m->kv_size);
    m->sched.init(m->p.k, m->n_seq);
    m->ckpts.resize(m->n_seq);

    // a cache that already holds sequences is copied once in full by the first decode
    for (uint32_t s = 0; s < m->n_seq; ++s) {
        if (mem->seq_pos_max((llama_seq_id) s) >= 0) {
            m->units.mark_all_written();
            break;
        }
    }

    if (m->kv) {
        m->kv->set_write_observer(m.get());
    }

    m->st.host_bytes = off;
    m->st.host_registered_pieces = m->tier.n_registered_pieces();

    LLAMA_LOG_INFO("%s: HSF-5.P kv host mirror ON: %s, %zu KV tensors, %zu recurrent tensors\n",
            __func__, llama_kvm::to_string(m->p).c_str(), kv_tensors.size(), rc_tensors.size());

    return m;
}

llama_kv_mirror::~llama_kv_mirror() {
    if (kv) {
        kv->set_write_observer(nullptr);
    }
    if (ops && dev) {
        for (const auto & t : tail) {
            ops->ticket_wait(dev, t.ticket);
        }
        for (auto & sc : ckpts) {
            for (auto & c : sc.c) {
                if (c.pending) {
                    ops->ticket_wait(dev, c.ticket);
                }
            }
        }
    }
    tier.release();
    staging.release();
    if (ops && dev) {
        ops->dev_free(dev);
    }
}

void llama_kv_mirror::on_cells_written(uint32_t strm, const uint32_t * idxs, size_t n) {
    const uint32_t base = strm * kv_size;
    for (size_t i = 0; i < n; ++i) {
        units.mark_written(base + idxs[i]);
    }
}

void llama_kv_mirror::on_invalidate_all() {
    units.mark_all_written();
}

void llama_kv_mirror::poll(bool wait) {
    size_t w = 0;
    for (size_t i = 0; i < tail.size(); ++i) {
        auto & t = tail[i];
        if (wait) {
            ops->ticket_wait(dev, t.ticket);
        }
        if (wait || ops->ticket_done(dev, t.ticket)) {
            units.on_complete(t.runs, t.epoch);
        } else {
            if (w != i) {
                tail[w] = std::move(t); // never self-move: a self-moved vector may come back empty
            }
            w++;
        }
    }
    tail.resize(w);
    finish_ckpts(wait);
}

void llama_kv_mirror::finish_ckpts(bool wait, llama_seq_id only) {
    for (uint32_t s = 0; s < n_seq; ++s) {
        if (only >= 0 && (llama_seq_id) s != only) {
            continue;
        }
        auto & sc = ckpts[s];
        for (int i = 0; i < 2; ++i) {
            auto & c = sc.c[i];
            if (!c.pending) {
                continue;
            }
            if (wait) {
                ops->ticket_wait(dev, c.ticket);
            }
            if (wait || ops->ticket_done(dev, c.ticket)) {
                c.pending = false;
                c.valid   = true;
                sc.cur    = i;
                sc.c[1 - i].valid = false; // the older one is now the next write target
                st.n_ckpt_complete++;
            }
        }
    }
}

bool llama_kv_mirror::issue_ckpt(llama_seq_id seq, bool device_wait) {
    if (!recr || ckpt_cap == 0) {
        return false;
    }
    auto & sc = ckpts[seq];
    const int tgt = sc.cur == 0 ? 1 : 0;
    auto & c = sc.c[tgt];
    if (c.pending || sc.c[1 - tgt].pending) {
        return false; // one checkpoint per sequence in flight
    }

    c.stream.clear();
    c.valid = false;
    frags.clear();

    const size_t slot_off = ckpt_off + ((size_t) seq * 2 + tgt) * ckpt_cap;
    llama_kvm::io_capture io(c.stream, tier, slot_off, ckpt_cap, frags);
    try {
        recr->state_write(io, seq, 0);
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: checkpoint of seq %d skipped: %s\n", __func__, seq, e.what());
        c.stream.clear();
        return false;
    }

    c.pos = recr->seq_pos_max(seq); // the position the recurrent state corresponds to
    if (frags.empty()) {
        // empty sequence: the literal metadata alone is the checkpoint
        c.ticket  = 0;
        c.pending = false;
        c.valid   = true;
        sc.cur    = tgt;
        sc.c[1 - tgt].valid = false;
        sched.on_issued(seq);
        return true;
    }

    llama_kvm::coalesce_frags(tier, frags);
    const uint64_t t = ops->copy_async(dev, frags.data(), frags.size(), GGML_KV_MIRROR_D2H, p.path);
    if (t == 0) {
        c.stream.clear();
        return false;
    }
    if (device_wait) {
        // design A': the next step rewrites this state in place, so it waits for the snapshot on the device
        ops->backend_wait(dev, t);
    }
    c.ticket  = t;
    c.pending = true;
    sched.on_issued(seq);

    st.n_ckpt_issued++;
    for (const auto & f : frags) {
        st.ckpt_bytes += f.bytes;
    }
    return true;
}

void llama_kv_mirror::on_decode_begin() {
    poll(false);

    if (!recr) {
        return;
    }

    // a sequence truncated below its checkpoint (new task, LCP trim) or emptied loses the checkpoint
    for (uint32_t s = 0; s < n_seq; ++s) {
        auto & sc = ckpts[s];
        if (sc.cur < 0 && !sc.c[0].pending && !sc.c[1].pending) {
            continue;
        }
        const llama_pos pmax = mem->seq_pos_max((llama_seq_id) s);
        for (auto & c : sc.c) {
            if ((c.valid || c.pending) && pmax < c.pos) {
                if (c.pending) {
                    ops->ticket_wait(dev, c.ticket); // not reachable in steady decode: truncation happens between tasks
                    c.pending = false;
                }
                c.valid = false;
            }
        }
        if (sc.cur >= 0 && !sc.c[sc.cur].valid) {
            sc.cur = -1;
            sched.on_reset((llama_seq_id) s);
        }
    }

    const size_t max_n = (n_seq + p.k - 1) / p.k;
    for (const int32_t s : sched.due(max_n)) {
        issue_ckpt(s, true);
    }
}

void llama_kv_mirror::on_decode_end(const llama_batch & batch) {
    st.n_steps++;

    step_seqs.clear();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int ns = batch.n_seq_id ? batch.n_seq_id[i] : 1;
        for (int j = 0; j < ns; ++j) {
            const llama_seq_id s = batch.seq_id ? batch.seq_id[i][j] : 0;
            if (s >= 0 && (uint32_t) s < n_seq) {
                step_seqs.push_back(s);
            }
        }
    }
    std::sort(step_seqs.begin(), step_seqs.end());
    step_seqs.erase(std::unique(step_seqs.begin(), step_seqs.end()), step_seqs.end());
    sched.on_step(step_seqs);

    issue_tail();
}

void llama_kv_mirror::issue_tail() {
    if (!kv || !units.has_pending()) {
        return;
    }

    uint32_t e = 0;
    llama_kvm::runs_t runs = units.take_pending(&e);

    frags.clear();
    llama_kvm::build_tail_frags(regions, runs, tier, frags);
    llama_kvm::coalesce_frags(tier, frags);

    // after the step's launch: the event inside copy_async orders the side stream after it; no host wait
    const uint64_t t = ops->copy_async(dev, frags.data(), frags.size(), GGML_KV_MIRROR_D2H, p.path);
    if (t == 0) {
        return; // the cells stay invalid in the mirror; a save reads them from the device
    }
    st.n_tail_copies++;
    st.n_tail_frags += frags.size();
    for (const auto & f : frags) {
        st.tail_bytes += f.bytes;
    }
    tail.push_back({ t, std::move(runs), e });
}

void llama_kv_mirror::on_state_read(llama_seq_id seq_id) {
    finish_ckpts(true);
    for (uint32_t s = 0; s < n_seq; ++s) {
        if (seq_id >= 0 && (llama_seq_id) s != seq_id) {
            continue;
        }
        auto & sc = ckpts[s];
        sc.c[0].valid = false;
        sc.c[1].valid = false;
        sc.cur = -1;
        sched.on_reset((llama_seq_id) s);
    }
}

size_t llama_kv_mirror::seq_save_file(const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count, size_t * n_token_saved,
                                      const std::function<void()> & sync) {
#if defined(__linux__)
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq) {
        return 0;
    }

    // save-time waits are allowed (this is the handover path, not post_decode): the previous step's tail copy
    // (tens of microseconds) and this sequence's in-flight checkpoint
    poll(true);

    // A completed lag checkpoint still predates the live recurrent state. Slot files
    // must describe the same boundary as their tokens and draft/spec appendix.
    if (recr) {
        if (!issue_ckpt(seq_id, true)) {
            return 0;
        }
        finish_ckpts(true, seq_id);
    }
    // Restores and cache shifts can dirty cells without another decode.
    issue_tail();
    poll(true);
    st.last_replay_tokens = 0;
    st.last_replay_steps  = 0;

    const int fd = ::open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LLAMA_LOG_ERROR("%s: cannot open %s: %s\n", __func__, filepath, strerror(errno));
        return 0;
    }

    size_t total = 0;
    try {
        llama_kvm::io_sink_fd sink(fd);

        const uint32_t hdr[3] = { LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, (uint32_t) n_token_count };
        sink.write(hdr, sizeof(hdr));
        sink.write(tokens, sizeof(llama_token) * n_token_count);

        if (kv) {
            bool synced = false;
            llama_kvm::io_from_mirror io(sink, regions, units, tier,
                    [&](ggml_tensor * t, void * dst, size_t off, size_t size) {
                        if (!synced) {
                            sync(); // only a range the mirror cannot serve touches the device
                            synced = true;
                        }
                        ggml_backend_tensor_get(t, dst, off, size);
                    });
            kv->state_write(io, seq_id, 0);
            st.save_mirror_bytes   += io.bytes_from_mirror;
            st.save_fallback_bytes += io.bytes_fallback;
        }
        if (recr) {
            const auto & sc = ckpts[seq_id];
            sc.c[sc.cur].stream.write_to(sink, tier.data());
            st.save_mirror_bytes += sc.c[sc.cur].stream.data_used;
        }
        total = sink.n_bytes();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        ::close(fd);
        return 0;
    }
    ::close(fd);

    st.n_saves++;
    if (n_token_saved) {
        *n_token_saved = n_token_count;
    }
    return total;
#else
    GGML_UNUSED(filepath); GGML_UNUSED(seq_id); GGML_UNUSED(tokens); GGML_UNUSED(n_token_count); GGML_UNUSED(n_token_saved); GGML_UNUSED(sync);
    return 0;
#endif
}

size_t llama_kv_mirror::seq_load_file(const char * filepath, llama_seq_id seq_id, llama_token * tokens_out, size_t n_token_capacity,
                                      size_t * n_token_count_out, const std::function<size_t(llama_io_read_i &)> & read_state) {
#if defined(__linux__)
    GGML_UNUSED(seq_id); // the reader is bound to seq_id by the caller
    const int fd = ::open(filepath, O_RDONLY);
    if (fd < 0) {
        LLAMA_LOG_ERROR("%s: cannot open %s: %s\n", __func__, filepath, strerror(errno));
        return 0;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < (off_t) (3*sizeof(uint32_t))) {
        ::close(fd);
        return 0;
    }
    const size_t size = (size_t) sb.st_size;

    if (staging.bytes() < size) {
        std::string err;
        const size_t want = kvm_align(std::max(size, staging.bytes() * 2));
        if (!staging.reserve(want, p.numa_node, p.piece_bytes, err)) {
            LLAMA_LOG_ERROR("%s: restore staging: %s\n", __func__, err.c_str());
            ::close(fd);
            return 0;
        }
        staging.register_with(ops, dev);
    }

    size_t got = 0;
    while (got < size) {
        const ssize_t r = ::pread(fd, staging.data() + got, size - got, (off_t) got);
        if (r < 0 && errno == EINTR) {
            continue;
        }
        if (r <= 0) {
            ::close(fd);
            return 0;
        }
        got += (size_t) r;
    }
    ::close(fd);

    const uint8_t * buf = staging.data();
    uint32_t hdr[3];
    memcpy(hdr, buf, sizeof(hdr));
    if (hdr[0] != LLAMA_STATE_SEQ_MAGIC || hdr[1] != LLAMA_STATE_SEQ_VERSION) {
        LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, hdr[0], hdr[1]);
        return 0;
    }
    const size_t n_tok = hdr[2];
    const size_t off = sizeof(hdr) + n_tok * sizeof(llama_token);
    if (n_tok > n_token_capacity || off > size) {
        LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %zu > %zu\n", __func__, n_tok, n_token_capacity);
        return 0;
    }
    memcpy(tokens_out, buf + sizeof(hdr), n_tok * sizeof(llama_token));

    const auto mbuft = buft;
    size_t nread = 0;
    {
        llama_kvm::io_to_device io(buf + off, size - off, &staging, ops, dev, p.path,
                [mbuft](const ggml_tensor * t) { return t->buffer != nullptr && ggml_backend_buffer_get_type(t->buffer) == mbuft; },
                [](ggml_tensor * t, const void * src, size_t o, size_t n) { ggml_backend_tensor_set(t, src, o, n); });
        try {
            nread = read_state(io);
            io.flush();
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
            return 0;
        }
        st.restore_device_bytes   += io.bytes_device;
        st.restore_device_frags   += io.n_frags_device;
        st.restore_fallback_bytes += io.bytes_fallback;
    }
    if (nread == 0) {
        return 0;
    }

    st.n_restores++;
    *n_token_count_out = n_tok;
    return off + nread;
#else
    GGML_UNUSED(filepath); GGML_UNUSED(seq_id); GGML_UNUSED(tokens_out); GGML_UNUSED(n_token_capacity);
    GGML_UNUSED(n_token_count_out); GGML_UNUSED(read_state);
    return 0;
#endif
}

void llama_kv_mirror::get_stats(llama_kv_mirror_stats * out) const {
    *out = st;
}

//
// llama_context members and the C API
//

bool llama_context::kv_mirror_enable(const char * spec) {
    if (kv_mirror) {
        LLAMA_LOG_WARN("%s: kv host mirror already enabled\n", __func__);
        return true;
    }
    synchronize();

    std::vector<ggml_backend_t> bks;
    for (auto & b : backends) {
        bks.push_back(b.get());
    }
    std::string err;
    kv_mirror = llama_kv_mirror::create(memory.get(), bks, cparams.n_seq_max, spec ? spec : "", err);
    if (!kv_mirror) {
        LLAMA_LOG_ERROR("%s: kv host mirror not enabled: %s\n", __func__, err.c_str());
        return false;
    }
    return true;
}

size_t llama_context::kv_mirror_seq_load_file(const char * filepath, llama_seq_id seq_id, llama_token * tokens_out,
                                              size_t n_token_capacity, size_t * n_token_count_out) {
    if (!kv_mirror) {
        return 0;
    }
    return kv_mirror->seq_load_file(filepath, seq_id, tokens_out, n_token_capacity, n_token_count_out,
            [this, seq_id](llama_io_read_i & io) { return state_seq_read_data(io, seq_id, 0); });
}

bool llama_kv_mirror_enable(llama_context * ctx, const char * spec) {
    return ctx->kv_mirror_enable(spec);
}

bool llama_kv_mirror_is_enabled(const llama_context * ctx) {
    return ctx->get_kv_mirror() != nullptr;
}

size_t llama_kv_mirror_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id,
                                     const llama_token * tokens, size_t n_token_count, size_t * n_token_saved) {
    auto * m = ctx->get_kv_mirror();
    if (!m) {
        return 0;
    }
    try {
        return m->seq_save_file(filepath, seq_id, tokens, n_token_count, n_token_saved, [ctx]() { ctx->synchronize(); });
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return 0;
    }
}

size_t llama_kv_mirror_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id,
                                     llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();
    try {
        return ctx->kv_mirror_seq_load_file(filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return 0;
    }
}

void llama_kv_mirror_get_stats(const llama_context * ctx, llama_kv_mirror_stats * out) {
    if (!out) {
        return;
    }
    *out = {};
    if (auto * m = ctx->get_kv_mirror()) {
        m->get_stats(out);
    }
}
