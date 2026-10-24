#include "llama-kv-host-mirror.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace llama_kvm {

//
// params
//

static bool parse_u64(const std::string & v, uint64_t & out) {
    if (v.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long x = std::strtoull(v.c_str(), &end, 10);
    if (errno != 0 || end == v.c_str() || *end != '\0') {
        return false;
    }
    out = x;
    return true;
}

bool parse_params(const std::string & spec, params & out, std::string & err) {
    params p;
    if (spec.empty() || spec == "on" || spec == "1") {
        out = p;
        return true;
    }

    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        const auto eq = item.find('=');
        if (eq == std::string::npos) {
            err = "expected key=value, got '" + item + "'";
            return false;
        }
        const std::string key = item.substr(0, eq);
        const std::string val = item.substr(eq + 1);
        uint64_t x = 0;
        if (key == "k") {
            if (!parse_u64(val, x) || x == 0 || x > 1000000) { err = "k must be a positive integer"; return false; }
            p.k = (uint32_t) x;
        } else if (key == "node") {
            if (val == "-1" || val == "none") {
                p.numa_node = -1;
            } else {
                if (!parse_u64(val, x) || x > 1023) { err = "node must be -1 or a NUMA node id"; return false; }
                p.numa_node = (int) x;
            }
        } else if (key == "mode") {
            if (val == "lag") {
                p.mode = save_mode::lag;
            } else if (val == "fresh") {
                p.mode = save_mode::fresh;
            } else {
                err = "mode must be lag or fresh"; return false;
            }
        } else if (key == "piece_mib") {
            if (!parse_u64(val, x) || x < 2 || x > (1u << 20)) { err = "piece_mib must be in [2, 1048576]"; return false; }
            p.piece_bytes = (size_t) x << 20;
        } else if (key == "path") {
            if (val == "auto") {
                p.path = GGML_KV_MIRROR_PATH_AUTO;
            } else if (val == "ce") {
                p.path = GGML_KV_MIRROR_PATH_COPY_ENGINE;
            } else if (val == "kernel") {
                p.path = GGML_KV_MIRROR_PATH_KERNEL;
            } else {
                err = "path must be auto, ce or kernel"; return false;
            }
        } else {
            err = "unknown key '" + key + "'";
            return false;
        }
    }

    out = p;
    return true;
}

std::string to_string(const params & p) {
    std::stringstream ss;
    ss << "k=" << p.k << ",node=" << p.numa_node << ",mode=" << (p.mode == save_mode::lag ? "lag" : "fresh")
       << ",piece_mib=" << (p.piece_bytes >> 20) << ",path="
       << (p.path == GGML_KV_MIRROR_PATH_AUTO ? "auto" : p.path == GGML_KV_MIRROR_PATH_COPY_ENGINE ? "ce" : "kernel");
    return ss.str();
}

//
// ckpt_schedule
//

void ckpt_schedule::init(uint32_t k_, uint32_t n_seq) {
    k = std::max<uint32_t>(1, k_);
    since.assign(n_seq, 0);
    has.assign(n_seq, false);
    ph.resize(n_seq);
    for (uint32_t s = 0; s < n_seq; ++s) {
        // spread the first checkpoints over the period: with n_seq <= k every sequence gets its own step residue
        ph[s] = n_seq == 0 ? 0 : (uint32_t) (((uint64_t) s * k) / n_seq) % k;
    }
}

void ckpt_schedule::on_step(const std::vector<int32_t> & seqs) {
    for (const int32_t s : seqs) {
        if (s >= 0 && (size_t) s < since.size()) {
            since[s]++;
        }
    }
}

std::vector<int32_t> ckpt_schedule::due(size_t max_n) const {
    std::vector<std::pair<int64_t, int32_t>> cand; // (overdue, seq)
    for (size_t s = 0; s < since.size(); ++s) {
        if (since[s] == 0) {
            continue; // nothing new since the last checkpoint (or the sequence never decoded)
        }
        const int64_t threshold = has[s] ? (int64_t) k : (int64_t) k - (int64_t) ph[s];
        const int64_t over = (int64_t) since[s] - threshold;
        if (over >= 0) {
            cand.emplace_back(over, (int32_t) s);
        }
    }
    std::stable_sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    std::vector<int32_t> res;
    for (const auto & c : cand) {
        if (res.size() >= max_n) {
            break;
        }
        res.push_back(c.second);
    }
    return res;
}

void ckpt_schedule::on_issued(int32_t seq) {
    since[seq] = 0;
    has[seq]   = true;
}

void ckpt_schedule::on_reset(int32_t seq) {
    since[seq] = 0;
    has[seq]   = false;
}

replay_bound compute_replay(int32_t pos_max, int32_t ckpt_pos, uint32_t steps_since, uint32_t k) {
    replay_bound r;
    if (ckpt_pos < 0 || pos_max < ckpt_pos) {
        return r;
    }
    r.n_tokens = pos_max - ckpt_pos;
    r.n_steps  = steps_since;
    r.ok       = steps_since <= k;
    return r;
}

//
// unit_tracker
//

void unit_tracker::init(uint32_t n_units) {
    cur = 1;
    full = false;
    written.assign(n_units, 0);
    mirrored.assign(n_units, 0);
    pending.clear();
}

void unit_tracker::mark_written(uint32_t u) {
    if (u >= written.size()) {
        return;
    }
    written[u] = cur;
    pending.push_back(u);
}

void unit_tracker::mark_all_written() {
    std::fill(written.begin(), written.end(), cur);
    pending.clear();
    full = true;
}

runs_t unit_tracker::take_pending(uint32_t * epoch_out) {
    runs_t runs;
    if (full) {
        if (!written.empty()) {
            runs.emplace_back(0, (uint32_t) written.size());
        }
        full = false;
        pending.clear();
    } else {
        runs = coalesce_units(std::move(pending));
        pending.clear();
    }
    *epoch_out = cur;
    cur++;
    return runs;
}

void unit_tracker::on_complete(const runs_t & runs, uint32_t e) {
    for (const auto & r : runs) {
        for (uint32_t u = r.first; u < r.second && u < written.size(); ++u) {
            // a cell written after the copy was issued (written > e) keeps its old mirror generation: invalid
            if (written[u] <= e && mirrored[u] < e) {
                mirrored[u] = e;
            }
        }
    }
}

bool unit_tracker::valid(uint32_t u0, uint32_t u1) const {
    if (u1 > written.size() || u0 > u1) {
        return false;
    }
    for (uint32_t u = u0; u < u1; ++u) {
        if (mirrored[u] < written[u]) {
            return false;
        }
    }
    return true;
}

runs_t coalesce_units(std::vector<uint32_t> units) {
    runs_t runs;
    if (units.empty()) {
        return runs;
    }
    std::sort(units.begin(), units.end());
    units.erase(std::unique(units.begin(), units.end()), units.end());
    uint32_t b = units[0];
    uint32_t e = units[0] + 1;
    for (size_t i = 1; i < units.size(); ++i) {
        if (units[i] == e) {
            e++;
        } else {
            runs.emplace_back(b, e);
            b = units[i];
            e = units[i] + 1;
        }
    }
    runs.emplace_back(b, e);
    return runs;
}

//
// host_tier
//

host_tier::~host_tier() {
    release();
}

bool host_tier::reserve(size_t bytes, int numa_node, size_t piece, std::string & err) {
    release();
    if (bytes == 0) {
        err = "empty host tier";
        return false;
    }
    piece_bytes = std::max<size_t>(piece, 2u << 20);
#if defined(__linux__)
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        err = std::string("mmap failed: ") + strerror(errno);
        return false;
    }
#ifdef MADV_HUGEPAGE
    (void) madvise(p, bytes, MADV_HUGEPAGE);
#endif
    if (numa_node >= 0) {
#ifdef SYS_mbind
        // MPOL_BIND = 2; the policy applies at first touch, i.e. when the pieces are pinned
        unsigned long mask[16] = {0};
        if (numa_node >= (int) (sizeof(mask)*8)) {
            munmap(p, bytes);
            err = "numa node out of range";
            return false;
        }
        mask[numa_node / (8*sizeof(unsigned long))] |= 1ul << (numa_node % (8*sizeof(unsigned long)));
        if (syscall(SYS_mbind, p, bytes, 2 /* MPOL_BIND */, mask, (unsigned long) (sizeof(mask)*8), 0) != 0) {
            const int e = errno;
            munmap(p, bytes);
            err = std::string("mbind to node ") + std::to_string(numa_node) + " failed: " + strerror(e);
            return false;
        }
#else
        munmap(p, bytes);
        err = "mbind not available on this platform";
        return false;
#endif
    }
    base = (uint8_t *) p;
#else
    if (numa_node >= 0) {
        err = "NUMA binding is only implemented on Linux";
        return false;
    }
    base = (uint8_t *) calloc(1, bytes);
    if (!base) {
        err = "allocation failed";
        return false;
    }
#endif
    n_bytes = bytes;
    owned = true;
    alias.assign((bytes + piece_bytes - 1) / piece_bytes, nullptr);
    n_reg = 0;
    return true;
}

bool host_tier::register_with(const ggml_kv_mirror_ops * ops, ggml_kv_mirror_dev_t dev) {
    reg_ops = ops;
    reg_dev = dev;
    n_reg = 0;
    for (size_t i = 0; i < alias.size(); ++i) {
        uint8_t * p = base + i*piece_bytes;
        const size_t n = std::min(piece_bytes, n_bytes - i*piece_bytes);
        void * a = nullptr;
        if (!ops->host_register(dev, p, n, &a)) {
            break; // the rest stays pageable; copies there fall back to synchronous tensor get/set
        }
        alias[i] = a;
        n_reg = i + 1;
    }
    return n_reg == alias.size();
}

void host_tier::release() {
    if (reg_ops && reg_dev) {
        for (size_t i = 0; i < n_reg; ++i) {
            reg_ops->host_unregister(reg_dev, base + i*piece_bytes);
        }
    }
    n_reg = 0;
    reg_ops = nullptr;
    reg_dev = nullptr;
    if (owned && base) {
#if defined(__linux__)
        munmap(base, n_bytes);
#else
        free(base);
#endif
    }
    base = nullptr;
    n_bytes = 0;
    owned = false;
    alias.clear();
}

void host_tier::adopt(uint8_t * p, size_t n, size_t piece, const std::vector<void *> & aliases) {
    release();
    base = p;
    n_bytes = n;
    piece_bytes = piece;
    owned = false;
    alias = aliases;
    alias.resize((n + piece - 1) / piece, nullptr);
    n_reg = alias.size();
}

bool host_tier::contains(const void * p, size_t n) const {
    const uintptr_t a = (uintptr_t) p;
    const uintptr_t b = (uintptr_t) base;
    return base != nullptr && a >= b && a - b <= n_bytes && n <= n_bytes - (a - b);
}

bool host_tier::registered(const void * p, size_t n) const {
    if (!contains(p, n)) {
        return false;
    }
    const size_t off = (uintptr_t) p - (uintptr_t) base;
    return n == 0 || (off + n - 1) / piece_bytes < n_reg;
}

void * host_tier::alias_of(const void * p) const {
    if (!contains(p, 1)) {
        return nullptr;
    }
    const size_t off = (uintptr_t) p - (uintptr_t) base;
    const size_t i = off / piece_bytes;
    if (i >= n_reg || alias[i] == nullptr) {
        return nullptr;
    }
    return (uint8_t *) alias[i] + (off - i*piece_bytes);
}

void append_frag(const host_tier & tier, uint8_t * dev, uint8_t * host, size_t bytes, std::vector<ggml_kv_mirror_frag> & out) {
    while (bytes > 0) {
        size_t n = bytes;
        if (tier.contains(host, 1)) {
            const size_t off = (uintptr_t) host - (uintptr_t) tier.data();
            const size_t piece_end = (off / tier.piece() + 1) * tier.piece();
            n = std::min(bytes, piece_end - off);
        }
        out.push_back({ dev, host, tier.alias_of(host), n });
        dev   += n;
        host  += n;
        bytes -= n;
    }
}

void coalesce_frags(const host_tier & tier, std::vector<ggml_kv_mirror_frag> & frags) {
    if (frags.size() < 2) {
        return;
    }
    size_t w = 0;
    for (size_t i = 1; i < frags.size(); ++i) {
        auto & a = frags[w];
        const auto & b = frags[i];
        const bool contig = (uint8_t *) a.dev + a.bytes == (uint8_t *) b.dev && (uint8_t *) a.host + a.bytes == (uint8_t *) b.host;
        bool same_piece = false;
        if (contig) {
            if (tier.contains(a.host, a.bytes + b.bytes)) {
                const size_t off = (uintptr_t) a.host - (uintptr_t) tier.data();
                same_piece = off / tier.piece() == (off + a.bytes + b.bytes - 1) / tier.piece();
            } else {
                same_piece = !tier.contains(a.host, 1) && !tier.contains(b.host, 1);
            }
        }
        const bool alias_ok = (a.host_alias == nullptr && b.host_alias == nullptr) ||
                              (a.host_alias != nullptr && (uint8_t *) a.host_alias + a.bytes == (uint8_t *) b.host_alias);
        if (contig && same_piece && alias_ok) {
            a.bytes += b.bytes;
        } else {
            frags[++w] = b;
        }
    }
    frags.resize(w + 1);
}

void build_tail_frags(const std::vector<kv_region> & regions, const runs_t & runs, host_tier & tier,
                      std::vector<ggml_kv_mirror_frag> & out) {
    for (const auto & r : regions) {
        const size_t n_rows = r.bytes / r.row_bytes;
        for (const auto & run : runs) {
            if (run.first >= n_rows) {
                continue;
            }
            const size_t u1 = std::min<size_t>(run.second, n_rows);
            const size_t off = (size_t) run.first * r.row_bytes;
            append_frag(tier, r.dev + off, tier.data() + r.host_off + off, (u1 - run.first) * r.row_bytes, out);
        }
    }
}

//
// captured_stream / io_capture
//

void captured_stream::write_to(llama_io_write_i & io, const uint8_t * tier_base) const {
    for (const auto & s : segs) {
        if (s.lit) {
            io.write(lit.data() + s.off, s.len);
        } else {
            io.write(tier_base + s.off, s.len);
        }
    }
}

io_capture::io_capture(captured_stream & out, host_tier & tier, size_t slot_off, size_t slot_cap,
                       std::vector<ggml_kv_mirror_frag> & frags)
    : out(out), tier(tier), slot_off(slot_off), slot_cap(slot_cap), frags(frags) {}

void io_capture::write(const void * src, size_t size) {
    const size_t off = out.lit.size();
    out.lit.insert(out.lit.end(), (const uint8_t *) src, (const uint8_t *) src + size);
    if (!out.segs.empty() && out.segs.back().lit && out.segs.back().off + out.segs.back().len == off) {
        out.segs.back().len += size;
    } else {
        out.segs.push_back({ true, off, size });
    }
    out.n_bytes += size;
}

void io_capture::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    if (out.data_used + size > slot_cap) {
        throw std::runtime_error("kv mirror: checkpoint slot too small");
    }
    const size_t toff = slot_off + out.data_used;
    append_frag(tier, (uint8_t *) tensor->data + offset, tier.data() + toff, size, frags);
    out.segs.push_back({ false, toff, size });
    out.data_used += size;
    out.n_bytes   += size;
}

//
// io_from_mirror
//

io_from_mirror::io_from_mirror(llama_io_write_i & sink, const std::vector<kv_region> & regions, const unit_tracker & units,
                               const host_tier & tier, tensor_get_fn fallback)
    : sink(sink), regions(regions), units(units), tier(tier), fallback(std::move(fallback)) {}

void io_from_mirror::write(const void * src, size_t size) {
    sink.write(src, size);
}

void io_from_mirror::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    const uint8_t * addr = (const uint8_t *) tensor->data + offset;
    for (const auto & r : regions) {
        if (addr < r.dev || addr + size > r.dev + r.bytes) {
            continue;
        }
        const size_t off = addr - r.dev;
        if (off % r.row_bytes != 0 || size % r.row_bytes != 0) {
            break;
        }
        const uint32_t u0 = (uint32_t) (off / r.row_bytes);
        const uint32_t u1 = (uint32_t) ((off + size) / r.row_bytes);
        if (!units.valid(u0, u1)) {
            break;
        }
        sink.write(tier.data() + r.host_off + off, size);
        bytes_from_mirror += size;
        return;
    }
    tmp.resize(size);
    fallback(tensor, tmp.data(), offset, size);
    sink.write(tmp.data(), size);
    bytes_fallback += size;
}

//
// io_to_device
//

io_to_device::io_to_device(const uint8_t * src, size_t size, const host_tier * tier,
                           const ggml_kv_mirror_ops * ops, ggml_kv_mirror_dev_t dev, ggml_kv_mirror_path path,
                           std::function<bool(const ggml_tensor *)> on_device, tensor_set_fn fallback)
    : ptr(src), left(size), tier(tier), ops(ops), dev(dev), path(path), on_device(std::move(on_device)), fallback(std::move(fallback)) {}

io_to_device::~io_to_device() {
    try {
        flush();
    } catch (...) {
    }
}

void io_to_device::read(void * dst, size_t size) {
    if (size > left) {
        throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(dst, ptr, size);
    ptr += size;
    left -= size;
    n_read += size;
}

void io_to_device::read_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    if (size > left) {
        throw std::runtime_error("unexpectedly reached end of buffer");
    }
    pending.push_back({ tensor, ptr, offset, size });
    ptr += size;
    left -= size;
    n_read += size;
}

bool io_to_device::flush() {
    if (pending.empty()) {
        return true;
    }
    std::vector<ggml_kv_mirror_frag> frags;
    std::vector<const pend *> slow;
    for (const auto & p : pending) {
        const bool dma = ops && dev && on_device(p.t) && tier && tier->registered(p.src, p.size);
        if (dma) {
            frags.push_back({ (uint8_t *) p.t->data + p.off, (void *) p.src, tier->alias_of(p.src), p.size });
        } else {
            slow.push_back(&p);
        }
    }
    bool ok = true;
    if (!frags.empty()) {
        if (tier) {
            coalesce_frags(*tier, frags);
        }
        const uint64_t t = ops->copy_async(dev, frags.data(), frags.size(), GGML_KV_MIRROR_H2D, path);
        if (t == 0) {
            ok = false;
        } else {
            ops->ticket_wait(dev, t);
            n_frags_device += frags.size();
            for (const auto & f : frags) {
                bytes_device += f.bytes;
            }
        }
    }
    if (!ok) {
        // nothing was enqueued: move everything the slow way
        slow.clear();
        for (const auto & p : pending) {
            slow.push_back(&p);
        }
        ok = true;
    }
    for (const auto * p : slow) {
        fallback(p->t, p->src, p->off, p->size);
        bytes_fallback += p->size;
    }
    pending.clear();
    return ok;
}

//
// sinks
//

void io_sink_buffer::write(const void * src, size_t size) {
    buf.insert(buf.end(), (const uint8_t *) src, (const uint8_t *) src + size);
}

void io_sink_buffer::write_tensor(ggml_tensor *, size_t, size_t) {
    throw std::runtime_error("io_sink_buffer: write_tensor not supported");
}

void io_sink_fd::write(const void * src, size_t size) {
#if defined(__linux__)
    const uint8_t * p = (const uint8_t *) src;
    size_t left = size;
    while (left > 0) {
        const ssize_t w = ::write(fd, p, left);
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("kv mirror: write failed: ") + strerror(errno));
        }
        p += w;
        left -= (size_t) w;
    }
    n += size;
#else
    GGML_UNUSED(src);
    GGML_UNUSED(size);
    throw std::runtime_error("io_sink_fd: not supported on this platform");
#endif
}

void io_sink_fd::write_tensor(ggml_tensor *, size_t, size_t) {
    throw std::runtime_error("io_sink_fd: write_tensor not supported");
}

//
// memcpy ops
//

namespace {

struct memcpy_dev_state {
    uint64_t next = 1;
};

ggml_kv_mirror_dev_t mc_init(ggml_backend_t) {
    return (ggml_kv_mirror_dev_t) new memcpy_dev_state;
}

void mc_free(ggml_kv_mirror_dev_t d) {
    delete (memcpy_dev_state *) d;
}

bool mc_register(ggml_kv_mirror_dev_t, void * host, size_t, void ** alias) {
    *alias = host;
    return true;
}

void mc_unregister(ggml_kv_mirror_dev_t, void *) {}

uint64_t mc_copy(ggml_kv_mirror_dev_t d, const ggml_kv_mirror_frag * f, size_t n, ggml_kv_mirror_dir dir, ggml_kv_mirror_path) {
    if (n == 0) {
        return 0;
    }
    for (size_t i = 0; i < n; ++i) {
        if (dir == GGML_KV_MIRROR_D2H) {
            memcpy(f[i].host, f[i].dev, f[i].bytes);
        } else {
            memcpy(f[i].dev, f[i].host, f[i].bytes);
        }
    }
    return ((memcpy_dev_state *) d)->next++;
}

bool mc_done(ggml_kv_mirror_dev_t, uint64_t) { return true; }
void mc_wait(ggml_kv_mirror_dev_t, uint64_t) {}
void mc_bwait(ggml_kv_mirror_dev_t, uint64_t) {}

const ggml_kv_mirror_ops k_memcpy_ops = {
    GGML_KV_MIRROR_OPS_VERSION, mc_init, mc_free, mc_register, mc_unregister, mc_copy, mc_done, mc_wait, mc_bwait,
};

} // namespace

const ggml_kv_mirror_ops * memcpy_ops() {
    return &k_memcpy_ops;
}

} // namespace llama_kvm
