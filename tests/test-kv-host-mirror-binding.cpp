// Model-free save/erase/restore regression: production memory serializers and
// mirror binding, with manually written deterministic cache rows. No decode.
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "llama-io.h"
#include "ggml-backend.h"
#include "../tools/server/server-slot-state.h"
#ifndef HSF_KVM_REFERENCE_ONLY
#include "llama-kv-host-mirror-bind.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * what) {
    if (!ok) {
        throw std::runtime_error(what);
    }
}

struct fixture_model : llama_model {
    fixture_model() : llama_model(llama_model_default_params()) {}
    void load_stats(llama_model_loader &) override { throw std::runtime_error("fixture has no weights"); }
    void load_hparams(llama_model_loader &) override { throw std::runtime_error("fixture has no weights"); }
    void load_vocab(llama_model_loader &) override { throw std::runtime_error("fixture has no weights"); }
    bool load_tensors(llama_model_loader &) override { throw std::runtime_error("fixture has no weights"); }
    void load_arch_hparams(llama_model_loader &) override { throw std::runtime_error("fixture has no graph"); }
    void load_arch_tensors(llama_model_loader &) override { throw std::runtime_error("fixture has no graph"); }
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        throw std::runtime_error("fixture never executes inference");
    }
};

struct state_writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * src, size_t n) override {
        const auto * p = (const uint8_t *) src;
        bytes.insert(bytes.end(), p, p + n);
    }
    void write_tensor(ggml_tensor * t, size_t off, size_t n) override {
        const size_t old = bytes.size();
        bytes.resize(old + n);
        ggml_backend_tensor_get(t, bytes.data() + old, off, n);
    }
    size_t n_bytes() override { return bytes.size(); }
};

struct state_reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t at = 0;
    explicit state_reader(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
    void read(void * dst, size_t n) override {
        require(at <= bytes.size() && n <= bytes.size() - at, "state read exceeded payload");
        memcpy(dst, bytes.data() + at, n);
        at += n;
    }
    void read_tensor(ggml_tensor * t, size_t off, size_t n) override {
        require(at <= bytes.size() && n <= bytes.size() - at, "tensor read exceeded payload");
        ggml_backend_tensor_set(t, bytes.data() + at, off, n);
        at += n;
    }
    size_t n_bytes() override { return at; }
};

static std::vector<uint8_t> state_bytes(llama_memory_i & mem) {
    state_writer out;
    mem.state_write(out, 0, 0);
    return out.bytes;
}

static std::vector<uint8_t> file_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    require((bool) in, "cannot read fixture file");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static void put_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write((const char *) bytes.data(), bytes.size());
    require((bool) out, "cannot write fixture file");
}

static std::vector<uint8_t> c5_file_bytes(llama_memory_i & mem, const std::vector<llama_token> & tokens) {
    // Same header and memory serialization as C5 llama_context::state_seq_save_file.
    state_writer out;
    const uint32_t header[] = {LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, (uint32_t) tokens.size()};
    out.write(header, sizeof(header));
    out.write(tokens.data(), tokens.size() * sizeof(llama_token));
    mem.state_write(out, 0, 0);
    return out.bytes;
}

static void append_step(llama_memory_hybrid & mem, std::vector<llama_token> & tokens, int n, int salt) {
    auto * kv = mem.get_mem_attn();
    auto * rc = mem.get_mem_recr();
    const size_t first = tokens.size();
    std::vector<llama_pos> pos(n);
    std::vector<llama_token> ids(n);
    std::vector<int32_t> counts(n, 1);
    llama_seq_id seq = 0;
    std::vector<llama_seq_id *> seqs(n, &seq);
    for (int i = 0; i < n; ++i) {
        pos[i] = (llama_pos) first + i;
        ids[i] = 100 + pos[i];
        tokens.push_back(ids[i]);
    }
    llama_ubatch batch{};
    batch.n_tokens = n;
    batch.n_seq_tokens = n;
    batch.n_seqs = batch.n_seqs_unq = batch.n_pos = 1;
    batch.b_equal_seqs = 1;
    batch.pos = pos.data();
    batch.token = ids.data();
    batch.n_seq_id = counts.data();
    batch.seq_id = seqs.data();
    batch.seq_id_unq = &seq;
    auto slots = kv->find_slot(batch, false);
    require(!slots.empty(), "fixture KV has no slots");
    kv->apply_ubatch(slots, batch);
    for (ggml_tensor * t : {kv->get_k_layer_raw(0), kv->get_v_layer_raw(0)}) {
        require(t != nullptr, "fixture attention tensor is absent");
        for (int i = 0; i < n; ++i) {
            std::vector<float> row(t->ne[0]);
            for (size_t j = 0; j < row.size(); ++j) {
                row[j] = (float) (salt * 1000 + pos[i] * 10 + j);
            }
            ggml_backend_tensor_set(t, row.data(), slots.idxs[0][i] * t->nb[1], row.size() * sizeof(float));
        }
    }
    // Exercise nonzero rollback rows. The serializer must capture the logical
    // current row, not the first physical row in the recurrent buffer.
    rc->cells[0].pos = (llama_pos) tokens.size() - 1;
    rc->cells[0].tail = rc->cells[0].src = rc->cells[0].src0 = 0;
    rc->cells[0].seq_id = {0};
    rc->used = 1;
    rc->set_rs_idx(0, (uint32_t) tokens.size() % 3);
    for (const auto * layers : {&rc->r_l, &rc->s_l, &rc->p_l}) {
        for (ggml_tensor * t : *layers) {
            if (!t) { continue; }
            std::vector<float> data(ggml_nelements(t));
            for (size_t j = 0; j < data.size(); ++j) {
                data[j] = (float) (salt * 1000 + tokens.size() * 10 + j);
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        }
    }
}

static void run(const std::string & prefix, const std::string & mode) {
    fixture_model model;
    model.hparams = {};
    auto & hp = model.hparams;
    hp.n_layer_all = 2;
    hp.n_embd = hp.n_embd_head_k_full = hp.n_embd_head_v_full = 8;
    hp.n_head_arr.fill(1);
    hp.n_head_kv_arr.fill(1);
    hp.is_recr_impl[1] = 1;
    hp.ssm_d_conv = 3;
    hp.ssm_d_inner = 8;
    hp.ssm_d_state = 4;
    llama_memory_hybrid target(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 64, 1, 0, LLAMA_SWA_TYPE_NONE,
            GGML_TYPE_F32, GGML_TYPE_F32, 2, 2, 2, false, true);
    llama_memory_hybrid draft(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 64, 1, 0, LLAMA_SWA_TYPE_NONE,
            GGML_TYPE_F32, GGML_TYPE_F32, 2, 2, 2, false, true);
#ifndef HSF_KVM_REFERENCE_ONLY
    std::unique_ptr<llama_kv_mirror> mirror;
    if (mode != "off") {
        std::string err;
        mirror = llama_kv_mirror::create(&target, {}, 2, "k=6,node=-1,mode=" + mode + ",piece_mib=2", err);
        require(mirror != nullptr, err.c_str());
    }
#else
    require(mode == "off", "C5 reference supports only off");
#endif
    std::vector<llama_token> tokens, draft_tokens;
    const int steps[] = {1, 1, 1, 1, 1, 1, 5, 7, 4, 9, 3, 9};
    for (int n : steps) {
#ifndef HSF_KVM_REFERENCE_ONLY
        if (mirror) { mirror->on_decode_begin(); }
#endif
        append_step(target, tokens, n, 1);
        append_step(draft, draft_tokens, n, 2);
#ifndef HSF_KVM_REFERENCE_ONLY
        llama_batch step{};
        step.n_tokens = n;
        if (mirror) { mirror->on_decode_end(step); }
#endif
    }
    require(tokens.size() == 43, "fixture must retain a 37-token tail after checkpoint position 5");
    const auto live_target = state_bytes(target);
    const auto live_draft = state_bytes(draft);
    const auto expected = c5_file_bytes(target, tokens);
    const std::string path = prefix + ".bin";
    size_t saved = tokens.size();
    size_t target_size = expected.size();
#ifndef HSF_KVM_REFERENCE_ONLY
    if (mirror) {
        target_size = mirror->seq_save_file(path.c_str(), 0, tokens.data(), tokens.size(), &saved, []() {});
        require(target_size != 0, "mirror save failed");
        require(file_bytes(path) == expected, "live mirror file differs from C5 serializer");
        llama_kv_mirror_stats stats{};
        mirror->get_stats(&stats);
        require(stats.n_steps == 12 && stats.n_ckpt_issued >= 2 && stats.n_ckpt_issued == stats.n_ckpt_complete,
                "save must advance a scheduled stale checkpoint");
        require(stats.save_mirror_bytes > 0 && stats.save_fallback_bytes == 0, "binding mirror path was not reached");
        require(stats.last_replay_tokens == 0 && stats.last_replay_steps == 0, "save retained a replay tail");
    } else
#endif
    {
        put_bytes(path, expected);
    }
    // Nonempty opaque speculative bytes exercise the production appendix wire
    // path; this is not a model/drafter behavioral or inference test.
    std::vector<uint8_t> spec(sizeof(llama_pos) + 3 * sizeof(float), 0x3c);
    const llama_pos boundary = target.seq_pos_max(0);
    memcpy(spec.data(), &boundary, sizeof(boundary));
    size_t appendix_size = 0;
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        require(server_slot_draft_write(out, live_draft, spec, appendix_size), "draft appendix write failed");
    }
    state_writer c5_draft;
    const uint32_t c5_header[] = {0x54464453, 1};
    c5_draft.write(c5_header, sizeof(c5_header));
    const std::vector<uint8_t> * c5_draft_parts[] = {&live_draft, &spec};
    for (const auto * data : c5_draft_parts) {
        const uint64_t n = data->size();
        c5_draft.write(&n, sizeof(n));
        c5_draft.write(data->data(), n);
    }
    auto c5_complete = expected;
    c5_complete.insert(c5_complete.end(), c5_draft.bytes.begin(), c5_draft.bytes.end());
    require(file_bytes(path) == c5_complete, "live target/draft/spec file differs from C5 wire bytes");
    const size_t erased = (size_t) target.seq_pos_max(0) + 1;
    require(target.seq_rm(0, -1, -1), "erase failed");
    require(draft.seq_rm(0, -1, -1), "draft erase failed");
    require(target.seq_pos_max(0) == -1 && draft.seq_pos_max(0) == -1, "erase did not clear both memories");
    std::vector<llama_token> restored_tokens(64);
    size_t restored = 0;
#ifndef HSF_KVM_REFERENCE_ONLY
    if (mirror) {
        const size_t got = mirror->seq_load_file(path.c_str(), 0, restored_tokens.data(), restored_tokens.size(), &restored,
                [&](llama_io_read_i & io) {
                    target.state_read(io, 0, 0);
                    mirror->on_state_read(0);
                    return io.n_bytes();
                });
        require(got == target_size, "binding consumed incorrect target bytes");
    } else
#endif
    {
        state_reader in(expected);
        uint32_t header[3];
        in.read(header, sizeof(header));
        restored = header[2];
        in.read(restored_tokens.data(), restored * sizeof(llama_token));
        target.state_read(in, 0, 0);
    }
    restored_tokens.resize(restored);
    require(saved == erased && restored == erased, "saved/restored tokens differ from erased live state");
    require(restored_tokens == tokens, "restored tokens differ");
    require(target.seq_pos_max(0) == boundary, "restored target position differs");
    require(state_bytes(target) == live_target, "restored target state differs (replay required)");
    std::vector<uint8_t> restored_draft, restored_spec;
    size_t n_read = 0;
    std::ifstream in(path, std::ios::binary);
    in.seekg(target_size);
    require(server_slot_draft_read(in, appendix_size, restored_draft, restored_spec, n_read), "draft appendix read failed");
    require(n_read == appendix_size && restored_spec == spec && restored_draft == live_draft, "live draft/spec appendix differs");
    state_reader draft_in(restored_draft);
    draft.state_read(draft_in, 0, 0);
    require(draft.seq_pos_max(0) == boundary && state_bytes(draft) == live_draft, "draft restored at a different boundary");
    put_bytes(prefix + ".target", live_target);
    put_bytes(prefix + ".draft", live_draft);
    printf("binding mode=%s saved=%zu erased=%zu restored=%zu replay=0 target_bytes=%zu draft_bytes=%zu spec_bytes=%zu OK\n",
            mode.c_str(), saved, erased, restored, live_target.size(), live_draft.size(), spec.size());
}

int main(int argc, char ** argv) {
    try {
        const char * tmp = std::getenv("TMPDIR");
        const std::string prefix = argc > 1 ? argv[1] : std::string(tmp ? tmp : ".") + "/hsf-kvm-binding";
        const std::string mode = argc > 2 ? argv[2] : "lag";
        require(mode == "off" || mode == "lag" || mode == "fresh", "mode must be off, lag or fresh");
        run(prefix, mode);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "test-kv-host-mirror-binding: %s\n", e.what());
        return 1;
    }
}
