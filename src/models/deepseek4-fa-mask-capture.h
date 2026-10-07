#pragma once

#include "ggml-backend.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <unistd.h>

static bool dsv4_mask_env_present(const char * key) {
    const char * value = std::getenv(key);
    return value && value[0];
}

static std::string dsv4_mask_json_string(const char * text) {
    std::string out = "\"";
    for (const unsigned char * p = (const unsigned char *) text; *p; ++p) {
        if (*p == '"' || *p == '\\') { out += '\\'; out += (char) *p; }
        else if (*p < 0x20) {
            char escaped[7];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", *p);
            out += escaped;
        } else { out += (char) *p; }
    }
    return out + '"';
}

// Publish a complete file atomically, preserving the first capture of a shape.
static bool dsv4_mask_publish(const std::string & path, const void * data, size_t bytes) {
    std::string pattern = path + ".part-XXXXXX";
    std::vector<char> temp(pattern.begin(), pattern.end());
    temp.push_back('\0');
    const int fd = mkstemp(temp.data());
    if (fd < 0) { return false; }
    FILE * file = fdopen(fd, "wb");
    if (!file) { close(fd); unlink(temp.data()); return false; }
    const bool wrote = fwrite(data, 1, bytes, file) == bytes;
    const bool closed = fclose(file) == 0;
    const bool published = wrote && closed && link(temp.data(), path.c_str()) == 0;
    unlink(temp.data());
    return published;
}

static void dsv4_capture_fa_mask(ggml_tensor * dst, const ggml_tensor * src,
                               int ith, int nth, void * userdata) {
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    // CPU graph dispatch calls every pool thread; its node barrier publishes thread 0's copy.
    if (ith != 0) { return; }
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    GGML_ASSERT(src->type == GGML_TYPE_F16);
    const size_t bytes = ggml_nbytes(src);
    std::vector<ggml_fp16_t> mask(ggml_nelements(src));
    ggml_backend_tensor_get(src, mask.data(), 0, bytes);
    std::memcpy(dst->data, mask.data(), bytes);

    const char * dir = std::getenv("AUTOKERNEL_DUMP_FA_MASK");
    if (!dir || !dir[0]) { return; }
    const uintptr_t capture_id = (uintptr_t) userdata;
    const int64_t kv = src->ne[0], nb = capture_id & 0x3ff;
    const int64_t layer = (capture_id >> 10) & 0xffff;
    const int64_t ratio = capture_id >> 26;
    if ((kv != 4096 && kv != 8192 && kv != 32768 && kv != 65536) || nb < 2 || nb > 5 ||
            src->ne[1] < nb || src->ne[2] != 1 || src->ne[3] != 1) { return; }
    GGML_ASSERT(ratio == 1 || ratio == 2);
    const std::string stem = std::string(dir) + "/ds41_realmask_kv" +
                            std::to_string(kv / 1024) + "k_nb" + std::to_string(nb);
    // FLASH_ATTN_EXT consumes the first q->ne[2] rows; mask rows may be padded.
    std::vector<unsigned char> little_endian((size_t) kv * nb * 2);
    for (size_t i = 0; i < little_endian.size() / 2; ++i) {
        little_endian[2*i] = mask[i] & 0xff;
        little_endian[2*i + 1] = mask[i] >> 8;
    }
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (const unsigned char byte : little_endian) { hash = (hash ^ byte) * UINT64_C(0x100000001b3); }
    char hash_text[17], timestamp[21];
    std::snprintf(hash_text, sizeof(hash_text), "%016" PRIx64, hash);
    const std::time_t now = std::time(nullptr);
    std::tm utc;
    gmtime_r(&now, &utc);
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);
    if (!dsv4_mask_publish(stem + ".mask.f16", little_endian.data(), little_endian.size())) { return; }
    const std::string metadata =
        "{\"schema\":\"epyc.autokernel.ds41_fa_mask.v1\",\"type\":\"f16\","
        "\"byte_order\":\"little\",\"layout\":\"token_kv\",\"ne\":[" +
        std::to_string(kv) + "," + std::to_string(nb) + ",1,1],\"n_q_heads\":64,"
        "\"n_kv_heads\":1,\"hsk\":512,\"hsv\":512,\"mask_kind\":\"raw_plus_compressed_top_k\","
        "\"captured_at\":" + dsv4_mask_json_string(timestamp) +
        ",\"mask_hash_algorithm\":\"fnv1a64\",\"mask_hash\":" + dsv4_mask_json_string(hash_text) +
        ",\"mask_bytes\":" + std::to_string(little_endian.size()) +
        ",\"original_ne\":[" + std::to_string(src->ne[0]) + "," + std::to_string(src->ne[1]) +
        "," + std::to_string(src->ne[2]) + "," + std::to_string(src->ne[3]) + "],"
        "\"original_nb\":[" + std::to_string(src->nb[0]) + "," + std::to_string(src->nb[1]) +
        "," + std::to_string(src->nb[2]) + "," + std::to_string(src->nb[3]) + "],"
        "\"slice\":{\"row_start\":0,\"row_count\":" + std::to_string(nb) +
        ",\"head_index\":0,\"stream_index\":0},\"layer\":" + std::to_string(layer) +
        ",\"compressed_ratio\":" + std::to_string(ratio) +
        ",\"run_id\":" + dsv4_mask_json_string(std::getenv("AUTOKERNEL_FA_MASK_RUN_ID")) + ","
        "\"source_commit\":" + dsv4_mask_json_string(std::getenv("AUTOKERNEL_FA_MASK_SOURCE_COMMIT")) +
        ",\"model\":" + dsv4_mask_json_string(std::getenv("AUTOKERNEL_FA_MASK_MODEL")) +
        ",\"model_sha256\":" + dsv4_mask_json_string(std::getenv("AUTOKERNEL_FA_MASK_MODEL_SHA256")) +
        ",\"recipe_sha256\":" + dsv4_mask_json_string(std::getenv("AUTOKERNEL_FA_MASK_RECIPE_SHA256")) +
        ",\"prompt_sha256\":" + dsv4_mask_json_string(std::getenv("AUTOKERNEL_FA_MASK_PROMPT_SHA256")) + "}\n";
    if (!dsv4_mask_publish(stem + ".mask.json", metadata.data(), metadata.size())) {
        LLAMA_LOG_ERROR("DS41 FA mask capture metadata failed: %s\n", stem.c_str());
    }
}
