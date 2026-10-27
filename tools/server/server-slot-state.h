#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

static constexpr uint32_t SERVER_SLOT_DRAFT_MAGIC = 0x54464453; // SDFT
static constexpr uint32_t SERVER_SLOT_DRAFT_VERSION = 1;

inline bool server_slot_draft_write(std::ostream & out, const std::vector<uint8_t> & draft,
                                    const std::vector<uint8_t> & spec, size_t & n_written) {
    const uint32_t header[] = { SERVER_SLOT_DRAFT_MAGIC, SERVER_SLOT_DRAFT_VERSION };
    out.write((const char *) header, sizeof(header));
    n_written += sizeof(header);
    for (const auto * data : { &draft, &spec }) {
        const uint64_t size = data->size();
        out.write((const char *) &size, sizeof(size));
        n_written += sizeof(size);
        if (size) {
            out.write((const char *) data->data(), size);
            n_written += size;
        }
    }
    out.flush();
    return (bool) out;
}

inline bool server_slot_draft_read(std::istream & in, size_t n_avail,
                                   std::vector<uint8_t> & draft, std::vector<uint8_t> & spec,
                                   size_t & n_read) {
    auto read = [&](void * dst, size_t size) {
        if (n_read > n_avail || size > n_avail - n_read || !in.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    };
    uint32_t header[2];
    if (!read(header, sizeof(header)) || header[0] != SERVER_SLOT_DRAFT_MAGIC ||
            header[1] != SERVER_SLOT_DRAFT_VERSION) {
        return false;
    }
    for (auto * data : { &draft, &spec }) {
        uint64_t size;
        if (!read(&size, sizeof(size)) || n_read > n_avail || size > n_avail - n_read) {
            return false;
        }
        data->resize(size);
        if (size && !read(data->data(), size)) {
            return false;
        }
    }
    return true;
}
