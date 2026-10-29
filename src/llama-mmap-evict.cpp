#include "llama-mmap-evict.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace llama_mmap_evict {

// This is only used on a refused range, to capture enough process-local
// provenance to distinguish a split/adjacent mmap view from an invalid span.
static std::string describe_vmas(uintptr_t begin, uintptr_t end) {
    std::ifstream maps("/proc/self/maps");
    std::ostringstream out;
    std::string line;
    bool any = false;
    while (std::getline(maps, line)) {
        std::istringstream fields(line);
        std::string address, perms, offset, device, inode;
        if (!(fields >> address >> perms >> offset >> device >> inode)) continue;
        const size_t dash = address.find('-');
        if (dash == std::string::npos) continue;
        uintptr_t vma_begin = 0, vma_end = 0;
        try {
            vma_begin = static_cast<uintptr_t>(std::stoull(address.substr(0, dash), nullptr, 16));
            vma_end = static_cast<uintptr_t>(std::stoull(address.substr(dash + 1), nullptr, 16));
        } catch (...) {
            continue;
        }
        if (vma_end < begin || vma_begin > end) continue;
        if (any) out << " ; ";
        any = true;
        out << "[" << std::hex << vma_begin << "," << vma_end << ")"
            << " perms=" << perms << " file_offset=0x" << offset
            << " dev=" << device << " inode=" << inode;
        std::string path;
        std::getline(fields, path);
        if (!path.empty()) out << " path=" << path;
    }
    return any ? out.str() : "unavailable";
}

bool can_enable(bool existing_eval_callback) noexcept {
    return !existing_eval_callback;
}

plan::plan(std::vector<mapping> mappings) : mappings_(std::move(mappings)), page_size_(0) {
#ifdef __linux__
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::runtime_error("cannot determine system page size for mmap eviction");
    }
    page_size_ = static_cast<size_t>(page_size);
#else
    throw std::runtime_error("mmap eviction prototype requires Linux madvise");
#endif
    for (const auto & map : mappings_) {
        if (map.address == 0 || map.size == 0 || map.address % page_size_ != 0 ||
                map.size > std::numeric_limits<uintptr_t>::max() - map.address) {
            throw std::invalid_argument("invalid mmap eviction mapping");
        }
    }
}

bool plan::add_tensor_use(size_t node_index, const void * data, size_t bytes) {
    if (finalized_) {
        throw std::logic_error("cannot add tensor use after mmap eviction plan finalization");
    }
    if (data == nullptr || bytes == 0) {
        return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
    if (bytes > std::numeric_limits<uintptr_t>::max() - begin) {
        throw std::invalid_argument("tensor range overflow in mmap eviction plan");
    }
    const uintptr_t end = begin + bytes;
    for (size_t i = 0; i < mappings_.size(); ++i) {
        const uintptr_t map_begin = mappings_[i].address;
        const uintptr_t map_end = map_begin + mappings_[i].size;
        if (begin >= map_begin && begin < map_end) {
            if (end > map_end) {
                throw std::invalid_argument("tensor range extends beyond mmap backing range");
            }
            const size_t offset_begin = static_cast<size_t>(begin - map_begin);
            const size_t offset_end = static_cast<size_t>(end - map_begin);
            const size_t aligned_begin = offset_begin - (offset_begin % page_size_);
            const size_t remainder = offset_end % page_size_;
            const size_t aligned_end = remainder == 0 ? offset_end :
                std::min(mappings_[i].size, offset_end + page_size_ - remainder);
            uses_.push_back({i, aligned_begin, aligned_end, node_index});
            return true;
        }
        if (begin < map_begin && end > map_begin) {
            throw std::invalid_argument("tensor range partially overlaps mmap backing range: data=[" +
                    std::to_string(begin) + "," + std::to_string(end) + ") mapping[" +
                    std::to_string(i) + "]=[" + std::to_string(map_begin) + "," +
                    std::to_string(map_end) + ") tensor_bytes=" + std::to_string(bytes) +
                    " all_mappings=" + [&]() {
                        std::string result;
                        for (size_t m = 0; m < mappings_.size(); ++m) {
                            if (!result.empty()) result += ";";
                            result += "[" + std::to_string(m) + "]=[" +
                                std::to_string(mappings_[m].address) + "," +
                                std::to_string(mappings_[m].address + mappings_[m].size) + ")";
                        }
                        return result;
                    }() + " vmas=" + describe_vmas(begin, end));
        }
    }
    return false;
}

void plan::finalize(size_t node_count) {
    if (finalized_) {
        throw std::logic_error("mmap eviction plan finalized twice");
    }
    after_.resize(node_count);
    struct event {
        size_t position;
        size_t node_index;
        bool start;
    };
    std::vector<std::vector<event>> events(mappings_.size());
    for (const auto & item : uses_) {
        if (item.node_index >= node_count || item.begin >= item.end || item.end > mappings_[item.mapping_index].size) {
            throw std::invalid_argument("invalid use in mmap eviction plan");
        }
        events[item.mapping_index].push_back({item.begin, item.node_index, true});
        events[item.mapping_index].push_back({item.end, item.node_index, false});
    }

    for (size_t map_index = 0; map_index < events.size(); ++map_index) {
        auto & map_events = events[map_index];
        std::sort(map_events.begin(), map_events.end(), [](const event & a, const event & b) {
            if (a.position != b.position) return a.position < b.position;
            return a.start < b.start; // remove half-open intervals before adding new ones
        });
        std::multiset<size_t> active;
        size_t cursor = 0;
        while (cursor < map_events.size()) {
            const size_t position = map_events[cursor].position;
            size_t next = cursor;
            while (next < map_events.size() && map_events[next].position == position) {
                const event & e = map_events[next];
                if (e.start) {
                    active.insert(e.node_index);
                } else {
                    const auto it = active.find(e.node_index);
                    if (it == active.end()) {
                        throw std::logic_error("unbalanced interval in mmap eviction plan");
                    }
                    active.erase(it);
                }
                ++next;
            }
            if (next < map_events.size() && !active.empty()) {
                const size_t end = map_events[next].position;
                const size_t last_use = *active.rbegin();
                if (end > position) {
                    auto & ranges = after_[last_use];
                    if (!ranges.empty() && ranges.back().mapping_index == map_index &&
                            ranges.back().offset + ranges.back().size == position) {
                        ranges.back().size += end - position;
                    } else {
                        ranges.push_back({map_index, position, end - position});
                    }
                }
            }
            cursor = next;
        }
    }
    finalized_ = true;
}

const std::vector<range> & plan::after(size_t node_index) const {
    if (!finalized_ || node_index >= after_.size()) {
        throw std::out_of_range("mmap eviction plan node index");
    }
    return after_[node_index];
}

bool plan::drop_after(size_t node_index) const {
    const auto & ranges = after(node_index);
#ifdef __linux__
    for (const auto & r : ranges) {
        const auto & map = mappings_.at(r.mapping_index);
        if (madvise(reinterpret_cast<void *>(map.address + r.offset), r.size, MADV_DONTNEED) != 0) {
            return false;
        }
    }
    return true;
#else
    (void) ranges;
    return false;
#endif
}

} // namespace llama_mmap_evict
