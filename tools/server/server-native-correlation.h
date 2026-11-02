#pragma once

#include <cctype>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>

enum class server_correlation_parse { absent, valid, invalid };

inline bool server_valid_native_correlation(const std::string & value) {
    if (value.size() != 32) return false;
    for (const char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

inline server_correlation_parse server_parse_native_correlation(
        const std::map<std::string, std::string> & headers, std::string & value) {
    bool found = false;
    for (const auto & item : headers) {
        std::string key = item.first;
        for (char & c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (key != "x-llama-native-correlation-id") continue;
        if (found || !server_valid_native_correlation(item.second)) return server_correlation_parse::invalid;
        found = true;
        value = item.second;
    }
    return found ? server_correlation_parse::valid : server_correlation_parse::absent;
}

inline bool server_erase_has_native_correlation(const std::map<std::string, std::string> & headers,
                                                const std::map<std::string, std::string> & params) {
    std::string ignored;
    return server_parse_native_correlation(headers, ignored) != server_correlation_parse::absent ||
           params.count("native_correlation_id") || params.count("expected_native_correlation_id");
}

// IDs are never reused within a server incarnation; capacity is bounded.
class server_native_correlation_registry {
public:
    explicit server_native_correlation_registry(std::size_t limit = 65536) : limit_(limit) {}
    bool reserve(const std::string & value) {
        if (!server_valid_native_correlation(value)) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (ids_.size() >= limit_) return false;
        return ids_.insert(value).second;
    }
private:
    const std::size_t limit_;
    std::mutex mutex_;
    std::unordered_set<std::string> ids_;
};
