#pragma once

#include <charconv>
#include <string>
#include <system_error>

// Parse only an explicit non-negative native task ID; -1 means no condition.
inline bool server_parse_expected_task_id(const std::string & value, int & id) {
    if (value.empty()) {
        return false;
    }
    for (const char ch : value) {
        if (ch < '0' || ch > '9') {
            return false;
        }
    }
    int parsed = -1;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() || parsed < 0) {
        return false;
    }
    id = parsed;
    return true;
}

inline bool server_slot_erase_matches(int expected, const std::string & expected_instance,
                                     const std::string & current_instance,
                                     bool processing, bool has_task, int task_id) {
    return expected < 0 || (!expected_instance.empty() && expected_instance == current_instance &&
                            processing && has_task && expected == task_id);
}

inline bool server_parse_conditional_erase(bool has_expected, const std::string & task_id,
                                           bool has_instance, const std::string & instance,
                                           int & expected) {
    if (has_expected != has_instance) {
        return false;
    }
    if (!has_expected) {
        expected = -1;
        return true;
    }
    return !instance.empty() && server_parse_expected_task_id(task_id, expected);
}
