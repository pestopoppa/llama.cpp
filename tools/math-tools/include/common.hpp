#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

namespace math_tools {

// Result structure for all commands
struct Result {
    bool success = false;
    std::string error;
    double elapsed_ms = 0.0;

    // Command-specific data stored as JSON string
    std::string data;
};

// Timing helper
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Command interface
class Command {
public:
    virtual ~Command() = default;
    virtual Result execute(const std::string& params_json) = 0;
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
};

} // namespace math_tools
