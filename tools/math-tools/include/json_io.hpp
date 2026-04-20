#pragma once

#include "json.hpp"
#include "common.hpp"
#include <iostream>
#include <sstream>

namespace math_tools {

using json = nlohmann::json;

// Read JSON from stdin
inline json read_input() {
    std::stringstream buffer;
    buffer << std::cin.rdbuf();
    std::string input = buffer.str();

    if (input.empty()) {
        throw std::runtime_error("Empty input");
    }

    return json::parse(input);
}

// Write result to stdout
inline void write_output(const Result& result) {
    json output;
    output["status"] = result.success ? "success" : "error";
    if (result.error.empty()) {
        output["error"] = nullptr;
    } else {
        output["error"] = result.error;
    }
    output["stats"]["elapsed_ms"] = result.elapsed_ms;

    if (!result.data.empty()) {
        output["result"] = json::parse(result.data);
    }

    std::cout << output.dump() << std::endl;
}

// Write error to stdout
inline void write_error(const std::string& error) {
    json output;
    output["status"] = "error";
    output["error"] = error;
    output["result"] = nullptr;
    std::cout << output.dump() << std::endl;
}

// Convert vector to JSON array
template <typename T>
json vector_to_json(const std::vector<T>& vec) {
    return json(vec);
}

// Convert JSON array to vector
template <typename T>
std::vector<T> json_to_vector(const json& j) {
    return j.get<std::vector<T>>();
}

// Parse matrix from JSON (row-major)
template <typename T>
std::vector<std::vector<T>> json_to_matrix(const json& j) {
    std::vector<std::vector<T>> matrix;
    for (const auto& row : j) {
        matrix.push_back(row.get<std::vector<T>>());
    }
    return matrix;
}

} // namespace math_tools
