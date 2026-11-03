#pragma once
#include "llama-lsa-native.h"
#include <map>

namespace llama_lsa {
// LSAMAP1: little endian, bounded, hashed BF16 payload; never loads Python/torch.
struct map_bundle {
    geometry shape;
    std::string model_identity;
    std::string source_sha256;
    std::string payload_sha256;
    std::map<int, projection_maps> layers;
};
// expected_source_sha256 is mandatory: it binds the map to an approved export.
bool load_map_bundle(const std::string & path, const std::string & model_identity,
                     const geometry & shape, const std::vector<int> & global_layers,
                     const std::string & expected_source_sha256, map_bundle & output,
                     std::string * error = nullptr);
std::string map_sha256(const void * data, size_t size);
}
