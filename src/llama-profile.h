#pragma once
#include "llama.h"
#include <cstdint>
#include <string>
// Observation-only, opt-in prospective capture. These are not serving-policy APIs.
LLAMA_API bool llama_profile_enabled();
class LLAMA_API llama_profile_scope {
public:
    llama_profile_scope(const char * json, const llama_context * target, const llama_context * drafter);
    ~llama_profile_scope();
    llama_profile_scope(const llama_profile_scope &) = delete;
    llama_profile_scope & operator=(const llama_profile_scope &) = delete;
private:
    bool active;
    uint64_t previous_scope;
    std::string previous;
    const llama_context * previous_target;
    const llama_context * previous_drafter;
};
LLAMA_API void llama_profile_record(const char * json_fields);
uint64_t llama_profile_begin(const llama_context * ctx, const llama_batch & batch, const char * api);
void llama_profile_end(uint64_t call, int32_t result, const char * api);
