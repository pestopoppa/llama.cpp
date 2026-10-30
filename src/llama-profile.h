#pragma once
#include "llama.h"
#include <cstdint>
// Observation-only, opt-in prospective capture. These are not serving-policy APIs.
LLAMA_API bool llama_profile_enabled();
LLAMA_API void llama_profile_slots(const char * json);
LLAMA_API void llama_profile_record(const char * json_fields);
uint64_t llama_profile_decode_begin(const llama_context * ctx, const llama_batch & batch);
void llama_profile_decode_end(uint64_t call, int32_t result);
