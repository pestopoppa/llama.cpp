#pragma once

#include "ggml.h"

// Q38FN-GEMV2 DRAFTVOCAB (llama-mtp-vocab.cpp). w_sub: [d, K] rows of the draft head, verbatim; idx: [K] I32 token ids.
struct llama_mtp_vocab {
    ggml_tensor * w_sub;
    ggml_tensor * idx;
};

// nullptr unless LLAMA_MTP_DRAFT_VOCAB names a readable token-id list (one id per line) and head_w is a plain host
// tensor. Built once per process on first use (LLAMA_MTP_DRAFT_VOCAB_K caps the list, default 32768).
const llama_mtp_vocab * llama_mtp_vocab_get(const ggml_tensor * head_w);
