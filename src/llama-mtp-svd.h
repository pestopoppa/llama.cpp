#pragma once

#include "ggml.h"

#include <cstdint>

// DRAFT-HEAD-SVD-CPU factors (llama-mtp-svd.cpp). v: [d, r] F32 (B^T); a: [r, V] (A = W_eff B); C: candidates rescored exactly.
struct llama_mtp_svd {
    ggml_tensor * v;
    ggml_tensor * a;
    int64_t       C;
};

// nullptr unless LLAMA_MTP_DRAFT_HEAD_SVD names a readable factor file whose shapes match head_w and head_w is a plain host
// tensor (the exact rescoring gathers its rows verbatim). Loaded once per process.
const llama_mtp_svd * llama_mtp_svd_get(const ggml_tensor * head_w);
