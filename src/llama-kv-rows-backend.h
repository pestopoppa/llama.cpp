#pragma once

#include "ggml-backend.h"
#include <vector>

// Execute an already validated row-gather/flash-attention reference graph on a
// selected backend. The original graph stays host-owned; the context must be
// exclusively owned during this call because temporary buffer bindings are used. Refuse
// unsupported ops and invalid output bindings; no implicit CPU fallback.
bool llama_kv_rows_backend_compute(ggml_backend_t backend, ggml_context * host_ctx, ggml_cgraph * graph,
        const std::vector<ggml_tensor *> & outputs, std::vector<std::vector<float>> & values);
