#pragma once

#include "llama-kv-rows-plan.h"
#include "ggml.h"

// Experimental CPU reference graph for validating per-sequence row plans.
// The caller owns ctx, the tensors, and must schedule these graphs on CPU.
// One output per slice is returned; failure returns false and leaves out empty.
// This deliberately does not modify the production attention graph or select a
// backend. Rebuild the graph for each plan until graph-input refresh is wired.
bool llama_kv_rows_reference_graph(
        ggml_context * ctx, const llama_kv_rows_plan & plan,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        const std::vector<ggml_fp16_t> & full_mask, float scale,
        std::vector<ggml_tensor *> & out);
