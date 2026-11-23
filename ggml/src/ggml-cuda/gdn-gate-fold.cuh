#pragma once

// AK-BW-GDN-GATE-FOLD: fold the GDN beta/alpha projections and gate math into one launch per layer (see
// gdn-gate-fold-plan.h). GGML_HIP_GDN_GATE_FOLD=1 enables (default off; read per graph evaluation).

#include "common.cuh"

#include "gdn-gate-fold-plan.h"

bool     ggml_cuda_gdn_gate_fold_enabled();
void     ggml_cuda_gdn_gate_fold_plan(const ggml_cgraph * cgraph, ggml_cuda_gdn_gate_plan & plan);
void     ggml_cuda_gdn_gate_fold_run(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_gate_anchor & an);
uint64_t ggml_cuda_gdn_gate_fold_n_calls();
uint64_t ggml_cuda_gdn_gate_fold_n_unsafe();  // test witness: anchors the C7R guard refused since process start   // test witness: fold launches since process start
