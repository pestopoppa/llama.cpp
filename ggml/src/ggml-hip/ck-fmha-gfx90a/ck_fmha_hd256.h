#pragma once

// AK-CK-HD256: plain-C++ boundary between ggml (fattn-asm-gfx90a.cu) and the CK-tile fmha forward instances in
// ck_fmha_fwd_hd256_f16.cpp. Neither side includes the other's headers (CK-tile and ggml's common.cuh both define
// half/bf16 helpers and macros), so this header must stay dependency-free.

#include <hip/hip_runtime.h>

// Batch mode, B = 1. Element (not byte) strides; Q/K/V/O f16; V row-major (seq x hdim), D = 256 for Q, K, V and O.
struct ggml_ck_fmha_hd256_args {
    const void * q;
    const void * k;
    const void * v;
    void *       o;
    int          seqlen_q;        // n_q
    int          seqlen_k;        // effective KV length L
    int          nhead_q;
    int          nhead_k;         // nhead_q % nhead_k == 0 (GQA)
    int          stride_q;        // next query row
    int          nhead_stride_q;  // next head
    int          stride_k;
    int          nhead_stride_k;
    int          stride_v;
    int          nhead_stride_v;
    int          stride_o;
    int          nhead_stride_o;
    float        scale;           // softmax scale applied to Q.K^T
    int          causal;          // 0 = no mask, 1 = bottom-right aligned causal (row r sees [0, L - n_q + r])
};

// Enqueues the kernel on stream and returns hipSuccess, or returns the launch error (nothing written to o then).
hipError_t ggml_ck_fmha_fwd_hd256_f16(const ggml_ck_fmha_hd256_args & args, hipStream_t stream);

// Name of the instance that serves args.causal (for logs / content proof).
const char * ggml_ck_fmha_fwd_hd256_f16_name(int causal);
