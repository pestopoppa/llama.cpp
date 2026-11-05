#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "mmvq.cuh"
#include "unary.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}


// AK-ASM-H8 arm (c), technique 1 of 3: expert-density column tiling (after AITER moe_sorting, which pads each
// expert's token run to the GEMM block size chosen for the *per-expert* token count, not the batch size).
// Upstream picks the MMQ column tile J from ncols_max = n_tokens, so at n_tokens=512 on Qwen3.6-35B-A3B
// (256 experts, top-8, mean 16 tokens/expert) every expert tile is J=128 wide for ~16 live columns: the MFMA
// work is ~8x the useful work. Here J is chosen from the mean tokens per expert times a headroom factor.
// Numerics: identical per-column dot products (same Q8_0 x / q8_1 y blocks, fp32 accumulation); only the
// stream-k K-split boundaries, and hence fp32 summation order of partial tiles, can move.
// Env: GGML_HIP_MMID_H8=0 restores upstream selection (kill switch); GGML_HIP_MMID_J_HEADROOM=<float>, default 1.5.
static bool ggml_cuda_mmid_h8_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_HIP_MMID_H8");
        return !(e && e[0] == '0');
    }();
    return enabled;
}

static float ggml_cuda_mmid_j_headroom() {
    static const float h = [] {
        const char * e = getenv("GGML_HIP_MMID_J_HEADROOM");
        const float v = e ? (float) atof(e) : 1.5f;
        return v >= 1.0f && v <= 64.0f ? v : 1.5f;
    }();
    return h;
}

static int64_t ggml_cuda_mmid_tile_hint(const int64_t n_tokens, const int64_t n_expert_used, const int64_t n_experts,
                                        const int64_t nrows_x, const int64_t ncols_x) {
    if (!ggml_cuda_mmid_h8_enabled() || n_experts <= 1) {
        return 0;
    }
    const double mean = double(n_tokens*n_expert_used) / double(n_experts);
    int64_t hint = (int64_t) (mean*ggml_cuda_mmid_j_headroom() + 0.999);
    hint = std::max<int64_t>(hint, 8);
    if (hint >= n_tokens) {
        return 0; // no narrower tile than upstream would pick
    }
    // The grid still spans n_tokens, now with narrower tiles: keep the stream-k kbc space far from int overflow
    // (launch_mul_mat_q asserts ntiles*blocks_per_ne00 < 2^30; I >= 16, qk >= 32).
    const int64_t ntx_worst = (n_tokens + 7) / 8;
    const int64_t work = ntx_worst * n_experts * ((nrows_x + 15) / 16) * ((ncols_x + 31) / 32);
    if (work >= (int64_t(1) << 29)) {
        return 0;
    }
    return hint;
}

// AK-ASM-H8 arm (c), technique 2 of 3: gate+up+GLU for the MMQ-id (prefill/verify) regime. One expert sort and one
// src1 quantization feed both GEMMs (upstream runs mm_ids_helper + quantize once per GEMM); up is written straight
// into the GLU output and the activation is applied in place, so the separate GLU op and the up intermediate are
// gone. Same MMQ kernels and the same op_silu/op_gelu formulas as unary.cu, so numerics match the unfused chain.
// Env: GGML_HIP_MMID_FUSED_GLU=0 disables (falls back to the unfused ops); GGML_HIP_MMID_H8=0 disables everything.
static bool ggml_cuda_mmid_fused_glu_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_HIP_MMID_FUSED_GLU");
        return !(e && e[0] == '0');
    }();
    return enabled && ggml_cuda_mmid_h8_enabled();
}

template <ggml_glu_op op>
static __global__ void mmid_glu_inplace_kernel(const float * __restrict__ g, float * __restrict__ x, const int64_t n) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float gi = g[i];
    float a;
    if constexpr (op == GGML_GLU_OP_SWIGLU) {
        a = ggml_cuda_op_silu_single(gi);
    } else {
        a = ggml_cuda_op_gelu_single(gi);
    }
    x[i] = a * x[i];
}

bool ggml_cuda_should_fuse_mmq_id_glu(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int cc) {
    if (!ggml_cuda_mmid_fused_glu_enabled()) {
        return false;
    }
    if (up->op != GGML_OP_MUL_MAT_ID || gate->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    if (ggml_get_op_params_i32(glu, 1) != 0 || glu->src[0] != gate || glu->src[1] != up) {
        return false;
    }
    const ggml_tensor * src0 = up->src[0];
    const ggml_tensor * src1 = up->src[1];
    if (!ggml_is_quantized(src0->type) || src0->type != gate->src[0]->type || !ggml_are_same_shape(src0, gate->src[0]) ||
            !ggml_are_same_stride(src0, gate->src[0])) {
        return false;
    }
    if (src1 != gate->src[1] || up->src[2] != gate->src[2] || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || !ggml_is_contiguous(up) || !ggml_are_same_shape(glu, up)) {
        return false;
    }
    // weights only: the compute-buffer padding clear in ggml_cuda_mul_mat_q is done for src0 alone
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_backend_buffer_get_usage(gate->src[0]->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    // only the regime ggml_cuda_mul_mat_id routes to MMQ (MMVQ-id has its own fused gate/up path)
    const int64_t n_tokens = src1->ne[2];
    if (n_tokens <= MMVQ_MAX_BATCH_SIZE && n_tokens <= get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }
    if (!ggml_cuda_should_use_mmq(src0->type, cc, n_tokens, src0->ne[2])) {
        return false;
    }
    if (blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4)) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_tensor * gate, const ggml_glu_op glu_op) {
    GGML_ASSERT(!gate || ids);
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (dedup_bcast) {
            // quantize each token once, scatter its block to all n_expert_used slots
            if (use_native_fp4) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            }
        } else if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12};
    args.ncols_tile_hint = ggml_cuda_mmid_tile_hint(ne12, n_expert_used, ne02, ne01, ne00);

    if (!gate) {
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    // fused gate+up+GLU: dst is the GLU output (same shape/layout as the up GEMM output)
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int64_t n_dst = ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> gate_out(ctx.pool(), n_dst);

    mmq_args args_gate = args;
    args_gate.x   = (const char *) gate->data;
    args_gate.dst = gate_out.get();
    ggml_cuda_mul_mat_q_switch_type(ctx, args_gate, stream);
    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream); // up -> dst

    const int nth = 256;
    const dim3 nblk((n_dst + nth - 1) / nth, 1, 1);
    if (glu_op == GGML_GLU_OP_SWIGLU) {
        mmid_glu_inplace_kernel<GGML_GLU_OP_SWIGLU><<<nblk, nth, 0, stream>>>(gate_out.get(), dst_d, n_dst);
    } else {
        GGML_ASSERT(glu_op == GGML_GLU_OP_GEGLU);
        mmid_glu_inplace_kernel<GGML_GLU_OP_GEGLU><<<nblk, nth, 0, stream>>>(gate_out.get(), dst_d, n_dst);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
