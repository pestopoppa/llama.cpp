#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// AK-MMID-SPLIT16 (default off): GGML_CUDA_MMVQ_MOE_SPLIT16=1 extends the multi-token MUL_MAT_ID MMVQ MoE path
// from ne2 <= 8 to ne2 <= 16 by launching the unchanged 8-column MoE kernel once per <= 8 token chunk, instead of
// falling through to MMQ_MMID (which also disables CUDA graphs). Only for types whose mmid cap is the full 8.
#define MMVQ_MOE_SPLIT16_MAX_BATCH 16
bool ggml_cuda_mmvq_moe_split16_enabled();
// Effective mmid batch cap: get_mmvq_mmid_max_batch(), raised to 16 when the knob is on and the type cap is 8.
int ggml_cuda_mmvq_mmid_effective_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
