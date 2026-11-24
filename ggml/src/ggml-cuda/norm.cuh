#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// AK-RF-27B-TAIL-FUSE G1 (GGML_CUDA_AK_ADD_RMS_NORM=1, default off): ADD + RMS_NORM + MUL(weight) in one launch.
bool     ggml_cuda_ak_add_rms_norm_enabled();
bool     ggml_cuda_ak_add_rms_norm_supported(const ggml_tensor * add, const ggml_tensor * rms_norm, const ggml_tensor * mul);
void     ggml_cuda_op_ak_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm,
                                            ggml_tensor * mul);
uint64_t ggml_cuda_ak_add_rms_norm_n_calls();

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
