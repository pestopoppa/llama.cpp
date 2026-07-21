//
// iqk port: link stubs for quant families our production stack does NOT use.
// These are NOT a substitute for required work — every quant the stack actually
// runs is FULLY ported with ik's real kernels:
//   Q4_K / Q5_K / Q6_K / Q2_K / Q3_K  -> iqk_gemm_kquants.cpp        (compiled, REAL)
//   Q8_0 / Q4_0 / Q5_0 / Q6_0 / Q4_1  -> iqk_gemm_legacy_quants.cpp  (compiled, REAL)
//   MoE mul_mat_id (iqk_mul_mat_moe / iqk_moe_fused_up_gate)         (real in iqk_mul_mat.cpp; Stage-2 hook)
// The registry shows ZERO use of IQ-quants / trellis / bitnet / float-mul-mat,
// so these families are stubbed to satisfy the linker (MulMat::prepare references
// them in its switch) and cleanly defer to v6's native kernel. If/when we adopt
// IQ-quants (e.g. future GLM IQ2), these MUST be replaced with the real ik kernels
// + their block types — do not leave them stubbed for a quant we deploy.
//
#include "iqk_config.h"

#if defined IQK_IMPLEMENT

#include "iqk_common.h"
#include "iqk_gemm_floats.h"
#include "iqk_gemm_iquants.h"
#include "iqk_gemm_iqk_quants.h"
#include "iqk_gemm_ktquants.h"
#include "iqk_gemm_1bit.h"

bool iqk_set_kernels_float(int, int, int, std::array<mul_mat_t, IQK_MAX_NY>&) { return false; }
// iqk_set_kernels_iquants: real kernel now compiled in (iqk_gemm_iquants.cpp) — stub removed 2026-07-21
bool iqk_set_kernels_iqk_quants(int, int, int, std::array<mul_mat_t, IQK_MAX_NY>&, mul_mat_t&) { return false; }
bool iqk_set_kernels_ktquants(int, int, int, std::array<mul_mat_t, IQK_MAX_NY>&, mul_mat_t&) { return false; }
bool iqk_set_kernels_1bit(int, int, int, std::array<mul_mat_t, IQK_MAX_NY>&, mul_mat_t&) { return false; }

// iqk_convert_iquants_q80_r8: real converter now compiled in (iqk_gemm_iquants.cpp) — stub removed 2026-07-21
bool iqk_convert_iqk_quants_q80_r8(int, int, const void *, size_t, void *, int) { return false; }
bool iqk_convert_1bit_q80_r8(int, int, const void *, size_t, void *, int) { return false; }

bool iqk_dequantize_ktquants(int, int, const void *, size_t, void *, size_t, int) { return false; }

#endif // IQK_IMPLEMENT
