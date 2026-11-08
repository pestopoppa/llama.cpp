// C8-4-H18: device-only translation unit for the spill-free hd256 code object (ck_fmha_fwd_hd256_f16.gfx90a.co).
// NOT part of the CMake build. gen_co.sh compiles it with a clang >= 23 (TheRock / ROCm 7+ llvm) and the SAME
// composable_kernel rocm-6.2.4 headers and flags as ck_fmha_fwd_hd256_f16.cpp; the library embeds the result and
// launches it with hipModuleLaunchKernel (ck_fmha_fwd_hd256_f16.cpp, ggml_ck_fmha_fwd_hd256_f16_co), building the
// kernargs from those same headers on the host.
// Why: ROCm 6.2 clang allocates these instances with 116-120 B/lane of scratch (occupancy 1); clang 24 allocates
// 345 unified VGPRs (89 AGPR) with private_segment_fixed_size 0 and no spills.
//
// Newer HIP headers make warpSize a non-constexpr device variable; ck_tile::get_warp_size() (rocm-6.2.4) needs a
// constant expression. gfx90a is wave64, so pin it while arch.hpp is parsed (its include guard keeps it pinned).
#include <hip/hip_runtime.h>
#define warpSize 64
#include "ck_tile/core/arch/arch.hpp"
#undef warpSize
#include "ck_fmha_fwd_hd256_f16.cpp"
