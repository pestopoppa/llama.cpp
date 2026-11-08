// AK-CK-HD256: CK-tile fmha forward instances for FLASH_ATTN_EXT head_dim 256 on gfx90a (fattn-asm-gfx90a.cu).
//
// Hand-instantiated equivalent of composable_kernel rocm-6.2.4 (6d90a7cf2eb64f6c956460ea94341e32ab4de7a5)
// example/ck_tile/01_fmha/generate.py FMHA_FWD_KERNEL_BODY for exactly two kernels:
//   hdim 256, fp16, batch mode, tile b128x128x32x256x32x256 r4x1x1 w32x32x16 (generate.py tile dict, '256'),
//   pipeline qr (QRKSVS; generate.py uses qr for hdim 256), V row-major, no bias, no LSE, no fp8 quant,
//   padding spad=t skpad=t dpad=f dvpad=f (arbitrary n_q / L, D fixed at 256: full-width vector loads on D),
//   mask SimplifiedGenericAttentionMask<false> (no mask) and <true> (causal, MASK_FROM_BOTTOM_RIGHT, left -1 right 0).
// Compiled with the ROCm 6.2 toolchain against the CK headers named by GGML_HIP_CK_FMHA_DIR (CMake); the trunk CK in
// aiter's 3rdparty needs gfx950 builtins that ROCm 6.2's clang lacks, the rocm-6.2.4 tag compiles cleanly.

#include "ck_fmha_hd256.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "fmha_fwd.hpp" // example/ck_tile/01_fmha: FmhaFwdTypeConfig, fmha_fwd_args, fmha_fwd_create_kargs_and_grids

namespace {

using ck_dtype = ck_tile::half_t;
using ck_cfg   = FmhaFwdTypeConfig<ck_dtype>;

constexpr bool k_spad  = true;
constexpr bool k_skpad = true;
constexpr bool k_dpad  = false;
constexpr bool k_dvpad = false;

using ck_block_tile  = ck_tile::sequence<128, 128, 32, 256, 32, 256>;
using ck_block_warps = ck_tile::sequence<4, 1, 1>;
using ck_warp_tile   = ck_tile::sequence<32, 32, 16>;
using ck_shape       = ck_tile::TileFmhaShape<ck_block_tile, ck_block_warps, ck_warp_tile, ck_block_warps, ck_warp_tile,
                                              /*IsVLayoutRowMajor=*/true>;
using ck_traits      = ck_tile::TileFmhaTraits<k_spad, k_skpad, k_dpad, k_dvpad, ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                               /*kStoreLSE=*/false, /*kDoFp8StaticQuant=*/false, /*kBlockPerCu=*/-1>;

template <bool kMasking>
struct ck_instance {
    using mask = ck_tile::SimplifiedGenericAttentionMask<kMasking>;
    using problem = ck_tile::BlockFmhaPipelineProblem<
        typename ck_cfg::QDataType, typename ck_cfg::KDataType, typename ck_cfg::VDataType,
        typename ck_cfg::SaccDataType, typename ck_cfg::SMPLComputeDataType, typename ck_cfg::BiasDataType,
        typename ck_cfg::LSEDataType, typename ck_cfg::PDataType, typename ck_cfg::OaccDataType,
        typename ck_cfg::ODataType, ck_shape, /*kIsGroupMode=*/false, mask, ck_traits>;
    using pipeline = ck_tile::BlockFmhaPipelineQRKSVS<problem>;
    using epilogue = ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
        typename ck_cfg::OaccDataType, typename ck_cfg::ODataType, k_spad, k_dvpad>>;
    using kernel = ck_tile::FmhaFwdKernel<ck_tile::FmhaFwdTilePartitioner<ck_shape>, pipeline, epilogue>;

    static hipError_t run(const fmha_fwd_args & a, hipStream_t stream) {
        using k_ = kernel;
        auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<k_>(a);
        constexpr dim3 blocks = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        (void) hipGetLastError();
        ck_tile::launch_kernel<blocks.x, kBlockPerCu>(ck_tile::stream_config{stream, false}, k_{}, grids, blocks, 0, kargs);
        return hipGetLastError();
    }

    // C8-4-H18: the same kernel from the embedded clang-24 code object (gen_co.sh). kentry<256, 1, kernel, Kargs>
    // takes (kernel f [empty, offset 0, size 1], Kargs kargs [offset 8]); kernargs are built from these headers.
    static hipError_t run_co(const fmha_fwd_args & a, hipFunction_t fn, hipStream_t stream) {
        using k_ = kernel;
        static_assert(k_::BlockSize().x == 256, "kentry<256,...> in the code object");
        static_assert(sizeof(typename k_::Kargs) == (kMasking ? 120 : 104), "kernarg layout differs from the code object");
        auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<k_>(a);
        k_ f{};
        void * params[2] = { &f, &kargs };
        return hipModuleLaunchKernel(fn, grids.x, grids.y, grids.z, k_::BlockSize().x, 1, 1, 0, stream, params, nullptr);
    }
};

} // namespace

const char * ggml_ck_fmha_fwd_hd256_f16_name(int causal) {
    return causal ? "ck_tile fmha_fwd d256 fp16 batch b128x128x32x256x32x256 qr row pssk mask(bottom-right causal) [ck rocm-6.2.4]"
                  : "ck_tile fmha_fwd d256 fp16 batch b128x128x32x256x32x256 qr row pssk nmask [ck rocm-6.2.4]";
}

const char * ggml_ck_fmha_fwd_hd256_f16_co_kernel(int causal) {
    // .name of the two kernels in ck_fmha_fwd_hd256_f16.gfx90a.co (llvm-readelf --notes)
    return causal
        ? "_ZN7ck_tile6kentryILi256ELi1ENS_13FmhaFwdKernelINS_22FmhaFwdTilePartitionerINS_13TileFmhaShapeINS_8sequenceIJLi128ELi128ELi32ELi256ELi32ELi256EEEENS4_IJLi4ELi1ELi1EEEENS4_IJLi32ELi32ELi16EEEES6_S7_Lb1EEEEENS_23BlockFmhaPipelineQRKSVSINS_24BlockFmhaPipelineProblemIDF16_DF16_DF16_ffDF16_fDF16_fDF16_S8_Lb0ENS_30SimplifiedGenericAttentionMaskILb1EEENS_14TileFmhaTraitsILb1ELb1ELb0ELb0ELNS_22BlockAttentionBiasEnumE0ELb0ELb0ELin1EEEEENS_35BlockFmhaPipelineQXKSVSCustomPolicyILb1ELb0ELb0ELi1ELi1EEEEENS_17Default2DEpilogueINS_24Default2DEpilogueProblemIfDF16_Lb1ELb0EEEvEEEEJNSP_21FmhaFwdBatchModeKargsEEEEvT1_DpT2_"
        : "_ZN7ck_tile6kentryILi256ELi1ENS_13FmhaFwdKernelINS_22FmhaFwdTilePartitionerINS_13TileFmhaShapeINS_8sequenceIJLi128ELi128ELi32ELi256ELi32ELi256EEEENS4_IJLi4ELi1ELi1EEEENS4_IJLi32ELi32ELi16EEEES6_S7_Lb1EEEEENS_23BlockFmhaPipelineQRKSVSINS_24BlockFmhaPipelineProblemIDF16_DF16_DF16_ffDF16_fDF16_fDF16_S8_Lb0ENS_30SimplifiedGenericAttentionMaskILb0EEENS_14TileFmhaTraitsILb1ELb1ELb0ELb0ELNS_22BlockAttentionBiasEnumE0ELb0ELb0ELin1EEEEENS_35BlockFmhaPipelineQXKSVSCustomPolicyILb1ELb0ELb0ELi1ELi1EEEEENS_17Default2DEpilogueINS_24Default2DEpilogueProblemIfDF16_Lb1ELb0EEEvEEEEJNSP_21FmhaFwdBatchModeKargsEEEEvT1_DpT2_";
}

hipError_t ggml_ck_fmha_fwd_hd256_f16(const ggml_ck_fmha_hd256_args & p, hipStream_t stream) {
    return ggml_ck_fmha_fwd_hd256_f16_co(p, nullptr, stream);
}

hipError_t ggml_ck_fmha_fwd_hd256_f16_co(const ggml_ck_fmha_hd256_args & p, hipFunction_t fn, hipStream_t stream) {
    fmha_fwd_args a{};
    a.q_ptr             = p.q;
    a.k_ptr             = p.k;
    a.v_ptr             = p.v;
    a.bias_ptr          = nullptr;
    a.lse_ptr           = nullptr;
    a.o_ptr             = p.o;
    a.seqstart_q_ptr    = nullptr;
    a.seqstart_k_ptr    = nullptr;
    a.seqlen_k_ptr      = nullptr;
    a.seqlen_q          = p.seqlen_q;
    a.seqlen_k          = p.seqlen_k;
    a.batch             = 1;
    a.max_seqlen_q      = p.seqlen_q;
    a.hdim_q            = 256;
    a.hdim_v            = 256;
    a.nhead_q           = p.nhead_q;
    a.nhead_k           = p.nhead_k;
    a.scale_s           = p.scale;
    a.scale_p           = 1.0f;
    a.scale_o           = 1.0f;
    a.stride_q          = p.stride_q;
    a.stride_k          = p.stride_k;
    a.stride_v          = p.stride_v;
    a.stride_bias       = 0;
    a.stride_o          = p.stride_o;
    a.nhead_stride_q    = p.nhead_stride_q;
    a.nhead_stride_k    = p.nhead_stride_k;
    a.nhead_stride_v    = p.nhead_stride_v;
    a.nhead_stride_bias = 0;
    a.nhead_stride_lse  = 0;
    a.nhead_stride_o    = p.nhead_stride_o;
    // B = 1: batch strides are never applied (i_batch == 0); set them to the full extents anyway
    a.batch_stride_q    = p.stride_q * p.seqlen_q;
    a.batch_stride_k    = p.stride_k * p.seqlen_k;
    a.batch_stride_v    = p.stride_v * p.seqlen_k;
    a.batch_stride_bias = 0;
    a.batch_stride_lse  = 0;
    a.batch_stride_o    = p.stride_o * p.seqlen_q;
    if (p.causal) {
        a.window_size_left  = -1;
        a.window_size_right = 0;
        a.mask_type         = static_cast<ck_tile::index_t>(ck_tile::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT);
        return fn ? ck_instance<true>::run_co(a, fn, stream) : ck_instance<true>::run(a, stream);
    }
    a.window_size_left  = -1;
    a.window_size_right = -1;
    a.mask_type         = static_cast<ck_tile::index_t>(ck_tile::GenericAttentionMaskEnum::NO_MASK);
    return fn ? ck_instance<false>::run_co(a, fn, stream) : ck_instance<false>::run(a, stream);
}
