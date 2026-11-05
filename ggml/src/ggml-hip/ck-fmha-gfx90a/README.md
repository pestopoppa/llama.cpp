# gfx90a CK-tile fmha forward, head_dim 256 (AK-CK-HD256)

`GGML_HIP_CK_FMHA_HD256=ON` compiles `ck_fmha_fwd_hd256_f16.cpp` into `libggml-hip` and lets the gfx90a FA fast-path
dispatcher (`ggml/src/ggml-cuda/fattn-asm-gfx90a.cu`, which also carries the hd128 asm path) send
FLASH_ATTN_EXT with D = 256 to a CK-tile fmha forward kernel. Requires `GGML_HIP_FA_ASM_GFX90A=ON`.

## Source of the kernels

Header-only: composable_kernel tag `rocm-6.2.4` (commit `6d90a7cf2eb64f6c956460ea94341e32ab4de7a5`, MIT), passed as
`-DGGML_HIP_CK_FMHA_DIR=<checkout>`. CMake refuses a checkout whose HEAD differs or whose tracked files are dirty
(`-DGGML_HIP_CK_FMHA_COMMIT=` disables the check). Nothing is vendored into this tree.

```bash
git clone --depth 1 --branch rocm-6.2.4 https://github.com/ROCm/composable_kernel <checkout>
```

Why not the CK in aiter's `3rdparty/` (the one behind the AK-CK-HD256 speed evidence): trunk CK (Oct 2026) uses
gfx950 builtins (`__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`, ...) that ROCm 6.2's clang does not declare;
`rocm-6.2.4` compiles cleanly with the production toolchain (ROCm 6.2.0, AMD clang 18).

## Instances (two)

Hand-instantiated equivalent of that tag's `example/ck_tile/01_fmha/generate.py` `FMHA_FWD_KERNEL_BODY`
(the generator itself is not run at build time):

| | |
|---|---|
| dtype | fp16 (Q/K/V/P/O), S/O accumulate fp32 |
| mode | batch, B = 1 |
| tile | `b128x128x32x256x32x256`, warps `4x1x1`, warp tile `32x32x16` (generate.py tile dict for hdim 256) |
| pipeline | `qr` (`BlockFmhaPipelineQRKSVS`; generate.py uses qr for hdim 256), V row-major |
| padding | `spad=t skpad=t dpad=f dvpad=f` (any n_q and L; D fixed, 16-byte vector loads along D) |
| bias / LSE / fp8 | none / not stored / off |
| mask | `SimplifiedGenericAttentionMask<false>` and `<true>` with `MASK_FROM_BOTTOM_RIGHT`, left -1, right 0 |
| flags | `-DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero -mllvm -enable-post-misched=0` (CK example) |

## Known risk (for the GPU gate)

ROCm 6.2 register allocation for these instances: 256 VGPR + 256 AGPR, occupancy 1, **scratch 116-120 B/lane**
(`-Rpass-analysis=kernel-resource-usage`). The aiter-built bf16 d256 kernels that produced the 4.96x evidence
(newer compiler, trunk CK) use ~330 registers and **no scratch**. The spill may cost part of that speedup; only the
GPU perf gate can say how much. If it does, the fallback is to embed the aiter-built gfx90a d256 code objects and
launch them with `hipModuleLoadData` (as the hd128 asm path does), at the cost of bf16 K/V conversion per call and a
hand-mirrored trunk `FmhaFwdBatchModeKargs` layout.
