# gfx90a FA-v3 forward asm code objects (AK-ASM-H5)

Two AMDGPU code objects (`.co`, ELF, code object v5, target `amdgcn-amd-amdhsa--gfx90a`) used by the
`GGML_HIP_FA_ASM_GFX90A` FLASH_ATTN_EXT fast path (`ggml/src/ggml-cuda/fattn-asm-gfx90a.cu`).
At configure time CMake verifies their sha256 and embeds the bytes into `libggml-hip` as arrays
(`fa-asm-gfx90a-blobs.cpp` in the build dir), so the build is relocatable; nothing reads these files at
run time. They are loaded with `hipModuleLoadData` on first use per device.

| File | Kernel symbol | sha256 |
|---|---|---|
| `fwd_hd128_bf16_rtne.gfx90a.co` | `_ZN5aiter24fmha_fwd_hd128_bf16_rtneE` | `404c2bf2bc814f068f6604c48416e27819061f7e49ac796dae7fb820c22661b3` |
| `fwd_hd128_bf16_causal_rtne.gfx90a.co` | `_ZN5aiter31fmha_fwd_hd128_bf16_causal_rtneE` | `f4cc9fce6c6cbc4cad4a5fd355f5b641dfde02c2ba99c383e8f9e4708d42725c` |

Provenance: AITER (MIT) `hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16{,_causal}_rtne.co` at
ROCm/aiter `3ba5ff7be51c04ad747a61da406cedf6c2d9c126`, retargeted to gfx90a offline by
`tools/retarget_fa.py` (sha256 `1822b68d…`) in the packet `/mnt/raid0/llm/tmp/ak-asm-h5-fa-retarget-20261105/`:
two MFMA opcode renames, a CFG-aware hazard audit at gfx90a MFMA pass counts with `s_nop` padding, a rebuilt
kernel descriptor, assembled with ROCm 6.2 `llvm-mc` + `ld.lld`. The packet's `IDENTITY.json` binds the same
digests. Verified on the MI210 against an fp32 SDPA reference (MHA and GQA 4, S 128/1024, causal and full)
and at S 8192/32768 (~114 TFLOPS); see that packet's `out/RESULT.json`.

Interface: packed 512-byte kernarg blob `fmha_fwd_v3_args` (aiter `csrc/include/mha_fwd.h:274`), block 512,
grid `(ceil(ceil(Sq/256)/tg), Hq, B)` with `tg = 2` for the causal kernel. Q/K/V/O bf16 with byte strides,
LSE fp32 `[B,H,Sq]`. The causal kernel is AITER's `mask_type 2` (bottom-right aligned) entry.
