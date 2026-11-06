# gfx90a topksoftmax asm code objects (AK-ASM-H11)

Seven AMDGPU code objects (`.co`, target `amdgcn-amd-amdhsa--gfx90a`) used by the `GGML_HIP_TOPK_ASM_GFX90A`
fast path of the topk-moe fusion (`ggml/src/ggml-cuda/topk-moe-asm-gfx90a.cu`). CMake verifies their sha256 at
configure time and embeds the bytes into `libggml-hip` (`topk-asm-gfx90a-blobs.cpp` in the build dir); nothing reads
these files at run time. They are loaded with `hipModuleLoadData` on first use per device.

| File | Kernel symbol | sha256 |
|---|---|---|
| `topksoftmax__topksoftmax_4x128x4.gfx90a.co` | `_ZN5aiter19topksoftmax_4x128x4E` | `667136718fd747c109e3608277bdc4c4ec03717b64a9d41d4d097b5b9b03c6e5` |
| `topksoftmax__topksoftmax_4x128x6.gfx90a.co` | `_ZN5aiter19topksoftmax_4x128x6E` | `3b7d38ac0318731d1401c58c4e5fb0582db33c0dc4bf76bac9e4edcf19751682` |
| `topksoftmax__topksoftmax_4x128x8.gfx90a.co` | `_ZN5aiter19topksoftmax_4x128x8E` | `75e3aa9b95a85988456c88f2470e5541c666733219a8b6cec2483fd77a50c0e1` |
| `topksoftmax__topksoftmax_4x256x6.gfx90a.co` | `_ZN5aiter19topksoftmax_4x256x6E` | `d0087b4fc4d86c8f23f5102206cff28b6295b6312a406df9413d7ed1d6a020ab` |
| `topksoftmax__topksoftmax_4x256x8.gfx90a.co` | `_ZN5aiter19topksoftmax_4x256x8E` | `a3723afaa7e51598c7af83df715fab4a4b80460c6c4fbbe534e53d7d767c57fc` |
| `topksoftmax__topksoftmax_4x384x8.gfx90a.co` | `_ZN5aiter19topksoftmax_4x384x8E` | `10bd0008e9a6804e22bea551777d2cd3d035ccf8d100880f622e8278f6d9a253` |
| `topksoftmax__topksoftmax_4x640x8.gfx90a.co` | `_ZN5aiter19topksoftmax_4x640x8E` | `43fcc6e2e21dbbea0b7572fa460759f56579a265354989d3a75da6a542349594` |

Provenance: AITER (MIT) `hsa/gfx942/topksoftmax/*.co` retargeted to gfx90a by the AK-ASM-T1 batch
(`/mnt/raid0/llm/tmp/ak-asm-t1-20261105/batch_out/`, identical bytes in `aiter-src-90a/hsa/gfx90a/topksoftmax/`).
The AK-ASM-T1 harness (`/mnt/raid0/llm/tmp/ak-asm-t1-harness-20261105/out/results.jsonl`, unit `topksoftmax`)
verified the **bf16** siblings `4x128x8_bf16` / `4x256x8_bf16` on the MI210 (12/12 checks, max_abs 1.2e-7); the fp32
objects embedded here come from the same retarget pass but were not exercised by that harness, so the
test-backend-ops TOPK_MOE corpus is their first hardware correctness check.

Interface (AITER `csrc/py_itfs_cu/asm_topksoftmax.cu`): 128-byte kernarg blob of 16-byte slots
`T, W, A, batch, expert, topk, renormalize, out_stride(bytes)`, block 256, grid `ceil(T / 4)`. fp32 logits `[T, E]`,
int32 ids and fp32 weights written at `t * out_stride`. No kernel exists for 512 experts / top-10 (Q38FN).
