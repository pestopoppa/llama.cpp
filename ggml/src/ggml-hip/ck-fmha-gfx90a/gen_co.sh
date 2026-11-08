#!/bin/bash
# C8-4-H18: regenerate ck_fmha_fwd_hd256_f16.gfx90a.co (see ck_fmha_fwd_hd256_f16_co.cpp). Fails unless both kernels
# have private_segment_fixed_size 0 and no VGPR/SGPR spills. Then update the sha256 in ggml/src/ggml-hip/CMakeLists.txt.
#   gen_co.sh <rocm root with clang >= 23> <composable_kernel rocm-6.2.4 checkout> [out.co]
# The committed object was built with /mnt/raid0/llm/rocm-10.1.0-q38p/install (AMD clang 24.0.0git d6f6cb691863).
set -euo pipefail
R=${1:?rocm root}
CK=${2:?ck checkout}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${3:-$HERE/ck_fmha_fwd_hd256_f16.gfx90a.co}
[ "$(git -C "$CK" rev-parse HEAD)" = 6d90a7cf2eb64f6c956460ea94341e32ab4de7a5 ] || { echo "CK checkout is not rocm-6.2.4" >&2; exit 1; }
env -u CPATH -u CPLUS_INCLUDE_PATH "$R/lib/llvm/bin/clang++" -x hip --rocm-path="$R" --hip-path="$R" --offload-arch=gfx90a \
    --cuda-device-only --no-gpu-bundle-output -mcode-object-version=5 -O3 -std=c++17 \
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -fgpu-flush-denormals-to-zero -mllvm -enable-post-misched=0 -w \
    -I"$CK/include" -I"$CK/example/ck_tile/01_fmha" -I"$HERE" "$HERE/ck_fmha_fwd_hd256_f16_co.cpp" -o "$OUT"
NOTES=$("$R/lib/llvm/bin/llvm-readelf" --notes "$OUT")
[ "$(grep -c 'private_segment_fixed_size: *0$' <<<"$NOTES")" = 2 ] || { echo "scratch in $OUT" >&2; grep -E 'private_segment|spill' <<<"$NOTES" >&2; exit 1; }
[ "$(grep -cE '(vgpr|sgpr)_spill_count: *0$' <<<"$NOTES")" = 4 ] || { echo "spills in $OUT" >&2; exit 1; }
grep -E '\.name:|private_segment|spill_count|vgpr_count|agpr_count|kernarg_segment_size|group_segment' <<<"$NOTES"
sha256sum "$OUT"
