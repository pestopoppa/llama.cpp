if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

# AK-ASM-H5: FLASH_ATTN_EXT fast path on gfx90a (MI210) through the retargeted AITER FA-v3 fwd hd128 bf16 asm
# kernels (fa-asm-gfx90a/README.md). Default ON only when gfx90a is among the HIP targets; the runtime check
# (device cc == CDNA2) still gates every launch, so a multi-arch build is safe either way.
set(GGML_HIP_FA_ASM_GFX90A_DEFAULT OFF)
if ("${CMAKE_HIP_ARCHITECTURES};${GPU_TARGETS};${AMDGPU_TARGETS}" MATCHES "gfx90a")
    set(GGML_HIP_FA_ASM_GFX90A_DEFAULT ON)
endif()
option(GGML_HIP_FA_ASM_GFX90A "ggml: gfx90a AITER FA-v3 hd128 bf16 asm fast path for FLASH_ATTN_EXT" ${GGML_HIP_FA_ASM_GFX90A_DEFAULT})

# AK-CK-HD256: head_dim 256 FLASH_ATTN_EXT on gfx90a through CK-tile fmha forward instances (ck-fmha-gfx90a/), compiled
# from the composable_kernel headers in GGML_HIP_CK_FMHA_DIR (a checkout of tag rocm-6.2.4, commit pinned below: the
# CK trunk needs gfx950 builtins ROCm 6.2's clang does not have). Rides on GGML_HIP_FA_ASM_GFX90A (same dispatcher).
option(GGML_HIP_CK_FMHA_HD256 "ggml: gfx90a CK-tile fmha fwd hd256 fp16 path for FLASH_ATTN_EXT (needs GGML_HIP_CK_FMHA_DIR)" OFF)
set(GGML_HIP_CK_FMHA_DIR "" CACHE PATH "ggml: composable_kernel source checkout (tag rocm-6.2.4) for GGML_HIP_CK_FMHA_HD256")
set(GGML_HIP_CK_FMHA_COMMIT "6d90a7cf2eb64f6c956460ea94341e32ab4de7a5" CACHE STRING "ggml: required git HEAD of GGML_HIP_CK_FMHA_DIR (empty = do not check)")

# AK-ASM-H11: MoE router softmax+top-k (topk-moe fusion) on gfx90a (MI210) through the retargeted AITER
# topksoftmax asm kernels (topk-asm-gfx90a/README.md). Default ON only when gfx90a is among the HIP targets; the
# runtime check (device cc == CDNA2) still gates every launch.
set(GGML_HIP_TOPK_ASM_GFX90A_DEFAULT OFF)
if ("${CMAKE_HIP_ARCHITECTURES};${GPU_TARGETS};${AMDGPU_TARGETS}" MATCHES "gfx90a")
    set(GGML_HIP_TOPK_ASM_GFX90A_DEFAULT ON)
endif()
option(GGML_HIP_TOPK_ASM_GFX90A "ggml: gfx90a AITER topksoftmax asm fast path for the topk-moe fusion" ${GGML_HIP_TOPK_ASM_GFX90A_DEFAULT})

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
endif()

if (GGML_HIP_FA_ASM_GFX90A)
    # Embed the two code objects as byte arrays (relocatable store: nothing is read from disk at run time).
    # The digests bind the bytes to the AK-ASM-H5 packet's IDENTITY.json; a mismatch is a configure error.
    set(GGML_HIP_FA_ASM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fa-asm-gfx90a)
    set(GGML_HIP_FA_ASM_GEN ${CMAKE_CURRENT_BINARY_DIR}/fa-asm-gfx90a-blobs.cpp)
    set(GGML_HIP_FA_ASM_SRC
        "// generated by ggml/src/ggml-hip/CMakeLists.txt (GGML_HIP_FA_ASM_GFX90A); do not edit\n#include <stddef.h>\n\n")
    foreach(GGML_HIP_FA_ASM_ENTRY
            "fwd_hd128_bf16_rtne|404c2bf2bc814f068f6604c48416e27819061f7e49ac796dae7fb820c22661b3"
            "fwd_hd128_bf16_causal_rtne|f4cc9fce6c6cbc4cad4a5fd355f5b641dfde02c2ba99c383e8f9e4708d42725c")
        string(REPLACE "|" ";" GGML_HIP_FA_ASM_PAIR "${GGML_HIP_FA_ASM_ENTRY}")
        list(GET GGML_HIP_FA_ASM_PAIR 0 GGML_HIP_FA_ASM_NAME)
        list(GET GGML_HIP_FA_ASM_PAIR 1 GGML_HIP_FA_ASM_SHA_WANT)
        set(GGML_HIP_FA_ASM_CO ${GGML_HIP_FA_ASM_DIR}/${GGML_HIP_FA_ASM_NAME}.gfx90a.co)
        file(SHA256 ${GGML_HIP_FA_ASM_CO} GGML_HIP_FA_ASM_SHA)
        if (NOT GGML_HIP_FA_ASM_SHA STREQUAL GGML_HIP_FA_ASM_SHA_WANT)
            message(FATAL_ERROR "GGML_HIP_FA_ASM_GFX90A: ${GGML_HIP_FA_ASM_CO} sha256 ${GGML_HIP_FA_ASM_SHA} != ${GGML_HIP_FA_ASM_SHA_WANT}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GGML_HIP_FA_ASM_CO})
        file(READ ${GGML_HIP_FA_ASM_CO} GGML_HIP_FA_ASM_HEX HEX)
        string(LENGTH "${GGML_HIP_FA_ASM_HEX}" GGML_HIP_FA_ASM_HEX_LEN)
        math(EXPR GGML_HIP_FA_ASM_NBYTES "${GGML_HIP_FA_ASM_HEX_LEN} / 2")
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," GGML_HIP_FA_ASM_BYTES "${GGML_HIP_FA_ASM_HEX}")
        string(APPEND GGML_HIP_FA_ASM_SRC
            "// ${GGML_HIP_FA_ASM_NAME}.gfx90a.co sha256 ${GGML_HIP_FA_ASM_SHA}\n"
            "extern \"C\" alignas(256) const unsigned char ggml_hip_fa_asm_${GGML_HIP_FA_ASM_NAME}[${GGML_HIP_FA_ASM_NBYTES}] = {${GGML_HIP_FA_ASM_BYTES}};\n"
            "extern \"C\" const size_t ggml_hip_fa_asm_${GGML_HIP_FA_ASM_NAME}_size = ${GGML_HIP_FA_ASM_NBYTES};\n"
            "extern \"C\" const char ggml_hip_fa_asm_${GGML_HIP_FA_ASM_NAME}_sha256[] = \"${GGML_HIP_FA_ASM_SHA}\";\n\n")
    endforeach()
    # write through a temp file so an unchanged blob does not retrigger the compile
    file(WRITE ${GGML_HIP_FA_ASM_GEN}.tmp "${GGML_HIP_FA_ASM_SRC}")
    configure_file(${GGML_HIP_FA_ASM_GEN}.tmp ${GGML_HIP_FA_ASM_GEN} COPYONLY)
    target_sources(ggml-hip PRIVATE ${GGML_HIP_FA_ASM_GEN})
    add_compile_definitions(GGML_HIP_FA_ASM_GFX90A)
    message(STATUS "GGML_HIP_FA_ASM_GFX90A: gfx90a FA-v3 hd128 bf16 asm fast path embedded")
endif()

if (GGML_HIP_CK_FMHA_HD256)
    if (NOT GGML_HIP_FA_ASM_GFX90A)
        message(FATAL_ERROR "GGML_HIP_CK_FMHA_HD256 requires GGML_HIP_FA_ASM_GFX90A (it shares its dispatcher)")
    endif()
    set(GGML_HIP_CK_FMHA_FWD_HPP ${GGML_HIP_CK_FMHA_DIR}/example/ck_tile/01_fmha/fmha_fwd.hpp)
    if (NOT EXISTS ${GGML_HIP_CK_FMHA_DIR}/include/ck_tile/ops/fmha.hpp OR NOT EXISTS ${GGML_HIP_CK_FMHA_FWD_HPP})
        message(FATAL_ERROR "GGML_HIP_CK_FMHA_HD256: GGML_HIP_CK_FMHA_DIR='${GGML_HIP_CK_FMHA_DIR}' is not a composable_kernel checkout")
    endif()
    if (GGML_HIP_CK_FMHA_COMMIT)
        find_package(Git REQUIRED)
        execute_process(COMMAND ${GIT_EXECUTABLE} -C ${GGML_HIP_CK_FMHA_DIR} rev-parse HEAD
                        OUTPUT_VARIABLE GGML_HIP_CK_FMHA_HEAD OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE GGML_HIP_CK_FMHA_RC)
        execute_process(COMMAND ${GIT_EXECUTABLE} -C ${GGML_HIP_CK_FMHA_DIR} status --porcelain --untracked-files=no
                        OUTPUT_VARIABLE GGML_HIP_CK_FMHA_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT GGML_HIP_CK_FMHA_RC EQUAL 0 OR NOT GGML_HIP_CK_FMHA_HEAD STREQUAL GGML_HIP_CK_FMHA_COMMIT OR GGML_HIP_CK_FMHA_DIRTY)
            message(FATAL_ERROR "GGML_HIP_CK_FMHA_HD256: ${GGML_HIP_CK_FMHA_DIR} HEAD '${GGML_HIP_CK_FMHA_HEAD}' (dirty: '${GGML_HIP_CK_FMHA_DIRTY}') != clean ${GGML_HIP_CK_FMHA_COMMIT}")
        endif()
    endif()
    set(GGML_HIP_CK_FMHA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ck-fmha-gfx90a/ck_fmha_fwd_hd256_f16.cpp)
    target_sources(ggml-hip PRIVATE ${GGML_HIP_CK_FMHA_SRC})
    if (CXX_IS_HIPCC)
        set_source_files_properties(${GGML_HIP_CK_FMHA_SRC} PROPERTIES LANGUAGE CXX)
    else()
        set_source_files_properties(${GGML_HIP_CK_FMHA_SRC} PROPERTIES LANGUAGE HIP)
    endif()
    # CK headers only for this TU (they clash with ggml's half/bf16 helpers); flags as CK's 01_fmha example
    # (FAST_EXP2 is also what the aiter build that produced the AK-CK-HD256 evidence used)
    set_source_files_properties(${GGML_HIP_CK_FMHA_SRC} PROPERTIES
        INCLUDE_DIRECTORIES "${GGML_HIP_CK_FMHA_DIR}/include;${GGML_HIP_CK_FMHA_DIR}/example/ck_tile/01_fmha"
        COMPILE_OPTIONS "-DCK_TILE_FMHA_FWD_FAST_EXP2=1;-fgpu-flush-denormals-to-zero;-mllvm;-enable-post-misched=0;-Wno-undefined-func-template;-Wno-float-equal")
    add_compile_definitions(GGML_HIP_CK_FMHA_HD256)
    # C8-4-H18: embed the spill-free clang-24 build of the same two instances (ck-fmha-gfx90a/gen_co.sh) and launch it
    # with hipModuleLaunchKernel (default; GGML_HIP_FA_CK_HD256_CO=0 at run time = the in-library ROCm 6.2 instance).
    set(GGML_HIP_CK_FMHA_CO ${CMAKE_CURRENT_SOURCE_DIR}/ck-fmha-gfx90a/ck_fmha_fwd_hd256_f16.gfx90a.co)
    set(GGML_HIP_CK_FMHA_CO_SHA_WANT "2a05c29ba1561c6d5399445d5d892d213d9eab77b55af5b4974582b86460829c")
    file(SHA256 ${GGML_HIP_CK_FMHA_CO} GGML_HIP_CK_FMHA_CO_SHA)
    if (NOT GGML_HIP_CK_FMHA_CO_SHA STREQUAL GGML_HIP_CK_FMHA_CO_SHA_WANT)
        message(FATAL_ERROR "GGML_HIP_CK_FMHA_HD256: ${GGML_HIP_CK_FMHA_CO} sha256 ${GGML_HIP_CK_FMHA_CO_SHA} != ${GGML_HIP_CK_FMHA_CO_SHA_WANT}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GGML_HIP_CK_FMHA_CO})
    file(READ ${GGML_HIP_CK_FMHA_CO} GGML_HIP_CK_FMHA_CO_HEX HEX)
    string(LENGTH "${GGML_HIP_CK_FMHA_CO_HEX}" GGML_HIP_CK_FMHA_CO_HEX_LEN)
    math(EXPR GGML_HIP_CK_FMHA_CO_NBYTES "${GGML_HIP_CK_FMHA_CO_HEX_LEN} / 2")
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," GGML_HIP_CK_FMHA_CO_BYTES "${GGML_HIP_CK_FMHA_CO_HEX}")
    set(GGML_HIP_CK_FMHA_CO_GEN ${CMAKE_CURRENT_BINARY_DIR}/ck-fmha-hd256-blob.cpp)
    file(WRITE ${GGML_HIP_CK_FMHA_CO_GEN}.tmp
        "// generated by ggml/src/ggml-hip/CMakeLists.txt (GGML_HIP_CK_FMHA_HD256, C8-4-H18); do not edit\n#include <stddef.h>\n\n"
        "// ck_fmha_fwd_hd256_f16.gfx90a.co sha256 ${GGML_HIP_CK_FMHA_CO_SHA}\n"
        "extern \"C\" alignas(256) const unsigned char ggml_hip_ck_fmha_hd256_co[${GGML_HIP_CK_FMHA_CO_NBYTES}] = {${GGML_HIP_CK_FMHA_CO_BYTES}};\n"
        "extern \"C\" const size_t ggml_hip_ck_fmha_hd256_co_size = ${GGML_HIP_CK_FMHA_CO_NBYTES};\n"
        "extern \"C\" const char ggml_hip_ck_fmha_hd256_co_sha256[] = \"GGML_HIP_CK_FMHA_HD256_CO sha256 ${GGML_HIP_CK_FMHA_CO_SHA}\";\n")
    configure_file(${GGML_HIP_CK_FMHA_CO_GEN}.tmp ${GGML_HIP_CK_FMHA_CO_GEN} COPYONLY)
    target_sources(ggml-hip PRIVATE ${GGML_HIP_CK_FMHA_CO_GEN})
    message(STATUS "GGML_HIP_CK_FMHA_HD256: gfx90a CK-tile fmha fwd hd256 fp16 path (CK ${GGML_HIP_CK_FMHA_HEAD} at ${GGML_HIP_CK_FMHA_DIR})")
endif()

if (GGML_HIP_TOPK_ASM_GFX90A)
    # Embed the fp32 topksoftmax code objects as byte arrays (relocatable: nothing is read from disk at run time).
    # Digests bind the bytes to the AK-ASM-T1 batch_out retarget; a mismatch is a configure error.
    set(GGML_HIP_TOPK_ASM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/topk-asm-gfx90a)
    set(GGML_HIP_TOPK_ASM_GEN ${CMAKE_CURRENT_BINARY_DIR}/topk-asm-gfx90a-blobs.cpp)
    set(GGML_HIP_TOPK_ASM_SRC
        "// generated by ggml/src/ggml-hip/CMakeLists.txt (GGML_HIP_TOPK_ASM_GFX90A); do not edit\n#include <stddef.h>\n\n")
    foreach(GGML_HIP_TOPK_ASM_ENTRY
            "4x128x4|667136718fd747c109e3608277bdc4c4ec03717b64a9d41d4d097b5b9b03c6e5"
            "4x128x6|3b7d38ac0318731d1401c58c4e5fb0582db33c0dc4bf76bac9e4edcf19751682"
            "4x128x8|75e3aa9b95a85988456c88f2470e5541c666733219a8b6cec2483fd77a50c0e1"
            "4x256x6|d0087b4fc4d86c8f23f5102206cff28b6295b6312a406df9413d7ed1d6a020ab"
            "4x256x8|a3723afaa7e51598c7af83df715fab4a4b80460c6c4fbbe534e53d7d767c57fc"
            "4x384x8|10bd0008e9a6804e22bea551777d2cd3d035ccf8d100880f622e8278f6d9a253"
            "4x640x8|43fcc6e2e21dbbea0b7572fa460759f56579a265354989d3a75da6a542349594")
        string(REPLACE "|" ";" GGML_HIP_TOPK_ASM_PAIR "${GGML_HIP_TOPK_ASM_ENTRY}")
        list(GET GGML_HIP_TOPK_ASM_PAIR 0 GGML_HIP_TOPK_ASM_NAME)
        list(GET GGML_HIP_TOPK_ASM_PAIR 1 GGML_HIP_TOPK_ASM_SHA_WANT)
        set(GGML_HIP_TOPK_ASM_CO ${GGML_HIP_TOPK_ASM_DIR}/topksoftmax__topksoftmax_${GGML_HIP_TOPK_ASM_NAME}.gfx90a.co)
        file(SHA256 ${GGML_HIP_TOPK_ASM_CO} GGML_HIP_TOPK_ASM_SHA)
        if (NOT GGML_HIP_TOPK_ASM_SHA STREQUAL GGML_HIP_TOPK_ASM_SHA_WANT)
            message(FATAL_ERROR "GGML_HIP_TOPK_ASM_GFX90A: ${GGML_HIP_TOPK_ASM_CO} sha256 ${GGML_HIP_TOPK_ASM_SHA} != ${GGML_HIP_TOPK_ASM_SHA_WANT}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GGML_HIP_TOPK_ASM_CO})
        file(READ ${GGML_HIP_TOPK_ASM_CO} GGML_HIP_TOPK_ASM_HEX HEX)
        string(LENGTH "${GGML_HIP_TOPK_ASM_HEX}" GGML_HIP_TOPK_ASM_HEX_LEN)
        math(EXPR GGML_HIP_TOPK_ASM_NBYTES "${GGML_HIP_TOPK_ASM_HEX_LEN} / 2")
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," GGML_HIP_TOPK_ASM_BYTES "${GGML_HIP_TOPK_ASM_HEX}")
        string(APPEND GGML_HIP_TOPK_ASM_SRC
            "// topksoftmax_${GGML_HIP_TOPK_ASM_NAME}.gfx90a.co sha256 ${GGML_HIP_TOPK_ASM_SHA}\n"
            "extern \"C\" alignas(256) const unsigned char ggml_hip_topk_asm_${GGML_HIP_TOPK_ASM_NAME}[${GGML_HIP_TOPK_ASM_NBYTES}] = {${GGML_HIP_TOPK_ASM_BYTES}};\n"
            "extern \"C\" const char ggml_hip_topk_asm_${GGML_HIP_TOPK_ASM_NAME}_sha256[] = \"${GGML_HIP_TOPK_ASM_SHA}\";\n\n")
    endforeach()
    file(WRITE ${GGML_HIP_TOPK_ASM_GEN}.tmp "${GGML_HIP_TOPK_ASM_SRC}")
    configure_file(${GGML_HIP_TOPK_ASM_GEN}.tmp ${GGML_HIP_TOPK_ASM_GEN} COPYONLY)
    target_sources(ggml-hip PRIVATE ${GGML_HIP_TOPK_ASM_GEN})
    add_compile_definitions(GGML_HIP_TOPK_ASM_GFX90A)
    message(STATUS "GGML_HIP_TOPK_ASM_GFX90A: gfx90a AITER topksoftmax asm fast path embedded")
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

# Fast math for HIP, like CUDA's -use_fast_math. Not -ffast-math: that implies -ffinite-math-only, which breaks ggml's INFINITY masking and produces NaNs.

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)
