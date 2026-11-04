llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-dsv41-engram.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-compress.cpp
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-hybrid-idx.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES})
set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_LSA_NATIVE)
    target_sources(llama PRIVATE
        llama-lsa-native.cpp
        llama-lsa-map.cpp
        llama-lsa-digest.cpp
        llama-lsa-graph.cpp
        llama-lsa-runtime.cpp)
    target_link_libraries(llama PRIVATE ggml-cpu)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    target_link_libraries(llama PRIVATE OpenSSL::Crypto)
    # Internal model/context headers contain gated fields; every consumer of
    # libllama must see the same class layout in this experimental build.
    target_compile_definitions(llama PUBLIC LLAMA_LSA_NATIVE)
    if (GGML_HIP)
        # ggml enables HIP inside its own subdirectory. The libllama HIP
        # translation unit needs the language and imported host target here.
        if (AMDGPU_TARGETS AND NOT GPU_TARGETS)
            set(GPU_TARGETS ${AMDGPU_TARGETS})
        endif()
        if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
        endif()
        enable_language(HIP)
        find_package(hip REQUIRED)
        set_source_files_properties(llama-lsa-native-hip.hip PROPERTIES LANGUAGE HIP)
        target_sources(llama PRIVATE llama-lsa-native-hip.hip llama-lsa-runtime-hip.cpp)
        if (CMAKE_HIP_ARCHITECTURES)
            set_property(TARGET llama PROPERTY HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
        endif()
        target_compile_definitions(llama PRIVATE GGML_USE_HIP LLAMA_LSA_RUNTIME_HIP)
        target_include_directories(llama PRIVATE ../ggml/src ../ggml/src/ggml-cuda)
        target_link_libraries(llama PRIVATE hip::host)
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)

# INF-77 DS41-C4: llama-context.cpp and llama-graph.cpp carry host-side phase timers guarded by
# GGML_CPU_PROF.  That define is PRIVATE to the ggml-cpu target, so until now it could never
# reach libllama and the host-side half of the profile did not exist in any binary.
if (GGML_CPU_PROF)
    target_compile_definitions(llama PRIVATE GGML_CPU_PROF)
endif()
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
