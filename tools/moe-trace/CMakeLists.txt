# llama-moe-trace: per-token MoE routing tap (INF-72 RT-1). Experimental branch only; tools/ change,
# no library or production-target code touched. See tools/moe-trace/README.md.
#
# LLAMA_BUILD_MOE_TRACE gates ONLY this directory's targets (default ON, like the sibling tools); it does
# not change any other target. -DLLAMA_BUILD_MOE_TRACE=OFF removes llama-moe-trace from the build entirely.
option(LLAMA_BUILD_MOE_TRACE "llama: build llama-moe-trace (MoE router tap, INF-72 RT-1)" ON)

if (LLAMA_BUILD_MOE_TRACE)
    set(TARGET llama-moe-trace)
    add_executable(${TARGET} moe-trace.cpp moe-trace-core.h moe-trace-tap.h)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    # model-free unit test of the core (stride-safe reader, RT-3 hazard guard, EOR, run writer): no llama, no GGUF
    if (LLAMA_BUILD_TESTS)
        add_executable(test-moe-trace-core tests/test-moe-trace-core.cpp moe-trace-core.h)
        target_compile_features(test-moe-trace-core PRIVATE cxx_std_17)
        add_test(NAME test-moe-trace-core COMMAND test-moe-trace-core ${CMAKE_CURRENT_BINARY_DIR}/synthetic-run)
        set_tests_properties(test-moe-trace-core PROPERTIES LABELS "moe-trace")

        # the eval callback against REAL ggml view tensors (ggml_argsort_top_k layout), still no model: links ggml-base only
        add_executable(test-moe-trace-tap tests/test-moe-trace-tap.cpp moe-trace-tap.h moe-trace-core.h)
        target_include_directories(test-moe-trace-tap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(test-moe-trace-tap PRIVATE ggml-base)
        target_compile_features(test-moe-trace-tap PRIVATE cxx_std_17)
        add_test(NAME test-moe-trace-tap COMMAND test-moe-trace-tap)
        set_tests_properties(test-moe-trace-tap PROPERTIES LABELS "moe-trace")
    endif()
endif()
