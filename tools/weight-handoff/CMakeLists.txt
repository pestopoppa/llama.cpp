# llama-weight-handoff: NEW-13 prefill/decode weight-set handoff probe (experimental branch only).
# Uses only llama.h, so it runs against the production store's libllama/libggml of the same commit.
set(TARGET llama-weight-handoff)
add_executable(${TARGET} weight-handoff.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
