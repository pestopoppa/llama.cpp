include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-prefill-order.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# GPU-ZOS-1a: compile the update_slots() pre_decode/decode/post_decode/sampling timers (server-context.cpp DEBUG_TIMINGS).
# Debug instrument only: it adds a llama_synchronize() after every decode and a log line every 5 s. Never in production.
option(LLAMA_SERVER_DEBUG_TIMINGS "llama-server: compile the DEBUG_TIMINGS update_slots timers (debug only)" OFF)
if (LLAMA_SERVER_DEBUG_TIMINGS)
    target_compile_definitions(${TARGET} PRIVATE DEBUG_TIMINGS)
    message(STATUS "llama-server: DEBUG_TIMINGS timers compiled in (LLAMA_SERVER_DEBUG_TIMINGS=ON)")
endif()

# UFH14-LB7.1: prompt-cache timing probes in server-context.cpp / server-task.cpp (debug only, never production).
if (LLAMA_LB7_PROBES)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_LB7_PROBES)
    message(STATUS "llama-server: UFH14-LB7.1 lb7p probes compiled in (LLAMA_LB7_PROBES=ON)")
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
