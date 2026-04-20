# llama-math-tools: Native computational tools for LLM orchestration
#
# Philosophy: LLMs reason. Tools execute.
# Every tool call should save 100-5000 tokens of LLM generation.

cmake_minimum_required(VERSION 3.14)
project(llama-math-tools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only dependencies
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen")
set(JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/nlohmann")

# Verify Eigen exists
if(NOT EXISTS "${EIGEN_DIR}/Eigen/Core")
    message(FATAL_ERROR "Eigen not found. Run: git clone --depth 1 https://gitlab.com/libeigen/eigen.git external/eigen")
endif()

# Verify nlohmann/json exists
if(NOT EXISTS "${JSON_DIR}/json.hpp")
    message(FATAL_ERROR "nlohmann/json not found at ${JSON_DIR}")
endif()

# Main executable
add_executable(llama-math-tools
    main.cpp
    commands/numerical/matrix_op.cpp
    commands/numerical/solve_ode.cpp
    commands/numerical/optimize.cpp
    commands/statistical/monte_carlo.cpp
    commands/statistical/mcmc.cpp
    commands/statistical/bayesopt.cpp
    commands/visualization/plot_braille.cpp
    commands/visualization/render_math.cpp
    commands/visualization/plot_sixel.cpp
)

target_include_directories(llama-math-tools PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_DIR}
    ${JSON_DIR}
)

# Compiler optimizations for numerical code
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(llama-math-tools PRIVATE
        -O3
        -march=native      # Enable AVX-512 on EPYC 9655
        -ffast-math        # Aggressive FP optimizations
        -funroll-loops
    )
endif()

# OpenMP for parallel Monte Carlo (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llama-math-tools PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(llama-math-tools PRIVATE HAVE_OPENMP)
endif()

# Install
install(TARGETS llama-math-tools
    RUNTIME DESTINATION bin
)

# Tests (optional)
option(BUILD_MATH_TOOLS_TESTS "Build math-tools tests" OFF)
if(BUILD_MATH_TOOLS_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
