cmake_minimum_required(VERSION 3.10)

project(GPU-LMFit LANGUAGES CXX)

# Gather GPU sources
set(LMFit_GPU_SOURCES
    lm_solver.cu
)

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA toolkit not found. The GPU component is required for this project. Configure CMake with a CUDA-capable generator and CUDA toolkit installed.")
endif()

# Enable CUDA language and create a CUDA static library for GPU code
enable_language(CUDA)

add_library(lmfit_gpu STATIC ${LMFit_GPU_SOURCES})
target_include_directories(lmfit_gpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(lmfit_gpu PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

# Set CUDA properties: separable compilation and position independent code
set_target_properties(lmfit_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)