cmake_minimum_required(VERSION 3.10)
project(Gauss_1D_GPU)

find_package(CUDAToolkit REQUIRED)

add_executable(Gauss_1D_gpu Gauss_1D.cpp)

target_include_directories(Gauss_1D_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against the GPU library created in src/gpu
target_link_libraries(Gauss_1D_gpu PRIVATE lmfit_gpu CUDA::cublas CUDA::cusolver)

set_target_properties(Gauss_1D_gpu PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)
