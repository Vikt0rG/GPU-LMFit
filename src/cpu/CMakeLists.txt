cmake_minimum_required(VERSION 3.10)

project(CPU-LMFit CXX)

set(LMFit_CPU_SOURCES
    lm_solver.cpp
)

# Build CPU core library lmfit_cpu
add_library(lmfit_cpu STATIC ${LMFit_CPU_SOURCES})
target_include_directories(lmfit_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(lmfit_cpu PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)

# Prefer static MSVC runtime when supported (CMake >= 3.15 property); fallback handled by generator flags.
if (MSVC)
    if(POLICY CMP0091)
        # Use the static runtime (/MT) if policy is available
        set_target_properties(lmfit_cpu PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()
