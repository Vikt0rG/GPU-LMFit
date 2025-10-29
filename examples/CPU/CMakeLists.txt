cmake_minimum_required(VERSION 3.5)

# Example: Gauss_1D
project(Gauss_1D_example CXX)

# Helper function to add CPU examples with consistent settings.
# Usage: add_cpu_example(<TargetName> <source1.cpp> <source2.cpp> ...)
function(add_cpu_example TARGET)
    # Remaining arguments are the example's source files
    set(sources ${ARGN})

    # Create the example executable from the provided sources. The common
    # lm-solver implementation is provided by the `lmfit_cpu` library.
    add_executable(${TARGET} ${sources})

    # Ensure the example can find the headers
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    # Link against the CPU core library which provides the lm solver
    # implementation and public headers.
    target_link_libraries(${TARGET} PRIVATE lmfit_cpu)

    # Common compile properties for CPU examples
    set_target_properties(${TARGET} PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
    )
endfunction()

# Add examples using the helper
add_cpu_example(Gauss_1D Gauss_1D.cpp)
add_cpu_example(Correlation_function_LogNorm Correlation_function_LogNorm.cpp)
