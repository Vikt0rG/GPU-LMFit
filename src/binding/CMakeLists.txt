cmake_minimum_required(VERSION 3.10)

# Include pybind11 from extern. Use the copy in the project root.
add_subdirectory(${CMAKE_SOURCE_DIR}/extern/pybind11 ${CMAKE_BINARY_DIR}/extern/pybind11 EXCLUDE_FROM_ALL)

# Create the python extension module from source
pybind11_add_module(_cpu_lmfit binding_cpu.cpp)

# Link the underlying CPU library into the module
target_link_libraries(_cpu_lmfit PRIVATE lmfit_cpu)

# Prefer static MSVC runtime for the extension if supported
if (MSVC)
	if(POLICY CMP0091)
		set_target_properties(_cpu_lmfit PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	endif()
endif()

# For MinGW/GNU toolchains, prefer static libgcc/libstdc++ to avoid distributing DLLs
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_link_options(_cpu_lmfit PRIVATE "-static-libgcc" "-static-libstdc++")
endif()

# Do NOT set target output directories into the source `package/` folder.
# When CMake places a target's runtime/library output directory inside the
# source tree, the build system (especially on MSVC) may also copy dependent
# DLLs and other build artifacts (e.g. Release/) into that directory. Keep 
# build outputs in the normal build tree and explicitly copy only the final 
# Python extension below.

# Ensure a copy of the final artifact (including Python ABI filename tag) is placed
# directly into the package folder with the exact target filename. This preserves
# names like _cpu_lmfit.cp314-win_amd64.pyd regardless of generator/config directories.
add_custom_command(TARGET _cpu_lmfit POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/package/cpu_lmfit
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:_cpu_lmfit>
		${CMAKE_SOURCE_DIR}/package/cpu_lmfit/$<TARGET_FILE_NAME:_cpu_lmfit>
	COMMENT "Copying python extension into package/cpu_lmfit"
)
