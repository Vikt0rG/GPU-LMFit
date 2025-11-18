# Simple Makefile wrapper for CMake-based build
# Usage:
#   make        -> configure + build (Release)
#   make clean  -> remove build directory
#   make run    -> run the Gauss_1D executable

.PHONY: default clean run rebuild configure show-config build-cpu build-gpu

# Detect Windows via the OS environment variable set by cmd.
ifeq ($(OS),Windows_NT)
# On Windows use the Visual Studio generator. Separate build dirs for cpu/gpu.
CMAKE_CONFIGURE_CPU = cmake -S . -B build-msvc-cpu -G "Visual Studio 17 2022" -A x64
CMAKE_BUILD_CPU = cmake --build build-msvc-cpu --config Release -v

# For GPU build rely on CMake to detect nvcc.
CMAKE_CONFIGURE_GPU = cmake -S . -B build-msvc-gpu -G "Visual Studio 17 2022" -A x64
CMAKE_BUILD_GPU = cmake --build build-msvc-gpu --config Release -v
else
# On Unix-like systems use separate out-of-source builds. Use GCC for CPU build.
CMAKE_CONFIGURE_CPU = CC=gcc CXX=g++ cmake -S . -B build-cpu
CMAKE_BUILD_CPU = cmake --build build-cpu --config Release -- -j

CMAKE_CONFIGURE_GPU = cmake -S . -B build-gpu
CMAKE_BUILD_GPU = cmake --build build-gpu --config Release -- -j
endif

default: build

configure:
	@echo $(if $(filter Windows_NT,$(OS)),"Detected Windows OS","Detected non-Windows OS")

show-config:
	@echo "CMAKE_BUILD: $(CMAKE_BUILD)"
	@echo "CMAKE_CONFIGURE: $(CMAKE_CONFIGURE)"

build:
	@echo "Building both CPU and GPU targets (build-cpu, build-gpu)"
	$(MAKE) build-cpu
	$(MAKE) build-gpu

build-cpu:
	@echo "Configuring and building CPU-only targets into build-cpu"
	$(CMAKE_CONFIGURE_CPU)
	$(CMAKE_BUILD_CPU)

build-gpu:
	@echo "Configuring and building GPU targets into build-gpu"
	$(CMAKE_CONFIGURE_GPU)
	$(CMAKE_BUILD_GPU)

rebuild: clean build

clean:
	@echo "Cleaning build artifacts..."
	@if [ -d build-cpu ]; then rm -rf build-cpu; fi
	@if [ -d build-gpu ]; then rm -rf build-gpu; fi
	@if [ -d build-msvc-cpu ]; then rm -rf build-msvc-cpu; fi
	@if [ -d build-msvc-gpu ]; then rm -rf build-msvc-gpu; fi
	@if [ -d package/cpu_lmfit/build ]; then rm -rf package/cpu_lmfit/build; fi
	@if [ -d package/cpu_lmfit/cpu_lmfit.egg-info ]; then rm -rf package/cpu_lmfit/cpu_lmfit.egg-info; fi
	@if [ -d package/cpu_lmfit/Release ]; then rm -rf package/cpu_lmfit/Release; fi
	# remove any copied extension files in the package root (abi-tagged or plain)
	@shopt -s nullglob 2>/dev/null || true
	@if ls package/cpu_lmfit/_cpu_lmfit* 1> /dev/null 2>&1; then rm -f package/cpu_lmfit/_cpu_lmfit*; fi

run:
	./build/Gauss_1D.exe
