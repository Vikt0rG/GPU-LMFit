# Simple Makefile wrapper for CMake-based build
# Usage:
#   make        -> configure + build (Release)
#   make clean  -> remove build directory
#   make run    -> run the Gauss_1D executable

.PHONY: default clean run rebuild configure show-config

# Detect Windows (native) via the OS environment variable set by cmd/powershell
ifeq ($(OS),Windows_NT)
CMAKE_CONFIGURE = cmake -S . -B build-msvc -G "Visual Studio 17 2022" -A x64
CMAKE_BUILD = cmake --build build-msvc --config Release --target _cpu_lmfit -v
else
CMAKE_CONFIGURE = cmake -S . -B build
CMAKE_BUILD = cmake --build build --config Release
endif

default: build

configure:
	@echo $(if $(filter Windows_NT,$(OS)),"Detected Windows OS","Detected non-Windows OS")

show-config:
	@echo "CMAKE_BUILD: $(CMAKE_BUILD)"
	@echo "CMAKE_CONFIGURE: $(CMAKE_CONFIGURE)"

build:
	$(CMAKE_CONFIGURE)
	$(CMAKE_BUILD)

rebuild: clean build

clean:
	@echo "Cleaning build artifacts..."
	@if [ -d build ]; then rm -rf build; fi
	@if [ -d build-msvc ]; then rm -rf build-msvc; fi
	@if [ -d package/cpu_lmfit/build ]; then rm -rf package/cpu_lmfit/build; fi
	@if [ -d package/cpu_lmfit/cpu_lmfit.egg-info ]; then rm -rf package/cpu_lmfit/cpu_lmfit.egg-info; fi
	@if [ -d package/cpu_lmfit/Release ]; then rm -rf package/cpu_lmfit/Release; fi
	# remove any copied extension files in the package root (abi-tagged or plain)
	@shopt -s nullglob 2>/dev/null || true
	@if ls package/cpu_lmfit/_cpu_lmfit* 1> /dev/null 2>&1; then rm -f package/cpu_lmfit/_cpu_lmfit*; fi

run:
	./build/Gauss_1D.exe
