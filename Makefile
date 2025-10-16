# Simple Makefile wrapper for CMake-based build
# Usage:
#   make        -> configure + build (Release)
#   make clean  -> remove build directory
#   make run    -> run the Gauss_1D executable

.PHONY: all clean run rebuild

all: build

build:
	cmake -S . -B build
	cmake --build build --config Release

rebuild: clean build

clean:
	@if [ -d build ]; then rm -rf build; fi

run:
	./build/Gauss_1D.exe
