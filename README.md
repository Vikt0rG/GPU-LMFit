# GPU-LMFit

Lightweight Levenberg–Marquardt fitting library with CPU and CUDA-ready GPU codepaths that uses forward difference method to approximate gradients which avoids having to define analytical derivatives by hand.

This repository contains a fitting library (CPU + GPU sources), Python binding wiring, as well as examples that demonstrate fitting a 1D Gaussian and a log-norm correlation function on the CPU (and a GPU example is still in progress).

## Current status

- CPU example: working and stable (see `examples/CPU/`).
- GPU fitting: work-in-progress — GPU kernels and estimators are present in `src/gpu/` but the integration is still unstable.
- Still tring to figure out the best approach for the ModelDescriptor (simple struct combining model definition and its derivative) adoptation in CUDA and how to avoid manual model definitions.

## Quick build (Windows, out-of-source)

The project uses CMake. The repository was developed using the Ninja generator on Windows, but any CMake generator supported by your platform should work.

From the repository root (PowerShell):

```powershell
mkdir build; cd build
cmake --build .
```

## Run the CPU example

After building, the CPU example executable `Gauss_1D` is available under the build output tree in `examples/CPU` (path depends on your CMake generator). From the `build` directory you can usually run:

```powershell
.\examples\CPU\Gauss_1D.exe
```

## Project layout and functionality (top-level sections)

- `include/` — public headers for models, types and utilities used by examples and library code.
- `src/` — implementation sources and bindings:
  - `src/cpu/` — CPU implementation of the Levenberg–Marquardt solver and helper code (used by the CPU example).
  - `src/gpu/` — CUDA `.cu/.cuh` sources with GPU kernels and estimators (WIP).
  - `src/binding/` — binding code that hooks the native library to higher-level language bindings (eg. pybind11-based Python bindings).
- `examples/` — example programs showing how to use the library:
  - `examples/CPU/` — working CPU examples (`Gauss_1D.cpp` / `.py` & `Correlation_function_LogNorm.cpp` / `.py`).
  - `examples/GPU/` — GPU example sources (requires a working CUDA toolchain and the GPU integration to be stable).
- `extern/pybind11/` — vendored `pybind11` sources used to expose C++ code to Python.
- `package/` — Python packaging helpers (minimal `cpu_lmfit` package and `setup.py`) After compiling the library `.dll` files are copied here automatically and allows direct installation with `pip`.
- `tests/` — experimental test sources and smoke tests.
- `docs/` — documentation, logs and notes.

## Top-level `CMakeLists.txt`: What it does

The top-level `CMakeLists.txt` (project root) configures the project and performs these main actions:

- Sets minimum CMake version and declares the project with languages `CXX` and `CUDA`.
- Configures the C++ standard to C++11.
- Adds common include directories used by the project (`include/`, `include/cpu`, `include/gpu`, and a `make` folder used for helper headers).
- Adds subdirectories that produce the library and examples:
  - `src/cpu` — builds the CPU solver library and/or objects.
  - `src/gpu` — builds CUDA sources (only if a CUDA toolchain is present).
  - `src/binding` — builds language bindings (pybind11 wiring present in `extern/`).
  - `examples/CPU` and `examples/GPU` — example executables.

The effect is that the top-level CMake lists manages all subprojects and ensures headers and sources are visible to the examples.

## `Makefile`: Its role and targets

- `Makefile` (top-level) is a small convenience wrapper around the canonical CMake workflow. It exposes a few common recipes (targets) that configure, build and clean various out-of-source build directories so developers can use `make` semantics instead of calling CMake directly.

Makefile targets and what they do

- `default` / `build`:
  - The default target. When you run `make` with no arguments it invokes `build`.
  - Orchestrates building CPU, GPU and example targets by invoking `build-cpu`, `build-gpu` and `build-examples` in sequence.

- `configure`:
  - Prints a short OS detection message. It does not run CMake configuration steps by itself; it is useful to quickly confirm whether the Makefile detected Windows (`Windows_NT`) or not.

- `show-config`:
  - Prints the `CMAKE_BUILD` and `CMAKE_CONFIGURE` variables as seen by the Makefile. Note: those variables are not set elsewhere in the Makefile, so this target may print empty values unless you exported them yourself.

- `build-cpu`:
  - Configures and builds CPU-only targets into a CPU-specific out-of-source build directory.
  - On Windows this uses the Visual Studio generator and a build directory named `build-msvc-cpu` (generator: "Visual Studio 17 2022" with x64 arch).
  - On Unix-like systems it sets `CC=gcc CXX=g++` and configures into `build-cpu`.
  - The recipe runs the configure step then invokes `cmake --build` for the chosen directory and configuration (Release).

- `build-gpu`:
  - Configures and builds GPU targets into a GPU-specific out-of-source build directory.
  - On Windows it uses `build-msvc-gpu` with the Visual Studio generator, and on Unix-like systems it uses `build-gpu`.
  - Rely on CMake to detect the CUDA toolchain (nvcc) when configuring the GPU build.

- `build-examples`:
  - Configures and builds the example targets (for instance the `Gauss_1D` example) into `build-examples`.
  - This target uses the default generator and is useful when you just want the example binaries and not full CPU/GPU libraries separately.

- `rebuild`:
  - Convenience target that runs `clean` followed by `build` to produce a fresh build directory and artifacts.

- `clean`:
  - Removes build directories and a few package-related build artefacts. The recipe deletes `build-cpu`, `build-gpu`, `build-examples`, `build-msvc-cpu`, `build-msvc-gpu` and some `package/cpu_lmfit` build outputs.
  - Implementation uses POSIX-style shell commands (`[ -d ... ]`, `rm -rf`, `shopt`) so behavior on native Windows `cmd.exe` may vary; it generally works in Unix-like environments or MSYS/MinGW shells that provide a POSIX shell.

- `run`:
  - Runs the example executable `Gauss_1D` from the `build-examples` output tree (`./build-examples/Gauss_1D.exe`).
  - Depending on your generator and platform (single-config vs multi-config generators such as Visual Studio) the actual executable may live in a configuration-specific subdirectory (e.g. `Release\` or `Debug\`) or a different path; adjust the path if needed.

Notes and platform caveats

- The Makefile contains `ifeq ($(OS),Windows_NT)` branching to select generator and build directories for Windows vs Unix-like systems. However, some recipes rely on a POSIX shell — so using `make` from a Unixlike environment on Windows (MSYS2, Git Bash, WSL) yields the smoothest behavior.
- The Makefile is intentionally minimal: it simply drives `cmake -S`/`-B` and `cmake --build` with a small set of well-chosen directories. Advanced or custom build workflows should invoke CMake directly for full control.

## Examples and bindings

- The CPU example (`examples/CPU/Gauss_1D.*`) demonstrates using the CPU solver to fit a 1D Gaussian and is the best starting point for understanding how to use the library.
- The `src/binding/` directory together with the vendored `extern/pybind11/` provide the pieces needed to build Python bindings. If you plan to build Python wheels or import the library from Python, inspect `src/binding/CMakeLists.txt` and `package/`.

## How to contribute / build tips

- If you only need the CPU example, configure CMake without CUDA or build only the `src/cpu` subproject and the `examples/CPU` target.
- To build GPU code you need a CUDA-capable system and a compatible CUDA toolkit; the CMake setup will try to enable CUDA language support when available. (For this I would first need to finally figure out the best approach for easy/user-friendly model initialization with low overhead 🙃)
- Use an out-of-source build directory (for example `build/`) to keep build artefacts separate from source.

## Roadmap / Notes / WIP

- Finally finish GPU integration and provide a GPU example binary.
- Close all the TODO's
- Add unit tests and CI for cross-platform builds.

## TODO / WIP

- src/cpu/lm_solver.cpp:
  - "WIP: Calculation of parameter_stddevs_ not yet implemented" — implement computation and storage of parameter standard deviations returned by the solver when transitioning to CUDA and cuBLAS (disregarded in the CPU version since CPU verison uses a "home-made" solver with Cholesky decomposition)
  - "WIP: Optimize by batch processing - compute all gradients at once rather than per parameter" — improve Jacobian computation performance (batch numerical differences or vectorized approach, also disregarded in the CPU verion sicne the main focus was on CUDA based approach).
  - "TODO: Add a report message if solver fails" — provide richer error reporting when normal-equation solve or decomposition fails.

- src/binding/binding_cpu.cpp:
  - "TODO: Add a binding to model functions that need speedup" — expose selected, performance-critical model helper functions directly to Python to avoid trampoline overhead.

- include/cpu/lm_solver.hpp:
  - "TODO: and their standard deviations" — public API currently contains a commented form of `copy_optimized_params` that would also return parameter stddevs; reinstate and wire it once stddevs are computed in the CUDA verison with cuBLAS.

- include/cpu/forward_difference.hpp:
  - "TODO: Tune the step size for better accuracy and stability." — empirically select or compute a better step-size rule for numeric differentiation.

## License

See the `LICENSE` file in the repository root.