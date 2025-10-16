# GPU-LMFit

Lightweight Levenberg–Marquardt fitting library with CUDA-ready GPU codepaths.

This repository contains a small fitting library (CPU + GPU sources) and an simple example CPU program that fits a 1D Gaussian.

## Current status

- CPU example: working (examples/CPU/Gauss_1D)
- GPU fitting: in progress — GPU kernels and CUDA sources are present in `src/gpu` and will be wired up soon :)

## Build (Windows, out-of-source)

This project uses CMake. The example was developed using Ninja as the generator but any generator supported by your platform should work.

From the repository root (PowerShell):

```powershell
mkdir build; cd build
cmake -G Ninja ..
cmake --build .
```

If you don't have Ninja, replace `-G Ninja` with the generator you prefer (like "Visual Studio 16 2019").

## Run the CPU example

After building, the CPU example executable `Gauss_1D` is available in the build subdirectory `examples/CPU`. It uses the headers from `include/` and the CPU solver implementation from `src/cpu`.

From the `build` directory you can run (adjust path if your generator places binaries elsewhere):

```powershell
.\examples\CPU\Gauss_1D.exe
```

## Project layout

- include/        — public headers (models, types, utilities)
- src/            — implementation (cpu, gpu, binding)
  - cpu/          — CPU solver and example wiring
  - gpu/          — CUDA kernels and GPU estimators (work in progress)
- examples/       — small example programs (currently CPU Gauss_1D)
- tests/          — test sources (experimental)

## Roadmap / Notes / WIP

- Finish GPU integration and provide a GPU example binary.
- (Add unit tests and CI for cross-platform builds.)

## License

See the `LICENSE` file in the repository root.