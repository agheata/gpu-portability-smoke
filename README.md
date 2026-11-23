# GPU portability smoke tests for VecGeom and AdePT

Small, self-contained tests that check whether a GPU and its toolchain provide what the
[VecGeom](https://gitlab.cern.ch/VecGeom/VecGeom) geometry library and the [AdePT](https://github.com/apt-sim/AdePT)
GPU transport need. They contain no code from VecGeom or AdePT and have no dependencies beyond a C++17 GPU
toolchain and CMake 3.21.

VecGeom builds one source for NVIDIA (CUDA) and AMD (HIP) GPUs. The tests reproduce, in miniature, the features its
design relies on, so one run tells whether a further GPU family can be supported and what is missing.

## Quick start

Requirements: a Linux machine with the GPU, its compiler and runtime, CMake 3.21 or newer, and bash. Clone the
repository on that machine and run `run.sh` with the backend that matches the toolchain:

| Your toolchain | Command |
| --- | --- |
| NVIDIA CUDA, or a compiler that CMake accepts as a CUDA compiler | `./run.sh -DGPU_BACKEND=CUDA [-DCMAKE_CUDA_COMPILER=<nvcc-compatible driver>]` |
| AMD HIP / ROCm, or a HIP-compatible stack | `./run.sh -DGPU_BACKEND=HIP -DCMAKE_HIP_COMPILER=<clang++> -DCMAKE_CXX_COMPILER=<clang++> -DCMAKE_HIP_ARCHITECTURES=<arch>` |
| Another compiler that accepts CUDA source, with a CUDA-like runtime under its own prefix | `./run.sh -DGPU_BACKEND=CUSTOM ...`, see [below](#other-cuda-like-toolchains) |

Examples on the reference machines:

```sh
./run.sh -DGPU_BACKEND=CUDA                    # NVIDIA, nvcc found in PATH
./run.sh -DGPU_BACKEND=HIP \
         -DCMAKE_HIP_COMPILER=amdclang++ -DCMAKE_CXX_COMPILER=amdclang++ \
         -DCMAKE_HIP_ARCHITECTURES=gfx1100       # AMD RX 7900 XTX
```

A run takes a few minutes. Further CMake options can be passed after `-DGPU_BACKEND`; environment variables:
`BUILD_DIR`, `OUT_DIR`, `TEST_TIMEOUT` (seconds per test, default 300), `STACK_BYTES` (stack per thread for the
recursion tests, default 16384). On machines with several GPUs, select one with `CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES` (or the equivalent of your runtime).

## Output

`run.sh` configures, builds every test target separately (a build failure is itself a result), runs each test in
its own process with a timeout, and writes:

| File | Content |
| --- | --- |
| `results/report.txt` | The summary below, printed at the end of the run |
| `results/results.txt` | One line per check: `RESULT <id> <PASS\|FAIL\|WARN\|INFO> <detail>` |
| `results/*.log` | Configure, build and test logs |
| `results.tar.gz` | All of the above: **please send this file back** |

The report starts with the machine, compiler and device, then gives one verdict per requirement, then the
measurements and every individual check. From the AMD reference run:

```text
GPU portability smoke tests - report
Backend:  HIP
Compiler: HIP .../amdclang++ (AMD clang version 24.0.0git ...)
Device:   AMD Radeon RX 7900 XTX, gfx1100, 48 multiprocessors, 24.0 GB, warp/wavefront 32

Requirements
  Virtual functions in device code (*)          PASS
  Device linking across static libraries (*)    PASS
  Recursion, settable stack size                PASS
  Single-source C++17, placement new            PASS
  CUDA-like runtime API                         PASS (1 optional item(s) missing)
  Atomics on device memory                      PASS (2 optional item(s) missing)
  FP64                                          PASS
  Build integration                             PASS
  (*) prerequisites

Measurements
  05.fp64.throughput INFO indicative FMA-chain microbenchmark: FP32 18023 GFLOP/s, FP64 455 GFLOP/s, ...

All results
  01.virtual.same_kernel PASS construct with placement new and call in one kernel
  ...
  06.atomics.host_or WARN or on mapped pinned host memory: 0x00000000
  07.runtime.trap WARN the runtime aborts the process on a device trap (exit code 134)
```

Verdicts: **PASS**, **FAIL** (a required feature does not work), **WARN** (an optional feature does not work),
**INFO** (a measurement). A FAIL names the check, and the logs show the compiler or runtime error; partial results
are useful too, so please send the archive whatever the outcome.

### Other CUDA-like toolchains

If the compiler accepts CUDA source (`__global__`, `__device__`, `<<<...>>>`) and the runtime API has CUDA names
with another prefix (`xyzMalloc`, `xyzMemcpy`, ...):

```sh
./run.sh -DGPU_BACKEND=CUSTOM -DCMAKE_CXX_COMPILER=<your compiler> \
         -DGPU_RUNTIME_HEADER=xyz_runtime.h -DGPU_API_PREFIX=xyz \
         -DGPU_COMPILE_FLAGS="<flags to compile .cu files as relocatable device code>" \
         -DGPU_LINK_FLAGS="<flags for the device link step>" -DGPU_LIBRARIES="<runtime library>"
```

Example, tested: clang in CUDA mode, as a stand-in for clang-based vendor compilers:

```sh
C=/usr/local/cuda-12.8
./run.sh -DGPU_BACKEND=CUSTOM -DCMAKE_CXX_COMPILER=clang++-21 \
         -DGPU_RUNTIME_HEADER=cuda_runtime.h -DGPU_API_PREFIX=cuda \
         "-DGPU_COMPILE_FLAGS=-x cuda --cuda-path=$C --cuda-gpu-arch=sm_89 -fgpu-rdc" \
         "-DGPU_LINK_FLAGS=-fgpu-rdc --offload-new-driver --cuda-path=$C --offload-arch=sm_89" \
         "-DGPU_LIBRARIES=-L$C/lib64;-lcudart" -DGPU_PURE_VIRTUAL_SHIM=ON
```

Without `-DGPU_PURE_VIRTUAL_SHIM=ON` this toolchain fails to link test 02: abstract classes need a device-side
`__cxa_pure_virtual`, which nvcc and HIP provide and clang's CUDA mode does not. The report names this case; the
option adds a small definition so that the other tests can still run.

All runtime calls go through `GPU(Name)` in [common/gpu.h](common/gpu.h), which pastes the prefix (the scheme VecGeom
uses); the few calls whose names differ between runtimes (pinned host memory, device properties) are wrapped there.
A toolchain with its own programming model needs a new section in that header and in `CMakeLists.txt`; the tests
themselves should not need changes.

## What is tested

| Test | Requirement | What it does |
| --- | --- | --- |
| 01_virtual | **Virtual functions in device code** (prerequisite) | Polymorphic objects built with placement new in a kernel, called through base pointers in the same and in later launches; nested virtual calls; virtual destructor |
| 02_link | **Device linking across static libraries** (prerequisite) | Three static libraries and a program in one executable; vtables, device functions and a device global cross library boundaries |
| 03_recursion | Recursion, settable stack size | Composite nodes calling children through virtual functions, depth 8 / 64 / 256 chosen at run time, with the default and with a raised stack |
| 04_image | Single-source C++17, placement new | VecGeom's geometry transfer in miniature: host-built memory image at device addresses, one copy, shells constructed in place, point location compared with the host |
| 05_fp64 | FP64 | Correctly rounded sqrt, division and fma (bit-identical to the host); FP64/FP32 throughput ratio |
| 06_atomics | Atomics on device memory | Integer add, max, CAS and exchange under contention; optional: native double add, atomics on mapped pinned host memory |
| 07_runtime | CUDA-like runtime API | Allocation and copies, pinned memory, streams and events, copies by symbol, launch errors, shared memory, printf; optional: managed memory, device trap |
| 08 (build) | Build integration | CMake configure and the build of every target |

Virtual functions in device code and device linking are prerequisites: the VecGeom design does not work without
them.

## Reference results

| | NVIDIA RTX 4090, CUDA 13.1 | AMD RX 7900 XTX, ROCm 10.1 | RTX 4090, clang 21 (CUSTOM) |
| --- | --- | --- | --- |
| Virtual functions in device code | PASS | PASS | PASS (with the pure-virtual shim) |
| Device linking across static libraries | PASS | PASS | PASS (with the pure-virtual shim) |
| Recursion, settable stack size | PASS | PASS | PASS |
| Single-source C++17, placement new | PASS | PASS | PASS |
| CUDA-like runtime API | PASS | PASS (a device trap aborts the process) | PASS |
| Atomics on device memory | PASS | PASS (sub and or on mapped host memory are lost) | PASS |
| FP64 | PASS (FP64/FP32 = 1/62) | PASS (FP64/FP32 = 1/40) | PASS (1/62 to 1/70) |
| Build integration | PASS | PASS | PASS |

The FP64/FP32 ratio comes from an indicative microbenchmark (independent fma chains), not from a tuned peak
measurement.

Full reports: [reference/](reference/).

## License

Apache License 2.0, see [LICENSE](LICENSE).
