// Backend selection for the smoke tests.
//
// Runtime calls are written once as GPU(Name), which pastes the backend prefix: GPU(Malloc) is cudaMalloc, hipMalloc,
// or <prefix>Malloc for a custom CUDA-like runtime. This is the scheme VecGeom uses (VECGEOM_DEVICE_API_CALL).
// The few calls whose names differ between runtimes are wrapped below.
#pragma once

#include <cstdio>
#include <string>

#if defined(GPU_BACKEND_CUDA)
#include <cuda_runtime.h>
#define GPU_PREFIX cuda
#define GPU_TRAP() __trap()
#elif defined(GPU_BACKEND_HIP)
#include <hip/hip_runtime.h>
#define GPU_PREFIX hip
#define GPU_TRAP() __builtin_trap()
#elif defined(GPU_BACKEND_CUSTOM)
// Custom CUDA-like runtime: -DGPU_RUNTIME_HEADER=<xyz_runtime.h> -DGPU_API_PREFIX=xyz (set by CMake)
#include GPU_RUNTIME_HEADER
#define GPU_PREFIX GPU_API_PREFIX
#ifndef GPU_TRAP
#define GPU_TRAP() __builtin_trap()
#endif
#else
#error "Define GPU_BACKEND_CUDA, GPU_BACKEND_HIP or GPU_BACKEND_CUSTOM"
#endif

#define GPU_CAT_(a, b) a##b
#define GPU_CAT(a, b) GPU_CAT_(a, b)
#define GPU(name) GPU_CAT(GPU_PREFIX, name)

/// Address of a __device__ variable for GPU(MemcpyToSymbol) / GPU(MemcpyFromSymbol)
#define GPU_SYMBOL(var) ((const void *)&(var))

/// Check a runtime call; prints the failing call and returns false on error
#define GPU_CHECK(call) gpu_check((call), #call, __FILE__, __LINE__)

inline bool gpu_check(GPU(Error_t) err, const char *what, const char *file, int line)
{
  if (err == GPU(Success)) return true;
  std::fprintf(stderr, "%s:%d: %s -> %s\n", file, line, what, GPU(GetErrorString)(err));
  return false;
}

// ---- calls whose names or types differ between runtimes
#if defined(GPU_BACKEND_HIP)
using gpuDeviceProp = hipDeviceProp_t;
inline GPU(Error_t) gpu_malloc_host(void **p, size_t n) { return hipHostMalloc(p, n, hipHostMallocDefault); }
inline GPU(Error_t) gpu_free_host(void *p) { return hipHostFree(p); }
inline GPU(Error_t) gpu_alloc_mapped(void **p, size_t n) { return hipHostMalloc(p, n, hipHostMallocMapped); }
inline std::string gpu_arch(const gpuDeviceProp &p) { return p.gcnArchName; }
#else
// CUDA and CUDA-like runtimes
using gpuDeviceProp = GPU(DeviceProp);
inline GPU(Error_t) gpu_malloc_host(void **p, size_t n) { return GPU(MallocHost)(p, n); }
inline GPU(Error_t) gpu_free_host(void *p) { return GPU(FreeHost)(p); }
inline GPU(Error_t) gpu_alloc_mapped(void **p, size_t n) { return GPU(HostAlloc)(p, n, GPU(HostAllocMapped)); }
inline std::string gpu_arch(const gpuDeviceProp &p)
{
  return "sm_" + std::to_string(p.major) + std::to_string(p.minor);
}
#endif
