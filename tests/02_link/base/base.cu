#include "shape.h"

#ifdef GPU_PURE_VIRTUAL_SHIM
// Some toolchains (e.g. clang in CUDA mode) do not provide the device-side handler that abstract classes reference
extern "C" __device__ void __cxa_pure_virtual() { GPU_TRAP(); }
#endif

__host__ __device__ Shape::~Shape() {}
__device__ double Shape::Twice(double x) const { return 2.0 * Distance(x); }

__device__ int gCalls = 0;
__device__ double Helper(double x) { return 2.0 * x; }

bool ResetCalls()
{
  int zero = 0;
  return GPU_CHECK(GPU(MemcpyToSymbol)(GPU_SYMBOL(gCalls), &zero, sizeof(int), 0, GPU(MemcpyHostToDevice)));
}

int ReadCalls()
{
  int n = -1;
  if (!GPU_CHECK(GPU(MemcpyFromSymbol)(&n, GPU_SYMBOL(gCalls), sizeof(int), 0, GPU(MemcpyDeviceToHost)))) return -1;
  return n;
}
