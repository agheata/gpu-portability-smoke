#include "kernels.h"
#include "slab.h"

#include <new>

__global__ void Evaluate(Shape *const *s, const double *x, double *out, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = s[0]->Twice(x[i]) + Helper(x[i]); // virtual call (lib 2) through a member of lib 1, helper of lib 1
    atomicAdd(&gCalls, 1);                     // device global of lib 1
  }
}

__global__ void BuildViaFactory(void *mem, double h, Shape **out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) *out = MakeSlabAt(mem, h); // device function of lib 2
}

__global__ void BuildHere(void *mem, double h, Shape **out)
{
  // Constructed here, so this library references the Slab vtable and constructor defined in lib 2
  if (blockIdx.x == 0 && threadIdx.x == 0) *out = new (mem) Slab(h);
}

bool LaunchEvaluate(Shape *const *s, const double *x, double *out, int n)
{
  Evaluate<<<(n + 255) / 256, 256>>>(s, x, out, n);
  return GPU_CHECK(GPU(GetLastError)());
}

bool LaunchBuildViaFactory(void *mem, double h, Shape **out)
{
  BuildViaFactory<<<1, 1>>>(mem, h, out);
  return GPU_CHECK(GPU(GetLastError)());
}

bool LaunchBuildHere(void *mem, double h, Shape **out)
{
  BuildHere<<<1, 1>>>(mem, h, out);
  return GPU_CHECK(GPU(GetLastError)());
}
