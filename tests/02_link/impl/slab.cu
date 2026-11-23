#include "slab.h"

#include <new>

__device__ Slab::Slab(double h_) : h(h_) {}
__device__ double Slab::Distance(double x) const { return fabs(x) - h; }

__device__ Shape *MakeSlabAt(void *mem, double h) { return new (mem) Slab(h); }

__global__ void ConstructSlab(void *mem, double h, Shape **out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) *out = new (mem) Slab(h);
}

bool LaunchConstructSlab(void *mem, double h, Shape **out)
{
  ConstructSlab<<<1, 1>>>(mem, h, out);
  return GPU_CHECK(GPU(GetLastError)());
}
