// Library 1 (base): abstract shape, a device helper and a device global, all defined in base.cu.
#pragma once
#include "gpu.h"

struct Shape {
  __host__ __device__ virtual ~Shape(); // out of line: the base vtable lives in base.cu
  __device__ virtual double Distance(double x) const = 0;
  __device__ double Twice(double x) const; // non-virtual member defined in base.cu
};

extern __device__ int gCalls;        // device global defined in base.cu, used by kernels in another library
__device__ double Helper(double x);  // device function defined in base.cu

// Host functions of the base library (copy to/from the device global by symbol, in its own translation unit)
bool ResetCalls();
int ReadCalls();
