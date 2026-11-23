// Library 2 (impl): a concrete shape whose virtual function and vtable live in slab.cu.
#pragma once
#include "shape.h"

struct Slab : Shape {
  double h;
  __device__ explicit Slab(double h_);
  __device__ double Distance(double x) const override; // out of line: the Slab vtable lives in slab.cu
};

/// Device factory, called from kernels of another library
__device__ Shape *MakeSlabAt(void *mem, double h);

/// Host launcher of a kernel of this library that constructs a Slab
bool LaunchConstructSlab(void *mem, double h, Shape **out);
