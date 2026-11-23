// Library 3 (kernels): kernels that use the shapes of libraries 1 and 2 only through their headers.
#pragma once
#include "shape.h"

// Kernels are launched through host functions of the library that defines them, as VecGeom and AdePT do.
bool LaunchEvaluate(Shape *const *s, const double *x, double *out, int n);
bool LaunchBuildViaFactory(void *mem, double h, Shape **out);
bool LaunchBuildHere(void *mem, double h, Shape **out);
