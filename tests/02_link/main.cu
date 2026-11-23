// 02 - Device linking across static libraries (prerequisite).
// Three static libraries (base, impl, kernels) and this program, linked into one executable. Virtual functions,
// vtables, a device function and a device global cross the library boundaries, as the device code of VecGeom,
// G4HepEm and AdePT does in one application.
#include "gpu.h"
#include "kernels.h"
#include "report.h"
#include "slab.h"

#include <cmath>
#include <vector>

int main()
{
  Report rep;
  const int n = 1 << 18;
  std::vector<double> x(n), out(n);
  for (int i = 0; i < n; ++i)
    x[i] = -4.0 + 8.0 * i / n;

  void *mem = nullptr;
  Shape **table = nullptr;
  double *dx = nullptr, *dout = nullptr;
  bool ok = GPU_CHECK(GPU(Malloc)(&mem, 64)) && GPU_CHECK(GPU(Malloc)((void **)&table, sizeof(Shape *))) &&
            GPU_CHECK(GPU(Malloc)((void **)&dx, n * sizeof(double))) &&
            GPU_CHECK(GPU(Malloc)((void **)&dout, n * sizeof(double))) &&
            GPU_CHECK(GPU(Memcpy)(dx, x.data(), n * sizeof(double), GPU(MemcpyHostToDevice)));
  if (!ok) {
    rep.check("02.link.setup", false, "device allocation failed");
    return rep.finish();
  }

  struct Case {
    const char *id, *what;
    int kind;
    double h;
  };
  const Case cases[] = {
      {"02.link.kernel_other_lib", "object built by a kernel of lib 2, called from a kernel of lib 3", 0, 1.0},
      {"02.link.device_function", "object built through a device function of lib 2 called from lib 3", 1, 2.0},
      {"02.link.vtable_reference", "object built in lib 3 with the vtable defined in lib 2", 2, 3.0}};
  for (const Case &c : cases) {
    bool run = c.kind == 0 ? LaunchConstructSlab(mem, c.h, table)
                           : (c.kind == 1 ? LaunchBuildViaFactory(mem, c.h, table) : LaunchBuildHere(mem, c.h, table));
    run      = run && GPU_CHECK(GPU(DeviceSynchronize)()) && ResetCalls();
    if (run) {
      run = LaunchEvaluate(table, dx, dout, n) && GPU_CHECK(GPU(DeviceSynchronize)()) &&
            GPU_CHECK(GPU(Memcpy)(out.data(), dout, n * sizeof(double), GPU(MemcpyDeviceToHost)));
    }
    int bad = -1;
    if (run) {
      bad = 0;
      for (int i = 0; i < n; ++i)
        bad += out[i] != 2.0 * (std::fabs(x[i]) - c.h) + 2.0 * x[i];
    }
    rep.check(c.id, run && bad == 0, "%s: %d mismatches", c.what, bad);
    if (c.kind == 0) {
      const int calls = run ? ReadCalls() : -1;
      rep.check("02.link.device_global", calls == n, "atomic counter defined in lib 1, incremented from lib 3: %d of %d",
                calls, n);
    }
  }

  // Host access by symbol to a device global defined in another translation unit
  int zero = 0, back = -1;
  bool sym = GPU_CHECK(GPU(MemcpyToSymbol)(GPU_SYMBOL(gCalls), &zero, sizeof(int), 0, GPU(MemcpyHostToDevice))) &&
             GPU_CHECK(GPU(MemcpyFromSymbol)(&back, GPU_SYMBOL(gCalls), sizeof(int), 0, GPU(MemcpyDeviceToHost)));
  rep.optional("02.link.host_symbol_other_tu", sym && back == 0,
               "MemcpyToSymbol from the program on a device global of lib 1");

  GPU(Free)(mem);
  GPU(Free)(table);
  GPU(Free)(dx);
  GPU(Free)(dout);
  return rep.finish();
}
