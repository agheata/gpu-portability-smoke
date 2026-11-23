// 01 - Virtual functions in device code (prerequisite).
// Polymorphic objects are built with placement new inside a kernel and called through base pointers, in the same
// kernel and in later launches; one override calls another virtual function (nested dispatch); the virtual
// destructor runs on the device. The same classes are used on the host for the reference values.
#include "gpu.h"
#include "report.h"

#include <cmath>
#include <new>
#include <vector>

struct Shape {
  __host__ __device__ virtual ~Shape() {}
  __host__ __device__ virtual double Distance(double x) const = 0;
  __host__ __device__ virtual int Id() const = 0;
};

struct Slab : Shape {
  double h;
  __host__ __device__ explicit Slab(double h_) : h(h_) {}
  __host__ __device__ double Distance(double x) const override { return fabs(x) - h; }
  __host__ __device__ int Id() const override { return 1; }
};

struct Shell : Shape {
  double r0, r1;
  __host__ __device__ Shell(double a, double b) : r0(a), r1(b) {}
  __host__ __device__ double Distance(double x) const override { return fmax(r0 - fabs(x), fabs(x) - r1); }
  __host__ __device__ int Id() const override { return 2; }
};

/// Calls the virtual function of another shape (nested dispatch, as composite solids do)
struct Scaled : Shape {
  const Shape *inner;
  double s;
  __host__ __device__ Scaled(const Shape *in, double sc) : inner(in), s(sc) {}
  __host__ __device__ double Distance(double x) const override { return inner->Distance(x / s) * s; }
  __host__ __device__ int Id() const override { return 3; }
};

constexpr int kShapes = 3;
constexpr int kSlot   = 64;
struct Storage {
  alignas(16) unsigned char bytes[kShapes][kSlot];
};
static_assert(sizeof(Slab) <= kSlot && sizeof(Shell) <= kSlot && sizeof(Scaled) <= kSlot, "slot too small");

__host__ __device__ void Build(Storage *st, Shape **table)
{
  table[0] = new (st->bytes[0]) Slab(1.5);
  table[1] = new (st->bytes[1]) Shell(0.5, 2.0);
  table[2] = new (st->bytes[2]) Scaled(table[0], 2.0);
}

__global__ void Construct(Storage *st, Shape **table)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) Build(st, table);
}

__global__ void ConstructAndCall(Storage *st, const double *x, double *out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    Shape *table[kShapes];
    Build(st, table);
    for (int k = 0; k < kShapes; ++k)
      out[k] = table[k]->Distance(x[k]);
  }
}

__global__ void Evaluate(Shape *const *table, const double *x, double *out, int *ids, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const Shape *s = table[i % kShapes]; // neighbouring threads take different types: divergent dispatch
    out[i]         = s->Distance(x[i]);
    ids[i]         = s->Id();
  }
}

__global__ void Destroy(Shape **table)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
    for (int k = 0; k < kShapes; ++k)
      table[k]->~Shape();
}

int main()
{
  Report rep;
  const int n = 1 << 20;
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = -4.0 + 8.0 * i / n;

  // Host reference built from the same classes
  Storage hst;
  Shape *htable[kShapes];
  Build(&hst, htable);
  std::vector<double> ref(n);
  std::vector<int> refId(n);
  for (int i = 0; i < n; ++i) {
    ref[i]   = htable[i % kShapes]->Distance(x[i]);
    refId[i] = htable[i % kShapes]->Id();
  }

  Storage *st = nullptr;
  Shape **table = nullptr;
  double *dx = nullptr, *dout = nullptr;
  int *dids  = nullptr;
  bool ok    = GPU_CHECK(GPU(Malloc)((void **)&st, sizeof(Storage))) &&
            GPU_CHECK(GPU(Malloc)((void **)&table, kShapes * sizeof(Shape *))) &&
            GPU_CHECK(GPU(Malloc)((void **)&dx, n * sizeof(double))) &&
            GPU_CHECK(GPU(Malloc)((void **)&dout, n * sizeof(double))) &&
            GPU_CHECK(GPU(Malloc)((void **)&dids, n * sizeof(int))) &&
            GPU_CHECK(GPU(Memcpy)(dx, x.data(), n * sizeof(double), GPU(MemcpyHostToDevice)));
  if (!ok) {
    rep.check("01.virtual.setup", false, "device allocation failed");
    return rep.finish();
  }

  // (a) construct and call in the same kernel
  ConstructAndCall<<<1, 1>>>(st, dx, dout);
  double same[kShapes] = {0, 0, 0};
  ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)()) &&
       GPU_CHECK(GPU(Memcpy)(same, dout, sizeof(same), GPU(MemcpyDeviceToHost)));
  bool match = ok;
  for (int k = 0; k < kShapes && ok; ++k)
    match = match && same[k] == htable[k]->Distance(x[k]);
  rep.check("01.virtual.same_kernel", match, "construct with placement new and call in one kernel");

  // (b) construct in one launch, call in later launches (vtables must survive between kernels)
  Construct<<<1, 1>>>(st, table);
  ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  int bad = -1, badId = -1;
  for (int launch = 0; launch < 2 && ok; ++launch) {
    Evaluate<<<(n + 255) / 256, 256>>>(table, dx, dout, dids, n);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  }
  std::vector<double> out(n);
  std::vector<int> ids(n);
  if (ok)
    ok = GPU_CHECK(GPU(Memcpy)(out.data(), dout, n * sizeof(double), GPU(MemcpyDeviceToHost))) &&
         GPU_CHECK(GPU(Memcpy)(ids.data(), dids, n * sizeof(int), GPU(MemcpyDeviceToHost)));
  if (ok) {
    bad = badId = 0;
    for (int i = 0; i < n; ++i) {
      bad += out[i] != ref[i];
      badId += ids[i] != refId[i];
    }
  }
  rep.check("01.virtual.later_kernel", ok && bad == 0 && badId == 0,
            "%d threads, 3 types interleaved, 2 launches after construction: %d value / %d type mismatches", n, bad,
            badId);
  int nested = 0, nestedRef = 0;
  for (int i = 0; i < n && ok; ++i) {
    nestedRef += refId[i] == 3;
    nested += refId[i] == 3 && ids[i] == 3 && out[i] == ref[i];
  }
  rep.check("01.virtual.nested", ok && nestedRef > 0 && nested == nestedRef,
            "virtual call from inside an override: %d correct", nested);

  // (c) virtual destructor on the device
  Destroy<<<1, 1>>>(table);
  ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  rep.check("01.virtual.destructor", ok, "virtual destructor called through base pointer");

  GPU(Free)(st);
  GPU(Free)(table);
  GPU(Free)(dx);
  GPU(Free)(dout);
  GPU(Free)(dids);
  return rep.finish();
}
