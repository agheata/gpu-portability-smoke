// 03 - Recursion through virtual calls, with a depth known only at run time.
// A chain of composite nodes (like a Boolean solid tree) calls its children through virtual functions, so the
// compiler cannot bound the stack; the program sets the device stack size through the runtime.
// Usage: 03_recursion <depth> <stack bytes per thread, 0 = keep the default>. run.sh runs several depths, each in its
// own process, since a stack overflow can leave the device context unusable.
#include "gpu.h"
#include "report.h"

#include <cmath>
#include <cstdlib>
#include <new>
#include <vector>

struct Node {
  __host__ __device__ virtual double Distance(double x) const = 0;
};

struct Leaf : Node {
  double h;
  __host__ __device__ explicit Leaf(double h_) : h(h_) {}
  __host__ __device__ double Distance(double x) const override { return fabs(x) - h; }
};

struct Union : Node {
  const Node *a, *b;
  __host__ __device__ Union(const Node *a_, const Node *b_) : a(a_), b(b_) {}
  __host__ __device__ double Distance(double x) const override { return fmin(a->Distance(x), b->Distance(x)); }
};

constexpr int kSlot = 32;
static_assert(sizeof(Leaf) <= kSlot && sizeof(Union) <= kSlot, "slot too small");

/// Chain of `depth` unions: U0(L0, U1(L1, ... U_{d-1}(L_{d-1}, L_d))); returns the root
__host__ __device__ const Node *BuildChain(unsigned char *mem, int depth)
{
  const Node *next = new (mem + 2 * depth * kSlot) Leaf(0.25 + depth);
  for (int k = depth - 1; k >= 0; --k) {
    const Node *leaf = new (mem + (2 * k + 1) * kSlot) Leaf(0.25 + k);
    next             = new (mem + 2 * k * kSlot) Union(leaf, next);
  }
  return next;
}

__global__ void Construct(unsigned char *mem, int depth, const Node **root)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) *root = BuildChain(mem, depth);
}

__global__ void Evaluate(const Node *const *root, const double *x, double *out, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (*root)->Distance(x[i]);
}

int main(int argc, char **argv)
{
  Report rep;
  const int depth    = argc > 1 ? std::atoi(argv[1]) : 16;
  const size_t stack = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 0;
  char id[64];
  std::snprintf(id, sizeof(id), "03.recursion.depth_%d%s", depth, stack ? "_stack_set" : "_stack_default");

  size_t limit = 0;
  if (stack) {
    const bool set = GPU_CHECK(GPU(DeviceSetLimit)(GPU(LimitStackSize), stack)) &&
                     GPU_CHECK(GPU(DeviceGetLimit)(&limit, GPU(LimitStackSize)));
    rep.check("03.recursion.stack_limit", set && limit >= stack, "requested %zu bytes per thread, got %zu", stack,
              limit);
  } else {
    GPU_CHECK(GPU(DeviceGetLimit)(&limit, GPU(LimitStackSize)));
  }

  const int n = 1 << 16;
  std::vector<double> x(n), out(n), ref(n);
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 * depth + 4.0 * depth * i / n;
  std::vector<unsigned char> hmem((2 * depth + 1) * kSlot);
  const Node *hroot = BuildChain(hmem.data(), depth);
  for (int i = 0; i < n; ++i)
    ref[i] = hroot->Distance(x[i]);

  unsigned char *mem = nullptr;
  const Node **root  = nullptr;
  double *dx = nullptr, *dout = nullptr;
  bool ok = GPU_CHECK(GPU(Malloc)((void **)&mem, hmem.size())) &&
            GPU_CHECK(GPU(Malloc)((void **)&root, sizeof(Node *))) &&
            GPU_CHECK(GPU(Malloc)((void **)&dx, n * sizeof(double))) &&
            GPU_CHECK(GPU(Malloc)((void **)&dout, n * sizeof(double))) &&
            GPU_CHECK(GPU(Memcpy)(dx, x.data(), n * sizeof(double), GPU(MemcpyHostToDevice)));
  if (ok) {
    Construct<<<1, 1>>>(mem, depth, root);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  }
  if (ok) {
    Evaluate<<<(n + 127) / 128, 128>>>(root, dx, dout, n);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)()) &&
         GPU_CHECK(GPU(Memcpy)(out.data(), dout, n * sizeof(double), GPU(MemcpyDeviceToHost)));
  }
  int bad = -1;
  if (ok) {
    bad = 0;
    for (int i = 0; i < n; ++i)
      bad += out[i] != ref[i];
  }
  if (stack)
    rep.check(id, ok && bad == 0, "recursion depth %d through virtual calls, stack %zu bytes per thread: %d mismatches",
              depth, limit, bad);
  else // with the default stack, deep recursion may legitimately fail: for information only
    rep.result(id, ok && bad == 0 ? "PASS" : "INFO", "recursion depth %d with the default stack (%zu bytes): %s", depth,
               limit, ok && bad == 0 ? "works" : "fails, a larger stack is needed");
  return rep.finish();
}
