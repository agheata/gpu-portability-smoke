// 06 - Atomics.
// Required: integer add, compare-and-swap and exchange on device memory under heavy contention, as AdePT queues and
// counters use them. Optional: native double atomicAdd, and atomics on pinned host memory mapped into the device
// (on some GPUs only part of the operations work over the bus; AdePT keeps all production atomics in device memory).
#include "gpu.h"
#include "report.h"

#include <cstdlib>
#include <cstring>
#include <vector>

struct Counters {
  unsigned int add32;
  int max32;
  unsigned long long add64;
  unsigned long long casDouble; // bit pattern of a double accumulated with a CAS loop
  double addDouble;
  unsigned int exch;
  unsigned int hist[1024];
};

constexpr int kCasThreads = 1 << 16;

__device__ void CasAddDouble(unsigned long long *addr, double v)
{
  unsigned long long old = *addr, assumed;
  do {
    assumed = old;
    old     = atomicCAS(addr, assumed, __double_as_longlong(__longlong_as_double(assumed) + v));
  } while (assumed != old);
}

__global__ void DeviceAtomics(Counters *c, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&c->add32, 1u);
  atomicMax(&c->max32, i);
  atomicAdd(&c->add64, 1ull);
  if (i < kCasThreads) CasAddDouble(&c->casDouble, 1.0); // a spin loop on one address: keep the contention bounded
  atomicExch(&c->exch, 7u);
  atomicAdd(&c->hist[i % 1024], 1u);
}

__global__ void DoubleAdd(Counters *c, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&c->addDouble, 1.0);
}

__global__ void HostAtomics(unsigned int *h, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&h[0], 1u);
  atomicSub(&h[1], 1u);
  atomicOr(&h[2], 1u << (i % 32));
  atomicCAS(&h[3], 0u, 5u);
}

int main(int argc, char **argv)
{
  Report rep;
  const int n = argc > 1 ? std::atoi(argv[1]) : 1 << 22;
  Counters *c = nullptr;
  bool ok     = GPU_CHECK(GPU(Malloc)((void **)&c, sizeof(Counters))) &&
            GPU_CHECK(GPU(Memset)(c, 0, sizeof(Counters)));
  if (ok) {
    DeviceAtomics<<<(n + 255) / 256, 256>>>(c, n);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  }
  Counters h{};
  if (ok) ok = GPU_CHECK(GPU(Memcpy)(&h, c, sizeof(Counters), GPU(MemcpyDeviceToHost)));
  bool hist = ok;
  for (int k = 0; k < 1024 && ok; ++k)
    hist = hist && h.hist[k] == (unsigned)(n / 1024);
  double cas = 0;
  memcpy(&cas, &h.casDouble, sizeof(double));
  rep.check("06.atomics.add32", ok && h.add32 == (unsigned)n, "%d threads on one address: %u", n, h.add32);
  rep.check("06.atomics.add64", ok && h.add64 == (unsigned long long)n, "64-bit add: %llu", h.add64);
  rep.check("06.atomics.max", ok && h.max32 == n - 1, "atomicMax: %d", h.max32);
  const int nCas = n < kCasThreads ? n : kCasThreads;
  rep.check("06.atomics.cas", ok && cas == (double)nCas, "CAS loop adding doubles, %d threads on one address: %.0f", nCas,
            cas);
  rep.check("06.atomics.exch", ok && h.exch == 7u, "atomicExch");
  rep.check("06.atomics.spread", hist, "1024 addresses, %d adds each", n / 1024);

  if (ok) {
    DoubleAdd<<<(n + 255) / 256, 256>>>(c, n);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)()) &&
         GPU_CHECK(GPU(Memcpy)(&h, c, sizeof(Counters), GPU(MemcpyDeviceToHost)));
  }
  rep.optional("06.atomics.add_f64", ok && h.addDouble == (double)n, "native double atomicAdd: %.0f", h.addDouble);

  // Pinned host memory mapped into the device
  unsigned int *pinned = nullptr;
  const int nh         = 1 << 16;
  bool mapped          = GPU_CHECK(gpu_alloc_mapped((void **)&pinned, 4 * sizeof(unsigned int)));
  if (mapped) {
    pinned[0] = 0, pinned[1] = (unsigned)nh, pinned[2] = 0, pinned[3] = 0;
    HostAtomics<<<(nh + 255) / 256, 256>>>(pinned, nh);
    mapped = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  }
  rep.optional("06.atomics.host_add", mapped && pinned[0] == (unsigned)nh, "add on mapped pinned host memory: %u of %d",
               mapped ? pinned[0] : 0, nh);
  rep.optional("06.atomics.host_sub", mapped && pinned[1] == 0u, "sub on mapped pinned host memory: %u left (0 expected)",
               mapped ? pinned[1] : 0);
  rep.optional("06.atomics.host_or", mapped && pinned[2] == 0xffffffffu, "or on mapped pinned host memory: 0x%08x",
               mapped ? pinned[2] : 0);
  rep.optional("06.atomics.host_cas", mapped && pinned[3] == 5u, "CAS on mapped pinned host memory");

  if (pinned) gpu_free_host(pinned);
  GPU(Free)(c);
  return rep.finish();
}
