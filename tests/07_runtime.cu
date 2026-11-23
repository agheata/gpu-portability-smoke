// 07 - Runtime API used by VecGeom and AdePT.
// Device query, allocation and copies, pinned host memory, streams and events, copies by symbol, launch-error
// reporting, dynamic shared memory, device printf, managed memory (optional).
// "07_runtime trap" runs a kernel that traps; run.sh runs it in its own process and checks that the host gets an
// error instead of a hang (some runtimes abort the process instead, reported as WARN).
#include "gpu.h"
#include "report.h"

#include <cstring>
#include <string>
#include <vector>

__device__ int gSymbol[4];

__global__ void Scale(double *v, double s, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] *= s;
}

__global__ void BlockSum(const int *in, int *out, int n)
{
  extern __shared__ int buf[];
  int i            = blockIdx.x * blockDim.x + threadIdx.x;
  buf[threadIdx.x] = i < n ? in[i] : 0;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, buf[0]);
}

__global__ void SymbolKernel()
{
  if (threadIdx.x < 4) gSymbol[threadIdx.x] *= 2;
}

__global__ void Hello() { printf("device printf ok from block %d\n", blockIdx.x); }

__global__ void Trap() { GPU_TRAP(); }

int main(int argc, char **argv)
{
  Report rep;
  if (argc > 1 && std::string(argv[1]) == "trap") {
    Trap<<<1, 1>>>();
    const GPU(Error_t) launch = GPU(GetLastError)();
    const GPU(Error_t) sync   = GPU(DeviceSynchronize)();
    rep.optional("07.runtime.trap", launch != GPU(Success) || sync != GPU(Success),
                 "device trap reported to the host as: %s", GPU(GetErrorString)(sync));
    return 0;
  }

  int count = 0, dev = 0;
  gpuDeviceProp p;
  bool ok = GPU_CHECK(GPU(GetDeviceCount)(&count)) && count > 0 && GPU_CHECK(GPU(GetDevice)(&dev)) &&
            GPU_CHECK(GPU(GetDeviceProperties)(&p, dev));
  rep.check("07.runtime.device_query", ok, "%d device(s)", count);
  if (!ok) return rep.finish();
  rep.result("07.runtime.device", "INFO", "%s, %s, %d multiprocessors, %.1f GB, warp/wavefront %d", p.name,
             gpu_arch(p).c_str(), p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, p.warpSize);

  // Allocation, memset and copies in all directions
  const int n = 1 << 20;
  std::vector<double> h(n), back(n);
  for (int i = 0; i < n; ++i)
    h[i] = i;
  double *d1 = nullptr, *d2 = nullptr;
  ok = GPU_CHECK(GPU(Malloc)((void **)&d1, n * sizeof(double))) &&
       GPU_CHECK(GPU(Malloc)((void **)&d2, n * sizeof(double))) && GPU_CHECK(GPU(Memset)(d2, 0, n * sizeof(double))) &&
       GPU_CHECK(GPU(Memcpy)(d1, h.data(), n * sizeof(double), GPU(MemcpyHostToDevice))) &&
       GPU_CHECK(GPU(Memcpy)(d2, d1, n * sizeof(double), GPU(MemcpyDeviceToDevice))) &&
       GPU_CHECK(GPU(Memcpy)(back.data(), d2, n * sizeof(double), GPU(MemcpyDeviceToHost)));
  rep.check("07.runtime.malloc_memcpy", ok && back == h, "Malloc, Memset, Memcpy host/device/device/host");

  // Pinned memory, two streams, async copies, events and a cross-stream dependency
  double *pinned = nullptr;
  GPU(Stream_t) s1, s2;
  GPU(Event_t) e0, e1, done;
  float ms  = -1;
  bool str = GPU_CHECK(gpu_malloc_host((void **)&pinned, n * sizeof(double))) &&
             GPU_CHECK(GPU(StreamCreate)(&s1)) && GPU_CHECK(GPU(StreamCreate)(&s2)) &&
             GPU_CHECK(GPU(EventCreate)(&e0)) && GPU_CHECK(GPU(EventCreate)(&e1)) &&
             GPU_CHECK(GPU(EventCreate)(&done));
  if (str) {
    std::memcpy(pinned, h.data(), n * sizeof(double));
    str = GPU_CHECK(GPU(EventRecord)(e0, s1)) &&
          GPU_CHECK(GPU(MemcpyAsync)(d1, pinned, n * sizeof(double), GPU(MemcpyHostToDevice), s1)) &&
          GPU_CHECK(GPU(EventRecord)(done, s1)) && GPU_CHECK(GPU(StreamWaitEvent)(s2, done, 0));
    Scale<<<(n + 255) / 256, 256, 0, s2>>>(d1, 2.0, n);
    str = str && GPU_CHECK(GPU(GetLastError)()) &&
          GPU_CHECK(GPU(MemcpyAsync)(pinned, d1, n * sizeof(double), GPU(MemcpyDeviceToHost), s2)) &&
          GPU_CHECK(GPU(EventRecord)(e1, s2)) && GPU_CHECK(GPU(EventSynchronize)(e1)) &&
          GPU_CHECK(GPU(EventElapsedTime)(&ms, e0, e1)) && GPU_CHECK(GPU(StreamSynchronize)(s1));
    for (int i = 0; i < n && str; ++i)
      str = pinned[i] == 2.0 * h[i];
  }
  rep.check("07.runtime.streams_events", str, "pinned memory, 2 streams, async copies, event dependency, %.3f ms", ms);

  // Copies by symbol
  int sym[4] = {1, 2, 3, 4}, symBack[4] = {0, 0, 0, 0};
  bool sy = GPU_CHECK(GPU(MemcpyToSymbol)(GPU_SYMBOL(gSymbol), sym, sizeof(sym), 0, GPU(MemcpyHostToDevice)));
  SymbolKernel<<<1, 32>>>();
  sy = sy && GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)()) &&
       GPU_CHECK(GPU(MemcpyFromSymbol)(symBack, GPU_SYMBOL(gSymbol), sizeof(sym), 0, GPU(MemcpyDeviceToHost)));
  rep.check("07.runtime.symbols", sy && symBack[0] == 2 && symBack[3] == 8, "MemcpyToSymbol / MemcpyFromSymbol");

  // Launch errors are reported, and the context stays usable
  Scale<<<1, 1 << 20>>>(d1, 1.0, n); // block size beyond any device limit
  const GPU(Error_t) bad = GPU(GetLastError)();
  Scale<<<1, 32>>>(d1, 1.0, n);
  const bool after = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  rep.check("07.runtime.launch_error", bad != GPU(Success) && after, "invalid launch reported as: %s",
            GPU(GetErrorString)(bad));

  // Dynamic shared memory and barriers
  std::vector<int> ones(n, 1);
  int *din = nullptr, *dsum = nullptr, sum = -1;
  bool sh = GPU_CHECK(GPU(Malloc)((void **)&din, n * sizeof(int))) && GPU_CHECK(GPU(Malloc)((void **)&dsum, sizeof(int))) &&
            GPU_CHECK(GPU(Memcpy)(din, ones.data(), n * sizeof(int), GPU(MemcpyHostToDevice))) &&
            GPU_CHECK(GPU(Memset)(dsum, 0, sizeof(int)));
  BlockSum<<<n / 256, 256, 256 * sizeof(int)>>>(din, dsum, n);
  sh = sh && GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)()) &&
       GPU_CHECK(GPU(Memcpy)(&sum, dsum, sizeof(int), GPU(MemcpyDeviceToHost)));
  rep.check("07.runtime.shared_memory", sh && sum == n, "dynamic shared memory reduction: %d", sum);

  // Device printf (run.sh also looks for the printed line)
  Hello<<<1, 1>>>();
  rep.check("07.runtime.printf", GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)()),
            "kernel with printf ran");

  // Managed memory (optional)
  double *managed = nullptr;
  bool mg         = GPU_CHECK(GPU(MallocManaged)((void **)&managed, 1024 * sizeof(double)));
  if (mg) {
    for (int i = 0; i < 1024; ++i)
      managed[i] = i;
    Scale<<<4, 256>>>(managed, 3.0, 1024);
    mg = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
    for (int i = 0; i < 1024 && mg; ++i)
      mg = managed[i] == 3.0 * i;
    GPU(Free)(managed);
  }
  rep.optional("07.runtime.managed", mg, "MallocManaged written on host, scaled on device, read on host");

  GPU(Free)(d1);
  GPU(Free)(d2);
  GPU(Free)(din);
  GPU(Free)(dsum);
  if (pinned) gpu_free_host(pinned);
  return rep.finish();
}
