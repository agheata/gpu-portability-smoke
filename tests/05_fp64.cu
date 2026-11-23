// 05 - Double precision: correctly rounded basic operations and FP64 throughput.
// Geometry tolerances need IEEE double precision: sqrt, division and fma must be correctly rounded (bit-identical to
// the host). A squared norm is also compared; its difference in ulp is reported for information, since compilers may
// contract a*b+c into fma differently on host and device. Throughput: independent fma chains
// in float and double, giving the FP64/FP32 rate.
#include "gpu.h"
#include "report.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

__global__ void BasicOps(const double *a, const double *b, const double *c, double *sq, double *dv, double *fm,
                         double *ray, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  sq[i] = sqrt(a[i]);
  dv[i] = a[i] / b[i];
  fm[i] = fma(a[i], b[i], c[i]);
  ray[i] = a[i] * a[i] + b[i] * b[i] + c[i] * c[i]; // squared norm, as in distance and safety formulas
}

template <typename T>
__global__ void FmaChains(T *out, int iters)
{
  T x0 = threadIdx.x * T(1e-3), x1 = x0 + T(1), x2 = x0 + T(2), x3 = x0 + T(3);
  T x4 = x0 + T(4), x5 = x0 + T(5), x6 = x0 + T(6), x7 = x0 + T(7);
  const T m = T(0.999999), s = T(1e-7);
  for (int k = 0; k < iters; ++k) {
    x0 = fma(x0, m, s), x1 = fma(x1, m, s), x2 = fma(x2, m, s), x3 = fma(x3, m, s);
    x4 = fma(x4, m, s), x5 = fma(x5, m, s), x6 = fma(x6, m, s), x7 = fma(x7, m, s);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

static int64_t Ulp(double x, double y)
{
  int64_t ix, iy;
  std::memcpy(&ix, &x, 8);
  std::memcpy(&iy, &y, 8);
  if (ix < 0) ix = INT64_MIN - ix;
  if (iy < 0) iy = INT64_MIN - iy;
  return ix > iy ? ix - iy : iy - ix;
}

template <typename T>
static double Gflops(int blocks, int threads, int iters, float &ms)
{
  T *out = nullptr;
  GPU(Event_t) e0, e1;
  if (!GPU_CHECK(GPU(Malloc)((void **)&out, (size_t)blocks * threads * sizeof(T)))) return 0;
  GPU_CHECK(GPU(EventCreate)(&e0));
  GPU_CHECK(GPU(EventCreate)(&e1));
  FmaChains<T><<<blocks, threads>>>(out, 16); // warm-up
  GPU_CHECK(GPU(EventRecord)(e0, 0));
  FmaChains<T><<<blocks, threads>>>(out, iters);
  GPU_CHECK(GPU(EventRecord)(e1, 0));
  const bool ok = GPU_CHECK(GPU(EventSynchronize)(e1)) && GPU_CHECK(GPU(EventElapsedTime)(&ms, e0, e1));
  GPU(EventDestroy)(e0);
  GPU(EventDestroy)(e1);
  GPU(Free)(out);
  return ok && ms > 0 ? 2.0 * 8.0 * iters * blocks * threads / (ms * 1e6) : 0;
}

int main()
{
  Report rep;
  const int n = 1 << 20;
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> u(-30.0, 30.0);
  std::vector<double> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) {
    a[i] = std::ldexp(1.0 + std::fabs(std::sin((double)i)), (int)u(rng)); // positive, wide range of exponents
    b[i] = std::ldexp(1.0 + std::fabs(std::cos((double)i)), (int)u(rng)) * (i % 2 ? 1 : -1);
    c[i] = u(rng);
  }
  double *d[7] = {nullptr};
  bool ok = true;
  for (auto &p : d)
    ok = ok && GPU_CHECK(GPU(Malloc)((void **)&p, n * sizeof(double)));
  ok = ok && GPU_CHECK(GPU(Memcpy)(d[0], a.data(), n * sizeof(double), GPU(MemcpyHostToDevice))) &&
       GPU_CHECK(GPU(Memcpy)(d[1], b.data(), n * sizeof(double), GPU(MemcpyHostToDevice))) &&
       GPU_CHECK(GPU(Memcpy)(d[2], c.data(), n * sizeof(double), GPU(MemcpyHostToDevice)));
  if (ok) {
    BasicOps<<<(n + 255) / 256, 256>>>(d[0], d[1], d[2], d[3], d[4], d[5], d[6], n);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  }
  std::vector<double> sq(n), dv(n), fm(n), ray(n);
  if (ok)
    ok = GPU_CHECK(GPU(Memcpy)(sq.data(), d[3], n * sizeof(double), GPU(MemcpyDeviceToHost))) &&
         GPU_CHECK(GPU(Memcpy)(dv.data(), d[4], n * sizeof(double), GPU(MemcpyDeviceToHost))) &&
         GPU_CHECK(GPU(Memcpy)(fm.data(), d[5], n * sizeof(double), GPU(MemcpyDeviceToHost))) &&
         GPU_CHECK(GPU(Memcpy)(ray.data(), d[6], n * sizeof(double), GPU(MemcpyDeviceToHost)));
  int bsq = -1, bdv = -1, bfm = -1;
  int64_t rayUlp = -1;
  if (ok) {
    bsq = bdv = bfm = 0;
    rayUlp          = 0;
    for (int i = 0; i < n; ++i) {
      bsq += sq[i] != std::sqrt(a[i]);
      bdv += dv[i] != a[i] / b[i];
      bfm += fm[i] != std::fma(a[i], b[i], c[i]);
      const double href = a[i] * a[i] + b[i] * b[i] + c[i] * c[i];
      const int64_t du  = Ulp(ray[i], href);
      if (du > rayUlp) rayUlp = du;
    }
  }
  rep.check("05.fp64.sqrt", ok && bsq == 0, "correctly rounded double sqrt: %d of %d differ from the host", bsq, n);
  rep.check("05.fp64.div", ok && bdv == 0, "correctly rounded double division: %d of %d differ from the host", bdv, n);
  rep.check("05.fp64.fma", ok && bfm == 0, "double fma: %d of %d differ from the host", bfm, n);
  rep.result("05.fp64.formula_ulp", "INFO", "squared norm a*a+b*b+c*c, max difference to the host: %lld ulp (fma contraction)",
             (long long)rayUlp);

  // Throughput
  int dev = 0;
  gpuDeviceProp prop;
  GPU_CHECK(GPU(GetDevice)(&dev));
  GPU_CHECK(GPU(GetDeviceProperties)(&prop, dev));
  const int blocks = prop.multiProcessorCount * 16, threads = 256;
  float ms32 = 0, ms64 = 0;
  const double g32 = Gflops<float>(blocks, threads, 4096, ms32);
  const double g64 = Gflops<double>(blocks, threads, 1024, ms64);
  rep.result("05.fp64.throughput", "INFO", "indicative FMA-chain microbenchmark: FP32 %.0f GFLOP/s, FP64 %.0f GFLOP/s, FP64/FP32 = 1/%.1f", g32,
             g64, g64 > 0 ? g32 / g64 : 0.0);
  rep.check("05.fp64.runs", g64 > 0, "double precision kernels run");

  for (auto p : d)
    GPU(Free)(p);
  return rep.finish();
}
