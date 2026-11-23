// 04 - Geometry transfer as VecGeom does it, in miniature.
// The host lays out a memory image at the final device addresses: plain data structs, arrays of device pointers,
// and empty slots for the polymorphic shells. One copy moves it to the device, one kernel constructs the shells in
// place with placement new (installing device vtables), and a navigation-like kernel locates points by walking the
// tree. The same classes build the host tree that gives the reference answers.
#include "gpu.h"
#include "report.h"

#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

struct ShapeData { // plain data, the same type on host and device
  double h[3];
};

struct Volume {
  const ShapeData *data;
  const Volume *const *daughters;
  int ndaughters, id;
  double t[3]; // position in the mother

  __host__ __device__ Volume(const ShapeData *d, const Volume *const *ds, int nd, int id_, const double *tr)
      : data(d), daughters(ds), ndaughters(nd), id(id_), t{tr[0], tr[1], tr[2]}
  {
  }
  __host__ __device__ virtual ~Volume() {}
  __host__ __device__ virtual bool Contains(const double *q) const = 0;

  /// Deepest volume containing p (given in the mother frame), -1 if outside
  __host__ __device__ int Locate(const double *p) const
  {
    const double q[3] = {p[0] - t[0], p[1] - t[1], p[2] - t[2]};
    if (!Contains(q)) return -1;
    for (int k = 0; k < ndaughters; ++k) {
      const int r = daughters[k]->Locate(q);
      if (r >= 0) return r;
    }
    return id;
  }
};

struct BoxVolume : Volume {
  using Volume::Volume;
  __host__ __device__ bool Contains(const double *q) const override
  {
    return fabs(q[0]) <= data->h[0] && fabs(q[1]) <= data->h[1] && fabs(q[2]) <= data->h[2];
  }
};

struct BallVolume : Volume {
  using Volume::Volume;
  __host__ __device__ bool Contains(const double *q) const override
  {
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] <= data->h[0] * data->h[0];
  }
};

enum Kind { kBox, kBall };
struct Desc {
  Kind kind;
  ShapeData data;
  int mother;
  double t[3];
};

/// Construction record: everything a shell constructor needs, as device addresses
struct Record {
  int kind, id, ndaughters;
  size_t slot;
  const ShapeData *data;
  const Volume *const *daughters;
  double t[3];
};

__global__ void ConstructShells(unsigned char *base, const Record *rec, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Record &r = rec[i];
  if (r.kind == kBox)
    new (base + r.slot) BoxVolume(r.data, r.daughters, r.ndaughters, r.id, r.t);
  else
    new (base + r.slot) BallVolume(r.data, r.daughters, r.ndaughters, r.id, r.t);
}

__global__ void LocatePoints(const Volume *world, const double *p, int *out, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = world->Locate(p + 3 * i);
}

static size_t Align(size_t at, size_t a) { return (at + a - 1) / a * a; }

int main()
{
  Report rep;
  // A small detector: world box, 4 boxes, 4 balls in each, a small box in each ball
  std::vector<Desc> desc;
  desc.push_back({kBox, {{100, 100, 100}}, -1, {0, 0, 0}});
  for (int a = 0; a < 4; ++a) {
    const int m1 = (int)desc.size();
    desc.push_back({kBox, {{40, 40, 40}}, 0, {a & 1 ? 50. : -50., a & 2 ? 50. : -50., 0}});
    for (int b = 0; b < 4; ++b) {
      const int m2 = (int)desc.size();
      desc.push_back({kBall, {{15, 0, 0}}, m1, {b & 1 ? 20. : -20., b & 2 ? 20. : -20., 0}});
      desc.push_back({kBox, {{5, 5, 5}}, m2, {0, 0, 0}});
    }
  }
  const int nv = (int)desc.size();
  std::vector<std::vector<int>> kids(nv);
  for (int v = 1; v < nv; ++v)
    kids[desc[v].mother].push_back(v);

  // Host tree from the same classes
  std::vector<std::vector<const Volume *>> hdaughters(nv);
  std::vector<Volume *> hvol(nv);
  for (int v = nv - 1; v >= 0; --v) {
    for (int k : kids[v])
      hdaughters[v].push_back(hvol[k]);
    if (desc[v].kind == kBox)
      hvol[v] = new BoxVolume(&desc[v].data, hdaughters[v].data(), (int)kids[v].size(), v, desc[v].t);
    else
      hvol[v] = new BallVolume(&desc[v].data, hdaughters[v].data(), (int)kids[v].size(), v, desc[v].t);
  }

  // Layout pass: offsets of data, daughter arrays and shell slots in the image
  const size_t slotSize = Align(sizeof(BoxVolume) > sizeof(BallVolume) ? sizeof(BoxVolume) : sizeof(BallVolume), 16);
  std::vector<size_t> dataAt(nv), kidsAt(nv), slotAt(nv);
  size_t size = 0;
  for (int v = 0; v < nv; ++v) {
    dataAt[v] = size = Align(size, alignof(ShapeData));
    size += sizeof(ShapeData);
    kidsAt[v] = size = Align(size, alignof(Volume *));
    size += kids[v].size() * sizeof(Volume *);
  }
  for (int v = 0; v < nv; ++v) {
    slotAt[v] = size = Align(size, 16);
    size += slotSize;
  }

  unsigned char *base = nullptr;
  Record *drec = nullptr;
  if (!GPU_CHECK(GPU(Malloc)((void **)&base, size)) || !GPU_CHECK(GPU(Malloc)((void **)&drec, nv * sizeof(Record)))) {
    rep.check("04.image.setup", false, "device allocation failed");
    return rep.finish();
  }

  // Image: data copied byte-wise, pointer arrays written as device addresses, shell slots left empty
  std::vector<unsigned char> image(size, 0);
  std::vector<Record> rec(nv);
  for (int v = 0; v < nv; ++v) {
    std::memcpy(image.data() + dataAt[v], &desc[v].data, sizeof(ShapeData));
    for (size_t k = 0; k < kids[v].size(); ++k) {
      const void *dev = base + slotAt[kids[v][k]];
      std::memcpy(image.data() + kidsAt[v] + k * sizeof(Volume *), &dev, sizeof(dev));
    }
    rec[v] = {desc[v].kind, v, (int)kids[v].size(), slotAt[v], (const ShapeData *)(base + dataAt[v]),
              (const Volume *const *)(base + kidsAt[v]), {desc[v].t[0], desc[v].t[1], desc[v].t[2]}};
  }
  bool ok = GPU_CHECK(GPU(Memcpy)(base, image.data(), size, GPU(MemcpyHostToDevice))) &&
            GPU_CHECK(GPU(Memcpy)(drec, rec.data(), nv * sizeof(Record), GPU(MemcpyHostToDevice)));
  rep.check("04.image.copy", ok, "%d volumes, image of %zu bytes copied in one call", nv, size);

  if (ok) {
    ConstructShells<<<(nv + 63) / 64, 64>>>(base, drec, nv);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)());
  }
  rep.check("04.image.construct", ok, "shells constructed in place by one kernel (placement new, device vtables)");

  // Locate points on a grid and compare with the host tree
  std::vector<double> pts;
  for (double x = -105; x <= 105; x += 1.5)
    for (double y = -105; y <= 105; y += 1.5)
      for (double z = -45; z <= 45; z += 7.5) {
        pts.push_back(x);
        pts.push_back(y);
        pts.push_back(z);
      }
  const int np = (int)pts.size() / 3;
  std::vector<int> got(np, -2);
  double *dp = nullptr;
  int *dout  = nullptr;
  if (ok) {
    ok = GPU_CHECK(GPU(Malloc)((void **)&dp, pts.size() * sizeof(double))) &&
         GPU_CHECK(GPU(Malloc)((void **)&dout, np * sizeof(int))) &&
         GPU_CHECK(GPU(Memcpy)(dp, pts.data(), pts.size() * sizeof(double), GPU(MemcpyHostToDevice)));
  }
  if (ok) {
    LocatePoints<<<(np + 127) / 128, 128>>>((const Volume *)(base + slotAt[0]), dp, dout, np);
    ok = GPU_CHECK(GPU(GetLastError)()) && GPU_CHECK(GPU(DeviceSynchronize)()) &&
         GPU_CHECK(GPU(Memcpy)(got.data(), dout, np * sizeof(int), GPU(MemcpyDeviceToHost)));
  }
  int bad = -1, inside = 0;
  if (ok) {
    bad = 0;
    for (int i = 0; i < np; ++i) {
      const int ref = hvol[0]->Locate(&pts[3 * i]);
      bad += got[i] != ref;
      inside += ref > 0;
    }
  }
  rep.check("04.image.locate", ok && bad == 0, "%d points located by walking the device tree: %d mismatches with the host "
            "tree (%d inside daughters)", np, bad, inside);

  for (Volume *v : hvol)
    delete v;
  GPU(Free)(base);
  GPU(Free)(drec);
  GPU(Free)(dp);
  GPU(Free)(dout);
  return rep.finish();
}
