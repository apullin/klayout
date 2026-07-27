/*
 * Conservative device-resident antenna clean certificates.
 */

#include "antenna_clean_certificate_gpu.cuh"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace {

namespace acc = klayout_cuda::antenna_clean_certificate;
namespace ac = klayout_cuda::antenna_connectivity;

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMalformed = 1u;
constexpr std::uint32_t kOverflow = 2u;
constexpr std::uint32_t kCapacity = 4u;

class CudaFailure : public std::runtime_error
{
public:
  explicit CudaFailure(const char *operation)
      : std::runtime_error(operation)
  {
  }
};

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) throw CudaFailure(operation);
}

bool checked_add_host(
    std::uint64_t first, std::uint64_t second,
    std::uint64_t *result)
{
  if (first > UINT64_MAX - second) return false;
  *result = first + second;
  return true;
}

bool checked_multiply_host(
    std::uint64_t first, std::uint64_t second,
    std::uint64_t *result)
{
  if (first && second > UINT64_MAX / first) return false;
  *result = first * second;
  return true;
}

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 0;
  const std::uint64_t blocks =
      (count + kThreads - 1) / kThreads;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(blocks, 65535));
}

bool valid_level(acc::MetalLevel level)
{
  return level == acc::MetalLevel::metal1 ||
         level == acc::MetalLevel::metal2 ||
         level == acc::MetalLevel::metal3 ||
         level == acc::MetalLevel::metal4;
}

struct DeviceGateScalars
{
  unsigned long long active_memberships = 0;
  unsigned long long candidate_visits = 0;
  unsigned long long positive_intersections = 0;
  unsigned long long gate_owners = 0;
  unsigned int status = 0;
};

struct DeviceBounds
{
  // Signed order is mapped to unsigned order with the sign bit flipped.
  unsigned long long minimum_x = ULLONG_MAX;
  unsigned long long minimum_y = ULLONG_MAX;
  unsigned long long maximum_x = 0;
  unsigned long long maximum_y = 0;
};

struct Grid
{
  std::int64_t base_x = 0;
  std::int64_t base_y = 0;
  std::int64_t cell_size = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

struct DeviceCheckpointScalars
{
  unsigned long long roots = 0;
  unsigned long long roots_with_metal = 0;
  unsigned long long roots_without_gate = 0;
  unsigned long long gate_roots_without_metal = 0;
  unsigned long long ratio_certified_roots = 0;
  unsigned long long uncertain_roots = 0;
  unsigned int status = 0;
};

__device__ bool valid_rectangle(
    const ac::RectI64 &rectangle, std::uint64_t owner_count)
{
  return rectangle.left < rectangle.right &&
         rectangle.bottom < rectangle.top &&
         rectangle.owner < owner_count;
}

__device__ bool checked_area(
    const ac::RectI64 &rectangle,
    unsigned long long *area)
{
  const unsigned long long width =
      static_cast<unsigned long long>(rectangle.right) -
      static_cast<unsigned long long>(rectangle.left);
  const unsigned long long height =
      static_cast<unsigned long long>(rectangle.top) -
      static_cast<unsigned long long>(rectangle.bottom);
  if (width && height > ULLONG_MAX / width) return false;
  *area = width * height;
  return true;
}

__device__ bool checked_intersection_area(
    const ac::RectI64 &poly, const ac::RectI64 &active,
    unsigned long long *area)
{
  const std::int64_t left =
      poly.left > active.left ? poly.left : active.left;
  const std::int64_t right =
      poly.right < active.right ? poly.right : active.right;
  const std::int64_t bottom =
      poly.bottom > active.bottom ? poly.bottom : active.bottom;
  const std::int64_t top =
      poly.top < active.top ? poly.top : active.top;
  if (left >= right || bottom >= top) {
    *area = 0;
    return true;
  }
  const unsigned long long width =
      static_cast<unsigned long long>(right) -
      static_cast<unsigned long long>(left);
  const unsigned long long height =
      static_cast<unsigned long long>(top) -
      static_cast<unsigned long long>(bottom);
  if (width && height > ULLONG_MAX / width) return false;
  *area = width * height;
  return true;
}

__device__ unsigned long long ordered_signed(std::int64_t value)
{
  return static_cast<unsigned long long>(value) ^
         (1ull << 63);
}

std::int64_t decode_ordered_signed(unsigned long long value)
{
  constexpr unsigned long long sign = 1ull << 63;
  return value < sign
             ? INT64_MIN + static_cast<std::int64_t>(value)
             : static_cast<std::int64_t>(value - sign);
}

std::int64_t floor_div_host(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

__device__ std::int64_t floor_div_device(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

__device__ bool grid_span(
    const ac::RectI64 &rectangle, const Grid &grid,
    std::int64_t *x0, std::int64_t *y0,
    std::int64_t *x1, std::int64_t *y1)
{
  *x0 = floor_div_device(rectangle.left, grid.cell_size);
  *y0 = floor_div_device(rectangle.bottom, grid.cell_size);
  // Rectangles are positive-area half-open covers in integer DBU space.
  // right-1/top-1 select the last occupied DBU cell and also ensure the loop
  // induction below can never increment INT64_MAX.
  *x1 = floor_div_device(rectangle.right - 1, grid.cell_size);
  *y1 = floor_div_device(rectangle.top - 1, grid.cell_size);
  const std::int64_t maximum_x =
      grid.base_x +
      static_cast<std::int64_t>(grid.width - 1);
  const std::int64_t maximum_y =
      grid.base_y +
      static_cast<std::int64_t>(grid.height - 1);
  return *x0 >= grid.base_x && *y0 >= grid.base_y &&
         *x1 <= maximum_x && *y1 <= maximum_y &&
         *x0 <= *x1 && *y0 <= *y1;
}

__device__ std::uint64_t grid_index(
    const Grid &grid, std::int64_t x, std::int64_t y)
{
  return static_cast<std::uint64_t>(y - grid.base_y) *
             grid.width +
         static_cast<std::uint64_t>(x - grid.base_x);
}

__device__ bool atomic_add_checked(
    unsigned long long *destination,
    unsigned long long value)
{
  unsigned long long observed =
      atomicCAS(destination, 0ull, 0ull);
  while (true) {
    if (observed > ULLONG_MAX - value) return false;
    const unsigned long long desired = observed + value;
    const unsigned long long prior =
        atomicCAS(destination, observed, desired);
    if (prior == observed) return true;
    observed = prior;
  }
}

__device__ bool atomic_add_limited(
    unsigned long long *destination,
    unsigned long long value, unsigned long long limit)
{
  unsigned long long observed =
      atomicCAS(destination, 0ull, 0ull);
  while (true) {
    if (observed > limit ||
        value > limit - observed) {
      return false;
    }
    const unsigned long long desired = observed + value;
    const unsigned long long prior =
        atomicCAS(destination, observed, desired);
    if (prior == observed) return true;
    observed = prior;
  }
}

__global__ void collect_bounds_kernel(
    const ac::RectI64 *rectangles, std::uint64_t count,
    std::uint64_t owner_count, DeviceBounds *bounds,
    unsigned int *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < count;
       index += stride) {
    if (!valid_rectangle(rectangles[index], owner_count)) {
      atomicOr(status, kMalformed);
      continue;
    }
    const ac::RectI64 rectangle = rectangles[index];
    atomicMin(&bounds->minimum_x, ordered_signed(rectangle.left));
    atomicMin(&bounds->minimum_y, ordered_signed(rectangle.bottom));
    atomicMax(
        &bounds->maximum_x,
        ordered_signed(rectangle.right - 1));
    atomicMax(
        &bounds->maximum_y,
        ordered_signed(rectangle.top - 1));
  }
}

__global__ void count_grid_kernel(
    const ac::RectI64 *active, std::uint64_t active_count,
    std::uint64_t owner_count, Grid grid,
    std::uint32_t max_cell_members,
    unsigned long long max_memberships,
    std::uint32_t *counts, DeviceGateScalars *scalars)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < active_count;
       index += stride) {
    const ac::RectI64 rectangle = active[index];
    if (!valid_rectangle(rectangle, owner_count)) {
      atomicOr(&scalars->status, kMalformed);
      continue;
    }
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!grid_span(rectangle, grid, &x0, &y0, &x1, &y1)) {
      atomicOr(&scalars->status, kMalformed);
      continue;
    }
    const unsigned long long columns =
        static_cast<unsigned long long>(x1 - x0 + 1);
    const unsigned long long rows =
        static_cast<unsigned long long>(y1 - y0 + 1);
    if (columns && rows > ULLONG_MAX / columns) {
      atomicOr(&scalars->status, kOverflow);
      continue;
    }
    const unsigned long long local = columns * rows;
    if (!atomic_add_limited(
            &scalars->active_memberships, local,
            max_memberships)) {
      atomicOr(&scalars->status, kCapacity);
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t prior = atomicAdd(counts + cell, 1u);
        if (prior == UINT32_MAX ||
            prior >= max_cell_members) {
          atomicOr(&scalars->status, kCapacity);
        }
      }
    }
  }
}

__global__ void fill_grid_kernel(
    const ac::RectI64 *active, std::uint64_t active_count,
    Grid grid, std::uint32_t *cursors,
    std::uint32_t *members, unsigned int *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < active_count;
       index += stride) {
    const ac::RectI64 rectangle = active[index];
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!grid_span(rectangle, grid, &x0, &y0, &x1, &y1)) {
      atomicOr(status, kMalformed);
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t slot =
            atomicAdd(cursors + cell, 1u);
        members[slot] = static_cast<std::uint32_t>(index);
      }
    }
  }
}

__global__ void count_gate_query_visits_grid_kernel(
    const ac::RectI64 *poly, std::uint64_t poly_count,
    Grid grid,
    const std::uint32_t *active_counts,
    unsigned long long max_query_visits,
    DeviceGateScalars *scalars)
{
  __shared__ unsigned long long thread_visits[kThreads];
  __shared__ unsigned int thread_status[kThreads];

  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  unsigned long long local_visits = 0;
  unsigned int local_status = 0;
  for (std::uint64_t poly_id = first; poly_id < poly_count;
       poly_id += stride) {
    const ac::RectI64 poly_rectangle = poly[poly_id];
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!grid_span(
            poly_rectangle, grid, &x0, &y0, &x1, &y1)) {
      local_status |= kMalformed;
      continue;
    }
    for (std::int64_t y = y0;
         y <= y1 && !(local_status & kCapacity); ++y) {
      for (std::int64_t x = x0;
           x <= x1 && !(local_status & kCapacity); ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t count = active_counts[cell];
        if (local_visits > max_query_visits ||
            count > max_query_visits - local_visits) {
          local_status |= kCapacity;
        } else {
          local_visits += count;
        }
      }
    }
    if (local_status & kCapacity) break;
  }

  thread_visits[threadIdx.x] = local_visits;
  thread_status[threadIdx.x] = local_status;
  __syncthreads();

  // The block leader performs a bounded reduction and makes one reservation
  // in the global counter.  This replaces one contended 64-bit CAS per
  // POLY/grid-cell membership with at most one CAS per block while preserving
  // the exact successful counter and fail-closed query-visit ceiling.
  if (threadIdx.x == 0) {
    unsigned long long block_visits = 0;
    unsigned int block_status = 0;
    for (std::uint32_t lane = 0; lane < blockDim.x; ++lane) {
      block_status |= thread_status[lane];
      const unsigned long long visits = thread_visits[lane];
      if (block_visits > max_query_visits ||
          visits > max_query_visits - block_visits) {
        block_status |= kCapacity;
      } else {
        block_visits += visits;
      }
    }
    if (!(block_status & kCapacity) && block_visits &&
        !atomic_add_limited(
            &scalars->candidate_visits, block_visits,
            max_query_visits)) {
      block_status |= kCapacity;
    }
    if (block_status) atomicOr(&scalars->status, block_status);
  }
}

__global__ void gate_intersections_grid_kernel(
    const ac::RectI64 *poly, std::uint64_t poly_count,
    const ac::RectI64 *active, Grid grid,
    const std::uint32_t *active_counts,
    const std::uint32_t *active_offsets,
    const std::uint32_t *active_members,
    std::uint32_t *gate_present,
    unsigned long long *gate_lower,
    DeviceGateScalars *scalars)
{
  // The preceding visit-count kernel is an exact admission pass.  CUDA
  // launches in the same stream are ordered, so a failed cap or malformed
  // span prevents all predicate work without a host round trip.
  if (scalars->status) return;

  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t poly_id = first; poly_id < poly_count;
       poly_id += stride) {
    const ac::RectI64 poly_rectangle = poly[poly_id];
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!grid_span(
            poly_rectangle, grid, &x0, &y0, &x1, &y1)) {
      atomicOr(&scalars->status, kMalformed);
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t begin = active_offsets[cell];
        const std::uint32_t count = active_counts[cell];
        const std::uint32_t end = begin + count;
        for (std::uint32_t slot = begin; slot < end; ++slot) {
          const ac::RectI64 active_rectangle =
              active[active_members[slot]];
          unsigned long long area = 0;
          if (!checked_intersection_area(
                  poly_rectangle, active_rectangle, &area)) {
            atomicOr(&scalars->status, kOverflow);
            continue;
          }
          if (!area) continue;
          const std::int64_t intersection_left =
              poly_rectangle.left > active_rectangle.left
                  ? poly_rectangle.left : active_rectangle.left;
          const std::int64_t intersection_bottom =
              poly_rectangle.bottom > active_rectangle.bottom
                  ? poly_rectangle.bottom : active_rectangle.bottom;
          // Rectangles can span many cells.  The cell containing the
          // positive intersection's lower-left point owns the pair exactly
          // once, including pairs originating in distinct hierarchy contexts.
          if (floor_div_device(
                  intersection_left, grid.cell_size) != x ||
              floor_div_device(
                  intersection_bottom, grid.cell_size) != y) {
            continue;
          }
          atomicExch(
              gate_present + poly_rectangle.owner,
              static_cast<std::uint32_t>(1));
          atomicMax(
              gate_lower + poly_rectangle.owner, area);
          atomicAdd(
              &scalars->positive_intersections, 1ull);
        }
      }
    }
  }
}

__global__ void count_gate_owners_kernel(
    const std::uint32_t *gate_present,
    std::uint64_t count, DeviceGateScalars *scalars)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < count;
       index += stride) {
    if (gate_present[index]) {
      atomicAdd(&scalars->gate_owners, 1ull);
    }
  }
}

__global__ void validate_labels_kernel(
    const std::uint32_t *labels, std::uint64_t count,
    unsigned int *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < count;
       index += stride) {
    const std::uint32_t root = labels[index];
    if (root >= count || root > index) {
      atomicOr(status, kMalformed);
      continue;
    }
    if (labels[root] != root) {
      atomicOr(status, kMalformed);
    }
  }
}

__global__ void reduce_gate_annotations_kernel(
    const std::uint32_t *owner_gate_present,
    const unsigned long long *owner_gate_lower,
    std::uint64_t annotation_count,
    const std::uint32_t *labels, std::uint64_t label_count,
    std::uint32_t *root_gate_present,
    unsigned long long *root_gate_lower,
    unsigned int *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t owner = first; owner < annotation_count;
       owner += stride) {
    if (!owner_gate_present[owner]) continue;
    const std::uint32_t root = labels[owner];
    if (root >= label_count || labels[root] != root) {
      atomicOr(status, kMalformed);
      continue;
    }
    atomicExch(
        root_gate_present + root,
        static_cast<std::uint32_t>(1));
    atomicMax(root_gate_lower + root, owner_gate_lower[owner]);
  }
}

__global__ void sum_metal_area_kernel(
    const ac::RectI64 *metal, std::uint64_t metal_count,
    const std::uint32_t *labels, std::uint64_t label_count,
    unsigned long long *root_metal_upper,
    unsigned int *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < metal_count;
       index += stride) {
    const ac::RectI64 rectangle = metal[index];
    if (!valid_rectangle(rectangle, label_count)) {
      atomicOr(status, kMalformed);
      continue;
    }
    const std::uint32_t root = labels[rectangle.owner];
    if (root >= label_count || labels[root] != root) {
      atomicOr(status, kMalformed);
      continue;
    }
    unsigned long long area = 0;
    if (!checked_area(rectangle, &area) ||
        !atomic_add_checked(root_metal_upper + root, area)) {
      atomicOr(status, kOverflow);
    }
  }
}

__global__ void evaluate_roots_kernel(
    const std::uint32_t *labels, std::uint64_t label_count,
    const std::uint32_t *root_gate_present,
    const unsigned long long *root_gate_lower,
    const unsigned long long *root_metal_upper,
    DeviceCheckpointScalars *scalars)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t root = first; root < label_count;
       root += stride) {
    if (labels[root] != root) continue;
    atomicAdd(&scalars->roots, 1ull);
    const unsigned long long metal = root_metal_upper[root];
    if (metal) atomicAdd(&scalars->roots_with_metal, 1ull);
    if (!root_gate_present[root]) {
      atomicAdd(&scalars->roots_without_gate, 1ull);
      continue;
    }
    if (!metal) {
      atomicAdd(&scalars->gate_roots_without_metal, 1ull);
      continue;
    }
    const unsigned long long gate = root_gate_lower[root];
    if (gate <= 1) {
      atomicAdd(&scalars->uncertain_roots, 1ull);
      continue;
    }
    if (gate > ULLONG_MAX / 300ull) {
      atomicOr(&scalars->status, kOverflow);
      atomicAdd(&scalars->uncertain_roots, 1ull);
      continue;
    }
    if (metal <= 300ull * gate) {
      atomicAdd(&scalars->ratio_certified_roots, 1ull);
    } else {
      atomicAdd(&scalars->uncertain_roots, 1ull);
    }
  }
}

class EventPair
{
public:
  EventPair()
  {
    cuda_require(cudaEventCreate(&m_start), "create start event");
    try {
      cuda_require(cudaEventCreate(&m_stop), "create stop event");
    } catch (...) {
      cudaEventDestroy(m_start);
      m_start = nullptr;
      throw;
    }
  }

  ~EventPair()
  {
    if (m_stop) cudaEventDestroy(m_stop);
    if (m_start) cudaEventDestroy(m_start);
  }

  void start()
  {
    cuda_require(cudaEventRecord(m_start), "record start event");
  }

  float stop()
  {
    cuda_require(cudaEventRecord(m_stop), "record stop event");
    cuda_require(cudaEventSynchronize(m_stop), "synchronize stop event");
    float milliseconds = 0.0f;
    cuda_require(
        cudaEventElapsedTime(&milliseconds, m_start, m_stop),
        "read elapsed event time");
    return milliseconds;
  }

private:
  cudaEvent_t m_start = nullptr;
  cudaEvent_t m_stop = nullptr;
};

class MemoryTracker
{
public:
  MemoryTracker(
      const acc::Config &config, std::uint64_t persistent)
      : m_config(config)
  {
    m_accounting.external_live_bytes =
        config.external_live_device_bytes;
    if (!checked_add_host(
            persistent, config.external_live_device_bytes,
            &m_accounting.current_live_bytes)) {
      m_valid = false;
      m_accounting.current_live_bytes = UINT64_MAX;
    }
    m_accounting.peak_live_bytes =
        m_accounting.current_live_bytes;
    m_accounting.persistent_bytes = persistent;
    if (m_accounting.current_live_bytes >
        config.limits.max_live_device_bytes) {
      m_valid = false;
    }
  }

  bool allocate(std::uint64_t bytes)
  {
    if (!m_valid) return false;
    std::uint64_t live = 0;
    std::uint64_t temporary = 0;
    if (!checked_add_host(
            m_accounting.current_live_bytes, bytes, &live) ||
        !checked_add_host(
            m_accounting.current_temporary_bytes, bytes,
            &temporary)) {
      return false;
    }
    if (live > m_config.limits.max_live_device_bytes) {
      return false;
    }
    m_accounting.current_live_bytes = live;
    m_accounting.current_temporary_bytes = temporary;
    m_accounting.peak_live_bytes =
        std::max(m_accounting.peak_live_bytes, live);
    m_accounting.peak_temporary_bytes = std::max(
        m_accounting.peak_temporary_bytes, temporary);
    notify(acc::MemoryEvent::temporary_allocate);
    return true;
  }

  void release_temporary(std::uint64_t bytes)
  {
    m_accounting.current_live_bytes -= bytes;
    m_accounting.current_temporary_bytes -= bytes;
    notify(acc::MemoryEvent::temporary_release);
  }

  void release_persistent(std::uint64_t bytes)
  {
    m_accounting.current_live_bytes -= bytes;
    m_accounting.persistent_bytes -= bytes;
    notify(acc::MemoryEvent::persistent_release);
  }

  void promote(std::uint64_t bytes)
  {
    m_accounting.current_temporary_bytes -= bytes;
    m_accounting.persistent_bytes += bytes;
    notify(acc::MemoryEvent::commit);
  }

  const acc::MemoryAccounting &accounting() const
  {
    return m_accounting;
  }

private:
  void notify(acc::MemoryEvent event)
  {
    if (!m_config.memory_hook) return;
    try {
      m_config.memory_hook(
          event, m_accounting, m_config.memory_hook_context);
    } catch (...) {
      // Observability must not affect the certificate.
    }
  }

  const acc::Config &m_config;
  acc::MemoryAccounting m_accounting;
  bool m_valid = true;
};

template <class T>
class DeviceBuffer
{
public:
  DeviceBuffer() = default;

  ~DeviceBuffer()
  {
    reset();
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  bool allocate(std::uint64_t count, MemoryTracker *tracker)
  {
    if (!count) {
      m_tracker = tracker;
      return true;
    }
    std::uint64_t bytes = 0;
    if (!checked_multiply_host(count, sizeof(T), &bytes) ||
        !tracker->allocate(bytes)) {
      return false;
    }
    void *pointer = nullptr;
    const cudaError_t status = cudaMalloc(&pointer, bytes);
    if (status != cudaSuccess) {
      tracker->release_temporary(bytes);
      if (status == cudaErrorMemoryAllocation) return false;
      throw CudaFailure("device allocation");
    }
    m_pointer = static_cast<T *>(pointer);
    m_bytes = bytes;
    m_tracker = tracker;
    return true;
  }

  void reset()
  {
    if (m_pointer) cudaFree(m_pointer);
    if (m_tracker && m_bytes) {
      m_tracker->release_temporary(m_bytes);
    }
    m_pointer = nullptr;
    m_bytes = 0;
    m_tracker = nullptr;
  }

  T *get() const
  {
    return m_pointer;
  }

  std::uint64_t bytes() const
  {
    return m_bytes;
  }

  T *promote()
  {
    if (m_tracker && m_bytes) m_tracker->promote(m_bytes);
    T *result = m_pointer;
    m_pointer = nullptr;
    m_bytes = 0;
    m_tracker = nullptr;
    return result;
  }

private:
  T *m_pointer = nullptr;
  std::uint64_t m_bytes = 0;
  MemoryTracker *m_tracker = nullptr;
};

acc::Status map_device_status(std::uint32_t flags)
{
  if (flags & kMalformed) return acc::Status::malformed_input;
  if (flags & kOverflow) return acc::Status::arithmetic_overflow;
  if (flags & kCapacity) return acc::Status::capacity_exceeded;
  return acc::Status::success;
}

}  // namespace

namespace klayout_cuda {
namespace antenna_clean_certificate {

struct Certificate::Impl
{
  explicit Impl(const Config &configuration)
      : config(configuration)
  {
    if (configuration.device < 0 ||
        configuration.grid_cell_size <= 0 ||
        !configuration.limits.max_live_device_bytes ||
        configuration.external_live_device_bytes >
            configuration.limits.max_live_device_bytes ||
        !configuration.limits.max_annotation_owners ||
        !configuration.limits.max_labels ||
        !configuration.limits.max_poly_tiles ||
        !configuration.limits.max_active_tiles ||
        !configuration.limits.max_metal_tiles ||
        !configuration.limits.max_grid_cells ||
        configuration.limits.max_grid_cells > UINT32_MAX ||
        !configuration.limits.max_active_memberships ||
        configuration.limits.max_active_memberships > UINT32_MAX ||
        !configuration.limits.max_query_visits ||
        !configuration.limits.max_cell_members) {
      config_status = Status::invalid_configuration;
    }
  }

  ~Impl()
  {
    if (gate_present || gate_lower) cudaSetDevice(config.device);
    if (gate_present) cudaFree(gate_present);
    if (gate_lower) cudaFree(gate_lower);
  }

  Config config;
  Status config_status = Status::success;
  std::uint32_t *gate_present = nullptr;
  unsigned long long *gate_lower = nullptr;
  std::uint64_t annotation_owner_count = 0;
  std::uint64_t persistent_bytes = 0;
  std::uint64_t epoch = 0;
  bool initialized = false;
};

Certificate::Certificate(const Config &config)
    : m_impl(new Impl(config))
{
}

Certificate::~Certificate() = default;

Status Certificate::configuration_status() const noexcept
{
  return m_impl ? m_impl->config_status : Status::host_error;
}

Status Certificate::set_external_live_device_bytes(
    std::uint64_t bytes) noexcept
{
  if (!m_impl || m_impl->config_status != Status::success) {
    return m_impl ? m_impl->config_status : Status::host_error;
  }
  if (bytes > m_impl->config.limits.max_live_device_bytes ||
      m_impl->persistent_bytes >
          m_impl->config.limits.max_live_device_bytes - bytes) {
    return Status::capacity_exceeded;
  }
  m_impl->config.external_live_device_bytes = bytes;
  return Status::success;
}

Status Certificate::build_gate_census(
    const antenna_connectivity::RectI64 *device_poly,
    std::uint64_t poly_count,
    const antenna_connectivity::RectI64 *device_active,
    std::uint64_t active_count,
    std::uint64_t annotation_owner_count,
    GateCensus *census) noexcept
{
  if (!m_impl || m_impl->config_status != Status::success) {
    return m_impl ? m_impl->config_status : Status::host_error;
  }
  if (!census || !annotation_owner_count ||
      (poly_count && !device_poly) ||
      (active_count && !device_active)) {
    return Status::malformed_input;
  }
  const Limits &limits = m_impl->config.limits;
  if (annotation_owner_count >
          limits.max_annotation_owners ||
      annotation_owner_count > UINT32_MAX ||
      poly_count > limits.max_poly_tiles ||
      active_count > limits.max_active_tiles ||
      active_count > UINT32_MAX) {
    return Status::capacity_exceeded;
  }

  try {
    cuda_require(
        cudaSetDevice(m_impl->config.device),
        "antenna certificate cudaSetDevice");
    MemoryTracker tracker(
        m_impl->config, m_impl->persistent_bytes);
    DeviceBuffer<std::uint32_t> new_gate_present;
    DeviceBuffer<unsigned long long> new_gate_lower;
    DeviceBuffer<DeviceGateScalars> device_scalars;
    DeviceBuffer<DeviceBounds> device_bounds;
    if (!new_gate_present.allocate(
            annotation_owner_count, &tracker) ||
        !new_gate_lower.allocate(
            annotation_owner_count, &tracker) ||
        !device_scalars.allocate(1, &tracker) ||
        !device_bounds.allocate(1, &tracker)) {
      return Status::capacity_exceeded;
    }
    cuda_require(
        cudaMemset(
            new_gate_present.get(), 0,
            new_gate_present.bytes()),
        "clear gate presence");
    cuda_require(
        cudaMemset(
            new_gate_lower.get(), 0,
            new_gate_lower.bytes()),
        "clear gate lower bounds");
    cuda_require(
        cudaMemset(
            device_scalars.get(), 0,
            device_scalars.bytes()),
        "clear gate scalars");
    const DeviceBounds initial_bounds;
    cuda_require(
        cudaMemcpy(
            device_bounds.get(), &initial_bounds,
            sizeof(initial_bounds), cudaMemcpyHostToDevice),
        "initialize scene bounds");

    EventPair events;
    events.start();
    if (poly_count) {
      collect_bounds_kernel<<<
          launch_blocks(poly_count), kThreads>>>(
          device_poly, poly_count, annotation_owner_count,
          device_bounds.get(),
          &device_scalars.get()->status);
      cuda_require(
          cudaGetLastError(),
          "validate and bound POLY rectangles");
    }
    if (active_count) {
      collect_bounds_kernel<<<
          launch_blocks(active_count), kThreads>>>(
          device_active, active_count, annotation_owner_count,
          device_bounds.get(),
          &device_scalars.get()->status);
      cuda_require(
          cudaGetLastError(),
          "validate and bound ACTIVE rectangles");
    }
    DeviceGateScalars host_scalars;
    cuda_require(
        cudaMemcpy(
            &host_scalars, device_scalars.get(),
            sizeof(host_scalars), cudaMemcpyDeviceToHost),
        "validation scalar D2H");
    Status device_status =
        map_device_status(host_scalars.status);
    if (device_status != Status::success) return device_status;

    std::uint64_t grid_cells = 0;
    DeviceBuffer<std::uint32_t> active_counts;
    DeviceBuffer<std::uint32_t> active_offsets;
    DeviceBuffer<std::uint32_t> active_cursors;
    DeviceBuffer<std::uint32_t> active_members;
    DeviceBuffer<std::uint8_t> scan_scratch;
    if (poly_count && active_count) {
      DeviceBounds host_bounds;
      cuda_require(
          cudaMemcpy(
              &host_bounds, device_bounds.get(),
              sizeof(host_bounds), cudaMemcpyDeviceToHost),
          "scene bounds D2H");
      device_bounds.reset();

      const std::int64_t minimum_x =
          decode_ordered_signed(host_bounds.minimum_x);
      const std::int64_t minimum_y =
          decode_ordered_signed(host_bounds.minimum_y);
      const std::int64_t maximum_x =
          decode_ordered_signed(host_bounds.maximum_x);
      const std::int64_t maximum_y =
          decode_ordered_signed(host_bounds.maximum_y);
      const std::int64_t base_x = floor_div_host(
          minimum_x, m_impl->config.grid_cell_size);
      const std::int64_t base_y = floor_div_host(
          minimum_y, m_impl->config.grid_cell_size);
      const std::int64_t last_x = floor_div_host(
          maximum_x, m_impl->config.grid_cell_size);
      const std::int64_t last_y = floor_div_host(
          maximum_y, m_impl->config.grid_cell_size);
      const __int128 width =
          static_cast<__int128>(last_x) - base_x + 1;
      const __int128 height =
          static_cast<__int128>(last_y) - base_y + 1;
      const __int128 cells = width * height;
      if (width <= 0 || height <= 0 ||
          width > UINT32_MAX || height > UINT32_MAX ||
          cells <= 0 || cells > INT_MAX ||
          cells > limits.max_grid_cells) {
        return Status::capacity_exceeded;
      }
      grid_cells = static_cast<std::uint64_t>(cells);
      const Grid grid = {
          base_x, base_y, m_impl->config.grid_cell_size,
          static_cast<std::uint32_t>(width),
          static_cast<std::uint32_t>(height)};

      if (!active_counts.allocate(grid_cells, &tracker) ||
          !active_offsets.allocate(grid_cells, &tracker)) {
        return Status::capacity_exceeded;
      }
      cuda_require(
          cudaMemset(
              active_counts.get(), 0, active_counts.bytes()),
          "clear ACTIVE grid counts");
      count_grid_kernel<<<
          launch_blocks(active_count), kThreads>>>(
          device_active, active_count, annotation_owner_count,
          grid, limits.max_cell_members,
          limits.max_active_memberships,
          active_counts.get(), device_scalars.get());
      cuda_require(
          cudaGetLastError(), "count ACTIVE grid memberships");
      cuda_require(
          cudaMemcpy(
              &host_scalars, device_scalars.get(),
              sizeof(host_scalars), cudaMemcpyDeviceToHost),
          "ACTIVE grid scalar D2H");
      device_status = map_device_status(host_scalars.status);
      if (device_status != Status::success) return device_status;

      std::size_t scan_bytes = 0;
      cuda_require(
          cub::DeviceScan::ExclusiveSum(
              nullptr, scan_bytes, active_counts.get(),
              active_offsets.get(),
              static_cast<int>(grid_cells)),
          "query CUB scan scratch");
      if (!scan_scratch.allocate(scan_bytes, &tracker)) {
        return Status::capacity_exceeded;
      }
      cuda_require(
          cub::DeviceScan::ExclusiveSum(
              scan_scratch.get(), scan_bytes,
              active_counts.get(), active_offsets.get(),
              static_cast<int>(grid_cells)),
          "scan ACTIVE grid offsets");
      // CUB receives all of its temporary storage explicitly.  Release it
      // before the fill frontier; there is no hidden Thrust allocation.
      scan_scratch.reset();

      if (!active_cursors.allocate(grid_cells, &tracker) ||
          !active_members.allocate(
              host_scalars.active_memberships, &tracker)) {
        return Status::capacity_exceeded;
      }
      cuda_require(
          cudaMemcpy(
              active_cursors.get(), active_offsets.get(),
              active_offsets.bytes(), cudaMemcpyDeviceToDevice),
          "initialize ACTIVE grid cursors");
      fill_grid_kernel<<<
          launch_blocks(active_count), kThreads>>>(
          device_active, active_count, grid,
          active_cursors.get(), active_members.get(),
          &device_scalars.get()->status);
      cuda_require(cudaGetLastError(), "fill ACTIVE grid");
      cuda_require(
          cudaMemcpy(
              &host_scalars, device_scalars.get(),
              sizeof(host_scalars), cudaMemcpyDeviceToHost),
          "ACTIVE fill scalar D2H");
      device_status = map_device_status(host_scalars.status);
      if (device_status != Status::success) return device_status;
      active_cursors.reset();

      count_gate_query_visits_grid_kernel<<<
          launch_blocks(poly_count), kThreads>>>(
          device_poly, poly_count, grid, active_counts.get(),
          limits.max_query_visits, device_scalars.get());
      cuda_require(
          cudaGetLastError(),
          "count gridded POLY/ACTIVE query visits");
      gate_intersections_grid_kernel<<<
          launch_blocks(poly_count), kThreads>>>(
          device_poly, poly_count, device_active, grid,
          active_counts.get(), active_offsets.get(),
          active_members.get(), new_gate_present.get(),
          new_gate_lower.get(), device_scalars.get());
      cuda_require(
          cudaGetLastError(),
          "census gridded POLY/ACTIVE intersections");
    }
    count_gate_owners_kernel<<<
        launch_blocks(annotation_owner_count), kThreads>>>(
        new_gate_present.get(), annotation_owner_count,
        device_scalars.get());
    cuda_require(cudaGetLastError(), "count gate owners");
    const float milliseconds = events.stop();

    cuda_require(
        cudaMemcpy(
            &host_scalars, device_scalars.get(),
            sizeof(host_scalars), cudaMemcpyDeviceToHost),
        "gate scalar census D2H");
    device_status = map_device_status(host_scalars.status);
    if (device_status != Status::success) return device_status;
    // The spatial index is single-use.  Drop every grid allocation before
    // committing the small persistent owner annotations.
    active_members.reset();
    active_offsets.reset();
    active_counts.reset();
    device_bounds.reset();

    std::uint64_t replacement_bytes = 0;
    if (!checked_add_host(
            new_gate_present.bytes(), new_gate_lower.bytes(),
            &replacement_bytes)) {
      return Status::capacity_exceeded;
    }
    if (m_impl->gate_present) cudaFree(m_impl->gate_present);
    if (m_impl->gate_lower) cudaFree(m_impl->gate_lower);
    tracker.release_persistent(m_impl->persistent_bytes);
    m_impl->gate_present = new_gate_present.promote();
    m_impl->gate_lower = new_gate_lower.promote();
    m_impl->annotation_owner_count = annotation_owner_count;
    m_impl->persistent_bytes = replacement_bytes;
    m_impl->initialized = true;
    ++m_impl->epoch;
    device_scalars.reset();

    GateCensus result;
    result.annotation_owners = annotation_owner_count;
    result.poly_tiles = poly_count;
    result.active_tiles = active_count;
    result.grid_cells = grid_cells;
    result.active_memberships =
        host_scalars.active_memberships;
    result.candidate_visits =
        host_scalars.candidate_visits;
    result.positive_intersections =
        host_scalars.positive_intersections;
    result.gate_owners = host_scalars.gate_owners;
    result.persistent_bytes = replacement_bytes;
    result.peak_temporary_bytes =
        tracker.accounting().peak_temporary_bytes;
    result.peak_live_bytes =
        tracker.accounting().peak_live_bytes;
    result.kernel_milliseconds = milliseconds;
    *census = result;
    return Status::success;
  } catch (const CudaFailure &) {
    return Status::cuda_error;
  } catch (const std::bad_alloc &) {
    return Status::host_error;
  } catch (...) {
    return Status::host_error;
  }
}

Status Certificate::evaluate_checkpoint(
    MetalLevel level,
    const antenna_connectivity::RectI64 *device_metal,
    std::uint64_t metal_count,
    const std::uint32_t *device_labels,
    std::uint64_t label_count,
    CheckpointCensus *census) const noexcept
{
  if (!m_impl || m_impl->config_status != Status::success) {
    return m_impl ? m_impl->config_status : Status::host_error;
  }
  if (!m_impl->initialized) return Status::not_initialized;
  if (!census || !valid_level(level) || !device_labels ||
      !label_count || (metal_count && !device_metal) ||
      label_count < m_impl->annotation_owner_count) {
    return Status::malformed_input;
  }
  const Limits &limits = m_impl->config.limits;
  if (label_count > limits.max_labels ||
      label_count > UINT32_MAX ||
      metal_count > limits.max_metal_tiles) {
    return Status::capacity_exceeded;
  }

  try {
    cuda_require(
        cudaSetDevice(m_impl->config.device),
        "antenna certificate cudaSetDevice");
    MemoryTracker tracker(
        m_impl->config, m_impl->persistent_bytes);
    DeviceBuffer<std::uint32_t> root_gate_present;
    DeviceBuffer<unsigned long long> root_gate_lower;
    DeviceBuffer<unsigned long long> root_metal_upper;
    DeviceBuffer<DeviceCheckpointScalars> device_scalars;
    if (!root_gate_present.allocate(label_count, &tracker) ||
        !root_gate_lower.allocate(label_count, &tracker) ||
        !root_metal_upper.allocate(label_count, &tracker) ||
        !device_scalars.allocate(1, &tracker)) {
      return Status::capacity_exceeded;
    }
    cuda_require(
        cudaMemset(
            root_gate_present.get(), 0,
            root_gate_present.bytes()),
        "clear root gate presence");
    cuda_require(
        cudaMemset(
            root_gate_lower.get(), 0,
            root_gate_lower.bytes()),
        "clear root gate lower bounds");
    cuda_require(
        cudaMemset(
            root_metal_upper.get(), 0,
            root_metal_upper.bytes()),
        "clear root metal upper bounds");
    cuda_require(
        cudaMemset(
            device_scalars.get(), 0,
            device_scalars.bytes()),
        "clear checkpoint scalars");

    EventPair events;
    events.start();
    validate_labels_kernel<<<
        launch_blocks(label_count), kThreads>>>(
        device_labels, label_count,
        &device_scalars.get()->status);
    cuda_require(cudaGetLastError(), "validate canonical labels");
    reduce_gate_annotations_kernel<<<
        launch_blocks(m_impl->annotation_owner_count),
        kThreads>>>(
        m_impl->gate_present, m_impl->gate_lower,
        m_impl->annotation_owner_count, device_labels,
        label_count, root_gate_present.get(),
        root_gate_lower.get(), &device_scalars.get()->status);
    cuda_require(cudaGetLastError(), "reduce gate annotations");
    if (metal_count) {
      sum_metal_area_kernel<<<
          launch_blocks(metal_count), kThreads>>>(
          device_metal, metal_count, device_labels,
          label_count, root_metal_upper.get(),
          &device_scalars.get()->status);
      cuda_require(cudaGetLastError(), "sum target-metal area");
    }
    evaluate_roots_kernel<<<
        launch_blocks(label_count), kThreads>>>(
        device_labels, label_count, root_gate_present.get(),
        root_gate_lower.get(), root_metal_upper.get(),
        device_scalars.get());
    cuda_require(cudaGetLastError(), "evaluate antenna roots");
    const float milliseconds = events.stop();

    DeviceCheckpointScalars host_scalars;
    cuda_require(
        cudaMemcpy(
            &host_scalars, device_scalars.get(),
            sizeof(host_scalars), cudaMemcpyDeviceToHost),
        "checkpoint scalar census D2H");
    const Status device_status =
        map_device_status(host_scalars.status);
    if (device_status != Status::success) return device_status;

    CheckpointCensus result;
    result.level = level;
    result.clean_certificate =
        host_scalars.uncertain_roots == 0;
    result.labels = label_count;
    result.metal_tiles = metal_count;
    result.roots = host_scalars.roots;
    result.roots_with_metal =
        host_scalars.roots_with_metal;
    result.roots_without_gate =
        host_scalars.roots_without_gate;
    result.gate_roots_without_metal =
        host_scalars.gate_roots_without_metal;
    result.ratio_certified_roots =
        host_scalars.ratio_certified_roots;
    result.uncertain_roots =
        host_scalars.uncertain_roots;
    result.persistent_bytes = m_impl->persistent_bytes;
    result.peak_temporary_bytes =
        tracker.accounting().peak_temporary_bytes;
    result.peak_live_bytes =
        tracker.accounting().peak_live_bytes;
    result.kernel_milliseconds = milliseconds;
    *census = result;
    return Status::success;
  } catch (const CudaFailure &) {
    return Status::cuda_error;
  } catch (const std::bad_alloc &) {
    return Status::host_error;
  } catch (...) {
    return Status::host_error;
  }
}

Status Certificate::device_gate_view(
    GateAnnotationDeviceView *view) const noexcept
{
  if (!m_impl || !view) return Status::malformed_input;
  if (m_impl->config_status != Status::success) {
    return m_impl->config_status;
  }
  if (!m_impl->initialized) return Status::not_initialized;
  GateAnnotationDeviceView result;
  result.gate_present = m_impl->gate_present;
  result.max_single_intersection_area =
      reinterpret_cast<const std::uint64_t *>(
          m_impl->gate_lower);
  result.count = m_impl->annotation_owner_count;
  result.epoch = m_impl->epoch;
  result.device = m_impl->config.device;
  *view = result;
  return Status::success;
}

const char *status_string(Status status) noexcept
{
  switch (status) {
  case Status::success:
    return "success";
  case Status::invalid_configuration:
    return "invalid_configuration";
  case Status::not_initialized:
    return "not_initialized";
  case Status::malformed_input:
    return "malformed_input";
  case Status::capacity_exceeded:
    return "capacity_exceeded";
  case Status::arithmetic_overflow:
    return "arithmetic_overflow";
  case Status::cuda_error:
    return "cuda_error";
  case Status::host_error:
    return "host_error";
  }
  return "unknown";
}

}  // namespace antenna_clean_certificate
}  // namespace klayout_cuda
