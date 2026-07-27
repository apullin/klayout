#include "active4_subset_resident.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace klayout_cuda {
namespace active4_subset_resident {
namespace {

namespace mu = manhattan_union;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaximumBlocks = 65535;
static_assert(
    sizeof(unsigned long long) == sizeof(std::uint64_t),
    "CUDA atomic counters require a 64-bit unsigned long long");

struct Counters
{
  unsigned long long rectangles_visited;
  unsigned long long rectangles_completed;
  unsigned long long slab_visits;
  unsigned long long interval_search_steps;
  unsigned long long witnesses;
  unsigned long long uncertain;
};

__device__ void checked_increment(
    unsigned long long *value, std::uint32_t *status)
{
  if (*value == ULLONG_MAX) {
    atomicOr(status, std::uint32_t(kVisitOverflow));
  } else {
    ++*value;
  }
}

__device__ void checked_add(
    unsigned long long *value, unsigned long long increment,
    std::uint32_t *status)
{
  if (*value > ULLONG_MAX - increment) {
    *value = ULLONG_MAX;
    atomicOr(status, std::uint32_t(kVisitOverflow));
  } else {
    *value += increment;
  }
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 1;
  const std::uint64_t needed = (count - 1) / kThreads + 1;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(needed, kMaximumBlocks));
}

void sample_memory(Result *result)
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_require(
      cudaMemGetInfo(&free_bytes, &total_bytes),
      "ACTIVE.4 subset cudaMemGetInfo");
  if (!result->device_total_bytes) {
    result->device_total_bytes = total_bytes;
    result->device_free_begin_bytes = free_bytes;
    result->device_free_low_bytes = free_bytes;
  } else if (result->device_total_bytes != total_bytes) {
    throw std::runtime_error(
        "ACTIVE.4 subset device-memory identity changed");
  } else {
    result->device_free_low_bytes =
        std::min<std::uint64_t>(result->device_free_low_bytes, free_bytes);
  }
}

void require_device_pointer(const void *pointer, int device, const char *name)
{
  if (!pointer) {
    throw std::runtime_error(std::string(name) + " is null");
  }
  cudaPointerAttributes attributes{};
  cuda_require(
      cudaPointerGetAttributes(&attributes, pointer),
      (std::string(name) + " pointer attributes").c_str());
#if CUDART_VERSION >= 10000
  const cudaMemoryType type = attributes.type;
#else
  const cudaMemoryType type = attributes.memoryType;
#endif
  if (type != cudaMemoryTypeDevice || attributes.device != device) {
    throw std::runtime_error(
        std::string(name) + " is not resident on the selected device");
  }
}

__global__ void validate_strips_kernel(
    DeviceStripView wells, std::uint64_t interval_count,
    std::uint32_t *status)
{
  for (std::uint64_t slab =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       slab < wells.x_slabs;
       slab += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (wells.xs[slab] >= wells.xs[slab + 1]) {
      atomicOr(status, std::uint32_t(kInvalidStripView));
    }
    const std::uint64_t offset = wells.slab_offsets[slab];
    const std::uint64_t count = wells.slab_counts[slab];
    if (offset > interval_count || count > interval_count - offset) {
      atomicOr(status, std::uint32_t(kInvalidStripView));
      continue;
    }
    if (!slab) {
      if (offset != 0) {
        atomicOr(status, std::uint32_t(kInvalidStripView));
      }
    } else {
      const std::uint64_t previous_offset =
          wells.slab_offsets[slab - 1];
      const std::uint64_t previous_count =
          wells.slab_counts[slab - 1];
      if (previous_offset > interval_count ||
          previous_count > interval_count - previous_offset ||
          offset != previous_offset + previous_count) {
        atomicOr(status, std::uint32_t(kInvalidStripView));
      }
    }
    for (std::uint64_t local = 0; local < count; ++local) {
      const mu::StripInterval interval = wells.intervals[offset + local];
      if (interval.slab != slab || interval.reserved ||
          interval.bottom >= interval.top) {
        atomicOr(status, std::uint32_t(kInvalidStripView));
      }
      if (local) {
        const mu::StripInterval previous =
            wells.intervals[offset + local - 1];
        if (previous.top >= interval.bottom) {
          atomicOr(status, std::uint32_t(kInvalidStripView));
        }
      }
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint64_t last = wells.x_slabs - 1;
    const std::uint64_t offset = wells.slab_offsets[last];
    const std::uint64_t count = wells.slab_counts[last];
    if (offset > interval_count || count > interval_count - offset ||
        offset + count != interval_count) {
      atomicOr(status, std::uint32_t(kInvalidStripView));
    }
  }
}

__device__ std::uint32_t upper_bound_x(
    const std::int64_t *xs, std::uint32_t count, std::int64_t value,
    unsigned long long *steps, std::uint32_t *status)
{
  std::uint32_t first = 0;
  std::uint32_t length = count;
  while (length) {
    checked_increment(steps, status);
    const std::uint32_t half = length / 2;
    const std::uint32_t middle = first + half;
    if (xs[middle] <= value) {
      first = middle + 1;
      length -= half + 1;
    } else {
      length = half;
    }
  }
  return first;
}

__device__ std::uint32_t lower_bound_x(
    const std::int64_t *xs, std::uint32_t count, std::int64_t value,
    unsigned long long *steps, std::uint32_t *status)
{
  std::uint32_t first = 0;
  std::uint32_t length = count;
  while (length) {
    checked_increment(steps, status);
    const std::uint32_t half = length / 2;
    const std::uint32_t middle = first + half;
    if (xs[middle] < value) {
      first = middle + 1;
      length -= half + 1;
    } else {
      length = half;
    }
  }
  return first;
}

__device__ bool interval_covers(
    const mu::StripInterval *intervals, std::uint64_t count,
    std::int64_t bottom, std::int64_t top,
    unsigned long long *steps, std::uint32_t *status)
{
  std::uint64_t first = 0;
  std::uint64_t length = count;
  while (length) {
    checked_increment(steps, status);
    const std::uint64_t half = length / 2;
    const std::uint64_t middle = first + half;
    if (intervals[middle].bottom <= bottom) {
      first = middle + 1;
      length -= half + 1;
    } else {
      length = half;
    }
  }
  if (!first) return false;
  const mu::StripInterval candidate = intervals[first - 1];
  return candidate.bottom <= bottom && candidate.top >= top;
}

__global__ void subset_query_kernel(
    DeviceStripView wells, DeviceRectangleView active,
    Limits limits, Counters *counters, std::uint32_t *status)
{
  unsigned long long local_slab_visits = 0;
  unsigned long long local_search_steps = 0;
  unsigned long long local_witnesses = 0;
  unsigned long long local_uncertain = 0;
  unsigned long long local_rectangles_visited = 0;
  unsigned long long local_rectangles_completed = 0;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < active.count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    checked_increment(&local_rectangles_visited, status);
    const mu::RectI64 rectangle = active.rectangles[index];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top) {
      atomicOr(status, std::uint32_t(kInvalidRectangle));
      checked_increment(&local_uncertain, status);
      continue;
    }
    if (rectangle.left < wells.xs[0] ||
        rectangle.right > wells.xs[wells.x_slabs]) {
      checked_increment(&local_witnesses, status);
      checked_increment(&local_rectangles_completed, status);
      continue;
    }
    const std::uint32_t x_count = wells.x_slabs + 1;
    const std::uint32_t upper = upper_bound_x(
        wells.xs, x_count, rectangle.left, &local_search_steps, status);
    const std::uint32_t end = lower_bound_x(
        wells.xs, x_count, rectangle.right, &local_search_steps, status);
    if (!upper || upper > wells.x_slabs || !end ||
        end > wells.x_slabs || upper - 1 >= end) {
      atomicOr(status, std::uint32_t(kInvalidRectangle));
      checked_increment(&local_uncertain, status);
      continue;
    }
    const std::uint32_t begin = upper - 1;
    const std::uint64_t span =
        static_cast<std::uint64_t>(end) - begin;
    if (span > limits.max_slabs_per_rectangle) {
      atomicOr(
          status,
          std::uint32_t(kVisitCapacity) |
              std::uint32_t(kTruncatedWork));
      checked_increment(&local_uncertain, status);
      continue;
    }
    checked_add(&local_slab_visits, span, status);
    bool covered = true;
    for (std::uint32_t slab = begin; slab < end; ++slab) {
      const std::uint64_t offset = wells.slab_offsets[slab];
      const std::uint64_t count = wells.slab_counts[slab];
      if (!interval_covers(
              wells.intervals + offset, count, rectangle.bottom,
              rectangle.top, &local_search_steps, status)) {
        covered = false;
      }
    }
    if (!covered) checked_increment(&local_witnesses, status);
    checked_increment(&local_rectangles_completed, status);
  }

  if (local_rectangles_visited) {
    const unsigned long long previous =
        atomicAdd(
            &counters->rectangles_visited,
            local_rectangles_visited);
    if (previous > ULLONG_MAX - local_rectangles_visited) {
      atomicOr(status, std::uint32_t(kVisitOverflow));
    }
  }
  if (local_rectangles_completed) {
    const unsigned long long previous =
        atomicAdd(
            &counters->rectangles_completed,
            local_rectangles_completed);
    if (previous > ULLONG_MAX - local_rectangles_completed) {
      atomicOr(status, std::uint32_t(kVisitOverflow));
    }
  }
  if (local_slab_visits) {
    const unsigned long long previous =
        atomicAdd(&counters->slab_visits, local_slab_visits);
    if (previous > ULLONG_MAX - local_slab_visits) {
      atomicOr(status, std::uint32_t(kVisitOverflow));
    } else if (previous + local_slab_visits > limits.max_slab_visits) {
      atomicOr(status, std::uint32_t(kVisitCapacity));
    }
  }
  if (local_search_steps) {
    const unsigned long long previous =
        atomicAdd(&counters->interval_search_steps, local_search_steps);
    if (previous > ULLONG_MAX - local_search_steps) {
      atomicOr(status, std::uint32_t(kVisitOverflow));
    } else if (
        previous + local_search_steps > limits.max_search_steps) {
      atomicOr(status, std::uint32_t(kSearchCapacity));
    }
  }
  if (local_witnesses) {
    atomicAdd(&counters->witnesses, local_witnesses);
  }
  if (local_uncertain) {
    atomicAdd(&counters->uncertain, local_uncertain);
  }
}

}  // namespace

Result certify_subset(
    cudaStream_t stream, DeviceStripView wells,
    DeviceRectangleView active, const Limits &limits, int device)
{
  const Clock::time_point total_begin = Clock::now();
  if (stream != nullptr) {
    throw std::runtime_error(
        "ACTIVE.4 subset requires the CUDA default stream");
  }
  if (device < 0 || !wells.xs || !wells.x_slabs ||
      !wells.intervals || !wells.interval_count ||
      !wells.slab_offsets || !wells.slab_counts ||
      (active.count && !active.rectangles) ||
      wells.x_slabs == std::numeric_limits<std::uint32_t>::max() ||
      wells.x_slabs > limits.max_x_slabs ||
      wells.interval_count > limits.max_intervals ||
      active.count > limits.max_rectangles ||
      !limits.max_slab_visits ||
      !limits.max_search_steps ||
      !limits.max_slabs_per_rectangle) {
    throw std::runtime_error(
        "invalid ACTIVE.4 resident subset request or capacity");
  }

  cuda_require(cudaSetDevice(device), "ACTIVE.4 subset cudaSetDevice");
  require_device_pointer(wells.xs, device, "ACTIVE.4 WELL x endpoints");
  require_device_pointer(
      wells.intervals, device, "ACTIVE.4 WELL intervals");
  require_device_pointer(
      wells.slab_offsets, device, "ACTIVE.4 WELL slab offsets");
  require_device_pointer(
      wells.slab_counts, device, "ACTIVE.4 WELL slab counts");
  if (active.count) {
    require_device_pointer(
        active.rectangles, device, "ACTIVE.4 ACTIVE rectangles");
  }

  Result result;
  result.rectangles = active.count;
  sample_memory(&result);
  thrust::device_vector<std::uint32_t> status(1, 0);
  thrust::device_vector<Counters> counters(1);
  cuda_require(
      cudaMemsetAsync(
          thrust::raw_pointer_cast(counters.data()), 0,
          sizeof(Counters), stream),
      "ACTIVE.4 subset counters clear");

  const Clock::time_point validation_begin = Clock::now();
  validate_strips_kernel<<<
      launch_blocks(wells.x_slabs), kThreads, 0, stream>>>(
      wells, wells.interval_count,
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(), "ACTIVE.4 strip validation launch");
  cuda_require(
      cudaStreamSynchronize(stream),
      "ACTIVE.4 strip validation synchronize");
  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "ACTIVE.4 strip status D2H");
  result.validation_ms = elapsed_ms(validation_begin, Clock::now());
  if (host_status) {
    throw std::runtime_error(
        "ACTIVE.4 canonical WELL strip invariant declined");
  }

  const Clock::time_point query_begin = Clock::now();
  if (active.count) {
    subset_query_kernel<<<
        launch_blocks(active.count), kThreads, 0, stream>>>(
        wells, active, limits, thrust::raw_pointer_cast(counters.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(), "ACTIVE.4 subset query launch");
  }
  cuda_require(
      cudaStreamSynchronize(stream),
      "ACTIVE.4 subset query synchronize");
  result.query_ms = elapsed_ms(query_begin, Clock::now());
  sample_memory(&result);

  const Clock::time_point d2h_begin = Clock::now();
  Counters host_counters{};
  cuda_require(
      cudaMemcpy(
          &host_counters, thrust::raw_pointer_cast(counters.data()),
          sizeof(host_counters), cudaMemcpyDeviceToHost),
      "ACTIVE.4 subset counters D2H");
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "ACTIVE.4 subset final status D2H");
  result.d2h_ms = elapsed_ms(d2h_begin, Clock::now());
  result.rectangles_visited = host_counters.rectangles_visited;
  result.rectangles_completed = host_counters.rectangles_completed;
  result.slab_visits = host_counters.slab_visits;
  result.interval_search_steps = host_counters.interval_search_steps;
  result.witnesses = host_counters.witnesses;
  result.uncertain = host_counters.uncertain;
  result.device_flags = host_status;
  result.all_work_completed =
      result.rectangles_visited == result.rectangles &&
      result.rectangles_completed == result.rectangles;
  result.certified_subset =
      !result.witnesses && !result.uncertain && !result.device_flags &&
      result.all_work_completed &&
      result.slab_visits <= limits.max_slab_visits;
  result.total_ms = elapsed_ms(total_begin, Clock::now());
  return result;
}

}  // namespace active4_subset_resident
}  // namespace klayout_cuda
