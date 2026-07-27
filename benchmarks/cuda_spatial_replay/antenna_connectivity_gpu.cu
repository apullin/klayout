/*
 * Exact staged rectangle connectivity for the FreePDK45 antenna CUDA plan.
 */

#include "antenna_connectivity_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <thrust/unique.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ac = klayout_cuda::antenna_connectivity;

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaximumBlocks = 65535;

enum DeviceFlag : std::uint32_t
{
  kMalformedRectangle = 1u << 0,
  kInvalidDomain = 1u << 1,
  kSpanOverflow = 1u << 2,
  kCellCapacity = 1u << 3,
  kPairCountOverflow = 1u << 4,
  kInvalidParent = 1u << 5,
  kOwnerDomainMismatch = 1u << 6,
  kMissingOwner = 1u << 7,
  kOverlappingOwnerTiles = 1u << 8,
  kDisconnectedOwnerTiles = 1u << 9,
  kClosedDomainInput = 1u << 10,
  kUnexpectedStreamKind = 1u << 11
};

class CudaFailure : public std::runtime_error
{
public:
  explicit CudaFailure(const std::string &message)
      : std::runtime_error(message)
  {
  }
};

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw CudaFailure(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

using ConnectivityClock = std::chrono::steady_clock;

bool connectivity_timing_enabled()
{
  const char *value =
      std::getenv("KLAYOUT_CUDA_CONNECTIVITY_TIMING");
  return value && *value &&
         !(value[0] == '0' && value[1] == '\0');
}

double elapsed_milliseconds(
    ConnectivityClock::time_point begin,
    ConnectivityClock::time_point end)
{
  return std::chrono::duration<double, std::milli>(
             end - begin)
      .count();
}

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 0;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          (count + kThreads - 1) / kThreads, kMaximumBlocks));
}

std::uint32_t launch_group_blocks(std::uint64_t group_count)
{
  if (!group_count) return 0;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(group_count, kMaximumBlocks));
}

template <class T>
void release_device_vector(thrust::device_vector<T> *values)
{
  thrust::device_vector<T>().swap(*values);
}

template <class T>
void compact_device_vector(thrust::device_vector<T> *values)
{
  thrust::device_vector<T> compact(values->begin(), values->end());
  values->swap(compact);
}

bool byte_product(
    std::uint64_t count, std::uint64_t size,
    std::uint64_t *bytes)
{
  if (size && count > UINT64_MAX / size) return false;
  *bytes = count * size;
  return true;
}

struct SaturatingAddU64
{
  __host__ __device__ std::uint64_t operator()(
      std::uint64_t first, std::uint64_t second) const
  {
    return first > UINT64_MAX - second
               ? UINT64_MAX
               : first + second;
  }
};

bool device_bytes_admitted(
    const ac::Config &config, std::uint64_t requested_bytes)
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_require(
      cudaMemGetInfo(&free_bytes, &total_bytes),
      "antenna connectivity cudaMemGetInfo");
  const std::uint64_t total = total_bytes;
  const std::uint64_t free = free_bytes;
  const std::uint64_t used = total - free;
  const std::uint64_t cap =
      std::min<std::uint64_t>(
          total, config.limits.max_device_bytes);
  const std::uint64_t under_cap =
      cap > used ? cap - used : 0;
  const std::uint64_t available =
      std::min(free, under_cap);
  const std::uint64_t reserve =
      config.limits.min_device_free_after_bytes;
  return available >= reserve &&
         requested_bytes <= available - reserve;
}

template <class T>
bool vector_allocation_admitted(
    const ac::Config &config, std::uint64_t count)
{
  std::uint64_t bytes = 0;
  return byte_product(count, sizeof(T), &bytes) &&
         device_bytes_admitted(config, bytes);
}

template <class T>
bool vector_growth_admitted(
    const ac::Config &config,
    const thrust::device_vector<T> &values,
    std::uint64_t target_count)
{
  return target_count <= values.capacity() ||
         vector_allocation_admitted<T>(config, target_count);
}

template <class T>
bool sort_scratch_admitted(
    const ac::Config &config, std::uint64_t count)
{
  std::uint64_t copy_bytes = 0;
  if (!byte_product(count, sizeof(T), &copy_bytes) ||
      copy_bytes >
          UINT64_MAX -
              config.limits.sort_scratch_fixed_guard_bytes) {
    return false;
  }
  return device_bytes_admitted(
      config,
      copy_bytes +
          config.limits.sort_scratch_fixed_guard_bytes);
}

ac::Status validate_config(const ac::Config &config)
{
  if (!config.domain_count ||
      config.domain_count > ac::kMaximumDomains ||
      config.bin_size <= 0 || config.device < 0 ||
      !config.limits.max_nodes ||
      config.limits.max_nodes > UINT32_MAX ||
      !config.limits.max_rectangles ||
      config.limits.max_rectangles > UINT32_MAX ||
      !config.limits.max_memberships ||
      !config.limits.max_pair_occurrences ||
      !config.limits.max_unique_candidates ||
      !config.limits.max_cell_members ||
      !config.limits.max_pair_tests_per_cell ||
      (config.exact_filter_before_materialization &&
       !config.limits.max_total_pair_tests) ||
      !config.limits.max_dsu_iterations ||
      !config.limits.max_device_bytes ||
      config.limits.min_device_free_after_bytes >=
          config.limits.max_device_bytes) {
    return ac::Status::invalid_configuration;
  }

  const std::uint64_t domain_mask =
      config.domain_count == ac::kMaximumDomains
          ? UINT64_MAX
          : (UINT64_C(1) << config.domain_count) - 1;
  for (std::uint32_t first = 0; first < ac::kMaximumDomains; ++first) {
    if (config.relation_rows[first] & ~domain_mask) {
      return ac::Status::invalid_configuration;
    }
    if (first >= config.domain_count && config.relation_rows[first]) {
      return ac::Status::invalid_configuration;
    }
  }
  for (std::uint32_t first = 0; first < config.domain_count; ++first) {
    for (std::uint32_t second = first + 1;
         second < config.domain_count; ++second) {
      const bool forward =
          (config.relation_rows[first] &
           (UINT64_C(1) << second)) != 0;
      const bool reverse =
          (config.relation_rows[second] &
           (UINT64_C(1) << first)) != 0;
      if (forward != reverse) {
        return ac::Status::invalid_configuration;
      }
    }
  }
  return ac::Status::success;
}

struct CellMember
{
  std::uint64_t cell;
  std::uint32_t node;
  std::uint32_t reserved;
};

static_assert(sizeof(ac::RectI64) == 40, "RectI64 size changed");
static_assert(sizeof(CellMember) == 16, "CellMember size changed");

struct CellMemberLess
{
  __host__ __device__ bool operator()(
      const CellMember &first, const CellMember &second) const
  {
    if (first.cell != second.cell) return first.cell < second.cell;
    return first.node < second.node;
  }
};

struct SameCell
{
  __host__ __device__ bool operator()(
      const CellMember &first, const CellMember &second) const
  {
    return first.cell == second.cell;
  }
};

struct CellBounds
{
  unsigned long long minimum_x;
  unsigned long long minimum_y;
  unsigned long long maximum_x;
  unsigned long long maximum_y;
};

static_assert(sizeof(CellBounds) == 32, "CellBounds size changed");

struct IsSet
{
  __host__ __device__ bool operator()(std::uint8_t value) const
  {
    return value != 0;
  }
};

struct KeepUnreleasedDomain
{
  std::uint64_t released_domains;

  __host__ __device__ bool operator()(
      const ac::RectI64 &rectangle) const
  {
    return (released_domains &
            (UINT64_C(1) << rectangle.domain)) == 0;
  }
};

struct IsReleasedDomain
{
  std::uint64_t released_domains;

  __host__ __device__ bool operator()(
      const ac::RectI64 &rectangle) const
  {
    return (released_domains &
            (UINT64_C(1) << rectangle.domain)) != 0;
  }
};

__device__ std::int64_t floor_div_device(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

__host__ __device__ std::uint64_t ordered_i64(
    std::int64_t value)
{
  return static_cast<std::uint64_t>(value) ^
         UINT64_C(0x8000000000000000);
}

__device__ bool inclusive_span_size(
    std::int64_t low, std::int64_t high, std::uint64_t *size)
{
  if (high < low) return false;
  const std::uint64_t difference =
      static_cast<std::uint64_t>(high) -
      static_cast<std::uint64_t>(low);
  if (difference == UINT64_MAX) return false;
  *size = difference + 1;
  return true;
}

__device__ bool allowed_relation(
    std::uint32_t first, std::uint32_t second,
    const std::uint64_t *rows)
{
  return (rows[first] & (UINT64_C(1) << second)) != 0;
}

__device__ std::uint64_t pair_key(
    std::uint32_t first, std::uint32_t second)
{
  const std::uint32_t low = min(first, second);
  const std::uint32_t high = max(first, second);
  return (static_cast<std::uint64_t>(low) << 32) | high;
}

__device__ void decode_pair(
    std::uint64_t key, std::uint32_t *first,
    std::uint32_t *second)
{
  *first = static_cast<std::uint32_t>(key >> 32);
  *second = static_cast<std::uint32_t>(key);
}

__device__ bool closed_touch_or_overlap(
    const ac::RectI64 &first, const ac::RectI64 &second)
{
  return first.left <= second.right &&
         second.left <= first.right &&
         first.bottom <= second.top &&
         second.bottom <= first.top;
}

__device__ bool positive_area_overlap(
    const ac::RectI64 &first, const ac::RectI64 &second)
{
  return max(first.left, second.left) <
             min(first.right, second.right) &&
         max(first.bottom, second.bottom) <
             min(first.top, second.top);
}

__global__ void count_memberships_kernel(
    const ac::RectI64 *rectangles, std::uint64_t count,
    std::uint64_t previous_rectangle_count,
    std::uint32_t stage_begin, std::uint32_t owner_count,
    std::uint32_t domain_count, std::int64_t bin_size,
    std::uint64_t closed_domains,
    std::uint32_t *owner_domains,
    std::uint64_t *membership_counts,
    unsigned long long *cell_bounds, std::uint32_t *status)
{
  CellBounds local = {ULLONG_MAX, ULLONG_MAX, 0, 0};
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       id < count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const ac::RectI64 rectangle = rectangles[id];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top) {
      atomicOr(status, std::uint32_t(kMalformedRectangle));
      membership_counts[id] = 0;
      continue;
    }
    const bool new_rectangle = id >= previous_rectangle_count;
    const bool new_owner = rectangle.owner >= stage_begin;
    if (rectangle.owner >= owner_count ||
        new_rectangle != new_owner) {
      atomicOr(status, std::uint32_t(kMalformedRectangle));
      membership_counts[id] = 0;
      continue;
    }
    if (rectangle.domain >= domain_count) {
      atomicOr(status, std::uint32_t(kInvalidDomain));
      membership_counts[id] = 0;
      continue;
    }
    if (new_rectangle &&
        (closed_domains &
         (UINT64_C(1) << rectangle.domain))) {
      atomicOr(status, std::uint32_t(kClosedDomainInput));
      membership_counts[id] = 0;
      continue;
    }
    const std::uint32_t prior_domain = atomicCAS(
        owner_domains + rectangle.owner, UINT32_MAX,
        rectangle.domain);
    if (prior_domain != UINT32_MAX &&
        prior_domain != rectangle.domain) {
      atomicOr(status, std::uint32_t(kOwnerDomainMismatch));
      membership_counts[id] = 0;
      continue;
    }

    const std::int64_t x0 =
        floor_div_device(rectangle.left, bin_size);
    const std::int64_t x1 =
        floor_div_device(rectangle.right, bin_size);
    const std::int64_t y0 =
        floor_div_device(rectangle.bottom, bin_size);
    const std::int64_t y1 =
        floor_div_device(rectangle.top, bin_size);
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    if (!inclusive_span_size(x0, x1, &width) ||
        !inclusive_span_size(y0, y1, &height) ||
        (height && width > UINT64_MAX / height)) {
      atomicOr(status, std::uint32_t(kSpanOverflow));
      membership_counts[id] = 0;
      continue;
    }
    membership_counts[id] = width * height;
    const unsigned long long ordered_x0 = ordered_i64(x0);
    const unsigned long long ordered_y0 = ordered_i64(y0);
    const unsigned long long ordered_x1 = ordered_i64(x1);
    const unsigned long long ordered_y1 = ordered_i64(y1);
    local.minimum_x = min(local.minimum_x, ordered_x0);
    local.minimum_y = min(local.minimum_y, ordered_y0);
    local.maximum_x = max(local.maximum_x, ordered_x1);
    local.maximum_y = max(local.maximum_y, ordered_y1);
  }

  __shared__ CellBounds block_bounds[kThreads];
  block_bounds[threadIdx.x] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride;
       stride /= 2) {
    if (threadIdx.x < stride) {
      const CellBounds other =
          block_bounds[threadIdx.x + stride];
      CellBounds &current = block_bounds[threadIdx.x];
      if (other.minimum_x <= other.maximum_x) {
        if (current.minimum_x <= current.maximum_x) {
          current.minimum_x =
              min(current.minimum_x, other.minimum_x);
          current.minimum_y =
              min(current.minimum_y, other.minimum_y);
          current.maximum_x =
              max(current.maximum_x, other.maximum_x);
          current.maximum_y =
              max(current.maximum_y, other.maximum_y);
        } else {
          current = other;
        }
      }
    }
    __syncthreads();
  }
  if (!threadIdx.x &&
      block_bounds[0].minimum_x <= block_bounds[0].maximum_x) {
    atomicMin(cell_bounds + 0, block_bounds[0].minimum_x);
    atomicMin(cell_bounds + 1, block_bounds[0].minimum_y);
    atomicMax(cell_bounds + 2, block_bounds[0].maximum_x);
    atomicMax(cell_bounds + 3, block_bounds[0].maximum_y);
  }
}

__global__ void validate_new_owners_kernel(
    const std::uint32_t *owner_domains,
    std::uint32_t stage_begin, std::uint32_t owner_count,
    std::uint32_t *status)
{
  for (std::uint64_t owner =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x + stage_begin;
       owner < owner_count;
       owner += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (owner_domains[owner] == UINT32_MAX) {
      atomicOr(status, std::uint32_t(kMissingOwner));
    }
  }
}

__global__ void fill_memberships_kernel(
    const ac::RectI64 *rectangles, std::uint64_t count,
    std::int64_t bin_size, const std::uint64_t *offsets,
    std::uint64_t minimum_x, std::uint64_t minimum_y,
    std::uint64_t grid_height, CellMember *members)
{
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       id < count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const ac::RectI64 rectangle = rectangles[id];
    const std::int64_t x0 =
        floor_div_device(rectangle.left, bin_size);
    const std::int64_t x1 =
        floor_div_device(rectangle.right, bin_size);
    const std::int64_t y0 =
        floor_div_device(rectangle.bottom, bin_size);
    const std::int64_t y1 =
        floor_div_device(rectangle.top, bin_size);
    std::uint64_t cursor = offsets[id];
    for (std::int64_t y = y0;; ++y) {
      for (std::int64_t x = x0;; ++x) {
        const std::uint64_t cell =
            (ordered_i64(x) - minimum_x) * grid_height +
            (ordered_i64(y) - minimum_y);
        members[cursor++] = {
            cell, static_cast<std::uint32_t>(id), 0};
        if (x == x1) break;
      }
      if (y == y1) break;
    }
  }
}

__device__ std::uint64_t full_pair_count(std::uint64_t count)
{
  const std::uint64_t half = count / 2;
  const std::uint64_t other =
      count & 1 ? count : count ? count - 1 : 0;
  return half * other;
}

__device__ std::uint64_t triangular_row_begin(
    std::uint64_t count, std::uint64_t row)
{
  // row * (2 * count - row - 1) / 2, divided before multiplying.
  std::uint64_t first = row;
  std::uint64_t second = 2 * count - row - 1;
  if (first & 1) {
    second /= 2;
  } else {
    first /= 2;
  }
  return first * second;
}

__device__ bool decode_triangular_pair(
    std::uint64_t count, std::uint64_t pair,
    std::uint64_t *first, std::uint64_t *second)
{
  if (count < 2 || pair >= full_pair_count(count)) return false;
  std::uint64_t low = 0;
  std::uint64_t high = count;
  while (low + 1 < high) {
    const std::uint64_t middle = low + (high - low) / 2;
    if (triangular_row_begin(count, middle) <= pair) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const std::uint64_t row_begin =
      triangular_row_begin(count, low);
  const std::uint64_t column = low + 1 + (pair - row_begin);
  if (column >= count) return false;
  *first = low;
  *second = column;
  return true;
}

__device__ std::uint64_t first_new_member_offset(
    const CellMember *members, std::uint64_t group_begin,
    std::uint64_t group_count,
    std::uint64_t previous_rectangle_count)
{
  /*
   * CellMemberLess orders equal-cell memberships by rectangle node.  Every
   * append adds all new rectangles after the retained prefix, so each cell
   * has one exact old/new split even when the appended owners span multiple
   * domains.
   */
  std::uint64_t low = 0;
  std::uint64_t high = group_count;
  while (low < high) {
    const std::uint64_t middle = low + (high - low) / 2;
    if (members[group_begin + middle].node <
        previous_rectangle_count) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

__device__ std::uint64_t staged_pair_count(
    std::uint64_t old_count, std::uint64_t group_count)
{
  const std::uint64_t new_count = group_count - old_count;
  return old_count * new_count + full_pair_count(new_count);
}

__device__ bool decode_staged_pair(
    std::uint64_t old_count, std::uint64_t group_count,
    std::uint64_t pair, std::uint64_t *first,
    std::uint64_t *second)
{
  const std::uint64_t new_count = group_count - old_count;
  if (!new_count || pair >= staged_pair_count(
                                old_count, group_count)) {
    return false;
  }
  const std::uint64_t old_new_count = old_count * new_count;
  if (pair < old_new_count) {
    *first = pair / new_count;
    *second = old_count + pair % new_count;
    return true;
  }
  if (!decode_triangular_pair(
          new_count, pair - old_new_count, first, second)) {
    return false;
  }
  *first += old_count;
  *second += old_count;
  return true;
}

__device__ std::int64_t decode_ordered_i64(
    std::uint64_t value)
{
  return static_cast<std::int64_t>(
      value ^ UINT64_C(0x8000000000000000));
}

__device__ std::int64_t saturating_cell_origin(
    std::int64_t cell, std::int64_t bin_size)
{
  /*
   * floor(INT64_MIN / bin_size) * bin_size can lie by at most
   * bin_size - 1 below INT64_MIN.  No positive quotient obtained from an
   * int64 coordinate can overflow.  Saturating the sole negative edge cell
   * preserves the canonical lower-bound comparison for every representable
   * rectangle coordinate.
   */
  if (cell < INT64_MIN / bin_size) return INT64_MIN;
  return cell * bin_size;
}

__device__ void cell_origin(
    std::uint64_t cell, std::uint64_t minimum_x,
    std::uint64_t minimum_y, std::uint64_t grid_height,
    std::int64_t bin_size, std::int64_t *left,
    std::int64_t *bottom)
{
  const std::uint64_t x_offset = cell / grid_height;
  const std::uint64_t y_offset = cell % grid_height;
  *left = saturating_cell_origin(
      decode_ordered_i64(minimum_x + x_offset), bin_size);
  *bottom = saturating_cell_origin(
      decode_ordered_i64(minimum_y + y_offset), bin_size);
}

__device__ bool exact_pair_key(
    const CellMember *members, std::uint64_t group_begin,
    std::uint64_t group_count, std::uint64_t local_pair,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    std::uint32_t stage_begin, std::int64_t cell_left,
    std::int64_t cell_bottom, std::uint64_t *key,
    std::uint32_t *status)
{
  std::uint64_t first_offset = 0;
  std::uint64_t second_offset = 0;
  if (!decode_triangular_pair(
          group_count, local_pair, &first_offset,
          &second_offset)) {
    atomicOr(status, std::uint32_t(kPairCountOverflow));
    return false;
  }
  const std::uint32_t first =
      members[group_begin + first_offset].node;
  const std::uint32_t second =
      members[group_begin + second_offset].node;
  const ac::RectI64 first_rectangle = rectangles[first];
  const ac::RectI64 second_rectangle = rectangles[second];
  if (first_rectangle.owner < stage_begin &&
      second_rectangle.owner < stage_begin) {
    return false;
  }
  if (first_rectangle.owner != second_rectangle.owner &&
      !allowed_relation(
          first_rectangle.domain, second_rectangle.domain,
          relation_rows)) {
    return false;
  }
  /*
   * Both rectangles contain the current cell.  It is their unique canonical
   * shared cell precisely when at least one rectangle begins in its x span
   * and at least one begins in its y span.  This is equivalent to
   * (max(lower_x), max(lower_y)) without a per-pair floor division.
   */
  if (max(first_rectangle.left, second_rectangle.left) <
          cell_left ||
      max(first_rectangle.bottom, second_rectangle.bottom) <
          cell_bottom) {
    return false;
  }
  if (first_rectangle.owner == second_rectangle.owner &&
      positive_area_overlap(first_rectangle, second_rectangle)) {
    atomicOr(status, std::uint32_t(kOverlappingOwnerTiles));
    return false;
  }
  if (!closed_touch_or_overlap(
          first_rectangle, second_rectangle)) {
    return false;
  }
  *key = pair_key(first, second);
  return true;
}

__global__ void count_staged_pair_tests_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts, std::uint64_t group_count,
    std::uint64_t previous_rectangle_count,
    std::uint32_t max_cell_members,
    std::uint64_t max_pair_tests_per_cell,
    std::uint64_t *pair_test_counts, std::uint32_t *status)
{
  for (std::uint64_t group =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       group < group_count;
       group += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t count = group_counts[group];
    if (count > max_cell_members) {
      atomicOr(status, std::uint32_t(kCellCapacity));
      pair_test_counts[group] = 0;
      continue;
    }
    const std::uint64_t old_count = first_new_member_offset(
        members, group_offsets[group], count,
        previous_rectangle_count);
    const std::uint64_t tests =
        staged_pair_count(old_count, count);
    if (tests > max_pair_tests_per_cell) {
      atomicOr(status, std::uint32_t(kCellCapacity));
      pair_test_counts[group] = 0;
      continue;
    }
    pair_test_counts[group] = tests;
  }
}

__global__ void count_exact_pair_occurrences_parallel_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    const std::uint64_t *pair_test_offsets,
    std::uint64_t total_pair_tests,
    std::uint64_t group_count,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    std::uint32_t stage_begin, std::int64_t bin_size,
    std::uint64_t minimum_x, std::uint64_t minimum_y,
    std::uint64_t grid_height,
    std::uint64_t *pair_counts, std::uint32_t *status)
{
  __shared__ unsigned long long partial[kThreads];
  __shared__ std::int64_t cell_left;
  __shared__ std::int64_t cell_bottom;
  for (std::uint64_t group = blockIdx.x; group < group_count;
       group += gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    const std::uint64_t test_begin = pair_test_offsets[group];
    const std::uint64_t test_end =
        group + 1 < group_count
            ? pair_test_offsets[group + 1]
            : total_pair_tests;
    const std::uint64_t tests = test_end - test_begin;
    if (!tests) {
      if (!threadIdx.x) pair_counts[group] = 0;
      continue;
    }
    if (!threadIdx.x) {
      cell_origin(
          members[begin].cell, minimum_x, minimum_y,
          grid_height, bin_size, &cell_left, &cell_bottom);
    }
    __syncthreads();
    unsigned long long local = 0;
    for (std::uint64_t pair = threadIdx.x; pair < tests;
         pair += blockDim.x) {
      std::uint64_t key = 0;
      if (exact_pair_key(
              members, begin, count, pair, rectangles,
              relation_rows, stage_begin, cell_left, cell_bottom,
              &key, status)) {
        ++local;
      }
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t offset = blockDim.x / 2; offset;
         offset /= 2) {
      if (threadIdx.x < offset) {
        partial[threadIdx.x] +=
            partial[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (!threadIdx.x) pair_counts[group] = partial[0];
    __syncthreads();
  }
}

__global__ void fill_exact_pair_occurrences_parallel_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    const std::uint64_t *pair_test_offsets,
    std::uint64_t total_pair_tests,
    std::uint64_t group_count,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    std::uint32_t stage_begin, std::int64_t bin_size,
    std::uint64_t minimum_x, std::uint64_t minimum_y,
    std::uint64_t grid_height,
    const std::uint64_t *pair_offsets,
    const std::uint64_t *pair_counts,
    std::uint64_t *pairs, std::uint32_t *status)
{
  __shared__ unsigned long long cursor;
  __shared__ std::int64_t cell_left;
  __shared__ std::int64_t cell_bottom;
  for (std::uint64_t group = blockIdx.x; group < group_count;
       group += gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    const std::uint64_t test_begin = pair_test_offsets[group];
    const std::uint64_t test_end =
        group + 1 < group_count
            ? pair_test_offsets[group + 1]
            : total_pair_tests;
    const std::uint64_t tests = test_end - test_begin;
    if (!tests) continue;
    if (!threadIdx.x) {
      cursor = pair_offsets[group];
      cell_origin(
          members[begin].cell, minimum_x, minimum_y,
          grid_height, bin_size, &cell_left, &cell_bottom);
    }
    __syncthreads();
    for (std::uint64_t pair = threadIdx.x; pair < tests;
         pair += blockDim.x) {
      std::uint64_t key = 0;
      if (exact_pair_key(
              members, begin, count, pair, rectangles,
              relation_rows, stage_begin, cell_left, cell_bottom,
              &key, status)) {
        const unsigned long long output =
            atomicAdd(&cursor, 1ull);
        pairs[output] = key;
      }
    }
    __syncthreads();
    if (!threadIdx.x &&
        cursor != pair_offsets[group] + pair_counts[group]) {
      atomicOr(status, std::uint32_t(kPairCountOverflow));
    }
    __syncthreads();
  }
}

enum ExactStreamKind : std::uint32_t
{
  kExactNone = 0,
  kExactTileEdge = 1,
  kExactSingletonOwnerEdge = 2,
  kExactExceptionOwnerEdge = 3
};

struct ExactStreamCounts
{
  unsigned long long tile_edges;
  unsigned long long singleton_owner_edges;
  unsigned long long exception_owner_edges;
};

__device__ void union_roots_atomic_cas(
    std::uint32_t *parents, std::uint32_t first,
    std::uint32_t second, std::uint64_t node_count,
    std::uint32_t *status);

__global__ void count_owner_rectangles_kernel(
    const ac::RectI64 *rectangles, std::uint64_t rectangle_count,
    std::uint32_t owner_count, std::uint32_t *owner_rectangles,
    std::uint32_t *status)
{
  for (std::uint64_t rectangle =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       rectangle < rectangle_count;
       rectangle += static_cast<std::uint64_t>(blockDim.x) *
                    gridDim.x) {
    const std::uint32_t owner = rectangles[rectangle].owner;
    if (owner >= owner_count) {
      atomicOr(status, std::uint32_t(kMalformedRectangle));
      continue;
    }
    const std::uint32_t prior =
        atomicAdd(owner_rectangles + owner, 1u);
    if (prior == UINT32_MAX) {
      atomicOr(status, std::uint32_t(kPairCountOverflow));
    }
  }
}

__device__ ExactStreamKind classify_exact_stream_pair(
    const CellMember *members, std::uint64_t group_begin,
    std::uint64_t first_offset, std::uint64_t second_offset,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    const std::uint32_t *owner_rectangles,
    std::uint32_t stage_begin, std::int64_t cell_left,
    std::int64_t cell_bottom, std::uint64_t *key,
    std::uint32_t *compact_relation, std::uint32_t domain_count,
    std::uint32_t *status)
{
  const std::uint32_t first =
      members[group_begin + first_offset].node;
  const std::uint32_t second =
      members[group_begin + second_offset].node;
  const ac::RectI64 first_rectangle = rectangles[first];
  const ac::RectI64 second_rectangle = rectangles[second];
  if (first_rectangle.owner < stage_begin &&
      second_rectangle.owner < stage_begin) {
    return kExactNone;
  }
  const bool same_owner =
      first_rectangle.owner == second_rectangle.owner;
  if (!same_owner &&
      !allowed_relation(
          first_rectangle.domain, second_rectangle.domain,
          relation_rows)) {
    return kExactNone;
  }
  if (max(first_rectangle.left, second_rectangle.left) <
          cell_left ||
      max(first_rectangle.bottom, second_rectangle.bottom) <
          cell_bottom) {
    return kExactNone;
  }
  if (same_owner &&
      positive_area_overlap(first_rectangle, second_rectangle)) {
    atomicOr(status, std::uint32_t(kOverlappingOwnerTiles));
    return kExactNone;
  }
  if (!closed_touch_or_overlap(
          first_rectangle, second_rectangle)) {
    return kExactNone;
  }
  if (same_owner) {
    *key = pair_key(first, second);
    return kExactTileEdge;
  }

  const std::uint32_t first_owner = first_rectangle.owner;
  const std::uint32_t second_owner = second_rectangle.owner;
  *key = pair_key(first_owner, second_owner);
  const std::uint32_t low_domain =
      min(first_rectangle.domain, second_rectangle.domain);
  const std::uint32_t high_domain =
      max(first_rectangle.domain, second_rectangle.domain);
  *compact_relation = low_domain * domain_count + high_domain;
  return owner_rectangles[first_owner] == 1 &&
                 owner_rectangles[second_owner] == 1
             ? kExactSingletonOwnerEdge
             : kExactExceptionOwnerEdge;
}

__global__ void count_exact_streams_parallel_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    const std::uint64_t *pair_test_offsets,
    std::uint64_t total_pair_tests,
    std::uint64_t group_count,
    std::uint64_t previous_rectangle_count,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    const std::uint32_t *owner_rectangles,
    std::uint32_t stage_begin, std::int64_t bin_size,
    std::uint64_t minimum_x, std::uint64_t minimum_y,
    std::uint64_t grid_height, std::uint32_t domain_count,
    std::uint32_t *owner_parents, std::uint64_t owner_count,
    unsigned long long *singleton_relation_census,
    std::uint64_t *tile_counts,
    std::uint64_t *singleton_counts,
    std::uint64_t *exception_counts, std::uint32_t *status)
{
  extern __shared__ unsigned long long relation_census[];
  __shared__ ExactStreamCounts partial[kThreads];
  __shared__ std::int64_t cell_left;
  __shared__ std::int64_t cell_bottom;
  __shared__ std::uint64_t old_count;
  const std::uint32_t relation_count =
      domain_count * domain_count;
  for (std::uint32_t relation = threadIdx.x;
       relation < relation_count; relation += blockDim.x) {
    relation_census[relation] = 0;
  }
  __syncthreads();
  for (std::uint64_t group = blockIdx.x; group < group_count;
       group += gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    const std::uint64_t test_begin = pair_test_offsets[group];
    const std::uint64_t test_end =
        group + 1 < group_count
            ? pair_test_offsets[group + 1]
            : total_pair_tests;
    const std::uint64_t tests = test_end - test_begin;
    if (!tests) {
      if (!threadIdx.x) {
        tile_counts[group] = 0;
        singleton_counts[group] = 0;
        exception_counts[group] = 0;
      }
      continue;
    }
    if (!threadIdx.x) {
      cell_origin(
          members[begin].cell, minimum_x, minimum_y,
          grid_height, bin_size, &cell_left, &cell_bottom);
      old_count = first_new_member_offset(
          members, begin, count, previous_rectangle_count);
    }
    __syncthreads();
    ExactStreamCounts local = {0, 0, 0};
    for (std::uint64_t pair = threadIdx.x; pair < tests;
         pair += blockDim.x) {
      std::uint64_t first_offset = 0;
      std::uint64_t second_offset = 0;
      if (!decode_staged_pair(
              old_count, count, pair, &first_offset,
              &second_offset)) {
        atomicOr(status, std::uint32_t(kPairCountOverflow));
        continue;
      }
      std::uint64_t key = 0;
      std::uint32_t relation = 0;
      const ExactStreamKind kind = classify_exact_stream_pair(
          members, begin, first_offset, second_offset,
          rectangles, relation_rows,
          owner_rectangles, stage_begin, cell_left, cell_bottom,
          &key, &relation, domain_count, status);
      if (kind == kExactTileEdge) {
        ++local.tile_edges;
      } else if (kind == kExactSingletonOwnerEdge) {
        std::uint32_t first_owner = 0;
        std::uint32_t second_owner = 0;
        decode_pair(key, &first_owner, &second_owner);
        union_roots_atomic_cas(
            owner_parents, first_owner, second_owner,
            owner_count, status);
        atomicAdd(relation_census + relation, 1ull);
        ++local.singleton_owner_edges;
      } else if (kind == kExactExceptionOwnerEdge) {
        ++local.exception_owner_edges;
      }
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride;
         stride /= 2) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x].tile_edges +=
            partial[threadIdx.x + stride].tile_edges;
        partial[threadIdx.x].singleton_owner_edges +=
            partial[threadIdx.x + stride].singleton_owner_edges;
        partial[threadIdx.x].exception_owner_edges +=
            partial[threadIdx.x + stride].exception_owner_edges;
      }
      __syncthreads();
    }
    if (!threadIdx.x) {
      tile_counts[group] = partial[0].tile_edges;
      singleton_counts[group] =
          partial[0].singleton_owner_edges;
      exception_counts[group] =
          partial[0].exception_owner_edges;
    }
    __syncthreads();
  }

  for (std::uint32_t relation = threadIdx.x;
       relation < relation_count; relation += blockDim.x) {
    const unsigned long long count = relation_census[relation];
    if (!count) continue;
    const std::uint32_t low_domain = relation / domain_count;
    const std::uint32_t high_domain = relation % domain_count;
    const std::size_t slot =
        static_cast<std::size_t>(low_domain) *
            ac::kMaximumDomains +
        high_domain;
    atomicAdd(singleton_relation_census + slot, count);
  }
}

__device__ std::uint32_t find_root_atomic(
    std::uint32_t *parents, std::uint32_t node,
    std::uint64_t node_count, std::uint32_t *status)
{
  const std::uint32_t start = node;
  std::uint32_t current = node;
  for (std::uint64_t step = 0; step < node_count; ++step) {
    const std::uint32_t parent =
        atomicAdd(parents + current, 0u);
    if (parent >= node_count || parent > current) {
      atomicOr(status, std::uint32_t(kInvalidParent));
      return UINT32_MAX;
    }
    if (parent == current) {
      atomicMin(parents + start, current);
      return current;
    }
    const std::uint32_t grandparent =
        atomicAdd(parents + parent, 0u);
    if (grandparent >= node_count || grandparent > parent) {
      atomicOr(status, std::uint32_t(kInvalidParent));
      return UINT32_MAX;
    }
    /*
     * Parent links only move toward smaller canonical labels.  Atomic path
     * halving therefore preserves the component/minimum-root invariant while
     * preventing a long streaming edge pass from repeatedly walking the
     * transient chains it creates.
     */
    atomicMin(parents + current, grandparent);
    current = grandparent;
  }
  atomicOr(status, std::uint32_t(kInvalidParent));
  return UINT32_MAX;
}

__device__ void union_roots_atomic_cas(
    std::uint32_t *parents, std::uint32_t first,
    std::uint32_t second, std::uint64_t node_count,
    std::uint32_t *status)
{
  for (std::uint64_t attempt = 0; attempt < node_count; ++attempt) {
    const std::uint32_t first_root =
        find_root_atomic(parents, first, node_count, status);
    const std::uint32_t second_root =
        find_root_atomic(parents, second, node_count, status);
    if (first_root == UINT32_MAX || second_root == UINT32_MAX ||
        first_root == second_root) {
      return;
    }
    const std::uint32_t low = min(first_root, second_root);
    const std::uint32_t high = max(first_root, second_root);
    if (atomicCAS(parents + high, high, low) == high) return;
  }
  atomicOr(status, std::uint32_t(kPairCountOverflow));
}

__global__ void count_multi_owner_members_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    std::uint64_t group_count,
    const ac::RectI64 *rectangles,
    const std::uint32_t *owner_rectangle_counts,
    std::uint64_t owner_count,
    std::uint64_t *multi_counts,
    std::uint32_t *status)
{
  __shared__ unsigned long long partial[kThreads];
  for (std::uint64_t group = blockIdx.x; group < group_count;
       group += gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    unsigned long long local = 0;
    for (std::uint64_t offset = threadIdx.x; offset < count;
         offset += blockDim.x) {
      const std::uint32_t rectangle =
          members[begin + offset].node;
      const std::uint32_t owner = rectangles[rectangle].owner;
      if (owner >= owner_count) {
        atomicOr(status, std::uint32_t(kMalformedRectangle));
      } else if (owner_rectangle_counts[owner] > 1) {
        ++local;
      }
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride;
         stride /= 2) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] +=
            partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (!threadIdx.x) multi_counts[group] = partial[0];
    __syncthreads();
  }
}

__global__ void fill_multi_owner_member_offsets_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    std::uint64_t group_count,
    const ac::RectI64 *rectangles,
    const std::uint32_t *owner_rectangle_counts,
    std::uint64_t owner_count,
    const std::uint64_t *multi_offsets,
    std::uint64_t multi_total,
    std::uint32_t *multi_member_offsets,
    std::uint32_t *status)
{
  for (std::uint64_t group = blockIdx.x; group < group_count;
       group += gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    if (!threadIdx.x) {
      std::uint64_t cursor = multi_offsets[group];
      const std::uint64_t end =
          group + 1 < group_count
              ? multi_offsets[group + 1]
              : multi_total;
      for (std::uint64_t offset = 0; offset < count; ++offset) {
        const std::uint32_t rectangle =
            members[begin + offset].node;
        const std::uint32_t owner = rectangles[rectangle].owner;
        if (owner >= owner_count) {
          atomicOr(status, std::uint32_t(kMalformedRectangle));
        } else if (owner_rectangle_counts[owner] > 1) {
          if (cursor < end) {
            multi_member_offsets[cursor++] =
                static_cast<std::uint32_t>(offset);
          } else {
            atomicOr(
                status, std::uint32_t(kPairCountOverflow));
          }
        }
      }
      if (cursor != end) {
        atomicOr(status, std::uint32_t(kPairCountOverflow));
      }
    }
    __syncthreads();
  }
}

__device__ std::uint64_t first_new_multi_member(
    const std::uint32_t *multi_member_offsets,
    std::uint64_t multi_begin, std::uint64_t multi_count,
    std::uint64_t old_count)
{
  std::uint64_t low = 0;
  std::uint64_t high = multi_count;
  while (low < high) {
    const std::uint64_t middle = low + (high - low) / 2;
    if (multi_member_offsets[multi_begin + middle] < old_count) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

__global__ void fill_exact_streams_parallel_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    const std::uint64_t *multi_offsets,
    std::uint64_t multi_total,
    const std::uint32_t *multi_member_offsets,
    std::uint64_t group_count,
    std::uint64_t previous_rectangle_count,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    const std::uint32_t *owner_rectangles,
    std::uint32_t stage_begin, std::int64_t bin_size,
    std::uint64_t minimum_x, std::uint64_t minimum_y,
    std::uint64_t grid_height, std::uint32_t domain_count,
    const std::uint64_t *tile_offsets,
    const std::uint64_t *exception_offsets,
    std::uint64_t tile_total, std::uint64_t exception_total,
    std::uint64_t *tile_edges, std::uint64_t *exception_keys,
    std::uint32_t *owner_parents, std::uint64_t owner_count,
    std::uint32_t *status)
{
  __shared__ unsigned long long tile_cursor;
  __shared__ unsigned long long exception_cursor;
  __shared__ std::int64_t cell_left;
  __shared__ std::int64_t cell_bottom;
  __shared__ std::uint64_t old_count;
  __shared__ std::uint64_t multi_begin;
  __shared__ std::uint64_t multi_count;
  __shared__ std::uint64_t old_multi_count;

  for (std::uint64_t group = blockIdx.x; group < group_count;
       group += gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    if (!threadIdx.x) {
      tile_cursor = tile_offsets[group];
      exception_cursor = exception_offsets[group];
      cell_origin(
          members[begin].cell, minimum_x, minimum_y,
          grid_height, bin_size, &cell_left, &cell_bottom);
      old_count = first_new_member_offset(
          members, begin, count, previous_rectangle_count);
      multi_begin = multi_offsets[group];
      const std::uint64_t multi_end =
          group + 1 < group_count
              ? multi_offsets[group + 1]
              : multi_total;
      multi_count = multi_end - multi_begin;
      old_multi_count = first_new_multi_member(
          multi_member_offsets, multi_begin, multi_count,
          old_count);
    }
    __syncthreads();
    const std::uint64_t new_count = count - old_count;
    const std::uint64_t new_multi_count =
        multi_count - old_multi_count;
    const std::uint64_t old_multi_new_tests =
        old_multi_count * new_count;
    const std::uint64_t new_multi_old_tests =
        new_multi_count * old_count;
    const std::uint64_t new_multi_new_tests =
        new_multi_count * new_count;
    const std::uint64_t tests =
        old_multi_new_tests + new_multi_old_tests +
        new_multi_new_tests;
    for (std::uint64_t pair = threadIdx.x; pair < tests;
         pair += blockDim.x) {
      std::uint64_t first_offset = 0;
      std::uint64_t second_offset = 0;
      if (pair < old_multi_new_tests) {
        first_offset = multi_member_offsets[
            multi_begin + pair / new_count];
        second_offset =
            old_count + pair % new_count;
      } else if (
          pair < old_multi_new_tests +
                     new_multi_old_tests) {
        const std::uint64_t local =
            pair - old_multi_new_tests;
        first_offset = local % old_count;
        second_offset = multi_member_offsets[
            multi_begin + old_multi_count +
            local / old_count];
        const std::uint32_t first_rectangle =
            members[begin + first_offset].node;
        const std::uint32_t first_owner =
            rectangles[first_rectangle].owner;
        if (owner_rectangles[first_owner] > 1) {
          continue;
        }
      } else {
        const std::uint64_t local =
            pair - old_multi_new_tests -
            new_multi_old_tests;
        first_offset = multi_member_offsets[
            multi_begin + old_multi_count +
            local / new_count];
        second_offset =
            old_count + local % new_count;
        if (first_offset == second_offset) continue;
        const std::uint32_t second_rectangle =
            members[begin + second_offset].node;
        const std::uint32_t second_owner =
            rectangles[second_rectangle].owner;
        if (owner_rectangles[second_owner] > 1 &&
            first_offset > second_offset) {
          continue;
        }
      }
      std::uint64_t key = 0;
      std::uint32_t relation = 0;
      const ExactStreamKind kind = classify_exact_stream_pair(
          members, begin, first_offset, second_offset,
          rectangles, relation_rows,
          owner_rectangles, stage_begin, cell_left, cell_bottom,
          &key, &relation, domain_count, status);
      if (kind == kExactTileEdge) {
        const std::uint64_t output = atomicAdd(&tile_cursor, 1ull);
        if (output < tile_total) {
          tile_edges[output] = key;
        } else {
          atomicOr(status, std::uint32_t(kPairCountOverflow));
        }
      } else if (kind == kExactExceptionOwnerEdge) {
        std::uint32_t first_owner = 0;
        std::uint32_t second_owner = 0;
        decode_pair(key, &first_owner, &second_owner);
        union_roots_atomic_cas(
            owner_parents, first_owner, second_owner,
            owner_count, status);
        const std::uint64_t output =
            atomicAdd(&exception_cursor, 1ull);
        if (output < exception_total) {
          exception_keys[output] = key;
        } else {
          atomicOr(status, std::uint32_t(kPairCountOverflow));
        }
      } else if (kind == kExactSingletonOwnerEdge) {
        atomicOr(status, std::uint32_t(kUnexpectedStreamKind));
      }
    }
    __syncthreads();
    if (!threadIdx.x) {
      const std::uint64_t tile_end =
          group + 1 < group_count
              ? tile_offsets[group + 1]
              : tile_total;
      const std::uint64_t exception_end =
          group + 1 < group_count
              ? exception_offsets[group + 1]
              : exception_total;
      if (tile_cursor != tile_end ||
          exception_cursor != exception_end) {
        atomicOr(status, std::uint32_t(kPairCountOverflow));
      }
    }
    __syncthreads();
  }
}

__global__ void census_owner_pairs_bounded_kernel(
    const std::uint64_t *pairs, std::uint64_t pair_count,
    const std::uint32_t *owner_domains,
    std::uint32_t domain_count,
    unsigned long long *global_census)
{
  extern __shared__ unsigned long long relation_census[];
  const std::uint32_t relation_count =
      domain_count * domain_count;
  for (std::uint32_t relation = threadIdx.x;
       relation < relation_count; relation += blockDim.x) {
    relation_census[relation] = 0;
  }
  __syncthreads();
  for (std::uint64_t pair_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       pair_id < pair_count;
       pair_id += static_cast<std::uint64_t>(blockDim.x) *
                  gridDim.x) {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    decode_pair(pairs[pair_id], &first, &second);
    const std::uint32_t low_domain =
        min(owner_domains[first], owner_domains[second]);
    const std::uint32_t high_domain =
        max(owner_domains[first], owner_domains[second]);
    atomicAdd(
        relation_census +
            low_domain * domain_count + high_domain,
        1ull);
  }
  __syncthreads();
  for (std::uint32_t relation = threadIdx.x;
       relation < relation_count; relation += blockDim.x) {
    const unsigned long long count = relation_census[relation];
    if (!count) continue;
    const std::uint32_t low_domain = relation / domain_count;
    const std::uint32_t high_domain = relation % domain_count;
    const std::size_t slot =
        static_cast<std::size_t>(low_domain) *
            ac::kMaximumDomains +
        high_domain;
    atomicAdd(global_census + slot, count);
  }
}

__global__ void count_pair_occurrences_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    std::uint64_t group_count,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    std::uint32_t stage_begin,
    bool exact_filter_before_materialization,
    std::uint32_t max_cell_members,
    std::uint64_t max_pair_tests_per_cell,
    std::uint64_t *pair_counts,
    std::uint32_t *status)
{
  for (std::uint64_t group =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       group < group_count;
       group += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    if (count > max_cell_members) {
      atomicOr(status, std::uint32_t(kCellCapacity));
      pair_counts[group] = 0;
      continue;
    }
    const std::uint64_t half = count / 2;
    const std::uint64_t other =
        count & 1 ? count : count ? count - 1 : 0;
    const std::uint64_t pair_tests = half * other;
    if (pair_tests > max_pair_tests_per_cell) {
      atomicOr(status, std::uint32_t(kCellCapacity));
      pair_counts[group] = 0;
      continue;
    }
    std::uint64_t pairs = 0;
    for (std::uint64_t first_offset = 0;
         first_offset < count; ++first_offset) {
      const std::uint32_t first =
          members[begin + first_offset].node;
      for (std::uint64_t second_offset = first_offset + 1;
           second_offset < count; ++second_offset) {
        const std::uint32_t second =
            members[begin + second_offset].node;
        const ac::RectI64 first_rectangle = rectangles[first];
        const ac::RectI64 second_rectangle = rectangles[second];
        if (first_rectangle.owner < stage_begin &&
            second_rectangle.owner < stage_begin) {
          continue;
        }
        if (first_rectangle.owner != second_rectangle.owner &&
            !allowed_relation(
                first_rectangle.domain, second_rectangle.domain,
                relation_rows)) {
          continue;
        }
        if (exact_filter_before_materialization) {
          if (first_rectangle.owner == second_rectangle.owner &&
              positive_area_overlap(
                  first_rectangle, second_rectangle)) {
            atomicOr(
                status, std::uint32_t(kOverlappingOwnerTiles));
            continue;
          }
          if (!closed_touch_or_overlap(
                  first_rectangle, second_rectangle)) {
            continue;
          }
        }
        if (pairs == UINT64_MAX) {
          atomicOr(status, std::uint32_t(kPairCountOverflow));
          pair_counts[group] = 0;
          return;
        }
        ++pairs;
      }
    }
    pair_counts[group] = pairs;
  }
}

__global__ void fill_pair_occurrences_kernel(
    const CellMember *members,
    const std::uint64_t *group_offsets,
    const std::uint64_t *group_counts,
    std::uint64_t group_count,
    const ac::RectI64 *rectangles,
    const std::uint64_t *relation_rows,
    std::uint32_t stage_begin,
    bool exact_filter_before_materialization,
    const std::uint64_t *pair_offsets,
    std::uint64_t *pairs)
{
  for (std::uint64_t group =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       group < group_count;
       group += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t begin = group_offsets[group];
    const std::uint64_t count = group_counts[group];
    std::uint64_t cursor = pair_offsets[group];
    for (std::uint64_t first_offset = 0;
         first_offset < count; ++first_offset) {
      const std::uint32_t first =
          members[begin + first_offset].node;
      for (std::uint64_t second_offset = first_offset + 1;
           second_offset < count; ++second_offset) {
        const std::uint32_t second =
            members[begin + second_offset].node;
        const ac::RectI64 first_rectangle = rectangles[first];
        const ac::RectI64 second_rectangle = rectangles[second];
        if (first_rectangle.owner < stage_begin &&
            second_rectangle.owner < stage_begin) {
          continue;
        }
        if (first_rectangle.owner != second_rectangle.owner &&
            !allowed_relation(
                first_rectangle.domain, second_rectangle.domain,
                relation_rows)) {
          continue;
        }
        if (exact_filter_before_materialization) {
          if (first_rectangle.owner == second_rectangle.owner &&
              positive_area_overlap(
                  first_rectangle, second_rectangle)) {
            continue;
          }
          if (!closed_touch_or_overlap(
                  first_rectangle, second_rectangle)) {
            continue;
          }
        }
        pairs[cursor++] = pair_key(first, second);
      }
    }
  }
}

__global__ void classify_owner_tiles_kernel(
    const std::uint64_t *pairs, std::uint64_t pair_count,
    const ac::RectI64 *rectangles,
    std::uint8_t *tile_edge_flags,
    std::uint32_t *status)
{
  for (std::uint64_t pair_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       pair_id < pair_count;
       pair_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    decode_pair(pairs[pair_id], &first, &second);
    const ac::RectI64 first_rectangle = rectangles[first];
    const ac::RectI64 second_rectangle = rectangles[second];
    tile_edge_flags[pair_id] = 0;
    if (first_rectangle.owner == second_rectangle.owner) {
      if (positive_area_overlap(
              first_rectangle, second_rectangle)) {
        atomicOr(
            status, std::uint32_t(kOverlappingOwnerTiles));
      } else if (closed_touch_or_overlap(
                     first_rectangle, second_rectangle)) {
        tile_edge_flags[pair_id] = 1;
      }
    }
  }
}

__global__ void classify_owner_pairs_kernel(
    const std::uint64_t *pairs, std::uint64_t pair_count,
    const ac::RectI64 *rectangles,
    std::uint64_t *owner_candidate_keys,
    std::uint8_t *owner_candidate_flags,
    std::uint64_t *owner_edge_keys,
    std::uint8_t *owner_edge_flags)
{
  for (std::uint64_t pair_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       pair_id < pair_count;
       pair_id += static_cast<std::uint64_t>(blockDim.x) *
                  gridDim.x) {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    decode_pair(pairs[pair_id], &first, &second);
    const ac::RectI64 first_rectangle = rectangles[first];
    const ac::RectI64 second_rectangle = rectangles[second];
    owner_candidate_flags[pair_id] = 0;
    owner_edge_flags[pair_id] = 0;
    if (first_rectangle.owner == second_rectangle.owner) continue;
    const std::uint64_t owner_pair = pair_key(
        first_rectangle.owner, second_rectangle.owner);
    owner_candidate_keys[pair_id] = owner_pair;
    owner_candidate_flags[pair_id] = 1;
    if (closed_touch_or_overlap(
            first_rectangle, second_rectangle)) {
      owner_edge_keys[pair_id] = owner_pair;
      owner_edge_flags[pair_id] = 1;
    }
  }
}

__global__ void census_owner_pairs_kernel(
    const std::uint64_t *pairs, std::uint64_t pair_count,
    const std::uint32_t *owner_domains,
    unsigned long long *relation_census)
{
  for (std::uint64_t pair_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       pair_id < pair_count;
       pair_id += static_cast<std::uint64_t>(blockDim.x) *
                  gridDim.x) {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    decode_pair(pairs[pair_id], &first, &second);
    const std::uint32_t low_domain =
        min(owner_domains[first], owner_domains[second]);
    const std::uint32_t high_domain =
        max(owner_domains[first], owner_domains[second]);
    const std::size_t relation =
        static_cast<std::size_t>(low_domain) *
            ac::kMaximumDomains +
        high_domain;
    atomicAdd(relation_census + relation, 1ULL);
  }
}

__device__ std::uint32_t find_root(
    const std::uint32_t *parents, std::uint32_t node,
    std::uint64_t node_count, std::uint32_t *status)
{
  std::uint32_t current = node;
  for (std::uint64_t step = 0; step < node_count; ++step) {
    const std::uint32_t parent = parents[current];
    if (parent >= node_count || parent > current) {
      atomicOr(status, std::uint32_t(kInvalidParent));
      return UINT32_MAX;
    }
    if (parent == current) return current;
    current = parent;
  }
  atomicOr(status, std::uint32_t(kInvalidParent));
  return UINT32_MAX;
}

__global__ void union_edges_kernel(
    const std::uint64_t *edges, std::uint64_t edge_count,
    std::uint32_t *parents, std::uint64_t node_count,
    std::uint32_t *changed, std::uint32_t *status)
{
  for (std::uint64_t edge_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       edge_id < edge_count;
       edge_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    decode_pair(edges[edge_id], &first, &second);
    for (std::uint64_t attempt = 0; attempt < node_count; ++attempt) {
      const std::uint32_t first_root =
          find_root(parents, first, node_count, status);
      const std::uint32_t second_root =
          find_root(parents, second, node_count, status);
      if (first_root == UINT32_MAX || second_root == UINT32_MAX ||
          first_root == second_root) {
        break;
      }
      const std::uint32_t low = min(first_root, second_root);
      const std::uint32_t high = max(first_root, second_root);
      const std::uint32_t prior = atomicMin(parents + high, low);
      if (prior > low) atomicExch(changed, 1u);
      if (prior == high || prior == low) break;
    }
  }
}

__global__ void compress_labels_kernel(
    std::uint32_t *parents, std::uint64_t node_count,
    std::uint32_t *changed, std::uint32_t *status)
{
  for (std::uint64_t node =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       node < node_count;
       node += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t root = find_root(
        parents, static_cast<std::uint32_t>(node),
        node_count, status);
    if (root == UINT32_MAX) continue;
    if (parents[node] != root) {
      parents[node] = root;
      atomicExch(changed, 1u);
    }
  }
}

__global__ void validate_labels_kernel(
    const std::uint32_t *parents, std::uint64_t node_count,
    std::uint32_t *status)
{
  for (std::uint64_t node =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       node < node_count;
       node += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t parent = parents[node];
    if (parent >= node_count || parent > node ||
        parents[parent] != parent) {
      atomicOr(status, std::uint32_t(kInvalidParent));
    }
  }
}

__global__ void first_rectangle_by_owner_kernel(
    const ac::RectI64 *rectangles, std::uint64_t rectangle_count,
    std::uint32_t *first_rectangle)
{
  for (std::uint64_t rectangle =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       rectangle < rectangle_count;
       rectangle += static_cast<std::uint64_t>(blockDim.x) *
                    gridDim.x) {
    atomicMin(
        first_rectangle + rectangles[rectangle].owner,
        static_cast<std::uint32_t>(rectangle));
  }
}

__global__ void validate_owner_tiles_kernel(
    const ac::RectI64 *rectangles,
    std::uint64_t previous_rectangle_count,
    std::uint64_t rectangle_count,
    const std::uint32_t *tile_labels,
    const std::uint32_t *first_rectangle,
    std::uint32_t *status)
{
  for (std::uint64_t rectangle =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x + previous_rectangle_count;
       rectangle < rectangle_count;
       rectangle += static_cast<std::uint64_t>(blockDim.x) *
                    gridDim.x) {
    const std::uint32_t owner = rectangles[rectangle].owner;
    const std::uint32_t first = first_rectangle[owner];
    if (first == UINT32_MAX ||
        tile_labels[rectangle] != tile_labels[first]) {
      atomicOr(
          status, std::uint32_t(kDisconnectedOwnerTiles));
    }
  }
}

std::uint32_t read_device_status(
    const thrust::device_vector<std::uint32_t> &status,
    const char *operation)
{
  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      operation);
  return host_status;
}

ac::Status map_device_status(std::uint32_t status)
{
  if (status & (kMalformedRectangle | kInvalidDomain |
                kInvalidParent | kOwnerDomainMismatch |
                kMissingOwner | kOverlappingOwnerTiles |
                kDisconnectedOwnerTiles |
                kClosedDomainInput | kUnexpectedStreamKind)) {
    return ac::Status::malformed_input;
  }
  if (status &
      (kSpanOverflow | kCellCapacity | kPairCountOverflow)) {
    return ac::Status::capacity_exceeded;
  }
  return ac::Status::success;
}

ac::Status run_min_dsu(
    const thrust::device_vector<std::uint64_t> &edges,
    thrust::device_vector<std::uint32_t> *parents,
    std::uint64_t node_count, std::uint32_t max_iterations,
    thrust::device_vector<std::uint32_t> *device_status,
    std::uint32_t *iterations)
{
  *iterations = 0;
  if (edges.empty()) return ac::Status::success;
  thrust::device_vector<std::uint32_t> changed(1, 0);
  bool converged = false;
  for (; *iterations < max_iterations; ++*iterations) {
    thrust::fill(
        thrust::device, changed.begin(), changed.end(), 0u);
    union_edges_kernel<<<
        launch_blocks(edges.size()), kThreads>>>(
        thrust::raw_pointer_cast(edges.data()), edges.size(),
        thrust::raw_pointer_cast(parents->data()), node_count,
        thrust::raw_pointer_cast(changed.data()),
        thrust::raw_pointer_cast(device_status->data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity DSU-union launch");
    compress_labels_kernel<<<
        launch_blocks(node_count), kThreads>>>(
        thrust::raw_pointer_cast(parents->data()), node_count,
        thrust::raw_pointer_cast(changed.data()),
        thrust::raw_pointer_cast(device_status->data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity DSU-compress launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "antenna connectivity DSU synchronize");
    const std::uint32_t dsu_flags = read_device_status(
        *device_status,
        "antenna connectivity DSU status D2H");
    if (dsu_flags) return map_device_status(dsu_flags);
    std::uint32_t host_changed = 0;
    cuda_require(
        cudaMemcpy(
            &host_changed,
            thrust::raw_pointer_cast(changed.data()),
            sizeof(host_changed), cudaMemcpyDeviceToHost),
        "antenna connectivity DSU changed D2H");
    if (!host_changed) {
      converged = true;
      ++*iterations;
      break;
    }
  }
  return converged ? ac::Status::success
                   : ac::Status::convergence_failure;
}

struct ExactStreamingResult
{
  std::uint64_t pair_occurrences = 0;
  std::uint64_t unique_candidates = 0;
  std::uint64_t edge_count = 0;
  std::uint32_t iterations = 0;
  thrust::device_vector<unsigned long long> candidate_census;
  thrust::device_vector<unsigned long long> edge_census;
};

ac::Status run_exact_streaming(
    const ac::Config &config, std::uint32_t stage_begin,
    std::uint64_t previous_rectangle_count,
    std::uint64_t rectangle_count, std::uint64_t total_count,
    std::uint64_t total_rectangle_count,
    std::uint64_t group_count, std::uint64_t minimum_x,
    std::uint64_t minimum_y, std::uint64_t grid_height,
    thrust::device_vector<ac::RectI64> *rectangles,
    thrust::device_vector<std::uint32_t> *owner_domains,
    thrust::device_vector<std::uint32_t> *owner_parents,
    thrust::device_vector<CellMember> *members,
    thrust::device_vector<std::uint64_t> *group_offsets,
    thrust::device_vector<std::uint64_t> *group_counts,
    thrust::device_vector<std::uint64_t> *relation_rows,
    thrust::device_vector<std::uint32_t> *device_status,
    ExactStreamingResult *result)
{
  if (!vector_allocation_admitted<std::uint32_t>(
          config, total_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint32_t>
      owner_rectangle_counts(total_count, 0);
  count_owner_rectangles_kernel<<<
      launch_blocks(total_rectangle_count), kThreads>>>(
      thrust::raw_pointer_cast(rectangles->data()),
      total_rectangle_count, static_cast<std::uint32_t>(total_count),
      thrust::raw_pointer_cast(owner_rectangle_counts.data()),
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity owner-rectangle-count launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity owner-rectangle-count synchronize");
  std::uint32_t flags = read_device_status(
      *device_status,
      "antenna connectivity owner-rectangle-count status D2H");
  if (flags) return map_device_status(flags);

  if (!vector_allocation_admitted<std::uint64_t>(
          config, group_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t>
      pair_test_counts(group_count);
  count_staged_pair_tests_kernel<<<
      launch_blocks(group_count), kThreads>>>(
      thrust::raw_pointer_cast(members->data()),
      thrust::raw_pointer_cast(group_offsets->data()),
      thrust::raw_pointer_cast(group_counts->data()),
      group_count, previous_rectangle_count,
      config.limits.max_cell_members,
      config.limits.max_pair_tests_per_cell,
      thrust::raw_pointer_cast(pair_test_counts.data()),
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming pair-work-count launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming pair-work-count synchronize");
  flags = read_device_status(
      *device_status,
      "antenna connectivity streaming pair-work-count status D2H");
  if (flags) return map_device_status(flags);
  const std::uint64_t total_pair_tests = thrust::reduce(
      thrust::device, pair_test_counts.begin(),
      pair_test_counts.end(), UINT64_C(0), SaturatingAddU64());
  if (total_pair_tests == UINT64_MAX ||
      total_pair_tests >
          config.limits.max_total_pair_tests) {
    return ac::Status::capacity_exceeded;
  }
  if (!vector_allocation_admitted<std::uint64_t>(
          config, group_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t>
      pair_test_offsets(group_count);
  thrust::exclusive_scan(
      thrust::device, pair_test_counts.begin(),
      pair_test_counts.end(), pair_test_offsets.begin(),
      UINT64_C(0));
  release_device_vector(&pair_test_counts);

  if (!vector_allocation_admitted<unsigned long long>(
          config, ac::kRelationSlots)) {
    return ac::Status::capacity_exceeded;
  }
  result->candidate_census.assign(ac::kRelationSlots, 0);
  const std::size_t relation_shared_bytes =
      static_cast<std::size_t>(config.domain_count) *
      config.domain_count * sizeof(unsigned long long);

  if (!vector_allocation_admitted<std::uint64_t>(
          config, group_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t> tile_counts(group_count);
  if (!vector_allocation_admitted<std::uint64_t>(
          config, group_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t>
      singleton_counts(group_count);
  if (!vector_allocation_admitted<std::uint64_t>(
          config, group_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t>
      exception_counts(group_count);
  const ConnectivityClock::time_point exact_count_begin =
      ConnectivityClock::now();
  count_exact_streams_parallel_kernel<<<
      launch_group_blocks(group_count), kThreads,
      relation_shared_bytes>>>(
      thrust::raw_pointer_cast(members->data()),
      thrust::raw_pointer_cast(group_offsets->data()),
      thrust::raw_pointer_cast(group_counts->data()),
      thrust::raw_pointer_cast(pair_test_offsets.data()),
      total_pair_tests, group_count,
      previous_rectangle_count,
      thrust::raw_pointer_cast(rectangles->data()),
      thrust::raw_pointer_cast(relation_rows->data()),
      thrust::raw_pointer_cast(owner_rectangle_counts.data()),
      stage_begin, config.bin_size, minimum_x, minimum_y,
      grid_height, config.domain_count,
      thrust::raw_pointer_cast(owner_parents->data()),
      total_count,
      thrust::raw_pointer_cast(
          result->candidate_census.data()),
      thrust::raw_pointer_cast(tile_counts.data()),
      thrust::raw_pointer_cast(singleton_counts.data()),
      thrust::raw_pointer_cast(exception_counts.data()),
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming exact-count launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming exact-count synchronize");
  const ConnectivityClock::time_point exact_count_end =
      ConnectivityClock::now();
  flags = read_device_status(
      *device_status,
      "antenna connectivity streaming exact-count status D2H");
  if (flags) return map_device_status(flags);

  const std::uint64_t tile_total = thrust::reduce(
      thrust::device, tile_counts.begin(), tile_counts.end(),
      UINT64_C(0), SaturatingAddU64());
  const std::uint64_t singleton_total = thrust::reduce(
      thrust::device, singleton_counts.begin(),
      singleton_counts.end(), UINT64_C(0), SaturatingAddU64());
  const std::uint64_t exception_total = thrust::reduce(
      thrust::device, exception_counts.begin(),
      exception_counts.end(), UINT64_C(0), SaturatingAddU64());
  if (tile_total == UINT64_MAX ||
      singleton_total == UINT64_MAX ||
      exception_total == UINT64_MAX ||
      singleton_total > UINT64_MAX - tile_total ||
      exception_total >
          UINT64_MAX - tile_total - singleton_total) {
    return ac::Status::capacity_exceeded;
  }
  result->pair_occurrences =
      tile_total + singleton_total + exception_total;
  if (result->pair_occurrences >
      config.limits.max_pair_occurrences) {
    return ac::Status::capacity_exceeded;
  }
  release_device_vector(&singleton_counts);
  release_device_vector(&pair_test_offsets);
  thrust::exclusive_scan(
      thrust::device, tile_counts.begin(), tile_counts.end(),
      tile_counts.begin(), UINT64_C(0));
  thrust::exclusive_scan(
      thrust::device, exception_counts.begin(),
      exception_counts.end(), exception_counts.begin(),
      UINT64_C(0));

  if (!vector_allocation_admitted<std::uint64_t>(
          config, tile_total)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t> tile_edges(tile_total);
  if (!vector_allocation_admitted<std::uint64_t>(
          config, exception_total)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t>
      exception_owner_keys(exception_total);

  if (!vector_allocation_admitted<std::uint64_t>(
          config, group_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint64_t>
      multi_offsets(group_count);
  const ConnectivityClock::time_point multi_index_begin =
      ConnectivityClock::now();
  count_multi_owner_members_kernel<<<
      launch_group_blocks(group_count), kThreads>>>(
      thrust::raw_pointer_cast(members->data()),
      thrust::raw_pointer_cast(group_offsets->data()),
      thrust::raw_pointer_cast(group_counts->data()),
      group_count,
      thrust::raw_pointer_cast(rectangles->data()),
      thrust::raw_pointer_cast(owner_rectangle_counts.data()),
      total_count,
      thrust::raw_pointer_cast(multi_offsets.data()),
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming multi-member-count launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming multi-member-count synchronize");
  flags = read_device_status(
      *device_status,
      "antenna connectivity streaming multi-member-count status D2H");
  if (flags) return map_device_status(flags);
  const std::uint64_t multi_total = thrust::reduce(
      thrust::device, multi_offsets.begin(), multi_offsets.end(),
      UINT64_C(0), SaturatingAddU64());
  if (multi_total == UINT64_MAX ||
      !vector_allocation_admitted<std::uint32_t>(
          config, multi_total)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::exclusive_scan(
      thrust::device, multi_offsets.begin(), multi_offsets.end(),
      multi_offsets.begin(), UINT64_C(0));
  thrust::device_vector<std::uint32_t>
      multi_member_offsets(multi_total);
  fill_multi_owner_member_offsets_kernel<<<
      launch_group_blocks(group_count), kThreads>>>(
      thrust::raw_pointer_cast(members->data()),
      thrust::raw_pointer_cast(group_offsets->data()),
      thrust::raw_pointer_cast(group_counts->data()),
      group_count,
      thrust::raw_pointer_cast(rectangles->data()),
      thrust::raw_pointer_cast(owner_rectangle_counts.data()),
      total_count,
      thrust::raw_pointer_cast(multi_offsets.data()),
      multi_total,
      multi_member_offsets.empty()
          ? nullptr
          : thrust::raw_pointer_cast(
                multi_member_offsets.data()),
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming multi-member-fill launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming multi-member-fill synchronize");
  const ConnectivityClock::time_point multi_index_end =
      ConnectivityClock::now();
  flags = read_device_status(
      *device_status,
      "antenna connectivity streaming multi-member-fill status D2H");
  if (flags) return map_device_status(flags);

  const ConnectivityClock::time_point exact_fill_begin =
      ConnectivityClock::now();
  fill_exact_streams_parallel_kernel<<<
      launch_group_blocks(group_count), kThreads>>>(
      thrust::raw_pointer_cast(members->data()),
      thrust::raw_pointer_cast(group_offsets->data()),
      thrust::raw_pointer_cast(group_counts->data()),
      thrust::raw_pointer_cast(multi_offsets.data()),
      multi_total,
      multi_member_offsets.empty()
          ? nullptr
          : thrust::raw_pointer_cast(
                multi_member_offsets.data()),
      group_count,
      previous_rectangle_count,
      thrust::raw_pointer_cast(rectangles->data()),
      thrust::raw_pointer_cast(relation_rows->data()),
      thrust::raw_pointer_cast(owner_rectangle_counts.data()),
      stage_begin, config.bin_size, minimum_x, minimum_y,
      grid_height, config.domain_count,
      thrust::raw_pointer_cast(tile_counts.data()),
      thrust::raw_pointer_cast(exception_counts.data()),
      tile_total, exception_total,
      tile_edges.empty()
          ? nullptr
          : thrust::raw_pointer_cast(tile_edges.data()),
      exception_owner_keys.empty()
          ? nullptr
          : thrust::raw_pointer_cast(exception_owner_keys.data()),
      thrust::raw_pointer_cast(owner_parents->data()),
      total_count,
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming exact-fill launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming exact-fill synchronize");
  const ConnectivityClock::time_point exact_fill_end =
      ConnectivityClock::now();
  flags = read_device_status(
      *device_status,
      "antenna connectivity streaming exact-fill status D2H");
  if (flags) return map_device_status(flags);
  if (connectivity_timing_enabled()) {
    std::fprintf(
        stderr,
        "KLAYOUT_CUDA_CONNECTIVITY_TIMING stage_begin=%u "
        "pair_tests=%llu singleton_edges=%llu "
        "exception_edges=%llu multi_members=%llu "
        "exact_count_union_ms=%.3f multi_index_ms=%.3f "
        "exact_fill_union_ms=%.3f\n",
        stage_begin,
        static_cast<unsigned long long>(total_pair_tests),
        static_cast<unsigned long long>(singleton_total),
        static_cast<unsigned long long>(exception_total),
        static_cast<unsigned long long>(multi_total),
        elapsed_milliseconds(
            exact_count_begin, exact_count_end),
        elapsed_milliseconds(
            multi_index_begin, multi_index_end),
        elapsed_milliseconds(
            exact_fill_begin, exact_fill_end));
  }
  release_device_vector(&owner_rectangle_counts);
  release_device_vector(&multi_member_offsets);
  release_device_vector(&multi_offsets);
  release_device_vector(&tile_counts);
  release_device_vector(&exception_counts);
  release_device_vector(members);
  release_device_vector(group_offsets);
  release_device_vector(group_counts);
  release_device_vector(relation_rows);

  if (!exception_owner_keys.empty()) {
    if (!sort_scratch_admitted<std::uint64_t>(
            config, exception_owner_keys.size())) {
      return ac::Status::capacity_exceeded;
    }
    thrust::sort(
        thrust::device, exception_owner_keys.begin(),
        exception_owner_keys.end());
    const auto end = thrust::unique(
        thrust::device, exception_owner_keys.begin(),
        exception_owner_keys.end());
    exception_owner_keys.resize(
        static_cast<std::size_t>(
            end - exception_owner_keys.begin()));
  }
  const std::uint64_t exception_unique =
      exception_owner_keys.size();
  if (exception_unique > UINT64_MAX - singleton_total) {
    return ac::Status::capacity_exceeded;
  }
  result->unique_candidates =
      singleton_total + exception_unique;
  result->edge_count = result->unique_candidates;
  if (result->unique_candidates >
      config.limits.max_unique_candidates) {
    return ac::Status::capacity_exceeded;
  }
  if (exception_unique) {
    census_owner_pairs_bounded_kernel<<<
        launch_blocks(exception_unique), kThreads,
        relation_shared_bytes>>>(
        thrust::raw_pointer_cast(exception_owner_keys.data()),
        exception_unique,
        thrust::raw_pointer_cast(owner_domains->data()),
        config.domain_count,
        thrust::raw_pointer_cast(
            result->candidate_census.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity streaming exception-census launch");
  }
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming relation-census synchronize");
  release_device_vector(&exception_owner_keys);
  if (!vector_allocation_admitted<unsigned long long>(
          config, ac::kRelationSlots)) {
    return ac::Status::capacity_exceeded;
  }
  result->edge_census = result->candidate_census;

  if (!vector_allocation_admitted<std::uint32_t>(
          config, total_rectangle_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint32_t>
      tile_parents(total_rectangle_count);
  thrust::sequence(
      thrust::device, tile_parents.begin(), tile_parents.end(), 0u);
  std::uint32_t tile_iterations = 0;
  ac::Status dsu_status = run_min_dsu(
      tile_edges, &tile_parents, total_rectangle_count,
      config.limits.max_dsu_iterations, device_status,
      &tile_iterations);
  if (dsu_status != ac::Status::success) return dsu_status;
  release_device_vector(&tile_edges);
  validate_labels_kernel<<<
      launch_blocks(total_rectangle_count), kThreads>>>(
      thrust::raw_pointer_cast(tile_parents.data()),
      total_rectangle_count,
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming tile-label validation launch");
  if (!vector_allocation_admitted<std::uint32_t>(
          config, total_count)) {
    return ac::Status::capacity_exceeded;
  }
  thrust::device_vector<std::uint32_t>
      first_rectangle(total_count, UINT32_MAX);
  first_rectangle_by_owner_kernel<<<
      launch_blocks(total_rectangle_count), kThreads>>>(
      thrust::raw_pointer_cast(rectangles->data()),
      total_rectangle_count,
      thrust::raw_pointer_cast(first_rectangle.data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming owner-first launch");
  validate_owner_tiles_kernel<<<
      launch_blocks(rectangle_count), kThreads>>>(
      thrust::raw_pointer_cast(rectangles->data()),
      previous_rectangle_count, total_rectangle_count,
      thrust::raw_pointer_cast(tile_parents.data()),
      thrust::raw_pointer_cast(first_rectangle.data()),
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming owner-tile validation launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming owner-tile validation synchronize");
  flags = read_device_status(
      *device_status,
      "antenna connectivity streaming owner-tile status D2H");
  if (flags) return map_device_status(flags);
  release_device_vector(&tile_parents);
  release_device_vector(&first_rectangle);

  const std::uint64_t cross_rectangle_edges =
      singleton_total + exception_total;
  if (cross_rectangle_edges) {
    if (!config.limits.max_dsu_iterations) {
      return ac::Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint32_t> changed(1, 0);
    compress_labels_kernel<<<
        launch_blocks(total_count), kThreads>>>(
        thrust::raw_pointer_cast(owner_parents->data()),
        total_count,
        thrust::raw_pointer_cast(changed.data()),
        thrust::raw_pointer_cast(device_status->data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity streaming owner-compress launch");
    result->iterations = 1;
  }
  validate_labels_kernel<<<
      launch_blocks(total_count), kThreads>>>(
      thrust::raw_pointer_cast(owner_parents->data()), total_count,
      thrust::raw_pointer_cast(device_status->data()));
  cuda_require(
      cudaGetLastError(),
      "antenna connectivity streaming label-validation launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna connectivity streaming label-validation synchronize");
  flags = read_device_status(
      *device_status,
      "antenna connectivity streaming label-validation status D2H");
  return flags ? map_device_status(flags) : ac::Status::success;
}

}  // namespace

namespace klayout_cuda {
namespace antenna_connectivity {

struct Connectivity::Impl
{
  explicit Impl(const Config &configuration)
      : config(configuration),
        config_status(validate_config(configuration))
  {
  }

  Config config;
  Status config_status;
  std::uint64_t owner_count = 0;
  std::uint64_t closed_domains = 0;
  std::uint64_t epoch = 0;
  bool poisoned = false;
  thrust::device_vector<RectI64> rectangles;
  thrust::device_vector<std::uint32_t> owner_domains;
  thrust::device_vector<std::uint32_t> parents;
};

Connectivity::Connectivity(const Config &config)
    : m_impl(new Impl(config))
{
}

Connectivity::~Connectivity() = default;

Status Connectivity::configuration_status() const noexcept
{
  if (!m_impl) return Status::host_error;
  return m_impl->poisoned ? Status::poisoned_state
                          : m_impl->config_status;
}

std::uint64_t Connectivity::node_count() const noexcept
{
  return m_impl
             ? (m_impl->poisoned ? 0 : m_impl->owner_count)
             : 0;
}

Status Connectivity::append_stage(
    const RectI64 *rectangles, std::uint64_t rectangle_count,
    std::uint64_t new_node_count,
    std::uint64_t close_domain_mask, InputMemory memory,
    std::vector<std::uint32_t> *labels,
    StageCensus *census, AppendMode mode) noexcept
{
  return append_stage_impl(
      rectangles, rectangle_count, new_node_count,
      close_domain_mask, memory, labels, census, mode, nullptr);
}

Status Connectivity::append_stage_consuming(
    thrust::device_vector<RectI64> &&rectangles,
    std::uint64_t new_node_count,
    std::uint64_t close_domain_mask,
    std::vector<std::uint32_t> *labels,
    StageCensus *census) noexcept
{
  const std::uint64_t rectangle_count = rectangles.size();
  const RectI64 *device_rectangles =
      rectangles.empty()
          ? nullptr
          : thrust::raw_pointer_cast(rectangles.data());
  return append_stage_impl(
      device_rectangles, rectangle_count, new_node_count,
      close_domain_mask, InputMemory::device, labels, census,
      AppendMode::consuming, &rectangles);
}

Status Connectivity::compact_retained_rectangles(
    std::uint64_t *retained_rectangle_capacity) noexcept
{
  if (!m_impl) return Status::host_error;
  if (m_impl->poisoned) return Status::poisoned_state;
  if (m_impl->config_status != Status::success) {
    return m_impl->config_status;
  }

  try {
    const Config &config = m_impl->config;
    cuda_require(
        cudaSetDevice(config.device),
        "antenna connectivity compaction cudaSetDevice");
    const std::uint64_t size = m_impl->rectangles.size();
    if (m_impl->rectangles.capacity() == size) {
      if (retained_rectangle_capacity) {
        *retained_rectangle_capacity = size;
      }
      return Status::success;
    }

    thrust::device_vector<RectI64> compact;
    if (size) {
      if (!vector_allocation_admitted<RectI64>(config, size)) {
        return Status::capacity_exceeded;
      }
      compact.assign(
          m_impl->rectangles.begin(), m_impl->rectangles.end());
      cuda_require(
          cudaDeviceSynchronize(),
          "antenna connectivity compaction synchronize");
    }

    m_impl->rectangles.swap(compact);
    if (retained_rectangle_capacity) {
      *retained_rectangle_capacity = m_impl->rectangles.capacity();
    }
    return Status::success;
  } catch (const CudaFailure &) {
    return Status::cuda_error;
  } catch (const thrust::system_error &) {
    return Status::cuda_error;
  } catch (const std::bad_alloc &) {
    return Status::host_error;
  } catch (...) {
    return Status::host_error;
  }
}

Status Connectivity::append_stage_impl(
    const RectI64 *rectangles, std::uint64_t rectangle_count,
    std::uint64_t new_node_count,
    std::uint64_t close_domain_mask, InputMemory memory,
    std::vector<std::uint32_t> *labels,
    StageCensus *census, AppendMode mode,
    thrust::device_vector<RectI64> *owned_input) noexcept
{
  if (!m_impl || m_impl->config_status != Status::success) {
    return m_impl ? m_impl->config_status : Status::host_error;
  }
  if (m_impl->poisoned) return Status::poisoned_state;
  if (!rectangles || !rectangle_count || !new_node_count ||
      !census ||
      (memory != InputMemory::host &&
       memory != InputMemory::device) ||
      (mode != AppendMode::transactional &&
       mode != AppendMode::consuming)) {
    return Status::malformed_input;
  }

  try {
    const Config &config = m_impl->config;
    const std::uint64_t domain_mask =
        config.domain_count == kMaximumDomains
            ? UINT64_MAX
            : (UINT64_C(1) << config.domain_count) - 1;
    if (close_domain_mask & ~domain_mask) {
      return Status::malformed_input;
    }
    const std::uint64_t previous_rectangle_count =
        m_impl->rectangles.size();
    const std::uint64_t previous_count = m_impl->owner_count;
    if (new_node_count >
            config.limits.max_nodes - previous_count ||
        previous_count + new_node_count > UINT32_MAX ||
        rectangle_count >
            config.limits.max_rectangles -
                previous_rectangle_count ||
        previous_rectangle_count + rectangle_count >
            UINT32_MAX) {
      return Status::capacity_exceeded;
    }
    const std::uint64_t total_count =
        previous_count + new_node_count;
    const std::uint64_t total_rectangle_count =
        previous_rectangle_count + rectangle_count;
    const std::uint32_t stage_begin =
        static_cast<std::uint32_t>(previous_count);

    cuda_require(
        cudaSetDevice(config.device),
        "antenna connectivity cudaSetDevice");

    thrust::device_vector<RectI64> work_rectangles;
    thrust::device_vector<std::uint32_t> work_owner_domains;
    thrust::device_vector<std::uint32_t> work_parents;
    const bool move_owned_first_stage =
        mode == AppendMode::consuming && owned_input &&
        previous_rectangle_count == 0;
    if (mode == AppendMode::consuming) {
      // From here on, any failure deliberately poisons this production-local
      // state.  Moving before growth lets device_vector release the old
      // allocation as soon as its replacement is installed instead of
      // retaining a transactional clone for the whole stage.
      if (move_owned_first_stage) {
        work_rectangles.swap(*owned_input);
      } else {
        work_rectangles.swap(m_impl->rectangles);
      }
      work_owner_domains.swap(m_impl->owner_domains);
      work_parents.swap(m_impl->parents);
      m_impl->poisoned = true;
      m_impl->owner_count = 0;
      ++m_impl->epoch;
      if (work_rectangles.size() != total_rectangle_count) {
        if (!vector_growth_admitted(
                config, work_rectangles,
                total_rectangle_count)) {
          return Status::capacity_exceeded;
        }
        work_rectangles.resize(total_rectangle_count);
      }
      if (!vector_growth_admitted(
              config, work_owner_domains, total_count)) {
        return Status::capacity_exceeded;
      }
      work_owner_domains.resize(total_count, UINT32_MAX);
      if (!vector_growth_admitted(
              config, work_parents, total_count)) {
        return Status::capacity_exceeded;
      }
      work_parents.resize(total_count);
    } else {
      if (!vector_allocation_admitted<RectI64>(
              config, total_rectangle_count)) {
        return Status::capacity_exceeded;
      }
      work_rectangles.resize(total_rectangle_count);
      if (previous_rectangle_count) {
        thrust::copy(
            thrust::device, m_impl->rectangles.begin(),
            m_impl->rectangles.end(), work_rectangles.begin());
      }
      if (!vector_allocation_admitted<std::uint32_t>(
              config, total_count)) {
        return Status::capacity_exceeded;
      }
      work_owner_domains.assign(total_count, UINT32_MAX);
      if (previous_count) {
        thrust::copy(
            thrust::device, m_impl->owner_domains.begin(),
            m_impl->owner_domains.end(),
            work_owner_domains.begin());
      }
      if (!vector_allocation_admitted<std::uint32_t>(
              config, total_count)) {
        return Status::capacity_exceeded;
      }
      work_parents.resize(total_count);
      if (previous_count) {
        thrust::copy(
            thrust::device, m_impl->parents.begin(),
            m_impl->parents.end(), work_parents.begin());
      }
    }
    if (!move_owned_first_stage) {
      cuda_require(
          cudaMemcpy(
              thrust::raw_pointer_cast(work_rectangles.data()) +
                  previous_rectangle_count,
              rectangles, rectangle_count * sizeof(RectI64),
              memory == InputMemory::host
                  ? cudaMemcpyHostToDevice
                  : cudaMemcpyDeviceToDevice),
          "antenna connectivity stage geometry copy");
      if (owned_input) release_device_vector(owned_input);
    }

    thrust::sequence(
        thrust::device,
        work_parents.begin() + previous_count,
        work_parents.end(), stage_begin);

    thrust::device_vector<std::uint64_t> relation_rows(
        config.relation_rows.begin(),
        config.relation_rows.end());
    thrust::device_vector<std::uint32_t> device_status(1, 0);
    const std::array<unsigned long long, 4> initial_cell_bounds = {
        ULLONG_MAX, ULLONG_MAX, 0, 0};
    thrust::device_vector<unsigned long long> cell_bounds(
        initial_cell_bounds.begin(), initial_cell_bounds.end());
    if (!vector_allocation_admitted<std::uint64_t>(
            config, total_rectangle_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        membership_counts(total_rectangle_count, 0);
    count_memberships_kernel<<<
        launch_blocks(total_rectangle_count), kThreads>>>(
        thrust::raw_pointer_cast(work_rectangles.data()),
        total_rectangle_count, previous_rectangle_count,
        stage_begin, static_cast<std::uint32_t>(total_count),
        config.domain_count, config.bin_size,
        m_impl->closed_domains,
        thrust::raw_pointer_cast(work_owner_domains.data()),
        thrust::raw_pointer_cast(membership_counts.data()),
        thrust::raw_pointer_cast(cell_bounds.data()),
        thrust::raw_pointer_cast(device_status.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity membership-count launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "antenna connectivity membership-count synchronize");
    validate_new_owners_kernel<<<
        launch_blocks(new_node_count), kThreads>>>(
        thrust::raw_pointer_cast(work_owner_domains.data()),
        stage_begin, static_cast<std::uint32_t>(total_count),
        thrust::raw_pointer_cast(device_status.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity owner-validation launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "antenna connectivity owner-validation synchronize");
    std::uint32_t flags = read_device_status(
        device_status,
        "antenna connectivity membership status D2H");
    if (flags) return map_device_status(flags);
    std::array<unsigned long long, 4> host_cell_bounds{};
    cuda_require(
        cudaMemcpy(
            host_cell_bounds.data(),
            thrust::raw_pointer_cast(cell_bounds.data()),
            sizeof(host_cell_bounds), cudaMemcpyDeviceToHost),
        "antenna connectivity cell bounds D2H");
    release_device_vector(&cell_bounds);
    const auto inclusive_ordered_span =
        [](std::uint64_t low, std::uint64_t high,
           std::uint64_t *size) {
          if (high < low || high - low == UINT64_MAX) {
            return false;
          }
          *size = high - low + 1;
          return true;
        };
    std::uint64_t grid_width = 0;
    std::uint64_t grid_height = 0;
    std::uint64_t grid_cells = 0;
    if (!inclusive_ordered_span(
            host_cell_bounds[0], host_cell_bounds[2],
            &grid_width) ||
        !inclusive_ordered_span(
            host_cell_bounds[1], host_cell_bounds[3],
            &grid_height) ||
        !byte_product(grid_width, grid_height, &grid_cells) ||
        !grid_cells) {
      return Status::capacity_exceeded;
    }

    const std::uint64_t membership_total = thrust::reduce(
        thrust::device, membership_counts.begin(),
        membership_counts.end(), UINT64_C(0),
        SaturatingAddU64());
    if (!membership_total || membership_total == UINT64_MAX ||
        membership_total > config.limits.max_memberships) {
      return Status::capacity_exceeded;
    }

    // Only scan after a saturating aggregate proves that every prefix and the
    // terminal offset fit uint64 and the configured allocation envelope.
    if (!vector_allocation_admitted<std::uint64_t>(
            config, total_rectangle_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        membership_offsets(total_rectangle_count);
    thrust::exclusive_scan(
        thrust::device, membership_counts.begin(),
        membership_counts.end(), membership_offsets.begin(),
        UINT64_C(0));

    if (!vector_allocation_admitted<CellMember>(
            config, membership_total)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<CellMember>
        members(membership_total);
    fill_memberships_kernel<<<
        launch_blocks(total_rectangle_count), kThreads>>>(
        thrust::raw_pointer_cast(work_rectangles.data()),
        total_rectangle_count, config.bin_size,
        thrust::raw_pointer_cast(membership_offsets.data()),
        host_cell_bounds[0], host_cell_bounds[1], grid_height,
        thrust::raw_pointer_cast(members.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity membership-fill launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "antenna connectivity membership-fill synchronize");
    release_device_vector(&membership_counts);
    release_device_vector(&membership_offsets);
    if (!sort_scratch_admitted<CellMember>(
            config, membership_total)) {
      return Status::capacity_exceeded;
    }
    thrust::sort(
        thrust::device, members.begin(), members.end(),
        CellMemberLess());

    if (!vector_allocation_admitted<std::uint64_t>(
            config, membership_total)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        group_counts(membership_total);
    const auto group_end = thrust::reduce_by_key(
        thrust::device, members.begin(), members.end(),
        thrust::make_constant_iterator<std::uint64_t>(1),
        thrust::make_discard_iterator(), group_counts.begin(),
        SameCell(),
        thrust::plus<std::uint64_t>());
    const std::uint64_t group_count =
        static_cast<std::uint64_t>(
            group_end.second - group_counts.begin());
    group_counts.resize(group_count);
    if (group_counts.capacity() > group_counts.size()) {
      if (!vector_allocation_admitted<std::uint64_t>(
              config, group_count)) {
        return Status::capacity_exceeded;
      }
      compact_device_vector(&group_counts);
    }
    if (!vector_allocation_admitted<std::uint64_t>(
            config, group_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        group_offsets(group_count);
    thrust::exclusive_scan(
        thrust::device, group_counts.begin(),
        group_counts.end(), group_offsets.begin(),
        UINT64_C(0));

    std::uint64_t pair_occurrences = 0;
    std::uint64_t unique_candidates = 0;
    std::uint64_t edge_count = 0;
    std::uint32_t iterations = 0;
    thrust::device_vector<unsigned long long> candidate_census;
    thrust::device_vector<unsigned long long> edge_census;

    if (config.exact_filter_before_materialization) {
      ExactStreamingResult exact_result;
      const Status exact_status = run_exact_streaming(
          config, stage_begin, previous_rectangle_count,
          rectangle_count, total_count, total_rectangle_count,
          group_count, host_cell_bounds[0], host_cell_bounds[1],
          grid_height, &work_rectangles, &work_owner_domains,
          &work_parents, &members, &group_offsets, &group_counts,
          &relation_rows, &device_status, &exact_result);
      if (exact_status != Status::success) return exact_status;
      pair_occurrences = exact_result.pair_occurrences;
      unique_candidates = exact_result.unique_candidates;
      edge_count = exact_result.edge_count;
      iterations = exact_result.iterations;
      candidate_census.swap(exact_result.candidate_census);
      edge_census.swap(exact_result.edge_census);
    } else {
    if (!vector_allocation_admitted<std::uint64_t>(
            config, group_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        pair_counts(group_count, 0);
    thrust::device_vector<std::uint64_t> pair_test_offsets;
    std::uint64_t total_pair_tests = 0;
    if (config.exact_filter_before_materialization) {
      if (!vector_allocation_admitted<std::uint64_t>(
              config, group_count)) {
        return Status::capacity_exceeded;
      }
      thrust::device_vector<std::uint64_t>
          pair_test_counts(group_count);
      count_staged_pair_tests_kernel<<<
          launch_blocks(group_count), kThreads>>>(
          thrust::raw_pointer_cast(members.data()),
          thrust::raw_pointer_cast(group_offsets.data()),
          thrust::raw_pointer_cast(group_counts.data()),
          group_count, previous_rectangle_count,
          config.limits.max_cell_members,
          config.limits.max_pair_tests_per_cell,
          thrust::raw_pointer_cast(pair_test_counts.data()),
          thrust::raw_pointer_cast(device_status.data()));
      cuda_require(
          cudaGetLastError(),
          "antenna connectivity pair-work-count launch");
      cuda_require(
          cudaDeviceSynchronize(),
          "antenna connectivity pair-work-count synchronize");
      flags = read_device_status(
          device_status,
          "antenna connectivity pair-work-count status D2H");
      if (flags) return map_device_status(flags);

      total_pair_tests = thrust::reduce(
          thrust::device, pair_test_counts.begin(),
          pair_test_counts.end(), UINT64_C(0),
          SaturatingAddU64());
      if (total_pair_tests == UINT64_MAX ||
          total_pair_tests >
              config.limits.max_total_pair_tests) {
        return Status::capacity_exceeded;
      }
      if (!vector_allocation_admitted<std::uint64_t>(
              config, group_count)) {
        return Status::capacity_exceeded;
      }
      pair_test_offsets.resize(group_count);
      thrust::exclusive_scan(
          thrust::device, pair_test_counts.begin(),
          pair_test_counts.end(), pair_test_offsets.begin(),
          UINT64_C(0));
      release_device_vector(&pair_test_counts);

      count_exact_pair_occurrences_parallel_kernel<<<
          launch_group_blocks(group_count), kThreads>>>(
          thrust::raw_pointer_cast(members.data()),
          thrust::raw_pointer_cast(group_offsets.data()),
          thrust::raw_pointer_cast(group_counts.data()),
          thrust::raw_pointer_cast(pair_test_offsets.data()),
          total_pair_tests, group_count,
          thrust::raw_pointer_cast(work_rectangles.data()),
          thrust::raw_pointer_cast(relation_rows.data()),
          stage_begin, config.bin_size, host_cell_bounds[0],
          host_cell_bounds[1], grid_height,
          thrust::raw_pointer_cast(pair_counts.data()),
          thrust::raw_pointer_cast(device_status.data()));
      cuda_require(
          cudaGetLastError(),
          "antenna connectivity exact pair-count launch");
      cuda_require(
          cudaDeviceSynchronize(),
          "antenna connectivity exact pair-count synchronize");
      flags = read_device_status(
          device_status,
          "antenna connectivity exact pair-count status D2H");
      if (flags) return map_device_status(flags);
    } else {
      count_pair_occurrences_kernel<<<
          launch_blocks(group_count), kThreads>>>(
          thrust::raw_pointer_cast(members.data()),
          thrust::raw_pointer_cast(group_offsets.data()),
          thrust::raw_pointer_cast(group_counts.data()),
          group_count,
          thrust::raw_pointer_cast(work_rectangles.data()),
          thrust::raw_pointer_cast(relation_rows.data()),
          stage_begin, false,
          config.limits.max_cell_members,
          config.limits.max_pair_tests_per_cell,
          thrust::raw_pointer_cast(pair_counts.data()),
          thrust::raw_pointer_cast(device_status.data()));
      cuda_require(
          cudaGetLastError(),
          "antenna connectivity pair-count launch");
      cuda_require(
          cudaDeviceSynchronize(),
          "antenna connectivity pair-count synchronize");
      flags = read_device_status(
          device_status,
          "antenna connectivity pair-count status D2H");
      if (flags) return map_device_status(flags);
    }

    pair_occurrences = thrust::reduce(
        thrust::device, pair_counts.begin(), pair_counts.end(),
        UINT64_C(0), SaturatingAddU64());
    if (pair_occurrences == UINT64_MAX ||
        pair_occurrences >
        config.limits.max_pair_occurrences) {
      return Status::capacity_exceeded;
    }

    if (!vector_allocation_admitted<std::uint64_t>(
            config, group_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        pair_offsets(group_count);
    thrust::exclusive_scan(
        thrust::device, pair_counts.begin(), pair_counts.end(),
        pair_offsets.begin(), UINT64_C(0));
    if (!config.exact_filter_before_materialization) {
      release_device_vector(&pair_counts);
    }
    if (!vector_allocation_admitted<std::uint64_t>(
            config, pair_occurrences)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        candidate_pairs(pair_occurrences);
    if (pair_occurrences) {
      if (config.exact_filter_before_materialization) {
        fill_exact_pair_occurrences_parallel_kernel<<<
            launch_group_blocks(group_count), kThreads>>>(
            thrust::raw_pointer_cast(members.data()),
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(group_counts.data()),
            thrust::raw_pointer_cast(pair_test_offsets.data()),
            total_pair_tests, group_count,
            thrust::raw_pointer_cast(work_rectangles.data()),
            thrust::raw_pointer_cast(relation_rows.data()),
            stage_begin, config.bin_size, host_cell_bounds[0],
            host_cell_bounds[1], grid_height,
            thrust::raw_pointer_cast(pair_offsets.data()),
            thrust::raw_pointer_cast(pair_counts.data()),
            thrust::raw_pointer_cast(candidate_pairs.data()),
            thrust::raw_pointer_cast(device_status.data()));
        cuda_require(
            cudaGetLastError(),
            "antenna connectivity exact pair-fill launch");
        cuda_require(
            cudaDeviceSynchronize(),
            "antenna connectivity exact pair-fill synchronize");
        flags = read_device_status(
            device_status,
            "antenna connectivity exact pair-fill status D2H");
        if (flags) return map_device_status(flags);
      } else {
        fill_pair_occurrences_kernel<<<
            launch_blocks(group_count), kThreads>>>(
            thrust::raw_pointer_cast(members.data()),
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(group_counts.data()),
            group_count,
            thrust::raw_pointer_cast(work_rectangles.data()),
            thrust::raw_pointer_cast(relation_rows.data()),
            stage_begin, false,
            thrust::raw_pointer_cast(pair_offsets.data()),
            thrust::raw_pointer_cast(candidate_pairs.data()));
        cuda_require(
            cudaGetLastError(),
            "antenna connectivity pair-fill launch");
        cuda_require(
            cudaDeviceSynchronize(),
            "antenna connectivity pair-fill synchronize");
      }
    }
    release_device_vector(&pair_counts);
    release_device_vector(&pair_test_offsets);
    release_device_vector(&members);
    release_device_vector(&group_offsets);
    release_device_vector(&group_counts);
    release_device_vector(&pair_offsets);
    release_device_vector(&relation_rows);
    if (pair_occurrences &&
        !config.exact_filter_before_materialization) {
      if (!sort_scratch_admitted<std::uint64_t>(
              config, pair_occurrences)) {
        return Status::capacity_exceeded;
      }
      thrust::sort(
          thrust::device, candidate_pairs.begin(),
          candidate_pairs.end());
      const auto unique_end = thrust::unique(
          thrust::device, candidate_pairs.begin(),
          candidate_pairs.end());
      candidate_pairs.resize(
          static_cast<std::size_t>(
              unique_end - candidate_pairs.begin()));
    }
    /*
     * Exact mode emits an unordered rectangle pair only in the unique cell
     * (max(lower_x), max(lower_y)).  Each rectangle has one membership per
     * cell and each group enumerates an unordered member pair once, so the
     * exact stream is already unique by rectangle-pair key.  Broad mode
     * deliberately retains per-cell occurrence semantics and still requires
     * sort/unique above.
     */
    if (!config.exact_filter_before_materialization &&
        candidate_pairs.capacity() > candidate_pairs.size()) {
      if (!vector_allocation_admitted<std::uint64_t>(
              config, candidate_pairs.size())) {
        return Status::capacity_exceeded;
      }
      compact_device_vector(&candidate_pairs);
    }

    const std::uint64_t rectangle_candidates =
        candidate_pairs.size();

    // Validate owner rectangulations first, then release all tile-only state
    // before allocating owner-pair arrays.
    if (!vector_allocation_admitted<std::uint8_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint8_t>
        tile_edge_flags(rectangle_candidates, 0);
    if (rectangle_candidates) {
      classify_owner_tiles_kernel<<<
          launch_blocks(rectangle_candidates), kThreads>>>(
          thrust::raw_pointer_cast(candidate_pairs.data()),
          rectangle_candidates,
          thrust::raw_pointer_cast(work_rectangles.data()),
          thrust::raw_pointer_cast(tile_edge_flags.data()),
          thrust::raw_pointer_cast(device_status.data()));
      cuda_require(
          cudaGetLastError(),
          "antenna connectivity owner-tile classify launch");
      cuda_require(
          cudaDeviceSynchronize(),
          "antenna connectivity owner-tile classify synchronize");
      flags = read_device_status(
          device_status,
          "antenna connectivity owner-tile classify status D2H");
      if (flags) return map_device_status(flags);
    }

    if (!vector_allocation_admitted<std::uint64_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        tile_edges(rectangle_candidates);
    const auto tile_edge_end = thrust::copy_if(
        thrust::device, candidate_pairs.begin(),
        candidate_pairs.end(), tile_edge_flags.begin(),
        tile_edges.begin(),
        IsSet());
    tile_edges.resize(static_cast<std::size_t>(
        tile_edge_end - tile_edges.begin()));
    release_device_vector(&tile_edge_flags);
    if (!vector_allocation_admitted<std::uint32_t>(
            config, total_rectangle_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint32_t>
        tile_parents(total_rectangle_count);
    thrust::sequence(
        thrust::device, tile_parents.begin(),
        tile_parents.end(), 0u);
    std::uint32_t tile_iterations = 0;
    Status dsu_status = run_min_dsu(
        tile_edges, &tile_parents, total_rectangle_count,
        config.limits.max_dsu_iterations, &device_status,
        &tile_iterations);
    if (dsu_status != Status::success) return dsu_status;
    validate_labels_kernel<<<
        launch_blocks(total_rectangle_count), kThreads>>>(
        thrust::raw_pointer_cast(tile_parents.data()),
        total_rectangle_count,
        thrust::raw_pointer_cast(device_status.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity tile-label validation launch");
    if (!vector_allocation_admitted<std::uint32_t>(
            config, total_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint32_t>
        first_rectangle(total_count, UINT32_MAX);
    first_rectangle_by_owner_kernel<<<
        launch_blocks(total_rectangle_count), kThreads>>>(
        thrust::raw_pointer_cast(work_rectangles.data()),
        total_rectangle_count,
        thrust::raw_pointer_cast(first_rectangle.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity owner-first launch");
    validate_owner_tiles_kernel<<<
        launch_blocks(rectangle_count), kThreads>>>(
        thrust::raw_pointer_cast(work_rectangles.data()),
        previous_rectangle_count, total_rectangle_count,
        thrust::raw_pointer_cast(tile_parents.data()),
        thrust::raw_pointer_cast(first_rectangle.data()),
        thrust::raw_pointer_cast(device_status.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity owner-tile validation launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "antenna connectivity owner-tile validation synchronize");
    flags = read_device_status(
        device_status,
        "antenna connectivity owner-tile status D2H");
    if (flags) return map_device_status(flags);
    release_device_vector(&tile_edges);
    release_device_vector(&tile_parents);
    release_device_vector(&first_rectangle);

    if (!vector_allocation_admitted<std::uint64_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        owner_candidate_keys(rectangle_candidates);
    if (!vector_allocation_admitted<std::uint8_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint8_t>
        owner_candidate_flags(rectangle_candidates, 0);
    if (!vector_allocation_admitted<std::uint64_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        owner_edge_keys(rectangle_candidates);
    if (!vector_allocation_admitted<std::uint8_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint8_t>
        owner_edge_flags(rectangle_candidates, 0);
    if (rectangle_candidates) {
      classify_owner_pairs_kernel<<<
          launch_blocks(rectangle_candidates), kThreads>>>(
          thrust::raw_pointer_cast(candidate_pairs.data()),
          rectangle_candidates,
          thrust::raw_pointer_cast(work_rectangles.data()),
          thrust::raw_pointer_cast(owner_candidate_keys.data()),
          thrust::raw_pointer_cast(owner_candidate_flags.data()),
          thrust::raw_pointer_cast(owner_edge_keys.data()),
          thrust::raw_pointer_cast(owner_edge_flags.data()));
      cuda_require(
          cudaGetLastError(),
          "antenna connectivity owner-pair classify launch");
      cuda_require(
          cudaDeviceSynchronize(),
          "antenna connectivity owner-pair classify synchronize");
    }
    release_device_vector(&candidate_pairs);

    if (!vector_allocation_admitted<std::uint64_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        owner_candidates(rectangle_candidates);
    const auto owner_candidate_end = thrust::copy_if(
        thrust::device, owner_candidate_keys.begin(),
        owner_candidate_keys.end(), owner_candidate_flags.begin(),
        owner_candidates.begin(), IsSet());
    owner_candidates.resize(static_cast<std::size_t>(
        owner_candidate_end - owner_candidates.begin()));
    release_device_vector(&owner_candidate_keys);
    release_device_vector(&owner_candidate_flags);
    if (!owner_candidates.empty()) {
      if (!sort_scratch_admitted<std::uint64_t>(
              config, owner_candidates.size())) {
        return Status::capacity_exceeded;
      }
      thrust::sort(
          thrust::device, owner_candidates.begin(),
          owner_candidates.end());
      const auto end = thrust::unique(
          thrust::device, owner_candidates.begin(),
          owner_candidates.end());
      owner_candidates.resize(static_cast<std::size_t>(
          end - owner_candidates.begin()));
    }
    unique_candidates = owner_candidates.size();
    if (unique_candidates >
        config.limits.max_unique_candidates) {
      return Status::capacity_exceeded;
    }

    if (!vector_allocation_admitted<std::uint64_t>(
            config, rectangle_candidates)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        edges(rectangle_candidates);
    const auto edge_end = thrust::copy_if(
        thrust::device, owner_edge_keys.begin(),
        owner_edge_keys.end(), owner_edge_flags.begin(),
        edges.begin(), IsSet());
    edges.resize(
        static_cast<std::size_t>(edge_end - edges.begin()));
    release_device_vector(&owner_edge_keys);
    release_device_vector(&owner_edge_flags);
    if (!edges.empty()) {
      if (!sort_scratch_admitted<std::uint64_t>(
              config, edges.size())) {
        return Status::capacity_exceeded;
      }
      thrust::sort(thrust::device, edges.begin(), edges.end());
      const auto end = thrust::unique(
          thrust::device, edges.begin(), edges.end());
      edges.resize(
          static_cast<std::size_t>(end - edges.begin()));
    }
    edge_count = edges.size();
    candidate_census.assign(kRelationSlots, 0);
    edge_census.assign(kRelationSlots, 0);
    if (unique_candidates) {
      census_owner_pairs_kernel<<<
          launch_blocks(unique_candidates), kThreads>>>(
          thrust::raw_pointer_cast(owner_candidates.data()),
          unique_candidates,
          thrust::raw_pointer_cast(work_owner_domains.data()),
          thrust::raw_pointer_cast(candidate_census.data()));
      cuda_require(
          cudaGetLastError(),
          "antenna connectivity candidate-census launch");
    }
    if (edge_count) {
      census_owner_pairs_kernel<<<
          launch_blocks(edge_count), kThreads>>>(
          thrust::raw_pointer_cast(edges.data()), edge_count,
          thrust::raw_pointer_cast(work_owner_domains.data()),
          thrust::raw_pointer_cast(edge_census.data()));
      cuda_require(
          cudaGetLastError(),
          "antenna connectivity edge-census launch");
    }
    cuda_require(
        cudaDeviceSynchronize(),
        "antenna connectivity relation-census synchronize");
    release_device_vector(&owner_candidates);

    dsu_status = run_min_dsu(
        edges, &work_parents, total_count,
        config.limits.max_dsu_iterations, &device_status,
        &iterations);
    if (dsu_status != Status::success) return dsu_status;
    release_device_vector(&edges);
    validate_labels_kernel<<<
        launch_blocks(total_count), kThreads>>>(
        thrust::raw_pointer_cast(work_parents.data()),
        total_count,
        thrust::raw_pointer_cast(device_status.data()));
    cuda_require(
        cudaGetLastError(),
        "antenna connectivity label-validation launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "antenna connectivity label-validation synchronize");
    flags = read_device_status(
        device_status,
        "antenna connectivity label-validation status D2H");
    if (flags) return map_device_status(flags);
    }

    std::vector<std::uint32_t> host_labels;
    if (labels) {
      host_labels.resize(total_count);
      cuda_require(
          cudaMemcpy(
              host_labels.data(),
              thrust::raw_pointer_cast(work_parents.data()),
              total_count * sizeof(std::uint32_t),
              cudaMemcpyDeviceToHost),
          "antenna connectivity labels D2H");
    }

    std::array<unsigned long long, kRelationSlots>
        host_candidate_census{};
    std::array<unsigned long long, kRelationSlots>
        host_edge_census{};
    cuda_require(
        cudaMemcpy(
            host_candidate_census.data(),
            thrust::raw_pointer_cast(candidate_census.data()),
            host_candidate_census.size() *
                sizeof(host_candidate_census[0]),
            cudaMemcpyDeviceToHost),
        "antenna connectivity candidate census D2H");
    cuda_require(
        cudaMemcpy(
            host_edge_census.data(),
            thrust::raw_pointer_cast(edge_census.data()),
            host_edge_census.size() *
                sizeof(host_edge_census[0]),
            cudaMemcpyDeviceToHost),
        "antenna connectivity edge census D2H");

    const std::uint64_t new_closed_domains =
        m_impl->closed_domains | close_domain_mask;
    std::uint64_t released_domains = 0;
    for (std::uint32_t domain = 0;
         domain < config.domain_count; ++domain) {
      const std::uint64_t bit = UINT64_C(1) << domain;
      if ((new_closed_domains & bit) &&
          !(config.relation_rows[domain] &
            ~new_closed_domains)) {
        released_domains |= bit;
      }
    }
    thrust::device_vector<RectI64> retained_rectangles;
    std::uint64_t retained_rectangle_count = 0;
    if (mode == AppendMode::consuming) {
      const auto retained_end = thrust::remove_if(
          thrust::device, work_rectangles.begin(),
          work_rectangles.end(),
          IsReleasedDomain{released_domains});
      work_rectangles.resize(static_cast<std::size_t>(
          retained_end - work_rectangles.begin()));
      retained_rectangle_count = work_rectangles.size();
    } else {
      retained_rectangles.resize(total_rectangle_count);
      const auto retained_end = thrust::copy_if(
          thrust::device, work_rectangles.begin(),
          work_rectangles.end(), retained_rectangles.begin(),
          KeepUnreleasedDomain{released_domains});
      retained_rectangles.resize(static_cast<std::size_t>(
          retained_end - retained_rectangles.begin()));
      retained_rectangle_count = retained_rectangles.size();
    }

    StageCensus host_census;
    host_census.previous_nodes = previous_count;
    host_census.appended_nodes = new_node_count;
    host_census.total_nodes = total_count;
    host_census.previous_rectangles =
        previous_rectangle_count;
    host_census.appended_rectangles = rectangle_count;
    host_census.total_rectangles = total_rectangle_count;
    host_census.retained_rectangles = retained_rectangle_count;
    host_census.retained_rectangle_capacity =
        mode == AppendMode::consuming
            ? work_rectangles.capacity()
            : retained_rectangles.capacity();
    host_census.released_rectangles =
        total_rectangle_count - retained_rectangle_count;
    host_census.closed_domain_mask = new_closed_domains;
    host_census.memberships = membership_total;
    host_census.occupied_cells = group_count;
    host_census.pair_occurrences = pair_occurrences;
    host_census.unique_candidates = unique_candidates;
    host_census.exact_edges = edge_count;
    host_census.dsu_iterations = iterations;
    for (std::size_t index = 0; index < kRelationSlots;
         ++index) {
      host_census.candidates_by_relation[index] =
          host_candidate_census[index];
      host_census.edges_by_relation[index] =
          host_edge_census[index];
    }

    // The only mutations of committed state and caller-visible outputs.
    if (mode == AppendMode::consuming) {
      m_impl->rectangles.swap(work_rectangles);
    } else {
      m_impl->rectangles.swap(retained_rectangles);
    }
    m_impl->owner_domains.swap(work_owner_domains);
    m_impl->parents.swap(work_parents);
    m_impl->owner_count = total_count;
    m_impl->closed_domains = new_closed_domains;
    m_impl->poisoned = false;
    if (mode == AppendMode::transactional) ++m_impl->epoch;
    if (labels) labels->swap(host_labels);
    *census = host_census;
    return Status::success;
  } catch (const CudaFailure &) {
    return Status::cuda_error;
  } catch (const thrust::system_error &) {
    return Status::cuda_error;
  } catch (const std::bad_alloc &) {
    return Status::host_error;
  } catch (...) {
    return Status::host_error;
  }
}

Status Connectivity::snapshot_labels(
    std::vector<std::uint32_t> *labels) const noexcept
{
  if (!m_impl || !labels) return Status::malformed_input;
  if (m_impl->poisoned) return Status::poisoned_state;
  if (m_impl->config_status != Status::success) {
    return m_impl->config_status;
  }
  try {
    cuda_require(
        cudaSetDevice(m_impl->config.device),
        "antenna connectivity snapshot cudaSetDevice");
    std::vector<std::uint32_t> host_labels(
        m_impl->parents.size());
    if (!host_labels.empty()) {
      cuda_require(
          cudaMemcpy(
              host_labels.data(),
              thrust::raw_pointer_cast(m_impl->parents.data()),
              host_labels.size() * sizeof(std::uint32_t),
              cudaMemcpyDeviceToHost),
          "antenna connectivity snapshot D2H");
    }
    labels->swap(host_labels);
    return Status::success;
  } catch (const CudaFailure &) {
    return Status::cuda_error;
  } catch (const thrust::system_error &) {
    return Status::cuda_error;
  } catch (const std::bad_alloc &) {
    return Status::host_error;
  } catch (...) {
    return Status::host_error;
  }
}

Status Connectivity::device_label_view(
    DeviceLabelView *view) const noexcept
{
  if (!m_impl || !view) return Status::malformed_input;
  if (m_impl->poisoned) return Status::poisoned_state;
  if (m_impl->config_status != Status::success) {
    return m_impl->config_status;
  }
  DeviceLabelView result;
  result.labels = m_impl->parents.empty()
                      ? nullptr
                      : thrust::raw_pointer_cast(
                            m_impl->parents.data());
  result.count = m_impl->owner_count;
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
  case Status::malformed_input:
    return "malformed_input";
  case Status::capacity_exceeded:
    return "capacity_exceeded";
  case Status::convergence_failure:
    return "convergence_failure";
  case Status::poisoned_state:
    return "poisoned_state";
  case Status::cuda_error:
    return "cuda_error";
  case Status::host_error:
    return "host_error";
  }
  return "unknown";
}

}  // namespace antenna_connectivity
}  // namespace klayout_cuda
