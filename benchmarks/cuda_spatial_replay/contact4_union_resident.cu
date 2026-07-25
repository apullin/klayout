/*
 * Exact resident-boundary CONTACT.4 qualification consumer.
 */

#include "contact4_union_resident.cuh"
#include "active3_exact_predicate.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

namespace c4 =
    klayout_cuda::contact4_union_resident;
namespace a3 = klayout_cuda::active3;
namespace mu = klayout_cuda::manhattan_union;

using Clock = std::chrono::steady_clock;

constexpr std::int64_t kCoordinateLimit = INT64_C(1000000000000);
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaximumBlocks = 65535;

enum DeviceFlag : std::uint32_t
{
  kInvalidContact = 1u << 0,
  kInvalidBoundary = 1u << 1,
  kCoordinateOverflow = 1u << 2,
  kPerEdgeCapacity = 1u << 3,
  kMembershipOverflow = 1u << 4,
  kPairCapacity = 1u << 5,
  kBoundaryCellCapacity = 1u << 6,
  kMemberVisitCapacity = 1u << 7,
};

struct Grid
{
  std::int64_t base_x;
  std::int64_t base_y;
  std::int64_t cell_size;
  std::int64_t distance;
  std::uint32_t width;
  std::uint32_t height;
};

struct Counters
{
  unsigned long long member_visits;
  unsigned long long candidate_pairs;
  unsigned long long hits;
  unsigned long long uncertain;
};

struct ValidatedHostRequest
{
  Grid grid;
  std::uint64_t memberships;
  c4::ContactBounds bounds;
};

double elapsed_ms(
    const Clock::time_point &begin,
    const Clock::time_point &end)
{
  return std::chrono::duration<double, std::milli>(
      end - begin).count();
}

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::uint64_t sample_memory(c4::Result *result)
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_require(
      cudaMemGetInfo(&free_bytes, &total_bytes),
      "CONTACT.4 resident cudaMemGetInfo");
  if (!result->device_total_bytes) {
    result->device_total_bytes = total_bytes;
    result->callback_free_begin_bytes = free_bytes;
    result->callback_free_low_bytes = free_bytes;
  } else {
    result->callback_free_low_bytes =
        std::min<std::uint64_t>(
            result->callback_free_low_bytes, free_bytes);
  }
  return free_bytes;
}

bool coordinate_qualified(std::int64_t value)
{
  return value >= -kCoordinateLimit && value <= kCoordinateLimit;
}

bool host_edge_valid(const a3::DirectedEdge &edge)
{
  return coordinate_qualified(edge.x1) &&
         coordinate_qualified(edge.y1) &&
         coordinate_qualified(edge.x2) &&
         coordinate_qualified(edge.y2) &&
         ! (edge.x1 == edge.x2 && edge.y1 == edge.y2) &&
         (edge.x1 == edge.x2 || edge.y1 == edge.y2);
}

std::int64_t floor_div_host(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 0;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          (count + kThreads - 1) / kThreads, kMaximumBlocks));
}

void require_contact_contours(const c4::Request &request)
{
  if (request.contact_direction_contract !=
      c4::ContactDirectionContract::
          validated_material_on_right_contours) {
    throw std::runtime_error(
        "resident CONTACT.4 direction contract declined");
  }

  std::uint64_t contour_begin = 0;
  while (contour_begin < request.contact_edge_count) {
    const a3::DirectedEdge &first =
        request.contact_edges[contour_begin];
    const std::int64_t start_x = first.x1;
    const std::int64_t start_y = first.y1;
    std::int64_t previous_x = start_x;
    std::int64_t previous_y = start_y;
    __int128 twice_area = 0;
    bool closed = false;
    std::uint64_t cursor = contour_begin;
    for (; cursor < request.contact_edge_count; ++cursor) {
      const a3::DirectedEdge &edge = request.contact_edges[cursor];
      if (!host_edge_valid(edge) ||
          edge.x1 != previous_x || edge.y1 != previous_y) {
        throw std::runtime_error(
            "resident CONTACT.4 contour is open or discontinuous");
      }
      twice_area +=
          static_cast<__int128>(edge.x1) * edge.y2 -
          static_cast<__int128>(edge.x2) * edge.y1;
      previous_x = edge.x2;
      previous_y = edge.y2;
      if (previous_x == start_x && previous_y == start_y) {
        ++cursor;
        closed = true;
        break;
      }
    }
    if (!closed || cursor - contour_begin < 4) {
      throw std::runtime_error(
          "resident CONTACT.4 contour is not a closed Manhattan ring");
    }
    if (twice_area >= 0) {
      throw std::runtime_error(
          "resident CONTACT.4 contour is not material-on-right");
    }
    contour_begin = cursor;
  }
}

__device__ bool add_checked(
    std::int64_t first, std::int64_t second,
    std::int64_t *result)
{
  if ((second > 0 && first > INT64_MAX - second) ||
      (second < 0 && first < INT64_MIN - second)) {
    return false;
  }
  *result = first + second;
  return true;
}

__device__ std::int64_t floor_div_device(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

__device__ bool edge_grid_span(
    const a3::DirectedEdge &edge, const Grid &grid,
    std::int64_t expansion, std::int64_t *x0,
    std::int64_t *y0, std::int64_t *x1,
    std::int64_t *y1)
{
  std::int64_t low_x = min(edge.x1, edge.x2);
  std::int64_t high_x = max(edge.x1, edge.x2);
  std::int64_t low_y = min(edge.y1, edge.y2);
  std::int64_t high_y = max(edge.y1, edge.y2);
  if (expansion &&
      (!add_checked(low_x, -expansion, &low_x) ||
       !add_checked(high_x, expansion, &high_x) ||
       !add_checked(low_y, -expansion, &low_y) ||
       !add_checked(high_y, expansion, &high_y))) {
    return false;
  }
  *x0 = floor_div_device(low_x, grid.cell_size);
  *x1 = floor_div_device(high_x, grid.cell_size);
  *y0 = floor_div_device(low_y, grid.cell_size);
  *y1 = floor_div_device(high_y, grid.cell_size);
  return true;
}

__device__ bool device_contact_valid(
    const a3::DirectedEdge &edge,
    const c4::ContactBounds &bounds)
{
  const bool coordinates_qualified =
      edge.x1 >= -kCoordinateLimit && edge.x1 <= kCoordinateLimit &&
      edge.y1 >= -kCoordinateLimit && edge.y1 <= kCoordinateLimit &&
      edge.x2 >= -kCoordinateLimit && edge.x2 <= kCoordinateLimit &&
      edge.y2 >= -kCoordinateLimit && edge.y2 <= kCoordinateLimit;
  const bool manhattan_nonzero =
      ! (edge.x1 == edge.x2 && edge.y1 == edge.y2) &&
      (edge.x1 == edge.x2 || edge.y1 == edge.y2);
  const bool inside_bounds =
      edge.x1 >= bounds.left && edge.x1 <= bounds.right &&
      edge.x2 >= bounds.left && edge.x2 <= bounds.right &&
      edge.y1 >= bounds.bottom && edge.y1 <= bounds.top &&
      edge.y2 >= bounds.bottom && edge.y2 <= bounds.top;
  return coordinates_qualified && manhattan_nonzero && inside_bounds;
}

__device__ bool clip_span(
    const Grid &grid, std::int64_t *x0,
    std::int64_t *y0, std::int64_t *x1,
    std::int64_t *y1)
{
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  if (*x1 < grid.base_x || *x0 > maximum_x ||
      *y1 < grid.base_y || *y0 > maximum_y) {
    return false;
  }
  *x0 = max(*x0, grid.base_x);
  *x1 = min(*x1, maximum_x);
  *y0 = max(*y0, grid.base_y);
  *y1 = min(*y1, maximum_y);
  return true;
}

__device__ bool span_inside_grid(
    const Grid &grid, std::int64_t x0,
    std::int64_t y0, std::int64_t x1,
    std::int64_t y1)
{
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  return x0 >= grid.base_x && y0 >= grid.base_y &&
         x1 <= maximum_x && y1 <= maximum_y;
}

__device__ std::uint64_t grid_index(
    const Grid &grid, std::int64_t x, std::int64_t y)
{
  return static_cast<std::uint64_t>(y - grid.base_y) *
             grid.width +
         static_cast<std::uint64_t>(x - grid.base_x);
}

__device__ bool boundary_to_edge(
    const mu::DirectedSegmentI64 &segment,
    a3::DirectedEdge *edge)
{
  if (segment.lo >= segment.hi ||
      (segment.side != -1 && segment.side != 1)) {
    return false;
  }
  if (segment.axis == mu::SegmentAxis::horizontal) {
    if (segment.side < 0) {
      *edge = {segment.hi, segment.fixed,
               segment.lo, segment.fixed};
    } else {
      *edge = {segment.lo, segment.fixed,
               segment.hi, segment.fixed};
    }
    return true;
  }
  if (segment.axis == mu::SegmentAxis::vertical) {
    if (segment.side < 0) {
      *edge = {segment.fixed, segment.lo,
               segment.fixed, segment.hi};
    } else {
      *edge = {segment.fixed, segment.hi,
               segment.fixed, segment.lo};
    }
    return true;
  }
  return false;
}

__global__ void count_contact_memberships_kernel(
    const a3::DirectedEdge *contacts, std::uint32_t contact_count,
    c4::ContactBounds bounds, Grid grid,
    std::uint32_t max_cells_per_edge,
    std::uint32_t *counts, unsigned long long *total,
    std::uint32_t *status)
{
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       id < contact_count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const a3::DirectedEdge edge = contacts[id];
    if (!device_contact_valid(edge, bounds)) {
      atomicOr(status, std::uint32_t(kInvalidContact));
      continue;
    }
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!edge_grid_span(edge, grid, 0, &x0, &y0, &x1, &y1) ||
        !span_inside_grid(grid, x0, y0, x1, y1)) {
      atomicOr(status, std::uint32_t(kInvalidContact));
      continue;
    }
    const std::uint64_t width =
        static_cast<std::uint64_t>(x1 - x0) + 1;
    const std::uint64_t height =
        static_cast<std::uint64_t>(y1 - y0) + 1;
    if (!width || !height ||
        width > max_cells_per_edge ||
        height > max_cells_per_edge ||
        width > max_cells_per_edge / height) {
      atomicOr(status, std::uint32_t(kPerEdgeCapacity));
      continue;
    }
    const std::uint64_t count = width * height;
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t previous = atomicAdd(counts + cell, 1u);
        if (previous == UINT32_MAX) {
          atomicOr(status, std::uint32_t(kMembershipOverflow));
        }
      }
    }
    const unsigned long long previous =
        atomicAdd(total, static_cast<unsigned long long>(count));
    if (previous > ULLONG_MAX - count) {
      atomicOr(status, std::uint32_t(kMembershipOverflow));
    }
  }
}

__global__ void fill_contact_memberships_kernel(
    const a3::DirectedEdge *contacts, std::uint32_t contact_count,
    Grid grid, unsigned long long *cursors,
    std::uint32_t *members,
    std::uint64_t member_capacity, std::uint32_t *status)
{
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       id < contact_count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!edge_grid_span(
          contacts[id], grid, 0, &x0, &y0, &x1, &y1) ||
        !span_inside_grid(grid, x0, y0, x1, y1)) {
      atomicOr(status, std::uint32_t(kInvalidContact));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const unsigned long long position =
            atomicAdd(cursors + cell, 1ULL);
        if (position >= member_capacity) {
          atomicOr(status, std::uint32_t(kMembershipOverflow));
        } else {
          members[position] = static_cast<std::uint32_t>(id);
        }
      }
    }
  }
}

__global__ void validate_grid_kernel(
    const std::uint32_t *counts,
    const std::uint64_t *offsets,
    const unsigned long long *cursors,
    std::uint64_t cell_count, std::uint32_t *status)
{
  for (std::uint64_t cell =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < cell_count;
       cell += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (cursors[cell] != offsets[cell] + counts[cell]) {
      atomicOr(status, std::uint32_t(kMembershipOverflow));
    }
  }
}

__global__ void preflight_boundary_kernel(
    const mu::DirectedSegmentI64 *boundary,
    std::uint64_t boundary_count, Grid grid,
    std::uint32_t max_cells_per_edge,
    unsigned long long *total_cells, std::uint32_t *status)
{
  unsigned long long local_cells = 0;
  for (std::uint64_t boundary_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       boundary_id < boundary_count;
       boundary_id +=
           static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    a3::DirectedEdge active{};
    if (!boundary_to_edge(boundary[boundary_id], &active)) {
      atomicOr(status, std::uint32_t(kInvalidBoundary));
      continue;
    }
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!edge_grid_span(
          active, grid, grid.distance, &x0, &y0, &x1, &y1)) {
      atomicOr(status, std::uint32_t(kCoordinateOverflow));
      continue;
    }
    if (!clip_span(grid, &x0, &y0, &x1, &y1)) continue;
    const std::uint64_t width =
        static_cast<std::uint64_t>(x1 - x0) + 1;
    const std::uint64_t height =
        static_cast<std::uint64_t>(y1 - y0) + 1;
    if (!width || !height ||
        width > max_cells_per_edge ||
        height > max_cells_per_edge ||
        width > max_cells_per_edge / height) {
      atomicOr(status, std::uint32_t(kBoundaryCellCapacity));
      continue;
    }
    const unsigned long long cells =
        static_cast<unsigned long long>(width * height);
    local_cells += cells;
  }
  if (local_cells) {
    atomicAdd(total_cells, local_cells);
  }
}

__global__ void preflight_member_visits_kernel(
    const mu::DirectedSegmentI64 *boundary,
    std::uint64_t boundary_count, Grid grid,
    const std::uint32_t *counts,
    unsigned long long *total_member_visits,
    std::uint32_t *status)
{
  unsigned long long local_visits = 0;
  for (std::uint64_t boundary_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       boundary_id < boundary_count;
       boundary_id +=
           static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    a3::DirectedEdge active{};
    if (!boundary_to_edge(boundary[boundary_id], &active)) {
      atomicOr(status, std::uint32_t(kInvalidBoundary));
      continue;
    }
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!edge_grid_span(
          active, grid, grid.distance, &x0, &y0, &x1, &y1)) {
      atomicOr(status, std::uint32_t(kCoordinateOverflow));
      continue;
    }
    if (!clip_span(grid, &x0, &y0, &x1, &y1)) continue;
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        local_visits += counts[grid_index(grid, x, y)];
      }
    }
  }
  if (local_visits) {
    atomicAdd(total_member_visits, local_visits);
  }
}

__global__ void query_boundary_kernel(
    const mu::DirectedSegmentI64 *boundary,
    std::uint64_t boundary_count,
    const a3::DirectedEdge *contacts,
    std::uint32_t contact_count, Grid grid,
    const std::uint32_t *counts,
    const std::uint64_t *offsets,
    const std::uint32_t *members,
    std::uint64_t max_pair_work, Counters *counters,
    std::uint32_t *status)
{
  unsigned long long local_candidates = 0;
  unsigned long long local_hits = 0;
  unsigned long long local_uncertain = 0;
  for (std::uint64_t boundary_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       boundary_id < boundary_count;
       boundary_id +=
           static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    a3::DirectedEdge active{};
    if (!boundary_to_edge(boundary[boundary_id], &active)) {
      atomicOr(status, std::uint32_t(kInvalidBoundary));
      continue;
    }
    std::int64_t active_x0 = 0;
    std::int64_t active_y0 = 0;
    std::int64_t active_x1 = 0;
    std::int64_t active_y1 = 0;
    if (!edge_grid_span(
          active, grid, grid.distance, &active_x0, &active_y0,
          &active_x1, &active_y1)) {
      atomicOr(status, std::uint32_t(kCoordinateOverflow));
      continue;
    }
    if (!clip_span(
          grid, &active_x0, &active_y0,
          &active_x1, &active_y1)) {
      continue;
    }
    for (std::int64_t y = active_y0; y <= active_y1; ++y) {
      for (std::int64_t x = active_x0; x <= active_x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint64_t begin = offsets[cell];
        const std::uint64_t end = begin + counts[cell];
        if (end < begin) {
          atomicOr(status, std::uint32_t(kMembershipOverflow));
          continue;
        }
        for (std::uint64_t position = begin;
             position < end; ++position) {
          const std::uint32_t contact_id = members[position];
          if (contact_id >= contact_count) {
            atomicOr(status, std::uint32_t(kInvalidContact));
            continue;
          }
          const a3::DirectedEdge contact = contacts[contact_id];
          std::int64_t contact_x0 = 0;
          std::int64_t contact_y0 = 0;
          std::int64_t contact_x1 = 0;
          std::int64_t contact_y1 = 0;
          if (!edge_grid_span(
                contact, grid, 0, &contact_x0, &contact_y0,
                &contact_x1, &contact_y1)) {
            atomicOr(status, std::uint32_t(kInvalidContact));
            continue;
          }
          // Each boundary/contact pair is owned by the lexicographically
          // first grid cell common to their expanded/unexpanded spans.
          if (x != max(active_x0, contact_x0) ||
              y != max(active_y0, contact_y0)) {
            continue;
          }
          ++local_candidates;
          const a3::Verdict verdict =
              a3::classify_pair_bounded(
                  a3::EdgePair{active, contact}, grid.distance);
          if (verdict == a3::Verdict::kViolation) {
            ++local_hits;
          } else if (verdict == a3::Verdict::kUncertain) {
            ++local_uncertain;
          } else if (verdict != a3::Verdict::kNoViolation) {
            atomicOr(status, std::uint32_t(kInvalidBoundary));
          }
        }
      }
    }
  }
  if (local_candidates) {
    const unsigned long long previous =
        atomicAdd(&counters->candidate_pairs, local_candidates);
    if (previous > ULLONG_MAX - local_candidates ||
        previous + local_candidates > max_pair_work) {
      atomicOr(status, std::uint32_t(kPairCapacity));
    }
  }
  if (local_hits) atomicAdd(&counters->hits, local_hits);
  if (local_uncertain) {
    atomicAdd(&counters->uncertain, local_uncertain);
  }
}

void validate_common_request(
    std::uint64_t contact_count,
    std::uint64_t boundary_count,
    std::int64_t distance,
    std::int64_t grid_cell_size,
    int device,
    c4::ContactDirectionContract direction_contract,
    const c4::Limits &limits)
{
  if (direction_contract !=
      c4::ContactDirectionContract::
          validated_material_on_right_contours) {
    throw std::runtime_error(
        "resident CONTACT.4 direction contract declined");
  }
  if (!contact_count ||
      contact_count > limits.max_contact_edges ||
      contact_count > UINT32_MAX ||
      !boundary_count ||
      distance !=
          a3::kContact4QualifiedSceneCoordinateDistance ||
      grid_cell_size <= 0 ||
      device < 0 ||
      !limits.max_grid_cells ||
      !limits.max_memberships ||
      !limits.max_boundary_cell_visits ||
      !limits.max_member_visits ||
      !limits.max_pair_work ||
      !limits.max_cells_per_contact_edge ||
      !limits.max_cells_per_boundary_edge) {
    throw std::runtime_error(
        "invalid resident CONTACT.4 request or capacity");
  }
}

Grid grid_from_bounds(
    const c4::ContactBounds &bounds,
    std::int64_t grid_cell_size,
    std::int64_t distance,
    const c4::Limits &limits)
{
  if (!coordinate_qualified(bounds.left) ||
      !coordinate_qualified(bounds.bottom) ||
      !coordinate_qualified(bounds.right) ||
      !coordinate_qualified(bounds.top) ||
      bounds.left >= bounds.right ||
      bounds.bottom >= bounds.top) {
    throw std::runtime_error(
        "resident CONTACT.4 contact bounds are invalid");
  }
  const std::int64_t base_x =
      floor_div_host(bounds.left, grid_cell_size);
  const std::int64_t base_y =
      floor_div_host(bounds.bottom, grid_cell_size);
  const std::int64_t maximum_x =
      floor_div_host(bounds.right, grid_cell_size);
  const std::int64_t maximum_y =
      floor_div_host(bounds.top, grid_cell_size);
  const __int128 width =
      static_cast<__int128>(maximum_x) - base_x + 1;
  const __int128 height =
      static_cast<__int128>(maximum_y) - base_y + 1;
  if (width <= 0 || height <= 0 ||
      width > UINT32_MAX || height > UINT32_MAX ||
      width * height >
          static_cast<__int128>(limits.max_grid_cells) ||
      width * height > UINT64_MAX) {
    throw std::runtime_error(
        "resident CONTACT.4 grid exceeds capacity");
  }
  return {
      base_x, base_y, grid_cell_size, distance,
      static_cast<std::uint32_t>(width),
      static_cast<std::uint32_t>(height)};
}

ValidatedHostRequest validate_host_request(
    const c4::Request &request,
    std::uint64_t boundary_count)
{
  if (!request.contact_edges) {
    throw std::runtime_error(
        "invalid resident CONTACT.4 request or capacity");
  }
  validate_common_request(
      request.contact_edge_count, boundary_count,
      request.distance, request.grid_cell_size, request.device,
      request.contact_direction_contract, request.limits);
  require_contact_contours(request);

  c4::ContactBounds bounds;
  for (std::uint64_t id = 0;
       id < request.contact_edge_count; ++id) {
    const a3::DirectedEdge &edge = request.contact_edges[id];
    if (!host_edge_valid(edge)) {
      throw std::runtime_error(
          "resident CONTACT.4 has an invalid contact edge");
    }
    const std::int64_t edge_left = std::min(edge.x1, edge.x2);
    const std::int64_t edge_bottom = std::min(edge.y1, edge.y2);
    const std::int64_t edge_right = std::max(edge.x1, edge.x2);
    const std::int64_t edge_top = std::max(edge.y1, edge.y2);
    if (!id) {
      bounds = {
          edge_left, edge_bottom, edge_right, edge_top};
    } else {
      bounds.left = std::min(bounds.left, edge_left);
      bounds.bottom = std::min(bounds.bottom, edge_bottom);
      bounds.right = std::max(bounds.right, edge_right);
      bounds.top = std::max(bounds.top, edge_top);
    }
  }
  const Grid grid = grid_from_bounds(
      bounds, request.grid_cell_size, request.distance,
      request.limits);

  __int128 membership_total = 0;
  for (std::uint64_t id = 0;
       id < request.contact_edge_count; ++id) {
    const a3::DirectedEdge &edge = request.contact_edges[id];
    const std::int64_t x0 = floor_div_host(
        std::min(edge.x1, edge.x2), request.grid_cell_size);
    const std::int64_t x1 = floor_div_host(
        std::max(edge.x1, edge.x2), request.grid_cell_size);
    const std::int64_t y0 = floor_div_host(
        std::min(edge.y1, edge.y2), request.grid_cell_size);
    const std::int64_t y1 = floor_div_host(
        std::max(edge.y1, edge.y2), request.grid_cell_size);
    const __int128 span_width =
        static_cast<__int128>(x1) - x0 + 1;
    const __int128 span_height =
        static_cast<__int128>(y1) - y0 + 1;
    const __int128 edge_memberships = span_width * span_height;
    if (span_width <= 0 || span_height <= 0 ||
        edge_memberships <= 0 ||
        edge_memberships >
            request.limits.max_cells_per_contact_edge) {
      throw std::runtime_error(
          "resident CONTACT.4 membership gate declined: "
          "per-contact-edge cell capacity");
    }
    membership_total += edge_memberships;
    if (membership_total > request.limits.max_memberships ||
        membership_total > UINT64_MAX) {
      throw std::runtime_error(
          "resident CONTACT.4 membership gate declined: "
          "total cell capacity");
    }
  }
  return {
      grid, static_cast<std::uint64_t>(membership_total), bounds};
}

Grid validate_device_request(
    const c4::DeviceRequest &request,
    std::uint64_t boundary_count)
{
  if (!request.contacts.device_edges) {
    throw std::runtime_error(
        "resident CONTACT.4 device view pointer is null");
  }
  validate_common_request(
      request.contacts.count, boundary_count,
      request.distance, request.grid_cell_size, request.device,
      request.contact_direction_contract, request.limits);
  const Grid grid = grid_from_bounds(
      request.contacts.bounds, request.grid_cell_size,
      request.distance, request.limits);

  cudaPointerAttributes attributes{};
  cuda_require(
      cudaPointerGetAttributes(
          &attributes, request.contacts.device_edges),
      "resident CONTACT.4 device view pointer attributes");
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != request.device) {
    throw std::runtime_error(
        "resident CONTACT.4 device view is not on the selected device");
  }
  return grid;
}

c4::Result consume_device_core(
    cudaStream_t stream,
    const mu::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count,
    const c4::DeviceRequest &request,
    const Grid &grid,
    std::uint64_t expected_memberships,
    c4::Result result,
    const Clock::time_point &total_begin)
{
  result.grid_cells =
      static_cast<std::uint64_t>(grid.width) * grid.height;
  if (result.boundary_segments >
          UINT64_MAX /
              request.limits.max_cells_per_boundary_edge ||
      request.limits.max_boundary_cell_visits >
          UINT64_MAX / request.contacts.count) {
    throw std::runtime_error(
        "resident CONTACT.4 traversal census can overflow");
  }

  thrust::device_vector<std::uint32_t> status(1, 0);
  thrust::device_vector<unsigned long long> membership_total(1, 0);
  thrust::device_vector<Counters> counters(1);
  unsigned long long *const device_member_visits =
      reinterpret_cast<unsigned long long *>(
          thrust::raw_pointer_cast(counters.data()));
  cuda_require(
      cudaMemsetAsync(
          thrust::raw_pointer_cast(counters.data()), 0,
          sizeof(Counters), stream),
      "resident CONTACT.4 counters clear");
  thrust::device_vector<std::uint32_t> cell_counts(
      result.grid_cells, 0);
  thrust::device_vector<std::uint64_t> cell_offsets(
      result.grid_cells);
  thrust::device_vector<unsigned long long> cell_cursors(
      result.grid_cells);

  const Clock::time_point count_begin = Clock::now();
  const std::uint32_t contact_blocks =
      launch_blocks(request.contacts.count);
  count_contact_memberships_kernel<<<
      contact_blocks, kThreads, 0, stream>>>(
      request.contacts.device_edges,
      static_cast<std::uint32_t>(request.contacts.count),
      request.contacts.bounds, grid,
      request.limits.max_cells_per_contact_edge,
      thrust::raw_pointer_cast(cell_counts.data()),
      thrust::raw_pointer_cast(membership_total.data()),
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(),
      "resident CONTACT.4 membership count launch");
  cuda_require(
      cudaStreamSynchronize(stream),
      "resident CONTACT.4 membership count synchronize");
  unsigned long long host_memberships = 0;
  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(
          &host_memberships,
          thrust::raw_pointer_cast(membership_total.data()),
          sizeof(host_memberships), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 membership total D2H");
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 count status D2H");
  result.device_flags = host_status;
  result.grid_count_ms = elapsed_ms(count_begin, Clock::now());
  if (host_status) {
    throw std::runtime_error(
        "resident CONTACT.4 device contact gate declined");
  }
  if (host_memberships > request.limits.max_memberships) {
    throw std::runtime_error(
        "resident CONTACT.4 membership gate declined: "
        "total cell capacity");
  }
  if (expected_memberships &&
      host_memberships != expected_memberships) {
    throw std::runtime_error(
        "resident CONTACT.4 membership census mismatch");
  }
  result.memberships = host_memberships;

  const Clock::time_point build_begin = Clock::now();
  const auto policy = thrust::cuda::par.on(stream);
  thrust::exclusive_scan(
      policy, cell_counts.begin(), cell_counts.end(),
      cell_offsets.begin(), std::uint64_t{0});
  cuda_require(
      cudaStreamSynchronize(stream),
      "resident CONTACT.4 offset scan synchronize");
  result.post_scan_free_bytes = sample_memory(&result);
  cuda_require(
      cudaMemcpyAsync(
          thrust::raw_pointer_cast(cell_cursors.data()),
          thrust::raw_pointer_cast(cell_offsets.data()),
          result.grid_cells * sizeof(std::uint64_t),
          cudaMemcpyDeviceToDevice, stream),
      "resident CONTACT.4 offsets-to-cursors D2D");
  thrust::device_vector<std::uint32_t> members(
      result.memberships);
  fill_contact_memberships_kernel<<<
      contact_blocks, kThreads, 0, stream>>>(
      request.contacts.device_edges,
      static_cast<std::uint32_t>(request.contacts.count),
      grid,
      thrust::raw_pointer_cast(cell_cursors.data()),
      thrust::raw_pointer_cast(members.data()), result.memberships,
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(),
      "resident CONTACT.4 membership fill launch");
  const std::uint32_t grid_blocks =
      launch_blocks(result.grid_cells);
  validate_grid_kernel<<<grid_blocks, kThreads, 0, stream>>>(
      thrust::raw_pointer_cast(cell_counts.data()),
      thrust::raw_pointer_cast(cell_offsets.data()),
      thrust::raw_pointer_cast(cell_cursors.data()),
      result.grid_cells,
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(),
      "resident CONTACT.4 grid validation launch");
  cuda_require(
      cudaStreamSynchronize(stream),
      "resident CONTACT.4 grid build synchronize");
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 grid status D2H");
  result.device_flags = host_status;
  sample_memory(&result);
  result.grid_build_ms = elapsed_ms(build_begin, Clock::now());
  if (host_status) {
    throw std::runtime_error(
        "resident CONTACT.4 grid invariant declined");
  }

  const Clock::time_point boundary_preflight_begin = Clock::now();
  thrust::device_vector<unsigned long long>
      boundary_cell_total(1, 0);
  if (horizontal_count) {
    preflight_boundary_kernel<<<
        launch_blocks(horizontal_count), kThreads, 0, stream>>>(
        horizontal, horizontal_count, grid,
        request.limits.max_cells_per_boundary_edge,
        thrust::raw_pointer_cast(boundary_cell_total.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "resident CONTACT.4 horizontal boundary preflight launch");
  }
  if (vertical_count) {
    preflight_boundary_kernel<<<
        launch_blocks(vertical_count), kThreads, 0, stream>>>(
        vertical, vertical_count, grid,
        request.limits.max_cells_per_boundary_edge,
        thrust::raw_pointer_cast(boundary_cell_total.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "resident CONTACT.4 vertical boundary preflight launch");
  }
  cuda_require(
      cudaStreamSynchronize(stream),
      "resident CONTACT.4 boundary preflight synchronize");
  unsigned long long host_boundary_cell_visits = 0;
  cuda_require(
      cudaMemcpy(
          &host_boundary_cell_visits,
          thrust::raw_pointer_cast(boundary_cell_total.data()),
          sizeof(host_boundary_cell_visits), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 boundary cell total D2H");
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 boundary preflight status D2H");
  result.boundary_cell_visits = host_boundary_cell_visits;
  result.device_flags = host_status;
  if (host_status ||
      result.boundary_cell_visits >
          request.limits.max_boundary_cell_visits) {
    throw std::runtime_error(
        "resident CONTACT.4 boundary traversal gate declined");
  }

  if (horizontal_count) {
    preflight_member_visits_kernel<<<
        launch_blocks(horizontal_count), kThreads, 0, stream>>>(
        horizontal, horizontal_count, grid,
        thrust::raw_pointer_cast(cell_counts.data()),
        device_member_visits,
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "resident CONTACT.4 horizontal member preflight launch");
  }
  if (vertical_count) {
    preflight_member_visits_kernel<<<
        launch_blocks(vertical_count), kThreads, 0, stream>>>(
        vertical, vertical_count, grid,
        thrust::raw_pointer_cast(cell_counts.data()),
        device_member_visits,
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "resident CONTACT.4 vertical member preflight launch");
  }
  cuda_require(
      cudaStreamSynchronize(stream),
      "resident CONTACT.4 member preflight synchronize");
  unsigned long long host_member_visits = 0;
  cuda_require(
      cudaMemcpy(
          &host_member_visits,
          device_member_visits,
          sizeof(host_member_visits), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 member visit total D2H");
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 member preflight status D2H");
  result.member_visits = host_member_visits;
  result.device_flags = host_status;
  result.boundary_preflight_ms =
      elapsed_ms(boundary_preflight_begin, Clock::now());
  if (host_status ||
      result.member_visits > request.limits.max_member_visits) {
    throw std::runtime_error(
        "resident CONTACT.4 query gate declined: "
        "member-visit capacity");
  }

  const Clock::time_point query_begin = Clock::now();
  if (horizontal_count) {
    query_boundary_kernel<<<
        launch_blocks(horizontal_count), kThreads, 0, stream>>>(
        horizontal, horizontal_count,
        request.contacts.device_edges,
        static_cast<std::uint32_t>(request.contacts.count),
        grid, thrust::raw_pointer_cast(cell_counts.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(members.data()),
        request.limits.max_pair_work,
        thrust::raw_pointer_cast(counters.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "resident CONTACT.4 horizontal query launch");
  }
  if (vertical_count) {
    query_boundary_kernel<<<
        launch_blocks(vertical_count), kThreads, 0, stream>>>(
        vertical, vertical_count,
        request.contacts.device_edges,
        static_cast<std::uint32_t>(request.contacts.count),
        grid, thrust::raw_pointer_cast(cell_counts.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(members.data()),
        request.limits.max_pair_work,
        thrust::raw_pointer_cast(counters.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "resident CONTACT.4 vertical query launch");
  }
  cuda_require(
      cudaStreamSynchronize(stream),
      "resident CONTACT.4 query synchronize");
  sample_memory(&result);
  result.query_ms = elapsed_ms(query_begin, Clock::now());

  const Clock::time_point d2h_begin = Clock::now();
  Counters host_counters{};
  cuda_require(
      cudaMemcpy(
          &host_counters, thrust::raw_pointer_cast(counters.data()),
          sizeof(host_counters), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 counters D2H");
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "resident CONTACT.4 final status D2H");
  result.candidate_pairs = host_counters.candidate_pairs;
  result.hits = host_counters.hits;
  result.uncertain = host_counters.uncertain;
  result.device_flags = host_status;
  result.d2h_ms = elapsed_ms(d2h_begin, Clock::now());
  if (host_status ||
      result.member_visits > request.limits.max_member_visits ||
      result.candidate_pairs > request.limits.max_pair_work) {
    throw std::runtime_error(
        "resident CONTACT.4 query gate declined");
  }
  result.certified_empty =
      result.hits == 0 && result.uncertain == 0;
  result.callback_incremental_peak_bytes =
      result.callback_free_begin_bytes -
      result.callback_free_low_bytes;
  result.total_ms = elapsed_ms(total_begin, Clock::now());
  return result;
}

c4::Result initialize_result(
    cudaStream_t stream, std::uint64_t contact_count,
    std::uint64_t horizontal_count,
    std::uint64_t vertical_count)
{
  if (stream != nullptr) {
    throw std::runtime_error(
        "resident CONTACT.4 v1 requires the default CUDA stream");
  }
  if (horizontal_count >
      std::numeric_limits<std::uint64_t>::max() - vertical_count) {
    throw std::runtime_error(
        "resident CONTACT.4 boundary census overflows");
  }
  c4::Result result;
  result.contact_edges = contact_count;
  result.boundary_segments = horizontal_count + vertical_count;
  return result;
}

void require_current_device(int requested_device)
{
  int current_device = -1;
  cuda_require(
      cudaGetDevice(&current_device),
      "resident CONTACT.4 cudaGetDevice");
  if (current_device != requested_device) {
    throw std::runtime_error(
        "resident CONTACT.4 device contract declined");
  }
}

c4::Result consume_host(
    cudaStream_t stream,
    const mu::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count,
    const c4::Request &request)
{
  const Clock::time_point total_begin = Clock::now();
  c4::Result result = initialize_result(
      stream, request.contact_edge_count,
      horizontal_count, vertical_count);

  const Clock::time_point setup_begin = Clock::now();
  const ValidatedHostRequest validated =
      validate_host_request(request, result.boundary_segments);
  require_current_device(request.device);
  sample_memory(&result);
  result.setup_ms = elapsed_ms(setup_begin, Clock::now());

  const Clock::time_point h2d_begin = Clock::now();
  thrust::device_vector<a3::DirectedEdge> contacts(
      request.contact_edge_count);
  cuda_require(
      cudaMemcpyAsync(
          thrust::raw_pointer_cast(contacts.data()),
          request.contact_edges,
          request.contact_edge_count * sizeof(a3::DirectedEdge),
          cudaMemcpyHostToDevice, stream),
      "resident CONTACT.4 contact H2D");
  cuda_require(
      cudaStreamSynchronize(stream),
      "resident CONTACT.4 contact H2D synchronize");
  sample_memory(&result);
  result.contact_h2d_ms = elapsed_ms(h2d_begin, Clock::now());

  c4::DeviceRequest device_request;
  device_request.contacts = {
      thrust::raw_pointer_cast(contacts.data()),
      request.contact_edge_count, validated.bounds};
  device_request.distance = request.distance;
  device_request.grid_cell_size = request.grid_cell_size;
  device_request.device = request.device;
  device_request.contact_direction_contract =
      request.contact_direction_contract;
  device_request.limits = request.limits;
  return consume_device_core(
      stream, horizontal, horizontal_count, vertical, vertical_count,
      device_request, validated.grid, validated.memberships,
      result, total_begin);
}

c4::Result consume_device(
    cudaStream_t stream,
    const mu::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count,
    const c4::DeviceRequest &request)
{
  const Clock::time_point total_begin = Clock::now();
  c4::Result result = initialize_result(
      stream, request.contacts.count,
      horizontal_count, vertical_count);

  const Clock::time_point setup_begin = Clock::now();
  require_current_device(request.device);
  const Grid grid =
      validate_device_request(request, result.boundary_segments);
  sample_memory(&result);
  result.setup_ms = elapsed_ms(setup_begin, Clock::now());
  return consume_device_core(
      stream, horizontal, horizontal_count, vertical, vertical_count,
      request, grid, 0, result, total_begin);
}

}  // namespace

namespace klayout_cuda {
namespace contact4_union_resident {

void consume_boundary_hook(
    cudaStream_t stream,
    const manhattan_union::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const manhattan_union::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count, void *opaque)
{
  ResidentContext *context =
      static_cast<ResidentContext *>(opaque);
  if (!context || context->invoked) {
    throw std::runtime_error(
        "invalid resident CONTACT.4 callback state");
  }
  context->invoked = true;
  context->result = consume_host(
      stream, horizontal, horizontal_count,
      vertical, vertical_count, context->request);
  if (!context->result.certified_empty) {
    throw std::runtime_error(
        "resident CONTACT.4 nonempty result declined");
  }
}

manhattan_union::ResidentBoundaryHook make_resident_hook(
    ResidentContext *context)
{
  if (!context || context->invoked) {
    throw std::runtime_error(
        "invalid resident CONTACT.4 hook context");
  }
  manhattan_union::ResidentBoundaryHook hook;
  hook.consume = consume_boundary_hook;
  hook.context = context;
  hook.stop_before_d2h = true;
  return hook;
}

void consume_device_boundary_hook(
    cudaStream_t stream,
    const manhattan_union::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const manhattan_union::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count, void *opaque)
{
  DeviceResidentContext *context =
      static_cast<DeviceResidentContext *>(opaque);
  if (!context || context->invoked) {
    throw std::runtime_error(
        "invalid resident CONTACT.4 device callback state");
  }
  context->invoked = true;
  context->result = consume_device(
      stream, horizontal, horizontal_count,
      vertical, vertical_count, context->request);
  if (!context->result.certified_empty) {
    throw std::runtime_error(
        "resident CONTACT.4 nonempty result declined");
  }
}

manhattan_union::ResidentBoundaryHook make_device_resident_hook(
    DeviceResidentContext *context)
{
  if (!context || context->invoked) {
    throw std::runtime_error(
        "invalid resident CONTACT.4 device hook context");
  }
  manhattan_union::ResidentBoundaryHook hook;
  hook.consume = consume_device_boundary_hook;
  hook.context = context;
  hook.stop_before_d2h = true;
  return hook;
}

}  // namespace contact4_union_resident
}  // namespace klayout_cuda
