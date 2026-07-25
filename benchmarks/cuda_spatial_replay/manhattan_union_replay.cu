/*
 * Exact CUDA Manhattan rectangle-union replay.
 *
 * This is a bounded feasibility harness, not a production KLayout hook.  It
 * converts expanded int64 rectangles into exact directed union-boundary
 * segments with no raster grid and no floating point:
 *
 *   x endpoint sort -> rectangle/slab memberships -> y event sort/scan
 *   -> disjoint strip intervals -> vertical XOR scan -> segment compaction
 *
 * Every allocation, H2D/D2H transfer, sort, scan and compaction is included in
 * total_ms.  A capacity or invariant failure returns an explicit fallback;
 * partial boundary output is never published.
 */

#include "manhattan_union_format.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using klayout_cuda::manhattan_union::DirectedSegmentI64;
using klayout_cuda::manhattan_union::RectI64;
using klayout_cuda::manhattan_union::SegmentAxis;
using Clock = std::chrono::steady_clock;

#define MU_HD __host__ __device__

constexpr std::uint32_t kThreads = 256;

enum DeviceStatus : std::uint32_t
{
  kStatusInvalidRectangle = 1u << 0,
  kStatusEndpointLookup = 1u << 1,
  kStatusPerRectangleCapacity = 1u << 2,
  kStatusCoverageInvariant = 1u << 3,
  kStatusTransitionInvariant = 1u << 4,
};

struct Limits
{
  std::uint64_t max_rectangles = UINT64_C(32000000);
  std::uint64_t max_x_slabs = UINT64_C(32000000);
  std::uint64_t max_memberships = UINT64_C(64000000);
  std::uint64_t max_events = UINT64_C(128000000);
  std::uint64_t max_segments = UINT64_C(64000000);
  std::uint32_t max_slabs_per_rectangle = 4096;
};

struct EventKey
{
  std::int64_t y;
  std::uint32_t slab;
  std::uint32_t reserved;
};

struct BoundaryKey
{
  std::int64_t y;
  std::uint32_t boundary;
  std::uint32_t reserved;
};

struct SideDelta
{
  std::int32_t left;
  std::int32_t right;
};

struct Transition
{
  std::int64_t y;
  std::uint32_t slab;
  std::int32_t kind;
};

struct StripInterval
{
  std::int64_t bottom;
  std::int64_t top;
  std::uint32_t slab;
  std::uint32_t reserved;
};

struct SegmentLine
{
  std::int64_t fixed;
  std::int32_t side;
  SegmentAxis axis;
};

static_assert(sizeof(EventKey) == 16, "unexpected event-key padding");
static_assert(sizeof(BoundaryKey) == 16, "unexpected boundary-key padding");
static_assert(sizeof(SideDelta) == 8, "unexpected side-delta padding");
static_assert(sizeof(Transition) == 16, "unexpected transition padding");
static_assert(sizeof(StripInterval) == 24,
              "unexpected strip-interval padding");
static_assert(sizeof(SegmentLine) == 16, "unexpected segment-line padding");

struct EventKeyLess
{
  MU_HD bool operator()(const EventKey &first, const EventKey &second) const
  {
    if (first.slab != second.slab) return first.slab < second.slab;
    return first.y < second.y;
  }
};

struct EventKeyEqual
{
  MU_HD bool operator()(const EventKey &first, const EventKey &second) const
  {
    return first.slab == second.slab && first.y == second.y;
  }
};

struct BoundaryKeyLess
{
  MU_HD bool operator()(const BoundaryKey &first,
                        const BoundaryKey &second) const
  {
    if (first.boundary != second.boundary)
      return first.boundary < second.boundary;
    return first.y < second.y;
  }
};

struct BoundaryKeyEqual
{
  MU_HD bool operator()(const BoundaryKey &first,
                        const BoundaryKey &second) const
  {
    return first.boundary == second.boundary && first.y == second.y;
  }
};

struct EventSlab
{
  MU_HD std::uint32_t operator()(const EventKey &key) const
  {
    return key.slab;
  }
};

struct BoundaryId
{
  MU_HD std::uint32_t operator()(const BoundaryKey &key) const
  {
    return key.boundary;
  }
};

struct SideDeltaPlus
{
  MU_HD SideDelta operator()(const SideDelta &first,
                             const SideDelta &second) const
  {
    return {first.left + second.left, first.right + second.right};
  }
};

struct ZeroEventDelta
{
  template <class Tuple>
  MU_HD bool operator()(const Tuple &entry) const
  {
    return thrust::get<1>(entry) == 0;
  }
};

struct ZeroSideDelta
{
  template <class Tuple>
  MU_HD bool operator()(const Tuple &entry) const
  {
    const SideDelta delta = thrust::get<1>(entry);
    return delta.left == 0 && delta.right == 0;
  }
};

struct InvalidTransition
{
  MU_HD bool operator()(const Transition &transition) const
  {
    return transition.kind == 0;
  }
};

struct InvalidSegment
{
  MU_HD bool operator()(const DirectedSegmentI64 &segment) const
  {
    return segment.axis == SegmentAxis::invalid;
  }
};

MU_HD bool same_segment_line(const DirectedSegmentI64 &first,
                             const DirectedSegmentI64 &second)
{
  return first.axis == second.axis && first.side == second.side &&
         first.fixed == second.fixed;
}

struct SegmentLess
{
  MU_HD bool operator()(const DirectedSegmentI64 &first,
                        const DirectedSegmentI64 &second) const
  {
    const auto first_axis = static_cast<std::uint32_t>(first.axis);
    const auto second_axis = static_cast<std::uint32_t>(second.axis);
    if (first_axis != second_axis) return first_axis < second_axis;
    if (first.side != second.side) return first.side < second.side;
    if (first.fixed != second.fixed) return first.fixed < second.fixed;
    if (first.lo != second.lo) return first.lo < second.lo;
    return first.hi < second.hi;
  }
};

struct SegmentMerge
{
  MU_HD DirectedSegmentI64 operator()(const DirectedSegmentI64 &first,
                                      const DirectedSegmentI64 &second) const
  {
    DirectedSegmentI64 result = first;
    result.lo = min(first.lo, second.lo);
    result.hi = max(first.hi, second.hi);
    return result;
  }
};

struct SegmentLineFromSegment
{
  MU_HD SegmentLine operator()(const DirectedSegmentI64 &segment) const
  {
    return {segment.fixed, segment.side, segment.axis};
  }
};

struct SegmentLineEqual
{
  MU_HD bool operator()(const SegmentLine &first,
                        const SegmentLine &second) const
  {
    return first.fixed == second.fixed && first.side == second.side &&
           first.axis == second.axis;
  }
};

struct SegmentHigh
{
  MU_HD std::int64_t operator()(const DirectedSegmentI64 &segment) const
  {
    return segment.hi;
  }
};

struct CpuEvent
{
  std::uint32_t slab;
  std::int64_t y;
  std::int32_t delta;
};

struct CpuBoundaryEvent
{
  std::uint32_t boundary;
  std::int64_t y;
  std::int32_t left;
  std::int32_t right;
};

struct UnionOutput
{
  bool fallback = false;
  std::string message;
  std::vector<DirectedSegmentI64> segments;
  std::uint64_t memberships = 0;
  std::uint64_t x_slabs = 0;
  std::uint64_t strip_intervals = 0;
  std::uint64_t raw_segments = 0;
  std::uint64_t digest = 0;
  double total_ms = 0.0;
  double h2d_ms = 0.0;
  double x_membership_ms = 0.0;
  double strip_scan_ms = 0.0;
  double boundary_ms = 0.0;
  double d2h_ms = 0.0;
};

double elapsed_ms(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 0;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>((count + kThreads - 1) / kThreads, 65535));
}

bool checked_multiply(std::uint64_t first, std::uint64_t second,
                      std::uint64_t *result)
{
  if (first && second > std::numeric_limits<std::uint64_t>::max() / first)
    return false;
  *result = first * second;
  return true;
}

bool valid_rectangle(const RectI64 &rectangle)
{
  return rectangle.left < rectangle.right &&
         rectangle.bottom < rectangle.top;
}

std::uint64_t digest_segments(
    const std::vector<DirectedSegmentI64> &segments)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&hash](std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
      hash *= UINT64_C(1099511628211);
    }
  };
  mix(segments.size());
  for (const auto &segment : segments) {
    mix(static_cast<std::uint32_t>(segment.axis));
    mix(static_cast<std::uint32_t>(segment.side));
    mix(static_cast<std::uint64_t>(segment.fixed));
    mix(static_cast<std::uint64_t>(segment.lo));
    mix(static_cast<std::uint64_t>(segment.hi));
  }
  return hash;
}

void canonicalize_segments(std::vector<DirectedSegmentI64> *segments)
{
  std::sort(segments->begin(), segments->end(), SegmentLess{});
  std::vector<DirectedSegmentI64> merged;
  merged.reserve(segments->size());
  for (const auto &segment : *segments) {
    if (segment.axis == SegmentAxis::invalid || segment.lo >= segment.hi)
      throw std::runtime_error("invalid raw CPU boundary segment");
    if (!merged.empty() && same_segment_line(merged.back(), segment) &&
        segment.lo <= merged.back().hi) {
      merged.back().hi = std::max(merged.back().hi, segment.hi);
    } else {
      merged.push_back(segment);
    }
  }
  segments->swap(merged);
}

UnionOutput cpu_union(const std::vector<RectI64> &rectangles,
                      const Limits &limits)
{
  const auto total_begin = Clock::now();
  UnionOutput output;
  if (rectangles.size() > limits.max_rectangles) {
    output.fallback = true;
    output.message = "rectangle capacity";
    return output;
  }
  if (rectangles.empty()) {
    output.digest = digest_segments(output.segments);
    output.total_ms = elapsed_ms(total_begin, Clock::now());
    return output;
  }

  std::vector<std::int64_t> xs;
  xs.reserve(rectangles.size() * 2);
  for (const auto &rectangle : rectangles) {
    if (!valid_rectangle(rectangle)) {
      output.fallback = true;
      output.message = "invalid or degenerate rectangle";
      output.total_ms = elapsed_ms(total_begin, Clock::now());
      return output;
    }
    xs.push_back(rectangle.left);
    xs.push_back(rectangle.right);
  }
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  output.x_slabs = xs.size() - 1;
  if (output.x_slabs > limits.max_x_slabs) {
    output.fallback = true;
    output.message = "x-slab capacity";
    output.total_ms = elapsed_ms(total_begin, Clock::now());
    return output;
  }

  std::vector<CpuEvent> events;
  for (const auto &rectangle : rectangles) {
    const std::uint64_t first = static_cast<std::uint64_t>(
        std::lower_bound(xs.begin(), xs.end(), rectangle.left) - xs.begin());
    const std::uint64_t last = static_cast<std::uint64_t>(
        std::lower_bound(xs.begin(), xs.end(), rectangle.right) - xs.begin());
    const std::uint64_t span = last - first;
    if (span > limits.max_slabs_per_rectangle ||
        span > limits.max_memberships ||
        output.memberships > limits.max_memberships - span) {
      output.fallback = true;
      output.message = span > limits.max_slabs_per_rectangle
                           ? "per-rectangle x-slab capacity"
                           : "membership capacity";
      output.total_ms = elapsed_ms(total_begin, Clock::now());
      return output;
    }
    output.memberships += span;
    for (std::uint64_t slab = first; slab < last; ++slab) {
      events.push_back(
          {static_cast<std::uint32_t>(slab), rectangle.bottom, 1});
      events.push_back(
          {static_cast<std::uint32_t>(slab), rectangle.top, -1});
    }
  }
  if (events.size() > limits.max_events) {
    output.fallback = true;
    output.message = "event capacity";
    output.total_ms = elapsed_ms(total_begin, Clock::now());
    return output;
  }

  std::sort(events.begin(), events.end(),
            [](const CpuEvent &first, const CpuEvent &second) {
              if (first.slab != second.slab)
                return first.slab < second.slab;
              return first.y < second.y;
            });

  std::vector<Transition> transitions;
  for (std::size_t begin = 0; begin < events.size();) {
    const std::uint32_t slab = events[begin].slab;
    std::int32_t coverage = 0;
    while (begin < events.size() && events[begin].slab == slab) {
      const std::int64_t y = events[begin].y;
      std::int32_t delta = 0;
      while (begin < events.size() && events[begin].slab == slab &&
             events[begin].y == y) {
        delta += events[begin].delta;
        ++begin;
      }
      if (!delta) continue;
      const std::int32_t previous = coverage;
      coverage += delta;
      if (coverage < 0)
        throw std::runtime_error("negative CPU sweep coverage");
      if (!previous && coverage > 0)
        transitions.push_back({y, slab, 1});
      else if (previous > 0 && !coverage)
        transitions.push_back({y, slab, -1});
    }
    if (coverage)
      throw std::runtime_error("unclosed CPU sweep coverage");
  }

  if (transitions.size() % 2)
    throw std::runtime_error("odd CPU transition count");
  std::vector<StripInterval> intervals;
  intervals.reserve(transitions.size() / 2);
  std::vector<DirectedSegmentI64> raw;
  raw.reserve(transitions.size() * 2);
  for (std::size_t index = 0; index < transitions.size(); index += 2) {
    const Transition &start = transitions[index];
    const Transition &finish = transitions[index + 1];
    if (start.kind != 1 || finish.kind != -1 ||
        start.slab != finish.slab || start.y >= finish.y) {
      throw std::runtime_error("invalid CPU transition pair");
    }
    intervals.push_back(
        {start.y, finish.y, start.slab, 0});
    raw.push_back(
        {start.y, xs[start.slab], xs[start.slab + 1], -1,
         SegmentAxis::horizontal});
    raw.push_back(
        {finish.y, xs[start.slab], xs[start.slab + 1], 1,
         SegmentAxis::horizontal});
  }
  output.strip_intervals = intervals.size();

  std::vector<CpuBoundaryEvent> boundary_events;
  boundary_events.reserve(intervals.size() * 4);
  for (const auto &interval : intervals) {
    boundary_events.push_back(
        {interval.slab, interval.bottom, 0, 1});
    boundary_events.push_back(
        {interval.slab, interval.top, 0, -1});
    boundary_events.push_back(
        {interval.slab + 1, interval.bottom, 1, 0});
    boundary_events.push_back(
        {interval.slab + 1, interval.top, -1, 0});
  }
  std::sort(
      boundary_events.begin(), boundary_events.end(),
      [](const CpuBoundaryEvent &first, const CpuBoundaryEvent &second) {
        if (first.boundary != second.boundary)
          return first.boundary < second.boundary;
        return first.y < second.y;
      });

  std::size_t position = 0;
  while (position < boundary_events.size()) {
    const std::uint32_t boundary = boundary_events[position].boundary;
    std::int32_t left = 0;
    std::int32_t right = 0;
    while (position < boundary_events.size() &&
           boundary_events[position].boundary == boundary) {
      const std::int64_t y = boundary_events[position].y;
      std::int32_t delta_left = 0;
      std::int32_t delta_right = 0;
      while (position < boundary_events.size() &&
             boundary_events[position].boundary == boundary &&
             boundary_events[position].y == y) {
        delta_left += boundary_events[position].left;
        delta_right += boundary_events[position].right;
        ++position;
      }
      left += delta_left;
      right += delta_right;
      if (left < 0 || right < 0)
        throw std::runtime_error("negative CPU boundary coverage");
      if (position < boundary_events.size() &&
          boundary_events[position].boundary == boundary &&
          y < boundary_events[position].y &&
          static_cast<bool>(left) != static_cast<bool>(right)) {
        raw.push_back(
            {xs[boundary], y, boundary_events[position].y,
             right ? -1 : 1, SegmentAxis::vertical});
      }
    }
    if (left || right)
      throw std::runtime_error("unclosed CPU boundary coverage");
  }

  output.raw_segments = raw.size();
  canonicalize_segments(&raw);
  output.segments.swap(raw);
  output.digest = digest_segments(output.segments);
  output.total_ms = elapsed_ms(total_begin, Clock::now());
  return output;
}

__global__ void emit_x_endpoints_kernel(
    const RectI64 *rectangles, std::uint64_t rectangle_count,
    std::int64_t *endpoints, std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < rectangle_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const RectI64 rectangle = rectangles[index];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusInvalidRectangle));
      continue;
    }
    endpoints[index * 2] = rectangle.left;
    endpoints[index * 2 + 1] = rectangle.right;
  }
}

__device__ std::uint64_t lower_bound_device(
    const std::int64_t *values, std::uint64_t count, std::int64_t target)
{
  std::uint64_t first = 0;
  while (first < count) {
    const std::uint64_t middle = first + (count - first) / 2;
    if (values[middle] < target)
      first = middle + 1;
    else
      count = middle;
  }
  return first;
}

__global__ void count_memberships_kernel(
    const RectI64 *rectangles, std::uint64_t rectangle_count,
    const std::int64_t *xs, std::uint64_t x_count,
    std::uint32_t max_slabs_per_rectangle, std::uint64_t *counts,
    std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < rectangle_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const RectI64 rectangle = rectangles[index];
    const std::uint64_t first =
        lower_bound_device(xs, x_count, rectangle.left);
    const std::uint64_t last =
        lower_bound_device(xs, x_count, rectangle.right);
    if (first >= x_count || last >= x_count ||
        xs[first] != rectangle.left || xs[last] != rectangle.right ||
        first >= last) {
      atomicOr(status, static_cast<std::uint32_t>(kStatusEndpointLookup));
      counts[index] = 0;
      continue;
    }
    const std::uint64_t span = last - first;
    if (span > max_slabs_per_rectangle) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(kStatusPerRectangleCapacity));
      counts[index] = 0;
      continue;
    }
    counts[index] = span;
  }
}

__global__ void fill_events_kernel(
    const RectI64 *rectangles, std::uint64_t rectangle_count,
    const std::int64_t *xs, std::uint64_t x_count,
    const std::uint64_t *offsets, EventKey *keys, std::int32_t *deltas)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < rectangle_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const RectI64 rectangle = rectangles[index];
    const std::uint64_t first =
        lower_bound_device(xs, x_count, rectangle.left);
    const std::uint64_t last =
        lower_bound_device(xs, x_count, rectangle.right);
    std::uint64_t output = offsets[index] * 2;
    for (std::uint64_t slab = first; slab < last; ++slab) {
      keys[output] = {rectangle.bottom,
                      static_cast<std::uint32_t>(slab), 0};
      deltas[output++] = 1;
      keys[output] = {rectangle.top,
                      static_cast<std::uint32_t>(slab), 0};
      deltas[output++] = -1;
    }
  }
}

__global__ void extract_transitions_kernel(
    const EventKey *keys, const std::int32_t *coverage,
    std::uint64_t event_count, Transition *transitions,
    std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < event_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::int32_t after = coverage[index];
    const std::int32_t before =
        index && keys[index - 1].slab == keys[index].slab
            ? coverage[index - 1]
            : 0;
    Transition transition = {0, 0, 0};
    if (after < 0) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusCoverageInvariant));
    } else if (!before && after > 0) {
      transition = {keys[index].y, keys[index].slab, 1};
    } else if (before > 0 && !after) {
      transition = {keys[index].y, keys[index].slab, -1};
    }
    if ((index + 1 == event_count ||
         keys[index + 1].slab != keys[index].slab) &&
        after != 0) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusCoverageInvariant));
    }
    transitions[index] = transition;
  }
}

__global__ void build_intervals_and_horizontal_kernel(
    const Transition *transitions, std::uint64_t transition_count,
    const std::int64_t *xs, StripInterval *intervals,
    DirectedSegmentI64 *horizontal, std::uint32_t *status)
{
  for (std::uint64_t pair =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       pair < transition_count / 2;
       pair += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const Transition first = transitions[pair * 2];
    const Transition second = transitions[pair * 2 + 1];
    if (first.kind != 1 || second.kind != -1 ||
        first.slab != second.slab || first.y >= second.y) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusTransitionInvariant));
      continue;
    }
    intervals[pair] = {first.y, second.y, first.slab, 0};
    horizontal[pair * 2] = {
        first.y, xs[first.slab], xs[first.slab + 1], -1,
        SegmentAxis::horizontal};
    horizontal[pair * 2 + 1] = {
        second.y, xs[first.slab], xs[first.slab + 1], 1,
        SegmentAxis::horizontal};
  }
}

__global__ void fill_boundary_events_kernel(
    const StripInterval *intervals, std::uint64_t interval_count,
    BoundaryKey *keys, SideDelta *deltas)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < interval_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const StripInterval interval = intervals[index];
    const std::uint64_t output = index * 4;
    keys[output] = {interval.bottom, interval.slab, 0};
    deltas[output] = {0, 1};
    keys[output + 1] = {interval.top, interval.slab, 0};
    deltas[output + 1] = {0, -1};
    keys[output + 2] = {interval.bottom, interval.slab + 1, 0};
    deltas[output + 2] = {1, 0};
    keys[output + 3] = {interval.top, interval.slab + 1, 0};
    deltas[output + 3] = {-1, 0};
  }
}

__global__ void extract_vertical_segments_kernel(
    const BoundaryKey *keys, const SideDelta *coverage,
    std::uint64_t event_count, const std::int64_t *xs,
    DirectedSegmentI64 *segments, std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < event_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    DirectedSegmentI64 segment = {
        0, 0, 0, 0, SegmentAxis::invalid};
    const SideDelta after = coverage[index];
    if (after.left < 0 || after.right < 0) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusCoverageInvariant));
    }
    const bool same_boundary =
        index + 1 < event_count &&
        keys[index + 1].boundary == keys[index].boundary;
    if (same_boundary && keys[index].y < keys[index + 1].y &&
        static_cast<bool>(after.left) !=
            static_cast<bool>(after.right)) {
      segment = {
          xs[keys[index].boundary], keys[index].y, keys[index + 1].y,
          after.right ? -1 : 1, SegmentAxis::vertical};
    }
    if (!same_boundary && (after.left || after.right)) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusCoverageInvariant));
    }
    segments[index] = segment;
  }
}

__global__ void mark_segment_groups_kernel(
    const DirectedSegmentI64 *segments, std::uint64_t segment_count,
    const std::int64_t *line_prefix_high, std::uint32_t *marks)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < segment_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (!index) {
      marks[index] = 1;
      continue;
    }
    const DirectedSegmentI64 previous = segments[index - 1];
    const DirectedSegmentI64 current = segments[index];
    marks[index] =
        !(same_segment_line(previous, current) &&
          current.lo <= line_prefix_high[index - 1]);
  }
}

std::uint64_t canonicalize_device_segments(
    thrust::device_vector<DirectedSegmentI64> *segments)
{
  const std::uint64_t segment_count = segments->size();
  if (!segment_count) return 0;
  thrust::sort(
      thrust::device, segments->begin(), segments->end(), SegmentLess{});

  /*
   * A simple comparison with the immediately previous hi is insufficient:
   * [0,100], [10,20], [30,40] is one interval even though the latter two do
   * not touch each other.  Segmented prefix-max makes the grouping exact for
   * arbitrary duplicate/overlapping raw fragments.
   */
  const auto line_keys = thrust::make_transform_iterator(
      segments->begin(), SegmentLineFromSegment{});
  const auto highs =
      thrust::make_transform_iterator(segments->begin(), SegmentHigh{});
  thrust::device_vector<std::int64_t> line_prefix_high(segment_count);
  thrust::inclusive_scan_by_key(
      thrust::device, line_keys, line_keys + segment_count, highs,
      line_prefix_high.begin(), SegmentLineEqual{},
      thrust::maximum<std::int64_t>{});

  thrust::device_vector<std::uint32_t> group_marks(segment_count);
  thrust::device_vector<std::uint32_t> group_ids(segment_count);
  mark_segment_groups_kernel<<<launch_blocks(segment_count), kThreads>>>(
      thrust::raw_pointer_cast(segments->data()), segment_count,
      thrust::raw_pointer_cast(line_prefix_high.data()),
      thrust::raw_pointer_cast(group_marks.data()));
  cuda_require(cudaGetLastError(), "mark segment groups");
  thrust::inclusive_scan(
      thrust::device, group_marks.begin(), group_marks.end(),
      group_ids.begin());

  std::uint32_t canonical_count = 0;
  cuda_require(
      cudaMemcpy(
          &canonical_count,
          thrust::raw_pointer_cast(group_ids.data()) + segment_count - 1,
          sizeof(canonical_count), cudaMemcpyDeviceToHost),
      "canonical group count D2H");
  thrust::device_vector<std::uint32_t> output_group_ids(canonical_count);
  thrust::device_vector<DirectedSegmentI64> canonical(canonical_count);
  const auto canonical_end = thrust::reduce_by_key(
      thrust::device, group_ids.begin(), group_ids.end(),
      segments->begin(), output_group_ids.begin(), canonical.begin(),
      thrust::equal_to<std::uint32_t>{}, SegmentMerge{});
  if (static_cast<std::uint64_t>(
          canonical_end.second - canonical.begin()) != canonical_count) {
    throw std::runtime_error("canonical segment count mismatch");
  }
  segments->swap(canonical);
  return canonical_count;
}

std::uint32_t copy_device_status(
    const thrust::device_vector<std::uint32_t> &status)
{
  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(&host_status, thrust::raw_pointer_cast(status.data()),
                 sizeof(host_status), cudaMemcpyDeviceToHost),
      "status D2H");
  return host_status;
}

std::string status_message(std::uint32_t status)
{
  std::ostringstream stream;
  if (status & kStatusInvalidRectangle) stream << "invalid rectangle;";
  if (status & kStatusEndpointLookup) stream << "endpoint lookup;";
  if (status & kStatusPerRectangleCapacity)
    stream << "per-rectangle slab capacity;";
  if (status & kStatusCoverageInvariant) stream << "coverage invariant;";
  if (status & kStatusTransitionInvariant)
    stream << "transition invariant;";
  return stream.str();
}

UnionOutput gpu_union(const std::vector<RectI64> &rectangles,
                      const Limits &limits, int device)
{
  const auto total_begin = Clock::now();
  UnionOutput output;
  auto pipeline = [&]() {
    if (rectangles.size() > limits.max_rectangles) {
      output.fallback = true;
      output.message = "rectangle capacity";
      return;
    }
    if (rectangles.empty()) {
      output.digest = digest_segments(output.segments);
      return;
    }
    for (const auto &rectangle : rectangles) {
      if (!valid_rectangle(rectangle)) {
        output.fallback = true;
        output.message = "invalid or degenerate rectangle";
        return;
      }
    }

    cuda_require(cudaSetDevice(device), "cudaSetDevice");
    thrust::device_vector<std::uint32_t> status(1, 0);

    const auto h2d_begin = Clock::now();
    thrust::device_vector<RectI64> device_rectangles(rectangles.size());
    cuda_require(
        cudaMemcpy(
            thrust::raw_pointer_cast(device_rectangles.data()),
            rectangles.data(), rectangles.size() * sizeof(RectI64),
            cudaMemcpyHostToDevice),
        "rectangle H2D");
    cuda_require(cudaDeviceSynchronize(), "rectangle H2D synchronize");
    output.h2d_ms = elapsed_ms(h2d_begin, Clock::now());

    const auto x_begin = Clock::now();
    std::uint64_t endpoint_count = 0;
    if (!checked_multiply(rectangles.size(), UINT64_C(2),
                          &endpoint_count)) {
      output.fallback = true;
      output.message = "endpoint count overflow";
      return;
    }
    thrust::device_vector<std::int64_t> xs(endpoint_count);
    const std::uint32_t rectangle_blocks =
        launch_blocks(rectangles.size());
    emit_x_endpoints_kernel<<<rectangle_blocks, kThreads>>>(
        thrust::raw_pointer_cast(device_rectangles.data()),
        rectangles.size(), thrust::raw_pointer_cast(xs.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "emit x endpoints");
    thrust::sort(thrust::device, xs.begin(), xs.end());
    const auto x_end =
        thrust::unique(thrust::device, xs.begin(), xs.end());
    const std::uint64_t x_count = x_end - xs.begin();
    xs.resize(x_count);
    if (x_count < 2) {
      output.fallback = true;
      output.message = "fewer than two x endpoints";
      return;
    }
    output.x_slabs = x_count - 1;
    if (output.x_slabs > limits.max_x_slabs ||
        output.x_slabs > std::numeric_limits<std::uint32_t>::max()) {
      output.fallback = true;
      output.message = "x-slab capacity";
      return;
    }

    thrust::device_vector<std::uint64_t> membership_counts(
        rectangles.size());
    thrust::device_vector<std::uint64_t> membership_offsets(
        rectangles.size());
    count_memberships_kernel<<<rectangle_blocks, kThreads>>>(
        thrust::raw_pointer_cast(device_rectangles.data()),
        rectangles.size(), thrust::raw_pointer_cast(xs.data()), x_count,
        limits.max_slabs_per_rectangle,
        thrust::raw_pointer_cast(membership_counts.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "count slab memberships");
    thrust::exclusive_scan(
        thrust::device, membership_counts.begin(), membership_counts.end(),
        membership_offsets.begin(), std::uint64_t{0});
    std::uint64_t last_offset = 0;
    std::uint64_t last_count = 0;
    cuda_require(
        cudaMemcpy(
            &last_offset,
            thrust::raw_pointer_cast(membership_offsets.data()) +
                rectangles.size() - 1,
            sizeof(last_offset), cudaMemcpyDeviceToHost),
        "last membership offset D2H");
    cuda_require(
        cudaMemcpy(
            &last_count,
            thrust::raw_pointer_cast(membership_counts.data()) +
                rectangles.size() - 1,
            sizeof(last_count), cudaMemcpyDeviceToHost),
        "last membership count D2H");
    output.memberships = last_offset + last_count;
    const std::uint32_t membership_status = copy_device_status(status);
    if (membership_status) {
      output.fallback = true;
      output.message = status_message(membership_status);
      return;
    }
    if (output.memberships > limits.max_memberships) {
      output.fallback = true;
      output.message = "membership capacity";
      return;
    }
    std::uint64_t event_count = 0;
    if (!checked_multiply(output.memberships, UINT64_C(2),
                          &event_count) ||
        event_count > limits.max_events) {
      output.fallback = true;
      output.message = "event capacity";
      return;
    }
    cuda_require(cudaDeviceSynchronize(), "x/membership synchronize");
    output.x_membership_ms = elapsed_ms(x_begin, Clock::now());

    const auto strip_begin = Clock::now();
    thrust::device_vector<EventKey> event_keys(event_count);
    thrust::device_vector<std::int32_t> event_deltas(event_count);
    fill_events_kernel<<<rectangle_blocks, kThreads>>>(
        thrust::raw_pointer_cast(device_rectangles.data()),
        rectangles.size(), thrust::raw_pointer_cast(xs.data()), x_count,
        thrust::raw_pointer_cast(membership_offsets.data()),
        thrust::raw_pointer_cast(event_keys.data()),
        thrust::raw_pointer_cast(event_deltas.data()));
    cuda_require(cudaGetLastError(), "fill slab events");
    thrust::sort_by_key(
        thrust::device, event_keys.begin(), event_keys.end(),
        event_deltas.begin(), EventKeyLess{});

    thrust::device_vector<EventKey> unique_event_keys(event_count);
    thrust::device_vector<std::int32_t> unique_event_deltas(event_count);
    auto reduced_events = thrust::reduce_by_key(
        thrust::device, event_keys.begin(), event_keys.end(),
        event_deltas.begin(), unique_event_keys.begin(),
        unique_event_deltas.begin(), EventKeyEqual{},
        thrust::plus<std::int32_t>{});
    std::uint64_t unique_event_count =
        reduced_events.first - unique_event_keys.begin();
    auto zipped_events = thrust::make_zip_iterator(
        thrust::make_tuple(unique_event_keys.begin(),
                           unique_event_deltas.begin()));
    const auto nonzero_event_end = thrust::remove_if(
        thrust::device, zipped_events, zipped_events + unique_event_count,
        ZeroEventDelta{});
    unique_event_count = nonzero_event_end - zipped_events;
    unique_event_keys.resize(unique_event_count);
    unique_event_deltas.resize(unique_event_count);
    if (!unique_event_count) {
      output.fallback = true;
      output.message = "empty event stream for nonempty input";
      return;
    }

    thrust::device_vector<std::int32_t> coverage(unique_event_count);
    const auto slab_ids = thrust::make_transform_iterator(
        unique_event_keys.begin(), EventSlab{});
    thrust::inclusive_scan_by_key(
        thrust::device, slab_ids, slab_ids + unique_event_count,
        unique_event_deltas.begin(), coverage.begin(),
        thrust::equal_to<std::uint32_t>{},
        thrust::plus<std::int32_t>{});

    thrust::device_vector<Transition> transitions(unique_event_count);
    extract_transitions_kernel<<<launch_blocks(unique_event_count), kThreads>>>(
        thrust::raw_pointer_cast(unique_event_keys.data()),
        thrust::raw_pointer_cast(coverage.data()), unique_event_count,
        thrust::raw_pointer_cast(transitions.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "extract strip transitions");
    const auto transition_end = thrust::remove_if(
        thrust::device, transitions.begin(), transitions.end(),
        InvalidTransition{});
    const std::uint64_t transition_count =
        transition_end - transitions.begin();
    transitions.resize(transition_count);
    if (transition_count % 2) {
      output.fallback = true;
      output.message = "odd transition count";
      return;
    }
    output.strip_intervals = transition_count / 2;
    std::uint64_t horizontal_count = transition_count;
    std::uint64_t boundary_event_count = 0;
    if (!checked_multiply(
            output.strip_intervals, UINT64_C(4),
            &boundary_event_count) ||
        boundary_event_count > limits.max_events ||
        horizontal_count > limits.max_segments) {
      output.fallback = true;
      output.message = "strip-output capacity";
      return;
    }
    thrust::device_vector<StripInterval> intervals(
        output.strip_intervals);
    thrust::device_vector<DirectedSegmentI64> horizontal(
        horizontal_count);
    if (output.strip_intervals) {
      build_intervals_and_horizontal_kernel<<<
          launch_blocks(output.strip_intervals), kThreads>>>(
          thrust::raw_pointer_cast(transitions.data()), transition_count,
          thrust::raw_pointer_cast(xs.data()),
          thrust::raw_pointer_cast(intervals.data()),
          thrust::raw_pointer_cast(horizontal.data()),
          thrust::raw_pointer_cast(status.data()));
      cuda_require(cudaGetLastError(), "build strip intervals");
    }
    cuda_require(cudaDeviceSynchronize(), "strip sweep synchronize");
    const std::uint32_t strip_status = copy_device_status(status);
    if (strip_status) {
      output.fallback = true;
      output.message = status_message(strip_status);
      return;
    }
    output.strip_scan_ms = elapsed_ms(strip_begin, Clock::now());

    const auto boundary_begin = Clock::now();
    thrust::device_vector<BoundaryKey> boundary_keys(
        boundary_event_count);
    thrust::device_vector<SideDelta> boundary_deltas(
        boundary_event_count);
    if (output.strip_intervals) {
      fill_boundary_events_kernel<<<
          launch_blocks(output.strip_intervals), kThreads>>>(
          thrust::raw_pointer_cast(intervals.data()),
          output.strip_intervals,
          thrust::raw_pointer_cast(boundary_keys.data()),
          thrust::raw_pointer_cast(boundary_deltas.data()));
      cuda_require(cudaGetLastError(), "fill boundary events");
    }
    thrust::sort_by_key(
        thrust::device, boundary_keys.begin(), boundary_keys.end(),
        boundary_deltas.begin(), BoundaryKeyLess{});
    thrust::device_vector<BoundaryKey> unique_boundary_keys(
        boundary_event_count);
    thrust::device_vector<SideDelta> unique_boundary_deltas(
        boundary_event_count);
    const auto reduced_boundaries = thrust::reduce_by_key(
        thrust::device, boundary_keys.begin(), boundary_keys.end(),
        boundary_deltas.begin(), unique_boundary_keys.begin(),
        unique_boundary_deltas.begin(), BoundaryKeyEqual{},
        SideDeltaPlus{});
    std::uint64_t unique_boundary_count =
        reduced_boundaries.first - unique_boundary_keys.begin();
    auto zipped_boundaries = thrust::make_zip_iterator(
        thrust::make_tuple(unique_boundary_keys.begin(),
                           unique_boundary_deltas.begin()));
    const auto nonzero_boundary_end = thrust::remove_if(
        thrust::device, zipped_boundaries,
        zipped_boundaries + unique_boundary_count, ZeroSideDelta{});
    unique_boundary_count = nonzero_boundary_end - zipped_boundaries;
    unique_boundary_keys.resize(unique_boundary_count);
    unique_boundary_deltas.resize(unique_boundary_count);

    thrust::device_vector<SideDelta> side_coverage(
        unique_boundary_count);
    const auto boundary_ids = thrust::make_transform_iterator(
        unique_boundary_keys.begin(), BoundaryId{});
    thrust::inclusive_scan_by_key(
        thrust::device, boundary_ids,
        boundary_ids + unique_boundary_count,
        unique_boundary_deltas.begin(), side_coverage.begin(),
        thrust::equal_to<std::uint32_t>{}, SideDeltaPlus{});
    thrust::device_vector<DirectedSegmentI64> vertical(
        unique_boundary_count);
    if (unique_boundary_count) {
      extract_vertical_segments_kernel<<<
          launch_blocks(unique_boundary_count), kThreads>>>(
          thrust::raw_pointer_cast(unique_boundary_keys.data()),
          thrust::raw_pointer_cast(side_coverage.data()),
          unique_boundary_count, thrust::raw_pointer_cast(xs.data()),
          thrust::raw_pointer_cast(vertical.data()),
          thrust::raw_pointer_cast(status.data()));
      cuda_require(cudaGetLastError(), "extract vertical segments");
    }
    const auto vertical_end = thrust::remove_if(
        thrust::device, vertical.begin(), vertical.end(),
        InvalidSegment{});
    const std::uint64_t vertical_count =
        vertical_end - vertical.begin();
    vertical.resize(vertical_count);
    std::uint64_t raw_segment_count = 0;
    if (horizontal_count >
            std::numeric_limits<std::uint64_t>::max() - vertical_count) {
      output.fallback = true;
      output.message = "raw segment count overflow";
      return;
    }
    raw_segment_count = horizontal_count + vertical_count;
    output.raw_segments = raw_segment_count;
    if (raw_segment_count > limits.max_segments) {
      output.fallback = true;
      output.message = "segment capacity";
      return;
    }

    thrust::device_vector<DirectedSegmentI64> raw_segments(
        raw_segment_count);
    thrust::copy(
        thrust::device, horizontal.begin(), horizontal.end(),
        raw_segments.begin());
    thrust::copy(
        thrust::device, vertical.begin(), vertical.end(),
        raw_segments.begin() + horizontal_count);
    const std::uint64_t canonical_count =
        canonicalize_device_segments(&raw_segments);
    cuda_require(cudaDeviceSynchronize(), "boundary synchronize");
    const std::uint32_t boundary_status = copy_device_status(status);
    if (boundary_status) {
      output.fallback = true;
      output.message = status_message(boundary_status);
      return;
    }
    output.boundary_ms = elapsed_ms(boundary_begin, Clock::now());

    const auto d2h_begin = Clock::now();
    output.segments.resize(canonical_count);
    if (canonical_count) {
      cuda_require(
          cudaMemcpy(
              output.segments.data(),
              thrust::raw_pointer_cast(raw_segments.data()),
              canonical_count * sizeof(DirectedSegmentI64),
              cudaMemcpyDeviceToHost),
          "boundary D2H");
    }
    cuda_require(cudaDeviceSynchronize(), "boundary D2H synchronize");
    output.d2h_ms = elapsed_ms(d2h_begin, Clock::now());
    output.digest = digest_segments(output.segments);
  };

  pipeline();
  /*
   * All device_vector destructors above run before total_ms is sampled because
   * they are scoped inside the pipeline lambda.  This intentionally charges
   * allocation and teardown as well as transfers and kernels.
   */
  output.total_ms = elapsed_ms(total_begin, Clock::now());
  return output;
}

std::vector<DirectedSegmentI64> gpu_canonicalize_for_test(
    const std::vector<DirectedSegmentI64> &raw, int device)
{
  cuda_require(cudaSetDevice(device), "canonical test cudaSetDevice");
  thrust::device_vector<DirectedSegmentI64> segments(
      raw.begin(), raw.end());
  canonicalize_device_segments(&segments);
  std::vector<DirectedSegmentI64> result(segments.size());
  if (!result.empty()) {
    cuda_require(
        cudaMemcpy(
            result.data(), thrust::raw_pointer_cast(segments.data()),
            result.size() * sizeof(DirectedSegmentI64),
            cudaMemcpyDeviceToHost),
        "canonical test D2H");
  }
  return result;
}

std::string segment_string(const DirectedSegmentI64 &segment)
{
  std::ostringstream stream;
  stream << (segment.axis == SegmentAxis::horizontal ? "H" : "V")
         << (segment.side < 0 ? "-" : "+") << " fixed=" << segment.fixed
         << " [" << segment.lo << "," << segment.hi << ")";
  return stream.str();
}

void require_equal(const std::string &name, const UnionOutput &cpu,
                   const UnionOutput &gpu)
{
  if (cpu.fallback || gpu.fallback) {
    throw std::runtime_error(
        name + ": unexpected fallback cpu='" + cpu.message + "' gpu='" +
        gpu.message + "'");
  }
  if (cpu.segments.size() != gpu.segments.size()) {
    std::ostringstream stream;
    stream << name << ": segment-count mismatch CPU="
           << cpu.segments.size() << " GPU=" << gpu.segments.size();
    throw std::runtime_error(stream.str());
  }
  for (std::size_t index = 0; index < cpu.segments.size(); ++index) {
    const auto &first = cpu.segments[index];
    const auto &second = gpu.segments[index];
    if (first.axis != second.axis || first.side != second.side ||
        first.fixed != second.fixed || first.lo != second.lo ||
        first.hi != second.hi) {
      throw std::runtime_error(
          name + ": segment mismatch at " + std::to_string(index) +
          " CPU=" + segment_string(first) +
          " GPU=" + segment_string(second));
    }
  }
  if (cpu.digest != gpu.digest)
    throw std::runtime_error(name + ": digest mismatch");
}

RectI64 rectangle(std::int64_t left, std::int64_t bottom,
                  std::int64_t right, std::int64_t top,
                  std::uint64_t token = 0)
{
  return {left, bottom, right, top, token, 0};
}

struct Fixture
{
  std::string name;
  std::vector<RectI64> rectangles;
};

std::vector<Fixture> directed_fixtures()
{
  return {
      {"empty", {}},
      {"single", {rectangle(0, 0, 10, 20)}},
      {"duplicate",
       {rectangle(0, 0, 10, 10), rectangle(0, 0, 10, 10)}},
      {"nested",
       {rectangle(-20, -20, 20, 20), rectangle(-5, -5, 5, 5)}},
      {"partial-overlap",
       {rectangle(0, 0, 10, 10), rectangle(5, 3, 15, 14)}},
      {"edge-touch-x",
       {rectangle(0, 0, 10, 10), rectangle(10, 0, 20, 10)}},
      {"edge-touch-y",
       {rectangle(0, 0, 10, 10), rectangle(0, 10, 10, 20)}},
      {"corner-touch-opposite-sides",
       {rectangle(0, 0, 10, 10), rectangle(10, 10, 20, 20)}},
      {"t-junction",
       {rectangle(0, 0, 30, 10), rectangle(10, 10, 20, 30)}},
      {"plus",
       {rectangle(-5, -20, 5, 20), rectangle(-20, -5, 20, 5)}},
      {"ring-with-hole",
       {rectangle(0, 0, 30, 5), rectangle(0, 25, 30, 30),
        rectangle(0, 5, 5, 25), rectangle(25, 5, 30, 25)}},
      {"covered-seam",
       {rectangle(0, 0, 10, 30), rectangle(10, 0, 20, 10),
        rectangle(10, 20, 20, 30), rectangle(5, 5, 15, 25)}},
      {"containment-bridge",
       {rectangle(0, 0, 10, 100), rectangle(10, 10, 20, 20),
        rectangle(10, 30, 20, 40), rectangle(20, 0, 30, 100)}},
      {"negative",
       {rectangle(-100, -90, -10, -20),
        rectangle(-60, -110, 20, -50)}},
      {"large-int64",
       {rectangle(INT64_C(-4000000000000000000),
                  INT64_C(-3000000000000000000),
                  INT64_C(-3999999999999999900),
                  INT64_C(-2999999999999999800)),
        rectangle(INT64_C(-3999999999999999950),
                  INT64_C(-2999999999999999900),
                  INT64_C(-3999999999999999800),
                  INT64_C(-2999999999999999700))}},
  };
}

std::vector<RectI64> random_rectangles(
    std::uint64_t seed, std::size_t count)
{
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<std::int64_t> coordinate(-80, 80);
  std::uniform_int_distribution<std::int64_t> extent(1, 30);
  std::vector<RectI64> rectangles;
  rectangles.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::int64_t left = coordinate(generator);
    const std::int64_t bottom = coordinate(generator);
    rectangles.push_back(
        rectangle(left, bottom, left + extent(generator),
                  bottom + extent(generator), index + 1));
  }
  return rectangles;
}

void run_self_test(int device)
{
  const Limits limits;
  std::size_t checks = 0;
  for (const auto &fixture : directed_fixtures()) {
    const UnionOutput cpu = cpu_union(fixture.rectangles, limits);
    const UnionOutput gpu = gpu_union(fixture.rectangles, limits, device);
    require_equal(fixture.name, cpu, gpu);
    std::cout << "MANHATTAN_UNION_FIXTURE ok name=" << fixture.name
              << " rectangles=" << fixture.rectangles.size()
              << " segments=" << gpu.segments.size()
              << " digest=0x" << std::hex << gpu.digest << std::dec
              << " gpu_ms=" << std::fixed << std::setprecision(3)
              << gpu.total_ms << "\n";
    ++checks;
  }

  for (std::uint64_t seed = 1; seed <= 64; ++seed) {
    const auto rectangles = random_rectangles(seed, 1 + seed % 47);
    const UnionOutput cpu = cpu_union(rectangles, limits);
    const UnionOutput gpu = gpu_union(rectangles, limits, device);
    require_equal("random-" + std::to_string(seed), cpu, gpu);
    ++checks;
  }
  std::cout << "MANHATTAN_UNION_RANDOM ok cases=64\n";

  /*
   * Directly guard the parallel canonicalizer's prefix-max requirement.
   * The middle fragments are contained by the first; the final fragment only
   * touches that first fragment, not its immediate predecessor.
   */
  std::vector<DirectedSegmentI64> canonical_bridge = {
      {7, 0, 100, -1, SegmentAxis::vertical},
      {7, 10, 20, -1, SegmentAxis::vertical},
      {7, 30, 40, -1, SegmentAxis::vertical},
      {7, 100, 120, -1, SegmentAxis::vertical},
      {7, 10, 20, 1, SegmentAxis::vertical}};
  std::vector<DirectedSegmentI64> cpu_canonical_bridge = canonical_bridge;
  canonicalize_segments(&cpu_canonical_bridge);
  const auto gpu_canonical_bridge =
      gpu_canonicalize_for_test(canonical_bridge, device);
  const bool canonical_equal =
      cpu_canonical_bridge.size() == gpu_canonical_bridge.size() &&
      std::equal(
          cpu_canonical_bridge.begin(), cpu_canonical_bridge.end(),
          gpu_canonical_bridge.begin(),
          [](const DirectedSegmentI64 &first,
             const DirectedSegmentI64 &second) {
            return first.axis == second.axis &&
                   first.side == second.side &&
                   first.fixed == second.fixed &&
                   first.lo == second.lo && first.hi == second.hi;
          });
  if (cpu_canonical_bridge.size() != 2 ||
      !canonical_equal ||
      gpu_canonical_bridge[0].lo != 0 ||
      gpu_canonical_bridge[0].hi != 120) {
    throw std::runtime_error("prefix-max canonical bridge mismatch");
  }
  ++checks;
  std::cout
      << "MANHATTAN_UNION_CANONICAL ok containment_bridge=1 "
         "prefix_max=1\n";

  const std::array<RectI64, 4> invalid = {
      rectangle(0, 0, 0, 1), rectangle(2, 3, 1, 4),
      rectangle(0, 5, 1, 5), rectangle(0, 9, 1, 8)};
  for (std::size_t index = 0; index < invalid.size(); ++index) {
    const UnionOutput cpu = cpu_union({invalid[index]}, limits);
    const UnionOutput gpu = gpu_union({invalid[index]}, limits, device);
    if (!cpu.fallback || !gpu.fallback || !gpu.segments.empty())
      throw std::runtime_error("degenerate rectangle did not fail closed");
    ++checks;
  }
  std::cout << "MANHATTAN_UNION_DEGENERATE ok cases=4 fallback=1\n";

  Limits bounded = limits;
  bounded.max_slabs_per_rectangle = 2;
  const std::vector<RectI64> capacity = {
      rectangle(0, 0, 100, 10), rectangle(10, 20, 20, 30),
      rectangle(30, 20, 40, 30), rectangle(50, 20, 60, 30)};
  const UnionOutput gpu_capacity = gpu_union(capacity, bounded, device);
  if (!gpu_capacity.fallback || !gpu_capacity.segments.empty() ||
      gpu_capacity.message.find("slab capacity") == std::string::npos) {
    throw std::runtime_error("per-rectangle capacity did not fail closed");
  }
  ++checks;
  std::cout << "MANHATTAN_UNION_CAPACITY ok fallback=1 message='"
            << gpu_capacity.message << "'\n";

  std::cout << "MANHATTAN_UNION_SELF_TEST PASS checks=" << checks
            << " directed=" << directed_fixtures().size()
            << " random=64 canonical=1 degeneracy=4 capacity=1\n";
}

std::vector<RectI64> touching_grid(std::uint32_t dimension)
{
  std::uint64_t count = static_cast<std::uint64_t>(dimension) * dimension;
  if (count > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("grid is too large for host");
  std::vector<RectI64> rectangles;
  rectangles.reserve(static_cast<std::size_t>(count));
  constexpr std::int64_t pitch = 16;
  for (std::uint32_t y = 0; y < dimension; ++y) {
    for (std::uint32_t x = 0; x < dimension; ++x) {
      rectangles.push_back(
          rectangle(
              static_cast<std::int64_t>(x) * pitch,
              static_cast<std::int64_t>(y) * pitch,
              static_cast<std::int64_t>(x + 1) * pitch,
              static_cast<std::int64_t>(y + 1) * pitch,
              static_cast<std::uint64_t>(y) * dimension + x + 1));
    }
  }
  return rectangles;
}

double median(std::vector<double> values)
{
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2) return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

void print_gpu_timing(std::uint32_t run, const UnionOutput &output)
{
  std::cout << "MANHATTAN_UNION_GPU run=" << run
            << " total_ms=" << std::fixed << std::setprecision(3)
            << output.total_ms << " h2d_ms=" << output.h2d_ms
            << " x_membership_ms=" << output.x_membership_ms
            << " strip_scan_ms=" << output.strip_scan_ms
            << " boundary_ms=" << output.boundary_ms
            << " d2h_ms=" << output.d2h_ms
            << " memberships=" << output.memberships
            << " strips=" << output.strip_intervals
            << " raw_segments=" << output.raw_segments
            << " segments=" << output.segments.size() << "\n";
}

void run_benchmark(std::uint32_t dimension, std::uint32_t repeat,
                   int device)
{
  if (!dimension || repeat < 2)
    throw std::runtime_error(
        "benchmark requires a nonzero grid and at least two runs");
  const auto rectangles = touching_grid(dimension);
  Limits limits;
  limits.max_slabs_per_rectangle = std::max<std::uint32_t>(
      limits.max_slabs_per_rectangle, dimension + 1);
  const UnionOutput cpu = cpu_union(rectangles, limits);
  if (cpu.fallback)
    throw std::runtime_error("CPU benchmark fallback: " + cpu.message);
  std::cout << "MANHATTAN_UNION_CPU rectangles=" << rectangles.size()
            << " total_ms=" << std::fixed << std::setprecision(3)
            << cpu.total_ms << " memberships=" << cpu.memberships
            << " strips=" << cpu.strip_intervals
            << " raw_segments=" << cpu.raw_segments
            << " segments=" << cpu.segments.size()
            << " digest=0x" << std::hex << cpu.digest << std::dec << "\n";

  std::vector<double> warm_times;
  UnionOutput last;
  double cold_time = 0.0;
  for (std::uint32_t run = 0; run < repeat; ++run) {
    last = gpu_union(rectangles, limits, device);
    require_equal("benchmark-" + std::to_string(run), cpu, last);
    print_gpu_timing(run, last);
    if (run)
      warm_times.push_back(last.total_ms);
    else
      cold_time = last.total_ms;
  }
  const double warm_median = median(warm_times);
  const double time_reduction =
      100.0 * (cpu.total_ms - warm_median) / cpu.total_ms;
  const double throughput_gain =
      100.0 * (cpu.total_ms / warm_median - 1.0);
  std::cout << "MANHATTAN_UNION_BENCH PASS rectangles="
            << rectangles.size() << " grid=" << dimension << "x"
            << dimension << " cold_gpu_ms=" << std::fixed
            << std::setprecision(3)
            << cold_time
            << " warm_gpu_median_ms=" << warm_median
            << " cpu_ms=" << cpu.total_ms
            << " time_reduction_pct=" << time_reduction
            << " throughput_gain_pct=" << throughput_gain
            << " digest=0x" << std::hex << cpu.digest << std::dec << "\n";
}

std::uint32_t parse_u32(const char *value, const char *option)
{
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end || parsed > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error(std::string("invalid ") + option);
  return static_cast<std::uint32_t>(parsed);
}

void print_help(const char *program)
{
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --self-test              run exact directed/random/fallback gates\n"
      << "  --benchmark-grid N       union an N-by-N touching rectangle grid\n"
      << "  --repeat N               benchmark process-local runs (default 5)\n"
      << "  --device N               CUDA device (default 0)\n"
      << "  --help                   show this text\n";
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    bool self_test = argc == 1;
    std::uint32_t benchmark_grid = 0;
    std::uint32_t repeat = 5;
    int device = 0;
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--self-test") {
        self_test = true;
      } else if (option == "--benchmark-grid" && index + 1 < argc) {
        benchmark_grid = parse_u32(argv[++index], "--benchmark-grid");
      } else if (option == "--repeat" && index + 1 < argc) {
        repeat = parse_u32(argv[++index], "--repeat");
      } else if (option == "--device" && index + 1 < argc) {
        device = static_cast<int>(parse_u32(argv[++index], "--device"));
      } else if (option == "--help") {
        print_help(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown or incomplete option: " + option);
      }
    }
    if (self_test) run_self_test(device);
    if (benchmark_grid) run_benchmark(benchmark_grid, repeat, device);
    if (!self_test && !benchmark_grid)
      throw std::runtime_error("no action requested");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MANHATTAN_UNION_FAIL " << error.what() << "\n";
    return 1;
  }
}
