/*
 * Exact CUDA Manhattan rectangle-union core.
 *
 * This bounded implementation converts expanded int64 rectangles into exact
 * directed union-boundary segments with no raster grid and no floating point:
 *
 *   x endpoint sort -> rectangle/slab memberships -> y event sort/scan
 *   -> disjoint strip intervals -> vertical XOR scan -> segment compaction
 *
 * Every allocation, H2D/D2H transfer, sort, scan and compaction is included in
 * total_ms.  A capacity or invariant failure returns an explicit fallback;
 * partial boundary output is never published.
 */

#include "manhattan_union_format.cuh"
#include "manhattan_union_gpu.cuh"

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
#include <atomic>
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
#include <thread>
#include <utility>
#include <vector>

namespace {

using klayout_cuda::manhattan_union::DirectedSegmentI64;
using klayout_cuda::manhattan_union::GpuUnionLimits;
using klayout_cuda::manhattan_union::GpuUnionOutput;
using klayout_cuda::manhattan_union::RectI64;
using klayout_cuda::manhattan_union::ResidentStripHook;
using klayout_cuda::manhattan_union::SegmentAxis;
using klayout_cuda::manhattan_union::StripInterval;
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

using Limits = GpuUnionLimits;

using PackedEventKey = std::uint64_t;

using PackedTransition = std::uint64_t;
constexpr PackedTransition kInvalidTransition = UINT64_MAX;

struct HorizontalRun
{
  std::uint64_t line;
  std::uint32_t first_slab;
  std::uint32_t last_slab;
};

struct SegmentLine
{
  std::int64_t fixed;
  std::int32_t side;
  SegmentAxis axis;
};

static_assert(sizeof(SegmentLine) == 16, "unexpected segment-line padding");
static_assert(sizeof(HorizontalRun) == 16,
              "unexpected horizontal-run padding");

struct EventSlab
{
  MU_HD std::uint32_t operator()(PackedEventKey key) const
  {
    return static_cast<std::uint32_t>(key >> 32);
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

struct InvalidTransition
{
  MU_HD bool operator()(PackedTransition transition) const
  {
    return transition == kInvalidTransition;
  }
};

struct HorizontalRunFromKey
{
  MU_HD HorizontalRun operator()(std::uint64_t key) const
  {
    const std::uint32_t slab =
        static_cast<std::uint32_t>(key & UINT64_C(0x7fffffff));
    return {key >> 31, slab, slab + 1};
  }
};

struct HorizontalRunMerge
{
  MU_HD HorizontalRun operator()(const HorizontalRun &first,
                                 const HorizontalRun &second) const
  {
    return {
        first.line, min(first.first_slab, second.first_slab),
        max(first.last_slab, second.last_slab)};
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

struct CpuTransition
{
  std::int64_t y;
  std::uint32_t slab;
  std::int32_t kind;
};

using UnionOutput = GpuUnionOutput;

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

void sample_device_memory(UnionOutput *output)
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_require(
      cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  if (!output->device_free_begin_bytes) {
    output->device_free_begin_bytes = free_bytes;
    output->device_free_low_bytes = free_bytes;
    output->device_total_bytes = total_bytes;
  } else {
    output->device_free_low_bytes =
        std::min<std::uint64_t>(
            output->device_free_low_bytes, free_bytes);
  }
}

template <class T>
void release_device_vector(thrust::device_vector<T> *values)
{
  thrust::device_vector<T> empty;
  values->swap(empty);
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
  output.rectangle_count = rectangles.size();
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
  output.event_count = events.size();

  std::sort(events.begin(), events.end(),
            [](const CpuEvent &first, const CpuEvent &second) {
              if (first.slab != second.slab)
                return first.slab < second.slab;
              return first.y < second.y;
            });

  std::vector<CpuTransition> transitions;
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
    const CpuTransition &start = transitions[index];
    const CpuTransition &finish = transitions[index + 1];
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
  if (output.raw_segments > limits.max_raw_segments) {
    output.fallback = true;
    output.message = "raw segment capacity";
    output.total_ms = elapsed_ms(total_begin, Clock::now());
    return output;
  }
  canonicalize_segments(&raw);
  if (raw.size() > limits.max_segments) {
    output.fallback = true;
    output.message = "segment capacity";
    output.total_ms = elapsed_ms(total_begin, Clock::now());
    return output;
  }
  output.segments.swap(raw);
  output.digest = digest_segments(output.segments);
  output.total_ms = elapsed_ms(total_begin, Clock::now());
  return output;
}

__global__ void emit_x_endpoints_kernel(
    const RectI64 *rectangles, std::uint64_t rectangle_count,
    std::int64_t y_base, std::int64_t y_high,
    std::int64_t *endpoints, std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < rectangle_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const RectI64 rectangle = rectangles[index];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top ||
        rectangle.bottom < y_base || rectangle.top > y_high) {
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

MU_HD PackedEventKey pack_event_key(
    std::uint32_t slab, std::int64_t y, std::int64_t y_base)
{
  return (static_cast<std::uint64_t>(slab) << 32) |
         static_cast<std::uint32_t>(y - y_base);
}

MU_HD std::int64_t unpack_event_y(
    PackedEventKey key, std::int64_t y_base)
{
  return y_base + static_cast<std::uint32_t>(key);
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
    std::int64_t y_base, const std::uint64_t *offsets,
    PackedEventKey *keys, std::int32_t *deltas)
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
      keys[output] = pack_event_key(
          static_cast<std::uint32_t>(slab), rectangle.bottom, y_base);
      deltas[output++] = 1;
      keys[output] = pack_event_key(
          static_cast<std::uint32_t>(slab), rectangle.top, y_base);
      deltas[output++] = -1;
    }
  }
}

__global__ void extract_packed_transitions_kernel(
    const PackedEventKey *keys, const std::int32_t *coverage,
    std::uint64_t event_count, PackedTransition *transitions,
    std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < event_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::int32_t after = coverage[index];
    const std::uint32_t slab =
        static_cast<std::uint32_t>(keys[index] >> 32);
    const std::int32_t before =
        index &&
                static_cast<std::uint32_t>(keys[index - 1] >> 32) == slab
            ? coverage[index - 1]
            : 0;
    PackedTransition transition = kInvalidTransition;
    if (after < 0) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusCoverageInvariant));
    } else if (!before && after > 0) {
      transition = keys[index];
    } else if (before > 0 && !after) {
      transition = keys[index] | (UINT64_C(1) << 63);
    }
    if ((index + 1 == event_count ||
         static_cast<std::uint32_t>(keys[index + 1] >> 32) != slab) &&
        after != 0) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusCoverageInvariant));
    }
    transitions[index] = transition;
  }
}

__global__ void build_intervals_kernel(
    const PackedTransition *transitions, std::uint64_t transition_count,
    std::int64_t y_base, StripInterval *intervals,
    std::uint32_t *status)
{
  for (std::uint64_t pair =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       pair < transition_count / 2;
       pair += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const PackedTransition first = transitions[pair * 2];
    const PackedTransition second = transitions[pair * 2 + 1];
    const bool first_end = (first >> 63) != 0;
    const bool second_end = (second >> 63) != 0;
    const std::uint32_t first_slab =
        static_cast<std::uint32_t>((first >> 32) & UINT64_C(0x7fffffff));
    const std::uint32_t second_slab =
        static_cast<std::uint32_t>((second >> 32) & UINT64_C(0x7fffffff));
    const std::int64_t first_y = unpack_event_y(first, y_base);
    const std::int64_t second_y = unpack_event_y(second, y_base);
    if (first_end || !second_end || first_slab != second_slab ||
        first_y >= second_y) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusTransitionInvariant));
      continue;
    }
    intervals[pair] = {first_y, second_y, first_slab, 0};
  }
}

__global__ void make_horizontal_keys_kernel(
    PackedTransition *transitions, std::uint64_t transition_count)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < transition_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const PackedTransition transition = transitions[index];
    const std::uint64_t side = transition >> 63;
    const std::uint32_t slab = static_cast<std::uint32_t>(
        (transition >> 32) & UINT64_C(0x7fffffff));
    const std::uint32_t y_offset =
        static_cast<std::uint32_t>(transition);
    transitions[index] =
        (side << 63) | (static_cast<std::uint64_t>(y_offset) << 31) |
        slab;
  }
}

__global__ void mark_horizontal_groups_kernel(
    const std::uint64_t *keys, std::uint64_t key_count,
    std::uint32_t *groups)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < key_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (!index) {
      groups[index] = 1;
      continue;
    }
    const std::uint64_t previous = keys[index - 1];
    const std::uint64_t current = keys[index];
    const bool same_line = (previous >> 31) == (current >> 31);
    const std::uint32_t previous_slab =
        static_cast<std::uint32_t>(previous & UINT64_C(0x7fffffff));
    const std::uint32_t current_slab =
        static_cast<std::uint32_t>(current & UINT64_C(0x7fffffff));
    groups[index] =
        !(same_line && current_slab == previous_slab + 1);
  }
}

__global__ void horizontal_runs_to_segments_kernel(
    const HorizontalRun *runs, std::uint64_t run_count,
    std::int64_t y_base, const std::int64_t *xs,
    DirectedSegmentI64 *segments)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < run_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const HorizontalRun run = runs[index];
    const std::int32_t side = (run.line >> 32) ? 1 : -1;
    const std::uint32_t y_offset =
        static_cast<std::uint32_t>(run.line);
    segments[index] = {
        y_base + y_offset, xs[run.first_slab], xs[run.last_slab],
        side, SegmentAxis::horizontal};
  }
}

__global__ void count_intervals_per_slab_kernel(
    const StripInterval *intervals, std::uint64_t interval_count,
    std::uint32_t *counts)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < interval_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    atomicAdd(counts + intervals[index].slab, 1u);
  }
}

__device__ std::uint64_t interval_difference_count(
    const StripInterval *primary, std::uint32_t primary_count,
    const StripInterval *mask, std::uint32_t mask_count)
{
  std::uint64_t output_count = 0;
  std::uint32_t mask_begin = 0;
  for (std::uint32_t primary_id = 0;
       primary_id < primary_count; ++primary_id) {
    std::int64_t lo = primary[primary_id].bottom;
    const std::int64_t hi = primary[primary_id].top;
    while (mask_begin < mask_count &&
           mask[mask_begin].top <= lo) {
      ++mask_begin;
    }
    std::uint32_t mask_id = mask_begin;
    while (mask_id < mask_count && mask[mask_id].bottom < hi) {
      if (mask[mask_id].bottom > lo) ++output_count;
      lo = max(lo, mask[mask_id].top);
      if (lo >= hi) break;
      ++mask_id;
    }
    if (lo < hi) ++output_count;
    mask_begin = mask_id;
  }
  return output_count;
}

__device__ std::uint64_t emit_interval_difference(
    const StripInterval *primary, std::uint32_t primary_count,
    const StripInterval *mask, std::uint32_t mask_count,
    std::int64_t fixed, std::int32_t side,
    DirectedSegmentI64 *output)
{
  std::uint64_t output_count = 0;
  std::uint32_t mask_begin = 0;
  for (std::uint32_t primary_id = 0;
       primary_id < primary_count; ++primary_id) {
    std::int64_t lo = primary[primary_id].bottom;
    const std::int64_t hi = primary[primary_id].top;
    while (mask_begin < mask_count &&
           mask[mask_begin].top <= lo) {
      ++mask_begin;
    }
    std::uint32_t mask_id = mask_begin;
    while (mask_id < mask_count && mask[mask_id].bottom < hi) {
      const std::int64_t end = min(mask[mask_id].bottom, hi);
      if (lo < end) {
        output[output_count++] = {
            fixed, lo, end, side, SegmentAxis::vertical};
      }
      lo = max(lo, mask[mask_id].top);
      if (lo >= hi) break;
      ++mask_id;
    }
    if (lo < hi) {
      output[output_count++] = {
          fixed, lo, hi, side, SegmentAxis::vertical};
    }
    mask_begin = mask_id;
  }
  return output_count;
}

__global__ void count_vertical_xor_kernel(
    const StripInterval *intervals, const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, std::uint32_t x_slabs,
    std::uint64_t *vertical_counts)
{
  const StripInterval *empty = intervals;
  for (std::uint64_t boundary =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       boundary <= x_slabs;
       boundary += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const bool has_left = boundary > 0;
    const bool has_right = boundary < x_slabs;
    const StripInterval *left =
        has_left ? intervals + slab_offsets[boundary - 1] : empty;
    const std::uint32_t left_count =
        has_left ? slab_counts[boundary - 1] : 0;
    const StripInterval *right =
        has_right ? intervals + slab_offsets[boundary] : empty;
    const std::uint32_t right_count =
        has_right ? slab_counts[boundary] : 0;
    vertical_counts[boundary] =
        interval_difference_count(
            left, left_count, right, right_count) +
        interval_difference_count(
            right, right_count, left, left_count);
  }
}

__global__ void emit_vertical_xor_kernel(
    const StripInterval *intervals, const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, std::uint32_t x_slabs,
    const std::int64_t *xs, const std::uint64_t *vertical_counts,
    const std::uint64_t *vertical_offsets, DirectedSegmentI64 *vertical,
    std::uint32_t *status)
{
  const StripInterval *empty = intervals;
  for (std::uint64_t boundary =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       boundary <= x_slabs;
       boundary += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const bool has_left = boundary > 0;
    const bool has_right = boundary < x_slabs;
    const StripInterval *left =
        has_left ? intervals + slab_offsets[boundary - 1] : empty;
    const std::uint32_t left_count =
        has_left ? slab_counts[boundary - 1] : 0;
    const StripInterval *right =
        has_right ? intervals + slab_offsets[boundary] : empty;
    const std::uint32_t right_count =
        has_right ? slab_counts[boundary] : 0;
    DirectedSegmentI64 *output = vertical + vertical_offsets[boundary];
    const std::uint64_t left_only = emit_interval_difference(
        left, left_count, right, right_count, xs[boundary], 1, output);
    const std::uint64_t right_only = emit_interval_difference(
        right, right_count, left, left_count, xs[boundary], -1,
        output + left_only);
    if (left_only + right_only != vertical_counts[boundary]) {
      atomicOr(status,
               static_cast<std::uint32_t>(kStatusTransitionInvariant));
    }
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

struct PreparedDeviceRectangles
{
  thrust::device_vector<RectI64> rectangles;
  double input_ms = 0.0;
};

template <class RectangleFactory>
UnionOutput gpu_union_prepared(
    std::uint64_t rectangle_count, std::int64_t y_base,
    std::int64_t y_high, const Limits &limits, int device,
    Clock::time_point total_begin, double input_prepare_ms,
    const ResidentStripHook *resident_hook,
    RectangleFactory rectangle_factory)
{
  UnionOutput output;
  output.rectangle_count = rectangle_count;
  output.input_prepare_ms = input_prepare_ms;
  auto pipeline = [&]() {
    if (!std::isfinite(input_prepare_ms) || input_prepare_ms < 0.0) {
      output.fallback = true;
      output.message = "invalid input preparation timing";
      return;
    }
    if (rectangle_count > limits.max_rectangles) {
      output.fallback = true;
      output.message = "rectangle capacity";
      return;
    }
    if (!rectangle_count) {
      output.digest = digest_segments(output.segments);
      return;
    }
    if (y_base >= y_high ||
        static_cast<__int128>(y_high) -
            static_cast<__int128>(y_base) >
        static_cast<__int128>(
            std::numeric_limits<std::uint32_t>::max())) {
      output.fallback = true;
      output.message = "packed y-coordinate range";
      return;
    }

    cuda_require(cudaSetDevice(device), "cudaSetDevice");
    sample_device_memory(&output);
    thrust::device_vector<std::uint32_t> status(1, 0);

    PreparedDeviceRectangles prepared = rectangle_factory();
    output.h2d_ms = prepared.input_ms;
    if (prepared.rectangles.size() != rectangle_count) {
      output.fallback = true;
      output.message = "prepared rectangle count";
      return;
    }
    thrust::device_vector<RectI64> device_rectangles =
        std::move(prepared.rectangles);

    const auto x_begin = Clock::now();
    std::uint64_t endpoint_count = 0;
    if (!checked_multiply(rectangle_count, UINT64_C(2),
                          &endpoint_count)) {
      output.fallback = true;
      output.message = "endpoint count overflow";
      return;
    }
    thrust::device_vector<std::int64_t> xs(endpoint_count);
    const std::uint32_t rectangle_blocks =
        launch_blocks(rectangle_count);
    emit_x_endpoints_kernel<<<rectangle_blocks, kThreads>>>(
        thrust::raw_pointer_cast(device_rectangles.data()),
        rectangle_count, y_base, y_high,
        thrust::raw_pointer_cast(xs.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "emit x endpoints");
    thrust::sort(thrust::device, xs.begin(), xs.end());
    const auto x_end =
        thrust::unique(thrust::device, xs.begin(), xs.end());
    const std::uint64_t x_count = x_end - xs.begin();
    xs.resize(x_count);
    xs.shrink_to_fit();
    if (x_count < 2) {
      output.fallback = true;
      output.message = "fewer than two x endpoints";
      return;
    }
    output.x_slabs = x_count - 1;
    if (output.x_slabs > limits.max_x_slabs ||
        output.x_slabs > UINT64_C(0x7fffffff)) {
      output.fallback = true;
      output.message = "x-slab capacity";
      return;
    }

    thrust::device_vector<std::uint64_t> membership_counts(
        rectangle_count);
    thrust::device_vector<std::uint64_t> membership_offsets(
        rectangle_count);
    count_memberships_kernel<<<rectangle_blocks, kThreads>>>(
        thrust::raw_pointer_cast(device_rectangles.data()),
        rectangle_count, thrust::raw_pointer_cast(xs.data()), x_count,
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
                rectangle_count - 1,
            sizeof(last_offset), cudaMemcpyDeviceToHost),
        "last membership offset D2H");
    cuda_require(
        cudaMemcpy(
            &last_count,
            thrust::raw_pointer_cast(membership_counts.data()) +
                rectangle_count - 1,
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
    output.event_count = event_count;
    cuda_require(cudaDeviceSynchronize(), "x/membership synchronize");
    output.x_membership_ms = elapsed_ms(x_begin, Clock::now());

    const auto strip_begin = Clock::now();
    thrust::device_vector<PackedEventKey> event_keys(event_count);
    thrust::device_vector<std::int32_t> event_deltas(event_count);
    fill_events_kernel<<<rectangle_blocks, kThreads>>>(
        thrust::raw_pointer_cast(device_rectangles.data()),
        rectangle_count, thrust::raw_pointer_cast(xs.data()), x_count,
        y_base,
        thrust::raw_pointer_cast(membership_offsets.data()),
        thrust::raw_pointer_cast(event_keys.data()),
        thrust::raw_pointer_cast(event_deltas.data()));
    cuda_require(cudaGetLastError(), "fill slab events");
    thrust::sort_by_key(
        thrust::device, event_keys.begin(), event_keys.end(),
        event_deltas.begin());
    sample_device_memory(&output);
    release_device_vector(&device_rectangles);
    release_device_vector(&membership_counts);
    release_device_vector(&membership_offsets);

    thrust::device_vector<PackedEventKey> unique_event_keys(event_count);
    thrust::device_vector<std::int32_t> unique_event_deltas(event_count);
    auto reduced_events = thrust::reduce_by_key(
        thrust::device, event_keys.begin(), event_keys.end(),
        event_deltas.begin(), unique_event_keys.begin(),
        unique_event_deltas.begin(), thrust::equal_to<PackedEventKey>{},
        thrust::plus<std::int32_t>{});
    sample_device_memory(&output);
    std::uint64_t unique_event_count =
        reduced_events.first - unique_event_keys.begin();
    release_device_vector(&event_keys);
    release_device_vector(&event_deltas);
    auto zipped_events = thrust::make_zip_iterator(
        thrust::make_tuple(unique_event_keys.begin(),
                           unique_event_deltas.begin()));
    const auto nonzero_event_end = thrust::remove_if(
        thrust::device, zipped_events, zipped_events + unique_event_count,
        ZeroEventDelta{});
    unique_event_count = nonzero_event_end - zipped_events;
    unique_event_keys.resize(unique_event_count);
    unique_event_deltas.resize(unique_event_count);
    unique_event_keys.shrink_to_fit();
    unique_event_deltas.shrink_to_fit();
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

    thrust::device_vector<PackedTransition> transitions(unique_event_count);
    extract_packed_transitions_kernel<<<
        launch_blocks(unique_event_count), kThreads>>>(
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
    release_device_vector(&unique_event_keys);
    release_device_vector(&unique_event_deltas);
    release_device_vector(&coverage);
    transitions.shrink_to_fit();
    if (transition_count > limits.max_raw_segments) {
      output.fallback = true;
      output.message = "raw segment capacity";
      return;
    }
    if (transition_count % 2) {
      output.fallback = true;
      output.message = "odd transition count";
      return;
    }
    output.strip_intervals = transition_count / 2;
    if (transition_count > std::numeric_limits<std::uint32_t>::max() ||
        output.strip_intervals >
            std::numeric_limits<std::uint32_t>::max()) {
      output.fallback = true;
      output.message = "transition index capacity";
      return;
    }
    thrust::device_vector<StripInterval> intervals(
        output.strip_intervals);
    if (output.strip_intervals) {
      build_intervals_kernel<<<
          launch_blocks(output.strip_intervals), kThreads>>>(
          thrust::raw_pointer_cast(transitions.data()), transition_count,
          y_base,
          thrust::raw_pointer_cast(intervals.data()),
          thrust::raw_pointer_cast(status.data()));
      cuda_require(cudaGetLastError(), "build strip intervals");
    }

    auto invoke_resident_consumer = [&](
        const thrust::device_vector<std::uint64_t> &slab_offsets,
        const thrust::device_vector<std::uint32_t> &slab_counts) {
      cudaStream_t stream = nullptr;
      resident_hook->consume(
          stream, thrust::raw_pointer_cast(xs.data()),
          static_cast<std::uint32_t>(output.x_slabs),
          thrust::raw_pointer_cast(intervals.data()),
          output.strip_intervals,
          thrust::raw_pointer_cast(slab_offsets.data()),
          thrust::raw_pointer_cast(slab_counts.data()),
          resident_hook->context);
      cuda_require(cudaGetLastError(), "resident strip consumer");
      cuda_require(
          cudaStreamSynchronize(stream),
          "resident strip consumer synchronize");
      sample_device_memory(&output);
      output.resident_consumer_completed = true;
    };

    /*
     * A proof consumer that does not need a host boundary can avoid all
     * horizontal compaction and vertical XOR materialization.  Validate the
     * strip status first, then expose only synchronous, stream-ordered views.
     */
    if (resident_hook && resident_hook->consume &&
        resident_hook->stop_before_boundary) {
      cuda_require(
          cudaDeviceSynchronize(),
          "pre-consumer strip sweep synchronize");
      const std::uint32_t strip_status = copy_device_status(status);
      if (strip_status) {
        output.fallback = true;
        output.message = status_message(strip_status);
        return;
      }
      release_device_vector(&transitions);
      thrust::device_vector<std::uint32_t> slab_interval_counts(
          output.x_slabs, 0);
      thrust::device_vector<std::uint64_t> slab_interval_offsets(
          output.x_slabs);
      if (output.strip_intervals) {
        count_intervals_per_slab_kernel<<<
            launch_blocks(output.strip_intervals), kThreads>>>(
            thrust::raw_pointer_cast(intervals.data()),
            output.strip_intervals,
            thrust::raw_pointer_cast(slab_interval_counts.data()));
        cuda_require(
            cudaGetLastError(), "count strip intervals per slab");
      }
      thrust::exclusive_scan(
          thrust::device, slab_interval_counts.begin(),
          slab_interval_counts.end(), slab_interval_offsets.begin(),
          std::uint64_t{0});
      invoke_resident_consumer(
          slab_interval_offsets, slab_interval_counts);
      output.strip_scan_ms = elapsed_ms(strip_begin, Clock::now());
      return;
    }

    make_horizontal_keys_kernel<<<
        launch_blocks(transition_count), kThreads>>>(
        thrust::raw_pointer_cast(transitions.data()), transition_count);
    cuda_require(cudaGetLastError(), "pack horizontal transition keys");
    thrust::sort(
        thrust::device, transitions.begin(), transitions.end());
    thrust::device_vector<std::uint32_t> horizontal_groups(
        transition_count);
    mark_horizontal_groups_kernel<<<
        launch_blocks(transition_count), kThreads>>>(
        thrust::raw_pointer_cast(transitions.data()), transition_count,
        thrust::raw_pointer_cast(horizontal_groups.data()));
    cuda_require(cudaGetLastError(), "mark horizontal groups");
    thrust::inclusive_scan(
        thrust::device, horizontal_groups.begin(),
        horizontal_groups.end(), horizontal_groups.begin());
    std::uint32_t horizontal_count_u32 = 0;
    cuda_require(
        cudaMemcpy(
            &horizontal_count_u32,
            thrust::raw_pointer_cast(horizontal_groups.data()) +
                transition_count - 1,
            sizeof(horizontal_count_u32), cudaMemcpyDeviceToHost),
        "horizontal group count D2H");
    const std::uint64_t horizontal_count = horizontal_count_u32;
    if (horizontal_count > limits.max_segments) {
      output.fallback = true;
      output.message = "horizontal segment capacity";
      return;
    }
    thrust::device_vector<std::uint32_t> compact_group_ids(
        horizontal_count);
    thrust::device_vector<HorizontalRun> horizontal_runs(
        horizontal_count);
    const auto run_values = thrust::make_transform_iterator(
        transitions.begin(), HorizontalRunFromKey{});
    const auto horizontal_reduce_end = thrust::reduce_by_key(
        thrust::device, horizontal_groups.begin(),
        horizontal_groups.end(), run_values, compact_group_ids.begin(),
        horizontal_runs.begin(), thrust::equal_to<std::uint32_t>{},
        HorizontalRunMerge{});
    if (static_cast<std::uint64_t>(
            horizontal_reduce_end.second - horizontal_runs.begin()) !=
        horizontal_count) {
      output.fallback = true;
      output.message = "horizontal group reduction mismatch";
      return;
    }
    thrust::device_vector<DirectedSegmentI64> horizontal(
        horizontal_count);
    if (horizontal_count) {
      horizontal_runs_to_segments_kernel<<<
          launch_blocks(horizontal_count), kThreads>>>(
          thrust::raw_pointer_cast(horizontal_runs.data()),
          horizontal_count, y_base, thrust::raw_pointer_cast(xs.data()),
          thrust::raw_pointer_cast(horizontal.data()));
      cuda_require(cudaGetLastError(), "emit canonical horizontal segments");
    }
    sample_device_memory(&output);
    cuda_require(cudaDeviceSynchronize(), "strip sweep synchronize");
    const std::uint32_t strip_status = copy_device_status(status);
    if (strip_status) {
      output.fallback = true;
      output.message = status_message(strip_status);
      return;
    }
    release_device_vector(&transitions);
    release_device_vector(&horizontal_groups);
    release_device_vector(&compact_group_ids);
    release_device_vector(&horizontal_runs);
    output.strip_scan_ms = elapsed_ms(strip_begin, Clock::now());

    const auto boundary_begin = Clock::now();
    thrust::device_vector<std::uint32_t> slab_interval_counts(
        output.x_slabs, 0);
    thrust::device_vector<std::uint64_t> slab_interval_offsets(
        output.x_slabs);
    if (output.strip_intervals) {
      count_intervals_per_slab_kernel<<<
          launch_blocks(output.strip_intervals), kThreads>>>(
          thrust::raw_pointer_cast(intervals.data()),
          output.strip_intervals,
          thrust::raw_pointer_cast(slab_interval_counts.data()));
      cuda_require(cudaGetLastError(), "count strip intervals per slab");
    }
    thrust::exclusive_scan(
        thrust::device, slab_interval_counts.begin(),
        slab_interval_counts.end(), slab_interval_offsets.begin(),
        std::uint64_t{0});

    if (resident_hook && resident_hook->consume) {
      invoke_resident_consumer(
          slab_interval_offsets, slab_interval_counts);
      if (resident_hook->stop_before_boundary) {
        return;
      }
    }

    const std::uint64_t boundary_count = output.x_slabs + 1;
    thrust::device_vector<std::uint64_t> vertical_counts(boundary_count);
    thrust::device_vector<std::uint64_t> vertical_offsets(boundary_count);
    count_vertical_xor_kernel<<<launch_blocks(boundary_count), kThreads>>>(
        thrust::raw_pointer_cast(intervals.data()),
        thrust::raw_pointer_cast(slab_interval_offsets.data()),
        thrust::raw_pointer_cast(slab_interval_counts.data()),
        static_cast<std::uint32_t>(output.x_slabs),
        thrust::raw_pointer_cast(vertical_counts.data()));
    cuda_require(cudaGetLastError(), "count adjacent-slab vertical XOR");
    thrust::exclusive_scan(
        thrust::device, vertical_counts.begin(), vertical_counts.end(),
        vertical_offsets.begin(), std::uint64_t{0});
    std::uint64_t final_vertical_offset = 0;
    std::uint64_t final_vertical_count = 0;
    cuda_require(
        cudaMemcpy(
            &final_vertical_offset,
            thrust::raw_pointer_cast(vertical_offsets.data()) +
                boundary_count - 1,
            sizeof(final_vertical_offset), cudaMemcpyDeviceToHost),
        "last vertical offset D2H");
    cuda_require(
        cudaMemcpy(
            &final_vertical_count,
            thrust::raw_pointer_cast(vertical_counts.data()) +
                boundary_count - 1,
            sizeof(final_vertical_count), cudaMemcpyDeviceToHost),
        "last vertical count D2H");
    if (final_vertical_count >
        std::numeric_limits<std::uint64_t>::max() -
            final_vertical_offset) {
      output.fallback = true;
      output.message = "vertical segment count overflow";
      return;
    }
    const std::uint64_t vertical_count =
        final_vertical_offset + final_vertical_count;
    if (vertical_count >
        limits.max_raw_segments - transition_count) {
      output.fallback = true;
      output.message = "raw segment capacity";
      return;
    }
    if (vertical_count > limits.max_segments - horizontal_count) {
      output.fallback = true;
      output.message = "segment capacity";
      return;
    }
    thrust::device_vector<DirectedSegmentI64> vertical(vertical_count);
    if (vertical_count) {
      emit_vertical_xor_kernel<<<launch_blocks(boundary_count), kThreads>>>(
          thrust::raw_pointer_cast(intervals.data()),
          thrust::raw_pointer_cast(slab_interval_offsets.data()),
          thrust::raw_pointer_cast(slab_interval_counts.data()),
          static_cast<std::uint32_t>(output.x_slabs),
          thrust::raw_pointer_cast(xs.data()),
          thrust::raw_pointer_cast(vertical_counts.data()),
          thrust::raw_pointer_cast(vertical_offsets.data()),
          thrust::raw_pointer_cast(vertical.data()),
          thrust::raw_pointer_cast(status.data()));
      cuda_require(cudaGetLastError(), "emit adjacent-slab vertical XOR");
    }
    sample_device_memory(&output);
    thrust::sort(
        thrust::device, vertical.begin(), vertical.end(), SegmentLess{});
    cuda_require(cudaDeviceSynchronize(), "vertical XOR synchronize");
    const std::uint32_t xor_status = copy_device_status(status);
    if (xor_status) {
      output.fallback = true;
      output.message = status_message(xor_status);
      return;
    }
    release_device_vector(&intervals);
    release_device_vector(&slab_interval_counts);
    release_device_vector(&slab_interval_offsets);
    release_device_vector(&vertical_counts);
    release_device_vector(&vertical_offsets);
    release_device_vector(&xs);

    std::uint64_t canonical_count = 0;
    if (horizontal_count >
            std::numeric_limits<std::uint64_t>::max() - vertical_count) {
      output.fallback = true;
      output.message = "canonical segment count overflow";
      return;
    }
    canonical_count = horizontal_count + vertical_count;
    if (canonical_count > limits.max_segments) {
      output.fallback = true;
      output.message = "segment capacity";
      return;
    }
    if (transition_count >
        std::numeric_limits<std::uint64_t>::max() - vertical_count) {
      output.fallback = true;
      output.message = "raw segment count overflow";
      return;
    }
    output.raw_segments = transition_count + vertical_count;
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
    if (horizontal_count) {
      cuda_require(
          cudaMemcpy(
              output.segments.data(),
              thrust::raw_pointer_cast(horizontal.data()),
              horizontal_count * sizeof(DirectedSegmentI64),
              cudaMemcpyDeviceToHost),
          "horizontal boundary D2H");
    }
    if (vertical_count) {
      cuda_require(
          cudaMemcpy(
              output.segments.data() + horizontal_count,
              thrust::raw_pointer_cast(vertical.data()),
              vertical_count * sizeof(DirectedSegmentI64),
              cudaMemcpyDeviceToHost),
          "vertical boundary D2H");
    }
    cuda_require(cudaDeviceSynchronize(), "boundary D2H synchronize");
    if (!std::is_sorted(
            output.segments.begin(), output.segments.end(),
            SegmentLess{})) {
      output.fallback = true;
      output.message = "noncanonical boundary order";
      output.segments.clear();
      return;
    }
    for (std::size_t index = 1; index < output.segments.size(); ++index) {
      const DirectedSegmentI64 &previous = output.segments[index - 1];
      const DirectedSegmentI64 &current = output.segments[index];
      if (same_segment_line(previous, current) &&
          current.lo <= previous.hi) {
        output.fallback = true;
        output.message = "nonmaximal canonical boundary";
        output.segments.clear();
        return;
      }
    }
    output.d2h_ms = elapsed_ms(d2h_begin, Clock::now());
    output.digest = digest_segments(output.segments);
  };

  try {
    pipeline();
  } catch (const std::exception &error) {
    output.fallback = true;
    output.message = std::string("CUDA pipeline exception: ") + error.what();
    output.segments.clear();
    (void)cudaGetLastError();
  }
  /*
   * All device_vector destructors above run before total_ms is sampled because
   * they are scoped inside the pipeline lambda.  This intentionally charges
   * allocation and teardown as well as transfers and kernels.
   */
  output.total_ms = elapsed_ms(total_begin, Clock::now());
  output.charged_total_ms =
      output.total_ms + output.input_prepare_ms;
  return output;
}

UnionOutput rejected_union(
    Clock::time_point total_begin, const std::string &message,
    std::uint64_t rectangle_count)
{
  UnionOutput output;
  output.rectangle_count = rectangle_count;
  output.fallback = true;
  output.message = message;
  output.total_ms = elapsed_ms(total_begin, Clock::now());
  output.charged_total_ms = output.total_ms;
  return output;
}

UnionOutput gpu_union(const std::vector<RectI64> &rectangles,
                      const Limits &limits, int device,
                      const ResidentStripHook *resident_hook = nullptr)
{
  const auto total_begin = Clock::now();
  if (rectangles.size() > limits.max_rectangles) {
    return rejected_union(
        total_begin, "rectangle capacity", rectangles.size());
  }
  if (rectangles.empty()) {
    return gpu_union_prepared(
        0, 0, 0, limits, device, total_begin, 0.0, resident_hook,
        []() { return PreparedDeviceRectangles{}; });
  }

  std::int64_t y_base = rectangles.front().bottom;
  std::int64_t y_high = rectangles.front().top;
  for (const RectI64 &rectangle : rectangles) {
    if (!valid_rectangle(rectangle)) {
      return rejected_union(
          total_begin, "invalid or degenerate rectangle",
          rectangles.size());
    }
    y_base = std::min(y_base, rectangle.bottom);
    y_high = std::max(y_high, rectangle.top);
  }

  return gpu_union_prepared(
      rectangles.size(), y_base, y_high, limits, device, total_begin,
      0.0, resident_hook,
      [&]() {
        const auto h2d_begin = Clock::now();
        PreparedDeviceRectangles prepared;
        prepared.rectangles.resize(rectangles.size());
        cuda_require(
            cudaMemcpy(
                thrust::raw_pointer_cast(prepared.rectangles.data()),
                rectangles.data(),
                rectangles.size() * sizeof(RectI64),
                cudaMemcpyHostToDevice),
            "rectangle H2D");
        cuda_require(
            cudaDeviceSynchronize(), "rectangle H2D synchronize");
        prepared.input_ms = elapsed_ms(h2d_begin, Clock::now());
        return prepared;
      });
}

UnionOutput gpu_union_resident_impl(
    thrust::device_vector<RectI64> &&rectangles,
    std::int64_t y_base, std::int64_t y_high,
    const Limits &limits, int device, double input_prepare_ms,
    const ResidentStripHook *resident_hook)
{
  const auto total_begin = Clock::now();
  const std::uint64_t rectangle_count = rectangles.size();
  return gpu_union_prepared(
      rectangle_count, y_base, y_high, limits, device, total_begin,
      input_prepare_ms, resident_hook,
      [&]() {
        PreparedDeviceRectangles prepared;
        prepared.rectangles = std::move(rectangles);
        return prepared;
      });
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

}  // namespace

namespace klayout_cuda {
namespace manhattan_union {

GpuUnionOutput gpu_union_host(
    const std::vector<RectI64> &rectangles,
    const GpuUnionLimits &limits, int device,
    const ResidentStripHook *resident_hook)
{
  return ::gpu_union(rectangles, limits, device, resident_hook);
}

GpuUnionOutput gpu_union_resident(
    thrust::device_vector<RectI64> &&rectangles,
    std::int64_t y_base, std::int64_t y_high,
    const GpuUnionLimits &limits, int device,
    double input_prepare_ms,
    const ResidentStripHook *resident_hook)
{
  return ::gpu_union_resident_impl(
      std::move(rectangles), y_base, y_high, limits, device,
      input_prepare_ms, resident_hook);
}

std::vector<DirectedSegmentI64> gpu_canonicalize_segments_for_test(
    const std::vector<DirectedSegmentI64> &raw, int device)
{
  return ::gpu_canonicalize_for_test(raw, device);
}

GpuUnionOutput cpu_union_reference_for_test(
    const std::vector<RectI64> &rectangles,
    const GpuUnionLimits &limits)
{
  return ::cpu_union(rectangles, limits);
}

void canonicalize_segments_reference_for_test(
    std::vector<DirectedSegmentI64> *segments)
{
  ::canonicalize_segments(segments);
}

}  // namespace manhattan_union
}  // namespace klayout_cuda
