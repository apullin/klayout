/*
 * Exact bounded resident-strip F90/F270 morphology.
 *
 * This is a reusable CUDA module.  It consumes the shared Manhattan union
 * core's explicit stream-ordered strip views and never includes an executable
 * translation unit.  The separately linked replay driver owns fixtures,
 * production input loading and golden comparison.
 */

#include "m2_resident_morphology_gpu.cuh"

#include "m1_width_space_exact_predicate.h"

#include <cuda_runtime.h>

#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

namespace morph = klayout_cuda::m2_resident_morphology;
namespace mu = klayout_cuda::manhattan_union;
namespace m1ws = klayout_cuda::m1_width_space;

using mu::DirectedSegmentI64;
using mu::SegmentAxis;
using mu::StripInterval;
using Clock = std::chrono::steady_clock;

#define MU_HD __host__ __device__

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMorphMaxActiveSlabs = 128;

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
  if (!count) return 0;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          (count + kThreads - 1) / kThreads, UINT64_C(65535)));
}

template <class T>
void release_device_vector(thrust::device_vector<T> *values)
{
  thrust::device_vector<T> empty;
  values->swap(empty);
}

std::uint32_t copy_device_status(
    const thrust::device_vector<std::uint32_t> &status,
    cudaStream_t stream)
{
  if (status.size() != 1) {
    throw std::runtime_error("invalid device status vector");
  }
  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpyAsync(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost, stream),
      "status D2H");
  cuda_require(cudaStreamSynchronize(stream), "status synchronize");
  return host_status;
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
  for (const DirectedSegmentI64 &segment : segments) {
    mix(static_cast<std::uint32_t>(segment.axis));
    mix(static_cast<std::uint32_t>(segment.side));
    mix(static_cast<std::uint64_t>(segment.fixed));
    mix(static_cast<std::uint64_t>(segment.lo));
    mix(static_cast<std::uint64_t>(segment.hi));
  }
  return hash;
}

struct SegmentLine
{
  std::int64_t fixed;
  std::int32_t side;
  SegmentAxis axis;
};

static_assert(sizeof(SegmentLine) == 16,
              "unexpected segment-line padding");

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

__device__ std::uint64_t interval_difference_count(
    const StripInterval *primary, std::uint32_t primary_count,
    const StripInterval *mask, std::uint32_t mask_count)
{
  std::uint64_t output_count = 0;
  std::uint32_t mask_begin = 0;
  for (std::uint32_t primary_id = 0; primary_id < primary_count;
       ++primary_id) {
    std::int64_t lo = primary[primary_id].bottom;
    const std::int64_t hi = primary[primary_id].top;
    while (mask_begin < mask_count && mask[mask_begin].top <= lo) {
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
  for (std::uint32_t primary_id = 0; primary_id < primary_count;
       ++primary_id) {
    std::int64_t lo = primary[primary_id].bottom;
    const std::int64_t hi = primary[primary_id].top;
    while (mask_begin < mask_count && mask[mask_begin].top <= lo) {
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

enum BoundaryStatus : std::uint32_t
{
  kBoundaryEmitMismatch = 1u << 0,
};

__global__ void emit_vertical_xor_kernel(
    const StripInterval *intervals, const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, std::uint32_t x_slabs,
    const std::int64_t *xs, const std::uint64_t *vertical_counts,
    const std::uint64_t *vertical_offsets,
    DirectedSegmentI64 *vertical, std::uint32_t *status)
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
               static_cast<std::uint32_t>(kBoundaryEmitMismatch));
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
    thrust::device_vector<DirectedSegmentI64> *segments,
    cudaStream_t stream)
{
  const std::uint64_t segment_count = segments->size();
  if (!segment_count) return 0;
  auto policy = thrust::cuda::par.on(stream);
  thrust::sort(policy, segments->begin(), segments->end(), SegmentLess{});
  const auto line_keys = thrust::make_transform_iterator(
      segments->begin(), SegmentLineFromSegment{});
  const auto highs =
      thrust::make_transform_iterator(segments->begin(), SegmentHigh{});
  thrust::device_vector<std::int64_t> line_prefix_high(segment_count);
  thrust::inclusive_scan_by_key(
      policy, line_keys, line_keys + segment_count, highs,
      line_prefix_high.begin(), SegmentLineEqual{},
      thrust::maximum<std::int64_t>{});

  thrust::device_vector<std::uint32_t> group_marks(segment_count);
  thrust::device_vector<std::uint32_t> group_ids(segment_count);
  mark_segment_groups_kernel<<<
      launch_blocks(segment_count), kThreads, 0, stream>>>(
      thrust::raw_pointer_cast(segments->data()), segment_count,
      thrust::raw_pointer_cast(line_prefix_high.data()),
      thrust::raw_pointer_cast(group_marks.data()));
  cuda_require(cudaGetLastError(), "mark morphology segment groups");
  thrust::inclusive_scan(
      policy, group_marks.begin(), group_marks.end(),
      group_ids.begin());

  std::uint32_t canonical_count = 0;
  cuda_require(
      cudaMemcpyAsync(
          &canonical_count,
          thrust::raw_pointer_cast(group_ids.data()) + segment_count - 1,
          sizeof(canonical_count), cudaMemcpyDeviceToHost, stream),
      "morphology canonical group count D2H");
  cuda_require(
      cudaStreamSynchronize(stream),
      "morphology canonical group count synchronize");
  if (!canonical_count || canonical_count > segment_count) {
    throw std::runtime_error(
        "morphology canonical segment count invariant");
  }
  thrust::device_vector<std::uint32_t> output_group_ids(canonical_count);
  thrust::device_vector<DirectedSegmentI64> canonical(canonical_count);
  const auto canonical_end = thrust::reduce_by_key(
      policy, group_ids.begin(), group_ids.end(), segments->begin(),
      output_group_ids.begin(), canonical.begin(),
      thrust::equal_to<std::uint32_t>{}, SegmentMerge{});
  if (static_cast<std::uint64_t>(
          canonical_end.second - canonical.begin()) != canonical_count) {
    throw std::runtime_error(
        "morphology canonical segment count mismatch");
  }
  segments->swap(canonical);
  return canonical_count;
}

enum MorphStatus : std::uint32_t
{
  kMorphTooManyActiveSlabs = 1u << 0,
  kMorphCoordinateOverflow = 1u << 1,
  kMorphPerSlabCountOverflow = 1u << 2,
  kMorphWorkCapacity = 1u << 3,
  kMorphEmitMismatch = 1u << 4,
  kMorphOutputInvariant = 1u << 5,
};

enum SourceStatus : std::uint32_t
{
  kSourceXOrder = 1u << 0,
  kSourceSlabRange = 1u << 1,
  kSourceIntervalInvariant = 1u << 2,
};

using MorphLimits = morph::Limits;
using DeviceBandView = morph::DeviceStripView;

__global__ void validate_source_slab_ranges_kernel(
    DeviceBandView source, std::uint32_t *status)
{
  for (std::uint64_t slab =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       slab < source.x_slabs;
       slab += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (source.xs[slab] >= source.xs[slab + 1]) {
      atomicOr(status, static_cast<std::uint32_t>(kSourceXOrder));
    }
    const std::uint64_t offset = source.slab_offsets[slab];
    const std::uint64_t count = source.slab_counts[slab];
    if (offset > source.interval_count ||
        count > source.interval_count - offset) {
      atomicOr(status, static_cast<std::uint32_t>(kSourceSlabRange));
      continue;
    }
    if (!slab) {
      if (offset != 0) {
        atomicOr(status, static_cast<std::uint32_t>(kSourceSlabRange));
      }
    } else {
      const std::uint64_t previous_offset =
          source.slab_offsets[slab - 1];
      const std::uint64_t previous_count =
          source.slab_counts[slab - 1];
      if (previous_offset > source.interval_count ||
          previous_count >
              source.interval_count - previous_offset ||
          offset != previous_offset + previous_count) {
        atomicOr(status, static_cast<std::uint32_t>(kSourceSlabRange));
      }
    }
    if (slab + 1 == source.x_slabs &&
        offset + count != source.interval_count) {
      atomicOr(status, static_cast<std::uint32_t>(kSourceSlabRange));
    }
  }
}

__global__ void validate_source_intervals_kernel(
    DeviceBandView source, std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < source.interval_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const StripInterval interval = source.intervals[index];
    if (interval.slab >= source.x_slabs ||
        interval.reserved != 0 || interval.bottom >= interval.top) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(kSourceIntervalInvariant));
      continue;
    }
    const std::uint64_t offset =
        source.slab_offsets[interval.slab];
    const std::uint64_t count =
        source.slab_counts[interval.slab];
    if (index < offset || index >= offset + count) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(kSourceIntervalInvariant));
      continue;
    }
    if (index > offset) {
      const StripInterval previous = source.intervals[index - 1];
      if (previous.slab != interval.slab ||
          previous.top >= interval.bottom) {
        atomicOr(
            status,
            static_cast<std::uint32_t>(kSourceIntervalInvariant));
      }
    }
  }
}

void validate_device_source(const DeviceBandView &source,
                            cudaStream_t stream)
{
  auto policy = thrust::cuda::par.on(stream);
  thrust::device_vector<std::uint32_t> status(1);
  thrust::fill(policy, status.begin(), status.end(), 0);
  validate_source_slab_ranges_kernel<<<
      launch_blocks(source.x_slabs), kThreads, 0, stream>>>(
      source, thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(), "validate resident source slab ranges");
  std::uint32_t host_status = copy_device_status(status, stream);
  if (host_status) {
    throw std::runtime_error(
        "resident source slab/x invariant");
  }

  thrust::fill(policy, status.begin(), status.end(), 0);
  validate_source_intervals_kernel<<<
      launch_blocks(source.interval_count), kThreads, 0, stream>>>(
      source, thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(), "validate resident source intervals");
  host_status = copy_device_status(status, stream);
  if (host_status) {
    throw std::runtime_error(
        "resident source interval invariant");
  }
}

struct DeviceBandSet
{
  thrust::device_vector<std::int64_t> xs;
  thrust::device_vector<StripInterval> intervals;
  thrust::device_vector<std::uint64_t> slab_offsets;
  thrust::device_vector<std::uint32_t> slab_counts;

  std::uint32_t x_slabs() const
  {
    return xs.empty() ? 0u : static_cast<std::uint32_t>(xs.size() - 1);
  }

  DeviceBandView view() const
  {
    return {
        xs.empty() ? nullptr : thrust::raw_pointer_cast(xs.data()),
        x_slabs(),
        intervals.empty()
            ? nullptr
            : thrust::raw_pointer_cast(intervals.data()),
        intervals.size(),
        slab_offsets.empty()
            ? nullptr
            : thrust::raw_pointer_cast(slab_offsets.data()),
        slab_counts.empty()
            ? nullptr
            : thrust::raw_pointer_cast(slab_counts.data())};
  }
};

using MorphMetrics = morph::PassMetrics;

struct OutsideCarrier
{
  std::int64_t low;
  std::int64_t high;

  MU_HD bool operator()(std::int64_t value) const
  {
    return value < low || value > high;
  }
};

__global__ void shifted_x_endpoints_kernel(
    const std::int64_t *xs, std::uint64_t endpoint_count,
    std::int64_t radius, std::int64_t *shifted)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < endpoint_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    shifted[2 * index] = xs[index] - radius;
    shifted[2 * index + 1] = xs[index] + radius;
  }
}

__device__ std::uint32_t first_intersecting_slab(
    const std::int64_t *xs, std::uint32_t x_slabs,
    std::int64_t lower2)
{
  std::uint32_t low = 0;
  std::uint32_t high = x_slabs;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2;
    if (2 * xs[middle + 1] > lower2) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return low;
}

__device__ std::uint32_t first_nonintersecting_slab(
    const std::int64_t *xs, std::uint32_t x_slabs,
    std::int64_t upper2)
{
  std::uint32_t low = 0;
  std::uint32_t high = x_slabs;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2;
    if (2 * xs[middle] >= upper2) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return low;
}

__device__ bool transformed_interval(
    const StripInterval &interval, std::int64_t radius, bool erosion,
    std::int64_t *lo, std::int64_t *hi, std::uint32_t *status)
{
  if (erosion) {
    if (interval.bottom >
            std::numeric_limits<std::int64_t>::max() - radius ||
        interval.top <
            std::numeric_limits<std::int64_t>::min() + radius) {
      atomicOr(status,
               static_cast<std::uint32_t>(kMorphCoordinateOverflow));
      return false;
    }
    *lo = interval.bottom + radius;
    *hi = interval.top - radius;
  } else {
    if (interval.bottom <
            std::numeric_limits<std::int64_t>::min() + radius ||
        interval.top >
            std::numeric_limits<std::int64_t>::max() - radius) {
      atomicOr(status,
               static_cast<std::uint32_t>(kMorphCoordinateOverflow));
      return false;
    }
    *lo = interval.bottom - radius;
    *hi = interval.top + radius;
  }
  return *lo < *hi;
}

__device__ bool current_transformed_interval(
    const DeviceBandView &source, std::uint32_t source_slab,
    std::uint32_t *cursor, std::int64_t radius, bool erosion,
    std::int64_t *lo, std::int64_t *hi, std::uint32_t *status)
{
  const std::uint32_t count = source.slab_counts[source_slab];
  const std::uint64_t offset = source.slab_offsets[source_slab];
  while (*cursor < count) {
    if (transformed_interval(
            source.intervals[offset + *cursor], radius, erosion,
            lo, hi, status)) {
      return true;
    }
    ++*cursor;
  }
  return false;
}

__device__ void publish_morph_interval(
    std::int64_t lo, std::int64_t hi, std::uint32_t output_slab,
    StripInterval *output, std::uint64_t *count)
{
  if (output) {
    output[*count] = {lo, hi, output_slab, 0};
  }
  ++*count;
}

__device__ std::uint64_t visit_dilated_intervals(
    const DeviceBandView &source, std::uint32_t first_source,
    std::uint32_t last_source, std::int64_t radius,
    std::uint32_t output_slab, StripInterval *output,
    std::uint32_t *status)
{
  std::uint32_t cursors[kMorphMaxActiveSlabs];
  const std::uint32_t active = last_source - first_source;
  for (std::uint32_t index = 0; index < active; ++index) {
    cursors[index] = 0;
  }

  bool have_accumulator = false;
  std::int64_t accumulator_lo = 0;
  std::int64_t accumulator_hi = 0;
  std::uint64_t output_count = 0;

  while (true) {
    std::uint32_t selected = active;
    std::int64_t selected_lo = 0;
    std::int64_t selected_hi = 0;
    for (std::uint32_t index = 0; index < active; ++index) {
      std::int64_t lo = 0;
      std::int64_t hi = 0;
      if (!current_transformed_interval(
              source, first_source + index, &cursors[index], radius,
              false, &lo, &hi, status)) {
        continue;
      }
      if (selected == active || lo < selected_lo ||
          (lo == selected_lo && hi < selected_hi)) {
        selected = index;
        selected_lo = lo;
        selected_hi = hi;
      }
    }
    if (selected == active) break;
    ++cursors[selected];

    if (!have_accumulator) {
      accumulator_lo = selected_lo;
      accumulator_hi = selected_hi;
      have_accumulator = true;
    } else if (selected_lo <= accumulator_hi) {
      accumulator_hi = max(accumulator_hi, selected_hi);
    } else {
      publish_morph_interval(
          accumulator_lo, accumulator_hi, output_slab, output,
          &output_count);
      accumulator_lo = selected_lo;
      accumulator_hi = selected_hi;
    }
  }

  if (have_accumulator) {
    publish_morph_interval(
        accumulator_lo, accumulator_hi, output_slab, output,
        &output_count);
  }
  return output_count;
}

__device__ std::uint64_t visit_eroded_intervals(
    const DeviceBandView &source, std::uint32_t first_source,
    std::uint32_t last_source, std::int64_t radius,
    std::uint32_t output_slab, StripInterval *output,
    std::uint32_t *status)
{
  std::uint32_t cursors[kMorphMaxActiveSlabs];
  const std::uint32_t active = last_source - first_source;
  for (std::uint32_t index = 0; index < active; ++index) {
    cursors[index] = 0;
  }

  bool have_accumulator = false;
  std::int64_t accumulator_lo = 0;
  std::int64_t accumulator_hi = 0;
  std::uint64_t output_count = 0;

  while (true) {
    std::int64_t intersection_lo =
        std::numeric_limits<std::int64_t>::min();
    std::int64_t intersection_hi =
        std::numeric_limits<std::int64_t>::max();
    std::int64_t current_hi[kMorphMaxActiveSlabs];
    for (std::uint32_t index = 0; index < active; ++index) {
      std::int64_t lo = 0;
      std::int64_t hi = 0;
      if (!current_transformed_interval(
              source, first_source + index, &cursors[index], radius,
              true, &lo, &hi, status)) {
        if (have_accumulator) {
          publish_morph_interval(
              accumulator_lo, accumulator_hi, output_slab, output,
              &output_count);
        }
        return output_count;
      }
      intersection_lo = max(intersection_lo, lo);
      intersection_hi = min(intersection_hi, hi);
      current_hi[index] = hi;
    }

    if (intersection_lo < intersection_hi) {
      if (!have_accumulator) {
        accumulator_lo = intersection_lo;
        accumulator_hi = intersection_hi;
        have_accumulator = true;
      } else if (intersection_lo <= accumulator_hi) {
        accumulator_hi = max(accumulator_hi, intersection_hi);
      } else {
        publish_morph_interval(
            accumulator_lo, accumulator_hi, output_slab, output,
            &output_count);
        accumulator_lo = intersection_lo;
        accumulator_hi = intersection_hi;
      }
    }

    for (std::uint32_t index = 0; index < active; ++index) {
      if (current_hi[index] == intersection_hi) {
        ++cursors[index];
      }
    }
  }
}

__device__ std::uint64_t visit_morph_intervals(
    const DeviceBandView &source, const std::int64_t *output_xs,
    std::uint32_t output_slab, std::int64_t radius, bool erosion,
    StripInterval *output, std::uint32_t *status,
    std::uint32_t max_active_slabs,
    std::uint64_t max_source_visits_per_band,
    unsigned long long *source_visits,
    std::uint32_t *observed_max_active)
{
  const std::int64_t midpoint2 =
      output_xs[output_slab] + output_xs[output_slab + 1];
  const std::int64_t lower2 = midpoint2 - 2 * radius;
  const std::int64_t upper2 = midpoint2 + 2 * radius;
  const std::uint32_t first_source = first_intersecting_slab(
      source.xs, source.x_slabs, lower2);
  const std::uint32_t last_source = first_nonintersecting_slab(
      source.xs, source.x_slabs, upper2);
  if (last_source < first_source) {
    atomicOr(status,
             static_cast<std::uint32_t>(kMorphOutputInvariant));
    return 0;
  }
  const std::uint32_t active = last_source - first_source;
  if (observed_max_active) {
    atomicMax(observed_max_active, active);
  }
  if (active > max_active_slabs ||
      active > kMorphMaxActiveSlabs) {
    atomicOr(status,
             static_cast<std::uint32_t>(kMorphTooManyActiveSlabs));
    return 0;
  }

  std::uint64_t visits = 0;
  for (std::uint32_t slab = first_source; slab < last_source; ++slab) {
    visits += source.slab_counts[slab];
  }
  if (source_visits) {
    atomicAdd(source_visits, static_cast<unsigned long long>(visits));
  }
  if (visits > max_source_visits_per_band) {
    atomicOr(status, static_cast<std::uint32_t>(kMorphWorkCapacity));
    return 0;
  }
  if (!active) return 0;

  return erosion
             ? visit_eroded_intervals(
                   source, first_source, last_source, radius,
                   output_slab, output, status)
             : visit_dilated_intervals(
                   source, first_source, last_source, radius,
                   output_slab, output, status);
}

__global__ void count_morph_intervals_kernel(
    DeviceBandView source, const std::int64_t *output_xs,
    std::uint32_t output_slabs, std::int64_t radius, bool erosion,
    std::uint32_t max_active_slabs,
    std::uint64_t max_source_visits_per_band,
    std::uint32_t *output_counts, std::uint32_t *status,
    unsigned long long *source_visits,
    std::uint32_t *observed_max_active)
{
  for (std::uint64_t output_slab =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       output_slab < output_slabs;
       output_slab +=
           static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t count = visit_morph_intervals(
        source, output_xs, static_cast<std::uint32_t>(output_slab),
        radius, erosion, nullptr, status, max_active_slabs,
        max_source_visits_per_band, source_visits,
        observed_max_active);
    if (count > std::numeric_limits<std::uint32_t>::max()) {
      atomicOr(status,
               static_cast<std::uint32_t>(
                   kMorphPerSlabCountOverflow));
      output_counts[output_slab] = 0;
    } else {
      output_counts[output_slab] =
          static_cast<std::uint32_t>(count);
    }
  }
}

__global__ void emit_morph_intervals_kernel(
    DeviceBandView source, const std::int64_t *output_xs,
    std::uint32_t output_slabs, std::int64_t radius, bool erosion,
    std::uint32_t max_active_slabs,
    std::uint64_t max_source_visits_per_band,
    const std::uint64_t *output_offsets,
    const std::uint32_t *output_counts, StripInterval *output,
    std::uint32_t *status)
{
  for (std::uint64_t output_slab =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       output_slab < output_slabs;
       output_slab +=
           static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t count = visit_morph_intervals(
        source, output_xs, static_cast<std::uint32_t>(output_slab),
        radius, erosion, output + output_offsets[output_slab], status,
        max_active_slabs, max_source_visits_per_band, nullptr, nullptr);
    if (count != output_counts[output_slab]) {
      atomicOr(status,
               static_cast<std::uint32_t>(kMorphEmitMismatch));
    }
  }
}

__global__ void validate_morph_intervals_kernel(
    const StripInterval *intervals, std::uint64_t interval_count,
    const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, std::uint32_t x_slabs,
    std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < interval_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const StripInterval interval = intervals[index];
    if (interval.slab >= x_slabs ||
        interval.bottom >= interval.top ||
        index < slab_offsets[interval.slab] ||
        index >= slab_offsets[interval.slab] +
                     slab_counts[interval.slab]) {
      atomicOr(status,
               static_cast<std::uint32_t>(kMorphOutputInvariant));
      continue;
    }
    if (index > slab_offsets[interval.slab]) {
      const StripInterval previous = intervals[index - 1];
      if (previous.slab != interval.slab ||
          previous.top >= interval.bottom) {
        atomicOr(status,
                 static_cast<std::uint32_t>(
                     kMorphOutputInvariant));
      }
    }
  }
}

void sample_morph_memory(MorphMetrics *metrics)
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_require(cudaMemGetInfo(&free_bytes, &total_bytes),
               "morph cudaMemGetInfo");
  if (!metrics->device_total_bytes) {
    metrics->device_total_bytes = total_bytes;
    metrics->device_free_begin_bytes = free_bytes;
    metrics->device_free_low_bytes = free_bytes;
  } else {
    metrics->device_free_low_bytes =
        std::min<std::uint64_t>(
            metrics->device_free_low_bytes, free_bytes);
  }
}

std::string morph_status_message(std::uint32_t status)
{
  std::ostringstream stream;
  if (status & kMorphTooManyActiveSlabs)
    stream << "active-slab capacity;";
  if (status & kMorphCoordinateOverflow)
    stream << "coordinate overflow;";
  if (status & kMorphPerSlabCountOverflow)
    stream << "per-slab count overflow;";
  if (status & kMorphWorkCapacity) stream << "work capacity;";
  if (status & kMorphEmitMismatch) stream << "emit mismatch;";
  if (status & kMorphOutputInvariant)
    stream << "output invariant;";
  return stream.str();
}

DeviceBandSet morph_bands(
    const DeviceBandView &source, std::int64_t radius, bool erosion,
    const MorphLimits &limits, MorphMetrics *metrics,
    cudaStream_t stream, bool count_only = false)
{
  const auto begin = Clock::now();
  if (!metrics) {
    throw std::runtime_error("null morphology metrics");
  }
  if (radius < 0) {
    throw std::runtime_error("negative morphology radius");
  }
  if (!source.x_slabs || !source.xs || !source.slab_offsets ||
      !source.slab_counts) {
    throw std::runtime_error("invalid empty source band view");
  }
  if (source.x_slabs >
      std::numeric_limits<std::uint32_t>::max() / 2 - 1) {
    throw std::runtime_error("shifted endpoint capacity");
  }

  std::int64_t source_low = 0;
  std::int64_t source_high = 0;
  cuda_require(
      cudaMemcpyAsync(
          &source_low, source.xs, sizeof(source_low),
          cudaMemcpyDeviceToHost, stream),
      "source lower x D2H");
  cuda_require(
      cudaMemcpyAsync(
          &source_high, source.xs + source.x_slabs,
          sizeof(source_high), cudaMemcpyDeviceToHost, stream),
      "source upper x D2H");
  cuda_require(
      cudaStreamSynchronize(stream), "source x bounds synchronize");
  const __int128 wide_low = source_low;
  const __int128 wide_high = source_high;
  const __int128 wide_radius = radius;
  const __int128 wide_min =
      std::numeric_limits<std::int64_t>::min();
  const __int128 wide_max =
      std::numeric_limits<std::int64_t>::max();
  if (source_low >
          std::numeric_limits<std::int64_t>::max() - radius ||
      source_low <
          std::numeric_limits<std::int64_t>::min() + radius ||
      source_high >
          std::numeric_limits<std::int64_t>::max() - radius ||
      source_high <
          std::numeric_limits<std::int64_t>::min() + radius ||
      2 * wide_low - 4 * wide_radius < wide_min ||
      2 * wide_high + 4 * wide_radius > wide_max) {
    throw std::runtime_error("x coordinate overflow");
  }

  DeviceBandSet result;
  sample_morph_memory(metrics);
  const std::uint64_t source_endpoint_count =
      static_cast<std::uint64_t>(source.x_slabs) + 1;
  result.xs.resize(2 * source_endpoint_count);
  shifted_x_endpoints_kernel<<<
      launch_blocks(source_endpoint_count), kThreads, 0, stream>>>(
      source.xs, source_endpoint_count, radius,
      thrust::raw_pointer_cast(result.xs.data()));
  cuda_require(cudaGetLastError(), "shift morphology x endpoints");
  auto policy = thrust::cuda::par.on(stream);
  thrust::sort(policy, result.xs.begin(), result.xs.end());
  result.xs.erase(
      thrust::unique(
          policy, result.xs.begin(), result.xs.end()),
      result.xs.end());

  if (erosion) {
    if (source_low > source_high - radius ||
        source_low + radius >= source_high - radius) {
      result.xs.clear();
      metrics->elapsed_ms = elapsed_ms(begin, Clock::now());
      return result;
    }
    const std::int64_t carrier_low = source_low + radius;
    const std::int64_t carrier_high = source_high - radius;
    result.xs.erase(
        thrust::remove_if(
            policy, result.xs.begin(), result.xs.end(),
            OutsideCarrier{carrier_low, carrier_high}),
        result.xs.end());
  }
  if (result.xs.size() < 2) {
    result.xs.clear();
    metrics->elapsed_ms = elapsed_ms(begin, Clock::now());
    return result;
  }

  const std::uint64_t output_slabs_u64 = result.xs.size() - 1;
  if (output_slabs_u64 > limits.max_output_slabs ||
      output_slabs_u64 >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("output slab capacity");
  }
  const std::uint32_t output_slabs =
      static_cast<std::uint32_t>(output_slabs_u64);
  result.slab_counts.resize(output_slabs);
  result.slab_offsets.resize(output_slabs);
  thrust::device_vector<std::uint32_t> status(1);
  thrust::device_vector<unsigned long long> source_visits(1);
  thrust::device_vector<std::uint32_t> observed_max_active(1);
  thrust::fill(
      policy, result.slab_counts.begin(), result.slab_counts.end(), 0);
  thrust::fill(policy, status.begin(), status.end(), 0);
  thrust::fill(
      policy, source_visits.begin(), source_visits.end(),
      static_cast<unsigned long long>(0));
  thrust::fill(
      policy, observed_max_active.begin(),
      observed_max_active.end(), 0);
  sample_morph_memory(metrics);

  count_morph_intervals_kernel<<<
      launch_blocks(output_slabs), kThreads, 0, stream>>>(
      source, thrust::raw_pointer_cast(result.xs.data()),
      output_slabs, radius, erosion, limits.max_active_slabs,
      limits.max_source_visits_per_band,
      thrust::raw_pointer_cast(result.slab_counts.data()),
      thrust::raw_pointer_cast(status.data()),
      thrust::raw_pointer_cast(source_visits.data()),
      thrust::raw_pointer_cast(observed_max_active.data()));
  cuda_require(cudaGetLastError(), "count morphology intervals");
  thrust::exclusive_scan(
      policy, result.slab_counts.begin(),
      result.slab_counts.end(), result.slab_offsets.begin(),
      std::uint64_t{0});

  std::uint64_t final_offset = 0;
  std::uint32_t final_count = 0;
  cuda_require(
      cudaMemcpyAsync(
          &final_offset,
          thrust::raw_pointer_cast(result.slab_offsets.data()) +
              output_slabs - 1,
          sizeof(final_offset), cudaMemcpyDeviceToHost, stream),
      "last morphology offset D2H");
  cuda_require(
      cudaMemcpyAsync(
          &final_count,
          thrust::raw_pointer_cast(result.slab_counts.data()) +
              output_slabs - 1,
          sizeof(final_count), cudaMemcpyDeviceToHost, stream),
      "last morphology count D2H");
  cuda_require(
      cudaStreamSynchronize(stream),
      "morphology count census synchronize");
  if (final_offset >
      std::numeric_limits<std::uint64_t>::max() - final_count) {
    throw std::runtime_error("output interval count overflow");
  }
  const std::uint64_t output_intervals = final_offset + final_count;

  std::uint32_t host_status = copy_device_status(status, stream);
  cuda_require(
      cudaMemcpyAsync(
          &metrics->source_visits,
          thrust::raw_pointer_cast(source_visits.data()),
          sizeof(metrics->source_visits), cudaMemcpyDeviceToHost,
          stream),
      "morph source visits D2H");
  cuda_require(
      cudaMemcpyAsync(
          &metrics->max_active_slabs,
          thrust::raw_pointer_cast(observed_max_active.data()),
          sizeof(metrics->max_active_slabs), cudaMemcpyDeviceToHost,
          stream),
      "morph max active D2H");
  cuda_require(
      cudaStreamSynchronize(stream), "morph metrics synchronize");
  if (metrics->source_visits > limits.max_total_source_visits) {
    host_status |= kMorphWorkCapacity;
  }
  if (host_status) {
    std::ostringstream stream;
    stream << "morph count failed: " << morph_status_message(host_status)
           << " observed_max_active=" << metrics->max_active_slabs
           << " source_visits=" << metrics->source_visits;
    throw std::runtime_error(stream.str());
  }
  if (output_intervals > limits.max_output_intervals) {
    throw std::runtime_error("output interval capacity");
  }
  metrics->output_intervals = output_intervals;
  if (count_only) {
    cuda_require(
        cudaStreamSynchronize(stream),
        "count-only morphology synchronize");
    sample_morph_memory(metrics);
    metrics->elapsed_ms = elapsed_ms(begin, Clock::now());
    return result;
  }

  result.intervals.resize(output_intervals);
  sample_morph_memory(metrics);
  if (output_intervals) {
    emit_morph_intervals_kernel<<<
        launch_blocks(output_slabs), kThreads, 0, stream>>>(
        source, thrust::raw_pointer_cast(result.xs.data()),
        output_slabs, radius, erosion, limits.max_active_slabs,
        limits.max_source_visits_per_band,
        thrust::raw_pointer_cast(result.slab_offsets.data()),
        thrust::raw_pointer_cast(result.slab_counts.data()),
        thrust::raw_pointer_cast(result.intervals.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "emit morphology intervals");
    validate_morph_intervals_kernel<<<
        launch_blocks(output_intervals), kThreads, 0, stream>>>(
        thrust::raw_pointer_cast(result.intervals.data()),
        output_intervals,
        thrust::raw_pointer_cast(result.slab_offsets.data()),
        thrust::raw_pointer_cast(result.slab_counts.data()),
        output_slabs, thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "validate morphology intervals");
  }
  cuda_require(cudaStreamSynchronize(stream), "morphology synchronize");
  host_status = copy_device_status(status, stream);
  if (host_status) {
    throw std::runtime_error(
        "morph emit failed: " + morph_status_message(host_status));
  }
  sample_morph_memory(metrics);
  metrics->elapsed_ms = elapsed_ms(begin, Clock::now());
  return result;
}

__global__ void emit_horizontal_band_boundary_kernel(
    const StripInterval *intervals, std::uint64_t interval_count,
    const std::int64_t *xs, DirectedSegmentI64 *segments)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < interval_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const StripInterval interval = intervals[index];
    const std::int64_t left = xs[interval.slab];
    const std::int64_t right = xs[interval.slab + 1];
    segments[2 * index] = {
        interval.bottom, left, right, -1, SegmentAxis::horizontal};
    segments[2 * index + 1] = {
        interval.top, left, right, 1, SegmentAxis::horizontal};
  }
}

thrust::device_vector<DirectedSegmentI64> device_boundary_from_bands(
    const DeviceBandSet &bands, std::uint64_t max_raw_segments,
    std::uint64_t max_segments, cudaStream_t stream,
    MorphMetrics *memory_metrics = nullptr)
{
  const DeviceBandView view = bands.view();
  if (!view.x_slabs || !view.interval_count) return {};
  const std::uint64_t boundary_count =
      static_cast<std::uint64_t>(view.x_slabs) + 1;
  thrust::device_vector<std::uint64_t> vertical_counts(boundary_count);
  thrust::device_vector<std::uint64_t> vertical_offsets(boundary_count);
  if (memory_metrics) sample_morph_memory(memory_metrics);
  count_vertical_xor_kernel<<<
      launch_blocks(boundary_count), kThreads, 0, stream>>>(
      view.intervals, view.slab_offsets, view.slab_counts,
      view.x_slabs,
      thrust::raw_pointer_cast(vertical_counts.data()));
  cuda_require(cudaGetLastError(), "count morphology vertical boundary");
  auto policy = thrust::cuda::par.on(stream);
  thrust::exclusive_scan(
      policy, vertical_counts.begin(), vertical_counts.end(),
      vertical_offsets.begin(), std::uint64_t{0});

  std::uint64_t final_offset = 0;
  std::uint64_t final_count = 0;
  cuda_require(
      cudaMemcpyAsync(
          &final_offset,
          thrust::raw_pointer_cast(vertical_offsets.data()) +
              boundary_count - 1,
          sizeof(final_offset), cudaMemcpyDeviceToHost, stream),
      "morph vertical offset D2H");
  cuda_require(
      cudaMemcpyAsync(
          &final_count,
          thrust::raw_pointer_cast(vertical_counts.data()) +
              boundary_count - 1,
          sizeof(final_count), cudaMemcpyDeviceToHost, stream),
      "morph vertical count D2H");
  cuda_require(
      cudaStreamSynchronize(stream),
      "morph vertical count synchronize");
  if (final_offset >
      std::numeric_limits<std::uint64_t>::max() - final_count) {
    throw std::runtime_error("morph vertical boundary overflow");
  }
  const std::uint64_t vertical_count = final_offset + final_count;
  if (view.interval_count >
      (std::numeric_limits<std::uint64_t>::max() - vertical_count) / 2) {
    throw std::runtime_error("morph boundary count overflow");
  }
  const std::uint64_t raw_count =
      2 * view.interval_count + vertical_count;
  if (raw_count > max_raw_segments) {
    throw std::runtime_error("morph raw boundary capacity");
  }

  thrust::device_vector<DirectedSegmentI64> raw(raw_count);
  if (memory_metrics) sample_morph_memory(memory_metrics);
  if (view.interval_count) {
    emit_horizontal_band_boundary_kernel<<<
        launch_blocks(view.interval_count), kThreads, 0, stream>>>(
        view.intervals, view.interval_count, view.xs,
        thrust::raw_pointer_cast(raw.data()));
    cuda_require(cudaGetLastError(), "emit morph horizontal boundary");
  }
  thrust::device_vector<std::uint32_t> status(1);
  thrust::fill(policy, status.begin(), status.end(), 0);
  if (vertical_count) {
    emit_vertical_xor_kernel<<<
        launch_blocks(boundary_count), kThreads, 0, stream>>>(
        view.intervals, view.slab_offsets, view.slab_counts,
        view.x_slabs, view.xs,
        thrust::raw_pointer_cast(vertical_counts.data()),
        thrust::raw_pointer_cast(vertical_offsets.data()),
        thrust::raw_pointer_cast(raw.data()) + 2 * view.interval_count,
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "emit morph vertical boundary");
  }
  cuda_require(
      cudaStreamSynchronize(stream),
      "morph boundary emit synchronize");
  if (copy_device_status(status, stream)) {
    throw std::runtime_error("morph vertical boundary invariant");
  }
  canonicalize_device_segments(&raw, stream);
  if (raw.size() > max_segments) {
    throw std::runtime_error("morph canonical boundary capacity");
  }
  cuda_require(
      cudaStreamSynchronize(stream),
      "morph boundary canonicalize synchronize");
  if (memory_metrics) sample_morph_memory(memory_metrics);
  return raw;
}

std::vector<DirectedSegmentI64> boundary_from_bands(
    const DeviceBandSet &bands, std::uint64_t max_raw_segments,
    std::uint64_t max_segments, cudaStream_t stream)
{
  thrust::device_vector<DirectedSegmentI64> raw =
      device_boundary_from_bands(
          bands, max_raw_segments, max_segments, stream);
  std::vector<DirectedSegmentI64> host(raw.size());
  if (!host.empty()) {
    cuda_require(
        cudaMemcpyAsync(
            host.data(), thrust::raw_pointer_cast(raw.data()),
            host.size() * sizeof(DirectedSegmentI64),
            cudaMemcpyDeviceToHost, stream),
        "morph boundary D2H");
    cuda_require(
        cudaStreamSynchronize(stream),
        "morph boundary D2H synchronize");
  }
  return host;
}

// This bounds the segment count, not the pair count: 4096 segments imply at
// most 8,386,560 unordered pairs in this deliberately bounded certificate.
constexpr std::uint64_t kF90LongSpacePairwiseSegmentCountCap =
    UINT64_C(4096);

struct LongSegment
{
  std::uint64_t minimum;

  MU_HD bool operator()(const DirectedSegmentI64 &segment) const
  {
    return static_cast<std::uint64_t>(segment.hi) -
               static_cast<std::uint64_t>(segment.lo) >=
           minimum;
  }
};

using LongSpaceCertificate = morph::LongSpaceCertificate;

m1ws::DirectedEdge directed_edge_from_boundary(
    const DirectedSegmentI64 &segment)
{
  if (segment.lo >= segment.hi ||
      (segment.side != -1 && segment.side != 1)) {
    throw std::runtime_error(
        "invalid canonical segment in F90 long-edge certificate");
  }
  if (segment.axis == SegmentAxis::horizontal) {
    return segment.side > 0
               ? m1ws::DirectedEdge{
                     segment.lo, segment.fixed, segment.hi, segment.fixed}
               : m1ws::DirectedEdge{
                     segment.hi, segment.fixed, segment.lo, segment.fixed};
  }
  if (segment.axis == SegmentAxis::vertical) {
    return segment.side < 0
               ? m1ws::DirectedEdge{
                     segment.fixed, segment.lo, segment.fixed, segment.hi}
               : m1ws::DirectedEdge{
                     segment.fixed, segment.hi, segment.fixed, segment.lo};
  }
  throw std::runtime_error(
      "invalid canonical axis in F90 long-edge certificate");
}

LongSpaceCertificate certify_f90_long_edge_space_impl(
    const std::vector<DirectedSegmentI64> &segments,
    std::uint64_t max_segments)
{
  if (segments.size() > max_segments) {
    throw std::runtime_error(
        "F90 long-edge pairwise certificate segment-count capacity");
  }
  const std::uint64_t segment_count =
      static_cast<std::uint64_t>(segments.size());
  if (segment_count > 1 &&
      segment_count >
          std::numeric_limits<std::uint64_t>::max() /
              (segment_count - 1)) {
    throw std::runtime_error(
        "F90 long-edge pairwise certificate pair-count overflow");
  }
  const std::uint64_t pair_count =
      segment_count > 1 ? segment_count * (segment_count - 1) / 2 : 0;

  LongSpaceCertificate result;
  result.pairs_checked = pair_count;
  std::uint64_t observed_pairs = 0;
  for (std::size_t first = 0; first < segments.size(); ++first) {
    const m1ws::DirectedEdge first_edge =
        directed_edge_from_boundary(segments[first]);
    for (std::size_t second = first + 1; second < segments.size();
         ++second) {
      const m1ws::CandidatePair candidate = {
          first_edge, directed_edge_from_boundary(segments[second]),
          0, 0, m1ws::Rule::kSpace};
      const m1ws::Verdict verdict = m1ws::classify_pair_bounded(
          candidate, m1ws::kM2F90LongSpaceCoordinateDistance);
      if (verdict == m1ws::Verdict::kViolation) {
        ++result.violations;
      } else if (verdict == m1ws::Verdict::kUncertain) {
        ++result.uncertain;
      }
      ++observed_pairs;
    }
  }
  if (observed_pairs != result.pairs_checked) {
    throw std::runtime_error(
        "F90 long-edge pair census mismatch");
  }
  return result;
}

struct ProductionMorphContext
{
  bool qualify_boundary = false;
  bool invoked = false;
  MorphLimits limits;
  MorphMetrics erode89;
  MorphMetrics dilate90;
  MorphMetrics boundary_memory;
  MorphMetrics erode269_count;
  std::uint64_t gt90_boundary_segments = 0;
  std::uint64_t gt90_long_segments = 0;
  std::uint64_t gt90_space_pairs_checked = 0;
  std::uint64_t gt90_space_violations = 0;
  std::uint64_t gt90_space_uncertain = 0;
  std::uint64_t gt270_eroded_intervals = 0;
  std::uint64_t callback_device_total_bytes = 0;
  std::uint64_t callback_device_free_begin_bytes = 0;
  std::uint64_t callback_device_free_low_bytes = 0;
  double boundary_ms = 0.0;
  double callback_ms = 0.0;
  std::vector<DirectedSegmentI64> qualified_boundary;
};

void include_memory_sample(
    const MorphMetrics &sample, ProductionMorphContext *context)
{
  if (!sample.device_total_bytes) return;
  context->callback_device_total_bytes = sample.device_total_bytes;
  context->callback_device_free_low_bytes =
      std::min(
          context->callback_device_free_low_bytes,
          sample.device_free_low_bytes);
}

void production_consume_strips(
    cudaStream_t stream, const std::int64_t *xs, std::uint32_t x_slabs,
    const StripInterval *intervals, std::uint64_t interval_count,
    const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque)
{
  auto *context = static_cast<ProductionMorphContext *>(opaque);
  if (!context || context->invoked) {
    throw std::runtime_error("production resident callback state");
  }
  context->invoked = true;
  const auto callback_begin = Clock::now();
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_require(
      cudaMemGetInfo(&free_bytes, &total_bytes),
      "production callback cudaMemGetInfo");
  context->callback_device_total_bytes = total_bytes;
  context->callback_device_free_begin_bytes = free_bytes;
  context->callback_device_free_low_bytes = free_bytes;

  if (!x_slabs || !xs || !intervals || !slab_offsets ||
      !slab_counts) {
    throw std::runtime_error("production source has no x slabs");
  }
  if (x_slabs > context->limits.max_input_x_slabs ||
      interval_count > context->limits.max_input_intervals) {
    throw std::runtime_error(
        "production resident source capacity");
  }
  std::uint64_t final_offset = 0;
  std::uint32_t final_count = 0;
  cuda_require(
      cudaMemcpyAsync(
          &final_offset, slab_offsets + x_slabs - 1,
          sizeof(final_offset), cudaMemcpyDeviceToHost, stream),
      "production source final offset D2H");
  cuda_require(
      cudaMemcpyAsync(
          &final_count, slab_counts + x_slabs - 1,
          sizeof(final_count), cudaMemcpyDeviceToHost, stream),
      "production source final count D2H");
  cuda_require(
      cudaStreamSynchronize(stream),
      "production source census synchronize");
  if (final_offset >
          std::numeric_limits<std::uint64_t>::max() - final_count ||
      final_offset + final_count != interval_count) {
    throw std::runtime_error(
        "production resident source interval census mismatch");
  }

  const DeviceBandView source = {
      xs, x_slabs, intervals, interval_count, slab_offsets, slab_counts};
  const MorphLimits &limits = context->limits;
  DeviceBandSet eroded89;
  try {
    eroded89 = morph_bands(
        source, 89, true, limits, &context->erode89, stream);
  } catch (const std::exception &error) {
    throw std::runtime_error(
        std::string("erode89: ") + error.what());
  }
  if (!eroded89.x_slabs() || !context->erode89.output_intervals) {
    throw std::runtime_error("production F90 erosion unexpectedly empty");
  }
  DeviceBandSet gt90;
  try {
    gt90 = morph_bands(
        eroded89.view(), 90, false, limits, &context->dilate90,
        stream);
  } catch (const std::exception &error) {
    throw std::runtime_error(
        std::string("dilate90: ") + error.what());
  }
  if (!gt90.x_slabs() || !context->dilate90.output_intervals) {
    throw std::runtime_error("production F90 dilation unexpectedly empty");
  }
  include_memory_sample(context->erode89, context);
  include_memory_sample(context->dilate90, context);
  release_device_vector(&eroded89.xs);
  release_device_vector(&eroded89.intervals);
  release_device_vector(&eroded89.slab_offsets);
  release_device_vector(&eroded89.slab_counts);

  const auto boundary_begin = Clock::now();
  thrust::device_vector<DirectedSegmentI64> boundary =
      device_boundary_from_bands(
          gt90, limits.max_raw_boundary_segments,
          limits.max_boundary_segments, stream,
          &context->boundary_memory);
  context->gt90_boundary_segments = boundary.size();
  auto policy = thrust::cuda::par.on(stream);
  context->gt90_long_segments = thrust::count_if(
      policy, boundary.begin(), boundary.end(),
      LongSegment{600});
  if (context->gt90_long_segments >
      limits.max_long_segments) {
    throw std::runtime_error(
        "production F90 long-edge pairwise capacity");
  }
  thrust::device_vector<DirectedSegmentI64> long_segments(
      context->gt90_long_segments);
  const auto long_end = thrust::copy_if(
      policy, boundary.begin(), boundary.end(),
      long_segments.begin(), LongSegment{600});
  if (static_cast<std::uint64_t>(
          long_end - long_segments.begin()) !=
      context->gt90_long_segments) {
    throw std::runtime_error(
        "production F90 long-edge compaction mismatch");
  }
  std::vector<DirectedSegmentI64> host_long_segments(
      context->gt90_long_segments);
  if (!host_long_segments.empty()) {
    cuda_require(
        cudaMemcpyAsync(
            host_long_segments.data(),
            thrust::raw_pointer_cast(long_segments.data()),
            host_long_segments.size() * sizeof(DirectedSegmentI64),
            cudaMemcpyDeviceToHost, stream),
        "production F90 long edges D2H");
    cuda_require(
        cudaStreamSynchronize(stream),
        "production F90 long edges synchronize");
  }
  const LongSpaceCertificate long_space =
      certify_f90_long_edge_space_impl(
          host_long_segments, limits.max_long_segments);
  context->gt90_space_pairs_checked = long_space.pairs_checked;
  context->gt90_space_violations = long_space.violations;
  context->gt90_space_uncertain = long_space.uncertain;
  if (context->gt90_space_violations ||
      context->gt90_space_uncertain) {
    throw std::runtime_error(
        "production F90 long-edge space certificate is not clean");
  }
  release_device_vector(&long_segments);
  if (context->qualify_boundary) {
    context->qualified_boundary.resize(boundary.size());
    if (!boundary.empty()) {
      cuda_require(
          cudaMemcpyAsync(
              context->qualified_boundary.data(),
              thrust::raw_pointer_cast(boundary.data()),
              boundary.size() * sizeof(DirectedSegmentI64),
              cudaMemcpyDeviceToHost, stream),
          "qualified gt90 boundary D2H");
    }
  }
  cuda_require(
      cudaStreamSynchronize(stream),
      "production gt90 boundary synchronize");
  context->boundary_ms = elapsed_ms(boundary_begin, Clock::now());
  include_memory_sample(context->boundary_memory, context);
  release_device_vector(&boundary);

  DeviceBandSet gt270_eroded;
  try {
    gt270_eroded = morph_bands(
        gt90.view(), 269, true, limits, &context->erode269_count,
        stream, true);
  } catch (const std::exception &error) {
    throw std::runtime_error(
        std::string("erode269-count: ") + error.what());
  }
  context->gt270_eroded_intervals =
      context->erode269_count.output_intervals;
  include_memory_sample(context->erode269_count, context);
  if (context->gt270_eroded_intervals != 0) {
    throw std::runtime_error(
        "production F270 count-only erosion is not empty");
  }
  context->callback_ms = elapsed_ms(callback_begin, Clock::now());
}


}  // namespace

namespace klayout_cuda {
namespace m2_resident_morphology {

namespace {

void validate_limits(const Limits &limits, bool m2_production_cap,
                     bool m1_production_cap)
{
  if (!limits.max_input_x_slabs || !limits.max_input_intervals ||
      !limits.max_output_slabs || !limits.max_output_intervals ||
      !limits.max_raw_boundary_segments ||
      !limits.max_boundary_segments ||
      !limits.max_total_source_visits ||
      !limits.max_source_visits_per_band ||
      !limits.max_active_slabs ||
      limits.max_active_slabs > kMorphMaxActiveSlabs ||
      !limits.max_long_segments ||
      limits.max_long_segments >
          kF90LongSpacePairwiseSegmentCountCap ||
      limits.max_boundary_segments >
          limits.max_raw_boundary_segments) {
    throw std::runtime_error("invalid resident morphology limits");
  }
  if (m2_production_cap && m1_production_cap) {
    throw std::runtime_error(
        "resident morphology production work-cap domains overlap");
  }
  if (m2_production_cap) {
    if (limits.max_total_source_visits !=
        kQualifiedProductionSourceVisitCap) {
      throw std::runtime_error(
          "qualified M2 production work cap must be exactly 8B");
    }
  } else if (m1_production_cap) {
    if (limits.max_total_source_visits !=
        kQualifiedM1ProductionSourceVisitCap) {
      throw std::runtime_error(
          "qualified M1 production work cap must be exactly 12B");
    }
  } else if (limits.max_total_source_visits >
             kUniversalSourceVisitCap) {
    throw std::runtime_error(
        "unqualified resident morphology work cap exceeds 2B");
  }
}

void validate_source(const DeviceStripView &source,
                     const Limits &limits)
{
  if (!source.xs || !source.x_slabs || !source.intervals ||
      !source.interval_count || !source.slab_offsets ||
      !source.slab_counts) {
    throw std::runtime_error("invalid resident strip view");
  }
  if (source.x_slabs > limits.max_input_x_slabs ||
      source.interval_count > limits.max_input_intervals) {
    throw std::runtime_error("resident strip input capacity");
  }
}

}  // namespace

std::vector<manhattan_union::DirectedSegmentI64>
qualification_boundary(
    cudaStream_t stream, const DeviceStripView &source,
    QualificationOperation operation, std::int64_t first_radius,
    std::int64_t second_radius, const Limits &limits)
{
  validate_limits(limits, false, false);
  validate_source(source, limits);
  validate_device_source(source, stream);
  if (first_radius < 0 || second_radius < 0) {
    throw std::runtime_error("negative qualification radius");
  }
  MorphMetrics first_metrics;
  const bool first_erosion =
      operation != QualificationOperation::dilate;
  DeviceBandSet first = morph_bands(
      source, first_radius, first_erosion, limits, &first_metrics,
      stream);
  if (!first.x_slabs()) return {};

  if (operation == QualificationOperation::erode_then_dilate) {
    MorphMetrics second_metrics;
    DeviceBandSet second = morph_bands(
        first.view(), second_radius, false, limits, &second_metrics,
        stream);
    return boundary_from_bands(
        second, limits.max_raw_boundary_segments,
        limits.max_boundary_segments, stream);
  }
  return boundary_from_bands(
      first, limits.max_raw_boundary_segments,
      limits.max_boundary_segments, stream);
}

LongSpaceCertificate certify_f90_long_edge_space(
    const std::vector<manhattan_union::DirectedSegmentI64> &segments,
    std::uint64_t max_segments)
{
  if (!max_segments ||
      max_segments > kF90LongSpacePairwiseSegmentCountCap) {
    throw std::runtime_error(
        "invalid F90 long-edge certificate capacity");
  }
  return certify_f90_long_edge_space_impl(segments, max_segments);
}

Result consume_f90_f270(cudaStream_t stream,
                        const DeviceStripView &source,
                        const Request &request)
{
  validate_limits(
      request.limits, request.allow_qualified_production_work_cap,
      request.allow_qualified_m1_production_work_cap);
  validate_source(source, request.limits);
  validate_device_source(source, stream);
  const auto begin = Clock::now();

  ProductionMorphContext context;
  context.qualify_boundary = request.copy_f90_boundary_to_host;
  context.limits = request.limits;
  production_consume_strips(
      stream, source.xs, source.x_slabs, source.intervals,
      source.interval_count, source.slab_offsets, source.slab_counts,
      &context);
  if (!context.invoked) {
    throw std::runtime_error(
        "resident morphology consumer was not invoked");
  }

  Result result;
  result.source_x_slabs = source.x_slabs;
  result.source_intervals = source.interval_count;
  result.erode89 = context.erode89;
  result.dilate90 = context.dilate90;
  result.boundary = context.boundary_memory;
  result.erode269_count = context.erode269_count;
  result.f90_boundary_segments = context.gt90_boundary_segments;
  result.f90_long_segments = context.gt90_long_segments;
  result.f90_space_pairs_checked =
      context.gt90_space_pairs_checked;
  result.f90_space_violations = context.gt90_space_violations;
  result.f90_space_uncertain = context.gt90_space_uncertain;
  result.f270_eroded_intervals =
      context.gt270_eroded_intervals;
  result.device_total_bytes =
      context.callback_device_total_bytes;
  result.device_free_begin_bytes =
      context.callback_device_free_begin_bytes;
  result.device_free_low_bytes =
      context.callback_device_free_low_bytes;
  result.boundary_and_long_space_ms = context.boundary_ms;
  result.total_ms = elapsed_ms(begin, Clock::now());
  result.f90_boundary = std::move(context.qualified_boundary);
  if (!result.f90_boundary.empty()) {
    if (result.f90_boundary.size() !=
        result.f90_boundary_segments) {
      throw std::runtime_error(
          "qualified F90 boundary census mismatch");
    }
    result.f90_boundary_fnv64 =
        digest_segments(result.f90_boundary);
  }
  return result;
}

void consume_f90_f270_hook(
    cudaStream_t stream, const std::int64_t *xs,
    std::uint32_t x_slabs,
    const manhattan_union::StripInterval *intervals,
    std::uint64_t interval_count,
    const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque)
{
  auto *context = static_cast<ResidentContext *>(opaque);
  if (!context || context->invoked) {
    throw std::runtime_error(
        "resident morphology hook state");
  }
  context->invoked = true;
  context->result = consume_f90_f270(
      stream,
      {xs, x_slabs, intervals, interval_count, slab_offsets,
       slab_counts},
      context->request);
}

manhattan_union::ResidentStripHook make_resident_hook(
    ResidentContext *context)
{
  if (!context || context->invoked) {
    throw std::runtime_error(
        "invalid resident morphology hook context");
  }
  manhattan_union::ResidentStripHook hook;
  hook.consume = consume_f90_f270_hook;
  hook.context = context;
  hook.stop_before_boundary = true;
  return hook;
}

}  // namespace m2_resident_morphology
}  // namespace klayout_cuda
