/*
 * Exact resident-strip square morphology.
 *
 * This qualification executable deliberately includes the Manhattan-union
 * replay implementation in the same translation unit.  The union exposes its
 * canonical x-band/y-interval representation to the callback below while all
 * arrays are still device resident.  Erosion and dilation consume that
 * representation directly; there is no contour reconstruction, component
 * labeling, or host geometry round trip between operations.
 */

#define KLAYOUT_MANHATTAN_UNION_REPLAY_NO_MAIN
#include "manhattan_union_replay.cu"
#include "m1_width_space_exact_predicate.h"

#include <set>
#include <thrust/count.h>
#include <thrust/copy.h>

namespace {

namespace m1ws = klayout_cuda::m1_width_space;

constexpr std::uint32_t kMorphMaxActiveSlabs = 128;

enum MorphStatus : std::uint32_t
{
  kMorphTooManyActiveSlabs = 1u << 0,
  kMorphCoordinateOverflow = 1u << 1,
  kMorphPerSlabCountOverflow = 1u << 2,
  kMorphWorkCapacity = 1u << 3,
  kMorphEmitMismatch = 1u << 4,
  kMorphOutputInvariant = 1u << 5,
};

struct MorphLimits
{
  std::uint64_t max_output_slabs = UINT64_C(1000000);
  std::uint64_t max_output_intervals = UINT64_C(64000000);
  std::uint64_t max_total_source_visits = UINT64_C(2000000000);
  std::uint64_t max_source_visits_per_band = UINT64_C(4000000);
  std::uint32_t max_active_slabs = kMorphMaxActiveSlabs;
};

struct DeviceBandView
{
  const std::int64_t *xs = nullptr;
  std::uint32_t x_slabs = 0;
  const StripInterval *intervals = nullptr;
  std::uint64_t interval_count = 0;
  const std::uint64_t *slab_offsets = nullptr;
  const std::uint32_t *slab_counts = nullptr;
};

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

struct MorphMetrics
{
  std::uint64_t output_intervals = 0;
  std::uint64_t source_visits = 0;
  std::uint32_t max_active_slabs = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_begin_bytes = 0;
  std::uint64_t device_free_low_bytes = 0;
  double elapsed_ms = 0.0;
};

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
    bool count_only = false)
{
  const auto begin = Clock::now();
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
      cudaMemcpy(&source_low, source.xs, sizeof(source_low),
                 cudaMemcpyDeviceToHost),
      "source lower x D2H");
  cuda_require(
      cudaMemcpy(
          &source_high, source.xs + source.x_slabs,
          sizeof(source_high), cudaMemcpyDeviceToHost),
      "source upper x D2H");
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
      launch_blocks(source_endpoint_count), kThreads>>>(
      source.xs, source_endpoint_count, radius,
      thrust::raw_pointer_cast(result.xs.data()));
  cuda_require(cudaGetLastError(), "shift morphology x endpoints");
  thrust::sort(thrust::device, result.xs.begin(), result.xs.end());
  result.xs.erase(
      thrust::unique(
          thrust::device, result.xs.begin(), result.xs.end()),
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
            thrust::device, result.xs.begin(), result.xs.end(),
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
  result.slab_counts.assign(output_slabs, 0);
  result.slab_offsets.resize(output_slabs);
  thrust::device_vector<std::uint32_t> status(1, 0);
  thrust::device_vector<unsigned long long> source_visits(1, 0);
  thrust::device_vector<std::uint32_t> observed_max_active(1, 0);
  sample_morph_memory(metrics);

  count_morph_intervals_kernel<<<
      launch_blocks(output_slabs), kThreads>>>(
      source, thrust::raw_pointer_cast(result.xs.data()),
      output_slabs, radius, erosion, limits.max_active_slabs,
      limits.max_source_visits_per_band,
      thrust::raw_pointer_cast(result.slab_counts.data()),
      thrust::raw_pointer_cast(status.data()),
      thrust::raw_pointer_cast(source_visits.data()),
      thrust::raw_pointer_cast(observed_max_active.data()));
  cuda_require(cudaGetLastError(), "count morphology intervals");
  thrust::exclusive_scan(
      thrust::device, result.slab_counts.begin(),
      result.slab_counts.end(), result.slab_offsets.begin(),
      std::uint64_t{0});

  std::uint64_t final_offset = 0;
  std::uint32_t final_count = 0;
  cuda_require(
      cudaMemcpy(
          &final_offset,
          thrust::raw_pointer_cast(result.slab_offsets.data()) +
              output_slabs - 1,
          sizeof(final_offset), cudaMemcpyDeviceToHost),
      "last morphology offset D2H");
  cuda_require(
      cudaMemcpy(
          &final_count,
          thrust::raw_pointer_cast(result.slab_counts.data()) +
              output_slabs - 1,
          sizeof(final_count), cudaMemcpyDeviceToHost),
      "last morphology count D2H");
  if (final_offset >
      std::numeric_limits<std::uint64_t>::max() - final_count) {
    throw std::runtime_error("output interval count overflow");
  }
  const std::uint64_t output_intervals = final_offset + final_count;

  std::uint32_t host_status = copy_device_status(status);
  cuda_require(
      cudaMemcpy(
          &metrics->source_visits,
          thrust::raw_pointer_cast(source_visits.data()),
          sizeof(metrics->source_visits), cudaMemcpyDeviceToHost),
      "morph source visits D2H");
  cuda_require(
      cudaMemcpy(
          &metrics->max_active_slabs,
          thrust::raw_pointer_cast(observed_max_active.data()),
          sizeof(metrics->max_active_slabs), cudaMemcpyDeviceToHost),
      "morph max active D2H");
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
        cudaDeviceSynchronize(), "count-only morphology synchronize");
    sample_morph_memory(metrics);
    metrics->elapsed_ms = elapsed_ms(begin, Clock::now());
    return result;
  }

  result.intervals.resize(output_intervals);
  sample_morph_memory(metrics);
  if (output_intervals) {
    emit_morph_intervals_kernel<<<
        launch_blocks(output_slabs), kThreads>>>(
        source, thrust::raw_pointer_cast(result.xs.data()),
        output_slabs, radius, erosion, limits.max_active_slabs,
        limits.max_source_visits_per_band,
        thrust::raw_pointer_cast(result.slab_offsets.data()),
        thrust::raw_pointer_cast(result.slab_counts.data()),
        thrust::raw_pointer_cast(result.intervals.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "emit morphology intervals");
    validate_morph_intervals_kernel<<<
        launch_blocks(output_intervals), kThreads>>>(
        thrust::raw_pointer_cast(result.intervals.data()),
        output_intervals,
        thrust::raw_pointer_cast(result.slab_offsets.data()),
        thrust::raw_pointer_cast(result.slab_counts.data()),
        output_slabs, thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "validate morphology intervals");
  }
  cuda_require(cudaDeviceSynchronize(), "morphology synchronize");
  host_status = copy_device_status(status);
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
    const DeviceBandSet &bands, std::uint64_t max_segments,
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
      launch_blocks(boundary_count), kThreads>>>(
      view.intervals, view.slab_offsets, view.slab_counts,
      view.x_slabs,
      thrust::raw_pointer_cast(vertical_counts.data()));
  cuda_require(cudaGetLastError(), "count morphology vertical boundary");
  thrust::exclusive_scan(
      thrust::device, vertical_counts.begin(), vertical_counts.end(),
      vertical_offsets.begin(), std::uint64_t{0});

  std::uint64_t final_offset = 0;
  std::uint64_t final_count = 0;
  cuda_require(
      cudaMemcpy(
          &final_offset,
          thrust::raw_pointer_cast(vertical_offsets.data()) +
              boundary_count - 1,
          sizeof(final_offset), cudaMemcpyDeviceToHost),
      "morph vertical offset D2H");
  cuda_require(
      cudaMemcpy(
          &final_count,
          thrust::raw_pointer_cast(vertical_counts.data()) +
              boundary_count - 1,
          sizeof(final_count), cudaMemcpyDeviceToHost),
      "morph vertical count D2H");
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
  if (raw_count > max_segments) {
    throw std::runtime_error("morph boundary capacity");
  }

  thrust::device_vector<DirectedSegmentI64> raw(raw_count);
  if (memory_metrics) sample_morph_memory(memory_metrics);
  if (view.interval_count) {
    emit_horizontal_band_boundary_kernel<<<
        launch_blocks(view.interval_count), kThreads>>>(
        view.intervals, view.interval_count, view.xs,
        thrust::raw_pointer_cast(raw.data()));
    cuda_require(cudaGetLastError(), "emit morph horizontal boundary");
  }
  thrust::device_vector<std::uint32_t> status(1, 0);
  if (vertical_count) {
    emit_vertical_xor_kernel<<<
        launch_blocks(boundary_count), kThreads>>>(
        view.intervals, view.slab_offsets, view.slab_counts,
        view.x_slabs, view.xs,
        thrust::raw_pointer_cast(vertical_counts.data()),
        thrust::raw_pointer_cast(vertical_offsets.data()),
        thrust::raw_pointer_cast(raw.data()) + 2 * view.interval_count,
        thrust::raw_pointer_cast(status.data()));
    cuda_require(cudaGetLastError(), "emit morph vertical boundary");
  }
  cuda_require(cudaDeviceSynchronize(), "morph boundary emit synchronize");
  if (copy_device_status(status)) {
    throw std::runtime_error("morph vertical boundary invariant");
  }
  canonicalize_device_segments(&raw);
  cuda_require(
      cudaDeviceSynchronize(), "morph boundary canonicalize synchronize");
  if (memory_metrics) sample_morph_memory(memory_metrics);
  return raw;
}

std::vector<DirectedSegmentI64> boundary_from_bands(
    const DeviceBandSet &bands, std::uint64_t max_segments)
{
  thrust::device_vector<DirectedSegmentI64> raw =
      device_boundary_from_bands(bands, max_segments);
  std::vector<DirectedSegmentI64> host(raw.size());
  if (!host.empty()) {
    cuda_require(
        cudaMemcpy(
            host.data(), thrust::raw_pointer_cast(raw.data()),
            host.size() * sizeof(DirectedSegmentI64),
            cudaMemcpyDeviceToHost),
        "morph boundary D2H");
  }
  return host;
}

enum class GateOperation
{
  erode,
  dilate,
  erode_then_dilate,
};

struct GateContext
{
  GateOperation operation = GateOperation::erode;
  std::int64_t first_radius = 1;
  std::int64_t second_radius = 0;
  bool invoked = false;
  MorphMetrics first_metrics;
  MorphMetrics second_metrics;
  std::vector<DirectedSegmentI64> boundary;
};

void gate_consume_strips(
    const std::int64_t *xs, std::uint32_t x_slabs,
    const StripInterval *intervals, std::uint64_t interval_count,
    const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque)
{
  auto *context = static_cast<GateContext *>(opaque);
  if (!context || context->invoked) {
    throw std::runtime_error("resident callback state");
  }
  context->invoked = true;
  if (!x_slabs) throw std::runtime_error("resident callback empty slabs");

  std::uint64_t final_offset = 0;
  std::uint32_t final_count = 0;
  cuda_require(
      cudaMemcpy(
          &final_offset, slab_offsets + x_slabs - 1,
          sizeof(final_offset), cudaMemcpyDeviceToHost),
      "resident final offset D2H");
  cuda_require(
      cudaMemcpy(
          &final_count, slab_counts + x_slabs - 1,
          sizeof(final_count), cudaMemcpyDeviceToHost),
      "resident final count D2H");
  if (final_offset + final_count != interval_count) {
    throw std::runtime_error("resident interval census mismatch");
  }

  const DeviceBandView source = {
      xs, x_slabs, intervals, interval_count, slab_offsets, slab_counts};
  const MorphLimits limits;
  const bool first_erosion =
      context->operation != GateOperation::dilate;
  DeviceBandSet first = morph_bands(
      source, context->first_radius, first_erosion, limits,
      &context->first_metrics);
  if (context->operation == GateOperation::erode_then_dilate) {
    if (!first.x_slabs()) {
      context->boundary.clear();
      return;
    }
    DeviceBandSet second = morph_bands(
        first.view(), context->second_radius, false, limits,
        &context->second_metrics);
    context->boundary = boundary_from_bands(
        second, limits.max_output_intervals * 4);
  } else {
    context->boundary = boundary_from_bands(
        first, limits.max_output_intervals * 4);
  }
}

using Cell = std::pair<int, int>;
using Cells = std::set<Cell>;

Cells rasterize(const std::vector<RectI64> &rectangles)
{
  Cells cells;
  for (const RectI64 &rectangle : rectangles) {
    for (std::int64_t x = rectangle.left; x < rectangle.right; ++x) {
      for (std::int64_t y = rectangle.bottom; y < rectangle.top; ++y) {
        cells.emplace(static_cast<int>(x), static_cast<int>(y));
      }
    }
  }
  return cells;
}

Cells raster_dilate(const Cells &input, int radius)
{
  Cells result;
  for (const Cell &cell : input) {
    for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        result.emplace(cell.first + dx, cell.second + dy);
      }
    }
  }
  return result;
}

Cells raster_erode(const Cells &input, int radius)
{
  Cells result;
  for (const Cell &cell : input) {
    bool keep = true;
    for (int dx = -radius; dx <= radius && keep; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        if (!input.count({cell.first + dx, cell.second + dy})) {
          keep = false;
          break;
        }
      }
    }
    if (keep) result.insert(cell);
  }
  return result;
}

std::vector<DirectedSegmentI64> raster_boundary(const Cells &cells)
{
  std::vector<DirectedSegmentI64> result;
  result.reserve(cells.size() * 2);
  for (const Cell &cell : cells) {
    const std::int64_t x = cell.first;
    const std::int64_t y = cell.second;
    if (!cells.count({cell.first, cell.second - 1})) {
      result.push_back(
          {y, x, x + 1, -1, SegmentAxis::horizontal});
    }
    if (!cells.count({cell.first, cell.second + 1})) {
      result.push_back(
          {y + 1, x, x + 1, 1, SegmentAxis::horizontal});
    }
    if (!cells.count({cell.first - 1, cell.second})) {
      result.push_back({x, y, y + 1, -1, SegmentAxis::vertical});
    }
    if (!cells.count({cell.first + 1, cell.second})) {
      result.push_back({x + 1, y, y + 1, 1, SegmentAxis::vertical});
    }
  }
  canonicalize_segments(&result);
  return result;
}

std::string operation_name(GateOperation operation)
{
  if (operation == GateOperation::erode) return "erode";
  if (operation == GateOperation::dilate) return "dilate";
  return "erode-dilate";
}

void require_boundary_equal(
    const std::string &name,
    const std::vector<DirectedSegmentI64> &expected,
    const std::vector<DirectedSegmentI64> &actual)
{
  if (expected.size() != actual.size()) {
    std::ostringstream stream;
    stream << name << ": boundary size expected=" << expected.size()
           << " actual=" << actual.size();
    throw std::runtime_error(stream.str());
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const DirectedSegmentI64 &left = expected[index];
    const DirectedSegmentI64 &right = actual[index];
    if (left.fixed != right.fixed || left.lo != right.lo ||
        left.hi != right.hi || left.side != right.side ||
        left.axis != right.axis) {
      throw std::runtime_error(
          name + ": boundary mismatch index=" + std::to_string(index) +
          " expected='" + segment_string(expected[index]) +
          "' actual='" + segment_string(actual[index]) + "'");
    }
  }
}

void run_gate_case(
    const std::string &name, const std::vector<RectI64> &rectangles,
    GateOperation operation, int first_radius, int second_radius,
    int device)
{
  Cells expected_cells = rasterize(rectangles);
  if (operation == GateOperation::erode) {
    expected_cells = raster_erode(expected_cells, first_radius);
  } else if (operation == GateOperation::dilate) {
    expected_cells = raster_dilate(expected_cells, first_radius);
  } else {
    expected_cells = raster_erode(expected_cells, first_radius);
    expected_cells = raster_dilate(expected_cells, second_radius);
  }
  const std::vector<DirectedSegmentI64> expected =
      raster_boundary(expected_cells);

  GateContext context;
  context.operation = operation;
  context.first_radius = first_radius;
  context.second_radius = second_radius;
  ResidentStripHook hook;
  hook.consume = gate_consume_strips;
  hook.context = &context;
  hook.stop_before_boundary = true;
  Limits limits;
  limits.max_slabs_per_rectangle = 4096;
  const UnionOutput union_output =
      gpu_union(rectangles, limits, device, &hook);
  if (union_output.fallback) {
    throw std::runtime_error(
        name + ": union/callback fallback: " + union_output.message);
  }
  if (!context.invoked) {
    throw std::runtime_error(name + ": resident callback not invoked");
  }
  require_boundary_equal(
      name + "/" + operation_name(operation), expected,
      context.boundary);
}

void run_morph_x_guard_case(
    const std::string &name, std::int64_t low, std::int64_t split,
    std::int64_t high, bool expect_fallback, int device)
{
  constexpr std::int64_t radius = 10;
  constexpr std::int64_t bottom = 0;
  constexpr std::int64_t top = 4;
  if (!(low < split && split < high)) {
    throw std::runtime_error(name + ": malformed guard fixture");
  }
  const std::vector<RectI64> rectangles = {
      {low, bottom, split, top, 0, 0},
      {split, bottom, high, top, 0, 0}};
  GateContext context;
  context.operation = GateOperation::dilate;
  context.first_radius = radius;
  ResidentStripHook hook;
  hook.consume = gate_consume_strips;
  hook.context = &context;
  hook.stop_before_boundary = true;
  Limits limits;
  limits.max_slabs_per_rectangle = 4096;
  const UnionOutput output = gpu_union(rectangles, limits, device, &hook);
  if (expect_fallback) {
    if (!output.fallback || !context.invoked ||
        !output.segments.empty() ||
        output.message.find("x coordinate overflow") ==
            std::string::npos) {
      throw std::runtime_error(
          name + ": unsafe x arithmetic did not fail closed: " +
          output.message);
    }
    return;
  }
  if (output.fallback || !context.invoked) {
    throw std::runtime_error(
        name + ": just-inside x arithmetic was rejected: " +
        output.message);
  }
  const std::vector<DirectedSegmentI64> expected = {
      {bottom - radius, low - radius, high + radius, -1,
       SegmentAxis::horizontal},
      {top + radius, low - radius, high + radius, 1,
       SegmentAxis::horizontal},
      {low - radius, bottom - radius, top + radius, -1,
       SegmentAxis::vertical},
      {high + radius, bottom - radius, top + radius, 1,
       SegmentAxis::vertical}};
  require_boundary_equal(name, expected, context.boundary);
}

void run_morph_x_guard_gate(int device)
{
  constexpr std::int64_t radius = 10;
  constexpr std::int64_t low_limit =
      std::numeric_limits<std::int64_t>::min() / 2;
  constexpr std::int64_t high_limit =
      std::numeric_limits<std::int64_t>::max() / 2;

  const std::int64_t rejected_low = low_limit + radius;
  run_morph_x_guard_case(
      "x-guard-low-reject", rejected_low, rejected_low + 1,
      rejected_low + 100, true, device);
  const std::int64_t accepted_low = low_limit + 2 * radius;
  run_morph_x_guard_case(
      "x-guard-low-accept", accepted_low, accepted_low + 1,
      accepted_low + 100, false, device);

  const std::int64_t rejected_high = high_limit - radius;
  run_morph_x_guard_case(
      "x-guard-high-reject", rejected_high - 100,
      rejected_high - 1, rejected_high, true, device);
  const std::int64_t accepted_high = high_limit - 2 * radius;
  run_morph_x_guard_case(
      "x-guard-high-accept", accepted_high - 100,
      accepted_high - 1, accepted_high, false, device);

  std::cout << "M2_RESIDENT_MORPH_X_GUARD PASS checks=4"
            << " rejected=2 accepted_exact=2\n";
}

std::vector<std::pair<std::string, std::vector<RectI64>>>
morphology_fixtures()
{
  return {
      {"single", {{0, 0, 7, 6}}},
      {"threshold-width-1", {{0, 0, 1, 7}}},
      {"threshold-width-2", {{0, 0, 2, 7}}},
      {"threshold-width-3", {{0, 0, 3, 7}}},
      {"threshold-width-4", {{0, 0, 4, 7}}},
      {"threshold-width-5", {{0, 0, 5, 7}}},
      {"gap-2", {{0, 0, 3, 4}, {5, 0, 8, 4}}},
      {"gap-3", {{0, 0, 3, 4}, {6, 0, 9, 4}}},
      {"corner-touch", {{0, 0, 3, 3}, {3, 3, 6, 6}}},
      {"edge-touch", {{0, 0, 3, 4}, {3, 1, 7, 5}}},
      {"notch", {{0, 0, 9, 3}, {0, 3, 3, 9}, {6, 3, 9, 9}}},
      {"hole",
       {{0, 0, 9, 2}, {0, 7, 9, 9}, {0, 2, 2, 7},
        {7, 2, 9, 7}}},
      {"thin-bridge",
       {{0, 0, 4, 6}, {8, 0, 12, 6}, {4, 2, 8, 3}}},
      {"overlap",
       {{-4, -2, 5, 3}, {-1, -5, 3, 7}, {2, 1, 8, 5}}},
  };
}

void run_differential_gate(int device)
{
  std::uint64_t checks = 0;
  for (const auto &fixture : morphology_fixtures()) {
    run_gate_case(
        fixture.first + "-e1", fixture.second,
        GateOperation::erode, 1, 0, device);
    ++checks;
    run_gate_case(
        fixture.first + "-d1", fixture.second,
        GateOperation::dilate, 1, 0, device);
    ++checks;
    run_gate_case(
        fixture.first + "-e1d2", fixture.second,
        GateOperation::erode_then_dilate, 1, 2, device);
    ++checks;
    run_gate_case(
        fixture.first + "-e2", fixture.second,
        GateOperation::erode, 2, 0, device);
    ++checks;
  }

  std::mt19937 generator(0x4d325f39);
  std::uniform_int_distribution<int> coordinate(-8, 7);
  std::uniform_int_distribution<int> extent(1, 7);
  std::uniform_int_distribution<int> rectangle_count(1, 9);
  for (int trial = 0; trial < 64; ++trial) {
    std::vector<RectI64> rectangles;
    const int count = rectangle_count(generator);
    for (int index = 0; index < count; ++index) {
      const int left = coordinate(generator);
      const int bottom = coordinate(generator);
      rectangles.push_back(
          {left, bottom, left + extent(generator),
           bottom + extent(generator)});
    }
    const std::string prefix = "random-" + std::to_string(trial);
    run_gate_case(
        prefix + "-e1", rectangles, GateOperation::erode, 1, 0,
        device);
    ++checks;
    run_gate_case(
        prefix + "-d2", rectangles, GateOperation::dilate, 2, 0,
        device);
    ++checks;
    run_gate_case(
        prefix + "-e1d2", rectangles,
        GateOperation::erode_then_dilate, 1, 2, device);
    ++checks;
  }
  std::cout << "M2_RESIDENT_MORPH_DIFFERENTIAL PASS checks=" << checks
            << " directed=" << morphology_fixtures().size()
            << " random=64\n";
}

constexpr char kGt90GoldenFileSha256[] =
    "e7149202ef0ace74618a01f56135ea1cde2b4a0bdb00102f9a5f4a072af2ea49";
constexpr char kGt90GoldenPayloadSha256[] =
    "233a611bc306126b0292763954aab2d984508f2466a56ced2d1cf08af6c526ff";
constexpr std::uint64_t kGt90GoldenSegments = UINT64_C(4254384);
constexpr std::uint64_t kGt90GoldenFnv64 =
    UINT64_C(2057677162565968634);
constexpr std::uint64_t kGt90LongSegments = UINT64_C(8);
constexpr std::uint64_t kGt90LongPairs =
    kGt90LongSegments * (kGt90LongSegments - 1) / 2;
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

struct LongSpaceCertificate
{
  std::uint64_t pairs_checked = 0;
  std::uint64_t violations = 0;
  std::uint64_t uncertain = 0;
};

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

LongSpaceCertificate certify_f90_long_edge_space(
    const std::vector<DirectedSegmentI64> &segments)
{
  if (segments.size() > kF90LongSpacePairwiseSegmentCountCap) {
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

void require_long_space_certificate(
    const std::string &name,
    const std::vector<DirectedSegmentI64> &segments,
    std::uint64_t expected_violations,
    std::uint64_t expected_uncertain)
{
  const LongSpaceCertificate result =
      certify_f90_long_edge_space(segments);
  const std::uint64_t expected_pairs =
      segments.size() > 1
          ? static_cast<std::uint64_t>(segments.size()) *
                static_cast<std::uint64_t>(segments.size() - 1) / 2
          : 0;
  if (result.pairs_checked != expected_pairs ||
      result.violations != expected_violations ||
      result.uncertain != expected_uncertain) {
    std::ostringstream message;
    message << name << ": pairs=" << result.pairs_checked
            << " violations=" << result.violations
            << " uncertain=" << result.uncertain;
    throw std::runtime_error(message.str());
  }
}

void run_f90_long_space_certificate_gate()
{
  const DirectedSegmentI64 east = {
      0, 0, 200, 1, SegmentAxis::horizontal};
  require_long_space_certificate(
      "horizontal-179",
      {east, {179, 0, 200, -1, SegmentAxis::horizontal}}, 1, 0);
  require_long_space_certificate(
      "horizontal-180",
      {east, {180, 0, 200, -1, SegmentAxis::horizontal}}, 0, 0);
  require_long_space_certificate(
      "horizontal-181",
      {east, {181, 0, 200, -1, SegmentAxis::horizontal}}, 0, 0);
  require_long_space_certificate(
      "wrong-exterior-side",
      {east, {-179, 0, 200, -1, SegmentAxis::horizontal}}, 0, 0);
  require_long_space_certificate(
      "vertical-179",
      {{0, 0, 200, 1, SegmentAxis::vertical},
       {179, 0, 200, -1, SegmentAxis::vertical}},
      1, 0);
  require_long_space_certificate(
      "corner-exact-108-144-180",
      {{0, 0, 100, 1, SegmentAxis::horizontal},
       {144, 208, 300, -1, SegmentAxis::horizontal}},
      0, 0);
  require_long_space_certificate(
      "corner-inside-107-144",
      {{0, 0, 100, 1, SegmentAxis::horizontal},
       {144, 207, 300, -1, SegmentAxis::horizontal}},
      1, 0);
  require_long_space_certificate(
      "unsafe-signed-span",
      {{0, std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max(), 1,
        SegmentAxis::horizontal},
       {179, std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max(), -1,
        SegmentAxis::horizontal}},
      0, 1);
  std::cout
      << "M2_RESIDENT_F90_LONG_SPACE_GATE PASS checks=8"
      << " threshold_dbu="
      << m1ws::kM2F90LongSpaceCoordinateDistance << "\n";
}

struct ProductionMorphContext
{
  bool qualify_boundary = false;
  bool invoked = false;
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
    const std::int64_t *xs, std::uint32_t x_slabs,
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

  if (!x_slabs) {
    throw std::runtime_error("production source has no x slabs");
  }
  std::uint64_t final_offset = 0;
  std::uint32_t final_count = 0;
  cuda_require(
      cudaMemcpy(
          &final_offset, slab_offsets + x_slabs - 1,
          sizeof(final_offset), cudaMemcpyDeviceToHost),
      "production source final offset D2H");
  cuda_require(
      cudaMemcpy(
          &final_count, slab_counts + x_slabs - 1,
          sizeof(final_count), cudaMemcpyDeviceToHost),
      "production source final count D2H");
  if (final_offset >
          std::numeric_limits<std::uint64_t>::max() - final_count ||
      final_offset + final_count != interval_count) {
    throw std::runtime_error(
        "production resident source interval census mismatch");
  }

  const DeviceBandView source = {
      xs, x_slabs, intervals, interval_count, slab_offsets, slab_counts};
  MorphLimits limits;
  // The exact r=269 count-only pass revisits 4.682 billion source
  // intervals across overlapping x windows on the pinned production scene.
  // Keep a measured, finite production allowance rather than disabling the
  // fail-closed work gate.
  limits.max_total_source_visits = UINT64_C(8000000000);
  DeviceBandSet eroded89;
  try {
    eroded89 = morph_bands(
        source, 89, true, limits, &context->erode89);
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
        eroded89.view(), 90, false, limits, &context->dilate90);
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
          gt90, limits.max_output_intervals * 4,
          &context->boundary_memory);
  context->gt90_boundary_segments = boundary.size();
  context->gt90_long_segments = thrust::count_if(
      thrust::device, boundary.begin(), boundary.end(),
      LongSegment{600});
  if (context->gt90_long_segments >
      kF90LongSpacePairwiseSegmentCountCap) {
    throw std::runtime_error(
        "production F90 long-edge pairwise capacity");
  }
  thrust::device_vector<DirectedSegmentI64> long_segments(
      context->gt90_long_segments);
  const auto long_end = thrust::copy_if(
      thrust::device, boundary.begin(), boundary.end(),
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
        cudaMemcpy(
            host_long_segments.data(),
            thrust::raw_pointer_cast(long_segments.data()),
            host_long_segments.size() * sizeof(DirectedSegmentI64),
            cudaMemcpyDeviceToHost),
        "production F90 long edges D2H");
  }
  const LongSpaceCertificate long_space =
      certify_f90_long_edge_space(host_long_segments);
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
          cudaMemcpy(
              context->qualified_boundary.data(),
              thrust::raw_pointer_cast(boundary.data()),
              boundary.size() * sizeof(DirectedSegmentI64),
              cudaMemcpyDeviceToHost),
          "qualified gt90 boundary D2H");
    }
  }
  cuda_require(
      cudaDeviceSynchronize(), "production gt90 boundary synchronize");
  context->boundary_ms = elapsed_ms(boundary_begin, Clock::now());
  include_memory_sample(context->boundary_memory, context);
  release_device_vector(&boundary);

  DeviceBandSet gt270_eroded;
  try {
    gt270_eroded = morph_bands(
        gt90.view(), 269, true, limits, &context->erode269_count, true);
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

void require_production_boundary_equal(
    const std::vector<m2oracle::DirectedSegmentI64> &expected,
    const std::vector<DirectedSegmentI64> &actual)
{
  if (expected.size() != kGt90GoldenSegments ||
      actual.size() != expected.size() ||
      digest_segments(actual) != kGt90GoldenFnv64) {
    std::ostringstream stream;
    stream << "gt90 boundary census/digest mismatch expected_count="
           << expected.size() << " actual_count=" << actual.size()
           << " actual_fnv64=" << digest_segments(actual);
    throw std::runtime_error(stream.str());
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const DirectedSegmentI64 &left = actual[index];
    const m2oracle::DirectedSegmentI64 &right = expected[index];
    if (left.fixed != right.fixed || left.lo != right.lo ||
        left.hi != right.hi || left.side != right.side ||
        static_cast<std::uint32_t>(left.axis) !=
            static_cast<std::uint32_t>(right.axis)) {
      throw std::runtime_error(
          "gt90 exact boundary first differs at segment " +
          std::to_string(index) + " actual='" + segment_string(left) +
          "'");
    }
  }
}

double production_resident_peak_delta_mib(
    const ProductionMorphContext &context)
{
  return (context.callback_device_free_begin_bytes -
          context.callback_device_free_low_bytes) /
         (1024.0 * 1024.0);
}

double production_resident_peak_in_use_mib(
    const ProductionMorphContext &context)
{
  return (context.callback_device_total_bytes -
          context.callback_device_free_low_bytes) /
         (1024.0 * 1024.0);
}

void print_production_morph_timing(
    std::uint32_t run, const UnionOutput &output,
    const ProductionMorphContext &context)
{
  std::cout
      << "M2_RESIDENT_F90_GPU"
      << " run=" << run
      << " qualification=" << context.qualify_boundary
      << " union_resident_total_ms=" << std::fixed
      << std::setprecision(3) << output.total_ms
      << " erode89_ms=" << context.erode89.elapsed_ms
      << " dilate90_ms=" << context.dilate90.elapsed_ms
      << " boundary_and_long_space_ms=" << context.boundary_ms
      << " erode269_count_ms=" << context.erode269_count.elapsed_ms
      << " resident_callback_ms=" << context.callback_ms
      << " gt90_intervals=" << context.dilate90.output_intervals
      << " gt90_segments=" << context.gt90_boundary_segments
      << " gt90_long_segments=" << context.gt90_long_segments
      << " gt90_space_pairs_checked="
      << context.gt90_space_pairs_checked
      << " gt90_space_violations="
      << context.gt90_space_violations
      << " gt90_space_uncertain="
      << context.gt90_space_uncertain
      << " gt270_eroded_intervals="
      << context.gt270_eroded_intervals
      << " erode89_max_active=" << context.erode89.max_active_slabs
      << " dilate90_max_active=" << context.dilate90.max_active_slabs
      << " erode269_max_active="
      << context.erode269_count.max_active_slabs
      << " resident_peak_delta_mib=" << std::setprecision(1)
      << production_resident_peak_delta_mib(context)
      << " resident_peak_in_use_mib="
      << production_resident_peak_in_use_mib(context)
      << std::setprecision(3) << "\n";
}

void run_production_morphology(
    const std::string &kact_path, const std::string &gt90_path,
    std::uint32_t repeat, int device)
{
  if (repeat < 2) {
    throw std::runtime_error(
        "production morphology requires qualification plus a warm run");
  }
  const auto all_begin = Clock::now();
  const auto load_begin = Clock::now();
  m2prod::LoadOptions load_options;
  load_options.expected_scene_sha256 = kProductionM2SceneSha256;
  load_options.expected_flat_polygons = kProductionM2FlatPolygons;
  load_options.expected_flat_rectangles = kProductionM2FlatRectangles;
  const m2prod::CompactScene scene =
      m2prod::load_kact_templates(kact_path, load_options);
  const double load_ms = elapsed_ms(load_begin, Clock::now());
  double host_expand_ms = 0.0;
  const std::vector<RectI64> rectangles =
      expand_production_rectangles(scene, &host_expand_ms);

  const auto golden_begin = Clock::now();
  const std::string gt90_file_sha256 =
      m2oracle::candidate_stream_file_sha256(gt90_path);
  if (gt90_file_sha256 != kGt90GoldenFileSha256) {
    throw std::runtime_error("gt90 golden file SHA-256 mismatch");
  }
  m2oracle::BoundaryOracle golden_identity;
  golden_identity.scene_sha256 = kProductionM2OracleSceneSha256;
  const std::vector<m2oracle::DirectedSegmentI64> golden =
      m2oracle::read_candidate_stream(gt90_path, golden_identity);
  if (m2oracle::canonical_boundary_sha256(golden) !=
          kGt90GoldenPayloadSha256 ||
      m2oracle::canonical_boundary_fnv64(golden) != kGt90GoldenFnv64 ||
      golden.size() != kGt90GoldenSegments) {
    throw std::runtime_error("gt90 golden payload identity mismatch");
  }
  const double golden_ms = elapsed_ms(golden_begin, Clock::now());

  Limits union_limits;
  union_limits.max_memberships = UINT64_C(100000000);
  union_limits.max_events = UINT64_C(200000000);
  union_limits.max_segments = UINT64_C(8000000);
  union_limits.max_slabs_per_rectangle = 64;

  std::vector<double> warm_union_resident_ms;
  std::vector<double> warm_callback_ms;
  std::vector<double> warm_peak_delta_mib;
  double qualification_ms = 0.0;
  for (std::uint32_t run = 0; run < repeat; ++run) {
    ProductionMorphContext context;
    context.qualify_boundary = run == 0;
    ResidentStripHook hook;
    hook.consume = production_consume_strips;
    hook.context = &context;
    hook.stop_before_boundary = true;
    const UnionOutput output =
        gpu_union(rectangles, union_limits, device, &hook);
    if (output.fallback) {
      throw std::runtime_error(
          "production union/resident morphology fallback: " +
          output.message);
    }
    if (!context.invoked ||
        context.gt90_boundary_segments != kGt90GoldenSegments ||
        context.gt90_long_segments != kGt90LongSegments ||
        context.gt90_space_pairs_checked != kGt90LongPairs ||
        context.gt90_space_violations != 0 ||
        context.gt90_space_uncertain != 0 ||
        context.gt270_eroded_intervals != 0) {
      throw std::runtime_error(
          "production resident morphology census mismatch");
    }
    if (run == 0) {
      require_production_boundary_equal(
          golden, context.qualified_boundary);
      qualification_ms = output.total_ms;
      std::cout
          << "M2_RESIDENT_F90_QUALIFICATION PASS"
          << " compared_edges=" << context.qualified_boundary.size()
          << " boundary_fnv64="
          << digest_segments(context.qualified_boundary)
          << " golden_file_sha256=" << gt90_file_sha256 << "\n";
    } else {
      warm_union_resident_ms.push_back(output.total_ms);
      warm_callback_ms.push_back(context.callback_ms);
      warm_peak_delta_mib.push_back(
          production_resident_peak_delta_mib(context));
    }
    print_production_morph_timing(run, output, context);
  }

  const double warm_union_resident =
      median(warm_union_resident_ms);
  const double warm_callback = median(warm_callback_ms);
  const double warm_peak_delta = median(warm_peak_delta_mib);
  // Keep the performance comparisons like-for-like.  The full 22.077-second
  // offline bridge also includes the separate 3.485-second M2.1/.2 checks,
  // which this resident morphology callback does not yet perform.
  constexpr double stock_f90_f270_ms = 14691.0;
  constexpr double stock_union_stitch_f90_f270_ms =
      1707.380 + 2194.0 + stock_f90_f270_ms;
  const double charged_resident_pipeline_ms =
      load_ms + host_expand_ms + warm_union_resident;
  const double callback_reduction =
      100.0 * (stock_f90_f270_ms - warm_callback) /
      stock_f90_f270_ms;
  const double charged_pipeline_reduction =
      100.0 *
      (stock_union_stitch_f90_f270_ms - charged_resident_pipeline_ms) /
      stock_union_stitch_f90_f270_ms;
  std::cout
      << "M2_RESIDENT_F90_PRODUCTION PASS"
      << " rectangles=" << rectangles.size()
      << " qualification_union_resident_ms=" << std::fixed
      << std::setprecision(3) << qualification_ms
      << " warm_union_resident_median_ms=" << warm_union_resident
      << " warm_callback_median_ms=" << warm_callback
      << " compact_load_ms=" << load_ms
      << " host_expand_ms=" << host_expand_ms
      << " golden_load_and_hash_ms=" << golden_ms
      << " warm_resident_peak_delta_mib=" << std::setprecision(1)
      << warm_peak_delta
      << std::setprecision(3)
      << " stock_f90_f270_ms=" << stock_f90_f270_ms
      << " resident_suffix_less_time_pct="
      << callback_reduction
      << " stock_union_stitch_f90_f270_ms="
      << stock_union_stitch_f90_f270_ms
      << " charged_resident_pipeline_ms="
      << charged_resident_pipeline_ms
      << " charged_pipeline_less_time_pct="
      << charged_pipeline_reduction
      << " verification_total_ms="
      << elapsed_ms(all_begin, Clock::now()) << "\n";
}

void print_morph_help(const char *program)
{
  std::cout
      << "Usage: " << program
      << " --self-test [--device N]\n"
      << "       " << program
      << " --production --kact FILE --gt90-golden FILE"
         " [--repeat N] [--device N]\n";
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    int device = 0;
    bool self_test = false;
    bool production = false;
    std::uint32_t repeat = 4;
    std::string kact_path;
    std::string gt90_path;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--self-test") {
        self_test = true;
      } else if (argument == "--production") {
        production = true;
      } else if (argument == "--kact" && index + 1 < argc) {
        kact_path = argv[++index];
      } else if (argument == "--gt90-golden" &&
                 index + 1 < argc) {
        gt90_path = argv[++index];
      } else if (argument == "--repeat" && index + 1 < argc) {
        repeat = parse_u32(argv[++index], "--repeat");
      } else if (argument == "--device" && index + 1 < argc) {
        device = std::stoi(argv[++index]);
      } else if (argument == "--help" || argument == "-h") {
        print_morph_help(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + argument);
      }
    }
    if (self_test == production) {
      print_morph_help(argv[0]);
      return 2;
    }
    cuda_require(cudaSetDevice(device), "morph cudaSetDevice");
    if (self_test) {
      run_morph_x_guard_gate(device);
      run_f90_long_space_certificate_gate();
      run_differential_gate(device);
    } else {
      if (kact_path.empty() || gt90_path.empty()) {
        print_morph_help(argv[0]);
        return 2;
      }
      run_production_morphology(
          kact_path, gt90_path, repeat, device);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M2_RESIDENT_MORPH_ERROR " << error.what() << "\n";
    return 1;
  }
}
