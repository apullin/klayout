/*
 * Differential and fail-closed gate for the opt-in bounded-event strip
 * producer.  This test intentionally forces seams through long rectangles,
 * empty gaps and overlapping coverage, then compares the one-shot resident
 * view byte-for-byte with the historical whole-input sweep.
 */

#include "manhattan_union_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace mu = klayout_cuda::manhattan_union;

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

mu::RectI64 rect(
    std::int64_t left, std::int64_t bottom, std::int64_t right,
    std::int64_t top)
{
  return {left, bottom, right, top, 0, 0};
}

struct StripCapture
{
  unsigned int invocations = 0;
  std::vector<std::int64_t> xs;
  std::vector<mu::StripInterval> intervals;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint32_t> counts;
};

void capture_strips(
    cudaStream_t stream, const std::int64_t *device_xs,
    std::uint32_t x_slabs,
    const mu::StripInterval *device_intervals,
    std::uint64_t interval_count,
    const std::uint64_t *device_offsets,
    const std::uint32_t *device_counts, void *opaque)
{
  auto *capture = static_cast<StripCapture *>(opaque);
  if (!capture || capture->invocations++) {
    throw std::runtime_error("strip callback invocation contract");
  }
  capture->xs.resize(static_cast<std::size_t>(x_slabs) + 1);
  capture->intervals.resize(interval_count);
  capture->offsets.resize(x_slabs);
  capture->counts.resize(x_slabs);
  cuda_require(
      cudaMemcpyAsync(
          capture->xs.data(), device_xs,
          capture->xs.size() * sizeof(capture->xs.front()),
          cudaMemcpyDeviceToHost, stream),
      "capture xs");
  if (interval_count) {
    cuda_require(
        cudaMemcpyAsync(
            capture->intervals.data(), device_intervals,
            capture->intervals.size() *
                sizeof(capture->intervals.front()),
            cudaMemcpyDeviceToHost, stream),
        "capture intervals");
  }
  cuda_require(
      cudaMemcpyAsync(
          capture->offsets.data(), device_offsets,
          capture->offsets.size() *
              sizeof(capture->offsets.front()),
          cudaMemcpyDeviceToHost, stream),
      "capture offsets");
  cuda_require(
      cudaMemcpyAsync(
          capture->counts.data(), device_counts,
          capture->counts.size() * sizeof(capture->counts.front()),
          cudaMemcpyDeviceToHost, stream),
      "capture counts");
  cuda_require(
      cudaStreamSynchronize(stream), "capture synchronize");
}

void reject_strips(
    cudaStream_t, const std::int64_t *, std::uint32_t,
    const mu::StripInterval *, std::uint64_t,
    const std::uint64_t *, const std::uint32_t *, void *)
{
  throw std::runtime_error("intentional window consumer rejection");
}

mu::ResidentStripHook capture_hook(StripCapture *capture)
{
  mu::ResidentStripHook hook;
  hook.consume = &capture_strips;
  hook.context = capture;
  hook.stop_before_boundary = true;
  return hook;
}

std::pair<std::int64_t, std::int64_t> y_bounds(
    const std::vector<mu::RectI64> &rectangles)
{
  if (rectangles.empty()) return {0, 0};
  std::int64_t low = rectangles.front().bottom;
  std::int64_t high = rectangles.front().top;
  for (const auto &rectangle : rectangles) {
    low = std::min(low, rectangle.bottom);
    high = std::max(high, rectangle.top);
  }
  return {low, high};
}

void validate_capture(const StripCapture &capture)
{
  if (capture.invocations != 1 || capture.xs.size() < 2 ||
      capture.offsets.size() + 1 != capture.xs.size() ||
      capture.counts.size() != capture.offsets.size()) {
    throw std::runtime_error("malformed captured strip view");
  }
  for (std::size_t slab = 0; slab < capture.offsets.size(); ++slab) {
    if (capture.xs[slab] >= capture.xs[slab + 1]) {
      throw std::runtime_error("noncanonical captured x endpoints");
    }
    const std::uint64_t expected =
        slab ? capture.offsets[slab - 1] + capture.counts[slab - 1]
             : 0;
    if (capture.offsets[slab] != expected ||
        capture.offsets[slab] + capture.counts[slab] >
            capture.intervals.size()) {
      throw std::runtime_error("malformed captured slab range");
    }
    for (std::uint64_t local = 0; local < capture.counts[slab];
         ++local) {
      const std::uint64_t index = capture.offsets[slab] + local;
      const mu::StripInterval &interval =
          capture.intervals[index];
      if (interval.slab != slab || interval.reserved ||
          interval.bottom >= interval.top ||
          (local &&
           capture.intervals[index - 1].top >= interval.bottom)) {
        throw std::runtime_error(
            "noncanonical captured strip interval");
      }
    }
  }
  if (capture.offsets.back() + capture.counts.back() !=
      capture.intervals.size()) {
    throw std::runtime_error("captured interval census mismatch");
  }
}

bool same_interval(
    const mu::StripInterval &first,
    const mu::StripInterval &second)
{
  return first.bottom == second.bottom &&
         first.top == second.top && first.slab == second.slab &&
         first.reserved == second.reserved;
}

void require_same_capture(
    const std::string &name, const StripCapture &expected,
    const StripCapture &actual)
{
  validate_capture(expected);
  validate_capture(actual);
  if (expected.xs != actual.xs ||
      expected.offsets != actual.offsets ||
      expected.counts != actual.counts ||
      expected.intervals.size() != actual.intervals.size()) {
    throw std::runtime_error(name + ": strip census mismatch");
  }
  for (std::size_t index = 0; index < expected.intervals.size();
       ++index) {
    if (!same_interval(
            expected.intervals[index], actual.intervals[index])) {
      throw std::runtime_error(
          name + ": strip interval mismatch at " +
          std::to_string(index));
    }
  }
}

void run_differential(
    const std::string &name,
    const std::vector<mu::RectI64> &rectangles, int device,
    std::uint64_t max_window_events,
    std::uint32_t max_window_slabs)
{
  const auto bounds = y_bounds(rectangles);
  mu::GpuUnionLimits limits;
  limits.max_rectangles = 10000;
  limits.max_x_slabs = 10000;
  limits.max_memberships = 1000000;
  limits.max_events = 2000000;
  limits.max_raw_segments = 2000000;
  limits.max_segments = 2000000;
  limits.max_slabs_per_rectangle = 10000;

  StripCapture historical_capture;
  mu::ResidentStripHook historical_hook =
      capture_hook(&historical_capture);
  const mu::GpuUnionOutput historical = mu::gpu_union_resident(
      thrust::device_vector<mu::RectI64>(
          rectangles.begin(), rectangles.end()),
      bounds.first, bounds.second, limits, device, 1.25,
      &historical_hook);

  StripCapture windowed_capture;
  mu::ResidentStripHook windowed_hook =
      capture_hook(&windowed_capture);
  mu::GpuUnionStripWindowLimits window_limits;
  window_limits.max_window_events = max_window_events;
  window_limits.max_strip_intervals = 100000;
  window_limits.max_windows = 10000;
  window_limits.max_window_slabs = max_window_slabs;
  const mu::GpuUnionOutput windowed =
      mu::gpu_union_resident_windowed_strips(
          thrust::device_vector<mu::RectI64>(
              rectangles.begin(), rectangles.end()),
          bounds.first, bounds.second, limits, window_limits, device,
          1.25, &windowed_hook);

  if (historical.fallback || windowed.fallback ||
      !historical.resident_consumer_completed ||
      !windowed.resident_consumer_completed ||
      historical.rectangle_count != windowed.rectangle_count ||
      historical.x_slabs != windowed.x_slabs ||
      historical.memberships != windowed.memberships ||
      historical.event_count != windowed.event_count ||
      historical.strip_intervals != windowed.strip_intervals ||
      windowed.h2d_ms != 0.0 ||
      windowed.input_prepare_ms != 1.25 ||
      std::abs(
          windowed.charged_total_ms -
          (windowed.total_ms + 1.25)) > 1e-9) {
    throw std::runtime_error(
        name + ": union telemetry mismatch historical='" +
        historical.message + "' windowed='" + windowed.message + "'");
  }
  require_same_capture(
      name, historical_capture, windowed_capture);
}

std::vector<mu::RectI64> random_fixture(
    std::uint64_t seed, std::size_t count)
{
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<std::int64_t> coordinate(-80, 80);
  std::uniform_int_distribution<std::int64_t> extent(1, 30);
  std::vector<mu::RectI64> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::int64_t left = coordinate(generator);
    const std::int64_t bottom = coordinate(generator);
    result.push_back(
        rect(
            left, bottom, left + extent(generator),
            bottom + extent(generator)));
  }
  return result;
}

mu::GpuUnionOutput run_capacity_case(
    const std::vector<mu::RectI64> &rectangles,
    const mu::GpuUnionLimits &limits,
    const mu::GpuUnionStripWindowLimits &window_limits, int device,
    mu::ResidentStripHook *hook)
{
  const auto bounds = y_bounds(rectangles);
  return mu::gpu_union_resident_windowed_strips(
      thrust::device_vector<mu::RectI64>(
          rectangles.begin(), rectangles.end()),
      bounds.first, bounds.second, limits, window_limits, device, 0.0,
      hook);
}

void require_fallback(
    const std::string &name, const mu::GpuUnionOutput &output,
    const std::string &message, const StripCapture &capture)
{
  if (!output.fallback ||
      output.message.find(message) == std::string::npos ||
      output.resident_consumer_completed || capture.invocations) {
    throw std::runtime_error(
        name + ": fail-closed gate mismatch: " + output.message);
  }
}

void run_capacity_gates(int device)
{
  mu::GpuUnionLimits limits;
  limits.max_rectangles = 100;
  limits.max_x_slabs = 100;
  limits.max_memberships = 1000;
  limits.max_events = 2000;
  limits.max_raw_segments = 2000;
  limits.max_segments = 2000;
  limits.max_slabs_per_rectangle = 100;

  {
    const std::vector<mu::RectI64> rectangles = {
        rect(0, 0, 10, 10), rect(0, 3, 10, 7)};
    StripCapture capture;
    mu::ResidentStripHook hook = capture_hook(&capture);
    mu::GpuUnionStripWindowLimits windows;
    windows.max_window_events = 2;
    windows.max_strip_intervals = 10;
    const auto output =
        run_capacity_case(rectangles, limits, windows, device, &hook);
    require_fallback(
        "single slab event cap", output,
        "single-slab window event capacity", capture);
  }

  {
    const std::vector<mu::RectI64> rectangles = {
        rect(0, 0, 10, 2), rect(0, 4, 10, 6)};
    StripCapture capture;
    mu::ResidentStripHook hook = capture_hook(&capture);
    mu::GpuUnionStripWindowLimits windows;
    windows.max_window_events = 16;
    windows.max_strip_intervals = 1;
    const auto output =
        run_capacity_case(rectangles, limits, windows, device, &hook);
    require_fallback(
        "strip interval cap", output, "strip interval capacity",
        capture);
  }

  {
    const std::vector<mu::RectI64> rectangles = {
        rect(0, 0, 3, 1), rect(1, 2, 2, 3)};
    StripCapture capture;
    mu::ResidentStripHook hook = capture_hook(&capture);
    mu::GpuUnionStripWindowLimits windows;
    windows.max_window_events = 100;
    windows.max_strip_intervals = 20;
    windows.max_window_slabs = 1;
    windows.max_windows = 2;
    const auto output =
        run_capacity_case(rectangles, limits, windows, device, &hook);
    require_fallback(
        "window count cap", output, "window count capacity", capture);
  }

  {
    const std::vector<mu::RectI64> rectangles = {
        rect(0, 0, 4, 1), rect(1, 2, 2, 3), rect(3, 2, 4, 3)};
    StripCapture capture;
    mu::ResidentStripHook hook = capture_hook(&capture);
    mu::GpuUnionStripWindowLimits windows;
    windows.max_window_events = 100;
    windows.max_strip_intervals = 20;
    mu::GpuUnionLimits narrow = limits;
    narrow.max_slabs_per_rectangle = 2;
    const auto output =
        run_capacity_case(rectangles, narrow, windows, device, &hook);
    require_fallback(
        "rectangle span cap", output,
        "per-rectangle slab capacity", capture);
  }

  {
    const std::vector<mu::RectI64> rectangles = {
        rect(0, 0, 4, 1), rect(1, 2, 2, 3)};
    StripCapture capture;
    mu::ResidentStripHook hook = capture_hook(&capture);
    hook.stop_before_boundary = false;
    mu::GpuUnionStripWindowLimits windows;
    windows.max_window_events = 100;
    windows.max_strip_intervals = 20;
    const auto output =
        run_capacity_case(rectangles, limits, windows, device, &hook);
    require_fallback(
        "terminal hook contract", output, "terminal resident hook",
        capture);
  }

  {
    const std::vector<mu::RectI64> rectangles = {
        rect(0, 0, 4, 1), rect(1, 2, 2, 3)};
    StripCapture capture;
    mu::ResidentStripHook hook = capture_hook(&capture);
    mu::GpuUnionStripWindowLimits windows;
    windows.max_window_events = 100;
    windows.max_strip_intervals =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) +
        1;
    const auto output =
        run_capacity_case(rectangles, limits, windows, device, &hook);
    require_fallback(
        "uint32 interval cap", output, "invalid strip window limits",
        capture);
  }

  {
    const std::vector<mu::RectI64> rectangles = {
        rect(0, 0, 4, 1), rect(1, 2, 2, 3)};
    mu::ResidentStripHook hook;
    hook.consume = &reject_strips;
    hook.stop_before_boundary = true;
    mu::GpuUnionStripWindowLimits windows;
    windows.max_window_events = 100;
    windows.max_strip_intervals = 20;
    const auto output =
        run_capacity_case(rectangles, limits, windows, device, &hook);
    StripCapture never;
    if (!output.fallback ||
        output.message.find("intentional window consumer rejection") ==
            std::string::npos ||
        output.resident_consumer_completed) {
      throw std::runtime_error(
          "consumer rejection did not fail closed: " +
          output.message);
    }
  }
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    int device = 0;
    if (argc == 3 && std::string(argv[1]) == "--device") {
      device = std::stoi(argv[2]);
    } else if (argc != 1) {
      throw std::runtime_error(
          "usage: manhattan_union_windowed_strips_test [--device N]");
    }
    cuda_require(cudaSetDevice(device), "cudaSetDevice");

    run_differential(
        "long-seam-crossing",
        {
            rect(0, 0, 12, 4), rect(2, 6, 10, 9),
            rect(4, 3, 8, 7), rect(6, -3, 7, 12),
            rect(1, 4, 3, 6), rect(9, 4, 11, 6),
        },
        device, 12, std::numeric_limits<std::uint32_t>::max());
    run_differential(
        "one-slab-windows-with-gaps",
        {
            rect(-20, 0, -10, 10), rect(-19, 3, -16, 7),
            rect(0, -5, 30, 5), rect(5, 8, 10, 12),
            rect(20, -8, 25, 9), rect(40, 1, 50, 2),
        },
        device, 1000, 1);
    run_differential(
        "nested-and-touching",
        {
            rect(0, 0, 30, 30), rect(5, 5, 25, 25),
            rect(30, 0, 40, 10), rect(30, 20, 40, 30),
            rect(10, 30, 20, 40), rect(40, 30, 50, 40),
        },
        device, 1000, 2);
    for (std::uint64_t seed = 1; seed <= 24; ++seed) {
      run_differential(
          "random-" + std::to_string(seed),
          random_fixture(seed, 80), device, 100000, 3);
    }
    run_capacity_gates(device);

    std::cout
        << "MANHATTAN_UNION_WINDOWED_STRIPS_TEST ok directed=3 "
        << "random=24 capacity=7\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "MANHATTAN_UNION_WINDOWED_STRIPS_TEST failed: "
        << error.what() << "\n";
    return 1;
  }
}
