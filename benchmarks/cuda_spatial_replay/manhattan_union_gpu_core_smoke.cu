/*
 * Link-level smoke gate for the shared exact CUDA Manhattan-union core.
 */

#include "manhattan_union_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <cmath>
#include <cstdint>
#include <iostream>
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

bool same_segment(
    const mu::DirectedSegmentI64 &first,
    const mu::DirectedSegmentI64 &second)
{
  return first.fixed == second.fixed && first.lo == second.lo &&
         first.hi == second.hi && first.side == second.side &&
         first.axis == second.axis;
}

void require_same_output(
    const mu::GpuUnionOutput &expected,
    const mu::GpuUnionOutput &actual)
{
  if (expected.fallback || actual.fallback ||
      expected.digest != actual.digest ||
      expected.segments.size() != actual.segments.size()) {
    throw std::runtime_error("host/resident union result mismatch");
  }
  for (std::size_t index = 0; index < expected.segments.size(); ++index) {
    if (!same_segment(expected.segments[index], actual.segments[index])) {
      throw std::runtime_error(
          "host/resident segment mismatch at " +
          std::to_string(index));
    }
  }
}

struct StripCapture
{
  bool invoked = false;
  bool valid = false;
};

struct BoundaryCapture
{
  bool invoked = false;
  std::vector<mu::DirectedSegmentI64> segments;
};

void capture_strips(
    cudaStream_t stream, const std::int64_t *device_xs,
    std::uint32_t x_slabs,
    const mu::StripInterval *device_intervals,
    std::uint64_t interval_count,
    const std::uint64_t *device_slab_offsets,
    const std::uint32_t *device_slab_counts, void *opaque)
{
  (void)stream;
  auto *capture = static_cast<StripCapture *>(opaque);
  capture->invoked = true;
  std::vector<std::int64_t> xs(x_slabs + 1);
  std::vector<mu::StripInterval> intervals(interval_count);
  std::vector<std::uint64_t> offsets(x_slabs);
  std::vector<std::uint32_t> counts(x_slabs);
  cuda_require(
      cudaMemcpy(
          xs.data(), device_xs, xs.size() * sizeof(xs.front()),
          cudaMemcpyDeviceToHost),
      "strip xs D2H");
  cuda_require(
      cudaMemcpy(
          intervals.data(), device_intervals,
          intervals.size() * sizeof(intervals.front()),
          cudaMemcpyDeviceToHost),
      "strip intervals D2H");
  cuda_require(
      cudaMemcpy(
          offsets.data(), device_slab_offsets,
          offsets.size() * sizeof(offsets.front()),
          cudaMemcpyDeviceToHost),
      "strip offsets D2H");
  cuda_require(
      cudaMemcpy(
          counts.data(), device_slab_counts,
          counts.size() * sizeof(counts.front()),
          cudaMemcpyDeviceToHost),
      "strip counts D2H");

  capture->valid =
      x_slabs == 2 && interval_count == 2 &&
      xs == std::vector<std::int64_t>({0, 4, 8}) &&
      offsets == std::vector<std::uint64_t>({0, 1}) &&
      counts == std::vector<std::uint32_t>({1, 1}) &&
      intervals[0].bottom == 0 && intervals[0].top == 3 &&
      intervals[0].slab == 0 && intervals[0].reserved == 0 &&
      intervals[1].bottom == 0 && intervals[1].top == 3 &&
      intervals[1].slab == 1 && intervals[1].reserved == 0;
}

void reject_strips(
    cudaStream_t, const std::int64_t *, std::uint32_t,
    const mu::StripInterval *, std::uint64_t, const std::uint64_t *,
    const std::uint32_t *, void *)
{
  throw std::runtime_error("intentional resident consumer rejection");
}

void capture_boundary(
    cudaStream_t, const mu::DirectedSegmentI64 *device_horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *device_vertical,
    std::uint64_t vertical_count, void *opaque)
{
  auto *capture = static_cast<BoundaryCapture *>(opaque);
  capture->invoked = true;
  capture->segments.resize(horizontal_count + vertical_count);
  if (horizontal_count) {
    cuda_require(
        cudaMemcpy(
            capture->segments.data(), device_horizontal,
            horizontal_count * sizeof(mu::DirectedSegmentI64),
            cudaMemcpyDeviceToHost),
        "resident horizontal boundary D2H");
  }
  if (vertical_count) {
    cuda_require(
        cudaMemcpy(
            capture->segments.data() + horizontal_count, device_vertical,
            vertical_count * sizeof(mu::DirectedSegmentI64),
            cudaMemcpyDeviceToHost),
        "resident vertical boundary D2H");
  }
}

void reject_boundary(
    cudaStream_t, const mu::DirectedSegmentI64 *, std::uint64_t,
    const mu::DirectedSegmentI64 *, std::uint64_t, void *)
{
  throw std::runtime_error(
      "intentional resident boundary consumer rejection");
}

thrust::device_vector<mu::RectI64> resident_copy(
    const std::vector<mu::RectI64> &rectangles)
{
  return thrust::device_vector<mu::RectI64>(
      rectangles.begin(), rectangles.end());
}

void require_boundary_gate_rejects(
    const std::vector<mu::DirectedSegmentI64> &horizontal,
    const std::vector<mu::DirectedSegmentI64> &vertical,
    int device, const std::string &name)
{
  std::string error;
  if (mu::gpu_validate_resident_boundary_for_test(
          horizontal, vertical, device, &error) ||
      error.find("canonical boundary invariant") ==
          std::string::npos) {
    throw std::runtime_error(
        name + " resident boundary corruption was accepted: " +
        error);
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
          "usage: manhattan_union_gpu_core_smoke [--device N]");
    }
    cuda_require(cudaSetDevice(device), "cudaSetDevice");

    const std::vector<mu::RectI64> rectangles = {
        {0, 0, 4, 3, 1, 0}, {4, 0, 8, 3, 2, 0}};
    const mu::GpuUnionLimits limits;
    const mu::GpuUnionOutput host =
        mu::gpu_union_host(rectangles, limits, device);
    const std::vector<mu::DirectedSegmentI64> expected = {
        {0, 0, 8, -1, mu::SegmentAxis::horizontal},
        {3, 0, 8, 1, mu::SegmentAxis::horizontal},
        {0, 0, 3, -1, mu::SegmentAxis::vertical},
        {8, 0, 3, 1, mu::SegmentAxis::vertical}};
    if (host.fallback || host.resident_consumer_completed ||
        host.resident_boundary_consumer_completed ||
        host.rectangle_count != 2 ||
        host.memberships != 2 || host.event_count != 4 ||
        host.x_slabs != 2 || host.strip_intervals != 2 ||
        host.raw_segments != 6 ||
        host.segments.size() != expected.size() ||
        host.input_prepare_ms != 0.0 ||
        host.charged_total_ms != host.total_ms) {
      throw std::runtime_error("host union fixture failed");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (!same_segment(expected[index], host.segments[index])) {
        throw std::runtime_error(
            "unexpected canonical fixture boundary at " +
            std::to_string(index));
      }
    }

    const std::vector<mu::DirectedSegmentI64> valid_horizontal(
        expected.begin(), expected.begin() + 2);
    const std::vector<mu::DirectedSegmentI64> valid_vertical(
        expected.begin() + 2, expected.end());
    std::string boundary_gate_error = "not cleared";
    if (!mu::gpu_validate_resident_boundary_for_test(
            valid_horizontal, valid_vertical, device,
            &boundary_gate_error) ||
        !boundary_gate_error.empty()) {
      throw std::runtime_error(
          "valid resident boundary gate declined: " +
          boundary_gate_error);
    }
    {
      auto corrupted = valid_horizontal;
      corrupted[0].axis = mu::SegmentAxis::vertical;
      require_boundary_gate_rejects(
          corrupted, valid_vertical, device, "wrong-axis");
    }
    {
      auto corrupted = valid_horizontal;
      corrupted[0].side = 0;
      require_boundary_gate_rejects(
          corrupted, valid_vertical, device, "bad-side");
    }
    {
      auto corrupted = valid_horizontal;
      corrupted[0].hi = corrupted[0].lo;
      require_boundary_gate_rejects(
          corrupted, valid_vertical, device, "empty-segment");
    }
    {
      auto corrupted = valid_horizontal;
      std::swap(corrupted[0], corrupted[1]);
      require_boundary_gate_rejects(
          corrupted, valid_vertical, device, "unsorted");
    }
    {
      const std::vector<mu::DirectedSegmentI64> touching = {
          {0, 0, 4, -1, mu::SegmentAxis::horizontal},
          {0, 4, 8, -1, mu::SegmentAxis::horizontal}};
      require_boundary_gate_rejects(
          touching, valid_vertical, device, "touching");
    }
    {
      const std::vector<mu::DirectedSegmentI64> overlapping = {
          {0, 0, 5, -1, mu::SegmentAxis::horizontal},
          {0, 4, 8, -1, mu::SegmentAxis::horizontal}};
      require_boundary_gate_rejects(
          overlapping, valid_vertical, device, "overlapping");
    }

    StripCapture capture;
    mu::ResidentStripHook hook;
    hook.consume = capture_strips;
    hook.context = &capture;
    const mu::GpuUnionOutput resident = mu::gpu_union_resident(
        resident_copy(rectangles), 0, 3, limits, device, 7.5, &hook);
    require_same_output(host, resident);
    if (!capture.invoked || !capture.valid ||
        !resident.resident_consumer_completed ||
        resident.input_prepare_ms != 7.5 ||
        resident.h2d_ms != 0.0 ||
        std::abs(
            resident.charged_total_ms -
            (resident.total_ms + 7.5)) > 1e-9) {
      throw std::runtime_error("resident strip hook mismatch");
    }

    StripCapture stopped_capture;
    hook.context = &stopped_capture;
    hook.stop_before_boundary = true;
    const mu::GpuUnionOutput stopped = mu::gpu_union_resident(
        resident_copy(rectangles), 0, 3, limits, device, 0.0, &hook);
    if (stopped.fallback || !stopped.resident_consumer_completed ||
        !stopped.segments.empty() ||
        !stopped_capture.invoked || !stopped_capture.valid) {
      throw std::runtime_error("resident strip early-stop mismatch");
    }

    hook.consume = reject_strips;
    hook.context = nullptr;
    const mu::GpuUnionOutput rejected = mu::gpu_union_resident(
        resident_copy(rectangles), 0, 3, limits, device, 0.0, &hook);
    if (!rejected.fallback || rejected.resident_consumer_completed ||
        !rejected.segments.empty() ||
        rejected.message.find("intentional resident consumer rejection") ==
            std::string::npos) {
      throw std::runtime_error(
          "resident consumer rejection did not fail closed");
    }

    BoundaryCapture boundary_capture;
    mu::ResidentBoundaryHook boundary_hook;
    boundary_hook.consume = capture_boundary;
    boundary_hook.context = &boundary_capture;
    const mu::GpuUnionOutput boundary_resident = mu::gpu_union_resident(
        resident_copy(rectangles), 0, 3, limits, device, 0.0, nullptr,
        &boundary_hook);
    require_same_output(host, boundary_resident);
    if (!boundary_capture.invoked ||
        !boundary_resident.resident_boundary_consumer_completed ||
        boundary_capture.segments.size() != expected.size()) {
      throw std::runtime_error("resident boundary hook mismatch");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (!same_segment(expected[index], boundary_capture.segments[index])) {
        throw std::runtime_error(
            "resident boundary hook segment mismatch at " +
            std::to_string(index));
      }
    }

    BoundaryCapture stopped_boundary_capture;
    boundary_hook.context = &stopped_boundary_capture;
    boundary_hook.stop_before_d2h = true;
    const mu::GpuUnionOutput stopped_boundary =
        mu::gpu_union_resident(
            resident_copy(rectangles), 0, 3, limits, device, 0.0,
            nullptr, &boundary_hook);
    if (stopped_boundary.fallback ||
        !stopped_boundary.resident_boundary_consumer_completed ||
        !stopped_boundary.segments.empty() ||
        !stopped_boundary_capture.invoked ||
        stopped_boundary_capture.segments.size() != expected.size() ||
        stopped_boundary.d2h_ms != 0.0) {
      throw std::runtime_error(
          "resident boundary early-stop mismatch");
    }

    boundary_hook.consume = reject_boundary;
    boundary_hook.context = nullptr;
    const mu::GpuUnionOutput rejected_boundary =
        mu::gpu_union_resident(
            resident_copy(rectangles), 0, 3, limits, device, 0.0,
            nullptr, &boundary_hook);
    if (!rejected_boundary.fallback ||
        rejected_boundary.resident_boundary_consumer_completed ||
        !rejected_boundary.segments.empty() ||
        rejected_boundary.message.find(
            "intentional resident boundary consumer rejection") ==
            std::string::npos) {
      throw std::runtime_error(
          "resident boundary rejection did not fail closed");
    }

    const mu::GpuUnionOutput bad_bounds = mu::gpu_union_resident(
        resident_copy(rectangles), 1, 3, limits, device);
    if (!bad_bounds.fallback || !bad_bounds.segments.empty()) {
      throw std::runtime_error("resident bounds did not fail closed");
    }

    mu::GpuUnionLimits separated_caps = limits;
    separated_caps.max_segments = 4;
    separated_caps.max_raw_segments = 6;
    const mu::GpuUnionOutput exact_caps =
        mu::gpu_union_host(rectangles, separated_caps, device);
    require_same_output(host, exact_caps);
    separated_caps.max_raw_segments = 5;
    const mu::GpuUnionOutput raw_cap =
        mu::gpu_union_host(rectangles, separated_caps, device);
    if (!raw_cap.fallback || !raw_cap.segments.empty() ||
        raw_cap.message != "raw segment capacity") {
      throw std::runtime_error(
          "raw/canonical segment caps were not independent");
    }

    std::cout
        << "MANHATTAN_UNION_GPU_CORE_SMOKE PASS host=1 resident=1 "
           "strip_hook=1 boundary_hook=1 early_stop=1 "
           "boundary_device_gate=7 "
           "bounds_fallback=1 consumer_rejection_fallback=1 "
           "boundary_rejection_fallback=1 separate_raw_final_caps=1 "
           "segments="
        << host.segments.size() << " digest=" << host.digest << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MANHATTAN_UNION_GPU_CORE_SMOKE FAIL reason='"
              << error.what() << "'\n";
    return 1;
  }
}
