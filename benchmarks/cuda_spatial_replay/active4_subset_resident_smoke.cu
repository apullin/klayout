#include "active4_subset_resident.cuh"
#include "manhattan_union_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace a4 = klayout_cuda::active4_subset_resident;
namespace mu = klayout_cuda::manhattan_union;

struct Capture
{
  const mu::RectI64 *active = nullptr;
  std::uint64_t active_count = 0;
  a4::Limits limits;
  a4::Result result;
  bool invoked = false;
};

void consume(
    cudaStream_t stream, const std::int64_t *xs,
    std::uint32_t x_slabs, const mu::StripInterval *intervals,
    std::uint64_t interval_count, const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque)
{
  Capture *capture = static_cast<Capture *>(opaque);
  if (!capture || capture->invoked) {
    throw std::runtime_error("ACTIVE.4 smoke callback contract");
  }
  capture->invoked = true;
  capture->result = a4::certify_subset(
      stream,
      a4::DeviceStripView{
          xs, x_slabs, intervals, interval_count,
          slab_offsets, slab_counts},
      a4::DeviceRectangleView{capture->active, capture->active_count},
      capture->limits, 0);
}

mu::RectI64 rectangle(
    std::int64_t left, std::int64_t bottom,
    std::int64_t right, std::int64_t top)
{
  return {left, bottom, right, top, 0, 0};
}

a4::Result run(
    const std::vector<mu::RectI64> &wells,
    const std::vector<mu::RectI64> &active,
    const a4::Limits &limits = a4::Limits{})
{
  thrust::device_vector<mu::RectI64> device_active(
      active.begin(), active.end());
  Capture capture;
  capture.active = active.empty()
      ? nullptr
      : thrust::raw_pointer_cast(device_active.data());
  capture.active_count = active.size();
  capture.limits = limits;
  mu::ResidentStripHook hook;
  hook.consume = &consume;
  hook.context = &capture;
  hook.stop_before_boundary = true;
  mu::GpuUnionLimits union_limits;
  union_limits.max_rectangles = 1024;
  union_limits.max_x_slabs = 1024;
  union_limits.max_memberships = 4096;
  union_limits.max_events = 8192;
  union_limits.max_raw_segments = 8192;
  union_limits.max_segments = 8192;
  union_limits.max_slabs_per_rectangle = 1024;
  const mu::GpuUnionOutput output =
      mu::gpu_union_host(wells, union_limits, 0, &hook);
  if (output.fallback || !output.resident_consumer_completed ||
      output.resident_boundary_consumer_completed ||
      !output.segments.empty() || !capture.invoked) {
    throw std::runtime_error(
        "ACTIVE.4 smoke union/callback did not complete: " +
        output.message);
  }
  return capture.result;
}

void require_subset(
    const std::vector<mu::RectI64> &wells,
    const std::vector<mu::RectI64> &active,
    const std::string &name)
{
  const a4::Result result = run(wells, active);
  if (!result.certified_subset || result.witnesses ||
      result.uncertain || result.device_flags ||
      !result.all_work_completed ||
      result.rectangles_visited != active.size() ||
      result.rectangles_completed != active.size()) {
    throw std::runtime_error(name + " was not certified");
  }
}

void require_witness(
    const std::vector<mu::RectI64> &wells,
    const std::vector<mu::RectI64> &active,
    const std::string &name)
{
  const a4::Result result = run(wells, active);
  if (result.certified_subset || !result.witnesses ||
      result.uncertain || result.device_flags ||
      !result.all_work_completed ||
      result.rectangles_visited != active.size() ||
      result.rectangles_completed != active.size()) {
    throw std::runtime_error(name + " did not produce an exact witness");
  }
}

}  // namespace

int main()
{
  try {
    const std::vector<mu::RectI64> box = {
        rectangle(0, 0, 100, 100)};
    require_subset(
        box, {rectangle(10, 10, 90, 90)}, "strictly-inside");
    require_subset(
        box, {rectangle(0, 0, 100, 100)}, "boundary-touch");
    require_subset(
        box,
        {rectangle(0, 0, 50, 100), rectangle(40, 10, 100, 90)},
        "overlapping-active");
    require_witness(
        box, {rectangle(200, 200, 220, 220)}, "far-outside");
    require_witness(
        box, {rectangle(90, 10, 110, 30)}, "partial-outside");

    const std::vector<mu::RectI64> ring = {
        rectangle(0, 0, 100, 20), rectangle(0, 80, 100, 100),
        rectangle(0, 20, 20, 80), rectangle(80, 20, 100, 80)};
    require_witness(
        ring, {rectangle(30, 30, 70, 70)}, "well-hole");
    require_subset(
        ring, {rectangle(5, 5, 95, 15)}, "ring-material");

    const std::vector<mu::RectI64> stepped = {
        rectangle(-100, -20, 0, 40), rectangle(0, -20, 100, 80)};
    require_subset(
        stepped, {rectangle(-75, -10, 75, 30)},
        "multi-slab-transform");
    require_witness(
        stepped, {rectangle(-75, 30, 75, 50)},
        "multi-slab-y-gap");

    a4::Limits tiny;
    tiny.max_slabs_per_rectangle = 1;
    const a4::Result capacity = run(
        stepped, {rectangle(-75, -10, 75, 30)}, tiny);
    if (capacity.certified_subset || !capacity.uncertain ||
        !(capacity.device_flags & a4::kVisitCapacity) ||
        !(capacity.device_flags & a4::kTruncatedWork) ||
        capacity.all_work_completed ||
        capacity.rectangles_visited != 1 ||
        capacity.rectangles_completed != 0) {
      throw std::runtime_error("capacity did not fail closed");
    }

    a4::Limits tiny_search;
    tiny_search.max_search_steps = 1;
    const a4::Result search_capacity =
        run(box, {rectangle(10, 10, 90, 90)}, tiny_search);
    if (search_capacity.certified_subset ||
        search_capacity.uncertain ||
        !(search_capacity.device_flags & a4::kSearchCapacity) ||
        (search_capacity.device_flags & a4::kTruncatedWork) ||
        !search_capacity.all_work_completed ||
        search_capacity.rectangles_visited != 1 ||
        search_capacity.rectangles_completed != 1) {
      throw std::runtime_error(
          "post-completion search capacity did not fail closed");
    }

    const a4::Result malformed =
        run(box, {rectangle(10, 10, 10, 20)});
    if (malformed.certified_subset || !malformed.uncertain ||
        !(malformed.device_flags & a4::kInvalidRectangle) ||
        malformed.all_work_completed ||
        malformed.rectangles_visited != 1 ||
        malformed.rectangles_completed != 0) {
      throw std::runtime_error("malformed rectangle did not fail closed");
    }

    std::cout
        << "ACTIVE4_SUBSET_RESIDENT_SMOKE ok"
        << " cases=12 exact=1 all_batches_completed=1"
        << " post_completion_cap=1 truncated_distinct=1\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ACTIVE4_SUBSET_RESIDENT_SMOKE failed: "
              << error.what() << "\n";
    return 1;
  }
}
