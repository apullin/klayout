/*
 * Shared exact CUDA Manhattan-union engine.
 *
 * This header is CUDA-only.  KLayout's optional DSO C ABI remains isolated in
 * dbCudaSpatialApi.h; the future M2 adapter owns the Thrust vector on its side
 * of that boundary and moves it into the resident entry point below.
 */

#ifndef KLAYOUT_CUDA_MANHATTAN_UNION_GPU_CUH
#define KLAYOUT_CUDA_MANHATTAN_UNION_GPU_CUH

#include "manhattan_union_format.cuh"

#include <cuda_runtime_api.h>

#include <thrust/device_vector.h>

#include <cstdint>
#include <string>
#include <vector>

namespace klayout_cuda {
namespace manhattan_union {

// Canonical y interval for one x slab.  These records, together with the
// resident x endpoints and per-slab ranges exposed by ResidentStripHook,
// describe the exact union before boundary materialization.
struct StripInterval
{
  std::int64_t bottom;
  std::int64_t top;
  std::uint32_t slab;
  std::uint32_t reserved;
};

static_assert(sizeof(StripInterval) == 24,
              "unexpected strip-interval padding");

struct ResidentStripHook
{
  using Consume = void (*)(
      cudaStream_t stream, const std::int64_t *xs,
      std::uint32_t x_slabs,
      const StripInterval *intervals, std::uint64_t interval_count,
      const std::uint64_t *slab_offsets,
      const std::uint32_t *slab_counts, void *context);

  // consume is invoked synchronously on the selected device after the strip
  // status has passed.  All input is ordered on stream; the callback must
  // launch any work on that stream and must not retain any pointer after it
  // returns.  The core synchronizes stream before reclaiming storage.
  Consume consume = nullptr;
  void *context = nullptr;
  bool stop_before_boundary = false;
};

/*
 * Synchronous device-resident boundary consumer.
 *
 * The union core invokes this hook only after the exact horizontal and
 * vertical boundary streams have passed every device invariant.  The two
 * streams are independently canonical and together form the complete
 * directed boundary.  Their storage remains owned by the union core and is
 * valid only for the duration of consume().
 *
 * At this point the high-water event/coverage/strip storage has already been
 * released.  A downstream spatial certificate can therefore allocate its
 * own index without overlapping the union sweep's peak memory.  Setting
 * stop_before_d2h skips host boundary materialization after a successful
 * callback.
 */
struct ResidentBoundaryHook
{
  using Consume = void (*)(
      cudaStream_t stream,
      const DirectedSegmentI64 *horizontal,
      std::uint64_t horizontal_count,
      const DirectedSegmentI64 *vertical,
      std::uint64_t vertical_count, void *context);

  Consume consume = nullptr;
  void *context = nullptr;
  bool stop_before_d2h = false;
};

struct GpuUnionLimits
{
  std::uint64_t max_rectangles = UINT64_C(32000000);
  std::uint64_t max_x_slabs = UINT64_C(32000000);
  std::uint64_t max_memberships = UINT64_C(64000000);
  std::uint64_t max_events = UINT64_C(128000000);
  // Raw strip transitions plus vertical XOR fragments are a distinct,
  // generally larger census than the canonical boundary below.
  std::uint64_t max_raw_segments = UINT64_MAX;
  std::uint64_t max_segments = UINT64_C(64000000);
  std::uint32_t max_slabs_per_rectangle = 4096;
};

/*
 * Optional bounded high-water configuration for the resident strip-only
 * producer.
 *
 * This is deliberately separate from GpuUnionLimits: max_memberships and
 * max_events remain whole-input census gates, while max_window_events bounds
 * only the event storage live during one x-slab window.  The producer keeps
 * one exact global strip array because the existing ResidentStripHook is a
 * one-shot whole-source contract.  max_strip_intervals is both its allocation
 * bound and its fail-closed capacity and may not exceed UINT32_MAX.  Before
 * the callback, the producer releases rectangle staging and compacts that
 * allocation to the exact interval census; the explicitly sampled compaction
 * high-water is bounded by twice max_strip_intervals.
 *
 * Windows partition canonical global x slabs.  Union coverage inside a slab
 * is independent of adjacent slabs, so no geometry halo is required.  A
 * downstream operation that consumes windows directly may require a
 * coordinate halo; this API intentionally does not expose partial windows and
 * invokes the existing hook only after all global strips have been stitched.
 */
struct GpuUnionStripWindowLimits
{
  std::uint64_t max_window_events = UINT64_C(64000000);
  std::uint64_t max_strip_intervals = UINT64_C(64000000);
  std::uint64_t max_windows = UINT64_C(4096);
  std::uint32_t max_window_slabs = UINT32_MAX;
};

struct GpuUnionOutput
{
  bool fallback = false;
  bool resident_consumer_completed = false;
  bool resident_boundary_consumer_completed = false;
  std::string message;
  std::vector<DirectedSegmentI64> segments;
  std::uint64_t rectangle_count = 0;
  std::uint64_t memberships = 0;
  std::uint64_t event_count = 0;
  std::uint64_t x_slabs = 0;
  std::uint64_t strip_intervals = 0;
  std::uint64_t raw_segments = 0;
  std::uint64_t digest = 0;
  double total_ms = 0.0;
  // Only upstream work explicitly supplied to gpu_union_resident is excluded
  // from total_ms.  charged_total_ms is total_ms + input_prepare_ms.
  double input_prepare_ms = 0.0;
  double charged_total_ms = 0.0;
  double h2d_ms = 0.0;
  double x_membership_ms = 0.0;
  double strip_scan_ms = 0.0;
  double boundary_ms = 0.0;
  double d2h_ms = 0.0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_begin_bytes = 0;
  std::uint64_t device_free_low_bytes = 0;
};

// Existing replay behavior: validates a host vector, copies it to the selected
// device, and charges that copy in h2d_ms and total_ms.
GpuUnionOutput gpu_union_host(
    const std::vector<RectI64> &rectangles,
    const GpuUnionLimits &limits, int device,
    const ResidentStripHook *resident_hook = nullptr,
    const ResidentBoundaryHook *boundary_hook = nullptr);

// Resident seam for a checked upstream device expander.  Ownership of
// rectangles is transferred to the union so it can release the 48-byte input
// records at the same peak-memory point as the historical host-vector path.
// y_base/y_high must bound every rectangle and fit in one uint32 packed range.
// input_prepare_ms reports already-charged compact upload/expansion work;
// total_ms starts at this call, while charged_total_ms includes both.
// h2d_ms remains zero because no host-to-device copy occurs in this call.
GpuUnionOutput gpu_union_resident(
    thrust::device_vector<RectI64> &&rectangles,
    std::int64_t y_base, std::int64_t y_high,
    const GpuUnionLimits &limits, int device,
    double input_prepare_ms = 0.0,
    const ResidentStripHook *resident_hook = nullptr,
    const ResidentBoundaryHook *boundary_hook = nullptr);

/*
 * Exact bounded-event resident strip producer.
 *
 * This path is opt-in and terminal: resident_hook must be non-null and set
 * stop_before_boundary.  It constructs the same canonical global x-slab
 * intervals as gpu_union_resident, but partitions event generation/sort/scan
 * by a histogram-planned sequence of slab windows.  The callback is still
 * invoked exactly once with the whole source, so existing whole-source
 * morphology consumers retain their semantics.  Boundary materialization is
 * intentionally unsupported here.
 */
GpuUnionOutput gpu_union_resident_windowed_strips(
    thrust::device_vector<RectI64> &&rectangles,
    std::int64_t y_base, std::int64_t y_high,
    const GpuUnionLimits &limits,
    const GpuUnionStripWindowLimits &window_limits, int device,
    double input_prepare_ms, const ResidentStripHook *resident_hook);

// Retained as a direct unit-test seam for the parallel canonicalizer.
std::vector<DirectedSegmentI64> gpu_canonicalize_segments_for_test(
    const std::vector<DirectedSegmentI64> &raw, int device);

// Direct qualification seam for the exact device gate that protects a
// ResidentBoundaryHook.  Production invokes the same gate immediately before
// exposing either device pointer.
bool gpu_validate_resident_boundary_for_test(
    const std::vector<DirectedSegmentI64> &horizontal,
    const std::vector<DirectedSegmentI64> &vertical,
    int device, std::string *error = nullptr);

// Exact serial references retained only for differential qualification of the
// shared core; production adapters must use the two GPU entry points above.
GpuUnionOutput cpu_union_reference_for_test(
    const std::vector<RectI64> &rectangles,
    const GpuUnionLimits &limits);

void canonicalize_segments_reference_for_test(
    std::vector<DirectedSegmentI64> *segments);

}  // namespace manhattan_union
}  // namespace klayout_cuda

#endif
