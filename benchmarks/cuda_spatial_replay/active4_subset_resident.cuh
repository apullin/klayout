/*
 * Exact bounded ACTIVE-subset-of-WELL certificate over resident Manhattan
 * geometry.
 *
 * WELL is supplied as the canonical x-slab/y-interval representation emitted
 * by manhattan_union_gpu.  ACTIVE is supplied as an exact rectangulation in
 * the same root coordinate frame.  Calls are synchronous and fail closed:
 * only a fully validated, all-rectangles-completed result with zero witnesses,
 * uncertainty, and device flags is a consumable subset certificate.
 */

#ifndef KLAYOUT_CUDA_ACTIVE4_SUBSET_RESIDENT_CUH
#define KLAYOUT_CUDA_ACTIVE4_SUBSET_RESIDENT_CUH

#include "manhattan_union_gpu.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace klayout_cuda {
namespace active4_subset_resident {

struct DeviceStripView
{
  const std::int64_t *xs = nullptr;
  std::uint32_t x_slabs = 0;
  const manhattan_union::StripInterval *intervals = nullptr;
  std::uint64_t interval_count = 0;
  const std::uint64_t *slab_offsets = nullptr;
  const std::uint32_t *slab_counts = nullptr;
};

struct DeviceRectangleView
{
  const manhattan_union::RectI64 *rectangles = nullptr;
  std::uint64_t count = 0;
};

struct Limits
{
  std::uint64_t max_rectangles = UINT64_C(64000000);
  std::uint64_t max_x_slabs = UINT64_C(32000000);
  std::uint64_t max_intervals = UINT64_C(64000000);
  std::uint64_t max_slab_visits = UINT64_C(1000000000);
  std::uint64_t max_search_steps = UINT64_C(2000000000);
  std::uint32_t max_slabs_per_rectangle = 16384;
};

enum DeviceFlag : std::uint32_t
{
  kInvalidStripView = 1u << 0,
  kInvalidRectangle = 1u << 1,
  kVisitOverflow = 1u << 2,
  kVisitCapacity = 1u << 3,
  kSearchCapacity = 1u << 4,
  kTruncatedWork = 1u << 5
};

struct Result
{
  bool certified_subset = false;
  bool all_work_completed = false;
  std::uint64_t rectangles = 0;
  std::uint64_t rectangles_visited = 0;
  std::uint64_t rectangles_completed = 0;
  std::uint64_t slab_visits = 0;
  std::uint64_t interval_search_steps = 0;
  std::uint64_t witnesses = 0;
  std::uint64_t uncertain = 0;
  std::uint32_t device_flags = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_begin_bytes = 0;
  std::uint64_t device_free_low_bytes = 0;
  double validation_ms = 0.0;
  double query_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

Result certify_subset(
    cudaStream_t stream, DeviceStripView wells,
    DeviceRectangleView active, const Limits &limits, int device);

}  // namespace active4_subset_resident
}  // namespace klayout_cuda

#endif
