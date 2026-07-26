/*
 * Reusable exact CUDA consumer for the FreePDK45 M2 F90/F270 suffix.
 *
 * The input view is the canonical x-slab/y-interval representation exposed
 * by manhattan_union_gpu_core.  Calls are synchronous: no input device pointer
 * is retained, and every submitted operation has completed before return.
 * Errors are reported by exception so a ResidentStripHook owner can fail
 * closed without publishing a partial certificate.
 */

#ifndef KLAYOUT_CUDA_M2_RESIDENT_MORPHOLOGY_GPU_CUH
#define KLAYOUT_CUDA_M2_RESIDENT_MORPHOLOGY_GPU_CUH

#include "manhattan_union_gpu.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace klayout_cuda {
namespace m2_resident_morphology {

inline constexpr std::uint64_t kUniversalSourceVisitCap =
    UINT64_C(2000000000);
inline constexpr std::uint64_t kQualifiedProductionSourceVisitCap =
    UINT64_C(8000000000);
inline constexpr std::uint64_t kQualifiedM1ProductionSourceVisitCap =
    UINT64_C(12000000000);
inline constexpr std::uint64_t kQualifiedF90BoundarySegments =
    UINT64_C(4254384);
inline constexpr std::uint64_t kQualifiedF90BoundaryFnv64 =
    UINT64_C(2057677162565968634);
inline constexpr std::uint64_t kQualifiedF90LongSegments = UINT64_C(8);
inline constexpr std::uint64_t kQualifiedF90LongPairs = UINT64_C(28);

struct DeviceStripView
{
  const std::int64_t *xs = nullptr;
  std::uint32_t x_slabs = 0;
  const manhattan_union::StripInterval *intervals = nullptr;
  std::uint64_t interval_count = 0;
  const std::uint64_t *slab_offsets = nullptr;
  const std::uint32_t *slab_counts = nullptr;
};

struct Limits
{
  std::uint64_t max_input_x_slabs = UINT64_C(32000000);
  std::uint64_t max_input_intervals = UINT64_C(64000000);
  std::uint64_t max_output_slabs = UINT64_C(1000000);
  std::uint64_t max_output_intervals = UINT64_C(64000000);
  std::uint64_t max_raw_boundary_segments = UINT64_C(256000000);
  std::uint64_t max_boundary_segments = UINT64_C(64000000);
  std::uint64_t max_total_source_visits = kUniversalSourceVisitCap;
  std::uint64_t max_source_visits_per_band = UINT64_C(4000000);
  std::uint32_t max_active_slabs = 128;
  std::uint64_t max_long_segments = UINT64_C(4096);
};

struct PassMetrics
{
  std::uint64_t output_intervals = 0;
  std::uint64_t source_visits = 0;
  std::uint32_t max_active_slabs = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_begin_bytes = 0;
  std::uint64_t device_free_low_bytes = 0;
  double elapsed_ms = 0.0;
};

struct Result
{
  std::uint32_t source_x_slabs = 0;
  std::uint64_t source_intervals = 0;
  PassMetrics erode89;
  PassMetrics dilate90;
  PassMetrics boundary;
  PassMetrics erode269_count;
  std::uint64_t f90_boundary_segments = 0;
  // Populated only with the optional host qualification boundary.
  std::uint64_t f90_boundary_fnv64 = 0;
  std::uint64_t f90_long_segments = 0;
  std::uint64_t f90_space_pairs_checked = 0;
  std::uint64_t f90_space_violations = 0;
  std::uint64_t f90_space_uncertain = 0;
  std::uint64_t f270_eroded_intervals = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_begin_bytes = 0;
  std::uint64_t device_free_low_bytes = 0;
  double boundary_and_long_space_ms = 0.0;
  double total_ms = 0.0;
  // Filled only when Request::copy_f90_boundary_to_host is true.
  std::vector<manhattan_union::DirectedSegmentI64> f90_boundary;
};

struct Request
{
  Limits limits;
  bool copy_f90_boundary_to_host = false;
  // The 8B allowance is qualified only for the pinned M2 production scene.
  bool allow_qualified_production_work_cap = false;
  // The distinct 12B allowance is qualified only for a pinned M1 production
  // scene by its owning adapter.  It must never be enabled together with the
  // M2 allowance or for an arbitrary caller-selected census.
  bool allow_qualified_m1_production_work_cap = false;
};

// Exact synchronous API.  The fixed operation is:
//   F90 = size(+90, size(-89, source))
//   F270 certificate = size(-269, F90) is empty
// plus the exact 600-DBU-long F90 edge space(180) certificate.
Result consume_f90_f270(cudaStream_t stream, const DeviceStripView &source,
                        const Request &request);

enum class QualificationOperation
{
  erode,
  dilate,
  erode_then_dilate,
};

// Qualification-only seam used by the independent raster differential gate.
// It exercises the same bounded kernels as consume_f90_f270 and intentionally
// returns a host boundary.
std::vector<manhattan_union::DirectedSegmentI64>
qualification_boundary(
    cudaStream_t stream, const DeviceStripView &source,
    QualificationOperation operation, std::int64_t first_radius,
    std::int64_t second_radius, const Limits &limits = Limits{});

struct LongSpaceCertificate
{
  std::uint64_t pairs_checked = 0;
  std::uint64_t violations = 0;
  std::uint64_t uncertain = 0;
};

// Host certificate seam retained for the exact 180-DBU predicate gate.
LongSpaceCertificate certify_f90_long_edge_space(
    const std::vector<manhattan_union::DirectedSegmentI64> &segments,
    std::uint64_t max_segments = UINT64_C(4096));

/*
 * Exact one-orientation M1.1/M1.2 certificate over canonical union strips.
 *
 * Every positive-width x slab is an exact vertical slice of the union.
 * An occupied y interval shorter than distance exposes an opposing
 * horizontal-boundary width violation; a gap shorter than distance between
 * consecutive intervals exposes an unshielded spacing violation.  A bounded
 * endpoint index additionally rejects every sub-distance Euclidean candidate
 * whose parallel-edge projections do not overlap.  Running the same
 * certificate once on the original geometry and once after x/y transposition
 * covers both Manhattan edge orientations without materializing the
 * multi-million-segment boundary.  Endpoint candidates are conservative:
 * finding one retains the CPU path, while finding none is an exact empty
 * certificate.
 */
struct BaseWidthSpaceResult
{
  std::uint64_t slabs_checked = 0;
  std::uint64_t intervals_checked = 0;
  std::uint64_t gaps_checked = 0;
  std::uint64_t width_violations = 0;
  std::uint64_t space_violations = 0;
  std::uint64_t corner_endpoint_count = 0;
  std::uint64_t corner_pair_work = 0;
  std::uint64_t corner_candidates = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t device_free_begin_bytes = 0;
  std::uint64_t device_free_low_bytes = 0;
  std::uint32_t device_flags = 0;
  double elapsed_ms = 0.0;
};

enum class BaseWidthSpaceProfile : std::uint32_t
{
  // FreePDK45 METAL1.1/.2: 65 nm at 0.5 nm/DBU.
  m1_130 = UINT32_C(0x4d313130),
  // FreePDK45 IMPLANT.3/.4: 45 nm at 0.5 nm/DBU.
  implant_90 = UINT32_C(0x49303930),
};

struct BaseWidthSpaceContext
{
  BaseWidthSpaceProfile profile = BaseWidthSpaceProfile::m1_130;
  std::int64_t distance = 0;
  std::int64_t origin_x = 0;
  std::int64_t origin_y = 0;
  std::uint64_t max_corner_endpoints = UINT64_C(64000000);
  std::uint64_t max_corner_pair_work = UINT64_C(2000000000);
  bool invoked = false;
  BaseWidthSpaceResult result;
};

void consume_base_width_space_hook(
    cudaStream_t stream, const std::int64_t *xs,
    std::uint32_t x_slabs,
    const manhattan_union::StripInterval *intervals,
    std::uint64_t interval_count, const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque);

manhattan_union::ResidentStripHook make_base_width_space_hook(
    BaseWidthSpaceContext *context,
    bool stop_before_boundary = true);

// Checked adapter for manhattan_union::ResidentStripHook.
struct ResidentContext
{
  Request request;
  bool invoked = false;
  Result result;
};

void consume_f90_f270_hook(
    cudaStream_t stream, const std::int64_t *xs, std::uint32_t x_slabs,
    const manhattan_union::StripInterval *intervals,
    std::uint64_t interval_count, const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque);

manhattan_union::ResidentStripHook make_resident_hook(
    ResidentContext *context);

}  // namespace m2_resident_morphology
}  // namespace klayout_cuda

#endif
