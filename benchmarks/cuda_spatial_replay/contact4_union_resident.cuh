/*
 * Exact bounded CONTACT.4 consumer for a resident Manhattan-union boundary.
 *
 * The core accepts an already-expanded DeviceContactView and indexes it only
 * from the resident-boundary callback, after the union engine has released
 * its sweep high-water storage.  Neither CONTACT nor ACTIVE geometry leaves
 * the device.  The original Request/ResidentContext API remains as a
 * qualification wrapper that validates and uploads host edges before entering
 * the same device-view core.
 */

#ifndef KLAYOUT_CUDA_CONTACT4_UNION_RESIDENT_CUH
#define KLAYOUT_CUDA_CONTACT4_UNION_RESIDENT_CUH

#include "active3_exact_predicate.h"
#include "manhattan_union_gpu.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace klayout_cuda {
namespace contact4_union_resident {

enum class ContactDirectionContract : std::uint32_t
{
  unspecified = 0,
  // The array consists of contiguous, closed, simple Manhattan contours in
  // material-on-right (clockwise hull) order.  Simplicity is a required
  // caller/live-serializer contract; this consumer independently rechecks
  // every edge, contour continuity/closure, and winding before device work.
  validated_material_on_right_contours = UINT32_C(0x43345231)
};

struct Limits
{
  std::uint64_t max_contact_edges = UINT64_C(64000000);
  std::uint64_t max_grid_cells = UINT64_C(4000000);
  std::uint64_t max_memberships = UINT64_C(200000000);
  std::uint64_t max_boundary_cell_visits = UINT64_C(200000000);
  // Hard ceiling on indexed member reads and therefore on predicate work.
  std::uint64_t max_member_visits = UINT64_C(1000000000);
  // Exact accepted-candidate census cap.  It is checked after classification;
  // max_member_visits is the hard pre-query work bound.
  std::uint64_t max_pair_work = UINT64_C(1000000000);
  std::uint32_t max_cells_per_contact_edge = 4096;
  std::uint32_t max_cells_per_boundary_edge = 4096;
};

struct ContactBounds
{
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
};

struct DeviceContactView
{
  // Non-owning pointer.  It must address `count` DirectedEdge records on
  // `DeviceRequest::device` and remain live until the synchronous resident
  // callback returns.
  const active3::DirectedEdge *device_edges = nullptr;
  std::uint64_t count = 0;

  // Inclusive coordinate bounds containing every edge endpoint.  Tight bounds
  // minimize the grid but are not required for correctness; undersized bounds
  // are detected on device and fail closed.
  ContactBounds bounds;
};

struct Request
{
  const active3::DirectedEdge *contact_edges = nullptr;
  std::uint64_t contact_edge_count = 0;
  std::int64_t distance =
      active3::kContact4QualifiedSceneCoordinateDistance;
  std::int64_t grid_cell_size = 2000;
  int device = 0;
  ContactDirectionContract contact_direction_contract =
      ContactDirectionContract::unspecified;
  Limits limits;
};

struct DeviceRequest
{
  DeviceContactView contacts;
  std::int64_t distance =
      active3::kContact4QualifiedSceneCoordinateDistance;
  std::int64_t grid_cell_size = 2000;
  int device = 0;

  // This is a required producer assertion for the device-resident view.
  // Per-edge geometry and containment in `contacts.bounds` are independently
  // checked on device, but contour simplicity/closure/winding are not copied
  // back to the host for revalidation.
  ContactDirectionContract contact_direction_contract =
      ContactDirectionContract::unspecified;
  Limits limits;
};

struct Result
{
  bool certified_empty = false;
  std::uint64_t contact_edges = 0;
  std::uint64_t boundary_segments = 0;
  std::uint64_t grid_cells = 0;
  std::uint64_t memberships = 0;
  std::uint64_t boundary_cell_visits = 0;
  std::uint64_t member_visits = 0;
  std::uint64_t candidate_pairs = 0;
  std::uint64_t hits = 0;
  std::uint64_t uncertain = 0;
  std::uint32_t device_flags = 0;

  // Memory is sampled at callback entry, after each allocating phase, and at
  // the query peak.  callback_free_begin_bytes is therefore the free memory
  // while only the union's final boundary remains live, not its sweep peak.
  std::uint64_t device_total_bytes = 0;
  std::uint64_t callback_free_begin_bytes = 0;
  std::uint64_t callback_free_low_bytes = 0;
  std::uint64_t post_scan_free_bytes = 0;
  // Incremental allocation above the still-live union boundary.  This is not
  // the full process high-water; callers combine callback_free_low_bytes with
  // the union's initial free-memory sample for that value.
  std::uint64_t callback_incremental_peak_bytes = 0;

  double setup_ms = 0.0;
  double boundary_preflight_ms = 0.0;
  double contact_h2d_ms = 0.0;
  double grid_count_ms = 0.0;
  double grid_build_ms = 0.0;
  double query_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

struct ResidentContext
{
  Request request;
  bool invoked = false;
  Result result;
};

struct DeviceResidentContext
{
  DeviceRequest request;
  bool invoked = false;
  Result result;
};

/*
 * Internal qualification C++ API.  Its layout is intentionally not a stable
 * binary ABI: every static consumer must be rebuilt with this header.
 *
 * v1 requires CUDA's legacy/default stream.  A future nondefault-stream
 * implementation must make device_vector initialization and every Thrust
 * dependency explicitly stream-ordered before relaxing this contract.
 */
void consume_boundary_hook(
    cudaStream_t stream,
    const manhattan_union::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const manhattan_union::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count, void *opaque);

manhattan_union::ResidentBoundaryHook make_resident_hook(
    ResidentContext *context);

void consume_device_boundary_hook(
    cudaStream_t stream,
    const manhattan_union::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const manhattan_union::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count, void *opaque);

manhattan_union::ResidentBoundaryHook make_device_resident_hook(
    DeviceResidentContext *context);

}  // namespace contact4_union_resident
}  // namespace klayout_cuda

#endif
