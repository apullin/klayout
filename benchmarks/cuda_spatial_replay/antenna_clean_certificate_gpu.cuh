/*
 * Conservative device-resident antenna clean certificates.
 *
 * This core deliberately proves only the easy, safe side of the antenna
 * predicate.  POLY/ACTIVE annotations and connectivity labels stay on the
 * device; each public operation copies back only a bounded scalar census.
 * A successful call with clean_certificate == false means UNCERTAIN, never a
 * reported antenna hit.  Every non-success return leaves caller output and
 * previously committed gate annotations unchanged.
 */

#ifndef KLAYOUT_CUDA_ANTENNA_CLEAN_CERTIFICATE_GPU_CUH
#define KLAYOUT_CUDA_ANTENNA_CLEAN_CERTIFICATE_GPU_CUH

#include "antenna_connectivity_gpu.cuh"

#include <cstdint>
#include <memory>

namespace klayout_cuda {
namespace antenna_clean_certificate {

enum class Status : std::uint32_t
{
  success = 0,
  invalid_configuration,
  not_initialized,
  malformed_input,
  capacity_exceeded,
  arithmetic_overflow,
  cuda_error,
  host_error
};

enum class MetalLevel : std::uint32_t
{
  metal1 = 1,
  metal2 = 2,
  metal3 = 3,
  metal4 = 4
};

struct Limits
{
  // Hard admission ceiling for all device bytes owned by this core plus the
  // caller-supplied external_live_device_bytes baseline.
  std::uint64_t max_live_device_bytes =
      UINT64_C(10) * 1024 * 1024 * 1024;
  std::uint64_t max_annotation_owners = UINT32_MAX;
  std::uint64_t max_labels = UINT32_MAX;
  std::uint64_t max_poly_tiles = UINT32_MAX;
  std::uint64_t max_active_tiles = UINT32_MAX;
  std::uint64_t max_metal_tiles = UINT32_MAX;
  std::uint64_t max_grid_cells = UINT64_C(100000000);
  std::uint64_t max_active_memberships = UINT32_MAX;
  std::uint64_t max_query_visits = UINT64_C(10000000000);
  std::uint32_t max_cell_members = UINT32_MAX;
};

enum class MemoryEvent : std::uint32_t
{
  temporary_allocate = 0,
  temporary_release,
  persistent_release,
  commit
};

struct MemoryAccounting
{
  std::uint64_t current_live_bytes = 0;
  std::uint64_t peak_live_bytes = 0;
  std::uint64_t current_temporary_bytes = 0;
  std::uint64_t peak_temporary_bytes = 0;
  std::uint64_t persistent_bytes = 0;
  std::uint64_t external_live_bytes = 0;
};

// The hook is observational and must not throw.  Hook exceptions are ignored
// so instrumentation can never change certificate semantics.
using MemoryHook = void (*)(
    MemoryEvent event, const MemoryAccounting &accounting,
    void *context);

struct Config
{
  int device = 0;
  std::int64_t grid_cell_size = 2000;
  // The core cannot discover memory owned by upstream resident geometry and
  // labels.  Production callers include those bytes here so the hard limit is
  // an end-to-end live-byte ceiling rather than component-only telemetry.
  std::uint64_t external_live_device_bytes = 0;
  Limits limits;
  MemoryHook memory_hook = nullptr;
  void *memory_hook_context = nullptr;
};

struct GateCensus
{
  std::uint64_t annotation_owners = 0;
  std::uint64_t poly_tiles = 0;
  std::uint64_t active_tiles = 0;
  std::uint64_t grid_cells = 0;
  std::uint64_t active_memberships = 0;
  // Exact grid candidate visits.  Multi-cell duplicate visits are counted
  // here, while deterministic intersection-cell ownership ensures each
  // positive POLY/ACTIVE pair contributes exactly once below.
  std::uint64_t candidate_visits = 0;
  std::uint64_t positive_intersections = 0;
  std::uint64_t gate_owners = 0;
  std::uint64_t persistent_bytes = 0;
  std::uint64_t peak_temporary_bytes = 0;
  std::uint64_t peak_live_bytes = 0;
  float kernel_milliseconds = 0.0f;
};

struct CheckpointCensus
{
  MetalLevel level = MetalLevel::metal1;
  bool clean_certificate = false;
  std::uint64_t labels = 0;
  std::uint64_t metal_tiles = 0;
  std::uint64_t roots = 0;
  std::uint64_t roots_with_metal = 0;
  std::uint64_t roots_without_gate = 0;
  std::uint64_t gate_roots_without_metal = 0;
  std::uint64_t ratio_certified_roots = 0;
  std::uint64_t uncertain_roots = 0;
  std::uint64_t persistent_bytes = 0;
  std::uint64_t peak_temporary_bytes = 0;
  std::uint64_t peak_live_bytes = 0;
  float kernel_milliseconds = 0.0f;
};

struct GateAnnotationDeviceView
{
  // Non-owning device pointers, invalidated by the next successful
  // build_gate_census call or Certificate destruction.
  const std::uint32_t *gate_present = nullptr;
  const std::uint64_t *max_single_intersection_area = nullptr;
  std::uint64_t count = 0;
  std::uint64_t epoch = 0;
  int device = 0;
};

class Certificate
{
public:
  explicit Certificate(const Config &config);
  ~Certificate();

  Certificate(const Certificate &) = delete;
  Certificate &operator=(const Certificate &) = delete;
  Certificate(Certificate &&) = delete;
  Certificate &operator=(Certificate &&) = delete;

  Status configuration_status() const noexcept;

  /*
   * Refresh device bytes owned by the surrounding resident transaction.
   * The update is transactional: an over-cap value leaves the prior
   * baseline unchanged.  Callers refresh it immediately before each bounded
   * certificate operation.
   */
  Status set_external_live_device_bytes(
      std::uint64_t bytes) noexcept;

  /*
   * Builds an exact complete positive-area POLY-vs-ACTIVE intersection
   * census.  Both rectangle arrays are device pointers.  Every valid
   * intersection sets gate_present for its POLY owner and atomically retains
   * that owner's maximum single intersection area.  The maximum is a safe
   * gate-area lower bound; it is intentionally not a union-area estimate.
   * annotation_owner_count is the current global owner-ID extent and must
   * cover both POLY and ACTIVE owner IDs; annotations are simply zero for
   * non-POLY owners.
   */
  Status build_gate_census(
      const antenna_connectivity::RectI64 *device_poly,
      std::uint64_t poly_count,
      const antenna_connectivity::RectI64 *device_active,
      std::uint64_t active_count,
      std::uint64_t annotation_owner_count,
      GateCensus *census) noexcept;

  /*
   * Reduces retained gate annotations through current canonical connectivity
   * labels and sums every exact target-metal tile area per root.  Summing
   * duplicate or overlapping metal tiles can only overestimate metal area,
   * preserving a conservative clean proof.
   *
   * device_labels is a const device pointer.  label_count may grow at later
   * connectivity stages but must cover every retained annotation owner and
   * every target-metal owner.
   */
  Status evaluate_checkpoint(
      MetalLevel level,
      const antenna_connectivity::RectI64 *device_metal,
      std::uint64_t metal_count,
      const std::uint32_t *device_labels,
      std::uint64_t label_count,
      CheckpointCensus *census) const noexcept;

  Status device_gate_view(
      GateAnnotationDeviceView *view) const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

const char *status_string(Status status) noexcept;

}  // namespace antenna_clean_certificate
}  // namespace klayout_cuda

#endif
