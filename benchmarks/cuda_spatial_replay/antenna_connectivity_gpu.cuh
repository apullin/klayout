/*
 * Exact, staged CUDA connectivity for closed Manhattan rectangles.
 *
 * The state is deliberately transactional.  append_stage builds candidate
 * pairs and a new DSU in temporary device storage, then commits only after all
 * validation, capacity checks, CUDA work, canonicalization, and D2H copies
 * succeed.  On every non-success return the state and both caller outputs are
 * unchanged.
 */

#ifndef KLAYOUT_CUDA_ANTENNA_CONNECTIVITY_GPU_CUH
#define KLAYOUT_CUDA_ANTENNA_CONNECTIVITY_GPU_CUH

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thrust/device_vector.h>
#include <vector>

namespace klayout_cuda {
namespace antenna_connectivity {

constexpr std::uint32_t kMaximumDomains = 64;
constexpr std::size_t kRelationSlots =
    static_cast<std::size_t>(kMaximumDomains) * kMaximumDomains;

struct RectI64
{
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
  // Stable polygon/node owner.  A node may be represented by multiple exact,
  // positive-area, interior-disjoint rectangles.  All rectangles for a new
  // node must arrive in one stage.  The core rejects overlapping or
  // disconnected owner decompositions; bounding-box substitution is never an
  // accepted interpretation.
  std::uint32_t owner = 0;
  std::uint32_t domain = 0;
};

enum class InputMemory : std::uint32_t
{
  host = 0,
  device = 1
};

enum class Status : std::uint32_t
{
  success = 0,
  invalid_configuration,
  malformed_input,
  capacity_exceeded,
  convergence_failure,
  poisoned_state,
  cuda_error,
  host_error
};

enum class AppendMode : std::uint32_t
{
  // State and resident views survive every failed append.
  transactional = 0,
  // Production-local low-peak mode.  Existing device vectors are consumed
  // before growth, avoiding a full committed-state clone.  Any failure after
  // consumption poisons the object and invalidates resident views; discard it.
  consuming = 1
};

struct Limits
{
  // Owner and rectangle pair keys are 32-bit per endpoint.  There is no
  // smaller semantic ceiling; production callers set lower admission caps
  // from available memory.
  std::uint64_t max_nodes = UINT32_MAX;
  std::uint64_t max_rectangles = UINT32_MAX;
  std::uint64_t max_memberships = UINT64_C(400000000);
  // Broad-mode pair occurrences are counted before multi-bin duplicate
  // removal.  Exact-filter mode counts canonical unique rectangle pairs.
  std::uint64_t max_pair_occurrences = UINT64_C(1000000000);
  std::uint64_t max_unique_candidates = UINT64_C(500000000);
  std::uint32_t max_cell_members = 1000000;
  // Conservative pre-filter work guard.  Broad mode guards the full
  // k*(k-1)/2 cell universe.  Exact staged mode guards only
  // old*new + new*(new-1)/2: immutable old-old pairs were processed by a
  // prior successful append and cannot become newly relevant.
  std::uint64_t max_pair_tests_per_cell = UINT64_C(100000000);
  // Aggregate selected-pair traversal guard for the exact-filtered parallel
  // path.  This is checked from the staged per-cell counts above before any
  // pair-test kernel is launched.
  std::uint64_t max_total_pair_tests = UINT64_C(100000000);
  std::uint32_t max_dsu_iterations = 128;
  // Hard admission envelope.  Every major allocation and sort scratch
  // frontier is checked against both cudaMemGetInfo() and this cap before the
  // allocation/algorithm.  The default is the qualification RTX 3080 ceiling.
  std::uint64_t max_device_bytes = UINT64_C(10240) * 1024 * 1024;
  std::uint64_t min_device_free_after_bytes =
      UINT64_C(256) * 1024 * 1024;
  // Thrust does not expose temporary-storage queries for the custom
  // CellMember comparator.  Admission reserves one full input copy plus this
  // fixed guard before every sort.  A production adapter may raise the fixed
  // guard; lowering it weakens the admission proof.
  std::uint64_t sort_scratch_fixed_guard_bytes =
      UINT64_C(64) * 1024 * 1024;
};

struct Config
{
  std::uint32_t domain_count = 0;
  // Connectivity is undirected.  The matrix must be symmetric and all bits
  // outside domain_count must be zero.  Diagonal bits opt each domain into
  // same-domain connectivity.
  std::array<std::uint64_t, kMaximumDomains> relation_rows{};
  // Production-local low-materialization mode.  Every grid-cell pair
  // involving at least one newly appended rectangle is traversed exactly;
  // old-old work is never repeated.  Only closed-touching same-owner tile
  // edges and closed-touching allowed cross-owner edges enter the occurrence
  // stream.
  // The default retains the historical broad-candidate census and behavior.
  bool exact_filter_before_materialization = false;
  /*
   * Exact-mode opt-in.  Before spatial memberships are built, identical
   * fully-transformed singleton rectangles are represented once and their
   * original owners are joined by canonical-min parent seeds.  Broad mode
   * rejects this option because its per-bin occurrence census is not a
   * logical owner-pair census.
   */
  bool quotient_identical_singletons = false;
  std::int64_t bin_size = 2000;
  int device = 0;
  Limits limits;
};

struct StageCensus
{
  std::uint64_t previous_nodes = 0;
  std::uint64_t appended_nodes = 0;
  std::uint64_t total_nodes = 0;
  std::uint64_t previous_rectangles = 0;
  std::uint64_t appended_rectangles = 0;
  std::uint64_t total_rectangles = 0;
  std::uint64_t retained_rectangles = 0;
  std::uint64_t retained_rectangle_capacity = 0;
  std::uint64_t released_rectangles = 0;
  // Physical post-quotient workload.  The legacy rectangle fields above keep
  // their exact raw/logical semantics.
  std::uint64_t physical_previous_rectangles = 0;
  std::uint64_t physical_appended_rectangles = 0;
  std::uint64_t physical_total_rectangles = 0;
  std::uint64_t physical_retained_rectangles = 0;
  std::uint64_t quotient_geometry_classes = 0;
  std::uint64_t quotient_collapsed_rectangles = 0;
  std::uint64_t quotient_weighted_internal_pairs = 0;
  std::uint64_t closed_domain_mask = 0;
  std::uint64_t memberships = 0;
  std::uint64_t physical_memberships = 0;
  std::uint64_t occupied_cells = 0;
  std::uint64_t exact_pair_tests = 0;
  std::uint64_t pair_occurrences = 0;
  std::uint64_t unique_candidates = 0;
  std::uint64_t exact_edges = 0;
  std::uint32_t dsu_iterations = 0;

  // Complete synchronous phase envelopes for production diagnosis.
  std::uint64_t quotient_ns = 0;
  std::uint64_t membership_count_ns = 0;
  std::uint64_t membership_fill_ns = 0;
  std::uint64_t membership_sort_ns = 0;
  std::uint64_t membership_group_ns = 0;
  std::uint64_t exact_count_ns = 0;
  std::uint64_t exact_fill_ns = 0;

  // Only the canonical slot min(domain_a, domain_b) * 64 +
  // max(domain_a, domain_b) is populated.
  std::array<std::uint64_t, kRelationSlots> candidates_by_relation{};
  std::array<std::uint64_t, kRelationSlots> edges_by_relation{};
};

struct DeviceLabelView
{
  // Non-owning pointer on device.  It remains valid through failed
  // transactional appends and snapshots, and is invalidated by the next
  // successful append, any consuming append attempt, or Connectivity
  // destruction.  append_stage is synchronous; a consumer may launch
  // dependent work after it returns without a host label round-trip.
  const std::uint32_t *labels = nullptr;
  std::uint64_t count = 0;
  std::uint64_t epoch = 0;
  int device = 0;
};

constexpr std::size_t relation_slot(
    std::uint32_t first, std::uint32_t second)
{
  return first <= second
             ? static_cast<std::size_t>(first) * kMaximumDomains + second
             : static_cast<std::size_t>(second) * kMaximumDomains + first;
}

class Connectivity
{
public:
  explicit Connectivity(const Config &config);
  ~Connectivity();

  Connectivity(const Connectivity &) = delete;
  Connectivity &operator=(const Connectivity &) = delete;
  Connectivity(Connectivity &&) = delete;
  Connectivity &operator=(Connectivity &&) = delete;

  Status configuration_status() const noexcept;
  std::uint64_t node_count() const noexcept;

  /*
   * Adds one stage and returns canonical-min labels for every owner accumulated
   * so far.  Owner IDs in this stage must cover the contiguous range
   * [node_count(), node_count() + new_node_count), with at least one rectangle
   * per owner and no rectangles for prior owners.  At least one endpoint of
   * each newly considered edge is therefore in this stage; old-old edges were
   * already resolved.
   *
   * close_domain_mask asserts that no later stage will add owners in those
   * domains.  The core rejects such later input.  It releases a closed
   * domain's rectangle decomposition only after every relation-neighbor
   * domain is also closed; owner labels and domains remain resident.  This
   * exact frontier rule is what permits M1->M2->M3->M4 streaming without
   * retaining the full expanded antenna geometry.
   *
   * rectangles may be host or device memory as selected by memory.  labels
   * may be null to keep the canonical labels device-resident and skip their
   * potentially hundreds-of-MB D2H copy.  Non-null labels and census are
   * assigned only on success.
   */
  Status append_stage(
      const RectI64 *rectangles, std::uint64_t rectangle_count,
      std::uint64_t new_node_count,
      std::uint64_t close_domain_mask, InputMemory memory,
      std::vector<std::uint32_t> *labels,
      StageCensus *census,
      AppendMode mode = AppendMode::transactional) noexcept;

  /*
   * Device-ownership consuming overload for production.  In consuming mode,
   * an empty-state append moves this vector directly into the core (including
   * reserved capacity).  Later stages copy into the preserved frontier
   * allocation and release this vector before broad-phase construction.
   * On a pre-consumption error the caller vector remains owned by the caller;
   * after consumption, any failure poisons the Connectivity object.
   */
  Status append_stage_consuming(
      thrust::device_vector<RectI64> &&rectangles,
      std::uint64_t new_node_count,
      std::uint64_t close_domain_mask,
      std::vector<std::uint32_t> *labels,
      StageCensus *census) noexcept;

  /*
   * Rebuilds the retained rectangle vector at its exact current size.  This is
   * intended for a stage boundary after a deliberately over-reserved
   * consuming frontier has served its purpose.  The replacement allocation
   * is admitted against the configured device cap before copying.  On every
   * failure the committed geometry, labels, and caller output are unchanged.
   *
   * retained_rectangle_capacity may be null.  A non-null output is assigned
   * only on success.
   */
  Status compact_retained_rectangles(
      std::uint64_t *retained_rectangle_capacity = nullptr) noexcept;

  // Copies the last committed canonical labels.  labels is unchanged on error.
  Status snapshot_labels(
      std::vector<std::uint32_t> *labels) const noexcept;

  // Returns a non-owning resident label view without synchronization or copy.
  // view is unchanged on error.
  Status device_label_view(DeviceLabelView *view) const noexcept;

private:
  Status append_stage_impl(
      const RectI64 *rectangles, std::uint64_t rectangle_count,
      std::uint64_t new_node_count,
      std::uint64_t close_domain_mask, InputMemory memory,
      std::vector<std::uint32_t> *labels,
      StageCensus *census, AppendMode mode,
      thrust::device_vector<RectI64> *owned_input) noexcept;

  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

const char *status_string(Status status) noexcept;

}  // namespace antenna_connectivity
}  // namespace klayout_cuda

#endif
