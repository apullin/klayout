/*
 * Combined production gate for the exact raw-M2 union transaction.
 *
 * This is the missing composition between the production CUDA DSO, KLayout's
 * runtime loader/copy/release wrapper, and the checked endpoint stitch.  The
 * pinned KACT is only a compact production request source.  The separately
 * decoded stock merged-boundary file remains the independent exact oracle.
 */

#define KLAYOUT_M2_UNION_PRODUCTION_SCENE_ONLY
#include "m2_union_production_backend_gate.cu"
#undef KLAYOUT_M2_UNION_PRODUCTION_SCENE_ONLY

#include "dbBox.h"
#include "dbCudaM2Rules.h"
#include "dbCudaSpatialBackend.h"
#include "dbRegion.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace combined_gate {

namespace oracle = klayout_cuda::m2_boundary_oracle;
using Clock = std::chrono::steady_clock;
using CounterFunction = std::uint64_t (*)(std::uint32_t);

constexpr std::uint64_t kContours = UINT64_C(14222);
constexpr std::uint64_t kMaxVertices = UINT64_C(2084);

enum CounterSelector : std::uint32_t
{
  kAllocationCount = 0,
  kReleaseCallCount = 1,
  kOwnedReleaseCount = 2,
  kOutstandingCount = 3
};

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double ns_ms(std::uint64_t nanoseconds)
{
  return static_cast<double>(nanoseconds) / 1000000.0;
}

std::uint64_t boundary_fnv64(
    const production_gate::Segment *segments, std::uint64_t count)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&hash](std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
      hash *= UINT64_C(1099511628211);
    }
  };
  mix(count);
  for (std::uint64_t index = 0; index < count; ++index) {
    mix(segments[index].axis);
    mix(static_cast<std::uint32_t>(segments[index].side));
    mix(static_cast<std::uint64_t>(segments[index].fixed));
    mix(static_cast<std::uint64_t>(segments[index].lo));
    mix(static_cast<std::uint64_t>(segments[index].hi));
  }
  return hash;
}

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

bool set_environment(const char *name, const char *value)
{
#if defined(_WIN32)
  return _putenv_s(name, value) == 0;
#else
  return setenv(name, value, 1) == 0;
#endif
}

class CounterModule
{
public:
  explicit CounterModule(const char *path)
      : m_handle(nullptr), m_counter(nullptr)
  {
#if defined(_WIN32)
    m_handle = LoadLibraryA(path);
    if (m_handle) {
      m_counter = reinterpret_cast<CounterFunction>(
          GetProcAddress(
              static_cast<HMODULE>(m_handle),
              "klayout_cuda_spatial_m2_union_test_counter_v1"));
    }
#else
    m_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (m_handle) {
      m_counter = reinterpret_cast<CounterFunction>(
          dlsym(
              m_handle,
              "klayout_cuda_spatial_m2_union_test_counter_v1"));
    }
#endif
    require(
        m_handle && m_counter,
        "production DSO has no ownership/release test counter");
  }

  CounterModule(const CounterModule &) = delete;
  CounterModule &operator=(const CounterModule &) = delete;

  ~CounterModule()
  {
#if defined(_WIN32)
    if (m_handle) FreeLibrary(static_cast<HMODULE>(m_handle));
#else
    if (m_handle) dlclose(m_handle);
#endif
  }

  std::uint64_t get(CounterSelector selector) const
  {
    return m_counter(static_cast<std::uint32_t>(selector));
  }

private:
#if defined(_WIN32)
  HMODULE m_handle;
#else
  void *m_handle;
#endif
  CounterFunction m_counter;
};

struct RegionSnapshot
{
  std::size_t count;
  db::Box bbox;
  bool merged_semantics;
  bool is_merged;
  std::string text;
};

RegionSnapshot snapshot(const db::Region &region)
{
  return RegionSnapshot {
      region.count(), region.bbox(), region.merged_semantics(),
      region.is_merged(), region.to_string()};
}

bool unchanged(const db::Region &region, const RegionSnapshot &before)
{
  return region.count() == before.count &&
         region.bbox() == before.bbox &&
         region.merged_semantics() == before.merged_semantics &&
         region.is_merged() == before.is_merged &&
         region.to_string() == before.text;
}

struct FlatStatsSnapshot
{
  std::uint64_t segment_count;
  std::uint64_t contour_count;
  std::uint64_t vertex_count;
  std::uint64_t max_vertices;
};

FlatStatsSnapshot snapshot(const db::CudaM2FlatUnionStats &stats)
{
  return FlatStatsSnapshot {
      stats.segment_count, stats.contour_count, stats.vertex_count,
      stats.max_vertices};
}

bool unchanged(
    const db::CudaM2FlatUnionStats &stats,
    const FlatStatsSnapshot &before)
{
  return stats.segment_count == before.segment_count &&
         stats.contour_count == before.contour_count &&
         stats.vertex_count == before.vertex_count &&
         stats.max_vertices == before.max_vertices;
}

struct Transaction
{
  db::CudaM2UnionAttempt backend;
  db::CudaM2UnionTiming timing{};
  db::CudaM2FlatUnionStats flat_stats;
  std::string topology_reason;
  double loader_ms = 0.0;
  double stitch_ms = 0.0;
  bool complete = false;
};

Transaction run_transaction(
    const production_gate::Request &request, db::Region &output)
{
  Transaction transaction;
  const Clock::time_point loader_begin = Clock::now();
  transaction.backend = db::cuda_spatial_try_m2_union_with_timing(
      request, &transaction.timing, sizeof(transaction.timing));
  const Clock::time_point loader_end = Clock::now();
  transaction.loader_ms = milliseconds(loader_begin, loader_end);
  if (transaction.backend.disposition !=
      db::CudaM2UnionAttempt::Complete) {
    return transaction;
  }

  const Clock::time_point stitch_begin = Clock::now();
  transaction.complete = db::cuda_m2_union_boundary_to_flat_region(
      transaction.backend.segments.data(),
      transaction.backend.segments.size(),
      transaction.backend.boundary_fnv64, output,
      &transaction.flat_stats, &transaction.topology_reason);
  transaction.stitch_ms = milliseconds(stitch_begin, Clock::now());
  return transaction;
}

void validate_production_attempt(
    const db::CudaM2UnionAttempt &attempt,
    const db::CudaM2UnionTiming &timing,
    const production_gate::RawScene &raw)
{
  require(
      attempt.disposition == db::CudaM2UnionAttempt::Complete,
      "real host loader did not complete the production request: " +
          attempt.message);
  require(
      attempt.fallback_flags == KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE &&
          attempt.device_flags == 0 &&
          attempt.context_count == raw.contexts.size() &&
          attempt.metal_context_count == raw.metal_contexts.size() &&
          attempt.cell_count == raw.cells.size() &&
          attempt.polygon_count == raw.polygons.size() &&
          attempt.edge_count == raw.edges.size() &&
          attempt.flat_polygon_count ==
              production_gate::kFlatPolygons &&
          attempt.flat_edge_count == production_gate::kFlatEdges &&
          attempt.rectangle_count == production_gate::kRectangles &&
          attempt.x_slab_count == production_gate::kXSlabs &&
          attempt.membership_count == production_gate::kMemberships &&
          attempt.event_count == production_gate::kEvents &&
          attempt.strip_interval_count == production_gate::kStrips &&
          attempt.raw_segment_count ==
              production_gate::kRawSegments &&
          attempt.segments.size() == production_gate::kSegments &&
          attempt.boundary_fnv64 ==
              production_gate::kBoundaryFnv64,
      "real host-loader proof differs from the production allowlist");

  const unsigned __int128 component_ns =
      static_cast<unsigned __int128>(timing.setup_ns) +
      timing.h2d_ns + timing.rectangle_expand_ns +
      timing.x_membership_ns + timing.strip_scan_ns +
      timing.boundary_ns + timing.d2h_ns;
  require(
      timing.format_version == db::CudaM2UnionTiming::FormatVersion &&
          timing.struct_size == sizeof(timing) &&
          timing.setup_ns && timing.h2d_ns &&
          timing.rectangle_expand_ns && timing.x_membership_ns &&
          timing.strip_scan_ns && timing.boundary_ns &&
          timing.d2h_ns && timing.total_ns &&
          timing.total_ns == attempt.total_ns &&
          component_ns <= timing.total_ns,
      "real host loader did not preserve charged backend component timing");
}

void compare_exact_oracle(
    const db::CudaM2UnionAttempt &attempt,
    const oracle::BoundaryOracle &expected)
{
  require(
      expected.segments.size() == production_gate::kSegments &&
          expected.boundary_fnv64 ==
              production_gate::kBoundaryFnv64 &&
          attempt.segments.size() == expected.segments.size() &&
          attempt.boundary_fnv64 == expected.boundary_fnv64,
      "combined transaction boundary census/FNV differs from the oracle");
  for (std::size_t index = 0; index < expected.segments.size(); ++index) {
    const production_gate::Segment &candidate =
        attempt.segments[index];
    const oracle::DirectedSegmentI64 &reference =
        expected.segments[index];
    if (candidate.fixed != reference.fixed ||
        candidate.lo != reference.lo ||
        candidate.hi != reference.hi ||
        candidate.side != reference.side ||
        candidate.axis !=
            static_cast<std::uint32_t>(reference.axis)) {
      throw std::runtime_error(
          "host-owned production boundary first differs from the exact "
          "oracle at segment " + std::to_string(index));
    }
  }
}

int run(
    const std::string &backend_path, const std::string &kact_path,
    const std::string &oracle_path, int device)
{
  require(
      set_environment(
          "KLAYOUT_CUDA_SPATIAL_BACKEND", backend_path.c_str()) &&
          set_environment("KLAYOUT_CUDA_M2_RULES", "1") &&
          set_environment("KLAYOUT_CUDA_M2_RULES_TELEMETRY", "0"),
      "unable to configure the real host-loader environment");

  const Clock::time_point verification_begin = Clock::now();
  CounterModule counters(backend_path.c_str());
  require(
      counters.get(kAllocationCount) == 0 &&
          counters.get(kReleaseCallCount) == 0 &&
          counters.get(kOwnedReleaseCount) == 0 &&
          counters.get(kOutstandingCount) == 0,
      "production DSO ownership counters were not initially clean");
  require(
      db::cuda_spatial_m2_union_requested(),
      "real host loader did not advertise the complete run/release pair");

  production_gate::Request incompatible_request{};
  db::CudaM2UnionTiming incompatible_timing{};
  const db::CudaM2UnionAttempt incompatible_attempt =
      db::cuda_spatial_try_m2_union_with_timing(
          incompatible_request, &incompatible_timing,
          sizeof(incompatible_timing) - 1);
  require(
      incompatible_attempt.disposition ==
              db::CudaM2UnionAttempt::InvalidResult &&
          counters.get(kAllocationCount) == 0 &&
          counters.get(kReleaseCallCount) == 0,
      "incompatible additive timing record did not decline before the DSO");

  const Clock::time_point raw_begin = Clock::now();
  LoadedScene source = load_and_validate(kact_path);
  require(
      hex_digest(source.header.scene_sha256, 32) ==
          production_gate::kProductionKactSha256 &&
          source.header.well_layer == 101 &&
          source.header.well_datatype == 0,
      "combined gate KACT identity/layer differs from the allowlist");
  LoweredScene hierarchy =
      lower_hierarchy(source, UINT64_C(4000000));
  production_gate::RawScene raw =
      production_gate::build_raw_scene(source, hierarchy, device);
  raw.bind();
  const double raw_prepare_ms =
      milliseconds(raw_begin, Clock::now());

  const Clock::time_point oracle_load_begin = Clock::now();
  oracle::LoadOptions options;
  options.expected_file_sha256 =
      production_gate::kOracleFileSha256;
  options.expected_scene_sha256 =
      production_gate::kOracleSceneSha256;
  options.expected_boundary_sha256 =
      production_gate::kBoundarySha256;
  const oracle::BoundaryOracle expected =
      oracle::load_cpu_merged_boundary(oracle_path, options);
  const double oracle_load_ms =
      milliseconds(oracle_load_begin, Clock::now());

  db::Region output(db::Box(-400, -300, -200, -100));
  const RegionSnapshot output_before = snapshot(output);
  Transaction success = run_transaction(raw.request, output);
  require(success.complete, "production boundary stitch declined: " +
                                success.topology_reason);
  validate_production_attempt(success.backend, success.timing, raw);
  require(
      counters.get(kAllocationCount) == 1 &&
          counters.get(kReleaseCallCount) == 1 &&
          counters.get(kOwnedReleaseCount) == 1 &&
          counters.get(kOutstandingCount) == 0,
      "real loader did not release exactly one DSO-owned result");
  require(
      !unchanged(output, output_before) &&
          output.merged_semantics() && output.is_merged() &&
          output.count() == kContours &&
          success.flat_stats.segment_count ==
              production_gate::kSegments &&
          success.flat_stats.contour_count == kContours &&
          success.flat_stats.vertex_count ==
              production_gate::kSegments &&
          success.flat_stats.max_vertices == kMaxVertices,
      "checked production stitch did not preserve its exact flat census");

  const Clock::time_point oracle_compare_begin = Clock::now();
  compare_exact_oracle(success.backend, expected);
  const double oracle_compare_ms =
      milliseconds(oracle_compare_begin, Clock::now());

  // A canonical, FNV-consistent open boundary must reach topology validation
  // and decline without publishing a partial replacement or partial stats.
  const production_gate::Segment open_segment = {
      500, 100, 900, 1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL};
  const std::uint64_t open_fnv =
      boundary_fnv64(&open_segment, 1);
  std::string canonical_reason;
  require(
      db::cuda_spatial_validate_m2_union_boundary(
          &open_segment, 1, open_fnv, &canonical_reason) &&
          canonical_reason.empty(),
      "open-endpoint fixture did not pass canonical/FNV validation");
  db::Region topology_output(db::Box(700, 800, 900, 1000));
  const RegionSnapshot topology_before = snapshot(topology_output);
  db::CudaM2FlatUnionStats topology_stats;
  topology_stats.segment_count = 11;
  topology_stats.contour_count = 12;
  topology_stats.vertex_count = 13;
  topology_stats.max_vertices = 14;
  const FlatStatsSnapshot topology_stats_before =
      snapshot(topology_stats);
  std::string topology_reason;
  const Clock::time_point topology_begin = Clock::now();
  const bool topology_accepted =
      db::cuda_m2_union_boundary_to_flat_region(
          &open_segment, 1, open_fnv, topology_output,
          &topology_stats, &topology_reason);
  const double topology_decline_ms =
      milliseconds(topology_begin, Clock::now());
  require(
      !topology_accepted &&
          topology_reason == "M2 boundary has an open endpoint" &&
          unchanged(topology_output, topology_before) &&
          unchanged(topology_stats, topology_stats_before),
      "checked stitch did not decline atomically");

  // Force a real, bounded DSO capacity fallback.  The host transaction must
  // preserve the caller's pristine output and still invoke release once.
  production_gate::Request fallback_request = raw.request;
  fallback_request.max_segments = production_gate::kSegments - 1;
  db::Region fallback_output(db::Box(1100, 1200, 1300, 1400));
  const RegionSnapshot fallback_before = snapshot(fallback_output);
  Transaction fallback =
      run_transaction(fallback_request, fallback_output);
  require(
      !fallback.complete &&
          fallback.backend.disposition ==
              db::CudaM2UnionAttempt::BackendFallback &&
          fallback.backend.fallback_flags ==
              KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY &&
          fallback.backend.segments.empty() &&
          unchanged(fallback_output, fallback_before),
      "real backend capacity fallback was not atomic");
  require(
      counters.get(kAllocationCount) == 1 &&
          counters.get(kReleaseCallCount) == 2 &&
          counters.get(kOwnedReleaseCount) == 1 &&
          counters.get(kOutstandingCount) == 0,
      "fallback path did not call release exactly once without leaking");

  const double backend_total_ms = ns_ms(success.timing.total_ns);
  const double loader_copy_validate_release_ms =
      std::max(0.0, success.loader_ms - backend_total_ms);
  const double charged_live_seam_ms =
      success.loader_ms + success.stitch_ms;
  const double charged_gate_end_to_end_ms =
      raw_prepare_ms + charged_live_seam_ms;

  std::cout
      << "M2_UNION_REAL_TRANSACTION_GATE PASS"
      << " contexts=" << success.backend.context_count
      << " metal_contexts=" << success.backend.metal_context_count
      << " rectangles=" << success.backend.rectangle_count
      << " x_slabs=" << success.backend.x_slab_count
      << " memberships=" << success.backend.membership_count
      << " events=" << success.backend.event_count
      << " strips=" << success.backend.strip_interval_count
      << " raw_segments=" << success.backend.raw_segment_count
      << " segments=" << success.backend.segments.size()
      << " boundary_fnv64=" << success.backend.boundary_fnv64
      << " contours=" << success.flat_stats.contour_count
      << " vertices=" << success.flat_stats.vertex_count
      << " max_vertices=" << success.flat_stats.max_vertices
      << " merged_semantics=" << output.merged_semantics()
      << " is_merged=" << output.is_merged()
      << " allocations=" << counters.get(kAllocationCount)
      << " release_calls=" << counters.get(kReleaseCallCount)
      << " owned_releases=" << counters.get(kOwnedReleaseCount)
      << " outstanding=" << counters.get(kOutstandingCount)
      << " timing_record_guard=1"
      << " fallback_atomic=1"
      << " topology_decline_atomic=1\n"
      << std::fixed << std::setprecision(3)
      << "M2_UNION_REAL_TRANSACTION_TIMING"
      << " gate_input_prepare_charged_ms=" << raw_prepare_ms
      << " backend_setup_ms=" << ns_ms(success.timing.setup_ns)
      << " backend_h2d_ms=" << ns_ms(success.timing.h2d_ns)
      << " backend_rectangle_expand_ms="
      << ns_ms(success.timing.rectangle_expand_ns)
      << " backend_x_membership_ms="
      << ns_ms(success.timing.x_membership_ns)
      << " backend_strip_scan_ms="
      << ns_ms(success.timing.strip_scan_ns)
      << " backend_boundary_ms="
      << ns_ms(success.timing.boundary_ns)
      << " backend_d2h_ms=" << ns_ms(success.timing.d2h_ns)
      << " backend_total_ms=" << backend_total_ms
      << " loader_observed_ms=" << success.loader_ms
      << " loader_copy_validate_release_ms="
      << loader_copy_validate_release_ms
      << " checked_stitch_ms=" << success.stitch_ms
      << " charged_live_seam_ms=" << charged_live_seam_ms
      << " charged_gate_end_to_end_ms="
      << charged_gate_end_to_end_ms
      << " fallback_loader_ms=" << fallback.loader_ms
      << " topology_decline_ms=" << topology_decline_ms
      << " oracle_load_qualification_ms=" << oracle_load_ms
      << " oracle_compare_qualification_ms=" << oracle_compare_ms
      << " verification_total_ms="
      << milliseconds(verification_begin, Clock::now()) << "\n";
  return 0;
}

}  // namespace combined_gate

int main(int argc, char **argv)
{
  try {
    if (argc < 4 || argc > 5) {
      std::cerr
          << "usage: m2_union_real_transaction_gate "
             "BACKEND KACT ORACLE [DEVICE]\n";
      return 2;
    }
    const long device = argc == 5 ? std::stol(argv[4]) : 0;
    if (device < 0 ||
        device > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error("invalid CUDA device");
    }
    return combined_gate::run(
        argv[1], argv[2], argv[3], static_cast<int>(device));
  } catch (const std::exception &error) {
    std::cerr << "M2_UNION_REAL_TRANSACTION_GATE FAIL: "
              << error.what() << "\n";
    return 1;
  }
}
