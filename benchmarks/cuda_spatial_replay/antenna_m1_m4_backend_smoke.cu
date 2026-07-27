/*
 * Directed ABI smoke for the exact resident M1-through-M4 adapter.
 */

#include "antenna_m1_m4_backend.cuh"
#ifdef KLAYOUT_ANTENNA_HOST_VALIDATOR
#include "dbCudaSpatialBackend.h"
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Request = klayout_cuda_spatial_antenna_m1_m4_request_v1;
using Result = klayout_cuda_spatial_antenna_m1_m4_result_v1;
using Context = klayout_cuda_spatial_m1_width_space_context_v1;
using Cell = klayout_cuda_spatial_antenna_m1_m4_cell_v1;
using Polygon = klayout_cuda_spatial_m1_width_space_polygon_v1;
using Edge = klayout_cuda_spatial_m1_width_space_edge_v1;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

struct Fixture
{
  Request request{};
  std::array<std::uint64_t, 1> source_cells{{42}};
  std::array<Context, 1> contexts{{{0, 0, 0, 0}}};
  std::array<std::uint32_t, 1> parents{{UINT32_MAX}};
  std::array<std::array<Cell, 1>, 12> cells{};
  std::array<std::array<Polygon, 1>, 12> polygons{};
  std::array<std::array<Edge, 4>, 12> edges{};

  Fixture()
  {
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.opcode =
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_SHARED_EMPTY;
    request.option_flags =
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_QUALIFIED_OPTIONS;
    request.format_version = 1;
    request.dbu_per_micron = 2000;
    request.requested_mask =
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_STAGES;
    request.stage_count =
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT;
    request.ratio_numerator =
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_NUMERATOR;
    request.ratio_denominator =
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_DENOMINATOR;
    request.domain_count =
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
    request.device = 0;

    request.hierarchy.struct_size = sizeof(request.hierarchy);
    request.hierarchy.format_version = 2;
    request.hierarchy.dbu_per_micron = 2000;
    request.hierarchy.root_cell = 0;
    request.hierarchy.source_root_cell_index = source_cells[0];
    request.hierarchy.source_cell_indices = source_cells.data();
    request.hierarchy.source_cell_count = source_cells.size();
    request.hierarchy.source_cell_index_record_bytes =
        sizeof(std::uint64_t);
    request.hierarchy.contexts = contexts.data();
    request.hierarchy.context_count = contexts.size();
    request.hierarchy.context_record_bytes = sizeof(Context);
    request.hierarchy.context_parent_ids = parents.data();
    request.hierarchy.context_parent_count = parents.size();
    request.hierarchy.context_parent_record_bytes =
        sizeof(std::uint32_t);

    auto &capacity = request.capacity;
    capacity.struct_size = sizeof(capacity);
    capacity.max_cells = 100;
    capacity.max_contexts = 100;
    capacity.max_stored_polygons = 100;
    capacity.max_stored_edges = 1000;
    capacity.max_flat_polygons = 1000;
    capacity.max_flat_edges = 10000;
    capacity.max_total_stored_bytes = UINT64_C(1) << 30;
    capacity.max_total_expanded_geometry_bytes = UINT64_C(1) << 30;
    capacity.max_estimated_peak_bytes = UINT64_C(2) << 30;
    capacity.max_nodes = 1000;
    capacity.max_rectangles = 1000;
    capacity.max_memberships = 100000;
    capacity.max_pair_occurrences = 100000;
    capacity.max_unique_candidates = 100000;
    capacity.max_cell_members = 1000;
    capacity.max_dsu_iterations = 128;
    capacity.max_rule_work = 100000;
    capacity.max_device_bytes = UINT64_C(1) << 30;

    static constexpr std::uint32_t physical_layers[12] = {
        9, 1, 4, 3, 10, 11, 12, 13, 14, 15, 16, 17};
    static constexpr const char *digest_domains[12] = {
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_POLY_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ACTIVE_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NPLUS_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NWELL_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_CONTACT_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M1_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA1_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M2_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA2_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M3_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA3_DIGEST_DOMAIN,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M4_DIGEST_DOMAIN};
    for (std::size_t role = 0; role < 12; ++role) {
      cells[role][0] = {0, 0, 1, 4};
      polygons[role][0] = {0, 0, 0, 10, 10, 0, 4};
      edges[role] = {{
          {0, 0, 0, 10},
          {0, 10, 10, 10},
          {10, 10, 10, 0},
          {10, 0, 0, 0}}};
      auto &domain = request.domains[role];
      domain.struct_size = sizeof(domain);
      domain.role = role;
      domain.physical_layer = physical_layers[role];
      domain.datatype = 0;
      domain.source_layer_index =
          static_cast<std::uint32_t>(100 + role);
      domain.cells = cells[role].data();
      domain.cell_count = 1;
      domain.cell_record_bytes = sizeof(Cell);
      domain.polygons = polygons[role].data();
      domain.polygon_count = 1;
      domain.polygon_record_bytes = sizeof(Polygon);
      domain.edges = edges[role].data();
      domain.edge_count = 4;
      domain.edge_record_bytes = sizeof(Edge);
      std::memcpy(domain.digest_domain, digest_domains[role], 8);
    }
    const char *error = nullptr;
    require(
        klayout_cuda::antenna_m1_m4_backend::prepare_test_request(
            &request, &error),
        std::string("prepare fixture: ") +
            (error ? error : "unknown failure"));
    require(
        request.domains[0].stored_bytes == 280 &&
            request.domains[0].expanded_geometry_bytes == 180 &&
            request.census.total_stored_bytes == 3572 &&
            request.census.total_expanded_geometry_bytes == 2160 &&
            request.census.estimated_peak_bytes == 5732,
        "fixture canonical byte census changed");
  }
};

void complete_smoke()
{
  Fixture fixture;
  Result result{};
  const int status =
      klayout_cuda_spatial_run_antenna_m1_m4_empty_v1(
          &fixture.request, &result);
  require(
      status == KLAYOUT_CUDA_SPATIAL_OK,
      std::string("complete transaction: ") + result.message);
  require(
      result.disposition ==
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_COMPLETE &&
          result.clean_mask ==
              KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_STAGES &&
          result.closed_domain_mask ==
              KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_DOMAINS,
      "complete transaction returned incomplete proof state");
#ifdef KLAYOUT_ANTENNA_HOST_VALIDATOR
  std::string host_error;
  require(
      db::cuda_spatial_validate_antenna_m1_m4_result(
          fixture.request, result, status, &host_error),
      std::string("host validator rejected real adapter result: ") +
          host_error);
#endif
  const std::uint64_t retained[4] = {1, 1, 1, 0};
  const std::uint64_t released[4] = {2, 2, 2, 3};
  /*
   * Every tiny rectangle occupies one connectivity bin.  These totals are
   * therefore a directed assertion that the adapter reports the exact sum of
   * its hidden domain sub-appends (3 sub-appends for M1, 2 thereafter), not
   * merely the final sub-append census.
   */
  const std::uint64_t memberships[4] = {5, 4, 4, 4};
  const std::uint64_t occupied_cells[4] = {3, 2, 2, 2};
  for (std::size_t stage = 0; stage < 4; ++stage) {
    require(
        result.stages[stage].retained_rectangle_count ==
            retained[stage] &&
            result.stages[stage].released_rectangle_count ==
                released[stage] &&
            result.stages[stage].membership_count ==
                memberships[stage] &&
            result.stages[stage].occupied_cell_count ==
                occupied_cells[stage] &&
            result.stages[stage].pair_occurrence_count == 2 &&
            result.stages[stage].unique_owner_candidate_count == 2 &&
            result.stages[stage].edge_count == 2 &&
            result.stages[stage].uncertainty_count == 0,
        "logical stage aggregation or closure census is inconsistent");
  }
}

void tamper_smoke()
{
  Fixture fixture;
  fixture.edges[0][0].x2 = 1;
  Result result{};
  const int status =
      klayout_cuda_spatial_run_antenna_m1_m4_empty_v1(
          &fixture.request, &result);
  require(
      status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT,
      "tampered contour was not rejected fail-closed");
}

void malformed_hierarchy_smoke()
{
  Fixture fixture;
  fixture.request.hierarchy.source_cell_indices = nullptr;
  Result result{};
  const int status =
      klayout_cuda_spatial_run_antenna_m1_m4_empty_v1(
          &fixture.request, &result);
  require(
      status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT,
      "null hierarchy array was not rejected before digest traversal");
}

void runtime_capacity_smoke()
{
  Fixture fixture;
  fixture.request.capacity.max_memberships = 1;
  Result result{};
  const int status =
      klayout_cuda_spatial_run_antenna_m1_m4_empty_v1(
          &fixture.request, &result);
  require(
      status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
          result.fallback_flags !=
              KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE,
      "runtime capacity exhaustion did not retain CPU fallback");
}

void bounded_device_admission_timeout_smoke()
{
  Fixture fixture;
  fixture.request.capacity.max_device_bytes = 1;
  require(
      setenv(
          "KLAYOUT_CUDA_ANTENNA_DEVICE_WAIT_MS", "25", 1) == 0,
      "set bounded device-admission wait environment");
  const auto begin = std::chrono::steady_clock::now();
  Result result{};
  const int status =
      klayout_cuda_spatial_run_antenna_m1_m4_empty_v1(
          &fixture.request, &result);
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - begin);
  require(
      unsetenv("KLAYOUT_CUDA_ANTENNA_DEVICE_WAIT_MS") == 0,
      "clear bounded device-admission wait environment");
  require(
      status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
          result.fallback_flags ==
              KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY &&
          elapsed.count() >= 20 && elapsed.count() < 2000,
      "temporary device pressure did not wait boundedly then fail closed");
}

}  // namespace

int main()
{
  try {
    complete_smoke();
    tamper_smoke();
    malformed_hierarchy_smoke();
    runtime_capacity_smoke();
    bounded_device_admission_timeout_smoke();
    std::cout << "antenna M1-M4 backend smoke: PASS\n";
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "antenna M1-M4 backend smoke: FAIL: "
              << exception.what() << "\n";
    return 1;
  }
}
