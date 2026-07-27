/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM4Evidence.h"
#include "dbCudaSpatialBackend.h"
#include "tlUnitTest.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace
{

const uint32_t physical_layers
  [KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT] = {
    9, 1, 4, 3, 10, 11, 12, 13, 14, 15, 16, 17
  };

const char *digest_domains
  [KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT] = {
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
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M4_DIGEST_DOMAIN
  };

struct ContractFixture
{
  uint64_t source_cell_index;
  klayout_cuda_spatial_m1_width_space_context_v1 context;
  uint32_t context_parent;
  klayout_cuda_spatial_antenna_m1_m4_cell_v1
    cells[KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT];
  klayout_cuda_spatial_m1_width_space_polygon_v1
    polygons[KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT];
  klayout_cuda_spatial_m1_width_space_edge_v1
    edges[KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT][4];
  klayout_cuda_spatial_antenna_m1_m4_request_v1 request;
  klayout_cuda_spatial_antenna_m1_m4_result_v1 result;

  ContractFixture ()
    : source_cell_index (42), context { 0, 0, 0, 0 },
      context_parent (0)
  {
    std::memset (cells, 0, sizeof (cells));
    std::memset (polygons, 0, sizeof (polygons));
    std::memset (edges, 0, sizeof (edges));
    std::memset (&request, 0, sizeof (request));
    std::memset (&result, 0, sizeof (result));

    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
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

    request.hierarchy.struct_size = sizeof (request.hierarchy);
    request.hierarchy.format_version = 2;
    request.hierarchy.dbu_per_micron = 2000;
    request.hierarchy.root_cell = 0;
    request.hierarchy.source_root_cell_index = source_cell_index;
    request.hierarchy.source_cell_indices = &source_cell_index;
    request.hierarchy.source_cell_count = 1;
    request.hierarchy.source_cell_index_record_bytes =
      sizeof (source_cell_index);
    request.hierarchy.contexts = &context;
    request.hierarchy.context_count = 1;
    request.hierarchy.context_record_bytes = sizeof (context);
    request.hierarchy.context_parent_ids = &context_parent;
    request.hierarchy.context_parent_count = 1;
    request.hierarchy.context_parent_record_bytes =
      sizeof (context_parent);
    std::memset (
      request.hierarchy.hierarchy_digest, 0x48,
      sizeof (request.hierarchy.hierarchy_digest));

    set_capacity ();
    uint64_t domain_stored_bytes = 0;
    uint64_t expanded_bytes = 0;
    for (size_t index = 0;
         index < KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
         ++index) {
      set_domain (index);
      domain_stored_bytes += request.domains [index].stored_bytes;
      expanded_bytes +=
        request.domains [index].expanded_geometry_bytes;
    }
    request.census.struct_size = sizeof (request.census);
    request.census.format_version = request.format_version;
    request.census.shared_cell_count = 1;
    request.census.shared_context_count = 1;
    request.census.context_parent_record_count = 1;
    request.census.stored_cell_record_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
    request.census.stored_polygon_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
    request.census.stored_edge_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT * 4;
    request.census.expanded_polygon_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
    request.census.expanded_edge_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT * 4;
    request.census.total_stored_bytes = domain_stored_bytes + 256;
    request.census.total_expanded_geometry_bytes = expanded_bytes;
    request.census.estimated_peak_bytes =
      request.census.total_stored_bytes + expanded_bytes;
    std::memset (
      request.lower_capture_digest, 0x4c,
      sizeof (request.lower_capture_digest));
    std::memset (
      request.capture_digest, 0x54,
      sizeof (request.capture_digest));

    set_complete_result ();
  }

  void set_capacity ()
  {
    request.capacity.struct_size = sizeof (request.capacity);
    request.capacity.max_cells = 64;
    request.capacity.max_contexts = 64;
    request.capacity.max_stored_polygons = 64;
    request.capacity.max_stored_edges = 256;
    request.capacity.max_flat_polygons = 64;
    request.capacity.max_flat_edges = 256;
    request.capacity.max_total_stored_bytes = 1024 * 1024;
    request.capacity.max_total_expanded_geometry_bytes = 1024 * 1024;
    request.capacity.max_estimated_peak_bytes = 2 * 1024 * 1024;
    request.capacity.max_nodes = 1024;
    request.capacity.max_rectangles = 1024;
    request.capacity.max_memberships = 4096;
    request.capacity.max_pair_occurrences = 4096;
    request.capacity.max_unique_candidates = 4096;
    request.capacity.max_cell_members = 128;
    request.capacity.max_dsu_iterations = 64;
    request.capacity.max_rule_work = 16384;
    request.capacity.max_device_bytes = 1024 * 1024;
  }

  void set_domain (size_t index)
  {
    const int64_t left = int64_t (index * 200);
    cells [index].polygon_count = 1;
    cells [index].edge_count = 4;
    polygons [index].left = left;
    polygons [index].bottom = 0;
    polygons [index].right = left + 100;
    polygons [index].top = 100;
    polygons [index].polygon_id = uint32_t (index);
    polygons [index].edge_count = 4;
    edges [index][0] = { left, 0, left, 100 };
    edges [index][1] = { left, 100, left + 100, 100 };
    edges [index][2] = { left + 100, 100, left + 100, 0 };
    edges [index][3] = { left + 100, 0, left, 0 };

    klayout_cuda_spatial_antenna_m1_m4_domain_v1 &domain =
      request.domains [index];
    domain.struct_size = sizeof (domain);
    domain.role = uint32_t (index);
    domain.physical_layer = physical_layers [index];
    domain.source_layer_index = uint32_t (100 + index);
    domain.cells = &cells [index];
    domain.cell_count = 1;
    domain.cell_record_bytes = sizeof (cells [index]);
    domain.polygons = &polygons [index];
    domain.polygon_count = 1;
    domain.polygon_record_bytes = sizeof (polygons [index]);
    domain.edges = edges [index];
    domain.edge_count = 4;
    domain.edge_record_bytes = sizeof (edges [index][0]);
    domain.nonempty_context_count = 1;
    domain.flat_polygon_count = 1;
    domain.flat_edge_count = 4;
    domain.stored_bytes =
      sizeof (cells [index]) + sizeof (polygons [index]) +
      sizeof (edges [index]);
    domain.expanded_geometry_bytes =
      sizeof (polygons [index]) + sizeof (uint32_t) +
      sizeof (edges [index]);
    domain.scene_left = left;
    domain.scene_bottom = 0;
    domain.scene_right = left + 100;
    domain.scene_top = 100;
    std::memcpy (
      domain.digest_domain, digest_domains [index],
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DIGEST_DOMAIN_BYTES);
    std::memset (
      domain.scene_digest, int (index + 1),
      sizeof (domain.scene_digest));
  }

  static void set_hierarchy_echo (
    const klayout_cuda_spatial_antenna_m1_m4_hierarchy_v1 &source,
    klayout_cuda_spatial_antenna_m1_m4_hierarchy_echo_v1 &echo)
  {
    echo.struct_size = sizeof (echo);
    echo.format_version = source.format_version;
    echo.dbu_per_micron = source.dbu_per_micron;
    echo.root_cell = source.root_cell;
    echo.source_root_cell_index = source.source_root_cell_index;
    echo.source_cell_count = source.source_cell_count;
    echo.source_cell_index_record_bytes =
      source.source_cell_index_record_bytes;
    echo.context_count = source.context_count;
    echo.context_record_bytes = source.context_record_bytes;
    echo.context_parent_count = source.context_parent_count;
    echo.context_parent_record_bytes =
      source.context_parent_record_bytes;
    std::memcpy (
      echo.hierarchy_digest, source.hierarchy_digest,
      sizeof (echo.hierarchy_digest));
  }

  static void set_domain_echo (
    const klayout_cuda_spatial_antenna_m1_m4_domain_v1 &source,
    klayout_cuda_spatial_antenna_m1_m4_domain_echo_v1 &echo)
  {
    echo.struct_size = sizeof (echo);
    echo.role = source.role;
    echo.physical_layer = source.physical_layer;
    echo.datatype = source.datatype;
    echo.source_layer_index = source.source_layer_index;
    echo.cell_count = source.cell_count;
    echo.cell_record_bytes = source.cell_record_bytes;
    echo.polygon_count = source.polygon_count;
    echo.polygon_record_bytes = source.polygon_record_bytes;
    echo.edge_count = source.edge_count;
    echo.edge_record_bytes = source.edge_record_bytes;
    echo.nonempty_context_count = source.nonempty_context_count;
    echo.flat_polygon_count = source.flat_polygon_count;
    echo.flat_edge_count = source.flat_edge_count;
    echo.stored_bytes = source.stored_bytes;
    echo.expanded_geometry_bytes = source.expanded_geometry_bytes;
    echo.scene_left = source.scene_left;
    echo.scene_bottom = source.scene_bottom;
    echo.scene_right = source.scene_right;
    echo.scene_top = source.scene_top;
    std::memcpy (
      echo.digest_domain, source.digest_domain,
      sizeof (echo.digest_domain));
    std::memcpy (
      echo.scene_digest, source.scene_digest,
      sizeof (echo.scene_digest));
  }

  void set_complete_result ()
  {
    result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    result.struct_size = sizeof (result);
    result.status = KLAYOUT_CUDA_SPATIAL_OK;
    result.disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_COMPLETE;
    result.opcode = request.opcode;
    result.option_flags = request.option_flags;
    result.format_version = request.format_version;
    result.dbu_per_micron = request.dbu_per_micron;
    result.requested_mask = request.requested_mask;
    result.certified_empty_mask = request.requested_mask;
    result.clean_mask = request.requested_mask;
    result.stage_count = request.stage_count;
    result.ratio_numerator = request.ratio_numerator;
    result.ratio_denominator = request.ratio_denominator;
    result.domain_count = request.domain_count;
    result.device = request.device;
    result.closed_domain_mask =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_DOMAINS;
    result.released_stage_mask = request.requested_mask;
    result.accounted_peak_device_bytes = 1024 * 1024;
    set_hierarchy_echo (request.hierarchy, result.hierarchy);
    for (size_t index = 0;
         index < KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
         ++index) {
      set_domain_echo (request.domains [index], result.domains [index]);
      result.domain_results [index].struct_size =
        sizeof (result.domain_results [index]);
      result.domain_results [index].role = uint32_t (index);
      result.domain_results [index].owner_count = 1;
      result.domain_results [index].rectangle_count = 1;
      result.domain_results [index].owner_range_count = 1;
      recompute_domain_digest (index);
    }
    result.census = request.census;
    result.capacity = request.capacity;
    std::memcpy (
      result.lower_capture_digest, request.lower_capture_digest,
      sizeof (result.lower_capture_digest));
    std::memcpy (
      result.capture_digest, request.capture_digest,
      sizeof (result.capture_digest));

    for (size_t index = 0;
         index < KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT;
         ++index) {
      klayout_cuda_spatial_antenna_m1_m4_stage_result_v1 &stage =
        result.stages [index];
      stage.struct_size = sizeof (stage);
      stage.stage = uint32_t (1u << index);
      stage.component_count = 3;
      stage.membership_count = 8;
      stage.occupied_cell_count = 4;
      stage.pair_occurrence_count = 6;
      stage.unique_owner_candidate_count = 3;
      stage.edge_count = 2;
      stage.gate_count = 1;
      //  Deliberately differs from connectivity candidates: one gate root
      //  reaches the conservative integer-ratio reduction.
      stage.evaluated_count = 1;
      stage.retained_rectangle_count =
        index + 1 == KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT ? 0 : 1;
      stage.released_rectangle_count =
        index + 1 == KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT ? 3 : 2;
      stage.dsu_iteration_count = 2;
      stage.work_count = 16;
      stage.stage_ns = 1;
      recompute_stage_digest (index);
    }
    result.setup_ns = 1;
    result.h2d_ns = 1;
    result.d2h_ns = 1;
    result.total_ns = 10;
  }

  void recompute_domain_digest (size_t index)
  {
    db::cuda_antenna_m1_m4_evidence::Digest digest;
    (void) db::cuda_antenna_m1_m4_evidence::domain_digest (
      request, index, result.domain_results [index], digest);
    std::memcpy (
      result.domain_results [index].rectangle_digest,
      digest.data (), digest.size ());
  }

  void recompute_stage_digest (size_t index)
  {
    db::cuda_antenna_m1_m4_evidence::Digest digest;
    (void) db::cuda_antenna_m1_m4_evidence::stage_digest (
      request, result, index, digest);
    std::memcpy (
      result.stages [index].stage_digest,
      digest.data (), digest.size ());
  }

  bool validate (std::string &error) const
  {
    return db::cuda_spatial_validate_antenna_m1_m4_result (
      request, result, KLAYOUT_CUDA_SPATIAL_OK, &error);
  }
};

} // anonymous namespace

TEST(1_CompleteConservativeCertificateAcceptsDistinctCandidateCounts)
{
  ContractFixture fixture;
  std::string error;
  EXPECT_EQ (fixture.validate (error), true);
  EXPECT_EQ (error, "");
  EXPECT_NE (
    fixture.result.stages [0].unique_owner_candidate_count,
    fixture.result.stages [0].evaluated_count);
}

TEST(2_AllIdentitiesCountsCapsAndDigestsAreBound)
{
  std::string error;
  {
    ContractFixture fixture;
    ++fixture.result.ratio_numerator;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    ++fixture.result.hierarchy.context_count;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    ++fixture.result.domains [7].polygon_count;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.domains [11].scene_digest [3] ^= 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    ++fixture.result.census.expanded_edge_count;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    ++fixture.result.capacity.max_pair_occurrences;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.capture_digest [0] ^= 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.domain_results [6].rectangle_digest [9] ^= 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    ++fixture.result.domain_results
      [KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ACTIVE_ROLE].rectangle_count;
    fixture.recompute_domain_digest (
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ACTIVE_ROLE);
    //  Even a valid updated domain digest invalidates every stale stage
    //  digest that binds the lower-domain evidence.
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    ++fixture.result.stages [2].work_count;
    //  The mutation remains within every numerical bound.  Its stale evidence
    //  digest alone must reject the result.
    EXPECT_EQ (fixture.validate (error), false);
    fixture.recompute_stage_digest (2);
    EXPECT_EQ (fixture.validate (error), true);
  }
}

TEST(3_OnlyFullZeroResultIsConsumable)
{
  std::string error;
  {
    ContractFixture fixture;
    fixture.result.certified_empty_mask &= ~1u;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.clean_mask &= ~2u;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_CAPACITY;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.device_flags = 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.accounted_peak_device_bytes =
      fixture.request.capacity.max_device_bytes + 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [2].hit_count = 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [1].uncertainty_count = 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_HITS;
    EXPECT_EQ (fixture.validate (error), false);
  }
}

TEST(4_StageAndFrontierEvidenceIsBound)
{
  std::string error;
  {
    ContractFixture fixture;
    fixture.result.closed_domain_mask &= ~1u;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.released_stage_mask &= ~4u;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [2].stage =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_M2;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    std::memset (
      fixture.result.stages [0].stage_digest, 0,
      sizeof (fixture.result.stages [0].stage_digest));
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    std::memcpy (
      fixture.result.stages [3].stage_digest,
      fixture.result.stages [1].stage_digest,
      sizeof (fixture.result.stages [3].stage_digest));
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [0].evaluated_count =
      fixture.result.stages [0].gate_count + 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [0].edge_count =
      fixture.result.stages [0].unique_owner_candidate_count + 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [0].retained_rectangle_count = 10;
    fixture.result.stages [0].released_rectangle_count = 3;
    fixture.recompute_stage_digest (0);
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [0].retained_rectangle_count = 0;
    fixture.result.stages [0].released_rectangle_count = 3;
    fixture.recompute_stage_digest (0);
    //  The total frontier is exact, but M1 itself was not retained for the
    //  VIA1 bridge, so the later staged proof must be rejected.
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [1].membership_count = 0;
    fixture.result.stages [1].occupied_cell_count = 0;
    fixture.recompute_stage_digest (1);
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [1].occupied_cell_count = 0;
    fixture.recompute_stage_digest (1);
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.result.stages [0].component_count = 4;
    fixture.recompute_stage_digest (0);
    EXPECT_EQ (fixture.validate (error), false);
  }
}

TEST(5_MalformedCompactRequestFailsClosed)
{
  std::string error;
  {
    ContractFixture fixture;
    fixture.request.hierarchy.contexts = 0;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.domains [4].role = 5;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.domains [8].source_layer_index =
      fixture.request.domains [7].source_layer_index;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.capacity.max_nodes = 0;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.census.total_expanded_geometry_bytes =
      fixture.request.capacity.max_total_expanded_geometry_bytes + 1;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.capacity.max_nodes = UINT64_C (1) << 32;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.capacity.max_rectangles = UINT64_C (1) << 32;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.capacity.max_cell_members = UINT64_C (1) << 32;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.capacity.max_dsu_iterations = UINT64_C (1) << 32;
    EXPECT_EQ (fixture.validate (error), false);
  }
  {
    ContractFixture fixture;
    fixture.request.capacity.max_nodes =
      fixture.request.census.expanded_polygon_count - 1;
    fixture.result.capacity.max_nodes =
      fixture.request.capacity.max_nodes;
    EXPECT_EQ (fixture.validate (error), false);
  }
}
