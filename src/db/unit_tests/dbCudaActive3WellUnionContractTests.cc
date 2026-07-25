/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaSpatialBackend.h"
#include "tlUnitTest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace
{

struct ContractFixture
{
  klayout_cuda_spatial_m1_width_space_context_v1 context;
  uint32_t layer_context;
  uint64_t polygon_offset;
  uint64_t edge_offset;
  klayout_cuda_spatial_m1_width_space_cell_v1 cell;
  klayout_cuda_spatial_m1_width_space_polygon_v1 polygon;
  klayout_cuda_spatial_m1_width_space_edge_v1 edges [4];
  klayout_cuda_spatial_active3_well_union_request_v1 request;
  klayout_cuda_spatial_active3_well_union_result_v1 result;

  ContractFixture ()
    : context { 0, 0, 0, 0 }, layer_context (0), polygon_offset (0),
      edge_offset (0), cell { 0, 0, 0, 1, 4 },
      polygon { 0, 0, 0, 100, 100, 0, 4 },
      edges {
        { 0, 0, 0, 100 },
        { 0, 100, 100, 100 },
        { 100, 100, 100, 0 },
        { 100, 0, 0, 0 }
      }
  {
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode = KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_QUALIFIED_OPTIONS;
    request.format_version = 1;
    request.dbu_per_micron = 2000;
    request.device = 0;
    request.distance = 110;
    request.grid_cell_size = 2000;
    request.secondary_well_layer = 2;
    request.secondary_well_datatype = 0;
    set_scene (
      request.wells,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_ROLE, 3,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_DIGEST_DOMAIN,
      0x39);
    set_scene (
      request.active,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_ROLE, 1,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_DIGEST_DOMAIN,
      0x61);

    request.max_contexts = 4;
    request.max_rectangles = 8;
    request.max_x_slabs = 16;
    request.max_union_memberships = 32;
    request.max_events = 64;
    request.max_raw_segments = 64;
    request.max_boundary_segments = 64;
    request.max_slabs_per_rectangle = 8;
    request.max_active_edges = 8;
    request.max_grid_cells = 16;
    request.max_active_memberships = 64;
    request.max_active_cell_visits = 64;
    request.max_member_visits = 64;
    request.max_pair_work = 64;
    request.max_cells_per_active_edge = 8;
    request.max_cells_per_well_edge = 8;

    std::memset (&result, 0, sizeof (result));
    result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    result.struct_size = sizeof (result);
    result.status = KLAYOUT_CUDA_SPATIAL_OK;
    result.disposition =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_COMPLETE;
    result.opcode = request.opcode;
    result.option_flags = request.option_flags;
    result.format_version = request.format_version;
    result.dbu_per_micron = request.dbu_per_micron;
    result.device = request.device;
    result.distance = request.distance;
    result.grid_cell_size = request.grid_cell_size;
    result.secondary_well_layer = request.secondary_well_layer;
    result.secondary_well_datatype =
      request.secondary_well_datatype;
    set_echo (request.wells, result.wells);
    set_echo (request.active, result.active);

    result.rectangle_count = 1;
    result.x_slab_count = 2;
    result.union_membership_count = 1;
    result.event_count = 2;
    result.strip_interval_count = 1;
    result.raw_segment_count = 4;
    result.boundary_segment_count = 4;
    result.active_expanded_edge_count = 4;
    result.grid_cell_count = 1;
    result.active_membership_count = 4;
    result.active_cell_visit_count = 1;
    result.member_visit_count = 4;
    result.device_total_bytes = 1000;
    result.union_free_begin_bytes = 900;
    result.union_free_low_bytes = 800;
    result.callback_free_begin_bytes = 900;
    result.callback_free_low_bytes = 700;
    result.post_scan_free_bytes = 800;
    result.callback_incremental_peak_bytes = 200;
    result.setup_ns = 5;
    result.wells_h2d_ns = 5;
    result.wells_expand_ns = 10;
    result.x_membership_ns = 10;
    result.strip_scan_ns = 10;
    result.boundary_ns = 10;
    result.active_h2d_ns = 5;
    result.active_expand_ns = 10;
    result.active_preflight_ns = 10;
    result.grid_count_ns = 5;
    result.grid_build_ns = 10;
    result.query_ns = 10;
    result.d2h_ns = 5;
    result.total_ns = 100;
  }

  void set_scene (
    klayout_cuda_spatial_active3_well_union_scene_v1 &scene,
    uint32_t role, uint32_t layer, const char *domain,
    uint8_t digest_byte)
  {
    std::memset (&scene, 0, sizeof (scene));
    scene.struct_size = sizeof (scene);
    scene.role = role;
    scene.format_version = 1;
    scene.dbu_per_micron = 2000;
    scene.root_cell = 0;
    scene.layer = layer;
    scene.datatype = 0;
    scene.contexts = &context;
    scene.context_count = 1;
    scene.context_record_bytes = sizeof (context);
    scene.layer_contexts = &layer_context;
    scene.layer_context_count = 1;
    scene.context_polygon_offsets = &polygon_offset;
    scene.context_polygon_offset_count = 1;
    scene.context_edge_offsets = &edge_offset;
    scene.context_edge_offset_count = 1;
    scene.cells = &cell;
    scene.cell_count = 1;
    scene.cell_record_bytes = sizeof (cell);
    scene.polygons = &polygon;
    scene.polygon_count = 1;
    scene.polygon_record_bytes = sizeof (polygon);
    scene.edges = edges;
    scene.edge_count = 4;
    scene.edge_record_bytes = sizeof (edges [0]);
    scene.flat_polygon_count = 1;
    scene.flat_edge_count = 4;
    scene.scene_left = 0;
    scene.scene_bottom = 0;
    scene.scene_right = 100;
    scene.scene_top = 100;
    std::memcpy (
      scene.digest_domain, domain,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_DIGEST_DOMAIN_BYTES);
    std::memset (
      scene.scene_digest, digest_byte, sizeof (scene.scene_digest));
  }

  static void set_echo (
    const klayout_cuda_spatial_active3_well_union_scene_v1 &scene,
    klayout_cuda_spatial_active3_well_union_scene_echo_v1 &echo)
  {
    std::memset (&echo, 0, sizeof (echo));
    echo.struct_size = sizeof (echo);
    echo.role = scene.role;
    echo.format_version = scene.format_version;
    echo.dbu_per_micron = scene.dbu_per_micron;
    echo.root_cell = scene.root_cell;
    echo.layer = scene.layer;
    echo.datatype = scene.datatype;
    echo.context_count = scene.context_count;
    echo.layer_context_count = scene.layer_context_count;
    echo.context_polygon_offset_count =
      scene.context_polygon_offset_count;
    echo.context_edge_offset_count =
      scene.context_edge_offset_count;
    echo.cell_count = scene.cell_count;
    echo.polygon_count = scene.polygon_count;
    echo.edge_count = scene.edge_count;
    echo.flat_polygon_count = scene.flat_polygon_count;
    echo.flat_edge_count = scene.flat_edge_count;
    echo.scene_left = scene.scene_left;
    echo.scene_bottom = scene.scene_bottom;
    echo.scene_right = scene.scene_right;
    echo.scene_top = scene.scene_top;
    std::memcpy (
      echo.digest_domain, scene.digest_domain,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_DIGEST_DOMAIN_BYTES);
    std::memcpy (
      echo.scene_digest, scene.scene_digest,
      sizeof (echo.scene_digest));
  }
};

static_assert (
  sizeof (klayout_cuda_spatial_active3_well_union_request_v1) == 776,
  "ACTIVE.3 WELL-union request ABI size changed");
static_assert (
  sizeof (klayout_cuda_spatial_active3_well_union_result_v1) == 960,
  "ACTIVE.3 WELL-union result ABI size changed");
static_assert (
  offsetof (
    klayout_cuda_spatial_active3_well_union_request_v1, wells) == 64,
  "ACTIVE.3 WELL descriptor offset changed");
static_assert (
  offsetof (
    klayout_cuda_spatial_active3_well_union_request_v1, active) == 344,
  "ACTIVE.3 ACTIVE descriptor offset changed");

} // anonymous namespace

TEST(1_QualifiedCompleteProof)
{
  ContractFixture fixture;
  std::string error;
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      fixture.request, fixture.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    true);
  EXPECT_EQ (error, "");
}

TEST(2_SecondaryLayerAndDomainsFailClosed)
{
  std::string error;
  ContractFixture bad_layer;
  bad_layer.request.secondary_well_layer = 4;
  bad_layer.result.secondary_well_layer = 4;
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      bad_layer.request, bad_layer.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error, "host supplied an unqualified ACTIVE.3 WELL-union request");

  ContractFixture bad_domain;
  std::memcpy (
    bad_domain.request.wells.digest_domain,
    KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_DIGEST_DOMAIN_BYTES);
  ContractFixture::set_echo (
    bad_domain.request.wells, bad_domain.result.wells);
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      bad_domain.request, bad_domain.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error, "host supplied an unqualified ACTIVE.3 WELL-union request");
}

TEST(3_EchoStatusAndCapacityFailures)
{
  std::string error;
  ContractFixture bad_echo;
  ++bad_echo.result.active.scene_digest [0];
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      bad_echo.request, bad_echo.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA ACTIVE.3 WELL-union backend returned a mismatched proof echo");

  ContractFixture bad_status;
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      bad_status.request, bad_status.result,
      KLAYOUT_CUDA_SPATIAL_FALLBACK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA ACTIVE.3 WELL-union backend returned inconsistent statuses");

  ContractFixture bad_capacity;
  bad_capacity.result.active_membership_count =
    bad_capacity.request.max_active_memberships + 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      bad_capacity.request, bad_capacity.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA ACTIVE.3 WELL-union backend returned impossible proof counters");

  ContractFixture impossible_visits;
  impossible_visits.result.active_cell_visit_count = 0;
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      impossible_visits.request, impossible_visits.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA ACTIVE.3 WELL-union backend returned impossible proof counters");
}

TEST(4_OnlyConsistentDispositionIsAccepted)
{
  std::string error;
  ContractFixture inconsistent;
  inconsistent.result.raw_hit_count = 1;
  inconsistent.result.candidate_pair_count = 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      inconsistent.request, inconsistent.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA ACTIVE.3 WELL-union backend returned an inconsistent disposition");

  ContractFixture raw_hit;
  raw_hit.result.disposition =
    KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_RAW_HITS;
  raw_hit.result.raw_hit_count = 1;
  raw_hit.result.candidate_pair_count = 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_active3_well_union_result (
      raw_hit.request, raw_hit.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    true);
}
