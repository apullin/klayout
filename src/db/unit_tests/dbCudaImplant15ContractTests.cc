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
  klayout_cuda_spatial_implant15_request_v1 request;
  klayout_cuda_spatial_implant15_result_v1 result;

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
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_RESIDENT_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_QUALIFIED_OPTIONS;
    request.format_version = 1;
    request.dbu_per_micron = 2000;
    request.requested_mask =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES;
    request.device = 0;
    request.implant1_distance = 140;
    request.implant2_distance = 50;
    request.implant3_distance = 90;
    request.implant4_distance = 90;
    request.grid_cell_size = 2000;

    set_scene (
      request.nplus, KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_ROLE,
      4, 0, KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_DIGEST_DOMAIN, 0x4e);
    set_scene (
      request.pplus, KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_ROLE,
      5, 0, KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_DIGEST_DOMAIN, 0x50);
    set_scene (
      request.gate, KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_ROLE,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_LAYER,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_DATATYPE,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_DIGEST_DOMAIN, 0x47);
    set_scene (
      request.contact, KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_ROLE,
      10, 0, KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_DIGEST_DOMAIN, 0x43);

    request.capacity.struct_size = sizeof (request.capacity);
    request.capacity.max_slabs_per_rectangle = 8;
    request.capacity.max_cells_per_secondary_edge = 8;
    request.capacity.max_cells_per_boundary_edge = 8;
    request.capacity.max_contexts = 4;
    request.capacity.max_rectangles = 8;
    request.capacity.max_x_slabs = 16;
    request.capacity.max_union_memberships = 32;
    request.capacity.max_events = 64;
    request.capacity.max_raw_segments = 64;
    request.capacity.max_boundary_segments = 64;
    request.capacity.max_gate_edges = 8;
    request.capacity.max_contact_edges = 8;
    request.capacity.max_grid_cells = 16;
    request.capacity.max_secondary_memberships = 64;
    request.capacity.max_gate_boundary_cell_visits = 64;
    request.capacity.max_contact_boundary_cell_visits = 64;
    request.capacity.max_member_visits = 64;
    request.capacity.max_pair_work = 64;
    request.capacity.max_morphology_work = 64;
    request.capacity.max_overlap_work = 64;

    std::memset (&result, 0, sizeof (result));
    result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    result.struct_size = sizeof (result);
    result.status = KLAYOUT_CUDA_SPATIAL_OK;
    result.disposition = KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_HITS;
    result.opcode = request.opcode;
    result.option_flags = request.option_flags;
    result.format_version = request.format_version;
    result.dbu_per_micron = request.dbu_per_micron;
    result.requested_mask = request.requested_mask;
    result.device = request.device;
    result.implant1_distance = request.implant1_distance;
    result.implant2_distance = request.implant2_distance;
    result.implant3_distance = request.implant3_distance;
    result.implant4_distance = request.implant4_distance;
    result.grid_cell_size = request.grid_cell_size;
    set_echo (request.nplus, result.nplus);
    set_echo (request.pplus, result.pplus);
    set_echo (request.gate, result.gate);
    set_echo (request.contact, result.contact);
    result.capacity = request.capacity;

    //  IMPLANT.5 can terminate after raw NPLUS/PPLUS expansion, before the
    //  union and IMPLANT.1-.4 phases.  The expanded rectangle census remains
    //  meaningful even though every union counter is still zero.
    result.nplus_rectangle_count = 1;
    result.pplus_rectangle_count = 1;
    result.implant_rectangle_count = 2;
    result.implant5_membership_count = 1;
    result.implant5_candidate_count = 1;
    result.implant5_hit_count = 1;
    result.implant5_ns = 1;
    result.total_ns = 1;
  }

  void set_scene (
    klayout_cuda_spatial_implant15_scene_v1 &scene,
    uint32_t role, uint32_t layer, uint32_t datatype,
    const char *domain, uint8_t digest_byte)
  {
    std::memset (&scene, 0, sizeof (scene));
    scene.struct_size = sizeof (scene);
    scene.role = role;
    scene.format_version = 1;
    scene.dbu_per_micron = 2000;
    scene.root_cell = 0;
    scene.layer = layer;
    scene.datatype = datatype;
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
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DIGEST_DOMAIN_BYTES);
    std::memset (scene.scene_digest, digest_byte, sizeof (scene.scene_digest));
  }

  static void set_echo (
    const klayout_cuda_spatial_implant15_scene_v1 &scene,
    klayout_cuda_spatial_implant15_scene_echo_v1 &echo)
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
    echo.context_polygon_offset_count = scene.context_polygon_offset_count;
    echo.context_edge_offset_count = scene.context_edge_offset_count;
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
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DIGEST_DOMAIN_BYTES);
    std::memcpy (
      echo.scene_digest, scene.scene_digest, sizeof (echo.scene_digest));
  }
};

} // anonymous namespace

TEST(1_EarlyImplant5HitCarriesExpandedRectangleCensus)
{
  ContractFixture fixture;
  std::string error;
  EXPECT_EQ (
    db::cuda_spatial_validate_implant15_result (
      fixture.request, fixture.result, KLAYOUT_CUDA_SPATIAL_OK, &error),
    true);
  EXPECT_EQ (error, "");

  fixture.result.implant_rectangle_count = 0;
  EXPECT_EQ (
    db::cuda_spatial_validate_implant15_result (
      fixture.request, fixture.result, KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (error.empty (), false);

  fixture.result.implant_rectangle_count = 3;
  EXPECT_EQ (
    db::cuda_spatial_validate_implant15_result (
      fixture.request, fixture.result, KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (error.empty (), false);
}
