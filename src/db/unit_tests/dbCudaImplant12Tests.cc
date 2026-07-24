/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaImplant12.h"
#include "dbCudaImplant12Digest.h"

#include "dbCell.h"
#include "dbDeepShapeStore.h"
#include "dbLayerProperties.h"
#include "dbLayout.h"
#include "dbObjectWithProperties.h"
#include "dbPolygon.h"
#include "dbRegion.h"
#include "tlUnitTest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace
{

struct SceneFixture
{
  db::DeepShapeStore store;
  db::DeepLayer implant;
  db::DeepLayer gate;
  db::DeepLayer contact;

  explicit SceneFixture (double dbu = 0.0005)
    : store ("TOP", dbu)
  {
    db::Region seed;
    seed.insert (db::Box (0, 0, 100, 100));
    implant = store.create_from_flat (seed, false);
    gate = implant.derived ();
    contact = implant.derived ();
    db::Cell &top = implant.initial_cell ();
    top.shapes (gate.layer ()).insert (db::Box (500, 0, 600, 100));
    top.shapes (contact.layer ()).insert (db::Box (700, 0, 800, 100));
    implant.layout ().set_properties (
      contact.layer (), db::LayerProperties (10, 0));
  }
};

db::CudaImplant12BuildSpec qualified_spec ()
{
  db::CudaImplant12BuildSpec spec;
  spec.implant_is_exact_merged = true;
  spec.gate_is_raw = true;
  spec.contact_is_raw = true;
  return spec;
}

} // anonymous namespace

TEST(1_DigestBindsProfileCensusAndRecords)
{
  klayout_cuda_spatial_implant12_context_v1 context =
    { 0, 0, 0, 0 };
  uint32_t implant_context = 0;
  uint64_t implant_offsets [2] = { 0, 4 };
  uint32_t gate_context = 0;
  uint32_t contact_context = 0;

  klayout_cuda_spatial_implant12_cell_v1 cell;
  std::memset (&cell, 0, sizeof (cell));
  cell.source_cell_index = 11;
  for (uint32_t domain = 0;
       domain < KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT; ++domain) {
    cell.domains [domain].contour_begin = domain;
    cell.domains [domain].edge_begin = uint64_t (domain) * 4;
    cell.domains [domain].polygon_count = 1;
    cell.domains [domain].contour_count = 1;
    cell.domains [domain].edge_count = 4;
  }

  klayout_cuda_spatial_implant12_contour_v1 contours [3];
  klayout_cuda_spatial_implant12_edge_v1 edges [12];
  std::memset (contours, 0, sizeof (contours));
  std::memset (edges, 0, sizeof (edges));
  for (uint32_t domain = 0; domain < 3; ++domain) {
    contours [domain].edge_begin = uint64_t (domain) * 4;
    contours [domain].edge_count = 4;
    contours [domain].flags = KLAYOUT_CUDA_SPATIAL_IMPLANT12_HULL;
    const int64_t x = int64_t (domain) * 100;
    edges [domain * 4 + 0] =
      klayout_cuda_spatial_implant12_edge_v1 { x, 0, x, 10 };
    edges [domain * 4 + 1] =
      klayout_cuda_spatial_implant12_edge_v1 { x, 10, x + 10, 10 };
    edges [domain * 4 + 2] =
      klayout_cuda_spatial_implant12_edge_v1 { x + 10, 10, x + 10, 0 };
    edges [domain * 4 + 3] =
      klayout_cuda_spatial_implant12_edge_v1 { x + 10, 0, x, 0 };
  }

  klayout_cuda_spatial_implant12_request_v1 request;
  std::memset (&request, 0, sizeof (request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof (request);
  request.opcode = KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_SUPERSET_EMPTY;
  request.option_flags = KLAYOUT_CUDA_SPATIAL_IMPLANT12_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.requested_mask = KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES;
  request.implant1_distance = 140;
  request.implant2_distance = 50;
  request.grid_cell_size = 2000;
  request.contexts = &context;
  request.context_count = 1;
  request.context_record_bytes = sizeof (context);
  request.implant_contexts = &implant_context;
  request.implant_context_count = 1;
  request.implant_edge_offsets = implant_offsets;
  request.implant_edge_offset_count = 2;
  request.gate_contexts = &gate_context;
  request.gate_context_count = 1;
  request.contact_contexts = &contact_context;
  request.contact_context_count = 1;
  request.cells = &cell;
  request.cell_count = 1;
  request.cell_record_bytes = sizeof (cell);
  request.contours = contours;
  request.contour_count = 3;
  request.contour_record_bytes = sizeof (contours [0]);
  request.edges = edges;
  request.edge_count = 12;
  request.edge_record_bytes = sizeof (edges [0]);
  request.flat_implant_polygon_count = 1;
  request.flat_gate_polygon_count = 1;
  request.flat_contact_polygon_count = 1;
  request.flat_implant_contour_count = 1;
  request.flat_gate_contour_count = 1;
  request.flat_contact_contour_count = 1;
  request.flat_implant_edge_count = 4;
  request.flat_gate_edge_count = 4;
  request.flat_contact_edge_count = 4;
  request.implant_right = 10;
  request.implant_top = 10;
  request.max_contexts = 10;
  request.max_grid_cells = 100;
  request.max_implant_memberships = 1000;
  request.max_gate_query_visits = 1000;
  request.max_gate_candidate_work = 1000;
  request.max_contact_query_visits = 1000;
  request.max_contact_candidate_work = 1000;
  request.max_flat_polygons = 1000;
  request.max_flat_contours = 1000;
  request.max_flat_edges = 10000;

  std::array<uint8_t, 32> first, repeated, changed;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, first), true);
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, repeated), true);
  EXPECT_EQ (first == repeated, true);

  context.tx = 1;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  context.tx = 0;
  implant_context = 1;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  implant_context = 0;
  ++implant_offsets [1];
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  --implant_offsets [1];
  gate_context = 1;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  gate_context = 0;
  contact_context = 1;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  contact_context = 0;
  ++cell.source_cell_index;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  --cell.source_cell_index;
  ++contours [0].polygon_id;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  --contours [0].polygon_id;
  edges [11].x2 = 1;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  edges [11].x2 = 0;

  ++request.max_contact_candidate_work;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  --request.max_contact_candidate_work;

#define EXPECT_IMPLANT12_SCALAR_BOUND(field) \
  do { \
    const auto saved = request.field; \
    ++request.field; \
    EXPECT_EQ ( \
      db::cuda_implant12_digest::request_digest (request, changed), true); \
    EXPECT_EQ (first == changed, false); \
    request.field = saved; \
  } while (false)
  EXPECT_IMPLANT12_SCALAR_BOUND (abi_version);
  EXPECT_IMPLANT12_SCALAR_BOUND (struct_size);
  EXPECT_IMPLANT12_SCALAR_BOUND (opcode);
  EXPECT_IMPLANT12_SCALAR_BOUND (option_flags);
  EXPECT_IMPLANT12_SCALAR_BOUND (format_version);
  EXPECT_IMPLANT12_SCALAR_BOUND (dbu_per_micron);
  EXPECT_IMPLANT12_SCALAR_BOUND (root_cell);
  EXPECT_IMPLANT12_SCALAR_BOUND (requested_mask);
  EXPECT_IMPLANT12_SCALAR_BOUND (device);
  EXPECT_IMPLANT12_SCALAR_BOUND (implant1_distance);
  EXPECT_IMPLANT12_SCALAR_BOUND (implant2_distance);
  EXPECT_IMPLANT12_SCALAR_BOUND (grid_cell_size);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_implant_polygon_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_gate_polygon_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_contact_polygon_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_implant_contour_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_gate_contour_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_contact_contour_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_implant_edge_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_gate_edge_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (flat_contact_edge_count);
  EXPECT_IMPLANT12_SCALAR_BOUND (implant_left);
  EXPECT_IMPLANT12_SCALAR_BOUND (implant_bottom);
  EXPECT_IMPLANT12_SCALAR_BOUND (implant_right);
  EXPECT_IMPLANT12_SCALAR_BOUND (implant_top);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_contexts);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_grid_cells);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_implant_memberships);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_gate_query_visits);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_gate_candidate_work);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_contact_query_visits);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_flat_polygons);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_flat_contours);
  EXPECT_IMPLANT12_SCALAR_BOUND (max_flat_edges);
#undef EXPECT_IMPLANT12_SCALAR_BOUND

  --request.context_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.context_count;
  --request.implant_context_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.implant_context_count;
  --request.implant_edge_offset_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.implant_edge_offset_count;
  --request.gate_context_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.gate_context_count;
  --request.contact_context_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.contact_context_count;
  --request.cell_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.cell_count;
  --request.contour_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.contour_count;
  --request.edge_count;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), true);
  EXPECT_EQ (first == changed, false);
  ++request.edge_count;

  ++request.edge_record_bytes;
  EXPECT_EQ (
    db::cuda_implant12_digest::request_digest (request, changed), false);
}

TEST(2_QualifiedSceneAndMirroredHierarchy)
{
  SceneFixture fixture;
  db::Layout &layout = fixture.implant.layout ();
  db::Cell &top = fixture.implant.initial_cell ();
  db::Cell &child = layout.cell (layout.add_cell ("CHILD"));
  child.shapes (fixture.implant.layer ()).insert (
    db::Box (0, 0, 20, 10));
  child.shapes (fixture.gate.layer ()).insert (
    db::Box (100, 0, 120, 10));
  child.shapes (fixture.contact.layer ()).insert (
    db::Box (200, 0, 220, 10));
  top.insert (
    db::CellInstArray (
      db::CellInst (child.cell_index ()),
      db::Trans (0, true, db::Vector (1000, 2000))));

  db::CudaImplant12SceneLimits limits;
  db::CudaImplant12Scene scene;
  std::string reason;
  EXPECT_EQ (
    db::cuda_implant12_build_scene (
      fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
      limits, scene, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (scene.cells.size (), size_t (2));
  EXPECT_EQ (scene.contexts.size (), size_t (2));
  EXPECT_EQ (scene.contexts [1].transform_code >= 4, true);
  EXPECT_EQ (scene.implant_edge_offsets.size (), size_t (3));
  EXPECT_EQ (scene.implant_edge_offsets [0], uint64_t (0));
  EXPECT_EQ (scene.implant_edge_offsets [1], uint64_t (4));
  EXPECT_EQ (scene.implant_edge_offsets [2], uint64_t (8));
  for (uint32_t domain = 0;
       domain < KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT; ++domain) {
    EXPECT_EQ (scene.flat_polygons [domain], uint64_t (2));
    EXPECT_EQ (scene.flat_contours [domain], uint64_t (2));
    EXPECT_EQ (scene.flat_edges [domain], uint64_t (8));
  }

  //  Local contours are clockwise.  Reflected world expansion must reverse
  //  endpoints in the backend; the context code above makes that contract
  //  visible to its smoke test.
  for (std::vector<klayout_cuda_spatial_implant12_contour_v1>::const_iterator
         contour = scene.contours.begin ();
       contour != scene.contours.end (); ++contour) {
    EXPECT_EQ (
      contour->flags, uint32_t (KLAYOUT_CUDA_SPATIAL_IMPLANT12_HULL));
    int64_t twice_area = 0;
    for (uint32_t local = 0; local < contour->edge_count; ++local) {
      const klayout_cuda_spatial_implant12_edge_v1 &edge =
        scene.edges [size_t (contour->edge_begin + local)];
      twice_area +=
        __int128 (edge.x1) * edge.y2 - __int128 (edge.x2) * edge.y1;
    }
    EXPECT_EQ (twice_area < 0, true);
  }

  limits.max_contexts = 1;
  scene.root_cell = 99;
  EXPECT_EQ (
    db::cuda_implant12_build_scene (
      fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
      limits, scene, &reason),
    false);
  EXPECT_EQ (scene.root_cell, uint32_t (99));
  EXPECT_EQ (reason.empty (), false);
}

TEST(3_FailClosedGeometryStateAndProvenance)
{
  {
    SceneFixture fixture;
    db::CudaImplant12BuildSpec spec = qualified_spec ();
    spec.gate_is_raw = false;
    db::CudaImplant12Scene scene;
    scene.root_cell = 17;
    std::string reason;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        fixture.implant, fixture.gate, fixture.contact, spec,
        db::CudaImplant12SceneLimits (), scene, &reason),
      false);
    EXPECT_EQ (scene.root_cell, uint32_t (17));
  }
  {
    SceneFixture fixture;
    fixture.implant.layout ().set_properties (
      fixture.implant.layer (), db::LayerProperties (4, 0));
    db::CudaImplant12Scene scene;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
        db::CudaImplant12SceneLimits (), scene, 0),
      false);
  }
  {
    SceneFixture fixture (0.001);
    db::CudaImplant12Scene scene;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
        db::CudaImplant12SceneLimits (), scene, 0),
      false);
  }
  {
    SceneFixture first;
    SceneFixture second;
    db::CudaImplant12Scene scene;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        first.implant, first.gate, second.contact, qualified_spec (),
        db::CudaImplant12SceneLimits (), scene, 0),
      false);
  }
}

TEST(4_FailClosedUnsupportedPolygonsAndCapacity)
{
  {
    SceneFixture fixture;
    db::Point hull [] = {
      db::Point (1000, 0), db::Point (1050, 50), db::Point (1100, 0)
    };
    db::Polygon diagonal;
    diagonal.assign_hull (hull, hull + 3);
    fixture.implant.initial_cell ().shapes (fixture.gate.layer ()).insert (
      diagonal);
    db::CudaImplant12Scene scene;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
        db::CudaImplant12SceneLimits (), scene, 0),
      false);
  }
  {
    SceneFixture fixture;
    db::Polygon with_hole (db::Box (1000, 0, 1200, 200));
    db::Point hole [] = {
      db::Point (1050, 50), db::Point (1050, 100),
      db::Point (1100, 100), db::Point (1100, 50)
    };
    with_hole.insert_hole (hole, hole + 4);
    fixture.implant.initial_cell ().shapes (fixture.gate.layer ()).insert (
      with_hole);
    db::CudaImplant12Scene scene;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
        db::CudaImplant12SceneLimits (), scene, 0),
      false);
  }
  {
    SceneFixture fixture;
    fixture.implant.initial_cell ().shapes (fixture.gate.layer ()).insert (
      db::object_with_properties<db::Box> (
        db::Box (1000, 0, 1100, 100), 1));
    db::CudaImplant12Scene scene;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
        db::CudaImplant12SceneLimits (), scene, 0),
      false);
  }
  {
    SceneFixture fixture;
    db::CudaImplant12SceneLimits limits;
    limits.max_flat_edges = 11;
    db::CudaImplant12Scene scene;
    scene.root_cell = 23;
    std::string reason;
    EXPECT_EQ (
      db::cuda_implant12_build_scene (
        fixture.implant, fixture.gate, fixture.contact, qualified_spec (),
        limits, scene, &reason),
      false);
    EXPECT_EQ (scene.root_cell, uint32_t (23));
    EXPECT_EQ (reason.empty (), false);
  }
}
