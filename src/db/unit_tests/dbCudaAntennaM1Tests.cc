/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM1.h"
#include "dbCudaAntennaM1Oracle.h"

#include "dbCell.h"
#include "dbDeepShapeStore.h"
#include "dbLayerProperties.h"
#include "dbLayout.h"
#include "dbObjectWithProperties.h"
#include "dbPolygon.h"
#include "dbRegion.h"
#include "tlUnitTest.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{

db::DeepLayer make_seed_layer (db::DeepShapeStore &store)
{
  db::Region seed;
  seed.insert (db::Box (0, 0, 100, 80));
  return store.create_from_flat (seed, false);
}

void mark_physical (
  db::DeepLayer &layer, int physical_layer, int datatype = 0)
{
  layer.layout ().set_properties (
    layer.layer (), db::LayerProperties (physical_layer, datatype));
}

struct AntennaFixture
{
  db::DeepShapeStore store;
  db::DeepLayer poly;
  db::DeepLayer active;
  db::DeepLayer nplus;
  db::DeepLayer nwell;
  db::DeepLayer contact;
  db::DeepLayer metal1;

  explicit AntennaFixture (double dbu = 0.0005)
    : store ("TOP", dbu),
      poly (make_seed_layer (store)),
      active (poly.derived ()),
      nplus (poly.derived ()),
      nwell (poly.derived ()),
      contact (poly.derived ()),
      metal1 (poly.derived ())
  {
    mark_physical (poly, 9);
    mark_physical (active, 1);
    mark_physical (nplus, 4);
    mark_physical (nwell, 3);
    mark_physical (contact, 10);
    mark_physical (metal1, 11);

    active.initial_cell ().shapes (active.layer ()).insert (
      db::Box (200, 0, 320, 80));
    nplus.initial_cell ().shapes (nplus.layer ()).insert (
      db::Box (400, 0, 530, 80));
    nwell.initial_cell ().shapes (nwell.layer ()).insert (
      db::Box (600, 0, 740, 80));
    contact.initial_cell ().shapes (contact.layer ()).insert (
      db::Box (800, 0, 950, 80));
    metal1.initial_cell ().shapes (metal1.layer ()).insert (
      db::Box (1000, 0, 1160, 80));
  }

  std::array<db::DeepLayer *, db::CudaAntennaM1DomainCount> layers ()
  {
    return {{
      &poly, &active, &nplus, &nwell, &contact, &metal1
    }};
  }

  void add_box_to_all (
    db::Cell &cell, db::Coord x, db::Coord y,
    db::Coord width, db::Coord height)
  {
    const std::array<
      db::DeepLayer *, db::CudaAntennaM1DomainCount> all = layers ();
    for (size_t domain = 0;
         domain < db::CudaAntennaM1DomainCount; ++domain) {
      cell.shapes (all [domain]->layer ()).insert (
        db::Box (x, y, x + width, y + height));
    }
  }

  void add_nested_eight_transform_hierarchy ()
  {
    db::Layout &layout = poly.layout ();
    db::Cell &top = poly.initial_cell ();
    db::Cell &child =
      layout.cell (layout.add_cell ("ANTENNA_CHILD"));
    db::Cell &grandchild =
      layout.cell (layout.add_cell ("ANTENNA_GRANDCHILD"));
    add_box_to_all (child, -30, -20, 40, 20);
    add_box_to_all (grandchild, 10, 15, 15, 25);
    child.insert (
      db::CellInstArray (
        db::CellInst (grandchild.cell_index ()),
        db::Trans (db::Vector (25, -40))));

    for (unsigned int code = 0; code < 8; ++code) {
      top.insert (
        db::CellInstArray (
          db::CellInst (child.cell_index ()),
          db::Trans (
            int (code & 3), code >= 4,
            db::Vector (
              db::Coord (2000 * int (code) - 7000),
              db::Coord (3000 - 250 * int (code))))));
    }
  }

  bool build (
    const db::CudaAntennaM1CaptureLimits &limits,
    db::CudaAntennaM1Capture &capture,
    std::string *reason = 0)
  {
    return db::cuda_antenna_m1_build_capture (
      poly, active, nplus, nwell, contact, metal1,
      limits, capture, reason);
  }
};

bool same_context (
  const db::CudaM1WidthSpaceContext &first,
  const db::CudaM1WidthSpaceContext &second)
{
  return
    first.tx == second.tx && first.ty == second.ty &&
    first.cell_id == second.cell_id &&
    first.transform_code == second.transform_code;
}

bool same_cell (
  const db::CudaM1WidthSpaceCell &first,
  const db::CudaM1WidthSpaceCell &second)
{
  return
    first.source_cell_index == second.source_cell_index &&
    first.polygon_begin == second.polygon_begin &&
    first.edge_begin == second.edge_begin &&
    first.polygon_count == second.polygon_count &&
    first.edge_count == second.edge_count;
}

bool same_polygon (
  const db::CudaM1WidthSpacePolygon &first,
  const db::CudaM1WidthSpacePolygon &second)
{
  return
    first.edge_begin == second.edge_begin &&
    first.left == second.left && first.bottom == second.bottom &&
    first.right == second.right && first.top == second.top &&
    first.polygon_id == second.polygon_id &&
    first.edge_count == second.edge_count;
}

bool same_edge (
  const db::CudaM1WidthSpaceEdge &first,
  const db::CudaM1WidthSpaceEdge &second)
{
  return
    first.x1 == second.x1 && first.y1 == second.y1 &&
    first.x2 == second.x2 && first.y2 == second.y2;
}

bool same_raw_scene (
  const db::CudaRawManhattanScene &first,
  const db::CudaRawManhattanScene &second)
{
  if (first.format_version != second.format_version ||
      first.dbu_per_micron != second.dbu_per_micron ||
      first.root_cell != second.root_cell ||
      first.reserved != second.reserved ||
      first.flat_polygon_count != second.flat_polygon_count ||
      first.flat_edge_count != second.flat_edge_count ||
      first.scene_left != second.scene_left ||
      first.scene_bottom != second.scene_bottom ||
      first.scene_right != second.scene_right ||
      first.scene_top != second.scene_top ||
      first.contexts.size () != second.contexts.size () ||
      first.metal_contexts != second.metal_contexts ||
      first.context_polygon_offsets != second.context_polygon_offsets ||
      first.context_edge_offsets != second.context_edge_offsets ||
      first.cells.size () != second.cells.size () ||
      first.polygons.size () != second.polygons.size () ||
      first.edges.size () != second.edges.size () ||
      first.digest != second.digest) {
    return false;
  }
  for (size_t index = 0; index < first.contexts.size (); ++index) {
    if (! same_context (first.contexts [index], second.contexts [index])) {
      return false;
    }
  }
  for (size_t index = 0; index < first.cells.size (); ++index) {
    if (! same_cell (first.cells [index], second.cells [index])) {
      return false;
    }
  }
  for (size_t index = 0; index < first.polygons.size (); ++index) {
    if (! same_polygon (first.polygons [index], second.polygons [index])) {
      return false;
    }
  }
  for (size_t index = 0; index < first.edges.size (); ++index) {
    if (! same_edge (first.edges [index], second.edges [index])) {
      return false;
    }
  }
  return true;
}

bool build_legacy_domain (
  AntennaFixture &fixture, size_t domain,
  db::CudaRawManhattanScene &scene, std::string *reason)
{
  const db::CudaM1WidthSpaceSceneLimits limits;
  switch (domain) {
  case db::CudaAntennaM1Poly:
    return db::cuda_poly_raw_manhattan_build_scene (
      fixture.poly, limits, scene, reason);
  case db::CudaAntennaM1Active:
    return db::cuda_active_raw_manhattan_build_scene (
      fixture.active, limits, scene, reason);
  case db::CudaAntennaM1Nplus:
    return db::cuda_nplus_raw_manhattan_build_scene (
      fixture.nplus, limits, scene, reason);
  case db::CudaAntennaM1Nwell:
    return db::cuda_nwell_raw_manhattan_build_scene (
      fixture.nwell, limits, scene, reason);
  case db::CudaAntennaM1Contact:
    return db::cuda_contact_raw_manhattan_build_scene (
      fixture.contact, limits, scene, reason);
  case db::CudaAntennaM1Metal1:
    return db::cuda_m1_raw_manhattan_build_scene (
      fixture.metal1, limits, scene, reason);
  default:
    return false;
  }
}

} // anonymous namespace

TEST(1_CaptureBindsSixDomainsHierarchyParentsCountsAndMemory)
{
  AntennaFixture fixture;
  fixture.add_nested_eight_transform_hierarchy ();
  db::CudaAntennaM1Capture capture;
  std::string reason;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (capture.format_version, uint32_t (2));
  EXPECT_EQ (capture.dbu_per_micron, uint32_t (2000));
  EXPECT_EQ (capture.source_cell_indices.size (), size_t (3));
  EXPECT_EQ (capture.contexts.size (), size_t (17));
  EXPECT_EQ (capture.context_parent_ids.size (), size_t (17));
  EXPECT_EQ (
    capture.context_parent_ids [0],
    std::numeric_limits<uint32_t>::max ());
  for (size_t context = 1; context <= 8; ++context) {
    EXPECT_EQ (capture.context_parent_ids [context], uint32_t (0));
  }
  for (size_t context = 9; context <= 16; ++context) {
    EXPECT_EQ (
      capture.context_parent_ids [context],
      uint32_t (context - 8));
  }

  db::CudaAntennaM1Census census;
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_census (
      capture, census, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (census.shared_cell_count, uint64_t (3));
  EXPECT_EQ (census.shared_context_count, uint64_t (17));
  EXPECT_EQ (census.context_parent_record_count, uint64_t (17));
  EXPECT_EQ (
    census.context_parent_bytes,
    uint64_t (17 * sizeof (uint32_t)));
  EXPECT_EQ (census.stored_cell_records, uint64_t (18));
  EXPECT_EQ (census.stored_context_records, uint64_t (17));
  EXPECT_EQ (census.legacy_stored_cell_records, uint64_t (18));
  EXPECT_EQ (census.legacy_stored_context_records, uint64_t (102));
  EXPECT_EQ (census.nonempty_context_records, uint64_t (102));
  EXPECT_EQ (census.stored_polygon_count, uint64_t (18));
  EXPECT_EQ (census.stored_edge_count, uint64_t (72));
  EXPECT_EQ (census.expanded_polygon_count, uint64_t (102));
  EXPECT_EQ (census.expanded_edge_count, uint64_t (408));
  EXPECT_EQ (
    census.estimated_peak_bytes,
    census.total_stored_bytes +
      census.total_expanded_geometry_bytes);
  EXPECT_EQ (
    census.legacy_estimated_peak_bytes,
    census.legacy_total_stored_bytes +
      census.total_expanded_geometry_bytes);
  EXPECT_EQ (
    census.total_stored_bytes < census.legacy_total_stored_bytes,
    true);
  const uint64_t shared_bytes =
    uint64_t (4 * sizeof (uint32_t) + sizeof (uint64_t) +
              db::CudaAntennaM1DomainCount * sizeof (uint32_t) + 64) +
    uint64_t (3 * sizeof (uint64_t)) +
    uint64_t (17 * sizeof (db::CudaM1WidthSpaceContext)) +
    uint64_t (17 * sizeof (uint32_t));
  const uint64_t domain_bytes =
    uint64_t (2 * sizeof (uint64_t) + 4 * sizeof (int64_t) + 32) +
    uint64_t (3 * sizeof (db::CudaAntennaM1DomainCell)) +
    uint64_t (3 * sizeof (db::CudaM1WidthSpacePolygon)) +
    uint64_t (12 * sizeof (db::CudaM1WidthSpaceEdge));
  EXPECT_EQ (
    census.total_stored_bytes,
    shared_bytes + db::CudaAntennaM1DomainCount * domain_bytes);
  EXPECT_EQ (census.capture_digest == capture.digest, true);
  EXPECT_EQ (
    census.hierarchy_digest == capture.hierarchy_digest, true);

  const uint32_t physical_layers [db::CudaAntennaM1DomainCount] =
    { 9, 1, 4, 3, 10, 11 };
  for (size_t domain = 0;
       domain < db::CudaAntennaM1DomainCount; ++domain) {
    const db::CudaAntennaM1DomainCensus &record =
      census.domains [domain];
    EXPECT_EQ (record.role, uint32_t (domain));
    EXPECT_EQ (record.physical_layer, physical_layers [domain]);
    EXPECT_EQ (record.datatype, uint32_t (0));
    EXPECT_EQ (record.stored_cell_count, uint64_t (3));
    EXPECT_EQ (record.stored_context_count, uint64_t (17));
    EXPECT_EQ (record.nonempty_context_count, uint64_t (17));
    EXPECT_EQ (record.stored_polygon_count, uint64_t (3));
    EXPECT_EQ (record.stored_edge_count, uint64_t (12));
    EXPECT_EQ (record.expanded_polygon_count, uint64_t (17));
    EXPECT_EQ (record.expanded_edge_count, uint64_t (68));
    EXPECT_EQ (
      record.scene_digest == capture.domains [domain].digest, true);
    EXPECT_EQ (record.stored_bytes < record.legacy_stored_bytes, true);

    db::CudaRawManhattanScene legacy;
    EXPECT_EQ (
      build_legacy_domain (fixture, domain, legacy, &reason), true);
    EXPECT_EQ (reason, "");
    db::CudaRawManhattanScene materialized;
    EXPECT_EQ (
      db::cuda_antenna_m1_materialize_domain_scene (
        capture, db::CudaAntennaM1Domain (domain),
        materialized, &reason),
      true);
    EXPECT_EQ (reason, "");
    EXPECT_EQ (same_raw_scene (legacy, materialized), true);
  }

  std::array<uint8_t, 32> recomputed;
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_digest (capture, recomputed),
    true);
  EXPECT_EQ (recomputed == capture.digest, true);

  const std::string text = db::cuda_antenna_m1_census_text (census);
  EXPECT_EQ (
    text.find ("antenna_m1_capture format=2") != std::string::npos,
    true);
  EXPECT_EQ (
    text.find ("shared_contexts=17") != std::string::npos, true);
  EXPECT_EQ (text.find ("POLY{physical=9/0") != std::string::npos, true);
  EXPECT_EQ (text.find ("M1{physical=11/0") != std::string::npos, true);
  EXPECT_EQ (
    text.find ("capture_sha256=") != std::string::npos, true);

  db::CudaAntennaM1Capture repeated;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), repeated, 0),
    true);
  EXPECT_EQ (repeated.digest == capture.digest, true);
  EXPECT_EQ (
    repeated.hierarchy_digest == capture.hierarchy_digest, true);

  fixture.active.initial_cell ().shapes (
    fixture.active.layer ()).insert (
      db::Box (5000, 0, 5100, 100));
  db::CudaAntennaM1Capture geometry_changed;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), geometry_changed, 0),
    true);
  EXPECT_EQ (
    geometry_changed.hierarchy_digest == capture.hierarchy_digest,
    true);
  EXPECT_EQ (geometry_changed.digest == capture.digest, false);
  EXPECT_EQ (
    geometry_changed.domains [db::CudaAntennaM1Active].digest ==
      capture.domains [db::CudaAntennaM1Active].digest,
    false);
}

TEST(2_CaptureFailsClosedOnIdentityRoleEmptyAndCapacity)
{
  AntennaFixture fixture;
  fixture.add_nested_eight_transform_hierarchy ();
  db::CudaAntennaM1Capture baseline;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), baseline, 0),
    true);
  db::CudaAntennaM1Census census;
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_census (
      baseline, census, 0),
    true);

  db::CudaAntennaM1Capture sentinel;
  sentinel.source_root_cell_index = 777;
  std::string reason;

  EXPECT_EQ (
    db::cuda_antenna_m1_build_capture (
      fixture.poly, fixture.active, fixture.active, fixture.nwell,
      fixture.contact, fixture.metal1,
      db::CudaAntennaM1CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (
    reason.find ("distinct") != std::string::npos, true);

  mark_physical (fixture.nplus, 99);
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (reason.find ("NPLUS") != std::string::npos, true);
  mark_physical (fixture.nplus, 4);

  db::DeepShapeStore other_store ("TOP", 0.0005);
  db::DeepLayer other_nwell = make_seed_layer (other_store);
  mark_physical (other_nwell, 3);
  EXPECT_EQ (
    db::cuda_antenna_m1_build_capture (
      fixture.poly, fixture.active, fixture.nplus, other_nwell,
      fixture.contact, fixture.metal1,
      db::CudaAntennaM1CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (
    reason.find ("one hierarchy") != std::string::npos, true);

  db::CudaAntennaM1CaptureLimits stored_limit;
  stored_limit.max_total_stored_bytes =
    census.total_stored_bytes - 1;
  EXPECT_EQ (
    fixture.build (stored_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (
    reason.find ("stored bytes") != std::string::npos, true);

  db::CudaAntennaM1CaptureLimits expanded_limit;
  expanded_limit.max_total_expanded_geometry_bytes =
    census.total_expanded_geometry_bytes - 1;
  EXPECT_EQ (
    fixture.build (expanded_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (
    reason.find ("expanded geometry bytes") != std::string::npos,
    true);

  db::CudaAntennaM1CaptureLimits scene_limit;
  scene_limit.scene.max_flat_edges = 67;
  EXPECT_EQ (
    fixture.build (scene_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (
    reason.find ("configured capacity") != std::string::npos, true);

  AntennaFixture empty;
  empty.active.initial_cell ().shapes (
    empty.active.layer ()).clear ();
  EXPECT_EQ (
    empty.build (
      db::CudaAntennaM1CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (reason.find ("ACTIVE") != std::string::npos, true);

  AntennaFixture wrong_dbu (0.001);
  EXPECT_EQ (
    wrong_dbu.build (
      db::CudaAntennaM1CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.source_root_cell_index, uint64_t (777));
  EXPECT_EQ (reason.find ("DBU") != std::string::npos, true);
}

TEST(3_CaptureRejectsUnsupportedGeometryPropertiesAndTransforms)
{
  {
    AntennaFixture fixture;
    db::Point diagonal_hull [] = {
      db::Point (2000, 0), db::Point (2100, 100),
      db::Point (2200, 0)
    };
    db::Polygon diagonal;
    diagonal.assign_hull (
      diagonal_hull, diagonal_hull + 3);
    fixture.nplus.initial_cell ().shapes (
      fixture.nplus.layer ()).insert (diagonal);

    db::CudaAntennaM1Capture capture;
    capture.source_root_cell_index = 91;
    std::string reason;
    EXPECT_EQ (
      fixture.build (
        db::CudaAntennaM1CaptureLimits (), capture, &reason),
      false);
    EXPECT_EQ (capture.source_root_cell_index, uint64_t (91));
    EXPECT_EQ (
      reason.find ("non-Manhattan") != std::string::npos, true);
  }

  {
    AntennaFixture fixture;
    fixture.active.initial_cell ().shapes (
      fixture.active.layer ()).insert (
        db::object_with_properties<db::Box> (
          db::Box (2000, 0, 2100, 100), 1));
    db::CudaAntennaM1Capture capture;
    std::string reason;
    EXPECT_EQ (
      fixture.build (
        db::CudaAntennaM1CaptureLimits (), capture, &reason),
      false);
    EXPECT_EQ (
      reason.find ("properties") != std::string::npos, true);
  }

  {
    AntennaFixture fixture;
    db::Cell &complex_child =
      fixture.poly.layout ().cell (
        fixture.poly.layout ().add_cell ("COMPLEX_CHILD"));
    fixture.add_box_to_all (complex_child, 0, 0, 20, 20);
    fixture.poly.initial_cell ().insert (
      db::CellInstArray (
        db::CellInst (complex_child.cell_index ()),
        db::ICplxTrans (
          1.0, 45.0, false, db::Vector (1000, 1000))));
    db::CudaAntennaM1Capture capture;
    std::string reason;
    EXPECT_EQ (
      fixture.build (
        db::CudaAntennaM1CaptureLimits (), capture, &reason),
      false);
    EXPECT_EQ (
      reason.find ("complex transform") != std::string::npos, true);
  }
}

TEST(4_CaptureDigestAndParentSidecarFailClosedOnDrift)
{
  AntennaFixture fixture;
  fixture.add_nested_eight_transform_hierarchy ();
  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);

  std::array<uint8_t, 32> sentinel;
  sentinel.fill (UINT8_C (0xa5));
  std::array<uint8_t, 32> digest = sentinel;

  db::CudaAntennaM1Capture geometry = capture;
  ++geometry.domains [db::CudaAntennaM1Contact].edges [0].x2;
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_digest (geometry, digest), false);
  EXPECT_EQ (digest == sentinel, true);
  db::CudaRawManhattanScene materialize_sentinel;
  materialize_sentinel.root_cell = 77;
  std::string materialize_reason;
  EXPECT_EQ (
    db::cuda_antenna_m1_materialize_domain_scene (
      geometry, db::CudaAntennaM1Contact,
      materialize_sentinel, &materialize_reason),
    false);
  EXPECT_EQ (materialize_sentinel.root_cell, uint32_t (77));

  db::CudaAntennaM1Capture parent = capture;
  parent.context_parent_ids [9] = 9;
  digest = sentinel;
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_digest (parent, digest), false);
  EXPECT_EQ (digest == sentinel, true);

  db::CudaAntennaM1Capture layer = capture;
  layer.source_layer_indices [db::CudaAntennaM1Active] =
    layer.source_layer_indices [db::CudaAntennaM1Poly];
  digest = sentinel;
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_digest (layer, digest), false);
  EXPECT_EQ (digest == sentinel, true);

  db::CudaAntennaM1Capture stale = capture;
  stale.digest [0] ^= UINT8_C (1);
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_digest (stale, digest), true);
  EXPECT_EQ (digest == capture.digest, true);
  db::CudaAntennaM1Census census;
  census.shared_cell_count = 777;
  std::string reason;
  EXPECT_EQ (
    db::cuda_antenna_m1_capture_census (
      stale, census, &reason),
    false);
  EXPECT_EQ (census.shared_cell_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("aggregate capture digest") != std::string::npos,
    true);

  std::vector<uint32_t> rebuilt;
  db::CudaRawManhattanScene materialized_poly;
  EXPECT_EQ (
    db::cuda_antenna_m1_materialize_domain_scene (
      capture, db::CudaAntennaM1Poly, materialized_poly, &reason),
    true);
  EXPECT_EQ (
    db::cuda_raw_manhattan_context_parents (
      fixture.poly,
      materialized_poly,
      db::CudaM1WidthSpaceSceneLimits (), rebuilt, &reason),
    true);
  EXPECT_EQ (rebuilt == capture.context_parent_ids, true);

  db::CudaM1WidthSpaceSceneLimits tight;
  tight.max_contexts = 16;
  rebuilt.clear ();
  rebuilt.push_back (77);
  EXPECT_EQ (
    db::cuda_raw_manhattan_context_parents (
      fixture.poly,
      materialized_poly,
      tight, rebuilt, &reason),
    false);
  EXPECT_EQ (rebuilt.size (), size_t (1));
  EXPECT_EQ (rebuilt [0], uint32_t (77));
}

TEST(5_OracleCanonicalizesExactFlatTouchGraph)
{
  AntennaFixture fixture;
  db::Cell &top = fixture.poly.initial_cell ();

  top.shapes (fixture.poly.layer ()).insert (
    db::Box (2000, 0, 2100, 100));
  top.shapes (fixture.poly.layer ()).insert (
    db::Box (2100, 100, 2200, 200));
  top.shapes (fixture.poly.layer ()).insert (
    db::Box (3000, 0, 3100, 100));
  top.shapes (fixture.contact.layer ()).insert (
    db::Box (3100, 0, 3200, 100));
  top.shapes (fixture.metal1.layer ()).insert (
    db::Box (3200, 100, 3300, 200));

  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);

  db::CudaAntennaM1Oracle oracle;
  std::string reason;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (),
      oracle, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (oracle.context_count, uint64_t (1));
  EXPECT_EQ (oracle.graph_node_count, uint64_t (8));
  EXPECT_EQ (oracle.component_count, uint64_t (5));
  EXPECT_EQ (oracle.canonical_root_count, uint64_t (5));
  EXPECT_EQ (oracle.top_context_component_count, uint64_t (5));
  EXPECT_EQ (oracle.graph_candidate_pair_count, uint64_t (20));
  EXPECT_EQ (oracle.graph_edge_count, uint64_t (3));
  EXPECT_EQ (oracle.annotation_candidate_pair_count, uint64_t (4));
  EXPECT_EQ (oracle.gate_component_count, uint64_t (0));
  EXPECT_EQ (oracle.diode_exempt_component_count, uint64_t (0));
  EXPECT_EQ (oracle.gate_area_dbu2, uint64_t (0));
  EXPECT_EQ (oracle.metal1_area_dbu2, uint64_t (22800));

  const uint64_t labels [] = { 0, 1, 1, 3, 4, 3, 6, 3 };
  EXPECT_EQ (
    oracle.canonical_labels ==
      std::vector<uint64_t> (labels, labels + 8),
    true);
  EXPECT_EQ (
    oracle.domain_component_counts [db::CudaAntennaM1Poly],
    uint64_t (3));
  EXPECT_EQ (
    oracle.domain_component_counts [db::CudaAntennaM1Contact],
    uint64_t (2));
  EXPECT_EQ (
    oracle.domain_component_counts [db::CudaAntennaM1Metal1],
    uint64_t (2));
  EXPECT_EQ (oracle.relation_edge_counts [0], uint64_t (1));
  EXPECT_EQ (oracle.relation_edge_counts [1], uint64_t (0));
  EXPECT_EQ (oracle.relation_edge_counts [2], uint64_t (0));
  EXPECT_EQ (oracle.relation_edge_counts [3], uint64_t (1));
  EXPECT_EQ (oracle.relation_edge_counts [4], uint64_t (1));

  db::CudaAntennaM1Oracle repeated;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (),
      repeated, 0),
    true);
  EXPECT_EQ (
    repeated.canonical_labels == oracle.canonical_labels, true);
  EXPECT_EQ (
    repeated.node_identity_digest == oracle.node_identity_digest,
    true);
  EXPECT_EQ (
    repeated.partition_digest == oracle.partition_digest, true);
  EXPECT_EQ (
    repeated.connectivity_digest == oracle.connectivity_digest,
    true);
  EXPECT_EQ (
    repeated.annotation_digest == oracle.annotation_digest, true);
  EXPECT_EQ (repeated.oracle_digest == oracle.oracle_digest, true);
  EXPECT_EQ (
    db::cuda_antenna_m1_oracle_text (repeated) ==
      db::cuda_antenna_m1_oracle_text (oracle),
    true);
  const std::string oracle_record =
    db::cuda_antenna_m1_oracle_text (oracle);
  EXPECT_EQ (
    oracle_record.find (
      "partition_sha256="
      "564616099e6c478e3938fb1588760f3cffc03151c947dff3486c64c1a21d4526") !=
      std::string::npos,
    true);
  EXPECT_EQ (
    oracle_record.find (
      "connectivity_sha256="
      "7381e5d2740a219901650daaee8970470c502628d818bbf1e49c7fe16fbafb53") !=
      std::string::npos,
    true);
  EXPECT_EQ (
    oracle_record.find (
      "oracle_sha256="
      "87a81358eb9526d414f7c8bbaec2c3dd69001df2e463fb16ef252748cda64174") !=
      std::string::npos,
    true);

  //  Annotation-only capture changes must not perturb the graph identity or
  //  connectivity certificate consumed by a future GPU DSU comparison.
  fixture.active.initial_cell ().shapes (
    fixture.active.layer ()).insert (
      db::Box (5000, 0, 5100, 100));
  db::CudaAntennaM1Capture annotation_changed_capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (),
      annotation_changed_capture, 0),
    true);
  db::CudaAntennaM1Oracle annotation_changed;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      annotation_changed_capture,
      db::CudaAntennaM1OracleLimits (),
      annotation_changed, 0),
    true);
  EXPECT_EQ (
    annotation_changed.node_identity_digest ==
      oracle.node_identity_digest,
    true);
  EXPECT_EQ (
    annotation_changed.connectivity_digest ==
      oracle.connectivity_digest,
    true);
  EXPECT_EQ (
    annotation_changed.partition_digest ==
      oracle.partition_digest,
    true);
  EXPECT_EQ (
    annotation_changed.annotation_digest ==
      oracle.annotation_digest,
    false);
  EXPECT_EQ (
    annotation_changed.oracle_digest == oracle.oracle_digest,
    false);
}

TEST(6_OracleAnnotatesGateAreaAndFactorZeroDiodeBoundary)
{
  AntennaFixture fixture;
  db::Cell &top = fixture.poly.initial_cell ();

  top.shapes (fixture.poly.layer ()).insert (
    db::Box (2000, 0, 2100, 100));
  top.shapes (fixture.contact.layer ()).insert (
    db::Box (2100, 0, 2200, 100));
  top.shapes (fixture.metal1.layer ()).insert (
    db::Box (2200, 0, 2300, 100));
  top.shapes (fixture.active.layer ()).insert (
    db::Box (2050, 0, 2150, 100));
  top.shapes (fixture.nplus.layer ()).insert (
    db::Box (2050, 0, 2150, 100));
  top.shapes (fixture.nwell.layer ()).insert (
    db::Box (2050, 0, 2100, 100));

  //  Gate only touches at x=3100, so its positive-area annotation is empty.
  top.shapes (fixture.poly.layer ()).insert (
    db::Box (3000, 0, 3100, 100));
  top.shapes (fixture.active.layer ()).insert (
    db::Box (3100, 0, 3200, 100));

  //  NPLUS only touches ACTIVE at x=4000, so this is not a diode.
  top.shapes (fixture.active.layer ()).insert (
    db::Box (3900, 0, 4000, 100));
  top.shapes (fixture.nplus.layer ()).insert (
    db::Box (4000, 0, 4100, 100));
  top.shapes (fixture.contact.layer ()).insert (
    db::Box (4000, 0, 4100, 100));

  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);
  db::CudaAntennaM1Oracle oracle;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (), oracle, 0),
    true);

  EXPECT_EQ (oracle.graph_node_count, uint64_t (8));
  EXPECT_EQ (oracle.component_count, uint64_t (6));
  EXPECT_EQ (oracle.graph_candidate_pair_count, uint64_t (22));
  EXPECT_EQ (oracle.graph_edge_count, uint64_t (2));
  EXPECT_EQ (oracle.annotation_candidate_pair_count, uint64_t (10));
  EXPECT_EQ (oracle.gate_component_count, uint64_t (1));
  EXPECT_EQ (oracle.diode_exempt_component_count, uint64_t (1));
  EXPECT_EQ (oracle.gate_area_dbu2, uint64_t (5000));
  EXPECT_EQ (oracle.metal1_area_dbu2, uint64_t (22800));

  const uint64_t labels [] = { 0, 1, 2, 3, 1, 5, 6, 1 };
  EXPECT_EQ (
    oracle.canonical_labels ==
      std::vector<uint64_t> (labels, labels + 8),
    true);
}

TEST(7_OraclePreservesParentChildAndSiblingOccurrenceIdentity)
{
  AntennaFixture fixture;
  db::Layout &layout = fixture.poly.layout ();
  db::Cell &top = fixture.poly.initial_cell ();

  db::Cell &child_contact =
    layout.cell (layout.add_cell ("ORACLE_CHILD_CONTACT"));
  child_contact.shapes (fixture.contact.layer ()).insert (
    db::Box (0, 0, 100, 100));
  child_contact.shapes (fixture.metal1.layer ()).insert (
    db::Box (0, 0, 100, 100));
  top.shapes (fixture.poly.layer ()).insert (
    db::Box (2000, 0, 2100, 100));
  top.insert (
    db::CellInstArray (
      db::CellInst (child_contact.cell_index ()),
      db::Trans (db::Vector (2100, 0))));

  db::Cell &child_poly =
    layout.cell (layout.add_cell ("ORACLE_CHILD_POLY"));
  child_poly.shapes (fixture.poly.layer ()).insert (
    db::Box (0, 0, 100, 100));
  db::Cell &sibling_contact =
    layout.cell (layout.add_cell ("ORACLE_SIBLING_CONTACT"));
  sibling_contact.shapes (fixture.contact.layer ()).insert (
    db::Box (0, 0, 100, 100));
  sibling_contact.shapes (fixture.metal1.layer ()).insert (
    db::Box (0, 0, 100, 100));
  top.insert (
    db::CellInstArray (
      db::CellInst (child_poly.cell_index ()),
      db::Trans (db::Vector (3000, 0))));
  top.insert (
    db::CellInstArray (
      db::CellInst (sibling_contact.cell_index ()),
      db::Trans (db::Vector (3100, 0))));

  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);
  db::CudaAntennaM1Oracle oracle;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (), oracle, 0),
    true);

  EXPECT_EQ (oracle.context_count, uint64_t (4));
  EXPECT_EQ (oracle.graph_node_count, uint64_t (9));
  EXPECT_EQ (oracle.component_count, uint64_t (5));
  EXPECT_EQ (oracle.top_context_component_count, uint64_t (4));
  EXPECT_EQ (oracle.graph_candidate_pair_count, uint64_t (27));
  EXPECT_EQ (oracle.graph_edge_count, uint64_t (4));
  EXPECT_EQ (oracle.annotation_candidate_pair_count, uint64_t (9));
  EXPECT_EQ (oracle.metal1_area_dbu2, uint64_t (32800));

  const uint64_t labels [] = { 0, 1, 2, 3, 1, 2, 6, 1, 2 };
  EXPECT_EQ (
    oracle.canonical_labels ==
      std::vector<uint64_t> (labels, labels + 9),
    true);
}

TEST(8_OracleBindsTransformsRegularArraysAndCoincidentContexts)
{
  AntennaFixture fixture;
  fixture.add_nested_eight_transform_hierarchy ();
  db::Layout &layout = fixture.poly.layout ();
  db::Cell &top = fixture.poly.initial_cell ();
  db::Cell &array_child =
    layout.cell (layout.add_cell ("ORACLE_ARRAY_CHILD"));
  fixture.add_box_to_all (array_child, 0, 0, 20, 10);
  top.insert (
    db::CellInstArray (
      db::CellInst (array_child.cell_index ()),
      db::Trans (1, true, db::Vector (30000, 20000)),
      db::Vector (100, 0), db::Vector (), 2, 1));

  //  Identical transforms remain two distinct contexts and occurrence nodes,
  //  even though exact geometry later joins their components.
  top.insert (
    db::CellInstArray (
      db::CellInst (array_child.cell_index ()),
      db::Trans (db::Vector (40000, 20000))));
  top.insert (
    db::CellInstArray (
      db::CellInst (array_child.cell_index ()),
      db::Trans (db::Vector (40000, 20000))));

  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);
  db::CudaAntennaM1Oracle oracle;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (), oracle, 0),
    true);

  EXPECT_EQ (oracle.context_count, uint64_t (21));
  EXPECT_EQ (oracle.graph_node_count, uint64_t (63));
  EXPECT_EQ (oracle.component_count, uint64_t (22));
  EXPECT_EQ (oracle.top_context_component_count, uint64_t (3));
  EXPECT_EQ (
    oracle.domain_node_counts [db::CudaAntennaM1Poly],
    uint64_t (21));
  EXPECT_EQ (
    oracle.domain_node_counts [db::CudaAntennaM1Contact],
    uint64_t (21));
  EXPECT_EQ (
    oracle.domain_node_counts [db::CudaAntennaM1Metal1],
    uint64_t (21));

  db::CudaAntennaM1Oracle repeated;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (), repeated, 0),
    true);
  EXPECT_EQ (repeated.oracle_digest == oracle.oracle_digest, true);
  EXPECT_EQ (
    repeated.canonical_labels == oracle.canonical_labels, true);
}

TEST(9_OracleLimitsAndCaptureDriftLeaveOutputUntouched)
{
  AntennaFixture fixture;
  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);

  db::CudaAntennaM1Oracle sentinel;
  sentinel.component_count = 777;
  sentinel.canonical_labels.push_back (77);
  std::string reason;

  db::CudaAntennaM1OracleLimits graph_limit;
  graph_limit.max_graph_nodes = 2;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, graph_limit, sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.component_count, uint64_t (777));
  EXPECT_EQ (sentinel.canonical_labels.size (), size_t (1));
  EXPECT_EQ (
    reason.find ("graph nodes") != std::string::npos, true);

  db::CudaAntennaM1OracleLimits total_limit;
  total_limit.max_total_flat_polygons = 5;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, total_limit, sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.component_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("total flat polygons") != std::string::npos, true);

  db::CudaAntennaM1OracleLimits edge_limit;
  edge_limit.max_total_flat_edges = 1;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, edge_limit, sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.component_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("total expanded edges") != std::string::npos,
    true);

  db::CudaAntennaM1OracleLimits graph_pair_limit;
  graph_pair_limit.max_candidate_pairs = 1;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, graph_pair_limit, sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.component_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("graph candidate-pair") != std::string::npos,
    true);

  db::CudaAntennaM1OracleLimits total_pair_limit;
  total_pair_limit.max_candidate_pairs = 2;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, total_pair_limit, sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.component_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("total candidate-pair") != std::string::npos,
    true);

  db::CudaAntennaM1Capture stale = capture;
  stale.digest [0] ^= UINT8_C (1);
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      stale, db::CudaAntennaM1OracleLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.component_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("capture is invalid") != std::string::npos, true);

#if defined(HAVE_64BIT_COORD)
  AntennaFixture wide;
  wide.metal1.initial_cell ().shapes (
    wide.metal1.layer ()).insert (
      db::Box (
        0, INT64_C (10000000000),
        INT64_C (5000000000), INT64_C (15000000000)));
  db::CudaAntennaM1Capture wide_capture;
  EXPECT_EQ (
    wide.build (
      db::CudaAntennaM1CaptureLimits (), wide_capture, 0),
    true);
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      wide_capture, db::CudaAntennaM1OracleLimits (),
      sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.component_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("M1 area exceeds uint64") != std::string::npos,
    true);
#endif
}

TEST(10_OracleFactorZeroDiodeAnnotatesButDoesNotUnionConductorRoots)
{
  AntennaFixture fixture;
  db::Cell &top = fixture.poly.initial_cell ();

  top.shapes (fixture.poly.layer ()).insert (
    db::Box (1900, 0, 2000, 100));
  top.shapes (fixture.contact.layer ()).insert (
    db::Box (2000, 0, 2100, 100));
  top.shapes (fixture.metal1.layer ()).insert (
    db::Box (2000, 100, 2100, 200));

  top.shapes (fixture.contact.layer ()).insert (
    db::Box (2400, 0, 2500, 100));
  top.shapes (fixture.poly.layer ()).insert (
    db::Box (2500, 0, 2600, 100));
  top.shapes (fixture.metal1.layer ()).insert (
    db::Box (2400, 100, 2500, 200));

  //  One factor-zero diode bar touches both otherwise disconnected CONTACT
  //  roots.  This conductor-only certificate marks both roots exempt but
  //  deliberately leaves production's diode-mediated union out of its DSU.
  top.shapes (fixture.active.layer ()).insert (
    db::Box (2050, 0, 2450, 100));
  top.shapes (fixture.nplus.layer ()).insert (
    db::Box (2050, 0, 2450, 100));

  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);
  db::CudaAntennaM1Oracle oracle;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (), oracle, 0),
    true);

  EXPECT_EQ (oracle.graph_node_count, uint64_t (9));
  EXPECT_EQ (oracle.component_count, uint64_t (5));
  EXPECT_EQ (oracle.graph_candidate_pair_count, uint64_t (27));
  EXPECT_EQ (oracle.graph_edge_count, uint64_t (4));
  EXPECT_EQ (oracle.annotation_candidate_pair_count, uint64_t (12));
  EXPECT_EQ (oracle.gate_component_count, uint64_t (0));
  EXPECT_EQ (oracle.diode_exempt_component_count, uint64_t (2));
  EXPECT_EQ (oracle.metal1_area_dbu2, uint64_t (32800));

  const uint64_t labels [] = { 0, 1, 2, 3, 1, 2, 6, 1, 2 };
  EXPECT_EQ (
    oracle.canonical_labels ==
      std::vector<uint64_t> (labels, labels + 9),
    true);
}

TEST(11_OracleGateEligibilityUsesProductionOneDbuSquaredBoundary)
{
  AntennaFixture fixture;
  db::Cell &top = fixture.poly.initial_cell ();

  top.shapes (fixture.poly.layer ()).insert (
    db::Box (2000, 0, 2001, 1));
  top.shapes (fixture.active.layer ()).insert (
    db::Box (2000, 0, 2001, 1));
  top.shapes (fixture.poly.layer ()).insert (
    db::Box (3000, 0, 3002, 1));
  top.shapes (fixture.active.layer ()).insert (
    db::Box (3000, 0, 3002, 1));

  db::CudaAntennaM1Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM1CaptureLimits (), capture, 0),
    true);
  db::CudaAntennaM1Oracle oracle;
  EXPECT_EQ (
    db::cuda_antenna_m1_cpu_oracle (
      capture, db::CudaAntennaM1OracleLimits (), oracle, 0),
    true);

  EXPECT_EQ (oracle.graph_node_count, uint64_t (5));
  EXPECT_EQ (oracle.component_count, uint64_t (5));
  EXPECT_EQ (oracle.graph_candidate_pair_count, uint64_t (7));
  EXPECT_EQ (oracle.graph_edge_count, uint64_t (0));
  EXPECT_EQ (oracle.annotation_candidate_pair_count, uint64_t (7));
  EXPECT_EQ (oracle.gate_area_dbu2, uint64_t (3));
  EXPECT_EQ (oracle.gate_component_count, uint64_t (1));
}
