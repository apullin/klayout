/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM4.h"

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

struct AntennaM4Fixture
{
  db::DeepShapeStore store;
  db::DeepLayer poly;
  db::DeepLayer active;
  db::DeepLayer nplus;
  db::DeepLayer nwell;
  db::DeepLayer contact;
  db::DeepLayer metal1;
  db::DeepLayer via1;
  db::DeepLayer metal2;
  db::DeepLayer via2;
  db::DeepLayer metal3;
  db::DeepLayer via3;
  db::DeepLayer metal4;

  explicit AntennaM4Fixture (double dbu = 0.0005)
    : store ("TOP", dbu),
      poly (make_seed_layer (store)),
      active (poly.derived ()),
      nplus (poly.derived ()),
      nwell (poly.derived ()),
      contact (poly.derived ()),
      metal1 (poly.derived ()),
      via1 (poly.derived ()),
      metal2 (poly.derived ()),
      via2 (poly.derived ()),
      metal3 (poly.derived ()),
      via3 (poly.derived ()),
      metal4 (poly.derived ())
  {
    const int physical [db::CudaAntennaM4DomainCount] =
      { 9, 1, 4, 3, 10, 11, 12, 13, 14, 15, 16, 17 };
    const std::array<
      db::DeepLayer *, db::CudaAntennaM4DomainCount> all = layers ();
    for (size_t domain = 0;
         domain < db::CudaAntennaM4DomainCount; ++domain) {
      mark_physical (*all [domain], physical [domain]);
      if (domain) {
        const db::Coord x = db::Coord (domain * 200);
        all [domain]->initial_cell ().shapes (
          all [domain]->layer ()).insert (
            db::Box (x, 0, x + 100 + db::Coord (domain), 80));
      }
    }
  }

  std::array<db::DeepLayer *, db::CudaAntennaM4DomainCount> layers ()
  {
    return {{
      &poly, &active, &nplus, &nwell, &contact, &metal1,
      &via1, &metal2, &via2, &metal3, &via3, &metal4
    }};
  }

  void add_box_to_all (
    db::Cell &cell, db::Coord x, db::Coord y,
    db::Coord width, db::Coord height)
  {
    const std::array<
      db::DeepLayer *, db::CudaAntennaM4DomainCount> all = layers ();
    for (size_t domain = 0;
         domain < db::CudaAntennaM4DomainCount; ++domain) {
      cell.shapes (all [domain]->layer ()).insert (
        db::Box (x, y, x + width, y + height));
    }
  }

  void add_nested_eight_transform_hierarchy ()
  {
    db::Layout &layout = poly.layout ();
    db::Cell &top = poly.initial_cell ();
    db::Cell &child =
      layout.cell (layout.add_cell ("ANTENNA_M4_CHILD"));
    db::Cell &grandchild =
      layout.cell (layout.add_cell ("ANTENNA_M4_GRANDCHILD"));
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
    const db::CudaAntennaM4CaptureLimits &limits,
    db::CudaAntennaM4Capture &capture,
    std::string *reason = 0)
  {
    return db::cuda_antenna_m4_build_capture (
      poly, active, nplus, nwell, contact, metal1,
      via1, metal2, via2, metal3, via3, metal4,
      limits, capture, reason);
  }
};

bool same_context (
  const db::CudaM1WidthSpaceContext &first,
  const db::CudaM1WidthSpaceContext &second)
{
  return first.tx == second.tx && first.ty == second.ty &&
         first.cell_id == second.cell_id &&
         first.transform_code == second.transform_code;
}

bool same_cell (
  const db::CudaM1WidthSpaceCell &first,
  const db::CudaM1WidthSpaceCell &second)
{
  return first.source_cell_index == second.source_cell_index &&
         first.polygon_begin == second.polygon_begin &&
         first.edge_begin == second.edge_begin &&
         first.polygon_count == second.polygon_count &&
         first.edge_count == second.edge_count;
}

bool same_polygon (
  const db::CudaM1WidthSpacePolygon &first,
  const db::CudaM1WidthSpacePolygon &second)
{
  return first.edge_begin == second.edge_begin &&
         first.left == second.left && first.bottom == second.bottom &&
         first.right == second.right && first.top == second.top &&
         first.polygon_id == second.polygon_id &&
         first.edge_count == second.edge_count;
}

bool same_edge (
  const db::CudaM1WidthSpaceEdge &first,
  const db::CudaM1WidthSpaceEdge &second)
{
  return first.x1 == second.x1 && first.y1 == second.y1 &&
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

} // anonymous namespace

TEST(1_CaptureSharesHierarchyPreservesM1AndBindsTwelveDomains)
{
  AntennaM4Fixture fixture;
  fixture.add_nested_eight_transform_hierarchy ();
  db::CudaAntennaM4Capture capture;
  std::string reason;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM4CaptureLimits (), capture, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (capture.format_version, uint32_t (1));
  EXPECT_EQ (capture.lower.format_version, uint32_t (2));
  EXPECT_EQ (capture.lower.source_cell_indices.size (), size_t (3));
  EXPECT_EQ (capture.lower.contexts.size (), size_t (17));
  EXPECT_EQ (capture.lower.context_parent_ids.size (), size_t (17));

  db::CudaAntennaM1Capture independent_lower;
  EXPECT_EQ (
    db::cuda_antenna_m1_build_capture (
      fixture.poly, fixture.active, fixture.nplus, fixture.nwell,
      fixture.contact, fixture.metal1,
      db::CudaAntennaM1CaptureLimits (), independent_lower, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (capture.lower.digest == independent_lower.digest, true);
  EXPECT_EQ (
    capture.lower.hierarchy_digest ==
      independent_lower.hierarchy_digest,
    true);

  db::CudaAntennaM4Census census;
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_census (
      capture, census, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (census.shared_cell_count, uint64_t (3));
  EXPECT_EQ (census.shared_context_count, uint64_t (17));
  EXPECT_EQ (census.context_parent_record_count, uint64_t (17));
  EXPECT_EQ (census.stored_cell_records, uint64_t (36));
  EXPECT_EQ (census.stored_polygon_count, uint64_t (36));
  EXPECT_EQ (census.stored_edge_count, uint64_t (144));
  EXPECT_EQ (census.expanded_polygon_count, uint64_t (204));
  EXPECT_EQ (census.expanded_edge_count, uint64_t (816));
  EXPECT_EQ (
    census.estimated_peak_bytes,
    census.total_stored_bytes +
      census.total_expanded_geometry_bytes);
  EXPECT_EQ (
    census.hierarchy_digest == capture.lower.hierarchy_digest, true);
  EXPECT_EQ (
    census.lower_capture_digest == capture.lower.digest, true);
  EXPECT_EQ (census.capture_digest == capture.digest, true);

  const uint32_t physical [db::CudaAntennaM4DomainCount] =
    { 9, 1, 4, 3, 10, 11, 12, 13, 14, 15, 16, 17 };
  for (size_t domain = 0;
       domain < db::CudaAntennaM4DomainCount; ++domain) {
    const db::CudaAntennaM1DomainCensus &record =
      census.domains [domain];
    EXPECT_EQ (record.role, uint32_t (domain));
    EXPECT_EQ (record.physical_layer, physical [domain]);
    EXPECT_EQ (record.datatype, uint32_t (0));
    EXPECT_EQ (record.stored_cell_count, uint64_t (3));
    EXPECT_EQ (record.stored_context_count, uint64_t (17));
    EXPECT_EQ (record.nonempty_context_count, uint64_t (17));
    EXPECT_EQ (record.stored_polygon_count, uint64_t (3));
    EXPECT_EQ (record.stored_edge_count, uint64_t (12));
    EXPECT_EQ (record.expanded_polygon_count, uint64_t (17));
    EXPECT_EQ (record.expanded_edge_count, uint64_t (68));

    db::CudaRawManhattanScene materialized;
    EXPECT_EQ (
      db::cuda_antenna_m4_materialize_domain_scene (
        capture, db::CudaAntennaM4Domain (domain),
        materialized, &reason),
      true);
    EXPECT_EQ (reason, "");
    EXPECT_EQ (materialized.contexts.size (), size_t (17));
    EXPECT_EQ (materialized.cells.size (), size_t (3));
    EXPECT_EQ (materialized.polygons.size (), size_t (3));
    EXPECT_EQ (materialized.edges.size (), size_t (12));
    EXPECT_EQ (materialized.digest == record.scene_digest, true);
  }

  db::CudaRawManhattanScene legacy_m2;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      fixture.metal2, db::CudaM1WidthSpaceSceneLimits (),
      legacy_m2, &reason),
    true);
  db::CudaRawManhattanScene materialized_m2;
  EXPECT_EQ (
    db::cuda_antenna_m4_materialize_domain_scene (
      capture, db::CudaAntennaM4Metal2, materialized_m2, &reason),
    true);
  EXPECT_EQ (same_raw_scene (legacy_m2, materialized_m2), true);

  std::array<uint8_t, 32> recomputed;
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_digest (capture, recomputed), true);
  EXPECT_EQ (recomputed == capture.digest, true);
  db::CudaAntennaM4Capture repeated;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM4CaptureLimits (), repeated, 0),
    true);
  EXPECT_EQ (repeated.digest == capture.digest, true);

  const std::string text = db::cuda_antenna_m4_census_text (census);
  EXPECT_EQ (
    text.find ("antenna_m4_capture format=1") != std::string::npos,
    true);
  EXPECT_EQ (text.find ("shared_contexts=17") != std::string::npos, true);
  EXPECT_EQ (text.find ("VIA1{physical=12/0") != std::string::npos, true);
  EXPECT_EQ (text.find ("M4{physical=17/0") != std::string::npos, true);
}

TEST(2_CaptureFailsClosedOnIdentityLayersEmptyAndCapacity)
{
  AntennaM4Fixture fixture;
  fixture.add_nested_eight_transform_hierarchy ();
  db::CudaAntennaM4Capture baseline;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM4CaptureLimits (), baseline, 0),
    true);
  db::CudaAntennaM4Census census;
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_census (baseline, census, 0), true);

  db::CudaAntennaM4Capture sentinel;
  sentinel.format_version = 77;
  std::string reason;
  EXPECT_EQ (
    db::cuda_antenna_m4_build_capture (
      fixture.poly, fixture.active, fixture.nplus, fixture.nwell,
      fixture.contact, fixture.metal1, fixture.via1, fixture.via1,
      fixture.via2, fixture.metal3, fixture.via3, fixture.metal4,
      db::CudaAntennaM4CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (reason.find ("distinct") != std::string::npos, true);

  mark_physical (fixture.via2, 99);
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM4CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (reason.find ("VIA2") != std::string::npos, true);
  mark_physical (fixture.via2, 14);

  db::DeepShapeStore other_store ("TOP", 0.0005);
  db::DeepLayer other_metal3 = make_seed_layer (other_store);
  mark_physical (other_metal3, 15);
  EXPECT_EQ (
    db::cuda_antenna_m4_build_capture (
      fixture.poly, fixture.active, fixture.nplus, fixture.nwell,
      fixture.contact, fixture.metal1, fixture.via1, fixture.metal2,
      fixture.via2, other_metal3, fixture.via3, fixture.metal4,
      db::CudaAntennaM4CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (
    reason.find ("one hierarchy") != std::string::npos, true);

  db::CudaAntennaM4CaptureLimits stored_limit;
  stored_limit.max_total_stored_bytes =
    census.total_stored_bytes - 1;
  EXPECT_EQ (
    fixture.build (stored_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (reason.find ("stored bytes") != std::string::npos, true);

  db::CudaAntennaM4CaptureLimits expanded_limit;
  expanded_limit.max_total_expanded_geometry_bytes =
    census.total_expanded_geometry_bytes - 1;
  EXPECT_EQ (
    fixture.build (expanded_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (
    reason.find ("expanded geometry bytes") != std::string::npos,
    true);

  db::CudaAntennaM4CaptureLimits peak_limit;
  peak_limit.max_estimated_peak_bytes =
    census.estimated_peak_bytes - 1;
  EXPECT_EQ (
    fixture.build (peak_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (
    reason.find ("estimated peak bytes") != std::string::npos, true);

  {
    db::CudaAntennaM4CaptureLimits zero;
    zero.max_total_stored_bytes = 0;
    EXPECT_EQ (fixture.build (zero, sentinel, &reason), false);
    EXPECT_EQ (sentinel.format_version, uint32_t (77));
    EXPECT_EQ (
      reason.find ("aggregate byte capacity is zero") !=
        std::string::npos,
      true);
  }
  {
    db::CudaAntennaM4CaptureLimits zero;
    zero.max_total_expanded_geometry_bytes = 0;
    EXPECT_EQ (fixture.build (zero, sentinel, &reason), false);
    EXPECT_EQ (sentinel.format_version, uint32_t (77));
    EXPECT_EQ (
      reason.find ("aggregate byte capacity is zero") !=
        std::string::npos,
      true);
  }
  {
    db::CudaAntennaM4CaptureLimits zero;
    zero.max_estimated_peak_bytes = 0;
    EXPECT_EQ (fixture.build (zero, sentinel, &reason), false);
    EXPECT_EQ (sentinel.format_version, uint32_t (77));
    EXPECT_EQ (
      reason.find ("aggregate byte capacity is zero") !=
        std::string::npos,
      true);
  }

  db::CudaAntennaM4CaptureLimits scene_limit;
  scene_limit.scene.max_flat_edges = 67;
  EXPECT_EQ (
    fixture.build (scene_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (
    reason.find ("configured capacity") != std::string::npos, true);

  AntennaM4Fixture empty;
  empty.metal4.initial_cell ().shapes (empty.metal4.layer ()).clear ();
  EXPECT_EQ (
    empty.build (
      db::CudaAntennaM4CaptureLimits (), sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (reason.find ("M4") != std::string::npos, true);

  uint64_t lower_and_header_bytes = census.total_stored_bytes;
  for (size_t domain = db::CudaAntennaM4Via1;
       domain < db::CudaAntennaM4DomainCount; ++domain) {
    lower_and_header_bytes -= census.domains [domain].stored_bytes;
  }
  db::CudaAntennaM4CaptureLimits incremental_limit;
  incremental_limit.max_total_stored_bytes =
    lower_and_header_bytes +
      census.domains [db::CudaAntennaM4Via1].stored_bytes - 1;
  fixture.metal4.initial_cell ().shapes (
    fixture.metal4.layer ()).insert (
      db::object_with_properties<db::Box> (
        db::Box (6000, 0, 6100, 100), 1));
  EXPECT_EQ (
    fixture.build (incremental_limit, sentinel, &reason), false);
  EXPECT_EQ (sentinel.format_version, uint32_t (77));
  EXPECT_EQ (
    reason.find ("VIA1 stored bytes") != std::string::npos, true);
}

TEST(3_CaptureRejectsUnsupportedUpperGeometryPropertiesAndTransforms)
{
  {
    AntennaM4Fixture fixture;
    fixture.via2.initial_cell ().shapes (
      fixture.via2.layer ()).insert (
        db::object_with_properties<db::Box> (
          db::Box (4000, 0, 4100, 100), 1));
    db::CudaAntennaM4Capture capture;
    capture.format_version = 81;
    std::string reason;
    EXPECT_EQ (
      fixture.build (
        db::CudaAntennaM4CaptureLimits (), capture, &reason),
      false);
    EXPECT_EQ (capture.format_version, uint32_t (81));
    EXPECT_EQ (reason.find ("properties") != std::string::npos, true);
  }

  {
    AntennaM4Fixture fixture;
    db::Point diagonal_hull [] = {
      db::Point (4000, 0), db::Point (4100, 100),
      db::Point (4200, 0)
    };
    db::Polygon diagonal;
    diagonal.assign_hull (
      diagonal_hull, diagonal_hull + 3);
    fixture.metal3.initial_cell ().shapes (
      fixture.metal3.layer ()).insert (diagonal);
    db::CudaAntennaM4Capture capture;
    std::string reason;
    EXPECT_EQ (
      fixture.build (
        db::CudaAntennaM4CaptureLimits (), capture, &reason),
      false);
    EXPECT_EQ (
      reason.find ("non-Manhattan") != std::string::npos, true);
  }

  {
    AntennaM4Fixture fixture;
    db::Cell &complex_child =
      fixture.poly.layout ().cell (
        fixture.poly.layout ().add_cell ("M4_COMPLEX_CHILD"));
    fixture.add_box_to_all (complex_child, 0, 0, 20, 20);
    fixture.poly.initial_cell ().insert (
      db::CellInstArray (
        db::CellInst (complex_child.cell_index ()),
        db::ICplxTrans (
          1.0, 45.0, false, db::Vector (1000, 1000))));
    db::CudaAntennaM4Capture capture;
    std::string reason;
    EXPECT_EQ (
      fixture.build (
        db::CudaAntennaM4CaptureLimits (), capture, &reason),
      false);
    EXPECT_EQ (
      reason.find ("complex transform") != std::string::npos, true);
  }
}

TEST(4_CaptureDigestAndMaterializationFailClosedOnDrift)
{
  AntennaM4Fixture fixture;
  fixture.add_nested_eight_transform_hierarchy ();
  db::CudaAntennaM4Capture capture;
  EXPECT_EQ (
    fixture.build (
      db::CudaAntennaM4CaptureLimits (), capture, 0),
    true);

  std::array<uint8_t, 32> sentinel;
  sentinel.fill (UINT8_C (0xa5));
  std::array<uint8_t, 32> digest = sentinel;
  db::CudaAntennaM4Capture geometry = capture;
  ++geometry.upper_domains [db::CudaAntennaM4Via2 -
      db::CudaAntennaM1DomainCount].edges [0].x2;
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_digest (geometry, digest), false);
  EXPECT_EQ (digest == sentinel, true);

  db::CudaRawManhattanScene materialize_sentinel;
  materialize_sentinel.root_cell = 77;
  std::string reason;
  EXPECT_EQ (
    db::cuda_antenna_m4_materialize_domain_scene (
      geometry, db::CudaAntennaM4Via2,
      materialize_sentinel, &reason),
    false);
  EXPECT_EQ (materialize_sentinel.root_cell, uint32_t (77));

  db::CudaAntennaM4Capture layer = capture;
  layer.upper_source_layer_indices [0] =
    layer.lower.source_layer_indices [db::CudaAntennaM1Metal1];
  digest = sentinel;
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_digest (layer, digest), false);
  EXPECT_EQ (digest == sentinel, true);

  db::CudaAntennaM4Capture stale = capture;
  stale.digest [0] ^= UINT8_C (1);
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_digest (stale, digest), true);
  EXPECT_EQ (digest == capture.digest, true);
  db::CudaAntennaM4Census census;
  census.shared_cell_count = 777;
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_census (
      stale, census, &reason),
    false);
  EXPECT_EQ (census.shared_cell_count, uint64_t (777));
  EXPECT_EQ (
    reason.find ("aggregate capture digest") != std::string::npos,
    true);

  db::CudaAntennaM4Capture lower_drift = capture;
  lower_drift.lower.context_parent_ids [9] = 9;
  digest = sentinel;
  EXPECT_EQ (
    db::cuda_antenna_m4_capture_digest (lower_drift, digest), false);
  EXPECT_EQ (digest == sentinel, true);
}
