/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaM1WidthSpace.h"

#include "dbCell.h"
#include "dbDeepShapeStore.h"
#include "dbLayerProperties.h"
#include "dbLayout.h"
#include "dbPolygon.h"
#include "dbRegion.h"
#include "dbText.h"
#include "tlUnitTest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace
{

typedef bool (*RawSceneBuilder) (
  const db::DeepLayer &, const db::CudaM1WidthSpaceSceneLimits &,
  db::CudaRawManhattanScene &, std::string *);

typedef bool (*RawSceneDigest) (
  const db::CudaRawManhattanScene &, std::array<uint8_t, 32> &);

struct RawSceneProfile
{
  int layer;
  int datatype;
  const char *label;
  RawSceneBuilder build;
  RawSceneDigest digest;
};

const RawSceneProfile profiles [] = {
  {
    13, 0, "raw M2",
    db::cuda_m2_raw_manhattan_build_scene,
    db::cuda_m2_raw_manhattan_scene_digest
  },
  {
    1, 0, "raw ACTIVE",
    db::cuda_active_raw_manhattan_build_scene,
    db::cuda_active_raw_manhattan_scene_digest
  },
  {
    10, 0, "raw CONTACT",
    db::cuda_contact_raw_manhattan_build_scene,
    db::cuda_contact_raw_manhattan_scene_digest
  }
};

void mark_physical (
  db::DeepLayer &layer, int physical_layer, int datatype = 0)
{
  layer.layout ().set_properties (
    layer.layer (), db::LayerProperties (physical_layer, datatype));
}

void mark_physical_m2 (db::DeepLayer &layer)
{
  mark_physical (layer, 13, 0);
}

db::DeepLayer make_raw_m2 (
  db::DeepShapeStore &store, const db::Region &seed)
{
  db::DeepLayer layer = store.create_from_flat (seed, false);
  mark_physical_m2 (layer);
  return layer;
}

std::string digest_hex (const std::array<uint8_t, 32> &digest)
{
  std::ostringstream text;
  text << std::hex << std::setfill ('0');
  for (size_t i = 0; i < digest.size (); ++i) {
    text << std::setw (2) << unsigned (digest [i]);
  }
  return text.str ();
}

template <class T>
bool same_records (const std::vector<T> &first,
                   const std::vector<T> &second)
{
  return
    first.size () == second.size () &&
    (first.empty () ||
     std::memcmp (
       first.data (), second.data (), first.size () * sizeof (T)) == 0);
}

bool same_geometry (
  const db::CudaRawManhattanScene &first,
  const db::CudaRawManhattanScene &second)
{
  return
    first.format_version == second.format_version &&
    first.dbu_per_micron == second.dbu_per_micron &&
    first.root_cell == second.root_cell &&
    first.reserved == second.reserved &&
    first.flat_polygon_count == second.flat_polygon_count &&
    first.flat_edge_count == second.flat_edge_count &&
    first.scene_left == second.scene_left &&
    first.scene_bottom == second.scene_bottom &&
    first.scene_right == second.scene_right &&
    first.scene_top == second.scene_top &&
    same_records (first.contexts, second.contexts) &&
    same_records (first.metal_contexts, second.metal_contexts) &&
    same_records (
      first.context_polygon_offsets, second.context_polygon_offsets) &&
    same_records (
      first.context_edge_offsets, second.context_edge_offsets) &&
    same_records (first.cells, second.cells) &&
    same_records (first.polygons, second.polygons) &&
    same_records (first.edges, second.edges);
}

} // anonymous namespace

TEST(1_RawSceneIsDistinctDeterministicAndUnmerged)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  seed.insert (db::Box (100, 0, 300, 100));
  db::DeepLayer raw_m2 = make_raw_m2 (store, seed);

  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaM2RawManhattanScene scene;
  std::string reason;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, scene, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (scene.cells.size (), size_t (1));
  EXPECT_EQ (scene.contexts.size (), size_t (1));
  EXPECT_EQ (scene.metal_contexts.size (), size_t (1));
  EXPECT_EQ (scene.polygons.size (), size_t (2));
  EXPECT_EQ (scene.edges.size (), size_t (8));
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (2));
  EXPECT_EQ (scene.flat_edge_count, uint64_t (8));

  std::array<uint8_t, 32> recomputed;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_scene_digest (scene, recomputed), true);
  EXPECT_EQ (recomputed == scene.digest, true);
  EXPECT_EQ (
    digest_hex (scene.digest),
    "a3417a95c51db78d4b1fed1209f825bea5b40afec75ff525a2fc0f0baf2bde89");

  db::CudaM2RawManhattanScene repeated;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, repeated, 0),
    true);
  EXPECT_EQ (repeated.digest == scene.digest, true);

  db::CudaM1WidthSpaceBuildSpec merged_spec;
  merged_spec.width_distance = 140;
  merged_spec.spacing_distance = 140;
  merged_spec.inputs_are_merged = true;
  db::CudaM1WidthSpaceScene profiled;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      raw_m2, raw_m2, merged_spec, limits, profiled, 0),
    true);
  EXPECT_EQ (profiled.digest == scene.digest, false);
}

TEST(2_PhysicalLayerAndCapacityFailClosed)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  db::DeepLayer raw_m2 = make_raw_m2 (store, seed);

  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaM2RawManhattanScene scene;
  scene.flat_polygon_count = 17;
  std::string reason;

  raw_m2.layout ().set_properties (
    raw_m2.layer (), db::LayerProperties (101, 0));
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, scene, &reason),
    false);
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (reason.empty (), false);

  mark_physical_m2 (raw_m2);
  db::Layout &layout = raw_m2.layout ();
  const unsigned int raw_layer_index = raw_m2.layer ();
  layout.delete_layer (raw_layer_index);
  layout.insert_special_layer (
    raw_layer_index, db::LayerProperties (13, 0));
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, scene, &reason),
    false);
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (reason.empty (), false);

  layout.delete_layer (raw_layer_index);
  layout.insert_layer (
    raw_layer_index, db::LayerProperties (13, 0));
  limits.max_flat_polygons = 0;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, scene, &reason),
    false);
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (reason.empty (), false);
}

TEST(3_RawHierarchyTransformsArePreserved)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 100, 100));
  db::DeepLayer raw_m2 = make_raw_m2 (store, seed);
  db::Layout &layout = raw_m2.layout ();
  db::Cell &top = raw_m2.initial_cell ();
  db::Cell &child = layout.cell (layout.add_cell ("CHILD"));
  child.shapes (raw_m2.layer ()).insert (db::Box (0, 0, 20, 10));
  top.insert (
    db::CellInstArray (
      db::CellInst (child.cell_index ()),
      db::Trans (1, false, db::Vector (1000, 2000)),
      db::Vector (100, 0), db::Vector (), 2, 1));

  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaM2RawManhattanScene scene;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, scene, 0),
    true);
  EXPECT_EQ (scene.cells.size (), size_t (2));
  EXPECT_EQ (scene.contexts.size (), size_t (3));
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (3));
  EXPECT_EQ (scene.flat_edge_count, uint64_t (12));
  EXPECT_EQ (scene.contexts [1].transform_code, uint32_t (1));
  EXPECT_EQ (scene.contexts [1].tx, int64_t (1000));
  EXPECT_EQ (scene.contexts [1].ty, int64_t (2000));
  EXPECT_EQ (scene.contexts [2].transform_code, uint32_t (1));
  EXPECT_EQ (scene.contexts [2].tx, int64_t (1100));
  EXPECT_EQ (scene.contexts [2].ty, int64_t (2000));
  EXPECT_EQ (scene.scene_right, int64_t (1100));
  EXPECT_EQ (scene.scene_top, int64_t (2020));
}

TEST(4_TextLabelsAreFilteredButOtherNonPolygonsDecline)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  db::DeepLayer raw_m2 = make_raw_m2 (store, seed);
  db::Shapes &shapes =
    raw_m2.initial_cell ().shapes (raw_m2.layer ());
  shapes.insert (
    db::Text ("ACTIVE_LABEL", db::Trans (db::Vector (10, 20))));

  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaM2RawManhattanScene scene;
  std::string reason;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, scene, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (scene.polygons.size (), size_t (1));
  EXPECT_EQ (scene.edges.size (), size_t (4));
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (1));

  shapes.insert (db::Edge (0, 0, 100, 0));
  db::CudaM2RawManhattanScene declined;
  declined.flat_polygon_count = 17;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_build_scene (
      raw_m2, limits, declined, &reason),
    false);
  EXPECT_EQ (declined.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (
    reason.find ("non-polygon shape") != std::string::npos,
    true);
}

TEST(5_FixedRawDomainsShareGeometryButNotIdentity)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  seed.insert (db::Box (100, 0, 300, 100));
  db::DeepLayer raw = store.create_from_flat (seed, false);
  db::CudaM1WidthSpaceSceneLimits limits;
  std::vector<db::CudaRawManhattanScene> scenes;

  for (size_t i = 0; i < sizeof (profiles) / sizeof (profiles [0]); ++i) {
    mark_physical (raw, profiles [i].layer, profiles [i].datatype);
    db::CudaRawManhattanScene scene;
    std::string reason;
    EXPECT_EQ (
      profiles [i].build (raw, limits, scene, &reason), true);
    EXPECT_EQ (reason, "");

    std::array<uint8_t, 32> recomputed;
    EXPECT_EQ (profiles [i].digest (scene, recomputed), true);
    EXPECT_EQ (recomputed == scene.digest, true);

    db::CudaRawManhattanScene repeated;
    EXPECT_EQ (
      profiles [i].build (raw, limits, repeated, 0), true);
    EXPECT_EQ (same_geometry (scene, repeated), true);
    EXPECT_EQ (scene.digest == repeated.digest, true);
    scenes.push_back (scene);
  }

  EXPECT_EQ (same_geometry (scenes [0], scenes [1]), true);
  EXPECT_EQ (same_geometry (scenes [0], scenes [2]), true);
  EXPECT_EQ (scenes [0].digest == scenes [1].digest, false);
  EXPECT_EQ (scenes [0].digest == scenes [2].digest, false);
  EXPECT_EQ (scenes [1].digest == scenes [2].digest, false);
  EXPECT_EQ (
    digest_hex (scenes [0].digest),
    "a3417a95c51db78d4b1fed1209f825bea5b40afec75ff525a2fc0f0baf2bde89");
  EXPECT_EQ (
    digest_hex (scenes [1].digest),
    "51c9bf6254a0d1a3e1183442048b920b35edad099141994778219743f81969f4");
  EXPECT_EQ (
    digest_hex (scenes [2].digest),
    "30bc97c324cd796d37c609267be7169488d1c48c23ae82f26e16bd0b1e094390");

  //  With identical geometry, these values differ only because the exact
  //  eight-byte domain magic is part of the canonical digest.
  std::array<uint8_t, 32> active_from_m2;
  std::array<uint8_t, 32> contact_from_m2;
  EXPECT_EQ (
    db::cuda_active_raw_manhattan_scene_digest (
      scenes [0], active_from_m2),
    true);
  EXPECT_EQ (
    db::cuda_contact_raw_manhattan_scene_digest (
      scenes [0], contact_from_m2),
    true);
  EXPECT_EQ (active_from_m2 == scenes [1].digest, true);
  EXPECT_EQ (contact_from_m2 == scenes [2].digest, true);
}

TEST(6_EachRawDomainRequiresItsExactPhysicalLayer)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  db::DeepLayer raw = store.create_from_flat (seed, false);
  db::CudaM1WidthSpaceSceneLimits limits;

  for (size_t i = 0; i < sizeof (profiles) / sizeof (profiles [0]); ++i) {
    const size_t wrong = (i + 1) %
      (sizeof (profiles) / sizeof (profiles [0]));
    mark_physical (
      raw, profiles [wrong].layer, profiles [wrong].datatype);
    db::CudaRawManhattanScene scene;
    scene.flat_polygon_count = 17;
    std::string reason;
    EXPECT_EQ (
      profiles [i].build (raw, limits, scene, &reason), false);
    EXPECT_EQ (scene.flat_polygon_count, uint64_t (17));
    EXPECT_EQ (
      reason.find (profiles [i].label) != std::string::npos, true);
    EXPECT_EQ (
      reason.find (
        std::to_string (profiles [i].layer) + "/" +
        std::to_string (profiles [i].datatype)) != std::string::npos,
      true);

    mark_physical (raw, profiles [i].layer, 1);
    EXPECT_EQ (
      profiles [i].build (raw, limits, scene, &reason), false);
    EXPECT_EQ (scene.flat_polygon_count, uint64_t (17));

    mark_physical (
      raw, profiles [i].layer, profiles [i].datatype);
    EXPECT_EQ (
      profiles [i].build (raw, limits, scene, &reason), true);
    EXPECT_EQ (reason, "");
    EXPECT_EQ (scene.flat_polygon_count, uint64_t (1));
  }
}

TEST(7_ActiveAndContactPreserveAllEightHierarchyTransforms)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 100, 100));
  db::DeepLayer raw = store.create_from_flat (seed, false);
  db::Layout &layout = raw.layout ();
  db::Cell &top = raw.initial_cell ();
  db::Cell &child = layout.cell (layout.add_cell ("CHILD"));
  child.shapes (raw.layer ()).insert (db::Box (0, 0, 20, 10));
  for (unsigned int code = 0; code < 8; ++code) {
    top.insert (
      db::CellInstArray (
        db::CellInst (child.cell_index ()),
        db::Trans (
          int (code & 3), code >= 4,
          db::Vector (db::Coord (1000 * code), 2000))));
  }

  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaRawManhattanScene active;
  mark_physical (raw, 1, 0);
  EXPECT_EQ (
    db::cuda_active_raw_manhattan_build_scene (
      raw, limits, active, 0),
    true);
  db::CudaRawManhattanScene contact;
  mark_physical (raw, 10, 0);
  EXPECT_EQ (
    db::cuda_contact_raw_manhattan_build_scene (
      raw, limits, contact, 0),
    true);

  EXPECT_EQ (same_geometry (active, contact), true);
  EXPECT_EQ (active.cells.size (), size_t (2));
  EXPECT_EQ (active.contexts.size (), size_t (9));
  EXPECT_EQ (active.flat_polygon_count, uint64_t (9));
  EXPECT_EQ (active.flat_edge_count, uint64_t (36));
  std::vector<uint32_t> codes;
  for (size_t i = 1; i < active.contexts.size (); ++i) {
    codes.push_back (active.contexts [i].transform_code);
  }
  std::sort (codes.begin (), codes.end ());
  for (uint32_t code = 0; code < 8; ++code) {
    EXPECT_EQ (codes [code], code);
  }
}

TEST(8_AllRawDomainsRetainTextAndUnsupportedShapeContract)
{
  for (size_t i = 0; i < sizeof (profiles) / sizeof (profiles [0]); ++i) {
    db::DeepShapeStore store ("TOP", 0.0005);
    db::Region seed;
    seed.insert (db::Box (0, 0, 200, 100));
    db::DeepLayer raw = store.create_from_flat (seed, false);
    mark_physical (raw, profiles [i].layer, profiles [i].datatype);
    db::Shapes &shapes = raw.initial_cell ().shapes (raw.layer ());
    shapes.insert (
      db::Text ("IGNORED_LABEL", db::Trans (db::Vector (10, 20))));

    db::CudaM1WidthSpaceSceneLimits limits;
    db::CudaRawManhattanScene scene;
    std::string reason;
    EXPECT_EQ (
      profiles [i].build (raw, limits, scene, &reason), true);
    EXPECT_EQ (scene.polygons.size (), size_t (1));
    EXPECT_EQ (scene.edges.size (), size_t (4));

    shapes.insert (db::Edge (0, 0, 100, 0));
    db::CudaRawManhattanScene declined;
    declined.flat_polygon_count = 17;
    EXPECT_EQ (
      profiles [i].build (raw, limits, declined, &reason), false);
    EXPECT_EQ (declined.flat_polygon_count, uint64_t (17));
    EXPECT_EQ (
      reason.find ("non-polygon shape") != std::string::npos, true);
  }
}

TEST(9_AllRawDomainsRejectPropertiesHolesAndNonManhattanContours)
{
  for (size_t i = 0; i < sizeof (profiles) / sizeof (profiles [0]); ++i) {
    {
      db::DeepShapeStore store ("TOP", 0.0005);
      db::Region seed;
      seed.insert (db::Box (0, 0, 200, 100));
      db::DeepLayer raw = store.create_from_flat (seed, false);
      mark_physical (raw, profiles [i].layer, profiles [i].datatype);
      raw.initial_cell ().shapes (raw.layer ()).insert (
        db::object_with_properties<db::Box> (
          db::Box (300, 0, 400, 100), 1));
      db::CudaRawManhattanScene scene;
      std::string reason;
      EXPECT_EQ (
        profiles [i].build (
          raw, db::CudaM1WidthSpaceSceneLimits (), scene, &reason),
        false);
      EXPECT_EQ (
        reason.find ("properties") != std::string::npos, true);
    }
    {
      db::DeepShapeStore store ("TOP", 0.0005);
      db::Region seed;
      seed.insert (db::Box (0, 0, 200, 200));
      db::DeepLayer raw = store.create_from_flat (seed, false);
      mark_physical (raw, profiles [i].layer, profiles [i].datatype);
      db::Polygon with_hole (db::Box (300, 0, 500, 200));
      db::Point hole [] = {
        db::Point (350, 50), db::Point (350, 100),
        db::Point (400, 100), db::Point (400, 50)
      };
      with_hole.insert_hole (hole, hole + 4);
      raw.initial_cell ().shapes (raw.layer ()).insert (with_hole);
      db::CudaRawManhattanScene scene;
      std::string reason;
      EXPECT_EQ (
        profiles [i].build (
          raw, db::CudaM1WidthSpaceSceneLimits (), scene, &reason),
        false);
      EXPECT_EQ (reason.find ("holes") != std::string::npos, true);
    }
    {
      db::DeepShapeStore store ("TOP", 0.0005);
      db::Region seed;
      seed.insert (db::Box (0, 0, 200, 100));
      db::DeepLayer raw = store.create_from_flat (seed, false);
      mark_physical (raw, profiles [i].layer, profiles [i].datatype);
      db::Point diagonal_hull [] = {
        db::Point (300, 0), db::Point (400, 100), db::Point (500, 0)
      };
      db::Polygon diagonal;
      diagonal.assign_hull (
        diagonal_hull, diagonal_hull + 3);
      raw.initial_cell ().shapes (raw.layer ()).insert (diagonal);
      db::CudaRawManhattanScene scene;
      std::string reason;
      EXPECT_EQ (
        profiles [i].build (
          raw, db::CudaM1WidthSpaceSceneLimits (), scene, &reason),
        false);
      EXPECT_EQ (
        reason.find ("non-Manhattan") != std::string::npos, true);
    }
  }
}

TEST(10_DomainDigestsFailClosedOnStructuralDrift)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  db::DeepLayer raw = store.create_from_flat (seed, false);
  mark_physical (raw, 1, 0);
  db::CudaRawManhattanScene scene;
  EXPECT_EQ (
    db::cuda_active_raw_manhattan_build_scene (
      raw, db::CudaM1WidthSpaceSceneLimits (), scene, 0),
    true);

  db::CudaRawManhattanScene reserved = scene;
  reserved.reserved = 1;
  std::array<uint8_t, 32> sentinel;
  sentinel.fill (UINT8_C (0xa5));
  std::array<uint8_t, 32> digest = sentinel;
  EXPECT_EQ (
    db::cuda_active_raw_manhattan_scene_digest (reserved, digest),
    false);
  EXPECT_EQ (digest == sentinel, true);

  db::CudaRawManhattanScene offset = scene;
  ++offset.context_edge_offsets [0];
  digest = sentinel;
  EXPECT_EQ (
    db::cuda_contact_raw_manhattan_scene_digest (offset, digest),
    false);
  EXPECT_EQ (digest == sentinel, true);

  db::CudaRawManhattanScene open = scene;
  ++open.edges [0].x2;
  digest = sentinel;
  EXPECT_EQ (
    db::cuda_m2_raw_manhattan_scene_digest (open, digest), false);
  EXPECT_EQ (digest == sentinel, true);
}

TEST(11_AllRawDomainsFailClosedAtRealCapacityBoundaries)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  db::DeepLayer raw = store.create_from_flat (seed, false);

  for (size_t i = 0; i < sizeof (profiles) / sizeof (profiles [0]); ++i) {
    mark_physical (raw, profiles [i].layer, profiles [i].datatype);
    db::CudaM1WidthSpaceSceneLimits limits;
    limits.max_flat_edges = 3;
    db::CudaRawManhattanScene scene;
    scene.flat_edge_count = 17;
    std::string reason;
    EXPECT_EQ (
      profiles [i].build (raw, limits, scene, &reason), false);
    EXPECT_EQ (scene.flat_edge_count, uint64_t (17));
    EXPECT_EQ (
      reason.find ("configured capacity") != std::string::npos, true);
  }
}

TEST(12_CombinedRawWellSceneIsDeterministicAndDomainBound)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region nwell_seed;
  nwell_seed.insert (db::Box (0, 0, 200, 100));
  db::DeepLayer nwell = store.create_from_flat (nwell_seed, false);
  db::DeepLayer pwell = nwell.derived ();
  pwell.initial_cell ().shapes (pwell.layer ()).insert (
    db::Box (300, 0, 500, 150));
  mark_physical (nwell, 3, 0);
  mark_physical (pwell, 2, 0);

  db::CudaRawManhattanScene scene;
  std::string reason;
  EXPECT_EQ (
    db::cuda_well_union_raw_manhattan_build_scene (
      nwell, pwell, db::CudaM1WidthSpaceSceneLimits (), scene, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (scene.cells.size (), size_t (1));
  EXPECT_EQ (scene.contexts.size (), size_t (1));
  EXPECT_EQ (scene.polygons.size (), size_t (2));
  EXPECT_EQ (scene.edges.size (), size_t (8));
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (2));
  EXPECT_EQ (scene.flat_edge_count, uint64_t (8));
  EXPECT_EQ (scene.polygons [0].polygon_id, uint32_t (0));
  EXPECT_EQ (scene.polygons [0].left, int64_t (0));
  EXPECT_EQ (scene.polygons [0].right, int64_t (200));
  EXPECT_EQ (scene.polygons [1].polygon_id, uint32_t (1));
  EXPECT_EQ (scene.polygons [1].left, int64_t (300));
  EXPECT_EQ (scene.polygons [1].right, int64_t (500));

  std::array<uint8_t, 32> recomputed;
  EXPECT_EQ (
    db::cuda_well_union_raw_manhattan_scene_digest (
      scene, recomputed),
    true);
  EXPECT_EQ (recomputed == scene.digest, true);
  std::array<uint8_t, 32> active_domain;
  EXPECT_EQ (
    db::cuda_active_raw_manhattan_scene_digest (
      scene, active_domain),
    true);
  EXPECT_EQ (active_domain == scene.digest, false);

  db::CudaRawManhattanScene repeated;
  EXPECT_EQ (
    db::cuda_well_union_raw_manhattan_build_scene (
      nwell, pwell, db::CudaM1WidthSpaceSceneLimits (), repeated, 0),
    true);
  EXPECT_EQ (same_geometry (scene, repeated), true);
  EXPECT_EQ (scene.digest == repeated.digest, true);
}

TEST(13_CombinedRawWellSceneDeclinesAtomically)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  db::DeepLayer nwell = store.create_from_flat (seed, false);
  db::DeepLayer pwell = nwell.derived ();
  pwell.initial_cell ().shapes (pwell.layer ()).insert (
    db::Box (300, 0, 500, 150));
  mark_physical (nwell, 3, 0);
  mark_physical (pwell, 2, 0);

  db::CudaRawManhattanScene sentinel;
  sentinel.flat_polygon_count = 17;
  std::string reason;

  mark_physical (pwell, 4, 0);
  EXPECT_EQ (
    db::cuda_well_union_raw_manhattan_build_scene (
      nwell, pwell, db::CudaM1WidthSpaceSceneLimits (),
      sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (reason.find ("PWELL") != std::string::npos, true);
  mark_physical (pwell, 2, 0);

  db::DeepShapeStore other_store ("TOP", 0.0005);
  db::DeepLayer other_pwell =
    other_store.create_from_flat (seed, false);
  mark_physical (other_pwell, 2, 0);
  EXPECT_EQ (
    db::cuda_well_union_raw_manhattan_build_scene (
      nwell, other_pwell, db::CudaM1WidthSpaceSceneLimits (),
      sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (
    reason.find ("hierarchy") != std::string::npos, true);

  store.add_breakout_cell (
    nwell.layout_index (), nwell.initial_cell ().cell_index ());
  EXPECT_EQ (
    db::cuda_well_union_raw_manhattan_build_scene (
      nwell, pwell, db::CudaM1WidthSpaceSceneLimits (),
      sentinel, &reason),
    false);
  EXPECT_EQ (sentinel.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (
    reason.find ("breakout") != std::string::npos, true);
}
