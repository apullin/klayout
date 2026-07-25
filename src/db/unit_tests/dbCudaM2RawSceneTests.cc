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
#include "dbRegion.h"
#include "dbText.h"
#include "tlUnitTest.h"

#include <array>
#include <cstdint>
#include <string>

namespace
{

void mark_physical_m2 (db::DeepLayer &layer)
{
  layer.layout ().set_properties (
    layer.layer (), db::LayerProperties (13, 0));
}

db::DeepLayer make_raw_m2 (
  db::DeepShapeStore &store, const db::Region &seed)
{
  db::DeepLayer layer = store.create_from_flat (seed, false);
  mark_physical_m2 (layer);
  return layer;
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
