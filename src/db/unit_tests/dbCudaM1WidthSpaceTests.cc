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
#include "dbLayout.h"
#include "dbRegion.h"
#include "tlUnitTest.h"

#include <array>
#include <cstdint>

namespace
{

// Keep the exported BuildSpec layout compatible with the original M1-only
// seam.  M2 is inferred from its exact 140/140 distance pair; adding a public
// profile field here would shift every legacy member in already-built clients.
struct LegacyCudaM1WidthSpaceBuildSpecLayout
{
  db::Coord width_distance;
  db::Coord spacing_distance;
  db::RegionCheckOptions width_options;
  db::RegionCheckOptions spacing_options;
  bool inputs_are_merged;
};

static_assert (
  sizeof (db::CudaM1WidthSpaceBuildSpec) ==
    sizeof (LegacyCudaM1WidthSpaceBuildSpecLayout),
  "CudaM1WidthSpaceBuildSpec public ABI size changed");
static_assert (
  alignof (db::CudaM1WidthSpaceBuildSpec) ==
    alignof (LegacyCudaM1WidthSpaceBuildSpecLayout),
  "CudaM1WidthSpaceBuildSpec public ABI alignment changed");

db::DeepLayer make_flat_m1 (db::DeepShapeStore &store)
{
  db::Region seed;
  seed.insert (db::Box (0, 0, 200, 100));
  seed.insert (db::Box (300, 0, 450, 100));
  return store.create_from_flat (seed, false);
}

} // anonymous namespace

TEST(1_FailClosedAndDigest)
{
  db::CudaM1WidthSpaceBuildSpec abi_spec;
  LegacyCudaM1WidthSpaceBuildSpecLayout legacy_abi_spec;
  const char *abi_base = reinterpret_cast<const char *> (&abi_spec);
  const char *legacy_base =
    reinterpret_cast<const char *> (&legacy_abi_spec);
  EXPECT_EQ (
    reinterpret_cast<const char *> (&abi_spec.width_distance) - abi_base,
    reinterpret_cast<const char *> (&legacy_abi_spec.width_distance) -
      legacy_base);
  EXPECT_EQ (
    reinterpret_cast<const char *> (&abi_spec.spacing_distance) - abi_base,
    reinterpret_cast<const char *> (&legacy_abi_spec.spacing_distance) -
      legacy_base);
  EXPECT_EQ (
    reinterpret_cast<const char *> (&abi_spec.width_options) - abi_base,
    reinterpret_cast<const char *> (&legacy_abi_spec.width_options) -
      legacy_base);
  EXPECT_EQ (
    reinterpret_cast<const char *> (&abi_spec.spacing_options) - abi_base,
    reinterpret_cast<const char *> (&legacy_abi_spec.spacing_options) -
      legacy_base);
  EXPECT_EQ (
    reinterpret_cast<const char *> (&abi_spec.inputs_are_merged) - abi_base,
    reinterpret_cast<const char *> (&legacy_abi_spec.inputs_are_merged) -
      legacy_base);

  db::DeepShapeStore store ("TOP", 0.0005);
  const db::DeepLayer metal1 = make_flat_m1 (store);

  db::CudaM1WidthSpaceBuildSpec spec;
  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaM1WidthSpaceScene scene;
  scene.flat_polygon_count = 17;
  std::string reason;

  //  Merged semantics must be asserted by the future integration site.
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, spec, limits, scene, &reason),
    false);
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (17));
  EXPECT_EQ (reason.empty (), false);

  spec.inputs_are_merged = true;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, spec, limits, scene, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (scene.cells.size (), size_t (1));
  EXPECT_EQ (scene.contexts.size (), size_t (1));
  EXPECT_EQ (scene.metal_contexts.size (), size_t (1));
  EXPECT_EQ (scene.polygons.size (), size_t (2));
  EXPECT_EQ (scene.edges.size (), size_t (8));
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (2));
  EXPECT_EQ (scene.flat_edge_count, uint64_t (8));
  EXPECT_EQ (scene.cells [0].polygon_count, uint32_t (2));
  EXPECT_EQ (scene.polygons [0].polygon_id, uint32_t (0));
  EXPECT_EQ (scene.polygons [1].polygon_id, uint32_t (1));

  for (size_t polygon_id = 0;
       polygon_id < scene.polygons.size (); ++polygon_id) {
    const db::CudaM1WidthSpacePolygon &polygon =
      scene.polygons [polygon_id];
    for (uint32_t local_edge = 0;
         local_edge < polygon.edge_count; ++local_edge) {
      const uint64_t edge_id = polygon.edge_begin + local_edge;
      const uint64_t next_id =
        local_edge + 1 == polygon.edge_count
          ? polygon.edge_begin
          : edge_id + 1;
      EXPECT_EQ (scene.edges [edge_id].x2, scene.edges [next_id].x1);
      EXPECT_EQ (scene.edges [edge_id].y2, scene.edges [next_id].y1);
    }
  }

  std::array<uint8_t, 32> recomputed;
  EXPECT_EQ (
    db::cuda_m1_width_space_scene_digest (scene, recomputed), true);
  EXPECT_EQ (recomputed == scene.digest, true);
  const std::array<uint8_t, 32> legacy_digest = {{
    0xdd, 0x86, 0xb0, 0xa8, 0x4f, 0xca, 0xc6, 0x19,
    0x04, 0x6c, 0x0a, 0x49, 0x86, 0x10, 0x4b, 0xc8,
    0xb5, 0x70, 0x01, 0x47, 0x3d, 0x6d, 0x0e, 0xf8,
    0x23, 0xf5, 0xbc, 0xc8, 0x66, 0xe8, 0x58, 0x3e
  }};
  EXPECT_EQ (scene.digest == legacy_digest, true);

  db::CudaM1WidthSpaceScene repeated;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, spec, limits, repeated, 0),
    true);
  EXPECT_EQ (repeated.digest == scene.digest, true);

  db::CudaM1WidthSpaceBuildSpec m2_spec;
  m2_spec.width_distance = 140;
  m2_spec.spacing_distance = 140;
  m2_spec.inputs_are_merged = true;
  db::CudaM1WidthSpaceScene m2_scene;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, m2_spec, limits, m2_scene, &reason),
    true);
  EXPECT_EQ (m2_scene.width_distance, int64_t (140));
  EXPECT_EQ (m2_scene.spacing_distance, int64_t (140));
  EXPECT_EQ (m2_scene.digest == scene.digest, false);

  m2_spec.width_distance = 130;
  m2_scene.flat_polygon_count = 31;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, m2_spec, limits, m2_scene, &reason),
    false);
  EXPECT_EQ (m2_scene.flat_polygon_count, uint64_t (31));
  m2_spec.width_distance = 141;
  m2_spec.spacing_distance = 141;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, m2_spec, limits, m2_scene, &reason),
    false);

  //  Even another layer in the same store/layout is not interchangeable.
  const db::DeepLayer other_layer = metal1.derived ();
  repeated.flat_polygon_count = 23;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, other_layer, spec, limits, repeated, &reason),
    false);
  EXPECT_EQ (repeated.flat_polygon_count, uint64_t (23));
}

TEST(2_HierarchyTransformsAndCapacity)
{
  db::DeepShapeStore store ("TOP", 0.0005);
  db::Region seed;
  seed.insert (db::Box (0, 0, 100, 100));
  db::DeepLayer metal1 = store.create_from_flat (seed, false);
  db::Layout &layout = metal1.layout ();
  db::Cell &top = metal1.initial_cell ();
  db::Cell &child = layout.cell (layout.add_cell ("CHILD"));
  child.shapes (metal1.layer ()).insert (db::Box (0, 0, 20, 10));
  top.insert (
    db::CellInstArray (
      db::CellInst (child.cell_index ()),
      db::Trans (1, false, db::Vector (1000, 2000)),
      db::Vector (100, 0), db::Vector (), 2, 1));

  db::CudaM1WidthSpaceBuildSpec spec;
  spec.inputs_are_merged = true;
  db::CudaM1WidthSpaceSceneLimits limits;
  db::CudaM1WidthSpaceScene scene;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, spec, limits, scene, 0),
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
  EXPECT_EQ (scene.scene_left, int64_t (0));
  EXPECT_EQ (scene.scene_bottom, int64_t (0));
  EXPECT_EQ (scene.scene_right, int64_t (1100));
  EXPECT_EQ (scene.scene_top, int64_t (2020));

  limits.max_flat_polygons = 2;
  scene.flat_polygon_count = 29;
  std::string reason;
  EXPECT_EQ (
    db::cuda_m1_width_space_build_scene (
      metal1, metal1, spec, limits, scene, &reason),
    false);
  EXPECT_EQ (scene.flat_polygon_count, uint64_t (29));
  EXPECT_EQ (reason.empty (), false);
}
