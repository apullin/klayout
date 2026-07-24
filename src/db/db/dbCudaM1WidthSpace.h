/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaM1WidthSpace
#define HDR_dbCudaM1WidthSpace

#include "dbCommon.h"
#include "dbRegionLocalOperations.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace db
{

class DeepLayer;

/**
 * Capacity limits for speculative M1 width/space scene construction.
 *
 * These limits protect only host lowering.  A future backend must apply its
 * own checked device-memory, grid-membership, and pair-work limits.
 */
struct DB_PUBLIC CudaM1WidthSpaceSceneLimits
{
  uint64_t max_cells;
  uint64_t max_contexts;
  uint64_t max_stored_polygons;
  uint64_t max_stored_edges;
  uint64_t max_flat_polygons;
  uint64_t max_flat_edges;

  CudaM1WidthSpaceSceneLimits ();
};

/**
 * The exact rule shape accepted by the first FreePDK45 scene builder.
 *
 * "inputs_are_merged" is an explicit assertion from the future integration
 * site.  DeepLayer itself does not retain a queryable merged-semantics bit.
 */
struct DB_PUBLIC CudaM1WidthSpaceBuildSpec
{
  db::Coord width_distance;
  db::Coord spacing_distance;
  db::RegionCheckOptions width_options;
  db::RegionCheckOptions spacing_options;
  bool inputs_are_merged;

  CudaM1WidthSpaceBuildSpec ();
};

/**
 * One expanded occurrence of a source cell.
 *
 * The transform code is db::Trans::rot(): the eight orthogonal rotations and
 * reflections.  Translation is in layout DBU.  The record is pointer-free.
 */
struct DB_PUBLIC CudaM1WidthSpaceContext
{
  int64_t tx;
  int64_t ty;
  uint32_t cell_id;
  uint32_t transform_code;
};

/**
 * One source cell and its contiguous polygon/edge ranges.
 */
struct DB_PUBLIC CudaM1WidthSpaceCell
{
  uint64_t source_cell_index;
  uint64_t polygon_begin;
  uint64_t edge_begin;
  uint32_t polygon_count;
  uint32_t edge_count;
};

/**
 * One source-cell polygon.
 *
 * polygon_id is local to the owning cell.  Its directed contour edges occupy
 * [edge_begin, edge_begin + edge_count) in source contour order.
 */
struct DB_PUBLIC CudaM1WidthSpacePolygon
{
  uint64_t edge_begin;
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
  uint32_t polygon_id;
  uint32_t edge_count;
};

/**
 * One directed local-coordinate contour edge.
 */
struct DB_PUBLIC CudaM1WidthSpaceEdge
{
  int64_t x1;
  int64_t y1;
  int64_t x2;
  int64_t y2;
};

/**
 * Owning host representation of a future fused M1 width/spacing request.
 *
 * Vector storage is owned here, but every serialized record is pointer-free
 * and refers to other records only through stable indices and ranges.
 */
struct DB_PUBLIC CudaM1WidthSpaceScene
{
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t reserved;
  int64_t width_distance;
  int64_t spacing_distance;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  std::vector<CudaM1WidthSpaceContext> contexts;
  std::vector<uint32_t> metal_contexts;
  std::vector<uint64_t> context_polygon_offsets;
  std::vector<uint64_t> context_edge_offsets;
  std::vector<CudaM1WidthSpaceCell> cells;
  std::vector<CudaM1WidthSpacePolygon> polygons;
  std::vector<CudaM1WidthSpaceEdge> edges;
  std::array<uint8_t, 32> digest;

  CudaM1WidthSpaceScene ();
  void swap (CudaM1WidthSpaceScene &other) noexcept;
};

/**
 * Compute the canonical digest of a structurally valid scene.
 *
 * Fields are hashed explicitly in little-endian form, so padding and host
 * object addresses cannot affect the result.  False means the scene is
 * structurally inconsistent and must not be published.
 */
DB_PUBLIC bool cuda_m1_width_space_scene_digest (
  const CudaM1WidthSpaceScene &scene, std::array<uint8_t, 32> &digest);

/**
 * Build the narrowly qualified merged-M1 scene.
 *
 * The two logical sources must be the identical DeepLayer (same store,
 * layout, layout index, top cell, and layer), with no breakout cells.  This
 * mirrors the two nodes of the existing width/space batch while serializing
 * the geometry once.  On any unsupported input, overflow, capacity excess, or
 * exception, false is returned and "scene" is left unchanged.
 *
 * No backend is loaded or called by this seam.
 */
DB_PUBLIC bool cuda_m1_width_space_build_scene (
  const db::DeepLayer &width_metal1,
  const db::DeepLayer &spacing_metal1,
  const CudaM1WidthSpaceBuildSpec &spec,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaM1WidthSpaceScene &scene,
  std::string *decline_reason = 0);

} // namespace db

#endif
