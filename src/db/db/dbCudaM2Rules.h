/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaM2Rules
#define HDR_dbCudaM2Rules

#include "dbCommon.h"
#include "dbCudaSpatialApi.h"

#include <stdint.h>
#include <string>

namespace db
{

class Region;

/**
 * Checked census from a canonical M2-union boundary materialization.
 */
struct DB_PUBLIC CudaM2FlatUnionStats
{
  CudaM2FlatUnionStats ();

  uint64_t segment_count;
  uint64_t contour_count;
  uint64_t vertex_count;
  uint64_t max_vertices;
};

/**
 * Stitch a canonical exact M2-union boundary into an owned flat Region.
 *
 * The boundary must first satisfy the loader's exact canonical ordering and
 * FNV-1a contract.  This function independently checks global Manhattan
 * topology, coordinate range, closed simple clockwise contours, and KLayout
 * polygon census conservation.  On success the replacement Region has
 * merged semantics and is already merged.
 *
 * Every failure leaves both "flat_union" and "stats" unchanged.  This helper
 * does not certify any design rule; in particular, it is not an M2.4/VIA2
 * integration hook.
 */
DB_PUBLIC bool cuda_m2_union_boundary_to_flat_region (
  const klayout_cuda_spatial_m2_union_segment_v1 *segments,
  uint64_t segment_count, uint64_t expected_fnv64,
  db::Region &flat_union, CudaM2FlatUnionStats *stats = 0,
  std::string *decline_reason = 0);

} // namespace db

#endif
