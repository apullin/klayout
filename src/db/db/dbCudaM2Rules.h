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

class DeepLayer;
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
 * Outcome and bounded telemetry for one raw-M2-to-flat-union transaction.
 *
 * Complete means only that the exact raw physical M2 scene was accepted by
 * the optional backend and that its validated boundary was materialized as
 * an owned merged flat Region.  It does not certify M2.4, VIA2, or any other
 * design rule.
 */
struct DB_PUBLIC CudaM2FlatUnionAttempt
{
  enum Disposition
  {
    Disabled,
    HostDeclined,
    BackendFallback,
    BackendError,
    InvalidResult,
    TopologyDeclined,
    Complete
  };

  CudaM2FlatUnionAttempt ();

  Disposition disposition;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t metal_context_count;
  uint64_t cell_count;
  uint64_t polygon_count;
  uint64_t edge_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  uint64_t rectangle_count;
  uint64_t x_slab_count;
  uint64_t membership_count;
  uint64_t event_count;
  uint64_t strip_interval_count;
  uint64_t raw_segment_count;
  uint64_t boundary_segment_count;
  uint64_t boundary_fnv64;
  uint64_t lowering_ns;
  uint64_t backend_ns;
  uint64_t materialize_ns;
  uint64_t live_total_ns;
  CudaM2FlatUnionStats flat_stats;
  std::string message;
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

/**
 * Try the complete optional raw-M2-to-flat-union host transaction.
 *
 * Backend capability is checked before "raw_metal2" is inspected or lowered.
 * The request uses the fixed, production-qualified bounded capacities of the
 * first exact Manhattan-union backend.  Every non-Complete outcome leaves
 * "flat_union" unchanged and requires the caller to use the pristine CPU
 * path.  Even Complete is a geometry result, not a design-rule certificate.
 */
DB_PUBLIC CudaM2FlatUnionAttempt cuda_m2_raw_manhattan_try_flat_union (
  const db::DeepLayer &raw_metal2, db::Region &flat_union,
  int32_t device = 0);

} // namespace db

#endif
