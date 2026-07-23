/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaSpatialBackend
#define HDR_dbCudaSpatialBackend

#include "dbCommon.h"
#include "dbCudaSpatialApi.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace db
{

struct DB_PUBLIC CudaSpatialAttempt
{
  enum Disposition
  {
    Disabled,
    BelowThreshold,
    Success,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaSpatialAttempt ();

  Disposition disposition;
  uint32_t fallback_flags;
  uint64_t membership_count;
  uint64_t occupied_cell_count;
  uint64_t pair_work_count;
  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t broad_phase_ns;
  uint64_t sort_unique_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  std::string message;
  std::vector<uint64_t> pair_keys;
};

struct DB_PUBLIC CudaActive3Attempt
{
  enum Disposition
  {
    Disabled,
    CertifiedEmpty,
    RawHits,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaActive3Attempt ();

  Disposition disposition;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t well_context_count;
  uint64_t active_context_count;
  uint64_t cell_count;
  uint64_t edge_count;
  uint64_t flat_well_edge_count;
  uint64_t flat_active_edge_count;
  uint64_t grid_cell_count;
  uint64_t membership_count;
  uint64_t candidate_pair_count;
  uint64_t raw_hit_count;
  uint64_t uncertain_count;
  uint64_t total_ns;
  std::string message;
};

/**
 * Try the optional CUDA bipartite broad phase.
 *
 * The backend is disabled unless KLAYOUT_CUDA_SPATIAL_BACKEND is set.  Any
 * loader, capacity, CUDA, or result-validation failure is represented by a
 * non-Success disposition and is expected to fall back to the CPU scanner.
 */
DB_PUBLIC CudaSpatialAttempt cuda_spatial_try_bipartite (
  const std::vector<klayout_cuda_spatial_aabb_v1> &subjects,
  const std::vector<klayout_cuda_spatial_aabb_v1> &intruders,
  int64_t enlargement);

/**
 * Try the optional CUDA self-AABB broad phase.
 *
 * Returned keys contain two distinct one-based record IDs in ascending order.
 * A backend without the optional self entry point fails closed to the caller.
 */
DB_PUBLIC CudaSpatialAttempt cuda_spatial_try_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement);

/** Return true when an enabled, loaded backend would accept this record count. */
DB_PUBLIC bool cuda_spatial_may_attempt (uint64_t subject_count,
                                         uint64_t intruder_count);

/** Return true when the optional self entry point accepts this record count. */
DB_PUBLIC bool cuda_spatial_may_attempt_self (uint64_t record_count);

/**
 * Check that a self-AABB request fits the configured per-record and aggregate
 * membership limits without launching the backend.
 *
 * On success, membership_count is the exact number of grid memberships the
 * backend will allocate for these records.  Coordinate, grid-span, or capacity
 * uncertainty fails closed.
 */
DB_PUBLIC bool cuda_spatial_preflight_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement, uint64_t &membership_count);

/** Return true only when the opt-in module was requested and loaded. */
DB_PUBLIC bool cuda_spatial_requested ();

/**
 * Invoke the optional live ACTIVE.3 raw-superset empty certificate.
 *
 * Only CertifiedEmpty is usable by a caller.  RawHits deliberately carries
 * no KLayout markers and requests the unchanged CPU implementation.
 */
DB_PUBLIC CudaActive3Attempt cuda_spatial_try_active3_empty (
  const klayout_cuda_spatial_active3_request_v1 &request);

/** Return true only when the independent ACTIVE.3 opt-in and symbol exist. */
DB_PUBLIC bool cuda_spatial_active3_requested ();

} // namespace db

#endif
