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

/** Return true when an enabled, loaded backend would accept this record count. */
DB_PUBLIC bool cuda_spatial_may_attempt (uint64_t subject_count,
                                         uint64_t intruder_count);

/** Return true only when the opt-in module was requested and loaded. */
DB_PUBLIC bool cuda_spatial_requested ();

} // namespace db

#endif
