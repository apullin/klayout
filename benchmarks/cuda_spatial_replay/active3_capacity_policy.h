/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef KLAYOUT_CUDA_ACTIVE3_CAPACITY_POLICY_H
#define KLAYOUT_CUDA_ACTIVE3_CAPACITY_POLICY_H

#include "dbCudaSpatialApi.h"

#include <cstdint>

namespace klayout_cuda {
namespace active3 {

inline bool uses_cartesian_pair_preflight(std::uint32_t opcode) {
  return opcode == KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY;
}

inline bool cartesian_pair_preflight_allows(
    std::uint32_t opcode, std::uint64_t well_edges,
    std::uint64_t active_edges, std::uint64_t maximum_pairs) {
  if (!uses_cartesian_pair_preflight(opcode)) {
    return true;
  }
  return well_edges != 0 && active_edges <= maximum_pairs / well_edges;
}

inline bool actual_candidate_capacity_exceeded(
    std::uint64_t candidates, std::uint64_t maximum_pairs) {
  return candidates > maximum_pairs;
}

}  // namespace active3
}  // namespace klayout_cuda

#endif
