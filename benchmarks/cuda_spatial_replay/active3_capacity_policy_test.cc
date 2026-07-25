/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "active3_capacity_policy.h"

#include <cstdint>
#include <iostream>
#include <limits>

namespace {

bool expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "ACTIVE.3 capacity policy failed: " << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  using klayout_cuda::active3::actual_candidate_capacity_exceeded;
  using klayout_cuda::active3::cartesian_pair_preflight_allows;

  bool good = true;
  good = expect(
             !cartesian_pair_preflight_allows(
                 KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY,
                 4, 4, 15),
             "legacy ACTIVE.3 must reject a 16-pair Cartesian bound at 15") &&
         good;
  good = expect(
             !cartesian_pair_preflight_allows(
                 KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY,
                 2, std::numeric_limits<std::uint64_t>::max(),
                 std::numeric_limits<std::uint64_t>::max()),
             "legacy ACTIVE.3 multiplication overflow must fail closed") &&
         good;
  good = expect(
             cartesian_pair_preflight_allows(
                 KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_BOTH_SUPERSET_EMPTY,
                 4, 4, 15),
             "raw-WELL ACTIVE.3 must bypass the Cartesian preflight") &&
         good;
  good = expect(
             cartesian_pair_preflight_allows(
                 KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_BOTH_SUPERSET_EMPTY,
                 std::numeric_limits<std::uint64_t>::max(),
                 std::numeric_limits<std::uint64_t>::max(), 1),
             "raw-WELL Cartesian overflow is telemetry, not rejection") &&
         good;
  good = expect(
             !actual_candidate_capacity_exceeded(15, 15),
             "an exact actual-candidate limit must be accepted") &&
         good;
  good = expect(
             actual_candidate_capacity_exceeded(16, 15),
             "raw-WELL actual candidates above the limit must fail closed") &&
         good;

  if (good) {
    std::cout << "ACTIVE.3 capacity policy passed: legacy Cartesian cap, "
                 "raw-WELL actual-candidate cap\n";
  }
  return good ? 0 : 1;
}
