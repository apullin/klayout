/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef KLAYOUT_CUDA_ACTIVE3_EXACT_PREDICATE_H
#define KLAYOUT_CUDA_ACTIVE3_EXACT_PREDICATE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace klayout_cuda {
namespace active3 {

// ACTIVE.3 is a 55 nm physical rule.  Both qualified KACTSCN1 captures use
// dbu=0.0005 um=0.5 nm, so the exact integer coordinate threshold is 110.
// Keeping all three values named prevents confusing nanometres with DBU.
inline constexpr std::int64_t kActive3RuleDistancePicometers = 55000;
inline constexpr std::int64_t kQualifiedSceneDbuPicometers = 500;
static_assert(kActive3RuleDistancePicometers %
                      kQualifiedSceneDbuPicometers ==
                  0,
              "qualified ACTIVE.3 distance must be integral in scene DBU");
inline constexpr std::int64_t kQualifiedSceneCoordinateDistance =
    kActive3RuleDistancePicometers / kQualifiedSceneDbuPicometers;
static_assert(kQualifiedSceneCoordinateDistance == 110,
              "0.055 um / 0.0005 um must be 110 DBU");

// A directed edge in the already-resolved hierarchy/context coordinate system.
struct alignas(16) DirectedEdge {
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
};

struct alignas(16) EdgePair {
  DirectedEdge well;
  DirectedEdge active;
};

// Fail-closed result for the fixed ACTIVE.3 relation:
//
//   well.enclosing(active, d, Euclidian, ignore_angle=90,
//                  whole_edges=false,
//                  IncludeZeroDistanceWhenTouching)
//
// The first accepted configuration is deliberately d=110 DBU, obtained
// exactly from the qualified scene's 0.0005-um DBU.  Any other distance is
// kUncertain, not an attempted generalization.
//
// kNoViolation and kViolation are exact.  kUncertain means that this bounded
// implementation deliberately declined the pair; callers must retain it for
// the existing CPU path and must never use it as part of a clean certificate.
enum class Verdict : std::uint8_t {
  kNoViolation = 0,
  kViolation = 1,
  kUncertain = 2,
};

static_assert(std::is_trivially_copyable<DirectedEdge>::value,
              "device edges must remain trivially copyable");
static_assert(std::is_trivially_copyable<EdgePair>::value,
              "device edge pairs must remain trivially copyable");
static_assert(sizeof(DirectedEdge) == 32, "unexpected edge padding");
static_assert(sizeof(EdgePair) == 64, "unexpected pair padding");

// Classifies a batch on the current CUDA device.  This owns only transient
// allocations; a future resident scene pipeline can invoke the same device
// classifier without these host transfers.
bool classify_batch(const EdgePair *pairs, std::size_t count,
                    std::int64_t distance, Verdict *results,
                    std::string *error);

const char *verdict_name(Verdict verdict);

}  // namespace active3
}  // namespace klayout_cuda

#endif
