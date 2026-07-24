/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef KLAYOUT_CUDA_IMPLANT12_EXACT_PREDICATE_H
#define KLAYOUT_CUDA_IMPLANT12_EXACT_PREDICATE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace klayout_cuda {
namespace implant12 {

inline constexpr std::int64_t kQualifiedSceneDbuPicometers = 500;
inline constexpr std::int64_t kImplant1RuleDistancePicometers = 70000;
inline constexpr std::int64_t kImplant2RuleDistancePicometers = 25000;
inline constexpr std::int64_t kImplant1Distance =
    kImplant1RuleDistancePicometers / kQualifiedSceneDbuPicometers;
inline constexpr std::int64_t kImplant2Distance =
    kImplant2RuleDistancePicometers / kQualifiedSceneDbuPicometers;
static_assert(kImplant1Distance == 140,
              "0.070 um / 0.0005 um must be 140 DBU");
static_assert(kImplant2Distance == 50,
              "0.025 um / 0.0005 um must be 50 DBU");

struct alignas(16) DirectedEdge {
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
};

struct alignas(16) EdgePair {
  DirectedEdge implant;
  DirectedEdge secondary;
};

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

bool classify_batch(const EdgePair *pairs, std::size_t count,
                    std::int64_t distance, Verdict *results,
                    std::string *error);

const char *verdict_name(Verdict verdict);

}  // namespace implant12
}  // namespace klayout_cuda

#endif
