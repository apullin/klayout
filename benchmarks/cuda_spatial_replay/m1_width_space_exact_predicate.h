/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef KLAYOUT_CUDA_M1_WIDTH_SPACE_EXACT_PREDICATE_H
#define KLAYOUT_CUDA_M1_WIDTH_SPACE_EXACT_PREDICATE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#if defined(__CUDACC__)
#define KLAYOUT_CUDA_M1_HD __host__ __device__
#else
#define KLAYOUT_CUDA_M1_HD
#endif

namespace klayout_cuda {
namespace m1_width_space {

// FreePDK45 METAL1.1 and METAL1.2 are both strict 65-nm rules.  Qualified
// KACTSCN1 scenes use dbu=0.0005 um=0.5 nm, making their integer threshold
// exactly 130 DBU.  A caller with any other DBU or rule distance must keep the
// existing CPU path.
inline constexpr std::int64_t kM1RuleDistancePicometers = 65000;
inline constexpr std::int64_t kQualifiedSceneDbuPicometers = 500;
static_assert(kM1RuleDistancePicometers % kQualifiedSceneDbuPicometers == 0,
              "qualified M1 distance must be integral in scene DBU");
inline constexpr std::int64_t kQualifiedSceneCoordinateDistance =
    kM1RuleDistancePicometers / kQualifiedSceneDbuPicometers;
static_assert(kQualifiedSceneCoordinateDistance == 130,
              "0.065 um / 0.0005 um must be 130 DBU");

// Polygon IDs must identify merged polygons in the fully resolved hierarchy
// context.  UINT64_MAX is reserved so missing topology metadata fails closed.
inline constexpr std::uint64_t kUnknownPolygonId = UINT64_MAX;

// A directed contour edge.  The bounded predicate assumes the same clockwise
// contour convention used by KLayout's polygon edge scanner: polygon interior
// is on the right of an edge and exterior is on the left.
struct alignas(16) DirectedEdge {
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
};

enum class Rule : std::uint8_t {
  kWidth = 1,
  kSpace = 2,
};

// One candidate emitted by a same-layer edge broad phase.  WidthRelation is a
// same-polygon check.  SpaceRelation checks both same-polygon notches and
// different-polygon spacing, matching edges_considered(false, false, ...).
struct alignas(16) CandidatePair {
  DirectedEdge first;
  DirectedEdge second;
  std::uint64_t first_polygon_id;
  std::uint64_t second_polygon_id;
  Rule rule;
};

// This is the result of the EdgeRelationFilter pair predicate before the
// scene-level shielding pass:
//
//   EdgeRelationFilter(rule, 130, Euclidian, ignore_angle=90,
//                      min_projection=0, max_projection=infinity,
//                      IncludeZeroDistanceWhenTouching)
//
// kNoViolation and kViolation are exact within the qualified domain.
// kViolation blocks a clean certificate, although a later shielding pass may
// remove its marker.  kUncertain also blocks certification and must retain the
// candidate for the CPU path.
enum class Verdict : std::uint8_t {
  kNoViolation = 0,
  kViolation = 1,
  kUncertain = 2,
};

static_assert(std::is_trivially_copyable<DirectedEdge>::value,
              "device edges must remain trivially copyable");
static_assert(std::is_trivially_copyable<CandidatePair>::value,
              "device candidates must remain trivially copyable");
static_assert(sizeof(DirectedEdge) == 32, "unexpected edge padding");
static_assert(sizeof(CandidatePair) == 96, "unexpected candidate padding");

namespace detail {

inline constexpr std::uint64_t kSignBit = UINT64_C(1) << 63;

KLAYOUT_CUDA_M1_HD inline std::uint64_t ordered_key(std::int64_t value) {
  // Flipping the sign bit maps the full signed range monotonically onto the
  // unsigned range.  Differences between ordered keys therefore cannot invoke
  // signed overflow.
  return static_cast<std::uint64_t>(value) ^ kSignBit;
}

KLAYOUT_CUDA_M1_HD inline std::uint64_t coordinate_gap(std::int64_t a,
                                                       std::int64_t b) {
  const std::uint64_t ak = ordered_key(a);
  const std::uint64_t bk = ordered_key(b);
  return ak < bk ? bk - ak : ak - bk;
}

KLAYOUT_CUDA_M1_HD inline std::int64_t minimum(std::int64_t a,
                                               std::int64_t b) {
  return a < b ? a : b;
}

KLAYOUT_CUDA_M1_HD inline std::int64_t maximum(std::int64_t a,
                                               std::int64_t b) {
  return a > b ? a : b;
}

KLAYOUT_CUDA_M1_HD inline bool intervals_touch(std::int64_t a0,
                                               std::int64_t a1,
                                               std::int64_t b0,
                                               std::int64_t b1) {
  const std::int64_t alo = minimum(a0, a1);
  const std::int64_t ahi = maximum(a0, a1);
  const std::int64_t blo = minimum(b0, b1);
  const std::int64_t bhi = maximum(b0, b1);
  return !(ahi < blo || bhi < alo);
}

KLAYOUT_CUDA_M1_HD inline std::uint64_t interval_gap(
    std::int64_t a0, std::int64_t a1, std::int64_t b0, std::int64_t b1) {
  const std::int64_t alo = minimum(a0, a1);
  const std::int64_t ahi = maximum(a0, a1);
  const std::int64_t blo = minimum(b0, b1);
  const std::int64_t bhi = maximum(b0, b1);
  if (ahi < blo) {
    return coordinate_gap(ahi, blo);
  }
  if (bhi < alo) {
    return coordinate_gap(bhi, alo);
  }
  return 0;
}

KLAYOUT_CUDA_M1_HD inline bool source_coordinate_differences_are_safe(
    const CandidatePair &pair) {
  std::int64_t xmin = pair.first.x1;
  std::int64_t xmax = pair.first.x1;
  std::int64_t ymin = pair.first.y1;
  std::int64_t ymax = pair.first.y1;
  const std::int64_t xs[3] = {
      pair.first.x2, pair.second.x1, pair.second.x2};
  const std::int64_t ys[3] = {
      pair.first.y2, pair.second.y1, pair.second.y2};
  for (int i = 0; i < 3; ++i) {
    xmin = minimum(xmin, xs[i]);
    xmax = maximum(xmax, xs[i]);
    ymin = minimum(ymin, ys[i]);
    ymax = maximum(ymax, ys[i]);
  }

  // KLayout forms signed Coord differences in its edge implementation.  Do
  // not certify pairs whose mathematical difference cannot be represented in
  // int64_t, even though the bounded implementation itself could continue.
  constexpr std::uint64_t kMaximumSignedDifference =
      static_cast<std::uint64_t>(INT64_MAX);
  return coordinate_gap(xmin, xmax) <= kMaximumSignedDifference &&
         coordinate_gap(ymin, ymax) <= kMaximumSignedDifference;
}

KLAYOUT_CUDA_M1_HD inline bool is_horizontal(const DirectedEdge &edge) {
  return edge.y1 == edge.y2 && edge.x1 != edge.x2;
}

KLAYOUT_CUDA_M1_HD inline bool is_vertical(const DirectedEdge &edge) {
  return edge.x1 == edge.x2 && edge.y1 != edge.y2;
}

// True when point/edge b lies strictly on the right-hand side of directed
// axial edge a.  The caller has already established that both edges are
// parallel, so testing either endpoint of b is sufficient.
KLAYOUT_CUDA_M1_HD inline bool second_is_right_of_first(
    const DirectedEdge &a, const DirectedEdge &b, bool horizontal) {
  if (horizontal) {
    return a.x2 > a.x1 ? b.y1 < a.y1 : b.y1 > a.y1;
  }
  return a.y2 > a.y1 ? b.x1 > a.x1 : b.x1 < a.x1;
}

}  // namespace detail

// Allocation-free hook for host and resident CUDA pipelines.  The exact
// domain is deliberately limited to nondegenerate Manhattan contour edges.
// Diagonal/dot geometry, unknown topology, unsafe source-coordinate spans,
// and any non-FreePDK45 distance return kUncertain.
KLAYOUT_CUDA_M1_HD inline Verdict classify_pair_bounded(
    const CandidatePair &pair, std::int64_t distance) {
  if (distance != kQualifiedSceneCoordinateDistance ||
      pair.first_polygon_id == kUnknownPolygonId ||
      pair.second_polygon_id == kUnknownPolygonId ||
      !detail::source_coordinate_differences_are_safe(pair)) {
    return Verdict::kUncertain;
  }

  if (pair.rule != Rule::kWidth && pair.rule != Rule::kSpace) {
    return Verdict::kUncertain;
  }

  const bool first_horizontal = detail::is_horizontal(pair.first);
  const bool first_vertical = detail::is_vertical(pair.first);
  const bool second_horizontal = detail::is_horizontal(pair.second);
  const bool second_vertical = detail::is_vertical(pair.second);
  if (!(first_horizontal || first_vertical) ||
      !(second_horizontal || second_vertical)) {
    return Verdict::kUncertain;
  }

  // SinglePolygonCheck never presents different polygons to WidthRelation.
  if (pair.rule == Rule::kWidth &&
      pair.first_polygon_id != pair.second_polygon_id) {
    return Verdict::kNoViolation;
  }

  // EdgeRelationFilter at ignore_angle=90 accepts only a strictly negative
  // scalar product.  Perpendicular Manhattan edges have product zero.
  if (first_horizontal != second_horizontal) {
    return Verdict::kNoViolation;
  }

  const bool first_positive =
      first_horizontal ? pair.first.x2 > pair.first.x1
                       : pair.first.y2 > pair.first.y1;
  const bool second_positive =
      second_horizontal ? pair.second.x2 > pair.second.x1
                        : pair.second.y2 > pair.second.y1;
  if (first_positive == second_positive) {
    return Verdict::kNoViolation;
  }

  const bool collinear =
      first_horizontal ? pair.first.y1 == pair.second.y1
                       : pair.first.x1 == pair.second.x1;
  const bool projections_touch =
      first_horizontal
          ? detail::intervals_touch(pair.first.x1, pair.first.x2,
                                    pair.second.x1, pair.second.x2)
          : detail::intervals_touch(pair.first.y1, pair.first.y2,
                                    pair.second.y1, pair.second.y2);

  if (collinear) {
    // IncludeZeroDistanceWhenTouching admits collinear overlap or an endpoint
    // touch.  For disjoint collinear edges, include_zero_flag is false and
    // euclidian_near_part_of_edge rejects the on-line side before distance.
    return projections_touch ? Verdict::kViolation
                             : Verdict::kNoViolation;
  }

  const bool second_on_right = detail::second_is_right_of_first(
      pair.first, pair.second, first_horizontal);
  // WidthRelation keeps the original orientations (inside/right sides face).
  // SpaceRelation reverses both edges (outside/left sides face).
  const bool sides_face =
      pair.rule == Rule::kWidth ? second_on_right : !second_on_right;
  if (!sides_face) {
    return Verdict::kNoViolation;
  }

  const std::uint64_t perpendicular_gap =
      first_horizontal
          ? detail::coordinate_gap(pair.first.y1, pair.second.y1)
          : detail::coordinate_gap(pair.first.x1, pair.second.x1);
  const std::uint64_t projection_gap =
      first_horizontal
          ? detail::interval_gap(pair.first.x1, pair.first.x2,
                                 pair.second.x1, pair.second.x2)
          : detail::interval_gap(pair.first.y1, pair.first.y2,
                                 pair.second.y1, pair.second.y2);
  constexpr std::uint64_t kDistance =
      static_cast<std::uint64_t>(kQualifiedSceneCoordinateDistance);

  // The source uses a strict distance test.  Checking each component first
  // keeps the subsequent squares tiny and overflow-free.
  if (perpendicular_gap >= kDistance || projection_gap >= kDistance) {
    return Verdict::kNoViolation;
  }
  const std::uint64_t squared_distance =
      perpendicular_gap * perpendicular_gap +
      projection_gap * projection_gap;
  return squared_distance < kDistance * kDistance
             ? Verdict::kViolation
             : Verdict::kNoViolation;
}

// Transient CUDA batch wrapper for focused parity tests.  A future scene
// pipeline should invoke classify_pair_bounded directly on resident data.
bool classify_batch(const CandidatePair *pairs, std::size_t count,
                    std::int64_t distance, Verdict *results,
                    std::string *error);

const char *verdict_name(Verdict verdict);
const char *rule_name(Rule rule);

}  // namespace m1_width_space
}  // namespace klayout_cuda

#undef KLAYOUT_CUDA_M1_HD

#endif
