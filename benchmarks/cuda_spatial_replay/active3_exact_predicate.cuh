/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef KLAYOUT_CUDA_ACTIVE3_EXACT_PREDICATE_CUH
#define KLAYOUT_CUDA_ACTIVE3_EXACT_PREDICATE_CUH

#include "active3_exact_predicate.h"

#include <cstdint>

namespace klayout_cuda {
namespace active3 {
namespace detail {

constexpr std::uint64_t kSignBit = UINT64_C(1) << 63;

__host__ __device__ inline std::uint64_t ordered_key(std::int64_t value) {
  // int64 -> uint64 is defined modulo 2^64.  Flipping the sign bit then maps
  // the entire signed range monotonically onto [0, 2^64), so subtraction of
  // ordered keys gives an overflow-free mathematical coordinate difference.
  return static_cast<std::uint64_t>(value) ^ kSignBit;
}

__host__ __device__ inline std::uint64_t coordinate_gap(std::int64_t low,
                                                        std::int64_t high) {
  return ordered_key(high) - ordered_key(low);
}

__host__ __device__ inline std::int64_t minimum(std::int64_t a,
                                                std::int64_t b) {
  return a < b ? a : b;
}

__host__ __device__ inline std::int64_t maximum(std::int64_t a,
                                                std::int64_t b) {
  return a > b ? a : b;
}

__host__ __device__ inline std::uint64_t interval_gap(
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

__host__ __device__ inline bool intervals_touch(
    std::int64_t a0, std::int64_t a1, std::int64_t b0, std::int64_t b1) {
  const std::int64_t alo = minimum(a0, a1);
  const std::int64_t ahi = maximum(a0, a1);
  const std::int64_t blo = minimum(b0, b1);
  const std::int64_t bhi = maximum(b0, b1);
  return !(ahi < blo || bhi < alo);
}

__host__ __device__ inline bool source_coordinate_differences_are_safe(
    const EdgePair &pair) {
  std::int64_t xmin = pair.well.x1;
  std::int64_t xmax = pair.well.x1;
  std::int64_t ymin = pair.well.y1;
  std::int64_t ymax = pair.well.y1;
  const std::int64_t xs[3] = {pair.well.x2, pair.active.x1,
                              pair.active.x2};
  const std::int64_t ys[3] = {pair.well.y2, pair.active.y1,
                              pair.active.y2};
  for (int i = 0; i < 3; ++i) {
    xmin = minimum(xmin, xs[i]);
    xmax = maximum(xmax, xs[i]);
    ymin = minimum(ymin, ys[i]);
    ymax = maximum(ymax, ys[i]);
  }
  // KLayout's relation source forms signed Coord differences before several
  // wider/double conversions.  Decline pairs whose mathematical difference
  // would overflow int64_t rather than certifying behavior outside that
  // source domain.
  constexpr std::uint64_t kMaximumSignedDifference =
      static_cast<std::uint64_t>(INT64_MAX);
  return coordinate_gap(xmin, xmax) <= kMaximumSignedDifference &&
         coordinate_gap(ymin, ymax) <= kMaximumSignedDifference;
}

}  // namespace detail

// Stable per-pair hook for a resident CUDA pipeline.  This performs no
// allocation and materializes no candidate list.  kNoViolation and
// kViolation are exact; kUncertain must remain on/fall back to the CPU path.
__host__ __device__ inline Verdict classify_pair_bounded(
    const EdgePair &pair, std::int64_t distance) {
  const DirectedEdge &a = pair.well;
  const DirectedEdge &b = pair.active;

  const bool a_horizontal = a.y1 == a.y2 && a.x1 != a.x2;
  const bool a_vertical = a.x1 == a.x2 && a.y1 != a.y2;
  const bool b_horizontal = b.y1 == b.y2 && b.x1 != b.x2;
  const bool b_vertical = b.x1 == b.x2 && b.y1 != b.y2;

  // This first proof deliberately has one auditable exact domain.  Dots and
  // diagonal edges are legal in general KLayout geometry, so they fall back.
  if (!(a_horizontal || a_vertical) || !(b_horizontal || b_vertical)) {
    return Verdict::kUncertain;
  }
  if (!detail::source_coordinate_differences_are_safe(pair) ||
      (distance != kQualifiedSceneCoordinateDistance &&
       distance != kContact4QualifiedSceneCoordinateDistance)) {
    return Verdict::kUncertain;
  }
  const std::uint64_t d = static_cast<std::uint64_t>(distance);

  // EdgeRelationFilter's OverlapRelation swaps the well edge for its
  // ignore-angle test.  At exactly 90 degrees the source accepts precisely
  // dot(original_well, original_active) > 0.  Thus perpendicular and
  // oppositely-directed Manhattan pairs are exact non-matches.
  if (a_horizontal != b_horizontal) {
    return Verdict::kNoViolation;
  }

  bool same_direction = false;
  bool active_is_right = false;
  bool collinear = false;
  std::uint64_t perpendicular_gap = 0;
  std::uint64_t projection_gap = 0;
  bool projections_touch = false;

  if (a_horizontal) {
    const bool a_positive = a.x2 > a.x1;
    const bool b_positive = b.x2 > b.x1;
    same_direction = a_positive == b_positive;
    if (!same_direction) {
      return Verdict::kNoViolation;
    }
    collinear = a.y1 == b.y1;
    // "Right" of an eastbound edge is south; right of a westbound edge is
    // north.  This is the half-plane retained by euclidian_near_part_of_edge
    // after OverlapRelation reverses the active edge.
    active_is_right = a_positive ? b.y1 < a.y1 : b.y1 > a.y1;
    perpendicular_gap =
        a.y1 < b.y1 ? detail::coordinate_gap(a.y1, b.y1)
                    : detail::coordinate_gap(b.y1, a.y1);
    projection_gap = detail::interval_gap(a.x1, a.x2, b.x1, b.x2);
    projections_touch = detail::intervals_touch(a.x1, a.x2, b.x1, b.x2);
  } else {
    const bool a_positive = a.y2 > a.y1;
    const bool b_positive = b.y2 > b.y1;
    same_direction = a_positive == b_positive;
    if (!same_direction) {
      return Verdict::kNoViolation;
    }
    collinear = a.x1 == b.x1;
    // "Right" of a northbound edge is east; right of a southbound edge is
    // west.
    active_is_right = a_positive ? b.x1 > a.x1 : b.x1 < a.x1;
    perpendicular_gap =
        a.x1 < b.x1 ? detail::coordinate_gap(a.x1, b.x1)
                    : detail::coordinate_gap(b.x1, a.x1);
    projection_gap = detail::interval_gap(a.y1, a.y2, b.y1, b.y2);
    projections_touch = detail::intervals_touch(a.y1, a.y2, b.y1, b.y2);
  }

  if (collinear) {
    // IncludeZeroDistanceWhenTouching changes the source's half-plane
    // threshold only when the collinear finite segments intersect.  A
    // collinear gap is therefore not an enclosing violation even if the two
    // closest endpoints are less than d apart.
    return projections_touch ? Verdict::kViolation
                             : Verdict::kNoViolation;
  }
  if (!active_is_right) {
    return Verdict::kNoViolation;
  }

  // The two source near-part calls are nonempty exactly when the minimum
  // Euclidean distance between these parallel finite segments is strictly
  // less than d.  For Manhattan parallels, that squared distance is the sum
  // of the perpendicular-line gap and the one-dimensional projection gap.
  if (perpendicular_gap >= d || projection_gap >= d) {
    return Verdict::kNoViolation;
  }
  const std::uint64_t distance_squared =
      perpendicular_gap * perpendicular_gap + projection_gap * projection_gap;
  const std::uint64_t threshold_squared = d * d;
  return distance_squared < threshold_squared ? Verdict::kViolation
                                              : Verdict::kNoViolation;
}

}  // namespace active3
}  // namespace klayout_cuda

#endif
