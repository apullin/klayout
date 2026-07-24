/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef KLAYOUT_CUDA_IMPLANT12_EXACT_PREDICATE_CUH
#define KLAYOUT_CUDA_IMPLANT12_EXACT_PREDICATE_CUH

#include "implant12_exact_predicate.h"

#include <cstdint>

namespace klayout_cuda {
namespace implant12 {
namespace detail {

constexpr std::uint64_t kSignBit = UINT64_C(1) << 63;

__host__ __device__ inline std::uint64_t ordered_key(std::int64_t value) {
  return static_cast<std::uint64_t>(value) ^ kSignBit;
}

__host__ __device__ inline std::uint64_t coordinate_gap(std::int64_t a,
                                                        std::int64_t b) {
  const std::uint64_t ak = ordered_key(a);
  const std::uint64_t bk = ordered_key(b);
  return ak < bk ? bk - ak : ak - bk;
}

__host__ __device__ inline std::int64_t minimum(std::int64_t a,
                                                std::int64_t b) {
  return a < b ? a : b;
}

__host__ __device__ inline std::int64_t maximum(std::int64_t a,
                                                std::int64_t b) {
  return a > b ? a : b;
}

__host__ __device__ inline bool source_differences_are_safe(
    const EdgePair &pair) {
  std::int64_t xmin = pair.implant.x1;
  std::int64_t xmax = pair.implant.x1;
  std::int64_t ymin = pair.implant.y1;
  std::int64_t ymax = pair.implant.y1;
  const std::int64_t xs[3] = {
      pair.implant.x2, pair.secondary.x1, pair.secondary.x2};
  const std::int64_t ys[3] = {
      pair.implant.y2, pair.secondary.y1, pair.secondary.y2};
  for (int i = 0; i < 3; ++i) {
    xmin = minimum(xmin, xs[i]);
    xmax = maximum(xmax, xs[i]);
    ymin = minimum(ymin, ys[i]);
    ymax = maximum(ymax, ys[i]);
  }
  constexpr std::uint64_t kMaximumSignedDifference =
      static_cast<std::uint64_t>(INT64_MAX);
  return coordinate_gap(xmin, xmax) <= kMaximumSignedDifference &&
         coordinate_gap(ymin, ymax) <= kMaximumSignedDifference;
}

__host__ __device__ inline bool intervals_overlap_strictly(
    std::int64_t a0, std::int64_t a1,
    std::int64_t b0, std::int64_t b1) {
  const std::int64_t low =
      maximum(minimum(a0, a1), minimum(b0, b1));
  const std::int64_t high =
      minimum(maximum(a0, a1), maximum(b0, b1));
  return low < high;
}

}  // namespace detail

/*
 * Exact bounded classifier for:
 *
 *   implant.separation(secondary, d, Projection,
 *                      ignore_angle=90, whole_edges=false,
 *                      IncludeZeroDistanceWhenTouching)
 *
 * KLayout's SpaceRelation keeps antiparallel edges, reverses both operands,
 * and applies projected_near_part_of_edge in both directions.  For
 * nondegenerate Manhattan parallels that is exactly: mutually exterior-facing
 * edges, strictly positive projection overlap and perpendicular distance < d.
 * Collinear positive overlap is included; endpoint-only projection contact is
 * not.  Geometry outside this deliberately narrow source-equivalent domain
 * fails closed as kUncertain.
 */
__host__ __device__ inline Verdict classify_pair_bounded(
    const EdgePair &pair, std::int64_t distance) {
  const DirectedEdge &a = pair.implant;
  const DirectedEdge &b = pair.secondary;
  const bool a_horizontal = a.y1 == a.y2 && a.x1 != a.x2;
  const bool a_vertical = a.x1 == a.x2 && a.y1 != a.y2;
  const bool b_horizontal = b.y1 == b.y2 && b.x1 != b.x2;
  const bool b_vertical = b.x1 == b.x2 && b.y1 != b.y2;

  if (!(a_horizontal || a_vertical) || !(b_horizontal || b_vertical)) {
    return Verdict::kUncertain;
  }
  if (!detail::source_differences_are_safe(pair) ||
      (distance != kImplant1Distance &&
       distance != kImplant2Distance)) {
    return Verdict::kUncertain;
  }
  if (a_horizontal != b_horizontal) {
    return Verdict::kNoViolation;
  }

  const bool a_positive =
      a_horizontal ? a.x2 > a.x1 : a.y2 > a.y1;
  const bool b_positive =
      b_horizontal ? b.x2 > b.x1 : b.y2 > b.y1;
  if (a_positive == b_positive) {
    return Verdict::kNoViolation;
  }

  const bool projection_overlap =
      a_horizontal
          ? detail::intervals_overlap_strictly(
                a.x1, a.x2, b.x1, b.x2)
          : detail::intervals_overlap_strictly(
                a.y1, a.y2, b.y1, b.y2);
  if (!projection_overlap) {
    return Verdict::kNoViolation;
  }

  const std::int64_t a_line = a_horizontal ? a.y1 : a.x1;
  const std::int64_t b_line = a_horizontal ? b.y1 : b.x1;
  const std::uint64_t gap = detail::coordinate_gap(a_line, b_line);
  if (gap >= static_cast<std::uint64_t>(distance)) {
    return Verdict::kNoViolation;
  }
  if (gap == 0) {
    return Verdict::kViolation;
  }

  // The secondary must lie on the original implant edge's exterior (left)
  // side.  With antiparallel edges the reverse half-plane test is then also
  // satisfied, matching the source's two projected-near-part calls.
  const bool secondary_on_implant_exterior =
      a_horizontal
          ? (a_positive ? b.y1 > a.y1 : b.y1 < a.y1)
          : (a_positive ? b.x1 < a.x1 : b.x1 > a.x1);
  return secondary_on_implant_exterior ? Verdict::kViolation
                                       : Verdict::kNoViolation;
}

}  // namespace implant12
}  // namespace klayout_cuda

#endif
