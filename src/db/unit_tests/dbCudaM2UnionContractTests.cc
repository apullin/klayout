/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaSpatialBackend.h"
#include "tlUnitTest.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace
{

typedef klayout_cuda_spatial_m2_union_segment_v1 Segment;

static_assert (sizeof (Segment) == 32, "M2 boundary ABI size changed");
static_assert (offsetof (Segment, fixed) == 0, "M2 fixed offset changed");
static_assert (offsetof (Segment, lo) == 8, "M2 lo offset changed");
static_assert (offsetof (Segment, hi) == 16, "M2 hi offset changed");
static_assert (offsetof (Segment, side) == 24, "M2 side offset changed");
static_assert (offsetof (Segment, axis) == 28, "M2 axis offset changed");

} // anonymous namespace

TEST(1_CanonicalOrderAndDigest)
{
  const Segment rectangle [] = {
    { 0, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 20, 0, 10, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 0, 0, 20, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL },
    { 10, 0, 20, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL }
  };
  const uint64_t rectangle_fnv64 = UINT64_C (11447980897846940057);
  std::string error;
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      rectangle, 4, rectangle_fnv64, &error),
    true);
  EXPECT_EQ (error, "");

  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      rectangle, 4, rectangle_fnv64 + 1, &error),
    false);
  EXPECT_EQ (error, "M2 boundary FNV-1a digest mismatch");

  //  This is ordered by the proven (axis, side, fixed, lo, hi) contract even
  //  though its fixed coordinates decrease.  Sorting fixed before side would
  //  incorrectly reject it.
  const Segment side_before_fixed [] = {
    { 100, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 0, 0, 10, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL }
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      side_before_fixed, 2, UINT64_C (11131890132215870808), &error),
    true);

  const Segment fixed_before_side [] = {
    side_before_fixed [1], side_before_fixed [0]
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      fixed_before_side, 2, 0, &error),
    false);
  EXPECT_EQ (
    error,
    "M2 boundary is not strictly ordered by (axis,side,fixed,lo,hi)");
}

TEST(2_NonmaximalAndMalformedFailClosed)
{
  std::string error;
  const Segment touching [] = {
    { 0, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 0, 10, 20, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL }
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      touching, 2, 0, &error),
    false);
  EXPECT_EQ (
    error,
    "M2 boundary has nonmaximal touching or overlapping segments");

  const Segment bad_axis [] = {
    { 0, 0, 10, -1, 2 }
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      bad_axis, 1, 0, &error),
    false);
  EXPECT_EQ (error, "M2 boundary has an invalid segment");

  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      0, 1, 0, &error),
    false);
  EXPECT_EQ (
    error, "nonempty M2 boundary has a null segment pointer");
}
