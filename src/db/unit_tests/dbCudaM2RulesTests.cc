/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaM2Rules.h"

#include "dbBox.h"
#include "dbDeepShapeStore.h"
#include "dbRegion.h"
#include "tlUnitTest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{

typedef klayout_cuda_spatial_m2_union_segment_v1 Segment;

static_assert (
  offsetof (db::CudaM2FlatUnionAttempt, disposition) == 0 &&
  offsetof (db::CudaM2FlatUnionAttempt, context_count) == 16 &&
  offsetof (db::CudaM2FlatUnionAttempt, boundary_fnv64) == 128 &&
  offsetof (db::CudaM2FlatUnionAttempt, lowering_ns) == 136 &&
  offsetof (db::CudaM2FlatUnionAttempt, flat_stats) == 168 &&
  offsetof (db::CudaM2FlatUnionAttempt, message) == 200 &&
  sizeof (db::CudaM2FlatUnionAttempt) ==
    offsetof (db::CudaM2FlatUnionAttempt, message) +
      sizeof (std::string),
  "legacy flat M2 attempt ABI layout changed");
static_assert (
  sizeof (db::CudaM2FlatUnionSuffixCertificate) == 24 &&
  offsetof (
    db::CudaM2FlatUnionSuffixCertificate, certified_empty_mask) == 8 &&
  offsetof (db::CudaM2FlatUnionSuffixCertificate, total_ns) == 16,
  "flat M2 suffix certificate ABI layout changed");

bool segment_less (const Segment &first, const Segment &second)
{
  if (first.axis != second.axis) {
    return first.axis < second.axis;
  }
  if (first.side != second.side) {
    return first.side < second.side;
  }
  if (first.fixed != second.fixed) {
    return first.fixed < second.fixed;
  }
  if (first.lo != second.lo) {
    return first.lo < second.lo;
  }
  return first.hi < second.hi;
}

uint64_t boundary_fnv64 (const std::vector<Segment> &segments)
{
  uint64_t hash = UINT64_C (1469598103934665603);
  const auto mix = [&hash] (uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C (0xff);
      hash *= UINT64_C (1099511628211);
    }
  };
  mix (segments.size ());
  for (std::vector<Segment>::const_iterator segment =
         segments.begin (); segment != segments.end (); ++segment) {
    mix (uint64_t (segment->axis));
    mix (uint64_t (uint32_t (segment->side)));
    mix (uint64_t (segment->fixed));
    mix (uint64_t (segment->lo));
    mix (uint64_t (segment->hi));
  }
  return hash;
}

std::vector<Segment> rectangle (
  int64_t left, int64_t bottom, int64_t right, int64_t top,
  bool clockwise = true)
{
  std::vector<Segment> result;
  if (clockwise) {
    result.push_back (
      Segment {
        bottom, left, right, -1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
      });
    result.push_back (
      Segment {
        top, left, right, 1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
      });
    result.push_back (
      Segment {
        left, bottom, top, -1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL
      });
    result.push_back (
      Segment {
        right, bottom, top, 1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL
      });
  } else {
    result.push_back (
      Segment {
        top, left, right, -1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
      });
    result.push_back (
      Segment {
        bottom, left, right, 1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
      });
    result.push_back (
      Segment {
        right, bottom, top, -1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL
      });
    result.push_back (
      Segment {
        left, bottom, top, 1,
        KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL
      });
  }
  std::sort (result.begin (), result.end (), segment_less);
  return result;
}

void append_rectangle (
  std::vector<Segment> &target,
  int64_t left, int64_t bottom, int64_t right, int64_t top)
{
  const std::vector<Segment> added =
    rectangle (left, bottom, right, top);
  target.insert (target.end (), added.begin (), added.end ());
  std::sort (target.begin (), target.end (), segment_less);
}

void expect_decline_unchanged (
  tl::TestBase *_this, const std::vector<Segment> &segments,
  const char *expected_reason = 0)
{
  db::Region output (db::Box (100, 200, 300, 400));
  db::CudaM2FlatUnionStats stats;
  stats.segment_count = 91;
  stats.contour_count = 92;
  stats.vertex_count = 93;
  stats.max_vertices = 94;
  std::string reason;
  EXPECT_EQ (
    db::cuda_m2_union_boundary_to_flat_region (
      segments.data (), segments.size (), boundary_fnv64 (segments),
      output, &stats, &reason),
    false);
  EXPECT_EQ (output.count (), size_t (1));
  EXPECT_EQ (output.bbox (), db::Box (100, 200, 300, 400));
  EXPECT_EQ (stats.segment_count, uint64_t (91));
  EXPECT_EQ (stats.contour_count, uint64_t (92));
  EXPECT_EQ (stats.vertex_count, uint64_t (93));
  EXPECT_EQ (stats.max_vertices, uint64_t (94));
  EXPECT_EQ (reason.empty (), false);
  if (expected_reason) {
    EXPECT_EQ (reason, expected_reason);
  }
}

} // anonymous namespace

TEST(1_ValidBoundaryBecomesOwnedMergedFlatRegion)
{
  const std::vector<Segment> box = rectangle (0, 0, 10, 20);
  EXPECT_EQ (boundary_fnv64 (box), UINT64_C (11447980897846940057));

  db::Region output (db::Box (100, 200, 300, 400));
  db::CudaM2FlatUnionStats stats;
  std::string reason;
  EXPECT_EQ (
    db::cuda_m2_union_boundary_to_flat_region (
      box.data (), box.size (), boundary_fnv64 (box),
      output, &stats, &reason),
    true);
  EXPECT_EQ (reason, "");
  EXPECT_EQ (output.merged_semantics (), true);
  EXPECT_EQ (output.is_merged (), true);
  EXPECT_EQ (output.count (), size_t (1));
  EXPECT_EQ (output.bbox (), db::Box (0, 0, 10, 20));
  EXPECT_EQ (stats.segment_count, uint64_t (4));
  EXPECT_EQ (stats.contour_count, uint64_t (1));
  EXPECT_EQ (stats.vertex_count, uint64_t (4));
  EXPECT_EQ (stats.max_vertices, uint64_t (4));

  std::vector<Segment> disjoint = box;
  append_rectangle (disjoint, 100, 200, 130, 240);
  EXPECT_EQ (
    db::cuda_m2_union_boundary_to_flat_region (
      disjoint.data (), disjoint.size (), boundary_fnv64 (disjoint),
      output, &stats, &reason),
    true);
  EXPECT_EQ (output.is_merged (), true);
  EXPECT_EQ (output.count (), size_t (2));
  EXPECT_EQ (stats.segment_count, uint64_t (8));
  EXPECT_EQ (stats.contour_count, uint64_t (2));
  EXPECT_EQ (stats.vertex_count, uint64_t (8));
}

TEST(2_TopologyAndOrientationFailClosed)
{
  std::vector<Segment> open = rectangle (0, 0, 10, 20);
  open.pop_back ();
  expect_decline_unchanged (_this, open);

  std::vector<Segment> kissing = rectangle (0, 0, 10, 10);
  append_rectangle (kissing, 10, 10, 20, 20);
  expect_decline_unchanged (
    _this, kissing,
    "M2 boundary has a repeated or kissing outgoing vertex");

  std::vector<Segment> crossing = rectangle (0, 0, 10, 10);
  append_rectangle (crossing, 5, -5, 15, 5);
  expect_decline_unchanged (_this, crossing);

  const std::vector<Segment> hole =
    rectangle (0, 0, 10, 20, false);
  expect_decline_unchanged (
    _this, hole,
    "M2 boundary contains a hole, zero area, or nonclockwise contour");
}

TEST(3_DigestCanonicalAndCoordinateFailuresAreAtomic)
{
  const std::vector<Segment> box = rectangle (0, 0, 10, 20);
  db::Region output (db::Box (100, 200, 300, 400));
  db::CudaM2FlatUnionStats stats;
  stats.segment_count = 77;
  std::string reason;
  EXPECT_EQ (
    db::cuda_m2_union_boundary_to_flat_region (
      box.data (), box.size (), boundary_fnv64 (box) + 1,
      output, &stats, &reason),
    false);
  EXPECT_EQ (output.bbox (), db::Box (100, 200, 300, 400));
  EXPECT_EQ (stats.segment_count, uint64_t (77));
  EXPECT_EQ (reason, "M2 boundary FNV-1a digest mismatch");

  std::vector<Segment> unordered = box;
  std::swap (unordered [0], unordered [1]);
  expect_decline_unchanged (_this, unordered);

  if (sizeof (db::Coord) < sizeof (int64_t)) {
    const int64_t left =
      int64_t (std::numeric_limits<db::Coord>::max ()) + 1;
    const std::vector<Segment> outside =
      rectangle (left, 0, left + 10, 20);
    expect_decline_unchanged (_this, outside);
  }
}

TEST(4_NullDegenerateAndDuplicateEndpointsFailClosed)
{
  db::Region output (db::Box (100, 200, 300, 400));
  db::CudaM2FlatUnionStats stats;
  stats.segment_count = 77;
  stats.contour_count = 78;
  stats.vertex_count = 79;
  stats.max_vertices = 80;
  std::string reason;
  EXPECT_EQ (
    db::cuda_m2_union_boundary_to_flat_region (
      0, 4, 0, output, &stats, &reason),
    false);
  EXPECT_EQ (reason, "nonempty M2 boundary has a null segment pointer");
  EXPECT_EQ (output.count (), size_t (1));
  EXPECT_EQ (output.bbox (), db::Box (100, 200, 300, 400));
  EXPECT_EQ (stats.segment_count, uint64_t (77));
  EXPECT_EQ (stats.contour_count, uint64_t (78));
  EXPECT_EQ (stats.vertex_count, uint64_t (79));
  EXPECT_EQ (stats.max_vertices, uint64_t (80));

  std::vector<Segment> degenerate;
  degenerate.push_back (
    Segment {
      0, 10, 10, -1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
    });
  expect_decline_unchanged (
    _this, degenerate, "M2 boundary has an invalid segment");

  std::vector<Segment> duplicate_incoming;
  duplicate_incoming.push_back (
    Segment {
      0, 0, 10, -1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
    });
  duplicate_incoming.push_back (
    Segment {
      0, 0, 10, -1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL
    });
  duplicate_incoming.push_back (
    Segment {
      0, 0, 10, 1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL
    });
  std::sort (
    duplicate_incoming.begin (), duplicate_incoming.end (),
    segment_less);
  expect_decline_unchanged (
    _this, duplicate_incoming,
    "M2 boundary has a repeated or kissing incoming vertex");
}

TEST(5_OppositeSideCollinearDegeneraciesFailClosed)
{
  std::vector<Segment> overlapping;
  overlapping.push_back (
    Segment {
      0, 0, 10, -1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
    });
  overlapping.push_back (
    Segment {
      0, 0, 10, 1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
    });
  std::sort (overlapping.begin (), overlapping.end (), segment_less);
  expect_decline_unchanged (
    _this, overlapping,
    "M2 boundary has a collinear overlap or point contact");

  std::vector<Segment> touching;
  touching.push_back (
    Segment {
      0, 0, 10, -1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
    });
  touching.push_back (
    Segment {
      0, 10, 20, 1,
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL
    });
  std::sort (touching.begin (), touching.end (), segment_less);
  expect_decline_unchanged (
    _this, touching,
    "M2 boundary has a repeated or kissing outgoing vertex");
}

TEST(6_MissingCapabilityPrecedesRawGeometryAccess)
{
  const db::DeepLayer deliberately_invalid;
  db::Region output (db::Box (100, 200, 300, 400));
  const db::CudaM2FlatUnionAttempt attempt =
    db::cuda_m2_raw_manhattan_try_flat_union (
      deliberately_invalid, output);
  EXPECT_EQ (
    attempt.disposition, db::CudaM2FlatUnionAttempt::Disabled);
  EXPECT_EQ (attempt.lowering_ns, uint64_t (0));
  EXPECT_EQ (attempt.boundary_segment_count, uint64_t (0));
  EXPECT_EQ (attempt.message, "");
  EXPECT_EQ (output.count (), size_t (1));
  EXPECT_EQ (output.bbox (), db::Box (100, 200, 300, 400));

  db::Region suffix_output (db::Box (100, 200, 300, 400));
  db::CudaM2FlatUnionSuffixCertificate certificate;
  std::memset (&certificate, 0xa5, sizeof (certificate));
  const db::CudaM2FlatUnionAttempt suffix_attempt =
    db::cuda_m2_raw_manhattan_try_flat_union_with_suffix (
      deliberately_invalid, suffix_output, &certificate,
      sizeof (certificate));
  EXPECT_EQ (
    suffix_attempt.disposition, db::CudaM2FlatUnionAttempt::Disabled);
  EXPECT_EQ (
    certificate.format_version,
    uint32_t (db::CudaM2FlatUnionSuffixCertificate::FormatVersion));
  EXPECT_EQ (certificate.struct_size, uint32_t (sizeof (certificate)));
  EXPECT_EQ (certificate.certified_empty_mask, uint32_t (0));
  EXPECT_EQ (certificate.reserved, uint32_t (0));
  EXPECT_EQ (certificate.total_ns, uint64_t (0));
  EXPECT_EQ (suffix_output.count (), size_t (1));
  EXPECT_EQ (
    suffix_output.bbox (), db::Box (100, 200, 300, 400));
}

TEST(7_SuffixRecordSizeFailsBeforeGeometryAccess)
{
  const db::DeepLayer deliberately_invalid;
  db::Region output (db::Box (100, 200, 300, 400));
  db::CudaM2FlatUnionSuffixCertificate certificate;
  std::memset (&certificate, 0xa5, sizeof (certificate));
  const db::CudaM2FlatUnionAttempt attempt =
    db::cuda_m2_raw_manhattan_try_flat_union_with_suffix (
      deliberately_invalid, output, &certificate,
      sizeof (certificate) - 1);
  EXPECT_EQ (
    attempt.disposition, db::CudaM2FlatUnionAttempt::InvalidResult);
  EXPECT_EQ (attempt.lowering_ns, uint64_t (0));
  const unsigned char *bytes =
    reinterpret_cast<const unsigned char *> (&certificate);
  EXPECT_EQ (
    std::find_if (
      bytes, bytes + sizeof (certificate),
      [] (unsigned char value) { return value != 0xa5; }) ==
        bytes + sizeof (certificate),
    true);
  EXPECT_EQ (output.count (), size_t (1));
  EXPECT_EQ (output.bbox (), db::Box (100, 200, 300, 400));
}
