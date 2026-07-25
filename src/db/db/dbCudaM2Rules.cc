/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaM2Rules.h"

#include "dbCudaManhattanContour.h"
#include "dbCudaSpatialBackend.h"
#include "dbFlatRegion.h"
#include "dbPolygon.h"
#include "dbRegion.h"
#include "dbShapes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace db
{

CudaM2FlatUnionStats::CudaM2FlatUnionStats ()
  : segment_count (0), contour_count (0), vertex_count (0),
    max_vertices (0)
{
  //  nothing yet
}

namespace
{

class M2FlatUnionDecline
  : public std::runtime_error
{
public:
  explicit M2FlatUnionDecline (const char *message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

struct PointKey
{
  db::Coord x;
  db::Coord y;

  bool operator== (const PointKey &other) const
  {
    return x == other.x && y == other.y;
  }
};

uint64_t mix_u64 (uint64_t value)
{
  value ^= value >> 30;
  value *= UINT64_C (0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C (0x94d049bb133111eb);
  value ^= value >> 31;
  return value;
}

struct PointKeyHash
{
  size_t operator() (const PointKey &point) const
  {
    const uint64_t x = mix_u64 (uint64_t (int64_t (point.x)));
    const uint64_t y = mix_u64 (uint64_t (int64_t (point.y)));
    return size_t (x ^ ((y << 1) | (y >> 63)));
  }
};

struct DirectedEdge
{
  PointKey first;
  PointKey second;
};

struct LineInterval
{
  uint32_t axis;
  int64_t fixed;
  int64_t lo;
  int64_t hi;
};

db::Coord narrow_coord (int64_t value)
{
  static_assert (
    std::numeric_limits<db::Coord>::is_signed,
    "KLayout coordinates must be signed");
  if (value < int64_t (std::numeric_limits<db::Coord>::min ()) ||
      value > int64_t (std::numeric_limits<db::Coord>::max ())) {
    throw M2FlatUnionDecline (
      "M2 boundary coordinate exceeds the KLayout coordinate range");
  }
  return db::Coord (value);
}

DirectedEdge directed_edge (
  const klayout_cuda_spatial_m2_union_segment_v1 &segment)
{
  const db::Coord fixed = narrow_coord (segment.fixed);
  const db::Coord lo = narrow_coord (segment.lo);
  const db::Coord hi = narrow_coord (segment.hi);

  //  KLayout polygon contours keep their material on the right.
  if (segment.axis ==
      KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL) {
    return segment.side < 0
      ? DirectedEdge { PointKey { hi, fixed }, PointKey { lo, fixed } }
      : DirectedEdge { PointKey { lo, fixed }, PointKey { hi, fixed } };
  }
  return segment.side < 0
    ? DirectedEdge { PointKey { fixed, lo }, PointKey { fixed, hi } }
    : DirectedEdge { PointKey { fixed, hi }, PointKey { fixed, lo } };
}

bool line_interval_less (
  const LineInterval &first, const LineInterval &second)
{
  if (first.axis != second.axis) {
    return first.axis < second.axis;
  }
  if (first.fixed != second.fixed) {
    return first.fixed < second.fixed;
  }
  if (first.lo != second.lo) {
    return first.lo < second.lo;
  }
  return first.hi < second.hi;
}

void validate_global_intersections (
  const klayout_cuda_spatial_m2_union_segment_v1 *segments,
  size_t segment_count)
{
  std::vector<LineInterval> line_intervals;
  std::vector<db::cuda_manhattan_contour::detail::SweepEvent>
    sweep_events;
  std::vector<int64_t> horizontal_y;
  line_intervals.reserve (segment_count);

  for (size_t index = 0; index < segment_count; ++index) {
    const klayout_cuda_spatial_m2_union_segment_v1 &segment =
      segments [index];
    line_intervals.push_back (
      LineInterval {
        segment.axis, segment.fixed, segment.lo, segment.hi
      });
    if (segment.axis ==
        KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL) {
      horizontal_y.push_back (segment.fixed);
      sweep_events.push_back (
        db::cuda_manhattan_contour::detail::SweepEvent {
          segment.lo,
          db::cuda_manhattan_contour::detail::SweepEvent::AddHorizontal,
          segment.fixed, segment.fixed
        });
      sweep_events.push_back (
        db::cuda_manhattan_contour::detail::SweepEvent {
          segment.hi,
          db::cuda_manhattan_contour::detail::SweepEvent::RemoveHorizontal,
          segment.fixed, segment.fixed
        });
    } else {
      sweep_events.push_back (
        db::cuda_manhattan_contour::detail::SweepEvent {
          segment.fixed,
          db::cuda_manhattan_contour::detail::SweepEvent::QueryVertical,
          segment.lo, segment.hi
        });
    }
  }

  std::sort (
    line_intervals.begin (), line_intervals.end (),
    line_interval_less);
  for (size_t index = 1; index < line_intervals.size (); ++index) {
    const LineInterval &previous = line_intervals [index - 1];
    const LineInterval &current = line_intervals [index];
    if (previous.axis == current.axis &&
        previous.fixed == current.fixed &&
        current.lo <= previous.hi) {
      throw M2FlatUnionDecline (
        "M2 boundary has a collinear overlap or point contact");
    }
  }

  if (! db::cuda_manhattan_contour::detail::
        cross_intersections_are_valid (
          sweep_events, horizontal_y, segment_count)) {
    throw M2FlatUnionDecline (
      "M2 boundary has an unexpected perpendicular crossing");
  }
}

__int128 checked_area_add (__int128 total, __int128 term)
{
  const __int128 maximum =
    __int128 ((~static_cast<unsigned __int128> (0)) >> 1);
  const __int128 minimum = -maximum - 1;
  if ((term > 0 && total > maximum - term) ||
      (term < 0 && total < minimum - term)) {
    throw M2FlatUnionDecline (
      "M2 boundary contour area exceeds the exact accumulator");
  }
  return total + term;
}

void set_reason (std::string *reason, const char *message)
{
  if (! reason) {
    return;
  }
  try {
    *reason = message;
  } catch (...) {
    //  Diagnostics cannot turn a decline into an exception.
  }
}

} // anonymous namespace

bool cuda_m2_union_boundary_to_flat_region (
  const klayout_cuda_spatial_m2_union_segment_v1 *segments,
  uint64_t segment_count, uint64_t expected_fnv64,
  db::Region &flat_union, CudaM2FlatUnionStats *stats,
  std::string *decline_reason)
{
  try {
    std::string canonical_error;
    if (! cuda_spatial_validate_m2_union_boundary (
          segments, segment_count, expected_fnv64,
          &canonical_error)) {
      throw M2FlatUnionDecline (canonical_error.c_str ());
    }
    if (! segment_count) {
      throw M2FlatUnionDecline (
        "a complete M2 union boundary cannot be empty");
    }
    if (segment_count > uint64_t (
          std::numeric_limits<size_t>::max ())) {
      throw M2FlatUnionDecline (
        "M2 boundary segment count exceeds the host address space");
    }

    const size_t count = size_t (segment_count);
    std::vector<DirectedEdge> edges;
    edges.reserve (count);
    std::unordered_map<PointKey, size_t, PointKeyHash> outgoing;
    outgoing.max_load_factor (0.75f);
    outgoing.reserve (count);

    for (size_t index = 0; index < count; ++index) {
      const DirectedEdge edge = directed_edge (segments [index]);
      if (! outgoing.insert (
            std::make_pair (edge.first, index)).second) {
        throw M2FlatUnionDecline (
          "M2 boundary has a repeated or kissing outgoing vertex");
      }
      edges.push_back (edge);
    }

    std::vector<size_t> next (count);
    std::vector<uint8_t> indegree (count, 0);
    for (size_t index = 0; index < count; ++index) {
      const std::unordered_map<
        PointKey, size_t, PointKeyHash>::const_iterator successor =
          outgoing.find (edges [index].second);
      if (successor == outgoing.end ()) {
        throw M2FlatUnionDecline (
          "M2 boundary has an open endpoint");
      }
      if (++indegree [successor->second] != 1) {
        throw M2FlatUnionDecline (
          "M2 boundary has a repeated or kissing incoming vertex");
      }
      next [index] = successor->second;
    }
    if (std::find (
          indegree.begin (), indegree.end (), uint8_t (0)) !=
        indegree.end ()) {
      throw M2FlatUnionDecline (
        "M2 boundary has a vertex without an incoming edge");
    }

    validate_global_intersections (segments, count);

    db::Shapes polygons (false);
    std::vector<uint8_t> visited (count, 0);
    std::vector<db::Point> points;
    CudaM2FlatUnionStats candidate_stats;
    candidate_stats.segment_count = segment_count;

    for (size_t seed = 0; seed < count; ++seed) {
      if (visited [seed]) {
        continue;
      }

      points.clear ();
      __int128 twice_area = 0;
      size_t current = seed;
      do {
        if (visited [current]) {
          throw M2FlatUnionDecline (
            "M2 boundary contour enters another cycle");
        }
        visited [current] = 1;
        const DirectedEdge &edge = edges [current];
        points.push_back (db::Point (edge.first.x, edge.first.y));
        const __int128 term =
          __int128 (edge.first.x) * __int128 (edge.second.y) -
          __int128 (edge.first.y) * __int128 (edge.second.x);
        twice_area = checked_area_add (twice_area, term);
        current = next [current];
        if (points.size () > count) {
          throw M2FlatUnionDecline (
            "M2 boundary contour does not close");
        }
      } while (current != seed);

      if (points.size () < 4) {
        throw M2FlatUnionDecline (
          "M2 boundary contour has fewer than four edges");
      }
      if (twice_area >= 0 || twice_area % 2 != 0) {
        throw M2FlatUnionDecline (
          "M2 boundary contains a hole, zero area, or nonclockwise contour");
      }

      db::Polygon polygon;
      polygon.assign_hull (
        points.begin (), points.end (), false, false, true);
      if (polygon.vertices () != points.size () ||
          polygon.holes () != 0 ||
          ! polygon.is_rectilinear ()) {
        throw M2FlatUnionDecline (
          "KLayout polygon materialization changed the M2 contour");
      }
      polygons.insert (polygon);
      ++candidate_stats.contour_count;
      candidate_stats.vertex_count += points.size ();
      candidate_stats.max_vertices = std::max<uint64_t> (
        candidate_stats.max_vertices, points.size ());
    }

    if (candidate_stats.vertex_count != segment_count ||
        ! candidate_stats.contour_count) {
      throw M2FlatUnionDecline (
        "stitched M2 contour census disagrees with the boundary");
    }

    std::unique_ptr<db::FlatRegion> flat (
      new db::FlatRegion (polygons, true));
    flat->set_merged_semantics (true);
    db::Region candidate (flat.release ());
    if (! candidate.merged_semantics () ||
        ! candidate.is_merged () ||
        candidate.count () != candidate_stats.contour_count) {
      throw M2FlatUnionDecline (
        "KLayout did not preserve the checked merged M2 region");
    }

    flat_union.swap (candidate);
    if (stats) {
      *stats = candidate_stats;
    }
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (
      decline_reason,
      "unknown exception while materializing the M2 union boundary");
  }
  return false;
}

} // namespace db
