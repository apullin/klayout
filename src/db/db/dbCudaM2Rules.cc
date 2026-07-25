/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaM2Rules.h"

#include "dbCudaM1WidthSpace.h"
#include "dbCudaManhattanContour.h"
#include "dbCudaSpatialBackend.h"
#include "dbFlatRegion.h"
#include "dbPolygon.h"
#include "dbRegion.h"
#include "dbShapes.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
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

CudaM2FlatUnionAttempt::CudaM2FlatUnionAttempt ()
  : disposition (Disabled), fallback_flags (0), device_flags (0),
    context_count (0), metal_context_count (0), cell_count (0),
    polygon_count (0), edge_count (0), flat_polygon_count (0),
    flat_edge_count (0), rectangle_count (0), x_slab_count (0),
    membership_count (0), event_count (0), strip_interval_count (0),
    raw_segment_count (0), boundary_segment_count (0), boundary_fnv64 (0),
    lowering_ns (0), backend_ns (0), materialize_ns (0),
    live_total_ns (0), flat_stats (), message ()
{
  //  nothing yet
}

namespace
{

const uint64_t m2_union_max_contexts = UINT64_C (4000000);
const uint64_t m2_union_max_rectangles = UINT64_C (32000000);
const uint64_t m2_union_max_x_slabs = UINT64_C (32000000);
const uint64_t m2_union_max_memberships = UINT64_C (100000000);
const uint64_t m2_union_max_events = UINT64_C (200000000);
const uint64_t m2_union_max_raw_segments = UINT64_C (12000000);
const uint64_t m2_union_max_segments = UINT64_C (8000000);
const uint32_t m2_union_max_slabs_per_rectangle = 64;

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

static_assert (
  std::is_standard_layout<CudaM1WidthSpaceContext>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceContext>::value &&
  sizeof (CudaM1WidthSpaceContext) ==
    sizeof (klayout_cuda_spatial_m1_width_space_context_v1) &&
  offsetof (CudaM1WidthSpaceContext, tx) ==
    offsetof (klayout_cuda_spatial_m1_width_space_context_v1, tx) &&
  offsetof (CudaM1WidthSpaceContext, ty) ==
    offsetof (klayout_cuda_spatial_m1_width_space_context_v1, ty) &&
  offsetof (CudaM1WidthSpaceContext, cell_id) ==
    offsetof (klayout_cuda_spatial_m1_width_space_context_v1, cell_id) &&
  offsetof (CudaM1WidthSpaceContext, transform_code) ==
    offsetof (
      klayout_cuda_spatial_m1_width_space_context_v1, transform_code),
  "raw M2 context ABI layout mismatch");
static_assert (
  std::is_standard_layout<CudaM1WidthSpaceCell>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceCell>::value &&
  sizeof (CudaM1WidthSpaceCell) ==
    sizeof (klayout_cuda_spatial_m1_width_space_cell_v1) &&
  offsetof (CudaM1WidthSpaceCell, source_cell_index) ==
    offsetof (
      klayout_cuda_spatial_m1_width_space_cell_v1, source_cell_index) &&
  offsetof (CudaM1WidthSpaceCell, polygon_begin) ==
    offsetof (klayout_cuda_spatial_m1_width_space_cell_v1, polygon_begin) &&
  offsetof (CudaM1WidthSpaceCell, edge_begin) ==
    offsetof (klayout_cuda_spatial_m1_width_space_cell_v1, edge_begin) &&
  offsetof (CudaM1WidthSpaceCell, polygon_count) ==
    offsetof (klayout_cuda_spatial_m1_width_space_cell_v1, polygon_count) &&
  offsetof (CudaM1WidthSpaceCell, edge_count) ==
    offsetof (klayout_cuda_spatial_m1_width_space_cell_v1, edge_count),
  "raw M2 cell ABI layout mismatch");
static_assert (
  std::is_standard_layout<CudaM1WidthSpacePolygon>::value &&
  std::is_trivially_copyable<CudaM1WidthSpacePolygon>::value &&
  sizeof (CudaM1WidthSpacePolygon) ==
    sizeof (klayout_cuda_spatial_m1_width_space_polygon_v1) &&
  offsetof (CudaM1WidthSpacePolygon, edge_begin) ==
    offsetof (
      klayout_cuda_spatial_m1_width_space_polygon_v1, edge_begin) &&
  offsetof (CudaM1WidthSpacePolygon, left) ==
    offsetof (klayout_cuda_spatial_m1_width_space_polygon_v1, left) &&
  offsetof (CudaM1WidthSpacePolygon, bottom) ==
    offsetof (klayout_cuda_spatial_m1_width_space_polygon_v1, bottom) &&
  offsetof (CudaM1WidthSpacePolygon, right) ==
    offsetof (klayout_cuda_spatial_m1_width_space_polygon_v1, right) &&
  offsetof (CudaM1WidthSpacePolygon, top) ==
    offsetof (klayout_cuda_spatial_m1_width_space_polygon_v1, top) &&
  offsetof (CudaM1WidthSpacePolygon, polygon_id) ==
    offsetof (
      klayout_cuda_spatial_m1_width_space_polygon_v1, polygon_id) &&
  offsetof (CudaM1WidthSpacePolygon, edge_count) ==
    offsetof (klayout_cuda_spatial_m1_width_space_polygon_v1, edge_count),
  "raw M2 polygon ABI layout mismatch");
static_assert (
  std::is_standard_layout<CudaM1WidthSpaceEdge>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceEdge>::value &&
  sizeof (CudaM1WidthSpaceEdge) ==
    sizeof (klayout_cuda_spatial_m1_width_space_edge_v1) &&
  offsetof (CudaM1WidthSpaceEdge, x1) ==
    offsetof (klayout_cuda_spatial_m1_width_space_edge_v1, x1) &&
  offsetof (CudaM1WidthSpaceEdge, y1) ==
    offsetof (klayout_cuda_spatial_m1_width_space_edge_v1, y1) &&
  offsetof (CudaM1WidthSpaceEdge, x2) ==
    offsetof (klayout_cuda_spatial_m1_width_space_edge_v1, x2) &&
  offsetof (CudaM1WidthSpaceEdge, y2) ==
    offsetof (klayout_cuda_spatial_m1_width_space_edge_v1, y2),
  "raw M2 edge ABI layout mismatch");

uint64_t elapsed_ns (
  const std::chrono::steady_clock::time_point &begin,
  const std::chrono::steady_clock::time_point &end)
{
  const std::chrono::nanoseconds elapsed =
    std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin);
  return elapsed.count () > 0 ? uint64_t (elapsed.count ()) : 0;
}

void make_m2_union_request (
  const CudaM2RawManhattanScene &scene, int32_t device,
  klayout_cuda_spatial_m2_union_request_v1 &request)
{
  std::memset (&request, 0, sizeof (request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof (request);
  request.opcode =
    KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY;
  request.option_flags =
    KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  request.format_version = scene.format_version;
  request.dbu_per_micron = scene.dbu_per_micron;
  request.root_cell = scene.root_cell;
  request.device = device;
  request.contexts = scene.contexts.data ();
  request.context_count = scene.contexts.size ();
  request.context_record_bytes = sizeof (CudaM1WidthSpaceContext);
  request.metal_contexts = scene.metal_contexts.data ();
  request.metal_context_count = scene.metal_contexts.size ();
  request.context_polygon_offsets =
    scene.context_polygon_offsets.data ();
  request.context_polygon_offset_count =
    scene.context_polygon_offsets.size ();
  request.context_edge_offsets = scene.context_edge_offsets.data ();
  request.context_edge_offset_count = scene.context_edge_offsets.size ();
  request.cells = scene.cells.data ();
  request.cell_count = scene.cells.size ();
  request.cell_record_bytes = sizeof (CudaM1WidthSpaceCell);
  request.polygons = scene.polygons.data ();
  request.polygon_count = scene.polygons.size ();
  request.polygon_record_bytes = sizeof (CudaM1WidthSpacePolygon);
  request.edges = scene.edges.data ();
  request.edge_count = scene.edges.size ();
  request.edge_record_bytes = sizeof (CudaM1WidthSpaceEdge);
  request.flat_polygon_count = scene.flat_polygon_count;
  request.flat_edge_count = scene.flat_edge_count;
  request.scene_left = scene.scene_left;
  request.scene_bottom = scene.scene_bottom;
  request.scene_right = scene.scene_right;
  request.scene_top = scene.scene_top;
  request.max_contexts = m2_union_max_contexts;
  request.max_rectangles = m2_union_max_rectangles;
  request.max_x_slabs = m2_union_max_x_slabs;
  request.max_memberships = m2_union_max_memberships;
  request.max_events = m2_union_max_events;
  request.max_raw_segments = m2_union_max_raw_segments;
  request.max_segments = m2_union_max_segments;
  request.max_slabs_per_rectangle =
    m2_union_max_slabs_per_rectangle;
  std::copy (
    scene.digest.begin (), scene.digest.end (), request.scene_digest);
}

void copy_backend_telemetry (
  const CudaM2UnionAttempt &backend,
  CudaM2FlatUnionAttempt &attempt)
{
  attempt.fallback_flags = backend.fallback_flags;
  attempt.device_flags = backend.device_flags;
  attempt.context_count = backend.context_count;
  attempt.metal_context_count = backend.metal_context_count;
  attempt.cell_count = backend.cell_count;
  attempt.polygon_count = backend.polygon_count;
  attempt.edge_count = backend.edge_count;
  attempt.flat_polygon_count = backend.flat_polygon_count;
  attempt.flat_edge_count = backend.flat_edge_count;
  attempt.rectangle_count = backend.rectangle_count;
  attempt.x_slab_count = backend.x_slab_count;
  attempt.membership_count = backend.membership_count;
  attempt.event_count = backend.event_count;
  attempt.strip_interval_count = backend.strip_interval_count;
  attempt.raw_segment_count = backend.raw_segment_count;
  attempt.boundary_segment_count = backend.segments.size ();
  attempt.boundary_fnv64 = backend.boundary_fnv64;
  attempt.backend_ns = backend.total_ns;
  attempt.message = backend.message;
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

CudaM2FlatUnionAttempt cuda_m2_raw_manhattan_try_flat_union (
  const db::DeepLayer &raw_metal2, db::Region &flat_union,
  int32_t device)
{
  CudaM2FlatUnionAttempt attempt;

  //  This check deliberately precedes every access to raw_metal2.  A host
  //  must not pay for hierarchy lowering when the complete optional
  //  run/release capability is absent.
  if (! cuda_spatial_m2_union_requested ()) {
    return attempt;
  }

  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  bool backend_called = false;
  try {
    if (device < 0) {
      attempt.disposition = CudaM2FlatUnionAttempt::HostDeclined;
      attempt.message = "the CUDA M2 union device must be nonnegative";
      attempt.live_total_ns =
        elapsed_ns (begin, std::chrono::steady_clock::now ());
      return attempt;
    }

    CudaM1WidthSpaceSceneLimits scene_limits;
    scene_limits.max_contexts = m2_union_max_contexts;
    scene_limits.max_flat_polygons = m2_union_max_rectangles;

    CudaM2RawManhattanScene scene;
    std::string reason;
    if (! cuda_m2_raw_manhattan_build_scene (
          raw_metal2, scene_limits, scene, &reason)) {
      attempt.disposition = CudaM2FlatUnionAttempt::HostDeclined;
      attempt.message.swap (reason);
      if (attempt.message.empty ()) {
        attempt.message =
          "unable to serialize the qualified raw physical M2 scene";
      }
      const std::chrono::steady_clock::time_point decline_end =
        std::chrono::steady_clock::now ();
      attempt.lowering_ns = elapsed_ns (begin, decline_end);
      attempt.live_total_ns = attempt.lowering_ns;
      return attempt;
    }

    klayout_cuda_spatial_m2_union_request_v1 request;
    make_m2_union_request (scene, device, request);
    const std::chrono::steady_clock::time_point lower_end =
      std::chrono::steady_clock::now ();
    attempt.lowering_ns = elapsed_ns (begin, lower_end);

    backend_called = true;
    const CudaM2UnionAttempt backend =
      cuda_spatial_try_m2_union (request);
    copy_backend_telemetry (backend, attempt);

    switch (backend.disposition) {
    case CudaM2UnionAttempt::BackendFallback:
      attempt.disposition = CudaM2FlatUnionAttempt::BackendFallback;
      break;
    case CudaM2UnionAttempt::BackendError:
      attempt.disposition = CudaM2FlatUnionAttempt::BackendError;
      break;
    case CudaM2UnionAttempt::InvalidResult:
      attempt.disposition = CudaM2FlatUnionAttempt::InvalidResult;
      break;
    case CudaM2UnionAttempt::Disabled:
      attempt.disposition = CudaM2FlatUnionAttempt::BackendError;
      if (attempt.message.empty ()) {
        attempt.message =
          "the advertised CUDA M2 union capability was not invoked";
      }
      break;
    case CudaM2UnionAttempt::Complete:
      break;
    }
    if (backend.disposition != CudaM2UnionAttempt::Complete) {
      attempt.live_total_ns =
        elapsed_ns (begin, std::chrono::steady_clock::now ());
      return attempt;
    }

    const std::chrono::steady_clock::time_point materialize_begin =
      std::chrono::steady_clock::now ();
    CudaM2FlatUnionStats flat_stats;
    std::string topology_reason;
    const bool materialized =
      cuda_m2_union_boundary_to_flat_region (
        backend.segments.data (), backend.segments.size (),
        backend.boundary_fnv64, flat_union, &flat_stats,
        &topology_reason);
    const std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now ();
    attempt.materialize_ns = elapsed_ns (materialize_begin, end);
    attempt.live_total_ns = elapsed_ns (begin, end);
    if (! materialized) {
      attempt.disposition = CudaM2FlatUnionAttempt::TopologyDeclined;
      attempt.message.swap (topology_reason);
      if (attempt.message.empty ()) {
        attempt.message =
          "unable to materialize the validated CUDA M2 boundary";
      }
      return attempt;
    }

    attempt.flat_stats = flat_stats;
    attempt.disposition = CudaM2FlatUnionAttempt::Complete;
    attempt.message.clear ();
    return attempt;
  } catch (const std::exception &ex) {
    attempt.disposition =
      backend_called ? CudaM2FlatUnionAttempt::BackendError
                     : CudaM2FlatUnionAttempt::HostDeclined;
    try {
      attempt.message = ex.what ();
    } catch (...) {
      //  Diagnostics cannot turn a fail-closed result into an exception.
    }
  } catch (...) {
    attempt.disposition =
      backend_called ? CudaM2FlatUnionAttempt::BackendError
                     : CudaM2FlatUnionAttempt::HostDeclined;
    try {
      attempt.message =
        "unknown exception in the raw M2 flat-union transaction";
    } catch (...) {
      //  Diagnostics cannot turn a fail-closed result into an exception.
    }
  }
  attempt.live_total_ns =
    elapsed_ns (begin, std::chrono::steady_clock::now ());
  return attempt;
}

} // namespace db
