/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

// Standalone proof of an atomic METAL1.1/METAL1.2 empty certificate.
//
// The host supplies compact, already-merged Manhattan polygon templates and
// resolved hierarchy contexts.  CUDA expands every directed edge occurrence,
// constructs a complete uniform-grid edge index, and applies the exact
// bounded WidthRelation and SpaceRelation predicates directly to unique edge
// pairs.  No candidate or marker stream returns to the host.
//
// This first island intentionally evaluates the unshielded relation
// superset.  KLayout shielding can only remove complete edge pairs, so zero
// raw hits is a valid shielded zero-hit certificate.  Any raw hit,
// unsupported geometry, uncertainty, capacity exhaustion, counter mismatch,
// or device flag declines the whole two-rule transaction.

#define main klayout_cuda_embedded_active3_scene_main
#include "active3_scene_island.cu"
#undef main

#include "m1_width_space_exact_predicate.h"
#include "m1_width_space_host_scene_format.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace m1p = klayout_cuda::m1_width_space;
namespace m1fmt = klayout_m1ws_scene;
using M1DirectedEdge = m1p::DirectedEdge;
using M1CandidatePair = m1p::CandidatePair;
using M1Rule = m1p::Rule;
using M1Verdict = m1p::Verdict;

constexpr std::int64_t kM1Distance =
    m1p::kQualifiedSceneCoordinateDistance;
constexpr std::int64_t kM1GridCell = 512;
constexpr std::uint64_t kM1DefaultMaxContexts = UINT64_C(4000000);
constexpr std::uint64_t kM1DefaultMaxGridCells = UINT64_C(16000000);
constexpr std::uint64_t kM1DefaultMaxMemberships = UINT64_C(120000000);
constexpr std::uint64_t kM1DefaultMaxPairWork = UINT64_C(2000000000);
constexpr std::uint64_t kM1DefaultMaxEdges = UINT64_C(100000000);
constexpr std::uint64_t kM1DefaultMaxPolygons = UINT64_C(50000000);
constexpr std::uint32_t kM1SampleCapacity = 8;
constexpr char kM1QualifiedHostSceneSha256[] =
    "df713200c1271e510ac2ecd1bdc060054e69451658f0c4327d64b228f8925235";
constexpr char kM1QualifiedSourceGdsSha256[] =
    "c3cef4f83d08bef36c109837f632fe9e4411479fb10bad4da08c16ee6d246846";
constexpr std::uint64_t kM1QualifiedContextCount = UINT64_C(590713);
constexpr std::uint64_t kM1QualifiedMetalContextCount = UINT64_C(535092);
constexpr std::uint64_t kM1QualifiedCellCount = UINT64_C(185);
constexpr std::uint64_t kM1QualifiedStoredPolygonCount = UINT64_C(543760);
constexpr std::uint64_t kM1QualifiedStoredEdgeCount = UINT64_C(7320532);
constexpr std::uint64_t kM1QualifiedFlatPolygonCount = UINT64_C(2680764);
constexpr std::uint64_t kM1QualifiedFlatEdgeCount = UINT64_C(24432912);

enum M1DeviceFlag : std::uint32_t {
  kM1DeviceOk = 0,
  kM1TransformOverflow = 1u << 0,
  kM1InvalidRecord = 1u << 1,
  kM1GridCounterOverflow = 1u << 2,
  kM1GridCapacityExceeded = 1u << 3,
  kM1PairCounterOverflow = 1u << 4,
  kM1PairCapacityExceeded = 1u << 5,
  kM1ConservationFailure = 1u << 6,
};

struct M1CellTemplate {
  std::uint64_t polygon_begin;
  std::uint64_t edge_begin;
  std::uint32_t polygon_count;
  std::uint32_t edge_count;
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
};

struct M1PolygonTemplate {
  std::uint64_t edge_begin;
  std::uint32_t edge_count;
  std::uint32_t reserved;
};

struct M1EdgeTemplate {
  M1DirectedEdge edge;
  std::uint32_t polygon_local;
  std::uint32_t edge_local;
};

struct M1CompactScene {
  std::string name;
  std::vector<ContextGpu> contexts;
  std::vector<M1CellTemplate> cells;
  std::vector<M1PolygonTemplate> polygons;
  std::vector<M1EdgeTemplate> edges;
  // The production seam must lower merged_deep_layer().  Fixtures which
  // deliberately model touching/coincident raw polygons clear this bit and
  // exercise fail-closed fallback instead of pretending they are merged.
  bool exact_merged_universe = false;
};

struct M1LoweredScene {
  std::vector<ContextGpu> contexts;
  std::vector<std::uint64_t> edge_offsets;
  std::vector<std::uint64_t> polygon_offsets;
  std::vector<M1CellTemplate> cells;
  std::vector<M1EdgeTemplate> edges;
  std::uint64_t edge_count = 0;
  std::uint64_t polygon_count = 0;
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
};

struct M1ExpandedEdge {
  M1DirectedEdge edge;
  std::uint64_t polygon_id;
  std::uint32_t context_id;
  std::uint32_t edge_local;
};

struct M1Grid {
  std::int64_t base_x;
  std::int64_t base_y;
  std::uint32_t width;
  std::uint32_t height;
};

struct M1DeviceCounters {
  unsigned long long expanded_edges;
  unsigned long long unique_edge_pairs;
  unsigned long long width_pairs;
  unsigned long long space_pairs;
  unsigned long long width_hits;
  unsigned long long space_hits;
  unsigned long long width_uncertain;
  unsigned long long space_uncertain;
};

struct M1Sample {
  std::uint32_t first_edge;
  std::uint32_t second_edge;
  std::uint32_t rule;
  std::uint32_t verdict;
  std::uint64_t first_polygon;
  std::uint64_t second_polygon;
};

struct M1HostOracle {
  std::uint64_t width_hits = 0;
  std::uint64_t space_hits = 0;
  std::uint64_t width_uncertain = 0;
  std::uint64_t space_uncertain = 0;
};

struct M1Timings {
  double packed_load_ms = 0.0;
  double validate_lower_ms = 0.0;
  double host_oracle_ms = 0.0;
  double cuda_init_ms = 0.0;
  double alloc_upload_ms = 0.0;
  double expand_ms = 0.0;
  double grid_count_ms = 0.0;
  double grid_build_ms = 0.0;
  double pair_count_ms = 0.0;
  double query_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

enum class M1Disposition {
  kComplete,
  kRawHits,
  kUncertain,
};

struct M1PlanResult {
  M1Disposition disposition = M1Disposition::kUncertain;
  M1DeviceCounters counters{};
  M1HostOracle oracle{};
  std::uint32_t device_flags = 0;
  std::uint64_t memberships = 0;
  std::uint64_t pair_work = 0;
  std::uint64_t grid_cells = 0;
  std::uint64_t edges = 0;
  std::uint64_t polygons = 0;
  std::uint64_t contexts = 0;
  M1Grid grid{};
  M1Timings timing{};
  std::array<M1Sample, kM1SampleCapacity> samples{};
  std::uint32_t sample_count = 0;
  std::string scene_digest;
  std::string message;
};

struct M1Options {
  bool self_test = false;
  bool host_oracle = true;
  bool trust_merged_layer0 = false;
  std::string packed_scene;
  std::string host_scene;
  std::string expected_scene_sha256;
  std::uint64_t benchmark_contexts = 0;
  std::uint64_t repetitions = 1;
  std::uint64_t max_contexts = kM1DefaultMaxContexts;
  std::uint64_t max_grid_cells = kM1DefaultMaxGridCells;
  std::uint64_t max_memberships = kM1DefaultMaxMemberships;
  std::uint64_t max_pair_work = kM1DefaultMaxPairWork;
  std::uint64_t max_edges = kM1DefaultMaxEdges;
  std::uint64_t max_polygons = kM1DefaultMaxPolygons;
};

struct M1HostCellRecord {
  std::uint64_t source_cell_index;
  std::uint64_t polygon_begin;
  std::uint64_t edge_begin;
  std::uint32_t polygon_count;
  std::uint32_t edge_count;
};

struct M1LoadedHostScene {
  M1CompactScene compact;
  std::string digest;
  m1fmt::SemanticHeaderV1 semantic{};
};

struct M1Point {
  std::int64_t x;
  std::int64_t y;
};

using M1PolygonPoints = std::vector<M1Point>;

const char *m1_disposition_name(M1Disposition disposition) {
  switch (disposition) {
    case M1Disposition::kComplete:
      return "COMPLETE";
    case M1Disposition::kRawHits:
      return "RAW_HITS";
    case M1Disposition::kUncertain:
      return "UNCERTAIN";
  }
  return "UNCERTAIN";
}

bool m1_checked_add_i64(std::int64_t a, std::int64_t b,
                        std::int64_t *result) {
  const __int128 sum = static_cast<__int128>(a) + b;
  if (sum < std::numeric_limits<std::int64_t>::min() ||
      sum > std::numeric_limits<std::int64_t>::max()) {
    return false;
  }
  *result = static_cast<std::int64_t>(sum);
  return true;
}

M1PolygonPoints m1_rectangle(std::int64_t left, std::int64_t bottom,
                             std::int64_t right, std::int64_t top) {
  if (left >= right || bottom >= top) {
    throw SceneError("fixture rectangle is empty");
  }
  // Clockwise: KLayout's normalized hull keeps its interior on the right.
  return {{left, bottom}, {left, top}, {right, top}, {right, bottom}};
}

M1PolygonPoints m1_notch(std::int64_t gap) {
  if (gap <= 0 || gap >= 300) {
    throw SceneError("fixture notch gap is out of range");
  }
  const std::int64_t notch_bottom = 185;
  const std::int64_t notch_top = notch_bottom + gap;
  return {{0, 0},
          {0, 500},
          {500, 500},
          {500, notch_top},
          {200, notch_top},
          {200, notch_bottom},
          {500, notch_bottom},
          {500, 0}};
}

M1CompactScene m1_make_scene(
    const std::string &name, const std::vector<M1PolygonPoints> &polygons,
    std::vector<ContextGpu> contexts = {{0, 0, 0, 0}}) {
  if (polygons.empty()) {
    throw SceneError("fixture scene has no polygons");
  }
  M1CompactScene scene;
  scene.name = name;
  scene.contexts = std::move(contexts);
  // This constructor is private to the deterministic synthetic fixtures and
  // benchmark below.  Their polygon sets are authored as a disjoint merged
  // universe; arbitrary packed/external scenes never inherit this assertion.
  scene.exact_merged_universe = true;

  M1CellTemplate cell{};
  cell.left = std::numeric_limits<std::int64_t>::max();
  cell.bottom = std::numeric_limits<std::int64_t>::max();
  cell.right = std::numeric_limits<std::int64_t>::min();
  cell.top = std::numeric_limits<std::int64_t>::min();
  for (std::size_t polygon_index = 0; polygon_index < polygons.size();
       ++polygon_index) {
    const M1PolygonPoints &points = polygons[polygon_index];
    if (points.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw SceneError("fixture polygon exceeds uint32 edge count");
    }
    M1PolygonTemplate polygon{};
    polygon.edge_begin = scene.edges.size();
    polygon.edge_count = static_cast<std::uint32_t>(points.size());
    for (std::size_t edge_index = 0; edge_index < points.size();
         ++edge_index) {
      const M1Point first = points[edge_index];
      const M1Point second = points[(edge_index + 1) % points.size()];
      scene.edges.push_back(
          {{first.x, first.y, second.x, second.y},
           static_cast<std::uint32_t>(polygon_index),
           static_cast<std::uint32_t>(edge_index)});
      cell.left = std::min(cell.left, first.x);
      cell.bottom = std::min(cell.bottom, first.y);
      cell.right = std::max(cell.right, first.x);
      cell.top = std::max(cell.top, first.y);
    }
    scene.polygons.push_back(polygon);
  }
  cell.polygon_begin = 0;
  cell.edge_begin = 0;
  cell.polygon_count = static_cast<std::uint32_t>(scene.polygons.size());
  cell.edge_count = static_cast<std::uint32_t>(scene.edges.size());
  scene.cells.push_back(cell);
  return scene;
}

M1CompactScene m1_scene_from_kact_layer0(
    const LoadedScene &packed, const M1Options &options) {
  if (!options.trust_merged_layer0) {
    throw SceneError(
        "packed layer 0 was not explicitly qualified as merged M1");
  }

  // Reuse the production-proven checked hierarchy expansion, then retain only
  // contexts whose cell has local layer-0 templates.  Descendant geometry is
  // represented by its own resolved context, so empty parent contexts carry
  // no edge work and can be dropped without flattening polygons.
  const LoweredScene hierarchy =
      lower_hierarchy(packed, options.max_contexts);
  std::vector<std::uint32_t> cell_map(
      packed.header.cell_count, std::numeric_limits<std::uint32_t>::max());
  M1CompactScene scene;
  scene.name = "packed_merged_m1";
  // This is an out-of-band producer/provenance assertion, not a property
  // inferred from KACT polygon bytes.  The caller reaches this conversion only
  // after load_and_validate has authenticated the complete KACT payload and
  // m1_run_packed_scene has matched its explicitly qualified scene digest.
  scene.exact_merged_universe = true;

  for (std::uint64_t old_cell_id = 0;
       old_cell_id < packed.header.cell_count; ++old_cell_id) {
    const CellRecord &old_cell = packed.cells[old_cell_id];
    std::uint32_t selected_polygons = 0;
    std::uint64_t selected_edges = 0;
    for (std::uint64_t local = 0; local < old_cell.polygon_count; ++local) {
      const PolygonRecord &polygon =
          packed.polygons[old_cell.polygon_begin + local];
      if (polygon.layer_code == 0) {
        ++selected_polygons;
        selected_edges += polygon.edge_count;
      }
    }
    if (!selected_polygons) {
      continue;
    }
    if (selected_edges > std::numeric_limits<std::uint32_t>::max() ||
        scene.cells.size() >= std::numeric_limits<std::uint32_t>::max()) {
      throw SceneError("packed per-cell M1 topology exceeds uint32");
    }

    const std::uint32_t new_cell_id =
        static_cast<std::uint32_t>(scene.cells.size());
    cell_map[old_cell_id] = new_cell_id;
    M1CellTemplate cell{};
    cell.polygon_begin = scene.polygons.size();
    cell.edge_begin = scene.edges.size();
    cell.polygon_count = selected_polygons;
    cell.edge_count = static_cast<std::uint32_t>(selected_edges);
    cell.left = std::numeric_limits<std::int64_t>::max();
    cell.bottom = std::numeric_limits<std::int64_t>::max();
    cell.right = std::numeric_limits<std::int64_t>::min();
    cell.top = std::numeric_limits<std::int64_t>::min();

    std::uint32_t polygon_local = 0;
    for (std::uint64_t local = 0; local < old_cell.polygon_count; ++local) {
      const PolygonRecord &source_polygon =
          packed.polygons[old_cell.polygon_begin + local];
      if (source_polygon.layer_code != 0) {
        continue;
      }
      M1PolygonTemplate polygon{};
      polygon.edge_begin = scene.edges.size();
      polygon.edge_count = source_polygon.edge_count;
      for (std::uint32_t edge_local = 0;
           edge_local < source_polygon.edge_count; ++edge_local) {
        const EdgeRecord &source =
            packed.edges[source_polygon.edge_begin + edge_local];
        scene.edges.push_back(
            {{source.x1, source.y1, source.x2, source.y2},
             polygon_local, edge_local});
      }
      scene.polygons.push_back(polygon);
      cell.left = std::min(cell.left, source_polygon.bbox[0]);
      cell.bottom = std::min(cell.bottom, source_polygon.bbox[1]);
      cell.right = std::max(cell.right, source_polygon.bbox[2]);
      cell.top = std::max(cell.top, source_polygon.bbox[3]);
      ++polygon_local;
    }
    if (polygon_local != selected_polygons ||
        scene.edges.size() != cell.edge_begin + selected_edges) {
      throw SceneError("packed M1 selection census mismatch");
    }
    scene.cells.push_back(cell);
  }
  if (scene.cells.empty()) {
    throw SceneError("packed layer 0 has no M1 polygon templates");
  }

  scene.contexts.reserve(hierarchy.contexts.size());
  for (ContextGpu context : hierarchy.contexts) {
    if (context.cell >= cell_map.size()) {
      throw SceneError("expanded packed context has invalid source cell");
    }
    const std::uint32_t mapped = cell_map[context.cell];
    if (mapped == std::numeric_limits<std::uint32_t>::max()) {
      continue;
    }
    context.cell = mapped;
    scene.contexts.push_back(context);
  }
  if (scene.contexts.empty()) {
    throw SceneError("packed layer 0 has no resolved M1 contexts");
  }
  return scene;
}

bool m1_ranges_overlap(std::int64_t a0, std::int64_t a1,
                       std::int64_t b0, std::int64_t b1) {
  return std::max(std::min(a0, a1), std::min(b0, b1)) <=
         std::min(std::max(a0, a1), std::max(b0, b1));
}

bool m1_segments_intersect(const M1DirectedEdge &a,
                           const M1DirectedEdge &b) {
  const bool ah = a.y1 == a.y2;
  const bool bh = b.y1 == b.y2;
  if (ah && bh) {
    return a.y1 == b.y1 && m1_ranges_overlap(a.x1, a.x2, b.x1, b.x2);
  }
  if (!ah && !bh) {
    return a.x1 == b.x1 && m1_ranges_overlap(a.y1, a.y2, b.y1, b.y2);
  }
  const M1DirectedEdge &h = ah ? a : b;
  const M1DirectedEdge &v = ah ? b : a;
  return std::min(h.x1, h.x2) <= v.x1 &&
         v.x1 <= std::max(h.x1, h.x2) &&
         std::min(v.y1, v.y2) <= h.y1 &&
         h.y1 <= std::max(v.y1, v.y2);
}

void m1_validate_polygon(const M1CompactScene &scene,
                         const M1PolygonTemplate &polygon,
                         std::uint32_t expected_polygon_local) {
  if (polygon.edge_count < 4 ||
      polygon.edge_begin > scene.edges.size() ||
      polygon.edge_count > scene.edges.size() - polygon.edge_begin) {
    throw SceneError("polygon edge range is invalid");
  }
  __int128 twice_area = 0;
  for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
    const M1EdgeTemplate &record = scene.edges[polygon.edge_begin + local];
    const M1DirectedEdge &edge = record.edge;
    const M1DirectedEdge &next =
        scene.edges[polygon.edge_begin + (local + 1) % polygon.edge_count]
            .edge;
    if (record.polygon_local != expected_polygon_local ||
        record.edge_local != local) {
      throw SceneError("edge topology metadata is not canonical");
    }
    if (edge.x2 != next.x1 || edge.y2 != next.y1) {
      throw SceneError("polygon contour is not closed");
    }
    const bool horizontal = edge.y1 == edge.y2 && edge.x1 != edge.x2;
    const bool vertical = edge.x1 == edge.x2 && edge.y1 != edge.y2;
    if (!(horizontal || vertical) || !bounded_coordinate(edge.x1) ||
        !bounded_coordinate(edge.y1) || !bounded_coordinate(edge.x2) ||
        !bounded_coordinate(edge.y2)) {
      throw SceneError("polygon contains unsupported geometry");
    }
    twice_area += static_cast<__int128>(edge.x1) * edge.y2 -
                  static_cast<__int128>(edge.x2) * edge.y1;
  }
  if (twice_area >= 0) {
    throw SceneError("polygon is not a nonempty clockwise hull");
  }

  // A self-crossing template has no trustworthy interior side.  Adjacent
  // edges share their expected endpoint and are excluded from this test.
  for (std::uint32_t first = 0; first < polygon.edge_count; ++first) {
    for (std::uint32_t second = first + 1; second < polygon.edge_count;
         ++second) {
      if (second == first + 1 ||
          (first == 0 && second + 1 == polygon.edge_count)) {
        continue;
      }
      if (m1_segments_intersect(
              scene.edges[polygon.edge_begin + first].edge,
              scene.edges[polygon.edge_begin + second].edge)) {
        throw SceneError("polygon contour self-intersects");
      }
    }
  }
}

std::array<std::int64_t, 4> m1_transform_box_host(
    const ContextGpu &context, const M1CellTemplate &cell) {
  const std::int64_t xs[4] = {
      cell.left, cell.left, cell.right, cell.right};
  const std::int64_t ys[4] = {
      cell.bottom, cell.top, cell.bottom, cell.top};
  std::array<std::int64_t, 4> result = {
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min()};
  for (int corner = 0; corner < 4; ++corner) {
    const auto transformed =
        transform_128(context.transform, xs[corner], ys[corner]);
    const std::int64_t x =
        narrow_i64(transformed.first + context.tx, "world M1 box x");
    const std::int64_t y =
        narrow_i64(transformed.second + context.ty, "world M1 box y");
    result[0] = std::min(result[0], x);
    result[1] = std::min(result[1], y);
    result[2] = std::max(result[2], x);
    result[3] = std::max(result[3], y);
  }
  return result;
}

M1LoweredScene m1_validate_and_lower(const M1CompactScene &scene,
                                     const M1Options &options) {
  if (!scene.exact_merged_universe) {
    throw SceneError("scene is not an exact merged-M1 universe");
  }
  if (scene.contexts.empty() || scene.contexts.size() > options.max_contexts ||
      scene.contexts.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("scene context count is empty or exceeds capacity");
  }
  if (scene.cells.empty() ||
      scene.cells.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("scene cell count is empty or exceeds uint32");
  }
  if (scene.edges.empty() || scene.edges.size() > options.max_edges ||
      scene.edges.size() > std::numeric_limits<std::uint32_t>::max() ||
      scene.polygons.empty() || scene.polygons.size() > options.max_polygons ||
      scene.polygons.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("stored M1 topology is empty or exceeds capacity");
  }

  std::uint64_t polygon_cursor = 0;
  std::uint64_t edge_cursor = 0;
  for (const M1CellTemplate &cell : scene.cells) {
    if (cell.polygon_begin != polygon_cursor ||
        cell.edge_begin != edge_cursor || !cell.polygon_count ||
        !cell.edge_count ||
        cell.polygon_count > scene.polygons.size() - polygon_cursor ||
        cell.edge_count > scene.edges.size() - edge_cursor ||
        cell.left >= cell.right || cell.bottom >= cell.top) {
      throw SceneError("cell topology is not a canonical partition");
    }
    std::uint64_t local_edge_cursor = cell.edge_begin;
    for (std::uint32_t local = 0; local < cell.polygon_count; ++local) {
      const M1PolygonTemplate &polygon =
          scene.polygons[cell.polygon_begin + local];
      if (polygon.edge_begin != local_edge_cursor) {
        throw SceneError("polygon edges are not canonically grouped");
      }
      m1_validate_polygon(scene, polygon, local);
      local_edge_cursor += polygon.edge_count;
    }
    if (local_edge_cursor != cell.edge_begin + cell.edge_count) {
      throw SceneError("cell/polygon edge censuses disagree");
    }
    polygon_cursor += cell.polygon_count;
    edge_cursor += cell.edge_count;
  }
  if (polygon_cursor != scene.polygons.size() ||
      edge_cursor != scene.edges.size()) {
    throw SceneError("cell table does not cover stored topology exactly");
  }

  M1LoweredScene lowered;
  lowered.contexts = scene.contexts;
  lowered.cells = scene.cells;
  lowered.edges = scene.edges;
  lowered.edge_offsets.reserve(scene.contexts.size());
  lowered.polygon_offsets.reserve(scene.contexts.size());
  bool have_box = false;
  for (const ContextGpu &context : scene.contexts) {
    if (context.cell >= scene.cells.size() || context.transform >= 8) {
      throw SceneError("context references an invalid cell or transform");
    }
    const M1CellTemplate &cell = scene.cells[context.cell];
    lowered.edge_offsets.push_back(lowered.edge_count);
    lowered.polygon_offsets.push_back(lowered.polygon_count);
    if (!checked_add_u64(
            lowered.edge_count, cell.edge_count, &lowered.edge_count) ||
        !checked_add_u64(lowered.polygon_count, cell.polygon_count,
                         &lowered.polygon_count) ||
        lowered.edge_count > options.max_edges ||
        lowered.polygon_count > options.max_polygons ||
        lowered.edge_count > std::numeric_limits<std::uint32_t>::max()) {
      throw SceneError("expanded M1 topology exceeds capacity");
    }
    const auto box = m1_transform_box_host(context, cell);
    if (!have_box) {
      lowered.left = box[0];
      lowered.bottom = box[1];
      lowered.right = box[2];
      lowered.top = box[3];
      have_box = true;
    } else {
      lowered.left = std::min(lowered.left, box[0]);
      lowered.bottom = std::min(lowered.bottom, box[1]);
      lowered.right = std::max(lowered.right, box[2]);
      lowered.top = std::max(lowered.top, box[3]);
    }
  }
  if (!have_box || !lowered.edge_count || !lowered.polygon_count) {
    throw SceneError("expanded M1 universe is empty");
  }
  return lowered;
}

bool m1_host_layout_equal(const m1fmt::FileLayoutV1 &first,
                          const m1fmt::FileLayoutV1 &second) {
  return first.file_bytes == second.file_bytes &&
         first.payload_offset == second.payload_offset &&
         first.payload_bytes == second.payload_bytes &&
         first.contexts_offset == second.contexts_offset &&
         first.metal_contexts_offset == second.metal_contexts_offset &&
         first.cells_offset == second.cells_offset &&
         first.polygons_offset == second.polygons_offset &&
         first.edges_offset == second.edges_offset;
}

bool m1_digest_is_nonzero(
    const std::array<std::uint8_t, m1fmt::kDigestBytes> &digest) {
  return std::any_of(digest.begin(), digest.end(),
                     [](std::uint8_t byte) { return byte != 0; });
}

bool m1_host_range_fits(std::uint64_t begin, std::uint64_t count,
                        std::uint64_t total) {
  std::uint64_t end = 0;
  return checked_add_u64(begin, count, &end) && end <= total;
}

M1LoadedHostScene m1_load_host_scene(const std::string &path,
                                     const M1Options &options) {
  if (options.expected_scene_sha256 != kM1QualifiedHostSceneSha256) {
    throw SceneError(
        "host-scene digest is not the qualified production M1 scene");
  }

  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw SceneError("cannot open M1 host scene: " + path);
  }
  const std::streamoff file_end = input.tellg();
  if (file_end < static_cast<std::streamoff>(m1fmt::kFileHeaderBytes)) {
    throw SceneError("M1 host scene is shorter than its fixed header");
  }
  if (static_cast<std::uint64_t>(file_end) >
          std::numeric_limits<std::size_t>::max() ||
      file_end > std::numeric_limits<std::streamsize>::max()) {
    throw SceneError("M1 host scene is too large for this host");
  }
  std::vector<std::uint8_t> storage(static_cast<std::size_t>(file_end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(storage.data()),
                  static_cast<std::streamsize>(file_end))) {
    throw SceneError("short read while loading M1 host scene");
  }

  m1fmt::FileHeaderV1 header{};
  if (!m1fmt::decode_file_header(storage.data(), storage.size(), header)) {
    throw SceneError("bad KM1WSCN1 file header or magic");
  }
  if (header.version != m1fmt::kFileVersion ||
      header.header_bytes != m1fmt::kFileHeaderBytes ||
      header.endian_tag != m1fmt::kEndianTag ||
      header.flags != m1fmt::kRequiredFlags ||
      header.source_layer != 101 || header.source_datatype != 0 ||
      header.reserved[0] || header.reserved[1] || header.reserved[2]) {
    throw SceneError("unsupported or unqualified KM1WSCN1 header");
  }

  m1fmt::FileLayoutV1 canonical_layout{};
  if (!m1fmt::compute_file_layout(
          header.context_count, header.metal_context_count,
          header.cell_count, header.polygon_count, header.edge_count,
          canonical_layout) ||
      !m1_host_layout_equal(header.layout, canonical_layout) ||
      header.layout.file_bytes != storage.size()) {
    throw SceneError("KM1WSCN1 layout is not canonical");
  }
  if (header.context_count != kM1QualifiedContextCount ||
      header.metal_context_count != kM1QualifiedMetalContextCount ||
      header.cell_count != kM1QualifiedCellCount ||
      header.polygon_count != kM1QualifiedStoredPolygonCount ||
      header.edge_count != kM1QualifiedStoredEdgeCount) {
    throw SceneError("KM1WSCN1 stored census is not the qualified scene");
  }
  if (!m1_digest_is_nonzero(header.scene_digest) ||
      !m1_digest_is_nonzero(header.transport_digest) ||
      !m1_digest_is_nonzero(header.source_digest)) {
    throw SceneError("KM1WSCN1 contains a missing provenance digest");
  }
  if (hex_digest(header.source_digest.data(), header.source_digest.size()) !=
      kM1QualifiedSourceGdsSha256) {
    throw SceneError("KM1WSCN1 source GDS digest is not qualified");
  }

  Sha256 transport_sha;
  transport_sha.update(storage.data(), m1fmt::kTransportDigestOffset);
  const std::array<std::uint8_t, m1fmt::kDigestBytes> zero_digest{};
  transport_sha.update(zero_digest.data(), zero_digest.size());
  transport_sha.update(
      storage.data() + m1fmt::kSourceDigestOffset,
      storage.size() - m1fmt::kSourceDigestOffset);
  const auto computed_transport = transport_sha.finish();
  if (!std::equal(computed_transport.begin(), computed_transport.end(),
                  header.transport_digest.begin())) {
    throw SceneError("KM1WSCN1 transport SHA-256 mismatch");
  }

  Sha256 payload_sha;
  payload_sha.update(
      storage.data() + header.layout.payload_offset,
      static_cast<std::size_t>(header.layout.payload_bytes));
  const auto computed_payload = payload_sha.finish();
  if (!std::equal(computed_payload.begin(), computed_payload.end(),
                  header.scene_digest.begin())) {
    throw SceneError("KM1WSCN1 canonical payload SHA-256 mismatch");
  }
  const std::string digest =
      hex_digest(header.scene_digest.data(), header.scene_digest.size());
  if (digest != options.expected_scene_sha256) {
    throw SceneError("KM1WSCN1 scene SHA-256 does not match expectation");
  }

  m1fmt::SemanticHeaderV1 semantic{};
  if (!m1fmt::decode_semantic_header(
          storage.data() + header.layout.payload_offset,
          static_cast<std::size_t>(header.layout.payload_bytes), semantic)) {
    throw SceneError("bad KM1WSCN1 semantic header or magic");
  }
  if (semantic.format_version != 1 || semantic.dbu_per_micron != 2000 ||
      semantic.root_cell != 0 || semantic.reserved != 0 ||
      semantic.width_distance != kM1Distance ||
      semantic.spacing_distance != kM1Distance ||
      semantic.context_count != header.context_count ||
      semantic.metal_context_count != header.metal_context_count ||
      semantic.cell_count != header.cell_count ||
      semantic.polygon_count != header.polygon_count ||
      semantic.edge_count != header.edge_count ||
      semantic.flat_polygon_count != kM1QualifiedFlatPolygonCount ||
      semantic.flat_edge_count != kM1QualifiedFlatEdgeCount ||
      semantic.scene_left >= semantic.scene_right ||
      semantic.scene_bottom >= semantic.scene_top ||
      !bounded_coordinate(semantic.scene_left) ||
      !bounded_coordinate(semantic.scene_bottom) ||
      !bounded_coordinate(semantic.scene_right) ||
      !bounded_coordinate(semantic.scene_top)) {
    throw SceneError("KM1WSCN1 semantic header is outside qualification");
  }
  if (semantic.metal_context_count > options.max_contexts ||
      semantic.polygon_count > options.max_polygons ||
      semantic.edge_count > options.max_edges ||
      semantic.flat_polygon_count > options.max_polygons ||
      semantic.flat_edge_count > options.max_edges ||
      semantic.metal_context_count >
          std::numeric_limits<std::uint32_t>::max() ||
      semantic.cell_count > std::numeric_limits<std::uint32_t>::max() ||
      semantic.polygon_count > std::numeric_limits<std::uint32_t>::max() ||
      semantic.edge_count > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("KM1WSCN1 census exceeds configured/uint32 capacity");
  }

  std::vector<ContextGpu> all_contexts;
  all_contexts.reserve(static_cast<std::size_t>(header.context_count));
  for (std::uint64_t context_id = 0; context_id < header.context_count;
       ++context_id) {
    const std::uint8_t *record =
        storage.data() + header.layout.contexts_offset +
        context_id * m1fmt::kContextRecordBytes;
    ContextGpu context{};
    context.tx = m1fmt::load_i64_le(record);
    context.ty = m1fmt::load_i64_le(record + 8);
    context.cell = m1fmt::load_u32_le(record + 16);
    context.transform = m1fmt::load_u32_le(record + 20);
    if (context.cell >= header.cell_count || context.transform >= 8 ||
        !bounded_coordinate(context.tx) || !bounded_coordinate(context.ty)) {
      throw SceneError("KM1WSCN1 context is invalid");
    }
    all_contexts.push_back(context);
  }
  const ContextGpu &root = all_contexts.front();
  if (root.tx != 0 || root.ty != 0 ||
      root.cell != semantic.root_cell || root.transform != 0) {
    throw SceneError("KM1WSCN1 root context is not canonical");
  }

  std::vector<M1HostCellRecord> host_cells;
  host_cells.reserve(static_cast<std::size_t>(header.cell_count));
  std::set<std::uint64_t> source_cells;
  std::uint64_t next_polygon = 0;
  std::uint64_t next_edge = 0;
  for (std::uint64_t cell_id = 0; cell_id < header.cell_count; ++cell_id) {
    const std::uint8_t *record =
        storage.data() + header.layout.cells_offset +
        cell_id * m1fmt::kCellRecordBytes;
    M1HostCellRecord cell{};
    cell.source_cell_index = m1fmt::load_u64_le(record);
    cell.polygon_begin = m1fmt::load_u64_le(record + 8);
    cell.edge_begin = m1fmt::load_u64_le(record + 16);
    cell.polygon_count = m1fmt::load_u32_le(record + 24);
    cell.edge_count = m1fmt::load_u32_le(record + 28);
    if (!source_cells.insert(cell.source_cell_index).second ||
        cell.polygon_begin != next_polygon ||
        cell.edge_begin != next_edge ||
        !m1_host_range_fits(
            cell.polygon_begin, cell.polygon_count, header.polygon_count) ||
        !m1_host_range_fits(
            cell.edge_begin, cell.edge_count, header.edge_count) ||
        ((!cell.polygon_count) != (!cell.edge_count))) {
      throw SceneError("KM1WSCN1 cell partition is not canonical");
    }
    next_polygon += cell.polygon_count;
    next_edge += cell.edge_count;
    host_cells.push_back(cell);
  }
  if (next_polygon != header.polygon_count ||
      next_edge != header.edge_count) {
    throw SceneError("KM1WSCN1 cell table does not cover stored topology");
  }

  M1LoadedHostScene loaded;
  loaded.digest = digest;
  loaded.semantic = semantic;
  M1CompactScene &scene = loaded.compact;
  scene.name = "host_merged_m1";
  std::vector<std::uint32_t> cell_map(
      static_cast<std::size_t>(header.cell_count),
      std::numeric_limits<std::uint32_t>::max());
  scene.polygons.reserve(static_cast<std::size_t>(header.polygon_count));
  scene.edges.reserve(static_cast<std::size_t>(header.edge_count));

  for (std::uint32_t old_cell_id = 0;
       old_cell_id < host_cells.size(); ++old_cell_id) {
    const M1HostCellRecord &source_cell = host_cells[old_cell_id];
    if (!source_cell.polygon_count) {
      continue;
    }
    const std::uint32_t new_cell_id =
        static_cast<std::uint32_t>(scene.cells.size());
    cell_map[old_cell_id] = new_cell_id;
    M1CellTemplate cell{};
    cell.polygon_begin = scene.polygons.size();
    cell.edge_begin = scene.edges.size();
    cell.polygon_count = source_cell.polygon_count;
    cell.edge_count = source_cell.edge_count;
    cell.left = std::numeric_limits<std::int64_t>::max();
    cell.bottom = std::numeric_limits<std::int64_t>::max();
    cell.right = std::numeric_limits<std::int64_t>::min();
    cell.top = std::numeric_limits<std::int64_t>::min();

    std::uint64_t cell_edge_cursor = source_cell.edge_begin;
    const std::uint64_t source_cell_edge_end =
        source_cell.edge_begin + source_cell.edge_count;
    for (std::uint32_t polygon_local = 0;
         polygon_local < source_cell.polygon_count; ++polygon_local) {
      const std::uint64_t polygon_id =
          source_cell.polygon_begin + polygon_local;
      const std::uint8_t *polygon_record =
          storage.data() + header.layout.polygons_offset +
          polygon_id * m1fmt::kPolygonRecordBytes;
      const std::uint64_t polygon_edge_begin =
          m1fmt::load_u64_le(polygon_record);
      const std::int64_t polygon_left =
          m1fmt::load_i64_le(polygon_record + 8);
      const std::int64_t polygon_bottom =
          m1fmt::load_i64_le(polygon_record + 16);
      const std::int64_t polygon_right =
          m1fmt::load_i64_le(polygon_record + 24);
      const std::int64_t polygon_top =
          m1fmt::load_i64_le(polygon_record + 32);
      const std::uint32_t stored_polygon_id =
          m1fmt::load_u32_le(polygon_record + 40);
      const std::uint32_t polygon_edge_count =
          m1fmt::load_u32_le(polygon_record + 44);
      if (polygon_edge_begin != cell_edge_cursor ||
          stored_polygon_id != polygon_local || polygon_edge_count < 4 ||
          !m1_host_range_fits(
              polygon_edge_begin, polygon_edge_count,
              source_cell_edge_end) ||
          polygon_left >= polygon_right ||
          polygon_bottom >= polygon_top ||
          !bounded_coordinate(polygon_left) ||
          !bounded_coordinate(polygon_bottom) ||
          !bounded_coordinate(polygon_right) ||
          !bounded_coordinate(polygon_top)) {
        throw SceneError("KM1WSCN1 polygon record is invalid");
      }

      M1PolygonTemplate polygon{};
      polygon.edge_begin = scene.edges.size();
      polygon.edge_count = polygon_edge_count;
      std::int64_t recomputed_left =
          std::numeric_limits<std::int64_t>::max();
      std::int64_t recomputed_bottom =
          std::numeric_limits<std::int64_t>::max();
      std::int64_t recomputed_right =
          std::numeric_limits<std::int64_t>::min();
      std::int64_t recomputed_top =
          std::numeric_limits<std::int64_t>::min();
      M1DirectedEdge first_edge{};
      M1DirectedEdge previous_edge{};
      __int128 twice_area = 0;
      for (std::uint32_t edge_local = 0;
           edge_local < polygon_edge_count; ++edge_local) {
        const std::uint64_t edge_id = polygon_edge_begin + edge_local;
        const std::uint8_t *edge_record =
            storage.data() + header.layout.edges_offset +
            edge_id * m1fmt::kEdgeRecordBytes;
        const M1DirectedEdge edge = {
            m1fmt::load_i64_le(edge_record),
            m1fmt::load_i64_le(edge_record + 8),
            m1fmt::load_i64_le(edge_record + 16),
            m1fmt::load_i64_le(edge_record + 24)};
        const bool horizontal =
            edge.y1 == edge.y2 && edge.x1 != edge.x2;
        const bool vertical =
            edge.x1 == edge.x2 && edge.y1 != edge.y2;
        if (!(horizontal || vertical) ||
            !bounded_coordinate(edge.x1) ||
            !bounded_coordinate(edge.y1) ||
            !bounded_coordinate(edge.x2) ||
            !bounded_coordinate(edge.y2) ||
            (edge_local &&
             (previous_edge.x2 != edge.x1 ||
              previous_edge.y2 != edge.y1))) {
          throw SceneError("KM1WSCN1 edge contour is invalid");
        }
        if (!edge_local) {
          first_edge = edge;
        }
        previous_edge = edge;
        recomputed_left =
            std::min(recomputed_left, std::min(edge.x1, edge.x2));
        recomputed_bottom =
            std::min(recomputed_bottom, std::min(edge.y1, edge.y2));
        recomputed_right =
            std::max(recomputed_right, std::max(edge.x1, edge.x2));
        recomputed_top =
            std::max(recomputed_top, std::max(edge.y1, edge.y2));
        twice_area += static_cast<__int128>(edge.x1) * edge.y2 -
                      static_cast<__int128>(edge.x2) * edge.y1;
        scene.edges.push_back(
            {edge, polygon_local, edge_local});
      }
      if (previous_edge.x2 != first_edge.x1 ||
          previous_edge.y2 != first_edge.y1 || twice_area >= 0 ||
          recomputed_left != polygon_left ||
          recomputed_bottom != polygon_bottom ||
          recomputed_right != polygon_right ||
          recomputed_top != polygon_top) {
        throw SceneError(
            "KM1WSCN1 polygon closure/orientation/bbox is invalid");
      }
      scene.polygons.push_back(polygon);
      cell.left = std::min(cell.left, polygon_left);
      cell.bottom = std::min(cell.bottom, polygon_bottom);
      cell.right = std::max(cell.right, polygon_right);
      cell.top = std::max(cell.top, polygon_top);
      cell_edge_cursor += polygon_edge_count;
    }
    if (cell_edge_cursor != source_cell_edge_end ||
        scene.polygons.size() !=
            cell.polygon_begin + cell.polygon_count ||
        scene.edges.size() != cell.edge_begin + cell.edge_count) {
      throw SceneError("KM1WSCN1 cell/polygon censuses disagree");
    }
    scene.cells.push_back(cell);
  }
  if (scene.cells.empty() ||
      scene.polygons.size() != header.polygon_count ||
      scene.edges.size() != header.edge_count) {
    throw SceneError("KM1WSCN1 M1 template universe is incomplete");
  }

  scene.contexts.reserve(
      static_cast<std::size_t>(header.metal_context_count));
  std::uint64_t metal_index = 0;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_edges = 0;
  bool have_scene_box = false;
  std::array<std::int64_t, 4> scene_box{};
  for (std::uint32_t context_id = 0;
       context_id < all_contexts.size(); ++context_id) {
    const ContextGpu &source_context = all_contexts[context_id];
    const M1HostCellRecord &host_cell =
        host_cells[source_context.cell];
    if (!host_cell.polygon_count) {
      continue;
    }
    if (metal_index >= header.metal_context_count) {
      throw SceneError("KM1WSCN1 metal-context table is truncated");
    }
    const std::uint8_t *record =
        storage.data() + header.layout.metal_contexts_offset +
        metal_index * m1fmt::kMetalContextRecordBytes;
    const std::uint32_t stored_context_id =
        m1fmt::load_u32_le(record);
    const std::uint64_t polygon_offset =
        m1fmt::load_u64_le(record + 4);
    const std::uint64_t edge_offset =
        m1fmt::load_u64_le(record + 12);
    if (stored_context_id != context_id ||
        polygon_offset != flat_polygons || edge_offset != flat_edges ||
        !checked_add_u64(
            flat_polygons, host_cell.polygon_count, &flat_polygons) ||
        !checked_add_u64(
            flat_edges, host_cell.edge_count, &flat_edges)) {
      throw SceneError(
          "KM1WSCN1 metal-context list/offsets are not canonical");
    }
    ContextGpu context = source_context;
    context.cell = cell_map[source_context.cell];
    if (context.cell == std::numeric_limits<std::uint32_t>::max()) {
      throw SceneError("KM1WSCN1 nonempty context lost its cell mapping");
    }
    scene.contexts.push_back(context);
    const auto box = m1_transform_box_host(
        context, scene.cells[context.cell]);
    if (!have_scene_box) {
      scene_box = box;
      have_scene_box = true;
    } else {
      scene_box[0] = std::min(scene_box[0], box[0]);
      scene_box[1] = std::min(scene_box[1], box[1]);
      scene_box[2] = std::max(scene_box[2], box[2]);
      scene_box[3] = std::max(scene_box[3], box[3]);
    }
    ++metal_index;
  }
  if (metal_index != header.metal_context_count ||
      scene.contexts.size() != header.metal_context_count ||
      flat_polygons != semantic.flat_polygon_count ||
      flat_edges != semantic.flat_edge_count || !have_scene_box ||
      scene_box[0] != semantic.scene_left ||
      scene_box[1] != semantic.scene_bottom ||
      scene_box[2] != semantic.scene_right ||
      scene_box[3] != semantic.scene_top) {
    throw SceneError("KM1WSCN1 flat census or scene bbox mismatch");
  }

  // This bit is set only after the complete transport/payload digests,
  // pinned production identity, merged assertion, semantic qualification,
  // and canonical topology/context censuses have all been checked.
  scene.exact_merged_universe = true;
  return loaded;
}

M1Grid m1_make_grid(const M1LoweredScene &scene,
                    const M1Options &options, std::uint64_t *cell_count) {
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
  if (!m1_checked_add_i64(scene.left, -kM1Distance, &left) ||
      !m1_checked_add_i64(scene.bottom, -kM1Distance, &bottom) ||
      !m1_checked_add_i64(scene.right, kM1Distance, &right) ||
      !m1_checked_add_i64(scene.top, kM1Distance, &top)) {
    throw SceneError("expanded M1 scene box overflows");
  }
  const std::int64_t x0 = host_floor_div(left, kM1GridCell);
  const std::int64_t y0 = host_floor_div(bottom, kM1GridCell);
  const std::int64_t x1 = host_floor_div(right, kM1GridCell);
  const std::int64_t y1 = host_floor_div(top, kM1GridCell);
  const __int128 width = static_cast<__int128>(x1) - x0 + 1;
  const __int128 height = static_cast<__int128>(y1) - y0 + 1;
  if (width <= 0 || height <= 0 ||
      width > std::numeric_limits<std::uint32_t>::max() ||
      height > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("M1 grid dimensions exceed uint32");
  }
  const __int128 cells = width * height;
  if (cells <= 0 || cells > options.max_grid_cells ||
      cells > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("M1 grid cell count exceeds configured/uint32 capacity");
  }
  *cell_count = static_cast<std::uint64_t>(cells);
  return {x0, y0, static_cast<std::uint32_t>(width),
          static_cast<std::uint32_t>(height)};
}

__device__ bool m1_transform_edge_checked(
    const ContextGpu &context, const M1DirectedEdge &source,
    M1DirectedEdge *destination) {
  M1DirectedEdge transformed{};
  if (!transform_point_checked(context, source.x1, source.y1,
                               &transformed.x1, &transformed.y1) ||
      !transform_point_checked(context, source.x2, source.y2,
                               &transformed.x2, &transformed.y2)) {
    return false;
  }
  // Reflections reverse a normalized hull.  Restore clockwise/right-interior
  // edge direction just as the live ACTIVE.3 lowerer does.
  if (context.transform >= 4) {
    destination->x1 = transformed.x2;
    destination->y1 = transformed.y2;
    destination->x2 = transformed.x1;
    destination->y2 = transformed.y1;
  } else {
    *destination = transformed;
  }
  return true;
}

M1DirectedEdge m1_transform_edge_host(const ContextGpu &context,
                                      const M1DirectedEdge &source) {
  const auto first = transform_128(context.transform, source.x1, source.y1);
  const auto second =
      transform_128(context.transform, source.x2, source.y2);
  M1DirectedEdge transformed = {
      narrow_i64(first.first + context.tx, "host edge x1"),
      narrow_i64(first.second + context.ty, "host edge y1"),
      narrow_i64(second.first + context.tx, "host edge x2"),
      narrow_i64(second.second + context.ty, "host edge y2")};
  if (context.transform >= 4) {
    std::swap(transformed.x1, transformed.x2);
    std::swap(transformed.y1, transformed.y2);
  }
  return transformed;
}

__device__ bool m1_add_checked(std::int64_t a, std::int64_t b,
                               std::int64_t *result) {
  return add_checked(a, b, result);
}

__device__ std::int64_t m1_floor_div(std::int64_t value) {
  std::int64_t quotient = value / kM1GridCell;
  if (value % kM1GridCell < 0) {
    --quotient;
  }
  return quotient;
}

__device__ bool m1_edge_span(const M1ExpandedEdge &record,
                             const M1Grid &grid, std::int64_t *x0,
                             std::int64_t *y0, std::int64_t *x1,
                             std::int64_t *y1) {
  std::int64_t left = min(record.edge.x1, record.edge.x2);
  std::int64_t bottom = min(record.edge.y1, record.edge.y2);
  std::int64_t right = max(record.edge.x1, record.edge.x2);
  std::int64_t top = max(record.edge.y1, record.edge.y2);
  if (!m1_add_checked(left, -kM1Distance, &left) ||
      !m1_add_checked(bottom, -kM1Distance, &bottom) ||
      !m1_add_checked(right, kM1Distance, &right) ||
      !m1_add_checked(top, kM1Distance, &top)) {
    return false;
  }
  *x0 = m1_floor_div(left);
  *y0 = m1_floor_div(bottom);
  *x1 = m1_floor_div(right);
  *y1 = m1_floor_div(top);
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  return *x0 >= grid.base_x && *y0 >= grid.base_y &&
         *x1 <= maximum_x && *y1 <= maximum_y;
}

__device__ std::uint64_t m1_grid_index(const M1Grid &grid,
                                       std::int64_t x,
                                       std::int64_t y) {
  return static_cast<std::uint64_t>(y - grid.base_y) * grid.width +
         static_cast<std::uint64_t>(x - grid.base_x);
}

__global__ void m1_expand_edges_kernel(
    const ContextGpu *contexts, const std::uint64_t *edge_offsets,
    const std::uint64_t *polygon_offsets, const M1CellTemplate *cells,
    const M1EdgeTemplate *templates, std::uint32_t context_count,
    M1ExpandedEdge *expanded, M1DeviceCounters *counters,
    std::uint32_t *status) {
  const std::uint32_t context_id = blockIdx.x;
  if (context_id >= context_count) {
    return;
  }
  const ContextGpu context = contexts[context_id];
  const M1CellTemplate cell = cells[context.cell];
  unsigned long long local_expanded = 0;
  for (std::uint32_t local = threadIdx.x; local < cell.edge_count;
       local += blockDim.x) {
    const M1EdgeTemplate source = templates[cell.edge_begin + local];
    if (source.polygon_local >= cell.polygon_count) {
      atomicOr(status, static_cast<std::uint32_t>(kM1InvalidRecord));
      continue;
    }
    M1DirectedEdge edge{};
    if (!m1_transform_edge_checked(context, source.edge, &edge)) {
      atomicOr(status, static_cast<std::uint32_t>(kM1TransformOverflow));
      continue;
    }
    expanded[edge_offsets[context_id] + local] = {
        edge, polygon_offsets[context_id] + source.polygon_local,
        context_id, source.edge_local};
    ++local_expanded;
  }
  if (local_expanded) {
    atomicAdd(&counters->expanded_edges, local_expanded);
  }
}

__global__ void m1_count_grid_kernel(
    const M1ExpandedEdge *edges, std::uint32_t edge_count, M1Grid grid,
    std::uint32_t *counts, unsigned long long *membership_total,
    std::uint32_t *status) {
  for (std::uint32_t edge_id = blockIdx.x * blockDim.x + threadIdx.x;
       edge_id < edge_count; edge_id += blockDim.x * gridDim.x) {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!m1_edge_span(edges[edge_id], grid, &x0, &y0, &x1, &y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kM1InvalidRecord));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = m1_grid_index(grid, x, y);
        const std::uint32_t previous = atomicAdd(counts + cell, 1u);
        if (previous == UINT32_MAX) {
          atomicOr(status,
                   static_cast<std::uint32_t>(kM1GridCounterOverflow));
        }
        atomicAdd(membership_total, 1ull);
      }
    }
  }
}

__global__ void m1_fill_grid_kernel(
    const M1ExpandedEdge *edges, std::uint32_t edge_count, M1Grid grid,
    std::uint32_t *cursors, std::uint32_t *members,
    std::uint64_t member_capacity, std::uint32_t *status) {
  for (std::uint32_t edge_id = blockIdx.x * blockDim.x + threadIdx.x;
       edge_id < edge_count; edge_id += blockDim.x * gridDim.x) {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!m1_edge_span(edges[edge_id], grid, &x0, &y0, &x1, &y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kM1InvalidRecord));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = m1_grid_index(grid, x, y);
        const std::uint32_t position = atomicAdd(cursors + cell, 1u);
        if (position >= member_capacity) {
          atomicOr(status,
                   static_cast<std::uint32_t>(kM1GridCapacityExceeded));
        } else {
          members[position] = edge_id;
        }
      }
    }
  }
}

__global__ void m1_validate_grid_kernel(
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *cursors, std::uint64_t cell_count,
    std::uint32_t *status) {
  for (std::uint64_t cell =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < cell_count;
       cell += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t expected =
        static_cast<std::uint64_t>(offsets[cell]) + counts[cell];
    if (expected > UINT32_MAX || cursors[cell] != expected) {
      atomicOr(status,
               static_cast<std::uint32_t>(kM1GridCounterOverflow));
    }
  }
}

__global__ void m1_count_pair_work_kernel(
    const std::uint32_t *counts, std::uint64_t cell_count,
    unsigned long long *pair_work, std::uint32_t *status) {
  for (std::uint64_t cell =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < cell_count;
       cell += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const unsigned long long count = counts[cell];
    const unsigned long long pairs = count * (count - (count != 0)) / 2;
    const unsigned long long previous = atomicAdd(pair_work, pairs);
    if (previous > ULLONG_MAX - pairs) {
      atomicOr(status,
               static_cast<std::uint32_t>(kM1PairCounterOverflow));
    }
  }
}

__device__ void m1_record_sample(
    M1Sample *samples, std::uint32_t *sample_count,
    std::uint32_t first_edge, std::uint32_t second_edge, M1Rule rule,
    M1Verdict verdict, std::uint64_t first_polygon,
    std::uint64_t second_polygon) {
  const std::uint32_t slot = atomicAdd(sample_count, 1u);
  if (slot < kM1SampleCapacity) {
    samples[slot] = {
        first_edge, second_edge, static_cast<std::uint32_t>(rule),
        static_cast<std::uint32_t>(verdict), first_polygon, second_polygon};
  }
}

__global__ void m1_query_pairs_kernel(
    const M1ExpandedEdge *edges, M1Grid grid,
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *members, std::uint64_t cell_count,
    M1DeviceCounters *counters, M1Sample *samples,
    std::uint32_t *sample_count, std::uint32_t *status) {
  const std::uint64_t cell_id = blockIdx.x;
  if (cell_id >= cell_count) {
    return;
  }
  const std::uint32_t count = counts[cell_id];
  const std::uint32_t begin = offsets[cell_id];
  const std::int64_t cell_x =
      grid.base_x + static_cast<std::int64_t>(cell_id % grid.width);
  const std::int64_t cell_y =
      grid.base_y + static_cast<std::int64_t>(cell_id / grid.width);

  unsigned long long local_unique = 0;
  unsigned long long local_width_pairs = 0;
  unsigned long long local_space_pairs = 0;
  unsigned long long local_width_hits = 0;
  unsigned long long local_space_hits = 0;
  unsigned long long local_width_uncertain = 0;
  unsigned long long local_space_uncertain = 0;
  for (std::uint32_t first_local = threadIdx.x; first_local < count;
       first_local += blockDim.x) {
    const std::uint32_t first_id = members[begin + first_local];
    const M1ExpandedEdge first = edges[first_id];
    std::int64_t first_x0 = 0;
    std::int64_t first_y0 = 0;
    std::int64_t first_x1 = 0;
    std::int64_t first_y1 = 0;
    if (!m1_edge_span(
            first, grid, &first_x0, &first_y0, &first_x1, &first_y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kM1InvalidRecord));
      continue;
    }
    for (std::uint32_t second_local = first_local + 1;
         second_local < count; ++second_local) {
      const std::uint32_t second_id = members[begin + second_local];
      if (first_id == second_id) {
        continue;
      }
      const M1ExpandedEdge second = edges[second_id];
      std::int64_t second_x0 = 0;
      std::int64_t second_y0 = 0;
      std::int64_t second_x1 = 0;
      std::int64_t second_y1 = 0;
      if (!m1_edge_span(
              second, grid, &second_x0, &second_y0,
              &second_x1, &second_y1)) {
        atomicOr(status, static_cast<std::uint32_t>(kM1InvalidRecord));
        continue;
      }
      // Expanded boxes can meet in several grid cells.  The componentwise
      // lowest common cell owns the unordered pair exactly once.
      if (cell_x != max(first_x0, second_x0) ||
          cell_y != max(first_y0, second_y0)) {
        continue;
      }
      ++local_unique;

      if (first.polygon_id == second.polygon_id) {
        ++local_width_pairs;
        const M1CandidatePair pair = {
            first.edge, second.edge, first.polygon_id, second.polygon_id,
            M1Rule::kWidth};
        const M1Verdict verdict =
            m1p::classify_pair_bounded(pair, kM1Distance);
        if (verdict == M1Verdict::kViolation) {
          ++local_width_hits;
          m1_record_sample(
              samples, sample_count, first_id, second_id, M1Rule::kWidth,
              verdict, first.polygon_id, second.polygon_id);
        } else if (verdict == M1Verdict::kUncertain) {
          ++local_width_uncertain;
          m1_record_sample(
              samples, sample_count, first_id, second_id, M1Rule::kWidth,
              verdict, first.polygon_id, second.polygon_id);
        }
      }

      ++local_space_pairs;
      const M1CandidatePair pair = {
          first.edge, second.edge, first.polygon_id, second.polygon_id,
          M1Rule::kSpace};
      const M1Verdict verdict =
          m1p::classify_pair_bounded(pair, kM1Distance);
      if (verdict == M1Verdict::kViolation) {
        ++local_space_hits;
        m1_record_sample(
            samples, sample_count, first_id, second_id, M1Rule::kSpace,
            verdict, first.polygon_id, second.polygon_id);
      } else if (verdict == M1Verdict::kUncertain) {
        ++local_space_uncertain;
        m1_record_sample(
            samples, sample_count, first_id, second_id, M1Rule::kSpace,
            verdict, first.polygon_id, second.polygon_id);
      }
    }
  }
  if (local_unique) {
    atomicAdd(&counters->unique_edge_pairs, local_unique);
  }
  if (local_width_pairs) {
    atomicAdd(&counters->width_pairs, local_width_pairs);
  }
  if (local_space_pairs) {
    atomicAdd(&counters->space_pairs, local_space_pairs);
  }
  if (local_width_hits) {
    atomicAdd(&counters->width_hits, local_width_hits);
  }
  if (local_space_hits) {
    atomicAdd(&counters->space_hits, local_space_hits);
  }
  if (local_width_uncertain) {
    atomicAdd(&counters->width_uncertain, local_width_uncertain);
  }
  if (local_space_uncertain) {
    atomicAdd(&counters->space_uncertain, local_space_uncertain);
  }
}

std::vector<M1ExpandedEdge> m1_expand_host(
    const M1LoweredScene &scene) {
  std::vector<M1ExpandedEdge> expanded(scene.edge_count);
  for (std::uint32_t context_id = 0;
       context_id < scene.contexts.size(); ++context_id) {
    const ContextGpu context = scene.contexts[context_id];
    const M1CellTemplate &cell = scene.cells[context.cell];
    for (std::uint32_t local = 0; local < cell.edge_count; ++local) {
      const M1EdgeTemplate &source = scene.edges[cell.edge_begin + local];
      expanded[scene.edge_offsets[context_id] + local] = {
          m1_transform_edge_host(context, source.edge),
          scene.polygon_offsets[context_id] + source.polygon_local,
          context_id, source.edge_local};
    }
  }
  return expanded;
}

M1HostOracle m1_host_oracle(const M1LoweredScene &scene,
                            std::uint64_t max_pairs) {
  const std::vector<M1ExpandedEdge> edges = m1_expand_host(scene);
  const __int128 pairs =
      static_cast<__int128>(edges.size()) * (edges.size() - 1) / 2;
  if (pairs > max_pairs) {
    throw SceneError("host oracle pair count exceeds configured capacity");
  }
  M1HostOracle oracle;
  for (std::size_t first_id = 0; first_id < edges.size(); ++first_id) {
    for (std::size_t second_id = first_id + 1;
         second_id < edges.size(); ++second_id) {
      const M1ExpandedEdge &first = edges[first_id];
      const M1ExpandedEdge &second = edges[second_id];
      if (first.polygon_id == second.polygon_id) {
        const M1CandidatePair width = {
            first.edge, second.edge, first.polygon_id, second.polygon_id,
            M1Rule::kWidth};
        const M1Verdict verdict =
            m1p::classify_pair_bounded(width, kM1Distance);
        oracle.width_hits += verdict == M1Verdict::kViolation;
        oracle.width_uncertain += verdict == M1Verdict::kUncertain;
      }
      const M1CandidatePair space = {
          first.edge, second.edge, first.polygon_id, second.polygon_id,
          M1Rule::kSpace};
      const M1Verdict verdict =
          m1p::classify_pair_bounded(space, kM1Distance);
      oracle.space_hits += verdict == M1Verdict::kViolation;
      oracle.space_uncertain += verdict == M1Verdict::kUncertain;
    }
  }
  return oracle;
}

void m1_upload_status_or(DeviceBuffer<std::uint32_t> *status,
                         std::uint32_t flag) {
  std::uint32_t current = 0;
  cuda_require(cudaMemcpy(
                   &current, status->get(), sizeof(current),
                   cudaMemcpyDeviceToHost),
               "cudaMemcpy M1 status read");
  current |= flag;
  cuda_require(cudaMemcpy(
                   status->get(), &current, sizeof(current),
                   cudaMemcpyHostToDevice),
               "cudaMemcpy M1 status write");
}

M1PlanResult m1_run_scene(const M1CompactScene &compact,
                          const M1Options &options) {
  const Clock::time_point total_begin = Clock::now();
  M1PlanResult result;
  Clock::time_point begin = Clock::now();
  const M1LoweredScene scene = m1_validate_and_lower(compact, options);
  result.grid = m1_make_grid(scene, options, &result.grid_cells);
  Clock::time_point end = Clock::now();
  result.timing.validate_lower_ms = milliseconds(begin, end);
  result.edges = scene.edge_count;
  result.polygons = scene.polygon_count;
  result.contexts = scene.contexts.size();

  if (options.host_oracle) {
    begin = Clock::now();
    result.oracle = m1_host_oracle(scene, UINT64_C(20000000));
    end = Clock::now();
    result.timing.host_oracle_ms = milliseconds(begin, end);
  }

  begin = Clock::now();
  cuda_require(cudaFree(nullptr), "CUDA M1 context initialization");
  end = Clock::now();
  result.timing.cuda_init_ms = milliseconds(begin, end);

  begin = Clock::now();
  DeviceBuffer<ContextGpu> d_contexts(scene.contexts.size());
  DeviceBuffer<std::uint64_t> d_edge_offsets(scene.edge_offsets.size());
  DeviceBuffer<std::uint64_t> d_polygon_offsets(
      scene.polygon_offsets.size());
  DeviceBuffer<M1CellTemplate> d_cells(scene.cells.size());
  DeviceBuffer<M1EdgeTemplate> d_templates(scene.edges.size());
  DeviceBuffer<M1ExpandedEdge> d_edges(scene.edge_count);
  DeviceBuffer<std::uint32_t> d_counts(result.grid_cells);
  DeviceBuffer<std::uint32_t> d_offsets(result.grid_cells + 1);
  DeviceBuffer<std::uint32_t> d_cursors(result.grid_cells);
  DeviceBuffer<unsigned long long> d_membership_total(1);
  DeviceBuffer<unsigned long long> d_pair_work(1);
  DeviceBuffer<M1DeviceCounters> d_counters(1);
  DeviceBuffer<M1Sample> d_samples(kM1SampleCapacity);
  DeviceBuffer<std::uint32_t> d_sample_count(1);
  DeviceBuffer<std::uint32_t> d_status(1);
  upload(&d_contexts, scene.contexts);
  upload(&d_edge_offsets, scene.edge_offsets);
  upload(&d_polygon_offsets, scene.polygon_offsets);
  upload(&d_cells, scene.cells);
  upload(&d_templates, scene.edges);
  cuda_require(cudaMemset(
                   d_counts.get(), 0,
                   result.grid_cells * sizeof(std::uint32_t)),
               "cudaMemset M1 grid counts");
  cuda_require(cudaMemset(
                   d_membership_total.get(), 0,
                   sizeof(unsigned long long)),
               "cudaMemset M1 membership total");
  cuda_require(cudaMemset(
                   d_pair_work.get(), 0, sizeof(unsigned long long)),
               "cudaMemset M1 pair work");
  cuda_require(cudaMemset(
                   d_counters.get(), 0, sizeof(M1DeviceCounters)),
               "cudaMemset M1 counters");
  cuda_require(cudaMemset(
                   d_sample_count.get(), 0, sizeof(std::uint32_t)),
               "cudaMemset M1 sample count");
  cuda_require(cudaMemset(
                   d_status.get(), 0, sizeof(std::uint32_t)),
               "cudaMemset M1 status");
  cuda_require(cudaDeviceSynchronize(), "M1 upload synchronize");
  end = Clock::now();
  result.timing.alloc_upload_ms = milliseconds(begin, end);

  begin = Clock::now();
  m1_expand_edges_kernel<<<
      static_cast<unsigned int>(scene.contexts.size()), 128>>>(
      d_contexts.get(), d_edge_offsets.get(), d_polygon_offsets.get(),
      d_cells.get(), d_templates.get(),
      static_cast<std::uint32_t>(scene.contexts.size()), d_edges.get(),
      d_counters.get(), d_status.get());
  cuda_require(cudaGetLastError(), "M1 edge expansion launch");
  cuda_require(cudaDeviceSynchronize(), "M1 edge expansion synchronize");
  std::uint32_t host_status = 0;
  cuda_require(cudaMemcpy(
                   &host_status, d_status.get(), sizeof(host_status),
                   cudaMemcpyDeviceToHost),
               "cudaMemcpy M1 expansion status");
  end = Clock::now();
  result.timing.expand_ms = milliseconds(begin, end);

  const unsigned int edge_blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>(65535, (scene.edge_count + 255) / 256));
  unsigned long long membership_total = 0;
  if (!host_status) {
    begin = Clock::now();
    m1_count_grid_kernel<<<edge_blocks, 256>>>(
        d_edges.get(), static_cast<std::uint32_t>(scene.edge_count),
        result.grid, d_counts.get(), d_membership_total.get(),
        d_status.get());
    cuda_require(cudaGetLastError(), "M1 grid count launch");
    cuda_require(cudaDeviceSynchronize(), "M1 grid count synchronize");
    cuda_require(cudaMemcpy(
                     &membership_total, d_membership_total.get(),
                     sizeof(membership_total), cudaMemcpyDeviceToHost),
                 "cudaMemcpy M1 membership total");
    cuda_require(cudaMemcpy(
                     &host_status, d_status.get(), sizeof(host_status),
                     cudaMemcpyDeviceToHost),
                 "cudaMemcpy M1 grid count status");
    end = Clock::now();
    result.timing.grid_count_ms = milliseconds(begin, end);
  }
  result.memberships = membership_total;
  if (!host_status &&
      (membership_total < scene.edge_count ||
      membership_total > options.max_memberships ||
       membership_total > std::numeric_limits<std::uint32_t>::max())) {
    m1_upload_status_or(&d_status, kM1GridCapacityExceeded);
    host_status |= kM1GridCapacityExceeded;
  }

  std::optional<DeviceBuffer<std::uint32_t>> d_members;
  if (!host_status) {
    begin = Clock::now();
    thrust::device_ptr<std::uint32_t> count_begin(d_counts.get());
    thrust::device_ptr<std::uint32_t> offset_begin(d_offsets.get());
    thrust::exclusive_scan(
        count_begin, count_begin + result.grid_cells, offset_begin);
    const std::uint32_t terminal =
        static_cast<std::uint32_t>(membership_total);
    cuda_require(cudaMemcpy(
                     d_offsets.get() + result.grid_cells, &terminal,
                     sizeof(terminal), cudaMemcpyHostToDevice),
                 "cudaMemcpy M1 terminal offset");
    cuda_require(cudaMemcpy(
                     d_cursors.get(), d_offsets.get(),
                     result.grid_cells * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToDevice),
                 "cudaMemcpy M1 offsets to cursors");
    d_members.emplace(membership_total);
    m1_fill_grid_kernel<<<edge_blocks, 256>>>(
        d_edges.get(), static_cast<std::uint32_t>(scene.edge_count),
        result.grid, d_cursors.get(), d_members->get(), membership_total,
        d_status.get());
    cuda_require(cudaGetLastError(), "M1 grid fill launch");
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>(
            65535, (result.grid_cells + 255) / 256));
    m1_validate_grid_kernel<<<grid_blocks, 256>>>(
        d_counts.get(), d_offsets.get(), d_cursors.get(),
        result.grid_cells, d_status.get());
    cuda_require(cudaGetLastError(), "M1 grid validation launch");
    cuda_require(cudaDeviceSynchronize(), "M1 grid build synchronize");
    cuda_require(cudaMemcpy(
                     &host_status, d_status.get(), sizeof(host_status),
                     cudaMemcpyDeviceToHost),
                 "cudaMemcpy M1 grid build status");
    end = Clock::now();
    result.timing.grid_build_ms = milliseconds(begin, end);
  }

  if (!host_status) {
    begin = Clock::now();
    const unsigned int grid_blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>(
            65535, (result.grid_cells + 255) / 256));
    m1_count_pair_work_kernel<<<grid_blocks, 256>>>(
        d_counts.get(), result.grid_cells, d_pair_work.get(),
        d_status.get());
    cuda_require(cudaGetLastError(), "M1 pair-work count launch");
    cuda_require(cudaDeviceSynchronize(), "M1 pair-work count synchronize");
    unsigned long long pair_work = 0;
    cuda_require(cudaMemcpy(
                     &pair_work, d_pair_work.get(), sizeof(pair_work),
                     cudaMemcpyDeviceToHost),
                 "cudaMemcpy M1 pair work");
    cuda_require(cudaMemcpy(
                     &host_status, d_status.get(), sizeof(host_status),
                     cudaMemcpyDeviceToHost),
                 "cudaMemcpy M1 pair-work status");
    result.pair_work = pair_work;
    if (pair_work > options.max_pair_work) {
      m1_upload_status_or(&d_status, kM1PairCapacityExceeded);
      host_status |= kM1PairCapacityExceeded;
    }
    end = Clock::now();
    result.timing.pair_count_ms = milliseconds(begin, end);
  }

  if (!host_status) {
    begin = Clock::now();
    m1_query_pairs_kernel<<<
        static_cast<unsigned int>(result.grid_cells), 128>>>(
        d_edges.get(), result.grid, d_counts.get(), d_offsets.get(),
        d_members->get(), result.grid_cells, d_counters.get(),
        d_samples.get(), d_sample_count.get(), d_status.get());
    cuda_require(cudaGetLastError(), "M1 pair query launch");
    cuda_require(cudaDeviceSynchronize(), "M1 pair query synchronize");
    end = Clock::now();
    result.timing.query_ms = milliseconds(begin, end);
  }

  begin = Clock::now();
  cuda_require(cudaMemcpy(
                   &result.counters, d_counters.get(),
                   sizeof(result.counters), cudaMemcpyDeviceToHost),
               "cudaMemcpy M1 counters");
  cuda_require(cudaMemcpy(
                   &result.sample_count, d_sample_count.get(),
                   sizeof(result.sample_count), cudaMemcpyDeviceToHost),
               "cudaMemcpy M1 sample count");
  cuda_require(cudaMemcpy(
                   &result.device_flags, d_status.get(),
                   sizeof(result.device_flags), cudaMemcpyDeviceToHost),
               "cudaMemcpy M1 final status");
  if (result.sample_count) {
    cuda_require(cudaMemcpy(
                     result.samples.data(), d_samples.get(),
                     sizeof(result.samples), cudaMemcpyDeviceToHost),
                 "cudaMemcpy M1 samples");
  }
  end = Clock::now();
  result.timing.d2h_ms = milliseconds(begin, end);

  if (result.counters.expanded_edges != scene.edge_count ||
      result.counters.width_hits + result.counters.width_uncertain >
          result.counters.width_pairs ||
      result.counters.space_hits + result.counters.space_uncertain >
          result.counters.space_pairs ||
      result.counters.space_pairs != result.counters.unique_edge_pairs) {
    result.device_flags |= kM1ConservationFailure;
  }
  if (options.host_oracle && !result.device_flags &&
      (result.oracle.width_hits != result.counters.width_hits ||
       result.oracle.space_hits != result.counters.space_hits ||
       result.oracle.width_uncertain !=
           result.counters.width_uncertain ||
       result.oracle.space_uncertain !=
           result.counters.space_uncertain)) {
    result.device_flags |= kM1ConservationFailure;
    result.message = "GPU result disagrees with exhaustive host oracle";
  }

  const bool uncertain =
      result.device_flags || result.counters.width_uncertain ||
      result.counters.space_uncertain;
  const bool raw_hits =
      result.counters.width_hits || result.counters.space_hits;
  result.disposition =
      uncertain ? M1Disposition::kUncertain
                : (raw_hits ? M1Disposition::kRawHits
                            : M1Disposition::kComplete);
  result.timing.total_ms = milliseconds(total_begin, Clock::now());
  return result;
}

void m1_print_result(const std::string &label,
                     const M1PlanResult &result) {
  std::cout << std::fixed << std::setprecision(3)
            << "M1_WIDTH_SPACE_GPU_ISLAND"
            << " case=" << label
            << " disposition=" << m1_disposition_name(result.disposition)
            << " certified_empty="
            << (result.disposition == M1Disposition::kComplete ? 1 : 0)
            << " distance_dbu=" << kM1Distance
            << " contexts=" << result.contexts
            << " polygons=" << result.polygons
            << " edges=" << result.edges
            << " expanded_edges=" << result.counters.expanded_edges
            << " grid=" << result.grid.width << "x" << result.grid.height
            << " grid_cells=" << result.grid_cells
            << " memberships=" << result.memberships
            << " pair_work=" << result.pair_work
            << " unique_edge_pairs=" << result.counters.unique_edge_pairs
            << " width_pairs=" << result.counters.width_pairs
            << " space_pairs=" << result.counters.space_pairs
            << " width_hits=" << result.counters.width_hits
            << " space_hits=" << result.counters.space_hits
            << " width_uncertain=" << result.counters.width_uncertain
            << " space_uncertain=" << result.counters.space_uncertain
            << " device_flags=" << result.device_flags;
  if (!result.scene_digest.empty()) {
    std::cout << " scene_sha256=" << result.scene_digest;
  }
  if (!result.message.empty()) {
    std::cout << " message=\"" << result.message << "\"";
  }
  std::cout << "\n";
  std::cout << "TIMING_MS"
            << " case=" << label
            << " packed_load=" << result.timing.packed_load_ms
            << " validate_lower=" << result.timing.validate_lower_ms
            << " host_oracle=" << result.timing.host_oracle_ms
            << " cuda_init=" << result.timing.cuda_init_ms
            << " alloc_upload=" << result.timing.alloc_upload_ms
            << " expand=" << result.timing.expand_ms
            << " grid_count=" << result.timing.grid_count_ms
            << " grid_build=" << result.timing.grid_build_ms
            << " pair_count=" << result.timing.pair_count_ms
            << " query=" << result.timing.query_ms
            << " d2h=" << result.timing.d2h_ms
            << " gpu_plan="
            << result.timing.alloc_upload_ms + result.timing.expand_ms +
                   result.timing.grid_count_ms +
                   result.timing.grid_build_ms +
                   result.timing.pair_count_ms + result.timing.query_ms +
                   result.timing.d2h_ms
            << " total=" << result.timing.total_ms << "\n";
  const std::uint32_t copied =
      std::min(result.sample_count, kM1SampleCapacity);
  for (std::uint32_t index = 0; index < copied; ++index) {
    const M1Sample &sample = result.samples[index];
    std::cout << "M1_SAMPLE"
              << " case=" << label
              << " first_edge=" << sample.first_edge
              << " second_edge=" << sample.second_edge
              << " first_polygon=" << sample.first_polygon
              << " second_polygon=" << sample.second_polygon
              << " rule=" << sample.rule
              << " verdict=" << sample.verdict << "\n";
  }
}

enum class M1Expected {
  kComplete,
  kNonClean,
  kUncertain,
};

struct M1Fixture {
  std::string name;
  M1CompactScene scene;
  M1Expected expected;
  std::uint64_t max_pair_work = kM1DefaultMaxPairWork;
};

std::vector<ContextGpu> m1_transform_array_contexts(
    std::uint32_t copies_per_transform, std::int64_t pitch) {
  std::vector<ContextGpu> contexts;
  for (std::uint32_t transform = 0; transform < 8; ++transform) {
    for (std::uint32_t copy = 0; copy < copies_per_transform; ++copy) {
      contexts.push_back(
          {0, transform,
           static_cast<std::int64_t>(transform) * pitch * 3,
           static_cast<std::int64_t>(copy) * pitch});
    }
  }
  return contexts;
}

std::vector<M1Fixture> m1_make_fixtures() {
  std::vector<M1Fixture> fixtures;
  fixtures.push_back({
      "width_axial_129",
      m1_make_scene(
          "width_axial_129", {m1_rectangle(0, 0, 600, 129)}),
      M1Expected::kNonClean});
  fixtures.push_back({
      "width_axial_130",
      m1_make_scene(
          "width_axial_130", {m1_rectangle(0, 0, 600, 130)}),
      M1Expected::kComplete});
  fixtures.push_back({
      "width_axial_131",
      m1_make_scene(
          "width_axial_131", {m1_rectangle(0, 0, 600, 131)}),
      M1Expected::kComplete});
  fixtures.push_back({
      "space_axial_129",
      m1_make_scene(
          "space_axial_129",
          {m1_rectangle(0, 0, 260, 260),
           m1_rectangle(389, 0, 649, 260)}),
      M1Expected::kNonClean});
  fixtures.push_back({
      "space_axial_130",
      m1_make_scene(
          "space_axial_130",
          {m1_rectangle(0, 0, 260, 260),
           m1_rectangle(390, 0, 650, 260)}),
      M1Expected::kComplete});
  fixtures.push_back({
      "space_axial_131",
      m1_make_scene(
          "space_axial_131",
          {m1_rectangle(0, 0, 260, 260),
           m1_rectangle(391, 0, 651, 260)}),
      M1Expected::kComplete});
  fixtures.push_back({
      "space_diagonal_below",
      m1_make_scene(
          "space_diagonal_below",
          {m1_rectangle(0, 0, 300, 300),
           m1_rectangle(350, 419, 650, 719)}),
      M1Expected::kNonClean});
  fixtures.push_back({
      "space_diagonal_equal",
      m1_make_scene(
          "space_diagonal_equal",
          {m1_rectangle(0, 0, 300, 300),
           m1_rectangle(350, 420, 650, 720)}),
      M1Expected::kComplete});
  fixtures.push_back({
      "space_diagonal_above",
      m1_make_scene(
          "space_diagonal_above",
          {m1_rectangle(0, 0, 300, 300),
           m1_rectangle(350, 421, 650, 721)}),
      M1Expected::kComplete});
  fixtures.push_back({
      "same_polygon_notch_129",
      m1_make_scene("same_polygon_notch_129", {m1_notch(129)}),
      M1Expected::kNonClean});
  fixtures.push_back({
      "same_polygon_notch_130",
      m1_make_scene("same_polygon_notch_130", {m1_notch(130)}),
      M1Expected::kComplete});
  fixtures.push_back({
      "transforms_array_clean",
      m1_make_scene(
          "transforms_array_clean",
          {m1_rectangle(-150, -150, 150, 150)},
          m1_transform_array_contexts(3, 700)),
      M1Expected::kComplete});
  fixtures.push_back({
      "transforms_array_width_hit",
      m1_make_scene(
          "transforms_array_width_hit",
          {m1_rectangle(-300, -64, 300, 65)},
          m1_transform_array_contexts(2, 800)),
      M1Expected::kNonClean});

  M1CompactScene default_untrusted;
  default_untrusted.name = "default_untrusted_scene";
  fixtures.push_back({
      "default_untrusted_scene", std::move(default_untrusted),
      M1Expected::kUncertain});

  M1CompactScene touching = m1_make_scene(
      "touching_raw_polygons",
      {m1_rectangle(0, 0, 260, 260),
       m1_rectangle(260, 0, 520, 260)});
  touching.exact_merged_universe = false;
  fixtures.push_back({
      "touching_raw_polygons", std::move(touching),
      M1Expected::kUncertain});

  M1CompactScene coincident = m1_make_scene(
      "coincident_raw_polygons",
      {m1_rectangle(0, 0, 260, 260),
       m1_rectangle(0, 0, 260, 260)});
  coincident.exact_merged_universe = false;
  fixtures.push_back({
      "coincident_raw_polygons", std::move(coincident),
      M1Expected::kUncertain});

  M1CompactScene diagonal = m1_make_scene(
      "unsupported_diagonal",
      {{{0, 0}, {0, 300}, {300, 300}, {400, 0}}});
  fixtures.push_back({
      "unsupported_diagonal", std::move(diagonal),
      M1Expected::kUncertain});

  M1CompactScene transform = m1_make_scene(
      "unsupported_transform", {m1_rectangle(0, 0, 300, 300)});
  transform.contexts[0].transform = 8;
  fixtures.push_back({
      "unsupported_transform", std::move(transform),
      M1Expected::kUncertain});

  fixtures.push_back({
      "pair_capacity",
      m1_make_scene(
          "pair_capacity", {m1_rectangle(0, 0, 600, 129)}),
      M1Expected::kUncertain, 1});
  return fixtures;
}

bool m1_expected_matches(M1Expected expected,
                         M1Disposition observed) {
  if (expected == M1Expected::kComplete) {
    return observed == M1Disposition::kComplete;
  }
  if (expected == M1Expected::kUncertain) {
    return observed == M1Disposition::kUncertain;
  }
  return observed == M1Disposition::kRawHits;
}

int m1_run_self_test(const M1Options &base_options) {
  cuda_require(cudaFree(nullptr), "CUDA M1 self-test initialization");
  std::size_t passed = 0;
  const std::vector<M1Fixture> fixtures = m1_make_fixtures();
  for (const M1Fixture &fixture : fixtures) {
    M1Disposition observed = M1Disposition::kUncertain;
    std::string error;
    try {
      M1Options options = base_options;
      options.host_oracle = true;
      options.max_pair_work = fixture.max_pair_work;
      const M1PlanResult result = m1_run_scene(fixture.scene, options);
      observed = result.disposition;
      m1_print_result(fixture.name, result);
    } catch (const std::exception &exception) {
      error = exception.what();
      std::cout << "M1_WIDTH_SPACE_GPU_ISLAND"
                << " case=" << fixture.name
                << " disposition=UNCERTAIN certified_empty=0"
                << " error=\"" << error << "\"\n";
    }
    const bool match = m1_expected_matches(fixture.expected, observed);
    std::cout << "M1_SELF_TEST"
              << " case=" << fixture.name
              << " result=" << (match ? "PASS" : "FAIL")
              << " observed=" << m1_disposition_name(observed) << "\n";
    passed += match;
  }
  std::cout << "M1_SELF_TEST_SUMMARY"
            << " passed=" << passed
            << " total=" << fixtures.size()
            << " result=" << (passed == fixtures.size() ? "PASS" : "FAIL")
            << "\n";
  return passed == fixtures.size() ? 0 : 1;
}

M1CompactScene m1_make_benchmark_scene(std::uint64_t contexts) {
  if (!contexts || contexts > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("benchmark context count is outside uint32 domain");
  }
  const std::uint64_t columns = static_cast<std::uint64_t>(
      std::ceil(std::sqrt(static_cast<long double>(contexts))));
  std::vector<ContextGpu> expanded;
  expanded.reserve(contexts);
  constexpr std::int64_t pitch = 640;
  for (std::uint64_t id = 0; id < contexts; ++id) {
    const std::uint64_t column = id % columns;
    const std::uint64_t row = id / columns;
    const __int128 tx = static_cast<__int128>(column) * pitch;
    const __int128 ty = static_cast<__int128>(row) * pitch;
    expanded.push_back({
        0, static_cast<std::uint32_t>(id % 8),
        narrow_i64(tx, "benchmark context x"),
        narrow_i64(ty, "benchmark context y")});
  }
  return m1_make_scene(
      "benchmark",
      {m1_rectangle(-150, -150, 150, 150)}, std::move(expanded));
}

std::uint64_t m1_parse_u64(const std::string &text,
                           const char *name) {
  if (text.empty() || text[0] == '-') {
    throw SceneError(std::string(name) + " must be a positive integer");
  }
  std::size_t consumed = 0;
  const std::uint64_t value = std::stoull(text, &consumed);
  if (!value || consumed != text.size()) {
    throw SceneError(std::string(name) + " must be a positive integer");
  }
  return value;
}

M1Options m1_parse_options(int argc, char **argv) {
  M1Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--self-test") {
      options.self_test = true;
    } else if (argument == "--no-host-oracle") {
      options.host_oracle = false;
    } else if (argument == "--trust-merged-layer0") {
      options.trust_merged_layer0 = true;
    } else if (argument.rfind("--packed-scene=", 0) == 0) {
      options.packed_scene =
          argument.substr(std::string("--packed-scene=").size());
      if (options.packed_scene.empty()) {
        throw SceneError("--packed-scene requires a path");
      }
    } else if (argument.rfind("--host-scene=", 0) == 0) {
      options.host_scene =
          argument.substr(std::string("--host-scene=").size());
      if (options.host_scene.empty()) {
        throw SceneError("--host-scene requires a path");
      }
    } else if (argument.rfind("--expect-scene-sha256=", 0) == 0) {
      options.expected_scene_sha256 =
          argument.substr(std::string("--expect-scene-sha256=").size());
      if (options.expected_scene_sha256.size() != 64 ||
          !std::all_of(
              options.expected_scene_sha256.begin(),
              options.expected_scene_sha256.end(),
              [](unsigned char character) {
                return std::isxdigit(character) != 0;
              })) {
        throw SceneError(
            "--expect-scene-sha256 requires exactly 64 hex digits");
      }
      std::transform(
          options.expected_scene_sha256.begin(),
          options.expected_scene_sha256.end(),
          options.expected_scene_sha256.begin(),
          [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
          });
    } else {
      const auto parse = [&](const char *prefix,
                             std::uint64_t *destination) {
        const std::string marker(prefix);
        if (argument.rfind(marker, 0) != 0) {
          return false;
        }
        *destination =
            m1_parse_u64(argument.substr(marker.size()), prefix);
        return true;
      };
      if (!parse("--benchmark-contexts=", &options.benchmark_contexts) &&
          !parse("--repetitions=", &options.repetitions) &&
          !parse("--max-contexts=", &options.max_contexts) &&
          !parse("--max-grid-cells=", &options.max_grid_cells) &&
          !parse("--max-memberships=", &options.max_memberships) &&
          !parse("--max-pair-work=", &options.max_pair_work) &&
          !parse("--max-edges=", &options.max_edges) &&
          !parse("--max-polygons=", &options.max_polygons)) {
        throw SceneError("unknown option: " + argument);
      }
    }
  }
  const unsigned int modes =
      (options.self_test ? 1u : 0u) +
      (options.benchmark_contexts ? 1u : 0u) +
      (!options.packed_scene.empty() ? 1u : 0u) +
      (!options.host_scene.empty() ? 1u : 0u);
  if (modes != 1) {
    throw SceneError(
        "usage: m1_width_space_scene_island "
        "(--self-test | --benchmark-contexts=N | "
        "--packed-scene=SCENE.kact --expect-scene-sha256=HEX "
        "--trust-merged-layer0 | "
        "--host-scene=SCENE.km1ws --expect-scene-sha256=HEX) "
        "[--repetitions=N] [--no-host-oracle] [capacity options]");
  }
  if (!options.packed_scene.empty() &&
      (options.expected_scene_sha256.empty() ||
       !options.trust_merged_layer0)) {
    throw SceneError(
        "packed mode requires an expected digest and explicit "
        "--trust-merged-layer0 qualification");
  }
  if (!options.host_scene.empty() &&
      options.expected_scene_sha256.empty()) {
    throw SceneError("host-scene mode requires an expected digest");
  }
  if (!options.host_scene.empty() && options.repetitions != 1) {
    throw SceneError("--repetitions is not supported in host-scene mode");
  }
  return options;
}

int m1_run_benchmark(const M1Options &options) {
  const M1CompactScene scene =
      m1_make_benchmark_scene(options.benchmark_contexts);
  M1Options run_options = options;
  // Exhaustive O(E^2) checking is deliberately limited to synthetic gates.
  if (options.benchmark_contexts > 512) {
    run_options.host_oracle = false;
  }
  for (std::uint64_t repetition = 0;
       repetition < options.repetitions; ++repetition) {
    const M1PlanResult result = m1_run_scene(scene, run_options);
    m1_print_result(
        "benchmark_" + std::to_string(repetition + 1), result);
    if (result.disposition != M1Disposition::kComplete) {
      return result.disposition == M1Disposition::kUncertain ? 2 : 3;
    }
  }
  return 0;
}

int m1_run_packed_scene(const M1Options &options) {
  const Clock::time_point begin = Clock::now();
  const LoadedScene packed = load_and_validate(options.packed_scene);
  const std::string digest =
      hex_digest(packed.header.scene_sha256, 32);
  if (digest != options.expected_scene_sha256) {
    throw SceneError("packed scene SHA-256 does not match expectation");
  }
  const M1CompactScene compact =
      m1_scene_from_kact_layer0(packed, options);
  const double packed_load_ms = milliseconds(begin, Clock::now());

  M1Options run_options = options;
  if (compact.contexts.size() > 512) {
    run_options.host_oracle = false;
  }
  const M1PlanResult base = m1_run_scene(compact, run_options);
  M1PlanResult result = base;
  result.scene_digest = digest;
  result.timing.packed_load_ms = packed_load_ms;
  result.timing.total_ms += packed_load_ms;
  m1_print_result("packed_merged_m1", result);
  return result.disposition == M1Disposition::kComplete
             ? 0
             : (result.disposition == M1Disposition::kUncertain ? 2 : 3);
}

int m1_run_host_scene(const M1Options &options) {
  const Clock::time_point begin = Clock::now();
  M1LoadedHostScene loaded =
      m1_load_host_scene(options.host_scene, options);
  const double packed_load_ms = milliseconds(begin, Clock::now());

  M1Options run_options = options;
  // Production scenes are validated structurally on the host, while the
  // exact all-pairs oracle remains restricted to the small synthetic gates.
  if (loaded.compact.contexts.size() > 512) {
    run_options.host_oracle = false;
  }
  const M1PlanResult base =
      m1_run_scene(loaded.compact, run_options);
  M1PlanResult result = base;
  result.scene_digest = loaded.digest;
  result.timing.packed_load_ms = packed_load_ms;
  result.timing.total_ms += packed_load_ms;
  m1_print_result("host_merged_m1", result);
  return result.disposition == M1Disposition::kComplete
             ? 0
             : (result.disposition == M1Disposition::kUncertain ? 2 : 3);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const M1Options options = m1_parse_options(argc, argv);
    if (options.self_test) {
      return m1_run_self_test(options);
    }
    if (!options.packed_scene.empty()) {
      return m1_run_packed_scene(options);
    }
    if (!options.host_scene.empty()) {
      return m1_run_host_scene(options);
    }
    return m1_run_benchmark(options);
  } catch (const std::exception &exception) {
    std::cerr << "M1_WIDTH_SPACE_GPU_ISLAND"
              << " disposition=UNCERTAIN certified_empty=0"
              << " error=\"" << exception.what() << "\"\n";
    return 2;
  }
}
