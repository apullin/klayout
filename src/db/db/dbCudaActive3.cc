/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaActive3.h"

#include "dbArray.h"
#include "dbCell.h"
#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialBackend.h"
#include "dbDeepShapeStore.h"
#include "dbLayout.h"
#include "dbPolygon.h"
#include "dbShape.h"
#include "dbShapes.h"
#include "tlLog.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace db
{

namespace
{

const uint64_t default_max_contexts = UINT64_C (4000000);
const uint64_t default_max_grid_cells = UINT64_C (16000000);
const uint64_t default_max_memberships = UINT64_C (100000000);
const uint64_t default_max_pair_work = UINT64_C (2000000000000);
const unsigned maximum_hierarchy_depth = 1024;
const int64_t accepted_coordinate_magnitude = INT64_C (1000000000000);
const int64_t qualified_distance = 110;
const int64_t qualified_grid_cell = 2000;
const uint32_t qualified_dbu_per_micron = 2000;

class Active3Decline
  : public std::runtime_error
{
public:
  explicit Active3Decline (const std::string &message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

bool env_enabled (const char *name)
{
  const char *value = std::getenv (name);
  return value && *value && std::strcmp (value, "0") != 0 &&
         std::strcmp (value, "false") != 0 &&
         std::strcmp (value, "off") != 0;
}

uint64_t env_u64 (const char *name, uint64_t default_value)
{
  const char *value = std::getenv (name);
  if (! value || ! *value || *value == '-') {
    return default_value;
  }
  const int saved_errno = errno;
  errno = 0;
  char *end = 0;
  const unsigned long long parsed = std::strtoull (value, &end, 0);
  const bool valid = errno == 0 && end != value && *end == 0;
  errno = saved_errno;
  return valid ? uint64_t (parsed) : default_value;
}

bool checked_add_u64 (uint64_t a, uint64_t b, uint64_t &result)
{
  if (b > std::numeric_limits<uint64_t>::max () - a) {
    return false;
  }
  result = a + b;
  return true;
}

bool checked_multiply_u64 (uint64_t a, uint64_t b, uint64_t &result)
{
  if (a && b > std::numeric_limits<uint64_t>::max () / a) {
    return false;
  }
  result = a * b;
  return true;
}

int64_t narrow_i64 (__int128 value, const char *what)
{
  if (value < std::numeric_limits<int64_t>::min () ||
      value > std::numeric_limits<int64_t>::max ()) {
    throw Active3Decline (std::string (what) + " overflows signed int64");
  }
  const int64_t result = int64_t (value);
  if (result < -accepted_coordinate_magnitude ||
      result > accepted_coordinate_magnitude) {
    throw Active3Decline (
      std::string (what) + " exceeds the qualified coordinate domain");
  }
  return result;
}

struct Matrix
{
  int xx, xy, yx, yy;
};

const Matrix transforms [8] = {
  { 1, 0, 0, 1 }, { 0, -1, 1, 0 }, { -1, 0, 0, -1 },
  { 0, 1, -1, 0 }, { 1, 0, 0, -1 }, { 0, 1, 1, 0 },
  { -1, 0, 0, 1 }, { 0, -1, -1, 0 }
};

std::pair<__int128, __int128>
transform_128 (uint32_t code, __int128 x, __int128 y)
{
  if (code >= 8) {
    throw Active3Decline ("invalid orthogonal transform code");
  }
  const Matrix &matrix = transforms [code];
  return std::make_pair (
    __int128 (matrix.xx) * x + __int128 (matrix.xy) * y,
    __int128 (matrix.yx) * x + __int128 (matrix.yy) * y);
}

uint32_t compose_transform (uint32_t outer, uint32_t inner)
{
  if (outer >= 8 || inner >= 8) {
    throw Active3Decline ("invalid transform during hierarchy expansion");
  }
  const Matrix &a = transforms [outer];
  const Matrix &b = transforms [inner];
  const Matrix product = {
    a.xx * b.xx + a.xy * b.yx,
    a.xx * b.xy + a.xy * b.yy,
    a.yx * b.xx + a.yy * b.yx,
    a.yx * b.xy + a.yy * b.yy
  };
  for (uint32_t code = 0; code < 8; ++code) {
    const Matrix &candidate = transforms [code];
    if (candidate.xx == product.xx && candidate.xy == product.xy &&
        candidate.yx == product.yx && candidate.yy == product.yy) {
      return code;
    }
  }
  throw Active3Decline ("orthogonal transform composition escaped its group");
}

struct InstanceTemplate
{
  uint32_t child_cell;
  uint32_t columns;
  uint32_t rows;
  uint32_t transform_code;
  int64_t dx, dy, ax, ay, bx, by;
  uint64_t occurrences;
};

struct CellTemplate
{
  std::vector<InstanceTemplate> instances;
};

struct LiveScene
{
  std::vector<klayout_cuda_spatial_active3_context_v1> contexts;
  std::vector<uint32_t> well_contexts;
  std::vector<uint64_t> well_offsets;
  std::vector<uint32_t> active_contexts;
  std::vector<klayout_cuda_spatial_active3_cell_v1> cells;
  std::vector<klayout_cuda_spatial_active3_edge_v1> edges;
  uint64_t flat_well_edges;
  uint64_t flat_active_edges;
  int64_t well_left, well_bottom, well_right, well_top;

  LiveScene ()
    : flat_well_edges (0), flat_active_edges (0),
      well_left (0), well_bottom (0), well_right (0), well_top (0)
  {
    //  nothing yet
  }
};

bool positive_collinear_overlap (
  const klayout_cuda_spatial_active3_edge_v1 &a,
  const klayout_cuda_spatial_active3_edge_v1 &b)
{
  if (a.y1 == a.y2 && b.y1 == b.y2 && a.y1 == b.y1) {
    return std::min (std::max (a.x1, a.x2), std::max (b.x1, b.x2)) >
           std::max (std::min (a.x1, a.x2), std::min (b.x1, b.x2));
  }
  if (a.x1 == a.x2 && b.x1 == b.x2 && a.x1 == b.x1) {
    return std::min (std::max (a.y1, a.y2), std::max (b.y1, b.y2)) >
           std::max (std::min (a.y1, a.y2), std::min (b.y1, b.y2));
  }
  return false;
}

bool manhattan_segments_intersect (
  const klayout_cuda_spatial_active3_edge_v1 &a,
  const klayout_cuda_spatial_active3_edge_v1 &b)
{
  if (a.y1 == a.y2 && b.y1 == b.y2) {
    return a.y1 == b.y1 &&
           std::max (std::min (a.x1, a.x2), std::min (b.x1, b.x2)) <=
           std::min (std::max (a.x1, a.x2), std::max (b.x1, b.x2));
  }
  if (a.x1 == a.x2 && b.x1 == b.x2) {
    return a.x1 == b.x1 &&
           std::max (std::min (a.y1, a.y2), std::min (b.y1, b.y2)) <=
           std::min (std::max (a.y1, a.y2), std::max (b.y1, b.y2));
  }
  const klayout_cuda_spatial_active3_edge_v1 &horizontal =
    a.y1 == a.y2 ? a : b;
  const klayout_cuda_spatial_active3_edge_v1 &vertical =
    a.y1 == a.y2 ? b : a;
  return
    std::min (horizontal.x1, horizontal.x2) <= vertical.x1 &&
    vertical.x1 <= std::max (horizontal.x1, horizontal.x2) &&
    std::min (vertical.y1, vertical.y2) <= horizontal.y1 &&
    horizontal.y1 <= std::max (vertical.y1, vertical.y2);
}

void append_polygon (
  const db::Shape &shape,
  std::vector<klayout_cuda_spatial_active3_edge_v1> &destination)
{
  if (shape.prop_id () != 0) {
    throw Active3Decline ("operand polygon has properties");
  }
  if (! shape.is_box () && ! shape.is_polygon ()) {
    throw Active3Decline ("operand layer contains a non-polygon shape");
  }

  db::Polygon polygon;
  if (! shape.polygon (polygon) || polygon.holes () != 0) {
    throw Active3Decline ("operand polygon is malformed or has holes");
  }

  std::vector<klayout_cuda_spatial_active3_edge_v1> contour;
  std::set<std::pair<int64_t, int64_t> > vertices;
  __int128 twice_area = 0;
  for (db::Polygon::polygon_edge_iterator edge = polygon.begin_edge ();
       ! edge.at_end (); ++edge) {
    const int64_t x1 = narrow_i64 ((*edge).p1 ().x (), "polygon x1");
    const int64_t y1 = narrow_i64 ((*edge).p1 ().y (), "polygon y1");
    const int64_t x2 = narrow_i64 ((*edge).p2 ().x (), "polygon x2");
    const int64_t y2 = narrow_i64 ((*edge).p2 ().y (), "polygon y2");
    if ((x1 == x2 && y1 == y2) || ! (x1 == x2 || y1 == y2)) {
      throw Active3Decline (
        "operand polygon has a degenerate or non-Manhattan edge");
    }
    if (! vertices.insert (std::make_pair (x1, y1)).second) {
      throw Active3Decline ("operand polygon repeats a contour vertex");
    }
    contour.push_back (
      klayout_cuda_spatial_active3_edge_v1 { x1, y1, x2, y2 });
    twice_area += __int128 (x1) * y2 - __int128 (x2) * y1;
  }
  if (contour.size () < 4 || twice_area >= 0) {
    throw Active3Decline (
      "operand polygon is too small or not clockwise");
  }
  for (size_t i = 0; i < contour.size (); ++i) {
    const size_t following = (i + 1) % contour.size ();
    if (contour [i].x2 != contour [following].x1 ||
        contour [i].y2 != contour [following].y1) {
      throw Active3Decline ("operand polygon contour is open");
    }
    for (size_t j = i + 1; j < contour.size (); ++j) {
      if (! manhattan_segments_intersect (contour [i], contour [j])) {
        continue;
      }
      const bool adjacent =
        j == i + 1 || (i == 0 && j + 1 == contour.size ());
      if (! adjacent ||
          positive_collinear_overlap (contour [i], contour [j])) {
        throw Active3Decline ("operand polygon contour self-intersects");
      }
    }
  }
  destination.insert (destination.end (), contour.begin (), contour.end ());
}

void append_cell_layer (
  const db::Cell &cell, unsigned int layer,
  std::vector<klayout_cuda_spatial_active3_edge_v1> &edges,
  uint64_t &begin, uint32_t &count)
{
  begin = edges.size ();
  const db::Shapes &shapes = cell.shapes (layer);
  for (db::Shapes::shape_iterator shape =
         shapes.begin (db::ShapeIterator::All);
       ! shape.at_end (); ++shape) {
    append_polygon (*shape, edges);
  }
  const uint64_t edge_count = uint64_t (edges.size ()) - begin;
  if (edge_count > std::numeric_limits<uint32_t>::max ()) {
    throw Active3Decline ("per-cell operand edge count exceeds uint32");
  }
  count = uint32_t (edge_count);
}

InstanceTemplate make_instance (
  const db::Instance &instance,
  const std::map<db::cell_index_type, uint32_t> &dense_cells)
{
  if (instance.prop_id () != 0 || instance.is_complex ()) {
    throw Active3Decline (
      "hierarchy has an instance property or complex transform");
  }

  const std::map<db::cell_index_type, uint32_t>::const_iterator child =
    dense_cells.find (instance.cell_index ());
  if (child == dense_cells.end ()) {
    throw Active3Decline ("hierarchy instance targets an unreachable cell");
  }

  const db::CellInstArray &array = instance.cell_inst ();
  const db::ArrayBase *delegate = array.delegate ();
  db::Vector a, b;
  unsigned long na = 1, nb = 1;
  const bool regular_delegate =
    delegate != 0 &&
    (typeid (*delegate) == typeid (db::regular_array<db::Coord>) ||
     typeid (*delegate) == typeid (db::regular_complex_array<db::Coord>));
  if (delegate != 0 &&
      (! regular_delegate || ! array.is_regular_array (a, b, na, nb))) {
    throw Active3Decline ("hierarchy has an irregular instance array");
  }
  if (delegate == 0) {
    a = db::Vector ();
    b = db::Vector ();
    na = nb = 1;
  }
  if (na == 0 || nb == 0 ||
      na > std::numeric_limits<uint32_t>::max () ||
      nb > std::numeric_limits<uint32_t>::max ()) {
    throw Active3Decline ("hierarchy has an invalid array dimension");
  }

  const db::Trans &trans = instance.front ();
  const int transform_code = trans.rot ();
  if (transform_code < 0 || transform_code >= 8) {
    throw Active3Decline ("hierarchy has an invalid orthogonal transform");
  }

  InstanceTemplate result;
  result.child_cell = child->second;
  result.columns = uint32_t (na);
  result.rows = uint32_t (nb);
  result.transform_code = uint32_t (transform_code);
  result.dx = narrow_i64 (trans.disp ().x (), "instance dx");
  result.dy = narrow_i64 (trans.disp ().y (), "instance dy");
  result.ax = result.columns == 1 ? 0 : narrow_i64 (a.x (), "array ax");
  result.ay = result.columns == 1 ? 0 : narrow_i64 (a.y (), "array ay");
  result.bx = result.rows == 1 ? 0 : narrow_i64 (b.x (), "array bx");
  result.by = result.rows == 1 ? 0 : narrow_i64 (b.y (), "array by");
  if ((result.columns > 1 && result.ax == 0 && result.ay == 0) ||
      (result.rows > 1 && result.bx == 0 && result.by == 0) ||
      ! checked_multiply_u64 (
          result.columns, result.rows, result.occurrences)) {
    throw Active3Decline ("hierarchy has a malformed regular array");
  }

  const __int128 ax = __int128 (result.columns - 1) * result.ax;
  const __int128 ay = __int128 (result.columns - 1) * result.ay;
  const __int128 bx = __int128 (result.rows - 1) * result.bx;
  const __int128 by = __int128 (result.rows - 1) * result.by;
  narrow_i64 (__int128 (result.dx) + ax + bx, "last array origin x");
  narrow_i64 (__int128 (result.dy) + ay + by, "last array origin y");
  return result;
}

uint64_t subtree_context_count (
  uint32_t cell, const std::vector<CellTemplate> &cells,
  std::vector<uint8_t> &state, std::vector<uint64_t> &memo,
  uint64_t maximum, unsigned depth)
{
  if (depth > maximum_hierarchy_depth) {
    throw Active3Decline (
      "hierarchy exceeds the qualified recursion depth");
  }
  if (state [cell] == 1) {
    throw Active3Decline ("hierarchy contains a cycle");
  }
  if (state [cell] == 2) {
    return memo [cell];
  }
  state [cell] = 1;
  uint64_t total = 1;
  for (std::vector<InstanceTemplate>::const_iterator instance =
         cells [cell].instances.begin ();
       instance != cells [cell].instances.end (); ++instance) {
    const uint64_t child = subtree_context_count (
      instance->child_cell, cells, state, memo, maximum, depth + 1);
    uint64_t contribution = 0;
    if (! checked_multiply_u64 (
          instance->occurrences, child, contribution) ||
        ! checked_add_u64 (total, contribution, total) ||
        total > maximum) {
      throw Active3Decline (
        "expanded hierarchy exceeds the configured context capacity");
    }
  }
  state [cell] = 2;
  memo [cell] = total;
  return total;
}

void expand_contexts (
  uint32_t root, const std::vector<CellTemplate> &templates,
  uint64_t max_contexts,
  std::vector<klayout_cuda_spatial_active3_context_v1> &contexts)
{
  std::vector<uint8_t> state (templates.size (), 0);
  std::vector<uint64_t> memo (templates.size (), 0);
  const uint64_t expected = subtree_context_count (
    root, templates, state, memo, max_contexts, 0);
  if (expected > std::numeric_limits<uint32_t>::max ()) {
    throw Active3Decline ("expanded hierarchy exceeds uint32 context IDs");
  }
  contexts.reserve (size_t (expected));
  contexts.push_back (
    klayout_cuda_spatial_active3_context_v1 { 0, 0, root, 0 });

  for (size_t parent_id = 0; parent_id < contexts.size (); ++parent_id) {
    const klayout_cuda_spatial_active3_context_v1 parent =
      contexts [parent_id];
    const std::vector<InstanceTemplate> &instances =
      templates [parent.cell_id].instances;
    for (std::vector<InstanceTemplate>::const_iterator instance =
           instances.begin (); instance != instances.end (); ++instance) {
      const uint32_t transform = compose_transform (
        parent.transform_code, instance->transform_code);
      for (uint32_t column = 0; column < instance->columns; ++column) {
        for (uint32_t row = 0; row < instance->rows; ++row) {
          const __int128 local_x =
            __int128 (instance->dx) + __int128 (column) * instance->ax +
            __int128 (row) * instance->bx;
          const __int128 local_y =
            __int128 (instance->dy) + __int128 (column) * instance->ay +
            __int128 (row) * instance->by;
          const std::pair<__int128, __int128> shifted =
            transform_128 (parent.transform_code, local_x, local_y);
          contexts.push_back (
            klayout_cuda_spatial_active3_context_v1 {
              narrow_i64 (
                shifted.first + parent.tx, "world context translation x"),
              narrow_i64 (
                shifted.second + parent.ty, "world context translation y"),
              instance->child_cell, transform
            });
        }
      }
    }
  }
  if (contexts.size () != expected) {
    throw Active3Decline (
      "expanded hierarchy disagrees with the checked context census");
  }
}

std::pair<int64_t, int64_t> transform_point (
  const klayout_cuda_spatial_active3_context_v1 &context,
  int64_t x, int64_t y)
{
  const std::pair<__int128, __int128> point =
    transform_128 (context.transform_code, x, y);
  return std::make_pair (
    narrow_i64 (point.first + context.tx, "world edge x"),
    narrow_i64 (point.second + context.ty, "world edge y"));
}

void derive_context_lists_and_well_box (LiveScene &scene)
{
  bool have_well_box = false;
  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const klayout_cuda_spatial_active3_context_v1 &context =
      scene.contexts [context_id];
    const klayout_cuda_spatial_active3_cell_v1 &cell =
      scene.cells [context.cell_id];
    if (cell.well_edge_count) {
      scene.well_contexts.push_back (uint32_t (context_id));
      scene.well_offsets.push_back (scene.flat_well_edges);
      if (! checked_add_u64 (
            scene.flat_well_edges, cell.well_edge_count,
            scene.flat_well_edges)) {
        throw Active3Decline ("flat WELL edge count overflow");
      }
      for (uint32_t local = 0; local < cell.well_edge_count; ++local) {
        const klayout_cuda_spatial_active3_edge_v1 &edge =
          scene.edges [cell.well_edge_begin + local];
        const std::pair<int64_t, int64_t> first =
          transform_point (context, edge.x1, edge.y1);
        const std::pair<int64_t, int64_t> second =
          transform_point (context, edge.x2, edge.y2);
        const int64_t left = std::min (first.first, second.first);
        const int64_t right = std::max (first.first, second.first);
        const int64_t bottom = std::min (first.second, second.second);
        const int64_t top = std::max (first.second, second.second);
        if (! have_well_box) {
          scene.well_left = left;
          scene.well_bottom = bottom;
          scene.well_right = right;
          scene.well_top = top;
          have_well_box = true;
        } else {
          scene.well_left = std::min (scene.well_left, left);
          scene.well_bottom = std::min (scene.well_bottom, bottom);
          scene.well_right = std::max (scene.well_right, right);
          scene.well_top = std::max (scene.well_top, top);
        }
      }
    }
    if (cell.active_edge_count) {
      scene.active_contexts.push_back (uint32_t (context_id));
      if (! checked_add_u64 (
            scene.flat_active_edges, cell.active_edge_count,
            scene.flat_active_edges)) {
        throw Active3Decline ("flat ACTIVE edge count overflow");
      }
    }
  }
  if (! have_well_box || ! scene.flat_well_edges ||
      ! scene.flat_active_edges ||
      scene.flat_well_edges > std::numeric_limits<uint32_t>::max ()) {
    throw Active3Decline (
      "qualified scene has an empty operand or too many WELL edges");
  }
}

LiveScene serialize_live_scene (
  const db::DeepLayer &well, const db::DeepLayer &active,
  uint64_t max_contexts)
{
  const db::Layout &layout = well.layout ();
  const db::cell_index_type top = well.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  well.initial_cell ().collect_called_cells (reachable);
  if (reachable.empty () ||
      reachable.size () > std::numeric_limits<uint32_t>::max ()) {
    throw Active3Decline ("reachable hierarchy has an invalid cell count");
  }

  std::map<db::cell_index_type, uint32_t> dense_cells;
  uint32_t dense = 0;
  for (std::set<db::cell_index_type>::const_iterator cell =
         reachable.begin (); cell != reachable.end (); ++cell, ++dense) {
    dense_cells.insert (std::make_pair (*cell, dense));
  }
  const std::map<db::cell_index_type, uint32_t>::const_iterator root =
    dense_cells.find (top);
  if (root == dense_cells.end ()) {
    throw Active3Decline ("initial cell is absent from the hierarchy census");
  }

  LiveScene scene;
  scene.cells.resize (reachable.size ());
  std::vector<CellTemplate> templates (reachable.size ());
  for (std::set<db::cell_index_type>::const_iterator source =
         reachable.begin (); source != reachable.end (); ++source) {
    const uint32_t cell_id = dense_cells.find (*source)->second;
    const db::Cell &cell = layout.cell (*source);
    for (db::Cell::const_iterator instance = cell.begin ();
         ! instance.at_end (); ++instance) {
      templates [cell_id].instances.push_back (
        make_instance (*instance, dense_cells));
    }

    klayout_cuda_spatial_active3_cell_v1 record;
    std::memset (&record, 0, sizeof (record));
    append_cell_layer (
      cell, well.layer (), scene.edges,
      record.well_edge_begin, record.well_edge_count);
    append_cell_layer (
      cell, active.layer (), scene.edges,
      record.active_edge_begin, record.active_edge_count);
    scene.cells [cell_id] = record;
  }

  expand_contexts (
    root->second, templates, max_contexts, scene.contexts);
  derive_context_lists_and_well_box (scene);
  return scene;
}

bool eligible (
  db::edge_relation_type relation, bool different_polygons,
  db::Coord distance, const db::RegionCheckOptions &options,
  const db::DeepLayer &well, const db::DeepLayer &active)
{
  return
    relation == db::OverlapRelation &&
    different_polygons &&
    distance == qualified_distance &&
    options.metrics == db::Euclidian &&
    options.ignore_angle == 90.0 &&
    ! options.whole_edges &&
    options.min_projection == 0 &&
    options.max_projection ==
      std::numeric_limits<db::RegionCheckOptions::distance_type>::max () &&
    options.shielded &&
    options.opposite_filter == db::NoOppositeFilter &&
    options.rect_filter == db::NoRectFilter &&
    ! options.negative &&
    options.prop_constraint == db::IgnoreProperties &&
    options.zd_mode == db::IncludeZeroDistanceWhenTouching &&
    well.store () == active.store () &&
    &well.layout () == &active.layout () &&
    well.layout_index () == active.layout_index () &&
    well.initial_cell ().cell_index () ==
      active.initial_cell ().cell_index () &&
    well.breakout_cells () == 0 &&
    active.breakout_cells () == 0 &&
    well.layer () != active.layer () &&
    well.layout ().dbu () == 0.0005;
}

} // anonymous namespace

bool cuda_active3_try_empty (
  db::edge_relation_type relation, bool different_polygons,
  db::Coord distance, const db::RegionCheckOptions &options,
  const db::DeepLayer &merged_well, const db::DeepLayer &raw_active)
{
  const bool telemetry = env_enabled ("KLAYOUT_CUDA_ACTIVE3_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    if (! db::cuda_spatial_active3_requested () ||
        ! eligible (
          relation, different_polygons, distance, options,
          merged_well, raw_active)) {
      return false;
    }

    const uint64_t max_contexts = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_MAX_CONTEXTS", default_max_contexts);
    const uint64_t max_grid_cells = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_MAX_GRID_CELLS", default_max_grid_cells);
    const uint64_t max_memberships = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_MAX_MEMBERSHIPS", default_max_memberships);
    const uint64_t max_pair_work = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_MAX_PAIR_WORK", default_max_pair_work);
    if (! max_contexts || ! max_grid_cells || ! max_memberships ||
        ! max_pair_work) {
      throw Active3Decline ("an ACTIVE.3 capacity is zero");
    }

    LiveScene scene = serialize_live_scene (
      merged_well, raw_active, max_contexts);
    uint64_t pair_work = 0;
    if (! checked_multiply_u64 (
          scene.flat_well_edges, scene.flat_active_edges, pair_work) ||
        pair_work > max_pair_work) {
      throw Active3Decline (
        "WELL/ACTIVE pair-work ceiling exceeds configured capacity");
    }

    klayout_cuda_spatial_active3_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode = KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY;
    request.option_flags = KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_OPTIONS;
    request.dbu_per_micron = qualified_dbu_per_micron;
    request.distance = qualified_distance;
    request.grid_cell_size = qualified_grid_cell;
    request.contexts = scene.contexts.data ();
    request.context_count = scene.contexts.size ();
    request.well_contexts = scene.well_contexts.data ();
    request.well_context_count = scene.well_contexts.size ();
    request.well_offsets = scene.well_offsets.data ();
    request.well_offset_count = scene.well_offsets.size ();
    request.active_contexts = scene.active_contexts.data ();
    request.active_context_count = scene.active_contexts.size ();
    request.cells = scene.cells.data ();
    request.cell_count = scene.cells.size ();
    request.edges = scene.edges.data ();
    request.edge_count = scene.edges.size ();
    request.flat_well_edge_count = scene.flat_well_edges;
    request.flat_active_edge_count = scene.flat_active_edges;
    request.well_left = scene.well_left;
    request.well_bottom = scene.well_bottom;
    request.well_right = scene.well_right;
    request.well_top = scene.well_top;
    request.max_contexts = max_contexts;
    request.max_grid_cells = max_grid_cells;
    request.max_memberships = max_memberships;
    request.max_pair_work = max_pair_work;
    std::array<uint8_t, 32> digest;
    if (! db::cuda_active3_digest::request_digest (request, digest)) {
      throw Active3Decline ("unable to digest the live scene request");
    }
    std::copy (digest.begin (), digest.end (), request.scene_digest);

    const double lower_ms =
      std::chrono::duration<double, std::milli> (
        std::chrono::steady_clock::now () - begin).count ();
    const db::CudaActive3Attempt attempt =
      db::cuda_spatial_try_active3_empty (request);
    if (telemetry) {
      tl::info << "CUDA ACTIVE.3 live lowering:"
               << " contexts=" << request.context_count
               << " well_contexts=" << request.well_context_count
               << " active_contexts=" << request.active_context_count
               << " cells=" << request.cell_count
               << " stored_edges=" << request.edge_count
               << " well_edges=" << request.flat_well_edge_count
               << " active_edges=" << request.flat_active_edge_count
               << " lower_ms=" << lower_ms;
    }
    return attempt.disposition == db::CudaActive3Attempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << "CUDA ACTIVE.3 live lowering:"
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << "CUDA ACTIVE.3 live lowering:"
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

} // namespace db
