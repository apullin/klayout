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
#include "dbCudaM1WidthSpace.h"
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
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
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
const int64_t qualified_contact4_distance = 10;
const int qualified_active_layer = 1;
const int qualified_pwell_layer = 2;
const int qualified_nwell_layer = 3;
const int qualified_active3_datatype = 0;
const int qualified_contact4_active_layer = 1;
const int qualified_contact4_contact_layer = 10;
const int qualified_contact4_datatype = 0;
const uint64_t contact4_union_max_rectangles = UINT64_C (32000000);
const uint64_t contact4_union_max_x_slabs = UINT64_C (32000000);
const uint64_t contact4_union_max_memberships = UINT64_C (128000000);
const uint64_t contact4_union_max_events = UINT64_C (256000000);
const uint64_t contact4_union_max_raw_segments = UINT64_C (128000000);
const uint64_t contact4_union_max_boundary_segments = UINT64_C (16000000);
const uint64_t contact4_union_max_contact_edges = UINT64_C (64000000);
const uint64_t contact4_union_max_contact_memberships = UINT64_C (200000000);
const uint64_t contact4_union_max_boundary_cell_visits =
  UINT64_C (200000000);
const uint64_t contact4_union_max_member_visits = UINT64_C (1200000000);
const uint64_t contact4_union_max_pair_work = UINT64_C (1200000000);
const uint64_t contact4_union_max_flat_edges = UINT64_C (128000000);
const uint32_t contact4_union_max_slabs_per_rectangle = 4096;
const uint32_t contact4_union_max_cells_per_edge = 4096;
const uint64_t active3_well_union_max_active_edges =
  UINT64_C (128000000);
const uint64_t active3_well_union_max_active_memberships =
  UINT64_C (300000000);
const uint64_t active3_well_union_max_active_cell_visits =
  UINT64_C (200000000);
const uint64_t active3_well_union_max_member_visits =
  UINT64_C (1200000000);
const uint64_t active3_well_union_max_pair_work =
  UINT64_C (1200000000);
// ACTIVE.4 uses this ABI field as an exact interval-search-step ceiling.
// The qualified x2 SRAM completes 2,069,888,091 checked steps without any
// geometry-sized allocation, so retain a bounded ~1.9x margin.  A dedicated
// environment override keeps this independent from ACTIVE.3's grid search.
const uint64_t active4_well_union_max_search_steps =
  UINT64_C (4000000000);
// The qualified x2 FreePDK45 SRAM contains physical WELL rectangles spanning
// more than 8192 exact union slabs.  Keep this independent from CONTACT.4:
// the cap remains bounded/fail-closed, while 16384 is the measured minimum
// that admits the production WELL scene.
const uint32_t active3_well_union_max_slabs_per_rectangle = 16384;

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
  uint64_t flat_first_well_edges;
  uint64_t flat_second_well_edges;
  uint64_t flat_active_edges;
  std::vector<uint32_t> first_well_edge_counts;
  std::vector<uint32_t> second_well_edge_counts;
  int64_t well_left, well_bottom, well_right, well_top;

  LiveScene ()
    : flat_well_edges (0), flat_first_well_edges (0),
      flat_second_well_edges (0), flat_active_edges (0),
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
      if (! checked_add_u64 (
            scene.flat_first_well_edges,
            scene.first_well_edge_counts [context.cell_id],
            scene.flat_first_well_edges) ||
          ! checked_add_u64 (
            scene.flat_second_well_edges,
            scene.second_well_edge_counts [context.cell_id],
            scene.flat_second_well_edges)) {
        throw Active3Decline ("flat raw-WELL edge count overflow");
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
  uint64_t split_well_edges = 0;
  if (! checked_add_u64 (
        scene.flat_first_well_edges, scene.flat_second_well_edges,
        split_well_edges) ||
      split_well_edges != scene.flat_well_edges) {
    throw Active3Decline ("raw-WELL edge census is inconsistent");
  }
}

LiveScene serialize_live_scene_impl (
  const db::DeepLayer &first_well, const db::DeepLayer *second_well,
  const db::DeepLayer &active,
  uint64_t max_contexts)
{
  const db::Layout &layout = first_well.layout ();
  const db::cell_index_type top = first_well.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  first_well.initial_cell ().collect_called_cells (reachable);
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
  scene.first_well_edge_counts.resize (reachable.size (), 0);
  scene.second_well_edge_counts.resize (reachable.size (), 0);
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
      cell, first_well.layer (), scene.edges,
      record.well_edge_begin, record.well_edge_count);
    scene.first_well_edge_counts [cell_id] = record.well_edge_count;
    if (second_well) {
      uint64_t second_begin = 0;
      uint32_t second_count = 0;
      append_cell_layer (
        cell, second_well->layer (), scene.edges,
        second_begin, second_count);
      const uint64_t expected_second_begin =
        record.well_edge_begin + record.well_edge_count;
      if (second_begin != expected_second_begin ||
          second_count >
            std::numeric_limits<uint32_t>::max () -
              record.well_edge_count) {
        throw Active3Decline (
          "concatenated raw-WELL cell record exceeds uint32");
      }
      record.well_edge_count += second_count;
      scene.second_well_edge_counts [cell_id] = second_count;
    }
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

LiveScene serialize_live_scene (
  const db::DeepLayer &well, const db::DeepLayer &active,
  uint64_t max_contexts)
{
  return serialize_live_scene_impl (well, 0, active, max_contexts);
}

LiveScene serialize_raw_wells_live_scene (
  const db::DeepLayer &nwell, const db::DeepLayer &pwell,
  const db::DeepLayer &active, uint64_t max_contexts)
{
  return serialize_live_scene_impl (nwell, &pwell, active, max_contexts);
}

bool physical_layer_is (
  const db::DeepLayer &operand, int layer, int datatype)
{
  if (operand.layer () >= operand.layout ().layers ()) {
    return false;
  }
  const db::LayerProperties &properties =
    operand.layout ().get_properties (operand.layer ());
  return properties.layer == layer && properties.datatype == datatype;
}

bool eligible_raw_wells (
  const db::DeepLayer &nwell, const db::DeepLayer &pwell,
  const db::DeepLayer &active)
{
  return
    physical_layer_is (
      nwell, qualified_nwell_layer, qualified_active3_datatype) &&
    physical_layer_is (
      pwell, qualified_pwell_layer, qualified_active3_datatype) &&
    physical_layer_is (
      active, qualified_active_layer, qualified_active3_datatype) &&
    nwell.store () == pwell.store () &&
    nwell.store () == active.store () &&
    &nwell.layout () == &pwell.layout () &&
    &nwell.layout () == &active.layout () &&
    nwell.layout_index () == pwell.layout_index () &&
    nwell.layout_index () == active.layout_index () &&
    nwell.initial_cell ().cell_index () ==
      pwell.initial_cell ().cell_index () &&
    nwell.initial_cell ().cell_index () ==
      active.initial_cell ().cell_index () &&
    nwell.breakout_cells () == 0 &&
    pwell.breakout_cells () == 0 &&
    active.breakout_cells () == 0 &&
    nwell.layer () != pwell.layer () &&
    nwell.layer () != active.layer () &&
    pwell.layer () != active.layer () &&
    nwell.layout ().dbu () == 0.0005;
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

bool eligible_contact4 (
  db::edge_relation_type relation, bool different_polygons,
  db::Coord distance, const db::RegionCheckOptions &options,
  const db::DeepLayer &active, const db::DeepLayer &raw_active,
  const db::DeepLayer &contact)
{
  if (raw_active.layer () >= raw_active.layout ().layers () ||
      contact.layer () >= contact.layout ().layers ()) {
    return false;
  }
  const db::LayerProperties &active_properties =
    raw_active.layout ().get_properties (raw_active.layer ());
  const db::LayerProperties &contact_properties =
    contact.layout ().get_properties (contact.layer ());
  return
    relation == db::OverlapRelation &&
    different_polygons &&
    distance == qualified_contact4_distance &&
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
    active.store () == contact.store () &&
    active.store () == raw_active.store () &&
    &active.layout () == &contact.layout () &&
    &active.layout () == &raw_active.layout () &&
    active.layout_index () == contact.layout_index () &&
    active.layout_index () == raw_active.layout_index () &&
    active.initial_cell ().cell_index () ==
      contact.initial_cell ().cell_index () &&
    active.initial_cell ().cell_index () ==
      raw_active.initial_cell ().cell_index () &&
    active.breakout_cells () == 0 &&
    raw_active.breakout_cells () == 0 &&
    contact.breakout_cells () == 0 &&
    active.layer () != contact.layer () &&
    raw_active.layer () != contact.layer () &&
    active_properties.layer == qualified_contact4_active_layer &&
    active_properties.datatype == qualified_contact4_datatype &&
    contact_properties.layer == qualified_contact4_contact_layer &&
    contact_properties.datatype == qualified_contact4_datatype &&
    active.layout ().dbu () == 0.0005;
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
  "raw CONTACT.4 context ABI layout mismatch");
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
  "raw CONTACT.4 cell ABI layout mismatch");
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
  "raw CONTACT.4 polygon ABI layout mismatch");
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
  "raw CONTACT.4 edge ABI layout mismatch");

void fill_contact4_active_union_scene (
  const CudaRawManhattanScene &source, uint32_t role, uint32_t layer,
  const char *digest_domain,
  klayout_cuda_spatial_contact4_active_union_scene_v1 &destination)
{
  std::memset (&destination, 0, sizeof (destination));
  destination.struct_size = sizeof (destination);
  destination.role = role;
  destination.format_version = source.format_version;
  destination.dbu_per_micron = source.dbu_per_micron;
  destination.root_cell = source.root_cell;
  destination.layer = layer;
  destination.datatype = qualified_contact4_datatype;
  destination.contexts = source.contexts.data ();
  destination.context_count = source.contexts.size ();
  destination.context_record_bytes = sizeof (CudaM1WidthSpaceContext);
  destination.layer_contexts = source.metal_contexts.data ();
  destination.layer_context_count = source.metal_contexts.size ();
  destination.context_polygon_offsets =
    source.context_polygon_offsets.data ();
  destination.context_polygon_offset_count =
    source.context_polygon_offsets.size ();
  destination.context_edge_offsets = source.context_edge_offsets.data ();
  destination.context_edge_offset_count =
    source.context_edge_offsets.size ();
  destination.cells = source.cells.data ();
  destination.cell_count = source.cells.size ();
  destination.cell_record_bytes = sizeof (CudaM1WidthSpaceCell);
  destination.polygons = source.polygons.data ();
  destination.polygon_count = source.polygons.size ();
  destination.polygon_record_bytes = sizeof (CudaM1WidthSpacePolygon);
  destination.edges = source.edges.data ();
  destination.edge_count = source.edges.size ();
  destination.edge_record_bytes = sizeof (CudaM1WidthSpaceEdge);
  destination.flat_polygon_count = source.flat_polygon_count;
  destination.flat_edge_count = source.flat_edge_count;
  destination.scene_left = source.scene_left;
  destination.scene_bottom = source.scene_bottom;
  destination.scene_right = source.scene_right;
  destination.scene_top = source.scene_top;
  std::memcpy (
    destination.digest_domain, digest_domain,
    KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES);
  std::copy (
    source.digest.begin (), source.digest.end (),
    destination.scene_digest);
}

bool contact4_scenes_share_hierarchy (
  const CudaRawManhattanScene &active,
  const CudaRawManhattanScene &contact)
{
  if (active.format_version != contact.format_version ||
      active.dbu_per_micron != contact.dbu_per_micron ||
      active.root_cell != contact.root_cell ||
      active.contexts.size () != contact.contexts.size () ||
      active.cells.size () != contact.cells.size ()) {
    return false;
  }
  for (size_t index = 0; index < active.contexts.size (); ++index) {
    const CudaM1WidthSpaceContext &first = active.contexts [index];
    const CudaM1WidthSpaceContext &second = contact.contexts [index];
    if (first.tx != second.tx || first.ty != second.ty ||
        first.cell_id != second.cell_id ||
        first.transform_code != second.transform_code) {
      return false;
    }
  }
  for (size_t index = 0; index < active.cells.size (); ++index) {
    if (active.cells [index].source_cell_index !=
        contact.cells [index].source_cell_index) {
      return false;
    }
  }
  return true;
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

bool cuda_active3_raw_wells_try_empty (
  const db::DeepLayer &raw_nwell, const db::DeepLayer &raw_pwell,
  const db::DeepLayer &raw_active)
{
  const bool telemetry =
    env_enabled ("KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    if (! db::cuda_spatial_active3_raw_wells_requested () ||
        ! eligible_raw_wells (raw_nwell, raw_pwell, raw_active)) {
      return false;
    }

    const uint64_t max_contexts = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_CONTEXTS",
      default_max_contexts);
    const uint64_t max_grid_cells = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_GRID_CELLS",
      default_max_grid_cells);
    const uint64_t max_memberships = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_MEMBERSHIPS",
      default_max_memberships);
    const uint64_t max_pair_work = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_PAIR_WORK",
      default_max_pair_work);
    if (! max_contexts || ! max_grid_cells || ! max_memberships ||
        ! max_pair_work) {
      throw Active3Decline ("a raw-WELL ACTIVE.3 capacity is zero");
    }

    LiveScene scene = serialize_raw_wells_live_scene (
      raw_nwell, raw_pwell, raw_active, max_contexts);
    uint64_t cartesian_pair_bound = 0;
    const bool cartesian_pair_bound_overflow =
      ! checked_multiply_u64 (
          scene.flat_well_edges, scene.flat_active_edges,
          cartesian_pair_bound);
    const std::string cartesian_pair_bound_text =
      cartesian_pair_bound_overflow
        ? std::string ("overflow")
        : std::to_string (cartesian_pair_bound);

    klayout_cuda_spatial_active3_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_BOTH_SUPERSET_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_QUALIFIED_OPTIONS;
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
      throw Active3Decline ("unable to digest the raw-WELL live scene");
    }
    std::copy (digest.begin (), digest.end (), request.scene_digest);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const db::CudaActive3Attempt attempt =
      db::cuda_spatial_try_active3_empty (request);
    const std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << "CUDA ACTIVE.3 raw-WELL live lowering:"
               << " outcome="
               << (attempt.disposition ==
                     db::CudaActive3Attempt::CertifiedEmpty
                     ? "certified-empty"
                     : "cpu-fallback")
               << " contexts=" << request.context_count
               << " well_contexts=" << request.well_context_count
               << " active_contexts=" << request.active_context_count
               << " cells=" << request.cell_count
               << " stored_edges=" << request.edge_count
               << " nwell_edges=" << scene.flat_first_well_edges
               << " pwell_edges=" << scene.flat_second_well_edges
               << " active_edges=" << request.flat_active_edge_count
               << " cartesian_bound=" << cartesian_pair_bound_text
               << " cartesian_bound_overflow="
               << (cartesian_pair_bound_overflow ? 1 : 0)
               << " grid_cells=" << attempt.grid_cell_count
               << " memberships=" << attempt.membership_count
               << " max_pair_work=" << max_pair_work
               << " candidates=" << attempt.candidate_pair_count
               << " raw_hits=" << attempt.raw_hit_count
               << " lower_ms="
               << std::chrono::duration<double, std::milli> (
                    call_begin - begin).count ()
               << " call_ms="
               << std::chrono::duration<double, std::milli> (
                    end - call_begin).count ()
               << " total_ms="
               << std::chrono::duration<double, std::milli> (
                    end - begin).count ();
    }
    return attempt.disposition == db::CudaActive3Attempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << "CUDA ACTIVE.3 raw-WELL live lowering:"
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << "CUDA ACTIVE.3 raw-WELL live lowering:"
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

static bool cuda_contact4_try_empty_impl (
  db::edge_relation_type relation, bool different_polygons,
  db::Coord distance, const db::RegionCheckOptions &options,
  const db::DeepLayer &active, const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_contact, uint32_t opcode, uint32_t option_flags,
  const char *primary_kind, bool raw_primary)
{
  const bool telemetry = env_enabled (
    raw_primary
      ? "KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY"
      : "KLAYOUT_CUDA_CONTACT4_TELEMETRY");
  const char *telemetry_prefix = raw_primary
    ? "CUDA CONTACT.4 raw-ACTIVE live lowering:"
    : "CUDA CONTACT.4 live lowering:";
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    const bool backend_ready = raw_primary
      ? db::cuda_spatial_contact4_raw_active_requested ()
      : db::cuda_spatial_contact4_requested ();
    if (! backend_ready ||
        ! eligible_contact4 (
          relation, different_polygons, distance, options,
          active, raw_active, raw_contact)) {
      return false;
    }

    const uint64_t max_contexts = env_u64 (
      raw_primary
        ? "KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_CONTEXTS"
        : "KLAYOUT_CUDA_CONTACT4_MAX_CONTEXTS",
      default_max_contexts);
    const uint64_t max_grid_cells = env_u64 (
      raw_primary
        ? "KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_GRID_CELLS"
        : "KLAYOUT_CUDA_CONTACT4_MAX_GRID_CELLS",
      default_max_grid_cells);
    const uint64_t max_memberships = env_u64 (
      raw_primary
        ? "KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_MEMBERSHIPS"
        : "KLAYOUT_CUDA_CONTACT4_MAX_MEMBERSHIPS",
      default_max_memberships);
    const uint64_t max_pair_work = env_u64 (
      raw_primary
        ? "KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_PAIR_WORK"
        : "KLAYOUT_CUDA_CONTACT4_MAX_PAIR_WORK",
      default_max_pair_work);
    if (! max_contexts || ! max_grid_cells || ! max_memberships ||
        ! max_pair_work) {
      throw Active3Decline ("a CONTACT.4 capacity is zero");
    }

    //  Reuse the proven hierarchical transport in its indexed/streamed
    //  order: historical "well" records carry the raw CONTACT secondary,
    //  while historical "active" records carry either the merged ACTIVE
    //  primary or its complete raw superset.
    LiveScene scene = serialize_live_scene (
      raw_contact, active, max_contexts);

    klayout_cuda_spatial_active3_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode = opcode;
    request.option_flags = option_flags;
    request.dbu_per_micron = qualified_dbu_per_micron;
    request.distance = qualified_contact4_distance;
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
      throw Active3Decline ("unable to digest the CONTACT.4 live request");
    }
    std::copy (digest.begin (), digest.end (), request.scene_digest);

    const double lower_ms =
      std::chrono::duration<double, std::milli> (
        std::chrono::steady_clock::now () - begin).count ();
    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const db::CudaActive3Attempt attempt = raw_primary
      ? db::cuda_spatial_try_contact4_raw_active_empty (request)
      : db::cuda_spatial_try_contact4_empty (request);
    const std::chrono::steady_clock::time_point done =
      std::chrono::steady_clock::now ();
    const double call_ms =
      std::chrono::duration<double, std::milli> (done - call_begin).count ();
    const double live_total_ms =
      std::chrono::duration<double, std::milli> (done - begin).count ();
    if (telemetry) {
      tl::info << telemetry_prefix
               << " primary=" << primary_kind
               << " contexts=" << request.context_count
               << " indexed_contact_contexts=" << request.well_context_count
               << " streamed_active_contexts=" << request.active_context_count
               << " cells=" << request.cell_count
               << " stored_edges=" << request.edge_count
               << " indexed_contact_edges=" << request.flat_well_edge_count
               << " streamed_active_edges=" << request.flat_active_edge_count
               << " lower_ms=" << lower_ms
               << " call_ms=" << call_ms
               << " live_total_ms=" << live_total_ms;
    }
    return attempt.disposition == db::CudaActive3Attempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << telemetry_prefix
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << telemetry_prefix
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

static bool cuda_well_union_try_empty_impl (
  const db::DeepLayer &raw_nwell, const db::DeepLayer &raw_pwell,
  const db::DeepLayer &raw_active, uint32_t opcode,
  uint32_t option_flags, const char *telemetry_env,
  const char *telemetry_prefix)
{
  const bool telemetry = env_enabled (telemetry_env);
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    // Capability discovery deliberately precedes hierarchy serialization.
    if (! db::cuda_spatial_active3_well_union_requested () ||
        ! eligible_raw_wells (raw_nwell, raw_pwell, raw_active)) {
      return false;
    }

    CudaM1WidthSpaceSceneLimits scene_limits;
    scene_limits.max_cells = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CELLS",
      scene_limits.max_cells);
    scene_limits.max_contexts = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CONTEXTS",
      scene_limits.max_contexts);
    scene_limits.max_stored_polygons = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_STORED_POLYGONS",
      scene_limits.max_stored_polygons);
    scene_limits.max_stored_edges = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_STORED_EDGES",
      scene_limits.max_stored_edges);

    const uint64_t max_rectangles = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_RECTANGLES",
      contact4_union_max_rectangles);
    scene_limits.max_flat_polygons = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_FLAT_POLYGONS",
      contact4_union_max_rectangles);
    scene_limits.max_flat_edges = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_FLAT_EDGES",
      contact4_union_max_flat_edges);

    const uint64_t max_x_slabs = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_X_SLABS",
      contact4_union_max_x_slabs);
    const uint64_t max_union_memberships = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_UNION_MEMBERSHIPS",
      contact4_union_max_memberships);
    const uint64_t max_events = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_EVENTS",
      contact4_union_max_events);
    const uint64_t max_raw_segments = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_RAW_SEGMENTS",
      contact4_union_max_raw_segments);
    const uint64_t max_boundary_segments = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_BOUNDARY_SEGMENTS",
      contact4_union_max_boundary_segments);
    const uint64_t max_slabs_per_rectangle = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_SLABS_PER_RECTANGLE",
      active3_well_union_max_slabs_per_rectangle);
    const uint64_t max_active_edges = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_ACTIVE_EDGES",
      active3_well_union_max_active_edges);
    const uint64_t max_grid_cells = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_GRID_CELLS",
      default_max_grid_cells);
    const uint64_t max_active_memberships = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_ACTIVE_MEMBERSHIPS",
      active3_well_union_max_active_memberships);
    const uint64_t max_active_cell_visits = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_ACTIVE_CELL_VISITS",
      active3_well_union_max_active_cell_visits);
    const bool active4_subset =
      opcode == KLAYOUT_CUDA_SPATIAL_ACTIVE4_WELL_UNION_SUBSET_EMPTY;
    const uint64_t max_member_visits = env_u64 (
      active4_subset
        ? "KLAYOUT_CUDA_ACTIVE4_WELL_UNION_MAX_SEARCH_STEPS"
        : "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_MEMBER_VISITS",
      active4_subset
        ? active4_well_union_max_search_steps
        : active3_well_union_max_member_visits);
    const uint64_t max_pair_work = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_PAIR_WORK",
      active3_well_union_max_pair_work);
    const uint64_t max_cells_per_active_edge = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CELLS_PER_ACTIVE_EDGE",
      contact4_union_max_cells_per_edge);
    const uint64_t max_cells_per_well_edge = env_u64 (
      "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CELLS_PER_WELL_EDGE",
      contact4_union_max_cells_per_edge);
    const uint64_t device =
      env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0);

    if (! scene_limits.max_cells || ! scene_limits.max_contexts ||
        ! scene_limits.max_stored_polygons ||
        ! scene_limits.max_stored_edges ||
        ! scene_limits.max_flat_polygons ||
        ! scene_limits.max_flat_edges || ! max_rectangles ||
        ! max_x_slabs || ! max_union_memberships || ! max_events ||
        ! max_raw_segments || ! max_boundary_segments ||
        ! max_slabs_per_rectangle ||
        max_slabs_per_rectangle >
          std::numeric_limits<uint32_t>::max () ||
        ! max_active_edges || ! max_grid_cells ||
        ! max_active_memberships || ! max_active_cell_visits ||
        ! max_member_visits || ! max_pair_work ||
        ! max_cells_per_active_edge ||
        max_cells_per_active_edge >
          std::numeric_limits<uint32_t>::max () ||
        ! max_cells_per_well_edge ||
        max_cells_per_well_edge >
          std::numeric_limits<uint32_t>::max () ||
        device > uint64_t (std::numeric_limits<int32_t>::max ())) {
      throw Active3Decline (
        "an ACTIVE.3 WELL-union capacity or device is invalid");
    }

    CudaRawManhattanScene well_scene;
    CudaRawManhattanScene active_scene;
    std::string reason;
    if (! cuda_well_union_raw_manhattan_build_scene (
          raw_nwell, raw_pwell, scene_limits, well_scene, &reason)) {
      throw Active3Decline (
        reason.empty ()
          ? "unable to serialize the combined raw-WELL scene"
          : reason);
    }
    if (! cuda_active_raw_manhattan_build_scene (
          raw_active, scene_limits, active_scene, &reason)) {
      throw Active3Decline (
        reason.empty ()
          ? "unable to serialize the qualified raw ACTIVE scene"
          : reason);
    }
    if (! contact4_scenes_share_hierarchy (
          well_scene, active_scene)) {
      throw Active3Decline (
        "combined raw WELL and ACTIVE scenes do not share one hierarchy "
        "identity");
    }

    klayout_cuda_spatial_active3_well_union_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode = opcode;
    request.option_flags = option_flags;
    request.format_version = well_scene.format_version;
    request.dbu_per_micron = well_scene.dbu_per_micron;
    request.device = int32_t (device);
    request.distance = qualified_distance;
    request.grid_cell_size = qualified_grid_cell;
    request.secondary_well_layer = qualified_pwell_layer;
    request.secondary_well_datatype = qualified_active3_datatype;
    fill_contact4_active_union_scene (
      well_scene,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_ROLE,
      qualified_nwell_layer,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_DIGEST_DOMAIN,
      request.wells);
    fill_contact4_active_union_scene (
      active_scene,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_ROLE,
      qualified_active_layer,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_DIGEST_DOMAIN,
      request.active);
    request.max_contexts = scene_limits.max_contexts;
    request.max_rectangles = max_rectangles;
    request.max_x_slabs = max_x_slabs;
    request.max_union_memberships = max_union_memberships;
    request.max_events = max_events;
    request.max_raw_segments = max_raw_segments;
    request.max_boundary_segments = max_boundary_segments;
    request.max_slabs_per_rectangle =
      uint32_t (max_slabs_per_rectangle);
    request.max_active_edges = max_active_edges;
    request.max_grid_cells = max_grid_cells;
    request.max_active_memberships = max_active_memberships;
    request.max_active_cell_visits = max_active_cell_visits;
    request.max_member_visits = max_member_visits;
    request.max_pair_work = max_pair_work;
    request.max_cells_per_active_edge =
      uint32_t (max_cells_per_active_edge);
    request.max_cells_per_well_edge =
      uint32_t (max_cells_per_well_edge);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const db::CudaActive3WellUnionAttempt attempt =
      db::cuda_spatial_try_active3_well_union_empty (request);
    const std::chrono::steady_clock::time_point done =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << telemetry_prefix
               << " well_contexts=" << request.wells.context_count
               << " active_contexts=" << request.active.context_count
               << " well_stored_polygons="
               << request.wells.polygon_count
               << " well_flat_polygons="
               << request.wells.flat_polygon_count
               << " well_flat_edges=" << request.wells.flat_edge_count
               << " active_stored_polygons="
               << request.active.polygon_count
               << " active_flat_polygons="
               << request.active.flat_polygon_count
               << " active_flat_edges="
               << request.active.flat_edge_count
               << " lower_ms="
               << std::chrono::duration<double, std::milli> (
                    call_begin - begin).count ()
               << " call_ms="
               << std::chrono::duration<double, std::milli> (
                    done - call_begin).count ()
               << " live_total_ms="
               << std::chrono::duration<double, std::milli> (
                    done - begin).count ();
    }
    return attempt.disposition ==
      db::CudaActive3WellUnionAttempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << telemetry_prefix
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        // Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << telemetry_prefix
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        // Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

bool cuda_active3_well_union_try_empty (
  const db::DeepLayer &raw_nwell, const db::DeepLayer &raw_pwell,
  const db::DeepLayer &raw_active)
{
  return cuda_well_union_try_empty_impl (
    raw_nwell, raw_pwell, raw_active,
    KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_EMPTY,
    KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_QUALIFIED_OPTIONS,
    "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_TELEMETRY",
    "CUDA ACTIVE.3 exact WELL-union live lowering:");
}

bool cuda_active4_well_union_try_empty (
  const db::DeepLayer &raw_nwell, const db::DeepLayer &raw_pwell,
  const db::DeepLayer &raw_active)
{
  if (! env_enabled ("KLAYOUT_CUDA_ACTIVE4_WELL_UNION")) {
    return false;
  }
  return cuda_well_union_try_empty_impl (
    raw_nwell, raw_pwell, raw_active,
    KLAYOUT_CUDA_SPATIAL_ACTIVE4_WELL_UNION_SUBSET_EMPTY,
    KLAYOUT_CUDA_SPATIAL_ACTIVE4_WELL_UNION_QUALIFIED_OPTIONS,
    "KLAYOUT_CUDA_ACTIVE4_WELL_UNION_TELEMETRY",
    "CUDA ACTIVE.4 exact WELL-union live lowering:");
}

bool cuda_contact4_try_empty (
  db::edge_relation_type relation, bool different_polygons,
  db::Coord distance, const db::RegionCheckOptions &options,
  const db::DeepLayer &merged_active, const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_contact)
{
  return cuda_contact4_try_empty_impl (
    relation, different_polygons, distance, options,
    merged_active, raw_active, raw_contact,
    KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_SUPERSET_EMPTY,
    KLAYOUT_CUDA_SPATIAL_CONTACT4_QUALIFIED_OPTIONS, "merged", false);
}

bool cuda_contact4_raw_active_try_empty (
  db::edge_relation_type relation, bool different_polygons,
  db::Coord distance, const db::RegionCheckOptions &options,
  const db::DeepLayer &raw_active, const db::DeepLayer &raw_contact)
{
  if (! env_enabled ("KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE")) {
    return false;
  }
  //  Do not pay for a second complete raw-scene lowering after the exact
  //  fused capability has already declined.  When available, that path
  //  strictly subsumes this conservative raw-superset certificate.
  if (db::cuda_spatial_contact4_active_union_requested ()) {
    return false;
  }
  return cuda_contact4_try_empty_impl (
    relation, different_polygons, distance, options,
    raw_active, raw_active, raw_contact,
    KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_SUPERSET_EMPTY,
    KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_QUALIFIED_OPTIONS, "raw", true);
}

bool cuda_contact4_active_union_try_empty (
  db::edge_relation_type relation, bool different_polygons,
  db::Coord distance, const db::RegionCheckOptions &options,
  const db::DeepLayer &raw_active, const db::DeepLayer &raw_contact)
{
  const bool telemetry =
    env_enabled ("KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    //  Capability discovery deliberately precedes hierarchy serialization.
    if (! db::cuda_spatial_contact4_active_union_requested () ||
        ! eligible_contact4 (
          relation, different_polygons, distance, options,
          raw_active, raw_active, raw_contact)) {
      return false;
    }

    CudaM1WidthSpaceSceneLimits scene_limits;
    scene_limits.max_cells = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_CELLS",
      scene_limits.max_cells);
    scene_limits.max_contexts = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_CONTEXTS",
      scene_limits.max_contexts);
    scene_limits.max_stored_polygons = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_STORED_POLYGONS",
      scene_limits.max_stored_polygons);
    scene_limits.max_stored_edges = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_STORED_EDGES",
      scene_limits.max_stored_edges);

    const uint64_t max_rectangles = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_RECTANGLES",
      contact4_union_max_rectangles);
    scene_limits.max_flat_polygons = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_FLAT_POLYGONS",
      contact4_union_max_rectangles);
    scene_limits.max_flat_edges = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_FLAT_EDGES",
      contact4_union_max_flat_edges);

    const uint64_t max_x_slabs = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_X_SLABS",
      contact4_union_max_x_slabs);
    const uint64_t max_union_memberships = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_UNION_MEMBERSHIPS",
      contact4_union_max_memberships);
    const uint64_t max_events = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_EVENTS",
      contact4_union_max_events);
    const uint64_t max_raw_segments = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_RAW_SEGMENTS",
      contact4_union_max_raw_segments);
    const uint64_t max_boundary_segments = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_BOUNDARY_SEGMENTS",
      contact4_union_max_boundary_segments);
    const uint64_t max_slabs_per_rectangle = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_SLABS_PER_RECTANGLE",
      contact4_union_max_slabs_per_rectangle);
    const uint64_t max_contact_edges = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_CONTACT_EDGES",
      contact4_union_max_contact_edges);
    const uint64_t max_grid_cells = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_GRID_CELLS",
      default_max_grid_cells);
    const uint64_t max_contact_memberships = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_CONTACT_MEMBERSHIPS",
      contact4_union_max_contact_memberships);
    const uint64_t max_boundary_cell_visits = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_BOUNDARY_CELL_VISITS",
      contact4_union_max_boundary_cell_visits);
    const uint64_t max_member_visits = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_MEMBER_VISITS",
      contact4_union_max_member_visits);
    const uint64_t max_pair_work = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_PAIR_WORK",
      contact4_union_max_pair_work);
    const uint64_t max_cells_per_contact_edge = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_CELLS_PER_CONTACT_EDGE",
      contact4_union_max_cells_per_edge);
    const uint64_t max_cells_per_boundary_edge = env_u64 (
      "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_MAX_CELLS_PER_BOUNDARY_EDGE",
      contact4_union_max_cells_per_edge);
    const uint64_t device =
      env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0);

    if (! scene_limits.max_cells || ! scene_limits.max_contexts ||
        ! scene_limits.max_stored_polygons ||
        ! scene_limits.max_stored_edges ||
        ! scene_limits.max_flat_polygons ||
        ! scene_limits.max_flat_edges || ! max_rectangles ||
        ! max_x_slabs || ! max_union_memberships || ! max_events ||
        ! max_raw_segments || ! max_boundary_segments ||
        ! max_slabs_per_rectangle ||
        max_slabs_per_rectangle >
          std::numeric_limits<uint32_t>::max () ||
        ! max_contact_edges || ! max_grid_cells ||
        ! max_contact_memberships || ! max_boundary_cell_visits ||
        ! max_member_visits || ! max_pair_work ||
        ! max_cells_per_contact_edge ||
        max_cells_per_contact_edge >
          std::numeric_limits<uint32_t>::max () ||
        ! max_cells_per_boundary_edge ||
        max_cells_per_boundary_edge >
          std::numeric_limits<uint32_t>::max () ||
        device > uint64_t (std::numeric_limits<int32_t>::max ())) {
      throw Active3Decline (
        "a CONTACT.4 ACTIVE-union capacity or device is invalid");
    }

    CudaRawManhattanScene active_scene;
    CudaRawManhattanScene contact_scene;
    std::string reason;
    if (! cuda_active_raw_manhattan_build_scene (
          raw_active, scene_limits, active_scene, &reason)) {
      throw Active3Decline (
        reason.empty ()
          ? "unable to serialize the qualified raw ACTIVE scene"
          : reason);
    }
    if (! cuda_contact_raw_manhattan_build_scene (
          raw_contact, scene_limits, contact_scene, &reason)) {
      throw Active3Decline (
        reason.empty ()
          ? "unable to serialize the qualified raw CONTACT scene"
          : reason);
    }
    if (! contact4_scenes_share_hierarchy (
          active_scene, contact_scene)) {
      throw Active3Decline (
        "raw ACTIVE and CONTACT scenes do not share one hierarchy identity");
    }

    klayout_cuda_spatial_contact4_active_union_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_QUALIFIED_OPTIONS;
    request.format_version = active_scene.format_version;
    request.dbu_per_micron = active_scene.dbu_per_micron;
    request.device = int32_t (device);
    request.distance = qualified_contact4_distance;
    request.grid_cell_size = qualified_grid_cell;
    fill_contact4_active_union_scene (
      active_scene, KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE,
      qualified_contact4_active_layer,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_DIGEST_DOMAIN,
      request.active);
    fill_contact4_active_union_scene (
      contact_scene, KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE,
      qualified_contact4_contact_layer,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_DIGEST_DOMAIN,
      request.contact);
    request.max_contexts = scene_limits.max_contexts;
    request.max_rectangles = max_rectangles;
    request.max_x_slabs = max_x_slabs;
    request.max_union_memberships = max_union_memberships;
    request.max_events = max_events;
    request.max_raw_segments = max_raw_segments;
    request.max_boundary_segments = max_boundary_segments;
    request.max_slabs_per_rectangle =
      uint32_t (max_slabs_per_rectangle);
    request.max_contact_edges = max_contact_edges;
    request.max_grid_cells = max_grid_cells;
    request.max_contact_memberships = max_contact_memberships;
    request.max_boundary_cell_visits = max_boundary_cell_visits;
    request.max_member_visits = max_member_visits;
    request.max_pair_work = max_pair_work;
    request.max_cells_per_contact_edge =
      uint32_t (max_cells_per_contact_edge);
    request.max_cells_per_boundary_edge =
      uint32_t (max_cells_per_boundary_edge);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const db::CudaContact4ActiveUnionAttempt attempt =
      db::cuda_spatial_try_contact4_active_union_empty (request);
    const std::chrono::steady_clock::time_point done =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << "CUDA CONTACT.4 fused ACTIVE-union live lowering:"
               << " active_contexts=" << request.active.context_count
               << " contact_contexts=" << request.contact.context_count
               << " active_stored_polygons="
               << request.active.polygon_count
               << " active_flat_polygons="
               << request.active.flat_polygon_count
               << " active_flat_edges=" << request.active.flat_edge_count
               << " contact_stored_polygons="
               << request.contact.polygon_count
               << " contact_flat_polygons="
               << request.contact.flat_polygon_count
               << " contact_flat_edges="
               << request.contact.flat_edge_count
               << " lower_ms="
               << std::chrono::duration<double, std::milli> (
                    call_begin - begin).count ()
               << " call_ms="
               << std::chrono::duration<double, std::milli> (
                    done - call_begin).count ()
               << " live_total_ms="
               << std::chrono::duration<double, std::milli> (
                    done - begin).count ();
    }
    return attempt.disposition ==
      db::CudaContact4ActiveUnionAttempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << "CUDA CONTACT.4 fused ACTIVE-union live lowering:"
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << "CUDA CONTACT.4 fused ACTIVE-union live lowering:"
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

} // namespace db
