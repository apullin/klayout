/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaVia1Stack.h"

#include "dbArray.h"
#include "dbCell.h"
#include "dbCudaSpatialBackend.h"
#include "dbCudaVia1StackDigest.h"
#include "dbDeepShapeStore.h"
#include "dbLayerProperties.h"
#include "dbLayout.h"
#include "dbPolygon.h"
#include "dbShape.h"
#include "dbShapes.h"
#include "tlLog.h"

#include <algorithm>
#include <array>
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
const uint64_t default_max_metal_memberships = UINT64_C (300000000);
const uint64_t default_max_via_memberships = UINT64_C (100000000);
const uint64_t default_max_pair_work = UINT64_C (2000000000000);
const unsigned maximum_hierarchy_depth = 1024;
const int64_t accepted_coordinate_magnitude = INT64_C (1000000000000);
const int64_t qualified_enclosure_distance = 70;
const int64_t qualified_cut_size = 130;
const int64_t qualified_spacing_distance = 150;
const int64_t qualified_grid_cell = 2000;
const uint32_t qualified_dbu_per_micron = 2000;

class Via1StackDecline
  : public std::runtime_error
{
public:
  explicit Via1StackDecline (const std::string &message)
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

int32_t env_device ()
{
  return int32_t (std::min<uint64_t> (
    env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0),
    uint64_t (std::numeric_limits<int32_t>::max ())));
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
    throw Via1StackDecline (std::string (what) + " overflows signed int64");
  }
  const int64_t result = int64_t (value);
  if (result < -accepted_coordinate_magnitude ||
      result > accepted_coordinate_magnitude) {
    throw Via1StackDecline (
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
    throw Via1StackDecline ("invalid orthogonal transform code");
  }
  const Matrix &matrix = transforms [code];
  return std::make_pair (
    __int128 (matrix.xx) * x + __int128 (matrix.xy) * y,
    __int128 (matrix.yx) * x + __int128 (matrix.yy) * y);
}

uint32_t compose_transform (uint32_t outer, uint32_t inner)
{
  if (outer >= 8 || inner >= 8) {
    throw Via1StackDecline (
      "invalid transform during hierarchy expansion");
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
  throw Via1StackDecline (
    "orthogonal transform composition escaped its group");
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

struct LocalBounds
{
  bool valid;
  int64_t left, bottom, right, top;

  LocalBounds ()
    : valid (false), left (0), bottom (0), right (0), top (0)
  {
    //  nothing yet
  }

  void add (const klayout_cuda_spatial_via1_stack_box_v1 &box)
  {
    if (! valid) {
      left = box.left;
      bottom = box.bottom;
      right = box.right;
      top = box.top;
      valid = true;
    } else {
      left = std::min (left, box.left);
      bottom = std::min (bottom, box.bottom);
      right = std::max (right, box.right);
      top = std::max (top, box.top);
    }
  }
};

struct LiveScene
{
  std::vector<klayout_cuda_spatial_via1_stack_context_v1> contexts;
  std::vector<uint32_t> metal1_contexts;
  std::vector<uint64_t> metal1_offsets;
  std::vector<uint32_t> via1_contexts;
  std::vector<uint64_t> via1_offsets;
  std::vector<uint32_t> metal2_contexts;
  std::vector<uint64_t> metal2_offsets;
  std::vector<klayout_cuda_spatial_via1_stack_cell_v1> cells;
  std::vector<klayout_cuda_spatial_via1_stack_box_v1> boxes;
  std::vector<LocalBounds> bounds;
  uint64_t flat_metal1_boxes;
  uint64_t flat_via1_boxes;
  uint64_t flat_metal2_boxes;
  bool have_scene_box;
  int64_t scene_left, scene_bottom, scene_right, scene_top;

  LiveScene ()
    : flat_metal1_boxes (0), flat_via1_boxes (0),
      flat_metal2_boxes (0), have_scene_box (false),
      scene_left (0), scene_bottom (0), scene_right (0), scene_top (0)
  {
    //  nothing yet
  }
};

struct ContourEdge
{
  int64_t x1, y1, x2, y2;
};

bool positive_collinear_overlap (
  const ContourEdge &a, const ContourEdge &b)
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
  const ContourEdge &a, const ContourEdge &b)
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
  const ContourEdge &horizontal = a.y1 == a.y2 ? a : b;
  const ContourEdge &vertical = a.y1 == a.y2 ? b : a;
  return
    std::min (horizontal.x1, horizontal.x2) <= vertical.x1 &&
    vertical.x1 <= std::max (horizontal.x1, horizontal.x2) &&
    std::min (vertical.y1, vertical.y2) <= horizontal.y1 &&
    horizontal.y1 <= std::max (vertical.y1, vertical.y2);
}

std::vector<ContourEdge> checked_contour (
  const db::Shape &shape, db::Box &bbox, __int128 &twice_area)
{
  if (shape.prop_id () != 0) {
    throw Via1StackDecline ("operand polygon has properties");
  }
  if (! shape.is_box () && ! shape.is_polygon ()) {
    throw Via1StackDecline ("operand layer contains a non-polygon shape");
  }

  db::Polygon polygon;
  if (! shape.polygon (polygon) || polygon.holes () != 0) {
    throw Via1StackDecline ("operand polygon is malformed or has holes");
  }
  bbox = polygon.box ();
  if (bbox.empty ()) {
    throw Via1StackDecline ("operand polygon has an empty bounding box");
  }

  std::vector<ContourEdge> contour;
  std::set<std::pair<int64_t, int64_t> > vertices;
  twice_area = 0;
  for (db::Polygon::polygon_edge_iterator edge = polygon.begin_edge ();
       ! edge.at_end (); ++edge) {
    const int64_t x1 = narrow_i64 ((*edge).p1 ().x (), "polygon x1");
    const int64_t y1 = narrow_i64 ((*edge).p1 ().y (), "polygon y1");
    const int64_t x2 = narrow_i64 ((*edge).p2 ().x (), "polygon x2");
    const int64_t y2 = narrow_i64 ((*edge).p2 ().y (), "polygon y2");
    if ((x1 == x2 && y1 == y2) || ! (x1 == x2 || y1 == y2)) {
      throw Via1StackDecline (
        "operand polygon has a degenerate or non-Manhattan edge");
    }
    if (! vertices.insert (std::make_pair (x1, y1)).second) {
      throw Via1StackDecline ("operand polygon repeats a contour vertex");
    }
    contour.push_back (ContourEdge { x1, y1, x2, y2 });
    twice_area += __int128 (x1) * y2 - __int128 (x2) * y1;
  }
  if (contour.size () < 4 || twice_area >= 0) {
    throw Via1StackDecline (
      "operand polygon is too small or not clockwise");
  }
  for (size_t i = 0; i < contour.size (); ++i) {
    const size_t following = (i + 1) % contour.size ();
    if (contour [i].x2 != contour [following].x1 ||
        contour [i].y2 != contour [following].y1) {
      throw Via1StackDecline ("operand polygon contour is open");
    }
    for (size_t j = i + 1; j < contour.size (); ++j) {
      if (! manhattan_segments_intersect (contour [i], contour [j])) {
        continue;
      }
      const bool adjacent =
        j == i + 1 || (i == 0 && j + 1 == contour.size ());
      if (! adjacent ||
          positive_collinear_overlap (contour [i], contour [j])) {
        throw Via1StackDecline (
          "operand polygon contour self-intersects");
      }
    }
  }
  return contour;
}

klayout_cuda_spatial_via1_stack_box_v1 box_record (const db::Box &box)
{
  return klayout_cuda_spatial_via1_stack_box_v1 {
    narrow_i64 (box.left (), "box left"),
    narrow_i64 (box.bottom (), "box bottom"),
    narrow_i64 (box.right (), "box right"),
    narrow_i64 (box.top (), "box top")
  };
}

bool contour_is_box (
  const std::vector<ContourEdge> &contour, const db::Box &bbox)
{
  if (contour.size () != 4) {
    return false;
  }
  uint32_t corner_mask = 0;
  for (std::vector<ContourEdge>::const_iterator edge = contour.begin ();
       edge != contour.end (); ++edge) {
    const bool right = edge->x1 == bbox.right ();
    const bool top = edge->y1 == bbox.top ();
    if ((! right && edge->x1 != bbox.left ()) ||
        (! top && edge->y1 != bbox.bottom ())) {
      return false;
    }
    corner_mask |= 1u << ((top ? 2u : 0u) | (right ? 1u : 0u));
  }
  return corner_mask == 0xfu;
}

std::vector<klayout_cuda_spatial_via1_stack_box_v1> decompose_y (
  const std::vector<ContourEdge> &contour, __int128 twice_area)
{
  std::vector<int64_t> coordinates;
  coordinates.reserve (contour.size ());
  for (std::vector<ContourEdge>::const_iterator edge = contour.begin ();
       edge != contour.end (); ++edge) {
    coordinates.push_back (edge->y1);
  }
  std::sort (coordinates.begin (), coordinates.end ());
  coordinates.erase (
    std::unique (coordinates.begin (), coordinates.end ()),
    coordinates.end ());
  if (coordinates.size () < 2 || twice_area >= 0) {
    throw Via1StackDecline ("invalid Y-slab decomposition input");
  }

  std::vector<klayout_cuda_spatial_via1_stack_box_v1> boxes;
  __int128 area = 0;
  for (size_t slab = 0; slab + 1 < coordinates.size (); ++slab) {
    const int64_t bottom = coordinates [slab];
    const int64_t top = coordinates [slab + 1];
    if (bottom >= top) {
      throw Via1StackDecline ("non-positive Y-slab");
    }
    std::vector<int64_t> crossings;
    for (std::vector<ContourEdge>::const_iterator edge = contour.begin ();
         edge != contour.end (); ++edge) {
      if (edge->x1 != edge->x2) {
        continue;
      }
      const int64_t edge_bottom = std::min (edge->y1, edge->y2);
      const int64_t edge_top = std::max (edge->y1, edge->y2);
      if (edge_bottom <= bottom && edge_top >= top) {
        crossings.push_back (edge->x1);
      }
    }
    std::sort (crossings.begin (), crossings.end ());
    if (crossings.empty () || crossings.size () % 2) {
      throw Via1StackDecline ("odd or empty Y-slab crossings");
    }
    for (size_t crossing = 0; crossing < crossings.size (); crossing += 2) {
      const int64_t left = crossings [crossing];
      const int64_t right = crossings [crossing + 1];
      if (left >= right) {
        throw Via1StackDecline ("non-positive Y-slab interval");
      }
      boxes.push_back (
        klayout_cuda_spatial_via1_stack_box_v1 {
          left, bottom, right, top });
      area += __int128 (right - left) * (top - bottom);
    }
  }
  if (boxes.empty () || area * 2 != -twice_area) {
    throw Via1StackDecline ("Y-slab decomposition area mismatch");
  }
  return boxes;
}

std::vector<klayout_cuda_spatial_via1_stack_box_v1> decompose_x (
  const std::vector<ContourEdge> &contour, __int128 twice_area)
{
  std::vector<int64_t> coordinates;
  coordinates.reserve (contour.size ());
  for (std::vector<ContourEdge>::const_iterator edge = contour.begin ();
       edge != contour.end (); ++edge) {
    coordinates.push_back (edge->x1);
  }
  std::sort (coordinates.begin (), coordinates.end ());
  coordinates.erase (
    std::unique (coordinates.begin (), coordinates.end ()),
    coordinates.end ());
  if (coordinates.size () < 2 || twice_area >= 0) {
    throw Via1StackDecline ("invalid X-slab decomposition input");
  }

  std::vector<klayout_cuda_spatial_via1_stack_box_v1> boxes;
  __int128 area = 0;
  for (size_t slab = 0; slab + 1 < coordinates.size (); ++slab) {
    const int64_t left = coordinates [slab];
    const int64_t right = coordinates [slab + 1];
    if (left >= right) {
      throw Via1StackDecline ("non-positive X-slab");
    }
    std::vector<int64_t> crossings;
    for (std::vector<ContourEdge>::const_iterator edge = contour.begin ();
         edge != contour.end (); ++edge) {
      if (edge->y1 != edge->y2) {
        continue;
      }
      const int64_t edge_left = std::min (edge->x1, edge->x2);
      const int64_t edge_right = std::max (edge->x1, edge->x2);
      if (edge_left <= left && edge_right >= right) {
        crossings.push_back (edge->y1);
      }
    }
    std::sort (crossings.begin (), crossings.end ());
    if (crossings.empty () || crossings.size () % 2) {
      throw Via1StackDecline ("odd or empty X-slab crossings");
    }
    for (size_t crossing = 0; crossing < crossings.size (); crossing += 2) {
      const int64_t bottom = crossings [crossing];
      const int64_t top = crossings [crossing + 1];
      if (bottom >= top) {
        throw Via1StackDecline ("non-positive X-slab interval");
      }
      boxes.push_back (
        klayout_cuda_spatial_via1_stack_box_v1 {
          left, bottom, right, top });
      area += __int128 (right - left) * (top - bottom);
    }
  }
  if (boxes.empty () || area * 2 != -twice_area) {
    throw Via1StackDecline ("X-slab decomposition area mismatch");
  }
  return boxes;
}

void append_cell_layer (
  const db::Cell &cell, unsigned int layer, bool cut_layer,
  std::vector<klayout_cuda_spatial_via1_stack_box_v1> &destination,
  LocalBounds &bounds, uint64_t &begin, uint32_t &count)
{
  begin = destination.size ();
  const db::Shapes &shapes = cell.shapes (layer);
  for (db::Shapes::shape_iterator shape =
         shapes.begin (db::ShapeIterator::All);
       ! shape.at_end (); ++shape) {
    db::Box bbox;
    __int128 twice_area = 0;
    const std::vector<ContourEdge> contour =
      checked_contour (*shape, bbox, twice_area);
    const klayout_cuda_spatial_via1_stack_box_v1 polygon_box =
      box_record (bbox);
    bounds.add (polygon_box);

    if (cut_layer) {
      if (! contour_is_box (contour, bbox)) {
        throw Via1StackDecline (
          "VIA1 layer contains a non-rectangular polygon");
      }
      if (polygon_box.right - polygon_box.left != qualified_cut_size ||
          polygon_box.top - polygon_box.bottom != qualified_cut_size) {
        throw Via1StackDecline (
          "VIA1 rectangle does not have the qualified 130x130 DBU size");
      }
      destination.push_back (polygon_box);
    } else if (contour_is_box (contour, bbox)) {
      destination.push_back (polygon_box);
    } else {
      std::vector<klayout_cuda_spatial_via1_stack_box_v1> boxes =
        decompose_y (contour, twice_area);
      const std::vector<klayout_cuda_spatial_via1_stack_box_v1> x_boxes =
        decompose_x (contour, twice_area);
      boxes.insert (boxes.end (), x_boxes.begin (), x_boxes.end ());
      destination.insert (destination.end (), boxes.begin (), boxes.end ());
    }

    const uint64_t new_count = uint64_t (destination.size ()) - begin;
    if (new_count > std::numeric_limits<uint32_t>::max ()) {
      throw Via1StackDecline ("per-cell box count exceeds uint32");
    }
    count = uint32_t (new_count);
  }
}

InstanceTemplate make_instance (
  const db::Instance &instance,
  const std::map<db::cell_index_type, uint32_t> &dense_cells)
{
  if (instance.prop_id () != 0 || instance.is_complex ()) {
    throw Via1StackDecline (
      "hierarchy has an instance property or complex transform");
  }

  const std::map<db::cell_index_type, uint32_t>::const_iterator child =
    dense_cells.find (instance.cell_index ());
  if (child == dense_cells.end ()) {
    throw Via1StackDecline (
      "hierarchy instance targets an unreachable cell");
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
    throw Via1StackDecline ("hierarchy has an irregular instance array");
  }
  if (delegate == 0) {
    a = db::Vector ();
    b = db::Vector ();
    na = nb = 1;
  }
  if (na == 0 || nb == 0 ||
      na > std::numeric_limits<uint32_t>::max () ||
      nb > std::numeric_limits<uint32_t>::max ()) {
    throw Via1StackDecline ("hierarchy has an invalid array dimension");
  }

  const db::Trans &trans = instance.front ();
  const int transform_code = trans.rot ();
  if (transform_code < 0 || transform_code >= 8) {
    throw Via1StackDecline (
      "hierarchy has an invalid orthogonal transform");
  }

  InstanceTemplate result;
  result.child_cell = child->second;
  result.columns = uint32_t (na);
  result.rows = uint32_t (nb);
  result.transform_code = uint32_t (transform_code);
  result.dx = narrow_i64 (trans.disp ().x (), "instance dx");
  result.dy = narrow_i64 (trans.disp ().y (), "instance dy");
  result.ax = result.columns == 1 ? 0 :
    narrow_i64 (a.x (), "array ax");
  result.ay = result.columns == 1 ? 0 :
    narrow_i64 (a.y (), "array ay");
  result.bx = result.rows == 1 ? 0 :
    narrow_i64 (b.x (), "array bx");
  result.by = result.rows == 1 ? 0 :
    narrow_i64 (b.y (), "array by");
  if ((result.columns > 1 && result.ax == 0 && result.ay == 0) ||
      (result.rows > 1 && result.bx == 0 && result.by == 0) ||
      ! checked_multiply_u64 (
          result.columns, result.rows, result.occurrences)) {
    throw Via1StackDecline ("hierarchy has a malformed regular array");
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
    throw Via1StackDecline (
      "hierarchy exceeds the qualified recursion depth");
  }
  if (state [cell] == 1) {
    throw Via1StackDecline ("hierarchy contains a cycle");
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
      throw Via1StackDecline (
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
  std::vector<klayout_cuda_spatial_via1_stack_context_v1> &contexts)
{
  std::vector<uint8_t> state (templates.size (), 0);
  std::vector<uint64_t> memo (templates.size (), 0);
  const uint64_t expected = subtree_context_count (
    root, templates, state, memo, max_contexts, 0);
  if (expected > std::numeric_limits<uint32_t>::max ()) {
    throw Via1StackDecline (
      "expanded hierarchy exceeds uint32 context IDs");
  }
  contexts.reserve (size_t (expected));
  contexts.push_back (
    klayout_cuda_spatial_via1_stack_context_v1 { 0, 0, root, 0 });

  for (size_t parent_id = 0; parent_id < contexts.size (); ++parent_id) {
    const klayout_cuda_spatial_via1_stack_context_v1 parent =
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
            klayout_cuda_spatial_via1_stack_context_v1 {
              narrow_i64 (
                shifted.first + parent.tx,
                "world context translation x"),
              narrow_i64 (
                shifted.second + parent.ty,
                "world context translation y"),
              instance->child_cell, transform
            });
        }
      }
    }
  }
  if (contexts.size () != expected) {
    throw Via1StackDecline (
      "expanded hierarchy disagrees with the checked context census");
  }
}

std::pair<int64_t, int64_t> transform_point (
  const klayout_cuda_spatial_via1_stack_context_v1 &context,
  int64_t x, int64_t y)
{
  const std::pair<__int128, __int128> point =
    transform_128 (context.transform_code, x, y);
  return std::make_pair (
    narrow_i64 (point.first + context.tx, "world box x"),
    narrow_i64 (point.second + context.ty, "world box y"));
}

void add_world_bounds (
  LiveScene &scene,
  const klayout_cuda_spatial_via1_stack_context_v1 &context,
  const LocalBounds &bounds)
{
  if (! bounds.valid) {
    return;
  }
  const int64_t xs [4] =
    { bounds.left, bounds.left, bounds.right, bounds.right };
  const int64_t ys [4] =
    { bounds.bottom, bounds.top, bounds.bottom, bounds.top };
  int64_t left = std::numeric_limits<int64_t>::max ();
  int64_t bottom = std::numeric_limits<int64_t>::max ();
  int64_t right = std::numeric_limits<int64_t>::min ();
  int64_t top = std::numeric_limits<int64_t>::min ();
  for (size_t corner = 0; corner < 4; ++corner) {
    const std::pair<int64_t, int64_t> point =
      transform_point (context, xs [corner], ys [corner]);
    left = std::min (left, point.first);
    bottom = std::min (bottom, point.second);
    right = std::max (right, point.first);
    top = std::max (top, point.second);
  }
  if (! scene.have_scene_box) {
    scene.scene_left = left;
    scene.scene_bottom = bottom;
    scene.scene_right = right;
    scene.scene_top = top;
    scene.have_scene_box = true;
  } else {
    scene.scene_left = std::min (scene.scene_left, left);
    scene.scene_bottom = std::min (scene.scene_bottom, bottom);
    scene.scene_right = std::max (scene.scene_right, right);
    scene.scene_top = std::max (scene.scene_top, top);
  }
}

void add_context_operand (
  uint32_t context_id, uint32_t local_count,
  std::vector<uint32_t> &context_list, std::vector<uint64_t> &offsets,
  uint64_t &flat_count, const char *what)
{
  if (! local_count) {
    return;
  }
  context_list.push_back (context_id);
  offsets.push_back (flat_count);
  if (! checked_add_u64 (flat_count, local_count, flat_count) ||
      flat_count > std::numeric_limits<uint32_t>::max ()) {
    throw Via1StackDecline (
      std::string ("logical ") + what + " box count exceeds uint32");
  }
}

void derive_context_lists_and_scene_box (LiveScene &scene)
{
  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const klayout_cuda_spatial_via1_stack_context_v1 &context =
      scene.contexts [context_id];
    const klayout_cuda_spatial_via1_stack_cell_v1 &cell =
      scene.cells [context.cell_id];
    add_context_operand (
      uint32_t (context_id), cell.metal1_box_count,
      scene.metal1_contexts, scene.metal1_offsets,
      scene.flat_metal1_boxes, "M1");
    add_context_operand (
      uint32_t (context_id), cell.via1_box_count,
      scene.via1_contexts, scene.via1_offsets,
      scene.flat_via1_boxes, "VIA1");
    add_context_operand (
      uint32_t (context_id), cell.metal2_box_count,
      scene.metal2_contexts, scene.metal2_offsets,
      scene.flat_metal2_boxes, "M2");
    add_world_bounds (scene, context, scene.bounds [context.cell_id]);
  }

  if (! scene.have_scene_box || ! scene.flat_metal1_boxes ||
      ! scene.flat_via1_boxes || ! scene.flat_metal2_boxes) {
    throw Via1StackDecline (
      "qualified VIA1-stack scene has an empty operand");
  }
}

LiveScene serialize_live_scene (
  const db::DeepLayer &metal1, const db::DeepLayer &via1,
  const db::DeepLayer &metal2, uint64_t max_contexts)
{
  const db::Layout &layout = metal1.layout ();
  const db::cell_index_type top = metal1.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  metal1.initial_cell ().collect_called_cells (reachable);
  if (reachable.empty () ||
      reachable.size () > std::numeric_limits<uint32_t>::max ()) {
    throw Via1StackDecline (
      "reachable hierarchy has an invalid cell count");
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
    throw Via1StackDecline (
      "initial cell is absent from the hierarchy census");
  }

  LiveScene scene;
  scene.cells.resize (reachable.size ());
  scene.bounds.resize (reachable.size ());
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

    klayout_cuda_spatial_via1_stack_cell_v1 record;
    std::memset (&record, 0, sizeof (record));
    append_cell_layer (
      cell, metal1.layer (), false, scene.boxes, scene.bounds [cell_id],
      record.metal1_box_begin, record.metal1_box_count);
    append_cell_layer (
      cell, via1.layer (), true, scene.boxes, scene.bounds [cell_id],
      record.via1_box_begin, record.via1_box_count);
    append_cell_layer (
      cell, metal2.layer (), false, scene.boxes, scene.bounds [cell_id],
      record.metal2_box_begin, record.metal2_box_count);
    scene.cells [cell_id] = record;
  }

  expand_contexts (
    root->second, templates, max_contexts, scene.contexts);
  derive_context_lists_and_scene_box (scene);
  return scene;
}

bool eligible (
  const db::DeepLayer &metal1, const db::DeepLayer &via1,
  const db::DeepLayer &metal2, bool contact_mode)
{
  const db::LayerProperties &metal1_properties =
    metal1.layout ().get_properties (metal1.layer ());
  const db::LayerProperties &via1_properties =
    via1.layout ().get_properties (via1.layer ());
  const db::LayerProperties &metal2_properties =
    metal2.layout ().get_properties (metal2.layer ());
  const bool layer_contract = contact_mode
    ? metal1.layer () != via1.layer () &&
      metal1.layer () == metal2.layer () &&
      metal1_properties.log_equal (db::LayerProperties (11, 0)) &&
      via1_properties.log_equal (db::LayerProperties (10, 0)) &&
      metal2_properties.log_equal (db::LayerProperties (11, 0))
    : metal1.layer () != via1.layer () &&
      metal1.layer () != metal2.layer () &&
      via1.layer () != metal2.layer () &&
      metal1_properties.log_equal (db::LayerProperties (11, 0)) &&
      via1_properties.log_equal (db::LayerProperties (12, 0)) &&
      metal2_properties.log_equal (db::LayerProperties (13, 0));
  return
    metal1.store () == via1.store () &&
    metal1.store () == metal2.store () &&
    &metal1.layout () == &via1.layout () &&
    &metal1.layout () == &metal2.layout () &&
    metal1.layout_index () == via1.layout_index () &&
    metal1.layout_index () == metal2.layout_index () &&
    metal1.initial_cell ().cell_index () ==
      via1.initial_cell ().cell_index () &&
    metal1.initial_cell ().cell_index () ==
      metal2.initial_cell ().cell_index () &&
    metal1.breakout_cells () == 0 &&
    via1.breakout_cells () == 0 &&
    metal2.breakout_cells () == 0 &&
    layer_contract &&
    metal1.layout ().dbu () == 0.0005;
}

} // anonymous namespace

static bool cuda_stack_try_empty_impl (
  const db::DeepLayer &raw_metal1, const db::DeepLayer &raw_via1,
  const db::DeepLayer &raw_metal2, bool contact_mode)
{
  const bool telemetry =
    env_enabled (
      contact_mode
        ? "KLAYOUT_CUDA_M1_CONTACT_TELEMETRY"
        : "KLAYOUT_CUDA_VIA1_STACK_TELEMETRY");
  const char *label =
    contact_mode ? "CUDA M1 contact live lowering:"
                 : "CUDA VIA1 stack live lowering:";
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    const bool requested = contact_mode
      ? db::cuda_spatial_m1_contact_requested ()
      : db::cuda_spatial_via1_stack_requested ();
    if (! requested ||
        ! eligible (raw_metal1, raw_via1, raw_metal2, contact_mode)) {
      return false;
    }

    const uint64_t max_contexts = env_u64 (
      contact_mode
        ? "KLAYOUT_CUDA_M1_CONTACT_MAX_CONTEXTS"
        : "KLAYOUT_CUDA_VIA1_STACK_MAX_CONTEXTS",
      default_max_contexts);
    const uint64_t max_grid_cells = env_u64 (
      contact_mode
        ? "KLAYOUT_CUDA_M1_CONTACT_MAX_GRID_CELLS"
        : "KLAYOUT_CUDA_VIA1_STACK_MAX_GRID_CELLS",
      default_max_grid_cells);
    const uint64_t max_metal_memberships = env_u64 (
      contact_mode
        ? "KLAYOUT_CUDA_M1_CONTACT_MAX_METAL_MEMBERSHIPS"
        : "KLAYOUT_CUDA_VIA1_STACK_MAX_METAL_MEMBERSHIPS",
      default_max_metal_memberships);
    const uint64_t max_via_memberships = env_u64 (
      contact_mode
        ? "KLAYOUT_CUDA_M1_CONTACT_MAX_CUT_MEMBERSHIPS"
        : "KLAYOUT_CUDA_VIA1_STACK_MAX_VIA_MEMBERSHIPS",
      default_max_via_memberships);
    const uint64_t max_pair_work = env_u64 (
      contact_mode
        ? "KLAYOUT_CUDA_M1_CONTACT_MAX_PAIR_WORK"
        : "KLAYOUT_CUDA_VIA1_STACK_MAX_PAIR_WORK",
      default_max_pair_work);
    if (! max_contexts || ! max_grid_cells || ! max_metal_memberships ||
        ! max_via_memberships || ! max_pair_work) {
      throw Via1StackDecline ("a VIA1-stack capacity is zero");
    }

    LiveScene scene = serialize_live_scene (
      raw_metal1, raw_via1, raw_metal2, max_contexts);
    klayout_cuda_spatial_via1_stack_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_VIA1_STACK_QUALIFIED_OPTIONS;
    request.requested_mask = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES;
    request.dbu_per_micron = qualified_dbu_per_micron;
    request.device = env_device ();
    request.enclosure_distance = qualified_enclosure_distance;
    request.cut_width = qualified_cut_size;
    request.cut_height = qualified_cut_size;
    request.spacing_distance = qualified_spacing_distance;
    request.grid_cell_size = qualified_grid_cell;
    request.contexts = scene.contexts.data ();
    request.context_count = scene.contexts.size ();
    request.metal1_contexts = scene.metal1_contexts.data ();
    request.metal1_context_count = scene.metal1_contexts.size ();
    request.metal1_offsets = scene.metal1_offsets.data ();
    request.metal1_offset_count = scene.metal1_offsets.size ();
    request.via1_contexts = scene.via1_contexts.data ();
    request.via1_context_count = scene.via1_contexts.size ();
    request.via1_offsets = scene.via1_offsets.data ();
    request.via1_offset_count = scene.via1_offsets.size ();
    request.metal2_contexts = scene.metal2_contexts.data ();
    request.metal2_context_count = scene.metal2_contexts.size ();
    request.metal2_offsets = scene.metal2_offsets.data ();
    request.metal2_offset_count = scene.metal2_offsets.size ();
    request.cells = scene.cells.data ();
    request.cell_count = scene.cells.size ();
    request.boxes = scene.boxes.data ();
    request.box_count = scene.boxes.size ();
    request.flat_metal1_box_count = scene.flat_metal1_boxes;
    request.flat_via1_box_count = scene.flat_via1_boxes;
    request.flat_metal2_box_count = scene.flat_metal2_boxes;
    request.scene_left = scene.scene_left;
    request.scene_bottom = scene.scene_bottom;
    request.scene_right = scene.scene_right;
    request.scene_top = scene.scene_top;
    request.max_contexts = max_contexts;
    request.max_grid_cells = max_grid_cells;
    request.max_metal_memberships = max_metal_memberships;
    request.max_via_memberships = max_via_memberships;
    request.max_pair_work = max_pair_work;
    std::array<uint8_t, 32> digest;
    if (! db::cuda_via1_stack_digest::request_digest (request, digest)) {
      throw Via1StackDecline (
        "unable to digest the live VIA1-stack request");
    }
    std::copy (digest.begin (), digest.end (), request.scene_digest);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const double lower_ms =
      std::chrono::duration<double, std::milli> (call_begin - begin).count ();
    const db::CudaVia1StackAttempt attempt =
      db::cuda_spatial_try_via1_stack_empty (request);
    const std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now ();
    const double call_ms =
      std::chrono::duration<double, std::milli> (end - call_begin).count ();
    const double live_total_ms =
      std::chrono::duration<double, std::milli> (end - begin).count ();
    if (telemetry) {
      tl::info << label
               << " contexts=" << request.context_count
               << " cells=" << request.cell_count
               << " stored_boxes=" << request.box_count
               << " m1_boxes=" << request.flat_metal1_box_count
               << " vias=" << request.flat_via1_box_count
               << " m2_boxes=" << request.flat_metal2_box_count
               << " lower_ms=" << lower_ms
               << " call_ms=" << call_ms
               << " live_total_ms=" << live_total_ms;
    }
    return
      attempt.disposition == db::CudaVia1StackAttempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << label
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << label
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

bool cuda_via1_stack_try_empty (
  const db::DeepLayer &raw_metal1, const db::DeepLayer &raw_via1,
  const db::DeepLayer &raw_metal2)
{
  return cuda_stack_try_empty_impl (
    raw_metal1, raw_via1, raw_metal2, false);
}

bool cuda_m1_contact_try_empty (
  const db::DeepLayer &raw_metal1, const db::DeepLayer &raw_contact)
{
  return cuda_stack_try_empty_impl (
    raw_metal1, raw_contact, raw_metal1, true);
}

} // namespace db
