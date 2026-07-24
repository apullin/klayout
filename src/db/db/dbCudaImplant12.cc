/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaImplant12.h"

#include "dbArray.h"
#include "dbCell.h"
#include "dbCudaImplant12Digest.h"
#include "dbCudaSpatialBackend.h"
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

const uint32_t scene_format_version = 1;
const uint32_t qualified_dbu_per_micron = 2000;
const int64_t qualified_implant1_distance = 140;
const int64_t qualified_implant2_distance = 50;
const int64_t qualified_grid_cell = 2000;
const unsigned maximum_hierarchy_depth = 1024;
const int64_t accepted_coordinate_magnitude = INT64_C (1000000000000);

const uint64_t default_max_contexts = UINT64_C (4000000);
const uint64_t default_max_grid_cells = UINT64_C (16000000);
const uint64_t default_max_implant_memberships = UINT64_C (300000000);
const uint64_t default_max_query_visits = UINT64_C (2000000000000);
const uint64_t default_max_candidate_work = UINT64_C (2000000000000);
const uint64_t default_max_flat_polygons = UINT64_C (100000000);
const uint64_t default_max_flat_contours = UINT64_C (100000000);
const uint64_t default_max_flat_edges = UINT64_C (1000000000);

class Implant12Decline
  : public std::runtime_error
{
public:
  explicit Implant12Decline (const std::string &message)
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
    throw Implant12Decline (std::string (what) + " overflows signed int64");
  }
  const int64_t result = int64_t (value);
  if (result < -accepted_coordinate_magnitude ||
      result > accepted_coordinate_magnitude) {
    throw Implant12Decline (
      std::string (what) + " exceeds the qualified coordinate domain");
  }
  return result;
}

uint64_t vector_size_u64 (size_t size, const char *what)
{
  if (size > std::numeric_limits<uint64_t>::max ()) {
    throw Implant12Decline (
      std::string (what) + " cannot be represented by uint64");
  }
  return uint64_t (size);
}

void require_room (
  uint64_t current, uint64_t additional, uint64_t maximum,
  const char *what)
{
  uint64_t total = 0;
  if (! checked_add_u64 (current, additional, total) || total > maximum) {
    throw Implant12Decline (
      std::string (what) + " exceeds the configured capacity");
  }
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
    throw Implant12Decline ("invalid orthogonal transform code");
  }
  const Matrix &matrix = transforms [code];
  return std::make_pair (
    __int128 (matrix.xx) * x + __int128 (matrix.xy) * y,
    __int128 (matrix.yx) * x + __int128 (matrix.yy) * y);
}

uint32_t compose_transform (uint32_t outer, uint32_t inner)
{
  if (outer >= 8 || inner >= 8) {
    throw Implant12Decline (
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
  throw Implant12Decline (
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

  void add (int64_t x, int64_t y)
  {
    if (! valid) {
      left = right = x;
      bottom = top = y;
      valid = true;
    } else {
      left = std::min (left, x);
      bottom = std::min (bottom, y);
      right = std::max (right, x);
      top = std::max (top, y);
    }
  }
};

bool positive_collinear_overlap (
  const klayout_cuda_spatial_implant12_edge_v1 &a,
  const klayout_cuda_spatial_implant12_edge_v1 &b)
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
  const klayout_cuda_spatial_implant12_edge_v1 &a,
  const klayout_cuda_spatial_implant12_edge_v1 &b)
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
  const klayout_cuda_spatial_implant12_edge_v1 &horizontal =
    a.y1 == a.y2 ? a : b;
  const klayout_cuda_spatial_implant12_edge_v1 &vertical =
    a.y1 == a.y2 ? b : a;
  return
    std::min (horizontal.x1, horizontal.x2) <= vertical.x1 &&
    vertical.x1 <= std::max (horizontal.x1, horizontal.x2) &&
    std::min (vertical.y1, vertical.y2) <= horizontal.y1 &&
    horizontal.y1 <= std::max (vertical.y1, vertical.y2);
}

void append_polygon (
  const db::Shape &shape, uint32_t polygon_id, LocalBounds *bounds,
  uint64_t max_stored_contours, uint64_t max_stored_edges,
  CudaImplant12Scene &scene)
{
  if (shape.prop_id () != 0) {
    throw Implant12Decline ("operand polygon has properties");
  }
  if (! shape.is_box () && ! shape.is_polygon ()) {
    throw Implant12Decline ("operand layer contains a non-polygon shape");
  }

  db::Polygon polygon;
  if (! shape.polygon (polygon) || polygon.holes () != 0) {
    throw Implant12Decline (
      "operand polygon is malformed or has holes");
  }

  std::vector<klayout_cuda_spatial_implant12_edge_v1> contour;
  std::set<std::pair<int64_t, int64_t> > vertices;
  __int128 twice_area = 0;
  for (db::Polygon::polygon_edge_iterator edge = polygon.begin_edge ();
       ! edge.at_end (); ++edge) {
    const int64_t x1 = narrow_i64 ((*edge).p1 ().x (), "polygon x1");
    const int64_t y1 = narrow_i64 ((*edge).p1 ().y (), "polygon y1");
    const int64_t x2 = narrow_i64 ((*edge).p2 ().x (), "polygon x2");
    const int64_t y2 = narrow_i64 ((*edge).p2 ().y (), "polygon y2");
    if ((x1 == x2 && y1 == y2) || ! (x1 == x2 || y1 == y2)) {
      throw Implant12Decline (
        "operand polygon has a degenerate or non-Manhattan edge");
    }
    if (! vertices.insert (std::make_pair (x1, y1)).second) {
      throw Implant12Decline ("operand polygon repeats a contour vertex");
    }
    contour.push_back (
      klayout_cuda_spatial_implant12_edge_v1 { x1, y1, x2, y2 });
    twice_area += __int128 (x1) * y2 - __int128 (x2) * y1;
    if (bounds) {
      bounds->add (x1, y1);
      bounds->add (x2, y2);
    }
  }
  if (contour.size () < 4 || twice_area >= 0 ||
      contour.size () > std::numeric_limits<uint32_t>::max ()) {
    throw Implant12Decline (
      "operand polygon is too small or not clockwise");
  }
  for (size_t i = 0; i < contour.size (); ++i) {
    const size_t following = (i + 1) % contour.size ();
    if (contour [i].x2 != contour [following].x1 ||
        contour [i].y2 != contour [following].y1) {
      throw Implant12Decline ("operand polygon contour is open");
    }
    for (size_t j = i + 1; j < contour.size (); ++j) {
      if (! manhattan_segments_intersect (contour [i], contour [j])) {
        continue;
      }
      const bool adjacent =
        j == i + 1 || (i == 0 && j + 1 == contour.size ());
      if (! adjacent ||
          positive_collinear_overlap (contour [i], contour [j])) {
        throw Implant12Decline (
          "operand polygon contour self-intersects");
      }
    }
  }

  const uint64_t stored_contours =
    vector_size_u64 (scene.contours.size (), "stored contour count");
  const uint64_t stored_edges =
    vector_size_u64 (scene.edges.size (), "stored edge count");
  require_room (
    stored_contours, 1, max_stored_contours, "stored contour count");
  require_room (
    stored_edges, uint64_t (contour.size ()), max_stored_edges,
    "stored edge count");
  if (scene.contours.size () == scene.contours.max_size () ||
      contour.size () > scene.edges.max_size () - scene.edges.size ()) {
    throw Implant12Decline (
      "host vector capacity cannot represent the IMPLANT scene");
  }

  scene.contours.push_back (
    klayout_cuda_spatial_implant12_contour_v1 {
      stored_edges, polygon_id, 0, uint32_t (contour.size ()),
      KLAYOUT_CUDA_SPATIAL_IMPLANT12_HULL
    });
  scene.edges.insert (scene.edges.end (), contour.begin (), contour.end ());
}

void append_cell_layer (
  const db::Cell &cell, unsigned int layer, uint32_t domain,
  uint64_t max_stored_contours, uint64_t max_stored_edges,
  CudaImplant12Scene &scene,
  klayout_cuda_spatial_implant12_domain_span_v1 &span,
  LocalBounds *bounds)
{
  if (domain >= KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT) {
    throw Implant12Decline ("invalid serialized geometry domain");
  }
  span.contour_begin =
    vector_size_u64 (scene.contours.size (), "cell contour begin");
  span.edge_begin =
    vector_size_u64 (scene.edges.size (), "cell edge begin");
  span.reserved0 = 0;

  uint32_t polygon_id = 0;
  const db::Shapes &shapes = cell.shapes (layer);
  for (db::Shapes::shape_iterator shape =
         shapes.begin (db::ShapeIterator::All);
       ! shape.at_end (); ++shape) {
    if (polygon_id == std::numeric_limits<uint32_t>::max ()) {
      throw Implant12Decline (
        "per-cell operand polygon count exceeds uint32");
    }
    append_polygon (
      *shape, polygon_id, bounds, max_stored_contours, max_stored_edges,
      scene);
    ++polygon_id;
  }

  const uint64_t contour_count =
    vector_size_u64 (scene.contours.size (), "stored contour count") -
    span.contour_begin;
  const uint64_t edge_count =
    vector_size_u64 (scene.edges.size (), "stored edge count") -
    span.edge_begin;
  if (contour_count != polygon_id ||
      contour_count > std::numeric_limits<uint32_t>::max () ||
      edge_count > std::numeric_limits<uint32_t>::max ()) {
    throw Implant12Decline (
      "per-cell operand census exceeds the first scene format");
  }
  span.polygon_count = polygon_id;
  span.contour_count = uint32_t (contour_count);
  span.edge_count = uint32_t (edge_count);
}

InstanceTemplate make_instance (
  const db::Instance &instance,
  const std::map<db::cell_index_type, uint32_t> &dense_cells)
{
  if (instance.prop_id () != 0 || instance.is_complex ()) {
    throw Implant12Decline (
      "hierarchy has an instance property or complex transform");
  }

  const std::map<db::cell_index_type, uint32_t>::const_iterator child =
    dense_cells.find (instance.cell_index ());
  if (child == dense_cells.end ()) {
    throw Implant12Decline (
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
    throw Implant12Decline (
      "hierarchy has an irregular instance array");
  }
  if (delegate == 0) {
    a = db::Vector ();
    b = db::Vector ();
    na = nb = 1;
  }
  if (na == 0 || nb == 0 ||
      na > std::numeric_limits<uint32_t>::max () ||
      nb > std::numeric_limits<uint32_t>::max ()) {
    throw Implant12Decline ("hierarchy has an invalid array dimension");
  }

  const db::Trans &trans = instance.front ();
  const int transform_code = trans.rot ();
  if (transform_code < 0 || transform_code >= 8) {
    throw Implant12Decline (
      "hierarchy has an invalid orthogonal transform");
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
    throw Implant12Decline ("hierarchy has a malformed regular array");
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
    throw Implant12Decline (
      "hierarchy exceeds the qualified recursion depth");
  }
  if (cell >= cells.size ()) {
    throw Implant12Decline (
      "hierarchy references an invalid dense cell");
  }
  if (state [cell] == 1) {
    throw Implant12Decline ("hierarchy contains a cycle");
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
      throw Implant12Decline (
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
  std::vector<klayout_cuda_spatial_implant12_context_v1> &contexts)
{
  std::vector<uint8_t> state (templates.size (), 0);
  std::vector<uint64_t> memo (templates.size (), 0);
  const uint64_t expected = subtree_context_count (
    root, templates, state, memo, max_contexts, 0);
  if (expected > std::numeric_limits<uint32_t>::max () ||
      expected > contexts.max_size ()) {
    throw Implant12Decline (
      "expanded hierarchy cannot be represented by context IDs");
  }

  contexts.reserve (size_t (expected));
  contexts.push_back (
    klayout_cuda_spatial_implant12_context_v1 { 0, 0, root, 0 });
  for (size_t parent_id = 0; parent_id < contexts.size (); ++parent_id) {
    const klayout_cuda_spatial_implant12_context_v1 parent =
      contexts [parent_id];
    if (parent.cell_id >= templates.size ()) {
      throw Implant12Decline (
        "expanded context references an invalid cell");
    }
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
            klayout_cuda_spatial_implant12_context_v1 {
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
    throw Implant12Decline (
      "expanded hierarchy disagrees with the checked context census");
  }
}

std::pair<int64_t, int64_t> transform_point (
  const klayout_cuda_spatial_implant12_context_v1 &context,
  int64_t x, int64_t y)
{
  const std::pair<__int128, __int128> point =
    transform_128 (context.transform_code, x, y);
  return std::make_pair (
    narrow_i64 (point.first + context.tx, "world polygon x"),
    narrow_i64 (point.second + context.ty, "world polygon y"));
}

void add_world_implant_bounds (
  CudaImplant12Scene &scene,
  const klayout_cuda_spatial_implant12_context_v1 &context,
  const LocalBounds &bounds)
{
  if (! bounds.valid) {
    return;
  }
  const int64_t xs [2] = { bounds.left, bounds.right };
  const int64_t ys [2] = { bounds.bottom, bounds.top };
  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      const std::pair<int64_t, int64_t> point =
        transform_point (context, xs [xi], ys [yi]);
      if (! scene.have_implant_bounds) {
        scene.implant_left = scene.implant_right = point.first;
        scene.implant_bottom = scene.implant_top = point.second;
        scene.have_implant_bounds = true;
      } else {
        scene.implant_left = std::min (scene.implant_left, point.first);
        scene.implant_bottom = std::min (scene.implant_bottom, point.second);
        scene.implant_right = std::max (scene.implant_right, point.first);
        scene.implant_top = std::max (scene.implant_top, point.second);
      }
    }
  }
}

void add_flat_count (
  uint64_t local, uint64_t maximum, uint64_t &flat, const char *what)
{
  if (! checked_add_u64 (flat, local, flat) || flat > maximum) {
    throw Implant12Decline (
      std::string ("flattened ") + what +
      " count exceeds the configured capacity");
  }
}

void derive_context_lists_and_bounds (
  CudaImplant12Scene &scene,
  const std::vector<LocalBounds> &implant_bounds,
  uint64_t max_flat_polygons, uint64_t max_flat_contours,
  uint64_t max_flat_edges)
{
  scene.implant_edge_offsets.push_back (0);
  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const klayout_cuda_spatial_implant12_context_v1 &context =
      scene.contexts [context_id];
    if (context.cell_id >= scene.cells.size ()) {
      throw Implant12Decline (
        "context references an invalid dense cell");
    }
    const klayout_cuda_spatial_implant12_cell_v1 &cell =
      scene.cells [context.cell_id];
    for (uint32_t domain = 0;
         domain < KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT; ++domain) {
      const klayout_cuda_spatial_implant12_domain_span_v1 &span =
        cell.domains [domain];
      if (! span.polygon_count) {
        if (span.contour_count || span.edge_count) {
          throw Implant12Decline (
            "empty domain span has a nonempty contour census");
        }
        continue;
      }
      if (context_id > std::numeric_limits<uint32_t>::max ()) {
        throw Implant12Decline ("nonempty context ID exceeds uint32");
      }
      if (! span.contour_count || ! span.edge_count) {
        throw Implant12Decline (
          "nonempty domain span has an empty contour census");
      }
      add_flat_count (
        span.polygon_count, max_flat_polygons,
        scene.flat_polygons [domain], "polygon");
      add_flat_count (
        span.contour_count, max_flat_contours,
        scene.flat_contours [domain], "contour");
      add_flat_count (
        span.edge_count, max_flat_edges,
        scene.flat_edges [domain], "edge");

      if (domain == KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN) {
        scene.implant_contexts.push_back (uint32_t (context_id));
        scene.implant_edge_offsets.push_back (scene.flat_edges [domain]);
        add_world_implant_bounds (
          scene, context, implant_bounds [context.cell_id]);
      } else if (domain == KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN) {
        scene.gate_contexts.push_back (uint32_t (context_id));
      } else {
        scene.contact_contexts.push_back (uint32_t (context_id));
      }
    }
  }

  uint64_t total_polygons = 0, total_contours = 0, total_edges = 0;
  for (uint32_t domain = 0;
       domain < KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT; ++domain) {
    if (! checked_add_u64 (
          total_polygons, scene.flat_polygons [domain], total_polygons) ||
        ! checked_add_u64 (
          total_contours, scene.flat_contours [domain], total_contours) ||
        ! checked_add_u64 (
          total_edges, scene.flat_edges [domain], total_edges)) {
      throw Implant12Decline ("aggregate flattened scene census overflow");
    }
  }
  if (total_polygons > max_flat_polygons ||
      total_contours > max_flat_contours ||
      total_edges > max_flat_edges) {
    throw Implant12Decline (
      "aggregate flattened scene exceeds the configured capacity");
  }
  if (! scene.have_implant_bounds ||
      scene.implant_contexts.empty () ||
      scene.gate_contexts.empty () ||
      scene.contact_contexts.empty () ||
      scene.flat_edges [KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN] >
        std::numeric_limits<uint32_t>::max () ||
      scene.implant_edge_offsets.size () !=
        scene.implant_contexts.size () + 1) {
    throw Implant12Decline (
      "qualified IMPLANT scene is empty or exceeds device edge IDs");
  }
}

void validate_inputs (
  const db::DeepLayer &implant, const db::DeepLayer &gate,
  const db::DeepLayer &contact, const CudaImplant12BuildSpec &spec,
  const CudaImplant12SceneLimits &limits)
{
  if (! spec.implant_is_exact_merged ||
      ! spec.gate_is_raw || ! spec.contact_is_raw) {
    throw Implant12Decline (
      "integration did not assert exact merged/raw operand provenance");
  }
  if (! limits.max_contexts || ! limits.max_flat_polygons ||
      ! limits.max_flat_contours || ! limits.max_flat_edges ||
      limits.max_contexts > std::numeric_limits<uint32_t>::max ()) {
    throw Implant12Decline (
      "an IMPLANT host-scene capacity is zero or exceeds context IDs");
  }
  if (implant.store () != gate.store () ||
      implant.store () != contact.store () ||
      &implant.layout () != &gate.layout () ||
      &implant.layout () != &contact.layout () ||
      implant.layout_index () != gate.layout_index () ||
      implant.layout_index () != contact.layout_index () ||
      implant.initial_cell ().cell_index () !=
        gate.initial_cell ().cell_index () ||
      implant.initial_cell ().cell_index () !=
        contact.initial_cell ().cell_index ()) {
    throw Implant12Decline (
      "IMPLANT operands do not share one store, layout and top cell");
  }
  if (implant.breakout_cells () != 0 ||
      gate.breakout_cells () != 0 ||
      contact.breakout_cells () != 0) {
    throw Implant12Decline ("IMPLANT scene has hierarchy breakout cells");
  }
  if (implant.layer () == gate.layer () ||
      implant.layer () == contact.layer () ||
      gate.layer () == contact.layer () ||
      implant.layer () >= implant.layout ().layers () ||
      gate.layer () >= gate.layout ().layers () ||
      contact.layer () >= contact.layout ().layers ()) {
    throw Implant12Decline (
      "IMPLANT operands do not have three valid distinct layers");
  }
  const db::LayerProperties &implant_properties =
    implant.layout ().get_properties (implant.layer ());
  const db::LayerProperties &gate_properties =
    gate.layout ().get_properties (gate.layer ());
  const db::LayerProperties &contact_properties =
    contact.layout ().get_properties (contact.layer ());
  if (! implant_properties.log_equal (db::LayerProperties ()) ||
      ! gate_properties.log_equal (db::LayerProperties ()) ||
      ! contact_properties.log_equal (db::LayerProperties (10, 0))) {
    throw Implant12Decline (
      "operands do not match derived IMPLANT/GATE and FreePDK45 CONTACT 10/0");
  }
  if (implant.layout ().dbu () != 0.0005) {
    throw Implant12Decline (
      "IMPLANT scene DBU is not the qualified 0.5 nm");
  }
}

CudaImplant12Scene serialize_live_scene (
  const db::DeepLayer &implant, const db::DeepLayer &gate,
  const db::DeepLayer &contact, uint64_t max_contexts,
  uint64_t max_flat_polygons, uint64_t max_flat_contours,
  uint64_t max_flat_edges)
{
  const db::Layout &layout = implant.layout ();
  const db::cell_index_type top = implant.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  implant.initial_cell ().collect_called_cells (reachable);
  if (reachable.empty () ||
      reachable.size () > std::numeric_limits<uint32_t>::max () ||
      reachable.size () > max_contexts) {
    throw Implant12Decline (
      "reachable hierarchy has an invalid or over-capacity cell count");
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
    throw Implant12Decline (
      "initial cell is absent from the hierarchy census");
  }

  CudaImplant12Scene scene;
  scene.root_cell = root->second;
  scene.cells.resize (reachable.size ());
  std::vector<LocalBounds> implant_bounds (reachable.size ());
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

    klayout_cuda_spatial_implant12_cell_v1 record;
    std::memset (&record, 0, sizeof (record));
    record.source_cell_index = uint64_t (*source);
    append_cell_layer (
      cell, implant.layer (),
      KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN,
      max_flat_contours, max_flat_edges, scene,
      record.domains [KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN],
      &implant_bounds [cell_id]);
    append_cell_layer (
      cell, gate.layer (), KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN,
      max_flat_contours, max_flat_edges, scene,
      record.domains [KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN], 0);
    append_cell_layer (
      cell, contact.layer (), KLAYOUT_CUDA_SPATIAL_IMPLANT12_CONTACT_DOMAIN,
      max_flat_contours, max_flat_edges, scene,
      record.domains [KLAYOUT_CUDA_SPATIAL_IMPLANT12_CONTACT_DOMAIN], 0);
    scene.cells [cell_id] = record;
  }

  if (scene.cells.size () > std::numeric_limits<uint32_t>::max () ||
      scene.contours.size () > std::numeric_limits<uint32_t>::max () ||
      scene.edges.size () > std::numeric_limits<uint32_t>::max ()) {
    throw Implant12Decline (
      "stored scene exceeds first-format device IDs");
  }
  expand_contexts (
    scene.root_cell, templates, max_contexts, scene.contexts);
  derive_context_lists_and_bounds (
    scene, implant_bounds,
    max_flat_polygons, max_flat_contours, max_flat_edges);
  return scene;
}

void validate_record_layouts ()
{
  static_assert (
    std::is_standard_layout<
      klayout_cuda_spatial_implant12_context_v1>::value,
    "IMPLANT context ABI must be standard-layout");
  static_assert (
    std::is_standard_layout<
      klayout_cuda_spatial_implant12_cell_v1>::value,
    "IMPLANT cell ABI must be standard-layout");
  static_assert (
    std::is_standard_layout<
      klayout_cuda_spatial_implant12_contour_v1>::value,
    "IMPLANT contour ABI must be standard-layout");
  static_assert (
    std::is_standard_layout<
      klayout_cuda_spatial_implant12_edge_v1>::value,
    "IMPLANT edge ABI must be standard-layout");
  static_assert (
    sizeof (klayout_cuda_spatial_implant12_context_v1) == 24,
    "unexpected IMPLANT context ABI size");
  static_assert (
    sizeof (klayout_cuda_spatial_implant12_domain_span_v1) == 32,
    "unexpected IMPLANT domain-span ABI size");
  static_assert (
    sizeof (klayout_cuda_spatial_implant12_cell_v1) == 104,
    "unexpected IMPLANT cell ABI size");
  static_assert (
    sizeof (klayout_cuda_spatial_implant12_contour_v1) == 24,
    "unexpected IMPLANT contour ABI size");
  static_assert (
    sizeof (klayout_cuda_spatial_implant12_edge_v1) == 32,
    "unexpected IMPLANT edge ABI size");
}

void set_reason (std::string *reason, const char *message)
{
  if (! reason) {
    return;
  }
  try {
    *reason = message ? message : "unknown IMPLANT exception";
  } catch (...) {
    //  Diagnostics cannot turn a fail-closed decline into an exception.
  }
}

} // anonymous namespace

CudaImplant12SceneLimits::CudaImplant12SceneLimits ()
  : max_contexts (default_max_contexts),
    max_flat_polygons (default_max_flat_polygons),
    max_flat_contours (default_max_flat_contours),
    max_flat_edges (default_max_flat_edges)
{
  //  nothing yet
}

CudaImplant12BuildSpec::CudaImplant12BuildSpec ()
  : implant_is_exact_merged (false), gate_is_raw (false),
    contact_is_raw (false)
{
  //  nothing yet
}

CudaImplant12Scene::CudaImplant12Scene ()
  : root_cell (0), have_implant_bounds (false),
    implant_left (0), implant_bottom (0), implant_right (0), implant_top (0)
{
  std::fill (
    flat_polygons,
    flat_polygons + KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT,
    uint64_t (0));
  std::fill (
    flat_contours,
    flat_contours + KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT,
    uint64_t (0));
  std::fill (
    flat_edges,
    flat_edges + KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT,
    uint64_t (0));
}

void CudaImplant12Scene::swap (CudaImplant12Scene &other) noexcept
{
  contexts.swap (other.contexts);
  implant_contexts.swap (other.implant_contexts);
  implant_edge_offsets.swap (other.implant_edge_offsets);
  gate_contexts.swap (other.gate_contexts);
  contact_contexts.swap (other.contact_contexts);
  cells.swap (other.cells);
  contours.swap (other.contours);
  edges.swap (other.edges);
  std::swap (root_cell, other.root_cell);
  for (uint32_t domain = 0;
       domain < KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT; ++domain) {
    std::swap (flat_polygons [domain], other.flat_polygons [domain]);
    std::swap (flat_contours [domain], other.flat_contours [domain]);
    std::swap (flat_edges [domain], other.flat_edges [domain]);
  }
  std::swap (have_implant_bounds, other.have_implant_bounds);
  std::swap (implant_left, other.implant_left);
  std::swap (implant_bottom, other.implant_bottom);
  std::swap (implant_right, other.implant_right);
  std::swap (implant_top, other.implant_top);
}

bool cuda_implant12_build_scene (
  const db::DeepLayer &merged_implant, const db::DeepLayer &raw_gate,
  const db::DeepLayer &raw_contact, const CudaImplant12BuildSpec &spec,
  const CudaImplant12SceneLimits &limits, CudaImplant12Scene &scene,
  std::string *decline_reason)
{
  try {
    validate_record_layouts ();
    validate_inputs (
      merged_implant, raw_gate, raw_contact, spec, limits);
    CudaImplant12Scene built = serialize_live_scene (
      merged_implant, raw_gate, raw_contact, limits.max_contexts,
      limits.max_flat_polygons, limits.max_flat_contours,
      limits.max_flat_edges);
    scene.swap (built);
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (
      decline_reason, "unknown IMPLANT scene-lowering exception");
  }
  return false;
}

bool cuda_implant12_try_empty (
  const db::DeepLayer &merged_implant, const db::DeepLayer &raw_gate,
  const db::DeepLayer &raw_contact)
{
  const bool telemetry = env_enabled ("KLAYOUT_CUDA_IMPLANT12_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    validate_record_layouts ();
    if (! db::cuda_spatial_implant12_requested ()) {
      return false;
    }

    const uint64_t max_contexts = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_CONTEXTS", default_max_contexts);
    const uint64_t max_grid_cells = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_GRID_CELLS", default_max_grid_cells);
    const uint64_t max_implant_memberships = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_IMPLANT_MEMBERSHIPS",
      default_max_implant_memberships);
    const uint64_t max_gate_query_visits = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_GATE_QUERY_VISITS",
      default_max_query_visits);
    const uint64_t max_gate_candidate_work = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_GATE_CANDIDATE_WORK",
      default_max_candidate_work);
    const uint64_t max_contact_query_visits = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_CONTACT_QUERY_VISITS",
      default_max_query_visits);
    const uint64_t max_contact_candidate_work = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_CONTACT_CANDIDATE_WORK",
      default_max_candidate_work);
    const uint64_t max_flat_polygons = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_FLAT_POLYGONS",
      default_max_flat_polygons);
    const uint64_t max_flat_contours = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_FLAT_CONTOURS",
      default_max_flat_contours);
    const uint64_t max_flat_edges = env_u64 (
      "KLAYOUT_CUDA_IMPLANT12_MAX_FLAT_EDGES", default_max_flat_edges);
    if (! max_contexts || ! max_grid_cells ||
        ! max_implant_memberships || ! max_gate_query_visits ||
        ! max_gate_candidate_work || ! max_contact_query_visits ||
        ! max_contact_candidate_work || ! max_flat_polygons ||
        ! max_flat_contours || ! max_flat_edges ||
        max_contexts > std::numeric_limits<uint32_t>::max () ||
        max_grid_cells > std::numeric_limits<uint32_t>::max () ||
        max_implant_memberships > std::numeric_limits<uint32_t>::max ()) {
      throw Implant12Decline (
        "an IMPLANT capacity is zero or exceeds first-format device IDs");
    }

    CudaImplant12BuildSpec spec;
    spec.implant_is_exact_merged = true;
    spec.gate_is_raw = true;
    spec.contact_is_raw = true;
    CudaImplant12SceneLimits limits;
    limits.max_contexts = max_contexts;
    limits.max_flat_polygons = max_flat_polygons;
    limits.max_flat_contours = max_flat_contours;
    limits.max_flat_edges = max_flat_edges;
    CudaImplant12Scene scene;
    std::string decline_reason;
    if (! cuda_implant12_build_scene (
          merged_implant, raw_gate, raw_contact, spec, limits,
          scene, &decline_reason)) {
      throw Implant12Decline (
        decline_reason.empty ()
          ? "unable to build the live IMPLANT scene"
          : decline_reason);
    }

    klayout_cuda_spatial_implant12_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_SUPERSET_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_IMPLANT12_QUALIFIED_OPTIONS;
    request.format_version = scene_format_version;
    request.dbu_per_micron = qualified_dbu_per_micron;
    request.root_cell = scene.root_cell;
    request.requested_mask = KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES;
    request.device = env_device ();
    request.implant1_distance = qualified_implant1_distance;
    request.implant2_distance = qualified_implant2_distance;
    request.grid_cell_size = qualified_grid_cell;
    request.contexts = scene.contexts.data ();
    request.context_count = scene.contexts.size ();
    request.context_record_bytes =
      sizeof (klayout_cuda_spatial_implant12_context_v1);
    request.implant_contexts = scene.implant_contexts.data ();
    request.implant_context_count = scene.implant_contexts.size ();
    request.implant_edge_offsets = scene.implant_edge_offsets.data ();
    request.implant_edge_offset_count = scene.implant_edge_offsets.size ();
    request.gate_contexts = scene.gate_contexts.data ();
    request.gate_context_count = scene.gate_contexts.size ();
    request.contact_contexts = scene.contact_contexts.data ();
    request.contact_context_count = scene.contact_contexts.size ();
    request.cells = scene.cells.data ();
    request.cell_count = scene.cells.size ();
    request.cell_record_bytes =
      sizeof (klayout_cuda_spatial_implant12_cell_v1);
    request.contours = scene.contours.data ();
    request.contour_count = scene.contours.size ();
    request.contour_record_bytes =
      sizeof (klayout_cuda_spatial_implant12_contour_v1);
    request.edges = scene.edges.data ();
    request.edge_count = scene.edges.size ();
    request.edge_record_bytes =
      sizeof (klayout_cuda_spatial_implant12_edge_v1);
    request.flat_implant_polygon_count =
      scene.flat_polygons [KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN];
    request.flat_gate_polygon_count =
      scene.flat_polygons [KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN];
    request.flat_contact_polygon_count =
      scene.flat_polygons [KLAYOUT_CUDA_SPATIAL_IMPLANT12_CONTACT_DOMAIN];
    request.flat_implant_contour_count =
      scene.flat_contours [KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN];
    request.flat_gate_contour_count =
      scene.flat_contours [KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN];
    request.flat_contact_contour_count =
      scene.flat_contours [KLAYOUT_CUDA_SPATIAL_IMPLANT12_CONTACT_DOMAIN];
    request.flat_implant_edge_count =
      scene.flat_edges [KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN];
    request.flat_gate_edge_count =
      scene.flat_edges [KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN];
    request.flat_contact_edge_count =
      scene.flat_edges [KLAYOUT_CUDA_SPATIAL_IMPLANT12_CONTACT_DOMAIN];
    request.implant_left = scene.implant_left;
    request.implant_bottom = scene.implant_bottom;
    request.implant_right = scene.implant_right;
    request.implant_top = scene.implant_top;
    request.max_contexts = max_contexts;
    request.max_grid_cells = max_grid_cells;
    request.max_implant_memberships = max_implant_memberships;
    request.max_gate_query_visits = max_gate_query_visits;
    request.max_gate_candidate_work = max_gate_candidate_work;
    request.max_contact_query_visits = max_contact_query_visits;
    request.max_contact_candidate_work = max_contact_candidate_work;
    request.max_flat_polygons = max_flat_polygons;
    request.max_flat_contours = max_flat_contours;
    request.max_flat_edges = max_flat_edges;

    std::array<uint8_t, 32> digest;
    if (! db::cuda_implant12_digest::request_digest (request, digest)) {
      throw Implant12Decline (
        "unable to digest the live IMPLANT request");
    }
    std::copy (digest.begin (), digest.end (), request.scene_digest);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const double lower_ms =
      std::chrono::duration<double, std::milli> (call_begin - begin).count ();
    const db::CudaImplant12Attempt attempt =
      db::cuda_spatial_try_implant12_empty (request);
    const std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << "CUDA IMPLANT.1/.2 live lowering:"
               << " contexts=" << request.context_count
               << " cells=" << request.cell_count
               << " stored_contours=" << request.contour_count
               << " stored_edges=" << request.edge_count
               << " implant_edges=" << request.flat_implant_edge_count
               << " gate_edges=" << request.flat_gate_edge_count
               << " contact_edges=" << request.flat_contact_edge_count
               << " lower_ms=" << lower_ms
               << " call_ms="
               << std::chrono::duration<double, std::milli> (
                    end - call_begin).count ()
               << " live_total_ms="
               << std::chrono::duration<double, std::milli> (
                    end - begin).count ();
    }
    return
      attempt.disposition == db::CudaImplant12Attempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << "CUDA IMPLANT.1/.2 live lowering:"
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << "CUDA IMPLANT.1/.2 live lowering:"
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

} // namespace db
