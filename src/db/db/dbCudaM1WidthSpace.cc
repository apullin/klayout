/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaM1WidthSpace.h"

#include "dbArray.h"
#include "dbCell.h"
#include "dbCudaActive3Digest.h"
#include "dbCudaManhattanContour.h"
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
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
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
const int64_t qualified_m1_distance = 130;
const int64_t qualified_m2_distance = 140;
const unsigned maximum_hierarchy_depth = 1024;
const int64_t accepted_coordinate_magnitude = INT64_C (1000000000000);

const uint64_t default_max_cells = UINT64_C (4000000);
const uint64_t default_max_contexts = UINT64_C (4000000);
const uint64_t default_max_stored_polygons = UINT64_C (20000000);
const uint64_t default_max_stored_edges = UINT64_C (100000000);
const uint64_t default_max_flat_polygons = UINT64_C (200000000);
const uint64_t default_max_flat_edges = UINT64_C (800000000);
const uint64_t default_max_grid_cells = UINT64_C (16000000);
const uint64_t default_max_memberships = UINT64_C (120000000);
const uint64_t default_max_pair_work = UINT64_C (2000000000);
const uint64_t backend_max_flat_polygons = UINT64_C (50000000);
const uint64_t backend_max_flat_edges = UINT64_C (100000000);
const int64_t qualified_grid_cell = 512;

const uint64_t m1_morph_max_contexts = UINT64_C (4000000);
const uint64_t m1_morph_max_rectangles = UINT64_C (64000000);
const uint64_t m1_morph_max_x_slabs = UINT64_C (32000000);
const uint64_t m1_morph_max_union_memberships = UINT64_C (256000000);
const uint64_t m1_morph_max_union_events = UINT64_C (512000000);
const uint64_t m1_morph_max_union_raw_segments = UINT64_C (256000000);
const uint64_t m1_morph_max_union_segments = UINT64_C (64000000);
const uint64_t m1_morph_max_slabs_per_rectangle = UINT64_C (256);
const uint64_t m1_morph_max_output_slabs = UINT64_C (1000000);
const uint64_t m1_morph_max_output_intervals = UINT64_C (64000000);
const uint64_t m1_morph_max_raw_boundary_segments = UINT64_C (256000000);
const uint64_t m1_morph_max_boundary_segments = UINT64_C (64000000);
const uint64_t m1_morph_max_source_visits_per_pass =
  UINT64_C (12000000000);
const uint64_t m1_morph_max_source_visits_per_band = UINT64_C (4000000);
const uint64_t m1_morph_max_long_segments = UINT64_C (4096);
const uint64_t m1_morph_max_active_slabs = UINT64_C (128);

struct RawManhattanProfile
{
  int layer;
  int datatype;
  const char *label;
  char digest_magic [8];
};

const RawManhattanProfile raw_m2_profile = {
  13, 0, "raw M2",
  { 'K', 'M', '2', 'R', 'A', 'W', '0', '1' }
};

const RawManhattanProfile raw_m1_profile = {
  11, 0, "raw M1",
  { 'K', 'M', '1', 'R', 'A', 'W', '0', '1' }
};

const RawManhattanProfile raw_active_profile = {
  1, 0, "raw ACTIVE",
  { 'K', 'A', 'R', 'A', 'W', '0', '0', '1' }
};

const RawManhattanProfile raw_contact_profile = {
  10, 0, "raw CONTACT",
  { 'K', 'C', 'R', 'A', 'W', '0', '0', '1' }
};

const RawManhattanProfile raw_well_union_profile = {
  3, 0, "combined raw WELL",
  { 'K', 'W', 'R', 'W', 'L', '0', '0', '1' }
};

class M1WidthSpaceDecline
  : public std::runtime_error
{
public:
  explicit M1WidthSpaceDecline (const std::string &message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

bool scene_distance_is_qualified (int64_t width_distance,
                                  int64_t spacing_distance)
{
  return width_distance == spacing_distance &&
         (width_distance == qualified_m1_distance ||
          width_distance == qualified_m2_distance);
}

uint64_t env_u64 (const char *name, uint64_t default_value)
{
  const char *value = std::getenv (name);
  if (! value || ! *value) {
    return default_value;
  }

  errno = 0;
  char *end = 0;
  const unsigned long long parsed = std::strtoull (value, &end, 0);
  if (errno != 0 || ! end || *end != 0) {
    return default_value;
  }
  return static_cast<uint64_t> (parsed);
}

bool env_enabled (const char *name)
{
  const char *value = std::getenv (name);
  return value && *value && std::strcmp (value, "0") != 0 &&
         std::strcmp (value, "false") != 0 &&
         std::strcmp (value, "off") != 0;
}

std::string digest_hex (const std::array<uint8_t, 32> &digest)
{
  static const char alphabet [] = "0123456789abcdef";
  std::string result (64, '0');
  for (size_t index = 0; index < digest.size (); ++index) {
    result [index * 2] = alphabet [digest [index] >> 4];
    result [index * 2 + 1] = alphabet [digest [index] & 0xf];
  }
  return result;
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
    throw M1WidthSpaceDecline (
      std::string (what) + " overflows signed int64");
  }
  const int64_t result = int64_t (value);
  if (result < -accepted_coordinate_magnitude ||
      result > accepted_coordinate_magnitude) {
    throw M1WidthSpaceDecline (
      std::string (what) + " exceeds the qualified coordinate domain");
  }
  return result;
}

uint64_t vector_size_u64 (size_t size, const char *what)
{
  if (size > std::numeric_limits<uint64_t>::max ()) {
    throw M1WidthSpaceDecline (
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
    throw M1WidthSpaceDecline (
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
    throw M1WidthSpaceDecline ("invalid orthogonal transform code");
  }
  const Matrix &matrix = transforms [code];
  return std::make_pair (
    __int128 (matrix.xx) * x + __int128 (matrix.xy) * y,
    __int128 (matrix.yx) * x + __int128 (matrix.yy) * y);
}

uint32_t compose_transform (uint32_t outer, uint32_t inner)
{
  if (outer >= 8 || inner >= 8) {
    throw M1WidthSpaceDecline (
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
  throw M1WidthSpaceDecline (
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

template <class Scene>
void append_polygon (
  const db::Shape &shape, uint32_t polygon_id,
  const CudaM1WidthSpaceSceneLimits &limits,
  Scene &scene,
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> &contour_cache)
{
  if (shape.prop_id () != 0) {
    throw M1WidthSpaceDecline ("Manhattan scene polygon has properties");
  }
  if (! shape.is_box () && ! shape.is_polygon ()) {
    throw M1WidthSpaceDecline (
      "Manhattan scene layer contains a non-polygon shape");
  }

  db::Polygon polygon;
  if (! shape.polygon (polygon)) {
    throw M1WidthSpaceDecline ("Manhattan scene polygon is malformed");
  }
  if (polygon.holes () != 0) {
    throw M1WidthSpaceDecline (
      "Manhattan scene polygon has holes, which are outside the first "
      "scene format");
  }

  std::vector<CudaM1WidthSpaceEdge> contour;
  __int128 twice_area = 0;
  bool have_bounds = false;
  int64_t left = 0, bottom = 0, right = 0, top = 0;
  for (db::Polygon::polygon_edge_iterator edge = polygon.begin_edge ();
       ! edge.at_end (); ++edge) {
    const int64_t x1 = narrow_i64 ((*edge).p1 ().x (), "polygon x1");
    const int64_t y1 = narrow_i64 ((*edge).p1 ().y (), "polygon y1");
    const int64_t x2 = narrow_i64 ((*edge).p2 ().x (), "polygon x2");
    const int64_t y2 = narrow_i64 ((*edge).p2 ().y (), "polygon y2");
    if ((x1 == x2 && y1 == y2) || ! (x1 == x2 || y1 == y2)) {
      throw M1WidthSpaceDecline (
        "Manhattan scene polygon has a degenerate or non-Manhattan edge");
    }
    contour.push_back (CudaM1WidthSpaceEdge { x1, y1, x2, y2 });
    twice_area += __int128 (x1) * y2 - __int128 (x2) * y1;
    if (! have_bounds) {
      left = std::min (x1, x2);
      bottom = std::min (y1, y2);
      right = std::max (x1, x2);
      top = std::max (y1, y2);
      have_bounds = true;
    } else {
      left = std::min (left, std::min (x1, x2));
      bottom = std::min (bottom, std::min (y1, y2));
      right = std::max (right, std::max (x1, x2));
      top = std::max (top, std::max (y1, y2));
    }
  }
  if (contour.size () < 4 || twice_area >= 0 || ! have_bounds ||
      left >= right || bottom >= top) {
    throw M1WidthSpaceDecline (
      "Manhattan scene polygon is too small, empty, or not clockwise");
  }
  const cuda_manhattan_contour::ValidationResult contour_result =
    contour_cache.validate_contour (contour);
  if (contour_result ==
      cuda_manhattan_contour::ValidationResult::OpenContour) {
    throw M1WidthSpaceDecline ("Manhattan scene polygon contour is open");
  }
  if (contour_result !=
      cuda_manhattan_contour::ValidationResult::Valid) {
    throw M1WidthSpaceDecline (
      "Manhattan scene polygon contour self-intersects");
  }
  if (contour.size () > std::numeric_limits<uint32_t>::max ()) {
    throw M1WidthSpaceDecline (
      "one Manhattan scene polygon has more than uint32 contour edges");
  }

  const uint64_t stored_polygons =
    vector_size_u64 (scene.polygons.size (), "stored polygon count");
  const uint64_t stored_edges =
    vector_size_u64 (scene.edges.size (), "stored edge count");
  require_room (
    stored_polygons, 1, limits.max_stored_polygons, "stored polygon count");
  require_room (
    stored_edges, uint64_t (contour.size ()), limits.max_stored_edges,
    "stored edge count");
  if (scene.polygons.size () == scene.polygons.max_size () ||
      contour.size () > scene.edges.max_size () - scene.edges.size ()) {
    throw M1WidthSpaceDecline (
      "host vector capacity cannot represent the Manhattan scene");
  }

  CudaM1WidthSpacePolygon record;
  record.edge_begin = stored_edges;
  record.left = left;
  record.bottom = bottom;
  record.right = right;
  record.top = top;
  record.polygon_id = polygon_id;
  record.edge_count = uint32_t (contour.size ());
  scene.polygons.push_back (record);
  scene.edges.insert (scene.edges.end (), contour.begin (), contour.end ());
}

template <class Scene>
void append_cell_layer (
  const db::Cell &cell, unsigned int layer, uint64_t source_cell_index,
  const CudaM1WidthSpaceSceneLimits &limits,
  Scene &scene, CudaM1WidthSpaceCell &record,
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> &contour_cache)
{
  record.source_cell_index = source_cell_index;
  record.polygon_begin =
    vector_size_u64 (scene.polygons.size (), "cell polygon begin");
  record.edge_begin =
    vector_size_u64 (scene.edges.size (), "cell edge begin");

  uint32_t polygon_id = 0;
  const db::Shapes &shapes = cell.shapes (layer);
  for (db::Shapes::shape_iterator shape =
         shapes.begin (db::ShapeIterator::All);
       ! shape.at_end (); ++shape) {
    //  Region polygon semantics ignore physical-layer labels.  Keep that
    //  filtering explicit at the raw scene boundary while continuing to
    //  decline every other unsupported non-polygon record.
    if (shape->is_text ()) {
      continue;
    }
    if (polygon_id == std::numeric_limits<uint32_t>::max ()) {
      throw M1WidthSpaceDecline (
        "per-cell Manhattan polygon count exceeds uint32");
    }
    append_polygon (
      *shape, polygon_id, limits, scene, contour_cache);
    ++polygon_id;
  }

  const uint64_t polygon_count =
    vector_size_u64 (scene.polygons.size (), "stored polygon count") -
    record.polygon_begin;
  const uint64_t edge_count =
    vector_size_u64 (scene.edges.size (), "stored edge count") -
    record.edge_begin;
  if (polygon_count > std::numeric_limits<uint32_t>::max () ||
      edge_count > std::numeric_limits<uint32_t>::max ()) {
    throw M1WidthSpaceDecline (
      "per-cell Manhattan polygon or edge count exceeds uint32");
  }
  record.polygon_count = uint32_t (polygon_count);
  record.edge_count = uint32_t (edge_count);
}

template <class Scene>
void append_cell_layer_pair (
  const db::Cell &cell, unsigned int first_layer,
  unsigned int second_layer, uint64_t source_cell_index,
  const CudaM1WidthSpaceSceneLimits &limits,
  Scene &scene, CudaM1WidthSpaceCell &record,
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> &contour_cache)
{
  record.source_cell_index = source_cell_index;
  record.polygon_begin =
    vector_size_u64 (scene.polygons.size (), "cell polygon begin");
  record.edge_begin =
    vector_size_u64 (scene.edges.size (), "cell edge begin");

  uint32_t polygon_id = 0;
  const unsigned int layers [2] = { first_layer, second_layer };
  for (size_t layer_index = 0; layer_index < 2; ++layer_index) {
    const db::Shapes &shapes = cell.shapes (layers [layer_index]);
    for (db::Shapes::shape_iterator shape =
           shapes.begin (db::ShapeIterator::All);
         ! shape.at_end (); ++shape) {
      if (shape->is_text ()) {
        continue;
      }
      if (polygon_id == std::numeric_limits<uint32_t>::max ()) {
        throw M1WidthSpaceDecline (
          "per-cell combined raw-WELL polygon count exceeds uint32");
      }
      append_polygon (
        *shape, polygon_id, limits, scene, contour_cache);
      ++polygon_id;
    }
  }

  const uint64_t polygon_count =
    vector_size_u64 (scene.polygons.size (), "stored polygon count") -
    record.polygon_begin;
  const uint64_t edge_count =
    vector_size_u64 (scene.edges.size (), "stored edge count") -
    record.edge_begin;
  if (polygon_count > std::numeric_limits<uint32_t>::max () ||
      edge_count > std::numeric_limits<uint32_t>::max ()) {
    throw M1WidthSpaceDecline (
      "per-cell combined raw-WELL polygon or edge count exceeds uint32");
  }
  record.polygon_count = uint32_t (polygon_count);
  record.edge_count = uint32_t (edge_count);
}

InstanceTemplate make_instance (
  const db::Instance &instance,
  const std::map<db::cell_index_type, uint32_t> &dense_cells)
{
  if (instance.prop_id () != 0 || instance.is_complex ()) {
    throw M1WidthSpaceDecline (
      "hierarchy has an instance property or complex transform");
  }

  const std::map<db::cell_index_type, uint32_t>::const_iterator child =
    dense_cells.find (instance.cell_index ());
  if (child == dense_cells.end ()) {
    throw M1WidthSpaceDecline (
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
    throw M1WidthSpaceDecline (
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
    throw M1WidthSpaceDecline (
      "hierarchy has an invalid array dimension");
  }

  const db::Trans &trans = instance.front ();
  const int transform_code = trans.rot ();
  if (transform_code < 0 || transform_code >= 8) {
    throw M1WidthSpaceDecline (
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
    throw M1WidthSpaceDecline (
      "hierarchy has a malformed regular array");
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
    throw M1WidthSpaceDecline (
      "hierarchy exceeds the qualified recursion depth");
  }
  if (cell >= cells.size ()) {
    throw M1WidthSpaceDecline (
      "hierarchy references an invalid dense cell");
  }
  if (state [cell] == 1) {
    throw M1WidthSpaceDecline ("hierarchy contains a cycle");
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
      throw M1WidthSpaceDecline (
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
  std::vector<CudaM1WidthSpaceContext> &contexts)
{
  std::vector<uint8_t> state (templates.size (), 0);
  std::vector<uint64_t> memo (templates.size (), 0);
  const uint64_t expected = subtree_context_count (
    root, templates, state, memo, max_contexts, 0);
  if (expected > std::numeric_limits<uint32_t>::max () ||
      expected > contexts.max_size ()) {
    throw M1WidthSpaceDecline (
      "expanded hierarchy cannot be represented by context IDs");
  }

  contexts.reserve (size_t (expected));
  contexts.push_back (CudaM1WidthSpaceContext { 0, 0, root, 0 });
  for (size_t parent_id = 0; parent_id < contexts.size (); ++parent_id) {
    const CudaM1WidthSpaceContext parent = contexts [parent_id];
    if (parent.cell_id >= templates.size ()) {
      throw M1WidthSpaceDecline (
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
            CudaM1WidthSpaceContext {
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
    throw M1WidthSpaceDecline (
      "expanded hierarchy disagrees with the checked context census");
  }
}

std::pair<int64_t, int64_t> transform_point (
  const CudaM1WidthSpaceContext &context, int64_t x, int64_t y)
{
  const std::pair<__int128, __int128> point =
    transform_128 (context.transform_code, x, y);
  return std::make_pair (
    narrow_i64 (point.first + context.tx, "world polygon x"),
    narrow_i64 (point.second + context.ty, "world polygon y"));
}

void add_world_point (
  int64_t x, int64_t y, bool &have_bounds,
  int64_t &left, int64_t &bottom, int64_t &right, int64_t &top)
{
  if (! have_bounds) {
    left = right = x;
    bottom = top = y;
    have_bounds = true;
  } else {
    left = std::min (left, x);
    bottom = std::min (bottom, y);
    right = std::max (right, x);
    top = std::max (top, y);
  }
}

void add_world_polygon_bounds (
  const CudaM1WidthSpaceContext &context,
  const CudaM1WidthSpacePolygon &polygon, bool &have_bounds,
  int64_t &left, int64_t &bottom, int64_t &right, int64_t &top)
{
  const int64_t xs [2] = { polygon.left, polygon.right };
  const int64_t ys [2] = { polygon.bottom, polygon.top };
  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      const std::pair<int64_t, int64_t> point =
        transform_point (context, xs [xi], ys [yi]);
      add_world_point (
        point.first, point.second, have_bounds,
        left, bottom, right, top);
    }
  }
}

template <class Scene>
void derive_context_lists_and_bounds (
  const CudaM1WidthSpaceSceneLimits &limits,
  Scene &scene)
{
  bool have_bounds = false;
  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const CudaM1WidthSpaceContext &context = scene.contexts [context_id];
    if (context.cell_id >= scene.cells.size ()) {
      throw M1WidthSpaceDecline (
        "context references an invalid dense cell");
    }
    const CudaM1WidthSpaceCell &cell = scene.cells [context.cell_id];
    if (! cell.polygon_count) {
      continue;
    }
    if (context_id > std::numeric_limits<uint32_t>::max ()) {
      throw M1WidthSpaceDecline (
        "nonempty context ID exceeds uint32");
    }

    scene.metal_contexts.push_back (uint32_t (context_id));
    scene.context_polygon_offsets.push_back (scene.flat_polygon_count);
    scene.context_edge_offsets.push_back (scene.flat_edge_count);
    if (! checked_add_u64 (
          scene.flat_polygon_count, cell.polygon_count,
          scene.flat_polygon_count) ||
        scene.flat_polygon_count > limits.max_flat_polygons) {
      throw M1WidthSpaceDecline (
        "flattened Manhattan polygon count exceeds the configured capacity");
    }
    if (! checked_add_u64 (
          scene.flat_edge_count, cell.edge_count,
          scene.flat_edge_count) ||
        scene.flat_edge_count > limits.max_flat_edges) {
      throw M1WidthSpaceDecline (
        "flattened Manhattan edge count exceeds the configured capacity");
    }

    const uint64_t polygon_end =
      cell.polygon_begin + uint64_t (cell.polygon_count);
    if (polygon_end > scene.polygons.size ()) {
      throw M1WidthSpaceDecline (
        "cell polygon range escapes the serialized array");
    }
    for (uint64_t polygon_id = cell.polygon_begin;
         polygon_id < polygon_end; ++polygon_id) {
      add_world_polygon_bounds (
        context, scene.polygons [size_t (polygon_id)], have_bounds,
        scene.scene_left, scene.scene_bottom,
        scene.scene_right, scene.scene_top);
    }
  }
  if (! have_bounds || ! scene.flat_polygon_count ||
      ! scene.flat_edge_count) {
    throw M1WidthSpaceDecline ("qualified Manhattan scene is empty");
  }
}

bool options_are_qualified (const db::RegionCheckOptions &options)
{
  return
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
    options.zd_mode == db::IncludeZeroDistanceWhenTouching;
}

void validate_scene_limits (
  const CudaM1WidthSpaceSceneLimits &limits)
{
  if (! limits.max_cells || ! limits.max_contexts ||
      ! limits.max_stored_polygons || ! limits.max_stored_edges ||
      ! limits.max_flat_polygons || ! limits.max_flat_edges) {
    throw M1WidthSpaceDecline ("a Manhattan scene capacity is zero");
  }
}

void validate_inputs (
  const db::DeepLayer &width_metal1,
  const db::DeepLayer &spacing_metal1,
  const CudaM1WidthSpaceBuildSpec &spec,
  const CudaM1WidthSpaceSceneLimits &limits)
{
  if (! spec.inputs_are_merged) {
    throw M1WidthSpaceDecline (
      "future integration did not assert merged M1 semantics");
  }
  if (! scene_distance_is_qualified (
        spec.width_distance, spec.spacing_distance) ||
      ! options_are_qualified (spec.width_options) ||
      ! options_are_qualified (spec.spacing_options)) {
    throw M1WidthSpaceDecline (
      "M1 width/spacing distances or options are outside the qualified form");
  }
  validate_scene_limits (limits);
  if (width_metal1.store () != spacing_metal1.store () ||
      &width_metal1.layout () != &spacing_metal1.layout () ||
      width_metal1.layout_index () != spacing_metal1.layout_index () ||
      width_metal1.initial_cell ().cell_index () !=
        spacing_metal1.initial_cell ().cell_index () ||
      width_metal1.layer () != spacing_metal1.layer ()) {
    throw M1WidthSpaceDecline (
      "width and spacing do not use the identical DeepLayer");
  }
  if (width_metal1.breakout_cells () != 0 ||
      spacing_metal1.breakout_cells () != 0) {
    throw M1WidthSpaceDecline (
      "M1 scene has hierarchy breakout cells");
  }
  if (width_metal1.layout ().dbu () != 0.0005) {
    throw M1WidthSpaceDecline (
      "M1 scene DBU is not the qualified 0.5 nm");
  }
}

void validate_raw_manhattan_input (
  const db::DeepLayer &raw_layer,
  const CudaM1WidthSpaceSceneLimits &limits,
  const RawManhattanProfile &profile)
{
  validate_scene_limits (limits);
  if (raw_layer.breakout_cells () != 0) {
    throw M1WidthSpaceDecline (
      std::string (profile.label) +
      " scene has hierarchy breakout cells");
  }
  if (raw_layer.layout ().dbu () != 0.0005) {
    throw M1WidthSpaceDecline (
      std::string (profile.label) +
      " scene DBU is not the qualified 0.5 nm");
  }
  if (! raw_layer.layout ().is_valid_layer (raw_layer.layer ())) {
    throw M1WidthSpaceDecline (
      std::string (profile.label) +
      " scene layer index is not a valid physical layer");
  }
  const db::LayerProperties &properties =
    raw_layer.layout ().get_properties (raw_layer.layer ());
  if (! properties.log_equal (
        db::LayerProperties (profile.layer, profile.datatype))) {
    throw M1WidthSpaceDecline (
      std::string (profile.label) +
      " scene is not physical FreePDK45 layer " +
      std::to_string (profile.layer) + "/" +
      std::to_string (profile.datatype));
  }
}

void validate_raw_well_union_inputs (
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_pwell,
  const CudaM1WidthSpaceSceneLimits &limits)
{
  validate_raw_manhattan_input (
    raw_nwell, limits, raw_well_union_profile);
  static const RawManhattanProfile raw_pwell_profile = {
    2, 0, "raw PWELL",
    { 'K', 'W', 'R', 'W', 'L', '0', '0', '1' }
  };
  validate_raw_manhattan_input (
    raw_pwell, limits, raw_pwell_profile);
  if (raw_nwell.store () != raw_pwell.store () ||
      &raw_nwell.layout () != &raw_pwell.layout () ||
      raw_nwell.layout_index () != raw_pwell.layout_index () ||
      raw_nwell.initial_cell ().cell_index () !=
        raw_pwell.initial_cell ().cell_index () ||
      raw_nwell.layer () == raw_pwell.layer ()) {
    throw M1WidthSpaceDecline (
      "raw NWELL and PWELL do not share one distinct-layer hierarchy");
  }
}

template <class Scene>
Scene serialize_layer_scene (
  const db::DeepLayer &metal1,
  const CudaM1WidthSpaceSceneLimits &limits)
{
  const db::Layout &layout = metal1.layout ();
  const db::cell_index_type top = metal1.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  metal1.initial_cell ().collect_called_cells (reachable);
  if (reachable.empty () ||
      reachable.size () > std::numeric_limits<uint32_t>::max () ||
      reachable.size () > limits.max_cells) {
    throw M1WidthSpaceDecline (
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
    throw M1WidthSpaceDecline (
      "initial cell is absent from the hierarchy census");
  }

  Scene scene;
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> contour_cache;
  scene.root_cell = root->second;
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

    CudaM1WidthSpaceCell record;
    std::memset (&record, 0, sizeof (record));
    append_cell_layer (
      cell, metal1.layer (), uint64_t (*source),
      limits, scene, record, contour_cache);
    scene.cells [cell_id] = record;
  }

  expand_contexts (
    root->second, templates, limits.max_contexts, scene.contexts);
  derive_context_lists_and_bounds (limits, scene);
  return scene;
}

CudaRawManhattanScene serialize_layer_pair_scene (
  const db::DeepLayer &first, const db::DeepLayer &second,
  const CudaM1WidthSpaceSceneLimits &limits)
{
  const db::Layout &layout = first.layout ();
  const db::cell_index_type top = first.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  first.initial_cell ().collect_called_cells (reachable);
  if (reachable.empty () ||
      reachable.size () > std::numeric_limits<uint32_t>::max () ||
      reachable.size () > limits.max_cells) {
    throw M1WidthSpaceDecline (
      "reachable combined raw-WELL hierarchy has an invalid or "
      "over-capacity cell count");
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
    throw M1WidthSpaceDecline (
      "initial cell is absent from the combined raw-WELL hierarchy census");
  }

  CudaRawManhattanScene scene;
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> contour_cache;
  scene.root_cell = root->second;
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

    CudaM1WidthSpaceCell record;
    std::memset (&record, 0, sizeof (record));
    append_cell_layer_pair (
      cell, first.layer (), second.layer (), uint64_t (*source),
      limits, scene, record, contour_cache);
    scene.cells [cell_id] = record;
  }

  expand_contexts (
    root->second, templates, limits.max_contexts, scene.contexts);
  derive_context_lists_and_bounds (limits, scene);
  return scene;
}

CudaM1WidthSpaceScene serialize_scene (
  const db::DeepLayer &metal1,
  const CudaM1WidthSpaceBuildSpec &spec,
  const CudaM1WidthSpaceSceneLimits &limits)
{
  CudaM1WidthSpaceScene scene =
    serialize_layer_scene<CudaM1WidthSpaceScene> (metal1, limits);
  scene.width_distance = spec.width_distance;
  scene.spacing_distance = spec.spacing_distance;
  return scene;
}

bool checked_range (uint64_t begin, uint64_t count, uint64_t size)
{
  uint64_t end = 0;
  return checked_add_u64 (begin, count, end) && end <= size;
}

template <class Scene>
bool geometry_structurally_valid (const Scene &scene)
{
  if (scene.cells.empty () || scene.contexts.empty () ||
      scene.polygons.empty () || scene.edges.empty () ||
      scene.root_cell >= scene.cells.size () ||
      scene.metal_contexts.size () !=
        scene.context_polygon_offsets.size () ||
      scene.metal_contexts.size () != scene.context_edge_offsets.size () ||
      scene.scene_left >= scene.scene_right ||
      scene.scene_bottom >= scene.scene_top) {
    return false;
  }
  const CudaM1WidthSpaceContext &root = scene.contexts.front ();
  if (root.tx != 0 || root.ty != 0 || root.cell_id != scene.root_cell ||
      root.transform_code != 0) {
    return false;
  }

  uint64_t next_polygon = 0;
  uint64_t next_edge = 0;
  std::set<uint64_t> source_cells;
  for (size_t cell_id = 0; cell_id < scene.cells.size (); ++cell_id) {
    const CudaM1WidthSpaceCell &cell = scene.cells [cell_id];
    if (! source_cells.insert (cell.source_cell_index).second ||
        cell.polygon_begin != next_polygon ||
        cell.edge_begin != next_edge ||
        ! checked_range (
          cell.polygon_begin, cell.polygon_count, scene.polygons.size ()) ||
        ! checked_range (
          cell.edge_begin, cell.edge_count, scene.edges.size ())) {
      return false;
    }

    uint64_t cell_edge = cell.edge_begin;
    for (uint32_t local = 0; local < cell.polygon_count; ++local) {
      const CudaM1WidthSpacePolygon &polygon =
        scene.polygons [size_t (cell.polygon_begin + local)];
      if (polygon.polygon_id != local ||
          polygon.edge_begin != cell_edge || polygon.edge_count < 4 ||
          polygon.left >= polygon.right || polygon.bottom >= polygon.top ||
          ! checked_range (
            polygon.edge_begin, polygon.edge_count, scene.edges.size ())) {
        return false;
      }
      uint64_t polygon_end = 0;
      if (! checked_add_u64 (
            polygon.edge_begin, polygon.edge_count, polygon_end)) {
        return false;
      }
      for (uint64_t edge_id = polygon.edge_begin;
           edge_id < polygon_end; ++edge_id) {
        const CudaM1WidthSpaceEdge &edge = scene.edges [size_t (edge_id)];
        if ((edge.x1 == edge.x2 && edge.y1 == edge.y2) ||
            ! (edge.x1 == edge.x2 || edge.y1 == edge.y2)) {
          return false;
        }
        const uint64_t following =
          edge_id + 1 == polygon_end ? polygon.edge_begin : edge_id + 1;
        const CudaM1WidthSpaceEdge &next =
          scene.edges [size_t (following)];
        if (edge.x2 != next.x1 || edge.y2 != next.y1) {
          return false;
        }
      }
      cell_edge = polygon_end;
    }
    uint64_t cell_edge_end = 0;
    if (! checked_add_u64 (
          cell.edge_begin, cell.edge_count, cell_edge_end) ||
        cell_edge != cell_edge_end) {
      return false;
    }
    next_polygon += cell.polygon_count;
    next_edge += cell.edge_count;
  }
  if (next_polygon != scene.polygons.size () ||
      next_edge != scene.edges.size ()) {
    return false;
  }

  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const CudaM1WidthSpaceContext &context = scene.contexts [context_id];
    if (context.cell_id >= scene.cells.size () ||
        context.transform_code >= 8) {
      return false;
    }
  }

  uint64_t flat_polygons = 0;
  uint64_t flat_edges = 0;
  uint32_t previous_context = 0;
  bool have_previous_context = false;
  for (size_t i = 0; i < scene.metal_contexts.size (); ++i) {
    const uint32_t context_id = scene.metal_contexts [i];
    if (context_id >= scene.contexts.size () ||
        (have_previous_context && context_id <= previous_context) ||
        scene.context_polygon_offsets [i] != flat_polygons ||
        scene.context_edge_offsets [i] != flat_edges) {
      return false;
    }
    const CudaM1WidthSpaceCell &cell =
      scene.cells [scene.contexts [context_id].cell_id];
    if (! cell.polygon_count ||
        ! checked_add_u64 (
          flat_polygons, cell.polygon_count, flat_polygons) ||
        ! checked_add_u64 (flat_edges, cell.edge_count, flat_edges)) {
      return false;
    }
    previous_context = context_id;
    have_previous_context = true;
  }
  return
    flat_polygons == scene.flat_polygon_count &&
    flat_edges == scene.flat_edge_count &&
    flat_polygons != 0 && flat_edges != 0;
}

bool structurally_valid (const CudaM1WidthSpaceScene &scene)
{
  return
    scene.format_version == scene_format_version &&
    scene.dbu_per_micron == qualified_dbu_per_micron &&
    scene.reserved == 0 &&
    scene_distance_is_qualified (
      scene.width_distance, scene.spacing_distance) &&
    geometry_structurally_valid (scene);
}

bool structurally_valid (const CudaM2RawManhattanScene &scene)
{
  return
    scene.format_version == scene_format_version &&
    scene.dbu_per_micron == qualified_dbu_per_micron &&
    scene.reserved == 0 &&
    geometry_structurally_valid (scene);
}

class CanonicalDigest
{
public:
  void bytes (const void *data, size_t size)
  {
    m_sha.update (data, size);
  }

  void u32 (uint32_t value)
  {
    uint8_t encoded [4];
    for (unsigned i = 0; i < 4; ++i) {
      encoded [i] = uint8_t (value >> (i * 8));
    }
    bytes (encoded, sizeof (encoded));
  }

  void u64 (uint64_t value)
  {
    uint8_t encoded [8];
    for (unsigned i = 0; i < 8; ++i) {
      encoded [i] = uint8_t (value >> (i * 8));
    }
    bytes (encoded, sizeof (encoded));
  }

  void i64 (int64_t value)
  {
    u64 (static_cast<uint64_t> (value));
  }

  std::array<uint8_t, 32> finish ()
  {
    return m_sha.finish ();
  }

private:
  db::cuda_active3_digest::Sha256 m_sha;
};

template <class Scene>
void digest_geometry_payload (
  CanonicalDigest &sha, const Scene &scene)
{
  sha.u64 (scene.contexts.size ());
  sha.u64 (scene.metal_contexts.size ());
  sha.u64 (scene.cells.size ());
  sha.u64 (scene.polygons.size ());
  sha.u64 (scene.edges.size ());
  sha.u64 (scene.flat_polygon_count);
  sha.u64 (scene.flat_edge_count);
  sha.i64 (scene.scene_left);
  sha.i64 (scene.scene_bottom);
  sha.i64 (scene.scene_right);
  sha.i64 (scene.scene_top);

  for (std::vector<CudaM1WidthSpaceContext>::const_iterator context =
         scene.contexts.begin (); context != scene.contexts.end ();
       ++context) {
    sha.i64 (context->tx);
    sha.i64 (context->ty);
    sha.u32 (context->cell_id);
    sha.u32 (context->transform_code);
  }
  for (size_t i = 0; i < scene.metal_contexts.size (); ++i) {
    sha.u32 (scene.metal_contexts [i]);
    sha.u64 (scene.context_polygon_offsets [i]);
    sha.u64 (scene.context_edge_offsets [i]);
  }
  for (std::vector<CudaM1WidthSpaceCell>::const_iterator cell =
         scene.cells.begin (); cell != scene.cells.end (); ++cell) {
    sha.u64 (cell->source_cell_index);
    sha.u64 (cell->polygon_begin);
    sha.u64 (cell->edge_begin);
    sha.u32 (cell->polygon_count);
    sha.u32 (cell->edge_count);
  }
  for (std::vector<CudaM1WidthSpacePolygon>::const_iterator polygon =
         scene.polygons.begin (); polygon != scene.polygons.end ();
       ++polygon) {
    sha.u64 (polygon->edge_begin);
    sha.i64 (polygon->left);
    sha.i64 (polygon->bottom);
    sha.i64 (polygon->right);
    sha.i64 (polygon->top);
    sha.u32 (polygon->polygon_id);
    sha.u32 (polygon->edge_count);
  }
  for (std::vector<CudaM1WidthSpaceEdge>::const_iterator edge =
         scene.edges.begin (); edge != scene.edges.end (); ++edge) {
    sha.i64 (edge->x1);
    sha.i64 (edge->y1);
    sha.i64 (edge->x2);
    sha.i64 (edge->y2);
  }
}

bool raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  const RawManhattanProfile &profile,
  std::array<uint8_t, 32> &digest)
{
  if (! structurally_valid (scene)) {
    return false;
  }

  CanonicalDigest sha;
  sha.bytes (profile.digest_magic, sizeof (profile.digest_magic));
  sha.u32 (scene.format_version);
  sha.u32 (scene.dbu_per_micron);
  sha.u32 (scene.root_cell);
  sha.u32 (scene.reserved);
  digest_geometry_payload (sha, scene);
  digest = sha.finish ();
  return true;
}

void set_reason (std::string *reason, const char *message)
{
  if (! reason) {
    return;
  }
  try {
    *reason = message ? message : "unknown exception";
  } catch (...) {
    //  Diagnostics cannot turn a fail-closed decline into an exception.
  }
}

bool build_raw_manhattan_scene (
  const db::DeepLayer &raw_layer,
  const CudaM1WidthSpaceSceneLimits &limits,
  const RawManhattanProfile &profile,
  CudaRawManhattanScene &scene,
  std::string *decline_reason)
{
  try {
    validate_raw_manhattan_input (raw_layer, limits, profile);
    CudaRawManhattanScene candidate =
      serialize_layer_scene<CudaRawManhattanScene> (
        raw_layer, limits);
    std::array<uint8_t, 32> digest;
    if (! raw_manhattan_scene_digest (candidate, profile, digest)) {
      throw M1WidthSpaceDecline (
        std::string ("serialized ") + profile.label +
        " scene failed structural digest validation");
    }
    candidate.digest = digest;
    scene.swap (candidate);
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

static_assert (
  std::is_standard_layout<CudaM1WidthSpaceContext>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceContext>::value,
  "M1 context records must remain pointer-free POD");
static_assert (
  std::is_standard_layout<CudaM1WidthSpaceCell>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceCell>::value,
  "M1 cell records must remain pointer-free POD");
static_assert (
  std::is_standard_layout<CudaM1WidthSpacePolygon>::value &&
  std::is_trivially_copyable<CudaM1WidthSpacePolygon>::value,
  "M1 polygon records must remain pointer-free POD");
static_assert (
  std::is_standard_layout<CudaM1WidthSpaceEdge>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceEdge>::value,
  "M1 edge records must remain pointer-free POD");

} // anonymous namespace

CudaM1WidthSpaceSceneLimits::CudaM1WidthSpaceSceneLimits ()
  : max_cells (default_max_cells),
    max_contexts (default_max_contexts),
    max_stored_polygons (default_max_stored_polygons),
    max_stored_edges (default_max_stored_edges),
    max_flat_polygons (default_max_flat_polygons),
    max_flat_edges (default_max_flat_edges)
{
  //  nothing yet
}

CudaM1WidthSpaceBuildSpec::CudaM1WidthSpaceBuildSpec ()
  : width_distance (qualified_m1_distance),
    spacing_distance (qualified_m1_distance),
    width_options (),
    spacing_options (),
    inputs_are_merged (false)
{
  //  nothing yet
}

CudaM1WidthSpaceScene::CudaM1WidthSpaceScene ()
  : format_version (scene_format_version),
    dbu_per_micron (qualified_dbu_per_micron),
    root_cell (0),
    reserved (0),
    width_distance (qualified_m1_distance),
    spacing_distance (qualified_m1_distance),
    flat_polygon_count (0),
    flat_edge_count (0),
    scene_left (0),
    scene_bottom (0),
    scene_right (0),
    scene_top (0),
    contexts (),
    metal_contexts (),
    context_polygon_offsets (),
    context_edge_offsets (),
    cells (),
    polygons (),
    edges (),
    digest ()
{
  //  nothing yet
}

void CudaM1WidthSpaceScene::swap (
  CudaM1WidthSpaceScene &other) noexcept
{
  using std::swap;
  swap (format_version, other.format_version);
  swap (dbu_per_micron, other.dbu_per_micron);
  swap (root_cell, other.root_cell);
  swap (reserved, other.reserved);
  swap (width_distance, other.width_distance);
  swap (spacing_distance, other.spacing_distance);
  swap (flat_polygon_count, other.flat_polygon_count);
  swap (flat_edge_count, other.flat_edge_count);
  swap (scene_left, other.scene_left);
  swap (scene_bottom, other.scene_bottom);
  swap (scene_right, other.scene_right);
  swap (scene_top, other.scene_top);
  contexts.swap (other.contexts);
  metal_contexts.swap (other.metal_contexts);
  context_polygon_offsets.swap (other.context_polygon_offsets);
  context_edge_offsets.swap (other.context_edge_offsets);
  cells.swap (other.cells);
  polygons.swap (other.polygons);
  edges.swap (other.edges);
  digest.swap (other.digest);
}

CudaM2RawManhattanScene::CudaM2RawManhattanScene ()
  : format_version (scene_format_version),
    dbu_per_micron (qualified_dbu_per_micron),
    root_cell (0),
    reserved (0),
    flat_polygon_count (0),
    flat_edge_count (0),
    scene_left (0),
    scene_bottom (0),
    scene_right (0),
    scene_top (0),
    contexts (),
    metal_contexts (),
    context_polygon_offsets (),
    context_edge_offsets (),
    cells (),
    polygons (),
    edges (),
    digest ()
{
  //  nothing yet
}

void CudaM2RawManhattanScene::swap (
  CudaM2RawManhattanScene &other) noexcept
{
  using std::swap;
  swap (format_version, other.format_version);
  swap (dbu_per_micron, other.dbu_per_micron);
  swap (root_cell, other.root_cell);
  swap (reserved, other.reserved);
  swap (flat_polygon_count, other.flat_polygon_count);
  swap (flat_edge_count, other.flat_edge_count);
  swap (scene_left, other.scene_left);
  swap (scene_bottom, other.scene_bottom);
  swap (scene_right, other.scene_right);
  swap (scene_top, other.scene_top);
  contexts.swap (other.contexts);
  metal_contexts.swap (other.metal_contexts);
  context_polygon_offsets.swap (other.context_polygon_offsets);
  context_edge_offsets.swap (other.context_edge_offsets);
  cells.swap (other.cells);
  polygons.swap (other.polygons);
  edges.swap (other.edges);
  digest.swap (other.digest);
}

bool cuda_m1_width_space_scene_digest (
  const CudaM1WidthSpaceScene &scene, std::array<uint8_t, 32> &digest)
{
  try {
    if (! structurally_valid (scene)) {
      return false;
    }

    static const char magic [8] =
      { 'K', 'M', '1', 'W', 'S', '0', '0', '1' };
    CanonicalDigest sha;
    sha.bytes (magic, sizeof (magic));
    sha.u32 (scene.format_version);
    sha.u32 (scene.dbu_per_micron);
    sha.u32 (scene.root_cell);
    sha.u32 (scene.reserved);
    sha.i64 (scene.width_distance);
    sha.i64 (scene.spacing_distance);
    digest_geometry_payload (sha, scene);
    digest = sha.finish ();
    return true;
  } catch (...) {
    return false;
  }
}

bool cuda_m2_raw_manhattan_scene_digest (
  const CudaM2RawManhattanScene &scene,
  std::array<uint8_t, 32> &digest)
{
  try {
    return raw_manhattan_scene_digest (
      scene, raw_m2_profile, digest);
  } catch (...) {
    return false;
  }
}

bool cuda_m1_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest)
{
  try {
    return raw_manhattan_scene_digest (
      scene, raw_m1_profile, digest);
  } catch (...) {
    return false;
  }
}

bool cuda_active_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest)
{
  try {
    return raw_manhattan_scene_digest (
      scene, raw_active_profile, digest);
  } catch (...) {
    return false;
  }
}

bool cuda_contact_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest)
{
  try {
    return raw_manhattan_scene_digest (
      scene, raw_contact_profile, digest);
  } catch (...) {
    return false;
  }
}

bool cuda_well_union_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest)
{
  try {
    return raw_manhattan_scene_digest (
      scene, raw_well_union_profile, digest);
  } catch (...) {
    return false;
  }
}

bool cuda_m1_width_space_build_scene (
  const db::DeepLayer &width_metal1,
  const db::DeepLayer &spacing_metal1,
  const CudaM1WidthSpaceBuildSpec &spec,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaM1WidthSpaceScene &scene,
  std::string *decline_reason)
{
  try {
    validate_inputs (
      width_metal1, spacing_metal1, spec, limits);
    CudaM1WidthSpaceScene candidate =
      serialize_scene (width_metal1, spec, limits);
    std::array<uint8_t, 32> digest;
    if (! cuda_m1_width_space_scene_digest (candidate, digest)) {
      throw M1WidthSpaceDecline (
        "serialized M1 scene failed structural digest validation");
    }
    candidate.digest = digest;
    scene.swap (candidate);
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

bool cuda_m2_raw_manhattan_build_scene (
  const db::DeepLayer &raw_metal2,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaM2RawManhattanScene &scene,
  std::string *decline_reason)
{
  return build_raw_manhattan_scene (
    raw_metal2, limits, raw_m2_profile, scene, decline_reason);
}

bool cuda_m1_raw_manhattan_build_scene (
  const db::DeepLayer &raw_metal1,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason)
{
  return build_raw_manhattan_scene (
    raw_metal1, limits, raw_m1_profile, scene, decline_reason);
}

bool cuda_active_raw_manhattan_build_scene (
  const db::DeepLayer &raw_active,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason)
{
  return build_raw_manhattan_scene (
    raw_active, limits, raw_active_profile, scene, decline_reason);
}

bool cuda_contact_raw_manhattan_build_scene (
  const db::DeepLayer &raw_contact,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason)
{
  return build_raw_manhattan_scene (
    raw_contact, limits, raw_contact_profile, scene, decline_reason);
}

bool cuda_well_union_raw_manhattan_build_scene (
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_pwell,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason)
{
  try {
    validate_raw_well_union_inputs (
      raw_nwell, raw_pwell, limits);
    CudaRawManhattanScene candidate =
      serialize_layer_pair_scene (
        raw_nwell, raw_pwell, limits);
    std::array<uint8_t, 32> digest;
    if (! cuda_well_union_raw_manhattan_scene_digest (
          candidate, digest)) {
      throw M1WidthSpaceDecline (
        "serialized combined raw-WELL scene failed structural digest "
        "validation");
    }
    candidate.digest = digest;
    scene.swap (candidate);
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

bool cuda_m1_5_9_try_empty (const db::DeepLayer &raw_metal1)
{
  const bool telemetry =
    env_enabled ("KLAYOUT_CUDA_M1_5_9_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    // Establish the complete optional capability before paying the
    // multi-million-polygon hierarchy-lowering cost.
    if (! db::cuda_spatial_m1_resident_morphology_requested ()) {
      return false;
    }

    CudaM1WidthSpaceSceneLimits scene_limits;
    scene_limits.max_cells = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_CELLS",
      scene_limits.max_cells);
    scene_limits.max_contexts = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_CONTEXTS",
      m1_morph_max_contexts);
    scene_limits.max_stored_polygons = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_STORED_POLYGONS",
      scene_limits.max_stored_polygons);
    scene_limits.max_stored_edges = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_STORED_EDGES",
      scene_limits.max_stored_edges);
    scene_limits.max_flat_polygons = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_FLAT_POLYGONS",
      m1_morph_max_rectangles);
    scene_limits.max_flat_edges = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_FLAT_EDGES",
      scene_limits.max_flat_edges);

    const uint64_t max_rectangles = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_RECTANGLES",
      m1_morph_max_rectangles);
    const uint64_t max_x_slabs = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_X_SLABS",
      m1_morph_max_x_slabs);
    const uint64_t max_union_memberships = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_UNION_MEMBERSHIPS",
      m1_morph_max_union_memberships);
    const uint64_t max_union_events = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_UNION_EVENTS",
      m1_morph_max_union_events);
    const uint64_t max_union_raw_segments = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_UNION_RAW_SEGMENTS",
      m1_morph_max_union_raw_segments);
    const uint64_t max_union_segments = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_UNION_SEGMENTS",
      m1_morph_max_union_segments);
    const uint64_t max_slabs_per_rectangle = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_SLABS_PER_RECTANGLE",
      m1_morph_max_slabs_per_rectangle);
    const uint64_t max_morph_output_slabs = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_OUTPUT_SLABS",
      m1_morph_max_output_slabs);
    const uint64_t max_morph_output_intervals = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_OUTPUT_INTERVALS",
      m1_morph_max_output_intervals);
    const uint64_t max_morph_raw_boundary_segments = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_RAW_BOUNDARY_SEGMENTS",
      m1_morph_max_raw_boundary_segments);
    const uint64_t max_morph_boundary_segments = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_BOUNDARY_SEGMENTS",
      m1_morph_max_boundary_segments);
    const uint64_t max_morph_source_visits_per_pass = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_SOURCE_VISITS_PER_PASS",
      m1_morph_max_source_visits_per_pass);
    const uint64_t max_morph_source_visits_per_band = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_SOURCE_VISITS_PER_BAND",
      m1_morph_max_source_visits_per_band);
    const uint64_t max_morph_long_segments = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_LONG_SEGMENTS",
      m1_morph_max_long_segments);
    const uint64_t max_morph_active_slabs = env_u64 (
      "KLAYOUT_CUDA_M1_5_9_MAX_MORPH_ACTIVE_SLABS",
      m1_morph_max_active_slabs);
    const uint64_t device =
      env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0);

    if (! scene_limits.max_cells || ! scene_limits.max_contexts ||
        ! scene_limits.max_stored_polygons ||
        ! scene_limits.max_stored_edges ||
        ! scene_limits.max_flat_polygons ||
        ! scene_limits.max_flat_edges ||
        ! max_rectangles || ! max_x_slabs ||
        ! max_union_memberships || ! max_union_events ||
        ! max_union_raw_segments || ! max_union_segments ||
        ! max_slabs_per_rectangle ||
        max_slabs_per_rectangle >
          std::numeric_limits<uint32_t>::max () ||
        ! max_morph_output_slabs ||
        max_morph_output_slabs >
          std::numeric_limits<uint32_t>::max () ||
        ! max_morph_output_intervals ||
        ! max_morph_raw_boundary_segments ||
        ! max_morph_boundary_segments ||
        ! max_morph_source_visits_per_pass ||
        ! max_morph_source_visits_per_band ||
        ! max_morph_long_segments ||
        ! max_morph_active_slabs ||
        max_morph_active_slabs >
          std::numeric_limits<uint32_t>::max () ||
        device > uint64_t (std::numeric_limits<int32_t>::max ())) {
      throw M1WidthSpaceDecline (
        "an M1.5-.9 capacity or device is invalid");
    }

    CudaRawManhattanScene scene;
    std::string reason;
    if (! cuda_m1_raw_manhattan_build_scene (
          raw_metal1, scene_limits, scene, &reason)) {
      throw M1WidthSpaceDecline (
        reason.empty ()
          ? "unable to serialize the qualified raw physical M1 scene"
          : reason);
    }

    klayout_cuda_spatial_m1_resident_morphology_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M15_9_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_M1_MORPH_QUALIFIED_OPTIONS;
    request.format_version = scene.format_version;
    request.dbu_per_micron = scene.dbu_per_micron;
    request.root_cell = scene.root_cell;
    request.device = int32_t (device);
    request.requested_mask =
      KLAYOUT_CUDA_SPATIAL_M1_MORPH_ALL_EMPTY;
    request.contexts = scene.contexts.data ();
    request.context_count = scene.contexts.size ();
    request.context_record_bytes = sizeof (CudaM1WidthSpaceContext);
    request.metal_contexts = scene.metal_contexts.data ();
    request.metal_context_count = scene.metal_contexts.size ();
    request.context_polygon_offsets =
      scene.context_polygon_offsets.data ();
    request.context_polygon_offset_count =
      scene.context_polygon_offsets.size ();
    request.context_edge_offsets =
      scene.context_edge_offsets.data ();
    request.context_edge_offset_count =
      scene.context_edge_offsets.size ();
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
    request.max_contexts = scene_limits.max_contexts;
    request.max_rectangles = max_rectangles;
    request.max_x_slabs = max_x_slabs;
    request.max_union_memberships = max_union_memberships;
    request.max_union_events = max_union_events;
    request.max_union_raw_segments = max_union_raw_segments;
    request.max_union_segments = max_union_segments;
    request.max_slabs_per_rectangle =
      uint32_t (max_slabs_per_rectangle);
    request.max_morph_output_slabs = max_morph_output_slabs;
    request.max_morph_output_intervals =
      max_morph_output_intervals;
    request.max_morph_raw_boundary_segments =
      max_morph_raw_boundary_segments;
    request.max_morph_boundary_segments =
      max_morph_boundary_segments;
    request.max_morph_source_visits_per_pass =
      max_morph_source_visits_per_pass;
    request.max_morph_source_visits_per_band =
      max_morph_source_visits_per_band;
    request.max_morph_long_segments = max_morph_long_segments;
    request.max_morph_active_slabs =
      uint32_t (max_morph_active_slabs);
    std::copy (
      scene.digest.begin (), scene.digest.end (),
      request.scene_digest);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const db::CudaM1ResidentMorphologyAttempt attempt =
      db::cuda_spatial_try_m1_resident_morphology_empty (request);
    const std::chrono::steady_clock::time_point done =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << "CUDA M1.5-.9 exact live lowering:"
               << " outcome="
               << (attempt.disposition ==
                     db::CudaM1ResidentMorphologyAttempt::CertifiedEmpty
                     ? "certified-empty" : "cpu-fallback")
               << " digest=" << digest_hex (scene.digest)
               << " contexts=" << request.context_count
               << " metal_contexts=" << request.metal_context_count
               << " cells=" << request.cell_count
               << " stored_polygons=" << request.polygon_count
               << " stored_edges=" << request.edge_count
               << " flat_polygons=" << request.flat_polygon_count
               << " flat_edges=" << request.flat_edge_count
               << " bounds=" << request.scene_left << ","
               << request.scene_bottom << ","
               << request.scene_right << ","
               << request.scene_top
               << " lower_ms="
               << std::chrono::duration<double, std::milli> (
                    call_begin - begin).count ()
               << " call_ms="
               << std::chrono::duration<double, std::milli> (
                    done - call_begin).count ()
               << " live_total_ms="
               << std::chrono::duration<double, std::milli> (
                    done - begin).count ()
               << " message="
               << (attempt.message.empty () ? "none" : attempt.message);
    }
    return attempt.disposition ==
      db::CudaM1ResidentMorphologyAttempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << "CUDA M1.5-.9 exact live lowering:"
                 << " outcome=cpu-fallback digest=unavailable"
                 << " message=" << ex.what ();
      } catch (...) {
        // Telemetry cannot turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << "CUDA M1.5-.9 exact live lowering:"
                 << " outcome=cpu-fallback digest=unavailable"
                 << " message=unknown exception";
      } catch (...) {
        // Telemetry cannot turn a speculative decline into an error.
      }
    }
  }
  return false;
}

bool cuda_m1_width_space_try_empty (
  const db::DeepLayer &merged_metal1,
  const CudaM1WidthSpaceBuildSpec &spec)
{
  const bool metal1 =
    spec.width_distance == qualified_m1_distance &&
    spec.spacing_distance == qualified_m1_distance;
  const bool metal2 =
    spec.width_distance == qualified_m2_distance &&
    spec.spacing_distance == qualified_m2_distance;
  const bool telemetry = env_enabled (
    metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY"
           : "KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    if (! (metal1 || metal2) ||
        ! (metal2 ? db::cuda_spatial_m2_width_space_requested ()
                  : db::cuda_spatial_m1_width_space_requested ())) {
      return false;
    }

    static_assert (
      std::is_trivially_copyable<CudaM1WidthSpaceContext>::value,
      "M1 context records must be trivially copyable");
    static_assert (
      std::is_trivially_copyable<CudaM1WidthSpaceCell>::value,
      "M1 cell records must be trivially copyable");
    static_assert (
      std::is_trivially_copyable<CudaM1WidthSpacePolygon>::value,
      "M1 polygon records must be trivially copyable");
    static_assert (
      std::is_trivially_copyable<CudaM1WidthSpaceEdge>::value,
      "M1 edge records must be trivially copyable");
    static_assert (
      sizeof (CudaM1WidthSpaceContext) ==
        sizeof (klayout_cuda_spatial_m1_width_space_context_v1) &&
      offsetof (CudaM1WidthSpaceContext, tx) ==
        offsetof (klayout_cuda_spatial_m1_width_space_context_v1, tx) &&
      offsetof (CudaM1WidthSpaceContext, ty) ==
        offsetof (klayout_cuda_spatial_m1_width_space_context_v1, ty) &&
      offsetof (CudaM1WidthSpaceContext, cell_id) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_context_v1, cell_id) &&
      offsetof (CudaM1WidthSpaceContext, transform_code) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_context_v1, transform_code),
      "M1 context ABI layout mismatch");
    static_assert (
      sizeof (CudaM1WidthSpaceCell) ==
        sizeof (klayout_cuda_spatial_m1_width_space_cell_v1) &&
      offsetof (CudaM1WidthSpaceCell, source_cell_index) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_cell_v1,
          source_cell_index) &&
      offsetof (CudaM1WidthSpaceCell, polygon_begin) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_cell_v1, polygon_begin) &&
      offsetof (CudaM1WidthSpaceCell, edge_begin) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_cell_v1, edge_begin) &&
      offsetof (CudaM1WidthSpaceCell, polygon_count) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_cell_v1, polygon_count) &&
      offsetof (CudaM1WidthSpaceCell, edge_count) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_cell_v1, edge_count),
      "M1 cell ABI layout mismatch");
    static_assert (
      sizeof (CudaM1WidthSpacePolygon) ==
        sizeof (klayout_cuda_spatial_m1_width_space_polygon_v1) &&
      offsetof (CudaM1WidthSpacePolygon, edge_begin) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_polygon_v1, edge_begin) &&
      offsetof (CudaM1WidthSpacePolygon, left) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_polygon_v1, left) &&
      offsetof (CudaM1WidthSpacePolygon, bottom) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_polygon_v1, bottom) &&
      offsetof (CudaM1WidthSpacePolygon, right) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_polygon_v1, right) &&
      offsetof (CudaM1WidthSpacePolygon, top) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_polygon_v1, top) &&
      offsetof (CudaM1WidthSpacePolygon, polygon_id) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_polygon_v1, polygon_id) &&
      offsetof (CudaM1WidthSpacePolygon, edge_count) ==
        offsetof (
          klayout_cuda_spatial_m1_width_space_polygon_v1, edge_count),
      "M1 polygon ABI layout mismatch");
    static_assert (
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
      "M1 edge ABI layout mismatch");

    CudaM1WidthSpaceSceneLimits limits;
    limits.max_cells = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_CELLS"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_CELLS",
      limits.max_cells);
    limits.max_contexts = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_CONTEXTS"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_CONTEXTS",
      limits.max_contexts);
    limits.max_stored_polygons = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_STORED_POLYGONS"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_STORED_POLYGONS",
      limits.max_stored_polygons);
    limits.max_stored_edges = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_STORED_EDGES"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_STORED_EDGES",
      limits.max_stored_edges);
    limits.max_flat_polygons = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_FLAT_POLYGONS"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_FLAT_POLYGONS",
      backend_max_flat_polygons);
    limits.max_flat_edges = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_FLAT_EDGES"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_FLAT_EDGES",
      backend_max_flat_edges);

    const uint64_t max_grid_cells = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_GRID_CELLS"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_GRID_CELLS",
      default_max_grid_cells);
    const uint64_t max_memberships = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_MEMBERSHIPS"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_MEMBERSHIPS",
      default_max_memberships);
    const uint64_t max_pair_work = env_u64 (
      metal2 ? "KLAYOUT_CUDA_M2_WIDTH_SPACE_MAX_PAIR_WORK"
             : "KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_PAIR_WORK",
      default_max_pair_work);
    const uint64_t device = env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0);
    if (! max_grid_cells || ! max_memberships || ! max_pair_work ||
        device > uint64_t (std::numeric_limits<int32_t>::max ())) {
      throw M1WidthSpaceDecline (
        "an M1 backend capacity or device is invalid");
    }

    CudaM1WidthSpaceScene scene;
    std::string reason;
    if (! cuda_m1_width_space_build_scene (
          merged_metal1, merged_metal1, spec, limits, scene, &reason)) {
      throw M1WidthSpaceDecline (
        reason.empty () ? "unable to build the live M1 scene" : reason);
    }

    klayout_cuda_spatial_m1_width_space_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode =
      metal2
        ? KLAYOUT_CUDA_SPATIAL_M2_WIDTH_SPACE_MERGED_EMPTY
        : KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_MERGED_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_QUALIFIED_OPTIONS;
    request.format_version = scene.format_version;
    request.dbu_per_micron = scene.dbu_per_micron;
    request.root_cell = scene.root_cell;
    request.scene_reserved = scene.reserved;
    request.device = int32_t (device);
    request.width_distance = scene.width_distance;
    request.spacing_distance = scene.spacing_distance;
    request.grid_cell_size = qualified_grid_cell;
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
    request.max_contexts = limits.max_contexts;
    request.max_grid_cells = max_grid_cells;
    request.max_memberships = max_memberships;
    request.max_pair_work = max_pair_work;
    request.max_flat_edges = limits.max_flat_edges;
    request.max_flat_polygons = limits.max_flat_polygons;
    std::copy (
      scene.digest.begin (), scene.digest.end (), request.scene_digest);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const double lower_ms =
      std::chrono::duration<double, std::milli> (
        call_begin - begin).count ();
    const db::CudaM1WidthSpaceAttempt attempt =
      db::cuda_spatial_try_m1_width_space_empty (request);
    const std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << (metal2
                    ? "CUDA M2 width/space live lowering:"
                    : "CUDA M1 width/space live lowering:")
               << " contexts=" << request.context_count
               << " metal_contexts=" << request.metal_context_count
               << " cells=" << request.cell_count
               << " stored_polygons=" << request.polygon_count
               << " stored_edges=" << request.edge_count
               << " flat_polygons=" << request.flat_polygon_count
               << " flat_edges=" << request.flat_edge_count
               << " lower_ms=" << lower_ms
               << " call_ms="
               << std::chrono::duration<double, std::milli> (
                    end - call_begin).count ()
               << " live_total_ms="
               << std::chrono::duration<double, std::milli> (
                    end - begin).count ();
    }
    return
      attempt.disposition == db::CudaM1WidthSpaceAttempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << (metal2
                      ? "CUDA M2 width/space live lowering:"
                      : "CUDA M1 width/space live lowering:")
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << (metal2
                      ? "CUDA M2 width/space live lowering:"
                      : "CUDA M1 width/space live lowering:")
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

} // namespace db
