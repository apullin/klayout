/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM1.h"

#include "dbArray.h"
#include "dbCell.h"
#include "dbCudaActive3Digest.h"
#include "dbCudaManhattanContour.h"
#include "dbDeepShapeStore.h"
#include "dbLayerProperties.h"
#include "dbLayout.h"
#include "dbPolygon.h"
#include "dbShape.h"
#include "dbShapes.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace db
{

namespace
{

const uint32_t capture_format_version = 2;
const uint32_t legacy_capture_format_version = 1;
const uint32_t raw_scene_format_version = 1;
const uint32_t qualified_dbu_per_micron = 2000;
const unsigned maximum_hierarchy_depth = 1024;
const int64_t accepted_coordinate_magnitude = INT64_C (1000000000000);

const uint64_t default_max_total_stored_bytes =
  UINT64_C (4) * 1024 * 1024 * 1024;
const uint64_t default_max_total_expanded_geometry_bytes =
  UINT64_C (8) * 1024 * 1024 * 1024;
const uint64_t default_max_estimated_peak_bytes =
  UINT64_C (12) * 1024 * 1024 * 1024;

typedef bool (*RawSceneDigest) (
  const db::CudaRawManhattanScene &, std::array<uint8_t, 32> &);

struct DomainProfile
{
  uint32_t role;
  uint32_t physical_layer;
  uint32_t datatype;
  const char *label;
  char digest_magic [8];
  RawSceneDigest digest;
};

const DomainProfile domain_profiles [CudaAntennaM1DomainCount] = {
  {
    CudaAntennaM1Poly, 9, 0, "POLY",
    { 'K', 'P', 'O', 'L', 'Y', '0', '0', '1' },
    db::cuda_poly_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Active, 1, 0, "ACTIVE",
    { 'K', 'A', 'R', 'A', 'W', '0', '0', '1' },
    db::cuda_active_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Nplus, 4, 0, "NPLUS",
    { 'K', 'N', 'P', 'L', 'S', '0', '0', '1' },
    db::cuda_nplus_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Nwell, 3, 0, "NWELL",
    { 'K', 'N', 'W', 'E', 'L', '0', '0', '1' },
    db::cuda_nwell_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Contact, 10, 0, "CONTACT",
    { 'K', 'C', 'R', 'A', 'W', '0', '0', '1' },
    db::cuda_contact_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Metal1, 11, 0, "M1",
    { 'K', 'M', '1', 'R', 'A', 'W', '0', '1' },
    db::cuda_m1_raw_manhattan_scene_digest
  }
};

class AntennaM1Decline
  : public std::runtime_error
{
public:
  explicit AntennaM1Decline (const std::string &message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

bool checked_add_u64 (uint64_t first, uint64_t second, uint64_t &result)
{
  if (second > std::numeric_limits<uint64_t>::max () - first) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply_u64 (
  uint64_t first, uint64_t second, uint64_t &result)
{
  if (first && second > std::numeric_limits<uint64_t>::max () / first) {
    return false;
  }
  result = first * second;
  return true;
}

uint64_t size_u64 (size_t value, const char *what)
{
  if (value > std::numeric_limits<uint64_t>::max ()) {
    throw AntennaM1Decline (
      std::string (what) + " cannot be represented by uint64");
  }
  return uint64_t (value);
}

void checked_accumulate (
  uint64_t value, uint64_t &total, const char *what)
{
  if (! checked_add_u64 (total, value, total)) {
    throw AntennaM1Decline (std::string (what) + " overflows uint64");
  }
}

uint64_t checked_array_bytes (
  uint64_t count, uint64_t stride, const char *what)
{
  uint64_t result = 0;
  if (! checked_multiply_u64 (count, stride, result)) {
    throw AntennaM1Decline (std::string (what) + " overflows uint64");
  }
  return result;
}

void require_room (
  uint64_t current, uint64_t additional, uint64_t maximum,
  const char *what)
{
  uint64_t total = 0;
  if (! checked_add_u64 (current, additional, total) || total > maximum) {
    throw AntennaM1Decline (
      std::string (what) + " exceeds the configured capacity");
  }
}

int64_t narrow_i64 (__int128 value, const char *what)
{
  if (value < std::numeric_limits<int64_t>::min () ||
      value > std::numeric_limits<int64_t>::max ()) {
    throw AntennaM1Decline (
      std::string (what) + " overflows signed int64");
  }
  const int64_t result = int64_t (value);
  if (result < -accepted_coordinate_magnitude ||
      result > accepted_coordinate_magnitude) {
    throw AntennaM1Decline (
      std::string (what) + " exceeds the qualified coordinate domain");
  }
  return result;
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
    for (unsigned int i = 0; i < 4; ++i) {
      encoded [i] = uint8_t (value >> (i * 8));
    }
    bytes (encoded, sizeof (encoded));
  }

  void u64 (uint64_t value)
  {
    uint8_t encoded [8];
    for (unsigned int i = 0; i < 8; ++i) {
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
    throw AntennaM1Decline ("invalid orthogonal transform code");
  }
  const Matrix &matrix = transforms [code];
  return std::make_pair (
    __int128 (matrix.xx) * x + __int128 (matrix.xy) * y,
    __int128 (matrix.yx) * x + __int128 (matrix.yy) * y);
}

uint32_t compose_transform (uint32_t outer, uint32_t inner)
{
  if (outer >= 8 || inner >= 8) {
    throw AntennaM1Decline (
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
  throw AntennaM1Decline (
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

InstanceTemplate make_instance (
  const db::Instance &instance,
  const std::map<db::cell_index_type, uint32_t> &dense_cells)
{
  if (instance.prop_id () != 0 || instance.is_complex ()) {
    throw AntennaM1Decline (
      "hierarchy has an instance property or complex transform");
  }

  const std::map<db::cell_index_type, uint32_t>::const_iterator child =
    dense_cells.find (instance.cell_index ());
  if (child == dense_cells.end ()) {
    throw AntennaM1Decline (
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
    throw AntennaM1Decline (
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
    throw AntennaM1Decline (
      "hierarchy has an invalid array dimension");
  }

  const db::Trans &trans = instance.front ();
  const int transform_code = trans.rot ();
  if (transform_code < 0 || transform_code >= 8) {
    throw AntennaM1Decline (
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
    throw AntennaM1Decline (
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
    throw AntennaM1Decline (
      "hierarchy exceeds the qualified recursion depth");
  }
  if (cell >= cells.size ()) {
    throw AntennaM1Decline (
      "hierarchy references an invalid dense cell");
  }
  if (state [cell] == 1) {
    throw AntennaM1Decline ("hierarchy contains a cycle");
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
      throw AntennaM1Decline (
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
  std::vector<CudaM1WidthSpaceContext> &contexts,
  std::vector<uint32_t> &parents)
{
  std::vector<uint8_t> state (templates.size (), 0);
  std::vector<uint64_t> memo (templates.size (), 0);
  const uint64_t expected = subtree_context_count (
    root, templates, state, memo, max_contexts, 0);
  if (expected > std::numeric_limits<uint32_t>::max () ||
      expected > contexts.max_size () ||
      expected > parents.max_size ()) {
    throw AntennaM1Decline (
      "expanded hierarchy cannot be represented by context IDs");
  }

  contexts.reserve (size_t (expected));
  parents.reserve (size_t (expected));
  contexts.push_back (CudaM1WidthSpaceContext { 0, 0, root, 0 });
  parents.push_back (std::numeric_limits<uint32_t>::max ());
  for (size_t parent_id = 0; parent_id < contexts.size (); ++parent_id) {
    const CudaM1WidthSpaceContext parent = contexts [parent_id];
    if (parent.cell_id >= templates.size ()) {
      throw AntennaM1Decline (
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
          parents.push_back (uint32_t (parent_id));
        }
      }
    }
  }
  if (contexts.size () != expected || parents.size () != contexts.size ()) {
    throw AntennaM1Decline (
      "expanded hierarchy disagrees with the checked context census");
  }
}

void append_polygon (
  const db::Shape &shape, uint32_t polygon_id,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaAntennaM1DomainScene &scene,
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> &contour_cache)
{
  if (shape.prop_id () != 0) {
    throw AntennaM1Decline ("Manhattan scene polygon has properties");
  }
  if (! shape.is_box () && ! shape.is_polygon ()) {
    throw AntennaM1Decline (
      "Manhattan scene layer contains a non-polygon shape");
  }

  db::Polygon polygon;
  if (! shape.polygon (polygon)) {
    throw AntennaM1Decline ("Manhattan scene polygon is malformed");
  }
  if (polygon.holes () != 0) {
    throw AntennaM1Decline (
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
      throw AntennaM1Decline (
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
    throw AntennaM1Decline (
      "Manhattan scene polygon is too small, empty, or not clockwise");
  }
  const cuda_manhattan_contour::ValidationResult contour_result =
    contour_cache.validate_contour (contour);
  if (contour_result ==
      cuda_manhattan_contour::ValidationResult::OpenContour) {
    throw AntennaM1Decline ("Manhattan scene polygon contour is open");
  }
  if (contour_result !=
      cuda_manhattan_contour::ValidationResult::Valid) {
    throw AntennaM1Decline (
      "Manhattan scene polygon contour self-intersects");
  }
  if (contour.size () > std::numeric_limits<uint32_t>::max ()) {
    throw AntennaM1Decline (
      "one Manhattan scene polygon has more than uint32 contour edges");
  }

  const uint64_t stored_polygons =
    size_u64 (scene.polygons.size (), "stored polygon count");
  const uint64_t stored_edges =
    size_u64 (scene.edges.size (), "stored edge count");
  require_room (
    stored_polygons, 1, limits.max_stored_polygons,
    "stored polygon count");
  require_room (
    stored_edges, uint64_t (contour.size ()), limits.max_stored_edges,
    "stored edge count");
  if (scene.polygons.size () == scene.polygons.max_size () ||
      contour.size () > scene.edges.max_size () - scene.edges.size ()) {
    throw AntennaM1Decline (
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

void append_cell_layer (
  const db::Cell &cell, unsigned int layer,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaAntennaM1DomainScene &scene,
  CudaAntennaM1DomainCell &record,
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> &contour_cache)
{
  record.polygon_begin =
    size_u64 (scene.polygons.size (), "cell polygon begin");
  record.edge_begin =
    size_u64 (scene.edges.size (), "cell edge begin");

  uint32_t polygon_id = 0;
  const db::Shapes &shapes = cell.shapes (layer);
  for (db::Shapes::shape_iterator shape =
         shapes.begin (db::ShapeIterator::All);
       ! shape.at_end (); ++shape) {
    if (shape->is_text ()) {
      continue;
    }
    if (polygon_id == std::numeric_limits<uint32_t>::max ()) {
      throw AntennaM1Decline (
        "per-cell Manhattan polygon count exceeds uint32");
    }
    append_polygon (*shape, polygon_id, limits, scene, contour_cache);
    ++polygon_id;
  }

  const uint64_t polygon_count =
    size_u64 (scene.polygons.size (), "stored polygon count") -
    record.polygon_begin;
  const uint64_t edge_count =
    size_u64 (scene.edges.size (), "stored edge count") -
    record.edge_begin;
  if (polygon_count > std::numeric_limits<uint32_t>::max () ||
      edge_count > std::numeric_limits<uint32_t>::max ()) {
    throw AntennaM1Decline (
      "per-cell Manhattan polygon or edge count exceeds uint32");
  }
  record.polygon_count = uint32_t (polygon_count);
  record.edge_count = uint32_t (edge_count);
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

struct DomainSummary
{
  uint64_t nonempty_context_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  DomainSummary ()
    : nonempty_context_count (0), flat_polygon_count (0),
      flat_edge_count (0), scene_left (0), scene_bottom (0),
      scene_right (0), scene_top (0)
  {
    //  nothing yet
  }
};

DomainSummary derive_domain_summary (
  const CudaAntennaM1Capture &capture,
  const CudaAntennaM1DomainScene &scene,
  const CudaM1WidthSpaceSceneLimits *limits,
  const char *label,
  bool derive_world_bounds)
{
  DomainSummary summary;
  //  The build path derives world bounds exactly once.  Subsequent census and
  //  digest validation follows the legacy raw-scene contract: bounds must be
  //  sane and are canonical-digest-bound, but validation does not flatten
  //  every polygon occurrence again.
  bool have_bounds = ! derive_world_bounds;
  if (! derive_world_bounds) {
    summary.scene_left = scene.scene_left;
    summary.scene_bottom = scene.scene_bottom;
    summary.scene_right = scene.scene_right;
    summary.scene_top = scene.scene_top;
  }
  for (size_t context_id = 0;
       context_id < capture.contexts.size (); ++context_id) {
    const CudaM1WidthSpaceContext &context =
      capture.contexts [context_id];
    if (context.cell_id >= scene.cells.size ()) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + label +
        " context references an invalid dense cell");
    }
    const CudaAntennaM1DomainCell &cell =
      scene.cells [context.cell_id];
    if (! cell.polygon_count) {
      continue;
    }
    if (context_id > std::numeric_limits<uint32_t>::max ()) {
      throw AntennaM1Decline (
        "nonempty context ID exceeds uint32");
    }
    checked_accumulate (
      1, summary.nonempty_context_count,
      "nonempty context count");
    checked_accumulate (
      cell.polygon_count, summary.flat_polygon_count,
      "flattened Manhattan polygon count");
    checked_accumulate (
      cell.edge_count, summary.flat_edge_count,
      "flattened Manhattan edge count");
    if (limits &&
        (summary.flat_polygon_count > limits->max_flat_polygons ||
         summary.flat_edge_count > limits->max_flat_edges)) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + label +
        " flattened geometry exceeds the configured capacity");
    }

    if (derive_world_bounds) {
      uint64_t polygon_end = 0;
      if (! checked_add_u64 (
            cell.polygon_begin, cell.polygon_count, polygon_end) ||
          polygon_end > scene.polygons.size ()) {
        throw AntennaM1Decline (
          std::string ("M1 antenna ") + label +
          " cell polygon range escapes the serialized array");
      }
      for (uint64_t polygon_id = cell.polygon_begin;
           polygon_id < polygon_end; ++polygon_id) {
        add_world_polygon_bounds (
          context, scene.polygons [size_t (polygon_id)], have_bounds,
          summary.scene_left, summary.scene_bottom,
          summary.scene_right, summary.scene_top);
      }
    }
  }
  if (! have_bounds || ! summary.flat_polygon_count ||
      ! summary.flat_edge_count) {
    throw AntennaM1Decline (
      std::string ("M1 antenna ") + label +
      " qualified Manhattan scene is empty");
  }
  return summary;
}

void apply_domain_summary (
  const DomainSummary &summary, CudaAntennaM1DomainScene &scene)
{
  scene.flat_polygon_count = summary.flat_polygon_count;
  scene.flat_edge_count = summary.flat_edge_count;
  scene.scene_left = summary.scene_left;
  scene.scene_bottom = summary.scene_bottom;
  scene.scene_right = summary.scene_right;
  scene.scene_top = summary.scene_top;
}

void validate_scene_limits (const CudaM1WidthSpaceSceneLimits &limits)
{
  if (! limits.max_cells || ! limits.max_contexts ||
      ! limits.max_stored_polygons || ! limits.max_stored_edges ||
      ! limits.max_flat_polygons || ! limits.max_flat_edges) {
    throw AntennaM1Decline ("a Manhattan scene capacity is zero");
  }
}

void validate_input_identity (
  const std::array<const db::DeepLayer *, CudaAntennaM1DomainCount>
    &inputs,
  const CudaM1WidthSpaceSceneLimits &limits)
{
  validate_scene_limits (limits);
  const db::DeepLayer &reference = *inputs [0];
  std::set<unsigned int> layers;
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    const db::DeepLayer &candidate = *inputs [domain];
    const DomainProfile &profile = domain_profiles [domain];
    if (candidate.store () != reference.store () ||
        &candidate.layout () != &reference.layout () ||
        candidate.layout_index () != reference.layout_index () ||
        candidate.initial_cell ().cell_index () !=
          reference.initial_cell ().cell_index ()) {
      throw AntennaM1Decline (
        "M1 antenna raw domains do not share one hierarchy");
    }
    if (! layers.insert (candidate.layer ()).second) {
      throw AntennaM1Decline (
        "M1 antenna raw domains do not use six distinct layers");
    }
    if (candidate.breakout_cells () != 0) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " scene has hierarchy breakout cells");
    }
    if (candidate.layout ().dbu () != 0.0005) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " scene DBU is not the qualified 0.5 nm");
    }
    if (! candidate.layout ().is_valid_layer (candidate.layer ())) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " scene layer index is not a valid physical layer");
    }
    const db::LayerProperties &properties =
      candidate.layout ().get_properties (candidate.layer ());
    if (! properties.log_equal (
          db::LayerProperties (
            profile.physical_layer, profile.datatype))) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " scene is not physical FreePDK45 layer " +
        std::to_string (profile.physical_layer) + "/" +
        std::to_string (profile.datatype));
    }
  }
}

void serialize_shared_capture (
  const std::array<const db::DeepLayer *, CudaAntennaM1DomainCount>
    &inputs,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaAntennaM1Capture &capture)
{
  const db::DeepLayer &reference = *inputs [0];
  const db::Layout &layout = reference.layout ();
  const db::cell_index_type top =
    reference.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  reference.initial_cell ().collect_called_cells (reachable);
  if (reachable.empty () ||
      reachable.size () > std::numeric_limits<uint32_t>::max () ||
      reachable.size () > limits.max_cells) {
    throw AntennaM1Decline (
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
    throw AntennaM1Decline (
      "initial cell is absent from the hierarchy census");
  }

  capture.dbu_per_micron = qualified_dbu_per_micron;
  capture.root_cell = root->second;
  capture.source_root_cell_index = uint64_t (top);
  capture.source_cell_indices.resize (reachable.size ());
  std::vector<CellTemplate> templates (reachable.size ());
  std::array<
    cuda_manhattan_contour::TranslationValidationCache<
      CudaM1WidthSpaceEdge>,
    CudaAntennaM1DomainCount> contour_caches;
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    capture.source_layer_indices [domain] = inputs [domain]->layer ();
    capture.domains [domain].cells.resize (reachable.size ());
  }

  for (std::set<db::cell_index_type>::const_iterator source =
         reachable.begin (); source != reachable.end (); ++source) {
    const uint32_t cell_id = dense_cells.find (*source)->second;
    const db::Cell &cell = layout.cell (*source);
    capture.source_cell_indices [cell_id] = uint64_t (*source);
    for (db::Cell::const_iterator instance = cell.begin ();
         ! instance.at_end (); ++instance) {
      templates [cell_id].instances.push_back (
        make_instance (*instance, dense_cells));
    }

    for (size_t domain = 0;
         domain < CudaAntennaM1DomainCount; ++domain) {
      CudaAntennaM1DomainCell record = {};
      append_cell_layer (
        cell, inputs [domain]->layer (), limits,
        capture.domains [domain], record, contour_caches [domain]);
      capture.domains [domain].cells [cell_id] = record;
    }
  }

  expand_contexts (
    root->second, templates, limits.max_contexts,
    capture.contexts, capture.context_parent_ids);
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    const DomainSummary summary = derive_domain_summary (
      capture, capture.domains [domain], &limits,
      domain_profiles [domain].label, true);
    apply_domain_summary (summary, capture.domains [domain]);
  }
}

bool checked_range (uint64_t begin, uint64_t count, uint64_t size)
{
  uint64_t end = 0;
  return checked_add_u64 (begin, count, end) && end <= size;
}

void validate_shared_hierarchy (const CudaAntennaM1Capture &capture)
{
  if (capture.format_version != capture_format_version ||
      capture.dbu_per_micron != qualified_dbu_per_micron ||
      capture.reserved != 0 ||
      capture.source_cell_indices.empty () ||
      capture.contexts.empty () ||
      capture.root_cell >= capture.source_cell_indices.size () ||
      capture.source_root_cell_index !=
        capture.source_cell_indices [capture.root_cell]) {
    throw AntennaM1Decline (
      "M1 antenna shared capture header or root is inconsistent");
  }
  const CudaM1WidthSpaceContext &root = capture.contexts.front ();
  if (root.tx != 0 || root.ty != 0 ||
      root.cell_id != capture.root_cell ||
      root.transform_code != 0) {
    throw AntennaM1Decline (
      "M1 antenna shared root context is inconsistent");
  }
  if (capture.context_parent_ids.size () != capture.contexts.size () ||
      capture.context_parent_ids.front () !=
        std::numeric_limits<uint32_t>::max ()) {
    throw AntennaM1Decline (
      "M1 antenna context-parent census is inconsistent");
  }

  std::set<uint64_t> source_cells;
  for (size_t cell = 0;
       cell < capture.source_cell_indices.size (); ++cell) {
    if (! source_cells.insert (
          capture.source_cell_indices [cell]).second) {
      throw AntennaM1Decline (
        "M1 antenna source-cell identity is duplicated");
    }
  }
  for (size_t context = 0;
       context < capture.contexts.size (); ++context) {
    const CudaM1WidthSpaceContext &record = capture.contexts [context];
    if (record.cell_id >= capture.source_cell_indices.size () ||
        record.transform_code >= 8 ||
        (context && capture.context_parent_ids [context] >= context)) {
      throw AntennaM1Decline (
        "M1 antenna shared context or parent is inconsistent");
    }
  }

  std::set<uint32_t> source_layers;
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    if (! source_layers.insert (
          capture.source_layer_indices [domain]).second) {
      throw AntennaM1Decline (
        "M1 antenna source-layer binding is inconsistent");
    }
  }
}

void validate_domain_geometry (
  const CudaAntennaM1Capture &capture, size_t domain,
  DomainSummary &summary)
{
  const DomainProfile &profile = domain_profiles [domain];
  const CudaAntennaM1DomainScene &scene = capture.domains [domain];
  if (profile.role != domain ||
      scene.cells.size () != capture.source_cell_indices.size () ||
      scene.polygons.empty () || scene.edges.empty () ||
      scene.scene_left >= scene.scene_right ||
      scene.scene_bottom >= scene.scene_top) {
    throw AntennaM1Decline (
      std::string ("M1 antenna ") + profile.label +
      " compact geometry header is inconsistent");
  }

  uint64_t next_polygon = 0;
  uint64_t next_edge = 0;
  for (size_t cell_id = 0; cell_id < scene.cells.size (); ++cell_id) {
    const CudaAntennaM1DomainCell &cell = scene.cells [cell_id];
    if (cell.polygon_begin != next_polygon ||
        cell.edge_begin != next_edge ||
        ! checked_range (
          cell.polygon_begin, cell.polygon_count, scene.polygons.size ()) ||
        ! checked_range (
          cell.edge_begin, cell.edge_count, scene.edges.size ())) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " cell range is inconsistent");
    }

    uint64_t cell_edge = cell.edge_begin;
    for (uint32_t local = 0; local < cell.polygon_count; ++local) {
      const CudaM1WidthSpacePolygon &polygon =
        scene.polygons [size_t (cell.polygon_begin + local)];
      if (polygon.polygon_id != local ||
          polygon.edge_begin != cell_edge || polygon.edge_count < 4 ||
          polygon.left >= polygon.right ||
          polygon.bottom >= polygon.top ||
          ! checked_range (
            polygon.edge_begin, polygon.edge_count, scene.edges.size ())) {
        throw AntennaM1Decline (
          std::string ("M1 antenna ") + profile.label +
          " polygon range is inconsistent");
      }
      uint64_t polygon_end = 0;
      if (! checked_add_u64 (
            polygon.edge_begin, polygon.edge_count, polygon_end)) {
        throw AntennaM1Decline (
          std::string ("M1 antenna ") + profile.label +
          " polygon edge range overflows");
      }
      for (uint64_t edge_id = polygon.edge_begin;
           edge_id < polygon_end; ++edge_id) {
        const CudaM1WidthSpaceEdge &edge =
          scene.edges [size_t (edge_id)];
        if ((edge.x1 == edge.x2 && edge.y1 == edge.y2) ||
            ! (edge.x1 == edge.x2 || edge.y1 == edge.y2)) {
          throw AntennaM1Decline (
            std::string ("M1 antenna ") + profile.label +
            " edge is degenerate or non-Manhattan");
        }
        const uint64_t following =
          edge_id + 1 == polygon_end ? polygon.edge_begin : edge_id + 1;
        const CudaM1WidthSpaceEdge &next =
          scene.edges [size_t (following)];
        if (edge.x2 != next.x1 || edge.y2 != next.y1) {
          throw AntennaM1Decline (
            std::string ("M1 antenna ") + profile.label +
            " polygon contour is open");
        }
      }
      cell_edge = polygon_end;
    }
    uint64_t cell_edge_end = 0;
    if (! checked_add_u64 (
          cell.edge_begin, cell.edge_count, cell_edge_end) ||
        cell_edge != cell_edge_end) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " cell edge census is inconsistent");
    }
    checked_accumulate (
      cell.polygon_count, next_polygon, "stored polygon count");
    checked_accumulate (
      cell.edge_count, next_edge, "stored edge count");
  }
  if (next_polygon != scene.polygons.size () ||
      next_edge != scene.edges.size ()) {
    throw AntennaM1Decline (
      std::string ("M1 antenna ") + profile.label +
      " geometry arrays are not fully covered");
  }

  summary = derive_domain_summary (
    capture, scene, 0, profile.label, false);
  if (scene.flat_polygon_count != summary.flat_polygon_count ||
      scene.flat_edge_count != summary.flat_edge_count ||
      scene.scene_left != summary.scene_left ||
      scene.scene_bottom != summary.scene_bottom ||
      scene.scene_right != summary.scene_right ||
      scene.scene_top != summary.scene_top) {
    throw AntennaM1Decline (
      std::string ("M1 antenna ") + profile.label +
      " flattened census or bounds are inconsistent");
  }
}

std::array<uint8_t, 32> legacy_domain_digest (
  const CudaAntennaM1Capture &capture, size_t domain,
  const DomainSummary &summary)
{
  const DomainProfile &profile = domain_profiles [domain];
  const CudaAntennaM1DomainScene &scene = capture.domains [domain];
  CanonicalDigest sha;
  sha.bytes (profile.digest_magic, sizeof (profile.digest_magic));
  sha.u32 (raw_scene_format_version);
  sha.u32 (capture.dbu_per_micron);
  sha.u32 (capture.root_cell);
  sha.u32 (capture.reserved);
  sha.u64 (capture.contexts.size ());
  sha.u64 (summary.nonempty_context_count);
  sha.u64 (scene.cells.size ());
  sha.u64 (scene.polygons.size ());
  sha.u64 (scene.edges.size ());
  sha.u64 (summary.flat_polygon_count);
  sha.u64 (summary.flat_edge_count);
  sha.i64 (summary.scene_left);
  sha.i64 (summary.scene_bottom);
  sha.i64 (summary.scene_right);
  sha.i64 (summary.scene_top);

  for (std::vector<CudaM1WidthSpaceContext>::const_iterator context =
         capture.contexts.begin (); context != capture.contexts.end ();
       ++context) {
    sha.i64 (context->tx);
    sha.i64 (context->ty);
    sha.u32 (context->cell_id);
    sha.u32 (context->transform_code);
  }

  uint64_t polygon_offset = 0;
  uint64_t edge_offset = 0;
  for (size_t context_id = 0;
       context_id < capture.contexts.size (); ++context_id) {
    const CudaAntennaM1DomainCell &cell =
      scene.cells [capture.contexts [context_id].cell_id];
    if (! cell.polygon_count) {
      continue;
    }
    sha.u32 (uint32_t (context_id));
    sha.u64 (polygon_offset);
    sha.u64 (edge_offset);
    checked_accumulate (
      cell.polygon_count, polygon_offset,
      "legacy context polygon offset");
    checked_accumulate (
      cell.edge_count, edge_offset,
      "legacy context edge offset");
  }

  for (size_t cell_id = 0; cell_id < scene.cells.size (); ++cell_id) {
    const CudaAntennaM1DomainCell &cell = scene.cells [cell_id];
    sha.u64 (capture.source_cell_indices [cell_id]);
    sha.u64 (cell.polygon_begin);
    sha.u64 (cell.edge_begin);
    sha.u32 (cell.polygon_count);
    sha.u32 (cell.edge_count);
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
  return sha.finish ();
}

std::array<uint8_t, 32> hierarchy_digest (
  const CudaAntennaM1Capture &capture)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'H', '1', '0', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (legacy_capture_format_version);
  sha.u32 (capture.dbu_per_micron);
  sha.u32 (capture.root_cell);
  sha.u32 (capture.reserved);
  sha.u64 (capture.source_root_cell_index);
  sha.u64 (capture.source_cell_indices.size ());
  sha.u64 (capture.contexts.size ());
  for (std::vector<uint64_t>::const_iterator cell =
         capture.source_cell_indices.begin ();
       cell != capture.source_cell_indices.end (); ++cell) {
    sha.u64 (*cell);
  }
  for (size_t context_id = 0;
       context_id < capture.contexts.size (); ++context_id) {
    const CudaM1WidthSpaceContext &context =
      capture.contexts [context_id];
    sha.i64 (context.tx);
    sha.i64 (context.ty);
    sha.u32 (context.cell_id);
    sha.u32 (context.transform_code);
    sha.u32 (capture.context_parent_ids [context_id]);
  }
  return sha.finish ();
}

uint64_t domain_stored_bytes (const CudaAntennaM1DomainScene &scene)
{
  uint64_t total =
    UINT64_C (2) * sizeof (uint64_t) +
    UINT64_C (4) * sizeof (int64_t) +
    UINT64_C (32);
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.cells.size (), "stored domain cell count"),
      sizeof (CudaAntennaM1DomainCell), "stored domain cell bytes"),
    total, "stored domain bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.polygons.size (), "stored polygon count"),
      sizeof (CudaM1WidthSpacePolygon), "stored polygon bytes"),
    total, "stored domain bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.edges.size (), "stored edge count"),
      sizeof (CudaM1WidthSpaceEdge), "stored edge bytes"),
    total, "stored domain bytes");
  return total;
}

uint64_t legacy_stored_scene_bytes (
  const CudaAntennaM1Capture &capture,
  const CudaAntennaM1DomainScene &scene,
  const DomainSummary &summary)
{
  uint64_t total =
    UINT64_C (4) * sizeof (uint32_t) +
    UINT64_C (2) * sizeof (uint64_t) +
    UINT64_C (4) * sizeof (int64_t) +
    UINT64_C (32);
  checked_accumulate (
    checked_array_bytes (
      size_u64 (capture.contexts.size (), "legacy context count"),
      sizeof (CudaM1WidthSpaceContext), "legacy context bytes"),
    total, "legacy stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      summary.nonempty_context_count, sizeof (uint32_t),
      "legacy nonempty context bytes"),
    total, "legacy stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      summary.nonempty_context_count, sizeof (uint64_t),
      "legacy context polygon-offset bytes"),
    total, "legacy stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      summary.nonempty_context_count, sizeof (uint64_t),
      "legacy context edge-offset bytes"),
    total, "legacy stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.cells.size (), "legacy stored cell count"),
      sizeof (CudaM1WidthSpaceCell), "legacy stored cell bytes"),
    total, "legacy stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.polygons.size (), "legacy stored polygon count"),
      sizeof (CudaM1WidthSpacePolygon), "legacy stored polygon bytes"),
    total, "legacy stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.edges.size (), "legacy stored edge count"),
      sizeof (CudaM1WidthSpaceEdge), "legacy stored edge bytes"),
    total, "legacy stored scene bytes");
  return total;
}

uint64_t shared_capture_bytes (const CudaAntennaM1Capture &capture)
{
  uint64_t total =
    UINT64_C (4) * sizeof (uint32_t) +
    sizeof (uint64_t) +
    CudaAntennaM1DomainCount * sizeof (uint32_t) +
    UINT64_C (2) * 32;
  checked_accumulate (
    checked_array_bytes (
      size_u64 (
        capture.source_cell_indices.size (), "shared source-cell count"),
      sizeof (uint64_t), "shared source-cell bytes"),
    total, "shared capture bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (capture.contexts.size (), "shared context count"),
      sizeof (CudaM1WidthSpaceContext), "shared context bytes"),
    total, "shared capture bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (
        capture.context_parent_ids.size (), "context-parent count"),
      sizeof (uint32_t), "context-parent bytes"),
    total, "shared capture bytes");
  return total;
}

uint64_t expanded_geometry_bytes (
  const CudaAntennaM1DomainScene &scene)
{
  uint64_t polygon_stride = 0;
  if (! checked_add_u64 (
        sizeof (CudaM1WidthSpacePolygon), sizeof (uint32_t),
        polygon_stride)) {
    throw AntennaM1Decline (
      "expanded polygon record stride overflows uint64");
  }
  uint64_t total = checked_array_bytes (
    scene.flat_polygon_count, polygon_stride,
    "expanded polygon bytes");
  checked_accumulate (
    checked_array_bytes (
      scene.flat_edge_count, sizeof (CudaM1WidthSpaceEdge),
      "expanded edge bytes"),
    total, "expanded geometry bytes");
  return total;
}

void derive_census (
  const CudaAntennaM1Capture &capture,
  CudaAntennaM1Census &census)
{
  validate_shared_hierarchy (capture);

  CudaAntennaM1Census candidate;
  candidate.format_version = capture.format_version;
  candidate.dbu_per_micron = capture.dbu_per_micron;
  candidate.root_cell = capture.root_cell;
  candidate.reserved = capture.reserved;
  candidate.source_root_cell_index = capture.source_root_cell_index;
  candidate.shared_cell_count =
    size_u64 (capture.source_cell_indices.size (), "shared cell count");
  candidate.shared_context_count =
    size_u64 (capture.contexts.size (), "shared context count");
  candidate.context_parent_record_count =
    size_u64 (
      capture.context_parent_ids.size (), "context-parent count");
  candidate.context_parent_bytes = checked_array_bytes (
    candidate.context_parent_record_count, sizeof (uint32_t),
    "context-parent bytes");
  candidate.stored_context_records = candidate.shared_context_count;
  candidate.total_stored_bytes = shared_capture_bytes (capture);

  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    DomainSummary summary;
    validate_domain_geometry (capture, domain, summary);
    const std::array<uint8_t, 32> digest =
      legacy_domain_digest (capture, domain, summary);
    const CudaAntennaM1DomainScene &scene = capture.domains [domain];
    if (digest != scene.digest) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") +
        domain_profiles [domain].label +
        " raw scene digest is inconsistent");
    }

    CudaAntennaM1DomainCensus &record =
      candidate.domains [domain];
    record.role = domain_profiles [domain].role;
    record.physical_layer = domain_profiles [domain].physical_layer;
    record.datatype = domain_profiles [domain].datatype;
    record.source_layer_index =
      capture.source_layer_indices [domain];
    record.stored_cell_count =
      size_u64 (scene.cells.size (), "stored cell count");
    record.stored_context_count = candidate.shared_context_count;
    record.nonempty_context_count = summary.nonempty_context_count;
    record.stored_polygon_count =
      size_u64 (scene.polygons.size (), "stored polygon count");
    record.stored_edge_count =
      size_u64 (scene.edges.size (), "stored edge count");
    record.expanded_polygon_count = scene.flat_polygon_count;
    record.expanded_edge_count = scene.flat_edge_count;
    record.stored_bytes = domain_stored_bytes (scene);
    record.legacy_stored_bytes =
      legacy_stored_scene_bytes (capture, scene, summary);
    record.expanded_geometry_bytes =
      expanded_geometry_bytes (scene);
    record.scene_digest = digest;

    checked_accumulate (
      record.stored_cell_count, candidate.stored_cell_records,
      "stored cell-record total");
    checked_accumulate (
      record.stored_cell_count,
      candidate.legacy_stored_cell_records,
      "legacy stored cell-record total");
    checked_accumulate (
      record.stored_context_count,
      candidate.legacy_stored_context_records,
      "legacy stored context-record total");
    checked_accumulate (
      record.nonempty_context_count,
      candidate.nonempty_context_records,
      "nonempty context-record total");
    checked_accumulate (
      record.stored_polygon_count,
      candidate.stored_polygon_count,
      "stored polygon total");
    checked_accumulate (
      record.stored_edge_count, candidate.stored_edge_count,
      "stored edge total");
    checked_accumulate (
      record.expanded_polygon_count,
      candidate.expanded_polygon_count,
      "expanded polygon total");
    checked_accumulate (
      record.expanded_edge_count,
      candidate.expanded_edge_count,
      "expanded edge total");
    checked_accumulate (
      record.stored_bytes, candidate.total_stored_bytes,
      "stored byte total");
    checked_accumulate (
      record.legacy_stored_bytes,
      candidate.legacy_total_stored_bytes,
      "legacy stored byte total");
    checked_accumulate (
      record.expanded_geometry_bytes,
      candidate.total_expanded_geometry_bytes,
      "expanded geometry byte total");
  }
  checked_accumulate (
    candidate.context_parent_bytes,
    candidate.legacy_total_stored_bytes,
    "legacy stored byte total");

  if (! checked_add_u64 (
        candidate.total_stored_bytes,
        candidate.total_expanded_geometry_bytes,
        candidate.estimated_peak_bytes) ||
      ! checked_add_u64 (
        candidate.legacy_total_stored_bytes,
        candidate.total_expanded_geometry_bytes,
        candidate.legacy_estimated_peak_bytes)) {
    throw AntennaM1Decline (
      "M1 antenna estimated peak bytes overflow uint64");
  }
  candidate.hierarchy_digest = hierarchy_digest (capture);
  if (candidate.hierarchy_digest != capture.hierarchy_digest) {
    throw AntennaM1Decline (
      "M1 antenna common hierarchy digest is inconsistent");
  }
  census = candidate;
}

std::array<uint8_t, 32> capture_digest (
  const CudaAntennaM1Capture &capture,
  const CudaAntennaM1Census &census)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'M', '1', '0', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (legacy_capture_format_version);
  sha.u32 (capture.dbu_per_micron);
  sha.u32 (capture.root_cell);
  sha.u32 (capture.reserved);
  sha.u64 (capture.source_root_cell_index);
  sha.bytes (
    capture.hierarchy_digest.data (),
    capture.hierarchy_digest.size ());
  sha.u32 (CudaAntennaM1DomainCount);
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    const CudaAntennaM1DomainCensus &record =
      census.domains [domain];
    sha.u32 (record.role);
    sha.u32 (record.physical_layer);
    sha.u32 (record.datatype);
    sha.u32 (record.source_layer_index);
    sha.u64 (record.stored_cell_count);
    sha.u64 (record.stored_context_count);
    sha.u64 (record.nonempty_context_count);
    sha.u64 (record.stored_polygon_count);
    sha.u64 (record.stored_edge_count);
    sha.u64 (record.expanded_polygon_count);
    sha.u64 (record.expanded_edge_count);
    sha.u64 (record.legacy_stored_bytes);
    sha.u64 (record.expanded_geometry_bytes);
    sha.bytes (record.scene_digest.data (), record.scene_digest.size ());
  }
  sha.u64 (census.shared_cell_count);
  sha.u64 (census.shared_context_count);
  sha.u64 (census.context_parent_record_count);
  sha.u64 (census.context_parent_bytes);
  sha.u64 (census.legacy_stored_cell_records);
  sha.u64 (census.legacy_stored_context_records);
  sha.u64 (census.nonempty_context_records);
  sha.u64 (census.stored_polygon_count);
  sha.u64 (census.stored_edge_count);
  sha.u64 (census.expanded_polygon_count);
  sha.u64 (census.expanded_edge_count);
  sha.u64 (census.legacy_total_stored_bytes);
  sha.u64 (census.total_expanded_geometry_bytes);
  sha.u64 (census.legacy_estimated_peak_bytes);
  return sha.finish ();
}

void validate_capture_limits (
  const CudaAntennaM1CaptureLimits &limits,
  const CudaAntennaM1Census &census)
{
  if (! limits.max_total_stored_bytes ||
      ! limits.max_total_expanded_geometry_bytes ||
      ! limits.max_estimated_peak_bytes) {
    throw AntennaM1Decline (
      "an M1 antenna aggregate byte capacity is zero");
  }
  if (census.total_stored_bytes >
      limits.max_total_stored_bytes) {
    throw AntennaM1Decline (
      "M1 antenna stored bytes exceed the configured capacity");
  }
  if (census.total_expanded_geometry_bytes >
      limits.max_total_expanded_geometry_bytes) {
    throw AntennaM1Decline (
      "M1 antenna expanded geometry bytes exceed the configured capacity");
  }
  if (census.estimated_peak_bytes >
      limits.max_estimated_peak_bytes) {
    throw AntennaM1Decline (
      "M1 antenna estimated peak bytes exceed the configured capacity");
  }
}

void materialize_domain_scene (
  const CudaAntennaM1Capture &capture, size_t domain,
  const DomainSummary &summary, CudaRawManhattanScene &scene)
{
  const CudaAntennaM1DomainScene &compact = capture.domains [domain];
  CudaRawManhattanScene candidate;
  candidate.format_version = raw_scene_format_version;
  candidate.dbu_per_micron = capture.dbu_per_micron;
  candidate.root_cell = capture.root_cell;
  candidate.reserved = capture.reserved;
  candidate.flat_polygon_count = summary.flat_polygon_count;
  candidate.flat_edge_count = summary.flat_edge_count;
  candidate.scene_left = summary.scene_left;
  candidate.scene_bottom = summary.scene_bottom;
  candidate.scene_right = summary.scene_right;
  candidate.scene_top = summary.scene_top;
  candidate.contexts = capture.contexts;
  candidate.cells.resize (compact.cells.size ());
  for (size_t cell_id = 0; cell_id < compact.cells.size (); ++cell_id) {
    const CudaAntennaM1DomainCell &source = compact.cells [cell_id];
    CudaM1WidthSpaceCell &destination = candidate.cells [cell_id];
    destination.source_cell_index =
      capture.source_cell_indices [cell_id];
    destination.polygon_begin = source.polygon_begin;
    destination.edge_begin = source.edge_begin;
    destination.polygon_count = source.polygon_count;
    destination.edge_count = source.edge_count;
  }
  candidate.polygons = compact.polygons;
  candidate.edges = compact.edges;

  uint64_t polygon_offset = 0;
  uint64_t edge_offset = 0;
  candidate.metal_contexts.reserve (
    size_t (summary.nonempty_context_count));
  candidate.context_polygon_offsets.reserve (
    size_t (summary.nonempty_context_count));
  candidate.context_edge_offsets.reserve (
    size_t (summary.nonempty_context_count));
  for (size_t context_id = 0;
       context_id < capture.contexts.size (); ++context_id) {
    const CudaAntennaM1DomainCell &cell =
      compact.cells [capture.contexts [context_id].cell_id];
    if (! cell.polygon_count) {
      continue;
    }
    candidate.metal_contexts.push_back (uint32_t (context_id));
    candidate.context_polygon_offsets.push_back (polygon_offset);
    candidate.context_edge_offsets.push_back (edge_offset);
    checked_accumulate (
      cell.polygon_count, polygon_offset,
      "materialized context polygon offset");
    checked_accumulate (
      cell.edge_count, edge_offset,
      "materialized context edge offset");
  }
  candidate.digest = compact.digest;

  std::array<uint8_t, 32> verified;
  if (! domain_profiles [domain].digest (candidate, verified) ||
      verified != compact.digest) {
    throw AntennaM1Decline (
      std::string ("M1 antenna ") + domain_profiles [domain].label +
      " materialized raw scene digest is inconsistent");
  }
  scene.swap (candidate);
}

std::string digest_hex (const std::array<uint8_t, 32> &digest)
{
  std::ostringstream text;
  text << std::hex << std::setfill ('0');
  for (size_t index = 0; index < digest.size (); ++index) {
    text << std::setw (2) << unsigned (digest [index]);
  }
  return text.str ();
}

static_assert (
  std::is_standard_layout<CudaAntennaM1DomainCell>::value &&
  std::is_trivially_copyable<CudaAntennaM1DomainCell>::value,
  "antenna domain-cell records must remain pointer-free POD");

} // anonymous namespace

CudaAntennaM1CaptureLimits::CudaAntennaM1CaptureLimits ()
  : scene (),
    max_total_stored_bytes (default_max_total_stored_bytes),
    max_total_expanded_geometry_bytes (
      default_max_total_expanded_geometry_bytes),
    max_estimated_peak_bytes (default_max_estimated_peak_bytes)
{
  //  nothing yet
}

CudaAntennaM1DomainScene::CudaAntennaM1DomainScene ()
  : flat_polygon_count (0), flat_edge_count (0),
    scene_left (0), scene_bottom (0), scene_right (0), scene_top (0),
    cells (), polygons (), edges (), digest ()
{
  //  nothing yet
}

void CudaAntennaM1DomainScene::swap (
  CudaAntennaM1DomainScene &other) noexcept
{
  using std::swap;
  swap (flat_polygon_count, other.flat_polygon_count);
  swap (flat_edge_count, other.flat_edge_count);
  swap (scene_left, other.scene_left);
  swap (scene_bottom, other.scene_bottom);
  swap (scene_right, other.scene_right);
  swap (scene_top, other.scene_top);
  cells.swap (other.cells);
  polygons.swap (other.polygons);
  edges.swap (other.edges);
  digest.swap (other.digest);
}

CudaAntennaM1DomainCensus::CudaAntennaM1DomainCensus ()
  : role (0), physical_layer (0), datatype (0), source_layer_index (0),
    stored_cell_count (0), stored_context_count (0),
    nonempty_context_count (0), stored_polygon_count (0),
    stored_edge_count (0), expanded_polygon_count (0),
    expanded_edge_count (0), stored_bytes (0),
    legacy_stored_bytes (0), expanded_geometry_bytes (0),
    scene_digest ()
{
  //  nothing yet
}

CudaAntennaM1Census::CudaAntennaM1Census ()
  : format_version (0), dbu_per_micron (0), root_cell (0), reserved (0),
    source_root_cell_index (0), shared_cell_count (0),
    shared_context_count (0), context_parent_record_count (0),
    context_parent_bytes (0), stored_cell_records (0),
    stored_context_records (0), nonempty_context_records (0),
    stored_polygon_count (0), stored_edge_count (0),
    expanded_polygon_count (0), expanded_edge_count (0),
    total_stored_bytes (0), legacy_stored_cell_records (0),
    legacy_stored_context_records (0), legacy_total_stored_bytes (0),
    total_expanded_geometry_bytes (0), estimated_peak_bytes (0),
    legacy_estimated_peak_bytes (0), domains (), hierarchy_digest (),
    capture_digest ()
{
  //  nothing yet
}

CudaAntennaM1Capture::CudaAntennaM1Capture ()
  : format_version (capture_format_version), dbu_per_micron (0),
    root_cell (0), reserved (0), source_root_cell_index (0),
    source_layer_indices (), source_cell_indices (), contexts (),
    domains (), context_parent_ids (), hierarchy_digest (), digest ()
{
  //  nothing yet
}

void CudaAntennaM1Capture::swap (
  CudaAntennaM1Capture &other) noexcept
{
  using std::swap;
  swap (format_version, other.format_version);
  swap (dbu_per_micron, other.dbu_per_micron);
  swap (root_cell, other.root_cell);
  swap (reserved, other.reserved);
  swap (source_root_cell_index, other.source_root_cell_index);
  source_layer_indices.swap (other.source_layer_indices);
  source_cell_indices.swap (other.source_cell_indices);
  contexts.swap (other.contexts);
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    domains [domain].swap (other.domains [domain]);
  }
  context_parent_ids.swap (other.context_parent_ids);
  hierarchy_digest.swap (other.hierarchy_digest);
  digest.swap (other.digest);
}

bool cuda_antenna_m1_build_capture (
  const db::DeepLayer &raw_poly,
  const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_contact,
  const db::DeepLayer &raw_metal1,
  const CudaAntennaM1CaptureLimits &limits,
  CudaAntennaM1Capture &capture,
  CudaAntennaM1Census &authenticated_census,
  std::string *decline_reason)
{
  try {
    const std::array<
      const db::DeepLayer *, CudaAntennaM1DomainCount> inputs = {{
        &raw_poly, &raw_active, &raw_nplus, &raw_nwell,
        &raw_contact, &raw_metal1
      }};
    validate_input_identity (inputs, limits.scene);
    if (! limits.max_total_stored_bytes ||
        ! limits.max_total_expanded_geometry_bytes ||
        ! limits.max_estimated_peak_bytes) {
      throw AntennaM1Decline (
        "an M1 antenna aggregate byte capacity is zero");
    }

    CudaAntennaM1Capture candidate;
    serialize_shared_capture (inputs, limits.scene, candidate);
    for (size_t domain = 0;
         domain < CudaAntennaM1DomainCount; ++domain) {
      const DomainSummary summary = derive_domain_summary (
        candidate, candidate.domains [domain], 0,
        domain_profiles [domain].label, false);
      candidate.domains [domain].digest =
        legacy_domain_digest (candidate, domain, summary);
    }
    candidate.hierarchy_digest = hierarchy_digest (candidate);

    CudaAntennaM1Census census;
    derive_census (candidate, census);
    validate_capture_limits (limits, census);
    candidate.digest = capture_digest (candidate, census);
    census.capture_digest = candidate.digest;

    capture.swap (candidate);
    authenticated_census = census;
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

bool cuda_antenna_m1_build_capture (
  const db::DeepLayer &raw_poly,
  const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_contact,
  const db::DeepLayer &raw_metal1,
  const CudaAntennaM1CaptureLimits &limits,
  CudaAntennaM1Capture &capture,
  std::string *decline_reason)
{
  CudaAntennaM1Census authenticated_census;
  return cuda_antenna_m1_build_capture (
    raw_poly, raw_active, raw_nplus, raw_nwell, raw_contact, raw_metal1,
    limits, capture, authenticated_census, decline_reason);
}

bool cuda_antenna_m1_materialize_domain_scene (
  const CudaAntennaM1Capture &capture,
  CudaAntennaM1Domain domain,
  CudaRawManhattanScene &scene,
  std::string *decline_reason)
{
  try {
    const size_t index = size_t (domain);
    if (index >= CudaAntennaM1DomainCount) {
      throw AntennaM1Decline (
        "M1 antenna domain ID is outside the fixed capture");
    }
    CudaAntennaM1Census census;
    derive_census (capture, census);
    DomainSummary summary;
    validate_domain_geometry (capture, index, summary);
    materialize_domain_scene (capture, index, summary, scene);
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

bool cuda_antenna_m1_capture_digest (
  const CudaAntennaM1Capture &capture,
  std::array<uint8_t, 32> &digest)
{
  try {
    CudaAntennaM1Census census;
    derive_census (capture, census);
    const std::array<uint8_t, 32> candidate =
      capture_digest (capture, census);
    digest = candidate;
    return true;
  } catch (...) {
    return false;
  }
}

bool cuda_antenna_m1_capture_census (
  const CudaAntennaM1Capture &capture,
  CudaAntennaM1Census &census,
  std::string *decline_reason)
{
  try {
    CudaAntennaM1Census candidate;
    derive_census (capture, candidate);
    candidate.capture_digest =
      capture_digest (capture, candidate);
    if (candidate.capture_digest != capture.digest) {
      throw AntennaM1Decline (
        "M1 antenna aggregate capture digest is inconsistent");
    }
    census = candidate;
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

std::string cuda_antenna_m1_census_text (
  const CudaAntennaM1Census &census)
{
  std::ostringstream text;
  text
    << "antenna_m1_capture"
    << " format=" << census.format_version
    << " dbu_per_micron=" << census.dbu_per_micron
    << " root=" << census.source_root_cell_index
    << " shared_cells=" << census.shared_cell_count
    << " shared_contexts=" << census.shared_context_count
    << " context_parent_records="
    << census.context_parent_record_count
    << " context_parent_bytes=" << census.context_parent_bytes
    << " stored_cell_records=" << census.stored_cell_records
    << " stored_context_records=" << census.stored_context_records
    << " derived_nonempty_context_records="
    << census.nonempty_context_records
    << " legacy_stored_cell_records="
    << census.legacy_stored_cell_records
    << " legacy_stored_context_records="
    << census.legacy_stored_context_records
    << " stored_polygons=" << census.stored_polygon_count
    << " stored_edges=" << census.stored_edge_count
    << " expanded_polygons=" << census.expanded_polygon_count
    << " expanded_edges=" << census.expanded_edge_count
    << " stored_bytes=" << census.total_stored_bytes
    << " legacy_stored_bytes=" << census.legacy_total_stored_bytes
    << " expanded_geometry_bytes="
    << census.total_expanded_geometry_bytes
    << " estimated_peak_bytes=" << census.estimated_peak_bytes
    << " legacy_estimated_peak_bytes="
    << census.legacy_estimated_peak_bytes
    << " hierarchy_sha256=" << digest_hex (census.hierarchy_digest)
    << " capture_sha256=" << digest_hex (census.capture_digest);
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    const CudaAntennaM1DomainCensus &record =
      census.domains [domain];
    text
      << " " << domain_profiles [domain].label
      << "{physical=" << record.physical_layer << "/"
      << record.datatype
      << ",internal=" << record.source_layer_index
      << ",stored_cells=" << record.stored_cell_count
      << ",shared_contexts=" << record.stored_context_count
      << ",derived_nonempty_contexts=" << record.nonempty_context_count
      << ",stored_polygons=" << record.stored_polygon_count
      << ",stored_edges=" << record.stored_edge_count
      << ",expanded_polygons=" << record.expanded_polygon_count
      << ",expanded_edges=" << record.expanded_edge_count
      << ",stored_bytes=" << record.stored_bytes
      << ",legacy_stored_bytes=" << record.legacy_stored_bytes
      << ",expanded_geometry_bytes=" << record.expanded_geometry_bytes
      << ",sha256=" << digest_hex (record.scene_digest)
      << "}";
  }
  return text.str ();
}

} // namespace db
