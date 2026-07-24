/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaPoly34.h"
#include "dbCudaPoly34Digest.h"
#include "dbCudaSpatialBackend.h"

#include "dbCell.h"
#include "dbDeepShapeStore.h"
#include "dbLayout.h"
#include "dbPolygonGenerators.h"
#include "dbPolygonTools.h"
#include "dbShapes.h"
#include "tlLog.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
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
const int64_t qualified_poly3_distance = 110;
const int64_t qualified_poly4_distance = 140;
const int64_t qualified_grid_cell = 2000;
const int64_t accepted_coordinate_magnitude = INT64_C (1000000000000);
const uint64_t default_max_contexts = UINT64_C (20000000);
const uint64_t default_max_flat_boxes = UINT64_C (100000000);
const uint64_t default_max_grid_cells = UINT64_C (100000000);
const uint64_t default_max_memberships = UINT64_C (500000000);
const uint64_t default_max_query_visits = UINT64_C (4000000000);
const uint64_t default_max_candidate_work = UINT64_C (500000000);
const unsigned maximum_hierarchy_depth = 512;

class Poly34Decline : public std::runtime_error
{
public:
  explicit Poly34Decline (const std::string &message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

uint64_t env_u64 (const char *name, uint64_t default_value)
{
  const char *value = std::getenv (name);
  if (! value || ! *value) {
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

bool env_enabled (const char *name)
{
  const char *value = std::getenv (name);
  return value && *value && std::strcmp (value, "0") != 0 &&
         std::strcmp (value, "false") != 0 &&
         std::strcmp (value, "off") != 0;
}

int32_t env_device ()
{
  return int32_t (std::min<uint64_t> (
    env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0),
    uint64_t (std::numeric_limits<int32_t>::max ())));
}

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

int64_t narrow_i64 (__int128 value, const char *what)
{
  if (value < -__int128 (accepted_coordinate_magnitude) ||
      value > __int128 (accepted_coordinate_magnitude)) {
    throw Poly34Decline (
      std::string (what) + " exceeds the qualified coordinate domain");
  }
  return int64_t (value);
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
    throw Poly34Decline ("invalid orthogonal transform code");
  }
  const Matrix &matrix = transforms [code];
  return std::make_pair (
    __int128 (matrix.xx) * x + __int128 (matrix.xy) * y,
    __int128 (matrix.yx) * x + __int128 (matrix.yy) * y);
}

uint32_t compose_transform (uint32_t outer, uint32_t inner)
{
  if (outer >= 8 || inner >= 8) {
    throw Poly34Decline (
      "invalid transform during POLY34 hierarchy expansion");
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
  throw Poly34Decline (
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

  void add (const klayout_cuda_spatial_poly34_box_v1 &box)
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

uint64_t vector_size_u64 (size_t size, const char *what)
{
  if (size > std::numeric_limits<uint64_t>::max ()) {
    throw Poly34Decline (std::string (what) + " exceeds uint64");
  }
  return uint64_t (size);
}

void append_box (
  const db::Box &box, uint64_t maximum,
  CudaPoly34Scene &scene, LocalBounds &bounds)
{
  if (box.empty () || scene.boxes.size () >= maximum ||
      scene.boxes.size () >= std::numeric_limits<uint32_t>::max ()) {
    throw Poly34Decline (
      "stored POLY34 box is empty or exceeds configured capacity");
  }
  const klayout_cuda_spatial_poly34_box_v1 record = {
    narrow_i64 (box.left (), "box left"),
    narrow_i64 (box.bottom (), "box bottom"),
    narrow_i64 (box.right (), "box right"),
    narrow_i64 (box.top (), "box top")
  };
  if (record.left >= record.right || record.bottom >= record.top) {
    throw Poly34Decline ("serialized POLY34 box is empty");
  }
  scene.boxes.push_back (record);
  bounds.add (record);
}

class BoxSink : public db::SimplePolygonSink
{
public:
  BoxSink (
    uint64_t maximum, CudaPoly34Scene &scene, LocalBounds &bounds)
    : m_maximum (maximum), m_scene (scene), m_bounds (bounds),
      m_area (0), m_count (0)
  {
    //  nothing yet
  }

  void put (const db::SimplePolygon &polygon)
  {
    if (! polygon.is_box ()) {
      throw Poly34Decline (
        "TD_simple emitted a non-box POLY34 primary component");
    }
    const db::Box box = polygon.box ();
    append_box (box, m_maximum, m_scene, m_bounds);
    m_area += __int128 (box.width ()) * box.height ();
    ++m_count;
  }

  __int128 area () const
  {
    return m_area;
  }

  uint64_t count () const
  {
    return m_count;
  }

private:
  uint64_t m_maximum;
  CudaPoly34Scene &m_scene;
  LocalBounds &m_bounds;
  __int128 m_area;
  uint64_t m_count;
};

db::Polygon checked_polygon (const db::Shape &shape)
{
  if (shape.prop_id () != 0 ||
      (! shape.is_box () && ! shape.is_polygon ())) {
    throw Poly34Decline (
      "POLY34 operand contains properties or a non-polygon shape");
  }
  db::Polygon polygon;
  if (! shape.polygon (polygon) || polygon.is_empty ()) {
    throw Poly34Decline ("POLY34 operand polygon is malformed or empty");
  }
  return polygon;
}

void append_primary_polygon (
  const db::Shape &shape, uint64_t maximum,
  CudaPoly34Scene &scene, LocalBounds &bounds)
{
  if (shape.is_box ()) {
    append_box (shape.box (), maximum, scene, bounds);
    return;
  }
  const db::Polygon polygon = checked_polygon (shape);
  if (! polygon.is_rectilinear ()) {
    throw Poly34Decline ("POLY34 primary polygon is non-Manhattan");
  }
  BoxSink sink (maximum, scene, bounds);
  db::decompose_trapezoids (polygon, db::TD_simple, sink);
  if (! sink.count () ||
      sink.area () != __int128 (polygon.area ())) {
    throw Poly34Decline (
      "POLY34 primary box decomposition failed exact area conservation");
  }
}

void append_gate_polygon (
  const db::Shape &shape, uint64_t maximum,
  CudaPoly34Scene &scene, LocalBounds &bounds)
{
  if (shape.is_box ()) {
    append_box (shape.box (), maximum, scene, bounds);
    return;
  }
  const db::Polygon polygon = checked_polygon (shape);
  if (! polygon.is_box ()) {
    throw Poly34Decline (
      "POLY34 GATE polygon is not an exact rectangle");
  }
  append_box (polygon.box (), maximum, scene, bounds);
}

void append_cell_layer (
  const db::Cell &cell, unsigned int layer, uint32_t domain,
  uint64_t maximum, CudaPoly34Scene &scene,
  klayout_cuda_spatial_poly34_domain_span_v1 &span,
  LocalBounds &bounds)
{
  if (domain >= KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT) {
    throw Poly34Decline ("invalid POLY34 geometry domain");
  }
  span.box_begin =
    vector_size_u64 (scene.boxes.size (), "POLY34 box begin");
  span.box_count = 0;
  span.reserved0 = 0;

  const db::Shapes &shapes = cell.shapes (layer);
  for (db::Shapes::shape_iterator shape =
         shapes.begin (db::ShapeIterator::All);
       ! shape.at_end (); ++shape) {
    const size_t before = scene.boxes.size ();
    if (domain == KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN) {
      append_gate_polygon (*shape, maximum, scene, bounds);
    } else {
      append_primary_polygon (*shape, maximum, scene, bounds);
    }
    const size_t added = scene.boxes.size () - before;
    if (added > std::numeric_limits<uint32_t>::max () - span.box_count) {
      throw Poly34Decline (
        "per-cell POLY34 box count exceeds uint32");
    }
    span.box_count += uint32_t (added);
  }
}

InstanceTemplate make_instance (
  const db::Instance &instance,
  const std::map<db::cell_index_type, uint32_t> &dense_cells)
{
  if (instance.prop_id () != 0 || instance.is_complex ()) {
    throw Poly34Decline (
      "POLY34 hierarchy has an instance property or complex transform");
  }
  const std::map<db::cell_index_type, uint32_t>::const_iterator child =
    dense_cells.find (instance.cell_index ());
  if (child == dense_cells.end ()) {
    throw Poly34Decline (
      "POLY34 hierarchy instance targets an unreachable cell");
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
    throw Poly34Decline (
      "POLY34 hierarchy has an irregular instance array");
  }
  if (delegate == 0) {
    a = db::Vector ();
    b = db::Vector ();
    na = nb = 1;
  }
  if (! na || ! nb ||
      na > std::numeric_limits<uint32_t>::max () ||
      nb > std::numeric_limits<uint32_t>::max ()) {
    throw Poly34Decline (
      "POLY34 hierarchy has an invalid array dimension");
  }

  const db::Trans &trans = instance.front ();
  const int transform_code = trans.rot ();
  if (transform_code < 0 || transform_code >= 8) {
    throw Poly34Decline (
      "POLY34 hierarchy has an invalid orthogonal transform");
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
    throw Poly34Decline (
      "POLY34 hierarchy has a malformed regular array");
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
    throw Poly34Decline (
      "POLY34 hierarchy exceeds the qualified recursion depth");
  }
  if (cell >= cells.size ()) {
    throw Poly34Decline (
      "POLY34 hierarchy references an invalid dense cell");
  }
  if (state [cell] == 1) {
    throw Poly34Decline ("POLY34 hierarchy contains a cycle");
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
      throw Poly34Decline (
        "expanded POLY34 hierarchy exceeds context capacity");
    }
  }
  state [cell] = 2;
  memo [cell] = total;
  return total;
}

void expand_contexts (
  uint32_t root, const std::vector<CellTemplate> &templates,
  uint64_t maximum,
  std::vector<klayout_cuda_spatial_poly34_context_v1> &contexts)
{
  std::vector<uint8_t> state (templates.size (), 0);
  std::vector<uint64_t> memo (templates.size (), 0);
  const uint64_t expected = subtree_context_count (
    root, templates, state, memo, maximum, 0);
  if (expected > std::numeric_limits<uint32_t>::max () ||
      expected > contexts.max_size ()) {
    throw Poly34Decline (
      "expanded POLY34 hierarchy cannot use uint32 context IDs");
  }

  contexts.reserve (size_t (expected));
  contexts.push_back (
    klayout_cuda_spatial_poly34_context_v1 { 0, 0, root, 0 });
  for (size_t parent_id = 0; parent_id < contexts.size (); ++parent_id) {
    const klayout_cuda_spatial_poly34_context_v1 parent =
      contexts [parent_id];
    if (parent.cell_id >= templates.size ()) {
      throw Poly34Decline (
        "POLY34 context references an invalid cell");
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
            klayout_cuda_spatial_poly34_context_v1 {
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
    throw Poly34Decline (
      "POLY34 context expansion disagrees with checked census");
  }
}

std::pair<int64_t, int64_t> transform_point (
  const klayout_cuda_spatial_poly34_context_v1 &context,
  int64_t x, int64_t y)
{
  const std::pair<__int128, __int128> point =
    transform_128 (context.transform_code, x, y);
  return std::make_pair (
    narrow_i64 (point.first + context.tx, "world box x"),
    narrow_i64 (point.second + context.ty, "world box y"));
}

void add_world_bounds (
  CudaPoly34Scene &scene,
  const klayout_cuda_spatial_poly34_context_v1 &context,
  const LocalBounds &local)
{
  if (! local.valid) {
    return;
  }
  const int64_t xs [4] =
    { local.left, local.left, local.right, local.right };
  const int64_t ys [4] =
    { local.bottom, local.top, local.bottom, local.top };
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

void add_context_domain (
  uint32_t context_id, uint32_t local_count, uint64_t maximum,
  std::vector<uint32_t> &contexts, std::vector<uint64_t> &offsets,
  uint64_t &flat_count)
{
  if (! local_count) {
    return;
  }
  contexts.push_back (context_id);
  offsets.push_back (flat_count);
  if (! checked_add_u64 (flat_count, local_count, flat_count) ||
      flat_count > maximum ||
      flat_count > std::numeric_limits<uint32_t>::max ()) {
    throw Poly34Decline (
      "flattened POLY34 box count exceeds configured capacity");
  }
}

void derive_context_lists_and_bounds (
  CudaPoly34Scene &scene, const std::vector<LocalBounds> &bounds,
  uint64_t maximum)
{
  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const klayout_cuda_spatial_poly34_context_v1 &context =
      scene.contexts [context_id];
    if (context.cell_id >= scene.cells.size ()) {
      throw Poly34Decline (
        "POLY34 context references an invalid dense cell");
    }
    const klayout_cuda_spatial_poly34_cell_v1 &cell =
      scene.cells [context.cell_id];
    add_context_domain (
      uint32_t (context_id),
      cell.domains [KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN].box_count,
      maximum, scene.poly_contexts, scene.poly_offsets,
      scene.flat_boxes [KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN]);
    add_context_domain (
      uint32_t (context_id),
      cell.domains [KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN].box_count,
      maximum, scene.active_contexts, scene.active_offsets,
      scene.flat_boxes [KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN]);
    add_context_domain (
      uint32_t (context_id),
      cell.domains [KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN].box_count,
      maximum, scene.gate_contexts, scene.gate_offsets,
      scene.flat_boxes [KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN]);
    add_world_bounds (scene, context, bounds [context.cell_id]);
  }

  uint64_t total = 0;
  for (uint32_t domain = 0;
       domain < KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT; ++domain) {
    if (! checked_add_u64 (total, scene.flat_boxes [domain], total)) {
      throw Poly34Decline ("aggregate POLY34 flat count overflows");
    }
  }
  if (total > maximum ||
      ! scene.have_scene_box ||
      scene.poly_contexts.empty () ||
      scene.active_contexts.empty () ||
      scene.gate_contexts.empty ()) {
    throw Poly34Decline (
      "POLY34 scene is empty or exceeds aggregate flat capacity");
  }
}

void validate_inputs (
  const db::DeepLayer &poly, const db::DeepLayer &active,
  const db::DeepLayer &gate, const CudaPoly34BuildSpec &spec,
  const CudaPoly34SceneLimits &limits)
{
  if (! spec.poly_is_exact_merged ||
      ! spec.active_is_exact_merged ||
      ! spec.gate_is_exact_merged ||
      ! spec.freepdk45_layer_contract) {
    throw Poly34Decline (
      "integration did not assert exact merged operands and the qualified "
      "FreePDK45 layer contract");
  }
  if (! limits.max_contexts || ! limits.max_flat_boxes ||
      limits.max_contexts > std::numeric_limits<uint32_t>::max ()) {
    throw Poly34Decline (
      "POLY34 scene capacity is zero or exceeds context IDs");
  }
  if (poly.store () != active.store () ||
      poly.store () != gate.store () ||
      &poly.layout () != &active.layout () ||
      &poly.layout () != &gate.layout () ||
      poly.layout_index () != active.layout_index () ||
      poly.layout_index () != gate.layout_index () ||
      poly.initial_cell ().cell_index () !=
        active.initial_cell ().cell_index () ||
      poly.initial_cell ().cell_index () !=
        gate.initial_cell ().cell_index ()) {
    throw Poly34Decline (
      "POLY34 operands do not share one store, layout and top cell");
  }
  if (poly.breakout_cells () != 0 ||
      active.breakout_cells () != 0 ||
      gate.breakout_cells () != 0) {
    throw Poly34Decline ("POLY34 scene has hierarchy breakout cells");
  }
  if (poly.layer () == active.layer () ||
      poly.layer () == gate.layer () ||
      active.layer () == gate.layer () ||
      poly.layer () >= poly.layout ().layers () ||
      active.layer () >= active.layout ().layers () ||
      gate.layer () >= gate.layout ().layers ()) {
    throw Poly34Decline (
      "POLY34 operands do not have three valid distinct layer identities");
  }
  if (poly.layout ().dbu () != 0.0005) {
    throw Poly34Decline (
      "POLY34 scene DBU is not the qualified 0.5 nm");
  }
}

CudaPoly34Scene serialize_live_scene (
  const db::DeepLayer &poly, const db::DeepLayer &active,
  const db::DeepLayer &gate, uint64_t max_contexts,
  uint64_t max_flat_boxes)
{
  const db::Layout &layout = poly.layout ();
  const db::cell_index_type top = poly.initial_cell ().cell_index ();
  std::set<db::cell_index_type> reachable;
  reachable.insert (top);
  poly.initial_cell ().collect_called_cells (reachable);
  if (reachable.empty () ||
      reachable.size () > std::numeric_limits<uint32_t>::max () ||
      reachable.size () > max_contexts) {
    throw Poly34Decline (
      "reachable POLY34 hierarchy has an invalid cell count");
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
    throw Poly34Decline (
      "POLY34 initial cell is absent from hierarchy census");
  }

  CudaPoly34Scene scene;
  scene.root_cell = root->second;
  scene.store_identity =
    uint64_t (reinterpret_cast<std::uintptr_t> (poly.store ()));
  scene.layout_identity =
    uint64_t (reinterpret_cast<std::uintptr_t> (&poly.layout ()));
  scene.top_cell_identity = uint64_t (top);
  scene.layer_ids [KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN] = poly.layer ();
  scene.layer_ids [KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN] =
    active.layer ();
  scene.layer_ids [KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN] = gate.layer ();
  scene.cells.resize (reachable.size ());
  std::vector<LocalBounds> bounds (reachable.size ());
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

    klayout_cuda_spatial_poly34_cell_v1 record;
    std::memset (&record, 0, sizeof (record));
    record.source_cell_index = uint64_t (*source);
    append_cell_layer (
      cell, poly.layer (), KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN,
      max_flat_boxes, scene,
      record.domains [KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN],
      bounds [cell_id]);
    append_cell_layer (
      cell, active.layer (), KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN,
      max_flat_boxes, scene,
      record.domains [KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN],
      bounds [cell_id]);
    append_cell_layer (
      cell, gate.layer (), KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN,
      max_flat_boxes, scene,
      record.domains [KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN],
      bounds [cell_id]);
    scene.cells [cell_id] = record;
  }
  expand_contexts (
    scene.root_cell, templates, max_contexts, scene.contexts);
  derive_context_lists_and_bounds (scene, bounds, max_flat_boxes);
  return scene;
}

void validate_record_layouts ()
{
  static_assert (
    std::is_standard_layout<
      klayout_cuda_spatial_poly34_context_v1>::value,
    "POLY34 context ABI must be standard-layout");
  static_assert (
    std::is_standard_layout<
      klayout_cuda_spatial_poly34_cell_v1>::value,
    "POLY34 cell ABI must be standard-layout");
  static_assert (
    std::is_standard_layout<
      klayout_cuda_spatial_poly34_box_v1>::value,
    "POLY34 box ABI must be standard-layout");
  static_assert (
    sizeof (klayout_cuda_spatial_poly34_context_v1) == 24,
    "unexpected POLY34 context ABI size");
  static_assert (
    sizeof (klayout_cuda_spatial_poly34_domain_span_v1) == 16,
    "unexpected POLY34 domain-span ABI size");
  static_assert (
    sizeof (klayout_cuda_spatial_poly34_cell_v1) == 56,
    "unexpected POLY34 cell ABI size");
  static_assert (
    sizeof (klayout_cuda_spatial_poly34_box_v1) == 32,
    "unexpected POLY34 box ABI size");
}

void set_reason (std::string *reason, const char *message)
{
  if (! reason) {
    return;
  }
  try {
    *reason = message ? message : "unknown POLY34 exception";
  } catch (...) {
    //  Diagnostics cannot turn a fail-closed decline into an exception.
  }
}

} // anonymous namespace

CudaPoly34SceneLimits::CudaPoly34SceneLimits ()
  : max_contexts (default_max_contexts),
    max_flat_boxes (default_max_flat_boxes)
{
  //  nothing yet
}

CudaPoly34BuildSpec::CudaPoly34BuildSpec ()
  : poly_is_exact_merged (false), active_is_exact_merged (false),
    gate_is_exact_merged (false), freepdk45_layer_contract (false)
{
  //  nothing yet
}

CudaPoly34Scene::CudaPoly34Scene ()
  : root_cell (0), have_scene_box (false),
    scene_left (0), scene_bottom (0), scene_right (0), scene_top (0),
    store_identity (0), layout_identity (0), top_cell_identity (0)
{
  std::fill (
    flat_boxes,
    flat_boxes + KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT,
    uint64_t (0));
  std::fill (
    layer_ids,
    layer_ids + KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT,
    uint32_t (0));
}

void CudaPoly34Scene::swap (CudaPoly34Scene &other) noexcept
{
  contexts.swap (other.contexts);
  poly_contexts.swap (other.poly_contexts);
  poly_offsets.swap (other.poly_offsets);
  active_contexts.swap (other.active_contexts);
  active_offsets.swap (other.active_offsets);
  gate_contexts.swap (other.gate_contexts);
  gate_offsets.swap (other.gate_offsets);
  cells.swap (other.cells);
  boxes.swap (other.boxes);
  std::swap (root_cell, other.root_cell);
  for (uint32_t domain = 0;
       domain < KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT; ++domain) {
    std::swap (flat_boxes [domain], other.flat_boxes [domain]);
    std::swap (layer_ids [domain], other.layer_ids [domain]);
  }
  std::swap (have_scene_box, other.have_scene_box);
  std::swap (scene_left, other.scene_left);
  std::swap (scene_bottom, other.scene_bottom);
  std::swap (scene_right, other.scene_right);
  std::swap (scene_top, other.scene_top);
  std::swap (store_identity, other.store_identity);
  std::swap (layout_identity, other.layout_identity);
  std::swap (top_cell_identity, other.top_cell_identity);
}

bool cuda_poly34_build_scene (
  const db::DeepLayer &merged_poly, const db::DeepLayer &merged_active,
  const db::DeepLayer &merged_gate, const CudaPoly34BuildSpec &spec,
  const CudaPoly34SceneLimits &limits, CudaPoly34Scene &scene,
  std::string *decline_reason)
{
  try {
    validate_record_layouts ();
    validate_inputs (
      merged_poly, merged_active, merged_gate, spec, limits);
    CudaPoly34Scene built = serialize_live_scene (
      merged_poly, merged_active, merged_gate,
      limits.max_contexts, limits.max_flat_boxes);
    scene.swap (built);
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (
      decline_reason, "unknown POLY34 scene-lowering exception");
  }
  return false;
}

bool cuda_poly34_try_empty (
  const db::DeepLayer &merged_poly, const db::DeepLayer &merged_active,
  const db::DeepLayer &merged_gate, const CudaPoly34BuildSpec &spec)
{
  const bool telemetry = env_enabled ("KLAYOUT_CUDA_POLY34_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    validate_record_layouts ();
    if (! db::cuda_spatial_poly34_requested ()) {
      return false;
    }

    const uint64_t max_contexts = env_u64 (
      "KLAYOUT_CUDA_POLY34_MAX_CONTEXTS", default_max_contexts);
    const uint64_t max_flat_boxes = env_u64 (
      "KLAYOUT_CUDA_POLY34_MAX_FLAT_BOXES", default_max_flat_boxes);
    const uint64_t max_grid_cells = env_u64 (
      "KLAYOUT_CUDA_POLY34_MAX_GRID_CELLS", default_max_grid_cells);
    const uint64_t max_poly_memberships = env_u64 (
      "KLAYOUT_CUDA_POLY34_MAX_POLY_MEMBERSHIPS",
      default_max_memberships);
    const uint64_t max_active_memberships = env_u64 (
      "KLAYOUT_CUDA_POLY34_MAX_ACTIVE_MEMBERSHIPS",
      default_max_memberships);
    const uint64_t max_query_visits = env_u64 (
      "KLAYOUT_CUDA_POLY34_MAX_QUERY_VISITS",
      default_max_query_visits);
    const uint64_t max_candidate_work = env_u64 (
      "KLAYOUT_CUDA_POLY34_MAX_CANDIDATE_WORK",
      default_max_candidate_work);
    if (! max_contexts || ! max_flat_boxes || ! max_grid_cells ||
        ! max_poly_memberships || ! max_active_memberships ||
        ! max_query_visits || ! max_candidate_work ||
        max_contexts > std::numeric_limits<uint32_t>::max () ||
        max_grid_cells > std::numeric_limits<uint32_t>::max () ||
        max_poly_memberships > std::numeric_limits<uint32_t>::max () ||
        max_active_memberships > std::numeric_limits<uint32_t>::max ()) {
      throw Poly34Decline (
        "a POLY34 capacity is zero or exceeds first-format IDs");
    }

    CudaPoly34SceneLimits limits;
    limits.max_contexts = max_contexts;
    limits.max_flat_boxes = max_flat_boxes;
    CudaPoly34Scene scene;
    std::string reason;
    if (! cuda_poly34_build_scene (
          merged_poly, merged_active, merged_gate, spec, limits,
          scene, &reason)) {
      throw Poly34Decline (
        reason.empty () ? "unable to build live POLY34 scene" : reason);
    }

    klayout_cuda_spatial_poly34_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode = KLAYOUT_CUDA_SPATIAL_POLY34_TERMINAL_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_POLY34_QUALIFIED_OPTIONS;
    request.format_version = scene_format_version;
    request.dbu_per_micron = qualified_dbu_per_micron;
    request.root_cell = scene.root_cell;
    request.requested_mask = KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES;
    request.device = env_device ();
    request.poly3_distance = qualified_poly3_distance;
    request.poly4_distance = qualified_poly4_distance;
    request.grid_cell_size = qualified_grid_cell;
    request.store_identity = scene.store_identity;
    request.layout_identity = scene.layout_identity;
    request.top_cell_identity = scene.top_cell_identity;
    request.poly_layer_id =
      scene.layer_ids [KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN];
    request.active_layer_id =
      scene.layer_ids [KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN];
    request.gate_layer_id =
      scene.layer_ids [KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN];
    request.contexts = scene.contexts.data ();
    request.context_count = scene.contexts.size ();
    request.context_record_bytes =
      sizeof (klayout_cuda_spatial_poly34_context_v1);
    request.poly_contexts = scene.poly_contexts.data ();
    request.poly_context_count = scene.poly_contexts.size ();
    request.poly_offsets = scene.poly_offsets.data ();
    request.poly_offset_count = scene.poly_offsets.size ();
    request.active_contexts = scene.active_contexts.data ();
    request.active_context_count = scene.active_contexts.size ();
    request.active_offsets = scene.active_offsets.data ();
    request.active_offset_count = scene.active_offsets.size ();
    request.gate_contexts = scene.gate_contexts.data ();
    request.gate_context_count = scene.gate_contexts.size ();
    request.gate_offsets = scene.gate_offsets.data ();
    request.gate_offset_count = scene.gate_offsets.size ();
    request.cells = scene.cells.data ();
    request.cell_count = scene.cells.size ();
    request.cell_record_bytes =
      sizeof (klayout_cuda_spatial_poly34_cell_v1);
    request.boxes = scene.boxes.data ();
    request.box_count = scene.boxes.size ();
    request.box_record_bytes =
      sizeof (klayout_cuda_spatial_poly34_box_v1);
    request.flat_poly_box_count =
      scene.flat_boxes [KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN];
    request.flat_active_box_count =
      scene.flat_boxes [KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN];
    request.flat_gate_box_count =
      scene.flat_boxes [KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN];
    request.scene_left = scene.scene_left;
    request.scene_bottom = scene.scene_bottom;
    request.scene_right = scene.scene_right;
    request.scene_top = scene.scene_top;
    request.max_contexts = max_contexts;
    request.max_flat_boxes = max_flat_boxes;
    request.max_grid_cells = max_grid_cells;
    request.max_poly_memberships = max_poly_memberships;
    request.max_active_memberships = max_active_memberships;
    request.max_query_visits = max_query_visits;
    request.max_candidate_work = max_candidate_work;
    request.max_candidates_per_gate = 64;

    std::array<uint8_t, 32> digest;
    if (! db::cuda_poly34_digest::request_digest (request, digest)) {
      throw Poly34Decline ("unable to digest live POLY34 request");
    }
    std::copy (digest.begin (), digest.end (), request.scene_digest);

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const double lower_ms =
      std::chrono::duration<double, std::milli> (
        call_begin - begin).count ();
    const db::CudaPoly34Attempt attempt =
      db::cuda_spatial_try_poly34_empty (request);
    const std::chrono::steady_clock::time_point end =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << "CUDA POLY.3/.4 live lowering:"
               << " contexts=" << request.context_count
               << " cells=" << request.cell_count
               << " stored_boxes=" << request.box_count
               << " poly_boxes=" << request.flat_poly_box_count
               << " active_boxes=" << request.flat_active_box_count
               << " gate_boxes=" << request.flat_gate_box_count
               << " lower_ms=" << lower_ms
               << " call_ms="
               << std::chrono::duration<double, std::milli> (
                    end - call_begin).count ()
               << " live_total_ms="
               << std::chrono::duration<double, std::milli> (
                    end - begin).count ();
    }
    return attempt.disposition == db::CudaPoly34Attempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << "CUDA POLY.3/.4 live lowering:"
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << "CUDA POLY.3/.4 live lowering:"
                 << " outcome=cpu-fallback message=unknown exception";
      } catch (...) {
        //  Telemetry must never turn a speculative decline into an error.
      }
    }
  }
  return false;
}

} // namespace db
