/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM4.h"

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
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace db
{

namespace
{

const uint32_t capture_format_version = 1;
const uint32_t raw_scene_format_version = 1;
const uint32_t qualified_dbu_per_micron = 2000;
const int64_t accepted_coordinate_magnitude = INT64_C (1000000000000);

const uint64_t default_max_total_stored_bytes =
  UINT64_C (4) * 1024 * 1024 * 1024;
const uint64_t default_max_total_expanded_geometry_bytes =
  UINT64_C (8) * 1024 * 1024 * 1024;
const uint64_t default_max_estimated_peak_bytes =
  UINT64_C (12) * 1024 * 1024 * 1024;

struct DomainProfile
{
  uint32_t role;
  uint32_t physical_layer;
  uint32_t datatype;
  const char *label;
  char digest_magic [8];
};

const DomainProfile domain_profiles [CudaAntennaM4DomainCount] = {
  {
    CudaAntennaM4Poly, 9, 0, "POLY",
    { 'K', 'P', 'O', 'L', 'Y', '0', '0', '1' }
  },
  {
    CudaAntennaM4Active, 1, 0, "ACTIVE",
    { 'K', 'A', 'R', 'A', 'W', '0', '0', '1' }
  },
  {
    CudaAntennaM4Nplus, 4, 0, "NPLUS",
    { 'K', 'N', 'P', 'L', 'S', '0', '0', '1' }
  },
  {
    CudaAntennaM4Nwell, 3, 0, "NWELL",
    { 'K', 'N', 'W', 'E', 'L', '0', '0', '1' }
  },
  {
    CudaAntennaM4Contact, 10, 0, "CONTACT",
    { 'K', 'C', 'R', 'A', 'W', '0', '0', '1' }
  },
  {
    CudaAntennaM4Metal1, 11, 0, "M1",
    { 'K', 'M', '1', 'R', 'A', 'W', '0', '1' }
  },
  {
    CudaAntennaM4Via1, 12, 0, "VIA1",
    { 'K', 'V', '1', 'R', 'A', 'W', '0', '1' }
  },
  {
    CudaAntennaM4Metal2, 13, 0, "M2",
    { 'K', 'M', '2', 'R', 'A', 'W', '0', '1' }
  },
  {
    CudaAntennaM4Via2, 14, 0, "VIA2",
    { 'K', 'V', '2', 'R', 'A', 'W', '0', '1' }
  },
  {
    CudaAntennaM4Metal3, 15, 0, "M3",
    { 'K', 'M', '3', 'R', 'A', 'W', '0', '1' }
  },
  {
    CudaAntennaM4Via3, 16, 0, "VIA3",
    { 'K', 'V', '3', 'R', 'A', 'W', '0', '1' }
  },
  {
    CudaAntennaM4Metal4, 17, 0, "M4",
    { 'K', 'M', '4', 'R', 'A', 'W', '0', '1' }
  }
};

class AntennaM4Decline
  : public std::runtime_error
{
public:
  explicit AntennaM4Decline (const std::string &message)
    : std::runtime_error (message)
  {
    //  nothing yet
  }
};

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
    throw AntennaM4Decline (
      std::string (what) + " cannot be represented by uint64");
  }
  return uint64_t (value);
}

void checked_accumulate (
  uint64_t value, uint64_t &total, const char *what)
{
  if (! checked_add_u64 (total, value, total)) {
    throw AntennaM4Decline (std::string (what) + " overflows uint64");
  }
}

uint64_t checked_array_bytes (
  uint64_t count, uint64_t stride, const char *what)
{
  uint64_t result = 0;
  if (! checked_multiply_u64 (count, stride, result)) {
    throw AntennaM4Decline (std::string (what) + " overflows uint64");
  }
  return result;
}

void require_room (
  uint64_t current, uint64_t additional, uint64_t maximum,
  const char *what)
{
  uint64_t total = 0;
  if (! checked_add_u64 (current, additional, total) || total > maximum) {
    throw AntennaM4Decline (
      std::string (what) + " exceeds the configured capacity");
  }
}

int64_t narrow_i64 (__int128 value, const char *what)
{
  if (value < std::numeric_limits<int64_t>::min () ||
      value > std::numeric_limits<int64_t>::max ()) {
    throw AntennaM4Decline (
      std::string (what) + " overflows signed int64");
  }
  const int64_t result = int64_t (value);
  if (result < -accepted_coordinate_magnitude ||
      result > accepted_coordinate_magnitude) {
    throw AntennaM4Decline (
      std::string (what) + " exceeds the qualified coordinate domain");
  }
  return result;
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
    throw AntennaM4Decline ("invalid orthogonal transform code");
  }
  const Matrix &matrix = transforms [code];
  return std::make_pair (
    __int128 (matrix.xx) * x + __int128 (matrix.xy) * y,
    __int128 (matrix.yx) * x + __int128 (matrix.yy) * y);
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

void validate_scene_limits (const CudaM1WidthSpaceSceneLimits &limits)
{
  if (! limits.max_cells || ! limits.max_contexts ||
      ! limits.max_stored_polygons || ! limits.max_stored_edges ||
      ! limits.max_flat_polygons || ! limits.max_flat_edges) {
    throw AntennaM4Decline ("a Manhattan scene capacity is zero");
  }
}

void validate_input_identity (
  const std::array<const db::DeepLayer *, CudaAntennaM4DomainCount>
    &inputs,
  const CudaAntennaM4CaptureLimits &limits)
{
  validate_scene_limits (limits.scene);
  if (! limits.max_total_stored_bytes ||
      ! limits.max_total_expanded_geometry_bytes ||
      ! limits.max_estimated_peak_bytes) {
    throw AntennaM4Decline (
      "an M1-through-M4 antenna aggregate byte capacity is zero");
  }

  const db::DeepLayer &reference = *inputs [0];
  std::set<unsigned int> layers;
  for (size_t domain = 0;
       domain < CudaAntennaM4DomainCount; ++domain) {
    const db::DeepLayer &candidate = *inputs [domain];
    const DomainProfile &profile = domain_profiles [domain];
    if (candidate.store () != reference.store () ||
        &candidate.layout () != &reference.layout () ||
        candidate.layout_index () != reference.layout_index () ||
        candidate.initial_cell ().cell_index () !=
          reference.initial_cell ().cell_index ()) {
      throw AntennaM4Decline (
        "M1-through-M4 antenna raw domains do not share one hierarchy");
    }
    if (! layers.insert (candidate.layer ()).second) {
      throw AntennaM4Decline (
        "M1-through-M4 antenna raw domains do not use twelve distinct "
        "layers");
    }
    if (candidate.breakout_cells () != 0) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + profile.label +
        " scene has hierarchy breakout cells");
    }
    if (candidate.layout ().dbu () != 0.0005) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + profile.label +
        " scene DBU is not the qualified 0.5 nm");
    }
    if (! candidate.layout ().is_valid_layer (candidate.layer ())) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + profile.label +
        " scene layer index is not a valid physical layer");
    }
    const db::LayerProperties &properties =
      candidate.layout ().get_properties (candidate.layer ());
    if (! properties.log_equal (
          db::LayerProperties (
            profile.physical_layer, profile.datatype))) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + profile.label +
        " scene is not physical FreePDK45 layer " +
        std::to_string (profile.physical_layer) + "/" +
        std::to_string (profile.datatype));
    }
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
    throw AntennaM4Decline ("Manhattan scene polygon has properties");
  }
  if (! shape.is_box () && ! shape.is_polygon ()) {
    throw AntennaM4Decline (
      "Manhattan scene layer contains a non-polygon shape");
  }

  db::Polygon polygon;
  if (! shape.polygon (polygon)) {
    throw AntennaM4Decline ("Manhattan scene polygon is malformed");
  }
  if (polygon.holes () != 0) {
    throw AntennaM4Decline (
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
      throw AntennaM4Decline (
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
    throw AntennaM4Decline (
      "Manhattan scene polygon is too small, empty, or not clockwise");
  }
  const cuda_manhattan_contour::ValidationResult contour_result =
    contour_cache.validate_contour (contour);
  if (contour_result ==
      cuda_manhattan_contour::ValidationResult::OpenContour) {
    throw AntennaM4Decline ("Manhattan scene polygon contour is open");
  }
  if (contour_result !=
      cuda_manhattan_contour::ValidationResult::Valid) {
    throw AntennaM4Decline (
      "Manhattan scene polygon contour self-intersects");
  }
  if (contour.size () > std::numeric_limits<uint32_t>::max ()) {
    throw AntennaM4Decline (
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
    throw AntennaM4Decline (
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
  record.edge_begin = size_u64 (scene.edges.size (), "cell edge begin");

  uint32_t polygon_id = 0;
  const db::Shapes &shapes = cell.shapes (layer);
  for (db::Shapes::shape_iterator shape =
         shapes.begin (db::ShapeIterator::All);
       ! shape.at_end (); ++shape) {
    if (shape->is_text ()) {
      continue;
    }
    if (polygon_id == std::numeric_limits<uint32_t>::max ()) {
      throw AntennaM4Decline (
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
    throw AntennaM4Decline (
      "per-cell Manhattan polygon or edge count exceeds uint32");
  }
  record.polygon_count = uint32_t (polygon_count);
  record.edge_count = uint32_t (edge_count);
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
  const CudaAntennaM1Capture &hierarchy,
  const CudaAntennaM1DomainScene &scene,
  const CudaM1WidthSpaceSceneLimits *limits,
  const char *label,
  bool derive_world_bounds)
{
  DomainSummary summary;
  bool have_bounds = ! derive_world_bounds;
  if (! derive_world_bounds) {
    summary.scene_left = scene.scene_left;
    summary.scene_bottom = scene.scene_bottom;
    summary.scene_right = scene.scene_right;
    summary.scene_top = scene.scene_top;
  }
  for (size_t context_id = 0;
       context_id < hierarchy.contexts.size (); ++context_id) {
    const CudaM1WidthSpaceContext &context =
      hierarchy.contexts [context_id];
    if (context.cell_id >= scene.cells.size ()) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + label +
        " context references an invalid dense cell");
    }
    const CudaAntennaM1DomainCell &cell =
      scene.cells [context.cell_id];
    if (! cell.polygon_count) {
      continue;
    }
    if (context_id > std::numeric_limits<uint32_t>::max ()) {
      throw AntennaM4Decline ("nonempty context ID exceeds uint32");
    }
    checked_accumulate (
      1, summary.nonempty_context_count, "nonempty context count");
    checked_accumulate (
      cell.polygon_count, summary.flat_polygon_count,
      "flattened Manhattan polygon count");
    checked_accumulate (
      cell.edge_count, summary.flat_edge_count,
      "flattened Manhattan edge count");
    if (limits &&
        (summary.flat_polygon_count > limits->max_flat_polygons ||
         summary.flat_edge_count > limits->max_flat_edges)) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + label +
        " flattened geometry exceeds the configured capacity");
    }

    if (derive_world_bounds) {
      uint64_t polygon_end = 0;
      if (! checked_add_u64 (
            cell.polygon_begin, cell.polygon_count, polygon_end) ||
          polygon_end > scene.polygons.size ()) {
        throw AntennaM4Decline (
          std::string ("M1-through-M4 antenna ") + label +
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
    throw AntennaM4Decline (
      std::string ("M1-through-M4 antenna ") + label +
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

void serialize_upper_domain (
  const db::Layout &layout, unsigned int layer,
  const CudaM1WidthSpaceSceneLimits &limits,
  const CudaAntennaM1Capture &hierarchy,
  CudaAntennaM1DomainScene &scene)
{
  scene.cells.resize (hierarchy.source_cell_indices.size ());
  cuda_manhattan_contour::TranslationValidationCache<
    CudaM1WidthSpaceEdge> contour_cache;
  for (size_t cell_id = 0;
       cell_id < hierarchy.source_cell_indices.size (); ++cell_id) {
    const uint64_t source = hierarchy.source_cell_indices [cell_id];
    if (source > std::numeric_limits<db::cell_index_type>::max () ||
        ! layout.is_valid_cell_index (db::cell_index_type (source))) {
      throw AntennaM4Decline (
        "shared source-cell identity is invalid in the input layout");
    }
    CudaAntennaM1DomainCell record = {};
    append_cell_layer (
      layout.cell (db::cell_index_type (source)), layer, limits,
      scene, record, contour_cache);
    scene.cells [cell_id] = record;
  }
}

bool checked_range (uint64_t begin, uint64_t count, uint64_t size)
{
  uint64_t end = 0;
  return checked_add_u64 (begin, count, end) && end <= size;
}

void validate_upper_domain_geometry (
  const CudaAntennaM4Capture &capture, size_t upper,
  DomainSummary &summary)
{
  const size_t domain = CudaAntennaM1DomainCount + upper;
  const DomainProfile &profile = domain_profiles [domain];
  const CudaAntennaM1DomainScene &scene =
    capture.upper_domains [upper];
  if (scene.cells.size () != capture.lower.source_cell_indices.size () ||
      scene.polygons.empty () || scene.edges.empty () ||
      scene.scene_left >= scene.scene_right ||
      scene.scene_bottom >= scene.scene_top) {
    throw AntennaM4Decline (
      std::string ("M1-through-M4 antenna ") + profile.label +
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
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + profile.label +
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
        throw AntennaM4Decline (
          std::string ("M1-through-M4 antenna ") + profile.label +
          " polygon range is inconsistent");
      }
      uint64_t polygon_end = 0;
      if (! checked_add_u64 (
            polygon.edge_begin, polygon.edge_count, polygon_end)) {
        throw AntennaM4Decline (
          std::string ("M1-through-M4 antenna ") + profile.label +
          " polygon edge range overflows");
      }
      for (uint64_t edge_id = polygon.edge_begin;
           edge_id < polygon_end; ++edge_id) {
        const CudaM1WidthSpaceEdge &edge =
          scene.edges [size_t (edge_id)];
        if ((edge.x1 == edge.x2 && edge.y1 == edge.y2) ||
            ! (edge.x1 == edge.x2 || edge.y1 == edge.y2)) {
          throw AntennaM4Decline (
            std::string ("M1-through-M4 antenna ") + profile.label +
            " edge is degenerate or non-Manhattan");
        }
        const uint64_t following =
          edge_id + 1 == polygon_end ? polygon.edge_begin : edge_id + 1;
        const CudaM1WidthSpaceEdge &next =
          scene.edges [size_t (following)];
        if (edge.x2 != next.x1 || edge.y2 != next.y1) {
          throw AntennaM4Decline (
            std::string ("M1-through-M4 antenna ") + profile.label +
            " polygon contour is open");
        }
      }
      cell_edge = polygon_end;
    }
    uint64_t cell_edge_end = 0;
    if (! checked_add_u64 (
          cell.edge_begin, cell.edge_count, cell_edge_end) ||
        cell_edge != cell_edge_end) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + profile.label +
        " cell edge census is inconsistent");
    }
    checked_accumulate (
      cell.polygon_count, next_polygon, "stored polygon count");
    checked_accumulate (
      cell.edge_count, next_edge, "stored edge count");
  }
  if (next_polygon != scene.polygons.size () ||
      next_edge != scene.edges.size ()) {
    throw AntennaM4Decline (
      std::string ("M1-through-M4 antenna ") + profile.label +
      " geometry arrays are not fully covered");
  }

  summary = derive_domain_summary (
    capture.lower, scene, 0, profile.label, false);
  if (scene.flat_polygon_count != summary.flat_polygon_count ||
      scene.flat_edge_count != summary.flat_edge_count ||
      scene.scene_left != summary.scene_left ||
      scene.scene_bottom != summary.scene_bottom ||
      scene.scene_right != summary.scene_right ||
      scene.scene_top != summary.scene_top) {
    throw AntennaM4Decline (
      std::string ("M1-through-M4 antenna ") + profile.label +
      " flattened census or bounds are inconsistent");
  }
}

std::array<uint8_t, 32> upper_domain_digest (
  const CudaAntennaM4Capture &capture, size_t upper,
  const DomainSummary &summary)
{
  const size_t domain = CudaAntennaM1DomainCount + upper;
  const DomainProfile &profile = domain_profiles [domain];
  const CudaAntennaM1DomainScene &scene =
    capture.upper_domains [upper];
  CanonicalDigest sha;
  sha.bytes (profile.digest_magic, sizeof (profile.digest_magic));
  sha.u32 (raw_scene_format_version);
  sha.u32 (capture.lower.dbu_per_micron);
  sha.u32 (capture.lower.root_cell);
  sha.u32 (capture.lower.reserved);
  sha.u64 (capture.lower.contexts.size ());
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
         capture.lower.contexts.begin ();
       context != capture.lower.contexts.end (); ++context) {
    sha.i64 (context->tx);
    sha.i64 (context->ty);
    sha.u32 (context->cell_id);
    sha.u32 (context->transform_code);
  }

  uint64_t polygon_offset = 0;
  uint64_t edge_offset = 0;
  for (size_t context_id = 0;
       context_id < capture.lower.contexts.size (); ++context_id) {
    const CudaAntennaM1DomainCell &cell =
      scene.cells [capture.lower.contexts [context_id].cell_id];
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
      cell.edge_count, edge_offset, "legacy context edge offset");
  }

  for (size_t cell_id = 0; cell_id < scene.cells.size (); ++cell_id) {
    const CudaAntennaM1DomainCell &cell = scene.cells [cell_id];
    sha.u64 (capture.lower.source_cell_indices [cell_id]);
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

uint64_t capture_header_stored_bytes ()
{
  return UINT64_C (2) * sizeof (uint32_t) +
         CudaAntennaM4UpperDomainCount * sizeof (uint32_t) +
         UINT64_C (32);
}

uint64_t legacy_stored_scene_bytes (
  const CudaAntennaM1Capture &hierarchy,
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
      size_u64 (hierarchy.contexts.size (), "legacy context count"),
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

uint64_t expanded_geometry_bytes (
  const CudaAntennaM1DomainScene &scene)
{
  uint64_t polygon_stride = 0;
  if (! checked_add_u64 (
        sizeof (CudaM1WidthSpacePolygon), sizeof (uint32_t),
        polygon_stride)) {
    throw AntennaM4Decline (
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

void validate_layer_bindings (const CudaAntennaM4Capture &capture)
{
  if (capture.format_version != capture_format_version ||
      capture.reserved != 0) {
    throw AntennaM4Decline (
      "M1-through-M4 antenna capture header is inconsistent");
  }
  std::set<uint32_t> layers;
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    if (! layers.insert (
          capture.lower.source_layer_indices [domain]).second) {
      throw AntennaM4Decline (
        "M1-through-M4 antenna source-layer binding is inconsistent");
    }
  }
  for (size_t upper = 0;
       upper < CudaAntennaM4UpperDomainCount; ++upper) {
    if (! layers.insert (
          capture.upper_source_layer_indices [upper]).second) {
      throw AntennaM4Decline (
        "M1-through-M4 antenna source-layer binding is inconsistent");
    }
  }
}

void derive_census_from_authenticated_lower (
  const CudaAntennaM4Capture &capture,
  const CudaAntennaM1Census &lower,
  CudaAntennaM4Census &census)
{
  validate_layer_bindings (capture);
  if (lower.format_version != capture.lower.format_version ||
      lower.dbu_per_micron != capture.lower.dbu_per_micron ||
      lower.root_cell != capture.lower.root_cell ||
      lower.source_root_cell_index !=
        capture.lower.source_root_cell_index ||
      lower.shared_cell_count != capture.lower.source_cell_indices.size () ||
      lower.shared_context_count != capture.lower.contexts.size () ||
      lower.context_parent_record_count !=
        capture.lower.context_parent_ids.size () ||
      lower.hierarchy_digest != capture.lower.hierarchy_digest ||
      lower.capture_digest != capture.lower.digest) {
    throw AntennaM4Decline (
      "builder-authenticated embedded M1 census is inconsistent");
  }

  CudaAntennaM4Census candidate;
  candidate.format_version = capture.format_version;
  candidate.reserved = capture.reserved;
  candidate.shared_cell_count = lower.shared_cell_count;
  candidate.shared_context_count = lower.shared_context_count;
  candidate.context_parent_record_count =
    lower.context_parent_record_count;
  candidate.stored_cell_records = lower.stored_cell_records;
  candidate.stored_polygon_count = lower.stored_polygon_count;
  candidate.stored_edge_count = lower.stored_edge_count;
  candidate.expanded_polygon_count = lower.expanded_polygon_count;
  candidate.expanded_edge_count = lower.expanded_edge_count;
  candidate.total_stored_bytes = lower.total_stored_bytes;
  checked_accumulate (
    capture_header_stored_bytes (),
    candidate.total_stored_bytes, "M1-through-M4 stored bytes");
  candidate.total_expanded_geometry_bytes =
    lower.total_expanded_geometry_bytes;
  candidate.hierarchy_digest = lower.hierarchy_digest;
  candidate.lower_capture_digest = lower.capture_digest;
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    candidate.domains [domain] = lower.domains [domain];
  }

  for (size_t upper = 0;
       upper < CudaAntennaM4UpperDomainCount; ++upper) {
    const size_t domain = CudaAntennaM1DomainCount + upper;
    const DomainProfile &profile = domain_profiles [domain];
    const CudaAntennaM1DomainScene &scene =
      capture.upper_domains [upper];
    DomainSummary summary;
    validate_upper_domain_geometry (capture, upper, summary);
    const std::array<uint8_t, 32> digest =
      upper_domain_digest (capture, upper, summary);
    if (digest != scene.digest) {
      throw AntennaM4Decline (
        std::string ("M1-through-M4 antenna ") + profile.label +
        " raw scene digest is inconsistent");
    }

    CudaAntennaM1DomainCensus &record =
      candidate.domains [domain];
    record.role = profile.role;
    record.physical_layer = profile.physical_layer;
    record.datatype = profile.datatype;
    record.source_layer_index =
      capture.upper_source_layer_indices [upper];
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
      legacy_stored_scene_bytes (capture.lower, scene, summary);
    record.expanded_geometry_bytes = expanded_geometry_bytes (scene);
    record.scene_digest = digest;

    checked_accumulate (
      record.stored_cell_count, candidate.stored_cell_records,
      "stored cell-record total");
    checked_accumulate (
      record.stored_polygon_count, candidate.stored_polygon_count,
      "stored polygon total");
    checked_accumulate (
      record.stored_edge_count, candidate.stored_edge_count,
      "stored edge total");
    checked_accumulate (
      record.expanded_polygon_count,
      candidate.expanded_polygon_count, "expanded polygon total");
    checked_accumulate (
      record.expanded_edge_count,
      candidate.expanded_edge_count, "expanded edge total");
    checked_accumulate (
      record.stored_bytes, candidate.total_stored_bytes,
      "stored byte total");
    checked_accumulate (
      record.expanded_geometry_bytes,
      candidate.total_expanded_geometry_bytes,
      "expanded geometry byte total");
  }
  if (! checked_add_u64 (
        candidate.total_stored_bytes,
        candidate.total_expanded_geometry_bytes,
        candidate.estimated_peak_bytes)) {
    throw AntennaM4Decline (
      "M1-through-M4 antenna estimated peak bytes overflow uint64");
  }
  census = candidate;
}

void derive_census (
  const CudaAntennaM4Capture &capture,
  CudaAntennaM4Census &census)
{
  CudaAntennaM1Census lower;
  std::string reason;
  if (! cuda_antenna_m1_capture_census (capture.lower, lower, &reason)) {
    throw AntennaM4Decline (
      std::string ("embedded M1 capture is inconsistent: ") + reason);
  }
  derive_census_from_authenticated_lower (capture, lower, census);
}

std::array<uint8_t, 32> capture_digest (
  const CudaAntennaM4Capture &capture,
  const CudaAntennaM4Census &census)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'M', '4', '0', '1' };
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (capture.format_version);
  sha.u32 (capture.reserved);
  sha.bytes (
    census.hierarchy_digest.data (), census.hierarchy_digest.size ());
  sha.bytes (
    census.lower_capture_digest.data (),
    census.lower_capture_digest.size ());
  sha.u32 (CudaAntennaM4DomainCount);
  for (size_t domain = 0;
       domain < CudaAntennaM4DomainCount; ++domain) {
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
    sha.u64 (record.stored_bytes);
    sha.u64 (record.legacy_stored_bytes);
    sha.u64 (record.expanded_geometry_bytes);
    sha.bytes (record.scene_digest.data (), record.scene_digest.size ());
  }
  sha.u64 (census.shared_cell_count);
  sha.u64 (census.shared_context_count);
  sha.u64 (census.context_parent_record_count);
  sha.u64 (census.stored_cell_records);
  sha.u64 (census.stored_polygon_count);
  sha.u64 (census.stored_edge_count);
  sha.u64 (census.expanded_polygon_count);
  sha.u64 (census.expanded_edge_count);
  sha.u64 (census.total_stored_bytes);
  sha.u64 (census.total_expanded_geometry_bytes);
  sha.u64 (census.estimated_peak_bytes);
  return sha.finish ();
}

void validate_capture_limits (
  const CudaAntennaM4CaptureLimits &limits,
  const CudaAntennaM4Census &census)
{
  if (census.total_stored_bytes > limits.max_total_stored_bytes) {
    throw AntennaM4Decline (
      "M1-through-M4 antenna stored bytes exceed the configured capacity");
  }
  if (census.total_expanded_geometry_bytes >
      limits.max_total_expanded_geometry_bytes) {
    throw AntennaM4Decline (
      "M1-through-M4 antenna expanded geometry bytes exceed the configured "
      "capacity");
  }
  if (census.estimated_peak_bytes > limits.max_estimated_peak_bytes) {
    throw AntennaM4Decline (
      "M1-through-M4 antenna estimated peak bytes exceed the configured "
      "capacity");
  }
}

void materialize_upper_domain_scene (
  const CudaAntennaM4Capture &capture, size_t upper,
  const DomainSummary &summary, CudaRawManhattanScene &scene)
{
  const CudaAntennaM1DomainScene &compact =
    capture.upper_domains [upper];
  CudaRawManhattanScene candidate;
  candidate.format_version = raw_scene_format_version;
  candidate.dbu_per_micron = capture.lower.dbu_per_micron;
  candidate.root_cell = capture.lower.root_cell;
  candidate.reserved = capture.lower.reserved;
  candidate.flat_polygon_count = summary.flat_polygon_count;
  candidate.flat_edge_count = summary.flat_edge_count;
  candidate.scene_left = summary.scene_left;
  candidate.scene_bottom = summary.scene_bottom;
  candidate.scene_right = summary.scene_right;
  candidate.scene_top = summary.scene_top;
  candidate.contexts = capture.lower.contexts;
  candidate.cells.resize (compact.cells.size ());
  for (size_t cell_id = 0; cell_id < compact.cells.size (); ++cell_id) {
    const CudaAntennaM1DomainCell &source = compact.cells [cell_id];
    CudaM1WidthSpaceCell &destination = candidate.cells [cell_id];
    destination.source_cell_index =
      capture.lower.source_cell_indices [cell_id];
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
       context_id < capture.lower.contexts.size (); ++context_id) {
    const CudaAntennaM1DomainCell &cell =
      compact.cells [capture.lower.contexts [context_id].cell_id];
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
  uint32_t (CudaAntennaM4Metal1) == uint32_t (CudaAntennaM1Metal1) &&
  uint32_t (CudaAntennaM1DomainCount) == uint32_t (CudaAntennaM4Via1),
  "the M4 capture must preserve the complete M1 domain prefix");
static_assert (
  std::is_standard_layout<CudaAntennaM1DomainCell>::value &&
  std::is_trivially_copyable<CudaAntennaM1DomainCell>::value,
  "antenna domain-cell records must remain pointer-free POD");

} // anonymous namespace

CudaAntennaM4CaptureLimits::CudaAntennaM4CaptureLimits ()
  : scene (),
    max_total_stored_bytes (default_max_total_stored_bytes),
    max_total_expanded_geometry_bytes (
      default_max_total_expanded_geometry_bytes),
    max_estimated_peak_bytes (default_max_estimated_peak_bytes)
{
  //  nothing yet
}

CudaAntennaM4Capture::CudaAntennaM4Capture ()
  : format_version (capture_format_version), reserved (0), lower (),
    upper_source_layer_indices (), upper_domains (), digest ()
{
  //  nothing yet
}

void CudaAntennaM4Capture::swap (
  CudaAntennaM4Capture &other) noexcept
{
  using std::swap;
  swap (format_version, other.format_version);
  swap (reserved, other.reserved);
  lower.swap (other.lower);
  upper_source_layer_indices.swap (other.upper_source_layer_indices);
  for (size_t upper = 0;
       upper < CudaAntennaM4UpperDomainCount; ++upper) {
    upper_domains [upper].swap (other.upper_domains [upper]);
  }
  digest.swap (other.digest);
}

CudaAntennaM4Census::CudaAntennaM4Census ()
  : format_version (0), reserved (0), shared_cell_count (0),
    shared_context_count (0), context_parent_record_count (0),
    stored_cell_records (0), stored_polygon_count (0),
    stored_edge_count (0), expanded_polygon_count (0),
    expanded_edge_count (0), total_stored_bytes (0),
    total_expanded_geometry_bytes (0), estimated_peak_bytes (0),
    domains (), hierarchy_digest (), lower_capture_digest (),
    capture_digest ()
{
  //  nothing yet
}

bool cuda_antenna_m4_build_capture (
  const db::DeepLayer &raw_poly,
  const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_contact,
  const db::DeepLayer &raw_metal1,
  const db::DeepLayer &raw_via1,
  const db::DeepLayer &raw_metal2,
  const db::DeepLayer &raw_via2,
  const db::DeepLayer &raw_metal3,
  const db::DeepLayer &raw_via3,
  const db::DeepLayer &raw_metal4,
  const CudaAntennaM4CaptureLimits &limits,
  CudaAntennaM4Capture &capture,
  CudaAntennaM4Census &authenticated_census,
  std::string *decline_reason)
{
  try {
    const std::array<
      const db::DeepLayer *, CudaAntennaM4DomainCount> inputs = {{
        &raw_poly, &raw_active, &raw_nplus, &raw_nwell,
        &raw_contact, &raw_metal1, &raw_via1, &raw_metal2,
        &raw_via2, &raw_metal3, &raw_via3, &raw_metal4
      }};
    validate_input_identity (inputs, limits);

    CudaAntennaM4Capture candidate;
    CudaAntennaM1CaptureLimits lower_limits;
    lower_limits.scene = limits.scene;
    lower_limits.max_total_stored_bytes =
      limits.max_total_stored_bytes;
    lower_limits.max_total_expanded_geometry_bytes =
      limits.max_total_expanded_geometry_bytes;
    lower_limits.max_estimated_peak_bytes =
      limits.max_estimated_peak_bytes;
    std::string lower_reason;
    CudaAntennaM1Census lower_census;
    if (! cuda_antenna_m1_build_capture (
          raw_poly, raw_active, raw_nplus, raw_nwell, raw_contact,
          raw_metal1, lower_limits, candidate.lower, lower_census,
          &lower_reason)) {
      throw AntennaM4Decline (
        std::string ("embedded M1 capture declined: ") + lower_reason);
    }

    uint64_t admitted_stored_bytes = lower_census.total_stored_bytes;
    checked_accumulate (
      capture_header_stored_bytes (), admitted_stored_bytes,
      "M1-through-M4 admitted stored bytes");
    if (admitted_stored_bytes > limits.max_total_stored_bytes) {
      throw AntennaM4Decline (
        "M1-through-M4 antenna stored bytes exceed the configured capacity "
        "before upper-domain allocation");
    }

    const db::Layout &layout = raw_poly.layout ();
    for (size_t upper = 0;
         upper < CudaAntennaM4UpperDomainCount; ++upper) {
      const size_t domain = CudaAntennaM1DomainCount + upper;
      candidate.upper_source_layer_indices [upper] =
        inputs [domain]->layer ();
      serialize_upper_domain (
        layout, inputs [domain]->layer (), limits.scene,
        candidate.lower, candidate.upper_domains [upper]);
      const DomainSummary summary = derive_domain_summary (
        candidate.lower, candidate.upper_domains [upper],
        &limits.scene, domain_profiles [domain].label, true);
      apply_domain_summary (
        summary, candidate.upper_domains [upper]);
      candidate.upper_domains [upper].digest =
        upper_domain_digest (candidate, upper, summary);
      const uint64_t upper_stored_bytes =
        domain_stored_bytes (candidate.upper_domains [upper]);
      uint64_t next_stored_bytes = 0;
      if (! checked_add_u64 (
            admitted_stored_bytes, upper_stored_bytes,
            next_stored_bytes) ||
          next_stored_bytes > limits.max_total_stored_bytes) {
        throw AntennaM4Decline (
          std::string ("M1-through-M4 antenna ") +
          domain_profiles [domain].label +
          " stored bytes exceed the configured capacity");
      }
      admitted_stored_bytes = next_stored_bytes;
    }

    CudaAntennaM4Census census;
    derive_census_from_authenticated_lower (
      candidate, lower_census, census);
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

bool cuda_antenna_m4_build_capture (
  const db::DeepLayer &raw_poly,
  const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_contact,
  const db::DeepLayer &raw_metal1,
  const db::DeepLayer &raw_via1,
  const db::DeepLayer &raw_metal2,
  const db::DeepLayer &raw_via2,
  const db::DeepLayer &raw_metal3,
  const db::DeepLayer &raw_via3,
  const db::DeepLayer &raw_metal4,
  const CudaAntennaM4CaptureLimits &limits,
  CudaAntennaM4Capture &capture,
  std::string *decline_reason)
{
  CudaAntennaM4Census authenticated_census;
  return cuda_antenna_m4_build_capture (
    raw_poly, raw_active, raw_nplus, raw_nwell, raw_contact, raw_metal1,
    raw_via1, raw_metal2, raw_via2, raw_metal3, raw_via3, raw_metal4,
    limits, capture, authenticated_census, decline_reason);
}

bool cuda_antenna_m4_materialize_domain_scene (
  const CudaAntennaM4Capture &capture,
  CudaAntennaM4Domain domain,
  CudaRawManhattanScene &scene,
  std::string *decline_reason)
{
  try {
    const size_t index = size_t (domain);
    if (index >= CudaAntennaM4DomainCount) {
      throw AntennaM4Decline (
        "M1-through-M4 antenna domain ID is outside the fixed capture");
    }
    CudaAntennaM4Census census;
    derive_census (capture, census);
    if (index < CudaAntennaM1DomainCount) {
      std::string reason;
      if (! cuda_antenna_m1_materialize_domain_scene (
            capture.lower, CudaAntennaM1Domain (index), scene, &reason)) {
        throw AntennaM4Decline (
          std::string ("embedded M1 materialization declined: ") + reason);
      }
    } else {
      const size_t upper = index - CudaAntennaM1DomainCount;
      DomainSummary summary;
      validate_upper_domain_geometry (capture, upper, summary);
      materialize_upper_domain_scene (capture, upper, summary, scene);
    }
    set_reason (decline_reason, "");
    return true;
  } catch (const std::exception &ex) {
    set_reason (decline_reason, ex.what ());
  } catch (...) {
    set_reason (decline_reason, "unknown exception");
  }
  return false;
}

bool cuda_antenna_m4_capture_digest (
  const CudaAntennaM4Capture &capture,
  std::array<uint8_t, 32> &digest)
{
  try {
    CudaAntennaM4Census census;
    derive_census (capture, census);
    const std::array<uint8_t, 32> candidate =
      capture_digest (capture, census);
    digest = candidate;
    return true;
  } catch (...) {
    return false;
  }
}

bool cuda_antenna_m4_capture_census (
  const CudaAntennaM4Capture &capture,
  CudaAntennaM4Census &census,
  std::string *decline_reason)
{
  try {
    CudaAntennaM4Census candidate;
    derive_census (capture, candidate);
    candidate.capture_digest = capture_digest (capture, candidate);
    if (candidate.capture_digest != capture.digest) {
      throw AntennaM4Decline (
        "M1-through-M4 antenna aggregate capture digest is inconsistent");
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

std::string cuda_antenna_m4_census_text (
  const CudaAntennaM4Census &census)
{
  std::ostringstream text;
  text
    << "antenna_m4_capture"
    << " format=" << census.format_version
    << " shared_cells=" << census.shared_cell_count
    << " shared_contexts=" << census.shared_context_count
    << " context_parent_records=" << census.context_parent_record_count
    << " stored_cell_records=" << census.stored_cell_records
    << " stored_polygons=" << census.stored_polygon_count
    << " stored_edges=" << census.stored_edge_count
    << " expanded_polygons=" << census.expanded_polygon_count
    << " expanded_edges=" << census.expanded_edge_count
    << " stored_bytes=" << census.total_stored_bytes
    << " expanded_geometry_bytes="
    << census.total_expanded_geometry_bytes
    << " estimated_peak_bytes=" << census.estimated_peak_bytes
    << " hierarchy_sha256=" << digest_hex (census.hierarchy_digest)
    << " lower_capture_sha256="
    << digest_hex (census.lower_capture_digest)
    << " capture_sha256=" << digest_hex (census.capture_digest);
  for (size_t domain = 0;
       domain < CudaAntennaM4DomainCount; ++domain) {
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
