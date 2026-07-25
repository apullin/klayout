/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM1.h"

#include "dbCudaActive3Digest.h"
#include "dbDeepShapeStore.h"
#include "dbLayout.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace db
{

namespace
{

const uint32_t capture_format_version = 1;
const uint64_t default_max_total_stored_bytes =
  UINT64_C (4) * 1024 * 1024 * 1024;
const uint64_t default_max_total_expanded_geometry_bytes =
  UINT64_C (8) * 1024 * 1024 * 1024;
const uint64_t default_max_estimated_peak_bytes =
  UINT64_C (12) * 1024 * 1024 * 1024;

typedef bool (*RawSceneBuilder) (
  const db::DeepLayer &, const db::CudaM1WidthSpaceSceneLimits &,
  db::CudaRawManhattanScene &, std::string *);

typedef bool (*RawSceneDigest) (
  const db::CudaRawManhattanScene &, std::array<uint8_t, 32> &);

struct DomainProfile
{
  uint32_t role;
  uint32_t physical_layer;
  uint32_t datatype;
  const char *label;
  RawSceneBuilder build;
  RawSceneDigest digest;
};

const DomainProfile domain_profiles [CudaAntennaM1DomainCount] = {
  {
    CudaAntennaM1Poly, 9, 0, "POLY",
    db::cuda_poly_raw_manhattan_build_scene,
    db::cuda_poly_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Active, 1, 0, "ACTIVE",
    db::cuda_active_raw_manhattan_build_scene,
    db::cuda_active_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Nplus, 4, 0, "NPLUS",
    db::cuda_nplus_raw_manhattan_build_scene,
    db::cuda_nplus_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Nwell, 3, 0, "NWELL",
    db::cuda_nwell_raw_manhattan_build_scene,
    db::cuda_nwell_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Contact, 10, 0, "CONTACT",
    db::cuda_contact_raw_manhattan_build_scene,
    db::cuda_contact_raw_manhattan_scene_digest
  },
  {
    CudaAntennaM1Metal1, 11, 0, "M1",
    db::cuda_m1_raw_manhattan_build_scene,
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

bool same_context (
  const CudaM1WidthSpaceContext &first,
  const CudaM1WidthSpaceContext &second)
{
  return
    first.tx == second.tx &&
    first.ty == second.ty &&
    first.cell_id == second.cell_id &&
    first.transform_code == second.transform_code;
}

bool common_hierarchy (
  const CudaRawManhattanScene &reference,
  const CudaRawManhattanScene &candidate)
{
  if (reference.format_version != candidate.format_version ||
      reference.dbu_per_micron != candidate.dbu_per_micron ||
      reference.root_cell != candidate.root_cell ||
      reference.contexts.size () != candidate.contexts.size () ||
      reference.cells.size () != candidate.cells.size ()) {
    return false;
  }
  for (size_t index = 0; index < reference.contexts.size (); ++index) {
    if (! same_context (
          reference.contexts [index], candidate.contexts [index])) {
      return false;
    }
  }
  for (size_t index = 0; index < reference.cells.size (); ++index) {
    if (reference.cells [index].source_cell_index !=
        candidate.cells [index].source_cell_index) {
      return false;
    }
  }
  return true;
}

std::array<uint8_t, 32> hierarchy_digest (
  const CudaAntennaM1Capture &capture)
{
  static const char magic [8] =
    { 'K', 'A', 'N', 'T', 'H', '1', '0', '1' };
  const CudaRawManhattanScene &scene =
    capture.domains [CudaAntennaM1Poly];
  CanonicalDigest sha;
  sha.bytes (magic, sizeof (magic));
  sha.u32 (capture.format_version);
  sha.u32 (capture.dbu_per_micron);
  sha.u32 (capture.root_cell);
  sha.u32 (capture.reserved);
  sha.u64 (capture.source_root_cell_index);
  sha.u64 (scene.cells.size ());
  sha.u64 (scene.contexts.size ());
  for (std::vector<CudaM1WidthSpaceCell>::const_iterator cell =
         scene.cells.begin (); cell != scene.cells.end (); ++cell) {
    sha.u64 (cell->source_cell_index);
  }
  for (size_t context_id = 0;
       context_id < scene.contexts.size (); ++context_id) {
    const CudaM1WidthSpaceContext &context =
      scene.contexts [context_id];
    sha.i64 (context.tx);
    sha.i64 (context.ty);
    sha.u32 (context.cell_id);
    sha.u32 (context.transform_code);
    sha.u32 (capture.context_parent_ids [context_id]);
  }
  return sha.finish ();
}

uint64_t stored_scene_bytes (const CudaRawManhattanScene &scene)
{
  //  Fixed scalar header, bounds and canonical digest.  std::vector object
  //  bookkeeping/capacity is deliberately excluded; record payload is exact.
  uint64_t total =
    UINT64_C (4) * sizeof (uint32_t) +
    UINT64_C (2) * sizeof (uint64_t) +
    UINT64_C (4) * sizeof (int64_t) +
    UINT64_C (32);
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.contexts.size (), "stored context count"),
      sizeof (CudaM1WidthSpaceContext), "stored context bytes"),
    total, "stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.metal_contexts.size (), "nonempty context count"),
      sizeof (uint32_t), "nonempty context bytes"),
    total, "stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (
        scene.context_polygon_offsets.size (),
        "context polygon-offset count"),
      sizeof (uint64_t), "context polygon-offset bytes"),
    total, "stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (
        scene.context_edge_offsets.size (),
        "context edge-offset count"),
      sizeof (uint64_t), "context edge-offset bytes"),
    total, "stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.cells.size (), "stored cell count"),
      sizeof (CudaM1WidthSpaceCell), "stored cell bytes"),
    total, "stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.polygons.size (), "stored polygon count"),
      sizeof (CudaM1WidthSpacePolygon), "stored polygon bytes"),
    total, "stored scene bytes");
  checked_accumulate (
    checked_array_bytes (
      size_u64 (scene.edges.size (), "stored edge count"),
      sizeof (CudaM1WidthSpaceEdge), "stored edge bytes"),
    total, "stored scene bytes");
  return total;
}

uint64_t expanded_geometry_bytes (const CudaRawManhattanScene &scene)
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
  if (capture.format_version != capture_format_version ||
      capture.reserved != 0) {
    throw AntennaM1Decline (
      "M1 antenna capture header is not qualified");
  }

  const CudaRawManhattanScene &reference =
    capture.domains [CudaAntennaM1Poly];
  if (reference.cells.empty () || reference.contexts.empty () ||
      reference.root_cell >= reference.cells.size () ||
      capture.dbu_per_micron != reference.dbu_per_micron ||
      capture.root_cell != reference.root_cell ||
      capture.source_root_cell_index !=
        reference.cells [reference.root_cell].source_cell_index) {
    throw AntennaM1Decline (
      "M1 antenna capture root hierarchy is inconsistent");
  }
  if (capture.context_parent_ids.size () !=
      reference.contexts.size () ||
      capture.context_parent_ids.front () !=
        std::numeric_limits<uint32_t>::max ()) {
    throw AntennaM1Decline (
      "M1 antenna context-parent census is inconsistent");
  }
  for (size_t context = 1;
       context < capture.context_parent_ids.size (); ++context) {
    if (capture.context_parent_ids [context] >= context) {
      throw AntennaM1Decline (
        "M1 antenna context parent is not earlier than its child");
    }
  }

  std::set<uint32_t> source_layers;
  CudaAntennaM1Census candidate;
  candidate.format_version = capture.format_version;
  candidate.dbu_per_micron = capture.dbu_per_micron;
  candidate.root_cell = capture.root_cell;
  candidate.reserved = capture.reserved;
  candidate.source_root_cell_index = capture.source_root_cell_index;
  candidate.shared_cell_count =
    size_u64 (reference.cells.size (), "shared cell count");
  candidate.shared_context_count =
    size_u64 (reference.contexts.size (), "shared context count");
  candidate.context_parent_record_count =
    size_u64 (
      capture.context_parent_ids.size (), "context-parent count");
  candidate.context_parent_bytes = checked_array_bytes (
    candidate.context_parent_record_count, sizeof (uint32_t),
    "context-parent bytes");

  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    const DomainProfile &profile = domain_profiles [domain];
    const CudaRawManhattanScene &scene = capture.domains [domain];
    if (profile.role != domain ||
        ! common_hierarchy (reference, scene) ||
        ! source_layers.insert (
          capture.source_layer_indices [domain]).second) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " hierarchy or source-layer binding is inconsistent");
    }

    std::array<uint8_t, 32> digest;
    if (! profile.digest (scene, digest) ||
        digest != scene.digest) {
      throw AntennaM1Decline (
        std::string ("M1 antenna ") + profile.label +
        " raw scene digest is inconsistent");
    }

    CudaAntennaM1DomainCensus &record =
      candidate.domains [domain];
    record.role = profile.role;
    record.physical_layer = profile.physical_layer;
    record.datatype = profile.datatype;
    record.source_layer_index =
      capture.source_layer_indices [domain];
    record.stored_cell_count =
      size_u64 (scene.cells.size (), "stored cell count");
    record.stored_context_count =
      size_u64 (scene.contexts.size (), "stored context count");
    record.nonempty_context_count =
      size_u64 (
        scene.metal_contexts.size (), "nonempty context count");
    record.stored_polygon_count =
      size_u64 (scene.polygons.size (), "stored polygon count");
    record.stored_edge_count =
      size_u64 (scene.edges.size (), "stored edge count");
    record.expanded_polygon_count = scene.flat_polygon_count;
    record.expanded_edge_count = scene.flat_edge_count;
    record.stored_bytes = stored_scene_bytes (scene);
    record.expanded_geometry_bytes =
      expanded_geometry_bytes (scene);
    record.scene_digest = digest;

    checked_accumulate (
      record.stored_cell_count, candidate.stored_cell_records,
      "stored cell-record total");
    checked_accumulate (
      record.stored_context_count,
      candidate.stored_context_records,
      "stored context-record total");
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
      record.expanded_geometry_bytes,
      candidate.total_expanded_geometry_bytes,
      "expanded geometry byte total");
  }
  checked_accumulate (
    candidate.context_parent_bytes, candidate.total_stored_bytes,
    "stored byte total");

  if (! checked_add_u64 (
        candidate.total_stored_bytes,
        candidate.total_expanded_geometry_bytes,
        candidate.estimated_peak_bytes)) {
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
  sha.u32 (capture.format_version);
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
    sha.u64 (record.stored_bytes);
    sha.u64 (record.expanded_geometry_bytes);
    sha.bytes (record.scene_digest.data (), record.scene_digest.size ());
  }
  sha.u64 (census.shared_cell_count);
  sha.u64 (census.shared_context_count);
  sha.u64 (census.context_parent_record_count);
  sha.u64 (census.context_parent_bytes);
  sha.u64 (census.stored_cell_records);
  sha.u64 (census.stored_context_records);
  sha.u64 (census.nonempty_context_records);
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

void validate_input_identity (
  const std::array<const db::DeepLayer *, CudaAntennaM1DomainCount>
    &inputs)
{
  const db::DeepLayer &reference = *inputs [0];
  std::set<unsigned int> layers;
  for (size_t domain = 0;
       domain < CudaAntennaM1DomainCount; ++domain) {
    const db::DeepLayer &candidate = *inputs [domain];
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
  }
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

CudaAntennaM1DomainCensus::CudaAntennaM1DomainCensus ()
  : role (0), physical_layer (0), datatype (0), source_layer_index (0),
    stored_cell_count (0), stored_context_count (0),
    nonempty_context_count (0), stored_polygon_count (0),
    stored_edge_count (0), expanded_polygon_count (0),
    expanded_edge_count (0), stored_bytes (0),
    expanded_geometry_bytes (0), scene_digest ()
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
    total_stored_bytes (0), total_expanded_geometry_bytes (0),
    estimated_peak_bytes (0), domains (), hierarchy_digest (),
    capture_digest ()
{
  //  nothing yet
}

CudaAntennaM1Capture::CudaAntennaM1Capture ()
  : format_version (capture_format_version), dbu_per_micron (0),
    root_cell (0), reserved (0), source_root_cell_index (0),
    source_layer_indices (), domains (), context_parent_ids (),
    hierarchy_digest (), digest ()
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
  std::string *decline_reason)
{
  try {
    const std::array<
      const db::DeepLayer *, CudaAntennaM1DomainCount> inputs = {{
        &raw_poly, &raw_active, &raw_nplus, &raw_nwell,
        &raw_contact, &raw_metal1
      }};
    validate_input_identity (inputs);

    CudaAntennaM1Capture candidate;
    for (size_t domain = 0;
         domain < CudaAntennaM1DomainCount; ++domain) {
      std::string reason;
      if (! domain_profiles [domain].build (
            *inputs [domain], limits.scene,
            candidate.domains [domain], &reason)) {
        throw AntennaM1Decline (
          std::string ("M1 antenna ") +
          domain_profiles [domain].label +
          " capture declined: " + reason);
      }
      candidate.source_layer_indices [domain] =
        inputs [domain]->layer ();
    }

    const CudaRawManhattanScene &reference =
      candidate.domains [CudaAntennaM1Poly];
    candidate.dbu_per_micron = reference.dbu_per_micron;
    candidate.root_cell = reference.root_cell;
    candidate.source_root_cell_index =
      reference.cells [reference.root_cell].source_cell_index;
    std::string parent_reason;
    if (! cuda_raw_manhattan_context_parents (
          raw_poly, reference, limits.scene,
          candidate.context_parent_ids, &parent_reason)) {
      throw AntennaM1Decline (
        "M1 antenna context-parent capture declined: " +
        parent_reason);
    }
    candidate.hierarchy_digest = hierarchy_digest (candidate);

    CudaAntennaM1Census census;
    derive_census (candidate, census);
    validate_capture_limits (limits, census);
    candidate.digest = capture_digest (candidate, census);

    CudaAntennaM1Census verified;
    derive_census (candidate, verified);
    if (capture_digest (candidate, verified) != candidate.digest) {
      throw AntennaM1Decline (
        "M1 antenna capture digest verification failed");
    }

    capture.swap (candidate);
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
    << " nonempty_context_records=" << census.nonempty_context_records
    << " stored_polygons=" << census.stored_polygon_count
    << " stored_edges=" << census.stored_edge_count
    << " expanded_polygons=" << census.expanded_polygon_count
    << " expanded_edges=" << census.expanded_edge_count
    << " stored_bytes=" << census.total_stored_bytes
    << " expanded_geometry_bytes="
    << census.total_expanded_geometry_bytes
    << " estimated_peak_bytes=" << census.estimated_peak_bytes
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
      << ",stored_contexts=" << record.stored_context_count
      << ",nonempty_contexts=" << record.nonempty_context_count
      << ",stored_polygons=" << record.stored_polygon_count
      << ",stored_edges=" << record.stored_edge_count
      << ",expanded_polygons=" << record.expanded_polygon_count
      << ",expanded_edges=" << record.expanded_edge_count
      << ",stored_bytes=" << record.stored_bytes
      << ",expanded_geometry_bytes=" << record.expanded_geometry_bytes
      << ",sha256=" << digest_hex (record.scene_digest)
      << "}";
  }
  return text.str ();
}

} // namespace db
