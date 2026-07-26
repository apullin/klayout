/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaImplant15.h"

#include "dbCudaM1WidthSpace.h"
#include "dbCudaSpatialApi.h"
#include "dbCudaSpatialBackend.h"
#include "dbDeepShapeStore.h"
#include "dbLayout.h"
#include "tlLog.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace db
{

namespace
{

const int64_t qualified_implant1_distance = 140;
const int64_t qualified_implant2_distance = 50;
const int64_t qualified_implant3_distance = 90;
const int64_t qualified_implant4_distance = 90;
const int64_t qualified_grid_cell_size = 2000;

const uint64_t default_max_rectangles = UINT64_C (32000000);
const uint64_t default_max_x_slabs = UINT64_C (32000000);
const uint64_t default_max_union_memberships = UINT64_C (256000000);
const uint64_t default_max_events = UINT64_C (512000000);
const uint64_t default_max_raw_segments = UINT64_C (256000000);
const uint64_t default_max_boundary_segments = UINT64_C (64000000);
const uint64_t default_max_secondary_edges = UINT64_C (128000000);
const uint64_t default_max_grid_cells = UINT64_C (16000000);
const uint64_t default_max_boundary_memberships = UINT64_C (300000000);
const uint64_t default_max_cell_visits = UINT64_C (200000000);
const uint64_t default_max_member_visits = UINT64_C (1200000000);
const uint64_t default_max_pair_work = UINT64_C (1200000000);
const uint64_t default_max_morphology_work = UINT64_C (2000000000000);
const uint64_t default_max_overlap_work = UINT64_C (1200000000);
const uint32_t default_max_slabs_per_rectangle = 16384;
const uint32_t default_max_cells_per_edge = 4096;

class Implant15Decline
  : public std::runtime_error
{
public:
  explicit Implant15Decline (const std::string &message)
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

uint32_t env_u32 (const char *name, uint32_t default_value)
{
  const uint64_t value = env_u64 (name, default_value);
  if (! value || value > std::numeric_limits<uint32_t>::max ()) {
    throw Implant15Decline (
      std::string (name) + " is zero or exceeds uint32");
  }
  return uint32_t (value);
}

void require_same_provenance (
  const db::DeepLayer &nplus, const db::DeepLayer &pplus,
  const db::DeepLayer &gate, const db::DeepLayer &contact)
{
  const db::DeepLayer *layers [4] = {
    &nplus, &pplus, &gate, &contact
  };
  for (size_t i = 1; i < 4; ++i) {
    if (layers [i]->store () != nplus.store () ||
        &layers [i]->layout () != &nplus.layout () ||
        layers [i]->layout_index () != nplus.layout_index () ||
        layers [i]->initial_cell ().cell_index () !=
          nplus.initial_cell ().cell_index ()) {
      throw Implant15Decline (
        "IMPLANT.1-.5 operands do not retain one store/layout/top identity");
    }
  }
  for (size_t first = 0; first < 4; ++first) {
    for (size_t second = first + 1; second < 4; ++second) {
      if (layers [first]->layer () == layers [second]->layer ()) {
        throw Implant15Decline (
          "IMPLANT.1-.5 operands do not retain four distinct layer "
          "provenances");
      }
    }
  }
}

bool scenes_share_hierarchy (
  const CudaRawManhattanScene &first,
  const CudaRawManhattanScene &second)
{
  if (first.format_version != second.format_version ||
      first.dbu_per_micron != second.dbu_per_micron ||
      first.root_cell != second.root_cell ||
      first.contexts.size () != second.contexts.size () ||
      first.cells.size () != second.cells.size ()) {
    return false;
  }
  for (size_t index = 0; index < first.contexts.size (); ++index) {
    const CudaM1WidthSpaceContext &a = first.contexts [index];
    const CudaM1WidthSpaceContext &b = second.contexts [index];
    if (a.tx != b.tx || a.ty != b.ty ||
        a.cell_id != b.cell_id ||
        a.transform_code != b.transform_code) {
      return false;
    }
  }
  for (size_t index = 0; index < first.cells.size (); ++index) {
    if (first.cells [index].source_cell_index !=
        second.cells [index].source_cell_index) {
      return false;
    }
  }
  return true;
}

void fill_scene (
  const CudaRawManhattanScene &source,
  uint32_t role, uint32_t layer, uint32_t datatype,
  const char *digest_domain,
  klayout_cuda_spatial_implant15_scene_v1 &destination)
{
  std::memset (&destination, 0, sizeof (destination));
  destination.struct_size = sizeof (destination);
  destination.role = role;
  destination.format_version = source.format_version;
  destination.dbu_per_micron = source.dbu_per_micron;
  destination.root_cell = source.root_cell;
  destination.layer = layer;
  destination.datatype = datatype;
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
    KLAYOUT_CUDA_SPATIAL_IMPLANT15_DIGEST_DOMAIN_BYTES);
  std::copy (
    source.digest.begin (), source.digest.end (),
    destination.scene_digest);
}

void fill_capacity (
  klayout_cuda_spatial_implant15_capacity_v1 &capacity)
{
  std::memset (&capacity, 0, sizeof (capacity));
  capacity.struct_size = sizeof (capacity);
  capacity.max_slabs_per_rectangle = env_u32 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_SLABS_PER_RECTANGLE",
    default_max_slabs_per_rectangle);
  capacity.max_cells_per_secondary_edge = env_u32 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_CELLS_PER_SECONDARY_EDGE",
    default_max_cells_per_edge);
  capacity.max_cells_per_boundary_edge = env_u32 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_CELLS_PER_BOUNDARY_EDGE",
    default_max_cells_per_edge);
  capacity.max_contexts = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_CONTEXTS", UINT64_C (4000000));
  capacity.max_rectangles = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_RECTANGLES", default_max_rectangles);
  capacity.max_x_slabs = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_X_SLABS", default_max_x_slabs);
  capacity.max_union_memberships = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_UNION_MEMBERSHIPS",
    default_max_union_memberships);
  capacity.max_events = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_EVENTS", default_max_events);
  capacity.max_raw_segments = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_RAW_SEGMENTS",
    default_max_raw_segments);
  capacity.max_boundary_segments = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_BOUNDARY_SEGMENTS",
    default_max_boundary_segments);
  capacity.max_gate_edges = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_GATE_EDGES",
    default_max_secondary_edges);
  capacity.max_contact_edges = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_CONTACT_EDGES",
    default_max_secondary_edges);
  capacity.max_grid_cells = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_GRID_CELLS", default_max_grid_cells);
  capacity.max_secondary_memberships = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_SECONDARY_MEMBERSHIPS",
    default_max_boundary_memberships);
  capacity.max_gate_boundary_cell_visits = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_GATE_BOUNDARY_CELL_VISITS",
    default_max_cell_visits);
  capacity.max_contact_boundary_cell_visits = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_CONTACT_BOUNDARY_CELL_VISITS",
    default_max_cell_visits);
  capacity.max_member_visits = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_MEMBER_VISITS",
    default_max_member_visits);
  capacity.max_pair_work = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_PAIR_WORK", default_max_pair_work);
  capacity.max_morphology_work = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_MORPHOLOGY_WORK",
    default_max_morphology_work);
  capacity.max_overlap_work = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_OVERLAP_WORK",
    default_max_overlap_work);

  if (! capacity.max_contexts || ! capacity.max_rectangles ||
      ! capacity.max_x_slabs || ! capacity.max_union_memberships ||
      ! capacity.max_events || ! capacity.max_raw_segments ||
      ! capacity.max_boundary_segments || ! capacity.max_gate_edges ||
      ! capacity.max_contact_edges || ! capacity.max_grid_cells ||
      ! capacity.max_secondary_memberships ||
      ! capacity.max_gate_boundary_cell_visits ||
      ! capacity.max_contact_boundary_cell_visits ||
      ! capacity.max_member_visits || ! capacity.max_pair_work ||
      ! capacity.max_morphology_work || ! capacity.max_overlap_work) {
    throw Implant15Decline (
      "an IMPLANT.1-.5 bounded-computation capacity is zero");
  }
}

void set_scene_limits (
  const klayout_cuda_spatial_implant15_capacity_v1 &capacity,
  CudaM1WidthSpaceSceneLimits &limits)
{
  limits.max_cells = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_CELLS", limits.max_cells);
  limits.max_contexts = capacity.max_contexts;
  limits.max_stored_polygons = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_STORED_POLYGONS",
    limits.max_stored_polygons);
  limits.max_stored_edges = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_STORED_EDGES",
    limits.max_stored_edges);
  limits.max_flat_polygons = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_FLAT_POLYGONS",
    capacity.max_rectangles);
  limits.max_flat_edges = env_u64 (
    "KLAYOUT_CUDA_IMPLANT15_MAX_FLAT_EDGES",
    default_max_secondary_edges);
  if (! limits.max_cells || ! limits.max_stored_polygons ||
      ! limits.max_stored_edges || ! limits.max_flat_polygons ||
      ! limits.max_flat_edges) {
    throw Implant15Decline (
      "an IMPLANT.1-.5 host scene capacity is zero");
  }
}

static_assert (
  std::is_standard_layout<CudaM1WidthSpaceContext>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceContext>::value &&
  sizeof (CudaM1WidthSpaceContext) ==
    sizeof (klayout_cuda_spatial_m1_width_space_context_v1),
  "IMPLANT.1-.5 raw context ABI layout mismatch");
static_assert (
  std::is_standard_layout<CudaM1WidthSpaceCell>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceCell>::value &&
  sizeof (CudaM1WidthSpaceCell) ==
    sizeof (klayout_cuda_spatial_m1_width_space_cell_v1),
  "IMPLANT.1-.5 raw cell ABI layout mismatch");
static_assert (
  std::is_standard_layout<CudaM1WidthSpacePolygon>::value &&
  std::is_trivially_copyable<CudaM1WidthSpacePolygon>::value &&
  sizeof (CudaM1WidthSpacePolygon) ==
    sizeof (klayout_cuda_spatial_m1_width_space_polygon_v1),
  "IMPLANT.1-.5 raw polygon ABI layout mismatch");
static_assert (
  std::is_standard_layout<CudaM1WidthSpaceEdge>::value &&
  std::is_trivially_copyable<CudaM1WidthSpaceEdge>::value &&
  sizeof (CudaM1WidthSpaceEdge) ==
    sizeof (klayout_cuda_spatial_m1_width_space_edge_v1),
  "IMPLANT.1-.5 raw edge ABI layout mismatch");

} // anonymous namespace

bool cuda_implant15_try_raw_empty (
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_pplus,
  const db::DeepLayer &derived_gate,
  const db::DeepLayer &raw_contact)
{
  const bool telemetry =
    env_enabled ("KLAYOUT_CUDA_IMPLANT15_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    if (! db::cuda_spatial_implant15_requested ()) {
      return false;
    }
    require_same_provenance (
      raw_nplus, raw_pplus, derived_gate, raw_contact);

    klayout_cuda_spatial_implant15_capacity_v1 capacity;
    fill_capacity (capacity);
    CudaM1WidthSpaceSceneLimits scene_limits;
    set_scene_limits (capacity, scene_limits);

    CudaRawManhattanScene nplus_scene;
    CudaRawManhattanScene pplus_scene;
    CudaRawManhattanScene gate_scene;
    CudaRawManhattanScene contact_scene;
    std::string reason;
    if (! cuda_nplus_raw_manhattan_build_scene (
          raw_nplus, scene_limits, nplus_scene, &reason)) {
      throw Implant15Decline (
        reason.empty () ? "unable to serialize raw NPLUS" : reason);
    }
    if (! cuda_pplus_raw_manhattan_build_scene (
          raw_pplus, scene_limits, pplus_scene, &reason)) {
      throw Implant15Decline (
        reason.empty () ? "unable to serialize raw PPLUS" : reason);
    }
    if (! cuda_gate_raw_manhattan_build_scene (
          derived_gate, scene_limits, gate_scene, &reason)) {
      throw Implant15Decline (
        reason.empty () ? "unable to serialize derived GATE" : reason);
    }
    if (! cuda_contact_raw_manhattan_build_scene (
          raw_contact, scene_limits, contact_scene, &reason)) {
      throw Implant15Decline (
        reason.empty () ? "unable to serialize raw CONTACT" : reason);
    }
    if (! scenes_share_hierarchy (nplus_scene, pplus_scene) ||
        ! scenes_share_hierarchy (nplus_scene, gate_scene) ||
        ! scenes_share_hierarchy (nplus_scene, contact_scene)) {
      throw Implant15Decline (
        "IMPLANT.1-.5 raw scenes lost their common hierarchy census");
    }

    const uint64_t device = env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0);
    if (device > uint64_t (std::numeric_limits<int32_t>::max ())) {
      throw Implant15Decline (
        "KLAYOUT_CUDA_SPATIAL_DEVICE exceeds int32");
    }

    klayout_cuda_spatial_implant15_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode = KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_RESIDENT_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_QUALIFIED_OPTIONS;
    request.format_version = nplus_scene.format_version;
    request.dbu_per_micron = nplus_scene.dbu_per_micron;
    request.requested_mask = KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES;
    request.device = int32_t (device);
    request.implant1_distance = qualified_implant1_distance;
    request.implant2_distance = qualified_implant2_distance;
    request.implant3_distance = qualified_implant3_distance;
    request.implant4_distance = qualified_implant4_distance;
    request.grid_cell_size = qualified_grid_cell_size;
    fill_scene (
      nplus_scene, KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_ROLE, 4, 0,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_DIGEST_DOMAIN,
      request.nplus);
    fill_scene (
      pplus_scene, KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_ROLE, 5, 0,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_DIGEST_DOMAIN,
      request.pplus);
    fill_scene (
      gate_scene, KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_ROLE,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_LAYER,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_DATATYPE,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_DIGEST_DOMAIN,
      request.gate);
    fill_scene (
      contact_scene, KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_ROLE, 10, 0,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_DIGEST_DOMAIN,
      request.contact);
    request.capacity = capacity;

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const CudaImplant15Attempt attempt =
      db::cuda_spatial_try_implant15_empty (request);
    const std::chrono::steady_clock::time_point done =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info << "CUDA IMPLANT.1-.5 raw live lowering:"
               << " nplus_contexts=" << request.nplus.context_count
               << " pplus_contexts=" << request.pplus.context_count
               << " gate_contexts=" << request.gate.context_count
               << " contact_contexts=" << request.contact.context_count
               << " nplus_flat_polygons="
               << request.nplus.flat_polygon_count
               << " pplus_flat_polygons="
               << request.pplus.flat_polygon_count
               << " gate_flat_polygons="
               << request.gate.flat_polygon_count
               << " contact_flat_polygons="
               << request.contact.flat_polygon_count
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
    return attempt.disposition == CudaImplant15Attempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info << "CUDA IMPLANT.1-.5 raw live lowering:"
                 << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry cannot alter the fail-closed result.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info << "CUDA IMPLANT.1-.5 raw live lowering:"
                 << " outcome=cpu-fallback message=unknown-host-exception";
      } catch (...) {
        //  Telemetry cannot alter the fail-closed result.
      }
    }
  }
  return false;
}

} // namespace db
