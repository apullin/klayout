/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaAntennaM4Transaction.h"

#include "dbCudaAntennaM4.h"
#include "dbCudaSpatialApi.h"
#include "dbCudaSpatialBackend.h"
#include "tlLog.h"

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

const uint64_t default_max_cells = UINT64_C (4000000);
const uint64_t default_max_contexts = UINT64_C (4000000);
const uint64_t default_max_stored_polygons = UINT64_C (20000000);
const uint64_t default_max_stored_edges = UINT64_C (100000000);
const uint64_t default_max_flat_polygons = UINT64_C (200000000);
const uint64_t default_max_flat_edges = UINT64_C (800000000);
const uint64_t default_max_total_stored_bytes =
  UINT64_C (4) * 1024 * 1024 * 1024;
const uint64_t default_max_streamed_expanded_bytes =
  std::numeric_limits<uint64_t>::max ();
const uint64_t default_max_nodes = UINT64_C (2000000000);
const uint64_t default_max_rectangles = UINT64_C (4000000000);
const uint64_t default_max_memberships = UINT64_C (16000000000);
const uint64_t default_max_pair_occurrences = UINT64_C (32000000000);
const uint64_t default_max_unique_candidates = UINT64_C (8000000000);
const uint64_t default_max_cell_members = UINT64_C (1048576);
const uint64_t default_max_dsu_iterations = UINT64_C (1024);
const uint64_t default_max_rule_work = UINT64_C (10000000000000);
//  Qualification target is a 10,240-MiB card; reserve 1,024 MiB for the
//  driver, module state and allocator fragmentation.
const uint64_t default_max_device_bytes =
  UINT64_C (9) * 1024 * 1024 * 1024;

const char *digest_domains
  [KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT] = {
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_POLY_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ACTIVE_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NPLUS_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NWELL_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_CONTACT_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M1_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA1_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M2_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA2_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M3_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA3_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M4_DIGEST_DOMAIN
  };

class AntennaM1M4Decline
  : public std::runtime_error
{
public:
  explicit AntennaM1M4Decline (const std::string &message)
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

void fill_capacity (
  klayout_cuda_spatial_antenna_m1_m4_capacity_v1 &capacity)
{
  std::memset (&capacity, 0, sizeof (capacity));
  capacity.struct_size = sizeof (capacity);
  capacity.max_cells = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_CELLS", default_max_cells);
  capacity.max_contexts = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_CONTEXTS", default_max_contexts);
  capacity.max_stored_polygons = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_STORED_POLYGONS",
    default_max_stored_polygons);
  capacity.max_stored_edges = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_STORED_EDGES",
    default_max_stored_edges);
  capacity.max_flat_polygons = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_FLAT_POLYGONS",
    default_max_flat_polygons);
  capacity.max_flat_edges = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_FLAT_EDGES",
    default_max_flat_edges);
  capacity.max_total_stored_bytes = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_TOTAL_STORED_BYTES",
    default_max_total_stored_bytes);
  //  The backend consumes occurrence geometry as a stream.  The historical
  //  8-GiB diagnostic estimate would reject valid compact captures despite
  //  never allocating that expanded representation on the host.
  capacity.max_total_expanded_geometry_bytes = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_TOTAL_EXPANDED_GEOMETRY_BYTES",
    default_max_streamed_expanded_bytes);
  capacity.max_estimated_peak_bytes = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_ESTIMATED_PEAK_BYTES",
    default_max_streamed_expanded_bytes);
  capacity.max_nodes = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_NODES", default_max_nodes);
  capacity.max_rectangles = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_RECTANGLES",
    default_max_rectangles);
  capacity.max_memberships = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_MEMBERSHIPS",
    default_max_memberships);
  capacity.max_pair_occurrences = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_PAIR_OCCURRENCES",
    default_max_pair_occurrences);
  capacity.max_unique_candidates = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_UNIQUE_CANDIDATES",
    default_max_unique_candidates);
  capacity.max_cell_members = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_CELL_MEMBERS",
    default_max_cell_members);
  capacity.max_dsu_iterations = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_DSU_ITERATIONS",
    default_max_dsu_iterations);
  capacity.max_rule_work = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_RULE_WORK",
    default_max_rule_work);
  capacity.max_device_bytes = env_u64 (
    "KLAYOUT_CUDA_ANTENNA_M1_M4_MAX_DEVICE_BYTES",
    default_max_device_bytes);

  if (! capacity.max_cells || ! capacity.max_contexts ||
      ! capacity.max_stored_polygons || ! capacity.max_stored_edges ||
      ! capacity.max_flat_polygons || ! capacity.max_flat_edges ||
      ! capacity.max_total_stored_bytes ||
      ! capacity.max_total_expanded_geometry_bytes ||
      ! capacity.max_estimated_peak_bytes || ! capacity.max_nodes ||
      ! capacity.max_rectangles || ! capacity.max_memberships ||
      ! capacity.max_pair_occurrences ||
      ! capacity.max_unique_candidates || ! capacity.max_cell_members ||
      ! capacity.max_dsu_iterations || ! capacity.max_rule_work ||
      ! capacity.max_device_bytes) {
    throw AntennaM1M4Decline (
      "an ANTENNA.M1-M4 bounded-computation capacity is zero");
  }
  if (capacity.max_cells > std::numeric_limits<uint32_t>::max () ||
      capacity.max_contexts > std::numeric_limits<uint32_t>::max () ||
      capacity.max_nodes > std::numeric_limits<uint32_t>::max () ||
      capacity.max_rectangles > std::numeric_limits<uint32_t>::max () ||
      capacity.max_cell_members > std::numeric_limits<uint32_t>::max () ||
      capacity.max_dsu_iterations > std::numeric_limits<uint32_t>::max ()) {
    throw AntennaM1M4Decline (
      "an ANTENNA.M1-M4 indexed capacity exceeds uint32");
  }
}

const CudaAntennaM1DomainScene &capture_domain (
  const CudaAntennaM4Capture &capture, size_t index)
{
  if (index < CudaAntennaM1DomainCount) {
    return capture.lower.domains [index];
  }
  return capture.upper_domains [index - CudaAntennaM1DomainCount];
}

uint32_t capture_source_layer (
  const CudaAntennaM4Capture &capture, size_t index)
{
  if (index < CudaAntennaM1DomainCount) {
    return capture.lower.source_layer_indices [index];
  }
  return capture.upper_source_layer_indices
    [index - CudaAntennaM1DomainCount];
}

void fill_hierarchy (
  const CudaAntennaM4Capture &capture,
  klayout_cuda_spatial_antenna_m1_m4_hierarchy_v1 &hierarchy)
{
  std::memset (&hierarchy, 0, sizeof (hierarchy));
  hierarchy.struct_size = sizeof (hierarchy);
  hierarchy.format_version = capture.lower.format_version;
  hierarchy.dbu_per_micron = capture.lower.dbu_per_micron;
  hierarchy.root_cell = capture.lower.root_cell;
  hierarchy.source_root_cell_index =
    capture.lower.source_root_cell_index;
  hierarchy.source_cell_indices =
    capture.lower.source_cell_indices.data ();
  hierarchy.source_cell_count =
    capture.lower.source_cell_indices.size ();
  hierarchy.source_cell_index_record_bytes = sizeof (uint64_t);
  hierarchy.contexts = capture.lower.contexts.data ();
  hierarchy.context_count = capture.lower.contexts.size ();
  hierarchy.context_record_bytes = sizeof (CudaM1WidthSpaceContext);
  hierarchy.context_parent_ids =
    capture.lower.context_parent_ids.data ();
  hierarchy.context_parent_count =
    capture.lower.context_parent_ids.size ();
  hierarchy.context_parent_record_bytes = sizeof (uint32_t);
  std::memcpy (
    hierarchy.hierarchy_digest, capture.lower.hierarchy_digest.data (),
    sizeof (hierarchy.hierarchy_digest));
}

void fill_domain (
  const CudaAntennaM4Capture &capture,
  const CudaAntennaM4Census &census, size_t index,
  klayout_cuda_spatial_antenna_m1_m4_domain_v1 &domain)
{
  const CudaAntennaM1DomainScene &scene =
    capture_domain (capture, index);
  const CudaAntennaM1DomainCensus &record = census.domains [index];
  std::memset (&domain, 0, sizeof (domain));
  domain.struct_size = sizeof (domain);
  domain.role = record.role;
  domain.physical_layer = record.physical_layer;
  domain.datatype = record.datatype;
  domain.source_layer_index = capture_source_layer (capture, index);
  domain.cells = scene.cells.data ();
  domain.cell_count = scene.cells.size ();
  domain.cell_record_bytes = sizeof (CudaAntennaM1DomainCell);
  domain.polygons = scene.polygons.data ();
  domain.polygon_count = scene.polygons.size ();
  domain.polygon_record_bytes = sizeof (CudaM1WidthSpacePolygon);
  domain.edges = scene.edges.data ();
  domain.edge_count = scene.edges.size ();
  domain.edge_record_bytes = sizeof (CudaM1WidthSpaceEdge);
  domain.nonempty_context_count = record.nonempty_context_count;
  domain.flat_polygon_count = scene.flat_polygon_count;
  domain.flat_edge_count = scene.flat_edge_count;
  domain.stored_bytes = record.stored_bytes;
  domain.expanded_geometry_bytes = record.expanded_geometry_bytes;
  domain.scene_left = scene.scene_left;
  domain.scene_bottom = scene.scene_bottom;
  domain.scene_right = scene.scene_right;
  domain.scene_top = scene.scene_top;
  std::memcpy (
    domain.digest_domain, digest_domains [index],
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DIGEST_DOMAIN_BYTES);
  std::memcpy (
    domain.scene_digest, scene.digest.data (),
    sizeof (domain.scene_digest));
}

void fill_census (
  const CudaAntennaM4Census &source,
  klayout_cuda_spatial_antenna_m1_m4_census_v1 &census)
{
  std::memset (&census, 0, sizeof (census));
  census.struct_size = sizeof (census);
  census.format_version = source.format_version;
  census.shared_cell_count = source.shared_cell_count;
  census.shared_context_count = source.shared_context_count;
  census.context_parent_record_count =
    source.context_parent_record_count;
  census.stored_cell_record_count = source.stored_cell_records;
  census.stored_polygon_count = source.stored_polygon_count;
  census.stored_edge_count = source.stored_edge_count;
  census.expanded_polygon_count = source.expanded_polygon_count;
  census.expanded_edge_count = source.expanded_edge_count;
  census.total_stored_bytes = source.total_stored_bytes;
  census.total_expanded_geometry_bytes =
    source.total_expanded_geometry_bytes;
  census.estimated_peak_bytes = source.estimated_peak_bytes;
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
  "ANTENNA.M1-M4 context ABI layout mismatch");
static_assert (
  std::is_standard_layout<CudaAntennaM1DomainCell>::value &&
  std::is_trivially_copyable<CudaAntennaM1DomainCell>::value &&
  sizeof (CudaAntennaM1DomainCell) ==
    sizeof (klayout_cuda_spatial_antenna_m1_m4_cell_v1) &&
  offsetof (CudaAntennaM1DomainCell, polygon_begin) ==
    offsetof (
      klayout_cuda_spatial_antenna_m1_m4_cell_v1, polygon_begin) &&
  offsetof (CudaAntennaM1DomainCell, edge_begin) ==
    offsetof (klayout_cuda_spatial_antenna_m1_m4_cell_v1, edge_begin) &&
  offsetof (CudaAntennaM1DomainCell, polygon_count) ==
    offsetof (
      klayout_cuda_spatial_antenna_m1_m4_cell_v1, polygon_count) &&
  offsetof (CudaAntennaM1DomainCell, edge_count) ==
    offsetof (klayout_cuda_spatial_antenna_m1_m4_cell_v1, edge_count),
  "ANTENNA.M1-M4 cell ABI layout mismatch");
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
    offsetof (
      klayout_cuda_spatial_m1_width_space_polygon_v1, edge_count),
  "ANTENNA.M1-M4 polygon ABI layout mismatch");
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
  "ANTENNA.M1-M4 edge ABI layout mismatch");

} // anonymous namespace

bool cuda_antenna_m1_m4_try_raw_empty (
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
  const db::DeepLayer &raw_metal4)
{
  const bool telemetry =
    env_enabled ("KLAYOUT_CUDA_ANTENNA_M1_M4_TELEMETRY");
  const std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now ();
  try {
    //  This is intentionally the first operation.  Disabled and
    //  symbol-incomplete configurations do not inspect or serialize any
    //  hierarchy state.
    if (! cuda_spatial_antenna_m1_m4_requested ()) {
      return false;
    }

    klayout_cuda_spatial_antenna_m1_m4_capacity_v1 capacity;
    fill_capacity (capacity);

    CudaAntennaM4CaptureLimits limits;
    limits.scene.max_cells = capacity.max_cells;
    limits.scene.max_contexts = capacity.max_contexts;
    limits.scene.max_stored_polygons = capacity.max_stored_polygons;
    limits.scene.max_stored_edges = capacity.max_stored_edges;
    limits.scene.max_flat_polygons = capacity.max_flat_polygons;
    limits.scene.max_flat_edges = capacity.max_flat_edges;
    limits.max_total_stored_bytes = capacity.max_total_stored_bytes;
    limits.max_total_expanded_geometry_bytes =
      capacity.max_total_expanded_geometry_bytes;
    limits.max_estimated_peak_bytes =
      capacity.max_estimated_peak_bytes;

    CudaAntennaM4Capture capture;
    std::string reason;
    if (! cuda_antenna_m4_build_capture (
          raw_poly, raw_active, raw_nplus, raw_nwell, raw_contact,
          raw_metal1, raw_via1, raw_metal2, raw_via2, raw_metal3,
          raw_via3, raw_metal4, limits, capture, &reason)) {
      throw AntennaM1M4Decline (
        reason.empty ()
          ? "unable to build the compact ANTENNA.M1-M4 capture"
          : reason);
    }
    CudaAntennaM4Census census;
    if (! cuda_antenna_m4_capture_census (capture, census, &reason)) {
      throw AntennaM1M4Decline (
        reason.empty ()
          ? "unable to validate the compact ANTENNA.M1-M4 capture"
          : reason);
    }

    const uint64_t device =
      env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0);
    if (device > uint64_t (std::numeric_limits<int32_t>::max ())) {
      throw AntennaM1M4Decline (
        "KLAYOUT_CUDA_SPATIAL_DEVICE exceeds int32");
    }

    klayout_cuda_spatial_antenna_m1_m4_request_v1 request;
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_SHARED_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_QUALIFIED_OPTIONS;
    request.format_version = capture.format_version;
    request.dbu_per_micron = capture.lower.dbu_per_micron;
    request.requested_mask =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_STAGES;
    request.stage_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT;
    request.ratio_numerator =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_NUMERATOR;
    request.ratio_denominator =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_DENOMINATOR;
    request.domain_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
    request.device = int32_t (device);
    fill_hierarchy (capture, request.hierarchy);
    for (size_t index = 0;
         index < KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
         ++index) {
      fill_domain (capture, census, index, request.domains [index]);
    }
    fill_census (census, request.census);
    request.capacity = capacity;
    std::memcpy (
      request.lower_capture_digest, census.lower_capture_digest.data (),
      sizeof (request.lower_capture_digest));
    std::memcpy (
      request.capture_digest, census.capture_digest.data (),
      sizeof (request.capture_digest));

    const std::chrono::steady_clock::time_point call_begin =
      std::chrono::steady_clock::now ();
    const CudaAntennaM1M4Attempt attempt =
      cuda_spatial_try_antenna_m1_m4_empty (request);
    const std::chrono::steady_clock::time_point done =
      std::chrono::steady_clock::now ();
    if (telemetry) {
      tl::info
        << "CUDA ANTENNA.M1-M4 live lowering:"
        << " cells=" << census.shared_cell_count
        << " contexts=" << census.shared_context_count
        << " stored_polygons=" << census.stored_polygon_count
        << " expanded_polygons=" << census.expanded_polygon_count
        << " capture_ms="
        << std::chrono::duration<double, std::milli> (
             call_begin - begin).count ()
        << " call_ms="
        << std::chrono::duration<double, std::milli> (
             done - call_begin).count ()
        << " live_total_ms="
        << std::chrono::duration<double, std::milli> (
             done - begin).count ();
    }
    return
      attempt.disposition == CudaAntennaM1M4Attempt::CertifiedEmpty;
  } catch (const std::exception &ex) {
    if (telemetry) {
      try {
        tl::info
          << "CUDA ANTENNA.M1-M4 live lowering:"
          << " outcome=cpu-fallback message=" << ex.what ();
      } catch (...) {
        //  Telemetry cannot alter the fail-closed result.
      }
    }
  } catch (...) {
    if (telemetry) {
      try {
        tl::info
          << "CUDA ANTENNA.M1-M4 live lowering:"
          << " outcome=cpu-fallback message=unknown-host-exception";
      } catch (...) {
        //  Telemetry cannot alter the fail-closed result.
      }
    }
  }
  return false;
}

} // namespace db
