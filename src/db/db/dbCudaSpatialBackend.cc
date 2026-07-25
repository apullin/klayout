/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaSpatialBackend.h"
#include "dbCudaImplant12Digest.h"
#include "dbCudaVia1StackDigest.h"
#include "tlLog.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <type_traits>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace db
{

static_assert (
  std::is_standard_layout<
    klayout_cuda_spatial_m2_union_segment_v1>::value &&
  std::is_trivially_copyable<
    klayout_cuda_spatial_m2_union_segment_v1>::value,
  "M2 union boundary segment must remain pointer-free POD");
static_assert (
  sizeof (klayout_cuda_spatial_m2_union_segment_v1) == 32 &&
  offsetof (klayout_cuda_spatial_m2_union_segment_v1, fixed) == 0 &&
  offsetof (klayout_cuda_spatial_m2_union_segment_v1, lo) == 8 &&
  offsetof (klayout_cuda_spatial_m2_union_segment_v1, hi) == 16 &&
  offsetof (klayout_cuda_spatial_m2_union_segment_v1, side) == 24 &&
  offsetof (klayout_cuda_spatial_m2_union_segment_v1, axis) == 28,
  "M2 union boundary segment ABI layout changed");
static_assert (
  std::is_standard_layout<CudaM2UnionTiming>::value &&
  std::is_trivially_copyable<CudaM2UnionTiming>::value &&
  sizeof (CudaM2UnionTiming) == 72 &&
  offsetof (CudaM2UnionTiming, format_version) == 0 &&
  offsetof (CudaM2UnionTiming, struct_size) == 4 &&
  offsetof (CudaM2UnionTiming, setup_ns) == 8 &&
  offsetof (CudaM2UnionTiming, total_ns) == 64,
  "additive M2 union component-timing ABI layout changed");
static_assert (
  sizeof (void *) != 8 ||
    (sizeof (klayout_cuda_spatial_contact4_active_union_scene_v1) == 280 &&
     offsetof (
       klayout_cuda_spatial_contact4_active_union_scene_v1,
       digest_domain) == 224 &&
     offsetof (
       klayout_cuda_spatial_contact4_active_union_scene_v1,
       scene_digest) == 232),
  "CONTACT.4 ACTIVE-union scene ABI layout changed");
static_assert (
  sizeof (void *) != 8 ||
    (sizeof (klayout_cuda_spatial_contact4_active_union_scene_echo_v1) ==
       192 &&
     sizeof (klayout_cuda_spatial_contact4_active_union_request_v1) == 760 &&
     offsetof (
       klayout_cuda_spatial_contact4_active_union_request_v1,
       active) == 48 &&
     offsetof (
       klayout_cuda_spatial_contact4_active_union_request_v1,
       contact) == 328 &&
     sizeof (klayout_cuda_spatial_contact4_active_union_result_v1) == 944 &&
     offsetof (
       klayout_cuda_spatial_contact4_active_union_result_v1,
       active) == 64 &&
     offsetof (
       klayout_cuda_spatial_contact4_active_union_result_v1,
       contact) == 256),
  "CONTACT.4 ACTIVE-union request/result ABI layout changed");
static_assert (
  sizeof (void *) != 8 ||
    (sizeof (klayout_cuda_spatial_active3_well_union_request_v1) == 776 &&
     offsetof (
       klayout_cuda_spatial_active3_well_union_request_v1, wells) == 64 &&
     offsetof (
       klayout_cuda_spatial_active3_well_union_request_v1, active) == 344 &&
     sizeof (klayout_cuda_spatial_active3_well_union_result_v1) == 960 &&
     offsetof (
       klayout_cuda_spatial_active3_well_union_result_v1, wells) == 80 &&
     offsetof (
       klayout_cuda_spatial_active3_well_union_result_v1, active) == 272),
  "ACTIVE.3 WELL-union request/result ABI layout changed");

CudaSpatialAttempt::CudaSpatialAttempt ()
  : disposition (Disabled), fallback_flags (0), membership_count (0),
    occupied_cell_count (0), pair_work_count (0), setup_ns (0), h2d_ns (0),
    broad_phase_ns (0), sort_unique_ns (0), d2h_ns (0), total_ns (0)
{
  //  nothing yet
}

CudaActive3Attempt::CudaActive3Attempt ()
  : disposition (Disabled), fallback_flags (0), device_flags (0),
    context_count (0), well_context_count (0), active_context_count (0),
    cell_count (0), edge_count (0), flat_well_edge_count (0),
    flat_active_edge_count (0), grid_cell_count (0), membership_count (0),
    candidate_pair_count (0), raw_hit_count (0), uncertain_count (0),
    total_ns (0)
{
  //  nothing yet
}

CudaContact4ActiveUnionAttempt::CudaContact4ActiveUnionAttempt ()
  : disposition (Disabled), fallback_flags (0), device_flags (0),
    active_context_count (0), contact_context_count (0),
    flat_active_polygon_count (0), flat_active_edge_count (0),
    flat_contact_polygon_count (0), flat_contact_edge_count (0),
    rectangle_count (0), x_slab_count (0), union_membership_count (0),
    strip_interval_count (0), boundary_segment_count (0),
    grid_cell_count (0), contact_membership_count (0),
    boundary_cell_visit_count (0), member_visit_count (0),
    candidate_pair_count (0), raw_hit_count (0), uncertain_count (0),
    total_ns (0)
{
  //  nothing yet
}

CudaActive3WellUnionAttempt::CudaActive3WellUnionAttempt ()
  : disposition (Disabled), fallback_flags (0), device_flags (0),
    well_context_count (0), active_context_count (0),
    flat_well_polygon_count (0), flat_well_edge_count (0),
    flat_active_polygon_count (0), flat_active_edge_count (0),
    rectangle_count (0), x_slab_count (0), union_membership_count (0),
    strip_interval_count (0), boundary_segment_count (0),
    grid_cell_count (0), active_membership_count (0),
    active_cell_visit_count (0), member_visit_count (0),
    candidate_pair_count (0), raw_hit_count (0), uncertain_count (0),
    total_ns (0)
{
  //  nothing yet
}

CudaM1WidthSpaceAttempt::CudaM1WidthSpaceAttempt ()
  : disposition (Disabled), fallback_flags (0), device_flags (0),
    context_count (0), metal_context_count (0), cell_count (0),
    polygon_count (0), edge_count (0), flat_polygon_count (0),
    flat_edge_count (0), grid_cell_count (0), membership_count (0),
    pair_work_count (0), unique_edge_pair_count (0), width_pair_count (0),
    space_pair_count (0), width_hit_count (0), space_hit_count (0),
    width_uncertain_count (0), space_uncertain_count (0), total_ns (0)
{
  //  nothing yet
}

CudaM2UnionAttempt::CudaM2UnionAttempt ()
  : disposition (Disabled), fallback_flags (0), device_flags (0),
    context_count (0), metal_context_count (0), cell_count (0),
    polygon_count (0), edge_count (0), flat_polygon_count (0),
    flat_edge_count (0), rectangle_count (0), x_slab_count (0),
    membership_count (0), event_count (0), strip_interval_count (0),
    raw_segment_count (0), boundary_fnv64 (0), total_ns (0)
{
  //  nothing yet
}

CudaPoly34Attempt::CudaPoly34Attempt ()
  : disposition (Disabled), certified_empty_mask (0), fallback_flags (0),
    device_flags (0), context_count (0), poly_context_count (0),
    active_context_count (0), gate_context_count (0), cell_count (0),
    box_count (0), flat_poly_box_count (0), flat_active_box_count (0),
    flat_gate_box_count (0), poly_membership_count (0),
    active_membership_count (0), poly_query_visit_count (0),
    active_query_visit_count (0), poly_candidate_count (0),
    active_candidate_count (0), poly_terminal_empty_count (0),
    active_terminal_empty_count (0), atomic_terminal_empty_count (0),
    fallback_gate_count (0), total_ns (0)
{
  //  nothing yet
}

CudaVia1StackAttempt::CudaVia1StackAttempt ()
  : disposition (Disabled), certified_empty_mask (0), fallback_flags (0),
    device_flags (0), context_count (0), flat_metal1_box_count (0),
    flat_via1_box_count (0), flat_metal2_box_count (0),
    via_expanded_count (0), via_size_checked_count (0),
    via_size_violation_count (0), metal1_expanded_count (0),
    metal2_expanded_count (0), grid_cell_count (0),
    via_membership_count (0), metal1_membership_count (0),
    metal2_membership_count (0), via_pair_queried_count (0),
    via_candidate_pair_count (0),
    duplicate_via_pair_count (0), unsafe_via_pair_count (0),
    spacing_violation_count (0), clean_via_pair_count (0),
    metal1_queried_count (0), metal1_candidate_count (0),
    metal1_certified_count (0), metal1_miss_count (0),
    metal2_queried_count (0), metal2_candidate_count (0),
    metal2_certified_count (0), metal2_miss_count (0), total_ns (0)
{
  //  nothing yet
}

CudaImplant12Attempt::CudaImplant12Attempt ()
  : disposition (Disabled), certified_empty_mask (0), clean_mask (0),
    fallback_flags (0), device_flags (0), context_count (0),
    implant_context_count (0), gate_context_count (0),
    contact_context_count (0), cell_count (0), contour_count (0),
    edge_count (0), flat_implant_polygon_count (0),
    flat_gate_polygon_count (0), flat_contact_polygon_count (0),
    flat_implant_contour_count (0), flat_gate_contour_count (0),
    flat_contact_contour_count (0), flat_implant_edge_count (0),
    flat_gate_edge_count (0), flat_contact_edge_count (0),
    implant_expanded_edge_count (0), gate_processed_edge_count (0),
    contact_processed_edge_count (0), grid_cell_count (0),
    implant_membership_count (0), gate_query_visit_count (0),
    gate_candidate_count (0), gate_raw_hit_count (0),
    gate_uncertain_count (0), contact_query_visit_count (0),
    contact_candidate_count (0), contact_raw_hit_count (0),
    contact_uncertain_count (0), setup_ns (0), h2d_ns (0),
    implant_expand_ns (0), grid_count_ns (0), grid_build_ns (0),
    gate_query_ns (0), contact_query_ns (0), d2h_ns (0), total_ns (0)
{
  //  nothing yet
}

namespace
{

bool checked_multiply_u64 (
  uint64_t first, uint64_t second, uint64_t &result)
{
  if (first && second > std::numeric_limits<uint64_t>::max () / first) {
    return false;
  }
  result = first * second;
  return true;
}

bool checked_add_u64 (
  uint64_t first, uint64_t second, uint64_t &result)
{
  if (second > std::numeric_limits<uint64_t>::max () - first) {
    return false;
  }
  result = first + second;
  return true;
}

bool array_bytes_fit (uint64_t count, uint64_t record_bytes)
{
  uint64_t bytes = 0;
  return checked_multiply_u64 (count, record_bytes, bytes) &&
         bytes <= std::numeric_limits<size_t>::max ();
}

bool m2_union_segment_less (
  const klayout_cuda_spatial_m2_union_segment_v1 &first,
  const klayout_cuda_spatial_m2_union_segment_v1 &second)
{
  if (first.axis != second.axis) {
    return first.axis < second.axis;
  }
  if (first.side != second.side) {
    return first.side < second.side;
  }
  if (first.fixed != second.fixed) {
    return first.fixed < second.fixed;
  }
  if (first.lo != second.lo) {
    return first.lo < second.lo;
  }
  return first.hi < second.hi;
}

bool m2_union_same_line (
  const klayout_cuda_spatial_m2_union_segment_v1 &first,
  const klayout_cuda_spatial_m2_union_segment_v1 &second)
{
  return first.axis == second.axis && first.side == second.side &&
         first.fixed == second.fixed;
}

uint64_t m2_union_boundary_fnv64 (
  const klayout_cuda_spatial_m2_union_segment_v1 *segments,
  uint64_t segment_count)
{
  uint64_t hash = UINT64_C (1469598103934665603);
  const auto mix = [&hash] (uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C (0xff);
      hash *= UINT64_C (1099511628211);
    }
  };
  mix (segment_count);
  for (uint64_t index = 0; index < segment_count; ++index) {
    const klayout_cuda_spatial_m2_union_segment_v1 &segment =
      segments [index];
    mix (uint64_t (segment.axis));
    mix (uint64_t (uint32_t (segment.side)));
    mix (uint64_t (segment.fixed));
    mix (uint64_t (segment.lo));
    mix (uint64_t (segment.hi));
  }
  return hash;
}

bool strictly_increasing_context_ids (
  const uint32_t *contexts, uint64_t count, uint64_t context_count)
{
  if (! contexts || ! count) {
    return false;
  }
  for (uint64_t index = 0; index < count; ++index) {
    if (contexts [index] >= context_count ||
        (index && contexts [index - 1] >= contexts [index])) {
      return false;
    }
  }
  return true;
}

bool qualified_m2_union_request (
  const klayout_cuda_spatial_m2_union_request_v1 &request)
{
  return
    request.abi_version == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
    request.struct_size == sizeof (request) &&
    (request.opcode ==
       KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY ||
     request.opcode ==
       KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY) &&
    request.option_flags == KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS &&
    request.format_version == 1 && request.dbu_per_micron == 2000 &&
    request.device >= 0 && request.context_reserved == 0 &&
    request.cell_reserved == 0 && request.polygon_reserved == 0 &&
    request.edge_reserved == 0 && request.reserved0 == 0 &&
    request.reserved1 [0] == 0 && request.reserved1 [1] == 0 &&
    request.context_count && request.contexts &&
    request.context_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_context_v1) &&
    request.metal_context_count && request.metal_contexts &&
    request.context_polygon_offset_count == request.metal_context_count &&
    request.context_polygon_offsets &&
    request.context_edge_offset_count == request.metal_context_count &&
    request.context_edge_offsets &&
    request.cell_count && request.cells &&
    request.cell_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_cell_v1) &&
    request.polygon_count && request.polygons &&
    request.polygon_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_polygon_v1) &&
    request.edge_count && request.edges &&
    request.edge_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_edge_v1) &&
    request.root_cell < request.cell_count &&
    request.flat_polygon_count && request.flat_edge_count &&
    request.scene_left < request.scene_right &&
    request.scene_bottom < request.scene_top &&
    request.max_contexts && request.max_rectangles &&
    request.max_x_slabs && request.max_memberships &&
    request.max_events && request.max_raw_segments &&
    request.max_segments &&
    request.max_slabs_per_rectangle &&
    request.context_count <= request.max_contexts &&
    request.context_count <= std::numeric_limits<uint32_t>::max () &&
    request.metal_context_count <= std::numeric_limits<uint32_t>::max () &&
    request.cell_count <= std::numeric_limits<uint32_t>::max () &&
    request.polygon_count <= std::numeric_limits<uint32_t>::max () &&
    request.edge_count <= std::numeric_limits<uint32_t>::max () &&
    request.flat_polygon_count <= std::numeric_limits<uint32_t>::max () &&
    request.flat_edge_count <= std::numeric_limits<uint32_t>::max () &&
    request.edge_count <= request.max_memberships &&
    request.max_x_slabs <= std::numeric_limits<uint32_t>::max () &&
    array_bytes_fit (request.context_count, request.context_record_bytes) &&
    array_bytes_fit (
      request.metal_context_count, sizeof (uint32_t)) &&
    array_bytes_fit (
      request.context_polygon_offset_count, sizeof (uint64_t)) &&
    array_bytes_fit (
      request.context_edge_offset_count, sizeof (uint64_t)) &&
    array_bytes_fit (request.cell_count, request.cell_record_bytes) &&
    array_bytes_fit (request.polygon_count, request.polygon_record_bytes) &&
    array_bytes_fit (request.edge_count, request.edge_record_bytes) &&
    array_bytes_fit (
      request.max_segments,
      sizeof (klayout_cuda_spatial_m2_union_segment_v1)) &&
    request.max_segments <=
      uint64_t (std::numeric_limits<std::ptrdiff_t>::max ()) &&
    strictly_increasing_context_ids (
      request.metal_contexts, request.metal_context_count,
      request.context_count);
}

bool contact4_active_union_domain_matches (
  const uint8_t domain [KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES],
  const char *expected)
{
  return std::memcmp (
           domain, expected,
           KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES) == 0;
}

bool qualified_contact4_active_union_scene (
  const klayout_cuda_spatial_contact4_active_union_scene_v1 &scene,
  uint32_t expected_role, uint32_t expected_layer,
  const char *expected_digest_domain, uint64_t max_contexts)
{
  uint64_t minimum_source_edges = 0;
  uint64_t minimum_flat_edges = 0;
  return
    checked_multiply_u64 (
      scene.polygon_count, UINT64_C (4), minimum_source_edges) &&
    checked_multiply_u64 (
      scene.flat_polygon_count, UINT64_C (4), minimum_flat_edges) &&
    scene.struct_size == sizeof (scene) &&
    scene.role == expected_role &&
    scene.format_version == 1 &&
    scene.dbu_per_micron == 2000 &&
    scene.layer == expected_layer && scene.datatype == 0 &&
    scene.reserved0 == 0 && scene.context_reserved == 0 &&
    scene.cell_reserved == 0 && scene.polygon_reserved == 0 &&
    scene.edge_reserved == 0 &&
    scene.reserved1 [0] == 0 && scene.reserved1 [1] == 0 &&
    contact4_active_union_domain_matches (
      scene.digest_domain, expected_digest_domain) &&
    scene.context_count && scene.contexts &&
    scene.context_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_context_v1) &&
    scene.layer_context_count && scene.layer_contexts &&
    scene.context_polygon_offset_count == scene.layer_context_count &&
    scene.context_polygon_offsets &&
    scene.context_edge_offset_count == scene.layer_context_count &&
    scene.context_edge_offsets &&
    scene.cell_count && scene.cells &&
    scene.cell_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_cell_v1) &&
    scene.polygon_count && scene.polygons &&
    scene.polygon_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_polygon_v1) &&
    scene.edge_count && scene.edges &&
    scene.edge_record_bytes ==
      sizeof (klayout_cuda_spatial_m1_width_space_edge_v1) &&
    scene.root_cell < scene.cell_count &&
    scene.flat_polygon_count && scene.flat_edge_count &&
    scene.edge_count >= minimum_source_edges &&
    scene.flat_edge_count >= minimum_flat_edges &&
    scene.scene_left < scene.scene_right &&
    scene.scene_bottom < scene.scene_top &&
    max_contexts &&
    scene.context_count <= max_contexts &&
    scene.context_count <= std::numeric_limits<uint32_t>::max () &&
    scene.layer_context_count <= std::numeric_limits<uint32_t>::max () &&
    scene.cell_count <= std::numeric_limits<uint32_t>::max () &&
    scene.polygon_count <= std::numeric_limits<uint32_t>::max () &&
    scene.edge_count <= std::numeric_limits<uint32_t>::max () &&
    scene.flat_polygon_count <= std::numeric_limits<uint32_t>::max () &&
    scene.flat_edge_count <= std::numeric_limits<uint32_t>::max () &&
    array_bytes_fit (scene.context_count, scene.context_record_bytes) &&
    array_bytes_fit (scene.layer_context_count, sizeof (uint32_t)) &&
    array_bytes_fit (
      scene.context_polygon_offset_count, sizeof (uint64_t)) &&
    array_bytes_fit (
      scene.context_edge_offset_count, sizeof (uint64_t)) &&
    array_bytes_fit (scene.cell_count, scene.cell_record_bytes) &&
    array_bytes_fit (scene.polygon_count, scene.polygon_record_bytes) &&
    array_bytes_fit (scene.edge_count, scene.edge_record_bytes) &&
    strictly_increasing_context_ids (
      scene.layer_contexts, scene.layer_context_count,
      scene.context_count);
}

bool qualified_contact4_active_union_request (
  const klayout_cuda_spatial_contact4_active_union_request_v1 &request)
{
  return
    request.abi_version == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
    request.struct_size == sizeof (request) &&
    request.opcode ==
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_EMPTY &&
    request.option_flags ==
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_QUALIFIED_OPTIONS &&
    request.format_version == 1 && request.dbu_per_micron == 2000 &&
    request.device >= 0 && request.reserved0 == 0 &&
    request.distance == 10 && request.grid_cell_size == 2000 &&
    request.union_reserved == 0 &&
    request.reserved1 [0] == 0 && request.reserved1 [1] == 0 &&
    request.reserved1 [2] == 0 && request.reserved1 [3] == 0 &&
    request.max_contexts &&
    request.max_contexts <= std::numeric_limits<uint32_t>::max () &&
    request.max_rectangles && request.max_x_slabs &&
    request.max_union_memberships && request.max_events &&
    request.max_raw_segments && request.max_boundary_segments &&
    request.max_slabs_per_rectangle &&
    request.max_x_slabs <= std::numeric_limits<uint32_t>::max () &&
    request.max_contact_edges &&
    request.max_contact_edges <= std::numeric_limits<uint32_t>::max () &&
    request.max_grid_cells &&
    request.max_grid_cells <= std::numeric_limits<uint32_t>::max () &&
    request.max_contact_memberships &&
    request.max_boundary_cell_visits &&
    request.max_member_visits && request.max_pair_work &&
    request.max_cells_per_contact_edge &&
    request.max_cells_per_boundary_edge &&
    qualified_contact4_active_union_scene (
      request.active, KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE, 1,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_DIGEST_DOMAIN,
      request.max_contexts) &&
    qualified_contact4_active_union_scene (
      request.contact, KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_DIGEST_DOMAIN,
      request.max_contexts) &&
    request.active.format_version == request.format_version &&
    request.contact.format_version == request.format_version &&
    request.active.dbu_per_micron == request.dbu_per_micron &&
    request.contact.dbu_per_micron == request.dbu_per_micron &&
    request.contact.flat_edge_count <= request.max_contact_edges &&
    array_bytes_fit (
      request.max_rectangles,
      sizeof (klayout_cuda_spatial_m1_width_space_polygon_v1)) &&
    array_bytes_fit (
      request.max_boundary_segments,
      sizeof (klayout_cuda_spatial_m2_union_segment_v1));
}

bool qualified_active3_well_union_request (
  const klayout_cuda_spatial_active3_well_union_request_v1 &request)
{
  return
    request.abi_version == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
    request.struct_size == sizeof (request) &&
    request.opcode ==
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_EMPTY &&
    request.option_flags ==
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_QUALIFIED_OPTIONS &&
    request.format_version == 1 && request.dbu_per_micron == 2000 &&
    request.device >= 0 && request.reserved0 == 0 &&
    request.distance == 110 && request.grid_cell_size == 2000 &&
    request.secondary_well_layer == 2 &&
    request.secondary_well_datatype == 0 &&
    request.layer_reserved == 0 && request.union_reserved == 0 &&
    request.reserved1 [0] == 0 && request.reserved1 [1] == 0 &&
    request.reserved1 [2] == 0 && request.reserved1 [3] == 0 &&
    request.max_contexts &&
    request.max_contexts <= std::numeric_limits<uint32_t>::max () &&
    request.max_rectangles && request.max_x_slabs &&
    request.max_union_memberships && request.max_events &&
    request.max_raw_segments && request.max_boundary_segments &&
    request.max_slabs_per_rectangle &&
    request.max_x_slabs <= std::numeric_limits<uint32_t>::max () &&
    request.max_active_edges &&
    request.max_active_edges <= std::numeric_limits<uint32_t>::max () &&
    request.max_grid_cells &&
    request.max_grid_cells <= std::numeric_limits<uint32_t>::max () &&
    request.max_active_memberships &&
    request.max_active_cell_visits &&
    request.max_member_visits && request.max_pair_work &&
    request.max_cells_per_active_edge &&
    request.max_cells_per_well_edge &&
    qualified_contact4_active_union_scene (
      request.wells,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_ROLE, 3,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_DIGEST_DOMAIN,
      request.max_contexts) &&
    qualified_contact4_active_union_scene (
      request.active,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_ROLE, 1,
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_DIGEST_DOMAIN,
      request.max_contexts) &&
    request.wells.format_version == request.format_version &&
    request.active.format_version == request.format_version &&
    request.wells.dbu_per_micron == request.dbu_per_micron &&
    request.active.dbu_per_micron == request.dbu_per_micron &&
    request.wells.root_cell == request.active.root_cell &&
    request.wells.context_count == request.active.context_count &&
    request.wells.cell_count == request.active.cell_count &&
    request.active.flat_edge_count <= request.max_active_edges &&
    array_bytes_fit (
      request.max_rectangles,
      sizeof (klayout_cuda_spatial_m1_width_space_polygon_v1)) &&
    array_bytes_fit (
      request.max_boundary_segments,
      sizeof (klayout_cuda_spatial_m2_union_segment_v1));
}

bool qualified_via1_stack_request (
  const klayout_cuda_spatial_via1_stack_request_v1 &request)
{
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size < sizeof (request) ||
      request.opcode != KLAYOUT_CUDA_SPATIAL_VIA1_STACK_EMPTY ||
      request.option_flags !=
        KLAYOUT_CUDA_SPATIAL_VIA1_STACK_QUALIFIED_OPTIONS ||
      request.requested_mask != KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES ||
      request.dbu_per_micron != 2000 || request.device < 0 ||
      request.reserved0 != 0 || request.reserved1 [0] != 0 ||
      request.reserved1 [1] != 0 ||
      request.enclosure_distance != 70 ||
      request.cut_width != 130 || request.cut_height != 130 ||
      request.spacing_distance != 150 ||
      request.grid_cell_size != 2000 ||
      ! request.context_count || ! request.contexts ||
      ! request.metal1_context_count || ! request.metal1_contexts ||
      request.metal1_offset_count != request.metal1_context_count ||
      ! request.metal1_offsets ||
      ! request.via1_context_count || ! request.via1_contexts ||
      request.via1_offset_count != request.via1_context_count ||
      ! request.via1_offsets ||
      ! request.metal2_context_count || ! request.metal2_contexts ||
      request.metal2_offset_count != request.metal2_context_count ||
      ! request.metal2_offsets ||
      ! request.cell_count || ! request.cells ||
      ! request.box_count || ! request.boxes ||
      ! request.flat_metal1_box_count ||
      ! request.flat_via1_box_count ||
      ! request.flat_metal2_box_count ||
      request.scene_left >= request.scene_right ||
      request.scene_bottom >= request.scene_top ||
      ! request.max_contexts || ! request.max_grid_cells ||
      ! request.max_metal_memberships ||
      ! request.max_via_memberships || ! request.max_pair_work ||
      request.context_count > request.max_contexts ||
      request.context_count > std::numeric_limits<uint32_t>::max () ||
      request.cell_count > std::numeric_limits<uint32_t>::max () ||
      request.flat_metal1_box_count >
        std::numeric_limits<uint32_t>::max () ||
      request.flat_via1_box_count >
        std::numeric_limits<uint32_t>::max () ||
      request.flat_metal2_box_count >
        std::numeric_limits<uint32_t>::max ()) {
    return false;
  }

  std::array<uint8_t, 32> digest;
  return db::cuda_via1_stack_digest::request_digest (request, digest) &&
         std::equal (
           digest.begin (), digest.end (), request.scene_digest);
}

bool qualified_implant12_request (
  const klayout_cuda_spatial_implant12_request_v1 &request)
{
  uint64_t expected_offset_count = 0;
  uint64_t flat_polygons = 0;
  uint64_t flat_contours = 0;
  uint64_t flat_edges = 0;
  uint64_t minimum_edges = 0;
  uint64_t minimum_implant_edges = 0;
  uint64_t minimum_gate_edges = 0;
  uint64_t minimum_contact_edges = 0;
  if (! checked_add_u64 (
        request.implant_context_count, 1, expected_offset_count) ||
      ! checked_add_u64 (
        request.flat_implant_polygon_count,
        request.flat_gate_polygon_count, flat_polygons) ||
      ! checked_add_u64 (
        flat_polygons, request.flat_contact_polygon_count, flat_polygons) ||
      ! checked_add_u64 (
        request.flat_implant_contour_count,
        request.flat_gate_contour_count, flat_contours) ||
      ! checked_add_u64 (
        flat_contours, request.flat_contact_contour_count, flat_contours) ||
      ! checked_add_u64 (
        request.flat_implant_edge_count,
        request.flat_gate_edge_count, flat_edges) ||
      ! checked_add_u64 (
        flat_edges, request.flat_contact_edge_count, flat_edges) ||
      ! checked_multiply_u64 (flat_contours, 4, minimum_edges) ||
      ! checked_multiply_u64 (
        request.flat_implant_contour_count, 4, minimum_implant_edges) ||
      ! checked_multiply_u64 (
        request.flat_gate_contour_count, 4, minimum_gate_edges) ||
      ! checked_multiply_u64 (
        request.flat_contact_contour_count, 4, minimum_contact_edges)) {
    return false;
  }

  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof (request) ||
      request.opcode !=
        KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_SUPERSET_EMPTY ||
      request.option_flags !=
        KLAYOUT_CUDA_SPATIAL_IMPLANT12_QUALIFIED_OPTIONS ||
      request.format_version != 1 ||
      request.dbu_per_micron != 2000 ||
      request.requested_mask != KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES ||
      request.device < 0 || request.reserved0 != 0 ||
      request.context_reserved != 0 || request.cell_reserved != 0 ||
      request.contour_reserved != 0 || request.edge_reserved != 0 ||
      request.reserved1 [0] != 0 || request.reserved1 [1] != 0 ||
      request.implant1_distance != 140 ||
      request.implant2_distance != 50 ||
      request.grid_cell_size != 2000 ||
      ! request.context_count || ! request.contexts ||
      request.context_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_context_v1) ||
      ! request.implant_context_count || ! request.implant_contexts ||
      request.implant_edge_offset_count != expected_offset_count ||
      ! request.implant_edge_offsets ||
      ! request.gate_context_count || ! request.gate_contexts ||
      ! request.contact_context_count || ! request.contact_contexts ||
      ! request.cell_count || ! request.cells ||
      request.cell_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_cell_v1) ||
      ! request.contour_count || ! request.contours ||
      request.contour_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_contour_v1) ||
      ! request.edge_count || ! request.edges ||
      request.edge_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_edge_v1) ||
      ! request.flat_implant_polygon_count ||
      ! request.flat_gate_polygon_count ||
      ! request.flat_contact_polygon_count ||
      request.flat_implant_polygon_count !=
        request.flat_implant_contour_count ||
      request.flat_gate_polygon_count !=
        request.flat_gate_contour_count ||
      request.flat_contact_polygon_count !=
        request.flat_contact_contour_count ||
      request.flat_implant_edge_count < minimum_implant_edges ||
      request.flat_gate_edge_count < minimum_gate_edges ||
      request.flat_contact_edge_count < minimum_contact_edges ||
      flat_edges < minimum_edges ||
      request.implant_left >= request.implant_right ||
      request.implant_bottom >= request.implant_top ||
      ! request.max_contexts || ! request.max_grid_cells ||
      ! request.max_implant_memberships ||
      ! request.max_gate_query_visits ||
      ! request.max_gate_candidate_work ||
      ! request.max_contact_query_visits ||
      ! request.max_contact_candidate_work ||
      ! request.max_flat_polygons ||
      ! request.max_flat_contours || ! request.max_flat_edges ||
      request.max_contexts > std::numeric_limits<uint32_t>::max () ||
      request.max_grid_cells > std::numeric_limits<uint32_t>::max () ||
      request.max_implant_memberships >
        std::numeric_limits<uint32_t>::max () ||
      request.context_count > request.max_contexts ||
      request.context_count > std::numeric_limits<uint32_t>::max () ||
      request.root_cell >= request.cell_count ||
      request.cell_count > request.context_count ||
      request.cell_count > std::numeric_limits<uint32_t>::max () ||
      request.contour_count > request.max_flat_contours ||
      request.contour_count > std::numeric_limits<uint32_t>::max () ||
      request.edge_count > request.max_flat_edges ||
      request.edge_count > std::numeric_limits<uint32_t>::max () ||
      request.implant_context_count > request.context_count ||
      request.gate_context_count > request.context_count ||
      request.contact_context_count > request.context_count ||
      request.flat_implant_edge_count >
        std::numeric_limits<uint32_t>::max () ||
      flat_polygons > request.max_flat_polygons ||
      flat_contours > request.max_flat_contours ||
      flat_edges > request.max_flat_edges ||
      request.implant_edge_offsets [0] != 0 ||
      request.implant_edge_offsets [request.implant_edge_offset_count - 1] !=
        request.flat_implant_edge_count) {
    return false;
  }

  if (! strictly_increasing_context_ids (
        request.implant_contexts, request.implant_context_count,
        request.context_count) ||
      ! strictly_increasing_context_ids (
        request.gate_contexts, request.gate_context_count,
        request.context_count) ||
      ! strictly_increasing_context_ids (
        request.contact_contexts, request.contact_context_count,
        request.context_count)) {
    return false;
  }
  for (uint64_t offset = 1;
       offset < request.implant_edge_offset_count; ++offset) {
    if (request.implant_edge_offsets [offset - 1] >=
        request.implant_edge_offsets [offset]) {
      return false;
    }
  }

  // Deliberately do not hash the (potentially very large) serialized scene
  // again in this wrapper.  The checks above qualify the scalar contract,
  // capacities, context lists, and implant offsets before dispatch.  The DSO
  // remains the trust boundary for complete record/topology validation and
  // recomputes and verifies scene_digest before launching any device work.
  return true;
}

uint64_t env_u64 (const char *name, uint64_t default_value)
{
  const char *value = std::getenv (name);
  if (! value || ! *value) {
    return default_value;
  }

  errno = 0;
  char *end = 0;
  unsigned long long parsed = std::strtoull (value, &end, 0);
  if (errno != 0 || ! end || *end != 0) {
    return default_value;
  }
  return static_cast<uint64_t> (parsed);
}

bool env_enabled (const char *name)
{
  const char *value = std::getenv (name);
  return value && *value && std::strcmp (value, "0") != 0 &&
         std::strcmp (value, "false") != 0 && std::strcmp (value, "off") != 0;
}

class CudaSpatialModule
{
public:
  CudaSpatialModule ()
    : m_enabled (false), m_telemetry (false),
      m_active3_enabled (env_enabled ("KLAYOUT_CUDA_ACTIVE3")),
      m_active3_telemetry (env_enabled ("KLAYOUT_CUDA_ACTIVE3_TELEMETRY")),
      m_active3_raw_wells_enabled (
        env_enabled ("KLAYOUT_CUDA_ACTIVE3_RAW_WELLS")),
      m_active3_raw_wells_telemetry (
        env_enabled ("KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_TELEMETRY")),
      m_active3_well_union_enabled (
        env_enabled ("KLAYOUT_CUDA_ACTIVE3_WELL_UNION")),
      m_active3_well_union_telemetry (
        env_enabled ("KLAYOUT_CUDA_ACTIVE3_WELL_UNION_TELEMETRY")),
      m_contact4_enabled (env_enabled ("KLAYOUT_CUDA_CONTACT4")),
      m_contact4_telemetry (
        env_enabled ("KLAYOUT_CUDA_CONTACT4_TELEMETRY")),
      m_contact4_raw_active_enabled (
        env_enabled ("KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE")),
      m_contact4_raw_active_telemetry (
        env_enabled ("KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY")),
      m_contact4_active_union_enabled (
        env_enabled ("KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION")),
      m_contact4_active_union_telemetry (
        env_enabled ("KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_TELEMETRY")),
      m_implant12_enabled (env_enabled ("KLAYOUT_CUDA_IMPLANT12")),
      m_implant12_telemetry (
        env_enabled ("KLAYOUT_CUDA_IMPLANT12_TELEMETRY")),
      m_m1_width_space_enabled (
        env_enabled ("KLAYOUT_CUDA_M1_WIDTH_SPACE")),
      m_m1_width_space_telemetry (
        env_enabled ("KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY")),
      m_m2_width_space_enabled (
        env_enabled ("KLAYOUT_CUDA_M2_WIDTH_SPACE")),
      m_m2_width_space_telemetry (
        env_enabled ("KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY")),
      m_m2_union_enabled (env_enabled ("KLAYOUT_CUDA_M2_RULES")),
      m_m2_union_telemetry (
        env_enabled ("KLAYOUT_CUDA_M2_RULES_TELEMETRY")),
      m_poly34_enabled (env_enabled ("KLAYOUT_CUDA_POLY34")),
      m_poly34_telemetry (
        env_enabled ("KLAYOUT_CUDA_POLY34_TELEMETRY")),
      m_via1_stack_enabled (env_enabled ("KLAYOUT_CUDA_VIA1_STACK")),
      m_via1_stack_telemetry (
        env_enabled ("KLAYOUT_CUDA_VIA1_STACK_TELEMETRY")),
      m_m1_contact_enabled (env_enabled ("KLAYOUT_CUDA_M1_CONTACT")),
      m_m1_contact_telemetry (
        env_enabled ("KLAYOUT_CUDA_M1_CONTACT_TELEMETRY")),
      m_handle (0), m_run_bipartite (0), m_run_self (0),
      m_run_active3 (0), m_run_contact4_raw_active (0),
      m_run_contact4_active_union (0),
      m_run_active3_well_union (0),
      m_run_implant12 (0), m_run_m1_width_space (0),
      m_run_m2_width_space (0), m_run_m2_union (0),
      m_release_m2_union (0), m_run_poly34 (0),
      m_run_via1_stack (0), m_release (0),
      m_min_records (100000)
  {
    const char *setting = std::getenv ("KLAYOUT_CUDA_SPATIAL_BACKEND");
    if (! setting || ! *setting || std::strcmp (setting, "0") == 0 ||
        std::strcmp (setting, "false") == 0 || std::strcmp (setting, "off") == 0) {
      return;
    }

    m_enabled = true;
    m_telemetry = env_enabled ("KLAYOUT_CUDA_SPATIAL_TELEMETRY");
    m_min_records = env_u64 ("KLAYOUT_CUDA_SPATIAL_MIN_RECORDS", m_min_records);

    std::string path (setting);
    if (path == "1" || path == "auto") {
#if defined(_WIN32)
      path = "klayout_cuda_spatial_backend.dll";
#elif defined(__APPLE__)
      path = "libklayout_cuda_spatial_backend.dylib";
#else
      path = "libklayout_cuda_spatial_backend.so";
#endif
    }

#if defined(_WIN32)
    m_handle = reinterpret_cast<void *> (LoadLibraryA (path.c_str ()));
    if (m_handle) {
      klayout_cuda_spatial_abi_version_func version =
        reinterpret_cast<klayout_cuda_spatial_abi_version_func> (
          GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_abi_version"));
      m_run_bipartite = reinterpret_cast<klayout_cuda_spatial_run_bipartite_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_run_bipartite_v1"));
      m_run_self = reinterpret_cast<klayout_cuda_spatial_run_self_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_run_self_v1"));
      m_run_active3 = reinterpret_cast<klayout_cuda_spatial_run_active3_empty_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_run_active3_empty_v1"));
      m_run_contact4_raw_active =
        reinterpret_cast<
          klayout_cuda_spatial_run_contact4_raw_active_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_contact4_raw_active_empty_v1"));
      m_run_contact4_active_union =
        reinterpret_cast<
          klayout_cuda_spatial_run_contact4_active_union_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_contact4_active_union_empty_v1"));
      m_run_active3_well_union =
        reinterpret_cast<
          klayout_cuda_spatial_run_active3_well_union_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_active3_well_union_empty_v1"));
      m_run_implant12 =
        reinterpret_cast<klayout_cuda_spatial_run_implant12_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_implant12_empty_v1"));
      m_run_m1_width_space =
        reinterpret_cast<klayout_cuda_spatial_run_m1_width_space_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_m1_width_space_empty_v1"));
      m_run_m2_width_space =
        reinterpret_cast<klayout_cuda_spatial_run_m2_width_space_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_m2_width_space_empty_v1"));
      m_run_m2_union =
        reinterpret_cast<klayout_cuda_spatial_run_m2_union_boundary_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_m2_union_boundary_v1"));
      m_release_m2_union =
        reinterpret_cast<
          klayout_cuda_spatial_release_m2_union_boundary_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_release_m2_union_boundary_v1"));
      m_run_poly34 =
        reinterpret_cast<klayout_cuda_spatial_run_poly34_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_poly34_empty_v1"));
      m_run_via1_stack =
        reinterpret_cast<klayout_cuda_spatial_run_via1_stack_empty_v1_func> (
          GetProcAddress (
            reinterpret_cast<HMODULE> (m_handle),
            "klayout_cuda_spatial_run_via1_stack_empty_v1"));
      m_release = reinterpret_cast<klayout_cuda_spatial_release_result_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_release_result_v1"));
      if (! version || version () != KLAYOUT_CUDA_SPATIAL_ABI_VERSION) {
        m_error = "CUDA spatial backend has an incompatible ABI";
      }
    } else {
      m_error = "unable to load CUDA spatial backend: " + path;
    }
#else
    m_handle = dlopen (path.c_str (), RTLD_NOW | RTLD_LOCAL);
    if (m_handle) {
      klayout_cuda_spatial_abi_version_func version =
        reinterpret_cast<klayout_cuda_spatial_abi_version_func> (
          dlsym (m_handle, "klayout_cuda_spatial_abi_version"));
      m_run_bipartite = reinterpret_cast<klayout_cuda_spatial_run_bipartite_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_run_bipartite_v1"));
      m_run_self = reinterpret_cast<klayout_cuda_spatial_run_self_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_run_self_v1"));
      m_run_active3 = reinterpret_cast<klayout_cuda_spatial_run_active3_empty_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_run_active3_empty_v1"));
      m_run_contact4_raw_active =
        reinterpret_cast<
          klayout_cuda_spatial_run_contact4_raw_active_empty_v1_func> (
          dlsym (
            m_handle,
            "klayout_cuda_spatial_run_contact4_raw_active_empty_v1"));
      m_run_contact4_active_union =
        reinterpret_cast<
          klayout_cuda_spatial_run_contact4_active_union_empty_v1_func> (
          dlsym (
            m_handle,
            "klayout_cuda_spatial_run_contact4_active_union_empty_v1"));
      m_run_active3_well_union =
        reinterpret_cast<
          klayout_cuda_spatial_run_active3_well_union_empty_v1_func> (
          dlsym (
            m_handle,
            "klayout_cuda_spatial_run_active3_well_union_empty_v1"));
      m_run_implant12 =
        reinterpret_cast<klayout_cuda_spatial_run_implant12_empty_v1_func> (
          dlsym (
            m_handle, "klayout_cuda_spatial_run_implant12_empty_v1"));
      m_run_m1_width_space =
        reinterpret_cast<klayout_cuda_spatial_run_m1_width_space_empty_v1_func> (
          dlsym (
            m_handle,
            "klayout_cuda_spatial_run_m1_width_space_empty_v1"));
      m_run_m2_width_space =
        reinterpret_cast<klayout_cuda_spatial_run_m2_width_space_empty_v1_func> (
          dlsym (
            m_handle,
            "klayout_cuda_spatial_run_m2_width_space_empty_v1"));
      m_run_m2_union =
        reinterpret_cast<klayout_cuda_spatial_run_m2_union_boundary_v1_func> (
          dlsym (
            m_handle,
            "klayout_cuda_spatial_run_m2_union_boundary_v1"));
      m_release_m2_union =
        reinterpret_cast<
          klayout_cuda_spatial_release_m2_union_boundary_v1_func> (
          dlsym (
            m_handle,
            "klayout_cuda_spatial_release_m2_union_boundary_v1"));
      m_run_poly34 =
        reinterpret_cast<klayout_cuda_spatial_run_poly34_empty_v1_func> (
          dlsym (
            m_handle, "klayout_cuda_spatial_run_poly34_empty_v1"));
      m_run_via1_stack =
        reinterpret_cast<klayout_cuda_spatial_run_via1_stack_empty_v1_func> (
          dlsym (
            m_handle, "klayout_cuda_spatial_run_via1_stack_empty_v1"));
      m_release = reinterpret_cast<klayout_cuda_spatial_release_result_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_release_result_v1"));
      if (! version || version () != KLAYOUT_CUDA_SPATIAL_ABI_VERSION) {
        m_error = "CUDA spatial backend has an incompatible ABI";
      }
    } else {
      const char *error = dlerror ();
      m_error = std::string ("unable to load CUDA spatial backend: ") +
                (error ? error : path.c_str ());
    }
#endif

    if (! m_error.empty ()) {
      m_run_bipartite = 0;
      m_run_self = 0;
      m_run_active3 = 0;
      m_run_contact4_raw_active = 0;
      m_run_contact4_active_union = 0;
      m_run_active3_well_union = 0;
      m_run_implant12 = 0;
      m_run_m1_width_space = 0;
      m_run_m2_width_space = 0;
      m_run_m2_union = 0;
      m_release_m2_union = 0;
      m_run_poly34 = 0;
      m_run_via1_stack = 0;
      m_release = 0;
      tl::warn << m_error;
    } else if (m_telemetry) {
      tl::info << "CUDA spatial backend loaded: " << path;
    }
  }

  bool enabled () const
  {
    return m_enabled;
  }

  bool ready () const
  {
    return m_run_bipartite && m_release;
  }

  bool self_ready () const
  {
    return m_run_self && m_release;
  }

  bool active3_ready () const
  {
    return m_active3_enabled && m_run_active3;
  }

  bool active3_enabled () const
  {
    return m_active3_enabled;
  }

  bool active3_raw_wells_ready () const
  {
    return m_active3_raw_wells_enabled && m_run_active3;
  }

  bool active3_raw_wells_enabled () const
  {
    return m_active3_raw_wells_enabled;
  }

  bool active3_well_union_ready () const
  {
    return m_active3_well_union_enabled && m_run_active3_well_union;
  }

  bool active3_well_union_enabled () const
  {
    return m_active3_well_union_enabled;
  }

  bool contact4_ready () const
  {
    return m_contact4_enabled && m_run_active3;
  }

  bool contact4_raw_active_ready () const
  {
    return m_contact4_raw_active_enabled && m_run_contact4_raw_active;
  }

  bool contact4_raw_active_enabled () const
  {
    return m_contact4_raw_active_enabled;
  }

  bool contact4_active_union_ready () const
  {
    return m_contact4_active_union_enabled && m_run_contact4_active_union;
  }

  bool contact4_active_union_enabled () const
  {
    return m_contact4_active_union_enabled;
  }

  bool contact4_enabled () const
  {
    return m_contact4_enabled;
  }

  bool implant12_ready () const
  {
    return m_implant12_enabled && m_run_implant12;
  }

  bool implant12_enabled () const
  {
    return m_implant12_enabled;
  }

  bool implant12_telemetry () const
  {
    return m_implant12_telemetry;
  }

  bool via1_stack_ready () const
  {
    return m_via1_stack_enabled && m_run_via1_stack;
  }

  bool m1_width_space_ready () const
  {
    return m_m1_width_space_enabled && m_run_m1_width_space;
  }

  bool m1_width_space_enabled () const
  {
    return m_m1_width_space_enabled;
  }

  bool m1_width_space_telemetry () const
  {
    return m_m1_width_space_telemetry;
  }

  bool m2_width_space_ready () const
  {
    return m_m2_width_space_enabled && m_run_m2_width_space;
  }

  bool m2_width_space_enabled () const
  {
    return m_m2_width_space_enabled;
  }

  bool m2_width_space_telemetry () const
  {
    return m_m2_width_space_telemetry;
  }

  bool m2_union_ready () const
  {
    return m_m2_union_enabled && m_run_m2_union && m_release_m2_union;
  }

  bool m2_union_enabled () const
  {
    return m_m2_union_enabled;
  }

  bool m2_union_telemetry () const
  {
    return m_m2_union_telemetry;
  }

  bool poly34_ready () const
  {
    return m_poly34_enabled && m_run_poly34;
  }

  bool poly34_enabled () const
  {
    return m_poly34_enabled;
  }

  bool poly34_telemetry () const
  {
    return m_poly34_telemetry;
  }

  bool via1_stack_enabled () const
  {
    return m_via1_stack_enabled;
  }

  bool m1_contact_ready () const
  {
    return m_m1_contact_enabled && m_run_via1_stack;
  }

  bool m1_contact_enabled () const
  {
    return m_m1_contact_enabled;
  }

  bool m1_contact_telemetry () const
  {
    return m_m1_contact_telemetry;
  }

  bool telemetry () const
  {
    return m_telemetry;
  }

  bool active3_telemetry () const
  {
    return m_active3_telemetry;
  }

  bool active3_raw_wells_telemetry () const
  {
    return m_active3_raw_wells_telemetry;
  }

  bool active3_well_union_telemetry () const
  {
    return m_active3_well_union_telemetry;
  }

  bool contact4_telemetry () const
  {
    return m_contact4_telemetry;
  }

  bool contact4_raw_active_telemetry () const
  {
    return m_contact4_raw_active_telemetry;
  }

  bool contact4_active_union_telemetry () const
  {
    return m_contact4_active_union_telemetry;
  }

  bool via1_stack_telemetry () const
  {
    return m_via1_stack_telemetry;
  }

  uint64_t min_records () const
  {
    return m_min_records;
  }

  const std::string &error () const
  {
    return m_error;
  }

  klayout_cuda_spatial_run_bipartite_v1_func run () const
  {
    return m_run_bipartite;
  }

  klayout_cuda_spatial_run_self_v1_func run_self () const
  {
    return m_run_self;
  }

  klayout_cuda_spatial_run_active3_empty_v1_func run_active3 () const
  {
    return m_run_active3;
  }

  klayout_cuda_spatial_run_contact4_raw_active_empty_v1_func
  run_contact4_raw_active () const
  {
    return m_run_contact4_raw_active;
  }

  klayout_cuda_spatial_run_contact4_active_union_empty_v1_func
  run_contact4_active_union () const
  {
    return m_run_contact4_active_union;
  }

  klayout_cuda_spatial_run_active3_well_union_empty_v1_func
  run_active3_well_union () const
  {
    return m_run_active3_well_union;
  }

  klayout_cuda_spatial_run_implant12_empty_v1_func run_implant12 () const
  {
    return m_run_implant12;
  }

  klayout_cuda_spatial_run_m1_width_space_empty_v1_func
  run_m1_width_space () const
  {
    return m_run_m1_width_space;
  }

  klayout_cuda_spatial_run_m2_width_space_empty_v1_func
  run_m2_width_space () const
  {
    return m_run_m2_width_space;
  }

  klayout_cuda_spatial_run_m2_union_boundary_v1_func
  run_m2_union () const
  {
    return m_run_m2_union;
  }

  klayout_cuda_spatial_release_m2_union_boundary_v1_func
  release_m2_union () const
  {
    return m_release_m2_union;
  }

  klayout_cuda_spatial_run_poly34_empty_v1_func run_poly34 () const
  {
    return m_run_poly34;
  }

  klayout_cuda_spatial_run_via1_stack_empty_v1_func run_via1_stack () const
  {
    return m_run_via1_stack;
  }

  klayout_cuda_spatial_release_result_v1_func release () const
  {
    return m_release;
  }

private:
  bool m_enabled;
  bool m_telemetry;
  bool m_active3_enabled;
  bool m_active3_telemetry;
  bool m_active3_raw_wells_enabled;
  bool m_active3_raw_wells_telemetry;
  bool m_active3_well_union_enabled;
  bool m_active3_well_union_telemetry;
  bool m_contact4_enabled;
  bool m_contact4_telemetry;
  bool m_contact4_raw_active_enabled;
  bool m_contact4_raw_active_telemetry;
  bool m_contact4_active_union_enabled;
  bool m_contact4_active_union_telemetry;
  bool m_implant12_enabled;
  bool m_implant12_telemetry;
  bool m_m1_width_space_enabled;
  bool m_m1_width_space_telemetry;
  bool m_m2_width_space_enabled;
  bool m_m2_width_space_telemetry;
  bool m_m2_union_enabled;
  bool m_m2_union_telemetry;
  bool m_poly34_enabled;
  bool m_poly34_telemetry;
  bool m_via1_stack_enabled;
  bool m_via1_stack_telemetry;
  bool m_m1_contact_enabled;
  bool m_m1_contact_telemetry;
  void *m_handle;
  klayout_cuda_spatial_run_bipartite_v1_func m_run_bipartite;
  klayout_cuda_spatial_run_self_v1_func m_run_self;
  klayout_cuda_spatial_run_active3_empty_v1_func m_run_active3;
  klayout_cuda_spatial_run_contact4_raw_active_empty_v1_func
    m_run_contact4_raw_active;
  klayout_cuda_spatial_run_contact4_active_union_empty_v1_func
    m_run_contact4_active_union;
  klayout_cuda_spatial_run_active3_well_union_empty_v1_func
    m_run_active3_well_union;
  klayout_cuda_spatial_run_implant12_empty_v1_func m_run_implant12;
  klayout_cuda_spatial_run_m1_width_space_empty_v1_func m_run_m1_width_space;
  klayout_cuda_spatial_run_m2_width_space_empty_v1_func m_run_m2_width_space;
  klayout_cuda_spatial_run_m2_union_boundary_v1_func m_run_m2_union;
  klayout_cuda_spatial_release_m2_union_boundary_v1_func
    m_release_m2_union;
  klayout_cuda_spatial_run_poly34_empty_v1_func m_run_poly34;
  klayout_cuda_spatial_run_via1_stack_empty_v1_func m_run_via1_stack;
  klayout_cuda_spatial_release_result_v1_func m_release;
  uint64_t m_min_records;
  std::string m_error;
};

CudaSpatialModule &cuda_spatial_module ()
{
  static CudaSpatialModule module;
  return module;
}

void log_attempt (const CudaSpatialAttempt &attempt, uint64_t subjects,
                  uint64_t intruders, const char *mode = "bipartite")
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaSpatialAttempt::Success: outcome = "success"; break;
  case CudaSpatialAttempt::BackendFallback: outcome = "fallback"; break;
  case CudaSpatialAttempt::BackendError: outcome = "error"; break;
  case CudaSpatialAttempt::InvalidResult: outcome = "invalid-result"; break;
  case CudaSpatialAttempt::BelowThreshold: outcome = "below-threshold"; break;
  case CudaSpatialAttempt::Disabled: outcome = "disabled"; break;
  }

  tl::info << "CUDA spatial broad phase: mode=" << mode
           << " outcome=" << outcome
           << " subjects=" << subjects << " intruders=" << intruders
           << " pairs=" << attempt.pair_keys.size ()
           << " memberships=" << attempt.membership_count
           << " pair_work=" << attempt.pair_work_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << (attempt.message.empty () ? "" : " message=") << attempt.message;
}

void log_active3_attempt (const CudaActive3Attempt &attempt, bool raw_wells)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (raw_wells
        ? ! module.active3_raw_wells_telemetry ()
        : ! module.active3_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaActive3Attempt::CertifiedEmpty: outcome = "certified-empty"; break;
  case CudaActive3Attempt::RawHits: outcome = "raw-hits-cpu-fallback"; break;
  case CudaActive3Attempt::BackendFallback: outcome = "fallback"; break;
  case CudaActive3Attempt::BackendError: outcome = "error"; break;
  case CudaActive3Attempt::InvalidResult: outcome = "invalid-result"; break;
  case CudaActive3Attempt::Disabled: outcome = "disabled"; break;
  }

  tl::info << (raw_wells
                ? "CUDA ACTIVE.3 raw-WELL empty certificate:"
                : "CUDA ACTIVE.3 empty certificate:")
           << " outcome=" << outcome
           << " contexts=" << attempt.context_count
           << " well_edges=" << attempt.flat_well_edge_count
           << " active_edges=" << attempt.flat_active_edge_count
           << " candidates=" << attempt.candidate_pair_count
           << " raw_hits=" << attempt.raw_hit_count
           << " uncertain=" << attempt.uncertain_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=") << attempt.message;
}

void log_contact4_attempt (const CudaActive3Attempt &attempt, bool raw_active)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (raw_active
        ? ! module.contact4_raw_active_telemetry ()
        : ! module.contact4_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaActive3Attempt::CertifiedEmpty: outcome = "certified-empty"; break;
  case CudaActive3Attempt::RawHits: outcome = "raw-hits-cpu-fallback"; break;
  case CudaActive3Attempt::BackendFallback: outcome = "fallback"; break;
  case CudaActive3Attempt::BackendError: outcome = "error"; break;
  case CudaActive3Attempt::InvalidResult: outcome = "invalid-result"; break;
  case CudaActive3Attempt::Disabled: outcome = "disabled"; break;
  }

  tl::info << (raw_active
                ? "CUDA CONTACT.4 raw-ACTIVE empty certificate:"
                : "CUDA CONTACT.4 empty certificate:")
           << " outcome=" << outcome
           << " contexts=" << attempt.context_count
           << " indexed_contact_edges=" << attempt.flat_well_edge_count
           << " streamed_active_edges=" << attempt.flat_active_edge_count
           << " candidates=" << attempt.candidate_pair_count
           << " raw_hits=" << attempt.raw_hit_count
           << " uncertain=" << attempt.uncertain_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=") << attempt.message;
}

void log_active3_profile_attempt (const CudaActive3Attempt &attempt,
                                  bool contact4, bool raw_contact4,
                                  bool raw_wells)
{
  if (contact4) {
    log_contact4_attempt (attempt, raw_contact4);
  } else {
    log_active3_attempt (attempt, raw_wells);
  }
}

void log_contact4_active_union_attempt (
  const CudaContact4ActiveUnionAttempt &attempt)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.contact4_active_union_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaContact4ActiveUnionAttempt::CertifiedEmpty:
    outcome = "certified-empty";
    break;
  case CudaContact4ActiveUnionAttempt::RawHits:
    outcome = "raw-hits-cpu-fallback";
    break;
  case CudaContact4ActiveUnionAttempt::BackendFallback:
    outcome = "fallback";
    break;
  case CudaContact4ActiveUnionAttempt::BackendError:
    outcome = "error";
    break;
  case CudaContact4ActiveUnionAttempt::InvalidResult:
    outcome = "invalid-result";
    break;
  case CudaContact4ActiveUnionAttempt::Disabled:
    outcome = "disabled";
    break;
  }

  tl::info << "CUDA CONTACT.4 fused ACTIVE-union empty certificate:"
           << " outcome=" << outcome
           << " active_contexts=" << attempt.active_context_count
           << " contact_contexts=" << attempt.contact_context_count
           << " active_polygons=" << attempt.flat_active_polygon_count
           << " contact_edges=" << attempt.flat_contact_edge_count
           << " rectangles=" << attempt.rectangle_count
           << " boundary_segments=" << attempt.boundary_segment_count
           << " grid_cells=" << attempt.grid_cell_count
           << " memberships=" << attempt.contact_membership_count
           << " boundary_visits=" << attempt.boundary_cell_visit_count
           << " member_visits=" << attempt.member_visit_count
           << " candidates=" << attempt.candidate_pair_count
           << " raw_hits=" << attempt.raw_hit_count
           << " uncertain=" << attempt.uncertain_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=")
           << attempt.message;
}

void log_active3_well_union_attempt (
  const CudaActive3WellUnionAttempt &attempt)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.active3_well_union_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaActive3WellUnionAttempt::CertifiedEmpty:
    outcome = "certified-empty";
    break;
  case CudaActive3WellUnionAttempt::RawHits:
    outcome = "raw-hits-cpu-fallback";
    break;
  case CudaActive3WellUnionAttempt::BackendFallback:
    outcome = "fallback";
    break;
  case CudaActive3WellUnionAttempt::BackendError:
    outcome = "error";
    break;
  case CudaActive3WellUnionAttempt::InvalidResult:
    outcome = "invalid-result";
    break;
  case CudaActive3WellUnionAttempt::Disabled:
    outcome = "disabled";
    break;
  }

  tl::info << "CUDA ACTIVE.3 exact resident WELL-union certificate:"
           << " outcome=" << outcome
           << " well_contexts=" << attempt.well_context_count
           << " active_contexts=" << attempt.active_context_count
           << " well_polygons=" << attempt.flat_well_polygon_count
           << " active_edges=" << attempt.flat_active_edge_count
           << " rectangles=" << attempt.rectangle_count
           << " boundary_segments=" << attempt.boundary_segment_count
           << " grid_cells=" << attempt.grid_cell_count
           << " memberships=" << attempt.active_membership_count
           << " active_visits=" << attempt.active_cell_visit_count
           << " member_visits=" << attempt.member_visit_count
           << " candidates=" << attempt.candidate_pair_count
           << " raw_hits=" << attempt.raw_hit_count
           << " uncertain=" << attempt.uncertain_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=")
           << attempt.message;
}

void log_metal_width_space_attempt (
  const CudaM1WidthSpaceAttempt &attempt, bool metal2)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! (metal2 ? module.m2_width_space_telemetry ()
                : module.m1_width_space_telemetry ())) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaM1WidthSpaceAttempt::CertifiedEmpty:
    outcome = "certified-empty";
    break;
  case CudaM1WidthSpaceAttempt::RawHits:
    outcome = "raw-hits-cpu-fallback";
    break;
  case CudaM1WidthSpaceAttempt::BackendFallback:
    outcome = "fallback";
    break;
  case CudaM1WidthSpaceAttempt::BackendError:
    outcome = "error";
    break;
  case CudaM1WidthSpaceAttempt::InvalidResult:
    outcome = "invalid-result";
    break;
  case CudaM1WidthSpaceAttempt::Disabled:
    outcome = "disabled";
    break;
  }

  tl::info << (metal2
                ? "CUDA M2 width/space empty certificate:"
                : "CUDA M1 width/space empty certificate:")
           << " outcome=" << outcome
           << " contexts=" << attempt.context_count
           << " metal_contexts=" << attempt.metal_context_count
           << " polygons=" << attempt.flat_polygon_count
           << " edges=" << attempt.flat_edge_count
           << " grid_cells=" << attempt.grid_cell_count
           << " memberships=" << attempt.membership_count
           << " pair_work=" << attempt.pair_work_count
           << " unique_pairs=" << attempt.unique_edge_pair_count
           << " width_pairs=" << attempt.width_pair_count
           << " space_pairs=" << attempt.space_pair_count
           << " width_hits=" << attempt.width_hit_count
           << " space_hits=" << attempt.space_hit_count
           << " uncertain="
           << (attempt.width_uncertain_count +
               attempt.space_uncertain_count)
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=")
           << attempt.message;
}

void log_m2_union_attempt (const CudaM2UnionAttempt &attempt)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.m2_union_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaM2UnionAttempt::Complete: outcome = "complete"; break;
  case CudaM2UnionAttempt::BackendFallback: outcome = "fallback"; break;
  case CudaM2UnionAttempt::BackendError: outcome = "error"; break;
  case CudaM2UnionAttempt::InvalidResult: outcome = "invalid-result"; break;
  case CudaM2UnionAttempt::Disabled: outcome = "disabled"; break;
  }

  tl::info << "CUDA M2 exact union boundary:"
           << " outcome=" << outcome
           << " contexts=" << attempt.context_count
           << " metal_contexts=" << attempt.metal_context_count
           << " rectangles=" << attempt.rectangle_count
           << " slabs=" << attempt.x_slab_count
           << " memberships=" << attempt.membership_count
           << " events=" << attempt.event_count
           << " strip_intervals=" << attempt.strip_interval_count
           << " raw_segments=" << attempt.raw_segment_count
           << " segments=" << attempt.segments.size ()
           << " fnv64=" << attempt.boundary_fnv64
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=")
           << attempt.message;
}

void log_poly34_attempt (const CudaPoly34Attempt &attempt)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.poly34_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaPoly34Attempt::CertifiedEmpty:
    outcome = "certified-empty";
    break;
  case CudaPoly34Attempt::NotEmpty:
    outcome = "not-empty-cpu-fallback";
    break;
  case CudaPoly34Attempt::BackendFallback:
    outcome = "fallback";
    break;
  case CudaPoly34Attempt::BackendError:
    outcome = "error";
    break;
  case CudaPoly34Attempt::InvalidResult:
    outcome = "invalid-result";
    break;
  case CudaPoly34Attempt::Disabled:
    outcome = "disabled";
    break;
  }

  tl::info << "CUDA POLY.3/.4 terminal-empty certificate:"
           << " outcome=" << outcome
           << " contexts=" << attempt.context_count
           << " poly_boxes=" << attempt.flat_poly_box_count
           << " active_boxes=" << attempt.flat_active_box_count
           << " gates=" << attempt.flat_gate_box_count
           << " certified_mask=" << attempt.certified_empty_mask
           << " poly_candidates=" << attempt.poly_candidate_count
           << " active_candidates=" << attempt.active_candidate_count
           << " atomic_empty=" << attempt.atomic_terminal_empty_count
           << " fallback_gates=" << attempt.fallback_gate_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=")
           << attempt.message;
}

void log_via1_stack_attempt (const CudaVia1StackAttempt &attempt)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.via1_stack_telemetry () &&
      ! module.m1_contact_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaVia1StackAttempt::CertifiedEmpty:
    outcome = "certified-empty";
    break;
  case CudaVia1StackAttempt::NotEmpty:
    outcome = "not-empty-cpu-fallback";
    break;
  case CudaVia1StackAttempt::BackendFallback:
    outcome = "fallback";
    break;
  case CudaVia1StackAttempt::BackendError:
    outcome = "error";
    break;
  case CudaVia1StackAttempt::InvalidResult:
    outcome = "invalid-result";
    break;
  case CudaVia1StackAttempt::Disabled:
    outcome = "disabled";
    break;
  }

  tl::info << "CUDA VIA1 stack empty certificate:"
           << " outcome=" << outcome
           << " contexts=" << attempt.context_count
           << " m1_boxes=" << attempt.flat_metal1_box_count
           << " vias=" << attempt.flat_via1_box_count
           << " m2_boxes=" << attempt.flat_metal2_box_count
           << " certified_mask=" << attempt.certified_empty_mask
           << " via_pairs=" << attempt.via_candidate_pair_count
           << " unsafe_pairs=" << attempt.unsafe_via_pair_count
           << " spacing_errors=" << attempt.spacing_violation_count
           << " m1_misses=" << attempt.metal1_miss_count
           << " m2_misses=" << attempt.metal2_miss_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=") << attempt.message;
}

void log_implant12_attempt (const CudaImplant12Attempt &attempt)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.implant12_telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaImplant12Attempt::CertifiedEmpty:
    outcome = "certified-empty";
    break;
  case CudaImplant12Attempt::RawHits:
    outcome = "raw-hits-cpu-fallback";
    break;
  case CudaImplant12Attempt::BackendFallback:
    outcome = "fallback";
    break;
  case CudaImplant12Attempt::BackendError:
    outcome = "error";
    break;
  case CudaImplant12Attempt::InvalidResult:
    outcome = "invalid-result";
    break;
  case CudaImplant12Attempt::Disabled:
    outcome = "disabled";
    break;
  }

  tl::info << "CUDA IMPLANT.1/.2 empty certificate:"
           << " outcome=" << outcome
           << " contexts=" << attempt.context_count
           << " implant_edges=" << attempt.flat_implant_edge_count
           << " gate_edges=" << attempt.flat_gate_edge_count
           << " contact_edges=" << attempt.flat_contact_edge_count
           << " certified_mask=" << attempt.certified_empty_mask
           << " clean_mask=" << attempt.clean_mask
           << " grid_cells=" << attempt.grid_cell_count
           << " memberships=" << attempt.implant_membership_count
           << " gate_candidates=" << attempt.gate_candidate_count
           << " gate_hits=" << attempt.gate_raw_hit_count
           << " gate_uncertain=" << attempt.gate_uncertain_count
           << " contact_candidates=" << attempt.contact_candidate_count
           << " contact_hits=" << attempt.contact_raw_hit_count
           << " contact_uncertain=" << attempt.contact_uncertain_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << " device_flags=" << attempt.device_flags
           << (attempt.message.empty () ? "" : " message=")
           << attempt.message;
}

klayout_cuda_spatial_config_v1 make_config ()
{
  klayout_cuda_spatial_config_v1 config;
  std::memset (&config, 0, sizeof (config));
  config.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  config.struct_size = sizeof (config);
  config.device = int32_t (env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0));
  config.cell_size = env_u64 ("KLAYOUT_CUDA_SPATIAL_CELL_SIZE", 128);
  config.max_cells_per_record = uint32_t (std::min<uint64_t> (
    env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD", 64),
    std::numeric_limits<uint32_t>::max ()));
  config.max_records_per_cell = uint32_t (std::min<uint64_t> (
    env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL", 4096),
    std::numeric_limits<uint32_t>::max ()));
  config.max_memberships = env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS", 16000000);
  config.max_pair_work = env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK", 64000000);
  config.max_candidates = env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES", 8000000);
  return config;
}

int64_t floor_div_i64 (int64_t value, int64_t divisor)
{
  int64_t quotient = value / divisor;
  if (value % divisor < 0) {
    --quotient;
  }
  return quotient;
}

bool self_memberships_fit (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement, const klayout_cuda_spatial_config_v1 &config,
  uint64_t &membership_count)
{
  membership_count = 0;
  if (records.empty () || enlargement < 0 || config.cell_size == 0 ||
      config.cell_size > uint64_t (std::numeric_limits<int64_t>::max ()) ||
      config.max_cells_per_record == 0 || config.max_memberships == 0) {
    return false;
  }

  const int64_t cell_size = int64_t (config.cell_size);
  for (std::vector<klayout_cuda_spatial_aabb_v1>::const_iterator record =
         records.begin ();
       record != records.end (); ++record) {
    if (record->left > record->right || record->bottom > record->top ||
        record->left < std::numeric_limits<int64_t>::min () + enlargement ||
        record->bottom < std::numeric_limits<int64_t>::min () + enlargement ||
        record->right > std::numeric_limits<int64_t>::max () - enlargement ||
        record->top > std::numeric_limits<int64_t>::max () - enlargement) {
      return false;
    }

    const int64_t x0 = floor_div_i64 (record->left - enlargement, cell_size);
    const int64_t x1 = floor_div_i64 (record->right + enlargement, cell_size);
    const int64_t y0 = floor_div_i64 (record->bottom - enlargement, cell_size);
    const int64_t y1 = floor_div_i64 (record->top + enlargement, cell_size);
    const uint64_t dx = uint64_t (x1) - uint64_t (x0);
    const uint64_t dy = uint64_t (y1) - uint64_t (y0);
    if (dx == std::numeric_limits<uint64_t>::max () ||
        dy == std::numeric_limits<uint64_t>::max ()) {
      return false;
    }

    const uint64_t width = dx + 1;
    const uint64_t height = dy + 1;
    const uint64_t per_record_limit = config.max_cells_per_record;
    if (width > per_record_limit || height > per_record_limit ||
        width > per_record_limit / height) {
      return false;
    }

    const uint64_t record_memberships = width * height;
    if (record_memberships > config.max_memberships - membership_count) {
      return false;
    }
    membership_count += record_memberships;
  }

  return true;
}

void copy_result_metadata (CudaSpatialAttempt &attempt,
                           const klayout_cuda_spatial_result_v1 &result)
{
  attempt.fallback_flags = result.fallback_flags;
  attempt.membership_count = result.membership_count;
  attempt.occupied_cell_count = result.occupied_cell_count;
  attempt.pair_work_count = result.pair_work_count;
  attempt.setup_ns = result.setup_ns;
  attempt.h2d_ns = result.h2d_ns;
  attempt.broad_phase_ns = result.broad_phase_ns;
  attempt.sort_unique_ns = result.sort_unique_ns;
  attempt.d2h_ns = result.d2h_ns;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (result.message, result.message + sizeof (result.message), '\0'));
}

bool valid_pair_keys (const klayout_cuda_spatial_result_v1 &result,
                      const klayout_cuda_spatial_config_v1 &config,
                      uint64_t subject_count, uint64_t intruder_count,
                      bool bipartite)
{
  if ((result.pair_count != 0 && ! result.pair_keys) ||
      result.pair_count > config.max_candidates ||
      result.pair_count > std::numeric_limits<size_t>::max ()) {
    return false;
  }

  uint64_t previous = 0;
  const uint64_t total_count = subject_count + intruder_count;
  for (uint64_t i = 0; i < result.pair_count; ++i) {
    const uint64_t key = result.pair_keys[i];
    const uint64_t first = key >> 32;
    const uint64_t second = key & uint64_t (0xffffffff);
    if ((i != 0 && key <= previous) || first == 0 || second == 0) {
      return false;
    }
    if (bipartite) {
      if (first > subject_count || second <= subject_count ||
          second > total_count) {
        return false;
      }
    } else if (first >= second || second > subject_count) {
      return false;
    }
    previous = key;
  }
  return true;
}

void interpret_result (CudaSpatialAttempt &attempt, int status,
                       const klayout_cuda_spatial_result_v1 &result,
                       const klayout_cuda_spatial_config_v1 &config,
                       uint64_t subject_count, uint64_t intruder_count,
                       bool bipartite)
{
  copy_result_metadata (attempt, result);
  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size < sizeof (result)) {
    attempt.disposition = CudaSpatialAttempt::InvalidResult;
    attempt.message = "CUDA spatial backend returned an incompatible result";
    return;
  }

  if (status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.status == KLAYOUT_CUDA_SPATIAL_OK) {
    if (result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        ! valid_pair_keys (result, config, subject_count, intruder_count,
                           bipartite)) {
      attempt.disposition = CudaSpatialAttempt::InvalidResult;
      attempt.message = "CUDA spatial backend returned invalid pair keys";
      return;
    }
    if (result.pair_count) {
      attempt.pair_keys.assign (result.pair_keys,
                                result.pair_keys + result.pair_count);
    }
    attempt.disposition = CudaSpatialAttempt::Success;
    return;
  }

  if (result.pair_count != 0 || result.pair_keys) {
    attempt.disposition = CudaSpatialAttempt::InvalidResult;
    attempt.message = "CUDA spatial backend published pairs after failure";
  } else if (status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
             result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
  } else {
    attempt.disposition = CudaSpatialAttempt::BackendError;
  }
}

bool contact4_active_union_scene_echo_matches (
  const klayout_cuda_spatial_contact4_active_union_scene_v1 &scene,
  const klayout_cuda_spatial_contact4_active_union_scene_echo_v1 &echo)
{
  return
    echo.struct_size == sizeof (echo) &&
    echo.role == scene.role &&
    echo.format_version == scene.format_version &&
    echo.dbu_per_micron == scene.dbu_per_micron &&
    echo.root_cell == scene.root_cell &&
    echo.layer == scene.layer && echo.datatype == scene.datatype &&
    echo.reserved0 == 0 &&
    echo.context_count == scene.context_count &&
    echo.layer_context_count == scene.layer_context_count &&
    echo.context_polygon_offset_count ==
      scene.context_polygon_offset_count &&
    echo.context_edge_offset_count == scene.context_edge_offset_count &&
    echo.cell_count == scene.cell_count &&
    echo.polygon_count == scene.polygon_count &&
    echo.edge_count == scene.edge_count &&
    echo.flat_polygon_count == scene.flat_polygon_count &&
    echo.flat_edge_count == scene.flat_edge_count &&
    echo.scene_left == scene.scene_left &&
    echo.scene_bottom == scene.scene_bottom &&
    echo.scene_right == scene.scene_right &&
    echo.scene_top == scene.scene_top &&
    std::equal (
      echo.digest_domain,
      echo.digest_domain +
        KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES,
      scene.digest_domain) &&
    std::equal (
      echo.scene_digest, echo.scene_digest + 32, scene.scene_digest) &&
    echo.reserved1 [0] == 0 && echo.reserved1 [1] == 0;
}

} // anonymous namespace

CudaSpatialAttempt cuda_spatial_try_bipartite (
  const std::vector<klayout_cuda_spatial_aabb_v1> &subjects,
  const std::vector<klayout_cuda_spatial_aabb_v1> &intruders,
  int64_t enlargement)
{
  CudaSpatialAttempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.enabled ()) {
    return attempt;
  }
  if (! module.ready ()) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is missing required legacy ABI entry points"
      : module.error ();
    log_attempt (attempt, subjects.size (), intruders.size ());
    return attempt;
  }

  const uint64_t total_records = uint64_t (subjects.size ()) + uint64_t (intruders.size ());
  if (total_records < module.min_records ()) {
    attempt.disposition = CudaSpatialAttempt::BelowThreshold;
    log_attempt (attempt, subjects.size (), intruders.size ());
    return attempt;
  }
  if (subjects.empty () || intruders.empty () || enlargement < 0 ||
      subjects.size () > std::numeric_limits<uint32_t>::max () ||
      intruders.size () > std::numeric_limits<uint32_t>::max () ||
      total_records > std::numeric_limits<uint32_t>::max ()) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
    attempt.fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    log_attempt (attempt, subjects.size (), intruders.size ());
    return attempt;
  }

  klayout_cuda_spatial_config_v1 config = make_config ();

  klayout_cuda_spatial_request_v1 request;
  std::memset (&request, 0, sizeof (request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof (request);
  request.subjects = subjects.data ();
  request.subject_count = subjects.size ();
  request.intruders = intruders.data ();
  request.intruder_count = intruders.size ();
  request.enlargement = enlargement;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;

  try {
    const int status = module.run () (&request, &result);
    interpret_result (attempt, status, result, config, subjects.size (),
                      intruders.size (), true);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = ex.what ();
  } catch (...) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = "unknown exception while calling CUDA spatial backend";
  }

  module.release () (&result);
  log_attempt (attempt, subjects.size (), intruders.size ());
  return attempt;
}

CudaSpatialAttempt cuda_spatial_try_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement)
{
  CudaSpatialAttempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.enabled ()) {
    return attempt;
  }
  if (! module.ready ()) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is missing required legacy ABI entry points"
      : module.error ();
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }
  if (! module.self_ready ()) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
    attempt.fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message = "CUDA spatial backend has no self-AABB entry point";
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }

  if (records.size () < module.min_records ()) {
    attempt.disposition = CudaSpatialAttempt::BelowThreshold;
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }
  if (records.empty () || enlargement < 0 ||
      records.size () > std::numeric_limits<uint32_t>::max ()) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
    attempt.fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }

  klayout_cuda_spatial_config_v1 config = make_config ();

  klayout_cuda_spatial_request_v1 request;
  std::memset (&request, 0, sizeof (request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof (request);
  request.subjects = records.data ();
  request.subject_count = records.size ();
  request.enlargement = enlargement;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;

  try {
    const int status = module.run_self () (&request, &result);
    interpret_result (attempt, status, result, config, records.size (), 0,
                      false);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = ex.what ();
  } catch (...) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = "unknown exception while calling CUDA spatial backend";
  }

  module.release () (&result);
  log_attempt (attempt, records.size (), 0, "self");
  return attempt;
}

bool cuda_spatial_may_attempt (uint64_t subject_count, uint64_t intruder_count)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.enabled () || ! module.ready () || subject_count == 0 ||
      intruder_count == 0) {
    return false;
  }
  return subject_count <= std::numeric_limits<uint32_t>::max () &&
         intruder_count <= std::numeric_limits<uint32_t>::max () &&
         subject_count + intruder_count <= std::numeric_limits<uint32_t>::max () &&
         subject_count + intruder_count >= module.min_records ();
}

bool cuda_spatial_may_attempt_self (uint64_t record_count)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.ready () && module.self_ready () &&
         record_count != 0 &&
         record_count <= std::numeric_limits<uint32_t>::max () &&
         record_count >= module.min_records ();
}

bool cuda_spatial_preflight_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement, uint64_t &membership_count)
{
  membership_count = 0;
  if (! cuda_spatial_may_attempt_self (records.size ())) {
    return false;
  }
  return self_memberships_fit (records, enlargement, make_config (),
                               membership_count);
}

bool cuda_spatial_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.ready ();
}

bool cuda_spatial_validate_contact4_active_union_result (
  const klayout_cuda_spatial_contact4_active_union_request_v1 &request,
  const klayout_cuda_spatial_contact4_active_union_result_v1 &result,
  int backend_status, std::string *error)
{
  const auto fail = [error] (const char *message) {
    if (error) {
      try {
        *error = message;
      } catch (...) {
        //  Diagnostics cannot turn a fail-closed result into an exception.
      }
    }
    return false;
  };

  try {
    if (! qualified_contact4_active_union_request (request)) {
      return fail (
        "host supplied an unqualified CONTACT.4 ACTIVE-union request");
    }
    if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
        result.struct_size != sizeof (result) ||
        result.reserved0 != 0 ||
        result.reserved1 [0] != 0 || result.reserved1 [1] != 0) {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend returned an incompatible result");
    }
    if (backend_status != int (result.status)) {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend returned inconsistent statuses");
    }
    if (backend_status != KLAYOUT_CUDA_SPATIAL_OK) {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend did not return a proof");
    }
    if (result.opcode != request.opcode ||
        result.option_flags != request.option_flags ||
        result.format_version != request.format_version ||
        result.dbu_per_micron != request.dbu_per_micron ||
        result.device != request.device ||
        result.distance != request.distance ||
        result.grid_cell_size != request.grid_cell_size ||
        ! contact4_active_union_scene_echo_matches (
            request.active, result.active) ||
        ! contact4_active_union_scene_echo_matches (
            request.contact, result.contact) ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend returned a mismatched proof echo");
    }

    uint64_t expected_event_count = 0;
    uint64_t maximum_contact_memberships = 0;
    uint64_t maximum_boundary_cell_visits = 0;
    if (! checked_multiply_u64 (
          result.union_membership_count, UINT64_C (2),
          expected_event_count) ||
        ! checked_multiply_u64 (
          result.contact_expanded_edge_count,
          request.max_cells_per_contact_edge,
          maximum_contact_memberships) ||
        ! checked_multiply_u64 (
          result.boundary_segment_count,
          request.max_cells_per_boundary_edge,
          maximum_boundary_cell_visits) ||
        result.rectangle_count < request.active.flat_polygon_count ||
        result.rectangle_count > request.max_rectangles ||
        ! result.x_slab_count ||
        result.x_slab_count > request.max_x_slabs ||
        ! result.union_membership_count ||
        result.union_membership_count > request.max_union_memberships ||
        result.event_count != expected_event_count ||
        result.event_count > request.max_events ||
        ! result.strip_interval_count ||
        result.strip_interval_count > result.union_membership_count ||
        ! result.boundary_segment_count ||
        result.boundary_segment_count > request.max_boundary_segments ||
        result.raw_segment_count < result.boundary_segment_count ||
        result.raw_segment_count > request.max_raw_segments ||
        result.contact_expanded_edge_count !=
          request.contact.flat_edge_count ||
        result.contact_expanded_edge_count > request.max_contact_edges ||
        ! result.grid_cell_count ||
        result.grid_cell_count > request.max_grid_cells ||
        result.contact_membership_count <
          result.contact_expanded_edge_count ||
        result.contact_membership_count >
          request.max_contact_memberships ||
        result.contact_membership_count > maximum_contact_memberships ||
        result.boundary_cell_visit_count >
          request.max_boundary_cell_visits ||
        result.boundary_cell_visit_count >
          maximum_boundary_cell_visits ||
        result.member_visit_count > request.max_member_visits ||
        result.candidate_pair_count > result.member_visit_count ||
        result.candidate_pair_count > request.max_pair_work ||
        result.raw_hit_count > result.candidate_pair_count ||
        result.uncertain_count > result.candidate_pair_count ||
        result.raw_hit_count >
          result.candidate_pair_count - result.uncertain_count) {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend returned impossible proof "
        "counters");
    }

    if (! result.device_total_bytes ||
        result.union_free_begin_bytes > result.device_total_bytes ||
        result.union_free_low_bytes > result.union_free_begin_bytes ||
        result.callback_free_begin_bytes > result.device_total_bytes ||
        result.callback_free_low_bytes >
          result.callback_free_begin_bytes ||
        result.post_scan_free_bytes > result.device_total_bytes ||
        result.callback_incremental_peak_bytes !=
          result.callback_free_begin_bytes -
            result.callback_free_low_bytes) {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend returned impossible memory "
        "telemetry");
    }

    // boundary_ns encloses the resident callback, whose ACTIVE expansion and
    // query stages are also reported below.  Those useful sub-timings overlap,
    // so validate each against total_ns rather than summing them.
    const uint64_t component_times [] = {
      result.setup_ns,
      result.active_h2d_ns,
      result.active_expand_ns,
      result.x_membership_ns,
      result.strip_scan_ns,
      result.boundary_ns,
      result.contact_h2d_ns,
      result.contact_expand_ns,
      result.boundary_preflight_ns,
      result.grid_count_ns,
      result.grid_build_ns,
      result.query_ns,
      result.d2h_ns
    };
    if (! result.total_ns) {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend returned impossible timing "
        "telemetry");
    }
    for (size_t index = 0;
         index < sizeof (component_times) / sizeof (component_times [0]);
         ++index) {
      if (component_times [index] > result.total_ns) {
        return fail (
          "CUDA CONTACT.4 ACTIVE-union backend returned impossible timing "
          "telemetry");
      }
    }

    if (result.disposition ==
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE &&
        result.raw_hit_count == 0 && result.uncertain_count == 0) {
      //  sole consumable outcome
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_RAW_HITS &&
      result.raw_hit_count != 0 && result.uncertain_count == 0) {
      //  valid diagnostic result; the wrapper retains CPU fallback
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_UNCERTAIN &&
      result.uncertain_count != 0) {
      //  valid bounded decline; the wrapper retains CPU fallback
    } else {
      return fail (
        "CUDA CONTACT.4 ACTIVE-union backend returned an inconsistent "
        "disposition");
    }

    if (error) {
      error->clear ();
    }
    return true;
  } catch (...) {
    return fail (
      "exception while validating the CUDA CONTACT.4 ACTIVE-union proof");
  }
}

bool cuda_spatial_validate_active3_well_union_result (
  const klayout_cuda_spatial_active3_well_union_request_v1 &request,
  const klayout_cuda_spatial_active3_well_union_result_v1 &result,
  int backend_status, std::string *error)
{
  const auto fail = [error] (const char *message) {
    if (error) {
      try {
        *error = message;
      } catch (...) {
        // Diagnostics cannot turn a fail-closed result into an exception.
      }
    }
    return false;
  };

  try {
    if (! qualified_active3_well_union_request (request)) {
      return fail (
        "host supplied an unqualified ACTIVE.3 WELL-union request");
    }
    if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
        result.struct_size != sizeof (result) ||
        result.reserved0 != 0 || result.layer_reserved != 0 ||
        result.reserved1 [0] != 0 || result.reserved1 [1] != 0) {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend returned an incompatible result");
    }
    if (backend_status != int (result.status)) {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend returned inconsistent statuses");
    }
    if (backend_status != KLAYOUT_CUDA_SPATIAL_OK) {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend did not return a proof");
    }
    if (result.opcode != request.opcode ||
        result.option_flags != request.option_flags ||
        result.format_version != request.format_version ||
        result.dbu_per_micron != request.dbu_per_micron ||
        result.device != request.device ||
        result.distance != request.distance ||
        result.grid_cell_size != request.grid_cell_size ||
        result.secondary_well_layer != request.secondary_well_layer ||
        result.secondary_well_datatype !=
          request.secondary_well_datatype ||
        ! contact4_active_union_scene_echo_matches (
            request.wells, result.wells) ||
        ! contact4_active_union_scene_echo_matches (
            request.active, result.active) ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend returned a mismatched proof echo");
    }

    uint64_t expected_event_count = 0;
    uint64_t maximum_active_memberships = 0;
    uint64_t maximum_active_cell_visits = 0;
    uint64_t maximum_member_visits = 0;
    if (! checked_multiply_u64 (
          result.union_membership_count, UINT64_C (2),
          expected_event_count) ||
        ! checked_multiply_u64 (
          result.active_expanded_edge_count,
          request.max_cells_per_active_edge,
          maximum_active_memberships) ||
        ! checked_multiply_u64 (
          result.boundary_segment_count,
          request.max_cells_per_well_edge,
          maximum_active_cell_visits) ||
        ! checked_multiply_u64 (
          result.active_cell_visit_count,
          result.active_expanded_edge_count,
          maximum_member_visits) ||
        result.rectangle_count < request.wells.flat_polygon_count ||
        result.rectangle_count > request.max_rectangles ||
        ! result.x_slab_count ||
        result.x_slab_count > request.max_x_slabs ||
        ! result.union_membership_count ||
        result.union_membership_count > request.max_union_memberships ||
        result.event_count != expected_event_count ||
        result.event_count > request.max_events ||
        ! result.strip_interval_count ||
        result.strip_interval_count > result.union_membership_count ||
        ! result.boundary_segment_count ||
        result.boundary_segment_count > request.max_boundary_segments ||
        result.raw_segment_count < result.boundary_segment_count ||
        result.raw_segment_count > request.max_raw_segments ||
        result.active_expanded_edge_count !=
          request.active.flat_edge_count ||
        result.active_expanded_edge_count > request.max_active_edges ||
        ! result.grid_cell_count ||
        result.grid_cell_count > request.max_grid_cells ||
        result.active_membership_count <
          result.active_expanded_edge_count ||
        result.active_membership_count >
          request.max_active_memberships ||
        result.active_membership_count > maximum_active_memberships ||
        result.active_cell_visit_count >
          request.max_active_cell_visits ||
        result.active_cell_visit_count > maximum_active_cell_visits ||
        result.member_visit_count > request.max_member_visits ||
        result.member_visit_count > maximum_member_visits ||
        result.candidate_pair_count > result.member_visit_count ||
        result.candidate_pair_count > request.max_pair_work ||
        result.raw_hit_count > result.candidate_pair_count ||
        result.uncertain_count > result.candidate_pair_count ||
        result.raw_hit_count >
          result.candidate_pair_count - result.uncertain_count) {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend returned impossible proof counters");
    }

    if (! result.device_total_bytes ||
        result.union_free_begin_bytes > result.device_total_bytes ||
        result.union_free_low_bytes > result.union_free_begin_bytes ||
        result.callback_free_begin_bytes > result.device_total_bytes ||
        result.callback_free_low_bytes >
          result.callback_free_begin_bytes ||
        result.post_scan_free_bytes > result.device_total_bytes ||
        result.callback_incremental_peak_bytes !=
          result.callback_free_begin_bytes -
            result.callback_free_low_bytes) {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend returned impossible memory "
        "telemetry");
    }

    const uint64_t component_times [] = {
      result.setup_ns,
      result.wells_h2d_ns,
      result.wells_expand_ns,
      result.x_membership_ns,
      result.strip_scan_ns,
      result.boundary_ns,
      result.active_h2d_ns,
      result.active_expand_ns,
      result.active_preflight_ns,
      result.grid_count_ns,
      result.grid_build_ns,
      result.query_ns,
      result.d2h_ns
    };
    if (! result.total_ns) {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend returned impossible timing "
        "telemetry");
    }
    for (size_t index = 0;
         index < sizeof (component_times) / sizeof (component_times [0]);
         ++index) {
      if (component_times [index] > result.total_ns) {
        return fail (
          "CUDA ACTIVE.3 WELL-union backend returned impossible timing "
          "telemetry");
      }
    }

    if (result.disposition ==
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_COMPLETE &&
        result.raw_hit_count == 0 && result.uncertain_count == 0) {
      // sole consumable outcome
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_RAW_HITS &&
      result.raw_hit_count != 0 && result.uncertain_count == 0) {
      // valid diagnostic result; retain CPU fallback
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_UNCERTAIN &&
      result.uncertain_count != 0) {
      // valid bounded decline; retain CPU fallback
    } else {
      return fail (
        "CUDA ACTIVE.3 WELL-union backend returned an inconsistent "
        "disposition");
    }

    if (error) {
      error->clear ();
    }
    return true;
  } catch (...) {
    return fail (
      "exception while validating the CUDA ACTIVE.3 WELL-union proof");
  }
}

static CudaActive3Attempt cuda_spatial_try_active3_profile_empty (
  const klayout_cuda_spatial_active3_request_v1 &request, bool contact4,
  bool raw_contact4)
{
  CudaActive3Attempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  const bool raw_wells =
    ! contact4 &&
    request.opcode ==
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_BOTH_SUPERSET_EMPTY;
  const char *profile = raw_contact4
    ? "CONTACT.4 raw-ACTIVE"
    : (contact4
        ? "CONTACT.4"
        : (raw_wells ? "ACTIVE.3 raw-WELL" : "ACTIVE.3"));
  const bool profile_enabled = raw_contact4
    ? module.contact4_raw_active_enabled ()
    : (contact4
        ? module.contact4_enabled ()
        : (raw_wells
            ? module.active3_raw_wells_enabled ()
            : module.active3_enabled ()));
  if (! profile_enabled) {
    return attempt;
  }
  const bool qualified_contact4 = raw_contact4
    ? request.opcode ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_SUPERSET_EMPTY &&
      request.option_flags ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_QUALIFIED_OPTIONS &&
      request.distance == 10
    : request.opcode ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_SUPERSET_EMPTY &&
      request.option_flags ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_QUALIFIED_OPTIONS &&
      request.distance == 10;
  const bool qualified_profile =
    contact4
      ? qualified_contact4
      : (raw_wells
          ? request.option_flags ==
              KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_QUALIFIED_OPTIONS
          : request.opcode ==
              KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY &&
            request.option_flags ==
              KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_OPTIONS) &&
        request.distance == 110;
  if (! qualified_profile || request.dbu_per_micron != 2000 ||
      request.grid_cell_size != 2000 || request.reserved0 != 0 ||
      request.reserved1 != 0) {
    attempt.disposition = CudaActive3Attempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message = std::string ("unqualified CUDA ") + profile +
      " empty-certificate request";
    log_active3_profile_attempt (
      attempt, contact4, raw_contact4, raw_wells);
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaActive3Attempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_active3_profile_attempt (
      attempt, contact4, raw_contact4, raw_wells);
    return attempt;
  }
  const bool ready = raw_contact4
    ? module.contact4_raw_active_ready ()
    : (contact4
        ? module.contact4_ready ()
        : (raw_wells
            ? module.active3_raw_wells_ready ()
            : module.active3_ready ()));
  if (! ready) {
    attempt.disposition = CudaActive3Attempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message = std::string ("CUDA spatial backend has no ") + profile +
      " empty-certificate entry point";
    log_active3_profile_attempt (
      attempt, contact4, raw_contact4, raw_wells);
    return attempt;
  }

  klayout_cuda_spatial_active3_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition = KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN;

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status = raw_contact4
      ? module.run_contact4_raw_active () (&request, &result)
      : module.run_active3 () (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaActive3Attempt::BackendError;
    attempt.message = ex.what ();
    log_active3_profile_attempt (
      attempt, contact4, raw_contact4, raw_wells);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaActive3Attempt::BackendError;
    attempt.message = std::string ("unknown exception while calling CUDA ") +
      profile + " backend";
    log_active3_profile_attempt (
      attempt, contact4, raw_contact4, raw_wells);
    return attempt;
  }

  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.context_count = result.context_count;
  attempt.well_context_count = result.well_context_count;
  attempt.active_context_count = result.active_context_count;
  attempt.cell_count = result.cell_count;
  attempt.edge_count = result.edge_count;
  attempt.flat_well_edge_count = result.flat_well_edge_count;
  attempt.flat_active_edge_count = result.flat_active_edge_count;
  attempt.grid_cell_count = result.grid_cell_count;
  attempt.membership_count = result.membership_count;
  attempt.candidate_pair_count = result.candidate_pair_count;
  attempt.raw_hit_count = result.raw_hit_count;
  attempt.uncertain_count = result.uncertain_count;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (result.message, result.message + sizeof (result.message), '\0'));

  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size < sizeof (result) || result.reserved0 != 0) {
    attempt.disposition = CudaActive3Attempt::InvalidResult;
    attempt.message = std::string ("CUDA ") + profile +
      " backend returned an incompatible result";
    log_active3_profile_attempt (
      attempt, contact4, raw_contact4, raw_wells);
    return attempt;
  }

  if (status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.status == KLAYOUT_CUDA_SPATIAL_OK) {
    const bool echo_matches =
      result.opcode == request.opcode &&
      result.option_flags == request.option_flags &&
      result.dbu_per_micron == request.dbu_per_micron &&
      result.distance == request.distance &&
      result.grid_cell_size == request.grid_cell_size &&
      std::equal (result.scene_digest, result.scene_digest + 32,
                  request.scene_digest) &&
      result.context_count == request.context_count &&
      result.well_context_count == request.well_context_count &&
      result.active_context_count == request.active_context_count &&
      result.cell_count == request.cell_count &&
      result.edge_count == request.edge_count &&
      result.flat_well_edge_count == request.flat_well_edge_count &&
      result.flat_active_edge_count == request.flat_active_edge_count;
    if (! echo_matches ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      attempt.disposition = CudaActive3Attempt::InvalidResult;
      attempt.message = std::string ("CUDA ") + profile +
        " backend returned a mismatched proof echo";
    } else if (
      result.grid_cell_count > request.max_grid_cells ||
      result.membership_count > request.max_memberships ||
      result.candidate_pair_count > request.max_pair_work ||
      result.raw_hit_count > result.candidate_pair_count ||
      result.uncertain_count > result.candidate_pair_count ||
      result.raw_hit_count >
        result.candidate_pair_count - result.uncertain_count) {
      attempt.disposition = CudaActive3Attempt::InvalidResult;
      attempt.message = std::string ("CUDA ") + profile +
        " backend returned impossible proof counters";
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_COMPLETE &&
      result.raw_hit_count == 0 && result.uncertain_count == 0) {
      attempt.disposition = CudaActive3Attempt::CertifiedEmpty;
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_HITS &&
      result.raw_hit_count != 0 && result.uncertain_count == 0) {
      attempt.disposition = CudaActive3Attempt::RawHits;
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN &&
      result.uncertain_count != 0) {
      attempt.disposition = CudaActive3Attempt::BackendFallback;
    } else {
      attempt.disposition = CudaActive3Attempt::InvalidResult;
      attempt.message = std::string ("CUDA ") + profile +
        " backend returned an inconsistent disposition";
    }
  } else if (
    status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
    result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaActive3Attempt::BackendFallback;
  } else {
    attempt.disposition = CudaActive3Attempt::BackendError;
  }

  log_active3_profile_attempt (
    attempt, contact4, raw_contact4, raw_wells);
  return attempt;
}

CudaActive3Attempt cuda_spatial_try_active3_empty (
  const klayout_cuda_spatial_active3_request_v1 &request)
{
  return cuda_spatial_try_active3_profile_empty (request, false, false);
}

CudaActive3Attempt cuda_spatial_try_contact4_empty (
  const klayout_cuda_spatial_active3_request_v1 &request)
{
  return cuda_spatial_try_active3_profile_empty (request, true, false);
}

CudaActive3Attempt cuda_spatial_try_contact4_raw_active_empty (
  const klayout_cuda_spatial_active3_request_v1 &request)
{
  return cuda_spatial_try_active3_profile_empty (request, true, true);
}

CudaContact4ActiveUnionAttempt
cuda_spatial_try_contact4_active_union_empty (
  const klayout_cuda_spatial_contact4_active_union_request_v1 &request)
{
  CudaContact4ActiveUnionAttempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.contact4_active_union_enabled ()) {
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_contact4_active_union_attempt (attempt);
    return attempt;
  }
  if (! module.contact4_active_union_ready ()) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message =
      "CUDA spatial backend has no fused CONTACT.4 ACTIVE-union capability";
    log_contact4_active_union_attempt (attempt);
    return attempt;
  }
  if (! qualified_contact4_active_union_request (request)) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::InvalidResult;
    attempt.message =
      "host supplied an unqualified CONTACT.4 ACTIVE-union request";
    log_contact4_active_union_attempt (attempt);
    return attempt;
  }

  klayout_cuda_spatial_contact4_active_union_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition =
    KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_UNCERTAIN;

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status = module.run_contact4_active_union () (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::BackendError;
    attempt.message = ex.what ();
    log_contact4_active_union_attempt (attempt);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::BackendError;
    attempt.message =
      "unknown exception while calling CUDA CONTACT.4 ACTIVE-union backend";
    log_contact4_active_union_attempt (attempt);
    return attempt;
  }

  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.active_context_count = result.active.context_count;
  attempt.contact_context_count = result.contact.context_count;
  attempt.flat_active_polygon_count = result.active.flat_polygon_count;
  attempt.flat_active_edge_count = result.active.flat_edge_count;
  attempt.flat_contact_polygon_count = result.contact.flat_polygon_count;
  attempt.flat_contact_edge_count = result.contact.flat_edge_count;
  attempt.rectangle_count = result.rectangle_count;
  attempt.x_slab_count = result.x_slab_count;
  attempt.union_membership_count = result.union_membership_count;
  attempt.strip_interval_count = result.strip_interval_count;
  attempt.boundary_segment_count = result.boundary_segment_count;
  attempt.grid_cell_count = result.grid_cell_count;
  attempt.contact_membership_count = result.contact_membership_count;
  attempt.boundary_cell_visit_count = result.boundary_cell_visit_count;
  attempt.member_visit_count = result.member_visit_count;
  attempt.candidate_pair_count = result.candidate_pair_count;
  attempt.raw_hit_count = result.raw_hit_count;
  attempt.uncertain_count = result.uncertain_count;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (result.message, result.message + sizeof (result.message), '\0'));

  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size != sizeof (result) ||
      result.reserved0 != 0 ||
      result.reserved1 [0] != 0 || result.reserved1 [1] != 0) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::InvalidResult;
    attempt.message =
      "CUDA CONTACT.4 ACTIVE-union backend returned an incompatible result";
  } else if (status != int (result.status)) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::InvalidResult;
    attempt.message =
      "CUDA CONTACT.4 ACTIVE-union backend returned inconsistent statuses";
  } else if (status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::BackendFallback;
  } else if (status != KLAYOUT_CUDA_SPATIAL_OK) {
    attempt.disposition = CudaContact4ActiveUnionAttempt::BackendError;
  } else {
    std::string validation_error;
    if (! cuda_spatial_validate_contact4_active_union_result (
          request, result, status, &validation_error)) {
      attempt.disposition = CudaContact4ActiveUnionAttempt::InvalidResult;
      attempt.message = validation_error;
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE) {
      attempt.disposition = CudaContact4ActiveUnionAttempt::CertifiedEmpty;
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_RAW_HITS) {
      attempt.disposition = CudaContact4ActiveUnionAttempt::RawHits;
    } else {
      attempt.disposition = CudaContact4ActiveUnionAttempt::BackendFallback;
    }
  }

  log_contact4_active_union_attempt (attempt);
  return attempt;
}

CudaActive3WellUnionAttempt
cuda_spatial_try_active3_well_union_empty (
  const klayout_cuda_spatial_active3_well_union_request_v1 &request)
{
  CudaActive3WellUnionAttempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.active3_well_union_enabled ()) {
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaActive3WellUnionAttempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_active3_well_union_attempt (attempt);
    return attempt;
  }
  if (! module.active3_well_union_ready ()) {
    attempt.disposition = CudaActive3WellUnionAttempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message =
      "CUDA spatial backend has no exact resident ACTIVE.3 WELL-union "
      "capability";
    log_active3_well_union_attempt (attempt);
    return attempt;
  }
  if (! qualified_active3_well_union_request (request)) {
    attempt.disposition = CudaActive3WellUnionAttempt::InvalidResult;
    attempt.message =
      "host supplied an unqualified ACTIVE.3 WELL-union request";
    log_active3_well_union_attempt (attempt);
    return attempt;
  }

  klayout_cuda_spatial_active3_well_union_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition =
    KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_UNCERTAIN;

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status = module.run_active3_well_union () (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaActive3WellUnionAttempt::BackendError;
    attempt.message = ex.what ();
    log_active3_well_union_attempt (attempt);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaActive3WellUnionAttempt::BackendError;
    attempt.message =
      "unknown exception while calling CUDA ACTIVE.3 WELL-union backend";
    log_active3_well_union_attempt (attempt);
    return attempt;
  }

  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.well_context_count = result.wells.context_count;
  attempt.active_context_count = result.active.context_count;
  attempt.flat_well_polygon_count =
    result.wells.flat_polygon_count;
  attempt.flat_well_edge_count = result.wells.flat_edge_count;
  attempt.flat_active_polygon_count =
    result.active.flat_polygon_count;
  attempt.flat_active_edge_count = result.active.flat_edge_count;
  attempt.rectangle_count = result.rectangle_count;
  attempt.x_slab_count = result.x_slab_count;
  attempt.union_membership_count = result.union_membership_count;
  attempt.strip_interval_count = result.strip_interval_count;
  attempt.boundary_segment_count = result.boundary_segment_count;
  attempt.grid_cell_count = result.grid_cell_count;
  attempt.active_membership_count = result.active_membership_count;
  attempt.active_cell_visit_count = result.active_cell_visit_count;
  attempt.member_visit_count = result.member_visit_count;
  attempt.candidate_pair_count = result.candidate_pair_count;
  attempt.raw_hit_count = result.raw_hit_count;
  attempt.uncertain_count = result.uncertain_count;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (
      result.message, result.message + sizeof (result.message), '\0'));

  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size != sizeof (result) ||
      result.reserved0 != 0 || result.layer_reserved != 0 ||
      result.reserved1 [0] != 0 || result.reserved1 [1] != 0) {
    attempt.disposition = CudaActive3WellUnionAttempt::InvalidResult;
    attempt.message =
      "CUDA ACTIVE.3 WELL-union backend returned an incompatible result";
  } else if (status != int (result.status)) {
    attempt.disposition = CudaActive3WellUnionAttempt::InvalidResult;
    attempt.message =
      "CUDA ACTIVE.3 WELL-union backend returned inconsistent statuses";
  } else if (status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaActive3WellUnionAttempt::BackendFallback;
  } else if (status != KLAYOUT_CUDA_SPATIAL_OK) {
    attempt.disposition = CudaActive3WellUnionAttempt::BackendError;
  } else {
    std::string validation_error;
    if (! cuda_spatial_validate_active3_well_union_result (
          request, result, status, &validation_error)) {
      attempt.disposition = CudaActive3WellUnionAttempt::InvalidResult;
      attempt.message = validation_error;
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_COMPLETE) {
      attempt.disposition = CudaActive3WellUnionAttempt::CertifiedEmpty;
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_RAW_HITS) {
      attempt.disposition = CudaActive3WellUnionAttempt::RawHits;
    } else {
      attempt.disposition = CudaActive3WellUnionAttempt::BackendFallback;
    }
  }

  log_active3_well_union_attempt (attempt);
  return attempt;
}

bool cuda_spatial_active3_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.active3_ready ();
}

bool cuda_spatial_active3_raw_wells_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.active3_raw_wells_ready ();
}

bool cuda_spatial_active3_well_union_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.active3_well_union_ready ();
}

bool cuda_spatial_contact4_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.contact4_ready ();
}

bool cuda_spatial_contact4_raw_active_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.contact4_raw_active_ready ();
}

bool cuda_spatial_contact4_active_union_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.contact4_active_union_ready ();
}

CudaM1WidthSpaceAttempt cuda_spatial_try_m1_width_space_empty (
  const klayout_cuda_spatial_m1_width_space_request_v1 &request)
{
  CudaM1WidthSpaceAttempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  const bool metal2 =
    request.opcode == KLAYOUT_CUDA_SPATIAL_M2_WIDTH_SPACE_MERGED_EMPTY;
  const bool metal1 =
    request.opcode == KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_MERGED_EMPTY;
  if (! (metal1 || metal2) ||
      ! (metal2 ? module.m2_width_space_enabled ()
                : module.m1_width_space_enabled ())) {
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaM1WidthSpaceAttempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_metal_width_space_attempt (attempt, metal2);
    return attempt;
  }
  if (! (metal2 ? module.m2_width_space_ready ()
                : module.m1_width_space_ready ())) {
    attempt.disposition = CudaM1WidthSpaceAttempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message = metal2
      ? "CUDA spatial backend has no M2 width/space empty-certificate entry point"
      : "CUDA spatial backend has no M1 width/space empty-certificate entry point";
    log_metal_width_space_attempt (attempt, metal2);
    return attempt;
  }

  klayout_cuda_spatial_m1_width_space_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition = KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN;

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status =
      (metal2 ? module.run_m2_width_space ()
              : module.run_m1_width_space ()) (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaM1WidthSpaceAttempt::BackendError;
    attempt.message = ex.what ();
    log_metal_width_space_attempt (attempt, metal2);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaM1WidthSpaceAttempt::BackendError;
    attempt.message =
      "unknown exception while calling CUDA M1 width/space backend";
    log_metal_width_space_attempt (attempt, metal2);
    return attempt;
  }

  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.context_count = result.context_count;
  attempt.metal_context_count = result.metal_context_count;
  attempt.cell_count = result.cell_count;
  attempt.polygon_count = result.polygon_count;
  attempt.edge_count = result.edge_count;
  attempt.flat_polygon_count = result.flat_polygon_count;
  attempt.flat_edge_count = result.flat_edge_count;
  attempt.grid_cell_count = result.grid_cell_count;
  attempt.membership_count = result.membership_count;
  attempt.pair_work_count = result.pair_work_count;
  attempt.unique_edge_pair_count = result.unique_edge_pair_count;
  attempt.width_pair_count = result.width_pair_count;
  attempt.space_pair_count = result.space_pair_count;
  attempt.width_hit_count = result.width_hit_count;
  attempt.space_hit_count = result.space_hit_count;
  attempt.width_uncertain_count = result.width_uncertain_count;
  attempt.space_uncertain_count = result.space_uncertain_count;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (result.message, result.message + sizeof (result.message), '\0'));

  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size < sizeof (result) || result.reserved0 != 0) {
    attempt.disposition = CudaM1WidthSpaceAttempt::InvalidResult;
    attempt.message =
      "CUDA M1 width/space backend returned an incompatible result";
    log_metal_width_space_attempt (attempt, metal2);
    return attempt;
  }

  if (status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.status == KLAYOUT_CUDA_SPATIAL_OK) {
    const bool echo_matches =
      result.opcode == request.opcode &&
      result.option_flags == request.option_flags &&
      result.format_version == request.format_version &&
      result.dbu_per_micron == request.dbu_per_micron &&
      result.root_cell == request.root_cell &&
      result.width_distance == request.width_distance &&
      result.spacing_distance == request.spacing_distance &&
      result.grid_cell_size == request.grid_cell_size &&
      result.scene_left == request.scene_left &&
      result.scene_bottom == request.scene_bottom &&
      result.scene_right == request.scene_right &&
      result.scene_top == request.scene_top &&
      std::equal (
        result.scene_digest, result.scene_digest + 32,
        request.scene_digest) &&
      result.context_count == request.context_count &&
      result.metal_context_count == request.metal_context_count &&
      result.cell_count == request.cell_count &&
      result.polygon_count == request.polygon_count &&
      result.edge_count == request.edge_count &&
      result.flat_polygon_count == request.flat_polygon_count &&
      result.flat_edge_count == request.flat_edge_count;
    if (! echo_matches ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      attempt.disposition = CudaM1WidthSpaceAttempt::InvalidResult;
      attempt.message =
        "CUDA M1 width/space backend returned a mismatched proof echo";
    } else if (
      result.grid_cell_count > request.max_grid_cells ||
      result.membership_count > request.max_memberships ||
      result.pair_work_count > request.max_pair_work ||
      result.unique_edge_pair_count > result.pair_work_count ||
      result.width_pair_count > result.unique_edge_pair_count ||
      result.space_pair_count != result.unique_edge_pair_count ||
      result.width_hit_count > result.width_pair_count ||
      result.space_hit_count > result.space_pair_count ||
      result.width_uncertain_count > result.width_pair_count ||
      result.space_uncertain_count > result.space_pair_count ||
      result.width_hit_count >
        result.width_pair_count - result.width_uncertain_count ||
      result.space_hit_count >
        result.space_pair_count - result.space_uncertain_count) {
      attempt.disposition = CudaM1WidthSpaceAttempt::InvalidResult;
      attempt.message =
        "CUDA M1 width/space backend returned impossible proof counters";
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_COMPLETE &&
      result.width_hit_count == 0 && result.space_hit_count == 0 &&
      result.width_uncertain_count == 0 &&
      result.space_uncertain_count == 0) {
      attempt.disposition = CudaM1WidthSpaceAttempt::CertifiedEmpty;
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_RAW_HITS &&
      (result.width_hit_count != 0 || result.space_hit_count != 0) &&
      result.width_uncertain_count == 0 &&
      result.space_uncertain_count == 0) {
      attempt.disposition = CudaM1WidthSpaceAttempt::RawHits;
    } else if (
      result.disposition ==
        KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      (result.width_uncertain_count != 0 ||
       result.space_uncertain_count != 0)) {
      attempt.disposition = CudaM1WidthSpaceAttempt::BackendFallback;
    } else {
      attempt.disposition = CudaM1WidthSpaceAttempt::InvalidResult;
      attempt.message =
        "CUDA M1 width/space backend returned an inconsistent disposition";
    }
  } else if (
    status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
    result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaM1WidthSpaceAttempt::BackendFallback;
  } else {
    attempt.disposition = CudaM1WidthSpaceAttempt::BackendError;
  }

  log_metal_width_space_attempt (attempt, metal2);
  return attempt;
}

bool cuda_spatial_m1_width_space_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.m1_width_space_ready ();
}

bool cuda_spatial_m2_width_space_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.m2_width_space_ready ();
}

bool cuda_spatial_validate_m2_union_boundary (
  const klayout_cuda_spatial_m2_union_segment_v1 *segments,
  uint64_t segment_count, uint64_t expected_fnv64, std::string *error)
{
  const auto fail = [error] (const char *message) {
    if (error) {
      try {
        *error = message;
      } catch (...) {
        //  Diagnostics cannot turn a validation failure into an exception.
      }
    }
    return false;
  };

  try {
    if (segment_count && ! segments) {
      return fail ("nonempty M2 boundary has a null segment pointer");
    }
    for (uint64_t index = 0; index < segment_count; ++index) {
      const klayout_cuda_spatial_m2_union_segment_v1 &segment =
        segments [index];
      if ((segment.axis != KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL &&
           segment.axis != KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL) ||
          (segment.side != -1 && segment.side != 1) ||
          segment.lo >= segment.hi) {
        return fail ("M2 boundary has an invalid segment");
      }
      if (index) {
        const klayout_cuda_spatial_m2_union_segment_v1 &previous =
          segments [index - 1];
        if (! m2_union_segment_less (previous, segment)) {
          return fail (
            "M2 boundary is not strictly ordered by "
            "(axis,side,fixed,lo,hi)");
        }
        if (m2_union_same_line (previous, segment) &&
            segment.lo <= previous.hi) {
          return fail (
            "M2 boundary has nonmaximal touching or overlapping segments");
        }
      }
    }
    if (m2_union_boundary_fnv64 (segments, segment_count) !=
        expected_fnv64) {
      return fail ("M2 boundary FNV-1a digest mismatch");
    }
    if (error) {
      error->clear ();
    }
    return true;
  } catch (...) {
    return fail ("exception while validating the M2 boundary");
  }
}

static CudaM2UnionAttempt cuda_spatial_try_m2_union_impl (
  const klayout_cuda_spatial_m2_union_request_v1 &request,
  CudaM2UnionTiming *timing, uint32_t timing_struct_size,
  CudaM2SuffixCertificate *certificate,
  uint32_t certificate_struct_size)
{
  CudaM2UnionAttempt attempt;
  if (timing) {
    if (timing_struct_size != sizeof (CudaM2UnionTiming)) {
      attempt.disposition = CudaM2UnionAttempt::InvalidResult;
      attempt.message = "host supplied an incompatible M2 timing record";
      log_m2_union_attempt (attempt);
      return attempt;
    }
    std::memset (timing, 0, sizeof (*timing));
    timing->format_version = CudaM2UnionTiming::FormatVersion;
    timing->struct_size = sizeof (*timing);
  }
  if (certificate) {
    if (certificate_struct_size != sizeof (CudaM2SuffixCertificate)) {
      attempt.disposition = CudaM2UnionAttempt::InvalidResult;
      attempt.message =
        "host supplied an incompatible M2 suffix certificate record";
      log_m2_union_attempt (attempt);
      return attempt;
    }
    std::memset (certificate, 0, sizeof (*certificate));
    certificate->format_version =
      CudaM2SuffixCertificate::FormatVersion;
    certificate->struct_size = sizeof (*certificate);
    if (request.opcode !=
          KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY) {
      attempt.disposition = CudaM2UnionAttempt::InvalidResult;
      attempt.message =
        "M2 suffix certificate wrapper requires the suffix opcode";
      log_m2_union_attempt (attempt);
      return attempt;
    }
  }
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.m2_union_enabled ()) {
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaM2UnionAttempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_m2_union_attempt (attempt);
    return attempt;
  }
  if (! module.m2_union_ready ()) {
    attempt.disposition = CudaM2UnionAttempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message =
      "CUDA spatial backend has no complete M2 union-boundary capability";
    log_m2_union_attempt (attempt);
    return attempt;
  }
  if (! qualified_m2_union_request (request)) {
    attempt.disposition = CudaM2UnionAttempt::InvalidResult;
    attempt.message = "host supplied an unqualified M2 union request";
    log_m2_union_attempt (attempt);
    return attempt;
  }

  klayout_cuda_spatial_m2_union_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition = KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN;

  class ResultReleaseGuard
  {
  public:
    ResultReleaseGuard (
      klayout_cuda_spatial_release_m2_union_boundary_v1_func release,
      klayout_cuda_spatial_m2_union_result_v1 *result)
      : mp_release (release), mp_result (result)
    {
      //  nothing yet
    }

    ~ResultReleaseGuard ()
    {
      try {
        mp_release (mp_result);
      } catch (...) {
        //  A backend cleanup failure cannot make an untrusted result usable.
      }
    }

  private:
    klayout_cuda_spatial_release_m2_union_boundary_v1_func mp_release;
    klayout_cuda_spatial_m2_union_result_v1 *mp_result;
  } release_guard (module.release_m2_union (), &result);

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status = module.run_m2_union () (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaM2UnionAttempt::BackendError;
    attempt.message = ex.what ();
    log_m2_union_attempt (attempt);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaM2UnionAttempt::BackendError;
    attempt.message =
      "unknown exception while calling CUDA M2 union backend";
    log_m2_union_attempt (attempt);
    return attempt;
  }

  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.context_count = result.context_count;
  attempt.metal_context_count = result.metal_context_count;
  attempt.cell_count = result.cell_count;
  attempt.polygon_count = result.polygon_count;
  attempt.edge_count = result.edge_count;
  attempt.flat_polygon_count = result.flat_polygon_count;
  attempt.flat_edge_count = result.flat_edge_count;
  attempt.rectangle_count = result.rectangle_count;
  attempt.x_slab_count = result.x_slab_count;
  attempt.membership_count = result.membership_count;
  attempt.event_count = result.event_count;
  attempt.strip_interval_count = result.strip_interval_count;
  attempt.raw_segment_count = result.raw_segment_count;
  attempt.boundary_fnv64 = result.boundary_fnv64;
  attempt.total_ns = result.total_ns;
  if (timing) {
    timing->setup_ns = result.setup_ns;
    timing->h2d_ns = result.h2d_ns;
    timing->rectangle_expand_ns = result.rectangle_expand_ns;
    timing->x_membership_ns = result.x_membership_ns;
    timing->strip_scan_ns = result.strip_scan_ns;
    timing->boundary_ns = result.boundary_ns;
    timing->d2h_ns = result.d2h_ns;
    timing->total_ns = result.total_ns;
  }
  attempt.message.assign (
    result.message,
    std::find (result.message, result.message + sizeof (result.message), '\0'));

  const bool backend_ok =
    status == KLAYOUT_CUDA_SPATIAL_OK &&
    result.status == KLAYOUT_CUDA_SPATIAL_OK;
  const bool suffix_fields_match =
    result.certificate_reserved == 0 &&
    (request.opcode ==
       KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY
       ? result.certified_empty_mask == 0 &&
         result.suffix_total_ns == 0
       : backend_ok
           ? result.certified_empty_mask ==
               KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY &&
             result.suffix_total_ns <= result.total_ns
           : result.certified_empty_mask == 0 &&
             result.suffix_total_ns == 0);
  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size != sizeof (result) ||
      result.segment_record_bytes !=
        sizeof (klayout_cuda_spatial_m2_union_segment_v1) ||
      ! suffix_fields_match) {
    attempt.disposition = CudaM2UnionAttempt::InvalidResult;
    attempt.message =
      "CUDA M2 union backend returned an incompatible result";
    log_m2_union_attempt (attempt);
    return attempt;
  }

  if (backend_ok) {
    const bool echo_matches =
      result.opcode == request.opcode &&
      result.option_flags == request.option_flags &&
      result.format_version == request.format_version &&
      result.dbu_per_micron == request.dbu_per_micron &&
      result.root_cell == request.root_cell &&
      std::equal (
        result.scene_digest, result.scene_digest + 32,
        request.scene_digest) &&
      result.context_count == request.context_count &&
      result.metal_context_count == request.metal_context_count &&
      result.cell_count == request.cell_count &&
      result.polygon_count == request.polygon_count &&
      result.edge_count == request.edge_count &&
      result.flat_polygon_count == request.flat_polygon_count &&
      result.flat_edge_count == request.flat_edge_count;
    if (! echo_matches ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      attempt.disposition = CudaM2UnionAttempt::InvalidResult;
      attempt.message =
        "CUDA M2 union backend returned a mismatched proof echo";
    } else if (
      result.disposition != KLAYOUT_CUDA_SPATIAL_M2_UNION_COMPLETE ||
      ! result.segment_count || ! result.segments) {
      attempt.disposition = CudaM2UnionAttempt::InvalidResult;
      attempt.message =
        "CUDA M2 union backend returned an incomplete empty boundary";
    } else {
      uint64_t expected_event_count = 0;
      const bool counters_match =
        checked_multiply_u64 (
          result.membership_count, UINT64_C (2), expected_event_count) &&
        result.rectangle_count >= result.flat_polygon_count &&
        result.rectangle_count <= request.max_rectangles &&
        result.x_slab_count && result.x_slab_count <= request.max_x_slabs &&
        result.membership_count &&
        result.membership_count <= request.max_memberships &&
        result.event_count == expected_event_count &&
        result.event_count <= request.max_events &&
        result.segment_count <= request.max_segments &&
        result.strip_interval_count &&
        result.strip_interval_count <= result.membership_count &&
        result.raw_segment_count >= result.segment_count &&
        result.raw_segment_count <= request.max_raw_segments;
      if (! counters_match) {
        attempt.disposition = CudaM2UnionAttempt::InvalidResult;
        attempt.message =
          "CUDA M2 union backend returned impossible proof counters";
      } else {
        try {
          attempt.segments.resize (size_t (result.segment_count));
          std::copy_n (
            result.segments, size_t (result.segment_count),
            attempt.segments.begin ());
          std::string boundary_error;
          if (! cuda_spatial_validate_m2_union_boundary (
                attempt.segments.data (), attempt.segments.size (),
                result.boundary_fnv64, &boundary_error)) {
            attempt.segments.clear ();
            attempt.disposition = CudaM2UnionAttempt::InvalidResult;
            attempt.message = boundary_error;
          } else {
            attempt.disposition = CudaM2UnionAttempt::Complete;
            if (certificate) {
              certificate->certified_empty_mask =
                result.certified_empty_mask;
              certificate->total_ns = result.suffix_total_ns;
            }
          }
        } catch (const std::exception &ex) {
          attempt.segments.clear ();
          attempt.disposition = CudaM2UnionAttempt::BackendError;
          attempt.message = ex.what ();
        } catch (...) {
          attempt.segments.clear ();
          attempt.disposition = CudaM2UnionAttempt::BackendError;
          attempt.message =
            "exception while copying the CUDA M2 union boundary";
        }
      }
    }
  } else if (
    status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
    result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaM2UnionAttempt::BackendFallback;
  } else {
    attempt.disposition = CudaM2UnionAttempt::BackendError;
  }

  log_m2_union_attempt (attempt);
  return attempt;
}

CudaM2UnionAttempt cuda_spatial_try_m2_union (
  const klayout_cuda_spatial_m2_union_request_v1 &request)
{
  return cuda_spatial_try_m2_union_impl (request, 0, 0, 0, 0);
}

CudaM2UnionAttempt cuda_spatial_try_m2_union_with_timing (
  const klayout_cuda_spatial_m2_union_request_v1 &request,
  CudaM2UnionTiming *timing, uint32_t timing_struct_size)
{
  return cuda_spatial_try_m2_union_impl (
    request, timing, timing_struct_size, 0, 0);
}

CudaM2UnionAttempt cuda_spatial_try_m2_union_with_certificate (
  const klayout_cuda_spatial_m2_union_request_v1 &request,
  CudaM2SuffixCertificate *certificate,
  uint32_t certificate_struct_size)
{
  return cuda_spatial_try_m2_union_impl (
    request, 0, 0, certificate, certificate_struct_size);
}

bool cuda_spatial_m2_union_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.m2_union_ready ();
}

CudaPoly34Attempt cuda_spatial_try_poly34_empty (
  const klayout_cuda_spatial_poly34_request_v1 &request)
{
  CudaPoly34Attempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.poly34_enabled ()) {
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaPoly34Attempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_poly34_attempt (attempt);
    return attempt;
  }
  if (! module.poly34_ready ()) {
    attempt.disposition = CudaPoly34Attempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message =
      "CUDA spatial backend has no POLY.3/.4 empty-certificate entry point";
    log_poly34_attempt (attempt);
    return attempt;
  }
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof (request) ||
      request.opcode != KLAYOUT_CUDA_SPATIAL_POLY34_TERMINAL_EMPTY ||
      request.option_flags !=
        KLAYOUT_CUDA_SPATIAL_POLY34_QUALIFIED_OPTIONS ||
      request.format_version != 1 ||
      request.dbu_per_micron != 2000 ||
      request.requested_mask != KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES ||
      request.device < 0 || request.reserved0 != 0 ||
      request.identity_reserved != 0 ||
      request.context_reserved != 0 || request.cell_reserved != 0 ||
      request.box_reserved != 0 || request.capacity_reserved != 0 ||
      request.reserved1 [0] != 0 || request.reserved1 [1] != 0 ||
      request.poly3_distance != 110 || request.poly4_distance != 140 ||
      request.grid_cell_size <= 0 ||
      request.max_candidates_per_gate != 64 ||
      ! request.context_count || ! request.contexts ||
      ! request.poly_context_count || ! request.poly_contexts ||
      request.poly_offset_count != request.poly_context_count ||
      ! request.poly_offsets ||
      ! request.active_context_count || ! request.active_contexts ||
      request.active_offset_count != request.active_context_count ||
      ! request.active_offsets ||
      ! request.gate_context_count || ! request.gate_contexts ||
      request.gate_offset_count != request.gate_context_count ||
      ! request.gate_offsets ||
      ! request.cell_count || ! request.cells ||
      ! request.box_count || ! request.boxes ||
      request.context_record_bytes !=
        sizeof (klayout_cuda_spatial_poly34_context_v1) ||
      request.cell_record_bytes !=
        sizeof (klayout_cuda_spatial_poly34_cell_v1) ||
      request.box_record_bytes !=
        sizeof (klayout_cuda_spatial_poly34_box_v1) ||
      ! request.flat_poly_box_count ||
      ! request.flat_active_box_count ||
      ! request.flat_gate_box_count ||
      request.scene_left >= request.scene_right ||
      request.scene_bottom >= request.scene_top ||
      ! request.max_contexts || ! request.max_flat_boxes ||
      ! request.max_grid_cells || ! request.max_poly_memberships ||
      ! request.max_active_memberships || ! request.max_query_visits ||
      ! request.max_candidate_work ||
      request.context_count > request.max_contexts ||
      request.context_count > std::numeric_limits<uint32_t>::max () ||
      request.cell_count > request.context_count ||
      request.cell_count > std::numeric_limits<uint32_t>::max () ||
      request.box_count > request.max_flat_boxes ||
      request.flat_poly_box_count >
        std::numeric_limits<uint32_t>::max () ||
      request.flat_active_box_count >
        std::numeric_limits<uint32_t>::max () ||
      request.flat_gate_box_count >
        std::numeric_limits<uint32_t>::max ()) {
    attempt.disposition = CudaPoly34Attempt::InvalidResult;
    attempt.message = "CUDA POLY34 caller supplied an unqualified request";
    log_poly34_attempt (attempt);
    return attempt;
  }

  klayout_cuda_spatial_poly34_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition = KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN;

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status = module.run_poly34 () (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaPoly34Attempt::BackendError;
    attempt.message = ex.what ();
    log_poly34_attempt (attempt);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaPoly34Attempt::BackendError;
    attempt.message =
      "unknown exception while calling CUDA POLY34 backend";
    log_poly34_attempt (attempt);
    return attempt;
  }

  attempt.certified_empty_mask = result.certified_empty_mask;
  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.context_count = result.context_count;
  attempt.poly_context_count = result.poly_context_count;
  attempt.active_context_count = result.active_context_count;
  attempt.gate_context_count = result.gate_context_count;
  attempt.cell_count = result.cell_count;
  attempt.box_count = result.box_count;
  attempt.flat_poly_box_count = result.flat_poly_box_count;
  attempt.flat_active_box_count = result.flat_active_box_count;
  attempt.flat_gate_box_count = result.flat_gate_box_count;
  attempt.poly_membership_count = result.poly_membership_count;
  attempt.active_membership_count = result.active_membership_count;
  attempt.poly_query_visit_count = result.poly_query_visit_count;
  attempt.active_query_visit_count = result.active_query_visit_count;
  attempt.poly_candidate_count = result.poly_candidate_count;
  attempt.active_candidate_count = result.active_candidate_count;
  attempt.poly_terminal_empty_count = result.poly_terminal_empty_count;
  attempt.active_terminal_empty_count = result.active_terminal_empty_count;
  attempt.atomic_terminal_empty_count =
    result.atomic_terminal_empty_count;
  attempt.fallback_gate_count = result.fallback_gate_count;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (
      result.message, result.message + sizeof (result.message), '\0'));

  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size < sizeof (result) ||
      result.reserved0 != 0 || result.reserved1 != 0 ||
      result.identity_reserved != 0) {
    attempt.disposition = CudaPoly34Attempt::InvalidResult;
    attempt.message =
      "CUDA POLY34 backend returned an incompatible result";
    log_poly34_attempt (attempt);
    return attempt;
  }

  if (status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.status == KLAYOUT_CUDA_SPATIAL_OK) {
    const bool echo_matches =
      result.opcode == request.opcode &&
      result.option_flags == request.option_flags &&
      result.format_version == request.format_version &&
      result.requested_mask == request.requested_mask &&
      result.dbu_per_micron == request.dbu_per_micron &&
      result.root_cell == request.root_cell &&
      result.device == request.device &&
      result.poly3_distance == request.poly3_distance &&
      result.poly4_distance == request.poly4_distance &&
      result.grid_cell_size == request.grid_cell_size &&
      result.store_identity == request.store_identity &&
      result.layout_identity == request.layout_identity &&
      result.top_cell_identity == request.top_cell_identity &&
      result.poly_layer_id == request.poly_layer_id &&
      result.active_layer_id == request.active_layer_id &&
      result.gate_layer_id == request.gate_layer_id &&
      std::equal (
        result.scene_digest, result.scene_digest + 32,
        request.scene_digest) &&
      result.context_count == request.context_count &&
      result.poly_context_count == request.poly_context_count &&
      result.active_context_count == request.active_context_count &&
      result.gate_context_count == request.gate_context_count &&
      result.cell_count == request.cell_count &&
      result.box_count == request.box_count &&
      result.flat_poly_box_count == request.flat_poly_box_count &&
      result.flat_active_box_count == request.flat_active_box_count &&
      result.flat_gate_box_count == request.flat_gate_box_count &&
      result.expanded_poly_box_count == request.flat_poly_box_count &&
      result.expanded_active_box_count == request.flat_active_box_count &&
      result.expanded_gate_box_count == request.flat_gate_box_count;
    const bool counters_possible =
      result.grid_cell_count <= request.max_grid_cells &&
      result.poly_membership_count <= request.max_poly_memberships &&
      result.active_membership_count <= request.max_active_memberships &&
      result.poly_query_visit_count <= request.max_query_visits &&
      result.active_query_visit_count <= request.max_query_visits &&
      result.poly_candidate_count <= request.max_candidate_work &&
      result.active_candidate_count <= request.max_candidate_work &&
      result.poly_terminal_empty_count <= request.flat_gate_box_count &&
      result.active_terminal_empty_count <= request.flat_gate_box_count &&
      result.atomic_terminal_empty_count <=
        result.poly_terminal_empty_count &&
      result.atomic_terminal_empty_count <=
        result.active_terminal_empty_count &&
      result.fallback_gate_count <= request.flat_gate_box_count &&
      result.atomic_terminal_empty_count + result.fallback_gate_count ==
        request.flat_gate_box_count &&
      result.maximum_poly_candidates <=
        request.max_candidates_per_gate &&
      result.maximum_active_candidates <=
        request.max_candidates_per_gate;
    if (! echo_matches || ! counters_possible ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      attempt.disposition = CudaPoly34Attempt::InvalidResult;
      attempt.message =
        "CUDA POLY34 backend returned a mismatched or impossible proof";
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
      result.certified_empty_mask ==
        KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES &&
      result.poly_terminal_empty_count == request.flat_gate_box_count &&
      result.active_terminal_empty_count == request.flat_gate_box_count &&
      result.atomic_terminal_empty_count == request.flat_gate_box_count &&
      result.fallback_gate_count == 0) {
      attempt.disposition = CudaPoly34Attempt::CertifiedEmpty;
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY &&
      result.certified_empty_mask !=
        KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES &&
      result.fallback_gate_count != 0) {
      attempt.disposition = CudaPoly34Attempt::NotEmpty;
    } else {
      attempt.disposition = CudaPoly34Attempt::InvalidResult;
      attempt.message =
        "CUDA POLY34 backend returned an inconsistent disposition";
    }
  } else if (
    status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
    result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaPoly34Attempt::BackendFallback;
  } else {
    attempt.disposition = CudaPoly34Attempt::BackendError;
  }

  log_poly34_attempt (attempt);
  return attempt;
}

bool cuda_spatial_poly34_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.poly34_enabled () && module.enabled () &&
         module.poly34_ready ();
}

CudaVia1StackAttempt cuda_spatial_try_via1_stack_empty (
  const klayout_cuda_spatial_via1_stack_request_v1 &request)
{
  CudaVia1StackAttempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.via1_stack_enabled () && ! module.m1_contact_enabled ()) {
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaVia1StackAttempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_via1_stack_attempt (attempt);
    return attempt;
  }
  if (! module.via1_stack_ready () && ! module.m1_contact_ready ()) {
    attempt.disposition = CudaVia1StackAttempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message =
      "CUDA spatial backend has no VIA1-stack empty-certificate entry point";
    log_via1_stack_attempt (attempt);
    return attempt;
  }
  if (! qualified_via1_stack_request (request)) {
    attempt.disposition = CudaVia1StackAttempt::InvalidResult;
    attempt.message =
      "CUDA VIA1-stack caller supplied an unqualified request";
    log_via1_stack_attempt (attempt);
    return attempt;
  }

  klayout_cuda_spatial_via1_stack_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_UNCERTAIN;

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status = module.run_via1_stack () (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaVia1StackAttempt::BackendError;
    attempt.message = ex.what ();
    log_via1_stack_attempt (attempt);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaVia1StackAttempt::BackendError;
    attempt.message =
      "unknown exception while calling CUDA VIA1-stack backend";
    log_via1_stack_attempt (attempt);
    return attempt;
  }

  attempt.certified_empty_mask = result.certified_empty_mask;
  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.context_count = result.context_count;
  attempt.flat_metal1_box_count = result.flat_metal1_box_count;
  attempt.flat_via1_box_count = result.flat_via1_box_count;
  attempt.flat_metal2_box_count = result.flat_metal2_box_count;
  attempt.via_expanded_count = result.via_expanded_count;
  attempt.via_size_checked_count = result.via_size_checked_count;
  attempt.via_size_violation_count = result.via_size_violation_count;
  attempt.metal1_expanded_count = result.metal1_expanded_count;
  attempt.metal2_expanded_count = result.metal2_expanded_count;
  attempt.grid_cell_count = result.grid_cell_count;
  attempt.via_membership_count = result.via_membership_count;
  attempt.metal1_membership_count = result.metal1_membership_count;
  attempt.metal2_membership_count = result.metal2_membership_count;
  attempt.via_pair_queried_count = result.via_pair_queried_count;
  attempt.via_candidate_pair_count = result.via_candidate_pair_count;
  attempt.duplicate_via_pair_count = result.duplicate_via_pair_count;
  attempt.unsafe_via_pair_count = result.unsafe_via_pair_count;
  attempt.spacing_violation_count = result.spacing_violation_count;
  attempt.clean_via_pair_count = result.clean_via_pair_count;
  attempt.metal1_queried_count = result.metal1_queried_count;
  attempt.metal1_candidate_count = result.metal1_candidate_count;
  attempt.metal1_certified_count = result.metal1_certified_count;
  attempt.metal1_miss_count = result.metal1_miss_count;
  attempt.metal2_queried_count = result.metal2_queried_count;
  attempt.metal2_candidate_count = result.metal2_candidate_count;
  attempt.metal2_certified_count = result.metal2_certified_count;
  attempt.metal2_miss_count = result.metal2_miss_count;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (result.message, result.message + sizeof (result.message), '\0'));

  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size < sizeof (result) || result.reserved0 != 0) {
    attempt.disposition = CudaVia1StackAttempt::InvalidResult;
    attempt.message =
      "CUDA VIA1-stack backend returned an incompatible result";
    log_via1_stack_attempt (attempt);
    return attempt;
  }

  const bool echo_matches =
    result.opcode == request.opcode &&
    result.option_flags == request.option_flags &&
    result.requested_mask == request.requested_mask &&
    result.dbu_per_micron == request.dbu_per_micron &&
    result.enclosure_distance == request.enclosure_distance &&
    result.cut_width == request.cut_width &&
    result.cut_height == request.cut_height &&
    result.spacing_distance == request.spacing_distance &&
    result.grid_cell_size == request.grid_cell_size &&
    std::equal (
      result.scene_digest, result.scene_digest + 32, request.scene_digest) &&
    result.context_count == request.context_count &&
    result.metal1_context_count == request.metal1_context_count &&
    result.via1_context_count == request.via1_context_count &&
    result.metal2_context_count == request.metal2_context_count &&
    result.cell_count == request.cell_count &&
    result.box_count == request.box_count &&
    result.flat_metal1_box_count == request.flat_metal1_box_count &&
    result.flat_via1_box_count == request.flat_via1_box_count &&
    result.flat_metal2_box_count == request.flat_metal2_box_count;
  const bool counters_fit =
    result.via_expanded_count == request.flat_via1_box_count &&
    result.via_size_checked_count == request.flat_via1_box_count &&
    result.via_size_violation_count <= result.via_size_checked_count &&
    result.metal1_expanded_count == request.flat_metal1_box_count &&
    result.metal2_expanded_count == request.flat_metal2_box_count &&
    result.grid_cell_count != 0 &&
    result.grid_cell_count <= request.max_grid_cells &&
    result.via_membership_count >= result.via_expanded_count &&
    result.via_membership_count <= request.max_via_memberships &&
    result.metal1_membership_count >= result.metal1_expanded_count &&
    result.metal1_membership_count <= request.max_metal_memberships &&
    result.metal2_membership_count >= result.metal2_expanded_count &&
    result.metal2_membership_count <= request.max_metal_memberships &&
    result.via_pair_queried_count == request.flat_via1_box_count &&
    result.via_candidate_pair_count <= request.max_pair_work &&
    result.duplicate_via_pair_count <= result.via_candidate_pair_count &&
    result.unsafe_via_pair_count <=
      result.via_candidate_pair_count - result.duplicate_via_pair_count &&
    result.spacing_violation_count <=
      result.via_candidate_pair_count -
        result.duplicate_via_pair_count - result.unsafe_via_pair_count &&
    result.clean_via_pair_count ==
      result.via_candidate_pair_count -
        result.duplicate_via_pair_count - result.unsafe_via_pair_count -
        result.spacing_violation_count &&
    result.metal1_queried_count == request.flat_via1_box_count &&
    result.metal1_certified_count <= result.metal1_queried_count &&
    result.metal1_miss_count ==
      result.metal1_queried_count - result.metal1_certified_count &&
    result.metal1_candidate_count >= result.metal1_certified_count &&
    result.metal1_candidate_count <= request.max_pair_work &&
    result.metal2_queried_count == request.flat_via1_box_count &&
    result.metal2_certified_count <= result.metal2_queried_count &&
    result.metal2_miss_count ==
      result.metal2_queried_count - result.metal2_certified_count &&
    result.metal2_candidate_count >= result.metal2_certified_count &&
    result.metal2_candidate_count <= request.max_pair_work &&
    (result.certified_empty_mask & ~request.requested_mask) == 0;

  uint64_t maximum_via_pairs = 0;
  if (request.flat_via1_box_count > 1) {
    const uint64_t even =
      request.flat_via1_box_count % 2
        ? request.flat_via1_box_count - 1
        : request.flat_via1_box_count;
    const uint64_t odd =
      request.flat_via1_box_count % 2
        ? request.flat_via1_box_count
        : request.flat_via1_box_count - 1;
    maximum_via_pairs = (even / 2) * odd;
  }
  uint64_t maximum_metal1_pairs = 0;
  uint64_t maximum_metal2_pairs = 0;
  const bool candidate_counts_fit =
    checked_multiply_u64 (
      request.flat_via1_box_count,
      request.flat_metal1_box_count, maximum_metal1_pairs) &&
    checked_multiply_u64 (
      request.flat_via1_box_count,
      request.flat_metal2_box_count, maximum_metal2_pairs) &&
    result.via_candidate_pair_count <= maximum_via_pairs &&
    result.metal1_candidate_count <= maximum_metal1_pairs &&
    result.metal2_candidate_count <= maximum_metal2_pairs;

  if (status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.status == KLAYOUT_CUDA_SPATIAL_OK) {
    if (! echo_matches || ! counters_fit || ! candidate_counts_fit ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      attempt.disposition = CudaVia1StackAttempt::InvalidResult;
      attempt.message =
        "CUDA VIA1-stack backend returned a mismatched proof echo";
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE &&
      result.certified_empty_mask == request.requested_mask &&
      result.via_size_violation_count == 0 &&
      result.unsafe_via_pair_count == 0 &&
      result.spacing_violation_count == 0 &&
      result.metal1_miss_count == 0 && result.metal2_miss_count == 0) {
      attempt.disposition = CudaVia1StackAttempt::CertifiedEmpty;
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY &&
      result.certified_empty_mask != request.requested_mask) {
      attempt.disposition = CudaVia1StackAttempt::NotEmpty;
    } else {
      attempt.disposition = CudaVia1StackAttempt::InvalidResult;
      attempt.message =
        "CUDA VIA1-stack backend returned an inconsistent disposition";
    }
  } else if (
    status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
    result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaVia1StackAttempt::BackendFallback;
  } else {
    attempt.disposition = CudaVia1StackAttempt::BackendError;
  }

  log_via1_stack_attempt (attempt);
  return attempt;
}

bool cuda_spatial_via1_stack_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.via1_stack_ready ();
}

bool cuda_spatial_m1_contact_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.m1_contact_ready ();
}

CudaImplant12Attempt cuda_spatial_try_implant12_empty (
  const klayout_cuda_spatial_implant12_request_v1 &request)
{
  CudaImplant12Attempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.implant12_enabled ()) {
    return attempt;
  }
  if (! module.enabled ()) {
    attempt.disposition = CudaImplant12Attempt::BackendError;
    attempt.message = module.error ().empty ()
      ? "CUDA spatial backend is unavailable"
      : module.error ();
    log_implant12_attempt (attempt);
    return attempt;
  }
  if (! module.implant12_ready ()) {
    attempt.disposition = CudaImplant12Attempt::BackendFallback;
    attempt.fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message =
      "CUDA spatial backend has no IMPLANT.1/.2 empty-certificate entry point";
    log_implant12_attempt (attempt);
    return attempt;
  }
  if (! qualified_implant12_request (request)) {
    attempt.disposition = CudaImplant12Attempt::InvalidResult;
    attempt.message =
      "CUDA IMPLANT.1/.2 caller supplied an unqualified request";
    log_implant12_attempt (attempt);
    return attempt;
  }

  klayout_cuda_spatial_implant12_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition = KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN;

  int status = KLAYOUT_CUDA_SPATIAL_ERROR;
  try {
    status = module.run_implant12 () (&request, &result);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaImplant12Attempt::BackendError;
    attempt.message = ex.what ();
    log_implant12_attempt (attempt);
    return attempt;
  } catch (...) {
    attempt.disposition = CudaImplant12Attempt::BackendError;
    attempt.message =
      "unknown exception while calling CUDA IMPLANT.1/.2 backend";
    log_implant12_attempt (attempt);
    return attempt;
  }

  attempt.certified_empty_mask = result.certified_empty_mask;
  attempt.clean_mask = result.clean_mask;
  attempt.fallback_flags = result.fallback_flags;
  attempt.device_flags = result.device_flags;
  attempt.context_count = result.context_count;
  attempt.implant_context_count = result.implant_context_count;
  attempt.gate_context_count = result.gate_context_count;
  attempt.contact_context_count = result.contact_context_count;
  attempt.cell_count = result.cell_count;
  attempt.contour_count = result.contour_count;
  attempt.edge_count = result.edge_count;
  attempt.flat_implant_polygon_count =
    result.flat_implant_polygon_count;
  attempt.flat_gate_polygon_count = result.flat_gate_polygon_count;
  attempt.flat_contact_polygon_count =
    result.flat_contact_polygon_count;
  attempt.flat_implant_contour_count =
    result.flat_implant_contour_count;
  attempt.flat_gate_contour_count = result.flat_gate_contour_count;
  attempt.flat_contact_contour_count =
    result.flat_contact_contour_count;
  attempt.flat_implant_edge_count = result.flat_implant_edge_count;
  attempt.flat_gate_edge_count = result.flat_gate_edge_count;
  attempt.flat_contact_edge_count = result.flat_contact_edge_count;
  attempt.implant_expanded_edge_count =
    result.implant_expanded_edge_count;
  attempt.gate_processed_edge_count = result.gate_processed_edge_count;
  attempt.contact_processed_edge_count =
    result.contact_processed_edge_count;
  attempt.grid_cell_count = result.grid_cell_count;
  attempt.implant_membership_count = result.implant_membership_count;
  attempt.gate_query_visit_count = result.gate_query_visit_count;
  attempt.gate_candidate_count = result.gate_candidate_count;
  attempt.gate_raw_hit_count = result.gate_raw_hit_count;
  attempt.gate_uncertain_count = result.gate_uncertain_count;
  attempt.contact_query_visit_count = result.contact_query_visit_count;
  attempt.contact_candidate_count = result.contact_candidate_count;
  attempt.contact_raw_hit_count = result.contact_raw_hit_count;
  attempt.contact_uncertain_count = result.contact_uncertain_count;
  attempt.setup_ns = result.setup_ns;
  attempt.h2d_ns = result.h2d_ns;
  attempt.implant_expand_ns = result.implant_expand_ns;
  attempt.grid_count_ns = result.grid_count_ns;
  attempt.grid_build_ns = result.grid_build_ns;
  attempt.gate_query_ns = result.gate_query_ns;
  attempt.contact_query_ns = result.contact_query_ns;
  attempt.d2h_ns = result.d2h_ns;
  attempt.total_ns = result.total_ns;
  attempt.message.assign (
    result.message,
    std::find (
      result.message, result.message + sizeof (result.message), '\0'));

  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size != sizeof (result) ||
      result.reserved0 != 0 || result.reserved1 != 0) {
    attempt.disposition = CudaImplant12Attempt::InvalidResult;
    attempt.message =
      "CUDA IMPLANT.1/.2 backend returned an incompatible result";
    log_implant12_attempt (attempt);
    return attempt;
  }

  if (status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.status == KLAYOUT_CUDA_SPATIAL_OK) {
    const bool echo_matches =
      result.opcode == request.opcode &&
      result.option_flags == request.option_flags &&
      result.format_version == request.format_version &&
      result.requested_mask == request.requested_mask &&
      result.dbu_per_micron == request.dbu_per_micron &&
      result.root_cell == request.root_cell &&
      result.implant1_distance == request.implant1_distance &&
      result.implant2_distance == request.implant2_distance &&
      result.grid_cell_size == request.grid_cell_size &&
      result.implant_left == request.implant_left &&
      result.implant_bottom == request.implant_bottom &&
      result.implant_right == request.implant_right &&
      result.implant_top == request.implant_top &&
      std::equal (
        result.scene_digest, result.scene_digest + 32,
        request.scene_digest) &&
      result.context_count == request.context_count &&
      result.implant_context_count == request.implant_context_count &&
      result.gate_context_count == request.gate_context_count &&
      result.contact_context_count == request.contact_context_count &&
      result.cell_count == request.cell_count &&
      result.contour_count == request.contour_count &&
      result.edge_count == request.edge_count &&
      result.flat_implant_polygon_count ==
        request.flat_implant_polygon_count &&
      result.flat_gate_polygon_count == request.flat_gate_polygon_count &&
      result.flat_contact_polygon_count ==
        request.flat_contact_polygon_count &&
      result.flat_implant_contour_count ==
        request.flat_implant_contour_count &&
      result.flat_gate_contour_count == request.flat_gate_contour_count &&
      result.flat_contact_contour_count ==
        request.flat_contact_contour_count &&
      result.flat_implant_edge_count == request.flat_implant_edge_count &&
      result.flat_gate_edge_count == request.flat_gate_edge_count &&
      result.flat_contact_edge_count == request.flat_contact_edge_count;

    uint64_t maximum_gate_candidates = 0;
    uint64_t maximum_contact_candidates = 0;
    const bool gate_product_fits = checked_multiply_u64 (
      request.flat_implant_edge_count, request.flat_gate_edge_count,
      maximum_gate_candidates);
    const bool contact_product_fits = checked_multiply_u64 (
      request.flat_implant_edge_count, request.flat_contact_edge_count,
      maximum_contact_candidates);
    const bool counters_fit =
      result.implant_expanded_edge_count ==
        request.flat_implant_edge_count &&
      result.gate_processed_edge_count == request.flat_gate_edge_count &&
      result.contact_processed_edge_count ==
        request.flat_contact_edge_count &&
      result.grid_cell_count != 0 &&
      result.grid_cell_count <= request.max_grid_cells &&
      result.implant_membership_count >=
        result.implant_expanded_edge_count &&
      result.implant_membership_count <=
        request.max_implant_memberships &&
      result.gate_query_visit_count <= request.max_gate_query_visits &&
      result.gate_candidate_count <= request.max_gate_candidate_work &&
      (! gate_product_fits ||
       result.gate_candidate_count <= maximum_gate_candidates) &&
      result.gate_raw_hit_count <= result.gate_candidate_count &&
      result.gate_uncertain_count <=
        result.gate_candidate_count - result.gate_raw_hit_count &&
      result.contact_query_visit_count <=
        request.max_contact_query_visits &&
      result.contact_candidate_count <=
        request.max_contact_candidate_work &&
      (! contact_product_fits ||
       result.contact_candidate_count <= maximum_contact_candidates) &&
      result.contact_raw_hit_count <= result.contact_candidate_count &&
      result.contact_uncertain_count <=
        result.contact_candidate_count - result.contact_raw_hit_count &&
      (result.certified_empty_mask & ~request.requested_mask) == 0 &&
      (result.clean_mask & ~request.requested_mask) == 0;

    if (! echo_matches || ! counters_fit ||
        result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        result.device_flags != 0) {
      attempt.disposition = CudaImplant12Attempt::InvalidResult;
      attempt.message =
        "CUDA IMPLANT.1/.2 backend returned a mismatched proof echo";
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_IMPLANT12_COMPLETE &&
      result.certified_empty_mask == request.requested_mask &&
      result.clean_mask == request.requested_mask &&
      result.gate_raw_hit_count == 0 &&
      result.gate_uncertain_count == 0 &&
      result.contact_raw_hit_count == 0 &&
      result.contact_uncertain_count == 0) {
      attempt.disposition = CudaImplant12Attempt::CertifiedEmpty;
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_HITS &&
      (result.gate_raw_hit_count != 0 ||
       result.contact_raw_hit_count != 0) &&
      result.gate_uncertain_count == 0 &&
      result.contact_uncertain_count == 0 &&
      (result.certified_empty_mask != request.requested_mask ||
       result.clean_mask != request.requested_mask)) {
      attempt.disposition = CudaImplant12Attempt::RawHits;
    } else if (
      result.disposition == KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN &&
      (result.gate_uncertain_count != 0 ||
       result.contact_uncertain_count != 0)) {
      attempt.disposition = CudaImplant12Attempt::BackendFallback;
    } else {
      attempt.disposition = CudaImplant12Attempt::InvalidResult;
      attempt.message =
        "CUDA IMPLANT.1/.2 backend returned an inconsistent disposition";
    }
  } else if (
    status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
    result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaImplant12Attempt::BackendFallback;
  } else {
    attempt.disposition = CudaImplant12Attempt::BackendError;
  }

  log_implant12_attempt (attempt);
  return attempt;
}

bool cuda_spatial_implant12_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.implant12_ready ();
}

} // namespace db
