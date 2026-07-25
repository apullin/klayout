/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaSpatialBackend
#define HDR_dbCudaSpatialBackend

#include "dbCommon.h"
#include "dbCudaSpatialApi.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace db
{

struct DB_PUBLIC CudaSpatialAttempt
{
  enum Disposition
  {
    Disabled,
    BelowThreshold,
    Success,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaSpatialAttempt ();

  Disposition disposition;
  uint32_t fallback_flags;
  uint64_t membership_count;
  uint64_t occupied_cell_count;
  uint64_t pair_work_count;
  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t broad_phase_ns;
  uint64_t sort_unique_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  std::string message;
  std::vector<uint64_t> pair_keys;
};

struct DB_PUBLIC CudaActive3Attempt
{
  enum Disposition
  {
    Disabled,
    CertifiedEmpty,
    RawHits,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaActive3Attempt ();

  Disposition disposition;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t well_context_count;
  uint64_t active_context_count;
  uint64_t cell_count;
  uint64_t edge_count;
  uint64_t flat_well_edge_count;
  uint64_t flat_active_edge_count;
  uint64_t grid_cell_count;
  uint64_t membership_count;
  uint64_t candidate_pair_count;
  uint64_t raw_hit_count;
  uint64_t uncertain_count;
  uint64_t total_ns;
  std::string message;
};

struct DB_PUBLIC CudaM1WidthSpaceAttempt
{
  enum Disposition
  {
    Disabled,
    CertifiedEmpty,
    RawHits,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaM1WidthSpaceAttempt ();

  Disposition disposition;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t metal_context_count;
  uint64_t cell_count;
  uint64_t polygon_count;
  uint64_t edge_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  uint64_t grid_cell_count;
  uint64_t membership_count;
  uint64_t pair_work_count;
  uint64_t unique_edge_pair_count;
  uint64_t width_pair_count;
  uint64_t space_pair_count;
  uint64_t width_hit_count;
  uint64_t space_hit_count;
  uint64_t width_uncertain_count;
  uint64_t space_uncertain_count;
  uint64_t total_ns;
  std::string message;
};

struct DB_PUBLIC CudaM2UnionAttempt
{
  enum Disposition
  {
    Disabled,
    Complete,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaM2UnionAttempt ();

  Disposition disposition;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t metal_context_count;
  uint64_t cell_count;
  uint64_t polygon_count;
  uint64_t edge_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  uint64_t rectangle_count;
  uint64_t x_slab_count;
  uint64_t membership_count;
  uint64_t event_count;
  uint64_t strip_interval_count;
  uint64_t raw_segment_count;
  uint64_t boundary_fnv64;
  uint64_t total_ns;
  std::string message;
  std::vector<klayout_cuda_spatial_m2_union_segment_v1> segments;
};

/**
 * Additive component timing returned only by the explicitly versioned timed
 * M2-union wrapper.  Keeping this POD separate preserves the established
 * returned-by-value CudaM2UnionAttempt binary layout.
 */
struct DB_PUBLIC CudaM2UnionTiming
{
  enum
  {
    FormatVersion = 1
  };

  uint32_t format_version;
  uint32_t struct_size;
  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t rectangle_expand_ns;
  uint64_t x_membership_ns;
  uint64_t strip_scan_ns;
  uint64_t boundary_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
};

struct DB_PUBLIC CudaPoly34Attempt
{
  enum Disposition
  {
    Disabled,
    CertifiedEmpty,
    NotEmpty,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaPoly34Attempt ();

  Disposition disposition;
  uint32_t certified_empty_mask;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t poly_context_count;
  uint64_t active_context_count;
  uint64_t gate_context_count;
  uint64_t cell_count;
  uint64_t box_count;
  uint64_t flat_poly_box_count;
  uint64_t flat_active_box_count;
  uint64_t flat_gate_box_count;
  uint64_t poly_membership_count;
  uint64_t active_membership_count;
  uint64_t poly_query_visit_count;
  uint64_t active_query_visit_count;
  uint64_t poly_candidate_count;
  uint64_t active_candidate_count;
  uint64_t poly_terminal_empty_count;
  uint64_t active_terminal_empty_count;
  uint64_t atomic_terminal_empty_count;
  uint64_t fallback_gate_count;
  uint64_t total_ns;
  std::string message;
};

struct DB_PUBLIC CudaVia1StackAttempt
{
  enum Disposition
  {
    Disabled,
    CertifiedEmpty,
    NotEmpty,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaVia1StackAttempt ();

  Disposition disposition;
  uint32_t certified_empty_mask;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t flat_metal1_box_count;
  uint64_t flat_via1_box_count;
  uint64_t flat_metal2_box_count;
  uint64_t via_expanded_count;
  uint64_t via_size_checked_count;
  uint64_t via_size_violation_count;
  uint64_t metal1_expanded_count;
  uint64_t metal2_expanded_count;
  uint64_t grid_cell_count;
  uint64_t via_membership_count;
  uint64_t metal1_membership_count;
  uint64_t metal2_membership_count;
  uint64_t via_pair_queried_count;
  uint64_t via_candidate_pair_count;
  uint64_t duplicate_via_pair_count;
  uint64_t unsafe_via_pair_count;
  uint64_t spacing_violation_count;
  uint64_t clean_via_pair_count;
  uint64_t metal1_queried_count;
  uint64_t metal1_candidate_count;
  uint64_t metal1_certified_count;
  uint64_t metal1_miss_count;
  uint64_t metal2_queried_count;
  uint64_t metal2_candidate_count;
  uint64_t metal2_certified_count;
  uint64_t metal2_miss_count;
  uint64_t total_ns;
  std::string message;
};

struct DB_PUBLIC CudaImplant12Attempt
{
  enum Disposition
  {
    Disabled,
    CertifiedEmpty,
    RawHits,
    BackendFallback,
    BackendError,
    InvalidResult
  };

  CudaImplant12Attempt ();

  Disposition disposition;
  uint32_t certified_empty_mask;
  uint32_t clean_mask;
  uint32_t fallback_flags;
  uint32_t device_flags;
  uint64_t context_count;
  uint64_t implant_context_count;
  uint64_t gate_context_count;
  uint64_t contact_context_count;
  uint64_t cell_count;
  uint64_t contour_count;
  uint64_t edge_count;
  uint64_t flat_implant_polygon_count;
  uint64_t flat_gate_polygon_count;
  uint64_t flat_contact_polygon_count;
  uint64_t flat_implant_contour_count;
  uint64_t flat_gate_contour_count;
  uint64_t flat_contact_contour_count;
  uint64_t flat_implant_edge_count;
  uint64_t flat_gate_edge_count;
  uint64_t flat_contact_edge_count;
  uint64_t implant_expanded_edge_count;
  uint64_t gate_processed_edge_count;
  uint64_t contact_processed_edge_count;
  uint64_t grid_cell_count;
  uint64_t implant_membership_count;
  uint64_t gate_query_visit_count;
  uint64_t gate_candidate_count;
  uint64_t gate_raw_hit_count;
  uint64_t gate_uncertain_count;
  uint64_t contact_query_visit_count;
  uint64_t contact_candidate_count;
  uint64_t contact_raw_hit_count;
  uint64_t contact_uncertain_count;
  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t implant_expand_ns;
  uint64_t grid_count_ns;
  uint64_t grid_build_ns;
  uint64_t gate_query_ns;
  uint64_t contact_query_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  std::string message;
};

/**
 * Try the optional CUDA bipartite broad phase.
 *
 * The backend is disabled unless KLAYOUT_CUDA_SPATIAL_BACKEND is set.  Any
 * loader, capacity, CUDA, or result-validation failure is represented by a
 * non-Success disposition and is expected to fall back to the CPU scanner.
 */
DB_PUBLIC CudaSpatialAttempt cuda_spatial_try_bipartite (
  const std::vector<klayout_cuda_spatial_aabb_v1> &subjects,
  const std::vector<klayout_cuda_spatial_aabb_v1> &intruders,
  int64_t enlargement);

/**
 * Try the optional CUDA self-AABB broad phase.
 *
 * Returned keys contain two distinct one-based record IDs in ascending order.
 * A backend without the optional self entry point fails closed to the caller.
 */
DB_PUBLIC CudaSpatialAttempt cuda_spatial_try_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement);

/** Return true when an enabled, loaded backend would accept this record count. */
DB_PUBLIC bool cuda_spatial_may_attempt (uint64_t subject_count,
                                         uint64_t intruder_count);

/** Return true when the optional self entry point accepts this record count. */
DB_PUBLIC bool cuda_spatial_may_attempt_self (uint64_t record_count);

/**
 * Check that a self-AABB request fits the configured per-record and aggregate
 * membership limits without launching the backend.
 *
 * On success, membership_count is the exact number of grid memberships the
 * backend will allocate for these records.  Coordinate, grid-span, or capacity
 * uncertainty fails closed.
 */
DB_PUBLIC bool cuda_spatial_preflight_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement, uint64_t &membership_count);

/** Return true only when the opt-in module was requested and loaded. */
DB_PUBLIC bool cuda_spatial_requested ();

/**
 * Invoke the optional live ACTIVE.3 raw-superset empty certificate.
 *
 * Only CertifiedEmpty is usable by a caller.  RawHits deliberately carries
 * no KLayout markers and requests the unchanged CPU implementation.
 */
DB_PUBLIC CudaActive3Attempt cuda_spatial_try_active3_empty (
  const klayout_cuda_spatial_active3_request_v1 &request);

/** Return true only when the independent ACTIVE.3 opt-in and symbol exist. */
DB_PUBLIC bool cuda_spatial_active3_requested ();

/**
 * Invoke the optional CONTACT.4 raw-CONTACT-superset empty certificate.
 *
 * The request reuses the ACTIVE.3 scene ABI: the historical WELL fields hold
 * indexed raw CONTACT (the secondary relation operand), while ACTIVE fields
 * hold streamed merged ACTIVE (the primary operand).  Only CertifiedEmpty is
 * consumable; raw hits and every bounded decline retain the CPU rule.
 */
DB_PUBLIC CudaActive3Attempt cuda_spatial_try_contact4_empty (
  const klayout_cuda_spatial_active3_request_v1 &request);

/** Return true only when the independent CONTACT.4 opt-in and symbol exist. */
DB_PUBLIC bool cuda_spatial_contact4_requested ();

/**
 * Invoke the optional atomic METAL1.1/METAL1.2 empty certificate.
 *
 * Only CertifiedEmpty is consumable.  Raw hits and every uncertain, malformed,
 * over-capacity, loader, or CUDA outcome request the unchanged two-rule CPU
 * batch.
 */
DB_PUBLIC CudaM1WidthSpaceAttempt cuda_spatial_try_m1_width_space_empty (
  const klayout_cuda_spatial_m1_width_space_request_v1 &request);

/** Return true only when the M1 width/space opt-in and symbol exist. */
DB_PUBLIC bool cuda_spatial_m1_width_space_requested ();

/** Return true only when the independent M2 width/space opt-in and symbol exist. */
DB_PUBLIC bool cuda_spatial_m2_width_space_requested ();

/**
 * Invoke the optional exact raw-M2 Manhattan-union boundary.
 *
 * Complete is usable only after this wrapper has validated the complete proof
 * echo, copied and released the backend-owned output, and established
 * canonical maximal segment order plus the echoed FNV-1a digest.
 */
DB_PUBLIC CudaM2UnionAttempt cuda_spatial_try_m2_union (
  const klayout_cuda_spatial_m2_union_request_v1 &request);

/**
 * Invoke the same wrapper while copying charged component timing into a
 * size-checked additive POD.  This distinct symbol makes a stale qmake DB
 * library fail at link/load time instead of silently changing attempt layout.
 */
DB_PUBLIC CudaM2UnionAttempt cuda_spatial_try_m2_union_with_timing (
  const klayout_cuda_spatial_m2_union_request_v1 &request,
  CudaM2UnionTiming *timing, uint32_t timing_struct_size);

/** Return true only when the M2-rules opt-in and both union symbols exist. */
DB_PUBLIC bool cuda_spatial_m2_union_requested ();

/**
 * Validate the proven canonical segment contract and FNV-1a digest.
 *
 * This small public helper keeps loader validation and focused unit tests on
 * the identical (axis, side, fixed, lo, hi) ordering rule.
 */
DB_PUBLIC bool cuda_spatial_validate_m2_union_boundary (
  const klayout_cuda_spatial_m2_union_segment_v1 *segments,
  uint64_t segment_count, uint64_t expected_fnv64,
  std::string *error = 0);

/**
 * Invoke the optional atomic POLY.3/POLY.4 terminal-empty certificate.
 *
 * Only CertifiedEmpty is consumable.  Every positive-area or conservative
 * miss, uncertainty, malformed echo, capacity, loader or CUDA outcome
 * requires the complete unchanged two-rule CPU transaction.
 */
DB_PUBLIC CudaPoly34Attempt cuda_spatial_try_poly34_empty (
  const klayout_cuda_spatial_poly34_request_v1 &request);

/** Return true only when the independent POLY.3/.4 opt-in and symbol exist. */
DB_PUBLIC bool cuda_spatial_poly34_requested ();

/**
 * Invoke the optional atomic M1/VIA1/M2 six-rule empty certificate.
 *
 * Only CertifiedEmpty is consumable.  Every other disposition requests the
 * complete unchanged CPU stack; partial result masks are telemetry only.
 */
DB_PUBLIC CudaVia1StackAttempt cuda_spatial_try_via1_stack_empty (
  const klayout_cuda_spatial_via1_stack_request_v1 &request);

/** Return true only when the VIA1-stack opt-in and optional symbol exist. */
DB_PUBLIC bool cuda_spatial_via1_stack_requested ();

/**
 * Return true when the CONTACT/METAL1.3 opt-in can reuse the qualified
 * VIA1-stack projection-certificate symbol.
 */
DB_PUBLIC bool cuda_spatial_m1_contact_requested ();

/**
 * Invoke the optional atomic IMPLANT.1/IMPLANT.2 empty certificate.
 *
 * Only CertifiedEmpty is consumable.  Raw hits, uncertainty, partial masks,
 * malformed echoes, capacity exhaustion and loader failures all retain both
 * unchanged CPU rules.
 */
DB_PUBLIC CudaImplant12Attempt cuda_spatial_try_implant12_empty (
  const klayout_cuda_spatial_implant12_request_v1 &request);

/** Return true only when the IMPLANT.1/.2 opt-in and symbol exist. */
DB_PUBLIC bool cuda_spatial_implant12_requested ();

} // namespace db

#endif
