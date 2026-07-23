/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaSpatialApi
#define HDR_dbCudaSpatialApi

/*
 * Versioned POD C ABI between KLayout and an optional spatial accelerator.
 * No KLayout object pointer crosses this boundary: the pointers below address
 * only caller-owned POD arrays or backend-owned result storage.  Keeping this
 * header independent of CUDA lets an ordinary KLayout build discover the
 * backend with dlopen/LoadLibrary at runtime.
 */

#include <stdint.h>

#if defined(_WIN32)
#  if defined(KLAYOUT_CUDA_SPATIAL_BACKEND_BUILD)
#    define KLAYOUT_CUDA_SPATIAL_EXPORT __declspec(dllexport)
#  else
#    define KLAYOUT_CUDA_SPATIAL_EXPORT
#  endif
#elif defined(__GNUC__)
#  define KLAYOUT_CUDA_SPATIAL_EXPORT __attribute__((visibility("default")))
#else
#  define KLAYOUT_CUDA_SPATIAL_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define KLAYOUT_CUDA_SPATIAL_ABI_VERSION 1u

enum klayout_cuda_spatial_status
{
  KLAYOUT_CUDA_SPATIAL_OK = 0,
  KLAYOUT_CUDA_SPATIAL_FALLBACK = 1,
  KLAYOUT_CUDA_SPATIAL_ERROR = 2,
  KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT = 3
};

enum klayout_cuda_spatial_fallback_flag
{
  KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE = 0,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_DENSE_CELL = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_CAPACITY = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT = 1u << 7
};

struct klayout_cuda_spatial_aabb_v1
{
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
};

struct klayout_cuda_spatial_config_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  int32_t device;
  uint32_t max_cells_per_record;
  uint32_t max_records_per_cell;
  uint32_t reserved0;
  uint64_t cell_size;
  uint64_t max_memberships;
  uint64_t max_pair_work;
  uint64_t max_candidates;
};

struct klayout_cuda_spatial_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  const struct klayout_cuda_spatial_aabb_v1 *subjects;
  uint64_t subject_count;
  const struct klayout_cuda_spatial_aabb_v1 *intruders;
  uint64_t intruder_count;
  int64_t enlargement;
  const struct klayout_cuda_spatial_config_v1 *config;
};

/*
 * For the bipartite entry point, a pair key stores the one-based subject ID
 * in bits 63..32 and the one-based global intruder ID in bits 31..0.
 * Intruder IDs start at subject_count + 1.
 *
 * For the self entry point, subjects is the sole input array: intruders must
 * be null and intruder_count must be zero.  A pair key stores two distinct
 * one-based subject IDs in ascending order.  The self scan enumerates each
 * unordered pair at most once per cell before cross-cell deduplication.
 *
 * Successful results from either entry point are strictly increasing and
 * unique.  pair_keys is owned by the backend until release_result is called.
 */
struct klayout_cuda_spatial_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  const uint64_t *pair_keys;
  uint64_t pair_count;
  uint64_t membership_count;
  uint64_t occupied_cell_count;
  uint64_t pair_work_count;
  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t broad_phase_ns;
  uint64_t sort_unique_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  char message[192];
};

typedef uint32_t (*klayout_cuda_spatial_abi_version_func) (void);
typedef int (*klayout_cuda_spatial_run_bipartite_v1_func) (
  const struct klayout_cuda_spatial_request_v1 *,
  struct klayout_cuda_spatial_result_v1 *);
typedef int (*klayout_cuda_spatial_run_self_v1_func) (
  const struct klayout_cuda_spatial_request_v1 *,
  struct klayout_cuda_spatial_result_v1 *);
typedef void (*klayout_cuda_spatial_release_result_v1_func) (
  struct klayout_cuda_spatial_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT uint32_t klayout_cuda_spatial_abi_version (void);
KLAYOUT_CUDA_SPATIAL_EXPORT int klayout_cuda_spatial_run_bipartite_v1 (
  const struct klayout_cuda_spatial_request_v1 *request,
  struct klayout_cuda_spatial_result_v1 *result);
KLAYOUT_CUDA_SPATIAL_EXPORT int klayout_cuda_spatial_run_self_v1 (
  const struct klayout_cuda_spatial_request_v1 *request,
  struct klayout_cuda_spatial_result_v1 *result);
KLAYOUT_CUDA_SPATIAL_EXPORT void klayout_cuda_spatial_release_result_v1 (
  struct klayout_cuda_spatial_result_v1 *result);

/*
 * Optional METAL1.3 enclosure certificate.
 *
 * This is an additive entry point beside the original v1 AABB API above.  It
 * deliberately keeps KLAYOUT_CUDA_SPATIAL_ABI_VERSION at 1 and does not change
 * the size or interpretation of any existing v1 structure.
 *
 * A contact ID is caller-defined, stable, nonzero and unique within a request.
 * Context IDs restrict comparisons to geometry represented in the same
 * flattened hierarchy context.  Directed M1 edges use KLayout's polygon
 * convention (material on the right side of the edge).
 *
 * COMPLETE is meaningful only relative to a caller-proven complete input
 * universe: every relevant merged contact and directed merged M1 edge must be
 * present in its interaction context.  Before using COMPLETE to skip a CPU
 * operation, the caller must also bind the request to an unchanged
 * scene/rule fingerprint.  This stateless proof API cannot establish those
 * hierarchy-serialization facts by itself.
 */
enum klayout_cuda_spatial_m1_opcode
{
  KLAYOUT_CUDA_SPATIAL_M1_ENCLOSED_PROJECTION_ONE_OR_OPPOSITE = 1
};

enum klayout_cuda_spatial_m1_disposition
{
  /* No contact has a surviving marker in the supplied complete input domain. */
  KLAYOUT_CUDA_SPATIAL_M1_COMPLETE = 0,
  /* At least one contact contained geometry the bounded proof cannot decide. */
  KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN = 1,
  /* The scan completed, but at least one rectangle mask was not waivable. */
  KLAYOUT_CUDA_SPATIAL_M1_DISALLOWED = 2
};

enum klayout_cuda_spatial_m1_survivor_flag
{
  KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_UNCERTAIN = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_DISALLOWED_MASK = 1u << 1
};

struct klayout_cuda_spatial_m1_contact_v1
{
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
  uint64_t contact_id;
  uint32_t context_id;
  uint32_t reserved0;
};

struct klayout_cuda_spatial_m1_edge_v1
{
  int64_t x1;
  int64_t y1;
  int64_t x2;
  int64_t y2;
  uint32_t context_id;
  uint32_t reserved0;
};

struct klayout_cuda_spatial_m1_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t reserved0;
  const struct klayout_cuda_spatial_m1_contact_v1 *contacts;
  uint64_t contact_count;
  const struct klayout_cuda_spatial_m1_edge_v1 *metal1_edges;
  uint64_t metal1_edge_count;
  int64_t distance;
  const struct klayout_cuda_spatial_config_v1 *config;
};

/*
 * A survivor is a contact which cannot be culled from a future fused CPU/GPU
 * path.  deficient_side_mask uses clockwise rectangle sides:
 *
 *   bit 0 = left, bit 1 = top, bit 2 = right, bit 3 = bottom.
 *
 * Masks 0, a singleton bit, 0x5 and 0xa are waivable.  A partial projection or
 * non-Manhattan candidate sets UNCERTAIN and keeps the contact in this list.
 * Survivors are sorted by contact_id.  The array is backend-owned until the
 * dedicated release function is called.
 */
struct klayout_cuda_spatial_m1_survivor_v1
{
  uint64_t contact_id;
  uint32_t deficient_side_mask;
  uint32_t flags;
};

struct klayout_cuda_spatial_m1_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t reserved0;
  const struct klayout_cuda_spatial_m1_survivor_v1 *survivors;
  uint64_t survivor_count;
  uint64_t contact_count;
  uint64_t metal1_edge_count;
  uint64_t membership_count;
  uint64_t occupied_cell_count;
  uint64_t pair_work_count;
  uint64_t broad_candidate_count;
  uint64_t full_side_hit_count;
  uint64_t partial_candidate_count;
  uint64_t non_manhattan_candidate_count;
  uint64_t uncertain_contact_count;
  uint64_t disallowed_contact_count;
  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t broad_phase_ns;
  uint64_t classify_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_m1_enclosure_v1_func) (
  const struct klayout_cuda_spatial_m1_request_v1 *,
  struct klayout_cuda_spatial_m1_result_v1 *);
typedef void (*klayout_cuda_spatial_release_m1_result_v1_func) (
  struct klayout_cuda_spatial_m1_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int klayout_cuda_spatial_run_m1_enclosure_v1 (
  const struct klayout_cuda_spatial_m1_request_v1 *request,
  struct klayout_cuda_spatial_m1_result_v1 *result);
KLAYOUT_CUDA_SPATIAL_EXPORT void klayout_cuda_spatial_release_m1_result_v1 (
  struct klayout_cuda_spatial_m1_result_v1 *result);

/*
 * Optional ACTIVE.3 empty certificate over a caller-qualified live hierarchy.
 *
 * This is another additive v1 entry point.  The caller expands the regular
 * hierarchy into compact contexts, but retains per-cell edge templates:
 * ACTIVE edges are streamed from templates on the device and are never
 * materialized as one flat array.
 *
 * The ACTIVE operand is intentionally the raw (unmerged) DeepLayer.  A raw
 * hit is not an exact KLayout marker and must only request pristine CPU
 * fallback.  COMPLETE is the sole consumable outcome and means that the
 * complete raw-ACTIVE superset had zero hits and zero uncertainty.
 */
enum klayout_cuda_spatial_active3_opcode
{
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY = 1
};

enum klayout_cuda_spatial_active3_option_flag
{
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_OVERLAP_RELATION = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_DIFFERENT_POLYGONS = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_EUCLIDIAN = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_IGNORE_ANGLE_90 = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WHOLE_EDGES_FALSE = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_PROJECTION_DEFAULTS = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_SHIELDED = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_NO_FILTERS_OR_NEGATIVE = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_IGNORE_PROPERTIES = 1u << 8,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_INCLUDE_TOUCHING = 1u << 9,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_SAME_STORE_AND_TOP = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_NO_BREAKOUT = 1u << 11,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_GEOMETRY = 1u << 12,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_ACTIVE_SUPERSET = 1u << 13
};

#define KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_OPTIONS \
  ((1u << 14) - 1u)

enum klayout_cuda_spatial_active3_disposition
{
  /* Complete qualified raw-ACTIVE universe, zero hits and zero uncertainty. */
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_COMPLETE = 0,
  /* Raw-superset hits were found.  They are not publishable KLayout markers. */
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_HITS = 1,
  /* The bounded backend could not establish either result exactly. */
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN = 2
};

struct klayout_cuda_spatial_active3_context_v1
{
  int64_t tx;
  int64_t ty;
  uint32_t cell_id;
  uint32_t transform_code;
};

struct klayout_cuda_spatial_active3_cell_v1
{
  uint64_t well_edge_begin;
  uint64_t active_edge_begin;
  uint32_t well_edge_count;
  uint32_t active_edge_count;
};

struct klayout_cuda_spatial_active3_edge_v1
{
  int64_t x1;
  int64_t y1;
  int64_t x2;
  int64_t y2;
};

struct klayout_cuda_spatial_active3_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t dbu_per_micron;
  uint32_t reserved0;
  int64_t distance;
  int64_t grid_cell_size;

  const struct klayout_cuda_spatial_active3_context_v1 *contexts;
  uint64_t context_count;
  const uint32_t *well_contexts;
  uint64_t well_context_count;
  const uint64_t *well_offsets;
  uint64_t well_offset_count;
  const uint32_t *active_contexts;
  uint64_t active_context_count;
  const struct klayout_cuda_spatial_active3_cell_v1 *cells;
  uint64_t cell_count;
  const struct klayout_cuda_spatial_active3_edge_v1 *edges;
  uint64_t edge_count;

  uint64_t flat_well_edge_count;
  uint64_t flat_active_edge_count;
  int64_t well_left;
  int64_t well_bottom;
  int64_t well_right;
  int64_t well_top;

  uint64_t max_contexts;
  uint64_t max_grid_cells;
  uint64_t max_memberships;
  uint64_t max_pair_work;
  uint8_t scene_digest[32];
  uint64_t reserved1;
};

struct klayout_cuda_spatial_active3_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t dbu_per_micron;
  int64_t distance;
  int64_t grid_cell_size;
  uint8_t scene_digest[32];

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
  uint32_t device_flags;
  uint32_t reserved0;

  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t well_expand_ns;
  uint64_t grid_build_ns;
  uint64_t active_query_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_active3_empty_v1_func) (
  const struct klayout_cuda_spatial_active3_request_v1 *,
  struct klayout_cuda_spatial_active3_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int klayout_cuda_spatial_run_active3_empty_v1 (
  const struct klayout_cuda_spatial_active3_request_v1 *request,
  struct klayout_cuda_spatial_active3_result_v1 *result);

/*
 * Optional atomic VIA1-stack empty certificate.
 *
 * The caller supplies one qualified hierarchy containing raw M1, VIA1 and M2
 * box templates.  Manhattan metal polygons are represented by conservative
 * rectangle witnesses (normally the union of exact X- and Y-slab
 * decompositions); VIA1 templates must be exact rectangles.  The backend
 * expands VIA1 once, retains it on the device, and certifies the complete
 * six-rule stack in one transaction.
 *
 * The first implementation accepts only requested_mask == ALL.  A partial
 * certified_empty_mask is diagnostic telemetry and must never be used to skip
 * an individual CPU rule.  A caller may bypass the historical six CPU checks
 * only when every requested bit is certified and every proof echo matches.
 */
enum klayout_cuda_spatial_via1_stack_opcode
{
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_EMPTY = 1
};

enum klayout_cuda_spatial_via1_stack_rule
{
  KLAYOUT_CUDA_SPATIAL_VIA1_METAL1_4 = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_VIA1_1 = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_VIA1_2 = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_VIA1_3 = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_VIA1_4 = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_VIA1_METAL2_3 = 1u << 5
};

#define KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES ((1u << 6) - 1u)

enum klayout_cuda_spatial_via1_stack_option_flag
{
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_SAME_STORE_AND_TOP = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NO_BREAKOUT = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_MANHATTAN_METAL = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_RECTANGULAR_CUT = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_EXACT_DUAL_SLABS = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_STRICT_EUCLIDEAN = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_IGNORE_PROPERTIES = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_RAW_COMPLETE_LAYERS = 1u << 7
};

#define KLAYOUT_CUDA_SPATIAL_VIA1_STACK_QUALIFIED_OPTIONS ((1u << 8) - 1u)

enum klayout_cuda_spatial_via1_stack_disposition
{
  /* Every requested rule is certified empty. */
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE = 0,
  /* The exact scan found a miss or violation; run all six CPU rules. */
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY = 1,
  /* Capacity, malformed input, or a device invariant prevented a proof. */
  KLAYOUT_CUDA_SPATIAL_VIA1_STACK_UNCERTAIN = 2
};

struct klayout_cuda_spatial_via1_stack_context_v1
{
  int64_t tx;
  int64_t ty;
  uint32_t cell_id;
  uint32_t transform_code;
};

struct klayout_cuda_spatial_via1_stack_box_v1
{
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
};

struct klayout_cuda_spatial_via1_stack_cell_v1
{
  uint64_t metal1_box_begin;
  uint64_t via1_box_begin;
  uint64_t metal2_box_begin;
  uint32_t metal1_box_count;
  uint32_t via1_box_count;
  uint32_t metal2_box_count;
  uint32_t reserved0;
};

struct klayout_cuda_spatial_via1_stack_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t requested_mask;
  uint32_t dbu_per_micron;
  int32_t device;
  uint32_t reserved0;

  int64_t enclosure_distance;
  int64_t cut_width;
  int64_t cut_height;
  int64_t spacing_distance;
  int64_t grid_cell_size;

  const struct klayout_cuda_spatial_via1_stack_context_v1 *contexts;
  uint64_t context_count;
  const uint32_t *metal1_contexts;
  uint64_t metal1_context_count;
  const uint64_t *metal1_offsets;
  uint64_t metal1_offset_count;
  const uint32_t *via1_contexts;
  uint64_t via1_context_count;
  const uint64_t *via1_offsets;
  uint64_t via1_offset_count;
  const uint32_t *metal2_contexts;
  uint64_t metal2_context_count;
  const uint64_t *metal2_offsets;
  uint64_t metal2_offset_count;
  const struct klayout_cuda_spatial_via1_stack_cell_v1 *cells;
  uint64_t cell_count;
  const struct klayout_cuda_spatial_via1_stack_box_v1 *boxes;
  uint64_t box_count;

  uint64_t flat_metal1_box_count;
  uint64_t flat_via1_box_count;
  uint64_t flat_metal2_box_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  uint64_t max_contexts;
  uint64_t max_grid_cells;
  uint64_t max_metal_memberships;
  uint64_t max_via_memberships;
  uint64_t max_pair_work;
  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

struct klayout_cuda_spatial_via1_stack_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t requested_mask;
  uint32_t certified_empty_mask;
  uint32_t dbu_per_micron;
  uint32_t device_flags;
  uint32_t reserved0;

  int64_t enclosure_distance;
  int64_t cut_width;
  int64_t cut_height;
  int64_t spacing_distance;
  int64_t grid_cell_size;
  uint8_t scene_digest[32];

  uint64_t context_count;
  uint64_t metal1_context_count;
  uint64_t via1_context_count;
  uint64_t metal2_context_count;
  uint64_t cell_count;
  uint64_t box_count;
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

  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t via_expand_ns;
  uint64_t via_grid_ns;
  uint64_t via_query_ns;
  uint64_t metal1_ns;
  uint64_t metal2_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_via1_stack_empty_v1_func) (
  const struct klayout_cuda_spatial_via1_stack_request_v1 *,
  struct klayout_cuda_spatial_via1_stack_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_via1_stack_empty_v1 (
  const struct klayout_cuda_spatial_via1_stack_request_v1 *request,
  struct klayout_cuda_spatial_via1_stack_result_v1 *result);

#ifdef __cplusplus
}
#endif

#endif
