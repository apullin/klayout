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
 * Optional qualified enclosure empty certificate over a live hierarchy.
 *
 * This is another additive v1 entry point.  The caller expands the regular
 * hierarchy into compact contexts, but retains per-cell edge templates.  The
 * operand in the historical ACTIVE fields is streamed from templates on the
 * device and is never materialized as one flat array.
 *
 * ACTIVE.3 indexes merged WELL and streams raw ACTIVE.  The original
 * CONTACT.4 profile indexes raw CONTACT (its secondary operand) and streams
 * merged ACTIVE (its primary).  The early CONTACT.4 profile streams raw
 * ACTIVE as a conservative superset before KLayout constructs merged ACTIVE.
 * In every profile, a raw hit is not an exact publishable KLayout marker and
 * must only request pristine CPU fallback.  COMPLETE is the sole consumable
 * outcome and means that the complete qualified raw superset had zero hits
 * and zero uncertainty.
 */
enum klayout_cuda_spatial_active3_opcode
{
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY = 1,
  /*
   * CONTACT.4 reuses this scene ABI with the historical WELL fields holding
   * the indexed raw-CONTACT secondary operand and the ACTIVE fields holding
   * the streamed merged-ACTIVE primary operand.
   */
  KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_SUPERSET_EMPTY = 2,
  /*
   * The indexed operand remains raw CONTACT, while the streamed primary is
   * complete raw ACTIVE.  Zero unshielded raw hits soundly certifies the
   * later merged-ACTIVE check empty; any hit or uncertainty falls back.
   */
  KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_SUPERSET_EMPTY = 3,
  /*
   * ACTIVE.3 before WELL construction.  The indexed historical WELL domain
   * is the concatenation of complete raw NWELL and raw PWELL contours, while
   * the streamed ACTIVE domain is also raw.  Every boundary of
   * union(NWELL, PWELL) and merged ACTIVE is an orientation-preserving
   * subsegment of this raw universe.  Consequently only zero unshielded raw
   * hits is consumable; every hit or uncertainty retains the exact WELL union
   * and historical ACTIVE.3 path.  Its Cartesian raw-edge product is
   * telemetry only: max_pair_work bounds actual spatial candidates, as in
   * CONTACT.4.
   */
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_BOTH_SUPERSET_EMPTY = 4
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
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_ACTIVE_SUPERSET = 1u << 13,
  /*
   * The indexed (historical WELL) operand is the relation's secondary
   * operand.  This bit is deliberately absent from the ACTIVE.3 profile.
   */
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_INDEXED_SECONDARY = 1u << 14,
  /*
   * The indexed domain is complete raw NWELL followed by complete raw PWELL,
   * rather than the exact merged WELL boundary used by the established
   * ACTIVE.3 profile.
   */
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_SUPERSET = 1u << 15
};

#define KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_OPTIONS \
  ((1u << 14) - 1u)

#define KLAYOUT_CUDA_SPATIAL_CONTACT4_QUALIFIED_OPTIONS \
  (((1u << 13) - 1u) | KLAYOUT_CUDA_SPATIAL_ACTIVE3_INDEXED_SECONDARY)

#define KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_QUALIFIED_OPTIONS \
  ((1u << 15) - 1u)

#define KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_QUALIFIED_OPTIONS \
  (KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_OPTIONS | \
   KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_WELLS_SUPERSET)

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
 * Optional early CONTACT.4 entry point.  Keeping this distinct from the
 * established ACTIVE.3/merged-ACTIVE symbol lets the host prove capability
 * before serializing the much larger raw-ACTIVE scene.  This entry point
 * accepts only KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_SUPERSET_EMPTY.
 */
typedef int
(*klayout_cuda_spatial_run_contact4_raw_active_empty_v1_func) (
  const struct klayout_cuda_spatial_active3_request_v1 *,
  struct klayout_cuda_spatial_active3_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_contact4_raw_active_empty_v1 (
  const struct klayout_cuda_spatial_active3_request_v1 *request,
  struct klayout_cuda_spatial_active3_result_v1 *result);

/*
 * Optional atomic FreePDK45 IMPLANT.1/IMPLANT.2 empty certificate.
 *
 * The caller supplies one qualified hierarchy with three ordered geometry
 * domains: an exact merged IMPLANT primary, a raw GATE secondary superset and
 * a raw CONTACT secondary superset.  The backend expands and indexes IMPLANT
 * once, then streams GATE and CONTACT while the grid remains resident.
 *
 * A raw hit is not a publishable KLayout marker.  It merely requests the
 * unchanged two-rule CPU transaction.  Both rules may be skipped only when
 * disposition is COMPLETE and both certified_empty_mask and clean_mask equal
 * ALL_RULES.  Partial clean masks are diagnostic and never independently
 * consumable.
 */
enum klayout_cuda_spatial_implant12_opcode
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_SUPERSET_EMPTY = 1
};

enum klayout_cuda_spatial_implant12_rule
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT1_RULE = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT2_RULE = 1u << 1
};

#define KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES ((1u << 2) - 1u)

enum klayout_cuda_spatial_implant12_domain
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_IMPLANT_DOMAIN = 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN = 1,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_CONTACT_DOMAIN = 2,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT = 3
};

enum klayout_cuda_spatial_implant12_option_flag
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_MERGED_IMPLANT_PRIMARY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_GATE_SUPERSET = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_CONTACT_SUPERSET = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_SAME_STORE_LAYOUT_TOP = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_NO_BREAKOUT = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_SPACE_RELATION = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_DIFFERENT_POLYGONS = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_PROJECTION = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_IGNORE_ANGLE_90 = 1u << 8,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_WHOLE_EDGES_FALSE = 1u << 9,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_PROJECTION_DEFAULTS = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_SHIELDED = 1u << 11,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_NO_FILTERS_OR_NEGATIVE = 1u << 12,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_IGNORE_PROPERTIES = 1u << 13,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_INCLUDE_TOUCHING = 1u << 14,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_ORDERED_OUTPUTS = 1u << 15,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_MANHATTAN_CONTOURS = 1u << 16
};

#define KLAYOUT_CUDA_SPATIAL_IMPLANT12_QUALIFIED_OPTIONS \
  ((1u << 17) - 1u)

enum klayout_cuda_spatial_implant12_contour_flag
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_HULL = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_HOLE = 1u << 1
};

enum klayout_cuda_spatial_implant12_disposition
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_HITS = 1,
  KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN = 2
};

struct klayout_cuda_spatial_implant12_context_v1
{
  int64_t tx;
  int64_t ty;
  uint32_t cell_id;
  uint32_t transform_code;
};

struct klayout_cuda_spatial_implant12_domain_span_v1
{
  uint64_t contour_begin;
  uint64_t edge_begin;
  uint32_t polygon_count;
  uint32_t contour_count;
  uint32_t edge_count;
  uint32_t reserved0;
};

struct klayout_cuda_spatial_implant12_cell_v1
{
  uint64_t source_cell_index;
  struct klayout_cuda_spatial_implant12_domain_span_v1
    domains[KLAYOUT_CUDA_SPATIAL_IMPLANT12_DOMAIN_COUNT];
};

struct klayout_cuda_spatial_implant12_contour_v1
{
  uint64_t edge_begin;
  uint32_t polygon_id;
  uint32_t contour_id;
  uint32_t edge_count;
  uint32_t flags;
};

struct klayout_cuda_spatial_implant12_edge_v1
{
  int64_t x1;
  int64_t y1;
  int64_t x2;
  int64_t y2;
};

/*
 * Record pointers are byte-addressed so a qualified C++ scene can bind its
 * existing vector storage after proving sizeof/offsetof compatibility,
 * without a multi-million-record repack or cross-type aliasing.
 */
struct klayout_cuda_spatial_implant12_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t requested_mask;
  int32_t device;
  uint32_t reserved0;
  int64_t implant1_distance;
  int64_t implant2_distance;
  int64_t grid_cell_size;

  const void *contexts;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t context_reserved;
  const uint32_t *implant_contexts;
  uint64_t implant_context_count;
  const uint64_t *implant_edge_offsets;
  uint64_t implant_edge_offset_count;
  const uint32_t *gate_contexts;
  uint64_t gate_context_count;
  const uint32_t *contact_contexts;
  uint64_t contact_context_count;
  const void *cells;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t cell_reserved;
  const void *contours;
  uint64_t contour_count;
  uint32_t contour_record_bytes;
  uint32_t contour_reserved;
  const void *edges;
  uint64_t edge_count;
  uint32_t edge_record_bytes;
  uint32_t edge_reserved;

  uint64_t flat_implant_polygon_count;
  uint64_t flat_gate_polygon_count;
  uint64_t flat_contact_polygon_count;
  uint64_t flat_implant_contour_count;
  uint64_t flat_gate_contour_count;
  uint64_t flat_contact_contour_count;
  uint64_t flat_implant_edge_count;
  uint64_t flat_gate_edge_count;
  uint64_t flat_contact_edge_count;
  int64_t implant_left;
  int64_t implant_bottom;
  int64_t implant_right;
  int64_t implant_top;

  uint64_t max_contexts;
  uint64_t max_grid_cells;
  uint64_t max_implant_memberships;
  uint64_t max_gate_query_visits;
  uint64_t max_gate_candidate_work;
  uint64_t max_contact_query_visits;
  uint64_t max_contact_candidate_work;
  uint64_t max_flat_polygons;
  uint64_t max_flat_contours;
  uint64_t max_flat_edges;
  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

struct klayout_cuda_spatial_implant12_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t requested_mask;
  uint32_t certified_empty_mask;
  uint32_t clean_mask;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t device_flags;
  uint32_t reserved0;
  uint32_t reserved1;
  int64_t implant1_distance;
  int64_t implant2_distance;
  int64_t grid_cell_size;
  int64_t implant_left;
  int64_t implant_bottom;
  int64_t implant_right;
  int64_t implant_top;
  uint8_t scene_digest[32];

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
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_implant12_empty_v1_func) (
  const struct klayout_cuda_spatial_implant12_request_v1 *,
  struct klayout_cuda_spatial_implant12_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_implant12_empty_v1 (
  const struct klayout_cuda_spatial_implant12_request_v1 *request,
  struct klayout_cuda_spatial_implant12_result_v1 *result);

/*
 * Optional atomic METAL1.1/METAL1.2 and METAL2.1/METAL2.2 empty certificate.
 *
 * The caller passes the pointer-free arrays published by
 * CudaM1WidthSpaceScene.  Array records below deliberately match those
 * records field-for-field, while the request carries the canonical scene
 * header, digest, complete hierarchy-context census, and bounded device
 * capacities.  No KLayout object or file-format record crosses this ABI.
 *
 * The backend evaluates an unshielded exact edge-pair superset.  Shielding can
 * remove complete pairs but cannot create one, so COMPLETE with zero raw hits
 * certifies both rules empty.  A hit is diagnostic only and, like uncertainty,
 * capacity exhaustion, a malformed request, or a CUDA error, requires the
 * caller to execute both pristine CPU rules.  The two profile-specific entry
 * points below accept only opcode 1 at 130/130 DBU and opcode 2 at 140/140
 * DBU, respectively.
 */
enum klayout_cuda_spatial_m1_width_space_opcode
{
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_MERGED_EMPTY = 1,
  KLAYOUT_CUDA_SPATIAL_M2_WIDTH_SPACE_MERGED_EMPTY = 2
};

enum klayout_cuda_spatial_m1_width_space_option_flag
{
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_EXACT_MERGED_M1 = 1u << 0,
  /* Additive profile-neutral spelling; the legacy M1 name remains ABI-stable. */
  KLAYOUT_CUDA_SPATIAL_METAL_WIDTH_SPACE_EXACT_MERGED_METAL = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_IDENTICAL_INPUT = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_SAME_STORE_LAYOUT_TOP_LAYER = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_NO_BREAKOUT = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_EUCLIDIAN = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_IGNORE_ANGLE_90 = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_WHOLE_EDGES_FALSE = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_PROJECTION_DEFAULTS = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_SHIELDED = 1u << 8,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_NO_FILTERS_OR_NEGATIVE = 1u << 9,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_IGNORE_PROPERTIES = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_INCLUDE_TOUCHING = 1u << 11
};

#define KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_QUALIFIED_OPTIONS \
  ((1u << 12) - 1u)

enum klayout_cuda_spatial_m1_width_space_disposition
{
  /* Both complete qualified rule universes have zero raw hits. */
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_COMPLETE = 0,
  /* At least one raw width/space hit blocks the atomic certificate. */
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_RAW_HITS = 1,
  /* The bounded backend could not establish a complete exact result. */
  KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN = 2
};

struct klayout_cuda_spatial_m1_width_space_context_v1
{
  int64_t tx;
  int64_t ty;
  uint32_t cell_id;
  uint32_t transform_code;
};

struct klayout_cuda_spatial_m1_width_space_cell_v1
{
  uint64_t source_cell_index;
  uint64_t polygon_begin;
  uint64_t edge_begin;
  uint32_t polygon_count;
  uint32_t edge_count;
};

struct klayout_cuda_spatial_m1_width_space_polygon_v1
{
  uint64_t edge_begin;
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
  uint32_t polygon_id;
  uint32_t edge_count;
};

struct klayout_cuda_spatial_m1_width_space_edge_v1
{
  int64_t x1;
  int64_t y1;
  int64_t x2;
  int64_t y2;
};

/*
 * Record-array pointers are byte-addressed deliberately.  A C++ caller may
 * pass the existing CudaM1WidthSpaceScene vector storage directly after
 * proving sizeof/offsetof compatibility with the ABI records above; the
 * backend validates every stride and reads host records through byte copies,
 * avoiding both a multi-million-record repack and cross-type aliasing.
 */
struct klayout_cuda_spatial_m1_width_space_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t scene_reserved;
  int32_t device;
  uint32_t reserved0;
  int64_t width_distance;
  int64_t spacing_distance;
  int64_t grid_cell_size;

  const void *contexts;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t context_reserved;
  const uint32_t *metal_contexts;
  uint64_t metal_context_count;
  const uint64_t *context_polygon_offsets;
  uint64_t context_polygon_offset_count;
  const uint64_t *context_edge_offsets;
  uint64_t context_edge_offset_count;
  const void *cells;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t cell_reserved;
  const void *polygons;
  uint64_t polygon_count;
  uint32_t polygon_record_bytes;
  uint32_t polygon_reserved;
  const void *edges;
  uint64_t edge_count;
  uint32_t edge_record_bytes;
  uint32_t edge_reserved;

  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  uint64_t max_contexts;
  uint64_t max_grid_cells;
  uint64_t max_memberships;
  uint64_t max_pair_work;
  uint64_t max_flat_edges;
  uint64_t max_flat_polygons;
  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

struct klayout_cuda_spatial_m1_width_space_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t device_flags;
  uint32_t reserved0;
  int64_t width_distance;
  int64_t spacing_distance;
  int64_t grid_cell_size;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  uint8_t scene_digest[32];

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

  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t edge_expand_ns;
  uint64_t grid_count_ns;
  uint64_t grid_build_ns;
  uint64_t pair_count_ns;
  uint64_t query_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_m1_width_space_empty_v1_func) (
  const struct klayout_cuda_spatial_m1_width_space_request_v1 *,
  struct klayout_cuda_spatial_m1_width_space_result_v1 *);
typedef int (*klayout_cuda_spatial_run_m2_width_space_empty_v1_func) (
  const struct klayout_cuda_spatial_m1_width_space_request_v1 *,
  struct klayout_cuda_spatial_m1_width_space_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m1_width_space_empty_v1 (
  const struct klayout_cuda_spatial_m1_width_space_request_v1 *request,
  struct klayout_cuda_spatial_m1_width_space_result_v1 *result);

/*
 * Independent optional M2 capability and entry point.  Hosts must require
 * this symbol before lowering an M2 scene: an ABI-v1 backend that exports
 * only the legacy M1 symbol is M1-only even though the request POD is shared.
 */
KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m2_width_space_empty_v1 (
  const struct klayout_cuda_spatial_m1_width_space_request_v1 *request,
  struct klayout_cuda_spatial_m1_width_space_result_v1 *result);

/*
 * Optional exact raw-M2 Manhattan-union boundary.
 *
 * This is an additive capability.  It deliberately does not reuse the
 * METAL2.1/.2 request: that request asserts an already-merged input, while
 * this entry point consumes the compact raw hierarchy before KLayout calls
 * merged_deep_layer().  The caller-owned arrays use the same pointer-free
 * record layouts as CudaM1WidthSpaceScene.  A backend expands the hierarchy
 * and orthogonal contours, unions their material exactly, and returns a
 * canonical directed boundary.
 *
 * Result segments are backend-owned until the dedicated release function is
 * called.  A host may consume them only after validating the complete proof
 * echo, capacities, canonical order, maximal collinear intervals, and digest.
 * Missing symbols, a non-COMPLETE disposition, malformed topology, or any
 * validation failure selects the untouched CPU rule block.
 */
enum klayout_cuda_spatial_m2_union_opcode
{
  KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY = 1,
  /*
   * Additive clean-only transaction: return the same exact boundary and also
   * certify the fixed FreePDK45 M2.5-.9 suffix from the device-resident union
   * strips.  A backend which knows only opcode 1 must reject this opcode; the
   * host then executes the complete historical CPU rule block.
   */
  KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY = 2
};

enum klayout_cuda_spatial_m2_suffix_rule_mask
{
  KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_M2_5_EMPTY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_M2_6_EMPTY = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_M2_7_EMPTY = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_M2_8_EMPTY = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_M2_9_EMPTY = 1u << 4
};

#define KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY \
  ((1u << 5) - 1u)

enum klayout_cuda_spatial_m2_union_option_flag
{
  KLAYOUT_CUDA_SPATIAL_M2_UNION_RAW_HIERARCHY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_SAME_STORE_LAYOUT_TOP_LAYER = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_NO_BREAKOUT = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_ORTHOGONAL_UNIT_TRANSFORMS = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_NO_PROPERTIES = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_CLOCKWISE_MANHATTAN_CONTOURS = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_EXACT_INTEGER_SET_UNION = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_CANONICAL_DIRECTED_BOUNDARY = 1u << 7
};

#define KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS \
  ((1u << 8) - 1u)

enum klayout_cuda_spatial_m2_union_disposition
{
  KLAYOUT_CUDA_SPATIAL_M2_UNION_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN = 1
};

enum klayout_cuda_spatial_m2_union_axis
{
  KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL = 0,
  KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL = 1
};

/*
 * A horizontal segment has y=fixed and x in [lo, hi); a vertical segment has
 * x=fixed and y in [lo, hi).  side is the outward-normal sign along the
 * fixed/perpendicular axis (-1 or +1).  Canonical order is
 * (axis, side, fixed, lo, hi), with touching or overlapping intervals on one
 * line already coalesced.
 */
struct klayout_cuda_spatial_m2_union_segment_v1
{
  int64_t fixed;
  int64_t lo;
  int64_t hi;
  int32_t side;
  uint32_t axis;
};

struct klayout_cuda_spatial_m2_union_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  int32_t device;

  const void *contexts;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t context_reserved;
  const uint32_t *metal_contexts;
  uint64_t metal_context_count;
  const uint64_t *context_polygon_offsets;
  uint64_t context_polygon_offset_count;
  const uint64_t *context_edge_offsets;
  uint64_t context_edge_offset_count;
  const void *cells;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t cell_reserved;
  const void *polygons;
  uint64_t polygon_count;
  uint32_t polygon_record_bytes;
  uint32_t polygon_reserved;
  const void *edges;
  uint64_t edge_count;
  uint32_t edge_record_bytes;
  uint32_t edge_reserved;

  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  uint64_t max_contexts;
  uint64_t max_rectangles;
  uint64_t max_x_slabs;
  uint64_t max_memberships;
  uint64_t max_events;
  uint64_t max_raw_segments;
  uint64_t max_segments;
  uint32_t max_slabs_per_rectangle;
  uint32_t reserved0;
  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

struct klayout_cuda_spatial_m2_union_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t device_flags;
  uint32_t segment_record_bytes;
  uint8_t scene_digest[32];

  const struct klayout_cuda_spatial_m2_union_segment_v1 *segments;
  uint64_t segment_count;
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

  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t rectangle_expand_ns;
  uint64_t x_membership_ns;
  uint64_t strip_scan_ns;
  uint64_t boundary_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  /*
   * These 16 bytes replace the ABI-v1 reserved tail without changing the
   * result size.  They must all be zero for opcode 1 and for every non-OK
   * result.  Opcode 2 is consumable only when certified_empty_mask is exactly
   * KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY and certificate_reserved is zero.
   */
  uint32_t certified_empty_mask;
  uint32_t certificate_reserved;
  uint64_t suffix_total_ns;
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_m2_union_boundary_v1_func) (
  const struct klayout_cuda_spatial_m2_union_request_v1 *,
  struct klayout_cuda_spatial_m2_union_result_v1 *);
typedef void (*klayout_cuda_spatial_release_m2_union_boundary_v1_func) (
  struct klayout_cuda_spatial_m2_union_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m2_union_boundary_v1 (
  const struct klayout_cuda_spatial_m2_union_request_v1 *request,
  struct klayout_cuda_spatial_m2_union_result_v1 *result);
KLAYOUT_CUDA_SPATIAL_EXPORT void
klayout_cuda_spatial_release_m2_union_boundary_v1 (
  struct klayout_cuda_spatial_m2_union_result_v1 *result);

/*
 * Optional exact raw-M1 resident-morphology certificate.
 *
 * This is deliberately a distinct, additive transaction rather than another
 * opcode on the raw-M2 boundary ABI.  The input hierarchy describes complete
 * raw FreePDK45 METAL1 (11/0) contours and is bound to the KM1RAW01 digest
 * domain.  The backend constructs the exact Manhattan union, retains its
 * canonical strip representation on the selected device and certifies the
 * fixed M1.5-.9 suffix:
 *
 *   F90  = size(+90, size(-89, union(M1)))
 *   M1.5 = F90.space(180, projection, length > 600)
 *   F270 = size(-269, F90)
 *   M1.6-.9 are the four directional F270 extents.
 *
 * COMPLETE is consumable only when every requested rule bit is certified,
 * every request/census/capacity echo matches, the exact F90 space predicate
 * has zero violations and uncertainty, F270 is empty, and device_flags is
 * zero.  No geometry pointer crosses this ABI and no full union/F90 boundary
 * is returned: the result is a scalar proof/telemetry record.  The current
 * exact F90 space predicate does copy its explicitly capped >=600-DBU segment
 * subset to the host; f90_long_segment_count exposes that census and d2h_ns
 * below remains the full-boundary/result-geometry D2H time (therefore zero).
 * Missing capability, a positive rule, bounded-capacity exhaustion, malformed
 * input, or any proof mismatch retains the complete historical CPU
 * transaction.
 */
enum klayout_cuda_spatial_m1_resident_morphology_opcode
{
  KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M15_9_EMPTY = 1,
  /*
   * Exact raw-M1 union followed by the fixed FreePDK45 M1.1/M1.2
   * width/space empty certificate.  The backend scans the canonical strip
   * representation in both coordinate orientations, so this opcode does not
   * require or publish a materialized union boundary.  Result union censuses
   * are the peak of the two independently capacity-bounded orientations;
   * charged phase times cover both passes (and morphology is a named subset
   * of strip-scan time rather than an additive component).
   */
  KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M11_2_EMPTY = 2
};

enum klayout_cuda_spatial_m1_resident_morphology_rule
{
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_5_EMPTY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_6_EMPTY = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_7_EMPTY = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_8_EMPTY = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_9_EMPTY = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_1_EMPTY = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_2_EMPTY = 1u << 6
};

#define KLAYOUT_CUDA_SPATIAL_M1_MORPH_ALL_EMPTY ((1u << 5) - 1u)
#define KLAYOUT_CUDA_SPATIAL_M1_BASE_ALL_EMPTY \
  (KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_1_EMPTY | \
   KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_2_EMPTY)

enum klayout_cuda_spatial_m1_resident_morphology_option_flag
{
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_RAW_HIERARCHY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_SAME_STORE_LAYOUT_TOP_LAYER = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_NO_BREAKOUT = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_ORTHOGONAL_UNIT_TRANSFORMS = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_NO_PROPERTIES = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_CLOCKWISE_MANHATTAN_CONTOURS = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_EXACT_INTEGER_SET_UNION = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_DEVICE_RESIDENT_F90_F270 = 1u << 7
};

#define KLAYOUT_CUDA_SPATIAL_M1_MORPH_QUALIFIED_OPTIONS \
  ((1u << 8) - 1u)

enum klayout_cuda_spatial_m1_resident_morphology_disposition
{
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_NOT_EMPTY = 1,
  KLAYOUT_CUDA_SPATIAL_M1_MORPH_UNCERTAIN = 2
};

/*
 * The compact hierarchy records use the pointer-free layouts already shared
 * by the raw-M2 adapter.  Byte strides are nevertheless carried and checked
 * independently so neither endpoint relies on C++ cross-type aliasing.
 */
struct klayout_cuda_spatial_m1_resident_morphology_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  int32_t device;
  uint32_t requested_mask;
  uint32_t reserved0;

  const void *contexts;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t context_reserved;
  const uint32_t *metal_contexts;
  uint64_t metal_context_count;
  const uint64_t *context_polygon_offsets;
  uint64_t context_polygon_offset_count;
  const uint64_t *context_edge_offsets;
  uint64_t context_edge_offset_count;
  const void *cells;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t cell_reserved;
  const void *polygons;
  uint64_t polygon_count;
  uint32_t polygon_record_bytes;
  uint32_t polygon_reserved;
  const void *edges;
  uint64_t edge_count;
  uint32_t edge_record_bytes;
  uint32_t edge_reserved;

  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  uint64_t max_contexts;
  uint64_t max_rectangles;
  uint64_t max_x_slabs;
  uint64_t max_union_memberships;
  uint64_t max_union_events;
  uint64_t max_union_raw_segments;
  uint64_t max_union_segments;
  uint32_t max_slabs_per_rectangle;
  uint32_t union_reserved;

  uint64_t max_morph_output_slabs;
  uint64_t max_morph_output_intervals;
  uint64_t max_morph_raw_boundary_segments;
  uint64_t max_morph_boundary_segments;
  /* Legacy engine limit, applied independently to every morphology pass. */
  uint64_t max_morph_source_visits_per_pass;
  uint64_t max_morph_source_visits_per_band;
  uint64_t max_morph_long_segments;
  uint32_t max_morph_active_slabs;
  uint32_t morph_reserved;

  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

struct klayout_cuda_spatial_m1_resident_morphology_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t requested_mask;
  uint32_t certified_empty_mask;
  uint32_t device_flags;
  uint32_t reserved0;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  uint8_t scene_digest[32];

  uint64_t context_count;
  uint64_t metal_context_count;
  uint64_t cell_count;
  uint64_t polygon_count;
  uint64_t edge_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;

  uint64_t max_contexts;
  uint64_t max_rectangles;
  uint64_t max_x_slabs;
  uint64_t max_union_memberships;
  uint64_t max_union_events;
  uint64_t max_union_raw_segments;
  uint64_t max_union_segments;
  uint32_t max_slabs_per_rectangle;
  uint32_t union_reserved;
  uint64_t max_morph_output_slabs;
  uint64_t max_morph_output_intervals;
  uint64_t max_morph_raw_boundary_segments;
  uint64_t max_morph_boundary_segments;
  uint64_t max_morph_source_visits_per_pass;
  uint64_t max_morph_source_visits_per_band;
  uint64_t max_morph_long_segments;
  uint32_t max_morph_active_slabs;
  uint32_t morph_reserved;

  uint64_t rectangle_count;
  uint64_t x_slab_count;
  uint64_t union_membership_count;
  uint64_t union_event_count;
  uint64_t strip_interval_count;
  uint64_t erode89_output_interval_count;
  uint64_t erode89_source_visit_count;
  uint64_t dilate90_output_interval_count;
  uint64_t dilate90_source_visit_count;
  uint64_t boundary_source_visit_count;
  uint64_t erode269_source_visit_count;
  uint64_t f90_boundary_segment_count;
  uint64_t f90_long_segment_count;
  uint64_t f90_space_pair_count;
  uint64_t f90_space_violation_count;
  uint64_t f90_space_uncertain_count;
  uint64_t f270_eroded_interval_count;
  uint64_t union_device_total_bytes;
  uint64_t union_device_free_begin_bytes;
  uint64_t union_device_free_low_bytes;
  uint64_t morph_device_total_bytes;
  uint64_t morph_device_free_begin_bytes;
  uint64_t morph_device_free_low_bytes;

  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t rectangle_expand_ns;
  uint64_t x_membership_ns;
  uint64_t strip_scan_ns;
  uint64_t morphology_ns;
  /* Full-boundary/result-geometry D2H only; see the capped predicate above. */
  uint64_t d2h_ns;
  uint64_t total_ns;
  char message[192];
};

typedef int
(*klayout_cuda_spatial_run_m1_resident_morphology_empty_v1_func) (
  const struct klayout_cuda_spatial_m1_resident_morphology_request_v1 *,
  struct klayout_cuda_spatial_m1_resident_morphology_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m1_resident_morphology_empty_v1 (
  const struct klayout_cuda_spatial_m1_resident_morphology_request_v1 *request,
  struct klayout_cuda_spatial_m1_resident_morphology_result_v1 *result);

/*
 * Optional atomic FreePDK45 POLY.3/POLY.4 terminal-empty certificate.
 *
 * Format 1 supplies three exact merged hierarchical domains: POLY, ACTIVE
 * and their already-derived GATE intersection.  Format 2 instead supplies
 * exact raw physical POLY/ACTIVE rectangle covers and requires the backend
 * to derive their complete positive-area intersection.  The backend expands
 * the compact box templates, constructs complete bounded candidate windows
 * for the fixed 110/140-DBU projection-enclosure profiles and applies the
 * exact zero-area-terminal certificate to every GATE tile.
 *
 * No partial result is consumable.  The historical two-rule CPU transaction
 * may be skipped only when disposition is COMPLETE, both rule bits are
 * certified, every GATE is atomically terminal-empty and every proof echo
 * matches.  A positive-area profile, conservative miss, malformed record,
 * capacity exhaustion or CUDA error requires both original CPU expressions.
 */
enum klayout_cuda_spatial_poly34_opcode
{
  KLAYOUT_CUDA_SPATIAL_POLY34_TERMINAL_EMPTY = 1,
  /*
   * Format 2 supplies exact raw physical POLY/ACTIVE rectangle covers and
   * derives GATE = POLY & ACTIVE on the selected device.  It remains an
   * empty-only transaction: no derived geometry is returned to the caller.
   */
  KLAYOUT_CUDA_SPATIAL_POLY34_RAW_TERMINAL_EMPTY = 2
};

enum klayout_cuda_spatial_poly34_rule
{
  KLAYOUT_CUDA_SPATIAL_POLY3_RULE = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_POLY4_RULE = 1u << 1
};

#define KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES ((1u << 2) - 1u)

enum klayout_cuda_spatial_poly34_domain
{
  KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN = 0,
  KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN = 1,
  KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN = 2,
  KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT = 3
};

enum klayout_cuda_spatial_poly34_option_flag
{
  KLAYOUT_CUDA_SPATIAL_POLY34_EXACT_MERGED_DOMAINS = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_POLY34_SAME_STORE_LAYOUT_TOP = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_POLY34_NO_BREAKOUT = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_POLY34_PROJECTION = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_POLY34_IGNORE_PROPERTIES = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_POLY34_INCLUDE_TOUCHING = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_POLY34_ZERO_AREA_TERMINAL = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_POLY34_RECTANGULAR_GATE = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_POLY34_EXACT_PRIMARY_BOX_UNIONS = 1u << 8,
  KLAYOUT_CUDA_SPATIAL_POLY34_ORDERED_OUTPUTS = 1u << 9,
  KLAYOUT_CUDA_SPATIAL_POLY34_RAW_PHYSICAL_DOMAINS = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_POLY34_DERIVE_GATE_INTERSECTION = 1u << 11,
  KLAYOUT_CUDA_SPATIAL_POLY34_EXACT_RECTANGLE_COVERS = 1u << 12
};

#define KLAYOUT_CUDA_SPATIAL_POLY34_QUALIFIED_OPTIONS ((1u << 10) - 1u)

/*
 * Unlike format 1, the raw transaction does not assert pre-merged operands
 * or a pre-materialized rectangular GATE.  Its exact rectangle covers retain
 * the same integer set as each physical input, and the backend derives the
 * complete intersection before applying the unchanged empty certificate.
 */
#define KLAYOUT_CUDA_SPATIAL_POLY34_RAW_QUALIFIED_OPTIONS ( \
  KLAYOUT_CUDA_SPATIAL_POLY34_SAME_STORE_LAYOUT_TOP | \
  KLAYOUT_CUDA_SPATIAL_POLY34_NO_BREAKOUT | \
  KLAYOUT_CUDA_SPATIAL_POLY34_PROJECTION | \
  KLAYOUT_CUDA_SPATIAL_POLY34_IGNORE_PROPERTIES | \
  KLAYOUT_CUDA_SPATIAL_POLY34_INCLUDE_TOUCHING | \
  KLAYOUT_CUDA_SPATIAL_POLY34_ZERO_AREA_TERMINAL | \
  KLAYOUT_CUDA_SPATIAL_POLY34_ORDERED_OUTPUTS | \
  KLAYOUT_CUDA_SPATIAL_POLY34_RAW_PHYSICAL_DOMAINS | \
  KLAYOUT_CUDA_SPATIAL_POLY34_DERIVE_GATE_INTERSECTION | \
  KLAYOUT_CUDA_SPATIAL_POLY34_EXACT_RECTANGLE_COVERS)

#define KLAYOUT_CUDA_SPATIAL_POLY34_NO_GATE_LAYER 0xffffffffu

enum klayout_cuda_spatial_poly34_disposition
{
  KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY = 1,
  KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN = 2
};

struct klayout_cuda_spatial_poly34_context_v1
{
  int64_t tx;
  int64_t ty;
  uint32_t cell_id;
  uint32_t transform_code;
};

struct klayout_cuda_spatial_poly34_box_v1
{
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
};

struct klayout_cuda_spatial_poly34_domain_span_v1
{
  uint64_t box_begin;
  uint32_t box_count;
  uint32_t reserved0;
};

struct klayout_cuda_spatial_poly34_cell_v1
{
  uint64_t source_cell_index;
  struct klayout_cuda_spatial_poly34_domain_span_v1
    domains[KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT];
};

struct klayout_cuda_spatial_poly34_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t requested_mask;
  int32_t device;
  uint32_t reserved0;

  int64_t poly3_distance;
  int64_t poly4_distance;
  int64_t grid_cell_size;
  uint64_t store_identity;
  uint64_t layout_identity;
  uint64_t top_cell_identity;
  uint32_t poly_layer_id;
  uint32_t active_layer_id;
  uint32_t gate_layer_id;
  uint32_t identity_reserved;

  const void *contexts;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t context_reserved;
  const uint32_t *poly_contexts;
  uint64_t poly_context_count;
  const uint64_t *poly_offsets;
  uint64_t poly_offset_count;
  const uint32_t *active_contexts;
  uint64_t active_context_count;
  const uint64_t *active_offsets;
  uint64_t active_offset_count;
  const uint32_t *gate_contexts;
  uint64_t gate_context_count;
  const uint64_t *gate_offsets;
  uint64_t gate_offset_count;
  const void *cells;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t cell_reserved;
  const void *boxes;
  uint64_t box_count;
  uint32_t box_record_bytes;
  uint32_t box_reserved;

  uint64_t flat_poly_box_count;
  uint64_t flat_active_box_count;
  uint64_t flat_gate_box_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  uint64_t max_contexts;
  uint64_t max_flat_boxes;
  uint64_t max_grid_cells;
  uint64_t max_poly_memberships;
  uint64_t max_active_memberships;
  uint64_t max_query_visits;
  uint64_t max_candidate_work;
  uint32_t max_candidates_per_gate;
  uint32_t capacity_reserved;
  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

struct klayout_cuda_spatial_poly34_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t requested_mask;
  uint32_t certified_empty_mask;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  int32_t device;
  uint32_t device_flags;
  uint32_t reserved0;
  uint32_t reserved1;

  int64_t poly3_distance;
  int64_t poly4_distance;
  int64_t grid_cell_size;
  uint64_t store_identity;
  uint64_t layout_identity;
  uint64_t top_cell_identity;
  uint32_t poly_layer_id;
  uint32_t active_layer_id;
  uint32_t gate_layer_id;
  uint32_t identity_reserved;
  uint8_t scene_digest[32];

  uint64_t context_count;
  uint64_t poly_context_count;
  uint64_t active_context_count;
  uint64_t gate_context_count;
  uint64_t cell_count;
  uint64_t box_count;
  uint64_t flat_poly_box_count;
  uint64_t flat_active_box_count;
  uint64_t flat_gate_box_count;
  uint64_t expanded_poly_box_count;
  uint64_t expanded_active_box_count;
  uint64_t expanded_gate_box_count;
  uint64_t grid_cell_count;
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
  uint64_t maximum_poly_candidates;
  uint64_t maximum_active_candidates;

  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t expand_ns;
  uint64_t poly_grid_ns;
  uint64_t poly_query_ns;
  uint64_t active_grid_ns;
  uint64_t active_query_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_poly34_empty_v1_func) (
  const struct klayout_cuda_spatial_poly34_request_v1 *,
  struct klayout_cuda_spatial_poly34_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_poly34_empty_v1 (
  const struct klayout_cuda_spatial_poly34_request_v1 *request,
  struct klayout_cuda_spatial_poly34_result_v1 *result);

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

/*
 * Optional fused raw-ACTIVE union / CONTACT.4 empty certificate.
 *
 * This additive entry point consumes two independently serialized raw
 * Manhattan hierarchies.  The backend expands and unions ACTIVE exactly,
 * retains its canonical directed boundary on the selected device, expands
 * CONTACT on that same device and applies the complete CONTACT.4 relation
 * without publishing either intermediate geometry to the host.
 *
 * The two digest domains are part of the proof contract.  Each digest is
 * SHA-256 over the fixed eight-byte domain below, followed by format_version,
 * dbu_per_micron, root_cell and the scene reserved word in little-endian form,
 * then the unchanged canonical raw-hierarchy geometry payload.  role, layer
 * and datatype bind the descriptor at this API boundary but are deliberately
 * not inserted into that existing host-serializer digest payload.
 *
 * COMPLETE is the sole consumable outcome.  A raw hit is diagnostic only and
 * requires the pristine CPU rule, as does every malformed echo, uncertainty,
 * bounded decline, loader failure or CUDA error.
 */
#define KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_DIGEST_DOMAIN "KARAW001"
#define KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_DIGEST_DOMAIN "KCRAW001"
#define KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES 8u

enum klayout_cuda_spatial_contact4_active_union_opcode
{
  KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_EMPTY = 1
};

enum klayout_cuda_spatial_contact4_active_union_role
{
  KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE = 1,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE = 2
};

enum klayout_cuda_spatial_contact4_active_union_option_flag
{
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_RAW_ACTIVE_HIERARCHY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_RAW_CONTACT_HIERARCHY = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_SAME_STORE_LAYOUT_TOP = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_NO_BREAKOUT = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_ORTHOGONAL_UNIT_TRANSFORMS = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_NO_PROPERTIES = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_CLOCKWISE_MANHATTAN_CONTOURS = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_EXACT_ACTIVE_INTEGER_SET_UNION = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_CANONICAL_ACTIVE_BOUNDARY = 1u << 8,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_CONTACT_INDEXED_SECONDARY = 1u << 9,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_OVERLAP_RELATION = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_DIFFERENT_POLYGONS = 1u << 11,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_EUCLIDIAN = 1u << 12,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_IGNORE_ANGLE_90 = 1u << 13,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_WHOLE_EDGES_FALSE = 1u << 14,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_PROJECTION_DEFAULTS = 1u << 15,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_SHIELDED = 1u << 16,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_NO_FILTERS_OR_NEGATIVE = 1u << 17,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_INCLUDE_TOUCHING = 1u << 18,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_UNION_DEFAULT_STREAM = 1u << 19
};

#define KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_QUALIFIED_OPTIONS \
  ((1u << 20) - 1u)

enum klayout_cuda_spatial_contact4_active_union_disposition
{
  KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_RAW_HITS = 1,
  KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_UNCERTAIN = 2
};

/*
 * Pointer-free records behind these byte-addressed arrays are the established
 * klayout_cuda_spatial_m1_width_space_{context,cell,polygon,edge}_v1 layouts.
 * Keeping a nested descriptor gives ACTIVE and CONTACT separate cell graphs,
 * context lists, flat censuses, bounds and digest domains without repacking
 * either owning host scene.
 */
struct klayout_cuda_spatial_contact4_active_union_scene_v1
{
  uint32_t struct_size;
  uint32_t role;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t layer;
  uint32_t datatype;
  uint32_t reserved0;

  const void *contexts;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t context_reserved;
  const uint32_t *layer_contexts;
  uint64_t layer_context_count;
  const uint64_t *context_polygon_offsets;
  uint64_t context_polygon_offset_count;
  const uint64_t *context_edge_offsets;
  uint64_t context_edge_offset_count;
  const void *cells;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t cell_reserved;
  const void *polygons;
  uint64_t polygon_count;
  uint32_t polygon_record_bytes;
  uint32_t polygon_reserved;
  const void *edges;
  uint64_t edge_count;
  uint32_t edge_record_bytes;
  uint32_t edge_reserved;

  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  uint8_t digest_domain[KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES];
  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

/*
 * Pointer-free scalar echo of one scene.  A COMPLETE result is accepted only
 * when every field exactly matches its request descriptor.
 */
struct klayout_cuda_spatial_contact4_active_union_scene_echo_v1
{
  uint32_t struct_size;
  uint32_t role;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t layer;
  uint32_t datatype;
  uint32_t reserved0;
  uint64_t context_count;
  uint64_t layer_context_count;
  uint64_t context_polygon_offset_count;
  uint64_t context_edge_offset_count;
  uint64_t cell_count;
  uint64_t polygon_count;
  uint64_t edge_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  uint8_t digest_domain[KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES];
  uint8_t scene_digest[32];
  uint64_t reserved1[2];
};

struct klayout_cuda_spatial_contact4_active_union_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  int32_t device;
  uint32_t reserved0;
  int64_t distance;
  int64_t grid_cell_size;

  struct klayout_cuda_spatial_contact4_active_union_scene_v1 active;
  struct klayout_cuda_spatial_contact4_active_union_scene_v1 contact;

  uint64_t max_contexts;
  uint64_t max_rectangles;
  uint64_t max_x_slabs;
  uint64_t max_union_memberships;
  uint64_t max_events;
  uint64_t max_raw_segments;
  uint64_t max_boundary_segments;
  uint32_t max_slabs_per_rectangle;
  uint32_t union_reserved;

  uint64_t max_contact_edges;
  uint64_t max_grid_cells;
  uint64_t max_contact_memberships;
  uint64_t max_boundary_cell_visits;
  uint64_t max_member_visits;
  uint64_t max_pair_work;
  uint32_t max_cells_per_contact_edge;
  uint32_t max_cells_per_boundary_edge;
  uint64_t reserved1[4];
};

struct klayout_cuda_spatial_contact4_active_union_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  int32_t device;
  uint32_t device_flags;
  uint32_t reserved0;
  int64_t distance;
  int64_t grid_cell_size;

  struct klayout_cuda_spatial_contact4_active_union_scene_echo_v1 active;
  struct klayout_cuda_spatial_contact4_active_union_scene_echo_v1 contact;

  uint64_t rectangle_count;
  uint64_t x_slab_count;
  uint64_t union_membership_count;
  uint64_t event_count;
  uint64_t strip_interval_count;
  uint64_t raw_segment_count;
  uint64_t boundary_segment_count;

  uint64_t contact_expanded_edge_count;
  uint64_t grid_cell_count;
  uint64_t contact_membership_count;
  uint64_t boundary_cell_visit_count;
  uint64_t member_visit_count;
  uint64_t candidate_pair_count;
  uint64_t raw_hit_count;
  uint64_t uncertain_count;

  uint64_t device_total_bytes;
  uint64_t union_free_begin_bytes;
  uint64_t union_free_low_bytes;
  uint64_t callback_free_begin_bytes;
  uint64_t callback_free_low_bytes;
  uint64_t post_scan_free_bytes;
  uint64_t callback_incremental_peak_bytes;

  uint64_t setup_ns;
  uint64_t active_h2d_ns;
  uint64_t active_expand_ns;
  uint64_t x_membership_ns;
  uint64_t strip_scan_ns;
  uint64_t boundary_ns;
  uint64_t contact_h2d_ns;
  uint64_t contact_expand_ns;
  uint64_t boundary_preflight_ns;
  uint64_t grid_count_ns;
  uint64_t grid_build_ns;
  uint64_t query_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  uint64_t reserved1[2];
  char message[192];
};

typedef int
(*klayout_cuda_spatial_run_contact4_active_union_empty_v1_func) (
  const struct klayout_cuda_spatial_contact4_active_union_request_v1 *,
  struct klayout_cuda_spatial_contact4_active_union_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_contact4_active_union_empty_v1 (
  const struct klayout_cuda_spatial_contact4_active_union_request_v1 *request,
  struct klayout_cuda_spatial_contact4_active_union_result_v1 *result);

/*
 * Optional exact raw-(NWELL union PWELL) / ACTIVE.3 empty certificate.
 *
 * This is intentionally a distinct additive ABI and symbol.  "wells" is one
 * deterministic KWRWL001 scene containing physical NWELL 3/0 followed by
 * PWELL 2/0 in every source cell.  The backend expands and unions that scene
 * exactly, retains its canonical boundary on-device, expands the complete raw
 * ACTIVE 1/0 hierarchy on the same device, and applies the exact ACTIVE.3
 * predicate with the WELL boundary as the first operand.  No geometry is
 * published to the host.
 *
 * COMPLETE is the sole consumable outcome.  Hits, uncertainty, capacity
 * declines, malformed echoes, loader failures, and CUDA errors all require
 * the unchanged CPU WELL union and ACTIVE.3 expression.
 */
#define KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_DIGEST_DOMAIN \
  "KWRWL001"
#define KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_DIGEST_DOMAIN \
  "KARAW001"
#define KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_DIGEST_DOMAIN_BYTES 8u

enum klayout_cuda_spatial_active3_well_union_opcode
{
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_EMPTY = 1
};

enum klayout_cuda_spatial_active3_well_union_role
{
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_ROLE = 1,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_ROLE = 2
};

enum klayout_cuda_spatial_active3_well_union_option_flag
{
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_RAW_WELLS_HIERARCHY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_RAW_ACTIVE_HIERARCHY = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_PHYSICAL_NWELL_PWELL = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_PHYSICAL_ACTIVE = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_SAME_STORE_LAYOUT_TOP = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_NO_BREAKOUT = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ORTHOGONAL_UNIT_TRANSFORMS =
    1u << 6,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_NO_PROPERTIES = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_CLOCKWISE_MANHATTAN_CONTOURS =
    1u << 8,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_EXACT_INTEGER_SET_UNION = 1u << 9,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_CANONICAL_WELL_BOUNDARY = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_INDEXED_SECONDARY = 1u << 11,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_OVERLAP_RELATION = 1u << 12,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_DIFFERENT_POLYGONS = 1u << 13,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_EUCLIDIAN = 1u << 14,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_IGNORE_ANGLE_90 = 1u << 15,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WHOLE_EDGES_FALSE = 1u << 16,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_PROJECTION_DEFAULTS = 1u << 17,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_SHIELDED = 1u << 18,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_NO_FILTERS_OR_NEGATIVE = 1u << 19,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_INCLUDE_TOUCHING = 1u << 20,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_DEFAULT_STREAM = 1u << 21
};

#define KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_QUALIFIED_OPTIONS \
  ((1u << 22) - 1u)

enum klayout_cuda_spatial_active3_well_union_disposition
{
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_RAW_HITS = 1,
  KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_UNCERTAIN = 2
};

/*
 * The pointer-bearing scene descriptor and pointer-free echo deliberately
 * reuse the already stable raw-Manhattan layouts.  The new role, physical
 * layer, digest domain, request, result, and entry point remain independent.
 */
typedef struct klayout_cuda_spatial_contact4_active_union_scene_v1
  klayout_cuda_spatial_active3_well_union_scene_v1;
typedef struct klayout_cuda_spatial_contact4_active_union_scene_echo_v1
  klayout_cuda_spatial_active3_well_union_scene_echo_v1;

struct klayout_cuda_spatial_active3_well_union_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  int32_t device;
  uint32_t reserved0;
  int64_t distance;
  int64_t grid_cell_size;
  uint32_t secondary_well_layer;
  uint32_t secondary_well_datatype;
  uint64_t layer_reserved;

  klayout_cuda_spatial_active3_well_union_scene_v1 wells;
  klayout_cuda_spatial_active3_well_union_scene_v1 active;

  uint64_t max_contexts;
  uint64_t max_rectangles;
  uint64_t max_x_slabs;
  uint64_t max_union_memberships;
  uint64_t max_events;
  uint64_t max_raw_segments;
  uint64_t max_boundary_segments;
  uint32_t max_slabs_per_rectangle;
  uint32_t union_reserved;

  uint64_t max_active_edges;
  uint64_t max_grid_cells;
  uint64_t max_active_memberships;
  uint64_t max_active_cell_visits;
  uint64_t max_member_visits;
  uint64_t max_pair_work;
  uint32_t max_cells_per_active_edge;
  uint32_t max_cells_per_well_edge;
  uint64_t reserved1[4];
};

struct klayout_cuda_spatial_active3_well_union_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  int32_t device;
  uint32_t device_flags;
  uint32_t reserved0;
  int64_t distance;
  int64_t grid_cell_size;
  uint32_t secondary_well_layer;
  uint32_t secondary_well_datatype;
  uint64_t layer_reserved;

  klayout_cuda_spatial_active3_well_union_scene_echo_v1 wells;
  klayout_cuda_spatial_active3_well_union_scene_echo_v1 active;

  uint64_t rectangle_count;
  uint64_t x_slab_count;
  uint64_t union_membership_count;
  uint64_t event_count;
  uint64_t strip_interval_count;
  uint64_t raw_segment_count;
  uint64_t boundary_segment_count;

  uint64_t active_expanded_edge_count;
  uint64_t grid_cell_count;
  uint64_t active_membership_count;
  uint64_t active_cell_visit_count;
  uint64_t member_visit_count;
  uint64_t candidate_pair_count;
  uint64_t raw_hit_count;
  uint64_t uncertain_count;

  uint64_t device_total_bytes;
  uint64_t union_free_begin_bytes;
  uint64_t union_free_low_bytes;
  uint64_t callback_free_begin_bytes;
  uint64_t callback_free_low_bytes;
  uint64_t post_scan_free_bytes;
  uint64_t callback_incremental_peak_bytes;

  uint64_t setup_ns;
  uint64_t wells_h2d_ns;
  uint64_t wells_expand_ns;
  uint64_t x_membership_ns;
  uint64_t strip_scan_ns;
  uint64_t boundary_ns;
  uint64_t active_h2d_ns;
  uint64_t active_expand_ns;
  uint64_t active_preflight_ns;
  uint64_t grid_count_ns;
  uint64_t grid_build_ns;
  uint64_t query_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  uint64_t reserved1[2];
  char message[192];
};

typedef int
(*klayout_cuda_spatial_run_active3_well_union_empty_v1_func) (
  const struct klayout_cuda_spatial_active3_well_union_request_v1 *,
  struct klayout_cuda_spatial_active3_well_union_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_active3_well_union_empty_v1 (
  const struct klayout_cuda_spatial_active3_well_union_request_v1 *request,
  struct klayout_cuda_spatial_active3_well_union_result_v1 *result);

/*
 * Optional exact raw FreePDK45 IMPLANT.1-.5 empty certificate.
 *
 * This additive transaction receives four independent, digest-bound
 * raw-Manhattan hierarchies: physical NPLUS 4/0, physical PPLUS 5/0, the
 * already-derived exact GATE integer set, and physical CONTACT 10/0.  GATE
 * deliberately carries the sentinel layer/datatype below: it is a qualified
 * derived KLayout layer, not a physical GDS layer.
 *
 * The backend expands all four scenes onto one device, forms and retains the
 * exact NPLUS-or-PPLUS IMPLANT union, and executes five ordered resident
 * phases matching the historical deck:
 *
 *   IMPLANT.1  implant.separation(gate, 140 DBU, projection), zero area only
 *   IMPLANT.2  implant.separation(contact, 50 DBU, projection), zero area only
 *   IMPLANT.3  implant.width(90 DBU, euclidian)
 *   IMPLANT.4  implant.space(90 DBU, euclidian)
 *   IMPLANT.5  nplus.and(pplus)
 *
 * Adjacent kernels may be fused internally for bandwidth, but the result
 * preserves distinct counts and timings for every semantic phase.  No
 * intermediate geometry crosses back to the host.  COMPLETE is consumable
 * only when certified_empty_mask and clean_mask both equal ALL_RULES and
 * every scalar, capacity, scene descriptor and digest echo matches exactly.
 * Every other outcome requires all five literal CPU expressions.
 */
#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_DIGEST_DOMAIN "KNPLS001"
#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_DIGEST_DOMAIN "KPPLS001"
#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_DIGEST_DOMAIN "KGATE001"
#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_DIGEST_DOMAIN "KCRAW001"
#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_DIGEST_DOMAIN_BYTES 8u
#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_LAYER 0xffffffffu
#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_DATATYPE 0xffffffffu

enum klayout_cuda_spatial_implant15_opcode
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_RESIDENT_EMPTY = 1
};

enum klayout_cuda_spatial_implant15_rule
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_1 = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_2 = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_3 = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_4 = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_5 = 1u << 4
};

#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES ((1u << 5) - 1u)

enum klayout_cuda_spatial_implant15_role
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_ROLE = 1,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_ROLE = 2,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_ROLE = 3,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_ROLE = 4
};

enum klayout_cuda_spatial_implant15_option_flag
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_NPLUS_HIERARCHY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_PPLUS_HIERARCHY = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_GATE_HIERARCHY = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_CONTACT_HIERARCHY = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_SAME_STORE_LAYOUT_TOP = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_NO_BREAKOUT = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_ORTHOGONAL_UNIT_TRANSFORMS = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_NO_PROPERTIES = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_CLOCKWISE_MANHATTAN_CONTOURS = 1u << 8,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_EXACT_IMPLANT_INTEGER_SET_UNION = 1u << 9,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_CANONICAL_IMPLANT_BOUNDARY = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE1_PROJECTION = 1u << 11,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE2_PROJECTION = 1u << 12,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE3_EUCLIDIAN_WIDTH = 1u << 13,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE4_EUCLIDIAN_SPACE = 1u << 14,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE5_EXACT_INTERSECTION = 1u << 15,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_DIFFERENT_POLYGONS = 1u << 16,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_IGNORE_ANGLE_90 = 1u << 17,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_WHOLE_EDGES_FALSE = 1u << 18,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_PROJECTION_DEFAULTS = 1u << 19,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_SHIELDED = 1u << 20,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_NO_FILTERS_OR_NEGATIVE = 1u << 21,
  /*
   * IMPLANT.5 is an integer-set AND: boundary-only touching has zero area
   * and is clean.  This bit binds the backend to strict positive-area
   * NPLUS/PPLUS overlap.
   */
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_POSITIVE_AREA_OVERLAP = 1u << 22,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_ORDERED_OUTPUTS = 1u << 23,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_DEFAULT_STREAM = 1u << 24
};

#define KLAYOUT_CUDA_SPATIAL_IMPLANT15_QUALIFIED_OPTIONS ((1u << 25) - 1u)

enum klayout_cuda_spatial_implant15_disposition
{
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_HITS = 1,
  KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN = 2
};

/*
 * These aliases intentionally preserve the established raw-Manhattan pointer
 * descriptor and pointer-free echo layouts.  Role, layer/datatype sentinel
 * and digest domain make every operand non-interchangeable.
 */
typedef struct klayout_cuda_spatial_contact4_active_union_scene_v1
  klayout_cuda_spatial_implant15_scene_v1;
typedef struct klayout_cuda_spatial_contact4_active_union_scene_echo_v1
  klayout_cuda_spatial_implant15_scene_echo_v1;

/*
 * All allocation/work ceilings live in one pointer-free POD so the result can
 * echo the complete bounded-computation contract byte for byte.
 */
struct klayout_cuda_spatial_implant15_capacity_v1
{
  uint32_t struct_size;
  uint32_t max_slabs_per_rectangle;
  uint32_t max_cells_per_secondary_edge;
  uint32_t max_cells_per_boundary_edge;
  uint64_t max_contexts;
  uint64_t max_rectangles;
  uint64_t max_x_slabs;
  uint64_t max_union_memberships;
  uint64_t max_events;
  uint64_t max_raw_segments;
  uint64_t max_boundary_segments;
  uint64_t max_gate_edges;
  uint64_t max_contact_edges;
  uint64_t max_grid_cells;
  uint64_t max_secondary_memberships;
  uint64_t max_gate_boundary_cell_visits;
  uint64_t max_contact_boundary_cell_visits;
  uint64_t max_member_visits;
  uint64_t max_pair_work;
  uint64_t max_morphology_work;
  uint64_t max_overlap_work;
  uint64_t reserved[4];
};

struct klayout_cuda_spatial_implant15_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t requested_mask;
  int32_t device;
  uint32_t reserved0;
  uint32_t reserved1;
  int64_t implant1_distance;
  int64_t implant2_distance;
  int64_t implant3_distance;
  int64_t implant4_distance;
  int64_t grid_cell_size;

  klayout_cuda_spatial_implant15_scene_v1 nplus;
  klayout_cuda_spatial_implant15_scene_v1 pplus;
  klayout_cuda_spatial_implant15_scene_v1 gate;
  klayout_cuda_spatial_implant15_scene_v1 contact;
  struct klayout_cuda_spatial_implant15_capacity_v1 capacity;
  uint64_t reserved2[4];
};

struct klayout_cuda_spatial_implant15_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t requested_mask;
  uint32_t certified_empty_mask;
  uint32_t clean_mask;
  int32_t device;
  uint32_t device_flags;
  uint32_t reserved0;
  uint32_t reserved1;
  int64_t implant1_distance;
  int64_t implant2_distance;
  int64_t implant3_distance;
  int64_t implant4_distance;
  int64_t grid_cell_size;

  klayout_cuda_spatial_implant15_scene_echo_v1 nplus;
  klayout_cuda_spatial_implant15_scene_echo_v1 pplus;
  klayout_cuda_spatial_implant15_scene_echo_v1 gate;
  klayout_cuda_spatial_implant15_scene_echo_v1 contact;
  struct klayout_cuda_spatial_implant15_capacity_v1 capacity;

  uint64_t nplus_rectangle_count;
  uint64_t pplus_rectangle_count;
  uint64_t implant_rectangle_count;
  uint64_t x_slab_count;
  uint64_t union_membership_count;
  uint64_t event_count;
  uint64_t strip_interval_count;
  uint64_t raw_segment_count;
  uint64_t boundary_segment_count;
  uint64_t gate_expanded_edge_count;
  uint64_t contact_expanded_edge_count;

  uint64_t implant1_grid_cell_count;
  uint64_t implant1_secondary_membership_count;
  uint64_t implant1_boundary_cell_visit_count;
  uint64_t implant1_member_visit_count;
  uint64_t implant1_candidate_count;
  uint64_t implant1_hit_count;
  uint64_t implant1_uncertain_count;
  uint64_t implant2_grid_cell_count;
  uint64_t implant2_secondary_membership_count;
  uint64_t implant2_boundary_cell_visit_count;
  uint64_t implant2_member_visit_count;
  uint64_t implant2_candidate_count;
  uint64_t implant2_hit_count;
  uint64_t implant2_uncertain_count;
  uint64_t implant3_candidate_count;
  uint64_t implant3_hit_count;
  uint64_t implant3_uncertain_count;
  uint64_t implant4_candidate_count;
  uint64_t implant4_hit_count;
  uint64_t implant4_uncertain_count;
  uint64_t implant5_membership_count;
  uint64_t implant5_candidate_count;
  uint64_t implant5_hit_count;
  uint64_t implant5_uncertain_count;

  uint64_t setup_ns;
  uint64_t nplus_h2d_ns;
  uint64_t nplus_expand_ns;
  uint64_t pplus_h2d_ns;
  uint64_t pplus_expand_ns;
  uint64_t implant_union_ns;
  uint64_t implant_boundary_ns;
  uint64_t gate_h2d_ns;
  uint64_t gate_expand_ns;
  uint64_t implant1_ns;
  uint64_t contact_h2d_ns;
  uint64_t contact_expand_ns;
  uint64_t implant2_ns;
  uint64_t implant3_ns;
  uint64_t implant4_ns;
  uint64_t implant5_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  uint64_t reserved2[4];
  char message[192];
};

typedef int (*klayout_cuda_spatial_run_implant15_raw_empty_v1_func) (
  const struct klayout_cuda_spatial_implant15_request_v1 *,
  struct klayout_cuda_spatial_implant15_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_implant15_raw_empty_v1 (
  const struct klayout_cuda_spatial_implant15_request_v1 *request,
  struct klayout_cuda_spatial_implant15_result_v1 *result);

/*
 * Optional exact FreePDK45 ANTENNA.M1-through-M4 empty certificate.
 *
 * The caller supplies twelve raw physical domains but owns hierarchy only
 * once.  Domain descriptors contain source-cell-local geometry ranges and
 * refer to the request's one shared source-cell/context/parent stream.  No
 * flattened occurrence geometry or repeated context array crosses this ABI.
 *
 * The backend executes four ordered, resident metal stages with the fixed
 * 300:1 antenna ratio.  COMPLETE is consumable only when every input identity,
 * count, capacity and digest is echoed exactly, all four requested stages are
 * certified clean, and every hit, uncertainty, device and fallback flag is
 * zero.  Every other outcome requires the complete literal CPU antenna path.
 */
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT 12u
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_DOMAINS 0x0fffu
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT 4u
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_STAGES 0x0fu
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_NUMERATOR 300u
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_DENOMINATOR 1u
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DIGEST_DOMAIN_BYTES 8u

#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_POLY_DIGEST_DOMAIN "KPOLY001"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ACTIVE_DIGEST_DOMAIN "KARAW001"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NPLUS_DIGEST_DOMAIN "KNPLS001"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NWELL_DIGEST_DOMAIN "KNWEL001"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_CONTACT_DIGEST_DOMAIN "KCRAW001"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M1_DIGEST_DOMAIN "KM1RAW01"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA1_DIGEST_DOMAIN "KV1RAW01"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M2_DIGEST_DOMAIN "KM2RAW01"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA2_DIGEST_DOMAIN "KV2RAW01"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M3_DIGEST_DOMAIN "KM3RAW01"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA3_DIGEST_DOMAIN "KV3RAW01"
#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M4_DIGEST_DOMAIN "KM4RAW01"

enum klayout_cuda_spatial_antenna_m1_m4_opcode
{
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_SHARED_EMPTY = 1
};

enum klayout_cuda_spatial_antenna_m1_m4_stage
{
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_M1 = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_M2 = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_M3 = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_M4 = 1u << 3
};

enum klayout_cuda_spatial_antenna_m1_m4_role
{
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_POLY_ROLE = 0,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ACTIVE_ROLE = 1,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NPLUS_ROLE = 2,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NWELL_ROLE = 3,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_CONTACT_ROLE = 4,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M1_ROLE = 5,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA1_ROLE = 6,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M2_ROLE = 7,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA2_ROLE = 8,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M3_ROLE = 9,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA3_ROLE = 10,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M4_ROLE = 11
};

enum klayout_cuda_spatial_antenna_m1_m4_option_flag
{
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_SHARED_HIERARCHY = 1u << 0,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_PHYSICAL_DOMAINS = 1u << 1,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NO_BREAKOUT = 1u << 2,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ORTHOGONAL_UNIT_TRANSFORMS = 1u << 3,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NO_PROPERTIES = 1u << 4,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_CLOCKWISE_MANHATTAN_CONTOURS = 1u << 5,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_EXACT_INTEGER_CONNECTIVITY = 1u << 6,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ORDERED_RESIDENT_STAGES = 1u << 7,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_FIXED_RATIO_300_TO_1 = 1u << 8,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DEFAULT_STREAM = 1u << 9,
  /*
   * Initial production proof is intentionally one-sided and clean-only:
   * raw target-metal area is an upper bound; the maximum single exact
   * positive-area POLY/ACTIVE intersection is the gate-area lower bound.
   * Integer cross multiplication checks 300:1 without rounding, and diode
   * exemptions are ignored.  These restrictions can create false fallback
   * but cannot create a false clean certificate.
   */
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_METAL_AREA_UPPER_BOUND = 1u << 10,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_MAX_SINGLE_GATE_LOWER_BOUND = 1u << 11,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_POSITIVE_AREA_POLY_ACTIVE_GATE = 1u << 12,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_INTEGER_RATIO_CROSS_MULTIPLY = 1u << 13,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DIODE_EXEMPTIONS_IGNORED = 1u << 14
};

#define KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_QUALIFIED_OPTIONS \
  ((1u << 15) - 1u)

enum klayout_cuda_spatial_antenna_m1_m4_disposition
{
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_COMPLETE = 0,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_HITS = 1,
  KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_UNCERTAIN = 2
};

struct klayout_cuda_spatial_antenna_m1_m4_cell_v1
{
  uint64_t polygon_begin;
  uint64_t edge_begin;
  uint32_t polygon_count;
  uint32_t edge_count;
};

struct klayout_cuda_spatial_antenna_m1_m4_hierarchy_v1
{
  uint32_t struct_size;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint64_t source_root_cell_index;
  const uint64_t *source_cell_indices;
  uint64_t source_cell_count;
  uint32_t source_cell_index_record_bytes;
  uint32_t reserved0;
  const void *contexts;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t reserved1;
  const uint32_t *context_parent_ids;
  uint64_t context_parent_count;
  uint32_t context_parent_record_bytes;
  uint32_t reserved2;
  uint8_t hierarchy_digest[32];
  uint64_t reserved3[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_hierarchy_echo_v1
{
  uint32_t struct_size;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint64_t source_root_cell_index;
  uint64_t source_cell_count;
  uint32_t source_cell_index_record_bytes;
  uint32_t reserved0;
  uint64_t context_count;
  uint32_t context_record_bytes;
  uint32_t reserved1;
  uint64_t context_parent_count;
  uint32_t context_parent_record_bytes;
  uint32_t reserved2;
  uint8_t hierarchy_digest[32];
  uint64_t reserved3[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_domain_v1
{
  uint32_t struct_size;
  uint32_t role;
  uint32_t physical_layer;
  uint32_t datatype;
  uint32_t source_layer_index;
  uint32_t reserved0;
  const void *cells;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t reserved1;
  const void *polygons;
  uint64_t polygon_count;
  uint32_t polygon_record_bytes;
  uint32_t reserved2;
  const void *edges;
  uint64_t edge_count;
  uint32_t edge_record_bytes;
  uint32_t reserved3;
  uint64_t nonempty_context_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  uint64_t stored_bytes;
  uint64_t expanded_geometry_bytes;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  uint8_t digest_domain[8];
  uint8_t scene_digest[32];
  uint64_t reserved4[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_domain_echo_v1
{
  uint32_t struct_size;
  uint32_t role;
  uint32_t physical_layer;
  uint32_t datatype;
  uint32_t source_layer_index;
  uint32_t reserved0;
  uint64_t cell_count;
  uint32_t cell_record_bytes;
  uint32_t reserved1;
  uint64_t polygon_count;
  uint32_t polygon_record_bytes;
  uint32_t reserved2;
  uint64_t edge_count;
  uint32_t edge_record_bytes;
  uint32_t reserved3;
  uint64_t nonempty_context_count;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  uint64_t stored_bytes;
  uint64_t expanded_geometry_bytes;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  uint8_t digest_domain[8];
  uint8_t scene_digest[32];
  uint64_t reserved4[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_census_v1
{
  uint32_t struct_size;
  uint32_t format_version;
  uint64_t shared_cell_count;
  uint64_t shared_context_count;
  uint64_t context_parent_record_count;
  uint64_t stored_cell_record_count;
  uint64_t stored_polygon_count;
  uint64_t stored_edge_count;
  uint64_t expanded_polygon_count;
  uint64_t expanded_edge_count;
  uint64_t total_stored_bytes;
  uint64_t total_expanded_geometry_bytes;
  uint64_t estimated_peak_bytes;
  uint64_t reserved[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_capacity_v1
{
  uint32_t struct_size;
  uint32_t reserved0;
  uint64_t max_cells;
  uint64_t max_contexts;
  uint64_t max_stored_polygons;
  uint64_t max_stored_edges;
  uint64_t max_flat_polygons;
  uint64_t max_flat_edges;
  uint64_t max_total_stored_bytes;
  uint64_t max_total_expanded_geometry_bytes;
  uint64_t max_estimated_peak_bytes;
  uint64_t max_nodes;
  uint64_t max_rectangles;
  uint64_t max_memberships;
  uint64_t max_pair_occurrences;
  uint64_t max_unique_candidates;
  uint64_t max_cell_members;
  uint64_t max_dsu_iterations;
  uint64_t max_rule_work;
  uint64_t max_device_bytes;
  uint64_t reserved1[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_request_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t requested_mask;
  uint32_t stage_count;
  uint32_t ratio_numerator;
  uint32_t ratio_denominator;
  uint32_t domain_count;
  int32_t device;
  struct klayout_cuda_spatial_antenna_m1_m4_hierarchy_v1 hierarchy;
  struct klayout_cuda_spatial_antenna_m1_m4_domain_v1
    domains[KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT];
  struct klayout_cuda_spatial_antenna_m1_m4_census_v1 census;
  struct klayout_cuda_spatial_antenna_m1_m4_capacity_v1 capacity;
  uint8_t lower_capture_digest[32];
  uint8_t capture_digest[32];
  uint64_t reserved[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_stage_result_v1
{
  uint32_t struct_size;
  uint32_t stage;
  uint64_t component_count;
  uint64_t membership_count;
  uint64_t occupied_cell_count;
  uint64_t pair_occurrence_count;
  /* Deduplicated geometric owner pairs considered for connectivity. */
  uint64_t unique_owner_candidate_count;
  uint64_t edge_count;
  /* Gate-bearing canonical components, independent of geometric pair count. */
  uint64_t gate_count;
  /* Gate components that reached the conservative integer ratio reduction. */
  uint64_t evaluated_count;
  uint64_t exempt_count;
  /*
   * Disjoint live-frontier subsets at this stage.  Their sum is exactly the
   * previous metal frontier plus the newly enabled VIA/metal rectangulation
   * (POLY/CONTACT/M1 at M1).  M1-M3 retain exactly the current metal domain;
   * M4 retains zero because the transaction closes there.  Released is the
   * remainder after that checkpoint is settled.
   */
  uint64_t retained_rectangle_count;
  uint64_t released_rectangle_count;
  uint64_t dsu_iteration_count;
  uint64_t hit_count;
  uint64_t uncertainty_count;
  uint64_t work_count;
  uint64_t stage_ns;
  /*
   * Host-recomputable digest over the request/capture identity, all twelve
   * per-domain rectangle-evidence records, stage ID, every counter above, the
   * transaction outcome flags and accounted peak device bytes.  Canonical
   * labels are not returned by this ABI and are therefore not claimed here.
   */
  uint8_t stage_digest[32];
  uint64_t reserved[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_domain_result_v1
{
  uint32_t struct_size;
  uint32_t role;
  uint64_t owner_count;
  uint64_t rectangle_count;
  uint64_t owner_range_count;
  /*
   * Host-recomputable digest over the capture/domain identity and the three
   * counters above.  It is evidence-integrity metadata, not an independently
   * reconstructed host rectangulation.
   */
  uint8_t rectangle_digest[32];
  uint64_t reserved[4];
};

struct klayout_cuda_spatial_antenna_m1_m4_result_v1
{
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t status;
  uint32_t fallback_flags;
  uint32_t disposition;
  uint32_t opcode;
  uint32_t option_flags;
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t requested_mask;
  uint32_t certified_empty_mask;
  uint32_t clean_mask;
  uint32_t stage_count;
  uint32_t ratio_numerator;
  uint32_t ratio_denominator;
  uint32_t domain_count;
  int32_t device;
  uint32_t device_flags;
  uint32_t closed_domain_mask;
  uint32_t released_stage_mask;
  /* Peak bytes conservatively accounted against the explicit device cap. */
  uint64_t accounted_peak_device_bytes;
  struct klayout_cuda_spatial_antenna_m1_m4_hierarchy_echo_v1 hierarchy;
  struct klayout_cuda_spatial_antenna_m1_m4_domain_echo_v1
    domains[KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT];
  struct klayout_cuda_spatial_antenna_m1_m4_census_v1 census;
  struct klayout_cuda_spatial_antenna_m1_m4_capacity_v1 capacity;
  uint8_t lower_capture_digest[32];
  uint8_t capture_digest[32];
  struct klayout_cuda_spatial_antenna_m1_m4_domain_result_v1
    domain_results[KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT];
  struct klayout_cuda_spatial_antenna_m1_m4_stage_result_v1
    stages[KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT];
  uint64_t setup_ns;
  uint64_t h2d_ns;
  uint64_t d2h_ns;
  uint64_t total_ns;
  uint64_t reserved[4];
  char message[192];
};

typedef int
(*klayout_cuda_spatial_run_antenna_m1_m4_empty_v1_func) (
  const struct klayout_cuda_spatial_antenna_m1_m4_request_v1 *,
  struct klayout_cuda_spatial_antenna_m1_m4_result_v1 *);

KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_antenna_m1_m4_empty_v1 (
  const struct klayout_cuda_spatial_antenna_m1_m4_request_v1 *request,
  struct klayout_cuda_spatial_antenna_m1_m4_result_v1 *result);

#ifdef __cplusplus
}
#endif

#endif
