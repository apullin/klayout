/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaM1WidthSpace
#define HDR_dbCudaM1WidthSpace

#include "dbCommon.h"
#include "dbRegionLocalOperations.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace db
{

class DeepLayer;

/**
 * Capacity limits for speculative M1 width/space scene construction.
 *
 * These limits protect only host lowering.  A future backend must apply its
 * own checked device-memory, grid-membership, and pair-work limits.
 */
struct DB_PUBLIC CudaM1WidthSpaceSceneLimits
{
  uint64_t max_cells;
  uint64_t max_contexts;
  uint64_t max_stored_polygons;
  uint64_t max_stored_edges;
  uint64_t max_flat_polygons;
  uint64_t max_flat_edges;

  CudaM1WidthSpaceSceneLimits ();
};

/**
 * The exact rule shape accepted by the first FreePDK45 scene builder.
 *
 * "inputs_are_merged" is an explicit assertion from the future integration
 * site.  DeepLayer itself does not retain a queryable merged-semantics bit.
 */
struct DB_PUBLIC CudaM1WidthSpaceBuildSpec
{
  db::Coord width_distance;
  db::Coord spacing_distance;
  db::RegionCheckOptions width_options;
  db::RegionCheckOptions spacing_options;
  bool inputs_are_merged;

  CudaM1WidthSpaceBuildSpec ();
};

/**
 * One expanded occurrence of a source cell.
 *
 * The transform code is db::Trans::rot(): the eight orthogonal rotations and
 * reflections.  Translation is in layout DBU.  The record is pointer-free.
 */
struct DB_PUBLIC CudaM1WidthSpaceContext
{
  int64_t tx;
  int64_t ty;
  uint32_t cell_id;
  uint32_t transform_code;
};

/**
 * One source cell and its contiguous polygon/edge ranges.
 */
struct DB_PUBLIC CudaM1WidthSpaceCell
{
  uint64_t source_cell_index;
  uint64_t polygon_begin;
  uint64_t edge_begin;
  uint32_t polygon_count;
  uint32_t edge_count;
};

/**
 * One source-cell polygon.
 *
 * polygon_id is local to the owning cell.  Its directed contour edges occupy
 * [edge_begin, edge_begin + edge_count) in source contour order.
 */
struct DB_PUBLIC CudaM1WidthSpacePolygon
{
  uint64_t edge_begin;
  int64_t left;
  int64_t bottom;
  int64_t right;
  int64_t top;
  uint32_t polygon_id;
  uint32_t edge_count;
};

/**
 * One directed local-coordinate contour edge.
 */
struct DB_PUBLIC CudaM1WidthSpaceEdge
{
  int64_t x1;
  int64_t y1;
  int64_t x2;
  int64_t y2;
};

/**
 * Owning host representation of a future fused M1 width/spacing request.
 *
 * Vector storage is owned here, but every serialized record is pointer-free
 * and refers to other records only through stable indices and ranges.
 */
struct DB_PUBLIC CudaM1WidthSpaceScene
{
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t reserved;
  int64_t width_distance;
  int64_t spacing_distance;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  std::vector<CudaM1WidthSpaceContext> contexts;
  std::vector<uint32_t> metal_contexts;
  std::vector<uint64_t> context_polygon_offsets;
  std::vector<uint64_t> context_edge_offsets;
  std::vector<CudaM1WidthSpaceCell> cells;
  std::vector<CudaM1WidthSpacePolygon> polygons;
  std::vector<CudaM1WidthSpaceEdge> edges;
  std::array<uint8_t, 32> digest;

  CudaM1WidthSpaceScene ();
  void swap (CudaM1WidthSpaceScene &other) noexcept;
};

/**
 * Exact compact serialization of one qualified raw physical Manhattan layer.
 *
 * The legacy type name is retained because M2 was the first consumer.  The
 * same pointer-free storage is also used by the fixed raw-ACTIVE and
 * raw-CONTACT builders below.  Role is bound by each builder's exact physical
 * layer contract and by a distinct canonical digest domain, not by an
 * additional field in this record.  This keeps the established KM2RAW01
 * descriptor and digest payload byte-for-byte stable.
 */
struct DB_PUBLIC CudaM2RawManhattanScene
{
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t reserved;
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;

  std::vector<CudaM1WidthSpaceContext> contexts;
  std::vector<uint32_t> metal_contexts;
  std::vector<uint64_t> context_polygon_offsets;
  std::vector<uint64_t> context_edge_offsets;
  std::vector<CudaM1WidthSpaceCell> cells;
  std::vector<CudaM1WidthSpacePolygon> polygons;
  std::vector<CudaM1WidthSpaceEdge> edges;
  std::array<uint8_t, 32> digest;

  CudaM2RawManhattanScene ();
  void swap (CudaM2RawManhattanScene &other) noexcept;
};

/**
 * Role-neutral spelling for the established raw-Manhattan scene storage.
 *
 * This alias deliberately adds no fields and changes no record layout.
 */
typedef CudaM2RawManhattanScene CudaRawManhattanScene;

/**
 * Compute the canonical digest of a structurally valid scene.
 *
 * Fields are hashed explicitly in little-endian form, so padding and host
 * object addresses cannot affect the result.  False means the scene is
 * structurally inconsistent and must not be published.
 */
DB_PUBLIC bool cuda_m1_width_space_scene_digest (
  const CudaM1WidthSpaceScene &scene, std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KM2RAW01 digest of a structurally valid raw scene.
 */
DB_PUBLIC bool cuda_m2_raw_manhattan_scene_digest (
  const CudaM2RawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KM1RAW01 digest of a structurally valid raw-M1 scene.
 *
 * The hashed field order after the magic is identical to KM2RAW01.  The
 * distinct domain prevents a valid M2 serialization from being replayed as
 * the raw physical M1 operand of the resident morphology certificate.
 */
DB_PUBLIC bool cuda_m1_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KARAW001 digest of a structurally valid raw-ACTIVE
 * scene.  The hashed field order after the magic is identical to KM2RAW01.
 */
DB_PUBLIC bool cuda_active_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KCRAW001 digest of a structurally valid raw-CONTACT
 * scene.  The hashed field order after the magic is identical to KM2RAW01.
 */
DB_PUBLIC bool cuda_contact_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KPOLY001 digest of a structurally valid raw-POLY
 * scene.  The hashed field order after the magic is identical to KM2RAW01.
 */
DB_PUBLIC bool cuda_poly_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KNPLS001 digest of a structurally valid raw-NPLUS
 * scene.  The hashed field order after the magic is identical to KM2RAW01.
 */
DB_PUBLIC bool cuda_nplus_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KPPLS001 digest of a structurally valid raw-PPLUS
 * scene.  The hashed field order after the magic is identical to KM2RAW01.
 */
DB_PUBLIC bool cuda_pplus_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KGATE001 digest of a structurally valid exact
 * derived-GATE scene.  This distinct domain cannot be replayed as a physical
 * GDS layer.
 */
DB_PUBLIC bool cuda_gate_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KNWEL001 digest of a structurally valid raw-NWELL
 * scene.  The hashed field order after the magic is identical to KM2RAW01.
 */
DB_PUBLIC bool cuda_nwell_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Compute the canonical KWRWL001 digest of one combined raw-NWELL/PWELL
 * scene.  The scene contains both physical layers in deterministic
 * NWELL-then-PWELL order within every source cell.
 */
DB_PUBLIC bool cuda_well_union_raw_manhattan_scene_digest (
  const CudaRawManhattanScene &scene,
  std::array<uint8_t, 32> &digest);

/**
 * Build the narrowly qualified merged-M1 scene.
 *
 * The two logical sources must be the identical DeepLayer (same store,
 * layout, layout index, top cell, and layer), with no breakout cells.  This
 * mirrors the two nodes of the existing width/space batch while serializing
 * the geometry once.  On any unsupported input, overflow, capacity excess, or
 * exception, false is returned and "scene" is left unchanged.
 *
 * No backend is loaded or called by this seam.
 */
DB_PUBLIC bool cuda_m1_width_space_build_scene (
  const db::DeepLayer &width_metal1,
  const db::DeepLayer &spacing_metal1,
  const CudaM1WidthSpaceBuildSpec &spec,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaM1WidthSpaceScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 M2 layer verbatim.
 *
 * The caller must pass DeepRegion::deep_layer(), never merged_deep_layer().
 * This builder does not and cannot infer a merged-semantics claim.  It
 * requires physical layer 13/0, 0.5-nm DBU, no breakout cells or properties,
 * supported orthogonal hierarchy, and simple clockwise Manhattan contours.
 * On every decline "scene" is left unchanged.
 */
DB_PUBLIC bool cuda_m2_raw_manhattan_build_scene (
  const db::DeepLayer &raw_metal2,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaM2RawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 M1 layer verbatim.
 *
 * The caller must pass DeepRegion::deep_layer(), never merged_deep_layer().
 * This builder requires physical layer 11/0 and otherwise applies the same
 * exact hierarchy, contour, property, DBU and capacity contract as raw M2.
 * Success publishes a KM1RAW01 digest; every decline leaves "scene"
 * unchanged.
 */
DB_PUBLIC bool cuda_m1_raw_manhattan_build_scene (
  const db::DeepLayer &raw_metal1,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 ACTIVE layer verbatim.
 *
 * This fixed-domain builder accepts only physical layer 1/0 at 0.5-nm DBU.
 * It otherwise applies the same hierarchy, contour, property and capacity
 * contract as the raw-M2 builder.  Success publishes a KARAW001 digest.
 */
DB_PUBLIC bool cuda_active_raw_manhattan_build_scene (
  const db::DeepLayer &raw_active,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 CONTACT layer verbatim.
 *
 * This fixed-domain builder accepts only physical layer 10/0 at 0.5-nm DBU.
 * It otherwise applies the same hierarchy, contour, property and capacity
 * contract as the raw-M2 builder.  Success publishes a KCRAW001 digest.
 */
DB_PUBLIC bool cuda_contact_raw_manhattan_build_scene (
  const db::DeepLayer &raw_contact,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 POLY layer verbatim.
 *
 * This fixed-domain builder accepts only physical layer 9/0 at 0.5-nm DBU.
 * It otherwise applies the established raw-Manhattan hierarchy, contour,
 * property and capacity contract.
 */
DB_PUBLIC bool cuda_poly_raw_manhattan_build_scene (
  const db::DeepLayer &raw_poly,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 NPLUS layer verbatim.
 *
 * This fixed-domain builder accepts only physical layer 4/0 at 0.5-nm DBU.
 */
DB_PUBLIC bool cuda_nplus_raw_manhattan_build_scene (
  const db::DeepLayer &raw_nplus,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 PPLUS layer verbatim.
 *
 * This fixed-domain builder accepts only physical layer 5/0 at 0.5-nm DBU.
 */
DB_PUBLIC bool cuda_pplus_raw_manhattan_build_scene (
  const db::DeepLayer &raw_pplus,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize an exact already-derived GATE integer set.
 *
 * Unlike physical raw-layer builders this deliberately makes no GDS
 * layer/datatype claim.  It accepts one valid internal DeepLayer with the
 * established hierarchy, 0.5-nm DBU, no breakout cells or properties, and
 * simple clockwise Manhattan polygons.  Success publishes KGATE001.
 */
DB_PUBLIC bool cuda_gate_raw_manhattan_build_scene (
  const db::DeepLayer &derived_gate,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Serialize the supplied raw physical FreePDK45 NWELL layer verbatim.
 *
 * This fixed-domain builder accepts only physical layer 3/0 at 0.5-nm DBU.
 */
DB_PUBLIC bool cuda_nwell_raw_manhattan_build_scene (
  const db::DeepLayer &raw_nwell,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Rebuild the parent occurrence of every established raw-scene context.
 *
 * The existing scene format intentionally remains byte-for-byte unchanged.
 * This sidecar supplies the hierarchy edge needed by graph-oriented
 * consumers such as antenna connectivity: entry zero is UINT32_MAX and every
 * later entry names an earlier parent context.  The supplied DeepLayer must
 * be the source hierarchy of "scene".  On every decline "parents" is left
 * unchanged.
 */
DB_PUBLIC bool cuda_raw_manhattan_context_parents (
  const db::DeepLayer &raw_layer,
  const CudaRawManhattanScene &scene,
  const CudaM1WidthSpaceSceneLimits &limits,
  std::vector<uint32_t> &parents,
  std::string *decline_reason = 0);

/**
 * Serialize physical FreePDK45 NWELL 3/0 and PWELL 2/0 into one raw scene.
 *
 * Both operands must share the identical store, layout, layout index and top
 * cell, have no breakout cells, and use distinct internal layers.  Each cell
 * record concatenates NWELL polygons followed by PWELL polygons so the
 * backend can form their exact integer-set union in one resident pass.
 * Success publishes a KWRWL001 digest.  On every decline "scene" is left
 * unchanged.
 */
DB_PUBLIC bool cuda_well_union_raw_manhattan_build_scene (
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_pwell,
  const CudaM1WidthSpaceSceneLimits &limits,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Try the exact default-off raw-M1 / resident M1.5-.9 certificate.
 *
 * True is returned only when all five unchanged FreePDK45 morphology rules
 * are strictly validated empty.  False is a normal decline and requires the
 * caller to execute the literal historical classify/space transaction.
 */
DB_PUBLIC bool cuda_m1_5_9_try_empty (
  const db::DeepLayer &raw_metal1);

/**
 * Try the exact raw-M1 / resident M1.1/M1.2 empty certificate.
 *
 * The backend constructs the exact Manhattan union and scans its canonical
 * strips in both coordinate orientations.  True certifies both complete
 * width and spacing rule universes empty without first constructing
 * merged_deep_layer().  False is a normal decline and requires the literal
 * historical merged-layer transaction.
 */
DB_PUBLIC bool cuda_m1_raw_width_space_try_empty (
  const db::DeepLayer &raw_metal1);

/**
 * Try the live atomic M1 width/spacing empty certificate.
 *
 * The build specification is validated again before any backend call.  True
 * means both complete rule universes are certified empty.  False is a normal
 * decline and requires the caller to execute both pristine CPU operations.
 */
DB_PUBLIC bool cuda_m1_width_space_try_empty (
  const db::DeepLayer &merged_metal1,
  const CudaM1WidthSpaceBuildSpec &spec);

} // namespace db

#endif
