/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaAntennaM1
#define HDR_dbCudaAntennaM1

#include "dbCommon.h"
#include "dbCudaM1WidthSpace.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace db
{

/**
 * Fixed raw physical domains in the first M1 antenna transaction.
 *
 * The order is part of the KANTM101 capture digest.  It mirrors the
 * production deck's gate/diode/connect prefix:
 *
 *   gate = POLY & ACTIVE
 *   diode = NPLUS & (ACTIVE - NWELL)
 *   gate-POLY-CONTACT-M1 and diode-CONTACT
 */
enum CudaAntennaM1Domain : uint32_t
{
  CudaAntennaM1Poly = 0,
  CudaAntennaM1Active = 1,
  CudaAntennaM1Nplus = 2,
  CudaAntennaM1Nwell = 3,
  CudaAntennaM1Contact = 4,
  CudaAntennaM1Metal1 = 5,
  CudaAntennaM1DomainCount = 6
};

/**
 * Host-capture and geometry-expansion capacity limits.
 *
 * "scene" protects each existing raw-Manhattan serialization.  The three
 * byte limits protect the fused transaction after summing all six domains.
 * Their defaults are conservative admission guards, not a promise that a
 * production design fits.  A later explicit census may raise them after
 * checking available host/device memory; exceeding them is always a normal
 * fail-closed decline.
 *
 * Stored bytes are canonical packed-record accounting, not std::vector
 * capacity or a measured runtime resident set.  Expanded bytes are a
 * geometry-only planning estimate: one expanded CudaM1WidthSpacePolygon plus
 * one uint32 context ID per flat polygon and one CudaM1WidthSpaceEdge per flat
 * edge.  Estimated peak is their checked sum.  These figures deliberately do
 * not claim to estimate a future connectivity graph or candidate workspace,
 * and capture does not allocate the expanded representation.
 */
struct DB_PUBLIC CudaAntennaM1CaptureLimits
{
  CudaM1WidthSpaceSceneLimits scene;
  uint64_t max_total_stored_bytes;
  uint64_t max_total_expanded_geometry_bytes;
  uint64_t max_estimated_peak_bytes;

  CudaAntennaM1CaptureLimits ();
};

/**
 * One source-cell range for a domain in the shared antenna capture.
 *
 * Source-cell identity is held once by CudaAntennaM1Capture.  Keeping only
 * domain-local geometry ranges here avoids repeating that identity six times.
 */
struct DB_PUBLIC CudaAntennaM1DomainCell
{
  uint64_t polygon_begin;
  uint64_t edge_begin;
  uint32_t polygon_count;
  uint32_t edge_count;
};

/**
 * Geometry owned by one physical domain in a shared antenna capture.
 *
 * Hierarchy contexts and the derived nonempty-context/offset streams are not
 * stored here.  They are reconstructed exactly from the capture's one shared
 * context stream and these cell ranges when a legacy raw scene is requested.
 */
struct DB_PUBLIC CudaAntennaM1DomainScene
{
  uint64_t flat_polygon_count;
  uint64_t flat_edge_count;
  int64_t scene_left;
  int64_t scene_bottom;
  int64_t scene_right;
  int64_t scene_top;
  std::vector<CudaAntennaM1DomainCell> cells;
  std::vector<CudaM1WidthSpacePolygon> polygons;
  std::vector<CudaM1WidthSpaceEdge> edges;
  std::array<uint8_t, 32> digest;

  CudaAntennaM1DomainScene ();
  void swap (CudaAntennaM1DomainScene &other) noexcept;
};

/**
 * Per-domain census derived from one compact shared-hierarchy domain.
 *
 * stored_bytes is the domain-local KANTM102 payload.  legacy_stored_bytes is
 * the byte count of the byte-identical KANTM101 raw scene reconstructed from
 * it, including the formerly duplicated hierarchy and offset streams.
 */
struct DB_PUBLIC CudaAntennaM1DomainCensus
{
  uint32_t role;
  uint32_t physical_layer;
  uint32_t datatype;
  uint32_t source_layer_index;
  uint64_t stored_cell_count;
  uint64_t stored_context_count;
  uint64_t nonempty_context_count;
  uint64_t stored_polygon_count;
  uint64_t stored_edge_count;
  uint64_t expanded_polygon_count;
  uint64_t expanded_edge_count;
  uint64_t stored_bytes;
  uint64_t legacy_stored_bytes;
  uint64_t expanded_geometry_bytes;
  std::array<uint8_t, 32> scene_digest;

  CudaAntennaM1DomainCensus ();
};

/**
 * Device-neutral census for a complete M1 antenna capture.
 *
 * The shared counts describe the one common hierarchy.  Current stored-record
 * and byte totals describe the compact KANTM102 representation.  The explicit
 * legacy totals describe the former six-scene KANTM101 representation and are
 * retained both for auditability and byte-identical legacy digest replay.
 */
struct DB_PUBLIC CudaAntennaM1Census
{
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t reserved;
  uint64_t source_root_cell_index;
  uint64_t shared_cell_count;
  uint64_t shared_context_count;
  uint64_t context_parent_record_count;
  uint64_t context_parent_bytes;
  uint64_t stored_cell_records;
  uint64_t stored_context_records;
  uint64_t nonempty_context_records;
  uint64_t stored_polygon_count;
  uint64_t stored_edge_count;
  uint64_t expanded_polygon_count;
  uint64_t expanded_edge_count;
  uint64_t total_stored_bytes;
  uint64_t legacy_stored_cell_records;
  uint64_t legacy_stored_context_records;
  uint64_t legacy_total_stored_bytes;
  uint64_t total_expanded_geometry_bytes;
  uint64_t estimated_peak_bytes;
  uint64_t legacy_estimated_peak_bytes;
  std::array<CudaAntennaM1DomainCensus, CudaAntennaM1DomainCount> domains;
  std::array<uint8_t, 32> hierarchy_digest;
  std::array<uint8_t, 32> capture_digest;

  CudaAntennaM1Census ();
};

/**
 * Default-off host foundation for a future fused M1 antenna transaction.
 *
 * No CUDA backend is loaded and no production DRC path calls this type.  One
 * hierarchy/context stream and one parent stream are shared by all domains.
 * Each domain owns only its source-cell geometry ranges and contours.  Legacy
 * raw scenes and their KANTM101 digests remain exactly reconstructible.
 */
struct DB_PUBLIC CudaAntennaM1Capture
{
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t reserved;
  uint64_t source_root_cell_index;
  std::array<uint32_t, CudaAntennaM1DomainCount> source_layer_indices;
  std::vector<uint64_t> source_cell_indices;
  std::vector<CudaM1WidthSpaceContext> contexts;
  std::array<CudaAntennaM1DomainScene, CudaAntennaM1DomainCount> domains;
  std::vector<uint32_t> context_parent_ids;
  std::array<uint8_t, 32> hierarchy_digest;
  std::array<uint8_t, 32> digest;

  CudaAntennaM1Capture ();
  void swap (CudaAntennaM1Capture &other) noexcept;
};

/**
 * Capture raw FreePDK45 POLY/ACTIVE/NPLUS/NWELL/CONTACT/M1 atomically.
 *
 * All inputs must share one store, layout, layout index, top cell and
 * unbroken hierarchy and must occupy six distinct physical layers.  Every
 * existing raw-Manhattan contour/property/transform/capacity restriction is
 * retained.  The established raw scene requires every domain to be globally
 * nonempty; an empty POLY/ACTIVE/NPLUS/NWELL/CONTACT/M1 domain therefore
 * declines safely to the future literal CPU path instead of inventing a
 * special empty encoding.  On every decline "capture" is left unchanged.
 */
DB_PUBLIC bool cuda_antenna_m1_build_capture (
  const db::DeepLayer &raw_poly,
  const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_contact,
  const db::DeepLayer &raw_metal1,
  const CudaAntennaM1CaptureLimits &limits,
  CudaAntennaM1Capture &capture,
  std::string *decline_reason = 0);

/**
 * Capture and return the census authenticated by that same build.
 *
 * This overload is for an immediate, immutable hand-off from the builder to
 * a transaction.  "capture" and "authenticated_census" are both left
 * unchanged on failure.  A capture received from any other source, or
 * modified after this call, must still use cuda_antenna_m1_capture_census for
 * a complete audit.
 */
DB_PUBLIC bool cuda_antenna_m1_build_capture (
  const db::DeepLayer &raw_poly,
  const db::DeepLayer &raw_active,
  const db::DeepLayer &raw_nplus,
  const db::DeepLayer &raw_nwell,
  const db::DeepLayer &raw_contact,
  const db::DeepLayer &raw_metal1,
  const CudaAntennaM1CaptureLimits &limits,
  CudaAntennaM1Capture &capture,
  CudaAntennaM1Census &authenticated_census,
  std::string *decline_reason = 0);

/**
 * Materialize one byte-identical legacy raw-Manhattan scene.
 *
 * The operation is explicit because it temporarily duplicates the shared
 * hierarchy and derives the nonempty-context and offset streams.  On failure
 * "scene" is unchanged.
 */
DB_PUBLIC bool cuda_antenna_m1_materialize_domain_scene (
  const CudaAntennaM1Capture &capture,
  CudaAntennaM1Domain domain,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Recompute the canonical legacy KANTM101 transaction digest.
 *
 * KANTM102 changes physical ownership only: the digest is intentionally
 * byte-identical to the six-scene KANTM101 representation.  False means a
 * domain, hierarchy, layer binding or count is structurally inconsistent.
 * "digest" is unchanged on failure.
 */
DB_PUBLIC bool cuda_antenna_m1_capture_digest (
  const CudaAntennaM1Capture &capture,
  std::array<uint8_t, 32> &digest);

/**
 * Validate a capture and derive its complete stored/expanded census.
 *
 * This is the focused host-only census hook for small fixtures and a later
 * explicitly requested production capture.  It does not inspect a GPU.
 */
DB_PUBLIC bool cuda_antenna_m1_capture_census (
  const CudaAntennaM1Capture &capture,
  CudaAntennaM1Census &census,
  std::string *decline_reason = 0);

/**
 * Render one stable, single-line census record suitable for benchmark logs.
 */
DB_PUBLIC std::string cuda_antenna_m1_census_text (
  const CudaAntennaM1Census &census);

} // namespace db

#endif
