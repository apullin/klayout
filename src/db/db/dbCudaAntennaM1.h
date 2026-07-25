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
 * Per-domain census derived from one established CudaRawManhattanScene.
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
  uint64_t expanded_geometry_bytes;
  std::array<uint8_t, 32> scene_digest;

  CudaAntennaM1DomainCensus ();
};

/**
 * Device-neutral census for a complete M1 antenna capture.
 *
 * The shared counts describe the one common hierarchy.  The stored-record
 * totals deliberately include the six copies currently owned by the six
 * raw-Manhattan scenes, so memory accounting cannot hide host duplication.
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
  uint64_t total_expanded_geometry_bytes;
  uint64_t estimated_peak_bytes;
  std::array<CudaAntennaM1DomainCensus, CudaAntennaM1DomainCount> domains;
  std::array<uint8_t, 32> hierarchy_digest;
  std::array<uint8_t, 32> capture_digest;

  CudaAntennaM1Census ();
};

/**
 * Default-off host foundation for a future fused M1 antenna transaction.
 *
 * No CUDA backend is loaded and no production DRC path calls this type.  Each
 * domain reuses CudaRawManhattanScene byte-for-byte.  The additional header
 * binds the six physical/internal layer roles and their common hierarchy.
 */
struct DB_PUBLIC CudaAntennaM1Capture
{
  uint32_t format_version;
  uint32_t dbu_per_micron;
  uint32_t root_cell;
  uint32_t reserved;
  uint64_t source_root_cell_index;
  std::array<uint32_t, CudaAntennaM1DomainCount> source_layer_indices;
  std::array<CudaRawManhattanScene, CudaAntennaM1DomainCount> domains;
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
 * Recompute the canonical KANTM101 transaction digest.
 *
 * False means a domain, hierarchy, layer binding or count is structurally
 * inconsistent.  "digest" is unchanged on failure.
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
