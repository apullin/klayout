/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaAntennaM4
#define HDR_dbCudaAntennaM4

#include "dbCommon.h"
#include "dbCudaAntennaM1.h"

#include <array>
#include <cstdint>
#include <string>

namespace db
{

/**
 * Fixed raw physical domains through the first four metal levels.
 *
 * The first six values deliberately equal CudaAntennaM1Domain.  A complete
 * capture embeds an unchanged KANTM102 lower capture, so existing M1 ABI and
 * all established lower-domain digests remain byte-for-byte stable.
 */
enum CudaAntennaM4Domain : uint32_t
{
  CudaAntennaM4Poly = 0,
  CudaAntennaM4Active = 1,
  CudaAntennaM4Nplus = 2,
  CudaAntennaM4Nwell = 3,
  CudaAntennaM4Contact = 4,
  CudaAntennaM4Metal1 = 5,
  CudaAntennaM4Via1 = 6,
  CudaAntennaM4Metal2 = 7,
  CudaAntennaM4Via2 = 8,
  CudaAntennaM4Metal3 = 9,
  CudaAntennaM4Via3 = 10,
  CudaAntennaM4Metal4 = 11,
  CudaAntennaM4DomainCount = 12,
  CudaAntennaM4UpperDomainCount = 6
};

/**
 * Host capture and hypothetical expanded-geometry admission limits.
 *
 * No expanded occurrence geometry is allocated.  The aggregate limits cover
 * all twelve domains and the one shared hierarchy.
 */
struct DB_PUBLIC CudaAntennaM4CaptureLimits
{
  CudaM1WidthSpaceSceneLimits scene;
  uint64_t max_total_stored_bytes;
  uint64_t max_total_expanded_geometry_bytes;
  uint64_t max_estimated_peak_bytes;

  CudaAntennaM4CaptureLimits ();
};

/**
 * Compact M1-through-M4 capture.
 *
 * "lower" is an ordinary, independently valid CudaAntennaM1Capture.  The six
 * upper domains store only source-cell-local contours and refer to lower's
 * source_cell_indices, contexts and context_parent_ids.  Thus hierarchy is
 * owned once rather than flattened or repeated for every physical layer.
 */
struct DB_PUBLIC CudaAntennaM4Capture
{
  uint32_t format_version;
  uint32_t reserved;
  CudaAntennaM1Capture lower;
  std::array<uint32_t, CudaAntennaM4UpperDomainCount>
    upper_source_layer_indices;
  std::array<CudaAntennaM1DomainScene, CudaAntennaM4UpperDomainCount>
    upper_domains;
  std::array<uint8_t, 32> digest;

  CudaAntennaM4Capture ();
  void swap (CudaAntennaM4Capture &other) noexcept;
};

/**
 * Device-neutral accounting for a complete compact capture.
 *
 * Domain records reuse the established M1 census record.  Roles 0..5 are the
 * byte-identical lower KANTM102 records and roles 6..11 describe the added
 * VIA1/M2/VIA2/M3/VIA3/M4 geometry.
 */
struct DB_PUBLIC CudaAntennaM4Census
{
  uint32_t format_version;
  uint32_t reserved;
  uint64_t shared_cell_count;
  uint64_t shared_context_count;
  uint64_t context_parent_record_count;
  uint64_t stored_cell_records;
  uint64_t stored_polygon_count;
  uint64_t stored_edge_count;
  uint64_t expanded_polygon_count;
  uint64_t expanded_edge_count;
  uint64_t total_stored_bytes;
  uint64_t total_expanded_geometry_bytes;
  uint64_t estimated_peak_bytes;
  std::array<CudaAntennaM1DomainCensus, CudaAntennaM4DomainCount>
    domains;
  std::array<uint8_t, 32> hierarchy_digest;
  std::array<uint8_t, 32> lower_capture_digest;
  std::array<uint8_t, 32> capture_digest;

  CudaAntennaM4Census ();
};

/**
 * Capture twelve raw FreePDK45 physical layers atomically.
 *
 * Inputs must share one unbroken hierarchy and occupy the exact distinct
 * physical layers 9/0, 1/0, 4/0, 3/0, 10/0, 11/0, 12/0, 13/0, 14/0,
 * 15/0, 16/0 and 17/0.  Every contour, property, transform and capacity
 * restriction of the established compact M1 capture is retained.  On every
 * decline "capture" remains unchanged.
 */
DB_PUBLIC bool cuda_antenna_m4_build_capture (
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
  const db::DeepLayer &raw_metal4,
  const CudaAntennaM4CaptureLimits &limits,
  CudaAntennaM4Capture &capture,
  std::string *decline_reason = 0);

/**
 * Capture and return the census authenticated by that same build.
 *
 * This overload is for an immediate, immutable hand-off from the builder to
 * a transaction.  "capture" and "authenticated_census" are both left
 * unchanged on failure.  A capture received from any other source, or
 * modified after this call, must still use cuda_antenna_m4_capture_census for
 * a complete audit.
 */
DB_PUBLIC bool cuda_antenna_m4_build_capture (
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
  const db::DeepLayer &raw_metal4,
  const CudaAntennaM4CaptureLimits &limits,
  CudaAntennaM4Capture &capture,
  CudaAntennaM4Census &authenticated_census,
  std::string *decline_reason = 0);

/**
 * Materialize one legacy-shaped raw scene without changing compact ownership.
 *
 * Lower domains delegate to the established M1 materializer.  Upper domains
 * derive the same context and offset streams from the one shared hierarchy.
 * On failure "scene" remains unchanged.
 */
DB_PUBLIC bool cuda_antenna_m4_materialize_domain_scene (
  const CudaAntennaM4Capture &capture,
  CudaAntennaM4Domain domain,
  CudaRawManhattanScene &scene,
  std::string *decline_reason = 0);

/**
 * Recompute the canonical KANTM401 transaction digest.
 *
 * A structurally inconsistent lower capture, upper domain, layer binding or
 * count fails closed and leaves "digest" unchanged.
 */
DB_PUBLIC bool cuda_antenna_m4_capture_digest (
  const CudaAntennaM4Capture &capture,
  std::array<uint8_t, 32> &digest);

/**
 * Validate a capture and derive exact compact/expanded accounting.
 */
DB_PUBLIC bool cuda_antenna_m4_capture_census (
  const CudaAntennaM4Capture &capture,
  CudaAntennaM4Census &census,
  std::string *decline_reason = 0);

/**
 * Render one stable single-line host census record.
 */
DB_PUBLIC std::string cuda_antenna_m4_census_text (
  const CudaAntennaM4Census &census);

} // namespace db

#endif
