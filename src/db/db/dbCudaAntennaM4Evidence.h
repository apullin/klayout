/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaAntennaM4Evidence
#define HDR_dbCudaAntennaM4Evidence

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace db
{
namespace cuda_antenna_m1_m4_evidence
{

typedef std::array<std::uint8_t, 32> Digest;

inline void add_u32 (
  cuda_active3_digest::Sha256 &sha, std::uint32_t value)
{
  const std::uint8_t bytes [4] = {
    std::uint8_t (value),
    std::uint8_t (value >> 8),
    std::uint8_t (value >> 16),
    std::uint8_t (value >> 24)
  };
  sha.update (bytes, sizeof (bytes));
}

inline void add_u64 (
  cuda_active3_digest::Sha256 &sha, std::uint64_t value)
{
  std::uint8_t bytes [8];
  for (unsigned int byte = 0; byte < 8; ++byte) {
    bytes [byte] = std::uint8_t (value >> (byte * 8));
  }
  sha.update (bytes, sizeof (bytes));
}

/**
 * Produce the host-recomputable evidence digest for one rectangulated domain.
 *
 * Pointer values and reserved bytes are intentionally excluded.  The digest
 * binds the immutable capture/domain identity and all three backend-produced
 * domain counters using a fixed little-endian encoding.
 */
inline bool domain_digest (
  const klayout_cuda_spatial_antenna_m1_m4_request_v1 &request,
  std::size_t index,
  const klayout_cuda_spatial_antenna_m1_m4_domain_result_v1 &result,
  Digest &digest)
{
  if (index >= KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT) {
    return false;
  }

  static const std::uint8_t magic [8] = {
    'K', 'A', 'D', 'O', 'M', 'V', '0', '1'
  };
  const klayout_cuda_spatial_antenna_m1_m4_domain_v1 &domain =
    request.domains [index];
  cuda_active3_digest::Sha256 sha;
  sha.update (magic, sizeof (magic));
  add_u32 (sha, request.abi_version);
  add_u32 (sha, request.opcode);
  add_u32 (sha, request.option_flags);
  add_u32 (sha, request.format_version);
  add_u32 (sha, request.dbu_per_micron);
  sha.update (
    request.hierarchy.hierarchy_digest,
    sizeof (request.hierarchy.hierarchy_digest));
  sha.update (
    request.lower_capture_digest, sizeof (request.lower_capture_digest));
  sha.update (request.capture_digest, sizeof (request.capture_digest));
  add_u32 (sha, std::uint32_t (index));
  add_u32 (sha, domain.role);
  add_u32 (sha, domain.physical_layer);
  add_u32 (sha, domain.datatype);
  add_u32 (sha, domain.source_layer_index);
  add_u64 (sha, domain.cell_count);
  add_u64 (sha, domain.polygon_count);
  add_u64 (sha, domain.edge_count);
  add_u64 (sha, domain.nonempty_context_count);
  add_u64 (sha, domain.flat_polygon_count);
  add_u64 (sha, domain.flat_edge_count);
  add_u64 (sha, domain.stored_bytes);
  add_u64 (sha, domain.expanded_geometry_bytes);
  add_u64 (sha, std::uint64_t (domain.scene_left));
  add_u64 (sha, std::uint64_t (domain.scene_bottom));
  add_u64 (sha, std::uint64_t (domain.scene_right));
  add_u64 (sha, std::uint64_t (domain.scene_top));
  sha.update (domain.digest_domain, sizeof (domain.digest_domain));
  sha.update (domain.scene_digest, sizeof (domain.scene_digest));
  add_u32 (sha, result.struct_size);
  add_u32 (sha, result.role);
  add_u64 (sha, result.owner_count);
  add_u64 (sha, result.rectangle_count);
  add_u64 (sha, result.owner_range_count);
  digest = sha.finish ();
  return true;
}

/**
 * Produce the host-recomputable evidence digest for one completed stage.
 *
 * Every stage binds all twelve rectangulated-domain evidence records.  This
 * includes ACTIVE, which participates in the gate census, and retains NPLUS
 * and NWELL identity even while diode handling is disabled.
 */
inline bool stage_digest (
  const klayout_cuda_spatial_antenna_m1_m4_request_v1 &request,
  const klayout_cuda_spatial_antenna_m1_m4_result_v1 &result,
  std::size_t index, Digest &digest)
{
  if (index >= KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT) {
    return false;
  }

  static const std::uint8_t magic [8] = {
    'K', 'A', 'S', 'T', 'G', 'V', '0', '1'
  };
  const klayout_cuda_spatial_antenna_m1_m4_stage_result_v1 &stage =
    result.stages [index];
  cuda_active3_digest::Sha256 sha;
  sha.update (magic, sizeof (magic));
  add_u32 (sha, request.abi_version);
  add_u32 (sha, request.opcode);
  add_u32 (sha, request.option_flags);
  add_u32 (sha, request.format_version);
  add_u32 (sha, request.dbu_per_micron);
  add_u32 (sha, request.requested_mask);
  add_u32 (sha, request.stage_count);
  add_u32 (sha, request.ratio_numerator);
  add_u32 (sha, request.ratio_denominator);
  add_u32 (sha, request.domain_count);
  add_u32 (sha, std::uint32_t (request.device));
  sha.update (
    request.hierarchy.hierarchy_digest,
    sizeof (request.hierarchy.hierarchy_digest));
  sha.update (
    request.lower_capture_digest, sizeof (request.lower_capture_digest));
  sha.update (request.capture_digest, sizeof (request.capture_digest));

  add_u32 (sha, result.status);
  add_u32 (sha, result.fallback_flags);
  add_u32 (sha, result.disposition);
  add_u32 (sha, result.certified_empty_mask);
  add_u32 (sha, result.clean_mask);
  add_u32 (sha, result.device_flags);
  add_u32 (sha, result.closed_domain_mask);
  add_u32 (sha, result.released_stage_mask);
  add_u64 (sha, result.accounted_peak_device_bytes);

  for (std::size_t role = 0;
       role < KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT; ++role) {
    sha.update (
      result.domain_results [role].rectangle_digest,
      sizeof (result.domain_results [role].rectangle_digest));
  }

  add_u32 (sha, stage.struct_size);
  add_u32 (sha, stage.stage);
  add_u64 (sha, stage.component_count);
  add_u64 (sha, stage.membership_count);
  add_u64 (sha, stage.occupied_cell_count);
  add_u64 (sha, stage.pair_occurrence_count);
  add_u64 (sha, stage.unique_owner_candidate_count);
  add_u64 (sha, stage.edge_count);
  add_u64 (sha, stage.gate_count);
  add_u64 (sha, stage.evaluated_count);
  add_u64 (sha, stage.exempt_count);
  add_u64 (sha, stage.retained_rectangle_count);
  add_u64 (sha, stage.released_rectangle_count);
  add_u64 (sha, stage.dsu_iteration_count);
  add_u64 (sha, stage.hit_count);
  add_u64 (sha, stage.uncertainty_count);
  add_u64 (sha, stage.work_count);
  add_u64 (sha, stage.stage_ns);
  digest = sha.finish ();
  return true;
}

} // namespace cuda_antenna_m1_m4_evidence
} // namespace db

#endif
