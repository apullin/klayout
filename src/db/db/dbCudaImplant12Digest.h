/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaImplant12Digest
#define HDR_dbCudaImplant12Digest

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace db
{
namespace cuda_implant12_digest
{

inline bool request_digest (
  const klayout_cuda_spatial_implant12_request_v1 &request,
  std::array<std::uint8_t, 32> &digest)
{
  if (request.context_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_context_v1) ||
      request.cell_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_cell_v1) ||
      request.contour_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_contour_v1) ||
      request.edge_record_bytes !=
        sizeof (klayout_cuda_spatial_implant12_edge_v1) ||
      (request.context_count && ! request.contexts) ||
      (request.implant_context_count && ! request.implant_contexts) ||
      (request.implant_edge_offset_count &&
        ! request.implant_edge_offsets) ||
      (request.gate_context_count && ! request.gate_contexts) ||
      (request.contact_context_count && ! request.contact_contexts) ||
      (request.cell_count && ! request.cells) ||
      (request.contour_count && ! request.contours) ||
      (request.edge_count && ! request.edges)) {
    return false;
  }

  static const char magic [16] = {
    'K', 'L', 'I', 'M', 'P', 'L', 'A', 'N',
    'T', '1', '2', 'V', '1', 0, 0, 0
  };
  db::cuda_active3_digest::Sha256 sha;
  sha.update (magic, sizeof (magic));

#define KLAYOUT_IMPLANT12_DIGEST_FIELD(field) \
  sha.update (&request.field, sizeof (request.field))
  KLAYOUT_IMPLANT12_DIGEST_FIELD (abi_version);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (struct_size);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (opcode);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (option_flags);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (format_version);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (dbu_per_micron);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (root_cell);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (requested_mask);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (device);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant1_distance);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant2_distance);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (grid_cell_size);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (context_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (context_record_bytes);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant_context_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant_edge_offset_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (gate_context_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (contact_context_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (cell_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (cell_record_bytes);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (contour_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (contour_record_bytes);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (edge_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (edge_record_bytes);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_implant_polygon_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_gate_polygon_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_contact_polygon_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_implant_contour_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_gate_contour_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_contact_contour_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_implant_edge_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_gate_edge_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (flat_contact_edge_count);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant_left);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant_bottom);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant_right);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (implant_top);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_contexts);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_grid_cells);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_implant_memberships);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_gate_query_visits);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_gate_candidate_work);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_contact_query_visits);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_contact_candidate_work);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_flat_polygons);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_flat_contours);
  KLAYOUT_IMPLANT12_DIGEST_FIELD (max_flat_edges);
#undef KLAYOUT_IMPLANT12_DIGEST_FIELD

  std::size_t bytes = 0;
#define KLAYOUT_IMPLANT12_DIGEST_ARRAY(pointer, count, record_bytes) \
  do { \
    if (! db::cuda_active3_digest::checked_bytes ( \
          request.count, request.record_bytes, bytes)) return false; \
    if (bytes) sha.update (request.pointer, bytes); \
  } while (false)
#define KLAYOUT_IMPLANT12_DIGEST_TYPED_ARRAY(pointer, count, type) \
  do { \
    if (! db::cuda_active3_digest::checked_bytes ( \
          request.count, sizeof (type), bytes)) return false; \
    if (bytes) sha.update (request.pointer, bytes); \
  } while (false)
  KLAYOUT_IMPLANT12_DIGEST_ARRAY (
    contexts, context_count, context_record_bytes);
  KLAYOUT_IMPLANT12_DIGEST_TYPED_ARRAY (
    implant_contexts, implant_context_count, uint32_t);
  KLAYOUT_IMPLANT12_DIGEST_TYPED_ARRAY (
    implant_edge_offsets, implant_edge_offset_count, uint64_t);
  KLAYOUT_IMPLANT12_DIGEST_TYPED_ARRAY (
    gate_contexts, gate_context_count, uint32_t);
  KLAYOUT_IMPLANT12_DIGEST_TYPED_ARRAY (
    contact_contexts, contact_context_count, uint32_t);
  KLAYOUT_IMPLANT12_DIGEST_ARRAY (
    cells, cell_count, cell_record_bytes);
  KLAYOUT_IMPLANT12_DIGEST_ARRAY (
    contours, contour_count, contour_record_bytes);
  KLAYOUT_IMPLANT12_DIGEST_ARRAY (
    edges, edge_count, edge_record_bytes);
#undef KLAYOUT_IMPLANT12_DIGEST_TYPED_ARRAY
#undef KLAYOUT_IMPLANT12_DIGEST_ARRAY

  digest = sha.finish ();
  return true;
}

} // namespace cuda_implant12_digest
} // namespace db

#endif
