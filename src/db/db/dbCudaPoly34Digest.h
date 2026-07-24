/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaPoly34Digest
#define HDR_dbCudaPoly34Digest

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace db
{
namespace cuda_poly34_digest
{

inline bool request_digest (
  const klayout_cuda_spatial_poly34_request_v1 &request,
  std::array<std::uint8_t, 32> &digest)
{
  if (request.context_record_bytes !=
        sizeof (klayout_cuda_spatial_poly34_context_v1) ||
      request.cell_record_bytes !=
        sizeof (klayout_cuda_spatial_poly34_cell_v1) ||
      request.box_record_bytes !=
        sizeof (klayout_cuda_spatial_poly34_box_v1) ||
      (request.context_count && ! request.contexts) ||
      (request.poly_context_count && ! request.poly_contexts) ||
      (request.poly_offset_count && ! request.poly_offsets) ||
      (request.active_context_count && ! request.active_contexts) ||
      (request.active_offset_count && ! request.active_offsets) ||
      (request.gate_context_count && ! request.gate_contexts) ||
      (request.gate_offset_count && ! request.gate_offsets) ||
      (request.cell_count && ! request.cells) ||
      (request.box_count && ! request.boxes)) {
    return false;
  }

  static const char magic [16] = {
    'K', 'L', 'P', 'O', 'L', 'Y', '3', '4',
    'V', '1', 0, 0, 0, 0, 0, 0
  };
  db::cuda_active3_digest::Sha256 sha;
  sha.update (magic, sizeof (magic));

#define KLAYOUT_POLY34_DIGEST_FIELD(field) \
  sha.update (&request.field, sizeof (request.field))
  KLAYOUT_POLY34_DIGEST_FIELD (abi_version);
  KLAYOUT_POLY34_DIGEST_FIELD (struct_size);
  KLAYOUT_POLY34_DIGEST_FIELD (opcode);
  KLAYOUT_POLY34_DIGEST_FIELD (option_flags);
  KLAYOUT_POLY34_DIGEST_FIELD (format_version);
  KLAYOUT_POLY34_DIGEST_FIELD (dbu_per_micron);
  KLAYOUT_POLY34_DIGEST_FIELD (root_cell);
  KLAYOUT_POLY34_DIGEST_FIELD (requested_mask);
  KLAYOUT_POLY34_DIGEST_FIELD (device);
  KLAYOUT_POLY34_DIGEST_FIELD (poly3_distance);
  KLAYOUT_POLY34_DIGEST_FIELD (poly4_distance);
  KLAYOUT_POLY34_DIGEST_FIELD (grid_cell_size);
  KLAYOUT_POLY34_DIGEST_FIELD (store_identity);
  KLAYOUT_POLY34_DIGEST_FIELD (layout_identity);
  KLAYOUT_POLY34_DIGEST_FIELD (top_cell_identity);
  KLAYOUT_POLY34_DIGEST_FIELD (poly_layer_id);
  KLAYOUT_POLY34_DIGEST_FIELD (active_layer_id);
  KLAYOUT_POLY34_DIGEST_FIELD (gate_layer_id);
  KLAYOUT_POLY34_DIGEST_FIELD (context_count);
  KLAYOUT_POLY34_DIGEST_FIELD (context_record_bytes);
  KLAYOUT_POLY34_DIGEST_FIELD (poly_context_count);
  KLAYOUT_POLY34_DIGEST_FIELD (poly_offset_count);
  KLAYOUT_POLY34_DIGEST_FIELD (active_context_count);
  KLAYOUT_POLY34_DIGEST_FIELD (active_offset_count);
  KLAYOUT_POLY34_DIGEST_FIELD (gate_context_count);
  KLAYOUT_POLY34_DIGEST_FIELD (gate_offset_count);
  KLAYOUT_POLY34_DIGEST_FIELD (cell_count);
  KLAYOUT_POLY34_DIGEST_FIELD (cell_record_bytes);
  KLAYOUT_POLY34_DIGEST_FIELD (box_count);
  KLAYOUT_POLY34_DIGEST_FIELD (box_record_bytes);
  KLAYOUT_POLY34_DIGEST_FIELD (flat_poly_box_count);
  KLAYOUT_POLY34_DIGEST_FIELD (flat_active_box_count);
  KLAYOUT_POLY34_DIGEST_FIELD (flat_gate_box_count);
  KLAYOUT_POLY34_DIGEST_FIELD (scene_left);
  KLAYOUT_POLY34_DIGEST_FIELD (scene_bottom);
  KLAYOUT_POLY34_DIGEST_FIELD (scene_right);
  KLAYOUT_POLY34_DIGEST_FIELD (scene_top);
  KLAYOUT_POLY34_DIGEST_FIELD (max_contexts);
  KLAYOUT_POLY34_DIGEST_FIELD (max_flat_boxes);
  KLAYOUT_POLY34_DIGEST_FIELD (max_grid_cells);
  KLAYOUT_POLY34_DIGEST_FIELD (max_poly_memberships);
  KLAYOUT_POLY34_DIGEST_FIELD (max_active_memberships);
  KLAYOUT_POLY34_DIGEST_FIELD (max_query_visits);
  KLAYOUT_POLY34_DIGEST_FIELD (max_candidate_work);
  KLAYOUT_POLY34_DIGEST_FIELD (max_candidates_per_gate);
#undef KLAYOUT_POLY34_DIGEST_FIELD

  std::size_t bytes = 0;
#define KLAYOUT_POLY34_DIGEST_ARRAY(pointer, count, record_bytes) \
  do { \
    if (! db::cuda_active3_digest::checked_bytes ( \
          request.count, request.record_bytes, bytes)) return false; \
    if (bytes) sha.update (request.pointer, bytes); \
  } while (false)
#define KLAYOUT_POLY34_DIGEST_TYPED_ARRAY(pointer, count, type) \
  do { \
    if (! db::cuda_active3_digest::checked_bytes ( \
          request.count, sizeof (type), bytes)) return false; \
    if (bytes) sha.update (request.pointer, bytes); \
  } while (false)
  KLAYOUT_POLY34_DIGEST_ARRAY (
    contexts, context_count, context_record_bytes);
  KLAYOUT_POLY34_DIGEST_TYPED_ARRAY (
    poly_contexts, poly_context_count, uint32_t);
  KLAYOUT_POLY34_DIGEST_TYPED_ARRAY (
    poly_offsets, poly_offset_count, uint64_t);
  KLAYOUT_POLY34_DIGEST_TYPED_ARRAY (
    active_contexts, active_context_count, uint32_t);
  KLAYOUT_POLY34_DIGEST_TYPED_ARRAY (
    active_offsets, active_offset_count, uint64_t);
  KLAYOUT_POLY34_DIGEST_TYPED_ARRAY (
    gate_contexts, gate_context_count, uint32_t);
  KLAYOUT_POLY34_DIGEST_TYPED_ARRAY (
    gate_offsets, gate_offset_count, uint64_t);
  KLAYOUT_POLY34_DIGEST_ARRAY (cells, cell_count, cell_record_bytes);
  KLAYOUT_POLY34_DIGEST_ARRAY (boxes, box_count, box_record_bytes);
#undef KLAYOUT_POLY34_DIGEST_TYPED_ARRAY
#undef KLAYOUT_POLY34_DIGEST_ARRAY

  digest = sha.finish ();
  return true;
}

} // namespace cuda_poly34_digest
} // namespace db

#endif
