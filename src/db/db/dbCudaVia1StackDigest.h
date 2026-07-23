/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef HDR_dbCudaVia1StackDigest
#define HDR_dbCudaVia1StackDigest

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace db
{
namespace cuda_via1_stack_digest
{

inline bool request_digest (
  const klayout_cuda_spatial_via1_stack_request_v1 &request,
  std::array<std::uint8_t, 32> &digest)
{
  if ((request.context_count && ! request.contexts) ||
      (request.metal1_context_count && ! request.metal1_contexts) ||
      (request.metal1_offset_count && ! request.metal1_offsets) ||
      (request.via1_context_count && ! request.via1_contexts) ||
      (request.via1_offset_count && ! request.via1_offsets) ||
      (request.metal2_context_count && ! request.metal2_contexts) ||
      (request.metal2_offset_count && ! request.metal2_offsets) ||
      (request.cell_count && ! request.cells) ||
      (request.box_count && ! request.boxes)) {
    return false;
  }

  static const char magic [8] =
    { 'K', 'A', 'C', 'T', 'V', 'I', 'A', '1' };
  db::cuda_active3_digest::Sha256 sha;
  sha.update (magic, sizeof (magic));
#define KLAYOUT_VIA1_STACK_DIGEST_FIELD(field) \
  sha.update (&request.field, sizeof (request.field))
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (abi_version);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (struct_size);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (opcode);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (option_flags);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (requested_mask);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (dbu_per_micron);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (device);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (enclosure_distance);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (cut_width);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (cut_height);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (spacing_distance);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (grid_cell_size);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (context_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (metal1_context_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (metal1_offset_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (via1_context_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (via1_offset_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (metal2_context_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (metal2_offset_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (cell_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (box_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (flat_metal1_box_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (flat_via1_box_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (flat_metal2_box_count);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (scene_left);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (scene_bottom);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (scene_right);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (scene_top);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (max_contexts);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (max_grid_cells);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (max_metal_memberships);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (max_via_memberships);
  KLAYOUT_VIA1_STACK_DIGEST_FIELD (max_pair_work);
#undef KLAYOUT_VIA1_STACK_DIGEST_FIELD

  std::size_t bytes = 0;
#define KLAYOUT_VIA1_STACK_DIGEST_ARRAY(pointer, count, type) \
  do { \
    if (! db::cuda_active3_digest::checked_bytes ( \
          request.count, sizeof (type), bytes)) return false; \
    if (bytes) sha.update (request.pointer, bytes); \
  } while (false)
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    contexts, context_count, klayout_cuda_spatial_via1_stack_context_v1);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    metal1_contexts, metal1_context_count, uint32_t);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    metal1_offsets, metal1_offset_count, uint64_t);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    via1_contexts, via1_context_count, uint32_t);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    via1_offsets, via1_offset_count, uint64_t);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    metal2_contexts, metal2_context_count, uint32_t);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    metal2_offsets, metal2_offset_count, uint64_t);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    cells, cell_count, klayout_cuda_spatial_via1_stack_cell_v1);
  KLAYOUT_VIA1_STACK_DIGEST_ARRAY (
    boxes, box_count, klayout_cuda_spatial_via1_stack_box_v1);
#undef KLAYOUT_VIA1_STACK_DIGEST_ARRAY

  digest = sha.finish ();
  return true;
}

} // namespace cuda_via1_stack_digest
} // namespace db

#endif
