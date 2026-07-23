/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

/*
 * CPU-only adversarial DSO for the host-side VIA1-stack result validator.
 *
 * Select a response with KLAYOUT_CUDA_VIA1_STACK_FAKE_MODE.  "well-formed"
 * returns a complete, internally consistent certificate for the minimal
 * request used by via1_stack_host_guard_smoke.cc.  Every other documented
 * mode corrupts one independent part of that result.
 */

#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace
{

using Request = klayout_cuda_spatial_via1_stack_request_v1;
using Result = klayout_cuda_spatial_via1_stack_result_v1;

const char *fake_mode ()
{
  const char *mode = std::getenv ("KLAYOUT_CUDA_VIA1_STACK_FAKE_MODE");
  return mode && *mode ? mode : "well-formed";
}

void set_message (Result *result, const char *message)
{
  std::strncpy (result->message, message, sizeof (result->message) - 1);
  result->message [sizeof (result->message) - 1] = '\0';
}

uint64_t plus_one_saturated (uint64_t value)
{
  return value == std::numeric_limits<uint64_t>::max ()
    ? value
    : value + 1;
}

uint64_t product_saturated (uint64_t first, uint64_t second)
{
  if (first && second > std::numeric_limits<uint64_t>::max () / first) {
    return std::numeric_limits<uint64_t>::max ();
  }
  return first * second;
}

void fill_well_formed (const Request &request, Result *result)
{
  std::memset (result, 0, sizeof (*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof (*result);
  result->status = KLAYOUT_CUDA_SPATIAL_OK;
  result->disposition = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE;
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->requested_mask = request.requested_mask;
  result->certified_empty_mask = request.requested_mask;
  result->dbu_per_micron = request.dbu_per_micron;
  result->enclosure_distance = request.enclosure_distance;
  result->cut_width = request.cut_width;
  result->cut_height = request.cut_height;
  result->spacing_distance = request.spacing_distance;
  result->grid_cell_size = request.grid_cell_size;
  std::copy (
    request.scene_digest, request.scene_digest + sizeof (request.scene_digest),
    result->scene_digest);

  result->context_count = request.context_count;
  result->metal1_context_count = request.metal1_context_count;
  result->via1_context_count = request.via1_context_count;
  result->metal2_context_count = request.metal2_context_count;
  result->cell_count = request.cell_count;
  result->box_count = request.box_count;
  result->flat_metal1_box_count = request.flat_metal1_box_count;
  result->flat_via1_box_count = request.flat_via1_box_count;
  result->flat_metal2_box_count = request.flat_metal2_box_count;

  result->via_expanded_count = request.flat_via1_box_count;
  result->via_size_checked_count = request.flat_via1_box_count;
  result->metal1_expanded_count = request.flat_metal1_box_count;
  result->metal2_expanded_count = request.flat_metal2_box_count;
  result->grid_cell_count = 1;
  result->via_membership_count = request.flat_via1_box_count;
  result->metal1_membership_count = request.flat_metal1_box_count;
  result->metal2_membership_count = request.flat_metal2_box_count;
  result->via_pair_queried_count = request.flat_via1_box_count;
  result->metal1_queried_count = request.flat_via1_box_count;
  result->metal1_candidate_count = request.flat_via1_box_count;
  result->metal1_certified_count = request.flat_via1_box_count;
  result->metal2_queried_count = request.flat_via1_box_count;
  result->metal2_candidate_count = request.flat_via1_box_count;
  result->metal2_certified_count = request.flat_via1_box_count;
  set_message (result, "fake well-formed complete certificate");
}

} // anonymous namespace

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT uint32_t
klayout_cuda_spatial_abi_version (void)
{
  return KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_via1_stack_empty_v1 (
  const Request *request, Result *result)
{
  if (! request || ! result) {
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }

  fill_well_formed (*request, result);
  const std::string mode (fake_mode ());
  if (mode == "well-formed") {
    return KLAYOUT_CUDA_SPATIAL_OK;
  } else if (mode == "zero-work") {
    result->via_expanded_count = 0;
    result->via_size_checked_count = 0;
    result->metal1_expanded_count = 0;
    result->metal2_expanded_count = 0;
    result->grid_cell_count = 0;
    result->via_membership_count = 0;
    result->metal1_membership_count = 0;
    result->metal2_membership_count = 0;
    result->via_pair_queried_count = 0;
    result->metal1_queried_count = 0;
    result->metal1_candidate_count = 0;
    result->metal1_certified_count = 0;
    result->metal2_queried_count = 0;
    result->metal2_candidate_count = 0;
    result->metal2_certified_count = 0;
  } else if (mode == "impossible-via-candidate") {
    result->via_candidate_pair_count = 1;
    result->clean_via_pair_count = 1;
  } else if (mode == "impossible-metal1-candidate") {
    result->metal1_candidate_count = plus_one_saturated (
      product_saturated (
        request->flat_via1_box_count, request->flat_metal1_box_count));
  } else if (mode == "impossible-metal2-candidate") {
    result->metal2_candidate_count = plus_one_saturated (
      product_saturated (
        request->flat_via1_box_count, request->flat_metal2_box_count));
  } else if (mode == "mutated-echo") {
    result->opcode ^= 1u;
  } else if (mode == "mutated-digest") {
    result->scene_digest [0] ^= 0x80u;
  } else if (mode == "bad-result-abi") {
    result->abi_version += 1;
  } else if (mode == "short-result") {
    result->struct_size = sizeof (*result) - 1;
  } else if (mode == "reserved-result") {
    result->reserved0 = 1;
  } else if (mode == "fallback-flag") {
    result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
  } else if (mode == "device-flag") {
    result->device_flags = 1;
  } else if (mode == "partial-mask") {
    result->certified_empty_mask &=
      ~uint32_t (KLAYOUT_CUDA_SPATIAL_VIA1_2);
  } else if (mode == "bad-conservation") {
    result->via_size_checked_count -= 1;
  } else if (mode == "zero-grid") {
    result->grid_cell_count = 0;
  } else if (mode == "result-error") {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
  } else if (mode == "return-error") {
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  } else {
    result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
    result->disposition = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_UNCERTAIN;
    result->certified_empty_mask = 0;
    set_message (result, "unknown fake-backend mode");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }

  set_message (result, mode.c_str ());
  return KLAYOUT_CUDA_SPATIAL_OK;
}
