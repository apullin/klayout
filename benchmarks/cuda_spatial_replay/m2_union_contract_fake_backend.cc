/*
 * Adversarial ABI-v1 backend for the host-side M2 union contract gate.
 *
 * Build variants deliberately omit either run or release so the host test can
 * prove that it checks the complete capability before constructing a scene.
 * The full variant returns controlled valid and malformed results selected by
 * KLAYOUT_CUDA_M2_UNION_FAKE_MODE.
 */

#include "dbCudaSpatialApi.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifndef KLAYOUT_M2_UNION_FAKE_WITH_RUN
#  define KLAYOUT_M2_UNION_FAKE_WITH_RUN 1
#endif

#ifndef KLAYOUT_M2_UNION_FAKE_WITH_RELEASE
#  define KLAYOUT_M2_UNION_FAKE_WITH_RELEASE 1
#endif

namespace
{

typedef klayout_cuda_spatial_m2_union_segment_v1 Segment;

std::atomic<int> s_run_count (0);
std::atomic<int> s_release_count (0);

const Segment s_rectangle [] = {
  { 0, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
  { 20, 0, 10, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
  { 0, 0, 20, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL },
  { 10, 0, 20, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL }
};

const Segment s_wrong_order [] = {
  s_rectangle [1], s_rectangle [0], s_rectangle [2], s_rectangle [3]
};

const Segment s_side_before_fixed [] = {
  { 100, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
  { 0, 0, 10, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL }
};

uint64_t boundary_fnv64 (const Segment *segments, uint64_t segment_count)
{
  uint64_t hash = UINT64_C (1469598103934665603);
  const auto mix = [&hash] (uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C (0xff);
      hash *= UINT64_C (1099511628211);
    }
  };
  mix (segment_count);
  for (uint64_t index = 0; index < segment_count; ++index) {
    const Segment &segment = segments [index];
    mix (uint64_t (segment.axis));
    mix (uint64_t (uint32_t (segment.side)));
    mix (uint64_t (segment.fixed));
    mix (uint64_t (segment.lo));
    mix (uint64_t (segment.hi));
  }
  return hash;
}

void fill_echo (
  const klayout_cuda_spatial_m2_union_request_v1 &request,
  klayout_cuda_spatial_m2_union_result_v1 &result)
{
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_OK;
  result.disposition = KLAYOUT_CUDA_SPATIAL_M2_UNION_COMPLETE;
  result.opcode = request.opcode;
  result.option_flags = request.option_flags;
  result.format_version = request.format_version;
  result.dbu_per_micron = request.dbu_per_micron;
  result.root_cell = request.root_cell;
  result.segment_record_bytes = sizeof (Segment);
  std::memcpy (result.scene_digest, request.scene_digest, 32);
  result.context_count = request.context_count;
  result.metal_context_count = request.metal_context_count;
  result.cell_count = request.cell_count;
  result.polygon_count = request.polygon_count;
  result.edge_count = request.edge_count;
  result.flat_polygon_count = request.flat_polygon_count;
  result.flat_edge_count = request.flat_edge_count;
  result.rectangle_count = request.flat_polygon_count;
  result.x_slab_count = 1;
  result.membership_count = 1;
  result.event_count = 2;
  result.strip_interval_count = 1;
  result.raw_segment_count = 4;
  result.segments = s_rectangle;
  result.segment_count = 4;
  result.boundary_fnv64 = boundary_fnv64 (s_rectangle, 4);
}

} // anonymous namespace

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT uint32_t
klayout_cuda_spatial_abi_version (void)
{
  return KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_m2_union_fake_run_count (void)
{
  return s_run_count.load ();
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_m2_union_fake_release_count (void)
{
  return s_release_count.load ();
}

#if KLAYOUT_M2_UNION_FAKE_WITH_RUN
extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m2_union_boundary_v1 (
  const klayout_cuda_spatial_m2_union_request_v1 *request,
  klayout_cuda_spatial_m2_union_result_v1 *result)
{
  ++s_run_count;
  if (! request || ! result) {
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }

  fill_echo (*request, *result);
  const char *mode = std::getenv ("KLAYOUT_CUDA_M2_UNION_FAKE_MODE");
  mode = mode ? mode : "ok";

  if (std::strcmp (mode, "backend_throw") == 0) {
    throw std::runtime_error ("synthetic M2 union backend exception");
  }
  if (boundary_fnv64 (s_rectangle, 4) !=
        UINT64_C (11447980897846940057) ||
      boundary_fnv64 (s_side_before_fixed, 2) !=
        UINT64_C (11131890132215870808)) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
  if (std::strcmp (mode, "live_caps") == 0 &&
      (request->max_contexts != UINT64_C (4000000) ||
       request->max_rectangles != UINT64_C (32000000) ||
       request->max_x_slabs != UINT64_C (32000000) ||
       request->max_memberships != UINT64_C (100000000) ||
       request->max_events != UINT64_C (200000000) ||
       request->max_raw_segments != UINT64_C (12000000) ||
       request->max_segments != UINT64_C (8000000) ||
       request->max_slabs_per_rectangle != 64)) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    std::strncpy (
      result->message, "live host used unexpected M2 union capacities",
      sizeof (result->message) - 1);
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
  if (std::strcmp (mode, "fallback") == 0) {
    result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
    result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    result->disposition = KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN;
    result->segments = 0;
    result->segment_count = 0;
    return KLAYOUT_CUDA_SPATIAL_FALLBACK;
  }
  if (std::strcmp (mode, "bad_echo") == 0) {
    ++result->root_cell;
  } else if (std::strcmp (mode, "side_order") == 0) {
    result->segments = s_side_before_fixed;
    result->segment_count = 2;
    result->boundary_fnv64 = boundary_fnv64 (s_side_before_fixed, 2);
  } else if (std::strcmp (mode, "bad_order") == 0) {
    result->segments = s_wrong_order;
    result->boundary_fnv64 = boundary_fnv64 (s_wrong_order, 4);
  } else if (std::strcmp (mode, "bad_digest") == 0) {
    ++result->boundary_fnv64;
  } else if (std::strcmp (mode, "bad_count") == 0) {
    result->rectangle_count = request->max_rectangles + 1;
  } else if (std::strcmp (mode, "copy_throw") == 0) {
    /*
     * The harness sets max_segments one above vector::max_size().  resize()
     * must throw before dereferencing this intentionally one-record buffer;
     * the host release guard must still call the dedicated release exactly
     * once.
     */
    result->segments = s_rectangle;
    result->segment_count = request->max_segments;
    result->raw_segment_count = request->max_segments;
    result->boundary_fnv64 = 0;
  }
  return KLAYOUT_CUDA_SPATIAL_OK;
}
#endif

#if KLAYOUT_M2_UNION_FAKE_WITH_RELEASE
extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT void
klayout_cuda_spatial_release_m2_union_boundary_v1 (
  klayout_cuda_spatial_m2_union_result_v1 *result)
{
  ++s_release_count;
  if (result) {
    result->segments = 0;
    result->segment_count = 0;
  }
}
#endif
