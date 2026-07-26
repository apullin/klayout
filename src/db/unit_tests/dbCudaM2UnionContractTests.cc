/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaSpatialBackend.h"
#include "tlUnitTest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace
{

typedef klayout_cuda_spatial_m2_union_segment_v1 Segment;
typedef klayout_cuda_spatial_m2_union_result_v1 Result;

static_assert (sizeof (Segment) == 32, "M2 boundary ABI size changed");
static_assert (offsetof (Segment, fixed) == 0, "M2 fixed offset changed");
static_assert (offsetof (Segment, lo) == 8, "M2 lo offset changed");
static_assert (offsetof (Segment, hi) == 16, "M2 hi offset changed");
static_assert (offsetof (Segment, side) == 24, "M2 side offset changed");
static_assert (offsetof (Segment, axis) == 28, "M2 axis offset changed");
static_assert (sizeof (Result) == 480, "M2 result ABI size changed");
static_assert (
  offsetof (Result, certified_empty_mask) == 272,
  "M2 suffix mask offset changed");
static_assert (
  offsetof (Result, certificate_reserved) == 276,
  "M2 suffix reserved offset changed");
static_assert (
  offsetof (Result, suffix_total_ns) == 280,
  "M2 suffix timing offset changed");
static_assert (
  offsetof (Result, message) == 288,
  "M2 result message offset changed");
static_assert (
  KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY == 31,
  "M2 suffix complete mask changed");
static_assert (
  sizeof (db::CudaM2SuffixCertificate) == 24,
  "M2 host suffix certificate size changed");

struct M1MorphologyContractFixture
{
  klayout_cuda_spatial_m1_width_space_context_v1 context;
  uint32_t metal_context;
  uint64_t polygon_offset;
  uint64_t edge_offset;
  klayout_cuda_spatial_m1_width_space_cell_v1 cell;
  klayout_cuda_spatial_m1_width_space_polygon_v1 polygon;
  klayout_cuda_spatial_m1_width_space_edge_v1 edges [4];
  klayout_cuda_spatial_m1_resident_morphology_request_v1 request;
  klayout_cuda_spatial_m1_resident_morphology_result_v1 result;

  M1MorphologyContractFixture ()
    : context { 0, 0, 0, 0 }, metal_context (0), polygon_offset (0),
      edge_offset (0), cell { 0, 0, 0, 1, 4 },
      polygon { 0, 0, 0, 100, 100, 0, 4 },
      edges {
        { 0, 0, 0, 100 },
        { 0, 100, 100, 100 },
        { 100, 100, 100, 0 },
        { 100, 0, 0, 0 }
      }
  {
    std::memset (&request, 0, sizeof (request));
    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof (request);
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M15_9_EMPTY;
    request.option_flags =
      KLAYOUT_CUDA_SPATIAL_M1_MORPH_QUALIFIED_OPTIONS;
    request.format_version = 1;
    request.dbu_per_micron = 2000;
    request.root_cell = 0;
    request.device = 0;
    request.requested_mask =
      KLAYOUT_CUDA_SPATIAL_M1_MORPH_ALL_EMPTY;
    request.contexts = &context;
    request.context_count = 1;
    request.context_record_bytes = sizeof (context);
    request.metal_contexts = &metal_context;
    request.metal_context_count = 1;
    request.context_polygon_offsets = &polygon_offset;
    request.context_polygon_offset_count = 1;
    request.context_edge_offsets = &edge_offset;
    request.context_edge_offset_count = 1;
    request.cells = &cell;
    request.cell_count = 1;
    request.cell_record_bytes = sizeof (cell);
    request.polygons = &polygon;
    request.polygon_count = 1;
    request.polygon_record_bytes = sizeof (polygon);
    request.edges = edges;
    request.edge_count = 4;
    request.edge_record_bytes = sizeof (edges [0]);
    request.flat_polygon_count = 1;
    request.flat_edge_count = 4;
    request.scene_left = 0;
    request.scene_bottom = 0;
    request.scene_right = 100;
    request.scene_top = 100;
    request.max_contexts = 4;
    request.max_rectangles = 8;
    request.max_x_slabs = 16;
    request.max_union_memberships = 32;
    request.max_union_events = 64;
    request.max_union_raw_segments = 64;
    request.max_union_segments = 64;
    request.max_slabs_per_rectangle = 8;
    request.max_morph_output_slabs = 16;
    request.max_morph_output_intervals = 64;
    request.max_morph_raw_boundary_segments = 128;
    request.max_morph_boundary_segments = 64;
    request.max_morph_source_visits_per_pass = 1000;
    request.max_morph_source_visits_per_band = 100;
    request.max_morph_long_segments = 64;
    request.max_morph_active_slabs = 8;
    std::memset (
      request.scene_digest, 0x4d, sizeof (request.scene_digest));

    std::memset (&result, 0, sizeof (result));
    result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    result.struct_size = sizeof (result);
    result.status = KLAYOUT_CUDA_SPATIAL_OK;
    result.disposition = KLAYOUT_CUDA_SPATIAL_M1_MORPH_COMPLETE;
    result.opcode = request.opcode;
    result.option_flags = request.option_flags;
    result.format_version = request.format_version;
    result.dbu_per_micron = request.dbu_per_micron;
    result.root_cell = request.root_cell;
    result.requested_mask = request.requested_mask;
    result.certified_empty_mask = request.requested_mask;
    result.scene_left = request.scene_left;
    result.scene_bottom = request.scene_bottom;
    result.scene_right = request.scene_right;
    result.scene_top = request.scene_top;
    std::memcpy (
      result.scene_digest, request.scene_digest,
      sizeof (result.scene_digest));
    result.context_count = request.context_count;
    result.metal_context_count = request.metal_context_count;
    result.cell_count = request.cell_count;
    result.polygon_count = request.polygon_count;
    result.edge_count = request.edge_count;
    result.flat_polygon_count = request.flat_polygon_count;
    result.flat_edge_count = request.flat_edge_count;
    result.max_contexts = request.max_contexts;
    result.max_rectangles = request.max_rectangles;
    result.max_x_slabs = request.max_x_slabs;
    result.max_union_memberships = request.max_union_memberships;
    result.max_union_events = request.max_union_events;
    result.max_union_raw_segments = request.max_union_raw_segments;
    result.max_union_segments = request.max_union_segments;
    result.max_slabs_per_rectangle =
      request.max_slabs_per_rectangle;
    result.max_morph_output_slabs =
      request.max_morph_output_slabs;
    result.max_morph_output_intervals =
      request.max_morph_output_intervals;
    result.max_morph_raw_boundary_segments =
      request.max_morph_raw_boundary_segments;
    result.max_morph_boundary_segments =
      request.max_morph_boundary_segments;
    result.max_morph_source_visits_per_pass =
      request.max_morph_source_visits_per_pass;
    result.max_morph_source_visits_per_band =
      request.max_morph_source_visits_per_band;
    result.max_morph_long_segments =
      request.max_morph_long_segments;
    result.max_morph_active_slabs =
      request.max_morph_active_slabs;

    result.rectangle_count = 1;
    result.x_slab_count = 2;
    result.union_membership_count = 1;
    result.union_event_count = 2;
    result.strip_interval_count = 1;
    result.erode89_output_interval_count = 1;
    result.erode89_source_visit_count = 800;
    result.dilate90_output_interval_count = 1;
    result.dilate90_source_visit_count = 900;
    result.boundary_source_visit_count = 1000;
    result.erode269_source_visit_count = 950;
    result.f90_boundary_segment_count = 4;
    //  This bounded long-segment subset is copied internally for the exact
    //  predicate.  It is intentionally not result-geometry D2H.
    result.f90_long_segment_count = 3;
    result.f90_space_pair_count = 3;
    result.union_device_total_bytes = 1000;
    result.union_device_free_begin_bytes = 900;
    result.union_device_free_low_bytes = 800;
    result.morph_device_total_bytes = 1000;
    result.morph_device_free_begin_bytes = 800;
    result.morph_device_free_low_bytes = 700;
    result.setup_ns = 5;
    result.h2d_ns = 5;
    result.rectangle_expand_ns = 10;
    result.x_membership_ns = 10;
    result.strip_scan_ns = 10;
    result.morphology_ns = 50;
    result.d2h_ns = 0;
    result.total_ns = 100;
  }

  void select_base_width_space ()
  {
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M11_2_EMPTY;
    request.requested_mask =
      KLAYOUT_CUDA_SPATIAL_M1_BASE_ALL_EMPTY;
    result.opcode = request.opcode;
    result.requested_mask = request.requested_mask;
    result.certified_empty_mask = request.requested_mask;
    result.erode89_output_interval_count = 0;
    result.erode89_source_visit_count = 0;
    result.dilate90_output_interval_count = 0;
    result.dilate90_source_visit_count = 0;
    result.boundary_source_visit_count = 0;
    result.erode269_source_visit_count = 0;
    result.f90_boundary_segment_count = 0;
    result.f90_long_segment_count = 0;
    result.f90_space_pair_count = 0;
    result.f90_space_violation_count = 0;
    result.f90_space_uncertain_count = 0;
    result.f270_eroded_interval_count = 0;
    result.morph_device_total_bytes = 0;
    result.morph_device_free_begin_bytes = 0;
    result.morph_device_free_low_bytes = 0;
  }
};

static_assert (
  sizeof (klayout_cuda_spatial_m1_resident_morphology_request_v1) == 408,
  "M1 resident morphology request ABI size changed");
static_assert (
  sizeof (klayout_cuda_spatial_m1_resident_morphology_result_v1) == 744,
  "M1 resident morphology result ABI size changed");

} // anonymous namespace

TEST(1_CanonicalOrderAndDigest)
{
  const Segment rectangle [] = {
    { 0, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 20, 0, 10, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 0, 0, 20, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL },
    { 10, 0, 20, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL }
  };
  const uint64_t rectangle_fnv64 = UINT64_C (11447980897846940057);
  std::string error;
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      rectangle, 4, rectangle_fnv64, &error),
    true);
  EXPECT_EQ (error, "");

  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      rectangle, 4, rectangle_fnv64 + 1, &error),
    false);
  EXPECT_EQ (error, "M2 boundary FNV-1a digest mismatch");

  //  This is ordered by the proven (axis, side, fixed, lo, hi) contract even
  //  though its fixed coordinates decrease.  Sorting fixed before side would
  //  incorrectly reject it.
  const Segment side_before_fixed [] = {
    { 100, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 0, 0, 10, 1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL }
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      side_before_fixed, 2, UINT64_C (11131890132215870808), &error),
    true);

  const Segment fixed_before_side [] = {
    side_before_fixed [1], side_before_fixed [0]
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      fixed_before_side, 2, 0, &error),
    false);
  EXPECT_EQ (
    error,
    "M2 boundary is not strictly ordered by (axis,side,fixed,lo,hi)");
}

TEST(2_NonmaximalAndMalformedFailClosed)
{
  std::string error;
  const Segment touching [] = {
    { 0, 0, 10, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL },
    { 0, 10, 20, -1, KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL }
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      touching, 2, 0, &error),
    false);
  EXPECT_EQ (
    error,
    "M2 boundary has nonmaximal touching or overlapping segments");

  const Segment bad_axis [] = {
    { 0, 0, 10, -1, 2 }
  };
  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      bad_axis, 1, 0, &error),
    false);
  EXPECT_EQ (error, "M2 boundary has an invalid segment");

  EXPECT_EQ (
    db::cuda_spatial_validate_m2_union_boundary (
      0, 1, 0, &error),
    false);
  EXPECT_EQ (
    error, "nonempty M2 boundary has a null segment pointer");
}

TEST(10_M1MorphologyQualifiedCompleteProof)
{
  M1MorphologyContractFixture fixture;
  std::string error;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      fixture.request, fixture.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    true);
  EXPECT_EQ (error, "");

  //  The reusable engine applies the source-visit budget independently to
  //  each pass.  The overlapping sum is deliberately much greater than the
  //  per-pass budget and must not invalidate the proof.
  EXPECT_EQ (
    fixture.result.erode89_source_visit_count +
      fixture.result.dilate90_source_visit_count +
      fixture.result.boundary_source_visit_count +
      fixture.result.erode269_source_visit_count >
        fixture.request.max_morph_source_visits_per_pass,
    true);
}

TEST(11_M1MorphologyAbiAndStatusFailClosed)
{
  std::string error;

  M1MorphologyContractFixture bad_abi;
  ++bad_abi.result.abi_version;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_abi.request, bad_abi.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned an incompatible result");

  M1MorphologyContractFixture bad_size;
  --bad_size.result.struct_size;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_size.request, bad_size.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned an incompatible result");

  M1MorphologyContractFixture inconsistent_status;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      inconsistent_status.request, inconsistent_status.result,
      KLAYOUT_CUDA_SPATIAL_FALLBACK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned inconsistent statuses");

  M1MorphologyContractFixture no_proof;
  no_proof.result.status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      no_proof.request, no_proof.result,
      KLAYOUT_CUDA_SPATIAL_FALLBACK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend did not return a proof");
}

TEST(12_M1MorphologyEchoDigestCountCapAndMaskFailClosed)
{
  std::string error;

  M1MorphologyContractFixture bad_echo;
  ++bad_echo.result.opcode;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_echo.request, bad_echo.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned a mismatched proof echo");

  M1MorphologyContractFixture bad_digest;
  ++bad_digest.result.scene_digest [17];
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_digest.request, bad_digest.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned a mismatched proof echo");

  M1MorphologyContractFixture bad_count_echo;
  ++bad_count_echo.result.flat_edge_count;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_count_echo.request, bad_count_echo.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned a mismatched proof echo");

  M1MorphologyContractFixture bad_cap_echo;
  ++bad_cap_echo.result.max_morph_boundary_segments;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_cap_echo.request, bad_cap_echo.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned a mismatched proof echo");

  M1MorphologyContractFixture extra_mask;
  extra_mask.result.certified_empty_mask |= 1u << 5;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      extra_mask.request, extra_mask.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned a mismatched proof echo");

  M1MorphologyContractFixture missing_mask;
  missing_mask.result.certified_empty_mask &=
    ~uint32_t (KLAYOUT_CUDA_SPATIAL_M1_MORPH_M1_9_EMPTY);
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      missing_mask.request, missing_mask.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned an inconsistent "
    "disposition");
}

TEST(13_M1MorphologyCounterAndPerPassCapacityFailures)
{
  std::string error;

  M1MorphologyContractFixture bad_events;
  ++bad_events.result.union_event_count;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_events.request, bad_events.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible proof "
    "counters");

  M1MorphologyContractFixture bad_pairs;
  ++bad_pairs.result.f90_space_pair_count;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_pairs.request, bad_pairs.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible proof "
    "counters");

  M1MorphologyContractFixture overlapping_pair_outcomes;
  overlapping_pair_outcomes.result.f90_space_pair_count = 1;
  overlapping_pair_outcomes.result.f90_long_segment_count = 2;
  overlapping_pair_outcomes.result.f90_space_violation_count = 1;
  overlapping_pair_outcomes.result.f90_space_uncertain_count = 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      overlapping_pair_outcomes.request,
      overlapping_pair_outcomes.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible proof "
    "counters");

  M1MorphologyContractFixture one_pass_over_cap;
  one_pass_over_cap.result.erode269_source_visit_count =
    one_pass_over_cap.request.max_morph_source_visits_per_pass + 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      one_pass_over_cap.request, one_pass_over_cap.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible proof "
    "counters");
}

TEST(14_M1MorphologyDispositionAndD2hScopeFailClosed)
{
  std::string error;

  M1MorphologyContractFixture positive_complete;
  positive_complete.result.f90_space_violation_count = 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      positive_complete.request, positive_complete.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned an inconsistent "
    "disposition");

  M1MorphologyContractFixture valid_positive;
  valid_positive.result.disposition =
    KLAYOUT_CUDA_SPATIAL_M1_MORPH_NOT_EMPTY;
  valid_positive.result.f90_space_violation_count = 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      valid_positive.request, valid_positive.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    true);

  M1MorphologyContractFixture invalid_disposition;
  invalid_disposition.result.disposition = 99;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      invalid_disposition.request, invalid_disposition.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned an inconsistent "
    "disposition");

  //  A nonempty long-segment census is valid (the internal bounded subset may
  //  cross to the host), but COMPLETE may not claim result-geometry D2H.
  M1MorphologyContractFixture result_geometry_d2h;
  result_geometry_d2h.result.d2h_ns = 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      result_geometry_d2h.request, result_geometry_d2h.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned an inconsistent "
    "disposition");
}

TEST(15_M1MorphologyMemoryAndTimingFailClosed)
{
  std::string error;

  M1MorphologyContractFixture bad_memory_order;
  bad_memory_order.result.morph_device_free_low_bytes =
    bad_memory_order.result.morph_device_free_begin_bytes + 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_memory_order.request, bad_memory_order.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible memory "
    "telemetry");

  M1MorphologyContractFixture mismatched_device;
  ++mismatched_device.result.morph_device_total_bytes;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      mismatched_device.request, mismatched_device.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible memory "
    "telemetry");

  M1MorphologyContractFixture zero_free_memory;
  zero_free_memory.result.union_device_free_low_bytes = 0;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      zero_free_memory.request, zero_free_memory.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible memory "
    "telemetry");

  M1MorphologyContractFixture bad_component_time;
  bad_component_time.result.morphology_ns =
    bad_component_time.result.total_ns + 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      bad_component_time.request, bad_component_time.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible timing "
    "telemetry");

  M1MorphologyContractFixture zero_total_time;
  zero_total_time.result.total_ns = 0;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      zero_total_time.request, zero_total_time.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 resident morphology backend returned impossible timing "
    "telemetry");
}

TEST(16_M1BaseWidthSpaceOpcodeAndDisposition)
{
  std::string error;

  M1MorphologyContractFixture complete;
  complete.select_base_width_space ();
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      complete.request, complete.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    true);
  EXPECT_EQ (error, "");

  M1MorphologyContractFixture positive;
  positive.select_base_width_space ();
  positive.result.disposition =
    KLAYOUT_CUDA_SPATIAL_M1_MORPH_NOT_EMPTY;
  positive.result.certified_empty_mask = 0;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      positive.request, positive.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    true);

  M1MorphologyContractFixture suffix_counter;
  suffix_counter.select_base_width_space ();
  suffix_counter.result.f90_boundary_segment_count = 1;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      suffix_counter.request, suffix_counter.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "CUDA M1 base width/space backend returned suffix counters");

  M1MorphologyContractFixture wrong_mask;
  wrong_mask.select_base_width_space ();
  wrong_mask.request.requested_mask =
    KLAYOUT_CUDA_SPATIAL_M1_MORPH_ALL_EMPTY;
  wrong_mask.result.requested_mask =
    wrong_mask.request.requested_mask;
  wrong_mask.result.certified_empty_mask =
    wrong_mask.request.requested_mask;
  EXPECT_EQ (
    db::cuda_spatial_validate_m1_resident_morphology_result (
      wrong_mask.request, wrong_mask.result,
      KLAYOUT_CUDA_SPATIAL_OK, &error),
    false);
  EXPECT_EQ (
    error,
    "host supplied an unqualified M1 resident morphology request");
}
