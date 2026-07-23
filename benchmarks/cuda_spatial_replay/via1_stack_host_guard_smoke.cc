/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

/*
 * CPU-only smoke test for the host consumer of atomic VIA1-stack results.
 *
 * The companion via1_stack_fake_backend.cc DSO is selected before KLayout's
 * process-global backend loader is first touched.  The DSO reads its mode for
 * every request, allowing this executable to exercise many result corruptions
 * without reloading the host library.
 */

#include "dbCudaSpatialBackend.h"
#include "dbCudaVia1StackDigest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Attempt = db::CudaVia1StackAttempt;
using Box = klayout_cuda_spatial_via1_stack_box_v1;
using Cell = klayout_cuda_spatial_via1_stack_cell_v1;
using Context = klayout_cuda_spatial_via1_stack_context_v1;
using Request = klayout_cuda_spatial_via1_stack_request_v1;

void require (bool condition, const std::string &message)
{
  if (! condition) {
    throw std::runtime_error (message);
  }
}

const char *disposition_name (Attempt::Disposition disposition)
{
  switch (disposition) {
  case Attempt::Disabled: return "disabled";
  case Attempt::CertifiedEmpty: return "certified-empty";
  case Attempt::NotEmpty: return "not-empty";
  case Attempt::BackendFallback: return "backend-fallback";
  case Attempt::BackendError: return "backend-error";
  case Attempt::InvalidResult: return "invalid-result";
  }
  return "unknown";
}

void select_fake_mode (const std::string &mode)
{
  require (
    setenv (
      "KLAYOUT_CUDA_VIA1_STACK_FAKE_MODE", mode.c_str (), 1) == 0,
    "unable to select fake-backend mode " + mode);
}

void seal (Request &request)
{
  std::array<std::uint8_t, 32> digest;
  require (
    db::cuda_via1_stack_digest::request_digest (request, digest),
    "unable to digest test request");
  std::copy (digest.begin (), digest.end (), request.scene_digest);
}

struct QualifiedScene
{
  std::array<Context, 1> contexts;
  std::array<uint32_t, 1> layer_contexts;
  std::array<uint64_t, 1> layer_offsets;
  std::array<Cell, 1> cells;
  std::array<Box, 3> boxes;

  QualifiedScene ()
    : contexts {{ Context { 0, 0, 0, 0 } }},
      layer_contexts {{ 0 }},
      layer_offsets {{ 0 }},
      cells {{ Cell { 0, 1, 2, 1, 1, 1, 0 } }},
      boxes {{
        Box { 0, 0, 270, 130 },
        Box { 70, 0, 200, 130 },
        Box { 0, 0, 270, 130 }
      }}
  {
    //  nothing yet
  }

  Request request () const
  {
    Request result {};
    result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    result.struct_size = sizeof (result);
    result.opcode = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_EMPTY;
    result.option_flags =
      KLAYOUT_CUDA_SPATIAL_VIA1_STACK_QUALIFIED_OPTIONS;
    result.requested_mask = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES;
    result.dbu_per_micron = 2000;
    result.device = 0;
    result.enclosure_distance = 70;
    result.cut_width = 130;
    result.cut_height = 130;
    result.spacing_distance = 150;
    result.grid_cell_size = 2000;
    result.contexts = contexts.data ();
    result.context_count = contexts.size ();
    result.metal1_contexts = layer_contexts.data ();
    result.metal1_context_count = layer_contexts.size ();
    result.metal1_offsets = layer_offsets.data ();
    result.metal1_offset_count = layer_offsets.size ();
    result.via1_contexts = layer_contexts.data ();
    result.via1_context_count = layer_contexts.size ();
    result.via1_offsets = layer_offsets.data ();
    result.via1_offset_count = layer_offsets.size ();
    result.metal2_contexts = layer_contexts.data ();
    result.metal2_context_count = layer_contexts.size ();
    result.metal2_offsets = layer_offsets.data ();
    result.metal2_offset_count = layer_offsets.size ();
    result.cells = cells.data ();
    result.cell_count = cells.size ();
    result.boxes = boxes.data ();
    result.box_count = boxes.size ();
    result.flat_metal1_box_count = 1;
    result.flat_via1_box_count = 1;
    result.flat_metal2_box_count = 1;
    result.scene_left = 0;
    result.scene_bottom = 0;
    result.scene_right = 270;
    result.scene_top = 130;
    result.max_contexts = 16;
    result.max_grid_cells = 16;
    result.max_metal_memberships = 16;
    result.max_via_memberships = 16;
    result.max_pair_work = 16;
    seal (result);
    return result;
  }
};

void expect_result (
  const QualifiedScene &scene, const std::string &mode,
  Attempt::Disposition expected)
{
  select_fake_mode (mode);
  const Attempt attempt =
    db::cuda_spatial_try_via1_stack_empty (scene.request ());
  require (
    attempt.disposition == expected,
    "result mode " + mode + ": expected " + disposition_name (expected) +
      ", got " + disposition_name (attempt.disposition) +
      (attempt.message.empty () ? "" : " message=" + attempt.message));
  std::cout << "VIA1_STACK_HOST_GUARD ok kind=result mode=" << mode
            << " disposition=" << disposition_name (attempt.disposition)
            << "\n";
}

void expect_invalid_request (
  const QualifiedScene &scene, const std::string &name,
  const std::function<void (Request &)> &mutate, bool reseal = true)
{
  Request request = scene.request ();
  mutate (request);
  if (reseal) {
    seal (request);
  }
  select_fake_mode ("well-formed");
  const Attempt attempt = db::cuda_spatial_try_via1_stack_empty (request);
  require (
    attempt.disposition == Attempt::InvalidResult,
    "request case " + name + ": expected invalid-result, got " +
      disposition_name (attempt.disposition) +
      (attempt.message.empty () ? "" : " message=" + attempt.message));
  std::cout << "VIA1_STACK_HOST_GUARD ok kind=request case=" << name
            << " disposition=" << disposition_name (attempt.disposition)
            << "\n";
}

} // anonymous namespace

int main (int argc, char **argv)
{
  try {
    require (
      argc == 2,
      "usage: via1_stack_host_guard_smoke /absolute/path/to/fake-backend.so");
    require (
      setenv ("KLAYOUT_CUDA_SPATIAL_BACKEND", argv [1], 1) == 0,
      "unable to select fake backend");
    require (
      setenv ("KLAYOUT_CUDA_VIA1_STACK", "1", 1) == 0,
      "unable to enable VIA1-stack backend");
    unsetenv ("KLAYOUT_CUDA_SPATIAL_TELEMETRY");
    unsetenv ("KLAYOUT_CUDA_VIA1_STACK_TELEMETRY");

    const QualifiedScene scene;
    require (
      db::cuda_spatial_via1_stack_requested (),
      "host did not load the fake VIA1-stack backend");

    expect_result (scene, "well-formed", Attempt::CertifiedEmpty);

    const std::vector<std::string> invalid_results = {
      "zero-work",
      "impossible-via-candidate",
      "impossible-metal1-candidate",
      "impossible-metal2-candidate",
      "mutated-echo",
      "mutated-digest",
      "bad-result-abi",
      "short-result",
      "reserved-result",
      "fallback-flag",
      "device-flag",
      "partial-mask",
      "bad-conservation",
      "zero-grid"
    };
    for (std::vector<std::string>::const_iterator mode =
           invalid_results.begin (); mode != invalid_results.end (); ++mode) {
      expect_result (scene, *mode, Attempt::InvalidResult);
    }
    expect_result (scene, "result-error", Attempt::BackendError);
    expect_result (scene, "return-error", Attempt::BackendError);

    expect_invalid_request (
      scene, "bad-abi", [] (Request &r) { ++r.abi_version; });
    expect_invalid_request (
      scene, "short-struct",
      [] (Request &r) { r.struct_size = sizeof (r) - 1; });
    expect_invalid_request (
      scene, "bad-opcode", [] (Request &r) { ++r.opcode; });
    expect_invalid_request (
      scene, "missing-options", [] (Request &r) { r.option_flags = 0; });
    expect_invalid_request (
      scene, "unknown-option",
      [] (Request &r) { r.option_flags |= uint32_t (1u << 31); });
    expect_invalid_request (
      scene, "zero-mask", [] (Request &r) { r.requested_mask = 0; });
    expect_invalid_request (
      scene, "partial-mask",
      [] (Request &r) {
        r.requested_mask &= ~uint32_t (KLAYOUT_CUDA_SPATIAL_VIA1_2);
      });
    expect_invalid_request (
      scene, "unknown-mask",
      [] (Request &r) { r.requested_mask |= uint32_t (1u << 31); });
    expect_invalid_request (
      scene, "wrong-dbu", [] (Request &r) { r.dbu_per_micron = 1000; });
    expect_invalid_request (
      scene, "negative-device", [] (Request &r) { r.device = -1; });
    expect_invalid_request (
      scene, "reserved0", [] (Request &r) { r.reserved0 = 1; });
    expect_invalid_request (
      scene, "reserved1", [] (Request &r) { r.reserved1 [1] = 1; });
    expect_invalid_request (
      scene, "wrong-enclosure",
      [] (Request &r) { --r.enclosure_distance; });
    expect_invalid_request (
      scene, "wrong-cut-width", [] (Request &r) { --r.cut_width; });
    expect_invalid_request (
      scene, "wrong-cut-height", [] (Request &r) { --r.cut_height; });
    expect_invalid_request (
      scene, "wrong-spacing", [] (Request &r) { --r.spacing_distance; });
    expect_invalid_request (
      scene, "wrong-grid-size", [] (Request &r) { --r.grid_cell_size; });
    expect_invalid_request (
      scene, "empty-contexts", [] (Request &r) { r.context_count = 0; });
    expect_invalid_request (
      scene, "offset-count-mismatch",
      [] (Request &r) { r.via1_offset_count = 0; });
    expect_invalid_request (
      scene, "empty-metal1",
      [] (Request &r) { r.flat_metal1_box_count = 0; });
    expect_invalid_request (
      scene, "empty-via1",
      [] (Request &r) { r.flat_via1_box_count = 0; });
    expect_invalid_request (
      scene, "empty-metal2",
      [] (Request &r) { r.flat_metal2_box_count = 0; });
    expect_invalid_request (
      scene, "empty-scene-x",
      [] (Request &r) { r.scene_right = r.scene_left; });
    expect_invalid_request (
      scene, "empty-scene-y",
      [] (Request &r) { r.scene_top = r.scene_bottom; });
    expect_invalid_request (
      scene, "zero-context-cap",
      [] (Request &r) { r.max_contexts = 0; });
    expect_invalid_request (
      scene, "context-over-cap",
      [] (Request &r) { r.max_contexts = r.context_count - 1; });
    expect_invalid_request (
      scene, "zero-grid-cap",
      [] (Request &r) { r.max_grid_cells = 0; });
    expect_invalid_request (
      scene, "zero-metal-membership-cap",
      [] (Request &r) { r.max_metal_memberships = 0; });
    expect_invalid_request (
      scene, "zero-via-membership-cap",
      [] (Request &r) { r.max_via_memberships = 0; });
    expect_invalid_request (
      scene, "zero-pair-cap",
      [] (Request &r) { r.max_pair_work = 0; });
    expect_invalid_request (
      scene, "bad-digest",
      [] (Request &r) { r.scene_digest [0] ^= 0x80u; }, false);
    expect_invalid_request (
      scene, "null-boxes",
      [] (Request &r) { r.boxes = 0; }, false);

    std::cout
      << "VIA1_STACK_HOST_GUARD PASS valid_results=1"
      << " invalid_results=" << invalid_results.size ()
      << " backend_errors=2 invalid_requests=32\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "VIA1_STACK_HOST_GUARD FAIL " << ex.what () << "\n";
    return 1;
  }
}
