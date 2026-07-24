#include "dbCudaPoly34Digest.h"
#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Box = klayout_cuda_spatial_poly34_box_v1;
using Cell = klayout_cuda_spatial_poly34_cell_v1;
using Context = klayout_cuda_spatial_poly34_context_v1;
using Request = klayout_cuda_spatial_poly34_request_v1;
using Result = klayout_cuda_spatial_poly34_result_v1;
using Span = klayout_cuda_spatial_poly34_domain_span_v1;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

struct Scene
{
  std::vector<Context> contexts;
  std::vector<std::uint32_t> domain_contexts;
  std::vector<std::uint64_t> poly_offsets;
  std::vector<std::uint64_t> active_offsets;
  std::vector<std::uint64_t> gate_offsets;
  std::vector<Cell> cells;
  std::vector<Box> boxes;
  Request request{};

  Scene(
      const std::vector<Box> &poly, const std::vector<Box> &active,
      const std::vector<Box> &gate,
      std::uint32_t context_count = 1, std::int64_t context_pitch = 0)
  {
    require(
        !poly.empty() && !active.empty() && !gate.empty() &&
            context_count != 0,
        "scene inputs must be nonempty");

    contexts.reserve(context_count);
    domain_contexts.reserve(context_count);
    poly_offsets.reserve(context_count);
    active_offsets.reserve(context_count);
    gate_offsets.reserve(context_count);
    for (std::uint32_t id = 0; id < context_count; ++id) {
      contexts.push_back(
          {static_cast<std::int64_t>(id) * context_pitch, 0, 0, 0});
      domain_contexts.push_back(id);
      poly_offsets.push_back(
          static_cast<std::uint64_t>(id) * poly.size());
      active_offsets.push_back(
          static_cast<std::uint64_t>(id) * active.size());
      gate_offsets.push_back(
          static_cast<std::uint64_t>(id) * gate.size());
    }

    boxes.reserve(poly.size() + active.size() + gate.size());
    boxes.insert(boxes.end(), poly.begin(), poly.end());
    boxes.insert(boxes.end(), active.begin(), active.end());
    boxes.insert(boxes.end(), gate.begin(), gate.end());
    const Span poly_span = {
        0, static_cast<std::uint32_t>(poly.size()), 0};
    const Span active_span = {
        poly.size(), static_cast<std::uint32_t>(active.size()), 0};
    const Span gate_span = {
        poly.size() + active.size(),
        static_cast<std::uint32_t>(gate.size()), 0};
    Cell cell{};
    cell.source_cell_index = 0x50335034u;
    cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN] = poly_span;
    cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN] = active_span;
    cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN] = gate_span;
    cells.push_back(cell);

    std::int64_t local_left = boxes.front().left;
    std::int64_t local_bottom = boxes.front().bottom;
    std::int64_t local_right = boxes.front().right;
    std::int64_t local_top = boxes.front().top;
    for (const Box &box : boxes) {
      local_left = std::min(local_left, box.left);
      local_bottom = std::min(local_bottom, box.bottom);
      local_right = std::max(local_right, box.right);
      local_top = std::max(local_top, box.top);
    }

    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.opcode = KLAYOUT_CUDA_SPATIAL_POLY34_TERMINAL_EMPTY;
    request.option_flags =
        KLAYOUT_CUDA_SPATIAL_POLY34_QUALIFIED_OPTIONS;
    request.format_version = 1;
    request.dbu_per_micron = 2000;
    request.root_cell = 0;
    request.requested_mask = KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES;
    request.device = 0;
    request.poly3_distance = 110;
    request.poly4_distance = 140;
    request.grid_cell_size = 2000;
    request.store_identity = 0x53544f5245ull;
    request.layout_identity = 0x4c41594f5554ull;
    request.top_cell_identity = 0x544f50ull;
    request.poly_layer_id = 9;
    request.active_layer_id = 1;
    request.gate_layer_id = 0x5034u;
    request.contexts = contexts.data();
    request.context_count = contexts.size();
    request.context_record_bytes = sizeof(Context);
    request.poly_contexts = domain_contexts.data();
    request.poly_context_count = domain_contexts.size();
    request.poly_offsets = poly_offsets.data();
    request.poly_offset_count = poly_offsets.size();
    request.active_contexts = domain_contexts.data();
    request.active_context_count = domain_contexts.size();
    request.active_offsets = active_offsets.data();
    request.active_offset_count = active_offsets.size();
    request.gate_contexts = domain_contexts.data();
    request.gate_context_count = domain_contexts.size();
    request.gate_offsets = gate_offsets.data();
    request.gate_offset_count = gate_offsets.size();
    request.cells = cells.data();
    request.cell_count = cells.size();
    request.cell_record_bytes = sizeof(Cell);
    request.boxes = boxes.data();
    request.box_count = boxes.size();
    request.box_record_bytes = sizeof(Box);
    request.flat_poly_box_count =
        static_cast<std::uint64_t>(context_count) * poly.size();
    request.flat_active_box_count =
        static_cast<std::uint64_t>(context_count) * active.size();
    request.flat_gate_box_count =
        static_cast<std::uint64_t>(context_count) * gate.size();
    request.scene_left = local_left;
    request.scene_bottom = local_bottom;
    request.scene_right =
        local_right +
        static_cast<std::int64_t>(context_count - 1) * context_pitch;
    request.scene_top = local_top;
    request.max_contexts = 100000;
    request.max_flat_boxes = 1000000;
    request.max_grid_cells = 1000000;
    request.max_poly_memberships = 1000000;
    request.max_active_memberships = 1000000;
    request.max_query_visits = 10000000;
    request.max_candidate_work = 1000000;
    request.max_candidates_per_gate = 64;
    seal();
  }

  void seal()
  {
    std::array<std::uint8_t, 32> digest{};
    require(
        db::cuda_poly34_digest::request_digest(request, digest),
        "request digest failed");
    std::copy(digest.begin(), digest.end(), request.scene_digest);
  }
};

Result run_ok(const std::string &name, Scene &scene)
{
  Result result{};
  const int status =
      klayout_cuda_spatial_run_poly34_empty_v1(
          &scene.request, &result);
  require(
      status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.status == KLAYOUT_CUDA_SPATIAL_OK,
      name + ": backend status=" + std::to_string(status) +
          " result=" + std::to_string(result.status) +
          " message=" + result.message);
  require(
      result.fallback_flags == 0 && result.device_flags == 0,
      name + ": unexpected fallback/device flags");
  require(
      result.opcode == scene.request.opcode &&
          result.option_flags == scene.request.option_flags &&
          result.format_version == scene.request.format_version &&
          result.requested_mask == scene.request.requested_mask &&
          result.dbu_per_micron == scene.request.dbu_per_micron &&
          result.root_cell == scene.request.root_cell &&
          result.store_identity == scene.request.store_identity &&
          result.layout_identity == scene.request.layout_identity &&
          result.top_cell_identity == scene.request.top_cell_identity &&
          result.poly_layer_id == scene.request.poly_layer_id &&
          result.active_layer_id == scene.request.active_layer_id &&
          result.gate_layer_id == scene.request.gate_layer_id &&
          std::memcmp(
              result.scene_digest, scene.request.scene_digest,
              sizeof(scene.request.scene_digest)) == 0,
      name + ": proof identity echo mismatch");
  require(
      result.expanded_poly_box_count ==
              scene.request.flat_poly_box_count &&
          result.expanded_active_box_count ==
              scene.request.flat_active_box_count &&
          result.expanded_gate_box_count ==
              scene.request.flat_gate_box_count &&
          result.poly_terminal_empty_count <=
              scene.request.flat_gate_box_count &&
          result.active_terminal_empty_count <=
              scene.request.flat_gate_box_count &&
          result.atomic_terminal_empty_count +
                  result.fallback_gate_count ==
              scene.request.flat_gate_box_count &&
          result.maximum_poly_candidates <=
              scene.request.max_candidates_per_gate &&
          result.maximum_active_candidates <=
              scene.request.max_candidates_per_gate,
      name + ": result conservation mismatch");
  return result;
}

}  // namespace

int main()
{
  try {
    const Box gate = {0, 0, 100, 180};

    Scene clean({gate}, {gate}, {gate});
    const Result clean_result = run_ok("clean", clean);
    require(
        clean_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            clean_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES &&
            clean_result.poly_terminal_empty_count == 1 &&
            clean_result.active_terminal_empty_count == 1 &&
            clean_result.atomic_terminal_empty_count == 1 &&
            clean_result.fallback_gate_count == 0,
        "clean: atomic terminal-empty certificate was not complete");

    Scene high_multiplicity(
        {gate}, {gate}, {gate}, 4096, 1000);
    const Result high_result =
        run_ok("high-multiplicity-clean", high_multiplicity);
    require(
        high_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            high_result.atomic_terminal_empty_count == 4096 &&
            high_result.poly_terminal_empty_count == 4096 &&
            high_result.active_terminal_empty_count == 4096 &&
            high_result.maximum_poly_candidates == 1 &&
            high_result.maximum_active_candidates == 1,
        "high-multiplicity-clean: reused hierarchy was not certified");

    const Box poly3_hit = {-1, 0, 100, 180};
    Scene poly3({poly3_hit}, {gate}, {gate});
    const Result poly3_result = run_ok("poly3-hit", poly3);
    require(
        poly3_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY &&
            poly3_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY4_RULE &&
            poly3_result.poly_terminal_empty_count == 0 &&
            poly3_result.active_terminal_empty_count == 1 &&
            poly3_result.atomic_terminal_empty_count == 0 &&
            poly3_result.fallback_gate_count == 1,
        "poly3-hit: independent positive-area profile was not retained");

    const Box poly4_hit = {0, -1, 100, 180};
    Scene poly4({gate}, {poly4_hit}, {gate});
    const Result poly4_result = run_ok("poly4-hit", poly4);
    require(
        poly4_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY &&
            poly4_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY3_RULE &&
            poly4_result.poly_terminal_empty_count == 1 &&
            poly4_result.active_terminal_empty_count == 0 &&
            poly4_result.atomic_terminal_empty_count == 0 &&
            poly4_result.fallback_gate_count == 1,
        "poly4-hit: independent positive-area profile was not retained");

    const Box gate2 = {1000, 0, 1100, 180};
    const Box poly4_hit2 = {1000, -1, 1100, 180};
    Scene mixed(
        {poly3_hit, gate2}, {gate, poly4_hit2}, {gate, gate2});
    const Result mixed_result = run_ok("mixed-atomic-fallback", mixed);
    require(
        mixed_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY &&
            mixed_result.certified_empty_mask == 0 &&
            mixed_result.poly_terminal_empty_count == 1 &&
            mixed_result.active_terminal_empty_count == 1 &&
            mixed_result.atomic_terminal_empty_count == 0 &&
            mixed_result.fallback_gate_count == 2,
        "mixed-atomic-fallback: partial rule results escaped atomically");

    Scene malformed({gate}, {gate}, {gate});
    malformed.request.scene_digest[0] ^= 0x80u;
    Result malformed_result{};
    const int malformed_status =
        klayout_cuda_spatial_run_poly34_empty_v1(
            &malformed.request, &malformed_result);
    require(
        malformed_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            malformed_result.status ==
                KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            malformed_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN &&
            malformed_result.certified_empty_mask == 0,
        "malformed: digest mismatch did not fail closed");

    Scene capacity({gate}, {gate}, {gate}, 2, 4000);
    capacity.request.max_grid_cells = 1;
    capacity.seal();
    Result capacity_result{};
    const int capacity_status =
        klayout_cuda_spatial_run_poly34_empty_v1(
            &capacity.request, &capacity_result);
    require(
        capacity_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
            capacity_result.status ==
                KLAYOUT_CUDA_SPATIAL_FALLBACK &&
            capacity_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN &&
            capacity_result.fallback_flags ==
                KLAYOUT_CUDA_SPATIAL_FALLBACK_DENSE_CELL &&
            capacity_result.certified_empty_mask == 0,
        "capacity: grid exhaustion did not request atomic CPU fallback");

    std::cout
        << "POLY34_BACKEND_SMOKE ok clean=1 "
           "high_multiplicity=4096 poly3_hit=1 poly4_hit=1 "
           "mixed_atomic_fallback=1 malformed=1 capacity=1\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "POLY34_BACKEND_SMOKE failed: " << error.what() << "\n";
    return 1;
  }
}
