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

struct RawCellInput
{
  std::vector<Box> poly;
  std::vector<Box> active;
};

struct RawScene
{
  std::vector<Context> contexts;
  std::vector<std::uint32_t> poly_contexts;
  std::vector<std::uint32_t> active_contexts;
  std::vector<std::uint64_t> poly_offsets;
  std::vector<std::uint64_t> active_offsets;
  std::vector<Cell> cells;
  std::vector<Box> boxes;
  Request request{};

  RawScene(
      const std::vector<Box> &poly, const std::vector<Box> &active)
      : RawScene(
            std::vector<RawCellInput>{{poly, active}},
            std::vector<Context>{{0, 0, 0, 0}})
  {
  }

  RawScene(
      const std::vector<RawCellInput> &input_cells,
      const std::vector<Context> &input_contexts)
      : contexts(input_contexts)
  {
    require(
        !input_cells.empty() && !contexts.empty(),
        "raw scene must contain cells and contexts");
    require(
        contexts.front().tx == 0 && contexts.front().ty == 0 &&
            contexts.front().cell_id == 0 &&
            contexts.front().transform_code == 0,
        "raw scene root context must be identity cell zero");

    cells.reserve(input_cells.size());
    for (std::size_t cell_id = 0; cell_id < input_cells.size();
         ++cell_id) {
      const RawCellInput &input = input_cells[cell_id];
      Cell cell{};
      cell.source_cell_index = 0x52415700u + cell_id;

      Span &poly =
          cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN];
      poly.box_begin = boxes.size();
      poly.box_count = static_cast<std::uint32_t>(input.poly.size());
      boxes.insert(boxes.end(), input.poly.begin(), input.poly.end());

      Span &active =
          cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN];
      active.box_begin = boxes.size();
      active.box_count = static_cast<std::uint32_t>(input.active.size());
      boxes.insert(boxes.end(), input.active.begin(), input.active.end());

      Span &gate =
          cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN];
      gate.box_begin = boxes.size();
      gate.box_count = 0;
      gate.reserved0 = 0;
      cells.push_back(cell);
    }
    require(!boxes.empty(), "raw scene must contain source geometry");

    std::uint64_t flat_poly = 0;
    std::uint64_t flat_active = 0;
    bool have_scene = false;
    std::int64_t scene_left = 0;
    std::int64_t scene_bottom = 0;
    std::int64_t scene_right = 0;
    std::int64_t scene_top = 0;
    for (std::uint32_t context_id = 0;
         context_id < contexts.size(); ++context_id) {
      const Context &context = contexts[context_id];
      require(
          context.cell_id < cells.size() &&
              context.transform_code == 0,
          "raw smoke supports only translated valid contexts");
      const Cell &cell = cells[context.cell_id];
      const Span &poly =
          cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN];
      const Span &active =
          cell.domains[KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN];
      if (poly.box_count) {
        poly_contexts.push_back(context_id);
        poly_offsets.push_back(flat_poly);
        flat_poly += poly.box_count;
      }
      if (active.box_count) {
        active_contexts.push_back(context_id);
        active_offsets.push_back(flat_active);
        flat_active += active.box_count;
      }
      for (std::uint32_t domain = 0;
           domain < KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN;
           ++domain) {
        const Span &span = cell.domains[domain];
        for (std::uint32_t local = 0; local < span.box_count; ++local) {
          const Box &box = boxes[span.box_begin + local];
          const std::int64_t left = box.left + context.tx;
          const std::int64_t bottom = box.bottom + context.ty;
          const std::int64_t right = box.right + context.tx;
          const std::int64_t top = box.top + context.ty;
          if (!have_scene) {
            scene_left = left;
            scene_bottom = bottom;
            scene_right = right;
            scene_top = top;
            have_scene = true;
          } else {
            scene_left = std::min(scene_left, left);
            scene_bottom = std::min(scene_bottom, bottom);
            scene_right = std::max(scene_right, right);
            scene_top = std::max(scene_top, top);
          }
        }
      }
    }
    require(
        have_scene && flat_poly && flat_active,
        "raw scene must expand both physical domains");

    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.opcode =
        KLAYOUT_CUDA_SPATIAL_POLY34_RAW_TERMINAL_EMPTY;
    request.option_flags =
        KLAYOUT_CUDA_SPATIAL_POLY34_RAW_QUALIFIED_OPTIONS;
    request.format_version = 2;
    request.dbu_per_micron = 2000;
    request.root_cell = 0;
    request.requested_mask = KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES;
    request.device = 0;
    request.poly3_distance = 110;
    request.poly4_distance = 140;
    request.grid_cell_size = 2000;
    request.store_identity = 0x52415753544f5245ull;
    request.layout_identity = 0x5241574c41594f55ull;
    request.top_cell_identity = 0x524157544f50ull;
    request.poly_layer_id = 9;
    request.active_layer_id = 1;
    request.gate_layer_id = KLAYOUT_CUDA_SPATIAL_POLY34_NO_GATE_LAYER;
    request.contexts = contexts.data();
    request.context_count = contexts.size();
    request.context_record_bytes = sizeof(Context);
    request.poly_contexts = poly_contexts.data();
    request.poly_context_count = poly_contexts.size();
    request.poly_offsets = poly_offsets.data();
    request.poly_offset_count = poly_offsets.size();
    request.active_contexts = active_contexts.data();
    request.active_context_count = active_contexts.size();
    request.active_offsets = active_offsets.data();
    request.active_offset_count = active_offsets.size();
    request.gate_contexts = nullptr;
    request.gate_context_count = 0;
    request.gate_offsets = nullptr;
    request.gate_offset_count = 0;
    request.cells = cells.data();
    request.cell_count = cells.size();
    request.cell_record_bytes = sizeof(Cell);
    request.boxes = boxes.data();
    request.box_count = boxes.size();
    request.box_record_bytes = sizeof(Box);
    request.flat_poly_box_count = flat_poly;
    request.flat_active_box_count = flat_active;
    request.flat_gate_box_count = 0;
    request.scene_left = scene_left;
    request.scene_bottom = scene_bottom;
    request.scene_right = scene_right;
    request.scene_top = scene_top;
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
        "raw request digest failed");
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

Result run_raw_ok(
    const std::string &name, RawScene &scene,
    std::uint64_t expected_gate_count)
{
  require(
      scene.request.flat_gate_box_count == 0 &&
          scene.request.gate_context_count == 0 &&
          scene.request.gate_offset_count == 0 &&
          scene.request.gate_contexts == nullptr &&
          scene.request.gate_offsets == nullptr,
      name + ": raw caller materialized a GATE census");
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
          result.context_count == scene.request.context_count &&
          result.poly_context_count ==
              scene.request.poly_context_count &&
          result.active_context_count ==
              scene.request.active_context_count &&
          result.gate_context_count == 0 &&
          result.cell_count == scene.request.cell_count &&
          result.box_count == scene.request.box_count &&
          result.flat_poly_box_count ==
              scene.request.flat_poly_box_count &&
          result.flat_active_box_count ==
              scene.request.flat_active_box_count &&
          std::memcmp(
              result.scene_digest, scene.request.scene_digest,
              sizeof(scene.request.scene_digest)) == 0,
      name + ": raw proof identity echo mismatch");
  require(
      result.expanded_poly_box_count ==
              scene.request.flat_poly_box_count &&
          result.expanded_active_box_count ==
              scene.request.flat_active_box_count &&
          result.flat_gate_box_count == expected_gate_count &&
          result.expanded_gate_box_count == expected_gate_count &&
          result.poly_terminal_empty_count <= expected_gate_count &&
          result.active_terminal_empty_count <= expected_gate_count &&
          result.atomic_terminal_empty_count +
                  result.fallback_gate_count ==
              expected_gate_count &&
          result.maximum_poly_candidates <=
              scene.request.max_candidates_per_gate &&
          result.maximum_active_candidates <=
              scene.request.max_candidates_per_gate,
      name + ": raw result conservation mismatch");
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

    RawScene raw_clean({gate}, {gate});
    const Result raw_clean_result =
        run_raw_ok("raw-clean", raw_clean, 1);
    require(
        raw_clean_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            raw_clean_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES &&
            raw_clean_result.poly_terminal_empty_count == 1 &&
            raw_clean_result.active_terminal_empty_count == 1 &&
            raw_clean_result.atomic_terminal_empty_count == 1 &&
            raw_clean_result.fallback_gate_count == 0,
        "raw-clean: device-derived GATE was not certified");

    RawScene raw_poly3({poly3_hit}, {gate});
    const Result raw_poly3_result =
        run_raw_ok("raw-poly3-hit", raw_poly3, 1);
    require(
        raw_poly3_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY &&
            raw_poly3_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY4_RULE &&
            raw_poly3_result.poly_terminal_empty_count == 0 &&
            raw_poly3_result.active_terminal_empty_count == 1 &&
            raw_poly3_result.atomic_terminal_empty_count == 0 &&
            raw_poly3_result.fallback_gate_count == 1,
        "raw-poly3-hit: independent POLY.3 fallback was not retained");

    RawScene raw_poly4({gate}, {poly4_hit});
    const Result raw_poly4_result =
        run_raw_ok("raw-poly4-hit", raw_poly4, 1);
    require(
        raw_poly4_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY &&
            raw_poly4_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY3_RULE &&
            raw_poly4_result.poly_terminal_empty_count == 1 &&
            raw_poly4_result.active_terminal_empty_count == 0 &&
            raw_poly4_result.atomic_terminal_empty_count == 0 &&
            raw_poly4_result.fallback_gate_count == 1,
        "raw-poly4-hit: independent POLY.4 fallback was not retained");

    const Box wide_gate = {0, 0, 400, 400};
    const Box overlapping_left = {0, 0, 250, 400};
    const Box overlapping_right = {150, 0, 400, 400};
    RawScene raw_fragmented(
        {overlapping_left, overlapping_right}, {wide_gate});
    const Result raw_fragmented_result =
        run_raw_ok("raw-overlapping-fragmented", raw_fragmented, 2);
    require(
        raw_fragmented_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            raw_fragmented_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES &&
            raw_fragmented_result.poly_terminal_empty_count == 2 &&
            raw_fragmented_result.active_terminal_empty_count == 2 &&
            raw_fragmented_result.atomic_terminal_empty_count == 2 &&
            raw_fragmented_result.fallback_gate_count == 0,
        "raw-overlapping-fragmented: exact cover was not certified");

    // The 1-DBU right fragment makes both enclosing bands at x=100 only
    // partially covered.  Per-tile evaluation therefore has to prove that
    // the shared side is internal to the exact GATE union; treating it as a
    // physical boundary would conservatively (but incorrectly) decline.
    const Box seam_left = {0, 0, 100, 400};
    const Box seam_right = {100, 0, 101, 400};
    const Box seam_active = {0, 0, 101, 400};
    RawScene raw_internal_seam(
        {seam_left, seam_right}, {seam_active});
    const Result raw_internal_seam_result =
        run_raw_ok("raw-internal-seam", raw_internal_seam, 2);
    require(
        raw_internal_seam_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            raw_internal_seam_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES &&
            raw_internal_seam_result.atomic_terminal_empty_count == 2 &&
            raw_internal_seam_result.fallback_gate_count == 0,
        "raw-internal-seam: proven non-boundary was not suppressed");

    const Box grid_boundary = {2000, 2000, 6100, 6100};
    RawScene raw_grid_boundary({grid_boundary}, {grid_boundary});
    const Result raw_grid_result =
        run_raw_ok("raw-grid-boundary-owner", raw_grid_boundary, 1);
    require(
        raw_grid_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            raw_grid_result.atomic_terminal_empty_count == 1,
        "raw-grid-boundary-owner: one pair was emitted more than once");

    const RawCellInput root_cell{};
    const RawCellInput poly_cell{{gate}, {}};
    const RawCellInput active_cell{{}, {gate}};
    RawScene raw_cross_context(
        {root_cell, poly_cell, active_cell},
        {{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 2, 0}});
    const Result raw_cross_result =
        run_raw_ok("raw-cross-context", raw_cross_context, 1);
    require(
        raw_cross_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            raw_cross_result.atomic_terminal_empty_count == 1,
        "raw-cross-context: global sibling intersection was missed");

    const Box disjoint = {500, 500, 600, 600};
    RawScene raw_empty_intersection({gate}, {disjoint});
    const Result raw_empty_result =
        run_raw_ok("raw-empty-intersection", raw_empty_intersection, 0);
    require(
        raw_empty_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE &&
            raw_empty_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES &&
            raw_empty_result.poly_terminal_empty_count == 0 &&
            raw_empty_result.active_terminal_empty_count == 0 &&
            raw_empty_result.atomic_terminal_empty_count == 0 &&
            raw_empty_result.fallback_gate_count == 0,
        "raw-empty-intersection: exact empty join did not certify");

    RawScene raw_malformed({gate}, {gate});
    raw_malformed.request.scene_digest[0] ^= 0x40u;
    Result raw_malformed_result{};
    const int raw_malformed_status =
        klayout_cuda_spatial_run_poly34_empty_v1(
            &raw_malformed.request, &raw_malformed_result);
    require(
        raw_malformed_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            raw_malformed_result.status ==
                KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            raw_malformed_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN &&
            raw_malformed_result.certified_empty_mask == 0,
        "raw-malformed: digest mismatch did not fail closed");

    RawScene raw_gate_smuggling({gate}, {gate});
    raw_gate_smuggling
        .cells[0]
        .domains[KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN]
        .box_count = 1;
    raw_gate_smuggling.seal();
    Result raw_smuggling_result{};
    const int raw_smuggling_status =
        klayout_cuda_spatial_run_poly34_empty_v1(
            &raw_gate_smuggling.request, &raw_smuggling_result);
    require(
        raw_smuggling_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            raw_smuggling_result.status ==
                KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            raw_smuggling_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN &&
            raw_smuggling_result.certified_empty_mask == 0,
        "raw-gate-smuggling: host GATE span did not fail closed");

    RawScene raw_capacity(
        {gate, gate, gate}, {gate, gate, gate});
    raw_capacity.request.max_flat_boxes = 6;
    raw_capacity.seal();
    Result raw_capacity_result{};
    const int raw_capacity_status =
        klayout_cuda_spatial_run_poly34_empty_v1(
            &raw_capacity.request, &raw_capacity_result);
    require(
        raw_capacity_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
            raw_capacity_result.status ==
                KLAYOUT_CUDA_SPATIAL_FALLBACK &&
            raw_capacity_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN &&
            raw_capacity_result.fallback_flags ==
                KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_CAPACITY &&
            raw_capacity_result.certified_empty_mask == 0 &&
            raw_capacity_result.flat_gate_box_count == 0,
        "raw-capacity: derived intersection overflow did not fall back");

    std::cout
        << "POLY34_BACKEND_SMOKE ok clean=1 "
           "high_multiplicity=4096 poly3_hit=1 poly4_hit=1 "
           "mixed_atomic_fallback=1 malformed=1 capacity=1 "
           "raw_clean=1 raw_poly3_hit=1 raw_poly4_hit=1 "
           "raw_overlapping_fragmented=2 raw_internal_seam=2 "
           "raw_grid_owner=1 "
           "raw_cross_context=1 raw_empty_intersection=1 "
           "raw_malformed=1 raw_gate_smuggling=1 raw_capacity=1\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "POLY34_BACKEND_SMOKE failed: " << error.what() << "\n";
    return 1;
  }
}
