#include "dbCudaSpatialApi.h"
#include "dbCudaVia1StackDigest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Box = klayout_cuda_spatial_via1_stack_box_v1;
using Cell = klayout_cuda_spatial_via1_stack_cell_v1;
using Context = klayout_cuda_spatial_via1_stack_context_v1;
using Request = klayout_cuda_spatial_via1_stack_request_v1;
using Result = klayout_cuda_spatial_via1_stack_result_v1;

void
require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

Result
run_case(
    const std::string &name, const std::vector<Box> &metal1,
    const std::vector<Box> &vias, const std::vector<Box> &metal2,
    bool corrupt_digest = false)
{
  const Context contexts[] = {{0, 0, 0, 0}};
  const std::uint32_t layer_contexts[] = {0};
  const std::uint64_t layer_offsets[] = {0};
  std::vector<Box> boxes;
  boxes.reserve(metal1.size() + vias.size() + metal2.size());
  boxes.insert(boxes.end(), metal1.begin(), metal1.end());
  boxes.insert(boxes.end(), vias.begin(), vias.end());
  boxes.insert(boxes.end(), metal2.begin(), metal2.end());

  const Cell cells[] = {{
      0,
      static_cast<std::uint64_t>(metal1.size()),
      static_cast<std::uint64_t>(metal1.size() + vias.size()),
      static_cast<std::uint32_t>(metal1.size()),
      static_cast<std::uint32_t>(vias.size()),
      static_cast<std::uint32_t>(metal2.size()),
      0}};

  std::int64_t scene_left = boxes.front().left;
  std::int64_t scene_bottom = boxes.front().bottom;
  std::int64_t scene_right = boxes.front().right;
  std::int64_t scene_top = boxes.front().top;
  for (const Box &box : boxes) {
    scene_left = std::min(scene_left, box.left);
    scene_bottom = std::min(scene_bottom, box.bottom);
    scene_right = std::max(scene_right, box.right);
    scene_top = std::max(scene_top, box.top);
  }

  Request request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_VIA1_STACK_QUALIFIED_OPTIONS;
  request.requested_mask = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES;
  request.dbu_per_micron = 2000;
  request.device = 0;
  request.enclosure_distance = 70;
  request.cut_width = 130;
  request.cut_height = 130;
  request.spacing_distance = 150;
  request.grid_cell_size = 2000;
  request.contexts = contexts;
  request.context_count = 1;
  request.metal1_contexts = layer_contexts;
  request.metal1_context_count = 1;
  request.metal1_offsets = layer_offsets;
  request.metal1_offset_count = 1;
  request.via1_contexts = layer_contexts;
  request.via1_context_count = 1;
  request.via1_offsets = layer_offsets;
  request.via1_offset_count = 1;
  request.metal2_contexts = layer_contexts;
  request.metal2_context_count = 1;
  request.metal2_offsets = layer_offsets;
  request.metal2_offset_count = 1;
  request.cells = cells;
  request.cell_count = 1;
  request.boxes = boxes.data();
  request.box_count = boxes.size();
  request.flat_metal1_box_count = metal1.size();
  request.flat_via1_box_count = vias.size();
  request.flat_metal2_box_count = metal2.size();
  request.scene_left = scene_left;
  request.scene_bottom = scene_bottom;
  request.scene_right = scene_right;
  request.scene_top = scene_top;
  request.max_contexts = 16;
  request.max_grid_cells = 1024;
  request.max_metal_memberships = 1024;
  request.max_via_memberships = 1024;
  request.max_pair_work = 1024;
  std::array<std::uint8_t, 32> digest;
  require(
      db::cuda_via1_stack_digest::request_digest(request, digest),
      name + ": request digest failed");
  std::copy(digest.begin(), digest.end(), request.scene_digest);
  if (corrupt_digest) request.scene_digest[0] ^= 0x80u;

  Result result{};
  const int status =
      klayout_cuda_spatial_run_via1_stack_empty_v1(&request, &result);
  if (corrupt_digest) {
    require(
        status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            result.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
            result.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_UNCERTAIN &&
            result.certified_empty_mask == 0,
        name + ": digest mismatch was not rejected fail-closed");
    return result;
  }
  require(
      status == KLAYOUT_CUDA_SPATIAL_OK,
      name + ": backend returned " + std::to_string(status) +
          " message=" + result.message);
  require(
      result.status == KLAYOUT_CUDA_SPATIAL_OK,
      name + ": result status was not OK");
  require(
      result.fallback_flags == 0 && result.device_flags == 0,
      name + ": backend set a fallback/device flag");
  require(
      result.requested_mask == request.requested_mask &&
          result.option_flags == request.option_flags &&
          result.context_count == request.context_count &&
          result.flat_via1_box_count == request.flat_via1_box_count &&
          std::memcmp(
              result.scene_digest, request.scene_digest,
              sizeof(request.scene_digest)) == 0,
      name + ": proof echo mismatch");
  require(
      result.via_expanded_count == request.flat_via1_box_count &&
          result.via_size_checked_count == request.flat_via1_box_count &&
          result.via_pair_queried_count == request.flat_via1_box_count &&
          result.metal1_expanded_count ==
              request.flat_metal1_box_count &&
          result.metal2_expanded_count ==
              request.flat_metal2_box_count &&
          result.metal1_queried_count ==
              request.flat_via1_box_count &&
          result.metal1_certified_count + result.metal1_miss_count ==
              result.metal1_queried_count &&
          result.metal2_queried_count ==
              request.flat_via1_box_count &&
          result.metal2_certified_count + result.metal2_miss_count ==
              result.metal2_queried_count &&
          result.duplicate_via_pair_count +
                  result.unsafe_via_pair_count +
                  result.spacing_violation_count +
                  result.clean_via_pair_count ==
              result.via_candidate_pair_count,
      name + ": result conservation proof mismatch");
  return result;
}

Result
run_case(
    const std::string &name, const Box &metal1,
    const std::vector<Box> &vias, const Box &metal2,
    bool corrupt_digest = false)
{
  return run_case(
      name, std::vector<Box>{metal1}, vias, std::vector<Box>{metal2},
      corrupt_digest);
}

}  // namespace

int
main()
{
  try {
    const Box clean_metal = {0, 0, 270, 130};
    const Box clean_via = {70, 0, 200, 130};
    const Result clean =
        run_case("clean", clean_metal, {clean_via}, clean_metal);
    require(
        clean.disposition == KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE &&
            clean.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES &&
            clean.metal1_miss_count == 0 &&
            clean.metal2_miss_count == 0,
        "clean: complete certificate was not returned");

    const Box union_via = {70, 70, 200, 200};
    const std::vector<Box> horizontal_union = {
        {0, 70, 135, 200}, {135, 70, 270, 200}};
    const std::vector<Box> vertical_union = {
        {70, 0, 200, 135}, {70, 135, 200, 270}};
    const Result union_clean =
        run_case(
            "union-clean", horizontal_union, {union_via},
            vertical_union);
    require(
        union_clean.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE &&
            union_clean.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES &&
            union_clean.metal1_miss_count == 0 &&
            union_clean.metal2_miss_count == 0,
        "union-clean: exact cross-strip unions were not certified");

    const std::vector<Box> one_dbu_gap = {
        {0, 70, 134, 200}, {135, 70, 270, 200}};
    const Result union_gap =
        run_case(
            "union-gap", one_dbu_gap, {union_via},
            vertical_union);
    require(
        union_gap.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY &&
            union_gap.metal1_miss_count == 1 &&
            union_gap.metal2_miss_count == 0 &&
            union_gap.certified_empty_mask == 0x36u,
        "union-gap: one-DBU cross-strip gap was not retained");

    const std::vector<Box> vertical_one_dbu_gap = {
        {70, 0, 200, 134}, {70, 135, 200, 270}};
    const Result vertical_union_gap =
        run_case(
            "vertical-union-gap", horizontal_union, {union_via},
            vertical_one_dbu_gap);
    require(
        vertical_union_gap.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY &&
            vertical_union_gap.metal1_miss_count == 0 &&
            vertical_union_gap.metal2_miss_count == 1 &&
            vertical_union_gap.certified_empty_mask == 0x0fu,
        "vertical-union-gap: one-DBU cross-strip gap was not retained");

    const Box grid_boundary_via = {1935, -130, 2065, 0};
    const std::vector<Box> boundary_union = {
        {1865, -130, 2000, 0}, {2000, -130, 2135, 0}};
    const Box boundary_clean_metal = {1800, -300, 2200, 200};
    const Result boundary_clean =
        run_case(
            "negative-grid-boundary", boundary_union, {grid_boundary_via},
            {boundary_clean_metal});
    require(
        boundary_clean.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE &&
            boundary_clean.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES &&
            boundary_clean.metal1_miss_count == 0 &&
            boundary_clean.metal2_miss_count == 0,
        "negative-grid-boundary: exact X=2000 union seam was not certified");

    const Box miss_metal2 = {0, 0, 260, 130};
    const Result miss =
        run_case("miss", clean_metal, {clean_via}, miss_metal2);
    require(
        miss.disposition == KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY &&
            miss.metal1_miss_count == 0 &&
            miss.metal2_miss_count == 1 &&
            miss.certified_empty_mask == 0x0fu,
        "miss: M2 projection miss was not isolated");

    const Box wide_metal = {0, 0, 549, 130};
    const Box spaced_via = {349, 0, 479, 130};
    const Result spacing =
        run_case(
            "spacing", wide_metal, {clean_via, spaced_via}, wide_metal);
    require(
        spacing.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY &&
            spacing.spacing_violation_count == 1 &&
            spacing.unsafe_via_pair_count == 0 &&
            spacing.via_candidate_pair_count == 1 &&
            spacing.certified_empty_mask ==
                (KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES &
                 ~KLAYOUT_CUDA_SPATIAL_VIA1_2),
        "spacing: strict 149-DBU violation was not classified");

    const Box equality_metal = {0, 0, 550, 130};
    const Box equality_via = {350, 0, 480, 130};
    const Result equality =
        run_case(
            "spacing-equality", equality_metal,
            {clean_via, equality_via}, equality_metal);
    require(
        equality.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE &&
            equality.spacing_violation_count == 0 &&
            equality.clean_via_pair_count == 1 &&
            equality.via_candidate_pair_count == 1,
        "spacing-equality: exact 150-DBU separation was not clean");

    const Box diagonal_metal = {0, 0, 490, 520};
    const Box diagonal_first = {70, 70, 200, 200};
    const Box diagonal_second = {290, 320, 420, 450};
    const Result diagonal_equality =
        run_case(
            "diagonal-equality", diagonal_metal,
            {diagonal_first, diagonal_second}, diagonal_metal);
    require(
        diagonal_equality.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE &&
            diagonal_equality.spacing_violation_count == 0 &&
            diagonal_equality.clean_via_pair_count == 1,
        "diagonal-equality: exact 90/120/150 separation was not clean");

    const Box short_via = {70, 0, 199, 130};
    const Box short_via_metal = {0, 0, 269, 130};
    const Result size =
        run_case(
            "size", short_via_metal, {short_via}, short_via_metal);
    require(
        size.disposition == KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY &&
            size.via_size_checked_count == 1 &&
            size.via_size_violation_count == 1 &&
            size.certified_empty_mask ==
                (KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES &
                 ~KLAYOUT_CUDA_SPATIAL_VIA1_1),
        "size: non-130x130 VIA was not isolated");

    const Box touch_metal = {0, 0, 400, 130};
    const Box touching_via = {200, 0, 330, 130};
    const Result touch =
        run_case(
            "touch", touch_metal, {clean_via, touching_via}, touch_metal);
    require(
        touch.disposition == KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY &&
            touch.unsafe_via_pair_count == 1 &&
            touch.certified_empty_mask == 0,
        "touch: nonidentical touching VIA boxes did not invalidate proof");

    const Box boundary_metal = {-135, -65, 135, 65};
    const Box boundary_via = {-65, -65, 65, 65};
    const Result duplicate =
        run_case(
            "duplicate", boundary_metal,
            {boundary_via, boundary_via}, boundary_metal);
    require(
        duplicate.disposition ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE &&
            duplicate.duplicate_via_pair_count == 1 &&
            duplicate.unsafe_via_pair_count == 0 &&
            duplicate.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES,
        "duplicate: exact duplicate VIA boxes were not accepted");

    (void) run_case(
        "bad-digest", clean_metal, {clean_via}, clean_metal, true);

    std::cout
        << "VIA1-stack backend smoke passed: clean, union-clean, "
           "horizontal-gap, vertical-gap, negative-grid-boundary, miss, "
           "spacing, spacing-equality, diagonal-equality, size, touch, "
           "duplicate, bad-digest\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "VIA1-stack backend smoke failed: " << error.what() << "\n";
    return 1;
  }
}
