/*
 * Atomic CUDA empty certificate for the qualified M1/VIA1/M2 rule stack.
 *
 * This translation unit deliberately shares no device symbols with the
 * generic spatial backend.  VIA1 is expanded and indexed once, then retained
 * while one metal scratch/index is reused for M1 and M2.
 */

#include "dbCudaSpatialApi.h"
#include "dbCudaVia1StackDigest.h"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using ViaContext = klayout_cuda_spatial_via1_stack_context_v1;
using ViaBox = klayout_cuda_spatial_via1_stack_box_v1;
using ViaCell = klayout_cuda_spatial_via1_stack_cell_v1;
using ViaRequest = klayout_cuda_spatial_via1_stack_request_v1;
using ViaResult = klayout_cuda_spatial_via1_stack_result_v1;
using Clock = std::chrono::steady_clock;

static_assert(std::is_trivially_copyable<ViaContext>::value,
              "VIA1-stack contexts must remain POD");
static_assert(std::is_trivially_copyable<ViaBox>::value,
              "VIA1-stack boxes must remain POD");
static_assert(std::is_trivially_copyable<ViaCell>::value,
              "VIA1-stack cells must remain POD");
static_assert(sizeof(ViaContext) == 24, "unexpected VIA1 context ABI padding");
static_assert(sizeof(ViaBox) == 32, "unexpected VIA1 box ABI padding");
static_assert(sizeof(ViaCell) == 40, "unexpected VIA1 cell ABI padding");
static_assert(sizeof(ViaRequest) == 360, "unexpected VIA1 request ABI padding");
static_assert(sizeof(ViaResult) == 640, "unexpected VIA1 result ABI padding");

constexpr std::int64_t kCoordinateLimit = INT64_C(1000000000000);
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kContextThreads = 128;

enum ViaDeviceFlag : std::uint32_t {
  kViaTransformOverflow = 1u << 0,
  kViaGridCounterOverflow = 1u << 1,
  kViaGridCapacityExceeded = 1u << 2,
  kViaInvalidDeviceRecord = 1u << 3,
  kViaWorkCounterOverflow = 1u << 4,
};

enum ViaLayer : std::uint32_t {
  kMetal1Layer = 0,
  kVia1Layer = 1,
  kMetal2Layer = 2,
};

struct ViaGrid {
  std::int64_t base_x;
  std::int64_t base_y;
  std::int64_t cell_size;
  std::uint32_t width;
  std::uint32_t height;
};

struct ExpandCounters {
  unsigned long long visited;
  unsigned long long expanded;
  unsigned long long size_checked;
  unsigned long long size_violations;
};

struct ViaPairCounters {
  unsigned long long vias_queried;
  unsigned long long candidate_pairs;
  unsigned long long duplicate_pairs;
  unsigned long long unsafe_pairs;
  unsigned long long spacing_pairs;
  unsigned long long clean_pairs;
};

struct ProjectionCounters {
  unsigned long long vias_queried;
  unsigned long long candidate_boxes;
  unsigned long long union_candidate_visits;
  unsigned long long certified;
  unsigned long long misses;
};

struct ViaPipelineResult {
  std::uint32_t fallback_flags = 0;
  std::uint32_t device_flags = 0;
  std::uint32_t certified_empty_mask = 0;
  std::uint64_t grid_cells = 0;
  std::uint64_t via_expanded = 0;
  std::uint64_t via_size_checked = 0;
  std::uint64_t via_size_violations = 0;
  std::uint64_t metal1_expanded = 0;
  std::uint64_t metal2_expanded = 0;
  std::uint64_t via_memberships = 0;
  std::uint64_t metal1_memberships = 0;
  std::uint64_t metal2_memberships = 0;
  std::uint64_t via_candidates = 0;
  std::uint64_t via_pair_queried = 0;
  std::uint64_t duplicate_pairs = 0;
  std::uint64_t unsafe_pairs = 0;
  std::uint64_t spacing_pairs = 0;
  std::uint64_t clean_pairs = 0;
  std::uint64_t metal1_queried = 0;
  std::uint64_t metal1_candidates = 0;
  std::uint64_t metal1_certified = 0;
  std::uint64_t metal1_misses = 0;
  std::uint64_t metal2_queried = 0;
  std::uint64_t metal2_candidates = 0;
  std::uint64_t metal2_certified = 0;
  std::uint64_t metal2_misses = 0;
  std::uint64_t setup_ns = 0;
  std::uint64_t h2d_ns = 0;
  std::uint64_t via_expand_ns = 0;
  std::uint64_t via_grid_ns = 0;
  std::uint64_t via_query_ns = 0;
  std::uint64_t metal1_ns = 0;
  std::uint64_t metal2_ns = 0;
  std::uint64_t d2h_ns = 0;
};

std::mutex &via_pipeline_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::uint64_t
elapsed_ns(Clock::time_point begin, Clock::time_point end)
{
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

void
cuda_require(cudaError_t error, const char *operation)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(error));
  }
}

void
set_via_message(ViaResult *result, const char *message)
{
  std::snprintf(
      result->message, sizeof(result->message), "%s", message ? message : "");
}

bool
coordinate_qualified(std::int64_t value)
{
  return value >= -kCoordinateLimit && value <= kCoordinateLimit;
}

bool
array_size_fits(std::uint64_t count, std::size_t record_size)
{
  return record_size &&
         count <= std::numeric_limits<std::size_t>::max() / record_size;
}

bool
checked_add_u64(std::uint64_t first, std::uint64_t second,
                std::uint64_t *result)
{
  if (second > std::numeric_limits<std::uint64_t>::max() - first) return false;
  *result = first + second;
  return true;
}

bool
checked_add_i64(std::int64_t first, std::int64_t second,
                std::int64_t *result)
{
  if ((second > 0 && first > INT64_MAX - second) ||
      (second < 0 && first < INT64_MIN - second)) {
    return false;
  }
  *result = first + second;
  return true;
}

std::int64_t
floor_div_host(std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

std::uint64_t
cell_layer_begin(const ViaCell &cell, ViaLayer layer)
{
  switch (layer) {
  case kMetal1Layer: return cell.metal1_box_begin;
  case kVia1Layer: return cell.via1_box_begin;
  case kMetal2Layer: return cell.metal2_box_begin;
  }
  return UINT64_MAX;
}

std::uint32_t
cell_layer_count(const ViaCell &cell, ViaLayer layer)
{
  switch (layer) {
  case kMetal1Layer: return cell.metal1_box_count;
  case kVia1Layer: return cell.via1_box_count;
  case kMetal2Layer: return cell.metal2_box_count;
  }
  return 0;
}

bool
valid_layer_contexts(
    const ViaRequest &request, ViaLayer layer,
    const std::uint32_t *layer_contexts, std::uint64_t layer_context_count,
    const std::uint64_t *layer_offsets, std::uint64_t layer_offset_count,
    std::uint64_t flat_box_count)
{
  if (layer_offset_count != layer_context_count ||
      (!!layer_context_count &&
       (!layer_contexts || !layer_offsets))) {
    return false;
  }
  std::uint64_t list_id = 0;
  std::uint64_t flat_count = 0;
  for (std::uint64_t context_id = 0;
       context_id < request.context_count; ++context_id) {
    const ViaContext &context = request.contexts[context_id];
    const std::uint32_t count =
        cell_layer_count(request.cells[context.cell_id], layer);
    if (!count) continue;
    if (list_id >= layer_context_count ||
        layer_contexts[list_id] != context_id ||
        layer_offsets[list_id] != flat_count ||
        !checked_add_u64(flat_count, count, &flat_count)) {
      return false;
    }
    ++list_id;
  }
  return list_id == layer_context_count && flat_count == flat_box_count;
}

bool
valid_via_request(const ViaRequest &request)
{
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size < sizeof(request) ||
      request.opcode != KLAYOUT_CUDA_SPATIAL_VIA1_STACK_EMPTY ||
      request.option_flags !=
          KLAYOUT_CUDA_SPATIAL_VIA1_STACK_QUALIFIED_OPTIONS ||
      request.requested_mask != KLAYOUT_CUDA_SPATIAL_VIA1_STACK_ALL_RULES ||
      request.dbu_per_micron != 2000 || request.device < 0 ||
      request.reserved0 != 0 || request.reserved1[0] != 0 ||
      request.reserved1[1] != 0 ||
      request.enclosure_distance != 70 ||
      request.cut_width != 130 || request.cut_height != 130 ||
      request.spacing_distance != 150 ||
      request.grid_cell_size != 2000 ||
      !request.context_count || !request.contexts ||
      !request.cell_count || !request.cells ||
      !request.box_count || !request.boxes ||
      !request.metal1_context_count ||
      !request.via1_context_count ||
      !request.metal2_context_count ||
      !request.flat_metal1_box_count ||
      !request.flat_via1_box_count ||
      !request.flat_metal2_box_count ||
      request.context_count > request.max_contexts ||
      request.context_count > UINT32_MAX ||
      request.metal1_context_count > UINT32_MAX ||
      request.via1_context_count > UINT32_MAX ||
      request.metal2_context_count > UINT32_MAX ||
      request.cell_count > UINT32_MAX ||
      request.flat_metal1_box_count > UINT32_MAX ||
      request.flat_via1_box_count > UINT32_MAX ||
      request.flat_metal2_box_count > UINT32_MAX ||
      !request.max_contexts || !request.max_grid_cells ||
      !request.max_metal_memberships || !request.max_via_memberships ||
      !request.max_pair_work ||
      !array_size_fits(request.context_count, sizeof(ViaContext)) ||
      !array_size_fits(
          request.metal1_context_count, sizeof(std::uint32_t)) ||
      !array_size_fits(
          request.metal1_offset_count, sizeof(std::uint64_t)) ||
      !array_size_fits(request.via1_context_count, sizeof(std::uint32_t)) ||
      !array_size_fits(request.via1_offset_count, sizeof(std::uint64_t)) ||
      !array_size_fits(
          request.metal2_context_count, sizeof(std::uint32_t)) ||
      !array_size_fits(
          request.metal2_offset_count, sizeof(std::uint64_t)) ||
      !array_size_fits(request.cell_count, sizeof(ViaCell)) ||
      !array_size_fits(request.box_count, sizeof(ViaBox)) ||
      request.scene_left >= request.scene_right ||
      request.scene_bottom >= request.scene_top ||
      !coordinate_qualified(request.scene_left) ||
      !coordinate_qualified(request.scene_bottom) ||
      !coordinate_qualified(request.scene_right) ||
      !coordinate_qualified(request.scene_top)) {
    return false;
  }

  std::uint64_t next_box = 0;
  for (std::uint64_t cell_id = 0; cell_id < request.cell_count; ++cell_id) {
    const ViaCell &cell = request.cells[cell_id];
    if (cell.reserved0 != 0 ||
        cell.metal1_box_begin != next_box ||
        !checked_add_u64(
            next_box, cell.metal1_box_count, &next_box) ||
        cell.via1_box_begin != next_box ||
        !checked_add_u64(
            next_box, cell.via1_box_count, &next_box) ||
        cell.metal2_box_begin != next_box ||
        !checked_add_u64(
            next_box, cell.metal2_box_count, &next_box)) {
      return false;
    }
    for (ViaLayer layer : {kMetal1Layer, kVia1Layer, kMetal2Layer}) {
      const std::uint64_t begin = cell_layer_begin(cell, layer);
      const std::uint64_t count = cell_layer_count(cell, layer);
      if (begin > request.box_count || count > request.box_count - begin) {
        return false;
      }
    }
  }
  if (next_box != request.box_count) return false;

  for (std::uint64_t box_id = 0; box_id < request.box_count; ++box_id) {
    const ViaBox &box = request.boxes[box_id];
    if (box.left >= box.right || box.bottom >= box.top ||
        !coordinate_qualified(box.left) ||
        !coordinate_qualified(box.bottom) ||
        !coordinate_qualified(box.right) ||
        !coordinate_qualified(box.top)) {
      return false;
    }
  }

  for (std::uint64_t context_id = 0;
       context_id < request.context_count; ++context_id) {
    const ViaContext &context = request.contexts[context_id];
    if (context.cell_id >= request.cell_count ||
        context.transform_code >= 8 ||
        !coordinate_qualified(context.tx) ||
        !coordinate_qualified(context.ty)) {
      return false;
    }
  }
  if (request.contexts[0].tx != 0 || request.contexts[0].ty != 0 ||
      request.contexts[0].transform_code != 0) {
    return false;
  }

  if (!valid_layer_contexts(
          request, kMetal1Layer, request.metal1_contexts,
          request.metal1_context_count, request.metal1_offsets,
          request.metal1_offset_count,
          request.flat_metal1_box_count) ||
      !valid_layer_contexts(
          request, kVia1Layer, request.via1_contexts,
          request.via1_context_count, request.via1_offsets,
          request.via1_offset_count, request.flat_via1_box_count) ||
      !valid_layer_contexts(
          request, kMetal2Layer, request.metal2_contexts,
          request.metal2_context_count, request.metal2_offsets,
          request.metal2_offset_count,
          request.flat_metal2_box_count)) {
    return false;
  }
  std::array<std::uint8_t, 32> digest;
  return db::cuda_via1_stack_digest::request_digest(request, digest) &&
         std::equal(
             digest.begin(), digest.end(), request.scene_digest);
}

void
echo_via_request(const ViaRequest &request, ViaResult *result)
{
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->requested_mask = request.requested_mask;
  result->dbu_per_micron = request.dbu_per_micron;
  result->enclosure_distance = request.enclosure_distance;
  result->cut_width = request.cut_width;
  result->cut_height = request.cut_height;
  result->spacing_distance = request.spacing_distance;
  result->grid_cell_size = request.grid_cell_size;
  std::copy(
      request.scene_digest, request.scene_digest + 32, result->scene_digest);
  result->context_count = request.context_count;
  result->metal1_context_count = request.metal1_context_count;
  result->via1_context_count = request.via1_context_count;
  result->metal2_context_count = request.metal2_context_count;
  result->cell_count = request.cell_count;
  result->box_count = request.box_count;
  result->flat_metal1_box_count = request.flat_metal1_box_count;
  result->flat_via1_box_count = request.flat_via1_box_count;
  result->flat_metal2_box_count = request.flat_metal2_box_count;
}

__device__ bool
via_negate_checked(std::int64_t value, std::int64_t *result)
{
  if (value == INT64_MIN) return false;
  *result = -value;
  return true;
}

__device__ bool
via_add_checked(
    std::int64_t first, std::int64_t second, std::int64_t *result)
{
  if ((second > 0 && first > INT64_MAX - second) ||
      (second < 0 && first < INT64_MIN - second)) {
    return false;
  }
  *result = first + second;
  return true;
}

__device__ bool
via_transform_point_checked(
    const ViaContext &context, std::int64_t x, std::int64_t y,
    std::int64_t *output_x, std::int64_t *output_y)
{
  std::int64_t transformed_x = 0;
  std::int64_t transformed_y = 0;
  switch (context.transform_code) {
  case 0: transformed_x = x; transformed_y = y; break;
  case 1:
    if (!via_negate_checked(y, &transformed_x)) return false;
    transformed_y = x;
    break;
  case 2:
    if (!via_negate_checked(x, &transformed_x) ||
        !via_negate_checked(y, &transformed_y)) {
      return false;
    }
    break;
  case 3:
    transformed_x = y;
    if (!via_negate_checked(x, &transformed_y)) return false;
    break;
  case 4:
    transformed_x = x;
    if (!via_negate_checked(y, &transformed_y)) return false;
    break;
  case 5: transformed_x = y; transformed_y = x; break;
  case 6:
    if (!via_negate_checked(x, &transformed_x)) return false;
    transformed_y = y;
    break;
  case 7:
    if (!via_negate_checked(y, &transformed_x) ||
        !via_negate_checked(x, &transformed_y)) {
      return false;
    }
    break;
  default: return false;
  }
  return via_add_checked(transformed_x, context.tx, output_x) &&
         via_add_checked(transformed_y, context.ty, output_y);
}

__device__ bool
via_transform_box_checked(
    const ViaContext &context, const ViaBox &source, ViaBox *destination)
{
  const std::int64_t xs[4] = {
      source.left, source.left, source.right, source.right};
  const std::int64_t ys[4] = {
      source.bottom, source.top, source.bottom, source.top};
  ViaBox transformed = {INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN};
  for (int corner = 0; corner < 4; ++corner) {
    std::int64_t x = 0;
    std::int64_t y = 0;
    if (!via_transform_point_checked(
            context, xs[corner], ys[corner], &x, &y)) {
      return false;
    }
    transformed.left = min(transformed.left, x);
    transformed.bottom = min(transformed.bottom, y);
    transformed.right = max(transformed.right, x);
    transformed.top = max(transformed.top, y);
  }
  if (transformed.left >= transformed.right ||
      transformed.bottom >= transformed.top) {
    return false;
  }
  *destination = transformed;
  return true;
}

__device__ std::uint64_t
via_cell_begin(const ViaCell &cell, std::uint32_t layer)
{
  if (layer == kMetal1Layer) return cell.metal1_box_begin;
  if (layer == kVia1Layer) return cell.via1_box_begin;
  return cell.metal2_box_begin;
}

__device__ std::uint32_t
via_cell_count(const ViaCell &cell, std::uint32_t layer)
{
  if (layer == kMetal1Layer) return cell.metal1_box_count;
  if (layer == kVia1Layer) return cell.via1_box_count;
  return cell.metal2_box_count;
}

__global__ void
via_transform_semantics_gate(std::uint32_t *status)
{
  const std::uint32_t code = threadIdx.x;
  if (blockIdx.x || code >= 8) return;
  const ViaContext context = {13, -7, 0, code};
  const ViaBox source = {-10, -20, 30, 40};
  const ViaBox expected[8] = {
      {3, -27, 43, 33},
      {-27, -17, 33, 23},
      {-17, -47, 23, 13},
      {-7, -37, 53, 3},
      {3, -47, 43, 13},
      {-7, -17, 53, 23},
      {-17, -27, 23, 33},
      {-27, -37, 33, 3}};
  ViaBox transformed{};
  if (!via_transform_box_checked(context, source, &transformed) ||
      transformed.left != expected[code].left ||
      transformed.bottom != expected[code].bottom ||
      transformed.right != expected[code].right ||
      transformed.top != expected[code].top) {
    atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
  }
}

__global__ void
via_expand_boxes_kernel(
    const ViaContext *contexts, const std::uint32_t *layer_contexts,
    const std::uint64_t *layer_offsets, const ViaCell *cells,
    const ViaBox *templates, std::uint32_t layer_context_count,
    std::uint32_t layer, std::int64_t scene_left,
    std::int64_t scene_bottom, std::int64_t scene_right,
    std::int64_t scene_top, std::int64_t cut_width,
    std::int64_t cut_height, ViaBox *destination,
    ExpandCounters *counters, std::uint32_t *status)
{
  const std::uint32_t list_id = blockIdx.x;
  if (list_id >= layer_context_count) return;
  const ViaContext context = contexts[layer_contexts[list_id]];
  const ViaCell cell = cells[context.cell_id];
  const std::uint32_t count = via_cell_count(cell, layer);
  const std::uint64_t begin = via_cell_begin(cell, layer);
  unsigned long long local_visited = 0;
  unsigned long long local_expanded = 0;
  unsigned long long local_size_checked = 0;
  unsigned long long local_size_violations = 0;
  for (std::uint32_t local = threadIdx.x;
       local < count; local += blockDim.x) {
    ++local_visited;
    ViaBox box{};
    if (!via_transform_box_checked(context, templates[begin + local], &box)) {
      atomicOr(status, std::uint32_t(kViaTransformOverflow));
      continue;
    }
    if (box.left < scene_left || box.bottom < scene_bottom ||
        box.right > scene_right || box.top > scene_top) {
      atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
      continue;
    }
    if (layer == kVia1Layer &&
        (box.right - box.left != cut_width ||
         box.top - box.bottom != cut_height)) {
      ++local_size_violations;
    }
    if (layer == kVia1Layer) ++local_size_checked;
    destination[layer_offsets[list_id] + local] = box;
    ++local_expanded;
  }
  if (local_visited) atomicAdd(&counters->visited, local_visited);
  if (local_expanded) atomicAdd(&counters->expanded, local_expanded);
  if (local_size_checked) {
    atomicAdd(&counters->size_checked, local_size_checked);
  }
  if (local_size_violations) {
    atomicAdd(&counters->size_violations, local_size_violations);
  }
}

__device__ std::int64_t
via_floor_div(std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

__device__ bool
via_box_span(
    const ViaBox &box, const ViaGrid &grid, std::int64_t *x0,
    std::int64_t *y0, std::int64_t *x1, std::int64_t *y1)
{
  *x0 = via_floor_div(box.left, grid.cell_size);
  *x1 = via_floor_div(box.right, grid.cell_size);
  *y0 = via_floor_div(box.bottom, grid.cell_size);
  *y1 = via_floor_div(box.top, grid.cell_size);
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  return *x0 >= grid.base_x && *x1 <= maximum_x &&
         *y0 >= grid.base_y && *y1 <= maximum_y;
}

__device__ bool
via_expanded_box_span(
    const ViaBox &box, const ViaGrid &grid, std::int64_t expansion,
    std::int64_t *x0, std::int64_t *y0,
    std::int64_t *x1, std::int64_t *y1)
{
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
  if (!via_add_checked(box.left, -expansion, &left) ||
      !via_add_checked(box.bottom, -expansion, &bottom) ||
      !via_add_checked(box.right, expansion, &right) ||
      !via_add_checked(box.top, expansion, &top)) {
    return false;
  }
  const ViaBox expanded = {left, bottom, right, top};
  return via_box_span(expanded, grid, x0, y0, x1, y1);
}

__device__ std::uint64_t
via_grid_index(const ViaGrid &grid, std::int64_t x, std::int64_t y)
{
  return static_cast<std::uint64_t>(y - grid.base_y) * grid.width +
         static_cast<std::uint64_t>(x - grid.base_x);
}

__global__ void
via_count_grid_kernel(
    const ViaBox *boxes, std::uint32_t box_count, ViaGrid grid,
    std::uint32_t *counts, unsigned long long *total,
    std::uint32_t *status)
{
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       id < box_count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!via_box_span(boxes[id], grid, &x0, &y0, &x1, &y1)) {
      atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t index = via_grid_index(grid, x, y);
        const std::uint32_t previous = atomicAdd(counts + index, 1u);
        if (previous == UINT32_MAX) {
          atomicOr(status, std::uint32_t(kViaGridCounterOverflow));
        }
        const unsigned long long previous_total = atomicAdd(total, 1ull);
        if (previous_total == ULLONG_MAX) {
          atomicOr(status, std::uint32_t(kViaGridCounterOverflow));
        }
      }
    }
  }
}

__global__ void
via_fill_grid_kernel(
    const ViaBox *boxes, std::uint32_t box_count, ViaGrid grid,
    std::uint32_t *cursors, std::uint32_t *members,
    std::uint64_t member_capacity, std::uint32_t *status)
{
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       id < box_count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!via_box_span(boxes[id], grid, &x0, &y0, &x1, &y1)) {
      atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t index = via_grid_index(grid, x, y);
        const std::uint32_t position = atomicAdd(cursors + index, 1u);
        if (position >= member_capacity) {
          atomicOr(status, std::uint32_t(kViaGridCapacityExceeded));
        } else {
          members[position] = static_cast<std::uint32_t>(id);
        }
      }
    }
  }
}

__global__ void
via_validate_grid_kernel(
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *cursors, std::uint64_t grid_cells,
    std::uint32_t *status)
{
  for (std::uint64_t cell =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < grid_cells;
       cell += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t expected =
        static_cast<std::uint64_t>(offsets[cell]) + counts[cell];
    if (expected > UINT32_MAX || cursors[cell] != expected) {
      atomicOr(status, std::uint32_t(kViaGridCounterOverflow));
    }
  }
}

__device__ std::uint64_t
via_ordered_distance(std::int64_t high, std::int64_t low)
{
  return static_cast<std::uint64_t>(high) -
         static_cast<std::uint64_t>(low);
}

__global__ void
via_query_pairs_kernel(
    const ViaBox *vias, std::uint32_t via_count, ViaGrid grid,
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *members, std::int64_t spacing,
    ViaPairCounters *counters, std::uint32_t *status)
{
  unsigned long long local_queried = 0;
  unsigned long long local_candidates = 0;
  unsigned long long local_duplicates = 0;
  unsigned long long local_unsafe = 0;
  unsigned long long local_spacing = 0;
  unsigned long long local_clean = 0;
  for (std::uint64_t via_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       via_id < via_count;
       via_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    ++local_queried;
    const ViaBox via = vias[via_id];
    std::int64_t query_x0 = 0, query_y0 = 0;
    std::int64_t query_x1 = 0, query_y1 = 0;
    if (!via_expanded_box_span(
            via, grid, spacing, &query_x0, &query_y0,
            &query_x1, &query_y1)) {
      atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
      continue;
    }
    for (std::int64_t y = query_y0; y <= query_y1; ++y) {
      for (std::int64_t x = query_x0; x <= query_x1; ++x) {
        const std::uint64_t cell_id = via_grid_index(grid, x, y);
        const std::uint32_t begin = offsets[cell_id];
        const std::uint32_t end = begin + counts[cell_id];
        for (std::uint32_t position = begin; position < end; ++position) {
          const std::uint32_t other_id = members[position];
          if (other_id <= via_id) {
            if (other_id >= via_count) {
              atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
            }
            continue;
          }
          if (other_id >= via_count) {
            atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
            continue;
          }
          const ViaBox other = vias[other_id];
          std::int64_t other_x0 = 0, other_y0 = 0;
          std::int64_t other_x1 = 0, other_y1 = 0;
          if (!via_box_span(
                  other, grid, &other_x0, &other_y0,
                  &other_x1, &other_y1)) {
            atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
            continue;
          }
          if (x != max(query_x0, other_x0) ||
              y != max(query_y0, other_y0)) {
            continue;
          }
          ++local_candidates;
          const std::uint64_t dx =
              via.right < other.left
                  ? via_ordered_distance(other.left, via.right)
                  : (other.right < via.left
                         ? via_ordered_distance(via.left, other.right)
                         : 0);
          const std::uint64_t dy =
              via.top < other.bottom
                  ? via_ordered_distance(other.bottom, via.top)
                  : (other.top < via.bottom
                         ? via_ordered_distance(via.bottom, other.top)
                         : 0);
          const std::uint64_t spacing_u =
              static_cast<std::uint64_t>(spacing);
          if (dx >= spacing_u || dy >= spacing_u) {
            ++local_clean;
          } else if (!dx && !dy) {
            const bool identical =
                via.left == other.left && via.bottom == other.bottom &&
                via.right == other.right && via.top == other.top;
            if (identical) {
              ++local_duplicates;
            } else {
              ++local_unsafe;
            }
          } else if (dx * dx + dy * dy < spacing_u * spacing_u) {
            ++local_spacing;
          } else {
            ++local_clean;
          }
        }
      }
    }
  }
  if (local_queried) atomicAdd(&counters->vias_queried, local_queried);
  if (local_candidates) {
    atomicAdd(&counters->candidate_pairs, local_candidates);
  }
  if (local_duplicates) {
    atomicAdd(&counters->duplicate_pairs, local_duplicates);
  }
  if (local_unsafe) atomicAdd(&counters->unsafe_pairs, local_unsafe);
  if (local_spacing) atomicAdd(&counters->spacing_pairs, local_spacing);
  if (local_clean) atomicAdd(&counters->clean_pairs, local_clean);
}

__device__ bool
via_projection_certificate(
    const ViaBox &metal, const ViaBox &via, std::int64_t distance)
{
  if (metal.left > via.left || metal.right < via.right ||
      metal.bottom > via.bottom || metal.top < via.top) {
    return false;
  }
  return (via.left - metal.left >= distance &&
          metal.right - via.right >= distance) ||
         (via.bottom - metal.bottom >= distance &&
          metal.top - via.top >= distance);
}

__device__ void
via_count_union_candidate(
    unsigned long long *candidate_visits, std::uint32_t *status)
{
  if (*candidate_visits == ULLONG_MAX) {
    atomicOr(status, std::uint32_t(kViaWorkCounterOverflow));
  } else {
    ++*candidate_visits;
  }
}

__device__ bool
via_union_covers_horizontal_strip(
    const ViaBox &via, const ViaBox *metals, std::uint32_t metal_count,
    ViaGrid grid, const std::uint32_t *counts,
    const std::uint32_t *offsets, const std::uint32_t *members,
    std::int64_t distance, unsigned long long *candidate_visits,
    std::uint32_t *status)
{
  const std::int64_t target_left = via.left - distance;
  const std::int64_t target_right = via.right + distance;
  const std::int64_t grid_x0 =
      via_floor_div(target_left, grid.cell_size);
  const std::int64_t grid_x1 =
      via_floor_div(target_right - 1, grid.cell_size);
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  if (grid_x0 < grid.base_x || grid_x1 > maximum_x) return false;

  for (std::int64_t y = via.bottom; y < via.top; ++y) {
    const std::int64_t grid_y = via_floor_div(y, grid.cell_size);
    if (grid_y < grid.base_y || grid_y > maximum_y) return false;
    std::int64_t cursor = target_left;
    while (cursor < target_right) {
      std::int64_t farthest = cursor;
      for (std::int64_t grid_x = grid_x0;
           grid_x <= grid_x1; ++grid_x) {
        const std::uint64_t cell_id =
            via_grid_index(grid, grid_x, grid_y);
        const std::uint32_t begin = offsets[cell_id];
        const std::uint32_t end = begin + counts[cell_id];
        for (std::uint32_t position = begin; position < end; ++position) {
          via_count_union_candidate(candidate_visits, status);
          const std::uint32_t metal_id = members[position];
          if (metal_id >= metal_count) {
            atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
            continue;
          }
          const ViaBox metal = metals[metal_id];
          if (metal.bottom <= y && metal.top >= y + 1 &&
              metal.left <= cursor && metal.right > farthest) {
            farthest = min(metal.right, target_right);
          }
        }
      }
      if (farthest == cursor) return false;
      cursor = farthest;
    }
  }
  return true;
}

__device__ bool
via_union_covers_vertical_strip(
    const ViaBox &via, const ViaBox *metals, std::uint32_t metal_count,
    ViaGrid grid, const std::uint32_t *counts,
    const std::uint32_t *offsets, const std::uint32_t *members,
    std::int64_t distance, unsigned long long *candidate_visits,
    std::uint32_t *status)
{
  const std::int64_t target_bottom = via.bottom - distance;
  const std::int64_t target_top = via.top + distance;
  const std::int64_t grid_y0 =
      via_floor_div(target_bottom, grid.cell_size);
  const std::int64_t grid_y1 =
      via_floor_div(target_top - 1, grid.cell_size);
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  if (grid_y0 < grid.base_y || grid_y1 > maximum_y) return false;

  for (std::int64_t x = via.left; x < via.right; ++x) {
    const std::int64_t grid_x = via_floor_div(x, grid.cell_size);
    if (grid_x < grid.base_x || grid_x > maximum_x) return false;
    std::int64_t cursor = target_bottom;
    while (cursor < target_top) {
      std::int64_t farthest = cursor;
      for (std::int64_t grid_y = grid_y0;
           grid_y <= grid_y1; ++grid_y) {
        const std::uint64_t cell_id =
            via_grid_index(grid, grid_x, grid_y);
        const std::uint32_t begin = offsets[cell_id];
        const std::uint32_t end = begin + counts[cell_id];
        for (std::uint32_t position = begin; position < end; ++position) {
          via_count_union_candidate(candidate_visits, status);
          const std::uint32_t metal_id = members[position];
          if (metal_id >= metal_count) {
            atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
            continue;
          }
          const ViaBox metal = metals[metal_id];
          if (metal.left <= x && metal.right >= x + 1 &&
              metal.bottom <= cursor && metal.top > farthest) {
            farthest = min(metal.top, target_top);
          }
        }
      }
      if (farthest == cursor) return false;
      cursor = farthest;
    }
  }
  return true;
}

__device__ bool
via_projection_union_certificate(
    const ViaBox &via, const ViaBox *metals, std::uint32_t metal_count,
    ViaGrid grid, const std::uint32_t *counts,
    const std::uint32_t *offsets, const std::uint32_t *members,
    std::int64_t distance, unsigned long long *candidate_visits,
    std::uint32_t *status)
{
  return via_union_covers_horizontal_strip(
             via, metals, metal_count, grid, counts, offsets, members,
             distance, candidate_visits, status) ||
         via_union_covers_vertical_strip(
             via, metals, metal_count, grid, counts, offsets, members,
             distance, candidate_visits, status);
}

__global__ void
via_query_projection_kernel(
    const ViaBox *vias, std::uint32_t via_count,
    const ViaBox *metals, std::uint32_t metal_count, ViaGrid grid,
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *members, std::int64_t distance,
    ProjectionCounters *counters, std::uint32_t *status)
{
  unsigned long long local_queried = 0;
  unsigned long long local_candidates = 0;
  unsigned long long local_union_candidates = 0;
  unsigned long long local_certified = 0;
  unsigned long long local_misses = 0;
  for (std::uint64_t via_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       via_id < via_count;
       via_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    ++local_queried;
    const ViaBox via = vias[via_id];
    const std::int64_t center_x = via.left + (via.right - via.left) / 2;
    const std::int64_t center_y = via.bottom + (via.top - via.bottom) / 2;
    const std::int64_t grid_x = via_floor_div(center_x, grid.cell_size);
    const std::int64_t grid_y = via_floor_div(center_y, grid.cell_size);
    const std::int64_t maximum_x =
        grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
    const std::int64_t maximum_y =
        grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
    bool certified = false;
    if (grid_x < grid.base_x || grid_x > maximum_x ||
        grid_y < grid.base_y || grid_y > maximum_y) {
      atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
    } else {
      const std::uint64_t cell_id = via_grid_index(grid, grid_x, grid_y);
      const std::uint32_t begin = offsets[cell_id];
      const std::uint32_t end = begin + counts[cell_id];
      for (std::uint32_t position = begin; position < end; ++position) {
        const std::uint32_t metal_id = members[position];
        if (metal_id >= metal_count) {
          atomicOr(status, std::uint32_t(kViaInvalidDeviceRecord));
          continue;
        }
        ++local_candidates;
        if (via_projection_certificate(metals[metal_id], via, distance)) {
          certified = true;
          break;
        }
      }
    }
    if (!certified) {
      certified = via_projection_union_certificate(
          via, metals, metal_count, grid, counts, offsets, members,
          distance, &local_union_candidates, status);
    }
    if (certified) {
      ++local_certified;
    } else {
      ++local_misses;
    }
  }
  if (local_queried) atomicAdd(&counters->vias_queried, local_queried);
  if (local_candidates) {
    atomicAdd(&counters->candidate_boxes, local_candidates);
  }
  if (local_union_candidates) {
    const unsigned long long previous = atomicAdd(
        &counters->union_candidate_visits, local_union_candidates);
    if (previous > ULLONG_MAX - local_union_candidates) {
      atomicOr(status, std::uint32_t(kViaWorkCounterOverflow));
    }
  }
  if (local_certified) {
    atomicAdd(&counters->certified, local_certified);
  }
  if (local_misses) atomicAdd(&counters->misses, local_misses);
}

template <class T>
T *
device_data(thrust::device_vector<T> &values)
{
  return values.empty() ? nullptr : thrust::raw_pointer_cast(values.data());
}

unsigned int
work_blocks(std::uint64_t count)
{
  return static_cast<unsigned int>(
      std::min<std::uint64_t>(65535, (count + kThreads - 1) / kThreads));
}

std::uint32_t
copy_device_status(thrust::device_vector<std::uint32_t> &status,
                   const char *operation)
{
  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(
          &host_status, device_data(status), sizeof(host_status),
          cudaMemcpyDeviceToHost),
      operation);
  return host_status;
}

bool
build_box_grid(
    const ViaBox *boxes, std::uint32_t box_count, const ViaGrid &grid,
    std::uint64_t grid_cells, std::uint64_t maximum_memberships,
    thrust::device_vector<std::uint32_t> &counts,
    thrust::device_vector<std::uint32_t> &offsets,
    thrust::device_vector<std::uint32_t> &cursors,
    thrust::device_vector<std::uint32_t> &members,
    thrust::device_vector<unsigned long long> &membership_total,
    thrust::device_vector<std::uint32_t> &status,
    std::uint64_t *host_memberships, ViaPipelineResult *result,
    const char *label)
{
  cuda_require(
      cudaMemset(device_data(counts), 0, grid_cells * sizeof(std::uint32_t)),
      "VIA1-stack grid count clear");
  cuda_require(
      cudaMemset(
          device_data(membership_total), 0, sizeof(unsigned long long)),
      "VIA1-stack membership clear");
  if (!box_count) {
    cuda_require(
        cudaMemset(
            device_data(offsets), 0,
            (grid_cells + 1) * sizeof(std::uint32_t)),
        "VIA1-stack empty offsets clear");
    cuda_require(
        cudaMemset(
            device_data(cursors), 0,
            grid_cells * sizeof(std::uint32_t)),
        "VIA1-stack empty cursors clear");
    members.clear();
    *host_memberships = 0;
    return true;
  }

  const unsigned int blocks = work_blocks(box_count);
  via_count_grid_kernel<<<blocks, kThreads>>>(
      boxes, box_count, grid, device_data(counts),
      device_data(membership_total), device_data(status));
  cuda_require(cudaGetLastError(), "VIA1-stack grid count launch");
  cuda_require(cudaDeviceSynchronize(), "VIA1-stack grid count synchronize");
  unsigned long long memberships = 0;
  cuda_require(
      cudaMemcpy(
          &memberships, device_data(membership_total), sizeof(memberships),
          cudaMemcpyDeviceToHost),
      "VIA1-stack membership D2H");
  result->device_flags =
      copy_device_status(status, "VIA1-stack grid-count status D2H");
  *host_memberships = memberships;
  if (result->device_flags) {
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return false;
  }
  if (memberships < box_count || memberships > maximum_memberships ||
      memberships > UINT32_MAX) {
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY;
    return false;
  }

  thrust::exclusive_scan(
      thrust::device, counts.begin(), counts.end(), offsets.begin());
  const std::uint32_t terminal = static_cast<std::uint32_t>(memberships);
  cuda_require(
      cudaMemcpy(
          device_data(offsets) + grid_cells, &terminal, sizeof(terminal),
          cudaMemcpyHostToDevice),
      "VIA1-stack terminal offset H2D");
  cuda_require(
      cudaMemcpy(
          device_data(cursors), device_data(offsets),
          grid_cells * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice),
      "VIA1-stack offsets-to-cursors D2D");
  members.resize(memberships);
  via_fill_grid_kernel<<<blocks, kThreads>>>(
      boxes, box_count, grid, device_data(cursors), device_data(members),
      memberships, device_data(status));
  cuda_require(cudaGetLastError(), "VIA1-stack grid fill launch");
  const unsigned int grid_blocks = work_blocks(grid_cells);
  via_validate_grid_kernel<<<grid_blocks, kThreads>>>(
      device_data(counts), device_data(offsets), device_data(cursors),
      grid_cells, device_data(status));
  cuda_require(cudaGetLastError(), "VIA1-stack grid validation launch");
  cuda_require(cudaDeviceSynchronize(), "VIA1-stack grid build synchronize");
  result->device_flags =
      copy_device_status(status, "VIA1-stack grid-build status D2H");
  if (result->device_flags) {
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return false;
  }
  (void) label;
  return true;
}

ViaPipelineResult
run_via_pipeline(const ViaRequest &request)
{
  ViaPipelineResult result;

  std::int64_t expanded_left = 0;
  std::int64_t expanded_bottom = 0;
  std::int64_t expanded_right = 0;
  std::int64_t expanded_top = 0;
  if (!checked_add_i64(
          request.scene_left, -request.spacing_distance, &expanded_left) ||
      !checked_add_i64(
          request.scene_bottom, -request.spacing_distance,
          &expanded_bottom) ||
      !checked_add_i64(
          request.scene_right, request.spacing_distance, &expanded_right) ||
      !checked_add_i64(
          request.scene_top, request.spacing_distance, &expanded_top)) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW;
    return result;
  }
  const std::int64_t base_x =
      floor_div_host(expanded_left, request.grid_cell_size);
  const std::int64_t base_y =
      floor_div_host(expanded_bottom, request.grid_cell_size);
  const std::int64_t maximum_x =
      floor_div_host(expanded_right, request.grid_cell_size);
  const std::int64_t maximum_y =
      floor_div_host(expanded_top, request.grid_cell_size);
  const __int128 width_128 =
      static_cast<__int128>(maximum_x) - base_x + 1;
  const __int128 height_128 =
      static_cast<__int128>(maximum_y) - base_y + 1;
  if (width_128 <= 0 || height_128 <= 0 ||
      width_128 > UINT32_MAX || height_128 > UINT32_MAX ||
      width_128 * height_128 > UINT32_MAX ||
      width_128 * height_128 >
          static_cast<__int128>(request.max_grid_cells)) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY;
    return result;
  }
  const std::uint64_t width = static_cast<std::uint64_t>(width_128);
  const std::uint64_t height = static_cast<std::uint64_t>(height_128);
  result.grid_cells = width * height;
  const ViaGrid grid = {
      base_x, base_y, request.grid_cell_size,
      static_cast<std::uint32_t>(width),
      static_cast<std::uint32_t>(height)};

  const auto setup_begin = Clock::now();
  cuda_require(cudaSetDevice(request.device), "VIA1-stack cudaSetDevice");
  cuda_require(cudaFree(nullptr), "VIA1-stack CUDA context initialization");
  cudaDeviceProp properties{};
  cuda_require(
      cudaGetDeviceProperties(&properties, request.device),
      "VIA1-stack cudaGetDeviceProperties");
  const std::uint64_t maximum_grid_x =
      static_cast<std::uint64_t>(properties.maxGridSize[0]);
  if (request.metal1_context_count > maximum_grid_x ||
      request.via1_context_count > maximum_grid_x ||
      request.metal2_context_count > maximum_grid_x) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    return result;
  }

  thrust::device_vector<ViaContext> contexts(request.context_count);
  thrust::device_vector<std::uint32_t>
      metal1_contexts(request.metal1_context_count);
  thrust::device_vector<std::uint64_t>
      metal1_offsets(request.metal1_offset_count);
  thrust::device_vector<std::uint32_t>
      via1_contexts(request.via1_context_count);
  thrust::device_vector<std::uint64_t>
      via1_offsets(request.via1_offset_count);
  thrust::device_vector<std::uint32_t>
      metal2_contexts(request.metal2_context_count);
  thrust::device_vector<std::uint64_t>
      metal2_offsets(request.metal2_offset_count);
  thrust::device_vector<ViaCell> cells(request.cell_count);
  thrust::device_vector<ViaBox> templates(request.box_count);
  thrust::device_vector<ViaBox> vias(request.flat_via1_box_count);
  const std::uint64_t metal_scratch_count =
      std::max(
          request.flat_metal1_box_count, request.flat_metal2_box_count);
  thrust::device_vector<ViaBox> metal_scratch(metal_scratch_count);
  thrust::device_vector<std::uint32_t> status(1, 0);
  thrust::device_vector<ExpandCounters> expand_counters(1);
  thrust::device_vector<ViaPairCounters> pair_counters(1);
  thrust::device_vector<ProjectionCounters> projection_counters(1);
  thrust::device_vector<unsigned long long> membership_total(1);

  thrust::device_vector<std::uint32_t>
      via_counts(result.grid_cells);
  thrust::device_vector<std::uint32_t>
      via_offsets(result.grid_cells + 1);
  thrust::device_vector<std::uint32_t>
      via_cursors(result.grid_cells);
  thrust::device_vector<std::uint32_t> via_members;
  thrust::device_vector<std::uint32_t>
      metal_counts(result.grid_cells);
  thrust::device_vector<std::uint32_t>
      metal_offsets(result.grid_cells + 1);
  thrust::device_vector<std::uint32_t>
      metal_cursors(result.grid_cells);
  thrust::device_vector<std::uint32_t> metal_members;
  result.setup_ns = elapsed_ns(setup_begin, Clock::now());

  const auto h2d_begin = Clock::now();
  const auto copy_to_device =
      [](void *destination, const void *source, std::uint64_t count,
         std::size_t record_size, const char *operation) {
        if (!count) return;
        cuda_require(
            cudaMemcpy(
                destination, source, count * record_size,
                cudaMemcpyHostToDevice),
            operation);
      };
  copy_to_device(
      device_data(contexts), request.contexts, request.context_count,
      sizeof(ViaContext), "VIA1-stack context H2D");
  copy_to_device(
      device_data(metal1_contexts), request.metal1_contexts,
      request.metal1_context_count, sizeof(std::uint32_t),
      "VIA1-stack M1-context H2D");
  copy_to_device(
      device_data(metal1_offsets), request.metal1_offsets,
      request.metal1_offset_count, sizeof(std::uint64_t),
      "VIA1-stack M1-offset H2D");
  copy_to_device(
      device_data(via1_contexts), request.via1_contexts,
      request.via1_context_count, sizeof(std::uint32_t),
      "VIA1-stack VIA-context H2D");
  copy_to_device(
      device_data(via1_offsets), request.via1_offsets,
      request.via1_offset_count, sizeof(std::uint64_t),
      "VIA1-stack VIA-offset H2D");
  copy_to_device(
      device_data(metal2_contexts), request.metal2_contexts,
      request.metal2_context_count, sizeof(std::uint32_t),
      "VIA1-stack M2-context H2D");
  copy_to_device(
      device_data(metal2_offsets), request.metal2_offsets,
      request.metal2_offset_count, sizeof(std::uint64_t),
      "VIA1-stack M2-offset H2D");
  copy_to_device(
      device_data(cells), request.cells, request.cell_count,
      sizeof(ViaCell), "VIA1-stack cell H2D");
  copy_to_device(
      device_data(templates), request.boxes, request.box_count,
      sizeof(ViaBox), "VIA1-stack box H2D");
  via_transform_semantics_gate<<<1, 8>>>(device_data(status));
  cuda_require(
      cudaGetLastError(), "VIA1-stack transform semantics gate launch");
  cuda_require(cudaDeviceSynchronize(), "VIA1-stack H2D synchronize");
  result.device_flags =
      copy_device_status(status, "VIA1-stack transform gate status D2H");
  result.h2d_ns = elapsed_ns(h2d_begin, Clock::now());
  if (result.device_flags) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return result;
  }

  const auto via_expand_begin = Clock::now();
  cuda_require(
      cudaMemset(
          device_data(expand_counters), 0, sizeof(ExpandCounters)),
      "VIA1-stack VIA expansion counter clear");
  if (request.via1_context_count) {
    via_expand_boxes_kernel<<<
        static_cast<unsigned int>(request.via1_context_count),
        kContextThreads>>>(
        device_data(contexts), device_data(via1_contexts),
        device_data(via1_offsets), device_data(cells),
        device_data(templates),
        static_cast<std::uint32_t>(request.via1_context_count),
        kVia1Layer, request.scene_left, request.scene_bottom,
        request.scene_right, request.scene_top, request.cut_width,
        request.cut_height, device_data(vias),
        device_data(expand_counters), device_data(status));
    cuda_require(
        cudaGetLastError(), "VIA1-stack VIA expansion launch");
  }
  cuda_require(
      cudaDeviceSynchronize(), "VIA1-stack VIA expansion synchronize");
  ExpandCounters via_expand_counters{};
  cuda_require(
      cudaMemcpy(
          &via_expand_counters, device_data(expand_counters),
          sizeof(via_expand_counters), cudaMemcpyDeviceToHost),
      "VIA1-stack VIA expansion counters D2H");
  result.device_flags =
      copy_device_status(status, "VIA1-stack VIA expansion status D2H");
  result.via_expand_ns = elapsed_ns(via_expand_begin, Clock::now());
  if (result.device_flags) {
    result.fallback_flags =
        (result.device_flags & kViaTransformOverflow)
            ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
            : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return result;
  }
  if (via_expand_counters.visited != request.flat_via1_box_count ||
      via_expand_counters.expanded != request.flat_via1_box_count ||
      via_expand_counters.size_checked != request.flat_via1_box_count ||
      via_expand_counters.size_violations >
          via_expand_counters.size_checked) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return result;
  }
  result.via_expanded = via_expand_counters.expanded;
  result.via_size_checked = via_expand_counters.size_checked;
  result.via_size_violations = via_expand_counters.size_violations;

  const auto via_grid_begin = Clock::now();
  if (!build_box_grid(
          device_data(vias),
          static_cast<std::uint32_t>(request.flat_via1_box_count),
          grid, result.grid_cells, request.max_via_memberships,
          via_counts, via_offsets, via_cursors, via_members,
          membership_total, status, &result.via_memberships,
          &result, "VIA1")) {
    result.via_grid_ns = elapsed_ns(via_grid_begin, Clock::now());
    return result;
  }
  result.via_grid_ns = elapsed_ns(via_grid_begin, Clock::now());

  const auto via_query_begin = Clock::now();
  cuda_require(
      cudaMemset(
          device_data(pair_counters), 0, sizeof(ViaPairCounters)),
      "VIA1-stack pair counter clear");
  if (request.flat_via1_box_count) {
    via_query_pairs_kernel<<<
        work_blocks(request.flat_via1_box_count), kThreads>>>(
        device_data(vias),
        static_cast<std::uint32_t>(request.flat_via1_box_count),
        grid, device_data(via_counts), device_data(via_offsets),
        device_data(via_members), request.spacing_distance,
        device_data(pair_counters), device_data(status));
    cuda_require(cudaGetLastError(), "VIA1-stack pair query launch");
  }
  cuda_require(cudaDeviceSynchronize(), "VIA1-stack pair query synchronize");
  ViaPairCounters host_pairs{};
  cuda_require(
      cudaMemcpy(
          &host_pairs, device_data(pair_counters), sizeof(host_pairs),
          cudaMemcpyDeviceToHost),
      "VIA1-stack pair counters D2H");
  result.device_flags =
      copy_device_status(status, "VIA1-stack pair status D2H");
  result.via_query_ns = elapsed_ns(via_query_begin, Clock::now());
  if (result.device_flags) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return result;
  }
  const __int128 classified_pairs =
      static_cast<__int128>(host_pairs.duplicate_pairs) +
      host_pairs.unsafe_pairs + host_pairs.spacing_pairs +
      host_pairs.clean_pairs;
  if (host_pairs.vias_queried != request.flat_via1_box_count ||
      classified_pairs != host_pairs.candidate_pairs) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return result;
  }
  result.via_candidates = host_pairs.candidate_pairs;
  result.via_pair_queried = host_pairs.vias_queried;
  result.duplicate_pairs = host_pairs.duplicate_pairs;
  result.unsafe_pairs = host_pairs.unsafe_pairs;
  result.spacing_pairs = host_pairs.spacing_pairs;
  result.clean_pairs = host_pairs.clean_pairs;
  if (result.via_candidates > request.max_pair_work) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY;
    return result;
  }

  const auto run_metal_pass =
      [&](ViaLayer layer,
          thrust::device_vector<std::uint32_t> &layer_contexts,
          thrust::device_vector<std::uint64_t> &layer_offsets,
          std::uint64_t layer_context_count, std::uint64_t flat_box_count,
          std::uint64_t *host_expanded, std::uint64_t *host_memberships,
          std::uint64_t *host_queried, std::uint64_t *host_candidates,
          std::uint64_t *host_certified, std::uint64_t *host_misses) -> bool {
        cuda_require(
            cudaMemset(
                device_data(expand_counters), 0,
                sizeof(ExpandCounters)),
            "VIA1-stack metal expansion counter clear");
        if (layer_context_count) {
          via_expand_boxes_kernel<<<
              static_cast<unsigned int>(layer_context_count),
              kContextThreads>>>(
              device_data(contexts), device_data(layer_contexts),
              device_data(layer_offsets), device_data(cells),
              device_data(templates),
              static_cast<std::uint32_t>(layer_context_count),
              layer, request.scene_left, request.scene_bottom,
              request.scene_right, request.scene_top, request.cut_width,
              request.cut_height, device_data(metal_scratch),
              device_data(expand_counters), device_data(status));
          cuda_require(
              cudaGetLastError(), "VIA1-stack metal expansion launch");
        }
        cuda_require(
            cudaDeviceSynchronize(),
            "VIA1-stack metal expansion synchronize");
        ExpandCounters host_expansion{};
        cuda_require(
            cudaMemcpy(
                &host_expansion, device_data(expand_counters),
                sizeof(host_expansion), cudaMemcpyDeviceToHost),
            "VIA1-stack metal expansion counters D2H");
        result.device_flags = copy_device_status(
            status, "VIA1-stack metal expansion status D2H");
        if (result.device_flags) {
          result.fallback_flags =
              (result.device_flags & kViaTransformOverflow)
                  ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
                  : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
          return false;
        }
        if (host_expansion.visited != flat_box_count ||
            host_expansion.expanded != flat_box_count ||
            host_expansion.size_checked != 0 ||
            host_expansion.size_violations != 0) {
          result.fallback_flags =
              KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
          return false;
        }
        *host_expanded = host_expansion.expanded;
        if (!build_box_grid(
                device_data(metal_scratch),
                static_cast<std::uint32_t>(flat_box_count), grid,
                result.grid_cells, request.max_metal_memberships,
                metal_counts, metal_offsets, metal_cursors,
                metal_members, membership_total, status,
                host_memberships, &result, "metal")) {
          return false;
        }
        cuda_require(
            cudaMemset(
                device_data(projection_counters), 0,
                sizeof(ProjectionCounters)),
            "VIA1-stack projection counter clear");
        if (request.flat_via1_box_count) {
          via_query_projection_kernel<<<
              work_blocks(request.flat_via1_box_count), kThreads>>>(
              device_data(vias),
              static_cast<std::uint32_t>(request.flat_via1_box_count),
              device_data(metal_scratch),
              static_cast<std::uint32_t>(flat_box_count), grid,
              device_data(metal_counts), device_data(metal_offsets),
              device_data(metal_members), request.enclosure_distance,
              device_data(projection_counters), device_data(status));
          cuda_require(
              cudaGetLastError(), "VIA1-stack projection query launch");
        }
        cuda_require(
            cudaDeviceSynchronize(),
            "VIA1-stack projection query synchronize");
        ProjectionCounters host_projection{};
        cuda_require(
            cudaMemcpy(
                &host_projection, device_data(projection_counters),
                sizeof(host_projection), cudaMemcpyDeviceToHost),
            "VIA1-stack projection counters D2H");
        result.device_flags = copy_device_status(
            status, "VIA1-stack projection status D2H");
        if (result.device_flags) {
          result.fallback_flags =
              KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
          return false;
        }
        if (host_projection.vias_queried !=
                request.flat_via1_box_count ||
            host_projection.certified + host_projection.misses !=
                host_projection.vias_queried) {
          result.fallback_flags =
              KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
          return false;
        }
        *host_queried = host_projection.vias_queried;
        *host_candidates = host_projection.candidate_boxes;
        *host_certified = host_projection.certified;
        *host_misses = host_projection.misses;
        const __int128 projection_work =
            static_cast<__int128>(host_projection.candidate_boxes) +
            host_projection.union_candidate_visits;
        if (projection_work > request.max_pair_work) {
          result.fallback_flags =
              KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY;
          return false;
        }
        return true;
      };

  const auto metal1_begin = Clock::now();
  if (!run_metal_pass(
          kMetal1Layer, metal1_contexts, metal1_offsets,
          request.metal1_context_count, request.flat_metal1_box_count,
          &result.metal1_expanded, &result.metal1_memberships,
          &result.metal1_queried, &result.metal1_candidates,
          &result.metal1_certified, &result.metal1_misses)) {
    result.metal1_ns = elapsed_ns(metal1_begin, Clock::now());
    return result;
  }
  result.metal1_ns = elapsed_ns(metal1_begin, Clock::now());

  const auto metal2_begin = Clock::now();
  if (!run_metal_pass(
          kMetal2Layer, metal2_contexts, metal2_offsets,
          request.metal2_context_count, request.flat_metal2_box_count,
          &result.metal2_expanded, &result.metal2_memberships,
          &result.metal2_queried, &result.metal2_candidates,
          &result.metal2_certified, &result.metal2_misses)) {
    result.metal2_ns = elapsed_ns(metal2_begin, Clock::now());
    return result;
  }
  result.metal2_ns = elapsed_ns(metal2_begin, Clock::now());

  const auto d2h_begin = Clock::now();
  result.device_flags =
      copy_device_status(status, "VIA1-stack final status D2H");
  result.d2h_ns = elapsed_ns(d2h_begin, Clock::now());
  if (result.device_flags) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    return result;
  }

  // Nonidentical touch/overlap invalidates the atomic proof. Exact duplicate
  // boxes are harmless because KLayout merges them to unchanged geometry.
  if (!result.unsafe_pairs) {
    if (!result.metal1_misses) {
      result.certified_empty_mask |=
          KLAYOUT_CUDA_SPATIAL_VIA1_METAL1_4 |
          KLAYOUT_CUDA_SPATIAL_VIA1_3;
    }
    if (!result.metal2_misses) {
      result.certified_empty_mask |=
          KLAYOUT_CUDA_SPATIAL_VIA1_METAL2_3 |
          KLAYOUT_CUDA_SPATIAL_VIA1_4;
    }
    if (!result.via_size_violations) {
      result.certified_empty_mask |= KLAYOUT_CUDA_SPATIAL_VIA1_1;
    }
    if (!result.spacing_pairs) {
      result.certified_empty_mask |= KLAYOUT_CUDA_SPATIAL_VIA1_2;
    }
  }
  return result;
}

int
run_via_request(const ViaRequest *request, ViaResult *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_UNCERTAIN;
  if (!request || !valid_via_request(*request)) {
    set_via_message(
        result, "unsupported or malformed atomic VIA1-stack request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  echo_via_request(*request, result);

  const auto total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> lock(via_pipeline_mutex());
    const ViaPipelineResult pipeline = run_via_pipeline(*request);
    result->fallback_flags = pipeline.fallback_flags;
    result->device_flags = pipeline.device_flags;
    result->certified_empty_mask = pipeline.certified_empty_mask;
    result->via_expanded_count = pipeline.via_expanded;
    result->via_size_checked_count = pipeline.via_size_checked;
    result->via_size_violation_count = pipeline.via_size_violations;
    result->metal1_expanded_count = pipeline.metal1_expanded;
    result->metal2_expanded_count = pipeline.metal2_expanded;
    result->grid_cell_count = pipeline.grid_cells;
    result->via_membership_count = pipeline.via_memberships;
    result->metal1_membership_count = pipeline.metal1_memberships;
    result->metal2_membership_count = pipeline.metal2_memberships;
    result->via_pair_queried_count = pipeline.via_pair_queried;
    result->via_candidate_pair_count = pipeline.via_candidates;
    result->duplicate_via_pair_count = pipeline.duplicate_pairs;
    result->unsafe_via_pair_count = pipeline.unsafe_pairs;
    result->spacing_violation_count = pipeline.spacing_pairs;
    result->clean_via_pair_count = pipeline.clean_pairs;
    result->metal1_queried_count = pipeline.metal1_queried;
    result->metal1_candidate_count = pipeline.metal1_candidates;
    result->metal1_certified_count = pipeline.metal1_certified;
    result->metal1_miss_count = pipeline.metal1_misses;
    result->metal2_queried_count = pipeline.metal2_queried;
    result->metal2_candidate_count = pipeline.metal2_candidates;
    result->metal2_certified_count = pipeline.metal2_certified;
    result->metal2_miss_count = pipeline.metal2_misses;
    result->setup_ns = pipeline.setup_ns;
    result->h2d_ns = pipeline.h2d_ns;
    result->via_expand_ns = pipeline.via_expand_ns;
    result->via_grid_ns = pipeline.via_grid_ns;
    result->via_query_ns = pipeline.via_query_ns;
    result->metal1_ns = pipeline.metal1_ns;
    result->metal2_ns = pipeline.metal2_ns;
    result->d2h_ns = pipeline.d2h_ns;
    if (pipeline.fallback_flags || pipeline.device_flags) {
      result->certified_empty_mask = 0;
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      set_via_message(
          result, "atomic VIA1-stack device or capacity gate declined");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    if ((pipeline.certified_empty_mask & request->requested_mask) ==
        request->requested_mask) {
      result->disposition = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_COMPLETE;
      set_via_message(
          result, "complete empty atomic M1/VIA1/M2 certificate");
    } else {
      result->disposition = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_NOT_EMPTY;
      set_via_message(
          result, "atomic VIA1-stack scan found a miss or violation");
    }
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return KLAYOUT_CUDA_SPATIAL_OK;
  } catch (const std::exception &error) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->certified_empty_mask = 0;
    set_via_message(result, error.what());
  } catch (...) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->certified_empty_mask = 0;
    set_via_message(result, "unknown CUDA atomic VIA1-stack exception");
  }
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return KLAYOUT_CUDA_SPATIAL_ERROR;
}

}  // namespace

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_via1_stack_empty_v1(
    const klayout_cuda_spatial_via1_stack_request_v1 *request,
    klayout_cuda_spatial_via1_stack_result_v1 *result)
{
  try {
    return run_via_request(request, result);
  } catch (...) {
    if (result) {
      std::memset(result, 0, sizeof(*result));
      result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result->struct_size = sizeof(*result);
      result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      result->disposition = KLAYOUT_CUDA_SPATIAL_VIA1_STACK_UNCERTAIN;
      set_via_message(
          result, "exception escaped the atomic VIA1-stack boundary");
    }
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}
