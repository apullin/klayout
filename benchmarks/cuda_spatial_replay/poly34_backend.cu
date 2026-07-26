/*
 * Atomic CUDA terminal-empty certificate for FreePDK45 POLY.3/POLY.4.
 *
 * The host supplies compact exact merged POLY, ACTIVE and derived GATE box
 * templates plus hierarchy contexts.  This backend expands all occurrences,
 * constructs complete bounded primary windows on device and returns only one
 * atomic clean/fallback decision.  No candidate or marker geometry crosses
 * the ABI.
 */

#include "dbCudaPoly34Digest.h"
#include "dbCudaSpatialApi.h"
#include "poly34_terminal_empty_certificate.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace certificate = klayout_cuda::poly34;
using PolyContext = klayout_cuda_spatial_poly34_context_v1;
using PolyBox = klayout_cuda_spatial_poly34_box_v1;
using PolyCell = klayout_cuda_spatial_poly34_cell_v1;
using PolySpan = klayout_cuda_spatial_poly34_domain_span_v1;
using PolyRequest = klayout_cuda_spatial_poly34_request_v1;
using PolyResult = klayout_cuda_spatial_poly34_result_v1;
using Clock = std::chrono::steady_clock;

static_assert(std::is_trivially_copyable<PolyContext>::value,
              "POLY34 context must remain POD");
static_assert(std::is_trivially_copyable<PolyBox>::value,
              "POLY34 box must remain POD");
static_assert(std::is_trivially_copyable<PolyCell>::value,
              "POLY34 cell must remain POD");
static_assert(sizeof(PolyContext) == 24,
              "unexpected POLY34 context ABI padding");
static_assert(sizeof(PolyBox) == 32,
              "unexpected POLY34 box ABI padding");
static_assert(sizeof(PolySpan) == 16,
              "unexpected POLY34 span ABI padding");
static_assert(sizeof(PolyCell) == 56,
              "unexpected POLY34 cell ABI padding");

constexpr std::int64_t kCoordinateLimit = INT64_C(1000000000000);
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kContextThreads = 128;

enum PolyDeviceFlag : std::uint32_t {
  kPolyTransformOverflow = 1u << 0,
  kPolyGridCounterOverflow = 1u << 1,
  kPolyMembershipCapacity = 1u << 2,
  kPolyCandidateCapacity = 1u << 3,
  kPolyQueryCapacity = 1u << 4,
  kPolyCandidateWorkCapacity = 1u << 5,
  kPolyInvalidDeviceRecord = 1u << 6,
  kPolyIntersectionCapacity = 1u << 7,
};

struct PolyGrid {
  std::int64_t base_x;
  std::int64_t base_y;
  std::int64_t cell_size;
  std::uint32_t width;
  std::uint32_t height;
};

struct ProfileCounters {
  unsigned long long query_visits;
  unsigned long long candidates;
  unsigned long long terminal_empty;
  unsigned long long maximum_candidates;
};

struct ProfileResult {
  std::uint64_t memberships = 0;
  std::uint64_t query_visits = 0;
  std::uint64_t candidates = 0;
  std::uint64_t terminal_empty = 0;
  std::uint64_t maximum_candidates = 0;
  std::uint64_t grid_ns = 0;
  std::uint64_t query_ns = 0;
};

struct PipelineResult {
  std::uint32_t fallback_flags = 0;
  std::uint32_t device_flags = 0;
  std::uint32_t certified_empty_mask = 0;
  std::uint32_t disposition = KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN;
  std::uint64_t grid_cells = 0;
  std::uint64_t expanded_poly = 0;
  std::uint64_t expanded_active = 0;
  std::uint64_t expanded_gate = 0;
  ProfileResult poly;
  ProfileResult active;
  std::uint64_t atomic_terminal_empty = 0;
  std::uint64_t fallback_gates = 0;
  std::uint64_t setup_ns = 0;
  std::uint64_t h2d_ns = 0;
  std::uint64_t expand_ns = 0;
  std::uint64_t d2h_ns = 0;
};

struct RawGateResult {
  thrust::device_vector<PolyBox> boxes;
  std::uint64_t query_visits = 0;
  std::uint64_t memberships = 0;
};

std::mutex &poly_pipeline_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end)
{
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          end - begin).count());
}

void cuda_require(cudaError_t error, const char *operation)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(error));
  }
}

void set_message(PolyResult *result, const char *message)
{
  std::snprintf(
      result->message, sizeof(result->message), "%s",
      message ? message : "");
}

bool coordinate_qualified(std::int64_t value)
{
  return value >= -kCoordinateLimit && value <= kCoordinateLimit;
}

bool checked_add_u64(std::uint64_t first, std::uint64_t second,
                     std::uint64_t *result)
{
  if (second > std::numeric_limits<std::uint64_t>::max() - first) {
    return false;
  }
  *result = first + second;
  return true;
}

bool raw_request(const PolyRequest &request)
{
  return request.opcode ==
             KLAYOUT_CUDA_SPATIAL_POLY34_RAW_TERMINAL_EMPTY &&
         request.option_flags ==
             KLAYOUT_CUDA_SPATIAL_POLY34_RAW_QUALIFIED_OPTIONS &&
         request.format_version == 2;
}

bool merged_request(const PolyRequest &request)
{
  return request.opcode ==
             KLAYOUT_CUDA_SPATIAL_POLY34_TERMINAL_EMPTY &&
         request.option_flags ==
             KLAYOUT_CUDA_SPATIAL_POLY34_QUALIFIED_OPTIONS &&
         request.format_version == 1;
}

std::int64_t floor_div_host(std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

bool transform_point_host(const PolyContext &context,
                          std::int64_t x, std::int64_t y,
                          std::int64_t *out_x, std::int64_t *out_y)
{
  if (context.transform_code >= 8) return false;
  static const int matrix[8][4] = {
      {1, 0, 0, 1}, {0, -1, 1, 0}, {-1, 0, 0, -1},
      {0, 1, -1, 0}, {1, 0, 0, -1}, {0, 1, 1, 0},
      {-1, 0, 0, 1}, {0, -1, -1, 0}};
  const int *m = matrix[context.transform_code];
  const __int128 tx =
      static_cast<__int128>(m[0]) * x +
      static_cast<__int128>(m[1]) * y + context.tx;
  const __int128 ty =
      static_cast<__int128>(m[2]) * x +
      static_cast<__int128>(m[3]) * y + context.ty;
  if (tx < -static_cast<__int128>(kCoordinateLimit) ||
      tx > static_cast<__int128>(kCoordinateLimit) ||
      ty < -static_cast<__int128>(kCoordinateLimit) ||
      ty > static_cast<__int128>(kCoordinateLimit)) {
    return false;
  }
  *out_x = static_cast<std::int64_t>(tx);
  *out_y = static_cast<std::int64_t>(ty);
  return true;
}

bool valid_context_domain(
    const PolyRequest &request, std::uint32_t domain,
    const std::uint32_t *contexts, std::uint64_t context_count,
    const std::uint64_t *offsets, std::uint64_t offset_count,
    std::uint64_t expected_flat_count)
{
  if (domain >= KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT ||
      !contexts || !offsets || !context_count ||
      offset_count != context_count) {
    return false;
  }
  std::uint64_t list_id = 0;
  std::uint64_t flat_count = 0;
  for (std::uint64_t context_id = 0;
       context_id < request.context_count; ++context_id) {
    const PolyContext &context =
        static_cast<const PolyContext *>(request.contexts)[context_id];
    if (context.cell_id >= request.cell_count) return false;
    const PolyCell &cell =
        static_cast<const PolyCell *>(request.cells)[context.cell_id];
    const std::uint32_t count = cell.domains[domain].box_count;
    if (!count) continue;
    if (list_id >= context_count ||
        contexts[list_id] != context_id ||
        offsets[list_id] != flat_count ||
        !checked_add_u64(flat_count, count, &flat_count)) {
      return false;
    }
    ++list_id;
  }
  return list_id == context_count && flat_count == expected_flat_count;
}

bool request_structurally_valid(const PolyRequest &request)
{
  const bool raw = raw_request(request);
  const bool merged = merged_request(request);
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof(request) ||
      (!raw && !merged) || request.dbu_per_micron != 2000 ||
      request.requested_mask != KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES ||
      request.device < 0 || request.reserved0 ||
      request.identity_reserved || request.context_reserved ||
      request.cell_reserved || request.box_reserved ||
      request.capacity_reserved || request.reserved1[0] ||
      request.reserved1[1] ||
      request.poly3_distance != certificate::kPoly3Distance ||
      request.poly4_distance != certificate::kPoly4Distance ||
      request.grid_cell_size <= 0 ||
      request.max_candidates_per_gate !=
          certificate::kMaximumCandidateBoxes ||
      !request.store_identity || !request.layout_identity ||
      request.poly_layer_id == request.active_layer_id ||
      (merged &&
       (request.poly_layer_id == request.gate_layer_id ||
        request.active_layer_id == request.gate_layer_id)) ||
      (raw &&
       request.gate_layer_id != KLAYOUT_CUDA_SPATIAL_POLY34_NO_GATE_LAYER) ||
      !request.context_count || !request.contexts ||
      request.context_record_bytes != sizeof(PolyContext) ||
      !request.cell_count || !request.cells ||
      request.cell_record_bytes != sizeof(PolyCell) ||
      !request.box_count || !request.boxes ||
      request.box_record_bytes != sizeof(PolyBox) ||
      !request.flat_poly_box_count || !request.flat_active_box_count ||
      (merged && !request.flat_gate_box_count) ||
      (raw &&
       (request.gate_contexts || request.gate_context_count ||
        request.gate_offsets || request.gate_offset_count ||
        request.flat_gate_box_count)) ||
      (merged &&
       (!request.gate_contexts || !request.gate_context_count ||
        !request.gate_offsets ||
        request.gate_offset_count != request.gate_context_count)) ||
      request.root_cell >= request.cell_count ||
      request.context_count > request.max_contexts ||
      request.context_count > UINT32_MAX ||
      request.cell_count > request.context_count ||
      request.cell_count > UINT32_MAX ||
      request.box_count > request.max_flat_boxes ||
      request.box_count > UINT32_MAX ||
      request.flat_poly_box_count > UINT32_MAX ||
      request.flat_active_box_count > UINT32_MAX ||
      request.flat_gate_box_count > UINT32_MAX ||
      request.scene_left >= request.scene_right ||
      request.scene_bottom >= request.scene_top ||
      !coordinate_qualified(request.scene_left) ||
      !coordinate_qualified(request.scene_bottom) ||
      !coordinate_qualified(request.scene_right) ||
      !coordinate_qualified(request.scene_top) ||
      !request.max_contexts || !request.max_flat_boxes ||
      !request.max_grid_cells || !request.max_poly_memberships ||
      !request.max_active_memberships || !request.max_query_visits ||
      !request.max_candidate_work) {
    return false;
  }

  std::array<std::uint8_t, 32> digest{};
  if (!db::cuda_poly34_digest::request_digest(request, digest) ||
      !std::equal(
          digest.begin(), digest.end(), request.scene_digest)) {
    return false;
  }

  const PolyContext *contexts =
      static_cast<const PolyContext *>(request.contexts);
  const PolyCell *cells = static_cast<const PolyCell *>(request.cells);
  const PolyBox *boxes = static_cast<const PolyBox *>(request.boxes);
  if (contexts[0].tx || contexts[0].ty ||
      contexts[0].cell_id != request.root_cell ||
      contexts[0].transform_code) {
    return false;
  }

  std::set<std::uint64_t> source_cells;
  std::vector<std::array<std::int64_t, 4>> cell_bounds(
      static_cast<std::size_t>(request.cell_count),
      {INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN});
  std::vector<std::uint8_t> cell_has_geometry(
      static_cast<std::size_t>(request.cell_count), 0);
  std::uint64_t next_box = 0;
  for (std::uint64_t cell_id = 0;
       cell_id < request.cell_count; ++cell_id) {
    const PolyCell &cell = cells[cell_id];
    if (!source_cells.insert(cell.source_cell_index).second) return false;
    for (std::uint32_t domain = 0;
         domain < KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT; ++domain) {
      const PolySpan &span = cell.domains[domain];
      if (span.reserved0 || span.box_begin != next_box ||
          span.box_count > request.box_count - next_box ||
          (raw &&
           domain == KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN &&
           span.box_count)) {
        return false;
      }
      for (std::uint32_t local = 0; local < span.box_count; ++local) {
        const PolyBox &box = boxes[span.box_begin + local];
        if (box.left >= box.right || box.bottom >= box.top ||
            !coordinate_qualified(box.left) ||
            !coordinate_qualified(box.bottom) ||
            !coordinate_qualified(box.right) ||
            !coordinate_qualified(box.top)) {
          return false;
        }
        std::array<std::int64_t, 4> &bounds = cell_bounds[cell_id];
        bounds[0] = std::min(bounds[0], box.left);
        bounds[1] = std::min(bounds[1], box.bottom);
        bounds[2] = std::max(bounds[2], box.right);
        bounds[3] = std::max(bounds[3], box.top);
        cell_has_geometry[cell_id] = 1;
      }
      next_box += span.box_count;
    }
  }
  if (next_box != request.box_count) return false;

  for (std::uint64_t context_id = 0;
       context_id < request.context_count; ++context_id) {
    const PolyContext &context = contexts[context_id];
    if (context.cell_id >= request.cell_count ||
        context.transform_code >= 8 ||
        !coordinate_qualified(context.tx) ||
        !coordinate_qualified(context.ty)) {
      return false;
    }
  }
  if (!valid_context_domain(
          request, KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN,
          request.poly_contexts, request.poly_context_count,
          request.poly_offsets, request.poly_offset_count,
          request.flat_poly_box_count) ||
      !valid_context_domain(
          request, KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN,
          request.active_contexts, request.active_context_count,
          request.active_offsets, request.active_offset_count,
          request.flat_active_box_count) ||
      (merged && !valid_context_domain(
          request, KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN,
          request.gate_contexts, request.gate_context_count,
          request.gate_offsets, request.gate_offset_count,
          request.flat_gate_box_count))) {
    return false;
  }

  bool have_scene = false;
  std::array<std::int64_t, 4> scene = {
      INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN};
  for (std::uint64_t context_id = 0;
       context_id < request.context_count; ++context_id) {
    const PolyContext &context = contexts[context_id];
    if (!cell_has_geometry[context.cell_id]) continue;
    const auto &local = cell_bounds[context.cell_id];
    const std::int64_t xs[4] =
        {local[0], local[0], local[2], local[2]};
    const std::int64_t ys[4] =
        {local[1], local[3], local[1], local[3]};
    for (int corner = 0; corner < 4; ++corner) {
      std::int64_t x = 0;
      std::int64_t y = 0;
      if (!transform_point_host(
              context, xs[corner], ys[corner], &x, &y)) {
        return false;
      }
      scene[0] = std::min(scene[0], x);
      scene[1] = std::min(scene[1], y);
      scene[2] = std::max(scene[2], x);
      scene[3] = std::max(scene[3], y);
      have_scene = true;
    }
  }
  return have_scene &&
         scene[0] == request.scene_left &&
         scene[1] == request.scene_bottom &&
         scene[2] == request.scene_right &&
         scene[3] == request.scene_top;
}

__device__ bool add_i64_checked(std::int64_t first, std::int64_t second,
                                std::int64_t *result)
{
  if ((second > 0 && first > INT64_MAX - second) ||
      (second < 0 && first < INT64_MIN - second)) {
    return false;
  }
  *result = first + second;
  return true;
}

__device__ bool transform_point_device(
    const PolyContext &context, std::int64_t x, std::int64_t y,
    std::int64_t *out_x, std::int64_t *out_y)
{
  if (context.transform_code >= 8) return false;
  int xx = 0, xy = 0, yx = 0, yy = 0;
  switch (context.transform_code) {
  case 0: xx = 1; yy = 1; break;
  case 1: xy = -1; yx = 1; break;
  case 2: xx = -1; yy = -1; break;
  case 3: xy = 1; yx = -1; break;
  case 4: xx = 1; yy = -1; break;
  case 5: xy = 1; yx = 1; break;
  case 6: xx = -1; yy = 1; break;
  case 7: xy = -1; yx = -1; break;
  }
  std::int64_t transformed_x = xx * x + xy * y;
  std::int64_t transformed_y = yx * x + yy * y;
  return add_i64_checked(
             transformed_x, context.tx, out_x) &&
         add_i64_checked(
             transformed_y, context.ty, out_y) &&
         *out_x >= -kCoordinateLimit && *out_x <= kCoordinateLimit &&
         *out_y >= -kCoordinateLimit && *out_y <= kCoordinateLimit;
}

__device__ bool transform_box_device(
    const PolyContext &context, const PolyBox &source, PolyBox *destination)
{
  const std::int64_t xs[4] =
      {source.left, source.left, source.right, source.right};
  const std::int64_t ys[4] =
      {source.bottom, source.top, source.bottom, source.top};
  PolyBox transformed = {INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN};
  for (int corner = 0; corner < 4; ++corner) {
    std::int64_t x = 0;
    std::int64_t y = 0;
    if (!transform_point_device(
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

__global__ void expand_boxes_kernel(
    const PolyContext *contexts,
    const std::uint32_t *domain_contexts,
    const std::uint64_t *domain_offsets,
    std::uint64_t domain_context_count,
    const PolyCell *cells, const PolyBox *templates,
    std::uint32_t domain, PolyBox *expanded,
    std::uint32_t *status)
{
  const std::uint64_t list_id = blockIdx.x;
  if (list_id >= domain_context_count ||
      domain >= KLAYOUT_CUDA_SPATIAL_POLY34_DOMAIN_COUNT) {
    return;
  }
  const PolyContext context = contexts[domain_contexts[list_id]];
  const PolySpan span = cells[context.cell_id].domains[domain];
  for (std::uint32_t local = threadIdx.x;
       local < span.box_count; local += blockDim.x) {
    PolyBox box{};
    if (!transform_box_device(
            context, templates[span.box_begin + local], &box)) {
      atomicOr(status, static_cast<std::uint32_t>(kPolyTransformOverflow));
      continue;
    }
    expanded[domain_offsets[list_id] + local] = box;
  }
}

__device__ std::int64_t floor_div_device(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

__device__ bool grid_span(
    const PolyBox &box, const PolyGrid &grid, std::int64_t enlargement,
    std::int64_t *x0, std::int64_t *y0,
    std::int64_t *x1, std::int64_t *y1)
{
  std::int64_t left = 0, bottom = 0, right = 0, top = 0;
  if (!add_i64_checked(box.left, -enlargement, &left) ||
      !add_i64_checked(box.bottom, -enlargement, &bottom) ||
      !add_i64_checked(box.right, enlargement, &right) ||
      !add_i64_checked(box.top, enlargement, &top)) {
    return false;
  }
  *x0 = floor_div_device(left, grid.cell_size);
  *y0 = floor_div_device(bottom, grid.cell_size);
  *x1 = floor_div_device(right, grid.cell_size);
  *y1 = floor_div_device(top, grid.cell_size);
  const std::int64_t max_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t max_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  return *x0 >= grid.base_x && *y0 >= grid.base_y &&
         *x1 <= max_x && *y1 <= max_y && *x0 <= *x1 && *y0 <= *y1;
}

__device__ std::uint64_t grid_index(
    const PolyGrid &grid, std::int64_t x, std::int64_t y)
{
  return static_cast<std::uint64_t>(y - grid.base_y) * grid.width +
         static_cast<std::uint64_t>(x - grid.base_x);
}

__global__ void count_grid_kernel(
    const PolyBox *boxes, std::uint64_t box_count, PolyGrid grid,
    std::uint32_t *counts, unsigned long long *memberships,
    std::uint32_t *status)
{
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       id < box_count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!grid_span(boxes[id], grid, 0, &x0, &y0, &x1, &y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
      continue;
    }
    unsigned long long local = 0;
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        if (atomicAdd(&counts[cell], 1u) == UINT32_MAX) {
          atomicOr(
              status,
              static_cast<std::uint32_t>(kPolyGridCounterOverflow));
        }
        ++local;
      }
    }
    atomicAdd(memberships, local);
  }
}

__global__ void fill_grid_kernel(
    const PolyBox *boxes, std::uint64_t box_count, PolyGrid grid,
    std::uint32_t *cursors, std::uint32_t *members,
    std::uint32_t *status)
{
  for (std::uint64_t id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       id < box_count;
       id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!grid_span(boxes[id], grid, 0, &x0, &y0, &x1, &y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t slot = atomicAdd(&cursors[cell], 1u);
        members[slot] = static_cast<std::uint32_t>(id);
      }
    }
  }
}

__device__ bool positive_box_intersection(
    const PolyBox &first, const PolyBox &second, PolyBox *intersection)
{
  PolyBox candidate = {
      max(first.left, second.left),
      max(first.bottom, second.bottom),
      min(first.right, second.right),
      min(first.top, second.top)};
  if (candidate.left >= candidate.right ||
      candidate.bottom >= candidate.top) {
    return false;
  }
  *intersection = candidate;
  return true;
}

__global__ void count_raw_gate_intersections_kernel(
    const PolyBox *poly, std::uint64_t poly_count,
    const PolyBox *active, PolyGrid grid,
    const std::uint32_t *active_counts,
    const std::uint32_t *active_offsets,
    const std::uint32_t *active_members,
    std::uint32_t *intersection_counts,
    unsigned long long *query_visits, std::uint32_t *status)
{
  for (std::uint64_t poly_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       poly_id < poly_count;
       poly_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const PolyBox source = poly[poly_id];
    std::int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!grid_span(source, grid, 0, &x0, &y0, &x1, &y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
      continue;
    }

    std::uint32_t local_count = 0;
    unsigned long long local_visits = 0;
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t begin = active_offsets[cell];
        const std::uint32_t end = begin + active_counts[cell];
        for (std::uint32_t slot = begin; slot < end; ++slot) {
          ++local_visits;
          PolyBox intersection{};
          if (!positive_box_intersection(
                  source, active[active_members[slot]], &intersection)) {
            continue;
          }
          // Both input boxes can span multiple grid cells.  Assign their
          // positive intersection to exactly the cell containing its
          // lower-left point so every pair is counted once without a
          // per-thread duplicate set.
          if (floor_div_device(intersection.left, grid.cell_size) != x ||
              floor_div_device(intersection.bottom, grid.cell_size) != y) {
            continue;
          }
          if (local_count == UINT32_MAX) {
            atomicOr(
                status,
                static_cast<std::uint32_t>(kPolyIntersectionCapacity));
          } else {
            ++local_count;
          }
        }
      }
    }
    intersection_counts[poly_id] = local_count;
    atomicAdd(query_visits, local_visits);
  }
}

__global__ void fill_raw_gate_intersections_kernel(
    const PolyBox *poly, std::uint64_t poly_count,
    const PolyBox *active, PolyGrid grid,
    const std::uint32_t *active_counts,
    const std::uint32_t *active_offsets,
    const std::uint32_t *active_members,
    const std::uint32_t *expected_counts,
    const std::uint64_t *intersection_offsets,
    std::uint64_t intersection_capacity, PolyBox *intersections,
    std::uint32_t *status)
{
  for (std::uint64_t poly_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       poly_id < poly_count;
       poly_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const PolyBox source = poly[poly_id];
    std::int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!grid_span(source, grid, 0, &x0, &y0, &x1, &y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
      continue;
    }

    const std::uint64_t output_begin = intersection_offsets[poly_id];
    std::uint32_t emitted = 0;
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t begin = active_offsets[cell];
        const std::uint32_t end = begin + active_counts[cell];
        for (std::uint32_t slot = begin; slot < end; ++slot) {
          PolyBox intersection{};
          if (!positive_box_intersection(
                  source, active[active_members[slot]], &intersection) ||
              floor_div_device(intersection.left, grid.cell_size) != x ||
              floor_div_device(intersection.bottom, grid.cell_size) != y) {
            continue;
          }
          const std::uint64_t output = output_begin + emitted;
          if (emitted >= expected_counts[poly_id] ||
              output >= intersection_capacity) {
            atomicOr(
                status,
                static_cast<std::uint32_t>(kPolyIntersectionCapacity));
          } else {
            intersections[output] = intersection;
            ++emitted;
          }
        }
      }
    }
    if (emitted != expected_counts[poly_id]) {
      atomicOr(
          status, static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
    }
  }
}

__device__ bool raw_gate_side_probe(
    const PolyBox &gate, std::uint32_t side, PolyBox *probe)
{
  *probe = gate;
  switch (side) {
    case 0:
      probe->right = gate.left;
      return add_i64_checked(gate.left, -1, &probe->left);
    case 1:
      probe->left = gate.right;
      return add_i64_checked(gate.right, 1, &probe->right);
    case 2:
      probe->top = gate.bottom;
      return add_i64_checked(gate.bottom, -1, &probe->bottom);
    case 3:
      probe->bottom = gate.top;
      return add_i64_checked(gate.top, 1, &probe->top);
    default:
      return false;
  }
}

__device__ bool box_covers_box(
    const PolyBox &cover, const PolyBox &target)
{
  return cover.left <= target.left && cover.bottom <= target.bottom &&
         cover.right >= target.right && cover.top >= target.top;
}

__global__ void mark_raw_gate_internal_sides_kernel(
    const PolyBox *gates, std::uint64_t gate_count, PolyGrid grid,
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *members, std::uint8_t *internal_sides,
    unsigned long long *query_visits, std::uint32_t *status)
{
  for (std::uint64_t gate_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       gate_id < gate_count;
       gate_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const PolyBox gate = gates[gate_id];
    std::uint8_t mask = 0;
    unsigned long long visits = 0;
    for (std::uint32_t side = 0; side < 4; ++side) {
      PolyBox probe{};
      if (!raw_gate_side_probe(gate, side, &probe)) {
        atomicOr(
            status,
            static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
        continue;
      }
      const std::int64_t x =
          floor_div_device(probe.left, grid.cell_size);
      const std::int64_t y =
          floor_div_device(probe.bottom, grid.cell_size);
      const std::int64_t max_x =
          grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
      const std::int64_t max_y =
          grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
      if (x < grid.base_x || x > max_x ||
          y < grid.base_y || y > max_y) {
        atomicOr(
            status,
            static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
        continue;
      }
      const std::uint64_t cell = grid_index(grid, x, y);
      const std::uint32_t begin = offsets[cell];
      const std::uint32_t end = begin + counts[cell];
      for (std::uint32_t slot = begin; slot < end; ++slot) {
        ++visits;
        const std::uint32_t candidate = members[slot];
        if (candidate != gate_id &&
            box_covers_box(gates[candidate], probe)) {
          mask |= std::uint8_t(1) << side;
          break;
        }
      }
    }
    internal_sides[gate_id] = mask;
    atomicAdd(query_visits, visits);
  }
}

__device__ bool window_candidate(
    const PolyBox &gate, const PolyBox &primary, std::int64_t distance)
{
  return primary.right > gate.left - distance &&
         primary.left < gate.right + distance &&
         primary.top > gate.bottom - distance &&
         primary.bottom < gate.top + distance;
}

__global__ void query_profile_kernel(
    const PolyBox *gates, std::uint64_t gate_count,
    const PolyBox *primary, PolyGrid grid,
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *members, std::int64_t distance,
    std::uint64_t max_query_visits, std::uint64_t max_candidate_work,
    std::uint32_t max_candidates,
    const std::uint8_t *proven_internal_sides,
    std::uint8_t *outcomes,
    ProfileCounters *counters, std::uint32_t *status)
{
  for (std::uint64_t gate_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       gate_id < gate_count;
       gate_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const PolyBox gate = gates[gate_id];
    std::int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (!grid_span(gate, grid, distance, &x0, &y0, &x1, &y1)) {
      outcomes[gate_id] =
          static_cast<std::uint8_t>(certificate::Certificate::kUnsupported);
      atomicOr(status, static_cast<std::uint32_t>(kPolyInvalidDeviceRecord));
      continue;
    }

    std::uint32_t candidate_ids[certificate::kMaximumCandidateBoxes];
    certificate::Box candidate_boxes[
        certificate::kMaximumCandidateBoxes];
    std::uint32_t candidate_count = 0;
    unsigned long long visits = 0;
    bool over_capacity = false;
    for (std::int64_t y = y0; y <= y1 && !over_capacity; ++y) {
      for (std::int64_t x = x0; x <= x1 && !over_capacity; ++x) {
        const std::uint64_t cell = grid_index(grid, x, y);
        const std::uint32_t begin = offsets[cell];
        const std::uint32_t end = begin + counts[cell];
        for (std::uint32_t slot = begin; slot < end; ++slot) {
          ++visits;
          const std::uint32_t primary_id = members[slot];
          const PolyBox box = primary[primary_id];
          if (!window_candidate(gate, box, distance)) continue;
          bool duplicate = false;
          for (std::uint32_t existing = 0;
               existing < candidate_count; ++existing) {
            duplicate |= candidate_ids[existing] == primary_id;
          }
          if (duplicate) continue;
          if (candidate_count >= max_candidates ||
              candidate_count >= certificate::kMaximumCandidateBoxes) {
            over_capacity = true;
            break;
          }
          candidate_ids[candidate_count] = primary_id;
          candidate_boxes[candidate_count] = {
              box.left, box.bottom, box.right, box.top};
          ++candidate_count;
        }
      }
    }

    atomicAdd(&counters->query_visits, visits);
    atomicAdd(
        &counters->candidates,
        static_cast<unsigned long long>(candidate_count));
    atomicMax(
        &counters->maximum_candidates,
        static_cast<unsigned long long>(candidate_count));
    if (over_capacity) {
      outcomes[gate_id] =
          static_cast<std::uint8_t>(certificate::Certificate::kFallback);
      atomicOr(status, static_cast<std::uint32_t>(kPolyCandidateCapacity));
      continue;
    }
    if (atomicAdd(
            &counters->query_visits, 0ULL) > max_query_visits) {
      atomicOr(status, static_cast<std::uint32_t>(kPolyQueryCapacity));
    }
    if (atomicAdd(
            &counters->candidates, 0ULL) > max_candidate_work) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(kPolyCandidateWorkCapacity));
    }

    const certificate::Certificate outcome =
        certificate::terminal_empty_profile(
            certificate::Box{
                gate.left, gate.bottom, gate.right, gate.top},
            candidate_boxes, candidate_count, distance, true,
            proven_internal_sides ?
                proven_internal_sides[gate_id] : 0);
    outcomes[gate_id] = static_cast<std::uint8_t>(outcome);
    if (outcome == certificate::Certificate::kTerminalEmpty) {
      atomicAdd(&counters->terminal_empty, 1ULL);
    }
  }
}

__global__ void combine_outcomes_kernel(
    const std::uint8_t *poly, const std::uint8_t *active,
    std::uint64_t gate_count, unsigned long long *atomic_empty,
    unsigned long long *fallback)
{
  unsigned long long local_empty = 0;
  unsigned long long local_fallback = 0;
  const std::uint8_t terminal = static_cast<std::uint8_t>(
      certificate::Certificate::kTerminalEmpty);
  for (std::uint64_t gate_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       gate_id < gate_count;
       gate_id += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (poly[gate_id] == terminal && active[gate_id] == terminal) {
      ++local_empty;
    } else {
      ++local_fallback;
    }
  }
  if (local_empty) atomicAdd(atomic_empty, local_empty);
  if (local_fallback) atomicAdd(fallback, local_fallback);
}

std::uint32_t fallback_from_device_flags(std::uint32_t flags)
{
  if (flags & kPolyTransformOverflow) {
    return KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW;
  }
  if (flags &
      (kPolyGridCounterOverflow | kPolyMembershipCapacity)) {
    return KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY;
  }
  if (flags & (kPolyCandidateCapacity | kPolyIntersectionCapacity)) {
    return KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_CAPACITY;
  }
  if (flags &
      (kPolyQueryCapacity | kPolyCandidateWorkCapacity)) {
    return KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY;
  }
  return flags ? KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT : 0;
}

unsigned int blocks_for(std::uint64_t records,
                        const cudaDeviceProp &properties)
{
  const std::uint64_t wanted = (records + kThreads - 1) / kThreads;
  return static_cast<unsigned int>(
      std::max<std::uint64_t>(
          1, std::min<std::uint64_t>(
                 wanted,
                 static_cast<std::uint64_t>(properties.maxGridSize[0]))));
}

RawGateResult derive_raw_gates(
    const PolyBox *poly, std::uint64_t poly_count,
    const PolyBox *active, std::uint64_t active_count,
    const PolyGrid &grid, std::uint64_t grid_cells,
    const PolyRequest &request, std::uint32_t *device_status,
    const cudaDeviceProp &properties)
{
  RawGateResult result;

  // Index the complete ACTIVE rectangle cover once for the exact global
  // POLY x ACTIVE positive-area join.
  thrust::device_vector<std::uint32_t> counts(grid_cells, 0);
  thrust::device_vector<std::uint32_t> offsets(grid_cells + 1);
  thrust::device_vector<std::uint32_t> cursors(grid_cells);
  thrust::device_vector<unsigned long long> membership_total(1, 0);
  count_grid_kernel<<<blocks_for(active_count, properties), kThreads>>>(
      active, active_count, grid,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(membership_total.data()),
      device_status);
  cuda_require(
      cudaGetLastError(), "POLY34 raw ACTIVE grid-count launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "POLY34 raw ACTIVE grid-count synchronize");

  unsigned long long memberships = 0;
  std::uint32_t status = 0;
  cuda_require(
      cudaMemcpy(
          &memberships,
          thrust::raw_pointer_cast(membership_total.data()),
          sizeof(memberships), cudaMemcpyDeviceToHost),
      "POLY34 raw ACTIVE membership count D2H");
  cuda_require(
      cudaMemcpy(
          &status, device_status, sizeof(status), cudaMemcpyDeviceToHost),
      "POLY34 raw ACTIVE grid status D2H");
  result.memberships = memberships;
  if (status) return result;
  if (memberships > request.max_active_memberships ||
      memberships > UINT32_MAX) {
    status = kPolyMembershipCapacity;
    cuda_require(
        cudaMemcpy(
            device_status, &status, sizeof(status), cudaMemcpyHostToDevice),
        "POLY34 raw ACTIVE membership-capacity H2D");
    return result;
  }

  thrust::exclusive_scan(
      thrust::device, counts.begin(), counts.end(), offsets.begin());
  const std::uint32_t terminal = static_cast<std::uint32_t>(memberships);
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(offsets.data()) + grid_cells,
          &terminal, sizeof(terminal), cudaMemcpyHostToDevice),
      "POLY34 raw ACTIVE terminal offset H2D");
  thrust::copy(
      thrust::device, offsets.begin(), offsets.begin() + grid_cells,
      cursors.begin());
  thrust::device_vector<std::uint32_t> members(
      static_cast<std::size_t>(memberships));
  fill_grid_kernel<<<blocks_for(active_count, properties), kThreads>>>(
      active, active_count, grid,
      thrust::raw_pointer_cast(cursors.data()),
      thrust::raw_pointer_cast(members.data()), device_status);
  cuda_require(
      cudaGetLastError(), "POLY34 raw ACTIVE grid-fill launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "POLY34 raw ACTIVE grid-fill synchronize");
  cuda_require(
      cudaMemcpy(
          &status, device_status, sizeof(status), cudaMemcpyDeviceToHost),
      "POLY34 raw ACTIVE fill status D2H");
  if (status) return result;

  // Count every positive-area pair intersection.  A deterministic owner grid
  // cell suppresses only duplicate visits of the same pair; overlapping or
  // duplicate input rectangles remain distinct exact cover members.
  thrust::device_vector<std::uint32_t> gate_counts(poly_count, 0);
  thrust::device_vector<std::uint64_t> gate_offsets(poly_count);
  thrust::device_vector<unsigned long long> query_visits(1, 0);
  count_raw_gate_intersections_kernel<<<
      blocks_for(poly_count, properties), kThreads>>>(
      poly, poly_count, active, grid,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(members.data()),
      thrust::raw_pointer_cast(gate_counts.data()),
      thrust::raw_pointer_cast(query_visits.data()), device_status);
  cuda_require(
      cudaGetLastError(), "POLY34 raw GATE count launch");
  cuda_require(
      cudaDeviceSynchronize(), "POLY34 raw GATE count synchronize");
  cuda_require(
      cudaMemcpy(
          &result.query_visits,
          thrust::raw_pointer_cast(query_visits.data()),
          sizeof(result.query_visits), cudaMemcpyDeviceToHost),
      "POLY34 raw GATE query visits D2H");
  cuda_require(
      cudaMemcpy(
          &status, device_status, sizeof(status), cudaMemcpyDeviceToHost),
      "POLY34 raw GATE count status D2H");
  if (status) return result;
  if (result.query_visits > request.max_query_visits) {
    status = kPolyQueryCapacity;
    cuda_require(
        cudaMemcpy(
            device_status, &status, sizeof(status), cudaMemcpyHostToDevice),
        "POLY34 raw GATE query-capacity H2D");
    return result;
  }

  const std::uint64_t gate_count = thrust::reduce(
      thrust::device, gate_counts.begin(), gate_counts.end(),
      std::uint64_t(0), thrust::plus<std::uint64_t>());
  if (gate_count > request.max_flat_boxes || gate_count > UINT32_MAX) {
    status = kPolyIntersectionCapacity;
    cuda_require(
        cudaMemcpy(
            device_status, &status, sizeof(status), cudaMemcpyHostToDevice),
        "POLY34 raw GATE capacity H2D");
    return result;
  }
  if (!gate_count) return result;

  thrust::exclusive_scan(
      thrust::device, gate_counts.begin(), gate_counts.end(),
      gate_offsets.begin());
  result.boxes.resize(static_cast<std::size_t>(gate_count));
  fill_raw_gate_intersections_kernel<<<
      blocks_for(poly_count, properties), kThreads>>>(
      poly, poly_count, active, grid,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(members.data()),
      thrust::raw_pointer_cast(gate_counts.data()),
      thrust::raw_pointer_cast(gate_offsets.data()), gate_count,
      thrust::raw_pointer_cast(result.boxes.data()), device_status);
  cuda_require(
      cudaGetLastError(), "POLY34 raw GATE fill launch");
  cuda_require(
      cudaDeviceSynchronize(), "POLY34 raw GATE fill synchronize");
  return result;
}

thrust::device_vector<std::uint8_t> prove_raw_gate_internal_sides(
    const PolyBox *gates, std::uint64_t gate_count,
    const PolyGrid &grid, std::uint64_t grid_cells,
    const PolyRequest &request, std::uint32_t *device_status,
    const cudaDeviceProp &properties)
{
  thrust::device_vector<std::uint8_t> masks;
  if (!gate_count) return masks;

  // Build a bounded index over the exact intersection cover.  A tile side is
  // marked internal only when one other GATE tile covers the complete
  // positive-width 1-DBU strip immediately across that side.  This is merely
  // a sufficient proof: split coverage retains the conservative fallback.
  thrust::device_vector<std::uint32_t> counts(grid_cells, 0);
  thrust::device_vector<std::uint32_t> offsets(grid_cells + 1);
  thrust::device_vector<std::uint32_t> cursors(grid_cells);
  thrust::device_vector<unsigned long long> membership_total(1, 0);
  count_grid_kernel<<<blocks_for(gate_count, properties), kThreads>>>(
      gates, gate_count, grid,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(membership_total.data()),
      device_status);
  cuda_require(
      cudaGetLastError(), "POLY34 raw GATE grid-count launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "POLY34 raw GATE grid-count synchronize");

  unsigned long long memberships = 0;
  std::uint32_t status = 0;
  cuda_require(
      cudaMemcpy(
          &memberships,
          thrust::raw_pointer_cast(membership_total.data()),
          sizeof(memberships), cudaMemcpyDeviceToHost),
      "POLY34 raw GATE membership count D2H");
  cuda_require(
      cudaMemcpy(
          &status, device_status, sizeof(status), cudaMemcpyDeviceToHost),
      "POLY34 raw GATE grid status D2H");
  if (status) return masks;
  const std::uint64_t maximum_memberships =
      std::max(
          request.max_poly_memberships,
          request.max_active_memberships);
  if (memberships > maximum_memberships || memberships > UINT32_MAX) {
    status = kPolyMembershipCapacity;
    cuda_require(
        cudaMemcpy(
            device_status, &status, sizeof(status), cudaMemcpyHostToDevice),
        "POLY34 raw GATE membership-capacity H2D");
    return masks;
  }

  thrust::exclusive_scan(
      thrust::device, counts.begin(), counts.end(), offsets.begin());
  const std::uint32_t terminal = static_cast<std::uint32_t>(memberships);
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(offsets.data()) + grid_cells,
          &terminal, sizeof(terminal), cudaMemcpyHostToDevice),
      "POLY34 raw GATE terminal offset H2D");
  thrust::copy(
      thrust::device, offsets.begin(), offsets.begin() + grid_cells,
      cursors.begin());
  thrust::device_vector<std::uint32_t> members(
      static_cast<std::size_t>(memberships));
  fill_grid_kernel<<<blocks_for(gate_count, properties), kThreads>>>(
      gates, gate_count, grid,
      thrust::raw_pointer_cast(cursors.data()),
      thrust::raw_pointer_cast(members.data()), device_status);
  cuda_require(
      cudaGetLastError(), "POLY34 raw GATE grid-fill launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "POLY34 raw GATE grid-fill synchronize");
  cuda_require(
      cudaMemcpy(
          &status, device_status, sizeof(status), cudaMemcpyDeviceToHost),
      "POLY34 raw GATE fill status D2H");
  if (status) return masks;

  masks.resize(static_cast<std::size_t>(gate_count));
  thrust::device_vector<unsigned long long> query_visits(1, 0);
  mark_raw_gate_internal_sides_kernel<<<
      blocks_for(gate_count, properties), kThreads>>>(
      gates, gate_count, grid,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(members.data()),
      thrust::raw_pointer_cast(masks.data()),
      thrust::raw_pointer_cast(query_visits.data()), device_status);
  cuda_require(
      cudaGetLastError(), "POLY34 raw GATE internal-side launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "POLY34 raw GATE internal-side synchronize");

  unsigned long long visits = 0;
  cuda_require(
      cudaMemcpy(
          &visits, thrust::raw_pointer_cast(query_visits.data()),
          sizeof(visits), cudaMemcpyDeviceToHost),
      "POLY34 raw GATE internal-side visits D2H");
  cuda_require(
      cudaMemcpy(
          &status, device_status, sizeof(status), cudaMemcpyDeviceToHost),
      "POLY34 raw GATE internal-side status D2H");
  if (!status && visits > request.max_query_visits) {
    status = kPolyQueryCapacity;
    cuda_require(
        cudaMemcpy(
            device_status, &status, sizeof(status), cudaMemcpyHostToDevice),
        "POLY34 raw GATE internal-side query-capacity H2D");
  }
  return masks;
}

ProfileResult run_profile(
    const PolyBox *primary, std::uint64_t primary_count,
    const PolyBox *gates, std::uint64_t gate_count,
    const PolyGrid &grid, std::uint64_t grid_cells,
    std::int64_t distance, std::uint64_t maximum_memberships,
    const PolyRequest &request,
    const std::uint8_t *proven_internal_sides,
    std::uint8_t *outcomes,
    std::uint32_t *device_status, const cudaDeviceProp &properties)
{
  ProfileResult result;
  const Clock::time_point grid_begin = Clock::now();
  thrust::device_vector<std::uint32_t> counts(grid_cells, 0);
  thrust::device_vector<std::uint32_t> offsets(grid_cells + 1);
  thrust::device_vector<std::uint32_t> cursors(grid_cells);
  thrust::device_vector<unsigned long long> membership_total(1, 0);
  count_grid_kernel<<<blocks_for(primary_count, properties), kThreads>>>(
      primary, primary_count, grid,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(membership_total.data()),
      device_status);
  cuda_require(cudaGetLastError(), "POLY34 grid-count launch");
  cuda_require(cudaDeviceSynchronize(), "POLY34 grid-count synchronize");

  unsigned long long memberships = 0;
  std::uint32_t status = 0;
  cuda_require(
      cudaMemcpy(
          &memberships,
          thrust::raw_pointer_cast(membership_total.data()),
          sizeof(memberships), cudaMemcpyDeviceToHost),
      "POLY34 membership count D2H");
  cuda_require(
      cudaMemcpy(
          &status, device_status, sizeof(status), cudaMemcpyDeviceToHost),
      "POLY34 grid-count status D2H");
  result.memberships = memberships;
  if (status || memberships > maximum_memberships ||
      memberships > UINT32_MAX) {
    if (!status) {
      status = kPolyMembershipCapacity;
      cuda_require(
          cudaMemcpy(
              device_status, &status, sizeof(status),
              cudaMemcpyHostToDevice),
          "POLY34 membership-capacity H2D");
    }
    result.grid_ns = elapsed_ns(grid_begin, Clock::now());
    return result;
  }

  thrust::exclusive_scan(
      thrust::device, counts.begin(), counts.end(), offsets.begin());
  const std::uint32_t terminal = static_cast<std::uint32_t>(memberships);
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(offsets.data()) + grid_cells,
          &terminal, sizeof(terminal), cudaMemcpyHostToDevice),
      "POLY34 terminal offset H2D");
  thrust::copy(
      thrust::device, offsets.begin(), offsets.begin() + grid_cells,
      cursors.begin());
  thrust::device_vector<std::uint32_t> members(
      static_cast<std::size_t>(memberships));
  fill_grid_kernel<<<blocks_for(primary_count, properties), kThreads>>>(
      primary, primary_count, grid,
      thrust::raw_pointer_cast(cursors.data()),
      thrust::raw_pointer_cast(members.data()), device_status);
  cuda_require(cudaGetLastError(), "POLY34 grid-fill launch");
  cuda_require(cudaDeviceSynchronize(), "POLY34 grid-fill synchronize");
  result.grid_ns = elapsed_ns(grid_begin, Clock::now());

  const Clock::time_point query_begin = Clock::now();
  thrust::device_vector<ProfileCounters> counters(1);
  cuda_require(
      cudaMemset(
          thrust::raw_pointer_cast(counters.data()), 0,
          sizeof(ProfileCounters)),
      "POLY34 profile counters clear");
  query_profile_kernel<<<blocks_for(gate_count, properties), kThreads>>>(
      gates, gate_count, primary, grid,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(members.data()), distance,
      request.max_query_visits, request.max_candidate_work,
      request.max_candidates_per_gate, proven_internal_sides, outcomes,
      thrust::raw_pointer_cast(counters.data()), device_status);
  cuda_require(cudaGetLastError(), "POLY34 profile query launch");
  cuda_require(cudaDeviceSynchronize(), "POLY34 profile query synchronize");
  ProfileCounters host{};
  cuda_require(
      cudaMemcpy(
          &host, thrust::raw_pointer_cast(counters.data()), sizeof(host),
          cudaMemcpyDeviceToHost),
      "POLY34 profile counters D2H");
  result.query_visits = host.query_visits;
  result.candidates = host.candidates;
  result.terminal_empty = host.terminal_empty;
  result.maximum_candidates = host.maximum_candidates;
  result.query_ns = elapsed_ns(query_begin, Clock::now());
  return result;
}

PipelineResult run_pipeline(const PolyRequest &request)
{
  PipelineResult result;
  const bool raw = raw_request(request);
  if (request.context_count > request.max_contexts ||
      request.box_count > request.max_flat_boxes ||
      request.flat_poly_box_count > request.max_flat_boxes ||
      request.flat_active_box_count > request.max_flat_boxes ||
      request.flat_gate_box_count > request.max_flat_boxes) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_CAPACITY;
    return result;
  }

  const __int128 expanded_left =
      static_cast<__int128>(request.scene_left) -
      request.poly4_distance;
  const __int128 expanded_bottom =
      static_cast<__int128>(request.scene_bottom) -
      request.poly4_distance;
  const __int128 expanded_right =
      static_cast<__int128>(request.scene_right) +
      request.poly4_distance;
  const __int128 expanded_top =
      static_cast<__int128>(request.scene_top) +
      request.poly4_distance;
  if (expanded_left < INT64_MIN || expanded_left > INT64_MAX ||
      expanded_bottom < INT64_MIN || expanded_bottom > INT64_MAX ||
      expanded_right < INT64_MIN || expanded_right > INT64_MAX ||
      expanded_top < INT64_MIN || expanded_top > INT64_MAX) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW;
    return result;
  }
  const std::int64_t base_x = floor_div_host(
      static_cast<std::int64_t>(expanded_left),
      request.grid_cell_size);
  const std::int64_t base_y = floor_div_host(
      static_cast<std::int64_t>(expanded_bottom),
      request.grid_cell_size);
  const std::int64_t maximum_x = floor_div_host(
      static_cast<std::int64_t>(expanded_right),
      request.grid_cell_size);
  const std::int64_t maximum_y = floor_div_host(
      static_cast<std::int64_t>(expanded_top),
      request.grid_cell_size);
  const __int128 width =
      static_cast<__int128>(maximum_x) - base_x + 1;
  const __int128 height =
      static_cast<__int128>(maximum_y) - base_y + 1;
  const __int128 cells = width * height;
  if (width <= 0 || height <= 0 ||
      width > UINT32_MAX || height > UINT32_MAX ||
      cells <= 0 || cells > UINT32_MAX ||
      cells > request.max_grid_cells) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_DENSE_CELL;
    return result;
  }
  result.grid_cells = static_cast<std::uint64_t>(cells);
  const PolyGrid grid = {
      base_x, base_y, request.grid_cell_size,
      static_cast<std::uint32_t>(width),
      static_cast<std::uint32_t>(height)};

  const Clock::time_point setup_begin = Clock::now();
  cuda_require(cudaSetDevice(request.device), "POLY34 cudaSetDevice");
  cuda_require(cudaFree(nullptr), "POLY34 CUDA context initialization");
  cudaDeviceProp properties{};
  cuda_require(
      cudaGetDeviceProperties(&properties, request.device),
      "POLY34 cudaGetDeviceProperties");
  if (request.poly_context_count >
          static_cast<std::uint64_t>(properties.maxGridSize[0]) ||
      request.active_context_count >
          static_cast<std::uint64_t>(properties.maxGridSize[0]) ||
      (!raw && request.gate_context_count >
          static_cast<std::uint64_t>(properties.maxGridSize[0]))) {
    result.fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    return result;
  }

  thrust::device_vector<PolyContext> contexts(request.context_count);
  thrust::device_vector<std::uint32_t>
      poly_contexts(request.poly_context_count);
  thrust::device_vector<std::uint64_t>
      poly_offsets(request.poly_offset_count);
  thrust::device_vector<std::uint32_t>
      active_contexts(request.active_context_count);
  thrust::device_vector<std::uint64_t>
      active_offsets(request.active_offset_count);
  thrust::device_vector<std::uint32_t>
      gate_contexts(raw ? 0 : request.gate_context_count);
  thrust::device_vector<std::uint64_t>
      gate_offsets(raw ? 0 : request.gate_offset_count);
  thrust::device_vector<PolyCell> cell_records(request.cell_count);
  thrust::device_vector<PolyBox> templates(request.box_count);
  thrust::device_vector<PolyBox> poly_boxes(request.flat_poly_box_count);
  thrust::device_vector<PolyBox> active_boxes(request.flat_active_box_count);
  thrust::device_vector<PolyBox> gate_boxes(
      raw ? 0 : request.flat_gate_box_count);
  thrust::device_vector<std::uint32_t> status(1, 0);
  result.setup_ns = elapsed_ns(setup_begin, Clock::now());

  const Clock::time_point h2d_begin = Clock::now();
#define POLY34_H2D(destination, source, count, type, label) \
  cuda_require( \
      cudaMemcpy( \
          thrust::raw_pointer_cast(destination.data()), source, \
          static_cast<std::size_t>(count) * sizeof(type), \
          cudaMemcpyHostToDevice), \
      label)
  POLY34_H2D(
      contexts, request.contexts, request.context_count,
      PolyContext, "POLY34 context H2D");
  POLY34_H2D(
      poly_contexts, request.poly_contexts,
      request.poly_context_count, std::uint32_t,
      "POLY34 POLY-context H2D");
  POLY34_H2D(
      poly_offsets, request.poly_offsets,
      request.poly_offset_count, std::uint64_t,
      "POLY34 POLY-offset H2D");
  POLY34_H2D(
      active_contexts, request.active_contexts,
      request.active_context_count, std::uint32_t,
      "POLY34 ACTIVE-context H2D");
  POLY34_H2D(
      active_offsets, request.active_offsets,
      request.active_offset_count, std::uint64_t,
      "POLY34 ACTIVE-offset H2D");
  if (!raw) {
    POLY34_H2D(
        gate_contexts, request.gate_contexts,
        request.gate_context_count, std::uint32_t,
        "POLY34 GATE-context H2D");
    POLY34_H2D(
        gate_offsets, request.gate_offsets,
        request.gate_offset_count, std::uint64_t,
        "POLY34 GATE-offset H2D");
  }
  POLY34_H2D(
      cell_records, request.cells, request.cell_count,
      PolyCell, "POLY34 cell H2D");
  POLY34_H2D(
      templates, request.boxes, request.box_count,
      PolyBox, "POLY34 box-template H2D");
#undef POLY34_H2D
  cuda_require(
      cudaDeviceSynchronize(), "POLY34 H2D synchronize");
  result.h2d_ns = elapsed_ns(h2d_begin, Clock::now());

  const Clock::time_point expand_begin = Clock::now();
  expand_boxes_kernel<<<
      static_cast<unsigned int>(request.poly_context_count),
      kContextThreads>>>(
      thrust::raw_pointer_cast(contexts.data()),
      thrust::raw_pointer_cast(poly_contexts.data()),
      thrust::raw_pointer_cast(poly_offsets.data()),
      request.poly_context_count,
      thrust::raw_pointer_cast(cell_records.data()),
      thrust::raw_pointer_cast(templates.data()),
      KLAYOUT_CUDA_SPATIAL_POLY34_POLY_DOMAIN,
      thrust::raw_pointer_cast(poly_boxes.data()),
      thrust::raw_pointer_cast(status.data()));
  expand_boxes_kernel<<<
      static_cast<unsigned int>(request.active_context_count),
      kContextThreads>>>(
      thrust::raw_pointer_cast(contexts.data()),
      thrust::raw_pointer_cast(active_contexts.data()),
      thrust::raw_pointer_cast(active_offsets.data()),
      request.active_context_count,
      thrust::raw_pointer_cast(cell_records.data()),
      thrust::raw_pointer_cast(templates.data()),
      KLAYOUT_CUDA_SPATIAL_POLY34_ACTIVE_DOMAIN,
      thrust::raw_pointer_cast(active_boxes.data()),
      thrust::raw_pointer_cast(status.data()));
  if (!raw) {
    expand_boxes_kernel<<<
        static_cast<unsigned int>(request.gate_context_count),
        kContextThreads>>>(
        thrust::raw_pointer_cast(contexts.data()),
        thrust::raw_pointer_cast(gate_contexts.data()),
        thrust::raw_pointer_cast(gate_offsets.data()),
        request.gate_context_count,
        thrust::raw_pointer_cast(cell_records.data()),
        thrust::raw_pointer_cast(templates.data()),
        KLAYOUT_CUDA_SPATIAL_POLY34_GATE_DOMAIN,
        thrust::raw_pointer_cast(gate_boxes.data()),
        thrust::raw_pointer_cast(status.data()));
  }
  cuda_require(cudaGetLastError(), "POLY34 expansion launch");
  cuda_require(cudaDeviceSynchronize(), "POLY34 expansion synchronize");
  result.expanded_poly = request.flat_poly_box_count;
  result.expanded_active = request.flat_active_box_count;

  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "POLY34 expansion status D2H");
  if (host_status) {
    result.device_flags = host_status;
    result.fallback_flags = fallback_from_device_flags(host_status);
    return result;
  }

  thrust::device_vector<std::uint8_t> proven_internal_sides;
  if (raw) {
    RawGateResult derived = derive_raw_gates(
        thrust::raw_pointer_cast(poly_boxes.data()),
        request.flat_poly_box_count,
        thrust::raw_pointer_cast(active_boxes.data()),
        request.flat_active_box_count, grid, result.grid_cells, request,
        thrust::raw_pointer_cast(status.data()), properties);
    gate_boxes.swap(derived.boxes);
    cuda_require(
        cudaMemcpy(
            &host_status, thrust::raw_pointer_cast(status.data()),
            sizeof(host_status), cudaMemcpyDeviceToHost),
        "POLY34 raw GATE status D2H");
    if (host_status) {
      result.device_flags = host_status;
      result.fallback_flags = fallback_from_device_flags(host_status);
      return result;
    }
    proven_internal_sides = prove_raw_gate_internal_sides(
        thrust::raw_pointer_cast(gate_boxes.data()),
        gate_boxes.size(), grid, result.grid_cells, request,
        thrust::raw_pointer_cast(status.data()), properties);
    cuda_require(
        cudaMemcpy(
            &host_status, thrust::raw_pointer_cast(status.data()),
            sizeof(host_status), cudaMemcpyDeviceToHost),
        "POLY34 raw GATE internal-side status D2H");
    if (host_status) {
      result.device_flags = host_status;
      result.fallback_flags = fallback_from_device_flags(host_status);
      return result;
    }
    if (proven_internal_sides.size() != gate_boxes.size()) {
      result.fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      return result;
    }
  }
  result.expanded_gate = gate_boxes.size();
  result.expand_ns = elapsed_ns(expand_begin, Clock::now());
  const std::uint64_t gate_count = result.expanded_gate;
  thrust::device_vector<std::uint8_t> poly_outcomes(gate_count);
  thrust::device_vector<std::uint8_t> active_outcomes(gate_count);

  result.poly = run_profile(
      thrust::raw_pointer_cast(poly_boxes.data()),
      request.flat_poly_box_count,
      thrust::raw_pointer_cast(gate_boxes.data()),
      gate_count, grid, result.grid_cells,
      request.poly3_distance, request.max_poly_memberships,
      request,
      raw ? thrust::raw_pointer_cast(proven_internal_sides.data()) : nullptr,
      thrust::raw_pointer_cast(poly_outcomes.data()),
      thrust::raw_pointer_cast(status.data()), properties);
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "POLY34 POLY-profile status D2H");
  if (host_status) {
    result.device_flags = host_status;
    result.fallback_flags = fallback_from_device_flags(host_status);
    return result;
  }

  result.active = run_profile(
      thrust::raw_pointer_cast(active_boxes.data()),
      request.flat_active_box_count,
      thrust::raw_pointer_cast(gate_boxes.data()),
      gate_count, grid, result.grid_cells,
      request.poly4_distance, request.max_active_memberships,
      request,
      raw ? thrust::raw_pointer_cast(proven_internal_sides.data()) : nullptr,
      thrust::raw_pointer_cast(active_outcomes.data()),
      thrust::raw_pointer_cast(status.data()), properties);
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "POLY34 ACTIVE-profile status D2H");
  if (host_status) {
    result.device_flags = host_status;
    result.fallback_flags = fallback_from_device_flags(host_status);
    return result;
  }

  const Clock::time_point d2h_begin = Clock::now();
  thrust::device_vector<unsigned long long> atomic_empty(1, 0);
  thrust::device_vector<unsigned long long> fallback(1, 0);
  combine_outcomes_kernel<<<
      blocks_for(gate_count, properties), kThreads>>>(
      thrust::raw_pointer_cast(poly_outcomes.data()),
      thrust::raw_pointer_cast(active_outcomes.data()),
      gate_count,
      thrust::raw_pointer_cast(atomic_empty.data()),
      thrust::raw_pointer_cast(fallback.data()));
  cuda_require(cudaGetLastError(), "POLY34 outcome reduction launch");
  cuda_require(
      cudaDeviceSynchronize(), "POLY34 outcome reduction synchronize");
  cuda_require(
      cudaMemcpy(
          &result.atomic_terminal_empty,
          thrust::raw_pointer_cast(atomic_empty.data()),
          sizeof(result.atomic_terminal_empty), cudaMemcpyDeviceToHost),
      "POLY34 atomic-empty D2H");
  cuda_require(
      cudaMemcpy(
          &result.fallback_gates,
          thrust::raw_pointer_cast(fallback.data()),
          sizeof(result.fallback_gates), cudaMemcpyDeviceToHost),
      "POLY34 fallback-gate D2H");
  result.d2h_ns = elapsed_ns(d2h_begin, Clock::now());

  if (result.poly.terminal_empty == gate_count) {
    result.certified_empty_mask |= KLAYOUT_CUDA_SPATIAL_POLY3_RULE;
  }
  if (result.active.terminal_empty == gate_count) {
    result.certified_empty_mask |= KLAYOUT_CUDA_SPATIAL_POLY4_RULE;
  }
  if (result.atomic_terminal_empty == gate_count &&
      result.fallback_gates == 0 &&
      result.certified_empty_mask ==
          KLAYOUT_CUDA_SPATIAL_POLY34_ALL_RULES) {
    result.disposition = KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE;
  } else {
    result.disposition = KLAYOUT_CUDA_SPATIAL_POLY34_NOT_EMPTY;
  }
  return result;
}

void echo_request(const PolyRequest &request, PolyResult *result)
{
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->format_version = request.format_version;
  result->requested_mask = request.requested_mask;
  result->dbu_per_micron = request.dbu_per_micron;
  result->root_cell = request.root_cell;
  result->device = request.device;
  result->poly3_distance = request.poly3_distance;
  result->poly4_distance = request.poly4_distance;
  result->grid_cell_size = request.grid_cell_size;
  result->store_identity = request.store_identity;
  result->layout_identity = request.layout_identity;
  result->top_cell_identity = request.top_cell_identity;
  result->poly_layer_id = request.poly_layer_id;
  result->active_layer_id = request.active_layer_id;
  result->gate_layer_id = request.gate_layer_id;
  std::copy(
      request.scene_digest, request.scene_digest + 32,
      result->scene_digest);
  result->context_count = request.context_count;
  result->poly_context_count = request.poly_context_count;
  result->active_context_count = request.active_context_count;
  result->gate_context_count = request.gate_context_count;
  result->cell_count = request.cell_count;
  result->box_count = request.box_count;
  result->flat_poly_box_count = request.flat_poly_box_count;
  result->flat_active_box_count = request.flat_active_box_count;
  result->flat_gate_box_count = request.flat_gate_box_count;
}

int run_request(const PolyRequest *request, PolyResult *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->disposition = KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN;
  if (!request || !request_structurally_valid(*request)) {
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    set_message(
        result,
        "unsupported, malformed, or digest-mismatched POLY34 request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  echo_request(*request, result);

  const Clock::time_point total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> lock(poly_pipeline_mutex());
    const PipelineResult pipeline = run_pipeline(*request);
    result->fallback_flags = pipeline.fallback_flags;
    result->device_flags = pipeline.device_flags;
    result->certified_empty_mask = pipeline.certified_empty_mask;
    result->disposition = pipeline.disposition;
    result->expanded_poly_box_count = pipeline.expanded_poly;
    result->expanded_active_box_count = pipeline.expanded_active;
    result->expanded_gate_box_count = pipeline.expanded_gate;
    if (raw_request(*request)) {
      // Format 2 has no host GATE census to echo.  Publish the complete
      // device-derived tile count in both result census fields so the host can
      // validate all terminal/fallback counters against one bound value.
      result->flat_gate_box_count = pipeline.expanded_gate;
    }
    result->grid_cell_count = pipeline.grid_cells;
    result->poly_membership_count = pipeline.poly.memberships;
    result->active_membership_count = pipeline.active.memberships;
    result->poly_query_visit_count = pipeline.poly.query_visits;
    result->active_query_visit_count = pipeline.active.query_visits;
    result->poly_candidate_count = pipeline.poly.candidates;
    result->active_candidate_count = pipeline.active.candidates;
    result->poly_terminal_empty_count = pipeline.poly.terminal_empty;
    result->active_terminal_empty_count = pipeline.active.terminal_empty;
    result->atomic_terminal_empty_count =
        pipeline.atomic_terminal_empty;
    result->fallback_gate_count = pipeline.fallback_gates;
    result->maximum_poly_candidates =
        pipeline.poly.maximum_candidates;
    result->maximum_active_candidates =
        pipeline.active.maximum_candidates;
    result->setup_ns = pipeline.setup_ns;
    result->h2d_ns = pipeline.h2d_ns;
    result->expand_ns = pipeline.expand_ns;
    result->poly_grid_ns = pipeline.poly.grid_ns;
    result->poly_query_ns = pipeline.poly.query_ns;
    result->active_grid_ns = pipeline.active.grid_ns;
    result->active_query_ns = pipeline.active.query_ns;
    result->d2h_ns = pipeline.d2h_ns;
    result->total_ns = elapsed_ns(total_begin, Clock::now());

    if (pipeline.fallback_flags || pipeline.device_flags) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->disposition = KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN;
      set_message(result, "POLY34 pipeline requested CPU fallback");
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }
    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    set_message(
        result,
        result->disposition == KLAYOUT_CUDA_SPATIAL_POLY34_COMPLETE
            ? "complete atomic POLY.3/.4 terminal-empty certificate"
            : "POLY.3/.4 certificate declined; run both CPU rules");
    return KLAYOUT_CUDA_SPATIAL_OK;
  } catch (const std::exception &error) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->disposition = KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN;
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    set_message(result, error.what());
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  } catch (...) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->disposition = KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN;
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    set_message(result, "unknown POLY34 backend exception");
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}

}  // namespace

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_poly34_empty_v1(
    const klayout_cuda_spatial_poly34_request_v1 *request,
    klayout_cuda_spatial_poly34_result_v1 *result)
{
  try {
    return run_request(request, result);
  } catch (...) {
    if (result) {
      std::memset(result, 0, sizeof(*result));
      result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result->struct_size = sizeof(*result);
      result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      result->disposition = KLAYOUT_CUDA_SPATIAL_POLY34_UNCERTAIN;
      set_message(result, "exception escaped POLY34 request boundary");
    }
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}
