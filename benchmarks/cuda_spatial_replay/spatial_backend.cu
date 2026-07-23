/*
 * Optional CUDA self and bipartite AABB broad phase for KLayout.
 *
 * This DSO intentionally exposes only the versioned POD C ABI.  KLayout
 * remains CUDA-free and loads it explicitly at runtime.
 */

#include "dbCudaSpatialApi.h"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

static_assert(
    std::is_trivially_copyable<klayout_cuda_spatial_m1_contact_v1>::value,
    "M1 contacts must remain POD across the DSO boundary");
static_assert(
    std::is_trivially_copyable<klayout_cuda_spatial_m1_edge_v1>::value,
    "M1 edges must remain POD across the DSO boundary");
static_assert(
    std::is_trivially_copyable<klayout_cuda_spatial_m1_survivor_v1>::value,
    "M1 survivors must remain POD across the DSO boundary");
static_assert(sizeof(klayout_cuda_spatial_m1_contact_v1) == 48,
              "unexpected M1 contact ABI padding");
static_assert(sizeof(klayout_cuda_spatial_m1_edge_v1) == 40,
              "unexpected M1 edge ABI padding");
static_assert(sizeof(klayout_cuda_spatial_m1_request_v1) == 64,
              "unexpected M1 request ABI padding");
static_assert(sizeof(klayout_cuda_spatial_m1_survivor_v1) == 16,
              "unexpected M1 survivor ABI padding");

struct alignas(16) PackedAabb {
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint32_t id;
  std::uint32_t side;
  std::uint32_t reserved0;
  std::uint32_t reserved1;
};

struct GridConfig {
  std::uint64_t cell_size;
  std::uint64_t enlargement;
  std::uint32_t max_cells_per_record;
  std::uint32_t max_records_per_cell;
  std::uint32_t bipartite;
};

struct CellRange {
  std::int64_t x0;
  std::int64_t x1;
  std::int64_t y0;
  std::int64_t y1;
};

struct alignas(8) CellKey {
  std::int64_t x;
  std::int64_t y;
  std::uint32_t context;
  std::uint32_t side;
};

struct CellKeyLess {
  __host__ __device__ bool operator()(const CellKey &a, const CellKey &b) const {
    if (a.context != b.context) return a.context < b.context;
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    return a.side < b.side;
  }
};

struct CellKeyEqual {
  __host__ __device__ bool operator()(const CellKey &a, const CellKey &b) const {
    // Side participates in ordering so all subjects precede intruders, but it
    // deliberately does not split the spatial cell during reduce_by_key.
    return a.context == b.context && a.x == b.x && a.y == b.y;
  }
};

struct PipelineResult {
  std::uint32_t fallback_flags = 0;
  std::uint64_t memberships = 0;
  std::uint64_t occupied_cells = 0;
  std::uint64_t pair_work = 0;
  std::vector<std::uint64_t> pairs;
  std::uint64_t setup_ns = 0;
  std::uint64_t h2d_ns = 0;
  std::uint64_t broad_phase_ns = 0;
  std::uint64_t sort_unique_ns = 0;
  std::uint64_t d2h_ns = 0;
};

struct alignas(16) PackedM1Endpoints {
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
};

struct M1PipelineResult {
  std::uint32_t fallback_flags = 0;
  std::uint32_t disposition = KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN;
  std::uint64_t memberships = 0;
  std::uint64_t occupied_cells = 0;
  std::uint64_t pair_work = 0;
  std::uint64_t broad_candidates = 0;
  std::uint64_t full_side_hits = 0;
  std::uint64_t partial_candidates = 0;
  std::uint64_t non_manhattan_candidates = 0;
  std::uint64_t uncertain_contacts = 0;
  std::uint64_t disallowed_contacts = 0;
  std::vector<klayout_cuda_spatial_m1_survivor_v1> survivors;
  std::uint64_t setup_ns = 0;
  std::uint64_t h2d_ns = 0;
  std::uint64_t broad_phase_ns = 0;
  std::uint64_t classify_ns = 0;
  std::uint64_t d2h_ns = 0;
};

enum M1Counter : std::uint32_t {
  kM1FullSideHits = 0,
  kM1PartialCandidates = 1,
  kM1NonManhattanCandidates = 2,
  kM1UncertainContacts = 3,
  kM1DisallowedContacts = 4,
  kM1CounterCount = 5
};

struct M1SurvivorLess {
  __host__ __device__ bool operator()(
      const klayout_cuda_spatial_m1_survivor_v1 &a,
      const klayout_cuda_spatial_m1_survivor_v1 &b) const {
    return a.contact_id < b.contact_id;
  }
};

struct M1SurvivorIsCulled {
  __host__ __device__ bool operator()(
      const klayout_cuda_spatial_m1_survivor_v1 &record) const {
    return record.flags == 0;
  }
};

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

std::uint64_t ceil_div_u64(std::uint64_t value, std::uint64_t divisor) {
  return value / divisor + (value % divisor != 0);
}

void cuda_check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

__host__ __device__ std::int64_t floor_div(std::int64_t value,
                                           std::int64_t divisor) {
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

__host__ __device__ CellRange cell_range(const PackedAabb &record,
                                         GridConfig config) {
  const std::int64_t cell = static_cast<std::int64_t>(config.cell_size);
  const std::int64_t enlargement =
      static_cast<std::int64_t>(config.enlargement);
  return CellRange{floor_div(record.left - enlargement, cell),
                   floor_div(record.right + enlargement, cell),
                   floor_div(record.bottom - enlargement, cell),
                   floor_div(record.top + enlargement, cell)};
}

__host__ __device__ bool membership_count_bounded(const CellRange &range,
                                                  std::uint64_t limit,
                                                  std::uint64_t &count) {
  const std::uint64_t dx = static_cast<std::uint64_t>(range.x1) -
                           static_cast<std::uint64_t>(range.x0);
  const std::uint64_t dy = static_cast<std::uint64_t>(range.y1) -
                           static_cast<std::uint64_t>(range.y0);
  if (dx == UINT64_MAX || dy == UINT64_MAX) return false;
  const std::uint64_t width = dx + 1;
  const std::uint64_t height = dy + 1;
  if (width > limit || height > limit || width > limit / height) return false;
  count = width * height;
  return true;
}

__host__ __device__ bool boxes_overlap_strict(const PackedAabb &a,
                                              const PackedAabb &b,
                                              std::uint64_t enlargement) {
  const std::int64_t e = static_cast<std::int64_t>(enlargement);
  return a.left < b.right + e && b.left < a.right + e &&
         a.bottom < b.top + e && b.bottom < a.top + e;
}

__host__ __device__ std::uint64_t unordered_pair_key(std::uint32_t a,
                                                     std::uint32_t b) {
  const std::uint32_t lo = a < b ? a : b;
  const std::uint32_t hi = a < b ? b : a;
  return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

__global__ void count_memberships_kernel(const PackedAabb *records,
                                         std::uint32_t record_count,
                                         GridConfig config,
                                         std::uint32_t *counts,
                                         std::uint32_t *fallback_flags) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < record_count; index += stride) {
    std::uint64_t count = 0;
    if (!membership_count_bounded(cell_range(records[index], config),
                                  config.max_cells_per_record, count)) {
      counts[index] = 0;
      atomicOr(fallback_flags,
               static_cast<std::uint32_t>(KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN));
    } else {
      counts[index] = static_cast<std::uint32_t>(count);
    }
  }
}

__global__ void fill_memberships_kernel(const PackedAabb *records,
                                        std::uint32_t record_count,
                                        GridConfig config,
                                        const std::uint64_t *offsets,
                                        CellKey *keys,
                                        std::uint32_t *record_indices) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < record_count; index += stride) {
    const CellRange range = cell_range(records[index], config);
    std::uint64_t output = offsets[index];
    for (std::int64_t y = range.y0;; ++y) {
      for (std::int64_t x = range.x0;; ++x) {
        keys[output] =
            CellKey{x, y, records[index].reserved0, records[index].side};
        record_indices[output++] = static_cast<std::uint32_t>(index);
        if (x == range.x1) break;
      }
      if (y == range.y1) break;
    }
  }
}

__global__ void count_pair_work_kernel(
    const PackedAabb *records, const std::uint32_t *record_indices,
    const std::uint64_t *cell_offsets, const std::uint32_t *cell_counts,
    std::uint64_t occupied_cells, GridConfig config,
    std::uint64_t *pair_work_counts, std::uint32_t *side_a_counts,
    std::uint32_t *fallback_flags) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t cell = first; cell < occupied_cells; cell += stride) {
    const std::uint32_t count = cell_counts[cell];
    if (count > config.max_records_per_cell) {
      pair_work_counts[cell] = 0;
      atomicOr(fallback_flags,
               static_cast<std::uint32_t>(KLAYOUT_CUDA_SPATIAL_FALLBACK_DENSE_CELL));
      continue;
    }
    if (config.bipartite) {
      const std::uint64_t offset = cell_offsets[cell];
      std::uint32_t side_a = 0;
      while (side_a < count &&
             records[record_indices[offset + side_a]].side == 0) {
        ++side_a;
      }
      side_a_counts[cell] = side_a;
      pair_work_counts[cell] =
          static_cast<std::uint64_t>(side_a) * (count - side_a);
    } else {
      side_a_counts[cell] = 0;
      pair_work_counts[cell] =
          static_cast<std::uint64_t>(count) * (count - 1) / 2;
    }
  }
}

__device__ std::uint64_t pair_row_start(std::uint32_t row,
                                        std::uint32_t count) {
  return static_cast<std::uint64_t>(row) * (2ULL * count - row - 1) / 2;
}

__global__ void mark_pair_candidates_kernel(
    const PackedAabb *records, const std::uint32_t *record_indices,
    const std::uint64_t *cell_offsets, const std::uint32_t *cell_counts,
    const std::uint32_t *side_a_counts,
    const std::uint64_t *pair_work_offsets, std::uint64_t occupied_cells,
    std::uint64_t total_pair_work, GridConfig config,
    std::uint64_t *candidate_or_zero) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t work = first; work < total_pair_work; work += stride) {
    std::uint64_t lo = 0;
    std::uint64_t hi = occupied_cells;
    while (lo < hi) {
      const std::uint64_t mid = lo + (hi - lo) / 2;
      if (pair_work_offsets[mid] <= work)
        lo = mid + 1;
      else
        hi = mid;
    }
    const std::uint64_t cell = lo - 1;
    const std::uint64_t local = work - pair_work_offsets[cell];
    const std::uint32_t count = cell_counts[cell];
    std::uint32_t ai = 0;
    std::uint32_t bi = 0;
    if (config.bipartite) {
      const std::uint32_t side_a = side_a_counts[cell];
      const std::uint32_t side_b = count - side_a;
      ai = static_cast<std::uint32_t>(local / side_b);
      bi = side_a + static_cast<std::uint32_t>(local % side_b);
    } else {
      std::uint32_t row_lo = 0;
      std::uint32_t row_hi = count - 1;
      while (row_lo < row_hi) {
        const std::uint32_t mid = row_lo + (row_hi - row_lo + 1) / 2;
        if (pair_row_start(mid, count) <= local) {
          row_lo = mid;
        } else {
          row_hi = mid - 1;
        }
      }
      ai = row_lo;
      bi = static_cast<std::uint32_t>(
          ai + 1 + local - pair_row_start(ai, count));
    }
    const std::uint64_t offset = cell_offsets[cell];
    const PackedAabb a = records[record_indices[offset + ai]];
    const PackedAabb b = records[record_indices[offset + bi]];
    candidate_or_zero[work] =
        boxes_overlap_strict(a, b, config.enlargement)
            ? (config.bipartite
                   ? (static_cast<std::uint64_t>(a.id) << 32) | b.id
                   : unordered_pair_key(a.id, b.id))
            : 0;
  }
}

__device__ bool ordered_gap_less_than(std::int64_t outer,
                                      std::int64_t inner,
                                      std::uint64_t distance) {
  // Preconditions: outer >= inner in signed coordinate order.  Unsigned
  // subtraction then gives the exact mathematical difference across the
  // entire int64 domain without signed overflow.
  return static_cast<std::uint64_t>(outer) -
             static_cast<std::uint64_t>(inner) <
         distance;
}

__device__ void mark_m1_side_candidate(
    const PackedAabb &contact, const PackedAabb &edge_box,
    const PackedM1Endpoints &edge, std::uint64_t distance,
    std::uint32_t contact_index, std::uint32_t *side_masks,
    std::uint32_t *uncertain, unsigned long long *counters) {
  std::uint32_t side_bit = 0;
  std::int64_t contact_lo = 0;
  std::int64_t contact_hi = 0;
  std::int64_t edge_lo = 0;
  std::int64_t edge_hi = 0;
  bool deficient_distance = false;

  if (edge.x1 == edge.x2 && edge.y1 != edge.y2) {
    edge_lo = edge_box.bottom;
    edge_hi = edge_box.top;
    contact_lo = contact.bottom;
    contact_hi = contact.top;
    if (edge.y2 > edge.y1) {
      // A clockwise rectangle's left edge points upward.
      side_bit = 0x1u;
      deficient_distance =
          edge.x1 <= contact.left &&
          ordered_gap_less_than(contact.left, edge.x1, distance);
    } else {
      // Its right edge points downward.
      side_bit = 0x4u;
      deficient_distance =
          edge.x1 >= contact.right &&
          ordered_gap_less_than(edge.x1, contact.right, distance);
    }
  } else if (edge.y1 == edge.y2 && edge.x1 != edge.x2) {
    edge_lo = edge_box.left;
    edge_hi = edge_box.right;
    contact_lo = contact.left;
    contact_hi = contact.right;
    if (edge.x2 > edge.x1) {
      // A clockwise rectangle's top edge points right.
      side_bit = 0x2u;
      deficient_distance =
          edge.y1 >= contact.top &&
          ordered_gap_less_than(edge.y1, contact.top, distance);
    } else {
      // Its bottom edge points left.
      side_bit = 0x8u;
      deficient_distance =
          edge.y1 <= contact.bottom &&
          ordered_gap_less_than(contact.bottom, edge.y1, distance);
    }
  } else {
    // Degenerate and non-Manhattan edges require the exact CPU predicate.
    atomicOr(uncertain + contact_index, 1u);
    atomicAdd(counters + kM1NonManhattanCandidates, 1ULL);
    return;
  }

  if (!deficient_distance || edge_hi < contact_lo ||
      edge_lo > contact_hi) {
    return;
  }

  if (edge_lo <= contact_lo && edge_hi >= contact_hi) {
    atomicOr(side_masks + contact_index, side_bit);
    atomicAdd(counters + kM1FullSideHits, 1ULL);
  } else {
    // Rectangle filtering matches literal whole subject edges.  Even a
    // one-point or collectively complete set of partial projections is not a
    // whole-edge match and therefore cannot be certified here.
    atomicOr(uncertain + contact_index, 1u);
    atomicAdd(counters + kM1PartialCandidates, 1ULL);
  }
}

__global__ void analyze_m1_candidates_kernel(
    const PackedAabb *records, const PackedM1Endpoints *metal1_edges,
    std::uint32_t contact_count, std::uint32_t record_count,
    const std::uint64_t *pair_keys, std::uint64_t pair_count,
    std::uint64_t distance,
    std::uint32_t *side_masks, std::uint32_t *uncertain,
    unsigned long long *counters, std::uint32_t *fallback_flags) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < pair_count; index += stride) {
    const std::uint64_t key = pair_keys[index];
    const std::uint32_t contact_id = static_cast<std::uint32_t>(key >> 32);
    const std::uint32_t edge_global_id =
        static_cast<std::uint32_t>(key & UINT64_C(0xffffffff));
    if (contact_id == 0 || contact_id > contact_count ||
        edge_global_id <= contact_count || edge_global_id > record_count) {
      atomicOr(
          fallback_flags,
          static_cast<std::uint32_t>(
              KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT));
      continue;
    }

    const std::uint32_t contact_index = contact_id - 1;
    const std::uint32_t edge_index = edge_global_id - contact_count - 1;
    const PackedAabb contact = records[contact_index];
    const PackedAabb edge_box = records[edge_global_id - 1];
    if (contact.reserved0 != edge_box.reserved0) {
      // The context-bearing sparse key makes this an internal invariant
      // failure.  Force the entire speculative request to fall back: silently
      // dropping the pair could otherwise produce a false clean certificate.
      atomicOr(
          fallback_flags,
          static_cast<std::uint32_t>(
              KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT));
      continue;
    }

    mark_m1_side_candidate(contact, edge_box, metal1_edges[edge_index],
                           distance, contact_index, side_masks, uncertain,
                           counters);
  }
}

__device__ bool m1_mask_is_waivable(std::uint32_t mask) {
  mask &= 0xfu;
  const bool singleton = mask != 0 && (mask & (mask - 1)) == 0;
  return mask == 0 || singleton || mask == 0x5u || mask == 0xau;
}

__global__ void classify_m1_contacts_kernel(
    const std::uint64_t *contact_ids, const std::uint32_t *side_masks,
    const std::uint32_t *uncertain, std::uint32_t contact_count,
    klayout_cuda_spatial_m1_survivor_v1 *survivors,
    unsigned long long *counters) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < contact_count; index += stride) {
    const std::uint32_t mask = side_masks[index] & 0xfu;
    std::uint32_t flags = 0;
    if (uncertain[index]) {
      flags |= KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_UNCERTAIN;
      atomicAdd(counters + kM1UncertainContacts, 1ULL);
    }
    if (!m1_mask_is_waivable(mask)) {
      flags |= KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_DISALLOWED_MASK;
      atomicAdd(counters + kM1DisallowedContacts, 1ULL);
    }
    survivors[index] = klayout_cuda_spatial_m1_survivor_v1{
        contact_ids[index], mask, flags};
  }
}

void set_message(klayout_cuda_spatial_result_v1 *result,
                 const std::string &message) {
  std::snprintf(result->message, sizeof(result->message), "%s", message.c_str());
}

void set_message(klayout_cuda_spatial_m1_result_v1 *result,
                 const std::string &message) {
  std::snprintf(result->message, sizeof(result->message), "%s", message.c_str());
}

PipelineResult run_pipeline(const std::vector<PackedAabb> &records,
                            const klayout_cuda_spatial_config_v1 &options,
                            std::uint64_t enlargement,
                            bool bipartite) {
  PipelineResult result;
  GridConfig config{options.cell_size, enlargement,
                    options.max_cells_per_record,
                    options.max_records_per_cell,
                    bipartite ? 1u : 0u};
  const std::uint32_t record_count = static_cast<std::uint32_t>(records.size());
  constexpr std::uint32_t threads = 256;
  const std::uint32_t record_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          ceil_div_u64(static_cast<std::uint64_t>(records.size()), threads),
          65535));

  const auto setup_begin = Clock::now();
  cuda_check(cudaSetDevice(options.device), "cudaSetDevice");
  cuda_check(cudaFree(nullptr), "CUDA context initialization");
  thrust::device_vector<PackedAabb> device_records(records.size());
  thrust::device_vector<std::uint32_t> device_status(1, 0);
  result.setup_ns = elapsed_ns(setup_begin, Clock::now());

  const auto h2d_begin = Clock::now();
  cuda_check(cudaMemcpy(thrust::raw_pointer_cast(device_records.data()),
                        records.data(), records.size() * sizeof(PackedAabb),
                        cudaMemcpyHostToDevice),
             "record H2D copy");
  result.h2d_ns = elapsed_ns(h2d_begin, Clock::now());

  const auto broad_begin = Clock::now();
  thrust::device_vector<std::uint32_t> counts(record_count);
  thrust::device_vector<std::uint64_t> offsets(record_count);
  count_memberships_kernel<<<record_blocks, threads>>>(
      thrust::raw_pointer_cast(device_records.data()), record_count, config,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(device_status.data()));
  cuda_check(cudaGetLastError(), "count-memberships launch");
  thrust::exclusive_scan(thrust::device, counts.begin(), counts.end(),
                         offsets.begin(), std::uint64_t{0});
  std::uint64_t last_offset = 0;
  std::uint32_t last_count = 0;
  cuda_check(cudaMemcpy(&last_offset,
                        thrust::raw_pointer_cast(offsets.data()) + record_count - 1,
                        sizeof(last_offset), cudaMemcpyDeviceToHost),
             "membership-offset control copy");
  cuda_check(cudaMemcpy(&last_count,
                        thrust::raw_pointer_cast(counts.data()) + record_count - 1,
                        sizeof(last_count), cudaMemcpyDeviceToHost),
             "membership-count control copy");
  result.memberships = last_offset + last_count;
  cuda_check(cudaMemcpy(&result.fallback_flags,
                        thrust::raw_pointer_cast(device_status.data()),
                        sizeof(result.fallback_flags), cudaMemcpyDeviceToHost),
             "membership-status control copy");
  if (result.memberships > options.max_memberships)
    result.fallback_flags |= KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY;
  if (result.fallback_flags) {
    cuda_check(cudaDeviceSynchronize(), "membership-stage synchronize");
    result.broad_phase_ns = elapsed_ns(broad_begin, Clock::now());
    return result;
  }

  thrust::device_vector<CellKey> membership_keys(result.memberships);
  thrust::device_vector<std::uint32_t> membership_records(result.memberships);
  fill_memberships_kernel<<<record_blocks, threads>>>(
      thrust::raw_pointer_cast(device_records.data()), record_count, config,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(membership_keys.data()),
      thrust::raw_pointer_cast(membership_records.data()));
  cuda_check(cudaGetLastError(), "fill-memberships launch");
  thrust::sort_by_key(thrust::device, membership_keys.begin(),
                      membership_keys.end(), membership_records.begin(),
                      CellKeyLess{});

  thrust::device_vector<CellKey> unique_cell_keys(result.memberships);
  thrust::device_vector<std::uint32_t> cell_counts(result.memberships);
  const auto reduced_end = thrust::reduce_by_key(
      thrust::device, membership_keys.begin(), membership_keys.end(),
      thrust::make_constant_iterator<std::uint32_t>(1), unique_cell_keys.begin(),
      cell_counts.begin(), CellKeyEqual{});
  result.occupied_cells =
      static_cast<std::uint64_t>(reduced_end.first - unique_cell_keys.begin());
  cell_counts.resize(result.occupied_cells);
  thrust::device_vector<std::uint64_t> cell_offsets(result.occupied_cells);
  thrust::exclusive_scan(thrust::device, cell_counts.begin(), cell_counts.end(),
                         cell_offsets.begin(), std::uint64_t{0});

  thrust::device_vector<std::uint64_t> pair_counts(result.occupied_cells);
  thrust::device_vector<std::uint64_t> pair_offsets(result.occupied_cells);
  thrust::device_vector<std::uint32_t> side_a_counts(result.occupied_cells);
  const std::uint32_t cell_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          ceil_div_u64(result.occupied_cells, threads), 65535));
  if (cell_blocks) {
    count_pair_work_kernel<<<cell_blocks, threads>>>(
        thrust::raw_pointer_cast(device_records.data()),
        thrust::raw_pointer_cast(membership_records.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(cell_counts.data()), result.occupied_cells,
        config, thrust::raw_pointer_cast(pair_counts.data()),
        thrust::raw_pointer_cast(side_a_counts.data()),
        thrust::raw_pointer_cast(device_status.data()));
    cuda_check(cudaGetLastError(), "count-pair-work launch");
  }
  thrust::exclusive_scan(thrust::device, pair_counts.begin(), pair_counts.end(),
                         pair_offsets.begin(), std::uint64_t{0});
  if (result.occupied_cells) {
    std::uint64_t final_offset = 0;
    std::uint64_t final_count = 0;
    cuda_check(cudaMemcpy(&final_offset,
                          thrust::raw_pointer_cast(pair_offsets.data()) +
                              result.occupied_cells - 1,
                          sizeof(final_offset), cudaMemcpyDeviceToHost),
               "pair-work-offset control copy");
    cuda_check(cudaMemcpy(&final_count,
                          thrust::raw_pointer_cast(pair_counts.data()) +
                              result.occupied_cells - 1,
                          sizeof(final_count), cudaMemcpyDeviceToHost),
               "pair-work-count control copy");
    result.pair_work = final_offset + final_count;
  }
  cuda_check(cudaMemcpy(&result.fallback_flags,
                        thrust::raw_pointer_cast(device_status.data()),
                        sizeof(result.fallback_flags), cudaMemcpyDeviceToHost),
             "pair-work-status control copy");
  if (result.pair_work > options.max_pair_work)
    result.fallback_flags |= KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY;
  if (result.fallback_flags) {
    cuda_check(cudaDeviceSynchronize(), "pair-work-stage synchronize");
    result.broad_phase_ns = elapsed_ns(broad_begin, Clock::now());
    return result;
  }

  thrust::device_vector<std::uint64_t> candidates(result.pair_work);
  const std::uint32_t pair_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(ceil_div_u64(result.pair_work, threads), 65535));
  if (pair_blocks) {
    mark_pair_candidates_kernel<<<pair_blocks, threads>>>(
        thrust::raw_pointer_cast(device_records.data()),
        thrust::raw_pointer_cast(membership_records.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(cell_counts.data()),
        thrust::raw_pointer_cast(side_a_counts.data()),
        thrust::raw_pointer_cast(pair_offsets.data()), result.occupied_cells,
        result.pair_work, config, thrust::raw_pointer_cast(candidates.data()));
    cuda_check(cudaGetLastError(), "mark-pair-candidates launch");
  }
  auto compact_end = thrust::remove(thrust::device, candidates.begin(),
                                    candidates.end(), std::uint64_t{0});
  cuda_check(cudaDeviceSynchronize(), "broad-phase synchronize");
  const std::uint64_t raw_candidates = compact_end - candidates.begin();
  if (raw_candidates > options.max_candidates)
    result.fallback_flags |= KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_CAPACITY;
  result.broad_phase_ns = elapsed_ns(broad_begin, Clock::now());
  if (result.fallback_flags) return result;

  const auto sort_begin = Clock::now();
  auto pair_end = candidates.begin() + static_cast<std::ptrdiff_t>(raw_candidates);
  thrust::sort(thrust::device, candidates.begin(), pair_end);
  pair_end = thrust::unique(thrust::device, candidates.begin(), pair_end);
  const std::size_t unique_count = pair_end - candidates.begin();
  cuda_check(cudaDeviceSynchronize(), "candidate sort/unique synchronize");
  result.sort_unique_ns = elapsed_ns(sort_begin, Clock::now());

  const auto d2h_begin = Clock::now();
  result.pairs.resize(unique_count);
  if (unique_count) {
    cuda_check(cudaMemcpy(result.pairs.data(),
                          thrust::raw_pointer_cast(candidates.data()),
                          unique_count * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost),
               "candidate D2H copy");
  }
  result.d2h_ns = elapsed_ns(d2h_begin, Clock::now());
  return result;
}

M1PipelineResult run_m1_pipeline(
    const std::vector<PackedAabb> &records,
    const std::vector<PackedM1Endpoints> &metal1_edges,
    const std::vector<std::uint64_t> &contact_ids,
    const klayout_cuda_spatial_config_v1 &options, std::uint64_t distance) {
  M1PipelineResult result;
  const std::uint32_t contact_count =
      static_cast<std::uint32_t>(contact_ids.size());
  const std::uint32_t record_count = static_cast<std::uint32_t>(records.size());
  GridConfig config{options.cell_size, distance,
                    options.max_cells_per_record,
                    options.max_records_per_cell, 1u};
  constexpr std::uint32_t threads = 256;
  const std::uint32_t record_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          ceil_div_u64(static_cast<std::uint64_t>(records.size()), threads),
          65535));

  const auto setup_begin = Clock::now();
  cuda_check(cudaSetDevice(options.device), "cudaSetDevice");
  cuda_check(cudaFree(nullptr), "CUDA context initialization");
  thrust::device_vector<PackedAabb> device_records(records.size());
  thrust::device_vector<PackedM1Endpoints> device_metal1_edges(
      metal1_edges.size());
  thrust::device_vector<std::uint64_t> device_contact_ids(contact_ids.size());
  thrust::device_vector<std::uint32_t> device_status(1, 0);
  result.setup_ns = elapsed_ns(setup_begin, Clock::now());

  const auto h2d_begin = Clock::now();
  cuda_check(cudaMemcpy(thrust::raw_pointer_cast(device_records.data()),
                        records.data(), records.size() * sizeof(PackedAabb),
                        cudaMemcpyHostToDevice),
             "M1 record H2D copy");
  if (!metal1_edges.empty()) {
    cuda_check(
        cudaMemcpy(thrust::raw_pointer_cast(device_metal1_edges.data()),
                   metal1_edges.data(),
                   metal1_edges.size() * sizeof(PackedM1Endpoints),
                   cudaMemcpyHostToDevice),
        "M1 endpoint H2D copy");
  }
  cuda_check(cudaMemcpy(thrust::raw_pointer_cast(device_contact_ids.data()),
                        contact_ids.data(),
                        contact_ids.size() * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice),
             "M1 contact-ID H2D copy");
  result.h2d_ns = elapsed_ns(h2d_begin, Clock::now());

  const auto broad_begin = Clock::now();
  thrust::device_vector<std::uint32_t> counts(record_count);
  thrust::device_vector<std::uint64_t> offsets(record_count);
  count_memberships_kernel<<<record_blocks, threads>>>(
      thrust::raw_pointer_cast(device_records.data()), record_count, config,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(device_status.data()));
  cuda_check(cudaGetLastError(), "M1 count-memberships launch");
  thrust::exclusive_scan(thrust::device, counts.begin(), counts.end(),
                         offsets.begin(), std::uint64_t{0});
  std::uint64_t last_offset = 0;
  std::uint32_t last_count = 0;
  cuda_check(cudaMemcpy(&last_offset,
                        thrust::raw_pointer_cast(offsets.data()) +
                            record_count - 1,
                        sizeof(last_offset), cudaMemcpyDeviceToHost),
             "M1 membership-offset control copy");
  cuda_check(cudaMemcpy(&last_count,
                        thrust::raw_pointer_cast(counts.data()) +
                            record_count - 1,
                        sizeof(last_count), cudaMemcpyDeviceToHost),
             "M1 membership-count control copy");
  result.memberships = last_offset + last_count;
  cuda_check(cudaMemcpy(&result.fallback_flags,
                        thrust::raw_pointer_cast(device_status.data()),
                        sizeof(result.fallback_flags), cudaMemcpyDeviceToHost),
             "M1 membership-status control copy");
  if (result.memberships > options.max_memberships) {
    result.fallback_flags |=
        KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY;
  }
  if (result.fallback_flags) {
    cuda_check(cudaDeviceSynchronize(), "M1 membership-stage synchronize");
    result.broad_phase_ns = elapsed_ns(broad_begin, Clock::now());
    return result;
  }

  thrust::device_vector<CellKey> membership_keys(result.memberships);
  thrust::device_vector<std::uint32_t> membership_records(result.memberships);
  fill_memberships_kernel<<<record_blocks, threads>>>(
      thrust::raw_pointer_cast(device_records.data()), record_count, config,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(membership_keys.data()),
      thrust::raw_pointer_cast(membership_records.data()));
  cuda_check(cudaGetLastError(), "M1 fill-memberships launch");
  thrust::sort_by_key(thrust::device, membership_keys.begin(),
                      membership_keys.end(), membership_records.begin(),
                      CellKeyLess{});

  thrust::device_vector<CellKey> unique_cell_keys(result.memberships);
  thrust::device_vector<std::uint32_t> cell_counts(result.memberships);
  const auto reduced_end = thrust::reduce_by_key(
      thrust::device, membership_keys.begin(), membership_keys.end(),
      thrust::make_constant_iterator<std::uint32_t>(1), unique_cell_keys.begin(),
      cell_counts.begin(), CellKeyEqual{});
  result.occupied_cells =
      static_cast<std::uint64_t>(reduced_end.first - unique_cell_keys.begin());
  cell_counts.resize(result.occupied_cells);
  thrust::device_vector<std::uint64_t> cell_offsets(result.occupied_cells);
  thrust::exclusive_scan(thrust::device, cell_counts.begin(), cell_counts.end(),
                         cell_offsets.begin(), std::uint64_t{0});

  thrust::device_vector<std::uint64_t> pair_counts(result.occupied_cells);
  thrust::device_vector<std::uint64_t> pair_offsets(result.occupied_cells);
  thrust::device_vector<std::uint32_t> side_a_counts(result.occupied_cells);
  const std::uint32_t cell_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          ceil_div_u64(result.occupied_cells, threads), 65535));
  if (cell_blocks) {
    count_pair_work_kernel<<<cell_blocks, threads>>>(
        thrust::raw_pointer_cast(device_records.data()),
        thrust::raw_pointer_cast(membership_records.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(cell_counts.data()), result.occupied_cells,
        config, thrust::raw_pointer_cast(pair_counts.data()),
        thrust::raw_pointer_cast(side_a_counts.data()),
        thrust::raw_pointer_cast(device_status.data()));
    cuda_check(cudaGetLastError(), "M1 count-pair-work launch");
  }
  thrust::exclusive_scan(thrust::device, pair_counts.begin(), pair_counts.end(),
                         pair_offsets.begin(), std::uint64_t{0});
  if (result.occupied_cells) {
    std::uint64_t final_offset = 0;
    std::uint64_t final_count = 0;
    cuda_check(cudaMemcpy(&final_offset,
                          thrust::raw_pointer_cast(pair_offsets.data()) +
                              result.occupied_cells - 1,
                          sizeof(final_offset), cudaMemcpyDeviceToHost),
               "M1 pair-work-offset control copy");
    cuda_check(cudaMemcpy(&final_count,
                          thrust::raw_pointer_cast(pair_counts.data()) +
                              result.occupied_cells - 1,
                          sizeof(final_count), cudaMemcpyDeviceToHost),
               "M1 pair-work-count control copy");
    result.pair_work = final_offset + final_count;
  }
  cuda_check(cudaMemcpy(&result.fallback_flags,
                        thrust::raw_pointer_cast(device_status.data()),
                        sizeof(result.fallback_flags), cudaMemcpyDeviceToHost),
             "M1 pair-work-status control copy");
  if (result.pair_work > options.max_pair_work) {
    result.fallback_flags |=
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY;
  }
  if (result.fallback_flags) {
    cuda_check(cudaDeviceSynchronize(), "M1 pair-work-stage synchronize");
    result.broad_phase_ns = elapsed_ns(broad_begin, Clock::now());
    return result;
  }

  thrust::device_vector<std::uint64_t> candidates(result.pair_work);
  const std::uint32_t pair_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          ceil_div_u64(result.pair_work, threads), 65535));
  if (pair_blocks) {
    mark_pair_candidates_kernel<<<pair_blocks, threads>>>(
        thrust::raw_pointer_cast(device_records.data()),
        thrust::raw_pointer_cast(membership_records.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(cell_counts.data()),
        thrust::raw_pointer_cast(side_a_counts.data()),
        thrust::raw_pointer_cast(pair_offsets.data()), result.occupied_cells,
        result.pair_work, config, thrust::raw_pointer_cast(candidates.data()));
    cuda_check(cudaGetLastError(), "M1 mark-pair-candidates launch");
  }
  auto compact_end = thrust::remove(thrust::device, candidates.begin(),
                                    candidates.end(), std::uint64_t{0});
  const std::uint64_t raw_candidates = compact_end - candidates.begin();
  if (raw_candidates > options.max_candidates) {
    result.fallback_flags |= KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_CAPACITY;
  }
  if (result.fallback_flags) {
    cuda_check(cudaDeviceSynchronize(), "M1 candidate-stage synchronize");
    result.broad_phase_ns = elapsed_ns(broad_begin, Clock::now());
    return result;
  }

  auto candidate_end =
      candidates.begin() + static_cast<std::ptrdiff_t>(raw_candidates);
  thrust::sort(thrust::device, candidates.begin(), candidate_end);
  candidate_end =
      thrust::unique(thrust::device, candidates.begin(), candidate_end);
  result.broad_candidates =
      static_cast<std::uint64_t>(candidate_end - candidates.begin());
  cuda_check(cudaDeviceSynchronize(), "M1 broad-phase synchronize");
  result.broad_phase_ns = elapsed_ns(broad_begin, Clock::now());

  const auto classify_begin = Clock::now();
  thrust::device_vector<std::uint32_t> side_masks(contact_count, 0);
  thrust::device_vector<std::uint32_t> uncertain(contact_count, 0);
  thrust::device_vector<unsigned long long> counters(kM1CounterCount, 0);
  thrust::device_vector<klayout_cuda_spatial_m1_survivor_v1> survivors(
      contact_count);
  const std::uint32_t candidate_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          ceil_div_u64(result.broad_candidates, threads), 65535));
  if (candidate_blocks) {
    analyze_m1_candidates_kernel<<<candidate_blocks, threads>>>(
        thrust::raw_pointer_cast(device_records.data()),
        thrust::raw_pointer_cast(device_metal1_edges.data()), contact_count,
        record_count, thrust::raw_pointer_cast(candidates.data()),
        result.broad_candidates, distance,
        thrust::raw_pointer_cast(side_masks.data()),
        thrust::raw_pointer_cast(uncertain.data()),
        thrust::raw_pointer_cast(counters.data()),
        thrust::raw_pointer_cast(device_status.data()));
    cuda_check(cudaGetLastError(), "M1 exact-candidate launch");
  }
  cuda_check(cudaMemcpy(&result.fallback_flags,
                        thrust::raw_pointer_cast(device_status.data()),
                        sizeof(result.fallback_flags), cudaMemcpyDeviceToHost),
             "M1 exact-candidate invariant-status copy");
  if (result.fallback_flags) {
    result.classify_ns = elapsed_ns(classify_begin, Clock::now());
    return result;
  }

  const std::uint32_t contact_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(ceil_div_u64(contact_count, threads), 65535));
  classify_m1_contacts_kernel<<<contact_blocks, threads>>>(
      thrust::raw_pointer_cast(device_contact_ids.data()),
      thrust::raw_pointer_cast(side_masks.data()),
      thrust::raw_pointer_cast(uncertain.data()), contact_count,
      thrust::raw_pointer_cast(survivors.data()),
      thrust::raw_pointer_cast(counters.data()));
  cuda_check(cudaGetLastError(), "M1 classify-contacts launch");
  auto survivor_end = thrust::remove_if(
      thrust::device, survivors.begin(), survivors.end(), M1SurvivorIsCulled{});
  thrust::sort(thrust::device, survivors.begin(), survivor_end,
               M1SurvivorLess{});
  cuda_check(cudaDeviceSynchronize(), "M1 classify synchronize");

  unsigned long long host_counters[kM1CounterCount] = {};
  cuda_check(cudaMemcpy(host_counters,
                        thrust::raw_pointer_cast(counters.data()),
                        sizeof(host_counters), cudaMemcpyDeviceToHost),
             "M1 counter D2H control copy");
  result.full_side_hits = host_counters[kM1FullSideHits];
  result.partial_candidates = host_counters[kM1PartialCandidates];
  result.non_manhattan_candidates =
      host_counters[kM1NonManhattanCandidates];
  result.uncertain_contacts = host_counters[kM1UncertainContacts];
  result.disallowed_contacts = host_counters[kM1DisallowedContacts];
  result.classify_ns = elapsed_ns(classify_begin, Clock::now());

  const auto d2h_begin = Clock::now();
  const std::size_t survivor_count = survivor_end - survivors.begin();
  result.survivors.resize(survivor_count);
  if (survivor_count) {
    cuda_check(
        cudaMemcpy(result.survivors.data(),
                   thrust::raw_pointer_cast(survivors.data()),
                   survivor_count *
                       sizeof(klayout_cuda_spatial_m1_survivor_v1),
                   cudaMemcpyDeviceToHost),
        "M1 survivor D2H copy");
  }
  result.d2h_ns = elapsed_ns(d2h_begin, Clock::now());

  // Shielding in the CPU implementation discards complete edge pairs; it
  // cannot add a side.  Therefore the shielded mask is a subset of this
  // unshielded mask.  The accepted family (0, singleton, 0x5, 0xa) is closed
  // under subsets, making an accepted complete contact a conservative cull.
  if (result.uncertain_contacts) {
    result.disposition = KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN;
  } else if (result.disallowed_contacts) {
    result.disposition = KLAYOUT_CUDA_SPATIAL_M1_DISALLOWED;
  } else {
    result.disposition = KLAYOUT_CUDA_SPATIAL_M1_COMPLETE;
  }
  return result;
}

bool valid_config(const klayout_cuda_spatial_config_v1 *config) {
  if (!config ||
      config->abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      config->struct_size < sizeof(*config)) {
    return false;
  }

  const bool pair_work_sum_cannot_overflow =
      config->max_records_per_cell <= 1 ||
      config->max_memberships <=
          UINT64_MAX / (config->max_records_per_cell - 1);
  return config->device >= 0 && config->cell_size != 0 &&
         config->reserved0 == 0 &&
         config->cell_size <= static_cast<std::uint64_t>(INT64_MAX) &&
         config->max_cells_per_record != 0 &&
         config->max_records_per_cell != 0 && config->max_memberships != 0 &&
         config->max_pair_work != 0 && config->max_candidates != 0 &&
         pair_work_sum_cannot_overflow;
}

std::mutex &pipeline_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool valid_request(const klayout_cuda_spatial_request_v1 &request,
                   bool bipartite) {
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size < sizeof(request) || !valid_config(request.config) ||
      request.subject_count == 0 ||
      !request.subjects || request.enlargement < 0 ||
      request.subject_count > UINT32_MAX) {
    return false;
  }

  if (bipartite) {
    return request.intruder_count != 0 && request.intruders &&
           request.intruder_count <= UINT32_MAX &&
           request.subject_count + request.intruder_count <= UINT32_MAX;
  }

  return request.intruder_count == 0 && request.intruders == nullptr;
}

bool append_records(std::vector<PackedAabb> &records,
                    const klayout_cuda_spatial_aabb_v1 *boxes,
                    std::uint64_t count, std::uint32_t first_id,
                    std::uint32_t side, std::int64_t enlargement) {
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto &box = boxes[i];
    if (box.left > box.right || box.bottom > box.top ||
        box.left < INT64_MIN + enlargement ||
        box.bottom < INT64_MIN + enlargement ||
        box.right > INT64_MAX - enlargement ||
        box.top > INT64_MAX - enlargement) {
      return false;
    }
    records.push_back(PackedAabb{box.left, box.bottom, box.right, box.top,
                                 first_id + static_cast<std::uint32_t>(i),
                                 side, 0, 0});
  }
  return true;
}

int run_request(const klayout_cuda_spatial_request_v1 *request,
                klayout_cuda_spatial_result_v1 *result, bool bipartite) {
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;

  if (!request || !valid_request(*request, bipartite)) {
    result->fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    set_message(result, bipartite
                            ? "unsupported or malformed bipartite request"
                            : "unsupported or malformed self request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }

  const auto total_begin = Clock::now();
  try {
    // The first PoC deliberately serializes both entry points.  It avoids
    // accidental interleaving on CUDA's legacy default stream; a broker with
    // persistent buffers is the intended production replacement.
    std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex());
    std::vector<PackedAabb> records;
    records.reserve(static_cast<std::size_t>(
        request->subject_count +
        (bipartite ? request->intruder_count : std::uint64_t{0})));
    const std::int64_t enlargement = request->enlargement;
    if (!append_records(records, request->subjects, request->subject_count, 1,
                        0, enlargement) ||
        (bipartite &&
         !append_records(
             records, request->intruders, request->intruder_count,
             static_cast<std::uint32_t>(request->subject_count) + 1, 1,
             enlargement))) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW;
      set_message(result, "non-normalized AABB or coordinate overflow risk");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    PipelineResult pipeline = run_pipeline(
        records, *request->config,
        static_cast<std::uint64_t>(request->enlargement), bipartite);
    result->fallback_flags = pipeline.fallback_flags;
    result->membership_count = pipeline.memberships;
    result->occupied_cell_count = pipeline.occupied_cells;
    result->pair_work_count = pipeline.pair_work;
    result->setup_ns = pipeline.setup_ns;
    result->h2d_ns = pipeline.h2d_ns;
    result->broad_phase_ns = pipeline.broad_phase_ns;
    result->sort_unique_ns = pipeline.sort_unique_ns;
    result->d2h_ns = pipeline.d2h_ns;
    if (pipeline.fallback_flags) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      set_message(result, "configured fail-closed capacity was exceeded");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    if (!pipeline.pairs.empty()) {
      std::uint64_t *pairs = new std::uint64_t[pipeline.pairs.size()];
      std::copy(pipeline.pairs.begin(), pipeline.pairs.end(), pairs);
      result->pair_keys = pairs;
      result->pair_count = pipeline.pairs.size();
    }
    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return KLAYOUT_CUDA_SPATIAL_OK;
  } catch (const std::exception &ex) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    set_message(result, ex.what());
  } catch (...) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    set_message(result, "unknown CUDA spatial backend exception");
  }
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return KLAYOUT_CUDA_SPATIAL_ERROR;
}

bool valid_m1_request(const klayout_cuda_spatial_m1_request_v1 &request) {
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size < sizeof(request) ||
      request.opcode !=
          KLAYOUT_CUDA_SPATIAL_M1_ENCLOSED_PROJECTION_ONE_OR_OPPOSITE ||
      request.reserved0 != 0 || !valid_config(request.config) ||
      request.contact_count == 0 || !request.contacts ||
      request.distance <= 0 || request.contact_count > UINT32_MAX ||
      request.metal1_edge_count > UINT32_MAX ||
      request.contact_count >
          static_cast<std::uint64_t>(UINT32_MAX) -
              request.metal1_edge_count) {
    return false;
  }
  return request.metal1_edge_count == 0
             ? request.metal1_edges == nullptr
             : request.metal1_edges != nullptr;
}

bool coordinate_has_margin(std::int64_t value, std::int64_t distance) {
  return value >= INT64_MIN + distance && value <= INT64_MAX - distance;
}

int run_m1_request(const klayout_cuda_spatial_m1_request_v1 *request,
                   klayout_cuda_spatial_m1_result_v1 *result) {
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->disposition = KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN;

  if (!request || !valid_m1_request(*request)) {
    result->fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    set_message(result, "unsupported or malformed M1 enclosure request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }

  result->contact_count = request->contact_count;
  result->metal1_edge_count = request->metal1_edge_count;
  const auto total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex());
    const std::uint32_t contact_count =
        static_cast<std::uint32_t>(request->contact_count);
    std::vector<PackedAabb> records;
    records.reserve(static_cast<std::size_t>(
        request->contact_count + request->metal1_edge_count));
    std::vector<PackedM1Endpoints> metal1_edges;
    metal1_edges.reserve(
        static_cast<std::size_t>(request->metal1_edge_count));
    std::vector<std::uint64_t> contact_ids;
    contact_ids.reserve(static_cast<std::size_t>(request->contact_count));

    bool geometry_good = true;
    for (std::uint64_t index = 0; index < request->contact_count; ++index) {
      const auto &contact = request->contacts[index];
      if (contact.reserved0 != 0 || contact.contact_id == 0 ||
          contact.left >= contact.right || contact.bottom >= contact.top ||
          !coordinate_has_margin(contact.left, request->distance) ||
          !coordinate_has_margin(contact.bottom, request->distance) ||
          !coordinate_has_margin(contact.right, request->distance) ||
          !coordinate_has_margin(contact.top, request->distance)) {
        geometry_good = false;
        break;
      }
      records.push_back(PackedAabb{
          contact.left, contact.bottom, contact.right, contact.top,
          static_cast<std::uint32_t>(index) + 1, 0, contact.context_id, 0});
      contact_ids.push_back(contact.contact_id);
    }

    for (std::uint64_t index = 0;
         geometry_good && index < request->metal1_edge_count; ++index) {
      const auto &edge = request->metal1_edges[index];
      if (edge.reserved0 != 0 ||
          !coordinate_has_margin(edge.x1, request->distance) ||
          !coordinate_has_margin(edge.y1, request->distance) ||
          !coordinate_has_margin(edge.x2, request->distance) ||
          !coordinate_has_margin(edge.y2, request->distance)) {
        geometry_good = false;
        break;
      }
      const std::int64_t left = std::min(edge.x1, edge.x2);
      const std::int64_t right = std::max(edge.x1, edge.x2);
      const std::int64_t bottom = std::min(edge.y1, edge.y2);
      const std::int64_t top = std::max(edge.y1, edge.y2);
      records.push_back(PackedAabb{
          left, bottom, right, top,
          contact_count + static_cast<std::uint32_t>(index) + 1, 1,
          edge.context_id, 0});
      metal1_edges.push_back(
          PackedM1Endpoints{edge.x1, edge.y1, edge.x2, edge.y2});
    }

    if (!geometry_good) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW;
      set_message(result,
                  "non-rectangular contact, reserved field, or coordinate "
                  "overflow risk");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    std::vector<std::uint64_t> sorted_ids(contact_ids);
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) !=
        sorted_ids.end()) {
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
      set_message(result, "M1 contact IDs must be nonzero and unique");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
    }

    M1PipelineResult pipeline = run_m1_pipeline(
        records, metal1_edges, contact_ids, *request->config,
        static_cast<std::uint64_t>(request->distance));
    result->fallback_flags = pipeline.fallback_flags;
    result->disposition = pipeline.disposition;
    result->membership_count = pipeline.memberships;
    result->occupied_cell_count = pipeline.occupied_cells;
    result->pair_work_count = pipeline.pair_work;
    result->broad_candidate_count = pipeline.broad_candidates;
    result->full_side_hit_count = pipeline.full_side_hits;
    result->partial_candidate_count = pipeline.partial_candidates;
    result->non_manhattan_candidate_count =
        pipeline.non_manhattan_candidates;
    result->uncertain_contact_count = pipeline.uncertain_contacts;
    result->disallowed_contact_count = pipeline.disallowed_contacts;
    result->setup_ns = pipeline.setup_ns;
    result->h2d_ns = pipeline.h2d_ns;
    result->broad_phase_ns = pipeline.broad_phase_ns;
    result->classify_ns = pipeline.classify_ns;
    result->d2h_ns = pipeline.d2h_ns;
    if (pipeline.fallback_flags) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->disposition = KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN;
      set_message(
          result,
          (pipeline.fallback_flags &
           KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT)
              ? "M1 certificate internal invariant failed"
              : "configured M1 certificate capacity was exceeded");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    if (!pipeline.survivors.empty()) {
      auto *survivors = new klayout_cuda_spatial_m1_survivor_v1[
          pipeline.survivors.size()];
      std::copy(pipeline.survivors.begin(), pipeline.survivors.end(),
                survivors);
      result->survivors = survivors;
      result->survivor_count = pipeline.survivors.size();
    }
    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    if (pipeline.disposition == KLAYOUT_CUDA_SPATIAL_M1_COMPLETE) {
      set_message(result, "complete empty M1 enclosure certificate");
    } else if (pipeline.disposition == KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN) {
      set_message(result, "M1 enclosure scan has uncertain survivors");
    } else {
      set_message(result, "M1 enclosure scan has disallowed-mask survivors");
    }
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return KLAYOUT_CUDA_SPATIAL_OK;
  } catch (const std::exception &ex) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    set_message(result, ex.what());
  } catch (...) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    set_message(result, "unknown CUDA M1 certificate exception");
  }
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return KLAYOUT_CUDA_SPATIAL_ERROR;
}

}  // namespace

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT std::uint32_t
klayout_cuda_spatial_abi_version(void) {
  return KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT void
klayout_cuda_spatial_release_result_v1(klayout_cuda_spatial_result_v1 *result) {
  if (!result) return;
  delete[] result->pair_keys;
  result->pair_keys = nullptr;
  result->pair_count = 0;
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_bipartite_v1(
    const klayout_cuda_spatial_request_v1 *request,
    klayout_cuda_spatial_result_v1 *result) {
  return run_request(request, result, true);
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_self_v1(
    const klayout_cuda_spatial_request_v1 *request,
    klayout_cuda_spatial_result_v1 *result) {
  return run_request(request, result, false);
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m1_enclosure_v1(
    const klayout_cuda_spatial_m1_request_v1 *request,
    klayout_cuda_spatial_m1_result_v1 *result) {
  return run_m1_request(request, result);
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT void
klayout_cuda_spatial_release_m1_result_v1(
    klayout_cuda_spatial_m1_result_v1 *result) {
  if (!result) return;
  delete[] result->survivors;
  result->survivors = nullptr;
  result->survivor_count = 0;
}
