/*
 * Optional CUDA bipartite AABB broad phase for KLayout.
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
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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
  std::uint32_t side;
  std::uint32_t reserved;
};

struct CellKeyLess {
  __host__ __device__ bool operator()(const CellKey &a, const CellKey &b) const {
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    return a.side < b.side;
  }
};

struct CellKeyEqual {
  __host__ __device__ bool operator()(const CellKey &a, const CellKey &b) const {
    // Side participates in ordering so all subjects precede intruders, but it
    // deliberately does not split the spatial cell during reduce_by_key.
    return a.x == b.x && a.y == b.y;
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

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
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

__host__ __device__ std::uint64_t pair_key(std::uint32_t a,
                                           std::uint32_t b) {
  return (static_cast<std::uint64_t>(a) << 32) | b;
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
        keys[output] = CellKey{x, y, records[index].side, 0};
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
    const std::uint64_t offset = cell_offsets[cell];
    std::uint32_t side_a = 0;
    while (side_a < count && records[record_indices[offset + side_a]].side == 0)
      ++side_a;
    side_a_counts[cell] = side_a;
    pair_work_counts[cell] =
        static_cast<std::uint64_t>(side_a) * (count - side_a);
  }
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
    const std::uint32_t side_a = side_a_counts[cell];
    const std::uint32_t side_b = count - side_a;
    const std::uint32_t ai = static_cast<std::uint32_t>(local / side_b);
    const std::uint32_t bi = side_a + static_cast<std::uint32_t>(local % side_b);
    const std::uint64_t offset = cell_offsets[cell];
    const PackedAabb a = records[record_indices[offset + ai]];
    const PackedAabb b = records[record_indices[offset + bi]];
    candidate_or_zero[work] = boxes_overlap_strict(a, b, config.enlargement)
                                  ? pair_key(a.id, b.id)
                                  : 0;
  }
}

void set_message(klayout_cuda_spatial_result_v1 *result,
                 const std::string &message) {
  std::snprintf(result->message, sizeof(result->message), "%s", message.c_str());
}

PipelineResult run_pipeline(const std::vector<PackedAabb> &records,
                            const klayout_cuda_spatial_config_v1 &options,
                            std::uint64_t enlargement) {
  PipelineResult result;
  GridConfig config{options.cell_size, enlargement,
                    options.max_cells_per_record,
                    options.max_records_per_cell};
  const std::uint32_t record_count = static_cast<std::uint32_t>(records.size());
  constexpr std::uint32_t threads = 256;
  const std::uint32_t record_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>((records.size() + threads - 1) / threads, 65535));

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
      std::min<std::uint64_t>((result.occupied_cells + threads - 1) / threads,
                              65535));
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
      std::min<std::uint64_t>((result.pair_work + threads - 1) / threads, 65535));
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

bool valid_config(const klayout_cuda_spatial_config_v1 &config) {
  return config.abi_version == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
         config.struct_size >= sizeof(config) && config.device >= 0 &&
         config.cell_size != 0 &&
         config.cell_size <= static_cast<std::uint64_t>(INT64_MAX) &&
         config.max_cells_per_record != 0 &&
         config.max_records_per_cell != 0 && config.max_memberships != 0 &&
         config.max_pair_work != 0 && config.max_candidates != 0;
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
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;

  if (!request || request->abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request->struct_size < sizeof(*request) || !request->config ||
      !valid_config(*request->config) || request->subject_count == 0 ||
      request->intruder_count == 0 || !request->subjects || !request->intruders ||
      request->enlargement < 0 ||
      static_cast<std::uint64_t>(request->enlargement) >
          static_cast<std::uint64_t>(INT64_MAX) ||
      request->subject_count > UINT32_MAX || request->intruder_count > UINT32_MAX ||
      request->subject_count + request->intruder_count > UINT32_MAX) {
    result->fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    set_message(result, "unsupported or malformed bipartite request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }

  const auto total_begin = Clock::now();
  try {
    // The first PoC deliberately serializes calls.  It avoids accidental
    // interleaving on CUDA's legacy default stream; a broker with persistent
    // buffers is the intended production replacement.
    static std::mutex pipeline_mutex;
    std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex);
    std::vector<PackedAabb> records;
    records.reserve(static_cast<std::size_t>(request->subject_count +
                                             request->intruder_count));
    const std::int64_t enlargement = request->enlargement;
    auto append = [&](const klayout_cuda_spatial_aabb_v1 *boxes,
                      std::uint64_t count, std::uint32_t first_id,
                      std::uint32_t side) -> bool {
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
    };
    if (!append(request->subjects, request->subject_count, 1, 0) ||
        !append(request->intruders, request->intruder_count,
                static_cast<std::uint32_t>(request->subject_count) + 1, 1)) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW;
      set_message(result, "non-normalized AABB or coordinate overflow risk");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    PipelineResult pipeline = run_pipeline(
        records, *request->config,
        static_cast<std::uint64_t>(request->enlargement));
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
