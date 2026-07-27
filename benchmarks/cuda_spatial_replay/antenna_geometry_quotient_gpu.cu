#include "antenna_geometry_quotient_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>

namespace {

namespace aq = klayout_cuda::antenna_geometry_quotient;
namespace ac = klayout_cuda::antenna_connectivity;

constexpr std::uint32_t kThreads = 256;

enum DeviceFlag : std::uint32_t
{
  kMalformed = 1u << 0,
  kMissingOwner = 1u << 1,
  kRelationOverflow = 1u << 2,
  kInternalInvariant = 1u << 3
};

class CudaFailure : public std::runtime_error
{
public:
  explicit CudaFailure(const char *operation)
      : std::runtime_error(operation)
  {
  }
};

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) throw CudaFailure(operation);
}

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 0;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          (count + kThreads - 1) / kThreads, 65535));
}

bool valid_config(const aq::DeviceConfig &config)
{
  const aq::Config &quotient = config.quotient;
  if (!quotient.domain_count ||
      quotient.domain_count > ac::kMaximumDomains ||
      !quotient.owner_count ||
      quotient.owner_count > quotient.limits.max_owners ||
      quotient.owner_begin >
          UINT32_MAX - (quotient.owner_count - 1) ||
      !config.device_limits.max_device_bytes ||
      config.device_limits.min_device_free_after_bytes >=
          config.device_limits.max_device_bytes) {
    return false;
  }
  const std::uint64_t domain_mask =
      quotient.domain_count == ac::kMaximumDomains
          ? UINT64_MAX
          : (UINT64_C(1) << quotient.domain_count) - 1;
  for (std::uint32_t domain = 0;
       domain < ac::kMaximumDomains; ++domain) {
    if (quotient.relation_rows[domain] & ~domain_mask) {
      return false;
    }
    if (domain >= quotient.domain_count &&
        quotient.relation_rows[domain]) {
      return false;
    }
    for (std::uint32_t other = 0;
         other < quotient.domain_count; ++other) {
      const bool forward =
          (quotient.relation_rows[domain] &
           (UINT64_C(1) << other)) != 0;
      const bool reverse =
          (quotient.relation_rows[other] &
           (UINT64_C(1) << domain)) != 0;
      if (forward != reverse) return false;
    }
  }
  return true;
}

bool byte_product(
    std::uint64_t count, std::uint64_t size,
    std::uint64_t *bytes)
{
  if (size && count > UINT64_MAX / size) return false;
  *bytes = count * size;
  return true;
}

bool bytes_admitted(
    const aq::DeviceConfig &config, std::uint64_t bytes)
{
  if (bytes > config.device_limits.max_device_bytes) {
    return false;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cuda_require(
      cudaMemGetInfo(&free_bytes, &total_bytes),
      "geometry quotient cudaMemGetInfo");
  (void) total_bytes;
  const std::uint64_t free_u64 = free_bytes;
  return bytes <= free_u64 &&
         config.device_limits.min_device_free_after_bytes <=
             free_u64 - bytes;
}

template <class T>
bool vector_admitted(
    const aq::DeviceConfig &config, std::uint64_t count)
{
  std::uint64_t bytes = 0;
  return byte_product(count, sizeof(T), &bytes) &&
         bytes_admitted(config, bytes);
}

bool singleton_sort_admitted(
    const aq::DeviceConfig &config, std::uint64_t count)
{
  std::uint64_t bytes = 0;
  if (!byte_product(
          count, sizeof(std::uint32_t), &bytes) ||
      bytes >
          UINT64_MAX -
              config.device_limits.sort_scratch_fixed_guard_bytes) {
    return false;
  }
  return bytes_admitted(
      config,
      bytes +
          config.device_limits.sort_scratch_fixed_guard_bytes);
}

template <class T>
void release_vector(thrust::device_vector<T> *values)
{
  thrust::device_vector<T>().swap(*values);
}

struct SaturatingAddU64
{
  __host__ __device__ std::uint64_t operator()(
      std::uint64_t first, std::uint64_t second) const
  {
    return first > UINT64_MAX - second
               ? UINT64_MAX
               : first + second;
  }
};

__global__ void validate_and_count_kernel(
    const ac::RectI64 *rectangles, std::uint64_t rectangle_count,
    std::uint32_t owner_begin, std::uint32_t owner_count,
    std::uint32_t domain_count, std::uint32_t *owner_rectangles,
    std::uint32_t *owner_domains, std::uint32_t *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  const std::uint64_t owner_end =
      static_cast<std::uint64_t>(owner_begin) + owner_count;
  for (std::uint64_t index = first;
       index < rectangle_count; index += stride) {
    const ac::RectI64 rectangle = rectangles[index];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top ||
        rectangle.owner < owner_begin ||
        rectangle.owner >= owner_end ||
        rectangle.domain >= domain_count) {
      atomicOr(status, std::uint32_t(kMalformed));
      continue;
    }
    const std::uint32_t local = rectangle.owner - owner_begin;
    atomicAdd(owner_rectangles + local, 1u);
    const std::uint32_t prior = atomicCAS(
        owner_domains + local, UINT32_MAX, rectangle.domain);
    if (prior != UINT32_MAX && prior != rectangle.domain) {
      atomicOr(status, std::uint32_t(kMalformed));
    }
  }
}

__global__ void validate_owners_kernel(
    const std::uint32_t *owner_rectangles,
    const std::uint32_t *owner_domains,
    std::uint32_t owner_count, std::uint32_t *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t owner = first;
       owner < owner_count; owner += stride) {
    if (!owner_rectangles[owner] ||
        owner_domains[owner] == UINT32_MAX) {
      atomicOr(status, std::uint32_t(kMissingOwner));
    }
  }
}

struct IsSingletonIndex
{
  const ac::RectI64 *rectangles;
  const std::uint32_t *owner_rectangles;
  std::uint32_t owner_begin;

  __host__ __device__ bool operator()(std::uint32_t index) const
  {
    const ac::RectI64 rectangle = rectangles[index];
    return owner_rectangles[rectangle.owner - owner_begin] == 1;
  }
};

struct IsExceptionRectangle
{
  const std::uint32_t *owner_rectangles;
  std::uint32_t owner_begin;

  __host__ __device__ bool operator()(
      const ac::RectI64 &rectangle) const
  {
    return owner_rectangles[rectangle.owner - owner_begin] > 1;
  }
};

struct GeometryIndexLess
{
  const ac::RectI64 *rectangles;

  __host__ __device__ bool operator()(
      std::uint32_t first_index,
      std::uint32_t second_index) const
  {
    const ac::RectI64 first = rectangles[first_index];
    const ac::RectI64 second = rectangles[second_index];
    if (first.domain != second.domain) {
      return first.domain < second.domain;
    }
    if (first.left != second.left) {
      return first.left < second.left;
    }
    if (first.bottom != second.bottom) {
      return first.bottom < second.bottom;
    }
    if (first.right != second.right) {
      return first.right < second.right;
    }
    if (first.top != second.top) {
      return first.top < second.top;
    }
    return first.owner < second.owner;
  }
};

__host__ __device__ bool same_geometry(
    const ac::RectI64 &first, const ac::RectI64 &second)
{
  return first.domain == second.domain &&
         first.left == second.left &&
         first.bottom == second.bottom &&
         first.right == second.right &&
         first.top == second.top;
}

__global__ void mark_class_heads_kernel(
    const ac::RectI64 *rectangles,
    const std::uint32_t *singleton_indices,
    std::uint64_t singleton_count,
    std::uint64_t self_connected_domains,
    std::uint32_t *class_heads)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first;
       index < singleton_count; index += stride) {
    bool head = !index;
    if (index) {
      const ac::RectI64 rectangle =
          rectangles[singleton_indices[index]];
      const bool self_connected =
          (self_connected_domains &
           (UINT64_C(1) << rectangle.domain)) != 0;
      head =
          !self_connected ||
          !same_geometry(
              rectangles[singleton_indices[index - 1]],
              rectangle);
    }
    class_heads[index] = head ? 1u : 0u;
  }
}

__global__ void write_class_heads_kernel(
    const ac::RectI64 *rectangles,
    const std::uint32_t *singleton_indices,
    const std::uint32_t *class_heads,
    const std::uint32_t *class_ids,
    std::uint64_t singleton_count,
    std::uint64_t *class_begins,
    ac::RectI64 *work_rectangles)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first;
       index < singleton_count; index += stride) {
    if (!class_heads[index]) continue;
    const std::uint32_t class_id = class_ids[index] - 1;
    class_begins[class_id] = index;
    work_rectangles[class_id] =
        rectangles[singleton_indices[index]];
  }
}

__global__ void seed_class_members_kernel(
    const ac::RectI64 *rectangles,
    const std::uint32_t *singleton_indices,
    const std::uint32_t *class_ids,
    const std::uint64_t *class_begins,
    std::uint64_t singleton_count,
    std::uint32_t owner_begin,
    std::uint32_t *parent_seeds,
    std::uint32_t *owner_multiplicities)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first;
       index < singleton_count; index += stride) {
    const std::uint32_t class_id = class_ids[index] - 1;
    const std::uint64_t class_begin = class_begins[class_id];
    const std::uint32_t representative =
        rectangles[singleton_indices[class_begin]].owner;
    const std::uint32_t owner =
        rectangles[singleton_indices[index]].owner;
    parent_seeds[owner - owner_begin] = representative;
    owner_multiplicities[owner - owner_begin] = 0;
  }
}

__global__ void finish_classes_kernel(
    const ac::RectI64 *work_rectangles,
    const std::uint64_t *class_begins,
    std::uint64_t class_count,
    std::uint64_t singleton_count,
    std::uint32_t owner_begin,
    std::uint32_t *rectangle_multiplicities,
    std::uint32_t *owner_multiplicities,
    std::uint64_t *internal_weights,
    unsigned long long *relation_census,
    std::uint32_t *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t class_id = first;
       class_id < class_count; class_id += stride) {
    const std::uint64_t begin = class_begins[class_id];
    const std::uint64_t end =
        class_id + 1 < class_count
            ? class_begins[class_id + 1]
            : singleton_count;
    if (end <= begin || end - begin > UINT32_MAX) {
      atomicOr(status, std::uint32_t(kInternalInvariant));
      continue;
    }
    const std::uint32_t multiplicity =
        static_cast<std::uint32_t>(end - begin);
    const ac::RectI64 representative =
        work_rectangles[class_id];
    rectangle_multiplicities[class_id] = multiplicity;
    owner_multiplicities[
        representative.owner - owner_begin] = multiplicity;
    const std::uint64_t half = multiplicity / 2;
    const std::uint64_t other =
        multiplicity & 1
            ? multiplicity
            : static_cast<std::uint64_t>(multiplicity) - 1;
    const std::uint64_t weight = half * other;
    internal_weights[class_id] = weight;
    if (!weight) continue;
    const std::size_t slot =
        static_cast<std::size_t>(representative.domain) *
            ac::kMaximumDomains +
        representative.domain;
    const unsigned long long prior =
        atomicAdd(
            relation_census + slot,
            static_cast<unsigned long long>(weight));
    if (prior > ULLONG_MAX - weight) {
      atomicOr(status, std::uint32_t(kRelationOverflow));
    }
  }
}

std::uint32_t read_status(
    const thrust::device_vector<std::uint32_t> &status)
{
  std::uint32_t host = 0;
  cuda_require(
      cudaMemcpy(
          &host, thrust::raw_pointer_cast(status.data()),
          sizeof(host), cudaMemcpyDeviceToHost),
      "geometry quotient status D2H");
  return host;
}

}  // namespace

namespace klayout_cuda {
namespace antenna_geometry_quotient {

Status build_device(
    const DeviceConfig &config,
    const ac::RectI64 *device_rectangles,
    std::uint64_t rectangle_count,
    DeviceResult *output) noexcept
{
  if (!output) return Status::malformed_input;
  if (!valid_config(config)) {
    return Status::invalid_configuration;
  }
  if (!device_rectangles || !rectangle_count) {
    return Status::malformed_input;
  }
  if (rectangle_count >
          config.quotient.limits.max_input_rectangles ||
      rectangle_count > UINT32_MAX) {
    return Status::capacity_exceeded;
  }

  try {
    cuda_require(
        cudaSetDevice(config.device),
        "geometry quotient cudaSetDevice");
    DeviceResult result;
    result.census.input_rectangles = rectangle_count;
    result.census.owners = config.quotient.owner_count;
    const thrust::device_ptr<const ac::RectI64> rectangle_begin =
        thrust::device_pointer_cast(device_rectangles);
    const thrust::device_ptr<const ac::RectI64> rectangle_end =
        rectangle_begin + rectangle_count;

    if (!vector_admitted<std::uint32_t>(
            config, config.quotient.owner_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint32_t> owner_rectangle_counts(
        config.quotient.owner_count, 0);
    if (!vector_admitted<std::uint32_t>(
            config, config.quotient.owner_count)) {
      return Status::capacity_exceeded;
    }
    result.owner_domains.assign(
        config.quotient.owner_count, UINT32_MAX);
    thrust::device_vector<std::uint32_t> status(1, 0);

    validate_and_count_kernel<<<
        launch_blocks(rectangle_count), kThreads>>>(
        device_rectangles,
        rectangle_count, config.quotient.owner_begin,
        config.quotient.owner_count,
        config.quotient.domain_count,
        thrust::raw_pointer_cast(
            owner_rectangle_counts.data()),
        thrust::raw_pointer_cast(result.owner_domains.data()),
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "geometry quotient validation launch");
    validate_owners_kernel<<<
        launch_blocks(config.quotient.owner_count), kThreads>>>(
        thrust::raw_pointer_cast(
            owner_rectangle_counts.data()),
        thrust::raw_pointer_cast(result.owner_domains.data()),
        config.quotient.owner_count,
        thrust::raw_pointer_cast(status.data()));
    cuda_require(
        cudaGetLastError(),
        "geometry quotient owner validation launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "geometry quotient validation synchronize");
    const std::uint32_t validation_flags = read_status(status);
    if (validation_flags &
        (kMalformed | kMissingOwner | kInternalInvariant)) {
      return Status::malformed_input;
    }
    if (validation_flags & kRelationOverflow) {
      return Status::capacity_exceeded;
    }

    if (!vector_admitted<std::uint32_t>(
            config, rectangle_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint32_t>
        singleton_indices(rectangle_count);
    const auto singleton_end = thrust::copy_if(
        thrust::device,
        thrust::make_counting_iterator<std::uint32_t>(0),
        thrust::make_counting_iterator<std::uint32_t>(
            static_cast<std::uint32_t>(rectangle_count)),
        singleton_indices.begin(),
        IsSingletonIndex{
            device_rectangles,
            thrust::raw_pointer_cast(
                owner_rectangle_counts.data()),
            config.quotient.owner_begin});
    const std::uint64_t singleton_count =
        singleton_end - singleton_indices.begin();
    singleton_indices.resize(
        static_cast<std::size_t>(singleton_count));
    result.census.singleton_owners = singleton_count;
    result.census.exception_owners =
        config.quotient.owner_count - singleton_count;
    const std::uint64_t exception_count =
        rectangle_count - singleton_count;
    result.census.exception_rectangles = exception_count;
    if (singleton_count >
            config.quotient.limits.max_class_members ||
        exception_count >
            config.quotient.limits.max_exception_rectangles) {
      return Status::capacity_exceeded;
    }

    if (singleton_count) {
      if (!singleton_sort_admitted(config, singleton_count)) {
        return Status::capacity_exceeded;
      }
      thrust::sort(
          thrust::device, singleton_indices.begin(),
          singleton_indices.end(),
          GeometryIndexLess{
              device_rectangles});
    }

    thrust::device_vector<std::uint32_t>
        class_heads;
    thrust::device_vector<std::uint32_t>
        class_ids;
    std::uint64_t self_connected_domains = 0;
    for (std::uint32_t domain = 0;
         domain < config.quotient.domain_count; ++domain) {
      if (config.quotient.relation_rows[domain] &
          (UINT64_C(1) << domain)) {
        self_connected_domains |= UINT64_C(1) << domain;
      }
    }
    std::uint32_t class_count_u32 = 0;
    if (singleton_count) {
      if (!vector_admitted<std::uint32_t>(
              config, singleton_count)) {
        return Status::capacity_exceeded;
      }
      class_heads.resize(
          static_cast<std::size_t>(singleton_count));
      if (!vector_admitted<std::uint32_t>(
              config, singleton_count)) {
        return Status::capacity_exceeded;
      }
      class_ids.resize(
          static_cast<std::size_t>(singleton_count));
      mark_class_heads_kernel<<<
          launch_blocks(singleton_count), kThreads>>>(
          device_rectangles,
          thrust::raw_pointer_cast(singleton_indices.data()),
          singleton_count, self_connected_domains,
          thrust::raw_pointer_cast(class_heads.data()));
      cuda_require(
          cudaGetLastError(),
          "geometry quotient class-head launch");
      thrust::inclusive_scan(
          thrust::device, class_heads.begin(),
          class_heads.end(), class_ids.begin());
      cuda_require(
          cudaMemcpy(
              &class_count_u32,
              thrust::raw_pointer_cast(class_ids.data()) +
                  singleton_count - 1,
              sizeof(class_count_u32),
              cudaMemcpyDeviceToHost),
          "geometry quotient class count D2H");
    }
    const std::uint64_t class_count = class_count_u32;
    if (class_count >
        config.quotient.limits.max_classes) {
      return Status::capacity_exceeded;
    }
    result.census.geometry_classes = class_count;
    result.census.collapsed_rectangles =
        singleton_count - class_count;
    result.census.star_edges =
        result.census.collapsed_rectangles;
    if (result.census.star_edges >
        config.quotient.limits.max_star_edges) {
      return Status::capacity_exceeded;
    }
    const std::uint64_t work_count =
        class_count + exception_count;
    result.census.work_rectangles = work_count;
    if (work_count >
        config.quotient.limits.max_work_rectangles) {
      return Status::capacity_exceeded;
    }

    if (!vector_admitted<ac::RectI64>(config, work_count)) {
      return Status::capacity_exceeded;
    }
    result.work_rectangles.resize(
        static_cast<std::size_t>(work_count));
    if (!vector_admitted<std::uint32_t>(
            config, work_count)) {
      return Status::capacity_exceeded;
    }
    result.rectangle_multiplicities.assign(
        static_cast<std::size_t>(work_count), 1);
    if (!vector_admitted<std::uint64_t>(
            config, class_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        class_begins(class_count);
    if (singleton_count) {
      write_class_heads_kernel<<<
          launch_blocks(singleton_count), kThreads>>>(
          device_rectangles,
          thrust::raw_pointer_cast(singleton_indices.data()),
          thrust::raw_pointer_cast(class_heads.data()),
          thrust::raw_pointer_cast(class_ids.data()),
          singleton_count,
          thrust::raw_pointer_cast(class_begins.data()),
          thrust::raw_pointer_cast(
              result.work_rectangles.data()));
      cuda_require(
          cudaGetLastError(),
          "geometry quotient class materialization launch");
    }
    const auto exception_end = thrust::copy_if(
        thrust::device, rectangle_begin, rectangle_end,
        result.work_rectangles.begin() + class_count,
        IsExceptionRectangle{
            thrust::raw_pointer_cast(
                owner_rectangle_counts.data()),
            config.quotient.owner_begin});
    if (exception_end != result.work_rectangles.end()) {
      return Status::host_error;
    }

    if (!vector_admitted<std::uint32_t>(
            config, config.quotient.owner_count)) {
      return Status::capacity_exceeded;
    }
    result.parent_seeds.resize(config.quotient.owner_count);
    thrust::sequence(
        thrust::device, result.parent_seeds.begin(),
        result.parent_seeds.end(),
        config.quotient.owner_begin);
    if (!vector_admitted<std::uint32_t>(
            config, config.quotient.owner_count)) {
      return Status::capacity_exceeded;
    }
    result.owner_multiplicities.assign(
        config.quotient.owner_count, 1);
    if (singleton_count) {
      seed_class_members_kernel<<<
          launch_blocks(singleton_count), kThreads>>>(
          device_rectangles,
          thrust::raw_pointer_cast(singleton_indices.data()),
          thrust::raw_pointer_cast(class_ids.data()),
          thrust::raw_pointer_cast(class_begins.data()),
          singleton_count, config.quotient.owner_begin,
          thrust::raw_pointer_cast(result.parent_seeds.data()),
          thrust::raw_pointer_cast(
              result.owner_multiplicities.data()));
      cuda_require(
          cudaGetLastError(),
          "geometry quotient parent seeding launch");
    }

    if (!vector_admitted<std::uint64_t>(
            config, class_count)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<std::uint64_t>
        internal_weights(class_count);
    if (!vector_admitted<unsigned long long>(
            config, ac::kRelationSlots)) {
      return Status::capacity_exceeded;
    }
    thrust::device_vector<unsigned long long>
        relation_census(ac::kRelationSlots, 0);
    if (class_count) {
      finish_classes_kernel<<<
          launch_blocks(class_count), kThreads>>>(
          thrust::raw_pointer_cast(
              result.work_rectangles.data()),
          thrust::raw_pointer_cast(class_begins.data()),
          class_count, singleton_count,
          config.quotient.owner_begin,
          thrust::raw_pointer_cast(
              result.rectangle_multiplicities.data()),
          thrust::raw_pointer_cast(
              result.owner_multiplicities.data()),
          thrust::raw_pointer_cast(internal_weights.data()),
          thrust::raw_pointer_cast(relation_census.data()),
          thrust::raw_pointer_cast(status.data()));
      cuda_require(
          cudaGetLastError(),
          "geometry quotient class finalization launch");
    }
    cuda_require(
        cudaDeviceSynchronize(),
        "geometry quotient finalization synchronize");
    const std::uint32_t final_flags = read_status(status);
    if (final_flags & kRelationOverflow) {
      return Status::capacity_exceeded;
    }
    if (final_flags) return Status::host_error;
    const std::uint64_t weighted_internal_pairs =
        thrust::reduce(
            thrust::device, internal_weights.begin(),
            internal_weights.end(), UINT64_C(0),
            SaturatingAddU64());
    if (weighted_internal_pairs == UINT64_MAX ||
        weighted_internal_pairs >
            config.quotient.limits
                .max_weighted_internal_pairs) {
      return Status::capacity_exceeded;
    }
    result.census.weighted_internal_pairs =
        weighted_internal_pairs;
    std::array<unsigned long long, ac::kRelationSlots>
        host_relation_census{};
    cuda_require(
        cudaMemcpy(
            host_relation_census.data(),
            thrust::raw_pointer_cast(relation_census.data()),
            sizeof(host_relation_census),
            cudaMemcpyDeviceToHost),
        "geometry quotient relation census D2H");
    for (std::size_t slot = 0;
         slot < ac::kRelationSlots; ++slot) {
      result.census
          .weighted_internal_pairs_by_relation[slot] =
          host_relation_census[slot];
    }

    release_vector(&owner_rectangle_counts);
    release_vector(&singleton_indices);
    release_vector(&class_heads);
    release_vector(&class_ids);
    release_vector(&class_begins);
    release_vector(&internal_weights);
    release_vector(&relation_census);
    *output = std::move(result);
    return Status::success;
  } catch (const CudaFailure &) {
    return Status::cuda_error;
  } catch (const thrust::system_error &) {
    return Status::cuda_error;
  } catch (const std::bad_alloc &) {
    return Status::host_error;
  } catch (...) {
    return Status::host_error;
  }
}

Status build_device(
    const DeviceConfig &config,
    const thrust::device_vector<ac::RectI64> &rectangles,
    DeviceResult *output) noexcept
{
  return build_device(
      config,
      rectangles.empty()
          ? nullptr
          : thrust::raw_pointer_cast(rectangles.data()),
      rectangles.size(), output);
}

}  // namespace antenna_geometry_quotient
}  // namespace klayout_cuda
