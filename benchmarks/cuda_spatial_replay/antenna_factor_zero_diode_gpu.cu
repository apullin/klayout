/*
 * Exact one-sided factor-zero diode witnesses.
 */

#include "antenna_factor_zero_diode_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <new>

namespace {

namespace afd = klayout_cuda::antenna_factor_zero_diode;
namespace ac = klayout_cuda::antenna_connectivity;

constexpr std::uint32_t kThreads = 256;

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 0;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          (count + kThreads - 1) / kThreads, 65535));
}

afd::Status map_connectivity(ac::Status status)
{
  switch (status) {
  case ac::Status::success:
    return afd::Status::success;
  case ac::Status::invalid_configuration:
    return afd::Status::invalid_configuration;
  case ac::Status::malformed_input:
  case ac::Status::poisoned_state:
    return afd::Status::malformed_input;
  case ac::Status::capacity_exceeded:
    return afd::Status::capacity_exceeded;
  case ac::Status::convergence_failure:
    return afd::Status::convergence_failure;
  case ac::Status::cuda_error:
    return afd::Status::cuda_error;
  case ac::Status::host_error:
    return afd::Status::host_error;
  }
  return afd::Status::host_error;
}

bool valid_config(const afd::Config &config)
{
  return config.bin_size > 0 && config.limits.max_nodes &&
         config.limits.max_rectangles &&
         config.limits.max_memberships &&
         config.limits.max_pair_occurrences &&
         config.limits.max_unique_candidates &&
         config.limits.max_cell_members &&
         config.limits.max_pair_tests_per_cell &&
         config.limits.max_total_pair_tests &&
         config.limits.max_dsu_iterations &&
         config.limits.max_device_bytes;
}

ac::Config connectivity_config(const afd::Config &input)
{
  ac::Config config;
  config.domain_count = 2;
  config.relation_rows[0] = UINT64_C(1) << 1;
  config.relation_rows[1] = UINT64_C(1) << 0;
  config.exact_filter_before_materialization = true;
  config.bin_size = input.bin_size;
  config.device = input.device;
  config.limits = input.limits;
  return config;
}

struct DeviceScalars
{
  unsigned long long local_nplus_active_contacts = 0;
  unsigned long long well_rejected_contacts = 0;
  unsigned long long exact_witness_contacts = 0;
  unsigned int status = 0;
};

__global__ void mark_nwell_component_bits_kernel(
    const std::uint32_t *labels,
    std::uint64_t label_count,
    std::uint64_t nwell_owner_begin,
    std::uint64_t nwell_owner_count,
    std::uint32_t *root_nwell_bits,
    unsigned int *status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t local = first;
       local < nwell_owner_count; local += stride) {
    const std::uint64_t owner = nwell_owner_begin + local;
    if (owner >= label_count) {
      atomicOr(status, 1u);
      continue;
    }
    const std::uint32_t root = labels[owner];
    if (root >= label_count || labels[root] != root) {
      atomicOr(status, 1u);
      continue;
    }
    atomicOr(
        root_nwell_bits + (root >> 5),
        static_cast<std::uint32_t>(1u << (root & 31)));
  }
}

__global__ void filter_contact_witnesses_kernel(
    const std::uint32_t *labels,
    std::uint64_t label_count,
    std::uint32_t *contact_present,
    std::uint64_t contact_count,
    const std::uint32_t *root_nwell_bits,
    DeviceScalars *scalars)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t owner = first;
       owner < contact_count; owner += stride) {
    if (!contact_present[owner]) continue;
    atomicAdd(
        &scalars->local_nplus_active_contacts, 1ull);
    if (owner >= label_count) {
      atomicOr(&scalars->status, 1u);
      contact_present[owner] = 0;
      continue;
    }
    const std::uint32_t root = labels[owner];
    if (root >= label_count || labels[root] != root) {
      atomicOr(&scalars->status, 1u);
      contact_present[owner] = 0;
      continue;
    }
    if (root_nwell_bits[root >> 5] &
        static_cast<std::uint32_t>(1u << (root & 31))) {
      contact_present[owner] = 0;
      atomicAdd(
          &scalars->well_rejected_contacts, 1ull);
    } else {
      atomicAdd(
          &scalars->exact_witness_contacts, 1ull);
    }
  }
}

}  // namespace

namespace klayout_cuda {
namespace antenna_factor_zero_diode {

Status filter_contact_witnesses(
    const Config &config,
    thrust::device_vector<ac::RectI64>
        &&contact_then_nwell,
    std::uint64_t contact_owner_count,
    std::uint64_t nwell_owner_count,
    thrust::device_vector<std::uint32_t> *contact_present,
    ContactWitnessCensus *census) noexcept
{
  if (!valid_config(config)) {
    return Status::invalid_configuration;
  }
  if (!contact_present || !census ||
      !contact_owner_count || !nwell_owner_count ||
      contact_present->size() != contact_owner_count ||
      contact_owner_count > UINT32_MAX - nwell_owner_count ||
      contact_then_nwell.empty()) {
    return Status::malformed_input;
  }

  try {
    const std::uint64_t total_nodes =
        contact_owner_count + nwell_owner_count;
    cudaError_t cuda_status = cudaSetDevice(config.device);
    if (cuda_status != cudaSuccess) return Status::cuda_error;

    ac::Connectivity detector(connectivity_config(config));
    Status status =
        map_connectivity(detector.configuration_status());
    if (status != Status::success) return status;
    ac::StageCensus graph;
    status = map_connectivity(
        detector.append_stage_consuming(
            std::move(contact_then_nwell), total_nodes,
            UINT64_C(3), nullptr, &graph));
    if (status != Status::success) return status;
    ac::DeviceLabelView labels;
    status = map_connectivity(
        detector.device_label_view(&labels));
    if (status != Status::success) return status;
    if (labels.count != total_nodes) {
      return Status::malformed_input;
    }

    const std::uint64_t root_word_count =
        (total_nodes + 31) / 32;
    thrust::device_vector<std::uint32_t> root_nwell_bits(
        static_cast<std::size_t>(root_word_count));
    thrust::device_vector<DeviceScalars> device_scalars(1);
    cuda_status = cudaMemset(
        thrust::raw_pointer_cast(root_nwell_bits.data()), 0,
        root_nwell_bits.size() * sizeof(std::uint32_t));
    if (cuda_status != cudaSuccess) return Status::cuda_error;
    cuda_status = cudaMemset(
        thrust::raw_pointer_cast(device_scalars.data()), 0,
        sizeof(DeviceScalars));
    if (cuda_status != cudaSuccess) return Status::cuda_error;

    mark_nwell_component_bits_kernel<<<
        launch_blocks(nwell_owner_count), kThreads>>>(
        labels.labels, labels.count, contact_owner_count,
        nwell_owner_count,
        thrust::raw_pointer_cast(root_nwell_bits.data()),
        &thrust::raw_pointer_cast(
             device_scalars.data())->status);
    if (cudaGetLastError() != cudaSuccess) {
      return Status::cuda_error;
    }
    filter_contact_witnesses_kernel<<<
        launch_blocks(contact_owner_count), kThreads>>>(
        labels.labels, labels.count,
        thrust::raw_pointer_cast(contact_present->data()),
        contact_owner_count,
        thrust::raw_pointer_cast(root_nwell_bits.data()),
        thrust::raw_pointer_cast(device_scalars.data()));
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      return Status::cuda_error;
    }

    DeviceScalars host_scalars;
    cuda_status = cudaMemcpy(
        &host_scalars,
        thrust::raw_pointer_cast(device_scalars.data()),
        sizeof(host_scalars), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) return Status::cuda_error;
    if (host_scalars.status ||
        host_scalars.local_nplus_active_contacts !=
            host_scalars.well_rejected_contacts +
                host_scalars.exact_witness_contacts) {
      return Status::malformed_input;
    }

    ContactWitnessCensus result;
    result.contact_owners = contact_owner_count;
    result.nwell_owners = nwell_owner_count;
    result.rectangles = graph.total_rectangles;
    result.local_nplus_active_contacts =
        host_scalars.local_nplus_active_contacts;
    result.well_rejected_contacts =
        host_scalars.well_rejected_contacts;
    result.exact_witness_contacts =
        host_scalars.exact_witness_contacts;
    result.memberships = graph.memberships;
    result.occupied_cells = graph.occupied_cells;
    result.pair_occurrences = graph.pair_occurrences;
    result.unique_candidates = graph.unique_candidates;
    result.exact_edges = graph.exact_edges;
    result.dsu_iterations = graph.dsu_iterations;
    *census = result;
    return Status::success;
  } catch (const std::bad_alloc &) {
    return Status::host_error;
  } catch (...) {
    return Status::host_error;
  }
}

const char *status_string(Status status) noexcept
{
  switch (status) {
  case Status::success: return "success";
  case Status::invalid_configuration:
    return "invalid_configuration";
  case Status::malformed_input: return "malformed_input";
  case Status::capacity_exceeded: return "capacity_exceeded";
  case Status::convergence_failure:
    return "convergence_failure";
  case Status::cuda_error: return "cuda_error";
  case Status::host_error: return "host_error";
  }
  return "unknown";
}

}  // namespace antenna_factor_zero_diode
}  // namespace klayout_cuda
