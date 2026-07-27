#include "antenna_geometry_quotient_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/host_vector.h>

namespace aq = klayout_cuda::antenna_geometry_quotient;
namespace ac = klayout_cuda::antenna_connectivity;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

ac::RectI64 rectangle(
    std::int64_t left, std::int64_t bottom,
    std::int64_t right, std::int64_t top,
    std::uint32_t owner, std::uint32_t domain)
{
  return {left, bottom, right, top, owner, domain};
}

bool equal_rectangle(
    const ac::RectI64 &first, const ac::RectI64 &second)
{
  return first.left == second.left &&
         first.bottom == second.bottom &&
         first.right == second.right &&
         first.top == second.top &&
         first.owner == second.owner &&
         first.domain == second.domain;
}

aq::DeviceConfig base_config(
    std::uint32_t owner_begin, std::uint32_t owner_count)
{
  aq::DeviceConfig config;
  config.quotient.domain_count = 3;
  config.quotient.owner_begin = owner_begin;
  config.quotient.owner_count = owner_count;
  config.quotient.relation_rows[0] =
      (UINT64_C(1) << 0) | (UINT64_C(1) << 1);
  config.quotient.relation_rows[1] =
      (UINT64_C(1) << 0) | (UINT64_C(1) << 1) |
      (UINT64_C(1) << 2);
  config.quotient.relation_rows[2] =
      (UINT64_C(1) << 1) | (UINT64_C(1) << 2);
  config.device_limits.max_device_bytes =
      UINT64_C(1024) * 1024 * 1024;
  config.device_limits.min_device_free_after_bytes =
      UINT64_C(16) * 1024 * 1024;
  config.device_limits.sort_scratch_fixed_guard_bytes =
      UINT64_C(1) * 1024 * 1024;
  return config;
}

template <class T>
std::vector<T> host_copy(
    const thrust::device_vector<T> &device)
{
  thrust::host_vector<T> host(device);
  return std::vector<T>(host.begin(), host.end());
}

void compare_with_oracle(
    const aq::DeviceConfig &config,
    const std::vector<ac::RectI64> &input,
    const std::string &name)
{
  aq::Result oracle;
  require(
      aq::build(
          config.quotient, input.data(), input.size(),
          &oracle) == aq::Status::success,
      name + ": host oracle");
  const thrust::device_vector<ac::RectI64> device_input(input);
  aq::DeviceResult device;
  require(
      aq::build_device(config, device_input, &device) ==
          aq::Status::success,
      name + ": device build");

  std::vector<ac::RectI64> expected_rectangles =
      oracle.representative_rectangles;
  expected_rectangles.insert(
      expected_rectangles.end(),
      oracle.exception_rectangles.begin(),
      oracle.exception_rectangles.end());
  const std::vector<ac::RectI64> actual_rectangles =
      host_copy(device.work_rectangles);
  require(
      actual_rectangles.size() == expected_rectangles.size(),
      name + ": work rectangle size");
  for (std::size_t index = 0;
       index < expected_rectangles.size(); ++index) {
    require(
        equal_rectangle(
            actual_rectangles[index],
            expected_rectangles[index]),
        name + ": work rectangle " +
            std::to_string(index));
  }

  std::vector<std::uint32_t> expected_rectangle_weights;
  expected_rectangle_weights.reserve(
      expected_rectangles.size());
  for (const aq::GeometryClass &geometry_class :
       oracle.classes) {
    expected_rectangle_weights.push_back(
        geometry_class.multiplicity);
  }
  expected_rectangle_weights.insert(
      expected_rectangle_weights.end(),
      oracle.exception_rectangles.size(), 1);
  require(
      host_copy(device.rectangle_multiplicities) ==
          expected_rectangle_weights,
      name + ": rectangle multiplicities");
  require(
      host_copy(device.parent_seeds) ==
          oracle.parent_seeds,
      name + ": parent seeds");
  require(
      host_copy(device.owner_domains) ==
          oracle.owner_domains,
      name + ": owner domains");
  require(
      host_copy(device.owner_multiplicities) ==
          oracle.owner_multiplicities,
      name + ": owner multiplicities");
  require(
      device.census.input_rectangles ==
              oracle.census.input_rectangles &&
          device.census.owners == oracle.census.owners &&
          device.census.singleton_owners ==
              oracle.census.singleton_owners &&
          device.census.exception_owners ==
              oracle.census.exception_owners &&
          device.census.exception_rectangles ==
              oracle.census.exception_rectangles &&
          device.census.geometry_classes ==
              oracle.census.geometry_classes &&
          device.census.collapsed_rectangles ==
              oracle.census.collapsed_rectangles &&
          device.census.work_rectangles ==
              oracle.census.work_rectangles &&
          device.census.star_edges ==
              oracle.census.star_edges &&
          device.census.weighted_internal_pairs ==
              oracle.census.weighted_internal_pairs &&
          device.census
                  .weighted_internal_pairs_by_relation ==
              oracle.census
                  .weighted_internal_pairs_by_relation,
      name + ": census");
}

void test_directed()
{
  {
    const aq::DeviceConfig config = base_config(10, 6);
    const std::vector<ac::RectI64> input = {
        rectangle(0, 0, 10, 10, 13, 0),
        rectangle(30, 0, 35, 10, 14, 1),
        rectangle(0, 0, 10, 10, 11, 0),
        rectangle(35, 0, 40, 10, 14, 1),
        rectangle(0, 0, 10, 10, 10, 0),
        rectangle(20, 0, 30, 10, 12, 2),
        rectangle(20, 0, 30, 10, 15, 2)};
    compare_with_oracle(config, input, "mixed directed");
  }
  {
    aq::DeviceConfig config = base_config(0, 4);
    config.quotient.relation_rows[2] = UINT64_C(1) << 1;
    const std::vector<ac::RectI64> input = {
        rectangle(0, 0, 4, 4, 0, 0),
        rectangle(0, 0, 4, 4, 1, 0),
        rectangle(0, 0, 4, 4, 2, 2),
        rectangle(0, 0, 4, 4, 3, 2)};
    compare_with_oracle(config, input, "disabled diagonal");
  }
  {
    const aq::DeviceConfig config = base_config(20, 2);
    const std::vector<ac::RectI64> input = {
        rectangle(0, 0, 5, 10, 20, 0),
        rectangle(5, 0, 10, 10, 20, 0),
        rectangle(20, 0, 25, 10, 21, 1),
        rectangle(25, 0, 30, 10, 21, 1)};
    compare_with_oracle(config, input, "all exceptions");
  }
}

void test_random_differential()
{
  for (std::uint32_t seed = 0; seed != 48; ++seed) {
    std::mt19937 random(seed * 65537 + 29);
    const std::uint32_t owner_begin = 1000 + seed * 128;
    const std::uint32_t owner_count = 12 + random() % 52;
    aq::DeviceConfig config =
        base_config(owner_begin, owner_count);
    if (seed & 1) {
      config.quotient.relation_rows[2] =
          UINT64_C(1) << 1;
    }
    std::vector<ac::RectI64> input;
    for (std::uint32_t local = 0;
         local != owner_count; ++local) {
      const std::uint32_t owner = owner_begin + local;
      const std::uint32_t domain = random() % 3;
      const std::int64_t left =
          static_cast<std::int64_t>(random() % 9) * 10 - 40;
      const std::int64_t bottom =
          static_cast<std::int64_t>(random() % 7) * 10 - 30;
      if (random() % 6 == 0) {
        input.push_back(
            rectangle(
                left, bottom, left + 5, bottom + 10,
                owner, domain));
        input.push_back(
            rectangle(
                left + 5, bottom, left + 10,
                bottom + 10, owner, domain));
      } else {
        input.push_back(
            rectangle(
                left, bottom, left + 10, bottom + 10,
                owner, domain));
      }
    }
    std::shuffle(input.begin(), input.end(), random);
    compare_with_oracle(
        config, input,
        "random seed " + std::to_string(seed));
  }
}

void require_unchanged_failure(
    const aq::DeviceConfig &config,
    const std::vector<ac::RectI64> &input,
    aq::Status expected, const std::string &name)
{
  const thrust::device_vector<ac::RectI64> device_input(input);
  aq::DeviceResult result;
  result.census.owners = 999;
  result.parent_seeds.assign(3, 7);
  const aq::Status status =
      aq::build_device(config, device_input, &result);
  require(status == expected, name + ": status");
  require(
      result.census.owners == 999 &&
          host_copy(result.parent_seeds) ==
              std::vector<std::uint32_t>({7, 7, 7}),
      name + ": output changed");
}

void test_fail_closed()
{
  const std::vector<ac::RectI64> valid = {
      rectangle(0, 0, 2, 2, 0, 0),
      rectangle(0, 0, 2, 2, 1, 0),
      rectangle(0, 0, 2, 2, 2, 0)};

  aq::DeviceConfig missing = base_config(0, 4);
  require_unchanged_failure(
      missing, valid, aq::Status::malformed_input,
      "missing owner");

  aq::DeviceConfig malformed = base_config(0, 3);
  auto malformed_input = valid;
  malformed_input[0].right = malformed_input[0].left;
  require_unchanged_failure(
      malformed, malformed_input,
      aq::Status::malformed_input, "empty rectangle");

  auto mismatch_input = valid;
  mismatch_input.push_back(
      rectangle(3, 0, 4, 2, 0, 1));
  require_unchanged_failure(
      malformed, mismatch_input,
      aq::Status::malformed_input, "owner domain mismatch");

  aq::DeviceConfig class_limit = malformed;
  class_limit.quotient.limits.max_classes = 0;
  require_unchanged_failure(
      class_limit, valid,
      aq::Status::capacity_exceeded, "class limit");

  aq::DeviceConfig weight_limit = malformed;
  weight_limit.quotient.limits
      .max_weighted_internal_pairs = 2;
  require_unchanged_failure(
      weight_limit, valid,
      aq::Status::capacity_exceeded, "weight limit");

  aq::DeviceConfig memory_limit = malformed;
  memory_limit.device_limits.max_device_bytes = 16;
  memory_limit.device_limits.min_device_free_after_bytes = 0;
  require_unchanged_failure(
      memory_limit, valid,
      aq::Status::capacity_exceeded, "device memory limit");
}

int main()
{
  try {
    int device_count = 0;
    require(
        cudaGetDeviceCount(&device_count) == cudaSuccess &&
            device_count > 0,
        "CUDA device unavailable");
    test_directed();
    test_random_differential();
    test_fail_closed();
    std::cout
        << "antenna_geometry_quotient_gpu_test: PASS"
        << " directed=3 random_seeds=48 fail_closed=6\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr
        << "antenna_geometry_quotient_gpu_test: FAIL: "
        << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
