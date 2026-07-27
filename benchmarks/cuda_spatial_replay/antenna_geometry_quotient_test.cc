#include "antenna_geometry_quotient.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace aq = klayout_cuda::antenna_geometry_quotient;
namespace ac = klayout_cuda::antenna_connectivity;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

aq::Config base_config(
    std::uint32_t owner_begin, std::uint32_t owner_count)
{
  aq::Config config;
  config.domain_count = 2;
  config.owner_begin = owner_begin;
  config.owner_count = owner_count;
  config.relation_rows[0] =
      (UINT64_C(1) << 0) | (UINT64_C(1) << 1);
  config.relation_rows[1] =
      (UINT64_C(1) << 0) | (UINT64_C(1) << 1);
  return config;
}

ac::RectI64 rectangle(
    std::int64_t left, std::int64_t bottom,
    std::int64_t right, std::int64_t top,
    std::uint32_t owner, std::uint32_t domain)
{
  return {left, bottom, right, top, owner, domain};
}

void test_identical_singleton_class()
{
  const aq::Config config = base_config(10, 4);
  const std::vector<ac::RectI64> input = {
      rectangle(0, 0, 10, 10, 13, 0),
      rectangle(20, 0, 30, 10, 12, 0),
      rectangle(0, 0, 10, 10, 11, 0),
      rectangle(0, 0, 10, 10, 10, 0)};
  aq::Result result;
  require(
      aq::build(config, input.data(), input.size(), &result) ==
          aq::Status::success,
      "identical singleton build");
  require(
      result.census.singleton_owners == 4 &&
          result.census.exception_owners == 0 &&
          result.census.geometry_classes == 2 &&
          result.census.collapsed_rectangles == 2 &&
          result.census.star_edges == 2 &&
          result.census.weighted_internal_pairs == 3 &&
          result.census.work_rectangles == 2,
      "identical singleton census");
  require(
      result.classes[0].rectangle.owner == 10 &&
          result.classes[0].multiplicity == 3 &&
          result.classes[0].member_begin == 0 &&
          result.class_members ==
              std::vector<std::uint32_t>({10, 11, 13, 12}),
      "identical singleton class order");
  require(
      result.parent_seeds ==
          std::vector<std::uint32_t>({10, 10, 12, 10}),
      "identical singleton parent seeds");
  require(
      result.census.weighted_internal_pairs_by_relation[
          ac::relation_slot(0, 0)] == 3,
      "identical singleton relation weight");
}

void test_domain_and_diagonal_partition()
{
  aq::Config config = base_config(0, 4);
  config.relation_rows[1] = UINT64_C(1) << 0;
  config.relation_rows[0] =
      (UINT64_C(1) << 0) | (UINT64_C(1) << 1);
  const std::vector<ac::RectI64> input = {
      rectangle(0, 0, 4, 4, 0, 0),
      rectangle(0, 0, 4, 4, 1, 0),
      rectangle(0, 0, 4, 4, 2, 1),
      rectangle(0, 0, 4, 4, 3, 1)};
  aq::Result result;
  require(
      aq::build(config, input.data(), input.size(), &result) ==
          aq::Status::success,
      "domain/diagonal build");
  require(
      result.classes.size() == 3 &&
          result.classes[0].multiplicity == 2 &&
          result.classes[1].multiplicity == 1 &&
          result.classes[2].multiplicity == 1,
      "domain/diagonal class partition");
  require(
      result.parent_seeds ==
          std::vector<std::uint32_t>({0, 0, 2, 3}) &&
          result.census.weighted_internal_pairs == 1,
      "domain/diagonal seeds and weight");
}

void test_multi_rectangle_exception()
{
  const aq::Config config = base_config(20, 3);
  const std::vector<ac::RectI64> input = {
      rectangle(0, 0, 5, 10, 20, 0),
      rectangle(5, 0, 10, 10, 20, 0),
      rectangle(0, 0, 10, 10, 21, 0),
      rectangle(0, 0, 10, 10, 22, 0)};
  aq::Result result;
  require(
      aq::build(config, input.data(), input.size(), &result) ==
          aq::Status::success,
      "exception build");
  require(
      result.census.singleton_owners == 2 &&
          result.census.exception_owners == 1 &&
          result.census.exception_rectangles == 2 &&
          result.census.geometry_classes == 1 &&
          result.census.star_edges == 1 &&
          result.census.weighted_internal_pairs == 1 &&
          result.census.work_rectangles == 3,
      "exception census");
  require(
      result.exception_rectangles[0].owner == 20 &&
          result.exception_rectangles[1].owner == 20 &&
          result.parent_seeds ==
              std::vector<std::uint32_t>({20, 21, 21}),
      "exception preservation");
}

void test_coordinate_extremes()
{
  const aq::Config config = base_config(0, 2);
  const auto minimum = std::numeric_limits<std::int64_t>::min();
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  const std::vector<ac::RectI64> input = {
      rectangle(minimum, maximum - 1, minimum + 1, maximum, 0, 0),
      rectangle(minimum, maximum - 1, minimum + 1, maximum, 1, 0)};
  aq::Result result;
  require(
      aq::build(config, input.data(), input.size(), &result) ==
          aq::Status::success &&
          result.classes.size() == 1 &&
          result.classes[0].multiplicity == 2 &&
          result.census.weighted_internal_pairs == 1,
      "signed coordinate extremes");
}

void require_unchanged_failure(
    const aq::Config &config,
    const std::vector<ac::RectI64> &input,
    aq::Status expected, const std::string &name)
{
  aq::Result result;
  result.census.owners = 999;
  result.parent_seeds = {7, 8, 9};
  const aq::Status status =
      aq::build(config, input.data(), input.size(), &result);
  require(status == expected, name + ": status");
  require(
      result.census.owners == 999 &&
          result.parent_seeds ==
              std::vector<std::uint32_t>({7, 8, 9}),
      name + ": output changed");
}

void test_fail_closed_inputs_and_limits()
{
  const std::vector<ac::RectI64> valid = {
      rectangle(0, 0, 2, 2, 0, 0),
      rectangle(0, 0, 2, 2, 1, 0),
      rectangle(0, 0, 2, 2, 2, 0)};

  aq::Config missing = base_config(0, 4);
  require_unchanged_failure(
      missing, valid, aq::Status::malformed_input,
      "missing owner");

  aq::Config malformed = base_config(0, 3);
  auto malformed_input = valid;
  malformed_input[0].right = malformed_input[0].left;
  require_unchanged_failure(
      malformed, malformed_input, aq::Status::malformed_input,
      "empty rectangle");

  auto mismatch_input = valid;
  mismatch_input.push_back(rectangle(3, 0, 4, 2, 0, 1));
  require_unchanged_failure(
      malformed, mismatch_input, aq::Status::malformed_input,
      "owner domain mismatch");

  aq::Config invalid = malformed;
  invalid.relation_rows[1] = UINT64_C(1) << 1;
  require_unchanged_failure(
      invalid, valid, aq::Status::invalid_configuration,
      "asymmetric relation");

  aq::Config class_limit = malformed;
  class_limit.limits.max_classes = 0;
  require_unchanged_failure(
      class_limit, valid, aq::Status::capacity_exceeded,
      "class limit");

  aq::Config star_limit = malformed;
  star_limit.limits.max_star_edges = 1;
  require_unchanged_failure(
      star_limit, valid, aq::Status::capacity_exceeded,
      "star limit");

  aq::Config weight_limit = malformed;
  weight_limit.limits.max_weighted_internal_pairs = 2;
  require_unchanged_failure(
      weight_limit, valid, aq::Status::capacity_exceeded,
      "weight limit");
}

struct OracleKey
{
  std::uint32_t domain;
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint32_t disambiguating_owner;

  bool operator<(const OracleKey &other) const
  {
    return std::tie(
               domain, left, bottom, right, top,
               disambiguating_owner) <
           std::tie(
               other.domain, other.left, other.bottom,
               other.right, other.top,
               other.disambiguating_owner);
  }
};

void test_random_differential()
{
  for (std::uint32_t seed = 0; seed != 40; ++seed) {
    std::mt19937 random(seed * 7919 + 17);
    const std::uint32_t owner_begin = 100 + seed * 64;
    const std::uint32_t owner_count = 8 + random() % 40;
    aq::Config config = base_config(owner_begin, owner_count);
    if (seed & 1) {
      config.relation_rows[1] = UINT64_C(1) << 0;
    }
    std::vector<ac::RectI64> input;
    std::vector<std::uint32_t> owner_rectangle_counts(owner_count);
    for (std::uint32_t local = 0; local != owner_count; ++local) {
      const std::uint32_t owner = owner_begin + local;
      const std::uint32_t domain = random() % 2;
      const std::int64_t left =
          static_cast<std::int64_t>(random() % 7) * 10 - 30;
      const std::int64_t bottom =
          static_cast<std::int64_t>(random() % 5) * 10 - 20;
      const bool exception = random() % 7 == 0;
      if (exception) {
        input.push_back(
            rectangle(left, bottom, left + 5, bottom + 10,
                      owner, domain));
        input.push_back(
            rectangle(left + 5, bottom, left + 10, bottom + 10,
                      owner, domain));
        owner_rectangle_counts[local] = 2;
      } else {
        input.push_back(
            rectangle(left, bottom, left + 10, bottom + 10,
                      owner, domain));
        owner_rectangle_counts[local] = 1;
      }
    }
    std::shuffle(input.begin(), input.end(), random);

    std::map<OracleKey, std::vector<std::uint32_t>> groups;
    std::uint64_t exception_rectangles = 0;
    std::uint64_t exception_owners = 0;
    for (std::uint32_t local = 0; local != owner_count; ++local) {
      exception_owners += owner_rectangle_counts[local] > 1;
    }
    for (const ac::RectI64 &item : input) {
      const std::uint32_t local = item.owner - owner_begin;
      if (owner_rectangle_counts[local] > 1) {
        ++exception_rectangles;
        continue;
      }
      const bool self_connected =
          (config.relation_rows[item.domain] &
           (UINT64_C(1) << item.domain)) != 0;
      groups[{item.domain, item.left, item.bottom, item.right,
              item.top, self_connected ? 0 : item.owner}]
          .push_back(item.owner);
    }

    aq::Result result;
    require(
        aq::build(config, input.data(), input.size(), &result) ==
            aq::Status::success,
        "random build seed " + std::to_string(seed));
    require(
        result.classes.size() == groups.size() &&
            result.census.exception_owners == exception_owners &&
            result.census.exception_rectangles ==
                exception_rectangles,
        "random census seed " + std::to_string(seed));

    std::uint64_t expected_weight = 0;
    std::vector<std::uint32_t> expected_parents(owner_count);
    for (std::uint32_t local = 0; local != owner_count; ++local) {
      expected_parents[local] = owner_begin + local;
    }
    std::size_t class_index = 0;
    for (auto &entry : groups) {
      auto &members = entry.second;
      std::sort(members.begin(), members.end());
      const std::uint32_t representative = members.front();
      for (std::uint32_t owner : members) {
        expected_parents[owner - owner_begin] = representative;
      }
      expected_weight +=
          static_cast<std::uint64_t>(members.size()) *
          (members.size() - 1) / 2;
      require(
          result.classes[class_index].rectangle.owner ==
                  representative &&
              result.classes[class_index].multiplicity ==
                  members.size(),
          "random class seed " + std::to_string(seed));
      ++class_index;
    }
    require(
        result.parent_seeds == expected_parents &&
            result.census.weighted_internal_pairs ==
                expected_weight &&
            result.census.star_edges ==
                result.census.collapsed_rectangles,
        "random parents/weights seed " + std::to_string(seed));
  }
}

void test_checked_weights()
{
  std::uint64_t weight = 0;
  require(
      aq::checked_cross_weight(17, 19, &weight) ==
              aq::Status::success &&
          weight == 323,
      "cross weight");
  require(
      aq::checked_internal_weight(5, &weight) ==
              aq::Status::success &&
          weight == 10,
      "internal odd weight");
  require(
      aq::checked_internal_weight(6, &weight) ==
              aq::Status::success &&
          weight == 15,
      "internal even weight");
  require(
      aq::checked_cross_weight(UINT64_MAX, 2, &weight) ==
          aq::Status::capacity_exceeded,
      "cross overflow");
  require(
      aq::checked_internal_weight(UINT64_MAX, &weight) ==
          aq::Status::capacity_exceeded,
      "internal overflow");
  require(
      aq::checked_cross_weight(0, 1, &weight) ==
              aq::Status::malformed_input &&
          aq::checked_internal_weight(0, &weight) ==
              aq::Status::malformed_input,
      "zero multiplicity");
}

int main()
{
  try {
    test_identical_singleton_class();
    test_domain_and_diagonal_partition();
    test_multi_rectangle_exception();
    test_coordinate_extremes();
    test_fail_closed_inputs_and_limits();
    test_random_differential();
    test_checked_weights();
    std::cout
        << "antenna_geometry_quotient_test: PASS"
        << " directed=7 random_seeds=40\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr
        << "antenna_geometry_quotient_test: FAIL: "
        << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
