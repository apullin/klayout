#include "antenna_geometry_quotient.h"

#include <algorithm>
#include <limits>
#include <new>
#include <tuple>
#include <utility>

namespace klayout_cuda {
namespace antenna_geometry_quotient {
namespace {

bool add_checked(
    std::uint64_t first, std::uint64_t second,
    std::uint64_t *result)
{
  if (first > UINT64_MAX - second) return false;
  *result = first + second;
  return true;
}

bool valid_config(const Config &config)
{
  if (!config.domain_count ||
      config.domain_count > ac::kMaximumDomains ||
      !config.owner_count ||
      config.owner_count > config.limits.max_owners ||
      config.owner_begin >
          UINT32_MAX - (config.owner_count - 1)) {
    return false;
  }
  const std::uint64_t domain_mask =
      config.domain_count == ac::kMaximumDomains
          ? UINT64_MAX
          : (UINT64_C(1) << config.domain_count) - 1;
  for (std::uint32_t domain = 0;
       domain < ac::kMaximumDomains; ++domain) {
    if (config.relation_rows[domain] & ~domain_mask) return false;
    if (domain >= config.domain_count &&
        config.relation_rows[domain]) {
      return false;
    }
    for (std::uint32_t other = 0;
         other < config.domain_count; ++other) {
      const bool forward =
          (config.relation_rows[domain] &
           (UINT64_C(1) << other)) != 0;
      const bool reverse =
          (config.relation_rows[other] &
           (UINT64_C(1) << domain)) != 0;
      if (forward != reverse) return false;
    }
  }
  return true;
}

bool geometry_less(
    const ac::RectI64 &first, const ac::RectI64 &second)
{
  return std::tie(
             first.domain, first.left, first.bottom,
             first.right, first.top, first.owner) <
         std::tie(
             second.domain, second.left, second.bottom,
             second.right, second.top, second.owner);
}

bool same_geometry(
    const ac::RectI64 &first, const ac::RectI64 &second)
{
  return first.domain == second.domain &&
         first.left == second.left &&
         first.bottom == second.bottom &&
         first.right == second.right &&
         first.top == second.top;
}

Status append_weight(
    const Config &config, std::uint32_t domain,
    std::uint64_t weight, Census *census)
{
  std::uint64_t total = 0;
  if (!add_checked(
          census->weighted_internal_pairs, weight, &total) ||
      total > config.limits.max_weighted_internal_pairs) {
    return Status::capacity_exceeded;
  }
  const std::size_t slot = ac::relation_slot(domain, domain);
  std::uint64_t relation_total = 0;
  if (!add_checked(
          census->weighted_internal_pairs_by_relation[slot],
          weight, &relation_total)) {
    return Status::capacity_exceeded;
  }
  census->weighted_internal_pairs = total;
  census->weighted_internal_pairs_by_relation[slot] =
      relation_total;
  return Status::success;
}

}  // namespace

Status checked_cross_weight(
    std::uint64_t first_multiplicity,
    std::uint64_t second_multiplicity,
    std::uint64_t *weight) noexcept
{
  if (!weight || !first_multiplicity || !second_multiplicity) {
    return Status::malformed_input;
  }
  if (first_multiplicity >
      UINT64_MAX / second_multiplicity) {
    return Status::capacity_exceeded;
  }
  *weight = first_multiplicity * second_multiplicity;
  return Status::success;
}

Status checked_internal_weight(
    std::uint64_t multiplicity, std::uint64_t *weight) noexcept
{
  if (!weight || !multiplicity) return Status::malformed_input;
  const std::uint64_t half = multiplicity / 2;
  const std::uint64_t other =
      multiplicity & 1 ? multiplicity : multiplicity - 1;
  if (half && other > UINT64_MAX / half) {
    return Status::capacity_exceeded;
  }
  *weight = half * other;
  return Status::success;
}

Status build(
    const Config &config, const ac::RectI64 *rectangles,
    std::uint64_t rectangle_count, Result *output) noexcept
{
  if (!output || !valid_config(config)) {
    return output ? Status::invalid_configuration
                  : Status::malformed_input;
  }
  if (!rectangles || !rectangle_count) {
    return Status::malformed_input;
  }
  if (rectangle_count > config.limits.max_input_rectangles ||
      rectangle_count > std::numeric_limits<std::size_t>::max()) {
    return Status::capacity_exceeded;
  }

  try {
    Result result;
    result.census.input_rectangles = rectangle_count;
    result.census.owners = config.owner_count;
    result.parent_seeds.resize(config.owner_count);
    result.owner_multiplicities.assign(config.owner_count, 1);
    std::vector<std::uint32_t> owner_rectangles(
        config.owner_count, 0);
    std::vector<std::uint32_t> owner_domains(
        config.owner_count, UINT32_MAX);

    for (std::uint32_t local = 0;
         local < config.owner_count; ++local) {
      result.parent_seeds[local] = config.owner_begin + local;
    }
    for (std::uint64_t index = 0;
         index < rectangle_count; ++index) {
      const ac::RectI64 &rectangle = rectangles[index];
      if (rectangle.left >= rectangle.right ||
          rectangle.bottom >= rectangle.top ||
          rectangle.owner < config.owner_begin ||
          rectangle.owner >=
              config.owner_begin + config.owner_count ||
          rectangle.domain >= config.domain_count) {
        return Status::malformed_input;
      }
      const std::uint32_t local =
          rectangle.owner - config.owner_begin;
      if (owner_rectangles[local] == UINT32_MAX) {
        return Status::capacity_exceeded;
      }
      ++owner_rectangles[local];
      if (owner_domains[local] == UINT32_MAX) {
        owner_domains[local] = rectangle.domain;
      } else if (owner_domains[local] != rectangle.domain) {
        return Status::malformed_input;
      }
    }

    for (std::uint32_t local = 0;
         local < config.owner_count; ++local) {
      if (!owner_rectangles[local]) return Status::malformed_input;
      if (owner_rectangles[local] == 1) {
        ++result.census.singleton_owners;
      } else {
        ++result.census.exception_owners;
      }
    }
    result.owner_domains = owner_domains;
    if (result.census.singleton_owners >
            config.limits.max_class_members ||
        rectangle_count - result.census.singleton_owners >
            config.limits.max_exception_rectangles) {
      return Status::capacity_exceeded;
    }

    std::vector<ac::RectI64> singleton_rectangles;
    singleton_rectangles.reserve(
        static_cast<std::size_t>(
            result.census.singleton_owners));
    result.exception_rectangles.reserve(
        static_cast<std::size_t>(
            rectangle_count - result.census.singleton_owners));
    for (std::uint64_t index = 0;
         index < rectangle_count; ++index) {
      const ac::RectI64 &rectangle = rectangles[index];
      const std::uint32_t local =
          rectangle.owner - config.owner_begin;
      if (owner_rectangles[local] == 1) {
        singleton_rectangles.push_back(rectangle);
      } else {
        result.exception_rectangles.push_back(rectangle);
      }
    }
    result.census.exception_rectangles =
        result.exception_rectangles.size();
    std::sort(
        singleton_rectangles.begin(),
        singleton_rectangles.end(), geometry_less);

    result.classes.reserve(singleton_rectangles.size());
    result.representative_rectangles.reserve(
        singleton_rectangles.size());
    result.class_members.reserve(singleton_rectangles.size());
    for (std::size_t begin = 0;
         begin < singleton_rectangles.size();) {
      const std::uint32_t domain =
          singleton_rectangles[begin].domain;
      const bool self_connected =
          (config.relation_rows[domain] &
           (UINT64_C(1) << domain)) != 0;
      std::size_t end = begin + 1;
      if (self_connected) {
        while (end < singleton_rectangles.size() &&
               same_geometry(
                   singleton_rectangles[begin],
                   singleton_rectangles[end])) {
          ++end;
        }
      }
      const std::uint64_t multiplicity = end - begin;
      if (result.classes.size() >= config.limits.max_classes ||
          multiplicity > UINT32_MAX) {
        return Status::capacity_exceeded;
      }
      const std::uint32_t representative =
          singleton_rectangles[begin].owner;
      GeometryClass geometry_class;
      geometry_class.rectangle = singleton_rectangles[begin];
      geometry_class.rectangle.owner = representative;
      geometry_class.member_begin = result.class_members.size();
      geometry_class.multiplicity =
          static_cast<std::uint32_t>(multiplicity);
      result.classes.push_back(geometry_class);
      result.representative_rectangles.push_back(
          geometry_class.rectangle);
      for (std::size_t index = begin; index < end; ++index) {
        const std::uint32_t owner =
            singleton_rectangles[index].owner;
        result.class_members.push_back(owner);
        result.parent_seeds[owner - config.owner_begin] =
            representative;
        result.owner_multiplicities[
            owner - config.owner_begin] = 0;
      }
      result.owner_multiplicities[
          representative - config.owner_begin] =
          geometry_class.multiplicity;
      if (multiplicity > 1) {
        std::uint64_t star_total = 0;
        if (!add_checked(
                result.census.star_edges, multiplicity - 1,
                &star_total) ||
            star_total > config.limits.max_star_edges) {
          return Status::capacity_exceeded;
        }
        result.census.star_edges = star_total;
        std::uint64_t weight = 0;
        const Status weight_status =
            checked_internal_weight(multiplicity, &weight);
        if (weight_status != Status::success) {
          return weight_status;
        }
        const Status append_status =
            append_weight(config, domain, weight, &result.census);
        if (append_status != Status::success) {
          return append_status;
        }
      }
      begin = end;
    }

    result.census.geometry_classes = result.classes.size();
    result.census.collapsed_rectangles =
        result.census.singleton_owners -
        result.census.geometry_classes;
    result.census.work_rectangles =
        result.census.geometry_classes +
        result.census.exception_rectangles;
    if (result.census.work_rectangles >
        config.limits.max_work_rectangles) {
      return Status::capacity_exceeded;
    }
    if (result.census.star_edges !=
        result.census.collapsed_rectangles) {
      return Status::host_error;
    }
    *output = std::move(result);
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
  case Status::cuda_error: return "cuda_error";
  case Status::host_error: return "host_error";
  }
  return "unknown";
}

}  // namespace antenna_geometry_quotient
}  // namespace klayout_cuda
