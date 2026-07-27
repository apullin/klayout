/*
 * Directed and seeded CPU-oracle differentials for staged antenna
 * connectivity.
 */

#include "antenna_connectivity_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ac = klayout_cuda::antenna_connectivity;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

void require_status(ac::Status actual, ac::Status expected,
                    const std::string &operation)
{
  if (actual != expected) {
    throw std::runtime_error(
        operation + ": expected " + ac::status_string(expected) +
        ", got " + ac::status_string(actual));
  }
}

void allow(ac::Config *config, std::uint32_t first,
           std::uint32_t second)
{
  config->relation_rows[first] |= UINT64_C(1) << second;
  config->relation_rows[second] |= UINT64_C(1) << first;
}

ac::Config base_config(std::uint32_t domains, std::int64_t bin_size)
{
  ac::Config config;
  config.domain_count = domains;
  config.bin_size = bin_size;
  config.device = 0;
  config.limits.max_nodes = 100000;
  config.limits.max_rectangles = 200000;
  config.limits.max_memberships = 2000000;
  config.limits.max_pair_occurrences = 10000000;
  config.limits.max_unique_candidates = 2000000;
  config.limits.max_cell_members = 10000;
  config.limits.max_total_pair_tests = 10000000;
  config.limits.max_dsu_iterations = 128;
  return config;
}

std::int64_t floor_div(std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

bool touches(const ac::RectI64 &first, const ac::RectI64 &second)
{
  return first.left <= second.right &&
         second.left <= first.right &&
         first.bottom <= second.top &&
         second.bottom <= first.top;
}

bool relation_allowed(const ac::Config &config,
                      const ac::RectI64 &first,
                      const ac::RectI64 &second)
{
  return (config.relation_rows[first.domain] &
          (UINT64_C(1) << second.domain)) != 0;
}

struct CpuDsu
{
  explicit CpuDsu(std::size_t count) : parent(count)
  {
    for (std::size_t index = 0; index < count; ++index) {
      parent[index] = static_cast<std::uint32_t>(index);
    }
  }

  std::uint32_t root(std::uint32_t node)
  {
    while (parent[node] != node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  }

  void join(std::uint32_t first, std::uint32_t second)
  {
    first = root(first);
    second = root(second);
    if (first == second) return;
    const std::uint32_t low = std::min(first, second);
    const std::uint32_t high = std::max(first, second);
    parent[high] = low;
  }

  std::vector<std::uint32_t> labels()
  {
    for (std::uint32_t node = 0; node < parent.size(); ++node) {
      parent[node] = root(node);
    }
    return parent;
  }

  std::vector<std::uint32_t> parent;
};

struct Oracle
{
  explicit Oracle(const ac::Config &configuration)
      : config(configuration)
  {
  }

  std::pair<std::vector<std::uint32_t>, ac::StageCensus>
  append(const std::vector<ac::RectI64> &stage,
         std::uint64_t new_node_count,
         std::uint64_t close_domain_mask)
  {
    const std::size_t previous_nodes = owner_count;
    const std::size_t previous_rectangles =
        active_rectangles.size();
    owner_count += new_node_count;
    all_rectangles.insert(
        all_rectangles.end(), stage.begin(), stage.end());
    active_rectangles.insert(
        active_rectangles.end(), stage.begin(), stage.end());

    ac::StageCensus census;
    census.previous_nodes = previous_nodes;
    census.appended_nodes = new_node_count;
    census.total_nodes = owner_count;
    census.previous_rectangles = previous_rectangles;
    census.appended_rectangles = stage.size();
    census.total_rectangles = active_rectangles.size();

    std::set<std::pair<std::int64_t, std::int64_t> >
        occupied_cells;
    for (const ac::RectI64 &rectangle : active_rectangles) {
      const std::int64_t x0 =
          floor_div(rectangle.left, config.bin_size);
      const std::int64_t x1 =
          floor_div(rectangle.right, config.bin_size);
      const std::int64_t y0 =
          floor_div(rectangle.bottom, config.bin_size);
      const std::int64_t y1 =
          floor_div(rectangle.top, config.bin_size);
      census.memberships +=
          static_cast<std::uint64_t>(x1 - x0 + 1) *
          static_cast<std::uint64_t>(y1 - y0 + 1);
      for (std::int64_t y = y0; y <= y1; ++y) {
        for (std::int64_t x = x0; x <= x1; ++x) {
          occupied_cells.emplace(x, y);
        }
      }
    }
    census.occupied_cells = occupied_cells.size();

    std::set<std::uint64_t> owner_candidates;
    std::set<std::uint64_t> owner_edges;
    for (std::uint32_t second = 0;
         second < active_rectangles.size(); ++second) {
      for (std::uint32_t first = 0; first < second; ++first) {
        const ac::RectI64 &first_rectangle =
            active_rectangles[first];
        const ac::RectI64 &second_rectangle =
            active_rectangles[second];
        if (first_rectangle.owner < previous_nodes &&
            second_rectangle.owner < previous_nodes) {
          continue;
        }
        const bool same_owner =
            first_rectangle.owner == second_rectangle.owner;
        if (!same_owner &&
            !relation_allowed(
                config, first_rectangle, second_rectangle)) {
          continue;
        }

        const std::int64_t first_x0 =
            floor_div(first_rectangle.left, config.bin_size);
        const std::int64_t first_x1 =
            floor_div(first_rectangle.right, config.bin_size);
        const std::int64_t first_y0 =
            floor_div(first_rectangle.bottom, config.bin_size);
        const std::int64_t first_y1 =
            floor_div(first_rectangle.top, config.bin_size);
        const std::int64_t second_x0 =
            floor_div(second_rectangle.left, config.bin_size);
        const std::int64_t second_x1 =
            floor_div(second_rectangle.right, config.bin_size);
        const std::int64_t second_y0 =
            floor_div(second_rectangle.bottom, config.bin_size);
        const std::int64_t second_y1 =
            floor_div(second_rectangle.top, config.bin_size);
        const std::int64_t overlap_x0 =
            std::max(first_x0, second_x0);
        const std::int64_t overlap_x1 =
            std::min(first_x1, second_x1);
        const std::int64_t overlap_y0 =
            std::max(first_y0, second_y0);
        const std::int64_t overlap_y1 =
            std::min(first_y1, second_y1);
        if (overlap_x0 > overlap_x1 ||
            overlap_y0 > overlap_y1) {
          continue;
        }
        const std::uint64_t occurrences =
            static_cast<std::uint64_t>(
                overlap_x1 - overlap_x0 + 1) *
            static_cast<std::uint64_t>(
                overlap_y1 - overlap_y0 + 1);
        const bool exact_touch =
            touches(first_rectangle, second_rectangle);
        if (!config.exact_filter_before_materialization ||
            exact_touch) {
          census.pair_occurrences += occurrences;
        }
        if (same_owner) continue;
        if (config.exact_filter_before_materialization &&
            !exact_touch) {
          continue;
        }
        const std::uint32_t low_owner = std::min(
            first_rectangle.owner, second_rectangle.owner);
        const std::uint32_t high_owner = std::max(
            first_rectangle.owner, second_rectangle.owner);
        const std::uint64_t owner_pair =
            (static_cast<std::uint64_t>(low_owner) << 32) |
            high_owner;
        owner_candidates.insert(owner_pair);
        if (exact_touch) {
          owner_edges.insert(owner_pair);
        }
      }
    }

    census.unique_candidates = owner_candidates.size();
    census.exact_edges = owner_edges.size();
    std::vector<std::uint32_t> owner_domains(owner_count, UINT32_MAX);
    for (const ac::RectI64 &rectangle : all_rectangles) {
      owner_domains[rectangle.owner] = rectangle.domain;
    }
    for (const std::uint64_t key : owner_candidates) {
      const std::uint32_t first =
          static_cast<std::uint32_t>(key >> 32);
      const std::uint32_t second =
          static_cast<std::uint32_t>(key);
      ++census.candidates_by_relation[ac::relation_slot(
          owner_domains[first], owner_domains[second])];
    }
    for (const std::uint64_t key : owner_edges) {
      const std::uint32_t first =
          static_cast<std::uint32_t>(key >> 32);
      const std::uint32_t second =
          static_cast<std::uint32_t>(key);
      ++census.edges_by_relation[ac::relation_slot(
          owner_domains[first], owner_domains[second])];
    }

    CpuDsu dsu(owner_count);
    for (std::uint32_t second = 0;
         second < all_rectangles.size(); ++second) {
      for (std::uint32_t first = 0; first < second; ++first) {
        const ac::RectI64 &first_rectangle =
            all_rectangles[first];
        const ac::RectI64 &second_rectangle =
            all_rectangles[second];
        if (first_rectangle.owner == second_rectangle.owner ||
            !relation_allowed(
                config, first_rectangle, second_rectangle)) {
          continue;
        }
        if (touches(first_rectangle, second_rectangle)) {
          dsu.join(
              first_rectangle.owner, second_rectangle.owner);
        }
      }
    }

    closed_domains |= close_domain_mask;
    std::uint64_t released_domains = 0;
    for (std::uint32_t domain = 0;
         domain < config.domain_count; ++domain) {
      const std::uint64_t bit = UINT64_C(1) << domain;
      if ((closed_domains & bit) &&
          !(config.relation_rows[domain] & ~closed_domains)) {
        released_domains |= bit;
      }
    }
    active_rectangles.erase(
        std::remove_if(
            active_rectangles.begin(), active_rectangles.end(),
            [released_domains](const ac::RectI64 &rectangle) {
              return (released_domains &
                      (UINT64_C(1) << rectangle.domain)) != 0;
            }),
        active_rectangles.end());
    census.retained_rectangles = active_rectangles.size();
    census.released_rectangles =
        census.total_rectangles - census.retained_rectangles;
    census.closed_domain_mask = closed_domains;
    return {dsu.labels(), census};
  }

  ac::Config config;
  std::uint32_t owner_count = 0;
  std::uint64_t closed_domains = 0;
  std::vector<ac::RectI64> all_rectangles;
  std::vector<ac::RectI64> active_rectangles;
};

void require_census_equal(
    const ac::StageCensus &actual,
    const ac::StageCensus &expected,
    const std::string &name)
{
  require(actual.previous_nodes == expected.previous_nodes,
          name + ": previous node census mismatch");
  require(actual.appended_nodes == expected.appended_nodes,
          name + ": appended node census mismatch");
  require(actual.total_nodes == expected.total_nodes,
          name + ": total node census mismatch");
  require(actual.previous_rectangles ==
              expected.previous_rectangles,
          name + ": previous rectangle census mismatch");
  require(actual.appended_rectangles ==
              expected.appended_rectangles,
          name + ": appended rectangle census mismatch");
  require(actual.total_rectangles ==
              expected.total_rectangles,
          name + ": total rectangle census mismatch");
  require(actual.retained_rectangles ==
              expected.retained_rectangles,
          name + ": retained rectangle census mismatch");
  require(actual.released_rectangles ==
              expected.released_rectangles,
          name + ": released rectangle census mismatch");
  require(actual.closed_domain_mask ==
              expected.closed_domain_mask,
          name + ": closed-domain census mismatch");
  require(actual.memberships == expected.memberships,
          name + ": membership census mismatch");
  require(actual.occupied_cells == expected.occupied_cells,
          name + ": occupied-cell census mismatch");
  require(actual.pair_occurrences == expected.pair_occurrences,
          name + ": pair-occurrence census mismatch");
  require(actual.unique_candidates == expected.unique_candidates,
          name + ": unique-candidate census mismatch");
  require(actual.exact_edges == expected.exact_edges,
          name + ": exact-edge census mismatch");
  require(actual.candidates_by_relation ==
              expected.candidates_by_relation,
          name + ": per-relation candidate census mismatch");
  require(actual.edges_by_relation == expected.edges_by_relation,
          name + ": per-relation edge census mismatch");
}

ac::StageCensus run_stage(
    ac::Connectivity *gpu, Oracle *oracle,
    const std::vector<ac::RectI64> &stage,
    std::uint64_t new_node_count, const std::string &name,
    std::uint64_t close_domain_mask = 0,
    ac::AppendMode mode = ac::AppendMode::transactional)
{
  const auto expected = oracle->append(
      stage, new_node_count, close_domain_mask);
  std::vector<std::uint32_t> labels = {UINT32_C(0xdeadbeef)};
  ac::StageCensus census;
  census.previous_nodes = UINT64_C(0xfeedface);
  require_status(
      gpu->append_stage(
          stage.data(), stage.size(), new_node_count,
          close_domain_mask, ac::InputMemory::host, &labels,
          &census, mode),
      ac::Status::success, name);
  require(labels == expected.first, name + ": label mismatch");
  require_census_equal(census, expected.second, name);
  return census;
}

void require_failure_unchanged(
    ac::Connectivity *gpu, const ac::RectI64 *stage,
    std::uint64_t count, ac::Status expected,
    const std::string &name, std::uint64_t new_node_count = 1,
    std::uint64_t close_domain_mask = 0);

void test_touching_and_relation_census()
{
  ac::Config config = base_config(2, 10);
  allow(&config, 0, 0);
  allow(&config, 1, 1);
  allow(&config, 0, 1);
  ac::Connectivity gpu(config);
  require_status(
      gpu.configuration_status(), ac::Status::success,
      "touch configuration");
  Oracle oracle(config);
  const std::vector<ac::RectI64> stage = {
      {0, 0, 10, 10, 0, 0},
      {10, 2, 20, 8, 1, 0},       // edge touch
      {20, 8, 30, 18, 2, 0},      // point touch
      {31, 8, 40, 18, 3, 0},      // separated, same broad bin
      {5, 5, 6, 6, 4, 1},         // contained cross-domain
      {40, 18, 45, 20, 5, 1}};    // point touch to node 3
  run_stage(
      &gpu, &oracle, stage, 6, "touch/point/separated");

  std::vector<std::uint32_t> labels;
  require_status(
      gpu.snapshot_labels(&labels), ac::Status::success,
      "touch snapshot");
  require(labels ==
              std::vector<std::uint32_t>({0, 0, 0, 3, 0, 3}),
          "directed touching labels are wrong");
}

void test_multibin_deduplication()
{
  ac::Config config = base_config(2, 10);
  allow(&config, 0, 1);
  ac::Connectivity gpu(config);
  Oracle oracle(config);
  const std::vector<ac::RectI64> stage = {
      {0, 0, 40, 40, 0, 0},
      {40, 10, 80, 30, 1, 1}};
  const auto expected = oracle.append(stage, 2, 0);
  std::vector<std::uint32_t> labels;
  ac::StageCensus census;
  require_status(
      gpu.append_stage(
          stage.data(), stage.size(), 2, 0,
          ac::InputMemory::host, &labels, &census),
      ac::Status::success, "multi-bin dedup");
  require(labels == expected.first, "multi-bin labels mismatch");
  require_census_equal(census, expected.second, "multi-bin");
  require(census.pair_occurrences > 1,
          "multi-bin fixture did not duplicate broad candidates");
  require(census.unique_candidates == 1 &&
              census.exact_edges == 1,
          "multi-bin pair was not deduplicated exactly once");
}

void test_exact_filter_before_materialization()
{
  ac::Config broad_config = base_config(2, 100);
  require(!broad_config.exact_filter_before_materialization,
          "exact filtering is not opt-in");
  allow(&broad_config, 0, 0);
  allow(&broad_config, 1, 1);
  allow(&broad_config, 0, 1);
  const std::vector<ac::RectI64> stage = {
      {0, 0, 10, 10, 0, 0},
      {90, 90, 99, 99, 1, 0},  // broad same-cell false candidate
      {10, 0, 20, 10, 2, 1},   // exact edge touch to owner 0
      {30, 30, 40, 40, 3, 1}}; // broad same-cell false candidate

  ac::Connectivity broad_gpu(broad_config);
  Oracle broad_oracle(broad_config);
  const ac::StageCensus broad = run_stage(
      &broad_gpu, &broad_oracle, stage, 4,
      "broad materialization control");
  require(broad.pair_occurrences == 6 &&
              broad.unique_candidates == 6 &&
              broad.exact_edges == 1,
          "broad control did not expose false candidates");

  ac::Config exact_config = broad_config;
  exact_config.exact_filter_before_materialization = true;
  ac::Connectivity exact_gpu(exact_config);
  Oracle exact_oracle(exact_config);
  const ac::StageCensus exact = run_stage(
      &exact_gpu, &exact_oracle, stage, 4,
      "exact filter before materialization");
  require(exact.pair_occurrences == 1 &&
              exact.unique_candidates == 1 &&
              exact.exact_edges == 1,
          "exact filter retained a broad false candidate");

  std::vector<ac::RectI64> strided_stage;
  for (std::uint32_t owner = 0; owner < 33; ++owner) {
    const std::int64_t left =
        static_cast<std::int64_t>(owner) * 2;
    strided_stage.push_back(
        {left, 0, left + 2, 10, owner, 0});
  }
  ac::Connectivity strided_gpu(exact_config);
  Oracle strided_oracle(exact_config);
  const ac::StageCensus strided = run_stage(
      &strided_gpu, &strided_oracle, strided_stage, 33,
      "exact parallel pair stride");
  require(strided.pair_occurrences == 32 &&
              strided.unique_candidates == 32 &&
              strided.exact_edges == 32,
          "exact parallel pair stride lost a neighbor edge");

  std::vector<std::uint32_t> broad_labels;
  std::vector<std::uint32_t> exact_labels;
  require_status(
      broad_gpu.snapshot_labels(&broad_labels),
      ac::Status::success, "broad control labels");
  require_status(
      exact_gpu.snapshot_labels(&exact_labels),
      ac::Status::success, "exact filter labels");
  require(exact_labels == broad_labels &&
              exact_labels ==
                  std::vector<std::uint32_t>({0, 1, 0, 3}),
          "exact filtering changed connectivity");

  exact_config.limits.max_pair_occurrences = 1;
  ac::Connectivity exact_bounded(exact_config);
  Oracle exact_bounded_oracle(exact_config);
  run_stage(
      &exact_bounded, &exact_bounded_oracle, stage, 4,
      "exact occurrence capacity");

  ac::Config exact_work_bounded = exact_config;
  exact_work_bounded.limits.max_pair_occurrences = 100;
  exact_work_bounded.limits.max_total_pair_tests = 5;
  ac::Connectivity exact_work_bounded_gpu(exact_work_bounded);
  require_failure_unchanged(
      &exact_work_bounded_gpu, stage.data(), stage.size(),
      ac::Status::capacity_exceeded,
      "exact aggregate pair-work capacity", 4);

  broad_config.limits.max_pair_occurrences = 1;
  ac::Connectivity broad_bounded(broad_config);
  require_failure_unchanged(
      &broad_bounded, stage.data(), stage.size(),
      ac::Status::capacity_exceeded,
      "broad occurrence capacity", 4);

  ac::Config overlap_config = base_config(1, 100);
  allow(&overlap_config, 0, 0);
  overlap_config.exact_filter_before_materialization = true;
  ac::Connectivity overlap_gpu(overlap_config);
  const std::array<ac::RectI64, 2> overlapping_tiles = {{
      {0, 0, 10, 10, 0, 0},
      {5, 0, 15, 10, 0, 0}}};
  require_failure_unchanged(
      &overlap_gpu, overlapping_tiles.data(),
      overlapping_tiles.size(), ac::Status::malformed_input,
      "exact-filter overlapping owner tiles");
}

void test_concave_owner_rectangulation()
{
  ac::Config config = base_config(1, 5);
  allow(&config, 0, 0);
  ac::Connectivity gpu(config);
  Oracle oracle(config);
  const ac::StageCensus census = run_stage(
      &gpu, &oracle,
      {{0, 0, 4, 10, 0, 0},
       {4, 0, 10, 4, 0, 0},   // owner 0: exact concave L
       {8, 4, 12, 8, 1, 0},   // touches the L boundary
       {30, 30, 35, 35, 2, 0}},
      3, "concave L rectangulation");
  require(census.appended_rectangles == 4 &&
              census.appended_nodes == 3,
          "L-shape owner/rectangle census mismatch");
  std::vector<std::uint32_t> labels;
  require_status(
      gpu.snapshot_labels(&labels), ac::Status::success,
      "L-shape snapshot");
  require(labels == std::vector<std::uint32_t>({0, 0, 2}),
          "concave owner did not connect exactly");
}

void test_staged_antenna_bridge(bool exact_filter)
{
  ac::Config config = base_config(10, 8);
  config.exact_filter_before_materialization = exact_filter;
  for (std::uint32_t domain = 0; domain < 10; ++domain) {
    allow(&config, domain, domain);
  }
  allow(&config, 0, 1);  // GATE-POLY
  allow(&config, 1, 2);  // POLY-CONTACT
  allow(&config, 2, 3);  // CONTACT-M1
  allow(&config, 3, 4);  // M1-VIA1
  allow(&config, 4, 5);  // VIA1-M2
  allow(&config, 5, 6);  // M2-VIA2
  allow(&config, 6, 7);  // VIA2-M3
  allow(&config, 7, 8);  // M3-VIA3
  allow(&config, 8, 9);  // VIA3-M4

  ac::Connectivity gpu(config);
  Oracle oracle(config);
  const ac::StageCensus m1 = run_stage(
      &gpu, &oracle,
      {{0, 0, 4, 4, 0, 0},
       {0, 0, 10, 4, 1, 1},
       {8, 0, 12, 4, 2, 2},
       {10, 0, 20, 4, 3, 3},
       {40, 0, 50, 4, 4, 3}},
      5, "antenna M1 checkpoint",
      (UINT64_C(1) << 0) | (UINT64_C(1) << 1) |
          (UINT64_C(1) << 2) | (UINT64_C(1) << 3),
      ac::AppendMode::consuming);
  require(m1.retained_rectangles == 2,
          "M1 frontier did not release GATE/POLY/CONTACT");
  const ac::StageCensus m2 = run_stage(
      &gpu, &oracle,
      {{18, 0, 42, 4, 5, 4},
       {20, 0, 45, 4, 6, 5}},
      2, "antenna M2 checkpoint",
      (UINT64_C(1) << 4) | (UINT64_C(1) << 5),
      ac::AppendMode::consuming);
  require(m2.retained_rectangles == 1,
          "M2 frontier did not release M1/VIA1");
  const ac::StageCensus m3 = run_stage(
      &gpu, &oracle,
      {{30, 0, 35, 4, 7, 6},
       {32, 0, 38, 4, 8, 7}},
      2, "antenna M3 checkpoint",
      (UINT64_C(1) << 6) | (UINT64_C(1) << 7),
      ac::AppendMode::consuming);
  require(m3.retained_rectangles == 1,
          "M3 frontier did not release M2/VIA2");
  const ac::StageCensus m4 = run_stage(
      &gpu, &oracle,
      {{34, 0, 36, 4, 9, 8},
       {35, 0, 39, 4, 10, 9}},
      2, "antenna M4 checkpoint",
      (UINT64_C(1) << 8) | (UINT64_C(1) << 9),
      ac::AppendMode::consuming);
  require(m4.retained_rectangles == 0,
          "final frontier did not release closed stack");

  std::vector<std::uint32_t> labels;
  require_status(
      gpu.snapshot_labels(&labels), ac::Status::success,
      "antenna final snapshot");
  require(
      std::all_of(
          labels.begin(), labels.end(),
          [](std::uint32_t label) { return label == 0; }),
      "staged bridge did not merge old components");
}

void require_failure_unchanged(
    ac::Connectivity *gpu, const ac::RectI64 *stage,
    std::uint64_t count, ac::Status expected,
    const std::string &name, std::uint64_t new_node_count,
    std::uint64_t close_domain_mask)
{
  const std::uint64_t old_count = gpu->node_count();
  std::vector<std::uint32_t> before;
  require_status(
      gpu->snapshot_labels(&before), ac::Status::success,
      name + " pre-snapshot");
  std::vector<std::uint32_t> labels = {
      UINT32_C(0x12345678), UINT32_C(0x87654321)};
  const std::vector<std::uint32_t> sentinel_labels = labels;
  ac::StageCensus census;
  census.previous_nodes = UINT64_C(0x1111222233334444);
  census.candidates_by_relation[17] =
      UINT64_C(0x5555666677778888);
  const std::uint64_t sentinel_previous =
      census.previous_nodes;
  const std::uint64_t sentinel_relation =
      census.candidates_by_relation[17];
  require_status(
      gpu->append_stage(
          stage, count, new_node_count, close_domain_mask,
          ac::InputMemory::host, &labels, &census),
      expected, name);
  require(labels == sentinel_labels,
          name + ": labels changed on failure");
  require(census.previous_nodes == sentinel_previous &&
              census.candidates_by_relation[17] ==
                  sentinel_relation,
          name + ": census changed on failure");
  require(gpu->node_count() == old_count,
          name + ": committed node count changed on failure");
  std::vector<std::uint32_t> after;
  require_status(
      gpu->snapshot_labels(&after), ac::Status::success,
      name + " post-snapshot");
  require(after == before, name + ": DSU state changed on failure");
}

void test_fail_closed_paths()
{
  ac::Config config = base_config(2, 10);
  allow(&config, 0, 0);
  allow(&config, 1, 1);
  allow(&config, 0, 1);
  config.limits.max_pair_occurrences = 2;
  ac::Connectivity gpu(config);
  Oracle oracle(config);
  run_stage(
      &gpu, &oracle, {{0, 0, 50, 50, 0, 0}},
      1, "failure baseline");

  const ac::RectI64 duplicate = {0, 0, 50, 50, 1, 0};
  require_failure_unchanged(
      &gpu, &duplicate, 1, ac::Status::capacity_exceeded,
      "pair capacity");

  const ac::RectI64 zero_width = {7, 0, 7, 10, 1, 0};
  require_failure_unchanged(
      &gpu, &zero_width, 1, ac::Status::malformed_input,
      "zero-width input");
  const ac::RectI64 bad_domain = {60, 0, 70, 10, 1, 2};
  require_failure_unchanged(
      &gpu, &bad_domain, 1, ac::Status::malformed_input,
      "invalid domain");
  ac::Config membership_config = base_config(1, 10);
  allow(&membership_config, 0, 0);
  membership_config.limits.max_memberships = 2;
  ac::Connectivity membership_gpu(membership_config);
  const ac::RectI64 broad = {0, 0, 100, 100, 0, 0};
  require_failure_unchanged(
      &membership_gpu, &broad, 1,
      ac::Status::capacity_exceeded, "membership capacity");

  ac::Config overflow_config = base_config(1, 2);
  allow(&overflow_config, 0, 0);
  overflow_config.limits.max_memberships = UINT64_MAX;
  ac::Connectivity overflow_gpu(overflow_config);
  const std::array<ac::RectI64, 2> overflowing_memberships = {{
      {INT64_MIN, 0, INT64_MAX, 1, 0, 0},
      {INT64_MIN, 0, INT64_MAX, 1, 1, 0}}};
  require_failure_unchanged(
      &overflow_gpu, overflowing_memberships.data(),
      overflowing_memberships.size(),
      ac::Status::capacity_exceeded,
      "saturating membership total", 2);

  ac::Config cell_work_config = base_config(1, 100);
  allow(&cell_work_config, 0, 0);
  cell_work_config.limits.max_pair_tests_per_cell = 2;
  ac::Connectivity cell_work_gpu(cell_work_config);
  const std::array<ac::RectI64, 3> dense_cell = {{
      {1, 1, 2, 2, 0, 0},
      {3, 1, 4, 2, 1, 0},
      {5, 1, 6, 2, 2, 0}}};
  require_failure_unchanged(
      &cell_work_gpu, dense_cell.data(), dense_cell.size(),
      ac::Status::capacity_exceeded,
      "dense-cell pair-test guard", 3);

  ac::Config byte_cap_config = base_config(1, 10);
  allow(&byte_cap_config, 0, 0);
  byte_cap_config.limits.max_device_bytes =
      byte_cap_config.limits.min_device_free_after_bytes + 4096;
  ac::Connectivity byte_cap_gpu(byte_cap_config);
  const ac::RectI64 tiny = {0, 0, 1, 1, 0, 0};
  require_failure_unchanged(
      &byte_cap_gpu, &tiny, 1, ac::Status::capacity_exceeded,
      "device-byte admission");

  ac::Config tiling_config = base_config(1, 10);
  allow(&tiling_config, 0, 0);
  ac::Connectivity overlap_gpu(tiling_config);
  const std::array<ac::RectI64, 2> overlapping_tiles = {{
      {0, 0, 10, 10, 0, 0},
      {5, 0, 15, 10, 0, 0}}};
  require_failure_unchanged(
      &overlap_gpu, overlapping_tiles.data(),
      overlapping_tiles.size(), ac::Status::malformed_input,
      "overlapping owner tiles");

  ac::Connectivity disconnected_gpu(tiling_config);
  const std::array<ac::RectI64, 2> disconnected_tiles = {{
      {0, 0, 10, 10, 0, 0},
      {30, 0, 40, 10, 0, 0}}};
  require_failure_unchanged(
      &disconnected_gpu, disconnected_tiles.data(),
      disconnected_tiles.size(), ac::Status::malformed_input,
      "disconnected owner tiles");

  ac::Connectivity missing_owner_gpu(tiling_config);
  const ac::RectI64 only_owner_zero = {0, 0, 10, 10, 0, 0};
  require_failure_unchanged(
      &missing_owner_gpu, &only_owner_zero, 1,
      ac::Status::malformed_input, "missing staged owner", 2);

  ac::Connectivity closed_gpu(tiling_config);
  Oracle closed_oracle(tiling_config);
  const ac::StageCensus closed_census = run_stage(
      &closed_gpu, &closed_oracle,
      {{0, 0, 10, 10, 0, 0}}, 1,
      "closed-domain baseline", UINT64_C(1));
  require(closed_census.retained_rectangles == 0,
          "closed domain geometry was not released");
  const ac::RectI64 reopened = {20, 0, 30, 10, 1, 0};
  require_failure_unchanged(
      &closed_gpu, &reopened, 1, ac::Status::malformed_input,
      "closed domain reopened");

  ac::Config asymmetric = base_config(2, 10);
  asymmetric.relation_rows[0] = UINT64_C(1) << 1;
  ac::Connectivity invalid(asymmetric);
  require_status(
      invalid.configuration_status(),
      ac::Status::invalid_configuration,
      "asymmetric matrix");
}

void test_device_input()
{
  ac::Config config = base_config(1, 7);
  allow(&config, 0, 0);
  ac::Connectivity gpu(config);
  const std::vector<ac::RectI64> stage = {
      {-8, -8, -1, -1, 0, 0},
      {-1, -1, 5, 5, 1, 0}};
  ac::RectI64 *device_stage = nullptr;
  require(
      cudaMalloc(
          reinterpret_cast<void **>(&device_stage),
          stage.size() * sizeof(ac::RectI64)) == cudaSuccess,
      "device-input cudaMalloc failed");
  try {
    require(
        cudaMemcpy(
            device_stage, stage.data(),
            stage.size() * sizeof(ac::RectI64),
            cudaMemcpyHostToDevice) == cudaSuccess,
        "device-input H2D failed");
    std::vector<std::uint32_t> labels;
    ac::StageCensus census;
    require_status(
        gpu.append_stage(
            device_stage, stage.size(), 2, 0,
            ac::InputMemory::device, &labels, &census),
        ac::Status::success, "device input");
    require(labels == std::vector<std::uint32_t>({0, 0}),
            "device-input labels mismatch");
  } catch (...) {
    cudaFree(device_stage);
    throw;
  }
  require(cudaFree(device_stage) == cudaSuccess,
          "device-input cudaFree failed");
}

std::vector<std::uint32_t> copy_resident_labels(
    const ac::DeviceLabelView &view)
{
  std::vector<std::uint32_t> labels(view.count);
  if (!labels.empty()) {
    require(view.labels != nullptr,
            "resident label pointer is null");
    require(
        cudaMemcpy(
            labels.data(), view.labels,
            labels.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost) == cudaSuccess,
        "resident label test D2H failed");
  }
  return labels;
}

void test_resident_label_view()
{
  ac::Config config = base_config(1, 8);
  allow(&config, 0, 0);
  ac::Connectivity gpu(config);
  const std::array<ac::RectI64, 2> first_stage = {{
      {0, 0, 10, 10, 0, 0},
      {10, 10, 20, 20, 1, 0}}};
  ac::StageCensus census;
  require_status(
      gpu.append_stage(
          first_stage.data(), first_stage.size(), 2, 0,
          ac::InputMemory::host, nullptr, &census),
      ac::Status::success, "resident null-label append");
  ac::DeviceLabelView first_view;
  require_status(
      gpu.device_label_view(&first_view), ac::Status::success,
      "resident first view");
  require(first_view.count == 2 && first_view.epoch == 1 &&
              first_view.device == 0,
          "resident first view metadata mismatch");
  require(copy_resident_labels(first_view) ==
              std::vector<std::uint32_t>({0, 0}),
          "resident first labels mismatch");

  const ac::RectI64 malformed = {30, 0, 30, 10, 2, 0};
  census.previous_nodes = UINT64_C(0xabcddcba);
  require_status(
      gpu.append_stage(
          &malformed, 1, 1, 0, ac::InputMemory::host,
          nullptr, &census),
      ac::Status::malformed_input,
      "resident failed null-label append");
  require(census.previous_nodes == UINT64_C(0xabcddcba),
          "resident failure changed census");
  ac::DeviceLabelView after_failure;
  require_status(
      gpu.device_label_view(&after_failure),
      ac::Status::success, "resident view after failure");
  require(after_failure.labels == first_view.labels &&
              after_failure.count == first_view.count &&
              after_failure.epoch == first_view.epoch,
          "failed append invalidated resident view");
  require(copy_resident_labels(after_failure) ==
              std::vector<std::uint32_t>({0, 0}),
          "failed append changed resident labels");

  const ac::RectI64 third = {40, 0, 50, 10, 2, 0};
  require_status(
      gpu.append_stage(
          &third, 1, 1, 0, ac::InputMemory::host, nullptr,
          &census),
      ac::Status::success, "resident second null-label append");
  ac::DeviceLabelView second_view;
  require_status(
      gpu.device_label_view(&second_view), ac::Status::success,
      "resident second view");
  require(second_view.count == 3 && second_view.epoch == 2,
          "successful append did not advance resident epoch");
  require(copy_resident_labels(second_view) ==
              std::vector<std::uint32_t>({0, 0, 2}),
          "resident second labels mismatch");
}

void test_consuming_device_ownership()
{
  ac::Config config = base_config(1, 8);
  allow(&config, 0, 0);
  ac::Connectivity gpu(config);
  const std::vector<ac::RectI64> first_host = {
      {0, 0, 10, 10, 0, 0},
      {10, 0, 20, 10, 1, 0}};
  thrust::device_vector<ac::RectI64> first_device;
  first_device.reserve(16);
  first_device.assign(first_host.begin(), first_host.end());
  require(first_device.capacity() >= 16,
          "owned input did not establish planned capacity");
  ac::StageCensus census;
  require_status(
      gpu.append_stage_consuming(
          std::move(first_device), 2, 0, nullptr, &census),
      ac::Status::success, "owned first stage");
  require(first_device.empty(),
          "owned first-stage vector was not consumed");
  require(census.retained_rectangle_capacity >= 16,
          "owned first-stage reserve was not preserved");

  thrust::device_vector<ac::RectI64> second_device(
      1, ac::RectI64{40, 0, 50, 10, 2, 0});
  require_status(
      gpu.append_stage_consuming(
          std::move(second_device), 1, 0, nullptr, &census),
      ac::Status::success, "owned second stage");
  require(second_device.empty(),
          "owned later-stage vector was not released");
  require(census.retained_rectangle_capacity >= 16,
          "later stage discarded planned rectangle reserve");
  ac::DeviceLabelView view;
  require_status(
      gpu.device_label_view(&view), ac::Status::success,
      "owned resident view");
  require(copy_resident_labels(view) ==
              std::vector<std::uint32_t>({0, 0, 2}),
          "owned consuming labels mismatch");

  ac::Connectivity poisoned(config);
  thrust::device_vector<ac::RectI64> malformed(
      1, ac::RectI64{0, 0, 0, 10, 0, 0});
  census.previous_nodes = UINT64_C(0x13579bdf);
  require_status(
      poisoned.append_stage_consuming(
          std::move(malformed), 1, 0, nullptr, &census),
      ac::Status::malformed_input,
      "owned consuming failure");
  require(malformed.empty(),
          "failed owned stage was not consumed");
  require(census.previous_nodes == UINT64_C(0x13579bdf),
          "failed owned stage changed census");
  require_status(
      poisoned.configuration_status(),
      ac::Status::poisoned_state,
      "failed consuming state poison");
  ac::DeviceLabelView poisoned_view;
  require_status(
      poisoned.device_label_view(&poisoned_view),
      ac::Status::poisoned_state,
      "poisoned resident view");
}

void test_seeded_random_differentials(bool exact_filter)
{
  constexpr std::uint32_t kSeeds = 12;
  constexpr std::uint32_t kStages = 4;
  constexpr std::uint32_t kRectanglesPerStage = 18;
  for (std::uint32_t seed = 0; seed < kSeeds; ++seed) {
    ac::Config config = base_config(6, 7);
    config.exact_filter_before_materialization = exact_filter;
    for (std::uint32_t domain = 0; domain < 6; ++domain) {
      allow(&config, domain, domain);
      if (domain + 1 < 6) allow(&config, domain, domain + 1);
    }
    ac::Connectivity gpu(config);
    Oracle oracle(config);
    std::mt19937_64 random(
        UINT64_C(0xa17e44c0ffee0000) + seed);
    std::uniform_int_distribution<std::int64_t>
        coordinate(-70, 70);
    std::uniform_int_distribution<std::int64_t> extent(1, 18);
    std::uniform_int_distribution<std::uint32_t> domain(0, 5);

    std::vector<ac::RectI64> prior;
    for (std::uint32_t stage_id = 0; stage_id < kStages;
         ++stage_id) {
      std::vector<ac::RectI64> stage;
      stage.reserve(kRectanglesPerStage);
      for (std::uint32_t index = 0;
           index < kRectanglesPerStage; ++index) {
        ac::RectI64 rectangle;
        const std::uint32_t owner =
            static_cast<std::uint32_t>(gpu.node_count()) + index;
        if (!prior.empty() && index % 9 == 0) {
          rectangle = prior[random() % prior.size()];
          rectangle.owner = owner;
        } else if (!prior.empty() && index % 5 == 0) {
          const ac::RectI64 &anchor =
              prior[random() % prior.size()];
          rectangle.left = anchor.right;
          rectangle.bottom = anchor.top;
          rectangle.right = rectangle.left + extent(random);
          rectangle.top = rectangle.bottom + extent(random);
          rectangle.owner = owner;
          rectangle.domain = anchor.domain;
        } else {
          rectangle.left = coordinate(random);
          rectangle.bottom = coordinate(random);
          rectangle.right = rectangle.left + extent(random);
          rectangle.top = rectangle.bottom + extent(random);
          rectangle.owner = owner;
          rectangle.domain = domain(random);
        }
        stage.push_back(rectangle);
      }
      const std::string name =
          std::string(exact_filter ? "exact " : "broad ") +
          "random seed " + std::to_string(seed) +
          " stage " + std::to_string(stage_id);
      run_stage(
          &gpu, &oracle, stage, kRectanglesPerStage, name);
      prior.insert(prior.end(), stage.begin(), stage.end());
    }
  }
}

}  // namespace

int main()
{
  try {
    int devices = 0;
    require(
        cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0,
        "no CUDA device available");
    test_touching_and_relation_census();
    test_multibin_deduplication();
    test_exact_filter_before_materialization();
    test_concave_owner_rectangulation();
    test_staged_antenna_bridge(false);
    test_staged_antenna_bridge(true);
    test_fail_closed_paths();
    test_device_input();
    test_resident_label_view();
    test_consuming_device_ownership();
    test_seeded_random_differentials(false);
    test_seeded_random_differentials(true);
    std::cout
        << "antenna_connectivity_gpu_test: PASS"
        << " directed=13 random_seeds=24 stages_per_seed=4"
        << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "antenna_connectivity_gpu_test: FAIL: "
              << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}
