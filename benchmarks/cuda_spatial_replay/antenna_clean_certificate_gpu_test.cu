/*
 * Directed and seeded CPU differentials for conservative antenna clean
 * certificates.
 */

#include "antenna_clean_certificate_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace acc = klayout_cuda::antenna_clean_certificate;
namespace ac = klayout_cuda::antenna_connectivity;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const std::string &operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
        operation + ": " + cudaGetErrorString(status));
  }
}

void require_status(
    acc::Status actual, acc::Status expected,
    const std::string &operation)
{
  if (actual != expected) {
    throw std::runtime_error(
        operation + ": expected " + acc::status_string(expected) +
        ", got " + acc::status_string(actual));
  }
}

ac::RectI64 rectangle(
    std::int64_t left, std::int64_t bottom,
    std::int64_t right, std::int64_t top,
    std::uint32_t owner, std::uint32_t domain = 0)
{
  return {left, bottom, right, top, owner, domain};
}

template <class T>
class DeviceArray
{
public:
  DeviceArray() = default;

  explicit DeviceArray(const std::vector<T> &values)
  {
    assign(values);
  }

  ~DeviceArray()
  {
    if (m_pointer) cudaFree(m_pointer);
  }

  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;

  void assign(const std::vector<T> &values)
  {
    if (m_pointer) cudaFree(m_pointer);
    m_pointer = nullptr;
    m_count = values.size();
    if (values.empty()) return;
    require_cuda(
        cudaMalloc(
            reinterpret_cast<void **>(&m_pointer),
            values.size() * sizeof(T)),
        "allocate test device array");
    require_cuda(
        cudaMemcpy(
            m_pointer, values.data(), values.size() * sizeof(T),
            cudaMemcpyHostToDevice),
        "upload test device array");
  }

  T *get() const
  {
    return m_pointer;
  }

  std::uint64_t size() const
  {
    return m_count;
  }

private:
  T *m_pointer = nullptr;
  std::uint64_t m_count = 0;
};

bool checked_multiply(
    std::uint64_t first, std::uint64_t second,
    std::uint64_t *result)
{
  if (first && second > UINT64_MAX / first) return false;
  *result = first * second;
  return true;
}

bool checked_add(
    std::uint64_t first, std::uint64_t second,
    std::uint64_t *result)
{
  if (first > UINT64_MAX - second) return false;
  *result = first + second;
  return true;
}

bool tile_area(const ac::RectI64 &tile, std::uint64_t *area)
{
  if (tile.left >= tile.right || tile.bottom >= tile.top) {
    return false;
  }
  const std::uint64_t width =
      static_cast<std::uint64_t>(tile.right) -
      static_cast<std::uint64_t>(tile.left);
  const std::uint64_t height =
      static_cast<std::uint64_t>(tile.top) -
      static_cast<std::uint64_t>(tile.bottom);
  return checked_multiply(width, height, area);
}

bool intersection_area(
    const ac::RectI64 &poly, const ac::RectI64 &active,
    std::uint64_t *area)
{
  const std::int64_t left = std::max(poly.left, active.left);
  const std::int64_t bottom =
      std::max(poly.bottom, active.bottom);
  const std::int64_t right =
      std::min(poly.right, active.right);
  const std::int64_t top = std::min(poly.top, active.top);
  if (left >= right || bottom >= top) {
    *area = 0;
    return true;
  }
  const std::uint64_t width =
      static_cast<std::uint64_t>(right) -
      static_cast<std::uint64_t>(left);
  const std::uint64_t height =
      static_cast<std::uint64_t>(top) -
      static_cast<std::uint64_t>(bottom);
  return checked_multiply(width, height, area);
}

struct GateOracle
{
  std::vector<std::uint32_t> present;
  std::vector<std::uint64_t> lower;
  std::uint64_t positive_intersections = 0;
};

GateOracle gate_oracle(
    const std::vector<ac::RectI64> &poly,
    const std::vector<ac::RectI64> &active,
    std::uint64_t owner_count)
{
  GateOracle result;
  result.present.assign(owner_count, 0);
  result.lower.assign(owner_count, 0);
  for (const ac::RectI64 &poly_tile : poly) {
    require(
        poly_tile.owner < owner_count,
        "CPU oracle POLY owner out of range");
    for (const ac::RectI64 &active_tile : active) {
      std::uint64_t area = 0;
      require(
          intersection_area(poly_tile, active_tile, &area),
          "CPU oracle gate area overflow");
      if (!area) continue;
      result.present[poly_tile.owner] = 1;
      result.lower[poly_tile.owner] =
          std::max(result.lower[poly_tile.owner], area);
      ++result.positive_intersections;
    }
  }
  return result;
}

acc::CheckpointCensus checkpoint_oracle(
    acc::MetalLevel level, const GateOracle &gate,
    const std::vector<ac::RectI64> &metal,
    const std::vector<std::uint32_t> &labels)
{
  require(
      labels.size() >= gate.present.size(),
      "CPU oracle labels too short");
  std::vector<std::uint32_t> root_present(labels.size(), 0);
  std::vector<std::uint64_t> root_lower(labels.size(), 0);
  std::vector<std::uint64_t> root_metal(labels.size(), 0);
  for (std::uint64_t owner = 0; owner < gate.present.size();
       ++owner) {
    if (!gate.present[owner]) continue;
    const std::uint32_t root = labels[owner];
    root_present[root] = 1;
    root_lower[root] =
        std::max(root_lower[root], gate.lower[owner]);
  }
  for (const ac::RectI64 &tile : metal) {
    require(tile.owner < labels.size(), "CPU metal owner out of range");
    std::uint64_t area = 0;
    require(tile_area(tile, &area), "CPU metal area overflow");
    const std::uint32_t root = labels[tile.owner];
    require(
        checked_add(root_metal[root], area, &root_metal[root]),
        "CPU metal sum overflow");
  }

  acc::CheckpointCensus result;
  result.level = level;
  result.labels = labels.size();
  result.metal_tiles = metal.size();
  for (std::uint64_t root = 0; root < labels.size(); ++root) {
    if (labels[root] != root) continue;
    ++result.roots;
    const std::uint64_t metal_area = root_metal[root];
    if (metal_area) ++result.roots_with_metal;
    if (!root_present[root]) {
      ++result.roots_without_gate;
    } else if (!metal_area) {
      ++result.gate_roots_without_metal;
    } else if (root_lower[root] <= 1) {
      ++result.uncertain_roots;
    } else {
      require(
          root_lower[root] <= UINT64_MAX / 300,
          "CPU ratio multiplication overflow");
      if (metal_area <= 300 * root_lower[root]) {
        ++result.ratio_certified_roots;
      } else {
        ++result.uncertain_roots;
      }
    }
  }
  result.clean_certificate = result.uncertain_roots == 0;
  return result;
}

std::int64_t floor_div(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

acc::CheckpointCensus root_cell_checkpoint_oracle(
    acc::MetalLevel level, const GateOracle &gate,
    const std::vector<ac::RectI64> &poly,
    const std::vector<ac::RectI64> &active,
    const std::vector<ac::RectI64> &metal,
    const std::vector<std::uint32_t> &labels,
    std::int64_t cell_size)
{
  std::vector<std::uint32_t> root_present(labels.size(), 0);
  std::vector<std::uint64_t> root_lower(labels.size(), 0);
  std::vector<std::uint64_t> root_metal(labels.size(), 0);
  for (std::uint64_t owner = 0; owner < gate.present.size();
       ++owner) {
    if (!gate.present[owner]) continue;
    const std::uint32_t root = labels[owner];
    root_present[root] = 1;
    root_lower[root] =
        std::max(root_lower[root], gate.lower[owner]);
  }
  for (const ac::RectI64 &tile : metal) {
    std::uint64_t area = 0;
    require(tile_area(tile, &area), "root-cell metal area");
    const std::uint32_t root = labels[tile.owner];
    require(
        checked_add(root_metal[root], area, &root_metal[root]),
        "root-cell metal sum");
  }

  std::vector<std::uint32_t> refine(labels.size(), 0);
  for (std::uint64_t root = 0; root < labels.size(); ++root) {
    if (labels[root] != root || !root_present[root] ||
        !root_metal[root]) {
      continue;
    }
    const std::uint64_t lower = root_lower[root];
    if (lower <= 1 ||
        root_metal[root] > 300 * lower) {
      refine[root] = 1;
      root_lower[root] = 0;
    }
  }

  using CellKey =
      std::tuple<std::uint32_t, std::int64_t, std::int64_t>;
  std::map<CellKey, std::uint64_t> cell_lower;
  for (const ac::RectI64 &poly_tile : poly) {
    const std::uint32_t root = labels[poly_tile.owner];
    if (!refine[root]) continue;
    for (const ac::RectI64 &active_tile : active) {
      const std::int64_t left =
          std::max(poly_tile.left, active_tile.left);
      const std::int64_t right =
          std::min(poly_tile.right, active_tile.right);
      const std::int64_t bottom =
          std::max(poly_tile.bottom, active_tile.bottom);
      const std::int64_t top =
          std::min(poly_tile.top, active_tile.top);
      if (left >= right || bottom >= top) continue;
      const std::int64_t x0 = floor_div(left, cell_size);
      const std::int64_t x1 = floor_div(right - 1, cell_size);
      const std::int64_t y0 = floor_div(bottom, cell_size);
      const std::int64_t y1 = floor_div(top - 1, cell_size);
      for (std::int64_t y = y0; y <= y1; ++y) {
        for (std::int64_t x = x0; x <= x1; ++x) {
          const std::int64_t clipped_left =
              std::max(left, x * cell_size);
          const std::int64_t clipped_right =
              std::min(right, (x + 1) * cell_size);
          const std::int64_t clipped_bottom =
              std::max(bottom, y * cell_size);
          const std::int64_t clipped_top =
              std::min(top, (y + 1) * cell_size);
          const std::uint64_t area =
              static_cast<std::uint64_t>(
                  clipped_right - clipped_left) *
              static_cast<std::uint64_t>(
                  clipped_top - clipped_bottom);
          CellKey key(root, x, y);
          cell_lower[key] =
              std::max(cell_lower[key], area);
        }
      }
    }
  }
  for (const auto &entry : cell_lower) {
    const std::uint32_t root = std::get<0>(entry.first);
    require(
        checked_add(
            root_lower[root], entry.second, &root_lower[root]),
        "root-cell lower sum");
  }

  acc::CheckpointCensus result;
  result.level = level;
  result.labels = labels.size();
  result.metal_tiles = metal.size();
  for (std::uint64_t root = 0; root < labels.size(); ++root) {
    if (labels[root] != root) continue;
    ++result.roots;
    if (root_metal[root]) ++result.roots_with_metal;
    if (!root_present[root]) {
      ++result.roots_without_gate;
    } else if (!root_metal[root]) {
      ++result.gate_roots_without_metal;
    } else if (
        root_lower[root] > 1 &&
        root_metal[root] <= 300 * root_lower[root]) {
      ++result.ratio_certified_roots;
    } else {
      ++result.uncertain_roots;
    }
  }
  result.clean_certificate = result.uncertain_roots == 0;
  return result;
}

void compare_checkpoint(
    const acc::CheckpointCensus &actual,
    const acc::CheckpointCensus &expected,
    const std::string &name)
{
  require(actual.level == expected.level, name + ": level");
  require(
      actual.clean_certificate == expected.clean_certificate,
      name + ": clean disposition");
  require(actual.labels == expected.labels, name + ": labels");
  require(
      actual.metal_tiles == expected.metal_tiles,
      name + ": metal tiles");
  require(actual.roots == expected.roots, name + ": roots");
  require(
      actual.roots_with_metal == expected.roots_with_metal,
      name + ": roots with metal");
  require(
      actual.roots_without_gate == expected.roots_without_gate,
      name + ": roots without gate");
  require(
      actual.gate_roots_without_metal ==
          expected.gate_roots_without_metal,
      name + ": gate roots without metal");
  require(
      actual.diode_exempt_roots ==
          expected.diode_exempt_roots,
      name + ": diode exempt roots");
  require(
      actual.ratio_certified_roots ==
          expected.ratio_certified_roots,
      name + ": ratio roots");
  require(
      actual.uncertain_roots == expected.uncertain_roots,
      name + ": uncertain roots");
}

acc::Config base_config()
{
  acc::Config config;
  config.device = 0;
  config.grid_cell_size = 10;
  config.limits.max_live_device_bytes =
      UINT64_C(256) * 1024 * 1024;
  config.limits.max_annotation_owners = 100000;
  config.limits.max_labels = 100000;
  config.limits.max_poly_tiles = 100000;
  config.limits.max_active_tiles = 100000;
  config.limits.max_metal_tiles = 100000;
  config.limits.max_grid_cells = 1000000;
  config.limits.max_active_memberships = 1000000;
  config.limits.max_query_visits = 10000000;
  config.limits.max_cell_members = 10000;
  return config;
}

void test_external_live_update_is_transactional()
{
  acc::Config config = base_config();
  config.limits.max_live_device_bytes = 1024;
  config.external_live_device_bytes = 64;
  acc::Certificate certificate(config);
  require_status(
      certificate.configuration_status(), acc::Status::success,
      "external baseline configuration");
  require_status(
      certificate.set_external_live_device_bytes(512),
      acc::Status::success, "external baseline update");
  require_status(
      certificate.set_external_live_device_bytes(2048),
      acc::Status::capacity_exceeded,
      "external baseline over-cap update");
  require_status(
      certificate.set_external_live_device_bytes(512),
      acc::Status::success,
      "external baseline remains usable after rejection");
}

void require_gate_view(
    acc::Certificate *certificate, const GateOracle &expected,
    const std::string &name)
{
  acc::GateAnnotationDeviceView view;
  require_status(
      certificate->device_gate_view(&view),
      acc::Status::success, name + " view");
  require(
      view.count == expected.present.size(),
      name + ": annotation count");
  std::vector<std::uint32_t> present(view.count);
  std::vector<std::uint64_t> lower(view.count);
  require_cuda(
      cudaMemcpy(
          present.data(), view.gate_present,
          present.size() * sizeof(present[0]),
          cudaMemcpyDeviceToHost),
      name + " presence D2H");
  require_cuda(
      cudaMemcpy(
          lower.data(), view.max_single_intersection_area,
          lower.size() * sizeof(lower[0]),
          cudaMemcpyDeviceToHost),
      name + " lower D2H");
  require(present == expected.present, name + ": gate presence");
  require(lower == expected.lower, name + ": gate lower");
}

acc::CheckpointCensus run_checkpoint(
    acc::Certificate *certificate, acc::MetalLevel level,
    const std::vector<ac::RectI64> &metal,
    const std::vector<std::uint32_t> &labels,
    const std::string &name)
{
  DeviceArray<ac::RectI64> device_metal(metal);
  DeviceArray<std::uint32_t> device_labels(labels);
  acc::CheckpointCensus result;
  require_status(
      certificate->evaluate_checkpoint(
          level, device_metal.get(), device_metal.size(),
          device_labels.get(), device_labels.size(), &result),
      acc::Status::success, name);
  require(
      result.persistent_bytes > 0 &&
          result.peak_live_bytes >= result.persistent_bytes &&
          result.peak_temporary_bytes > 0,
      name + ": memory accounting");
  return result;
}

acc::CheckpointCensus run_root_cell_checkpoint(
    acc::Certificate *certificate, acc::MetalLevel level,
    const std::vector<ac::RectI64> &poly,
    const std::vector<ac::RectI64> &active,
    const std::vector<ac::RectI64> &metal,
    const std::vector<std::uint32_t> &labels,
    const std::string &name)
{
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);
  DeviceArray<ac::RectI64> device_metal(metal);
  DeviceArray<std::uint32_t> device_labels(labels);
  acc::CheckpointCensus preliminary;
  require_status(
      certificate->evaluate_checkpoint(
          level, device_metal.get(), device_metal.size(),
          device_labels.get(), device_labels.size(),
          &preliminary),
      acc::Status::success, name + " preliminary");
  acc::CheckpointCensus result;
  require_status(
      certificate->evaluate_checkpoint_root_cell_refined(
          level, device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(),
          device_metal.get(), device_metal.size(),
          device_labels.get(), device_labels.size(), &result,
          nullptr, &preliminary),
      acc::Status::success, name);
  require(
      result.preliminary_uncertain_roots > 0 &&
          result.refinement_records > 0 &&
          result.refinement_candidate_visits > 0,
      name + ": refinement census");
  return result;
}

void test_cross_context_tile_splits_and_touches()
{
  const std::vector<ac::RectI64> poly = {
      rectangle(-15, -10, -5, 0, 0, 7),
      rectangle(-5, -10, 5, 0, 0, 19),
      rectangle(20, 0, 30, 10, 1, 3),
      rectangle(40, 0, 50, 10, 2, 31)};
  const std::vector<ac::RectI64> active = {
      // Cross-context-style owner/domain and negative coordinates.
      rectangle(-8, -8, 2, -2, 3, 44),
      // Edge touch only against POLY owner 1.
      rectangle(30, 0, 35, 10, 4, 5),
      // Corner touch only against POLY owner 2.
      rectangle(50, 10, 55, 15, 5, 6)};
  const GateOracle oracle = gate_oracle(poly, active, 8);
  require(oracle.present[0] == 1, "cross-context overlap missing");
  require(oracle.lower[0] == 42, "tile-split max lower");
  require(oracle.present[1] == 0, "edge touch became gate");
  require(oracle.present[2] == 0, "corner touch became gate");

  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);
  acc::Certificate certificate(base_config());
  acc::GateCensus census;
  require_status(
      certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(), 8, &census),
      acc::Status::success, "cross-context gate census");
  require(
      census.positive_intersections ==
          oracle.positive_intersections,
      "cross-context positive census");
  require(census.gate_owners == 1, "cross-context gate owners");
  require(census.grid_cells > 0, "spatial grid absent");
  require_gate_view(&certificate, oracle, "cross-context");

  const std::vector<std::uint32_t> labels =
      {0, 1, 2, 3, 4, 5, 0, 7};
  const std::vector<ac::RectI64> metal = {
      rectangle(100, 0, 110, 10, 6)};
  const acc::CheckpointCensus actual = run_checkpoint(
      &certificate, acc::MetalLevel::metal1, metal, labels,
      "cross-context checkpoint");
  compare_checkpoint(
      actual,
      checkpoint_oracle(
          acc::MetalLevel::metal1, oracle, metal, labels),
      "cross-context checkpoint");
  require(actual.clean_certificate, "safe ratio should certify");
}

void test_one_dbu_boundary_and_duplicate_metal_upper()
{
  {
    const std::vector<ac::RectI64> poly = {
        rectangle(0, 0, 10, 10, 0)};
    const std::vector<ac::RectI64> active = {
        rectangle(9, 9, 20, 20, 1)};
    const GateOracle oracle = gate_oracle(poly, active, 4);
    require(oracle.lower[0] == 1, "one-DBU lower bound");
    DeviceArray<ac::RectI64> device_poly(poly);
    DeviceArray<ac::RectI64> device_active(active);
    acc::Certificate certificate(base_config());
    acc::GateCensus gate_census;
    require_status(
        certificate.build_gate_census(
            device_poly.get(), device_poly.size(),
            device_active.get(), device_active.size(), 2,
            &gate_census),
        acc::Status::success, "one-DBU build");
    const std::vector<std::uint32_t> labels = {0, 1, 0, 3};
    const std::vector<ac::RectI64> metal = {
        rectangle(30, 0, 31, 1, 2)};
    const acc::CheckpointCensus actual = run_checkpoint(
        &certificate, acc::MetalLevel::metal1, metal, labels,
        "one-DBU checkpoint");
    require(
        !actual.clean_certificate && actual.uncertain_roots == 1,
        "one-DBU gate with metal must be uncertain");
  }

  {
    const std::vector<ac::RectI64> poly = {
        rectangle(0, 0, 2, 2, 0)};
    const std::vector<ac::RectI64> active = {
        rectangle(0, 0, 2, 1, 1)};
    const GateOracle oracle = gate_oracle(poly, active, 3);
    require(oracle.lower[0] == 2, "duplicate test lower");
    DeviceArray<ac::RectI64> device_poly(poly);
    DeviceArray<ac::RectI64> device_active(active);
    acc::Certificate certificate(base_config());
    acc::GateCensus gate_census;
    require_status(
        certificate.build_gate_census(
            device_poly.get(), device_poly.size(),
            device_active.get(), device_active.size(), 2,
            &gate_census),
        acc::Status::success, "duplicate metal build");
    const std::vector<std::uint32_t> labels = {0, 1, 0};
    const ac::RectI64 duplicate =
        rectangle(20, 0, 40, 20, 2);
    const std::vector<ac::RectI64> metal =
        {duplicate, duplicate};
    const acc::CheckpointCensus actual = run_checkpoint(
        &certificate, acc::MetalLevel::metal2, metal, labels,
        "duplicate metal checkpoint");
    require(
        !actual.clean_certificate && actual.uncertain_roots == 1,
        "overlapping metal must be summed as safe upper bound");
  }
}

void test_merged_roots_use_max_gate_lower()
{
  const std::vector<ac::RectI64> poly = {
      rectangle(0, 0, 2, 2, 0),
      rectangle(20, 0, 22, 2, 1)};
  const std::vector<ac::RectI64> active = {
      rectangle(0, 0, 2, 1, 2),
      rectangle(20, 0, 22, 1, 3)};
  const GateOracle oracle = gate_oracle(poly, active, 5);
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);
  acc::Certificate certificate(base_config());
  acc::GateCensus gate_census;
  require_status(
      certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(), 4,
          &gate_census),
      acc::Status::success, "merged-root build");
  const std::vector<std::uint32_t> labels = {0, 0, 2, 3, 0};
  const std::vector<ac::RectI64> metal = {
      rectangle(100, 0, 107, 100, 4)};
  const acc::CheckpointCensus actual = run_checkpoint(
      &certificate, acc::MetalLevel::metal3, metal, labels,
      "merged-root checkpoint");
  require(
      !actual.clean_certificate && actual.uncertain_roots == 1,
      "merged roots must max, not sum, gate lower bounds");
  compare_checkpoint(
      actual,
      checkpoint_oracle(
          acc::MetalLevel::metal3, oracle, metal, labels),
      "merged-root checkpoint");
}

void test_root_cell_refinement_is_disjoint_and_fail_closed()
{
  {
    const std::vector<ac::RectI64> poly = {
        rectangle(-20, 0, -16, 4, 0),
        rectangle(0, 0, 4, 4, 1)};
    const std::vector<ac::RectI64> active = {
        rectangle(-20, 0, -16, 4, 2),
        // A raw duplicate must not add area within one cell.
        rectangle(-20, 0, -16, 4, 2),
        rectangle(0, 0, 4, 4, 3)};
    const std::vector<std::uint32_t> labels =
        {0, 0, 2, 3, 0, 5};
    const std::vector<ac::RectI64> metal = {
        rectangle(100, 0, 160, 100, 4)};
    const GateOracle gate = gate_oracle(poly, active, 4);
    DeviceArray<ac::RectI64> device_poly(poly);
    DeviceArray<ac::RectI64> device_active(active);
    acc::Certificate certificate(base_config());
    acc::GateCensus gate_census;
    require_status(
        certificate.build_gate_census(
            device_poly.get(), device_poly.size(),
            device_active.get(), device_active.size(), 4,
            &gate_census),
        acc::Status::success, "root-cell disjoint build");
    const acc::CheckpointCensus preliminary = run_checkpoint(
        &certificate, acc::MetalLevel::metal1, metal, labels,
        "root-cell disjoint preliminary");
    require(
        !preliminary.clean_certificate &&
            preliminary.uncertain_roots == 1,
        "single-fragment lower unexpectedly certified");
    const acc::CheckpointCensus refined =
        run_root_cell_checkpoint(
            &certificate, acc::MetalLevel::metal1, poly, active,
            metal, labels, "root-cell disjoint refined");
    const acc::CheckpointCensus expected =
        root_cell_checkpoint_oracle(
            acc::MetalLevel::metal1, gate, poly, active, metal,
            labels, base_config().grid_cell_size);
    compare_checkpoint(refined, expected, "root-cell disjoint");
    require(
        refined.clean_certificate &&
            refined.preliminary_uncertain_roots == 1 &&
            refined.refinement_root_cells == 2,
        "disjoint root cells did not strengthen the lower bound");
  }

  {
    const std::vector<ac::RectI64> poly = {
        rectangle(0, 0, 4, 4, 0),
        // Overlapping POLY owners belong to one connectivity root.
        rectangle(0, 0, 4, 4, 1)};
    const std::vector<ac::RectI64> active = {
        rectangle(0, 0, 4, 4, 2),
        rectangle(0, 0, 4, 4, 3)};
    const std::vector<std::uint32_t> labels = {0, 0, 2, 3, 0};
    const std::vector<ac::RectI64> metal = {
        rectangle(100, 0, 160, 100, 4)};
    const GateOracle gate = gate_oracle(poly, active, 4);
    DeviceArray<ac::RectI64> device_poly(poly);
    DeviceArray<ac::RectI64> device_active(active);
    acc::Certificate certificate(base_config());
    acc::GateCensus gate_census;
    require_status(
        certificate.build_gate_census(
            device_poly.get(), device_poly.size(),
            device_active.get(), device_active.size(), 4,
            &gate_census),
        acc::Status::success, "root-cell overlap build");
    const acc::CheckpointCensus refined =
        run_root_cell_checkpoint(
            &certificate, acc::MetalLevel::metal1, poly, active,
            metal, labels, "root-cell overlap refined");
    const acc::CheckpointCensus expected =
        root_cell_checkpoint_oracle(
            acc::MetalLevel::metal1, gate, poly, active, metal,
            labels, base_config().grid_cell_size);
    compare_checkpoint(refined, expected, "root-cell overlap");
    require(
        !refined.clean_certificate &&
            refined.uncertain_roots == 1 &&
            refined.refinement_root_cells == 1,
        "overlapping raw shapes inflated a root-cell lower bound");

    acc::Config limited_config = base_config();
    limited_config.limits.max_refinement_records = 1;
    acc::Certificate limited(limited_config);
    require_status(
        limited.build_gate_census(
            device_poly.get(), device_poly.size(),
            device_active.get(), device_active.size(), 4,
            &gate_census),
        acc::Status::success, "root-cell capacity build");
    DeviceArray<ac::RectI64> device_metal(metal);
    DeviceArray<std::uint32_t> device_labels(labels);
    acc::CheckpointCensus sentinel;
    sentinel.labels = UINT64_C(0x12345678);
    sentinel.uncertain_roots = UINT64_C(0x87654321);
    const acc::CheckpointCensus sentinel_expected = sentinel;
    require_status(
        limited.evaluate_checkpoint_root_cell_refined(
            acc::MetalLevel::metal1, device_poly.get(),
            device_poly.size(), device_active.get(),
            device_active.size(), device_metal.get(),
            device_metal.size(), device_labels.get(),
            device_labels.size(), &sentinel),
        acc::Status::capacity_exceeded,
        "root-cell exact record cap");
    require(
        std::memcmp(
            &sentinel, &sentinel_expected, sizeof(sentinel)) == 0,
        "root-cell capacity failure changed output");
  }
}

void test_factor_zero_diode_exemption_is_annotation_only()
{
  const std::vector<ac::RectI64> poly = {
      rectangle(0, 0, 2, 2, 0)};
  const std::vector<ac::RectI64> active = {
      rectangle(0, 0, 2, 2, 1)};
  const std::vector<ac::RectI64> metal = {
      rectangle(100, 0, 200, 100, 2)};
  const std::vector<std::uint32_t> labels = {0, 1, 0, 0};
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);
  DeviceArray<ac::RectI64> device_metal(metal);
  DeviceArray<std::uint32_t> device_labels(labels);
  DeviceArray<std::uint32_t> device_diode(
      std::vector<std::uint32_t>{1, 1});

  acc::Certificate certificate(base_config());
  acc::GateCensus gate_census;
  require_status(
      certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(), 2,
          &gate_census),
      acc::Status::success, "factor-zero diode build");
  const acc::CheckpointCensus baseline = run_checkpoint(
      &certificate, acc::MetalLevel::metal1, metal, labels,
      "factor-zero diode baseline");
  require(
      baseline.uncertain_roots == 1 &&
          !baseline.clean_certificate,
      "factor-zero diode fixture did not begin uncertain");

  acc::FactorZeroDiodeDeviceView diode_view;
  diode_view.contact_present = device_diode.get();
  diode_view.owner_begin = 2;
  diode_view.count = 2;
  acc::CheckpointCensus exempt;
  require_status(
      certificate.evaluate_checkpoint(
          acc::MetalLevel::metal1, device_metal.get(),
          device_metal.size(), device_labels.get(),
          device_labels.size(), &exempt, &diode_view),
      acc::Status::success, "factor-zero diode checkpoint");
  require(
      exempt.clean_certificate &&
          exempt.diode_exempt_roots == 1 &&
          exempt.uncertain_roots == 0 &&
          exempt.ratio_certified_roots == 0 &&
          exempt.roots == baseline.roots,
      "factor-zero diode did not exempt exactly one root");

  acc::CheckpointCensus refined;
  require_status(
      certificate.evaluate_checkpoint_root_cell_refined(
          acc::MetalLevel::metal1, device_poly.get(),
          device_poly.size(), device_active.get(),
          device_active.size(), device_metal.get(),
          device_metal.size(), device_labels.get(),
          device_labels.size(), &refined, &diode_view, &exempt),
      acc::Status::success,
      "factor-zero diode refined checkpoint");
  require(
      refined.clean_certificate &&
          refined.diode_exempt_roots == 1 &&
          refined.preliminary_uncertain_roots == 0 &&
          refined.refinement_records == 0,
      "factor-zero diode should avoid unnecessary refinement");

  acc::CheckpointCensus mismatched_preliminary = exempt;
  --mismatched_preliminary.labels;
  acc::CheckpointCensus reuse_sentinel;
  reuse_sentinel.labels = UINT64_C(0x13572468);
  const acc::CheckpointCensus reuse_sentinel_expected =
      reuse_sentinel;
  require_status(
      certificate.evaluate_checkpoint_root_cell_refined(
          acc::MetalLevel::metal1, device_poly.get(),
          device_poly.size(), device_active.get(),
          device_active.size(), device_metal.get(),
          device_metal.size(), device_labels.get(),
          device_labels.size(), &reuse_sentinel, &diode_view,
          &mismatched_preliminary),
      acc::Status::malformed_input,
      "mismatched preliminary checkpoint reuse");
  require(
      std::memcmp(
          &reuse_sentinel, &reuse_sentinel_expected,
          sizeof(reuse_sentinel)) == 0,
      "mismatched preliminary reuse changed output");

  acc::FactorZeroDiodeDeviceView malformed = diode_view;
  malformed.owner_begin = labels.size();
  malformed.count = 1;
  acc::CheckpointCensus sentinel;
  sentinel.labels = UINT64_C(0x12345678);
  const acc::CheckpointCensus sentinel_expected = sentinel;
  require_status(
      certificate.evaluate_checkpoint(
          acc::MetalLevel::metal1, device_metal.get(),
          device_metal.size(), device_labels.get(),
          device_labels.size(), &sentinel, &malformed),
      acc::Status::malformed_input,
      "factor-zero diode malformed range");
  require(
      std::memcmp(
          &sentinel, &sentinel_expected, sizeof(sentinel)) == 0,
      "factor-zero diode malformed view changed output");
}

struct HookState
{
  std::uint64_t calls = 0;
  std::uint64_t observed_peak = 0;
  std::uint64_t commits = 0;
};

void memory_hook(
    acc::MemoryEvent event,
    const acc::MemoryAccounting &accounting, void *context)
{
  HookState *state = static_cast<HookState *>(context);
  ++state->calls;
  state->observed_peak =
      std::max(state->observed_peak, accounting.peak_live_bytes);
  if (event == acc::MemoryEvent::commit) ++state->commits;
}

void test_failures_are_transactional_and_budgeted()
{
  acc::Config config = base_config();
  config.limits.max_live_device_bytes = 4096;
  HookState hook;
  config.memory_hook = memory_hook;
  config.memory_hook_context = &hook;
  acc::Certificate certificate(config);

  const std::vector<ac::RectI64> poly = {
      rectangle(0, 0, 4, 4, 0)};
  const std::vector<ac::RectI64> active = {
      rectangle(0, 0, 4, 2, 1)};
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);
  acc::GateCensus initial;
  require_status(
      certificate.build_gate_census(
          device_poly.get(), 1, device_active.get(), 1, 4,
          &initial),
      acc::Status::success, "transactional initial build");
  acc::GateAnnotationDeviceView before;
  require_status(
      certificate.device_gate_view(&before),
      acc::Status::success, "transactional initial view");
  require(
      hook.calls > 0 && hook.commits == 2 &&
          hook.observed_peak == initial.peak_live_bytes,
      "memory hook did not observe exact allocations");

  const std::vector<ac::RectI64> malformed = {
      rectangle(0, 0, 0, 4, 0)};
  DeviceArray<ac::RectI64> device_malformed(malformed);
  acc::GateCensus gate_sentinel;
  gate_sentinel.annotation_owners = 0x12345678;
  gate_sentinel.candidate_visits = 0x87654321;
  const acc::GateCensus gate_expected = gate_sentinel;
  require_status(
      certificate.build_gate_census(
          device_malformed.get(), 1, device_active.get(), 1,
          4, &gate_sentinel),
      acc::Status::malformed_input, "malformed rebuild");
  require(
      std::memcmp(
          &gate_sentinel, &gate_expected, sizeof(gate_sentinel)) == 0,
      "malformed build changed output");

  acc::GateAnnotationDeviceView after_malformed;
  require_status(
      certificate.device_gate_view(&after_malformed),
      acc::Status::success, "view after malformed build");
  require(
      after_malformed.epoch == before.epoch &&
          after_malformed.gate_present == before.gate_present &&
          after_malformed.max_single_intersection_area ==
              before.max_single_intersection_area,
      "malformed build changed persistent annotations");

  require_status(
      certificate.build_gate_census(
          device_poly.get(), 1, device_active.get(), 1, 1000,
          &gate_sentinel),
      acc::Status::capacity_exceeded, "hard live-byte cap");
  require(
      std::memcmp(
          &gate_sentinel, &gate_expected, sizeof(gate_sentinel)) == 0,
      "cap rejection changed output");
  acc::GateAnnotationDeviceView after_cap;
  require_status(
      certificate.device_gate_view(&after_cap),
      acc::Status::success, "view after cap rejection");
  require(
      after_cap.epoch == before.epoch &&
          after_cap.gate_present == before.gate_present,
      "cap rejection changed persistent state");

  const std::vector<std::uint32_t> malformed_labels = {1, 1, 2, 3};
  const std::vector<ac::RectI64> metal = {
      rectangle(10, 0, 12, 2, 2)};
  DeviceArray<std::uint32_t> device_labels(malformed_labels);
  DeviceArray<ac::RectI64> device_metal(metal);
  acc::CheckpointCensus checkpoint_sentinel;
  checkpoint_sentinel.labels = 0xa5a5a5a5;
  checkpoint_sentinel.uncertain_roots = 0x5a5a5a5a;
  const acc::CheckpointCensus checkpoint_expected =
      checkpoint_sentinel;
  require_status(
      certificate.evaluate_checkpoint(
          acc::MetalLevel::metal1, device_metal.get(), 1,
          device_labels.get(), device_labels.size(),
          &checkpoint_sentinel),
      acc::Status::malformed_input, "malformed labels");
  require(
      std::memcmp(
          &checkpoint_sentinel, &checkpoint_expected,
          sizeof(checkpoint_sentinel)) == 0,
      "malformed checkpoint changed output");
}

void test_arithmetic_overflow_is_uncertain_failure()
{
  const std::vector<ac::RectI64> poly = {
      rectangle(0, 0, 4, 4, 0)};
  const std::vector<ac::RectI64> active = {
      rectangle(0, 0, 4, 2, 1)};
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);
  acc::Certificate certificate(base_config());
  acc::GateCensus gate_census;
  require_status(
      certificate.build_gate_census(
          device_poly.get(), 1, device_active.get(), 1, 2,
          &gate_census),
      acc::Status::success, "overflow setup");

  const std::vector<ac::RectI64> metal = {
      rectangle(
          INT64_MIN, 0, INT64_MAX, 2, 2)};
  const std::vector<std::uint32_t> labels = {0, 1, 0};
  DeviceArray<ac::RectI64> device_metal(metal);
  DeviceArray<std::uint32_t> device_labels(labels);
  acc::CheckpointCensus sentinel;
  sentinel.labels = 991;
  sentinel.uncertain_roots = 773;
  const acc::CheckpointCensus expected = sentinel;
  require_status(
      certificate.evaluate_checkpoint(
          acc::MetalLevel::metal4, device_metal.get(), 1,
          device_labels.get(), labels.size(), &sentinel),
      acc::Status::arithmetic_overflow,
      "metal multiplication overflow");
  require(
      std::memcmp(&sentinel, &expected, sizeof(sentinel)) == 0,
      "overflow changed checkpoint output");

  constexpr std::int64_t two_to_31 = INT64_C(2147483648);
  constexpr std::int64_t two_to_32 = INT64_C(4294967296);
  const std::vector<ac::RectI64> exact_maximum_sum = {
      // 2^32 * 2^31 = 2^63.
      rectangle(0, 0, two_to_32, two_to_31, 2),
      // 1 * INT64_MAX = 2^63 - 1.
      rectangle(
          two_to_32 + 1, 0, two_to_32 + 2,
          INT64_MAX, 2)};
  DeviceArray<ac::RectI64> device_exact_maximum(
      exact_maximum_sum);
  acc::CheckpointCensus exact_maximum;
  require_status(
      certificate.evaluate_checkpoint(
          acc::MetalLevel::metal4,
          device_exact_maximum.get(),
          device_exact_maximum.size(), device_labels.get(),
          labels.size(), &exact_maximum),
      acc::Status::success,
      "checked atomic exact ULLONG_MAX sum");
  require(
      exact_maximum.roots_with_metal == 1 &&
          exact_maximum.uncertain_roots == 1,
      "exact ULLONG_MAX metal sum changed checkpoint census");

  const std::vector<ac::RectI64> overflowing_sum = {
      rectangle(0, 0, two_to_32, two_to_31, 2),
      rectangle(
          two_to_32 + 1, 0, 2 * two_to_32 + 1,
          two_to_31, 2)};
  DeviceArray<ac::RectI64> device_overflowing_sum(
      overflowing_sum);
  acc::CheckpointCensus addition_sentinel;
  addition_sentinel.labels = 31337;
  addition_sentinel.uncertain_roots = 424242;
  const acc::CheckpointCensus addition_expected =
      addition_sentinel;
  require_status(
      certificate.evaluate_checkpoint(
          acc::MetalLevel::metal4,
          device_overflowing_sum.get(),
          device_overflowing_sum.size(), device_labels.get(),
          labels.size(), &addition_sentinel),
      acc::Status::arithmetic_overflow,
      "checked atomic addition overflow");
  require(
      std::memcmp(
          &addition_sentinel, &addition_expected,
          sizeof(addition_sentinel)) == 0,
      "atomic addition overflow changed checkpoint output");
}

void test_spatial_candidate_scaling()
{
  std::vector<ac::RectI64> poly;
  std::vector<ac::RectI64> active;
  constexpr std::uint32_t count = 200;
  for (std::uint32_t id = 0; id < count; ++id) {
    const std::int64_t x = static_cast<std::int64_t>(id) * 100;
    poly.push_back(rectangle(x, 0, x + 2, 2, id));
    active.push_back(
        rectangle(x + 1, 1, x + 3, 3, count + id));
  }
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);
  acc::Config config = base_config();
  config.limits.max_annotation_owners = 2 * count;
  acc::Certificate certificate(config);
  acc::GateCensus census;
  require_status(
      certificate.build_gate_census(
          device_poly.get(), poly.size(), device_active.get(),
          active.size(), 2 * count, &census),
      acc::Status::success, "spatial scaling build");
  require(
      census.positive_intersections == count,
      "spatial scaling exact overlap count");
  require(
      census.candidate_visits <
          static_cast<std::uint64_t>(count) * count / 10,
      "spatial path regressed toward Cartesian product");
}

void test_query_visit_block_reduction_and_exact_cap()
{
  constexpr std::uint32_t poly_count = 257;
  const std::uint64_t owner_count =
      static_cast<std::uint64_t>(poly_count) + 1;
  std::vector<ac::RectI64> poly;
  poly.reserve(poly_count);
  for (std::uint32_t id = 0; id < poly_count; ++id) {
    poly.push_back(rectangle(0, 0, 10, 10, id));
  }
  const std::vector<ac::RectI64> active = {
      rectangle(0, 0, 10, 10, poly_count)};
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);

  acc::Config exact_config = base_config();
  exact_config.limits.max_query_visits = poly_count;
  acc::Certificate exact_certificate(exact_config);
  acc::GateCensus exact;
  require_status(
      exact_certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(),
          owner_count, &exact),
      acc::Status::success, "block query-visit exact cap");
  require(
      exact.candidate_visits == poly_count,
      "block query-visit reduction changed exact counter");
  require(
      exact.positive_intersections == poly_count &&
          exact.gate_owners == poly_count,
      "block query-visit reduction changed exact predicates");

  acc::Config limited_config = base_config();
  limited_config.limits.max_query_visits = poly_count - 1;
  acc::Certificate limited_certificate(limited_config);
  acc::GateCensus baseline;
  require_status(
      limited_certificate.build_gate_census(
          device_poly.get(), 1, device_active.get(),
          device_active.size(), owner_count, &baseline),
      acc::Status::success, "block query-visit cap baseline");
  acc::GateAnnotationDeviceView before;
  require_status(
      limited_certificate.device_gate_view(&before),
      acc::Status::success, "block query-visit cap baseline view");

  acc::GateCensus sentinel;
  sentinel.annotation_owners = UINT64_C(0x12345678);
  sentinel.candidate_visits = UINT64_C(0x87654321);
  const acc::GateCensus expected = sentinel;
  require_status(
      limited_certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(),
          owner_count, &sentinel),
      acc::Status::capacity_exceeded,
      "block query-visit one-below cap");
  require(
      std::memcmp(&sentinel, &expected, sizeof(sentinel)) == 0,
      "block query-visit cap changed output");
  acc::GateAnnotationDeviceView after;
  require_status(
      limited_certificate.device_gate_view(&after),
      acc::Status::success, "block query-visit post-cap view");
  require(
      after.epoch == before.epoch &&
          after.gate_present == before.gate_present &&
          after.max_single_intersection_area ==
              before.max_single_intersection_area,
      "block query-visit cap changed persistent annotations");
}

void test_active_membership_block_reduction_and_exact_cap()
{
  constexpr std::uint32_t active_count = 257;
  const std::uint64_t owner_count =
      static_cast<std::uint64_t>(active_count) + 1;
  const std::vector<ac::RectI64> poly = {
      rectangle(0, 0, 10, 10, 0)};
  std::vector<ac::RectI64> active;
  active.reserve(active_count);
  for (std::uint32_t id = 0; id < active_count; ++id) {
    active.push_back(rectangle(0, 0, 10, 10, id + 1));
  }
  DeviceArray<ac::RectI64> device_poly(poly);
  DeviceArray<ac::RectI64> device_active(active);

  acc::Config exact_config = base_config();
  exact_config.limits.max_active_memberships = active_count;
  acc::Certificate exact_certificate(exact_config);
  acc::GateCensus exact;
  require_status(
      exact_certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(),
          owner_count, &exact),
      acc::Status::success, "block ACTIVE-membership exact cap");
  require(
      exact.active_memberships == active_count,
      "block ACTIVE-membership reduction changed exact counter");
  require(
      exact.candidate_visits == active_count &&
          exact.positive_intersections == active_count &&
          exact.gate_owners == 1,
      "block ACTIVE-membership reduction changed grid predicates");

  acc::Config limited_config = base_config();
  limited_config.limits.max_active_memberships = active_count - 1;
  acc::Certificate limited_certificate(limited_config);
  acc::GateCensus baseline;
  require_status(
      limited_certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), 1, owner_count, &baseline),
      acc::Status::success, "block ACTIVE-membership cap baseline");
  acc::GateAnnotationDeviceView before;
  require_status(
      limited_certificate.device_gate_view(&before),
      acc::Status::success, "block ACTIVE-membership cap baseline view");

  acc::GateCensus sentinel;
  sentinel.annotation_owners = UINT64_C(0x12345678);
  sentinel.active_memberships = UINT64_C(0x87654321);
  const acc::GateCensus expected = sentinel;
  require_status(
      limited_certificate.build_gate_census(
          device_poly.get(), device_poly.size(),
          device_active.get(), device_active.size(),
          owner_count, &sentinel),
      acc::Status::capacity_exceeded,
      "block ACTIVE-membership one-below cap");
  require(
      std::memcmp(&sentinel, &expected, sizeof(sentinel)) == 0,
      "block ACTIVE-membership cap changed output");
  acc::GateAnnotationDeviceView after;
  require_status(
      limited_certificate.device_gate_view(&after),
      acc::Status::success, "block ACTIVE-membership post-cap view");
  require(
      after.epoch == before.epoch &&
          after.gate_present == before.gate_present &&
          after.max_single_intersection_area ==
              before.max_single_intersection_area,
      "block ACTIVE-membership cap changed persistent annotations");
}

void test_extreme_and_negative_grid_boundaries()
{
  {
    const std::vector<ac::RectI64> poly = {
        rectangle(
            INT64_MAX - 4, -11, INT64_MAX, -1, 0)};
    const std::vector<ac::RectI64> active = {
        rectangle(
            INT64_MAX - 3, -10, INT64_MAX, -1, 1)};
    const GateOracle oracle = gate_oracle(poly, active, 2);
    require(
        oracle.lower[0] == 27,
        "INT64_MAX fixture intersection area");
    DeviceArray<ac::RectI64> device_poly(poly);
    DeviceArray<ac::RectI64> device_active(active);
    acc::Certificate certificate(base_config());
    acc::GateCensus census;
    require_status(
        certificate.build_gate_census(
            device_poly.get(), 1, device_active.get(), 1, 2,
            &census),
        acc::Status::success, "INT64_MAX grid boundary");
    require_gate_view(
        &certificate, oracle, "INT64_MAX grid boundary");
  }

  {
    const std::vector<ac::RectI64> poly = {
        rectangle(-20, -10, -10, 0, 0)};
    const std::vector<ac::RectI64> active = {
        // Exact negative grid boundary touch: no positive area.
        rectangle(-10, -10, 0, 0, 1)};
    const GateOracle oracle = gate_oracle(poly, active, 2);
    DeviceArray<ac::RectI64> device_poly(poly);
    DeviceArray<ac::RectI64> device_active(active);
    acc::Certificate certificate(base_config());
    acc::GateCensus census;
    require_status(
        certificate.build_gate_census(
            device_poly.get(), 1, device_active.get(), 1, 2,
            &census),
        acc::Status::success, "negative exact grid boundary");
    require(
        census.positive_intersections == 0,
        "negative boundary touch became overlap");
    require_gate_view(
        &certificate, oracle, "negative exact grid boundary");
  }
}

void test_seeded_cpu_differential()
{
  std::mt19937_64 random(UINT64_C(0x6b6c61796f7574));
  for (std::uint32_t trial = 0; trial < 80; ++trial) {
    constexpr std::uint32_t poly_owners = 8;
    constexpr std::uint32_t active_owners = 4;
    constexpr std::uint32_t metal_owners = 8;
    constexpr std::uint32_t annotation_owners =
        poly_owners + active_owners;
    constexpr std::uint32_t label_count =
        annotation_owners + metal_owners;
    std::vector<ac::RectI64> poly;
    std::vector<ac::RectI64> active;
    std::vector<ac::RectI64> metal;

    for (std::uint32_t owner = 0; owner < poly_owners; ++owner) {
      const std::int64_t base =
          static_cast<std::int64_t>(owner) * 40 - 100;
      const std::int64_t split =
          2 + static_cast<std::int64_t>(random() % 8);
      poly.push_back(
          rectangle(base, -5, base + split, 5, owner, trial));
      poly.push_back(
          rectangle(base + split, -5, base + 12, 5, owner,
                    trial + 100));
    }
    for (std::uint32_t local = 0; local < active_owners; ++local) {
      const std::int64_t x =
          static_cast<std::int64_t>(random() % 330) - 120;
      const std::int64_t y =
          static_cast<std::int64_t>(random() % 17) - 8;
      const std::int64_t width =
          1 + static_cast<std::int64_t>(random() % 30);
      const std::int64_t height =
          1 + static_cast<std::int64_t>(random() % 12);
      active.push_back(rectangle(
          x, y, x + width, y + height,
          poly_owners + local, trial + 200));
    }
    for (std::uint32_t local = 0; local < metal_owners; ++local) {
      const std::int64_t width =
          1 + static_cast<std::int64_t>(random() % 50);
      const std::int64_t height =
          1 + static_cast<std::int64_t>(random() % 50);
      metal.push_back(rectangle(
          static_cast<std::int64_t>(local) * 60, 50,
          static_cast<std::int64_t>(local) * 60 + width,
          50 + height, annotation_owners + local, trial + 300));
    }

    std::vector<std::uint32_t> labels(label_count);
    for (std::uint32_t owner = 0; owner < label_count; ++owner) {
      labels[owner] = owner;
    }
    // Randomly merge later owners into an earlier canonical root.  Roots are
    // direct and fixed points, matching Connectivity::DeviceLabelView.
    for (std::uint32_t owner = 1; owner < label_count; ++owner) {
      if ((random() & 3) == 0) {
        labels[owner] =
            static_cast<std::uint32_t>(random() % owner);
        labels[owner] = labels[labels[owner]];
      }
    }

    const GateOracle gate =
        gate_oracle(poly, active, annotation_owners);
    DeviceArray<ac::RectI64> device_poly(poly);
    DeviceArray<ac::RectI64> device_active(active);
    acc::Certificate certificate(base_config());
    acc::GateCensus gate_census;
    require_status(
        certificate.build_gate_census(
            device_poly.get(), poly.size(), device_active.get(),
            active.size(), annotation_owners, &gate_census),
        acc::Status::success,
        "seeded gate trial " + std::to_string(trial));
    require(
        gate_census.positive_intersections ==
            gate.positive_intersections,
        "seeded positive census trial " + std::to_string(trial));
    require_gate_view(
        &certificate, gate,
        "seeded gate trial " + std::to_string(trial));

    const acc::MetalLevel level = static_cast<acc::MetalLevel>(
        1 + trial % 4);
    const acc::CheckpointCensus actual = run_checkpoint(
        &certificate, level, metal, labels,
        "seeded checkpoint trial " + std::to_string(trial));
    const acc::CheckpointCensus expected =
        checkpoint_oracle(level, gate, metal, labels);
    compare_checkpoint(
        actual, expected,
        "seeded checkpoint trial " + std::to_string(trial));
    if (expected.uncertain_roots) {
      const acc::CheckpointCensus refined =
          run_root_cell_checkpoint(
              &certificate, level, poly, active, metal, labels,
              "seeded root-cell trial " +
                  std::to_string(trial));
      compare_checkpoint(
          refined,
          root_cell_checkpoint_oracle(
              level, gate, poly, active, metal, labels,
              base_config().grid_cell_size),
          "seeded root-cell trial " + std::to_string(trial));
    }
  }
}

}  // namespace

int main()
{
  try {
    require_cuda(cudaSetDevice(0), "select CUDA device");
    test_external_live_update_is_transactional();
    test_cross_context_tile_splits_and_touches();
    test_one_dbu_boundary_and_duplicate_metal_upper();
    test_merged_roots_use_max_gate_lower();
    test_root_cell_refinement_is_disjoint_and_fail_closed();
    test_factor_zero_diode_exemption_is_annotation_only();
    test_failures_are_transactional_and_budgeted();
    test_arithmetic_overflow_is_uncertain_failure();
    test_spatial_candidate_scaling();
    test_query_visit_block_reduction_and_exact_cap();
    test_active_membership_block_reduction_and_exact_cap();
    test_extreme_and_negative_grid_boundaries();
    test_seeded_cpu_differential();
    std::cout << "antenna_clean_certificate_gpu_test: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr
        << "antenna_clean_certificate_gpu_test: FAIL: "
        << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
