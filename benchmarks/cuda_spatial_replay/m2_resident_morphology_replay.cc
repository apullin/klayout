/*
 * Qualification driver for m2_resident_morphology_gpu.
 *
 * The executable is intentionally separate from both CUDA libraries.  It
 * supplies independent raster fixtures, production hierarchy expansion and
 * a checked stock F90 boundary artifact.
 */

#include "m2_resident_morphology_gpu.cuh"

#include "m1_width_space_host_scene_format.h"
#include "m2_manhattan_production_loader.h"
#include "m2_merged_boundary_oracle.h"

#include <cuda_runtime_api.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace format = klayout_m1ws_scene;
namespace morph = klayout_cuda::m2_resident_morphology;
namespace mu = klayout_cuda::manhattan_union;
namespace m2oracle = klayout_cuda::m2_boundary_oracle;
namespace m2prod = klayout_cuda::m2_production;

using Clock = std::chrono::steady_clock;
using mu::DirectedSegmentI64;
using mu::GpuUnionLimits;
using mu::GpuUnionOutput;
using mu::RectI64;
using mu::ResidentStripHook;
using mu::SegmentAxis;
using mu::StripInterval;

constexpr char kProductionM2SceneSha256[] =
    "dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2";
constexpr char kProductionM2OracleSceneSha256[] =
    "441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480";
constexpr std::uint64_t kProductionM2FlatPolygons = UINT64_C(22945976);
constexpr std::uint64_t kProductionM2FlatRectangles = UINT64_C(22946444);
constexpr char kGt90GoldenFileSha256[] =
    "e7149202ef0ace74618a01f56135ea1cde2b4a0bdb00102f9a5f4a072af2ea49";
constexpr char kGt90GoldenPayloadSha256[] =
    "233a611bc306126b0292763954aab2d984508f2466a56ced2d1cf08af6c526ff";
constexpr std::uint32_t kGt90GoldenHeaderBytes = 128;

double elapsed_ms(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::uint32_t parse_u32(const char *value, const char *option)
{
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end || parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string("invalid ") + option);
  }
  return static_cast<std::uint32_t>(parsed);
}

double median(std::vector<double> values)
{
  if (values.empty()) {
    throw std::runtime_error("median of empty sample");
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2
             ? values[middle]
             : (values[middle - 1] + values[middle]) / 2.0;
}

std::string hex_digest(const std::uint8_t *bytes, std::size_t count)
{
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < count; ++index) {
    stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return stream.str();
}

std::string segment_string(const DirectedSegmentI64 &segment)
{
  std::ostringstream stream;
  stream << static_cast<std::uint32_t>(segment.axis) << ":"
         << segment.side << ":" << segment.fixed << ":"
         << segment.lo << ":" << segment.hi;
  return stream.str();
}

void require_boundary_equal(
    const std::string &name,
    const std::vector<DirectedSegmentI64> &expected,
    const std::vector<DirectedSegmentI64> &actual)
{
  if (expected.size() != actual.size()) {
    std::ostringstream message;
    message << name << ": boundary size expected=" << expected.size()
            << " actual=" << actual.size();
    throw std::runtime_error(message.str());
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const DirectedSegmentI64 &left = expected[index];
    const DirectedSegmentI64 &right = actual[index];
    if (left.fixed != right.fixed || left.lo != right.lo ||
        left.hi != right.hi || left.side != right.side ||
        left.axis != right.axis) {
      throw std::runtime_error(
          name + ": boundary mismatch index=" + std::to_string(index) +
          " expected='" + segment_string(left) + "' actual='" +
          segment_string(right) + "'");
    }
  }
}

using Cell = std::pair<int, int>;
using Cells = std::set<Cell>;

Cells rasterize(const std::vector<RectI64> &rectangles)
{
  Cells cells;
  for (const RectI64 &rectangle : rectangles) {
    for (std::int64_t x = rectangle.left; x < rectangle.right; ++x) {
      for (std::int64_t y = rectangle.bottom; y < rectangle.top; ++y) {
        cells.emplace(static_cast<int>(x), static_cast<int>(y));
      }
    }
  }
  return cells;
}

Cells raster_dilate(const Cells &input, int radius)
{
  Cells result;
  for (const Cell &cell : input) {
    for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        result.emplace(cell.first + dx, cell.second + dy);
      }
    }
  }
  return result;
}

Cells raster_erode(const Cells &input, int radius)
{
  Cells result;
  for (const Cell &cell : input) {
    bool keep = true;
    for (int dx = -radius; dx <= radius && keep; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        if (!input.count({cell.first + dx, cell.second + dy})) {
          keep = false;
          break;
        }
      }
    }
    if (keep) result.insert(cell);
  }
  return result;
}

std::vector<DirectedSegmentI64> raster_boundary(const Cells &cells)
{
  std::vector<DirectedSegmentI64> result;
  result.reserve(cells.size() * 2);
  for (const Cell &cell : cells) {
    const std::int64_t x = cell.first;
    const std::int64_t y = cell.second;
    if (!cells.count({cell.first, cell.second - 1})) {
      result.push_back(
          {y, x, x + 1, -1, SegmentAxis::horizontal});
    }
    if (!cells.count({cell.first, cell.second + 1})) {
      result.push_back(
          {y + 1, x, x + 1, 1, SegmentAxis::horizontal});
    }
    if (!cells.count({cell.first - 1, cell.second})) {
      result.push_back(
          {x, y, y + 1, -1, SegmentAxis::vertical});
    }
    if (!cells.count({cell.first + 1, cell.second})) {
      result.push_back(
          {x + 1, y, y + 1, 1, SegmentAxis::vertical});
    }
  }
  mu::canonicalize_segments_reference_for_test(&result);
  return result;
}

struct QualificationContext
{
  morph::QualificationOperation operation =
      morph::QualificationOperation::erode;
  std::int64_t first_radius = 1;
  std::int64_t second_radius = 0;
  bool invoked = false;
  std::vector<DirectedSegmentI64> boundary;
};

void qualification_consume(
    cudaStream_t stream, const std::int64_t *xs,
    std::uint32_t x_slabs, const StripInterval *intervals,
    std::uint64_t interval_count, const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque)
{
  auto *context = static_cast<QualificationContext *>(opaque);
  if (!context || context->invoked) {
    throw std::runtime_error("qualification callback state");
  }
  context->invoked = true;
  context->boundary = morph::qualification_boundary(
      stream,
      {xs, x_slabs, intervals, interval_count, slab_offsets,
       slab_counts},
      context->operation, context->first_radius,
      context->second_radius);
}

std::string operation_name(morph::QualificationOperation operation)
{
  if (operation == morph::QualificationOperation::erode) return "erode";
  if (operation == morph::QualificationOperation::dilate) return "dilate";
  return "erode-dilate";
}

void run_gate_case(
    const std::string &name, const std::vector<RectI64> &rectangles,
    morph::QualificationOperation operation, int first_radius,
    int second_radius, int device)
{
  Cells expected_cells = rasterize(rectangles);
  if (operation == morph::QualificationOperation::erode) {
    expected_cells = raster_erode(expected_cells, first_radius);
  } else if (operation == morph::QualificationOperation::dilate) {
    expected_cells = raster_dilate(expected_cells, first_radius);
  } else {
    expected_cells = raster_erode(expected_cells, first_radius);
    expected_cells = raster_dilate(expected_cells, second_radius);
  }
  const std::vector<DirectedSegmentI64> expected =
      raster_boundary(expected_cells);

  QualificationContext context;
  context.operation = operation;
  context.first_radius = first_radius;
  context.second_radius = second_radius;
  ResidentStripHook hook;
  hook.consume = qualification_consume;
  hook.context = &context;
  hook.stop_before_boundary = true;
  GpuUnionLimits limits;
  limits.max_slabs_per_rectangle = 4096;
  const GpuUnionOutput output =
      mu::gpu_union_host(rectangles, limits, device, &hook);
  if (output.fallback) {
    throw std::runtime_error(
        name + ": union/callback fallback: " + output.message);
  }
  if (!context.invoked || !output.resident_consumer_completed) {
    throw std::runtime_error(
        name + ": resident callback not completed");
  }
  require_boundary_equal(
      name + "/" + operation_name(operation), expected,
      context.boundary);
}

void run_morph_x_guard_case(
    const std::string &name, std::int64_t low, std::int64_t split,
    std::int64_t high, bool expect_fallback, int device)
{
  constexpr std::int64_t radius = 10;
  constexpr std::int64_t bottom = 0;
  constexpr std::int64_t top = 4;
  if (!(low < split && split < high)) {
    throw std::runtime_error(name + ": malformed guard fixture");
  }
  const std::vector<RectI64> rectangles = {
      {low, bottom, split, top, 0, 0},
      {split, bottom, high, top, 0, 0}};
  QualificationContext context;
  context.operation = morph::QualificationOperation::dilate;
  context.first_radius = radius;
  ResidentStripHook hook;
  hook.consume = qualification_consume;
  hook.context = &context;
  hook.stop_before_boundary = true;
  GpuUnionLimits limits;
  limits.max_slabs_per_rectangle = 4096;
  const GpuUnionOutput output =
      mu::gpu_union_host(rectangles, limits, device, &hook);
  if (expect_fallback) {
    if (!output.fallback || !context.invoked ||
        !output.segments.empty() ||
        output.message.find("x coordinate overflow") ==
            std::string::npos) {
      throw std::runtime_error(
          name + ": unsafe x arithmetic did not fail closed: " +
          output.message);
    }
    return;
  }
  if (output.fallback || !context.invoked ||
      !output.resident_consumer_completed) {
    throw std::runtime_error(
        name + ": just-inside x arithmetic was rejected: " +
        output.message);
  }
  const std::vector<DirectedSegmentI64> expected = {
      {bottom - radius, low - radius, high + radius, -1,
       SegmentAxis::horizontal},
      {top + radius, low - radius, high + radius, 1,
       SegmentAxis::horizontal},
      {low - radius, bottom - radius, top + radius, -1,
       SegmentAxis::vertical},
      {high + radius, bottom - radius, top + radius, 1,
       SegmentAxis::vertical}};
  require_boundary_equal(name, expected, context.boundary);
}

void run_morph_x_guard_gate(int device)
{
  constexpr std::int64_t radius = 10;
  constexpr std::int64_t low_limit =
      std::numeric_limits<std::int64_t>::min() / 2;
  constexpr std::int64_t high_limit =
      std::numeric_limits<std::int64_t>::max() / 2;
  const std::int64_t rejected_low = low_limit + radius;
  run_morph_x_guard_case(
      "x-guard-low-reject", rejected_low, rejected_low + 1,
      rejected_low + 100, true, device);
  const std::int64_t accepted_low = low_limit + 2 * radius;
  run_morph_x_guard_case(
      "x-guard-low-accept", accepted_low, accepted_low + 1,
      accepted_low + 100, false, device);
  const std::int64_t rejected_high = high_limit - radius;
  run_morph_x_guard_case(
      "x-guard-high-reject", rejected_high - 100,
      rejected_high - 1, rejected_high, true, device);
  const std::int64_t accepted_high = high_limit - 2 * radius;
  run_morph_x_guard_case(
      "x-guard-high-accept", accepted_high - 100,
      accepted_high - 1, accepted_high, false, device);
  std::cout << "M2_RESIDENT_MORPH_X_GUARD PASS checks=4"
            << " rejected=2 accepted_exact=2\n";
}

std::vector<std::pair<std::string, std::vector<RectI64>>>
morphology_fixtures()
{
  const auto rectangle = [](
      std::int64_t left, std::int64_t bottom,
      std::int64_t right, std::int64_t top) {
    return RectI64{left, bottom, right, top, 0, 0};
  };
  return {
      {"single", {rectangle(0, 0, 7, 6)}},
      {"threshold-width-1", {rectangle(0, 0, 1, 7)}},
      {"threshold-width-2", {rectangle(0, 0, 2, 7)}},
      {"threshold-width-3", {rectangle(0, 0, 3, 7)}},
      {"threshold-width-4", {rectangle(0, 0, 4, 7)}},
      {"threshold-width-5", {rectangle(0, 0, 5, 7)}},
      {"gap-2",
       {rectangle(0, 0, 3, 4), rectangle(5, 0, 8, 4)}},
      {"gap-3",
       {rectangle(0, 0, 3, 4), rectangle(6, 0, 9, 4)}},
      {"corner-touch",
       {rectangle(0, 0, 3, 3), rectangle(3, 3, 6, 6)}},
      {"edge-touch",
       {rectangle(0, 0, 3, 4), rectangle(3, 1, 7, 5)}},
      {"notch",
       {rectangle(0, 0, 9, 3), rectangle(0, 3, 3, 9),
        rectangle(6, 3, 9, 9)}},
      {"hole",
       {rectangle(0, 0, 9, 2), rectangle(0, 7, 9, 9),
        rectangle(0, 2, 2, 7), rectangle(7, 2, 9, 7)}},
      {"thin-bridge",
       {rectangle(0, 0, 4, 6), rectangle(8, 0, 12, 6),
        rectangle(4, 2, 8, 3)}},
      {"overlap",
       {rectangle(-4, -2, 5, 3), rectangle(-1, -5, 3, 7),
        rectangle(2, 1, 8, 5)}},
  };
}

void run_differential_gate(int device)
{
  std::uint64_t checks = 0;
  for (const auto &fixture : morphology_fixtures()) {
    run_gate_case(
        fixture.first + "-e1", fixture.second,
        morph::QualificationOperation::erode, 1, 0, device);
    ++checks;
    run_gate_case(
        fixture.first + "-d1", fixture.second,
        morph::QualificationOperation::dilate, 1, 0, device);
    ++checks;
    run_gate_case(
        fixture.first + "-e1d2", fixture.second,
        morph::QualificationOperation::erode_then_dilate, 1, 2,
        device);
    ++checks;
    run_gate_case(
        fixture.first + "-e2", fixture.second,
        morph::QualificationOperation::erode, 2, 0, device);
    ++checks;
  }

  std::mt19937 generator(0x4d325f39);
  std::uniform_int_distribution<int> coordinate(-8, 7);
  std::uniform_int_distribution<int> extent(1, 7);
  std::uniform_int_distribution<int> rectangle_count(1, 9);
  for (int trial = 0; trial < 64; ++trial) {
    std::vector<RectI64> rectangles;
    const int count = rectangle_count(generator);
    for (int index = 0; index < count; ++index) {
      const int left = coordinate(generator);
      const int bottom = coordinate(generator);
      rectangles.push_back(
          {left, bottom, left + extent(generator),
           bottom + extent(generator), 0, 0});
    }
    const std::string prefix = "random-" + std::to_string(trial);
    run_gate_case(
        prefix + "-e1", rectangles,
        morph::QualificationOperation::erode, 1, 0, device);
    ++checks;
    run_gate_case(
        prefix + "-d2", rectangles,
        morph::QualificationOperation::dilate, 2, 0, device);
    ++checks;
    run_gate_case(
        prefix + "-e1d2", rectangles,
        morph::QualificationOperation::erode_then_dilate, 1, 2,
        device);
    ++checks;
  }
  if (checks != 248) {
    throw std::runtime_error("differential check census mismatch");
  }
  std::cout << "M2_RESIDENT_MORPH_DIFFERENTIAL PASS checks=" << checks
            << " directed=" << morphology_fixtures().size()
            << " random=64\n";
}

void require_long_space_certificate(
    const std::string &name,
    const std::vector<DirectedSegmentI64> &segments,
    std::uint64_t expected_violations,
    std::uint64_t expected_uncertain)
{
  const morph::LongSpaceCertificate result =
      morph::certify_f90_long_edge_space(segments);
  const std::uint64_t count = segments.size();
  const std::uint64_t expected_pairs =
      count > 1 ? count * (count - 1) / 2 : 0;
  if (result.pairs_checked != expected_pairs ||
      result.violations != expected_violations ||
      result.uncertain != expected_uncertain) {
    std::ostringstream message;
    message << name << ": pairs=" << result.pairs_checked
            << " violations=" << result.violations
            << " uncertain=" << result.uncertain;
    throw std::runtime_error(message.str());
  }
}

void run_f90_long_space_certificate_gate()
{
  const DirectedSegmentI64 east = {
      0, 0, 200, 1, SegmentAxis::horizontal};
  require_long_space_certificate(
      "horizontal-179",
      {east, {179, 0, 200, -1, SegmentAxis::horizontal}}, 1, 0);
  require_long_space_certificate(
      "horizontal-180",
      {east, {180, 0, 200, -1, SegmentAxis::horizontal}}, 0, 0);
  require_long_space_certificate(
      "horizontal-181",
      {east, {181, 0, 200, -1, SegmentAxis::horizontal}}, 0, 0);
  require_long_space_certificate(
      "wrong-exterior-side",
      {east, {-179, 0, 200, -1, SegmentAxis::horizontal}}, 0, 0);
  require_long_space_certificate(
      "vertical-179",
      {{0, 0, 200, 1, SegmentAxis::vertical},
       {179, 0, 200, -1, SegmentAxis::vertical}},
      1, 0);
  require_long_space_certificate(
      "corner-exact-108-144-180",
      {{0, 0, 100, 1, SegmentAxis::horizontal},
       {144, 208, 300, -1, SegmentAxis::horizontal}},
      0, 0);
  require_long_space_certificate(
      "corner-inside-107-144",
      {{0, 0, 100, 1, SegmentAxis::horizontal},
       {144, 207, 300, -1, SegmentAxis::horizontal}},
      1, 0);
  require_long_space_certificate(
      "unsafe-signed-span",
      {{0, std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max(), 1,
        SegmentAxis::horizontal},
       {179, std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max(), -1,
        SegmentAxis::horizontal}},
      0, 1);
  std::cout << "M2_RESIDENT_F90_LONG_SPACE_GATE PASS checks=8"
            << " threshold_dbu=180\n";
}

template <class Function>
void require_throw(const std::string &name, const std::string &needle,
                   Function function)
{
  try {
    function();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(needle) == std::string::npos) {
      throw std::runtime_error(
          name + ": wrong rejection: " + error.what());
    }
    return;
  }
  throw std::runtime_error(name + ": did not reject");
}

void run_limit_contract_gate()
{
  morph::Request request;
  request.limits.max_active_slabs = 129;
  require_throw(
      "active-cap", "invalid resident morphology limits",
      [&]() {
        (void)morph::consume_f90_f270(nullptr, {}, request);
      });

  request = {};
  request.limits.max_total_source_visits =
      morph::kUniversalSourceVisitCap + 1;
  require_throw(
      "unqualified-work-cap", "exceeds 2B",
      [&]() {
        (void)morph::consume_f90_f270(nullptr, {}, request);
      });

  request = {};
  request.allow_qualified_production_work_cap = true;
  require_throw(
      "production-work-cap", "must be exactly 8B",
      [&]() {
        (void)morph::consume_f90_f270(nullptr, {}, request);
      });

  request = {};
  request.limits.max_raw_boundary_segments = 10;
  request.limits.max_boundary_segments = 11;
  require_throw(
      "boundary-cap-order", "invalid resident morphology limits",
      [&]() {
        (void)morph::consume_f90_f270(nullptr, {}, request);
      });

  require_throw(
      "null-hook", "invalid resident morphology hook context",
      [&]() {
        (void)morph::make_resident_hook(nullptr);
      });

  std::cout << "M2_RESIDENT_MORPH_LIMIT_GATE PASS checks=5"
            << " universal_work_cap="
            << morph::kUniversalSourceVisitCap
            << " production_work_cap="
            << morph::kQualifiedProductionSourceVisitCap << "\n";
}

void run_device_source_invariant_gate()
{
  thrust::device_vector<std::int64_t> xs = {0, 1, 2};
  thrust::device_vector<StripInterval> intervals = {
      {0, 1, 0, 0}};
  thrust::device_vector<std::uint64_t> offsets = {0, 1};
  thrust::device_vector<std::uint32_t> counts = {1, 1};
  const auto view = [&]() {
    return morph::DeviceStripView{
        thrust::raw_pointer_cast(xs.data()), 2,
        thrust::raw_pointer_cast(intervals.data()),
        intervals.size(),
        thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(counts.data())};
  };

  require_throw(
      "source-range", "resident source slab/x invariant",
      [&]() {
        (void)morph::qualification_boundary(
            nullptr, view(),
            morph::QualificationOperation::dilate, 1, 0);
      });

  const std::uint32_t valid_counts[2] = {1, 0};
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(counts.data()), valid_counts,
          sizeof(valid_counts), cudaMemcpyHostToDevice),
      "source invariant counts H2D");
  const StripInterval bad_reserved = {0, 1, 0, 1};
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(intervals.data()), &bad_reserved,
          sizeof(bad_reserved), cudaMemcpyHostToDevice),
      "source invariant bad interval H2D");
  require_throw(
      "source-interval", "resident source interval invariant",
      [&]() {
        (void)morph::qualification_boundary(
            nullptr, view(),
            morph::QualificationOperation::dilate, 1, 0);
      });

  const StripInterval valid_interval = {0, 1, 0, 0};
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(intervals.data()), &valid_interval,
          sizeof(valid_interval), cudaMemcpyHostToDevice),
      "source invariant valid interval H2D");
  const std::int64_t duplicate_x = 0;
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(xs.data()) + 1, &duplicate_x,
          sizeof(duplicate_x), cudaMemcpyHostToDevice),
      "source invariant duplicate x H2D");
  require_throw(
      "source-x-order", "resident source slab/x invariant",
      [&]() {
        (void)morph::qualification_boundary(
            nullptr, view(),
            morph::QualificationOperation::dilate, 1, 0);
      });

  std::cout
      << "M2_RESIDENT_MORPH_SOURCE_INVARIANT_GATE PASS checks=3\n";
}

class StreamGuard
{
public:
  StreamGuard()
  {
    cuda_require(
        cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking),
        "qualification stream create");
  }

  ~StreamGuard()
  {
    if (m_stream) cudaStreamDestroy(m_stream);
  }

  StreamGuard(const StreamGuard &) = delete;
  StreamGuard &operator=(const StreamGuard &) = delete;

  operator cudaStream_t() const { return m_stream; }

private:
  cudaStream_t m_stream = nullptr;
};

void run_explicit_stream_gate()
{
  thrust::device_vector<std::int64_t> xs = {0, 4};
  thrust::device_vector<StripInterval> intervals = {
      {0, 4, 0, 0}};
  thrust::device_vector<std::uint64_t> offsets = {0};
  thrust::device_vector<std::uint32_t> counts = {1};
  StreamGuard stream;
  const std::vector<DirectedSegmentI64> actual =
      morph::qualification_boundary(
          stream,
          {thrust::raw_pointer_cast(xs.data()), 1,
           thrust::raw_pointer_cast(intervals.data()),
           intervals.size(),
           thrust::raw_pointer_cast(offsets.data()),
           thrust::raw_pointer_cast(counts.data())},
          morph::QualificationOperation::dilate, 1, 0);
  const std::vector<DirectedSegmentI64> expected = {
      {-1, -1, 5, -1, SegmentAxis::horizontal},
      {5, -1, 5, 1, SegmentAxis::horizontal},
      {-1, -1, 5, -1, SegmentAxis::vertical},
      {5, -1, 5, 1, SegmentAxis::vertical}};
  require_boundary_equal("explicit-stream", expected, actual);
  std::cout << "M2_RESIDENT_MORPH_STREAM_GATE PASS checks=1"
            << " nonblocking=1\n";
}

bool valid_rectangle(const RectI64 &rectangle)
{
  return rectangle.left < rectangle.right &&
         rectangle.bottom < rectangle.top;
}

bool transform_production_rectangle(
    const m2prod::ResolvedContextI64 &context,
    const m2prod::RectTemplateI64 &source,
    std::uint64_t context_token, RectI64 *destination)
{
  if (context.transform >= 8 || !destination) return false;
  static constexpr int matrix[8][4] = {
      {1, 0, 0, 1}, {0, -1, 1, 0}, {-1, 0, 0, -1},
      {0, 1, -1, 0}, {1, 0, 0, -1}, {0, 1, 1, 0},
      {-1, 0, 0, 1}, {0, -1, -1, 0}};
  const int *transform = matrix[context.transform];
  const std::int64_t xs[4] = {
      source.left, source.left, source.right, source.right};
  const std::int64_t ys[4] = {
      source.bottom, source.top, source.bottom, source.top};
  RectI64 result = {
      INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN,
      source.source_token, context_token};
  for (unsigned int corner = 0; corner < 4; ++corner) {
    const __int128 x =
        static_cast<__int128>(transform[0]) * xs[corner] +
        static_cast<__int128>(transform[1]) * ys[corner] + context.tx;
    const __int128 y =
        static_cast<__int128>(transform[2]) * xs[corner] +
        static_cast<__int128>(transform[3]) * ys[corner] + context.ty;
    if (x < std::numeric_limits<std::int64_t>::min() ||
        x > std::numeric_limits<std::int64_t>::max() ||
        y < std::numeric_limits<std::int64_t>::min() ||
        y > std::numeric_limits<std::int64_t>::max()) {
      return false;
    }
    const std::int64_t xx = static_cast<std::int64_t>(x);
    const std::int64_t yy = static_cast<std::int64_t>(y);
    result.left = std::min(result.left, xx);
    result.bottom = std::min(result.bottom, yy);
    result.right = std::max(result.right, xx);
    result.top = std::max(result.top, yy);
  }
  if (!valid_rectangle(result)) return false;
  *destination = result;
  return true;
}

std::vector<RectI64> expand_production_rectangles(
    const m2prod::CompactScene &scene, double *elapsed)
{
  const auto begin = Clock::now();
  if (!elapsed ||
      scene.flat_rectangles != kProductionM2FlatRectangles ||
      scene.rectangle_offsets.size() != scene.m2_contexts.size()) {
    throw std::runtime_error(
        "production compact-scene rectangle census is not qualified");
  }
  std::vector<RectI64> rectangles(scene.flat_rectangles);
  std::atomic<std::uint64_t> next{0};
  std::atomic<std::uint32_t> failed{0};
  const unsigned int detected =
      std::max(1u, std::thread::hardware_concurrency());
  const unsigned int worker_count = std::min(32u, detected);
  constexpr std::uint64_t chunk = 64;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (unsigned int worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&]() {
      while (!failed.load(std::memory_order_relaxed)) {
        const std::uint64_t first =
            next.fetch_add(chunk, std::memory_order_relaxed);
        if (first >= scene.m2_contexts.size()) return;
        const std::uint64_t last = std::min<std::uint64_t>(
            first + chunk, scene.m2_contexts.size());
        for (std::uint64_t list_id = first; list_id < last; ++list_id) {
          const std::uint32_t context_id = scene.m2_contexts[list_id];
          if (context_id >= scene.contexts.size()) {
            failed.store(1, std::memory_order_relaxed);
            return;
          }
          const m2prod::ResolvedContextI64 &context =
              scene.contexts[context_id];
          if (context.cell >= scene.cells.size()) {
            failed.store(1, std::memory_order_relaxed);
            return;
          }
          const m2prod::CellTemplate &cell =
              scene.cells[context.cell];
          const std::uint64_t output_begin =
              scene.rectangle_offsets[list_id];
          if (cell.rectangle_begin >
                  scene.rectangles.size() ||
              cell.rectangle_count >
                  scene.rectangles.size() - cell.rectangle_begin ||
              output_begin > rectangles.size() ||
              cell.rectangle_count >
                  rectangles.size() - output_begin) {
            failed.store(1, std::memory_order_relaxed);
            return;
          }
          for (std::uint32_t local = 0;
               local < cell.rectangle_count; ++local) {
            if (!transform_production_rectangle(
                    context,
                    scene.rectangles[cell.rectangle_begin + local],
                    context_id, &rectangles[output_begin + local])) {
              failed.store(1, std::memory_order_relaxed);
              return;
            }
          }
        }
      }
    });
  }
  for (std::thread &worker : workers) worker.join();
  if (failed.load(std::memory_order_relaxed)) {
    throw std::runtime_error(
        "production host hierarchy expansion failed");
  }
  *elapsed = elapsed_ms(begin, Clock::now());
  return rectangles;
}

std::vector<m2oracle::DirectedSegmentI64> read_gt90_golden_v1(
    const std::string &path)
{
  if (m2oracle::candidate_stream_file_sha256(path) !=
      kGt90GoldenFileSha256) {
    throw std::runtime_error("gt90 golden file SHA-256 mismatch");
  }
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open gt90 golden");
  }
  const std::streamoff end = input.tellg();
  if (end < kGt90GoldenHeaderBytes ||
      static_cast<std::uint64_t>(end) >
          std::numeric_limits<std::size_t>::max() ||
      end > std::numeric_limits<std::streamsize>::max()) {
    throw std::runtime_error("gt90 golden size is invalid");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(bytes.data()), end)) {
    throw std::runtime_error("short read of gt90 golden");
  }
  constexpr char magic[8] =
      {'K', 'M', '2', 'B', 'N', 'D', '0', '1'};
  if (std::memcmp(bytes.data(), magic, sizeof(magic)) != 0 ||
      format::load_u32_le(bytes.data() + 8) != 1 ||
      format::load_u32_le(bytes.data() + 12) !=
          kGt90GoldenHeaderBytes ||
      format::load_u32_le(bytes.data() + 16) != UINT32_C(0x01020304) ||
      format::load_u32_le(bytes.data() + 20) != 32 ||
      format::load_u64_le(bytes.data() + 24) != bytes.size() ||
      hex_digest(bytes.data() + 40, 32) !=
          kGt90GoldenPayloadSha256 ||
      hex_digest(bytes.data() + 72, 32) !=
          kProductionM2OracleSceneSha256 ||
      std::any_of(
          bytes.begin() + 104, bytes.begin() + 128,
          [](std::uint8_t byte) { return byte != 0; })) {
    throw std::runtime_error("gt90 golden header is invalid");
  }
  const std::uint64_t count = format::load_u64_le(bytes.data() + 32);
  if (count != morph::kQualifiedF90BoundarySegments ||
      count >
          (std::numeric_limits<std::uint64_t>::max() -
           kGt90GoldenHeaderBytes) /
              32 ||
      kGt90GoldenHeaderBytes + count * 32 != bytes.size()) {
    throw std::runtime_error("gt90 golden count/length is invalid");
  }
  std::vector<m2oracle::DirectedSegmentI64> result;
  result.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    const std::uint8_t *record =
        bytes.data() + kGt90GoldenHeaderBytes + index * 32;
    const std::uint32_t side_bits = format::load_u32_le(record + 24);
    std::int32_t side = 0;
    std::memcpy(&side, &side_bits, sizeof(side));
    result.push_back(
        {format::load_i64_le(record),
         format::load_i64_le(record + 8),
         format::load_i64_le(record + 16), side,
         static_cast<m2oracle::SegmentAxis>(
             format::load_u32_le(record + 28))});
  }
  if (m2oracle::canonical_boundary_sha256(result) !=
          kGt90GoldenPayloadSha256 ||
      m2oracle::canonical_boundary_fnv64(result) !=
          morph::kQualifiedF90BoundaryFnv64) {
    throw std::runtime_error("gt90 golden payload identity mismatch");
  }
  return result;
}

void require_production_boundary_equal(
    const std::vector<m2oracle::DirectedSegmentI64> &expected,
    const std::vector<DirectedSegmentI64> &actual)
{
  if (expected.size() != morph::kQualifiedF90BoundarySegments ||
      actual.size() != expected.size()) {
    throw std::runtime_error(
        "gt90 boundary census mismatch");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const DirectedSegmentI64 &left = actual[index];
    const m2oracle::DirectedSegmentI64 &right = expected[index];
    if (left.fixed != right.fixed || left.lo != right.lo ||
        left.hi != right.hi || left.side != right.side ||
        static_cast<std::uint32_t>(left.axis) !=
            static_cast<std::uint32_t>(right.axis)) {
      throw std::runtime_error(
          "gt90 exact boundary first differs at segment " +
          std::to_string(index) + " actual='" +
          segment_string(left) + "'");
    }
  }
}

void require_production_census(const morph::ResidentContext &context)
{
  const morph::Result &result = context.result;
  if (!context.invoked ||
      result.f90_boundary_segments !=
          morph::kQualifiedF90BoundarySegments ||
      result.f90_long_segments !=
          morph::kQualifiedF90LongSegments ||
      result.f90_space_pairs_checked !=
          morph::kQualifiedF90LongPairs ||
      result.f90_space_violations != 0 ||
      result.f90_space_uncertain != 0 ||
      result.f270_eroded_intervals != 0) {
    throw std::runtime_error(
        "production resident morphology census mismatch");
  }
}

void print_production_timing(
    std::uint32_t run, const GpuUnionOutput &output,
    const morph::Result &result)
{
  const double peak_delta_mib =
      (result.device_free_begin_bytes -
       result.device_free_low_bytes) /
      (1024.0 * 1024.0);
  std::cout
      << "M2_RESIDENT_F90_GPU"
      << " run=" << run
      << " qualification=" << !result.f90_boundary.empty()
      << " union_resident_total_ms=" << std::fixed
      << std::setprecision(3) << output.total_ms
      << " erode89_ms=" << result.erode89.elapsed_ms
      << " dilate90_ms=" << result.dilate90.elapsed_ms
      << " boundary_and_long_space_ms="
      << result.boundary_and_long_space_ms
      << " erode269_count_ms="
      << result.erode269_count.elapsed_ms
      << " resident_callback_ms=" << result.total_ms
      << " gt90_intervals=" << result.dilate90.output_intervals
      << " gt90_segments=" << result.f90_boundary_segments
      << " gt90_long_segments=" << result.f90_long_segments
      << " gt90_space_pairs_checked="
      << result.f90_space_pairs_checked
      << " gt270_eroded_intervals="
      << result.f270_eroded_intervals
      << " erode89_source_visits="
      << result.erode89.source_visits
      << " dilate90_source_visits="
      << result.dilate90.source_visits
      << " erode269_source_visits="
      << result.erode269_count.source_visits
      << " resident_peak_delta_mib=" << std::setprecision(1)
      << peak_delta_mib << std::setprecision(3) << "\n";
}

void run_production_morphology(
    const std::string &kact_path, const std::string &gt90_path,
    std::uint32_t repeat, int device)
{
  if (repeat < 2) {
    throw std::runtime_error(
        "production morphology requires qualification plus a warm run");
  }
  const auto all_begin = Clock::now();
  const auto load_begin = Clock::now();
  m2prod::LoadOptions load_options;
  load_options.expected_scene_sha256 = kProductionM2SceneSha256;
  load_options.expected_flat_polygons = kProductionM2FlatPolygons;
  load_options.expected_flat_rectangles = kProductionM2FlatRectangles;
  const m2prod::CompactScene scene =
      m2prod::load_kact_templates(kact_path, load_options);
  const double load_ms = elapsed_ms(load_begin, Clock::now());
  double host_expand_ms = 0.0;
  const std::vector<RectI64> rectangles =
      expand_production_rectangles(scene, &host_expand_ms);

  const auto golden_begin = Clock::now();
  const std::vector<m2oracle::DirectedSegmentI64> golden =
      read_gt90_golden_v1(gt90_path);
  const double golden_ms = elapsed_ms(golden_begin, Clock::now());

  GpuUnionLimits union_limits;
  union_limits.max_memberships = UINT64_C(100000000);
  union_limits.max_events = UINT64_C(200000000);
  union_limits.max_raw_segments = UINT64_C(12000000);
  union_limits.max_segments = UINT64_C(8000000);
  union_limits.max_slabs_per_rectangle = 64;

  std::vector<double> warm_union_resident_ms;
  std::vector<double> warm_callback_ms;
  double qualification_ms = 0.0;
  for (std::uint32_t run = 0; run < repeat; ++run) {
    morph::ResidentContext context;
    context.request.copy_f90_boundary_to_host = run == 0;
    context.request.allow_qualified_production_work_cap = true;
    context.request.limits.max_total_source_visits =
        morph::kQualifiedProductionSourceVisitCap;
    ResidentStripHook hook = morph::make_resident_hook(&context);
    const GpuUnionOutput output =
        mu::gpu_union_host(rectangles, union_limits, device, &hook);
    if (output.fallback) {
      throw std::runtime_error(
          "production union/resident morphology fallback: " +
          output.message);
    }
    if (!output.resident_consumer_completed) {
      throw std::runtime_error(
          "production resident consumer did not complete");
    }
    require_production_census(context);
    if (run == 0) {
      require_production_boundary_equal(
          golden, context.result.f90_boundary);
      if (context.result.f90_boundary_fnv64 !=
          morph::kQualifiedF90BoundaryFnv64) {
        throw std::runtime_error(
            "qualified F90 boundary digest mismatch");
      }
      qualification_ms = output.total_ms;
      std::cout
          << "M2_RESIDENT_F90_QUALIFICATION PASS"
          << " compared_edges="
          << context.result.f90_boundary.size()
          << " boundary_fnv64="
          << context.result.f90_boundary_fnv64
          << " golden_file_sha256="
          << kGt90GoldenFileSha256 << "\n";
    } else {
      warm_union_resident_ms.push_back(output.total_ms);
      warm_callback_ms.push_back(context.result.total_ms);
    }
    print_production_timing(run, output, context.result);
  }

  const double warm_union_resident =
      median(warm_union_resident_ms);
  const double warm_callback = median(warm_callback_ms);
  constexpr double stock_f90_f270_ms = 14691.0;
  constexpr double stock_union_stitch_f90_f270_ms =
      1707.380 + 2194.0 + stock_f90_f270_ms;
  const double charged_resident_pipeline_ms =
      load_ms + host_expand_ms + warm_union_resident;
  const double callback_reduction =
      100.0 * (stock_f90_f270_ms - warm_callback) /
      stock_f90_f270_ms;
  const double charged_pipeline_reduction =
      100.0 *
      (stock_union_stitch_f90_f270_ms -
       charged_resident_pipeline_ms) /
      stock_union_stitch_f90_f270_ms;
  std::cout
      << "M2_RESIDENT_F90_PRODUCTION PASS"
      << " rectangles=" << rectangles.size()
      << " qualification_union_resident_ms=" << std::fixed
      << std::setprecision(3) << qualification_ms
      << " warm_union_resident_median_ms=" << warm_union_resident
      << " warm_callback_median_ms=" << warm_callback
      << " compact_load_ms=" << load_ms
      << " host_expand_ms=" << host_expand_ms
      << " golden_load_and_hash_ms=" << golden_ms
      << " stock_f90_f270_ms=" << stock_f90_f270_ms
      << " resident_suffix_less_time_pct=" << callback_reduction
      << " stock_union_stitch_f90_f270_ms="
      << stock_union_stitch_f90_f270_ms
      << " charged_resident_pipeline_ms="
      << charged_resident_pipeline_ms
      << " charged_pipeline_less_time_pct="
      << charged_pipeline_reduction
      << " verification_total_ms="
      << elapsed_ms(all_begin, Clock::now()) << "\n";
}

void print_help(const char *program)
{
  std::cout
      << "Usage: " << program
      << " --self-test [--device N]\n"
      << "       " << program
      << " --production --kact FILE --gt90-golden FILE"
         " [--repeat N] [--device N]\n";
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    int device = 0;
    bool self_test = false;
    bool production = false;
    std::uint32_t repeat = 4;
    std::string kact_path;
    std::string gt90_path;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--self-test") {
        self_test = true;
      } else if (argument == "--production") {
        production = true;
      } else if (argument == "--kact" && index + 1 < argc) {
        kact_path = argv[++index];
      } else if (argument == "--gt90-golden" &&
                 index + 1 < argc) {
        gt90_path = argv[++index];
      } else if (argument == "--repeat" && index + 1 < argc) {
        repeat = parse_u32(argv[++index], "--repeat");
      } else if (argument == "--device" && index + 1 < argc) {
        device = static_cast<int>(
            parse_u32(argv[++index], "--device"));
      } else if (argument == "--help" || argument == "-h") {
        print_help(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + argument);
      }
    }
    if (self_test == production) {
      print_help(argv[0]);
      return 2;
    }
    cuda_require(cudaSetDevice(device), "morph cudaSetDevice");
    if (self_test) {
      run_limit_contract_gate();
      run_device_source_invariant_gate();
      run_explicit_stream_gate();
      run_morph_x_guard_gate(device);
      run_f90_long_space_certificate_gate();
      run_differential_gate(device);
    } else {
      if (kact_path.empty() || gt90_path.empty()) {
        print_help(argv[0]);
        return 2;
      }
      run_production_morphology(
          kact_path, gt90_path, repeat, device);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M2_RESIDENT_MORPH_ERROR " << error.what() << "\n";
    return 1;
  }
}
