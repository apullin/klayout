/*
 * Standalone replay and production driver for the shared exact CUDA
 * Manhattan-union core.
 */

#include "manhattan_union_gpu.cuh"
#include "m2_manhattan_production_loader.h"
#include "m2_merged_boundary_oracle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace mu = klayout_cuda::manhattan_union;
namespace m2prod = klayout_cuda::m2_production;
namespace m2oracle = klayout_cuda::m2_boundary_oracle;

using Clock = std::chrono::steady_clock;
using mu::DirectedSegmentI64;
using mu::GpuUnionLimits;
using mu::GpuUnionOutput;
using mu::RectI64;
using mu::SegmentAxis;
using Limits = GpuUnionLimits;
using UnionOutput = GpuUnionOutput;

constexpr char kProductionM2SceneSha256[] =
    "dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2";
constexpr char kProductionM2OracleFileSha256[] =
    "980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f";
constexpr char kProductionM2OracleSceneSha256[] =
    "441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480";
constexpr char kProductionM2BoundarySha256[] =
    "94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d";
constexpr std::uint64_t kProductionM2FlatPolygons = UINT64_C(22945976);
constexpr std::uint64_t kProductionM2FlatRectangles = UINT64_C(22946444);
constexpr std::uint64_t kProductionM2BoundarySegments = UINT64_C(4385384);
constexpr std::uint64_t kProductionM2BoundaryFnv64 =
    UINT64_C(7541395996791771514);

double elapsed_ms(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool valid_rectangle(const RectI64 &rectangle)
{
  return rectangle.left < rectangle.right &&
         rectangle.bottom < rectangle.top;
}

UnionOutput gpu_union(
    const std::vector<RectI64> &rectangles, const Limits &limits,
    int device)
{
  return mu::gpu_union_host(rectangles, limits, device);
}

UnionOutput cpu_union(
    const std::vector<RectI64> &rectangles, const Limits &limits)
{
  return mu::cpu_union_reference_for_test(rectangles, limits);
}

void canonicalize_segments(std::vector<DirectedSegmentI64> *segments)
{
  mu::canonicalize_segments_reference_for_test(segments);
}

std::vector<DirectedSegmentI64> gpu_canonicalize_for_test(
    const std::vector<DirectedSegmentI64> &raw, int device)
{
  return mu::gpu_canonicalize_segments_for_test(raw, device);
}

std::string segment_string(const DirectedSegmentI64 &segment)
{
  std::ostringstream stream;
  stream << (segment.axis == SegmentAxis::horizontal ? "H" : "V")
         << (segment.side < 0 ? "-" : "+") << " fixed=" << segment.fixed
         << " [" << segment.lo << "," << segment.hi << ")";
  return stream.str();
}

void require_equal(const std::string &name, const UnionOutput &cpu,
                   const UnionOutput &gpu)
{
  if (cpu.fallback || gpu.fallback) {
    throw std::runtime_error(
        name + ": unexpected fallback cpu='" + cpu.message + "' gpu='" +
        gpu.message + "'");
  }
  if (cpu.segments.size() != gpu.segments.size()) {
    std::ostringstream stream;
    stream << name << ": segment-count mismatch CPU="
           << cpu.segments.size() << " GPU=" << gpu.segments.size();
    throw std::runtime_error(stream.str());
  }
  for (std::size_t index = 0; index < cpu.segments.size(); ++index) {
    const DirectedSegmentI64 &first = cpu.segments[index];
    const DirectedSegmentI64 &second = gpu.segments[index];
    if (first.axis != second.axis || first.side != second.side ||
        first.fixed != second.fixed || first.lo != second.lo ||
        first.hi != second.hi) {
      throw std::runtime_error(
          name + ": segment mismatch at " + std::to_string(index) +
          " CPU=" + segment_string(first) +
          " GPU=" + segment_string(second));
    }
  }
  if (cpu.digest != gpu.digest) {
    throw std::runtime_error(name + ": digest mismatch");
  }
}

RectI64 rectangle(std::int64_t left, std::int64_t bottom,
                  std::int64_t right, std::int64_t top,
                  std::uint64_t token = 0)
{
  return {left, bottom, right, top, token, 0};
}

struct Fixture
{
  std::string name;
  std::vector<RectI64> rectangles;
};

std::vector<Fixture> directed_fixtures()
{
  return {
      {"empty", {}},
      {"single", {rectangle(0, 0, 10, 20)}},
      {"duplicate",
       {rectangle(0, 0, 10, 10), rectangle(0, 0, 10, 10)}},
      {"nested",
       {rectangle(-20, -20, 20, 20), rectangle(-5, -5, 5, 5)}},
      {"partial-overlap",
       {rectangle(0, 0, 10, 10), rectangle(5, 3, 15, 14)}},
      {"edge-touch-x",
       {rectangle(0, 0, 10, 10), rectangle(10, 0, 20, 10)}},
      {"edge-touch-y",
       {rectangle(0, 0, 10, 10), rectangle(0, 10, 10, 20)}},
      {"corner-touch-opposite-sides",
       {rectangle(0, 0, 10, 10), rectangle(10, 10, 20, 20)}},
      {"t-junction",
       {rectangle(0, 0, 30, 10), rectangle(10, 10, 20, 30)}},
      {"plus",
       {rectangle(-5, -20, 5, 20), rectangle(-20, -5, 20, 5)}},
      {"ring-with-hole",
       {rectangle(0, 0, 30, 5), rectangle(0, 25, 30, 30),
        rectangle(0, 5, 5, 25), rectangle(25, 5, 30, 25)}},
      {"covered-seam",
       {rectangle(0, 0, 10, 30), rectangle(10, 0, 20, 10),
        rectangle(10, 20, 20, 30), rectangle(5, 5, 15, 25)}},
      {"containment-bridge",
       {rectangle(0, 0, 10, 100), rectangle(10, 10, 20, 20),
        rectangle(10, 30, 20, 40), rectangle(20, 0, 30, 100)}},
      {"negative",
       {rectangle(-100, -90, -10, -20),
        rectangle(-60, -110, 20, -50)}},
      {"large-int64",
       {rectangle(INT64_C(-4000000000000000000),
                  INT64_C(-3000000000000000000),
                  INT64_C(-3999999999999999900),
                  INT64_C(-2999999999999999800)),
        rectangle(INT64_C(-3999999999999999950),
                  INT64_C(-2999999999999999900),
                  INT64_C(-3999999999999999800),
                  INT64_C(-2999999999999999700))}},
  };
}

std::vector<RectI64> random_rectangles(
    std::uint64_t seed, std::size_t count)
{
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<std::int64_t> coordinate(-80, 80);
  std::uniform_int_distribution<std::int64_t> extent(1, 30);
  std::vector<RectI64> rectangles;
  rectangles.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::int64_t left = coordinate(generator);
    const std::int64_t bottom = coordinate(generator);
    rectangles.push_back(
        rectangle(left, bottom, left + extent(generator),
                  bottom + extent(generator), index + 1));
  }
  return rectangles;
}

void run_self_test(int device)
{
  const Limits limits;
  std::size_t checks = 0;
  for (const Fixture &fixture : directed_fixtures()) {
    const UnionOutput cpu = cpu_union(fixture.rectangles, limits);
    const UnionOutput gpu = gpu_union(fixture.rectangles, limits, device);
    require_equal(fixture.name, cpu, gpu);
    std::cout << "MANHATTAN_UNION_FIXTURE ok name=" << fixture.name
              << " rectangles=" << fixture.rectangles.size()
              << " segments=" << gpu.segments.size()
              << " digest=0x" << std::hex << gpu.digest << std::dec
              << " gpu_ms=" << std::fixed << std::setprecision(3)
              << gpu.total_ms << "\n";
    ++checks;
  }

  for (std::uint64_t seed = 1; seed <= 64; ++seed) {
    const std::vector<RectI64> rectangles =
        random_rectangles(seed, 1 + seed % 47);
    const UnionOutput cpu = cpu_union(rectangles, limits);
    const UnionOutput gpu = gpu_union(rectangles, limits, device);
    require_equal("random-" + std::to_string(seed), cpu, gpu);
    ++checks;
  }
  std::cout << "MANHATTAN_UNION_RANDOM ok cases=64\n";

  std::vector<DirectedSegmentI64> canonical_bridge = {
      {7, 0, 100, -1, SegmentAxis::vertical},
      {7, 10, 20, -1, SegmentAxis::vertical},
      {7, 30, 40, -1, SegmentAxis::vertical},
      {7, 100, 120, -1, SegmentAxis::vertical},
      {7, 10, 20, 1, SegmentAxis::vertical}};
  std::vector<DirectedSegmentI64> cpu_canonical_bridge =
      canonical_bridge;
  canonicalize_segments(&cpu_canonical_bridge);
  const std::vector<DirectedSegmentI64> gpu_canonical_bridge =
      gpu_canonicalize_for_test(canonical_bridge, device);
  const bool canonical_equal =
      cpu_canonical_bridge.size() == gpu_canonical_bridge.size() &&
      std::equal(
          cpu_canonical_bridge.begin(), cpu_canonical_bridge.end(),
          gpu_canonical_bridge.begin(),
          [](const DirectedSegmentI64 &first,
             const DirectedSegmentI64 &second) {
            return first.axis == second.axis &&
                   first.side == second.side &&
                   first.fixed == second.fixed &&
                   first.lo == second.lo && first.hi == second.hi;
          });
  if (cpu_canonical_bridge.size() != 2 || !canonical_equal ||
      gpu_canonical_bridge[0].lo != 0 ||
      gpu_canonical_bridge[0].hi != 120) {
    throw std::runtime_error("prefix-max canonical bridge mismatch");
  }
  ++checks;
  std::cout
      << "MANHATTAN_UNION_CANONICAL ok containment_bridge=1 "
         "prefix_max=1\n";

  const std::array<RectI64, 4> invalid = {
      rectangle(0, 0, 0, 1), rectangle(2, 3, 1, 4),
      rectangle(0, 5, 1, 5), rectangle(0, 9, 1, 8)};
  for (std::size_t index = 0; index < invalid.size(); ++index) {
    const UnionOutput cpu = cpu_union({invalid[index]}, limits);
    const UnionOutput gpu = gpu_union({invalid[index]}, limits, device);
    if (!cpu.fallback || !gpu.fallback || !gpu.segments.empty()) {
      throw std::runtime_error(
          "degenerate rectangle did not fail closed");
    }
    ++checks;
  }
  std::cout << "MANHATTAN_UNION_DEGENERATE ok cases=4 fallback=1\n";

  Limits bounded = limits;
  bounded.max_slabs_per_rectangle = 2;
  const std::vector<RectI64> capacity = {
      rectangle(0, 0, 100, 10), rectangle(10, 20, 20, 30),
      rectangle(30, 20, 40, 30), rectangle(50, 20, 60, 30)};
  const UnionOutput gpu_capacity =
      gpu_union(capacity, bounded, device);
  if (!gpu_capacity.fallback || !gpu_capacity.segments.empty() ||
      gpu_capacity.message.find("slab capacity") == std::string::npos) {
    throw std::runtime_error(
        "per-rectangle capacity did not fail closed");
  }
  ++checks;
  std::cout << "MANHATTAN_UNION_CAPACITY ok fallback=1 message='"
            << gpu_capacity.message << "'\n";

  const std::vector<RectI64> wide_y = {
      rectangle(0, INT64_MIN + 10, 10, INT64_MIN + 20),
      rectangle(20, INT64_MAX - 20, 30, INT64_MAX - 10)};
  const UnionOutput gpu_wide_y = gpu_union(wide_y, limits, device);
  if (!gpu_wide_y.fallback || !gpu_wide_y.segments.empty() ||
      gpu_wide_y.message != "packed y-coordinate range") {
    throw std::runtime_error(
        "packed y-range capacity did not fail closed");
  }
  ++checks;
  std::cout
      << "MANHATTAN_UNION_Y_RANGE ok fallback=1 packed_u32=1\n";

  std::cout << "MANHATTAN_UNION_SELF_TEST PASS checks=" << checks
            << " directed=" << directed_fixtures().size()
            << " random=64 canonical=1 degeneracy=4 capacity=2\n";
}

std::vector<RectI64> touching_grid(std::uint32_t dimension)
{
  const std::uint64_t count =
      static_cast<std::uint64_t>(dimension) * dimension;
  if (count > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("grid is too large for host");
  }
  std::vector<RectI64> rectangles;
  rectangles.reserve(static_cast<std::size_t>(count));
  constexpr std::int64_t pitch = 16;
  for (std::uint32_t y = 0; y < dimension; ++y) {
    for (std::uint32_t x = 0; x < dimension; ++x) {
      rectangles.push_back(
          rectangle(
              static_cast<std::int64_t>(x) * pitch,
              static_cast<std::int64_t>(y) * pitch,
              static_cast<std::int64_t>(x + 1) * pitch,
              static_cast<std::int64_t>(y + 1) * pitch,
              static_cast<std::uint64_t>(y) * dimension + x + 1));
    }
  }
  return rectangles;
}

double median(std::vector<double> values)
{
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2) return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

void print_gpu_timing(std::uint32_t run, const UnionOutput &output)
{
  std::cout << "MANHATTAN_UNION_GPU run=" << run
            << " total_ms=" << std::fixed << std::setprecision(3)
            << output.total_ms << " h2d_ms=" << output.h2d_ms
            << " x_membership_ms=" << output.x_membership_ms
            << " strip_scan_ms=" << output.strip_scan_ms
            << " boundary_ms=" << output.boundary_ms
            << " d2h_ms=" << output.d2h_ms
            << " memberships=" << output.memberships
            << " strips=" << output.strip_intervals
            << " raw_segments=" << output.raw_segments
            << " segments=" << output.segments.size()
            << " sampled_live_allocation_delta_mib="
            << std::setprecision(1)
            << (output.device_free_begin_bytes -
                output.device_free_low_bytes) /
                   (1024.0 * 1024.0)
            << std::setprecision(3) << "\n";
}

void run_benchmark(std::uint32_t dimension, std::uint32_t repeat,
                   int device)
{
  if (!dimension || repeat < 2) {
    throw std::runtime_error(
        "benchmark requires a nonzero grid and at least two runs");
  }
  const std::vector<RectI64> rectangles = touching_grid(dimension);
  Limits limits;
  limits.max_slabs_per_rectangle = std::max<std::uint32_t>(
      limits.max_slabs_per_rectangle, dimension + 1);
  const UnionOutput cpu = cpu_union(rectangles, limits);
  if (cpu.fallback) {
    throw std::runtime_error(
        "CPU benchmark fallback: " + cpu.message);
  }
  std::cout << "MANHATTAN_UNION_CPU rectangles=" << rectangles.size()
            << " total_ms=" << std::fixed << std::setprecision(3)
            << cpu.total_ms << " memberships=" << cpu.memberships
            << " strips=" << cpu.strip_intervals
            << " raw_segments=" << cpu.raw_segments
            << " segments=" << cpu.segments.size()
            << " digest=0x" << std::hex << cpu.digest << std::dec
            << "\n";

  std::vector<double> warm_times;
  UnionOutput last;
  double cold_time = 0.0;
  for (std::uint32_t run = 0; run < repeat; ++run) {
    last = gpu_union(rectangles, limits, device);
    require_equal("benchmark-" + std::to_string(run), cpu, last);
    print_gpu_timing(run, last);
    if (run) {
      warm_times.push_back(last.total_ms);
    } else {
      cold_time = last.total_ms;
    }
  }
  const double warm_median = median(warm_times);
  const double time_reduction =
      100.0 * (cpu.total_ms - warm_median) / cpu.total_ms;
  const double throughput_gain =
      100.0 * (cpu.total_ms / warm_median - 1.0);
  std::cout << "MANHATTAN_UNION_BENCH PASS rectangles="
            << rectangles.size() << " grid=" << dimension << "x"
            << dimension << " cold_gpu_ms=" << std::fixed
            << std::setprecision(3) << cold_time
            << " warm_gpu_median_ms=" << warm_median
            << " cpu_ms=" << cpu.total_ms
            << " time_reduction_pct=" << time_reduction
            << " throughput_gain_pct=" << throughput_gain
            << " digest=0x" << std::hex << cpu.digest << std::dec
            << "\n";
}

bool transform_production_rectangle(
    const m2prod::ResolvedContextI64 &context,
    const m2prod::RectTemplateI64 &source,
    std::uint64_t context_token, RectI64 *destination)
{
  if (context.transform >= 8) return false;
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
  if (scene.flat_rectangles != kProductionM2FlatRectangles ||
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
          if (cell.rectangle_begin + cell.rectangle_count >
                  scene.rectangles.size() ||
              output_begin + cell.rectangle_count > rectangles.size()) {
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
        "production host hierarchy expansion failed exact qualification");
  }
  *elapsed = elapsed_ms(begin, Clock::now());
  return rectangles;
}

void compare_production_boundary(
    const m2oracle::BoundaryOracle &oracle,
    const UnionOutput &candidate)
{
  if (candidate.fallback) {
    throw std::runtime_error(
        "production CUDA union fell back: " + candidate.message);
  }
  if (candidate.segments.size() != kProductionM2BoundarySegments ||
      oracle.segments.size() != kProductionM2BoundarySegments ||
      candidate.digest != kProductionM2BoundaryFnv64 ||
      oracle.boundary_fnv64 != kProductionM2BoundaryFnv64) {
    std::ostringstream stream;
    stream << "production boundary census/digest mismatch candidate_count="
           << candidate.segments.size() << " oracle_count="
           << oracle.segments.size() << " candidate_fnv="
           << candidate.digest << " oracle_fnv="
           << oracle.boundary_fnv64;
    throw std::runtime_error(stream.str());
  }
  for (std::size_t index = 0;
       index < candidate.segments.size(); ++index) {
    const DirectedSegmentI64 &first = candidate.segments[index];
    const m2oracle::DirectedSegmentI64 &second =
        oracle.segments[index];
    if (first.fixed != second.fixed || first.lo != second.lo ||
        first.hi != second.hi || first.side != second.side ||
        static_cast<std::uint32_t>(first.axis) !=
            static_cast<std::uint32_t>(second.axis)) {
      throw std::runtime_error(
          "production boundary first differs at segment " +
          std::to_string(index));
    }
  }
}

void validate_production_boundary_identity(
    const UnionOutput &candidate)
{
  if (candidate.fallback) {
    throw std::runtime_error(
        "production CUDA union fell back: " + candidate.message);
  }
  if (candidate.rectangle_count != kProductionM2FlatRectangles ||
      candidate.memberships != UINT64_C(92386704) ||
      candidate.event_count != UINT64_C(184773408) ||
      candidate.x_slabs != UINT64_C(46383) ||
      candidate.raw_segments != UINT64_C(9575624) ||
      candidate.segments.size() != kProductionM2BoundarySegments ||
      candidate.digest != kProductionM2BoundaryFnv64) {
    std::ostringstream stream;
    stream
        << "production boundary census/digest mismatch rectangles="
        << candidate.rectangle_count
        << " memberships=" << candidate.memberships
        << " events=" << candidate.event_count
        << " x_slabs=" << candidate.x_slabs
        << " raw_segments=" << candidate.raw_segments
        << " candidate_count=" << candidate.segments.size()
        << " candidate_fnv=" << candidate.digest;
    throw std::runtime_error(stream.str());
  }
}

std::vector<m2oracle::DirectedSegmentI64> portable_candidate(
    const UnionOutput &candidate)
{
  validate_production_boundary_identity(candidate);
  std::vector<m2oracle::DirectedSegmentI64> result;
  result.reserve(candidate.segments.size());
  for (const DirectedSegmentI64 &segment : candidate.segments) {
    result.push_back(
        {segment.fixed, segment.lo, segment.hi, segment.side,
         static_cast<m2oracle::SegmentAxis>(
             static_cast<std::uint32_t>(segment.axis))});
  }
  return result;
}

void run_production_m2(
    const std::string &kact_path, const std::string &oracle_path,
    const std::string &candidate_output_path, std::uint32_t repeat,
    int device)
{
  if (!repeat) {
    throw std::runtime_error(
        "production repeat count must be nonzero");
  }
  const auto all_begin = Clock::now();
  const auto load_begin = Clock::now();
  m2prod::LoadOptions load_options;
  load_options.expected_scene_sha256 = kProductionM2SceneSha256;
  load_options.expected_flat_polygons = kProductionM2FlatPolygons;
  load_options.expected_flat_rectangles =
      kProductionM2FlatRectangles;
  const m2prod::CompactScene scene =
      m2prod::load_kact_templates(kact_path, load_options);
  const double load_ms = elapsed_ms(load_begin, Clock::now());

  double host_expand_ms = 0.0;
  const std::vector<RectI64> rectangles =
      expand_production_rectangles(scene, &host_expand_ms);

  m2oracle::BoundaryOracle oracle;
  double oracle_ms = 0.0;
  if (!oracle_path.empty()) {
    const auto oracle_begin = Clock::now();
    m2oracle::LoadOptions oracle_options;
    oracle_options.expected_file_sha256 =
        kProductionM2OracleFileSha256;
    oracle_options.expected_scene_sha256 =
        kProductionM2OracleSceneSha256;
    oracle_options.expected_boundary_sha256 =
        kProductionM2BoundarySha256;
    oracle =
        m2oracle::load_cpu_merged_boundary(oracle_path, oracle_options);
    oracle_ms = elapsed_ms(oracle_begin, Clock::now());
  }

  Limits limits;
  limits.max_memberships = UINT64_C(100000000);
  limits.max_events = UINT64_C(200000000);
  limits.max_raw_segments = UINT64_C(12000000);
  limits.max_segments = UINT64_C(8000000);
  limits.max_slabs_per_rectangle = 64;

  std::vector<double> warm_times;
  double cold_time = 0.0;
  UnionOutput candidate;
  for (std::uint32_t run = 0; run < repeat; ++run) {
    candidate = gpu_union(rectangles, limits, device);
    validate_production_boundary_identity(candidate);
    if (!oracle_path.empty()) {
      compare_production_boundary(oracle, candidate);
    }
    print_gpu_timing(run, candidate);
    if (run) {
      warm_times.push_back(candidate.total_ms);
    } else {
      cold_time = candidate.total_ms;
    }
  }
  double candidate_write_ms = 0.0;
  if (!candidate_output_path.empty()) {
    const auto write_begin = Clock::now();
    const std::vector<m2oracle::DirectedSegmentI64> segments =
        portable_candidate(candidate);
    m2oracle::CandidateStreamIdentity identity;
    identity.producer_scene_sha256 = kProductionM2SceneSha256;
    identity.qualification_scene_sha256 =
        kProductionM2OracleSceneSha256;
    identity.boundary_sha256 = kProductionM2BoundarySha256;
    identity.segment_count = kProductionM2BoundarySegments;
    identity.boundary_fnv64 = kProductionM2BoundaryFnv64;
    m2oracle::write_candidate_stream(
        candidate_output_path, segments, identity);
    candidate_write_ms = elapsed_ms(write_begin, Clock::now());
  }
  const double warm_median =
      warm_times.empty() ? cold_time : median(warm_times);
  const double pipeline_ms = load_ms + host_expand_ms + warm_median;
  const double published_pipeline_ms =
      load_ms + host_expand_ms + candidate.total_ms +
      candidate_write_ms;
  std::cout
      << "M2_MANHATTAN_PRODUCTION_UNION PASS"
      << " rectangles=" << candidate.rectangle_count
      << " memberships=" << candidate.memberships
      << " events=" << candidate.event_count
      << " x_slabs=" << candidate.x_slabs
      << " strips=" << candidate.strip_intervals
      << " raw_segments=" << candidate.raw_segments
      << " segments=" << candidate.segments.size()
      << " boundary_fnv64=" << candidate.digest
      << " load_ms=" << std::fixed << std::setprecision(3) << load_ms
      << " host_expand_ms=" << host_expand_ms
      << " oracle_qualification=" << (oracle_path.empty() ? 0 : 1)
      << " oracle_ms=" << oracle_ms
      << " cold_gpu_ms=" << cold_time
      << " warm_gpu_median_ms=" << warm_median
      << " charged_host_roundtrip_pipeline_ms=" << pipeline_ms
      << " candidate_output="
      << (candidate_output_path.empty() ? "none"
                                        : candidate_output_path)
      << " candidate_gpu_ms=" << candidate.total_ms
      << " candidate_write_ms=" << candidate_write_ms
      << " published_candidate_pipeline_ms=" << published_pipeline_ms
      << " sampled_live_allocation_delta_mib="
      << std::setprecision(1)
      << (candidate.device_free_begin_bytes -
          candidate.device_free_low_bytes) /
             (1024.0 * 1024.0)
      << " verification_total_ms=" << std::setprecision(3)
      << elapsed_ms(all_begin, Clock::now()) << "\n";
}

std::uint32_t parse_u32(const char *value, const char *option)
{
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string("invalid ") + option);
  }
  return static_cast<std::uint32_t>(parsed);
}

void print_help(const char *program)
{
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --self-test              run exact directed/random/fallback gates\n"
      << "  --benchmark-grid N       union an N-by-N touching rectangle grid\n"
      << "  --production-m2-kact P   qualified raw hierarchical M2 capture\n"
      << "  --production-m2-oracle P qualified CPU-merged KM1WS oracle\n"
      << "  --production-m2-candidate-out P write exact GPU KM2BND02 result\n"
      << "  --repeat N               benchmark process-local runs (default 5)\n"
      << "  --device N               CUDA device (default 0)\n"
      << "  --help                   show this text\n";
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    bool self_test = argc == 1;
    std::uint32_t benchmark_grid = 0;
    std::uint32_t repeat = 5;
    int device = 0;
    std::string production_kact;
    std::string production_oracle;
    std::string production_candidate_output;
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--self-test") {
        self_test = true;
      } else if (option == "--benchmark-grid" &&
                 index + 1 < argc) {
        benchmark_grid =
            parse_u32(argv[++index], "--benchmark-grid");
      } else if (option == "--repeat" && index + 1 < argc) {
        repeat = parse_u32(argv[++index], "--repeat");
      } else if (option == "--production-m2-kact" &&
                 index + 1 < argc) {
        production_kact = argv[++index];
      } else if (option == "--production-m2-oracle" &&
                 index + 1 < argc) {
        production_oracle = argv[++index];
      } else if (option == "--production-m2-candidate-out" &&
                 index + 1 < argc) {
        production_candidate_output = argv[++index];
      } else if (option == "--device" && index + 1 < argc) {
        device =
            static_cast<int>(parse_u32(argv[++index], "--device"));
      } else if (option == "--help") {
        print_help(argv[0]);
        return 0;
      } else {
        throw std::runtime_error(
            "unknown or incomplete option: " + option);
      }
    }
    if (self_test) run_self_test(device);
    if (benchmark_grid) {
      run_benchmark(benchmark_grid, repeat, device);
    }
    if (production_kact.empty() &&
        (!production_oracle.empty() ||
         !production_candidate_output.empty())) {
      throw std::runtime_error(
          "production M2 oracle/output requires a KACT path");
    }
    if (!production_kact.empty() &&
        production_oracle.empty() &&
        production_candidate_output.empty()) {
      throw std::runtime_error(
          "production M2 requires an oracle or candidate output");
    }
    if (!production_kact.empty()) {
      run_production_m2(
          production_kact, production_oracle,
          production_candidate_output, repeat, device);
    }
    if (!self_test && !benchmark_grid && production_kact.empty()) {
      throw std::runtime_error("no action requested");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MANHATTAN_UNION_FAIL " << error.what() << "\n";
    return 1;
  }
}
