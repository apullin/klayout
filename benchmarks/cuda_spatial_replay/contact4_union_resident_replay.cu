/*
 * Differential qualification for the resident ACTIVE-union/CONTACT.4 seam.
 *
 * The empty-only callback is a proof certificate: a nonempty or uncertain
 * answer deliberately throws so the union pipeline falls back.  The replay
 * retains the populated Result and checks it against an independent serial
 * union/predicate/grid oracle on both success and fail-closed paths.
 */

#include "active3_exact_predicate.cuh"
#include "contact4_union_resident.cuh"
#include "manhattan_union_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace a3 = klayout_cuda::active3;
namespace c4 = klayout_cuda::contact4_union_resident;
namespace mu = klayout_cuda::manhattan_union;

struct Box
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
};

struct CellSpan
{
  std::int64_t x0;
  std::int64_t y0;
  std::int64_t x1;
  std::int64_t y1;
};

struct HostGrid
{
  std::int64_t base_x;
  std::int64_t base_y;
  std::int64_t maximum_x;
  std::int64_t maximum_y;
  std::int64_t cell_size;
};

struct Oracle
{
  std::uint64_t candidates = 0;
  std::uint64_t hits = 0;
  std::uint64_t uncertain = 0;
};

struct Aggregate
{
  std::uint64_t cases = 0;
  std::uint64_t oracle_pairs = 0;
  std::uint64_t oracle_candidates = 0;
  std::uint64_t gpu_candidates = 0;
  std::uint64_t callbacks = 0;
  std::uint64_t expected_fallbacks = 0;
  std::uint64_t maximum_contact_edges = 0;
  std::uint64_t minimum_grid_cells = UINT64_MAX;
  std::uint64_t union_peak_bytes = 0;
  std::uint64_t callback_incremental_peak_bytes = 0;
  std::uint64_t full_process_callback_high_water_bytes = 0;
  std::uint64_t device_view_cases = 0;
  std::uint64_t device_view_expected_fallbacks = 0;
};

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::int64_t floor_div(
    std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

std::int64_t checked_add(
    std::int64_t value, std::int64_t delta)
{
  const __int128 sum =
      static_cast<__int128>(value) + delta;
  require(
      sum >= INT64_MIN && sum <= INT64_MAX,
      "replay coordinate expansion overflow");
  return static_cast<std::int64_t>(sum);
}

std::vector<a3::DirectedEdge> box_edges(const Box &box)
{
  require(
      box.left < box.right && box.bottom < box.top,
      "invalid replay box");
  // Clockwise in KLayout's y-up coordinates: material is on the right.
  return {
      {box.left, box.bottom, box.left, box.top},
      {box.left, box.top, box.right, box.top},
      {box.right, box.top, box.right, box.bottom},
      {box.right, box.bottom, box.left, box.bottom}};
}

void append_box_edges(
    const Box &box, std::vector<a3::DirectedEdge> *edges)
{
  const std::vector<a3::DirectedEdge> produced = box_edges(box);
  edges->insert(edges->end(), produced.begin(), produced.end());
}

std::vector<mu::RectI64> rectangles(
    const std::vector<Box> &boxes)
{
  std::vector<mu::RectI64> result;
  result.reserve(boxes.size());
  for (std::size_t id = 0; id < boxes.size(); ++id) {
    const Box &box = boxes[id];
    result.push_back(
        {box.left, box.bottom, box.right, box.top,
         static_cast<std::uint64_t>(id + 1), 0});
  }
  return result;
}

a3::DirectedEdge directed_edge(
    const mu::DirectedSegmentI64 &segment)
{
  require(
      segment.lo < segment.hi &&
          (segment.side == -1 || segment.side == 1),
      "invalid oracle union segment");
  if (segment.axis == mu::SegmentAxis::horizontal) {
    return segment.side < 0
        ? a3::DirectedEdge{
              segment.hi, segment.fixed,
              segment.lo, segment.fixed}
        : a3::DirectedEdge{
              segment.lo, segment.fixed,
              segment.hi, segment.fixed};
  }
  require(
      segment.axis == mu::SegmentAxis::vertical,
      "invalid oracle union segment axis");
  return segment.side < 0
      ? a3::DirectedEdge{
            segment.fixed, segment.lo,
            segment.fixed, segment.hi}
      : a3::DirectedEdge{
            segment.fixed, segment.hi,
            segment.fixed, segment.lo};
}

HostGrid contact_grid(
    const std::vector<a3::DirectedEdge> &contacts,
    std::int64_t cell_size)
{
  require(!contacts.empty() && cell_size > 0, "invalid oracle grid");
  std::int64_t left = std::min(contacts[0].x1, contacts[0].x2);
  std::int64_t right = std::max(contacts[0].x1, contacts[0].x2);
  std::int64_t bottom = std::min(contacts[0].y1, contacts[0].y2);
  std::int64_t top = std::max(contacts[0].y1, contacts[0].y2);
  for (const a3::DirectedEdge &edge : contacts) {
    left = std::min(left, std::min(edge.x1, edge.x2));
    right = std::max(right, std::max(edge.x1, edge.x2));
    bottom = std::min(bottom, std::min(edge.y1, edge.y2));
    top = std::max(top, std::max(edge.y1, edge.y2));
  }
  return {
      floor_div(left, cell_size), floor_div(bottom, cell_size),
      floor_div(right, cell_size), floor_div(top, cell_size),
      cell_size};
}

c4::ContactBounds contact_bounds(
    const std::vector<a3::DirectedEdge> &contacts)
{
  require(!contacts.empty(), "empty CONTACT bounds input");
  c4::ContactBounds bounds = {
      std::min(contacts[0].x1, contacts[0].x2),
      std::min(contacts[0].y1, contacts[0].y2),
      std::max(contacts[0].x1, contacts[0].x2),
      std::max(contacts[0].y1, contacts[0].y2)};
  for (const a3::DirectedEdge &edge : contacts) {
    bounds.left =
        std::min(bounds.left, std::min(edge.x1, edge.x2));
    bounds.bottom =
        std::min(bounds.bottom, std::min(edge.y1, edge.y2));
    bounds.right =
        std::max(bounds.right, std::max(edge.x1, edge.x2));
    bounds.top =
        std::max(bounds.top, std::max(edge.y1, edge.y2));
  }
  return bounds;
}

CellSpan edge_span(
    const a3::DirectedEdge &edge, const HostGrid &grid,
    std::int64_t expansion)
{
  return {
      floor_div(
          checked_add(std::min(edge.x1, edge.x2), -expansion),
          grid.cell_size),
      floor_div(
          checked_add(std::min(edge.y1, edge.y2), -expansion),
          grid.cell_size),
      floor_div(
          checked_add(std::max(edge.x1, edge.x2), expansion),
          grid.cell_size),
      floor_div(
          checked_add(std::max(edge.y1, edge.y2), expansion),
          grid.cell_size)};
}

bool clip_to_grid(CellSpan *span, const HostGrid &grid)
{
  if (span->x1 < grid.base_x || span->x0 > grid.maximum_x ||
      span->y1 < grid.base_y || span->y0 > grid.maximum_y) {
    return false;
  }
  span->x0 = std::max(span->x0, grid.base_x);
  span->x1 = std::min(span->x1, grid.maximum_x);
  span->y0 = std::max(span->y0, grid.base_y);
  span->y1 = std::min(span->y1, grid.maximum_y);
  return true;
}

bool spans_intersect(
    const CellSpan &first, const CellSpan &second)
{
  return std::max(first.x0, second.x0) <=
             std::min(first.x1, second.x1) &&
         std::max(first.y0, second.y0) <=
             std::min(first.y1, second.y1);
}

Oracle oracle(
    const std::vector<mu::DirectedSegmentI64> &boundary,
    const std::vector<a3::DirectedEdge> &contacts,
    std::int64_t grid_cell_size, Aggregate *aggregate)
{
  const HostGrid grid = contact_grid(contacts, grid_cell_size);
  Oracle result;
  for (const mu::DirectedSegmentI64 &segment : boundary) {
    const a3::DirectedEdge active = directed_edge(segment);
    CellSpan active_span = edge_span(
        active, grid,
        a3::kContact4QualifiedSceneCoordinateDistance);
    const bool active_reaches_grid =
        clip_to_grid(&active_span, grid);
    for (const a3::DirectedEdge &contact : contacts) {
      ++aggregate->oracle_pairs;
      const a3::Verdict verdict =
          a3::classify_pair_bounded(
              a3::EdgePair{active, contact},
              a3::kContact4QualifiedSceneCoordinateDistance);
      if (verdict == a3::Verdict::kViolation) {
        ++result.hits;
      } else if (verdict == a3::Verdict::kUncertain) {
        ++result.uncertain;
      }
      if (active_reaches_grid &&
          spans_intersect(
              active_span, edge_span(contact, grid, 0))) {
        ++result.candidates;
      }
    }
  }
  aggregate->oracle_candidates += result.candidates;
  return result;
}

c4::ResidentContext qualified_context(
    const std::vector<a3::DirectedEdge> &contacts,
    int device, std::int64_t grid_cell_size = 2000)
{
  c4::ResidentContext context;
  context.request.contact_edges = contacts.data();
  context.request.contact_edge_count = contacts.size();
  context.request.device = device;
  context.request.grid_cell_size = grid_cell_size;
  context.request.contact_direction_contract =
      c4::ContactDirectionContract::
          validated_material_on_right_contours;
  return context;
}

c4::DeviceResidentContext qualified_device_context(
    const a3::DirectedEdge *device_edges,
    const std::vector<a3::DirectedEdge> &host_edges,
    int device, std::int64_t grid_cell_size = 2000)
{
  c4::DeviceResidentContext context;
  context.request.contacts = {
      device_edges, host_edges.size(), contact_bounds(host_edges)};
  context.request.device = device;
  context.request.grid_cell_size = grid_cell_size;
  context.request.contact_direction_contract =
      c4::ContactDirectionContract::
          validated_material_on_right_contours;
  return context;
}

void account_memory(
    const mu::GpuUnionOutput &output,
    const c4::Result &result, Aggregate *aggregate)
{
  require(
      output.device_total_bytes != 0 &&
          output.device_total_bytes == result.device_total_bytes &&
          output.device_free_begin_bytes != 0 &&
          output.device_free_low_bytes != 0 &&
          result.callback_free_begin_bytes != 0 &&
          result.callback_free_low_bytes != 0 &&
          result.post_scan_free_bytes != 0 &&
          result.callback_free_begin_bytes >=
              result.callback_free_low_bytes &&
          output.device_free_begin_bytes >=
              result.callback_free_low_bytes,
      "resident full-process memory accounting is incomplete");
  const std::uint64_t callback_incremental =
      result.callback_free_begin_bytes -
      result.callback_free_low_bytes;
  require(
      result.callback_incremental_peak_bytes ==
          callback_incremental,
      "resident callback incremental peak mismatch");
  aggregate->union_peak_bytes = std::max(
      aggregate->union_peak_bytes,
      output.device_free_begin_bytes -
          output.device_free_low_bytes);
  aggregate->callback_incremental_peak_bytes = std::max(
      aggregate->callback_incremental_peak_bytes,
      callback_incremental);
  aggregate->full_process_callback_high_water_bytes = std::max(
      aggregate->full_process_callback_high_water_bytes,
      output.device_free_begin_bytes -
          result.callback_free_low_bytes);
}

void run_case(
    const std::string &name, const std::vector<Box> &active_boxes,
    const std::vector<a3::DirectedEdge> &contacts,
    int device, Aggregate *aggregate,
    std::int64_t grid_cell_size = 2000,
    bool require_more_edges_than_cells = false,
    bool require_multiple_cells = false)
{
  const std::vector<mu::RectI64> input =
      rectangles(active_boxes);
  mu::GpuUnionLimits union_limits;
  const mu::GpuUnionOutput reference =
      mu::cpu_union_reference_for_test(input, union_limits);
  require(
      !reference.fallback && !reference.segments.empty(),
      name + ": CPU union reference declined");
  const Oracle expected =
      oracle(
          reference.segments, contacts, grid_cell_size, aggregate);
  const bool expected_empty =
      expected.hits == 0 && expected.uncertain == 0;

  c4::ResidentContext context =
      qualified_context(contacts, device, grid_cell_size);
  mu::ResidentBoundaryHook hook =
      c4::make_resident_hook(&context);
  const mu::GpuUnionOutput actual =
      mu::gpu_union_host(
          input, union_limits, device, nullptr, &hook);
  require(context.invoked, name + ": callback was not invoked");
  require(
      actual.fallback == !expected_empty &&
          actual.resident_boundary_consumer_completed ==
              expected_empty &&
          actual.segments.empty() && actual.d2h_ms == 0.0,
      name + ": empty-only callback completion contract mismatch: " +
          actual.message);
  if (!expected_empty) {
    require(
        actual.message.find(
            "resident CONTACT.4 nonempty result declined") !=
            std::string::npos,
        name + ": nonempty result did not propagate fail-closed");
    ++aggregate->expected_fallbacks;
  }
  require(
      context.result.hits == expected.hits &&
          context.result.uncertain == expected.uncertain &&
          context.result.candidate_pairs == expected.candidates &&
          context.result.certified_empty == expected_empty,
      name + ": resident result differs from exact oracle");
  require(
      context.result.boundary_cell_visits <=
              context.request.limits.max_boundary_cell_visits &&
          context.result.member_visits <=
              context.request.limits.max_member_visits &&
          context.result.candidate_pairs <=
              context.request.limits.max_pair_work,
      name + ": bounded-work telemetry exceeds its cap");
  if (require_more_edges_than_cells) {
    require(
        context.result.contact_edges >
            context.result.grid_cells,
        name + ": regression did not exceed per-cell array length");
  }
  if (require_multiple_cells) {
    require(
        context.result.grid_cells > 1,
        name + ": fixture did not exercise multiple grid cells");
  }
  account_memory(actual, context.result, aggregate);
  ++aggregate->cases;
  ++aggregate->callbacks;
  aggregate->gpu_candidates +=
      context.result.candidate_pairs;
  aggregate->maximum_contact_edges = std::max(
      aggregate->maximum_contact_edges,
      context.result.contact_edges);
  aggregate->minimum_grid_cells = std::min(
      aggregate->minimum_grid_cells,
      context.result.grid_cells);
}

void run_named_cases(int device, Aggregate *aggregate)
{
  run_case(
      "clean", {{0, 0, 100, 100}},
      box_edges({20, 20, 80, 80}), device, aggregate);
  run_case(
      "strict-nine-hit", {{0, 0, 100, 100}},
      box_edges({9, 20, 80, 80}), device, aggregate);
  run_case(
      "strict-ten-clean", {{0, 0, 100, 100}},
      box_edges({10, 20, 80, 80}), device, aggregate);
  run_case(
      "raw-overlap-false-positive-eliminated",
      {{0, 0, 60, 100}, {40, 0, 100, 100}},
      box_edges({45, 20, 55, 80}), device, aggregate);
  run_case(
      "t-junction",
      {{0, 0, 100, 40}, {40, 40, 100, 100}},
      box_edges({50, 50, 80, 80}), device, aggregate);
  run_case(
      "ring-hole-hit",
      {{0, 0, 100, 20}, {0, 80, 100, 100},
       {0, 20, 20, 80}, {80, 20, 100, 80}},
      box_edges({29, 30, 40, 70}), device, aggregate);

  // Original per-edge/per-cell bug: 128 contact edges occupy one cell.
  std::vector<a3::DirectedEdge> dense_contacts;
  for (std::int64_t index = 0; index < 32; ++index) {
    append_box_edges(
        {20 + index, 20 + index, 80 + index, 80 + index},
        &dense_contacts);
  }
  run_case(
      "contact-edges-exceed-grid-cells",
      {{0, 0, 1000, 1000}}, dense_contacts, device, aggregate,
      2000, true);

  // Small-grid fixtures cover negative floor division, coordinates exactly
  // on grid lines, complete clipping, long spans, and multi-cell duplicate
  // ownership.  Exact unique candidates are checked independently above.
  run_case(
      "negative-floor-division", {{-50, -50, 50, 50}},
      box_edges({-30, -30, -10, -10}), device, aggregate,
      10, false, true);
  run_case(
      "exact-grid-line-endpoints", {{0, 0, 60, 60}},
      box_edges({10, 10, 30, 30}), device, aggregate,
      10, false, true);
  run_case(
      "active-clipped-outside-contact-grid",
      {{-100, -100, 100, 100}},
      box_edges({0, 0, 20, 20}), device, aggregate,
      10, false, true);
  run_case(
      "long-edges-span-many-cells", {{0, 0, 100, 100}},
      box_edges({20, 20, 80, 80}), device, aggregate,
      7, false, true);
  run_case(
      "multi-cell-lexicographic-owner", {{0, 0, 100, 100}},
      box_edges({5, 20, 60, 80}), device, aggregate,
      10, false, true);
}

void run_device_view_case(
    const std::string &name,
    const std::vector<Box> &active_boxes,
    const std::vector<a3::DirectedEdge> &contacts,
    int device, Aggregate *aggregate,
    std::int64_t grid_cell_size = 2000)
{
  const std::vector<mu::RectI64> input =
      rectangles(active_boxes);
  mu::GpuUnionLimits union_limits;
  const mu::GpuUnionOutput reference =
      mu::cpu_union_reference_for_test(input, union_limits);
  require(
      !reference.fallback && !reference.segments.empty(),
      name + ": device-view CPU union reference declined");
  Aggregate oracle_accounting;
  const Oracle expected = oracle(
      reference.segments, contacts, grid_cell_size,
      &oracle_accounting);
  const bool expected_empty =
      expected.hits == 0 && expected.uncertain == 0;

  cuda_require(
      cudaSetDevice(device),
      "device-view replay cudaSetDevice");
  thrust::device_vector<a3::DirectedEdge> device_contacts(
      contacts.begin(), contacts.end());
  c4::DeviceResidentContext context =
      qualified_device_context(
          thrust::raw_pointer_cast(device_contacts.data()),
          contacts, device, grid_cell_size);
  mu::ResidentBoundaryHook hook =
      c4::make_device_resident_hook(&context);
  const mu::GpuUnionOutput actual =
      mu::gpu_union_host(
          input, union_limits, device, nullptr, &hook);

  require(
      context.invoked &&
          actual.fallback == !expected_empty &&
          actual.resident_boundary_consumer_completed ==
              expected_empty &&
          actual.segments.empty() && actual.d2h_ms == 0.0,
      name + ": device-view empty-only contract mismatch: " +
          actual.message);
  if (!expected_empty) {
    require(
        actual.message.find(
            "resident CONTACT.4 nonempty result declined") !=
            std::string::npos,
        name + ": device-view nonempty result did not fail closed");
    ++aggregate->device_view_expected_fallbacks;
  }
  require(
      context.result.contact_h2d_ms == 0.0 &&
          context.result.contact_edges == contacts.size() &&
          context.result.hits == expected.hits &&
          context.result.uncertain == expected.uncertain &&
          context.result.candidate_pairs == expected.candidates &&
          context.result.certified_empty == expected_empty,
      name + ": device-view result differs from exact oracle");
  ++aggregate->device_view_cases;
}

void run_device_view_cases(int device, Aggregate *aggregate)
{
  run_device_view_case(
      "device-clean", {{0, 0, 100, 100}},
      box_edges({20, 20, 80, 80}), device, aggregate);
  run_device_view_case(
      "device-strict-nine-hit", {{0, 0, 100, 100}},
      box_edges({9, 20, 80, 80}), device, aggregate);
  run_device_view_case(
      "device-negative-multicell", {{-50, -50, 50, 50}},
      box_edges({-30, -30, -10, -10}), device, aggregate, 10);
  run_device_view_case(
      "device-lexicographic-owner", {{0, 0, 100, 100}},
      box_edges({5, 20, 60, 80}), device, aggregate, 10);
}

mu::GpuUnionOutput run_context(
    const std::vector<mu::RectI64> &input, int device,
    c4::ResidentContext *context)
{
  mu::GpuUnionLimits union_limits;
  mu::ResidentBoundaryHook hook =
      c4::make_resident_hook(context);
  return mu::gpu_union_host(
      input, union_limits, device, nullptr, &hook);
}

mu::GpuUnionOutput run_device_context(
    const std::vector<mu::RectI64> &input, int device,
    c4::DeviceResidentContext *context)
{
  mu::GpuUnionLimits union_limits;
  mu::ResidentBoundaryHook hook =
      c4::make_device_resident_hook(context);
  return mu::gpu_union_host(
      input, union_limits, device, nullptr, &hook);
}

void require_decline(
    const std::string &name, const mu::GpuUnionOutput &output,
    const c4::ResidentContext &context,
    const std::string &message_fragment)
{
  require(
      output.fallback && context.invoked &&
          !output.resident_boundary_consumer_completed &&
          output.segments.empty() &&
          output.message.find(message_fragment) !=
              std::string::npos,
      name + " did not propagate fail-closed: " + output.message);
}

void require_device_decline(
    const std::string &name, const mu::GpuUnionOutput &output,
    const c4::DeviceResidentContext &context,
    const std::string &message_fragment)
{
  require(
      output.fallback && context.invoked &&
          !output.resident_boundary_consumer_completed &&
          output.segments.empty() &&
          output.message.find(message_fragment) !=
              std::string::npos,
      name + " device view did not fail closed: " + output.message);
}

void run_capacity_cases(int device)
{
  const std::vector<mu::RectI64> input =
      rectangles({{0, 0, 100, 100}});
  const std::vector<a3::DirectedEdge> contacts =
      box_edges({5, 20, 80, 80});

  c4::ResidentContext membership =
      qualified_context(contacts, device, 10);
  membership.request.limits.max_memberships = 1;
  require_decline(
      "membership capacity", run_context(input, device, &membership),
      membership, "resident CONTACT.4 membership gate declined");

  c4::ResidentContext boundary_edge =
      qualified_context(contacts, device, 10);
  boundary_edge.request.limits.max_cells_per_boundary_edge = 1;
  require_decline(
      "per-boundary-edge capacity",
      run_context(input, device, &boundary_edge), boundary_edge,
      "resident CONTACT.4 boundary traversal gate declined");

  c4::ResidentContext boundary_total =
      qualified_context(contacts, device, 10);
  boundary_total.request.limits.max_boundary_cell_visits = 1;
  require_decline(
      "boundary-cell total capacity",
      run_context(input, device, &boundary_total), boundary_total,
      "resident CONTACT.4 boundary traversal gate declined");

  c4::ResidentContext member_visits =
      qualified_context(contacts, device, 10);
  member_visits.request.limits.max_member_visits = 1;
  require_decline(
      "member-visit capacity",
      run_context(input, device, &member_visits), member_visits,
      "resident CONTACT.4 query gate declined");

  c4::ResidentContext pair_work =
      qualified_context(contacts, device, 10);
  pair_work.request.limits.max_pair_work = 1;
  require_decline(
      "candidate-pair capacity",
      run_context(input, device, &pair_work), pair_work,
      "resident CONTACT.4 query gate declined");
}

void run_contract_cases(int device)
{
  const std::vector<mu::RectI64> input =
      rectangles({{0, 0, 100, 100}});
  const std::vector<a3::DirectedEdge> contacts =
      box_edges({20, 20, 80, 80});

  c4::ResidentContext unspecified;
  unspecified.request.contact_edges = contacts.data();
  unspecified.request.contact_edge_count = contacts.size();
  unspecified.request.device = device;
  require_decline(
      "unspecified direction contract",
      run_context(input, device, &unspecified), unspecified,
      "resident CONTACT.4 direction contract declined");

  std::vector<a3::DirectedEdge> reversed = contacts;
  std::reverse(reversed.begin(), reversed.end());
  for (a3::DirectedEdge &edge : reversed) {
    std::swap(edge.x1, edge.x2);
    std::swap(edge.y1, edge.y2);
  }
  c4::ResidentContext reversed_context =
      qualified_context(reversed, device);
  require_decline(
      "reversed contour",
      run_context(input, device, &reversed_context),
      reversed_context,
      "resident CONTACT.4 contour is not material-on-right");

  c4::ResidentContext mismatched_device =
      qualified_context(contacts, device + 1);
  require_decline(
      "device mismatch",
      run_context(input, device, &mismatched_device),
      mismatched_device,
      "resident CONTACT.4 device contract declined");

  c4::ResidentContext stream_context =
      qualified_context(contacts, device);
  bool stream_declined = false;
  try {
    c4::consume_boundary_hook(
        reinterpret_cast<cudaStream_t>(UINTPTR_MAX),
        nullptr, 0, nullptr, 0, &stream_context);
  } catch (const std::exception &error) {
    stream_declined =
        std::string(error.what()).find(
            "requires the default CUDA stream") !=
        std::string::npos;
  }
  require(
      stream_declined && stream_context.invoked,
      "nondefault stream contract did not decline before device work");

  c4::ResidentContext empty_context;
  mu::ResidentBoundaryHook empty_hook =
      c4::make_resident_hook(&empty_context);
  mu::GpuUnionLimits limits;
  const mu::GpuUnionOutput empty_output =
      mu::gpu_union_host({}, limits, device, nullptr, &empty_hook);
  require(
      !empty_output.fallback && !empty_context.invoked &&
          !empty_output.resident_boundary_consumer_completed,
      "empty union success was incorrectly treated as a certificate");
}

void run_device_view_contract_cases(int device)
{
  cuda_require(
      cudaSetDevice(device),
      "device-view contract cudaSetDevice");
  const std::vector<mu::RectI64> input =
      rectangles({{0, 0, 100, 100}});
  const std::vector<a3::DirectedEdge> contacts =
      box_edges({20, 20, 80, 80});
  thrust::device_vector<a3::DirectedEdge> device_contacts(
      contacts.begin(), contacts.end());
  const a3::DirectedEdge *const device_pointer =
      thrust::raw_pointer_cast(device_contacts.data());

  c4::DeviceResidentContext null_pointer =
      qualified_device_context(nullptr, contacts, device);
  require_device_decline(
      "null pointer",
      run_device_context(input, device, &null_pointer),
      null_pointer, "device view pointer is null");

  c4::DeviceResidentContext host_pointer =
      qualified_device_context(contacts.data(), contacts, device);
  require_device_decline(
      "host pointer",
      run_device_context(input, device, &host_pointer),
      host_pointer, "device view");

  c4::DeviceResidentContext missing_contract =
      qualified_device_context(device_pointer, contacts, device);
  missing_contract.request.contact_direction_contract =
      c4::ContactDirectionContract::unspecified;
  require_device_decline(
      "missing producer contract",
      run_device_context(input, device, &missing_contract),
      missing_contract, "direction contract declined");

  c4::DeviceResidentContext undersized_bounds =
      qualified_device_context(device_pointer, contacts, device);
  --undersized_bounds.request.contacts.bounds.right;
  require_device_decline(
      "undersized bounds",
      run_device_context(input, device, &undersized_bounds),
      undersized_bounds, "device contact gate declined");

  const std::vector<a3::DirectedEdge> diagonal = {
      {0, 0, 10, 10}};
  thrust::device_vector<a3::DirectedEdge> device_diagonal(
      diagonal.begin(), diagonal.end());
  c4::DeviceResidentContext invalid_edge =
      qualified_device_context(
          thrust::raw_pointer_cast(device_diagonal.data()),
          diagonal, device);
  require_device_decline(
      "diagonal edge",
      run_device_context(input, device, &invalid_edge),
      invalid_edge, "device contact gate declined");
}

void run_random_cases(int device, Aggregate *aggregate)
{
  std::mt19937_64 random(UINT64_C(0xc04a4e5eeda11));
  for (std::uint32_t case_id = 0; case_id < 64; ++case_id) {
    std::vector<Box> active;
    const std::uint32_t active_count =
        1 + static_cast<std::uint32_t>(random() % 5);
    for (std::uint32_t index = 0; index < active_count; ++index) {
      const std::int64_t left =
          static_cast<std::int64_t>(random() % 160) - 80;
      const std::int64_t bottom =
          static_cast<std::int64_t>(random() % 160) - 80;
      active.push_back(
          {left, bottom,
           left + 8 + static_cast<std::int64_t>(random() % 80),
           bottom + 8 + static_cast<std::int64_t>(random() % 80)});
    }
    std::vector<a3::DirectedEdge> contacts;
    const std::uint32_t contact_count =
        1 + static_cast<std::uint32_t>(random() % 6);
    for (std::uint32_t index = 0; index < contact_count; ++index) {
      const std::int64_t left =
          static_cast<std::int64_t>(random() % 180) - 90;
      const std::int64_t bottom =
          static_cast<std::int64_t>(random() % 180) - 90;
      append_box_edges(
          {left, bottom,
           left + 4 + static_cast<std::int64_t>(random() % 50),
           bottom + 4 + static_cast<std::int64_t>(random() % 50)},
          &contacts);
    }
    run_case(
        "random-" + std::to_string(case_id),
        active, contacts, device, aggregate, 17);
  }
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    int device = 0;
    if (argc == 3 && std::string(argv[1]) == "--device") {
      device = std::stoi(argv[2]);
    } else if (argc != 1) {
      throw std::runtime_error(
          "usage: contact4_union_resident_replay [--device N]");
    }
    Aggregate aggregate;
    run_named_cases(device, &aggregate);
    run_device_view_cases(device, &aggregate);
    run_capacity_cases(device);
    run_contract_cases(device);
    run_device_view_contract_cases(device);
    run_random_cases(device, &aggregate);
    require(
        aggregate.oracle_candidates == aggregate.gpu_candidates,
        "aggregate unique-candidate census mismatch");
    std::cout
        << "CONTACT4_UNION_RESIDENT_REPLAY PASS"
        << " cases=" << aggregate.cases
        << " callbacks=" << aggregate.callbacks
        << " expected_nonempty_fallbacks="
        << aggregate.expected_fallbacks
        << " oracle_pairs=" << aggregate.oracle_pairs
        << " exact_unique_candidates="
        << aggregate.oracle_candidates
        << " gpu_candidates=" << aggregate.gpu_candidates
        << " max_contact_edges=" << aggregate.maximum_contact_edges
        << " min_grid_cells=" << aggregate.minimum_grid_cells
        << " edge_gt_cell_regression=1"
        << " multi_cell_fixtures=5"
        << " capacity_fallbacks=5"
        << " contract_fallbacks=4"
        << " device_view_exact_cases="
        << aggregate.device_view_cases
        << " device_view_expected_fallbacks="
        << aggregate.device_view_expected_fallbacks
        << " device_view_contract_fallbacks=5"
        << " device_view_h2d_zero=1"
        << " empty_hook_regression=1"
        << " union_peak_mib="
        << static_cast<double>(aggregate.union_peak_bytes) /
               (1024.0 * 1024.0)
        << " callback_incremental_peak_mib="
        << static_cast<double>(
               aggregate.callback_incremental_peak_bytes) /
               (1024.0 * 1024.0)
        << " full_process_callback_high_water_mib="
        << static_cast<double>(
               aggregate.full_process_callback_high_water_bytes) /
               (1024.0 * 1024.0)
        << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "CONTACT4_UNION_RESIDENT_REPLAY FAIL reason='"
        << error.what() << "'\n";
    return 1;
  }
}
