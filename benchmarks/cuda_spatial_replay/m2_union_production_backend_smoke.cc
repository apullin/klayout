/*
 * Adversarial DSO-level gate for the production raw-M2 union adapter.
 *
 * The clean fixture places one integer-grid L under all eight orthogonal
 * transforms.  Its expected boundary is built independently from occupied
 * unit cells, not by either the adapter decomposition or the shared union
 * core.  Mutations then exercise every trust-boundary class before the result
 * ownership contract is checked.
 */

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C" void
klayout_cuda_spatial_m2_union_test_fault_v1(std::uint32_t selector);
extern "C" std::uint64_t
klayout_cuda_spatial_m2_union_test_counter_v1(std::uint32_t selector);

namespace {

using Context = klayout_cuda_spatial_m1_width_space_context_v1;
using Cell = klayout_cuda_spatial_m1_width_space_cell_v1;
using Polygon = klayout_cuda_spatial_m1_width_space_polygon_v1;
using Edge = klayout_cuda_spatial_m1_width_space_edge_v1;
using Request = klayout_cuda_spatial_m2_union_request_v1;
using Result = klayout_cuda_spatial_m2_union_result_v1;
using Segment = klayout_cuda_spatial_m2_union_segment_v1;

struct CanonicalDigest
{
  void bytes(const void *data, std::size_t size) { sha.update(data, size); }

  void u32(std::uint32_t value)
  {
    std::uint8_t encoded[4];
    for (unsigned int byte = 0; byte < 4; ++byte) {
      encoded[byte] =
          static_cast<std::uint8_t>(value >> (byte * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void u64(std::uint64_t value)
  {
    std::uint8_t encoded[8];
    for (unsigned int byte = 0; byte < 8; ++byte) {
      encoded[byte] =
          static_cast<std::uint8_t>(value >> (byte * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void i64(std::int64_t value)
  {
    u64(static_cast<std::uint64_t>(value));
  }

  std::array<std::uint8_t, 32> finish() { return sha.finish(); }

  db::cuda_active3_digest::Sha256 sha;
};

std::pair<std::int64_t, std::int64_t>
transform_point(const Context &context, std::int64_t x, std::int64_t y)
{
  std::int64_t transformed_x = 0;
  std::int64_t transformed_y = 0;
  switch (context.transform_code) {
  case 0: transformed_x = x; transformed_y = y; break;
  case 1: transformed_x = -y; transformed_y = x; break;
  case 2: transformed_x = -x; transformed_y = -y; break;
  case 3: transformed_x = y; transformed_y = -x; break;
  case 4: transformed_x = x; transformed_y = -y; break;
  case 5: transformed_x = y; transformed_y = x; break;
  case 6: transformed_x = -x; transformed_y = y; break;
  case 7: transformed_x = -y; transformed_y = -x; break;
  default: throw std::runtime_error("invalid fixture transform");
  }
  return {
      transformed_x + context.tx,
      transformed_y + context.ty};
}

bool segment_less(const Segment &first, const Segment &second)
{
  return std::tie(
             first.axis, first.side, first.fixed, first.lo, first.hi) <
         std::tie(
             second.axis, second.side, second.fixed, second.lo,
             second.hi);
}

bool same_line(const Segment &first, const Segment &second)
{
  return first.axis == second.axis &&
         first.side == second.side &&
         first.fixed == second.fixed;
}

std::uint64_t boundary_fnv64(
    const Segment *segments, std::uint64_t count)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&hash](std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
      hash *= UINT64_C(1099511628211);
    }
  };
  mix(count);
  for (std::uint64_t index = 0; index < count; ++index) {
    mix(segments[index].axis);
    mix(static_cast<std::uint32_t>(segments[index].side));
    mix(static_cast<std::uint64_t>(segments[index].fixed));
    mix(static_cast<std::uint64_t>(segments[index].lo));
    mix(static_cast<std::uint64_t>(segments[index].hi));
  }
  return hash;
}

struct Scene
{
  std::vector<Context> contexts;
  std::vector<std::uint32_t> metal_contexts;
  std::vector<std::uint64_t> polygon_offsets;
  std::vector<std::uint64_t> edge_offsets;
  std::vector<Cell> cells;
  std::vector<Polygon> polygons;
  std::vector<Edge> edges;
  Request request{};

  void bind()
  {
    request.contexts = contexts.data();
    request.context_count = contexts.size();
    request.metal_contexts = metal_contexts.data();
    request.metal_context_count = metal_contexts.size();
    request.context_polygon_offsets = polygon_offsets.data();
    request.context_polygon_offset_count = polygon_offsets.size();
    request.context_edge_offsets = edge_offsets.data();
    request.context_edge_offset_count = edge_offsets.size();
    request.cells = cells.data();
    request.cell_count = cells.size();
    request.polygons = polygons.data();
    request.polygon_count = polygons.size();
    request.edges = edges.data();
    request.edge_count = edges.size();
  }

  void digest()
  {
    bind();
    static const char magic[8] =
        {'K', 'M', '2', 'R', 'A', 'W', '0', '1'};
    CanonicalDigest output;
    output.bytes(magic, sizeof(magic));
    output.u32(request.format_version);
    output.u32(request.dbu_per_micron);
    output.u32(request.root_cell);
    output.u32(0);
    output.u64(request.context_count);
    output.u64(request.metal_context_count);
    output.u64(request.cell_count);
    output.u64(request.polygon_count);
    output.u64(request.edge_count);
    output.u64(request.flat_polygon_count);
    output.u64(request.flat_edge_count);
    output.i64(request.scene_left);
    output.i64(request.scene_bottom);
    output.i64(request.scene_right);
    output.i64(request.scene_top);
    for (const Context &context : contexts) {
      output.i64(context.tx);
      output.i64(context.ty);
      output.u32(context.cell_id);
      output.u32(context.transform_code);
    }
    for (std::size_t index = 0;
         index < metal_contexts.size(); ++index) {
      output.u32(metal_contexts[index]);
      output.u64(polygon_offsets[index]);
      output.u64(edge_offsets[index]);
    }
    for (const Cell &cell : cells) {
      output.u64(cell.source_cell_index);
      output.u64(cell.polygon_begin);
      output.u64(cell.edge_begin);
      output.u32(cell.polygon_count);
      output.u32(cell.edge_count);
    }
    for (const Polygon &polygon : polygons) {
      output.u64(polygon.edge_begin);
      output.i64(polygon.left);
      output.i64(polygon.bottom);
      output.i64(polygon.right);
      output.i64(polygon.top);
      output.u32(polygon.polygon_id);
      output.u32(polygon.edge_count);
    }
    for (const Edge &edge : edges) {
      output.i64(edge.x1);
      output.i64(edge.y1);
      output.i64(edge.x2);
      output.i64(edge.y2);
    }
    const auto value = output.finish();
    std::copy(
        value.begin(), value.end(), request.scene_digest);
  }
};

void update_bounds(Scene *scene)
{
  bool have_bounds = false;
  for (const Context &context : scene->contexts) {
    for (const Polygon &polygon : scene->polygons) {
      const std::int64_t xs[2] =
          {polygon.left, polygon.right};
      const std::int64_t ys[2] =
          {polygon.bottom, polygon.top};
      for (int x_index = 0; x_index < 2; ++x_index) {
        for (int y_index = 0; y_index < 2; ++y_index) {
          const auto point = transform_point(
              context, xs[x_index], ys[y_index]);
          if (!have_bounds) {
            scene->request.scene_left =
                scene->request.scene_right = point.first;
            scene->request.scene_bottom =
                scene->request.scene_top = point.second;
            have_bounds = true;
          } else {
            scene->request.scene_left = std::min(
                scene->request.scene_left, point.first);
            scene->request.scene_bottom = std::min(
                scene->request.scene_bottom, point.second);
            scene->request.scene_right = std::max(
                scene->request.scene_right, point.first);
            scene->request.scene_top = std::max(
                scene->request.scene_top, point.second);
          }
        }
      }
    }
  }
}

Scene make_scene()
{
  Scene scene;
  for (std::uint32_t code = 0; code < 8; ++code) {
    scene.contexts.push_back(
        {static_cast<std::int64_t>(code) * 20,
         static_cast<std::int64_t>(code) * 20, 0, code});
    scene.metal_contexts.push_back(code);
    scene.polygon_offsets.push_back(code);
    scene.edge_offsets.push_back(
        static_cast<std::uint64_t>(code) * 6);
  }
  scene.cells.push_back({7, 0, 0, 1, 6});
  scene.polygons.push_back({0, 0, 0, 4, 4, 0, 6});
  scene.edges = {
      {0, 0, 0, 4}, {0, 4, 2, 4}, {2, 4, 2, 2},
      {2, 2, 4, 2}, {4, 2, 4, 0}, {4, 0, 0, 0}};

  Request &request = scene.request;
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.root_cell = 0;
  request.device = 0;
  request.context_record_bytes = sizeof(Context);
  request.cell_record_bytes = sizeof(Cell);
  request.polygon_record_bytes = sizeof(Polygon);
  request.edge_record_bytes = sizeof(Edge);
  request.flat_polygon_count = 8;
  request.flat_edge_count = 48;
  request.max_contexts = 8;
  request.max_rectangles = 16;
  request.max_x_slabs = 256;
  request.max_memberships = 1024;
  request.max_events = 2048;
  request.max_raw_segments = 1024;
  request.max_segments = 256;
  request.max_slabs_per_rectangle = 256;
  update_bounds(&scene);
  scene.digest();
  return scene;
}

Scene make_rectangle_scene(std::int64_t width, std::int64_t height)
{
  Scene scene;
  scene.contexts.push_back({0, 0, 0, 0});
  scene.metal_contexts.push_back(0);
  scene.polygon_offsets.push_back(0);
  scene.edge_offsets.push_back(0);
  scene.cells.push_back({7, 0, 0, 1, 4});
  scene.polygons.push_back(
      {0, 0, 0, width, height, 0, 4});
  scene.edges = {
      {0, 0, 0, height},
      {0, height, width, height},
      {width, height, width, 0},
      {width, 0, 0, 0}};

  Request &request = scene.request;
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.root_cell = 0;
  request.device = 0;
  request.context_record_bytes = sizeof(Context);
  request.cell_record_bytes = sizeof(Cell);
  request.polygon_record_bytes = sizeof(Polygon);
  request.edge_record_bytes = sizeof(Edge);
  request.flat_polygon_count = 1;
  request.flat_edge_count = 4;
  request.max_contexts = 1;
  request.max_rectangles = 1;
  request.max_x_slabs = 16;
  request.max_memberships = 64;
  request.max_events = 128;
  request.max_raw_segments = 64;
  request.max_segments = 32;
  request.max_slabs_per_rectangle = 16;
  update_bounds(&scene);
  scene.digest();
  return scene;
}

std::vector<Segment>
independent_boundary(const Scene &scene)
{
  std::set<std::pair<std::int64_t, std::int64_t>> cells;
  for (const Context &context : scene.contexts) {
    for (std::int64_t x = 0; x < 4; ++x) {
      for (std::int64_t y = 0; y < 4; ++y) {
        if (x >= 2 && y >= 2) continue;
        const auto first = transform_point(context, x, y);
        const auto second = transform_point(
            context, x + 1, y + 1);
        cells.insert(
            {std::min(first.first, second.first),
             std::min(first.second, second.second)});
      }
    }
  }

  std::vector<Segment> raw;
  for (const auto &cell : cells) {
    const std::int64_t x = cell.first;
    const std::int64_t y = cell.second;
    if (!cells.count({x, y - 1})) {
      raw.push_back(
          {y, x, x + 1, -1,
           KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL});
    }
    if (!cells.count({x, y + 1})) {
      raw.push_back(
          {y + 1, x, x + 1, 1,
           KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL});
    }
    if (!cells.count({x - 1, y})) {
      raw.push_back(
          {x, y, y + 1, -1,
           KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL});
    }
    if (!cells.count({x + 1, y})) {
      raw.push_back(
          {x + 1, y, y + 1, 1,
           KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL});
    }
  }
  std::sort(raw.begin(), raw.end(), segment_less);
  std::vector<Segment> result;
  for (const Segment &segment : raw) {
    if (!result.empty() &&
        same_line(result.back(), segment) &&
        result.back().hi == segment.lo) {
      result.back().hi = segment.hi;
    } else {
      result.push_back(segment);
    }
  }
  return result;
}

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

void expect_decline(Scene scene, const std::string &label,
                    std::uint32_t expected_status = UINT32_MAX,
                    std::uint32_t expected_flag = 0)
{
  scene.bind();
  Result result{};
  const int status =
      klayout_cuda_spatial_run_m2_union_boundary_v1(
          &scene.request, &result);
  require(
      status != KLAYOUT_CUDA_SPATIAL_OK &&
          result.status != KLAYOUT_CUDA_SPATIAL_OK &&
          result.disposition ==
              KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN &&
          !result.segments && !result.segment_count &&
          result.certified_empty_mask == 0 &&
          result.certificate_reserved == 0 &&
          result.suffix_total_ns == 0,
      label + " did not fail closed");
  if (expected_status != UINT32_MAX) {
    require(
        result.status == expected_status,
        label + " returned the wrong status");
  }
  if (expected_flag) {
    require(
        result.fallback_flags & expected_flag,
        label + " returned the wrong fallback flag");
  }
  klayout_cuda_spatial_release_m2_union_boundary_v1(&result);
}

void reverse_contour(Scene *scene)
{
  const std::array<std::pair<std::int64_t, std::int64_t>, 6> points = {{
      {0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}}};
  for (std::size_t index = 0; index < points.size(); ++index) {
    const auto first = points[index];
    const auto second = points[(index + 1) % points.size()];
    scene->edges[index] = {
        first.first, first.second, second.first, second.second};
  }
}

}  // namespace

int main()
{
  try {
    Scene clean = make_scene();
    const std::vector<Segment> expected =
        independent_boundary(clean);
    clean.bind();
    Result result{};
    const int status =
        klayout_cuda_spatial_run_m2_union_boundary_v1(
            &clean.request, &result);
    require(
        status == KLAYOUT_CUDA_SPATIAL_OK &&
            result.status == KLAYOUT_CUDA_SPATIAL_OK &&
            result.disposition ==
                KLAYOUT_CUDA_SPATIAL_M2_UNION_COMPLETE,
        std::string("clean all-transform scene declined: ") +
            result.message);
    require(
        result.rectangle_count == 16 &&
            result.context_count == 8 &&
            result.metal_context_count == 8 &&
            result.polygon_count == 1 &&
            result.edge_count == 6 &&
            result.flat_polygon_count == 8 &&
            result.flat_edge_count == 48 &&
            result.event_count == 2 * result.membership_count &&
            result.raw_segment_count >= result.segment_count,
        "clean result counters are inconsistent");
    require(
        result.segment_count == expected.size() &&
            std::equal(
                expected.begin(), expected.end(), result.segments,
                [](const Segment &first, const Segment &second) {
                  return first.fixed == second.fixed &&
                         first.lo == second.lo &&
                         first.hi == second.hi &&
                         first.side == second.side &&
                         first.axis == second.axis;
                }),
        "clean boundary differs from independent unit-cell oracle");
    require(
        result.boundary_fnv64 ==
            boundary_fnv64(expected.data(), expected.size()),
        "clean boundary digest differs from independent oracle");
    require(
        result.certified_empty_mask == 0 &&
            result.certificate_reserved == 0 &&
            result.suffix_total_ns == 0,
        "geometry-only opcode published an M2 suffix certificate");
    require(
        result.setup_ns && result.h2d_ns &&
            result.rectangle_expand_ns &&
            result.x_membership_ns && result.strip_scan_ns &&
            result.boundary_ns && result.d2h_ns &&
            result.total_ns,
        "clean result omitted a production timing");
    const Segment *owned = result.segments;
    klayout_cuda_spatial_release_m2_union_boundary_v1(&result);
    require(
        owned && !result.segments && !result.segment_count,
        "release did not clear backend-owned output");
    klayout_cuda_spatial_release_m2_union_boundary_v1(&result);

    Scene suffix_clean = make_rectangle_scene(200, 500);
    suffix_clean.bind();
    Result suffix_result{};
    const int suffix_status =
        klayout_cuda_spatial_run_m2_union_boundary_v1(
            &suffix_clean.request, &suffix_result);
    require(
        suffix_status == KLAYOUT_CUDA_SPATIAL_OK &&
            suffix_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
            suffix_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_M2_UNION_COMPLETE &&
            suffix_result.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY &&
            suffix_result.certificate_reserved == 0 &&
            suffix_result.suffix_total_ns &&
            suffix_result.suffix_total_ns <= suffix_result.total_ns &&
            suffix_result.segment_count == 4 &&
            suffix_result.segments,
        std::string("clean M2 suffix scene declined: ") +
            suffix_result.message);
    klayout_cuda_spatial_release_m2_union_boundary_v1(
        &suffix_result);

    require(
        klayout_cuda_spatial_m2_union_test_counter_v1(3) == 0,
        "M2 backend retained ownership before the fault gate");
    klayout_cuda_spatial_m2_union_test_fault_v1(1);
    Scene suffix_fault = make_rectangle_scene(200, 500);
    expect_decline(
        suffix_fault, "post-suffix output-validation fault",
        KLAYOUT_CUDA_SPATIAL_ERROR,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT);
    require(
        klayout_cuda_spatial_m2_union_test_counter_v1(3) == 0,
        "post-suffix fault leaked an M2 boundary allocation");

    Scene suffix_hit = make_rectangle_scene(600, 600);
    expect_decline(
        suffix_hit, "M2 suffix F270 hit",
        KLAYOUT_CUDA_SPATIAL_FALLBACK);

    Scene malformed_stride = clean;
    --malformed_stride.request.context_record_bytes;
    expect_decline(
        malformed_stride, "record-size mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT);

    Scene malformed_options = clean;
    malformed_options.request.option_flags ^= 1u;
    expect_decline(
        malformed_options, "option mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT);

    Scene malformed_reserved = clean;
    malformed_reserved.request.reserved1[1] = 1;
    expect_decline(
        malformed_reserved, "reserved mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT);

    Scene digest_bad = clean;
    digest_bad.request.scene_digest[7] ^= 0x80u;
    expect_decline(
        digest_bad, "digest mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST);

    Scene span_bad = clean;
    span_bad.cells[0].polygon_begin = 1;
    span_bad.digest();
    expect_decline(
        span_bad, "cell-span mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN);

    Scene counterclockwise = clean;
    reverse_contour(&counterclockwise);
    counterclockwise.digest();
    expect_decline(
        counterclockwise, "contour-orientation mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT);

    Scene open_contour = clean;
    open_contour.edges[2].y2 = 1;
    open_contour.digest();
    expect_decline(
        open_contour, "open-contour mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT);

    Scene offset_bad = clean;
    offset_bad.polygon_offsets[3] = 2;
    offset_bad.digest();
    expect_decline(
        offset_bad, "flattened-offset mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT);

    Scene bounds_bad = clean;
    ++bounds_bad.request.scene_right;
    bounds_bad.digest();
    expect_decline(
        bounds_bad, "world-bounds mutation",
        KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT);

    Scene rectangle_cap = clean;
    rectangle_cap.request.max_rectangles = 15;
    expect_decline(
        rectangle_cap, "rectangle-cap mutation",
        KLAYOUT_CUDA_SPATIAL_FALLBACK,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY);

    Scene slab_cap = clean;
    slab_cap.request.max_slabs_per_rectangle = 1;
    expect_decline(
        slab_cap, "slab-span mutation",
        KLAYOUT_CUDA_SPATIAL_FALLBACK,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY);

    Scene coordinate_bad = clean;
    coordinate_bad.contexts[5].tx =
        INT64_C(1000000000000);
    coordinate_bad.digest();
    expect_decline(
        coordinate_bad, "coordinate-overflow mutation",
        KLAYOUT_CUDA_SPATIAL_FALLBACK,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW);

    std::cout
        << "M2_UNION_PRODUCTION_BACKEND_SMOKE PASS"
        << " transforms=8 rectangles=16 segments="
        << expected.size()
        << " fnv64="
        << boundary_fnv64(expected.data(), expected.size())
        << " adversarial=12 release_idempotent=1"
        << " suffix_clean=1 suffix_fault_zeroed=1"
        << " suffix_hit_fallback=1\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "M2_UNION_PRODUCTION_BACKEND_SMOKE FAIL: "
        << error.what() << "\n";
    return 1;
  }
}
