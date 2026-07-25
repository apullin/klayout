/*
 * Full production gate for the raw-hierarchy M2 union DSO entry point.
 *
 * KACTSCN1 is used only as a pinned, independently validated source of the
 * compact hierarchy and local contours.  This gate rebuilds the KM2RAW01 ABI
 * scene (without host-expanding world rectangles), calls the production DSO,
 * and compares every returned segment with the independently generated stock
 * KLayout merged-boundary oracle.
 */

#define main klayout_cuda_embedded_active3_scene_main
#include "active3_scene_island.cu"
#undef main

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"
#include "m2_merged_boundary_oracle.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace production_gate {

namespace oracle = klayout_cuda::m2_boundary_oracle;

using AbiContext =
    klayout_cuda_spatial_m1_width_space_context_v1;
using AbiCell =
    klayout_cuda_spatial_m1_width_space_cell_v1;
using AbiPolygon =
    klayout_cuda_spatial_m1_width_space_polygon_v1;
using AbiEdge =
    klayout_cuda_spatial_m1_width_space_edge_v1;
using Request =
    klayout_cuda_spatial_m2_union_request_v1;
using Result =
    klayout_cuda_spatial_m2_union_result_v1;
using Segment =
    klayout_cuda_spatial_m2_union_segment_v1;
using GateClock = std::chrono::steady_clock;

constexpr char kProductionKactSha256[] =
    "dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2";
constexpr char kOracleFileSha256[] =
    "980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f";
constexpr char kOracleSceneSha256[] =
    "441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480";
constexpr char kBoundarySha256[] =
    "94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d";
constexpr std::uint64_t kFlatPolygons = UINT64_C(22945976);
constexpr std::uint64_t kFlatEdges = UINT64_C(91784840);
constexpr std::uint64_t kRectangles = UINT64_C(22946444);
constexpr std::uint64_t kXSlabs = UINT64_C(46383);
constexpr std::uint64_t kMemberships = UINT64_C(92386704);
constexpr std::uint64_t kEvents = UINT64_C(184773408);
constexpr std::uint64_t kStrips = UINT64_C(3691466);
constexpr std::uint64_t kRawSegments = UINT64_C(9575624);
constexpr std::uint64_t kSegments = UINT64_C(4385384);
constexpr std::uint64_t kBoundaryFnv64 =
    UINT64_C(7541395996791771514);

struct RawDigest
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

struct RawScene
{
  std::vector<AbiContext> contexts;
  std::vector<std::uint32_t> metal_contexts;
  std::vector<std::uint64_t> polygon_offsets;
  std::vector<std::uint64_t> edge_offsets;
  std::vector<AbiCell> cells;
  std::vector<AbiPolygon> polygons;
  std::vector<AbiEdge> edges;
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
};

double gate_ms(GateClock::time_point begin, GateClock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double ns_ms(std::uint64_t nanoseconds)
{
  return static_cast<double>(nanoseconds) / 1000000.0;
}

void gate_require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

std::pair<std::int64_t, std::int64_t>
raw_transform_point(const AbiContext &context,
                    std::int64_t x, std::int64_t y)
{
  __int128 transformed_x = 0;
  __int128 transformed_y = 0;
  switch (context.transform_code) {
  case 0: transformed_x = x; transformed_y = y; break;
  case 1: transformed_x = -static_cast<__int128>(y);
          transformed_y = x; break;
  case 2: transformed_x = -static_cast<__int128>(x);
          transformed_y = -static_cast<__int128>(y); break;
  case 3: transformed_x = y;
          transformed_y = -static_cast<__int128>(x); break;
  case 4: transformed_x = x;
          transformed_y = -static_cast<__int128>(y); break;
  case 5: transformed_x = y; transformed_y = x; break;
  case 6: transformed_x = -static_cast<__int128>(x);
          transformed_y = y; break;
  case 7: transformed_x = -static_cast<__int128>(y);
          transformed_y = -static_cast<__int128>(x); break;
  default: throw std::runtime_error("invalid production transform code");
  }
  transformed_x += context.tx;
  transformed_y += context.ty;
  gate_require(
      transformed_x >= std::numeric_limits<std::int64_t>::min() &&
          transformed_x <=
              std::numeric_limits<std::int64_t>::max() &&
          transformed_y >=
              std::numeric_limits<std::int64_t>::min() &&
          transformed_y <=
              std::numeric_limits<std::int64_t>::max(),
      "production raw transform overflows int64");
  return {
      static_cast<std::int64_t>(transformed_x),
      static_cast<std::int64_t>(transformed_y)};
}

void raw_add_bounds(
    const AbiContext &context, const AbiPolygon &polygon,
    bool *have_bounds, Request *request)
{
  const std::int64_t xs[2] = {polygon.left, polygon.right};
  const std::int64_t ys[2] = {polygon.bottom, polygon.top};
  for (int x_index = 0; x_index < 2; ++x_index) {
    for (int y_index = 0; y_index < 2; ++y_index) {
      const auto point = raw_transform_point(
          context, xs[x_index], ys[y_index]);
      if (!*have_bounds) {
        request->scene_left = request->scene_right = point.first;
        request->scene_bottom = request->scene_top = point.second;
        *have_bounds = true;
      } else {
        request->scene_left =
            std::min(request->scene_left, point.first);
        request->scene_bottom =
            std::min(request->scene_bottom, point.second);
        request->scene_right =
            std::max(request->scene_right, point.first);
        request->scene_top =
            std::max(request->scene_top, point.second);
      }
    }
  }
}

std::array<std::uint8_t, 32> raw_scene_digest(const RawScene &scene)
{
  const Request &request = scene.request;
  static const char magic[8] =
      {'K', 'M', '2', 'R', 'A', 'W', '0', '1'};
  RawDigest output;
  output.bytes(magic, sizeof(magic));
  output.u32(request.format_version);
  output.u32(request.dbu_per_micron);
  output.u32(request.root_cell);
  output.u32(0);
  output.u64(scene.contexts.size());
  output.u64(scene.metal_contexts.size());
  output.u64(scene.cells.size());
  output.u64(scene.polygons.size());
  output.u64(scene.edges.size());
  output.u64(request.flat_polygon_count);
  output.u64(request.flat_edge_count);
  output.i64(request.scene_left);
  output.i64(request.scene_bottom);
  output.i64(request.scene_right);
  output.i64(request.scene_top);
  for (const AbiContext &context : scene.contexts) {
    output.i64(context.tx);
    output.i64(context.ty);
    output.u32(context.cell_id);
    output.u32(context.transform_code);
  }
  for (std::size_t index = 0;
       index < scene.metal_contexts.size(); ++index) {
    output.u32(scene.metal_contexts[index]);
    output.u64(scene.polygon_offsets[index]);
    output.u64(scene.edge_offsets[index]);
  }
  for (const AbiCell &cell : scene.cells) {
    output.u64(cell.source_cell_index);
    output.u64(cell.polygon_begin);
    output.u64(cell.edge_begin);
    output.u32(cell.polygon_count);
    output.u32(cell.edge_count);
  }
  for (const AbiPolygon &polygon : scene.polygons) {
    output.u64(polygon.edge_begin);
    output.i64(polygon.left);
    output.i64(polygon.bottom);
    output.i64(polygon.right);
    output.i64(polygon.top);
    output.u32(polygon.polygon_id);
    output.u32(polygon.edge_count);
  }
  for (const AbiEdge &edge : scene.edges) {
    output.i64(edge.x1);
    output.i64(edge.y1);
    output.i64(edge.x2);
    output.i64(edge.y2);
  }
  return output.finish();
}

std::string raw_hex(const std::uint8_t *bytes, std::size_t count)
{
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < count; ++index) {
    stream << std::setw(2)
           << static_cast<unsigned int>(bytes[index]);
  }
  return stream.str();
}

RawScene build_raw_scene(const LoadedScene &source,
                         const LoweredScene &hierarchy,
                         int device,
                         std::uint32_t opcode =
                             KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY)
{
  gate_require(
      opcode ==
              KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY ||
          opcode ==
              KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY,
      "production gate received an unsupported M2 opcode");
  RawScene scene;
  gate_require(
      source.header.cell_count <=
          std::numeric_limits<std::uint32_t>::max() &&
          source.header.root_cell <=
              std::numeric_limits<std::uint32_t>::max(),
      "production KACT cell IDs exceed raw ABI");
  scene.contexts.reserve(hierarchy.contexts.size());
  for (const ContextGpu &context : hierarchy.contexts) {
    scene.contexts.push_back(
        {context.tx, context.ty, context.cell, context.transform});
  }

  scene.cells.resize(
      static_cast<std::size_t>(source.header.cell_count));
  for (std::uint64_t cell_id = 0;
       cell_id < source.header.cell_count; ++cell_id) {
    const CellRecord &source_cell = source.cells[cell_id];
    AbiCell &cell = scene.cells[static_cast<std::size_t>(cell_id)];
    cell.source_cell_index = source_cell.cell_id;
    cell.polygon_begin = scene.polygons.size();
    cell.edge_begin = scene.edges.size();
    std::uint32_t local_polygon = 0;
    for (std::uint64_t local = 0;
         local < source_cell.polygon_count; ++local) {
      const PolygonRecord &source_polygon =
          source.polygons[source_cell.polygon_begin + local];
      if (source_polygon.layer_code != kWellLayer) continue;
      gate_require(
          source_polygon.edge_count >= 4 &&
              source_polygon.edge_count <=
                  std::numeric_limits<std::uint32_t>::max(),
          "production M2 polygon edge count exceeds raw ABI");
      AbiPolygon polygon{};
      polygon.edge_begin = scene.edges.size();
      polygon.left = source_polygon.bbox[0];
      polygon.bottom = source_polygon.bbox[1];
      polygon.right = source_polygon.bbox[2];
      polygon.top = source_polygon.bbox[3];
      polygon.polygon_id = local_polygon++;
      polygon.edge_count = source_polygon.edge_count;
      scene.polygons.push_back(polygon);
      for (std::uint32_t edge_local = 0;
           edge_local < source_polygon.edge_count; ++edge_local) {
        const EdgeRecord &source_edge =
            source.edges[
                source_polygon.edge_begin + edge_local];
        gate_require(
            source_edge.layer_code == kWellLayer &&
                source_edge.polygon_id ==
                    source_polygon.polygon_id,
            "production M2 polygon/edge ownership mismatch");
        scene.edges.push_back(
            {source_edge.x1, source_edge.y1,
             source_edge.x2, source_edge.y2});
      }
    }
    const std::uint64_t polygon_count =
        scene.polygons.size() - cell.polygon_begin;
    const std::uint64_t edge_count =
        scene.edges.size() - cell.edge_begin;
    gate_require(
        polygon_count <=
            std::numeric_limits<std::uint32_t>::max() &&
            edge_count <=
                std::numeric_limits<std::uint32_t>::max(),
        "production per-cell M2 census exceeds raw ABI");
    cell.polygon_count =
        static_cast<std::uint32_t>(polygon_count);
    cell.edge_count = static_cast<std::uint32_t>(edge_count);
  }

  Request &request = scene.request;
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode = opcode;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.root_cell =
      static_cast<std::uint32_t>(source.header.root_cell);
  request.device = device;
  request.context_record_bytes = sizeof(AbiContext);
  request.cell_record_bytes = sizeof(AbiCell);
  request.polygon_record_bytes = sizeof(AbiPolygon);
  request.edge_record_bytes = sizeof(AbiEdge);
  request.max_contexts = UINT64_C(4000000);
  request.max_rectangles = UINT64_C(32000000);
  request.max_x_slabs = UINT64_C(32000000);
  request.max_memberships = UINT64_C(100000000);
  request.max_events = UINT64_C(200000000);
  request.max_raw_segments = UINT64_C(12000000);
  request.max_segments = UINT64_C(8000000);
  request.max_slabs_per_rectangle = 64;

  bool have_bounds = false;
  for (std::uint32_t context_id = 0;
       context_id < scene.contexts.size(); ++context_id) {
    const AbiContext &context = scene.contexts[context_id];
    const AbiCell &cell = scene.cells[context.cell_id];
    if (!cell.polygon_count) continue;
    scene.metal_contexts.push_back(context_id);
    scene.polygon_offsets.push_back(
        request.flat_polygon_count);
    scene.edge_offsets.push_back(request.flat_edge_count);
    request.flat_polygon_count += cell.polygon_count;
    request.flat_edge_count += cell.edge_count;
    const std::uint64_t polygon_end =
        cell.polygon_begin + cell.polygon_count;
    for (std::uint64_t polygon_id = cell.polygon_begin;
         polygon_id < polygon_end; ++polygon_id) {
      raw_add_bounds(
          context,
          scene.polygons[static_cast<std::size_t>(polygon_id)],
          &have_bounds, &request);
    }
  }
  gate_require(
      have_bounds &&
          request.flat_polygon_count == kFlatPolygons &&
          request.flat_edge_count == kFlatEdges,
      "production raw flattened census differs from allowlist");
  scene.bind();
  const auto digest = raw_scene_digest(scene);
  std::copy(digest.begin(), digest.end(), request.scene_digest);
  return scene;
}

void compare_oracle(const Result &result,
                    const oracle::BoundaryOracle &expected)
{
  gate_require(
      expected.segments.size() == result.segment_count &&
          expected.boundary_fnv64 == result.boundary_fnv64,
      "production boundary census/digest differs from oracle");
  for (std::uint64_t index = 0;
       index < result.segment_count; ++index) {
    const Segment &candidate = result.segments[index];
    const oracle::DirectedSegmentI64 &reference =
        expected.segments[static_cast<std::size_t>(index)];
    if (candidate.fixed != reference.fixed ||
        candidate.lo != reference.lo ||
        candidate.hi != reference.hi ||
        candidate.side != reference.side ||
        candidate.axis !=
            static_cast<std::uint32_t>(reference.axis)) {
      throw std::runtime_error(
          "production boundary first differs from oracle at segment " +
          std::to_string(index));
    }
  }
}

void validate_counters(const Result &result)
{
  gate_require(
      result.rectangle_count == kRectangles &&
          result.x_slab_count == kXSlabs &&
          result.membership_count == kMemberships &&
          result.event_count == kEvents &&
          result.strip_interval_count == kStrips &&
          result.raw_segment_count == kRawSegments &&
          result.segment_count == kSegments &&
          result.boundary_fnv64 == kBoundaryFnv64,
      "production result counters differ from exact allowlist");
}

double median(std::vector<double> values)
{
  gate_require(!values.empty(), "cannot take an empty median");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() & 1
             ? values[middle]
             : (values[middle - 1] + values[middle]) * 0.5;
}

#if !defined(KLAYOUT_M2_UNION_PRODUCTION_SCENE_ONLY)
int run_gate(const std::string &kact_path,
             const std::string &oracle_path,
             std::uint32_t repeat, int device)
{
  gate_require(repeat >= 1, "production repeat must be nonzero");
  const auto all_begin = GateClock::now();
  const auto source_begin = GateClock::now();
  LoadedScene source = load_and_validate(kact_path);
  gate_require(
      hex_digest(source.header.scene_sha256, 32) ==
          kProductionKactSha256,
      "production KACT digest differs from allowlist");
  gate_require(
      source.header.well_layer == 101 &&
          source.header.well_datatype == 0,
      "production KACT logical M2 source is not 101/0");
  LoweredScene hierarchy =
      lower_hierarchy(source, UINT64_C(4000000));
  RawScene raw = build_raw_scene(
      source, hierarchy, device,
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY);
  raw.bind();
  const double raw_build_ms =
      gate_ms(source_begin, GateClock::now());

  const auto oracle_begin = GateClock::now();
  oracle::LoadOptions options;
  options.expected_file_sha256 = kOracleFileSha256;
  options.expected_scene_sha256 = kOracleSceneSha256;
  options.expected_boundary_sha256 = kBoundarySha256;
  const oracle::BoundaryOracle expected =
      oracle::load_cpu_merged_boundary(oracle_path, options);
  const double oracle_ms =
      gate_ms(oracle_begin, GateClock::now());

  const char *prepare_only =
      std::getenv("KLAYOUT_M2_UNION_BACKEND_PREPARE_ONLY");
  if (prepare_only && *prepare_only &&
      std::strcmp(prepare_only, "0") != 0) {
    std::cout
        << "M2_UNION_PRODUCTION_BACKEND_PREPARE PASS"
        << " contexts=" << raw.contexts.size()
        << " metal_contexts=" << raw.metal_contexts.size()
        << " cells=" << raw.cells.size()
        << " stored_polygons=" << raw.polygons.size()
        << " stored_edges=" << raw.edges.size()
        << " flat_polygons=" << raw.request.flat_polygon_count
        << " flat_edges=" << raw.request.flat_edge_count
        << " bounds=" << raw.request.scene_left << ","
        << raw.request.scene_bottom << ","
        << raw.request.scene_right << ","
        << raw.request.scene_top
        << " raw_scene_sha256="
        << raw_hex(raw.request.scene_digest, 32)
        << " oracle_segments=" << expected.segments.size()
        << " oracle_fnv64=" << expected.boundary_fnv64
        << " raw_build_ms=" << std::fixed << std::setprecision(3)
        << raw_build_ms
        << " oracle_ms=" << oracle_ms
        << " verification_total_ms="
        << gate_ms(all_begin, GateClock::now()) << "\n";
    return 0;
  }

  std::vector<double> warm_total_ms;
  for (std::uint32_t run = 0; run < repeat; ++run) {
    Result result{};
    const auto call_begin = GateClock::now();
    const int status =
        klayout_cuda_spatial_run_m2_union_boundary_v1(
            &raw.request, &result);
    const double call_ms =
        gate_ms(call_begin, GateClock::now());
    try {
      gate_require(
          status == KLAYOUT_CUDA_SPATIAL_OK &&
              result.status == KLAYOUT_CUDA_SPATIAL_OK &&
              result.disposition ==
                  KLAYOUT_CUDA_SPATIAL_M2_UNION_COMPLETE,
          std::string("production backend declined: ") +
              result.message);
      validate_counters(result);
      compare_oracle(result, expected);
      gate_require(
          result.context_count == raw.contexts.size() &&
              result.metal_context_count ==
                  raw.metal_contexts.size() &&
              result.cell_count == raw.cells.size() &&
              result.polygon_count == raw.polygons.size() &&
              result.edge_count == raw.edges.size() &&
              result.flat_polygon_count == kFlatPolygons &&
              result.flat_edge_count == kFlatEdges &&
              std::equal(
                  result.scene_digest,
                  result.scene_digest + 32,
                  raw.request.scene_digest),
          "production backend proof echo differs from raw scene");
      gate_require(
          result.certified_empty_mask ==
                  KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY &&
              result.certificate_reserved == 0 &&
              result.suffix_total_ns > 0 &&
              result.suffix_total_ns <= result.total_ns,
          "production backend did not publish the complete M2.5-.9 "
          "empty certificate");
      std::cout
          << "M2_UNION_PRODUCTION_BACKEND run=" << run
          << " setup_ms=" << std::fixed << std::setprecision(3)
          << ns_ms(result.setup_ns)
          << " h2d_ms=" << ns_ms(result.h2d_ns)
          << " rectangle_expand_ms="
          << ns_ms(result.rectangle_expand_ns)
          << " x_membership_ms="
          << ns_ms(result.x_membership_ns)
          << " strip_scan_ms=" << ns_ms(result.strip_scan_ns)
          << " boundary_ms=" << ns_ms(result.boundary_ns)
          << " d2h_ms=" << ns_ms(result.d2h_ns)
          << " suffix_mask=" << result.certified_empty_mask
          << " suffix_ms=" << ns_ms(result.suffix_total_ns)
          << " backend_total_ms=" << ns_ms(result.total_ns)
          << " observed_call_ms=" << call_ms << "\n";
      if (run) warm_total_ms.push_back(call_ms);
    } catch (...) {
      klayout_cuda_spatial_release_m2_union_boundary_v1(&result);
      throw;
    }
    klayout_cuda_spatial_release_m2_union_boundary_v1(&result);
    gate_require(
        !result.segments && !result.segment_count,
        "production result release failed");
  }

  const std::string raw_sha =
      raw_hex(raw.request.scene_digest, 32);
  std::cout
      << "M2_UNION_PRODUCTION_BACKEND_GATE PASS"
      << " contexts=" << raw.contexts.size()
      << " metal_contexts=" << raw.metal_contexts.size()
      << " cells=" << raw.cells.size()
      << " stored_polygons=" << raw.polygons.size()
      << " stored_edges=" << raw.edges.size()
      << " flat_polygons=" << kFlatPolygons
      << " flat_edges=" << kFlatEdges
      << " bounds=" << raw.request.scene_left << ","
      << raw.request.scene_bottom << ","
      << raw.request.scene_right << ","
      << raw.request.scene_top
      << " rectangles=" << kRectangles
      << " memberships=" << kMemberships
      << " events=" << kEvents
      << " x_slabs=" << kXSlabs
      << " strips=" << kStrips
      << " raw_segments=" << kRawSegments
      << " segments=" << kSegments
      << " boundary_fnv64=" << kBoundaryFnv64
      << " suffix_mask="
      << KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY
      << " raw_scene_sha256=" << raw_sha
      << " raw_build_ms=" << std::fixed << std::setprecision(3)
      << raw_build_ms
      << " oracle_ms=" << oracle_ms
      << " warm_call_median_ms="
      << (warm_total_ms.empty() ? 0.0 : median(warm_total_ms))
      << " verification_total_ms="
      << gate_ms(all_begin, GateClock::now()) << "\n";
  return 0;
}
#endif

}  // namespace production_gate

#if !defined(KLAYOUT_M2_UNION_PRODUCTION_SCENE_ONLY)
int main(int argc, char **argv)
{
  try {
    if (argc < 3 || argc > 5) {
      std::cerr
          << "usage: m2_union_production_backend_gate "
             "KACT ORACLE [REPEAT] [DEVICE]\n";
      return 2;
    }
    const unsigned long repeat =
        argc >= 4 ? std::stoul(argv[3]) : 3;
    const long device = argc >= 5 ? std::stol(argv[4]) : 0;
    if (!repeat ||
        repeat > std::numeric_limits<std::uint32_t>::max() ||
        device < 0 ||
        device > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error("invalid repeat or device");
    }
    return production_gate::run_gate(
        argv[1], argv[2], static_cast<std::uint32_t>(repeat),
        static_cast<int>(device));
  } catch (const std::exception &error) {
    std::cerr << "M2_UNION_PRODUCTION_BACKEND_GATE FAIL: "
              << error.what() << "\n";
    return 1;
  }
}
#endif
