/*
 * Full-size source-bound qualification of the fused raw-ACTIVE-union /
 * CONTACT.4 CUDA entry point.
 *
 * The KACTSCN1 reader below is the same independently checked reader used by
 * the other production scene replays.  This bridge retains the shared cell
 * graph and occurrence contexts, serializes only the two small sets of local
 * polygon templates, and never constructs a host-flat ACTIVE or CONTACT
 * stream.
 */

#define main klayout_cuda_embedded_active3_scene_main
#include "active3_scene_island.cu"
#undef main

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace contact4_kact_fused {

using AbiContext = klayout_cuda_spatial_m1_width_space_context_v1;
using AbiCell = klayout_cuda_spatial_m1_width_space_cell_v1;
using AbiPolygon = klayout_cuda_spatial_m1_width_space_polygon_v1;
using AbiEdge = klayout_cuda_spatial_m1_width_space_edge_v1;
using Scene = klayout_cuda_spatial_contact4_active_union_scene_v1;
using Request = klayout_cuda_spatial_contact4_active_union_request_v1;
using Result = klayout_cuda_spatial_contact4_active_union_result_v1;
using Clock = std::chrono::steady_clock;

constexpr char kKactFileSha256[] =
    "4c1609873c64c36b5ecf97796cc43cb4f164c4e872d199514ea07893449772bd";
constexpr char kKactSceneSha256[] =
    "88cf7064183aba158e80486463b21c2481c3861490c0d0d4215311b2491be23e";

constexpr std::uint64_t kExpectedCells = UINT64_C(273);
constexpr std::uint64_t kExpectedInstances = UINT64_C(569514);
constexpr std::uint64_t kExpectedLocalPolygons = UINT64_C(1248);
constexpr std::uint64_t kExpectedLocalEdges = UINT64_C(5040);

// Exact counters established by two byte-identical, independently validated
// captures and two repeated fused executions of the pinned scene.
constexpr std::uint64_t kExpectedContexts = UINT64_C(848485);
constexpr std::uint64_t kExpectedActiveLayerContexts = UINT64_C(788174);
constexpr std::uint64_t kExpectedContactLayerContexts = UINT64_C(777474);
constexpr std::uint64_t kExpectedActiveLocalPolygons = UINT64_C(716);
constexpr std::uint64_t kExpectedActiveLocalEdges = UINT64_C(2912);
constexpr std::uint64_t kExpectedContactLocalPolygons = UINT64_C(532);
constexpr std::uint64_t kExpectedContactLocalEdges = UINT64_C(2128);
constexpr std::uint64_t kExpectedActiveFlatPolygons = UINT64_C(24687816);
constexpr std::uint64_t kExpectedActiveFlatEdges = UINT64_C(98754896);
constexpr std::uint64_t kExpectedContactFlatPolygons = UINT64_C(10353606);
constexpr std::uint64_t kExpectedContactFlatEdges = UINT64_C(41414424);
constexpr char kExpectedActiveDigest[] =
    "d079f29b7d38fdc50f566a78fe680463319e5bc14ee9430efe743793b897eff0";
constexpr char kExpectedContactDigest[] =
    "adc4614073dcabdb88c608c6e777eb281fee21babba623690bb911ade38489af";
constexpr std::uint64_t kExpectedRectangles = UINT64_C(24689164);
constexpr std::uint64_t kExpectedXSlabs = UINT64_C(40259);
constexpr std::uint64_t kExpectedUnionMemberships = UINT64_C(114973212);
constexpr std::uint64_t kExpectedStripIntervals = UINT64_C(40936780);
constexpr std::uint64_t kExpectedRawSegments = UINT64_C(87241928);
constexpr std::uint64_t kExpectedBoundarySegments = UINT64_C(10736736);
constexpr std::uint64_t kExpectedGridCells = UINT64_C(660231);
constexpr std::uint64_t kExpectedContactMemberships = UINT64_C(44084722);
constexpr std::uint64_t kExpectedBoundaryCellVisits = UINT64_C(13416293);
constexpr std::uint64_t kExpectedMemberVisits = UINT64_C(1099731476);
constexpr std::uint64_t kExpectedCandidatePairs = UINT64_C(1091588254);

struct CanonicalDigest
{
  db::cuda_active3_digest::Sha256 sha;

  void bytes(const void *data, std::size_t count)
  {
    sha.update(data, count);
  }

  void u32(std::uint32_t value)
  {
    std::uint8_t encoded[4];
    for (unsigned int index = 0; index < 4; ++index) {
      encoded[index] =
          static_cast<std::uint8_t>(value >> (index * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void u64(std::uint64_t value)
  {
    std::uint8_t encoded[8];
    for (unsigned int index = 0; index < 8; ++index) {
      encoded[index] =
          static_cast<std::uint8_t>(value >> (index * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void i64(std::int64_t value)
  {
    u64(static_cast<std::uint64_t>(value));
  }

  std::array<std::uint8_t, 32> finish()
  {
    return sha.finish();
  }
};

struct CompactLayerScene
{
  std::vector<AbiContext> contexts;
  std::vector<std::uint32_t> layer_contexts;
  std::vector<std::uint64_t> polygon_offsets;
  std::vector<std::uint64_t> edge_offsets;
  std::vector<AbiCell> cells;
  std::vector<AbiPolygon> polygons;
  std::vector<AbiEdge> edges;
  Scene descriptor{};

  void bind()
  {
    descriptor.contexts = contexts.data();
    descriptor.context_count = contexts.size();
    descriptor.layer_contexts = layer_contexts.data();
    descriptor.layer_context_count = layer_contexts.size();
    descriptor.context_polygon_offsets = polygon_offsets.data();
    descriptor.context_polygon_offset_count = polygon_offsets.size();
    descriptor.context_edge_offsets = edge_offsets.data();
    descriptor.context_edge_offset_count = edge_offsets.size();
    descriptor.cells = cells.data();
    descriptor.cell_count = cells.size();
    descriptor.polygons = polygons.data();
    descriptor.polygon_count = polygons.size();
    descriptor.edges = edges.data();
    descriptor.edge_count = edges.size();
  }
};

struct ProofCounters
{
  std::uint64_t rectangle_count;
  std::uint64_t x_slab_count;
  std::uint64_t union_membership_count;
  std::uint64_t event_count;
  std::uint64_t strip_interval_count;
  std::uint64_t raw_segment_count;
  std::uint64_t boundary_segment_count;
  std::uint64_t contact_expanded_edge_count;
  std::uint64_t grid_cell_count;
  std::uint64_t contact_membership_count;
  std::uint64_t boundary_cell_visit_count;
  std::uint64_t member_visit_count;
  std::uint64_t candidate_pair_count;
  std::uint64_t raw_hit_count;
  std::uint64_t uncertain_count;

  bool operator==(const ProofCounters &other) const
  {
    return rectangle_count == other.rectangle_count &&
        x_slab_count == other.x_slab_count &&
        union_membership_count == other.union_membership_count &&
        event_count == other.event_count &&
        strip_interval_count == other.strip_interval_count &&
        raw_segment_count == other.raw_segment_count &&
        boundary_segment_count == other.boundary_segment_count &&
        contact_expanded_edge_count ==
            other.contact_expanded_edge_count &&
        grid_cell_count == other.grid_cell_count &&
        contact_membership_count == other.contact_membership_count &&
        boundary_cell_visit_count ==
            other.boundary_cell_visit_count &&
        member_visit_count == other.member_visit_count &&
        candidate_pair_count == other.candidate_pair_count &&
        raw_hit_count == other.raw_hit_count &&
        uncertain_count == other.uncertain_count;
  }
};

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double ns_ms(std::uint64_t nanoseconds)
{
  return static_cast<double>(nanoseconds) / 1000000.0;
}

std::string digest_hex(const std::uint8_t *bytes, std::size_t count)
{
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < count; ++index) {
    output << std::setw(2)
           << static_cast<unsigned int>(bytes[index]);
  }
  return output.str();
}

std::string file_sha256(const LoadedScene &source)
{
  db::cuda_active3_digest::Sha256 sha;
  sha.update(source.storage.data(), source.storage.size());
  const auto digest = sha.finish();
  return digest_hex(digest.data(), digest.size());
}

std::pair<std::int64_t, std::int64_t> transform_point(
    const AbiContext &context, std::int64_t x, std::int64_t y)
{
  __int128 output_x = 0;
  __int128 output_y = 0;
  switch (context.transform_code) {
  case 0: output_x = x; output_y = y; break;
  case 1: output_x = -static_cast<__int128>(y); output_y = x; break;
  case 2:
    output_x = -static_cast<__int128>(x);
    output_y = -static_cast<__int128>(y);
    break;
  case 3: output_x = y; output_y = -static_cast<__int128>(x); break;
  case 4: output_x = x; output_y = -static_cast<__int128>(y); break;
  case 5: output_x = y; output_y = x; break;
  case 6: output_x = -static_cast<__int128>(x); output_y = y; break;
  case 7:
    output_x = -static_cast<__int128>(y);
    output_y = -static_cast<__int128>(x);
    break;
  default:
    throw std::runtime_error("invalid KACT transform code");
  }
  output_x += context.tx;
  output_y += context.ty;
  require(
      output_x >= std::numeric_limits<std::int64_t>::min() &&
          output_x <= std::numeric_limits<std::int64_t>::max() &&
          output_y >= std::numeric_limits<std::int64_t>::min() &&
          output_y <= std::numeric_limits<std::int64_t>::max(),
      "scene transform overflows int64");
  return {
      static_cast<std::int64_t>(output_x),
      static_cast<std::int64_t>(output_y)};
}

void add_bounds(const AbiContext &context, const AbiPolygon &polygon,
                bool *have_bounds, Scene *scene)
{
  const std::int64_t xs[2] = {polygon.left, polygon.right};
  const std::int64_t ys[2] = {polygon.bottom, polygon.top};
  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      const auto point =
          transform_point(context, xs[xi], ys[yi]);
      if (!*have_bounds) {
        scene->scene_left = scene->scene_right = point.first;
        scene->scene_bottom = scene->scene_top = point.second;
        *have_bounds = true;
      } else {
        scene->scene_left = std::min(scene->scene_left, point.first);
        scene->scene_bottom =
            std::min(scene->scene_bottom, point.second);
        scene->scene_right =
            std::max(scene->scene_right, point.first);
        scene->scene_top = std::max(scene->scene_top, point.second);
      }
    }
  }
}

std::array<std::uint8_t, 32> scene_digest(
    const CompactLayerScene &storage, const char magic[8])
{
  const Scene &scene = storage.descriptor;
  CanonicalDigest digest;
  digest.bytes(magic, 8);
  digest.u32(scene.format_version);
  digest.u32(scene.dbu_per_micron);
  digest.u32(scene.root_cell);
  digest.u32(scene.reserved0);
  digest.u64(storage.contexts.size());
  digest.u64(storage.layer_contexts.size());
  digest.u64(storage.cells.size());
  digest.u64(storage.polygons.size());
  digest.u64(storage.edges.size());
  digest.u64(scene.flat_polygon_count);
  digest.u64(scene.flat_edge_count);
  digest.i64(scene.scene_left);
  digest.i64(scene.scene_bottom);
  digest.i64(scene.scene_right);
  digest.i64(scene.scene_top);
  for (const AbiContext &context : storage.contexts) {
    digest.i64(context.tx);
    digest.i64(context.ty);
    digest.u32(context.cell_id);
    digest.u32(context.transform_code);
  }
  for (std::size_t index = 0;
       index < storage.layer_contexts.size(); ++index) {
    digest.u32(storage.layer_contexts[index]);
    digest.u64(storage.polygon_offsets[index]);
    digest.u64(storage.edge_offsets[index]);
  }
  for (const AbiCell &cell : storage.cells) {
    digest.u64(cell.source_cell_index);
    digest.u64(cell.polygon_begin);
    digest.u64(cell.edge_begin);
    digest.u32(cell.polygon_count);
    digest.u32(cell.edge_count);
  }
  for (const AbiPolygon &polygon : storage.polygons) {
    digest.u64(polygon.edge_begin);
    digest.i64(polygon.left);
    digest.i64(polygon.bottom);
    digest.i64(polygon.right);
    digest.i64(polygon.top);
    digest.u32(polygon.polygon_id);
    digest.u32(polygon.edge_count);
  }
  for (const AbiEdge &edge : storage.edges) {
    digest.i64(edge.x1);
    digest.i64(edge.y1);
    digest.i64(edge.x2);
    digest.i64(edge.y2);
  }
  return digest.finish();
}

void rebuild_scene_index_and_digest(
    CompactLayerScene *output, const char digest_magic[8])
{
  Scene &scene = output->descriptor;
  output->layer_contexts.clear();
  output->polygon_offsets.clear();
  output->edge_offsets.clear();
  scene.flat_polygon_count = 0;
  scene.flat_edge_count = 0;
  scene.scene_left = 0;
  scene.scene_bottom = 0;
  scene.scene_right = 0;
  scene.scene_top = 0;
  bool have_bounds = false;
  for (std::uint32_t context_id = 0;
       context_id < output->contexts.size(); ++context_id) {
    const AbiContext &context = output->contexts[context_id];
    require(
        context.cell_id < output->cells.size(),
        "scene context references an absent cell");
    const AbiCell &cell = output->cells[context.cell_id];
    if (!cell.polygon_count) continue;
    output->layer_contexts.push_back(context_id);
    output->polygon_offsets.push_back(scene.flat_polygon_count);
    output->edge_offsets.push_back(scene.flat_edge_count);
    require(
        scene.flat_polygon_count <=
                std::numeric_limits<std::uint64_t>::max() -
                    cell.polygon_count &&
            scene.flat_edge_count <=
                std::numeric_limits<std::uint64_t>::max() -
                    cell.edge_count,
        "flat layer census overflows uint64");
    scene.flat_polygon_count += cell.polygon_count;
    scene.flat_edge_count += cell.edge_count;
    const std::uint64_t polygon_end =
        cell.polygon_begin + cell.polygon_count;
    require(
        polygon_end <= output->polygons.size(),
        "cell polygon range is outside compact storage");
    for (std::uint64_t polygon_id = cell.polygon_begin;
         polygon_id < polygon_end; ++polygon_id) {
      add_bounds(
          context,
          output->polygons[static_cast<std::size_t>(polygon_id)],
          &have_bounds, &scene);
    }
  }
  require(have_bounds, "selected KACT layer is empty");
  output->bind();
  const auto digest = scene_digest(*output, digest_magic);
  std::copy(digest.begin(), digest.end(), scene.scene_digest);
}

CompactLayerScene build_layer_scene(
    const LoadedScene &source, const LoweredScene &hierarchy,
    std::uint32_t logical_layer, std::uint32_t role,
    std::uint32_t physical_layer, const char digest_magic[8])
{
  CompactLayerScene output;
  require(
      source.header.cell_count <=
              std::numeric_limits<std::uint32_t>::max() &&
          source.header.root_cell <=
              std::numeric_limits<std::uint32_t>::max(),
      "KACT cell IDs exceed raw scene ABI");
  output.contexts.reserve(hierarchy.contexts.size());
  for (const ContextGpu &context : hierarchy.contexts) {
    output.contexts.push_back(
        {context.tx, context.ty, context.cell, context.transform});
  }

  output.cells.resize(
      static_cast<std::size_t>(source.header.cell_count));
  for (std::uint64_t cell_id = 0;
       cell_id < source.header.cell_count; ++cell_id) {
    const CellRecord &source_cell = source.cells[cell_id];
    AbiCell &cell = output.cells[static_cast<std::size_t>(cell_id)];
    cell.source_cell_index = source_cell.cell_id;
    cell.polygon_begin = output.polygons.size();
    cell.edge_begin = output.edges.size();
    std::uint32_t local_polygon_id = 0;
    for (std::uint64_t local = 0;
         local < source_cell.polygon_count; ++local) {
      const PolygonRecord &source_polygon =
          source.polygons[source_cell.polygon_begin + local];
      if (source_polygon.layer_code != logical_layer) continue;
      require(
          source_polygon.edge_count >= 4 &&
              source_polygon.edge_count <=
                  std::numeric_limits<std::uint32_t>::max(),
          "KACT polygon edge count exceeds raw scene ABI");
      AbiPolygon polygon{};
      polygon.edge_begin = output.edges.size();
      polygon.left = source_polygon.bbox[0];
      polygon.bottom = source_polygon.bbox[1];
      polygon.right = source_polygon.bbox[2];
      polygon.top = source_polygon.bbox[3];
      polygon.polygon_id = local_polygon_id++;
      polygon.edge_count = source_polygon.edge_count;
      output.polygons.push_back(polygon);
      for (std::uint32_t edge_local = 0;
           edge_local < source_polygon.edge_count; ++edge_local) {
        const EdgeRecord &source_edge =
            source.edges[source_polygon.edge_begin + edge_local];
        require(
            source_edge.layer_code == logical_layer &&
                source_edge.polygon_id == source_polygon.polygon_id,
            "KACT polygon/edge ownership mismatch");
        output.edges.push_back(
            {source_edge.x1, source_edge.y1,
             source_edge.x2, source_edge.y2});
      }
    }
    const std::uint64_t polygon_count =
        output.polygons.size() - cell.polygon_begin;
    const std::uint64_t edge_count =
        output.edges.size() - cell.edge_begin;
    require(
        polygon_count <= std::numeric_limits<std::uint32_t>::max() &&
            edge_count <= std::numeric_limits<std::uint32_t>::max(),
        "per-cell layer census exceeds raw scene ABI");
    cell.polygon_count = static_cast<std::uint32_t>(polygon_count);
    cell.edge_count = static_cast<std::uint32_t>(edge_count);
  }

  Scene &scene = output.descriptor;
  scene.struct_size = sizeof(scene);
  scene.role = role;
  scene.format_version = 1;
  scene.dbu_per_micron = 2000;
  scene.root_cell =
      static_cast<std::uint32_t>(source.header.root_cell);
  scene.layer = physical_layer;
  scene.datatype = 0;
  scene.context_record_bytes = sizeof(AbiContext);
  scene.cell_record_bytes = sizeof(AbiCell);
  scene.polygon_record_bytes = sizeof(AbiPolygon);
  scene.edge_record_bytes = sizeof(AbiEdge);
  std::copy(
      digest_magic, digest_magic + 8, scene.digest_domain);

  rebuild_scene_index_and_digest(&output, digest_magic);
  return output;
}

struct WorldBox
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
};

WorldBox transformed_box(
    const AbiContext &context, const AbiPolygon &polygon)
{
  WorldBox box{
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min()};
  const std::int64_t xs[2] = {polygon.left, polygon.right};
  const std::int64_t ys[2] = {polygon.bottom, polygon.top};
  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      const auto point =
          transform_point(context, xs[xi], ys[yi]);
      box.left = std::min(box.left, point.first);
      box.bottom = std::min(box.bottom, point.second);
      box.right = std::max(box.right, point.first);
      box.top = std::max(box.top, point.second);
    }
  }
  return box;
}

CompactLayerScene build_hit_mutation(
    const CompactLayerScene &clean_contact,
    const CompactLayerScene &active, WorldBox *injected_box)
{
  WorldBox support{};
  bool found = false;
  for (std::uint32_t context_id : active.layer_contexts) {
    const AbiContext &context = active.contexts[context_id];
    const AbiCell &cell = active.cells[context.cell_id];
    const std::uint64_t end =
        cell.polygon_begin + cell.polygon_count;
    for (std::uint64_t polygon_id = cell.polygon_begin;
         polygon_id < end; ++polygon_id) {
      const AbiPolygon &polygon =
          active.polygons[static_cast<std::size_t>(polygon_id)];
      if (polygon.edge_count != 4) continue;
      const WorldBox candidate =
          transformed_box(context, polygon);
      if (candidate.right == active.descriptor.scene_right &&
          candidate.right - candidate.left >= 25 &&
          candidate.top - candidate.bottom >= 30) {
        support = candidate;
        found = true;
        break;
      }
    }
    if (found) break;
  }
  require(
      found,
      "could not locate an exposed rectangular ACTIVE support for hit oracle");

  CompactLayerScene output = clean_contact;
  const std::uint32_t root = output.descriptor.root_cell;
  require(
      root < output.cells.size() &&
          !output.contexts.empty() &&
          output.contexts[0].cell_id == root &&
          output.contexts[0].transform_code == 0 &&
          output.contexts[0].tx == 0 &&
          output.contexts[0].ty == 0,
      "KACT root is not the canonical identity context");
  AbiCell &root_cell = output.cells[root];
  const std::uint64_t polygon_insert =
      root_cell.polygon_begin + root_cell.polygon_count;
  const std::uint64_t edge_insert =
      root_cell.edge_begin + root_cell.edge_count;
  require(
      polygon_insert <= output.polygons.size() &&
          edge_insert <= output.edges.size() &&
          root_cell.polygon_count <
              std::numeric_limits<std::uint32_t>::max() &&
          root_cell.edge_count <=
              std::numeric_limits<std::uint32_t>::max() - 4,
      "root CONTACT insertion exceeds compact storage");

  const WorldBox contact{
      support.right - 15,
      support.bottom + 10,
      support.right - 5,
      support.bottom + 20};
  require(
      contact.left > support.left &&
          contact.right < support.right &&
          contact.bottom > support.bottom &&
          contact.top < support.top,
      "derived CONTACT hit oracle is not strictly inside ACTIVE");
  const AbiEdge edges[4] = {
      {contact.left, contact.bottom, contact.left, contact.top},
      {contact.left, contact.top, contact.right, contact.top},
      {contact.right, contact.top, contact.right, contact.bottom},
      {contact.right, contact.bottom, contact.left, contact.bottom}};
  output.edges.insert(
      output.edges.begin() + static_cast<std::ptrdiff_t>(edge_insert),
      std::begin(edges), std::end(edges));
  for (AbiPolygon &polygon : output.polygons) {
    if (polygon.edge_begin >= edge_insert) {
      polygon.edge_begin += 4;
    }
  }
  AbiPolygon polygon{};
  polygon.edge_begin = edge_insert;
  polygon.left = contact.left;
  polygon.bottom = contact.bottom;
  polygon.right = contact.right;
  polygon.top = contact.top;
  polygon.polygon_id = root_cell.polygon_count;
  polygon.edge_count = 4;
  output.polygons.insert(
      output.polygons.begin() +
          static_cast<std::ptrdiff_t>(polygon_insert),
      polygon);
  for (std::uint32_t cell_id = 0;
       cell_id < output.cells.size(); ++cell_id) {
    if (cell_id == root) continue;
    AbiCell &cell = output.cells[cell_id];
    if (cell.polygon_begin >= polygon_insert) {
      ++cell.polygon_begin;
    }
    if (cell.edge_begin >= edge_insert) {
      cell.edge_begin += 4;
    }
  }
  ++root_cell.polygon_count;
  root_cell.edge_count += 4;
  rebuild_scene_index_and_digest(
      &output,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_DIGEST_DOMAIN);
  *injected_box = contact;
  return output;
}

void require_expected(std::uint64_t actual, std::uint64_t expected,
                      const char *label)
{
  if (expected && actual != expected) {
    throw std::runtime_error(
        std::string(label) + " differs from pinned exact census: " +
        std::to_string(actual) + " != " + std::to_string(expected));
  }
}

void validate_scene_echo(const Scene &scene,
                         const klayout_cuda_spatial_contact4_active_union_scene_echo_v1
                             &echo)
{
  require(
      echo.struct_size == sizeof(echo) &&
          echo.role == scene.role &&
          echo.format_version == scene.format_version &&
          echo.dbu_per_micron == scene.dbu_per_micron &&
          echo.root_cell == scene.root_cell &&
          echo.layer == scene.layer &&
          echo.datatype == scene.datatype &&
          echo.context_count == scene.context_count &&
          echo.layer_context_count == scene.layer_context_count &&
          echo.context_polygon_offset_count ==
              scene.context_polygon_offset_count &&
          echo.context_edge_offset_count ==
              scene.context_edge_offset_count &&
          echo.cell_count == scene.cell_count &&
          echo.polygon_count == scene.polygon_count &&
          echo.edge_count == scene.edge_count &&
          echo.flat_polygon_count == scene.flat_polygon_count &&
          echo.flat_edge_count == scene.flat_edge_count &&
          echo.scene_left == scene.scene_left &&
          echo.scene_bottom == scene.scene_bottom &&
          echo.scene_right == scene.scene_right &&
          echo.scene_top == scene.scene_top &&
          std::equal(
              echo.digest_domain, echo.digest_domain + 8,
              scene.digest_domain) &&
          std::equal(
              echo.scene_digest, echo.scene_digest + 32,
              scene.scene_digest),
      "fused backend scene echo differs from compact request");
}

ProofCounters counters(const Result &result)
{
  return {
      result.rectangle_count,
      result.x_slab_count,
      result.union_membership_count,
      result.event_count,
      result.strip_interval_count,
      result.raw_segment_count,
      result.boundary_segment_count,
      result.contact_expanded_edge_count,
      result.grid_cell_count,
      result.contact_membership_count,
      result.boundary_cell_visit_count,
      result.member_visit_count,
      result.candidate_pair_count,
      result.raw_hit_count,
      result.uncertain_count};
}

void validate_complete(
    const Request &request, int backend_status, const Result &result)
{
  if (backend_status != KLAYOUT_CUDA_SPATIAL_OK ||
      result.status != KLAYOUT_CUDA_SPATIAL_OK ||
      result.disposition !=
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE) {
    std::cerr
        << "CONTACT4_ACTIVE_UNION_KACT_FUSED decline"
        << " status=" << backend_status << "/" << result.status
        << " fallback=" << result.fallback_flags
        << " disposition=" << result.disposition
        << " rectangles=" << result.rectangle_count
        << " memberships=" << result.union_membership_count
        << " raw_segments=" << result.raw_segment_count
        << " boundary_segments=" << result.boundary_segment_count
        << " contact_edges=" << result.contact_expanded_edge_count
        << " grid_cells=" << result.grid_cell_count
        << " contact_memberships=" << result.contact_membership_count
        << " boundary_cell_visits="
        << result.boundary_cell_visit_count
        << " member_visits=" << result.member_visit_count
        << " candidates=" << result.candidate_pair_count
        << " hits=" << result.raw_hit_count
        << " uncertain=" << result.uncertain_count
        << " message=" << result.message << "\n";
  }
  require(
      backend_status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.fallback_flags ==
              KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE &&
          result.disposition ==
              KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE &&
          result.device_flags == 0 &&
          result.raw_hit_count == 0 &&
          result.uncertain_count == 0,
      std::string("full-size fused backend declined: ") + result.message);
  require(
      result.abi_version == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
          result.struct_size == sizeof(result) &&
          result.opcode == request.opcode &&
          result.option_flags == request.option_flags &&
          result.format_version == request.format_version &&
          result.dbu_per_micron == request.dbu_per_micron &&
          result.device == request.device &&
          result.distance == request.distance &&
          result.grid_cell_size == request.grid_cell_size,
      "fused backend scalar proof echo differs from request");
  validate_scene_echo(request.active, result.active);
  validate_scene_echo(request.contact, result.contact);

  require(
      result.rectangle_count >= request.active.flat_polygon_count &&
          result.rectangle_count <= request.max_rectangles &&
          result.x_slab_count &&
          result.x_slab_count <= request.max_x_slabs &&
          result.union_membership_count &&
          result.union_membership_count <=
              request.max_union_memberships &&
          result.event_count == result.union_membership_count * 2 &&
          result.event_count <= request.max_events &&
          result.strip_interval_count &&
          result.strip_interval_count <=
              result.union_membership_count &&
          result.raw_segment_count >= result.boundary_segment_count &&
          result.raw_segment_count <= request.max_raw_segments &&
          result.boundary_segment_count &&
          result.boundary_segment_count <=
              request.max_boundary_segments &&
          result.contact_expanded_edge_count ==
              request.contact.flat_edge_count &&
          result.contact_expanded_edge_count <=
              request.max_contact_edges &&
          result.grid_cell_count &&
          result.grid_cell_count <= request.max_grid_cells &&
          result.contact_membership_count >=
              result.contact_expanded_edge_count &&
          result.contact_membership_count <=
              request.max_contact_memberships &&
          result.boundary_cell_visit_count <=
              request.max_boundary_cell_visits &&
          result.member_visit_count <= request.max_member_visits &&
          result.candidate_pair_count <= result.member_visit_count &&
          result.candidate_pair_count <= request.max_pair_work,
      "fused backend proof counters violate exact conservation/caps");
  require(
      result.device_total_bytes &&
          result.union_free_begin_bytes <= result.device_total_bytes &&
          result.union_free_low_bytes <=
              result.union_free_begin_bytes &&
          result.callback_free_begin_bytes <=
              result.device_total_bytes &&
          result.callback_free_low_bytes <=
              result.callback_free_begin_bytes &&
          result.post_scan_free_bytes <= result.device_total_bytes &&
          result.callback_incremental_peak_bytes ==
              result.callback_free_begin_bytes -
                  result.callback_free_low_bytes &&
          result.callback_free_begin_bytes >
              result.union_free_low_bytes,
      "fused backend memory phasing did not prove union high-water release");

  require_expected(
      result.rectangle_count, kExpectedRectangles, "rectangle count");
  require_expected(result.x_slab_count, kExpectedXSlabs, "x-slab count");
  require_expected(
      result.union_membership_count, kExpectedUnionMemberships,
      "union membership count");
  require_expected(
      result.strip_interval_count, kExpectedStripIntervals,
      "strip interval count");
  require_expected(
      result.raw_segment_count, kExpectedRawSegments,
      "raw segment count");
  require_expected(
      result.boundary_segment_count, kExpectedBoundarySegments,
      "boundary segment count");
  require_expected(
      result.grid_cell_count, kExpectedGridCells, "grid-cell count");
  require_expected(
      result.contact_membership_count, kExpectedContactMemberships,
      "contact membership count");
  require_expected(
      result.boundary_cell_visit_count, kExpectedBoundaryCellVisits,
      "boundary cell visit count");
  require_expected(
      result.member_visit_count, kExpectedMemberVisits,
      "member visit count");
  require_expected(
      result.candidate_pair_count, kExpectedCandidatePairs,
      "candidate-pair count");
}

void validate_raw_hit(
    const Request &request, int backend_status, const Result &result)
{
  require(
      backend_status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.fallback_flags ==
              KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE &&
          result.disposition ==
              KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_RAW_HITS &&
          result.device_flags == 0 &&
          result.raw_hit_count &&
          result.uncertain_count == 0,
      std::string("derived hit oracle did not return exact RAW_HITS: ") +
          result.message);
  require(
      result.opcode == request.opcode &&
          result.option_flags == request.option_flags &&
          result.distance == request.distance &&
          result.grid_cell_size == request.grid_cell_size &&
          result.contact_expanded_edge_count ==
              request.contact.flat_edge_count &&
          result.boundary_segment_count ==
              kExpectedBoundarySegments &&
          result.candidate_pair_count <= result.member_visit_count &&
          result.raw_hit_count <= result.candidate_pair_count,
      "derived hit oracle returned inconsistent exact counters");
  validate_scene_echo(request.active, result.active);
  validate_scene_echo(request.contact, result.contact);
}

Request build_request(
    CompactLayerScene *active, CompactLayerScene *contact, int device)
{
  active->bind();
  contact->bind();
  Request request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.device = device;
  request.distance = 10;
  request.grid_cell_size = 2000;
  request.active = active->descriptor;
  request.contact = contact->descriptor;

  request.max_contexts = UINT64_C(4000000);
  request.max_rectangles = UINT64_C(32000000);
  request.max_x_slabs = UINT64_C(32000000);
  request.max_union_memberships = UINT64_C(128000000);
  request.max_events = UINT64_C(256000000);
  request.max_raw_segments = UINT64_C(128000000);
  request.max_boundary_segments = UINT64_C(64000000);
  // The x2 hierarchy contains long top-level ACTIVE rails spanning many
  // local x coordinates; this is a work cap, not a semantic restriction.
  request.max_slabs_per_rectangle = 4096;
  request.max_contact_edges = UINT64_C(64000000);
  request.max_grid_cells = UINT64_C(16000000);
  request.max_contact_memberships = UINT64_C(200000000);
  request.max_boundary_cell_visits = UINT64_C(200000000);
  request.max_member_visits = UINT64_C(1200000000);
  request.max_pair_work = UINT64_C(1200000000);
  request.max_cells_per_contact_edge = 4096;
  request.max_cells_per_boundary_edge = 4096;
  return request;
}

int run(const std::string &kact_path, std::uint32_t repeat, int device)
{
  require(repeat >= 2, "full-size qualification requires at least two runs");
  const auto all_begin = Clock::now();
  const auto load_begin = Clock::now();
  LoadedScene source = load_and_validate(kact_path);
  require(
      file_sha256(source) == kKactFileSha256,
      "KACT file SHA-256 differs from pinned capture");
  require(
      hex_digest(source.header.scene_sha256, 32) ==
          kKactSceneSha256,
      "KACT embedded scene SHA-256 differs from pinned capture");
  require(
      source.header.well_layer == 1 &&
          source.header.well_datatype == 0 &&
          source.header.active_layer == 10 &&
          source.header.active_datatype == 0 &&
          source.header.cell_count == kExpectedCells &&
          source.header.instance_count == kExpectedInstances &&
          source.header.polygon_count == kExpectedLocalPolygons &&
          source.header.edge_count == kExpectedLocalEdges,
      "KACT physical layers or local census differ from qualification");
  LoweredScene hierarchy =
      lower_hierarchy(source, UINT64_C(4000000));
  require_expected(
      hierarchy.contexts.size(), kExpectedContexts, "context count");

  CompactLayerScene active = build_layer_scene(
      source, hierarchy, kWellLayer,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE, 1,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_DIGEST_DOMAIN);
  CompactLayerScene contact = build_layer_scene(
      source, hierarchy, kActiveLayer,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
      KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_DIGEST_DOMAIN);
  require(
      active.contexts.size() == contact.contexts.size() &&
          active.cells.size() == contact.cells.size(),
      "ACTIVE and CONTACT descriptors do not retain one shared hierarchy");
  for (std::size_t index = 0; index < active.contexts.size(); ++index) {
    const AbiContext &a = active.contexts[index];
    const AbiContext &c = contact.contexts[index];
    require(
        a.tx == c.tx && a.ty == c.ty &&
            a.cell_id == c.cell_id &&
            a.transform_code == c.transform_code,
        "ACTIVE and CONTACT occurrence contexts differ");
  }
  for (std::size_t index = 0; index < active.cells.size(); ++index) {
    require(
        active.cells[index].source_cell_index ==
            contact.cells[index].source_cell_index,
        "ACTIVE and CONTACT source cell identities differ");
  }
  require_expected(
      active.descriptor.flat_polygon_count,
      kExpectedActiveFlatPolygons, "ACTIVE flat polygon count");
  require_expected(
      active.descriptor.flat_edge_count,
      kExpectedActiveFlatEdges, "ACTIVE flat edge count");
  require_expected(
      contact.descriptor.flat_polygon_count,
      kExpectedContactFlatPolygons, "CONTACT flat polygon count");
  require_expected(
      contact.descriptor.flat_edge_count,
      kExpectedContactFlatEdges, "CONTACT flat edge count");
  require(
      active.layer_contexts.size() ==
              kExpectedActiveLayerContexts &&
          contact.layer_contexts.size() ==
              kExpectedContactLayerContexts &&
          active.polygons.size() ==
              kExpectedActiveLocalPolygons &&
          active.edges.size() == kExpectedActiveLocalEdges &&
          contact.polygons.size() ==
              kExpectedContactLocalPolygons &&
          contact.edges.size() == kExpectedContactLocalEdges &&
          digest_hex(active.descriptor.scene_digest, 32) ==
              kExpectedActiveDigest &&
          digest_hex(contact.descriptor.scene_digest, 32) ==
              kExpectedContactDigest,
      "compact local scene census or canonical digest differs from pin");

  Request request =
      build_request(&active, &contact, device);
  const double compact_build_ms =
      elapsed_ms(load_begin, Clock::now());
  const std::uint64_t local_geometry_records =
      active.polygons.size() + active.edges.size() +
      contact.polygons.size() + contact.edges.size();
  require(
      local_geometry_records ==
          source.header.polygon_count + source.header.edge_count,
      "compact descriptors duplicated or dropped KACT local geometry");
  require(
      local_geometry_records <
          active.descriptor.flat_polygon_count +
              active.descriptor.flat_edge_count +
              contact.descriptor.flat_polygon_count +
              contact.descriptor.flat_edge_count,
      "qualification accidentally materialized host-flat geometry");

  ProofCounters reference{};
  bool have_reference = false;
  std::vector<double> observed_ms;
  for (std::uint32_t run_id = 0; run_id < repeat; ++run_id) {
    Result result{};
    const auto call_begin = Clock::now();
    const int status =
        klayout_cuda_spatial_run_contact4_active_union_empty_v1(
            &request, &result);
    const double call_ms =
        elapsed_ms(call_begin, Clock::now());
    validate_complete(request, status, result);
    const ProofCounters proof = counters(result);
    if (have_reference) {
      require(
          proof == reference,
          "full-size fused proof counters changed across exact repeats");
    } else {
      reference = proof;
      have_reference = true;
    }
    observed_ms.push_back(call_ms);
    std::cout
        << "CONTACT4_ACTIVE_UNION_KACT_FUSED run=" << run_id
        << " call_ms=" << std::fixed << std::setprecision(3)
        << call_ms
        << " backend_ms=" << ns_ms(result.total_ns)
        << " setup_ms=" << ns_ms(result.setup_ns)
        << " active_h2d_ms=" << ns_ms(result.active_h2d_ns)
        << " active_expand_ms=" << ns_ms(result.active_expand_ns)
        << " x_membership_ms=" << ns_ms(result.x_membership_ns)
        << " strip_scan_ms=" << ns_ms(result.strip_scan_ns)
        << " boundary_ms=" << ns_ms(result.boundary_ns)
        << " contact_h2d_ms=" << ns_ms(result.contact_h2d_ns)
        << " contact_expand_ms=" << ns_ms(result.contact_expand_ns)
        << " grid_count_ms=" << ns_ms(result.grid_count_ns)
        << " grid_build_ms=" << ns_ms(result.grid_build_ns)
        << " query_ms=" << ns_ms(result.query_ns)
        << " scalar_d2h_ms=" << ns_ms(result.d2h_ns)
        << " geometry_d2h_bytes=0"
        << " union_low_mib="
        << (result.union_free_begin_bytes -
            result.union_free_low_bytes) / 1048576.0
        << " callback_peak_mib="
        << result.callback_incremental_peak_bytes / 1048576.0
        << "\n";
  }

  WorldBox injected{};
  CompactLayerScene hit_contact =
      build_hit_mutation(contact, active, &injected);
  require(
      hit_contact.descriptor.flat_polygon_count ==
              contact.descriptor.flat_polygon_count + 1 &&
          hit_contact.descriptor.flat_edge_count ==
              contact.descriptor.flat_edge_count + 4,
      "root-only hit mutation expanded through more than one context");
  Request hit_request =
      build_request(&active, &hit_contact, device);
  Result hit_result{};
  const auto hit_begin = Clock::now();
  const int hit_status =
      klayout_cuda_spatial_run_contact4_active_union_empty_v1(
          &hit_request, &hit_result);
  const double hit_ms = elapsed_ms(hit_begin, Clock::now());
  validate_raw_hit(hit_request, hit_status, hit_result);
  std::cout
      << "CONTACT4_ACTIVE_UNION_KACT_FUSED_HIT_ORACLE PASS"
      << " injected_box=" << injected.left << ","
      << injected.bottom << "," << injected.right << ","
      << injected.top
      << " contact_digest="
      << digest_hex(hit_contact.descriptor.scene_digest, 32)
      << " hits=" << hit_result.raw_hit_count
      << " candidates=" << hit_result.candidate_pair_count
      << " call_ms=" << std::fixed << std::setprecision(3)
      << hit_ms
      << " geometry_d2h_bytes=0\n";

  std::sort(observed_ms.begin(), observed_ms.end());
  const std::size_t middle = observed_ms.size() / 2;
  const double median_ms =
      observed_ms.size() & 1
          ? observed_ms[middle]
          : (observed_ms[middle - 1] + observed_ms[middle]) * 0.5;
  std::cout
      << "CONTACT4_ACTIVE_UNION_KACT_FUSED_GATE PASS"
      << " kact_file_sha256=" << kKactFileSha256
      << " kact_scene_sha256=" << kKactSceneSha256
      << " contexts=" << active.contexts.size()
      << " cells=" << active.cells.size()
      << " active_layer_contexts=" << active.layer_contexts.size()
      << " contact_layer_contexts=" << contact.layer_contexts.size()
      << " active_local_polygons=" << active.polygons.size()
      << " active_local_edges=" << active.edges.size()
      << " contact_local_polygons=" << contact.polygons.size()
      << " contact_local_edges=" << contact.edges.size()
      << " active_flat_polygons="
      << active.descriptor.flat_polygon_count
      << " active_flat_edges=" << active.descriptor.flat_edge_count
      << " contact_flat_polygons="
      << contact.descriptor.flat_polygon_count
      << " contact_flat_edges=" << contact.descriptor.flat_edge_count
      << " active_digest="
      << digest_hex(active.descriptor.scene_digest, 32)
      << " contact_digest="
      << digest_hex(contact.descriptor.scene_digest, 32)
      << " rectangles=" << reference.rectangle_count
      << " x_slabs=" << reference.x_slab_count
      << " union_memberships=" << reference.union_membership_count
      << " events=" << reference.event_count
      << " strips=" << reference.strip_interval_count
      << " raw_segments=" << reference.raw_segment_count
      << " boundary_segments=" << reference.boundary_segment_count
      << " contact_edges=" << reference.contact_expanded_edge_count
      << " grid_cells=" << reference.grid_cell_count
      << " contact_memberships=" << reference.contact_membership_count
      << " boundary_cell_visits="
      << reference.boundary_cell_visit_count
      << " member_visits=" << reference.member_visit_count
      << " candidate_pairs=" << reference.candidate_pair_count
      << " compact_build_ms=" << compact_build_ms
      << " median_call_ms=" << median_ms
      << " verification_total_ms="
      << elapsed_ms(all_begin, Clock::now())
      << "\n";
  return 0;
}

}  // namespace contact4_kact_fused

int main(int argc, char **argv)
{
  try {
    if (argc < 2 || argc > 4) {
      std::cerr
          << "usage: contact4_active_union_kact_fused_replay "
             "SCENE.kact [REPEAT>=2] [DEVICE]\n";
      return 2;
    }
    const unsigned long repeat =
        argc >= 3 ? std::stoul(argv[2]) : 2;
    const long device = argc >= 4 ? std::stol(argv[3]) : 0;
    if (repeat < 2 ||
        repeat > std::numeric_limits<std::uint32_t>::max() ||
        device < 0 ||
        device > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error("invalid repeat or device");
    }
    return contact4_kact_fused::run(
        argv[1], static_cast<std::uint32_t>(repeat),
        static_cast<int>(device));
  } catch (const std::exception &error) {
    std::cerr
        << "CONTACT4_ACTIVE_UNION_KACT_FUSED_GATE FAIL: "
        << error.what() << "\n";
    return 1;
  }
}
