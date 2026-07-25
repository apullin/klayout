#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Context = klayout_cuda_spatial_m1_width_space_context_v1;
using Cell = klayout_cuda_spatial_m1_width_space_cell_v1;
using Polygon = klayout_cuda_spatial_m1_width_space_polygon_v1;
using Edge = klayout_cuda_spatial_m1_width_space_edge_v1;
using Scene = klayout_cuda_spatial_contact4_active_union_scene_v1;
using Request = klayout_cuda_spatial_contact4_active_union_request_v1;
using Result = klayout_cuda_spatial_contact4_active_union_result_v1;

struct Box
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
};

struct Digest
{
  void bytes(const void *data, std::size_t size)
  {
    sha.update(data, size);
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

  db::cuda_active3_digest::Sha256 sha;
};

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

std::pair<std::int64_t, std::int64_t> transform_point(
    std::uint32_t code, std::int64_t x, std::int64_t y)
{
  switch (code) {
  case 0: return {x, y};
  case 1: return {-y, x};
  case 2: return {-x, -y};
  case 3: return {y, -x};
  case 4: return {x, -y};
  case 5: return {y, x};
  case 6: return {-x, y};
  case 7: return {-y, -x};
  default: throw std::runtime_error("invalid transform");
  }
}

struct StoredScene
{
  std::vector<Context> contexts;
  std::vector<std::uint32_t> layer_contexts;
  std::vector<std::uint64_t> polygon_offsets;
  std::vector<std::uint64_t> edge_offsets;
  std::vector<Cell> cells;
  std::vector<Polygon> polygons;
  std::vector<Edge> edges;
  Scene scene{};
};

std::array<std::uint8_t, 32> digest_scene(const Scene &scene)
{
  Digest digest;
  digest.bytes(scene.digest_domain, 8);
  digest.u32(scene.format_version);
  digest.u32(scene.dbu_per_micron);
  digest.u32(scene.root_cell);
  digest.u32(0);
  digest.u64(scene.context_count);
  digest.u64(scene.layer_context_count);
  digest.u64(scene.cell_count);
  digest.u64(scene.polygon_count);
  digest.u64(scene.edge_count);
  digest.u64(scene.flat_polygon_count);
  digest.u64(scene.flat_edge_count);
  digest.i64(scene.scene_left);
  digest.i64(scene.scene_bottom);
  digest.i64(scene.scene_right);
  digest.i64(scene.scene_top);
  for (std::uint64_t index = 0;
       index < scene.context_count; ++index) {
    const Context &context =
        static_cast<const Context *>(scene.contexts)[index];
    digest.i64(context.tx);
    digest.i64(context.ty);
    digest.u32(context.cell_id);
    digest.u32(context.transform_code);
  }
  for (std::uint64_t index = 0;
       index < scene.layer_context_count; ++index) {
    digest.u32(scene.layer_contexts[index]);
    digest.u64(scene.context_polygon_offsets[index]);
    digest.u64(scene.context_edge_offsets[index]);
  }
  for (std::uint64_t index = 0; index < scene.cell_count; ++index) {
    const Cell &cell =
        static_cast<const Cell *>(scene.cells)[index];
    digest.u64(cell.source_cell_index);
    digest.u64(cell.polygon_begin);
    digest.u64(cell.edge_begin);
    digest.u32(cell.polygon_count);
    digest.u32(cell.edge_count);
  }
  for (std::uint64_t index = 0;
       index < scene.polygon_count; ++index) {
    const Polygon &polygon =
        static_cast<const Polygon *>(scene.polygons)[index];
    digest.u64(polygon.edge_begin);
    digest.i64(polygon.left);
    digest.i64(polygon.bottom);
    digest.i64(polygon.right);
    digest.i64(polygon.top);
    digest.u32(polygon.polygon_id);
    digest.u32(polygon.edge_count);
  }
  for (std::uint64_t index = 0; index < scene.edge_count; ++index) {
    const Edge &edge =
        static_cast<const Edge *>(scene.edges)[index];
    digest.i64(edge.x1);
    digest.i64(edge.y1);
    digest.i64(edge.x2);
    digest.i64(edge.y2);
  }
  return digest.sha.finish();
}

StoredScene make_scene(
    const std::vector<Box> &boxes, std::uint32_t role,
    std::uint32_t layer, const char (&domain)[9],
    std::uint32_t transform)
{
  require(!boxes.empty(), "fixture scene is empty");
  StoredScene storage;
  storage.contexts.push_back(Context{0, 0, 0, 0});
  if (transform) {
    storage.contexts.push_back(Context{0, 0, 1, transform});
    storage.layer_contexts.push_back(1);
  } else {
    storage.layer_contexts.push_back(0);
  }
  storage.polygon_offsets.push_back(0);
  storage.edge_offsets.push_back(0);
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    const Box &box = boxes[index];
    require(
        box.left < box.right && box.bottom < box.top,
        "fixture box is empty");
    const std::uint64_t edge_begin = storage.edges.size();
    storage.edges.push_back(
        Edge{box.left, box.bottom, box.left, box.top});
    storage.edges.push_back(
        Edge{box.left, box.top, box.right, box.top});
    storage.edges.push_back(
        Edge{box.right, box.top, box.right, box.bottom});
    storage.edges.push_back(
        Edge{box.right, box.bottom, box.left, box.bottom});
    storage.polygons.push_back(
        Polygon{edge_begin, box.left, box.bottom,
                box.right, box.top,
                static_cast<std::uint32_t>(index), 4});
  }
  if (transform) {
    storage.cells.push_back(
        Cell{UINT64_C(0x1234), 0, 0, 0, 0});
  }
  storage.cells.push_back(
      Cell{transform ? UINT64_C(0x1235) : UINT64_C(0x1234),
           0, 0,
           static_cast<std::uint32_t>(storage.polygons.size()),
           static_cast<std::uint32_t>(storage.edges.size())});

  bool have_bounds = false;
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
  for (const Box &box : boxes) {
    const std::int64_t xs[2] = {box.left, box.right};
    const std::int64_t ys[2] = {box.bottom, box.top};
    for (int xi = 0; xi < 2; ++xi) {
      for (int yi = 0; yi < 2; ++yi) {
        const auto point =
            transform_point(transform, xs[xi], ys[yi]);
        if (!have_bounds) {
          left = right = point.first;
          bottom = top = point.second;
          have_bounds = true;
        } else {
          left = std::min(left, point.first);
          bottom = std::min(bottom, point.second);
          right = std::max(right, point.first);
          top = std::max(top, point.second);
        }
      }
    }
  }

  Scene &scene = storage.scene;
  scene.struct_size = sizeof(scene);
  scene.role = role;
  scene.format_version = 1;
  scene.dbu_per_micron = 2000;
  scene.root_cell = 0;
  scene.layer = layer;
  scene.datatype = 0;
  scene.contexts = storage.contexts.data();
  scene.context_count = storage.contexts.size();
  scene.context_record_bytes = sizeof(Context);
  scene.layer_contexts = storage.layer_contexts.data();
  scene.layer_context_count = storage.layer_contexts.size();
  scene.context_polygon_offsets = storage.polygon_offsets.data();
  scene.context_polygon_offset_count = storage.polygon_offsets.size();
  scene.context_edge_offsets = storage.edge_offsets.data();
  scene.context_edge_offset_count = storage.edge_offsets.size();
  scene.cells = storage.cells.data();
  scene.cell_count = storage.cells.size();
  scene.cell_record_bytes = sizeof(Cell);
  scene.polygons = storage.polygons.data();
  scene.polygon_count = storage.polygons.size();
  scene.polygon_record_bytes = sizeof(Polygon);
  scene.edges = storage.edges.data();
  scene.edge_count = storage.edges.size();
  scene.edge_record_bytes = sizeof(Edge);
  scene.flat_polygon_count = storage.polygons.size();
  scene.flat_edge_count = storage.edges.size();
  scene.scene_left = left;
  scene.scene_bottom = bottom;
  scene.scene_right = right;
  scene.scene_top = top;
  std::copy(domain, domain + 8, scene.digest_domain);
  const std::array<std::uint8_t, 32> digest = digest_scene(scene);
  std::copy(digest.begin(), digest.end(), scene.scene_digest);
  return storage;
}

Request make_request(const Scene &active, const Scene &contact)
{
  Request request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.device = 0;
  request.distance = 10;
  request.grid_cell_size = 2000;
  request.active = active;
  request.contact = contact;
  request.max_contexts = 16;
  request.max_rectangles = 1024;
  request.max_x_slabs = 1024;
  request.max_union_memberships = 8192;
  request.max_events = 16384;
  request.max_raw_segments = 8192;
  request.max_boundary_segments = 8192;
  request.max_slabs_per_rectangle = 64;
  request.max_contact_edges = 4096;
  request.max_grid_cells = 4096;
  request.max_contact_memberships = 65536;
  request.max_boundary_cell_visits = 65536;
  request.max_member_visits = 1048576;
  request.max_pair_work = 1048576;
  request.max_cells_per_contact_edge = 64;
  request.max_cells_per_boundary_edge = 64;
  return request;
}

Result run(const StoredScene &active, const StoredScene &contact)
{
  Request request = make_request(active.scene, contact.scene);
  Result result{};
  const int status =
      klayout_cuda_spatial_run_contact4_active_union_empty_v1(
          &request, &result);
  require(status == static_cast<int>(result.status),
          "return/result status mismatch");
  return result;
}

void require_clean(const Result &result, const char *fixture)
{
  require(
      result.status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.disposition ==
              KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE &&
          !result.fallback_flags && !result.device_flags &&
          !result.raw_hit_count && !result.uncertain_count &&
          result.rectangle_count &&
          result.boundary_segment_count &&
          result.contact_expanded_edge_count &&
          result.total_ns,
      std::string(fixture) + " did not certify empty");
}

void require_hit(const Result &result, const char *fixture)
{
  if (result.status != KLAYOUT_CUDA_SPATIAL_OK ||
      result.disposition !=
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_RAW_HITS ||
      result.fallback_flags || result.device_flags ||
      !result.raw_hit_count || result.uncertain_count) {
    throw std::runtime_error(
        std::string(fixture) + " did not report a raw hit:"
        " status=" + std::to_string(result.status) +
        " disposition=" + std::to_string(result.disposition) +
        " fallback=" + std::to_string(result.fallback_flags) +
        " device=" + std::to_string(result.device_flags) +
        " candidates=" + std::to_string(result.candidate_pair_count) +
        " hits=" + std::to_string(result.raw_hit_count) +
        " uncertain=" + std::to_string(result.uncertain_count) +
        " message='" + result.message + "'");
  }
}

}  // namespace

int main()
{
  try {
    const char active_domain[9] =
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_DIGEST_DOMAIN;
    const char contact_domain[9] =
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_DIGEST_DOMAIN;

    StoredScene active = make_scene(
        {{0, 0, 100, 100}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE, 1,
        active_domain, 0);
    StoredScene far_contact = make_scene(
        {{200, 0, 210, 10}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
        contact_domain, 0);
    require_clean(run(active, far_contact), "far");

    StoredScene strict_contact = make_scene(
        {{80, 40, 90, 60}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
        contact_domain, 0);
    require_clean(run(active, strict_contact), "strict-10");

    StoredScene hit_contact = make_scene(
        {{85, 40, 95, 60}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
        contact_domain, 0);
    require_hit(run(active, hit_contact), "gap-5");

    StoredScene mirrored_active = make_scene(
        {{0, 0, 100, 100}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE, 1,
        active_domain, 4);
    StoredScene mirrored_contact = make_scene(
        {{85, 40, 95, 60}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
        contact_domain, 4);
    require_hit(
        run(mirrored_active, mirrored_contact), "mirrored-gap-5");

    StoredScene kissing_active = make_scene(
        {{0, 0, 100, 100}, {100, 100, 200, 200}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE, 1,
        active_domain, 0);
    require_clean(run(kissing_active, far_contact), "kissing-clean");

    Request corrupt = make_request(active.scene, far_contact.scene);
    corrupt.active.scene_digest[0] ^= 1;
    Result corrupt_result{};
    const int corrupt_status =
        klayout_cuda_spatial_run_contact4_active_union_empty_v1(
            &corrupt, &corrupt_result);
    require(
        corrupt_status != KLAYOUT_CUDA_SPATIAL_OK &&
            corrupt_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_UNCERTAIN,
        "corrupt digest did not fail closed");

    std::cout
        << "CONTACT4_ACTIVE_UNION_BACKEND_SMOKE PASS"
        << " clean=3 hits=2 reflected=1 kissing=1 corrupt=1\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "CONTACT4_ACTIVE_UNION_BACKEND_SMOKE FAIL: "
        << error.what() << "\n";
    return 1;
  }
}
