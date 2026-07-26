#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Context = klayout_cuda_spatial_m1_width_space_context_v1;
using Cell = klayout_cuda_spatial_m1_width_space_cell_v1;
using Polygon = klayout_cuda_spatial_m1_width_space_polygon_v1;
using Edge = klayout_cuda_spatial_m1_width_space_edge_v1;
using Scene = klayout_cuda_spatial_implant15_scene_v1;
using Request = klayout_cuda_spatial_implant15_request_v1;
using Result = klayout_cuda_spatial_implant15_result_v1;

struct Box
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
};

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

struct Digest
{
  db::cuda_active3_digest::Sha256 sha;

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
};

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
  for (std::uint64_t index = 0;
       index < scene.cell_count; ++index) {
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
  for (std::uint64_t index = 0;
       index < scene.edge_count; ++index) {
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
    std::uint32_t layer, std::uint32_t datatype,
    const char (&domain)[9])
{
  require(!boxes.empty(), "fixture scene is empty");
  StoredScene stored;
  stored.contexts.push_back(Context{0, 0, 0, 0});
  stored.layer_contexts.push_back(0);
  stored.polygon_offsets.push_back(0);
  stored.edge_offsets.push_back(0);
  for (std::size_t id = 0; id < boxes.size(); ++id) {
    const Box &box = boxes[id];
    require(
        box.left < box.right && box.bottom < box.top,
        "fixture box is empty");
    const std::uint64_t edge_begin = stored.edges.size();
    // Clockwise: material remains on the right of every directed edge.
    stored.edges.push_back(
        Edge{box.left, box.bottom, box.left, box.top});
    stored.edges.push_back(
        Edge{box.left, box.top, box.right, box.top});
    stored.edges.push_back(
        Edge{box.right, box.top, box.right, box.bottom});
    stored.edges.push_back(
        Edge{box.right, box.bottom, box.left, box.bottom});
    stored.polygons.push_back(
        Polygon{
            edge_begin, box.left, box.bottom, box.right, box.top,
            static_cast<std::uint32_t>(id), 4});
  }
  stored.cells.push_back(
      Cell{
          UINT64_C(0x1234), 0, 0,
          static_cast<std::uint32_t>(stored.polygons.size()),
          static_cast<std::uint32_t>(stored.edges.size())});
  Scene &scene = stored.scene;
  scene.struct_size = sizeof(scene);
  scene.role = role;
  scene.format_version = 1;
  scene.dbu_per_micron = 2000;
  scene.root_cell = 0;
  scene.layer = layer;
  scene.datatype = datatype;
  scene.contexts = stored.contexts.data();
  scene.context_count = stored.contexts.size();
  scene.context_record_bytes = sizeof(Context);
  scene.layer_contexts = stored.layer_contexts.data();
  scene.layer_context_count = stored.layer_contexts.size();
  scene.context_polygon_offsets = stored.polygon_offsets.data();
  scene.context_polygon_offset_count = stored.polygon_offsets.size();
  scene.context_edge_offsets = stored.edge_offsets.data();
  scene.context_edge_offset_count = stored.edge_offsets.size();
  scene.cells = stored.cells.data();
  scene.cell_count = stored.cells.size();
  scene.cell_record_bytes = sizeof(Cell);
  scene.polygons = stored.polygons.data();
  scene.polygon_count = stored.polygons.size();
  scene.polygon_record_bytes = sizeof(Polygon);
  scene.edges = stored.edges.data();
  scene.edge_count = stored.edges.size();
  scene.edge_record_bytes = sizeof(Edge);
  scene.flat_polygon_count = stored.polygons.size();
  scene.flat_edge_count = stored.edges.size();
  scene.scene_left = boxes.front().left;
  scene.scene_bottom = boxes.front().bottom;
  scene.scene_right = boxes.front().right;
  scene.scene_top = boxes.front().top;
  for (const Box &box : boxes) {
    scene.scene_left = std::min(scene.scene_left, box.left);
    scene.scene_bottom = std::min(scene.scene_bottom, box.bottom);
    scene.scene_right = std::max(scene.scene_right, box.right);
    scene.scene_top = std::max(scene.scene_top, box.top);
  }
  std::copy(domain, domain + 8, scene.digest_domain);
  const std::array<std::uint8_t, 32> digest =
      digest_scene(scene);
  std::copy(digest.begin(), digest.end(), scene.scene_digest);
  return stored;
}

Request make_request(
    const Scene &nplus, const Scene &pplus,
    const Scene &gate, const Scene &contact)
{
  Request request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_RESIDENT_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.requested_mask =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES;
  request.device = 0;
  request.implant1_distance = 140;
  request.implant2_distance = 50;
  request.implant3_distance = 90;
  request.implant4_distance = 90;
  request.grid_cell_size = 2000;
  request.nplus = nplus;
  request.pplus = pplus;
  request.gate = gate;
  request.contact = contact;
  request.capacity.struct_size = sizeof(request.capacity);
  request.capacity.max_slabs_per_rectangle = 256;
  request.capacity.max_cells_per_secondary_edge = 64;
  request.capacity.max_cells_per_boundary_edge = 64;
  request.capacity.max_contexts = 64;
  request.capacity.max_rectangles = 4096;
  request.capacity.max_x_slabs = 4096;
  request.capacity.max_union_memberships = 65536;
  request.capacity.max_events = 131072;
  request.capacity.max_raw_segments = 65536;
  request.capacity.max_boundary_segments = 65536;
  request.capacity.max_gate_edges = 65536;
  request.capacity.max_contact_edges = 65536;
  request.capacity.max_grid_cells = 65536;
  request.capacity.max_secondary_memberships = 65536;
  request.capacity.max_gate_boundary_cell_visits = 65536;
  request.capacity.max_contact_boundary_cell_visits = 65536;
  request.capacity.max_member_visits = 1048576;
  request.capacity.max_pair_work = 1048576;
  request.capacity.max_morphology_work = 1048576;
  request.capacity.max_overlap_work = 1048576;
  return request;
}

Result run(
    const StoredScene &nplus, const StoredScene &pplus,
    const StoredScene &gate, const StoredScene &contact)
{
  Request request = make_request(
      nplus.scene, pplus.scene, gate.scene, contact.scene);
  Result result{};
  const int status =
      klayout_cuda_spatial_run_implant15_raw_empty_v1(
          &request, &result);
  require(
      status == static_cast<int>(result.status),
      "return/result status mismatch");
  return result;
}

std::string result_text(const Result &result)
{
  return " status=" + std::to_string(result.status) +
         " disposition=" + std::to_string(result.disposition) +
         " fallback=" + std::to_string(result.fallback_flags) +
         " device=" + std::to_string(result.device_flags) +
         " clean=" + std::to_string(result.clean_mask) +
         " certified=" +
         std::to_string(result.certified_empty_mask) +
         " i1=" + std::to_string(result.implant1_hit_count) +
         " i2=" + std::to_string(result.implant2_hit_count) +
         " i3=" + std::to_string(result.implant3_hit_count) +
         " i4=" + std::to_string(result.implant4_hit_count) +
         " i5=" + std::to_string(result.implant5_hit_count) +
         " message='" + result.message + "'";
}

void require_clean(const Result &result, const char *fixture)
{
  require(
      result.status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.disposition ==
              KLAYOUT_CUDA_SPATIAL_IMPLANT15_COMPLETE &&
          result.certified_empty_mask ==
              KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES &&
          result.clean_mask ==
              KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES &&
          !result.fallback_flags && !result.device_flags &&
          !result.implant1_hit_count &&
          !result.implant2_hit_count &&
          !result.implant3_hit_count &&
          !result.implant4_hit_count &&
          !result.implant5_hit_count &&
          result.event_count ==
              2 * result.union_membership_count &&
          result.raw_segment_count >=
              result.boundary_segment_count,
      std::string(fixture) + " did not certify empty:" +
          result_text(result));
}

void require_rule_hit(
    const Result &result, std::uint32_t rule,
    const char *fixture)
{
  const std::uint64_t hits[] = {
      result.implant1_hit_count,
      result.implant2_hit_count,
      result.implant3_hit_count,
      result.implant4_hit_count,
      result.implant5_hit_count};
  require(
      result.status == KLAYOUT_CUDA_SPATIAL_OK &&
          result.disposition ==
              KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_HITS &&
          rule >= 1 && rule <= 5 && hits[rule - 1] != 0 &&
          !result.certified_empty_mask,
      std::string(fixture) + " did not report rule hit:" +
          result_text(result));
}

void require_morphology_hit_with_corner_uncertainty(
    const Result &result, std::uint32_t rule,
    const char *fixture)
{
  const std::uint64_t hits[] = {
      result.implant1_hit_count,
      result.implant2_hit_count,
      result.implant3_hit_count,
      result.implant4_hit_count,
      result.implant5_hit_count};
  require(
      result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
          result.disposition ==
              KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN &&
          (rule == 3 || rule == 4) && hits[rule - 1] != 0 &&
          result.implant3_uncertain_count != 0 &&
          result.implant4_uncertain_count != 0 &&
          !result.certified_empty_mask,
      std::string(fixture) +
          " did not fail closed on its exact strip hit plus"
          " ambiguous corner attribution:" +
          result_text(result));
}

StoredScene nplus(const std::vector<Box> &boxes)
{
  const char domain[9] =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_DIGEST_DOMAIN;
  return make_scene(
      boxes, KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_ROLE,
      4, 0, domain);
}

StoredScene pplus(const std::vector<Box> &boxes)
{
  const char domain[9] =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_DIGEST_DOMAIN;
  return make_scene(
      boxes, KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_ROLE,
      5, 0, domain);
}

StoredScene gate(const std::vector<Box> &boxes)
{
  const char domain[9] =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_DIGEST_DOMAIN;
  return make_scene(
      boxes, KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_ROLE,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_LAYER,
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_DATATYPE,
      domain);
}

StoredScene contact(const std::vector<Box> &boxes)
{
  const char domain[9] =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_DIGEST_DOMAIN;
  return make_scene(
      boxes, KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_ROLE,
      10, 0, domain);
}

}  // namespace

int main()
{
  try {
    StoredScene clean_n = nplus({{0, 0, 400, 400}});
    StoredScene clean_p = pplus({{600, 0, 1000, 400}});
    StoredScene far_gate = gate({{1400, 0, 1500, 100}});
    StoredScene far_contact = contact({{1600, 0, 1700, 100}});
    require_clean(
        run(clean_n, clean_p, far_gate, far_contact),
        "clean");

    // IMPLANT.5 is set intersection, so a shared boundary is clean.
    StoredScene touching_p = pplus({{400, 0, 800, 400}});
    require_clean(
        run(clean_n, touching_p, far_gate, far_contact),
        "touching-clean");

    StoredScene overlap_p = pplus({{399, 0, 799, 400}});
    require_rule_hit(
        run(clean_n, overlap_p, far_gate, far_contact),
        5, "one-dbu-overlap");

    StoredScene narrow_n = nplus({{0, 0, 89, 400}});
    require_morphology_hit_with_corner_uncertainty(
        run(narrow_n, clean_p, far_gate, far_contact),
        3, "width-89");

    StoredScene threshold_width_n = nplus({{0, 0, 90, 400}});
    require_clean(
        run(
            threshold_width_n, clean_p, far_gate,
            far_contact),
        "width-90-clean");

    StoredScene close_p = pplus({{489, 0, 889, 400}});
    require_morphology_hit_with_corner_uncertainty(
        run(clean_n, close_p, far_gate, far_contact),
        4, "space-89");

    StoredScene threshold_space_p =
        pplus({{490, 0, 890, 400}});
    require_clean(
        run(
            clean_n, threshold_space_p, far_gate,
            far_contact),
        "space-90-clean");

    StoredScene near_gate = gate({{500, 100, 600, 300}});
    const Result gate_hit =
        run(clean_n, clean_p, near_gate, far_contact);
    require_rule_hit(gate_hit, 1, "gate-gap-100");
    require(
        gate_hit.contact_expanded_edge_count ==
            far_contact.scene.flat_edge_count &&
            gate_hit.implant2_secondary_membership_count,
        "IMPLANT.1 hit did not continue through IMPLANT.2");

    StoredScene threshold_gate =
        gate({{1140, 100, 1240, 300}});
    require_clean(
        run(
            clean_n, clean_p, threshold_gate,
            far_contact),
        "gate-gap-140-clean");

    StoredScene near_contact = contact({{425, 100, 475, 300}});
    require_rule_hit(
        run(clean_n, clean_p, far_gate, near_contact),
        2, "contact-gap-25");

    StoredScene threshold_contact =
        contact({{1050, 100, 1100, 300}});
    require_clean(
        run(
            clean_n, clean_p, far_gate,
            threshold_contact),
        "contact-gap-50-clean");

    Request malformed = make_request(
        clean_n.scene, clean_p.scene,
        far_gate.scene, far_contact.scene);
    malformed.nplus.scene_digest[0] ^= 1;
    Result malformed_result{};
    const int malformed_status =
        klayout_cuda_spatial_run_implant15_raw_empty_v1(
            &malformed, &malformed_result);
    require(
        malformed_status != KLAYOUT_CUDA_SPATIAL_OK &&
            malformed_result.disposition ==
                KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN,
        "digest corruption did not fail closed");

    std::cout
        << "IMPLANT15_BACKEND_SMOKE PASS"
        << " clean=1 touching_clean=1 overlap_1dbu=1"
        << " width=1 space=1 gate=1 contact=1"
        << " threshold_equal_clean=4 malformed=1\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "IMPLANT15_BACKEND_SMOKE FAIL: "
        << error.what() << "\n";
    return 1;
  }
}
