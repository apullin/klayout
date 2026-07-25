/*
 * Offline feasibility gate for bridging an exact sorted CUDA Manhattan-union
 * boundary into KLayout's stock flat Region morphology.
 *
 * This is intentionally not a live hook.  It accepts only the pinned
 * production CPU-merged KM1WS oracle, reconstructs simple clockwise contours
 * with checked endpoint topology, marks the resulting flat Region as already
 * merged through the public C++ Shapes constructor, and times the exact
 * cumulative M2.5-.9 operations.
 */

#include "m2_merged_boundary_oracle.h"

#include "dbEdgePairs.h"
#include "dbEdges.h"
#include "dbEdgesUtils.h"
#include "dbFlatRegion.h"
#include "dbPolygon.h"
#include "dbRegion.h"
#include "dbShapes.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace oracle = klayout_cuda::m2_boundary_oracle;
using Clock = std::chrono::steady_clock;

constexpr char kFileSha256[] =
    "980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f";
constexpr char kSceneSha256[] =
    "441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480";
constexpr char kRawSceneSha256[] =
    "dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2";
constexpr char kBoundarySha256[] =
    "94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d";
constexpr std::uint64_t kBoundarySegments = UINT64_C(4385384);
constexpr std::uint64_t kBoundaryFnv64 =
    UINT64_C(7541395996791771514);
constexpr std::uint64_t kContours = UINT64_C(14222);
constexpr std::uint64_t kGt90RawPolygons = UINT64_C(1063596);
constexpr std::uint64_t kGt90MergedEdges = UINT64_C(4254384);
constexpr std::uint64_t kGt90LongEdges = UINT64_C(8);

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Point
{
  std::int32_t x;
  std::int32_t y;

  bool operator==(const Point &other) const
  {
    return x == other.x && y == other.y;
  }
};

struct PointHash
{
  std::size_t operator()(const Point &point) const
  {
    std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(point.x))
         << 32) |
        static_cast<std::uint32_t>(point.y);
    key ^= key >> 30;
    key *= UINT64_C(0xbf58476d1ce4e5b9);
    key ^= key >> 27;
    key *= UINT64_C(0x94d049bb133111eb);
    key ^= key >> 31;
    return static_cast<std::size_t>(key);
  }
};

struct DirectedEdge
{
  Point first;
  Point second;
};

std::int32_t narrow_coord(std::int64_t value)
{
  if (value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    throw std::runtime_error("boundary coordinate exceeds this 32-bit build");
  }
  return static_cast<std::int32_t>(value);
}

DirectedEdge directed_edge(const oracle::DirectedSegmentI64 &segment)
{
  if (segment.lo >= segment.hi ||
      (segment.side != -1 && segment.side != 1)) {
    throw std::runtime_error("invalid canonical boundary segment");
  }
  const std::int32_t fixed = narrow_coord(segment.fixed);
  const std::int32_t lo = narrow_coord(segment.lo);
  const std::int32_t hi = narrow_coord(segment.hi);

  // KLayout polygons keep their material on the right side of each edge.
  if (segment.axis == oracle::SegmentAxis::horizontal) {
    return segment.side < 0
               ? DirectedEdge{{hi, fixed}, {lo, fixed}}
               : DirectedEdge{{lo, fixed}, {hi, fixed}};
  }
  if (segment.axis == oracle::SegmentAxis::vertical) {
    return segment.side < 0
               ? DirectedEdge{{fixed, lo}, {fixed, hi}}
               : DirectedEdge{{fixed, hi}, {fixed, lo}};
  }
  throw std::runtime_error("invalid canonical boundary axis");
}

struct StitchResult
{
  db::Shapes polygons{false};
  std::uint64_t contours = 0;
  std::uint64_t vertices = 0;
  std::uint64_t max_vertices = 0;
};

StitchResult stitch(const std::vector<oracle::DirectedSegmentI64> &segments,
                    std::uint64_t expected_segments,
                    std::uint64_t expected_contours)
{
  if (segments.size() != expected_segments ||
      segments.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("unqualified boundary segment census");
  }

  const std::size_t count = segments.size();
  std::vector<DirectedEdge> edges;
  edges.reserve(count);
  std::unordered_map<Point, std::uint32_t, PointHash> outgoing;
  outgoing.reserve(count);
  outgoing.max_load_factor(0.75f);

  for (std::size_t index = 0; index < count; ++index) {
    const DirectedEdge edge = directed_edge(segments[index]);
    if (!outgoing.emplace(edge.first, static_cast<std::uint32_t>(index))
             .second) {
      throw std::runtime_error("degree/topology failure: duplicate outgoing "
                               "boundary endpoint");
    }
    edges.push_back(edge);
  }

  std::vector<std::uint32_t> next(count);
  std::vector<std::uint8_t> indegree(count, 0);
  for (std::size_t index = 0; index < count; ++index) {
    const auto successor = outgoing.find(edges[index].second);
    if (successor == outgoing.end()) {
      throw std::runtime_error(
          "degree/topology failure: open boundary endpoint");
    }
    const std::uint32_t next_index = successor->second;
    if (++indegree[next_index] != 1) {
      throw std::runtime_error(
          "degree/topology failure: duplicate incoming boundary endpoint");
    }
    next[index] = next_index;
  }
  if (std::find(indegree.begin(), indegree.end(), std::uint8_t{0}) !=
      indegree.end()) {
    throw std::runtime_error(
        "degree/topology failure: missing incoming boundary endpoint");
  }

  StitchResult result;
  result.polygons.reserve(db::Polygon::tag(), expected_contours);
  std::vector<std::uint8_t> visited(count, 0);
  std::vector<db::Point> points;
  for (std::uint32_t seed = 0; seed < count; ++seed) {
    if (visited[seed]) continue;
    points.clear();
    std::uint32_t current = seed;
    do {
      if (visited[current]) {
        throw std::runtime_error(
            "degree/topology failure: contour enters another cycle");
      }
      visited[current] = 1;
      points.emplace_back(edges[current].first.x, edges[current].first.y);
      current = next[current];
      if (points.size() > count) {
        throw std::runtime_error("degree/topology failure: unclosed contour");
      }
    } while (current != seed);
    if (points.size() < 4) {
      throw std::runtime_error(
          "degree/topology failure: contour has fewer than four edges");
    }

    db::Polygon polygon;
    // Normalize is retained intentionally: the bridge must hand stock KLayout
    // a fully valid Polygon, not rely on its current tolerance for pre-oriented
    // point streams.
    polygon.assign_hull(points.begin(), points.end(), false, false, true);
    if (polygon.vertices() != points.size() || polygon.holes() != 0) {
      throw std::runtime_error(
          "KLayout polygon materialization changed canonical contour");
    }
    result.polygons.insert(polygon);
    ++result.contours;
    result.vertices += points.size();
    result.max_vertices =
        std::max<std::uint64_t>(result.max_vertices, points.size());
  }

  if (result.contours != expected_contours || result.vertices != count) {
    throw std::runtime_error("stitched contour census disagrees with oracle");
  }
  return result;
}

oracle::DirectedSegmentI64 segment(
    std::int64_t fixed, std::int64_t lo, std::int64_t hi,
    std::int32_t side, oracle::SegmentAxis axis)
{
  return {fixed, lo, hi, side, axis};
}

std::vector<oracle::DirectedSegmentI64> rectangle(
    std::int64_t left, std::int64_t bottom, std::int64_t right,
    std::int64_t top)
{
  return {
      segment(bottom, left, right, -1,
              oracle::SegmentAxis::horizontal),
      segment(top, left, right, 1, oracle::SegmentAxis::horizontal),
      segment(left, bottom, top, -1, oracle::SegmentAxis::vertical),
      segment(right, bottom, top, 1, oracle::SegmentAxis::vertical)};
}

template <class Mutator>
void require_topology_fallback(const std::string &name, Mutator mutator)
{
  std::vector<oracle::DirectedSegmentI64> fixture =
      rectangle(0, 0, 10, 10);
  mutator(fixture);
  try {
    (void)stitch(fixture, fixture.size(), 1);
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(name + " did not fail closed");
}

void self_test()
{
  const std::vector<oracle::DirectedSegmentI64> box =
      rectangle(0, 0, 10, 20);
  const StitchResult valid = stitch(box, 4, 1);
  if (valid.contours != 1 || valid.vertices != 4 ||
      valid.max_vertices != 4) {
    throw std::runtime_error("directed rectangle stitch failed");
  }

  require_topology_fallback(
      "open endpoint",
      [](std::vector<oracle::DirectedSegmentI64> &fixture) {
        fixture.pop_back();
      });
  require_topology_fallback(
      "duplicate outgoing endpoint",
      [](std::vector<oracle::DirectedSegmentI64> &fixture) {
        fixture.push_back(fixture.front());
      });
  require_topology_fallback(
      "degenerate segment",
      [](std::vector<oracle::DirectedSegmentI64> &fixture) {
        fixture.front().hi = fixture.front().lo;
      });
  require_topology_fallback(
      "degree-four kissing point",
      [](std::vector<oracle::DirectedSegmentI64> &fixture) {
        const auto second = rectangle(10, 10, 20, 20);
        fixture.insert(fixture.end(), second.begin(), second.end());
      });

  std::cout << "M2_FLAT_REGION_BRIDGE_SELF_TEST PASS gates=5\n";
}

std::uint64_t edge_count(const db::Edges &edges)
{
  return static_cast<std::uint64_t>(edges.count());
}

std::uint64_t pair_count(const db::EdgePairs &pairs)
{
  return static_cast<std::uint64_t>(pairs.count());
}

void run(const std::string &path, bool candidate_stream,
         bool audit_full_boundary, bool audit_m2_width_space)
{
  const auto all_begin = Clock::now();

  const auto load_begin = Clock::now();
  oracle::BoundaryOracle boundary;
  if (candidate_stream) {
    oracle::CandidateStreamIdentity identity;
    identity.producer_scene_sha256 = kRawSceneSha256;
    identity.qualification_scene_sha256 = kSceneSha256;
    identity.boundary_sha256 = kBoundarySha256;
    identity.segment_count = kBoundarySegments;
    identity.boundary_fnv64 = kBoundaryFnv64;
    boundary.segments =
        oracle::read_candidate_stream(path, identity);
    boundary.scene_sha256 = kRawSceneSha256;
    boundary.boundary_sha256 = kBoundarySha256;
    boundary.boundary_fnv64 = kBoundaryFnv64;
  } else {
    oracle::LoadOptions options;
    options.expected_file_sha256 = kFileSha256;
    options.expected_scene_sha256 = kSceneSha256;
    options.expected_boundary_sha256 = kBoundarySha256;
    boundary = oracle::load_cpu_merged_boundary(path, options);
  }
  const auto load_end = Clock::now();

  const auto stitch_begin = Clock::now();
  StitchResult stitched =
      stitch(boundary.segments, kBoundarySegments, kContours);
  const auto stitch_end = Clock::now();

  const auto region_begin = Clock::now();
  // Region(Shapes, ..., is_merged=true) currently inserts one shape at a time,
  // which invalidates the requested merged flag.  The direct public delegate
  // constructor copies the already validated Shapes payload and preserves it.
  auto *flat = new db::FlatRegion(stitched.polygons, true);
  flat->set_merged_semantics(true);
  db::Region m2(flat);
  const auto region_end = Clock::now();
  if (!m2.merged_semantics() || !m2.is_merged() ||
      m2.count() != kContours) {
    throw std::runtime_error(
        "flat bridge did not preserve exact already-merged state");
  }

  double m2_width_ms = 0.0;
  double m2_space_ms = 0.0;
  std::uint64_t m2_width_pairs = 0;
  std::uint64_t m2_space_pairs = 0;
  if (audit_m2_width_space) {
    const auto width_begin = Clock::now();
    const db::EdgePairs width = m2.width_check(140);
    const auto width_end = Clock::now();
    const auto space_begin = Clock::now();
    const db::EdgePairs space = m2.space_check(140);
    const auto space_end = Clock::now();
    m2_width_ms = milliseconds(width_begin, width_end);
    m2_space_ms = milliseconds(space_begin, space_end);
    m2_width_pairs = pair_count(width);
    m2_space_pairs = pair_count(space);
    if (m2_width_pairs != 0 || m2_space_pairs != 0) {
      throw std::runtime_error("flat M2.1/M2.2 stock certificate is not clean");
    }
  }

  const auto shrink90_begin = Clock::now();
  db::Region gt90_shrunk = m2.sized(db::Coord(-89));
  const auto shrink90_end = Clock::now();
  if (!gt90_shrunk.is_merged()) {
    throw std::runtime_error("negative F90 sizing lost merged state");
  }

  const auto grow90_begin = Clock::now();
  db::Region gt90 = gt90_shrunk.sized(db::Coord(90));
  const auto grow90_end = Clock::now();
  const std::uint64_t gt90_raw = gt90.count();
  if (gt90_raw != kGt90RawPolygons || gt90.is_merged()) {
    throw std::runtime_error("F90 raw polygon census/state mismatch");
  }

  db::EdgeLengthFilter long_filter(
      600, std::numeric_limits<db::Edge::distance_type>::max(), false);
  const auto edges90_begin = Clock::now();
  std::uint64_t gt90_edge_count = 0;
  db::Edges long_edges;
  if (audit_full_boundary) {
    db::Edges gt90_edges = gt90.edges();
    gt90_edge_count = edge_count(gt90_edges);
    if (gt90_edge_count != kGt90MergedEdges) {
      throw std::runtime_error("F90 merged-edge census mismatch");
    }
    long_edges = gt90_edges.filtered(long_filter);
  } else {
    // Exact stock operation, but apply the length filter while generating
    // edges so 4.25 million unselected edges never enter a second collection.
    long_edges = gt90.edges(long_filter);
  }
  const auto edges90_end = Clock::now();
  const std::uint64_t gt90_long = edge_count(long_edges);
  if (gt90_long != kGt90LongEdges) {
    throw std::runtime_error("F90 long-edge census mismatch");
  }

  const auto space_begin = Clock::now();
  db::EdgePairs spacing = long_edges.space_check(180);
  const auto space_end = Clock::now();
  if (!spacing.empty()) {
    throw std::runtime_error("F90 long-edge spacing certificate is not clean");
  }

  const auto shrink270_begin = Clock::now();
  db::Region gt270_shrunk = gt90.sized(db::Coord(-269));
  const auto shrink270_end = Clock::now();
  const std::uint64_t gt270_shrunk_count = gt270_shrunk.count();
  if (gt270_shrunk_count != 0 || !gt270_shrunk.empty()) {
    throw std::runtime_error("F270 erosion did not collapse to empty");
  }

  const auto grow270_begin = Clock::now();
  db::Region gt270 = gt270_shrunk.sized(db::Coord(270));
  const auto grow270_end = Clock::now();
  if (!gt270.empty() || gt270.count() != 0) {
    throw std::runtime_error("F270 output is not empty");
  }

  std::cout
      << "M2_FLAT_REGION_BRIDGE PASS"
      << " input_mode="
      << (candidate_stream ? "gpu-km2bnd" : "cpu-km1ws-oracle")
      << " producer_scene_sha256="
      << (candidate_stream ? kRawSceneSha256 : boundary.scene_sha256)
      << " qualification_scene_sha256=" << kSceneSha256
      << " boundary_sha256=" << boundary.boundary_sha256
      << " boundary_fnv64=" << boundary.boundary_fnv64
      << " segments=" << boundary.segments.size()
      << " contours=" << stitched.contours
      << " vertices=" << stitched.vertices
      << " max_vertices=" << stitched.max_vertices
      << " merged_semantics=" << m2.merged_semantics()
      << " is_merged=" << m2.is_merged()
      << " m2_width_pairs="
      << (audit_m2_width_space ? std::to_string(m2_width_pairs)
                               : "not-audited")
      << " m2_space_pairs="
      << (audit_m2_width_space ? std::to_string(m2_space_pairs)
                               : "not-audited")
      << " gt90_raw_polygons=" << gt90_raw
      << " gt90_merged_edges="
      << (audit_full_boundary ? std::to_string(gt90_edge_count) : "not-audited")
      << " gt90_long_edges=" << gt90_long
      << " gt90_space_pairs=" << pair_count(spacing)
      << " gt270_shrunk_polygons=" << gt270_shrunk_count
      << " gt270_polygons=" << gt270.count() << "\n"
      << std::fixed << std::setprecision(3)
      << "TIMING"
      << " input_load_validate_ms=" << milliseconds(load_begin, load_end)
      << " oracle_qualification_ms="
      << (candidate_stream ? 0.0 : milliseconds(load_begin, load_end))
      << " endpoint_stitch_and_shapes_ms="
      << milliseconds(stitch_begin, stitch_end)
      << " region_copy_ms=" << milliseconds(region_begin, region_end)
      << " m2_width_ms=" << m2_width_ms
      << " m2_space_ms=" << m2_space_ms
      << " shrink90_ms=" << milliseconds(shrink90_begin, shrink90_end)
      << " grow90_ms=" << milliseconds(grow90_begin, grow90_end)
      << " edge_extract_mode="
      << (audit_full_boundary ? "full-then-filter" : "fused-length-filter")
      << " edge_extract_ms=" << milliseconds(edges90_begin, edges90_end)
      << " long_space_ms=" << milliseconds(space_begin, space_end)
      << " shrink270_ms=" << milliseconds(shrink270_begin, shrink270_end)
      << " grow270_ms=" << milliseconds(grow270_begin, grow270_end)
      << " morphology_ms="
      << milliseconds(shrink90_begin, grow270_end)
      << " total_ms=" << milliseconds(all_begin, Clock::now()) << "\n";
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    bool audit_full_boundary = false;
    bool audit_m2_width_space = false;
    bool run_self_test = false;
    bool candidate_stream = false;
    std::string path;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--audit-full-boundary") {
        audit_full_boundary = true;
      } else if (argument == "--audit-m2-width-space") {
        audit_m2_width_space = true;
      } else if (argument == "--self-test") {
        run_self_test = true;
      } else if (argument == "--candidate" && index + 1 < argc) {
        if (!path.empty()) {
          path.clear();
          break;
        }
        candidate_stream = true;
        path = argv[++index];
      } else if (argument.rfind("--candidate=", 0) == 0) {
        if (!path.empty()) {
          path.clear();
          break;
        }
        candidate_stream = true;
        path = argument.substr(std::string("--candidate=").size());
      } else if (path.empty()) {
        path = argument;
      } else {
        path.clear();
        break;
      }
    }
    if (run_self_test) self_test();
    if (path.empty() && run_self_test) return EXIT_SUCCESS;
    if (path.empty()) {
      std::cerr << "usage: " << argv[0]
                << " [--self-test] [--audit-full-boundary] "
                   "[--audit-m2-width-space] "
                   "[--candidate GPU_BOUNDARY.km2bnd | "
                   "CPU_MERGED_M2.km1ws]\n";
      return EXIT_FAILURE;
    }
    run(path, candidate_stream, audit_full_boundary,
        audit_m2_width_space);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "M2_FLAT_REGION_BRIDGE FAIL " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
