/*
 * Independent exact boundary oracle for a qualified CPU-merged M2
 * KM1WSCN1 scene.
 */

#ifndef KLAYOUT_CUDA_M2_MERGED_BOUNDARY_ORACLE_H
#define KLAYOUT_CUDA_M2_MERGED_BOUNDARY_ORACLE_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace klayout_cuda {
namespace m2_boundary_oracle {

enum class SegmentAxis : std::uint32_t
{
  horizontal = 0,
  vertical = 1,
  invalid = 2
};

// Byte-compatible with manhattan_union_format.cuh::DirectedSegmentI64.
struct DirectedSegmentI64
{
  std::int64_t fixed;
  std::int64_t lo;
  std::int64_t hi;
  std::int32_t side;
  SegmentAxis axis;
};

static_assert(std::is_trivially_copyable<DirectedSegmentI64>::value,
              "boundary segment must remain POD");
static_assert(sizeof(DirectedSegmentI64) == 32,
              "unexpected boundary-segment ABI padding");

struct LoadOptions
{
  // Both independent allowlist values are mandatory.
  std::string expected_file_sha256;
  std::string expected_scene_sha256;
  std::string expected_boundary_sha256;
  std::uint64_t max_contexts = UINT64_C(1000000);
  std::uint64_t max_stored_polygons = UINT64_C(32000000);
  std::uint64_t max_stored_edges = UINT64_C(64000000);
  std::uint64_t max_flat_polygons = UINT64_C(32000000);
  std::uint64_t max_flat_edges = UINT64_C(64000000);
};

struct BoundaryStats
{
  std::uint64_t stored_contexts = 0;
  std::uint64_t nonempty_contexts = 0;
  std::uint64_t stored_cells = 0;
  std::uint64_t stored_contours = 0;
  std::uint64_t stored_edges = 0;
  std::uint64_t flat_contours = 0;
  std::uint64_t flat_edges = 0;
  std::uint64_t horizontal_segments = 0;
  std::uint64_t vertical_segments = 0;
  std::uint64_t negative_side_segments = 0;
  std::uint64_t positive_side_segments = 0;
  std::uint64_t max_contour_edges = 0;
  std::uint64_t max_contour_id = 0;
  std::uint64_t max_contour_context = 0;
  std::uint64_t max_contour_source_polygon = 0;
  std::uint64_t max_contour_perimeter = 0;
  std::uint64_t longest_segment = 0;
  std::int64_t max_contour_width = 0;
  std::int64_t max_contour_height = 0;
  std::string max_contour_area_dbu2;
  std::string total_area_dbu2;
  std::uint64_t total_perimeter = 0;
  std::uint64_t adjacent_collinear = 0;
  std::uint64_t hole_contours = 0;
  std::uint64_t repeated_vertices = 0;
  std::uint64_t kissing_vertices = 0;
  std::uint64_t duplicate_segments = 0;
  std::uint64_t opposite_segments = 0;
  std::uint64_t collinear_overlaps = 0;
  std::uint64_t unexpected_crossings = 0;
  std::uint64_t contour_cache_full_validations = 0;
  std::uint64_t contour_cache_hits = 0;
  std::uint64_t contour_cache_edges = 0;
};

struct BoundaryOracle
{
  std::vector<DirectedSegmentI64> segments;
  BoundaryStats stats;
  std::string file_sha256;
  std::string scene_sha256;
  std::string source_sha256;
  std::string boundary_sha256;
  std::uint64_t boundary_fnv64 = 0;
};

struct Comparison
{
  bool equal = false;
  std::uint64_t first_mismatch = 0;
  std::string message;
};

BoundaryOracle load_cpu_merged_boundary(const std::string &path,
                                        const LoadOptions &options);

Comparison compare_candidate(
    const BoundaryOracle &oracle,
    const std::vector<DirectedSegmentI64> &candidate);

// Digests use the checked little-endian 32-byte candidate record encoding,
// never the host ABI representation.
std::string canonical_boundary_sha256(
    const std::vector<DirectedSegmentI64> &segments);

std::uint64_t canonical_boundary_fnv64(
    const std::vector<DirectedSegmentI64> &segments);

// Candidate stream is a checked little-endian 128-byte header followed by
// canonical 32-byte DirectedSegmentI64 records.
std::vector<DirectedSegmentI64> read_candidate_stream(
    const std::string &path, const BoundaryOracle &oracle);

void write_candidate_stream(const std::string &path,
                            const BoundaryOracle &oracle);

}  // namespace m2_boundary_oracle
}  // namespace klayout_cuda

#endif
