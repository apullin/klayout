#include "m2_merged_boundary_oracle.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace oracle = klayout_cuda::m2_boundary_oracle;

struct Options
{
  std::string scene;
  std::string expected_file_sha256;
  std::string expected_scene_sha256;
  std::string expected_boundary_sha256;
  std::string candidate_producer_scene_sha256;
  std::string candidate;
  std::string write_oracle;
};

bool take(const std::string &argument, const char *prefix,
          std::string *destination)
{
  const std::string marker(prefix);
  if (argument.rfind(marker, 0) != 0) {
    return false;
  }
  *destination = argument.substr(marker.size());
  return true;
}

Options parse_options(int argc, char **argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (take(argument, "--expect-file-sha256=",
             &options.expected_file_sha256) ||
        take(argument, "--expect-scene-sha256=",
             &options.expected_scene_sha256) ||
        take(argument, "--expect-boundary-sha256=",
             &options.expected_boundary_sha256) ||
        take(argument, "--candidate-producer-scene-sha256=",
             &options.candidate_producer_scene_sha256) ||
        take(argument, "--candidate=", &options.candidate) ||
        take(argument, "--write-oracle=", &options.write_oracle)) {
      continue;
    }
    if (!argument.empty() && argument[0] == '-') {
      throw std::runtime_error("unknown option: " + argument);
    }
    if (!options.scene.empty()) {
      throw std::runtime_error("exactly one KM1WSCN1 scene is required");
    }
    options.scene = argument;
  }
  if (options.scene.empty() || options.expected_file_sha256.empty() ||
      options.expected_scene_sha256.empty()) {
    throw std::runtime_error(
        "usage: m2_merged_boundary_oracle "
        "--expect-file-sha256=HEX --expect-scene-sha256=HEX "
        "[--expect-boundary-sha256=HEX] "
        "[--candidate=FILE --candidate-producer-scene-sha256=HEX] "
        "[--write-oracle=FILE] SCENE.km1ws");
  }
  return options;
}

double elapsed_ms(Clock::time_point begin)
{
  return std::chrono::duration<double, std::milli>(
             Clock::now() - begin)
      .count();
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    const Options options = parse_options(argc, argv);
    oracle::LoadOptions load;
    load.expected_file_sha256 = options.expected_file_sha256;
    load.expected_scene_sha256 = options.expected_scene_sha256;
    load.expected_boundary_sha256 = options.expected_boundary_sha256;
    const Clock::time_point begin = Clock::now();
    const oracle::BoundaryOracle boundary =
        oracle::load_cpu_merged_boundary(options.scene, load);
    const double oracle_ms = elapsed_ms(begin);

    std::cout
        << "M2_MERGED_BOUNDARY_ORACLE verdict=COMPLETE"
        << " file_sha256=" << boundary.file_sha256
        << " scene_sha256=" << boundary.scene_sha256
        << " source_sha256=" << boundary.source_sha256
        << " boundary_sha256=" << boundary.boundary_sha256
        << " boundary_fnv64=" << boundary.boundary_fnv64
        << " stored_contexts=" << boundary.stats.stored_contexts
        << " nonempty_contexts=" << boundary.stats.nonempty_contexts
        << " stored_cells=" << boundary.stats.stored_cells
        << " stored_contours=" << boundary.stats.stored_contours
        << " stored_edges=" << boundary.stats.stored_edges
        << " flat_contours=" << boundary.stats.flat_contours
        << " flat_edges=" << boundary.stats.flat_edges
        << " segments=" << boundary.segments.size() << "\n";
    std::cout
        << "BOUNDARY_TOPOLOGY"
        << " horizontal=" << boundary.stats.horizontal_segments
        << " vertical=" << boundary.stats.vertical_segments
        << " negative_side=" << boundary.stats.negative_side_segments
        << " positive_side=" << boundary.stats.positive_side_segments
        << " adjacent_collinear=" << boundary.stats.adjacent_collinear
        << " holes=" << boundary.stats.hole_contours
        << " repeated_vertices=" << boundary.stats.repeated_vertices
        << " kissing_vertices=" << boundary.stats.kissing_vertices
        << " duplicates=" << boundary.stats.duplicate_segments
        << " opposites=" << boundary.stats.opposite_segments
        << " collinear_overlaps="
        << boundary.stats.collinear_overlaps
        << " unexpected_crossings="
        << boundary.stats.unexpected_crossings << "\n";
    std::cout
        << "MAX_CONTOUR"
        << " edges=" << boundary.stats.max_contour_edges
        << " contour=" << boundary.stats.max_contour_id
        << " context=" << boundary.stats.max_contour_context
        << " source_polygon="
        << boundary.stats.max_contour_source_polygon
        << " area_dbu2=" << boundary.stats.max_contour_area_dbu2
        << " perimeter=" << boundary.stats.max_contour_perimeter
        << " width=" << boundary.stats.max_contour_width
        << " height=" << boundary.stats.max_contour_height
        << " longest_segment=" << boundary.stats.longest_segment
        << " total_area_dbu2=" << boundary.stats.total_area_dbu2
        << " total_perimeter=" << boundary.stats.total_perimeter
        << " cache_full="
        << boundary.stats.contour_cache_full_validations
        << " cache_hits=" << boundary.stats.contour_cache_hits
        << " cache_edges=" << boundary.stats.contour_cache_edges << "\n";

    if (!options.write_oracle.empty()) {
      const Clock::time_point write_begin = Clock::now();
      oracle::write_candidate_stream(options.write_oracle, boundary);
      std::cout << std::fixed << std::setprecision(3)
                << "WRITE_ORACLE path=" << options.write_oracle
                << " ms=" << elapsed_ms(write_begin) << "\n";
    }
    if (!options.candidate.empty()) {
      const Clock::time_point compare_begin = Clock::now();
      std::vector<oracle::DirectedSegmentI64> candidate;
      if (options.candidate_producer_scene_sha256.empty()) {
        candidate =
            oracle::read_candidate_stream(options.candidate, boundary);
      } else {
        oracle::CandidateStreamIdentity identity;
        identity.producer_scene_sha256 =
            options.candidate_producer_scene_sha256;
        identity.qualification_scene_sha256 =
            boundary.scene_sha256;
        identity.boundary_sha256 = boundary.boundary_sha256;
        identity.segment_count = boundary.segments.size();
        identity.boundary_fnv64 = boundary.boundary_fnv64;
        candidate =
            oracle::read_candidate_stream(options.candidate, identity);
      }
      const oracle::Comparison comparison =
          oracle::compare_candidate(boundary, candidate);
      std::cout << "CANDIDATE_COMPARISON"
                << " equal=" << (comparison.equal ? 1 : 0)
                << " candidate_segments=" << candidate.size()
                << " first_mismatch=" << comparison.first_mismatch
                << " message=\"" << comparison.message << "\""
                << " ms=" << std::fixed << std::setprecision(3)
                << elapsed_ms(compare_begin) << "\n";
      if (!comparison.equal) {
        return 3;
      }
    }
    std::cout << std::fixed << std::setprecision(3)
              << "TIMING_MS oracle=" << oracle_ms << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M2_MERGED_BOUNDARY_ORACLE verdict=UNCERTAIN"
              << " error=\"" << error.what() << "\"\n";
    return 2;
  }
}
