#include "m2_merged_boundary_oracle.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <class Operation>
void require_rejection(const char *name, Operation operation)
{
  try {
    operation();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(std::string(name) + " was accepted");
}

}  // namespace

int main()
{
  namespace oracle = klayout_cuda::m2_boundary_oracle;
  oracle::BoundaryOracle reference;
  reference.segments = {
      {0, 0, 10, -1, oracle::SegmentAxis::horizontal},
      {10, 0, 10, 1, oracle::SegmentAxis::horizontal},
      {0, 0, 10, -1, oracle::SegmentAxis::vertical},
      {10, 0, 10, 1, oracle::SegmentAxis::vertical}};

  if (!oracle::compare_candidate(reference, reference.segments).equal) {
    std::cerr << "exact candidate did not match\n";
    return 1;
  }

  std::vector<oracle::DirectedSegmentI64> changed =
      reference.segments;
  changed[1].fixed = 11;
  if (oracle::compare_candidate(reference, changed).equal) {
    std::cerr << "changed candidate matched\n";
    return 1;
  }

  std::vector<oracle::DirectedSegmentI64> short_candidate =
      reference.segments;
  short_candidate.pop_back();
  if (oracle::compare_candidate(reference, short_candidate).equal) {
    std::cerr << "short candidate matched\n";
    return 1;
  }

  std::vector<oracle::DirectedSegmentI64> noncanonical =
      reference.segments;
  noncanonical[1] = noncanonical[0];
  if (oracle::compare_candidate(reference, noncanonical).equal) {
    std::cerr << "duplicate candidate matched\n";
    return 1;
  }

  try {
    reference.scene_sha256 = std::string(64, '1');
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("m2-candidate-stream-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".km2bnd");
    struct Cleanup
    {
      std::filesystem::path path;
      ~Cleanup()
      {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
      }
    } cleanup{path};

    oracle::write_candidate_stream(path.string(), reference);
    const auto roundtrip =
        oracle::read_candidate_stream(path.string(), reference);
    if (!oracle::compare_candidate(reference, roundtrip).equal) {
      throw std::runtime_error("candidate stream roundtrip changed payload");
    }

    oracle::BoundaryOracle wrong_scene = reference;
    wrong_scene.scene_sha256 = std::string(64, '2');
    require_rejection("wrong producer/qualification scene", [&]() {
      (void)oracle::read_candidate_stream(path.string(), wrong_scene);
    });

    {
      std::fstream corrupt(
          path, std::ios::binary | std::ios::in | std::ios::out);
      corrupt.seekg(160);
      char byte = 0;
      corrupt.read(&byte, 1);
      byte ^= 1;
      corrupt.seekp(160);
      corrupt.write(&byte, 1);
    }
    require_rejection("corrupt payload", [&]() {
      (void)oracle::read_candidate_stream(path.string(), reference);
    });

    oracle::write_candidate_stream(path.string(), reference);
    std::filesystem::resize_file(
        path, std::filesystem::file_size(path) - 1);
    require_rejection("truncated payload", [&]() {
      (void)oracle::read_candidate_stream(path.string(), reference);
    });

    oracle::BoundaryOracle bad_order = reference;
    std::swap(bad_order.segments[0], bad_order.segments[1]);
    require_rejection("noncanonical output", [&]() {
      oracle::write_candidate_stream(path.string(), bad_order);
    });

    oracle::BoundaryOracle nonmaximal = reference;
    nonmaximal.segments[0].hi = 5;
    nonmaximal.segments.insert(
        nonmaximal.segments.begin() + 1,
        {0, 5, 10, -1, oracle::SegmentAxis::horizontal});
    require_rejection("nonmaximal output", [&]() {
      oracle::write_candidate_stream(path.string(), nonmaximal);
    });
  } catch (const std::exception &error) {
    std::cerr << "candidate stream test failed: " << error.what() << "\n";
    return 1;
  }

  std::cout
      << "M2 merged-boundary oracle comparator/stream: 10 tests passed\n";
  return 0;
}
