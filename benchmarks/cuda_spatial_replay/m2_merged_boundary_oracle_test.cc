#include "m2_merged_boundary_oracle.h"

#include <iostream>
#include <vector>

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

  std::cout << "M2 merged-boundary oracle comparator: 4 tests passed\n";
  return 0;
}
