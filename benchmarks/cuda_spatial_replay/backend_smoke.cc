#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

std::uint64_t pair_key(std::uint32_t subject, std::uint32_t intruder) {
  return (static_cast<std::uint64_t>(subject) << 32) | intruder;
}

bool boxes_overlap(const klayout_cuda_spatial_aabb_v1 &a,
                   const klayout_cuda_spatial_aabb_v1 &b,
                   std::int64_t enlargement) {
  // Use a wider intermediate so the oracle itself cannot overflow if this
  // fixture is later extended toward the int64 coordinate limits.
  const __int128 e = static_cast<__int128>(enlargement);
  return static_cast<__int128>(a.left) <
             static_cast<__int128>(b.right) + e &&
         static_cast<__int128>(b.left) <
             static_cast<__int128>(a.right) + e &&
         static_cast<__int128>(a.bottom) <
             static_cast<__int128>(b.top) + e &&
         static_cast<__int128>(b.bottom) <
             static_cast<__int128>(a.top) + e;
}

klayout_cuda_spatial_config_v1 make_config(std::uint64_t cell_size) {
  klayout_cuda_spatial_config_v1 config{};
  config.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  config.struct_size = sizeof(config);
  config.device = 0;
  config.max_cells_per_record = 64;
  config.max_records_per_cell = 512;
  config.cell_size = cell_size;
  config.max_memberships = 1000000;
  config.max_pair_work = 10000000;
  config.max_candidates = 2000000;
  return config;
}

bool valid_result(const klayout_cuda_spatial_result_v1 &result, int status) {
  return status == KLAYOUT_CUDA_SPATIAL_OK &&
         result.status == KLAYOUT_CUDA_SPATIAL_OK &&
         result.abi_version == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
         result.struct_size >= sizeof(result) &&
         (result.pair_count == 0 || result.pair_keys != nullptr);
}

void report_failure(const char *name, int status,
                    const klayout_cuda_spatial_result_v1 &result) {
  std::cerr << name << " failed: call_status=" << status
            << " result_status=" << result.status
            << " fallback_flags=" << result.fallback_flags
            << " pair_count=" << result.pair_count
            << " message=" << result.message << '\n';
}

bool run_two_pair_smoke() {
  const klayout_cuda_spatial_aabb_v1 subjects[] = {
      {0, 0, 10, 10},
      {100, 100, 110, 110},
  };
  const klayout_cuda_spatial_aabb_v1 intruders[] = {
      {5, 5, 15, 15},
      {105, 105, 115, 115},
      {1000, 1000, 1010, 1010},
  };

  auto config = make_config(32);
  klayout_cuda_spatial_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.subjects = subjects;
  request.subject_count = 2;
  request.intruders = intruders;
  request.intruder_count = 3;
  request.enlargement = 0;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result{};
  const int status = klayout_cuda_spatial_run_bipartite_v1(&request, &result);
  const bool good =
      valid_result(result, status) && result.pair_count == 2 &&
      result.pair_keys[0] == pair_key(1, 3) &&
      result.pair_keys[1] == pair_key(2, 4);
  if (!good) {
    report_failure("CUDA backend two-pair ABI smoke", status, result);
  } else {
    std::cout << "CUDA backend ABI smoke passed: pairs=(1,3),(2,4)"
              << " total_ms=" << (double(result.total_ns) / 1.0e6) << '\n';
  }
  klayout_cuda_spatial_release_result_v1(&result);
  return good;
}

std::uint64_t next_random(std::uint64_t &state) {
  // A fixed xorshift64* stream keeps this gate deterministic without pulling
  // in a library RNG whose sequence can vary across standard libraries.
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * UINT64_C(2685821657736338717);
}

std::vector<klayout_cuda_spatial_aabb_v1> make_boxes(std::size_t count,
                                                     std::uint64_t seed) {
  std::vector<klayout_cuda_spatial_aabb_v1> boxes;
  boxes.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::int64_t left =
        static_cast<std::int64_t>(next_random(seed) % 4096) - 2048;
    const std::int64_t bottom =
        static_cast<std::int64_t>(next_random(seed) % 4096) - 2048;
    const std::int64_t width =
        1 + static_cast<std::int64_t>(next_random(seed) % 127);
    const std::int64_t height =
        1 + static_cast<std::int64_t>(next_random(seed) % 127);
    boxes.push_back({left, bottom, left + width, bottom + height});
  }
  return boxes;
}

bool run_exact_oracle_gate() {
  constexpr std::size_t kSubjectCount = 1024;
  constexpr std::size_t kIntruderCount = 1024;
  constexpr std::int64_t kEnlargement = 7;
  auto subjects =
      make_boxes(kSubjectCount, UINT64_C(0x6a09e667f3bcc909));
  auto intruders =
      make_boxes(kIntruderCount, UINT64_C(0xbb67ae8584caa73b));

  // Pin explicit cases around signed cell boundaries.  The first intruder has
  // a gap of enlargement-1 and must match; the second has a gap exactly equal
  // to enlargement and must not.  The third crosses zero and a cell boundary.
  subjects[0] = {-64, -64, -32, -32};
  intruders[0] = {-26, -60, -10, -40};
  intruders[1] = {-25, -60, -9, -40};
  subjects[1] = {-1, -1, 64, 64};
  intruders[2] = {64, 0, 96, 32};

  std::vector<std::uint64_t> expected;
  for (std::size_t i = 0; i < subjects.size(); ++i) {
    for (std::size_t j = 0; j < intruders.size(); ++j) {
      if (boxes_overlap(subjects[i], intruders[j], kEnlargement)) {
        expected.push_back(pair_key(
            static_cast<std::uint32_t>(i + 1),
            static_cast<std::uint32_t>(subjects.size() + j + 1)));
      }
    }
  }

  auto config = make_config(64);
  klayout_cuda_spatial_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.subjects = subjects.data();
  request.subject_count = subjects.size();
  request.intruders = intruders.data();
  request.intruder_count = intruders.size();
  request.enlargement = kEnlargement;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result{};
  const int status = klayout_cuda_spatial_run_bipartite_v1(&request, &result);
  bool good = valid_result(result, status) &&
              result.pair_count == expected.size();
  if (good) {
    good = std::equal(expected.begin(), expected.end(), result.pair_keys);
  }

  if (!good) {
    report_failure("CUDA backend exact CPU-oracle gate", status, result);
    if (valid_result(result, status)) {
      const std::size_t actual_size =
          static_cast<std::size_t>(result.pair_count);
      const std::size_t common = std::min(expected.size(), actual_size);
      std::size_t mismatch = 0;
      while (mismatch < common &&
             expected[mismatch] == result.pair_keys[mismatch]) {
        ++mismatch;
      }
      if (mismatch < common) {
        std::cerr << "first mismatch at " << mismatch
                  << ": expected=" << expected[mismatch]
                  << " actual=" << result.pair_keys[mismatch] << '\n';
      } else {
        std::cerr << "pair vector size mismatch: expected=" << expected.size()
                  << " actual=" << actual_size << '\n';
      }
    }
  } else {
    std::cout << "CUDA backend exact CPU-oracle gate passed: checks="
              << (subjects.size() * intruders.size())
              << " exact_pairs=" << expected.size()
              << " memberships=" << result.membership_count
              << " pair_work=" << result.pair_work_count
              << " total_ms=" << (double(result.total_ns) / 1.0e6) << '\n';
  }
  klayout_cuda_spatial_release_result_v1(&result);
  return good;
}

}  // namespace

int main() {
  if (klayout_cuda_spatial_abi_version() !=
      KLAYOUT_CUDA_SPATIAL_ABI_VERSION) {
    std::cerr << "CUDA backend ABI version function returned an incompatible "
                 "version\n";
    return 1;
  }
  const bool smoke_good = run_two_pair_smoke();
  const bool oracle_good = smoke_good && run_exact_oracle_gate();
  return oracle_good ? 0 : 1;
}
