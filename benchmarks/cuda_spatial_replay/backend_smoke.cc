#include "dbCudaSpatialApi.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

bool valid_self_pairs(const klayout_cuda_spatial_result_v1 &result,
                      std::size_t record_count) {
  std::uint64_t previous = 0;
  for (std::uint64_t i = 0; i < result.pair_count; ++i) {
    const std::uint64_t key = result.pair_keys[i];
    const std::uint64_t first = key >> 32;
    const std::uint64_t second = key & UINT64_C(0xffffffff);
    if ((i != 0 && key <= previous) || first == 0 || first >= second ||
        second > record_count) {
      return false;
    }
    previous = key;
  }
  return true;
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

bool run_self_boundary_smoke() {
  constexpr std::int64_t kEnlargement = 7;
  const klayout_cuda_spatial_aabb_v1 records[] = {
      {0, 0, 10, 10},
      {16, 0, 26, 10},    // gap enlargement-1: included
      {0, 100, 10, 110},
      {17, 100, 27, 110}  // gap exactly enlargement: excluded
  };
  const std::uint64_t expected = pair_key(1, 2);

  auto config = make_config(32);
  klayout_cuda_spatial_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.subjects = records;
  request.subject_count = sizeof(records) / sizeof(records[0]);
  request.enlargement = kEnlargement;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result{};
  const int status = klayout_cuda_spatial_run_self_v1(&request, &result);
  const bool good = valid_result(result, status) &&
                    valid_self_pairs(result, request.subject_count) &&
                    result.pair_count == 1 && result.pair_keys[0] == expected;
  if (!good) {
    report_failure("CUDA backend self strict-boundary smoke", status, result);
  } else {
    std::cout << "CUDA backend self strict-boundary smoke passed: "
                 "unordered_pair=(1,2) total_ms="
              << (double(result.total_ns) / 1.0e6) << '\n';
  }
  klayout_cuda_spatial_release_result_v1(&result);
  return good && result.pair_keys == nullptr && result.pair_count == 0;
}

bool run_self_complete_cell_smoke() {
  const klayout_cuda_spatial_aabb_v1 records[] = {
      {0, 0, 10, 10},
      {0, 0, 10, 10},
      {0, 0, 10, 10},
      {0, 0, 10, 10},
      {0, 0, 10, 10},
  };
  std::vector<std::uint64_t> expected;
  for (std::uint32_t first = 1; first <= 5; ++first) {
    for (std::uint32_t second = first + 1; second <= 5; ++second) {
      expected.push_back(pair_key(first, second));
    }
  }

  auto config = make_config(64);
  klayout_cuda_spatial_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.subjects = records;
  request.subject_count = sizeof(records) / sizeof(records[0]);
  request.enlargement = 0;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result{};
  const int status = klayout_cuda_spatial_run_self_v1(&request, &result);
  bool good = valid_result(result, status) &&
              valid_self_pairs(result, request.subject_count) &&
              result.pair_work_count == expected.size() &&
              result.pair_count == expected.size();
  if (good) {
    good = std::equal(expected.begin(), expected.end(), result.pair_keys);
  }
  if (!good) {
    report_failure("CUDA backend self complete-cell smoke", status, result);
  } else {
    std::cout << "CUDA backend self complete-cell smoke passed: records=5 "
                 "unordered_pairs=10\n";
  }
  klayout_cuda_spatial_release_result_v1(&result);
  return good;
}

bool run_self_exact_oracle_gate() {
  constexpr std::size_t kRecordCount = 1024;
  constexpr std::int64_t kEnlargement = 7;
  auto records = make_boxes(kRecordCount, UINT64_C(0x3c6ef372fe94f82b));

  records[0] = {-64, -64, -32, -32};
  records[1] = {-26, -60, -10, -40};  // gap 6: included
  records[2] = {10000, 0, 10032, 32};
  records[3] = {10039, 0, 10055, 16};  // gap 7: excluded

  std::vector<std::uint64_t> expected;
  for (std::size_t i = 0; i < records.size(); ++i) {
    for (std::size_t j = i + 1; j < records.size(); ++j) {
      if (boxes_overlap(records[i], records[j], kEnlargement)) {
        expected.push_back(pair_key(static_cast<std::uint32_t>(i + 1),
                                    static_cast<std::uint32_t>(j + 1)));
      }
    }
  }
  const bool pinned_boundaries =
      std::binary_search(expected.begin(), expected.end(), pair_key(1, 2)) &&
      !std::binary_search(expected.begin(), expected.end(), pair_key(3, 4));

  auto config = make_config(64);
  klayout_cuda_spatial_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.subjects = records.data();
  request.subject_count = records.size();
  request.enlargement = kEnlargement;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result{};
  const int status = klayout_cuda_spatial_run_self_v1(&request, &result);
  bool good = pinned_boundaries && valid_result(result, status) &&
              valid_self_pairs(result, records.size()) &&
              result.pair_count == expected.size();
  if (good) {
    good = std::equal(expected.begin(), expected.end(), result.pair_keys);
  }
  if (!good) {
    report_failure("CUDA backend self exact CPU-oracle gate", status, result);
  } else {
    std::cout << "CUDA backend self exact CPU-oracle gate passed: checks="
              << (records.size() * (records.size() - 1) / 2)
              << " exact_pairs=" << expected.size()
              << " memberships=" << result.membership_count
              << " pair_work=" << result.pair_work_count
              << " total_ms=" << (double(result.total_ns) / 1.0e6) << '\n';
  }
  klayout_cuda_spatial_release_result_v1(&result);
  return good;
}

bool run_self_fail_closed_gate() {
  const klayout_cuda_spatial_aabb_v1 records[] = {
      {0, 0, 10, 10},
      {1, 1, 11, 11},
      {2, 2, 12, 12},
  };
  auto config = make_config(64);
  klayout_cuda_spatial_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.subjects = records;
  request.subject_count = sizeof(records) / sizeof(records[0]);
  request.enlargement = 0;
  request.config = &config;

  // Self mode has exactly one input array.  Supplying the bipartite fields is
  // malformed and must never reach a device pipeline.
  request.intruders = records;
  request.intruder_count = 1;
  klayout_cuda_spatial_result_v1 malformed{};
  const int malformed_status =
      klayout_cuda_spatial_run_self_v1(&request, &malformed);
  bool good = malformed_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
              malformed.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
              (malformed.fallback_flags &
               KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST) != 0 &&
              malformed.pair_count == 0 && malformed.pair_keys == nullptr;
  if (!good) {
    report_failure("CUDA backend self malformed-request gate",
                   malformed_status, malformed);
  }
  klayout_cuda_spatial_release_result_v1(&malformed);

  // A matching ABI with a truncated config prefix must be rejected before the
  // backend consults any fields beyond struct_size.
  request.intruders = nullptr;
  request.intruder_count = 0;
  auto short_config = config;
  short_config.struct_size = 2 * sizeof(std::uint32_t);
  request.config = &short_config;
  klayout_cuda_spatial_result_v1 short_config_result{};
  const int short_config_status =
      klayout_cuda_spatial_run_self_v1(&request, &short_config_result);
  const bool short_config_good =
      short_config_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      short_config_result.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      (short_config_result.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST) != 0 &&
      short_config_result.pair_count == 0 &&
      short_config_result.pair_keys == nullptr;
  if (!short_config_good) {
    report_failure("CUDA backend self short-config gate",
                   short_config_status, short_config_result);
  }
  good = good && short_config_good;
  klayout_cuda_spatial_release_result_v1(&short_config_result);
  request.config = &config;

  // A non-normalized box is a data-dependent CPU fallback, not partial output.
  const klayout_cuda_spatial_aabb_v1 invalid_records[] = {
      {10, 0, 0, 10},
      {0, 0, 10, 10},
  };
  request.subjects = invalid_records;
  request.subject_count = sizeof(invalid_records) / sizeof(invalid_records[0]);
  request.intruders = nullptr;
  request.intruder_count = 0;
  klayout_cuda_spatial_result_v1 invalid{};
  const int invalid_status =
      klayout_cuda_spatial_run_self_v1(&request, &invalid);
  const bool invalid_good =
      invalid_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      invalid.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      (invalid.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW) != 0 &&
      invalid.pair_count == 0 && invalid.pair_keys == nullptr;
  if (!invalid_good) {
    report_failure("CUDA backend self invalid-AABB gate", invalid_status,
                   invalid);
  }
  good = good && invalid_good;
  klayout_cuda_spatial_release_result_v1(&invalid);

  // Three records in one cell require three unordered comparisons.
  request.subjects = records;
  request.subject_count = sizeof(records) / sizeof(records[0]);
  config.max_pair_work = 1;
  klayout_cuda_spatial_result_v1 capacity{};
  const int capacity_status =
      klayout_cuda_spatial_run_self_v1(&request, &capacity);
  const bool capacity_good =
      capacity_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      (capacity.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY) != 0 &&
      capacity.pair_count == 0 && capacity.pair_keys == nullptr;
  if (!capacity_good) {
    report_failure("CUDA backend self capacity gate", capacity_status,
                   capacity);
  }
  good = good && capacity_good;
  klayout_cuda_spatial_release_result_v1(&capacity);

  if (good) {
    std::cout << "CUDA backend self fail-closed gates passed: "
                 "malformed, short-config, invalid-AABB, "
                 "pair-work-capacity\n";
  }
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
  bool good = true;
  good = run_two_pair_smoke() && good;
  good = run_exact_oracle_gate() && good;
  good = run_self_boundary_smoke() && good;
  good = run_self_complete_cell_smoke() && good;
  good = run_self_exact_oracle_gate() && good;
  good = run_self_fail_closed_gate() && good;
  return good ? 0 : 1;
}
