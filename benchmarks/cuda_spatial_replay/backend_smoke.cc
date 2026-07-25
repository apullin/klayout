#include "dbCudaSpatialApi.h"
#include "dbCudaActive3Digest.h"
#include "dbCudaImplant12Digest.h"

#include <algorithm>
#include <array>
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

bool valid_m1_result(const klayout_cuda_spatial_m1_result_v1 &result,
                     int status) {
  if (status != KLAYOUT_CUDA_SPATIAL_OK ||
      result.status != KLAYOUT_CUDA_SPATIAL_OK ||
      result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size < sizeof(result) ||
      (result.survivor_count != 0 && !result.survivors)) {
    return false;
  }
  std::uint64_t previous = 0;
  for (std::uint64_t index = 0; index < result.survivor_count; ++index) {
    const auto &survivor = result.survivors[index];
    if (survivor.contact_id == 0 ||
        (index != 0 && survivor.contact_id <= previous) ||
        (survivor.deficient_side_mask & ~0xfu) != 0 ||
        (survivor.flags &
         ~(KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_UNCERTAIN |
           KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_DISALLOWED_MASK)) != 0) {
      return false;
    }
    previous = survivor.contact_id;
  }
  return true;
}

void report_m1_failure(const char *name, int status,
                       const klayout_cuda_spatial_m1_result_v1 &result) {
  std::cerr << name << " failed: call_status=" << status
            << " result_status=" << result.status
            << " disposition=" << result.disposition
            << " fallback_flags=" << result.fallback_flags
            << " survivors=" << result.survivor_count
            << " uncertain=" << result.uncertain_contact_count
            << " disallowed=" << result.disallowed_contact_count
            << " full_hits=" << result.full_side_hit_count
            << " partials=" << result.partial_candidate_count
            << " non_manhattan="
            << result.non_manhattan_candidate_count
            << " message=" << result.message << '\n';
}

int run_m1_request(
    const std::vector<klayout_cuda_spatial_m1_contact_v1> &contacts,
    const std::vector<klayout_cuda_spatial_m1_edge_v1> &edges,
    klayout_cuda_spatial_config_v1 &config,
    klayout_cuda_spatial_m1_result_v1 &result,
    std::int64_t distance = 35) {
  klayout_cuda_spatial_m1_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_M1_ENCLOSED_PROJECTION_ONE_OR_OPPOSITE;
  request.contacts = contacts.data();
  request.contact_count = contacts.size();
  request.metal1_edges = edges.empty() ? nullptr : edges.data();
  request.metal1_edge_count = edges.size();
  request.distance = distance;
  request.config = &config;
  return klayout_cuda_spatial_run_m1_enclosure_v1(&request, &result);
}

klayout_cuda_spatial_m1_contact_v1 make_m1_contact(
    std::int64_t left, std::int64_t bottom, std::uint64_t id,
    std::uint32_t context, std::int64_t size = 65) {
  return {left, bottom, left + size, bottom + size, id, context, 0};
}

void add_m1_mask_edges(
    std::vector<klayout_cuda_spatial_m1_edge_v1> &edges,
    const klayout_cuda_spatial_m1_contact_v1 &contact, std::uint32_t mask,
    std::int64_t gap = 34) {
  if (mask & 0x1u) {
    edges.push_back({contact.left - gap, contact.bottom, contact.left - gap,
                     contact.top, contact.context_id, 0});
  }
  if (mask & 0x2u) {
    edges.push_back({contact.left, contact.top + gap, contact.right,
                     contact.top + gap, contact.context_id, 0});
  }
  if (mask & 0x4u) {
    edges.push_back({contact.right + gap, contact.top, contact.right + gap,
                     contact.bottom, contact.context_id, 0});
  }
  if (mask & 0x8u) {
    edges.push_back({contact.right, contact.bottom - gap, contact.left,
                     contact.bottom - gap, contact.context_id, 0});
  }
}

bool m1_mask_is_expected_waivable(std::uint32_t mask) {
  return mask == 0 || (mask != 0 && (mask & (mask - 1)) == 0) ||
         mask == 0x5u || mask == 0xau;
}

bool run_m1_all_masks_gate() {
  auto config = make_config(128);
  bool good = true;
  for (std::uint32_t mask = 0; mask < 16; ++mask) {
    std::vector<klayout_cuda_spatial_m1_contact_v1> contacts{
        make_m1_contact(-32, -16, UINT64_C(0x100000000) + mask, 7)};
    std::vector<klayout_cuda_spatial_m1_edge_v1> edges;
    add_m1_mask_edges(edges, contacts.front(), mask);

    klayout_cuda_spatial_m1_result_v1 result{};
    const int status = run_m1_request(contacts, edges, config, result);
    const bool waivable = m1_mask_is_expected_waivable(mask);
    bool case_good =
        valid_m1_result(result, status) &&
        result.full_side_hit_count ==
            static_cast<std::uint64_t>(__builtin_popcount(mask)) &&
        result.partial_candidate_count == 0 &&
        result.non_manhattan_candidate_count == 0 &&
        result.uncertain_contact_count == 0 &&
        result.disallowed_contact_count == (waivable ? 0u : 1u);
    if (waivable) {
      case_good =
          case_good &&
          result.disposition == KLAYOUT_CUDA_SPATIAL_M1_COMPLETE &&
          result.survivor_count == 0;
    } else {
      case_good =
          case_good &&
          result.disposition == KLAYOUT_CUDA_SPATIAL_M1_DISALLOWED &&
          result.survivor_count == 1 &&
          result.survivors[0].contact_id == contacts[0].contact_id &&
          result.survivors[0].deficient_side_mask == mask &&
          result.survivors[0].flags ==
              KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_DISALLOWED_MASK;
    }
    if (!case_good) {
      report_m1_failure("CUDA M1 exhaustive mask oracle", status, result);
      std::cerr << "  mask=0x" << std::hex << mask << std::dec
                << " expected_waivable=" << waivable << '\n';
    }
    good = case_good && good;
    klayout_cuda_spatial_release_m1_result_v1(&result);
  }
  if (good) {
    std::cout << "CUDA M1 certificate exhaustive mask oracle passed: "
                 "all 16 masks, accepted 0/singleton/0x5/0xa\n";
  }
  return good;
}

bool run_m1_boundary_context_uncertainty_gate() {
  auto config = make_config(128);
  bool good = true;

  // Strict threshold: 34 is deficient, while 35 and 36 are not.
  std::vector<klayout_cuda_spatial_m1_contact_v1> threshold_contacts{
      make_m1_contact(0, 0, 700, 11)};
  std::vector<klayout_cuda_spatial_m1_edge_v1> threshold_edges;
  const auto &c = threshold_contacts.front();
  threshold_edges.push_back(
      {c.left - 34, c.bottom, c.left - 34, c.top, c.context_id, 0});
  threshold_edges.push_back(
      {c.left, c.top + 35, c.right, c.top + 35, c.context_id, 0});
  threshold_edges.push_back(
      {c.right + 36, c.top, c.right + 36, c.bottom, c.context_id, 0});
  klayout_cuda_spatial_m1_result_v1 threshold{};
  const int threshold_status =
      run_m1_request(threshold_contacts, threshold_edges, config, threshold);
  const bool threshold_good =
      valid_m1_result(threshold, threshold_status) &&
      threshold.disposition == KLAYOUT_CUDA_SPATIAL_M1_COMPLETE &&
      threshold.full_side_hit_count == 1 && threshold.survivor_count == 0;
  if (!threshold_good) {
    report_m1_failure("CUDA M1 34/35/36 threshold gate", threshold_status,
                      threshold);
  }
  good = threshold_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&threshold);

  // Three disallowed masks arrive in intentionally non-sorted stable-ID order.
  // The DSO must return only the compact stable-ID survivor list.
  std::vector<klayout_cuda_spatial_m1_contact_v1> sorted_contacts{
      make_m1_contact(0, 0, 900, 21),
      make_m1_contact(1000, 0, 100, 22),
      make_m1_contact(2000, 0, 500, 23)};
  std::vector<klayout_cuda_spatial_m1_edge_v1> sorted_edges;
  add_m1_mask_edges(sorted_edges, sorted_contacts[0], 0x3);
  add_m1_mask_edges(sorted_edges, sorted_contacts[1], 0x7);
  add_m1_mask_edges(sorted_edges, sorted_contacts[2], 0xf);
  klayout_cuda_spatial_m1_result_v1 sorted{};
  const int sorted_status =
      run_m1_request(sorted_contacts, sorted_edges, config, sorted);
  const bool sorted_good =
      valid_m1_result(sorted, sorted_status) &&
      sorted.disposition == KLAYOUT_CUDA_SPATIAL_M1_DISALLOWED &&
      sorted.survivor_count == 3 && sorted.survivors[0].contact_id == 100 &&
      sorted.survivors[0].deficient_side_mask == 0x7 &&
      sorted.survivors[1].contact_id == 500 &&
      sorted.survivors[1].deficient_side_mask == 0xf &&
      sorted.survivors[2].contact_id == 900 &&
      sorted.survivors[2].deficient_side_mask == 0x3;
  if (!sorted_good) {
    report_m1_failure("CUDA M1 sorted survivor gate", sorted_status, sorted);
  }
  good = sorted_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&sorted);

  // A wrong-context disallowed mask must not interact with this contact.
  std::vector<klayout_cuda_spatial_m1_contact_v1> context_contacts{
      make_m1_contact(-100, -100, 701, 31)};
  auto wrong_context = context_contacts.front();
  wrong_context.context_id = 32;
  std::vector<klayout_cuda_spatial_m1_edge_v1> context_edges;
  add_m1_mask_edges(context_edges, wrong_context, 0xf);
  klayout_cuda_spatial_m1_result_v1 context{};
  const int context_status =
      run_m1_request(context_contacts, context_edges, config, context);
  const bool context_good =
      valid_m1_result(context, context_status) &&
      context.disposition == KLAYOUT_CUDA_SPATIAL_M1_COMPLETE &&
      context.full_side_hit_count == 0 && context.survivor_count == 0;
  if (!context_good) {
    report_m1_failure("CUDA M1 hierarchy-context gate", context_status,
                      context);
  }
  good = context_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&context);

  // Whole top plus a partial left projection keeps the contact as uncertain.
  std::vector<klayout_cuda_spatial_m1_contact_v1> uncertain_contacts{
      make_m1_contact(0, 0, 702, 41)};
  std::vector<klayout_cuda_spatial_m1_edge_v1> uncertain_edges;
  add_m1_mask_edges(uncertain_edges, uncertain_contacts[0], 0x2);
  uncertain_edges.push_back(
      {-34, 10, -34, 55, uncertain_contacts[0].context_id, 0});
  uncertain_edges.push_back(
      {-20, 0, 20, 65, uncertain_contacts[0].context_id, 0});
  klayout_cuda_spatial_m1_result_v1 uncertain{};
  const int uncertain_status =
      run_m1_request(uncertain_contacts, uncertain_edges, config, uncertain);
  const bool uncertain_good =
      valid_m1_result(uncertain, uncertain_status) &&
      uncertain.disposition == KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN &&
      uncertain.survivor_count == 1 &&
      uncertain.survivors[0].contact_id == 702 &&
      uncertain.survivors[0].deficient_side_mask == 0x2 &&
      (uncertain.survivors[0].flags &
       KLAYOUT_CUDA_SPATIAL_M1_SURVIVOR_UNCERTAIN) != 0 &&
      uncertain.partial_candidate_count == 1 &&
      uncertain.non_manhattan_candidate_count == 1 &&
      uncertain.uncertain_contact_count == 1;
  if (!uncertain_good) {
    report_m1_failure("CUDA M1 partial/non-Manhattan uncertainty gate",
                      uncertain_status, uncertain);
  }
  good = uncertain_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&uncertain);

  if (good) {
    std::cout << "CUDA M1 threshold/context/survivor/uncertainty gates "
                 "passed\n";
  }
  return good;
}

bool run_m1_extrema_and_fail_closed_gate() {
  auto config = make_config(128);
  bool good = true;

  const std::int64_t hi = std::numeric_limits<std::int64_t>::max() - 256;
  const std::int64_t lo = std::numeric_limits<std::int64_t>::min() + 128;
  std::vector<klayout_cuda_spatial_m1_contact_v1> extrema_contacts{
      make_m1_contact(hi - 65, hi - 65, 0xfeed, 51),
      make_m1_contact(lo, lo, 0xbeef, 52)};
  std::vector<klayout_cuda_spatial_m1_edge_v1> extrema_edges;
  add_m1_mask_edges(extrema_edges, extrema_contacts[0], 0x5);
  add_m1_mask_edges(extrema_edges, extrema_contacts[1], 0x3);
  klayout_cuda_spatial_m1_result_v1 extrema{};
  const int extrema_status =
      run_m1_request(extrema_contacts, extrema_edges, config, extrema);
  const bool extrema_good =
      valid_m1_result(extrema, extrema_status) &&
      extrema.disposition == KLAYOUT_CUDA_SPATIAL_M1_DISALLOWED &&
      extrema.survivor_count == 1 &&
      extrema.survivors[0].contact_id == 0xbeef &&
      extrema.survivors[0].deficient_side_mask == 0x3;
  if (!extrema_good) {
    report_m1_failure("CUDA M1 signed-coordinate extrema/rotation gate",
                      extrema_status, extrema);
  }
  good = extrema_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&extrema);
  const bool release_good =
      extrema.survivors == nullptr && extrema.survivor_count == 0;
  good = release_good && good;

  std::vector<klayout_cuda_spatial_m1_contact_v1> malformed_contacts{
      make_m1_contact(0, 0, 1, 61)};
  malformed_contacts[0].right = malformed_contacts[0].left;
  std::vector<klayout_cuda_spatial_m1_edge_v1> no_edges;
  klayout_cuda_spatial_m1_result_v1 malformed{};
  const int malformed_status =
      run_m1_request(malformed_contacts, no_edges, config, malformed);
  const bool malformed_good =
      malformed_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      malformed.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      malformed.disposition == KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN &&
      (malformed.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW) != 0 &&
      malformed.survivors == nullptr && malformed.survivor_count == 0;
  if (!malformed_good) {
    report_m1_failure("CUDA M1 malformed-contact gate", malformed_status,
                      malformed);
  }
  good = malformed_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&malformed);

  std::vector<klayout_cuda_spatial_m1_contact_v1> capacity_contacts{
      make_m1_contact(0, 0, 2, 62)};
  std::vector<klayout_cuda_spatial_m1_edge_v1> capacity_edges;
  add_m1_mask_edges(capacity_edges, capacity_contacts[0], 0xf);
  auto capacity_config = config;
  capacity_config.max_pair_work = 1;
  klayout_cuda_spatial_m1_result_v1 capacity{};
  const int capacity_status = run_m1_request(
      capacity_contacts, capacity_edges, capacity_config, capacity);
  const bool capacity_good =
      capacity_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity.disposition == KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN &&
      (capacity.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY) != 0 &&
      capacity.survivors == nullptr && capacity.survivor_count == 0;
  if (!capacity_good) {
    report_m1_failure("CUDA M1 pair-work-capacity gate", capacity_status,
                      capacity);
  }
  good = capacity_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&capacity);

  auto duplicate_contacts = capacity_contacts;
  duplicate_contacts.push_back(make_m1_contact(1000, 0, 2, 63));
  klayout_cuda_spatial_m1_result_v1 duplicate{};
  const int duplicate_status =
      run_m1_request(duplicate_contacts, no_edges, config, duplicate);
  const bool duplicate_good =
      duplicate_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      duplicate.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      (duplicate.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST) != 0 &&
      duplicate.survivors == nullptr;
  if (!duplicate_good) {
    report_m1_failure("CUDA M1 duplicate-stable-ID gate", duplicate_status,
                      duplicate);
  }
  good = duplicate_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&duplicate);

  auto reserved_config = config;
  reserved_config.reserved0 = 1;
  klayout_cuda_spatial_m1_result_v1 reserved{};
  const int reserved_status =
      run_m1_request(capacity_contacts, no_edges, reserved_config, reserved);
  const bool reserved_good =
      reserved_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      reserved.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      reserved.disposition == KLAYOUT_CUDA_SPATIAL_M1_UNCERTAIN &&
      (reserved.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST) != 0 &&
      reserved.survivors == nullptr && reserved.survivor_count == 0;
  if (!reserved_good) {
    report_m1_failure("CUDA M1 reserved-config gate", reserved_status,
                      reserved);
  }
  good = reserved_good && good;
  klayout_cuda_spatial_release_m1_result_v1(&reserved);

  if (good) {
    std::cout << "CUDA M1 coordinate-extrema and fail-closed gates passed: "
                 "malformed, duplicate ID, reserved config, capacity\n";
  }
  return good;
}

void report_active3_failure(
    const char *name, int status,
    const klayout_cuda_spatial_active3_result_v1 &result) {
  std::cerr << name << " failed: call_status=" << status
            << " result_status=" << result.status
            << " disposition=" << result.disposition
            << " fallback_flags=" << result.fallback_flags
            << " device_flags=" << result.device_flags
            << " candidates=" << result.candidate_pair_count
            << " raw_hits=" << result.raw_hit_count
            << " uncertain=" << result.uncertain_count
            << " message=" << result.message << '\n';
}

bool set_active3_digest(
    klayout_cuda_spatial_active3_request_v1 &request) {
  std::array<std::uint8_t, 32> digest{};
  if (!db::cuda_active3_digest::request_digest(request, digest)) {
    return false;
  }
  std::copy(digest.begin(), digest.end(), request.scene_digest);
  return true;
}

bool run_active3_abi_smoke() {
  const klayout_cuda_spatial_active3_context_v1 contexts[] = {
      {0, 0, 0, 0},
  };
  const std::uint32_t well_contexts[] = {0};
  const std::uint64_t well_offsets[] = {0};
  const std::uint32_t active_contexts[] = {0};
  const klayout_cuda_spatial_active3_cell_v1 cells[] = {
      {0, 4, 4, 4},
  };
  klayout_cuda_spatial_active3_edge_v1 edges[] = {
      // Clockwise WELL: directed-edge material is on the right.
      {0, 0, 0, 1000},
      {0, 1000, 1000, 1000},
      {1000, 1000, 1000, 0},
      {1000, 0, 0, 0},
      // Clockwise ACTIVE, initially inset by 200 DBU (> d=110).
      {200, 200, 200, 800},
      {200, 800, 800, 800},
      {800, 800, 800, 200},
      {800, 200, 200, 200},
  };

  klayout_cuda_spatial_active3_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_SUPERSET_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_OPTIONS;
  request.dbu_per_micron = 2000;
  request.distance = 110;
  request.grid_cell_size = 2000;
  request.contexts = contexts;
  request.context_count = 1;
  request.well_contexts = well_contexts;
  request.well_context_count = 1;
  request.well_offsets = well_offsets;
  request.well_offset_count = 1;
  request.active_contexts = active_contexts;
  request.active_context_count = 1;
  request.cells = cells;
  request.cell_count = 1;
  request.edges = edges;
  request.edge_count = sizeof(edges) / sizeof(edges[0]);
  request.flat_well_edge_count = 4;
  request.flat_active_edge_count = 4;
  request.well_left = 0;
  request.well_bottom = 0;
  request.well_right = 1000;
  request.well_top = 1000;
  request.max_contexts = 16;
  request.max_grid_cells = 1024;
  request.max_memberships = 1024;
  request.max_pair_work = 1024;

  bool good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 clean{};
  const int clean_status =
      good ? klayout_cuda_spatial_run_active3_empty_v1(&request, &clean)
           : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool clean_good =
      good && clean_status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean.status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_COMPLETE &&
      clean.fallback_flags == 0 && clean.device_flags == 0 &&
      clean.raw_hit_count == 0 && clean.uncertain_count == 0;
  if (!clean_good) {
    report_active3_failure(
        "CUDA ACTIVE.3 clean-certificate smoke", clean_status, clean);
  }
  good = clean_good;

  // Move ACTIVE within 50 DBU of every WELL edge: raw hits must request
  // pristine CPU fallback and must never masquerade as publishable markers.
  edges[4] = {50, 50, 50, 950};
  edges[5] = {50, 950, 950, 950};
  edges[6] = {950, 950, 950, 50};
  edges[7] = {950, 50, 50, 50};
  const bool hit_digest_good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 raw_hit{};
  const int hit_status =
      hit_digest_good
          ? klayout_cuda_spatial_run_active3_empty_v1(&request, &raw_hit)
          : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool hit_good =
      hit_digest_good && hit_status == KLAYOUT_CUDA_SPATIAL_OK &&
      raw_hit.status == KLAYOUT_CUDA_SPATIAL_OK &&
      raw_hit.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_HITS &&
      raw_hit.fallback_flags == 0 && raw_hit.device_flags == 0 &&
      raw_hit.raw_hit_count != 0 && raw_hit.uncertain_count == 0;
  if (!hit_good) {
    report_active3_failure(
        "CUDA ACTIVE.3 raw-hit fallback smoke", hit_status, raw_hit);
  }
  good = hit_good && good;

  // The digest binds the exact serialized scene.  Any mismatch must be
  // rejected before the device pipeline can produce a consumable result.
  request.scene_digest[0] ^= 0x80u;
  klayout_cuda_spatial_active3_result_v1 tampered{};
  const int tampered_status =
      klayout_cuda_spatial_run_active3_empty_v1(&request, &tampered);
  const bool tampered_good =
      tampered_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      tampered.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      tampered.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN &&
      (tampered.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST) != 0 &&
      tampered.device_flags == 0 && tampered.raw_hit_count == 0 &&
      tampered.uncertain_count == 0;
  if (!tampered_good) {
    report_active3_failure(
        "CUDA ACTIVE.3 tampered-digest gate", tampered_status, tampered);
  }
  good = tampered_good && good;

  // Reject a count whose byte span cannot be represented before any array
  // walk, digest, allocation, or H2D copy can observe the supplied pointer.
  klayout_cuda_spatial_active3_request_v1 oversized = request;
  oversized.edge_count =
      std::numeric_limits<std::size_t>::max() /
          sizeof(klayout_cuda_spatial_active3_edge_v1) +
      1;
  klayout_cuda_spatial_active3_result_v1 oversized_result{};
  const int oversized_status =
      klayout_cuda_spatial_run_active3_empty_v1(
          &oversized, &oversized_result);
  const bool oversized_good =
      oversized_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      oversized_result.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      oversized_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN &&
      (oversized_result.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST) != 0 &&
      oversized_result.device_flags == 0 &&
      oversized_result.raw_hit_count == 0 &&
      oversized_result.uncertain_count == 0;
  if (!oversized_good) {
    report_active3_failure(
        "CUDA ACTIVE.3 oversized-payload gate",
        oversized_status, oversized_result);
  }
  good = oversized_good && good;

  if (good) {
    std::cout << "CUDA ACTIVE.3 additive ABI smoke passed: "
                 "clean certificate, raw-hit fallback, digest/count "
                 "rejection\n";
  }
  return good;
}

bool run_contact4_abi_smoke() {
  const klayout_cuda_spatial_active3_context_v1 contexts[] = {
      {0, 0, 0, 0},
  };
  const std::uint32_t indexed_contact_contexts[] = {0};
  const std::uint64_t indexed_contact_offsets[] = {0};
  const std::uint32_t streamed_active_contexts[] = {0};
  const klayout_cuda_spatial_active3_cell_v1 cells[] = {
      {0, 4, 4, 4},
  };
  klayout_cuda_spatial_active3_edge_v1 edges[] = {
      // Historical WELL slots: clockwise raw CONTACT, initially 200 DBU
      // inside ACTIVE and therefore clean for the 10-DBU CONTACT.4 rule.
      {200, 200, 200, 800},
      {200, 800, 800, 800},
      {800, 800, 800, 200},
      {800, 200, 200, 200},
      // Historical ACTIVE slots: clockwise merged ACTIVE primary.
      {0, 0, 0, 1000},
      {0, 1000, 1000, 1000},
      {1000, 1000, 1000, 0},
      {1000, 0, 0, 0},
  };

  klayout_cuda_spatial_active3_request_v1 request{};
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_SUPERSET_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_QUALIFIED_OPTIONS;
  request.dbu_per_micron = 2000;
  request.distance = 10;
  request.grid_cell_size = 2000;
  request.contexts = contexts;
  request.context_count = 1;
  request.well_contexts = indexed_contact_contexts;
  request.well_context_count = 1;
  request.well_offsets = indexed_contact_offsets;
  request.well_offset_count = 1;
  request.active_contexts = streamed_active_contexts;
  request.active_context_count = 1;
  request.cells = cells;
  request.cell_count = 1;
  request.edges = edges;
  request.edge_count = sizeof(edges) / sizeof(edges[0]);
  request.flat_well_edge_count = 4;
  request.flat_active_edge_count = 4;
  request.well_left = 200;
  request.well_bottom = 200;
  request.well_right = 800;
  request.well_top = 800;
  request.max_contexts = 16;
  request.max_grid_cells = 1024;
  request.max_memberships = 1024;
  // Deliberately smaller than the 4x4 Cartesian product.  CONTACT.4 must
  // qualify bounded spatial candidates rather than reject this request.
  request.max_pair_work = 15;

  bool good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 clean{};
  const int clean_status =
      good ? klayout_cuda_spatial_run_active3_empty_v1(&request, &clean)
           : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool clean_good =
      good && clean_status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean.status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_COMPLETE &&
      clean.opcode == KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_SUPERSET_EMPTY &&
      clean.option_flags ==
          KLAYOUT_CUDA_SPATIAL_CONTACT4_QUALIFIED_OPTIONS &&
      clean.distance == 10 && clean.fallback_flags == 0 &&
      clean.device_flags == 0 && clean.raw_hit_count == 0 &&
      clean.uncertain_count == 0 &&
      clean.candidate_pair_count <= request.max_pair_work;
  if (!clean_good) {
    report_active3_failure(
        "CUDA CONTACT.4 clean-certificate smoke", clean_status, clean);
  }
  good = clean_good;

  // Bring raw CONTACT to 5 DBU inside the merged ACTIVE boundary.  Correct
  // primary/secondary ordering reports raw hits; reversing the predicate
  // arguments would incorrectly call this scene clean.
  edges[0] = {5, 5, 5, 995};
  edges[1] = {5, 995, 995, 995};
  edges[2] = {995, 995, 995, 5};
  edges[3] = {995, 5, 5, 5};
  request.well_left = 5;
  request.well_bottom = 5;
  request.well_right = 995;
  request.well_top = 995;
  const bool hit_digest_good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 raw_hit{};
  const int hit_status =
      hit_digest_good
          ? klayout_cuda_spatial_run_active3_empty_v1(&request, &raw_hit)
          : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool hit_good =
      hit_digest_good && hit_status == KLAYOUT_CUDA_SPATIAL_OK &&
      raw_hit.status == KLAYOUT_CUDA_SPATIAL_OK &&
      raw_hit.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_RAW_HITS &&
      raw_hit.fallback_flags == 0 && raw_hit.device_flags == 0 &&
      raw_hit.raw_hit_count != 0 && raw_hit.uncertain_count == 0 &&
      raw_hit.candidate_pair_count <= request.max_pair_work;
  if (!hit_good) {
    report_active3_failure(
        "CUDA CONTACT.4 reversed-operand raw-hit smoke",
        hit_status, raw_hit);
  }
  good = hit_good && good;

  // The actual candidate limiter must fail closed after the Cartesian
  // preflight has intentionally been bypassed.
  request.max_pair_work = 1;
  const bool capacity_digest_good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 capacity{};
  const int capacity_status =
      capacity_digest_good
          ? klayout_cuda_spatial_run_active3_empty_v1(&request, &capacity)
          : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool capacity_good =
      capacity_digest_good &&
      capacity_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN &&
      (capacity.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY) != 0 &&
      capacity.device_flags != 0 &&
      capacity.candidate_pair_count > request.max_pair_work;
  if (!capacity_good) {
    report_active3_failure(
        "CUDA CONTACT.4 actual-candidate capacity gate",
        capacity_status, capacity);
  }
  good = capacity_good && good;

  // Profile fields are inseparable: CONTACT.4 opcode with ACTIVE.3 options,
  // or CONTACT.4 options with the ACTIVE.3 distance, must be rejected.
  request.max_pair_work = 15;
  request.option_flags = KLAYOUT_CUDA_SPATIAL_ACTIVE3_QUALIFIED_OPTIONS;
  const bool wrong_options_digest_good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 wrong_options{};
  const int wrong_options_status =
      wrong_options_digest_good
          ? klayout_cuda_spatial_run_active3_empty_v1(
                &request, &wrong_options)
          : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool wrong_options_good =
      wrong_options_digest_good &&
      wrong_options_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      wrong_options.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      wrong_options.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN;
  if (!wrong_options_good) {
    report_active3_failure(
        "CUDA CONTACT.4 strict-option gate",
        wrong_options_status, wrong_options);
  }
  good = wrong_options_good && good;

  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_QUALIFIED_OPTIONS;
  request.distance = 110;
  const bool wrong_distance_digest_good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 wrong_distance{};
  const int wrong_distance_status =
      wrong_distance_digest_good
          ? klayout_cuda_spatial_run_active3_empty_v1(
                &request, &wrong_distance)
          : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool wrong_distance_good =
      wrong_distance_digest_good &&
      wrong_distance_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      wrong_distance.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      wrong_distance.disposition == KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN;
  if (!wrong_distance_good) {
    report_active3_failure(
        "CUDA CONTACT.4 strict-distance gate",
        wrong_distance_status, wrong_distance);
  }
  good = wrong_distance_good && good;

  // The early raw-ACTIVE profile has a distinct optional symbol.  This lets a
  // host reject a stale backend before lowering the much larger raw scene and
  // prevents either entry point from silently accepting the other's profile.
  edges[0] = {200, 200, 200, 800};
  edges[1] = {200, 800, 800, 800};
  edges[2] = {800, 800, 800, 200};
  edges[3] = {800, 200, 200, 200};
  request.well_left = 200;
  request.well_bottom = 200;
  request.well_right = 800;
  request.well_top = 800;
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_SUPERSET_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_QUALIFIED_OPTIONS;
  request.distance = 10;
  const bool raw_active_digest_good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 raw_active_clean{};
  const int raw_active_status =
      raw_active_digest_good
          ? klayout_cuda_spatial_run_contact4_raw_active_empty_v1(
                &request, &raw_active_clean)
          : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool raw_active_good =
      raw_active_digest_good &&
      raw_active_status == KLAYOUT_CUDA_SPATIAL_OK &&
      raw_active_clean.status == KLAYOUT_CUDA_SPATIAL_OK &&
      raw_active_clean.disposition ==
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_COMPLETE &&
      raw_active_clean.opcode ==
          KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_SUPERSET_EMPTY &&
      raw_active_clean.option_flags ==
          KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_BOTH_QUALIFIED_OPTIONS &&
      raw_active_clean.fallback_flags == 0 &&
      raw_active_clean.device_flags == 0 &&
      raw_active_clean.raw_hit_count == 0 &&
      raw_active_clean.uncertain_count == 0;
  if (!raw_active_good) {
    report_active3_failure(
        "CUDA CONTACT.4 raw-ACTIVE symbol smoke",
        raw_active_status, raw_active_clean);
  }
  good = raw_active_good && good;

  klayout_cuda_spatial_active3_result_v1 raw_through_old{};
  const int raw_through_old_status =
      klayout_cuda_spatial_run_active3_empty_v1(
          &request, &raw_through_old);
  const bool raw_through_old_good =
      raw_through_old_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      raw_through_old.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      raw_through_old.disposition ==
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN;
  if (!raw_through_old_good) {
    report_active3_failure(
        "CUDA CONTACT.4 raw profile rejected by established symbol",
        raw_through_old_status, raw_through_old);
  }
  good = raw_through_old_good && good;

  request.opcode =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_RAW_SUPERSET_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_QUALIFIED_OPTIONS;
  const bool merged_digest_good = set_active3_digest(request);
  klayout_cuda_spatial_active3_result_v1 merged_through_raw{};
  const int merged_through_raw_status =
      merged_digest_good
          ? klayout_cuda_spatial_run_contact4_raw_active_empty_v1(
                &request, &merged_through_raw)
          : KLAYOUT_CUDA_SPATIAL_ERROR;
  const bool merged_through_raw_good =
      merged_digest_good &&
      merged_through_raw_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      merged_through_raw.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      merged_through_raw.disposition ==
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_UNCERTAIN;
  if (!merged_through_raw_good) {
    report_active3_failure(
        "CUDA CONTACT.4 merged profile rejected by raw symbol",
        merged_through_raw_status, merged_through_raw);
  }
  good = merged_through_raw_good && good;

  if (good) {
    std::cout << "CUDA CONTACT.4 additive profile smoke passed: "
                 "clean, reversed-operand raw hit, actual-candidate cap, "
                 "strict options/distance, distinct raw-ACTIVE symbol\n";
  }
  return good;
}

class M1WsSmokeDigest {
 public:
  void bytes(const void *data, std::size_t size) {
    sha_.update(data, size);
  }

  void u32(std::uint32_t value) {
    std::uint8_t encoded[4];
    for (unsigned int i = 0; i < 4; ++i) {
      encoded[i] = static_cast<std::uint8_t>(value >> (i * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void u64(std::uint64_t value) {
    std::uint8_t encoded[8];
    for (unsigned int i = 0; i < 8; ++i) {
      encoded[i] = static_cast<std::uint8_t>(value >> (i * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void i64(std::int64_t value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }

  std::array<std::uint8_t, 32> finish() {
    return sha_.finish();
  }

 private:
  db::cuda_active3_digest::Sha256 sha_;
};

struct M1WsSmokePoint {
  std::int64_t x;
  std::int64_t y;
};

struct M1WsSmokeScene {
  std::vector<klayout_cuda_spatial_m1_width_space_context_v1> contexts;
  std::vector<std::uint32_t> metal_contexts;
  std::vector<std::uint64_t> polygon_offsets;
  std::vector<std::uint64_t> edge_offsets;
  std::vector<klayout_cuda_spatial_m1_width_space_cell_v1> cells;
  std::vector<klayout_cuda_spatial_m1_width_space_polygon_v1> polygons;
  std::vector<klayout_cuda_spatial_m1_width_space_edge_v1> edges;
  klayout_cuda_spatial_m1_width_space_request_v1 request{};
};

M1WsSmokeScene make_m1ws_scene(
    const std::vector<std::vector<M1WsSmokePoint>> &shapes) {
  M1WsSmokeScene scene;
  scene.contexts.push_back({0, 0, 0, 0});
  scene.metal_contexts.push_back(0);
  scene.polygon_offsets.push_back(0);
  scene.edge_offsets.push_back(0);
  std::int64_t scene_left = INT64_MAX;
  std::int64_t scene_bottom = INT64_MAX;
  std::int64_t scene_right = INT64_MIN;
  std::int64_t scene_top = INT64_MIN;
  for (std::size_t polygon_id = 0; polygon_id < shapes.size();
       ++polygon_id) {
    const auto &points = shapes[polygon_id];
    klayout_cuda_spatial_m1_width_space_polygon_v1 polygon{};
    polygon.edge_begin = scene.edges.size();
    polygon.polygon_id = static_cast<std::uint32_t>(polygon_id);
    polygon.edge_count = static_cast<std::uint32_t>(points.size());
    polygon.left = INT64_MAX;
    polygon.bottom = INT64_MAX;
    polygon.right = INT64_MIN;
    polygon.top = INT64_MIN;
    for (std::size_t edge_id = 0; edge_id < points.size(); ++edge_id) {
      const auto first = points[edge_id];
      const auto second = points[(edge_id + 1) % points.size()];
      scene.edges.push_back(
          {first.x, first.y, second.x, second.y});
      polygon.left = std::min(polygon.left, first.x);
      polygon.bottom = std::min(polygon.bottom, first.y);
      polygon.right = std::max(polygon.right, first.x);
      polygon.top = std::max(polygon.top, first.y);
    }
    scene_left = std::min(scene_left, polygon.left);
    scene_bottom = std::min(scene_bottom, polygon.bottom);
    scene_right = std::max(scene_right, polygon.right);
    scene_top = std::max(scene_top, polygon.top);
    scene.polygons.push_back(polygon);
  }
  scene.cells.push_back({
      0, 0, 0, static_cast<std::uint32_t>(scene.polygons.size()),
      static_cast<std::uint32_t>(scene.edges.size())});

  auto &request = scene.request;
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_MERGED_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.root_cell = 0;
  request.device = 0;
  request.width_distance = 130;
  request.spacing_distance = 130;
  request.grid_cell_size = 512;
  request.flat_polygon_count = scene.polygons.size();
  request.flat_edge_count = scene.edges.size();
  request.scene_left = scene_left;
  request.scene_bottom = scene_bottom;
  request.scene_right = scene_right;
  request.scene_top = scene_top;
  request.max_contexts = 1024;
  request.max_grid_cells = 1000000;
  request.max_memberships = 10000000;
  request.max_pair_work = 100000000;
  request.max_flat_edges = 1000000;
  request.max_flat_polygons = 1000000;
  return scene;
}

void bind_m1ws_scene(M1WsSmokeScene &scene) {
  auto &request = scene.request;
  request.contexts = scene.contexts.data();
  request.context_count = scene.contexts.size();
  request.context_record_bytes =
      sizeof(klayout_cuda_spatial_m1_width_space_context_v1);
  request.metal_contexts = scene.metal_contexts.data();
  request.metal_context_count = scene.metal_contexts.size();
  request.context_polygon_offsets = scene.polygon_offsets.data();
  request.context_polygon_offset_count = scene.polygon_offsets.size();
  request.context_edge_offsets = scene.edge_offsets.data();
  request.context_edge_offset_count = scene.edge_offsets.size();
  request.cells = scene.cells.data();
  request.cell_count = scene.cells.size();
  request.cell_record_bytes =
      sizeof(klayout_cuda_spatial_m1_width_space_cell_v1);
  request.polygons = scene.polygons.data();
  request.polygon_count = scene.polygons.size();
  request.polygon_record_bytes =
      sizeof(klayout_cuda_spatial_m1_width_space_polygon_v1);
  request.edges = scene.edges.data();
  request.edge_count = scene.edges.size();
  request.edge_record_bytes =
      sizeof(klayout_cuda_spatial_m1_width_space_edge_v1);
}

bool set_m1ws_digest(M1WsSmokeScene &scene) {
  bind_m1ws_scene(scene);
  const auto &request = scene.request;
  M1WsSmokeDigest sha;
  const char magic[8] = {'K', 'M', '1', 'W', 'S', '0', '0', '1'};
  sha.bytes(magic, sizeof(magic));
  sha.u32(request.format_version);
  sha.u32(request.dbu_per_micron);
  sha.u32(request.root_cell);
  sha.u32(request.scene_reserved);
  sha.i64(request.width_distance);
  sha.i64(request.spacing_distance);
  sha.u64(request.context_count);
  sha.u64(request.metal_context_count);
  sha.u64(request.cell_count);
  sha.u64(request.polygon_count);
  sha.u64(request.edge_count);
  sha.u64(request.flat_polygon_count);
  sha.u64(request.flat_edge_count);
  sha.i64(request.scene_left);
  sha.i64(request.scene_bottom);
  sha.i64(request.scene_right);
  sha.i64(request.scene_top);
  for (const auto &context : scene.contexts) {
    sha.i64(context.tx);
    sha.i64(context.ty);
    sha.u32(context.cell_id);
    sha.u32(context.transform_code);
  }
  for (std::size_t id = 0; id < scene.metal_contexts.size(); ++id) {
    sha.u32(scene.metal_contexts[id]);
    sha.u64(scene.polygon_offsets[id]);
    sha.u64(scene.edge_offsets[id]);
  }
  for (const auto &cell : scene.cells) {
    sha.u64(cell.source_cell_index);
    sha.u64(cell.polygon_begin);
    sha.u64(cell.edge_begin);
    sha.u32(cell.polygon_count);
    sha.u32(cell.edge_count);
  }
  for (const auto &polygon : scene.polygons) {
    sha.u64(polygon.edge_begin);
    sha.i64(polygon.left);
    sha.i64(polygon.bottom);
    sha.i64(polygon.right);
    sha.i64(polygon.top);
    sha.u32(polygon.polygon_id);
    sha.u32(polygon.edge_count);
  }
  for (const auto &edge : scene.edges) {
    sha.i64(edge.x1);
    sha.i64(edge.y1);
    sha.i64(edge.x2);
    sha.i64(edge.y2);
  }
  const auto digest = sha.finish();
  std::copy(
      digest.begin(), digest.end(), scene.request.scene_digest);
  return true;
}

void report_m1ws_failure(
    const char *name, int status,
    const klayout_cuda_spatial_m1_width_space_result_v1 &result) {
  std::cerr << name << " failed: call_status=" << status
            << " result_status=" << result.status
            << " disposition=" << result.disposition
            << " fallback_flags=" << result.fallback_flags
            << " device_flags=" << result.device_flags
            << " width_hits=" << result.width_hit_count
            << " space_hits=" << result.space_hit_count
            << " width_uncertain=" << result.width_uncertain_count
            << " space_uncertain=" << result.space_uncertain_count
            << " message=" << result.message << '\n';
}

bool valid_m1ws_counters(
    const klayout_cuda_spatial_m1_width_space_result_v1 &result) {
  return result.width_pair_count <= result.unique_edge_pair_count &&
         result.space_pair_count == result.unique_edge_pair_count &&
         result.width_hit_count + result.width_uncertain_count <=
             result.width_pair_count &&
         result.space_hit_count + result.space_uncertain_count <=
             result.space_pair_count;
}

bool run_m1ws_abi_smoke() {
  const auto rectangle = [](std::int64_t left, std::int64_t bottom,
                            std::int64_t right, std::int64_t top) {
    return std::vector<M1WsSmokePoint>{
        {left, bottom}, {left, top}, {right, top}, {right, bottom}};
  };
  bool good = true;

  M1WsSmokeScene clean =
      make_m1ws_scene({rectangle(0, 0, 300, 130)});
  set_m1ws_digest(clean);
  klayout_cuda_spatial_m1_width_space_result_v1 clean_result{};
  const int clean_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &clean.request, &clean_result);
  const bool clean_good =
      clean_status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_COMPLETE &&
      clean_result.fallback_flags == 0 &&
      clean_result.device_flags == 0 &&
      clean_result.width_hit_count == 0 &&
      clean_result.space_hit_count == 0 &&
      clean_result.width_uncertain_count == 0 &&
      clean_result.space_uncertain_count == 0 &&
      clean_result.context_count == 1 &&
      clean_result.flat_edge_count == 4 &&
      clean_result.grid_cell_size == clean.request.grid_cell_size &&
      clean_result.scene_left == clean.request.scene_left &&
      clean_result.scene_bottom == clean.request.scene_bottom &&
      clean_result.scene_right == clean.request.scene_right &&
      clean_result.scene_top == clean.request.scene_top &&
      std::equal(
          clean.request.scene_digest,
          clean.request.scene_digest + 32,
          clean_result.scene_digest) &&
      valid_m1ws_counters(clean_result);
  if (!clean_good) {
    report_m1ws_failure(
        "CUDA M1 width/space clean certificate", clean_status,
        clean_result);
  }
  good = clean_good && good;

  M1WsSmokeScene m2_clean =
      make_m1ws_scene({rectangle(0, 0, 300, 140)});
  m2_clean.request.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_WIDTH_SPACE_MERGED_EMPTY;
  m2_clean.request.width_distance = 140;
  m2_clean.request.spacing_distance = 140;
  set_m1ws_digest(m2_clean);
  klayout_cuda_spatial_m1_width_space_result_v1 m2_clean_result{};
  const int m2_clean_status =
      klayout_cuda_spatial_run_m2_width_space_empty_v1(
          &m2_clean.request, &m2_clean_result);
  const bool m2_clean_good =
      m2_clean_status == KLAYOUT_CUDA_SPATIAL_OK &&
      m2_clean_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      m2_clean_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_COMPLETE &&
      m2_clean_result.opcode == m2_clean.request.opcode &&
      m2_clean_result.width_distance == 140 &&
      m2_clean_result.spacing_distance == 140 &&
      m2_clean_result.width_hit_count == 0 &&
      m2_clean_result.space_hit_count == 0 &&
      m2_clean_result.width_uncertain_count == 0 &&
      m2_clean_result.space_uncertain_count == 0 &&
      m2_clean_result.fallback_flags == 0 &&
      m2_clean_result.device_flags == 0 &&
      valid_m1ws_counters(m2_clean_result);
  if (!m2_clean_good) {
    report_m1ws_failure(
        "CUDA M2 width/space clean certificate", m2_clean_status,
        m2_clean_result);
  }
  good = m2_clean_good && good;

  M1WsSmokeScene m2_width_hit =
      make_m1ws_scene({rectangle(0, 0, 300, 139)});
  m2_width_hit.request.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_WIDTH_SPACE_MERGED_EMPTY;
  m2_width_hit.request.width_distance = 140;
  m2_width_hit.request.spacing_distance = 140;
  set_m1ws_digest(m2_width_hit);
  klayout_cuda_spatial_m1_width_space_result_v1 m2_width_result{};
  const int m2_width_status =
      klayout_cuda_spatial_run_m2_width_space_empty_v1(
          &m2_width_hit.request, &m2_width_result);
  const bool m2_width_good =
      m2_width_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      m2_width_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      m2_width_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_RAW_HITS &&
      m2_width_result.fallback_flags == 0 &&
      m2_width_result.device_flags == 0 &&
      m2_width_result.width_hit_count != 0 &&
      m2_width_result.space_hit_count == 0 &&
      m2_width_result.width_uncertain_count == 0 &&
      m2_width_result.space_uncertain_count == 0 &&
      valid_m1ws_counters(m2_width_result);
  if (!m2_width_good) {
    report_m1ws_failure(
        "CUDA M2 139-DBU width raw-hit fallback", m2_width_status,
        m2_width_result);
  }
  good = m2_width_good && good;

  M1WsSmokeScene m2_space_hit = make_m1ws_scene({
      rectangle(0, 0, 200, 300),
      rectangle(339, 0, 539, 300)});
  m2_space_hit.request.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_WIDTH_SPACE_MERGED_EMPTY;
  m2_space_hit.request.width_distance = 140;
  m2_space_hit.request.spacing_distance = 140;
  set_m1ws_digest(m2_space_hit);
  klayout_cuda_spatial_m1_width_space_result_v1 m2_space_result{};
  const int m2_space_status =
      klayout_cuda_spatial_run_m2_width_space_empty_v1(
          &m2_space_hit.request, &m2_space_result);
  const bool m2_space_good =
      m2_space_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      m2_space_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      m2_space_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_RAW_HITS &&
      m2_space_result.fallback_flags == 0 &&
      m2_space_result.device_flags == 0 &&
      m2_space_result.space_hit_count != 0 &&
      m2_space_result.width_hit_count == 0 &&
      m2_space_result.width_uncertain_count == 0 &&
      m2_space_result.space_uncertain_count == 0 &&
      valid_m1ws_counters(m2_space_result);
  if (!m2_space_good) {
    report_m1ws_failure(
        "CUDA M2 139-DBU spacing raw-hit fallback", m2_space_status,
        m2_space_result);
  }
  good = m2_space_good && good;

  M1WsSmokeScene opcode1_distance140 =
      make_m1ws_scene({rectangle(0, 0, 300, 140)});
  opcode1_distance140.request.width_distance = 140;
  opcode1_distance140.request.spacing_distance = 140;
  set_m1ws_digest(opcode1_distance140);
  klayout_cuda_spatial_m1_width_space_result_v1 opcode1_distance140_result{};
  const int opcode1_distance140_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &opcode1_distance140.request, &opcode1_distance140_result);
  const bool opcode1_distance140_good =
      opcode1_distance140_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      opcode1_distance140_result.status ==
          KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      opcode1_distance140_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      opcode1_distance140_result.fallback_flags ==
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  if (!opcode1_distance140_good) {
    report_m1ws_failure(
        "CUDA opcode1 plus 140/140 rejection",
        opcode1_distance140_status, opcode1_distance140_result);
  }
  good = opcode1_distance140_good && good;

  M1WsSmokeScene opcode2_distance130 =
      make_m1ws_scene({rectangle(0, 0, 300, 130)});
  opcode2_distance130.request.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_WIDTH_SPACE_MERGED_EMPTY;
  set_m1ws_digest(opcode2_distance130);
  klayout_cuda_spatial_m1_width_space_result_v1 opcode2_distance130_result{};
  const int opcode2_distance130_status =
      klayout_cuda_spatial_run_m2_width_space_empty_v1(
          &opcode2_distance130.request, &opcode2_distance130_result);
  const bool opcode2_distance130_good =
      opcode2_distance130_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      opcode2_distance130_result.status ==
          KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      opcode2_distance130_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      opcode2_distance130_result.fallback_flags ==
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  if (!opcode2_distance130_good) {
    report_m1ws_failure(
        "CUDA opcode2 plus 130/130 rejection",
        opcode2_distance130_status, opcode2_distance130_result);
  }
  good = opcode2_distance130_good && good;

  klayout_cuda_spatial_m1_width_space_result_v1 m2_on_m1_symbol{};
  const int m2_on_m1_symbol_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &m2_clean.request, &m2_on_m1_symbol);
  const bool m2_on_m1_symbol_good =
      m2_on_m1_symbol_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      m2_on_m1_symbol.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      m2_on_m1_symbol.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      m2_on_m1_symbol.fallback_flags ==
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  if (!m2_on_m1_symbol_good) {
    report_m1ws_failure(
        "CUDA M2 request on legacy M1 symbol rejection",
        m2_on_m1_symbol_status, m2_on_m1_symbol);
  }
  good = m2_on_m1_symbol_good && good;

  klayout_cuda_spatial_m1_width_space_result_v1 m1_on_m2_symbol{};
  const int m1_on_m2_symbol_status =
      klayout_cuda_spatial_run_m2_width_space_empty_v1(
          &clean.request, &m1_on_m2_symbol);
  const bool m1_on_m2_symbol_good =
      m1_on_m2_symbol_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      m1_on_m2_symbol.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      m1_on_m2_symbol.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      m1_on_m2_symbol.fallback_flags ==
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  if (!m1_on_m2_symbol_good) {
    report_m1ws_failure(
        "CUDA M1 request on M2 symbol rejection",
        m1_on_m2_symbol_status, m1_on_m2_symbol);
  }
  good = m1_on_m2_symbol_good && good;

  M1WsSmokeScene width_hit =
      make_m1ws_scene({rectangle(0, 0, 300, 129)});
  set_m1ws_digest(width_hit);
  klayout_cuda_spatial_m1_width_space_result_v1 width_result{};
  const int width_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &width_hit.request, &width_result);
  const bool width_good =
      width_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      width_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      width_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_RAW_HITS &&
      width_result.device_flags == 0 &&
      width_result.width_hit_count != 0 &&
      width_result.width_uncertain_count == 0 &&
      width_result.space_uncertain_count == 0 &&
      valid_m1ws_counters(width_result);
  if (!width_good) {
    report_m1ws_failure(
        "CUDA M1 width raw-hit fallback", width_status, width_result);
  }
  good = width_good && good;

  M1WsSmokeScene space_hit = make_m1ws_scene({
      rectangle(0, 0, 200, 300),
      rectangle(329, 0, 529, 300)});
  set_m1ws_digest(space_hit);
  klayout_cuda_spatial_m1_width_space_result_v1 space_result{};
  const int space_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &space_hit.request, &space_result);
  const bool space_good =
      space_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      space_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      space_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_RAW_HITS &&
      space_result.device_flags == 0 &&
      space_result.space_hit_count != 0 &&
      space_result.width_hit_count == 0 &&
      space_result.width_uncertain_count == 0 &&
      space_result.space_uncertain_count == 0 &&
      valid_m1ws_counters(space_result);
  if (!space_good) {
    report_m1ws_failure(
        "CUDA M1 spacing raw-hit fallback", space_status, space_result);
  }
  good = space_good && good;

  clean.request.scene_digest[0] ^= 0x80u;
  klayout_cuda_spatial_m1_width_space_result_v1 tampered{};
  const int tampered_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &clean.request, &tampered);
  const bool tampered_good =
      tampered_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      tampered.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      tampered.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      tampered.fallback_flags ==
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST &&
      tampered.width_hit_count == 0 &&
      tampered.space_hit_count == 0;
  if (!tampered_good) {
    report_m1ws_failure(
        "CUDA M1 width/space digest gate", tampered_status, tampered);
  }
  good = tampered_good && good;
  clean.request.scene_digest[0] ^= 0x80u;

  const std::uint32_t saved_stride = clean.request.edge_record_bytes;
  clean.request.edge_record_bytes = saved_stride + 8;
  klayout_cuda_spatial_m1_width_space_result_v1 stride{};
  const int stride_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &clean.request, &stride);
  const bool stride_good =
      stride_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      stride.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      stride.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      stride.fallback_flags ==
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  if (!stride_good) {
    report_m1ws_failure(
        "CUDA M1 width/space stride gate", stride_status, stride);
  }
  good = stride_good && good;
  clean.request.edge_record_bytes = saved_stride;

  clean.request.max_memberships = 1;
  klayout_cuda_spatial_m1_width_space_result_v1 capacity{};
  const int capacity_status =
      klayout_cuda_spatial_run_m1_width_space_empty_v1(
          &clean.request, &capacity);
  const bool capacity_good =
      capacity_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity.disposition ==
          KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN &&
      (capacity.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY) != 0 &&
      capacity.width_hit_count == 0 &&
      capacity.space_hit_count == 0;
  if (!capacity_good) {
    report_m1ws_failure(
        "CUDA M1 width/space capacity gate", capacity_status, capacity);
  }
  good = capacity_good && good;

  if (good) {
    std::cout << "CUDA M1/M2 width/space additive ABI smoke passed: "
                 "profile-specific symbols, atomic clean profiles, exact "
                 "129/139-DBU width/space raw-hit fallback, cross-profile "
                 "rejection, digest/stride/capacity rejection\n";
  }
  return good;
}

struct Implant12Fixture {
  std::vector<klayout_cuda_spatial_implant12_context_v1> contexts;
  std::vector<std::uint32_t> implant_contexts;
  std::vector<std::uint64_t> implant_offsets;
  std::vector<std::uint32_t> gate_contexts;
  std::vector<std::uint32_t> contact_contexts;
  std::vector<klayout_cuda_spatial_implant12_cell_v1> cells;
  std::vector<klayout_cuda_spatial_implant12_contour_v1> contours;
  std::vector<klayout_cuda_spatial_implant12_edge_v1> edges;
  klayout_cuda_spatial_implant12_request_v1 request{};

  static void append_box(
      std::vector<klayout_cuda_spatial_implant12_edge_v1> *destination,
      std::int64_t left, std::int64_t bottom,
      std::int64_t right, std::int64_t top) {
    destination->push_back({left, bottom, left, top});
    destination->push_back({left, top, right, top});
    destination->push_back({right, top, right, bottom});
    destination->push_back({right, bottom, left, bottom});
  }

  Implant12Fixture(std::int64_t gate_left, std::int64_t gate_bottom,
                   std::int64_t contact_left,
                   std::int64_t contact_bottom) {
    contexts = {
        {0, 0, 0, 0},
        {5000, 0, 0, 4},
    };
    implant_contexts = {0, 1};
    implant_offsets = {0, 4, 8};
    gate_contexts = {0, 1};
    contact_contexts = {0, 1};
    cells.resize(1);
    cells[0].source_cell_index = 17;

    const std::int64_t lefts[3] = {0, gate_left, contact_left};
    const std::int64_t bottoms[3] = {0, gate_bottom, contact_bottom};
    const std::int64_t rights[3] = {
        100, gate_left + 100, contact_left + 100};
    const std::int64_t tops[3] = {
        100, gate_bottom + 20, contact_bottom + 20};
    for (std::uint32_t domain = 0; domain < 3; ++domain) {
      auto &span = cells[0].domains[domain];
      span.contour_begin = contours.size();
      span.edge_begin = edges.size();
      span.polygon_count = 1;
      span.contour_count = 1;
      span.edge_count = 4;
      contours.push_back({
          static_cast<std::uint64_t>(edges.size()), 0, 0, 4,
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_HULL});
      append_box(
          &edges, lefts[domain], bottoms[domain],
          rights[domain], tops[domain]);
    }

    request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.opcode =
        KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_SUPERSET_EMPTY;
    request.option_flags =
        KLAYOUT_CUDA_SPATIAL_IMPLANT12_QUALIFIED_OPTIONS;
    request.format_version = 1;
    request.dbu_per_micron = 2000;
    request.root_cell = 0;
    request.requested_mask =
        KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES;
    request.device = 0;
    request.implant1_distance = 140;
    request.implant2_distance = 50;
    request.grid_cell_size = 2000;
    request.contexts = contexts.data();
    request.context_count = contexts.size();
    request.context_record_bytes =
        sizeof(klayout_cuda_spatial_implant12_context_v1);
    request.implant_contexts = implant_contexts.data();
    request.implant_context_count = implant_contexts.size();
    request.implant_edge_offsets = implant_offsets.data();
    request.implant_edge_offset_count = implant_offsets.size();
    request.gate_contexts = gate_contexts.data();
    request.gate_context_count = gate_contexts.size();
    request.contact_contexts = contact_contexts.data();
    request.contact_context_count = contact_contexts.size();
    request.cells = cells.data();
    request.cell_count = cells.size();
    request.cell_record_bytes =
        sizeof(klayout_cuda_spatial_implant12_cell_v1);
    request.contours = contours.data();
    request.contour_count = contours.size();
    request.contour_record_bytes =
        sizeof(klayout_cuda_spatial_implant12_contour_v1);
    request.edges = edges.data();
    request.edge_count = edges.size();
    request.edge_record_bytes =
        sizeof(klayout_cuda_spatial_implant12_edge_v1);
    request.flat_implant_polygon_count = 2;
    request.flat_gate_polygon_count = 2;
    request.flat_contact_polygon_count = 2;
    request.flat_implant_contour_count = 2;
    request.flat_gate_contour_count = 2;
    request.flat_contact_contour_count = 2;
    request.flat_implant_edge_count = 8;
    request.flat_gate_edge_count = 8;
    request.flat_contact_edge_count = 8;
    request.implant_left = 0;
    request.implant_bottom = -100;
    request.implant_right = 5100;
    request.implant_top = 100;
    request.max_contexts = 100;
    request.max_grid_cells = 100;
    request.max_implant_memberships = 1000;
    request.max_gate_query_visits = 10000;
    request.max_gate_candidate_work = 10000;
    request.max_contact_query_visits = 10000;
    request.max_contact_candidate_work = 10000;
    request.max_flat_polygons = 1000;
    request.max_flat_contours = 1000;
    request.max_flat_edges = 10000;
    refresh_digest();
  }

  bool refresh_digest() {
    request.contexts = contexts.data();
    request.implant_contexts = implant_contexts.data();
    request.implant_edge_offsets = implant_offsets.data();
    request.gate_contexts = gate_contexts.data();
    request.contact_contexts = contact_contexts.data();
    request.cells = cells.data();
    request.contours = contours.data();
    request.edges = edges.data();
    std::array<std::uint8_t, 32> digest{};
    if (!db::cuda_implant12_digest::request_digest(request, digest)) {
      return false;
    }
    std::copy(digest.begin(), digest.end(), request.scene_digest);
    return true;
  }
};

void report_implant12_failure(
    const char *name, int status,
    const klayout_cuda_spatial_implant12_result_v1 &result) {
  std::cerr << name << " failed: call_status=" << status
            << " result_status=" << result.status
            << " disposition=" << result.disposition
            << " clean_mask=" << result.clean_mask
            << " certified_mask=" << result.certified_empty_mask
            << " gate_candidates=" << result.gate_candidate_count
            << " gate_hits=" << result.gate_raw_hit_count
            << " contact_candidates=" << result.contact_candidate_count
            << " contact_hits=" << result.contact_raw_hit_count
            << " device_flags=" << result.device_flags
            << " fallback_flags=" << result.fallback_flags
            << " message=" << result.message << '\n';
}

bool run_implant12_abi_smoke() {
  bool good = true;

  // Both secondaries are exactly on the strict d boundary.  The second
  // context is mirrored and translated, exercising endpoint reversal while
  // remaining geometrically equivalent and cross-context disjoint.
  Implant12Fixture clean(0, 240, 0, 150);
  klayout_cuda_spatial_implant12_result_v1 clean_result{};
  const int clean_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &clean.request, &clean_result);
  const bool clean_good =
      clean_status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      clean_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_COMPLETE &&
      clean_result.clean_mask ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES &&
      clean_result.certified_empty_mask ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES &&
      clean_result.gate_raw_hit_count == 0 &&
      clean_result.contact_raw_hit_count == 0 &&
      clean_result.gate_processed_edge_count == 8 &&
      clean_result.contact_processed_edge_count == 8 &&
      clean_result.implant_expanded_edge_count == 8 &&
      clean_result.grid_cell_count <= clean.request.max_grid_cells &&
      clean_result.implant_membership_count <=
          clean.request.max_implant_memberships &&
      std::equal(
          clean_result.scene_digest,
          clean_result.scene_digest + 32,
          clean.request.scene_digest);
  if (!clean_good) {
    report_implant12_failure(
        "CUDA IMPLANT boundary/mirror clean", clean_status, clean_result);
  }
  good = clean_good && good;

  Implant12Fixture gate_hit(0, 239, 0, 150);
  klayout_cuda_spatial_implant12_result_v1 gate_result{};
  const int gate_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &gate_hit.request, &gate_result);
  const bool gate_good =
      gate_status == KLAYOUT_CUDA_SPATIAL_OK &&
      gate_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      gate_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_HITS &&
      gate_result.gate_raw_hit_count == 2 &&
      gate_result.contact_raw_hit_count == 0 &&
      gate_result.clean_mask == KLAYOUT_CUDA_SPATIAL_IMPLANT2_RULE &&
      gate_result.certified_empty_mask == 0;
  if (!gate_good) {
    report_implant12_failure(
        "CUDA IMPLANT.1 strict d-1 hit", gate_status, gate_result);
  }
  good = gate_good && good;

  Implant12Fixture contact_hit(0, 240, 0, 149);
  klayout_cuda_spatial_implant12_result_v1 contact_result{};
  const int contact_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &contact_hit.request, &contact_result);
  const bool contact_good =
      contact_status == KLAYOUT_CUDA_SPATIAL_OK &&
      contact_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      contact_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_HITS &&
      contact_result.gate_raw_hit_count == 0 &&
      contact_result.contact_raw_hit_count == 2 &&
      contact_result.clean_mask == KLAYOUT_CUDA_SPATIAL_IMPLANT1_RULE &&
      contact_result.certified_empty_mask == 0;
  if (!contact_good) {
    report_implant12_failure(
        "CUDA IMPLANT.2 strict d-1 hit",
        contact_status, contact_result);
  }
  good = contact_good && good;

  Implant12Fixture both_hit(0, 100, 0, 100);
  klayout_cuda_spatial_implant12_result_v1 both_result{};
  const int both_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &both_hit.request, &both_result);
  const bool both_good =
      both_status == KLAYOUT_CUDA_SPATIAL_OK &&
      both_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      both_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_HITS &&
      both_result.gate_raw_hit_count != 0 &&
      both_result.contact_raw_hit_count != 0 &&
      both_result.clean_mask == 0 &&
      both_result.certified_empty_mask == 0;
  if (!both_good) {
    report_implant12_failure(
        "CUDA IMPLANT collinear/mirrored both-hit",
        both_status, both_result);
  }
  good = both_good && good;

  Implant12Fixture endpoint_clean(100, 101, 100, 101);
  klayout_cuda_spatial_implant12_result_v1 endpoint_result{};
  const int endpoint_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &endpoint_clean.request, &endpoint_result);
  const bool endpoint_good =
      endpoint_status == KLAYOUT_CUDA_SPATIAL_OK &&
      endpoint_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      endpoint_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_COMPLETE &&
      endpoint_result.clean_mask ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES &&
      endpoint_result.certified_empty_mask ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_ALL_RULES &&
      endpoint_result.gate_raw_hit_count == 0 &&
      endpoint_result.contact_raw_hit_count == 0;
  if (!endpoint_good) {
    report_implant12_failure(
        "CUDA IMPLANT endpoint-only projection",
        endpoint_status, endpoint_result);
  }
  good = endpoint_good && good;

  // One long negative-coordinate polygon spans six X cells.  Its one
  // IMPLANT.1 facing pair must be classified exactly once despite sharing
  // multiple grid cells; IMPLANT.2 remains on its strict boundary.
  Implant12Fixture long_negative(0, 240, 0, 150);
  long_negative.contexts.resize(1);
  long_negative.implant_contexts = {0};
  long_negative.implant_offsets = {0, 4};
  long_negative.gate_contexts = {0};
  long_negative.contact_contexts = {0};
  long_negative.edges.clear();
  Implant12Fixture::append_box(
      &long_negative.edges, -4100, -100, 4100, 0);
  Implant12Fixture::append_box(
      &long_negative.edges, -4100, 139, 4100, 159);
  Implant12Fixture::append_box(
      &long_negative.edges, -4100, 50, 4100, 70);
  long_negative.request.context_count = 1;
  long_negative.request.implant_context_count = 1;
  long_negative.request.implant_edge_offset_count = 2;
  long_negative.request.gate_context_count = 1;
  long_negative.request.contact_context_count = 1;
  long_negative.request.flat_implant_polygon_count = 1;
  long_negative.request.flat_gate_polygon_count = 1;
  long_negative.request.flat_contact_polygon_count = 1;
  long_negative.request.flat_implant_contour_count = 1;
  long_negative.request.flat_gate_contour_count = 1;
  long_negative.request.flat_contact_contour_count = 1;
  long_negative.request.flat_implant_edge_count = 4;
  long_negative.request.flat_gate_edge_count = 4;
  long_negative.request.flat_contact_edge_count = 4;
  long_negative.request.implant_left = -4100;
  long_negative.request.implant_bottom = -100;
  long_negative.request.implant_right = 4100;
  long_negative.request.implant_top = 0;
  long_negative.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 long_result{};
  const int long_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &long_negative.request, &long_result);
  const bool long_good =
      long_status == KLAYOUT_CUDA_SPATIAL_OK &&
      long_result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      long_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_RAW_HITS &&
      long_result.gate_raw_hit_count == 1 &&
      long_result.contact_raw_hit_count == 0 &&
      long_result.clean_mask == KLAYOUT_CUDA_SPATIAL_IMPLANT2_RULE &&
      long_result.grid_cell_count > 4 &&
      long_result.implant_membership_count >
          long_result.implant_expanded_edge_count;
  if (!long_good) {
    report_implant12_failure(
        "CUDA IMPLANT negative long-edge canonicalization",
        long_status, long_result);
  }
  good = long_good && good;

  Implant12Fixture digest_bad(0, 240, 0, 150);
  digest_bad.request.scene_digest[0] ^= 1;
  klayout_cuda_spatial_implant12_result_v1 digest_result{};
  const int digest_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &digest_bad.request, &digest_result);
  const bool digest_good =
      digest_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      digest_result.status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      digest_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN &&
      digest_result.certified_empty_mask == 0;
  if (!digest_good) {
    report_implant12_failure(
        "CUDA IMPLANT digest rejection", digest_status, digest_result);
  }
  good = digest_good && good;

  Implant12Fixture order_bad(0, 240, 0, 150);
  std::swap(
      order_bad.request.implant1_distance,
      order_bad.request.implant2_distance);
  order_bad.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 order_result{};
  const int order_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &order_bad.request, &order_result);
  const bool order_good =
      order_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      order_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN;
  if (!order_good) {
    report_implant12_failure(
        "CUDA IMPLANT ordered-distance rejection",
        order_status, order_result);
  }
  good = order_good && good;

  Implant12Fixture span_bad(0, 240, 0, 150);
  ++span_bad.cells[0]
        .domains[KLAYOUT_CUDA_SPATIAL_IMPLANT12_GATE_DOMAIN]
        .edge_begin;
  span_bad.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 span_result{};
  const int span_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &span_bad.request, &span_result);
  const bool span_good =
      span_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      span_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN;
  if (!span_good) {
    report_implant12_failure(
        "CUDA IMPLANT span rejection", span_status, span_result);
  }
  good = span_good && good;

  Implant12Fixture list_bad(0, 240, 0, 150);
  std::swap(list_bad.gate_contexts[0], list_bad.gate_contexts[1]);
  list_bad.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 list_result{};
  const int list_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &list_bad.request, &list_result);
  const bool list_good =
      list_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      list_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN;
  if (!list_good) {
    report_implant12_failure(
        "CUDA IMPLANT context-list rejection", list_status, list_result);
  }
  good = list_good && good;

  Implant12Fixture offset_bad(0, 240, 0, 150);
  ++offset_bad.implant_offsets[1];
  offset_bad.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 offset_result{};
  const int offset_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &offset_bad.request, &offset_result);
  const bool offset_good =
      offset_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      offset_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN;
  if (!offset_good) {
    report_implant12_failure(
        "CUDA IMPLANT offset rejection", offset_status, offset_result);
  }
  good = offset_good && good;

  Implant12Fixture orientation_bad(0, 240, 0, 150);
  orientation_bad.edges[0] = {0, 0, 100, 0};
  orientation_bad.edges[1] = {100, 0, 100, 100};
  orientation_bad.edges[2] = {100, 100, 0, 100};
  orientation_bad.edges[3] = {0, 100, 0, 0};
  orientation_bad.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 orientation_result{};
  const int orientation_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &orientation_bad.request, &orientation_result);
  const bool orientation_good =
      orientation_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      orientation_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN;
  if (!orientation_good) {
    report_implant12_failure(
        "CUDA IMPLANT contour-orientation rejection",
        orientation_status, orientation_result);
  }
  good = orientation_good && good;

  Implant12Fixture census_bad(0, 240, 0, 150);
  ++census_bad.request.flat_gate_edge_count;
  census_bad.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 census_result{};
  const int census_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &census_bad.request, &census_result);
  const bool census_good =
      census_status == KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT &&
      census_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN;
  if (!census_good) {
    report_implant12_failure(
        "CUDA IMPLANT flat-census rejection",
        census_status, census_result);
  }
  good = census_good && good;

  Implant12Fixture capacity(0, 239, 0, 149);
  capacity.request.max_gate_candidate_work = 1;
  capacity.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 capacity_result{};
  const int capacity_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &capacity.request, &capacity_result);
  const bool capacity_good =
      capacity_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      capacity_result.disposition ==
          KLAYOUT_CUDA_SPATIAL_IMPLANT12_UNCERTAIN &&
      (capacity_result.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY) != 0 &&
      capacity_result.certified_empty_mask == 0;
  if (!capacity_good) {
    report_implant12_failure(
        "CUDA IMPLANT actual-work capacity",
        capacity_status, capacity_result);
  }
  good = capacity_good && good;

  Implant12Fixture visit_capacity(0, 240, 0, 150);
  visit_capacity.request.max_gate_query_visits = 1;
  visit_capacity.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 visit_result{};
  const int visit_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &visit_capacity.request, &visit_result);
  const bool visit_good =
      visit_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      visit_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      (visit_result.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY) != 0 &&
      visit_result.certified_empty_mask == 0;
  if (!visit_good) {
    report_implant12_failure(
        "CUDA IMPLANT query-visit capacity", visit_status, visit_result);
  }
  good = visit_good && good;

  Implant12Fixture grid_capacity(0, 240, 0, 150);
  grid_capacity.request.max_grid_cells = 1;
  grid_capacity.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 grid_result{};
  const int grid_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &grid_capacity.request, &grid_result);
  const bool grid_good =
      grid_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      grid_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      (grid_result.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY) != 0 &&
      grid_result.certified_empty_mask == 0;
  if (!grid_good) {
    report_implant12_failure(
        "CUDA IMPLANT grid capacity", grid_status, grid_result);
  }
  good = grid_good && good;

  Implant12Fixture membership_capacity(0, 240, 0, 150);
  membership_capacity.request.max_implant_memberships = 1;
  membership_capacity.refresh_digest();
  klayout_cuda_spatial_implant12_result_v1 membership_result{};
  const int membership_status =
      klayout_cuda_spatial_run_implant12_empty_v1(
          &membership_capacity.request, &membership_result);
  const bool membership_good =
      membership_status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      membership_result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK &&
      (membership_result.fallback_flags &
       KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY) != 0 &&
      membership_result.certified_empty_mask == 0;
  if (!membership_good) {
    report_implant12_failure(
        "CUDA IMPLANT membership capacity",
        membership_status, membership_result);
  }
  good = membership_good && good;

  if (good) {
    std::cout
        << "CUDA IMPLANT.1/.2 additive ABI smoke passed: atomic clean, "
           "strict per-lane/mirrored/collinear hits, endpoint exclusion, "
           "digest/order/span/capacity rejection\n";
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
  good = run_m1_all_masks_gate() && good;
  good = run_m1_boundary_context_uncertainty_gate() && good;
  good = run_m1_extrema_and_fail_closed_gate() && good;
  good = run_active3_abi_smoke() && good;
  good = run_contact4_abi_smoke() && good;
  good = run_m1ws_abi_smoke() && good;
  good = run_implant12_abi_smoke() && good;
  return good ? 0 : 1;
}
