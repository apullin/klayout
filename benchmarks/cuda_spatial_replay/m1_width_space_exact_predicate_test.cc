/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "m1_width_space_exact_predicate.h"

#include "dbEdgePairRelations.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using klayout_cuda::m1_width_space::CandidatePair;
using klayout_cuda::m1_width_space::DirectedEdge;
using klayout_cuda::m1_width_space::Rule;
using klayout_cuda::m1_width_space::Verdict;

constexpr std::int64_t kDistance =
    klayout_cuda::m1_width_space::kQualifiedSceneCoordinateDistance;
constexpr std::int64_t kM2Distance =
    klayout_cuda::m1_width_space::kM2QualifiedSceneCoordinateDistance;
constexpr std::uint64_t kPolygonA = 17;
constexpr std::uint64_t kPolygonB = 29;

DirectedEdge edge(std::int64_t x1, std::int64_t y1, std::int64_t x2,
                  std::int64_t y2) {
  return {x1, y1, x2, y2};
}

CandidatePair pair(DirectedEdge first, DirectedEdge second,
                   std::uint64_t first_polygon_id,
                   std::uint64_t second_polygon_id, Rule rule) {
  return {first, second, first_polygon_id, second_polygon_id, rule};
}

struct NamedCase {
  const char *name;
  CandidatePair candidate;
  std::int64_t distance;
  Verdict expected;
  bool compare_with_klayout;
};

Verdict klayout_edge_relation_oracle(const CandidatePair &candidate,
                                     std::int64_t distance) {
  if (candidate.rule == Rule::kWidth &&
      candidate.first_polygon_id != candidate.second_polygon_id) {
    // SinglePolygonCheck/edges_considered filters this pair before invoking
    // EdgeRelationFilter.
    return Verdict::kNoViolation;
  }

  const db::edge_relation_type relation =
      candidate.rule == Rule::kWidth ? db::WidthRelation
                                     : db::SpaceRelation;
  db::EdgeRelationFilter filter(
      relation,
      static_cast<db::EdgeRelationFilter::distance_type>(distance),
      db::Euclidian, 90.0, 0,
      std::numeric_limits<db::EdgeRelationFilter::distance_type>::max(),
      db::IncludeZeroDistanceWhenTouching);
  const db::Edge first(
      static_cast<db::Coord>(candidate.first.x1),
      static_cast<db::Coord>(candidate.first.y1),
      static_cast<db::Coord>(candidate.first.x2),
      static_cast<db::Coord>(candidate.first.y2));
  const db::Edge second(
      static_cast<db::Coord>(candidate.second.x1),
      static_cast<db::Coord>(candidate.second.y1),
      static_cast<db::Coord>(candidate.second.x2),
      static_cast<db::Coord>(candidate.second.y2));
  return filter.check(first, second, nullptr) ? Verdict::kViolation
                                              : Verdict::kNoViolation;
}

std::string describe(const CandidatePair &candidate) {
  return std::string(
             klayout_cuda::m1_width_space::rule_name(candidate.rule)) +
         " p=(" + std::to_string(candidate.first_polygon_id) + "," +
         std::to_string(candidate.second_polygon_id) + ") first=(" +
         std::to_string(candidate.first.x1) + "," +
         std::to_string(candidate.first.y1) + ")->(" +
         std::to_string(candidate.first.x2) + "," +
         std::to_string(candidate.first.y2) + ") second=(" +
         std::to_string(candidate.second.x1) + "," +
         std::to_string(candidate.second.y1) + ")->(" +
         std::to_string(candidate.second.x2) + "," +
         std::to_string(candidate.second.y2) + ")";
}

void require_verdict(const std::string &label, Verdict actual,
                     Verdict expected) {
  if (actual != expected) {
    throw std::runtime_error(
        label + ": expected " +
        klayout_cuda::m1_width_space::verdict_name(expected) + ", got " +
        klayout_cuda::m1_width_space::verdict_name(actual));
  }
}

std::vector<NamedCase> make_named_cases() {
  constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();
  const DirectedEdge east = edge(0, 0, 200, 0);

  return {
      // Strict 130-DBU boundary for METAL1.1 width.
      {"width horizontal 129",
       pair(east, edge(200, -129, 0, -129), kPolygonA, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kViolation, true},
      {"width horizontal 130",
       pair(east, edge(200, -130, 0, -130), kPolygonA, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},
      {"width horizontal 131",
       pair(east, edge(200, -131, 0, -131), kPolygonA, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},
      {"width wrong side",
       pair(east, edge(200, 129, 0, 129), kPolygonA, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},
      {"width west-facing interior",
       pair(edge(200, 0, 0, 0), edge(0, 129, 200, 129), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kViolation, true},
      {"width north-facing interior",
       pair(edge(0, 0, 0, 200), edge(129, 200, 129, 0), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kViolation, true},
      {"width south-facing interior",
       pair(edge(0, 200, 0, 0), edge(-129, 0, -129, 200), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kViolation, true},
      {"width pair order symmetry",
       pair(edge(200, -129, 0, -129), east, kPolygonA, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kViolation, true},

      // Strict 130-DBU boundary for METAL1.2 space.  Space reverses both
      // source edges, so this is the opposite side from the width cases.
      {"space horizontal 129 same polygon",
       pair(east, edge(200, 129, 0, 129), kPolygonA, kPolygonA,
            Rule::kSpace),
       kDistance, Verdict::kViolation, true},
      {"space horizontal 129 different polygons",
       pair(east, edge(200, 129, 0, 129), kPolygonA, kPolygonB,
            Rule::kSpace),
       kDistance, Verdict::kViolation, true},
      {"space horizontal 130",
       pair(east, edge(200, 130, 0, 130), kPolygonA, kPolygonB,
            Rule::kSpace),
       kDistance, Verdict::kNoViolation, true},
      {"space horizontal 131",
       pair(east, edge(200, 131, 0, 131), kPolygonA, kPolygonB,
            Rule::kSpace),
       kDistance, Verdict::kNoViolation, true},
      {"space wrong side",
       pair(east, edge(200, -129, 0, -129), kPolygonA, kPolygonB,
            Rule::kSpace),
       kDistance, Verdict::kNoViolation, true},
      {"space west-facing exterior",
       pair(edge(200, 0, 0, 0), edge(0, -129, 200, -129), kPolygonA,
            kPolygonB, Rule::kSpace),
       kDistance, Verdict::kViolation, true},
      {"space vertical",
       pair(edge(0, 200, 0, 0), edge(129, 0, 129, 200), kPolygonA,
            kPolygonB, Rule::kSpace),
       kDistance, Verdict::kViolation, true},

      // Independent strict 140-DBU METAL2.1/.2 profile.
      {"m2 width horizontal 139",
       pair(east, edge(200, -139, 0, -139), kPolygonA, kPolygonA,
            Rule::kWidth),
       kM2Distance, Verdict::kViolation, true},
      {"m2 width horizontal 140",
       pair(east, edge(200, -140, 0, -140), kPolygonA, kPolygonA,
            Rule::kWidth),
       kM2Distance, Verdict::kNoViolation, true},
      {"m2 width horizontal 141",
       pair(east, edge(200, -141, 0, -141), kPolygonA, kPolygonA,
            Rule::kWidth),
       kM2Distance, Verdict::kNoViolation, true},
      {"m2 space horizontal 139",
       pair(east, edge(200, 139, 0, 139), kPolygonA, kPolygonB,
            Rule::kSpace),
       kM2Distance, Verdict::kViolation, true},
      {"m2 space horizontal 140",
       pair(east, edge(200, 140, 0, 140), kPolygonA, kPolygonB,
            Rule::kSpace),
       kM2Distance, Verdict::kNoViolation, true},
      {"m2 corner vector inside circle",
       pair(edge(0, 0, 100, 0), edge(300, -112, 183, -112), kPolygonA,
            kPolygonA, Rule::kWidth),
       kM2Distance, Verdict::kViolation, true},
      {"m2 corner vector exact 84-112-140",
       pair(edge(0, 0, 100, 0), edge(300, -112, 184, -112), kPolygonA,
            kPolygonA, Rule::kWidth),
       kM2Distance, Verdict::kNoViolation, true},

      // The scanner does not present different polygons to WidthRelation.
      {"width different polygon metadata",
       pair(east, edge(200, -1, 0, -1), kPolygonA, kPolygonB,
            Rule::kWidth),
       kDistance, Verdict::kNoViolation, false},

      // Finite-segment Euclidean endpoint/corner distance.  50-120-130 is an
      // exact boundary even though both component gaps are below the rule.
      {"corner vector inside circle",
       pair(edge(0, 0, 100, 0), edge(200, -120, 149, -120), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kViolation, true},
      {"corner vector exact 50-120-130",
       pair(edge(0, 0, 100, 0), edge(200, -120, 150, -120), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},
      {"corner vector outside circle",
       pair(edge(0, 0, 100, 0), edge(200, -120, 151, -120), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},
      {"component gaps below threshold but hypotenuse outside",
       pair(edge(0, 0, 100, 0), edge(200, -129, 120, -129), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},

      // IncludeZeroDistanceWhenTouching behavior from include_zero_flag.
      {"collinear endpoint touch width",
       pair(edge(0, 0, 100, 0), edge(200, 0, 100, 0), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kViolation, true},
      {"collinear endpoint touch space",
       pair(edge(0, 0, 100, 0), edge(200, 0, 100, 0), kPolygonA,
            kPolygonB, Rule::kSpace),
       kDistance, Verdict::kViolation, true},
      {"collinear overlap",
       pair(edge(0, 0, 100, 0), edge(150, 0, 50, 0), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kViolation, true},
      {"collinear one-DBU gap excluded",
       pair(edge(0, 0, 100, 0), edge(200, 0, 101, 0), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},
      {"perpendicular shared corner ignored at 90 degrees",
       pair(edge(0, 0, 100, 0), edge(100, 100, 100, 0), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},
      {"same direction ignored",
       pair(edge(0, 0, 100, 0), edge(0, -1, 100, -1), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kNoViolation, true},

      // Explicit fail-closed boundaries.
      {"diagonal edges unsupported",
       pair(edge(0, 0, 200, 200), edge(200, 71, 0, -129), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kUncertain, false},
      {"axial versus diagonal unsupported",
       pair(east, edge(200, -129, 0, -128), kPolygonA, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kUncertain, false},
      {"degenerate edge unsupported",
       pair(edge(0, 0, 0, 0), edge(0, 1, 100, 1), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kUncertain, false},
      {"unknown first polygon",
       pair(east, edge(200, -1, 0, -1),
            klayout_cuda::m1_width_space::kUnknownPolygonId, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kUncertain, false},
      {"unknown second polygon",
       pair(east, edge(200, 1, 0, 1), kPolygonA,
            klayout_cuda::m1_width_space::kUnknownPolygonId, Rule::kSpace),
       kDistance, Verdict::kUncertain, false},
      {"invalid relation",
       pair(east, edge(200, -1, 0, -1), kPolygonA, kPolygonA,
            static_cast<Rule>(99)),
       kDistance, Verdict::kUncertain, false},
      {"unqualified distance 129",
       pair(east, edge(200, -1, 0, -1), kPolygonA, kPolygonA,
            Rule::kWidth),
       129, Verdict::kUncertain, false},
      {"unqualified distance 131",
       pair(east, edge(200, -1, 0, -1), kPolygonA, kPolygonA,
            Rule::kWidth),
       131, Verdict::kUncertain, false},
      {"unsafe full signed x span",
       pair(edge(lo, 0, hi, 0), edge(hi, -129, lo, -129), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kUncertain, false},
      {"unsafe full signed y span",
       pair(edge(0, lo, 0, hi), edge(129, hi, 129, lo), kPolygonA,
            kPolygonA, Rule::kWidth),
       kDistance, Verdict::kUncertain, false},
      {"near INT64_MIN remains exact",
       pair(edge(lo + 100, 0, lo + 300, 0),
            edge(lo + 300, -129, lo + 100, -129), kPolygonA, kPolygonA,
            Rule::kWidth),
       kDistance, Verdict::kViolation, false},
      {"near INT64_MAX remains exact",
       pair(edge(hi - 300, 0, hi - 100, 0),
            edge(hi - 100, 129, hi - 300, 129), kPolygonA, kPolygonB,
            Rule::kSpace),
       kDistance, Verdict::kViolation, false},
  };
}

void test_named_cases(std::vector<CandidatePair> *m1_batch,
                      std::vector<CandidatePair> *m2_batch,
                      std::uint64_t *checked_count) {
  for (const NamedCase &test : make_named_cases()) {
    const Verdict host =
        klayout_cuda::m1_width_space::classify_pair_bounded(
            test.candidate, test.distance);
    require_verdict(test.name, host, test.expected);
    if (test.compare_with_klayout) {
      const Verdict source =
          klayout_edge_relation_oracle(test.candidate, test.distance);
      require_verdict(std::string(test.name) + " KLayout source oracle", host,
                      source);
    }
    if (test.distance == kDistance) {
      m1_batch->push_back(test.candidate);
    } else if (test.distance == kM2Distance) {
      m2_batch->push_back(test.candidate);
    }
    ++*checked_count;
  }
}

void test_random_source_differential(
    std::vector<CandidatePair> *batch, std::int64_t distance,
    std::uint64_t seed, std::uint64_t *checked_count) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::int64_t> coordinate(-2000, 2000);
  std::uniform_int_distribution<std::int64_t> length(1, 500);
  std::uniform_int_distribution<std::int64_t> displacement(-220, 220);
  std::uniform_int_distribution<int> bit(0, 1);
  constexpr std::size_t kCaseCount = 50000;

  batch->reserve(batch->size() + kCaseCount);
  for (std::size_t i = 0; i < kCaseCount; ++i) {
    const bool horizontal = bit(rng) != 0;
    const bool perpendicular = (rng() % 7) == 0;
    const bool first_positive = bit(rng) != 0;
    const bool second_positive = bit(rng) != 0;
    const Rule rule = bit(rng) != 0 ? Rule::kWidth : Rule::kSpace;
    const bool same_polygon = bit(rng) != 0;

    const std::int64_t base_x = coordinate(rng);
    const std::int64_t base_y = coordinate(rng);
    const std::int64_t first_length = length(rng);
    const std::int64_t second_length = length(rng);
    const std::int64_t along = displacement(rng);
    const std::int64_t across = displacement(rng);

    DirectedEdge first;
    DirectedEdge second;
    if (horizontal) {
      first = first_positive
                  ? edge(base_x, base_y, base_x + first_length, base_y)
                  : edge(base_x + first_length, base_y, base_x, base_y);
      if (perpendicular) {
        second = second_positive
                     ? edge(base_x + along, base_y + across,
                            base_x + along, base_y + across + second_length)
                     : edge(base_x + along, base_y + across + second_length,
                            base_x + along, base_y + across);
      } else {
        second = second_positive
                     ? edge(base_x + along, base_y + across,
                            base_x + along + second_length, base_y + across)
                     : edge(base_x + along + second_length, base_y + across,
                            base_x + along, base_y + across);
      }
    } else {
      first = first_positive
                  ? edge(base_x, base_y, base_x, base_y + first_length)
                  : edge(base_x, base_y + first_length, base_x, base_y);
      if (perpendicular) {
        second = second_positive
                     ? edge(base_x + across, base_y + along,
                            base_x + across + second_length, base_y + along)
                     : edge(base_x + across + second_length, base_y + along,
                            base_x + across, base_y + along);
      } else {
        second = second_positive
                     ? edge(base_x + across, base_y + along,
                            base_x + across, base_y + along + second_length)
                     : edge(base_x + across, base_y + along + second_length,
                            base_x + across, base_y + along);
      }
    }

    const CandidatePair candidate =
        pair(first, second, kPolygonA,
             same_polygon ? kPolygonA : kPolygonB, rule);
    const Verdict host =
        klayout_cuda::m1_width_space::classify_pair_bounded(candidate,
                                                            distance);
    const Verdict source =
        klayout_edge_relation_oracle(candidate, distance);
    if (host != source) {
      throw std::runtime_error(
          "random KLayout differential mismatch at index " +
          std::to_string(i) + ": " + describe(candidate) + ", predicate=" +
          klayout_cuda::m1_width_space::verdict_name(host) + ", source=" +
          klayout_cuda::m1_width_space::verdict_name(source));
    }
    batch->push_back(candidate);
    ++*checked_count;
  }
}

void test_cuda_parity(const std::vector<CandidatePair> &batch,
                      std::int64_t distance, const char *profile,
                      std::uint64_t *checked_count) {
  std::vector<Verdict> device(batch.size());
  std::string error;
  if (!klayout_cuda::m1_width_space::classify_batch(
          batch.data(), batch.size(), distance, device.data(), &error)) {
    throw std::runtime_error(std::string(profile) +
                             " CUDA batch failed: " + error);
  }
  for (std::size_t i = 0; i < batch.size(); ++i) {
    const Verdict host =
        klayout_cuda::m1_width_space::classify_pair_bounded(batch[i],
                                                            distance);
    if (device[i] != host) {
      throw std::runtime_error(
          "host/device mismatch at index " + std::to_string(i) + ": " +
          describe(batch[i]) + ", host=" +
          klayout_cuda::m1_width_space::verdict_name(host) + ", device=" +
          klayout_cuda::m1_width_space::verdict_name(device[i]));
    }
  }
  *checked_count += batch.size();
}

void test_invalid_distance(std::uint64_t *checked_count) {
  std::string error;
  const CandidatePair invalid_distance_pair =
      pair(edge(0, 0, 100, 0), edge(100, -1, 0, -1), kPolygonA,
           kPolygonA, Rule::kWidth);
  for (const std::int64_t invalid_distance : std::array<std::int64_t, 4>{
           129, 131, 139, 141}) {
    require_verdict(
        "host invalid distance",
        klayout_cuda::m1_width_space::classify_pair_bounded(
            invalid_distance_pair, invalid_distance),
        Verdict::kUncertain);
    Verdict device_verdict = Verdict::kNoViolation;
    if (!klayout_cuda::m1_width_space::classify_batch(
            &invalid_distance_pair, 1, invalid_distance, &device_verdict,
            &error)) {
      throw std::runtime_error("CUDA invalid-distance batch failed: " +
                               error);
    }
    require_verdict("device invalid distance", device_verdict,
                    Verdict::kUncertain);
    ++*checked_count;
  }
}

void test_batch_contract(std::uint64_t *checked_count) {
  std::string error = "not cleared";
  if (!klayout_cuda::m1_width_space::classify_batch(
          nullptr, 0, kDistance, nullptr, &error)) {
    throw std::runtime_error("zero-sized batch unexpectedly failed");
  }
  if (!error.empty()) {
    throw std::runtime_error("zero-sized batch did not clear error");
  }
  Verdict output = Verdict::kNoViolation;
  if (klayout_cuda::m1_width_space::classify_batch(
          nullptr, 1, kDistance, &output, &error) ||
      error != "null batch input or output") {
    throw std::runtime_error("null-input contract was not enforced");
  }
  ++*checked_count;
}

}  // namespace

int main() {
  try {
    std::uint64_t checked_count = 0;
    std::vector<CandidatePair> m1_batch;
    std::vector<CandidatePair> m2_batch;
    test_named_cases(&m1_batch, &m2_batch, &checked_count);
    test_random_source_differential(
        &m1_batch, kDistance, UINT64_C(0x4d31574944544853),
        &checked_count);
    test_random_source_differential(
        &m2_batch, kM2Distance, UINT64_C(0x4d32574944544853),
        &checked_count);
    test_cuda_parity(m1_batch, kDistance, "M1", &checked_count);
    test_cuda_parity(m2_batch, kM2Distance, "M2", &checked_count);
    test_invalid_distance(&checked_count);
    test_batch_contract(&checked_count);
    std::cout << "m1 width/space exact predicate: PASS (" << checked_count
              << " checks; " << (m1_batch.size() + m2_batch.size())
              << " KLayout-source differential/device pairs)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "m1 width/space exact predicate: FAIL: " << error.what()
              << "\n";
    return 1;
  }
}
