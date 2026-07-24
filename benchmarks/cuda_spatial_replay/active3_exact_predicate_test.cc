/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "active3_exact_predicate.h"

#include "dbEdgePairRelations.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using klayout_cuda::active3::DirectedEdge;
using klayout_cuda::active3::EdgePair;
using klayout_cuda::active3::Verdict;

constexpr std::int64_t kActive3Distance =
    klayout_cuda::active3::kQualifiedSceneCoordinateDistance;
constexpr std::int64_t kContact4Distance =
    klayout_cuda::active3::kContact4QualifiedSceneCoordinateDistance;
constexpr std::uint64_t kSignBit = UINT64_C(1) << 63;

std::uint64_t ordered_key(std::int64_t value) {
  return static_cast<std::uint64_t>(value) ^ kSignBit;
}

std::uint64_t coordinate_gap(std::int64_t a, std::int64_t b) {
  const std::uint64_t ak = ordered_key(a);
  const std::uint64_t bk = ordered_key(b);
  return ak < bk ? bk - ak : ak - bk;
}

std::uint64_t point_interval_gap(std::int64_t point, std::int64_t first,
                                 std::int64_t second) {
  if (second < first) {
    std::swap(first, second);
  }
  if (point < first) {
    return coordinate_gap(point, first);
  }
  if (point > second) {
    return coordinate_gap(second, point);
  }
  return 0;
}

bool intervals_touch(std::int64_t a0, std::int64_t a1, std::int64_t b0,
                     std::int64_t b1) {
  if (a1 < a0) {
    std::swap(a0, a1);
  }
  if (b1 < b0) {
    std::swap(b0, b1);
  }
  return !(a1 < b0 || b1 < a0);
}

bool source_coordinate_differences_are_safe(const EdgePair &pair) {
  const std::array<std::int64_t, 4> xs = {
      pair.well.x1, pair.well.x2, pair.active.x1, pair.active.x2};
  const std::array<std::int64_t, 4> ys = {
      pair.well.y1, pair.well.y2, pair.active.y1, pair.active.y2};
  const auto xbounds = std::minmax_element(xs.begin(), xs.end());
  const auto ybounds = std::minmax_element(ys.begin(), ys.end());
  return coordinate_gap(*xbounds.first, *xbounds.second) <=
             static_cast<std::uint64_t>(INT64_MAX) &&
         coordinate_gap(*ybounds.first, *ybounds.second) <=
             static_cast<std::uint64_t>(INT64_MAX);
}

bool squared_distance_less(std::uint64_t dx, std::uint64_t dy,
                           std::uint64_t distance) {
  if (dx >= distance || dy >= distance) {
    return false;
  }
  return dx * dx + dy * dy < distance * distance;
}

// Independent exact distance oracle: unlike the CUDA classifier's single
// perpendicular-gap/projection-gap identity, this projects each of the four
// endpoints onto the opposite finite segment and tests their distances.  All
// coordinate differences are order-key differences, so INT64_MIN..INT64_MAX
// cannot overflow; both fixed profile distances make the squared sums exact.
bool point_segment_distance_less_than(std::int64_t x, std::int64_t y,
                                      const DirectedEdge &segment,
                                      std::uint64_t distance) {
  std::uint64_t dx = 0;
  std::uint64_t dy = 0;
  if (segment.y1 == segment.y2) {
    dx = point_interval_gap(x, segment.x1, segment.x2);
    dy = coordinate_gap(y, segment.y1);
  } else {
    dx = coordinate_gap(x, segment.x1);
    dy = point_interval_gap(y, segment.y1, segment.y2);
  }
  return squared_distance_less(dx, dy, distance);
}

Verdict oracle(const EdgePair &pair, std::int64_t distance) {
  const DirectedEdge &a = pair.well;
  const DirectedEdge &b = pair.active;
  const bool a_horizontal = a.y1 == a.y2 && a.x1 != a.x2;
  const bool a_vertical = a.x1 == a.x2 && a.y1 != a.y2;
  const bool b_horizontal = b.y1 == b.y2 && b.x1 != b.x2;
  const bool b_vertical = b.x1 == b.x2 && b.y1 != b.y2;
  if (!(a_horizontal || a_vertical) || !(b_horizontal || b_vertical)) {
    return Verdict::kUncertain;
  }
  if (!source_coordinate_differences_are_safe(pair) ||
      (distance != kActive3Distance && distance != kContact4Distance)) {
    return Verdict::kUncertain;
  }

  // This is the source EdgeRelationFilter angle criterion after
  // OverlapRelation reverses its temporary copy of the first edge.
  if (a_horizontal != b_horizontal) {
    return Verdict::kNoViolation;
  }

  const bool a_positive =
      a_horizontal ? a.x2 > a.x1 : a.y2 > a.y1;
  const bool b_positive =
      b_horizontal ? b.x2 > b.x1 : b.y2 > b.y1;
  if (a_positive != b_positive) {
    return Verdict::kNoViolation;
  }

  const bool collinear =
      a_horizontal ? a.y1 == b.y1 : a.x1 == b.x1;
  if (collinear) {
    const bool touching =
        a_horizontal
            ? intervals_touch(a.x1, a.x2, b.x1, b.x2)
            : intervals_touch(a.y1, a.y2, b.y1, b.y2);
    return touching ? Verdict::kViolation : Verdict::kNoViolation;
  }

  const bool active_is_right =
      a_horizontal
          ? (a_positive ? b.y1 < a.y1 : b.y1 > a.y1)
          : (a_positive ? b.x1 > a.x1 : b.x1 < a.x1);
  if (!active_is_right) {
    return Verdict::kNoViolation;
  }

  const std::uint64_t d = static_cast<std::uint64_t>(distance);
  const bool close =
      point_segment_distance_less_than(a.x1, a.y1, b, d) ||
      point_segment_distance_less_than(a.x2, a.y2, b, d) ||
      point_segment_distance_less_than(b.x1, b.y1, a, d) ||
      point_segment_distance_less_than(b.x2, b.y2, a, d);
  return close ? Verdict::kViolation : Verdict::kNoViolation;
}

Verdict klayout_oracle(const EdgePair &pair, std::int64_t distance) {
  db::EdgeRelationFilter filter(
      db::OverlapRelation,
      static_cast<db::EdgeRelationFilter::distance_type>(distance),
      db::Euclidian, 90.0, 0,
      std::numeric_limits<db::EdgeRelationFilter::distance_type>::max(),
      db::IncludeZeroDistanceWhenTouching);
  const db::Edge first(
      static_cast<db::Coord>(pair.well.x1),
      static_cast<db::Coord>(pair.well.y1),
      static_cast<db::Coord>(pair.well.x2),
      static_cast<db::Coord>(pair.well.y2));
  const db::Edge second(
      static_cast<db::Coord>(pair.active.x1),
      static_cast<db::Coord>(pair.active.y1),
      static_cast<db::Coord>(pair.active.x2),
      static_cast<db::Coord>(pair.active.y2));
  return filter.check(first, second, nullptr) ? Verdict::kViolation
                                              : Verdict::kNoViolation;
}

struct NamedCase {
  std::string name;
  EdgePair pair;
  std::int64_t distance;
  Verdict expected;
};

DirectedEdge edge(std::int64_t x1, std::int64_t y1, std::int64_t x2,
                  std::int64_t y2) {
  return {x1, y1, x2, y2};
}

EdgePair pair(DirectedEdge well, DirectedEdge active) {
  return {well, active};
}

void run_batch_and_compare(const std::vector<EdgePair> &pairs,
                           std::int64_t distance,
                           std::uint64_t *checked_count) {
  std::vector<Verdict> device(pairs.size());
  std::string error;
  if (!klayout_cuda::active3::classify_batch(
          pairs.data(), pairs.size(), distance, device.data(), &error)) {
    throw std::runtime_error("CUDA batch failed: " + error);
  }
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const Verdict expected = oracle(pairs[i], distance);
    if (device[i] != expected) {
      throw std::runtime_error(
          "oracle mismatch at batch index " + std::to_string(i) +
          ", d=" + std::to_string(distance) + ": GPU=" +
          klayout_cuda::active3::verdict_name(device[i]) + ", oracle=" +
          klayout_cuda::active3::verdict_name(expected));
    }
    if (expected != Verdict::kUncertain) {
      const Verdict klayout = klayout_oracle(pairs[i], distance);
      if (device[i] != klayout) {
        throw std::runtime_error(
            "KLayout EdgeRelationFilter mismatch at batch index " +
            std::to_string(i) + ", d=" + std::to_string(distance) +
            ": GPU=" +
            klayout_cuda::active3::verdict_name(device[i]) +
            ", KLayout=" +
            klayout_cuda::active3::verdict_name(klayout));
      }
    }
  }
  *checked_count += pairs.size();
}

void test_named_cases(std::uint64_t *checked_count) {
  constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();
  const DirectedEdge east = edge(0, 0, 100, 0);

  const std::vector<NamedCase> cases = {
      {"east/right/d-1 endpoint",
       pair(east, edge(100, -109, 200, -109)), kActive3Distance,
       Verdict::kViolation},
      {"east/right/d endpoint",
       pair(east, edge(100, -110, 200, -110)), kActive3Distance,
       Verdict::kNoViolation},
      {"east/right/d+1 endpoint",
       pair(east, edge(100, -111, 200, -111)), kActive3Distance,
       Verdict::kNoViolation},
      {"wrong half-plane", pair(east, edge(0, 109, 100, 109)),
       kActive3Distance,
       Verdict::kNoViolation},
      {"opposite direction", pair(east, edge(100, -1, 0, -1)),
       kActive3Distance,
       Verdict::kNoViolation},
      {"perpendicular", pair(east, edge(50, -10, 50, 10)),
       kActive3Distance,
       Verdict::kNoViolation},
      {"west/right",
       pair(edge(100, 0, 0, 0), edge(100, 109, 0, 109)),
       kActive3Distance,
       Verdict::kViolation},
      {"north/right",
       pair(edge(0, 0, 0, 100), edge(109, 0, 109, 100)),
       kActive3Distance,
       Verdict::kViolation},
      {"south/right",
       pair(edge(0, 100, 0, 0), edge(-109, 100, -109, 0)),
       kActive3Distance,
       Verdict::kViolation},
      {"projection overlap", pair(east, edge(50, -109, 150, -109)),
       kActive3Distance,
       Verdict::kViolation},
      {"projection gap inside circle",
       pair(east, edge(101, -109, 200, -109)), kActive3Distance,
       Verdict::kViolation},
      {"3-4-5 boundary minus one",
       pair(east, edge(187, -66, 250, -66)), kActive3Distance,
       Verdict::kViolation},
      {"3-4-5 exact boundary",
       pair(east, edge(188, -66, 250, -66)), kActive3Distance,
       Verdict::kNoViolation},
      {"3-4-5 boundary plus one",
       pair(east, edge(189, -66, 250, -66)), kActive3Distance,
       Verdict::kNoViolation},
      {"collinear endpoint touch", pair(east, edge(100, 0, 200, 0)),
       kActive3Distance,
       Verdict::kViolation},
      {"collinear overlap", pair(east, edge(50, 0, 150, 0)),
       kActive3Distance,
       Verdict::kViolation},
      {"identical directed edges", pair(east, east), kActive3Distance,
       Verdict::kViolation},
      {"collinear opposite direction", pair(east, edge(100, 0, 0, 0)),
       kActive3Distance,
       Verdict::kNoViolation},
      {"collinear gap is excluded", pair(east, edge(101, 0, 200, 0)),
       kActive3Distance,
       Verdict::kNoViolation},
      {"negative coordinates",
       pair(edge(-200, -100, -100, -100),
            edge(-150, -209, -50, -209)),
       kActive3Distance, Verdict::kViolation},
      {"full signed range projection",
       pair(edge(lo, 0, hi, 0), edge(lo, -109, hi, -109)),
       kActive3Distance,
       Verdict::kUncertain},
      {"full signed range axial gap",
       pair(edge(hi - 100, 0, hi, 0), edge(lo, -1, lo + 100, -1)),
       kActive3Distance,
       Verdict::kUncertain},
      {"full signed vertical range",
       pair(edge(0, lo, 0, hi), edge(109, lo, 109, hi)),
       kActive3Distance,
       Verdict::kUncertain},
      {"near INT64_MIN remains exact",
       pair(edge(lo + 100, -100, lo + 200, -100),
            edge(lo + 100, -209, lo + 200, -209)),
       kActive3Distance, Verdict::kViolation},
      {"near INT64_MAX remains exact",
       pair(edge(hi - 200, 100, hi - 100, 100),
            edge(hi - 200, -9, hi - 100, -9)),
       kActive3Distance, Verdict::kViolation},
      {"degenerate is fail-closed",
       pair(edge(0, 0, 0, 0), edge(0, 0, 100, 0)), kActive3Distance,
       Verdict::kUncertain},
      {"diagonal is fail-closed",
       pair(edge(0, 0, 100, 100), edge(0, -1, 100, 99)),
       kActive3Distance,
       Verdict::kUncertain},
      {"CONTACT.4 parallel gap d-1",
       pair(east, edge(0, -9, 100, -9)), kContact4Distance,
       Verdict::kViolation},
      {"CONTACT.4 parallel gap d",
       pair(east, edge(0, -10, 100, -10)), kContact4Distance,
       Verdict::kNoViolation},
      {"CONTACT.4 endpoint 6/7 inside",
       pair(east, edge(106, -7, 200, -7)), kContact4Distance,
       Verdict::kViolation},
      {"CONTACT.4 endpoint 6/8 boundary",
       pair(east, edge(106, -8, 200, -8)), kContact4Distance,
       Verdict::kNoViolation},
      {"CONTACT.4 reversed relation order is distinct",
       pair(edge(0, -9, 100, -9), east), kContact4Distance,
       Verdict::kNoViolation},
      {"CONTACT.4 collinear touch",
       pair(east, edge(100, 0, 200, 0)), kContact4Distance,
       Verdict::kViolation},
      {"CONTACT.4 collinear separated",
       pair(east, edge(101, 0, 200, 0)), kContact4Distance,
       Verdict::kNoViolation},
      {"non-ACTIVE.3 distance is fail-closed",
       pair(east, edge(0, -1, 100, -1)), 111,
       Verdict::kUncertain},
      {"physical nanometres are not coordinate DBU",
       pair(east, edge(0, -1, 100, -1)), 55,
       Verdict::kUncertain},
      {"zero distance configuration is fail-closed",
       pair(east, edge(0, -1, 100, -1)), 0, Verdict::kUncertain},
      {"negative distance configuration is fail-closed",
       pair(east, edge(0, -1, 100, -1)), -1,
       Verdict::kUncertain},
  };

  for (const NamedCase &test : cases) {
    const Verdict expected_by_oracle = oracle(test.pair, test.distance);
    if (expected_by_oracle != test.expected) {
      throw std::runtime_error("bad named oracle expectation: " + test.name);
    }
    std::vector<Verdict> actual(1);
    std::string error;
    if (!klayout_cuda::active3::classify_batch(
            &test.pair, 1, test.distance, actual.data(), &error)) {
      throw std::runtime_error("CUDA named case failed: " + error);
    }
    if (actual[0] != test.expected) {
      throw std::runtime_error(
          "named case '" + test.name + "': expected " +
          klayout_cuda::active3::verdict_name(test.expected) + ", got " +
          klayout_cuda::active3::verdict_name(actual[0]));
    }
    ++*checked_count;
  }
}

void test_api_bounds() {
  EdgePair dummy_pair{};
  Verdict dummy_result = Verdict::kUncertain;
  std::string error;
  const std::size_t overflowing_count =
      std::numeric_limits<std::size_t>::max() / sizeof(EdgePair) + 1;
  if (klayout_cuda::active3::classify_batch(
          &dummy_pair, overflowing_count, kActive3Distance, &dummy_result,
          &error) ||
      error != "batch byte-size overflow") {
    throw std::runtime_error("batch byte-size overflow was not rejected");
  }
  error = "not cleared";
  if (!klayout_cuda::active3::classify_batch(
          nullptr, 0, kActive3Distance, nullptr, &error) ||
      !error.empty()) {
    throw std::runtime_error("empty batch contract failed");
  }
}

std::vector<DirectedEdge> exhaustive_edges() {
  std::vector<DirectedEdge> edges;
  for (std::int64_t fixed = -2; fixed <= 2; ++fixed) {
    for (std::int64_t first = -2; first <= 2; ++first) {
      for (std::int64_t second = -2; second <= 2; ++second) {
        if (first == second) {
          continue;
        }
        edges.push_back(edge(first, fixed, second, fixed));
        edges.push_back(edge(fixed, first, fixed, second));
      }
    }
  }
  return edges;
}

void test_exhaustive_lattice(std::uint64_t *checked_count) {
  const std::vector<DirectedEdge> edges = exhaustive_edges();
  std::vector<EdgePair> pairs;
  pairs.reserve(edges.size() * edges.size());
  for (const DirectedEdge &well : edges) {
    for (const DirectedEdge &active : edges) {
      pairs.push_back(pair(well, active));
    }
  }
  run_batch_and_compare(pairs, kActive3Distance, checked_count);
  run_batch_and_compare(pairs, kContact4Distance, checked_count);
}

std::int64_t random_coordinate(std::mt19937_64 *random) {
  static constexpr std::array<std::int64_t, 5> anchors = {
      std::numeric_limits<std::int64_t>::min() + 4096, -1000000, 0,
      1000000, std::numeric_limits<std::int64_t>::max() - 4096};
  const std::int64_t anchor = anchors[(*random)() % anchors.size()];
  const std::int64_t delta =
      static_cast<std::int64_t>((*random)() % 4097) - 2048;
  return anchor + delta;
}

DirectedEdge random_edge(std::mt19937_64 *random) {
  const std::int64_t x = random_coordinate(random);
  const std::int64_t y = random_coordinate(random);
  const std::int64_t span =
      static_cast<std::int64_t>((*random)() % 1024) + 1;
  const bool reverse = ((*random)() & 1U) != 0;
  const bool horizontal = ((*random)() & 1U) != 0;

  // Anchors leave enough headroom for this bounded span.
  if (horizontal) {
    return reverse ? edge(x + span, y, x, y) : edge(x, y, x + span, y);
  }
  return reverse ? edge(x, y + span, x, y) : edge(x, y, x, y + span);
}

void test_random(std::uint64_t *checked_count) {
  std::mt19937_64 random(UINT64_C(0x41c71e3a5eed));
  static constexpr std::array<std::int64_t, 2> distances = {
      kActive3Distance, kContact4Distance};

  for (std::int64_t distance : distances) {
    std::vector<EdgePair> pairs;
    pairs.reserve(200000);
    for (std::size_t i = 0; i < 200000; ++i) {
      DirectedEdge well = random_edge(&random);
      DirectedEdge active = random_edge(&random);

      // Ensure a substantial population close enough to exercise the exact
      // positive and threshold paths, rather than only trivially distant
      // pairs around independent int64 anchors.
      if ((i % 2) == 0) {
        const std::int64_t shift =
            static_cast<std::int64_t>(random() % 129) - 64;
        if (well.y1 == well.y2) {
          const std::int64_t ax0 = std::min(well.x1, well.x2);
          const std::int64_t ax1 = std::max(well.x1, well.x2);
          const std::int64_t span = ax1 - ax0;
          const bool positive = well.x2 > well.x1;
          const std::int64_t by = well.y1 + shift;
          active = positive ? edge(ax0, by, ax0 + span, by)
                            : edge(ax1, by, ax1 - span, by);
        } else {
          const std::int64_t ay0 = std::min(well.y1, well.y2);
          const std::int64_t ay1 = std::max(well.y1, well.y2);
          const std::int64_t span = ay1 - ay0;
          const bool positive = well.y2 > well.y1;
          const std::int64_t bx = well.x1 + shift;
          active = positive ? edge(bx, ay0, bx, ay0 + span)
                            : edge(bx, ay1, bx, ay1 - span);
        }
      }

      // Ten percent deliberately leave the supported geometry domain.
      if ((i % 10) == 0) {
        active.x2 = active.x1 + 1;
        active.y2 = active.y1 + 1;
      } else if ((i % 10) == 1) {
        active.x2 = active.x1;
        active.y2 = active.y1;
      }
      pairs.push_back(pair(well, active));
    }
    run_batch_and_compare(pairs, distance, checked_count);
  }
}

}  // namespace

int main() {
  try {
    std::uint64_t checked_count = 0;
    test_api_bounds();
    test_named_cases(&checked_count);
    test_exhaustive_lattice(&checked_count);
    test_random(&checked_count);
    std::cout << "ACTIVE.3/CONTACT.4 exact predicate: PASS (" << checked_count
              << " GPU/oracle/KLayout EdgeRelationFilter classifications)"
              << std::endl;
    std::cout << "exact domain: directed nondegenerate Manhattan edges, "
                 "d=110 DBU ACTIVE.3 or d=10 DBU CONTACT.4 "
                 "(0.5 nm/DBU)"
              << std::endl;
    std::cout << "fallback domain: degenerate/diagonal edges, unsafe signed "
                 "coordinate differences, and other configurations"
              << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ACTIVE.3/CONTACT.4 exact predicate: FAIL: " << error.what()
              << std::endl;
    return 1;
  }
}
