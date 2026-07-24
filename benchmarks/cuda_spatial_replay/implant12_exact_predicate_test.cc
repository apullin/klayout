/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "implant12_exact_predicate.h"

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

using klayout_cuda::implant12::DirectedEdge;
using klayout_cuda::implant12::EdgePair;
using klayout_cuda::implant12::Verdict;

constexpr std::int64_t kImplant1 =
    klayout_cuda::implant12::kImplant1Distance;
constexpr std::int64_t kImplant2 =
    klayout_cuda::implant12::kImplant2Distance;
constexpr std::uint64_t kSignBit = UINT64_C(1) << 63;

std::uint64_t ordered_key(std::int64_t value) {
  return static_cast<std::uint64_t>(value) ^ kSignBit;
}

std::uint64_t gap(std::int64_t a, std::int64_t b) {
  const std::uint64_t ak = ordered_key(a);
  const std::uint64_t bk = ordered_key(b);
  return ak < bk ? bk - ak : ak - bk;
}

bool safe_source_differences(const EdgePair &pair) {
  const std::array<std::int64_t, 4> xs = {
      pair.implant.x1, pair.implant.x2,
      pair.secondary.x1, pair.secondary.x2};
  const std::array<std::int64_t, 4> ys = {
      pair.implant.y1, pair.implant.y2,
      pair.secondary.y1, pair.secondary.y2};
  const auto xb = std::minmax_element(xs.begin(), xs.end());
  const auto yb = std::minmax_element(ys.begin(), ys.end());
  return gap(*xb.first, *xb.second) <=
             static_cast<std::uint64_t>(INT64_MAX) &&
         gap(*yb.first, *yb.second) <=
             static_cast<std::uint64_t>(INT64_MAX);
}

Verdict independent_oracle(const EdgePair &pair, std::int64_t distance) {
  const DirectedEdge &a = pair.implant;
  const DirectedEdge &b = pair.secondary;
  const bool ah = a.y1 == a.y2 && a.x1 != a.x2;
  const bool av = a.x1 == a.x2 && a.y1 != a.y2;
  const bool bh = b.y1 == b.y2 && b.x1 != b.x2;
  const bool bv = b.x1 == b.x2 && b.y1 != b.y2;
  if (!(ah || av) || !(bh || bv) ||
      !safe_source_differences(pair) ||
      (distance != kImplant1 && distance != kImplant2)) {
    return Verdict::kUncertain;
  }
  if (ah != bh) return Verdict::kNoViolation;

  const std::int64_t ad = ah ? a.x2 - a.x1 : a.y2 - a.y1;
  const std::int64_t bd = bh ? b.x2 - b.x1 : b.y2 - b.y1;
  if ((ad > 0) == (bd > 0)) return Verdict::kNoViolation;

  const std::int64_t alo = ah ? std::min(a.x1, a.x2)
                              : std::min(a.y1, a.y2);
  const std::int64_t ahi = ah ? std::max(a.x1, a.x2)
                              : std::max(a.y1, a.y2);
  const std::int64_t blo = bh ? std::min(b.x1, b.x2)
                              : std::min(b.y1, b.y2);
  const std::int64_t bhi = bh ? std::max(b.x1, b.x2)
                              : std::max(b.y1, b.y2);
  if (std::max(alo, blo) >= std::min(ahi, bhi)) {
    return Verdict::kNoViolation;
  }

  const std::int64_t aline = ah ? a.y1 : a.x1;
  const std::int64_t bline = bh ? b.y1 : b.x1;
  const std::uint64_t separation = gap(aline, bline);
  if (separation >= static_cast<std::uint64_t>(distance)) {
    return Verdict::kNoViolation;
  }
  if (separation == 0) return Verdict::kViolation;

  const bool exterior =
      ah ? (ad > 0 ? bline > aline : bline < aline)
         : (ad > 0 ? bline < aline : bline > aline);
  return exterior ? Verdict::kViolation : Verdict::kNoViolation;
}

Verdict klayout_oracle(const EdgePair &pair, std::int64_t distance) {
  db::EdgeRelationFilter filter(
      db::SpaceRelation,
      static_cast<db::EdgeRelationFilter::distance_type>(distance),
      db::Projection, 90.0, 0,
      std::numeric_limits<db::EdgeRelationFilter::distance_type>::max(),
      db::IncludeZeroDistanceWhenTouching);
  const db::Edge first(
      static_cast<db::Coord>(pair.implant.x1),
      static_cast<db::Coord>(pair.implant.y1),
      static_cast<db::Coord>(pair.implant.x2),
      static_cast<db::Coord>(pair.implant.y2));
  const db::Edge second(
      static_cast<db::Coord>(pair.secondary.x1),
      static_cast<db::Coord>(pair.secondary.y1),
      static_cast<db::Coord>(pair.secondary.x2),
      static_cast<db::Coord>(pair.secondary.y2));
  return filter.check(first, second, nullptr) ? Verdict::kViolation
                                              : Verdict::kNoViolation;
}

DirectedEdge edge(std::int64_t x1, std::int64_t y1,
                  std::int64_t x2, std::int64_t y2) {
  return {x1, y1, x2, y2};
}

EdgePair pair(DirectedEdge implant, DirectedEdge secondary) {
  return {implant, secondary};
}

struct NamedCase {
  const char *name;
  EdgePair edges;
  std::int64_t distance;
  Verdict expected;
};

void run_and_compare(const std::vector<EdgePair> &pairs,
                     std::int64_t distance, std::uint64_t *checked) {
  std::vector<Verdict> gpu(pairs.size());
  std::string error;
  if (!klayout_cuda::implant12::classify_batch(
          pairs.data(), pairs.size(), distance, gpu.data(), &error)) {
    throw std::runtime_error("CUDA batch failed: " + error);
  }
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const Verdict expected = independent_oracle(pairs[i], distance);
    if (gpu[i] != expected) {
      throw std::runtime_error(
          "independent-oracle mismatch at " + std::to_string(i) +
          ", d=" + std::to_string(distance) + ": GPU=" +
          klayout_cuda::implant12::verdict_name(gpu[i]) +
          ", oracle=" +
          klayout_cuda::implant12::verdict_name(expected));
    }
    if (expected != Verdict::kUncertain) {
      const Verdict direct = klayout_oracle(pairs[i], distance);
      if (gpu[i] != direct) {
        throw std::runtime_error(
            "direct KLayout mismatch at " + std::to_string(i) +
            ", d=" + std::to_string(distance) + ": GPU=" +
            klayout_cuda::implant12::verdict_name(gpu[i]) +
            ", KLayout=" +
            klayout_cuda::implant12::verdict_name(direct));
      }
    }
  }
  *checked += pairs.size();
}

void test_named(std::uint64_t *checked) {
  constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();
  const DirectedEdge east = edge(0, 0, 100, 0);
  const std::vector<NamedCase> cases = {
      {"east exterior d-1",
       pair(east, edge(100, 139, 0, 139)), kImplant1,
       Verdict::kViolation},
      {"east exterior d",
       pair(east, edge(100, 140, 0, 140)), kImplant1,
       Verdict::kNoViolation},
      {"east interior",
       pair(east, edge(100, -139, 0, -139)), kImplant1,
       Verdict::kNoViolation},
      {"west exterior",
       pair(edge(100, 0, 0, 0), edge(0, -139, 100, -139)),
       kImplant1, Verdict::kViolation},
      {"north exterior",
       pair(edge(0, 0, 0, 100), edge(-139, 100, -139, 0)),
       kImplant1, Verdict::kViolation},
      {"south exterior",
       pair(edge(0, 100, 0, 0), edge(139, 0, 139, 100)),
       kImplant1, Verdict::kViolation},
      {"same direction",
       pair(east, edge(0, 1, 100, 1)), kImplant1,
       Verdict::kNoViolation},
      {"perpendicular",
       pair(east, edge(50, -10, 50, 10)), kImplant1,
       Verdict::kNoViolation},
      {"positive projection overlap",
       pair(east, edge(200, 1, 99, 1)), kImplant1,
       Verdict::kViolation},
      {"endpoint-only projection",
       pair(east, edge(200, 1, 100, 1)), kImplant1,
       Verdict::kNoViolation},
      {"separate projection",
       pair(east, edge(201, 1, 101, 1)), kImplant1,
       Verdict::kNoViolation},
      {"collinear overlap",
       pair(east, edge(150, 0, 50, 0)), kImplant1,
       Verdict::kViolation},
      {"collinear endpoint touch",
       pair(east, edge(200, 0, 100, 0)), kImplant1,
       Verdict::kNoViolation},
      {"IMPLANT.2 d-1",
       pair(east, edge(100, 49, 0, 49)), kImplant2,
       Verdict::kViolation},
      {"IMPLANT.2 d",
       pair(east, edge(100, 50, 0, 50)), kImplant2,
       Verdict::kNoViolation},
      {"negative coordinates",
       pair(edge(-200, -100, -100, -100),
            edge(-100, 39, -200, 39)),
       kImplant1, Verdict::kViolation},
      {"diagonal fail closed",
       pair(edge(0, 0, 100, 100), edge(100, 1, 0, 101)),
       kImplant1, Verdict::kUncertain},
      {"dot fail closed",
       pair(edge(0, 0, 0, 0), edge(100, 1, 0, 1)),
       kImplant1, Verdict::kUncertain},
      {"unqualified distance",
       pair(east, edge(100, 49, 0, 49)), 51,
       Verdict::kUncertain},
      {"full signed range fail closed",
       pair(edge(lo, 0, hi, 0), edge(hi, 1, lo, 1)),
       kImplant1, Verdict::kUncertain},
      {"near INT64 min exact",
       pair(edge(lo + 10, -1, lo + 110, -1),
            edge(lo + 110, 138, lo + 10, 138)),
       kImplant1, Verdict::kViolation},
      {"near INT64 max exact",
       pair(edge(hi - 110, 1, hi - 10, 1),
            edge(hi - 10, 140, hi - 110, 140)),
       kImplant1, Verdict::kViolation},
  };

  for (const NamedCase &test : cases) {
    std::vector<Verdict> gpu(1);
    std::string error;
    if (!klayout_cuda::implant12::classify_batch(
            &test.edges, 1, test.distance, gpu.data(), &error)) {
      throw std::runtime_error(
          std::string(test.name) + ": CUDA batch failed: " + error);
    }
    if (gpu[0] != test.expected) {
      throw std::runtime_error(
          std::string(test.name) + ": expected " +
          klayout_cuda::implant12::verdict_name(test.expected) +
          ", got " + klayout_cuda::implant12::verdict_name(gpu[0]));
    }
    const Verdict independent =
        independent_oracle(test.edges, test.distance);
    if (independent != test.expected) {
      throw std::runtime_error(
          std::string(test.name) + ": independent oracle mismatch");
    }
    if (test.expected != Verdict::kUncertain &&
        klayout_oracle(test.edges, test.distance) != test.expected) {
      throw std::runtime_error(
          std::string(test.name) + ": direct KLayout mismatch");
    }
    ++*checked;
  }
}

std::vector<EdgePair> random_pairs(std::size_t count, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::int64_t> coord(-1000000, 1000000);
  std::uniform_int_distribution<std::int64_t> length(1, 2000);
  std::uniform_int_distribution<std::int64_t> line_delta(-220, 220);
  std::uniform_int_distribution<std::int64_t> projection_delta(-2100, 2100);
  std::uniform_int_distribution<int> bit(0, 1);

  std::vector<EdgePair> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const bool horizontal = bit(rng) != 0;
    const bool positive = bit(rng) != 0;
    const std::int64_t axis = coord(rng);
    const std::int64_t line = coord(rng);
    const std::int64_t alen = length(rng);
    const std::int64_t blen = length(rng);
    const std::int64_t baxis = axis + projection_delta(rng);
    const std::int64_t bline = line + line_delta(rng);
    DirectedEdge a{};
    DirectedEdge b{};
    if (horizontal) {
      a = positive ? edge(axis, line, axis + alen, line)
                   : edge(axis + alen, line, axis, line);
      b = positive ? edge(baxis + blen, bline, baxis, bline)
                   : edge(baxis, bline, baxis + blen, bline);
    } else {
      a = positive ? edge(line, axis, line, axis + alen)
                   : edge(line, axis + alen, line, axis);
      b = positive ? edge(bline, baxis + blen, bline, baxis)
                   : edge(bline, baxis, bline, baxis + blen);
    }
    result.push_back(pair(a, b));
  }
  return result;
}

}  // namespace

int main() {
  try {
    std::uint64_t checked = 0;
    test_named(&checked);
    constexpr std::size_t kPerProfile = 240000;
    const std::vector<EdgePair> implant1 =
        random_pairs(kPerProfile, UINT64_C(0x1a2b3c4d5e6f7788));
    const std::vector<EdgePair> implant2 =
        random_pairs(kPerProfile, UINT64_C(0x8877665544332211));
    run_and_compare(implant1, kImplant1, &checked);
    run_and_compare(implant2, kImplant2, &checked);
    std::cout << "implant12 exact predicate PASS comparisons=" << checked
              << " gpu=oracle=direct-klayout\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "implant12 exact predicate FAIL: " << error.what() << "\n";
    return 1;
  }
}
