/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#ifndef KLAYOUT_POLY34_TERMINAL_EMPTY_CERTIFICATE_CUH
#define KLAYOUT_POLY34_TERMINAL_EMPTY_CERTIFICATE_CUH

#include <cstddef>
#include <cstdint>

namespace klayout_cuda {
namespace poly34 {

constexpr std::int64_t kPoly3Distance = 110;
constexpr std::int64_t kPoly4Distance = 140;
constexpr std::uint32_t kMaximumCandidateBoxes = 64;

struct Box {
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
};

enum class Certificate : std::uint8_t {
  kUnsupported = 0,
  kFallback = 1,
  kTerminalEmpty = 2
};

#if defined(__CUDACC__)
#define KLAYOUT_POLY34_HD __host__ __device__
#else
#define KLAYOUT_POLY34_HD
#endif

KLAYOUT_POLY34_HD inline bool valid_box(const Box &box) {
  return box.left < box.right && box.bottom < box.top;
}

KLAYOUT_POLY34_HD inline bool add_checked(
    std::int64_t value, std::int64_t delta, std::int64_t *result) {
  if ((delta > 0 && value > INT64_MAX - delta) ||
      (delta < 0 && value < INT64_MIN - delta)) {
    return false;
  }
  *result = value + delta;
  return true;
}

KLAYOUT_POLY34_HD inline bool positive_area_intersection(
    const Box &first, const Box &second) {
  return first.left < second.right && first.right > second.left &&
         first.bottom < second.top && first.top > second.bottom;
}

// Proves that target is covered by the union of boxes.  The Y endpoints of
// all rectangles partition target into slabs with constant active intervals.
// Within each slab a greedy interval-union walk proves complete X coverage.
// This intentionally remains bounded and allocation-free for device use.
KLAYOUT_POLY34_HD inline bool union_covers(
    const Box &target, const Box *boxes, std::uint32_t box_count) {
  if (!valid_box(target) || !boxes || !box_count) {
    return false;
  }
  std::int64_t y = target.bottom;
  while (y < target.top) {
    std::int64_t next_y = target.top;
    for (std::uint32_t i = 0; i < box_count; ++i) {
      const Box &box = boxes[i];
      if (box.bottom > y && box.bottom < next_y && box.bottom < target.top) {
        next_y = box.bottom;
      }
      if (box.top > y && box.top < next_y && box.top < target.top) {
        next_y = box.top;
      }
    }
    if (next_y <= y) {
      return false;
    }

    std::int64_t x = target.left;
    while (x < target.right) {
      std::int64_t farthest = x;
      for (std::uint32_t i = 0; i < box_count; ++i) {
        const Box &box = boxes[i];
        if (box.bottom <= y && box.top >= next_y &&
            box.left <= x && box.right > farthest) {
          farthest = box.right;
        }
      }
      if (farthest <= x) {
        return false;
      }
      x = farthest;
    }
    y = next_y;
  }
  return true;
}

KLAYOUT_POLY34_HD inline bool union_misses(
    const Box &target, const Box *boxes, std::uint32_t box_count) {
  for (std::uint32_t i = 0; i < box_count; ++i) {
    if (positive_area_intersection(target, boxes[i])) {
      return false;
    }
  }
  return true;
}

KLAYOUT_POLY34_HD inline Certificate terminal_empty_profile(
    const Box &gate,
    const Box *primary_boxes,
    std::uint32_t primary_count,
    std::int64_t distance,
    bool supported = true,
    std::uint8_t proven_internal_sides = 0) {
  if (!supported || !valid_box(gate) || !primary_boxes || !primary_count ||
      primary_count > kMaximumCandidateBoxes ||
      (distance != kPoly3Distance && distance != kPoly4Distance)) {
    return Certificate::kUnsupported;
  }
  for (std::uint32_t i = 0; i < primary_count; ++i) {
    if (!valid_box(primary_boxes[i])) {
      return Certificate::kUnsupported;
    }
  }

  // A live caller will pass gate = poly & active.  Re-prove containment here
  // so malformed or incompletely gathered candidate windows fail closed.
  if (!union_covers(gate, primary_boxes, primary_count)) {
    return Certificate::kFallback;
  }

  Box bands[4] = {
      {gate.left, gate.bottom, gate.left, gate.top},
      {gate.right, gate.bottom, gate.right, gate.top},
      {gate.left, gate.bottom, gate.right, gate.bottom},
      {gate.left, gate.top, gate.right, gate.top}};
  if (!add_checked(gate.left, -distance, &bands[0].left) ||
      !add_checked(gate.right, distance, &bands[1].right) ||
      !add_checked(gate.bottom, -distance, &bands[2].bottom) ||
      !add_checked(gate.top, distance, &bands[3].top)) {
    return Certificate::kUnsupported;
  }

  // For each projected side, one of two sufficient facts must hold:
  //
  //   * the open sub-threshold band contains no primary area, so any reported
  //     facing pair is coincident and normalizes to a zero-area polygon; or
  //   * the complete band is covered, so every facing boundary is at least
  //     the qualified distance away.
  //
  // A mixture of coincident and fully extended segments may also be clean in
  // KLayout, but is deliberately declined here.  Partial coverage, unrelated
  // nearby primary geometry, and candidate-window ambiguity therefore cannot
  // become a false terminal-empty certificate.
  for (std::uint32_t side = 0; side < 4; ++side) {
    // A raw rectangle cover can split one physical GATE component into
    // several tiles.  Its caller may suppress a tile side only after
    // independently proving that a positive-width strip immediately across
    // the complete side belongs to GATE too.  Such a side is internal to the
    // exact union and therefore cannot participate in KLayout's enclosing
    // result.  Every unproved or partially internal side retains the original
    // conservative test below.
    if (proven_internal_sides & (std::uint8_t(1) << side)) {
      continue;
    }
    const Box &band = bands[side];
    if (!union_misses(band, primary_boxes, primary_count) &&
        !union_covers(band, primary_boxes, primary_count)) {
      return Certificate::kFallback;
    }
  }
  return Certificate::kTerminalEmpty;
}

KLAYOUT_POLY34_HD inline const char *certificate_name(Certificate certificate) {
  switch (certificate) {
    case Certificate::kUnsupported:
      return "UNSUPPORTED";
    case Certificate::kFallback:
      return "FALLBACK";
    case Certificate::kTerminalEmpty:
      return "TERMINAL_EMPTY";
  }
  return "INVALID";
}

#undef KLAYOUT_POLY34_HD

}  // namespace poly34
}  // namespace klayout_cuda

#endif
