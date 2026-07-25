/*
 * Exact rectangle decomposition for one simple, hole-free Manhattan contour.
 *
 * This header is deliberately CUDA-free so the host qualification logic can
 * be exercised without creating a CUDA context.  The caller remains
 * responsible for its outer coordinate-domain and record-span checks.
 */

#ifndef KLAYOUT_CUDA_M2_MANHATTAN_DECOMPOSE_H
#define KLAYOUT_CUDA_M2_MANHATTAN_DECOMPOSE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace klayout_cuda {
namespace m2_manhattan_decompose {

struct EdgeI64
{
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
};

struct RectangleI64
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint64_t source_token;
};

enum class Status
{
  complete,
  malformed,
  capacity
};

struct Result
{
  Status status = Status::malformed;
  std::string message;
  std::vector<RectangleI64> rectangles;
};

inline Result failure(Status status, const char *message)
{
  Result result;
  result.status = status;
  result.message = message;
  return result;
}

inline bool intervals_overlap_inclusive(
    std::int64_t first_lo, std::int64_t first_hi,
    std::int64_t second_lo, std::int64_t second_hi)
{
  if (first_lo > first_hi) std::swap(first_lo, first_hi);
  if (second_lo > second_hi) std::swap(second_lo, second_hi);
  return std::max(first_lo, second_lo) <=
         std::min(first_hi, second_hi);
}

inline bool edges_intersect(const EdgeI64 &first,
                            const EdgeI64 &second)
{
  const bool first_vertical = first.x1 == first.x2;
  const bool second_vertical = second.x1 == second.x2;
  if (first_vertical && second_vertical) {
    return first.x1 == second.x1 &&
           intervals_overlap_inclusive(
               first.y1, first.y2, second.y1, second.y2);
  }
  if (!first_vertical && !second_vertical) {
    return first.y1 == second.y1 &&
           intervals_overlap_inclusive(
               first.x1, first.x2, second.x1, second.x2);
  }
  const EdgeI64 &vertical = first_vertical ? first : second;
  const EdgeI64 &horizontal = first_vertical ? second : first;
  return intervals_overlap_inclusive(
             vertical.y1, vertical.y2,
             horizontal.y1, horizontal.y1) &&
         intervals_overlap_inclusive(
             horizontal.x1, horizontal.x2,
             vertical.x1, vertical.x1);
}

inline Result slab_partition(
    const std::vector<EdgeI64> &edges,
    std::int64_t expected_left, std::int64_t expected_bottom,
    std::int64_t expected_right, std::int64_t expected_top,
    std::uint64_t source_token, std::uint64_t max_rectangles)
{
  std::vector<std::int64_t> y_levels;
  y_levels.reserve(edges.size());
  for (std::size_t index = 0; index < edges.size(); ++index) {
    y_levels.push_back(edges[index].y1);
  }

  std::sort(y_levels.begin(), y_levels.end());
  y_levels.erase(
      std::unique(y_levels.begin(), y_levels.end()),
      y_levels.end());
  if (y_levels.size() < 2 ||
      y_levels.front() != expected_bottom ||
      y_levels.back() != expected_top) {
    return failure(
        Status::malformed,
        "contour y-coordinate census is inconsistent");
  }

  Result result;
  result.status = Status::complete;
  std::vector<std::int64_t> crossings;
  crossings.reserve(edges.size() / 2);
  typedef std::pair<std::int64_t, std::int64_t> Interval;
  std::map<Interval, std::size_t> previous;
  std::map<Interval, std::size_t> current;

  for (std::size_t band = 0; band + 1 < y_levels.size(); ++band) {
    const std::int64_t bottom = y_levels[band];
    const std::int64_t top = y_levels[band + 1];
    if (bottom >= top) {
      return failure(
          Status::malformed,
          "contour has a nonpositive horizontal slab");
    }
    const __int128 midpoint2 =
        static_cast<__int128>(bottom) + top;
    crossings.clear();
    for (std::size_t index = 0; index < edges.size(); ++index) {
      const EdgeI64 &edge = edges[index];
      if (edge.x1 != edge.x2) continue;
      const std::int64_t lo = std::min(edge.y1, edge.y2);
      const std::int64_t hi = std::max(edge.y1, edge.y2);
      if (static_cast<__int128>(lo) * 2 < midpoint2 &&
          midpoint2 < static_cast<__int128>(hi) * 2) {
        crossings.push_back(edge.x1);
      }
    }
    std::sort(crossings.begin(), crossings.end());
    if (crossings.empty() || (crossings.size() & 1u)) {
      return failure(
          Status::malformed,
          "contour has an invalid scanline crossing parity");
    }
    current.clear();
    for (std::size_t crossing = 0;
         crossing < crossings.size(); crossing += 2) {
      const std::int64_t left = crossings[crossing];
      const std::int64_t right = crossings[crossing + 1];
      if (left >= right ||
          left < expected_left || right > expected_right ||
          (crossing && crossings[crossing - 1] >= left)) {
        return failure(
            Status::malformed,
            "contour has duplicate or inverted scanline crossings");
      }
      const Interval interval(left, right);
      const std::map<Interval, std::size_t>::const_iterator prior =
          previous.find(interval);
      if (prior != previous.end() &&
          result.rectangles[prior->second].top == bottom) {
        result.rectangles[prior->second].top = top;
        current.insert(std::make_pair(interval, prior->second));
      } else {
        if (result.rectangles.size() >= max_rectangles ||
            result.rectangles.size() ==
                result.rectangles.max_size()) {
          return failure(
              Status::capacity,
              "rectangle decomposition exceeds capacity");
        }
        const std::size_t rectangle = result.rectangles.size();
        result.rectangles.push_back(
            RectangleI64{
                left, bottom, right, top, source_token});
        current.insert(std::make_pair(interval, rectangle));
      }
    }
    previous.swap(current);
  }

  if (result.rectangles.empty()) {
    return failure(
        Status::malformed,
        "contour decomposition produced no rectangles");
  }
  return result;
}

inline bool validate_partition(
    const std::vector<RectangleI64> &rectangles,
    std::int64_t expected_left, std::int64_t expected_bottom,
    std::int64_t expected_right, std::int64_t expected_top,
    std::uint64_t source_token, __int128 twice_area)
{
  if (rectangles.empty()) return false;
  __int128 rectangle_area = 0;
  for (std::size_t index = 0; index < rectangles.size(); ++index) {
    const RectangleI64 &rectangle = rectangles[index];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top ||
        rectangle.left < expected_left ||
        rectangle.bottom < expected_bottom ||
        rectangle.right > expected_right ||
        rectangle.top > expected_top ||
        rectangle.source_token != source_token) {
      return false;
    }
    rectangle_area +=
        (static_cast<__int128>(rectangle.right) - rectangle.left) *
        (static_cast<__int128>(rectangle.top) - rectangle.bottom);
  }
  return rectangle_area * 2 == -twice_area;
}

inline Result decompose(
    const std::vector<EdgeI64> &edges,
    std::int64_t expected_left, std::int64_t expected_bottom,
    std::int64_t expected_right, std::int64_t expected_top,
    std::uint64_t source_token, std::uint64_t max_rectangles)
{
  if (edges.size() < 4 || (edges.size() & 1u)) {
    return failure(
        Status::malformed,
        "contour does not have an even edge count of at least four");
  }
  if (expected_left >= expected_right ||
      expected_bottom >= expected_top) {
    return failure(Status::malformed, "contour bounding box is empty");
  }

  std::int64_t derived_left = 0;
  std::int64_t derived_bottom = 0;
  std::int64_t derived_right = 0;
  std::int64_t derived_top = 0;
  __int128 twice_area = 0;
  for (std::size_t index = 0; index < edges.size(); ++index) {
    const EdgeI64 &edge = edges[index];
    if ((edge.x1 == edge.x2 && edge.y1 == edge.y2) ||
        !(edge.x1 == edge.x2 || edge.y1 == edge.y2)) {
      return failure(
          Status::malformed,
          "contour contains a degenerate or non-Manhattan edge");
    }
    const EdgeI64 &following = edges[(index + 1) % edges.size()];
    if (edge.x2 != following.x1 || edge.y2 != following.y1 ||
        (edge.x1 == edge.x2) ==
            (following.x1 == following.x2)) {
      return failure(
          Status::malformed,
          "contour is open or contains a redundant turn");
    }
    if (!index) {
      derived_left = derived_right = edge.x1;
      derived_bottom = derived_top = edge.y1;
    } else {
      derived_left = std::min(derived_left, edge.x1);
      derived_bottom = std::min(derived_bottom, edge.y1);
      derived_right = std::max(derived_right, edge.x1);
      derived_top = std::max(derived_top, edge.y1);
    }
    twice_area +=
        static_cast<__int128>(edge.x1) * edge.y2 -
        static_cast<__int128>(edge.x2) * edge.y1;

    for (std::size_t other = index + 1;
         other < edges.size(); ++other) {
      const bool adjacent =
          other == index + 1 ||
          (index == 0 && other + 1 == edges.size());
      if (!adjacent && edges_intersect(edge, edges[other])) {
        return failure(Status::malformed, "contour self-intersects");
      }
    }
  }

  if (derived_left != expected_left ||
      derived_bottom != expected_bottom ||
      derived_right != expected_right ||
      derived_top != expected_top) {
    return failure(
        Status::malformed,
        "contour bounding-box echo is inconsistent");
  }
  if (twice_area >= 0) {
    return failure(Status::malformed, "contour is not clockwise");
  }

  Result horizontal = slab_partition(
      edges, expected_left, expected_bottom,
      expected_right, expected_top, source_token, max_rectangles);
  std::vector<EdgeI64> transposed;
  transposed.reserve(edges.size());
  for (std::size_t index = 0; index < edges.size(); ++index) {
    const EdgeI64 &edge = edges[index];
    transposed.push_back(
        EdgeI64{edge.y1, edge.x1, edge.y2, edge.x2});
  }
  Result vertical = slab_partition(
      transposed, expected_bottom, expected_left,
      expected_top, expected_right, source_token, max_rectangles);
  if (vertical.status == Status::complete) {
    for (std::size_t index = 0;
         index < vertical.rectangles.size(); ++index) {
      RectangleI64 &rectangle = vertical.rectangles[index];
      const RectangleI64 original = rectangle;
      rectangle.left = original.bottom;
      rectangle.bottom = original.left;
      rectangle.right = original.top;
      rectangle.top = original.right;
    }
  }

  Result result;
  if (horizontal.status == Status::complete &&
      (vertical.status != Status::complete ||
       horizontal.rectangles.size() <=
           vertical.rectangles.size())) {
    result = std::move(horizontal);
  } else if (vertical.status == Status::complete) {
    result = std::move(vertical);
  } else if (horizontal.status == Status::capacity ||
             vertical.status == Status::capacity) {
    return failure(
        Status::capacity,
        "rectangle decomposition exceeds capacity in both orientations");
  } else {
    return failure(
        Status::malformed,
        "validated contour could not be partitioned into rectangles");
  }

  if (!validate_partition(
          result.rectangles, expected_left, expected_bottom,
          expected_right, expected_top, source_token, twice_area)) {
    return failure(
        Status::malformed,
        "rectangle decomposition changed exact area, bounds, or source token");
  }
  return result;
}

}  // namespace m2_manhattan_decompose
}  // namespace klayout_cuda

#endif
