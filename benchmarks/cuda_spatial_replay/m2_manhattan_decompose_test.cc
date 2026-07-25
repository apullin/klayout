#include "m2_manhattan_decompose.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace md = klayout_cuda::m2_manhattan_decompose;

using Point = std::pair<std::int64_t, std::int64_t>;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

__int128 twice_area(const std::vector<Point> &points)
{
  __int128 area = 0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Point &first = points[index];
    const Point &second = points[(index + 1) % points.size()];
    area += static_cast<__int128>(first.first) * second.second -
            static_cast<__int128>(second.first) * first.second;
  }
  return area;
}

std::vector<md::EdgeI64> edges_for(const std::vector<Point> &points)
{
  std::vector<md::EdgeI64> edges;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Point &first = points[index];
    const Point &second = points[(index + 1) % points.size()];
    edges.push_back(
        {first.first, first.second, second.first, second.second});
  }
  return edges;
}

std::array<std::int64_t, 4> bounds_for(
    const std::vector<Point> &points)
{
  std::array<std::int64_t, 4> bounds = {
      points[0].first, points[0].second,
      points[0].first, points[0].second};
  for (std::size_t index = 1; index < points.size(); ++index) {
    bounds[0] = std::min(bounds[0], points[index].first);
    bounds[1] = std::min(bounds[1], points[index].second);
    bounds[2] = std::max(bounds[2], points[index].first);
    bounds[3] = std::max(bounds[3], points[index].second);
  }
  return bounds;
}

bool point_inside(const std::vector<Point> &points,
                  __int128 x2, __int128 y2)
{
  bool inside = false;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Point &first = points[index];
    const Point &second = points[(index + 1) % points.size()];
    if (first.first != second.first) continue;
    const __int128 first_y2 =
        static_cast<__int128>(first.second) * 2;
    const __int128 second_y2 =
        static_cast<__int128>(second.second) * 2;
    if ((first_y2 > y2) != (second_y2 > y2) &&
        static_cast<__int128>(first.first) * 2 > x2) {
      inside = !inside;
    }
  }
  return inside;
}

void prove_exact_cover(
    const std::vector<Point> &points,
    const std::vector<md::RectangleI64> &rectangles,
    std::uint64_t token)
{
  std::vector<std::int64_t> xs;
  std::vector<std::int64_t> ys;
  for (std::size_t index = 0; index < points.size(); ++index) {
    xs.push_back(points[index].first);
    ys.push_back(points[index].second);
  }
  for (std::size_t index = 0; index < rectangles.size(); ++index) {
    const md::RectangleI64 &rectangle = rectangles[index];
    require(
        rectangle.source_token == token,
        "rectangle changed its source token");
    xs.push_back(rectangle.left);
    xs.push_back(rectangle.right);
    ys.push_back(rectangle.bottom);
    ys.push_back(rectangle.top);
  }
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  std::sort(ys.begin(), ys.end());
  ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

  for (std::size_t xi = 0; xi + 1 < xs.size(); ++xi) {
    for (std::size_t yi = 0; yi + 1 < ys.size(); ++yi) {
      const __int128 x2 =
          static_cast<__int128>(xs[xi]) + xs[xi + 1];
      const __int128 y2 =
          static_cast<__int128>(ys[yi]) + ys[yi + 1];
      const bool expected = point_inside(points, x2, y2);
      std::size_t covering = 0;
      for (std::size_t rectangle = 0;
           rectangle < rectangles.size(); ++rectangle) {
        const md::RectangleI64 &candidate =
            rectangles[rectangle];
        if (static_cast<__int128>(candidate.left) * 2 < x2 &&
            x2 < static_cast<__int128>(candidate.right) * 2 &&
            static_cast<__int128>(candidate.bottom) * 2 < y2 &&
            y2 < static_cast<__int128>(candidate.top) * 2) {
          ++covering;
        }
      }
      require(
          covering == (expected ? 1u : 0u),
          "rectangles do not form a disjoint exact cover");
    }
  }
}

md::Result accept(const std::vector<Point> &points,
                  std::uint64_t token,
                  std::uint64_t capacity = 64)
{
  require(twice_area(points) < 0, "fixture is not clockwise");
  const std::array<std::int64_t, 4> bounds = bounds_for(points);
  const md::Result result = md::decompose(
      edges_for(points), bounds[0], bounds[1],
      bounds[2], bounds[3], token, capacity);
  require(
      result.status == md::Status::complete,
      std::string("qualified contour declined: ") + result.message);
  prove_exact_cover(points, result.rectangles, token);
  return result;
}

std::vector<Point> transformed_clockwise(
    const std::vector<Point> &source, std::uint32_t code)
{
  std::vector<Point> result;
  for (std::size_t index = 0; index < source.size(); ++index) {
    const std::int64_t x = source[index].first;
    const std::int64_t y = source[index].second;
    switch (code) {
    case 0: result.push_back({x, y}); break;
    case 1: result.push_back({-y, x}); break;
    case 2: result.push_back({-x, -y}); break;
    case 3: result.push_back({y, -x}); break;
    case 4: result.push_back({x, -y}); break;
    case 5: result.push_back({y, x}); break;
    case 6: result.push_back({-x, y}); break;
    case 7: result.push_back({-y, -x}); break;
    default: throw std::runtime_error("invalid fixture transform");
    }
  }
  if (twice_area(result) > 0) {
    std::reverse(result.begin(), result.end());
  }
  return result;
}

bool same_rectangles(
    const std::vector<md::RectangleI64> &first,
    const std::vector<md::RectangleI64> &second)
{
  if (first.size() != second.size()) return false;
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (first[index].left != second[index].left ||
        first[index].bottom != second[index].bottom ||
        first[index].right != second[index].right ||
        first[index].top != second[index].top ||
        first[index].source_token != second[index].source_token) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main()
{
  try {
    const std::array<std::vector<Point>, 4> production = {{
        {{1775, 2585}, {1775, 3585}, {1875, 3585},
         {1875, 4645}, {2055, 4645}, {2055, 3585},
         {2265, 3585}, {2265, 2585}},
        {{3045, 2300}, {3045, 4300}, {3255, 4300},
         {3255, 3655}, {3535, 3655}, {3535, 2655},
         {3255, 2655}, {3255, 2300}},
        {{4975, 2585}, {4975, 3785}, {4695, 3785},
         {4695, 4285}, {4905, 4285}, {4905, 4585},
         {5115, 4585}, {5115, 2585}},
        {{1895, 330}, {1895, 1200}, {1775, 1200},
         {1775, 1700}, {2265, 1700}, {2265, 1200},
         {2075, 1200}, {2075, 330}}
    }};
    const std::array<std::size_t, 4>
        expected_production_rectangles = {{2, 2, 3, 2}};

    std::size_t production_rectangles = 0;
    for (std::size_t index = 0; index < production.size(); ++index) {
      const md::Result result =
          accept(production[index], UINT64_C(0xabc000) + index);
      require(
          result.rectangles.size() ==
              expected_production_rectangles[index],
          "production eight-edge contour used an unexpected rectangle count");
      production_rectangles += result.rectangles.size();
      for (std::uint32_t transform = 0; transform < 8; ++transform) {
        const md::Result transformed = accept(
            transformed_clockwise(production[index], transform),
            UINT64_C(0xdef000) + index * 8 + transform);
        require(
            transformed.rectangles.size() == result.rectangles.size(),
            "orthogonal transform changed rectangle census");
      }
      for (std::size_t start = 0;
           start < production[index].size(); ++start) {
        std::vector<Point> rotated = production[index];
        std::rotate(
            rotated.begin(), rotated.begin() + start, rotated.end());
        const md::Result cyclic = accept(
            rotated, UINT64_C(0xabc000) + index);
        require(
            same_rectangles(result.rectangles, cyclic.rectangles),
            "cyclic start vertex changed deterministic decomposition");
      }
    }

    const std::vector<Point> staircase = {
        {0, 0}, {0, 9}, {3, 9}, {3, 7},
        {6, 7}, {6, 5}, {9, 5}, {9, 0}};
    const md::Result staircase_result =
        accept(staircase, UINT64_C(0xffffffffffffffff));
    require(
        staircase_result.rectangles.size() == 3,
        "staircase did not decompose into three rectangles");

    std::vector<Point> u_shape = {
        {0, 0}, {6, 0}, {6, 6}, {4, 6},
        {4, 2}, {2, 2}, {2, 6}, {0, 6}};
    std::reverse(u_shape.begin(), u_shape.end());
    const md::Result u_result =
        accept(u_shape, UINT64_C(0x5555555555555555));
    require(
        u_result.rectangles.size() == 3,
        "multi-band U did not decompose into three rectangles");

    std::vector<Point> near_limit = production[3];
    for (std::size_t index = 0; index < near_limit.size(); ++index) {
      near_limit[index].first += INT64_C(999999990000);
      near_limit[index].second -= INT64_C(999999990000);
    }
    accept(near_limit, UINT64_C(0x7777777777777777));

    std::vector<Point> wrong_winding = production[0];
    std::reverse(wrong_winding.begin(), wrong_winding.end());
    const std::array<std::int64_t, 4> wrong_bounds =
        bounds_for(wrong_winding);
    const md::Result winding_result = md::decompose(
        edges_for(wrong_winding), wrong_bounds[0], wrong_bounds[1],
        wrong_bounds[2], wrong_bounds[3], 7, 64);
    require(
        winding_result.status == md::Status::malformed &&
            winding_result.message.find("clockwise") !=
                std::string::npos,
        "counterclockwise contour did not fail closed");

    const md::Result capacity_result = md::decompose(
        edges_for(staircase), 0, 0, 9, 9, 9, 2);
    require(
        capacity_result.status == md::Status::capacity,
        "rectangle capacity did not fail closed");

    const std::vector<md::EdgeI64> bow_tie = {
        {0, 0, 0, 4}, {0, 4, 3, 4},
        {3, 4, 3, 1}, {3, 1, 1, 1},
        {1, 1, 1, 3}, {1, 3, 4, 3},
        {4, 3, 4, 0}, {4, 0, 0, 0}};
    const md::Result self_intersection =
        md::decompose(bow_tie, 0, 0, 4, 4, 11, 64);
    require(
        self_intersection.status == md::Status::malformed,
        "self-intersecting contour did not fail closed");

    std::cout
        << "M2_MANHATTAN_DECOMPOSE_TEST PASS"
        << " production_contours=" << production.size()
        << " production_rectangles=" << production_rectangles
        << " transforms=32 cyclic_starts=32"
        << " adversarial=5 source_tokens=71\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "M2_MANHATTAN_DECOMPOSE_TEST FAIL: "
        << error.what() << "\n";
    return 1;
  }
}
