#include "manhattan_union_oracle.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using klayout_cuda::manhattan_union_oracle::BoundaryFragment;
using klayout_cuda::manhattan_union_oracle::Cell;
using klayout_cuda::manhattan_union_oracle::Component;
using klayout_cuda::manhattan_union_oracle::Coord;
using klayout_cuda::manhattan_union_oracle::KissingDiagonal;
using klayout_cuda::manhattan_union_oracle::KissingVertex;
using klayout_cuda::manhattan_union_oracle::Limits;
using klayout_cuda::manhattan_union_oracle::Rect;
using klayout_cuda::manhattan_union_oracle::UnionResult;
using klayout_cuda::manhattan_union_oracle::UnionStatus;
using klayout_cuda::manhattan_union_oracle::describe;
using klayout_cuda::manhattan_union_oracle::unite;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t distance(Coord lower, Coord upper)
{
  return static_cast<std::uint64_t>(upper) -
         static_cast<std::uint64_t>(lower);
}

class DisjointSet
{
public:
  explicit DisjointSet(std::size_t size) : m_parent(size), m_rank(size, 0)
  {
    std::iota(m_parent.begin(), m_parent.end(), std::size_t(0));
  }

  std::size_t find(std::size_t item)
  {
    std::size_t root = item;
    while (m_parent[root] != root) root = m_parent[root];
    while (m_parent[item] != item) {
      const std::size_t parent = m_parent[item];
      m_parent[item] = root;
      item = parent;
    }
    return root;
  }

  void unite(std::size_t first, std::size_t second)
  {
    first = find(first);
    second = find(second);
    if (first == second) return;

    if (m_rank[first] < m_rank[second]) std::swap(first, second);
    m_parent[second] = first;
    if (m_rank[first] == m_rank[second]) ++m_rank[first];
  }

private:
  std::vector<std::size_t> m_parent;
  std::vector<unsigned char> m_rank;
};

/*
 * Independent dense reference used only by the tests below.
 *
 * Unlike the sparse oracle implementation, this builds a rectangular bitmap,
 * labels it with a disjoint-set pass, and emits already-maximal fragments by
 * scanning complete horizontal and vertical lattice lines.  The deliberate
 * algorithm/code-path duplication makes randomized differential tests useful.
 */
UnionResult dense_reference(const std::vector<Rect> &rectangles)
{
  UnionResult result;
  if (rectangles.empty()) return result;

  Coord min_x = rectangles.front().left;
  Coord min_y = rectangles.front().bottom;
  Coord max_x = rectangles.front().right;
  Coord max_y = rectangles.front().top;
  for (const Rect &rectangle : rectangles) {
    require(rectangle.left < rectangle.right &&
                rectangle.bottom < rectangle.top,
            "dense reference received malformed rectangle");
    min_x = std::min(min_x, rectangle.left);
    min_y = std::min(min_y, rectangle.bottom);
    max_x = std::max(max_x, rectangle.right);
    max_y = std::max(max_y, rectangle.top);
  }

  const std::size_t width = static_cast<std::size_t>(distance(min_x, max_x));
  const std::size_t height = static_cast<std::size_t>(distance(min_y, max_y));
  require(width <= 256 && height <= 256,
          "dense reference fixture exceeds bounded domain");

  const auto index = [width](std::size_t x, std::size_t y) {
    return y * width + x;
  };

  std::vector<unsigned char> filled(width * height, 0);
  for (const Rect &rectangle : rectangles) {
    const std::size_t left =
        static_cast<std::size_t>(distance(min_x, rectangle.left));
    const std::size_t right =
        static_cast<std::size_t>(distance(min_x, rectangle.right));
    const std::size_t bottom =
        static_cast<std::size_t>(distance(min_y, rectangle.bottom));
    const std::size_t top =
        static_cast<std::size_t>(distance(min_y, rectangle.top));
    for (std::size_t y = bottom; y < top; ++y)
      for (std::size_t x = left; x < right; ++x)
        filled[index(x, y)] = 1;
  }

  result.area_cells =
      static_cast<std::uint64_t>(std::count(filled.begin(), filled.end(), 1));

  DisjointSet sets(filled.size());
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      if (!filled[index(x, y)]) continue;
      if (x > 0 && filled[index(x - 1, y)])
        sets.unite(index(x, y), index(x - 1, y));
      if (y > 0 && filled[index(x, y - 1)])
        sets.unite(index(x, y), index(x, y - 1));
    }
  }

  struct Description
  {
    std::size_t root;
    Cell least;
    std::uint64_t area;
  };

  std::map<std::size_t, Description> by_root;
  for (std::size_t x = 0; x < width; ++x) {
    for (std::size_t y = 0; y < height; ++y) {
      const std::size_t item = index(x, y);
      if (!filled[item]) continue;
      const std::size_t root = sets.find(item);
      const Cell cell = {min_x + static_cast<Coord>(x),
                         min_y + static_cast<Coord>(y)};
      const auto inserted =
          by_root.emplace(root, Description{root, cell, 0});
      Description &description = inserted.first->second;
      if (std::tie(cell.x, cell.y) <
          std::tie(description.least.x, description.least.y))
        description.least = cell;
      ++description.area;
    }
  }

  std::vector<Description> descriptions;
  for (const auto &entry : by_root) descriptions.push_back(entry.second);
  std::sort(descriptions.begin(), descriptions.end(),
            [](const Description &first, const Description &second) {
              return std::tie(first.least.x, first.least.y) <
                     std::tie(second.least.x, second.least.y);
            });

  std::map<std::size_t, std::uint32_t> root_to_component;
  for (std::size_t id = 0; id < descriptions.size(); ++id) {
    const Description &description = descriptions[id];
    root_to_component.emplace(description.root,
                              static_cast<std::uint32_t>(id));
    result.components.push_back(
        {static_cast<std::uint32_t>(id), description.least,
         description.area});
  }

  std::vector<std::uint32_t> labels(
      filled.size(), std::numeric_limits<std::uint32_t>::max());
  for (std::size_t item = 0; item < filled.size(); ++item)
    if (filled[item])
      labels[item] = root_to_component.at(sets.find(item));

  /*
   * Independently classify checkerboard vertices as unsupported for the
   * maximum-coherence production contract.
   */
  for (std::size_t x = 0; x <= width; ++x) {
    for (std::size_t y = 0; y <= height; ++y) {
      const bool southwest =
          x > 0 && y > 0 && filled[index(x - 1, y - 1)];
      const bool southeast =
          x < width && y > 0 && filled[index(x, y - 1)];
      const bool northwest =
          x > 0 && y < height && filled[index(x - 1, y)];
      const bool northeast =
          x < width && y < height && filled[index(x, y)];

      if (southwest && northeast && !southeast && !northwest) {
        result.kissing_vertices.push_back(
            {min_x + static_cast<Coord>(x),
             min_y + static_cast<Coord>(y),
             KissingDiagonal::SouthwestNortheast,
             labels[index(x - 1, y - 1)], labels[index(x, y)]});
      } else if (southeast && northwest && !southwest && !northeast) {
        result.kissing_vertices.push_back(
            {min_x + static_cast<Coord>(x),
             min_y + static_cast<Coord>(y),
             KissingDiagonal::NorthwestSoutheast,
             labels[index(x - 1, y)], labels[index(x, y - 1)]});
      }
    }
  }
  if (!result.kissing_vertices.empty())
    result.status = UnionStatus::UnsupportedKissingVertex;

  const auto horizontal = [&result](std::uint32_t component, int direction,
                                    Coord fixed, Coord lo, Coord hi) {
    if (direction > 0)
      result.boundary.push_back({component, lo, fixed, hi, fixed});
    else
      result.boundary.push_back({component, hi, fixed, lo, fixed});
  };

  const auto vertical = [&result](std::uint32_t component, int direction,
                                  Coord fixed, Coord lo, Coord hi) {
    if (direction > 0)
      result.boundary.push_back({component, fixed, lo, fixed, hi});
    else
      result.boundary.push_back({component, fixed, hi, fixed, lo});
  };

  /*
   * Horizontal lattice scan.  Occupied above means a bottom side and therefore
   * westward orientation; occupied below means an eastward top side.
   */
  for (std::size_t line = 0; line <= height; ++line) {
    bool running = false;
    std::uint32_t run_component = 0;
    int run_direction = 0;
    std::size_t run_start = 0;

    for (std::size_t x = 0; x <= width; ++x) {
      bool boundary = false;
      std::uint32_t component = 0;
      int direction = 0;

      if (x < width) {
        const bool below = line > 0 && filled[index(x, line - 1)];
        const bool above = line < height && filled[index(x, line)];
        if (below != above) {
          boundary = true;
          if (above) {
            component = labels[index(x, line)];
            direction = -1;
          } else {
            component = labels[index(x, line - 1)];
            direction = +1;
          }
        }
      }

      if (running &&
          (!boundary || component != run_component ||
           direction != run_direction)) {
        horizontal(run_component, run_direction,
                   min_y + static_cast<Coord>(line),
                   min_x + static_cast<Coord>(run_start),
                   min_x + static_cast<Coord>(x));
        running = false;
      }
      if (boundary && !running) {
        running = true;
        run_component = component;
        run_direction = direction;
        run_start = x;
      }
    }
  }

  /*
   * Vertical lattice scan.  Occupied left means a southward right side;
   * occupied right means a northward left side.
   */
  for (std::size_t line = 0; line <= width; ++line) {
    bool running = false;
    std::uint32_t run_component = 0;
    int run_direction = 0;
    std::size_t run_start = 0;

    for (std::size_t y = 0; y <= height; ++y) {
      bool boundary = false;
      std::uint32_t component = 0;
      int direction = 0;

      if (y < height) {
        const bool left = line > 0 && filled[index(line - 1, y)];
        const bool right = line < width && filled[index(line, y)];
        if (left != right) {
          boundary = true;
          if (left) {
            component = labels[index(line - 1, y)];
            direction = -1;
          } else {
            component = labels[index(line, y)];
            direction = +1;
          }
        }
      }

      if (running &&
          (!boundary || component != run_component ||
           direction != run_direction)) {
        vertical(run_component, run_direction,
                 min_x + static_cast<Coord>(line),
                 min_y + static_cast<Coord>(run_start),
                 min_y + static_cast<Coord>(y));
        running = false;
      }
      if (boundary && !running) {
        running = true;
        run_component = component;
        run_direction = direction;
        run_start = y;
      }
    }
  }

  std::sort(result.boundary.begin(), result.boundary.end(),
            [](const BoundaryFragment &first,
               const BoundaryFragment &second) {
              return std::tie(first.component, first.x1, first.y1, first.x2,
                              first.y2) <
                     std::tie(second.component, second.x1, second.y1,
                              second.x2, second.y2);
            });
  return result;
}

struct Fixture
{
  const char *name;
  std::vector<Rect> rectangles;
  std::uint64_t area;
  std::size_t components;
  std::size_t fragments;
  UnionStatus status;
  std::size_t kissing_vertices;
};

std::vector<Fixture> fixtures()
{
  return {
      {"overlap",
       {{0, 0, 3, 2}, {2, 1, 5, 4}},
       14,
       1,
       8,
       UnionStatus::Exact,
       0},
      {"containment",
       {{0, 0, 5, 5}, {1, 1, 4, 4}},
       25,
       1,
       4,
       UnionStatus::Exact,
       0},
      {"coincident",
       {{-2, -1, 3, 4}, {-2, -1, 3, 4}, {-2, -1, 3, 4}},
       25,
       1,
       4,
       UnionStatus::Exact,
       0},
      {"touching-edge",
       {{0, 0, 2, 2}, {2, 0, 5, 2}},
       10,
       1,
       4,
       UnionStatus::Exact,
       0},
      {"touching-corner",
       {{0, 0, 2, 2}, {2, 2, 5, 4}},
       10,
       2,
       8,
       UnionStatus::UnsupportedKissingVertex,
       1},
      {"l-shape",
       {{0, 0, 4, 1}, {0, 1, 1, 4}},
       7,
       1,
       6,
       UnionStatus::Exact,
       0},
      {"ring-with-hole",
       {{0, 0, 5, 1}, {0, 4, 5, 5}, {0, 1, 1, 4}, {4, 1, 5, 4}},
       16,
       1,
       8,
       UnionStatus::Exact,
       0},
      {"one-cell-channel",
       {{0, 0, 2, 5}, {5, 0, 7, 5}, {2, 2, 5, 3}},
       23,
       1,
       12,
       UnionStatus::Exact,
       0},
      /*
       * SW and NE cells kiss at (1,1), but are connected by the lower/right
       * path.  Four directed boundary fragments meet at the same vertex with
       * one component label: path reconstruction must not pair them
       * arbitrarily.
       */
      {"degree-4-self-kiss",
       {{0, -1, 3, 0}, {0, 0, 1, 1}, {2, 0, 3, 2}, {1, 1, 2, 2}},
       7,
       1,
       10,
       UnionStatus::UnsupportedKissingVertex,
       1},
  };
}

bool contains(const UnionResult &result, const BoundaryFragment &wanted)
{
  return std::find(result.boundary.begin(), result.boundary.end(), wanted) !=
         result.boundary.end();
}

std::int64_t signed_double_area(const UnionResult &result,
                                std::uint32_t component)
{
  std::int64_t sum = 0;
  for (const BoundaryFragment &fragment : result.boundary)
    if (fragment.component == component)
      sum += fragment.x1 * fragment.y2 - fragment.x2 * fragment.y1;
  return sum;
}

void compare(const std::string &name, const UnionResult &actual,
             const UnionResult &expected)
{
  if (actual == expected) return;
  std::ostringstream message;
  message << name << ": sparse/dense mismatch\nSPARSE\n"
          << describe(actual) << "\nDENSE\n"
          << describe(expected);
  throw std::runtime_error(message.str());
}

void test_fixtures()
{
  for (const Fixture &fixture : fixtures()) {
    const UnionResult sparse = unite(fixture.rectangles);
    const UnionResult dense = dense_reference(fixture.rectangles);
    compare(fixture.name, sparse, dense);

    require(sparse.area_cells == fixture.area,
            std::string(fixture.name) + ": unexpected area");
    require(sparse.components.size() == fixture.components,
            std::string(fixture.name) + ": unexpected component count");
    require(sparse.boundary.size() == fixture.fragments,
            std::string(fixture.name) + ": unexpected fragment count");
    require(sparse.status == fixture.status,
            std::string(fixture.name) + ": unexpected topology status");
    require(sparse.kissing_vertices.size() == fixture.kissing_vertices,
            std::string(fixture.name) +
                ": unexpected kissing-vertex count");

    for (const Component &component : sparse.components)
      require(signed_double_area(sparse, component.id) ==
                  -2 * static_cast<std::int64_t>(component.area_cells),
              std::string(fixture.name) +
                  ": boundary is not clockwise/material-right");

    std::vector<Rect> permuted = fixture.rectangles;
    std::reverse(permuted.begin(), permuted.end());
    require(unite(permuted) == sparse,
            std::string(fixture.name) +
                ": result depends on rectangle order");

    std::cout << "MANHATTAN_UNION_ORACLE_FIXTURE ok name=" << fixture.name
              << " area=" << sparse.area_cells
              << " components=" << sparse.components.size()
              << " fragments=" << sparse.boundary.size()
              << " status="
              << (sparse.status == UnionStatus::Exact
                      ? "Exact"
                      : "UnsupportedKissingVertex")
              << " kissing=" << sparse.kissing_vertices.size() << '\n';
  }
}

void test_orientation()
{
  const UnionResult rectangle = unite({{0, 0, 3, 2}});
  require(contains(rectangle, {0, 3, 0, 0, 0}), "bottom is not westward");
  require(contains(rectangle, {0, 0, 0, 0, 2}), "left is not northward");
  require(contains(rectangle, {0, 0, 2, 3, 2}), "top is not eastward");
  require(contains(rectangle, {0, 3, 2, 3, 0}), "right is not southward");
  require(signed_double_area(rectangle, 0) == -12,
          "outer rectangle is not clockwise");

  const UnionResult ring = unite(
      {{0, 0, 5, 1}, {0, 4, 5, 5}, {0, 1, 1, 4}, {4, 1, 5, 4}});
  require(contains(ring, {0, 5, 0, 0, 0}), "outer bottom orientation");
  require(contains(ring, {0, 0, 0, 0, 5}), "outer left orientation");
  require(contains(ring, {0, 0, 5, 5, 5}), "outer top orientation");
  require(contains(ring, {0, 5, 5, 5, 0}), "outer right orientation");
  require(contains(ring, {0, 1, 1, 4, 1}), "hole bottom orientation");
  require(contains(ring, {0, 4, 1, 4, 4}), "hole right orientation");
  require(contains(ring, {0, 4, 4, 1, 4}), "hole top orientation");
  require(contains(ring, {0, 1, 4, 1, 1}), "hole left orientation");
  require(signed_double_area(ring, 0) == -32,
          "ring net orientation/area mismatch");

  const UnionResult kissing =
      unite({{0, -1, 3, 0}, {0, 0, 1, 1}, {2, 0, 3, 2},
             {1, 1, 2, 2}});
  const auto incident = std::count_if(
      kissing.boundary.begin(), kissing.boundary.end(),
      [](const BoundaryFragment &fragment) {
        return (fragment.x1 == 1 && fragment.y1 == 1) ||
               (fragment.x2 == 1 && fragment.y2 == 1);
      });
  require(incident == 4, "degree-4 kissing vertex was spuriously paired");
  require(kissing.components.size() == 1,
          "degree-4 self-kiss lost its positive-area connection");
  require(kissing.status == UnionStatus::UnsupportedKissingVertex &&
              kissing.kissing_vertices.size() == 1,
          "degree-4 self-kiss was not failed closed");
  require(kissing.kissing_vertices.front() ==
              KissingVertex{1, 1, KissingDiagonal::SouthwestNortheast, 0, 0},
          "degree-4 self-kiss diagnostic is not canonical");

  const UnionResult corner =
      unite({{0, 0, 2, 2}, {2, 2, 5, 4}});
  require(corner.status == UnionStatus::UnsupportedKissingVertex &&
              corner.kissing_vertices.size() == 1,
          "separate-component corner kiss was not failed closed");
  require(corner.kissing_vertices.front() ==
              KissingVertex{2, 2, KissingDiagonal::SouthwestNortheast, 0, 1},
          "separate-component kissing diagnostic is not canonical");

  std::cout << "MANHATTAN_UNION_ORACLE_ORIENTATION ok outer=clockwise"
            << " holes=counter-clockwise interior=right"
            << " kissing=UnsupportedKissingVertex\n";
}

std::vector<Rect> random_rectangles(std::mt19937_64 &random)
{
  std::uniform_int_distribution<int> count_distribution(0, 16);
  std::uniform_int_distribution<int> coordinate_distribution(-10, 10);
  std::vector<Rect> rectangles;
  const int count = count_distribution(random);
  rectangles.reserve(static_cast<std::size_t>(count) + 2);

  for (int index = 0; index < count; ++index) {
    int x1 = coordinate_distribution(random);
    int x2 = coordinate_distribution(random);
    int y1 = coordinate_distribution(random);
    int y2 = coordinate_distribution(random);
    if (x1 == x2) x2 += x2 < 10 ? 1 : -1;
    if (y1 == y2) y2 += y2 < 10 ? 1 : -1;
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    rectangles.push_back({x1, y1, x2, y2});

    /* Exercise exact duplicate elimination without changing the domain. */
    if ((random() & 31U) == 0) rectangles.push_back(rectangles.back());
  }
  return rectangles;
}

std::vector<Rect> split_one_rectangle(const std::vector<Rect> &rectangles)
{
  std::vector<Rect> split = rectangles;
  for (std::size_t index = 0; index < split.size(); ++index) {
    const Rect rectangle = split[index];
    if (distance(rectangle.left, rectangle.right) > 1) {
      const Coord middle = rectangle.left +
                           static_cast<Coord>(
                               distance(rectangle.left, rectangle.right) / 2);
      split[index] = {rectangle.left, rectangle.bottom, middle, rectangle.top};
      split.push_back(
          {middle, rectangle.bottom, rectangle.right, rectangle.top});
      return split;
    }
    if (distance(rectangle.bottom, rectangle.top) > 1) {
      const Coord middle =
          rectangle.bottom +
          static_cast<Coord>(
              distance(rectangle.bottom, rectangle.top) / 2);
      split[index] =
          {rectangle.left, rectangle.bottom, rectangle.right, middle};
      split.push_back(
          {rectangle.left, middle, rectangle.right, rectangle.top});
      return split;
    }
  }
  return split;
}

void test_randomized()
{
  constexpr std::uint64_t seed = UINT64_C(0x4d324f5241434c45);
  constexpr std::size_t cases = 5000;
  std::mt19937_64 random(seed);

  for (std::size_t index = 0; index < cases; ++index) {
    const std::vector<Rect> rectangles = random_rectangles(random);
    const UnionResult sparse = unite(rectangles);
    const UnionResult dense = dense_reference(rectangles);
    compare("random-" + std::to_string(index), sparse, dense);

    std::vector<Rect> shuffled = rectangles;
    std::shuffle(shuffled.begin(), shuffled.end(), random);
    require(unite(shuffled) == sparse,
            "random input-order invariance failure at case " +
                std::to_string(index));

    const std::vector<Rect> split = split_one_rectangle(rectangles);
    require(unite(split) == sparse,
            "rectangle-decomposition invariance failure at case " +
                std::to_string(index));

    for (const Component &component : sparse.components)
      require(signed_double_area(sparse, component.id) ==
                  -2 * static_cast<std::int64_t>(component.area_cells),
              "random orientation failure at case " +
                  std::to_string(index));
  }

  std::cout << "MANHATTAN_UNION_ORACLE_RANDOM ok seed=0x" << std::hex
            << seed << std::dec << " cases=" << cases
            << " differentials=dense"
            << " metamorphic=input-order,rectangle-split\n";
}

template <class Exception, class Function>
void require_throws(const std::string &name, Function function)
{
  try {
    function();
  } catch (const Exception &) {
    return;
  } catch (const std::exception &error) {
    throw std::runtime_error(name + ": wrong exception: " + error.what());
  }
  throw std::runtime_error(name + ": expected exception was not thrown");
}

void test_limits()
{
  require(unite({}) == UnionResult(), "empty input is not empty");
  require_throws<std::invalid_argument>(
      "zero-width", [] { static_cast<void>(unite({{0, 0, 0, 1}})); });
  require_throws<std::invalid_argument>(
      "reversed", [] { static_cast<void>(unite({{2, 0, 1, 1}})); });

  Limits axis;
  axis.max_axis_span = 4;
  require_throws<std::length_error>(
      "axis limit",
      [&axis] { static_cast<void>(unite({{0, 0, 5, 1}}, axis)); });

  Limits cells;
  cells.max_occupied_cells = 3;
  require_throws<std::length_error>(
      "cell limit",
      [&cells] { static_cast<void>(unite({{0, 0, 2, 2}}, cells)); });

  Limits count;
  count.max_rectangles = 1;
  require_throws<std::length_error>(
      "rectangle limit", [&count] {
        static_cast<void>(
            unite({{0, 0, 1, 1}, {2, 2, 3, 3}}, count));
      });

  const Coord maximum = std::numeric_limits<Coord>::max();
  const Coord minimum = std::numeric_limits<Coord>::min();
  const UnionResult high = unite({{maximum - 1, maximum - 1, maximum,
                                   maximum}});
  const UnionResult low =
      unite({{minimum, minimum, minimum + 1, minimum + 1}});
  require(high.area_cells == 1 && low.area_cells == 1,
          "int64 edge coordinates are not exact");

  std::cout << "MANHATTAN_UNION_ORACLE_LIMITS ok malformed=2 capacity=3"
            << " int64-edges=2\n";
}

}  // namespace

int main()
{
  try {
    test_fixtures();
    test_orientation();
    test_randomized();
    test_limits();
    std::cout << "MANHATTAN_UNION_ORACLE PASS fixtures=" << fixtures().size()
              << " random=5000 orientation=material-right"
              << " components=4-neighbour-diagnostic"
              << " max-coherence-kissing=unsupported\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MANHATTAN_UNION_ORACLE FAIL " << error.what() << '\n';
    return 1;
  }
}
