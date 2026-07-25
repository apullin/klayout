#include "manhattan_union_oracle.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace klayout_cuda {
namespace manhattan_union_oracle {

namespace {

struct CellLess
{
  bool operator()(const Cell &first, const Cell &second) const
  {
    return std::tie(first.x, first.y) < std::tie(second.x, second.y);
  }
};

enum class Axis : std::uint8_t
{
  horizontal,
  vertical
};

struct UnitBoundary
{
  std::uint32_t component;
  Axis axis;
  Coord fixed;
  Coord lo;
  Coord hi;
  int direction;
};

bool unit_boundary_less(const UnitBoundary &first,
                        const UnitBoundary &second)
{
  return std::tie(first.component, first.axis, first.fixed, first.direction,
                  first.lo, first.hi) <
         std::tie(second.component, second.axis, second.fixed, second.direction,
                  second.lo, second.hi);
}

bool fragment_less(const BoundaryFragment &first,
                   const BoundaryFragment &second)
{
  return std::tie(first.component, first.x1, first.y1, first.x2, first.y2) <
         std::tie(second.component, second.x1, second.y1, second.x2, second.y2);
}

std::uint64_t unsigned_distance(Coord lower, Coord upper)
{
  return static_cast<std::uint64_t>(upper) -
         static_cast<std::uint64_t>(lower);
}

BoundaryFragment make_fragment(const UnitBoundary &edge)
{
  if (edge.axis == Axis::horizontal) {
    if (edge.direction > 0)
      return {edge.component, edge.lo, edge.fixed, edge.hi, edge.fixed};
    return {edge.component, edge.hi, edge.fixed, edge.lo, edge.fixed};
  }

  if (edge.direction > 0)
    return {edge.component, edge.fixed, edge.lo, edge.fixed, edge.hi};
  return {edge.component, edge.fixed, edge.hi, edge.fixed, edge.lo};
}

}  // namespace

bool operator==(const Cell &first, const Cell &second)
{
  return first.x == second.x && first.y == second.y;
}

bool operator==(const BoundaryFragment &first,
                const BoundaryFragment &second)
{
  return first.component == second.component && first.x1 == second.x1 &&
         first.y1 == second.y1 && first.x2 == second.x2 &&
         first.y2 == second.y2;
}

bool operator==(const KissingVertex &first, const KissingVertex &second)
{
  return first.x == second.x && first.y == second.y &&
         first.diagonal == second.diagonal &&
         first.first_component == second.first_component &&
         first.second_component == second.second_component;
}

bool operator==(const Component &first, const Component &second)
{
  return first.id == second.id && first.least_cell == second.least_cell &&
         first.area_cells == second.area_cells;
}

bool operator==(const UnionResult &first, const UnionResult &second)
{
  return first.status == second.status &&
         first.area_cells == second.area_cells &&
         first.components == second.components &&
         first.boundary == second.boundary &&
         first.kissing_vertices == second.kissing_vertices;
}

UnionResult unite(const std::vector<Rect> &rectangles, const Limits &limits)
{
  if (rectangles.size() > limits.max_rectangles)
    throw std::length_error("rectangle-count oracle limit exceeded");

  UnionResult result;
  if (rectangles.empty()) return result;

  Coord min_x = rectangles.front().left;
  Coord min_y = rectangles.front().bottom;
  Coord max_x = rectangles.front().right;
  Coord max_y = rectangles.front().top;

  for (const Rect &rectangle : rectangles) {
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top)
      throw std::invalid_argument("rectangle is empty or reversed");

    min_x = std::min(min_x, rectangle.left);
    min_y = std::min(min_y, rectangle.bottom);
    max_x = std::max(max_x, rectangle.right);
    max_y = std::max(max_y, rectangle.top);
  }

  const std::uint64_t width = unsigned_distance(min_x, max_x);
  const std::uint64_t height = unsigned_distance(min_y, max_y);
  if (width > limits.max_axis_span || height > limits.max_axis_span)
    throw std::length_error("axis-span oracle limit exceeded");

  std::set<Cell, CellLess> occupied;
  for (const Rect &rectangle : rectangles) {
    for (Coord x = rectangle.left; x < rectangle.right; ++x) {
      for (Coord y = rectangle.bottom; y < rectangle.top; ++y) {
        occupied.insert({x, y});
        if (occupied.size() > limits.max_occupied_cells)
          throw std::length_error("occupied-cell oracle limit exceeded");
      }
    }
  }

  result.area_cells = occupied.size();
  std::map<Cell, std::uint32_t, CellLess> labels;

  constexpr std::array<std::pair<Coord, Coord>, 4> neighbours = {
      std::pair<Coord, Coord>{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

  for (const Cell &seed : occupied) {
    if (labels.find(seed) != labels.end()) continue;

    const std::uint32_t id =
        static_cast<std::uint32_t>(result.components.size());
    std::queue<Cell> pending;
    pending.push(seed);
    labels.emplace(seed, id);
    std::uint64_t area = 0;

    while (!pending.empty()) {
      const Cell cell = pending.front();
      pending.pop();
      ++area;

      for (const auto &delta : neighbours) {
        /*
         * The bounded span guarantees ordinary cases, but avoid signed
         * overflow so adversarial fixtures fail closed rather than invoking
         * undefined behaviour.
         */
        if ((delta.first < 0 &&
             cell.x == std::numeric_limits<Coord>::min()) ||
            (delta.first > 0 &&
             cell.x == std::numeric_limits<Coord>::max()) ||
            (delta.second < 0 &&
             cell.y == std::numeric_limits<Coord>::min()) ||
            (delta.second > 0 &&
             cell.y == std::numeric_limits<Coord>::max()))
          continue;

        const Cell next = {cell.x + delta.first, cell.y + delta.second};
        if (occupied.find(next) == occupied.end()) continue;
        if (labels.emplace(next, id).second) pending.push(next);
      }
    }

    result.components.push_back({id, seed, area});
  }

  const auto label_at = [&labels](Coord x, Coord y,
                                  std::uint32_t *component) {
    const auto found = labels.find({x, y});
    if (found == labels.end()) return false;
    *component = found->second;
    return true;
  };

  /*
   * Detect every checkerboard lattice vertex explicitly.  Four-neighbour
   * occupancy labels remain useful diagnostics, but they are not claimed to
   * reproduce KLayout's maximum-coherence contour pairing.
   */
  for (Coord x = min_x;; ++x) {
    for (Coord y = min_y;; ++y) {
      std::uint32_t southwest_component = 0;
      std::uint32_t southeast_component = 0;
      std::uint32_t northwest_component = 0;
      std::uint32_t northeast_component = 0;
      const bool has_west = x != std::numeric_limits<Coord>::min();
      const bool has_south = y != std::numeric_limits<Coord>::min();
      const bool southwest =
          has_west && has_south &&
          label_at(x - 1, y - 1, &southwest_component);
      const bool southeast =
          has_south && label_at(x, y - 1, &southeast_component);
      const bool northwest =
          has_west && label_at(x - 1, y, &northwest_component);
      const bool northeast =
          label_at(x, y, &northeast_component);

      if (southwest && northeast && !southeast && !northwest) {
        result.kissing_vertices.push_back(
            {x, y, KissingDiagonal::SouthwestNortheast,
             southwest_component, northeast_component});
      } else if (southeast && northwest && !southwest && !northeast) {
        result.kissing_vertices.push_back(
            {x, y, KissingDiagonal::NorthwestSoutheast,
             northwest_component, southeast_component});
      }

      if (y == max_y) break;
    }
    if (x == max_x) break;
  }
  if (!result.kissing_vertices.empty())
    result.status = UnionStatus::UnsupportedKissingVertex;

  std::vector<UnitBoundary> unit_edges;
  unit_edges.reserve(occupied.size() * 2);

  const auto is_occupied = [&occupied](Coord x, Coord y) {
    return occupied.find({x, y}) != occupied.end();
  };

  for (const Cell &cell : occupied) {
    const std::uint32_t component = labels.at(cell);

    /*
     * Direction is selected so occupied material is on the right, matching
     * KLayout's canonical hull convention:
     * bottom west, right south, top east, left north.
     */
    if (cell.y == std::numeric_limits<Coord>::min() ||
        !is_occupied(cell.x, cell.y - 1))
      unit_edges.push_back(
          {component, Axis::horizontal, cell.y, cell.x, cell.x + 1, -1});

    if (!is_occupied(cell.x + 1, cell.y))
      unit_edges.push_back(
          {component, Axis::vertical, cell.x + 1, cell.y, cell.y + 1, -1});

    if (!is_occupied(cell.x, cell.y + 1))
      unit_edges.push_back({component, Axis::horizontal, cell.y + 1, cell.x,
                            cell.x + 1, +1});

    if (cell.x == std::numeric_limits<Coord>::min() ||
        !is_occupied(cell.x - 1, cell.y))
      unit_edges.push_back(
          {component, Axis::vertical, cell.x, cell.y, cell.y + 1, +1});
  }

  std::sort(unit_edges.begin(), unit_edges.end(), unit_boundary_less);
  std::vector<UnitBoundary> maximal;
  maximal.reserve(unit_edges.size());

  for (const UnitBoundary &edge : unit_edges) {
    if (!maximal.empty()) {
      UnitBoundary &previous = maximal.back();
      if (previous.component == edge.component &&
          previous.axis == edge.axis && previous.fixed == edge.fixed &&
          previous.direction == edge.direction && previous.hi == edge.lo) {
        previous.hi = edge.hi;
        continue;
      }
    }
    maximal.push_back(edge);
  }

  result.boundary.reserve(maximal.size());
  for (const UnitBoundary &edge : maximal)
    result.boundary.push_back(make_fragment(edge));

  std::sort(result.boundary.begin(), result.boundary.end(), fragment_less);
  return result;
}

std::string describe(const UnionResult &result)
{
  std::ostringstream stream;
  stream << "status="
         << (result.status == UnionStatus::Exact ? "Exact"
                                                : "UnsupportedKissingVertex")
         << " area=" << result.area_cells
         << " components=" << result.components.size()
         << " fragments=" << result.boundary.size()
         << " kissing_vertices=" << result.kissing_vertices.size();

  for (const Component &component : result.components)
    stream << "\nC " << component.id << " least=(" << component.least_cell.x
           << ',' << component.least_cell.y << ") area="
           << component.area_cells;

  for (const BoundaryFragment &fragment : result.boundary)
    stream << "\nE " << fragment.component << " (" << fragment.x1 << ','
           << fragment.y1 << ")->(" << fragment.x2 << ',' << fragment.y2
           << ')';

  for (const KissingVertex &vertex : result.kissing_vertices)
    stream << "\nK (" << vertex.x << ',' << vertex.y << ") diagonal="
           << (vertex.diagonal == KissingDiagonal::SouthwestNortheast
                   ? "SW-NE"
                   : "NW-SE")
           << " components=" << vertex.first_component << ','
           << vertex.second_component;

  return stream.str();
}

}  // namespace manhattan_union_oracle
}  // namespace klayout_cuda
