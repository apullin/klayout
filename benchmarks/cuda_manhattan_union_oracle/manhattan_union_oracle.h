/*
 * Deliberately small, deliberately slow exact oracle for Manhattan unions.
 *
 * This code is independent of the CUDA sweep/scan implementation.  It expands
 * bounded integer-coordinate rectangles into occupied unit cells, labels
 * positive-area (4-neighbour) connected components, and extracts the exact
 * oriented boundary of that occupancy.  It is intended for differential
 * testing, not production geometry.
 */

#ifndef KLAYOUT_CUDA_MANHATTAN_UNION_ORACLE_H
#define KLAYOUT_CUDA_MANHATTAN_UNION_ORACLE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace klayout_cuda {
namespace manhattan_union_oracle {

using Coord = std::int64_t;

struct Rect
{
  Coord left;
  Coord bottom;
  Coord right;
  Coord top;
};

struct Cell
{
  Coord x;
  Coord y;
};

/*
 * A maximal collinear boundary fragment.  The occupied union is always on the
 * right while walking from (x1, y1) to (x2, y2).  This matches KLayout's
 * canonical hull convention: outer boundaries are clockwise and hole
 * boundaries are counter-clockwise.
 */
struct BoundaryFragment
{
  std::uint32_t component;
  Coord x1;
  Coord y1;
  Coord x2;
  Coord y2;
};

enum class UnionStatus : std::uint8_t
{
  Exact = 0,
  UnsupportedKissingVertex = 1
};

enum class KissingDiagonal : std::uint8_t
{
  SouthwestNortheast = 0,
  NorthwestSoutheast = 1
};

/*
 * A degree-4/checkerboard lattice vertex.  The occupied quadrants touch only
 * at (x,y).  KLayout's production default uses maximum coherence to pair such
 * corners into fewer polygons.  The bounded oracle deliberately does not
 * reproduce that contour-pairing policy: presence of any such record sets
 * status to UnsupportedKissingVertex so a qualified fast path fails closed.
 */
struct KissingVertex
{
  Coord x;
  Coord y;
  KissingDiagonal diagonal;
  /* first/second follow the order named by diagonal. */
  std::uint32_t first_component;
  std::uint32_t second_component;
};

struct Component
{
  /*
   * Components are numbered by their lexicographically least occupied cell
   * (x first, then y), making labels independent of input rectangle order.
   */
  std::uint32_t id;
  Cell least_cell;
  std::uint64_t area_cells;
};

struct UnionResult
{
  UnionStatus status = UnionStatus::Exact;
  std::uint64_t area_cells = 0;
  std::vector<Component> components;
  std::vector<BoundaryFragment> boundary;
  std::vector<KissingVertex> kissing_vertices;
};

struct Limits
{
  std::uint64_t max_axis_span = 256;
  std::uint64_t max_occupied_cells = 65536;
  std::size_t max_rectangles = 4096;
};

bool operator==(const Cell &first, const Cell &second);
bool operator==(const BoundaryFragment &first,
                const BoundaryFragment &second);
bool operator==(const KissingVertex &first, const KissingVertex &second);
bool operator==(const Component &first, const Component &second);
bool operator==(const UnionResult &first, const UnionResult &second);

/*
 * Throws std::invalid_argument for malformed rectangles and
 * std::length_error when the deliberately bounded oracle limits are exceeded.
 */
UnionResult unite(const std::vector<Rect> &rectangles,
                  const Limits &limits = Limits());

std::string describe(const UnionResult &result);

}  // namespace manhattan_union_oracle
}  // namespace klayout_cuda

#endif
