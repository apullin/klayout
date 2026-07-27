/*
 * CPU-only production-capture census for the singleton geometry quotient.
 *
 * This reuses the authenticated compact hierarchy, exact Manhattan
 * decomposition, and exact integer transforms.  It never invokes CUDA.  The
 * tool is intentionally a proof harness: it reports the pre-membership
 * rectangle and membership universes that a device implementation must
 * reproduce before it is allowed into Connectivity.
 */

#include "antenna_geometry_quotient.h"
#include "antenna_m1_m4_capture_file.h"
#include "m2_manhattan_decompose.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aq = klayout_cuda::antenna_geometry_quotient;
namespace ac = klayout_cuda::antenna_connectivity;
namespace cap = klayout_cuda::antenna_m1_m4_capture;
namespace md = klayout_cuda::m2_manhattan_decompose;

struct LocalRectangle
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint32_t local_owner;
};

struct CellGeometry
{
  std::vector<LocalRectangle> rectangles;
  std::uint32_t owners = 0;
};

std::int64_t floor_div(std::int64_t value, std::int64_t divisor)
{
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) --quotient;
  return quotient;
}

std::uint64_t membership_count(
    const ac::RectI64 &rectangle, std::int64_t bin_size)
{
  const std::int64_t x0 = floor_div(rectangle.left, bin_size);
  const std::int64_t x1 = floor_div(rectangle.right, bin_size);
  const std::int64_t y0 = floor_div(rectangle.bottom, bin_size);
  const std::int64_t y1 = floor_div(rectangle.top, bin_size);
  return static_cast<std::uint64_t>(x1 - x0 + 1) *
         static_cast<std::uint64_t>(y1 - y0 + 1);
}

ac::RectI64 transform(
    const cap::Context &context, const LocalRectangle &input,
    std::uint32_t owner, std::uint32_t domain)
{
  const std::int64_t xs[2] = {input.left, input.right};
  const std::int64_t ys[2] = {input.bottom, input.top};
  ac::RectI64 output;
  bool first = true;
  for (int xi = 0; xi != 2; ++xi) {
    for (int yi = 0; yi != 2; ++yi) {
      __int128 x = xs[xi];
      __int128 y = ys[yi];
      __int128 tx = 0;
      __int128 ty = 0;
      switch (context.transform_code) {
      case 0: tx = x;  ty = y;  break;
      case 1: tx = -y; ty = x;  break;
      case 2: tx = -x; ty = -y; break;
      case 3: tx = y;  ty = -x; break;
      case 4: tx = x;  ty = -y; break;
      case 5: tx = y;  ty = x;  break;
      case 6: tx = -x; ty = y;  break;
      case 7: tx = -y; ty = -x; break;
      default: throw std::runtime_error("invalid transform code");
      }
      tx += context.tx;
      ty += context.ty;
      if (tx < std::numeric_limits<std::int64_t>::min() ||
          tx > std::numeric_limits<std::int64_t>::max() ||
          ty < std::numeric_limits<std::int64_t>::min() ||
          ty > std::numeric_limits<std::int64_t>::max()) {
        throw std::runtime_error("transformed coordinate overflow");
      }
      const std::int64_t xx = static_cast<std::int64_t>(tx);
      const std::int64_t yy = static_cast<std::int64_t>(ty);
      if (first) {
        output = {xx, yy, xx, yy, owner, domain};
        first = false;
      } else {
        output.left = std::min(output.left, xx);
        output.bottom = std::min(output.bottom, yy);
        output.right = std::max(output.right, xx);
        output.top = std::max(output.top, yy);
      }
    }
  }
  return output;
}

std::vector<CellGeometry> lower_cells(
    const cap::DomainStorage &storage)
{
  std::vector<CellGeometry> cells(storage.cells.size());
  for (std::size_t cell_id = 0;
       cell_id != storage.cells.size(); ++cell_id) {
    const cap::Cell &cell = storage.cells[cell_id];
    CellGeometry &target = cells[cell_id];
    target.owners = cell.polygon_count;
    for (std::uint32_t local_owner = 0;
         local_owner != cell.polygon_count; ++local_owner) {
      const cap::Polygon &polygon =
          storage.polygons[cell.polygon_begin + local_owner];
      if (polygon.edge_count == 4) {
        target.rectangles.push_back(
            {polygon.left, polygon.bottom, polygon.right, polygon.top,
             local_owner});
        continue;
      }
      std::vector<md::EdgeI64> edges;
      edges.reserve(polygon.edge_count);
      for (std::uint64_t edge_id = polygon.edge_begin;
           edge_id != polygon.edge_begin + polygon.edge_count;
           ++edge_id) {
        const cap::Edge &edge = storage.edges[edge_id];
        edges.push_back({edge.x1, edge.y1, edge.x2, edge.y2});
      }
      const md::Result decomposition = md::decompose(
          edges, polygon.left, polygon.bottom,
          polygon.right, polygon.top, local_owner,
          std::numeric_limits<std::uint64_t>::max());
      if (decomposition.status != md::Status::complete) {
        throw std::runtime_error(
            "exact Manhattan decomposition failed: " +
            decomposition.message);
      }
      for (const md::RectangleI64 &rectangle :
           decomposition.rectangles) {
        target.rectangles.push_back(
            {rectangle.left, rectangle.bottom,
             rectangle.right, rectangle.top, local_owner});
      }
    }
  }
  return cells;
}

std::vector<ac::RectI64> expand_domain(
    const cap::OwnedRequest &owned, std::uint32_t role,
    std::uint64_t *owner_count)
{
  const std::vector<CellGeometry> cells =
      lower_cells(owned.domains[role]);
  std::uint64_t rectangle_count = 0;
  for (const cap::Context &context : owned.contexts) {
    rectangle_count += cells[context.cell_id].rectangles.size();
  }
  if (rectangle_count > UINT32_MAX) {
    throw std::runtime_error("expanded rectangle count exceeds uint32");
  }
  std::vector<ac::RectI64> rectangles;
  rectangles.reserve(static_cast<std::size_t>(rectangle_count));
  std::uint64_t owners = 0;
  for (const cap::Context &context : owned.contexts) {
    const CellGeometry &cell = cells[context.cell_id];
    if (owners > UINT32_MAX - cell.owners) {
      throw std::runtime_error("expanded owner count exceeds uint32");
    }
    for (const LocalRectangle &rectangle : cell.rectangles) {
      rectangles.push_back(transform(
          context, rectangle,
          static_cast<std::uint32_t>(
              owners + rectangle.local_owner),
          role));
    }
    owners += cell.owners;
  }
  *owner_count = owners;
  return rectangles;
}

aq::Config quotient_config(
    std::uint32_t role, std::uint32_t owner_count)
{
  aq::Config config;
  config.domain_count = 12;
  config.owner_count = owner_count;
  config.owner_begin = 0;
  const std::uint32_t graph_roles[] = {
      0, 4, 5, 6, 7, 8, 9, 10, 11};
  for (std::uint32_t domain : graph_roles) {
    config.relation_rows[domain] |= UINT64_C(1) << domain;
  }
  const std::uint32_t adjacent[][2] = {
      {0, 4}, {4, 5}, {5, 6}, {6, 7},
      {7, 8}, {8, 9}, {9, 10}, {10, 11}};
  for (const auto &edge : adjacent) {
    config.relation_rows[edge[0]] |= UINT64_C(1) << edge[1];
    config.relation_rows[edge[1]] |= UINT64_C(1) << edge[0];
  }
  if (!(config.relation_rows[role] & (UINT64_C(1) << role))) {
    throw std::runtime_error("requested role is not a graph domain");
  }
  return config;
}

void require_production_x2(
    std::uint32_t role, const aq::Result &result)
{
  struct Expected
  {
    std::uint64_t owners;
    std::uint64_t rectangles;
    std::uint64_t classes;
    std::uint64_t star_edges;
    std::uint64_t weighted_internal_pairs;
  };
  const Expected expected[] = {
      {41093878, 41098970, 15965424, 25125674, 257379546},
      {20178022, 20178022, 1875312, 18302710, 248071522},
      {22945976, 22946444, 5927678, 17017830, 232292204}};
  const std::uint32_t roles[] = {5, 6, 7};
  for (std::size_t index = 0; index != 3; ++index) {
    if (roles[index] != role) continue;
    const Expected &item = expected[index];
    if (result.census.owners != item.owners ||
        result.census.input_rectangles != item.rectangles ||
        result.census.geometry_classes != item.classes ||
        result.census.star_edges != item.star_edges ||
        result.census.weighted_internal_pairs !=
            item.weighted_internal_pairs) {
      throw std::runtime_error(
          "production-x2 quotient census mismatch for role " +
          std::to_string(role));
    }
    return;
  }
  throw std::runtime_error("unexpected production role");
}

int main(int argc, char **argv)
{
  try {
    if (argc < 2 || argc > 4) {
      std::cerr
          << "usage: " << argv[0]
          << " CAPTURE.kam4 [BIN_SIZE] [--expect-production-x2]\n";
      return 2;
    }
    std::int64_t bin_size = 1000;
    bool expect_production = false;
    for (int argument = 2; argument < argc; ++argument) {
      const std::string value = argv[argument];
      if (value == "--expect-production-x2") {
        expect_production = true;
      } else {
        char *end = nullptr;
        const long long parsed =
            std::strtoll(value.c_str(), &end, 10);
        if (!end || *end || parsed <= 0) {
          throw std::runtime_error("invalid bin size");
        }
        bin_size = parsed;
      }
    }

    cap::OwnedRequest owned;
    std::string error;
    if (!cap::load_request(argv[1], owned, &error)) {
      throw std::runtime_error("capture load failed: " + error);
    }
    const std::uint32_t roles[] = {5, 6, 7};
    for (std::uint32_t role : roles) {
      std::uint64_t owner_count = 0;
      std::vector<ac::RectI64> rectangles =
          expand_domain(owned, role, &owner_count);
      if (owner_count !=
          owned.request.domains[role].flat_polygon_count) {
        throw std::runtime_error(
            "expanded owner count disagrees with authenticated census");
      }
      std::uint64_t input_memberships = 0;
      for (const ac::RectI64 &rectangle : rectangles) {
        input_memberships += membership_count(rectangle, bin_size);
      }
      aq::Result quotient;
      const aq::Status status = aq::build(
          quotient_config(
              role, static_cast<std::uint32_t>(owner_count)),
          rectangles.data(), rectangles.size(), &quotient);
      if (status != aq::Status::success) {
        throw std::runtime_error(
            std::string("quotient failed: ") +
            aq::status_string(status));
      }
      std::uint64_t representative_memberships = 0;
      for (const ac::RectI64 &rectangle :
           quotient.representative_rectangles) {
        representative_memberships +=
            membership_count(rectangle, bin_size);
      }
      std::uint64_t exception_memberships = 0;
      for (const ac::RectI64 &rectangle :
           quotient.exception_rectangles) {
        exception_memberships +=
            membership_count(rectangle, bin_size);
      }
      const std::uint64_t quotient_memberships =
          representative_memberships + exception_memberships;
      if (expect_production) {
        require_production_x2(role, quotient);
      }
      std::cout
          << "QUOTIENT role=" << role
          << " owners=" << quotient.census.owners
          << " input_rectangles="
          << quotient.census.input_rectangles
          << " singleton_owners="
          << quotient.census.singleton_owners
          << " exception_owners="
          << quotient.census.exception_owners
          << " exception_rectangles="
          << quotient.census.exception_rectangles
          << " classes=" << quotient.census.geometry_classes
          << " collapsed="
          << quotient.census.collapsed_rectangles
          << " work_rectangles="
          << quotient.census.work_rectangles
          << " star_edges=" << quotient.census.star_edges
          << " weighted_internal_pairs="
          << quotient.census.weighted_internal_pairs
          << " input_memberships=" << input_memberships
          << " representative_memberships="
          << representative_memberships
          << " exception_memberships="
          << exception_memberships
          << " quotient_memberships="
          << quotient_memberships
          << " rectangle_reduction_pct="
          << (100.0 *
              static_cast<double>(
                  quotient.census.input_rectangles -
                  quotient.census.work_rectangles) /
              quotient.census.input_rectangles)
          << " membership_reduction_pct="
          << (100.0 *
              static_cast<double>(
                  input_memberships - quotient_memberships) /
              input_memberships)
          << "\n";
    }
    std::cout
        << "antenna_geometry_quotient_capture_census: PASS"
        << " exact_transform=1 exact_decomposition=1"
        << " production_expectation="
        << (expect_production ? 1 : 0) << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &exception) {
    std::cerr
        << "antenna_geometry_quotient_capture_census: FAIL: "
        << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}
