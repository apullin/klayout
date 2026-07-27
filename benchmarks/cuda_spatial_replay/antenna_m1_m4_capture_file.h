/*
 * Portable capture file for the compact ANTENNA.M1-through-M4 request ABI.
 *
 * The file owns values, never ABI pointers or reserved storage.  All integer
 * fields and records are encoded explicitly as little-endian values and every
 * payload section is authenticated by a SHA-256 digest which is itself bound
 * by the header digest.
 */

#ifndef KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_FILE_H
#define KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_FILE_H

#include "dbCudaSpatialApi.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace klayout_cuda {
namespace antenna_m1_m4_capture {

using Request = klayout_cuda_spatial_antenna_m1_m4_request_v1;
using Context = klayout_cuda_spatial_m1_width_space_context_v1;
using Cell = klayout_cuda_spatial_antenna_m1_m4_cell_v1;
using Polygon = klayout_cuda_spatial_m1_width_space_polygon_v1;
using Edge = klayout_cuda_spatial_m1_width_space_edge_v1;

struct DomainStorage
{
  std::vector<Cell> cells;
  std::vector<Polygon> polygons;
  std::vector<Edge> edges;
};

/*
 * request is always rebound to this object's vectors after construction,
 * copy, move, assignment, and a successful load_request call.
 */
struct OwnedRequest
{
  Request request;
  std::vector<std::uint64_t> source_cell_indices;
  std::vector<Context> contexts;
  std::vector<std::uint32_t> context_parent_ids;
  std::array<DomainStorage,
             KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT> domains;

  OwnedRequest () noexcept;
  OwnedRequest (const OwnedRequest &other);
  OwnedRequest (OwnedRequest &&other) noexcept;
  OwnedRequest &operator= (const OwnedRequest &other);
  OwnedRequest &operator= (OwnedRequest &&other) noexcept;

  void rebind () noexcept;
};

/*
 * Both functions are fail-closed and do not throw.  error is cleared on
 * success and receives a diagnostic on failure when non-null.
 */
bool dump_request (const std::string &path, const Request &request,
                   std::string *error = nullptr) noexcept;

bool load_request (const std::string &path, OwnedRequest &request,
                   std::string *error = nullptr) noexcept;

}  // namespace antenna_m1_m4_capture
}  // namespace klayout_cuda

#endif
