/*
 * Fail-closed production bridge from KACTSCN1 raw M2 hierarchy to an
 * expanded CUDA rectangle stream.
 *
 * This experiment deliberately stops before allocating the exact union's
 * y-events.  It establishes the production rectangle census and measures the
 * global coordinate-compressed x-slab work that the first Manhattan-union
 * prototype would have to materialize.
 */

#define main klayout_cuda_embedded_active3_scene_main
#include "active3_scene_island.cu"
#undef main

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kM2LogicalLayer = 0;
constexpr std::uint32_t kQualifiedM2Layer = 101;
constexpr std::uint32_t kQualifiedM2Datatype = 0;
constexpr std::uint32_t kCensusThreads = 256;
constexpr std::uint64_t kDefaultMaxRectangles = UINT64_C(32000000);
constexpr std::uint64_t kDefaultMaxEndpoints = UINT64_C(64000000);

enum M2CensusDeviceFlag : std::uint32_t {
  kM2CensusTransformOverflow = 1u << 0,
  kM2CensusInvalidRecord = 1u << 1,
  kM2CensusEndpointLookup = 1u << 2,
};

// This is byte-for-byte compatible with the standalone Manhattan-union
// replay's RectI64 seam.  source_token identifies the stored KACT polygon;
// context_token identifies the exact resolved hierarchy occurrence.
struct M2CensusRect {
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint64_t source_token;
  std::uint64_t context_token;
};

static_assert(sizeof(M2CensusRect) == 48,
              "unexpected production rectangle ABI padding");

struct M2CensusRectTemplate {
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint64_t source_polygon;
};

struct M2CensusCell {
  std::uint64_t rectangle_begin;
  std::uint32_t rectangle_count;
  std::uint32_t polygon_count;
  std::uint32_t l_shape_count;
  std::uint32_t reserved;
};

struct M2CensusCompactScene {
  std::vector<ContextGpu> contexts;
  std::vector<std::uint32_t> m2_contexts;
  std::vector<std::uint64_t> rectangle_offsets;
  std::vector<M2CensusCell> cells;
  std::vector<M2CensusRectTemplate> rectangles;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_l_shapes = 0;
  std::uint64_t flat_rectangles = 0;
  std::uint64_t local_polygons = 0;
  std::uint64_t local_l_shapes = 0;
};

struct M2CensusOptions {
  std::string path;
  std::string expected_scene_sha256;
  bool verify_expanded_host = false;
  std::uint64_t expected_flat_polygons = 0;
  std::uint64_t expected_flat_rectangles = 0;
  std::uint64_t max_contexts = kDefaultMaxContexts;
  std::uint64_t max_rectangles = kDefaultMaxRectangles;
  std::uint64_t max_endpoints = kDefaultMaxEndpoints;
};

struct M2CensusTiming {
  double load_validate_ms = 0.0;
  double hierarchy_lower_ms = 0.0;
  double template_lower_ms = 0.0;
  double cuda_init_ms = 0.0;
  double upload_ms = 0.0;
  double expand_ms = 0.0;
  double host_verify_ms = 0.0;
  double x_sort_unique_ms = 0.0;
  double membership_ms = 0.0;
  double cleanup_ms = 0.0;
  double total_ms = 0.0;
};

struct M2CensusGpuCounters {
  unsigned long long expanded;
  unsigned long long span_over_4096;
  unsigned long long span_over_65535;
};

struct M2CensusPoint {
  std::int64_t x;
  std::int64_t y;
};

using M2CensusPolygon = std::vector<M2CensusPoint>;

std::uint64_t m2_parse_u64(const std::string &text, const char *name) {
  if (text.empty() || text[0] == '-') {
    throw SceneError(std::string(name) + " must be a positive integer");
  }
  std::size_t consumed = 0;
  const std::uint64_t value = std::stoull(text, &consumed);
  if (consumed != text.size() || !value) {
    throw SceneError(std::string(name) + " must be a positive integer");
  }
  return value;
}

M2CensusOptions m2_parse_options(int argc, char **argv) {
  M2CensusOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto take = [&](const char *prefix, std::uint64_t *destination) {
      const std::string marker = std::string(prefix) + "=";
      if (argument.rfind(marker, 0) != 0) {
        return false;
      }
      *destination =
          m2_parse_u64(argument.substr(marker.size()), prefix);
      return true;
    };
    if (take("--expect-flat-polygons", &options.expected_flat_polygons) ||
        take("--expect-flat-rectangles",
             &options.expected_flat_rectangles) ||
        take("--max-contexts", &options.max_contexts) ||
        take("--max-rectangles", &options.max_rectangles) ||
        take("--max-endpoints", &options.max_endpoints)) {
      continue;
    }
    const std::string sha_prefix = "--expect-scene-sha256=";
    if (argument.rfind(sha_prefix, 0) == 0) {
      options.expected_scene_sha256 =
          argument.substr(sha_prefix.size());
      if (options.expected_scene_sha256.size() != 64 ||
          !std::all_of(options.expected_scene_sha256.begin(),
                       options.expected_scene_sha256.end(),
                       [](unsigned char character) {
                         return std::isxdigit(character) != 0;
                       })) {
        throw SceneError(
            "--expect-scene-sha256 requires exactly 64 hex digits");
      }
      std::transform(options.expected_scene_sha256.begin(),
                     options.expected_scene_sha256.end(),
                     options.expected_scene_sha256.begin(),
                     [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                     });
      continue;
    }
    if (argument == "--verify-expanded-host") {
      options.verify_expanded_host = true;
      continue;
    }
    if (!argument.empty() && argument[0] == '-') {
      throw SceneError("unknown option: " + argument);
    }
    if (!options.path.empty()) {
      throw SceneError("exactly one KACTSCN1 input is required");
    }
    options.path = argument;
  }
  if (options.path.empty() || options.expected_scene_sha256.empty()) {
    throw SceneError(
        "usage: m2_manhattan_production_census "
        "--expect-scene-sha256=HEX [--expect-flat-polygons=N] "
        "[--expect-flat-rectangles=N] [--verify-expanded-host] "
        "[capacity options] SCENE.kact");
  }
  return options;
}

__int128 m2_abs_i128(__int128 value) {
  return value < 0 ? -value : value;
}

__int128 m2_polygon_twice_area(const M2CensusPolygon &points) {
  __int128 area = 0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const M2CensusPoint first = points[index];
    const M2CensusPoint second = points[(index + 1) % points.size()];
    area += static_cast<__int128>(first.x) * second.y -
            static_cast<__int128>(second.x) * first.y;
  }
  return area;
}

bool m2_point_inside(const M2CensusPolygon &points,
                     std::int64_t left, std::int64_t bottom,
                     std::int64_t right, std::int64_t top) {
  // Test the exact doubled-coordinate center.  The point is strictly inside a
  // compressed cell, so it can never lie on a polygon edge.
  const __int128 px2 =
      static_cast<__int128>(left) + static_cast<__int128>(right);
  const __int128 py2 =
      static_cast<__int128>(bottom) + static_cast<__int128>(top);
  bool inside = false;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const M2CensusPoint first = points[index];
    const M2CensusPoint second = points[(index + 1) % points.size()];
    if (first.x != second.x) {
      continue;
    }
    const __int128 y1 = static_cast<__int128>(first.y) * 2;
    const __int128 y2 = static_cast<__int128>(second.y) * 2;
    if ((y1 > py2) != (y2 > py2) &&
        static_cast<__int128>(first.x) * 2 > px2) {
      inside = !inside;
    }
  }
  return inside;
}

void m2_append_rectangle(
    std::vector<M2CensusRectTemplate> *destination,
    std::int64_t left, std::int64_t bottom, std::int64_t right,
    std::int64_t top, std::uint64_t source_polygon) {
  if (left >= right || bottom >= top) {
    throw SceneError("M2 rectangle decomposition produced an empty box");
  }
  destination->push_back(
      {left, bottom, right, top, source_polygon});
}

std::uint32_t m2_decompose_polygon(
    const LoadedScene &scene, const PolygonRecord &record,
    std::vector<M2CensusRectTemplate> *destination) {
  if (record.layer_code != kM2LogicalLayer ||
      (record.edge_count != 4 && record.edge_count != 6)) {
    throw SceneError(
        "raw M2 contains an unsupported non-box/non-L polygon");
  }

  M2CensusPolygon points;
  points.reserve(record.edge_count);
  std::vector<std::int64_t> xs;
  std::vector<std::int64_t> ys;
  xs.reserve(record.edge_count);
  ys.reserve(record.edge_count);
  for (std::uint32_t local = 0; local < record.edge_count; ++local) {
    const EdgeRecord &edge = scene.edges[record.edge_begin + local];
    points.push_back({edge.x1, edge.y1});
    xs.push_back(edge.x1);
    ys.push_back(edge.y1);
  }
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  std::sort(ys.begin(), ys.end());
  ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

  const __int128 polygon_area2 = m2_polygon_twice_area(points);
  if (polygon_area2 >= 0) {
    throw SceneError("raw M2 polygon lost clockwise normalization");
  }
  const std::size_t begin = destination->size();
  if (record.edge_count == 4) {
    if (xs.size() != 2 || ys.size() != 2 ||
        xs.front() != record.bbox[0] ||
        ys.front() != record.bbox[1] ||
        xs.back() != record.bbox[2] ||
        ys.back() != record.bbox[3]) {
      throw SceneError("four-edge raw M2 polygon is not a rectangle");
    }
    m2_append_rectangle(destination, xs[0], ys[0], xs[1], ys[1],
                        record.polygon_id);
  } else {
    if (xs.size() != 3 || ys.size() != 3 ||
        xs.front() != record.bbox[0] ||
        ys.front() != record.bbox[1] ||
        xs.back() != record.bbox[2] ||
        ys.back() != record.bbox[3]) {
      throw SceneError("six-edge raw M2 polygon is not a simple L");
    }
    bool occupied[2][2] = {};
    std::uint32_t occupied_count = 0;
    for (std::uint32_t x = 0; x < 2; ++x) {
      for (std::uint32_t y = 0; y < 2; ++y) {
        occupied[x][y] =
            m2_point_inside(points, xs[x], ys[y], xs[x + 1], ys[y + 1]);
        occupied_count += occupied[x][y] ? 1u : 0u;
      }
    }
    if (occupied_count != 3) {
      throw SceneError("six-edge raw M2 polygon is not a 3-cell L");
    }

    // One deterministic, nonoverlapping decomposition only: the complete
    // x-column first, followed by the occupied cell in the other column.
    // This deliberately avoids the prior VIA-stack path's simultaneous X and
    // Y slab decompositions.
    const std::uint32_t full_x =
        occupied[0][0] && occupied[0][1] ? 0u : 1u;
    const std::uint32_t other_x = 1u - full_x;
    if (!(occupied[full_x][0] && occupied[full_x][1]) ||
        occupied[other_x][0] == occupied[other_x][1]) {
      throw SceneError("six-edge raw M2 L has no unique full x-column");
    }
    m2_append_rectangle(destination, xs[full_x], ys[0],
                        xs[full_x + 1], ys[2], record.polygon_id);
    const std::uint32_t other_y = occupied[other_x][0] ? 0u : 1u;
    m2_append_rectangle(destination, xs[other_x], ys[other_y],
                        xs[other_x + 1], ys[other_y + 1],
                        record.polygon_id);
  }

  __int128 rectangle_area = 0;
  for (std::size_t index = begin; index < destination->size(); ++index) {
    const M2CensusRectTemplate &rectangle = (*destination)[index];
    rectangle_area +=
        static_cast<__int128>(rectangle.right - rectangle.left) *
        static_cast<__int128>(rectangle.top - rectangle.bottom);
  }
  if (rectangle_area * 2 != m2_abs_i128(polygon_area2)) {
    throw SceneError("raw M2 rectangle decomposition changed exact area");
  }
  return record.edge_count == 6 ? 1u : 0u;
}

M2CensusCompactScene m2_lower_scene(
    const LoadedScene &scene, const LoweredScene &hierarchy,
    std::uint64_t max_rectangles) {
  M2CensusCompactScene result;
  result.contexts = hierarchy.contexts;
  result.cells.resize(scene.header.cell_count);

  for (std::uint64_t cell_id = 0; cell_id < scene.header.cell_count;
       ++cell_id) {
    const CellRecord &source_cell = scene.cells[cell_id];
    M2CensusCell &cell = result.cells[cell_id];
    cell.rectangle_begin = result.rectangles.size();
    for (std::uint64_t local = 0; local < source_cell.polygon_count;
         ++local) {
      const PolygonRecord &polygon =
          scene.polygons[source_cell.polygon_begin + local];
      if (polygon.layer_code != kM2LogicalLayer) {
        continue;
      }
      if (cell.polygon_count == std::numeric_limits<std::uint32_t>::max()) {
        throw SceneError("per-cell raw M2 polygon count exceeds uint32");
      }
      ++cell.polygon_count;
      cell.l_shape_count +=
          m2_decompose_polygon(scene, polygon, &result.rectangles);
    }
    const std::uint64_t count =
        result.rectangles.size() - cell.rectangle_begin;
    if (count > std::numeric_limits<std::uint32_t>::max()) {
      throw SceneError("per-cell M2 rectangle count exceeds uint32");
    }
    cell.rectangle_count = static_cast<std::uint32_t>(count);
    result.local_polygons += cell.polygon_count;
    result.local_l_shapes += cell.l_shape_count;
  }

  for (std::uint32_t context_id = 0;
       context_id < result.contexts.size(); ++context_id) {
    const ContextGpu &context = result.contexts[context_id];
    const M2CensusCell &cell = result.cells[context.cell];
    if (!cell.rectangle_count) {
      continue;
    }
    result.m2_contexts.push_back(context_id);
    result.rectangle_offsets.push_back(result.flat_rectangles);
    if (!checked_add_u64(result.flat_rectangles, cell.rectangle_count,
                         &result.flat_rectangles) ||
        !checked_add_u64(result.flat_polygons, cell.polygon_count,
                         &result.flat_polygons) ||
        !checked_add_u64(result.flat_l_shapes, cell.l_shape_count,
                         &result.flat_l_shapes)) {
      throw SceneError("flat raw M2 census overflow");
    }
    if (result.flat_rectangles > max_rectangles) {
      throw SceneError("flat raw M2 rectangles exceed configured capacity");
    }
  }
  if (result.flat_rectangles !=
          result.flat_polygons + result.flat_l_shapes ||
      result.rectangles.size() !=
          result.local_polygons + result.local_l_shapes ||
      result.flat_rectangles == 0 || result.m2_contexts.empty()) {
    throw SceneError("raw M2 rectangle conservation failed");
  }
  return result;
}

M2CensusRect m2_transform_rectangle_host(
    const ContextGpu &context, const M2CensusRectTemplate &source,
    std::uint64_t context_token) {
  const std::int64_t xs[4] = {
      source.left, source.left, source.right, source.right};
  const std::int64_t ys[4] = {
      source.bottom, source.top, source.bottom, source.top};
  M2CensusRect destination{};
  for (int corner = 0; corner < 4; ++corner) {
    const auto transformed =
        transform_128(context.transform, xs[corner], ys[corner]);
    const std::int64_t x =
        narrow_i64(transformed.first + context.tx,
                   "host M2 rectangle x");
    const std::int64_t y =
        narrow_i64(transformed.second + context.ty,
                   "host M2 rectangle y");
    if (!corner) {
      destination.left = x;
      destination.right = x;
      destination.bottom = y;
      destination.top = y;
    } else {
      destination.left = std::min(destination.left, x);
      destination.right = std::max(destination.right, x);
      destination.bottom = std::min(destination.bottom, y);
      destination.top = std::max(destination.top, y);
    }
  }
  destination.source_token = source.source_polygon;
  destination.context_token = context_token;
  if (destination.left >= destination.right ||
      destination.bottom >= destination.top) {
    throw SceneError("host M2 transform produced an empty rectangle");
  }
  return destination;
}

bool m2_same_rectangle(const M2CensusRect &first,
                       const M2CensusRect &second) {
  return first.left == second.left &&
         first.bottom == second.bottom &&
         first.right == second.right && first.top == second.top &&
         first.source_token == second.source_token &&
         first.context_token == second.context_token;
}

__device__ bool m2_transform_rectangle(
    const ContextGpu &context, const M2CensusRectTemplate &source,
    M2CensusRect *destination, std::uint64_t context_token) {
  const std::int64_t xs[4] = {
      source.left, source.left, source.right, source.right};
  const std::int64_t ys[4] = {
      source.bottom, source.top, source.bottom, source.top};
  std::int64_t output_x = 0;
  std::int64_t output_y = 0;
  if (!transform_point_checked(
          context, xs[0], ys[0], &output_x, &output_y)) {
    return false;
  }
  destination->left = output_x;
  destination->right = output_x;
  destination->bottom = output_y;
  destination->top = output_y;
  for (int corner = 1; corner < 4; ++corner) {
    if (!transform_point_checked(
            context, xs[corner], ys[corner], &output_x, &output_y)) {
      return false;
    }
    destination->left = min(destination->left, output_x);
    destination->right = max(destination->right, output_x);
    destination->bottom = min(destination->bottom, output_y);
    destination->top = max(destination->top, output_y);
  }
  destination->source_token = source.source_polygon;
  destination->context_token = context_token;
  return destination->left < destination->right &&
         destination->bottom < destination->top;
}

__global__ void m2_expand_rectangles_kernel(
    const ContextGpu *contexts, const std::uint32_t *m2_contexts,
    const std::uint64_t *rectangle_offsets, const M2CensusCell *cells,
    const M2CensusRectTemplate *templates,
    std::uint64_t context_count, M2CensusRect *rectangles,
    M2CensusGpuCounters *counters, std::uint32_t *status) {
  for (std::uint64_t list_id = blockIdx.x; list_id < context_count;
       list_id += gridDim.x) {
    const std::uint32_t context_id = m2_contexts[list_id];
    const ContextGpu context = contexts[context_id];
    const M2CensusCell cell = cells[context.cell];
    for (std::uint32_t local = threadIdx.x;
         local < cell.rectangle_count; local += blockDim.x) {
      M2CensusRect rectangle{};
      if (!m2_transform_rectangle(
              context, templates[cell.rectangle_begin + local],
              &rectangle, context_id)) {
        atomicOr(status,
                 static_cast<std::uint32_t>(
                     kM2CensusTransformOverflow |
                     kM2CensusInvalidRecord));
        continue;
      }
      rectangles[rectangle_offsets[list_id] + local] = rectangle;
      atomicAdd(&counters->expanded, 1ull);
    }
  }
}

__global__ void m2_emit_x_endpoints_kernel(
    const M2CensusRect *rectangles, std::uint64_t rectangle_count,
    std::int64_t *endpoints, std::uint32_t *status) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < rectangle_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const M2CensusRect rectangle = rectangles[index];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top) {
      atomicOr(status,
               static_cast<std::uint32_t>(kM2CensusInvalidRecord));
      continue;
    }
    endpoints[index * 2] = rectangle.left;
    endpoints[index * 2 + 1] = rectangle.right;
  }
}

__device__ std::uint64_t m2_lower_bound_device(
    const std::int64_t *values, std::uint64_t count,
    std::int64_t target) {
  std::uint64_t first = 0;
  while (count) {
    const std::uint64_t step = count / 2;
    const std::uint64_t middle = first + step;
    if (values[middle] < target) {
      first = middle + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

__global__ void m2_count_slab_memberships_kernel(
    const M2CensusRect *rectangles, std::uint64_t rectangle_count,
    const std::int64_t *unique_x, std::uint64_t unique_x_count,
    std::uint64_t *membership_counts,
    M2CensusGpuCounters *counters, std::uint32_t *status) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < rectangle_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const M2CensusRect rectangle = rectangles[index];
    const std::uint64_t first =
        m2_lower_bound_device(unique_x, unique_x_count, rectangle.left);
    const std::uint64_t last =
        m2_lower_bound_device(unique_x, unique_x_count, rectangle.right);
    if (first >= unique_x_count || last >= unique_x_count ||
        unique_x[first] != rectangle.left ||
        unique_x[last] != rectangle.right || first >= last) {
      atomicOr(status,
               static_cast<std::uint32_t>(kM2CensusEndpointLookup));
      membership_counts[index] = 0;
      continue;
    }
    const std::uint64_t span = last - first;
    membership_counts[index] = span;
    if (span > 4096) {
      atomicAdd(&counters->span_over_4096, 1ull);
    }
    if (span > 65535) {
      atomicAdd(&counters->span_over_65535, 1ull);
    }
  }
}

std::uint32_t m2_launch_blocks(std::uint64_t count) {
  if (!count) {
    return 0;
  }
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
          (count + kCensusThreads - 1) / kCensusThreads, 65535));
}

int m2_run_census(const M2CensusOptions &options,
                  Clock::time_point total_begin) {
  M2CensusTiming timing;
  auto begin = Clock::now();
  LoadedScene scene = load_and_validate(options.path);
  timing.load_validate_ms = milliseconds(begin, Clock::now());
  if (hex_digest(scene.header.scene_sha256, 32) !=
      options.expected_scene_sha256) {
    throw SceneError("scene SHA-256 does not match explicit expectation");
  }
  if (scene.header.well_layer != kQualifiedM2Layer ||
      scene.header.well_datatype != kQualifiedM2Datatype) {
    throw SceneError("logical slot 0 is not qualified M2 layer 101/0");
  }

  begin = Clock::now();
  LoweredScene hierarchy =
      lower_hierarchy(scene, options.max_contexts);
  timing.hierarchy_lower_ms = milliseconds(begin, Clock::now());

  begin = Clock::now();
  M2CensusCompactScene compact =
      m2_lower_scene(scene, hierarchy, options.max_rectangles);
  timing.template_lower_ms = milliseconds(begin, Clock::now());
  if (options.expected_flat_polygons &&
      compact.flat_polygons != options.expected_flat_polygons) {
    throw SceneError("flat M2 polygon census disagrees with expectation");
  }
  if (options.expected_flat_rectangles &&
      compact.flat_rectangles != options.expected_flat_rectangles) {
    throw SceneError("flat M2 rectangle census disagrees with expectation");
  }
  std::uint64_t endpoint_count = 0;
  if (!checked_mul_u64(compact.flat_rectangles, 2, &endpoint_count) ||
      endpoint_count > options.max_endpoints) {
    throw SceneError("global x endpoints exceed configured capacity");
  }

  begin = Clock::now();
  cuda_require(cudaFree(nullptr), "CUDA context initialization");
  cuda_require(cudaDeviceSynchronize(),
               "CUDA context initialization synchronize");
  timing.cuda_init_ms = milliseconds(begin, Clock::now());

  begin = Clock::now();
  DeviceBuffer<ContextGpu> d_contexts(compact.contexts.size());
  DeviceBuffer<std::uint32_t> d_m2_contexts(
      compact.m2_contexts.size());
  DeviceBuffer<std::uint64_t> d_rectangle_offsets(
      compact.rectangle_offsets.size());
  DeviceBuffer<M2CensusCell> d_cells(compact.cells.size());
  DeviceBuffer<M2CensusRectTemplate> d_templates(
      compact.rectangles.size());
  DeviceBuffer<M2CensusRect> d_rectangles(compact.flat_rectangles);
  DeviceBuffer<std::int64_t> d_endpoints(endpoint_count);
  DeviceBuffer<std::uint64_t> d_membership_counts(
      compact.flat_rectangles);
  DeviceBuffer<M2CensusGpuCounters> d_counters(1);
  DeviceBuffer<std::uint32_t> d_status(1);
  upload(&d_contexts, compact.contexts);
  upload(&d_m2_contexts, compact.m2_contexts);
  upload(&d_rectangle_offsets, compact.rectangle_offsets);
  upload(&d_cells, compact.cells);
  upload(&d_templates, compact.rectangles);
  cuda_require(cudaMemset(d_counters.get(), 0,
                          sizeof(M2CensusGpuCounters)),
               "clear M2 census counters");
  cuda_require(cudaMemset(d_status.get(), 0, sizeof(std::uint32_t)),
               "clear M2 census status");
  cuda_require(cudaDeviceSynchronize(), "M2 census upload synchronize");
  timing.upload_ms = milliseconds(begin, Clock::now());

  begin = Clock::now();
  const std::uint32_t context_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(compact.m2_contexts.size(), 65535));
  m2_expand_rectangles_kernel<<<context_blocks, 128>>>(
      d_contexts.get(), d_m2_contexts.get(), d_rectangle_offsets.get(),
      d_cells.get(), d_templates.get(), compact.m2_contexts.size(),
      d_rectangles.get(), d_counters.get(), d_status.get());
  cuda_require(cudaGetLastError(), "expand production M2 rectangles");
  cuda_require(cudaDeviceSynchronize(),
               "expand production M2 rectangles synchronize");
  timing.expand_ms = milliseconds(begin, Clock::now());

  std::uint32_t host_status = 0;
  M2CensusGpuCounters counters{};
  cuda_require(cudaMemcpy(&host_status, d_status.get(), sizeof(host_status),
                          cudaMemcpyDeviceToHost),
               "M2 expansion status D2H");
  cuda_require(cudaMemcpy(&counters, d_counters.get(), sizeof(counters),
                          cudaMemcpyDeviceToHost),
               "M2 expansion counters D2H");
  if (host_status || counters.expanded != compact.flat_rectangles) {
    throw SceneError(
        "device M2 expansion failed exact count/status conservation");
  }

  if (options.verify_expanded_host) {
    begin = Clock::now();
    std::vector<M2CensusRect> host_rectangles(compact.flat_rectangles);
    cuda_require(
        cudaMemcpy(host_rectangles.data(), d_rectangles.get(),
                   host_rectangles.size() * sizeof(M2CensusRect),
                   cudaMemcpyDeviceToHost),
        "expanded production M2 rectangles D2H");
    std::uint64_t compared = 0;
    for (std::size_t list_id = 0;
         list_id < compact.m2_contexts.size(); ++list_id) {
      const std::uint32_t context_id = compact.m2_contexts[list_id];
      const ContextGpu &context = compact.contexts[context_id];
      const M2CensusCell &cell = compact.cells[context.cell];
      if (compact.rectangle_offsets[list_id] != compared) {
        throw SceneError("host M2 oracle found a noncanonical offset");
      }
      for (std::uint32_t local = 0; local < cell.rectangle_count;
           ++local) {
        const M2CensusRect expected = m2_transform_rectangle_host(
            context, compact.rectangles[cell.rectangle_begin + local],
            context_id);
        if (!m2_same_rectangle(expected, host_rectangles[compared])) {
          std::ostringstream message;
          message << "GPU M2 rectangle differs from host oracle at "
                  << compared;
          throw SceneError(message.str());
        }
        ++compared;
      }
    }
    if (compared != compact.flat_rectangles) {
      throw SceneError("host M2 rectangle oracle count mismatch");
    }
    timing.host_verify_ms = milliseconds(begin, Clock::now());
  }

  begin = Clock::now();
  const std::uint32_t rectangle_blocks =
      m2_launch_blocks(compact.flat_rectangles);
  m2_emit_x_endpoints_kernel<<<rectangle_blocks, kCensusThreads>>>(
      d_rectangles.get(), compact.flat_rectangles, d_endpoints.get(),
      d_status.get());
  cuda_require(cudaGetLastError(), "emit production M2 x endpoints");
  thrust::device_ptr<std::int64_t> x_begin(d_endpoints.get());
  thrust::sort(thrust::device, x_begin, x_begin + endpoint_count);
  const auto x_end =
      thrust::unique(thrust::device, x_begin, x_begin + endpoint_count);
  const std::uint64_t unique_x_count = x_end - x_begin;
  cuda_require(cudaDeviceSynchronize(),
               "production M2 x sort/unique synchronize");
  timing.x_sort_unique_ms = milliseconds(begin, Clock::now());
  if (unique_x_count < 2) {
    throw SceneError("production M2 has fewer than two unique x endpoints");
  }

  begin = Clock::now();
  m2_count_slab_memberships_kernel<<<rectangle_blocks, kCensusThreads>>>(
      d_rectangles.get(), compact.flat_rectangles, d_endpoints.get(),
      unique_x_count, d_membership_counts.get(), d_counters.get(),
      d_status.get());
  cuda_require(cudaGetLastError(),
               "count production M2 x-slab memberships");
  thrust::device_ptr<std::uint64_t> count_begin(
      d_membership_counts.get());
  const std::uint64_t membership_count = thrust::reduce(
      thrust::device, count_begin,
      count_begin + compact.flat_rectangles, std::uint64_t{0});
  const std::uint64_t max_memberships_per_rectangle =
      *thrust::max_element(
          thrust::device, count_begin,
          count_begin + compact.flat_rectangles);
  cuda_require(cudaDeviceSynchronize(),
               "production M2 membership synchronize");
  timing.membership_ms = milliseconds(begin, Clock::now());
  cuda_require(cudaMemcpy(&host_status, d_status.get(), sizeof(host_status),
                          cudaMemcpyDeviceToHost),
               "M2 membership status D2H");
  cuda_require(cudaMemcpy(&counters, d_counters.get(), sizeof(counters),
                          cudaMemcpyDeviceToHost),
               "M2 membership counters D2H");
  if (host_status) {
    throw SceneError(
        "device M2 x-slab census failed exact endpoint lookup");
  }

  begin = Clock::now();
  cudaError_t cleanup_status = cudaSuccess;
  const auto release = [&](auto *buffer) {
    const cudaError_t status = buffer->release();
    if (cleanup_status == cudaSuccess && status != cudaSuccess) {
      cleanup_status = status;
    }
  };
  release(&d_status);
  release(&d_counters);
  release(&d_membership_counts);
  release(&d_endpoints);
  release(&d_rectangles);
  release(&d_templates);
  release(&d_cells);
  release(&d_rectangle_offsets);
  release(&d_m2_contexts);
  release(&d_contexts);
  if (cleanup_status != cudaSuccess) {
    throw SceneError(std::string("CUDA cleanup: ") +
                     cudaGetErrorString(cleanup_status));
  }
  timing.cleanup_ms = milliseconds(begin, Clock::now());
  timing.total_ms = milliseconds(total_begin, Clock::now());

  std::cout << "M2_MANHATTAN_PRODUCTION_CENSUS"
            << " verdict=COMPLETE"
            << " scene_sha256="
            << hex_digest(scene.header.scene_sha256, 32)
            << " contexts=" << compact.contexts.size()
            << " m2_contexts=" << compact.m2_contexts.size()
            << " local_polygons=" << compact.local_polygons
            << " local_l_shapes=" << compact.local_l_shapes
            << " local_rectangles=" << compact.rectangles.size()
            << " flat_polygons=" << compact.flat_polygons
            << " flat_l_shapes=" << compact.flat_l_shapes
            << " flat_rectangles=" << compact.flat_rectangles
            << " unique_x=" << unique_x_count
            << " x_slabs=" << unique_x_count - 1
            << " slab_memberships=" << membership_count
            << " max_slabs_per_rectangle="
            << max_memberships_per_rectangle
            << " rectangles_over_4096_slabs="
            << counters.span_over_4096
            << " rectangles_over_65535_slabs="
            << counters.span_over_65535
            << " device_flags=" << host_status << "\n";
  std::cout << std::fixed << std::setprecision(3)
            << "TIMING_MS"
            << " load_validate=" << timing.load_validate_ms
            << " hierarchy_lower=" << timing.hierarchy_lower_ms
            << " template_lower=" << timing.template_lower_ms
            << " cuda_init=" << timing.cuda_init_ms
            << " upload=" << timing.upload_ms
            << " expand=" << timing.expand_ms
            << " host_verify=" << timing.host_verify_ms
            << " x_sort_unique=" << timing.x_sort_unique_ms
            << " membership=" << timing.membership_ms
            << " cleanup=" << timing.cleanup_ms
            << " gpu_census="
            << timing.upload_ms + timing.expand_ms +
                   timing.x_sort_unique_ms + timing.membership_ms
            << " warm_total=" << timing.total_ms - timing.cuda_init_ms
            << " cold_total=" << timing.total_ms << "\n";
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  const Clock::time_point total_begin = Clock::now();
  try {
    return m2_run_census(m2_parse_options(argc, argv), total_begin);
  } catch (const std::exception &error) {
    std::cerr << "M2_MANHATTAN_PRODUCTION_CENSUS"
              << " verdict=UNCERTAIN error=\"" << error.what() << "\"\n";
    return 2;
  }
}
