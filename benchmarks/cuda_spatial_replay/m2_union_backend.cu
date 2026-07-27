/*
 * Exact raw-hierarchy M2 Manhattan-union production adapter.
 *
 * The C ABI deliberately carries the compact host hierarchy rather than a
 * multi-gigabyte flattened rectangle stream.  This adapter independently
 * validates that hierarchy and its KM2RAW01 digest, decomposes only exact
 * simple hole-free Manhattan contours into disjoint rectangles, uploads
 * compact templates, expands all eight orthogonal transforms on the GPU, and
 * transfers ownership of the resulting device rectangles directly into the
 * shared exact union core.
 */

#include "dbCudaActive3Digest.h"
#include "dbCudaSpatialApi.h"
#include "contact4_union_resident.cuh"
#include "cuda_device_phase_lease.h"
#include "m1_width_space_exact_predicate.h"
#include "m2_manhattan_decompose.h"
#include "m2_resident_morphology_gpu.cuh"
#include "manhattan_union_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace mu = klayout_cuda::manhattan_union;
namespace md = klayout_cuda::m2_manhattan_decompose;
namespace m2m = klayout_cuda::m2_resident_morphology;
namespace c4 = klayout_cuda::contact4_union_resident;
namespace a3 = klayout_cuda::active3;

using Context =
    klayout_cuda_spatial_m1_width_space_context_v1;
using Cell =
    klayout_cuda_spatial_m1_width_space_cell_v1;
using Polygon =
    klayout_cuda_spatial_m1_width_space_polygon_v1;
using Edge =
    klayout_cuda_spatial_m1_width_space_edge_v1;
using Request =
    klayout_cuda_spatial_m2_union_request_v1;
using Result =
    klayout_cuda_spatial_m2_union_result_v1;
using Segment =
    klayout_cuda_spatial_m2_union_segment_v1;
using Contact4Scene =
    klayout_cuda_spatial_contact4_active_union_scene_v1;
using Contact4SceneEcho =
    klayout_cuda_spatial_contact4_active_union_scene_echo_v1;
using Contact4Request =
    klayout_cuda_spatial_contact4_active_union_request_v1;
using Contact4Result =
    klayout_cuda_spatial_contact4_active_union_result_v1;
using Active3WellScene =
    klayout_cuda_spatial_active3_well_union_scene_v1;
using Active3WellSceneEcho =
    klayout_cuda_spatial_active3_well_union_scene_echo_v1;
using Active3WellRequest =
    klayout_cuda_spatial_active3_well_union_request_v1;
using Active3WellResult =
    klayout_cuda_spatial_active3_well_union_result_v1;
using Implant15Scene =
    klayout_cuda_spatial_implant15_scene_v1;
using Implant15SceneEcho =
    klayout_cuda_spatial_implant15_scene_echo_v1;
using Implant15Capacity =
    klayout_cuda_spatial_implant15_capacity_v1;
using Implant15Request =
    klayout_cuda_spatial_implant15_request_v1;
using Implant15Result =
    klayout_cuda_spatial_implant15_result_v1;
using M1MorphRequest =
    klayout_cuda_spatial_m1_resident_morphology_request_v1;
using M1MorphResult =
    klayout_cuda_spatial_m1_resident_morphology_result_v1;
using Clock = std::chrono::steady_clock;

static_assert(std::is_trivially_copyable<Context>::value,
              "M2 contexts must remain POD");
static_assert(std::is_trivially_copyable<Cell>::value,
              "M2 cells must remain POD");
static_assert(std::is_trivially_copyable<Polygon>::value,
              "M2 polygons must remain POD");
static_assert(std::is_trivially_copyable<Edge>::value,
              "M2 edges must remain POD");
static_assert(sizeof(Context) == 24, "unexpected M2 context ABI padding");
static_assert(sizeof(Cell) == 32, "unexpected M2 cell ABI padding");
static_assert(sizeof(Polygon) == 48, "unexpected M2 polygon ABI padding");
static_assert(sizeof(Edge) == 32, "unexpected M2 edge ABI padding");
static_assert(sizeof(Request) == 336, "unexpected M2 request ABI padding");
static_assert(sizeof(Result) == 480, "unexpected M2 result ABI padding");
static_assert(sizeof(Segment) == 32, "unexpected M2 segment ABI padding");
static_assert(sizeof(Contact4Scene) == 280,
              "unexpected CONTACT4 scene ABI padding");
static_assert(sizeof(Contact4SceneEcho) == 192,
              "unexpected CONTACT4 scene echo ABI padding");
static_assert(sizeof(Contact4Request) == 760,
              "unexpected CONTACT4 request ABI padding");
static_assert(sizeof(Contact4Result) == 944,
              "unexpected CONTACT4 result ABI padding");
static_assert(sizeof(Active3WellRequest) == 776,
              "unexpected ACTIVE3 WELL-union request ABI padding");
static_assert(sizeof(Active3WellResult) == 960,
              "unexpected ACTIVE3 WELL-union result ABI padding");
static_assert(sizeof(M1MorphRequest) == 408,
              "unexpected M1 resident-morphology request ABI padding");
static_assert(sizeof(M1MorphResult) == 744,
              "unexpected M1 resident-morphology result ABI padding");
static_assert(std::is_trivially_copyable<Implant15Scene>::value,
              "IMPLANT.1-.5 scenes must remain POD");
static_assert(std::is_trivially_copyable<Implant15Capacity>::value,
              "IMPLANT.1-.5 capacities must remain POD");
static_assert(std::is_trivially_copyable<Implant15Request>::value,
              "IMPLANT.1-.5 requests must remain POD");
static_assert(std::is_trivially_copyable<Implant15Result>::value,
              "IMPLANT.1-.5 results must remain POD");
static_assert(sizeof(a3::DirectedEdge) == sizeof(Edge),
              "CONTACT4 and raw-scene edge layouts diverged");
static_assert(sizeof(mu::DirectedSegmentI64) == sizeof(Segment),
              "shared union and C ABI segment sizes diverged");
static_assert(
    static_cast<std::uint32_t>(mu::SegmentAxis::horizontal) ==
        KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL &&
        static_cast<std::uint32_t>(mu::SegmentAxis::vertical) ==
            KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL,
    "shared union and C ABI segment axes diverged");

constexpr std::int64_t kCoordinateLimit = INT64_C(1000000000000);
constexpr std::uint32_t kExpandThreads = 256;
constexpr std::uint32_t kMaximumBlocks = 65535;
constexpr std::uint64_t kM1MaxStitchedStripIntervals =
    UINT64_C(128000000);
constexpr std::uint64_t kM1BaseMaxStitchedStripIntervals =
    UINT64_C(160000000);
constexpr char kM2RawDigestMagic[8] =
    {'K', 'M', '2', 'R', 'A', 'W', '0', '1'};
constexpr char kM1RawDigestMagic[8] =
    {'K', 'M', '1', 'R', 'A', 'W', '0', '1'};
constexpr char kActiveRawDigestMagic[8] =
    {'K', 'A', 'R', 'A', 'W', '0', '0', '1'};
constexpr char kContactRawDigestMagic[8] =
    {'K', 'C', 'R', 'A', 'W', '0', '0', '1'};
constexpr char kWellUnionRawDigestMagic[8] =
    {'K', 'W', 'R', 'W', 'L', '0', '0', '1'};
constexpr char kNplusRawDigestMagic[8] =
    {'K', 'N', 'P', 'L', 'S', '0', '0', '1'};
constexpr char kPplusRawDigestMagic[8] =
    {'K', 'P', 'P', 'L', 'S', '0', '0', '1'};
constexpr char kGateRawDigestMagic[8] =
    {'K', 'G', 'A', 'T', 'E', '0', '0', '1'};
constexpr std::array<std::uint8_t, 32>
    kQualifiedCompactM2SceneDigest = {
        0x66, 0xec, 0x73, 0xea, 0xf6, 0x86, 0xc6, 0xf6,
        0x30, 0xeb, 0x91, 0xe6, 0x39, 0x49, 0xb7, 0x03,
        0x01, 0xca, 0x2f, 0x24, 0x72, 0x6e, 0xe4, 0x03,
        0x3d, 0xd7, 0x7a, 0x5f, 0x89, 0x08, 0xc1, 0xc3};
constexpr std::array<std::uint8_t, 32>
    kQualifiedLiveM2SceneDigest = {
        0x08, 0x8c, 0xf4, 0x59, 0xf9, 0xea, 0xa6, 0x21,
        0xcd, 0x06, 0xfd, 0x71, 0xce, 0x0f, 0x50, 0x2d,
        0x8a, 0x18, 0xce, 0x08, 0x72, 0x7a, 0x1d, 0xf7,
        0x46, 0x35, 0x57, 0x73, 0xe3, 0x09, 0x96, 0x27};
constexpr std::array<std::uint8_t, 32>
    kQualifiedLiveM1MorphSceneDigest = {
        0x02, 0x74, 0x2f, 0x05, 0x42, 0x8d, 0x84, 0x76,
        0x9f, 0x0b, 0xcd, 0xd2, 0xeb, 0x28, 0x78, 0xdf,
        0x65, 0x30, 0xe6, 0xab, 0xbc, 0x04, 0xf3, 0x44,
        0x58, 0x75, 0xce, 0x81, 0xb0, 0x71, 0xc1, 0x18};
constexpr std::array<std::uint8_t, 32>
    kQualifiedFocusedM1MorphSceneDigest = {
        0xb8, 0x1e, 0xb8, 0x97, 0x44, 0x85, 0x54, 0x4c,
        0xf4, 0xc9, 0xea, 0xe0, 0x5b, 0x9b, 0x44, 0xd5,
        0x79, 0x0b, 0xa6, 0xdb, 0xfc, 0xa0, 0xa9, 0x24,
        0x50, 0x4b, 0xad, 0xac, 0xc9, 0x09, 0x59, 0xea};

enum ExpandFlag : std::uint32_t {
  kExpandTransformOverflow = 1u << 0,
  kExpandBoundsMismatch = 1u << 1,
  kExpandInvalidRecord = 1u << 2,
  kExpandTransposeInvariant = 1u << 3,
};

enum class DeclineKind {
  bad_argument,
  capacity,
  coordinate,
};

class M2Decline : public std::runtime_error
{
public:
  M2Decline(DeclineKind kind, std::uint32_t fallback_flags,
            const std::string &message)
      : std::runtime_error(message), m_kind(kind),
        m_fallback_flags(fallback_flags)
  {
  }

  DeclineKind kind() const { return m_kind; }
  std::uint32_t fallback_flags() const { return m_fallback_flags; }

private:
  DeclineKind m_kind;
  std::uint32_t m_fallback_flags;
};

struct Point
{
  std::int64_t x;
  std::int64_t y;
};

struct RectangleTemplate
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint64_t source_token;
};

struct LoweredCell
{
  std::uint64_t rectangle_begin;
  std::uint32_t rectangle_count;
  std::uint32_t polygon_count;
  std::uint32_t l_shape_count;
  std::uint32_t reserved;
};

static_assert(sizeof(RectangleTemplate) == 40,
              "unexpected M2 rectangle-template padding");
static_assert(sizeof(LoweredCell) == 24,
              "unexpected lowered M2 cell padding");

struct CellCensus
{
  std::uint64_t rectangle_begin = 0;
  std::uint32_t rectangle_count = 0;
  std::uint32_t polygon_count = 0;
  std::uint32_t l_shape_count = 0;
};

struct LoweredScene
{
  std::vector<LoweredCell> cells;
  std::vector<RectangleTemplate> rectangles;
  std::vector<std::uint64_t> rectangle_offsets;
  std::uint64_t flat_rectangles = 0;
};

template <class T>
class DeviceBuffer
{
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::uint64_t count) { allocate(count); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : m_data(other.m_data), m_count(other.m_count)
  {
    other.m_data = nullptr;
    other.m_count = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
  {
    if (this != &other) {
      reset();
      m_data = other.m_data;
      m_count = other.m_count;
      other.m_data = nullptr;
      other.m_count = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { reset(); }

  void allocate(std::uint64_t count)
  {
    reset();
    if (!count) return;
    if (count >
        std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::runtime_error("CUDA buffer byte count overflows size_t");
    }
    cuda_require(
        cudaMalloc(reinterpret_cast<void **>(&m_data),
                   static_cast<std::size_t>(count) * sizeof(T)),
        "cudaMalloc compact M2 buffer");
    m_count = count;
  }

  void reset() noexcept
  {
    if (m_data) {
      (void)cudaFree(m_data);
      m_data = nullptr;
      m_count = 0;
    }
  }

  T *get() { return m_data; }
  const T *get() const { return m_data; }
  std::uint64_t size() const { return m_count; }

private:
  static void cuda_require(cudaError_t error, const char *operation)
  {
    if (error != cudaSuccess) {
      throw std::runtime_error(
          std::string(operation) + ": " + cudaGetErrorString(error));
    }
  }

  T *m_data = nullptr;
  std::uint64_t m_count = 0;
};

struct CanonicalDigest
{
  void bytes(const void *data, std::size_t size) { sha.update(data, size); }

  void u32(std::uint32_t value)
  {
    std::uint8_t encoded[4];
    for (unsigned int byte = 0; byte < 4; ++byte) {
      encoded[byte] =
          static_cast<std::uint8_t>(value >> (byte * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void u64(std::uint64_t value)
  {
    std::uint8_t encoded[8];
    for (unsigned int byte = 0; byte < 8; ++byte) {
      encoded[byte] =
          static_cast<std::uint8_t>(value >> (byte * 8));
    }
    bytes(encoded, sizeof(encoded));
  }

  void i64(std::int64_t value)
  {
    u64(static_cast<std::uint64_t>(value));
  }

  std::array<std::uint8_t, 32> finish() { return sha.finish(); }

  db::cuda_active3_digest::Sha256 sha;
};

std::mutex &pipeline_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::mutex &allocation_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unordered_set<const Segment *> &owned_allocations()
{
  static std::unordered_set<const Segment *> allocations;
  return allocations;
}

struct AllocationCounters
{
  std::uint64_t allocations = 0;
  std::uint64_t release_calls = 0;
  std::uint64_t owned_releases = 0;
};

AllocationCounters &allocation_counters()
{
  static AllocationCounters counters;
  return counters;
}

std::atomic<std::uint32_t> &m2_test_fault()
{
  static std::atomic<std::uint32_t> fault{0};
  return fault;
}

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end)
{
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          end - begin)
          .count());
}

std::uint64_t milliseconds_to_ns(double milliseconds)
{
  if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
    throw std::runtime_error("shared M2 union returned invalid timing");
  }
  const long double nanoseconds =
      static_cast<long double>(milliseconds) * 1000000.0L;
  if (nanoseconds >
      static_cast<long double>(
          std::numeric_limits<std::uint64_t>::max())) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(nanoseconds + 0.5L);
}

void cuda_require(cudaError_t error, const char *operation)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(error));
  }
}

void set_message(Result *result, const char *message)
{
  std::snprintf(
      result->message, sizeof(result->message), "%s",
      message ? message : "");
}

void clear_suffix_certificate(Result *result) noexcept
{
  result->certified_empty_mask = 0;
  result->certificate_reserved = 0;
  result->suffix_total_ns = 0;
}

void set_message(Contact4Result *result, const char *message)
{
  std::snprintf(
      result->message, sizeof(result->message), "%s",
      message ? message : "");
}

void set_message(Active3WellResult *result, const char *message)
{
  std::snprintf(
      result->message, sizeof(result->message), "%s",
      message ? message : "");
}

void set_message(Implant15Result *result, const char *message)
{
  std::snprintf(
      result->message, sizeof(result->message), "%s",
      message ? message : "");
}

void set_message(M1MorphResult *result, const char *message)
{
  std::snprintf(
      result->message, sizeof(result->message), "%s",
      message ? message : "");
}

bool coordinate_qualified(std::int64_t value)
{
  return value >= -kCoordinateLimit && value <= kCoordinateLimit;
}

bool checked_add_u64(std::uint64_t first, std::uint64_t second,
                     std::uint64_t *result)
{
  if (second >
      std::numeric_limits<std::uint64_t>::max() - first) {
    return false;
  }
  *result = first + second;
  return true;
}

bool checked_multiply_u64(std::uint64_t first, std::uint64_t second,
                          std::uint64_t *result)
{
  if (first && second >
                   std::numeric_limits<std::uint64_t>::max() / first) {
    return false;
  }
  *result = first * second;
  return true;
}

bool array_size_fits(std::uint64_t count, std::uint64_t record_size)
{
  std::uint64_t bytes = 0;
  return record_size &&
         checked_multiply_u64(count, record_size, &bytes) &&
         bytes <= std::numeric_limits<std::size_t>::max();
}

template <class T>
T load_record(const void *records, std::uint64_t index,
              std::uint32_t record_bytes)
{
  T result;
  const std::uint8_t *source =
      static_cast<const std::uint8_t *>(records) +
      static_cast<std::size_t>(index) * record_bytes;
  std::memcpy(&result, source, sizeof(result));
  return result;
}

template <class T>
T load_scalar(const T *records, std::uint64_t index)
{
  T result;
  std::memcpy(&result, records + static_cast<std::size_t>(index),
              sizeof(result));
  return result;
}

[[noreturn]] void malformed(const std::string &message,
                            std::uint32_t flags =
                                KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST)
{
  throw M2Decline(DeclineKind::bad_argument, flags, message);
}

[[noreturn]] void capacity(const std::string &message)
{
  throw M2Decline(
      DeclineKind::capacity,
      KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY,
      message);
}

[[noreturn]] void coordinate_decline(const std::string &message)
{
  throw M2Decline(
      DeclineKind::coordinate,
      KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW,
      message);
}

Request m1_morph_as_union_request(const M1MorphRequest &request)
{
  Request adapted{};
  adapted.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  adapted.struct_size = sizeof(adapted);
  adapted.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY;
  adapted.option_flags =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  adapted.format_version = request.format_version;
  adapted.dbu_per_micron = request.dbu_per_micron;
  adapted.root_cell = request.root_cell;
  adapted.device = request.device;
  adapted.contexts = request.contexts;
  adapted.context_count = request.context_count;
  adapted.context_record_bytes = request.context_record_bytes;
  adapted.context_reserved = request.context_reserved;
  adapted.metal_contexts = request.metal_contexts;
  adapted.metal_context_count = request.metal_context_count;
  adapted.context_polygon_offsets =
      request.context_polygon_offsets;
  adapted.context_polygon_offset_count =
      request.context_polygon_offset_count;
  adapted.context_edge_offsets = request.context_edge_offsets;
  adapted.context_edge_offset_count =
      request.context_edge_offset_count;
  adapted.cells = request.cells;
  adapted.cell_count = request.cell_count;
  adapted.cell_record_bytes = request.cell_record_bytes;
  adapted.cell_reserved = request.cell_reserved;
  adapted.polygons = request.polygons;
  adapted.polygon_count = request.polygon_count;
  adapted.polygon_record_bytes = request.polygon_record_bytes;
  adapted.polygon_reserved = request.polygon_reserved;
  adapted.edges = request.edges;
  adapted.edge_count = request.edge_count;
  adapted.edge_record_bytes = request.edge_record_bytes;
  adapted.edge_reserved = request.edge_reserved;
  adapted.flat_polygon_count = request.flat_polygon_count;
  adapted.flat_edge_count = request.flat_edge_count;
  adapted.scene_left = request.scene_left;
  adapted.scene_bottom = request.scene_bottom;
  adapted.scene_right = request.scene_right;
  adapted.scene_top = request.scene_top;
  adapted.max_contexts = request.max_contexts;
  adapted.max_rectangles = request.max_rectangles;
  adapted.max_x_slabs = request.max_x_slabs;
  adapted.max_memberships = request.max_union_memberships;
  adapted.max_events = request.max_union_events;
  adapted.max_raw_segments = request.max_union_raw_segments;
  adapted.max_segments = request.max_union_segments;
  adapted.max_slabs_per_rectangle =
      request.max_slabs_per_rectangle;
  std::copy(
      request.scene_digest, request.scene_digest + 32,
      adapted.scene_digest);
  return adapted;
}

bool valid_basic_request(const Request &request)
{
  return
      request.abi_version == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
      request.struct_size == sizeof(request) &&
      (request.opcode ==
           KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY ||
       request.opcode ==
           KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY) &&
      request.option_flags ==
          KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS &&
      request.format_version == 1 &&
      request.dbu_per_micron == 2000 && request.device >= 0 &&
      request.context_reserved == 0 &&
      request.cell_reserved == 0 &&
      request.polygon_reserved == 0 &&
      request.edge_reserved == 0 && request.reserved0 == 0 &&
      request.reserved1[0] == 0 && request.reserved1[1] == 0 &&
      request.context_count && request.contexts &&
      request.context_record_bytes == sizeof(Context) &&
      request.metal_context_count && request.metal_contexts &&
      request.context_polygon_offset_count ==
          request.metal_context_count &&
      request.context_polygon_offsets &&
      request.context_edge_offset_count ==
          request.metal_context_count &&
      request.context_edge_offsets && request.cell_count &&
      request.cells && request.cell_record_bytes == sizeof(Cell) &&
      request.polygon_count && request.polygons &&
      request.polygon_record_bytes == sizeof(Polygon) &&
      request.edge_count && request.edges &&
      request.edge_record_bytes == sizeof(Edge) &&
      request.root_cell < request.cell_count &&
      request.flat_polygon_count && request.flat_edge_count &&
      request.scene_left < request.scene_right &&
      request.scene_bottom < request.scene_top &&
      request.max_contexts && request.max_rectangles &&
      request.max_x_slabs && request.max_memberships &&
      request.max_events && request.max_raw_segments &&
      request.max_segments && request.max_slabs_per_rectangle &&
      request.context_count <= request.max_contexts &&
      request.cell_count <= request.context_count &&
      request.context_count <=
          std::numeric_limits<std::uint32_t>::max() &&
      request.metal_context_count <=
          std::numeric_limits<std::uint32_t>::max() &&
      request.cell_count <=
          std::numeric_limits<std::uint32_t>::max() &&
      request.polygon_count <=
          std::numeric_limits<std::uint32_t>::max() &&
      request.edge_count <=
          std::numeric_limits<std::uint32_t>::max() &&
      request.flat_polygon_count <=
          std::numeric_limits<std::uint32_t>::max() &&
      request.flat_edge_count <=
          std::numeric_limits<std::uint32_t>::max() &&
      request.polygon_count <= request.edge_count / 4 &&
      request.flat_polygon_count <= request.flat_edge_count / 4 &&
      request.flat_polygon_count <= request.max_rectangles &&
      request.edge_count <= request.max_memberships &&
      request.max_x_slabs <=
          std::numeric_limits<std::uint32_t>::max() &&
      array_size_fits(
          request.context_count, request.context_record_bytes) &&
      array_size_fits(
          request.metal_context_count, sizeof(std::uint32_t)) &&
      array_size_fits(
          request.context_polygon_offset_count, sizeof(std::uint64_t)) &&
      array_size_fits(
          request.context_edge_offset_count, sizeof(std::uint64_t)) &&
      array_size_fits(
          request.cell_count, request.cell_record_bytes) &&
      array_size_fits(
          request.polygon_count, request.polygon_record_bytes) &&
      array_size_fits(
          request.edge_count, request.edge_record_bytes) &&
      array_size_fits(request.max_segments, sizeof(Segment)) &&
      request.max_segments <= static_cast<std::uint64_t>(
          std::numeric_limits<std::ptrdiff_t>::max()) &&
      coordinate_qualified(request.scene_left) &&
      coordinate_qualified(request.scene_bottom) &&
      coordinate_qualified(request.scene_right) &&
      coordinate_qualified(request.scene_top);
}

bool valid_m1_morph_request(const M1MorphRequest &request)
{
  const bool suffix_request =
      request.opcode ==
          KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M15_9_EMPTY &&
      request.requested_mask ==
          KLAYOUT_CUDA_SPATIAL_M1_MORPH_ALL_EMPTY;
  const bool base_request =
      request.opcode ==
          KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M11_2_EMPTY &&
      request.requested_mask ==
          KLAYOUT_CUDA_SPATIAL_M1_BASE_ALL_EMPTY;
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof(request) ||
      (!suffix_request && !base_request) ||
      request.option_flags !=
          KLAYOUT_CUDA_SPATIAL_M1_MORPH_QUALIFIED_OPTIONS ||
      request.format_version != 1 ||
      request.dbu_per_micron != 2000 ||
      request.device < 0 ||
      request.reserved0 ||
      request.context_reserved || request.cell_reserved ||
      request.polygon_reserved || request.edge_reserved ||
      request.union_reserved || request.morph_reserved ||
      request.reserved1[0] || request.reserved1[1] ||
      !request.max_morph_output_slabs ||
      !request.max_morph_output_intervals ||
      !request.max_morph_raw_boundary_segments ||
      !request.max_morph_boundary_segments ||
      !request.max_morph_source_visits_per_pass ||
      !request.max_morph_source_visits_per_band ||
      !request.max_morph_long_segments ||
      !request.max_morph_active_slabs ||
      request.max_morph_boundary_segments >
          request.max_morph_raw_boundary_segments) {
    return false;
  }

  const m2m::Limits qualified_limits;
  if (request.max_morph_output_slabs >
          qualified_limits.max_output_slabs ||
      request.max_morph_output_intervals >
          qualified_limits.max_output_intervals ||
      request.max_morph_raw_boundary_segments >
          qualified_limits.max_raw_boundary_segments ||
      request.max_morph_boundary_segments >
          qualified_limits.max_boundary_segments ||
      request.max_morph_source_visits_per_band >
          qualified_limits.max_source_visits_per_band ||
      request.max_morph_long_segments >
          qualified_limits.max_long_segments ||
      request.max_morph_active_slabs >
          qualified_limits.max_active_slabs ||
      (request.max_morph_source_visits_per_pass >
           m2m::kUniversalSourceVisitCap &&
       request.max_morph_source_visits_per_pass !=
           m2m::kQualifiedM1ProductionSourceVisitCap)) {
    return false;
  }

  Request structural = m1_morph_as_union_request(request);
  // A bounded cap below the supplied census is a valid transaction which
  // must decline as FALLBACK/CAPACITY rather than BAD_ARGUMENT.  Raise only
  // the cap-dependent fields for the pointer/record-shape precheck.
  structural.max_rectangles = std::max(
      structural.max_rectangles, structural.flat_polygon_count);
  structural.max_memberships = std::max(
      structural.max_memberships, structural.edge_count);
  return valid_basic_request(structural);
}

bool qualified_production_m2_suffix_scene(const Request &request)
{
  const bool compact_scene =
      std::equal(
          request.scene_digest, request.scene_digest + 32,
          kQualifiedCompactM2SceneDigest.begin()) &&
      request.context_count == UINT64_C(587201) &&
      request.cell_count == UINT64_C(143);
  const bool live_scene =
      std::equal(
          request.scene_digest, request.scene_digest + 32,
          kQualifiedLiveM2SceneDigest.begin()) &&
      request.context_count == UINT64_C(849265) &&
      request.cell_count == UINT64_C(273);
  return (compact_scene || live_scene) &&
      request.metal_context_count == UINT64_C(568632) &&
      request.polygon_count == UINT64_C(45960) &&
      request.edge_count == UINT64_C(183852) &&
      request.flat_polygon_count == UINT64_C(22945976) &&
      request.flat_edge_count == UINT64_C(91784840) &&
      request.scene_left == INT64_C(6230) &&
      request.scene_bottom == INT64_C(6225) &&
      request.scene_right == INT64_C(1788415) &&
      request.scene_top == INT64_C(1487300);
}

bool qualified_production_m1_morph_scene(
    const M1MorphRequest &request)
{
  const bool full_deck_scene =
      std::equal(
          request.scene_digest, request.scene_digest + 32,
          kQualifiedLiveM1MorphSceneDigest.begin()) &&
      request.context_count == UINT64_C(849265);
  const bool focused_scene =
      std::equal(
          request.scene_digest, request.scene_digest + 32,
          kQualifiedFocusedM1MorphSceneDigest.begin()) &&
      request.context_count == UINT64_C(848497);
  return (full_deck_scene || focused_scene) &&
      request.metal_context_count == UINT64_C(830950) &&
      request.cell_count == UINT64_C(273) &&
      request.polygon_count == UINT64_C(1099662) &&
      request.edge_count == UINT64_C(4398792) &&
      request.flat_polygon_count == UINT64_C(41093878) &&
      request.flat_edge_count == UINT64_C(164386520) &&
      request.scene_left == INT64_C(6165) &&
      request.scene_bottom == INT64_C(6230) &&
      request.scene_right == INT64_C(1788420) &&
      request.scene_top == INT64_C(1486245);
}

Request scene_as_union_request(
    const Contact4Scene &scene, const Contact4Request &request)
{
  Request adapted{};
  adapted.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  adapted.struct_size = sizeof(adapted);
  adapted.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY;
  adapted.option_flags =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  adapted.format_version = scene.format_version;
  adapted.dbu_per_micron = scene.dbu_per_micron;
  adapted.root_cell = scene.root_cell;
  adapted.device = request.device;
  adapted.contexts = scene.contexts;
  adapted.context_count = scene.context_count;
  adapted.context_record_bytes = scene.context_record_bytes;
  adapted.metal_contexts = scene.layer_contexts;
  adapted.metal_context_count = scene.layer_context_count;
  adapted.context_polygon_offsets =
      scene.context_polygon_offsets;
  adapted.context_polygon_offset_count =
      scene.context_polygon_offset_count;
  adapted.context_edge_offsets = scene.context_edge_offsets;
  adapted.context_edge_offset_count =
      scene.context_edge_offset_count;
  adapted.cells = scene.cells;
  adapted.cell_count = scene.cell_count;
  adapted.cell_record_bytes = scene.cell_record_bytes;
  adapted.polygons = scene.polygons;
  adapted.polygon_count = scene.polygon_count;
  adapted.polygon_record_bytes = scene.polygon_record_bytes;
  adapted.edges = scene.edges;
  adapted.edge_count = scene.edge_count;
  adapted.edge_record_bytes = scene.edge_record_bytes;
  adapted.flat_polygon_count = scene.flat_polygon_count;
  adapted.flat_edge_count = scene.flat_edge_count;
  adapted.scene_left = scene.scene_left;
  adapted.scene_bottom = scene.scene_bottom;
  adapted.scene_right = scene.scene_right;
  adapted.scene_top = scene.scene_top;
  adapted.max_contexts = request.max_contexts;
  adapted.max_rectangles = request.max_rectangles;
  adapted.max_x_slabs = request.max_x_slabs;
  adapted.max_memberships = request.max_union_memberships;
  adapted.max_events = request.max_events;
  adapted.max_raw_segments = request.max_raw_segments;
  adapted.max_segments = request.max_boundary_segments;
  adapted.max_slabs_per_rectangle =
      request.max_slabs_per_rectangle;
  std::copy(
      scene.scene_digest, scene.scene_digest + 32,
      adapted.scene_digest);
  return adapted;
}

Request active3_scene_as_union_request(
    const Active3WellScene &scene,
    const Active3WellRequest &request)
{
  Request adapted{};
  adapted.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  adapted.struct_size = sizeof(adapted);
  adapted.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY;
  adapted.option_flags =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  adapted.format_version = scene.format_version;
  adapted.dbu_per_micron = scene.dbu_per_micron;
  adapted.root_cell = scene.root_cell;
  adapted.device = request.device;
  adapted.contexts = scene.contexts;
  adapted.context_count = scene.context_count;
  adapted.context_record_bytes = scene.context_record_bytes;
  adapted.metal_contexts = scene.layer_contexts;
  adapted.metal_context_count = scene.layer_context_count;
  adapted.context_polygon_offsets =
      scene.context_polygon_offsets;
  adapted.context_polygon_offset_count =
      scene.context_polygon_offset_count;
  adapted.context_edge_offsets = scene.context_edge_offsets;
  adapted.context_edge_offset_count =
      scene.context_edge_offset_count;
  adapted.cells = scene.cells;
  adapted.cell_count = scene.cell_count;
  adapted.cell_record_bytes = scene.cell_record_bytes;
  adapted.polygons = scene.polygons;
  adapted.polygon_count = scene.polygon_count;
  adapted.polygon_record_bytes = scene.polygon_record_bytes;
  adapted.edges = scene.edges;
  adapted.edge_count = scene.edge_count;
  adapted.edge_record_bytes = scene.edge_record_bytes;
  adapted.flat_polygon_count = scene.flat_polygon_count;
  adapted.flat_edge_count = scene.flat_edge_count;
  adapted.scene_left = scene.scene_left;
  adapted.scene_bottom = scene.scene_bottom;
  adapted.scene_right = scene.scene_right;
  adapted.scene_top = scene.scene_top;
  adapted.max_contexts = request.max_contexts;
  adapted.max_rectangles = request.max_rectangles;
  adapted.max_x_slabs = request.max_x_slabs;
  adapted.max_memberships = request.max_union_memberships;
  adapted.max_events = request.max_events;
  adapted.max_raw_segments = request.max_raw_segments;
  adapted.max_segments = request.max_boundary_segments;
  adapted.max_slabs_per_rectangle =
      request.max_slabs_per_rectangle;
  std::copy(
      scene.scene_digest, scene.scene_digest + 32,
      adapted.scene_digest);
  return adapted;
}

Request implant15_scene_as_union_request(
    const Implant15Scene &scene,
    const Implant15Request &request)
{
  Request adapted{};
  adapted.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  adapted.struct_size = sizeof(adapted);
  adapted.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY;
  adapted.option_flags =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  adapted.format_version = scene.format_version;
  adapted.dbu_per_micron = scene.dbu_per_micron;
  adapted.root_cell = scene.root_cell;
  adapted.device = request.device;
  adapted.contexts = scene.contexts;
  adapted.context_count = scene.context_count;
  adapted.context_record_bytes = scene.context_record_bytes;
  adapted.metal_contexts = scene.layer_contexts;
  adapted.metal_context_count = scene.layer_context_count;
  adapted.context_polygon_offsets =
      scene.context_polygon_offsets;
  adapted.context_polygon_offset_count =
      scene.context_polygon_offset_count;
  adapted.context_edge_offsets = scene.context_edge_offsets;
  adapted.context_edge_offset_count =
      scene.context_edge_offset_count;
  adapted.cells = scene.cells;
  adapted.cell_count = scene.cell_count;
  adapted.cell_record_bytes = scene.cell_record_bytes;
  adapted.polygons = scene.polygons;
  adapted.polygon_count = scene.polygon_count;
  adapted.polygon_record_bytes = scene.polygon_record_bytes;
  adapted.edges = scene.edges;
  adapted.edge_count = scene.edge_count;
  adapted.edge_record_bytes = scene.edge_record_bytes;
  adapted.flat_polygon_count = scene.flat_polygon_count;
  adapted.flat_edge_count = scene.flat_edge_count;
  adapted.scene_left = scene.scene_left;
  adapted.scene_bottom = scene.scene_bottom;
  adapted.scene_right = scene.scene_right;
  adapted.scene_top = scene.scene_top;
  adapted.max_contexts = request.capacity.max_contexts;
  adapted.max_rectangles = request.capacity.max_rectangles;
  adapted.max_x_slabs = request.capacity.max_x_slabs;
  adapted.max_memberships =
      request.capacity.max_union_memberships;
  adapted.max_events = request.capacity.max_events;
  adapted.max_raw_segments =
      request.capacity.max_raw_segments;
  adapted.max_segments =
      request.capacity.max_boundary_segments;
  adapted.max_slabs_per_rectangle =
      request.capacity.max_slabs_per_rectangle;
  std::copy(
      scene.scene_digest, scene.scene_digest + 32,
      adapted.scene_digest);
  return adapted;
}

bool exact_bytes(
    const std::uint8_t *bytes, const char (&expected)[8])
{
  return std::equal(bytes, bytes + 8, expected);
}

bool valid_implant15_scene_descriptor(
    const Implant15Scene &scene, std::uint32_t role,
    std::uint32_t layer, std::uint32_t datatype,
    const char (&digest_magic)[8],
    const Implant15Request &request)
{
  if (scene.struct_size != sizeof(scene) ||
      scene.role != role || scene.format_version != 1 ||
      scene.dbu_per_micron != 2000 || scene.layer != layer ||
      scene.datatype != datatype || scene.reserved0 ||
      scene.context_reserved || scene.cell_reserved ||
      scene.polygon_reserved || scene.edge_reserved ||
      scene.reserved1[0] || scene.reserved1[1] ||
      !exact_bytes(scene.digest_domain, digest_magic)) {
    return false;
  }
  Request structural =
      implant15_scene_as_union_request(scene, request);
  structural.max_rectangles = std::max(
      structural.max_rectangles, structural.flat_polygon_count);
  structural.max_memberships = std::max(
      structural.max_memberships, structural.edge_count);
  return valid_basic_request(structural);
}

bool valid_contact4_scene_descriptor(
    const Contact4Scene &scene, std::uint32_t role,
    std::uint32_t layer, const char (&digest_magic)[8],
    const Contact4Request &request)
{
  if (scene.struct_size != sizeof(scene) ||
      scene.role != role || scene.format_version != 1 ||
      scene.dbu_per_micron != 2000 || scene.layer != layer ||
      scene.datatype != 0 || scene.reserved0 ||
      scene.context_reserved || scene.cell_reserved ||
      scene.polygon_reserved || scene.edge_reserved ||
      scene.reserved1[0] || scene.reserved1[1] ||
      !exact_bytes(scene.digest_domain, digest_magic)) {
    return false;
  }
  Request structural = scene_as_union_request(scene, request);
  // Operational caps are allowed to be smaller than the input census: that
  // is a valid bounded request which must return FALLBACK, not BAD_ARGUMENT.
  // Raise only the two cap-dependent fields for this pointer/record-shape
  // precheck; validate_and_lower receives the original limits and emits the
  // checked capacity decline before any large allocation.
  structural.max_rectangles = std::max(
      structural.max_rectangles, structural.flat_polygon_count);
  structural.max_memberships = std::max(
      structural.max_memberships, structural.edge_count);
  return valid_basic_request(structural);
}

bool valid_active3_well_scene_descriptor(
    const Active3WellScene &scene, std::uint32_t role,
    std::uint32_t layer, const char (&digest_magic)[8],
    const Active3WellRequest &request)
{
  if (scene.struct_size != sizeof(scene) ||
      scene.role != role || scene.format_version != 1 ||
      scene.dbu_per_micron != 2000 || scene.layer != layer ||
      scene.datatype != 0 || scene.reserved0 ||
      scene.context_reserved || scene.cell_reserved ||
      scene.polygon_reserved || scene.edge_reserved ||
      scene.reserved1[0] || scene.reserved1[1] ||
      !exact_bytes(scene.digest_domain, digest_magic)) {
    return false;
  }
  Request structural =
      active3_scene_as_union_request(scene, request);
  structural.max_rectangles = std::max(
      structural.max_rectangles, structural.flat_polygon_count);
  structural.max_memberships = std::max(
      structural.max_memberships, structural.edge_count);
  return valid_basic_request(structural);
}

bool valid_contact4_request(const Contact4Request &request)
{
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof(request) ||
      request.opcode !=
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_EMPTY ||
      request.option_flags !=
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_QUALIFIED_OPTIONS ||
      request.format_version != 1 ||
      request.dbu_per_micron != 2000 || request.device < 0 ||
      request.reserved0 || request.distance != 10 ||
      request.grid_cell_size != 2000 ||
      !request.max_contexts || !request.max_rectangles ||
      !request.max_x_slabs || !request.max_union_memberships ||
      !request.max_events || !request.max_raw_segments ||
      !request.max_boundary_segments ||
      !request.max_slabs_per_rectangle ||
      request.union_reserved || !request.max_contact_edges ||
      !request.max_grid_cells ||
      !request.max_contact_memberships ||
      !request.max_boundary_cell_visits ||
      !request.max_member_visits || !request.max_pair_work ||
      !request.max_cells_per_contact_edge ||
      !request.max_cells_per_boundary_edge ||
      request.reserved1[0] || request.reserved1[1] ||
      request.reserved1[2] || request.reserved1[3]) {
    return false;
  }
  if (!valid_contact4_scene_descriptor(
          request.active,
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE, 1,
          kActiveRawDigestMagic, request) ||
      !valid_contact4_scene_descriptor(
          request.contact,
          KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
          kContactRawDigestMagic, request) ||
      request.active.format_version != request.format_version ||
      request.contact.format_version != request.format_version ||
      request.active.dbu_per_micron != request.dbu_per_micron ||
      request.contact.dbu_per_micron != request.dbu_per_micron ||
      request.active.root_cell != request.contact.root_cell ||
      request.active.context_count != request.contact.context_count ||
      request.active.cell_count != request.contact.cell_count ||
      request.active.context_count > request.max_contexts ||
      request.contact.context_count > request.max_contexts ||
      request.contact.flat_edge_count >
          request.max_contact_edges ||
      request.contact.flat_edge_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  return true;
}

bool valid_active3_well_request(
    const Active3WellRequest &request)
{
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof(request) ||
      request.opcode !=
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_EMPTY ||
      request.option_flags !=
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_QUALIFIED_OPTIONS ||
      request.format_version != 1 ||
      request.dbu_per_micron != 2000 || request.device < 0 ||
      request.reserved0 || request.distance != 110 ||
      request.grid_cell_size != 2000 ||
      request.secondary_well_layer != 2 ||
      request.secondary_well_datatype != 0 ||
      request.layer_reserved ||
      !request.max_contexts || !request.max_rectangles ||
      !request.max_x_slabs || !request.max_union_memberships ||
      !request.max_events || !request.max_raw_segments ||
      !request.max_boundary_segments ||
      !request.max_slabs_per_rectangle ||
      request.union_reserved || !request.max_active_edges ||
      !request.max_grid_cells ||
      !request.max_active_memberships ||
      !request.max_active_cell_visits ||
      !request.max_member_visits || !request.max_pair_work ||
      !request.max_cells_per_active_edge ||
      !request.max_cells_per_well_edge ||
      request.reserved1[0] || request.reserved1[1] ||
      request.reserved1[2] || request.reserved1[3]) {
    return false;
  }
  if (!valid_active3_well_scene_descriptor(
          request.wells,
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_WELLS_ROLE, 3,
          kWellUnionRawDigestMagic, request) ||
      !valid_active3_well_scene_descriptor(
          request.active,
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_ACTIVE_ROLE, 1,
          kActiveRawDigestMagic, request) ||
      request.wells.format_version != request.format_version ||
      request.active.format_version != request.format_version ||
      request.wells.dbu_per_micron != request.dbu_per_micron ||
      request.active.dbu_per_micron != request.dbu_per_micron ||
      request.wells.root_cell != request.active.root_cell ||
      request.wells.context_count != request.active.context_count ||
      request.wells.cell_count != request.active.cell_count ||
      request.wells.context_count > request.max_contexts ||
      request.active.context_count > request.max_contexts ||
      request.active.flat_edge_count >
          request.max_active_edges ||
      request.active.flat_edge_count >
          std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  return true;
}

bool valid_implant15_request(const Implant15Request &request)
{
  const Implant15Capacity &capacity = request.capacity;
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof(request) ||
      request.opcode !=
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_RESIDENT_EMPTY ||
      request.option_flags !=
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_QUALIFIED_OPTIONS ||
      request.format_version != 1 ||
      request.dbu_per_micron != 2000 ||
      request.requested_mask !=
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES ||
      request.device < 0 || request.reserved0 ||
      request.reserved1 ||
      request.implant1_distance != 140 ||
      request.implant2_distance != 50 ||
      request.implant3_distance != 90 ||
      request.implant4_distance != 90 ||
      request.grid_cell_size != 2000 ||
      request.reserved2[0] || request.reserved2[1] ||
      request.reserved2[2] || request.reserved2[3] ||
      capacity.struct_size != sizeof(capacity) ||
      !capacity.max_slabs_per_rectangle ||
      !capacity.max_cells_per_secondary_edge ||
      !capacity.max_cells_per_boundary_edge ||
      !capacity.max_contexts || !capacity.max_rectangles ||
      !capacity.max_x_slabs ||
      !capacity.max_union_memberships ||
      !capacity.max_events || !capacity.max_raw_segments ||
      !capacity.max_boundary_segments ||
      !capacity.max_gate_edges || !capacity.max_contact_edges ||
      !capacity.max_grid_cells ||
      !capacity.max_secondary_memberships ||
      !capacity.max_gate_boundary_cell_visits ||
      !capacity.max_contact_boundary_cell_visits ||
      !capacity.max_member_visits ||
      !capacity.max_pair_work ||
      !capacity.max_morphology_work ||
      !capacity.max_overlap_work ||
      capacity.reserved[0] || capacity.reserved[1] ||
      capacity.reserved[2] || capacity.reserved[3]) {
    return false;
  }
  if (!valid_implant15_scene_descriptor(
          request.nplus,
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_NPLUS_ROLE,
          4, 0, kNplusRawDigestMagic, request) ||
      !valid_implant15_scene_descriptor(
          request.pplus,
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_PPLUS_ROLE,
          5, 0, kPplusRawDigestMagic, request) ||
      !valid_implant15_scene_descriptor(
          request.gate,
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_GATE_ROLE,
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_LAYER,
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_DERIVED_DATATYPE,
          kGateRawDigestMagic, request) ||
      !valid_implant15_scene_descriptor(
          request.contact,
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_CONTACT_ROLE,
          10, 0, kContactRawDigestMagic, request)) {
    return false;
  }
  const Implant15Scene *const scenes[] = {
      &request.nplus, &request.pplus,
      &request.gate, &request.contact};
  for (const Implant15Scene *scene : scenes) {
    if (scene->format_version != request.format_version ||
        scene->dbu_per_micron != request.dbu_per_micron ||
        scene->context_count > capacity.max_contexts) {
      return false;
    }
  }
  if (request.gate.flat_edge_count > capacity.max_gate_edges ||
      request.contact.flat_edge_count >
          capacity.max_contact_edges ||
      request.gate.flat_edge_count > UINT32_MAX ||
      request.contact.flat_edge_count > UINT32_MAX) {
    return false;
  }
  return true;
}

void echo_request(const Request &request, Result *result)
{
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->format_version = request.format_version;
  result->dbu_per_micron = request.dbu_per_micron;
  result->root_cell = request.root_cell;
  result->segment_record_bytes = sizeof(Segment);
  std::copy(
      request.scene_digest, request.scene_digest + 32,
      result->scene_digest);
  result->context_count = request.context_count;
  result->metal_context_count = request.metal_context_count;
  result->cell_count = request.cell_count;
  result->polygon_count = request.polygon_count;
  result->edge_count = request.edge_count;
  result->flat_polygon_count = request.flat_polygon_count;
  result->flat_edge_count = request.flat_edge_count;
}

void echo_m1_morph_request(
    const M1MorphRequest &request, M1MorphResult *result)
{
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->format_version = request.format_version;
  result->dbu_per_micron = request.dbu_per_micron;
  result->root_cell = request.root_cell;
  result->requested_mask = request.requested_mask;
  result->scene_left = request.scene_left;
  result->scene_bottom = request.scene_bottom;
  result->scene_right = request.scene_right;
  result->scene_top = request.scene_top;
  std::copy(
      request.scene_digest, request.scene_digest + 32,
      result->scene_digest);
  result->context_count = request.context_count;
  result->metal_context_count = request.metal_context_count;
  result->cell_count = request.cell_count;
  result->polygon_count = request.polygon_count;
  result->edge_count = request.edge_count;
  result->flat_polygon_count = request.flat_polygon_count;
  result->flat_edge_count = request.flat_edge_count;
  result->max_contexts = request.max_contexts;
  result->max_rectangles = request.max_rectangles;
  result->max_x_slabs = request.max_x_slabs;
  result->max_union_memberships =
      request.max_union_memberships;
  result->max_union_events = request.max_union_events;
  result->max_union_raw_segments =
      request.max_union_raw_segments;
  result->max_union_segments = request.max_union_segments;
  result->max_slabs_per_rectangle =
      request.max_slabs_per_rectangle;
  result->max_morph_output_slabs =
      request.max_morph_output_slabs;
  result->max_morph_output_intervals =
      request.max_morph_output_intervals;
  result->max_morph_raw_boundary_segments =
      request.max_morph_raw_boundary_segments;
  result->max_morph_boundary_segments =
      request.max_morph_boundary_segments;
  result->max_morph_source_visits_per_pass =
      request.max_morph_source_visits_per_pass;
  result->max_morph_source_visits_per_band =
      request.max_morph_source_visits_per_band;
  result->max_morph_long_segments =
      request.max_morph_long_segments;
  result->max_morph_active_slabs =
      request.max_morph_active_slabs;
}

std::array<std::uint8_t, 32>
request_digest(const Request &request, const char (&magic)[8])
{
  CanonicalDigest digest;
  digest.bytes(magic, sizeof(magic));
  digest.u32(request.format_version);
  digest.u32(request.dbu_per_micron);
  digest.u32(request.root_cell);
  digest.u32(0);
  digest.u64(request.context_count);
  digest.u64(request.metal_context_count);
  digest.u64(request.cell_count);
  digest.u64(request.polygon_count);
  digest.u64(request.edge_count);
  digest.u64(request.flat_polygon_count);
  digest.u64(request.flat_edge_count);
  digest.i64(request.scene_left);
  digest.i64(request.scene_bottom);
  digest.i64(request.scene_right);
  digest.i64(request.scene_top);

  for (std::uint64_t index = 0;
       index < request.context_count; ++index) {
    const Context context = load_record<Context>(
        request.contexts, index, request.context_record_bytes);
    digest.i64(context.tx);
    digest.i64(context.ty);
    digest.u32(context.cell_id);
    digest.u32(context.transform_code);
  }
  for (std::uint64_t index = 0;
       index < request.metal_context_count; ++index) {
    digest.u32(load_scalar(request.metal_contexts, index));
    digest.u64(load_scalar(request.context_polygon_offsets, index));
    digest.u64(load_scalar(request.context_edge_offsets, index));
  }
  for (std::uint64_t index = 0; index < request.cell_count; ++index) {
    const Cell cell = load_record<Cell>(
        request.cells, index, request.cell_record_bytes);
    digest.u64(cell.source_cell_index);
    digest.u64(cell.polygon_begin);
    digest.u64(cell.edge_begin);
    digest.u32(cell.polygon_count);
    digest.u32(cell.edge_count);
  }
  for (std::uint64_t index = 0;
       index < request.polygon_count; ++index) {
    const Polygon polygon = load_record<Polygon>(
        request.polygons, index, request.polygon_record_bytes);
    digest.u64(polygon.edge_begin);
    digest.i64(polygon.left);
    digest.i64(polygon.bottom);
    digest.i64(polygon.right);
    digest.i64(polygon.top);
    digest.u32(polygon.polygon_id);
    digest.u32(polygon.edge_count);
  }
  for (std::uint64_t index = 0; index < request.edge_count; ++index) {
    const Edge edge = load_record<Edge>(
        request.edges, index, request.edge_record_bytes);
    digest.i64(edge.x1);
    digest.i64(edge.y1);
    digest.i64(edge.x2);
    digest.i64(edge.y2);
  }
  return digest.finish();
}

void validate_shared_contact4_hierarchy(
    const Request &active, const Request &contact)
{
  if (active.root_cell != contact.root_cell ||
      active.context_count != contact.context_count ||
      active.cell_count != contact.cell_count) {
    malformed(
        "ACTIVE and CONTACT do not share one hierarchy identity");
  }
  for (std::uint64_t index = 0;
       index < active.context_count; ++index) {
    const Context first = load_record<Context>(
        active.contexts, index, active.context_record_bytes);
    const Context second = load_record<Context>(
        contact.contexts, index, contact.context_record_bytes);
    if (first.tx != second.tx || first.ty != second.ty ||
        first.cell_id != second.cell_id ||
        first.transform_code != second.transform_code) {
      malformed(
          "ACTIVE and CONTACT context hierarchies differ");
    }
  }
  for (std::uint64_t index = 0;
       index < active.cell_count; ++index) {
    const Cell first = load_record<Cell>(
        active.cells, index, active.cell_record_bytes);
    const Cell second = load_record<Cell>(
        contact.cells, index, contact.cell_record_bytes);
    if (first.source_cell_index != second.source_cell_index) {
      malformed(
          "ACTIVE and CONTACT source-cell identities differ");
    }
  }
}

void validate_shared_active3_hierarchy(
    const Request &wells, const Request &active)
{
  if (wells.root_cell != active.root_cell ||
      wells.context_count != active.context_count ||
      wells.cell_count != active.cell_count) {
    malformed(
        "combined WELL and ACTIVE do not share one hierarchy identity");
  }
  for (std::uint64_t index = 0;
       index < wells.context_count; ++index) {
    const Context first = load_record<Context>(
        wells.contexts, index, wells.context_record_bytes);
    const Context second = load_record<Context>(
        active.contexts, index, active.context_record_bytes);
    if (first.tx != second.tx || first.ty != second.ty ||
        first.cell_id != second.cell_id ||
        first.transform_code != second.transform_code) {
      malformed(
          "combined WELL and ACTIVE context hierarchies differ");
    }
  }
  for (std::uint64_t index = 0;
       index < wells.cell_count; ++index) {
    const Cell first = load_record<Cell>(
        wells.cells, index, wells.cell_record_bytes);
    const Cell second = load_record<Cell>(
        active.cells, index, active.cell_record_bytes);
    if (first.source_cell_index != second.source_cell_index) {
      malformed(
          "combined WELL and ACTIVE source-cell identities differ");
    }
  }
}

__int128 twice_area(const std::array<Point, 6> &points,
                    std::uint32_t count)
{
  __int128 area = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    const Point first = points[index];
    const Point second = points[(index + 1) % count];
    area += static_cast<__int128>(first.x) * second.y -
            static_cast<__int128>(second.x) * first.y;
  }
  return area;
}

__int128 abs_i128(__int128 value)
{
  return value < 0 ? -value : value;
}

bool intervals_overlap_inclusive(std::int64_t first_lo,
                                 std::int64_t first_hi,
                                 std::int64_t second_lo,
                                 std::int64_t second_hi)
{
  if (first_lo > first_hi) std::swap(first_lo, first_hi);
  if (second_lo > second_hi) std::swap(second_lo, second_hi);
  return std::max(first_lo, second_lo) <=
         std::min(first_hi, second_hi);
}

bool edges_intersect(const Edge &first, const Edge &second)
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
  const Edge &vertical = first_vertical ? first : second;
  const Edge &horizontal = first_vertical ? second : first;
  return intervals_overlap_inclusive(
             vertical.y1, vertical.y2,
             horizontal.y1, horizontal.y1) &&
         intervals_overlap_inclusive(
             horizontal.x1, horizontal.x2,
             vertical.x1, vertical.x1);
}

bool point_inside_cell(const std::array<Point, 6> &points,
                       std::uint32_t count, std::int64_t left,
                       std::int64_t bottom, std::int64_t right,
                       std::int64_t top)
{
  const __int128 x2 =
      static_cast<__int128>(left) + right;
  const __int128 y2 =
      static_cast<__int128>(bottom) + top;
  bool inside = false;
  for (std::uint32_t index = 0; index < count; ++index) {
    const Point first = points[index];
    const Point second = points[(index + 1) % count];
    if (first.x != second.x) continue;
    const __int128 first_y2 = static_cast<__int128>(first.y) * 2;
    const __int128 second_y2 = static_cast<__int128>(second.y) * 2;
    if ((first_y2 > y2) != (second_y2 > y2) &&
        static_cast<__int128>(first.x) * 2 > x2) {
      inside = !inside;
    }
  }
  return inside;
}

std::uint32_t decompose_polygon(
    const Request &request, const Polygon &polygon,
    std::uint64_t global_polygon,
    RectangleTemplate *destination,
    std::uint64_t destination_capacity)
{
  if (polygon.edge_count < 4 || (polygon.edge_count & 1u)) {
    malformed(
        "raw M2 polygon does not have an even edge count of at least four");
  }
  if (polygon.left >= polygon.right ||
      polygon.bottom >= polygon.top ||
      !coordinate_qualified(polygon.left) ||
      !coordinate_qualified(polygon.bottom) ||
      !coordinate_qualified(polygon.right) ||
      !coordinate_qualified(polygon.top)) {
    malformed("raw M2 polygon has an invalid bounding box");
  }
  std::uint64_t polygon_end = 0;
  if (!checked_add_u64(
          polygon.edge_begin, polygon.edge_count, &polygon_end) ||
      polygon_end > request.edge_count) {
    malformed(
        "raw M2 polygon edge range escapes the scene",
        KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN);
  }

  /*
   * Preserve the allocation-free common box/L path.  Larger contours are
   * rare stored templates, so their independent exact slab decomposition can
   * afford dynamic scratch without imposing it on every production box.
   */
  if (polygon.edge_count > 6) {
    std::vector<md::EdgeI64> contour;
    contour.reserve(polygon.edge_count);
    for (std::uint32_t local = 0;
         local < polygon.edge_count; ++local) {
      const Edge edge = load_record<Edge>(
          request.edges, polygon.edge_begin + local,
          request.edge_record_bytes);
      if (!coordinate_qualified(edge.x1) ||
          !coordinate_qualified(edge.y1) ||
          !coordinate_qualified(edge.x2) ||
          !coordinate_qualified(edge.y2)) {
        malformed(
            "raw M2 arbitrary contour exceeds the coordinate domain");
      }
      contour.push_back(
          md::EdgeI64{edge.x1, edge.y1, edge.x2, edge.y2});
    }
    md::Result decomposition = md::decompose(
        contour, polygon.left, polygon.bottom,
        polygon.right, polygon.top, global_polygon,
        request.max_rectangles);
    if (decomposition.status == md::Status::capacity) {
      capacity(
          std::string("raw M2 arbitrary contour: ") +
          decomposition.message);
    }
    if (decomposition.status != md::Status::complete) {
      malformed(
          std::string("raw M2 arbitrary contour: ") +
          decomposition.message);
    }
    if (decomposition.rectangles.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      capacity(
          "raw M2 arbitrary contour rectangle count exceeds capacity");
    }
    if (destination &&
        decomposition.rectangles.size() > destination_capacity) {
      malformed(
          "raw M2 arbitrary second-pass decomposition diverged");
    }
    for (std::size_t index = 0;
         destination && index < decomposition.rectangles.size();
         ++index) {
      const md::RectangleI64 &source =
          decomposition.rectangles[index];
      destination[index] = RectangleTemplate{
          source.left, source.bottom, source.right, source.top,
          source.source_token};
    }
    return static_cast<std::uint32_t>(
        decomposition.rectangles.size());
  }
  if (polygon.edge_count != 4 && polygon.edge_count != 6) {
    malformed("raw M2 box/L contour has an invalid edge count");
  }

  std::array<Point, 6> points{};
  std::array<Edge, 6> edges{};
  std::array<std::int64_t, 6> xs{};
  std::array<std::int64_t, 6> ys{};
  std::int64_t derived_left = 0;
  std::int64_t derived_bottom = 0;
  std::int64_t derived_right = 0;
  std::int64_t derived_top = 0;
  for (std::uint32_t local = 0;
       local < polygon.edge_count; ++local) {
    const Edge edge = load_record<Edge>(
        request.edges, polygon.edge_begin + local,
        request.edge_record_bytes);
    edges[local] = edge;
    if (!coordinate_qualified(edge.x1) ||
        !coordinate_qualified(edge.y1) ||
        !coordinate_qualified(edge.x2) ||
        !coordinate_qualified(edge.y2) ||
        (edge.x1 == edge.x2 && edge.y1 == edge.y2) ||
        !(edge.x1 == edge.x2 || edge.y1 == edge.y2)) {
      malformed("raw M2 contains an invalid Manhattan edge");
    }
    points[local] = {edge.x1, edge.y1};
    xs[local] = edge.x1;
    ys[local] = edge.y1;
    if (!local) {
      derived_left = derived_right = edge.x1;
      derived_bottom = derived_top = edge.y1;
    } else {
      derived_left = std::min(derived_left, edge.x1);
      derived_bottom = std::min(derived_bottom, edge.y1);
      derived_right = std::max(derived_right, edge.x1);
      derived_top = std::max(derived_top, edge.y1);
    }
  }
  for (std::uint32_t local = 0;
       local < polygon.edge_count; ++local) {
    const Edge &edge = edges[local];
    const Edge &following =
        edges[(local + 1) % polygon.edge_count];
    if (edge.x2 != following.x1 ||
        edge.y2 != following.y1 ||
        (edge.x1 == edge.x2) ==
            (following.x1 == following.x2)) {
      malformed(
          "raw M2 contour is open or has a redundant turn");
    }
    for (std::uint32_t other = local + 1;
         other < polygon.edge_count; ++other) {
      const bool adjacent =
          other == local + 1 ||
          (local == 0 && other + 1 == polygon.edge_count);
      if (!adjacent && edges_intersect(edge, edges[other])) {
        malformed("raw M2 contour self-intersects");
      }
    }
  }
  if (derived_left != polygon.left ||
      derived_bottom != polygon.bottom ||
      derived_right != polygon.right ||
      derived_top != polygon.top) {
    malformed("raw M2 polygon bounding-box echo is inconsistent");
  }

  const auto sort_prefix = [count = polygon.edge_count](
                               std::array<std::int64_t, 6> *values) {
    for (std::uint32_t index = 1; index < count; ++index) {
      const std::int64_t value = (*values)[index];
      std::uint32_t insertion = index;
      while (insertion && value < (*values)[insertion - 1]) {
        (*values)[insertion] = (*values)[insertion - 1];
        --insertion;
      }
      (*values)[insertion] = value;
    }
  };
  sort_prefix(&xs);
  sort_prefix(&ys);
  const auto x_end = std::unique(
      xs.begin(), xs.begin() + polygon.edge_count);
  const auto y_end = std::unique(
      ys.begin(), ys.begin() + polygon.edge_count);
  const std::size_t x_count =
      static_cast<std::size_t>(x_end - xs.begin());
  const std::size_t y_count =
      static_cast<std::size_t>(y_end - ys.begin());
  const __int128 polygon_area2 =
      twice_area(points, polygon.edge_count);
  if (polygon_area2 >= 0) {
    malformed("raw M2 polygon is not clockwise");
  }

  std::array<RectangleTemplate, 2> rectangles{};
  std::uint32_t rectangle_count = 0;
  if (polygon.edge_count == 4) {
    if (x_count != 2 || y_count != 2 ||
        xs[0] != polygon.left || ys[0] != polygon.bottom ||
        xs[1] != polygon.right || ys[1] != polygon.top) {
      malformed("four-edge raw M2 polygon is not an exact box");
    }
    rectangles[0] = {
        xs[0], ys[0], xs[1], ys[1], global_polygon};
    rectangle_count = 1;
  } else {
    if (x_count != 3 || y_count != 3 ||
        xs[0] != polygon.left || ys[0] != polygon.bottom ||
        xs[2] != polygon.right || ys[2] != polygon.top) {
      malformed("six-edge raw M2 polygon is not an exact L");
    }
    bool occupied[2][2] = {};
    std::uint32_t occupied_count = 0;
    for (std::uint32_t x = 0; x < 2; ++x) {
      for (std::uint32_t y = 0; y < 2; ++y) {
        occupied[x][y] = point_inside_cell(
            points, polygon.edge_count, xs[x], ys[y],
            xs[x + 1], ys[y + 1]);
        occupied_count += occupied[x][y] ? 1u : 0u;
      }
    }
    if (occupied_count != 3) {
      malformed("six-edge raw M2 polygon is not a three-cell L");
    }
    const std::uint32_t full_x =
        occupied[0][0] && occupied[0][1] ? 0u : 1u;
    const std::uint32_t other_x = 1u - full_x;
    if (!(occupied[full_x][0] && occupied[full_x][1]) ||
        occupied[other_x][0] == occupied[other_x][1]) {
      malformed("six-edge raw M2 L has no unique full column");
    }
    rectangles[0] = {
        xs[full_x], ys[0], xs[full_x + 1], ys[2],
        global_polygon};
    const std::uint32_t other_y =
        occupied[other_x][0] ? 0u : 1u;
    rectangles[1] = {
        xs[other_x], ys[other_y], xs[other_x + 1],
        ys[other_y + 1], global_polygon};
    rectangle_count = 2;
  }

  __int128 rectangle_area = 0;
  for (std::uint32_t index = 0;
       index < rectangle_count; ++index) {
    const RectangleTemplate &rectangle = rectangles[index];
    if (rectangle.left >= rectangle.right ||
        rectangle.bottom >= rectangle.top ||
        rectangle.source_token != global_polygon) {
      malformed(
          "raw M2 decomposition changed bounds or source token");
    }
    rectangle_area +=
        static_cast<__int128>(
            rectangle.right - rectangle.left) *
        static_cast<__int128>(
            rectangle.top - rectangle.bottom);
  }
  if (rectangle_area * 2 != abs_i128(polygon_area2)) {
    malformed("raw M2 decomposition changed exact polygon area");
  }
  if (destination && rectangle_count > destination_capacity) {
    malformed("raw M2 box/L second-pass decomposition diverged");
  }
  if (destination) {
    for (std::uint32_t index = 0;
         index < rectangle_count; ++index) {
      destination[index] = rectangles[index];
    }
  }
  return rectangle_count;
}

std::pair<std::int64_t, std::int64_t>
transform_point_host(const Context &context,
                     std::int64_t x, std::int64_t y)
{
  __int128 transformed_x = 0;
  __int128 transformed_y = 0;
  switch (context.transform_code) {
  case 0: transformed_x = x; transformed_y = y; break;
  case 1: transformed_x = -static_cast<__int128>(y);
          transformed_y = x; break;
  case 2: transformed_x = -static_cast<__int128>(x);
          transformed_y = -static_cast<__int128>(y); break;
  case 3: transformed_x = y;
          transformed_y = -static_cast<__int128>(x); break;
  case 4: transformed_x = x;
          transformed_y = -static_cast<__int128>(y); break;
  case 5: transformed_x = y; transformed_y = x; break;
  case 6: transformed_x = -static_cast<__int128>(x);
          transformed_y = y; break;
  case 7: transformed_x = -static_cast<__int128>(y);
          transformed_y = -static_cast<__int128>(x); break;
  default: malformed("raw M2 context has an invalid transform");
  }
  transformed_x += context.tx;
  transformed_y += context.ty;
  if (transformed_x <
          std::numeric_limits<std::int64_t>::min() ||
      transformed_x >
          std::numeric_limits<std::int64_t>::max() ||
      transformed_y <
          std::numeric_limits<std::int64_t>::min() ||
      transformed_y >
          std::numeric_limits<std::int64_t>::max()) {
    coordinate_decline("raw M2 transform overflows int64");
  }
  const std::int64_t result_x =
      static_cast<std::int64_t>(transformed_x);
  const std::int64_t result_y =
      static_cast<std::int64_t>(transformed_y);
  if (!coordinate_qualified(result_x) ||
      !coordinate_qualified(result_y)) {
    coordinate_decline(
        "raw M2 transform exceeds the qualified coordinate domain");
  }
  return {result_x, result_y};
}

void add_world_polygon_bounds(
    const Context &context, const Polygon &polygon,
    bool *have_bounds, std::int64_t *left,
    std::int64_t *bottom, std::int64_t *right,
    std::int64_t *top)
{
  const std::int64_t xs[2] =
      {polygon.left, polygon.right};
  const std::int64_t ys[2] =
      {polygon.bottom, polygon.top};
  for (int x_index = 0; x_index < 2; ++x_index) {
    for (int y_index = 0; y_index < 2; ++y_index) {
      const auto point =
          transform_point_host(
              context, xs[x_index], ys[y_index]);
      if (!*have_bounds) {
        *left = *right = point.first;
        *bottom = *top = point.second;
        *have_bounds = true;
      } else {
        *left = std::min(*left, point.first);
        *bottom = std::min(*bottom, point.second);
        *right = std::max(*right, point.first);
        *top = std::max(*top, point.second);
      }
    }
  }
}

std::vector<CellCensus>
validate_cells_and_polygons(const Request &request)
{
  std::vector<CellCensus> census(
      static_cast<std::size_t>(request.cell_count));
  std::unordered_set<std::uint64_t> source_cells;
  source_cells.reserve(
      static_cast<std::size_t>(request.cell_count));

  std::uint64_t next_polygon = 0;
  std::uint64_t next_edge = 0;
  std::uint64_t next_rectangle = 0;
  for (std::uint64_t cell_id = 0;
       cell_id < request.cell_count; ++cell_id) {
    const Cell cell = load_record<Cell>(
        request.cells, cell_id, request.cell_record_bytes);
    if (!source_cells.insert(cell.source_cell_index).second ||
        cell.polygon_begin != next_polygon ||
        cell.edge_begin != next_edge) {
      malformed(
          "raw M2 cell ranges or source identities are inconsistent",
          KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN);
    }
    std::uint64_t polygon_end = 0;
    std::uint64_t edge_end = 0;
    if (!checked_add_u64(
            cell.polygon_begin, cell.polygon_count, &polygon_end) ||
        polygon_end > request.polygon_count ||
        !checked_add_u64(
            cell.edge_begin, cell.edge_count, &edge_end) ||
        edge_end > request.edge_count) {
      malformed(
          "raw M2 cell range escapes its record array",
          KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN);
    }
    CellCensus &cell_census =
        census[static_cast<std::size_t>(cell_id)];
    cell_census.rectangle_begin = next_rectangle;
    cell_census.polygon_count = cell.polygon_count;
    std::uint64_t cell_edge = cell.edge_begin;
    std::uint64_t rectangle_count = 0;
    std::uint64_t l_shape_count = 0;
    for (std::uint32_t local = 0;
         local < cell.polygon_count; ++local) {
      const std::uint64_t polygon_id =
          cell.polygon_begin + local;
      const Polygon polygon = load_record<Polygon>(
          request.polygons, polygon_id,
          request.polygon_record_bytes);
      if (polygon.polygon_id != local ||
          polygon.edge_begin != cell_edge) {
        malformed(
            "raw M2 polygon order escapes its owning cell",
            KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN);
      }
      const std::uint32_t produced =
          decompose_polygon(
              request, polygon, polygon_id, nullptr, 0);
      if (!checked_add_u64(
              rectangle_count, produced, &rectangle_count) ||
          !checked_add_u64(
              cell_edge, polygon.edge_count, &cell_edge)) {
        malformed("raw M2 polygon census overflows uint64");
      }
      if (polygon.edge_count == 6 && produced == 2) {
        ++l_shape_count;
      }
    }
    if (cell_edge != edge_end ||
        rectangle_count >
            std::numeric_limits<std::uint32_t>::max() ||
        l_shape_count >
            std::numeric_limits<std::uint32_t>::max() ||
        !checked_add_u64(
            next_rectangle, rectangle_count, &next_rectangle)) {
      malformed(
          "raw M2 per-cell rectangle or edge census is inconsistent",
          KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN);
    }
    if (next_rectangle > request.max_rectangles) {
      capacity(
          "stored raw M2 rectangle templates exceed capacity");
    }
    cell_census.rectangle_count =
        static_cast<std::uint32_t>(rectangle_count);
    cell_census.l_shape_count =
        static_cast<std::uint32_t>(l_shape_count);
    next_polygon = polygon_end;
    next_edge = edge_end;
  }
  if (next_polygon != request.polygon_count ||
      next_edge != request.edge_count) {
    malformed(
        "raw M2 record arrays have unowned trailing records",
        KLAYOUT_CUDA_SPATIAL_FALLBACK_RECORD_CELL_SPAN);
  }
  return census;
}

std::uint64_t validate_contexts_and_census(
    const Request &request,
    const std::vector<CellCensus> &cells)
{
  for (std::uint64_t context_id = 0;
       context_id < request.context_count; ++context_id) {
    const Context context = load_record<Context>(
        request.contexts, context_id,
        request.context_record_bytes);
    if (context.cell_id >= request.cell_count ||
        context.transform_code >= 8 ||
        !coordinate_qualified(context.tx) ||
        !coordinate_qualified(context.ty)) {
      malformed("raw M2 context is invalid");
    }
  }
  const Context root = load_record<Context>(
      request.contexts, 0, request.context_record_bytes);
  if (root.tx != 0 || root.ty != 0 ||
      root.cell_id != request.root_cell ||
      root.transform_code != 0) {
    malformed("raw M2 root context is not canonical");
  }

  bool have_bounds = false;
  std::int64_t scene_left = 0;
  std::int64_t scene_bottom = 0;
  std::int64_t scene_right = 0;
  std::int64_t scene_top = 0;
  std::uint64_t list_id = 0;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_edges = 0;
  std::uint64_t flat_rectangles = 0;
  for (std::uint64_t context_id = 0;
       context_id < request.context_count; ++context_id) {
    const Context context = load_record<Context>(
        request.contexts, context_id,
        request.context_record_bytes);
    const Cell cell = load_record<Cell>(
        request.cells, context.cell_id,
        request.cell_record_bytes);
    if (!cell.polygon_count) continue;
    if (list_id >= request.metal_context_count ||
        load_scalar(request.metal_contexts, list_id) !=
            context_id ||
        load_scalar(
            request.context_polygon_offsets, list_id) !=
            flat_polygons ||
        load_scalar(
            request.context_edge_offsets, list_id) !=
            flat_edges ||
        !checked_add_u64(
            flat_polygons, cell.polygon_count,
            &flat_polygons) ||
        !checked_add_u64(
            flat_edges, cell.edge_count, &flat_edges) ||
        !checked_add_u64(
            flat_rectangles,
            cells[context.cell_id].rectangle_count,
            &flat_rectangles)) {
      malformed("raw M2 flattened context census is inconsistent");
    }
    if (flat_rectangles > request.max_rectangles) {
      capacity("flattened raw M2 rectangles exceed capacity");
    }
    const std::uint64_t polygon_end =
        cell.polygon_begin + cell.polygon_count;
    for (std::uint64_t polygon_id = cell.polygon_begin;
         polygon_id < polygon_end; ++polygon_id) {
      const Polygon polygon = load_record<Polygon>(
          request.polygons, polygon_id,
          request.polygon_record_bytes);
      add_world_polygon_bounds(
          context, polygon, &have_bounds, &scene_left,
          &scene_bottom, &scene_right, &scene_top);
    }
    ++list_id;
  }
  if (!have_bounds ||
      list_id != request.metal_context_count ||
      flat_polygons != request.flat_polygon_count ||
      flat_edges != request.flat_edge_count ||
      !flat_rectangles ||
      scene_left != request.scene_left ||
      scene_bottom != request.scene_bottom ||
      scene_right != request.scene_right ||
      scene_top != request.scene_top) {
    malformed("raw M2 world bounds or flattened census mismatch");
  }
  const __int128 y_range =
      static_cast<__int128>(request.scene_top) -
      request.scene_bottom;
  if (y_range <= 0 ||
      y_range >
          std::numeric_limits<std::uint32_t>::max()) {
    coordinate_decline(
        "raw M2 y range exceeds exact packed-union capacity");
  }
  return flat_rectangles;
}

LoweredScene lower_scene(
    const Request &request,
    const std::vector<CellCensus> &census,
    std::uint64_t flat_rectangles)
{
  LoweredScene lowered;
  lowered.flat_rectangles = flat_rectangles;
  lowered.cells.resize(census.size());
  std::uint64_t local_rectangles = 0;
  if (!census.empty()) {
    const CellCensus &last = census.back();
    if (!checked_add_u64(
            last.rectangle_begin, last.rectangle_count,
            &local_rectangles)) {
      malformed("raw M2 local rectangle census overflows");
    }
  }
  if (local_rectangles >
          std::numeric_limits<std::size_t>::max() /
              sizeof(RectangleTemplate) ||
      flat_rectangles >
          std::numeric_limits<std::size_t>::max() /
              sizeof(mu::RectI64)) {
    capacity("raw M2 rectangle allocation exceeds host size_t");
  }
  lowered.rectangles.resize(
      static_cast<std::size_t>(local_rectangles));

  for (std::uint64_t cell_id = 0;
       cell_id < request.cell_count; ++cell_id) {
    const Cell source = load_record<Cell>(
        request.cells, cell_id, request.cell_record_bytes);
    const CellCensus &source_census =
        census[static_cast<std::size_t>(cell_id)];
    LoweredCell &destination =
        lowered.cells[static_cast<std::size_t>(cell_id)];
    destination.rectangle_begin =
        source_census.rectangle_begin;
    destination.rectangle_count =
        source_census.rectangle_count;
    destination.polygon_count =
        source_census.polygon_count;
    destination.l_shape_count =
        source_census.l_shape_count;
    destination.reserved = 0;

    std::uint64_t rectangle_id =
        source_census.rectangle_begin;
    std::uint64_t cell_rectangle_end = 0;
    if (!checked_add_u64(
            source_census.rectangle_begin,
            source_census.rectangle_count,
            &cell_rectangle_end)) {
      malformed("raw M2 second-pass cell rectangle span overflows");
    }
    for (std::uint32_t local = 0;
         local < source.polygon_count; ++local) {
      const std::uint64_t polygon_id =
          source.polygon_begin + local;
      const Polygon polygon = load_record<Polygon>(
          request.polygons, polygon_id,
          request.polygon_record_bytes);
      if (rectangle_id > cell_rectangle_end ||
          cell_rectangle_end > lowered.rectangles.size()) {
        malformed(
            "raw M2 second-pass rectangle census diverged");
      }
      const std::uint64_t remaining =
          cell_rectangle_end - rectangle_id;
      RectangleTemplate *produced =
          remaining
              ? lowered.rectangles.data() +
                    static_cast<std::size_t>(rectangle_id)
              : nullptr;
      const std::uint32_t count = decompose_polygon(
          request, polygon, polygon_id, produced, remaining);
      if (count > remaining) {
        malformed(
            "raw M2 second-pass rectangle census diverged");
      }
      rectangle_id += count;
    }
    if (rectangle_id != cell_rectangle_end) {
      malformed("raw M2 second-pass cell decomposition diverged");
    }
  }

  lowered.rectangle_offsets.resize(
      static_cast<std::size_t>(
          request.metal_context_count));
  std::uint64_t offset = 0;
  for (std::uint64_t list_id = 0;
       list_id < request.metal_context_count; ++list_id) {
    lowered.rectangle_offsets[
        static_cast<std::size_t>(list_id)] = offset;
    const std::uint32_t context_id =
        load_scalar(request.metal_contexts, list_id);
    const Context context = load_record<Context>(
        request.contexts, context_id,
        request.context_record_bytes);
    if (!checked_add_u64(
            offset, census[context.cell_id].rectangle_count,
            &offset)) {
      malformed("raw M2 rectangle offsets overflow uint64");
    }
  }
  if (offset != flat_rectangles) {
    malformed("raw M2 rectangle-offset conservation failed");
  }
  return lowered;
}

LoweredScene validate_and_lower(
    const Request &request, const char (&digest_magic)[8])
{
  const std::vector<CellCensus> cells =
      validate_cells_and_polygons(request);
  const std::uint64_t flat_rectangles =
      validate_contexts_and_census(request, cells);
  const std::array<std::uint8_t, 32> digest =
      request_digest(request, digest_magic);
  if (!std::equal(
          digest.begin(), digest.end(), request.scene_digest)) {
    malformed("raw Manhattan scene digest mismatch");
  }
  // Only compact host templates and offsets are materialized here.  Device
  // allocation and the 22.9M-record production world stream remain strictly
  // after the complete structural/census/digest gate.
  return lower_scene(request, cells, flat_rectangles);
}

void validate_without_lowering(
    const Request &request, const char (&digest_magic)[8])
{
  const std::vector<CellCensus> cells =
      validate_cells_and_polygons(request);
  validate_contexts_and_census(request, cells);
  const std::array<std::uint8_t, 32> digest =
      request_digest(request, digest_magic);
  if (!std::equal(
          digest.begin(), digest.end(), request.scene_digest)) {
    malformed("raw Manhattan scene digest mismatch");
  }
}

__device__ bool negate_checked(std::int64_t value,
                               std::int64_t *result)
{
  if (value == INT64_MIN) return false;
  *result = -value;
  return true;
}

__device__ bool add_checked(std::int64_t first,
                            std::int64_t second,
                            std::int64_t *result)
{
  if ((second > 0 && first > INT64_MAX - second) ||
      (second < 0 && first < INT64_MIN - second)) {
    return false;
  }
  *result = first + second;
  return true;
}

__device__ bool transform_point_device(
    const Context &context, std::int64_t x, std::int64_t y,
    std::int64_t *output_x, std::int64_t *output_y)
{
  std::int64_t transformed_x = 0;
  std::int64_t transformed_y = 0;
  switch (context.transform_code) {
  case 0: transformed_x = x; transformed_y = y; break;
  case 1:
    if (!negate_checked(y, &transformed_x)) return false;
    transformed_y = x;
    break;
  case 2:
    if (!negate_checked(x, &transformed_x) ||
        !negate_checked(y, &transformed_y)) {
      return false;
    }
    break;
  case 3:
    transformed_x = y;
    if (!negate_checked(x, &transformed_y)) return false;
    break;
  case 4:
    transformed_x = x;
    if (!negate_checked(y, &transformed_y)) return false;
    break;
  case 5: transformed_x = y; transformed_y = x; break;
  case 6:
    if (!negate_checked(x, &transformed_x)) return false;
    transformed_y = y;
    break;
  case 7:
    if (!negate_checked(y, &transformed_x) ||
        !negate_checked(x, &transformed_y)) {
      return false;
    }
    break;
  default: return false;
  }
  return add_checked(transformed_x, context.tx, output_x) &&
         add_checked(transformed_y, context.ty, output_y);
}

__device__ bool transform_rectangle_device(
    const Context &context,
    const RectangleTemplate &source,
    mu::RectI64 *destination,
    std::uint64_t context_token)
{
  const std::int64_t xs[4] = {
      source.left, source.left, source.right, source.right};
  const std::int64_t ys[4] = {
      source.bottom, source.top, source.bottom, source.top};
  mu::RectI64 transformed = {
      INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN,
      source.source_token, context_token};
  for (int corner = 0; corner < 4; ++corner) {
    std::int64_t x = 0;
    std::int64_t y = 0;
    if (!transform_point_device(
            context, xs[corner], ys[corner], &x, &y)) {
      return false;
    }
    transformed.left = min(transformed.left, x);
    transformed.bottom = min(transformed.bottom, y);
    transformed.right = max(transformed.right, x);
    transformed.top = max(transformed.top, y);
  }
  if (transformed.left >= transformed.right ||
      transformed.bottom >= transformed.top) {
    return false;
  }
  *destination = transformed;
  return true;
}

__global__ void expand_rectangles_kernel(
    const Context *contexts,
    const std::uint32_t *metal_contexts,
    const std::uint64_t *rectangle_offsets,
    const LoweredCell *cells,
    const RectangleTemplate *templates,
    std::uint64_t metal_context_count,
    std::int64_t scene_left, std::int64_t scene_bottom,
    std::int64_t scene_right, std::int64_t scene_top,
    mu::RectI64 *rectangles, std::uint32_t *status)
{
  for (std::uint64_t list_id = blockIdx.x;
       list_id < metal_context_count;
       list_id += gridDim.x) {
    const std::uint32_t context_id =
        metal_contexts[list_id];
    const Context context = contexts[context_id];
    const LoweredCell cell = cells[context.cell_id];
    for (std::uint32_t local = threadIdx.x;
         local < cell.rectangle_count;
         local += blockDim.x) {
      mu::RectI64 rectangle{};
      if (!transform_rectangle_device(
              context,
              templates[cell.rectangle_begin + local],
              &rectangle, context_id)) {
        atomicOr(status, std::uint32_t(kExpandTransformOverflow));
        continue;
      }
      if (rectangle.left < scene_left ||
          rectangle.bottom < scene_bottom ||
          rectangle.right > scene_right ||
          rectangle.top > scene_top ||
          rectangle.left < -kCoordinateLimit ||
          rectangle.bottom < -kCoordinateLimit ||
          rectangle.right > kCoordinateLimit ||
          rectangle.top > kCoordinateLimit) {
        atomicOr(status, std::uint32_t(kExpandBoundsMismatch));
        continue;
      }
      const std::uint64_t output =
          rectangle_offsets[list_id] + local;
      rectangles[output] = rectangle;
    }
  }
}

/*
 * The M1 suffix is invariant under the exact isometry (x, y) -> (y, x):
 * both morphology radii use square structuring elements, the long-edge
 * space predicate checks both axes symmetrically, and a COMPLETE certificate
 * requires F270 to be empty before any directional M1.6-.9 result is
 * consumed.  Transposing only this private resident stream lets horizontal
 * M1 rails be swept along their narrow axis without changing the request
 * digest, hierarchy census, or any caller-owned coordinate.
 */
__global__ void transpose_m1_rectangles_kernel(
    mu::RectI64 *rectangles, std::uint64_t rectangle_count,
    std::int64_t scene_left, std::int64_t scene_bottom,
    std::int64_t scene_right, std::int64_t scene_top,
    std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < rectangle_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const mu::RectI64 source = rectangles[index];
    if (source.left >= source.right ||
        source.bottom >= source.top ||
        source.left < scene_left ||
        source.bottom < scene_bottom ||
        source.right > scene_right ||
        source.top > scene_top ||
        source.left < -kCoordinateLimit ||
        source.bottom < -kCoordinateLimit ||
        source.right > kCoordinateLimit ||
        source.top > kCoordinateLimit) {
      atomicOr(status, std::uint32_t(kExpandTransposeInvariant));
      continue;
    }
    const mu::RectI64 transposed = {
        source.bottom, source.left, source.top, source.right,
        source.source_token, source.context_token};
    if (transposed.left >= transposed.right ||
        transposed.bottom >= transposed.top ||
        transposed.left < scene_bottom ||
        transposed.bottom < scene_left ||
        transposed.right > scene_top ||
        transposed.top > scene_right) {
      atomicOr(status, std::uint32_t(kExpandTransposeInvariant));
      continue;
    }
    rectangles[index] = transposed;
  }
}

/*
 * Exact IMPLANT.5 operator.  PPLUS rectangles are indexed in a bounded
 * uniform grid and NPLUS rectangles query that index.  Rectangle covers are
 * exact decompositions of their source polygons, so one strict positive-area
 * box intersection is equivalent to a nonempty NPLUS & PPLUS result.
 * Boundary and corner touching deliberately fail the strict inequalities.
 */
struct Implant15OverlapGrid
{
  std::int64_t base_x;
  std::int64_t base_y;
  std::int64_t cell_size;
  std::uint32_t width;
  std::uint32_t height;
};

enum Implant15OverlapFlag : std::uint32_t
{
  kImplant15OverlapInvalidRectangle = 1u << 0,
  kImplant15OverlapCoordinate = 1u << 1,
  kImplant15OverlapCountOverflow = 1u << 2,
  kImplant15OverlapFillMismatch = 1u << 3,
  kImplant15OverlapWorkCapacity = 1u << 4,
};

struct Implant15OverlapCounters
{
  unsigned long long member_visits;
  unsigned long long hits;
};

__host__ __device__ std::int64_t implant15_floor_div(
    std::int64_t value, std::int64_t divisor)
{
  const std::int64_t quotient = value / divisor;
  const std::int64_t remainder = value % divisor;
  return remainder < 0 ? quotient - 1 : quotient;
}

__device__ bool implant15_rectangle_span(
    const mu::RectI64 &rectangle, Implant15OverlapGrid grid,
    std::int64_t *x0, std::int64_t *y0,
    std::int64_t *x1, std::int64_t *y1)
{
  if (rectangle.left >= rectangle.right ||
      rectangle.bottom >= rectangle.top ||
      rectangle.left < -kCoordinateLimit ||
      rectangle.bottom < -kCoordinateLimit ||
      rectangle.right > kCoordinateLimit ||
      rectangle.top > kCoordinateLimit ||
      rectangle.right == INT64_MIN ||
      rectangle.top == INT64_MIN) {
    return false;
  }
  const std::int64_t global_x0 =
      implant15_floor_div(rectangle.left, grid.cell_size);
  const std::int64_t global_y0 =
      implant15_floor_div(rectangle.bottom, grid.cell_size);
  const std::int64_t global_x1 =
      implant15_floor_div(rectangle.right - 1, grid.cell_size);
  const std::int64_t global_y1 =
      implant15_floor_div(rectangle.top - 1, grid.cell_size);
  if (global_x0 < grid.base_x || global_y0 < grid.base_y) {
    return false;
  }
  *x0 = global_x0 - grid.base_x;
  *y0 = global_y0 - grid.base_y;
  *x1 = global_x1 - grid.base_x;
  *y1 = global_y1 - grid.base_y;
  return *x0 >= 0 && *y0 >= 0 && *x0 <= *x1 && *y0 <= *y1 &&
         static_cast<std::uint64_t>(*x1) < grid.width &&
         static_cast<std::uint64_t>(*y1) < grid.height;
}

__device__ std::uint64_t implant15_overlap_grid_index(
    Implant15OverlapGrid grid, std::int64_t x, std::int64_t y)
{
  return static_cast<std::uint64_t>(y) * grid.width +
         static_cast<std::uint64_t>(x);
}

__global__ void implant15_count_overlap_memberships_kernel(
    const mu::RectI64 *pplus, std::uint64_t pplus_count,
    Implant15OverlapGrid grid, std::uint32_t *cell_counts,
    std::uint64_t max_work,
    unsigned long long *membership_total, std::uint32_t *status)
{
  unsigned long long local_total = 0;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < pplus_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!implant15_rectangle_span(
            pplus[index], grid, &x0, &y0, &x1, &y1)) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapInvalidRectangle));
      continue;
    }
    const std::uint64_t width =
        static_cast<std::uint64_t>(x1 - x0) + 1;
    const std::uint64_t height =
        static_cast<std::uint64_t>(y1 - y0) + 1;
    if (!width || !height || width > UINT64_MAX / height) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapCoordinate));
      continue;
    }
    const std::uint64_t rectangle_memberships = width * height;
    if (local_total > max_work ||
        rectangle_memberships > max_work - local_total) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapWorkCapacity));
      local_total = max_work;
    } else {
      local_total += rectangle_memberships;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell =
            implant15_overlap_grid_index(grid, x, y);
        if (atomicAdd(cell_counts + cell, 1u) == UINT32_MAX) {
          atomicOr(
              status,
              static_cast<std::uint32_t>(
                  kImplant15OverlapCountOverflow));
        }
      }
    }
  }
  if (local_total) {
    const unsigned long long previous =
        atomicAdd(membership_total, local_total);
    if (previous > ULLONG_MAX - local_total) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapCountOverflow));
    }
  }
}

__global__ void implant15_fill_overlap_memberships_kernel(
    const mu::RectI64 *pplus, std::uint64_t pplus_count,
    Implant15OverlapGrid grid, unsigned long long *cell_cursors,
    std::uint32_t *members, std::uint64_t member_capacity,
    std::uint32_t *status)
{
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       index < pplus_count;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!implant15_rectangle_span(
            pplus[index], grid, &x0, &y0, &x1, &y1)) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapInvalidRectangle));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t cell =
            implant15_overlap_grid_index(grid, x, y);
        const unsigned long long position =
            atomicAdd(cell_cursors + cell, 1ull);
        if (position >= member_capacity || index > UINT32_MAX) {
          atomicOr(
              status,
              static_cast<std::uint32_t>(
                  kImplant15OverlapFillMismatch));
        } else {
          members[position] = static_cast<std::uint32_t>(index);
        }
      }
    }
  }
}

__global__ void implant15_validate_overlap_grid_kernel(
    const std::uint32_t *counts, const std::uint64_t *offsets,
    const unsigned long long *cursors, std::uint64_t cell_count,
    std::uint32_t *status)
{
  for (std::uint64_t cell =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       cell < cell_count;
       cell += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    if (cursors[cell] != offsets[cell] + counts[cell]) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapFillMismatch));
    }
  }
}

__global__ void implant15_query_overlap_kernel(
    const mu::RectI64 *nplus, std::uint64_t nplus_count,
    const mu::RectI64 *pplus, std::uint64_t pplus_count,
    Implant15OverlapGrid grid, const std::uint32_t *cell_counts,
    const std::uint64_t *cell_offsets, const std::uint32_t *members,
    std::uint64_t max_work, Implant15OverlapCounters *counters,
    std::uint32_t *status)
{
  unsigned long long local_visits = 0;
  unsigned long long local_hits = 0;
  for (std::uint64_t nplus_id =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       nplus_id < nplus_count;
       nplus_id +=
           static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const mu::RectI64 first = nplus[nplus_id];
    std::int64_t first_x0 = 0;
    std::int64_t first_y0 = 0;
    std::int64_t first_x1 = 0;
    std::int64_t first_y1 = 0;
    if (!implant15_rectangle_span(
            first, grid, &first_x0, &first_y0,
            &first_x1, &first_y1)) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapInvalidRectangle));
      continue;
    }
    for (std::int64_t y = first_y0; y <= first_y1; ++y) {
      for (std::int64_t x = first_x0; x <= first_x1; ++x) {
        const std::uint64_t cell =
            implant15_overlap_grid_index(grid, x, y);
        const std::uint64_t begin = cell_offsets[cell];
        const std::uint64_t end = begin + cell_counts[cell];
        for (std::uint64_t position = begin;
             position < end; ++position) {
          if (local_visits == max_work) {
            atomicOr(
                status,
                static_cast<std::uint32_t>(
                    kImplant15OverlapWorkCapacity));
          } else {
            ++local_visits;
          }
          const std::uint32_t pplus_id = members[position];
          if (pplus_id >= pplus_count) {
            atomicOr(
                status,
                static_cast<std::uint32_t>(
                    kImplant15OverlapFillMismatch));
            continue;
          }
          const mu::RectI64 second = pplus[pplus_id];
          std::int64_t second_x0 = 0;
          std::int64_t second_y0 = 0;
          std::int64_t second_x1 = 0;
          std::int64_t second_y1 = 0;
          if (!implant15_rectangle_span(
                  second, grid, &second_x0, &second_y0,
                  &second_x1, &second_y1)) {
            atomicOr(
                status,
                static_cast<std::uint32_t>(
                    kImplant15OverlapInvalidRectangle));
            continue;
          }
          // Exactly one grid cell owns every rectangle pair.
          if (x != max(first_x0, second_x0) ||
              y != max(first_y0, second_y0)) {
            continue;
          }
          if (first.left < second.right &&
              second.left < first.right &&
              first.bottom < second.top &&
              second.bottom < first.top) {
            ++local_hits;
          }
        }
      }
    }
  }
  if (local_visits) {
    const unsigned long long previous =
        atomicAdd(&counters->member_visits, local_visits);
    if (previous > ULLONG_MAX - local_visits ||
        previous > max_work ||
        local_visits > max_work - previous) {
      atomicOr(
          status,
          static_cast<std::uint32_t>(
              kImplant15OverlapWorkCapacity));
    }
  }
  if (local_hits) atomicAdd(&counters->hits, local_hits);
}

struct Implant15OverlapResult
{
  std::uint64_t grid_cells = 0;
  std::uint64_t memberships = 0;
  std::uint64_t candidate_visits = 0;
  std::uint64_t hits = 0;
  std::uint32_t device_flags = 0;
  std::uint64_t elapsed_ns = 0;
};

__global__ void expand_contact_edges_kernel(
    const Context *contexts, std::uint64_t context_count,
    const std::uint32_t *layer_contexts,
    const std::uint64_t *edge_offsets,
    std::uint64_t layer_context_count, const Cell *cells,
    std::uint64_t cell_count, const Polygon *polygons,
    std::uint64_t polygon_count, const Edge *templates,
    std::uint64_t template_count,
    std::int64_t scene_left, std::int64_t scene_bottom,
    std::int64_t scene_right, std::int64_t scene_top,
    a3::DirectedEdge *expanded, std::uint64_t expanded_count,
    std::uint32_t *status)
{
  for (std::uint64_t list_id = blockIdx.x;
       list_id < layer_context_count;
       list_id += gridDim.x) {
    const std::uint32_t context_id = layer_contexts[list_id];
    if (context_id >= context_count) {
      atomicOr(status, std::uint32_t(kExpandInvalidRecord));
      continue;
    }
    const Context context = contexts[context_id];
    if (context.cell_id >= cell_count) {
      atomicOr(status, std::uint32_t(kExpandInvalidRecord));
      continue;
    }
    const Cell cell = cells[context.cell_id];
    const std::uint64_t output_base = edge_offsets[list_id];
    if (output_base > expanded_count ||
        cell.edge_count > expanded_count - output_base ||
        cell.polygon_begin > polygon_count ||
        cell.polygon_count >
            polygon_count - cell.polygon_begin ||
        cell.edge_begin > template_count ||
        cell.edge_count > template_count - cell.edge_begin) {
      atomicOr(status, std::uint32_t(kExpandInvalidRecord));
      continue;
    }

    const bool mirrored = context.transform_code >= 4;
    for (std::uint32_t polygon_local = 0;
         polygon_local < cell.polygon_count; ++polygon_local) {
      const Polygon polygon =
          polygons[cell.polygon_begin + polygon_local];
      if (polygon.edge_begin < cell.edge_begin ||
          polygon.edge_begin > template_count ||
          polygon.edge_count >
              template_count - polygon.edge_begin) {
        atomicOr(status, std::uint32_t(kExpandInvalidRecord));
        continue;
      }
      const std::uint64_t cell_edge_local =
          polygon.edge_begin - cell.edge_begin;
      if (cell_edge_local > cell.edge_count ||
          polygon.edge_count >
              cell.edge_count - cell_edge_local) {
        atomicOr(status, std::uint32_t(kExpandInvalidRecord));
        continue;
      }

      for (std::uint32_t output_local = threadIdx.x;
           output_local < polygon.edge_count;
           output_local += blockDim.x) {
        const std::uint32_t source_local =
            mirrored
                ? polygon.edge_count - 1 - output_local
                : output_local;
        const Edge source =
            templates[polygon.edge_begin + source_local];
        std::int64_t x1 = 0;
        std::int64_t y1 = 0;
        std::int64_t x2 = 0;
        std::int64_t y2 = 0;
        if (!transform_point_device(
                context, source.x1, source.y1, &x1, &y1) ||
            !transform_point_device(
                context, source.x2, source.y2, &x2, &y2)) {
          atomicOr(
              status, std::uint32_t(kExpandTransformOverflow));
          continue;
        }
        a3::DirectedEdge destination = mirrored
            ? a3::DirectedEdge{x2, y2, x1, y1}
            : a3::DirectedEdge{x1, y1, x2, y2};
        if ((destination.x1 == destination.x2 &&
             destination.y1 == destination.y2) ||
            !(destination.x1 == destination.x2 ||
              destination.y1 == destination.y2) ||
            destination.x1 < scene_left ||
            destination.x1 > scene_right ||
            destination.x2 < scene_left ||
            destination.x2 > scene_right ||
            destination.y1 < scene_bottom ||
            destination.y1 > scene_top ||
            destination.y2 < scene_bottom ||
            destination.y2 > scene_top) {
          atomicOr(status, std::uint32_t(kExpandBoundsMismatch));
          continue;
        }
        const std::uint64_t output =
            output_base + cell_edge_local + output_local;
        if (output >= expanded_count) {
          atomicOr(status, std::uint32_t(kExpandInvalidRecord));
          continue;
        }
        expanded[output] = destination;
      }
    }
  }
}

std::uint32_t fallback_flags_for_union(
    const mu::GpuUnionOutput &output)
{
  if (output.message.find("packed y") != std::string::npos ||
      output.message.find("coordinate") != std::string::npos) {
    return KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW;
  }
  if (output.message.find("capacity") != std::string::npos ||
      output.message.find("slab") != std::string::npos) {
    return KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY;
  }
  return KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
}

bool segment_less(const Segment &first, const Segment &second)
{
  if (first.axis != second.axis) return first.axis < second.axis;
  if (first.side != second.side) return first.side < second.side;
  if (first.fixed != second.fixed) return first.fixed < second.fixed;
  if (first.lo != second.lo) return first.lo < second.lo;
  return first.hi < second.hi;
}

bool same_segment_line(const Segment &first, const Segment &second)
{
  return first.axis == second.axis &&
         first.side == second.side &&
         first.fixed == second.fixed;
}

std::uint64_t boundary_fnv64(
    const Segment *segments, std::uint64_t count)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&hash](std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
      hash *= UINT64_C(1099511628211);
    }
  };
  mix(count);
  for (std::uint64_t index = 0; index < count; ++index) {
    const Segment &segment = segments[index];
    mix(segment.axis);
    mix(static_cast<std::uint32_t>(segment.side));
    mix(static_cast<std::uint64_t>(segment.fixed));
    mix(static_cast<std::uint64_t>(segment.lo));
    mix(static_cast<std::uint64_t>(segment.hi));
  }
  return hash;
}

void validate_output(
    const Request &request, const LoweredScene &lowered,
    const mu::GpuUnionOutput &output,
    const std::vector<Segment> &segments)
{
  std::uint64_t expected_events = 0;
  if (output.rectangle_count != lowered.flat_rectangles ||
      output.rectangle_count < request.flat_polygon_count ||
      output.rectangle_count > request.max_rectangles ||
      !output.x_slabs ||
      output.x_slabs > request.max_x_slabs ||
      !output.memberships ||
      output.memberships > request.max_memberships ||
      !checked_multiply_u64(
          output.memberships, 2, &expected_events) ||
      output.event_count != expected_events ||
      output.event_count > request.max_events ||
      !output.strip_intervals ||
      output.strip_intervals > output.memberships ||
      output.raw_segments < segments.size() ||
      output.raw_segments > request.max_raw_segments ||
      segments.empty() ||
      segments.size() > request.max_segments) {
    throw std::runtime_error(
        "shared M2 union returned impossible proof counters");
  }
  for (std::size_t index = 0; index < segments.size(); ++index) {
    const Segment &segment = segments[index];
    if ((segment.axis !=
             KLAYOUT_CUDA_SPATIAL_M2_UNION_HORIZONTAL &&
         segment.axis !=
             KLAYOUT_CUDA_SPATIAL_M2_UNION_VERTICAL) ||
        (segment.side != -1 && segment.side != 1) ||
        segment.lo >= segment.hi) {
      throw std::runtime_error(
          "shared M2 union returned an invalid segment");
    }
    if (index) {
      const Segment &previous = segments[index - 1];
      if (!segment_less(previous, segment) ||
          (same_segment_line(previous, segment) &&
           segment.lo <= previous.hi)) {
        throw std::runtime_error(
            "shared M2 union returned a noncanonical boundary");
      }
    }
  }
  if (boundary_fnv64(segments.data(), segments.size()) !=
      output.digest) {
    throw std::runtime_error(
        "shared M2 union boundary FNV mismatch");
  }
}

void register_allocation(const Segment *segments)
{
  std::lock_guard<std::mutex> lock(allocation_mutex());
  if (!owned_allocations().insert(segments).second) {
    throw std::runtime_error(
        "duplicate M2 boundary allocation identity");
  }
  ++allocation_counters().allocations;
}

void fill_success_result(
    const Request &request, const LoweredScene &lowered,
    const mu::GpuUnionOutput &output, Result *result)
{
  std::vector<Segment> portable(output.segments.size());
  for (std::size_t index = 0;
       index < output.segments.size(); ++index) {
    const mu::DirectedSegmentI64 &source =
        output.segments[index];
    portable[index] = {
        source.fixed, source.lo, source.hi, source.side,
        static_cast<std::uint32_t>(source.axis)};
  }
  validate_output(request, lowered, output, portable);
  if (m2_test_fault().exchange(
          0, std::memory_order_acq_rel) == 1) {
    throw std::runtime_error(
        "synthetic failure after M2 output validation");
  }

  std::unique_ptr<Segment[]> storage(
      new Segment[portable.size()]);
  std::copy(portable.begin(), portable.end(), storage.get());
  register_allocation(storage.get());
  result->segments = storage.release();
  result->segment_count = portable.size();
  result->rectangle_count = output.rectangle_count;
  result->x_slab_count = output.x_slabs;
  result->membership_count = output.memberships;
  result->event_count = output.event_count;
  result->strip_interval_count = output.strip_intervals;
  result->raw_segment_count = output.raw_segments;
  result->boundary_fnv64 = output.digest;
}

struct ExpandedRectangles
{
  thrust::device_vector<mu::RectI64> rectangles;
  std::uint32_t status = 0;
  std::uint64_t h2d_ns = 0;
  std::uint64_t expand_ns = 0;
};

ExpandedRectangles expand_rectangles_resident(
    const Request &request, const LoweredScene &lowered)
{
  ExpandedRectangles expanded;
  const auto h2d_begin = Clock::now();
  cuda_require(
      cudaSetDevice(request.device), "raw Manhattan cudaSetDevice");
  {
    DeviceBuffer<Context> device_contexts(request.context_count);
    DeviceBuffer<std::uint32_t> device_metal_contexts(
        request.metal_context_count);
    DeviceBuffer<std::uint64_t> device_offsets(
        request.metal_context_count);
    DeviceBuffer<LoweredCell> device_cells(request.cell_count);
    DeviceBuffer<RectangleTemplate> device_templates(
        lowered.rectangles.size());
    DeviceBuffer<std::uint32_t> device_status(1);
    expanded.rectangles.resize(
        static_cast<std::size_t>(lowered.flat_rectangles));

    cuda_require(
        cudaMemcpy(
            device_contexts.get(), request.contexts,
            static_cast<std::size_t>(request.context_count) *
                sizeof(Context),
            cudaMemcpyHostToDevice),
        "raw Manhattan context H2D");
    cuda_require(
        cudaMemcpy(
            device_metal_contexts.get(), request.metal_contexts,
            static_cast<std::size_t>(
                request.metal_context_count) *
                sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        "raw Manhattan context-list H2D");
    cuda_require(
        cudaMemcpy(
            device_offsets.get(), lowered.rectangle_offsets.data(),
            lowered.rectangle_offsets.size() *
                sizeof(std::uint64_t),
            cudaMemcpyHostToDevice),
        "raw Manhattan rectangle-offset H2D");
    cuda_require(
        cudaMemcpy(
            device_cells.get(), lowered.cells.data(),
            lowered.cells.size() * sizeof(LoweredCell),
            cudaMemcpyHostToDevice),
        "raw Manhattan cell H2D");
    cuda_require(
        cudaMemcpy(
            device_templates.get(), lowered.rectangles.data(),
            lowered.rectangles.size() *
                sizeof(RectangleTemplate),
            cudaMemcpyHostToDevice),
        "raw Manhattan rectangle-template H2D");
    cuda_require(
        cudaMemset(
            device_status.get(), 0, sizeof(std::uint32_t)),
        "raw Manhattan expansion status clear");
    cuda_require(
        cudaDeviceSynchronize(),
        "raw Manhattan compact H2D synchronize");
    expanded.h2d_ns = elapsed_ns(h2d_begin, Clock::now());

    const auto expand_begin = Clock::now();
    const std::uint32_t blocks =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            request.metal_context_count, kMaximumBlocks));
    expand_rectangles_kernel<<<blocks, kExpandThreads>>>(
        device_contexts.get(), device_metal_contexts.get(),
        device_offsets.get(), device_cells.get(),
        device_templates.get(), request.metal_context_count,
        request.scene_left, request.scene_bottom,
        request.scene_right, request.scene_top,
        thrust::raw_pointer_cast(expanded.rectangles.data()),
        device_status.get());
    cuda_require(
        cudaGetLastError(), "raw Manhattan rectangle expansion launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "raw Manhattan rectangle expansion synchronize");
    cuda_require(
        cudaMemcpy(
            &expanded.status, device_status.get(),
            sizeof(expanded.status), cudaMemcpyDeviceToHost),
        "raw Manhattan rectangle expansion status D2H");
    expanded.expand_ns =
        elapsed_ns(expand_begin, Clock::now());
  }
  return expanded;
}

std::uint64_t transpose_m1_rectangles_resident(
    const Request &request, ExpandedRectangles *expanded)
{
  if (!expanded || expanded->status ||
      expanded->rectangles.empty()) {
    throw std::runtime_error(
        "invalid M1 resident transpose input");
  }
  const auto begin = Clock::now();
  DeviceBuffer<std::uint32_t> device_status(1);
  cuda_require(
      cudaMemset(device_status.get(), 0, sizeof(std::uint32_t)),
      "raw M1 transpose status clear");
  const std::uint64_t rectangle_count =
      expanded->rectangles.size();
  const std::uint32_t blocks =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          (rectangle_count + kExpandThreads - 1) /
              kExpandThreads,
          kMaximumBlocks));
  transpose_m1_rectangles_kernel<<<blocks, kExpandThreads>>>(
      thrust::raw_pointer_cast(expanded->rectangles.data()),
      rectangle_count, request.scene_left, request.scene_bottom,
      request.scene_right, request.scene_top,
      device_status.get());
  cuda_require(
      cudaGetLastError(), "raw M1 resident transpose launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "raw M1 resident transpose synchronize");
  std::uint32_t transpose_status = 0;
  cuda_require(
      cudaMemcpy(
          &transpose_status, device_status.get(),
          sizeof(transpose_status), cudaMemcpyDeviceToHost),
      "raw M1 resident transpose status D2H");
  expanded->status |= transpose_status;
  return elapsed_ns(begin, Clock::now());
}

Implant15OverlapGrid implant15_make_overlap_grid(
    const Implant15Request &request)
{
  const std::int64_t left =
      std::min(request.nplus.scene_left, request.pplus.scene_left);
  const std::int64_t bottom =
      std::min(request.nplus.scene_bottom, request.pplus.scene_bottom);
  const std::int64_t right =
      std::max(request.nplus.scene_right, request.pplus.scene_right);
  const std::int64_t top =
      std::max(request.nplus.scene_top, request.pplus.scene_top);
  const std::int64_t base_x =
      implant15_floor_div(left, request.grid_cell_size);
  const std::int64_t base_y =
      implant15_floor_div(bottom, request.grid_cell_size);
  const std::int64_t high_x =
      implant15_floor_div(right - 1, request.grid_cell_size);
  const std::int64_t high_y =
      implant15_floor_div(top - 1, request.grid_cell_size);
  const __int128 width =
      static_cast<__int128>(high_x) - base_x + 1;
  const __int128 height =
      static_cast<__int128>(high_y) - base_y + 1;
  if (right <= left || top <= bottom ||
      right == INT64_MIN || top == INT64_MIN ||
      width <= 0 || height <= 0 ||
      width > UINT32_MAX || height > UINT32_MAX ||
      width * height >
          static_cast<__int128>(
              request.capacity.max_grid_cells) ||
      width * height > UINT64_MAX) {
    capacity("IMPLANT.5 overlap grid exceeds capacity");
  }
  return {
      base_x, base_y, request.grid_cell_size,
      static_cast<std::uint32_t>(width),
      static_cast<std::uint32_t>(height)};
}

Implant15OverlapResult run_implant15_overlap(
    const thrust::device_vector<mu::RectI64> &nplus,
    const thrust::device_vector<mu::RectI64> &pplus,
    const Implant15Request &request)
{
  const auto begin = Clock::now();
  Implant15OverlapResult result;
  if (nplus.empty() || pplus.empty()) {
    result.elapsed_ns = elapsed_ns(begin, Clock::now());
    return result;
  }
  if (nplus.size() > UINT32_MAX || pplus.size() > UINT32_MAX) {
    capacity("IMPLANT.5 rectangle-id domain exceeds uint32");
  }
  const Implant15OverlapGrid grid =
      implant15_make_overlap_grid(request);
  result.grid_cells =
      static_cast<std::uint64_t>(grid.width) * grid.height;
  thrust::device_vector<std::uint32_t> status(1, 0);
  thrust::device_vector<unsigned long long>
      membership_total(1, 0);
  thrust::device_vector<std::uint32_t> cell_counts(
      result.grid_cells, 0);
  thrust::device_vector<std::uint64_t> cell_offsets(
      result.grid_cells);
  thrust::device_vector<unsigned long long> cell_cursors(
      result.grid_cells);
  const std::uint32_t pplus_blocks =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          (pplus.size() + kExpandThreads - 1) /
              kExpandThreads,
          kMaximumBlocks));
  implant15_count_overlap_memberships_kernel<<<
      pplus_blocks, kExpandThreads>>>(
      thrust::raw_pointer_cast(pplus.data()), pplus.size(), grid,
      thrust::raw_pointer_cast(cell_counts.data()),
      request.capacity.max_overlap_work,
      thrust::raw_pointer_cast(membership_total.data()),
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(),
      "IMPLANT.5 overlap membership-count launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "IMPLANT.5 overlap membership-count synchronize");
  unsigned long long memberships = 0;
  cuda_require(
      cudaMemcpy(
          &memberships,
          thrust::raw_pointer_cast(membership_total.data()),
          sizeof(memberships), cudaMemcpyDeviceToHost),
      "IMPLANT.5 overlap membership count D2H");
  cuda_require(
      cudaMemcpy(
          &result.device_flags,
          thrust::raw_pointer_cast(status.data()),
          sizeof(result.device_flags), cudaMemcpyDeviceToHost),
      "IMPLANT.5 overlap count status D2H");
  if (result.device_flags) {
    throw std::runtime_error(
        "IMPLANT.5 overlap membership invariant");
  }
  result.memberships = memberships;
  if (result.memberships >
      request.capacity.max_overlap_work) {
    capacity("IMPLANT.5 overlap membership capacity");
  }

  thrust::exclusive_scan(
      thrust::device, cell_counts.begin(), cell_counts.end(),
      cell_offsets.begin(), std::uint64_t{0});
  cuda_require(
      cudaMemcpy(
          thrust::raw_pointer_cast(cell_cursors.data()),
          thrust::raw_pointer_cast(cell_offsets.data()),
          result.grid_cells * sizeof(unsigned long long),
          cudaMemcpyDeviceToDevice),
      "IMPLANT.5 overlap offsets-to-cursors D2D");
  thrust::device_vector<std::uint32_t> members(
      result.memberships);
  implant15_fill_overlap_memberships_kernel<<<
      pplus_blocks, kExpandThreads>>>(
      thrust::raw_pointer_cast(pplus.data()), pplus.size(), grid,
      thrust::raw_pointer_cast(cell_cursors.data()),
      thrust::raw_pointer_cast(members.data()),
      result.memberships,
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(),
      "IMPLANT.5 overlap membership-fill launch");
  implant15_validate_overlap_grid_kernel<<<
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          (result.grid_cells + kExpandThreads - 1) /
              kExpandThreads,
          kMaximumBlocks)),
      kExpandThreads>>>(
      thrust::raw_pointer_cast(cell_counts.data()),
      thrust::raw_pointer_cast(cell_offsets.data()),
      thrust::raw_pointer_cast(cell_cursors.data()),
      result.grid_cells,
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(),
      "IMPLANT.5 overlap grid-validation launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "IMPLANT.5 overlap grid-build synchronize");

  thrust::device_vector<Implant15OverlapCounters> counters(1);
  cuda_require(
      cudaMemset(
          thrust::raw_pointer_cast(counters.data()), 0,
          sizeof(Implant15OverlapCounters)),
      "IMPLANT.5 overlap counters clear");
  const std::uint32_t nplus_blocks =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          (nplus.size() + kExpandThreads - 1) /
              kExpandThreads,
          kMaximumBlocks));
  implant15_query_overlap_kernel<<<
      nplus_blocks, kExpandThreads>>>(
      thrust::raw_pointer_cast(nplus.data()), nplus.size(),
      thrust::raw_pointer_cast(pplus.data()), pplus.size(),
      grid, thrust::raw_pointer_cast(cell_counts.data()),
      thrust::raw_pointer_cast(cell_offsets.data()),
      thrust::raw_pointer_cast(members.data()),
      request.capacity.max_overlap_work,
      thrust::raw_pointer_cast(counters.data()),
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(), "IMPLANT.5 overlap query launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "IMPLANT.5 overlap query synchronize");
  Implant15OverlapCounters host{};
  cuda_require(
      cudaMemcpy(
          &host, thrust::raw_pointer_cast(counters.data()),
          sizeof(host), cudaMemcpyDeviceToHost),
      "IMPLANT.5 overlap counters D2H");
  cuda_require(
      cudaMemcpy(
          &result.device_flags,
          thrust::raw_pointer_cast(status.data()),
          sizeof(result.device_flags), cudaMemcpyDeviceToHost),
      "IMPLANT.5 overlap final status D2H");
  result.candidate_visits = host.member_visits;
  result.hits = host.hits;
  result.elapsed_ns = elapsed_ns(begin, Clock::now());
  if (result.device_flags &
      static_cast<std::uint32_t>(
          kImplant15OverlapWorkCapacity)) {
    capacity("IMPLANT.5 overlap candidate-work capacity");
  }
  if (result.device_flags) {
    throw std::runtime_error(
        "IMPLANT.5 overlap device invariant");
  }
  if (result.candidate_visits >
      request.capacity.max_overlap_work) {
    capacity("IMPLANT.5 overlap candidate-work capacity");
  }
  return result;
}

struct ExpandedContacts
{
  thrust::device_vector<a3::DirectedEdge> edges;
  std::uint32_t status = 0;
  std::uint64_t h2d_ns = 0;
  std::uint64_t expand_ns = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t free_low_bytes = 0;
};

ExpandedContacts expand_contact_edges_resident(
    const Request &request)
{
  ExpandedContacts expanded;
  const auto h2d_begin = Clock::now();
  Clock::time_point expand_begin;
  cuda_require(
      cudaSetDevice(request.device),
      "raw CONTACT cudaSetDevice");
  {
    DeviceBuffer<Context> device_contexts(request.context_count);
    DeviceBuffer<std::uint32_t> device_layer_contexts(
        request.metal_context_count);
    DeviceBuffer<std::uint64_t> device_offsets(
        request.metal_context_count);
    DeviceBuffer<Cell> device_cells(request.cell_count);
    DeviceBuffer<Polygon> device_polygons(request.polygon_count);
    DeviceBuffer<Edge> device_templates(request.edge_count);
    DeviceBuffer<std::uint32_t> device_status(1);
    expanded.edges.resize(
        static_cast<std::size_t>(request.flat_edge_count));

    cuda_require(
        cudaMemcpy(
            device_contexts.get(), request.contexts,
            static_cast<std::size_t>(request.context_count) *
                sizeof(Context),
            cudaMemcpyHostToDevice),
        "raw CONTACT context H2D");
    cuda_require(
        cudaMemcpy(
            device_layer_contexts.get(), request.metal_contexts,
            static_cast<std::size_t>(
                request.metal_context_count) *
                sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        "raw CONTACT context-list H2D");
    cuda_require(
        cudaMemcpy(
            device_offsets.get(), request.context_edge_offsets,
            static_cast<std::size_t>(
                request.context_edge_offset_count) *
                sizeof(std::uint64_t),
            cudaMemcpyHostToDevice),
        "raw CONTACT edge-offset H2D");
    cuda_require(
        cudaMemcpy(
            device_cells.get(), request.cells,
            static_cast<std::size_t>(request.cell_count) *
                sizeof(Cell),
            cudaMemcpyHostToDevice),
        "raw CONTACT cell H2D");
    cuda_require(
        cudaMemcpy(
            device_polygons.get(), request.polygons,
            static_cast<std::size_t>(request.polygon_count) *
                sizeof(Polygon),
            cudaMemcpyHostToDevice),
        "raw CONTACT polygon H2D");
    cuda_require(
        cudaMemcpy(
            device_templates.get(), request.edges,
            static_cast<std::size_t>(request.edge_count) *
                sizeof(Edge),
            cudaMemcpyHostToDevice),
        "raw CONTACT edge-template H2D");
    cuda_require(
        cudaMemset(
            device_status.get(), 0, sizeof(std::uint32_t)),
        "raw CONTACT expansion status clear");
    cuda_require(
        cudaDeviceSynchronize(),
        "raw CONTACT compact H2D synchronize");
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cuda_require(
        cudaMemGetInfo(&free_bytes, &total_bytes),
        "raw CONTACT staging cudaMemGetInfo");
    expanded.device_total_bytes = total_bytes;
    expanded.free_low_bytes = free_bytes;
    expanded.h2d_ns = elapsed_ns(h2d_begin, Clock::now());

    expand_begin = Clock::now();
    const std::uint32_t blocks =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            request.metal_context_count, kMaximumBlocks));
    expand_contact_edges_kernel<<<blocks, kExpandThreads>>>(
        device_contexts.get(), request.context_count,
        device_layer_contexts.get(), device_offsets.get(),
        request.metal_context_count, device_cells.get(),
        request.cell_count, device_polygons.get(),
        request.polygon_count, device_templates.get(),
        request.edge_count, request.scene_left,
        request.scene_bottom, request.scene_right,
        request.scene_top,
        thrust::raw_pointer_cast(expanded.edges.data()),
        request.flat_edge_count, device_status.get());
    cuda_require(
        cudaGetLastError(), "raw CONTACT edge expansion launch");
    cuda_require(
        cudaDeviceSynchronize(),
        "raw CONTACT edge expansion synchronize");
    cuda_require(
        cudaMemcpy(
            &expanded.status, device_status.get(),
            sizeof(expanded.status), cudaMemcpyDeviceToHost),
        "raw CONTACT edge expansion status D2H");
  }
  // Include destruction of the compact staging buffers.  The expanded edge
  // array intentionally remains resident for the relation consumer.
  expanded.expand_ns =
      elapsed_ns(expand_begin, Clock::now());
  return expanded;
}

struct Contact4CallbackContext
{
  const Request *contact = nullptr;
  const Contact4Request *outer = nullptr;
  bool invoked = false;
  std::uint32_t expansion_status = 0;
  std::uint64_t contact_h2d_ns = 0;
  std::uint64_t contact_expand_ns = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t callback_free_begin_bytes = 0;
  std::uint64_t callback_free_low_bytes = 0;
  c4::DeviceResidentContext resident;
};

void consume_contact4_active_union_boundary(
    cudaStream_t stream,
    const mu::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count, void *opaque)
{
  Contact4CallbackContext *context =
      static_cast<Contact4CallbackContext *>(opaque);
  if (!context || context->invoked || !context->contact ||
      !context->outer) {
    throw std::runtime_error(
        "CONTACT4 fused boundary callback contract");
  }
  context->invoked = true;
  if (stream != nullptr) {
    throw std::runtime_error(
        "CONTACT4 fused callback requires the default stream");
  }

  std::size_t callback_free_begin = 0;
  std::size_t device_total = 0;
  cuda_require(
      cudaMemGetInfo(&callback_free_begin, &device_total),
      "CONTACT4 fused callback-entry cudaMemGetInfo");
  context->device_total_bytes = device_total;
  context->callback_free_begin_bytes = callback_free_begin;
  context->callback_free_low_bytes = callback_free_begin;

  ExpandedContacts expanded =
      expand_contact_edges_resident(*context->contact);
  context->expansion_status = expanded.status;
  context->contact_h2d_ns = expanded.h2d_ns;
  context->contact_expand_ns = expanded.expand_ns;
  if (expanded.device_total_bytes != context->device_total_bytes ||
      !expanded.free_low_bytes) {
    throw std::runtime_error(
        "raw CONTACT memory telemetry identity mismatch");
  }
  context->callback_free_low_bytes = std::min(
      context->callback_free_low_bytes, expanded.free_low_bytes);
  if (expanded.status) {
    throw std::runtime_error(
        "raw CONTACT device expansion failed its exact gate");
  }

  c4::DeviceRequest &device_request =
      context->resident.request;
  device_request.contacts.device_edges =
      thrust::raw_pointer_cast(expanded.edges.data());
  device_request.contacts.count =
      context->contact->flat_edge_count;
  device_request.contacts.bounds = c4::ContactBounds{
      context->contact->scene_left,
      context->contact->scene_bottom,
      context->contact->scene_right,
      context->contact->scene_top};
  device_request.distance = context->outer->distance;
  device_request.grid_cell_size =
      context->outer->grid_cell_size;
  device_request.device = context->outer->device;
  device_request.contact_direction_contract =
      c4::ContactDirectionContract::
          validated_material_on_right_contours;
  device_request.limits.max_contact_edges =
      context->outer->max_contact_edges;
  device_request.limits.max_grid_cells =
      context->outer->max_grid_cells;
  device_request.limits.max_memberships =
      context->outer->max_contact_memberships;
  device_request.limits.max_boundary_cell_visits =
      context->outer->max_boundary_cell_visits;
  device_request.limits.max_member_visits =
      context->outer->max_member_visits;
  device_request.limits.max_pair_work =
      context->outer->max_pair_work;
  device_request.limits.max_cells_per_contact_edge =
      context->outer->max_cells_per_contact_edge;
  device_request.limits.max_cells_per_boundary_edge =
      context->outer->max_cells_per_boundary_edge;

  const auto reconcile_memory = [context]() {
    c4::Result &result = context->resident.result;
    if (!result.device_total_bytes) return;
    if (result.device_total_bytes != context->device_total_bytes ||
        !result.callback_free_begin_bytes ||
        !result.callback_free_low_bytes) {
      throw std::runtime_error(
          "CONTACT4 fused resident memory telemetry mismatch");
    }
    result.callback_free_begin_bytes =
        context->callback_free_begin_bytes;
    result.callback_free_low_bytes = std::min(
        context->callback_free_low_bytes,
        result.callback_free_low_bytes);
    result.callback_incremental_peak_bytes =
        result.callback_free_begin_bytes -
        result.callback_free_low_bytes;
  };
  try {
    c4::consume_device_boundary_hook(
        stream, horizontal, horizontal_count, vertical,
        vertical_count, &context->resident);
  } catch (...) {
    reconcile_memory();
    throw;
  }
  reconcile_memory();
}

mu::ResidentBoundaryHook make_contact4_active_union_hook(
    Contact4CallbackContext *context)
{
  mu::ResidentBoundaryHook hook;
  hook.consume = &consume_contact4_active_union_boundary;
  hook.context = context;
  hook.stop_before_d2h = true;
  return hook;
}

struct Active3WellCallbackContext
{
  const Request *active = nullptr;
  const Active3WellRequest *outer = nullptr;
  bool invoked = false;
  std::uint32_t expansion_status = 0;
  std::uint64_t active_h2d_ns = 0;
  std::uint64_t active_expand_ns = 0;
  std::uint64_t device_total_bytes = 0;
  std::uint64_t callback_free_begin_bytes = 0;
  std::uint64_t callback_free_low_bytes = 0;
  c4::DeviceResidentContext resident;
};

void consume_active3_well_union_boundary(
    cudaStream_t stream,
    const mu::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count, void *opaque)
{
  Active3WellCallbackContext *context =
      static_cast<Active3WellCallbackContext *>(opaque);
  if (!context || context->invoked || !context->active ||
      !context->outer) {
    throw std::runtime_error(
        "ACTIVE3 WELL-union boundary callback contract");
  }
  context->invoked = true;
  if (stream != nullptr) {
    throw std::runtime_error(
        "ACTIVE3 WELL-union callback requires the default stream");
  }

  std::size_t callback_free_begin = 0;
  std::size_t device_total = 0;
  cuda_require(
      cudaMemGetInfo(&callback_free_begin, &device_total),
      "ACTIVE3 WELL-union callback-entry cudaMemGetInfo");
  context->device_total_bytes = device_total;
  context->callback_free_begin_bytes = callback_free_begin;
  context->callback_free_low_bytes = callback_free_begin;

  ExpandedContacts expanded =
      expand_contact_edges_resident(*context->active);
  context->expansion_status = expanded.status;
  context->active_h2d_ns = expanded.h2d_ns;
  context->active_expand_ns = expanded.expand_ns;
  if (expanded.device_total_bytes != context->device_total_bytes ||
      !expanded.free_low_bytes) {
    throw std::runtime_error(
        "raw ACTIVE memory telemetry identity mismatch");
  }
  context->callback_free_low_bytes = std::min(
      context->callback_free_low_bytes, expanded.free_low_bytes);
  if (expanded.status) {
    throw std::runtime_error(
        "raw ACTIVE device expansion failed its exact gate");
  }

  c4::DeviceRequest &device_request =
      context->resident.request;
  device_request.contacts.device_edges =
      thrust::raw_pointer_cast(expanded.edges.data());
  device_request.contacts.count =
      context->active->flat_edge_count;
  device_request.contacts.bounds = c4::ContactBounds{
      context->active->scene_left,
      context->active->scene_bottom,
      context->active->scene_right,
      context->active->scene_top};
  device_request.distance = context->outer->distance;
  device_request.grid_cell_size =
      context->outer->grid_cell_size;
  device_request.device = context->outer->device;
  device_request.contact_direction_contract =
      c4::ContactDirectionContract::
          validated_active3_secondary_material_on_right_contours;
  device_request.limits.max_contact_edges =
      context->outer->max_active_edges;
  device_request.limits.max_grid_cells =
      context->outer->max_grid_cells;
  device_request.limits.max_memberships =
      context->outer->max_active_memberships;
  device_request.limits.max_boundary_cell_visits =
      context->outer->max_active_cell_visits;
  device_request.limits.max_member_visits =
      context->outer->max_member_visits;
  device_request.limits.max_pair_work =
      context->outer->max_pair_work;
  device_request.limits.max_cells_per_contact_edge =
      context->outer->max_cells_per_active_edge;
  device_request.limits.max_cells_per_boundary_edge =
      context->outer->max_cells_per_well_edge;

  const auto reconcile_memory = [context]() {
    c4::Result &result = context->resident.result;
    if (!result.device_total_bytes) return;
    if (result.device_total_bytes != context->device_total_bytes ||
        !result.callback_free_begin_bytes ||
        !result.callback_free_low_bytes) {
      throw std::runtime_error(
          "ACTIVE3 WELL-union resident memory telemetry mismatch");
    }
    result.callback_free_begin_bytes =
        context->callback_free_begin_bytes;
    result.callback_free_low_bytes = std::min(
        context->callback_free_low_bytes,
        result.callback_free_low_bytes);
    result.callback_incremental_peak_bytes =
        result.callback_free_begin_bytes -
        result.callback_free_low_bytes;
  };
  try {
    c4::consume_device_boundary_hook(
        stream, horizontal, horizontal_count, vertical,
        vertical_count, &context->resident);
  } catch (...) {
    reconcile_memory();
    throw;
  }
  reconcile_memory();
}

mu::ResidentBoundaryHook make_active3_well_union_hook(
    Active3WellCallbackContext *context)
{
  mu::ResidentBoundaryHook hook;
  hook.consume = &consume_active3_well_union_boundary;
  hook.context = context;
  hook.stop_before_d2h = true;
  return hook;
}

struct Implant15BoundaryContext
{
  const Request *gate = nullptr;
  const Request *contact = nullptr;
  const Implant15Request *outer = nullptr;
  bool invoked = false;
  std::uint32_t expansion_status = 0;
  std::uint64_t gate_h2d_ns = 0;
  std::uint64_t gate_expand_ns = 0;
  std::uint64_t contact_h2d_ns = 0;
  std::uint64_t contact_expand_ns = 0;
  c4::DeviceResidentContext implant1;
  c4::DeviceResidentContext implant2;
};

void consume_implant15_secondary(
    cudaStream_t stream,
    const mu::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count,
    const Request &secondary, std::int64_t distance,
    std::uint64_t max_edges, std::uint64_t max_cell_visits,
    const Implant15Request &outer,
    std::uint64_t *h2d_ns, std::uint64_t *expand_ns,
    std::uint32_t *expansion_status,
    c4::DeviceResidentContext *resident)
{
  if (!h2d_ns || !expand_ns || !expansion_status || !resident) {
    throw std::runtime_error(
        "IMPLANT.1/.2 secondary callback state");
  }
  ExpandedContacts expanded =
      expand_contact_edges_resident(secondary);
  *h2d_ns = expanded.h2d_ns;
  *expand_ns = expanded.expand_ns;
  *expansion_status |= expanded.status;
  if (expanded.status) {
    throw std::runtime_error(
        "IMPLANT.1/.2 secondary expansion failed");
  }

  c4::DeviceRequest &device_request = resident->request;
  device_request.contacts.device_edges =
      thrust::raw_pointer_cast(expanded.edges.data());
  device_request.contacts.count = secondary.flat_edge_count;
  device_request.contacts.bounds = c4::ContactBounds{
      secondary.scene_left, secondary.scene_bottom,
      secondary.scene_right, secondary.scene_top};
  device_request.distance = distance;
  device_request.grid_cell_size = outer.grid_cell_size;
  device_request.device = outer.device;
  device_request.contact_direction_contract =
      c4::ContactDirectionContract::
          validated_implant12_secondary_material_on_right_contours;
  device_request.limits.max_contact_edges = max_edges;
  device_request.limits.max_grid_cells =
      outer.capacity.max_grid_cells;
  device_request.limits.max_memberships =
      outer.capacity.max_secondary_memberships;
  device_request.limits.max_boundary_cell_visits =
      max_cell_visits;
  device_request.limits.max_member_visits =
      outer.capacity.max_member_visits;
  device_request.limits.max_pair_work =
      outer.capacity.max_pair_work;
  device_request.limits.max_cells_per_contact_edge =
      outer.capacity.max_cells_per_secondary_edge;
  device_request.limits.max_cells_per_boundary_edge =
      outer.capacity.max_cells_per_boundary_edge;
  c4::consume_device_boundary_result_hook(
      stream, horizontal, horizontal_count, vertical,
      vertical_count, resident);
}

void consume_implant15_boundary(
    cudaStream_t stream,
    const mu::DirectedSegmentI64 *horizontal,
    std::uint64_t horizontal_count,
    const mu::DirectedSegmentI64 *vertical,
    std::uint64_t vertical_count, void *opaque)
{
  Implant15BoundaryContext *context =
      static_cast<Implant15BoundaryContext *>(opaque);
  if (!context || context->invoked || !context->gate ||
      !context->contact || !context->outer) {
    throw std::runtime_error(
        "IMPLANT.1-.5 boundary callback contract");
  }
  context->invoked = true;
  if (stream != nullptr) {
    throw std::runtime_error(
        "IMPLANT.1-.5 boundary callback requires default stream");
  }
  consume_implant15_secondary(
      stream, horizontal, horizontal_count, vertical,
      vertical_count, *context->gate,
      context->outer->implant1_distance,
      context->outer->capacity.max_gate_edges,
      context->outer->capacity.max_gate_boundary_cell_visits,
      *context->outer, &context->gate_h2d_ns,
      &context->gate_expand_ns, &context->expansion_status,
      &context->implant1);
  consume_implant15_secondary(
      stream, horizontal, horizontal_count, vertical,
      vertical_count, *context->contact,
      context->outer->implant2_distance,
      context->outer->capacity.max_contact_edges,
      context->outer->capacity.max_contact_boundary_cell_visits,
      *context->outer, &context->contact_h2d_ns,
      &context->contact_expand_ns,
      &context->expansion_status, &context->implant2);
}

mu::ResidentBoundaryHook make_implant15_boundary_hook(
    Implant15BoundaryContext *context)
{
  if (!context || context->invoked) {
    throw std::runtime_error(
        "invalid IMPLANT.1-.5 boundary hook context");
  }
  mu::ResidentBoundaryHook hook;
  hook.consume = &consume_implant15_boundary;
  hook.context = context;
  hook.stop_before_d2h = true;
  return hook;
}

void echo_contact4_scene(
    const Contact4Scene &source, Contact4SceneEcho *echo)
{
  std::memset(echo, 0, sizeof(*echo));
  echo->struct_size = sizeof(*echo);
  echo->role = source.role;
  echo->format_version = source.format_version;
  echo->dbu_per_micron = source.dbu_per_micron;
  echo->root_cell = source.root_cell;
  echo->layer = source.layer;
  echo->datatype = source.datatype;
  echo->context_count = source.context_count;
  echo->layer_context_count = source.layer_context_count;
  echo->context_polygon_offset_count =
      source.context_polygon_offset_count;
  echo->context_edge_offset_count =
      source.context_edge_offset_count;
  echo->cell_count = source.cell_count;
  echo->polygon_count = source.polygon_count;
  echo->edge_count = source.edge_count;
  echo->flat_polygon_count = source.flat_polygon_count;
  echo->flat_edge_count = source.flat_edge_count;
  echo->scene_left = source.scene_left;
  echo->scene_bottom = source.scene_bottom;
  echo->scene_right = source.scene_right;
  echo->scene_top = source.scene_top;
  std::copy(
      source.digest_domain,
      source.digest_domain +
          KLAYOUT_CUDA_SPATIAL_CONTACT4_DIGEST_DOMAIN_BYTES,
      echo->digest_domain);
  std::copy(
      source.scene_digest, source.scene_digest + 32,
      echo->scene_digest);
}

void echo_implant15_request(
    const Implant15Request &request, Implant15Result *result)
{
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->format_version = request.format_version;
  result->dbu_per_micron = request.dbu_per_micron;
  result->requested_mask = request.requested_mask;
  result->device = request.device;
  result->implant1_distance = request.implant1_distance;
  result->implant2_distance = request.implant2_distance;
  result->implant3_distance = request.implant3_distance;
  result->implant4_distance = request.implant4_distance;
  result->grid_cell_size = request.grid_cell_size;
  echo_contact4_scene(request.nplus, &result->nplus);
  echo_contact4_scene(request.pplus, &result->pplus);
  echo_contact4_scene(request.gate, &result->gate);
  echo_contact4_scene(request.contact, &result->contact);
  result->capacity = request.capacity;
}

void echo_contact4_request(
    const Contact4Request &request, Contact4Result *result)
{
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->format_version = request.format_version;
  result->dbu_per_micron = request.dbu_per_micron;
  result->device = request.device;
  result->distance = request.distance;
  result->grid_cell_size = request.grid_cell_size;
  echo_contact4_scene(request.active, &result->active);
  echo_contact4_scene(request.contact, &result->contact);
}

void copy_contact4_pipeline_result(
    const mu::GpuUnionOutput &output,
    const Contact4CallbackContext &callback,
    Contact4Result *result)
{
  const c4::Result &contact = callback.resident.result;
  result->rectangle_count = output.rectangle_count;
  result->x_slab_count = output.x_slabs;
  result->union_membership_count = output.memberships;
  result->event_count = output.event_count;
  result->strip_interval_count = output.strip_intervals;
  result->raw_segment_count = output.raw_segments;
  result->boundary_segment_count = contact.boundary_segments;
  result->contact_expanded_edge_count = contact.contact_edges;
  result->grid_cell_count = contact.grid_cells;
  result->contact_membership_count = contact.memberships;
  result->boundary_cell_visit_count =
      contact.boundary_cell_visits;
  result->member_visit_count = contact.member_visits;
  result->candidate_pair_count = contact.candidate_pairs;
  result->raw_hit_count = contact.hits;
  result->uncertain_count = contact.uncertain;
  result->device_flags = contact.device_flags;
  result->device_total_bytes = std::max(
      output.device_total_bytes, contact.device_total_bytes);
  result->union_free_begin_bytes =
      output.device_free_begin_bytes;
  result->union_free_low_bytes = output.device_free_low_bytes;
  result->callback_free_begin_bytes =
      contact.callback_free_begin_bytes;
  result->callback_free_low_bytes =
      contact.callback_free_low_bytes;
  result->post_scan_free_bytes = contact.post_scan_free_bytes;
  result->callback_incremental_peak_bytes =
      contact.callback_incremental_peak_bytes;
  result->x_membership_ns =
      milliseconds_to_ns(output.x_membership_ms);
  result->strip_scan_ns =
      milliseconds_to_ns(output.strip_scan_ms);
  result->boundary_ns =
      milliseconds_to_ns(output.boundary_ms);
  result->contact_h2d_ns = callback.contact_h2d_ns;
  result->contact_expand_ns = callback.contact_expand_ns;
  result->boundary_preflight_ns =
      milliseconds_to_ns(contact.boundary_preflight_ms);
  result->grid_count_ns =
      milliseconds_to_ns(contact.grid_count_ms);
  result->grid_build_ns =
      milliseconds_to_ns(contact.grid_build_ms);
  result->query_ns = milliseconds_to_ns(contact.query_ms);
  result->d2h_ns =
      milliseconds_to_ns(output.d2h_ms + contact.d2h_ms);
}

void echo_active3_well_request(
    const Active3WellRequest &request,
    Active3WellResult *result)
{
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->format_version = request.format_version;
  result->dbu_per_micron = request.dbu_per_micron;
  result->device = request.device;
  result->distance = request.distance;
  result->grid_cell_size = request.grid_cell_size;
  result->secondary_well_layer =
      request.secondary_well_layer;
  result->secondary_well_datatype =
      request.secondary_well_datatype;
  echo_contact4_scene(request.wells, &result->wells);
  echo_contact4_scene(request.active, &result->active);
}

void copy_active3_well_pipeline_result(
    const mu::GpuUnionOutput &output,
    const Active3WellCallbackContext &callback,
    Active3WellResult *result)
{
  const c4::Result &active = callback.resident.result;
  result->rectangle_count = output.rectangle_count;
  result->x_slab_count = output.x_slabs;
  result->union_membership_count = output.memberships;
  result->event_count = output.event_count;
  result->strip_interval_count = output.strip_intervals;
  result->raw_segment_count = output.raw_segments;
  result->boundary_segment_count = active.boundary_segments;
  result->active_expanded_edge_count = active.contact_edges;
  result->grid_cell_count = active.grid_cells;
  result->active_membership_count = active.memberships;
  result->active_cell_visit_count =
      active.boundary_cell_visits;
  result->member_visit_count = active.member_visits;
  result->candidate_pair_count = active.candidate_pairs;
  result->raw_hit_count = active.hits;
  result->uncertain_count = active.uncertain;
  result->device_flags = active.device_flags;
  result->device_total_bytes = std::max(
      output.device_total_bytes, active.device_total_bytes);
  result->union_free_begin_bytes =
      output.device_free_begin_bytes;
  result->union_free_low_bytes = output.device_free_low_bytes;
  result->callback_free_begin_bytes =
      active.callback_free_begin_bytes;
  result->callback_free_low_bytes =
      active.callback_free_low_bytes;
  result->post_scan_free_bytes = active.post_scan_free_bytes;
  result->callback_incremental_peak_bytes =
      active.callback_incremental_peak_bytes;
  result->x_membership_ns =
      milliseconds_to_ns(output.x_membership_ms);
  result->strip_scan_ns =
      milliseconds_to_ns(output.strip_scan_ms);
  result->boundary_ns =
      milliseconds_to_ns(output.boundary_ms);
  result->active_h2d_ns = callback.active_h2d_ns;
  result->active_expand_ns = callback.active_expand_ns;
  result->active_preflight_ns =
      milliseconds_to_ns(active.boundary_preflight_ms);
  result->grid_count_ns =
      milliseconds_to_ns(active.grid_count_ms);
  result->grid_build_ns =
      milliseconds_to_ns(active.grid_build_ms);
  result->query_ns = milliseconds_to_ns(active.query_ms);
  result->d2h_ns =
      milliseconds_to_ns(output.d2h_ms + active.d2h_ms);
}

int run_contact4_active_union_request(
    const Contact4Request *request, Contact4Result *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_UNCERTAIN;
  if (!request || !valid_contact4_request(*request)) {
    set_message(
        result,
        "unsupported or malformed CONTACT4 ACTIVE-union request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  echo_contact4_request(*request, result);

  const auto total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> lock(pipeline_mutex());
    const auto setup_begin = Clock::now();
    const Request active =
        scene_as_union_request(request->active, *request);
    Request contact =
        scene_as_union_request(request->contact, *request);
    // CONTACT is structurally/digest validated but never decomposed into the
    // ACTIVE union's rectangle stream.  Give that validation its independent
    // exact upper bound (one rectangle cannot require more source edges than
    // the whole flattened contour stream) instead of coupling it to the
    // operational ACTIVE rectangle cap.
    contact.max_rectangles = contact.flat_edge_count;
    validate_shared_contact4_hierarchy(active, contact);
    const LoweredScene active_lowered =
        validate_and_lower(active, kActiveRawDigestMagic);
    validate_without_lowering(contact, kContactRawDigestMagic);
    result->setup_ns = elapsed_ns(setup_begin, Clock::now());
    klayout_cuda::DevicePhaseLease device_lease(
        request->device, "contact4_active_union");

    ExpandedRectangles expanded_active =
        expand_rectangles_resident(active, active_lowered);
    result->active_h2d_ns = expanded_active.h2d_ns;
    result->active_expand_ns = expanded_active.expand_ns;
    if (expanded_active.status) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          expanded_active.status & kExpandTransformOverflow
              ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
              : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      set_message(
          result,
          "raw ACTIVE device expansion failed its exact gate");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    mu::GpuUnionLimits limits;
    limits.max_rectangles = request->max_rectangles;
    limits.max_x_slabs = request->max_x_slabs;
    limits.max_memberships = request->max_union_memberships;
    limits.max_events = request->max_events;
    limits.max_raw_segments = request->max_raw_segments;
    limits.max_segments = request->max_boundary_segments;
    limits.max_slabs_per_rectangle =
        request->max_slabs_per_rectangle;

    Contact4CallbackContext callback;
    callback.contact = &contact;
    callback.outer = request;
    const mu::ResidentBoundaryHook hook =
        make_contact4_active_union_hook(&callback);
    const double input_prepare_ms =
        static_cast<double>(
            result->active_h2d_ns + result->active_expand_ns) /
        1000000.0;
    const mu::GpuUnionOutput output = mu::gpu_union_resident(
        std::move(expanded_active.rectangles),
        active.scene_bottom, active.scene_top, limits,
        request->device, input_prepare_ms, nullptr, &hook);
    copy_contact4_pipeline_result(output, callback, result);

    if (output.fallback) {
      if (callback.resident.result.hits &&
          !callback.resident.result.uncertain &&
          !callback.resident.result.device_flags) {
        result->status = KLAYOUT_CUDA_SPATIAL_OK;
        result->fallback_flags =
            KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
        result->disposition =
            KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_RAW_HITS;
        set_message(
            result,
            "exact ACTIVE union has raw CONTACT4 hits");
        result->total_ns = elapsed_ns(total_begin, Clock::now());
        return KLAYOUT_CUDA_SPATIAL_OK;
      }
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          callback.expansion_status & kExpandTransformOverflow
              ? static_cast<std::uint32_t>(
                    KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW)
              : fallback_flags_for_union(output);
      set_message(result, output.message.c_str());
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    const c4::Result &contact_result =
        callback.resident.result;
    if (!callback.invoked || !callback.resident.invoked ||
        !output.resident_boundary_consumer_completed ||
        !output.segments.empty() || output.d2h_ms != 0.0 ||
        !contact_result.certified_empty ||
        contact_result.contact_edges !=
            request->contact.flat_edge_count ||
        !contact_result.boundary_segments ||
        contact_result.hits || contact_result.uncertain ||
        contact_result.device_flags ||
        contact_result.grid_cells > request->max_grid_cells ||
        contact_result.memberships >
            request->max_contact_memberships ||
        contact_result.boundary_cell_visits >
            request->max_boundary_cell_visits ||
        contact_result.member_visits >
            request->max_member_visits ||
        contact_result.candidate_pairs >
            request->max_pair_work) {
      throw std::runtime_error(
          "CONTACT4 fused completion invariant failed");
    }

    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE;
    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    set_message(
        result,
        "complete resident ACTIVE-union CONTACT4 empty certificate");
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return KLAYOUT_CUDA_SPATIAL_OK;
  } catch (const M2Decline &decline) {
    result->fallback_flags = decline.fallback_flags();
    result->status =
        decline.kind() == DeclineKind::bad_argument
            ? KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT
            : KLAYOUT_CUDA_SPATIAL_FALLBACK;
    set_message(result, decline.what());
  } catch (const std::exception &error) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    set_message(result, error.what());
  } catch (...) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    set_message(
        result,
        "unknown resident ACTIVE-union CONTACT4 exception");
  }
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_UNCERTAIN;
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return result->status;
}

int run_active3_well_union_request(
    const Active3WellRequest *request,
    Active3WellResult *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_UNCERTAIN;
  if (!request || !valid_active3_well_request(*request)) {
    set_message(
        result,
        "unsupported or malformed ACTIVE3 WELL-union request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  echo_active3_well_request(*request, result);

  const auto total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> lock(pipeline_mutex());
    const auto setup_begin = Clock::now();
    const Request wells =
        active3_scene_as_union_request(
            request->wells, *request);
    Request active =
        active3_scene_as_union_request(
            request->active, *request);
    // ACTIVE is digest/topology validated but is expanded directly as
    // directed edges in the resident callback, not decomposed into union
    // rectangles.
    active.max_rectangles = active.flat_edge_count;
    validate_shared_active3_hierarchy(wells, active);
    const LoweredScene wells_lowered =
        validate_and_lower(wells, kWellUnionRawDigestMagic);
    validate_without_lowering(active, kActiveRawDigestMagic);
    result->setup_ns = elapsed_ns(setup_begin, Clock::now());

    ExpandedRectangles expanded_wells =
        expand_rectangles_resident(wells, wells_lowered);
    result->wells_h2d_ns = expanded_wells.h2d_ns;
    result->wells_expand_ns = expanded_wells.expand_ns;
    if (expanded_wells.status) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          expanded_wells.status & kExpandTransformOverflow
              ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
              : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      set_message(
          result,
          "combined raw WELL device expansion failed its exact gate");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    mu::GpuUnionLimits limits;
    limits.max_rectangles = request->max_rectangles;
    limits.max_x_slabs = request->max_x_slabs;
    limits.max_memberships = request->max_union_memberships;
    limits.max_events = request->max_events;
    limits.max_raw_segments = request->max_raw_segments;
    limits.max_segments = request->max_boundary_segments;
    limits.max_slabs_per_rectangle =
        request->max_slabs_per_rectangle;

    Active3WellCallbackContext callback;
    callback.active = &active;
    callback.outer = request;
    const mu::ResidentBoundaryHook hook =
        make_active3_well_union_hook(&callback);
    const double input_prepare_ms =
        static_cast<double>(
            result->wells_h2d_ns + result->wells_expand_ns) /
        1000000.0;
    const mu::GpuUnionOutput output = mu::gpu_union_resident(
        std::move(expanded_wells.rectangles),
        wells.scene_bottom, wells.scene_top, limits,
        request->device, input_prepare_ms, nullptr, &hook);
    copy_active3_well_pipeline_result(
        output, callback, result);

    if (output.fallback) {
      if (callback.resident.result.hits &&
          !callback.resident.result.uncertain &&
          !callback.resident.result.device_flags) {
        result->status = KLAYOUT_CUDA_SPATIAL_OK;
        result->fallback_flags =
            KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
        result->disposition =
            KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_RAW_HITS;
        set_message(
            result,
            "exact WELL union has ACTIVE3 hits");
        result->total_ns = elapsed_ns(total_begin, Clock::now());
        return KLAYOUT_CUDA_SPATIAL_OK;
      }
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          callback.expansion_status & kExpandTransformOverflow
              ? static_cast<std::uint32_t>(
                    KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW)
              : fallback_flags_for_union(output);
      set_message(result, output.message.c_str());
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    const c4::Result &active_result =
        callback.resident.result;
    if (!callback.invoked || !callback.resident.invoked ||
        !output.resident_boundary_consumer_completed ||
        !output.segments.empty() || output.d2h_ms != 0.0 ||
        !active_result.certified_empty ||
        active_result.contact_edges !=
            request->active.flat_edge_count ||
        !active_result.boundary_segments ||
        active_result.hits || active_result.uncertain ||
        active_result.device_flags ||
        active_result.grid_cells > request->max_grid_cells ||
        active_result.memberships >
            request->max_active_memberships ||
        active_result.boundary_cell_visits >
            request->max_active_cell_visits ||
        active_result.member_visits >
            request->max_member_visits ||
        active_result.candidate_pairs >
            request->max_pair_work) {
      throw std::runtime_error(
          "ACTIVE3 WELL-union completion invariant failed");
    }

    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_COMPLETE;
    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    set_message(
        result,
        "complete resident exact-WELL-union ACTIVE3 empty certificate");
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return KLAYOUT_CUDA_SPATIAL_OK;
  } catch (const M2Decline &decline) {
    result->fallback_flags = decline.fallback_flags();
    result->status =
        decline.kind() == DeclineKind::bad_argument
            ? KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT
            : KLAYOUT_CUDA_SPATIAL_FALLBACK;
    set_message(result, decline.what());
  } catch (const std::exception &error) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    set_message(result, error.what());
  } catch (...) {
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    set_message(
        result,
        "unknown resident WELL-union ACTIVE3 exception");
  }
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_UNCERTAIN;
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return result->status;
}

void copy_implant15_boundary_result(
    const c4::DeviceResidentContext &source, bool implant1,
    Implant15Result *result)
{
  if (!source.invoked) return;
  const c4::Result &relation = source.result;
  if (implant1) {
    result->gate_expanded_edge_count = relation.contact_edges;
    result->implant1_grid_cell_count = relation.grid_cells;
    result->implant1_secondary_membership_count =
        relation.memberships;
    result->implant1_boundary_cell_visit_count =
        relation.boundary_cell_visits;
    result->implant1_member_visit_count =
        relation.member_visits;
    result->implant1_candidate_count =
        relation.candidate_pairs;
    result->implant1_hit_count = relation.hits;
    result->implant1_uncertain_count = relation.uncertain;
    result->implant1_ns =
        milliseconds_to_ns(relation.total_ms);
  } else {
    result->contact_expanded_edge_count = relation.contact_edges;
    result->implant2_grid_cell_count = relation.grid_cells;
    result->implant2_secondary_membership_count =
        relation.memberships;
    result->implant2_boundary_cell_visit_count =
        relation.boundary_cell_visits;
    result->implant2_member_visit_count =
        relation.member_visits;
    result->implant2_candidate_count =
        relation.candidate_pairs;
    result->implant2_hit_count = relation.hits;
    result->implant2_uncertain_count = relation.uncertain;
    result->implant2_ns =
        milliseconds_to_ns(relation.total_ms);
  }
}

void copy_implant15_width_space_result(
    const m2m::BaseWidthSpaceResult &original,
    const m2m::BaseWidthSpaceResult &transposed,
    const Implant15Request &request, Implant15Result *result)
{
  const auto add = [](std::uint64_t first, std::uint64_t second,
                      const char *label) {
    std::uint64_t value = 0;
    if (!checked_add_u64(first, second, &value)) {
      throw std::runtime_error(
          std::string("IMPLANT morphology ") + label +
          " overflows uint64");
    }
    return value;
  };
  const std::uint64_t interval_work =
      add(original.intervals_checked,
          transposed.intervals_checked, "width census");
  const std::uint64_t gaps =
      add(original.gaps_checked, transposed.gaps_checked,
          "gap census");
  const std::uint64_t corner_work =
      add(original.corner_pair_work,
          transposed.corner_pair_work, "corner-work census");
  const std::uint64_t corner_hits =
      add(original.corner_candidates,
          transposed.corner_candidates,
          "corner-hit census");
  /*
   * The exact corner endpoint classifier is deliberately conservative across
   * the width/space attribution boundary.  It proves that the combined
   * morphology certificate cannot be consumed, but it has intentionally lost
   * the polygon identity required to attribute the candidate honestly to
   * IMPLANT.3 or IMPLANT.4.  Charge the work and uncertainty to both rules;
   * direct strip width and gap violations remain rule-specific hits.
   */
  result->implant3_candidate_count =
      add(interval_work, corner_work, "width/corner census");
  result->implant3_hit_count =
      add(original.width_violations,
          transposed.width_violations,
          "width-hit census");
  result->implant4_candidate_count =
      add(gaps, corner_work, "spacing census");
  result->implant4_hit_count =
      add(original.space_violations,
          transposed.space_violations,
          "parallel-space-hit census");
  result->implant3_uncertain_count =
      add(
          corner_hits,
          original.device_flags || transposed.device_flags,
          "width uncertainty census");
  result->implant4_uncertain_count =
      add(
          corner_hits,
          original.device_flags || transposed.device_flags,
          "space uncertainty census");
  if (result->implant3_candidate_count >
          request.capacity.max_morphology_work ||
      result->implant4_candidate_count >
          request.capacity.max_morphology_work) {
    capacity("IMPLANT.3/.4 morphology-work capacity");
  }
  /*
   * Width and direct-gap spacing intentionally share one strip scan in each
   * orientation.  Charge that fused scan once to IMPLANT.3; IMPLANT.4 is the
   * zero-additional-latency sibling certificate from the same operator.
   */
  result->implant3_ns = milliseconds_to_ns(
      original.elapsed_ms + transposed.elapsed_ms);
  result->implant4_ns = 0;
}

mu::GpuUnionLimits implant15_union_limits(
    const Implant15Request &request)
{
  mu::GpuUnionLimits limits;
  limits.max_rectangles = request.capacity.max_rectangles;
  limits.max_x_slabs = request.capacity.max_x_slabs;
  limits.max_memberships =
      request.capacity.max_union_memberships;
  limits.max_events = request.capacity.max_events;
  limits.max_raw_segments =
      request.capacity.max_raw_segments;
  limits.max_segments =
      request.capacity.max_boundary_segments;
  limits.max_slabs_per_rectangle =
      request.capacity.max_slabs_per_rectangle;
  return limits;
}

mu::GpuUnionStripWindowLimits implant15_window_limits(
    const Implant15Request &request)
{
  mu::GpuUnionStripWindowLimits limits;
  limits.max_window_events = std::min(
      limits.max_window_events, request.capacity.max_events);
  limits.max_strip_intervals = std::min(
      {request.capacity.max_union_memberships,
       request.capacity.max_raw_segments / 2,
       static_cast<std::uint64_t>(UINT32_MAX)});
  limits.max_windows = std::min(
      limits.max_windows, request.capacity.max_x_slabs);
  limits.max_window_slabs =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          limits.max_window_slabs,
          request.capacity.max_x_slabs));
  return limits;
}

thrust::device_vector<mu::RectI64>
implant15_concatenate_rectangles(
    const thrust::device_vector<mu::RectI64> &nplus,
    const thrust::device_vector<mu::RectI64> &pplus)
{
  if (pplus.size() >
      std::numeric_limits<std::size_t>::max() - nplus.size()) {
    capacity("IMPLANT rectangle concatenation exceeds size_t");
  }
  thrust::device_vector<mu::RectI64> combined(
      nplus.size() + pplus.size());
  if (!nplus.empty()) {
    cuda_require(
        cudaMemcpy(
            thrust::raw_pointer_cast(combined.data()),
            thrust::raw_pointer_cast(nplus.data()),
            nplus.size() * sizeof(mu::RectI64),
            cudaMemcpyDeviceToDevice),
        "IMPLANT NPLUS rectangle concatenation D2D");
  }
  if (!pplus.empty()) {
    cuda_require(
        cudaMemcpy(
            thrust::raw_pointer_cast(combined.data()) + nplus.size(),
            thrust::raw_pointer_cast(pplus.data()),
            pplus.size() * sizeof(mu::RectI64),
            cudaMemcpyDeviceToDevice),
        "IMPLANT PPLUS rectangle concatenation D2D");
  }
  cuda_require(
      cudaDeviceSynchronize(),
      "IMPLANT rectangle concatenation synchronize");
  return combined;
}

std::uint64_t transpose_implant15_rectangles(
    thrust::device_vector<mu::RectI64> *rectangles,
    std::int64_t left, std::int64_t bottom,
    std::int64_t right, std::int64_t top)
{
  if (!rectangles || rectangles->empty()) {
    throw std::runtime_error(
        "invalid resident IMPLANT transpose input");
  }
  const auto begin = Clock::now();
  thrust::device_vector<std::uint32_t> status(1, 0);
  transpose_m1_rectangles_kernel<<<
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          (rectangles->size() + kExpandThreads - 1) /
              kExpandThreads,
          kMaximumBlocks)),
      kExpandThreads>>>(
      thrust::raw_pointer_cast(rectangles->data()),
      rectangles->size(), left, bottom, right, top,
      thrust::raw_pointer_cast(status.data()));
  cuda_require(
      cudaGetLastError(),
      "IMPLANT resident transpose launch");
  cuda_require(
      cudaDeviceSynchronize(),
      "IMPLANT resident transpose synchronize");
  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(
          &host_status, thrust::raw_pointer_cast(status.data()),
          sizeof(host_status), cudaMemcpyDeviceToHost),
      "IMPLANT transpose status D2H");
  if (host_status) {
    coordinate_decline(
        "IMPLANT resident transpose failed exact bounds gate");
  }
  return elapsed_ns(begin, Clock::now());
}

int run_implant15_request(
    const Implant15Request *request, Implant15Result *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN;
  if (!request || !valid_implant15_request(*request)) {
    set_message(
        result,
        "unsupported or malformed raw IMPLANT.1-.5 request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  echo_implant15_request(*request, result);
  const auto total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> lock(pipeline_mutex());
    const auto setup_begin = Clock::now();
    const Request nplus =
        implant15_scene_as_union_request(
            request->nplus, *request);
    const Request pplus =
        implant15_scene_as_union_request(
            request->pplus, *request);
    Request gate =
        implant15_scene_as_union_request(
            request->gate, *request);
    Request contact =
        implant15_scene_as_union_request(
            request->contact, *request);
    gate.max_rectangles = gate.flat_edge_count;
    contact.max_rectangles = contact.flat_edge_count;
    validate_shared_contact4_hierarchy(nplus, pplus);
    validate_shared_contact4_hierarchy(nplus, gate);
    validate_shared_contact4_hierarchy(nplus, contact);
    const LoweredScene nplus_lowered =
        validate_and_lower(nplus, kNplusRawDigestMagic);
    const LoweredScene pplus_lowered =
        validate_and_lower(pplus, kPplusRawDigestMagic);
    validate_without_lowering(gate, kGateRawDigestMagic);
    validate_without_lowering(contact, kContactRawDigestMagic);
    result->setup_ns = elapsed_ns(setup_begin, Clock::now());
    result->nplus_rectangle_count =
        nplus_lowered.flat_rectangles;
    result->pplus_rectangle_count =
        pplus_lowered.flat_rectangles;
    if (!checked_add_u64(
            result->nplus_rectangle_count,
            result->pplus_rectangle_count,
            &result->implant_rectangle_count) ||
        result->implant_rectangle_count >
            request->capacity.max_rectangles) {
      capacity("combined raw IMPLANT rectangles exceed capacity");
    }
    klayout_cuda::DevicePhaseLease device_lease(
        request->device, "implant15");

    ExpandedRectangles expanded_nplus =
        expand_rectangles_resident(nplus, nplus_lowered);
    ExpandedRectangles expanded_pplus =
        expand_rectangles_resident(pplus, pplus_lowered);
    result->nplus_h2d_ns = expanded_nplus.h2d_ns;
    result->nplus_expand_ns = expanded_nplus.expand_ns;
    result->pplus_h2d_ns = expanded_pplus.h2d_ns;
    result->pplus_expand_ns = expanded_pplus.expand_ns;
    if (expanded_nplus.status || expanded_pplus.status) {
      result->device_flags =
          expanded_nplus.status | expanded_pplus.status;
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          result->device_flags & kExpandTransformOverflow
              ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
              : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      set_message(
          result,
          "raw NPLUS/PPLUS device expansion failed exact gate");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }

    const Implant15OverlapResult overlap =
        run_implant15_overlap(
            expanded_nplus.rectangles,
            expanded_pplus.rectangles, *request);
    result->implant5_membership_count = overlap.memberships;
    result->implant5_candidate_count =
        overlap.candidate_visits;
    result->implant5_hit_count = overlap.hits;
    result->implant5_uncertain_count =
        overlap.device_flags != 0;
    result->implant5_ns = overlap.elapsed_ns;
    result->device_flags |= overlap.device_flags;
    if (overlap.hits || overlap.device_flags) {
      result->status = overlap.hits
          ? KLAYOUT_CUDA_SPATIAL_OK
          : KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          overlap.device_flags
              ? KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT
              : KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
      result->disposition = overlap.hits
          ? KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_HITS
          : KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN;
      set_message(
          result,
          overlap.hits
              ? "strict positive-area NPLUS/PPLUS overlap requires CPU fallback"
              : "IMPLANT.5 device invariant declined");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }
    result->clean_mask |= KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_5;

    const std::int64_t implant_left =
        std::min(nplus.scene_left, pplus.scene_left);
    const std::int64_t implant_bottom =
        std::min(nplus.scene_bottom, pplus.scene_bottom);
    const std::int64_t implant_right =
        std::max(nplus.scene_right, pplus.scene_right);
    const std::int64_t implant_top =
        std::max(nplus.scene_top, pplus.scene_top);
    const __int128 x_range =
        static_cast<__int128>(implant_right) - implant_left;
    const __int128 y_range =
        static_cast<__int128>(implant_top) - implant_bottom;
    if (x_range <= 0 || y_range <= 0 ||
        x_range > UINT32_MAX || y_range > UINT32_MAX) {
      coordinate_decline(
          "combined raw IMPLANT range exceeds packed union capacity");
    }
    const mu::GpuUnionLimits union_limits =
        implant15_union_limits(*request);

    // Y-oriented pass first.  It consumes only the temporary transposed copy.
    thrust::device_vector<mu::RectI64> transposed =
        implant15_concatenate_rectangles(
            expanded_nplus.rectangles,
            expanded_pplus.rectangles);
    const std::uint64_t transpose_ns =
        transpose_implant15_rectangles(
            &transposed, implant_left, implant_bottom,
            implant_right, implant_top);
    m2m::BaseWidthSpaceContext transposed_certificate;
    transposed_certificate.profile =
        m2m::BaseWidthSpaceProfile::implant_90;
    transposed_certificate.distance =
        request->implant3_distance;
    transposed_certificate.origin_x = implant_bottom;
    transposed_certificate.origin_y = implant_left;
    transposed_certificate.max_corner_endpoints =
        request->capacity.max_boundary_segments;
    transposed_certificate.max_corner_pair_work =
        request->capacity.max_morphology_work;
    const mu::ResidentStripHook transposed_hook =
        m2m::make_base_width_space_hook(
            &transposed_certificate, true);
    const mu::GpuUnionOutput transposed_output =
        mu::gpu_union_resident_windowed_strips(
            std::move(transposed), implant_left, implant_right,
            union_limits, implant15_window_limits(*request),
            request->device,
            static_cast<double>(transpose_ns) / 1000000.0,
            &transposed_hook);
    if (transposed_output.fallback ||
        !transposed_certificate.invoked) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          fallback_flags_for_union(transposed_output);
      set_message(
          result,
          ("transposed raw IMPLANT union: " +
           transposed_output.message).c_str());
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }

    // Original orientation continues from exact strips into one canonical
    // boundary, then two separately indexed projection-relation operators.
    thrust::device_vector<mu::RectI64> original =
        implant15_concatenate_rectangles(
            expanded_nplus.rectangles,
            expanded_pplus.rectangles);
    {
      // Destroy the source allocations now.  Named empty vectors at function
      // scope would retain both buffers through the original union and its
      // resident GATE/CONTACT consumers.
      thrust::device_vector<mu::RectI64> empty_nplus;
      expanded_nplus.rectangles.swap(empty_nplus);
      thrust::device_vector<mu::RectI64> empty_pplus;
      expanded_pplus.rectangles.swap(empty_pplus);
    }
    m2m::BaseWidthSpaceContext original_certificate;
    original_certificate.profile =
        m2m::BaseWidthSpaceProfile::implant_90;
    original_certificate.distance = request->implant3_distance;
    original_certificate.origin_x = implant_left;
    original_certificate.origin_y = implant_bottom;
    original_certificate.max_corner_endpoints =
        request->capacity.max_boundary_segments;
    original_certificate.max_corner_pair_work =
        request->capacity.max_morphology_work;
    const mu::ResidentStripHook original_hook =
        m2m::make_base_width_space_hook(
            &original_certificate, false);
    Implant15BoundaryContext boundary;
    boundary.gate = &gate;
    boundary.contact = &contact;
    boundary.outer = request;
    const mu::ResidentBoundaryHook boundary_hook =
        make_implant15_boundary_hook(&boundary);
    const mu::GpuUnionOutput original_output =
        mu::gpu_union_resident(
            std::move(original), implant_bottom, implant_top,
            union_limits, request->device, 0.0,
            &original_hook, &boundary_hook);
    result->x_slab_count = std::max(
        original_output.x_slabs, transposed_output.x_slabs);
    result->union_membership_count = std::max(
        original_output.memberships,
        transposed_output.memberships);
    result->event_count = std::max(
        original_output.event_count,
        transposed_output.event_count);
    result->strip_interval_count = std::max(
        original_output.strip_intervals,
        transposed_output.strip_intervals);
    result->raw_segment_count = original_output.raw_segments;
    result->boundary_segment_count =
        boundary.implant1.invoked
            ? boundary.implant1.result.boundary_segments
            : boundary.implant2.invoked
                  ? boundary.implant2.result.boundary_segments
                  : 0;
    result->implant_union_ns = milliseconds_to_ns(
        original_output.x_membership_ms +
        original_output.strip_scan_ms +
        transposed_output.x_membership_ms +
        transposed_output.strip_scan_ms);
    result->implant_boundary_ns =
        milliseconds_to_ns(original_output.boundary_ms);
    result->gate_h2d_ns = boundary.gate_h2d_ns;
    result->gate_expand_ns = boundary.gate_expand_ns;
    result->contact_h2d_ns = boundary.contact_h2d_ns;
    result->contact_expand_ns = boundary.contact_expand_ns;
    result->device_flags |= boundary.expansion_status;
    copy_implant15_boundary_result(
        boundary.implant1, true, result);
    copy_implant15_boundary_result(
        boundary.implant2, false, result);
    copy_implant15_width_space_result(
        original_certificate.result,
        transposed_certificate.result, *request, result);

    const bool raw_hits =
        result->implant1_hit_count ||
        result->implant2_hit_count ||
        result->implant3_hit_count ||
        result->implant4_hit_count;
    const bool uncertain =
        result->device_flags ||
        result->implant1_uncertain_count ||
        result->implant2_uncertain_count ||
        result->implant3_uncertain_count ||
        result->implant4_uncertain_count;
    const bool pipeline_complete =
        !original_output.fallback &&
        original_output.resident_consumer_completed &&
        original_output.resident_boundary_consumer_completed &&
        original_output.segments.empty() &&
        original_output.d2h_ms == 0.0 &&
        original_certificate.invoked &&
        boundary.invoked &&
        boundary.implant1.invoked &&
        boundary.implant2.invoked;
    if (boundary.implant1.invoked &&
        !result->implant1_hit_count &&
        !result->implant1_uncertain_count) {
      result->clean_mask |=
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_1;
    }
    if (boundary.implant2.invoked &&
        !result->implant2_hit_count &&
        !result->implant2_uncertain_count) {
      result->clean_mask |=
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_2;
    }
    if (!result->implant3_hit_count &&
        !result->implant3_uncertain_count) {
      result->clean_mask |=
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_3;
    }
    if (!result->implant4_hit_count &&
        !result->implant4_uncertain_count) {
      result->clean_mask |=
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_RULE_4;
    }
    if (raw_hits && !uncertain && pipeline_complete) {
      result->status = KLAYOUT_CUDA_SPATIAL_OK;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_RAW_HITS;
      set_message(
          result,
          "raw IMPLANT.1-.5 hit requires pristine CPU fallback");
    } else if (uncertain || !pipeline_complete) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          original_output.fallback
              ? fallback_flags_for_union(original_output)
              : uncertain
                    ? static_cast<std::uint32_t>(
                          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST)
                    : static_cast<std::uint32_t>(
                          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT);
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN;
      set_message(
          result,
          "raw IMPLANT.1-.5 resident operator declined");
    } else {
      result->clean_mask =
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES;
      result->certified_empty_mask =
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_ALL_RULES;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_COMPLETE;
      result->status = KLAYOUT_CUDA_SPATIAL_OK;
      set_message(
          result,
          "complete chained resident IMPLANT.1-.5 empty certificate");
    }
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return result->status;
  } catch (const M2Decline &decline) {
    result->certified_empty_mask = 0;
    result->fallback_flags = decline.fallback_flags();
    result->status =
        decline.kind() == DeclineKind::bad_argument
            ? KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT
            : KLAYOUT_CUDA_SPATIAL_FALLBACK;
    set_message(result, decline.what());
  } catch (const std::exception &error) {
    result->certified_empty_mask = 0;
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    set_message(result, error.what());
  } catch (...) {
    result->certified_empty_mask = 0;
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    set_message(
        result,
        "unknown raw IMPLANT.1-.5 backend exception");
  }
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN;
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return result->status;
}

struct M1MorphCallbackContext
{
  m2m::Request request;
  bool invoked = false;
  bool complete = false;
  m2m::Result result;
  std::string error;
};

void consume_m1_morphology_strips(
    cudaStream_t stream, const std::int64_t *xs,
    std::uint32_t x_slabs,
    const mu::StripInterval *intervals,
    std::uint64_t interval_count,
    const std::uint64_t *slab_offsets,
    const std::uint32_t *slab_counts, void *opaque)
{
  M1MorphCallbackContext *context =
      static_cast<M1MorphCallbackContext *>(opaque);
  if (!context || context->invoked) {
    throw std::runtime_error(
        "M1 resident-morphology callback contract");
  }
  context->invoked = true;
  try {
    context->result = m2m::consume_f90_f270(
        stream,
        m2m::DeviceStripView{
            xs, x_slabs, intervals, interval_count,
            slab_offsets, slab_counts},
        context->request);
    context->complete = true;
  } catch (const std::exception &error) {
    context->error = error.what();
  } catch (...) {
    context->error =
        "unknown M1 resident-morphology callback exception";
  }
}

mu::ResidentStripHook make_m1_morphology_hook(
    M1MorphCallbackContext *context)
{
  if (!context) {
    throw std::runtime_error(
        "invalid M1 resident-morphology callback context");
  }
  mu::ResidentStripHook hook;
  hook.consume = &consume_m1_morphology_strips;
  hook.context = context;
  hook.stop_before_boundary = true;
  return hook;
}

bool morphology_capacity_error(const std::string &message)
{
  return message.find("capacity") != std::string::npos ||
         message.find("work cap") != std::string::npos ||
         message.find("exceeds") != std::string::npos;
}

bool morphology_known_not_empty(const std::string &message)
{
  return message.find(
             "F270 count-only erosion is not empty") !=
         std::string::npos;
}

void copy_m1_union_telemetry(
    const mu::GpuUnionOutput &output, M1MorphResult *result)
{
  result->rectangle_count = output.rectangle_count;
  result->x_slab_count = output.x_slabs;
  result->union_membership_count = output.memberships;
  result->union_event_count = output.event_count;
  result->strip_interval_count = output.strip_intervals;
  result->x_membership_ns =
      milliseconds_to_ns(output.x_membership_ms);
  result->strip_scan_ns =
      milliseconds_to_ns(output.strip_scan_ms);
  result->d2h_ns = milliseconds_to_ns(output.d2h_ms);
  result->union_device_total_bytes =
      output.device_total_bytes;
  result->union_device_free_begin_bytes =
      output.device_free_begin_bytes;
  result->union_device_free_low_bytes =
      output.device_free_low_bytes;
}

void copy_m1_morph_telemetry(
    const m2m::Result &morph, M1MorphResult *result)
{
  result->erode89_output_interval_count =
      morph.erode89.output_intervals;
  result->erode89_source_visit_count =
      morph.erode89.source_visits;
  result->dilate90_output_interval_count =
      morph.dilate90.output_intervals;
  result->dilate90_source_visit_count =
      morph.dilate90.source_visits;
  result->boundary_source_visit_count =
      morph.boundary.source_visits;
  result->erode269_source_visit_count =
      morph.erode269_count.source_visits;
  result->f90_boundary_segment_count =
      morph.f90_boundary_segments;
  result->f90_long_segment_count =
      morph.f90_long_segments;
  result->f90_space_pair_count =
      morph.f90_space_pairs_checked;
  result->f90_space_violation_count =
      morph.f90_space_violations;
  result->f90_space_uncertain_count =
      morph.f90_space_uncertain;
  result->f270_eroded_interval_count =
      morph.f270_eroded_intervals;
  result->morph_device_total_bytes =
      morph.device_total_bytes;
  result->morph_device_free_begin_bytes =
      morph.device_free_begin_bytes;
  result->morph_device_free_low_bytes =
      morph.device_free_low_bytes;
  result->morphology_ns =
      milliseconds_to_ns(morph.total_ms);
}

void validate_m1_morph_complete(
    const M1MorphRequest &request,
    const LoweredScene &lowered,
    const mu::GpuUnionOutput &output,
    const M1MorphCallbackContext &callback)
{
  const m2m::Result &morph = callback.result;
  std::uint64_t expected_events = 0;
  std::uint64_t expected_pairs = 0;
  if (!callback.invoked || !callback.complete ||
      !callback.error.empty() ||
      !output.resident_consumer_completed ||
      output.resident_boundary_consumer_completed ||
      !output.segments.empty() || output.raw_segments ||
      output.digest || output.boundary_ms != 0.0 ||
      output.d2h_ms != 0.0 ||
      output.rectangle_count != lowered.flat_rectangles ||
      output.rectangle_count < request.flat_polygon_count ||
      output.rectangle_count > request.max_rectangles ||
      !output.x_slabs ||
      output.x_slabs > request.max_x_slabs ||
      !output.memberships ||
      output.memberships > request.max_union_memberships ||
      !checked_multiply_u64(
          output.memberships, 2, &expected_events) ||
      output.event_count != expected_events ||
      output.event_count > request.max_union_events ||
      !output.strip_intervals ||
      output.strip_intervals > output.memberships ||
      morph.source_x_slabs != output.x_slabs ||
      morph.source_intervals != output.strip_intervals ||
      !morph.erode89.output_intervals ||
      morph.erode89.output_intervals >
          request.max_morph_output_intervals ||
      !morph.dilate90.output_intervals ||
      morph.dilate90.output_intervals >
          request.max_morph_output_intervals ||
      morph.erode89.source_visits >
          request.max_morph_source_visits_per_pass ||
      morph.dilate90.source_visits >
          request.max_morph_source_visits_per_pass ||
      morph.boundary.source_visits >
          request.max_morph_source_visits_per_pass ||
      morph.erode269_count.source_visits >
          request.max_morph_source_visits_per_pass ||
      morph.erode89.max_active_slabs >
          request.max_morph_active_slabs ||
      morph.dilate90.max_active_slabs >
          request.max_morph_active_slabs ||
      morph.erode269_count.max_active_slabs >
          request.max_morph_active_slabs ||
      !morph.f90_boundary_segments ||
      morph.f90_boundary_segments >
          request.max_morph_boundary_segments ||
      morph.f90_long_segments >
          request.max_morph_long_segments ||
      !checked_multiply_u64(
          morph.f90_long_segments,
          morph.f90_long_segments
              ? morph.f90_long_segments - 1
              : 0,
          &expected_pairs) ||
      (expected_pairs /= 2) !=
          morph.f90_space_pairs_checked ||
      morph.f90_space_violations ||
      morph.f90_space_uncertain ||
      morph.f270_eroded_intervals ||
      !morph.f90_boundary.empty() ||
      morph.f90_boundary_fnv64 ||
      !output.device_total_bytes ||
      !output.device_free_begin_bytes ||
      !output.device_free_low_bytes ||
      output.device_free_begin_bytes >
          output.device_total_bytes ||
      output.device_free_low_bytes >
          output.device_free_begin_bytes ||
      !morph.device_total_bytes ||
      morph.device_total_bytes != output.device_total_bytes ||
      !morph.device_free_begin_bytes ||
      !morph.device_free_low_bytes ||
      morph.device_free_begin_bytes >
          morph.device_total_bytes ||
      morph.device_free_low_bytes >
          morph.device_free_begin_bytes) {
    throw std::runtime_error(
        "M1 resident-morphology completion invariant failed");
  }
}

struct M1BasePass
{
  mu::GpuUnionOutput output;
  m2m::BaseWidthSpaceContext certificate;
  std::uint64_t h2d_ns = 0;
  std::uint64_t rectangle_expand_ns = 0;
};

M1BasePass run_m1_base_width_space_pass(
    const Request &raw, const LoweredScene &lowered,
    const M1MorphRequest &request, bool transpose)
{
  const __int128 packed_x_range =
      transpose
          ? static_cast<__int128>(raw.scene_top) -
                raw.scene_bottom
          : static_cast<__int128>(raw.scene_right) -
                raw.scene_left;
  const __int128 packed_y_range =
      transpose
          ? static_cast<__int128>(raw.scene_right) -
                raw.scene_left
          : static_cast<__int128>(raw.scene_top) -
                raw.scene_bottom;
  if (packed_x_range <= 0 || packed_y_range <= 0 ||
      packed_x_range >
          std::numeric_limits<std::uint32_t>::max() ||
      packed_y_range >
          std::numeric_limits<std::uint32_t>::max()) {
    coordinate_decline(
        transpose
            ? "transposed raw M1 range exceeds exact packed-union capacity"
            : "raw M1 range exceeds exact packed-union capacity");
  }

  ExpandedRectangles expanded =
      expand_rectangles_resident(raw, lowered);
  M1BasePass pass;
  pass.h2d_ns = expanded.h2d_ns;
  pass.rectangle_expand_ns = expanded.expand_ns;
  if (expanded.status) {
    throw M2Decline(
        DeclineKind::coordinate,
        expanded.status & kExpandTransformOverflow
            ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
            : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT,
        "raw M1 device expansion failed its exact bounds gate");
  }
  if (transpose) {
    const std::uint64_t transpose_ns =
        transpose_m1_rectangles_resident(raw, &expanded);
    if (!checked_add_u64(
            pass.rectangle_expand_ns, transpose_ns,
            &pass.rectangle_expand_ns)) {
      throw std::runtime_error(
          "raw M1 expansion/transpose timing overflow");
    }
    if (expanded.status) {
      throw M2Decline(
          DeclineKind::coordinate,
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT,
          "raw M1 resident coordinate transpose failed its exact "
          "bounds gate");
    }
  }

  mu::GpuUnionLimits union_limits;
  union_limits.max_rectangles = request.max_rectangles;
  union_limits.max_x_slabs = request.max_x_slabs;
  union_limits.max_memberships =
      request.max_union_memberships;
  union_limits.max_events = request.max_union_events;
  union_limits.max_raw_segments =
      request.max_union_raw_segments;
  union_limits.max_segments = request.max_union_segments;
  union_limits.max_slabs_per_rectangle =
      request.max_slabs_per_rectangle;

  mu::GpuUnionStripWindowLimits window_limits;
  window_limits.max_window_events = std::min(
      window_limits.max_window_events, union_limits.max_events);
  window_limits.max_strip_intervals = std::min(
      {kM1BaseMaxStitchedStripIntervals,
       union_limits.max_memberships,
       union_limits.max_raw_segments / 2});
  window_limits.max_windows = std::min(
      window_limits.max_windows, union_limits.max_x_slabs);
  window_limits.max_window_slabs =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          window_limits.max_window_slabs,
          union_limits.max_x_slabs));

  pass.certificate.distance =
      klayout_cuda::m1_width_space::
          kQualifiedSceneCoordinateDistance;
  pass.certificate.origin_x =
      transpose ? raw.scene_bottom : raw.scene_left;
  pass.certificate.origin_y =
      transpose ? raw.scene_left : raw.scene_bottom;
  pass.certificate.max_corner_endpoints =
      request.max_union_segments;
  pass.certificate.max_corner_pair_work =
      request.max_union_events;
  const mu::ResidentStripHook hook =
      m2m::make_base_width_space_hook(&pass.certificate);
  const double input_prepare_ms =
      static_cast<double>(
          pass.h2d_ns + pass.rectangle_expand_ns) /
      1000000.0;
  pass.output = mu::gpu_union_resident_windowed_strips(
      std::move(expanded.rectangles),
      transpose ? raw.scene_left : raw.scene_bottom,
      transpose ? raw.scene_right : raw.scene_top,
      union_limits, window_limits, raw.device,
      input_prepare_ms, &hook);
  const m2m::BaseWidthSpaceResult &certificate =
      pass.certificate.result;
  if (certificate.device_total_bytes) {
    if (!pass.output.device_total_bytes ||
        pass.output.device_total_bytes !=
            certificate.device_total_bytes ||
        !certificate.device_free_begin_bytes ||
        !certificate.device_free_low_bytes ||
        certificate.device_free_begin_bytes >
            certificate.device_total_bytes ||
        certificate.device_free_low_bytes >
            certificate.device_free_begin_bytes) {
      throw std::runtime_error(
          "M1 base-width/space callback memory invariant");
    }
    pass.output.device_free_low_bytes =
        std::min(
            pass.output.device_free_low_bytes,
            certificate.device_free_low_bytes);
  }
  return pass;
}

void validate_m1_base_pass_complete(
    const M1MorphRequest &request,
    const LoweredScene &lowered,
    const M1BasePass &pass)
{
  const mu::GpuUnionOutput &output = pass.output;
  const m2m::BaseWidthSpaceResult &certificate =
      pass.certificate.result;
  std::uint64_t expected_events = 0;
  if (!pass.certificate.invoked ||
      !output.resident_consumer_completed ||
      output.resident_boundary_consumer_completed ||
      !output.segments.empty() || output.raw_segments ||
      output.digest || output.boundary_ms != 0.0 ||
      output.d2h_ms != 0.0 ||
      output.rectangle_count != lowered.flat_rectangles ||
      output.rectangle_count < request.flat_polygon_count ||
      output.rectangle_count > request.max_rectangles ||
      !output.x_slabs ||
      output.x_slabs > request.max_x_slabs ||
      !output.memberships ||
      output.memberships > request.max_union_memberships ||
      !checked_multiply_u64(
          output.memberships, 2, &expected_events) ||
      output.event_count != expected_events ||
      output.event_count > request.max_union_events ||
      !output.strip_intervals ||
      output.strip_intervals > output.memberships ||
      certificate.slabs_checked != output.x_slabs ||
      certificate.intervals_checked !=
          output.strip_intervals ||
      certificate.gaps_checked >
          certificate.intervals_checked ||
      certificate.width_violations >
          certificate.intervals_checked ||
      certificate.space_violations >
          certificate.gaps_checked ||
      !certificate.corner_endpoint_count ||
      certificate.corner_endpoint_count >
          request.max_union_segments ||
      certificate.corner_pair_work >
          request.max_union_events ||
      certificate.corner_candidates >
          certificate.corner_pair_work ||
      !certificate.device_total_bytes ||
      certificate.device_total_bytes !=
          output.device_total_bytes ||
      !certificate.device_free_begin_bytes ||
      !certificate.device_free_low_bytes ||
      certificate.device_free_begin_bytes >
          certificate.device_total_bytes ||
      certificate.device_free_low_bytes >
          certificate.device_free_begin_bytes ||
      output.device_free_low_bytes >
          certificate.device_free_low_bytes ||
      certificate.device_flags != 0 ||
      !output.device_total_bytes ||
      !output.device_free_begin_bytes ||
      !output.device_free_low_bytes ||
      output.device_free_begin_bytes >
          output.device_total_bytes ||
      output.device_free_low_bytes >
          output.device_free_begin_bytes) {
    throw std::runtime_error(
        "M1 resident base-width/space completion invariant failed");
  }
}

int run_m1_base_width_space_request(
    const M1MorphRequest &request, M1MorphResult *result,
    const Request &raw, const LoweredScene &lowered,
    Clock::time_point total_begin)
{
  M1BasePass original =
      run_m1_base_width_space_pass(
          raw, lowered, request, false);
  copy_m1_union_telemetry(original.output, result);
  if (original.output.fallback) {
    result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
    result->fallback_flags =
        fallback_flags_for_union(original.output);
    const std::string message =
        "raw M1 base-width/space x pass: " +
        original.output.message;
    set_message(result, message.c_str());
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return result->status;
  }
  validate_m1_base_pass_complete(
      request, lowered, original);

  M1BasePass transposed =
      run_m1_base_width_space_pass(
          raw, lowered, request, true);
  if (transposed.output.fallback) {
    result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
    result->fallback_flags =
        fallback_flags_for_union(transposed.output);
    const std::string message =
        "raw M1 base-width/space y pass: " +
        transposed.output.message;
    set_message(result, message.c_str());
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return result->status;
  }
  validate_m1_base_pass_complete(
      request, lowered, transposed);

  result->x_slab_count =
      std::max(
          original.output.x_slabs,
          transposed.output.x_slabs);
  result->union_membership_count =
      std::max(
          original.output.memberships,
          transposed.output.memberships);
  result->union_event_count =
      std::max(
          original.output.event_count,
          transposed.output.event_count);
  result->strip_interval_count =
      std::max(
          original.output.strip_intervals,
          transposed.output.strip_intervals);
  result->union_device_free_low_bytes =
      std::min(
          original.output.device_free_low_bytes,
          transposed.output.device_free_low_bytes);
  const std::uint64_t original_x_membership_ns =
      milliseconds_to_ns(original.output.x_membership_ms);
  const std::uint64_t transposed_x_membership_ns =
      milliseconds_to_ns(transposed.output.x_membership_ms);
  const std::uint64_t original_strip_scan_ns =
      milliseconds_to_ns(original.output.strip_scan_ms);
  const std::uint64_t transposed_strip_scan_ns =
      milliseconds_to_ns(transposed.output.strip_scan_ms);
  if (!checked_add_u64(
          original.h2d_ns, transposed.h2d_ns,
          &result->h2d_ns) ||
      !checked_add_u64(
          original.rectangle_expand_ns,
          transposed.rectangle_expand_ns,
          &result->rectangle_expand_ns) ||
      !checked_add_u64(
          original_x_membership_ns,
          transposed_x_membership_ns,
          &result->x_membership_ns) ||
      !checked_add_u64(
          original_strip_scan_ns,
          transposed_strip_scan_ns,
          &result->strip_scan_ns)) {
    throw std::runtime_error(
        "M1 base-width/space telemetry overflow");
  }
  const double certificate_ms =
      original.certificate.result.elapsed_ms +
      transposed.certificate.result.elapsed_ms;
  result->morphology_ns =
      milliseconds_to_ns(certificate_ms);

  const bool width_violation =
      original.certificate.result.width_violations ||
      transposed.certificate.result.width_violations;
  const bool space_violation =
      original.certificate.result.space_violations ||
      transposed.certificate.result.space_violations;
  const bool corner_candidate =
      original.certificate.result.corner_candidates ||
      transposed.certificate.result.corner_candidates;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
  result->device_flags = 0;
  result->status = KLAYOUT_CUDA_SPATIAL_OK;
  if (width_violation || space_violation || corner_candidate) {
    result->certified_empty_mask = 0;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M1_MORPH_NOT_EMPTY;
    set_message(
        result,
        "exact raw M1 union has a possible base width/space violation");
  } else {
    result->certified_empty_mask =
        KLAYOUT_CUDA_SPATIAL_M1_BASE_ALL_EMPTY;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M1_MORPH_COMPLETE;
    set_message(
        result,
        "complete exact raw M1 resident M1.1/M1.2 empty certificate");
  }
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return result->status;
}

int run_m1_morph_request(
    const M1MorphRequest *request, M1MorphResult *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_M1_MORPH_UNCERTAIN;
  if (!request || !valid_m1_morph_request(*request)) {
    set_message(
        result,
        "unsupported or malformed raw M1 morphology request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  echo_m1_morph_request(*request, result);

  const auto total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> lock(pipeline_mutex());
    const auto setup_begin = Clock::now();
    const Request raw = m1_morph_as_union_request(*request);
    const LoweredScene lowered =
        validate_and_lower(raw, kM1RawDigestMagic);
    result->setup_ns = elapsed_ns(setup_begin, Clock::now());
    klayout_cuda::DevicePhaseLease device_lease(
        request->device, "m1_resident_morphology");
    if (request->opcode ==
        KLAYOUT_CUDA_SPATIAL_M1_RAW_MANHATTAN_M11_2_EMPTY) {
      return run_m1_base_width_space_request(
          *request, result, raw, lowered, total_begin);
    }
    const __int128 transposed_y_range =
        static_cast<__int128>(raw.scene_right) -
        raw.scene_left;
    if (transposed_y_range <= 0 ||
        transposed_y_range >
            std::numeric_limits<std::uint32_t>::max()) {
      coordinate_decline(
          "raw M1 transposed y range exceeds exact packed-union capacity");
    }

    const bool qualified_production_scene =
        qualified_production_m1_morph_scene(*request);
    if (request->max_morph_source_visits_per_pass >
            m2m::kUniversalSourceVisitCap &&
        !qualified_production_scene) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
      set_message(
          result,
          "raw M1 scene is not qualified for the 12B morphology work cap");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }

    ExpandedRectangles expanded =
        expand_rectangles_resident(raw, lowered);
    result->h2d_ns = expanded.h2d_ns;
    result->rectangle_expand_ns = expanded.expand_ns;
    if (expanded.status) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          expanded.status & kExpandTransformOverflow
              ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
              : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      set_message(
          result,
          "raw M1 device expansion failed its exact bounds gate");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }
    const std::uint64_t transpose_ns =
        transpose_m1_rectangles_resident(raw, &expanded);
    if (!checked_add_u64(
            result->rectangle_expand_ns, transpose_ns,
            &result->rectangle_expand_ns)) {
      throw std::runtime_error(
          "raw M1 expansion/transpose timing overflow");
    }
    if (expanded.status) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      set_message(
          result,
          "raw M1 resident coordinate transpose failed its exact bounds gate");
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }

    mu::GpuUnionLimits union_limits;
    union_limits.max_rectangles = request->max_rectangles;
    union_limits.max_x_slabs = request->max_x_slabs;
    union_limits.max_memberships =
        request->max_union_memberships;
    union_limits.max_events = request->max_union_events;
    union_limits.max_raw_segments =
        request->max_union_raw_segments;
    union_limits.max_segments = request->max_union_segments;
    union_limits.max_slabs_per_rectangle =
        request->max_slabs_per_rectangle;

    /*
     * The raw-M1 terminal consumer needs the canonical strips, not a
     * materialized boundary.  Bound each window independently while retaining
     * the request's whole-scene membership/event gates.  The internal default
     * caps temporary event storage.  Raw M1 may retain up to 128M exact
     * stitched intervals, still clamped to the caller's existing membership
     * and raw-transition gates.  Each strip interval consumes two raw
     * transitions.
     */
    mu::GpuUnionStripWindowLimits window_limits;
    window_limits.max_window_events = std::min(
        window_limits.max_window_events, union_limits.max_events);
    window_limits.max_strip_intervals = std::min(
        {kM1MaxStitchedStripIntervals,
         union_limits.max_memberships,
         union_limits.max_raw_segments / 2});
    window_limits.max_windows = std::min(
        window_limits.max_windows, union_limits.max_x_slabs);
    window_limits.max_window_slabs =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(
            window_limits.max_window_slabs,
            union_limits.max_x_slabs));

    M1MorphCallbackContext callback;
    callback.request.limits.max_input_x_slabs =
        request->max_x_slabs;
    callback.request.limits.max_input_intervals =
        request->max_union_memberships;
    callback.request.limits.max_output_slabs =
        request->max_morph_output_slabs;
    callback.request.limits.max_output_intervals =
        request->max_morph_output_intervals;
    callback.request.limits.max_raw_boundary_segments =
        request->max_morph_raw_boundary_segments;
    callback.request.limits.max_boundary_segments =
        request->max_morph_boundary_segments;
    callback.request.limits.max_total_source_visits =
        request->max_morph_source_visits_per_pass;
    callback.request.limits.max_source_visits_per_band =
        request->max_morph_source_visits_per_band;
    callback.request.limits.max_active_slabs =
        request->max_morph_active_slabs;
    callback.request.limits.max_long_segments =
        request->max_morph_long_segments;
    callback.request.allow_qualified_m1_production_work_cap =
        qualified_production_scene &&
        request->max_morph_source_visits_per_pass ==
            m2m::kQualifiedM1ProductionSourceVisitCap;
    const mu::ResidentStripHook hook =
        make_m1_morphology_hook(&callback);
    const double input_prepare_ms =
        static_cast<double>(
            result->h2d_ns + result->rectangle_expand_ns) /
        1000000.0;
    const mu::GpuUnionOutput output =
        mu::gpu_union_resident_windowed_strips(
        std::move(expanded.rectangles), raw.scene_left,
        raw.scene_right, union_limits, window_limits, raw.device,
        input_prepare_ms, &hook);
    copy_m1_union_telemetry(output, result);

    if (output.fallback) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags = fallback_flags_for_union(output);
      const std::string message =
          "transposed raw M1 union: " + output.message;
      set_message(result, message.c_str());
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }
    if (!callback.complete) {
      if (morphology_known_not_empty(callback.error)) {
        result->status = KLAYOUT_CUDA_SPATIAL_OK;
        result->fallback_flags =
            KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
        result->disposition =
            KLAYOUT_CUDA_SPATIAL_M1_MORPH_NOT_EMPTY;
      } else {
        result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
        result->fallback_flags =
            morphology_capacity_error(callback.error)
                ? KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY
                : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      }
      set_message(result, callback.error.c_str());
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return result->status;
    }

    copy_m1_morph_telemetry(callback.result, result);
    validate_m1_morph_complete(
        *request, lowered, output, callback);
    result->certified_empty_mask =
        KLAYOUT_CUDA_SPATIAL_M1_MORPH_ALL_EMPTY;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
    result->device_flags = 0;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M1_MORPH_COMPLETE;
    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    set_message(
        result,
        "complete exact transposed raw M1 resident M1.5-.9 empty certificate");
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return result->status;
  } catch (const M2Decline &decline) {
    result->certified_empty_mask = 0;
    result->fallback_flags = decline.fallback_flags();
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M1_MORPH_UNCERTAIN;
    result->status =
        decline.kind() == DeclineKind::bad_argument
            ? KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT
            : KLAYOUT_CUDA_SPATIAL_FALLBACK;
    set_message(result, decline.what());
  } catch (const std::exception &error) {
    result->certified_empty_mask = 0;
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M1_MORPH_UNCERTAIN;
    set_message(result, error.what());
  } catch (...) {
    result->certified_empty_mask = 0;
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M1_MORPH_UNCERTAIN;
    set_message(
        result,
        "unknown exact raw M1 resident-morphology exception");
  }
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return result->status;
}

int run_request(const Request *request, Result *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN;
  result->segment_record_bytes = sizeof(Segment);
  if (!request || !valid_basic_request(*request)) {
    set_message(
        result, "unsupported or malformed raw M2 union request");
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  echo_request(*request, result);

  const auto total_begin = Clock::now();
  try {
    std::lock_guard<std::mutex> lock(pipeline_mutex());

    const auto setup_begin = Clock::now();
    const LoweredScene lowered =
        validate_and_lower(*request, kM2RawDigestMagic);
    result->setup_ns = elapsed_ns(setup_begin, Clock::now());

    ExpandedRectangles expanded =
        expand_rectangles_resident(*request, lowered);
    result->h2d_ns = expanded.h2d_ns;
    result->rectangle_expand_ns = expanded.expand_ns;
    if (expanded.status) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags =
          expanded.status & kExpandTransformOverflow
              ? KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW
              : KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      set_message(
          result,
          "raw M2 device expansion failed its exact bounds gate");
      result->total_ns =
          elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    mu::GpuUnionLimits limits;
    limits.max_rectangles = request->max_rectangles;
    limits.max_x_slabs = request->max_x_slabs;
    limits.max_memberships = request->max_memberships;
    limits.max_events = request->max_events;
    limits.max_raw_segments = request->max_raw_segments;
    limits.max_segments = request->max_segments;
    limits.max_slabs_per_rectangle =
        request->max_slabs_per_rectangle;
    const double input_prepare_ms =
        static_cast<double>(
            result->h2d_ns + result->rectangle_expand_ns) /
        1000000.0;
    m2m::ResidentContext suffix_context;
    mu::ResidentStripHook suffix_hook;
    const mu::ResidentStripHook *suffix_hook_ptr = nullptr;
    const bool suffix_requested =
        request->opcode ==
        KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY;
    std::uint32_t certified_empty_mask = 0;
    std::uint64_t suffix_total_ns = 0;
    const bool qualified_production_suffix =
        suffix_requested &&
        qualified_production_m2_suffix_scene(*request);
    if (suffix_requested) {
      if (qualified_production_suffix) {
        suffix_context.request.allow_qualified_production_work_cap =
            true;
        suffix_context.request.limits.max_total_source_visits =
            m2m::kQualifiedProductionSourceVisitCap;
      }
      suffix_hook = m2m::make_resident_hook(&suffix_context);
      // The same transaction still needs the canonical boundary for the
      // checked host FlatRegion bridge used by M2.1/.2/.4.
      suffix_hook.stop_before_boundary = false;
      suffix_hook_ptr = &suffix_hook;
    }
    const mu::GpuUnionOutput output = mu::gpu_union_resident(
        std::move(expanded.rectangles), request->scene_bottom,
        request->scene_top, limits, request->device,
        input_prepare_ms, suffix_hook_ptr);
    result->x_membership_ns =
        milliseconds_to_ns(output.x_membership_ms);
    result->strip_scan_ns =
        milliseconds_to_ns(output.strip_scan_ms);
    result->boundary_ns =
        milliseconds_to_ns(output.boundary_ms);
    result->d2h_ns = milliseconds_to_ns(output.d2h_ms);
    result->rectangle_count = output.rectangle_count;
    result->x_slab_count = output.x_slabs;
    result->membership_count = output.memberships;
    result->event_count = output.event_count;
    result->strip_interval_count = output.strip_intervals;
    result->raw_segment_count = output.raw_segments;
    result->boundary_fnv64 = output.digest;
    if (output.fallback) {
      result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
      result->fallback_flags = fallback_flags_for_union(output);
      set_message(result, output.message.c_str());
      result->total_ns = elapsed_ns(total_begin, Clock::now());
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    if (suffix_requested) {
      const m2m::Result &suffix = suffix_context.result;
      const bool suffix_clean =
          suffix_context.invoked &&
          output.resident_consumer_completed &&
          suffix.source_x_slabs == output.x_slabs &&
          suffix.source_intervals == output.strip_intervals &&
          suffix.f90_space_violations == 0 &&
          suffix.f90_space_uncertain == 0 &&
          suffix.f270_eroded_intervals == 0;
      const bool production_census_matches =
          !qualified_production_suffix ||
          (suffix.f90_boundary_segments ==
               m2m::kQualifiedF90BoundarySegments &&
           suffix.f90_long_segments ==
               m2m::kQualifiedF90LongSegments &&
           suffix.f90_space_pairs_checked ==
               m2m::kQualifiedF90LongPairs);
      if (!suffix_clean || !production_census_matches) {
        result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
        result->fallback_flags =
            KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
        set_message(
            result,
            "resident M2.5-.9 certificate failed its exact integrity gate");
        result->total_ns = elapsed_ns(total_begin, Clock::now());
        return KLAYOUT_CUDA_SPATIAL_FALLBACK;
      }
      certified_empty_mask =
          KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY;
      suffix_total_ns = milliseconds_to_ns(suffix.total_ms);
    }

    fill_success_result(*request, lowered, output, result);
    result->certified_empty_mask = certified_empty_mask;
    result->suffix_total_ns = suffix_total_ns;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
    result->device_flags = 0;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M2_UNION_COMPLETE;
    result->status = KLAYOUT_CUDA_SPATIAL_OK;
    set_message(
        result,
        suffix_requested
            ? "complete exact raw M2 union and M2.5-.9 empty certificate"
            : "complete exact raw M2 Manhattan-union boundary");
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return KLAYOUT_CUDA_SPATIAL_OK;
  } catch (const M2Decline &decline) {
    clear_suffix_certificate(result);
    result->fallback_flags = decline.fallback_flags();
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN;
    result->status =
        decline.kind() == DeclineKind::bad_argument
            ? KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT
            : KLAYOUT_CUDA_SPATIAL_FALLBACK;
    set_message(result, decline.what());
    result->total_ns = elapsed_ns(total_begin, Clock::now());
    return result->status;
  } catch (const std::exception &error) {
    clear_suffix_certificate(result);
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN;
    set_message(result, error.what());
  } catch (...) {
    clear_suffix_certificate(result);
    result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    result->disposition =
        KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN;
    set_message(
        result, "unknown exact raw M2 union exception");
  }
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  return KLAYOUT_CUDA_SPATIAL_ERROR;
}

void release_result(Result *result) noexcept
{
  if (!result) return;
  const Segment *segments = result->segments;
  bool owned = false;
  if (segments) {
    std::lock_guard<std::mutex> lock(allocation_mutex());
    ++allocation_counters().release_calls;
    owned = owned_allocations().erase(segments) == 1;
    if (owned) ++allocation_counters().owned_releases;
  } else {
    std::lock_guard<std::mutex> lock(allocation_mutex());
    ++allocation_counters().release_calls;
  }
  if (owned) delete[] segments;
  result->segments = nullptr;
  result->segment_count = 0;
}

}  // namespace

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m2_union_boundary_v1(
    const klayout_cuda_spatial_m2_union_request_v1 *request,
    klayout_cuda_spatial_m2_union_result_v1 *result)
{
  try {
    return run_request(request, result);
  } catch (...) {
    if (result) {
      std::memset(result, 0, sizeof(*result));
      result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result->struct_size = sizeof(*result);
      result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_M2_UNION_UNCERTAIN;
      result->segment_record_bytes = sizeof(Segment);
      set_message(
          result,
          "exception escaped the exact raw M2 union boundary");
    }
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m1_resident_morphology_empty_v1(
    const klayout_cuda_spatial_m1_resident_morphology_request_v1 *request,
    klayout_cuda_spatial_m1_resident_morphology_result_v1 *result)
{
  try {
    return run_m1_morph_request(request, result);
  } catch (...) {
    if (result) {
      std::memset(result, 0, sizeof(*result));
      result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result->struct_size = sizeof(*result);
      result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_M1_MORPH_UNCERTAIN;
      set_message(
          result,
          "exception escaped exact raw M1 resident morphology");
    }
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_contact4_active_union_empty_v1(
    const klayout_cuda_spatial_contact4_active_union_request_v1 *request,
    klayout_cuda_spatial_contact4_active_union_result_v1 *result)
{
  try {
    return run_contact4_active_union_request(request, result);
  } catch (...) {
    if (result) {
      std::memset(result, 0, sizeof(*result));
      result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result->struct_size = sizeof(*result);
      result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_UNCERTAIN;
      set_message(
          result,
          "exception escaped resident ACTIVE-union CONTACT4 boundary");
    }
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_active3_well_union_empty_v1(
    const klayout_cuda_spatial_active3_well_union_request_v1 *request,
    klayout_cuda_spatial_active3_well_union_result_v1 *result)
{
  try {
    return run_active3_well_union_request(request, result);
  } catch (...) {
    if (result) {
      std::memset(result, 0, sizeof(*result));
      result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result->struct_size = sizeof(*result);
      result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_ACTIVE3_WELL_UNION_UNCERTAIN;
      set_message(
          result,
          "exception escaped resident exact-WELL-union ACTIVE3 boundary");
    }
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_implant15_raw_empty_v1(
    const klayout_cuda_spatial_implant15_request_v1 *request,
    klayout_cuda_spatial_implant15_result_v1 *result)
{
  try {
    return run_implant15_request(request, result);
  } catch (...) {
    if (result) {
      std::memset(result, 0, sizeof(*result));
      result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result->struct_size = sizeof(*result);
      result->status = KLAYOUT_CUDA_SPATIAL_ERROR;
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      result->disposition =
          KLAYOUT_CUDA_SPATIAL_IMPLANT15_UNCERTAIN;
      set_message(
          result,
          "exception escaped chained resident IMPLANT.1-.5 boundary");
    }
    return KLAYOUT_CUDA_SPATIAL_ERROR;
  }
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT void
klayout_cuda_spatial_release_m2_union_boundary_v1(
    klayout_cuda_spatial_m2_union_result_v1 *result)
{
  try {
    release_result(result);
  } catch (...) {
    if (result) {
      result->segments = nullptr;
      result->segment_count = 0;
    }
  }
}

/*
 * Benchmark-only ownership probe.  This is deliberately outside the public
 * backend ABI: the combined production gate resolves it with dlsym and uses
 * it only to prove that the real loader released each DSO-owned result.
 *
 * selector 0: successful owned allocations
 * selector 1: release entry-point calls
 * selector 2: releases that matched an owned allocation
 * selector 3: currently outstanding owned allocations
 */
extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT std::uint64_t
klayout_cuda_spatial_m2_union_test_counter_v1(std::uint32_t selector)
{
  std::lock_guard<std::mutex> lock(allocation_mutex());
  switch (selector) {
  case 0: return allocation_counters().allocations;
  case 1: return allocation_counters().release_calls;
  case 2: return allocation_counters().owned_releases;
  case 3: return owned_allocations().size();
  default: return std::numeric_limits<std::uint64_t>::max();
  }
}

/*
 * Benchmark-only one-shot fault injection.  Selector 1 throws after the
 * suffix proof and canonical-output validation, but before output ownership
 * or certificate publication.  It is intentionally absent from the public
 * backend header and exists only to lock the fail-closed DSO contract.
 */
extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT void
klayout_cuda_spatial_m2_union_test_fault_v1(std::uint32_t selector)
{
  m2_test_fault().store(selector, std::memory_order_release);
}
