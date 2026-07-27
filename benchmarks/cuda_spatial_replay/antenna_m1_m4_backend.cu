/*
 * Exact high-memory CUDA adapter for the compact M1-through-M4 antenna
 * transaction.
 *
 * It validates and decomposes each stored contour once, expands occurrences
 * on the GPU, and executes one ordered resident connectivity/certificate
 * transaction.  Connectivity is appended one domain at a time so the compact
 * source plus the exact retained frontier fits a 10-GiB device.  All bounded
 * failures return FALLBACK; COMPLETE is emitted only after all four exact
 * checkpoints are clean.
 */

#include "antenna_m1_m4_backend.cuh"

#include "antenna_clean_certificate_gpu.cuh"
#include "antenna_connectivity_gpu.cuh"
#include "antenna_factor_zero_diode_gpu.cuh"
#include "dbCudaActive3Digest.h"
#include "dbCudaAntennaM4Evidence.h"
#include "m2_manhattan_decompose.h"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace ac = klayout_cuda::antenna_connectivity;
namespace afd = klayout_cuda::antenna_factor_zero_diode;
namespace cert = klayout_cuda::antenna_clean_certificate;
namespace md = klayout_cuda::m2_manhattan_decompose;

using Clock = std::chrono::steady_clock;
using Request = klayout_cuda_spatial_antenna_m1_m4_request_v1;
using Result = klayout_cuda_spatial_antenna_m1_m4_result_v1;
using Context = klayout_cuda_spatial_m1_width_space_context_v1;
using Cell = klayout_cuda_spatial_antenna_m1_m4_cell_v1;
using Polygon = klayout_cuda_spatial_m1_width_space_polygon_v1;
using Edge = klayout_cuda_spatial_m1_width_space_edge_v1;

static_assert(sizeof(Context) == 24, "context ABI padding changed");
static_assert(sizeof(Cell) == 24, "compact cell ABI padding changed");
static_assert(sizeof(Polygon) == 48, "polygon ABI padding changed");
static_assert(sizeof(Edge) == 32, "edge ABI padding changed");

constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kMaximumBlocks = 65535;
constexpr std::uint32_t kConnectivityBinsPerMicron = 2;
constexpr std::uint32_t kCertificateBinsPerMicron = 8;
constexpr std::int64_t kCoordinateLimit = INT64_C(1000000000000);
constexpr std::uint32_t kPhysicalLayers[12] = {
    9, 1, 4, 3, 10, 11, 12, 13, 14, 15, 16, 17};
constexpr char kDigestDomains[12][9] = {
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_POLY_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ACTIVE_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NPLUS_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_NWELL_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_CONTACT_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M1_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA1_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M2_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA2_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M3_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_VIA3_DIGEST_DOMAIN,
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_M4_DIGEST_DOMAIN};

class Decline : public std::runtime_error
{
public:
  Decline(std::uint32_t status, std::uint32_t fallback,
          const std::string &message)
      : std::runtime_error(message), status(status), fallback(fallback)
  {
  }

  std::uint32_t status;
  std::uint32_t fallback;
};

[[noreturn]] void malformed(const std::string &message)
{
  throw Decline(
      KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT,
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST, message);
}

[[noreturn]] void capacity(const std::string &message,
                           std::uint32_t flag =
                               KLAYOUT_CUDA_SPATIAL_FALLBACK_MEMBERSHIP_CAPACITY)
{
  throw Decline(KLAYOUT_CUDA_SPATIAL_FALLBACK, flag, message);
}

[[noreturn]] void internal_decline(const std::string &message)
{
  throw Decline(
      KLAYOUT_CUDA_SPATIAL_FALLBACK,
      KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT, message);
}

void cuda_require(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    internal_decline(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

bool backend_phase_timing_enabled()
{
  const char *value =
      std::getenv("KLAYOUT_CUDA_ANTENNA_PHASE_TIMING");
  return value && *value &&
         !(value[0] == '0' && value[1] == '\0');
}

void report_backend_phase(
    std::size_t stage, const char *phase,
    Clock::time_point begin)
{
  if (!backend_phase_timing_enabled()) return;
  cuda_require(
      cudaDeviceSynchronize(),
      "antenna backend phase timing synchronize");
  const double milliseconds =
      std::chrono::duration<double, std::milli>(
          Clock::now() - begin)
          .count();
  std::fprintf(
      stderr,
      "KLAYOUT_CUDA_ANTENNA_PHASE stage=%zu phase=%s ms=%.3f\n",
      stage + 1, phase, milliseconds);
}

class DeviceMemoryAccount
{
public:
  explicit DeviceMemoryAccount(std::uint64_t cap) : m_cap(cap)
  {
    query(&m_total, &m_baseline_used);
    m_peak_global_used = m_baseline_used;
    if (!m_cap || m_baseline_used >= m_cap) {
      capacity("CUDA device is already at the transaction memory cap");
    }
  }

  void admit_growth(std::uint64_t bytes, const char *what)
  {
    std::uint64_t total = 0;
    std::uint64_t used = 0;
    query(&total, &used);
    if (total != m_total || used > m_cap ||
        bytes > m_cap - used) {
      capacity(std::string(what) + " exceeds the global device cap");
    }
    m_peak_global_used =
        std::max(m_peak_global_used, used + bytes);
  }

  void observe()
  {
    std::uint64_t total = 0;
    std::uint64_t used = 0;
    query(&total, &used);
    if (total != m_total || used > m_cap) {
      capacity("observed CUDA residency exceeds the global device cap");
    }
    m_peak_global_used = std::max(m_peak_global_used, used);
  }

  void observe_component_peak(std::uint64_t baseline_relative_peak)
  {
    if (baseline_relative_peak > UINT64_MAX - m_baseline_used) {
      capacity("global device peak overflows uint64");
    }
    const std::uint64_t global_peak =
        m_baseline_used + baseline_relative_peak;
    if (global_peak > m_cap) {
      capacity("certificate peak exceeds the global device cap");
    }
    m_peak_global_used =
        std::max(m_peak_global_used, global_peak);
  }

  std::uint64_t external_live_bytes(
      std::uint64_t certificate_persistent_bytes)
  {
    std::uint64_t total = 0;
    std::uint64_t used = 0;
    query(&total, &used);
    if (total != m_total || used > m_cap) {
      capacity("CUDA residency exceeds the global device cap");
    }
    m_peak_global_used = std::max(m_peak_global_used, used);
    const std::uint64_t relative =
        used > m_baseline_used ? used - m_baseline_used : 0;
    return relative > certificate_persistent_bytes
               ? relative - certificate_persistent_bytes
               : 0;
  }

private:
  static void query(std::uint64_t *total, std::uint64_t *used)
  {
    std::size_t free_size = 0;
    std::size_t total_size = 0;
    cuda_require(
        cudaMemGetInfo(&free_size, &total_size),
        "query CUDA memory accounting");
    *total = total_size;
    *used = total_size - free_size;
  }

  std::uint64_t m_cap = 0;
  std::uint64_t m_total = 0;
  std::uint64_t m_baseline_used = 0;
  std::uint64_t m_peak_global_used = 0;
};

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end)
{
  const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end - begin).count();
  return value > 0 ? static_cast<std::uint64_t>(value) : 1;
}

class OptionalCpuPhaseClock
{
public:
  explicit OptionalCpuPhaseClock(bool enabled) : m_enabled(enabled)
  {
    if (m_enabled) m_begin = Clock::now();
  }

  std::uint64_t split()
  {
    if (!m_enabled) return 0;
    const Clock::time_point end = Clock::now();
    const std::uint64_t duration = elapsed_ns(m_begin, end);
    m_begin = end;
    return duration;
  }

private:
  bool m_enabled = false;
  Clock::time_point m_begin{};
};

std::uint32_t launch_blocks(std::uint64_t count)
{
  if (!count) return 0;
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      (count + kThreads - 1) / kThreads, kMaximumBlocks));
}

bool checked_add(std::uint64_t first, std::uint64_t second,
                 std::uint64_t *result)
{
  if (second > UINT64_MAX - first) return false;
  *result = first + second;
  return true;
}

bool checked_multiply(std::uint64_t first, std::uint64_t second,
                      std::uint64_t *result)
{
  if (first && second > UINT64_MAX / first) return false;
  *result = first * second;
  return true;
}

std::uint64_t add_or_malformed(
    std::uint64_t first, std::uint64_t second, const char *what)
{
  std::uint64_t result = 0;
  if (!checked_add(first, second, &result)) {
    malformed(std::string(what) + " overflows uint64");
  }
  return result;
}

std::uint64_t multiply_or_malformed(
    std::uint64_t first, std::uint64_t second, const char *what)
{
  std::uint64_t result = 0;
  if (!checked_multiply(first, second, &result)) {
    malformed(std::string(what) + " overflows uint64");
  }
  return result;
}

template <class T>
T load_record(const void *base, std::uint64_t index, std::uint32_t stride)
{
  if (!base || stride != sizeof(T) ||
      index > std::numeric_limits<std::size_t>::max() / stride) {
    malformed("record array is null, mis-strided, or overflows size_t");
  }
  T value;
  const auto *bytes = static_cast<const std::uint8_t *>(base);
  std::memcpy(
      &value, bytes + static_cast<std::size_t>(index) * stride,
      sizeof(value));
  return value;
}

bool coordinate_qualified(std::int64_t value)
{
  return value >= -kCoordinateLimit && value <= kCoordinateLimit;
}

struct CanonicalDigest
{
  void bytes(const void *data, std::size_t size) { sha.update(data, size); }
  void u32(std::uint32_t value)
  {
    std::uint8_t encoded[4];
    for (unsigned int i = 0; i < 4; ++i) {
      encoded[i] = static_cast<std::uint8_t>(value >> (i * 8));
    }
    bytes(encoded, sizeof(encoded));
  }
  void u64(std::uint64_t value)
  {
    std::uint8_t encoded[8];
    for (unsigned int i = 0; i < 8; ++i) {
      encoded[i] = static_cast<std::uint8_t>(value >> (i * 8));
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

struct RectangleTemplate
{
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
  std::uint32_t polygon_local = 0;
  std::uint32_t reserved = 0;
};

struct LoweredCell
{
  std::uint64_t rectangle_begin = 0;
  std::uint32_t rectangle_count = 0;
  std::uint32_t polygon_count = 0;
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
};

struct DomainSummary
{
  std::uint64_t nonempty_contexts = 0;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_edges = 0;
  std::uint64_t flat_rectangles = 0;
  std::uint64_t stored_bytes = 0;
  std::uint64_t legacy_stored_bytes = 0;
  std::uint64_t expanded_bytes = 0;
  std::int64_t left = 0;
  std::int64_t bottom = 0;
  std::int64_t right = 0;
  std::int64_t top = 0;
  std::vector<LoweredCell> cells;
  std::vector<RectangleTemplate> rectangles;
  std::vector<std::uint64_t> context_rectangle_offsets;
  std::vector<std::uint64_t> context_owner_offsets;
  std::array<std::uint8_t, 32> scene_digest{};
};

struct Identity
{
  std::array<DomainSummary, 12> domains;
  std::array<std::uint8_t, 32> hierarchy_digest{};
  std::array<std::uint8_t, 32> lower_capture_digest{};
  std::array<std::uint8_t, 32> capture_digest{};
  std::uint64_t stored_cells = 0;
  std::uint64_t stored_polygons = 0;
  std::uint64_t stored_edges = 0;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_edges = 0;
  std::uint64_t total_stored_bytes = 0;
  std::uint64_t total_expanded_bytes = 0;
  std::uint64_t estimated_peak_bytes = 0;
};

struct DomainSetupTiming
{
  std::uint64_t header_ns = 0;
  std::uint64_t templates_ns = 0;
  std::uint64_t contexts_ns = 0;
  std::uint64_t accounting_ns = 0;
  std::uint64_t scene_digest_ns = 0;
};

struct SetupTiming
{
  bool enabled = false;
  bool populate = false;
  std::uint64_t total_ns = 0;
  std::uint64_t telemetry_init_ns = 0;
  std::uint64_t request_header_ns = 0;
  std::uint64_t hierarchy_validate_ns = 0;
  std::uint64_t hierarchy_digest_ns = 0;
  std::array<std::uint64_t, 12> domain_total_ns{};
  std::array<DomainSetupTiming, 12> domains{};
  std::uint64_t domain_lower_wall_ns = 0;
  std::uint64_t domain_finalize_wall_ns = 0;
  std::uint64_t aggregate_capacity_ns = 0;
  std::uint64_t lower_capture_digest_ns = 0;
  std::uint64_t full_capture_digest_ns = 0;
  std::uint64_t census_ns = 0;
};

struct LoweredDomain
{
  DomainSummary summary;
  DomainSetupTiming timing;
  std::uint64_t total_ns = 0;
};

bool setup_timing_enabled()
{
  const char *value =
      std::getenv("KLAYOUT_CUDA_ANTENNA_SETUP_TIMING");
  return value && *value &&
         !(value[0] == '0' && value[1] == '\0');
}

double nanoseconds_to_milliseconds(std::uint64_t nanoseconds)
{
  return static_cast<double>(nanoseconds) / 1000000.0;
}

void report_setup_timing(const SetupTiming &timing)
{
  if (!timing.enabled) return;
  std::uint64_t domain_cpu_ns = 0;
  for (const std::uint64_t duration : timing.domain_total_ns) {
    domain_cpu_ns += duration;
  }
  const std::uint64_t domain_wall_ns =
      timing.domain_lower_wall_ns +
      timing.domain_finalize_wall_ns;
  const std::uint64_t attributed_ns =
      timing.telemetry_init_ns + timing.request_header_ns +
      timing.hierarchy_validate_ns + timing.hierarchy_digest_ns +
      domain_wall_ns + timing.aggregate_capacity_ns +
      timing.lower_capture_digest_ns +
      timing.full_capture_digest_ns + timing.census_ns;
  const std::uint64_t unattributed_ns =
      timing.total_ns > attributed_ns
          ? timing.total_ns - attributed_ns
          : 0;
  std::fprintf(
      stderr,
      "KLAYOUT_CUDA_ANTENNA_SETUP populate=%u total_ms=%.3f "
      "telemetry_init_ms=%.3f request_header_ms=%.3f "
      "hierarchy_validate_ms=%.3f hierarchy_digest_ms=%.3f "
      "domains_ms=%.3f domain_lower_wall_ms=%.3f "
      "domain_finalize_wall_ms=%.3f domain_cpu_ms=%.3f "
      "aggregate_capacity_ms=%.3f "
      "lower_capture_digest_ms=%.3f full_capture_digest_ms=%.3f "
      "census_ms=%.3f unattributed_ms=%.3f\n",
      timing.populate ? 1u : 0u,
      nanoseconds_to_milliseconds(timing.total_ns),
      nanoseconds_to_milliseconds(timing.telemetry_init_ns),
      nanoseconds_to_milliseconds(timing.request_header_ns),
      nanoseconds_to_milliseconds(timing.hierarchy_validate_ns),
      nanoseconds_to_milliseconds(timing.hierarchy_digest_ns),
      nanoseconds_to_milliseconds(domain_wall_ns),
      nanoseconds_to_milliseconds(timing.domain_lower_wall_ns),
      nanoseconds_to_milliseconds(timing.domain_finalize_wall_ns),
      nanoseconds_to_milliseconds(domain_cpu_ns),
      nanoseconds_to_milliseconds(timing.aggregate_capacity_ns),
      nanoseconds_to_milliseconds(timing.lower_capture_digest_ns),
      nanoseconds_to_milliseconds(timing.full_capture_digest_ns),
      nanoseconds_to_milliseconds(timing.census_ns),
      nanoseconds_to_milliseconds(unattributed_ns));
  for (std::size_t role = 0; role < timing.domains.size(); ++role) {
    const DomainSetupTiming &domain = timing.domains[role];
    const std::uint64_t internal_ns =
        domain.header_ns + domain.templates_ns +
        domain.contexts_ns + domain.accounting_ns +
        domain.scene_digest_ns;
    const std::uint64_t finalize_ns =
        timing.domain_total_ns[role] > internal_ns
            ? timing.domain_total_ns[role] - internal_ns
            : 0;
    std::fprintf(
        stderr,
        "KLAYOUT_CUDA_ANTENNA_SETUP_DOMAIN role=%zu total_ms=%.3f "
        "header_ms=%.3f templates_ms=%.3f contexts_ms=%.3f "
        "accounting_ms=%.3f scene_digest_ms=%.3f finalize_ms=%.3f\n",
        role, nanoseconds_to_milliseconds(
                  timing.domain_total_ns[role]),
        nanoseconds_to_milliseconds(domain.header_ns),
        nanoseconds_to_milliseconds(domain.templates_ns),
        nanoseconds_to_milliseconds(domain.contexts_ns),
        nanoseconds_to_milliseconds(domain.accounting_ns),
        nanoseconds_to_milliseconds(domain.scene_digest_ns),
        nanoseconds_to_milliseconds(finalize_ns));
  }
}

bool rectangle_contains(
    const RectangleTemplate &outer,
    const RectangleTemplate &inner)
{
  return outer.left <= inner.left &&
         outer.bottom <= inner.bottom &&
         outer.right >= inner.right &&
         outer.top >= inner.top;
}

std::vector<std::uint8_t> local_diode_contact_witness_templates(
    const Identity &identity)
{
  const DomainSummary &active = identity.domains[1];
  const DomainSummary &nplus = identity.domains[2];
  const DomainSummary &contact = identity.domains[4];
  if (active.cells.size() != contact.cells.size() ||
      nplus.cells.size() != contact.cells.size()) {
    internal_decline(
        "diode witness domains do not share one cell table");
  }

  std::vector<std::uint8_t> witnesses(
      contact.rectangles.size(), 0);
  for (std::size_t cell_id = 0;
       cell_id < contact.cells.size(); ++cell_id) {
    const LoweredCell &contact_cell = contact.cells[cell_id];
    const LoweredCell &active_cell = active.cells[cell_id];
    const LoweredCell &nplus_cell = nplus.cells[cell_id];
    for (std::uint32_t local = 0;
         local < contact_cell.rectangle_count; ++local) {
      const std::uint64_t contact_index =
          contact_cell.rectangle_begin + local;
      const RectangleTemplate &candidate =
          contact.rectangles[contact_index];
      bool inside_active = false;
      for (std::uint32_t active_local = 0;
           active_local < active_cell.rectangle_count;
           ++active_local) {
        if (rectangle_contains(
                active.rectangles[
                    active_cell.rectangle_begin + active_local],
                candidate)) {
          inside_active = true;
          break;
        }
      }
      if (!inside_active) continue;
      for (std::uint32_t nplus_local = 0;
           nplus_local < nplus_cell.rectangle_count;
           ++nplus_local) {
        if (rectangle_contains(
                nplus.rectangles[
                    nplus_cell.rectangle_begin + nplus_local],
                candidate)) {
          witnesses[contact_index] = 1;
          break;
        }
      }
    }
  }
  return witnesses;
}

std::pair<std::int64_t, std::int64_t>
transform_point_host(const Context &context, std::int64_t x, std::int64_t y)
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
  default: malformed("context has an invalid orthogonal transform");
  }
  transformed_x += context.tx;
  transformed_y += context.ty;
  if (transformed_x < INT64_MIN || transformed_x > INT64_MAX ||
      transformed_y < INT64_MIN || transformed_y > INT64_MAX) {
    throw Decline(
        KLAYOUT_CUDA_SPATIAL_FALLBACK,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW,
        "context transform overflows int64");
  }
  const auto output_x = static_cast<std::int64_t>(transformed_x);
  const auto output_y = static_cast<std::int64_t>(transformed_y);
  if (!coordinate_qualified(output_x) || !coordinate_qualified(output_y)) {
    throw Decline(
        KLAYOUT_CUDA_SPATIAL_FALLBACK,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW,
        "context transform exceeds the qualified coordinate domain");
  }
  return {output_x, output_y};
}

std::array<std::uint8_t, 32>
hierarchy_digest(const Request &request)
{
  static constexpr char magic[8] =
      {'K', 'A', 'N', 'T', 'H', '1', '0', '1'};
  CanonicalDigest digest;
  digest.bytes(magic, sizeof(magic));
  digest.u32(1);
  digest.u32(request.dbu_per_micron);
  digest.u32(request.hierarchy.root_cell);
  digest.u32(0);
  digest.u64(request.hierarchy.source_root_cell_index);
  digest.u64(request.hierarchy.source_cell_count);
  digest.u64(request.hierarchy.context_count);
  for (std::uint64_t i = 0;
       i < request.hierarchy.source_cell_count; ++i) {
    digest.u64(request.hierarchy.source_cell_indices[i]);
  }
  for (std::uint64_t i = 0; i < request.hierarchy.context_count; ++i) {
    const Context context = load_record<Context>(
        request.hierarchy.contexts, i,
        request.hierarchy.context_record_bytes);
    digest.i64(context.tx);
    digest.i64(context.ty);
    digest.u32(context.cell_id);
    digest.u32(context.transform_code);
    digest.u32(request.hierarchy.context_parent_ids[i]);
  }
  return digest.finish();
}

bool bytes_zero(const void *data, std::size_t size)
{
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < size; ++i) {
    if (bytes[i]) return false;
  }
  return true;
}

void require_capacity_header(const Request &request)
{
  const auto &c = request.capacity;
  if (c.struct_size != sizeof(c) || c.reserved0 ||
      !bytes_zero(c.reserved1, sizeof(c.reserved1)) ||
      !c.max_cells || !c.max_contexts || !c.max_stored_polygons ||
      !c.max_stored_edges || !c.max_flat_polygons || !c.max_flat_edges ||
      !c.max_total_stored_bytes ||
      !c.max_total_expanded_geometry_bytes ||
      !c.max_estimated_peak_bytes || !c.max_nodes ||
      !c.max_rectangles || !c.max_memberships ||
      !c.max_pair_occurrences || !c.max_unique_candidates ||
      !c.max_cell_members || !c.max_dsu_iterations ||
      !c.max_rule_work || !c.max_device_bytes ||
      c.max_cells > UINT32_MAX || c.max_contexts > UINT32_MAX ||
      c.max_nodes > UINT32_MAX || c.max_rectangles > UINT32_MAX ||
      c.max_cell_members > UINT32_MAX ||
      c.max_dsu_iterations > UINT32_MAX) {
    malformed("antenna capacity header is not qualified");
  }
}

std::vector<Context> validate_hierarchy(
    const Request &request, bool verify_digest)
{
  const auto &h = request.hierarchy;
  if (h.struct_size != sizeof(h) || h.format_version != 2 ||
      h.dbu_per_micron != 2000 || h.reserved0 || h.reserved1 ||
      h.reserved2 || !bytes_zero(h.reserved3, sizeof(h.reserved3)) ||
      !h.source_cell_indices || !h.source_cell_count ||
      h.source_cell_count > request.capacity.max_cells ||
      h.root_cell >= h.source_cell_count ||
      h.source_cell_index_record_bytes != sizeof(std::uint64_t) ||
      !h.contexts || !h.context_count ||
      h.context_count > request.capacity.max_contexts ||
      h.context_record_bytes != sizeof(Context) ||
      !h.context_parent_ids ||
      h.context_parent_count != h.context_count ||
      h.context_parent_record_bytes != sizeof(std::uint32_t) ||
      h.source_root_cell_index != h.source_cell_indices[h.root_cell]) {
    malformed("shared hierarchy header is not qualified");
  }

  std::set<std::uint64_t> source_cells;
  for (std::uint64_t i = 0; i < h.source_cell_count; ++i) {
    if (!source_cells.insert(h.source_cell_indices[i]).second) {
      malformed("source-cell identity is duplicated");
    }
  }

  std::vector<Context> contexts;
  contexts.reserve(static_cast<std::size_t>(h.context_count));
  for (std::uint64_t i = 0; i < h.context_count; ++i) {
    const Context context =
        load_record<Context>(h.contexts, i, h.context_record_bytes);
    const std::uint32_t parent = h.context_parent_ids[i];
    if (!coordinate_qualified(context.tx) ||
        !coordinate_qualified(context.ty) ||
        context.cell_id >= h.source_cell_count ||
        context.transform_code >= 8 ||
        (!i && (context.tx || context.ty ||
                context.cell_id != h.root_cell ||
                context.transform_code ||
                parent != UINT32_MAX)) ||
        (i && parent >= i)) {
      malformed("shared hierarchy context or parent is inconsistent");
    }
    contexts.push_back(context);
  }
  if (verify_digest) {
    const auto digest = hierarchy_digest(request);
    if (std::memcmp(
            digest.data(), h.hierarchy_digest, digest.size()) != 0) {
      malformed("shared hierarchy digest is inconsistent");
    }
  }
  return contexts;
}

std::uint64_t domain_stored_bytes(
    const klayout_cuda_spatial_antenna_m1_m4_domain_v1 &domain)
{
  std::uint64_t total = 80;
  total = add_or_malformed(
      total, multiply_or_malformed(
                 domain.cell_count, sizeof(Cell), "domain cell bytes"),
      "domain stored bytes");
  total = add_or_malformed(
      total, multiply_or_malformed(
                 domain.polygon_count, sizeof(Polygon),
                 "domain polygon bytes"),
      "domain stored bytes");
  total = add_or_malformed(
      total, multiply_or_malformed(
                 domain.edge_count, sizeof(Edge), "domain edge bytes"),
      "domain stored bytes");
  return total;
}

std::uint64_t legacy_domain_stored_bytes(
    const Request &request,
    const klayout_cuda_spatial_antenna_m1_m4_domain_v1 &domain,
    std::uint64_t nonempty_contexts)
{
  std::uint64_t total = 96;
  total = add_or_malformed(
      total, multiply_or_malformed(
                 request.hierarchy.context_count, sizeof(Context),
                 "legacy context bytes"),
      "legacy scene bytes");
  total = add_or_malformed(
      total, multiply_or_malformed(
                 nonempty_contexts, 20, "legacy nonempty context bytes"),
      "legacy scene bytes");
  total = add_or_malformed(
      total, multiply_or_malformed(
                 domain.cell_count, 32, "legacy cell bytes"),
      "legacy scene bytes");
  total = add_or_malformed(
      total, multiply_or_malformed(
                 domain.polygon_count, sizeof(Polygon),
                 "legacy polygon bytes"),
      "legacy scene bytes");
  return add_or_malformed(
      total, multiply_or_malformed(
                 domain.edge_count, sizeof(Edge), "legacy edge bytes"),
      "legacy scene bytes");
}

std::uint64_t expanded_geometry_bytes(
    std::uint64_t flat_polygons, std::uint64_t flat_edges)
{
  return add_or_malformed(
      multiply_or_malformed(flat_polygons, sizeof(Polygon) + 4,
                            "expanded polygon bytes"),
      multiply_or_malformed(flat_edges, sizeof(Edge),
                            "expanded edge bytes"),
      "expanded geometry bytes");
}

void include_cell_world_bounds(
    const Context &context, const LoweredCell &cell, bool *have_bounds,
    std::int64_t *left, std::int64_t *bottom,
    std::int64_t *right, std::int64_t *top)
{
  if (!cell.polygon_count) return;
  const std::int64_t xs[2] = {cell.left, cell.right};
  const std::int64_t ys[2] = {cell.bottom, cell.top};
  for (int xi = 0; xi < 2; ++xi) {
    for (int yi = 0; yi < 2; ++yi) {
      const auto point = transform_point_host(context, xs[xi], ys[yi]);
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

std::array<std::uint8_t, 32> domain_scene_digest(
    const Request &request, std::size_t role,
    const DomainSummary &summary)
{
  const auto &domain = request.domains[role];
  CanonicalDigest digest;
  digest.bytes(kDigestDomains[role], 8);
  digest.u32(1);
  digest.u32(request.dbu_per_micron);
  digest.u32(request.hierarchy.root_cell);
  digest.u32(0);
  digest.u64(request.hierarchy.context_count);
  digest.u64(summary.nonempty_contexts);
  digest.u64(domain.cell_count);
  digest.u64(domain.polygon_count);
  digest.u64(domain.edge_count);
  digest.u64(summary.flat_polygons);
  digest.u64(summary.flat_edges);
  digest.i64(summary.left);
  digest.i64(summary.bottom);
  digest.i64(summary.right);
  digest.i64(summary.top);
  for (std::uint64_t i = 0; i < request.hierarchy.context_count; ++i) {
    const Context context = load_record<Context>(
        request.hierarchy.contexts, i,
        request.hierarchy.context_record_bytes);
    digest.i64(context.tx);
    digest.i64(context.ty);
    digest.u32(context.cell_id);
    digest.u32(context.transform_code);
  }
  std::uint64_t polygon_offset = 0;
  std::uint64_t edge_offset = 0;
  for (std::uint64_t i = 0; i < request.hierarchy.context_count; ++i) {
    const Context context = load_record<Context>(
        request.hierarchy.contexts, i,
        request.hierarchy.context_record_bytes);
    const Cell cell =
        load_record<Cell>(domain.cells, context.cell_id,
                          domain.cell_record_bytes);
    if (!cell.polygon_count) continue;
    digest.u32(static_cast<std::uint32_t>(i));
    digest.u64(polygon_offset);
    digest.u64(edge_offset);
    polygon_offset = add_or_malformed(
        polygon_offset, cell.polygon_count, "context polygon offset");
    edge_offset = add_or_malformed(
        edge_offset, cell.edge_count, "context edge offset");
  }
  for (std::uint64_t i = 0; i < domain.cell_count; ++i) {
    const Cell cell =
        load_record<Cell>(domain.cells, i, domain.cell_record_bytes);
    digest.u64(request.hierarchy.source_cell_indices[i]);
    digest.u64(cell.polygon_begin);
    digest.u64(cell.edge_begin);
    digest.u32(cell.polygon_count);
    digest.u32(cell.edge_count);
  }
  for (std::uint64_t i = 0; i < domain.polygon_count; ++i) {
    const Polygon polygon = load_record<Polygon>(
        domain.polygons, i, domain.polygon_record_bytes);
    digest.u64(polygon.edge_begin);
    digest.i64(polygon.left);
    digest.i64(polygon.bottom);
    digest.i64(polygon.right);
    digest.i64(polygon.top);
    digest.u32(polygon.polygon_id);
    digest.u32(polygon.edge_count);
  }
  for (std::uint64_t i = 0; i < domain.edge_count; ++i) {
    const Edge edge =
        load_record<Edge>(domain.edges, i, domain.edge_record_bytes);
    digest.i64(edge.x1);
    digest.i64(edge.y1);
    digest.i64(edge.x2);
    digest.i64(edge.y2);
  }
  return digest.finish();
}

void require_request_header(const Request &request)
{
  if (request.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      request.struct_size != sizeof(request) ||
      request.opcode !=
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_SHARED_EMPTY ||
      request.option_flags !=
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_QUALIFIED_OPTIONS ||
      request.format_version != 1 || request.dbu_per_micron != 2000 ||
      request.requested_mask !=
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_STAGES ||
      request.stage_count !=
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT ||
      request.ratio_numerator !=
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_NUMERATOR ||
      request.ratio_denominator !=
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_DENOMINATOR ||
      request.domain_count !=
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT ||
      request.device < 0 ||
      !bytes_zero(request.reserved, sizeof(request.reserved))) {
    malformed("antenna transaction header is not qualified");
  }
  require_capacity_header(request);
}

DomainSummary lower_domain(
    const Request &request, std::size_t role,
    const std::vector<Context> &contexts,
    DomainSetupTiming *timing)
{
  OptionalCpuPhaseClock phases(timing != nullptr);
  const auto &domain = request.domains[role];
  if (domain.struct_size != sizeof(domain) || domain.role != role ||
      domain.physical_layer != kPhysicalLayers[role] ||
      domain.datatype || domain.reserved0 || domain.reserved1 ||
      domain.reserved2 || domain.reserved3 ||
      !bytes_zero(domain.reserved4, sizeof(domain.reserved4)) ||
      std::memcmp(domain.digest_domain, kDigestDomains[role], 8) ||
      !domain.cells ||
      domain.cell_count != request.hierarchy.source_cell_count ||
      domain.cell_record_bytes != sizeof(Cell) ||
      !domain.polygons || !domain.polygon_count ||
      domain.polygon_count > request.capacity.max_stored_polygons ||
      domain.polygon_count > UINT32_MAX ||
      domain.polygon_record_bytes != sizeof(Polygon) ||
      !domain.edges || !domain.edge_count ||
      domain.edge_count > request.capacity.max_stored_edges ||
      domain.edge_count > UINT32_MAX ||
      domain.edge_record_bytes != sizeof(Edge)) {
    malformed("compact physical-domain header is not qualified");
  }
  if (timing) timing->header_ns = phases.split();

  DomainSummary summary;
  summary.cells.resize(static_cast<std::size_t>(domain.cell_count));
  std::uint64_t next_polygon = 0;
  std::uint64_t next_edge = 0;
  for (std::uint64_t cell_id = 0; cell_id < domain.cell_count; ++cell_id) {
    const Cell cell =
        load_record<Cell>(domain.cells, cell_id, domain.cell_record_bytes);
    if (cell.polygon_begin != next_polygon ||
        cell.edge_begin != next_edge ||
        next_polygon > domain.polygon_count ||
        next_edge > domain.edge_count ||
        cell.polygon_count > domain.polygon_count - next_polygon ||
        cell.edge_count > domain.edge_count - next_edge) {
      malformed("compact cell ranges are not contiguous");
    }
    LoweredCell lowered;
    lowered.rectangle_begin = summary.rectangles.size();
    lowered.polygon_count = cell.polygon_count;
    bool have_cell_bounds = false;
    std::uint64_t cell_edge = cell.edge_begin;
    for (std::uint32_t local = 0; local < cell.polygon_count; ++local) {
      const std::uint64_t polygon_id = cell.polygon_begin + local;
      const Polygon polygon = load_record<Polygon>(
          domain.polygons, polygon_id, domain.polygon_record_bytes);
      if (polygon.polygon_id != local ||
          polygon.edge_begin != cell_edge || polygon.edge_count < 4 ||
          (polygon.edge_count & 1u) ||
          polygon.left >= polygon.right ||
          polygon.bottom >= polygon.top ||
          !coordinate_qualified(polygon.left) ||
          !coordinate_qualified(polygon.bottom) ||
          !coordinate_qualified(polygon.right) ||
          !coordinate_qualified(polygon.top) ||
          polygon.edge_begin > domain.edge_count ||
          polygon.edge_count >
              domain.edge_count - polygon.edge_begin) {
        malformed("compact polygon range or bounds are inconsistent");
      }
      std::vector<md::EdgeI64> edges;
      edges.reserve(polygon.edge_count);
      for (std::uint64_t edge_id = polygon.edge_begin;
           edge_id < polygon.edge_begin + polygon.edge_count; ++edge_id) {
        const Edge edge =
            load_record<Edge>(domain.edges, edge_id,
                              domain.edge_record_bytes);
        if (!coordinate_qualified(edge.x1) ||
            !coordinate_qualified(edge.y1) ||
            !coordinate_qualified(edge.x2) ||
            !coordinate_qualified(edge.y2)) {
          malformed("compact edge exceeds qualified coordinate domain");
        }
        edges.push_back({edge.x1, edge.y1, edge.x2, edge.y2});
      }
      const std::uint64_t remaining =
          request.capacity.max_rectangles > summary.rectangles.size()
              ? request.capacity.max_rectangles -
                    summary.rectangles.size()
              : 0;
      md::Result decomposition = md::decompose(
          edges, polygon.left, polygon.bottom,
          polygon.right, polygon.top, polygon_id, remaining);
      if (decomposition.status == md::Status::capacity) {
        capacity("exact rectangulation exceeds rectangle capacity");
      }
      if (decomposition.status != md::Status::complete ||
          decomposition.rectangles.empty()) {
        malformed(
            std::string("compact Manhattan contour is malformed: ") +
            decomposition.message);
      }
      for (const auto &rectangle : decomposition.rectangles) {
        summary.rectangles.push_back(
            {rectangle.left, rectangle.bottom,
             rectangle.right, rectangle.top, local, 0});
      }
      if (!have_cell_bounds) {
        lowered.left = polygon.left;
        lowered.bottom = polygon.bottom;
        lowered.right = polygon.right;
        lowered.top = polygon.top;
        have_cell_bounds = true;
      } else {
        lowered.left = std::min(lowered.left, polygon.left);
        lowered.bottom = std::min(lowered.bottom, polygon.bottom);
        lowered.right = std::max(lowered.right, polygon.right);
        lowered.top = std::max(lowered.top, polygon.top);
      }
      cell_edge = add_or_malformed(
          polygon.edge_begin, polygon.edge_count, "polygon edge end");
    }
    if (cell_edge != cell.edge_begin + cell.edge_count) {
      malformed("compact cell edge census is inconsistent");
    }
    const std::uint64_t rectangle_count =
        summary.rectangles.size() - lowered.rectangle_begin;
    if (rectangle_count > UINT32_MAX) {
      capacity("one cell rectangulation exceeds uint32");
    }
    lowered.rectangle_count =
        static_cast<std::uint32_t>(rectangle_count);
    summary.cells[static_cast<std::size_t>(cell_id)] = lowered;
    next_polygon = add_or_malformed(
        next_polygon, cell.polygon_count, "stored polygon count");
    next_edge = add_or_malformed(
        next_edge, cell.edge_count, "stored edge count");
  }
  if (next_polygon != domain.polygon_count ||
      next_edge != domain.edge_count) {
    malformed("compact geometry arrays are not fully covered");
  }
  if (timing) timing->templates_ns = phases.split();

  summary.context_rectangle_offsets.reserve(contexts.size() + 1);
  summary.context_owner_offsets.reserve(contexts.size() + 1);
  summary.context_rectangle_offsets.push_back(0);
  summary.context_owner_offsets.push_back(0);
  bool have_bounds = false;
  for (const Context &context : contexts) {
    const LoweredCell &cell = summary.cells[context.cell_id];
    if (cell.polygon_count) ++summary.nonempty_contexts;
    summary.flat_polygons = add_or_malformed(
        summary.flat_polygons, cell.polygon_count,
        "flat polygon count");
    const Cell source_cell = load_record<Cell>(
        domain.cells, context.cell_id, domain.cell_record_bytes);
    summary.flat_edges = add_or_malformed(
        summary.flat_edges, source_cell.edge_count,
        "flat edge count");
    summary.flat_rectangles = add_or_malformed(
        summary.flat_rectangles, cell.rectangle_count,
        "flat rectangle count");
    summary.context_owner_offsets.push_back(summary.flat_polygons);
    summary.context_rectangle_offsets.push_back(
        summary.flat_rectangles);
    include_cell_world_bounds(
        context, cell, &have_bounds, &summary.left, &summary.bottom,
        &summary.right, &summary.top);
  }
  if (!have_bounds || !summary.nonempty_contexts ||
      !summary.flat_polygons || !summary.flat_edges ||
      summary.flat_polygons > request.capacity.max_flat_polygons ||
      summary.flat_edges > request.capacity.max_flat_edges ||
      summary.flat_rectangles > request.capacity.max_rectangles) {
    capacity("flattened domain census is empty or over capacity");
  }
  if (timing) timing->contexts_ns = phases.split();
  summary.stored_bytes = domain_stored_bytes(domain);
  summary.legacy_stored_bytes = legacy_domain_stored_bytes(
      request, domain, summary.nonempty_contexts);
  summary.expanded_bytes = expanded_geometry_bytes(
      summary.flat_polygons, summary.flat_edges);
  if (timing) timing->accounting_ns = phases.split();
  summary.scene_digest = domain_scene_digest(request, role, summary);
  if (timing) timing->scene_digest_ns = phases.split();
  return summary;
}

std::array<std::uint8_t, 32> lower_capture_digest(
    const Request &request, const Identity &identity)
{
  static constexpr char magic[8] =
      {'K', 'A', 'N', 'T', 'M', '1', '0', '1'};
  CanonicalDigest digest;
  digest.bytes(magic, 8);
  digest.u32(1);
  digest.u32(request.dbu_per_micron);
  digest.u32(request.hierarchy.root_cell);
  digest.u32(0);
  digest.u64(request.hierarchy.source_root_cell_index);
  digest.bytes(identity.hierarchy_digest.data(), 32);
  digest.u32(6);
  std::uint64_t legacy_total = 0;
  std::uint64_t lower_expanded = 0;
  std::uint64_t nonempty = 0;
  std::uint64_t polygons = 0;
  std::uint64_t edges = 0;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_edges = 0;
  for (std::size_t role = 0; role < 6; ++role) {
    const auto &domain = request.domains[role];
    const auto &summary = identity.domains[role];
    digest.u32(role);
    digest.u32(kPhysicalLayers[role]);
    digest.u32(0);
    digest.u32(domain.source_layer_index);
    digest.u64(domain.cell_count);
    digest.u64(request.hierarchy.context_count);
    digest.u64(summary.nonempty_contexts);
    digest.u64(domain.polygon_count);
    digest.u64(domain.edge_count);
    digest.u64(summary.flat_polygons);
    digest.u64(summary.flat_edges);
    digest.u64(summary.legacy_stored_bytes);
    digest.u64(summary.expanded_bytes);
    digest.bytes(summary.scene_digest.data(), 32);
    legacy_total = add_or_malformed(
        legacy_total, summary.legacy_stored_bytes,
        "lower legacy bytes");
    lower_expanded = add_or_malformed(
        lower_expanded, summary.expanded_bytes,
        "lower expanded bytes");
    nonempty = add_or_malformed(
        nonempty, summary.nonempty_contexts,
        "lower nonempty contexts");
    polygons = add_or_malformed(
        polygons, domain.polygon_count, "lower polygons");
    edges = add_or_malformed(
        edges, domain.edge_count, "lower edges");
    flat_polygons = add_or_malformed(
        flat_polygons, summary.flat_polygons, "lower flat polygons");
    flat_edges = add_or_malformed(
        flat_edges, summary.flat_edges, "lower flat edges");
  }
  const std::uint64_t parent_bytes = multiply_or_malformed(
      request.hierarchy.context_count, sizeof(std::uint32_t),
      "parent bytes");
  legacy_total =
      add_or_malformed(legacy_total, parent_bytes, "lower legacy bytes");
  digest.u64(request.hierarchy.source_cell_count);
  digest.u64(request.hierarchy.context_count);
  digest.u64(request.hierarchy.context_count);
  digest.u64(parent_bytes);
  digest.u64(request.hierarchy.source_cell_count * 6);
  digest.u64(request.hierarchy.context_count * 6);
  digest.u64(nonempty);
  digest.u64(polygons);
  digest.u64(edges);
  digest.u64(flat_polygons);
  digest.u64(flat_edges);
  digest.u64(legacy_total);
  digest.u64(lower_expanded);
  digest.u64(add_or_malformed(
      legacy_total, lower_expanded, "lower legacy peak"));
  return digest.finish();
}

std::array<std::uint8_t, 32> full_capture_digest(
    const Request &request, const Identity &identity)
{
  static constexpr char magic[8] =
      {'K', 'A', 'N', 'T', 'M', '4', '0', '1'};
  CanonicalDigest digest;
  digest.bytes(magic, 8);
  digest.u32(1);
  digest.u32(0);
  digest.bytes(identity.hierarchy_digest.data(), 32);
  digest.bytes(identity.lower_capture_digest.data(), 32);
  digest.u32(12);
  for (std::size_t role = 0; role < 12; ++role) {
    const auto &domain = request.domains[role];
    const auto &summary = identity.domains[role];
    digest.u32(role);
    digest.u32(kPhysicalLayers[role]);
    digest.u32(0);
    digest.u32(domain.source_layer_index);
    digest.u64(domain.cell_count);
    digest.u64(request.hierarchy.context_count);
    digest.u64(summary.nonempty_contexts);
    digest.u64(domain.polygon_count);
    digest.u64(domain.edge_count);
    digest.u64(summary.flat_polygons);
    digest.u64(summary.flat_edges);
    digest.u64(summary.stored_bytes);
    digest.u64(summary.legacy_stored_bytes);
    digest.u64(summary.expanded_bytes);
    digest.bytes(summary.scene_digest.data(), 32);
  }
  digest.u64(request.hierarchy.source_cell_count);
  digest.u64(request.hierarchy.context_count);
  digest.u64(request.hierarchy.context_count);
  digest.u64(identity.stored_cells);
  digest.u64(identity.stored_polygons);
  digest.u64(identity.stored_edges);
  digest.u64(identity.flat_polygons);
  digest.u64(identity.flat_edges);
  digest.u64(identity.total_stored_bytes);
  digest.u64(identity.total_expanded_bytes);
  digest.u64(identity.estimated_peak_bytes);
  return digest.finish();
}

std::uint64_t shared_lower_stored_bytes(const Request &request)
{
  std::uint64_t total = 112;
  total = add_or_malformed(
      total, multiply_or_malformed(
                 request.hierarchy.source_cell_count,
                 sizeof(std::uint64_t), "source-cell bytes"),
      "shared lower bytes");
  total = add_or_malformed(
      total, multiply_or_malformed(
                 request.hierarchy.context_count, sizeof(Context),
                 "shared context bytes"),
      "shared lower bytes");
  return add_or_malformed(
      total, multiply_or_malformed(
                 request.hierarchy.context_count,
                 sizeof(std::uint32_t), "shared parent bytes"),
      "shared lower bytes");
}

Identity derive_identity(Request &request, bool populate)
{
  const Clock::time_point setup_begin = Clock::now();
  SetupTiming timing;
  timing.enabled = setup_timing_enabled();
  timing.populate = populate;
  if (timing.enabled) {
    timing.telemetry_init_ns =
        elapsed_ns(setup_begin, Clock::now());
  }
  OptionalCpuPhaseClock phases(timing.enabled);
  require_request_header(request);
  if (timing.enabled) {
    timing.request_header_ns = phases.split();
  }
  const std::vector<Context> contexts =
      validate_hierarchy(request, false);
  if (timing.enabled) {
    timing.hierarchy_validate_ns = phases.split();
  }
  const auto derived_hierarchy_digest = hierarchy_digest(request);
  if (populate) {
    std::memcpy(
        request.hierarchy.hierarchy_digest,
        derived_hierarchy_digest.data(), derived_hierarchy_digest.size());
  } else if (std::memcmp(
                 request.hierarchy.hierarchy_digest,
                 derived_hierarchy_digest.data(),
                 derived_hierarchy_digest.size()) != 0) {
    malformed("shared hierarchy digest is inconsistent");
  }
  if (timing.enabled) {
    timing.hierarchy_digest_ns = phases.split();
  }

  Identity identity;
  identity.hierarchy_digest = derived_hierarchy_digest;
  identity.total_stored_bytes = shared_lower_stored_bytes(request);
  std::set<std::uint32_t> source_layers;
  for (std::size_t role = 0; role < 12; ++role) {
    if (!source_layers.insert(
            request.domains[role].source_layer_index).second) {
      malformed("source-layer identity is duplicated");
    }
  }

  std::array<LoweredDomain, 12> lowered_domains;
  std::array<std::exception_ptr, 12> lower_errors{};
  const unsigned int advertised_workers =
      std::thread::hardware_concurrency();
  const std::size_t worker_count = std::min<std::size_t>(
      12, advertised_workers ? advertised_workers : 1);
  const auto lower_worker = [&](std::size_t worker) {
    for (std::size_t role = worker; role < 12;
         role += worker_count) {
      try {
        LoweredDomain lowered;
        OptionalCpuPhaseClock total(timing.enabled);
        lowered.summary = lower_domain(
            request, role, contexts,
            timing.enabled ? &lowered.timing : nullptr);
        if (timing.enabled) lowered.total_ns = total.split();
        lowered_domains[role] = std::move(lowered);
      } catch (...) {
        lower_errors[role] = std::current_exception();
      }
    }
  };
  if (worker_count == 1) {
    lower_worker(0);
  } else {
    std::array<std::future<void>, 12> workers;
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers[worker] = std::async(
          std::launch::async, lower_worker, worker);
    }
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers[worker].get();
    }
  }
  for (std::size_t role = 0; role < 12; ++role) {
    if (lower_errors[role]) {
      std::rethrow_exception(lower_errors[role]);
    }
  }
  if (timing.enabled) {
    timing.domain_lower_wall_ns = phases.split();
  }

  std::uint64_t total_rectangles = 0;
  for (std::size_t role = 0; role < 12; ++role) {
    OptionalCpuPhaseClock finalize(timing.enabled);
    auto &domain = request.domains[role];
    identity.domains[role] =
        std::move(lowered_domains[role].summary);
    if (timing.enabled) {
      timing.domains[role] =
          lowered_domains[role].timing;
    }
    const DomainSummary &summary = identity.domains[role];
    if (populate) {
      domain.nonempty_context_count = summary.nonempty_contexts;
      domain.flat_polygon_count = summary.flat_polygons;
      domain.flat_edge_count = summary.flat_edges;
      domain.stored_bytes = summary.stored_bytes;
      domain.expanded_geometry_bytes = summary.expanded_bytes;
      domain.scene_left = summary.left;
      domain.scene_bottom = summary.bottom;
      domain.scene_right = summary.right;
      domain.scene_top = summary.top;
      std::memcpy(
          domain.scene_digest, summary.scene_digest.data(),
          summary.scene_digest.size());
    } else if (
        domain.nonempty_context_count != summary.nonempty_contexts ||
        domain.flat_polygon_count != summary.flat_polygons ||
        domain.flat_edge_count != summary.flat_edges ||
        domain.stored_bytes != summary.stored_bytes ||
        domain.expanded_geometry_bytes != summary.expanded_bytes ||
        domain.scene_left != summary.left ||
        domain.scene_bottom != summary.bottom ||
        domain.scene_right != summary.right ||
        domain.scene_top != summary.top ||
        std::memcmp(
            domain.scene_digest, summary.scene_digest.data(),
            summary.scene_digest.size()) != 0) {
      malformed("compact domain identity or census is inconsistent");
    }
    identity.stored_cells = add_or_malformed(
        identity.stored_cells, domain.cell_count,
        "stored cell census");
    identity.stored_polygons = add_or_malformed(
        identity.stored_polygons, domain.polygon_count,
        "stored polygon census");
    identity.stored_edges = add_or_malformed(
        identity.stored_edges, domain.edge_count,
        "stored edge census");
    identity.flat_polygons = add_or_malformed(
        identity.flat_polygons, summary.flat_polygons,
        "flat polygon census");
    identity.flat_edges = add_or_malformed(
        identity.flat_edges, summary.flat_edges,
        "flat edge census");
    identity.total_expanded_bytes = add_or_malformed(
        identity.total_expanded_bytes, summary.expanded_bytes,
        "expanded byte census");
    total_rectangles = add_or_malformed(
        total_rectangles, summary.flat_rectangles,
        "rectangle census");
    if (role == 6) {
      identity.total_stored_bytes = add_or_malformed(
          identity.total_stored_bytes, 64,
          "M1-through-M4 capture header bytes");
    }
    identity.total_stored_bytes = add_or_malformed(
        identity.total_stored_bytes, summary.stored_bytes,
        "stored byte census");
    if (timing.enabled) {
      timing.domain_total_ns[role] =
          lowered_domains[role].total_ns + finalize.split();
    }
  }
  if (timing.enabled) {
    timing.domain_finalize_wall_ns = phases.split();
  }
  identity.estimated_peak_bytes = add_or_malformed(
      identity.total_stored_bytes, identity.total_expanded_bytes,
      "estimated peak bytes");
  if (identity.flat_polygons > request.capacity.max_nodes ||
      total_rectangles > request.capacity.max_rectangles ||
      identity.total_stored_bytes >
          request.capacity.max_total_stored_bytes ||
      identity.total_expanded_bytes >
          request.capacity.max_total_expanded_geometry_bytes ||
      identity.estimated_peak_bytes >
          request.capacity.max_estimated_peak_bytes) {
    capacity("derived transaction census exceeds capacity");
  }
  if (timing.enabled) {
    timing.aggregate_capacity_ns = phases.split();
  }

  identity.lower_capture_digest =
      lower_capture_digest(request, identity);
  if (populate) {
    std::memcpy(
        request.lower_capture_digest,
        identity.lower_capture_digest.data(), 32);
  } else if (std::memcmp(
                 request.lower_capture_digest,
                 identity.lower_capture_digest.data(), 32) != 0) {
    malformed("embedded M1 capture digest is inconsistent");
  }
  if (timing.enabled) {
    timing.lower_capture_digest_ns = phases.split();
  }
  identity.capture_digest = full_capture_digest(request, identity);
  if (populate) {
    std::memcpy(
        request.capture_digest, identity.capture_digest.data(), 32);
  } else if (std::memcmp(
                 request.capture_digest,
                 identity.capture_digest.data(), 32) != 0) {
    malformed("M1-through-M4 capture digest is inconsistent");
  }
  if (timing.enabled) {
    timing.full_capture_digest_ns = phases.split();
  }

  auto &census = request.census;
  if (populate) {
    std::memset(&census, 0, sizeof(census));
    census.struct_size = sizeof(census);
    census.format_version = request.format_version;
    census.shared_cell_count = request.hierarchy.source_cell_count;
    census.shared_context_count = request.hierarchy.context_count;
    census.context_parent_record_count =
        request.hierarchy.context_count;
    census.stored_cell_record_count = identity.stored_cells;
    census.stored_polygon_count = identity.stored_polygons;
    census.stored_edge_count = identity.stored_edges;
    census.expanded_polygon_count = identity.flat_polygons;
    census.expanded_edge_count = identity.flat_edges;
    census.total_stored_bytes = identity.total_stored_bytes;
    census.total_expanded_geometry_bytes =
        identity.total_expanded_bytes;
    census.estimated_peak_bytes = identity.estimated_peak_bytes;
  } else if (
      census.struct_size != sizeof(census) ||
      census.format_version != request.format_version ||
      !bytes_zero(census.reserved, sizeof(census.reserved)) ||
      census.shared_cell_count !=
          request.hierarchy.source_cell_count ||
      census.shared_context_count !=
          request.hierarchy.context_count ||
      census.context_parent_record_count !=
          request.hierarchy.context_count ||
      census.stored_cell_record_count != identity.stored_cells ||
      census.stored_polygon_count != identity.stored_polygons ||
      census.stored_edge_count != identity.stored_edges ||
      census.expanded_polygon_count != identity.flat_polygons ||
      census.expanded_edge_count != identity.flat_edges ||
      census.total_stored_bytes != identity.total_stored_bytes ||
      census.total_expanded_geometry_bytes !=
          identity.total_expanded_bytes ||
      census.estimated_peak_bytes != identity.estimated_peak_bytes) {
    malformed("aggregate transaction census is inconsistent");
  }
  if (timing.enabled) {
    timing.census_ns = phases.split();
    timing.total_ns = elapsed_ns(setup_begin, Clock::now());
    report_setup_timing(timing);
  }
  return identity;
}

__device__ std::uint64_t ordered_i64(std::int64_t value)
{
  return static_cast<std::uint64_t>(value) ^
         UINT64_C(0x8000000000000000);
}

__device__ void transform_point_device(
    const Context &context, std::int64_t x, std::int64_t y,
    std::int64_t *output_x, std::int64_t *output_y)
{
  switch (context.transform_code) {
  case 0: *output_x = x; *output_y = y; break;
  case 1: *output_x = -y; *output_y = x; break;
  case 2: *output_x = -x; *output_y = -y; break;
  case 3: *output_x = y; *output_y = -x; break;
  case 4: *output_x = x; *output_y = -y; break;
  case 5: *output_x = y; *output_y = x; break;
  case 6: *output_x = -x; *output_y = y; break;
  default: *output_x = -y; *output_y = -x; break;
  }
  *output_x += context.tx;
  *output_y += context.ty;
}

__global__ void expand_domain_kernel(
    const Context *contexts, std::uint64_t context_count,
    const LoweredCell *cells,
    const RectangleTemplate *templates,
    const std::uint64_t *rectangle_offsets,
    const std::uint64_t *owner_offsets,
    std::uint64_t owner_base, std::uint32_t domain,
    bool collapse_owner,
    const std::uint8_t *template_annotations,
    std::uint32_t *owner_annotations,
    ac::RectI64 *output,
    unsigned long long *bounds_and_status)
{
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t context_id = first;
       context_id < context_count; context_id += stride) {
    const Context context = contexts[context_id];
    const LoweredCell cell = cells[context.cell_id];
    const std::uint64_t output_begin = rectangle_offsets[context_id];
    for (std::uint32_t local = 0; local < cell.rectangle_count; ++local) {
      const RectangleTemplate input =
          templates[cell.rectangle_begin + local];
      const std::int64_t xs[2] = {input.left, input.right};
      const std::int64_t ys[2] = {input.bottom, input.top};
      std::int64_t left = 0;
      std::int64_t bottom = 0;
      std::int64_t right = 0;
      std::int64_t top = 0;
      for (int xi = 0; xi < 2; ++xi) {
        for (int yi = 0; yi < 2; ++yi) {
          std::int64_t x = 0;
          std::int64_t y = 0;
          transform_point_device(context, xs[xi], ys[yi], &x, &y);
          if (x < -kCoordinateLimit || x > kCoordinateLimit ||
              y < -kCoordinateLimit || y > kCoordinateLimit) {
            atomicOr(bounds_and_status + 4, 1ull);
          }
          if (!xi && !yi) {
            left = right = x;
            bottom = top = y;
          } else {
            left = min(left, x);
            bottom = min(bottom, y);
            right = max(right, x);
            top = max(top, y);
          }
        }
      }
      const std::uint64_t owner =
          owner_base + owner_offsets[context_id] +
          input.polygon_local;
      if (owner > UINT32_MAX) {
        atomicOr(bounds_and_status + 4, 2ull);
      }
      ac::RectI64 rectangle;
      rectangle.left = left;
      rectangle.bottom = bottom;
      rectangle.right = right;
      rectangle.top = top;
      rectangle.owner =
          collapse_owner ? 0 : static_cast<std::uint32_t>(owner);
      rectangle.domain = domain;
      output[output_begin + local] = rectangle;
      if (template_annotations &&
          owner_annotations &&
          template_annotations[
              cell.rectangle_begin + local]) {
        atomicExch(
            owner_annotations +
                owner_offsets[context_id] +
                input.polygon_local,
            1u);
      }
      atomicMin(bounds_and_status + 0, ordered_i64(left));
      atomicMin(bounds_and_status + 1, ordered_i64(bottom));
      atomicMax(bounds_and_status + 2, ordered_i64(right));
      atomicMax(bounds_and_status + 3, ordered_i64(top));
    }
  }
}

struct Expansion
{
  thrust::device_vector<ac::RectI64> rectangles;
  thrust::device_vector<std::uint32_t> owner_annotations;
};

class DeviceExpander
{
public:
  DeviceExpander(
      const Request &request, const std::vector<Context> &contexts,
      DeviceMemoryAccount *memory, std::uint64_t *h2d_ns,
      std::uint64_t *d2h_ns)
      : m_request(request), m_memory(memory),
        m_h2d_ns(h2d_ns), m_d2h_ns(d2h_ns)
  {
    m_memory->admit_growth(
        multiply_or_malformed(
            contexts.size(), sizeof(Context), "device context bytes"),
        "shared context upload");
    const auto begin = Clock::now();
    m_contexts = contexts;
    cuda_require(cudaDeviceSynchronize(), "upload shared contexts");
    m_memory->observe();
    *m_h2d_ns = add_or_malformed(
        *m_h2d_ns, elapsed_ns(begin, Clock::now()), "H2D timing");
  }

  Expansion expand(
      const DomainSummary &summary, std::uint32_t role,
      std::uint64_t owner_base, bool collapse_owner,
      const std::vector<std::uint8_t> *template_annotations =
          nullptr)
  {
    if (template_annotations &&
        template_annotations->size() !=
            summary.rectangles.size()) {
      internal_decline(
          "domain template annotation size is inconsistent");
    }
    std::uint64_t known_bytes = 0;
    known_bytes = add_or_malformed(
        known_bytes,
        multiply_or_malformed(
            summary.cells.size(), sizeof(LoweredCell),
            "device cell bytes"),
        "known device bytes");
    known_bytes = add_or_malformed(
        known_bytes,
        multiply_or_malformed(
            summary.rectangles.size(), sizeof(RectangleTemplate),
            "device template bytes"),
        "known device bytes");
    known_bytes = add_or_malformed(
        known_bytes,
        multiply_or_malformed(
            summary.context_rectangle_offsets.size() +
                summary.context_owner_offsets.size(),
            sizeof(std::uint64_t), "device offset bytes"),
        "known device bytes");
    known_bytes = add_or_malformed(
        known_bytes,
        multiply_or_malformed(
            summary.flat_rectangles, sizeof(ac::RectI64),
            "device expanded rectangle bytes"),
        "known device bytes");
    known_bytes = add_or_malformed(
        known_bytes, 5 * sizeof(unsigned long long),
        "known device bytes");
    if (template_annotations) {
      known_bytes = add_or_malformed(
          known_bytes,
          multiply_or_malformed(
              template_annotations->size(),
              sizeof(std::uint8_t),
              "device template annotation bytes"),
          "known device bytes");
      known_bytes = add_or_malformed(
          known_bytes,
          multiply_or_malformed(
              summary.flat_polygons,
              sizeof(std::uint32_t),
              "device owner annotation bytes"),
          "known device bytes");
    }
    m_memory->admit_growth(known_bytes, "compact domain expansion");

    const auto h2d_begin = Clock::now();
    thrust::device_vector<LoweredCell> cells(summary.cells);
    thrust::device_vector<RectangleTemplate> templates(
        summary.rectangles);
    thrust::device_vector<std::uint64_t> rectangle_offsets(
        summary.context_rectangle_offsets);
    thrust::device_vector<std::uint64_t> owner_offsets(
        summary.context_owner_offsets);
    thrust::device_vector<std::uint8_t> annotations;
    if (template_annotations) {
      annotations = *template_annotations;
    }
    thrust::device_vector<unsigned long long> bounds(5);
    const unsigned long long initial_bounds[5] = {
        ULLONG_MAX, ULLONG_MAX, 0, 0, 0};
    cuda_require(
        cudaMemcpy(
            thrust::raw_pointer_cast(bounds.data()), initial_bounds,
            sizeof(initial_bounds), cudaMemcpyHostToDevice),
        "initialize domain bounds");
    Expansion expansion;
    expansion.rectangles.resize(
        static_cast<std::size_t>(summary.flat_rectangles));
    if (template_annotations) {
      expansion.owner_annotations.resize(
          static_cast<std::size_t>(summary.flat_polygons));
      cuda_require(
          cudaMemset(
              thrust::raw_pointer_cast(
                  expansion.owner_annotations.data()),
              0,
              expansion.owner_annotations.size() *
                  sizeof(std::uint32_t)),
          "clear expanded owner annotations");
    }
    cuda_require(cudaDeviceSynchronize(), "upload compact domain");
    m_memory->observe();
    *m_h2d_ns = add_or_malformed(
        *m_h2d_ns, elapsed_ns(h2d_begin, Clock::now()), "H2D timing");

    expand_domain_kernel<<<
        launch_blocks(m_request.hierarchy.context_count), kThreads>>>(
        thrust::raw_pointer_cast(m_contexts.data()),
        m_request.hierarchy.context_count,
        thrust::raw_pointer_cast(cells.data()),
        thrust::raw_pointer_cast(templates.data()),
        thrust::raw_pointer_cast(rectangle_offsets.data()),
        thrust::raw_pointer_cast(owner_offsets.data()),
        owner_base, role, collapse_owner,
        template_annotations
            ? thrust::raw_pointer_cast(annotations.data())
            : nullptr,
        template_annotations
            ? thrust::raw_pointer_cast(
                  expansion.owner_annotations.data())
            : nullptr,
        thrust::raw_pointer_cast(expansion.rectangles.data()),
        thrust::raw_pointer_cast(bounds.data()));
    cuda_require(cudaGetLastError(), "launch domain expansion");
    cuda_require(cudaDeviceSynchronize(), "complete domain expansion");

    const auto d2h_begin = Clock::now();
    unsigned long long host_bounds[5] = {};
    cuda_require(
        cudaMemcpy(
            host_bounds, thrust::raw_pointer_cast(bounds.data()),
            sizeof(host_bounds), cudaMemcpyDeviceToHost),
        "copy domain bounds");
    *m_d2h_ns = add_or_malformed(
        *m_d2h_ns, elapsed_ns(d2h_begin, Clock::now()), "D2H timing");
    if (host_bounds[4] & 1ull) {
      throw Decline(
          KLAYOUT_CUDA_SPATIAL_FALLBACK,
          KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW,
          "device expansion exceeded qualified coordinate domain");
    }
    if (host_bounds[4] & 2ull) {
      capacity("device expansion owner ID exceeds uint32");
    }
    const auto decode = [](unsigned long long value) {
      return static_cast<std::int64_t>(
          value ^ UINT64_C(0x8000000000000000));
    };
    if (decode(host_bounds[0]) != summary.left ||
        decode(host_bounds[1]) != summary.bottom ||
        decode(host_bounds[2]) != summary.right ||
        decode(host_bounds[3]) != summary.top) {
      internal_decline(
          "GPU-expanded bounds disagree with compact identity");
    }
    return expansion;
  }

private:
  const Request &m_request;
  DeviceMemoryAccount *m_memory;
  std::uint64_t *m_h2d_ns;
  std::uint64_t *m_d2h_ns;
  thrust::device_vector<Context> m_contexts;
};

void require_connectivity(ac::Status status, const char *operation)
{
  if (status == ac::Status::success) return;
  if (status == ac::Status::capacity_exceeded) {
    capacity(
        std::string(operation) + ": " + ac::status_string(status));
  }
  if (status == ac::Status::malformed_input ||
      status == ac::Status::invalid_configuration) {
    internal_decline(
        std::string(operation) + ": " + ac::status_string(status));
  }
  internal_decline(
      std::string(operation) + ": " + ac::status_string(status));
}

void require_certificate(cert::Status status, const char *operation)
{
  if (status == cert::Status::success) return;
  if (status == cert::Status::capacity_exceeded) {
    capacity(
        std::string(operation) + ": " + cert::status_string(status),
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY);
  }
  if (status == cert::Status::arithmetic_overflow) {
    throw Decline(
        KLAYOUT_CUDA_SPATIAL_FALLBACK,
        KLAYOUT_CUDA_SPATIAL_FALLBACK_COORDINATE_OVERFLOW,
        std::string(operation) + ": " + cert::status_string(status));
  }
  internal_decline(
      std::string(operation) + ": " + cert::status_string(status));
}

void require_factor_zero_diode(
    afd::Status status, const char *operation)
{
  if (status == afd::Status::success) return;
  if (status == afd::Status::capacity_exceeded) {
    capacity(
        std::string(operation) + ": " +
            afd::status_string(status),
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY);
  }
  internal_decline(
      std::string(operation) + ": " +
      afd::status_string(status));
}

ac::Config connectivity_config(const Request &request)
{
  ac::Config config;
  config.domain_count = 12;
  config.exact_filter_before_materialization = true;
  config.bin_size = std::max<std::int64_t>(
      1, (request.dbu_per_micron +
          kConnectivityBinsPerMicron - 1) /
             kConnectivityBinsPerMicron);
  config.device = request.device;
  const std::uint32_t neighbors[9][3] = {
      {0, 4, UINT32_MAX},
      {4, 0, 5},
      {5, 4, 6},
      {6, 5, 7},
      {7, 6, 8},
      {8, 7, 9},
      {9, 8, 10},
      {10, 9, 11},
      {11, 10, UINT32_MAX}};
  for (const auto &row : neighbors) {
    const std::uint32_t role = row[0];
    config.relation_rows[role] |= UINT64_C(1) << role;
    for (int i = 1; i < 3; ++i) {
      if (row[i] == UINT32_MAX) continue;
      config.relation_rows[role] |= UINT64_C(1) << row[i];
      config.relation_rows[row[i]] |= UINT64_C(1) << role;
    }
  }
  config.limits.max_nodes = request.capacity.max_nodes;
  config.limits.max_rectangles = request.capacity.max_rectangles;
  config.limits.max_memberships = request.capacity.max_memberships;
  config.limits.max_pair_occurrences =
      request.capacity.max_pair_occurrences;
  config.limits.max_unique_candidates =
      request.capacity.max_unique_candidates;
  config.limits.max_cell_members =
      static_cast<std::uint32_t>(request.capacity.max_cell_members);
  config.limits.max_pair_tests_per_cell =
      request.capacity.max_rule_work;
  config.limits.max_total_pair_tests =
      request.capacity.max_rule_work;
  config.limits.max_dsu_iterations =
      static_cast<std::uint32_t>(
          request.capacity.max_dsu_iterations);
  config.limits.max_device_bytes = request.capacity.max_device_bytes;
  return config;
}

afd::Config factor_zero_diode_config(const Request &request)
{
  afd::Config config;
  config.device = request.device;
  config.bin_size = std::max<std::int64_t>(
      1, (request.dbu_per_micron +
          kConnectivityBinsPerMicron - 1) /
             kConnectivityBinsPerMicron);
  config.limits = connectivity_config(request).limits;
  return config;
}

void filter_factor_zero_diode_contacts(
    const Request &request, DeviceMemoryAccount *memory,
    std::uint64_t contact_owner_count,
    std::uint64_t nwell_owner_count,
    Expansion *contact, Expansion *nwell)
{
  if (!contact || !nwell ||
      contact->owner_annotations.size() !=
          contact_owner_count) {
    internal_decline(
        "factor-zero diode witness inputs are inconsistent");
  }
  const std::uint64_t contact_rectangle_count =
      contact->rectangles.size();
  const std::uint64_t total_rectangles = add_or_malformed(
      contact_rectangle_count, nwell->rectangles.size(),
      "diode witness rectangle count");
  memory->admit_growth(
      multiply_or_malformed(
          total_rectangles, sizeof(ac::RectI64),
          "diode witness concatenation bytes"),
      "concatenate diode witness geometry");
  contact->rectangles.reserve(
      static_cast<std::size_t>(total_rectangles));
  contact->rectangles.resize(
      static_cast<std::size_t>(total_rectangles));
  if (!nwell->rectangles.empty()) {
    cuda_require(
        cudaMemcpy(
            thrust::raw_pointer_cast(contact->rectangles.data()) +
                contact_rectangle_count,
            thrust::raw_pointer_cast(nwell->rectangles.data()),
            nwell->rectangles.size() * sizeof(ac::RectI64),
            cudaMemcpyDeviceToDevice),
        "concatenate diode witness NWELL geometry");
  }
  nwell->rectangles.clear();
  nwell->rectangles.shrink_to_fit();
  cuda_require(
      cudaDeviceSynchronize(),
      "complete diode witness geometry concatenation");
  memory->observe();

  afd::ContactWitnessCensus census;
  require_factor_zero_diode(
      afd::filter_contact_witnesses(
          factor_zero_diode_config(request),
          std::move(contact->rectangles),
          contact_owner_count, nwell_owner_count,
          &contact->owner_annotations, &census),
      "filter factor-zero diode CONTACT witnesses");
  if (!contact->rectangles.empty()) {
    internal_decline(
        "factor-zero diode filter did not consume geometry");
  }
  std::fprintf(
      stderr,
      "ANTENNA_DIODE_WITNESS "
      "local_nplus_active_contacts=%llu "
      "well_rejected_contacts=%llu "
      "exact_witness_contacts=%llu memberships=%llu "
      "pair_occurrences=%llu exact_edges=%llu\n",
      static_cast<unsigned long long>(
          census.local_nplus_active_contacts),
      static_cast<unsigned long long>(
          census.well_rejected_contacts),
      static_cast<unsigned long long>(
          census.exact_witness_contacts),
      static_cast<unsigned long long>(census.memberships),
      static_cast<unsigned long long>(census.pair_occurrences),
      static_cast<unsigned long long>(census.exact_edges));
  memory->observe();
}

cert::Config certificate_config(const Request &request)
{
  cert::Config config;
  config.device = request.device;
  config.grid_cell_size = std::max<std::int64_t>(
      1, (request.dbu_per_micron +
          kCertificateBinsPerMicron - 1) /
             kCertificateBinsPerMicron);
  config.limits.max_live_device_bytes =
      request.capacity.max_device_bytes;
  config.limits.max_annotation_owners =
      request.capacity.max_nodes;
  config.limits.max_labels = request.capacity.max_nodes;
  config.limits.max_poly_tiles = request.capacity.max_rectangles;
  config.limits.max_active_tiles = request.capacity.max_rectangles;
  config.limits.max_metal_tiles = request.capacity.max_rectangles;
  /*
   * The certificate core uses 32-bit indices for these two arrays.  A
   * broader transaction cap therefore maps to the strongest representable
   * internal ceiling; exceeding it declines safely.
   */
  config.limits.max_grid_cells = std::min<std::uint64_t>(
      request.capacity.max_memberships, UINT32_MAX);
  config.limits.max_active_memberships = std::min<std::uint64_t>(
      request.capacity.max_memberships, UINT32_MAX);
  config.limits.max_query_visits = request.capacity.max_rule_work;
  config.limits.max_refinement_records =
      std::min<std::uint64_t>(
          request.capacity.max_memberships, INT_MAX);
  config.limits.max_cell_members =
      static_cast<std::uint32_t>(request.capacity.max_cell_members);
  return config;
}

void fill_hierarchy_echo(const Request &request, Result *result)
{
  auto &echo = result->hierarchy;
  std::memset(&echo, 0, sizeof(echo));
  echo.struct_size = sizeof(echo);
  echo.format_version = request.hierarchy.format_version;
  echo.dbu_per_micron = request.hierarchy.dbu_per_micron;
  echo.root_cell = request.hierarchy.root_cell;
  echo.source_root_cell_index =
      request.hierarchy.source_root_cell_index;
  echo.source_cell_count = request.hierarchy.source_cell_count;
  echo.source_cell_index_record_bytes =
      request.hierarchy.source_cell_index_record_bytes;
  echo.context_count = request.hierarchy.context_count;
  echo.context_record_bytes =
      request.hierarchy.context_record_bytes;
  echo.context_parent_count =
      request.hierarchy.context_parent_count;
  echo.context_parent_record_bytes =
      request.hierarchy.context_parent_record_bytes;
  std::memcpy(
      echo.hierarchy_digest, request.hierarchy.hierarchy_digest, 32);
}

void fill_result_echo(const Request &request, Result *result)
{
  result->abi_version = request.abi_version;
  result->struct_size = sizeof(*result);
  result->opcode = request.opcode;
  result->option_flags = request.option_flags;
  result->format_version = request.format_version;
  result->dbu_per_micron = request.dbu_per_micron;
  result->requested_mask = request.requested_mask;
  result->stage_count = request.stage_count;
  result->ratio_numerator = request.ratio_numerator;
  result->ratio_denominator = request.ratio_denominator;
  result->domain_count = request.domain_count;
  result->device = request.device;
  fill_hierarchy_echo(request, result);
  for (std::size_t role = 0; role < 12; ++role) {
    const auto &domain = request.domains[role];
    auto &echo = result->domains[role];
    std::memset(&echo, 0, sizeof(echo));
    echo.struct_size = sizeof(echo);
    echo.role = domain.role;
    echo.physical_layer = domain.physical_layer;
    echo.datatype = domain.datatype;
    echo.source_layer_index = domain.source_layer_index;
    echo.cell_count = domain.cell_count;
    echo.cell_record_bytes = domain.cell_record_bytes;
    echo.polygon_count = domain.polygon_count;
    echo.polygon_record_bytes = domain.polygon_record_bytes;
    echo.edge_count = domain.edge_count;
    echo.edge_record_bytes = domain.edge_record_bytes;
    echo.nonempty_context_count = domain.nonempty_context_count;
    echo.flat_polygon_count = domain.flat_polygon_count;
    echo.flat_edge_count = domain.flat_edge_count;
    echo.stored_bytes = domain.stored_bytes;
    echo.expanded_geometry_bytes = domain.expanded_geometry_bytes;
    echo.scene_left = domain.scene_left;
    echo.scene_bottom = domain.scene_bottom;
    echo.scene_right = domain.scene_right;
    echo.scene_top = domain.scene_top;
    std::memcpy(echo.digest_domain, domain.digest_domain, 8);
    std::memcpy(echo.scene_digest, domain.scene_digest, 32);
  }
  result->census = request.census;
  result->capacity = request.capacity;
  std::memcpy(
      result->lower_capture_digest, request.lower_capture_digest, 32);
  std::memcpy(result->capture_digest, request.capture_digest, 32);
}

std::array<std::uint64_t, 12> graph_owner_bases(
    const Identity &identity, std::uint64_t *total_nodes)
{
  static constexpr std::uint32_t roles[9] =
      {0, 4, 5, 6, 7, 8, 9, 10, 11};
  std::array<std::uint64_t, 12> bases{};
  std::uint64_t cursor = 0;
  for (std::uint32_t role : roles) {
    bases[role] = cursor;
    cursor = add_or_malformed(
        cursor, identity.domains[role].flat_polygons,
        "graph owner count");
  }
  if (cursor > UINT32_MAX) {
    capacity("graph owner count exceeds uint32");
  }
  *total_nodes = cursor;
  return bases;
}

void populate_domain_evidence(
    const Request &request, const Identity &identity, Result *result)
{
  for (std::size_t role = 0; role < 12; ++role) {
    auto &output = result->domain_results[role];
    std::memset(&output, 0, sizeof(output));
    output.struct_size = sizeof(output);
    output.role = role;
    output.owner_count = identity.domains[role].flat_polygons;
    output.rectangle_count =
        identity.domains[role].flat_rectangles;
    output.owner_range_count = output.owner_count;
    db::cuda_antenna_m1_m4_evidence::Digest digest;
    if (!db::cuda_antenna_m1_m4_evidence::domain_digest(
            request, role, output, digest)) {
      internal_decline("domain evidence digest rejected a valid role");
    }
    std::memcpy(
        output.rectangle_digest, digest.data(), digest.size());
  }
}

void merge_stage_census(
    ac::StageCensus *aggregate, const ac::StageCensus &part,
    bool first)
{
  const std::uint64_t part_nodes = add_or_malformed(
      part.previous_nodes, part.appended_nodes,
      "sub-append node census");
  const std::uint64_t part_rectangles = add_or_malformed(
      part.previous_rectangles, part.appended_rectangles,
      "sub-append rectangle census");
  const std::uint64_t part_disposition = add_or_malformed(
      part.retained_rectangles, part.released_rectangles,
      "sub-append closure census");
  if (part.total_nodes != part_nodes ||
      part.total_rectangles != part_rectangles ||
      part.total_rectangles != part_disposition) {
    internal_decline("sub-append returned an inconsistent census");
  }

  if (first) {
    *aggregate = part;
    return;
  }
  if (part.previous_nodes != aggregate->total_nodes ||
      part.previous_rectangles != aggregate->retained_rectangles ||
      (part.closed_domain_mask & aggregate->closed_domain_mask) !=
          aggregate->closed_domain_mask) {
    internal_decline("sub-append frontier is not continuous");
  }

  aggregate->appended_nodes = add_or_malformed(
      aggregate->appended_nodes, part.appended_nodes,
      "logical-stage node census");
  aggregate->total_nodes = part.total_nodes;
  aggregate->appended_rectangles = add_or_malformed(
      aggregate->appended_rectangles, part.appended_rectangles,
      "logical-stage rectangle census");
  aggregate->total_rectangles = add_or_malformed(
      aggregate->previous_rectangles,
      aggregate->appended_rectangles,
      "logical-stage rectangle census");
  aggregate->retained_rectangles = part.retained_rectangles;
  aggregate->retained_rectangle_capacity =
      part.retained_rectangle_capacity;
  aggregate->released_rectangles = add_or_malformed(
      aggregate->released_rectangles, part.released_rectangles,
      "logical-stage closure census");
  aggregate->closed_domain_mask = part.closed_domain_mask;
  aggregate->memberships = add_or_malformed(
      aggregate->memberships, part.memberships,
      "logical-stage membership census");
  aggregate->occupied_cells = add_or_malformed(
      aggregate->occupied_cells, part.occupied_cells,
      "logical-stage occupied-cell census");
  aggregate->pair_occurrences = add_or_malformed(
      aggregate->pair_occurrences, part.pair_occurrences,
      "logical-stage pair census");
  aggregate->unique_candidates = add_or_malformed(
      aggregate->unique_candidates, part.unique_candidates,
      "logical-stage candidate census");
  aggregate->exact_edges = add_or_malformed(
      aggregate->exact_edges, part.exact_edges,
      "logical-stage edge census");
  if (part.dsu_iterations >
      std::numeric_limits<std::uint32_t>::max() -
          aggregate->dsu_iterations) {
    internal_decline("logical-stage DSU census overflows uint32");
  }
  aggregate->dsu_iterations += part.dsu_iterations;
  for (std::size_t index = 0; index < ac::kRelationSlots; ++index) {
    aggregate->candidates_by_relation[index] = add_or_malformed(
        aggregate->candidates_by_relation[index],
        part.candidates_by_relation[index],
        "logical-stage relation candidate census");
    aggregate->edges_by_relation[index] = add_or_malformed(
        aggregate->edges_by_relation[index],
        part.edges_by_relation[index],
        "logical-stage relation edge census");
  }

  if (aggregate->total_nodes !=
          add_or_malformed(
              aggregate->previous_nodes, aggregate->appended_nodes,
              "logical-stage node census") ||
      aggregate->total_rectangles !=
          add_or_malformed(
              aggregate->retained_rectangles,
              aggregate->released_rectangles,
              "logical-stage closure census")) {
    internal_decline("logical-stage aggregate is inconsistent");
  }
}

std::uint64_t maximum_connectivity_frontier(
    const Identity &identity)
{
  const auto pair_frontier =
      [&](std::size_t first, std::size_t second) {
        return add_or_malformed(
            identity.domains[first].flat_rectangles,
            identity.domains[second].flat_rectangles,
            "connectivity frontier");
      };
  std::uint64_t maximum = identity.domains[0].flat_rectangles;
  const std::size_t adjacent_domains[][2] = {
      {0, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8},
      {8, 9}, {9, 10}, {10, 11}};
  for (const auto &domains : adjacent_domains) {
    maximum = std::max(
        maximum, pair_frontier(domains[0], domains[1]));
  }
  return maximum;
}

void require_logical_stage_capacity(
    const ac::StageCensus &graph, const Request &request)
{
  if (graph.memberships > request.capacity.max_memberships ||
      graph.occupied_cells > graph.memberships) {
    capacity("logical-stage membership census exceeds capacity");
  }
  if (graph.pair_occurrences >
      request.capacity.max_pair_occurrences) {
    capacity(
        "logical-stage pair census exceeds capacity",
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY);
  }
  if (graph.unique_candidates > graph.pair_occurrences ||
      graph.unique_candidates >
          request.capacity.max_unique_candidates) {
    capacity(
        "logical-stage candidate census exceeds capacity",
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY);
  }
  if (graph.exact_edges > graph.unique_candidates ||
      graph.exact_edges > request.capacity.max_rule_work) {
    capacity(
        "logical-stage edge census exceeds rule-work capacity",
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY);
  }
  if (graph.dsu_iterations >
      request.capacity.max_dsu_iterations) {
    capacity(
        "logical-stage DSU census exceeds capacity",
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY);
  }
}

void fill_stage_result(
    std::size_t index, const ac::StageCensus &graph,
    const cert::CheckpointCensus &checkpoint,
    std::uint64_t stage_ns, const Request &request, Result *result)
{
  auto &stage = result->stages[index];
  std::memset(&stage, 0, sizeof(stage));
  stage.struct_size = sizeof(stage);
  stage.stage = static_cast<std::uint32_t>(1u << index);
  stage.component_count = checkpoint.roots;
  stage.membership_count = graph.memberships;
  stage.occupied_cell_count = graph.occupied_cells;
  stage.pair_occurrence_count = graph.pair_occurrences;
  stage.unique_owner_candidate_count = graph.unique_candidates;
  stage.edge_count = graph.exact_edges;
  stage.gate_count =
      checkpoint.roots - checkpoint.roots_without_gate;
  stage.evaluated_count =
      checkpoint.ratio_certified_roots +
      checkpoint.uncertain_roots;
  stage.exempt_count = checkpoint.diode_exempt_roots;
  stage.retained_rectangle_count = graph.retained_rectangles;
  stage.released_rectangle_count = graph.released_rectangles;
  stage.dsu_iteration_count = graph.dsu_iterations;
  stage.hit_count = 0;
  stage.uncertainty_count = checkpoint.uncertain_roots;
  stage.work_count = add_or_malformed(
      checkpoint.roots, checkpoint.metal_tiles,
      "checkpoint work count");
  if (stage.work_count > request.capacity.max_rule_work) {
    capacity(
        "checkpoint work exceeds rule-work capacity",
        KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY);
  }
  stage.stage_ns = stage_ns;
}

void run_transaction(
    const Request &request, const Identity &identity,
    std::uint64_t setup_ns, Clock::time_point total_begin,
    Result *result)
{
  cuda_require(cudaSetDevice(request.device), "select CUDA device");
  DeviceMemoryAccount memory(request.capacity.max_device_bytes);

  std::uint64_t total_nodes = 0;
  const auto owner_bases =
      graph_owner_bases(identity, &total_nodes);
  if (total_nodes > request.capacity.max_nodes) {
    capacity("graph owner count exceeds node capacity");
  }

  std::uint64_t h2d_ns = 0;
  std::uint64_t d2h_ns = 0;
  const std::vector<Context> contexts =
      validate_hierarchy(request, true);
  DeviceExpander expander(
      request, contexts, &memory, &h2d_ns, &d2h_ns);
  ac::Connectivity connectivity(connectivity_config(request));
  require_connectivity(
      connectivity.configuration_status(),
      "configure staged connectivity");
  cert::Certificate certificate(certificate_config(request));
  require_certificate(
      certificate.configuration_status(),
      "configure clean certificate");
  const std::vector<std::uint8_t>
      diode_contact_template_witnesses =
          local_diode_contact_witness_templates(identity);
  thrust::device_vector<std::uint32_t>
      diode_contact_witnesses;

  Expansion poly = expander.expand(
      identity.domains[0], 0, owner_bases[0], false);
  Expansion active = expander.expand(
      identity.domains[1], 1, 0, true);
  cert::GateCensus gate_census;
  require_certificate(
      certificate.set_external_live_device_bytes(
          memory.external_live_bytes(0)),
      "account external gate-census residency");
  require_certificate(
      certificate.build_gate_census(
          thrust::raw_pointer_cast(poly.rectangles.data()),
          poly.rectangles.size(),
          thrust::raw_pointer_cast(active.rectangles.data()),
          active.rectangles.size(),
          identity.domains[0].flat_polygons, &gate_census),
      "build POLY/ACTIVE gate census");
  memory.observe_component_peak(gate_census.peak_live_bytes);
  memory.observe();
  active.rectangles.clear();
  active.rectangles.shrink_to_fit();

  /*
   * The first consuming append adopts POLY's allocation.  Reserve the exact
   * largest adjacent-domain frontier now, while ACTIVE and gate-census
   * scratch are gone.  Every later append then copies into this stable
   * allocation and releases its source before constructing grid/sort
   * scratch; device_vector's geometric growth cannot transiently cross the
   * admitted cap.
   */
  const std::uint64_t frontier_rectangles =
      maximum_connectivity_frontier(identity);
  if (frontier_rectangles > poly.rectangles.capacity()) {
    memory.admit_growth(
        multiply_or_malformed(
            frontier_rectangles, sizeof(ac::RectI64),
            "reserved connectivity frontier bytes"),
        "reserve exact connectivity frontier");
    poly.rectangles.reserve(
        static_cast<std::size_t>(frontier_rectangles));
    cuda_require(
        cudaDeviceSynchronize(),
        "reserve exact connectivity frontier");
    memory.observe();
  }

  auto append_domain =
      [&](Expansion *expansion, std::uint64_t new_nodes,
          std::uint64_t close_mask, ac::StageCensus *aggregate,
          bool first) {
        ac::StageCensus part;
        require_connectivity(
            connectivity.append_stage_consuming(
                std::move(expansion->rectangles), new_nodes,
                close_mask, nullptr, &part),
            "append resident antenna domain");
        memory.observe();
        if (!expansion->rectangles.empty() ||
            part.closed_domain_mask != close_mask) {
          internal_decline(
              "consuming sub-append did not consume or close exactly");
        }
        merge_stage_census(aggregate, part, first);
      };

  auto checkpoint_stage =
      [&](std::size_t index, std::uint32_t metal_role,
          cert::MetalLevel level, const ac::StageCensus &graph,
          Clock::time_point stage_begin) {
        const auto checkpoint_begin = Clock::now();
        require_logical_stage_capacity(graph, request);
        for (std::uint32_t low = 0; low < 12; ++low) {
          for (std::uint32_t high = low; high < 12; ++high) {
            const std::size_t slot =
                static_cast<std::size_t>(low) *
                    ac::kMaximumDomains +
                high;
            if (!graph.candidates_by_relation[slot] &&
                !graph.edges_by_relation[slot]) {
              continue;
            }
            std::fprintf(
                stderr,
                "ANTENNA_CONNECTIVITY_RELATION "
                "stage=%zu low=%u high=%u "
                "candidates=%llu edges=%llu\n",
                index + 1, low, high,
                static_cast<unsigned long long>(
                    graph.candidates_by_relation[slot]),
                static_cast<unsigned long long>(
                    graph.edges_by_relation[slot]));
          }
        }
        /*
         * Re-expand only after append_stage_consuming has destroyed its
         * membership/sort scratch and released the compact append source.
         * The retained graph copy and this certificate view are the only
         * expanded copies alive at the checkpoint.
         */
        const auto metal_expand_begin = Clock::now();
        Expansion metal = expander.expand(
            identity.domains[metal_role], metal_role,
            owner_bases[metal_role], false);
        report_backend_phase(
            index, "checkpoint_expand_metal",
            metal_expand_begin);
        const auto label_view_begin = Clock::now();
        ac::DeviceLabelView labels;
        require_connectivity(
            connectivity.device_label_view(&labels),
            "obtain resident connectivity labels");
        report_backend_phase(
            index, "checkpoint_label_view",
            label_view_begin);
        cert::CheckpointCensus checkpoint;
        cert::FactorZeroDiodeDeviceView diode_view;
        const cert::FactorZeroDiodeDeviceView *diode_view_ptr =
            nullptr;
        if (!diode_contact_witnesses.empty()) {
          diode_view.contact_present =
              thrust::raw_pointer_cast(
                  diode_contact_witnesses.data());
          diode_view.owner_begin = owner_bases[4];
          diode_view.count =
              diode_contact_witnesses.size();
          diode_view_ptr = &diode_view;
        }
        require_certificate(
            certificate.set_external_live_device_bytes(
                memory.external_live_bytes(
                    gate_census.persistent_bytes)),
            "account external checkpoint residency");
        const auto evaluate_begin = Clock::now();
        require_certificate(
            certificate.evaluate_checkpoint(
                level,
                thrust::raw_pointer_cast(metal.rectangles.data()),
                metal.rectangles.size(), labels.labels, labels.count,
                &checkpoint, diode_view_ptr),
            "evaluate resident antenna checkpoint");
        report_backend_phase(
            index, "checkpoint_evaluate",
            evaluate_begin);
        memory.observe_component_peak(checkpoint.peak_live_bytes);
        memory.observe();
        if (!checkpoint.clean_certificate &&
            checkpoint.uncertain_roots) {
          /*
           * The inexpensive certificate intentionally retains only one
           * gate-intersection witness per owner.  Re-expand POLY and ACTIVE
           * only for an actually uncertain checkpoint, then strengthen that
           * lower bound by summing maxima across disjoint spatial cells.
           * Arbitrary raw overlap within a cell remains max-reduced.
           */
          const auto refine_begin = Clock::now();
          Expansion refined_poly = expander.expand(
              identity.domains[0], 0, owner_bases[0], false);
          Expansion refined_active = expander.expand(
              identity.domains[1], 1, 0, true);
          require_certificate(
              certificate.set_external_live_device_bytes(
                  memory.external_live_bytes(
                      gate_census.persistent_bytes)),
              "account external root-cell refinement residency");
          cert::CheckpointCensus refined_checkpoint;
          require_certificate(
              certificate.evaluate_checkpoint_root_cell_refined(
                  level,
                  thrust::raw_pointer_cast(
                      refined_poly.rectangles.data()),
                  refined_poly.rectangles.size(),
                  thrust::raw_pointer_cast(
                      refined_active.rectangles.data()),
                  refined_active.rectangles.size(),
                  thrust::raw_pointer_cast(
                      metal.rectangles.data()),
                  metal.rectangles.size(), labels.labels,
                  labels.count, &refined_checkpoint,
                  diode_view_ptr, &checkpoint),
              "refine antenna gate lower bound by root cell");
          report_backend_phase(
              index, "checkpoint_refine",
              refine_begin);
          memory.observe_component_peak(
              refined_checkpoint.peak_live_bytes);
          memory.observe();
          checkpoint = refined_checkpoint;
        }
        if (!checkpoint.clean_certificate ||
            checkpoint.uncertain_roots) {
          char message[256];
          std::snprintf(
              message, sizeof(message),
              "antenna checkpoint uncertain: stage=%zu "
              "preliminary=%llu remaining=%llu "
              "gate_records=%llu root_cells=%llu certified=%llu",
              index + 1,
              static_cast<unsigned long long>(
                  checkpoint.preliminary_uncertain_roots),
              static_cast<unsigned long long>(
                  checkpoint.uncertain_roots),
              static_cast<unsigned long long>(
                  checkpoint.refinement_records),
              static_cast<unsigned long long>(
                  checkpoint.refinement_root_cells),
              static_cast<unsigned long long>(
                  checkpoint.ratio_certified_roots));
          throw Decline(
              KLAYOUT_CUDA_SPATIAL_FALLBACK,
              KLAYOUT_CUDA_SPATIAL_FALLBACK_PAIR_WORK_CAPACITY,
              message);
        }
        fill_stage_result(
            index, graph, checkpoint,
            elapsed_ns(stage_begin, Clock::now()), request, result);
        report_backend_phase(
            index, "checkpoint_total",
            checkpoint_begin);
      };

  {
    const auto stage_begin = Clock::now();
    ac::StageCensus graph;
    append_domain(
        &poly, identity.domains[0].flat_polygons,
        (UINT64_C(1) << 4) - 1, &graph, true);
    Expansion diode_contact = expander.expand(
        identity.domains[4], 0, 0, false,
        &diode_contact_template_witnesses);
    Expansion diode_nwell = expander.expand(
        identity.domains[3], 1,
        identity.domains[4].flat_polygons, false);
    filter_factor_zero_diode_contacts(
        request, &memory,
        identity.domains[4].flat_polygons,
        identity.domains[3].flat_polygons,
        &diode_contact, &diode_nwell);
    diode_contact_witnesses =
        std::move(diode_contact.owner_annotations);
    Expansion contact = expander.expand(
        identity.domains[4], 4, owner_bases[4], false);
    append_domain(
        &contact, identity.domains[4].flat_polygons,
        (UINT64_C(1) << 5) - 1, &graph, false);
    Expansion m1 = expander.expand(
        identity.domains[5], 5, owner_bases[5], false);
    append_domain(
        &m1, identity.domains[5].flat_polygons,
        (UINT64_C(1) << 6) - 1, &graph, false);
    checkpoint_stage(
        0, 5, cert::MetalLevel::metal1, graph, stage_begin);
  }

  {
    const auto stage_begin = Clock::now();
    ac::StageCensus graph;
    const auto via1_expand_begin = Clock::now();
    Expansion via1 = expander.expand(
        identity.domains[6], 6, owner_bases[6], false);
    report_backend_phase(
        1, "expand_via1", via1_expand_begin);
    const auto via1_append_begin = Clock::now();
    append_domain(
        &via1, identity.domains[6].flat_polygons,
        (UINT64_C(1) << 7) - 1, &graph, true);
    report_backend_phase(
        1, "append_via1", via1_append_begin);
    const auto m2_expand_begin = Clock::now();
    Expansion m2 = expander.expand(
        identity.domains[7], 7, owner_bases[7], false);
    report_backend_phase(
        1, "expand_m2", m2_expand_begin);
    const auto m2_append_begin = Clock::now();
    append_domain(
        &m2, identity.domains[7].flat_polygons,
        (UINT64_C(1) << 8) - 1, &graph, false);
    report_backend_phase(
        1, "append_m2", m2_append_begin);
    const auto compact_begin = Clock::now();
    require_connectivity(
        connectivity.compact_retained_rectangles(
            &graph.retained_rectangle_capacity),
        "compact post-M2 connectivity frontier");
    report_backend_phase(
        1, "compact_retained", compact_begin);
    memory.observe();
    checkpoint_stage(
        1, 7, cert::MetalLevel::metal2, graph, stage_begin);
  }

  {
    const auto stage_begin = Clock::now();
    ac::StageCensus graph;
    Expansion via2 = expander.expand(
        identity.domains[8], 8, owner_bases[8], false);
    append_domain(
        &via2, identity.domains[8].flat_polygons,
        (UINT64_C(1) << 9) - 1, &graph, true);
    Expansion m3 = expander.expand(
        identity.domains[9], 9, owner_bases[9], false);
    append_domain(
        &m3, identity.domains[9].flat_polygons,
        (UINT64_C(1) << 10) - 1, &graph, false);
    checkpoint_stage(
        2, 9, cert::MetalLevel::metal3, graph, stage_begin);
  }

  {
    const auto stage_begin = Clock::now();
    ac::StageCensus graph;
    Expansion via3 = expander.expand(
        identity.domains[10], 10, owner_bases[10], false);
    append_domain(
        &via3, identity.domains[10].flat_polygons,
        (UINT64_C(1) << 11) - 1, &graph, true);
    Expansion m4 = expander.expand(
        identity.domains[11], 11, owner_bases[11], false);
    append_domain(
        &m4, identity.domains[11].flat_polygons,
        KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_DOMAINS,
        &graph, false);
    checkpoint_stage(
        3, 11, cert::MetalLevel::metal4, graph, stage_begin);
  }

  result->status = KLAYOUT_CUDA_SPATIAL_OK;
  result->fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_COMPLETE;
  result->certified_empty_mask = request.requested_mask;
  result->clean_mask = request.requested_mask;
  result->device_flags = 0;
  result->closed_domain_mask =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_DOMAINS;
  result->released_stage_mask = request.requested_mask;
  /*
   * Adapter and certificate peaks are tracked exactly above.  Connectivity
   * enforces the same all-device cap before every allocation, but does not
   * yet publish its transient peak.  The admitted cap is therefore the
   * conservative transaction-wide upper bound; never under-report a
   * post-checkpoint observation as the peak.
   */
  result->accounted_peak_device_bytes =
      request.capacity.max_device_bytes;
  result->setup_ns = setup_ns;
  result->h2d_ns = h2d_ns;
  result->d2h_ns = d2h_ns;
  result->total_ns = elapsed_ns(total_begin, Clock::now());
  populate_domain_evidence(request, identity, result);
  for (std::size_t index = 0; index < 4; ++index) {
    db::cuda_antenna_m1_m4_evidence::Digest digest;
    if (!db::cuda_antenna_m1_m4_evidence::stage_digest(
            request, *result, index, digest)) {
      internal_decline("stage evidence digest rejected a valid stage");
    }
    std::memcpy(
        result->stages[index].stage_digest,
        digest.data(), digest.size());
  }
  std::snprintf(
      result->message, sizeof(result->message),
      "exact resident CUDA M1-M4 clean certificate");
}

std::mutex &adapter_mutex()
{
  static std::mutex mutex;
  return mutex;
}

void report_decline(Result *result, const Decline &decline)
{
  result->status = decline.status;
  result->fallback_flags = decline.fallback;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_UNCERTAIN;
  std::snprintf(
      result->message, sizeof(result->message), "%s", decline.what());
}

void report_internal(Result *result, const char *message)
{
  result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_UNCERTAIN;
  std::snprintf(
      result->message, sizeof(result->message), "%s", message);
}

}  // namespace

namespace klayout_cuda {
namespace antenna_m1_m4_backend {

bool prepare_test_request(Request *request, const char **error) noexcept
{
  static thread_local std::string last_error;
  if (error) *error = nullptr;
  if (!request) {
    last_error = "request is null";
    if (error) *error = last_error.c_str();
    return false;
  }
  try {
    (void)derive_identity(*request, true);
    return true;
  } catch (const std::exception &exception) {
    last_error = exception.what();
  } catch (...) {
    last_error = "unknown identity preparation failure";
  }
  if (error) *error = last_error.c_str();
  return false;
}

}  // namespace antenna_m1_m4_backend
}  // namespace klayout_cuda

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_antenna_m1_m4_empty_v1(
    const Request *request, Result *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_UNCERTAIN;
  if (!request) {
    std::snprintf(
        result->message, sizeof(result->message), "request is null");
    return result->status;
  }

  std::lock_guard<std::mutex> lock(adapter_mutex());
  const auto total_begin = Clock::now();
  try {
    Request mutable_request = *request;
    const auto setup_begin = Clock::now();
    const Identity identity = derive_identity(mutable_request, false);
    const std::uint64_t setup_ns =
        elapsed_ns(setup_begin, Clock::now());
    fill_result_echo(*request, result);
    run_transaction(
        *request, identity, setup_ns, total_begin, result);
  } catch (const Decline &decline) {
    report_decline(result, decline);
  } catch (const std::exception &exception) {
    report_internal(result, exception.what());
  } catch (...) {
    report_internal(result, "unknown CUDA antenna backend failure");
  }
  if (!result->total_ns) {
    result->total_ns = elapsed_ns(total_begin, Clock::now());
  }
  return result->status;
}
