/*
 * Portable, endian-explicit and self-authenticating capture of the compact
 * ANTENNA.M1-through-M4 request ABI.
 *
 * Version 1 layout (all integers little-endian):
 *
 *   128-byte header
 *   40 canonical 80-byte directory entries
 *   metadata, shared source-cell/context/parent sections
 *   cells/polygons/edges for domains 0 through 11 in role order
 *
 * Every directory entry carries its exact count, wire stride, contiguous
 * range and section SHA-256.  The header SHA-256 covers the header with its
 * digest slot zeroed followed by the complete directory.  Thus the one header
 * digest binds all scalar layout metadata and every section digest.  The
 * declared file extent is exact: gaps, overlaps, truncation, and trailing
 * bytes are all rejected.
 */

#include "antenna_m1_m4_capture_file.h"

#include "dbCudaActive3Digest.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace klayout_cuda {
namespace antenna_m1_m4_capture {
namespace {

using Sha256 = db::cuda_active3_digest::Sha256;
using Digest = std::array<std::uint8_t, 32>;

constexpr std::uint8_t kMagic[8] =
    {'K', 'A', 'M', '4', 'R', 'E', 'Q', '1'};
constexpr std::uint32_t kFileVersion = 1;
constexpr std::uint32_t kMetadataVersion = 1;
constexpr std::uint32_t kEndianMarker = UINT32_C(0x01020304);
constexpr std::uint32_t kHeaderBytes = 128;
constexpr std::uint32_t kDirectoryEntryBytes = 80;
constexpr std::uint32_t kSectionCount = 40;
constexpr std::uint64_t kDirectoryOffset = kHeaderBytes;
constexpr std::uint64_t kDirectoryBytes =
    static_cast<std::uint64_t>(kSectionCount) * kDirectoryEntryBytes;
constexpr std::uint64_t kPayloadOffset =
    kDirectoryOffset + kDirectoryBytes;
constexpr std::uint32_t kNoDomain = UINT32_MAX;
constexpr std::size_t kDomainCount =
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;

enum SectionKind : std::uint32_t
{
  kMetadata = 1,
  kSourceCellIndices = 2,
  kContexts = 3,
  kContextParents = 4,
  kDomainCells = 10,
  kDomainPolygons = 11,
  kDomainEdges = 12
};

constexpr std::uint32_t kU64WireBytes = 8;
constexpr std::uint32_t kContextWireBytes = 24;
constexpr std::uint32_t kCellWireBytes = 24;
constexpr std::uint32_t kPolygonWireBytes = 48;
constexpr std::uint32_t kEdgeWireBytes = 32;

constexpr std::uint32_t kPhysicalLayers[kDomainCount] = {
    9, 1, 4, 3, 10, 11, 12, 13, 14, 15, 16, 17};

constexpr char kDigestDomains[kDomainCount][9] = {
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

struct Section
{
  std::uint32_t kind = 0;
  std::uint32_t domain = kNoDomain;
  std::uint64_t count = 0;
  std::uint32_t record_bytes = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  Digest digest{};
};

[[noreturn]] void fail (const std::string &message)
{
  throw std::runtime_error(message);
}

bool checked_add (std::uint64_t first, std::uint64_t second,
                  std::uint64_t &result)
{
  if (second > UINT64_MAX - first) return false;
  result = first + second;
  return true;
}

bool checked_multiply (std::uint64_t first, std::uint64_t second,
                       std::uint64_t &result)
{
  if (first && second > UINT64_MAX / first) return false;
  result = first * second;
  return true;
}

std::uint64_t add_or_fail (std::uint64_t first, std::uint64_t second,
                           const char *what)
{
  std::uint64_t result = 0;
  if (!checked_add(first, second, result)) {
    fail(std::string(what) + " overflows uint64");
  }
  return result;
}

std::uint64_t multiply_or_fail (std::uint64_t first, std::uint64_t second,
                                const char *what)
{
  std::uint64_t result = 0;
  if (!checked_multiply(first, second, result)) {
    fail(std::string(what) + " overflows uint64");
  }
  return result;
}

bool bytes_zero (const void *data, std::size_t bytes)
{
  const auto *value = static_cast<const std::uint8_t *>(data);
  for (std::size_t index = 0; index < bytes; ++index) {
    if (value[index]) return false;
  }
  return true;
}

template <class T>
bool count_fits_vector (std::uint64_t count)
{
  return count <=
         static_cast<std::uint64_t>(
             std::numeric_limits<std::size_t>::max() / sizeof(T));
}

template <class T>
const T *nonnull_data (const std::vector<T> &values)
{
  return values.empty() ? nullptr : values.data();
}

template <class T>
T *nonnull_data (std::vector<T> &values)
{
  return values.empty() ? nullptr : values.data();
}

class Encoder
{
public:
  explicit Encoder (std::vector<std::uint8_t> *bytes = nullptr,
                    std::ostream *stream = nullptr, Sha256 *sha = nullptr)
      : m_bytes(bytes), m_stream(stream), m_sha(sha), m_buffer(),
        m_buffer_bytes(0)
  {
  }

  void raw (const void *source, std::size_t bytes)
  {
    if (!bytes) return;
    if (m_bytes) {
      const auto *begin = static_cast<const std::uint8_t *>(source);
      m_bytes->insert(m_bytes->end(), begin, begin + bytes);
    }
    if (!m_stream && !m_sha) return;
    const auto *input = static_cast<const std::uint8_t *>(source);
    while (bytes) {
      const std::size_t available = m_buffer.size() - m_buffer_bytes;
      const std::size_t take = std::min(available, bytes);
      std::memcpy(m_buffer.data() + m_buffer_bytes, input, take);
      m_buffer_bytes += take;
      input += take;
      bytes -= take;
      if (m_buffer_bytes == m_buffer.size()) flush();
    }
  }

  void finish ()
  {
    flush();
  }

  void u8 (std::uint8_t value) { raw(&value, 1); }

  void u32 (std::uint32_t value)
  {
    std::uint8_t bytes[4];
    for (unsigned int index = 0; index < 4; ++index) {
      bytes[index] = static_cast<std::uint8_t>(value >> (8 * index));
    }
    raw(bytes, sizeof(bytes));
  }

  void i32 (std::int32_t value)
  {
    u32(static_cast<std::uint32_t>(value));
  }

  void u64 (std::uint64_t value)
  {
    std::uint8_t bytes[8];
    for (unsigned int index = 0; index < 8; ++index) {
      bytes[index] = static_cast<std::uint8_t>(value >> (8 * index));
    }
    raw(bytes, sizeof(bytes));
  }

  void i64 (std::int64_t value)
  {
    u64(static_cast<std::uint64_t>(value));
  }

private:
  void flush ()
  {
    if (!m_buffer_bytes) return;
    if (m_stream) {
      m_stream->write(
          reinterpret_cast<const char *>(m_buffer.data()),
          static_cast<std::streamsize>(m_buffer_bytes));
      if (!*m_stream) fail("capture write failed");
    }
    if (m_sha) m_sha->update(m_buffer.data(), m_buffer_bytes);
    m_buffer_bytes = 0;
  }

  std::vector<std::uint8_t> *m_bytes;
  std::ostream *m_stream;
  Sha256 *m_sha;
  std::array<std::uint8_t, 65536> m_buffer;
  std::size_t m_buffer_bytes;
};

class Cursor
{
public:
  Cursor (const std::uint8_t *data, std::size_t bytes)
      : m_data(data), m_bytes(bytes), m_offset(0)
  {
  }

  void raw (void *target, std::size_t bytes)
  {
    if (bytes > m_bytes - m_offset) fail("truncated encoded record");
    std::memcpy(target, m_data + m_offset, bytes);
    m_offset += bytes;
  }

  std::uint8_t u8 ()
  {
    std::uint8_t value = 0;
    raw(&value, 1);
    return value;
  }

  std::uint32_t u32 ()
  {
    std::uint8_t bytes[4];
    raw(bytes, sizeof(bytes));
    std::uint32_t value = 0;
    for (unsigned int index = 0; index < 4; ++index) {
      value |= static_cast<std::uint32_t>(bytes[index]) << (8 * index);
    }
    return value;
  }

  std::int32_t i32 ()
  {
    const std::uint32_t value = u32();
    if (value <= static_cast<std::uint32_t>(INT32_MAX)) {
      return static_cast<std::int32_t>(value);
    }
    return static_cast<std::int32_t>(
        -1 - static_cast<std::int64_t>(UINT32_MAX - value));
  }

  std::uint64_t u64 ()
  {
    std::uint8_t bytes[8];
    raw(bytes, sizeof(bytes));
    std::uint64_t value = 0;
    for (unsigned int index = 0; index < 8; ++index) {
      value |= static_cast<std::uint64_t>(bytes[index]) << (8 * index);
    }
    return value;
  }

  std::int64_t i64 ()
  {
    const std::uint64_t value = u64();
    if (value <= static_cast<std::uint64_t>(INT64_MAX)) {
      return static_cast<std::int64_t>(value);
    }
    return -1 - static_cast<std::int64_t>(UINT64_MAX - value);
  }

  bool empty () const { return m_offset == m_bytes; }
  std::size_t remaining () const { return m_bytes - m_offset; }

private:
  const std::uint8_t *m_data;
  std::size_t m_bytes;
  std::size_t m_offset;
};

template <class T>
T load_record (const void *base, std::uint64_t index,
               std::uint32_t stride)
{
  if (!base || stride != sizeof(T) ||
      index > std::numeric_limits<std::size_t>::max() / stride) {
    fail("record pointer, stride, or index is invalid");
  }
  T value;
  const auto *bytes = static_cast<const std::uint8_t *>(base);
  std::memcpy(&value, bytes + static_cast<std::size_t>(index) * stride,
              sizeof(value));
  return value;
}

void encode_context (Encoder &encoder, const Context &context)
{
  encoder.i64(context.tx);
  encoder.i64(context.ty);
  encoder.u32(context.cell_id);
  encoder.u32(context.transform_code);
}

void encode_cell (Encoder &encoder, const Cell &cell)
{
  encoder.u64(cell.polygon_begin);
  encoder.u64(cell.edge_begin);
  encoder.u32(cell.polygon_count);
  encoder.u32(cell.edge_count);
}

void encode_polygon (Encoder &encoder, const Polygon &polygon)
{
  encoder.u64(polygon.edge_begin);
  encoder.i64(polygon.left);
  encoder.i64(polygon.bottom);
  encoder.i64(polygon.right);
  encoder.i64(polygon.top);
  encoder.u32(polygon.polygon_id);
  encoder.u32(polygon.edge_count);
}

void encode_edge (Encoder &encoder, const Edge &edge)
{
  encoder.i64(edge.x1);
  encoder.i64(edge.y1);
  encoder.i64(edge.x2);
  encoder.i64(edge.y2);
}

Context decode_context (Cursor &cursor)
{
  Context value{};
  value.tx = cursor.i64();
  value.ty = cursor.i64();
  value.cell_id = cursor.u32();
  value.transform_code = cursor.u32();
  return value;
}

Cell decode_cell (Cursor &cursor)
{
  Cell value{};
  value.polygon_begin = cursor.u64();
  value.edge_begin = cursor.u64();
  value.polygon_count = cursor.u32();
  value.edge_count = cursor.u32();
  return value;
}

Polygon decode_polygon (Cursor &cursor)
{
  Polygon value{};
  value.edge_begin = cursor.u64();
  value.left = cursor.i64();
  value.bottom = cursor.i64();
  value.right = cursor.i64();
  value.top = cursor.i64();
  value.polygon_id = cursor.u32();
  value.edge_count = cursor.u32();
  return value;
}

Edge decode_edge (Cursor &cursor)
{
  Edge value{};
  value.x1 = cursor.i64();
  value.y1 = cursor.i64();
  value.x2 = cursor.i64();
  value.y2 = cursor.i64();
  return value;
}

void encode_metadata (Encoder &encoder, const Request &request)
{
  encoder.u32(kMetadataVersion);
  encoder.u32(static_cast<std::uint32_t>(kDomainCount));

  encoder.u32(request.abi_version);
  encoder.u32(request.struct_size);
  encoder.u32(request.opcode);
  encoder.u32(request.option_flags);
  encoder.u32(request.format_version);
  encoder.u32(request.dbu_per_micron);
  encoder.u32(request.requested_mask);
  encoder.u32(request.stage_count);
  encoder.u32(request.ratio_numerator);
  encoder.u32(request.ratio_denominator);
  encoder.u32(request.domain_count);
  encoder.i32(request.device);

  const auto &hierarchy = request.hierarchy;
  encoder.u32(hierarchy.struct_size);
  encoder.u32(hierarchy.format_version);
  encoder.u32(hierarchy.dbu_per_micron);
  encoder.u32(hierarchy.root_cell);
  encoder.u64(hierarchy.source_root_cell_index);
  encoder.u64(hierarchy.source_cell_count);
  encoder.u32(hierarchy.source_cell_index_record_bytes);
  encoder.u64(hierarchy.context_count);
  encoder.u32(hierarchy.context_record_bytes);
  encoder.u64(hierarchy.context_parent_count);
  encoder.u32(hierarchy.context_parent_record_bytes);
  encoder.raw(hierarchy.hierarchy_digest, sizeof(hierarchy.hierarchy_digest));

  for (std::size_t index = 0; index < kDomainCount; ++index) {
    const auto &domain = request.domains[index];
    encoder.u32(domain.struct_size);
    encoder.u32(domain.role);
    encoder.u32(domain.physical_layer);
    encoder.u32(domain.datatype);
    encoder.u32(domain.source_layer_index);
    encoder.u64(domain.cell_count);
    encoder.u32(domain.cell_record_bytes);
    encoder.u64(domain.polygon_count);
    encoder.u32(domain.polygon_record_bytes);
    encoder.u64(domain.edge_count);
    encoder.u32(domain.edge_record_bytes);
    encoder.u64(domain.nonempty_context_count);
    encoder.u64(domain.flat_polygon_count);
    encoder.u64(domain.flat_edge_count);
    encoder.u64(domain.stored_bytes);
    encoder.u64(domain.expanded_geometry_bytes);
    encoder.i64(domain.scene_left);
    encoder.i64(domain.scene_bottom);
    encoder.i64(domain.scene_right);
    encoder.i64(domain.scene_top);
    encoder.raw(domain.digest_domain, sizeof(domain.digest_domain));
    encoder.raw(domain.scene_digest, sizeof(domain.scene_digest));
  }

  const auto &census = request.census;
  encoder.u32(census.struct_size);
  encoder.u32(census.format_version);
  encoder.u64(census.shared_cell_count);
  encoder.u64(census.shared_context_count);
  encoder.u64(census.context_parent_record_count);
  encoder.u64(census.stored_cell_record_count);
  encoder.u64(census.stored_polygon_count);
  encoder.u64(census.stored_edge_count);
  encoder.u64(census.expanded_polygon_count);
  encoder.u64(census.expanded_edge_count);
  encoder.u64(census.total_stored_bytes);
  encoder.u64(census.total_expanded_geometry_bytes);
  encoder.u64(census.estimated_peak_bytes);

  const auto &capacity = request.capacity;
  encoder.u32(capacity.struct_size);
  encoder.u64(capacity.max_cells);
  encoder.u64(capacity.max_contexts);
  encoder.u64(capacity.max_stored_polygons);
  encoder.u64(capacity.max_stored_edges);
  encoder.u64(capacity.max_flat_polygons);
  encoder.u64(capacity.max_flat_edges);
  encoder.u64(capacity.max_total_stored_bytes);
  encoder.u64(capacity.max_total_expanded_geometry_bytes);
  encoder.u64(capacity.max_estimated_peak_bytes);
  encoder.u64(capacity.max_nodes);
  encoder.u64(capacity.max_rectangles);
  encoder.u64(capacity.max_memberships);
  encoder.u64(capacity.max_pair_occurrences);
  encoder.u64(capacity.max_unique_candidates);
  encoder.u64(capacity.max_cell_members);
  encoder.u64(capacity.max_dsu_iterations);
  encoder.u64(capacity.max_rule_work);
  encoder.u64(capacity.max_device_bytes);

  encoder.raw(request.lower_capture_digest,
              sizeof(request.lower_capture_digest));
  encoder.raw(request.capture_digest, sizeof(request.capture_digest));
}

void decode_metadata (Cursor &cursor, Request &request)
{
  if (cursor.u32() != kMetadataVersion) {
    fail("unknown metadata version");
  }
  if (cursor.u32() != kDomainCount) {
    fail("metadata domain count is not twelve");
  }

  std::memset(&request, 0, sizeof(request));
  request.abi_version = cursor.u32();
  request.struct_size = cursor.u32();
  request.opcode = cursor.u32();
  request.option_flags = cursor.u32();
  request.format_version = cursor.u32();
  request.dbu_per_micron = cursor.u32();
  request.requested_mask = cursor.u32();
  request.stage_count = cursor.u32();
  request.ratio_numerator = cursor.u32();
  request.ratio_denominator = cursor.u32();
  request.domain_count = cursor.u32();
  request.device = cursor.i32();

  auto &hierarchy = request.hierarchy;
  hierarchy.struct_size = cursor.u32();
  hierarchy.format_version = cursor.u32();
  hierarchy.dbu_per_micron = cursor.u32();
  hierarchy.root_cell = cursor.u32();
  hierarchy.source_root_cell_index = cursor.u64();
  hierarchy.source_cell_count = cursor.u64();
  hierarchy.source_cell_index_record_bytes = cursor.u32();
  hierarchy.context_count = cursor.u64();
  hierarchy.context_record_bytes = cursor.u32();
  hierarchy.context_parent_count = cursor.u64();
  hierarchy.context_parent_record_bytes = cursor.u32();
  cursor.raw(hierarchy.hierarchy_digest, sizeof(hierarchy.hierarchy_digest));

  for (std::size_t index = 0; index < kDomainCount; ++index) {
    auto &domain = request.domains[index];
    domain.struct_size = cursor.u32();
    domain.role = cursor.u32();
    domain.physical_layer = cursor.u32();
    domain.datatype = cursor.u32();
    domain.source_layer_index = cursor.u32();
    domain.cell_count = cursor.u64();
    domain.cell_record_bytes = cursor.u32();
    domain.polygon_count = cursor.u64();
    domain.polygon_record_bytes = cursor.u32();
    domain.edge_count = cursor.u64();
    domain.edge_record_bytes = cursor.u32();
    domain.nonempty_context_count = cursor.u64();
    domain.flat_polygon_count = cursor.u64();
    domain.flat_edge_count = cursor.u64();
    domain.stored_bytes = cursor.u64();
    domain.expanded_geometry_bytes = cursor.u64();
    domain.scene_left = cursor.i64();
    domain.scene_bottom = cursor.i64();
    domain.scene_right = cursor.i64();
    domain.scene_top = cursor.i64();
    cursor.raw(domain.digest_domain, sizeof(domain.digest_domain));
    cursor.raw(domain.scene_digest, sizeof(domain.scene_digest));
  }

  auto &census = request.census;
  census.struct_size = cursor.u32();
  census.format_version = cursor.u32();
  census.shared_cell_count = cursor.u64();
  census.shared_context_count = cursor.u64();
  census.context_parent_record_count = cursor.u64();
  census.stored_cell_record_count = cursor.u64();
  census.stored_polygon_count = cursor.u64();
  census.stored_edge_count = cursor.u64();
  census.expanded_polygon_count = cursor.u64();
  census.expanded_edge_count = cursor.u64();
  census.total_stored_bytes = cursor.u64();
  census.total_expanded_geometry_bytes = cursor.u64();
  census.estimated_peak_bytes = cursor.u64();

  auto &capacity = request.capacity;
  capacity.struct_size = cursor.u32();
  capacity.max_cells = cursor.u64();
  capacity.max_contexts = cursor.u64();
  capacity.max_stored_polygons = cursor.u64();
  capacity.max_stored_edges = cursor.u64();
  capacity.max_flat_polygons = cursor.u64();
  capacity.max_flat_edges = cursor.u64();
  capacity.max_total_stored_bytes = cursor.u64();
  capacity.max_total_expanded_geometry_bytes = cursor.u64();
  capacity.max_estimated_peak_bytes = cursor.u64();
  capacity.max_nodes = cursor.u64();
  capacity.max_rectangles = cursor.u64();
  capacity.max_memberships = cursor.u64();
  capacity.max_pair_occurrences = cursor.u64();
  capacity.max_unique_candidates = cursor.u64();
  capacity.max_cell_members = cursor.u64();
  capacity.max_dsu_iterations = cursor.u64();
  capacity.max_rule_work = cursor.u64();
  capacity.max_device_bytes = cursor.u64();

  cursor.raw(request.lower_capture_digest,
             sizeof(request.lower_capture_digest));
  cursor.raw(request.capture_digest, sizeof(request.capture_digest));
  if (!cursor.empty()) fail("metadata has trailing bytes");
}

std::vector<std::uint8_t> metadata_bytes (const Request &request)
{
  std::vector<std::uint8_t> bytes;
  bytes.reserve(2600);
  Encoder encoder(&bytes);
  encode_metadata(encoder, request);
  return bytes;
}

void validate_request (const Request &request)
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
      request.domain_count != kDomainCount || request.device < 0 ||
      !bytes_zero(request.reserved, sizeof(request.reserved))) {
    fail("request header is not the qualified ANTENNA.M1-M4 v1 identity");
  }

  const auto &hierarchy = request.hierarchy;
  if (hierarchy.struct_size != sizeof(hierarchy) ||
      hierarchy.format_version != 2 ||
      hierarchy.dbu_per_micron != request.dbu_per_micron ||
      hierarchy.source_cell_count == 0 ||
      hierarchy.source_cell_count > UINT32_MAX ||
      hierarchy.root_cell >= hierarchy.source_cell_count ||
      !hierarchy.source_cell_indices ||
      hierarchy.source_cell_index_record_bytes != sizeof(std::uint64_t) ||
      hierarchy.context_count == 0 || hierarchy.context_count > UINT32_MAX ||
      !hierarchy.contexts ||
      hierarchy.context_record_bytes != sizeof(Context) ||
      !hierarchy.context_parent_ids ||
      hierarchy.context_parent_count != hierarchy.context_count ||
      hierarchy.context_parent_record_bytes != sizeof(std::uint32_t) ||
      hierarchy.reserved0 || hierarchy.reserved1 || hierarchy.reserved2 ||
      !bytes_zero(hierarchy.reserved3, sizeof(hierarchy.reserved3)) ||
      !count_fits_vector<std::uint64_t>(hierarchy.source_cell_count) ||
      !count_fits_vector<Context>(hierarchy.context_count) ||
      !count_fits_vector<std::uint32_t>(hierarchy.context_parent_count)) {
    fail("request hierarchy is not qualified");
  }

  const auto &capacity = request.capacity;
  if (capacity.struct_size != sizeof(capacity) || capacity.reserved0 ||
      !bytes_zero(capacity.reserved1, sizeof(capacity.reserved1)) ||
      !capacity.max_cells || !capacity.max_contexts ||
      !capacity.max_stored_polygons || !capacity.max_stored_edges ||
      !capacity.max_flat_polygons || !capacity.max_flat_edges ||
      !capacity.max_total_stored_bytes ||
      !capacity.max_total_expanded_geometry_bytes ||
      !capacity.max_estimated_peak_bytes || !capacity.max_nodes ||
      !capacity.max_rectangles || !capacity.max_memberships ||
      !capacity.max_pair_occurrences ||
      !capacity.max_unique_candidates || !capacity.max_cell_members ||
      !capacity.max_dsu_iterations || !capacity.max_rule_work ||
      !capacity.max_device_bytes || capacity.max_cells > UINT32_MAX ||
      capacity.max_contexts > UINT32_MAX ||
      capacity.max_nodes > UINT32_MAX ||
      capacity.max_rectangles > UINT32_MAX ||
      capacity.max_cell_members > UINT32_MAX ||
      capacity.max_dsu_iterations > UINT32_MAX) {
    fail("request capacity is not qualified");
  }

  if (hierarchy.source_cell_count > capacity.max_cells ||
      hierarchy.context_count > capacity.max_contexts) {
    fail("hierarchy counts exceed request capacity");
  }

  std::uint64_t stored_cells = 0;
  std::uint64_t stored_polygons = 0;
  std::uint64_t stored_edges = 0;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_edges = 0;
  std::uint64_t domain_stored_bytes = 0;
  std::uint64_t expanded_bytes = 0;
  std::set<std::uint32_t> source_layers;
  for (std::size_t index = 0; index < kDomainCount; ++index) {
    const auto &domain = request.domains[index];
    const std::uint64_t minimum_edges =
        multiply_or_fail(domain.polygon_count, 4, "source edge minimum");
    const std::uint64_t minimum_flat_edges =
        multiply_or_fail(domain.flat_polygon_count, 4,
                         "expanded edge minimum");
    if (domain.struct_size != sizeof(domain) || domain.role != index ||
        domain.physical_layer != kPhysicalLayers[index] ||
        domain.datatype != 0 || domain.reserved0 || domain.reserved1 ||
        domain.reserved2 || domain.reserved3 ||
        !bytes_zero(domain.reserved4, sizeof(domain.reserved4)) ||
        std::memcmp(domain.digest_domain, kDigestDomains[index], 8) != 0 ||
        !source_layers.insert(domain.source_layer_index).second ||
        !domain.cells ||
        domain.cell_count != hierarchy.source_cell_count ||
        domain.cell_count > capacity.max_cells ||
        domain.cell_record_bytes != sizeof(Cell) || !domain.polygons ||
        !domain.polygon_count ||
        domain.polygon_count > capacity.max_stored_polygons ||
        domain.polygon_count > UINT32_MAX ||
        domain.polygon_record_bytes != sizeof(Polygon) || !domain.edges ||
        domain.edge_count < minimum_edges ||
        domain.edge_count > capacity.max_stored_edges ||
        domain.edge_count > UINT32_MAX ||
        domain.edge_record_bytes != sizeof(Edge) ||
        !domain.nonempty_context_count ||
        domain.nonempty_context_count > hierarchy.context_count ||
        domain.flat_polygon_count < domain.polygon_count ||
        domain.flat_polygon_count > capacity.max_flat_polygons ||
        domain.flat_polygon_count > UINT32_MAX ||
        domain.flat_edge_count < minimum_flat_edges ||
        domain.flat_edge_count > capacity.max_flat_edges ||
        domain.flat_edge_count > UINT32_MAX || !domain.stored_bytes ||
        !domain.expanded_geometry_bytes ||
        domain.scene_left >= domain.scene_right ||
        domain.scene_bottom >= domain.scene_top ||
        !count_fits_vector<Cell>(domain.cell_count) ||
        !count_fits_vector<Polygon>(domain.polygon_count) ||
        !count_fits_vector<Edge>(domain.edge_count)) {
      std::ostringstream message;
      message << "domain " << index << " is not qualified";
      fail(message.str());
    }
    stored_cells =
        add_or_fail(stored_cells, domain.cell_count, "stored cell census");
    stored_polygons = add_or_fail(
        stored_polygons, domain.polygon_count, "stored polygon census");
    stored_edges =
        add_or_fail(stored_edges, domain.edge_count, "stored edge census");
    flat_polygons = add_or_fail(
        flat_polygons, domain.flat_polygon_count, "flat polygon census");
    flat_edges =
        add_or_fail(flat_edges, domain.flat_edge_count, "flat edge census");
    domain_stored_bytes = add_or_fail(
        domain_stored_bytes, domain.stored_bytes, "stored byte census");
    expanded_bytes = add_or_fail(
        expanded_bytes, domain.expanded_geometry_bytes,
        "expanded byte census");
  }

  const auto &census = request.census;
  const std::uint64_t estimated_peak =
      add_or_fail(census.total_stored_bytes,
                  census.total_expanded_geometry_bytes,
                  "estimated peak census");
  if (census.struct_size != sizeof(census) ||
      census.format_version != request.format_version ||
      !bytes_zero(census.reserved, sizeof(census.reserved)) ||
      census.shared_cell_count != hierarchy.source_cell_count ||
      census.shared_context_count != hierarchy.context_count ||
      census.context_parent_record_count !=
          hierarchy.context_parent_count ||
      census.stored_cell_record_count != stored_cells ||
      census.stored_polygon_count != stored_polygons ||
      census.stored_edge_count != stored_edges ||
      census.expanded_polygon_count != flat_polygons ||
      census.expanded_edge_count != flat_edges ||
      census.total_stored_bytes < domain_stored_bytes ||
      census.total_expanded_geometry_bytes != expanded_bytes ||
      census.estimated_peak_bytes != estimated_peak ||
      census.expanded_polygon_count > capacity.max_nodes ||
      census.total_stored_bytes > capacity.max_total_stored_bytes ||
      census.total_expanded_geometry_bytes >
          capacity.max_total_expanded_geometry_bytes ||
      census.estimated_peak_bytes > capacity.max_estimated_peak_bytes) {
    fail("aggregate request census is not qualified");
  }
}

std::vector<Section> make_sections (const Request &request,
                                    std::size_t metadata_size)
{
  std::vector<Section> sections;
  sections.reserve(kSectionCount);
  sections.push_back(
      {kMetadata, kNoDomain, 1, static_cast<std::uint32_t>(metadata_size)});
  sections.push_back(
      {kSourceCellIndices, kNoDomain,
       request.hierarchy.source_cell_count, kU64WireBytes});
  sections.push_back(
      {kContexts, kNoDomain, request.hierarchy.context_count,
       kContextWireBytes});
  sections.push_back(
      {kContextParents, kNoDomain,
       request.hierarchy.context_parent_count, 4});
  for (std::size_t index = 0; index < kDomainCount; ++index) {
    const auto &domain = request.domains[index];
    sections.push_back(
        {kDomainCells, static_cast<std::uint32_t>(index),
         domain.cell_count, kCellWireBytes});
    sections.push_back(
        {kDomainPolygons, static_cast<std::uint32_t>(index),
         domain.polygon_count, kPolygonWireBytes});
    sections.push_back(
        {kDomainEdges, static_cast<std::uint32_t>(index),
         domain.edge_count, kEdgeWireBytes});
  }
  if (sections.size() != kSectionCount) {
    fail("internal section-count invariant failed");
  }

  std::uint64_t offset = kPayloadOffset;
  for (auto &section : sections) {
    section.bytes =
        multiply_or_fail(section.count, section.record_bytes,
                         "section byte count");
    section.offset = offset;
    offset = add_or_fail(offset, section.bytes, "capture file size");
  }
  return sections;
}

void encode_section (Encoder &encoder, const Request &request,
                     const std::vector<std::uint8_t> &metadata,
                     const Section &section)
{
  if (section.kind == kMetadata) {
    encoder.raw(metadata.data(), metadata.size());
    return;
  }
  if (section.kind == kSourceCellIndices) {
    for (std::uint64_t index = 0; index < section.count; ++index) {
      encoder.u64(request.hierarchy.source_cell_indices[index]);
    }
    return;
  }
  if (section.kind == kContexts) {
    for (std::uint64_t index = 0; index < section.count; ++index) {
      encode_context(
          encoder, load_record<Context>(
                       request.hierarchy.contexts, index,
                       request.hierarchy.context_record_bytes));
    }
    return;
  }
  if (section.kind == kContextParents) {
    for (std::uint64_t index = 0; index < section.count; ++index) {
      encoder.u32(request.hierarchy.context_parent_ids[index]);
    }
    return;
  }
  if (section.domain >= kDomainCount) {
    fail("section has an invalid domain");
  }
  const auto &domain = request.domains[section.domain];
  if (section.kind == kDomainCells) {
    for (std::uint64_t index = 0; index < section.count; ++index) {
      encode_cell(
          encoder,
          load_record<Cell>(domain.cells, index, domain.cell_record_bytes));
    }
  } else if (section.kind == kDomainPolygons) {
    for (std::uint64_t index = 0; index < section.count; ++index) {
      encode_polygon(
          encoder, load_record<Polygon>(
                       domain.polygons, index,
                       domain.polygon_record_bytes));
    }
  } else if (section.kind == kDomainEdges) {
    for (std::uint64_t index = 0; index < section.count; ++index) {
      encode_edge(
          encoder,
          load_record<Edge>(domain.edges, index, domain.edge_record_bytes));
    }
  } else {
    fail("unknown section kind");
  }
}

std::vector<std::uint8_t> encode_directory (
    const std::vector<Section> &sections)
{
  std::vector<std::uint8_t> bytes;
  bytes.reserve(static_cast<std::size_t>(kDirectoryBytes));
  Encoder encoder(&bytes);
  for (const auto &section : sections) {
    encoder.u32(section.kind);
    encoder.u32(section.domain);
    encoder.u64(section.count);
    encoder.u32(section.record_bytes);
    encoder.u32(0);
    encoder.u64(section.offset);
    encoder.u64(section.bytes);
    encoder.raw(section.digest.data(), section.digest.size());
    encoder.u64(0);
  }
  if (bytes.size() != kDirectoryBytes) {
    fail("internal directory-size invariant failed");
  }
  return bytes;
}

std::vector<std::uint8_t> encode_header (
    std::uint64_t file_bytes, const Digest &header_digest)
{
  std::vector<std::uint8_t> bytes;
  bytes.reserve(kHeaderBytes);
  Encoder encoder(&bytes);
  encoder.raw(kMagic, sizeof(kMagic));
  encoder.u32(kFileVersion);
  encoder.u32(kHeaderBytes);
  encoder.u32(kEndianMarker);
  encoder.u32(kSectionCount);
  encoder.u32(kDirectoryEntryBytes);
  encoder.u32(0);
  encoder.u64(kDirectoryOffset);
  encoder.u64(kDirectoryBytes);
  encoder.u64(kPayloadOffset);
  encoder.u64(file_bytes);
  encoder.raw(header_digest.data(), header_digest.size());
  for (unsigned int index = 0; index < 32; ++index) encoder.u8(0);
  if (bytes.size() != kHeaderBytes) {
    fail("internal header-size invariant failed");
  }
  return bytes;
}

Digest header_digest (const std::vector<std::uint8_t> &zeroed_header,
                      const std::vector<std::uint8_t> &directory)
{
  Sha256 sha;
  sha.update(zeroed_header.data(), zeroed_header.size());
  sha.update(directory.data(), directory.size());
  return sha.finish();
}

Digest digest_bytes (const std::vector<std::uint8_t> &bytes)
{
  Sha256 sha;
  if (!bytes.empty()) sha.update(bytes.data(), bytes.size());
  return sha.finish();
}

std::vector<std::uint8_t> read_exact (std::istream &stream,
                                      std::uint64_t count,
                                      const char *what)
{
  if (count > std::numeric_limits<std::size_t>::max() ||
      count > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamsize>::max())) {
    fail(std::string(what) + " is too large for this host");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(count));
  if (count) {
    stream.read(reinterpret_cast<char *>(bytes.data()),
                static_cast<std::streamsize>(count));
    if (stream.gcount() != static_cast<std::streamsize>(count)) {
      fail(std::string("truncated ") + what);
    }
  }
  return bytes;
}

std::vector<Section> decode_directory (
    const std::vector<std::uint8_t> &bytes)
{
  Cursor cursor(bytes.data(), bytes.size());
  std::vector<Section> sections;
  sections.reserve(kSectionCount);
  for (std::size_t index = 0; index < kSectionCount; ++index) {
    Section section;
    section.kind = cursor.u32();
    section.domain = cursor.u32();
    section.count = cursor.u64();
    section.record_bytes = cursor.u32();
    const std::uint32_t flags = cursor.u32();
    section.offset = cursor.u64();
    section.bytes = cursor.u64();
    cursor.raw(section.digest.data(), section.digest.size());
    const std::uint64_t reserved = cursor.u64();
    if (flags || reserved) fail("directory reserved fields are nonzero");
    sections.push_back(section);
  }
  if (!cursor.empty()) fail("directory has trailing bytes");
  return sections;
}

void require_section (const Section &section, std::uint32_t kind,
                      std::uint32_t domain, std::uint32_t record_bytes,
                      std::uint64_t count, std::size_t index)
{
  if (section.kind != kind || section.domain != domain ||
      section.record_bytes != record_bytes || section.count != count) {
    std::ostringstream message;
    message << "section " << index << " descriptor disagrees with metadata";
    fail(message.str());
  }
}

template <class T, class Decode>
void decode_records (const std::vector<std::uint8_t> &bytes,
                     std::uint64_t count, std::uint32_t record_bytes,
                     std::vector<T> &output, Decode decode)
{
  if (!count_fits_vector<T>(count)) {
    fail("section record count exceeds host address space");
  }
  const std::uint64_t expected =
      multiply_or_fail(count, record_bytes, "decoded section bytes");
  if (expected != bytes.size()) {
    fail("decoded section byte count is inconsistent");
  }
  output.resize(static_cast<std::size_t>(count));
  Cursor cursor(bytes.data(), bytes.size());
  for (std::size_t index = 0; index < output.size(); ++index) {
    output[index] = decode(cursor);
  }
  if (!cursor.empty()) fail("record section has trailing bytes");
}

void verify_section_digest (const Section &section,
                            const std::vector<std::uint8_t> &bytes,
                            std::size_t index)
{
  if (digest_bytes(bytes) != section.digest) {
    std::ostringstream message;
    message << "section " << index << " SHA-256 mismatch";
    fail(message.str());
  }
}

std::uint64_t stream_file_size (std::ifstream &stream)
{
  stream.seekg(0, std::ios::end);
  if (!stream) fail("unable to seek capture file");
  const std::streamoff end = stream.tellg();
  if (end < 0) fail("unable to determine capture file size");
  stream.seekg(0, std::ios::beg);
  if (!stream) fail("unable to rewind capture file");
  return static_cast<std::uint64_t>(end);
}

std::string errno_message (const char *operation,
                           const std::string &path)
{
  std::ostringstream message;
  message << operation << " '" << path << "'";
  if (errno) message << ": " << std::strerror(errno);
  return message.str();
}

void set_error (std::string *error, const char *message) noexcept
{
  if (!error) return;
  try {
    *error = message ? message : "unknown capture-file error";
  } catch (...) {
    error->clear();
  }
}

}  // namespace

OwnedRequest::OwnedRequest () noexcept
    : request{}, source_cell_indices(), contexts(), context_parent_ids(),
      domains()
{
}

OwnedRequest::OwnedRequest (const OwnedRequest &other)
    : request(other.request),
      source_cell_indices(other.source_cell_indices),
      contexts(other.contexts),
      context_parent_ids(other.context_parent_ids),
      domains(other.domains)
{
  rebind();
}

OwnedRequest::OwnedRequest (OwnedRequest &&other) noexcept
    : request(other.request),
      source_cell_indices(std::move(other.source_cell_indices)),
      contexts(std::move(other.contexts)),
      context_parent_ids(std::move(other.context_parent_ids)),
      domains(std::move(other.domains))
{
  rebind();
  other.rebind();
}

OwnedRequest &OwnedRequest::operator= (const OwnedRequest &other)
{
  if (this != &other) {
    OwnedRequest replacement(other);
    *this = std::move(replacement);
  }
  return *this;
}

OwnedRequest &OwnedRequest::operator= (OwnedRequest &&other) noexcept
{
  if (this != &other) {
    request = other.request;
    source_cell_indices = std::move(other.source_cell_indices);
    contexts = std::move(other.contexts);
    context_parent_ids = std::move(other.context_parent_ids);
    domains = std::move(other.domains);
    rebind();
    other.rebind();
  }
  return *this;
}

void OwnedRequest::rebind () noexcept
{
  request.hierarchy.source_cell_indices = nonnull_data(source_cell_indices);
  request.hierarchy.contexts = nonnull_data(contexts);
  request.hierarchy.context_parent_ids = nonnull_data(context_parent_ids);
  for (std::size_t index = 0; index < kDomainCount; ++index) {
    request.domains[index].cells = nonnull_data(domains[index].cells);
    request.domains[index].polygons =
        nonnull_data(domains[index].polygons);
    request.domains[index].edges = nonnull_data(domains[index].edges);
  }
}

bool dump_request (const std::string &path, const Request &request,
                   std::string *error) noexcept
{
  bool touched_output = false;
  try {
    if (path.empty()) fail("capture output path is empty");
    validate_request(request);

    const std::vector<std::uint8_t> metadata = metadata_bytes(request);
    if (metadata.size() > UINT32_MAX) {
      fail("metadata section is too large");
    }
    std::vector<Section> sections =
        make_sections(request, metadata.size());
    for (auto &section : sections) {
      Sha256 sha;
      Encoder encoder(nullptr, nullptr, &sha);
      encode_section(encoder, request, metadata, section);
      encoder.finish();
      section.digest = sha.finish();
    }

    const std::vector<std::uint8_t> directory =
        encode_directory(sections);
    const std::uint64_t file_bytes =
        sections.empty() ? kPayloadOffset :
                           add_or_fail(sections.back().offset,
                                       sections.back().bytes,
                                       "capture file size");
    const Digest zero_digest{};
    const std::vector<std::uint8_t> zeroed_header =
        encode_header(file_bytes, zero_digest);
    const Digest digest = header_digest(zeroed_header, directory);
    const std::vector<std::uint8_t> header =
        encode_header(file_bytes, digest);

    errno = 0;
    std::ofstream stream(path.c_str(),
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream) fail(errno_message("unable to create capture", path));
    touched_output = true;
    stream.write(reinterpret_cast<const char *>(header.data()),
                 static_cast<std::streamsize>(header.size()));
    stream.write(reinterpret_cast<const char *>(directory.data()),
                 static_cast<std::streamsize>(directory.size()));
    if (!stream) fail("capture header write failed");
    for (const auto &section : sections) {
      Encoder encoder(nullptr, &stream, nullptr);
      encode_section(encoder, request, metadata, section);
      encoder.finish();
    }
    stream.flush();
    if (!stream) fail("capture flush failed");
    stream.close();
    if (!stream) fail("capture close failed");
    if (error) error->clear();
    return true;
  } catch (const std::exception &exception) {
    if (touched_output) std::remove(path.c_str());
    set_error(error, exception.what());
  } catch (...) {
    if (touched_output) std::remove(path.c_str());
    set_error(error, "unknown exception while writing capture");
  }
  return false;
}

bool load_request (const std::string &path, OwnedRequest &output,
                   std::string *error) noexcept
{
  try {
    if (path.empty()) fail("capture input path is empty");
    errno = 0;
    std::ifstream stream(path.c_str(), std::ios::binary | std::ios::in);
    if (!stream) fail(errno_message("unable to open capture", path));
    const std::uint64_t actual_file_bytes = stream_file_size(stream);
    if (actual_file_bytes < kPayloadOffset) {
      fail("capture is truncated before its payload");
    }

    std::vector<std::uint8_t> header =
        read_exact(stream, kHeaderBytes, "capture header");
    Cursor header_cursor(header.data(), header.size());
    std::uint8_t magic[8];
    header_cursor.raw(magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
      fail("capture magic is invalid");
    }
    if (header_cursor.u32() != kFileVersion) {
      fail("unknown capture file version");
    }
    if (header_cursor.u32() != kHeaderBytes ||
        header_cursor.u32() != kEndianMarker ||
        header_cursor.u32() != kSectionCount ||
        header_cursor.u32() != kDirectoryEntryBytes ||
        header_cursor.u32() != 0 ||
        header_cursor.u64() != kDirectoryOffset ||
        header_cursor.u64() != kDirectoryBytes ||
        header_cursor.u64() != kPayloadOffset) {
      fail("capture header layout is unsupported or ambiguous");
    }
    const std::uint64_t declared_file_bytes = header_cursor.u64();
    Digest stored_header_digest{};
    header_cursor.raw(stored_header_digest.data(),
                      stored_header_digest.size());
    std::uint8_t header_reserved[32];
    header_cursor.raw(header_reserved, sizeof(header_reserved));
    if (!header_cursor.empty() ||
        !bytes_zero(header_reserved, sizeof(header_reserved))) {
      fail("capture header reserved bytes are nonzero");
    }
    if (declared_file_bytes != actual_file_bytes) {
      fail(declared_file_bytes < actual_file_bytes
               ? "capture has ambiguous trailing bytes"
               : "capture is truncated");
    }

    std::vector<std::uint8_t> directory =
        read_exact(stream, kDirectoryBytes, "capture directory");
    std::vector<std::uint8_t> zeroed_header = header;
    std::fill(zeroed_header.begin() + 64,
              zeroed_header.begin() + 96, 0);
    if (header_digest(zeroed_header, directory) !=
        stored_header_digest) {
      fail("capture header/directory SHA-256 mismatch");
    }
    const std::vector<Section> sections =
        decode_directory(directory);

    std::uint64_t expected_offset = kPayloadOffset;
    for (std::size_t index = 0; index < sections.size(); ++index) {
      const auto &section = sections[index];
      const std::uint64_t expected_bytes =
          multiply_or_fail(section.count, section.record_bytes,
                           "directory section byte count");
      if (section.offset != expected_offset ||
          section.bytes != expected_bytes) {
        fail("capture sections overlap, contain gaps, or have bad sizes");
      }
      expected_offset =
          add_or_fail(expected_offset, section.bytes,
                      "directory payload extent");
    }
    if (expected_offset != declared_file_bytes) {
      fail("capture payload extent does not equal declared file size");
    }

    require_section(sections[0], kMetadata, kNoDomain,
                    sections[0].record_bytes, 1, 0);
    if (!sections[0].record_bytes || sections[0].bytes > 65536) {
      fail("metadata section has an invalid size");
    }
    std::vector<std::uint8_t> metadata =
        read_exact(stream, sections[0].bytes, "metadata section");
    verify_section_digest(sections[0], metadata, 0);

    OwnedRequest candidate;
    Cursor metadata_cursor(metadata.data(), metadata.size());
    decode_metadata(metadata_cursor, candidate.request);

    require_section(sections[1], kSourceCellIndices, kNoDomain,
                    kU64WireBytes,
                    candidate.request.hierarchy.source_cell_count, 1);
    require_section(sections[2], kContexts, kNoDomain,
                    kContextWireBytes,
                    candidate.request.hierarchy.context_count, 2);
    require_section(sections[3], kContextParents, kNoDomain, 4,
                    candidate.request.hierarchy.context_parent_count, 3);
    for (std::size_t domain = 0; domain < kDomainCount; ++domain) {
      const std::size_t base = 4 + domain * 3;
      require_section(
          sections[base], kDomainCells,
          static_cast<std::uint32_t>(domain), kCellWireBytes,
          candidate.request.domains[domain].cell_count, base);
      require_section(
          sections[base + 1], kDomainPolygons,
          static_cast<std::uint32_t>(domain), kPolygonWireBytes,
          candidate.request.domains[domain].polygon_count, base + 1);
      require_section(
          sections[base + 2], kDomainEdges,
          static_cast<std::uint32_t>(domain), kEdgeWireBytes,
          candidate.request.domains[domain].edge_count, base + 2);
    }

    auto read_section = [&](std::size_t index, const char *what) {
      std::vector<std::uint8_t> bytes =
          read_exact(stream, sections[index].bytes, what);
      verify_section_digest(sections[index], bytes, index);
      return bytes;
    };

    {
      const auto bytes = read_section(1, "source-cell section");
      decode_records<std::uint64_t>(
          bytes, sections[1].count, kU64WireBytes,
          candidate.source_cell_indices,
          [](Cursor &cursor) { return cursor.u64(); });
    }
    {
      const auto bytes = read_section(2, "context section");
      decode_records<Context>(
          bytes, sections[2].count, kContextWireBytes,
          candidate.contexts, decode_context);
    }
    {
      const auto bytes = read_section(3, "context-parent section");
      decode_records<std::uint32_t>(
          bytes, sections[3].count, 4, candidate.context_parent_ids,
          [](Cursor &cursor) { return cursor.u32(); });
    }
    for (std::size_t domain = 0; domain < kDomainCount; ++domain) {
      const std::size_t base = 4 + domain * 3;
      {
        const auto bytes = read_section(base, "domain-cell section");
        decode_records<Cell>(
            bytes, sections[base].count, kCellWireBytes,
            candidate.domains[domain].cells, decode_cell);
      }
      {
        const auto bytes =
            read_section(base + 1, "domain-polygon section");
        decode_records<Polygon>(
            bytes, sections[base + 1].count, kPolygonWireBytes,
            candidate.domains[domain].polygons, decode_polygon);
      }
      {
        const auto bytes =
            read_section(base + 2, "domain-edge section");
        decode_records<Edge>(
            bytes, sections[base + 2].count, kEdgeWireBytes,
            candidate.domains[domain].edges, decode_edge);
      }
    }
    if (stream.peek() != std::char_traits<char>::eof()) {
      fail("capture has ambiguous trailing data");
    }

    candidate.rebind();
    validate_request(candidate.request);
    output = std::move(candidate);
    if (error) error->clear();
    return true;
  } catch (const std::exception &exception) {
    set_error(error, exception.what());
  } catch (...) {
    set_error(error, "unknown exception while reading capture");
  }
  return false;
}

}  // namespace antenna_m1_m4_capture
}  // namespace klayout_cuda
