#include "m2_merged_boundary_oracle.h"

#include "m1_width_space_host_scene_format.h"

#include "dbCudaActive3Digest.h"
#include "dbCudaManhattanContour.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace klayout_cuda {
namespace m2_boundary_oracle {

namespace {

namespace format = klayout_m1ws_scene;
using Sha256 = db::cuda_active3_digest::Sha256;

constexpr char kQualifiedFileSha256[] =
    "980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f";
constexpr char kQualifiedSceneSha256[] =
    "441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480";
constexpr char kQualifiedSourceSha256[] =
    "8630e7ca7a2a72d03a4ba4fe61fd9f048a04d5370cd488b297c484bb19754558";
constexpr std::uint64_t kQualifiedContextCount = UINT64_C(39573);
constexpr std::uint64_t kQualifiedMetalContextCount = UINT64_C(334);
constexpr std::uint64_t kQualifiedCellCount = UINT64_C(121);
constexpr std::uint64_t kQualifiedStoredContourCount = UINT64_C(13166);
constexpr std::uint64_t kQualifiedStoredEdgeCount = UINT64_C(4380228);
constexpr std::uint64_t kQualifiedFlatContourCount = UINT64_C(14222);
constexpr std::uint64_t kQualifiedFlatEdgeCount = UINT64_C(4385384);
constexpr std::int64_t kQualifiedDistance = INT64_C(140);
constexpr std::int64_t kCoordinateLimit = INT64_C(1000000000000);
constexpr std::int64_t kQualifiedLeft = INT64_C(6230);
constexpr std::int64_t kQualifiedBottom = INT64_C(6225);
constexpr std::int64_t kQualifiedRight = INT64_C(1788415);
constexpr std::int64_t kQualifiedTop = INT64_C(1487300);
constexpr std::uint32_t kCandidateVersion = 1;
constexpr std::uint32_t kCandidateHeaderBytes = 128;
constexpr std::uint32_t kEndianTag = UINT32_C(0x01020304);
constexpr char kCandidateMagic[8] =
    {'K', 'M', '2', 'B', 'N', 'D', '0', '1'};

struct Context
{
  std::int64_t tx;
  std::int64_t ty;
  std::uint32_t cell;
  std::uint32_t transform;
};

struct Cell
{
  std::uint64_t source_cell;
  std::uint64_t polygon_begin;
  std::uint64_t edge_begin;
  std::uint32_t polygon_count;
  std::uint32_t edge_count;
};

struct Polygon
{
  std::uint64_t edge_begin;
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint32_t polygon_id;
  std::uint32_t edge_count;
};

struct Edge
{
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
};

struct MetalContext
{
  std::uint32_t context;
  std::uint64_t polygon_offset;
  std::uint64_t edge_offset;
};

struct Vertex
{
  std::int64_t x;
  std::int64_t y;
  std::uint64_t contour;
};

struct Matrix
{
  int xx;
  int xy;
  int yx;
  int yy;
};

constexpr Matrix kTransforms[8] = {
    {1, 0, 0, 1},   {0, -1, 1, 0}, {-1, 0, 0, -1},
    {0, 1, -1, 0},  {1, 0, 0, -1}, {0, 1, 1, 0},
    {-1, 0, 0, 1},  {0, -1, -1, 0}};

bool coordinate_valid(std::int64_t value)
{
  return value >= -kCoordinateLimit && value <= kCoordinateLimit;
}

bool checked_add_u64(std::uint64_t first, std::uint64_t second,
                     std::uint64_t *result)
{
  if (second > std::numeric_limits<std::uint64_t>::max() - first) {
    return false;
  }
  *result = first + second;
  return true;
}

bool range_valid(std::uint64_t begin, std::uint64_t count,
                 std::uint64_t total)
{
  std::uint64_t end = 0;
  return checked_add_u64(begin, count, &end) && end <= total;
}

std::string hex_digest(const std::uint8_t *bytes, std::size_t count)
{
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < count; ++index) {
    stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return stream.str();
}

std::string normalized_digest(const std::string &digest, const char *what)
{
  if (digest.size() != 64 ||
      !std::all_of(digest.begin(), digest.end(),
                   [](unsigned char character) {
                     return std::isxdigit(character) != 0;
                   })) {
    throw std::runtime_error(std::string(what) +
                             " must contain 64 hexadecimal digits");
  }
  std::string result = digest;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return result;
}

std::array<std::uint8_t, 32> digest_from_hex(const std::string &digest)
{
  const std::string normalized = normalized_digest(digest, "digest");
  std::array<std::uint8_t, 32> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const auto nibble = [](char character) -> std::uint8_t {
      if (character >= '0' && character <= '9') {
        return static_cast<std::uint8_t>(character - '0');
      }
      return static_cast<std::uint8_t>(character - 'a' + 10);
    };
    bytes[index] = static_cast<std::uint8_t>(
        (nibble(normalized[index * 2]) << 4) |
        nibble(normalized[index * 2 + 1]));
  }
  return bytes;
}

std::string sha256(const void *data, std::size_t bytes)
{
  Sha256 sha;
  sha.update(data, bytes);
  const auto digest = sha.finish();
  return hex_digest(digest.data(), digest.size());
}

bool digest_nonzero(const std::array<std::uint8_t, 32> &digest)
{
  return std::any_of(digest.begin(), digest.end(),
                     [](std::uint8_t byte) { return byte != 0; });
}

bool layout_equal(const format::FileLayoutV1 &first,
                  const format::FileLayoutV1 &second)
{
  return first.file_bytes == second.file_bytes &&
         first.payload_offset == second.payload_offset &&
         first.payload_bytes == second.payload_bytes &&
         first.contexts_offset == second.contexts_offset &&
         first.metal_contexts_offset == second.metal_contexts_offset &&
         first.cells_offset == second.cells_offset &&
         first.polygons_offset == second.polygons_offset &&
         first.edges_offset == second.edges_offset;
}

std::pair<__int128, __int128> transform_point_128(
    std::uint32_t transform, std::int64_t x, std::int64_t y)
{
  if (transform >= 8) {
    throw std::runtime_error("invalid orthogonal transform");
  }
  const Matrix matrix = kTransforms[transform];
  return {
      static_cast<__int128>(matrix.xx) * x +
          static_cast<__int128>(matrix.xy) * y,
      static_cast<__int128>(matrix.yx) * x +
          static_cast<__int128>(matrix.yy) * y};
}

std::int64_t narrow_i64(__int128 value, const char *what)
{
  if (value < std::numeric_limits<std::int64_t>::min() ||
      value > std::numeric_limits<std::int64_t>::max()) {
    throw std::runtime_error(std::string(what) +
                             " exceeds signed int64");
  }
  return static_cast<std::int64_t>(value);
}

Edge transform_edge(const Context &context, const Edge &source)
{
  const auto first =
      transform_point_128(context.transform, source.x1, source.y1);
  const auto second =
      transform_point_128(context.transform, source.x2, source.y2);
  Edge result{
      narrow_i64(first.first + context.tx, "world edge x1"),
      narrow_i64(first.second + context.ty, "world edge y1"),
      narrow_i64(second.first + context.tx, "world edge x2"),
      narrow_i64(second.second + context.ty, "world edge y2")};
  if (context.transform >= 4) {
    std::swap(result.x1, result.x2);
    std::swap(result.y1, result.y2);
  }
  return result;
}

bool horizontal(const Edge &edge)
{
  return edge.y1 == edge.y2 && edge.x1 != edge.x2;
}

DirectedSegmentI64 normalize_edge(const Edge &edge)
{
  if (horizontal(edge)) {
    return {edge.y1, std::min(edge.x1, edge.x2),
            std::max(edge.x1, edge.x2),
            edge.x2 > edge.x1 ? 1 : -1, SegmentAxis::horizontal};
  }
  if (edge.x1 == edge.x2 && edge.y1 != edge.y2) {
    return {edge.x1, std::min(edge.y1, edge.y2),
            std::max(edge.y1, edge.y2),
            edge.y2 > edge.y1 ? -1 : 1, SegmentAxis::vertical};
  }
  throw std::runtime_error("world edge is degenerate or non-Manhattan");
}

bool canonical_less(const DirectedSegmentI64 &first,
                    const DirectedSegmentI64 &second)
{
  const auto first_axis = static_cast<std::uint32_t>(first.axis);
  const auto second_axis = static_cast<std::uint32_t>(second.axis);
  if (first_axis != second_axis) {
    return first_axis < second_axis;
  }
  if (first.side != second.side) {
    return first.side < second.side;
  }
  if (first.fixed != second.fixed) {
    return first.fixed < second.fixed;
  }
  if (first.lo != second.lo) {
    return first.lo < second.lo;
  }
  return first.hi < second.hi;
}

bool geometry_less(const DirectedSegmentI64 &first,
                   const DirectedSegmentI64 &second)
{
  const auto first_axis = static_cast<std::uint32_t>(first.axis);
  const auto second_axis = static_cast<std::uint32_t>(second.axis);
  if (first_axis != second_axis) {
    return first_axis < second_axis;
  }
  if (first.fixed != second.fixed) {
    return first.fixed < second.fixed;
  }
  if (first.lo != second.lo) {
    return first.lo < second.lo;
  }
  if (first.hi != second.hi) {
    return first.hi < second.hi;
  }
  return first.side < second.side;
}

bool same_segment(const DirectedSegmentI64 &first,
                  const DirectedSegmentI64 &second)
{
  return first.axis == second.axis && first.side == second.side &&
         first.fixed == second.fixed && first.lo == second.lo &&
         first.hi == second.hi;
}

std::uint64_t delta_magnitude(std::int64_t first, std::int64_t second)
{
  return first >= second
             ? static_cast<std::uint64_t>(first) -
                   static_cast<std::uint64_t>(second)
             : static_cast<std::uint64_t>(second) -
                   static_cast<std::uint64_t>(first);
}

void add_checked(std::uint64_t value, std::uint64_t *sum,
                 const char *what)
{
  if (value > std::numeric_limits<std::uint64_t>::max() - *sum) {
    throw std::runtime_error(std::string(what) + " overflows uint64");
  }
  *sum += value;
}

std::string decimal_i128(__int128 value)
{
  if (value == 0) {
    return "0";
  }
  const bool negative = value < 0;
  unsigned __int128 magnitude =
      negative ? static_cast<unsigned __int128>(-(value + 1)) + 1
               : static_cast<unsigned __int128>(value);
  std::string digits;
  while (magnitude) {
    digits.push_back(
        static_cast<char>('0' + static_cast<unsigned int>(magnitude % 10)));
    magnitude /= 10;
  }
  if (negative) {
    digits.push_back('-');
  }
  std::reverse(digits.begin(), digits.end());
  return digits;
}

void encode_segment(const DirectedSegmentI64 &segment,
                    std::uint8_t *record)
{
  format::store_i64_le(record, segment.fixed);
  format::store_i64_le(record + 8, segment.lo);
  format::store_i64_le(record + 16, segment.hi);
  format::store_u32_le(
      record + 24, static_cast<std::uint32_t>(segment.side));
  format::store_u32_le(
      record + 28, static_cast<std::uint32_t>(segment.axis));
}

DirectedSegmentI64 decode_segment(const std::uint8_t *record)
{
  const std::uint32_t side_bits = format::load_u32_le(record + 24);
  std::int32_t side = 0;
  std::memcpy(&side, &side_bits, sizeof(side));
  return {format::load_i64_le(record),
          format::load_i64_le(record + 8),
          format::load_i64_le(record + 16), side,
          static_cast<SegmentAxis>(format::load_u32_le(record + 28))};
}

std::string boundary_sha256(
    const std::vector<DirectedSegmentI64> &segments)
{
  Sha256 sha;
  std::array<std::uint8_t, 32> record{};
  for (const DirectedSegmentI64 &segment : segments) {
    encode_segment(segment, record.data());
    sha.update(record.data(), record.size());
  }
  const auto digest = sha.finish();
  return hex_digest(digest.data(), digest.size());
}

std::uint64_t boundary_fnv64(
    const std::vector<DirectedSegmentI64> &segments)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&hash](std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & UINT64_C(0xff);
      hash *= UINT64_C(1099511628211);
    }
  };
  mix(segments.size());
  for (const DirectedSegmentI64 &segment : segments) {
    mix(static_cast<std::uint32_t>(segment.axis));
    mix(static_cast<std::uint32_t>(segment.side));
    mix(static_cast<std::uint64_t>(segment.fixed));
    mix(static_cast<std::uint64_t>(segment.lo));
    mix(static_cast<std::uint64_t>(segment.hi));
  }
  return hash;
}

void validate_canonical_candidate(
    const std::vector<DirectedSegmentI64> &segments)
{
  for (std::size_t index = 0; index < segments.size(); ++index) {
    const DirectedSegmentI64 &segment = segments[index];
    if ((segment.axis != SegmentAxis::horizontal &&
         segment.axis != SegmentAxis::vertical) ||
        (segment.side != -1 && segment.side != 1) ||
        segment.lo >= segment.hi) {
      throw std::runtime_error("candidate has an invalid boundary segment");
    }
    if (index && !canonical_less(segments[index - 1], segment)) {
      throw std::runtime_error(
          "candidate boundary is not strict canonical order");
    }
    if (index) {
      const DirectedSegmentI64 &previous = segments[index - 1];
      if (previous.axis == segment.axis &&
          previous.side == segment.side &&
          previous.fixed == segment.fixed &&
          segment.lo <= previous.hi) {
        throw std::runtime_error(
            "candidate has overlapping/adjacent canonical fragments");
      }
    }
  }
}

}  // namespace

BoundaryOracle load_cpu_merged_boundary(const std::string &path,
                                        const LoadOptions &options)
{
  const std::string expected_file =
      normalized_digest(options.expected_file_sha256,
                        "expected file SHA-256");
  const std::string expected_scene =
      normalized_digest(options.expected_scene_sha256,
                        "expected scene SHA-256");
  if (expected_file != kQualifiedFileSha256 ||
      expected_scene != kQualifiedSceneSha256) {
    throw std::runtime_error(
        "KM1WSCN1 identity is outside the qualified production M2 oracle");
  }

  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open CPU-merged M2 oracle: " + path);
  }
  const std::streamoff file_end = input.tellg();
  if (file_end < static_cast<std::streamoff>(format::kFileHeaderBytes) ||
      static_cast<std::uint64_t>(file_end) >
          std::numeric_limits<std::size_t>::max() ||
      file_end > std::numeric_limits<std::streamsize>::max()) {
    throw std::runtime_error("KM1WSCN1 file size is invalid");
  }
  std::vector<std::uint8_t> storage(
      static_cast<std::size_t>(file_end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(storage.data()),
                  static_cast<std::streamsize>(file_end))) {
    throw std::runtime_error("short read of KM1WSCN1");
  }
  const std::string file_digest =
      sha256(storage.data(), storage.size());
  if (file_digest != expected_file) {
    throw std::runtime_error("KM1WSCN1 file SHA-256 mismatch");
  }

  format::FileHeaderV1 header{};
  if (!format::decode_file_header(
          storage.data(), storage.size(), header) ||
      header.version != format::kFileVersion ||
      header.header_bytes != format::kFileHeaderBytes ||
      header.endian_tag != format::kEndianTag ||
      header.flags != format::kRequiredFlags ||
      header.source_layer != 101 || header.source_datatype != 0 ||
      header.reserved[0] || header.reserved[1] ||
      header.reserved[2]) {
    throw std::runtime_error(
        "KM1WSCN1 fixed header is malformed or unqualified");
  }
  format::FileLayoutV1 canonical_layout{};
  if (!format::compute_file_layout(
          header.context_count, header.metal_context_count,
          header.cell_count, header.polygon_count, header.edge_count,
          canonical_layout) ||
      !layout_equal(header.layout, canonical_layout) ||
      header.layout.file_bytes != storage.size()) {
    throw std::runtime_error("KM1WSCN1 layout is not canonical");
  }
  if (header.context_count != kQualifiedContextCount ||
      header.metal_context_count != kQualifiedMetalContextCount ||
      header.cell_count != kQualifiedCellCount ||
      header.polygon_count != kQualifiedStoredContourCount ||
      header.edge_count != kQualifiedStoredEdgeCount ||
      header.context_count > options.max_contexts ||
      header.polygon_count > options.max_stored_polygons ||
      header.edge_count > options.max_stored_edges ||
      header.context_count >
          std::numeric_limits<std::uint32_t>::max() ||
      header.cell_count > std::numeric_limits<std::uint32_t>::max() ||
      header.polygon_count >
          std::numeric_limits<std::uint32_t>::max() ||
      header.edge_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "KM1WSCN1 stored census is outside qualification/capacity");
  }
  if (!digest_nonzero(header.scene_digest) ||
      !digest_nonzero(header.transport_digest) ||
      !digest_nonzero(header.source_digest) ||
      hex_digest(header.scene_digest.data(), 32) != expected_scene ||
      hex_digest(header.source_digest.data(), 32) !=
          kQualifiedSourceSha256) {
    throw std::runtime_error("KM1WSCN1 provenance digest mismatch");
  }

  Sha256 transport_sha;
  transport_sha.update(storage.data(), format::kTransportDigestOffset);
  const std::array<std::uint8_t, 32> zero_digest{};
  transport_sha.update(zero_digest.data(), zero_digest.size());
  transport_sha.update(
      storage.data() + format::kSourceDigestOffset,
      storage.size() - format::kSourceDigestOffset);
  const auto computed_transport = transport_sha.finish();
  if (!std::equal(computed_transport.begin(), computed_transport.end(),
                  header.transport_digest.begin())) {
    throw std::runtime_error("KM1WSCN1 transport SHA-256 mismatch");
  }
  Sha256 payload_sha;
  payload_sha.update(
      storage.data() + header.layout.payload_offset,
      static_cast<std::size_t>(header.layout.payload_bytes));
  const auto computed_payload = payload_sha.finish();
  if (!std::equal(computed_payload.begin(), computed_payload.end(),
                  header.scene_digest.begin())) {
    throw std::runtime_error("KM1WSCN1 payload SHA-256 mismatch");
  }

  format::SemanticHeaderV1 semantic{};
  if (!format::decode_semantic_header(
          storage.data() + header.layout.payload_offset,
          static_cast<std::size_t>(header.layout.payload_bytes),
          semantic) ||
      semantic.format_version != 1 ||
      semantic.dbu_per_micron != 2000 || semantic.root_cell != 0 ||
      semantic.reserved || semantic.width_distance != kQualifiedDistance ||
      semantic.spacing_distance != kQualifiedDistance ||
      semantic.context_count != header.context_count ||
      semantic.metal_context_count != header.metal_context_count ||
      semantic.cell_count != header.cell_count ||
      semantic.polygon_count != header.polygon_count ||
      semantic.edge_count != header.edge_count ||
      semantic.flat_polygon_count != kQualifiedFlatContourCount ||
      semantic.flat_edge_count != kQualifiedFlatEdgeCount ||
      semantic.flat_polygon_count > options.max_flat_polygons ||
      semantic.flat_edge_count > options.max_flat_edges ||
      semantic.scene_left != kQualifiedLeft ||
      semantic.scene_bottom != kQualifiedBottom ||
      semantic.scene_right != kQualifiedRight ||
      semantic.scene_top != kQualifiedTop) {
    throw std::runtime_error(
        "KM1WSCN1 semantic header is outside M2 qualification");
  }

  std::vector<Context> contexts;
  contexts.reserve(static_cast<std::size_t>(header.context_count));
  for (std::uint64_t index = 0; index < header.context_count; ++index) {
    const std::uint8_t *record =
        storage.data() + header.layout.contexts_offset +
        index * format::kContextRecordBytes;
    const Context context{
        format::load_i64_le(record),
        format::load_i64_le(record + 8),
        format::load_u32_le(record + 16),
        format::load_u32_le(record + 20)};
    if (context.cell >= header.cell_count || context.transform >= 8 ||
        !coordinate_valid(context.tx) || !coordinate_valid(context.ty)) {
      throw std::runtime_error("KM1WSCN1 context is invalid");
    }
    contexts.push_back(context);
  }
  if (contexts.empty() || contexts.front().tx ||
      contexts.front().ty || contexts.front().cell != semantic.root_cell ||
      contexts.front().transform) {
    throw std::runtime_error(
        "KM1WSCN1 root context is not canonical");
  }

  std::vector<Cell> cells;
  cells.reserve(static_cast<std::size_t>(header.cell_count));
  std::set<std::uint64_t> source_cells;
  std::uint64_t next_polygon = 0;
  std::uint64_t next_edge = 0;
  for (std::uint64_t index = 0; index < header.cell_count; ++index) {
    const std::uint8_t *record =
        storage.data() + header.layout.cells_offset +
        index * format::kCellRecordBytes;
    const Cell cell{
        format::load_u64_le(record),
        format::load_u64_le(record + 8),
        format::load_u64_le(record + 16),
        format::load_u32_le(record + 24),
        format::load_u32_le(record + 28)};
    if (!source_cells.insert(cell.source_cell).second ||
        cell.polygon_begin != next_polygon ||
        cell.edge_begin != next_edge ||
        !range_valid(cell.polygon_begin, cell.polygon_count,
                     header.polygon_count) ||
        !range_valid(cell.edge_begin, cell.edge_count,
                     header.edge_count) ||
        ((!cell.polygon_count) != (!cell.edge_count))) {
      throw std::runtime_error(
          "KM1WSCN1 cell partition is not canonical");
    }
    next_polygon += cell.polygon_count;
    next_edge += cell.edge_count;
    cells.push_back(cell);
  }
  if (next_polygon != header.polygon_count ||
      next_edge != header.edge_count) {
    throw std::runtime_error(
        "KM1WSCN1 cells do not partition stored topology");
  }

  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(header.edge_count));
  for (std::uint64_t index = 0; index < header.edge_count; ++index) {
    const std::uint8_t *record =
        storage.data() + header.layout.edges_offset +
        index * format::kEdgeRecordBytes;
    edges.push_back(
        {format::load_i64_le(record),
         format::load_i64_le(record + 8),
         format::load_i64_le(record + 16),
         format::load_i64_le(record + 24)});
  }

  std::vector<Polygon> polygons;
  polygons.reserve(static_cast<std::size_t>(header.polygon_count));
  db::cuda_manhattan_contour::TranslationValidationCache<Edge>
      contour_cache;
  std::uint64_t polygon_owner = 0;
  std::uint64_t expected_edge = 0;
  for (std::uint64_t index = 0; index < header.polygon_count; ++index) {
    while (polygon_owner + 1 < cells.size() &&
           index >= cells[polygon_owner].polygon_begin +
                        cells[polygon_owner].polygon_count) {
      ++polygon_owner;
    }
    const std::uint8_t *record =
        storage.data() + header.layout.polygons_offset +
        index * format::kPolygonRecordBytes;
    const Polygon polygon{
        format::load_u64_le(record),
        format::load_i64_le(record + 8),
        format::load_i64_le(record + 16),
        format::load_i64_le(record + 24),
        format::load_i64_le(record + 32),
        format::load_u32_le(record + 40),
        format::load_u32_le(record + 44)};
    const Cell &cell = cells[polygon_owner];
    if (polygon.edge_begin != expected_edge ||
        polygon.polygon_id != index - cell.polygon_begin ||
        polygon.edge_count < 4 ||
        !range_valid(polygon.edge_begin, polygon.edge_count,
                     cell.edge_begin + cell.edge_count) ||
        polygon.left >= polygon.right ||
        polygon.bottom >= polygon.top ||
        !coordinate_valid(polygon.left) ||
        !coordinate_valid(polygon.bottom) ||
        !coordinate_valid(polygon.right) ||
        !coordinate_valid(polygon.top)) {
      throw std::runtime_error("KM1WSCN1 polygon record is invalid");
    }
    std::vector<Edge> contour;
    contour.reserve(polygon.edge_count);
    std::int64_t left = std::numeric_limits<std::int64_t>::max();
    std::int64_t bottom = std::numeric_limits<std::int64_t>::max();
    std::int64_t right = std::numeric_limits<std::int64_t>::min();
    std::int64_t top = std::numeric_limits<std::int64_t>::min();
    __int128 area2 = 0;
    for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
      const Edge edge = edges[polygon.edge_begin + local];
      if ((!horizontal(edge) &&
           !(edge.x1 == edge.x2 && edge.y1 != edge.y2)) ||
          !coordinate_valid(edge.x1) || !coordinate_valid(edge.y1) ||
          !coordinate_valid(edge.x2) || !coordinate_valid(edge.y2)) {
        throw std::runtime_error(
            "KM1WSCN1 contour has an invalid edge");
      }
      const Edge following =
          edges[polygon.edge_begin +
                (local + 1 == polygon.edge_count ? 0 : local + 1)];
      if (edge.x2 != following.x1 || edge.y2 != following.y1) {
        throw std::runtime_error("KM1WSCN1 contour is open");
      }
      if (horizontal(edge) == horizontal(following)) {
        throw std::runtime_error(
            "KM1WSCN1 contour has adjacent collinear fragments");
      }
      left = std::min(left, std::min(edge.x1, edge.x2));
      bottom = std::min(bottom, std::min(edge.y1, edge.y2));
      right = std::max(right, std::max(edge.x1, edge.x2));
      top = std::max(top, std::max(edge.y1, edge.y2));
      area2 += static_cast<__int128>(edge.x1) * edge.y2 -
               static_cast<__int128>(edge.x2) * edge.y1;
      contour.push_back(edge);
    }
    if (area2 >= 0 || left != polygon.left ||
        bottom != polygon.bottom || right != polygon.right ||
        top != polygon.top ||
        contour_cache.validate_contour(contour) !=
            db::cuda_manhattan_contour::ValidationResult::Valid) {
      throw std::runtime_error(
          "KM1WSCN1 contour topology/orientation/bbox is invalid");
    }
    expected_edge += polygon.edge_count;
    polygons.push_back(polygon);
  }
  if (expected_edge != header.edge_count) {
    throw std::runtime_error(
        "KM1WSCN1 polygons do not partition the edge table");
  }

  std::vector<MetalContext> metal_contexts;
  metal_contexts.reserve(
      static_cast<std::size_t>(header.metal_context_count));
  std::uint64_t metal_cursor = 0;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_edges = 0;
  for (std::uint32_t context_id = 0;
       context_id < contexts.size(); ++context_id) {
    const Context &context = contexts[context_id];
    const Cell &cell = cells[context.cell];
    if (!cell.polygon_count) {
      continue;
    }
    if (metal_cursor >= header.metal_context_count) {
      throw std::runtime_error(
          "KM1WSCN1 nonempty-context table is truncated");
    }
    const std::uint8_t *record =
        storage.data() + header.layout.metal_contexts_offset +
        metal_cursor * format::kMetalContextRecordBytes;
    const MetalContext metal{
        format::load_u32_le(record),
        format::load_u64_le(record + 4),
        format::load_u64_le(record + 12)};
    if (metal.context != context_id ||
        metal.polygon_offset != flat_polygons ||
        metal.edge_offset != flat_edges ||
        !checked_add_u64(flat_polygons, cell.polygon_count,
                         &flat_polygons) ||
        !checked_add_u64(flat_edges, cell.edge_count, &flat_edges)) {
      throw std::runtime_error(
          "KM1WSCN1 nonempty-context offsets are not canonical");
    }
    metal_contexts.push_back(metal);
    ++metal_cursor;
  }
  if (metal_cursor != header.metal_context_count ||
      flat_polygons != semantic.flat_polygon_count ||
      flat_edges != semantic.flat_edge_count) {
    throw std::runtime_error(
        "KM1WSCN1 flat context census is inconsistent");
  }

  BoundaryOracle oracle;
  oracle.file_sha256 = file_digest;
  oracle.scene_sha256 = expected_scene;
  oracle.source_sha256 =
      hex_digest(header.source_digest.data(), header.source_digest.size());
  oracle.segments.reserve(static_cast<std::size_t>(flat_edges));
  std::vector<Vertex> vertices;
  vertices.reserve(static_cast<std::size_t>(flat_edges));
  oracle.stats.stored_contexts = contexts.size();
  oracle.stats.nonempty_contexts = metal_contexts.size();
  oracle.stats.stored_cells = cells.size();
  oracle.stats.stored_contours = polygons.size();
  oracle.stats.stored_edges = edges.size();
  oracle.stats.flat_contours = flat_polygons;
  oracle.stats.flat_edges = flat_edges;

  std::uint64_t world_contour = 0;
  std::uint64_t world_edge_count = 0;
  bool have_world_box = false;
  std::int64_t world_left = 0;
  std::int64_t world_bottom = 0;
  std::int64_t world_right = 0;
  std::int64_t world_top = 0;
  __int128 max_area = 0;
  __int128 total_area = 0;

  for (const MetalContext &metal : metal_contexts) {
    const Context &context = contexts[metal.context];
    const Cell &cell = cells[context.cell];
    for (std::uint32_t local_polygon = 0;
         local_polygon < cell.polygon_count; ++local_polygon) {
      const std::uint64_t source_polygon_id =
          cell.polygon_begin + local_polygon;
      const Polygon &polygon = polygons[source_polygon_id];
      std::vector<Edge> world;
      world.reserve(polygon.edge_count);
      if (context.transform < 4) {
        for (std::uint32_t local = 0; local < polygon.edge_count;
             ++local) {
          world.push_back(
              transform_edge(context, edges[polygon.edge_begin + local]));
        }
      } else {
        for (std::uint32_t local = polygon.edge_count; local > 0;
             --local) {
          world.push_back(
              transform_edge(context,
                             edges[polygon.edge_begin + local - 1]));
        }
      }

      __int128 area2 = 0;
      std::uint64_t perimeter = 0;
      std::int64_t left = std::numeric_limits<std::int64_t>::max();
      std::int64_t bottom = std::numeric_limits<std::int64_t>::max();
      std::int64_t right = std::numeric_limits<std::int64_t>::min();
      std::int64_t top = std::numeric_limits<std::int64_t>::min();
      for (std::uint32_t local = 0; local < world.size(); ++local) {
        const Edge &edge = world[local];
        const Edge &following = world[(local + 1) % world.size()];
        if (edge.x2 != following.x1 || edge.y2 != following.y1 ||
            horizontal(edge) == horizontal(following)) {
          throw std::runtime_error(
              "world contour lost closure/alternating axes");
        }
        area2 += static_cast<__int128>(edge.x1) * edge.y2 -
                 static_cast<__int128>(edge.x2) * edge.y1;
        const std::uint64_t length =
            delta_magnitude(edge.x1, edge.x2) +
            delta_magnitude(edge.y1, edge.y2);
        add_checked(length, &perimeter, "contour perimeter");
        oracle.stats.longest_segment =
            std::max(oracle.stats.longest_segment, length);
        left = std::min(left, std::min(edge.x1, edge.x2));
        bottom = std::min(bottom, std::min(edge.y1, edge.y2));
        right = std::max(right, std::max(edge.x1, edge.x2));
        top = std::max(top, std::max(edge.y1, edge.y2));
        vertices.push_back({edge.x1, edge.y1, world_contour});
        const DirectedSegmentI64 segment = normalize_edge(edge);
        oracle.segments.push_back(segment);
        if (segment.axis == SegmentAxis::horizontal) {
          ++oracle.stats.horizontal_segments;
        } else {
          ++oracle.stats.vertical_segments;
        }
        if (segment.side < 0) {
          ++oracle.stats.negative_side_segments;
        } else {
          ++oracle.stats.positive_side_segments;
        }
      }
      if (area2 >= 0 || area2 % 2) {
        ++oracle.stats.hole_contours;
        throw std::runtime_error(
            "world contour is a hole/nonclockwise or has half-unit area");
      }
      const __int128 area = -area2 / 2;
      total_area += area;
      if (area > max_area) {
        max_area = area;
      }
      add_checked(perimeter, &oracle.stats.total_perimeter,
                  "total perimeter");
      oracle.stats.max_contour_perimeter =
          std::max(oracle.stats.max_contour_perimeter, perimeter);
      oracle.stats.max_contour_width =
          std::max(oracle.stats.max_contour_width, right - left);
      oracle.stats.max_contour_height =
          std::max(oracle.stats.max_contour_height, top - bottom);
      if (world.size() > oracle.stats.max_contour_edges) {
        oracle.stats.max_contour_edges = world.size();
        oracle.stats.max_contour_id = world_contour;
        oracle.stats.max_contour_context = metal.context;
        oracle.stats.max_contour_source_polygon = source_polygon_id;
      }
      if (!have_world_box) {
        world_left = left;
        world_bottom = bottom;
        world_right = right;
        world_top = top;
        have_world_box = true;
      } else {
        world_left = std::min(world_left, left);
        world_bottom = std::min(world_bottom, bottom);
        world_right = std::max(world_right, right);
        world_top = std::max(world_top, top);
      }
      world_edge_count += world.size();
      ++world_contour;
    }
  }
  if (world_contour != flat_polygons ||
      world_edge_count != flat_edges ||
      oracle.segments.size() != flat_edges ||
      vertices.size() != flat_edges || !have_world_box ||
      world_left != semantic.scene_left ||
      world_bottom != semantic.scene_bottom ||
      world_right != semantic.scene_right ||
      world_top != semantic.scene_top) {
    throw std::runtime_error(
        "world boundary census/bbox conservation failed");
  }
  oracle.stats.max_contour_area_dbu2 = decimal_i128(max_area);
  oracle.stats.total_area_dbu2 = decimal_i128(total_area);
  const auto cache_stats = contour_cache.statistics();
  oracle.stats.contour_cache_full_validations =
      cache_stats.full_validations;
  oracle.stats.contour_cache_hits = cache_stats.cache_hits;
  oracle.stats.contour_cache_edges = cache_stats.cached_edges;

  std::sort(vertices.begin(), vertices.end(),
            [](const Vertex &first, const Vertex &second) {
              if (first.x != second.x) {
                return first.x < second.x;
              }
              if (first.y != second.y) {
                return first.y < second.y;
              }
              return first.contour < second.contour;
            });
  for (std::size_t begin = 0; begin < vertices.size();) {
    std::size_t end = begin + 1;
    while (end < vertices.size() &&
           vertices[end].x == vertices[begin].x &&
           vertices[end].y == vertices[begin].y) {
      ++end;
    }
    if (end - begin > 1) {
      bool same_contour = false;
      for (std::size_t index = begin + 1; index < end; ++index) {
        same_contour =
            same_contour ||
            vertices[index].contour == vertices[index - 1].contour;
      }
      if (same_contour) {
        oracle.stats.repeated_vertices += end - begin - 1;
      } else {
        oracle.stats.kissing_vertices += end - begin - 1;
      }
    }
    begin = end;
  }
  if (oracle.stats.repeated_vertices ||
      oracle.stats.kissing_vertices) {
    throw std::runtime_error(
        "world boundary contains repeated/kissing vertices");
  }
  std::vector<Vertex>().swap(vertices);

  std::sort(oracle.segments.begin(), oracle.segments.end(),
            geometry_less);
  for (std::size_t begin = 0; begin < oracle.segments.size();) {
    std::size_t end = begin + 1;
    while (end < oracle.segments.size() &&
           oracle.segments[end].axis == oracle.segments[begin].axis &&
           oracle.segments[end].fixed ==
               oracle.segments[begin].fixed) {
      ++end;
    }
    std::int64_t furthest = oracle.segments[begin].hi;
    DirectedSegmentI64 previous = oracle.segments[begin];
    for (std::size_t index = begin + 1; index < end; ++index) {
      const DirectedSegmentI64 &current = oracle.segments[index];
      if (current.lo < furthest) {
        if (current.lo == previous.lo &&
            current.hi == previous.hi) {
          if (current.side == previous.side) {
            ++oracle.stats.duplicate_segments;
          } else {
            ++oracle.stats.opposite_segments;
          }
        } else {
          ++oracle.stats.collinear_overlaps;
        }
      } else if (current.lo == furthest) {
        ++oracle.stats.adjacent_collinear;
      }
      if (current.hi > furthest) {
        furthest = current.hi;
      }
      previous = current;
    }
    begin = end;
  }
  if (oracle.stats.duplicate_segments ||
      oracle.stats.opposite_segments ||
      oracle.stats.collinear_overlaps ||
      oracle.stats.adjacent_collinear) {
    throw std::runtime_error(
        "world boundary has duplicate/opposite/overlapping/adjacent "
        "collinear fragments");
  }

  std::vector<db::cuda_manhattan_contour::detail::SweepEvent>
      sweep_events;
  std::vector<std::int64_t> horizontal_y;
  sweep_events.reserve(
      oracle.stats.horizontal_segments * 2 +
      oracle.stats.vertical_segments);
  horizontal_y.reserve(oracle.stats.horizontal_segments);
  for (const DirectedSegmentI64 &segment : oracle.segments) {
    if (segment.axis == SegmentAxis::horizontal) {
      horizontal_y.push_back(segment.fixed);
      sweep_events.push_back(
          {segment.lo,
           db::cuda_manhattan_contour::detail::SweepEvent::
               AddHorizontal,
           segment.fixed, segment.fixed});
      sweep_events.push_back(
          {segment.hi,
           db::cuda_manhattan_contour::detail::SweepEvent::
               RemoveHorizontal,
           segment.fixed, segment.fixed});
    } else {
      sweep_events.push_back(
          {segment.fixed,
           db::cuda_manhattan_contour::detail::SweepEvent::QueryVertical,
           segment.lo, segment.hi});
    }
  }
  if (!db::cuda_manhattan_contour::detail::
          cross_intersections_are_valid(
              sweep_events, horizontal_y, oracle.segments.size())) {
    oracle.stats.unexpected_crossings = 1;
    throw std::runtime_error(
        "world boundary has an unexpected perpendicular crossing");
  }

  std::sort(oracle.segments.begin(), oracle.segments.end(),
            canonical_less);
  validate_canonical_candidate(oracle.segments);
  oracle.boundary_sha256 = boundary_sha256(oracle.segments);
  oracle.boundary_fnv64 = boundary_fnv64(oracle.segments);
  if (!options.expected_boundary_sha256.empty() &&
      oracle.boundary_sha256 !=
          normalized_digest(options.expected_boundary_sha256,
                            "expected boundary SHA-256")) {
    throw std::runtime_error(
        "canonical boundary SHA-256 disagrees with expectation");
  }
  return oracle;
}

Comparison compare_candidate(
    const BoundaryOracle &oracle,
    const std::vector<DirectedSegmentI64> &candidate)
{
  Comparison result;
  try {
    validate_canonical_candidate(candidate);
  } catch (const std::exception &error) {
    result.message = error.what();
    return result;
  }
  const std::size_t common =
      std::min(oracle.segments.size(), candidate.size());
  for (std::size_t index = 0; index < common; ++index) {
    if (!same_segment(oracle.segments[index], candidate[index])) {
      result.first_mismatch = index;
      result.message = "candidate first differs from CPU oracle";
      return result;
    }
  }
  if (candidate.size() != oracle.segments.size()) {
    result.first_mismatch = common;
    result.message = "candidate boundary segment count differs";
    return result;
  }
  result.equal = true;
  result.first_mismatch = candidate.size();
  result.message = "exact canonical boundary match";
  return result;
}

std::string canonical_boundary_sha256(
    const std::vector<DirectedSegmentI64> &segments)
{
  validate_canonical_candidate(segments);
  return boundary_sha256(segments);
}

std::uint64_t canonical_boundary_fnv64(
    const std::vector<DirectedSegmentI64> &segments)
{
  validate_canonical_candidate(segments);
  return boundary_fnv64(segments);
}

std::vector<DirectedSegmentI64> read_candidate_stream(
    const std::string &path, const BoundaryOracle &oracle)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open boundary candidate: " + path);
  }
  const std::streamoff file_end = input.tellg();
  if (file_end < kCandidateHeaderBytes ||
      static_cast<std::uint64_t>(file_end) >
          std::numeric_limits<std::size_t>::max() ||
      file_end > std::numeric_limits<std::streamsize>::max()) {
    throw std::runtime_error("candidate stream size is invalid");
  }
  std::vector<std::uint8_t> storage(
      static_cast<std::size_t>(file_end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(storage.data()),
                  static_cast<std::streamsize>(file_end))) {
    throw std::runtime_error("short read of candidate stream");
  }
  if (std::memcmp(storage.data(), kCandidateMagic, 8) != 0 ||
      format::load_u32_le(storage.data() + 8) != kCandidateVersion ||
      format::load_u32_le(storage.data() + 12) !=
          kCandidateHeaderBytes ||
      format::load_u32_le(storage.data() + 16) != kEndianTag ||
      format::load_u32_le(storage.data() + 20) !=
          sizeof(DirectedSegmentI64)) {
    throw std::runtime_error("candidate stream header is invalid");
  }
  const std::uint64_t file_bytes =
      format::load_u64_le(storage.data() + 24);
  const std::uint64_t count =
      format::load_u64_le(storage.data() + 32);
  std::uint64_t payload_bytes = 0;
  std::uint64_t expected_bytes = 0;
  if (count > UINT64_C(64000000) ||
      !format::checked_mul_u64(
          count, sizeof(DirectedSegmentI64), payload_bytes) ||
      !format::checked_add_u64(
          kCandidateHeaderBytes, payload_bytes, expected_bytes) ||
      file_bytes != expected_bytes || file_bytes != storage.size()) {
    throw std::runtime_error(
        "candidate stream count/length is invalid");
  }
  if (hex_digest(storage.data() + 72, 32) != oracle.scene_sha256 ||
      std::any_of(storage.begin() + 104, storage.begin() + 128,
                  [](std::uint8_t byte) { return byte != 0; })) {
    throw std::runtime_error(
        "candidate stream scene identity/reserved bytes mismatch");
  }
  const std::string payload_digest =
      sha256(storage.data() + kCandidateHeaderBytes,
             static_cast<std::size_t>(payload_bytes));
  if (payload_digest != hex_digest(storage.data() + 40, 32)) {
    throw std::runtime_error(
        "candidate stream payload SHA-256 mismatch");
  }
  std::vector<DirectedSegmentI64> result;
  result.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    result.push_back(
        decode_segment(storage.data() + kCandidateHeaderBytes +
                       index * sizeof(DirectedSegmentI64)));
  }
  validate_canonical_candidate(result);
  return result;
}

void write_candidate_stream(const std::string &path,
                            const BoundaryOracle &oracle)
{
  validate_canonical_candidate(oracle.segments);
  if (normalized_digest(
          oracle.boundary_sha256, "candidate boundary SHA-256") !=
      boundary_sha256(oracle.segments)) {
    throw std::runtime_error(
        "candidate boundary SHA-256 disagrees with payload");
  }
  const std::uint64_t payload_bytes =
      oracle.segments.size() * sizeof(DirectedSegmentI64);
  const std::uint64_t file_bytes =
      kCandidateHeaderBytes + payload_bytes;
  std::array<std::uint8_t, kCandidateHeaderBytes> header{};
  std::memcpy(header.data(), kCandidateMagic, 8);
  format::store_u32_le(header.data() + 8, kCandidateVersion);
  format::store_u32_le(header.data() + 12, kCandidateHeaderBytes);
  format::store_u32_le(header.data() + 16, kEndianTag);
  format::store_u32_le(
      header.data() + 20, sizeof(DirectedSegmentI64));
  format::store_u64_le(header.data() + 24, file_bytes);
  format::store_u64_le(
      header.data() + 32, oracle.segments.size());
  const auto boundary_digest =
      digest_from_hex(oracle.boundary_sha256);
  const auto scene_digest = digest_from_hex(oracle.scene_sha256);
  std::memcpy(header.data() + 40, boundary_digest.data(), 32);
  std::memcpy(header.data() + 72, scene_digest.data(), 32);

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create candidate stream: " + path);
  }
  output.write(reinterpret_cast<const char *>(header.data()),
               header.size());
  std::array<std::uint8_t, 32> record{};
  for (const DirectedSegmentI64 &segment : oracle.segments) {
    encode_segment(segment, record.data());
    output.write(reinterpret_cast<const char *>(record.data()),
                 record.size());
  }
  output.flush();
  if (!output) {
    throw std::runtime_error("failed writing candidate stream: " + path);
  }
}

}  // namespace m2_boundary_oracle
}  // namespace klayout_cuda
