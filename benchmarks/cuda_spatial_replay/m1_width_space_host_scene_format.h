/*
 * Exact merged-M1 host-scene replay format.
 *
 * This is a narrow development bridge between db::CudaM1WidthSpaceScene and
 * the standalone CUDA M1 width/spacing island.  It is not a public KLayout
 * file format or ABI.
 *
 * All integer fields are little-endian.  The 256-byte transport header is
 * followed immediately by a canonical scene payload.  The payload is byte
 * for byte the stream hashed by cuda_m1_width_space_scene_digest(), so its
 * SHA-256 must equal scene_digest in the transport header.
 *
 * transport_digest is SHA-256 of the complete file with header bytes
 * [160, 192) treated as zero.  source_digest is SHA-256 of the input capture
 * file and is provenance metadata, not a claim that the source was merged.
 */

#ifndef KLAYOUT_M1_WIDTH_SPACE_HOST_SCENE_FORMAT_H
#define KLAYOUT_M1_WIDTH_SPACE_HOST_SCENE_FORMAT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace klayout_m1ws_scene
{

static const std::uint8_t kFileMagic[8] =
  { 'K', 'M', '1', 'W', 'S', 'C', 'N', '1' };
static const std::uint8_t kSemanticMagic[8] =
  { 'K', 'M', '1', 'W', 'S', '0', '0', '1' };

constexpr std::uint32_t kFileVersion = 1;
constexpr std::uint32_t kEndianTag = UINT32_C (0x01020304);
constexpr std::uint32_t kFlagCanonicalPayload = UINT32_C (1) << 0;
constexpr std::uint32_t kFlagMergedAssertion = UINT32_C (1) << 1;
constexpr std::uint32_t kFlagSourceDigest = UINT32_C (1) << 2;
constexpr std::uint32_t kRequiredFlags =
  kFlagCanonicalPayload | kFlagMergedAssertion | kFlagSourceDigest;

constexpr std::uint64_t kFileHeaderBytes = UINT64_C (256);
constexpr std::uint64_t kSemanticHeaderBytes = UINT64_C (128);
constexpr std::uint64_t kContextRecordBytes = UINT64_C (24);
constexpr std::uint64_t kMetalContextRecordBytes = UINT64_C (20);
constexpr std::uint64_t kCellRecordBytes = UINT64_C (32);
constexpr std::uint64_t kPolygonRecordBytes = UINT64_C (48);
constexpr std::uint64_t kEdgeRecordBytes = UINT64_C (32);

constexpr std::size_t kSceneDigestOffset = 128;
constexpr std::size_t kTransportDigestOffset = 160;
constexpr std::size_t kSourceDigestOffset = 192;
constexpr std::size_t kDigestBytes = 32;

struct FileLayoutV1
{
  std::uint64_t file_bytes;
  std::uint64_t payload_offset;
  std::uint64_t payload_bytes;
  std::uint64_t contexts_offset;
  std::uint64_t metal_contexts_offset;
  std::uint64_t cells_offset;
  std::uint64_t polygons_offset;
  std::uint64_t edges_offset;
};

struct FileHeaderV1
{
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t endian_tag;
  std::uint32_t flags;
  FileLayoutV1 layout;
  std::uint64_t context_count;
  std::uint64_t metal_context_count;
  std::uint64_t cell_count;
  std::uint64_t polygon_count;
  std::uint64_t edge_count;
  std::array<std::uint8_t, kDigestBytes> scene_digest;
  std::array<std::uint8_t, kDigestBytes> transport_digest;
  std::array<std::uint8_t, kDigestBytes> source_digest;
  std::uint32_t source_layer;
  std::uint32_t source_datatype;
  std::uint64_t reserved[3];
};

struct SemanticHeaderV1
{
  std::uint32_t format_version;
  std::uint32_t dbu_per_micron;
  std::uint32_t root_cell;
  std::uint32_t reserved;
  std::int64_t width_distance;
  std::int64_t spacing_distance;
  std::uint64_t context_count;
  std::uint64_t metal_context_count;
  std::uint64_t cell_count;
  std::uint64_t polygon_count;
  std::uint64_t edge_count;
  std::uint64_t flat_polygon_count;
  std::uint64_t flat_edge_count;
  std::int64_t scene_left;
  std::int64_t scene_bottom;
  std::int64_t scene_right;
  std::int64_t scene_top;
};

inline std::uint32_t load_u32_le (const std::uint8_t *p)
{
  return
    std::uint32_t (p[0]) |
    (std::uint32_t (p[1]) << 8) |
    (std::uint32_t (p[2]) << 16) |
    (std::uint32_t (p[3]) << 24);
}

inline std::uint64_t load_u64_le (const std::uint8_t *p)
{
  std::uint64_t value = 0;
  for (unsigned int i = 0; i < 8; ++i) {
    value |= std::uint64_t (p[i]) << (i * 8);
  }
  return value;
}

inline std::int64_t load_i64_le (const std::uint8_t *p)
{
  const std::uint64_t bits = load_u64_le (p);
  std::int64_t value = 0;
  std::memcpy (&value, &bits, sizeof (value));
  return value;
}

inline void store_u32_le (std::uint8_t *p, std::uint32_t value)
{
  for (unsigned int i = 0; i < 4; ++i) {
    p[i] = std::uint8_t (value >> (i * 8));
  }
}

inline void store_u64_le (std::uint8_t *p, std::uint64_t value)
{
  for (unsigned int i = 0; i < 8; ++i) {
    p[i] = std::uint8_t (value >> (i * 8));
  }
}

inline void store_i64_le (std::uint8_t *p, std::int64_t value)
{
  std::uint64_t bits = 0;
  std::memcpy (&bits, &value, sizeof (bits));
  store_u64_le (p, bits);
}

inline bool checked_add_u64 (
  std::uint64_t a, std::uint64_t b, std::uint64_t &result)
{
  if (b > std::numeric_limits<std::uint64_t>::max () - a) {
    return false;
  }
  result = a + b;
  return true;
}

inline bool checked_mul_u64 (
  std::uint64_t a, std::uint64_t b, std::uint64_t &result)
{
  if (a && b > std::numeric_limits<std::uint64_t>::max () / a) {
    return false;
  }
  result = a * b;
  return true;
}

inline bool append_section (
  std::uint64_t &cursor, std::uint64_t count, std::uint64_t record_bytes,
  std::uint64_t &offset)
{
  std::uint64_t bytes = 0;
  offset = cursor;
  return
    checked_mul_u64 (count, record_bytes, bytes) &&
    checked_add_u64 (cursor, bytes, cursor);
}

inline bool compute_file_layout (
  std::uint64_t context_count,
  std::uint64_t metal_context_count,
  std::uint64_t cell_count,
  std::uint64_t polygon_count,
  std::uint64_t edge_count,
  FileLayoutV1 &layout)
{
  layout.payload_offset = kFileHeaderBytes;
  std::uint64_t cursor = 0;
  if (! checked_add_u64 (
        layout.payload_offset, kSemanticHeaderBytes, cursor) ||
      ! append_section (
        cursor, context_count, kContextRecordBytes,
        layout.contexts_offset) ||
      ! append_section (
        cursor, metal_context_count, kMetalContextRecordBytes,
        layout.metal_contexts_offset) ||
      ! append_section (
        cursor, cell_count, kCellRecordBytes, layout.cells_offset) ||
      ! append_section (
        cursor, polygon_count, kPolygonRecordBytes,
        layout.polygons_offset) ||
      ! append_section (
        cursor, edge_count, kEdgeRecordBytes, layout.edges_offset)) {
    return false;
  }
  layout.file_bytes = cursor;
  layout.payload_bytes = cursor - layout.payload_offset;
  return true;
}

inline std::array<std::uint8_t, kFileHeaderBytes> encode_file_header (
  const FileHeaderV1 &header)
{
  std::array<std::uint8_t, kFileHeaderBytes> bytes = {};
  std::memcpy (bytes.data (), kFileMagic, sizeof (kFileMagic));
  store_u32_le (bytes.data () + 8, header.version);
  store_u32_le (bytes.data () + 12, header.header_bytes);
  store_u32_le (bytes.data () + 16, header.endian_tag);
  store_u32_le (bytes.data () + 20, header.flags);
  store_u64_le (bytes.data () + 24, header.layout.file_bytes);
  store_u64_le (bytes.data () + 32, header.layout.payload_offset);
  store_u64_le (bytes.data () + 40, header.layout.payload_bytes);
  store_u64_le (bytes.data () + 48, header.layout.contexts_offset);
  store_u64_le (bytes.data () + 56, header.layout.metal_contexts_offset);
  store_u64_le (bytes.data () + 64, header.layout.cells_offset);
  store_u64_le (bytes.data () + 72, header.layout.polygons_offset);
  store_u64_le (bytes.data () + 80, header.layout.edges_offset);
  store_u64_le (bytes.data () + 88, header.context_count);
  store_u64_le (bytes.data () + 96, header.metal_context_count);
  store_u64_le (bytes.data () + 104, header.cell_count);
  store_u64_le (bytes.data () + 112, header.polygon_count);
  store_u64_le (bytes.data () + 120, header.edge_count);
  std::memcpy (
    bytes.data () + kSceneDigestOffset,
    header.scene_digest.data (), kDigestBytes);
  std::memcpy (
    bytes.data () + kTransportDigestOffset,
    header.transport_digest.data (), kDigestBytes);
  std::memcpy (
    bytes.data () + kSourceDigestOffset,
    header.source_digest.data (), kDigestBytes);
  store_u32_le (bytes.data () + 224, header.source_layer);
  store_u32_le (bytes.data () + 228, header.source_datatype);
  store_u64_le (bytes.data () + 232, header.reserved[0]);
  store_u64_le (bytes.data () + 240, header.reserved[1]);
  store_u64_le (bytes.data () + 248, header.reserved[2]);
  return bytes;
}

inline bool decode_file_header (
  const std::uint8_t *bytes, std::size_t size, FileHeaderV1 &header)
{
  if (size < kFileHeaderBytes ||
      std::memcmp (bytes, kFileMagic, sizeof (kFileMagic)) != 0) {
    return false;
  }
  header.version = load_u32_le (bytes + 8);
  header.header_bytes = load_u32_le (bytes + 12);
  header.endian_tag = load_u32_le (bytes + 16);
  header.flags = load_u32_le (bytes + 20);
  header.layout.file_bytes = load_u64_le (bytes + 24);
  header.layout.payload_offset = load_u64_le (bytes + 32);
  header.layout.payload_bytes = load_u64_le (bytes + 40);
  header.layout.contexts_offset = load_u64_le (bytes + 48);
  header.layout.metal_contexts_offset = load_u64_le (bytes + 56);
  header.layout.cells_offset = load_u64_le (bytes + 64);
  header.layout.polygons_offset = load_u64_le (bytes + 72);
  header.layout.edges_offset = load_u64_le (bytes + 80);
  header.context_count = load_u64_le (bytes + 88);
  header.metal_context_count = load_u64_le (bytes + 96);
  header.cell_count = load_u64_le (bytes + 104);
  header.polygon_count = load_u64_le (bytes + 112);
  header.edge_count = load_u64_le (bytes + 120);
  std::memcpy (
    header.scene_digest.data (), bytes + kSceneDigestOffset, kDigestBytes);
  std::memcpy (
    header.transport_digest.data (),
    bytes + kTransportDigestOffset, kDigestBytes);
  std::memcpy (
    header.source_digest.data (), bytes + kSourceDigestOffset, kDigestBytes);
  header.source_layer = load_u32_le (bytes + 224);
  header.source_datatype = load_u32_le (bytes + 228);
  header.reserved[0] = load_u64_le (bytes + 232);
  header.reserved[1] = load_u64_le (bytes + 240);
  header.reserved[2] = load_u64_le (bytes + 248);
  return true;
}

inline std::array<std::uint8_t, kSemanticHeaderBytes>
encode_semantic_header (const SemanticHeaderV1 &header)
{
  std::array<std::uint8_t, kSemanticHeaderBytes> bytes = {};
  std::memcpy (bytes.data (), kSemanticMagic, sizeof (kSemanticMagic));
  store_u32_le (bytes.data () + 8, header.format_version);
  store_u32_le (bytes.data () + 12, header.dbu_per_micron);
  store_u32_le (bytes.data () + 16, header.root_cell);
  store_u32_le (bytes.data () + 20, header.reserved);
  store_i64_le (bytes.data () + 24, header.width_distance);
  store_i64_le (bytes.data () + 32, header.spacing_distance);
  store_u64_le (bytes.data () + 40, header.context_count);
  store_u64_le (bytes.data () + 48, header.metal_context_count);
  store_u64_le (bytes.data () + 56, header.cell_count);
  store_u64_le (bytes.data () + 64, header.polygon_count);
  store_u64_le (bytes.data () + 72, header.edge_count);
  store_u64_le (bytes.data () + 80, header.flat_polygon_count);
  store_u64_le (bytes.data () + 88, header.flat_edge_count);
  store_i64_le (bytes.data () + 96, header.scene_left);
  store_i64_le (bytes.data () + 104, header.scene_bottom);
  store_i64_le (bytes.data () + 112, header.scene_right);
  store_i64_le (bytes.data () + 120, header.scene_top);
  return bytes;
}

inline bool decode_semantic_header (
  const std::uint8_t *bytes, std::size_t size, SemanticHeaderV1 &header)
{
  if (size < kSemanticHeaderBytes ||
      std::memcmp (bytes, kSemanticMagic, sizeof (kSemanticMagic)) != 0) {
    return false;
  }
  header.format_version = load_u32_le (bytes + 8);
  header.dbu_per_micron = load_u32_le (bytes + 12);
  header.root_cell = load_u32_le (bytes + 16);
  header.reserved = load_u32_le (bytes + 20);
  header.width_distance = load_i64_le (bytes + 24);
  header.spacing_distance = load_i64_le (bytes + 32);
  header.context_count = load_u64_le (bytes + 40);
  header.metal_context_count = load_u64_le (bytes + 48);
  header.cell_count = load_u64_le (bytes + 56);
  header.polygon_count = load_u64_le (bytes + 64);
  header.edge_count = load_u64_le (bytes + 72);
  header.flat_polygon_count = load_u64_le (bytes + 80);
  header.flat_edge_count = load_u64_le (bytes + 88);
  header.scene_left = load_i64_le (bytes + 96);
  header.scene_bottom = load_i64_le (bytes + 104);
  header.scene_right = load_i64_le (bytes + 112);
  header.scene_top = load_i64_le (bytes + 120);
  return true;
}

}  // namespace klayout_m1ws_scene

#endif
