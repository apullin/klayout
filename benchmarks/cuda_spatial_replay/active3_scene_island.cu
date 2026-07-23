/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

// Standalone proof of a contiguous GPU-owned ACTIVE.3 scene island.
//
// The host validates KACTSCN1 and lowers the retained hierarchy to compact
// cell contexts.  The device then owns the complete WELL expansion, uniform
// index construction, ACTIVE edge stream, exact relation predicate, cull, and
// compact reduction.  ACTIVE edges are never flattened on either host or
// device.

#include "active3_exact_predicate.cuh"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using klayout_cuda::active3::DirectedEdge;
using klayout_cuda::active3::EdgePair;
using klayout_cuda::active3::Verdict;

constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kHeaderBytes = 256;
constexpr std::uint32_t kEndianTag = 0x01020304;
constexpr std::uint32_t kFormatFlags = 1;
constexpr std::uint32_t kCoordinateBits = 64;
constexpr std::uint32_t kLayerCount = 2;
constexpr std::uint32_t kWellLayer = 0;
constexpr std::uint32_t kActiveLayer = 1;
constexpr std::int64_t kGridCell = 2000;
constexpr double kQualifiedDbuMicrometres = 0.0005;
constexpr double kRuleDistanceMicrometres = 0.055;
constexpr std::int64_t kQualifiedDistance = 110;
constexpr std::int64_t kAcceptedCoordinateMagnitude =
    INT64_C(1000000000000);
constexpr std::uint64_t kDefaultMaxContexts = UINT64_C(4000000);
constexpr std::uint64_t kDefaultMaxGridCells = UINT64_C(16000000);
constexpr std::uint64_t kDefaultMaxMemberships = UINT64_C(100000000);
constexpr std::uint64_t kDefaultMaxPairWork = UINT64_C(2000000000000);
constexpr std::uint32_t kSampleCapacity = 16;

enum DeviceStatus : std::uint32_t {
  kDeviceOk = 0,
  kTransformOverflow = 1u << 0,
  kGridCounterOverflow = 1u << 1,
  kGridCapacityExceeded = 1u << 2,
  kInvalidDeviceRecord = 1u << 3,
};

#pragma pack(push, 1)
struct SceneHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t endian_tag;
  std::uint32_t flags;
  std::uint32_t coordinate_bits;
  std::uint32_t layer_count;
  std::uint32_t cell_record_bytes;
  std::uint32_t instance_record_bytes;
  std::uint32_t polygon_record_bytes;
  std::uint32_t edge_record_bytes;
  std::uint32_t well_layer;
  std::uint32_t well_datatype;
  std::uint32_t active_layer;
  std::uint32_t active_datatype;
  double dbu;
  std::uint64_t root_cell;
  std::uint64_t cell_count;
  std::uint64_t instance_count;
  std::uint64_t polygon_count;
  std::uint64_t edge_count;
  std::uint64_t names_offset;
  std::uint64_t names_bytes;
  std::uint64_t cells_offset;
  std::uint64_t cells_bytes;
  std::uint64_t instances_offset;
  std::uint64_t instances_bytes;
  std::uint64_t polygons_offset;
  std::uint64_t polygons_bytes;
  std::uint64_t edges_offset;
  std::uint64_t edges_bytes;
  std::uint64_t file_bytes;
  std::uint64_t payload_offset;
  std::uint64_t payload_bytes;
  std::uint8_t scene_sha256[32];
  std::uint64_t reserved;
};

struct CellRecord {
  std::uint64_t cell_id;
  std::uint64_t name_offset;
  std::uint32_t name_bytes;
  std::uint32_t local_layer_mask;
  std::uint32_t subtree_layer_mask;
  std::uint32_t flags;
  std::uint64_t instance_begin;
  std::uint64_t instance_count;
  std::uint64_t polygon_begin;
  std::uint64_t polygon_count;
  std::uint64_t edge_begin;
  std::uint64_t edge_count;
  std::int64_t local_bbox[2][4];
  std::int64_t subtree_bbox[2][4];
};

struct InstanceRecord {
  std::uint64_t instance_id;
  std::uint64_t parent_cell_id;
  std::uint64_t child_cell_id;
  std::uint64_t occurrence_count;
  std::int64_t dx;
  std::int64_t dy;
  std::int64_t ax;
  std::int64_t ay;
  std::int64_t bx;
  std::int64_t by;
  std::uint32_t columns;
  std::uint32_t rows;
  std::uint32_t transform_code;
  std::uint32_t flags;
};

struct PolygonRecord {
  std::uint64_t polygon_id;
  std::uint64_t cell_id;
  std::uint64_t edge_begin;
  std::uint32_t edge_count;
  std::uint32_t layer_code;
  std::int64_t bbox[4];
};

struct EdgeRecord {
  std::uint64_t edge_id;
  std::uint64_t polygon_id;
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
  std::uint32_t contour_index;
  std::uint32_t layer_code;
};
#pragma pack(pop)

static_assert(sizeof(SceneHeader) == 256, "KACTSCN1 header ABI changed");
static_assert(offsetof(SceneHeader, scene_sha256) == 216,
              "KACTSCN1 digest offset changed");
static_assert(sizeof(CellRecord) == 208, "KACTSCN1 cell ABI changed");
static_assert(sizeof(InstanceRecord) == 96, "KACTSCN1 instance ABI changed");
static_assert(sizeof(PolygonRecord) == 64, "KACTSCN1 polygon ABI changed");
static_assert(sizeof(EdgeRecord) == 56, "KACTSCN1 edge ABI changed");

class SceneError : public std::runtime_error {
 public:
  explicit SceneError(const std::string &message)
      : std::runtime_error(message) {}
};

double milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool checked_add_u64(std::uint64_t a, std::uint64_t b,
                     std::uint64_t *result) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    return false;
  }
  *result = a + b;
  return true;
}

bool checked_mul_u64(std::uint64_t a, std::uint64_t b,
                     std::uint64_t *result) {
  if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
    return false;
  }
  *result = a * b;
  return true;
}

std::uint64_t align_up_64(std::uint64_t value) {
  std::uint64_t expanded = 0;
  if (!checked_add_u64(value, 63, &expanded)) {
    throw SceneError("aligned offset overflow");
  }
  return expanded & ~UINT64_C(63);
}

bool bounded_coordinate(std::int64_t value) {
  return value >= -kAcceptedCoordinateMagnitude &&
         value <= kAcceptedCoordinateMagnitude;
}

struct Sha256 {
  std::array<std::uint32_t, 8> state = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::array<std::uint8_t, 64> block{};
  std::uint64_t total_bytes = 0;
  std::size_t block_bytes = 0;

  static std::uint32_t rotate_right(std::uint32_t value,
                                    unsigned int amount) {
    return (value >> amount) | (value << (32 - amount));
  }

  void transform(const std::uint8_t *input) {
    static constexpr std::uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::uint32_t words[64];
    for (int i = 0; i < 16; ++i) {
      words[i] = static_cast<std::uint32_t>(input[i * 4]) << 24 |
                 static_cast<std::uint32_t>(input[i * 4 + 1]) << 16 |
                 static_cast<std::uint32_t>(input[i * 4 + 2]) << 8 |
                 static_cast<std::uint32_t>(input[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                               rotate_right(words[i - 15], 18) ^
                               (words[i - 15] >> 3);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                               rotate_right(words[i - 2], 19) ^
                               (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t sum1 =
          rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t temp1 =
          h + sum1 + choose + constants[i] + words[i];
      const std::uint32_t sum0 =
          rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  void update(const std::uint8_t *data, std::size_t bytes) {
    total_bytes += bytes;
    while (bytes) {
      const std::size_t available = block.size() - block_bytes;
      const std::size_t take = std::min(available, bytes);
      std::memcpy(block.data() + block_bytes, data, take);
      block_bytes += take;
      data += take;
      bytes -= take;
      if (block_bytes == block.size()) {
        transform(block.data());
        block_bytes = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finish() {
    const std::uint64_t bit_count = total_bytes * 8;
    const std::uint8_t marker = 0x80;
    update(&marker, 1);
    const std::uint8_t zero = 0;
    while (block_bytes != 56) {
      update(&zero, 1);
    }
    std::uint8_t length[8];
    for (int i = 0; i < 8; ++i) {
      length[7 - i] = static_cast<std::uint8_t>(bit_count >> (i * 8));
    }
    update(length, sizeof(length));
    std::array<std::uint8_t, 32> digest{};
    for (int i = 0; i < 8; ++i) {
      digest[i * 4] = static_cast<std::uint8_t>(state[i] >> 24);
      digest[i * 4 + 1] = static_cast<std::uint8_t>(state[i] >> 16);
      digest[i * 4 + 2] = static_cast<std::uint8_t>(state[i] >> 8);
      digest[i * 4 + 3] = static_cast<std::uint8_t>(state[i]);
    }
    return digest;
  }
};

std::string hex_digest(const std::uint8_t *digest, std::size_t bytes) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < bytes; ++i) {
    output << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  return output.str();
}

struct Box {
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;

  bool operator==(const Box &other) const {
    return left == other.left && bottom == other.bottom &&
           right == other.right && top == other.top;
  }

  bool operator!=(const Box &other) const { return !(*this == other); }
};

using LayerBoxes = std::array<std::optional<Box>, 2>;

Box box_from(const std::int64_t values[4]) {
  return {values[0], values[1], values[2], values[3]};
}

void include_box(std::optional<Box> *destination, const Box &source) {
  if (!*destination) {
    *destination = source;
    return;
  }
  destination->value().left =
      std::min(destination->value().left, source.left);
  destination->value().bottom =
      std::min(destination->value().bottom, source.bottom);
  destination->value().right =
      std::max(destination->value().right, source.right);
  destination->value().top =
      std::max(destination->value().top, source.top);
}

struct Matrix {
  int xx;
  int xy;
  int yx;
  int yy;
};

constexpr Matrix kTransforms[8] = {
    {1, 0, 0, 1},   {0, -1, 1, 0}, {-1, 0, 0, -1}, {0, 1, -1, 0},
    {1, 0, 0, -1},  {0, 1, 1, 0},  {-1, 0, 0, 1},  {0, -1, -1, 0}};

std::pair<__int128, __int128> transform_128(std::uint32_t code,
                                            __int128 x, __int128 y) {
  if (code >= 8) {
    throw SceneError("invalid transform code");
  }
  const Matrix matrix = kTransforms[code];
  return {matrix.xx * x + matrix.xy * y,
          matrix.yx * x + matrix.yy * y};
}

std::uint32_t compose_transform(std::uint32_t outer, std::uint32_t inner) {
  const Matrix a = kTransforms[outer];
  const Matrix b = kTransforms[inner];
  const Matrix product = {
      a.xx * b.xx + a.xy * b.yx, a.xx * b.xy + a.xy * b.yy,
      a.yx * b.xx + a.yy * b.yx, a.yx * b.xy + a.yy * b.yy};
  for (std::uint32_t code = 0; code < 8; ++code) {
    const Matrix candidate = kTransforms[code];
    if (candidate.xx == product.xx && candidate.xy == product.xy &&
        candidate.yx == product.yx && candidate.yy == product.yy) {
      return code;
    }
  }
  throw SceneError("orthogonal transform composition escaped the group");
}

std::int64_t narrow_i64(__int128 value, const char *what) {
  if (value < std::numeric_limits<std::int64_t>::min() ||
      value > std::numeric_limits<std::int64_t>::max()) {
    throw SceneError(std::string(what) + " overflows signed int64");
  }
  return static_cast<std::int64_t>(value);
}

Box transform_array_box(const Box &source, const InstanceRecord &instance) {
  std::array<std::pair<__int128, __int128>, 4> corners = {
      transform_128(instance.transform_code, source.left, source.bottom),
      transform_128(instance.transform_code, source.left, source.top),
      transform_128(instance.transform_code, source.right, source.bottom),
      transform_128(instance.transform_code, source.right, source.top)};
  __int128 left = corners[0].first + instance.dx;
  __int128 right = left;
  __int128 bottom = corners[0].second + instance.dy;
  __int128 top = bottom;
  for (const auto &corner : corners) {
    const __int128 x = corner.first + instance.dx;
    const __int128 y = corner.second + instance.dy;
    left = std::min(left, x);
    right = std::max(right, x);
    bottom = std::min(bottom, y);
    top = std::max(top, y);
  }
  const __int128 ax =
      static_cast<__int128>(instance.columns - 1) * instance.ax;
  const __int128 ay =
      static_cast<__int128>(instance.columns - 1) * instance.ay;
  const __int128 bx = static_cast<__int128>(instance.rows - 1) * instance.bx;
  const __int128 by = static_cast<__int128>(instance.rows - 1) * instance.by;
  const std::array<__int128, 4> xs = {0, ax, bx, ax + bx};
  const std::array<__int128, 4> ys = {0, ay, by, ay + by};
  return {narrow_i64(left + *std::min_element(xs.begin(), xs.end()),
                     "array bbox left"),
          narrow_i64(bottom + *std::min_element(ys.begin(), ys.end()),
                     "array bbox bottom"),
          narrow_i64(right + *std::max_element(xs.begin(), xs.end()),
                     "array bbox right"),
          narrow_i64(top + *std::max_element(ys.begin(), ys.end()),
                     "array bbox top")};
}

struct PairHash {
  std::size_t operator()(
      const std::pair<std::int64_t, std::int64_t> &point) const {
    const std::uint64_t x = static_cast<std::uint64_t>(point.first);
    const std::uint64_t y = static_cast<std::uint64_t>(point.second);
    return static_cast<std::size_t>(
        x ^ (y + UINT64_C(0x9e3779b97f4a7c15) + (x << 6) + (x >> 2)));
  }
};

struct LoadedScene {
  std::vector<std::uint8_t> storage;
  SceneHeader header{};
  const char *names = nullptr;
  const CellRecord *cells = nullptr;
  const InstanceRecord *instances = nullptr;
  const PolygonRecord *polygons = nullptr;
  const EdgeRecord *edges = nullptr;
  std::int64_t distance = 0;
};

void require_range(std::uint64_t offset, std::uint64_t bytes,
                   std::uint64_t file_bytes, const char *what) {
  std::uint64_t end = 0;
  if (!checked_add_u64(offset, bytes, &end) || offset < kHeaderBytes ||
      end > file_bytes) {
    throw SceneError(std::string(what) + " section is outside the file");
  }
  if (offset & 63u) {
    throw SceneError(std::string(what) + " section is not 64-byte aligned");
  }
}

void require_zero(const std::vector<std::uint8_t> &data,
                  std::uint64_t begin, std::uint64_t end,
                  const char *what) {
  if (begin > end || end > data.size()) {
    throw SceneError(std::string(what) + " has an invalid range");
  }
  if (std::any_of(data.begin() + begin, data.begin() + end,
                  [](std::uint8_t value) { return value != 0; })) {
    throw SceneError(std::string(what) + " contains nonzero padding");
  }
}

LoadedScene load_and_validate(const std::string &path) {
  LoadedScene scene;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw SceneError("cannot open scene: " + path);
  }
  const std::streamoff end = input.tellg();
  if (end < static_cast<std::streamoff>(sizeof(SceneHeader))) {
    throw SceneError("scene is shorter than the fixed header");
  }
  if (static_cast<std::uint64_t>(end) >
      std::numeric_limits<std::size_t>::max()) {
    throw SceneError("scene is too large for this host");
  }
  scene.storage.resize(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(scene.storage.data()), end)) {
    throw SceneError("short read while loading scene");
  }
  std::memcpy(&scene.header, scene.storage.data(), sizeof(scene.header));
  const SceneHeader &header = scene.header;
  const char expected_magic[8] = {'K', 'A', 'C', 'T', 'S', 'C', 'N', '\0'};
  if (std::memcmp(header.magic, expected_magic, 8) != 0) {
    throw SceneError("bad KACTSCN1 magic");
  }
  const std::uint16_t endian_probe = 1;
  if (*reinterpret_cast<const std::uint8_t *>(&endian_probe) != 1) {
    throw SceneError("this reader requires a little-endian host");
  }
  if (header.version != kVersion || header.header_bytes != kHeaderBytes ||
      header.endian_tag != kEndianTag || header.flags != kFormatFlags ||
      header.coordinate_bits != kCoordinateBits ||
      header.layer_count != kLayerCount) {
    throw SceneError("unsupported KACTSCN1 header configuration");
  }
  if (header.cell_record_bytes != sizeof(CellRecord) ||
      header.instance_record_bytes != sizeof(InstanceRecord) ||
      header.polygon_record_bytes != sizeof(PolygonRecord) ||
      header.edge_record_bytes != sizeof(EdgeRecord)) {
    throw SceneError("KACTSCN1 record-size mismatch");
  }
  if (header.well_layer == header.active_layer &&
      header.well_datatype == header.active_datatype) {
    throw SceneError("WELL and ACTIVE source layers alias");
  }
  if (!std::isfinite(header.dbu) ||
      header.dbu != kQualifiedDbuMicrometres) {
    throw SceneError(
        "scene DBU is outside the qualified 0.0005um ACTIVE.3 domain");
  }
  const long double distance =
      static_cast<long double>(kRuleDistanceMicrometres) /
      static_cast<long double>(header.dbu);
  const long double rounded = std::round(distance);
  if (std::fabs(distance - rounded) >
          std::numeric_limits<double>::epsilon() * 8 * std::fabs(distance) ||
      rounded != kQualifiedDistance) {
    throw SceneError("55nm rule distance is not exactly 110 scene units");
  }
  scene.distance = static_cast<std::int64_t>(rounded);
  if (!header.cell_count || header.root_cell >= header.cell_count) {
    throw SceneError("invalid cell count or root cell");
  }
  if (header.reserved || header.file_bytes != scene.storage.size() ||
      header.payload_offset != kHeaderBytes ||
      header.payload_bytes != scene.storage.size() - kHeaderBytes) {
    throw SceneError("noncanonical file/payload size or reserved field");
  }

  std::uint64_t expected_bytes = 0;
  if (!checked_mul_u64(header.cell_count, sizeof(CellRecord),
                       &expected_bytes) ||
      expected_bytes != header.cells_bytes ||
      !checked_mul_u64(header.instance_count, sizeof(InstanceRecord),
                       &expected_bytes) ||
      expected_bytes != header.instances_bytes ||
      !checked_mul_u64(header.polygon_count, sizeof(PolygonRecord),
                       &expected_bytes) ||
      expected_bytes != header.polygons_bytes ||
      !checked_mul_u64(header.edge_count, sizeof(EdgeRecord),
                       &expected_bytes) ||
      expected_bytes != header.edges_bytes) {
    throw SceneError("record count/section byte mismatch");
  }
  require_range(header.names_offset, header.names_bytes, header.file_bytes,
                "names");
  require_range(header.cells_offset, header.cells_bytes, header.file_bytes,
                "cells");
  require_range(header.instances_offset, header.instances_bytes,
                header.file_bytes, "instances");
  require_range(header.polygons_offset, header.polygons_bytes,
                header.file_bytes, "polygons");
  require_range(header.edges_offset, header.edges_bytes, header.file_bytes,
                "edges");
  if (header.names_offset != kHeaderBytes) {
    throw SceneError("names section does not begin after the header");
  }
  std::uint64_t previous_end = kHeaderBytes;
  const std::array<std::pair<std::uint64_t, std::uint64_t>, 5> sections = {{
      {header.names_offset, header.names_bytes},
      {header.cells_offset, header.cells_bytes},
      {header.instances_offset, header.instances_bytes},
      {header.polygons_offset, header.polygons_bytes},
      {header.edges_offset, header.edges_bytes},
  }};
  for (const auto &section : sections) {
    if (section.first != align_up_64(previous_end)) {
      throw SceneError("noncanonical section offset");
    }
    require_zero(scene.storage, previous_end, section.first,
                 "inter-section padding");
    previous_end = section.first + section.second;
  }
  if (align_up_64(previous_end) != header.file_bytes) {
    throw SceneError("noncanonical aligned file size");
  }
  require_zero(scene.storage, previous_end, header.file_bytes,
               "trailing padding");

  std::vector<std::uint8_t> digest_input(scene.storage);
  std::fill(digest_input.begin() + offsetof(SceneHeader, scene_sha256),
            digest_input.begin() + kHeaderBytes, 0);
  Sha256 sha;
  sha.update(digest_input.data(), digest_input.size());
  const auto digest = sha.finish();
  if (!std::equal(digest.begin(), digest.end(), header.scene_sha256)) {
    throw SceneError("scene SHA-256 mismatch");
  }

  scene.names = reinterpret_cast<const char *>(
      scene.storage.data() + header.names_offset);
  scene.cells = reinterpret_cast<const CellRecord *>(
      scene.storage.data() + header.cells_offset);
  scene.instances = reinterpret_cast<const InstanceRecord *>(
      scene.storage.data() + header.instances_offset);
  scene.polygons = reinterpret_cast<const PolygonRecord *>(
      scene.storage.data() + header.polygons_offset);
  scene.edges = reinterpret_cast<const EdgeRecord *>(
      scene.storage.data() + header.edges_offset);

  std::uint64_t next_name = 0;
  std::uint64_t next_instance = 0;
  std::uint64_t next_polygon = 0;
  std::uint64_t next_edge = 0;
  std::string previous_name;
  std::vector<LayerBoxes> recomputed_local(header.cell_count);
  for (std::uint64_t cell_id = 0; cell_id < header.cell_count; ++cell_id) {
    const CellRecord &cell = scene.cells[cell_id];
    if (cell.cell_id != cell_id || cell.flags ||
        (cell.local_layer_mask & ~3u) ||
        (cell.subtree_layer_mask & ~3u) ||
        (cell.local_layer_mask & ~cell.subtree_layer_mask)) {
      throw SceneError("invalid cell ID, flags, or layer mask");
    }
    std::uint64_t name_end = 0;
    if (cell.name_offset != next_name || !cell.name_bytes ||
        !checked_add_u64(cell.name_offset, cell.name_bytes, &name_end) ||
        name_end > header.names_bytes) {
      throw SceneError("noncontiguous or invalid cell name");
    }
    const std::string name(scene.names + cell.name_offset, cell.name_bytes);
    if (name.find('\0') != std::string::npos ||
        (cell_id && name <= previous_name)) {
      throw SceneError("cell names are not strict canonical bytes");
    }
    previous_name = name;
    next_name = name_end;
    if (cell.instance_begin != next_instance ||
        cell.polygon_begin != next_polygon || cell.edge_begin != next_edge ||
        !checked_add_u64(next_instance, cell.instance_count,
                         &next_instance) ||
        !checked_add_u64(next_polygon, cell.polygon_count, &next_polygon) ||
        !checked_add_u64(next_edge, cell.edge_count, &next_edge) ||
        next_instance > header.instance_count ||
        next_polygon > header.polygon_count || next_edge > header.edge_count) {
      throw SceneError("cell ranges do not partition record tables");
    }
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
      const bool local_present = cell.local_layer_mask & (1u << layer);
      const bool subtree_present = cell.subtree_layer_mask & (1u << layer);
      const Box local = box_from(cell.local_bbox[layer]);
      const Box subtree = box_from(cell.subtree_bbox[layer]);
      if (local_present) {
        if (local.left > local.right || local.bottom > local.top) {
          throw SceneError("invalid local cell bbox");
        }
      } else if (local != Box{0, 0, 0, 0}) {
        throw SceneError("nonzero absent local cell bbox");
      }
      if (subtree_present) {
        if (subtree.left > subtree.right || subtree.bottom > subtree.top) {
          throw SceneError("invalid subtree cell bbox");
        }
      } else if (subtree != Box{0, 0, 0, 0}) {
        throw SceneError("nonzero absent subtree cell bbox");
      }
      for (std::int64_t coordinate : cell.local_bbox[layer]) {
        if (!bounded_coordinate(coordinate)) {
          throw SceneError("cell local bbox exceeds qualified coordinate bound");
        }
      }
      for (std::int64_t coordinate : cell.subtree_bbox[layer]) {
        if (!bounded_coordinate(coordinate)) {
          throw SceneError(
              "cell subtree bbox exceeds qualified coordinate bound");
        }
      }
    }
  }
  if (next_name != header.names_bytes ||
      next_instance != header.instance_count ||
      next_polygon != header.polygon_count || next_edge != header.edge_count) {
    throw SceneError("cell ranges do not exactly partition sections");
  }

  std::uint64_t instance_owner = 0;
  for (std::uint64_t id = 0; id < header.instance_count; ++id) {
    while (instance_owner + 1 < header.cell_count &&
           id >= scene.cells[instance_owner].instance_begin +
                     scene.cells[instance_owner].instance_count) {
      ++instance_owner;
    }
    const InstanceRecord &instance = scene.instances[id];
    std::uint64_t occurrences = 0;
    if (instance.instance_id != id ||
        instance.parent_cell_id != instance_owner ||
        instance.child_cell_id >= header.cell_count || !instance.columns ||
        !instance.rows ||
        !checked_mul_u64(instance.columns, instance.rows, &occurrences) ||
        occurrences != instance.occurrence_count ||
        instance.transform_code >= 8 || instance.flags ||
        (instance.columns == 1 && (instance.ax || instance.ay)) ||
        (instance.rows == 1 && (instance.bx || instance.by)) ||
        (instance.columns > 1 && !instance.ax && !instance.ay) ||
        (instance.rows > 1 && !instance.bx && !instance.by)) {
      throw SceneError("malformed instance/array record");
    }
    const std::int64_t values[] = {
        instance.dx, instance.dy, instance.ax,
        instance.ay, instance.bx, instance.by};
    for (std::int64_t value : values) {
      if (!bounded_coordinate(value)) {
        throw SceneError("instance coordinate exceeds qualified bound");
      }
    }
    const __int128 last_x =
        static_cast<__int128>(instance.columns - 1) * instance.ax +
        static_cast<__int128>(instance.rows - 1) * instance.bx + instance.dx;
    const __int128 last_y =
        static_cast<__int128>(instance.columns - 1) * instance.ay +
        static_cast<__int128>(instance.rows - 1) * instance.by + instance.dy;
    narrow_i64(last_x, "array final x origin");
    narrow_i64(last_y, "array final y origin");
  }

  std::uint64_t polygon_owner = 0;
  std::uint64_t polygon_edge_end = 0;
  for (std::uint64_t id = 0; id < header.polygon_count; ++id) {
    while (polygon_owner + 1 < header.cell_count &&
           id >= scene.cells[polygon_owner].polygon_begin +
                     scene.cells[polygon_owner].polygon_count) {
      ++polygon_owner;
    }
    const PolygonRecord &polygon = scene.polygons[id];
    std::uint64_t edge_end = 0;
    if (polygon.polygon_id != id || polygon.cell_id != polygon_owner ||
        polygon.layer_code >= 2 || polygon.edge_count < 4 ||
        polygon.edge_begin != polygon_edge_end ||
        !checked_add_u64(polygon.edge_begin, polygon.edge_count, &edge_end) ||
        edge_end > header.edge_count || polygon.bbox[0] > polygon.bbox[2] ||
        polygon.bbox[1] > polygon.bbox[3]) {
      throw SceneError("malformed polygon record");
    }
    polygon_edge_end = edge_end;
    const Box box = box_from(polygon.bbox);
    include_box(&recomputed_local[polygon.cell_id][polygon.layer_code], box);
    std::unordered_set<std::pair<std::int64_t, std::int64_t>, PairHash>
        vertices;
    std::int64_t min_x = std::numeric_limits<std::int64_t>::max();
    std::int64_t min_y = std::numeric_limits<std::int64_t>::max();
    std::int64_t max_x = std::numeric_limits<std::int64_t>::min();
    std::int64_t max_y = std::numeric_limits<std::int64_t>::min();
    __int128 twice_area = 0;
    for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
      const EdgeRecord &edge = scene.edges[polygon.edge_begin + local];
      const EdgeRecord &following =
          scene.edges[polygon.edge_begin +
                      (local + 1 == polygon.edge_count ? 0 : local + 1)];
      if (edge.edge_id != polygon.edge_begin + local ||
          edge.polygon_id != id || edge.contour_index != local ||
          edge.layer_code != polygon.layer_code ||
          (edge.x1 == edge.x2 && edge.y1 == edge.y2) ||
          !(edge.x1 == edge.x2 || edge.y1 == edge.y2) ||
          edge.x2 != following.x1 || edge.y2 != following.y1) {
        throw SceneError("malformed, non-Manhattan, or open directed edge");
      }
      const std::int64_t coordinates[] = {edge.x1, edge.y1, edge.x2, edge.y2};
      for (std::int64_t coordinate : coordinates) {
        if (!bounded_coordinate(coordinate)) {
          throw SceneError("edge coordinate exceeds qualified bound");
        }
      }
      if (!vertices.emplace(edge.x1, edge.y1).second) {
        throw SceneError("polygon repeats a contour vertex");
      }
      min_x = std::min(min_x, edge.x1);
      min_y = std::min(min_y, edge.y1);
      max_x = std::max(max_x, edge.x1);
      max_y = std::max(max_y, edge.y1);
      twice_area += static_cast<__int128>(edge.x1) * edge.y2 -
                    static_cast<__int128>(edge.x2) * edge.y1;
    }
    // KLayout canonicalizes polygon hulls clockwise.  Direction is semantic
    // for EdgeRelationFilter, so a counter-clockwise template is outside this
    // island's exact domain instead of being silently reinterpreted.
    if (twice_area >= 0 ||
        Box{min_x, min_y, max_x, max_y} != box) {
      throw SceneError(
          "polygon is not clockwise or its polygon bbox mismatches");
    }
  }
  if (polygon_edge_end != header.edge_count) {
    throw SceneError("polygon ranges do not partition the edge table");
  }
  for (std::uint64_t cell = 0; cell < header.cell_count; ++cell) {
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
      const bool present = scene.cells[cell].local_layer_mask & (1u << layer);
      if (present != recomputed_local[cell][layer].has_value() ||
          (present && *recomputed_local[cell][layer] !=
                          box_from(scene.cells[cell].local_bbox[layer]))) {
        throw SceneError("recomputed local cell bbox mismatch");
      }
    }
  }

  std::vector<std::uint8_t> state(header.cell_count, 0);
  std::vector<LayerBoxes> subtree(header.cell_count);
  std::vector<std::uint8_t> reachable(header.cell_count, 0);
  const auto visit = [&](const auto &self, std::uint64_t cell_id)
      -> const LayerBoxes & {
    if (state[cell_id] == 1) {
      throw SceneError("hierarchy cycle");
    }
    if (state[cell_id] == 2) {
      reachable[cell_id] = 1;
      return subtree[cell_id];
    }
    state[cell_id] = 1;
    reachable[cell_id] = 1;
    subtree[cell_id] = recomputed_local[cell_id];
    const CellRecord &cell = scene.cells[cell_id];
    for (std::uint64_t index = 0; index < cell.instance_count; ++index) {
      const InstanceRecord &instance =
          scene.instances[cell.instance_begin + index];
      const LayerBoxes &child = self(self, instance.child_cell_id);
      for (std::uint32_t layer = 0; layer < 2; ++layer) {
        if (child[layer]) {
          include_box(&subtree[cell_id][layer],
                      transform_array_box(*child[layer], instance));
        }
      }
    }
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
      const bool present =
          scene.cells[cell_id].subtree_layer_mask & (1u << layer);
      if (present != subtree[cell_id][layer].has_value() ||
          (present && *subtree[cell_id][layer] !=
                          box_from(scene.cells[cell_id]
                                       .subtree_bbox[layer]))) {
        throw SceneError("recomputed subtree cell bbox mismatch");
      }
    }
    state[cell_id] = 2;
    return subtree[cell_id];
  };
  visit(visit, header.root_cell);
  if (std::find(reachable.begin(), reachable.end(), 0) != reachable.end()) {
    throw SceneError("cell table contains unreachable cells");
  }
  return scene;
}

struct ContextGpu {
  std::uint32_t cell;
  std::uint32_t transform;
  std::int64_t tx;
  std::int64_t ty;
};

struct CellGpu {
  std::uint64_t well_edge_begin;
  std::uint64_t active_edge_begin;
  std::uint32_t well_edge_count;
  std::uint32_t active_edge_count;
};

struct EdgeGpu {
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
};

struct LoweredScene {
  std::vector<ContextGpu> contexts;
  std::vector<std::uint32_t> well_contexts;
  std::vector<std::uint64_t> well_offsets;
  std::vector<std::uint32_t> active_contexts;
  std::vector<CellGpu> cells;
  std::vector<EdgeGpu> edges;
  std::uint64_t well_edge_count = 0;
  std::uint64_t active_edge_count = 0;
};

std::uint64_t subtree_context_count(
    const LoadedScene &scene, std::uint64_t cell_id,
    std::vector<std::optional<std::uint64_t>> *memo,
    std::uint64_t maximum) {
  if ((*memo)[cell_id]) {
    return *(*memo)[cell_id];
  }
  std::uint64_t total = 1;
  const CellRecord &cell = scene.cells[cell_id];
  for (std::uint64_t i = 0; i < cell.instance_count; ++i) {
    const InstanceRecord &instance = scene.instances[cell.instance_begin + i];
    const std::uint64_t child =
        subtree_context_count(scene, instance.child_cell_id, memo, maximum);
    std::uint64_t contribution = 0;
    if (!checked_mul_u64(instance.occurrence_count, child, &contribution) ||
        !checked_add_u64(total, contribution, &total) || total > maximum) {
      throw SceneError("expanded context count exceeds configured capacity");
    }
  }
  (*memo)[cell_id] = total;
  return total;
}

LoweredScene lower_hierarchy(const LoadedScene &scene,
                             std::uint64_t max_contexts) {
  LoweredScene lowered;
  std::vector<std::optional<std::uint64_t>> memo(scene.header.cell_count);
  const std::uint64_t expected_contexts =
      subtree_context_count(scene, scene.header.root_cell, &memo,
                            max_contexts);
  if (expected_contexts > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("context count exceeds the device context-ID domain");
  }
  lowered.contexts.reserve(expected_contexts);
  lowered.contexts.push_back(
      {static_cast<std::uint32_t>(scene.header.root_cell), 0, 0, 0});
  for (std::size_t context_id = 0;
       context_id < lowered.contexts.size(); ++context_id) {
    const ContextGpu parent = lowered.contexts[context_id];
    const CellRecord &cell = scene.cells[parent.cell];
    for (std::uint64_t index = 0; index < cell.instance_count; ++index) {
      const InstanceRecord &instance =
          scene.instances[cell.instance_begin + index];
      const std::uint32_t transform =
          compose_transform(parent.transform, instance.transform_code);
      for (std::uint32_t column = 0; column < instance.columns; ++column) {
        for (std::uint32_t row = 0; row < instance.rows; ++row) {
          const __int128 local_x =
              static_cast<__int128>(instance.dx) +
              static_cast<__int128>(column) * instance.ax +
              static_cast<__int128>(row) * instance.bx;
          const __int128 local_y =
              static_cast<__int128>(instance.dy) +
              static_cast<__int128>(column) * instance.ay +
              static_cast<__int128>(row) * instance.by;
          const auto shifted =
              transform_128(parent.transform, local_x, local_y);
          const std::int64_t tx =
              narrow_i64(shifted.first + parent.tx, "context x translation");
          const std::int64_t ty =
              narrow_i64(shifted.second + parent.ty, "context y translation");
          lowered.contexts.push_back(
              {static_cast<std::uint32_t>(instance.child_cell_id),
               transform, tx, ty});
        }
      }
    }
  }
  if (lowered.contexts.size() != expected_contexts) {
    throw SceneError("hierarchy context count disagrees with checked DP");
  }

  lowered.cells.resize(scene.header.cell_count);
  lowered.edges.resize(scene.header.edge_count);
  for (std::uint64_t id = 0; id < scene.header.edge_count; ++id) {
    const EdgeRecord &edge = scene.edges[id];
    lowered.edges[id] = {edge.x1, edge.y1, edge.x2, edge.y2};
  }
  for (std::uint64_t cell_id = 0; cell_id < scene.header.cell_count;
       ++cell_id) {
    const CellRecord &cell = scene.cells[cell_id];
    CellGpu compact{};
    bool saw_active = false;
    for (std::uint64_t offset = 0; offset < cell.edge_count; ++offset) {
      const std::uint64_t edge_id = cell.edge_begin + offset;
      const std::uint32_t layer = scene.edges[edge_id].layer_code;
      if (layer == kWellLayer) {
        if (saw_active) {
          throw SceneError("cell edge layers are not canonically grouped");
        }
        if (!compact.well_edge_count) {
          compact.well_edge_begin = edge_id;
        }
        if (compact.well_edge_count ==
            std::numeric_limits<std::uint32_t>::max()) {
          throw SceneError("per-cell WELL edge count exceeds uint32");
        }
        ++compact.well_edge_count;
      } else if (layer == kActiveLayer) {
        saw_active = true;
        if (!compact.active_edge_count) {
          compact.active_edge_begin = edge_id;
        }
        if (compact.active_edge_count ==
            std::numeric_limits<std::uint32_t>::max()) {
          throw SceneError("per-cell ACTIVE edge count exceeds uint32");
        }
        ++compact.active_edge_count;
      } else {
        throw SceneError("unsupported logical edge layer");
      }
    }
    lowered.cells[cell_id] = compact;
  }

  for (std::uint32_t id = 0; id < lowered.contexts.size(); ++id) {
    const CellGpu &cell = lowered.cells[lowered.contexts[id].cell];
    if (cell.well_edge_count) {
      lowered.well_contexts.push_back(id);
      lowered.well_offsets.push_back(lowered.well_edge_count);
      if (!checked_add_u64(lowered.well_edge_count, cell.well_edge_count,
                           &lowered.well_edge_count)) {
        throw SceneError("flat WELL edge count overflow");
      }
    }
    if (cell.active_edge_count) {
      lowered.active_contexts.push_back(id);
      if (!checked_add_u64(lowered.active_edge_count,
                           cell.active_edge_count,
                           &lowered.active_edge_count)) {
        throw SceneError("flat ACTIVE edge count overflow");
      }
    }
  }
  if (lowered.well_edge_count > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("flat WELL edge count exceeds uint32 index domain");
  }
  return lowered;
}

__device__ bool negate_checked(std::int64_t value, std::int64_t *result) {
  if (value == INT64_MIN) {
    return false;
  }
  *result = -value;
  return true;
}

__device__ bool add_checked(std::int64_t a, std::int64_t b,
                            std::int64_t *result) {
  if ((b > 0 && a > INT64_MAX - b) ||
      (b < 0 && a < INT64_MIN - b)) {
    return false;
  }
  *result = a + b;
  return true;
}

__device__ bool transform_point_checked(const ContextGpu &context,
                                        std::int64_t x, std::int64_t y,
                                        std::int64_t *output_x,
                                        std::int64_t *output_y) {
  std::int64_t tx = 0;
  std::int64_t ty = 0;
  switch (context.transform) {
    case 0:
      tx = x;
      ty = y;
      break;
    case 1:
      if (!negate_checked(y, &tx)) return false;
      ty = x;
      break;
    case 2:
      if (!negate_checked(x, &tx) || !negate_checked(y, &ty)) return false;
      break;
    case 3:
      tx = y;
      if (!negate_checked(x, &ty)) return false;
      break;
    case 4:
      tx = x;
      if (!negate_checked(y, &ty)) return false;
      break;
    case 5:
      tx = y;
      ty = x;
      break;
    case 6:
      if (!negate_checked(x, &tx)) return false;
      ty = y;
      break;
    case 7:
      if (!negate_checked(y, &tx) || !negate_checked(x, &ty)) return false;
      break;
    default:
      return false;
  }
  return add_checked(tx, context.tx, output_x) &&
         add_checked(ty, context.ty, output_y);
}

__device__ bool transform_edge_checked(const ContextGpu &context,
                                       const EdgeGpu &source,
                                       DirectedEdge *destination) {
  DirectedEdge transformed{};
  if (!transform_point_checked(context, source.x1, source.y1,
                               &transformed.x1, &transformed.y1) ||
      !transform_point_checked(context, source.x2, source.y2,
                               &transformed.x2, &transformed.y2)) {
    return false;
  }
  // KLayout normalizes a reflected polygon hull back to clockwise.  The
  // affine transform alone reverses its orientation, so reverse every
  // directed edge under composed mirror codes 4..7.  Edge order is irrelevant
  // to this streamed relation, but the directed half-plane is not.
  if (context.transform >= 4) {
    destination->x1 = transformed.x2;
    destination->y1 = transformed.y2;
    destination->x2 = transformed.x1;
    destination->y2 = transformed.y1;
  } else {
    *destination = transformed;
  }
  return true;
}

__global__ void transform_semantics_gate(std::uint32_t *status) {
  const std::uint32_t code = threadIdx.x;
  if (blockIdx.x || code >= 8) {
    return;
  }
  // Clockwise rectangle.  The sum below remains negative for every KLayout
  // Trans code only if reflected directed edges are normalized correctly.
  const EdgeGpu source[4] = {
      {0, 0, 0, 20}, {0, 20, 10, 20},
      {10, 20, 10, 0}, {10, 0, 0, 0}};
  const ContextGpu context = {0, code, 13, -7};
  long long twice_area = 0;
  for (int edge_id = 0; edge_id < 4; ++edge_id) {
    DirectedEdge edge{};
    if (!transform_edge_checked(context, source[edge_id], &edge)) {
      atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
      return;
    }
    twice_area += edge.x1 * edge.y2 - edge.x2 * edge.y1;
  }
  if (twice_area != -400) {
    atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
  }
}

__device__ std::int64_t floor_div(std::int64_t value,
                                  std::int64_t divisor) {
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) {
    --quotient;
  }
  return quotient;
}

struct Grid {
  std::int64_t base_x;
  std::int64_t base_y;
  std::uint32_t width;
  std::uint32_t height;
};

__device__ bool edge_grid_span(const DirectedEdge &edge, const Grid &grid,
                               std::int64_t expansion,
                               std::int64_t *x0, std::int64_t *y0,
                               std::int64_t *x1, std::int64_t *y1) {
  std::int64_t low_x = min(edge.x1, edge.x2);
  std::int64_t high_x = max(edge.x1, edge.x2);
  std::int64_t low_y = min(edge.y1, edge.y2);
  std::int64_t high_y = max(edge.y1, edge.y2);
  if (expansion) {
    if (!add_checked(low_x, -expansion, &low_x) ||
        !add_checked(high_x, expansion, &high_x) ||
        !add_checked(low_y, -expansion, &low_y) ||
        !add_checked(high_y, expansion, &high_y)) {
      return false;
    }
  }
  *x0 = floor_div(low_x, kGridCell);
  *x1 = floor_div(high_x, kGridCell);
  *y0 = floor_div(low_y, kGridCell);
  *y1 = floor_div(high_y, kGridCell);
  return true;
}

__device__ bool clip_span(const Grid &grid, std::int64_t *x0,
                          std::int64_t *y0, std::int64_t *x1,
                          std::int64_t *y1) {
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  if (*x1 < grid.base_x || *x0 > maximum_x || *y1 < grid.base_y ||
      *y0 > maximum_y) {
    return false;
  }
  *x0 = max(*x0, grid.base_x);
  *x1 = min(*x1, maximum_x);
  *y0 = max(*y0, grid.base_y);
  *y1 = min(*y1, maximum_y);
  return true;
}

__device__ bool span_inside_grid(const Grid &grid, std::int64_t x0,
                                 std::int64_t y0, std::int64_t x1,
                                 std::int64_t y1) {
  const std::int64_t maximum_x =
      grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
  const std::int64_t maximum_y =
      grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
  return x0 >= grid.base_x && x1 <= maximum_x &&
         y0 >= grid.base_y && y1 <= maximum_y;
}

__device__ std::uint64_t grid_index(const Grid &grid, std::int64_t x,
                                    std::int64_t y) {
  return static_cast<std::uint64_t>(y - grid.base_y) * grid.width +
         static_cast<std::uint64_t>(x - grid.base_x);
}

__global__ void expand_well_kernel(
    const ContextGpu *contexts, const std::uint32_t *well_contexts,
    const std::uint64_t *well_offsets, const CellGpu *cells,
    const EdgeGpu *templates, std::uint32_t context_count,
    DirectedEdge *well_edges, std::uint32_t *status) {
  const std::uint32_t list_index = blockIdx.x;
  if (list_index >= context_count) {
    return;
  }
  const ContextGpu context = contexts[well_contexts[list_index]];
  const CellGpu cell = cells[context.cell];
  for (std::uint32_t local = threadIdx.x; local < cell.well_edge_count;
       local += blockDim.x) {
    DirectedEdge edge{};
    if (!transform_edge_checked(
            context, templates[cell.well_edge_begin + local], &edge)) {
      atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
      continue;
    }
    well_edges[well_offsets[list_index] + local] = edge;
  }
}

__global__ void count_grid_kernel(const DirectedEdge *well_edges,
                                  std::uint32_t well_count, Grid grid,
                                  std::uint32_t *counts,
                                  unsigned long long *total,
                                  std::uint32_t *status) {
  for (std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
       id < well_count; id += blockDim.x * gridDim.x) {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!edge_grid_span(well_edges[id], grid, 0, &x0, &y0, &x1, &y1) ||
        !span_inside_grid(grid, x0, y0, x1, y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t index = grid_index(grid, x, y);
        const std::uint32_t previous = atomicAdd(counts + index, 1u);
        if (previous == UINT32_MAX) {
          atomicOr(status,
                   static_cast<std::uint32_t>(kGridCounterOverflow));
        }
        atomicAdd(total, 1ull);
      }
    }
  }
}

__global__ void fill_grid_kernel(const DirectedEdge *well_edges,
                                 std::uint32_t well_count, Grid grid,
                                 std::uint32_t *cursors,
                                 std::uint32_t *members,
                                 std::uint64_t member_capacity,
                                 std::uint32_t *status) {
  for (std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
       id < well_count; id += blockDim.x * gridDim.x) {
    std::int64_t x0 = 0;
    std::int64_t y0 = 0;
    std::int64_t x1 = 0;
    std::int64_t y1 = 0;
    if (!edge_grid_span(well_edges[id], grid, 0, &x0, &y0, &x1, &y1) ||
        !span_inside_grid(grid, x0, y0, x1, y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
      continue;
    }
    for (std::int64_t y = y0; y <= y1; ++y) {
      for (std::int64_t x = x0; x <= x1; ++x) {
        const std::uint64_t index = grid_index(grid, x, y);
        const std::uint32_t position = atomicAdd(cursors + index, 1u);
        if (position >= member_capacity) {
          atomicOr(status,
                   static_cast<std::uint32_t>(kGridCapacityExceeded));
        } else {
          members[position] = id;
        }
      }
    }
  }
}

__global__ void validate_grid_kernel(const std::uint32_t *counts,
                                     const std::uint32_t *offsets,
                                     const std::uint32_t *cursors,
                                     std::uint64_t cell_count,
                                     std::uint32_t *status) {
  for (std::uint64_t cell =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cell_count;
       cell += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t expected =
        static_cast<std::uint64_t>(offsets[cell]) + counts[cell];
    if (expected > UINT32_MAX ||
        cursors[cell] != expected) {
      atomicOr(status, static_cast<std::uint32_t>(kGridCounterOverflow));
    }
  }
}

struct DeviceCounters {
  unsigned long long candidate_pairs;
  unsigned long long violations;
  unsigned long long uncertain;
};

struct HitSample {
  std::uint32_t context_id;
  std::uint32_t active_edge_local;
  std::uint32_t well_edge;
  std::uint32_t verdict;
};

__global__ void query_active_kernel(
    const ContextGpu *contexts, const std::uint32_t *active_contexts,
    std::uint32_t active_context_count, const CellGpu *cells,
    const EdgeGpu *templates, const DirectedEdge *well_edges,
    std::uint32_t well_edge_count, Grid grid,
    const std::uint32_t *counts, const std::uint32_t *offsets,
    const std::uint32_t *members, std::int64_t distance,
    DeviceCounters *counters, HitSample *samples,
    std::uint32_t *sample_count, std::uint32_t *status) {
  const std::uint32_t active_list_id = blockIdx.x;
  if (active_list_id >= active_context_count) {
    return;
  }
  const std::uint32_t context_id = active_contexts[active_list_id];
  const ContextGpu context = contexts[context_id];
  const CellGpu cell = cells[context.cell];
  unsigned long long local_candidates = 0;
  unsigned long long local_violations = 0;
  unsigned long long local_uncertain = 0;
  for (std::uint32_t local = threadIdx.x; local < cell.active_edge_count;
       local += blockDim.x) {
    DirectedEdge active{};
    if (!transform_edge_checked(
            context, templates[cell.active_edge_begin + local], &active)) {
      atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
      continue;
    }
    std::int64_t active_x0 = 0;
    std::int64_t active_y0 = 0;
    std::int64_t active_x1 = 0;
    std::int64_t active_y1 = 0;
    if (!edge_grid_span(active, grid, distance, &active_x0, &active_y0,
                        &active_x1, &active_y1)) {
      atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
      continue;
    }
    if (!clip_span(grid, &active_x0, &active_y0, &active_x1, &active_y1)) {
      continue;
    }
    for (std::int64_t y = active_y0; y <= active_y1; ++y) {
      for (std::int64_t x = active_x0; x <= active_x1; ++x) {
        const std::uint64_t cell_index = grid_index(grid, x, y);
        const std::uint32_t begin = offsets[cell_index];
        const std::uint32_t end = begin + counts[cell_index];
        for (std::uint32_t position = begin; position < end; ++position) {
          const std::uint32_t well_id = members[position];
          if (well_id >= well_edge_count) {
            atomicOr(status,
                     static_cast<std::uint32_t>(kInvalidDeviceRecord));
            continue;
          }
          const DirectedEdge well = well_edges[well_id];
          std::int64_t well_x0 = 0;
          std::int64_t well_y0 = 0;
          std::int64_t well_x1 = 0;
          std::int64_t well_y1 = 0;
          if (!edge_grid_span(well, grid, 0, &well_x0, &well_y0, &well_x1,
                              &well_y1)) {
            atomicOr(status,
                     static_cast<std::uint32_t>(kInvalidDeviceRecord));
            continue;
          }
          // A pair can share several cells.  Its componentwise-lowest cell
          // in the span intersection is a deterministic unique owner.
          if (x != max(active_x0, well_x0) ||
              y != max(active_y0, well_y0)) {
            continue;
          }
          // The uniform grid is deliberately coarse.  Cull pairs which only
          // share a grid cell but whose exact axis-aligned spans do not meet
          // the distance-expanded ACTIVE span.  This makes candidate counters
          // independent of grid-cell aliasing while retaining all Euclidean
          // pairs with distance < d.
          std::int64_t expanded_left = min(active.x1, active.x2);
          std::int64_t expanded_right = max(active.x1, active.x2);
          std::int64_t expanded_bottom = min(active.y1, active.y2);
          std::int64_t expanded_top = max(active.y1, active.y2);
          if (!add_checked(expanded_left, -distance, &expanded_left) ||
              !add_checked(expanded_right, distance, &expanded_right) ||
              !add_checked(expanded_bottom, -distance, &expanded_bottom) ||
              !add_checked(expanded_top, distance, &expanded_top)) {
            atomicOr(status,
                     static_cast<std::uint32_t>(kTransformOverflow));
            continue;
          }
          const std::int64_t well_left = min(well.x1, well.x2);
          const std::int64_t well_right = max(well.x1, well.x2);
          const std::int64_t well_bottom = min(well.y1, well.y2);
          const std::int64_t well_top = max(well.y1, well.y2);
          if (well_right < expanded_left || well_left > expanded_right ||
              well_top < expanded_bottom || well_bottom > expanded_top) {
            continue;
          }
          ++local_candidates;
          const Verdict verdict =
              klayout_cuda::active3::classify_pair_bounded(
                  EdgePair{well, active}, distance);
          if (verdict == Verdict::kViolation) {
            ++local_violations;
          } else if (verdict == Verdict::kUncertain) {
            ++local_uncertain;
          } else if (verdict == Verdict::kNoViolation) {
            continue;
          } else {
            atomicOr(status,
                     static_cast<std::uint32_t>(kInvalidDeviceRecord));
            continue;
          }
          const std::uint32_t sample = atomicAdd(sample_count, 1u);
          if (sample < kSampleCapacity) {
            samples[sample] =
                {context_id, local, well_id,
                 static_cast<std::uint32_t>(verdict)};
          }
        }
      }
    }
  }
  if (local_candidates) {
    atomicAdd(&counters->candidate_pairs, local_candidates);
  }
  if (local_violations) {
    atomicAdd(&counters->violations, local_violations);
  }
  if (local_uncertain) {
    atomicAdd(&counters->uncertain, local_uncertain);
  }
}

void cuda_require(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw SceneError(std::string(operation) + ": " +
                     cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() {
    if (pointer_) {
      cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
      : pointer_(other.pointer_), count_(other.count_) {
    other.pointer_ = nullptr;
    other.count_ = 0;
  }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (pointer_) {
        cudaFree(pointer_);
      }
      pointer_ = other.pointer_;
      count_ = other.count_;
      other.pointer_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }
  void allocate(std::size_t count) {
    if (pointer_ || !count) {
      if (!count) {
        count_ = 0;
        return;
      }
      throw SceneError("invalid device-buffer allocation state");
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw SceneError("device-buffer byte-size overflow");
    }
    cuda_require(cudaMalloc(&pointer_, count * sizeof(T)), "cudaMalloc");
    count_ = count;
  }
  T *get() { return pointer_; }
  const T *get() const { return pointer_; }
  std::size_t size() const { return count_; }
  cudaError_t release() {
    if (!pointer_) {
      return cudaSuccess;
    }
    T *pointer = pointer_;
    pointer_ = nullptr;
    count_ = 0;
    return cudaFree(pointer);
  }

 private:
  T *pointer_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T>
void upload(DeviceBuffer<T> *destination, const std::vector<T> &source) {
  if (!source.empty()) {
    cuda_require(cudaMemcpy(destination->get(), source.data(),
                            source.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D");
  }
}

std::int64_t host_floor_div(std::int64_t value, std::int64_t divisor) {
  std::int64_t quotient = value / divisor;
  if (value % divisor < 0) {
    --quotient;
  }
  return quotient;
}

struct Options {
  std::string path;
  std::string expected_scene_sha256;
  bool verify_bruteforce = false;
  std::uint64_t max_contexts = kDefaultMaxContexts;
  std::uint64_t max_grid_cells = kDefaultMaxGridCells;
  std::uint64_t max_memberships = kDefaultMaxMemberships;
  std::uint64_t max_pair_work = kDefaultMaxPairWork;
  std::uint64_t max_bruteforce_pairs = UINT64_C(10000000);
};

std::uint64_t parse_u64(const std::string &text, const char *name) {
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

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto take = [&](const char *prefix, std::uint64_t *destination) {
      const std::string marker = std::string(prefix) + "=";
      if (argument.rfind(marker, 0) == 0) {
        *destination = parse_u64(argument.substr(marker.size()), prefix);
        return true;
      }
      return false;
    };
    if (take("--max-contexts", &options.max_contexts) ||
        take("--max-grid-cells", &options.max_grid_cells) ||
        take("--max-memberships", &options.max_memberships) ||
        take("--max-pair-work", &options.max_pair_work) ||
        take("--max-bruteforce-pairs", &options.max_bruteforce_pairs)) {
      continue;
    }
    const std::string sha_prefix = "--expect-scene-sha256=";
    if (argument.rfind(sha_prefix, 0) == 0) {
      options.expected_scene_sha256 = argument.substr(sha_prefix.size());
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
    if (argument == "--verify-bruteforce") {
      options.verify_bruteforce = true;
      continue;
    }
    if (!argument.empty() && argument[0] == '-') {
      throw SceneError("unknown option: " + argument);
    }
    if (!options.path.empty()) {
      throw SceneError("exactly one packed scene path is required");
    }
    options.path = argument;
  }
  if (options.path.empty()) {
    throw SceneError(
        "usage: active3_scene_island --expect-scene-sha256=HEX "
        "[capacity options] SCENE.kact");
  }
  if (options.expected_scene_sha256.empty()) {
    throw SceneError(
        "an external --expect-scene-sha256 fingerprint is mandatory");
  }
  return options;
}

struct Timings {
  double load_validate_ms = 0;
  double cpu_lower_ms = 0;
  double bruteforce_verify_ms = 0;
  double cuda_init_ms = 0;
  double alloc_upload_ms = 0;
  double well_expand_ms = 0;
  double grid_count_ms = 0;
  double grid_build_ms = 0;
  double active_query_ms = 0;
  double d2h_ms = 0;
  double cuda_cleanup_ms = 0;
  double total_ms = 0;
};

DirectedEdge transform_edge_host(const ContextGpu &context,
                                 const EdgeGpu &source) {
  const auto first = transform_128(context.transform, source.x1, source.y1);
  const auto second = transform_128(context.transform, source.x2, source.y2);
  DirectedEdge edge = {
      narrow_i64(first.first + context.tx, "host edge x1"),
      narrow_i64(first.second + context.ty, "host edge y1"),
      narrow_i64(second.first + context.tx, "host edge x2"),
      narrow_i64(second.second + context.ty, "host edge y2")};
  if (context.transform >= 4) {
    std::swap(edge.x1, edge.x2);
    std::swap(edge.y1, edge.y2);
  }
  return edge;
}

struct BruteForceResult {
  std::uint64_t candidates = 0;
  std::uint64_t violations = 0;
  std::uint64_t uncertain = 0;
};

BruteForceResult verify_bruteforce(const LoweredScene &lowered,
                                   std::int64_t distance,
                                   std::uint64_t maximum_pairs) {
  std::uint64_t pair_count = 0;
  if (!checked_mul_u64(lowered.well_edge_count,
                       lowered.active_edge_count, &pair_count) ||
      pair_count > maximum_pairs) {
    throw SceneError(
        "CPU brute-force pair count exceeds configured capacity");
  }
  std::vector<DirectedEdge> wells;
  std::vector<DirectedEdge> actives;
  wells.reserve(lowered.well_edge_count);
  actives.reserve(lowered.active_edge_count);
  for (std::uint32_t context_id : lowered.well_contexts) {
    const ContextGpu &context = lowered.contexts[context_id];
    const CellGpu &cell = lowered.cells[context.cell];
    for (std::uint32_t local = 0; local < cell.well_edge_count; ++local) {
      wells.push_back(transform_edge_host(
          context, lowered.edges[cell.well_edge_begin + local]));
    }
  }
  for (std::uint32_t context_id : lowered.active_contexts) {
    const ContextGpu &context = lowered.contexts[context_id];
    const CellGpu &cell = lowered.cells[context.cell];
    for (std::uint32_t local = 0; local < cell.active_edge_count; ++local) {
      actives.push_back(transform_edge_host(
          context, lowered.edges[cell.active_edge_begin + local]));
    }
  }
  if (wells.size() != lowered.well_edge_count ||
      actives.size() != lowered.active_edge_count) {
    throw SceneError("CPU brute-force flattened count mismatch");
  }
  BruteForceResult result;
  for (const DirectedEdge &active : actives) {
    const __int128 active_left =
        static_cast<__int128>(std::min(active.x1, active.x2)) - distance;
    const __int128 active_right =
        static_cast<__int128>(std::max(active.x1, active.x2)) + distance;
    const __int128 active_bottom =
        static_cast<__int128>(std::min(active.y1, active.y2)) - distance;
    const __int128 active_top =
        static_cast<__int128>(std::max(active.y1, active.y2)) + distance;
    for (const DirectedEdge &well : wells) {
      const std::int64_t well_left = std::min(well.x1, well.x2);
      const std::int64_t well_right = std::max(well.x1, well.x2);
      const std::int64_t well_bottom = std::min(well.y1, well.y2);
      const std::int64_t well_top = std::max(well.y1, well.y2);
      if (well_right < active_left || well_left > active_right ||
          well_top < active_bottom || well_bottom > active_top) {
        continue;
      }
      ++result.candidates;
      const Verdict verdict =
          klayout_cuda::active3::classify_pair_bounded(
              EdgePair{well, active}, distance);
      if (verdict == Verdict::kViolation) {
        ++result.violations;
      } else if (verdict == Verdict::kUncertain) {
        ++result.uncertain;
      }
    }
  }
  return result;
}

int run(const Options &options, Clock::time_point total_begin) {
  Timings timing;
  auto begin = Clock::now();
  LoadedScene scene = load_and_validate(options.path);
  auto end = Clock::now();
  timing.load_validate_ms = milliseconds(begin, end);
  if (!options.expected_scene_sha256.empty() &&
      options.expected_scene_sha256 !=
          hex_digest(scene.header.scene_sha256, 32)) {
    throw SceneError("scene SHA-256 does not match explicit expectation");
  }

  begin = Clock::now();
  LoweredScene lowered = lower_hierarchy(scene, options.max_contexts);
  end = Clock::now();
  timing.cpu_lower_ms = milliseconds(begin, end);

  std::optional<BruteForceResult> brute_force;
  if (options.verify_bruteforce) {
    begin = Clock::now();
    brute_force = verify_bruteforce(
        lowered, scene.distance, options.max_bruteforce_pairs);
    end = Clock::now();
    timing.bruteforce_verify_ms = milliseconds(begin, end);
  }

  if (!lowered.well_edge_count || !lowered.active_edge_count) {
    throw SceneError("qualified scene must contain both WELL and ACTIVE edges");
  }
  std::uint64_t maximum_pair_work = 0;
  if (!checked_mul_u64(lowered.well_edge_count,
                       lowered.active_edge_count, &maximum_pair_work) ||
      maximum_pair_work > options.max_pair_work) {
    throw SceneError("WELL/ACTIVE pair-work ceiling exceeds capacity");
  }
  const Box root_well =
      box_from(scene.cells[scene.header.root_cell].subtree_bbox[kWellLayer]);
  const std::int64_t base_x = host_floor_div(root_well.left, kGridCell);
  const std::int64_t base_y = host_floor_div(root_well.bottom, kGridCell);
  const std::int64_t maximum_x = host_floor_div(root_well.right, kGridCell);
  const std::int64_t maximum_y = host_floor_div(root_well.top, kGridCell);
  const std::uint64_t grid_width =
      static_cast<std::uint64_t>(maximum_x - base_x) + 1;
  const std::uint64_t grid_height =
      static_cast<std::uint64_t>(maximum_y - base_y) + 1;
  std::uint64_t grid_cells = 0;
  if (!checked_mul_u64(grid_width, grid_height, &grid_cells) ||
      grid_width > std::numeric_limits<std::uint32_t>::max() ||
      grid_height > std::numeric_limits<std::uint32_t>::max() ||
      grid_cells > options.max_grid_cells ||
      grid_cells > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("uniform grid exceeds configured/index capacity");
  }
  const Grid grid = {base_x, base_y,
                     static_cast<std::uint32_t>(grid_width),
                     static_cast<std::uint32_t>(grid_height)};

  begin = Clock::now();
  cuda_require(cudaFree(nullptr), "CUDA context initialization");
  cuda_require(cudaDeviceSynchronize(), "CUDA initialization synchronize");
  end = Clock::now();
  timing.cuda_init_ms = milliseconds(begin, end);

  begin = Clock::now();
  DeviceBuffer<ContextGpu> d_contexts(lowered.contexts.size());
  DeviceBuffer<std::uint32_t> d_well_contexts(
      lowered.well_contexts.size());
  DeviceBuffer<std::uint64_t> d_well_offsets(lowered.well_offsets.size());
  DeviceBuffer<std::uint32_t> d_active_contexts(
      lowered.active_contexts.size());
  DeviceBuffer<CellGpu> d_cells(lowered.cells.size());
  DeviceBuffer<EdgeGpu> d_edges(lowered.edges.size());
  DeviceBuffer<DirectedEdge> d_well_edges(lowered.well_edge_count);
  DeviceBuffer<std::uint32_t> d_counts(grid_cells);
  DeviceBuffer<std::uint32_t> d_offsets(grid_cells + 1);
  DeviceBuffer<std::uint32_t> d_cursors(grid_cells);
  DeviceBuffer<unsigned long long> d_membership_total(1);
  DeviceBuffer<std::uint32_t> d_status(1);
  DeviceBuffer<DeviceCounters> d_counters(1);
  DeviceBuffer<HitSample> d_samples(kSampleCapacity);
  DeviceBuffer<std::uint32_t> d_sample_count(1);
  upload(&d_contexts, lowered.contexts);
  upload(&d_well_contexts, lowered.well_contexts);
  upload(&d_well_offsets, lowered.well_offsets);
  upload(&d_active_contexts, lowered.active_contexts);
  upload(&d_cells, lowered.cells);
  upload(&d_edges, lowered.edges);
  cuda_require(cudaMemset(d_counts.get(), 0,
                          grid_cells * sizeof(std::uint32_t)),
               "cudaMemset grid counts");
  cuda_require(cudaMemset(d_membership_total.get(), 0,
                          sizeof(unsigned long long)),
               "cudaMemset membership total");
  cuda_require(cudaMemset(d_status.get(), 0, sizeof(std::uint32_t)),
               "cudaMemset status");
  cuda_require(cudaMemset(d_counters.get(), 0, sizeof(DeviceCounters)),
               "cudaMemset counters");
  cuda_require(cudaMemset(d_sample_count.get(), 0, sizeof(std::uint32_t)),
               "cudaMemset sample count");
  transform_semantics_gate<<<1, 8>>>(d_status.get());
  cuda_require(cudaGetLastError(), "transform_semantics_gate launch");
  cuda_require(cudaDeviceSynchronize(), "upload synchronize");
  end = Clock::now();
  timing.alloc_upload_ms = milliseconds(begin, end);

  begin = Clock::now();
  expand_well_kernel<<<static_cast<unsigned int>(
                           lowered.well_contexts.size()),
                       128>>>(
      d_contexts.get(), d_well_contexts.get(), d_well_offsets.get(),
      d_cells.get(), d_edges.get(),
      static_cast<std::uint32_t>(lowered.well_contexts.size()),
      d_well_edges.get(), d_status.get());
  cuda_require(cudaGetLastError(), "expand_well_kernel launch");
  cuda_require(cudaDeviceSynchronize(), "expand_well_kernel synchronize");
  end = Clock::now();
  timing.well_expand_ms = milliseconds(begin, end);

  std::uint32_t host_status = 0;
  cuda_require(cudaMemcpy(&host_status, d_status.get(), sizeof(host_status),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy WELL status D2H");
  if (host_status) {
    throw SceneError("device WELL expansion declined the scene, flags=" +
                     std::to_string(host_status));
  }

  begin = Clock::now();
  const unsigned int well_blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>(
          65535, (lowered.well_edge_count + 255) / 256));
  count_grid_kernel<<<well_blocks, 256>>>(
      d_well_edges.get(),
      static_cast<std::uint32_t>(lowered.well_edge_count), grid,
      d_counts.get(), d_membership_total.get(), d_status.get());
  cuda_require(cudaGetLastError(), "count_grid_kernel launch");
  cuda_require(cudaDeviceSynchronize(), "count_grid_kernel synchronize");
  unsigned long long membership_total = 0;
  cuda_require(cudaMemcpy(&membership_total, d_membership_total.get(),
                          sizeof(membership_total), cudaMemcpyDeviceToHost),
               "cudaMemcpy membership total D2H");
  cuda_require(cudaMemcpy(&host_status, d_status.get(), sizeof(host_status),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy grid count status D2H");
  end = Clock::now();
  timing.grid_count_ms = milliseconds(begin, end);
  if (host_status) {
    throw SceneError("device grid count declined the scene, flags=" +
                     std::to_string(host_status));
  }
  if (!membership_total || membership_total > options.max_memberships ||
      membership_total > std::numeric_limits<std::uint32_t>::max()) {
    throw SceneError("grid memberships exceed configured/uint32 capacity");
  }

  begin = Clock::now();
  thrust::device_ptr<std::uint32_t> count_begin(d_counts.get());
  thrust::device_ptr<std::uint32_t> offset_begin(d_offsets.get());
  thrust::exclusive_scan(count_begin, count_begin + grid_cells,
                         offset_begin);
  const std::uint32_t total_u32 =
      static_cast<std::uint32_t>(membership_total);
  cuda_require(cudaMemcpy(d_offsets.get() + grid_cells, &total_u32,
                          sizeof(total_u32), cudaMemcpyHostToDevice),
               "cudaMemcpy terminal grid offset H2D");
  cuda_require(cudaMemcpy(d_cursors.get(), d_offsets.get(),
                          grid_cells * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToDevice),
               "cudaMemcpy offsets to cursors D2D");
  DeviceBuffer<std::uint32_t> d_members(membership_total);
  fill_grid_kernel<<<well_blocks, 256>>>(
      d_well_edges.get(),
      static_cast<std::uint32_t>(lowered.well_edge_count), grid,
      d_cursors.get(), d_members.get(), membership_total, d_status.get());
  cuda_require(cudaGetLastError(), "fill_grid_kernel launch");
  const unsigned int grid_blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>(65535, (grid_cells + 255) / 256));
  validate_grid_kernel<<<grid_blocks, 256>>>(
      d_counts.get(), d_offsets.get(), d_cursors.get(), grid_cells,
      d_status.get());
  cuda_require(cudaGetLastError(), "validate_grid_kernel launch");
  cuda_require(cudaDeviceSynchronize(), "grid build synchronize");
  cuda_require(cudaMemcpy(&host_status, d_status.get(), sizeof(host_status),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy grid build status D2H");
  end = Clock::now();
  timing.grid_build_ms = milliseconds(begin, end);
  if (host_status) {
    throw SceneError("device grid build declined the scene, flags=" +
                     std::to_string(host_status));
  }

  begin = Clock::now();
  if (lowered.active_contexts.size() >
      std::numeric_limits<unsigned int>::max()) {
    throw SceneError("ACTIVE context launch exceeds CUDA grid-x domain");
  }
  query_active_kernel<<<static_cast<unsigned int>(
                            lowered.active_contexts.size()),
                        128>>>(
      d_contexts.get(), d_active_contexts.get(),
      static_cast<std::uint32_t>(lowered.active_contexts.size()),
      d_cells.get(), d_edges.get(), d_well_edges.get(),
      static_cast<std::uint32_t>(lowered.well_edge_count), grid,
      d_counts.get(), d_offsets.get(), d_members.get(), scene.distance,
      d_counters.get(), d_samples.get(), d_sample_count.get(),
      d_status.get());
  cuda_require(cudaGetLastError(), "query_active_kernel launch");
  cuda_require(cudaDeviceSynchronize(), "query_active_kernel synchronize");
  end = Clock::now();
  timing.active_query_ms = milliseconds(begin, end);

  begin = Clock::now();
  DeviceCounters counters{};
  std::uint32_t sample_count = 0;
  std::array<HitSample, kSampleCapacity> samples{};
  cuda_require(cudaMemcpy(&counters, d_counters.get(), sizeof(counters),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy counters D2H");
  cuda_require(cudaMemcpy(&sample_count, d_sample_count.get(),
                          sizeof(sample_count), cudaMemcpyDeviceToHost),
               "cudaMemcpy sample count D2H");
  cuda_require(cudaMemcpy(&host_status, d_status.get(), sizeof(host_status),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy final status D2H");
  if (sample_count) {
    cuda_require(cudaMemcpy(samples.data(), d_samples.get(),
                            sizeof(samples), cudaMemcpyDeviceToHost),
                 "cudaMemcpy samples D2H");
  }
  end = Clock::now();
  timing.d2h_ms = milliseconds(begin, end);
  if (brute_force &&
      (brute_force->candidates != counters.candidate_pairs ||
       brute_force->violations != counters.violations ||
       brute_force->uncertain != counters.uncertain)) {
    std::ostringstream mismatch;
    mismatch << "GPU spatial result disagrees with CPU brute force: GPU="
             << counters.candidate_pairs << "/" << counters.violations
             << "/" << counters.uncertain << " CPU="
             << brute_force->candidates << "/" << brute_force->violations
             << "/" << brute_force->uncertain;
    throw SceneError(mismatch.str());
  }
  begin = Clock::now();
  cudaError_t cleanup_status = cudaSuccess;
  const auto release = [&](auto *buffer) {
    const cudaError_t status = buffer->release();
    if (cleanup_status == cudaSuccess && status != cudaSuccess) {
      cleanup_status = status;
    }
  };
  release(&d_sample_count);
  release(&d_samples);
  release(&d_counters);
  release(&d_status);
  release(&d_membership_total);
  release(&d_members);
  release(&d_cursors);
  release(&d_offsets);
  release(&d_counts);
  release(&d_well_edges);
  release(&d_edges);
  release(&d_cells);
  release(&d_active_contexts);
  release(&d_well_offsets);
  release(&d_well_contexts);
  release(&d_contexts);
  if (cleanup_status != cudaSuccess) {
    throw SceneError(std::string("CUDA cleanup: ") +
                     cudaGetErrorString(cleanup_status));
  }
  end = Clock::now();
  timing.cuda_cleanup_ms = milliseconds(begin, end);
  timing.total_ms = milliseconds(total_begin, Clock::now());

  const char *verdict = "CLEAN";
  if (host_status || counters.uncertain) {
    verdict = "UNCERTAIN";
  } else if (counters.violations) {
    // The captured ACTIVE layer contains raw contours. KLayout unions those
    // intruders locally, which can erase internal raw edges. A raw hit is
    // therefore a mandatory CPU-fallback signal, not an exact final marker.
    // Only the zero-hit direction is consumed as a clean certificate.
    verdict = "RAW_HIT_FALLBACK";
  }
  std::cout << std::fixed << std::setprecision(3)
            << "ACTIVE3_GPU_ISLAND"
            << " verdict=" << verdict
            << " distance_dbu=" << scene.distance
            << " dbu_um=" << std::setprecision(7) << scene.header.dbu
            << std::setprecision(3)
            << " contexts=" << lowered.contexts.size()
            << " well_contexts=" << lowered.well_contexts.size()
            << " active_contexts=" << lowered.active_contexts.size()
            << " well_edges=" << lowered.well_edge_count
            << " active_edges=" << lowered.active_edge_count
            << " max_pair_work=" << maximum_pair_work
            << " grid=" << grid.width << "x" << grid.height
            << " grid_cells=" << grid_cells
            << " memberships=" << membership_total
            << " candidate_pairs=" << counters.candidate_pairs
            << " raw_hits=" << counters.violations
            << " uncertain=" << counters.uncertain
            << " device_flags=" << host_status
            << " scene_sha256="
            << hex_digest(scene.header.scene_sha256, 32) << "\n";
  std::cout << "TIMING_MS"
            << " load_validate=" << timing.load_validate_ms
            << " cpu_lower=" << timing.cpu_lower_ms
            << " bruteforce_verify=" << timing.bruteforce_verify_ms
            << " cuda_init=" << timing.cuda_init_ms
            << " alloc_upload=" << timing.alloc_upload_ms
            << " well_expand=" << timing.well_expand_ms
            << " grid_count=" << timing.grid_count_ms
            << " grid_build=" << timing.grid_build_ms
            << " active_query=" << timing.active_query_ms
            << " d2h=" << timing.d2h_ms
            << " cuda_cleanup=" << timing.cuda_cleanup_ms
            << " gpu_island="
            << timing.alloc_upload_ms + timing.well_expand_ms +
                   timing.grid_count_ms + timing.grid_build_ms +
                   timing.active_query_ms + timing.d2h_ms +
                   timing.cuda_cleanup_ms
            << " warm_context_total="
            << timing.total_ms - timing.cuda_init_ms
            << " cold_standalone_total=" << timing.total_ms
            << " total=" << timing.total_ms << "\n";
  const std::uint32_t copied_samples =
      std::min(sample_count, kSampleCapacity);
  for (std::uint32_t i = 0; i < copied_samples; ++i) {
    std::cout << "SAMPLE"
              << " context=" << samples[i].context_id
              << " active_local=" << samples[i].active_edge_local
              << " well=" << samples[i].well_edge
              << " verdict=" << samples[i].verdict << "\n";
  }
  if (std::string(verdict) == "UNCERTAIN") {
    return 2;
  }
  return std::string(verdict) == "RAW_HIT_FALLBACK" ? 3 : 0;
}

}  // namespace

int main(int argc, char **argv) {
  const Clock::time_point total_begin = Clock::now();
  try {
    return run(parse_options(argc, argv), total_begin);
  } catch (const std::exception &error) {
    std::cerr << "ACTIVE3_GPU_ISLAND verdict=UNCERTAIN error=\""
              << error.what() << "\"\n";
    return 2;
  }
}
