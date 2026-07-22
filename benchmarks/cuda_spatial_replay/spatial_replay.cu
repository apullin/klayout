/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/

// Standalone CUDA broad-phase feasibility harness for KLayout geometry work.
//
// This intentionally does not depend on KLayout internals.  It accepts compact
// AABB/context records, enumerates spatial candidates on a GPU, and
// returns a deterministic sorted/unique list of record-ID pairs.  Exact DRC
// predicates remain out of scope and belong on the CPU after this broad phase.

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum RecordFlag : std::uint32_t {
  kRecordHasEndpoints = 1u << 0,
  kRecordSideB = 1u << 1,
};

// Host/replay record.  Generic scanners need only the AABB; edge-based exact
// replay can additionally consume the optional endpoints.
struct alignas(16) InputRecord {
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::int64_t x1;
  std::int64_t y1;
  std::int64_t x2;
  std::int64_t y2;
  std::uint32_t id;
  std::uint32_t property;
  std::uint32_t context;
  std::uint32_t flags;
};

// Device broad-phase record.  It deliberately omits optional endpoint data.
struct alignas(16) PackedAabb {
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint32_t id;
  std::uint32_t property;
  std::uint32_t context;
  std::uint32_t flags;
};

static_assert(std::is_trivially_copyable<InputRecord>::value,
              "replay records must remain trivially copyable");
static_assert(std::is_trivially_copyable<PackedAabb>::value,
              "device records must remain trivially copyable");
static_assert(sizeof(InputRecord) == 80, "unexpected replay-record padding");
static_assert(sizeof(PackedAabb) == 48, "unexpected device-record padding");

struct ReplayHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_size;
  std::uint32_t record_size;
  std::uint32_t coordinate_bits;
  std::uint64_t record_count;
};

static_assert(sizeof(ReplayHeader) == 32, "unexpected replay-header padding");

constexpr std::array<char, 8> kReplayMagic = {'K', 'S', 'P', 'A', 'T', '0', '1', '\0'};
constexpr std::uint32_t kReplayVersion = 1;

enum FallbackFlag : std::uint32_t {
  kFallbackNone = 0,
  kFallbackCoordinateOverflow = 1u << 0,
  kFallbackRecordCellSpan = 1u << 1,
  kFallbackMembershipCapacity = 1u << 2,
  kFallbackDenseCell = 1u << 3,
  kFallbackPairWorkCapacity = 1u << 4,
  kFallbackPairCapacity = 1u << 5,
};

struct Options {
  std::uint64_t record_count = 200000;
  std::uint32_t context_count = 32;
  std::uint64_t world_size = 100000;
  std::uint64_t object_size = 64;
  std::uint64_t enlargement = 32;
  std::uint64_t cell_size = 128;
  std::uint64_t seed = 1;
  std::uint32_t max_cells_per_record = 64;
  std::uint32_t max_edges_per_cell = 4096;
  std::uint64_t max_memberships = 16000000;
  std::uint64_t max_pair_work = 64000000;
  std::uint64_t max_candidates = 8000000;
  int device = 0;
  std::uint32_t warmup = 0;
  std::uint32_t repeat = 1;
  std::string reference = "grid";
  std::string geometry = "aabb";
  std::string mode = "self";
  std::string fixture = "synthetic";
  std::string input_path;
  std::string write_input_path;
  std::string output_pairs_path;
};

struct GridConfig {
  std::uint64_t cell_size = 0;
  std::uint64_t enlargement = 0;
  std::uint32_t max_cells_per_record = 0;
  std::uint32_t max_edges_per_cell = 0;
  std::uint32_t bipartite = 0;
  std::uint64_t pair_capacity = 0;
};

struct CellRange {
  std::int64_t x0;
  std::int64_t x1;
  std::int64_t y0;
  std::int64_t y1;
};

struct alignas(8) CellKey {
  std::int64_t x;
  std::int64_t y;
  std::uint32_t context;
  std::uint32_t reserved;
};

struct CellKeyLess {
  __host__ __device__ bool operator()(const CellKey &a, const CellKey &b) const {
    if (a.context != b.context) return a.context < b.context;
    if (a.y != b.y) return a.y < b.y;
    if (a.x != b.x) return a.x < b.x;
    return a.reserved < b.reserved;
  }
};

struct CellKeyEqual {
  __host__ __device__ bool operator()(const CellKey &a, const CellKey &b) const {
    return a.context == b.context && a.x == b.x && a.y == b.y;
  }
};

struct PipelineResult {
  std::uint32_t fallback_flags = kFallbackNone;
  std::uint64_t memberships = 0;
  std::uint64_t occupied_cells = 0;
  std::uint64_t pair_work = 0;
  std::uint64_t raw_candidates = 0;
  std::vector<std::uint64_t> pairs;
  double setup_ms = 0.0;
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double sort_dedup_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
  double enumeration_theoretical_occupancy_pct = 0.0;
  std::uint32_t enumeration_blocks = 0;
};

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

[[noreturn]] void usage_error(const std::string &message) {
  throw std::runtime_error(message + " (use --help for usage)");
}

std::uint64_t parse_u64(const char *text, const char *name) {
  std::size_t consumed = 0;
  std::uint64_t result = 0;
  try {
    result = std::stoull(text, &consumed, 0);
  } catch (const std::exception &) {
    usage_error(std::string("invalid value for ") + name + ": " + text);
  }
  if (text[consumed] != '\0') {
    usage_error(std::string("invalid value for ") + name + ": " + text);
  }
  return result;
}

std::uint32_t parse_u32(const char *text, const char *name) {
  const std::uint64_t result = parse_u64(text, name);
  if (result > std::numeric_limits<std::uint32_t>::max()) {
    usage_error(std::string("value is too large for ") + name);
  }
  return static_cast<std::uint32_t>(result);
}

void print_help(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Input (synthetic unless --input is supplied):\n"
      << "  --input PATH                  read a v1 compact replay\n"
      << "  --write-input PATH            write the packed input replay\n"
      << "  --records N                   synthetic record count (default 200000)\n"
      << "  --contexts N                  synthetic context count (default 32)\n"
      << "  --world-size N                synthetic coordinate extent\n"
      << "  --object-size N               maximum synthetic AABB/edge size\n"
      << "  --geometry aabb|edges         synthetic geometry (default aabb)\n"
      << "  --mode self|bipartite         scanner mode (default self)\n"
      << "  --fixture synthetic|boundary|extrema|overflow\n"
      << "  --seed N                      deterministic synthetic seed\n\n"
      << "Broad phase:\n"
      << "  --enlargement N               strict KLayout-style range (default 32)\n"
      << "  --cell-size N                 uniform-grid cell size (default 128)\n"
      << "  --max-cells-per-record N      fail-closed span limit (default 64)\n"
      << "  --max-edges-per-cell N        fail-closed density limit (default 4096)\n"
      << "  --max-memberships N           fail-closed grid-entry limit\n"
      << "  --max-pair-work N             fail-closed cell-pair work limit\n"
      << "  --max-candidates N            fail-closed raw-pair limit\n"
      << "  --device N                    CUDA device index (default 0)\n"
      << "  --warmup N                    unreported warmup pipelines\n"
      << "  --repeat N                    measured end-to-end repetitions\n\n"
      << "Verification/output:\n"
      << "  --reference grid|exhaustive|none (default grid)\n"
      << "  --output-pairs PATH           write sorted uint64 pair keys\n"
      << "  --help                        show this text\n";
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    auto value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) usage_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (argument == "--help") {
      print_help(argv[0]);
      std::exit(0);
    } else if (argument == "--input") {
      options.input_path = value("--input");
    } else if (argument == "--write-input") {
      options.write_input_path = value("--write-input");
    } else if (argument == "--output-pairs") {
      options.output_pairs_path = value("--output-pairs");
    } else if (argument == "--records" || argument == "--edges") {
      options.record_count = parse_u64(value(argument.c_str()), argument.c_str());
    } else if (argument == "--contexts") {
      options.context_count = parse_u32(value("--contexts"), "--contexts");
    } else if (argument == "--world-size") {
      options.world_size = parse_u64(value("--world-size"), "--world-size");
    } else if (argument == "--object-size" || argument == "--edge-length") {
      options.object_size = parse_u64(value(argument.c_str()), argument.c_str());
    } else if (argument == "--geometry") {
      options.geometry = value("--geometry");
      if (options.geometry != "aabb" && options.geometry != "edges") {
        usage_error("--geometry must be aabb or edges");
      }
    } else if (argument == "--mode") {
      options.mode = value("--mode");
      if (options.mode != "self" && options.mode != "bipartite") {
        usage_error("--mode must be self or bipartite");
      }
    } else if (argument == "--fixture") {
      options.fixture = value("--fixture");
      if (options.fixture != "synthetic" && options.fixture != "boundary" &&
          options.fixture != "extrema" && options.fixture != "overflow") {
        usage_error("--fixture must be synthetic, boundary, extrema, or overflow");
      }
    } else if (argument == "--seed") {
      options.seed = parse_u64(value("--seed"), "--seed");
    } else if (argument == "--enlargement" || argument == "--padding") {
      options.enlargement = parse_u64(value(argument.c_str()), argument.c_str());
    } else if (argument == "--cell-size") {
      options.cell_size = parse_u64(value("--cell-size"), "--cell-size");
    } else if (argument == "--max-cells-per-record" ||
               argument == "--max-cells-per-edge") {
      options.max_cells_per_record =
          parse_u32(value(argument.c_str()), argument.c_str());
    } else if (argument == "--max-edges-per-cell") {
      options.max_edges_per_cell =
          parse_u32(value("--max-edges-per-cell"), "--max-edges-per-cell");
    } else if (argument == "--max-memberships") {
      options.max_memberships =
          parse_u64(value("--max-memberships"), "--max-memberships");
    } else if (argument == "--max-pair-work") {
      options.max_pair_work =
          parse_u64(value("--max-pair-work"), "--max-pair-work");
    } else if (argument == "--max-candidates") {
      options.max_candidates =
          parse_u64(value("--max-candidates"), "--max-candidates");
    } else if (argument == "--device") {
      options.device = static_cast<int>(parse_u32(value("--device"), "--device"));
    } else if (argument == "--warmup") {
      options.warmup = parse_u32(value("--warmup"), "--warmup");
    } else if (argument == "--repeat") {
      options.repeat = parse_u32(value("--repeat"), "--repeat");
    } else if (argument == "--reference") {
      options.reference = value("--reference");
      if (options.reference != "grid" && options.reference != "exhaustive" &&
          options.reference != "none") {
        usage_error("--reference must be grid, exhaustive, or none");
      }
    } else {
      usage_error("unknown option: " + argument);
    }
  }

  if (options.record_count == 0) usage_error("--records must be nonzero");
  if (options.record_count > std::numeric_limits<std::uint32_t>::max())
    usage_error("--records exceeds the prototype's uint32 index space");
  if (options.context_count == 0) usage_error("--contexts must be nonzero");
  if (options.world_size == 0) usage_error("--world-size must be nonzero");
  if (options.object_size == 0) usage_error("--object-size must be nonzero");
  if (options.world_size >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      options.object_size >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      options.world_size >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) -
              options.object_size) {
    usage_error("synthetic coordinate extent exceeds int64");
  }
  if (options.cell_size == 0) usage_error("--cell-size must be nonzero");
  if (options.max_cells_per_record == 0)
    usage_error("--max-cells-per-record must be nonzero");
  if (options.max_edges_per_cell < 2)
    usage_error("--max-edges-per-cell must be at least two");
  if (options.max_memberships == 0) usage_error("--max-memberships must be nonzero");
  if (options.max_pair_work == 0) usage_error("--max-pair-work must be nonzero");
  if (options.max_candidates == 0) usage_error("--max-candidates must be nonzero");
  if (options.repeat == 0) usage_error("--repeat must be nonzero");
  return options;
}

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

 private:
  std::uint64_t state_;
};

InputRecord make_aabb(std::int64_t left, std::int64_t bottom,
                      std::int64_t right, std::int64_t top, std::uint32_t id,
                      std::uint32_t property, std::uint32_t context,
                      std::uint32_t flags = 0) {
  return InputRecord{left, bottom, right, top, 0, 0, 0, 0, id, property,
                     context, flags};
}

std::vector<InputRecord> generate_boundary_fixture(const Options &options) {
  if (options.enlargement >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() - 32)) {
    throw std::runtime_error("boundary fixture enlargement is too large");
  }
  const std::int64_t enlargement =
      static_cast<std::int64_t>(options.enlargement);
  const std::int64_t passing_left =
      enlargement == 0 ? 9 : 10 + enlargement - 1;
  const std::int64_t failing_left = 10 + enlargement;
  return {
      make_aabb(0, 0, 10, 10, 1, 0, 0),
      make_aabb(passing_left, 0, passing_left + 10, 10, 2, 0, 0,
                kRecordSideB),
      make_aabb(0, 100, 10, 110, 3, 0, 1),
      make_aabb(failing_left, 100, failing_left + 10, 110, 4, 0, 1,
                kRecordSideB),
      make_aabb(0, 200, 10, 210, 5, 0, 2),
      make_aabb(10, 200, 20, 210, 6, 0, 2, kRecordSideB),
  };
}

std::vector<InputRecord> generate_records(const Options &options) {
  if (options.fixture == "boundary") return generate_boundary_fixture(options);
  if (options.fixture == "extrema") {
    if (options.enlargement > 1024) {
      throw std::runtime_error("extrema fixture requires enlargement <= 1024");
    }
    const std::int64_t e = static_cast<std::int64_t>(options.enlargement);
    const std::int64_t lo = std::numeric_limits<std::int64_t>::min() + e;
    const std::int64_t hi = std::numeric_limits<std::int64_t>::max() - e;
    return {make_aabb(lo, lo, lo + 8, lo + 8, 1, 0, 0),
            make_aabb(lo + 4, lo + 4, lo + 12, lo + 12, 2, 0, 0,
                      kRecordSideB),
            make_aabb(hi - 12, hi - 12, hi - 4, hi - 4, 3, 0, 1),
            make_aabb(hi - 8, hi - 8, hi, hi, 4, 0, 1, kRecordSideB)};
  }
  if (options.fixture == "overflow") {
    return {make_aabb(std::numeric_limits<std::int64_t>::min(), 0,
                      std::numeric_limits<std::int64_t>::min() + 8, 8, 1, 0,
                      0),
            make_aabb(0, 0, 8, 8, 2, 0, 0, kRecordSideB)};
  }
  std::vector<InputRecord> records;
  records.reserve(static_cast<std::size_t>(options.record_count));
  SplitMix64 random(options.seed);
  const std::uint64_t coordinate_extent = std::max<std::uint64_t>(1, options.world_size);
  for (std::uint64_t i = 0; i < options.record_count; ++i) {
    const std::int64_t x = static_cast<std::int64_t>(random.next() % coordinate_extent);
    const std::int64_t y = static_cast<std::int64_t>(random.next() % coordinate_extent);
    const std::int64_t first_size =
        static_cast<std::int64_t>(1 + random.next() % options.object_size);
    const std::int64_t second_size =
        static_cast<std::int64_t>(1 + random.next() % options.object_size);
    const std::uint32_t id = static_cast<std::uint32_t>(i + 1);
    const std::uint32_t property = static_cast<std::uint32_t>(random.next());
    const std::uint32_t context =
        static_cast<std::uint32_t>(random.next() % options.context_count);
    if (options.geometry == "aabb") {
      records.push_back(make_aabb(x, y, x + first_size, y + second_size, id,
                                  property, context,
                                  (i & 1u) ? kRecordSideB : 0));
    } else {
      const bool horizontal = (random.next() & 1u) != 0;
      const bool reverse = (random.next() & 1u) != 0;
      const std::int64_t dx = horizontal ? (reverse ? -first_size : first_size) : 0;
      const std::int64_t dy = horizontal ? 0 : (reverse ? -first_size : first_size);
      const std::int64_t x2 = x + dx;
      const std::int64_t y2 = y + dy;
      records.push_back(InputRecord{std::min(x, x2), std::min(y, y2),
                                    std::max(x, x2), std::max(y, y2), x, y, x2,
                                    y2, id, property, context,
                                    kRecordHasEndpoints |
                                        ((i & 1u) ? kRecordSideB : 0)});
    }
  }
  return records;
}

bool host_is_little_endian() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t *>(&value) == 1;
}

std::vector<InputRecord> read_replay(const std::string &path) {
  if (!host_is_little_endian()) {
    throw std::runtime_error("v1 replay reader requires a little-endian host");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open replay: " + path);
  ReplayHeader header{};
  input.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!input || !std::equal(kReplayMagic.begin(), kReplayMagic.end(), header.magic) ||
      header.version != kReplayVersion || header.header_size != sizeof(ReplayHeader) ||
      header.record_size != sizeof(InputRecord) || header.coordinate_bits != 64) {
    throw std::runtime_error("invalid or unsupported replay header: " + path);
  }
  if (header.record_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("replay exceeds the prototype's uint32 index space");
  }
  std::vector<InputRecord> records(static_cast<std::size_t>(header.record_count));
  input.read(reinterpret_cast<char *>(records.data()),
             static_cast<std::streamsize>(records.size() * sizeof(InputRecord)));
  if (!input) throw std::runtime_error("truncated replay: " + path);
  if (input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("trailing data after replay records: " + path);
  }
  return records;
}

void write_replay(const std::string &path,
                  const std::vector<InputRecord> &records) {
  if (!host_is_little_endian()) {
    throw std::runtime_error("v1 replay writer requires a little-endian host");
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create replay: " + path);
  ReplayHeader header{};
  std::copy(kReplayMagic.begin(), kReplayMagic.end(), header.magic);
  header.version = kReplayVersion;
  header.header_size = sizeof(ReplayHeader);
  header.record_size = sizeof(InputRecord);
  header.coordinate_bits = 64;
  header.record_count = records.size();
  output.write(reinterpret_cast<const char *>(&header), sizeof(header));
  output.write(reinterpret_cast<const char *>(records.data()),
               static_cast<std::streamsize>(records.size() * sizeof(InputRecord)));
  if (!output) throw std::runtime_error("failed writing replay: " + path);
}

std::vector<PackedAabb> pack_records(const std::vector<InputRecord> &input) {
  std::vector<PackedAabb> packed;
  packed.reserve(input.size());
  std::vector<std::uint32_t> ids;
  ids.reserve(input.size());
  for (const InputRecord &record : input) {
    if (record.left > record.right || record.bottom > record.top) {
      throw std::runtime_error("input contains a non-normalized AABB");
    }
    if ((record.flags & kRecordHasEndpoints) != 0 &&
        (record.left != std::min(record.x1, record.x2) ||
         record.right != std::max(record.x1, record.x2) ||
         record.bottom != std::min(record.y1, record.y2) ||
         record.top != std::max(record.y1, record.y2))) {
      throw std::runtime_error("endpoint sidecar does not match its AABB");
    }
    packed.push_back(PackedAabb{record.left, record.bottom, record.right,
                                record.top, record.id, record.property,
                                record.context, record.flags});
    ids.push_back(record.id);
  }
  std::sort(ids.begin(), ids.end());
  if (ids.empty() || ids.front() == 0) {
    throw std::runtime_error("record IDs must be nonzero");
  }
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    throw std::runtime_error("record IDs must be globally unique");
  }
  return packed;
}

__host__ __device__ std::int64_t grid_min_x(const PackedAabb &box,
                                             std::uint64_t enlargement) {
  return box.left - static_cast<std::int64_t>(enlargement);
}
__host__ __device__ std::int64_t grid_max_x(const PackedAabb &box,
                                             std::uint64_t enlargement) {
  return box.right + static_cast<std::int64_t>(enlargement);
}
__host__ __device__ std::int64_t grid_min_y(const PackedAabb &box,
                                             std::uint64_t enlargement) {
  return box.bottom - static_cast<std::int64_t>(enlargement);
}
__host__ __device__ std::int64_t grid_max_y(const PackedAabb &box,
                                             std::uint64_t enlargement) {
  return box.top + static_cast<std::int64_t>(enlargement);
}

// Exact broad-phase predicate used by KLayout's db::bs_boxes_overlap: strict
// comparisons and one enlargement, not two padded boxes and not <=.
__host__ __device__ bool boxes_overlap_strict(const PackedAabb &a,
                                              const PackedAabb &b,
                                              std::uint64_t enlargement) {
  const std::int64_t e = static_cast<std::int64_t>(enlargement);
  return a.left < b.right + e && b.left < a.right + e &&
         a.bottom < b.top + e && b.bottom < a.top + e;
}

bool boxes_overlap_strict_host(const PackedAabb &a, const PackedAabb &b,
                               std::uint64_t enlargement) {
  const __int128 e = static_cast<__int128>(enlargement);
  return static_cast<__int128>(a.left) < static_cast<__int128>(b.right) + e &&
         static_cast<__int128>(b.left) < static_cast<__int128>(a.right) + e &&
         static_cast<__int128>(a.bottom) < static_cast<__int128>(b.top) + e &&
         static_cast<__int128>(b.bottom) < static_cast<__int128>(a.top) + e;
}

GridConfig make_grid(const std::vector<PackedAabb> &records,
                     const Options &options,
                     std::uint32_t &fallback_flags) {
  GridConfig config{};
  config.cell_size = options.cell_size;
  config.enlargement = options.enlargement;
  config.max_cells_per_record = options.max_cells_per_record;
  config.max_edges_per_cell = options.max_edges_per_cell;
  config.bipartite = options.mode == "bipartite" ? 1u : 0u;
  config.pair_capacity = options.max_candidates;
  if (options.enlargement >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      options.cell_size >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    fallback_flags |= kFallbackCoordinateOverflow;
    return config;
  }
  const std::int64_t enlargement =
      static_cast<std::int64_t>(options.enlargement);
  for (const PackedAabb &record : records) {
    if (record.left < std::numeric_limits<std::int64_t>::min() + enlargement ||
        record.bottom < std::numeric_limits<std::int64_t>::min() + enlargement ||
        record.right > std::numeric_limits<std::int64_t>::max() - enlargement ||
        record.top > std::numeric_limits<std::int64_t>::max() - enlargement) {
      fallback_flags |= kFallbackCoordinateOverflow;
      return config;
    }
  }
  return config;
}

__host__ __device__ std::int64_t floor_div(std::int64_t value,
                                           std::int64_t divisor) {
  std::int64_t quotient = value / divisor;
  const std::int64_t remainder = value % divisor;
  if (remainder < 0) --quotient;
  return quotient;
}

__host__ __device__ CellRange cell_range(const PackedAabb &record,
                                         const GridConfig &config) {
  const std::int64_t divisor = static_cast<std::int64_t>(config.cell_size);
  return CellRange{
      floor_div(grid_min_x(record, config.enlargement), divisor),
      floor_div(grid_max_x(record, config.enlargement), divisor),
      floor_div(grid_min_y(record, config.enlargement), divisor),
      floor_div(grid_max_y(record, config.enlargement), divisor)};
}

__host__ __device__ bool membership_count_bounded(const CellRange &range,
                                                  std::uint64_t limit,
                                                  std::uint64_t &count) {
  const std::uint64_t dx = static_cast<std::uint64_t>(range.x1) -
                           static_cast<std::uint64_t>(range.x0);
  const std::uint64_t dy = static_cast<std::uint64_t>(range.y1) -
                           static_cast<std::uint64_t>(range.y0);
  if (dx == UINT64_MAX || dy == UINT64_MAX) {
    return false;
  }
  const std::uint64_t width = dx + 1;
  const std::uint64_t height = dy + 1;
  if (width > limit || height > limit || width > limit / height) return false;
  count = width * height;
  return true;
}

__host__ __device__ CellKey cell_key(std::uint32_t context, std::int64_t x,
                                     std::int64_t y, std::uint32_t side) {
  return CellKey{x, y, context, side};
}

__host__ __device__ std::uint64_t pair_key(std::uint32_t a, std::uint32_t b) {
  const std::uint32_t lo = a < b ? a : b;
  const std::uint32_t hi = a < b ? b : a;
  return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

__global__ void count_memberships_kernel(const PackedAabb *records,
                                         std::uint32_t record_count,
                                         GridConfig config,
                                         std::uint32_t *counts,
                                         std::uint32_t *fallback_flags) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < record_count; index += stride) {
    const CellRange range = cell_range(records[index], config);
    std::uint64_t count = 0;
    if (!membership_count_bounded(range, config.max_cells_per_record, count)) {
      counts[index] = 0;
      atomicOr(fallback_flags,
               static_cast<std::uint32_t>(kFallbackRecordCellSpan));
    } else {
      counts[index] = static_cast<std::uint32_t>(count);
    }
  }
}

__global__ void fill_memberships_kernel(const PackedAabb *records,
                                        std::uint32_t record_count,
                                        GridConfig config,
                                        const std::uint64_t *offsets,
                                        CellKey *keys,
                                        std::uint32_t *record_indices) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < record_count; index += stride) {
    const CellRange range = cell_range(records[index], config);
    std::uint64_t output = offsets[index];
    for (std::int64_t y = range.y0;; ++y) {
      for (std::int64_t x = range.x0;; ++x) {
        keys[output] = cell_key(
            records[index].context, x, y,
            (records[index].flags & kRecordSideB) != 0 ? 1u : 0u);
        record_indices[output] = static_cast<std::uint32_t>(index);
        ++output;
        if (x == range.x1) break;
      }
      if (y == range.y1) break;
    }
  }
}

__global__ void count_pair_work_kernel(const PackedAabb *records,
                                       const std::uint32_t *record_indices,
                                       const std::uint64_t *cell_offsets,
                                       const std::uint32_t *cell_counts,
                                       std::uint64_t occupied_cells,
                                       GridConfig config,
                                       std::uint64_t *pair_work_counts,
                                       std::uint32_t *side_a_counts,
                                       std::uint32_t *fallback_flags) {
  const std::uint64_t first_cell =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t cell = first_cell; cell < occupied_cells; cell += stride) {
    const std::uint32_t count = cell_counts[cell];
    if (count > config.max_edges_per_cell) {
      pair_work_counts[cell] = 0;
      atomicOr(fallback_flags, static_cast<std::uint32_t>(kFallbackDenseCell));
    } else {
      if (config.bipartite) {
        const std::uint64_t offset = cell_offsets[cell];
        std::uint32_t side_a = 0;
        while (side_a < count &&
               (records[record_indices[offset + side_a]].flags &
                kRecordSideB) == 0) {
          ++side_a;
        }
        side_a_counts[cell] = side_a;
        pair_work_counts[cell] =
            static_cast<std::uint64_t>(side_a) * (count - side_a);
      } else {
        side_a_counts[cell] = 0;
        pair_work_counts[cell] =
            static_cast<std::uint64_t>(count) * (count - 1) / 2;
      }
    }
  }
}

__device__ std::uint64_t pair_row_start(std::uint32_t row,
                                        std::uint32_t count) {
  return static_cast<std::uint64_t>(row) * (2ULL * count - row - 1) / 2;
}

__global__ void mark_pair_candidates_kernel(
    const PackedAabb *records, const std::uint32_t *record_indices,
    const std::uint64_t *cell_offsets, const std::uint32_t *cell_counts,
    const std::uint32_t *side_a_counts,
    const std::uint64_t *pair_work_offsets, std::uint64_t occupied_cells,
    std::uint64_t total_pair_work, GridConfig config,
    std::uint64_t *candidate_or_zero) {
  const std::uint64_t first_work =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t work = first_work; work < total_pair_work;
       work += stride) {
    std::uint64_t lo = 0;
    std::uint64_t hi = occupied_cells;
    while (lo < hi) {
      const std::uint64_t mid = lo + (hi - lo) / 2;
      if (pair_work_offsets[mid] <= work) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const std::uint64_t cell = lo - 1;
    const std::uint64_t local = work - pair_work_offsets[cell];
    const std::uint32_t count = cell_counts[cell];
    std::uint32_t first_local = 0;
    std::uint32_t second_local = 0;
    if (config.bipartite) {
      const std::uint32_t side_a = side_a_counts[cell];
      const std::uint32_t side_b = count - side_a;
      first_local = static_cast<std::uint32_t>(local / side_b);
      second_local =
          side_a + static_cast<std::uint32_t>(local % side_b);
    } else {
      std::uint32_t row_lo = 0;
      std::uint32_t row_hi = count - 1;
      while (row_lo < row_hi) {
        const std::uint32_t mid = row_lo + (row_hi - row_lo + 1) / 2;
        if (pair_row_start(mid, count) <= local) {
          row_lo = mid;
        } else {
          row_hi = mid - 1;
        }
      }
      first_local = row_lo;
      second_local = static_cast<std::uint32_t>(
          first_local + 1 + local - pair_row_start(first_local, count));
    }
    const std::uint64_t membership_offset = cell_offsets[cell];
    const PackedAabb first =
        records[record_indices[membership_offset + first_local]];
    const PackedAabb second =
        records[record_indices[membership_offset + second_local]];
    candidate_or_zero[work] =
        boxes_overlap_strict(first, second, config.enlargement)
            ? pair_key(first.id, second.id)
            : 0;
  }
}

void cuda_check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

PipelineResult run_gpu_stages(const std::vector<PackedAabb> &records,
                              const GridConfig &config,
                              const Options &options,
                              std::uint32_t initial_fallback_flags) {
  PipelineResult result;
  result.fallback_flags = initial_fallback_flags;
  if (result.fallback_flags != kFallbackNone) {
    return result;
  }

  const auto setup_begin = Clock::now();
  cuda_check(cudaSetDevice(options.device), "cudaSetDevice");
  cuda_check(cudaFree(nullptr), "CUDA context initialization");
  thrust::device_vector<PackedAabb> device_records(records.size());
  thrust::device_vector<std::uint32_t> device_status(1, 0);
  const auto setup_end = Clock::now();
  result.setup_ms = elapsed_ms(setup_begin, setup_end);

  const auto h2d_begin = Clock::now();
  cuda_check(cudaMemcpy(thrust::raw_pointer_cast(device_records.data()),
                        records.data(), records.size() * sizeof(PackedAabb),
                        cudaMemcpyHostToDevice),
             "record H2D copy");
  const auto h2d_end = Clock::now();
  result.h2d_ms = elapsed_ms(h2d_begin, h2d_end);

  const auto kernel_begin = Clock::now();
  const std::uint32_t record_count =
      static_cast<std::uint32_t>(records.size());
  constexpr std::uint32_t threads = 256;
  const std::uint32_t record_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>((static_cast<std::uint64_t>(record_count) +
                               threads - 1) /
                                  threads,
                              65535));
  thrust::device_vector<std::uint32_t> counts(record_count);
  thrust::device_vector<std::uint64_t> offsets(record_count);
  count_memberships_kernel<<<record_blocks, threads>>>(
      thrust::raw_pointer_cast(device_records.data()), record_count, config,
      thrust::raw_pointer_cast(counts.data()),
      thrust::raw_pointer_cast(device_status.data()));
  cuda_check(cudaGetLastError(), "count-memberships launch");
  thrust::exclusive_scan(thrust::device, counts.begin(), counts.end(),
                         offsets.begin(), std::uint64_t{0});

  std::uint64_t final_offset = 0;
  std::uint32_t final_count = 0;
  cuda_check(cudaMemcpy(&final_offset,
                        thrust::raw_pointer_cast(offsets.data()) + record_count - 1,
                        sizeof(final_offset), cudaMemcpyDeviceToHost),
             "membership-offset control copy");
  cuda_check(cudaMemcpy(&final_count,
                        thrust::raw_pointer_cast(counts.data()) + record_count - 1,
                        sizeof(final_count), cudaMemcpyDeviceToHost),
             "membership-count control copy");
  result.memberships = final_offset + final_count;
  cuda_check(cudaMemcpy(&result.fallback_flags,
                        thrust::raw_pointer_cast(device_status.data()),
                        sizeof(result.fallback_flags), cudaMemcpyDeviceToHost),
             "membership-status control copy");
  if (result.memberships > options.max_memberships) {
    result.fallback_flags |= kFallbackMembershipCapacity;
  }
  if (result.fallback_flags != kFallbackNone) {
    cuda_check(cudaDeviceSynchronize(), "membership-stage synchronize");
    result.kernel_ms = elapsed_ms(kernel_begin, Clock::now());
    return result;
  }

  thrust::device_vector<CellKey> membership_keys(result.memberships);
  thrust::device_vector<std::uint32_t> membership_records(result.memberships);
  fill_memberships_kernel<<<record_blocks, threads>>>(
      thrust::raw_pointer_cast(device_records.data()), record_count, config,
      thrust::raw_pointer_cast(offsets.data()),
      thrust::raw_pointer_cast(membership_keys.data()),
      thrust::raw_pointer_cast(membership_records.data()));
  cuda_check(cudaGetLastError(), "fill-memberships launch");
  thrust::sort_by_key(thrust::device, membership_keys.begin(),
                      membership_keys.end(), membership_records.begin(),
                      CellKeyLess{});

  thrust::device_vector<CellKey> unique_cell_keys(result.memberships);
  thrust::device_vector<std::uint32_t> cell_counts(result.memberships);
  const auto reduced_end = thrust::reduce_by_key(
      thrust::device, membership_keys.begin(), membership_keys.end(),
      thrust::make_constant_iterator<std::uint32_t>(1), unique_cell_keys.begin(),
      cell_counts.begin(), CellKeyEqual{});
  result.occupied_cells =
      static_cast<std::uint64_t>(reduced_end.first - unique_cell_keys.begin());
  unique_cell_keys.resize(result.occupied_cells);
  cell_counts.resize(result.occupied_cells);
  thrust::device_vector<std::uint64_t> cell_offsets(result.occupied_cells);
  thrust::exclusive_scan(thrust::device, cell_counts.begin(), cell_counts.end(),
                         cell_offsets.begin(), std::uint64_t{0});

  thrust::device_vector<std::uint64_t> pair_work_counts(result.occupied_cells);
  thrust::device_vector<std::uint64_t> pair_work_offsets(result.occupied_cells);
  thrust::device_vector<std::uint32_t> side_a_counts(result.occupied_cells);
  const std::uint32_t cell_count_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>((result.occupied_cells + threads - 1) / threads,
                              65535));
  if (cell_count_blocks != 0) {
    count_pair_work_kernel<<<cell_count_blocks, threads>>>(
        thrust::raw_pointer_cast(device_records.data()),
        thrust::raw_pointer_cast(membership_records.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(cell_counts.data()), result.occupied_cells,
        config,
        thrust::raw_pointer_cast(pair_work_counts.data()),
        thrust::raw_pointer_cast(side_a_counts.data()),
        thrust::raw_pointer_cast(device_status.data()));
    cuda_check(cudaGetLastError(), "count-pair-work launch");
  }
  thrust::exclusive_scan(thrust::device, pair_work_counts.begin(),
                         pair_work_counts.end(), pair_work_offsets.begin(),
                         std::uint64_t{0});
  if (result.occupied_cells != 0) {
    std::uint64_t last_offset = 0;
    std::uint64_t last_count = 0;
    cuda_check(cudaMemcpy(
                   &last_offset,
                   thrust::raw_pointer_cast(pair_work_offsets.data()) +
                       result.occupied_cells - 1,
                   sizeof(last_offset), cudaMemcpyDeviceToHost),
               "pair-work-offset control copy");
    cuda_check(cudaMemcpy(
                   &last_count,
                   thrust::raw_pointer_cast(pair_work_counts.data()) +
                       result.occupied_cells - 1,
                   sizeof(last_count), cudaMemcpyDeviceToHost),
               "pair-work-count control copy");
    result.pair_work = last_offset + last_count;
  }
  cuda_check(cudaMemcpy(&result.fallback_flags,
                        thrust::raw_pointer_cast(device_status.data()),
                        sizeof(result.fallback_flags), cudaMemcpyDeviceToHost),
             "pair-work-status control copy");
  if (result.pair_work > options.max_pair_work) {
    result.fallback_flags |= kFallbackPairWorkCapacity;
  }
  if (result.fallback_flags != kFallbackNone) {
    cuda_check(cudaDeviceSynchronize(), "pair-work-stage synchronize");
    result.kernel_ms = elapsed_ms(kernel_begin, Clock::now());
    return result;
  }

  thrust::device_vector<std::uint64_t> candidate_pairs(result.pair_work);
  const std::uint64_t required_blocks =
      (result.pair_work + threads - 1) / threads;
  const std::uint32_t pair_blocks = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(required_blocks, 65535));
  result.enumeration_blocks = pair_blocks;
  int active_blocks_per_sm = 0;
  cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &active_blocks_per_sm, mark_pair_candidates_kernel, threads, 0),
             "enumeration occupancy query");
  cudaDeviceProp property{};
  cuda_check(cudaGetDeviceProperties(&property, options.device),
             "enumeration device-properties query");
  result.enumeration_theoretical_occupancy_pct =
      100.0 * active_blocks_per_sm * threads /
      static_cast<double>(property.maxThreadsPerMultiProcessor);
  if (pair_blocks != 0) {
    mark_pair_candidates_kernel<<<pair_blocks, threads>>>(
        thrust::raw_pointer_cast(device_records.data()),
        thrust::raw_pointer_cast(membership_records.data()),
        thrust::raw_pointer_cast(cell_offsets.data()),
        thrust::raw_pointer_cast(cell_counts.data()),
        thrust::raw_pointer_cast(side_a_counts.data()),
        thrust::raw_pointer_cast(pair_work_offsets.data()), result.occupied_cells,
        result.pair_work, config,
        thrust::raw_pointer_cast(candidate_pairs.data()));
    cuda_check(cudaGetLastError(), "mark-pair-candidates launch");
  }
  auto compact_end = thrust::remove(thrust::device, candidate_pairs.begin(),
                                    candidate_pairs.end(), std::uint64_t{0});
  cuda_check(cudaDeviceSynchronize(), "broad-phase synchronize");
  result.raw_candidates =
      static_cast<std::uint64_t>(compact_end - candidate_pairs.begin());
  if (result.raw_candidates > options.max_candidates) {
    result.fallback_flags |= kFallbackPairCapacity;
  }
  result.kernel_ms = elapsed_ms(kernel_begin, Clock::now());
  if (result.fallback_flags != kFallbackNone) {
    return result;
  }

  const auto sort_begin = Clock::now();
  auto pair_begin = candidate_pairs.begin();
  auto pair_end = pair_begin + static_cast<std::ptrdiff_t>(result.raw_candidates);
  thrust::sort(thrust::device, pair_begin, pair_end);
  pair_end = thrust::unique(thrust::device, pair_begin, pair_end);
  const std::size_t unique_count = static_cast<std::size_t>(pair_end - pair_begin);
  cuda_check(cudaDeviceSynchronize(), "candidate sort/unique synchronize");
  result.sort_dedup_ms = elapsed_ms(sort_begin, Clock::now());

  const auto d2h_begin = Clock::now();
  result.pairs.resize(unique_count);
  if (unique_count != 0) {
    cuda_check(cudaMemcpy(result.pairs.data(),
                          thrust::raw_pointer_cast(candidate_pairs.data()),
                          unique_count * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost),
               "candidate D2H copy");
  }
  result.d2h_ms = elapsed_ms(d2h_begin, Clock::now());
  return result;
}

// Keep the outer timer outside run_gpu_stages so all per-call device-vector
// destructors run before the charged pipeline wall is sampled.
PipelineResult run_gpu(const std::vector<PackedAabb> &records,
                       const GridConfig &config, const Options &options,
                       std::uint32_t initial_fallback_flags) {
  const auto total_begin = Clock::now();
  PipelineResult result =
      run_gpu_stages(records, config, options, initial_fallback_flags);
  result.total_ms = elapsed_ms(total_begin, Clock::now());
  return result;
}

std::vector<std::uint64_t> cpu_grid_reference(
    const std::vector<PackedAabb> &records, const GridConfig &config) {
  std::uint64_t total_memberships = 0;
  for (const PackedAabb &record : records) {
    std::uint64_t count = 0;
    if (!membership_count_bounded(cell_range(record, config),
                                  std::numeric_limits<std::uint64_t>::max(),
                                  count) ||
        total_memberships > std::numeric_limits<std::uint64_t>::max() - count) {
      throw std::runtime_error("CPU reference membership size overflow");
    }
    total_memberships += count;
  }
  if (total_memberships > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("CPU reference membership size overflow");
  }
  std::vector<std::pair<CellKey, std::uint32_t>> memberships;
  memberships.reserve(static_cast<std::size_t>(total_memberships));
  for (std::uint32_t index = 0; index < records.size(); ++index) {
    const CellRange range = cell_range(records[index], config);
    for (std::int64_t y = range.y0;; ++y) {
      for (std::int64_t x = range.x0;; ++x) {
        memberships.emplace_back(
            cell_key(records[index].context, x, y,
                     (records[index].flags & kRecordSideB) != 0 ? 1u : 0u),
            index);
        if (x == range.x1) break;
      }
      if (y == range.y1) break;
    }
  }
  std::sort(memberships.begin(), memberships.end(),
            [](const auto &a, const auto &b) {
              const CellKeyLess less;
              if (less(a.first, b.first)) return true;
              if (less(b.first, a.first)) return false;
              return a.second < b.second;
            });
  std::vector<std::uint64_t> pairs;
  for (std::size_t begin = 0; begin < memberships.size();) {
    std::size_t end = begin + 1;
    while (end < memberships.size() &&
           CellKeyEqual{}(memberships[end].first, memberships[begin].first)) {
      ++end;
    }
    for (std::size_t i = begin; i < end; ++i) {
      for (std::size_t j = i + 1; j < end; ++j) {
        const PackedAabb &a = records[memberships[i].second];
        const PackedAabb &b = records[memberships[j].second];
        const bool side_ok =
            !config.bipartite || ((a.flags ^ b.flags) & kRecordSideB) != 0;
        if (side_ok && boxes_overlap_strict_host(a, b, config.enlargement)) {
          pairs.push_back(pair_key(a.id, b.id));
        }
      }
    }
    begin = end;
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  return pairs;
}

std::vector<std::uint64_t> cpu_exhaustive_reference(
    const std::vector<PackedAabb> &records, const GridConfig &config) {
  std::vector<std::uint64_t> pairs;
  for (std::size_t i = 0; i < records.size(); ++i) {
    for (std::size_t j = i + 1; j < records.size(); ++j) {
      const bool side_ok =
          !config.bipartite ||
          ((records[i].flags ^ records[j].flags) & kRecordSideB) != 0;
      if (records[i].context == records[j].context && side_ok &&
          boxes_overlap_strict_host(records[i], records[j],
                                    config.enlargement)) {
        pairs.push_back(pair_key(records[i].id, records[j].id));
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  return pairs;
}

std::uint64_t pair_hash(const std::vector<std::uint64_t> &pairs) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint64_t pair : pairs) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
      hash ^= static_cast<std::uint8_t>(pair >> shift);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

std::string fallback_text(std::uint32_t flags) {
  if (flags == kFallbackNone) return "none";
  std::string result;
  auto append = [&](const char *name) {
    if (!result.empty()) result += ',';
    result += name;
  };
  if (flags & kFallbackCoordinateOverflow) append("coordinate-overflow");
  if (flags & kFallbackRecordCellSpan) append("record-cell-span");
  if (flags & kFallbackMembershipCapacity) append("membership-capacity");
  if (flags & kFallbackDenseCell) append("dense-cell");
  if (flags & kFallbackPairWorkCapacity) append("pair-work-capacity");
  if (flags & kFallbackPairCapacity) append("pair-capacity");
  return result;
}

void write_pairs(const std::string &path,
                 const std::vector<std::uint64_t> &pairs) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create pair output: " + path);
  const std::uint64_t count = pairs.size();
  output.write(reinterpret_cast<const char *>(&count), sizeof(count));
  output.write(reinterpret_cast<const char *>(pairs.data()),
               static_cast<std::streamsize>(pairs.size() * sizeof(std::uint64_t)));
  if (!output) throw std::runtime_error("failed writing pair output: " + path);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto input_begin = Clock::now();
    const std::vector<InputRecord> input = options.input_path.empty()
                                               ? generate_records(options)
                                               : read_replay(options.input_path);
    const auto input_end = Clock::now();
    if (input.empty()) throw std::runtime_error("input contains no records");

    const auto pack_begin = Clock::now();
    const std::vector<PackedAabb> records = pack_records(input);
    const auto pack_end = Clock::now();
    if (!options.write_input_path.empty()) {
      write_replay(options.write_input_path, input);
    }

    std::uint32_t initial_fallback_flags = kFallbackNone;
    const GridConfig config =
        make_grid(records, options, initial_fallback_flags);

    cudaDeviceProp property{};
    cuda_check(cudaGetDeviceProperties(&property, options.device),
               "cudaGetDeviceProperties");
    for (std::uint32_t i = 0; i < options.warmup; ++i) {
      const PipelineResult warmup =
          run_gpu(records, config, options, initial_fallback_flags);
      if (warmup.fallback_flags != kFallbackNone) break;
    }

    PipelineResult gpu;
    double setup_sum = 0.0;
    double h2d_sum = 0.0;
    double kernel_sum = 0.0;
    double sort_sum = 0.0;
    double d2h_sum = 0.0;
    double total_sum = 0.0;
    for (std::uint32_t i = 0; i < options.repeat; ++i) {
      PipelineResult current =
          run_gpu(records, config, options, initial_fallback_flags);
      setup_sum += current.setup_ms;
      h2d_sum += current.h2d_ms;
      kernel_sum += current.kernel_ms;
      sort_sum += current.sort_dedup_ms;
      d2h_sum += current.d2h_ms;
      total_sum += current.total_ms;
      if (i == 0) {
        gpu = std::move(current);
      } else if (current.fallback_flags != gpu.fallback_flags ||
                 current.memberships != gpu.memberships ||
                 current.occupied_cells != gpu.occupied_cells ||
                 current.pair_work != gpu.pair_work ||
                 current.raw_candidates != gpu.raw_candidates ||
                 current.pairs != gpu.pairs) {
        throw std::runtime_error("measured repetitions were not deterministic");
      }
    }
    const double repetitions = static_cast<double>(options.repeat);
    gpu.setup_ms = setup_sum / repetitions;
    gpu.h2d_ms = h2d_sum / repetitions;
    gpu.kernel_ms = kernel_sum / repetitions;
    gpu.sort_dedup_ms = sort_sum / repetitions;
    gpu.d2h_ms = d2h_sum / repetitions;
    gpu.total_ms = total_sum / repetitions;

    const bool fallback_required = gpu.fallback_flags != kFallbackNone;
    std::vector<std::uint64_t> reference_pairs;
    double reference_ms = 0.0;
    bool reference_available = false;
    // The grid oracle shares coordinate-expansion arithmetic with the GPU
    // path, so it must not run after any fail-closed signal.  The explicitly
    // requested exhaustive oracle remains safe and useful for small fixtures.
    if (options.reference != "none" &&
        (!fallback_required || options.reference == "exhaustive")) {
      const auto reference_begin = Clock::now();
      reference_pairs = options.reference == "exhaustive"
                            ? cpu_exhaustive_reference(records, config)
                            : cpu_grid_reference(records, config);
      reference_ms = elapsed_ms(reference_begin, Clock::now());
      reference_available = true;
    }

    const bool compared = reference_available && !fallback_required;
    const bool equal = compared && gpu.pairs == reference_pairs;
    if (!fallback_required && options.fixture == "boundary") {
      const std::size_t expected = options.enlargement == 0 ? 1 : 2;
      if (gpu.pairs.size() != expected) {
        throw std::runtime_error("strict-enlargement boundary fixture failed");
      }
    }
    const std::vector<std::uint64_t> &authoritative_pairs =
        fallback_required && reference_available ? reference_pairs : gpu.pairs;
    if (!options.output_pairs_path.empty()) {
      if (fallback_required && !reference_available) {
        throw std::runtime_error(
            "cannot write incomplete GPU pairs after a fallback signal");
      }
      write_pairs(options.output_pairs_path, authoritative_pairs);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "device=\"" << property.name << "\" cc=" << property.major << '.'
              << property.minor << " sm_count=" << property.multiProcessorCount
              << " global_memory_bytes=" << property.totalGlobalMem << '\n';
    std::cout << "records=" << records.size()
              << " replay_record_bytes=" << sizeof(InputRecord)
              << " device_record_bytes=" << sizeof(PackedAabb)
              << " contexts_requested=" << options.context_count
              << " mode=" << options.mode << " geometry=" << options.geometry
              << " enlargement=" << options.enlargement
              << " cell_size=" << options.cell_size << '\n';
    std::cout << "memberships=" << gpu.memberships
              << " occupied_cells=" << gpu.occupied_cells
              << " pair_work=" << gpu.pair_work
              << " raw_candidates=" << gpu.raw_candidates
              << " unique_gpu_pairs=" << gpu.pairs.size() << '\n';
    std::cout << "fallback_required=" << (fallback_required ? "true" : "false")
              << " fallback_flags=\"" << fallback_text(gpu.fallback_flags) << "\"\n";
    std::cout << "timing_ms input=" << elapsed_ms(input_begin, input_end)
              << " pack=" << elapsed_ms(pack_begin, pack_end)
              << " setup=" << gpu.setup_ms << " h2d=" << gpu.h2d_ms
              << " kernel=" << gpu.kernel_ms
              << " sort_dedup=" << gpu.sort_dedup_ms << " d2h=" << gpu.d2h_ms
              << " gpu_pipeline_total=" << gpu.total_ms
              << " host_to_host_candidate_generation="
              << elapsed_ms(pack_begin, pack_end) + gpu.total_ms
              << " cpu_reference=" << reference_ms
              << " warmup=" << options.warmup << " repeat=" << options.repeat
              << " reported_gpu_timings=per-repeat-mean\n";
    std::cout << "enumeration_launch_blocks=" << gpu.enumeration_blocks
              << " enumeration_theoretical_occupancy_pct="
              << gpu.enumeration_theoretical_occupancy_pct
              << " achieved_gpu_utilization_pct=external-profiler-required\n";
    std::cout << std::hex << std::setfill('0');
    std::cout << "gpu_pair_hash=0x" << std::setw(16) << pair_hash(gpu.pairs);
    if (reference_available) {
      std::cout << " reference_pairs=" << std::dec << reference_pairs.size() << std::hex
                << " reference_pair_hash=0x" << std::setw(16)
                << pair_hash(reference_pairs);
    }
    std::cout << std::dec << '\n';
    if (fallback_required) {
      std::cout << "comparison=FALLBACK cpu_reference_available="
                << (reference_available ? "true" : "false") << '\n';
      return 3;
    }
    if (options.reference == "none") {
      std::cout << "comparison=SKIPPED\n";
      return 0;
    }
    std::cout << "comparison=" << (equal ? "PASS" : "FAIL") << '\n';
    return equal ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
