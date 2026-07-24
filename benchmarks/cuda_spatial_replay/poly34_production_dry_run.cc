/*
 * Read-only production census for the conservative POLY.3/POLY.4
 * terminal-empty certificate.
 *
 * This program does not alter KLayout's host ABI or DRC execution.  It loads a
 * layout through KLayout's database library, builds the exact flat merged
 * POLY, ACTIVE, and GATE regions, and answers the bounded-volume questions
 * needed before a live transaction is attempted:
 *
 *   - how many canonical primary rectangles are in each gate's complete
 *     projection candidate window;
 *   - whether the 64-rectangle certificate capacity is sufficient;
 *   - how often each conservative profile and their atomic conjunction can
 *     certify terminal empty; and
 *   - how many exact KLayout edge pairs disappear only because the deck
 *     normalizes them to zero-area polygons and applies without_area(0).
 */

#include "poly34_terminal_empty_certificate.cuh"

#include "dbBoxScanner.h"
#include "dbEdgePairs.h"
#include "dbGDS2Reader.h"
#include "dbLayout.h"
#include "dbLoadLayoutOptions.h"
#include "dbPolygonGenerators.h"
#include "dbPolygonTools.h"
#include "dbRecursiveShapeIterator.h"
#include "dbRegion.h"
#include "dbRegionLocalOperations.h"
#include "tlStream.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace poly34 = klayout_cuda::poly34;
using Clock = std::chrono::steady_clock;

class DryRunError : public std::runtime_error {
 public:
  explicit DryRunError(const std::string &message)
      : std::runtime_error(message) {}
};

struct Options {
  std::string input;
  std::string top;
  std::uint32_t poly_layer = 9;
  std::uint32_t poly_datatype = 0;
  std::uint32_t active_layer = 1;
  std::uint32_t active_datatype = 0;
  std::uint32_t maximum_candidates = poly34::kMaximumCandidateBoxes;
  bool skip_exact_terminal = false;
};

struct RegionBoxes {
  std::vector<db::Box> boxes;
  std::uint64_t merged_polygons = 0;
  std::uint64_t decomposed_polygons = 0;
  std::uint64_t non_rectilinear = 0;
  std::uint64_t polygons_with_holes = 0;
  std::uint64_t nonbox_trapezoids = 0;
};

struct WindowCensus {
  std::vector<std::uint32_t> counts;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint32_t> candidate_ids;
  std::uint64_t total_candidates = 0;
  std::uint64_t capacity_gates = 0;
  std::uint32_t maximum = 0;
};

struct CertificateCensus {
  std::vector<std::uint8_t> outcomes;
  std::uint64_t terminal_empty = 0;
  std::uint64_t fallback = 0;
  std::uint64_t unsupported = 0;
};

struct TerminalCensus {
  std::uint64_t raw_edge_pairs = 0;
  std::uint64_t zero_area_polygons = 0;
  std::uint64_t nonzero_area_polygons = 0;
  double seconds = 0.0;
};

double elapsed_seconds(const Clock::time_point &begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::uint32_t parse_u32(const std::string &text, const char *what) {
  if (text.empty() || text[0] == '-') {
    throw DryRunError(std::string("invalid ") + what + ": " + text);
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno || !end || *end ||
      value > std::numeric_limits<std::uint32_t>::max()) {
    throw DryRunError(std::string("invalid ") + what + ": " + text);
  }
  return static_cast<std::uint32_t>(value);
}

bool take_value(const std::string &argument, const char *prefix,
                std::string *value) {
  const std::size_t length = std::strlen(prefix);
  if (argument.compare(0, length, prefix) != 0) {
    return false;
  }
  *value = argument.substr(length);
  return true;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    std::string value;
    if (take_value(argument, "--input=", &value)) {
      options.input = value;
    } else if (take_value(argument, "--top=", &value)) {
      options.top = value;
    } else if (take_value(argument, "--poly-layer=", &value)) {
      options.poly_layer = parse_u32(value, "POLY layer");
    } else if (take_value(argument, "--poly-datatype=", &value)) {
      options.poly_datatype = parse_u32(value, "POLY datatype");
    } else if (take_value(argument, "--active-layer=", &value)) {
      options.active_layer = parse_u32(value, "ACTIVE layer");
    } else if (take_value(argument, "--active-datatype=", &value)) {
      options.active_datatype = parse_u32(value, "ACTIVE datatype");
    } else if (take_value(argument, "--maximum-candidates=", &value)) {
      options.maximum_candidates = parse_u32(value, "maximum candidates");
    } else if (argument == "--skip-exact-terminal") {
      options.skip_exact_terminal = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "usage: poly34_production_dry_run --input=LAYOUT.gds --top=TOP "
          << "[--poly-layer=9] [--poly-datatype=0] "
          << "[--active-layer=1] [--active-datatype=0] "
          << "[--maximum-candidates=64] [--skip-exact-terminal]\n";
      std::exit(0);
    } else {
      throw DryRunError("unknown argument: " + argument);
    }
  }
  if (options.input.empty() || options.top.empty()) {
    throw DryRunError("--input and --top are required");
  }
  if (!options.maximum_candidates ||
      options.maximum_candidates > poly34::kMaximumCandidateBoxes) {
    throw DryRunError(
        "--maximum-candidates must be in the qualified range 1..64");
  }
  return options;
}

unsigned int find_layer(const db::Layout &layout, std::uint32_t layer,
                        std::uint32_t datatype, const char *name) {
  bool found = false;
  unsigned int result = 0;
  for (db::Layout::layer_iterator candidate = layout.begin_layers();
       candidate != layout.end_layers(); ++candidate) {
    if ((*candidate).second->layer == static_cast<int>(layer) &&
        (*candidate).second->datatype == static_cast<int>(datatype)) {
      if (found) {
        throw DryRunError(std::string(name) + " layer is ambiguous");
      }
      found = true;
      result = (*candidate).first;
    }
  }
  if (!found) {
    throw DryRunError(std::string(name) + " layer is absent");
  }
  return result;
}

db::Region load_region(const db::Layout &layout, db::cell_index_type top,
                       unsigned int layer) {
  const db::RecursiveShapeIterator iterator(
      layout, layout.cell(top), layer);
  db::Region raw(iterator);
  return raw.merged();
}

class BoxSink : public db::SimplePolygonSink {
 public:
  explicit BoxSink(RegionBoxes *result) : result_(result) {}

  void put(const db::SimplePolygon &polygon) override {
    ++result_->decomposed_polygons;
    if (!polygon.is_box()) {
      ++result_->nonbox_trapezoids;
      return;
    }
    const db::Box box = polygon.box();
    if (box.empty()) {
      throw DryRunError("trapezoid decomposition emitted an empty box");
    }
    result_->boxes.push_back(box);
  }

 private:
  RegionBoxes *result_;
};

RegionBoxes decompose_region(const db::Region &region) {
  RegionBoxes result;
  BoxSink sink(&result);
  for (db::Region::const_iterator polygon = region.begin_merged();
       !polygon.at_end(); ++polygon) {
    ++result.merged_polygons;
    result.polygons_with_holes += polygon->holes() != 0;
    if (!polygon->is_rectilinear()) {
      ++result.non_rectilinear;
      continue;
    }
    db::decompose_trapezoids(*polygon, db::TD_simple, sink);
  }
  return result;
}

poly34::Box certificate_box(const db::Box &box) {
  return {box.left(), box.bottom(), box.right(), box.top()};
}

class CountReceiver
    : public db::box_scanner_receiver2<db::Box, std::uint32_t, db::Box,
                                      std::uint32_t> {
 public:
  explicit CountReceiver(std::vector<std::uint32_t> *counts)
      : counts_(counts) {}

  void add(const db::Box *, const std::uint32_t &gate_id, const db::Box *,
           const std::uint32_t &) {
    std::uint32_t &count = (*counts_)[gate_id];
    if (count == std::numeric_limits<std::uint32_t>::max()) {
      throw DryRunError("per-gate candidate count overflow");
    }
    ++count;
  }

 private:
  std::vector<std::uint32_t> *counts_;
};

class FillReceiver
    : public db::box_scanner_receiver2<db::Box, std::uint32_t, db::Box,
                                      std::uint32_t> {
 public:
  FillReceiver(const std::vector<std::uint64_t> *offsets,
               std::vector<std::uint64_t> *cursors,
               std::vector<std::uint32_t> *candidate_ids)
      : offsets_(offsets), cursors_(cursors), candidate_ids_(candidate_ids) {}

  void add(const db::Box *, const std::uint32_t &gate_id, const db::Box *,
           const std::uint32_t &primary_id) {
    std::uint64_t &cursor = (*cursors_)[gate_id];
    const std::uint64_t end = (*offsets_)[gate_id + 1];
    if (cursor < end) {
      (*candidate_ids_)[cursor++] = primary_id;
    }
  }

 private:
  const std::vector<std::uint64_t> *offsets_;
  std::vector<std::uint64_t> *cursors_;
  std::vector<std::uint32_t> *candidate_ids_;
};

template <class Receiver>
void scan_windows(const std::vector<db::Box> &gates,
                  const std::vector<db::Box> &primary, db::Coord distance,
                  Receiver *receiver) {
  if (gates.size() > std::numeric_limits<std::uint32_t>::max() ||
      primary.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw DryRunError("box census exceeds uint32 scanner identity capacity");
  }
  db::box_scanner2<db::Box, std::uint32_t, db::Box, std::uint32_t> scanner;
  scanner.reserve1(gates.size());
  scanner.reserve2(primary.size());
  for (std::uint32_t i = 0; i < gates.size(); ++i) {
    scanner.insert1(&gates[i], i);
  }
  for (std::uint32_t i = 0; i < primary.size(); ++i) {
    scanner.insert2(&primary[i], i);
  }
  scanner.process(*receiver, distance, db::box_convert<db::Box>(),
                  db::box_convert<db::Box>());
}

WindowCensus build_windows(const std::vector<db::Box> &gates,
                           const std::vector<db::Box> &primary,
                           db::Coord distance,
                           std::uint32_t maximum_candidates) {
  if (distance <= 0) {
    throw DryRunError("candidate-window distance must be positive");
  }
  const db::Coord lower =
      std::numeric_limits<db::Coord>::min() + distance;
  const db::Coord upper =
      std::numeric_limits<db::Coord>::max() - distance;
  const auto coordinates_are_safe =
      [lower, upper](const std::vector<db::Box> &boxes) {
        return std::all_of(
            boxes.begin(), boxes.end(),
            [lower, upper](const db::Box &box) {
              return box.left() >= lower && box.bottom() >= lower &&
                     box.right() <= upper && box.top() <= upper;
            });
      };
  // db::box_scanner adds the positive enlargement to box endpoints in
  // db::Coord. Reject both coordinate extremes conservatively so neither the
  // broad scan nor a future symmetric implementation can wrap.
  if (!coordinates_are_safe(gates) || !coordinates_are_safe(primary)) {
    throw DryRunError(
        "candidate-window coordinate extent is unsafe for scanner arithmetic");
  }

  WindowCensus result;
  result.counts.resize(gates.size(), 0);
  CountReceiver counter(&result.counts);
  scan_windows(gates, primary, distance, &counter);

  result.offsets.resize(gates.size() + 1, 0);
  for (std::size_t i = 0; i < gates.size(); ++i) {
    const std::uint32_t count = result.counts[i];
    result.total_candidates += count;
    result.maximum = std::max(result.maximum, count);
    result.capacity_gates += count > maximum_candidates;
    const std::uint64_t retained =
        std::min<std::uint32_t>(count, maximum_candidates);
    if (result.offsets[i] >
        std::numeric_limits<std::uint64_t>::max() - retained) {
      throw DryRunError("retained candidate offset overflow");
    }
    result.offsets[i + 1] = result.offsets[i] + retained;
  }
  if (result.offsets.back() >
      std::uint64_t(std::numeric_limits<std::size_t>::max())) {
    throw DryRunError("retained candidate storage exceeds size_t");
  }
  result.candidate_ids.resize(
      static_cast<std::size_t>(result.offsets.back()));
  std::vector<std::uint64_t> cursors(result.offsets.begin(),
                                     result.offsets.end() - 1);
  FillReceiver filler(&result.offsets, &cursors, &result.candidate_ids);
  scan_windows(gates, primary, distance, &filler);
  for (std::size_t i = 0; i < gates.size(); ++i) {
    if (cursors[i] != result.offsets[i + 1]) {
      throw DryRunError("candidate fill census disagrees with count pass");
    }
  }
  return result;
}

std::uint64_t percentile(const std::vector<std::uint32_t> &sorted,
                         std::uint64_t numerator,
                         std::uint64_t denominator) {
  if (sorted.empty() || !denominator || numerator > denominator) {
    throw DryRunError("invalid percentile request");
  }
  const std::uint64_t rank =
      (std::uint64_t(sorted.size()) * numerator + denominator - 1) /
      denominator;
  return sorted[std::max<std::uint64_t>(1, rank) - 1];
}

void print_window_census(const char *profile, const WindowCensus &census,
                         std::uint32_t maximum_candidates, double seconds) {
  std::array<std::uint64_t, 10> histogram = {};
  std::vector<std::uint32_t> sorted = census.counts;
  for (const std::uint32_t count : sorted) {
    std::size_t bucket = 9;
    if (count == 0) bucket = 0;
    else if (count == 1) bucket = 1;
    else if (count == 2) bucket = 2;
    else if (count == 3) bucket = 3;
    else if (count == 4) bucket = 4;
    else if (count <= 8) bucket = 5;
    else if (count <= 16) bucket = 6;
    else if (count <= 32) bucket = 7;
    else if (count <= 64) bucket = 8;
    ++histogram[bucket];
  }
  std::sort(sorted.begin(), sorted.end());
  std::cout << "POLY34_CANDIDATE_WINDOWS"
            << " profile=" << profile
            << " gates=" << census.counts.size()
            << " raw_candidates=" << census.total_candidates
            << " maximum=" << census.maximum
            << " p50=" << percentile(sorted, 50, 100)
            << " p90=" << percentile(sorted, 90, 100)
            << " p99=" << percentile(sorted, 99, 100)
            << " p999=" << percentile(sorted, 999, 1000)
            << " capacity=" << maximum_candidates
            << " capacity_gates=" << census.capacity_gates
            << " seconds=" << seconds << "\n";
  std::cout << "POLY34_CANDIDATE_HISTOGRAM"
            << " profile=" << profile
            << " n0=" << histogram[0]
            << " n1=" << histogram[1]
            << " n2=" << histogram[2]
            << " n3=" << histogram[3]
            << " n4=" << histogram[4]
            << " n5_8=" << histogram[5]
            << " n9_16=" << histogram[6]
            << " n17_32=" << histogram[7]
            << " n33_64=" << histogram[8]
            << " n65_plus=" << histogram[9] << "\n";
}

CertificateCensus classify_profile(
    const std::vector<db::Box> &gates,
    const std::vector<db::Box> &primary,
    const WindowCensus &windows,
    std::int64_t distance,
    std::uint32_t maximum_candidates) {
  CertificateCensus result;
  result.outcomes.resize(gates.size());

  for (std::size_t gate_id = 0; gate_id < gates.size(); ++gate_id) {
    const std::uint32_t count = windows.counts[gate_id];
    const std::uint64_t begin = windows.offsets[gate_id];
    const std::uint64_t end = windows.offsets[gate_id + 1];
    if (end < begin ||
        end - begin != std::min(count, maximum_candidates) ||
        end > windows.candidate_ids.size()) {
      throw DryRunError("retained candidate range is inconsistent");
    }
    for (std::uint64_t i = begin; i < end; ++i) {
      if (windows.candidate_ids[i] >= primary.size()) {
        throw DryRunError("candidate primary identity is out of range");
      }
    }
  }

#pragma omp parallel
  {
    std::array<poly34::Box, poly34::kMaximumCandidateBoxes> candidates;
    std::uint64_t local_terminal = 0;
    std::uint64_t local_fallback = 0;
    std::uint64_t local_unsupported = 0;

#pragma omp for schedule(static)
    for (std::int64_t gate_id = 0;
         gate_id < static_cast<std::int64_t>(gates.size()); ++gate_id) {
      poly34::Certificate certificate = poly34::Certificate::kUnsupported;
      const std::uint32_t count = windows.counts[gate_id];
      if (count <= maximum_candidates) {
        const std::uint64_t begin = windows.offsets[gate_id];
        const std::uint64_t end = windows.offsets[gate_id + 1];
        for (std::uint64_t i = begin; i < end; ++i) {
          const std::uint32_t primary_id = windows.candidate_ids[i];
          candidates[i - begin] = certificate_box(primary[primary_id]);
        }
        certificate = poly34::terminal_empty_profile(
            certificate_box(gates[gate_id]), candidates.data(), count,
            distance, true);
      }
      result.outcomes[gate_id] = static_cast<std::uint8_t>(certificate);
      if (certificate == poly34::Certificate::kTerminalEmpty) {
        ++local_terminal;
      } else if (certificate == poly34::Certificate::kFallback) {
        ++local_fallback;
      } else {
        ++local_unsupported;
      }
    }

#pragma omp atomic
    result.terminal_empty += local_terminal;
#pragma omp atomic
    result.fallback += local_fallback;
#pragma omp atomic
    result.unsupported += local_unsupported;
  }
  return result;
}

TerminalCensus exact_terminal_census(const db::Region &primary,
                                     const db::Region &gate,
                                     db::Coord distance) {
  const Clock::time_point begin = Clock::now();
  const db::RegionCheckOptions options(false, db::Projection);
  const db::EdgePairs pairs =
      primary.enclosing_check(gate, distance, options);
  TerminalCensus result;
  for (db::EdgePairs::const_iterator pair = pairs.begin();
       !pair.at_end(); ++pair) {
    ++result.raw_edge_pairs;
    const db::Polygon marker = pair->normalized().to_polygon(0);
    if (marker.area() == 0) {
      ++result.zero_area_polygons;
    } else {
      ++result.nonzero_area_polygons;
    }
  }
  result.seconds = elapsed_seconds(begin);
  if (result.raw_edge_pairs !=
      result.zero_area_polygons + result.nonzero_area_polygons) {
    throw DryRunError("exact terminal census arithmetic is inconsistent");
  }
  return result;
}

std::uint64_t checked_add(std::uint64_t first, std::uint64_t second,
                          const char *what) {
  if (first > std::numeric_limits<std::uint64_t>::max() - second) {
    throw DryRunError(std::string(what) + " byte estimate overflow");
  }
  return first + second;
}

std::uint64_t checked_multiply(std::uint64_t first, std::uint64_t second,
                               const char *what) {
  if (first && second > std::numeric_limits<std::uint64_t>::max() / first) {
    throw DryRunError(std::string(what) + " byte estimate overflow");
  }
  return first * second;
}

void print_memory_census(std::uint64_t gates, std::uint64_t poly_boxes,
                         std::uint64_t active_boxes,
                         const WindowCensus &poly_windows,
                         const WindowCensus &active_windows) {
  const std::uint64_t offset_records =
      checked_multiply(2, checked_add(gates, 1, "candidate offsets"),
                       "candidate offsets");
  const std::uint64_t primary_records =
      checked_add(poly_boxes, active_boxes, "primary boxes");
  const std::uint64_t candidate_records =
      checked_add(poly_windows.total_candidates,
                  active_windows.total_candidates, "candidate identities");
  const std::uint64_t result_records =
      checked_multiply(3, gates, "results");
  std::uint64_t bytes = 0;
  bytes = checked_add(
      bytes, checked_multiply(gates, sizeof(poly34::Box), "gate"),
      "device payload");
  bytes = checked_add(
      bytes, checked_multiply(poly_boxes, sizeof(poly34::Box), "POLY box"),
      "device payload");
  bytes = checked_add(
      bytes, checked_multiply(active_boxes, sizeof(poly34::Box), "ACTIVE box"),
      "device payload");
  bytes = checked_add(
      bytes, checked_multiply(offset_records, sizeof(std::uint64_t),
                              "candidate offsets"),
      "device payload");
  bytes = checked_add(
      bytes,
      checked_multiply(candidate_records,
                       sizeof(std::uint32_t), "candidate identities"),
      "device payload");
  bytes = checked_add(
      bytes,
      checked_multiply(result_records, sizeof(std::uint8_t), "results"),
      "device payload");
  std::cout << "POLY34_MEMORY_ESTIMATE"
            << " gate_box_bytes="
            << checked_multiply(gates, sizeof(poly34::Box), "gate")
            << " primary_box_bytes="
            << checked_multiply(primary_records,
                                sizeof(poly34::Box), "primary")
            << " candidate_offset_bytes="
            << checked_multiply(offset_records, sizeof(std::uint64_t),
                                "candidate offsets")
            << " candidate_identity_bytes="
            << checked_multiply(candidate_records,
                                sizeof(std::uint32_t),
                                "candidate identities")
            << " result_bytes="
            << checked_multiply(result_records, sizeof(std::uint8_t),
                                "results")
            << " minimum_device_payload_bytes=" << bytes << "\n";
}

int run(const Options &options) {
  const Clock::time_point total_begin = Clock::now();
  const Clock::time_point load_begin = Clock::now();
  db::Layout layout;
  {
    tl::InputStream input(options.input);
    db::GDS2Reader reader(input);
    reader.read(layout, db::LoadLayoutOptions());
  }
  const std::pair<bool, db::cell_index_type> top =
      layout.cell_by_name(options.top.c_str());
  if (!top.first) {
    throw DryRunError("top cell is absent: " + options.top);
  }
  const unsigned int poly_layer =
      find_layer(layout, options.poly_layer, options.poly_datatype, "POLY");
  const unsigned int active_layer =
      find_layer(layout, options.active_layer, options.active_datatype,
                 "ACTIVE");
  constexpr double kQualifiedDbu = 0.0005;
  if (layout.dbu() != kQualifiedDbu) {
    throw DryRunError(
        "layout DBU differs from qualified 0.0005 micron profile");
  }
  std::cout << std::fixed << std::setprecision(6)
            << "POLY34_LOAD"
            << " cells=" << layout.cells()
            << " dbu=" << layout.dbu()
            << " seconds=" << elapsed_seconds(load_begin) << "\n";

  const Clock::time_point region_begin = Clock::now();
  const db::Region poly = load_region(layout, top.second, poly_layer);
  const db::Region active = load_region(layout, top.second, active_layer);
  const db::Region gate = (poly & active).merged();
  std::cout << "POLY34_REGIONS seconds=" << elapsed_seconds(region_begin)
            << "\n";

  const Clock::time_point decompose_begin = Clock::now();
  const RegionBoxes poly_boxes = decompose_region(poly);
  const RegionBoxes active_boxes = decompose_region(active);
  const RegionBoxes gate_boxes = decompose_region(gate);
  const bool geometry_supported =
      !poly_boxes.non_rectilinear && !poly_boxes.nonbox_trapezoids &&
      !active_boxes.non_rectilinear && !active_boxes.nonbox_trapezoids &&
      !gate_boxes.non_rectilinear && !gate_boxes.nonbox_trapezoids &&
      gate_boxes.merged_polygons == gate_boxes.boxes.size();
  std::cout << "POLY34_CANONICAL_BOXES"
            << " layer=poly"
            << " merged_polygons=" << poly_boxes.merged_polygons
            << " rectangles=" << poly_boxes.boxes.size()
            << " holes=" << poly_boxes.polygons_with_holes
            << " non_rectilinear=" << poly_boxes.non_rectilinear
            << " nonbox_trapezoids=" << poly_boxes.nonbox_trapezoids << "\n";
  std::cout << "POLY34_CANONICAL_BOXES"
            << " layer=active"
            << " merged_polygons=" << active_boxes.merged_polygons
            << " rectangles=" << active_boxes.boxes.size()
            << " holes=" << active_boxes.polygons_with_holes
            << " non_rectilinear=" << active_boxes.non_rectilinear
            << " nonbox_trapezoids=" << active_boxes.nonbox_trapezoids << "\n";
  std::cout << "POLY34_CANONICAL_BOXES"
            << " layer=gate"
            << " merged_polygons=" << gate_boxes.merged_polygons
            << " rectangles=" << gate_boxes.boxes.size()
            << " holes=" << gate_boxes.polygons_with_holes
            << " non_rectilinear=" << gate_boxes.non_rectilinear
            << " nonbox_trapezoids=" << gate_boxes.nonbox_trapezoids
            << " seconds=" << elapsed_seconds(decompose_begin) << "\n";
  if (!geometry_supported || gate_boxes.boxes.empty()) {
    throw DryRunError("canonical production geometry is unsupported");
  }

  const Clock::time_point poly_window_begin = Clock::now();
  const WindowCensus poly_windows =
      build_windows(gate_boxes.boxes, poly_boxes.boxes,
                    poly34::kPoly3Distance, options.maximum_candidates);
  print_window_census("poly3", poly_windows, options.maximum_candidates,
                      elapsed_seconds(poly_window_begin));
  const Clock::time_point poly_classify_begin = Clock::now();
  const CertificateCensus poly_certificates =
      classify_profile(gate_boxes.boxes, poly_boxes.boxes, poly_windows,
                       poly34::kPoly3Distance, options.maximum_candidates);
  std::cout << "POLY34_CERTIFICATE_COVERAGE"
            << " profile=poly3"
            << " terminal_empty=" << poly_certificates.terminal_empty
            << " fallback=" << poly_certificates.fallback
            << " unsupported=" << poly_certificates.unsupported
            << " seconds=" << elapsed_seconds(poly_classify_begin) << "\n";

  const Clock::time_point active_window_begin = Clock::now();
  const WindowCensus active_windows =
      build_windows(gate_boxes.boxes, active_boxes.boxes,
                    poly34::kPoly4Distance, options.maximum_candidates);
  print_window_census("poly4", active_windows, options.maximum_candidates,
                      elapsed_seconds(active_window_begin));
  const Clock::time_point active_classify_begin = Clock::now();
  const CertificateCensus active_certificates =
      classify_profile(gate_boxes.boxes, active_boxes.boxes, active_windows,
                       poly34::kPoly4Distance, options.maximum_candidates);
  std::cout << "POLY34_CERTIFICATE_COVERAGE"
            << " profile=poly4"
            << " terminal_empty=" << active_certificates.terminal_empty
            << " fallback=" << active_certificates.fallback
            << " unsupported=" << active_certificates.unsupported
            << " seconds=" << elapsed_seconds(active_classify_begin) << "\n";

  std::uint64_t atomic_terminal_empty = 0;
  for (std::size_t i = 0; i < gate_boxes.boxes.size(); ++i) {
    atomic_terminal_empty +=
        poly_certificates.outcomes[i] ==
            static_cast<std::uint8_t>(poly34::Certificate::kTerminalEmpty) &&
        active_certificates.outcomes[i] ==
            static_cast<std::uint8_t>(poly34::Certificate::kTerminalEmpty);
  }
  std::cout << "POLY34_ATOMIC_COVERAGE"
            << " gates=" << gate_boxes.boxes.size()
            << " terminal_empty=" << atomic_terminal_empty
            << " fallback_or_unsupported="
            << gate_boxes.boxes.size() - atomic_terminal_empty << "\n";
  print_memory_census(gate_boxes.boxes.size(), poly_boxes.boxes.size(),
                      active_boxes.boxes.size(), poly_windows, active_windows);

  TerminalCensus poly_terminal;
  TerminalCensus active_terminal;
  if (!options.skip_exact_terminal) {
    poly_terminal =
        exact_terminal_census(poly, gate, poly34::kPoly3Distance);
    std::cout << "POLY34_EXACT_TERMINAL"
              << " profile=poly3"
              << " raw_edge_pairs=" << poly_terminal.raw_edge_pairs
              << " zero_area_polygons=" << poly_terminal.zero_area_polygons
              << " nonzero_area_polygons="
              << poly_terminal.nonzero_area_polygons
              << " seconds=" << poly_terminal.seconds << "\n";
    active_terminal =
        exact_terminal_census(active, gate, poly34::kPoly4Distance);
    std::cout << "POLY34_EXACT_TERMINAL"
              << " profile=poly4"
              << " raw_edge_pairs=" << active_terminal.raw_edge_pairs
              << " zero_area_polygons=" << active_terminal.zero_area_polygons
              << " nonzero_area_polygons="
              << active_terminal.nonzero_area_polygons
              << " seconds=" << active_terminal.seconds << "\n";
  }

  const bool capacity_ok =
      !poly_windows.capacity_gates && !active_windows.capacity_gates;
  const bool complete_coverage =
      atomic_terminal_empty == gate_boxes.boxes.size();
  const bool exact_terminal_empty =
      options.skip_exact_terminal ||
      (!poly_terminal.nonzero_area_polygons &&
       !active_terminal.nonzero_area_polygons);
  const bool go =
      geometry_supported && capacity_ok && complete_coverage &&
      exact_terminal_empty;
  std::cout << "POLY34_PRODUCTION_DRY_RUN"
            << " verdict=" << (go ? "GO" : "NO_GO")
            << " geometry_supported=" << geometry_supported
            << " capacity_ok=" << capacity_ok
            << " complete_atomic_coverage=" << complete_coverage
            << " exact_terminal_empty=" << exact_terminal_empty
            << " exact_terminal_skipped=" << options.skip_exact_terminal
            << " total_seconds=" << elapsed_seconds(total_begin) << "\n";
  return go ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    return run(parse_options(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "POLY34_PRODUCTION_DRY_RUN verdict=NO_GO error=\""
              << error.what() << "\"\n";
    return 2;
  }
}
