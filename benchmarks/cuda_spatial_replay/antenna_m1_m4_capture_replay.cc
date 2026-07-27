/*
 * Standalone loader/replayer and small corruption-oriented self-test for the
 * portable ANTENNA.M1-through-M4 capture file.
 */

#include "antenna_m1_m4_capture_file.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#elif !defined(KLAYOUT_ANTENNA_M1_M4_REPLAY_DIRECT)
#  include <dlfcn.h>
#endif

namespace {

namespace capture = klayout_cuda::antenna_m1_m4_capture;
using Request = capture::Request;
using Result = klayout_cuda_spatial_antenna_m1_m4_result_v1;
using RunFunction =
    klayout_cuda_spatial_run_antenna_m1_m4_empty_v1_func;

constexpr std::size_t kDomainCount =
    KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
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

struct DynamicBackend
{
  RunFunction run = nullptr;
  std::string path;
#if defined(_WIN32)
  HMODULE handle = nullptr;
#elif !defined(KLAYOUT_ANTENNA_M1_M4_REPLAY_DIRECT)
  void *handle = nullptr;
#endif

  ~DynamicBackend ()
  {
#if defined(_WIN32)
    if (handle) FreeLibrary(handle);
#elif !defined(KLAYOUT_ANTENNA_M1_M4_REPLAY_DIRECT)
    if (handle) dlclose(handle);
#endif
  }

  DynamicBackend () = default;
  DynamicBackend (const DynamicBackend &) = delete;
  DynamicBackend &operator= (const DynamicBackend &) = delete;
};

struct ReplayOverrides
{
  bool has_device = false;
  std::int32_t device = 0;
  bool has_max_device_bytes = false;
  std::uint64_t max_device_bytes = 0;
  bool has_max_estimated_peak_bytes = false;
  std::uint64_t max_estimated_peak_bytes = 0;

  bool empty () const
  {
    return !has_device && !has_max_device_bytes &&
           !has_max_estimated_peak_bytes;
  }
};

bool parse_u64 (const char *text, std::uint64_t &value)
{
  if (!text || !*text || *text == '-') return false;
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 0);
  if (errno || !end || *end ||
      parsed > static_cast<unsigned long long>(UINT64_MAX)) {
    return false;
  }
  value = static_cast<std::uint64_t>(parsed);
  return true;
}

bool apply_overrides (const ReplayOverrides &overrides,
                      capture::OwnedRequest &owned, std::string &error)
{
  if (overrides.has_device) {
    owned.request.device = overrides.device;
  }
  if (overrides.has_max_device_bytes) {
    if (!overrides.max_device_bytes) {
      error = "--max-device-bytes must be nonzero";
      return false;
    }
    owned.request.capacity.max_device_bytes =
        overrides.max_device_bytes;
  }
  if (overrides.has_max_estimated_peak_bytes) {
    if (overrides.max_estimated_peak_bytes <
        owned.request.census.estimated_peak_bytes) {
      std::ostringstream message;
      message
          << "--max-estimated-peak-bytes is below the captured census ("
          << owned.request.census.estimated_peak_bytes << ")";
      error = message.str();
      return false;
    }
    owned.request.capacity.max_estimated_peak_bytes =
        overrides.max_estimated_peak_bytes;
  }
  error.clear();
  return true;
}

std::string default_backend_path ()
{
  const char *setting = std::getenv("KLAYOUT_CUDA_SPATIAL_BACKEND");
  if (setting && *setting && std::strcmp(setting, "0") != 0 &&
      std::strcmp(setting, "false") != 0 &&
      std::strcmp(setting, "off") != 0) {
    if (std::strcmp(setting, "1") != 0 &&
        std::strcmp(setting, "auto") != 0) {
      return setting;
    }
  }
#if defined(_WIN32)
  return "klayout_cuda_spatial_backend.dll";
#elif defined(__APPLE__)
  return "libklayout_cuda_spatial_backend.dylib";
#else
  return "libklayout_cuda_spatial_backend.so";
#endif
}

bool load_backend (const std::string &path, DynamicBackend &backend,
                   std::string &error)
{
  backend.path = path;
#if defined(KLAYOUT_ANTENNA_M1_M4_REPLAY_DIRECT)
  (void)path;
  backend.run = &klayout_cuda_spatial_run_antenna_m1_m4_empty_v1;
  error.clear();
  return true;
#else
  klayout_cuda_spatial_abi_version_func version = nullptr;
#  if defined(_WIN32)
  backend.handle = LoadLibraryA(path.c_str());
  if (!backend.handle) {
    error = "LoadLibrary failed for " + path;
    return false;
  }
  version = reinterpret_cast<klayout_cuda_spatial_abi_version_func>(
      GetProcAddress(backend.handle, "klayout_cuda_spatial_abi_version"));
  backend.run = reinterpret_cast<RunFunction>(
      GetProcAddress(
          backend.handle,
          "klayout_cuda_spatial_run_antenna_m1_m4_empty_v1"));
#  else
  dlerror();
  backend.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!backend.handle) {
    const char *message = dlerror();
    error = message ? message : "dlopen failed";
    return false;
  }
  version = reinterpret_cast<klayout_cuda_spatial_abi_version_func>(
      dlsym(backend.handle, "klayout_cuda_spatial_abi_version"));
  backend.run = reinterpret_cast<RunFunction>(
      dlsym(backend.handle,
            "klayout_cuda_spatial_run_antenna_m1_m4_empty_v1"));
#  endif
  if (!version) {
    error = "backend has no klayout_cuda_spatial_abi_version export";
    return false;
  }
  if (version() != KLAYOUT_CUDA_SPATIAL_ABI_VERSION) {
    error = "backend ABI version is incompatible";
    return false;
  }
  if (!backend.run) {
    error =
        "backend has no ANTENNA.M1-M4 replay export";
    return false;
  }
  error.clear();
  return true;
#endif
}

const char *status_name (int status)
{
  switch (status) {
  case KLAYOUT_CUDA_SPATIAL_OK:
    return "OK";
  case KLAYOUT_CUDA_SPATIAL_FALLBACK:
    return "FALLBACK";
  case KLAYOUT_CUDA_SPATIAL_ERROR:
    return "ERROR";
  case KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT:
    return "BAD_ARGUMENT";
  default:
    return "UNKNOWN";
  }
}

std::string result_message (const Result &result)
{
  const char *end =
      std::find(result.message,
                result.message + sizeof(result.message), '\0');
  return std::string(result.message, end);
}

double ns_to_ms (std::uint64_t value)
{
  return static_cast<double>(value) / 1.0e6;
}

void print_result (const capture::OwnedRequest &owned,
                   const DynamicBackend &backend, const std::string &capture_path,
                   const Result &result, int returned_status,
                   double wall_ms)
{
  std::cout << "capture=" << capture_path << "\n"
            << "backend=" << backend.path << "\n"
            << "input cells="
            << owned.request.hierarchy.source_cell_count
            << " contexts=" << owned.request.hierarchy.context_count
            << " stored_polygons="
            << owned.request.census.stored_polygon_count
            << " stored_edges="
            << owned.request.census.stored_edge_count
            << " expanded_polygons="
            << owned.request.census.expanded_polygon_count
            << " expanded_edges="
            << owned.request.census.expanded_edge_count << "\n"
            << "request_device=" << owned.request.device
            << " max_device_bytes="
            << owned.request.capacity.max_device_bytes
            << " max_estimated_peak_bytes="
            << owned.request.capacity.max_estimated_peak_bytes << "\n"
            << "return_status=" << returned_status << " ("
            << status_name(returned_status) << ")"
            << " result_status=" << result.status << " ("
            << status_name(static_cast<int>(result.status)) << ")"
            << " disposition=" << result.disposition
            << " fallback_flags=0x" << std::hex
            << result.fallback_flags
            << " device_flags=0x" << result.device_flags
            << std::dec << "\n"
            << std::fixed << std::setprecision(3)
            << "wall_ms=" << wall_ms
            << " backend_total_ms=" << ns_to_ms(result.total_ns)
            << " setup_ms=" << ns_to_ms(result.setup_ns)
            << " h2d_ms=" << ns_to_ms(result.h2d_ns)
            << " d2h_ms=" << ns_to_ms(result.d2h_ns) << "\n"
            << "accounted_peak_device_bytes="
            << result.accounted_peak_device_bytes
            << " certified_empty_mask=0x" << std::hex
            << result.certified_empty_mask << " clean_mask=0x"
            << result.clean_mask << " closed_domain_mask=0x"
            << result.closed_domain_mask << " released_stage_mask=0x"
            << result.released_stage_mask << std::dec << "\n";

  for (std::size_t index = 0; index < kDomainCount; ++index) {
    const auto &domain = result.domain_results[index];
    std::cout << "domain[" << index << "]"
              << " role=" << domain.role
              << " owners=" << domain.owner_count
              << " rectangles=" << domain.rectangle_count
              << " owner_ranges=" << domain.owner_range_count << "\n";
  }
  for (std::size_t index = 0;
       index < KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT;
       ++index) {
    const auto &stage = result.stages[index];
    std::cout << "stage[" << index << "]"
              << " id=0x" << std::hex << stage.stage << std::dec
              << " components=" << stage.component_count
              << " memberships=" << stage.membership_count
              << " occupied_cells=" << stage.occupied_cell_count
              << " pair_occurrences=" << stage.pair_occurrence_count
              << " unique_candidates="
              << stage.unique_owner_candidate_count
              << " edges=" << stage.edge_count
              << " gates=" << stage.gate_count
              << " evaluated=" << stage.evaluated_count
              << " retained=" << stage.retained_rectangle_count
              << " released=" << stage.released_rectangle_count
              << " dsu_iterations=" << stage.dsu_iteration_count
              << " hits=" << stage.hit_count
              << " uncertainties=" << stage.uncertainty_count
              << " work=" << stage.work_count
              << " ms=" << ns_to_ms(stage.stage_ns) << "\n";
  }
  std::cout << "message=" << result_message(result) << "\n";
}

capture::OwnedRequest make_fixture ()
{
  capture::OwnedRequest owned;
  owned.source_cell_indices = {UINT64_C(0x0123456789abcdef)};
  capture::Context context{};
  context.tx = -17;
  context.ty = 23;
  context.cell_id = 0;
  context.transform_code = 0;
  owned.contexts.push_back(context);
  owned.context_parent_ids.push_back(UINT32_MAX);

  for (std::size_t index = 0; index < kDomainCount; ++index) {
    capture::Cell cell{};
    cell.polygon_begin = 0;
    cell.edge_begin = 0;
    cell.polygon_count = 1;
    cell.edge_count = 4;
    owned.domains[index].cells.push_back(cell);

    capture::Polygon polygon{};
    polygon.edge_begin = 0;
    polygon.left = 0;
    polygon.bottom = 0;
    polygon.right = 10 + static_cast<std::int64_t>(index);
    polygon.top = 20 + static_cast<std::int64_t>(index);
    polygon.polygon_id = static_cast<std::uint32_t>(index);
    polygon.edge_count = 4;
    owned.domains[index].polygons.push_back(polygon);

    const std::int64_t right = polygon.right;
    const std::int64_t top = polygon.top;
    owned.domains[index].edges = {
        {0, 0, 0, top}, {0, top, right, top},
        {right, top, right, 0}, {right, 0, 0, 0}};
  }

  Request &request = owned.request;
  std::memset(&request, 0, sizeof(request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof(request);
  request.opcode =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RAW_SHARED_EMPTY;
  request.option_flags =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.requested_mask =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_ALL_STAGES;
  request.stage_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_STAGE_COUNT;
  request.ratio_numerator =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_NUMERATOR;
  request.ratio_denominator =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_RATIO_DENOMINATOR;
  request.domain_count =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_DOMAIN_COUNT;
  request.device = 0;

  auto &hierarchy = request.hierarchy;
  hierarchy.struct_size = sizeof(hierarchy);
  hierarchy.format_version = 2;
  hierarchy.dbu_per_micron = 2000;
  hierarchy.root_cell = 0;
  hierarchy.source_root_cell_index = owned.source_cell_indices[0];
  hierarchy.source_cell_count = owned.source_cell_indices.size();
  hierarchy.source_cell_index_record_bytes = sizeof(std::uint64_t);
  hierarchy.context_count = owned.contexts.size();
  hierarchy.context_record_bytes = sizeof(capture::Context);
  hierarchy.context_parent_count = owned.context_parent_ids.size();
  hierarchy.context_parent_record_bytes = sizeof(std::uint32_t);
  for (std::size_t index = 0;
       index < sizeof(hierarchy.hierarchy_digest); ++index) {
    hierarchy.hierarchy_digest[index] =
        static_cast<std::uint8_t>(0x20 + index);
  }

  std::uint64_t stored_cells = 0;
  std::uint64_t stored_polygons = 0;
  std::uint64_t stored_edges = 0;
  std::uint64_t total_stored = 0;
  std::uint64_t total_expanded = 0;
  for (std::size_t index = 0; index < kDomainCount; ++index) {
    auto &domain = request.domains[index];
    domain.struct_size = sizeof(domain);
    domain.role = static_cast<std::uint32_t>(index);
    domain.physical_layer = kPhysicalLayers[index];
    domain.datatype = 0;
    domain.source_layer_index = 100 + static_cast<std::uint32_t>(index);
    domain.cell_count = owned.domains[index].cells.size();
    domain.cell_record_bytes = sizeof(capture::Cell);
    domain.polygon_count = owned.domains[index].polygons.size();
    domain.polygon_record_bytes = sizeof(capture::Polygon);
    domain.edge_count = owned.domains[index].edges.size();
    domain.edge_record_bytes = sizeof(capture::Edge);
    domain.nonempty_context_count = 1;
    domain.flat_polygon_count = 1;
    domain.flat_edge_count = 4;
    domain.stored_bytes =
        sizeof(capture::Cell) + sizeof(capture::Polygon) +
        4 * sizeof(capture::Edge);
    domain.expanded_geometry_bytes =
        sizeof(capture::Polygon) + 4 * sizeof(capture::Edge);
    domain.scene_left = 0;
    domain.scene_bottom = 0;
    domain.scene_right = 10 + static_cast<std::int64_t>(index);
    domain.scene_top = 20 + static_cast<std::int64_t>(index);
    std::memcpy(domain.digest_domain, kDigestDomains[index], 8);
    for (std::size_t byte = 0; byte < sizeof(domain.scene_digest);
         ++byte) {
      domain.scene_digest[byte] =
          static_cast<std::uint8_t>(index * 17 + byte);
    }
    stored_cells += domain.cell_count;
    stored_polygons += domain.polygon_count;
    stored_edges += domain.edge_count;
    total_stored += domain.stored_bytes;
    total_expanded += domain.expanded_geometry_bytes;
  }

  auto &census = request.census;
  census.struct_size = sizeof(census);
  census.format_version = 1;
  census.shared_cell_count = hierarchy.source_cell_count;
  census.shared_context_count = hierarchy.context_count;
  census.context_parent_record_count = hierarchy.context_parent_count;
  census.stored_cell_record_count = stored_cells;
  census.stored_polygon_count = stored_polygons;
  census.stored_edge_count = stored_edges;
  census.expanded_polygon_count = stored_polygons;
  census.expanded_edge_count = stored_edges;
  census.total_stored_bytes = total_stored;
  census.total_expanded_geometry_bytes = total_expanded;
  census.estimated_peak_bytes = total_stored + total_expanded;

  auto &capacity = request.capacity;
  capacity.struct_size = sizeof(capacity);
  capacity.max_cells = 1024;
  capacity.max_contexts = 1024;
  capacity.max_stored_polygons = 1024;
  capacity.max_stored_edges = 4096;
  capacity.max_flat_polygons = 1024;
  capacity.max_flat_edges = 4096;
  capacity.max_total_stored_bytes = UINT64_C(1) << 30;
  capacity.max_total_expanded_geometry_bytes = UINT64_C(1) << 30;
  capacity.max_estimated_peak_bytes = UINT64_C(2) << 30;
  capacity.max_nodes = 4096;
  capacity.max_rectangles = 4096;
  capacity.max_memberships = 65536;
  capacity.max_pair_occurrences = 65536;
  capacity.max_unique_candidates = 65536;
  capacity.max_cell_members = 4096;
  capacity.max_dsu_iterations = 4096;
  capacity.max_rule_work = 65536;
  capacity.max_device_bytes = UINT64_C(2) << 30;
  for (std::size_t index = 0;
       index < sizeof(request.lower_capture_digest); ++index) {
    request.lower_capture_digest[index] =
        static_cast<std::uint8_t>(0x40 + index);
    request.capture_digest[index] =
        static_cast<std::uint8_t>(0x80 + index);
  }
  owned.rebind();
  return owned;
}

std::vector<std::uint8_t> read_file (const std::string &path)
{
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("unable to open " + path);
  stream.seekg(0, std::ios::end);
  const std::streamoff count = stream.tellg();
  if (count < 0) throw std::runtime_error("unable to size " + path);
  stream.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(count));
  if (!bytes.empty()) {
    stream.read(reinterpret_cast<char *>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
  }
  if (!stream) throw std::runtime_error("unable to read " + path);
  return bytes;
}

void write_file (const std::string &path,
                 const std::vector<std::uint8_t> &bytes)
{
  std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to create " + path);
  if (!bytes.empty()) {
    stream.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
  }
  if (!stream) throw std::runtime_error("unable to write " + path);
}

bool expect_rejected (const std::string &path, const char *label,
                      std::string &error)
{
  capture::OwnedRequest ignored;
  if (capture::load_request(path, ignored, &error)) {
    error = std::string(label) + " unexpectedly loaded";
    return false;
  }
  if (error.empty()) {
    error = std::string(label) + " rejection had no diagnostic";
    return false;
  }
  return true;
}

bool set_capture_output (const std::string &path, std::string &error)
{
#if defined(_WIN32)
  if (_putenv_s("KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_OUT",
                path.c_str()) != 0) {
    error = "unable to set capture-output environment variable";
    return false;
  }
#else
  if (setenv("KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_OUT",
             path.c_str(), 1) != 0) {
    error = "unable to set capture-output environment variable";
    return false;
  }
#endif
  return true;
}

void clear_capture_output ()
{
#if defined(_WIN32)
  (void)_putenv_s("KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_OUT", "");
#else
  (void)unsetenv("KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_OUT");
#endif
}

int run_self_test (const std::string &base,
                   const std::string &capture_backend_path)
{
  const std::string roundtrip = base + ".roundtrip";
  const std::string corrupt = base + ".corrupt";
  const std::string truncated = base + ".truncated";
  const std::string trailing = base + ".trailing";
  const std::string backend_capture = base + ".backend";
  const auto cleanup = [&]() {
    std::remove(base.c_str());
    std::remove(roundtrip.c_str());
    std::remove(corrupt.c_str());
    std::remove(truncated.c_str());
    std::remove(trailing.c_str());
    std::remove(backend_capture.c_str());
  };

  try {
    cleanup();
    capture::OwnedRequest fixture = make_fixture();
    std::string error;
    if (!capture::dump_request(base, fixture.request, &error)) {
      throw std::runtime_error("fixture dump failed: " + error);
    }
    capture::OwnedRequest loaded;
    if (!capture::load_request(base, loaded, &error)) {
      throw std::runtime_error("fixture load failed: " + error);
    }
    if (!capture::dump_request(roundtrip, loaded.request, &error)) {
      throw std::runtime_error("roundtrip dump failed: " + error);
    }
    const std::vector<std::uint8_t> canonical = read_file(base);
    if (canonical != read_file(roundtrip)) {
      throw std::runtime_error("roundtrip file is not byte-identical");
    }

    ReplayOverrides raised;
    raised.has_device = true;
    raised.device = 7;
    raised.has_max_device_bytes = true;
    raised.max_device_bytes = UINT64_C(96) << 30;
    raised.has_max_estimated_peak_bytes = true;
    raised.max_estimated_peak_bytes =
        loaded.request.census.estimated_peak_bytes + 4096;
    capture::OwnedRequest overridden = loaded;
    if (!apply_overrides(raised, overridden, error) ||
        overridden.request.device != raised.device ||
        overridden.request.capacity.max_device_bytes !=
            raised.max_device_bytes ||
        overridden.request.capacity.max_estimated_peak_bytes !=
            raised.max_estimated_peak_bytes) {
      throw std::runtime_error("valid replay override was rejected");
    }
    ReplayOverrides too_small;
    too_small.has_max_estimated_peak_bytes = true;
    too_small.max_estimated_peak_bytes =
        loaded.request.census.estimated_peak_bytes - 1;
    if (apply_overrides(too_small, overridden, error)) {
      throw std::runtime_error(
          "under-census replay override was accepted");
    }

    std::vector<std::uint8_t> mutated = canonical;
    mutated.back() ^= 0x80;
    write_file(corrupt, mutated);
    if (!expect_rejected(corrupt, "corruption", error)) {
      throw std::runtime_error(error);
    }

    mutated = canonical;
    mutated.pop_back();
    write_file(truncated, mutated);
    if (!expect_rejected(truncated, "truncation", error)) {
      throw std::runtime_error(error);
    }

    mutated = canonical;
    mutated.push_back(0);
    write_file(trailing, mutated);
    if (!expect_rejected(trailing, "trailing byte", error)) {
      throw std::runtime_error(error);
    }

    if (!capture_backend_path.empty()) {
      DynamicBackend backend;
      if (!load_backend(capture_backend_path, backend, error)) {
        throw std::runtime_error("capture-backend load failed: " + error);
      }
      if (!set_capture_output(backend_capture, error)) {
        throw std::runtime_error(error);
      }
      Result result{};
      result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
      result.struct_size = sizeof(result);
      result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
      result.disposition =
          KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_UNCERTAIN;
      const int status = backend.run(&fixture.request, &result);
      clear_capture_output();
      if (status != KLAYOUT_CUDA_SPATIAL_FALLBACK ||
          status != static_cast<int>(result.status)) {
        throw std::runtime_error(
            "capture-only backend did not return consistent FALLBACK");
      }
      capture::OwnedRequest backend_loaded;
      if (!capture::load_request(
              backend_capture, backend_loaded, &error)) {
        throw std::runtime_error(
            "capture-only backend output failed to load: " + error);
      }
      if (canonical != read_file(backend_capture)) {
        throw std::runtime_error(
            "capture-only backend output is not canonical");
      }
    }

    cleanup();
    std::cout
        << "antenna_m1_m4_capture_replay self-test: PASS"
        << " (canonical roundtrip, replay overrides, SHA corruption, "
           "truncation, trailing ambiguity";
    if (!capture_backend_path.empty()) {
      std::cout << ", capture-only backend";
    }
    std::cout << ")\n";
    return 0;
  } catch (const std::exception &exception) {
    clear_capture_output();
    cleanup();
    std::cerr << "antenna_m1_m4_capture_replay self-test: FAIL: "
              << exception.what() << "\n";
    return 1;
  }
}

void usage (const char *program)
{
  std::cerr
      << "usage: " << program
      << " [--backend LIBRARY] [--device N]"
         " [--max-device-bytes BYTES]"
         " [--max-estimated-peak-bytes BYTES] CAPTURE\n"
      << "       " << program
      << " [--backend CAPTURE_BACKEND] --self-test [TEMP_PREFIX]\n";
}

}  // namespace

int main (int argc, char **argv)
{
  std::string backend_path = default_backend_path();
  bool backend_explicit = false;
  ReplayOverrides overrides;
  std::string capture_path;
  bool self_test = false;
  std::string self_test_path =
      "/tmp/antenna_m1_m4_capture_self_test.kam4";
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--backend") {
      if (++index == argc) {
        usage(argv[0]);
        return 64;
      }
      backend_path = argv[index];
      backend_explicit = true;
    } else if (argument == "--device") {
      std::uint64_t value = 0;
      if (++index == argc || !parse_u64(argv[index], value) ||
          value > static_cast<std::uint64_t>(INT32_MAX)) {
        std::cerr << "--device requires an int32 device index\n";
        return 64;
      }
      overrides.has_device = true;
      overrides.device = static_cast<std::int32_t>(value);
    } else if (argument == "--max-device-bytes") {
      if (++index == argc ||
          !parse_u64(argv[index], overrides.max_device_bytes) ||
          !overrides.max_device_bytes) {
        std::cerr
            << "--max-device-bytes requires a nonzero uint64 byte count\n";
        return 64;
      }
      overrides.has_max_device_bytes = true;
    } else if (argument == "--max-estimated-peak-bytes") {
      if (++index == argc ||
          !parse_u64(
              argv[index], overrides.max_estimated_peak_bytes) ||
          !overrides.max_estimated_peak_bytes) {
        std::cerr << "--max-estimated-peak-bytes requires a nonzero "
                     "uint64 byte count\n";
        return 64;
      }
      overrides.has_max_estimated_peak_bytes = true;
    } else if (argument == "--self-test") {
      self_test = true;
      if (index + 1 < argc && argv[index + 1][0] != '-') {
        self_test_path = argv[++index];
      }
    } else if (!argument.empty() && argument[0] == '-') {
      usage(argv[0]);
      return 64;
    } else if (capture_path.empty()) {
      capture_path = argument;
    } else {
      usage(argv[0]);
      return 64;
    }
  }
  if (self_test) {
    if (!capture_path.empty() || !overrides.empty()) {
      usage(argv[0]);
      return 64;
    }
    return run_self_test(
        self_test_path, backend_explicit ? backend_path : std::string());
  }
  if (capture_path.empty()) {
    usage(argv[0]);
    return 64;
  }

  capture::OwnedRequest owned;
  std::string error;
  if (!capture::load_request(capture_path, owned, &error)) {
    std::cerr << "capture load failed: " << error << "\n";
    return 65;
  }
  if (!apply_overrides(overrides, owned, error)) {
    std::cerr << "replay override rejected: " << error << "\n";
    return 64;
  }

  DynamicBackend backend;
  if (!load_backend(backend_path, backend, error)) {
    std::cerr << "backend load failed: " << error << "\n";
    return 66;
  }

  Result result{};
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof(result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  result.disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_UNCERTAIN;
  const auto begin = std::chrono::steady_clock::now();
  const int status = backend.run(&owned.request, &result);
  const auto end = std::chrono::steady_clock::now();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  print_result(owned, backend, capture_path, result, status, wall_ms);
  if (status != static_cast<int>(result.status)) {
    std::cerr << "backend returned inconsistent status values\n";
    return 67;
  }
  return status == KLAYOUT_CUDA_SPATIAL_OK
             ? 0
             : (status == KLAYOUT_CUDA_SPATIAL_FALLBACK ? 2 : 1);
}
