/*
 * End-to-end host-loader contract test for the optional raw-M2 union ABI.
 *
 * Each case runs in a fresh process because CudaSpatialModule intentionally
 * snapshots its environment and DSO symbols on first use.
 */

#include "dbCudaSpatialBackend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace
{

typedef klayout_cuda_spatial_m2_union_segment_v1 Segment;
typedef int (*CountFunction) (void);

bool set_environment (const char *name, const char *value)
{
#if defined(_WIN32)
  return _putenv_s (name, value) == 0;
#else
  return setenv (name, value, 1) == 0;
#endif
}

struct CounterModule
{
#if defined(_WIN32)
  HMODULE handle;
#else
  void *handle;
#endif
  CountFunction run_count;
  CountFunction release_count;
};

CounterModule load_counters (const char *path)
{
#if defined(_WIN32)
  HMODULE handle = LoadLibraryA (path);
  return CounterModule {
    handle,
    handle
      ? reinterpret_cast<CountFunction> (
          GetProcAddress (handle, "klayout_cuda_m2_union_fake_run_count"))
      : 0,
    handle
      ? reinterpret_cast<CountFunction> (
          GetProcAddress (
            handle, "klayout_cuda_m2_union_fake_release_count"))
      : 0
  };
#else
  void *handle = dlopen (path, RTLD_NOW | RTLD_LOCAL);
  return CounterModule {
    handle,
    handle
      ? reinterpret_cast<CountFunction> (
          dlsym (handle, "klayout_cuda_m2_union_fake_run_count"))
      : 0,
    handle
      ? reinterpret_cast<CountFunction> (
          dlsym (handle, "klayout_cuda_m2_union_fake_release_count"))
      : 0
  };
#endif
}

void close_counters (CounterModule &module)
{
#if defined(_WIN32)
  if (module.handle) {
    FreeLibrary (module.handle);
  }
#else
  if (module.handle) {
    dlclose (module.handle);
  }
#endif
  module.handle = 0;
}

klayout_cuda_spatial_m2_union_request_v1 make_request (
  const char *mode,
  const klayout_cuda_spatial_m1_width_space_context_v1 *contexts,
  const uint32_t *metal_contexts, const uint64_t *polygon_offsets,
  const uint64_t *edge_offsets,
  const klayout_cuda_spatial_m1_width_space_cell_v1 *cells,
  const klayout_cuda_spatial_m1_width_space_polygon_v1 *polygons,
  const klayout_cuda_spatial_m1_width_space_edge_v1 *edges)
{
  klayout_cuda_spatial_m2_union_request_v1 request;
  std::memset (&request, 0, sizeof (request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof (request);
  request.opcode =
    KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_BOUNDARY;
  if (std::strncmp (mode, "suffix_", 7) == 0) {
    request.opcode =
      KLAYOUT_CUDA_SPATIAL_M2_RAW_MANHATTAN_UNION_M25_9_EMPTY;
  }
  request.option_flags = KLAYOUT_CUDA_SPATIAL_M2_UNION_QUALIFIED_OPTIONS;
  request.format_version = 1;
  request.dbu_per_micron = 2000;
  request.root_cell = 0;
  request.device = 0;
  request.contexts = contexts;
  request.context_count = 1;
  request.context_record_bytes = sizeof (*contexts);
  request.metal_contexts = metal_contexts;
  request.metal_context_count = 1;
  request.context_polygon_offsets = polygon_offsets;
  request.context_polygon_offset_count = 1;
  request.context_edge_offsets = edge_offsets;
  request.context_edge_offset_count = 1;
  request.cells = cells;
  request.cell_count = 1;
  request.cell_record_bytes = sizeof (*cells);
  request.polygons = polygons;
  request.polygon_count = 1;
  request.polygon_record_bytes = sizeof (*polygons);
  request.edges = edges;
  request.edge_count = 4;
  request.edge_record_bytes = sizeof (*edges);
  request.flat_polygon_count = 1;
  request.flat_edge_count = 4;
  request.scene_left = 0;
  request.scene_bottom = 0;
  request.scene_right = 10;
  request.scene_top = 20;
  request.max_contexts = 1;
  request.max_rectangles = 1;
  request.max_x_slabs = 4;
  request.max_memberships = 16;
  request.max_events = 16;
  request.max_raw_segments = 16;
  request.max_segments = 8;
  request.max_slabs_per_rectangle = 4;
  for (unsigned int index = 0; index < 32; ++index) {
    request.scene_digest [index] = uint8_t (index);
  }

  if (std::strcmp (mode, "copy_throw") == 0) {
    const uint64_t max_size =
      uint64_t (std::vector<Segment> ().max_size ());
    if (max_size == std::numeric_limits<uint64_t>::max ()) {
      request.max_segments = max_size;
    } else {
      request.max_segments = max_size + 1;
    }
    request.max_raw_segments = request.max_segments;
  }
  return request;
}

bool expected_disposition (
  const db::CudaM2UnionAttempt &attempt, const std::string &expected)
{
  if (expected == "complete") {
    return attempt.disposition == db::CudaM2UnionAttempt::Complete;
  } else if (expected == "fallback") {
    return attempt.disposition == db::CudaM2UnionAttempt::BackendFallback;
  } else if (expected == "invalid") {
    return attempt.disposition == db::CudaM2UnionAttempt::InvalidResult;
  } else if (expected == "error") {
    return attempt.disposition == db::CudaM2UnionAttempt::BackendError;
  }
  return false;
}

} // anonymous namespace

int main (int argc, char **argv)
{
  if (argc != 4) {
    std::cerr
      << "usage: m2_union_backend_contract_smoke "
         "BACKEND MODE unavailable|complete|fallback|invalid|error\n";
    return 2;
  }

  if (! set_environment ("KLAYOUT_CUDA_SPATIAL_BACKEND", argv [1]) ||
      ! set_environment ("KLAYOUT_CUDA_M2_RULES", "1") ||
      ! set_environment ("KLAYOUT_CUDA_M2_RULES_TELEMETRY", "0") ||
      ! set_environment ("KLAYOUT_CUDA_M2_UNION_FAKE_MODE", argv [2])) {
    std::cerr << "unable to configure the contract-test environment\n";
    return 2;
  }

  CounterModule counters = load_counters (argv [1]);
  if (! counters.handle || ! counters.run_count || ! counters.release_count) {
    std::cerr << "unable to load fake-backend counters\n";
    close_counters (counters);
    return 2;
  }

  const std::string expected (argv [3]);
  const bool available = db::cuda_spatial_m2_union_requested ();
  if (expected == "unavailable") {
    const bool good =
      ! available && counters.run_count () == 0 &&
      counters.release_count () == 0;
    close_counters (counters);
    if (! good) {
      std::cerr
        << "incomplete run/release capability was advertised or invoked\n";
      return 1;
    }
    std::cout
      << "capability gate passed before request/scene construction\n";
    return 0;
  }
  if (! available) {
    std::cerr << "complete run/release capability was not advertised\n";
    close_counters (counters);
    return 1;
  }

  const klayout_cuda_spatial_m1_width_space_context_v1 contexts [] = {
    { 0, 0, 0, 0 }
  };
  const uint32_t metal_contexts [] = { 0 };
  const uint64_t polygon_offsets [] = { 0 };
  const uint64_t edge_offsets [] = { 0 };
  const klayout_cuda_spatial_m1_width_space_cell_v1 cells [] = {
    { 0, 0, 0, 1, 4 }
  };
  const klayout_cuda_spatial_m1_width_space_polygon_v1 polygons [] = {
    { 0, 0, 0, 10, 20, 0, 4 }
  };
  const klayout_cuda_spatial_m1_width_space_edge_v1 edges [] = {
    { 0, 0, 0, 20 },
    { 0, 20, 10, 20 },
    { 10, 20, 10, 0 },
    { 10, 0, 0, 0 }
  };
  const klayout_cuda_spatial_m2_union_request_v1 request =
    make_request (
      argv [2], contexts, metal_contexts, polygon_offsets, edge_offsets,
      cells, polygons, edges);

  db::CudaM2SuffixCertificate certificate;
  std::memset (&certificate, 0xa5, sizeof (certificate));
  const bool suffix_case =
    std::strncmp (argv [2], "suffix_", 7) == 0;
  const bool bad_host_size =
    std::strcmp (argv [2], "suffix_bad_host_size") == 0;
  const db::CudaM2UnionAttempt attempt =
    suffix_case
      ? db::cuda_spatial_try_m2_union_with_certificate (
          request, &certificate,
          bad_host_size ? sizeof (certificate) - 1
                        : sizeof (certificate))
      : db::cuda_spatial_try_m2_union (request);
  const int run_count = counters.run_count ();
  const int release_count = counters.release_count ();
  const bool side_order = std::strcmp (argv [2], "side_order") == 0;
  const uint64_t expected_fnv64 =
    side_order ? UINT64_C (11131890132215870808)
               : UINT64_C (11447980897846940057);
  const unsigned char *certificate_bytes =
    reinterpret_cast<const unsigned char *> (&certificate);
  const bool untouched_bad_size =
    std::find_if (
      certificate_bytes, certificate_bytes + sizeof (certificate),
      [] (unsigned char value) { return value != 0xa5; }) ==
    certificate_bytes + sizeof (certificate);
  const bool certificate_good =
    ! suffix_case ||
    (bad_host_size
       ? untouched_bad_size
       : certificate.format_version ==
           db::CudaM2SuffixCertificate::FormatVersion &&
         certificate.struct_size == sizeof (certificate) &&
         certificate.reserved == 0 &&
         (expected == "complete"
            ? certificate.certified_empty_mask ==
                KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY &&
              certificate.total_ns == 500
            : certificate.certified_empty_mask == 0 &&
              certificate.total_ns == 0));
  const int expected_calls = bad_host_size ? 0 : 1;
  const bool good =
    expected_disposition (attempt, expected) &&
    run_count == expected_calls && release_count == expected_calls &&
    certificate_good &&
    (expected == "complete"
       ? attempt.segments.size () == (side_order ? 2 : 4) &&
         attempt.boundary_fnv64 == expected_fnv64
       : attempt.segments.empty ());
  close_counters (counters);
  if (! good) {
    std::cerr
      << "contract case failed: mode=" << argv [2]
      << " expected=" << expected
      << " disposition=" << int (attempt.disposition)
      << " segments=" << attempt.segments.size ()
      << " run_count=" << run_count
      << " release_count=" << release_count
      << " certificate_mask=" << certificate.certified_empty_mask
      << " certificate_total_ns=" << certificate.total_ns
      << " message=" << attempt.message << "\n";
    return 1;
  }
  std::cout
    << "contract case passed: mode=" << argv [2]
    << " disposition=" << expected
    << " release_count=1\n";
  return 0;
}
