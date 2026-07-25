/*
 * End-to-end fake-DSO gate for the raw-M2-to-flat-union host transaction.
 *
 * Every invocation is a fresh process because CudaSpatialModule deliberately
 * snapshots its environment and optional symbols on first use.
 */

#include "dbBox.h"
#include "dbCudaM2Rules.h"
#include "dbDeepShapeStore.h"
#include "dbLayerProperties.h"
#include "dbRegion.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace
{

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

bool expected_disposition (
  const db::CudaM2FlatUnionAttempt &attempt,
  const std::string &expected)
{
  if (expected == "disabled") {
    return attempt.disposition == db::CudaM2FlatUnionAttempt::Disabled;
  } else if (expected == "host") {
    return attempt.disposition == db::CudaM2FlatUnionAttempt::HostDeclined;
  } else if (expected == "complete") {
    return attempt.disposition == db::CudaM2FlatUnionAttempt::Complete;
  } else if (expected == "fallback") {
    return
      attempt.disposition == db::CudaM2FlatUnionAttempt::BackendFallback;
  } else if (expected == "invalid") {
    return
      attempt.disposition == db::CudaM2FlatUnionAttempt::InvalidResult;
  } else if (expected == "topology") {
    return
      attempt.disposition == db::CudaM2FlatUnionAttempt::TopologyDeclined;
  } else if (expected == "error") {
    return
      attempt.disposition == db::CudaM2FlatUnionAttempt::BackendError;
  }
  return false;
}

} // anonymous namespace

int main (int argc, char **argv)
{
  if (argc != 4) {
    std::cerr
      << "usage: m2_union_live_transaction_smoke "
         "BACKEND MODE "
         "disabled|host|complete|fallback|invalid|topology|error\n";
    return 2;
  }
  if (! set_environment ("KLAYOUT_CUDA_SPATIAL_BACKEND", argv [1]) ||
      ! set_environment ("KLAYOUT_CUDA_M2_RULES", "1") ||
      ! set_environment ("KLAYOUT_CUDA_M2_RULES_TELEMETRY", "0") ||
      ! set_environment ("KLAYOUT_CUDA_M2_UNION_FAKE_MODE", argv [2])) {
    std::cerr << "unable to configure the live-transaction environment\n";
    return 2;
  }

  CounterModule counters = load_counters (argv [1]);
  if (! counters.handle || ! counters.run_count || ! counters.release_count) {
    std::cerr << "unable to load fake-backend counters\n";
    close_counters (counters);
    return 2;
  }

  const std::string expected (argv [3]);
  const bool legacy_api =
    std::string (argv [2]) == "legacy_opcode";
  const bool bad_host_size =
    std::string (argv [2]) == "suffix_bad_host_size";
  db::Region output (db::Box (100, 200, 300, 400));
  db::CudaM2FlatUnionAttempt attempt;
  db::CudaM2FlatUnionSuffixCertificate certificate = {};
  if (bad_host_size) {
    std::memset (&certificate, 0xa5, sizeof (certificate));
  }
  if (expected == "disabled" || expected == "host") {
    const db::DeepLayer deliberately_invalid;
    attempt = legacy_api
      ? db::cuda_m2_raw_manhattan_try_flat_union (
          deliberately_invalid, output)
      : db::cuda_m2_raw_manhattan_try_flat_union_with_suffix (
          deliberately_invalid, output, &certificate,
          bad_host_size ? sizeof (certificate) - 1
                        : sizeof (certificate));
  } else {
    db::DeepShapeStore store ("TOP", 0.0005);
    db::Region seed (db::Box (0, 0, 10, 20));
    db::DeepLayer raw_m2 = store.create_from_flat (seed, false);
    raw_m2.layout ().set_properties (
      raw_m2.layer (), db::LayerProperties (13, 0));
    attempt = legacy_api
      ? db::cuda_m2_raw_manhattan_try_flat_union (raw_m2, output)
      : db::cuda_m2_raw_manhattan_try_flat_union_with_suffix (
          raw_m2, output, &certificate,
          bad_host_size ? sizeof (certificate) - 1
                        : sizeof (certificate));
  }

  const int run_count = counters.run_count ();
  const int release_count = counters.release_count ();
  const bool complete = expected == "complete";
  const bool output_ok =
    complete
      ? output.count () == size_t (1) &&
        output.bbox () == db::Box (0, 0, 10, 20) &&
        output.merged_semantics () && output.is_merged () &&
        attempt.flat_stats.segment_count == 4 &&
        attempt.flat_stats.contour_count == 1 &&
        attempt.flat_stats.vertex_count == 4 &&
        (legacy_api
           ? certificate.certified_empty_mask == 0 &&
             certificate.total_ns == 0
           : certificate.certified_empty_mask ==
               KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY &&
             certificate.total_ns == 500)
      : output.count () == size_t (1) &&
        output.bbox () == db::Box (100, 200, 300, 400);
  const unsigned char *certificate_bytes =
    reinterpret_cast<const unsigned char *> (&certificate);
  const bool certificate_untouched =
    std::find_if (
      certificate_bytes, certificate_bytes + sizeof (certificate),
      [] (unsigned char value) { return value != 0xa5; }) ==
        certificate_bytes + sizeof (certificate);
  const bool certificate_ok =
    legacy_api
      ? certificate.certified_empty_mask == 0 &&
        certificate.total_ns == 0
      : bad_host_size
        ? certificate_untouched
        : certificate.format_version ==
            db::CudaM2FlatUnionSuffixCertificate::FormatVersion &&
          certificate.struct_size == sizeof (certificate) &&
          certificate.reserved == 0 &&
          (complete
             ? certificate.certified_empty_mask ==
                 KLAYOUT_CUDA_SPATIAL_M2_SUFFIX_ALL_EMPTY &&
               certificate.total_ns == 500
             : certificate.certified_empty_mask == 0 &&
               certificate.total_ns == 0);
  const bool counters_ok =
    (expected == "disabled" || expected == "host" || bad_host_size)
      ? run_count == 0 && release_count == 0
      : run_count == 1 && release_count == 1;
  const bool telemetry_ok =
    bad_host_size
      ? attempt.lowering_ns == 0 && attempt.boundary_segment_count == 0
    : expected == "disabled"
      ? attempt.lowering_ns == 0 && attempt.boundary_segment_count == 0
      : expected == "host"
        ? attempt.boundary_segment_count == 0 &&
          ! attempt.message.empty ()
      : expected == "error"
        ? attempt.context_count == 0 &&
          attempt.boundary_segment_count == 0
      : attempt.context_count == 1 &&
        attempt.metal_context_count == 1 &&
        attempt.cell_count == 1 &&
        attempt.polygon_count == 1 &&
        attempt.edge_count == 4;
  const bool good =
    expected_disposition (attempt, expected) &&
    output_ok && certificate_ok && counters_ok && telemetry_ok;
  close_counters (counters);
  if (! good) {
    std::cerr
      << "live transaction case failed: mode=" << argv [2]
      << " expected=" << expected
      << " disposition=" << int (attempt.disposition)
      << " output=" << output.to_string ()
      << " run_count=" << run_count
      << " release_count=" << release_count
      << " contexts=" << attempt.context_count
      << " segments=" << attempt.boundary_segment_count
      << " suffix_mask=" << certificate.certified_empty_mask
      << " suffix_ns=" << certificate.total_ns
      << " message=" << attempt.message << "\n";
    return 1;
  }

  std::cout
    << "live transaction case passed: mode=" << argv [2]
    << " disposition=" << expected
    << " run_count=" << run_count
    << " release_count=" << release_count << "\n";
  return 0;
}
