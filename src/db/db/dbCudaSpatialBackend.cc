/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

#include "dbCudaSpatialBackend.h"
#include "tlLog.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace db
{

CudaSpatialAttempt::CudaSpatialAttempt ()
  : disposition (Disabled), fallback_flags (0), membership_count (0),
    occupied_cell_count (0), pair_work_count (0), setup_ns (0), h2d_ns (0),
    broad_phase_ns (0), sort_unique_ns (0), d2h_ns (0), total_ns (0)
{
  //  nothing yet
}

namespace
{

uint64_t env_u64 (const char *name, uint64_t default_value)
{
  const char *value = std::getenv (name);
  if (! value || ! *value) {
    return default_value;
  }

  errno = 0;
  char *end = 0;
  unsigned long long parsed = std::strtoull (value, &end, 0);
  if (errno != 0 || ! end || *end != 0) {
    return default_value;
  }
  return static_cast<uint64_t> (parsed);
}

bool env_enabled (const char *name)
{
  const char *value = std::getenv (name);
  return value && *value && std::strcmp (value, "0") != 0 &&
         std::strcmp (value, "false") != 0 && std::strcmp (value, "off") != 0;
}

class CudaSpatialModule
{
public:
  CudaSpatialModule ()
    : m_enabled (false), m_telemetry (false), m_handle (0), m_run (0),
      m_release (0), m_min_records (100000)
  {
    const char *setting = std::getenv ("KLAYOUT_CUDA_SPATIAL_BACKEND");
    if (! setting || ! *setting || std::strcmp (setting, "0") == 0 ||
        std::strcmp (setting, "false") == 0 || std::strcmp (setting, "off") == 0) {
      return;
    }

    m_enabled = true;
    m_telemetry = env_enabled ("KLAYOUT_CUDA_SPATIAL_TELEMETRY");
    m_min_records = env_u64 ("KLAYOUT_CUDA_SPATIAL_MIN_RECORDS", m_min_records);

    std::string path (setting);
    if (path == "1" || path == "auto") {
#if defined(_WIN32)
      path = "klayout_cuda_spatial_backend.dll";
#elif defined(__APPLE__)
      path = "libklayout_cuda_spatial_backend.dylib";
#else
      path = "libklayout_cuda_spatial_backend.so";
#endif
    }

#if defined(_WIN32)
    m_handle = reinterpret_cast<void *> (LoadLibraryA (path.c_str ()));
    if (m_handle) {
      klayout_cuda_spatial_abi_version_func version =
        reinterpret_cast<klayout_cuda_spatial_abi_version_func> (
          GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_abi_version"));
      m_run = reinterpret_cast<klayout_cuda_spatial_run_bipartite_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_run_bipartite_v1"));
      m_release = reinterpret_cast<klayout_cuda_spatial_release_result_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_release_result_v1"));
      if (! version || version () != KLAYOUT_CUDA_SPATIAL_ABI_VERSION) {
        m_error = "CUDA spatial backend has an incompatible ABI";
      }
    } else {
      m_error = "unable to load CUDA spatial backend: " + path;
    }
#else
    m_handle = dlopen (path.c_str (), RTLD_NOW | RTLD_LOCAL);
    if (m_handle) {
      klayout_cuda_spatial_abi_version_func version =
        reinterpret_cast<klayout_cuda_spatial_abi_version_func> (
          dlsym (m_handle, "klayout_cuda_spatial_abi_version"));
      m_run = reinterpret_cast<klayout_cuda_spatial_run_bipartite_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_run_bipartite_v1"));
      m_release = reinterpret_cast<klayout_cuda_spatial_release_result_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_release_result_v1"));
      if (! version || version () != KLAYOUT_CUDA_SPATIAL_ABI_VERSION) {
        m_error = "CUDA spatial backend has an incompatible ABI";
      }
    } else {
      const char *error = dlerror ();
      m_error = std::string ("unable to load CUDA spatial backend: ") +
                (error ? error : path.c_str ());
    }
#endif

    if (m_handle && (! m_run || ! m_release) && m_error.empty ()) {
      m_error = "CUDA spatial backend is missing required ABI entry points";
    }

    if (! m_error.empty ()) {
      m_run = 0;
      m_release = 0;
      tl::warn << m_error;
    } else if (m_telemetry) {
      tl::info << "CUDA spatial backend loaded: " << path;
    }
  }

  bool enabled () const
  {
    return m_enabled;
  }

  bool ready () const
  {
    return m_run && m_release;
  }

  bool telemetry () const
  {
    return m_telemetry;
  }

  uint64_t min_records () const
  {
    return m_min_records;
  }

  const std::string &error () const
  {
    return m_error;
  }

  klayout_cuda_spatial_run_bipartite_v1_func run () const
  {
    return m_run;
  }

  klayout_cuda_spatial_release_result_v1_func release () const
  {
    return m_release;
  }

private:
  bool m_enabled;
  bool m_telemetry;
  void *m_handle;
  klayout_cuda_spatial_run_bipartite_v1_func m_run;
  klayout_cuda_spatial_release_result_v1_func m_release;
  uint64_t m_min_records;
  std::string m_error;
};

CudaSpatialModule &cuda_spatial_module ()
{
  static CudaSpatialModule module;
  return module;
}

void log_attempt (const CudaSpatialAttempt &attempt, uint64_t subjects,
                  uint64_t intruders)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.telemetry ()) {
    return;
  }

  const char *outcome = "unknown";
  switch (attempt.disposition) {
  case CudaSpatialAttempt::Success: outcome = "success"; break;
  case CudaSpatialAttempt::BackendFallback: outcome = "fallback"; break;
  case CudaSpatialAttempt::BackendError: outcome = "error"; break;
  case CudaSpatialAttempt::InvalidResult: outcome = "invalid-result"; break;
  case CudaSpatialAttempt::BelowThreshold: outcome = "below-threshold"; break;
  case CudaSpatialAttempt::Disabled: outcome = "disabled"; break;
  }

  tl::info << "CUDA spatial broad phase: outcome=" << outcome
           << " subjects=" << subjects << " intruders=" << intruders
           << " pairs=" << attempt.pair_keys.size ()
           << " memberships=" << attempt.membership_count
           << " pair_work=" << attempt.pair_work_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << (attempt.message.empty () ? "" : " message=") << attempt.message;
}

} // anonymous namespace

CudaSpatialAttempt cuda_spatial_try_bipartite (
  const std::vector<klayout_cuda_spatial_aabb_v1> &subjects,
  const std::vector<klayout_cuda_spatial_aabb_v1> &intruders,
  int64_t enlargement)
{
  CudaSpatialAttempt attempt;
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.enabled ()) {
    return attempt;
  }
  if (! module.ready ()) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = module.error ();
    log_attempt (attempt, subjects.size (), intruders.size ());
    return attempt;
  }

  const uint64_t total_records = uint64_t (subjects.size ()) + uint64_t (intruders.size ());
  if (total_records < module.min_records ()) {
    attempt.disposition = CudaSpatialAttempt::BelowThreshold;
    log_attempt (attempt, subjects.size (), intruders.size ());
    return attempt;
  }
  if (subjects.empty () || intruders.empty () || enlargement < 0 ||
      subjects.size () > std::numeric_limits<uint32_t>::max () ||
      intruders.size () > std::numeric_limits<uint32_t>::max () ||
      total_records > std::numeric_limits<uint32_t>::max ()) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
    attempt.fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    log_attempt (attempt, subjects.size (), intruders.size ());
    return attempt;
  }

  klayout_cuda_spatial_config_v1 config;
  std::memset (&config, 0, sizeof (config));
  config.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  config.struct_size = sizeof (config);
  config.device = int32_t (env_u64 ("KLAYOUT_CUDA_SPATIAL_DEVICE", 0));
  config.cell_size = env_u64 ("KLAYOUT_CUDA_SPATIAL_CELL_SIZE", 128);
  config.max_cells_per_record = uint32_t (std::min<uint64_t> (
    env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD", 64),
    std::numeric_limits<uint32_t>::max ()));
  config.max_records_per_cell = uint32_t (std::min<uint64_t> (
    env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL", 4096),
    std::numeric_limits<uint32_t>::max ()));
  config.max_memberships = env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS", 16000000);
  config.max_pair_work = env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK", 64000000);
  config.max_candidates = env_u64 ("KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES", 8000000);

  klayout_cuda_spatial_request_v1 request;
  std::memset (&request, 0, sizeof (request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof (request);
  request.subjects = subjects.data ();
  request.subject_count = subjects.size ();
  request.intruders = intruders.data ();
  request.intruder_count = intruders.size ();
  request.enlargement = enlargement;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);

  try {
    const int status = module.run () (&request, &result);
    attempt.fallback_flags = result.fallback_flags;
    attempt.membership_count = result.membership_count;
    attempt.occupied_cell_count = result.occupied_cell_count;
    attempt.pair_work_count = result.pair_work_count;
    attempt.setup_ns = result.setup_ns;
    attempt.h2d_ns = result.h2d_ns;
    attempt.broad_phase_ns = result.broad_phase_ns;
    attempt.sort_unique_ns = result.sort_unique_ns;
    attempt.d2h_ns = result.d2h_ns;
    attempt.total_ns = result.total_ns;
    attempt.message.assign (result.message,
                            std::find (result.message, result.message + sizeof (result.message), '\0'));

    if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
        result.struct_size < sizeof (result)) {
      attempt.disposition = CudaSpatialAttempt::InvalidResult;
      attempt.message = "CUDA spatial backend returned an incompatible result";
    } else if (status == KLAYOUT_CUDA_SPATIAL_OK &&
        result.status == KLAYOUT_CUDA_SPATIAL_OK) {
      if ((result.pair_count != 0 && ! result.pair_keys) ||
          result.pair_count > config.max_candidates ||
          result.pair_count > std::numeric_limits<size_t>::max ()) {
        attempt.disposition = CudaSpatialAttempt::InvalidResult;
      } else {
        if (result.pair_count) {
          attempt.pair_keys.assign (result.pair_keys, result.pair_keys + result.pair_count);
        }
        attempt.disposition = CudaSpatialAttempt::Success;
      }
    } else if (status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
               result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
      attempt.disposition = CudaSpatialAttempt::BackendFallback;
    } else {
      attempt.disposition = CudaSpatialAttempt::BackendError;
    }
  } catch (const std::exception &ex) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = ex.what ();
  } catch (...) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = "unknown exception while calling CUDA spatial backend";
  }

  module.release () (&result);
  log_attempt (attempt, subjects.size (), intruders.size ());
  return attempt;
}

bool cuda_spatial_may_attempt (uint64_t subject_count, uint64_t intruder_count)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  if (! module.enabled () || ! module.ready () || subject_count == 0 ||
      intruder_count == 0) {
    return false;
  }
  return subject_count <= std::numeric_limits<uint32_t>::max () &&
         intruder_count <= std::numeric_limits<uint32_t>::max () &&
         subject_count + intruder_count <= std::numeric_limits<uint32_t>::max () &&
         subject_count + intruder_count >= module.min_records ();
}

bool cuda_spatial_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.ready ();
}

} // namespace db
