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
    : m_enabled (false), m_telemetry (false), m_handle (0),
      m_run_bipartite (0), m_run_self (0), m_release (0),
      m_min_records (100000)
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
      m_run_bipartite = reinterpret_cast<klayout_cuda_spatial_run_bipartite_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_run_bipartite_v1"));
      m_run_self = reinterpret_cast<klayout_cuda_spatial_run_self_v1_func> (
        GetProcAddress (reinterpret_cast<HMODULE> (m_handle), "klayout_cuda_spatial_run_self_v1"));
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
      m_run_bipartite = reinterpret_cast<klayout_cuda_spatial_run_bipartite_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_run_bipartite_v1"));
      m_run_self = reinterpret_cast<klayout_cuda_spatial_run_self_v1_func> (
        dlsym (m_handle, "klayout_cuda_spatial_run_self_v1"));
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

    if (m_handle && (! m_run_bipartite || ! m_release) && m_error.empty ()) {
      m_error = "CUDA spatial backend is missing required ABI entry points";
    }

    if (! m_error.empty ()) {
      m_run_bipartite = 0;
      m_run_self = 0;
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
    return m_run_bipartite && m_release;
  }

  bool self_ready () const
  {
    return m_run_self && m_release;
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
    return m_run_bipartite;
  }

  klayout_cuda_spatial_run_self_v1_func run_self () const
  {
    return m_run_self;
  }

  klayout_cuda_spatial_release_result_v1_func release () const
  {
    return m_release;
  }

private:
  bool m_enabled;
  bool m_telemetry;
  void *m_handle;
  klayout_cuda_spatial_run_bipartite_v1_func m_run_bipartite;
  klayout_cuda_spatial_run_self_v1_func m_run_self;
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
                  uint64_t intruders, const char *mode = "bipartite")
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

  tl::info << "CUDA spatial broad phase: mode=" << mode
           << " outcome=" << outcome
           << " subjects=" << subjects << " intruders=" << intruders
           << " pairs=" << attempt.pair_keys.size ()
           << " memberships=" << attempt.membership_count
           << " pair_work=" << attempt.pair_work_count
           << " total_ms=" << (double (attempt.total_ns) / 1.0e6)
           << " fallback_flags=" << attempt.fallback_flags
           << (attempt.message.empty () ? "" : " message=") << attempt.message;
}

klayout_cuda_spatial_config_v1 make_config ()
{
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
  return config;
}

int64_t floor_div_i64 (int64_t value, int64_t divisor)
{
  int64_t quotient = value / divisor;
  if (value % divisor < 0) {
    --quotient;
  }
  return quotient;
}

bool self_memberships_fit (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement, const klayout_cuda_spatial_config_v1 &config,
  uint64_t &membership_count)
{
  membership_count = 0;
  if (records.empty () || enlargement < 0 || config.cell_size == 0 ||
      config.cell_size > uint64_t (std::numeric_limits<int64_t>::max ()) ||
      config.max_cells_per_record == 0 || config.max_memberships == 0) {
    return false;
  }

  const int64_t cell_size = int64_t (config.cell_size);
  for (std::vector<klayout_cuda_spatial_aabb_v1>::const_iterator record =
         records.begin ();
       record != records.end (); ++record) {
    if (record->left > record->right || record->bottom > record->top ||
        record->left < std::numeric_limits<int64_t>::min () + enlargement ||
        record->bottom < std::numeric_limits<int64_t>::min () + enlargement ||
        record->right > std::numeric_limits<int64_t>::max () - enlargement ||
        record->top > std::numeric_limits<int64_t>::max () - enlargement) {
      return false;
    }

    const int64_t x0 = floor_div_i64 (record->left - enlargement, cell_size);
    const int64_t x1 = floor_div_i64 (record->right + enlargement, cell_size);
    const int64_t y0 = floor_div_i64 (record->bottom - enlargement, cell_size);
    const int64_t y1 = floor_div_i64 (record->top + enlargement, cell_size);
    const uint64_t dx = uint64_t (x1) - uint64_t (x0);
    const uint64_t dy = uint64_t (y1) - uint64_t (y0);
    if (dx == std::numeric_limits<uint64_t>::max () ||
        dy == std::numeric_limits<uint64_t>::max ()) {
      return false;
    }

    const uint64_t width = dx + 1;
    const uint64_t height = dy + 1;
    const uint64_t per_record_limit = config.max_cells_per_record;
    if (width > per_record_limit || height > per_record_limit ||
        width > per_record_limit / height) {
      return false;
    }

    const uint64_t record_memberships = width * height;
    if (record_memberships > config.max_memberships - membership_count) {
      return false;
    }
    membership_count += record_memberships;
  }

  return true;
}

void copy_result_metadata (CudaSpatialAttempt &attempt,
                           const klayout_cuda_spatial_result_v1 &result)
{
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
  attempt.message.assign (
    result.message,
    std::find (result.message, result.message + sizeof (result.message), '\0'));
}

bool valid_pair_keys (const klayout_cuda_spatial_result_v1 &result,
                      const klayout_cuda_spatial_config_v1 &config,
                      uint64_t subject_count, uint64_t intruder_count,
                      bool bipartite)
{
  if ((result.pair_count != 0 && ! result.pair_keys) ||
      result.pair_count > config.max_candidates ||
      result.pair_count > std::numeric_limits<size_t>::max ()) {
    return false;
  }

  uint64_t previous = 0;
  const uint64_t total_count = subject_count + intruder_count;
  for (uint64_t i = 0; i < result.pair_count; ++i) {
    const uint64_t key = result.pair_keys[i];
    const uint64_t first = key >> 32;
    const uint64_t second = key & uint64_t (0xffffffff);
    if ((i != 0 && key <= previous) || first == 0 || second == 0) {
      return false;
    }
    if (bipartite) {
      if (first > subject_count || second <= subject_count ||
          second > total_count) {
        return false;
      }
    } else if (first >= second || second > subject_count) {
      return false;
    }
    previous = key;
  }
  return true;
}

void interpret_result (CudaSpatialAttempt &attempt, int status,
                       const klayout_cuda_spatial_result_v1 &result,
                       const klayout_cuda_spatial_config_v1 &config,
                       uint64_t subject_count, uint64_t intruder_count,
                       bool bipartite)
{
  copy_result_metadata (attempt, result);
  if (result.abi_version != KLAYOUT_CUDA_SPATIAL_ABI_VERSION ||
      result.struct_size < sizeof (result)) {
    attempt.disposition = CudaSpatialAttempt::InvalidResult;
    attempt.message = "CUDA spatial backend returned an incompatible result";
    return;
  }

  if (status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.status == KLAYOUT_CUDA_SPATIAL_OK) {
    if (result.fallback_flags != KLAYOUT_CUDA_SPATIAL_FALLBACK_NONE ||
        ! valid_pair_keys (result, config, subject_count, intruder_count,
                           bipartite)) {
      attempt.disposition = CudaSpatialAttempt::InvalidResult;
      attempt.message = "CUDA spatial backend returned invalid pair keys";
      return;
    }
    if (result.pair_count) {
      attempt.pair_keys.assign (result.pair_keys,
                                result.pair_keys + result.pair_count);
    }
    attempt.disposition = CudaSpatialAttempt::Success;
    return;
  }

  if (result.pair_count != 0 || result.pair_keys) {
    attempt.disposition = CudaSpatialAttempt::InvalidResult;
    attempt.message = "CUDA spatial backend published pairs after failure";
  } else if (status == KLAYOUT_CUDA_SPATIAL_FALLBACK ||
             result.status == KLAYOUT_CUDA_SPATIAL_FALLBACK) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
  } else {
    attempt.disposition = CudaSpatialAttempt::BackendError;
  }
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

  klayout_cuda_spatial_config_v1 config = make_config ();

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
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;

  try {
    const int status = module.run () (&request, &result);
    interpret_result (attempt, status, result, config, subjects.size (),
                      intruders.size (), true);
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

CudaSpatialAttempt cuda_spatial_try_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
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
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }
  if (! module.self_ready ()) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
    attempt.fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    attempt.message = "CUDA spatial backend has no self-AABB entry point";
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }

  if (records.size () < module.min_records ()) {
    attempt.disposition = CudaSpatialAttempt::BelowThreshold;
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }
  if (records.empty () || enlargement < 0 ||
      records.size () > std::numeric_limits<uint32_t>::max ()) {
    attempt.disposition = CudaSpatialAttempt::BackendFallback;
    attempt.fallback_flags = KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    log_attempt (attempt, records.size (), 0, "self");
    return attempt;
  }

  klayout_cuda_spatial_config_v1 config = make_config ();

  klayout_cuda_spatial_request_v1 request;
  std::memset (&request, 0, sizeof (request));
  request.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  request.struct_size = sizeof (request);
  request.subjects = records.data ();
  request.subject_count = records.size ();
  request.enlargement = enlargement;
  request.config = &config;

  klayout_cuda_spatial_result_v1 result;
  std::memset (&result, 0, sizeof (result));
  result.abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result.struct_size = sizeof (result);
  result.status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;

  try {
    const int status = module.run_self () (&request, &result);
    interpret_result (attempt, status, result, config, records.size (), 0,
                      false);
  } catch (const std::exception &ex) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = ex.what ();
  } catch (...) {
    attempt.disposition = CudaSpatialAttempt::BackendError;
    attempt.message = "unknown exception while calling CUDA spatial backend";
  }

  module.release () (&result);
  log_attempt (attempt, records.size (), 0, "self");
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

bool cuda_spatial_may_attempt_self (uint64_t record_count)
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.ready () && module.self_ready () &&
         record_count != 0 &&
         record_count <= std::numeric_limits<uint32_t>::max () &&
         record_count >= module.min_records ();
}

bool cuda_spatial_preflight_self (
  const std::vector<klayout_cuda_spatial_aabb_v1> &records,
  int64_t enlargement, uint64_t &membership_count)
{
  membership_count = 0;
  if (! cuda_spatial_may_attempt_self (records.size ())) {
    return false;
  }
  return self_memberships_fit (records, enlargement, make_config (),
                               membership_count);
}

bool cuda_spatial_requested ()
{
  CudaSpatialModule &module = cuda_spatial_module ();
  return module.enabled () && module.ready ();
}

} // namespace db
