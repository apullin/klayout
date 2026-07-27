/*
 * Capture-only ANTENNA.M1-through-M4 shared backend.
 *
 * Build this file with antenna_m1_m4_capture_file.cc as a standalone shared
 * object.  It captures the exact qualified request and deliberately returns
 * FALLBACK so KLayout executes the unchanged literal CPU antenna path.
 */

#include "antenna_m1_m4_capture_file.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

void set_message (klayout_cuda_spatial_antenna_m1_m4_result_v1 *result,
                  const std::string &message)
{
  std::snprintf(result->message, sizeof(result->message), "%s",
                message.c_str());
}

void initialize_fallback (
    klayout_cuda_spatial_antenna_m1_m4_result_v1 *result)
{
  std::memset(result, 0, sizeof(*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof(*result);
  result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
  result->fallback_flags =
      KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition =
      KLAYOUT_CUDA_SPATIAL_ANTENNA_M1_M4_UNCERTAIN;
}

std::mutex &capture_mutex ()
{
  static std::mutex mutex;
  return mutex;
}

}  // namespace

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT std::uint32_t
klayout_cuda_spatial_abi_version (void)
{
  return KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_antenna_m1_m4_empty_v1 (
    const klayout_cuda_spatial_antenna_m1_m4_request_v1 *request,
    klayout_cuda_spatial_antenna_m1_m4_result_v1 *result)
{
  if (!result) return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  initialize_fallback(result);
  if (!request) {
    result->status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
    set_message(result, "capture backend received a null request");
    return static_cast<int>(result->status);
  }

  try {
    const char *path =
        std::getenv("KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_OUT");
    if (!path || !*path) {
      set_message(
          result,
          "KLAYOUT_CUDA_ANTENNA_M1_M4_CAPTURE_OUT is unset; CPU fallback");
      return KLAYOUT_CUDA_SPATIAL_FALLBACK;
    }

    std::string error;
    bool captured = false;
    {
      std::lock_guard<std::mutex> lock(capture_mutex());
      captured =
          klayout_cuda::antenna_m1_m4_capture::dump_request(
              path, *request, &error);
    }
    if (!captured) {
      result->fallback_flags =
          KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
      set_message(result, std::string("capture failed: ") +
                              (error.empty() ? "unknown error" : error));
    } else {
      set_message(result,
                  "qualified ANTENNA.M1-M4 request captured; deliberate CPU "
                  "fallback");
    }
  } catch (...) {
    result->fallback_flags =
        KLAYOUT_CUDA_SPATIAL_FALLBACK_INTERNAL_INVARIANT;
    std::snprintf(
        result->message, sizeof(result->message), "%s",
        "capture backend caught an exception; CPU fallback");
  }
  return KLAYOUT_CUDA_SPATIAL_FALLBACK;
}
