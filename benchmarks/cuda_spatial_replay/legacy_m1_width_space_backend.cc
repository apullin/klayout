/*
 * Deliberately M1-only ABI-v1 backend used to prove that the host does not
 * mistake the legacy shared entry point for M2 capability.
 */

#include "dbCudaSpatialApi.h"

#include <cstring>

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT uint32_t
klayout_cuda_spatial_abi_version (void)
{
  return KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
}

extern "C" KLAYOUT_CUDA_SPATIAL_EXPORT int
klayout_cuda_spatial_run_m1_width_space_empty_v1 (
  const klayout_cuda_spatial_m1_width_space_request_v1 *,
  klayout_cuda_spatial_m1_width_space_result_v1 *result)
{
  if (! result) {
    return KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
  }
  std::memset (result, 0, sizeof (*result));
  result->abi_version = KLAYOUT_CUDA_SPATIAL_ABI_VERSION;
  result->struct_size = sizeof (*result);
  result->status = KLAYOUT_CUDA_SPATIAL_FALLBACK;
  result->fallback_flags =
    KLAYOUT_CUDA_SPATIAL_FALLBACK_UNSUPPORTED_REQUEST;
  result->disposition = KLAYOUT_CUDA_SPATIAL_M1_WIDTH_SPACE_UNCERTAIN;
  return KLAYOUT_CUDA_SPATIAL_FALLBACK;
}
