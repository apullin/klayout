/*
 * Production adapter for the compact four-stage antenna transaction.
 */

#ifndef KLAYOUT_CUDA_ANTENNA_M1_M4_BACKEND_CUH
#define KLAYOUT_CUDA_ANTENNA_M1_M4_BACKEND_CUH

#include "dbCudaSpatialApi.h"

namespace klayout_cuda {
namespace antenna_m1_m4_backend {

/*
 * Test-fixture helper.  It derives the canonical hierarchy/domain/capture
 * identity and aggregate census from already populated POD record arrays.
 * Production never calls this mutating helper.
 */
bool prepare_test_request(
    klayout_cuda_spatial_antenna_m1_m4_request_v1 *request,
    const char **error = nullptr) noexcept;

}  // namespace antenna_m1_m4_backend
}  // namespace klayout_cuda

#endif
