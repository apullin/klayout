/*
 * Device-resident exact singleton geometry quotient.
 *
 * This is the production-shaped pre-membership seam for antenna
 * connectivity.  It validates an appended owner range, groups fully
 * transformed singleton rectangles by exact (domain, coordinates), and
 * returns the reduced physical work stream plus the metadata required to
 * preserve the original logical owner graph.
 */

#ifndef KLAYOUT_CUDA_ANTENNA_GEOMETRY_QUOTIENT_GPU_CUH
#define KLAYOUT_CUDA_ANTENNA_GEOMETRY_QUOTIENT_GPU_CUH

#include "antenna_geometry_quotient.h"

#include <cstdint>

#include <thrust/device_vector.h>

namespace klayout_cuda {
namespace antenna_geometry_quotient {

struct DeviceLimits
{
  std::uint64_t max_device_bytes =
      UINT64_C(10240) * 1024 * 1024;
  std::uint64_t min_device_free_after_bytes =
      UINT64_C(256) * 1024 * 1024;
  // Conservative temporary-storage reserve before sorting singleton indices.
  std::uint64_t sort_scratch_fixed_guard_bytes =
      UINT64_C(64) * 1024 * 1024;
};

struct DeviceConfig
{
  Config quotient;
  int device = 0;
  DeviceLimits device_limits;
};

struct DeviceResult
{
  /*
   * Physical stream consumed by spatial membership construction.  Exact
   * singleton representatives come first in sorted geometry order; unchanged
   * multi-rectangle exceptions follow in original input order.
   */
  thrust::device_vector<ac::RectI64> work_rectangles;
  /*
   * Logical rectangle multiplicity aligned 1:1 with work_rectangles.
   * Representative entries carry their class size; every exception tile is 1.
   */
  thrust::device_vector<std::uint32_t> rectangle_multiplicities;
  // Global canonical-min parent seed for each local owner.
  thrust::device_vector<std::uint32_t> parent_seeds;
  // Validated input domain for each local owner.
  thrust::device_vector<std::uint32_t> owner_domains;
  /*
   * Spatial endpoint weight for each local owner: representative=n,
   * collapsed=0, exception=1.
   */
  thrust::device_vector<std::uint32_t> owner_multiplicities;
  Census census;
};

/*
 * Builds all output in temporary device storage and replaces output only on
 * success.  Input remains owned by the caller and is never modified.  The
 * caller may release it immediately after a successful call.
 */
Status build_device(
    const DeviceConfig &config,
    const ac::RectI64 *device_rectangles,
    std::uint64_t rectangle_count,
    DeviceResult *output) noexcept;

Status build_device(
    const DeviceConfig &config,
    const thrust::device_vector<ac::RectI64> &rectangles,
    DeviceResult *output) noexcept;

}  // namespace antenna_geometry_quotient
}  // namespace klayout_cuda

#endif
