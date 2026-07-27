/*
 * Exact one-sided factor-zero diode witnesses for the staged antenna core.
 *
 * A caller first proves that selected CONTACT owners lie inside raw NPLUS
 * and ACTIVE.  This module removes every selected owner whose complete
 * CONTACT component has any closed touch with raw NWELL.  Every survivor
 * therefore contains positive area in
 *
 *   NPLUS intersect (ACTIVE subtract NWELL)
 *
 * and is a sound factor-zero diode witness.  Missed witnesses only cause
 * conservative fallback.
 */

#ifndef KLAYOUT_CUDA_ANTENNA_FACTOR_ZERO_DIODE_GPU_CUH
#define KLAYOUT_CUDA_ANTENNA_FACTOR_ZERO_DIODE_GPU_CUH

#include "antenna_connectivity_gpu.cuh"

#include <cstdint>

#include <thrust/device_vector.h>

namespace klayout_cuda {
namespace antenna_factor_zero_diode {

enum class Status : std::uint32_t
{
  success = 0,
  invalid_configuration,
  malformed_input,
  capacity_exceeded,
  convergence_failure,
  cuda_error,
  host_error
};

struct Config
{
  int device = 0;
  std::int64_t bin_size = 1000;
  antenna_connectivity::Limits limits;
};

struct ContactWitnessCensus
{
  std::uint64_t contact_owners = 0;
  std::uint64_t nwell_owners = 0;
  std::uint64_t rectangles = 0;
  std::uint64_t local_nplus_active_contacts = 0;
  std::uint64_t well_rejected_contacts = 0;
  std::uint64_t exact_witness_contacts = 0;
  std::uint64_t memberships = 0;
  std::uint64_t occupied_cells = 0;
  std::uint64_t pair_occurrences = 0;
  std::uint64_t unique_candidates = 0;
  std::uint64_t exact_edges = 0;
  std::uint32_t dsu_iterations = 0;
};

/*
 * contact_then_nwell contains two exact rectangle covers.  CONTACT owners
 * occupy [0, contact_owner_count), domain 0.  NWELL owners occupy the
 * immediately following range, domain 1.  The function consumes the
 * geometry.  contact_present has one uint32 per CONTACT owner and initially
 * marks the owners already proven inside NPLUS and ACTIVE.  On success it is
 * filtered in place to the exact one-sided witnesses described above.
 */
Status filter_contact_witnesses(
    const Config &config,
    thrust::device_vector<antenna_connectivity::RectI64>
        &&contact_then_nwell,
    std::uint64_t contact_owner_count,
    std::uint64_t nwell_owner_count,
    thrust::device_vector<std::uint32_t> *contact_present,
    ContactWitnessCensus *census) noexcept;

const char *status_string(Status status) noexcept;

}  // namespace antenna_factor_zero_diode
}  // namespace klayout_cuda

#endif
