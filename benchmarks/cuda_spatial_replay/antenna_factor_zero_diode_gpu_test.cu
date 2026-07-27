/*
 * Directed exact/fail-closed tests for factor-zero diode witnesses.
 */

#include "antenna_clean_certificate_gpu.cuh"
#include "antenna_factor_zero_diode_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ac = klayout_cuda::antenna_connectivity;
namespace afd = klayout_cuda::antenna_factor_zero_diode;
namespace cert = klayout_cuda::antenna_clean_certificate;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

ac::RectI64 rectangle(
    std::int64_t left, std::int64_t bottom,
    std::int64_t right, std::int64_t top,
    std::uint32_t owner, std::uint32_t domain)
{
  ac::RectI64 result;
  result.left = left;
  result.bottom = bottom;
  result.right = right;
  result.top = top;
  result.owner = owner;
  result.domain = domain;
  return result;
}

afd::Config diode_config()
{
  afd::Config config;
  config.device = 0;
  config.bin_size = 16;
  config.limits.max_nodes = 1000;
  config.limits.max_rectangles = 1000;
  config.limits.max_memberships = 100000;
  config.limits.max_pair_occurrences = 100000;
  config.limits.max_unique_candidates = 100000;
  config.limits.max_cell_members = 1000;
  config.limits.max_pair_tests_per_cell = 100000;
  config.limits.max_total_pair_tests = 100000;
  config.limits.max_dsu_iterations = 128;
  config.limits.max_device_bytes = UINT64_C(1) << 30;
  return config;
}

void filter_witnesses_test()
{
  /*
   * CONTACT 0 overlaps NWELL, CONTACT 1 only boundary-touches NWELL and
   * CONTACT 2 is disjoint.  Closed-touch rejection is deliberately stricter
   * than the positive-overlap Boolean and therefore fail-closed.
   */
  const std::vector<ac::RectI64> geometry = {
      rectangle(0, 0, 10, 10, 0, 0),
      rectangle(20, 0, 30, 10, 1, 0),
      rectangle(40, 0, 50, 10, 2, 0),
      rectangle(5, 5, 6, 6, 3, 1),
      rectangle(30, 0, 35, 10, 4, 1)};
  thrust::device_vector<ac::RectI64> device_geometry(
      geometry);
  thrust::device_vector<std::uint32_t> witnesses(
      3, 1u);
  afd::ContactWitnessCensus census;
  require(
      afd::filter_contact_witnesses(
          diode_config(), std::move(device_geometry),
          3, 2, &witnesses, &census) ==
          afd::Status::success,
      "directed CONTACT/NWELL filter failed");
  const std::vector<std::uint32_t> actual(
      witnesses.begin(), witnesses.end());
  require(
      actual == std::vector<std::uint32_t>({0, 0, 1}),
      "CONTACT/NWELL filter accepted a touched contact");
  require(
      census.local_nplus_active_contacts == 3 &&
          census.well_rejected_contacts == 2 &&
          census.exact_witness_contacts == 1 &&
          census.exact_edges == 2,
      "CONTACT/NWELL exact census changed");
}

cert::Config certificate_config()
{
  cert::Config config;
  config.device = 0;
  config.grid_cell_size = 16;
  config.limits.max_live_device_bytes =
      UINT64_C(256) * 1024 * 1024;
  config.limits.max_annotation_owners = 1000;
  config.limits.max_labels = 1000;
  config.limits.max_poly_tiles = 1000;
  config.limits.max_active_tiles = 1000;
  config.limits.max_metal_tiles = 1000;
  config.limits.max_grid_cells = 10000;
  config.limits.max_active_memberships = 10000;
  config.limits.max_query_visits = 100000;
  config.limits.max_cell_members = 1000;
  return config;
}

cert::CheckpointCensus evaluate_with_contacts(
    cert::Certificate *certificate,
    thrust::device_vector<std::uint32_t> *contacts,
    const std::uint32_t *labels,
    const ac::RectI64 *metal)
{
  cert::FactorZeroDiodeDeviceView view;
  view.contact_present =
      thrust::raw_pointer_cast(contacts->data());
  view.owner_begin = 2;
  view.count = contacts->size();
  cert::CheckpointCensus census;
  require(
      certificate->evaluate_checkpoint(
          cert::MetalLevel::metal1, metal, 2, labels, 6,
          &census, &view) == cert::Status::success,
      "factor-zero checkpoint failed");
  return census;
}

void exemptions_do_not_join_roots_test()
{
  const std::vector<ac::RectI64> poly = {
      rectangle(0, 0, 2, 2, 0, 0),
      rectangle(10, 0, 12, 2, 1, 0)};
  const std::vector<ac::RectI64> active = poly;
  const std::vector<ac::RectI64> metal = {
      rectangle(0, 20, 20, 120, 4, 5),
      rectangle(30, 20, 50, 120, 5, 5)};
  const std::vector<std::uint32_t> labels =
      {0, 1, 0, 1, 0, 1};
  thrust::device_vector<ac::RectI64> device_poly(poly);
  thrust::device_vector<ac::RectI64> device_active(active);
  thrust::device_vector<ac::RectI64> device_metal(metal);
  thrust::device_vector<std::uint32_t> device_labels(labels);

  cert::Certificate certificate(certificate_config());
  cert::GateCensus gate;
  require(
      certificate.build_gate_census(
          thrust::raw_pointer_cast(device_poly.data()),
          device_poly.size(),
          thrust::raw_pointer_cast(device_active.data()),
          device_active.size(), 2, &gate) ==
          cert::Status::success,
      "factor-zero gate setup failed");

  thrust::device_vector<std::uint32_t> one_contact(
      std::vector<std::uint32_t>({1, 0}));
  cert::CheckpointCensus one = evaluate_with_contacts(
      &certificate, &one_contact,
      thrust::raw_pointer_cast(device_labels.data()),
      thrust::raw_pointer_cast(device_metal.data()));
  require(
      !one.clean_certificate &&
          one.diode_exempt_roots == 1 &&
          one.uncertain_roots == 1,
      "one diode witness exempted another conductor root");

  thrust::device_vector<std::uint32_t> both_contacts(
      std::vector<std::uint32_t>({1, 1}));
  cert::CheckpointCensus both = evaluate_with_contacts(
      &certificate, &both_contacts,
      thrust::raw_pointer_cast(device_labels.data()),
      thrust::raw_pointer_cast(device_metal.data()));
  require(
      both.clean_certificate &&
          both.diode_exempt_roots == 2 &&
          both.uncertain_roots == 0 &&
          both.ratio_certified_roots == 0,
      "two factor-zero witnesses did not exempt exactly two roots");
  const std::vector<std::uint32_t> labels_after(
      device_labels.begin(), device_labels.end());
  require(
      labels_after == labels,
      "factor-zero annotation changed conductor labels");

  cert::FactorZeroDiodeDeviceView malformed;
  malformed.contact_present =
      thrust::raw_pointer_cast(both_contacts.data());
  malformed.owner_begin = 5;
  malformed.count = 2;
  cert::CheckpointCensus sentinel;
  sentinel.labels = 123456;
  const cert::CheckpointCensus expected = sentinel;
  require(
      certificate.evaluate_checkpoint(
          cert::MetalLevel::metal1,
          thrust::raw_pointer_cast(device_metal.data()), 2,
          thrust::raw_pointer_cast(device_labels.data()), 6,
          &sentinel, &malformed) ==
          cert::Status::malformed_input,
      "out-of-range diode view was not rejected");
  require(
      std::memcmp(
          &sentinel, &expected, sizeof(sentinel)) == 0,
      "malformed diode view changed checkpoint output");
}

}  // namespace

int main()
{
  try {
    filter_witnesses_test();
    exemptions_do_not_join_roots_test();
    std::cout
        << "antenna_factor_zero_diode_gpu_test: PASS\n";
    return 0;
  } catch (const std::exception &exception) {
    std::cerr
        << "antenna_factor_zero_diode_gpu_test: FAIL: "
        << exception.what() << "\n";
    return 1;
  }
}
