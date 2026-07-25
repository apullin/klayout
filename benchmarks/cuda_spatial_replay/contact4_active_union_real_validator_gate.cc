/*
 * Focused bridge gate for the fused CONTACT.4 backend and its production
 * host-side result validator.
 *
 * Reuse the backend smoke's exact scene serializer in this translation unit
 * so this gate cannot drift to a second, subtly different digest builder.
 * Its main is renamed; this file supplies the validator-specific entry point.
 */
#define main contact4_active_union_backend_smoke_entry_not_used
#include "contact4_active_union_backend_smoke.cc"
#undef main

#include "dbCudaSpatialBackend.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct BackendOutcome
{
  Request request{};
  Result result{};
  int status = KLAYOUT_CUDA_SPATIAL_BAD_ARGUMENT;
};

BackendOutcome run_backend(
    const StoredScene &active, const StoredScene &contact)
{
  BackendOutcome outcome;
  outcome.request = make_request(active.scene, contact.scene);
  outcome.status =
      klayout_cuda_spatial_run_contact4_active_union_empty_v1(
          &outcome.request, &outcome.result);
  require(
      outcome.status == static_cast<int>(outcome.result.status),
      "backend return/result status mismatch");
  return outcome;
}

void require_production_validator_accepts(
    const BackendOutcome &outcome, const char *fixture)
{
  std::string error = "validator did not clear its diagnostic";
  const bool accepted =
      db::cuda_spatial_validate_contact4_active_union_result(
          outcome.request, outcome.result, outcome.status, &error);
  require(
      accepted,
      std::string(fixture) +
          " real backend result failed production validation: " + error);
  require(
      error.empty(),
      std::string(fixture) +
          " production validator accepted but retained a diagnostic");
}

void require_production_validator_rejects(
    const Request &request, const Result &result, int status,
    const char *expected_error, const char *fixture)
{
  std::string error;
  const bool accepted =
      db::cuda_spatial_validate_contact4_active_union_result(
          request, result, status, &error);
  require(
      !accepted,
      std::string(fixture) + " malformed result was accepted");
  require(
      error == expected_error,
      std::string(fixture) + " unexpected validator diagnostic: " + error);
}

bool is_consumable_empty_proof(const Result &result)
{
  return
      result.status == KLAYOUT_CUDA_SPATIAL_OK &&
      result.disposition ==
          KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE &&
      result.raw_hit_count == 0 && result.uncertain_count == 0;
}

}  // namespace

int main()
{
  try {
    const char active_domain[9] =
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_DIGEST_DOMAIN;
    const char contact_domain[9] =
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_DIGEST_DOMAIN;

    StoredScene active = make_scene(
        {{0, 0, 100, 100}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_ROLE, 1,
        active_domain, 0);
    StoredScene strict_contact = make_scene(
        {{80, 40, 90, 60}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
        contact_domain, 0);
    StoredScene hit_contact = make_scene(
        {{85, 40, 95, 60}},
        KLAYOUT_CUDA_SPATIAL_CONTACT4_CONTACT_ROLE, 10,
        contact_domain, 0);

    const BackendOutcome complete =
        run_backend(active, strict_contact);
    require_production_validator_accepts(complete, "strict-10");
    require(
        is_consumable_empty_proof(complete.result),
        "validated COMPLETE result was not consumable");

    const BackendOutcome raw_hits = run_backend(active, hit_contact);
    require_production_validator_accepts(raw_hits, "gap-5");
    require(
        raw_hits.result.disposition ==
            KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_RAW_HITS &&
        raw_hits.result.raw_hit_count != 0 &&
        raw_hits.result.uncertain_count == 0,
        "validated gap-5 result was not RAW_HITS");
    require(
        !is_consumable_empty_proof(raw_hits.result),
        "validated RAW_HITS result became consumable");

    Result bad_echo = complete.result;
    bad_echo.active.scene_digest[0] ^= 1;
    require_production_validator_rejects(
        complete.request, bad_echo, complete.status,
        "CUDA CONTACT.4 ACTIVE-union backend returned a mismatched proof echo",
        "bad-echo");

    Result bad_counter = complete.result;
    ++bad_counter.event_count;
    require_production_validator_rejects(
        complete.request, bad_counter, complete.status,
        "CUDA CONTACT.4 ACTIVE-union backend returned impossible proof "
        "counters",
        "bad-counter");

    Result bad_disposition = raw_hits.result;
    bad_disposition.disposition =
        KLAYOUT_CUDA_SPATIAL_CONTACT4_ACTIVE_UNION_COMPLETE;
    require_production_validator_rejects(
        raw_hits.request, bad_disposition, raw_hits.status,
        "CUDA CONTACT.4 ACTIVE-union backend returned an inconsistent "
        "disposition",
        "bad-disposition");

    std::cout
        << "CONTACT4_ACTIVE_UNION_REAL_VALIDATOR_GATE PASS"
        << " complete_rectangles=" << complete.result.rectangle_count
        << " complete_boundary=" << complete.result.boundary_segment_count
        << " complete_candidates=" << complete.result.candidate_pair_count
        << " raw_hit_candidates=" << raw_hits.result.candidate_pair_count
        << " raw_hits=" << raw_hits.result.raw_hit_count
        << " malformed=3"
        << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr
        << "CONTACT4_ACTIVE_UNION_REAL_VALIDATOR_GATE FAIL: "
        << error.what() << "\n";
    return 1;
  }
}
