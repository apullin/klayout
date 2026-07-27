/*
 * Exact host oracle and contract for quotienting singleton antenna geometry.
 *
 * This module changes only the graph used to discover connectivity.  Every
 * owner remains present for annotations and component reductions.  Owners
 * represented by one exactly equal transformed rectangle in a self-connected
 * domain have identical closed-touch neighborhoods, so they may be joined to
 * the minimum owner and represented by one rectangle before spatial
 * membership construction.  Multi-rectangle owners remain on the exact
 * exception path.
 */

#ifndef KLAYOUT_CUDA_ANTENNA_GEOMETRY_QUOTIENT_H
#define KLAYOUT_CUDA_ANTENNA_GEOMETRY_QUOTIENT_H

#include "antenna_connectivity_gpu.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace klayout_cuda {
namespace antenna_geometry_quotient {

namespace ac = antenna_connectivity;

enum class Status : std::uint32_t
{
  success = 0,
  invalid_configuration,
  malformed_input,
  capacity_exceeded,
  cuda_error,
  host_error
};

struct Limits
{
  std::uint64_t max_owners = UINT32_MAX;
  std::uint64_t max_input_rectangles = UINT32_MAX;
  std::uint64_t max_classes = UINT32_MAX;
  std::uint64_t max_class_members = UINT32_MAX;
  std::uint64_t max_exception_rectangles = UINT32_MAX;
  std::uint64_t max_work_rectangles = UINT32_MAX;
  std::uint64_t max_star_edges = UINT32_MAX;
  std::uint64_t max_weighted_internal_pairs = UINT64_MAX;
};

struct Config
{
  std::uint32_t domain_count = 0;
  std::array<std::uint64_t, ac::kMaximumDomains> relation_rows{};
  std::uint32_t owner_begin = 0;
  std::uint32_t owner_count = 0;
  Limits limits;
};

struct GeometryClass
{
  // Exact transformed representative.  owner is the minimum class owner.
  ac::RectI64 rectangle;
  std::uint64_t member_begin = 0;
  std::uint32_t multiplicity = 0;
};

struct Census
{
  std::uint64_t input_rectangles = 0;
  std::uint64_t owners = 0;
  std::uint64_t singleton_owners = 0;
  std::uint64_t exception_owners = 0;
  std::uint64_t exception_rectangles = 0;
  std::uint64_t geometry_classes = 0;
  std::uint64_t collapsed_rectangles = 0;
  std::uint64_t work_rectangles = 0;
  std::uint64_t star_edges = 0;
  std::uint64_t weighted_internal_pairs = 0;

  // Exact owner-edge contribution of identical singleton classes.  Only the
  // canonical min(domain_a, domain_b) * 64 + max(...) slot is populated.
  std::array<std::uint64_t, ac::kRelationSlots>
      weighted_internal_pairs_by_relation{};
};

struct Result
{
  /*
   * One rectangle per enabled singleton equivalence class.  A singleton in a
   * domain without diagonal connectivity forms a class of multiplicity one.
   */
  std::vector<ac::RectI64> representative_rectangles;
  std::vector<GeometryClass> classes;
  // Global owner IDs, contiguous by class and beginning at member_begin.
  std::vector<std::uint32_t> class_members;
  // All rectangles of every multi-rectangle owner, unchanged and in input
  // order.  These must retain the existing exact tile/exception treatment.
  std::vector<ac::RectI64> exception_rectangles;
  /*
   * One global canonical-min parent seed per owner in
   * [owner_begin, owner_begin + owner_count).  Singleton class members point
   * directly to their representative; exception owners point to themselves.
   */
  std::vector<std::uint32_t> parent_seeds;
  // One validated domain per owner, in the same local-owner order.
  std::vector<std::uint32_t> owner_domains;
  /*
   * Exact owner-pair weight carried by a spatial edge whose endpoint is this
   * owner.  A class representative carries the class multiplicity, collapsed
   * members carry zero (they never enter the spatial stream), and every
   * exception owner carries one.  This makes a representative-representative
   * edge weigh n*m and an exception-representative edge weigh m without
   * materializing the original owner-pair universe.
   */
  std::vector<std::uint32_t> owner_multiplicities;
  Census census;
};

/*
 * Builds into temporary storage and replaces output only on success.
 * rectangles may be null only when rectangle_count is zero, which is rejected
 * because every configured owner must have at least one rectangle.
 */
Status build(
    const Config &config, const ac::RectI64 *rectangles,
    std::uint64_t rectangle_count, Result *output) noexcept;

/*
 * Checked exact census weights used by the later device seam.  Distinct
 * singleton classes contribute first_multiplicity * second_multiplicity
 * owner pairs.  One identical class contributes C(multiplicity, 2).
 */
Status checked_cross_weight(
    std::uint64_t first_multiplicity,
    std::uint64_t second_multiplicity,
    std::uint64_t *weight) noexcept;

Status checked_internal_weight(
    std::uint64_t multiplicity, std::uint64_t *weight) noexcept;

const char *status_string(Status status) noexcept;

}  // namespace antenna_geometry_quotient
}  // namespace klayout_cuda

#endif
