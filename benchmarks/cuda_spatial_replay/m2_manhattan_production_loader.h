/*
 * Checked compact production input for the exact CUDA Manhattan union.
 *
 * This C++ seam intentionally returns hierarchy contexts and local rectangle
 * templates, not a 1.1-GiB host-expanded stream.  A resident CUDA consumer can
 * upload these POD vectors and expand directly into
 * manhattan_union_format.cuh's RectI64 device buffer.
 */

#ifndef KLAYOUT_CUDA_M2_MANHATTAN_PRODUCTION_LOADER_H
#define KLAYOUT_CUDA_M2_MANHATTAN_PRODUCTION_LOADER_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace klayout_cuda {
namespace m2_production {

struct ResolvedContextI64
{
  std::uint32_t cell;
  std::uint32_t transform;
  std::int64_t tx;
  std::int64_t ty;
};

struct RectTemplateI64
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint64_t source_token;
};

struct CellTemplate
{
  std::uint64_t rectangle_begin;
  std::uint32_t rectangle_count;
  std::uint32_t polygon_count;
  std::uint32_t l_shape_count;
  std::uint32_t reserved;
};

static_assert(std::is_trivially_copyable<ResolvedContextI64>::value,
              "M2 production contexts must remain POD");
static_assert(std::is_trivially_copyable<RectTemplateI64>::value,
              "M2 production rectangle templates must remain POD");
static_assert(std::is_trivially_copyable<CellTemplate>::value,
              "M2 production cell templates must remain POD");
static_assert(sizeof(ResolvedContextI64) == 24,
              "unexpected M2 production context padding");
static_assert(sizeof(RectTemplateI64) == 40,
              "unexpected M2 production rectangle-template padding");
static_assert(sizeof(CellTemplate) == 24,
              "unexpected M2 production cell-template padding");

struct LoadOptions
{
  // Mandatory independent qualification; KACT's embedded digest is not
  // accepted as its own provenance assertion.
  std::string expected_scene_sha256;
  std::uint64_t expected_flat_polygons = 0;
  std::uint64_t expected_flat_rectangles = 0;
  std::uint64_t max_contexts = UINT64_C(4000000);
  std::uint64_t max_rectangles = UINT64_C(32000000);
};

struct CompactScene
{
  std::vector<ResolvedContextI64> contexts;
  // Canonical indices into contexts for cells with local M2 rectangles.
  std::vector<std::uint32_t> m2_contexts;
  // One device output offset per entry of m2_contexts.
  std::vector<std::uint64_t> rectangle_offsets;
  std::vector<CellTemplate> cells;
  std::vector<RectTemplateI64> rectangles;

  std::string scene_sha256;
  std::uint64_t flat_polygons = 0;
  std::uint64_t flat_l_shapes = 0;
  std::uint64_t flat_rectangles = 0;
  std::uint64_t local_polygons = 0;
  std::uint64_t local_l_shapes = 0;
};

// Throws std::runtime_error on every unsupported semantic, malformed record,
// digest mismatch, overflow, conservation failure, or capacity exhaustion.
CompactScene load_kact_templates(const std::string &path,
                                 const LoadOptions &options);

}  // namespace m2_production
}  // namespace klayout_cuda

#endif
