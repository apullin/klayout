/*
 * Pointer-free device seam for the exact Manhattan-union replay.
 *
 * The first milestone consumes normalized, already-expanded rectangles.  The
 * POLY34 and VIA1-stack backends already produce this shape after applying one
 * of KLayout's eight orthogonal transforms.  Keeping that seam explicit lets a
 * future resident pipeline write these records directly in device memory
 * without a host round trip.
 */

#ifndef KLAYOUT_CUDA_MANHATTAN_UNION_FORMAT_CUH
#define KLAYOUT_CUDA_MANHATTAN_UNION_FORMAT_CUH

#include <cstdint>
#include <type_traits>

namespace klayout_cuda {
namespace manhattan_union {

struct RectI64
{
  std::int64_t left;
  std::int64_t bottom;
  std::int64_t right;
  std::int64_t top;
  std::uint64_t source_token;
  std::uint64_t context_token;
};

enum class SegmentAxis : std::uint32_t
{
  horizontal = 0,
  vertical = 1,
  invalid = 2
};

/*
 * side is the outward normal on the varying coordinate:
 *   horizontal -1/+1 -> bottom/top
 *   vertical   -1/+1 -> left/right
 *
 * Directed sides preserve exact boundary topology at point-touching corners;
 * two collinear segments with opposite interior sides must not be coalesced.
 */
struct DirectedSegmentI64
{
  std::int64_t fixed;
  std::int64_t lo;
  std::int64_t hi;
  std::int32_t side;
  SegmentAxis axis;
};

static_assert(std::is_trivially_copyable<RectI64>::value,
              "rectangle replay record must remain POD");
static_assert(std::is_trivially_copyable<DirectedSegmentI64>::value,
              "boundary replay record must remain POD");
static_assert(sizeof(RectI64) == 48, "unexpected rectangle record padding");
static_assert(sizeof(DirectedSegmentI64) == 32,
              "unexpected boundary record padding");

}  // namespace manhattan_union
}  // namespace klayout_cuda

#endif
