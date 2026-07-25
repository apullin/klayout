/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

// Standalone exact clean-only certificate for FreePDK45 METAL2.4.
//
// KACTSCN1 layer zero is raw M2 and layer one is raw VIA2.  We reuse the
// checked hierarchy loader, Manhattan polygon decomposition, transform
// expansion, and uniform-grid construction from the projection-enclosure
// proof.  For every rectangular VIA2, this proof asks whether the union of
// nearby raw M2 rectangles covers either:
//
//   [via.left - 70, via.right + 70] x [via.bottom, via.top], or
//   [via.left, via.right] x [via.bottom - 70, via.top + 70].
//
// Those rectangles are exactly "the via plus both opposite 35 nm projection
// strips" in X and Y respectively at the qualified 0.5 nm DBU.  Coverage is
// exact: candidate X endpoints partition the query into slabs, and each slab
// must have gap-free Y coverage.  No sampling or floating point is used.
//
// This is deliberately a clean-only transaction.  A non-rectangle VIA2,
// unsupported scene, arithmetic failure, digest mismatch, candidate overflow,
// allocation failure, or device invariant failure returns UNCERTAIN so the
// caller can run the untouched CPU rule.

#define KLAYOUT_CUDA_PROJECTION_ENCLOSURE_NO_MAIN
#include "projection_enclosure_scene_island.cu"
#undef KLAYOUT_CUDA_PROJECTION_ENCLOSURE_NO_MAIN

namespace {

constexpr std::uint32_t kM2HardMaxQueryCandidates = 128;
constexpr std::uint32_t kM2QueryCapacityExceeded = 1u << 4;
constexpr std::uint64_t kM2DefaultMaxCpuPairs = UINT64_C(10000000);

struct M2QueryCandidate
{
  std::uint32_t id;
  ProjectionBox box;
};

struct M2CertificateCounters
{
  unsigned long long vias;
  unsigned long long grid_members_visited;
  unsigned long long unique_candidates;
  unsigned long long x_certified;
  unsigned long long y_certified;
  unsigned long long misses;
  std::uint32_t maximum_query_candidates;
  std::uint32_t reserved;
};

struct M2CertificateSample
{
  std::uint32_t context_id;
  std::uint32_t cut_local;
  ProjectionBox cut;
};

struct M2CertificateOptions
{
  std::string path;
  std::string expected_scene_sha256;
  bool verify_cpu = false;
  std::uint64_t expected_metal_boxes = 0;
  std::uint64_t expected_vias = 0;
  std::uint64_t max_contexts = kDefaultMaxContexts;
  std::uint64_t max_grid_cells = kDefaultMaxGridCells;
  std::uint64_t max_memberships = kDefaultMaxMemberships;
  std::uint64_t max_metal_boxes = kDefaultMaxMetalBoxes;
  std::uint64_t max_via_boxes = kDefaultMaxCutBoxes;
  std::uint64_t max_cpu_pairs = kM2DefaultMaxCpuPairs;
  std::uint32_t max_query_candidates = kM2HardMaxQueryCandidates;
};

struct M2CertificateCpuResult
{
  std::uint64_t vias = 0;
  std::uint64_t x_certified = 0;
  std::uint64_t y_certified = 0;
  std::uint64_t misses = 0;
};

struct M2CertificateTimings
{
  double load_validate_ms = 0.0;
  double cpu_lower_ms = 0.0;
  double cpu_verify_ms = 0.0;
  double cuda_init_ms = 0.0;
  double alloc_upload_ms = 0.0;
  double metal_expand_ms = 0.0;
  double grid_count_ms = 0.0;
  double grid_build_ms = 0.0;
  double query_ms = 0.0;
  double d2h_ms = 0.0;
  double cleanup_ms = 0.0;
  double total_ms = 0.0;
};

__host__ __device__ bool
m2_positive_intersection(
    const ProjectionBox &first, const ProjectionBox &second)
{
  return first.left < second.right && first.right > second.left &&
         first.bottom < second.top && first.top > second.bottom;
}

__host__ __device__ ProjectionBox
m2_clip_box(const ProjectionBox &source, const ProjectionBox &clip)
{
  return {
      max(source.left, clip.left), max(source.bottom, clip.bottom),
      min(source.right, clip.right), min(source.top, clip.top)};
}

__device__ bool
m2_collect_query_candidates(
    const ProjectionBox &query, const ProjectionBox *metal_boxes,
    std::uint32_t metal_count, Grid grid, const std::uint32_t *counts,
    const std::uint32_t *offsets, const std::uint32_t *members,
    std::uint32_t maximum_candidates, M2QueryCandidate *candidates,
    std::uint32_t *candidate_count, unsigned long long *visited,
    std::uint32_t *status)
{
  std::int64_t x0 = 0;
  std::int64_t y0 = 0;
  std::int64_t x1 = 0;
  std::int64_t y1 = 0;
  if (!projection_box_span(query, grid, &x0, &y0, &x1, &y1)) {
    atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
    return false;
  }

  std::uint32_t size = 0;
  unsigned long long local_visited = 0;
  for (std::int64_t y = y0; y <= y1; ++y) {
    for (std::int64_t x = x0; x <= x1; ++x) {
      const std::uint64_t cell_id = grid_index(grid, x, y);
      const std::uint32_t begin = offsets[cell_id];
      const std::uint32_t count = counts[cell_id];
      if (begin > UINT32_MAX - count) {
        atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
        return false;
      }
      const std::uint32_t end = begin + count;
      for (std::uint32_t position = begin; position < end; ++position) {
        ++local_visited;
        const std::uint32_t metal_id = members[position];
        if (metal_id >= metal_count) {
          atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
          return false;
        }
        const ProjectionBox metal = metal_boxes[metal_id];
        if (!m2_positive_intersection(metal, query)) {
          continue;
        }
        bool duplicate = false;
        for (std::uint32_t existing = 0; existing < size; ++existing) {
          if (candidates[existing].id == metal_id) {
            duplicate = true;
            break;
          }
        }
        if (duplicate) {
          continue;
        }
        if (size >= maximum_candidates ||
            size >= kM2HardMaxQueryCandidates) {
          atomicOr(status, kM2QueryCapacityExceeded);
          *visited += local_visited;
          return false;
        }
        candidates[size++] = {metal_id, m2_clip_box(metal, query)};
      }
    }
  }
  *candidate_count = size;
  *visited += local_visited;
  return true;
}

__device__ bool
m2_union_covers_query(
    const ProjectionBox &query, const M2QueryCandidate *candidates,
    std::uint32_t candidate_count)
{
  if (!candidate_count) {
    return false;
  }

  // Two query endpoints plus two endpoints per clipped candidate.
  std::int64_t xs[2 + 2 * kM2HardMaxQueryCandidates];
  std::uint32_t x_count = 0;
  xs[x_count++] = query.left;
  xs[x_count++] = query.right;
  for (std::uint32_t candidate = 0; candidate < candidate_count; ++candidate) {
    xs[x_count++] = candidates[candidate].box.left;
    xs[x_count++] = candidates[candidate].box.right;
  }

  // The bounded lists are intentionally small.  Insertion sort avoids a
  // second global sort/scan pipeline and is deterministic across devices.
  for (std::uint32_t index = 1; index < x_count; ++index) {
    const std::int64_t value = xs[index];
    std::uint32_t position = index;
    while (position && value < xs[position - 1]) {
      xs[position] = xs[position - 1];
      --position;
    }
    xs[position] = value;
  }
  std::uint32_t unique_count = 0;
  for (std::uint32_t index = 0; index < x_count; ++index) {
    if (!unique_count || xs[index] != xs[unique_count - 1]) {
      xs[unique_count++] = xs[index];
    }
  }
  if (unique_count < 2 || xs[0] != query.left ||
      xs[unique_count - 1] != query.right) {
    return false;
  }

  for (std::uint32_t slab = 0; slab + 1 < unique_count; ++slab) {
    const std::int64_t slab_left = xs[slab];
    const std::int64_t slab_right = xs[slab + 1];
    if (slab_left >= slab_right) {
      return false;
    }
    std::int64_t covered_top = query.bottom;
    while (covered_top < query.top) {
      std::int64_t next_top = covered_top;
      for (std::uint32_t candidate = 0; candidate < candidate_count;
           ++candidate) {
        const ProjectionBox &box = candidates[candidate].box;
        if (box.left <= slab_left && box.right >= slab_right &&
            box.bottom <= covered_top && box.top > next_top) {
          next_top = box.top;
        }
      }
      if (next_top == covered_top) {
        return false;
      }
      covered_top = next_top;
    }
  }
  return true;
}

__device__ bool
m2_make_projection_query(
    const ProjectionBox &cut, std::int64_t distance, bool x_axis,
    ProjectionBox *query)
{
  *query = cut;
  if (x_axis) {
    return add_checked(cut.left, -distance, &query->left) &&
           add_checked(cut.right, distance, &query->right);
  }
  return add_checked(cut.bottom, -distance, &query->bottom) &&
         add_checked(cut.top, distance, &query->top);
}

__global__ void
m2_via2_certificate_kernel(
    const ContextGpu *contexts, const std::uint32_t *cut_contexts,
    std::uint32_t cut_context_count, const ProjectionCell *cells,
    const ProjectionBox *templates, const ProjectionBox *metal_boxes,
    std::uint32_t metal_count, Grid grid, const std::uint32_t *counts,
    const std::uint32_t *offsets, const std::uint32_t *members,
    std::int64_t distance, std::uint32_t maximum_candidates,
    M2CertificateCounters *counters, M2CertificateSample *samples,
    std::uint32_t *sample_count, std::uint32_t *status)
{
  const std::uint32_t list_id = blockIdx.x;
  if (list_id >= cut_context_count) {
    return;
  }
  const std::uint32_t context_id = cut_contexts[list_id];
  const ContextGpu context = contexts[context_id];
  const ProjectionCell cell = cells[context.cell];

  unsigned long long local_vias = 0;
  unsigned long long local_visited = 0;
  unsigned long long local_unique = 0;
  unsigned long long local_x = 0;
  unsigned long long local_y = 0;
  unsigned long long local_misses = 0;
  std::uint32_t local_maximum = 0;

  for (std::uint32_t local = threadIdx.x; local < cell.cut_count;
       local += blockDim.x) {
    ++local_vias;
    ProjectionBox cut{};
    if (!transform_projection_box_checked(
            context, templates[cell.cut_begin + local], &cut)) {
      atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
      ++local_misses;
      continue;
    }

    bool certified = false;
    bool x_certified = false;
    bool y_certified = false;
    for (int axis = 0; axis < 2 && !certified; ++axis) {
      ProjectionBox query{};
      if (!m2_make_projection_query(cut, distance, axis == 0, &query)) {
        atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
        break;
      }
      M2QueryCandidate candidates[kM2HardMaxQueryCandidates];
      std::uint32_t candidate_count = 0;
      if (!m2_collect_query_candidates(
              query, metal_boxes, metal_count, grid, counts, offsets, members,
              maximum_candidates, candidates, &candidate_count, &local_visited,
              status)) {
        break;
      }
      local_unique += candidate_count;
      local_maximum = max(local_maximum, candidate_count);
      certified =
          m2_union_covers_query(query, candidates, candidate_count);
      if (certified) {
        x_certified = axis == 0;
        y_certified = axis == 1;
      }
    }

    if (x_certified) {
      ++local_x;
    } else if (y_certified) {
      ++local_y;
    } else {
      ++local_misses;
      const std::uint32_t sample = atomicAdd(sample_count, 1u);
      if (sample < kSampleCapacity) {
        samples[sample] = {context_id, local, cut};
      }
    }
  }

  if (local_vias) atomicAdd(&counters->vias, local_vias);
  if (local_visited)
    atomicAdd(&counters->grid_members_visited, local_visited);
  if (local_unique)
    atomicAdd(&counters->unique_candidates, local_unique);
  if (local_x) atomicAdd(&counters->x_certified, local_x);
  if (local_y) atomicAdd(&counters->y_certified, local_y);
  if (local_misses) atomicAdd(&counters->misses, local_misses);
  atomicMax(&counters->maximum_query_candidates, local_maximum);
}

ProjectionBox
m2_make_projection_query_host(
    const ProjectionBox &cut, std::int64_t distance, bool x_axis)
{
  ProjectionBox query = cut;
  if (x_axis) {
    query.left = narrow_i64(
        static_cast<__int128>(cut.left) - distance, "M2 X query left");
    query.right = narrow_i64(
        static_cast<__int128>(cut.right) + distance, "M2 X query right");
  } else {
    query.bottom = narrow_i64(
        static_cast<__int128>(cut.bottom) - distance, "M2 Y query bottom");
    query.top = narrow_i64(
        static_cast<__int128>(cut.top) + distance, "M2 Y query top");
  }
  return query;
}

bool
m2_union_covers_query_host(
    const ProjectionBox &query, std::vector<ProjectionBox> candidates)
{
  if (candidates.empty()) {
    return false;
  }
  std::vector<std::int64_t> xs = {query.left, query.right};
  for (ProjectionBox &box : candidates) {
    box = m2_clip_box(box, query);
    xs.push_back(box.left);
    xs.push_back(box.right);
  }
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  if (xs.size() < 2 || xs.front() != query.left ||
      xs.back() != query.right) {
    return false;
  }
  for (std::size_t slab = 0; slab + 1 < xs.size(); ++slab) {
    std::int64_t covered_top = query.bottom;
    while (covered_top < query.top) {
      std::int64_t next_top = covered_top;
      for (const ProjectionBox &box : candidates) {
        if (box.left <= xs[slab] && box.right >= xs[slab + 1] &&
            box.bottom <= covered_top && box.top > next_top) {
          next_top = box.top;
        }
      }
      if (next_top == covered_top) {
        return false;
      }
      covered_top = next_top;
    }
  }
  return true;
}

M2CertificateCpuResult
m2_verify_cpu(
    const ProjectionLowered &lowered, std::uint64_t maximum_pairs)
{
  std::uint64_t pair_count = 0;
  if (!checked_mul_u64(lowered.metal_count, lowered.cut_count, &pair_count) ||
      pair_count > maximum_pairs) {
    throw SceneError("M2.4 CPU differential pair count exceeds capacity");
  }

  std::vector<ProjectionBox> metals;
  std::vector<ProjectionBox> cuts;
  metals.reserve(lowered.metal_count);
  cuts.reserve(lowered.cut_count);
  for (std::uint32_t context_id : lowered.metal_contexts) {
    const ContextGpu &context = lowered.contexts[context_id];
    const ProjectionCell &cell = lowered.cells[context.cell];
    for (std::uint32_t local = 0; local < cell.metal_count; ++local) {
      metals.push_back(transform_projection_box_host(
          context, lowered.templates[cell.metal_begin + local]));
    }
  }
  for (std::uint32_t context_id : lowered.cut_contexts) {
    const ContextGpu &context = lowered.contexts[context_id];
    const ProjectionCell &cell = lowered.cells[context.cell];
    for (std::uint32_t local = 0; local < cell.cut_count; ++local) {
      cuts.push_back(transform_projection_box_host(
          context, lowered.templates[cell.cut_begin + local]));
    }
  }

  M2CertificateCpuResult result;
  result.vias = cuts.size();
  for (const ProjectionBox &cut : cuts) {
    bool certified = false;
    for (int axis = 0; axis < 2 && !certified; ++axis) {
      const ProjectionBox query =
          m2_make_projection_query_host(cut, kProjectionDistance, axis == 0);
      std::vector<ProjectionBox> candidates;
      for (const ProjectionBox &metal : metals) {
        if (m2_positive_intersection(metal, query)) {
          candidates.push_back(metal);
        }
      }
      certified = m2_union_covers_query_host(query, std::move(candidates));
      if (certified) {
        if (axis == 0)
          ++result.x_certified;
        else
          ++result.y_certified;
      }
    }
    if (!certified) {
      ++result.misses;
    }
  }
  return result;
}

M2CertificateOptions
m2_parse_options(int argc, char **argv)
{
  M2CertificateOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto take_u64 = [&](const char *name, std::uint64_t *destination) {
      const std::string prefix = std::string(name) + "=";
      if (argument.rfind(prefix, 0) != 0) {
        return false;
      }
      *destination = parse_u64(argument.substr(prefix.size()), name);
      return true;
    };
    if (take_u64("--expect-metal-boxes", &options.expected_metal_boxes) ||
        take_u64("--expect-vias", &options.expected_vias) ||
        take_u64("--max-contexts", &options.max_contexts) ||
        take_u64("--max-grid-cells", &options.max_grid_cells) ||
        take_u64("--max-memberships", &options.max_memberships) ||
        take_u64("--max-metal-boxes", &options.max_metal_boxes) ||
        take_u64("--max-via-boxes", &options.max_via_boxes) ||
        take_u64("--max-cpu-pairs", &options.max_cpu_pairs)) {
      continue;
    }
    const std::string candidate_prefix = "--max-query-candidates=";
    if (argument.rfind(candidate_prefix, 0) == 0) {
      const std::uint64_t value = parse_u64(
          argument.substr(candidate_prefix.size()),
          "--max-query-candidates");
      if (!value || value > kM2HardMaxQueryCandidates) {
        throw SceneError(
            "--max-query-candidates must be in the range 1..128");
      }
      options.max_query_candidates = static_cast<std::uint32_t>(value);
      continue;
    }
    const std::string sha_prefix = "--expect-scene-sha256=";
    if (argument.rfind(sha_prefix, 0) == 0) {
      options.expected_scene_sha256 = argument.substr(sha_prefix.size());
      if (options.expected_scene_sha256.size() != 64 ||
          !std::all_of(
              options.expected_scene_sha256.begin(),
              options.expected_scene_sha256.end(),
              [](unsigned char character) {
                return std::isxdigit(character) != 0;
              })) {
        throw SceneError(
            "--expect-scene-sha256 requires exactly 64 hex digits");
      }
      std::transform(
          options.expected_scene_sha256.begin(),
          options.expected_scene_sha256.end(),
          options.expected_scene_sha256.begin(),
          [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
          });
      continue;
    }
    if (argument == "--verify-cpu") {
      options.verify_cpu = true;
      continue;
    }
    if (!argument.empty() && argument[0] == '-') {
      throw SceneError("unknown option: " + argument);
    }
    if (!options.path.empty()) {
      throw SceneError("exactly one packed scene path is required");
    }
    options.path = argument;
  }
  if (options.path.empty() || options.expected_scene_sha256.empty()) {
    throw SceneError(
        "usage: m2_via2_certificate --expect-scene-sha256=HEX "
        "[--verify-cpu] [capacity options] SCENE.kact");
  }
  return options;
}

int
m2_run_certificate(
    const M2CertificateOptions &options, Clock::time_point total_begin)
{
  M2CertificateTimings timing;
  auto begin = Clock::now();
  LoadedScene scene = load_and_validate(options.path);
  auto end = Clock::now();
  timing.load_validate_ms = milliseconds(begin, end);

  const std::string scene_digest = hex_digest(scene.header.scene_sha256, 32);
  if (options.expected_scene_sha256 != scene_digest) {
    throw SceneError("scene SHA-256 does not match explicit expectation");
  }
  if (scene.header.dbu != kQualifiedDbuMicrometres) {
    throw SceneError("M2.4 certificate requires the qualified 0.5nm DBU");
  }

  begin = Clock::now();
  ProjectionLowered lowered = lower_projection_scene(
      scene, options.max_contexts, options.max_metal_boxes,
      options.max_via_boxes);
  end = Clock::now();
  timing.cpu_lower_ms = milliseconds(begin, end);
  if (options.expected_metal_boxes &&
      options.expected_metal_boxes != lowered.metal_count) {
    throw SceneError("logical M2 rectangle census differs from expectation");
  }
  if (options.expected_vias && options.expected_vias != lowered.cut_count) {
    throw SceneError("logical VIA2 census differs from expectation");
  }

  std::optional<M2CertificateCpuResult> cpu;
  if (options.verify_cpu) {
    begin = Clock::now();
    cpu = m2_verify_cpu(lowered, options.max_cpu_pairs);
    end = Clock::now();
    timing.cpu_verify_ms = milliseconds(begin, end);
  }

  const Box root_metal =
      box_from(scene.cells[scene.header.root_cell].subtree_bbox[kWellLayer]);
  const Box root_via =
      box_from(scene.cells[scene.header.root_cell].subtree_bbox[kActiveLayer]);
  const std::int64_t root_left = narrow_i64(
      static_cast<__int128>(std::min(root_metal.left, root_via.left)) -
          kProjectionDistance,
      "M2.4 grid left");
  const std::int64_t root_bottom = narrow_i64(
      static_cast<__int128>(std::min(root_metal.bottom, root_via.bottom)) -
          kProjectionDistance,
      "M2.4 grid bottom");
  const std::int64_t root_right = narrow_i64(
      static_cast<__int128>(std::max(root_metal.right, root_via.right)) +
          kProjectionDistance,
      "M2.4 grid right");
  const std::int64_t root_top = narrow_i64(
      static_cast<__int128>(std::max(root_metal.top, root_via.top)) +
          kProjectionDistance,
      "M2.4 grid top");
  const std::int64_t base_x = host_floor_div(root_left, kGridCell);
  const std::int64_t base_y = host_floor_div(root_bottom, kGridCell);
  const std::int64_t maximum_x = host_floor_div(root_right, kGridCell);
  const std::int64_t maximum_y = host_floor_div(root_top, kGridCell);
  const std::uint64_t grid_width =
      static_cast<std::uint64_t>(maximum_x - base_x) + 1;
  const std::uint64_t grid_height =
      static_cast<std::uint64_t>(maximum_y - base_y) + 1;
  std::uint64_t grid_cells = 0;
  if (!checked_mul_u64(grid_width, grid_height, &grid_cells) ||
      grid_width > UINT32_MAX || grid_height > UINT32_MAX ||
      grid_cells > options.max_grid_cells || grid_cells > UINT32_MAX) {
    throw SceneError("M2.4 uniform grid exceeds capacity");
  }
  const Grid grid = {
      base_x, base_y, static_cast<std::uint32_t>(grid_width),
      static_cast<std::uint32_t>(grid_height)};

  begin = Clock::now();
  cuda_require(cudaFree(nullptr), "M2.4 CUDA context initialization");
  cuda_require(
      cudaDeviceSynchronize(), "M2.4 CUDA initialization synchronize");
  end = Clock::now();
  timing.cuda_init_ms = milliseconds(begin, end);

  begin = Clock::now();
  DeviceBuffer<ContextGpu> d_contexts(lowered.contexts.size());
  DeviceBuffer<std::uint32_t> d_metal_contexts(
      lowered.metal_contexts.size());
  DeviceBuffer<std::uint64_t> d_metal_offsets(lowered.metal_offsets.size());
  DeviceBuffer<std::uint32_t> d_cut_contexts(lowered.cut_contexts.size());
  DeviceBuffer<ProjectionCell> d_cells(lowered.cells.size());
  DeviceBuffer<ProjectionBox> d_templates(lowered.templates.size());
  DeviceBuffer<ProjectionBox> d_metal_boxes(lowered.metal_count);
  DeviceBuffer<std::uint32_t> d_counts(grid_cells);
  DeviceBuffer<std::uint32_t> d_offsets(grid_cells + 1);
  DeviceBuffer<std::uint32_t> d_cursors(grid_cells);
  DeviceBuffer<unsigned long long> d_membership_total(1);
  DeviceBuffer<std::uint32_t> d_status(1);
  DeviceBuffer<M2CertificateCounters> d_counters(1);
  DeviceBuffer<M2CertificateSample> d_samples(kSampleCapacity);
  DeviceBuffer<std::uint32_t> d_sample_count(1);
  upload(&d_contexts, lowered.contexts);
  upload(&d_metal_contexts, lowered.metal_contexts);
  upload(&d_metal_offsets, lowered.metal_offsets);
  upload(&d_cut_contexts, lowered.cut_contexts);
  upload(&d_cells, lowered.cells);
  upload(&d_templates, lowered.templates);
  cuda_require(
      cudaMemset(d_counts.get(), 0, grid_cells * sizeof(std::uint32_t)),
      "cudaMemset M2.4 grid counts");
  cuda_require(
      cudaMemset(d_membership_total.get(), 0, sizeof(unsigned long long)),
      "cudaMemset M2.4 membership total");
  cuda_require(
      cudaMemset(d_status.get(), 0, sizeof(std::uint32_t)),
      "cudaMemset M2.4 status");
  cuda_require(
      cudaMemset(d_counters.get(), 0, sizeof(M2CertificateCounters)),
      "cudaMemset M2.4 counters");
  cuda_require(
      cudaMemset(d_sample_count.get(), 0, sizeof(std::uint32_t)),
      "cudaMemset M2.4 sample count");
  projection_transform_gate<<<1, 8>>>(d_status.get());
  cuda_require(cudaGetLastError(), "M2.4 transform gate launch");
  cuda_require(cudaDeviceSynchronize(), "M2.4 upload synchronize");
  end = Clock::now();
  timing.alloc_upload_ms = milliseconds(begin, end);

  begin = Clock::now();
  expand_projection_metal_kernel<<<
      static_cast<unsigned int>(lowered.metal_contexts.size()), 128>>>(
      d_contexts.get(), d_metal_contexts.get(), d_metal_offsets.get(),
      d_cells.get(), d_templates.get(),
      static_cast<std::uint32_t>(lowered.metal_contexts.size()),
      d_metal_boxes.get(), d_status.get());
  cuda_require(cudaGetLastError(), "M2.4 metal expansion launch");
  cuda_require(cudaDeviceSynchronize(), "M2.4 metal expansion sync");
  end = Clock::now();
  timing.metal_expand_ms = milliseconds(begin, end);

  std::uint32_t host_status = 0;
  cuda_require(
      cudaMemcpy(
          &host_status, d_status.get(), sizeof(host_status),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy M2.4 expansion status");
  if (host_status) {
    throw SceneError(
        "device M2 expansion declined the scene, flags=" +
        std::to_string(host_status));
  }

  begin = Clock::now();
  const unsigned int metal_blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>(
          65535, (lowered.metal_count + 255) / 256));
  count_projection_grid_kernel<<<metal_blocks, 256>>>(
      d_metal_boxes.get(), static_cast<std::uint32_t>(lowered.metal_count),
      grid, d_counts.get(), d_membership_total.get(), d_status.get());
  cuda_require(cudaGetLastError(), "M2.4 grid count launch");
  cuda_require(cudaDeviceSynchronize(), "M2.4 grid count sync");
  unsigned long long membership_total = 0;
  cuda_require(
      cudaMemcpy(
          &membership_total, d_membership_total.get(),
          sizeof(membership_total), cudaMemcpyDeviceToHost),
      "cudaMemcpy M2.4 membership total");
  cuda_require(
      cudaMemcpy(
          &host_status, d_status.get(), sizeof(host_status),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy M2.4 grid count status");
  end = Clock::now();
  timing.grid_count_ms = milliseconds(begin, end);
  if (host_status) {
    throw SceneError(
        "device M2.4 grid count declined the scene, flags=" +
        std::to_string(host_status));
  }
  if (membership_total < lowered.metal_count ||
      membership_total > options.max_memberships ||
      membership_total > UINT32_MAX) {
    throw SceneError("M2.4 grid memberships exceed capacity");
  }

  begin = Clock::now();
  thrust::device_ptr<std::uint32_t> count_begin(d_counts.get());
  thrust::device_ptr<std::uint32_t> offset_begin(d_offsets.get());
  thrust::exclusive_scan(
      count_begin, count_begin + grid_cells, offset_begin);
  const std::uint32_t terminal =
      static_cast<std::uint32_t>(membership_total);
  cuda_require(
      cudaMemcpy(
          d_offsets.get() + grid_cells, &terminal, sizeof(terminal),
          cudaMemcpyHostToDevice),
      "cudaMemcpy M2.4 terminal offset");
  cuda_require(
      cudaMemcpy(
          d_cursors.get(), d_offsets.get(),
          grid_cells * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice),
      "cudaMemcpy M2.4 offsets to cursors");
  DeviceBuffer<std::uint32_t> d_members(membership_total);
  fill_projection_grid_kernel<<<metal_blocks, 256>>>(
      d_metal_boxes.get(), static_cast<std::uint32_t>(lowered.metal_count),
      grid, d_cursors.get(), d_members.get(), membership_total,
      d_status.get());
  cuda_require(cudaGetLastError(), "M2.4 grid fill launch");
  const unsigned int grid_blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>(65535, (grid_cells + 255) / 256));
  validate_grid_kernel<<<grid_blocks, 256>>>(
      d_counts.get(), d_offsets.get(), d_cursors.get(), grid_cells,
      d_status.get());
  cuda_require(cudaGetLastError(), "M2.4 grid validate launch");
  cuda_require(cudaDeviceSynchronize(), "M2.4 grid build sync");
  cuda_require(
      cudaMemcpy(
          &host_status, d_status.get(), sizeof(host_status),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy M2.4 grid build status");
  end = Clock::now();
  timing.grid_build_ms = milliseconds(begin, end);
  if (host_status) {
    throw SceneError(
        "device M2.4 grid build declined the scene, flags=" +
        std::to_string(host_status));
  }

  begin = Clock::now();
  if (lowered.cut_contexts.size() >
      std::numeric_limits<unsigned int>::max()) {
    throw SceneError("M2.4 VIA2 context launch exceeds CUDA grid-x");
  }
  m2_via2_certificate_kernel<<<
      static_cast<unsigned int>(lowered.cut_contexts.size()), 128>>>(
      d_contexts.get(), d_cut_contexts.get(),
      static_cast<std::uint32_t>(lowered.cut_contexts.size()),
      d_cells.get(), d_templates.get(), d_metal_boxes.get(),
      static_cast<std::uint32_t>(lowered.metal_count), grid,
      d_counts.get(), d_offsets.get(), d_members.get(), kProjectionDistance,
      options.max_query_candidates, d_counters.get(), d_samples.get(),
      d_sample_count.get(), d_status.get());
  cuda_require(cudaGetLastError(), "M2.4 exact coverage query launch");
  cuda_require(cudaDeviceSynchronize(), "M2.4 exact coverage query sync");
  end = Clock::now();
  timing.query_ms = milliseconds(begin, end);

  begin = Clock::now();
  M2CertificateCounters counters{};
  std::uint32_t sample_count = 0;
  std::array<M2CertificateSample, kSampleCapacity> samples{};
  cuda_require(
      cudaMemcpy(
          &counters, d_counters.get(), sizeof(counters),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy M2.4 counters");
  cuda_require(
      cudaMemcpy(
          &sample_count, d_sample_count.get(), sizeof(sample_count),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy M2.4 sample count");
  cuda_require(
      cudaMemcpy(
          &host_status, d_status.get(), sizeof(host_status),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy M2.4 final status");
  if (sample_count) {
    cuda_require(
        cudaMemcpy(
            samples.data(), d_samples.get(), sizeof(samples),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy M2.4 samples");
  }
  end = Clock::now();
  timing.d2h_ms = milliseconds(begin, end);

  if (counters.vias != lowered.cut_count ||
      counters.x_certified + counters.y_certified + counters.misses !=
          counters.vias) {
    throw SceneError("M2.4 device counters violate conservation");
  }
  if (cpu &&
      (cpu->vias != counters.vias ||
       cpu->x_certified != counters.x_certified ||
       cpu->y_certified != counters.y_certified ||
       cpu->misses != counters.misses)) {
    throw SceneError("M2.4 GPU result disagrees with CPU differential");
  }

  begin = Clock::now();
  cudaError_t cleanup_status = cudaSuccess;
  const auto release = [&](auto *buffer) {
    const cudaError_t status = buffer->release();
    if (cleanup_status == cudaSuccess && status != cudaSuccess) {
      cleanup_status = status;
    }
  };
  release(&d_sample_count);
  release(&d_samples);
  release(&d_counters);
  release(&d_status);
  release(&d_membership_total);
  release(&d_members);
  release(&d_cursors);
  release(&d_offsets);
  release(&d_counts);
  release(&d_metal_boxes);
  release(&d_templates);
  release(&d_cells);
  release(&d_cut_contexts);
  release(&d_metal_offsets);
  release(&d_metal_contexts);
  release(&d_contexts);
  if (cleanup_status != cudaSuccess) {
    throw SceneError(
        std::string("M2.4 CUDA cleanup: ") +
        cudaGetErrorString(cleanup_status));
  }
  end = Clock::now();
  timing.cleanup_ms = milliseconds(begin, end);
  timing.total_ms = milliseconds(total_begin, Clock::now());

  const bool uncertain = host_status != 0;
  const bool clean = !uncertain && counters.misses == 0;
  const char *verdict = uncertain ? "UNCERTAIN" : (clean ? "CLEAN" : "FALLBACK");
  const int exit_code = uncertain ? 2 : (clean ? 0 : 3);
  std::cout << std::fixed << std::setprecision(3)
            << "M2_VIA2_GPU_CERTIFICATE"
            << " verdict=" << verdict
            << " distance_dbu=" << kProjectionDistance
            << " dbu_um=" << std::setprecision(7) << scene.header.dbu
            << std::setprecision(3)
            << " contexts=" << lowered.contexts.size()
            << " metal_contexts=" << lowered.metal_contexts.size()
            << " via_contexts=" << lowered.cut_contexts.size()
            << " metal_boxes=" << lowered.metal_count
            << " vias=" << counters.vias
            << " grid=" << grid.width << "x" << grid.height
            << " grid_cells=" << grid_cells
            << " memberships=" << membership_total
            << " grid_members_visited=" << counters.grid_members_visited
            << " unique_candidates=" << counters.unique_candidates
            << " maximum_query_candidates="
            << counters.maximum_query_candidates
            << " x_certified=" << counters.x_certified
            << " y_certified=" << counters.y_certified
            << " misses=" << counters.misses
            << " decomposed_metal_templates="
            << lowered.decomposed_metal_templates
            << " decomposition_rectangles="
            << lowered.decomposition_rectangles
            << " device_flags=" << host_status
            << " scene_sha256=" << scene_digest << "\n";
  std::cout << "TIMING_MS"
            << " load_validate=" << timing.load_validate_ms
            << " cpu_lower=" << timing.cpu_lower_ms
            << " cpu_verify=" << timing.cpu_verify_ms
            << " cuda_init=" << timing.cuda_init_ms
            << " alloc_upload=" << timing.alloc_upload_ms
            << " metal_expand=" << timing.metal_expand_ms
            << " grid_count=" << timing.grid_count_ms
            << " grid_build=" << timing.grid_build_ms
            << " query=" << timing.query_ms
            << " d2h=" << timing.d2h_ms
            << " cleanup=" << timing.cleanup_ms
            << " gpu_plan="
            << timing.alloc_upload_ms + timing.metal_expand_ms +
                   timing.grid_count_ms + timing.grid_build_ms +
                   timing.query_ms + timing.d2h_ms + timing.cleanup_ms
            << " warm_context_total="
            << timing.total_ms - timing.cuda_init_ms
            << " cold_standalone_total=" << timing.total_ms
            << " total=" << timing.total_ms << "\n";
  const std::uint32_t copied_samples =
      std::min(sample_count, kSampleCapacity);
  for (std::uint32_t index = 0; index < copied_samples; ++index) {
    const M2CertificateSample &sample = samples[index];
    std::cout << "SAMPLE"
              << " context=" << sample.context_id
              << " via_local=" << sample.cut_local
              << " box=" << sample.cut.left << "," << sample.cut.bottom
              << "," << sample.cut.right << "," << sample.cut.top << "\n";
  }
  return exit_code;
}

}  // namespace

int
main(int argc, char **argv)
{
  const Clock::time_point total_begin = Clock::now();
  try {
    return m2_run_certificate(m2_parse_options(argc, argv), total_begin);
  } catch (const std::exception &error) {
    std::cerr << "M2_VIA2_GPU_CERTIFICATE"
              << " verdict=UNCERTAIN error=\"" << error.what() << "\"\n";
    return 2;
  }
}
