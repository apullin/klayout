/*

  KLayout Layout Viewer
  Copyright (C) 2006-2026 Matthias Koefferlein

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

*/

// Standalone proof of a device-owned projection-enclosure clean plan.
//
// KACTSCN1 layer zero is the enclosing metal and layer one is the enclosed
// cut.  The host validates the canonical packed scene and lowers only its
// hierarchy.  CUDA expands rectangular metal occurrences, builds a spatial
// index, streams cut occurrences through it, and returns one clean/fallback
// decision.  No intermediate edge pairs or polygons leave the device.
//
// This first plan deliberately implements a sufficient clean certificate:
// every cut must be contained by one rectangle which is itself a subset of a
// raw metal polygon, with the required margin on both X sides or both Y
// sides. Manhattan polygons are lowered to exact X- and Y-slab rectangles.
// Success is therefore conservative with respect to the merged metal region.
// A non-rectangular cut declines the scene.
//
// Cuts are also expanded and self-indexed once. Exact coincident duplicates
// are accepted because KLayout merges them into one unchanged physical cut;
// any other touch/overlap declines the scene. The same pass proves the
// qualified cut size and strict Euclidean cut-spacing rules when clean.

#define main klayout_cuda_embedded_active3_scene_main
#include "active3_scene_island.cu"
#undef main

namespace {

constexpr std::int64_t kProjectionDistance = 70;  // 35 nm at 0.5 nm/DBU.
constexpr std::int64_t kQualifiedCutSize = 130;   // 65 nm at 0.5 nm/DBU.
constexpr std::int64_t kCutSpacingDistance = 150; // 75 nm at 0.5 nm/DBU.
constexpr std::uint64_t kDefaultMaxMetalBoxes = UINT64_C(100000000);
constexpr std::uint64_t kDefaultMaxCutBoxes = UINT64_C(100000000);
constexpr std::uint64_t kDefaultMaxBruteforceBoxPairs = UINT64_C(10000000);

struct ProjectionBox {
    std::int64_t left;
    std::int64_t bottom;
    std::int64_t right;
    std::int64_t top;
};

struct ProjectionCell {
    std::uint64_t metal_begin;
    std::uint64_t cut_begin;
    std::uint32_t metal_count;
    std::uint32_t cut_count;
};

struct ProjectionLowered {
    std::vector<ContextGpu> contexts;
    std::vector<std::uint32_t> metal_contexts;
    std::vector<std::uint64_t> metal_offsets;
    std::vector<std::uint32_t> cut_contexts;
    std::vector<std::uint64_t> cut_offsets;
    std::vector<ProjectionCell> cells;
    std::vector<ProjectionBox> templates;
    std::uint64_t metal_count = 0;
    std::uint64_t cut_count = 0;
    std::uint64_t decomposed_metal_templates = 0;
    std::uint64_t decomposition_rectangles = 0;
    bool qualified_cut_size = true;
};

bool
is_rectangle_template(const LoadedScene &scene, const PolygonRecord &polygon)
{
    if (polygon.edge_count != 4 || polygon.bbox[0] >= polygon.bbox[2] ||
        polygon.bbox[1] >= polygon.bbox[3]) {
        return false;
    }
    std::uint32_t corner_mask = 0;
    for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
        const EdgeRecord &edge = scene.edges[polygon.edge_begin + local];
        if ((edge.x1 == edge.x2) == (edge.y1 == edge.y2)) {
            return false;
        }
        const bool right = edge.x1 == polygon.bbox[2];
        const bool top = edge.y1 == polygon.bbox[3];
        if ((!right && edge.x1 != polygon.bbox[0]) || (!top && edge.y1 != polygon.bbox[1])) {
            return false;
        }
        corner_mask |= 1u << ((top ? 2u : 0u) | (right ? 1u : 0u));
    }
    return corner_mask == 0xfu;
}

ProjectionBox
polygon_box(const PolygonRecord &polygon)
{
    return {polygon.bbox[0], polygon.bbox[1], polygon.bbox[2], polygon.bbox[3]};
}

std::vector<ProjectionBox>
decompose_manhattan_polygon(const LoadedScene &scene, const PolygonRecord &polygon)
{
    // KACTSCN1 validation has already proven that this is one simple,
    // hole-free Manhattan contour.  Between consecutive vertex Y values its
    // filled X intervals are constant, so the even/odd vertical crossings
    // yield an exact disjoint rectangle decomposition.
    std::vector<std::int64_t> ys;
    ys.reserve(polygon.edge_count);
    __int128 twice_area = 0;
    for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
        const EdgeRecord &edge = scene.edges[polygon.edge_begin + local];
        ys.push_back(edge.y1);
        twice_area +=
            static_cast<__int128>(edge.x1) * edge.y2 - static_cast<__int128>(edge.x2) * edge.y1;
    }
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    if (ys.size() < 2 || twice_area >= 0) {
        throw SceneError("invalid Manhattan polygon decomposition input");
    }

    std::vector<ProjectionBox> rectangles;
    __int128 rectangle_area = 0;
    for (std::size_t slab = 0; slab + 1 < ys.size(); ++slab) {
        const std::int64_t bottom = ys[slab];
        const std::int64_t top = ys[slab + 1];
        if (bottom >= top) {
            throw SceneError("non-positive Manhattan decomposition slab");
        }
        std::vector<std::int64_t> crossings;
        for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
            const EdgeRecord &edge = scene.edges[polygon.edge_begin + local];
            if (edge.x1 != edge.x2) {
                continue;
            }
            const std::int64_t edge_bottom = std::min(edge.y1, edge.y2);
            const std::int64_t edge_top = std::max(edge.y1, edge.y2);
            if (edge_bottom <= bottom && edge_top >= top) {
                crossings.push_back(edge.x1);
            }
        }
        std::sort(crossings.begin(), crossings.end());
        if (crossings.empty() || crossings.size() % 2) {
            throw SceneError("odd/empty Manhattan decomposition crossings");
        }
        for (std::size_t crossing = 0; crossing < crossings.size(); crossing += 2) {
            const std::int64_t left = crossings[crossing];
            const std::int64_t right = crossings[crossing + 1];
            if (left >= right) {
                throw SceneError("non-positive Manhattan decomposition interval");
            }
            rectangles.push_back({left, bottom, right, top});
            rectangle_area += static_cast<__int128>(right - left) * (top - bottom);
        }
    }
    if (rectangles.empty() || rectangle_area * 2 != -twice_area) {
        throw SceneError("Manhattan decomposition area mismatch");
    }
    return rectangles;
}

std::vector<ProjectionBox>
decompose_manhattan_polygon_x(const LoadedScene &scene, const PolygonRecord &polygon)
{
    // The orthogonal companion to the Y-slab decomposition above.  Retaining
    // both exact decompositions exposes horizontal and vertical rectangle
    // witnesses without implementing polygon containment in the CUDA query.
    std::vector<std::int64_t> xs;
    xs.reserve(polygon.edge_count);
    __int128 twice_area = 0;
    for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
        const EdgeRecord &edge = scene.edges[polygon.edge_begin + local];
        xs.push_back(edge.x1);
        twice_area +=
            static_cast<__int128>(edge.x1) * edge.y2 - static_cast<__int128>(edge.x2) * edge.y1;
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    if (xs.size() < 2 || twice_area >= 0) {
        throw SceneError("invalid X-slab decomposition input");
    }

    std::vector<ProjectionBox> rectangles;
    __int128 rectangle_area = 0;
    for (std::size_t slab = 0; slab + 1 < xs.size(); ++slab) {
        const std::int64_t left = xs[slab];
        const std::int64_t right = xs[slab + 1];
        if (left >= right) {
            throw SceneError("non-positive X-slab decomposition slab");
        }
        std::vector<std::int64_t> crossings;
        for (std::uint32_t local = 0; local < polygon.edge_count; ++local) {
            const EdgeRecord &edge = scene.edges[polygon.edge_begin + local];
            if (edge.y1 != edge.y2) {
                continue;
            }
            const std::int64_t edge_left = std::min(edge.x1, edge.x2);
            const std::int64_t edge_right = std::max(edge.x1, edge.x2);
            if (edge_left <= left && edge_right >= right) {
                crossings.push_back(edge.y1);
            }
        }
        std::sort(crossings.begin(), crossings.end());
        if (crossings.empty() || crossings.size() % 2) {
            throw SceneError("odd/empty X-slab decomposition crossings");
        }
        for (std::size_t crossing = 0; crossing < crossings.size(); crossing += 2) {
            const std::int64_t bottom = crossings[crossing];
            const std::int64_t top = crossings[crossing + 1];
            if (bottom >= top) {
                throw SceneError("non-positive X-slab decomposition interval");
            }
            rectangles.push_back({left, bottom, right, top});
            rectangle_area += static_cast<__int128>(right - left) * (top - bottom);
        }
    }
    if (rectangles.empty() || rectangle_area * 2 != -twice_area) {
        throw SceneError("X-slab decomposition area mismatch");
    }
    return rectangles;
}

ProjectionLowered
lower_projection_scene(
    const LoadedScene &scene,
    std::uint64_t max_contexts,
    std::uint64_t max_metal_boxes,
    std::uint64_t max_cut_boxes)
{
    // Reuse the fully checked hierarchy expansion from the ACTIVE.3 island.
    // Its semantic layer names do not affect transforms or context ordering.
    LoweredScene hierarchy = lower_hierarchy(scene, max_contexts);

    ProjectionLowered lowered;
    lowered.contexts = std::move(hierarchy.contexts);
    lowered.cells.resize(scene.header.cell_count);
    lowered.templates.reserve(scene.header.polygon_count);

    for (std::uint64_t cell_id = 0; cell_id < scene.header.cell_count; ++cell_id) {
        const CellRecord &source_cell = scene.cells[cell_id];
        ProjectionCell cell{};
        cell.metal_begin = lowered.templates.size();
        for (std::uint64_t local = 0; local < source_cell.polygon_count; ++local) {
            const PolygonRecord &polygon = scene.polygons[source_cell.polygon_begin + local];
            if (polygon.layer_code != kWellLayer) {
                continue;
            }
            if (is_rectangle_template(scene, polygon)) {
                lowered.templates.push_back(polygon_box(polygon));
                if (cell.metal_count == std::numeric_limits<std::uint32_t>::max()) {
                    throw SceneError("per-cell metal rectangle count exceeds uint32");
                }
                ++cell.metal_count;
            } else {
                std::vector<ProjectionBox> decomposition =
                    decompose_manhattan_polygon(scene, polygon);
                const std::vector<ProjectionBox> x_decomposition =
                    decompose_manhattan_polygon_x(scene, polygon);
                decomposition.insert(
                    decomposition.end(), x_decomposition.begin(), x_decomposition.end());
                if (decomposition.size() >
                    std::numeric_limits<std::uint32_t>::max() - cell.metal_count) {
                    throw SceneError("per-cell decomposed metal count exceeds uint32");
                }
                lowered.templates.insert(
                    lowered.templates.end(), decomposition.begin(), decomposition.end());
                cell.metal_count += static_cast<std::uint32_t>(decomposition.size());
                ++lowered.decomposed_metal_templates;
                lowered.decomposition_rectangles += decomposition.size();
            }
        }
        cell.cut_begin = lowered.templates.size();
        for (std::uint64_t local = 0; local < source_cell.polygon_count; ++local) {
            const PolygonRecord &polygon = scene.polygons[source_cell.polygon_begin + local];
            if (polygon.layer_code != kActiveLayer) {
                continue;
            }
            if (!is_rectangle_template(scene, polygon)) {
                throw SceneError("projection-enclosure cut layer contains a non-rectangle");
            }
            const ProjectionBox box = polygon_box(polygon);
            if (box.right - box.left != kQualifiedCutSize ||
                box.top - box.bottom != kQualifiedCutSize) {
                lowered.qualified_cut_size = false;
            }
            if (cell.cut_count == std::numeric_limits<std::uint32_t>::max()) {
                throw SceneError("per-cell cut rectangle count exceeds uint32");
            }
            lowered.templates.push_back(box);
            ++cell.cut_count;
        }
        lowered.cells[cell_id] = cell;
    }

    for (std::uint32_t context_id = 0; context_id < lowered.contexts.size(); ++context_id) {
        const ProjectionCell &cell = lowered.cells[lowered.contexts[context_id].cell];
        if (cell.metal_count) {
            lowered.metal_contexts.push_back(context_id);
            lowered.metal_offsets.push_back(lowered.metal_count);
            if (!checked_add_u64(lowered.metal_count, cell.metal_count, &lowered.metal_count) ||
                lowered.metal_count > max_metal_boxes) {
                throw SceneError("logical metal rectangle count exceeds configured capacity");
            }
        }
        if (cell.cut_count) {
            lowered.cut_contexts.push_back(context_id);
            lowered.cut_offsets.push_back(lowered.cut_count);
            if (!checked_add_u64(lowered.cut_count, cell.cut_count, &lowered.cut_count) ||
                lowered.cut_count > max_cut_boxes) {
                throw SceneError("logical cut rectangle count exceeds configured capacity");
            }
        }
    }
    if (!lowered.metal_count || !lowered.cut_count) {
        throw SceneError("projection-enclosure scene requires metal and cut rectangles");
    }
    if (lowered.metal_count > std::numeric_limits<std::uint32_t>::max()) {
        throw SceneError("logical metal rectangle count exceeds uint32");
    }
    if (lowered.cut_count > std::numeric_limits<std::uint32_t>::max()) {
        throw SceneError("logical cut rectangle count exceeds uint32");
    }
    return lowered;
}

__device__ bool
transform_projection_box_checked(
    const ContextGpu &context, const ProjectionBox &source, ProjectionBox *destination)
{
    const std::int64_t xs[4] = {source.left, source.left, source.right, source.right};
    const std::int64_t ys[4] = {source.bottom, source.top, source.bottom, source.top};
    ProjectionBox transformed = {INT64_MAX, INT64_MAX, INT64_MIN, INT64_MIN};
    for (int corner = 0; corner < 4; ++corner) {
        std::int64_t x = 0;
        std::int64_t y = 0;
        if (!transform_point_checked(context, xs[corner], ys[corner], &x, &y)) {
            return false;
        }
        transformed.left = min(transformed.left, x);
        transformed.bottom = min(transformed.bottom, y);
        transformed.right = max(transformed.right, x);
        transformed.top = max(transformed.top, y);
    }
    if (transformed.left >= transformed.right || transformed.bottom >= transformed.top) {
        return false;
    }
    *destination = transformed;
    return true;
}

ProjectionBox
transform_projection_box_host(const ContextGpu &context, const ProjectionBox &source)
{
    const std::int64_t xs[4] = {source.left, source.left, source.right, source.right};
    const std::int64_t ys[4] = {source.bottom, source.top, source.bottom, source.top};
    ProjectionBox transformed = {
        std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::min()};
    for (int corner = 0; corner < 4; ++corner) {
        const auto point = transform_128(context.transform, xs[corner], ys[corner]);
        const std::int64_t x = narrow_i64(point.first + context.tx, "projection box x");
        const std::int64_t y = narrow_i64(point.second + context.ty, "projection box y");
        transformed.left = std::min(transformed.left, x);
        transformed.bottom = std::min(transformed.bottom, y);
        transformed.right = std::max(transformed.right, x);
        transformed.top = std::max(transformed.top, y);
    }
    return transformed;
}

__device__ bool
projection_box_span(
    const ProjectionBox &box,
    const Grid &grid,
    std::int64_t *x0,
    std::int64_t *y0,
    std::int64_t *x1,
    std::int64_t *y1)
{
    *x0 = floor_div(box.left, kGridCell);
    *x1 = floor_div(box.right, kGridCell);
    *y0 = floor_div(box.bottom, kGridCell);
    *y1 = floor_div(box.top, kGridCell);
    return span_inside_grid(grid, *x0, *y0, *x1, *y1);
}

__device__ bool
projection_certificate(const ProjectionBox &metal, const ProjectionBox &cut, std::int64_t distance)
{
    const bool contained_x = metal.left <= cut.left && metal.right >= cut.right;
    const bool contained_y = metal.bottom <= cut.bottom && metal.top >= cut.top;
    if (!contained_x || !contained_y) {
        return false;
    }
    const bool x_margin = cut.left - metal.left >= distance && metal.right - cut.right >= distance;
    const bool y_margin = cut.bottom - metal.bottom >= distance && metal.top - cut.top >= distance;
    return x_margin || y_margin;
}

bool
projection_certificate_host(
    const ProjectionBox &metal, const ProjectionBox &cut, std::int64_t distance)
{
    const bool contained_x = metal.left <= cut.left && metal.right >= cut.right;
    const bool contained_y = metal.bottom <= cut.bottom && metal.top >= cut.top;
    if (!contained_x || !contained_y) {
        return false;
    }
    return (cut.left - metal.left >= distance && metal.right - cut.right >= distance) ||
           (cut.bottom - metal.bottom >= distance && metal.top - cut.top >= distance);
}

__global__ void
projection_transform_gate(std::uint32_t *status)
{
    const std::uint32_t code = threadIdx.x;
    if (blockIdx.x || code >= 8) {
        return;
    }
    const ProjectionBox source = {-10, -20, 30, 40};
    const ContextGpu context = {0, code, 13, -7};
    ProjectionBox transformed{};
    if (!transform_projection_box_checked(context, source, &transformed) ||
        transformed.right - transformed.left != ((code & 1u) ? 60 : 40) ||
        transformed.top - transformed.bottom != ((code & 1u) ? 40 : 60)) {
        atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
    }
}

__global__ void
expand_projection_metal_kernel(
    const ContextGpu *contexts,
    const std::uint32_t *metal_contexts,
    const std::uint64_t *metal_offsets,
    const ProjectionCell *cells,
    const ProjectionBox *templates,
    std::uint32_t context_count,
    ProjectionBox *metal_boxes,
    std::uint32_t *status)
{
    const std::uint32_t list_id = blockIdx.x;
    if (list_id >= context_count) {
        return;
    }
    const ContextGpu context = contexts[metal_contexts[list_id]];
    const ProjectionCell cell = cells[context.cell];
    for (std::uint32_t local = threadIdx.x; local < cell.metal_count; local += blockDim.x) {
        ProjectionBox box{};
        if (!transform_projection_box_checked(context, templates[cell.metal_begin + local], &box)) {
            atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
            continue;
        }
        metal_boxes[metal_offsets[list_id] + local] = box;
    }
}

__global__ void
expand_projection_cut_kernel(
    const ContextGpu *contexts,
    const std::uint32_t *cut_contexts,
    const std::uint64_t *cut_offsets,
    const ProjectionCell *cells,
    const ProjectionBox *templates,
    std::uint32_t context_count,
    ProjectionBox *cut_boxes,
    std::uint32_t *status)
{
    const std::uint32_t list_id = blockIdx.x;
    if (list_id >= context_count) {
        return;
    }
    const ContextGpu context = contexts[cut_contexts[list_id]];
    const ProjectionCell cell = cells[context.cell];
    for (std::uint32_t local = threadIdx.x; local < cell.cut_count; local += blockDim.x) {
        ProjectionBox box{};
        if (!transform_projection_box_checked(context, templates[cell.cut_begin + local], &box)) {
            atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
            continue;
        }
        cut_boxes[cut_offsets[list_id] + local] = box;
    }
}

__global__ void
count_projection_grid_kernel(
    const ProjectionBox *metal_boxes,
    std::uint32_t metal_count,
    Grid grid,
    std::uint32_t *counts,
    unsigned long long *total,
    std::uint32_t *status)
{
    for (std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < metal_count;
         id += blockDim.x * gridDim.x) {
        std::int64_t x0 = 0;
        std::int64_t y0 = 0;
        std::int64_t x1 = 0;
        std::int64_t y1 = 0;
        if (!projection_box_span(metal_boxes[id], grid, &x0, &y0, &x1, &y1)) {
            atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
            continue;
        }
        for (std::int64_t y = y0; y <= y1; ++y) {
            for (std::int64_t x = x0; x <= x1; ++x) {
                const std::uint64_t index = grid_index(grid, x, y);
                const std::uint32_t previous = atomicAdd(counts + index, 1u);
                if (previous == UINT32_MAX) {
                    atomicOr(status, static_cast<std::uint32_t>(kGridCounterOverflow));
                }
                atomicAdd(total, 1ull);
            }
        }
    }
}

__global__ void
fill_projection_grid_kernel(
    const ProjectionBox *metal_boxes,
    std::uint32_t metal_count,
    Grid grid,
    std::uint32_t *cursors,
    std::uint32_t *members,
    std::uint64_t member_capacity,
    std::uint32_t *status)
{
    for (std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < metal_count;
         id += blockDim.x * gridDim.x) {
        std::int64_t x0 = 0;
        std::int64_t y0 = 0;
        std::int64_t x1 = 0;
        std::int64_t y1 = 0;
        if (!projection_box_span(metal_boxes[id], grid, &x0, &y0, &x1, &y1)) {
            atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
            continue;
        }
        for (std::int64_t y = y0; y <= y1; ++y) {
            for (std::int64_t x = x0; x <= x1; ++x) {
                const std::uint64_t index = grid_index(grid, x, y);
                const std::uint32_t position = atomicAdd(cursors + index, 1u);
                if (position >= member_capacity) {
                    atomicOr(status, static_cast<std::uint32_t>(kGridCapacityExceeded));
                } else {
                    members[position] = id;
                }
            }
        }
    }
}

__device__ bool
expanded_projection_box_span(
    const ProjectionBox &box,
    const Grid &grid,
    std::int64_t expansion,
    std::int64_t *x0,
    std::int64_t *y0,
    std::int64_t *x1,
    std::int64_t *y1)
{
    std::int64_t left = 0;
    std::int64_t bottom = 0;
    std::int64_t right = 0;
    std::int64_t top = 0;
    if (!add_checked(box.left, -expansion, &left) ||
        !add_checked(box.bottom, -expansion, &bottom) ||
        !add_checked(box.right, expansion, &right) || !add_checked(box.top, expansion, &top)) {
        return false;
    }
    *x0 = floor_div(left, kGridCell);
    *x1 = floor_div(right, kGridCell);
    *y0 = floor_div(bottom, kGridCell);
    *y1 = floor_div(top, kGridCell);
    return true;
}

struct CutPairCounters {
    unsigned long long cuts_queried;
    unsigned long long candidate_pairs;
    unsigned long long duplicate_pairs;
    unsigned long long unsafe_touching_pairs;
    unsigned long long spacing_pairs;
    unsigned long long clean_pairs;
    std::uint32_t sample_claimed;
    std::uint32_t sample_cut;
    std::uint32_t sample_other;
    std::uint32_t reserved;
    ProjectionBox sample_cut_box;
    ProjectionBox sample_other_box;
};

__global__ void
query_projection_cut_pairs_kernel(
    const ProjectionBox *cut_boxes,
    std::uint32_t cut_count,
    Grid grid,
    const std::uint32_t *counts,
    const std::uint32_t *offsets,
    const std::uint32_t *members,
    std::int64_t spacing,
    CutPairCounters *counters,
    std::uint32_t *status)
{
    unsigned long long local_cuts = 0;
    unsigned long long local_candidates = 0;
    unsigned long long local_duplicates = 0;
    unsigned long long local_unsafe_touching = 0;
    unsigned long long local_spacing = 0;
    unsigned long long local_clean = 0;
    for (std::uint32_t cut_id = blockIdx.x * blockDim.x + threadIdx.x; cut_id < cut_count;
         cut_id += blockDim.x * gridDim.x) {
        const ProjectionBox cut = cut_boxes[cut_id];
        ++local_cuts;
        std::int64_t query_x0 = 0;
        std::int64_t query_y0 = 0;
        std::int64_t query_x1 = 0;
        std::int64_t query_y1 = 0;
        if (!expanded_projection_box_span(
                cut, grid, spacing, &query_x0, &query_y0, &query_x1, &query_y1) ||
            !clip_span(grid, &query_x0, &query_y0, &query_x1, &query_y1)) {
            atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
            continue;
        }
        for (std::int64_t y = query_y0; y <= query_y1; ++y) {
            for (std::int64_t x = query_x0; x <= query_x1; ++x) {
                const std::uint64_t cell_id = grid_index(grid, x, y);
                const std::uint32_t begin = offsets[cell_id];
                const std::uint32_t end = begin + counts[cell_id];
                for (std::uint32_t position = begin; position < end; ++position) {
                    const std::uint32_t other_id = members[position];
                    if (other_id <= cut_id || other_id >= cut_count) {
                        if (other_id >= cut_count) {
                            atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
                        }
                        continue;
                    }
                    const ProjectionBox other = cut_boxes[other_id];
                    std::int64_t other_x0 = 0;
                    std::int64_t other_y0 = 0;
                    std::int64_t other_x1 = 0;
                    std::int64_t other_y1 = 0;
                    if (!projection_box_span(
                            other, grid, &other_x0, &other_y0, &other_x1, &other_y1)) {
                        atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
                        continue;
                    }
                    // A pair can share several grid cells. The componentwise-lowest
                    // shared cell is its unique owner.
                    if (x != max(query_x0, other_x0) || y != max(query_y0, other_y0)) {
                        continue;
                    }
                    const std::uint64_t dx =
                        cut.right < other.left
                            ? static_cast<std::uint64_t>(other.left - cut.right)
                            : (other.right < cut.left
                                   ? static_cast<std::uint64_t>(cut.left - other.right)
                                   : UINT64_C(0));
                    const std::uint64_t dy =
                        cut.top < other.bottom
                            ? static_cast<std::uint64_t>(other.bottom - cut.top)
                            : (other.top < cut.bottom
                                   ? static_cast<std::uint64_t>(cut.bottom - other.top)
                                   : UINT64_C(0));
                    const std::uint64_t spacing_u = static_cast<std::uint64_t>(spacing);
                    if (dx >= spacing_u || dy >= spacing_u) {
                        continue;
                    }
                    ++local_candidates;
                    if (dx == 0 && dy == 0) {
                        const bool identical = cut.left == other.left &&
                                               cut.bottom == other.bottom &&
                                               cut.right == other.right && cut.top == other.top;
                        if (identical) {
                            ++local_duplicates;
                        } else {
                            ++local_unsafe_touching;
                            if (atomicCAS(&counters->sample_claimed, 0u, 1u) == 0u) {
                                counters->sample_cut = cut_id;
                                counters->sample_other = other_id;
                                counters->sample_cut_box = cut;
                                counters->sample_other_box = other;
                            }
                        }
                    } else if (dx * dx + dy * dy < spacing_u * spacing_u) {
                        ++local_spacing;
                    } else {
                        ++local_clean;
                    }
                }
            }
        }
    }
    if (local_cuts) {
        atomicAdd(&counters->cuts_queried, local_cuts);
    }
    if (local_candidates) {
        atomicAdd(&counters->candidate_pairs, local_candidates);
    }
    if (local_duplicates) {
        atomicAdd(&counters->duplicate_pairs, local_duplicates);
    }
    if (local_unsafe_touching) {
        atomicAdd(&counters->unsafe_touching_pairs, local_unsafe_touching);
    }
    if (local_spacing) {
        atomicAdd(&counters->spacing_pairs, local_spacing);
    }
    if (local_clean) {
        atomicAdd(&counters->clean_pairs, local_clean);
    }
}

struct ProjectionCounters {
    unsigned long long cuts;
    unsigned long long candidate_boxes;
    unsigned long long certified;
    unsigned long long misses;
};

struct ProjectionSample {
    std::uint32_t context_id;
    std::uint32_t cut_local;
    std::int64_t left;
    std::int64_t bottom;
    std::int64_t right;
    std::int64_t top;
};

__global__ void
query_projection_cuts_kernel(
    const ContextGpu *contexts,
    const std::uint32_t *cut_contexts,
    std::uint32_t cut_context_count,
    const ProjectionCell *cells,
    const ProjectionBox *templates,
    const ProjectionBox *metal_boxes,
    std::uint32_t metal_count,
    Grid grid,
    const std::uint32_t *counts,
    const std::uint32_t *offsets,
    const std::uint32_t *members,
    std::int64_t distance,
    ProjectionCounters *counters,
    ProjectionSample *samples,
    std::uint32_t *sample_count,
    std::uint32_t *status)
{
    const std::uint32_t list_id = blockIdx.x;
    if (list_id >= cut_context_count) {
        return;
    }
    const std::uint32_t context_id = cut_contexts[list_id];
    const ContextGpu context = contexts[context_id];
    const ProjectionCell cell = cells[context.cell];
    unsigned long long local_cuts = 0;
    unsigned long long local_candidates = 0;
    unsigned long long local_certified = 0;
    unsigned long long local_misses = 0;
    for (std::uint32_t local = threadIdx.x; local < cell.cut_count; local += blockDim.x) {
        ++local_cuts;
        ProjectionBox cut{};
        if (!transform_projection_box_checked(context, templates[cell.cut_begin + local], &cut)) {
            atomicOr(status, static_cast<std::uint32_t>(kTransformOverflow));
            ++local_misses;
            continue;
        }
        const std::int64_t center_x = cut.left + (cut.right - cut.left) / 2;
        const std::int64_t center_y = cut.bottom + (cut.top - cut.bottom) / 2;
        const std::int64_t grid_x = floor_div(center_x, kGridCell);
        const std::int64_t grid_y = floor_div(center_y, kGridCell);
        const std::int64_t maximum_x = grid.base_x + static_cast<std::int64_t>(grid.width) - 1;
        const std::int64_t maximum_y = grid.base_y + static_cast<std::int64_t>(grid.height) - 1;
        bool certified = false;
        if (grid_x >= grid.base_x && grid_x <= maximum_x && grid_y >= grid.base_y &&
            grid_y <= maximum_y) {
            const std::uint64_t grid_id = grid_index(grid, grid_x, grid_y);
            const std::uint32_t begin = offsets[grid_id];
            const std::uint32_t end = begin + counts[grid_id];
            for (std::uint32_t position = begin; position < end; ++position) {
                const std::uint32_t metal_id = members[position];
                if (metal_id >= metal_count) {
                    atomicOr(status, static_cast<std::uint32_t>(kInvalidDeviceRecord));
                    continue;
                }
                ++local_candidates;
                if (projection_certificate(metal_boxes[metal_id], cut, distance)) {
                    certified = true;
                    break;
                }
            }
        }
        if (certified) {
            ++local_certified;
        } else {
            ++local_misses;
            const std::uint32_t sample = atomicAdd(sample_count, 1u);
            if (sample < kSampleCapacity) {
                samples[sample] = {context_id, local, cut.left, cut.bottom, cut.right, cut.top};
            }
        }
    }
    if (local_cuts) {
        atomicAdd(&counters->cuts, local_cuts);
    }
    if (local_candidates) {
        atomicAdd(&counters->candidate_boxes, local_candidates);
    }
    if (local_certified) {
        atomicAdd(&counters->certified, local_certified);
    }
    if (local_misses) {
        atomicAdd(&counters->misses, local_misses);
    }
}

struct ProjectionOptions {
    std::string path;
    std::string expected_scene_sha256;
    bool verify_bruteforce = false;
    std::uint64_t max_contexts = kDefaultMaxContexts;
    std::uint64_t max_grid_cells = kDefaultMaxGridCells;
    std::uint64_t max_memberships = kDefaultMaxMemberships;
    std::uint64_t max_metal_boxes = kDefaultMaxMetalBoxes;
    std::uint64_t max_cut_boxes = kDefaultMaxCutBoxes;
    std::uint64_t max_bruteforce_pairs = kDefaultMaxBruteforceBoxPairs;
};

ProjectionOptions
parse_projection_options(int argc, char **argv)
{
    ProjectionOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto take = [&](const char *prefix, std::uint64_t *destination) {
            const std::string marker = std::string(prefix) + "=";
            if (argument.rfind(marker, 0) == 0) {
                *destination = parse_u64(argument.substr(marker.size()), prefix);
                return true;
            }
            return false;
        };
        if (take("--max-contexts", &options.max_contexts) ||
            take("--max-grid-cells", &options.max_grid_cells) ||
            take("--max-memberships", &options.max_memberships) ||
            take("--max-metal-boxes", &options.max_metal_boxes) ||
            take("--max-cut-boxes", &options.max_cut_boxes) ||
            take("--max-bruteforce-pairs", &options.max_bruteforce_pairs)) {
            continue;
        }
        const std::string sha_prefix = "--expect-scene-sha256=";
        if (argument.rfind(sha_prefix, 0) == 0) {
            options.expected_scene_sha256 = argument.substr(sha_prefix.size());
            if (options.expected_scene_sha256.size() != 64 ||
                !std::all_of(
                    options.expected_scene_sha256.begin(), options.expected_scene_sha256.end(),
                    [](unsigned char character) { return std::isxdigit(character) != 0; })) {
                throw SceneError("--expect-scene-sha256 requires exactly 64 hex digits");
            }
            std::transform(
                options.expected_scene_sha256.begin(), options.expected_scene_sha256.end(),
                options.expected_scene_sha256.begin(),
                [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
            continue;
        }
        if (argument == "--verify-bruteforce") {
            options.verify_bruteforce = true;
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
            "usage: projection_enclosure_scene_island "
            "--expect-scene-sha256=HEX [capacity options] SCENE.kact");
    }
    return options;
}

struct ProjectionBruteforce {
    std::uint64_t cuts = 0;
    std::uint64_t candidate_boxes = 0;
    std::uint64_t certified = 0;
    std::uint64_t misses = 0;
};

ProjectionBruteforce
verify_projection_bruteforce(const ProjectionLowered &lowered, std::uint64_t maximum_pairs)
{
    std::uint64_t pair_count = 0;
    if (!checked_mul_u64(lowered.metal_count, lowered.cut_count, &pair_count) ||
        pair_count > maximum_pairs) {
        throw SceneError("projection CPU brute-force pair count exceeds capacity");
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
            cuts.push_back(
                transform_projection_box_host(context, lowered.templates[cell.cut_begin + local]));
        }
    }
    ProjectionBruteforce result;
    result.cuts = cuts.size();
    for (const ProjectionBox &cut : cuts) {
        bool certified = false;
        for (const ProjectionBox &metal : metals) {
            ++result.candidate_boxes;
            if (projection_certificate_host(metal, cut, kProjectionDistance)) {
                certified = true;
                break;
            }
        }
        if (certified) {
            ++result.certified;
        } else {
            ++result.misses;
        }
    }
    return result;
}

struct ProjectionTimings {
    double load_validate_ms = 0;
    double cpu_lower_ms = 0;
    double bruteforce_ms = 0;
    double cuda_init_ms = 0;
    double alloc_upload_ms = 0;
    double metal_expand_ms = 0;
    double cut_expand_ms = 0;
    double grid_count_ms = 0;
    double grid_build_ms = 0;
    double cut_grid_count_ms = 0;
    double cut_grid_build_ms = 0;
    double cut_pair_query_ms = 0;
    double cut_query_ms = 0;
    double d2h_ms = 0;
    double cleanup_ms = 0;
    double total_ms = 0;
};

int
run_projection_plan(const ProjectionOptions &options, Clock::time_point total_begin)
{
    ProjectionTimings timing;
    auto begin = Clock::now();
    LoadedScene scene = load_and_validate(options.path);
    auto end = Clock::now();
    timing.load_validate_ms = milliseconds(begin, end);
    if (options.expected_scene_sha256 != hex_digest(scene.header.scene_sha256, 32)) {
        throw SceneError("scene SHA-256 does not match explicit expectation");
    }
    if (scene.header.dbu != kQualifiedDbuMicrometres) {
        throw SceneError("projection plan requires the qualified 0.5nm DBU");
    }

    begin = Clock::now();
    ProjectionLowered lowered = lower_projection_scene(
        scene, options.max_contexts, options.max_metal_boxes, options.max_cut_boxes);
    end = Clock::now();
    timing.cpu_lower_ms = milliseconds(begin, end);

    std::optional<ProjectionBruteforce> brute_force;
    if (options.verify_bruteforce) {
        begin = Clock::now();
        brute_force = verify_projection_bruteforce(lowered, options.max_bruteforce_pairs);
        end = Clock::now();
        timing.bruteforce_ms = milliseconds(begin, end);
    }

    const Box root_metal = box_from(scene.cells[scene.header.root_cell].subtree_bbox[kWellLayer]);
    const Box root_cut = box_from(scene.cells[scene.header.root_cell].subtree_bbox[kActiveLayer]);
    const std::int64_t root_left = narrow_i64(
        static_cast<__int128>(std::min(root_metal.left, root_cut.left)) - kCutSpacingDistance,
        "projection grid left");
    const std::int64_t root_bottom = narrow_i64(
        static_cast<__int128>(std::min(root_metal.bottom, root_cut.bottom)) - kCutSpacingDistance,
        "projection grid bottom");
    const std::int64_t root_right = narrow_i64(
        static_cast<__int128>(std::max(root_metal.right, root_cut.right)) + kCutSpacingDistance,
        "projection grid right");
    const std::int64_t root_top = narrow_i64(
        static_cast<__int128>(std::max(root_metal.top, root_cut.top)) + kCutSpacingDistance,
        "projection grid top");
    const std::int64_t base_x = host_floor_div(root_left, kGridCell);
    const std::int64_t base_y = host_floor_div(root_bottom, kGridCell);
    const std::int64_t maximum_x = host_floor_div(root_right, kGridCell);
    const std::int64_t maximum_y = host_floor_div(root_top, kGridCell);
    const std::uint64_t grid_width = static_cast<std::uint64_t>(maximum_x - base_x) + 1;
    const std::uint64_t grid_height = static_cast<std::uint64_t>(maximum_y - base_y) + 1;
    std::uint64_t grid_cells = 0;
    if (!checked_mul_u64(grid_width, grid_height, &grid_cells) ||
        grid_width > std::numeric_limits<std::uint32_t>::max() ||
        grid_height > std::numeric_limits<std::uint32_t>::max() ||
        grid_cells > options.max_grid_cells ||
        grid_cells > std::numeric_limits<std::uint32_t>::max()) {
        throw SceneError("projection uniform grid exceeds capacity");
    }
    const Grid grid = {
        base_x, base_y, static_cast<std::uint32_t>(grid_width),
        static_cast<std::uint32_t>(grid_height)};

    begin = Clock::now();
    cuda_require(cudaFree(nullptr), "CUDA context initialization");
    cuda_require(cudaDeviceSynchronize(), "CUDA initialization synchronize");
    end = Clock::now();
    timing.cuda_init_ms = milliseconds(begin, end);

    begin = Clock::now();
    DeviceBuffer<ContextGpu> d_contexts(lowered.contexts.size());
    DeviceBuffer<std::uint32_t> d_metal_contexts(lowered.metal_contexts.size());
    DeviceBuffer<std::uint64_t> d_metal_offsets(lowered.metal_offsets.size());
    DeviceBuffer<std::uint32_t> d_cut_contexts(lowered.cut_contexts.size());
    DeviceBuffer<std::uint64_t> d_cut_offsets(lowered.cut_offsets.size());
    DeviceBuffer<ProjectionCell> d_cells(lowered.cells.size());
    DeviceBuffer<ProjectionBox> d_templates(lowered.templates.size());
    DeviceBuffer<ProjectionBox> d_metal_boxes(lowered.metal_count);
    DeviceBuffer<ProjectionBox> d_cut_boxes(lowered.cut_count);
    DeviceBuffer<std::uint32_t> d_counts(grid_cells);
    DeviceBuffer<std::uint32_t> d_offsets(grid_cells + 1);
    DeviceBuffer<std::uint32_t> d_cursors(grid_cells);
    DeviceBuffer<unsigned long long> d_membership_total(1);
    DeviceBuffer<std::uint32_t> d_cut_counts(grid_cells);
    DeviceBuffer<std::uint32_t> d_cut_grid_offsets(grid_cells + 1);
    DeviceBuffer<std::uint32_t> d_cut_cursors(grid_cells);
    DeviceBuffer<unsigned long long> d_cut_membership_total(1);
    DeviceBuffer<std::uint32_t> d_status(1);
    DeviceBuffer<ProjectionCounters> d_counters(1);
    DeviceBuffer<CutPairCounters> d_cut_pair_counters(1);
    DeviceBuffer<ProjectionSample> d_samples(kSampleCapacity);
    DeviceBuffer<std::uint32_t> d_sample_count(1);
    upload(&d_contexts, lowered.contexts);
    upload(&d_metal_contexts, lowered.metal_contexts);
    upload(&d_metal_offsets, lowered.metal_offsets);
    upload(&d_cut_contexts, lowered.cut_contexts);
    upload(&d_cut_offsets, lowered.cut_offsets);
    upload(&d_cells, lowered.cells);
    upload(&d_templates, lowered.templates);
    cuda_require(
        cudaMemset(d_counts.get(), 0, grid_cells * sizeof(std::uint32_t)),
        "cudaMemset projection grid counts");
    cuda_require(
        cudaMemset(d_membership_total.get(), 0, sizeof(unsigned long long)),
        "cudaMemset projection membership total");
    cuda_require(
        cudaMemset(d_cut_counts.get(), 0, grid_cells * sizeof(std::uint32_t)),
        "cudaMemset cut grid counts");
    cuda_require(
        cudaMemset(d_cut_membership_total.get(), 0, sizeof(unsigned long long)),
        "cudaMemset cut membership total");
    cuda_require(
        cudaMemset(d_status.get(), 0, sizeof(std::uint32_t)), "cudaMemset projection status");
    cuda_require(
        cudaMemset(d_counters.get(), 0, sizeof(ProjectionCounters)),
        "cudaMemset projection counters");
    cuda_require(
        cudaMemset(d_cut_pair_counters.get(), 0, sizeof(CutPairCounters)),
        "cudaMemset cut pair counters");
    cuda_require(
        cudaMemset(d_sample_count.get(), 0, sizeof(std::uint32_t)),
        "cudaMemset projection sample count");
    projection_transform_gate<<<1, 8>>>(d_status.get());
    cuda_require(cudaGetLastError(), "projection transform gate launch");
    cuda_require(cudaDeviceSynchronize(), "projection upload synchronize");
    end = Clock::now();
    timing.alloc_upload_ms = milliseconds(begin, end);

    begin = Clock::now();
    expand_projection_metal_kernel<<<
        static_cast<unsigned int>(lowered.metal_contexts.size()), 128>>>(
        d_contexts.get(), d_metal_contexts.get(), d_metal_offsets.get(), d_cells.get(),
        d_templates.get(), static_cast<std::uint32_t>(lowered.metal_contexts.size()),
        d_metal_boxes.get(), d_status.get());
    cuda_require(cudaGetLastError(), "projection metal expansion launch");
    cuda_require(cudaDeviceSynchronize(), "projection metal expansion sync");
    end = Clock::now();
    timing.metal_expand_ms = milliseconds(begin, end);

    begin = Clock::now();
    expand_projection_cut_kernel<<<static_cast<unsigned int>(lowered.cut_contexts.size()), 128>>>(
        d_contexts.get(), d_cut_contexts.get(), d_cut_offsets.get(), d_cells.get(),
        d_templates.get(), static_cast<std::uint32_t>(lowered.cut_contexts.size()),
        d_cut_boxes.get(), d_status.get());
    cuda_require(cudaGetLastError(), "projection cut expansion launch");
    cuda_require(cudaDeviceSynchronize(), "projection cut expansion sync");
    end = Clock::now();
    timing.cut_expand_ms = milliseconds(begin, end);

    std::uint32_t host_status = 0;
    cuda_require(
        cudaMemcpy(&host_status, d_status.get(), sizeof(host_status), cudaMemcpyDeviceToHost),
        "cudaMemcpy projection metal status");
    if (host_status) {
        throw SceneError(
            "device metal expansion declined the scene, flags=" + std::to_string(host_status));
    }

    begin = Clock::now();
    const unsigned int metal_blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>(65535, (lowered.metal_count + 255) / 256));
    count_projection_grid_kernel<<<metal_blocks, 256>>>(
        d_metal_boxes.get(), static_cast<std::uint32_t>(lowered.metal_count), grid, d_counts.get(),
        d_membership_total.get(), d_status.get());
    cuda_require(cudaGetLastError(), "projection grid count launch");
    cuda_require(cudaDeviceSynchronize(), "projection grid count sync");
    unsigned long long membership_total = 0;
    cuda_require(
        cudaMemcpy(
            &membership_total, d_membership_total.get(), sizeof(membership_total),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy projection membership total");
    cuda_require(
        cudaMemcpy(&host_status, d_status.get(), sizeof(host_status), cudaMemcpyDeviceToHost),
        "cudaMemcpy projection grid count status");
    end = Clock::now();
    timing.grid_count_ms = milliseconds(begin, end);
    if (host_status) {
        throw SceneError(
            "device grid count declined the scene, flags=" + std::to_string(host_status));
    }
    if (membership_total < lowered.metal_count || membership_total > options.max_memberships ||
        membership_total > std::numeric_limits<std::uint32_t>::max()) {
        throw SceneError("projection grid memberships exceed configured/uint32 capacity");
    }

    begin = Clock::now();
    thrust::device_ptr<std::uint32_t> count_begin(d_counts.get());
    thrust::device_ptr<std::uint32_t> offset_begin(d_offsets.get());
    thrust::exclusive_scan(count_begin, count_begin + grid_cells, offset_begin);
    const std::uint32_t terminal = static_cast<std::uint32_t>(membership_total);
    cuda_require(
        cudaMemcpy(
            d_offsets.get() + grid_cells, &terminal, sizeof(terminal), cudaMemcpyHostToDevice),
        "cudaMemcpy projection terminal offset");
    cuda_require(
        cudaMemcpy(
            d_cursors.get(), d_offsets.get(), grid_cells * sizeof(std::uint32_t),
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy projection offsets to cursors");
    DeviceBuffer<std::uint32_t> d_members(membership_total);
    fill_projection_grid_kernel<<<metal_blocks, 256>>>(
        d_metal_boxes.get(), static_cast<std::uint32_t>(lowered.metal_count), grid, d_cursors.get(),
        d_members.get(), membership_total, d_status.get());
    cuda_require(cudaGetLastError(), "projection grid fill launch");
    const unsigned int grid_blocks =
        static_cast<unsigned int>(std::min<std::uint64_t>(65535, (grid_cells + 255) / 256));
    validate_grid_kernel<<<grid_blocks, 256>>>(
        d_counts.get(), d_offsets.get(), d_cursors.get(), grid_cells, d_status.get());
    cuda_require(cudaGetLastError(), "projection grid validate launch");
    cuda_require(cudaDeviceSynchronize(), "projection grid build sync");
    cuda_require(
        cudaMemcpy(&host_status, d_status.get(), sizeof(host_status), cudaMemcpyDeviceToHost),
        "cudaMemcpy projection grid build status");
    end = Clock::now();
    timing.grid_build_ms = milliseconds(begin, end);
    if (host_status) {
        throw SceneError(
            "device grid build declined the scene, flags=" + std::to_string(host_status));
    }

    begin = Clock::now();
    const unsigned int cut_blocks =
        static_cast<unsigned int>(std::min<std::uint64_t>(65535, (lowered.cut_count + 255) / 256));
    count_projection_grid_kernel<<<cut_blocks, 256>>>(
        d_cut_boxes.get(), static_cast<std::uint32_t>(lowered.cut_count), grid, d_cut_counts.get(),
        d_cut_membership_total.get(), d_status.get());
    cuda_require(cudaGetLastError(), "cut grid count launch");
    cuda_require(cudaDeviceSynchronize(), "cut grid count sync");
    unsigned long long cut_membership_total = 0;
    cuda_require(
        cudaMemcpy(
            &cut_membership_total, d_cut_membership_total.get(), sizeof(cut_membership_total),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy cut membership total");
    cuda_require(
        cudaMemcpy(&host_status, d_status.get(), sizeof(host_status), cudaMemcpyDeviceToHost),
        "cudaMemcpy cut grid count status");
    end = Clock::now();
    timing.cut_grid_count_ms = milliseconds(begin, end);
    if (host_status) {
        throw SceneError(
            "device cut grid count declined the scene, flags=" + std::to_string(host_status));
    }
    if (cut_membership_total < lowered.cut_count ||
        cut_membership_total > options.max_memberships ||
        cut_membership_total > std::numeric_limits<std::uint32_t>::max()) {
        throw SceneError("cut grid memberships exceed configured/uint32 capacity");
    }

    begin = Clock::now();
    thrust::device_ptr<std::uint32_t> cut_count_begin(d_cut_counts.get());
    thrust::device_ptr<std::uint32_t> cut_offset_begin(d_cut_grid_offsets.get());
    thrust::exclusive_scan(cut_count_begin, cut_count_begin + grid_cells, cut_offset_begin);
    const std::uint32_t cut_terminal = static_cast<std::uint32_t>(cut_membership_total);
    cuda_require(
        cudaMemcpy(
            d_cut_grid_offsets.get() + grid_cells, &cut_terminal, sizeof(cut_terminal),
            cudaMemcpyHostToDevice),
        "cudaMemcpy cut terminal offset");
    cuda_require(
        cudaMemcpy(
            d_cut_cursors.get(), d_cut_grid_offsets.get(), grid_cells * sizeof(std::uint32_t),
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy cut offsets to cursors");
    DeviceBuffer<std::uint32_t> d_cut_members(cut_membership_total);
    fill_projection_grid_kernel<<<cut_blocks, 256>>>(
        d_cut_boxes.get(), static_cast<std::uint32_t>(lowered.cut_count), grid, d_cut_cursors.get(),
        d_cut_members.get(), cut_membership_total, d_status.get());
    cuda_require(cudaGetLastError(), "cut grid fill launch");
    validate_grid_kernel<<<grid_blocks, 256>>>(
        d_cut_counts.get(), d_cut_grid_offsets.get(), d_cut_cursors.get(), grid_cells,
        d_status.get());
    cuda_require(cudaGetLastError(), "cut grid validate launch");
    cuda_require(cudaDeviceSynchronize(), "cut grid build sync");
    cuda_require(
        cudaMemcpy(&host_status, d_status.get(), sizeof(host_status), cudaMemcpyDeviceToHost),
        "cudaMemcpy cut grid build status");
    end = Clock::now();
    timing.cut_grid_build_ms = milliseconds(begin, end);
    if (host_status) {
        throw SceneError(
            "device cut grid build declined the scene, flags=" + std::to_string(host_status));
    }

    begin = Clock::now();
    query_projection_cut_pairs_kernel<<<cut_blocks, 256>>>(
        d_cut_boxes.get(), static_cast<std::uint32_t>(lowered.cut_count), grid, d_cut_counts.get(),
        d_cut_grid_offsets.get(), d_cut_members.get(), kCutSpacingDistance,
        d_cut_pair_counters.get(), d_status.get());
    cuda_require(cudaGetLastError(), "cut pair query launch");
    cuda_require(cudaDeviceSynchronize(), "cut pair query sync");
    end = Clock::now();
    timing.cut_pair_query_ms = milliseconds(begin, end);

    begin = Clock::now();
    if (lowered.cut_contexts.size() > std::numeric_limits<unsigned int>::max()) {
        throw SceneError("cut context launch exceeds CUDA grid-x domain");
    }
    query_projection_cuts_kernel<<<static_cast<unsigned int>(lowered.cut_contexts.size()), 128>>>(
        d_contexts.get(), d_cut_contexts.get(),
        static_cast<std::uint32_t>(lowered.cut_contexts.size()), d_cells.get(), d_templates.get(),
        d_metal_boxes.get(), static_cast<std::uint32_t>(lowered.metal_count), grid, d_counts.get(),
        d_offsets.get(), d_members.get(), kProjectionDistance, d_counters.get(), d_samples.get(),
        d_sample_count.get(), d_status.get());
    cuda_require(cudaGetLastError(), "projection cut query launch");
    cuda_require(cudaDeviceSynchronize(), "projection cut query sync");
    end = Clock::now();
    timing.cut_query_ms = milliseconds(begin, end);

    begin = Clock::now();
    ProjectionCounters counters{};
    CutPairCounters cut_pair_counters{};
    std::uint32_t sample_count = 0;
    std::array<ProjectionSample, kSampleCapacity> samples{};
    cuda_require(
        cudaMemcpy(&counters, d_counters.get(), sizeof(counters), cudaMemcpyDeviceToHost),
        "cudaMemcpy projection counters");
    cuda_require(
        cudaMemcpy(
            &cut_pair_counters, d_cut_pair_counters.get(), sizeof(cut_pair_counters),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy cut pair counters");
    cuda_require(
        cudaMemcpy(
            &sample_count, d_sample_count.get(), sizeof(sample_count), cudaMemcpyDeviceToHost),
        "cudaMemcpy projection sample count");
    cuda_require(
        cudaMemcpy(&host_status, d_status.get(), sizeof(host_status), cudaMemcpyDeviceToHost),
        "cudaMemcpy projection final status");
    if (sample_count) {
        cuda_require(
            cudaMemcpy(samples.data(), d_samples.get(), sizeof(samples), cudaMemcpyDeviceToHost),
            "cudaMemcpy projection samples");
    }
    end = Clock::now();
    timing.d2h_ms = milliseconds(begin, end);

    if (counters.cuts != lowered.cut_count ||
        counters.certified + counters.misses != counters.cuts) {
        throw SceneError("projection device counters violate conservation");
    }
    const __int128 classified_pairs = static_cast<__int128>(cut_pair_counters.duplicate_pairs) +
                                      cut_pair_counters.unsafe_touching_pairs +
                                      cut_pair_counters.spacing_pairs +
                                      cut_pair_counters.clean_pairs;
    if (cut_pair_counters.cuts_queried != lowered.cut_count ||
        classified_pairs != cut_pair_counters.candidate_pairs) {
        throw SceneError("cut pair device counters violate conservation");
    }
    if (cut_pair_counters.sample_claimed > 1 ||
        (!!cut_pair_counters.unsafe_touching_pairs != !!cut_pair_counters.sample_claimed) ||
        (cut_pair_counters.sample_claimed &&
         (cut_pair_counters.sample_cut >= lowered.cut_count ||
          cut_pair_counters.sample_other >= lowered.cut_count ||
          cut_pair_counters.sample_cut == cut_pair_counters.sample_other))) {
        throw SceneError("cut pair diagnostic sample violates invariants");
    }
    if (brute_force &&
        (brute_force->cuts != counters.cuts || brute_force->certified != counters.certified ||
         brute_force->misses != counters.misses)) {
        throw SceneError("projection GPU result disagrees with CPU brute force");
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
    release(&d_cut_pair_counters);
    release(&d_counters);
    release(&d_status);
    release(&d_cut_membership_total);
    release(&d_cut_members);
    release(&d_cut_cursors);
    release(&d_cut_grid_offsets);
    release(&d_cut_counts);
    release(&d_membership_total);
    release(&d_members);
    release(&d_cursors);
    release(&d_offsets);
    release(&d_counts);
    release(&d_cut_boxes);
    release(&d_metal_boxes);
    release(&d_templates);
    release(&d_cells);
    release(&d_cut_contexts);
    release(&d_cut_offsets);
    release(&d_metal_offsets);
    release(&d_metal_contexts);
    release(&d_contexts);
    if (cleanup_status != cudaSuccess) {
        throw SceneError(std::string("CUDA cleanup: ") + cudaGetErrorString(cleanup_status));
    }
    end = Clock::now();
    timing.cleanup_ms = milliseconds(begin, end);
    timing.total_ms = milliseconds(total_begin, Clock::now());

    const char *verdict = "CLEAN";
    int exit_code = 0;
    const bool projection_clean =
        !host_status && !counters.misses && !cut_pair_counters.unsafe_touching_pairs;
    const bool via1_size_clean =
        !host_status && lowered.qualified_cut_size && !cut_pair_counters.unsafe_touching_pairs;
    const bool via1_spacing_clean = !host_status && !cut_pair_counters.unsafe_touching_pairs &&
                                    !cut_pair_counters.spacing_pairs;
    if (host_status) {
        verdict = "UNCERTAIN";
        exit_code = 2;
    } else if (counters.misses || cut_pair_counters.unsafe_touching_pairs) {
        verdict = "FALLBACK";
        exit_code = 3;
    }
    std::cout << std::fixed << std::setprecision(3) << "PROJECTION_ENCLOSURE_GPU_ISLAND"
              << " verdict=" << verdict << " distance_dbu=" << kProjectionDistance
              << " dbu_um=" << std::setprecision(7) << scene.header.dbu << std::setprecision(3)
              << " contexts=" << lowered.contexts.size()
              << " metal_contexts=" << lowered.metal_contexts.size()
              << " cut_contexts=" << lowered.cut_contexts.size()
              << " metal_boxes=" << lowered.metal_count << " cuts=" << counters.cuts
              << " grid=" << grid.width << "x" << grid.height << " grid_cells=" << grid_cells
              << " memberships=" << membership_total << " cut_memberships=" << cut_membership_total
              << " candidate_boxes=" << counters.candidate_boxes
              << " certified=" << counters.certified << " misses=" << counters.misses
              << " cuts_pair_queried=" << cut_pair_counters.cuts_queried
              << " cut_pair_candidates=" << cut_pair_counters.candidate_pairs
              << " duplicate_cut_pairs=" << cut_pair_counters.duplicate_pairs
              << " unsafe_touching_cut_pairs=" << cut_pair_counters.unsafe_touching_pairs
              << " via1_spacing_violations=" << cut_pair_counters.spacing_pairs
              << " cut_pair_clean=" << cut_pair_counters.clean_pairs
              << " via1_size_clean=" << (via1_size_clean ? 1 : 0)
              << " via1_spacing_clean=" << (via1_spacing_clean ? 1 : 0)
              << " projection_clean=" << (projection_clean ? 1 : 0)
              << " decomposed_metal_templates=" << lowered.decomposed_metal_templates
              << " decomposition_rectangles=" << lowered.decomposition_rectangles
              << " device_flags=" << host_status
              << " scene_sha256=" << hex_digest(scene.header.scene_sha256, 32) << "\n";
    std::cout << "TIMING_MS"
              << " load_validate=" << timing.load_validate_ms
              << " cpu_lower=" << timing.cpu_lower_ms
              << " bruteforce_verify=" << timing.bruteforce_ms
              << " cuda_init=" << timing.cuda_init_ms << " alloc_upload=" << timing.alloc_upload_ms
              << " metal_expand=" << timing.metal_expand_ms
              << " cut_expand=" << timing.cut_expand_ms << " grid_count=" << timing.grid_count_ms
              << " grid_build=" << timing.grid_build_ms
              << " cut_grid_count=" << timing.cut_grid_count_ms
              << " cut_grid_build=" << timing.cut_grid_build_ms
              << " cut_pair_query=" << timing.cut_pair_query_ms
              << " cut_query=" << timing.cut_query_ms << " d2h=" << timing.d2h_ms
              << " cleanup=" << timing.cleanup_ms << " gpu_plan="
              << timing.alloc_upload_ms + timing.metal_expand_ms + timing.cut_expand_ms +
                     timing.grid_count_ms + timing.grid_build_ms + timing.cut_grid_count_ms +
                     timing.cut_grid_build_ms + timing.cut_pair_query_ms + timing.cut_query_ms +
                     timing.d2h_ms + timing.cleanup_ms
              << " warm_context_total=" << timing.total_ms - timing.cuda_init_ms
              << " cold_standalone_total=" << timing.total_ms << " total=" << timing.total_ms
              << "\n";
    const std::uint32_t copied_samples = std::min(sample_count, kSampleCapacity);
    for (std::uint32_t i = 0; i < copied_samples; ++i) {
        std::cout << "SAMPLE"
                  << " context=" << samples[i].context_id << " cut_local=" << samples[i].cut_local
                  << " box=" << samples[i].left << "," << samples[i].bottom << ","
                  << samples[i].right << "," << samples[i].top << "\n";
    }
    if (cut_pair_counters.sample_claimed) {
        const ProjectionBox &first = cut_pair_counters.sample_cut_box;
        const ProjectionBox &second = cut_pair_counters.sample_other_box;
        std::cout << "CUT_PAIR_SAMPLE"
                  << " cut=" << cut_pair_counters.sample_cut
                  << " other=" << cut_pair_counters.sample_other << " first=" << first.left << ","
                  << first.bottom << "," << first.right << "," << first.top
                  << " second=" << second.left << "," << second.bottom << "," << second.right << ","
                  << second.top << "\n";
    }
    return exit_code;
}

} // namespace

int
main(int argc, char **argv)
{
    const Clock::time_point total_begin = Clock::now();
    try {
        return run_projection_plan(parse_projection_options(argc, argv), total_begin);
    } catch (const std::exception &error) {
        std::cerr << "PROJECTION_ENCLOSURE_GPU_ISLAND"
                  << " verdict=UNCERTAIN error=\"" << error.what() << "\"\n";
        return 2;
    }
}
