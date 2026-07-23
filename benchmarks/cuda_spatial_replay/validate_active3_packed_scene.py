#!/usr/bin/env python3
"""Fail-closed reader and structural validator for KACTSCN1 packed scenes."""

from __future__ import annotations

import argparse
import hashlib
import math
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MAGIC = b"KACTSCN\x00"
VERSION = 1
HEADER_BYTES = 256
ENDIAN_TAG = 0x01020304
FORMAT_FLAGS = 1
COORDINATE_BITS = 64
LAYER_COUNT = 2
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

HEADER = struct.Struct("<8s14Id18Q32sQ")
CELL = struct.Struct("<QQIIII6Q16q")
INSTANCE = struct.Struct("<4Q6q4I")
POLYGON = struct.Struct("<3Q2I4q")
EDGE = struct.Struct("<2Q4q2I")


class SceneError(ValueError):
    pass


@dataclass(frozen=True)
class InstanceRecord:
    ident: int
    parent: int
    child: int
    occurrences: int
    dx: int
    dy: int
    ax: int
    ay: int
    bx: int
    by: int
    columns: int
    rows: int
    transform: int


@dataclass(frozen=True)
class PolygonRecord:
    ident: int
    cell: int
    edge_begin: int
    edge_count: int
    layer: int
    bbox: tuple[int, int, int, int]


def checked_i64(value: int, what: str) -> int:
    if value < INT64_MIN or value > INT64_MAX:
        raise SceneError(f"{what} overflows signed int64")
    return value


def valid_bbox(box: tuple[int, int, int, int]) -> bool:
    return box[0] <= box[2] and box[1] <= box[3]


def union_bbox(
    a: Optional[tuple[int, int, int, int]],
    b: Optional[tuple[int, int, int, int]],
) -> Optional[tuple[int, int, int, int]]:
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def transform_point(code: int, x: int, y: int) -> tuple[int, int]:
    transforms = (
        (x, y),
        (-y, x),
        (-x, -y),
        (y, -x),
        (x, -y),
        (y, x),
        (-x, y),
        (-y, -x),
    )
    if code >= len(transforms):
        raise SceneError(f"invalid transform code {code}")
    tx, ty = transforms[code]
    # Python integers do not overflow. Check the raw orthogonal transform
    # before translation so an int64 consumer cannot encounter -INT64_MIN
    # even if the following displacement would cancel the overflow.
    return (
        checked_i64(tx, "transformed x before translation"),
        checked_i64(ty, "transformed y before translation"),
    )


def manhattan_segments_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    if ay1 == ay2 and by1 == by2:
        return ay1 == by1 and max(min(ax1, ax2), min(bx1, bx2)) <= min(
            max(ax1, ax2), max(bx1, bx2)
        )
    if ax1 == ax2 and bx1 == bx2:
        return ax1 == bx1 and max(min(ay1, ay2), min(by1, by2)) <= min(
            max(ay1, ay2), max(by1, by2)
        )
    horizontal, vertical = (
        (first, second) if ay1 == ay2 else (second, first)
    )
    hx1, hy, hx2, _ = horizontal
    vx, vy1, _, vy2 = vertical
    return (
        min(hx1, hx2) <= vx <= max(hx1, hx2)
        and min(vy1, vy2) <= hy <= max(vy1, vy2)
    )


def positive_collinear_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    if ay1 == ay2 and by1 == by2 and ay1 == by1:
        return min(max(ax1, ax2), max(bx1, bx2)) > max(
            min(ax1, ax2), min(bx1, bx2)
        )
    if ax1 == ax2 and bx1 == bx2 and ax1 == bx1:
        return min(max(ay1, ay2), max(by1, by2)) > max(
            min(ay1, ay2), min(by1, by2)
        )
    return False


def transformed_array_bbox(
    box: tuple[int, int, int, int], instance: InstanceRecord
) -> tuple[int, int, int, int]:
    corners = []
    for x, y in (
        (box[0], box[1]),
        (box[0], box[3]),
        (box[2], box[1]),
        (box[2], box[3]),
    ):
        tx, ty = transform_point(instance.transform, x, y)
        corners.append(
            (
                checked_i64(tx + instance.dx, "transformed x"),
                checked_i64(ty + instance.dy, "transformed y"),
            )
        )
    base = (
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    )

    a_last_x = checked_i64((instance.columns - 1) * instance.ax, "column x extent")
    a_last_y = checked_i64((instance.columns - 1) * instance.ay, "column y extent")
    b_last_x = checked_i64((instance.rows - 1) * instance.bx, "row x extent")
    b_last_y = checked_i64((instance.rows - 1) * instance.by, "row y extent")
    offsets_x = (0, a_last_x, b_last_x, checked_i64(a_last_x + b_last_x, "array x extent"))
    offsets_y = (0, a_last_y, b_last_y, checked_i64(a_last_y + b_last_y, "array y extent"))
    return (
        checked_i64(base[0] + min(offsets_x), "array bbox left"),
        checked_i64(base[1] + min(offsets_y), "array bbox bottom"),
        checked_i64(base[2] + max(offsets_x), "array bbox right"),
        checked_i64(base[3] + max(offsets_y), "array bbox top"),
    )


def section(
    data: mmap.mmap, offset: int, size: int, alignment: int, label: str
) -> None:
    if offset < HEADER_BYTES or offset % alignment:
        raise SceneError(f"{label} offset is not {alignment}-byte aligned")
    end = offset + size
    if end < offset or end > len(data):
        raise SceneError(f"{label} range is outside the file")


def require_zero(data: mmap.mmap, begin: int, end: int, label: str) -> None:
    if begin > end:
        raise SceneError(f"{label} has inverted range")
    if any(data[begin:end]):
        raise SceneError(f"{label} contains nonzero padding")


def parse_scene(path: Path) -> str:
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size < HEADER_BYTES:
            raise SceneError("file is shorter than the fixed header")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            values = HEADER.unpack_from(data)
            (
                magic,
                version,
                header_bytes,
                endian_tag,
                flags,
                coordinate_bits,
                layer_count,
                cell_record_bytes,
                instance_record_bytes,
                polygon_record_bytes,
                edge_record_bytes,
                well_layer,
                well_datatype,
                active_layer,
                active_datatype,
                dbu,
                root_cell,
                cell_count,
                instance_count,
                polygon_count,
                edge_count,
                names_offset,
                names_bytes,
                cells_offset,
                cells_bytes,
                instances_offset,
                instances_bytes,
                polygons_offset,
                polygons_bytes,
                edges_offset,
                edges_bytes,
                file_bytes,
                payload_offset,
                payload_bytes,
                scene_sha256,
                reserved,
            ) = values

            if magic != MAGIC:
                raise SceneError(f"bad magic {magic!r}")
            if version != VERSION or header_bytes != HEADER_BYTES:
                raise SceneError("unsupported version/header size")
            if endian_tag != ENDIAN_TAG:
                raise SceneError("bad endian tag")
            if flags != FORMAT_FLAGS:
                raise SceneError("unsupported format flags")
            if coordinate_bits != COORDINATE_BITS or layer_count != LAYER_COUNT:
                raise SceneError("unsupported coordinate width/layer count")
            expected_sizes = (CELL.size, INSTANCE.size, POLYGON.size, EDGE.size)
            actual_sizes = (
                cell_record_bytes,
                instance_record_bytes,
                polygon_record_bytes,
                edge_record_bytes,
            )
            if actual_sizes != expected_sizes:
                raise SceneError(f"record-size mismatch: {actual_sizes} != {expected_sizes}")
            if (well_layer, well_datatype) == (active_layer, active_datatype):
                raise SceneError("WELL and ACTIVE layers alias")
            if not math.isfinite(dbu) or dbu <= 0.0:
                raise SceneError("database unit is not finite and positive")
            if not cell_count or root_cell >= cell_count:
                raise SceneError("invalid cell count/root ID")
            if reserved:
                raise SceneError("nonzero reserved header field")
            if file_bytes != len(data):
                raise SceneError(f"header file size {file_bytes} != actual {len(data)}")
            if payload_offset != HEADER_BYTES or payload_bytes != len(data) - HEADER_BYTES:
                raise SceneError("noncanonical payload range")

            digest = hashlib.sha256()
            digest.update(data[:216])
            digest.update(b"\x00" * 40)
            digest.update(data[payload_offset : payload_offset + payload_bytes])
            if digest.digest() != scene_sha256:
                raise SceneError("scene SHA-256 mismatch")

            if cells_bytes != cell_count * CELL.size:
                raise SceneError("cell section byte count mismatch")
            if instances_bytes != instance_count * INSTANCE.size:
                raise SceneError("instance section byte count mismatch")
            if polygons_bytes != polygon_count * POLYGON.size:
                raise SceneError("polygon section byte count mismatch")
            if edges_bytes != edge_count * EDGE.size:
                raise SceneError("edge section byte count mismatch")
            if names_offset != HEADER_BYTES:
                raise SceneError("name section must begin immediately after the header")

            section(data, names_offset, names_bytes, 64, "names")
            section(data, cells_offset, cells_bytes, 64, "cells")
            section(data, instances_offset, instances_bytes, 64, "instances")
            section(data, polygons_offset, polygons_bytes, 64, "polygons")
            section(data, edges_offset, edges_bytes, 64, "edges")
            ordered_sections = (
                ("names", names_offset, names_bytes),
                ("cells", cells_offset, cells_bytes),
                ("instances", instances_offset, instances_bytes),
                ("polygons", polygons_offset, polygons_bytes),
                ("edges", edges_offset, edges_bytes),
            )
            previous_end = HEADER_BYTES
            for label, offset, size in ordered_sections:
                expected_offset = (previous_end + 63) & ~63
                if offset != expected_offset:
                    raise SceneError(f"{label} section has a noncanonical offset")
                require_zero(data, previous_end, offset, f"padding before {label}")
                previous_end = offset + size
            if len(data) != (previous_end + 63) & ~63:
                raise SceneError("file has a noncanonical aligned size")
            require_zero(data, previous_end, len(data), "trailing padding")

            cells = []
            previous_name: Optional[bytes] = None
            next_name = 0
            next_instance = next_polygon = next_edge = 0
            for ident in range(cell_count):
                record = CELL.unpack_from(data, cells_offset + ident * CELL.size)
                (
                    record_id,
                    name_offset,
                    name_size,
                    local_mask,
                    subtree_mask,
                    cell_flags,
                    instance_begin,
                    record_instance_count,
                    polygon_begin,
                    record_polygon_count,
                    edge_begin,
                    record_edge_count,
                    *bbox_values,
                ) = record
                if record_id != ident:
                    raise SceneError(f"cell {ident}: non-dense ID {record_id}")
                if cell_flags or local_mask & ~3 or subtree_mask & ~3:
                    raise SceneError(f"cell {ident}: invalid flags/layer masks")
                if local_mask & ~subtree_mask:
                    raise SceneError(f"cell {ident}: local layer is absent from subtree")
                if name_offset != next_name or name_offset + name_size > names_bytes:
                    raise SceneError(f"cell {ident}: noncontiguous/out-of-range name")
                name = data[
                    names_offset + name_offset : names_offset + name_offset + name_size
                ]
                if not name or b"\x00" in name:
                    raise SceneError(f"cell {ident}: empty/NUL-containing name")
                if previous_name is not None and name <= previous_name:
                    raise SceneError("cell names are not in strict bytewise order")
                previous_name = name
                next_name += name_size
                if instance_begin != next_instance:
                    raise SceneError(f"cell {ident}: noncontiguous instance range")
                if polygon_begin != next_polygon:
                    raise SceneError(f"cell {ident}: noncontiguous polygon range")
                if edge_begin != next_edge:
                    raise SceneError(f"cell {ident}: noncontiguous edge range")
                next_instance += record_instance_count
                next_polygon += record_polygon_count
                next_edge += record_edge_count
                if next_instance > instance_count or next_polygon > polygon_count or next_edge > edge_count:
                    raise SceneError(f"cell {ident}: record range exceeds section")

                local_boxes = []
                subtree_boxes = []
                for layer in range(2):
                    local_box = tuple(bbox_values[layer * 4 : layer * 4 + 4])
                    subtree_box = tuple(bbox_values[8 + layer * 4 : 12 + layer * 4])
                    if local_mask & (1 << layer):
                        if not valid_bbox(local_box):
                            raise SceneError(f"cell {ident}: invalid local bbox on layer {layer}")
                        local_boxes.append(local_box)
                    else:
                        if local_box != (0, 0, 0, 0):
                            raise SceneError(f"cell {ident}: nonzero absent local bbox")
                        local_boxes.append(None)
                    if subtree_mask & (1 << layer):
                        if not valid_bbox(subtree_box):
                            raise SceneError(f"cell {ident}: invalid subtree bbox on layer {layer}")
                        subtree_boxes.append(subtree_box)
                    else:
                        if subtree_box != (0, 0, 0, 0):
                            raise SceneError(f"cell {ident}: nonzero absent subtree bbox")
                        subtree_boxes.append(None)
                cells.append(
                    {
                        "name": name,
                        "instance_begin": instance_begin,
                        "instance_count": record_instance_count,
                        "polygon_begin": polygon_begin,
                        "polygon_count": record_polygon_count,
                        "edge_begin": edge_begin,
                        "edge_count": record_edge_count,
                        "local_boxes": local_boxes,
                        "subtree_boxes": subtree_boxes,
                    }
                )
            if (next_instance, next_polygon, next_edge) != (
                instance_count,
                polygon_count,
                edge_count,
            ):
                raise SceneError("cell ranges do not partition record sections")
            if next_name != names_bytes:
                raise SceneError("cell names do not partition the name section")

            instances: list[InstanceRecord] = []
            instance_owner = 0
            previous_instance_key: Optional[tuple[int, ...]] = None
            for ident in range(instance_count):
                old_owner = instance_owner
                while (
                    instance_owner + 1 < cell_count
                    and ident
                    >= cells[instance_owner]["instance_begin"]
                    + cells[instance_owner]["instance_count"]
                ):
                    instance_owner += 1
                if instance_owner != old_owner:
                    previous_instance_key = None
                record = INSTANCE.unpack_from(
                    data, instances_offset + ident * INSTANCE.size
                )
                (
                    record_id,
                    parent,
                    child,
                    occurrences,
                    dx,
                    dy,
                    ax,
                    ay,
                    bx,
                    by,
                    columns,
                    rows,
                    transform,
                    instance_flags,
                ) = record
                if record_id != ident or parent != instance_owner:
                    raise SceneError(f"instance {ident}: bad dense ID/parent")
                if child >= cell_count:
                    raise SceneError(f"instance {ident}: child outside cell table")
                if not columns or not rows or occurrences != columns * rows:
                    raise SceneError(f"instance {ident}: malformed array dimensions")
                if transform >= 8 or instance_flags:
                    raise SceneError(f"instance {ident}: invalid transform/flags")
                if columns == 1 and (ax or ay):
                    raise SceneError(f"instance {ident}: noncanonical singleton column pitch")
                if rows == 1 and (bx or by):
                    raise SceneError(f"instance {ident}: noncanonical singleton row pitch")
                if columns > 1 and not (ax or ay):
                    raise SceneError(f"instance {ident}: zero repeated-column pitch")
                if rows > 1 and not (bx or by):
                    raise SceneError(f"instance {ident}: zero repeated-row pitch")
                a_last_x = checked_i64(
                    (columns - 1) * ax, f"instance {ident} column x extent"
                )
                a_last_y = checked_i64(
                    (columns - 1) * ay, f"instance {ident} column y extent"
                )
                b_last_x = checked_i64(
                    (rows - 1) * bx, f"instance {ident} row x extent"
                )
                b_last_y = checked_i64(
                    (rows - 1) * by, f"instance {ident} row y extent"
                )
                for offset_x, offset_y in (
                    (0, 0),
                    (a_last_x, a_last_y),
                    (b_last_x, b_last_y),
                    (
                        checked_i64(a_last_x + b_last_x, "combined array x extent"),
                        checked_i64(a_last_y + b_last_y, "combined array y extent"),
                    ),
                ):
                    checked_i64(dx + offset_x, f"instance {ident} array origin x")
                    checked_i64(dy + offset_y, f"instance {ident} array origin y")
                instance_key = (
                    child,
                    transform,
                    dx,
                    dy,
                    columns,
                    rows,
                    ax,
                    ay,
                    bx,
                    by,
                )
                if (
                    previous_instance_key is not None
                    and instance_key < previous_instance_key
                ):
                    raise SceneError(
                        f"instance {ident}: parent-local records are not canonical"
                    )
                previous_instance_key = instance_key
                instances.append(
                    InstanceRecord(
                        record_id,
                        parent,
                        child,
                        occurrences,
                        dx,
                        dy,
                        ax,
                        ay,
                        bx,
                        by,
                        columns,
                        rows,
                        transform,
                    )
                )

            polygons: list[PolygonRecord] = []
            polygon_owner = 0
            next_polygon_edge = 0
            recomputed_local: list[list[Optional[tuple[int, int, int, int]]]] = [
                [None, None] for _ in range(cell_count)
            ]
            for ident in range(polygon_count):
                while (
                    polygon_owner + 1 < cell_count
                    and ident
                    >= cells[polygon_owner]["polygon_begin"]
                    + cells[polygon_owner]["polygon_count"]
                ):
                    polygon_owner += 1
                record = POLYGON.unpack_from(
                    data, polygons_offset + ident * POLYGON.size
                )
                record_id, cell_id, polygon_edge_begin, polygon_edge_count, layer, *box = record
                bbox = tuple(box)
                if record_id != ident or cell_id != polygon_owner:
                    raise SceneError(f"polygon {ident}: bad dense ID/cell")
                if layer >= 2 or polygon_edge_count < 4 or not valid_bbox(bbox):
                    raise SceneError(f"polygon {ident}: invalid layer/count/bbox")
                if polygon_edge_begin != next_polygon_edge:
                    raise SceneError(f"polygon {ident}: noncontiguous edge range")
                next_polygon_edge += polygon_edge_count
                if next_polygon_edge > edge_count:
                    raise SceneError(f"polygon {ident}: edge range outside table")
                recomputed_local[cell_id][layer] = union_bbox(
                    recomputed_local[cell_id][layer], bbox
                )
                polygons.append(
                    PolygonRecord(
                        record_id,
                        cell_id,
                        polygon_edge_begin,
                        polygon_edge_count,
                        layer,
                        bbox,
                    )
                )
            if next_polygon_edge != edge_count:
                raise SceneError("polygon ranges do not partition edge table")
            for cell_id, cell in enumerate(cells):
                begin = cell["polygon_begin"]
                end = begin + cell["polygon_count"]
                cell_polygons = polygons[begin:end]
                expected_edge_count = sum(
                    polygon.edge_count for polygon in cell_polygons
                )
                if expected_edge_count != cell["edge_count"]:
                    raise SceneError(
                        f"cell {cell_id}: edge count does not match its polygons"
                    )
                if cell_polygons and cell_polygons[0].edge_begin != cell["edge_begin"]:
                    raise SceneError(
                        f"cell {cell_id}: edge range does not begin with its polygons"
                    )

            polygon_vertices: list[list[tuple[int, int]]] = [[] for _ in polygons]
            edge_owner = 0
            for ident in range(edge_count):
                while (
                    edge_owner + 1 < polygon_count
                    and ident
                    >= polygons[edge_owner].edge_begin + polygons[edge_owner].edge_count
                ):
                    edge_owner += 1
                record = EDGE.unpack_from(data, edges_offset + ident * EDGE.size)
                record_id, polygon_id, x1, y1, x2, y2, local_index, layer = record
                if record_id != ident or polygon_id != edge_owner:
                    raise SceneError(f"edge {ident}: bad dense ID/polygon")
                polygon = polygons[polygon_id]
                if local_index != ident - polygon.edge_begin or layer != polygon.layer:
                    raise SceneError(f"edge {ident}: bad local index/layer")
                if (x1 == x2 and y1 == y2) or not (x1 == x2 or y1 == y2):
                    raise SceneError(f"edge {ident}: degenerate/non-Manhattan")
                polygon_vertices[polygon_id].append((x1, y1))

            for polygon, vertices in zip(polygons, polygon_vertices):
                if len(vertices) != len(set(vertices)):
                    raise SceneError(f"polygon {polygon.ident}: repeated vertex")
                contour_edges = []
                for index, point in enumerate(vertices):
                    following = vertices[(index + 1) % len(vertices)]
                    edge = EDGE.unpack_from(
                        data,
                        edges_offset + (polygon.edge_begin + index) * EDGE.size,
                    )
                    if (edge[4], edge[5]) != following:
                        raise SceneError(f"polygon {polygon.ident}: open edge chain")
                    contour_edges.append(
                        (point[0], point[1], following[0], following[1])
                    )
                for first in range(len(contour_edges)):
                    for second in range(first + 1, len(contour_edges)):
                        if not manhattan_segments_intersect(
                            contour_edges[first], contour_edges[second]
                        ):
                            continue
                        adjacent = second == first + 1 or (
                            first == 0 and second == len(contour_edges) - 1
                        )
                        if adjacent and not positive_collinear_overlap(
                            contour_edges[first], contour_edges[second]
                        ):
                            continue
                        raise SceneError(
                            f"polygon {polygon.ident}: self-intersecting "
                            f"edges {first} and {second}"
                        )
                twice_area = sum(
                    point[0] * vertices[(index + 1) % len(vertices)][1]
                    - vertices[(index + 1) % len(vertices)][0] * point[1]
                    for index, point in enumerate(vertices)
                )
                if twice_area >= 0:
                    raise SceneError(
                        f"polygon {polygon.ident}: outer contour is not clockwise"
                    )
                bbox = (
                    min(point[0] for point in vertices),
                    min(point[1] for point in vertices),
                    max(point[0] for point in vertices),
                    max(point[1] for point in vertices),
                )
                if bbox != polygon.bbox:
                    raise SceneError(f"polygon {polygon.ident}: bbox mismatch")
                if vertices != min(vertices[index:] + vertices[:index] for index in range(len(vertices))):
                    raise SceneError(f"polygon {polygon.ident}: contour is not canonically rotated")

            for cell_id, cell in enumerate(cells):
                if recomputed_local[cell_id] != cell["local_boxes"]:
                    raise SceneError(f"cell {cell_id}: local bbox does not match polygons")
                begin = cell["polygon_begin"]
                end = begin + cell["polygon_count"]
                previous_polygon_key = None
                for polygon in polygons[begin:end]:
                    polygon_key = (
                        polygon.layer,
                        tuple(
                            coordinate
                            for point in polygon_vertices[polygon.ident]
                            for coordinate in point
                        ),
                    )
                    if (
                        previous_polygon_key is not None
                        and polygon_key < previous_polygon_key
                    ):
                        raise SceneError(
                            f"cell {cell_id}: polygons are not in canonical order"
                        )
                    previous_polygon_key = polygon_key

            reachable = set()
            frontier = [root_cell]
            while frontier:
                cell_id = frontier.pop()
                if cell_id in reachable:
                    continue
                reachable.add(cell_id)
                begin = cells[cell_id]["instance_begin"]
                end = begin + cells[cell_id]["instance_count"]
                frontier.extend(instance.child for instance in instances[begin:end])
            if len(reachable) != cell_count:
                raise SceneError("cell table contains cells unreachable from the root")

            state = [0] * cell_count
            recomputed_subtree: list[
                Optional[list[Optional[tuple[int, int, int, int]]]]
            ] = [None] * cell_count

            def compute_subtree(cell_id: int):
                if state[cell_id] == 1:
                    raise SceneError(f"hierarchy cycle through cell {cell_id}")
                if state[cell_id] == 2:
                    return recomputed_subtree[cell_id]
                state[cell_id] = 1
                boxes = list(recomputed_local[cell_id])
                begin = cells[cell_id]["instance_begin"]
                end = begin + cells[cell_id]["instance_count"]
                for instance in instances[begin:end]:
                    child_boxes = compute_subtree(instance.child)
                    assert child_boxes is not None
                    for layer, child_box in enumerate(child_boxes):
                        if child_box is not None:
                            boxes[layer] = union_bbox(
                                boxes[layer], transformed_array_bbox(child_box, instance)
                            )
                recomputed_subtree[cell_id] = boxes
                state[cell_id] = 2
                return boxes

            compute_subtree(root_cell)
            for cell_id in range(cell_count):
                expected = compute_subtree(cell_id)
                if expected != cells[cell_id]["subtree_boxes"]:
                    raise SceneError(f"cell {cell_id}: subtree bbox mismatch")

            total_occurrences = sum(instance.occurrences for instance in instances)
            full_sha = hashlib.sha256(data).hexdigest()
            return (
                f"KACTSCN1 ok path={path} bytes={len(data)} dbu={dbu:.12g} "
                f"root={root_cell} cells={cell_count} instances={instance_count} "
                f"array_occurrences={total_occurrences} polygons={polygon_count} "
                f"edges={edge_count} layers={well_layer}/{well_datatype},{active_layer}/{active_datatype} "
                f"scene_sha256={scene_sha256.hex()} file_sha256={full_sha}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate deterministic KACTSCN1 hierarchy/geometry files"
    )
    parser.add_argument("scene", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        for path in args.scene:
            print(parse_scene(path))
    except (OSError, SceneError, struct.error) as error:
        print(f"KACTSCN1 invalid: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
