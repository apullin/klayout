# ACTIVE.3 packed scene (`KACTSCN1`)

Status: capture/replay development format. It is deliberately narrower than
GDS and is not yet a public or stable KLayout ABI.

`active3_packed_scene_export.rb` lowers the exact derived-layout capture into a
pointer-free, memory-mappable scene. It retains cells and regular arrays, so a
24-million-polygon flat workload is represented by its stored templates and
hierarchy rather than by 24 million copied polygons. The companion
`validate_active3_packed_scene.py` reader performs an independent fail-closed
validation before this data is suitable for a CUDA upload.

## Qualified input

The v1 exporter accepts exactly two polygon operand layers, defaulting to
101/0 (WELL) and 102/0 (ACTIVE). It accepts:

- a named, reachable root hierarchy;
- simple orthogonal transforms encoded with KLayout codes `r0..m135`;
- scalar instances and compressed regular arrays;
- property-free boxes and polygons;
- closed, simple, hole-free, clockwise, nondegenerate Manhattan contours; and
- signed 64-bit integer coordinates and all intermediate transformed boxes.

It rejects complex or irregular transforms/arrays, hierarchy cycles, missing
children, properties, paths/texts/edges or other non-polygons on operand
layers, holes, non-Manhattan or malformed contours, zero array pitch in a
repeated dimension, count/offset overflow, and coordinate arithmetic overflow.
Unsupported input produces no accepted output. Output creation is exclusive,
and publication is atomic and no-clobber: the exporter writes and fsyncs a
private sibling, hard-links the complete inode at the final path, and fsyncs
the destination directory. A caught failure after publication removes only
the final path created by that exporter. A process killed before publication
can leave a hidden temporary file, but never a partial final path.

The clockwise outer-contour requirement is semantic, not merely canonical
formatting: ACTIVE.3 uses each directed edge's inside/outside orientation.
Consumers must preserve that direction. Consumers must also use checked or
widened arithmetic for every transform, negation, array offset, and
translation; a final in-range coordinate does not make an overflowing
intermediate operation acceptable.

## Byte order and identity

Every integer and the IEEE-754 database-unit double are little-endian. IDs are
dense unsigned 64-bit indices starting at zero. Cells are ordered by strict
bytewise cell name. Instances are ordered by their complete semantic tuple.
Polygons are ordered by logical layer and contour; a directed contour is
cyclically rotated to its lexicographically least vertex sequence but is never
reversed. These rules make repeated export of one source byte-identical.

Layer code 0 is WELL and layer code 1 is ACTIVE. Cell boxes contain inclusive
coordinate extrema. A layer-mask bit distinguishes an absent box (stored as
four zeroes) from a real zero-sized-coordinate box.

Stored local contours are clockwise. KLayout also normalizes a polygon
transformed through a mirrored hierarchy context back to clockwise; simply
transforming its stored vertices would instead reverse its direction. A scene
consumer must therefore track the composed context's mirror parity. For a
mirrored context, it emits each directed edge as `T(p2) -> T(p1)` (and reverses
contour traversal if ordered-contour output is required). This restoration of
clockwise/interior-right semantics is mandatory for direction-sensitive
checks such as ACTIVE.3.

The file and every fixed-record section are 64-byte aligned. Alignment bytes
are zero. The header contains SHA-256 of the complete header and payload with
the digest and reserved header fields treated as zero. This binds metadata,
records, and padding. It detects corruption; it is not an authenticity claim.

### External provenance is mandatory

Structural validation does **not** prove that a self-consistent scene came
from the intended source GDS, top cell, derivation deck, exporter revision, or
rule invocation. Before a clean GPU certificate can replace production work,
the caller must bind those external identities to an expected scene SHA-256
(or an equivalently strong in-process revision token) and reject a mismatch.
The validated-capture table below records that benchmark provenance, but it
is not embedded as an authenticity assertion in `KACTSCN1`.

## Header

The fixed 256-byte header has this layout:

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | `KACTSCN\0` |
| 8 | `u32` | format version, 1 |
| 12 | `u32` | header bytes, 256 |
| 16 | `u32` | endian tag, `0x01020304` |
| 20 | `u32` | flags, 1 (inclusive-extrema boxes) |
| 24 | `u32` | coordinate bits, 64 |
| 28 | `u32` | logical layer count, 2 |
| 32..44 | `u32[4]` | cell/instance/polygon/edge record sizes |
| 48..60 | `u32[4]` | WELL layer/datatype, ACTIVE layer/datatype |
| 64 | `f64` | database unit in micrometres |
| 72 | `u64` | root cell ID |
| 80..104 | `u64[4]` | cell, instance, polygon, edge counts |
| 112..184 | `u64[10]` | offset/byte-size pairs for names, cells, instances, polygons, edges |
| 192 | `u64` | total file bytes |
| 200 | `u64` | payload offset (256) |
| 208 | `u64` | payload bytes |
| 216 | `u8[32]` | scene SHA-256; hash bytes 216..255 as zero |
| 248 | `u64` | reserved, zero |

The names section concatenates opaque, nonempty, NUL-free cell-name bytes.
Cell name offsets are relative to the beginning of that section.

## POD records

### Cell: 208 bytes

```text
u64 cell_id, name_offset
u32 name_bytes, local_layer_mask, subtree_layer_mask, flags_zero
u64 instance_begin, instance_count
u64 polygon_begin, polygon_count
u64 edge_begin, edge_count
i64 local_bbox[2][4]       # WELL then ACTIVE; left,bottom,right,top
i64 subtree_bbox[2][4]
```

Cell ranges partition their corresponding global tables. A subtree box is the
checked union of local geometry and every transformed child-array subtree box.

### Instance/array: 96 bytes

```text
u64 instance_id, parent_cell_id, child_cell_id, occurrence_count
i64 dx, dy, ax, ay, bx, by
u32 columns, rows, transform_code, flags_zero
```

`transform_code` uses KLayout's 0..7 `r0,r90,r180,r270,m0,m45,m90,m135`
mapping. The transform is applied to child coordinates, then `(dx,dy)` and
the parent-coordinate array offset `column*(ax,ay)+row*(bx,by)` are added.
Singleton dimensions have a canonical zero pitch.

The transform code also contributes its mirror parity to the composed
hierarchy context. Consumers apply the clockwise direction restoration
described under "Byte order and identity" after transforming local geometry.

### Polygon: 64 bytes

```text
u64 polygon_id, cell_id, edge_begin
u32 edge_count, layer_code
i64 bbox[4]
```

### Directed edge: 56 bytes

```text
u64 edge_id, polygon_id
i64 x1, y1, x2, y2
u32 contour_index, layer_code
```

Edges for a polygon are contiguous, closed in order, and preserve the source
contour direction. No KLayout pointer, property handle, iterator, or receiver
object appears in the file.

## Export and validate

```sh
KLAYOUT=/path/to/klayout
"$KLAYOUT" -b -r active3_packed_scene_export.rb \
  -rd input=/tmp/active3-scene-64k.gds \
  -rd topcell=KLAYOUT_CUDA_ACTIVE3_SCENE \
  -rd output=/tmp/active3-scene-64k.kact

python3 validate_active3_packed_scene.py /tmp/active3-scene-64k.kact
```

The validator checks the header and SHA-256; all offsets, padding, dense IDs,
and parent ranges; bytewise name order; array dimensions and transforms;
directed contour closure/canonicalization, simplicity, clockwise orientation,
Manhattan geometry, areas and boxes; hierarchy acyclicity; and independently
recomputed per-cell local and subtree boxes without flattening.

A CUDA consumer additionally needs a differential fixture covering all eight
transform codes and nested mirror parity. Expected world contours remain
clockwise, matching KLayout's recursive transformed-polygon semantics.

## Validated captures

The initial 2026-07-23 qualification used both exact derived captures:

| Capture | Source GDS SHA-256 | Packed bytes | Cells | Instance records / represented occurrences | Polygons | Edges | Packed file SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| 64K | `7374a7ccc286d96fa8deca534b5dcd6a1959c12d26f6173e43e2ad57d31068da` | 1,430,656 | 132 | 12,610 / 78,187 | 637 | 2,602 | `c9098b76f08ef114bd0377d0a66d96a98e330f39c7e8fc78307ed3c3662b62d0` |
| x2 | `98d754b4448a0cab7b0b1e994c0b524a35c36d579b284621e016062234415e4d` | 55,339,648 | 273 | 570,294 / 570,294 | 1,790 | 7,268 | `fc3d31bc34a9fe0ad9515ead95e0543d095276a84076f2faf99bd4d611da9eda` |

Both files passed the independent full validator. Two separate exports of the
64K capture were byte-identical. The x2 scene's 716 stored ACTIVE templates
still represent 24,687,816 flat ACTIVE polygons; that flat count is deliberately
not materialized in `KACTSCN1`.

## Deliberate residuals

This format is a lowering checkpoint, not yet the GPU scene ABI. It does not
encode properties, holes, non-Manhattan contours, complex transforms,
breakout/waiver state, marker ownership, rule parameters, or revision tokens.
The CUDA loader should either use the same fixed tables directly or copy them
once into a versioned resident scene. Exact ACTIVE.3 candidate generation,
Euclidean enclosure predicates, device-side culling, and compact result
materialization remain the next island stages.
