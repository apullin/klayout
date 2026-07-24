# Exact merged-M1 host scene (`KM1WSCN1`)

Status: narrow capture/replay development format. It is not a public KLayout
format or ABI.

`m1_width_space_host_scene_export.cc` reads an explicitly asserted merged
polygon layer from GDS, constructs the committed
`db::CudaM1WidthSpaceScene`, and writes the scene without routing it through
the generic Ruby `KACTSCN1` exporter.

The producer must be invoked with `--assert-merged`. That assertion is valid
only when external provenance establishes that the selected GDS layer came
from the intended `merged_deep_layer()` derivation. The input-file SHA-256 is
recorded in the transport header, but a digest alone does not prove that
derivation.

## Identity and publication

All integers are little-endian. The file has two independent SHA-256 values:

- `scene_digest` is the canonical digest produced by
  `cuda_m1_width_space_scene_digest()`. The payload is exactly the byte stream
  hashed by that function, so `SHA256(payload) == scene_digest`.
- `transport_digest` covers the complete file with the transport-digest field
  itself treated as zero. It binds the transport header, source provenance,
  canonical payload, and file length.

The exporter first recomputes the host scene digest. It writes and fsyncs a
private sibling file, publishes with an atomic no-clobber hard link, fsyncs
the destination directory, and removes the private name. A failure never
replaces an existing output or publishes a partial scene.

## Transport header

The fixed header is 256 bytes.

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `char[8]` | `KM1WSCN1` |
| 8 | `u32` | file version, 1 |
| 12 | `u32` | header bytes, 256 |
| 16 | `u32` | endian tag, `0x01020304` |
| 20 | `u32` | required flags: canonical payload, merged assertion, source digest |
| 24 | `u64` | total file bytes |
| 32 | `u64` | payload offset, 256 |
| 40 | `u64` | payload bytes |
| 48 | `u64` | context section offset |
| 56 | `u64` | nonempty/metal-context section offset |
| 64 | `u64` | cell section offset |
| 72 | `u64` | polygon section offset |
| 80 | `u64` | edge section offset |
| 88..120 | `u64[5]` | context, metal-context, cell, polygon, edge counts |
| 128 | `u8[32]` | canonical host `scene_digest` |
| 160 | `u8[32]` | full-file `transport_digest`; hash these bytes as zero |
| 192 | `u8[32]` | input capture file SHA-256 |
| 224 | `u32` | input GDS layer number |
| 228 | `u32` | input GDS datatype |
| 232 | `u64[3]` | reserved, zero |

There is no padding between payload sections. Consumers must use checked
count-times-record-size and offset addition, recompute the canonical offsets
with `compute_file_layout()`, and reject any disagreement.

## Canonical payload

The first 128 payload bytes are:

```text
char magic[8] = "KM1WS001"
u32 format_version, dbu_per_micron, root_cell, reserved_zero
i64 width_distance, spacing_distance
u64 context_count, metal_context_count, cell_count, polygon_count, edge_count
u64 flat_polygon_count, flat_edge_count
i64 scene_left, scene_bottom, scene_right, scene_top
```

The records immediately follow in this order.

### Context: 24 bytes

```text
i64 tx, ty
u32 cell_id, transform_code
```

### Nonempty/metal context: 20 bytes

```text
u32 context_id
u64 flat_polygon_offset
u64 flat_edge_offset
```

Only these contexts have local M1 templates. IDs are strictly increasing.
The two offsets form canonical prefix sums and finish at the payload's flat
polygon/edge counts.

### Cell: 32 bytes

```text
u64 source_cell_index
u64 polygon_begin, edge_begin
u32 polygon_count, edge_count
```

Cell ranges canonically partition the polygon and edge tables. Empty cells
are allowed because the context hierarchy is complete.

### Polygon: 48 bytes

```text
u64 edge_begin
i64 left, bottom, right, top
u32 polygon_id, edge_count
```

`polygon_id` is local to the owning cell. Directed edges are contiguous,
closed, clockwise, Manhattan, nondegenerate, and in source contour order.

### Edge: 32 bytes

```text
i64 x1, y1, x2, y2
```

## Required replay checks

A fail-closed consumer must:

1. require an externally allowlisted `scene_digest`;
2. validate the fixed header, required flags, reserved fields, actual file
   length, count arithmetic, and every canonical offset;
3. verify the full transport digest and canonical payload digest;
4. cross-check every duplicated count in the transport and semantic headers;
5. require the qualified DBU, width/spacing distances, root context, and
   scene bounds;
6. validate all context, cell, polygon, edge, and prefix-sum references;
7. restore clockwise edge direction for reflected hierarchy contexts; and
8. treat any malformed topology, overflow, capacity exhaustion, raw hit,
   predicate uncertainty, or device error as a decline of the entire atomic
   METAL1.1/METAL1.2 transaction.

The authoritative constants and little-endian codec are in
`m1_width_space_host_scene_format.h`.
