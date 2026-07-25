# FreePDK45 M2 all-rule GPU transaction

This note pins the semantics and production oracle for extending the raw M2
Manhattan-union replay through the rest of the FreePDK45 M2 rule owner.  It is
an audit/design note, not a claim that the transaction is implemented.

## The integration boundary must be atomic

The production M2 owner consumes the same `metal2` region in this order:

1. merged M2 width and space (`METAL2.1` and `METAL2.2`);
2. M2 enclosure around VIA2 (`METAL2.4`);
3. five cumulative classify-by-width morphology stages; and
4. length-filtered edge-space rules (`METAL2.5` through `METAL2.9`).

A GPU union used only for `METAL2.1/.2` does not remove the native merge.
`metal2.sized(...)` subsequently calls `merged_deep_layer()` on the original
region.  Unless the generic merged cache is mutated, that call performs the
same native merge again.

An isolated production run, with no preceding `drc_batch` cache warm-up,
measured:

```text
first gt90 shrink (includes native merge)  51.17 s
gt90 grow                                  2.27 s
gt270 shrink to empty                     26.59 s
remaining seven sizes                      0.00 s
```

The safe useful seam is therefore one fail-closed transaction before any
stock M2 operation:

```text
raw M2 hierarchy + raw VIA2
  -> exact resident Manhattan union
  -> METAL2.1/.2/.4/.5/.6/.7/.8/.9 empty decisions
  -> one scalar certified-empty/fallback result
```

On success, the deck emits empty results for all eight rules.  On any hit,
unsupported input, uncertainty, capacity failure, or integrity failure, it
runs the untouched CPU block.  `METAL2.3` remains owned by the existing VIA1
stack transaction; the all-M2 gate must require that ownership arrangement or
also cover `METAL2.3`.

## Exact classify-by-width semantics

The deck function mutates `layer` inside `collect`:

```ruby
dimensions.collect do |d|
  layer = layer.sized(-0.5 * (d - 1.dbu)).
                sized( 0.5 * (d - 1.dbu))
end
```

The five outputs are cumulative, not five independent transforms of raw M2.
The DRC engine converts a floating sizing value with
`floor(0.5 + value / dbu)`.  At the production DBU of 0.5 nm this rounding is
asymmetric:

| class | dimension (DBU) | shrink (DBU) | grow (DBU) |
| --- | ---: | ---: | ---: |
| gt90 | 180 | -89 | +90 |
| gt270 | 540 | -269 | +270 |
| gt500 | 1000 | -499 | +500 |
| gt900 | 1800 | -899 | +900 |
| gt1500 | 3000 | -1499 | +1500 |

The default sizing mode is `2` (`square_limit`).  For qualified Manhattan
polygons this is exact square, or L-infinity, morphology.  In set notation one
stage is:

```text
F_D(U) = merge(dilate_square(D/2,
                  erode_square(D/2 - 1, merge(U))))
```

The grow is deliberately one DBU larger than the shrink.  Positive sizing is
not marked merged; the next sizing call, or `edges`, merges it using the
region's default maximum-coherence setting (`min_coherence == false`).

The first implementation should fail closed on properties, breakouts,
non-orthogonal or non-unit transforms, holes, non-Manhattan contours, and
degree-four kissing vertices.  Supporting any of those later requires matching
KLayout's contour/coherence semantics rather than merely matching area.

## Edge and M2.4 predicates

`edges.with_length(min, nil)` retains edges with length **greater than or equal
to** `min`.  `Edge::length()` is rounded Euclidean length; on the qualified
Manhattan output it is exactly the absolute coordinate delta.  Collinear
boundary fragments must therefore be coalesced before filtering.

`edges.space(d, euclidian)` checks every selected outward-facing edge pair,
including pairs from the same contour.  A positive-length violation is at
strict distance **less than** `d`; equality passes.  The default angle cutoff
is 90 degrees and the default zero-distance mode includes touching.  The
existing exact CUDA edge predicate can be reused after the morphology boundary
edges are length-filtered.

`METAL2.4` is:

```ruby
bad = metal2.enclosing(via2, 35.nm, projection).second_edges
corners = bad.width(angle_limit(100.0), 1.dbu)
via2.interacting(corners.polygons(1.dbu))
```

The first operation returns VIA2 edge fragments with less than 35 nm projected
M2 enclosure.  The 100-degree angle limit makes the 1-DBU width check detect
adjacent bad sides at their shared 90-degree corner.  This implements “two
opposite sides”: a via is valid when either both horizontal sides or both
vertical sides have adequate enclosure.  VIA2 outside M2 is handled separately
by `VIA2.3`.

For the clean-only transaction, a simpler sufficient certificate is exact:
for each qualified rectangular VIA2, require the via itself to be covered and
require both opposite 35-nm projection strips to be covered in X or in Y.
Failure to prove either pair selects CPU fallback.  This may decline inputs
that the stock fragment-level rule would accept, but cannot falsely certify an
error.

## Resident exact square morphology

The union output need not be materialized as CPU contours.  Keep each nonempty
context as canonical disjoint y bands containing sorted x intervals:

1. Square dilation is separable.  Expand each x interval by `r`, merge interval
   overlaps, expand its y band by `r`, then resolve the y events with the same
   sort/scan union primitive.
2. Square erosion uses
   `erode_r(U) = complement(dilate_r(complement(U)))`.  Build complement
   intervals in a per-context sentinel domain padded by at least `r`, mark the
   exterior as complement, run the same dilation, and retain the complement
   inside the original domain.
3. Canonicalize after every shrink and grow.  Extract covered/uncovered
   transitions, orient them with interior on the right, join collinear
   degree-two fragments, and assign component IDs.  A seam multiplicity,
   open contour, zero/diagonal edge, ambiguous degree-four vertex, or overflow
   is fallback.
4. Run the gt90 300-nm length filter and 90-nm exact space predicate.  Run the
   gt270 shrink next.  On the qualified production scene it is empty, so
   `METAL2.6` through `.9` are certified empty without their grow or later
   stages.

The interval form avoids a global X-by-Y cross product and maps to GPU radix
sort, segmented scan, interval merge, and compaction.  An equivalent
rectangle-event implementation is valid if it uses the same complement
sentinels and proves the same canonical boundary invariants.

Useful integrity counters are input/output area, bbox, context census,
interval-event census, boundary closure, component count, and deterministic
payload digest.  All arithmetic and allocation products must be checked before
launch.

## Qualified production oracle

Workload:

```text
sram_1rw0r0w_64_4096_freepdk45__x2__independent_sref.gds
top cell sram_1rw0r0w_64_4096_freepdk45__x2
DBU 0.0005 um
```

Raw hierarchy:

```text
M2 stored / flat polygons       45,960 / 22,945,976
VIA2 stored / flat polygons          8 /     10,128
```

The checked-in `m2_classify_width_oracle.lydrc` produced:

```text
gt90 raw polygons       1,063,596 flat / 1,063,594 hierarchical
gt90 merged edges       4,254,384 flat / 4,254,376 hierarchical
gt90 edges >= 300 nm            8 flat /         4 hierarchical
gt270 raw polygons              0 flat /         0 hierarchical
gt270 edges >= 900 nm           0 flat /         0 hierarchical
```

The four stored long-edge records are two edges in each of the two
`dff_buf_0` variants:

```text
(2.632,1.183;3.403,1.183)
(3.403,1.047;2.632,1.047)
```

They are 771 nm long and 136 nm apart.  Their orientations are width-facing,
not space-facing.  The exact oracle report generated from the checked-in
relative deck path is 4,565 bytes with SHA-256:

```text
9a66be0d7bb80a7b6f8fbfc128c738846684ea87c644fd93a5912a7048b7a3cf
```

The report digest pins the retained boundary records and the empty gt270
category; the report generator path is part of the XML and must be held fixed
when reproducing that byte digest.

The checked-in M2/VIA2 capture deck produced a raw hierarchical GDS and the
existing `active3_packed_scene_export.rb` converted it to `KACTSCN1`:

```text
capture GDS SHA-256
  8c123e3a215d47a08591f53c474e3f225f5b0fc6c0298c47003becfd7192e813
packed file SHA-256
  2176a29ecd0ea1a55551bd7db0e3cc02068a2ed65f958c77cc20406b0bd492dc
packed scene SHA-256
  3a511538eb520292eed7e00f4ad1763360502e4a6751a1bb66fab1e8838678a5
```

The GDS digest identifies that captured artifact but is not a reproducibility
golden because the GDS header contains write timestamps.  A second capture
produced the same packed file and scene digests bit-for-bit.  The packed
digests are the canonical goldens.

Packed census:

```text
bytes       67,847,296
cells              143
instances      568,456
polygons        45,968
edges          183,884
layers         101/0 (M2), 102/0 (VIA2)
```

The production CPU `METAL2.4` chain generated 15,948 flat / 15,876
hierarchical inadequate-enclosure edge fragments, but zero adjacent-corner
errors and therefore zero rule markers.

## Required differential gates

Before live integration:

- compare every boundary edge of gt90 against a CPU capture, not just area;
- reproduce the four stored/eight flat long-edge records above;
- reproduce the empty gt270 erosion and all eight empty M2 rule lanes;
- test dimensions, lengths, and spaces at threshold minus one, equal, and
  threshold plus one DBU;
- test thin bridges, notches, holes, disappearing components, dilation joins,
  long edges created by collinear joining, mirrored hierarchy, and kissing
  corners;
- compare random rectangle unions after every cumulative morphology stage to
  KLayout, with deterministic canonical digests; and
- inject unsupported topology, properties, transforms, overflows, and capacity
  failures and verify untouched CPU fallback.

## Reproduction

Run the classify oracle:

```sh
klayout -b \
  -r benchmarks/cuda_spatial_replay/m2_classify_width_oracle.lydrc \
  -rd input=/path/to/x2.gds \
  -rd topcell=sram_1rw0r0w_64_4096_freepdk45__x2 \
  -rd output=/path/to/m2-classify-oracle.lyrdb
```

Capture and validate raw M2/VIA2:

```sh
klayout -b \
  -r benchmarks/cuda_spatial_replay/m2_via2_projection_capture.lydrc \
  -rd input=/path/to/x2.gds \
  -rd topcell=sram_1rw0r0w_64_4096_freepdk45__x2 \
  -rd scene_output=/path/to/m2-via2.gds \
  -rd output=/path/to/m2-via2.lyrdb

klayout -b \
  -r benchmarks/cuda_spatial_replay/active3_packed_scene_export.rb \
  -rd input=/path/to/m2-via2.gds \
  -rd topcell=KLAYOUT_CUDA_M2_VIA2_SCENE \
  -rd output=/path/to/m2-via2.kact \
  -rd well_layer=101 -rd well_datatype=0 \
  -rd active_layer=102 -rd active_datatype=0

python3 benchmarks/cuda_spatial_replay/validate_active3_packed_scene.py \
  /path/to/m2-via2.kact
```
