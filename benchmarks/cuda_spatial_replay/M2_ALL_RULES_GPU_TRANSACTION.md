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

## A clean-only certificate does not need component IDs

The merged boundary's outward side is enough for a conservative empty
certificate.  Let `C` be the boundary obtained by coalescing every touching
collinear fragment with the same `(axis, fixed_coordinate, outward_side)`,
without regard to contour or component.  Every KLayout contour edge is
contained in one member of `C`, with the same orientation and material side.

For `METAL2.1/.2`, send every nearby unordered pair in `C` through both the
width-facing and space-facing exact relation predicates.  Do not apply the
width predicate's same-polygon eligibility filter.  This is conservative:

- every stock width pair is included because all same-polygon pairs are in the
  all-pairs set;
- every stock space pair is included already;
- extending an axial edge cannot increase its projection gap or Euclidean
  near-part distance; and
- a real violating pair is antiparallel, so its two members cannot have been
  collapsed into one same-side member of `C`.

Thus zero raw hits from both relations proves both stock result sets empty.
Extra cross-component width-facing hits merely select CPU fallback.  KLayout's
shielding pass can only remove raw hits, so it is not needed on a zero-hit
path.

The same argument removes component IDs from `METAL2.5`: form the exact gt90
material boundary, coalesce by axis/side, retain conservative segments of
length at least 600 DBU, and run the space predicate on every nearby pair.
Every stock selected edge is contained in a selected conservative segment.
Any stock `<180`-DBU violation therefore remains a conservative violation.

This is not only a paper shortcut.  A focused production diagnostic changed
the existing M2 CUDA query from same-polygon width eligibility to both
relations on every unique pair.  A Release `sm_86` run produced:

```text
unique pairs   34,011,017
width pairs    34,011,017
space pairs    34,011,017
width hits              0
space hits              0
uncertain                0
GPU total       397.051540 ms
```

The run retained the exact empty report digest
`55da1c410253ef51f00af69758bd899a67ce70fe6161d1711ef77271f5ca3a0a`.
Qualification evidence is
`.scratchpad/m2-morph-certificate/both-relations-gpu.dRTeEO`; its run-log
SHA-256 is
`a46a13d99be9af4070f1bc115977d5eff8f225b0255c58858bb72a973f072c39`
and the diagnostic backend SHA-256 is
`78be6ad9d5988899baf6ea44eccd3ebb30d17957197b32e2088de3dee2736c3e`.
The timing is a single semantic gate, not a performance comparison.

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
context as canonical x bands containing sorted disjoint y intervals.  The
fastest exact first implementation is a direct separable sliding-window
transform, not another global rectangle sort.

For an erosion radius `r`:

1. In every input x band, replace `[bottom, top)` with
   `[bottom+r, top-r)` and discard it unless `top-bottom > 2r`.  This is exact
   y erosion.
2. Sort/unique every input x boundary shifted by `-r` and `+r`.  Between two
   consecutive shifted boundaries, the set of input slabs intersected by the
   horizontal `2r` window is constant.
3. One CTA per output x band intersects the sorted y-interval lists of all
   slabs in that window.  An exterior slab contributes the empty list.  Use a
   count/prefix/emit pair and discard zero-length intersections.

For dilation radius `r`, first expand and merge y intervals locally, then use
the same shifted x boundaries and take the union, rather than intersection, of
the active interval lists.  Touching intervals are merged.  These are exactly
the separable identities
`erode_square = erode_x(erode_y(U))` and
`dilate_square = dilate_x(dilate_y(U))`.

The qualified raw union has 46,383 x slabs and 3,691,466 strip intervals.
Each transform has at most 92,767 nonzero candidate output x bands.  At the
89/90-DBU radii, the active window is local, so this route replaces a global
sort with tens of millions of expected interval visits and two compact strip
buffers.  Runtime must still cap the active slabs per output band, total
interval visits, output intervals, and every allocation product; a cap miss is
fallback.  With 16-byte intervals, two production-sized interval buffers plus
band offsets are expected to stay in the low hundreds of MiB, rather than the
multi-GiB scratch of the original union.

Complement dilation remains a valuable independent oracle and simpler
fallback prototype:

```text
erode_r(U) = D \ dilate_r(D \ U)
```

For the production strips, `D\U` has at most
`3,691,466 + 46,383 + 2 = 3,737,851` rectangles.  Expanding those rectangles
and reusing the union engine is exact with a checked `r`-padded sentinel
domain, but it requires two additional global membership/event sorts for the
gt90 shrink and grow.  Depending on local x-coordinate density, each can
produce tens of millions of memberships and roughly twice as many packed
events.  It should be used to differential-test the direct transform, not as
the preferred resident production kernel.

After exact `F90 = dilate_90(erode_89(U))`:

1. Extract its exact outward-side boundary from adjacent strip XORs.
2. Coalesce by `(axis, fixed, side)`, retain lengths `>=600`, and run the exact
   conservative space test at strict distance `<180`.
3. Invoke the erosion count pass on `F90` with `r=269`, but do not emit its
   intervals.  If every output-band count is zero, the gt270 shrink is exactly
   empty.  Its `+270` grow is therefore empty, and induction makes every
   cumulative gt500/gt900/gt1500 class empty.
4. If the gt270 count is nonzero, the first implementation falls back for the
   whole atomic block rather than implementing later classes.

The tempting cheaper proof
`erode_269(dilate_90(raw_M2)) == empty` is valid as an upper-bound test but
does not pass this workload: an exact KLayout audit retained 152 flat / 148
hierarchical shapes.  The exact first opening is therefore the smallest robust
production target.

No contour stitching or component labeling is required on this zero-hit path.
Required resident invariants are exact axial coordinates, correct outward
sides, sorted positive-width bands, sorted positive-length non-touching
intervals, checked arithmetic, and conservation/digest counters.  Properties,
unqualified transforms, malformed source geometry, or a failed strip
invariant select fallback.

Useful integrity counters are input/output area, bbox, context census,
interval-work census, boundary-side conservation, and deterministic payload
digest.  All arithmetic and allocation products must be checked before launch.

## Flat stock-morphology bridge

There is a useful interim bridge if the resident F90 kernels take longer to
land.  The qualified 4,385,384 directed union segments can be stitched into
14,222 flat clockwise polygons by sorting directed start vertices, requiring
exactly one successor and predecessor, traversing every cycle once, and
checking area/orientation and full segment conservation.  Constructing

```cpp
db::FlatRegion *flat = new db::FlatRegion(
    shapes, true /* already merged */);
flat->set_merged_semantics(true);
db::Region merged_flat(flat);
```

then lets stock flat sizing skip the original deep merge.  Flattening does not
change geometric clean/dirty semantics, and on a hit or any stitch failure the
transaction discards the flat result and reruns the untouched deep block, so
hierarchical marker attribution is preserved on every nonempty result.

The first exact bridge probe passed the production goldens: 4,385,384
segments stitched into 14,222 contours, the gt90 boundary and retained
long-edge results matched, and the gt270 result was empty.  Its preliminary
stock-flat phase timings were about 4.76 seconds for the gt90 shrink, 2.44
seconds for its grow, and 4.22 seconds for the gt270 shrink, plus the gt90
edge/length work.  In particular, flattening and asserting the exact union as
already merged removes the earlier approximately 26.59-second deep
merge/shrink behavior; that number is not a remaining bridge pole.

These timings remain preliminary until a stable charged gate includes
boundary D2H, cycle stitching, 4.4-million-vertex `db::Polygon`
materialization, every stock-flat phase, allocation/teardown, and golden
validation.  The result nevertheless makes the bridge a promising way to reap
the measured 46-second first merge before resident F90 kernels land.

Semantic gates are strict: no properties, every segment used once, unique
directed successor/predecessor, no open or zero-area cycle, no hole/CCW cycle
in the first version, no degree-four kissing ambiguity, and exact
merged-boundary area/perimeter/digest agreement.  The `FlatRegion(shapes,
true)` already-merged flag is an unchecked assertion inside KLayout, so none
of these checks may be inferred from that flag.  Constructing
`Region(shapes, merged_semantics, is_merged)` is not equivalent here: that
path inserts shapes one by one and loses the already-merged state.  The
resident component-free certificate remains the end state because it avoids
the host materialization and stock suffix, not because the flat bridge retains
the old deep gt270 pole.

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
- test gt90 rectangle widths/heights 177, 178, and 179 DBU: exactly 178
  disappears under `-89`, while 179 leaves one DBU before the `+90` grow;
- test dilation gaps 179, 180, and 181 DBU, including the exact-touch
  max-coherence case;
- test gt270 widths/heights 537, 538, and 539 DBU: exactly 538 disappears
  under `-269`, while 539 is nonempty;
- test edge lengths 599, 600, and 601 DBU (`with_length` is inclusive), and
  edge spaces 179, 180, and 181 DBU (space is strict);
- test an exact Euclidean diagonal at, immediately below, and immediately
  above the 180-DBU radius, in both width-facing and space-facing
  orientations;
- test thin bridges, notches, holes, disappearing components, dilation joins,
  long edges created by collinear joining, mirrored hierarchy, and kissing
  corners;
- verify the component-free superset on separate components whose nearby
  edges are width-facing only, so false hits are shown to fallback rather than
  certify;
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
