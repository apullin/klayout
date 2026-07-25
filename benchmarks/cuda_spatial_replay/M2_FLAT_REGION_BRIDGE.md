# Exact merged-M2 flat Region bridge

Status: production offline feasibility gate passed; live integration is not
implemented.

## Result

The exact CUDA Manhattan-union boundary can be converted into a stock KLayout
`Region` without paying another polygon union:

```text
4,385,384 sorted directed segments
        -> 14,222 checked simple contours
        -> db::FlatRegion(polygons, is_merged=true)
        -> stock cumulative M2 morphology and checks
```

The production result is exact:

```text
input boundary segments                 4,385,384
stitched contours                          14,222
largest contour edges                       2,084
gt90 raw polygons                       1,063,596
gt90 merged boundary edges              4,254,384
gt90 edges >= 300 nm                             8
gt90 90-nm space violations                     0
gt270 eroded/output polygons                     0
stock M2.1 70-nm width violations                0
stock M2.2 70-nm space violations                0
```

These counts match the pinned CPU `classify_by_width` oracle.  The eight long
edges and empty gt270 stage are the useful clean-certificate outputs: once the
second cumulative erosion is empty, the 500/900/1500-nm classes are necessarily
empty too.

This is a practical earlier landing point than porting the full morphology
chain to CUDA.  The GPU still removes the expensive raw 22.95-million-polygon
union; KLayout receives the exact merged boundary and performs the remaining
set operations on only 14,222 flat contours.

## Trust boundary and topology

The bridge consumes the canonical `DirectedSegmentI64` stream.  `side` is the
outward normal.  It is converted to KLayout's material-on-the-right direction
as follows:

| segment | directed edge |
| --- | --- |
| horizontal, side -1 (bottom) | high X to low X |
| horizontal, side +1 (top) | low X to high X |
| vertical, side -1 (left) | low Y to high Y |
| vertical, side +1 (right) | high Y to low Y |

An expected-O(E) hash maps each directed start point to its edge.  A second
linear pass resolves every end point and requires exactly one incoming and one
outgoing edge at every vertex.  Cycle walking then requires all edges to be
visited exactly once.  The bridge rejects:

- open contours;
- duplicate incoming or outgoing endpoints;
- zero-length, diagonal, invalid-axis, or invalid-side segments;
- degree-four kissing vertices;
- contours shorter than four edges;
- coordinate narrowing overflow;
- contour/edge census drift; and
- any KLayout normalization that changes the vertex census.

The production boundary has no holes, kissing vertices, repeated vertices, or
adjacent collinear fragments.  Inputs outside that qualified topology must
fall back to the untouched deep CPU transaction.

The public `Region(Shapes, merged_semantics, is_merged)` constructor currently
inserts each shape after creating its delegate, which clears the requested
merged flag.  The gate therefore uses the equally public delegate path:

```cpp
auto *flat = new db::FlatRegion(polygons, true);
flat->set_merged_semantics(true);
db::Region merged(flat);
```

The gate asserts both `merged_semantics()==true` and `is_merged()==true` before
any stock operation.

## Exact stock certificate

The production DBU is 0.5 nm.  The gate executes the rounded deck operations
with sizing mode 2:

```text
gt90  = size(+90, size(-89, merged_M2))
long  = gt90.edges(length >= 600 DBU)
clean = long.space(180 DBU).empty?
gt270 erosion = size(-269, gt90)
```

The positive F90 size is intentionally not marked merged.  Asking KLayout for
its filtered edges merges it with the default maximum-coherence semantics.
Applying the length filter while generating the edges is identical to
`edges.with_length(600, nil)`, but avoids materializing and rescanning a second
4.25-million-edge collection.

The optional complete host audit also runs stock `width_check(140)` and
`space_check(140)` on the reconstructed merged M2 Region.  Both are empty.
This makes the flat bridge a viable M2.1/M2.2 host certificate if avoiding a
second boundary-backend integration is more valuable than its roughly
0.6-second advantage.

## Timing

Host: the same machine and optimized qmake KLayout DB build
(`83f9c606887cd1cd618c3b5c079af3f1b9aab665`) used by the M2 ABI audit.
Five independent fused-certificate processes were run sequentially.

| phase | median | range |
| --- | ---: | ---: |
| checked endpoint stitch + `Shapes` | 2.188 s | 2.163-2.280 s |
| already-merged `Region` copy | 0.006 s | 0.006-0.007 s |
| shrink F90 | 4.692 s | 4.676-4.713 s |
| grow F90 | 2.432 s | 2.423-2.458 s |
| fused merged-edge/length extraction | 3.355 s | 3.350-3.381 s |
| long-edge spacing | <0.001 s | <0.001 s |
| shrink F270 to empty | 4.206 s | 4.199-4.220 s |
| complete F90/F270 certificate | **14.691 s** | **14.654-14.772 s** |

The full-boundary qualification form first materializes all 4,254,384 merged
F90 edges and then filters them.  It took 18.706 s for morphology in the
qualification run.  The fused form is 21.5% less time while retaining the same
eight long edges and empty spacing result.

The exact stock flat M2.1/M2.2 audit was:

```text
width_check(140 DBU)  1.066 s, 0 pairs
space_check(140 DBU)  2.419 s, 0 pairs
```

The KM1WS load, hashing, and independent topology validation took a median
5.944 s.  That is qualification-only and is not charged to a live path: the
live bridge receives the already validated CUDA output vector.

The proven production union pipeline currently charges 1.707 s for compact
scene loading, host expansion, warm GPU union, and the 4.385-million-segment
D2H transfer.  Composing that measured pipeline with the bridge medians gives:

```text
union pipeline                         1.707 s
endpoint stitch + Region               2.194 s
stock F90/F270 certificate             14.691 s
                                     ----------
composed bridge opportunity            18.592 s

optional stock M2.1/M2.2                3.485 s
                                     ----------
complete host certificate opportunity  22.077 s
```

This is not an integrated end-to-end measurement.  It is a conservative
composed opportunity.  The old 67.612-second denominator contains only the
two separately measured merge payments (46.332 s and 21.280 s), while the new
22.077-second numerator includes the proven union pipeline, contour bridge,
full F90/F270 morphology, and stock M2.1/M2.2.  Even with that deliberately
unequal conservative accounting, the opportunity is 45.535 seconds, or
67.35% less time.  A live gate must be measured again as one transaction, and
concurrent full-launcher wall time will remain governed by the next owner
until the other near-equal poles are reduced.

## Reproduction

The gate needs an optimized qmake KLayout build and the pinned production
KM1WS oracle:

```sh
KLAYOUT_QMAKE_BUILD_DIR=/path/to/klayout-build \
KLAYOUT_M2_FLAT_BRIDGE_ORACLE=/tmp/m2-width-space-census-exact.km1ws \
  benchmarks/cuda_spatial_replay/run_m2_flat_region_bridge.sh \
  /path/to/standalone-build
```

The runner executes five directed topology self-tests, the full production
boundary/M2.1/M2.2 audit, and the fused production certificate.  The oracle
loader pins all three independent digests:

```text
KM1WS file
  980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f
KM1WS scene
  441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480
canonical boundary
  94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d
```

