# POLY.3/POLY.4 terminal-empty correctness milestone

This work began with the correctness question that had to be answered before a
host ABI or live-deck transaction could be added:

> Can a bounded rectangle-union CUDA classifier safely prove that
> `enclosing(..., projection).polygons.without_area(0)` is empty for the
> qualified 110 and 140 DBU profiles?

The standalone predicate, production-volume census, initial merged-GATE live
transaction, and direct raw POLY+ACTIVE owner transaction are now all present.
Both CUDA protocols remain additive and fail closed.  The default deck remains
byte-identical; the separately enabled raw owner path now has an exact
production performance result.

## Exact terminal semantics

The deck does not require the raw enclosure edge-pair collection to be empty.
Coincident edge pairs normalize to zero-area polygons and are removed by the
terminal chain. The certificate therefore accepts a projected gate side only
when the primary rectangle union proves one of two sufficient conditions:

1. The open sub-threshold band contains no primary area. A facing pair is
   coincident and its normalized polygon has zero area.
2. The complete band is covered. Every facing boundary is at least the
   qualified distance away.

Mixed coincidence/full coverage, positive partial coverage, unrelated nearby
primary geometry, incomplete candidate windows, unsupported geometry, and
capacity overflow all decline to pristine CPU fallback. The fused decision is
complete only when both the 110 DBU POLY.3 and 140 DBU POLY.4 profiles certify
terminal empty.

The allocation-free host/device implementation is in
`poly34_terminal_empty_certificate.cuh`. The standalone CUDA island compares
that implementation with an independent Ruby oracle and with terminal outcomes
computed by KLayout itself through:

```ruby
primary.enclosing_check(
  gate, distance, false, RBA::Region::Projection, nil, nil, nil
).polygons.with_area(0, true)
```

`with_area(0, true)` is the direct Region API form of `.without_area(0)`.

## Deterministic and randomized gate

`run_poly34_terminal_empty_gate.sh` covers:

- exact coincidence;
- exact 110/140 DBU boundaries;
- 1 DBU and distance-minus-1 partial margins;
- different POLY.3/POLY.4 outcomes over the same gate;
- mixed coincidence/full segments, deliberately declined;
- disconnected geometry inside and exactly at the projection boundary;
- tiled rectangle-union coverage;
- a 1 DBU band gap;
- incomplete gate coverage;
- unsupported primary geometry flags; and
- bounded candidate overflow.

It then generates 20,000 seeded randomized dual-primary rectangle-union cases
by default. A real KLayout process supplies both terminal outcomes. Acceptance
requires:

- exact host/CUDA certificate agreement;
- exact agreement with the independent conservative oracle; and
- zero CUDA terminal-empty decisions when KLayout's filtered terminal region
  is nonempty.

Reproduce with:

```sh
bash benchmarks/cuda_spatial_replay/run_poly34_terminal_empty_gate.sh \
  --klayout /path/to/klayout \
  --random-count 20000
```

The qualified 20,000-random-case run on 2026-07-24 reported:

```text
cases=20015
raw_edge_pairs=133116
zero_area_only_profiles=17887
actual_nonempty_profiles=13946
gpu_mismatches=0
false_clean=0
verdict=GO
```

Its charged standalone gate wall was 9.114 seconds, including fixture
generation through KLayout, CUDA compilation, upload, classification, and
validation. That is test-harness runtime only, not a production performance
measurement or saving claim.

## Production geometry census

The read-only census is:

```sh
/path/to/klayout -b \
  -r benchmarks/cuda_spatial_replay/poly34_geometry_census.rb \
  -rd input=/path/to/design.gds \
  -rd topcell=TOP \
  -rd poly_layer=9 -rd poly_datatype=0 \
  -rd active_layer=1 -rd active_datatype=0
```

On the pinned FreePDK45 x2 production input
`sram_1rw0r0w_64_4096_freepdk45__x2__independent_sref.gds`, the exact merged
`gate = poly & active` result was:

```text
gate_merged=3401254
gate_boxes=3401254
gate_manhattan_nonboxes=0
gate_non_manhattan=0
gate_with_holes=0
```

Every gate was exactly 100 DBU wide. The source hierarchy contained 6,164 POLY
box templates plus 34 Manhattan polygon templates, and 700 ACTIVE box templates
plus 16 Manhattan polygon templates. The remaining 80 POLY and 8 ACTIVE layer
records were texts, not geometry. No non-Manhattan polygon, path, or polygon
with holes was present.

## Decision boundary

The standalone rectangle-union terminal classifier is a **correctness GO** once
the full seeded differential gate passes: it is conservative and has no
observed false-clean path.

The subsequent production-volume milestone is also a **GO for live
integration**. `poly34_production_dry_run.cc` reads the pinned layout through
KLayout's database library, constructs the exact flat merged POLY, ACTIVE, and
GATE regions, and decomposes each rectilinear primary union into canonical
`TD_simple` boxes. A two-pass KLayout box scan then gathers every primary box
whose bounding box is less than the rule distance from each gate. Geometry
outside that square window cannot intersect the projected side bands examined
by the certificate. Boundary-only boxes at exactly the distance contribute no
positive area and are intentionally excluded.

On the FreePDK45 x2 input, the complete candidate census was:

```text
profile  gates      raw candidates  maximum  p50  p99  capacity >64
POLY.3   3,401,254  4,462,594       2        1    2    0
POLY.4   3,401,254  3,403,326       4        1    1    0
```

Both conservative profiles certified all 3,401,254 gates, so their atomic
conjunction also had 100% coverage with zero fallback or unsupported
occurrences. The unchanged KLayout projection checks independently produced
6,802,508 raw edge pairs for each profile. Every one normalized to a zero-area
polygon; the deck's `without_area(0)` therefore retained zero markers, exactly
matching the atomic certificate.

A fully materialized 64-bit-box/CSR representation requires at least
451,288,546 bytes (430.4 MiB) before allocator and spatial-index workspace:

```text
gate boxes             108,840,128
primary boxes          246,360,896
candidate offsets       54,420,080
candidate identities    31,463,680
profile/atomic results  10,203,762
```

The qualified read-only run took 234.90 seconds internally and 242.06 seconds
charged, with 8,304,532 KiB (7.92 GiB) peak RSS. This is deliberately a
one-time exhaustive census, not a live implementation estimate: 156.56 seconds
were spent flattening and merging production regions, the two exact KLayout
oracle checks took 33.62 and 28.41 seconds, the complete candidate scans took
4.84 and 3.98 seconds, and both certificate passes together took about 0.044
seconds.
Evidence is in
`cuda-runs/poly34-production-dry-run.ktYpv1`.

Reproduce the deterministic boundary gate with:

```sh
bash benchmarks/cuda_spatial_replay/run_poly34_production_dry_run_gate.sh \
  --klayout /path/to/klayout
```

Or run the production census directly:

```sh
bash benchmarks/cuda_spatial_replay/run_poly34_production_dry_run.sh \
  --input /path/to/layout.gds --top TOP
```

## Initial merged-GATE live transaction

The separately enabled live path avoids the 159-second flat census
construction.  Its host bridge first checks both the POLY.3/.4 opt-in and the
independent backend symbol, then explicitly merges the same hierarchical POLY,
ACTIVE, and derived GATE operands used by the historical rules.  It packs only
exact per-cell box unions and regular orthogonal instance contexts.  Store,
layout, top-cell, layer, count, capacity, and complete-scene identities are
bound into a SHA-256 digest.

The `nvcc`-compiled backend revalidates the request and digest, expands the
compact hierarchy on device, constructs bounded POLY and ACTIVE spatial
windows, and applies both exact terminal predicates atomically.  It returns
clean only when every GATE occurrence certifies both profiles.  A genuine hit,
uncertain predicate, malformed request, disabled or missing backend, capacity
limit, or CUDA error executes both pristine CPU expressions.

At this historical milestone, the focused live gate had seven CPU oracles and
seven accelerated cases: flat clean, clean Manhattan non-box primary
decomposition, 2,048-occurrence hierarchical clean, a six-occurrence
asymmetric rotated-array hit, independent genuine POLY.3 and POLY.4 hits, and
a mixed two-rule hit.  It also proved that an unqualified source layer, a
configured backend with the feature disabled, and a missing backend performed
no scene lowering, and that a forced device-capacity failure fell back
atomically.  All 18 complete, generator-stripped `.lyrdb` reports were
canonical-identical to their CPU oracle, including ordered categories and
cell/report structure:

```text
POLY34_LIVE_GATE ok gate=cpu-oracle cases=7 categories=2
POLY34_LIVE_GATE ok gate=cuda clean=3 hits=4 reports=cpu-identical atomic=1 transformed-hit=1 manhattan-primary=1
POLY34_LIVE_GATE ok gate=wrong-layer report=cpu-identical no-lowering=1
POLY34_LIVE_GATE ok gate=backend-off report=cpu-identical
POLY34_LIVE_GATE ok gate=missing-backend report=cpu-identical
POLY34_LIVE_GATE ok gate=capacity report=cpu-identical
POLY34_LIVE_GATE PASS oracles=7 cuda=7 wrong-layer=1 backend-off=1 missing=1 capacity=1 reports=18
```

Reproduce against built host and backend artifacts with:

```sh
bash benchmarks/cuda_spatial_replay/run_poly34_live_gate.sh \
  --klayout /path/to/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so
```

The existing aggregate CUDA ABI/oracle smoke and VIA1-stack backend smoke also
pass with the new independent symbol linked into the shared backend.  This is
an integrity milestone, not a production/full-design performance gate, so no
whole-run saving is booked here.

## Direct raw POLY+ACTIVE owner transaction

The next transaction moves the expensive Boolean boundary itself.  In the
sole-consumer `m1_enclosure` owner, the generated deck calls
`cuda_poly34_raw_clean?` before constructing `gate = poly & active`.  The host
walks the hierarchy once, performs exact Manhattan `TD_simple` decomposition
of the raw POLY and ACTIVE layers, and serializes their compact cell templates
and complete occurrence contexts.  It sends no GATE layer, GATE context, or
GATE box span.

The device expands both raw layers into the common top-cell coordinate system
and performs a global spatial join.  The construction is exact because

```text
(union_i P_i) intersect (union_j A_j)
  = union_i,j (P_i intersect A_j)
```

Every positive-area `P_i intersect A_j` rectangle is emitted.  A deterministic
grid owner prevents one pair from being emitted repeatedly when it spans
multiple grid cells.  The resulting rectangles are an exact cover of GATE;
they need not be merged, disjoint, or equal in count to the historical merged
GATE polygons.  A complete join with no positive intersection also proves
that GATE is empty.

An exact cover introduces artificial tile sides that are not physical GATE
boundaries.  Before applying either terminal profile, the backend tests each
tile side against the derived GATE cover.  It suppresses the side only when
one other exact GATE tile covers the complete positive-width 1-DBU strip
immediately across that side.  On the integer-DBU geometry lattice, that is a
sufficient proof that the whole side lies inside GATE and therefore cannot
participate in KLayout's enclosing result.  Split or partial coverage is not
accepted.  A directed fixture with a 100-DBU tile beside a 1-DBU fragment
would falsely look partially covered if the seam were treated as an external
boundary; proving the full 1-DBU strip internal lets both rules certify it
exactly.  Unproved seams, incomplete windows, unsupported geometry, arithmetic
overflow, or any capacity error still decline atomically.

The raw request is an additive format-2 opcode.  The original merged-GATE
opcode, request format, and non-owner deck hook are unchanged.  A raw owner
decline never retries that legacy CUDA path: it lazily constructs the
historical `gate = poly & active` region and executes both pristine CPU
expressions.  Ruby exceptions, missing or symbol-incomplete backends, capacity
declines, and device errors follow the same fail-closed rule.  Non-owner modes
can continue to use the legacy transaction.

The current focused live gate has eight CPU oracles and eight raw CUDA cases.
The added clean case stores POLY and ACTIVE in different sibling leaves and
makes them intersect only after both contexts are transformed into the common
top coordinate system.  This proves the device join is global rather than
unsafely context-local.  All 20 complete canonical reports match:

```text
POLY34_LIVE_GATE ok gate=cpu-oracle cases=8 categories=2
POLY34_LIVE_GATE ok gate=cuda clean=4 hits=4 reports=cpu-identical atomic=1 transformed-hit=1 manhattan-primary=1 cross-context=1
POLY34_LIVE_GATE ok gate=wrong-layer report=cpu-identical no-lowering=1
POLY34_LIVE_GATE ok gate=backend-off report=cpu-identical
POLY34_LIVE_GATE ok gate=missing-backend report=cpu-identical
POLY34_LIVE_GATE ok gate=capacity report=cpu-identical
POLY34_LIVE_GATE PASS oracles=8 cuda=8 wrong-layer=1 backend-off=1 missing=1 capacity=1 reports=20
```

On the pinned FreePDK45 x2 production owner, the raw compact scene contained:

```text
contexts             849,265
cells                    273
stored source boxes     7,004
expanded raw POLY  16,207,964
expanded raw ACTIVE 24,689,320
derived GATE tiles   3,420,536
```

The **3,420,536** value counts device-derived cover tiles, not merged GATE
polygons.  The host raw lowering took **316.150 ms** and the complete backend
call took **652.485 ms**, for **968.635 ms** combined.  Telemetry confirms
that this lane performed no host GATE construction and no legacy merged-input
lowering.

The exact immediate-parent production A/B changed **78.74 -> 34.04 seconds**,
removing **44.70 wall-seconds / 56.769% of owner time**.  Peak RSS changed
**1,850,420 -> 704,812 KiB**, or **61.911% less**.  Both complete reports have
canonical SHA-256
`d056b808e6f2134e60286e35247a92e3a2f6d2eaa26b463fd572aa7e0652146d`.
This comparison is against the immediately preceding implementation and is
not additive with the older merged-GATE production result.

Reproduce the focused and production gates with:

```sh
bash benchmarks/cuda_spatial_replay/run_poly34_live_gate.sh \
  --klayout /path/to/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so

bash benchmarks/cuda_spatial_replay/run_poly34_production_owner_gate.sh \
  --klayout /path/to/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so \
  --device-smoke /path/to/cuda_spatial_backend_smoke \
  --source-deck /path/to/source.lydrc \
  --default-deck-reference /path/to/default.lydrc \
  --input /path/to/layout.gds \
  --top-cell TOP
```

A second serialized production A/B measured **77.16 -> 33.93 seconds**,
removing **43.23 seconds / 56.026% of owner time**.  The complete six-lane
gate also proved exact reports for injected exception, forced capacity,
missing-library, and ABI-compatible symbol-incomplete-library fallback.  Its
durable evidence is in
`.scratchpad/cuda-runs/klayout-poly34-raw-owner.M4P5M5`.
