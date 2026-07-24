# POLY.3/POLY.4 terminal-empty correctness milestone

This milestone is intentionally limited to the correctness question that must
be answered before a host ABI or live-deck transaction is added:

> Can a bounded rectangle-union CUDA classifier safely prove that
> `enclosing(..., projection).polygons.without_area(0)` is empty for the
> qualified 110 and 140 DBU profiles?

It does not make a performance claim and does not alter the production host,
ABI, deck, or report path.

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

The next step may add a separately enabled, digest-bound host/backend
transaction. It must reuse the already-derived hierarchical layers or lower
them into a packed scene; the 159-second flat census construction is not an
acceptable live path. No whole-run saving is booked by this dry-run milestone.
