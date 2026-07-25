# Exact CUDA Manhattan-union replay

This standalone replay tests the expensive operation immediately before the
existing M1/M2 CUDA predicates: unioning a large set of expanded orthogonal
geometry.  It is an integrity and feasibility milestone, not yet a live
KLayout bypass.

## Exact contract

Input is a pointer-free array of signed-int64 half-open rectangles:
`[left,right) × [bottom,top)`.  Degenerate or inverted input fails closed.
Output is a deterministic, sorted set of maximal directed boundary segments.
Each segment records its axis and outward side, so collinear fragments with
opposite material sides remain distinct at point-touching corners.

There is no floating point, tolerance, dense global raster, or image
approximation.  Tensor cores do not apply: the useful primitives are integer
radix/comparison sort, segmented scan, reduction and compaction.

The current accelerated event key binds a uint32 slab ID and a checked uint32
`y-y_base` offset into one uint64.  Signed-int64 absolute coordinates remain
exact, but a scene whose total y span exceeds `UINT32_MAX` fails closed.  This
bound covers the qualified M2 scene by more than three orders of magnitude.

`RectI64` is deliberately the normalized output of hierarchy expansion.  The
existing POLY34 and VIA1-stack CUDA paths already apply KLayout's eight
orthogonal transforms to boxes in device code.  A resident production path can
write this 48-byte record directly to device memory and enter the union without
a host round trip.

## GPU pipeline

1. Copy expanded rectangles and sort/unique all x endpoints.
2. Map each rectangle to every exact x slab it covers.  A configured maximum
   span and total membership capacity make worst-case growth explicit.
3. Emit signed y events per membership into packed uint64 `(slab,y-y_base)`
   keys, radix-sort/reduce identical events, and segmented-scan coverage.
4. Compact only zero-to-positive and positive-to-zero transitions into
   disjoint covered strip intervals.
5. Retain per-slab offsets into those already-sorted disjoint intervals.  One
   count/prefix/emit pass directly XOR-merges the two interval lists adjacent
   to each x boundary.  This emits exact vertical boundaries in linear work
   without a second four-events-per-interval sort.
6. Sort horizontal and vertical fragments and merge collinear intervals.
   Segment grouping uses a per-line segmented prefix maximum, which remains
   exact for nested fragments such as `[0,100], [10,20], [30,40], [100,120]`.
7. Copy only canonical boundary segments to the host.

All device allocation and teardown, H2D/D2H transfers, sorting, scanning,
compaction, and output hashing are charged in `total_ms`.  The first run also
charges CUDA context initialization.  Later process-local runs model a warm
server retaining the CUDA runtime, but this milestone intentionally does not
retain scene buffers between calls.

Any rectangle, event, membership, segment, arithmetic or internal coverage
invariant failure returns a fallback with no partial result.

## Integrity gates

`run_manhattan_union_replay.sh` builds and runs:

- 15 directed fixtures: overlap, duplicate, nesting, edge/corner touch,
  T-junction, plus, hole, covered seams, a containment bridge, negative
  coordinates and large signed-int64 coordinates;
- 64 fixed-seed randomized comparisons against an independent CPU sweep;
- a direct nested-fragment canonicalization regression;
- four invalid/degenerate fail-closed cases;
- a forced per-rectangle membership-capacity fallback;
- a forced packed-y-range fallback.

The current gate is 86/86 CPU-identical/fail-closed.

The output boundary set is exact for point-touching inputs, but a live KLayout
integration also needs its polygon/component identity and maximum-coherence
topology convention.  The first production path should therefore reject
degree-four/checkerboard kissing vertices until component labeling and that
convention have their own oracle gate.

## Charged synthetic result

On the project RTX 3080, an exact union of a touching 1024×1024 grid
(1,048,576 rectangle records) produced the same four canonical boundary
segments and digest as the CPU sweep:

| path | charged time |
|---|---:|
| CPU oracle | 377.784 ms |
| GPU first/cold call | 178.846 ms |
| GPU warm median, four calls | 14.562 ms |

The warm replay used **96.15% less time** than this CPU oracle
(**+2494.25% throughput**).  This synthetic case demonstrates the mechanics;
it is not a claim of a 95% KLayout end-to-end reduction.

## Production M2 sizing and next gate

The captured FreePDK45 M2 scene is favorable but larger than the first generic
defaults:

- 45,954 local box templates plus six simple six-edge Manhattan templates;
- 568,456 hierarchy contexts;
- 22,945,976 expanded polygon occurrences;
- 22,946,444 exact rectangle records after deterministic decomposition;
- 46,384 unique world x coordinates;
- 92,386,704 rectangle/slab memberships, maximum span 43;
- 184,773,408 y events before equal-key reduction.

The capture is
`/tmp/m2-via1-x2.39i7kG/m2-via1-x2.kact`; the existing merged contour oracle is
`/tmp/m2-width-space-census-exact.km1ws`.

The checked packed event key and linear adjacent-slab XOR are now implemented,
and phase-dead buffers are explicitly released before the next full-volume
allocation.  The immediate production gate is therefore the global scene:
compare its canonical directed boundary multiset against the 4,385,384-edge
merged KM1WS oracle.  If the measured sort scratch still exceeds 10 GB, the
exact fallback design is bounded contiguous x-slab tiles with a one-slab halo.

After that boundary proof, a live path must add exact component IDs, fail
closed on kissing vertices, and feed the existing M2 width/spacing predicate
without materializing KLayout polygons on the CPU.

## Build

```sh
benchmarks/cuda_spatial_replay/run_manhattan_union_replay.sh \
  /home/pullin/personal/klayout/.scratchpad/build-manhattan-union
```

The optional environment variables
`KLAYOUT_CUDA_MANHATTAN_UNION_GRID` and
`KLAYOUT_CUDA_MANHATTAN_UNION_REPEAT` select the synthetic grid and number of
process-local calls.
