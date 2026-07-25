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
4. Compact only zero-to-positive and positive-to-zero transitions into packed
   uint64 records, then pair them into disjoint covered strip intervals.
5. Reuse the transition allocation for packed `(side,y,slab)` horizontal keys.
   Sort those keys and reduce contiguous slabs on each directed line into
   maximal horizontal runs before materializing 32-byte output segments.
6. Retain per-slab offsets into the already-sorted disjoint intervals.  One
   count/prefix/emit pass directly XOR-merges the two interval lists adjacent
   to each x boundary.  This emits exact vertical boundaries in linear work
   without a second four-events-per-interval sort.
7. Sort only the vertical output.  Horizontal and vertical streams are each
   already canonical, so copy them to the host in axis order without a final
   full-output device sort or concatenation buffer.

All device allocation and teardown, H2D/D2H transfers, sorting, scanning,
compaction, and output hashing are charged in `total_ms`.  The first run also
charges CUDA context initialization.  Later process-local runs model a warm
server retaining the CUDA runtime, but this milestone intentionally does not
retain scene buffers between calls.

Any rectangle, event, membership, segment, arithmetic or internal coverage
invariant failure returns a fallback with no partial result.

`sampled_live_allocation_delta_mib` is the largest live-allocation delta seen
at explicit phase boundaries.  It is not an allocator high-water mark:
temporary CUB/Thrust sort scratch can be allocated and released between
samples.

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
| CPU oracle | 357.804 ms |
| GPU first/cold call | 179.958 ms |
| GPU warm median, four calls | 14.532 ms |

The warm replay used **95.94% less time** than this CPU oracle
(**+2362.15% throughput**).  This synthetic case demonstrates the mechanics;
it is not a claim of a 95% KLayout end-to-end reduction.

## Exact production M2 result

The qualified raw FreePDK45 M2 capture contains:

- 45,960 stored M2 polygons, including six simple six-edge L shapes;
- 568,632 M2 hierarchy contexts;
- 22,945,976 expanded polygon occurrences;
- 22,946,444 exact rectangle records after deterministic decomposition;
- 46,384 unique world x coordinates;
- 92,386,704 rectangle/slab memberships, maximum span 43;
- 184,773,408 y events before equal-key reduction.

The capture is
`/tmp/m2-via1-x2.39i7kG/m2-via1-x2.kact`; the existing merged contour oracle is
`/tmp/m2-width-space-census-exact.km1ws`.

The global exact gate now passes.  Six consecutive GPU calls independently
matched all 4,385,384 canonical directed boundary segments from the CPU-merged
oracle, not merely their count or hash.  The pinned result is:

```text
boundary SHA-256  94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d
boundary FNV-1a   7541395996791771514
horizontal        2192692
vertical          2192692
```

The six-call charged run on the RTX 3080 measured:

| phase | wall time |
|---|---:|
| standalone KACT validation/load | 564.005 ms |
| 32-thread host hierarchy expansion | 562.740 ms |
| first/cold complete GPU union | 820.188 ms |
| warm complete GPU union, median of five | 580.634 ms |
| conservative host-roundtrip pipeline | 1707.380 ms |
| independent oracle load/validation | 5978.154 ms |

The GPU timing includes allocation, hierarchy-expanded rectangle H2D, all
sort/scan/compaction work, canonical-boundary D2H, and teardown.  The
1.707-second pipeline adds input load and host expansion once; the
5.978-second oracle validation is qualification-only and is not included.
Explicit phase-boundary samples saw a 4,234 MiB live-allocation delta, and the
complete run fit on the 10-GiB card.

This is a standalone exact replacement candidate, not yet a KLayout
end-to-end result.  Against the separately measured 46.331706-second native
merge stage, its conservative 1.707380-second warm host-roundtrip path models
**44.624326 real seconds removed**, or **96.315% less stage time**.  That is a
like-for-like stage opportunity, not a measured whole-run saving.  The wider
48.617 CPU-second telemetry residual also includes construction and
integration work and remains an invalid denominator until the live seam is
wired.

The next live step is to expand directly from compact hierarchy records into a
resident device buffer, reject unsupported kissing-vertex topology, and feed
the existing M2 width/spacing predicate without materializing merged KLayout
polygons on the CPU.  Production can either retain exact component identity or
conservatively test the superset of all properly oriented boundary pairs and
fall back on any hit; a zero result still proves clean.  The qualified oracle
has zero kissing, crossing, overlap, duplicate, or nonmaximal-boundary errors.

## Build

```sh
benchmarks/cuda_spatial_replay/run_manhattan_union_replay.sh \
  /home/pullin/personal/klayout/.scratchpad/build-manhattan-union
```

The optional environment variables
`KLAYOUT_CUDA_MANHATTAN_UNION_GRID` and
`KLAYOUT_CUDA_MANHATTAN_UNION_REPEAT` select the synthetic grid and number of
process-local calls.  Setting both
`KLAYOUT_CUDA_MANHATTAN_UNION_PRODUCTION_KACT` and
`KLAYOUT_CUDA_MANHATTAN_UNION_PRODUCTION_ORACLE` adds the pinned production
comparison to the same build-and-test invocation; setting only one fails
before the build.

The production gate is also available directly from the CMake-built binary:

```sh
manhattan_union_replay \
  --production-m2-kact /path/to/m2-via1-x2.kact \
  --production-m2-oracle /path/to/m2-width-space-census-exact.km1ws \
  --repeat 6
```

To publish the actual final GPU result without loading the CPU oracle in the
producer numerator, use:

```sh
manhattan_union_replay \
  --production-m2-kact /path/to/m2-via1-x2.kact \
  --production-m2-candidate-out /path/to/gpu-boundary.km2bnd \
  --repeat 4
```

The writer accepts only the pinned production segment census/FNV/SHA and emits
`KM2BND02`, which binds the output to both the raw KACT producer identity and
the independent merged-oracle qualification identity.  The reported
`published_candidate_pipeline_ms` charges raw-scene load, host expansion, the
actual GPU call producing the file, D2H/teardown, portable vector conversion,
hashing, and serialization.  See `M2_FLAT_REGION_BRIDGE.md` for the complete
producer-to-stock-morphology gate.
