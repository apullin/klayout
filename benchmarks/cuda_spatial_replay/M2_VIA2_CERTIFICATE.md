# Exact CUDA clean certificate for METAL2.4

`m2_via2_certificate.cu` is a standalone, fail-closed proof for the
FreePDK45 `METAL2.4` rule. It is not wired into the live DRC transaction.

The stock rule requires 35 nm M2 enclosure on two opposite VIA2 sides:

```ruby
bad = metal2.enclosing(via2, 35.nm, projection).second_edges
corners = bad.width(angle_limit(100.0), 1.dbu)
via2.interacting(corners.polygons(1.dbu))
```

At the qualified 0.5 nm DBU, the certificate proves that every rectangular
VIA2 is covered by the union of raw M2 and that either:

```text
[via.left - 70, via.right + 70] x [via.bottom, via.top]
```

or:

```text
[via.left, via.right] x [via.bottom - 70, via.top + 70]
```

is completely covered. Each query is the VIA2 plus both opposite 35 nm
projection strips. Therefore a `CLEAN` result is sufficient for an empty
`METAL2.4` lane. A failed proof is only `FALLBACK`; it is not reported as a
violation.

## Exact coverage algorithm

The implementation reuses the checked `KACTSCN1` loader, hierarchy lowering,
all eight orthogonal transforms, Manhattan polygon decomposition, and
uniform-grid construction from `projection_enclosure_scene_island.cu`.

For one projection query:

1. the uniform grid gathers every intersecting raw-M2 rectangle;
2. rectangle IDs are deduplicated, clipped to the query, and bounded by an
   explicit candidate capacity;
3. the clipped X endpoints partition the query into exact integer slabs; and
4. each slab must have gap-free union coverage from query bottom to query top.

The predicate uses only signed 64-bit integer coordinates. It does not sample,
use floating point, or require one rectangle to cover the query. Touching
rectangles tile exactly; a one-DBU gap fails. Arithmetic, allocation, grid,
candidate-capacity, transform, scene-digest, and census failures all return
`UNCERTAIN`.

The hard per-query candidate bound is 128. It is also runtime-configurable
downward with `--max-query-candidates=N` so the capacity path is testable.

## Differential and integrity gate

The gate compares certified-clean cases with both a bounded independent CPU
rectangle-union predicate and KLayout's stock `METAL2.4` chain. It covers:

- X and Y projection distances at 69, 70, and 71 DBU;
- adjacent partial sides and the choice between opposite X and Y pairs;
- a clean query tiled by three M2 rectangles, with no single-rectangle
  witness;
- a one-DBU tiled gap and an adjacent-corner failure;
- all eight unit orthogonal hierarchy transforms;
- nonrectangular VIA2 fallback;
- candidate-capacity, expected-census, explicit-digest, and corrupted-payload
  fallback; and
- byte-identical repeated scene export.

The checked run passed:

```text
M2_VIA2_CERTIFICATE_GATE passed
  clean=7 fallback=5 fail_closed=5 deterministic=1
```

The one-DBU gap is intentionally a conservative fallback. Stock
`METAL2.4` alone can accept it because the inadequate fragments are opposite;
the separate `VIA2.3` rule owns VIA2-outside-M2. The all-M2 clean transaction
requires the VIA2 itself to be covered, so declining this case is correct.

Run the gate:

```sh
benchmarks/cuda_spatial_replay/run_m2_via2_certificate_gate.sh \
  --klayout /path/to/klayout \
  --build-dir /path/to/build
```

## Qualified production result

The pinned production scene is the downstream-composed two-copy FreePDK45
SRAM workload documented in `M2_ALL_RULES_GPU_TRANSACTION.md`.

```text
packed file SHA-256
  2176a29ecd0ea1a55551bd7db0e3cc02068a2ed65f958c77cc20406b0bd492dc
packed scene SHA-256
  3a511538eb520292eed7e00f4ad1763360502e4a6751a1bb66fab1e8838678a5

contexts                         587,201
M2 contexts                      568,632
logical M2 rectangles         22,947,380
VIA2 contexts / occurrences       10,128 / 10,128
X-certified / Y-certified          2,200 / 7,928
fallbacks                              0
maximum unique candidates/query        7
device flags                           0
```

The production CPU chain independently reported 15,948 flat / 15,876
hierarchical inadequate-enclosure edge fragments, followed by zero
adjacent-corner edge pairs and zero markers. Its byte-exact eight-category
report has SHA-256:

```text
55da1c410253ef51f00af69758bd899a67ce70fe6161d1711ef77271f5ca3a0a
```

The exact production CUDA result was:

```text
M2_VIA2_GPU_CERTIFICATE verdict=CLEAN vias=10128 misses=0

TIMING_MS
  alloc_upload       3.567
  metal_expand       1.446
  grid_count         2.906
  grid_build         2.108
  exact_query        2.606
  d2h                0.025
  cleanup           10.549
  charged GPU plan  23.207
```

The 23.207 ms figure includes upload, rebuilding the standalone M2 index,
the exact query, scalar download, and cleanup. With resident M2 state, the
incremental predicate is the 2.606 ms query rather than another index build.
The cold standalone process took 750.176 ms, dominated by 432.317 ms of
packed-file validation and 238.498 ms of CUDA context initialization; neither
is claimed as kernel time.

For context, the stock production `enclosing` and adjacent-corner `width`
operations took about 0.590 s after the native M2 merge. The charged 23.207 ms
standalone GPU plan is 96.1% less time for this bounded rule section. This is
not yet a whole-run claim: the useful integration boundary remains the atomic
M2 transaction described in `M2_ALL_RULES_GPU_TRANSACTION.md`.

Run the production proof:

```sh
build/m2_via2_certificate \
  --expect-scene-sha256=3a511538eb520292eed7e00f4ad1763360502e4a6751a1bb66fab1e8838678a5 \
  --expect-metal-boxes=22947380 \
  --expect-vias=10128 \
  /path/to/m2-via2-x2.kact
```
