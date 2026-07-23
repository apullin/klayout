# Standalone ACTIVE.3 GPU scene island

Status: experimental proof, not a production CPU-skip integration.

`active3_scene_island.cu` is the first contiguous GPU-owned implementation of
the captured FreePDK45 ACTIVE.3 clean-certificate path. It starts after a
checked host lowering of `KACTSCN1` hierarchy into compact cell contexts. From
the first upload through result reduction, the device owns:

1. hierarchy-aware expansion of the small WELL edge stream;
2. construction of a dense uniform-grid CSR index;
3. streaming of every ACTIVE template edge through every retained context,
   with transforms applied in registers;
4. exact AABB culling and the bounded Euclidean ACTIVE.3 predicate; and
5. compact violation/uncertain counters and a bounded diagnostic sample.

The implementation never creates a flat ACTIVE-edge array on the host or
device. On the x2 workload, 2,912 stored ACTIVE edges are streamed as
98,754,896 world edges.

## Trust boundary

The executable performs bounded loader validation: fixed ABI/header checks,
SHA-256, canonical section ranges and padding, dense IDs and ranges, directed
Manhattan contour closure, clockwise templates, local/subtree bounding boxes,
array arithmetic, reachability, and hierarchy acyclicity. It is not a
replacement for the fuller independent Python validator's simplicity and
canonical-order checks.

An accepted invocation therefore requires both:

- a prior successful `validate_active3_packed_scene.py` run; and
- an explicit scene digest from an external trusted manifest, passed as
  `--expect-scene-sha256=HEX`.

Omitting or mismatching that fingerprint returns `UNCERTAIN`. Unsupported DBU,
non-integral rule conversion, malformed geometry, overflow, hierarchy cycles,
capacity exhaustion, CUDA errors, or any exact-predicate uncertainty also
return `UNCERTAIN`; none can produce a clean certificate.

## Clean-only production meaning

The captured WELL operand is merged, but the captured ACTIVE operand retains
raw polygon contours. KLayout unions those ACTIVE intruders locally before
reporting the enclosing check. Every boundary segment of that union is a
directed subsegment of a raw contour edge. With partial-edge Euclidean
semantics, extending such a segment back to its raw parent cannot remove an
existential `< 110` match. Additional raw internal edges, cross-context pairs,
or shielding can only add candidates or remove later markers. Therefore:

```text
complete traversal + zero raw hits + zero uncertainty => safe CLEAN certificate
any raw hit                                  => run pristine CPU check
```

The reverse implication is false. For example, a second ACTIVE polygon can
cover the first polygon's near internal edge while extending the merged union
outside WELL; KLayout then reports no enclosure marker although the raw edge
pair matched. The standalone executable consequently reports
`RAW_HIT_FALLBACK` with exit status 3 for any raw hit. It never presents raw
hit counts as exact KLayout violations or markers. Exact nonempty results
would require reproducing ACTIVE union boundaries, shielding, marker
fragments, and hierarchy ownership.

The only qualified unit conversion is:

```text
scene DBU = 0.0005 um = 0.5 nm
ACTIVE.3 = 0.055 um = 55 nm = 110 scene coordinates
```

The shared predicate rejects the earlier mistaken 55-coordinate value.

KLayout keeps transformed polygon hulls clockwise. For composed mirror
transforms `m0..m135`, the island applies the affine transform and reverses
each emitted directed edge. `active3_mirror_transform_differential.rb`
compares this rule with KLayout under all eight simple transforms.

## Spatial-candidate semantics

WELL edges occupy every 2000-coordinate grid cell touched by their exact
axis-aligned span. Each ACTIVE edge queries cells touched by its span expanded
by 110 coordinates. A pair which shares multiple cells is assigned to the
componentwise-lowest cell in the two grid-span intersection, so it is visited
once. A final exact AABB test removes pairs that merely alias in a coarse grid
cell.

Consequently, `candidate_pairs` counts each WELL/ACTIVE pair whose exact WELL
span intersects the distance-expanded ACTIVE span exactly once. It is
independent of coarse-cell multiplicity. Every such pair is classified on the
device immediately; there is no candidate list or host round trip.

## Build and run

The standalone build does not require a KLayout relink:

```sh
TMPDIR=/tmp/klayout-active3-island-build/tmp /usr/bin/nvcc \
  -O3 -std=c++17 -arch=sm_86 -lineinfo \
  -ccbin /usr/bin/g++-13 -Xcompiler=-Wall,-Wextra \
  benchmarks/cuda_spatial_replay/active3_scene_island.cu \
  -o /tmp/klayout-active3-island-build/active3_scene_island

/tmp/klayout-active3-island-build/active3_scene_island \
  --expect-scene-sha256=8104f7e61d943038e2f55312efcf5f48a8f33d95228bd754cdc89ac97e51bafe \
  /tmp/active3-scene-x2-final.kact
```

`run_active3_scene_island.sh` wraps the same compile. The benchmark CMake
project also provides an `active3_scene_island` executable target using the
same optimization flags as `active3_exact_predicate_test`; the source includes
the shared `active3_exact_predicate.cuh` hook directly.

## Correctness gates

The initial qualification on an RTX 3080 passed:

- full independent packed-scene validation for the 64K and x2 captures;
- exact scene fingerprints on every accepted real-scene run;
- all-eight-transform KLayout directed-edge differential;
- a mirrored deliberate-hit scene: KLayout 32 flat edge pairs and GPU 32 raw
  hits on that already disjoint fixture, with zero uncertain;
- an ACTIVE-union counterexample: KLayout reports one marker for the first raw
  polygon alone but zero after adding an overlapping covering polygon; the GPU
  sees the one internal raw hit and returns `RAW_HIT_FALLBACK` with status 3;
- a deterministic randomized/grid-boundary scene: CPU brute force and GPU
  both found 1,536 unique candidates and 384 violations;
- twelve fail-closed gates: truncated, corrupt hash, unsupported DBU/distance,
  reserved header, hierarchy cycle, coordinate-domain overflow, context/grid/
  membership/pair-work capacity, and missing/mismatched external
  fingerprints; and
- `compute-sanitizer --tool memcheck` on both the deliberate fixture and the
  64K scene, with zero reported errors.

The randomized and deliberate fixtures are produced by
`active3_scene_island_random_fixture.rb` and
`active3_scene_island_fixture.rb`; the raw-hit fallback regression is
`active3_scene_island_merge_fallback_fixture.rb`. The optional
`--verify-bruteforce` mode is capacity-bounded and intended only for such small
differential scenes.

## Measured x2 result

An independent final five-process reproduction produced:

```text
external wall: 0.78 s, 0.70 s, 0.70 s, 0.71 s, 0.69 s
external median: 0.70 s
first-driver-cold CUDA initialization: 244.6 ms
subsequent CUDA initialization: 164--172 ms
GPU-owned island (upload through checked CUDA cleanup): 28.210--28.296 ms
ACTIVE stream/query: 23.239--23.245 ms
```

Every run had the same deterministic census:

```text
contexts=849265
well_edges=8924
active_edges=98754896
grid_cells=660231
memberships=774466
candidate_pairs=44623826
raw_hits=0
uncertain=0
device_flags=0
```

The approximately 41.58-second isolated CPU ACTIVE.3 observation is useful
workload context, but the 0.69--0.70-second number is not yet production DRC
wall: this executable starts from an already captured derived scene and has
not replaced KLayout's live operation. The defensible result is that the
standalone clean-certificate scene path at the 0.70-second median is about
98.3% less wall than that reference (roughly 59x). Even the
first-driver-cold 0.78-second tail is about 98.1% less wall (roughly 53x).
Production integration and its source/capture boundary remain to be measured.

The remaining large host costs are bounded scene load/SHA validation
(about 355 ms) and CPU hierarchy lowering (about 40 ms). CUDA context creation
costs about 160 ms in each standalone process but would be amortized by a
resident backend. The next implementation bites are a GPU hierarchy BFS,
resident scene reuse, and integration behind the full external provenance and
CPU-fallback gate.

Nsight Compute 2024.1 was invoked for
`query_active_kernel`, but the host denied access with
`ERR_NVGPUCTRPERM`. SM throughput, DRAM throughput, and achieved occupancy are
therefore still unverified; no utilization percentage is inferred from kernel
wall time.
