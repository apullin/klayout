# Standalone M1 width/spacing GPU scene island

Status: exact production-scene replay passed; live CPU-skip integration remains
open.

`m1_width_space_scene_island.cu` moves one contiguous METAL1.1/METAL1.2
interval onto CUDA.  Starting from compact merged-M1 polygon templates and
resolved hierarchy contexts, the device:

1. expands directed edge occurrences and stable world-polygon identities;
2. builds one shared uniform-grid CSR index;
3. assigns each unordered candidate edge pair to one deterministic grid cell;
4. evaluates the exact qualified Euclidean width and spacing predicates; and
5. returns only complete/raw-hit/uncertain state, counters, and bounded samples.

No candidate stream or marker geometry returns to the host.  This first path
is clean-only: a complete traversal with zero raw hits and zero uncertainty can
certify that both rule outputs are empty.  Any hit or uncertainty must run the
untouched CPU batch.  Shielding can remove raw edge-pair hits, but it cannot
create a hit from a complete zero-hit superset, so shielding is not needed for
the empty certificate.

## Trust boundary

The only qualified operation is the FreePDK45 compound batch:

```text
width(euclidian) < 65 nm
space(euclidian) < 65 nm
scene DBU = 0.5 nm
strict integer distance = 130 DBU
```

The input must be the exact merged-M1 universe.  This property cannot be
inferred from polygon bytes: treating overlapping raw polygons as merged can
lose same-world-polygon width candidates.  A live caller must therefore obtain
the scene from `cuda_m1_width_space_build_scene` at the
`merged_deep_layer()` seam and bind its canonical digest to the request.
Standalone input additionally requires externally qualified scene and
source-GDS digests plus an explicit producer assertion.  The assertion defaults
false; it is not a file-content heuristic.

Malformed topology, non-Manhattan or self-intersecting contours, holes,
unsupported transforms, coordinate overflow, counter overflow, capacity
exhaustion, CUDA failure, digest mismatch, and predicate uncertainty all
decline the entire transaction.

## Build and run

The direct wrapper builds and runs the fixtures plus a synthetic benchmark:

```sh
benchmarks/cuda_spatial_replay/run_m1_width_space_scene_island.sh
```

The CMake project also exposes `m1_width_space_scene_island`.  Useful focused
commands are:

```sh
m1_width_space_scene_island --self-test

m1_width_space_scene_island \
  --benchmark-contexts=2680764 \
  --repetitions=5 \
  --no-host-oracle
```

Packed-scene mode is an interoperability gate, not the intended live
production seam:

```sh
m1_width_space_scene_island \
  --packed-scene=qualified.kact \
  --expect-scene-sha256=HEX \
  --trust-merged-layer0 \
  --no-host-oracle
```

The exact production replay uses the narrower canonical host-scene format:

```sh
m1_width_space_scene_island \
  --host-scene=/tmp/m1-width-space-merged-x2-direct.km1ws \
  --expect-scene-sha256=df713200c1271e510ac2ecd1bdc060054e69451658f0c4327d64b228f8925235 \
  --no-host-oracle
```

See `M1_WIDTH_SPACE_HOST_SCENE_FORMAT.md` for the transport and producer trust
contract.

## Current gates and measurement

The RTX 3080 qualification passed:

- 19/19 scene fixtures covering strict 129/130/131 axial boundaries,
  Euclidean corner distance, same-polygon notches, all eight transforms,
  default-untrusted and raw touching/coincident provenance rejection,
  unsupported geometry, and pair-capacity fallback;
- the separate exact-predicate differential with 100,077 checks, including
  50,036 direct KLayout-source versus device pairs;
- a digest-qualified hierarchical KACT ingestion smoke with zero device flags;
  and
- five repetitions of a deliberately regular full-flat-count stress scene.

That stress scene contains 2,680,764 polygons, expands 10,723,056 edges, and
classifies 101,117,574 unique candidate pairs.  Warm observations were:

```text
host validate/lower: 122.9--126.8 ms
CUDA upload through compact result: 66.9--67.4 ms
warm total: 189.9--193.9 ms
```

The exact merged-M1 x2 replay then consumed a source- and scene-digest-pinned
`KM1WSCN1` capture with 543,760 stored / 2,680,764 flat polygons and
7,320,532 stored / 24,432,912 flat edges.  It classified 276,872,266 unique
pairs with zero hits, uncertainty, or device flags:

```text
CUDA upload through compact result: 200.5--212.2 ms
standalone file load: 2.87--2.91 s
exhaustive host validate/lower: 8.37--9.37 s
standalone total: 11.68--12.73 s
```

The CPU reference spacing operation took 117.42 aggregate seconds.  The
approximately 0.20-second device plan is therefore about 99.8% less compute
time for the exact production geometry.  The standalone total is deliberately
conservative: live integration consumes the canonical scene in memory and can
avoid the 285-MB file load, redundant validation, and standalone CUDA context
initialization.

The next gate is live atomic integration and whole-run timing.  The current
critical-path model predicts about 14.7--15.6 seconds (roughly 10--11%) of
immediate whole-run reduction because the parallel implant/contact shard then
becomes the approximately 118--124-second owner.
