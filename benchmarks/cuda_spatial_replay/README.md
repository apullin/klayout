# CUDA spatial-candidate replay prototype

This directory contains both the standalone feasibility harness and the first
opt-in KLayout integration proof of concept. They test whether a GPU can turn a
large batch of copied geometry records into the same deterministic broad-phase
candidate pairs as a CPU oracle after charging packing, transfers, device work,
deduplication, and result copies.

The integration is deliberately narrow. The audited
`scan_shape2shape_different_layers` seam can submit a bipartite request, and an
independent opt-in `DeepEdges` certificate can submit a self request before a
merged `EdgeLengthFilter` operation. Exact CPU AABB revalidation, hierarchy
ownership, receiver calls, proof checks, and fail-closed fallback remain
KLayout responsibilities. The earlier description of the integration as
different-layer-only is therefore stale. The normal build has no CUDA headers
or CUDA link dependency.

The later contiguous clean-certificate islands are documented separately:
[`ACTIVE3_GPU_ISLAND.md`](ACTIVE3_GPU_ISLAND.md) and
[`M1_WIDTH_SPACE_GPU_ISLAND.md`](M1_WIDTH_SPACE_GPU_ISLAND.md).

The exact integer rectangle-union feasibility harness is documented in
[`MANHATTAN_UNION_GPU.md`](MANHATTAN_UNION_GPU.md).  It moves the seam farther
upstream than the rule-specific islands: expanded Manhattan rectangles are
sorted and scan-converted on the GPU into exact directed boundary segments,
without floating point or a dense pixel raster.  Its qualified production M2
gate compares all 4,385,384 output segments against an independently decoded
CPU-merged boundary oracle.

The first exact producer-to-consumer bridge is documented in
[`M2_FLAT_REGION_BRIDGE.md`](M2_FLAT_REGION_BRIDGE.md).  Its offline production
gate serializes the actual GPU boundary as dual-provenance `KM2BND02`, rejects
corruption/noncanonical topology before materialization, constructs an
already-merged flat KLayout `Region`, and runs stock M2.1/.2 plus F90/F270.
Serialization and read validation are charged; the heavyweight CPU oracle is
qualification-only.

The raw production host transaction and its combined real-DSO gate are
documented in
[`M2_LIVE_UNION_HOST_SEAM.md`](M2_LIVE_UNION_HOST_SEAM.md).  The gate composes
the production DSO with KLayout's runtime loader/copy/release wrapper and the
checked flat-region stitch, reproducing the exact 4,385,384-segment oracle in
one process.  GSI/deck ownership and the complete stock M2-rule suffix remain
separate unfinished milestones.

The reusable no-host-geometry F90/F270 successor is documented in
[`M2_RESIDENT_MORPHOLOGY_GPU.md`](M2_RESIDENT_MORPHOLOGY_GPU.md).  Its
separately linked production gate consumes the union core's resident strip
view, matches all 4,254,384 stock F90 edges, certifies the bounded eight-edge
space(180) set, and proves the F270 erosion empty.

## KLayout-pointer-free record contract

The on-disk/host replay record is an 80-byte POD with:

- a normalized signed-int64 AABB;
- optional signed-int64 edge endpoints;
- a globally stable nonzero uint32 record ID;
- uint32 property and hierarchy/context tokens;
- flags for endpoint presence and bipartite side A/B.

Packing produces the 48-byte device AABB record. The default broad phase does
not allocate or transfer endpoint data. The opt-in
`--edge-filter projection-overlap` path additionally packs and transfers a
separate, index-aligned 32-byte endpoint sidecar. Pointer identity is never
serialized or compared. The CPU seam maps returned IDs back to its copied
records, runs any remaining exact predicate, and publishes accepted records and
finish events in original order.

`--mode self` enumerates unordered pairs within each context. `--mode
bipartite` emits only A-to-B pairs, matching KLayout's two-input
`box_scanner2` shape/instance and instance/instance uses without paying for a
same-side superset.

Output is a sorted, duplicate-free vector of uint64 keys. Self requests use
`min(record_id) << 32 | max(record_id)`. Bipartite requests preserve side
identity and use `subject_id << 32 | intruder_id`. Multiple owner processes
must keep an owner/request namespace outside that local pair key.

## Spatial semantics

Records are inserted conservatively into uniform cells identified by signed
int64 `(cell_x, cell_y, context)` tuples. There is no global origin or dense
global grid, so sparse, far-apart hierarchy contexts do not inflate an extent
or collide in a truncated linear key.

The final emitted-pair predicate exactly mirrors `db::bs_boxes_overlap`:

```text
a.left   < b.right + enlargement
b.left   < a.right + enlargement
a.bottom < b.top   + enlargement
b.bottom < a.top   + enlargement
```

It is strict, uses one enlargement, and therefore rejects a gap exactly equal
to the enlargement. It is not inclusive intersection of two padded boxes.
`run.sh` checks gap `enlargement-1` versus gap `enlargement`, plus touching at
enlargement zero and one, in both self and bipartite modes.

Potential pairs from all occupied cells are flattened with a prefix sum and
mapped one-per-thread. Bipartite cells use an exact A-count times B-count work
range. The kernel writes a pair key or zero, a device compaction removes misses,
and device sort/unique makes duplicate-cell output deterministic.

## Projection-overlap edge filter

`--edge-filter projection-overlap` fuses a conservative exact-edge filter into
the AABB candidate kernel. It targets the FreePDK45 M2 enclosure profile:
`OverlapRelation`, projection metrics, a 90-degree angle limit, `[0, max)`
projection limits, and `IncludeZeroDistanceWhenTouching`.

The device makes a final decision only for two nondegenerate Manhattan edges in
ordered bipartite mode. Perpendicular and opposite-direction pairs are exact
angle-gate rejections. Parallel pairs require equal original direction, a
right-normal gap in `[0, distance)`, and a positive 1-D projection; zero gap
therefore covers the profile's collinear-overlap case. Missing endpoints,
diagonal or degenerate edges, and unordered scanner modes conservatively retain
the broad candidate for authoritative CPU replay. Thus unsupported geometry can
reduce the win but cannot cause a false negative.

The CPU reference implements the same conservative contract. `run.sh` includes
an exhaustive random-edge comparison and a deterministic fixture covering
accepted horizontal/vertical pairs, the strict distance boundary, zero
projection, wrong side, opposite direction, and every pass-through class.
Output reports `broad_raw_candidates`, `filtered_raw_pairs`, and
`edge_filter_charged`; the last field is the already-charged fused
enumeration/filter/compaction interval inside `kernel`, not an additional time.

## Replay format and fail-closed behavior

The v1 little-endian header is 32 bytes: `KSPAT01\0`, uint32 version, uint32
header size, uint32 record size, explicit uint32 coordinate width (64), and
uint64 record count. `--write-input` and `--input` provide a round-trip gate.

No limit silently truncates output. Status 3 and
`fallback_required=true` are returned for:

- signed coordinate overflow before any `left-enlargement` or
  `right+enlargement` arithmetic;
- a record spanning more than `--max-cells-per-record` cells;
- `--max-memberships` exhaustion;
- a cell exceeding `--max-edges-per-cell`;
- `--max-pair-work` exhaustion before allocation;
- `--max-candidates` exhaustion after exact AABB compaction.

GPU output is incomplete after any fallback signal and must not be published.
The fixtures include safe INT64_MIN/INT64_MAX-adjacent records and a separate
adversarial coordinate-overflow case.

`--edge-capture PATH` natively reads the 192-byte-header `KEDGER1` files emitted
by the opt-in KLayout M2 scanner capture. It validates all section
sizes/offsets, profile metadata, sorted-unique pair oracles, endpoint/AABB
records, and the full uint64 property before any checked narrowing to the
standalone replay's uint32 property. Capture mode takes its distance from the
file and uses ordered bipartite edge semantics. With no edge filter, GPU output
must equal the captured broad oracle. With the conservative filter, output must
remain a subset of the broad oracle and a superset of the captured exact
oracle; both oracle counts and hashes are printed.

`--edge-capture-dir PATH --edge-capture-min-records N` aggregates all `.ker`
requests at or above the cutoff into one bounded replay. Record IDs are
globally remapped and every request receives a distinct context, preventing
cross-request candidate pairs. All selected requests must have identical rule
profiles. The aggregate broad and exact CPU oracles are rekeyed, checked, and
reported together with request, record, pair, callback, and captured scanner
time totals.

## Build and run

CUDA 12.4 rejects this host's default GCC 15. Use GCC 13 explicitly and compile
for the RTX 3080's SM 8.6:

```sh
mkdir -p build/tmp
TMPDIR=$PWD/build/tmp /usr/bin/nvcc \
  -O3 -std=c++17 -arch=sm_86 -ccbin /usr/bin/g++-13 \
  spatial_replay.cu -o build/cuda_spatial_replay
./build/cuda_spatial_replay --help
```

`./run.sh` builds and runs exhaustive self/bipartite tests, strict-boundary and
int64 fixtures, a replay round trip, the million-record grid-oracle case, a
million-edge projection-filter A/B, and expected dense/overflow fallbacks. It
also builds the optional backend DSO and runs bipartite two-pair and
1024-by-1024 exact CPU-oracle gates, plus self strict-boundary, complete-cell,
1024-record exact-oracle, malformed-request, short-config, invalid-AABB, and
capacity gates. Its optional first argument selects a build directory.

CMake is also supported:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
```

## Opt-in KLayout integration

`libklayout_cuda_spatial_backend` exposes a versioned POD C ABI, free of
KLayout object pointers, from `dbCudaSpatialApi.h`. KLayout discovers it with
`dlopen`/`LoadLibrary`; ABI or symbol mismatch, CUDA errors, malformed output,
and configured capacity limits all return to the authoritative CPU scanner
before any receiver callback. Successful pair vectors must be sorted and
unique. KLayout validates all indices and reruns `db::bs_boxes_overlap` on the
CPU for every pair before publishing the first callback.

The original `klayout_cuda_spatial_run_bipartite_v1` contract is unchanged.
The optional `klayout_cuda_spatial_run_self_v1` entry point reuses the request
POD with only `subjects` populated (`intruders=null`, `intruder_count=0`) and
returns each unordered pair as two distinct ascending one-based subject IDs.
Older v1 backend DSOs remain usable for bipartite scans; the CPU loader treats
a missing self symbol as a fail-closed unsupported self request.

Build the DSO with CMake, then enable it with an explicit path:

```sh
cmake -S benchmarks/cuda_spatial_replay \
  -B build-cuda-spatial -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-spatial --target klayout_cuda_spatial_backend

export KLAYOUT_CUDA_SPATIAL_BACKEND="$PWD/build-cuda-spatial/libklayout_cuda_spatial_backend.so"
export KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
```

With the backend variable absent, the existing CPU path is retained. `auto` or
`1` asks the platform loader to find the conventional library name. The PoC
serializes DSO calls on one device; persistent buffers and a multi-owner broker
are intentionally deferred.

Runtime tuning variables and defaults are:

- `KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=100000`;
- `KLAYOUT_CUDA_SPATIAL_DEVICE=0`;
- `KLAYOUT_CUDA_SPATIAL_CELL_SIZE=128`;
- `KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64`;
- `KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096`;
- `KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=16000000`;
- `KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=64000000`;
- `KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=8000000`.

These are fail-closed resource limits, not truncation knobs. Telemetry reports
handled/fallback status, record and pair-work counts, stage times, and backend
messages. The GPU publishes candidates in deterministic pair-key order, not
the CPU sweep's callback order; the audited interaction receiver is insensitive
to that ordering.

### Atomic live M1/VIA1/M2 transaction

`KLAYOUT_CUDA_VIA1_STACK=1` enables the additive
`klayout_cuda_spatial_run_via1_stack_empty_v1` entry point. The qualified
FreePDK45 path serializes raw layers 11/0, 12/0, and 13/0 from one
`DeepShapeStore`, uploads their shared hierarchy once, expands VIA1 once, and
retains its CSR grid across M1 and M2 projection-enclosure passes. One metal
scratch allocation and grid are reused. The CPU receives one digest-bound
six-bit result covering METAL1.4, VIA1.1--1.4, and METAL2.3.

Only a complete all-six empty certificate can bypass DRC work. A malformed or
partial request/result, unsupported geometry or hierarchy, capacity limit,
size/spacing/enclosure violation, nonidentical VIA touch/overlap, CUDA error,
or exception runs all six historical CPU chains. Exact coincident VIA
duplicates are safe under the required merged Region semantics. The live hook
also requires exact source-layer provenance, 0.5 nm DBU, no breakout cells,
hole-free Manhattan metal, rectangular 65 nm cuts, and checked orthogonal
hierarchy transforms.

The generated deck preserves its original CPU shard owners when the opt-in is
off. When requested, it moves the six decisions into
`via1_upper_active12`; a missing method/backend or any proof decline performs
the complete local CPU fallback. Generate and qualify that deck with:

```sh
python3 benchmarks/cuda_spatial_replay/make_via1_stack_live_deck.py \
  --input freepdk45-eight-way-dual-repack.lydrc \
  --output freepdk45-via1-stack-live.lydrc

benchmarks/cuda_spatial_replay/run_via1_stack_live_gate.sh \
  --stock-klayout /path/to/stock/klayout \
  --live-klayout /path/to/live/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so \
  --deck freepdk45-eight-way-dual-repack.lydrc
```

The gate covers 17 live-layout cases and three lanes: stock requested
fallback, live requested fallback, and CUDA. It checks original/off ownership,
requested atomic ownership, exact enclosure and spacing boundaries, diagonal
distance, duplicates, X/Y slab witnesses, hierarchy transforms, nine
fail-closed cases, and stock-identical reports. The CPU-only host attack matrix
builds and tests a fake backend without requiring a GPU:

```sh
benchmarks/cuda_spatial_replay/run_via1_stack_host_guard.sh \
  --klayout-bin /path/to/klayout/bin
```

On the downstream-composed FreePDK45 x2 scene, the fused backend handled
849,265 contexts, 41,109,338 M1 rectangles, 20,178,022 VIA occurrences,
22,947,380 M2 rectangles, and 672,673,286 uniquely owned VIA candidates in
537.582 ms. Live lowering took 1,172.094 ms. The same-binary eight-owner A/B
changed `m1_via_class` from 259.526 to 59.832 s (**76.95% less wall**),
`m2_rules` from 230.567 to 100.867 s (**56.25% less**), and
`via1_upper_active12` from 187.276 to 59.982 s (**67.97% less**). Canonical
merged reports were identical with SHA-256
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`.
The parallel launcher changed only 278.20 to 275.97 s (**0.80% less**) because
the unchanged `m1_enclosure` owner remained critical; the lane savings are not
misreported as a whole-run win.

The projection passes also accept an exact union of Manhattan metal rectangles
as their witness. A single rectangle remains the fast path. Only misses enter
an integer-DBU strip proof, which greedily covers every row or column of the
required projection cross. Touching seams are accepted; a one-DBU gap or hole
remains a miss and forces fallback.

### Live METAL1.1/METAL1.2 certificate

`KLAYOUT_CUDA_M1_WIDTH_SPACE=1` enables one atomic clean-only transaction for
the exact merged deep-region batch
`drc_batch([width(euclidian) < 65.nm, space(euclidian) < 65.nm])` at 0.5 nm
DBU. The host lowers the hierarchy and hole-free Manhattan contours once; the
backend expands occurrences, builds one edge index, evaluates both exact
predicates, and returns a digest-bound result. Only a complete result with zero
width hits, spacing hits, uncertainty, fallback flags, and device flags may
publish the two empty edge-pair outputs. Every hit or unsupported condition
runs the untouched CPU batch exactly once.

Telemetry is enabled with
`KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1`. The focused live gate checks clean
certification, width-only and spacing-only CPU fallback, raw semantics, changed
options, reversed output order, a missing backend, and forced capacity
fallback:

```sh
benchmarks/cuda_spatial_replay/run_m1_width_space_live_gate.sh \
  --klayout /path/to/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so
```

On the downstream-composed FreePDK45 x2 M1 owner, the same generic-`O2` binary
changed 174.87 to 78.66 s: **96.21 real seconds / 55.02% less wall time**.
Generator-stripped reports were byte-identical at SHA-256
`8bb8f17de680d0b74e940e5a8ec3231a569f8e3057a6d6cde958992abfca4769`.
The live speculative path took 29.586 s, including 12.331 s of scene lowering
and a 17.255 s backend call; the backend device-pipeline timer was 504.110 ms.

`KLAYOUT_CUDA_M2_WIDTH_SPACE=1` independently enables the corresponding exact
140/140-DBU `METAL2.1/.2` profile. The host infers M1 versus M2 only from the
qualified distance pair, preserving the original exported
`CudaM1WidthSpaceBuildSpec` C++ layout. M2 lowering additionally requires the
additive `klayout_cuda_spatial_run_m2_width_space_empty_v1` symbol. An ABI-v1
backend exposing only the legacy M1 entry point therefore remains M1-only and
declines M2 before constructing its scene. Each entry point rejects the other
profile's opcode/distance combination.

### Live CONTACT.1-.3/METAL1.3 certificate

`KLAYOUT_CUDA_M1_CONTACT=1` enables a separate fail-closed use of the VIA1
stack ABI for the fixed FreePDK45 CONTACT.1-.3 and METAL1.3 rules. The receiver
must be raw CONTACT 10/0, the witness must be raw M1 11/0, and the shared layout
must use 0.5 nm DBU. Every cut must be a 65 nm rectangle, satisfy 75 nm spacing,
be contained by the M1 union, and have at least 35 nm projection enclosure on
two opposite sides. M1 containment is stronger than CONTACT.3's
active-or-poly-or-M1 union, so one complete certificate proves all four output
categories empty. The host currently serializes `(M1, CONTACT, M1)` into one
request; this is not a persistent or fused ACTIVE.3-through-METAL1.3
transaction.

The generated deck invokes one transaction when either `implant_contact` or
`m1_enclosure` owns a consumer (and still only one in unsplit `all` mode).
Only a complete empty certificate emits empty CONTACT.1-.3 or METAL1.3
categories. Properties, unsupported geometry or transforms, malformed or
partial backend results, capacity limits, cut size/spacing failures, missing
M1 containment, or an enclosure miss execute the untouched historical CPU
rules exactly once in the current owner. The transform matches the original
CONTACT.1-.3 text exactly and refuses a changed rule rather than accelerating
it. Telemetry is enabled with `KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1`. The
fail-closed capacity variables and defaults are:

- `KLAYOUT_CUDA_M1_CONTACT_MAX_CONTEXTS=4000000`;
- `KLAYOUT_CUDA_M1_CONTACT_MAX_GRID_CELLS=16000000`;
- `KLAYOUT_CUDA_M1_CONTACT_MAX_METAL_MEMBERSHIPS=300000000`;
- `KLAYOUT_CUDA_M1_CONTACT_MAX_CUT_MEMBERSHIPS=100000000`;
- `KLAYOUT_CUDA_M1_CONTACT_MAX_PAIR_WORK=2000000000000`.

Generate and qualify the live deck with:

```sh
python3 benchmarks/cuda_spatial_replay/make_via1_stack_live_deck.py \
  --input freepdk45-eight-way-dual-repack.lydrc \
  --output freepdk45-m1-contact-live.lydrc \
  --m1-contact

benchmarks/cuda_spatial_replay/run_m1_contact_live_gate.sh \
  --stock-klayout /path/to/stock/klayout \
  --live-klayout /path/to/live/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so \
  --deck freepdk45-eight-way-dual-repack.lydrc
```

The live gate covers both `implant_contact` and `m1_enclosure` owners over 15
deterministic layouts across source-off, generated-off, stock-requested
fallback, live-requested fallback, and CUDA lanes. It checks exact
CONTACT.1-.3 marker counts, equality and deficient-side combinations, exact
cut spacing, overlap/touch/duplicates, outside-M1 and malformed domains,
hierarchy transforms, a split-M1 union witness, and rejection of a changed
CONTACT.2 rule. Backend smokes additionally cover negative coordinates, a
grid-boundary seam, and horizontal and vertical one-DBU gaps.

On the downstream-composed FreePDK45 x2 `m1_enclosure` shard, a same-binary
external-wall A/B changed 202.73 to 88.06 s: 114.67 s, or **56.56%**, less
lane time. The complete host-to-host transaction took 3,063.94 ms, including
1,823.34 ms of hierarchy lowering and a 1,240.59 ms validated backend call;
the backend's complete CUDA pipeline reported 235.58 ms. Canonical reports
were identical with SHA-256
`d056b808e6f2134e60286e35247a92e3a2f6d2eaa26b463fd572aa7e0652146d`.
These are critical-lane and transaction measurements, not a claim that the
parallel full launcher fell by 114.67 s.

The later `implant_contact` same-binary production gate changed 219.20 to
132.33 s: **86.87 real seconds / 39.63% less owner wall time**. Both reports
retained canonical SHA-256
`b0e94aa57f09535c1b283e47838fba1830ffd17f9f88c9e15c9c2512aff95f56`.
This owner-local result includes the CONTACT.1-.3 transaction but predates the
separate CONTACT.4 certificate below.

### Live CONTACT.4 certificate

`KLAYOUT_CUDA_CONTACT4=1` enables a narrowly qualified clean-only certificate
for the exact FreePDK45 expression
`active.enclosing(cont, 5.nm, euclidian)` at 0.5 nm DBU. The host requires a
merged ACTIVE 1/0 primary and raw CONTACT 10/0 secondary from the same
DeepShapeStore, hierarchy root, and layout. It lowers the complete hierarchy
once, indexes raw CONTACT edges, streams merged ACTIVE edges, and restores
primary/secondary order before the exact 10-DBU predicate. A zero-hit result
is consumable only when the digest and all request/result census fields echo,
the actual candidate count is within capacity, and hit, uncertainty, fallback,
and device flags are all zero. Every other outcome runs the untouched CPU
expression exactly once.

Telemetry is enabled with `KLAYOUT_CUDA_CONTACT4_TELEMETRY=1`. The focused
gate covers strict 9/10-DBU spacing, Euclidean 6/7 versus 6/8 endpoint
distance, partial projection, collinear touch/overlap/separation, raw geometry
whose merge changes edge subsegments, hierarchy transforms and arrays, raw
primary semantics, changed options, reversed operands, wrong DBU, missing
backend, and capacity fallback:

```sh
benchmarks/cuda_spatial_replay/run_contact4_live_gate.sh \
  --klayout /path/to/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so
```

The gate passes 480,038 direct GPU/oracle/KLayout predicate comparisons and
19 live CPU-oracle cases: 14 qualified CUDA cases, five qualification
declines, and two explicit fail-closed lanes. On the production x2
`implant_contact` owner, a same-binary external-wall A/B changed **132.50 to
120.29 s: 12.21 real seconds / 9.22% less owner wall time**. The backend
classified exactly 31,899,588 candidates in 3.978 s after 3.379 s of host
lowering, with zero hits or uncertainty. Both reports retained canonical
SHA-256
`b0e94aa57f09535c1b283e47838fba1830ffd17f9f88c9e15c9c2512aff95f56`.

Two same-binary full-gate controls and two CONTACT.4 candidates changed mean
wall from **142.18 to 129.385 s: 12.795 real seconds / 9.00% less full wall
time**. Individual control walls were 141.67 and 142.69 s; candidates were
129.39 and 129.38 s. Mean `implant_contact` owner wall changed 137.240 to
124.473 s: **12.767 real seconds / 9.30% less owner wall time**. Every merged
report retained canonical SHA-256
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`.

### Live ACTIVE.4 WELL-union certificate

`KLAYOUT_CUDA_ACTIVE4_WELL_UNION=1` enables a clean-only certificate for the
qualified FreePDK45 `active.not(nwell.or(pwell))` expression. The host captures
raw NWELL, PWELL, and ACTIVE from one hierarchy and authenticates their shared
layout, root, transforms, DBU, source layers, and immutable scene digests. The
backend expands NWELL and PWELL on device, constructs their exact canonical
x-slab union, then proves that every rectangle in the exact qualified ACTIVE
rectangulation is covered across every slab it spans. The proof accounts for
all rectangles, slab visits, and interval searches with checked 64-bit
counters; partial coverage, holes, cap exhaustion, incomplete work, malformed
input, CUDA errors, and nonzero device flags all decline the transaction.

Only a complete zero-hit certificate suppresses `active.not(well)`. A hit or
any unsupported condition executes the untouched literal CPU expression, so
the GPU never synthesizes diagnostic markers. Co-owned WELL rules retain the
literal union; the certificate removes only the expensive ACTIVE.4
difference. Telemetry is enabled with
`KLAYOUT_CUDA_ACTIVE4_WELL_UNION_TELEMETRY=1`.

The focused gate covers clean, outside, partial-overlap, hole, hierarchy,
forced-cap, and missing-backend lanes. Its production-scene CPU control versus
CUDA candidate changed **33.303 to 6.150 s: 27.153 real seconds / 81.53% less
focused wall time**, with identical reports. The balanced production harness
may additionally set
`KLAYOUT_CUDA_ACTIVE4_WELL_UNION_TERMINAL_RESET=1`; this is restricted to the
terminal, sole-CUDA-owner ACTIVE.4 shard. It synchronizes, checks and trims the
default pool, then performs a checked device reset before releasing the shared
device lease. The normal feature remains opt-in and fail-closed without
assuming terminal ownership.

### Balanced full-launch gate

`run_balanced_full_gate.sh` packages the qualified configuration-level gate.
It regenerates the live VIA1-stack plus CONTACT.1-.3/METAL1.3 deck from the
eight-owner source, applies the three-way antenna transform, and moves only
`CONTACT.6` into the underloaded grid owner. The compound `METAL1.1` and
`METAL1.2` traversal remains intact.

The runner fixes the qualified ten-owner launch order, selectable bounded
process concurrency, CUDA resource limits, and certificate opt-ins. The source
deck controls inner KLayout threads. It then requires ACTIVE.3,
METAL1.1/METAL1.2, VIA1-stack,
CONTACT.1-.3/METAL1.3, and CONTACT.4 telemetry before comparing the canonical
merged report. The older CONTACT.1 selected-empty certificate is intentionally
not required because the atomic M1-contact certificate bypasses that CPU
expression. `--without-contact4` retains every other opt-in and provides an
explicit same-binary performance control. Supply the deck-bound manifest
created from a CPU `drc_shard=all` reference and the ten CUDA-enabled shard
reports. Do not create the reference with CUDA enabled: that changes
historical category order even when the semantic report is otherwise equal.

`--split-lower-antenna` and `--split-upper-antenna` are independent,
default-off owner splits. The lower mode replaces `antenna_m1_m2` with
independent `antenna_m1` and `antenna_m2` owners. The upper mode replaces
`antenna_m3_m10` with independent `antenna_m3` and `antenna_m4_m10` owners.
The selected launch has ten owners by default, eleven with either split, and
twelve with both; its `--jobs` value may not exceed that owner count. Every
mode requires its own deck-bound manifest. The accepted upper-only
32-core-budget screen used a source deck with `threads(2)`, so the eleven
processes request 22 inner workers. Its full report retained canonical SHA-256
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`;
the antenna critical lane fell from 96.431 to 57.769 s while unchanged M2
kept full wall flat at 101.36 -> 101.66 s.

The mutually exclusive `--fuse-metal-antenna` mode selects one
`antenna_m1_m4` owner for all ten metal outputs. It preserves the literal
staged CPU chain and is the fallback/hook point for the atomic M1-through-M4
CUDA transaction. With no other owner splits it selects nine owners; with
both implant/contact and ACTIVE.1/.2 splits it selects eleven.

```sh
bash benchmarks/cuda_spatial_replay/run_balanced_full_gate.sh \
  --klayout /path/to/cuda/klayout \
  --backend /path/to/libklayout_cuda_spatial_backend.so \
  --source-deck /path/to/freepdk45-eight-way-dual-repack.lydrc \
  --manifest /path/to/freepdk45-balanced-bound.json \
  --input /path/to/sram_1rw0r0w_64_4096_freepdk45__x2.gds \
  --top-cell sram_1rw0r0w_64_4096_freepdk45__x2 \
  --reference /path/to/trusted-full-report.lyrdb \
  --keep-work
```

The qualified source deck, generated balanced deck, manifest, x2 input, and
canonical report SHA-256 values are respectively:

- `3e981b9389a67c6c1c4b08f0640d8750ca78990c868c5686fa8ebbd401cba72c`
- `5b2d78c211cff3a68388f22b6d74e67011272d0b851d635da28123c8c95bda33`
- `769fd241d6f07fbab2543e30a2f15ee257ad5b960216321cce72d7c189b0bdc1`
- `74911a2111a3421912e54538bf55cd12e50164f43cd1a8c47411602e64c91d98`
- `01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`

The runner records every supplied and generated artifact hash, launcher
provenance, external wall time, per-owner timings, and CUDA telemetry under
its temporary work directory. Without `--keep-work`, that directory is
removed after the gate.

The current `--with-active4-well-union` jobs-10 configuration also selects the
exact raw-M1 base-width/space certificate. Its generic strip producer still
supports 64-million-event windows, but the 10 GiB-qualified M1.1/M1.2 owner
uses deterministic 16-million-event windows. Window boundaries do not change
the canonical global strip view or predicates; they only bound temporary
sort/reduction storage. The larger 64M and 32M plans are retained as rejected
allocation-cliff observations, while three 16M full gates completed exact with
the same 134,495,138 stitched intervals.

Those three internal full-launch walls were **35.458, 35.488, and 35.376 s**
(35.441 s mean, 0.111 s range). Against the preceding accepted 39.841-second
mean, that is **4.400 real seconds / 11.05% less full-launch wall time**.
Child-plus-merge wall averaged 30.623 s, the `antenna_feol` owner averaged
24.298 s, and the independent ACTIVE.1/.2 owner became the 30.598-second
mean roof. Every required transaction certified, all runtime artifacts were
unchanged across execution, and all merged reports retained canonical
SHA-256
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`.

### Production host control-build result

Rebuilding the current CUDA host at clean commit `f15739559610a8c3`, tree
`3f5ec23262ed2a27`, with Clang 22, full LTO/LLD, and `znver2` code generation
turned the qualified balanced configuration into a second end-to-end win. The
CUDA backend, deck, manifest, x2 input, owner order, and eight-job limit were
unchanged.

Three preceding balanced full-launch walls were 184.556784, 186.398209, and
184.242187 seconds. Three production-build runs took 155.075069, 156.075237,
and 156.136431 seconds. The means are 185.065727 -> 155.762246 seconds:
**29.303481 real seconds / 15.83% less wall time and +18.81% throughput**.
The new full-range spread is 0.68%. Child-plus-merge fell 18.17%, from
180.514787 to 147.723113 seconds.

M1 width/space remains the pole at 147.715584 seconds mean, 18.17% below its
preceding 180.507539 seconds. Implant/contact follows at 132.263260 seconds,
17.57% below 160.457722 seconds. Every counted run produced all four required
CUDA certificate families. All three raw reports are byte-identical at
SHA-256
`89fa723caf5ecd620993d62c2d2e63ca6eaf8c14fe0dac25a7abf9c2992c1472`;
all canonical reports retain
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`.

The 29:26.31 source-bound build produced:

- executable SHA-256
  `007f6ddc0750d7b6b8d6c8287756a2e7a94dca298e5a4ded0c4e4bb6cb5cc252`;
- build-manifest SHA-256
  `26b7137e6fe9ff45ed0051e302d255af7fc8c1cbc4ff39eef1269fc6f3982fce`;
- lifecycle-state SHA-256
  `3a9383a9df70cf57df21425941ee83b1eb36aa6beacbefe69d2cf238f0aedb78`.

### Current-host mixed-PGO qualification

The same clean source commit
`f15739559610a8c31b21bf47a185ee22eaeded86`, tree
`3f5ec23262ed2a2797f5b8a6f47d5fa0ff979337`, was retrained with one balanced
FreePDK45 CUDA x2 run and one real Sky130 HG0 S5 run.  Counter-balanced
FreePDK45:Sky130 merge weights of 1:4 left 7.832967875% weighted imbalance.
The merged profile SHA-256 is
`81e0493fd312dce2df88d5dad2ff3ae0f3bacd9d7e756a1dab4d109f95974a8d`.
The control and PGO-use executable SHA-256 values are respectively
`007f6ddc0750d7b6b8d6c8287756a2e7a94dca298e5a4ded0c4e4bb6cb5cc252`
and
`8e4a435d76ccbc9d986af43c61772dcbe45f82000d6a4deb382ba6b1e73e5d40`.

Formal serial qualification used three observations per treatment and PDK.
FreePDK45 CUDA x2 full-launch means were 154.78227078542113 s control versus
140.58269528571205 s PGO: **14.199575499709084 real seconds /
9.173903075% less wall time and +10.100514484% throughput**.  Sky130 HG0 S5
means were 215.93017702068514 versus 188.39364087659246 s:
**27.53653614409268 real seconds / 12.752518672% less wall time and
+14.616489185% throughput**.  All four full-range spreads were below 2%, and
all twelve primary reports were exact.

FreePDK45 raw and canonical reports retained SHA-256
`89fa723caf5ecd620993d62c2d2e63ca6eaf8c14fe0dac25a7abf9c2992c1472`
and
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`;
Sky130 S5 retained normalized SHA-256
`2c9f660d7b2d7186329c510333083bfe19ab17779fe42d66c936feac0b45fdb4`.
The nonempty sentinels and the 86-item antenna and 1,587-item mixed-hierarchy
fixtures also passed.  Profile warnings while linking auxiliary buddy
executables reflect their use of the main-program profile and are not a
qualification claim for those buddies; the timed main `klayout` executable
was source-bound and fully qualified.

The original publisher failed after freezing every timed sample because it
treated one recorded Sky report mapping as a path.  The audited recovery
reran no builds, training, or measurements, retained frozen-evidence SHA-256
`df0d42e761810f3b3a5dd83ad22a15d246b94cd4367f2cf5574d1aff17f03388`,
and published formal receipt SHA-256
`0c55bfc620d75077d7ba8550bea18c6b1510173fd0fc17dddb36fce20d84a662`.

### Opt-in `DeepEdges` merge certificate

`KLAYOUT_CUDA_DISCONNECTED_MERGE=1` enables a separate, fail-closed
certificate in `DeepEdges::filtered()`. The setting is read once per process,
so set it before the first eligible filter call. Disabled builds and processes
retain the legacy hierarchy-aware merge path.

The certificate is attempted only for the true-only result of the exact
concrete `EdgeLengthFilter`, with merged (not raw) semantics and no usable
merged cache. It additionally requires:

- no breakout cells, non-orthogonal/complex instance transform, transform or
  hierarchy-count overflow, or nonzero source property ID;
- at most `KLAYOUT_DEEP_EDGE_CERT_MAX_STORED_EDGES` stored source edges
  (default `100000`) and at least 8x flat occurrence reuse;
- an enabled self-request backend above
  `KLAYOUT_CUDA_SPATIAL_MIN_RECORDS`, plus an exact host-side membership
  capacity preflight;
- successful, complete GPU output with every returned index and transformed
  AABB validated on the CPU.

Local canonicalization is not itself accepted as a global merge. KLayout first
groups canonical edges by original-cluster provenance and endpoint
connectivity, then submits one overflow-checked marker AABB per hierarchy
occurrence. An empty interaction set proves that no merge cluster crosses an
occurrence boundary; only then is the locally canonicalized candidate installed
as the merged cache.

There is one narrower nonempty result, reported as `selected-empty`, for
coincident duplicate rectangles. It is valid only when local EdgeOr preserved
the complete oriented edge multiset including multiplicity, every marker group
is exactly the four nondegenerate axis-aligned sides of its provenance box,
the requested length filter rejects every canonical edge, and every GPU pair
has byte-identical transformed AABBs. Global merging can then only retain,
cancel, or reorient edges of the same rejected lengths, so the selected output
is provably empty. KLayout returns a fresh merged empty result and deliberately
does not publish a merged cache on the source object. Equal AABBs that are not
literal rectangle boundaries, touching or partially overlapping boxes,
containment, cancellation, selected sides, split/false-output filtering, or
any incomplete backend result all fall back to the unchanged CPU path.

`KLAYOUT_DEEP_EDGE_CERT_PROFILE=1` emits one
`KLAYOUT_DEEP_EDGE_CERT` line per attempted certificate. `accelerator_ms`
measures the backend call, while `decision_ms` measures the certificate
decision through the reported outcome; neither is a whole-filter or whole-run
timer. A nonempty backend result also emits a
`KLAYOUT_DEEP_EDGE_CERT_PAIRS` classification.

The focused integration runner exercises the successful disconnected and
selected-empty outcomes and the touching, complex-transform, cancellation,
stored-edge, reuse, unsafe-mode, and nonrectangle fallbacks:

```sh
bash benchmarks/cuda_spatial_replay/run_deep_edges_integration.sh \
  [backend.so] [klayout-build-dir]
```

It requires a built backend DSO, `ut_runner`, and `db_tests.ut`, selects
`dbDeepEdgesTests:24` through `:33`, checks the expected telemetry, and fails if
obsolete `total_ms` certificate telemetry appears. The standalone `run.sh`
still qualifies the pointer-free backend ABI and GPU/CPU oracles; the
integration runner is the additional KLayout proof gate.

On the downstream-composed x2 FreePDK45 CONTACT shard, the same-binary control
took 218.32 s and the selected-empty candidate took 155.10 s: 63.22 s, or
28.96%, less shard wall time. The reports were identical with SHA-256
`9511c638ae7ed175e9bca5ece71068602b6c804bd230aecc5e56fa3cbefad305`.
The 369.611 ms GPU self request and 2,412.654 ms complete certificate decision
are included in the candidate wall time. This is an x2 CONTACT-shard result,
not a 63.22 s reduction of the original 156.692 s parallel full launcher.
Local evidence is retained at
`/home/pullin/personal/klayout/cuda-evidence-temp/contact-x2-duplicate-empty-ab-20260723T154225Z`.

## Charged timings

The executable reports input generation/read, CPU pack/ID validation, CUDA
setup/allocation, H2D, grid/sort/enumeration/compaction (`kernel`), candidate
sort/dedup, final D2H, complete GPU-pipeline wall, and CPU-reference wall.
`--warmup N --repeat N` provides a steady repeated end-to-end mode; reported
GPU timings are per-repeat means and every output is checked for determinism.

On the local RTX 3080, the exact million-record bipartite case
(`4` contexts, `24,076,886` flattened cell pairs) produced `2,552,846`
sorted unique pairs and matched the CPU grid oracle exactly, including hash
`0x51417d671935870e`, in five independent launches. Each launch used one warmup
and three measured GPU repetitions. Packing and the CPU oracle each ran once
per launch; GPU fields are the mean of that launch's three repetitions.

Across the five launches:

- CPU pack averaged 42.922 ms (42.011-44.532 ms);
- the device pipeline averaged 36.014 ms (35.703-36.244 ms), including setup,
  allocation, transfers, kernels, device sort/dedup, result copy, and per-call
  device-buffer teardown;
- pack plus device pipeline averaged 78.936 ms (78.107-80.236 ms), the charged
  host-to-host candidate-generation total;
- the CPU grid oracle averaged 1,349.768 ms (1,342.496-1,360.444 ms).

These are synthetic replay numbers, not a KLayout whole-run speedup. The
host-to-host number was 94.2% less wall time than this particular CPU oracle,
but excludes synthetic input generation (98.484 ms average) and downstream
exact CPU replay. The reference algorithm is an oracle rather than the accepted
KLayout scanner.

## Utilization result

The executable queries theoretical launch occupancy, but explicitly labels it
as distinct from achieved utilization. Nsight Compute hardware counters are
currently unavailable to this user (`ERR_NVGPUCTRPERM`), so SOL/achieved-
occupancy claims cannot be made yet.

One repeated bipartite process sampled at 82-84% `nvidia-smi dmon` SM activity.
Three concurrent processes, modeling the three dominant owners, sustained
95-97% SM activity for ten consecutive one-second samples (memory activity
34-52%). Thus greater-than-90% aggregate GPU activity is feasible on this box
without changing correctness. It did not create extra GPU throughput:
per-process pipeline wall rose from about 18.16 ms warm to 55.38-56.12 ms, so
three-process aggregate throughput stayed near the single-GPU ceiling.

The utilization-only runs used `--reference none` only after the corresponding
exactness gate. To reproduce a long sample:

```sh
nvidia-smi dmon -s u -d 1
./build/cuda_spatial_replay \
  --records 1000000 --contexts 4 --world-size 20000 --object-size 64 \
  --mode bipartite --enlargement 32 --cell-size 128 \
  --warmup 2 --repeat 250 --reference none \
  --max-memberships 16000000 --max-pair-work 64000000 \
  --max-candidates 16000000
```

When counter permissions are enabled, use Nsight Compute for useful SM
throughput and achieved occupancy, and Nsight Systems for copy/kernel overlap:

```sh
ncu --target-processes all \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  ./build/cuda_spatial_replay --records 1000000 --reference none

nsys profile --trace=cuda --stats=true \
  ./build/cuda_spatial_replay --records 1000000 --reference none
```

Activity alone is not an acceptance criterion; serialization, CPU replay, and
whole-run savings still have to clear the project gate.

## Synthetic fused-filter result

On the deterministic million-edge bipartite sample, the projection filter
reduced raw candidates from `2,729,090` to `1,613,186` (40.9% fewer) and unique
CPU replay pairs from `1,337,218` to `791,540` (40.8% fewer), with exact
GPU/CPU agreement. The fused filter interval was 1.446 ms. Device-pipeline wall
was 29.872 ms versus 29.233 ms without the filter, so this synthetic broad-phase
run alone was 2.2% slower before charging the downstream exact CPU work it
eliminates. The real M2 capture/oracle gate, including CPU replay, is therefore
the performance decision point; candidate reduction by itself is not booked as
a whole-run win.

## Eight-owner integration and next kernels

The accepted DRC launcher has eight independent KLayout owner processes, with
three dominant owners. The local CUDA toolkit does not install
`nvidia-cuda-mps-control`; the measured three-process run therefore used normal
driver context scheduling. Two production experiments remain:

1. Install/qualify MPS only if the actual GeForce/driver combination supports
   it. It preserves independent processes but duplicates device allocations.
2. Prefer a single GPU broker owning persistent buffers and 2-4 non-default
   streams. Owners send pointer-free records through bounded shared-memory
   queues. The broker namespaces requests, batches the three critical owners,
   and overlaps H2D for batch N+1, compute for N, and D2H for N-1 with
   double-buffered pinned memory. Completion is routed back in receiver order.

Profiling shows edge broad phase alone is too small to justify integration, so
the generic AABB/self/bipartite contract targets higher-ceiling hierarchy
polygon-to-instance and instance-to-instance scanners as well. A second
device-neutral replay candidate is shielding incidence: sort edge/shield
incidences and perform set difference on GPU, return stable IDs, then run the
final geometric `shields()` decision and ordered publication on CPU. Both
candidates must retain exact CPU fallbacks and charge broker queueing and every
transfer.
