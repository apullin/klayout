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
