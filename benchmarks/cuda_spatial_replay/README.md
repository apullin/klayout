# CUDA spatial-candidate replay prototype

This directory contains both the standalone feasibility harness and the first
opt-in KLayout integration proof of concept. They test whether a GPU can turn a
large batch of copied geometry records into the same deterministic broad-phase
candidate pairs as a CPU oracle after charging packing, transfers, device work,
deduplication, and result copies.

The integration is deliberately narrow: only the audited
`scan_shape2shape_different_layers` seam can call the backend. Exact CPU AABB
revalidation, hierarchy ownership, receiver calls, and fail-closed fallback
remain KLayout responsibilities. The normal build has no CUDA headers or CUDA
link dependency.

## KLayout-pointer-free record contract

The on-disk/host replay record is an 80-byte POD with:

- a normalized signed-int64 AABB;
- optional signed-int64 edge endpoints;
- a globally stable nonzero uint32 record ID;
- uint32 property and hierarchy/context tokens;
- flags for endpoint presence and bipartite side A/B.

Packing produces the 48-byte device AABB record; optional endpoints never cross
PCIe for the broad phase. Pointer identity is never serialized or compared.
The CPU seam maps returned IDs back to its copied records, runs any remaining
exact predicate, and publishes accepted records and finish events in original
order.

`--mode self` enumerates unordered pairs within each context. `--mode
bipartite` emits only A-to-B pairs, matching KLayout's two-input
`box_scanner2` shape/instance and instance/instance uses without paying for a
same-side superset.

Output is a sorted, duplicate-free vector of uint64 keys:
`min(record_id) << 32 | max(record_id)`. Multiple owner processes must keep an
owner/request namespace outside that local pair key.

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
int64 fixtures, a replay round trip, the million-record grid-oracle case, and
expected dense/overflow fallbacks. It also builds the optional backend DSO and
runs both a two-pair ABI smoke and a deterministic 1024-by-1024 exact CPU-oracle
gate through that ABI. Its optional first argument selects a build directory.

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
