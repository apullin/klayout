# Raw-M2 live union seam: host ABI scaffold

This implementation contains the fail-closed host ABI/loader scaffold, the
qualified compact raw-M2 scene serializer, and the production CUDA
run/release adapter.  It does not yet stitch a backend boundary into a live
flat region, expose a GSI method, or rewrite the deck.  Therefore it does not
complete or cross off the live M2 transaction.

## Status

- [x] Add a pointer-free compact raw-hierarchy request and canonical directed
  boundary result to `dbCudaSpatialApi.h`.
- [x] Require both the run and dedicated release symbols before advertising
  the optional capability.
- [x] Copy, release, and validate every backend-owned boundary before returning
  `CudaM2UnionAttempt::Complete`.
- [x] Pin segment layout, canonical order
  `(axis, side, fixed, lo, hi)`, maximal same-line intervals, and FNV-1a.
- [x] Exercise missing-symbol, valid, fallback, malformed echo/order/digest/
  count, and throwing host-copy paths against the real loader wrapper.
- [x] Implement and export the production CUDA union entry and release
  functions.
- [x] Build the qualified compact scene from raw live M2 before
  `merged_deep_layer()`.
- [ ] Stitch the validated boundary, run the complete stock flat M2 suffix, and
  return one atomic clean decision.
- [ ] Add `cuda_m2_rules_clean?` to the Region GSI.
- [ ] Add the atomic M2 owner/fallback rewrite to the deck generator and pass
  live CPU/CUDA differential gates.

## Smallest useful live transaction

The hook belongs at the beginning of the deck's M2 owner, before width, space,
`sized`, `edges`, or enclosure touches M2:

```text
raw deep M2 + raw VIA2
  |
  | capability gate (run AND release), before geometry lowering
  v
compact stored hierarchy, contexts, contours, and digest
  |
  | device-side hierarchy/rectangle expansion
  v
exact CUDA Manhattan union -> canonical directed boundary
  |
  | bounded host copy, dedicated release, proof validation
  v
checked endpoint stitch -> Shapes -> FlatRegion(shapes, true)
  |
  | set_merged_semantics(true), require merged_semantics && is_merged
  v
stock flat M2.1/.2/.4/.5-.9 clean checks
  |
  +-- every result empty: true
  `-- hit, decline, exception, topology mismatch, or malformed proof: false
       and execute the complete pristine CPU block
```

The existing compound width/space hook is too late for this purpose:
`region_cop_multi_with_properties_impl` obtains
`region->merged_deep_layer()` before it attempts the CUDA certificate.  A new
GSI transaction must inspect `DeepRegion::deep_layer()` only after
`cuda_spatial_m2_union_requested()` proves that the opt-in, DSO ABI, run
symbol, and release symbol are all present.

The atomic rule set is M2.1, M2.2, M2.4, and M2.5 through M2.9.  M2.3 remains
owned by the VIA1-stack transaction.  The deck rewrite must require that
ownership arrangement and must never suppress the original M2.3 expression
without either a successful VIA1 certificate or its local CPU fallback.

## Device work versus host work

Device-side hierarchy expansion is the qualified production route.  The
captured workload has 45,960 stored M2 polygons and 568,632 relevant contexts,
but expands to 22,946,444 rectangles.  Materializing that world stream on the
host would create a roughly 1.026-GiB payload and pay avoidable host work and
transfer.  The existing production loader already proves the compact
context/template representation and all eight orthogonal transforms.

The first live bridge should therefore:

1. reuse/refactor the raw hierarchy traversal from `dbCudaM1WidthSpace.cc`
   without asserting `inputs_are_merged=true`;
2. transfer compact contexts, cell spans, polygon templates, and contour
   edges;
3. expand and union them on the device; and
4. return only the exact canonical boundary.

The host performs the integrity-sensitive bridge: it validates the proof,
stitches directed segments into simple closed contours, constructs the
already-merged flat `Region`, and runs stock rule code.  This is an interim
boundary.  A later resident all-rule kernel can retain strips and morphology
on the GPU, but is not required to cash the already-proven union.

The offline exact flat bridge reconstructed 4,385,384 segments into 14,222
contours.  Its measured full host-certificate opportunity was about 22.077
seconds versus 67.612 seconds of replaced native merge payments, a 67.35%
reduction in that composed stage.  Those are offline feasibility numbers, not
a live end-to-end claim.

## Fail-closed contract

The request is accepted only for the complete qualified option mask, ABI and
record sizes, 2000 DBU/micron, nonempty bounded arrays, strictly increasing
M2-context IDs, checked byte products, and explicit capacity limits.  The
live builder additionally enforces the proven raw-scene rules: FreePDK45
physical M2 layer 13/0, same store/layout/top layer, no breakout, no
properties, supported orthogonal unit transforms, clockwise Manhattan
contours, structural spans, coordinate safety, census conservation, and a
canonical scene digest.  The live VIA2 operand is physical layer 14/0;
offline capture/remap slots 101/0 and 102/0 are not live operands.

The loader returns `Complete` only after all of these result checks:

- exact ABI size, opcode/options/profile/root/scene-digest echo;
- exact input censuses and zero fallback/device flags;
- complete disposition and a nonempty boundary;
- bounded, internally consistent rectangle/slab/membership/event/strip/raw-
  segment/canonical-segment counters, with independent raw and canonical
  capacities;
- a successful host copy;
- valid axis, side, and positive interval for every segment;
- strict `(axis, side, fixed, lo, hi)` order;
- no touching or overlapping interval on one `(axis, side, fixed)` line; and
- exact FNV-1a over the canonical stream.

The result buffer remains backend-owned until
`klayout_cuda_spatial_release_m2_union_boundary_v1`.  A local RAII guard calls
that dedicated release exactly once after every backend invocation, including
OK, fallback, malformed-result, backend-throw, and host-copy-throw paths.

After validation, stitching must still reject open endpoints, duplicate
incoming/outgoing endpoints, degree-four kissing points, short cycles,
coordinate narrowing, holes, changed vertex censuses, and any contour-count
mismatch.  Construct the region through:

```cpp
auto *flat = new db::FlatRegion(shapes, true);
flat->set_merged_semantics(true);
db::Region merged_m2(flat);
```

Do not use `Region(Shapes, true, true)`: its incremental insertion currently
clears the requested merged flag.  Require both
`merged_m2.merged_semantics()` and `merged_m2.is_merged()` before running the
stock suffix.

M2.4 needs special operand handling.  Do not silently mix the new flat M2
delegate with a deep VIA2 delegate.  Copy/flatten the qualified VIA2 operand
into a separate temporary flat region, or consume a separately proven
resident projection-strip certificate.  The original deep M2 and VIA2
operands must remain untouched so every decline can execute the historical
CPU expressions.

## Implementation map

The scaffold in this commit touches:

- `src/db/db/dbCudaSpatialApi.h`: additive raw-M2 request/result/release ABI;
- `src/db/db/dbCudaSpatialBackend.{h,cc}`: optional-symbol discovery,
  capability gate, result ownership, and proof validation;
- `src/db/unit_tests/dbCudaM2UnionContractTests.cc`: layout/order/digest
  validator tests; and
- `benchmarks/cuda_spatial_replay/m2_union_*`: adversarial DSO/loader gate.

The smallest subsequent live implementation should add:

- `dbCudaM2Rules.{h,cc}` for scene ownership, boundary stitching, flat stock
  checks, and the one boolean transaction;
- one guarded `cuda_m2_rules_clean?` binding in `gsiDeclDbRegion.cc`; and
- an atomic, exact-count rewrite in `make_via1_stack_live_deck.py`.

No generic `DeepRegion` merged cache should be mutated by the first version.
The validated flat region is transaction-local and the pristine deep inputs
remain the sole fallback operands.

The production adapter is
`benchmarks/cuda_spatial_replay/m2_union_backend.cu`.  Before its first CUDA
allocation it independently validates record sizes and spans, root and
context order, stored and flat censuses, exact world bounds, coordinate
safety, and the complete `KM2RAW01` digest.  It accepts only checked
four-edge boxes and six-edge three-cell L contours, conserves exact area,
uploads compact local templates, expands all eight transforms on the device,
and moves the 22,946,444-record resident buffer into the shared union core.
The 1.026-GiB world stream is never materialized on the host.

## Contract gate

Using an existing qmake KLayout build for the loader's normal digest/log
dependencies:

```sh
cmake -S benchmarks/cuda_spatial_replay \
  -B /tmp/klayout-m2-live-seam-contract-build \
  -DKLAYOUT_QMAKE_BUILD_DIR=/tmp/klayout-m1ws-live-build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /tmp/klayout-m2-live-seam-contract-build \
  --target m2_union_backend_contract_gate -j 8
```

Each mode runs in a fresh process because the loader intentionally snapshots
environment and DSO symbols on first use.  The gate requires exactly zero
backend calls for incomplete capabilities and exactly one release for every
invoked result path.

## Production backend gate

The DSO-level small gate covers all eight transforms against an independent
integer-cell boundary oracle, eleven malformed/capacity cases, and idempotent
release.  The full gate reconstructs the compact raw ABI from the pinned
KACT capture, recomputes its independent `KM2RAW01` identity, and compares
all 4,385,384 returned segments with the pinned stock KLayout merged-boundary
oracle:

```sh
/tmp/klayout_cuda_workbench.sh \
  m2-union-production-backend-v1 production
```

Evidence from commit `e3ad0ee` plus the production-gate follow-up is in
`m2-union-production-backend.RGYZyn`.  Three exact calls produced:

- 587,201 contexts, 568,632 nonempty M2 contexts, 143 cells;
- 45,960 stored polygons and 183,852 stored edges;
- 22,945,976 flat polygons and 91,784,840 flat edges;
- 22,946,444 rectangles, 92,386,704 memberships, 184,773,408 events;
- 46,383 x slabs, 3,691,466 strip intervals, 9,575,624 raw segments;
- 4,385,384 canonical segments and FNV64
  `7541395996791771514`; and
- raw scene SHA-256
  `66ec73eaf686c6f630eb91e63949b70301ca2f24726ee4033dd77a5f8908c1c3`.

Warm observed adapter calls were 1,517.197 and 1,517.459 ms (median
1,517.328 ms).  Their charged internal components were approximately
691 ms independent host validation/lowering, 5–6 ms compact upload,
2.5 ms device rectangle expansion, 21.1 ms x membership, 138.7 ms strip
scan, 9.5 ms boundary work, and 104.3 ms core D2H.  Remaining charged time is
the core's other exact sort/scan/allocation work plus conversion into the
backend-owned ABI buffer.  The 2.162-second KACT reconstruction and
5.917-second heavyweight oracle load are qualification harness costs outside
the backend call.  These are production-corpus DSO/replay measurements, not a
live KLayout transaction or full-signoff runtime.
