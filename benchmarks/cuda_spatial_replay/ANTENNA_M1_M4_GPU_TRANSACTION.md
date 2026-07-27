# Staged M1-M4 antenna CUDA transaction

## Scope and roofline

The qualified FreePDK45 independent-x2 launch currently has four metal
antenna owners that start together:

| Owner | Mean wall |
|---|---:|
| `antenna_m1` | 32.636 s |
| `antenna_m2` | 38.824 s |
| `antenna_m3` | 40.634 s |
| `antenna_m4_m10` | 40.986 s |

Each owner independently constructs the same GATE, factor-zero DIODE, and
lower-metal connectivity prefix.  M5-M10 are empty on this workload, so the
last owner effectively pays only for M4.

The target is one atomic, clean-only M1-through-M4 transaction.  Collapsing
the four owners into one reduces the production plan from 14 shards to 11,
which also lets every remaining owner start in the first wave.  If the fused
owner finishes below the current 33.9-second GRID owner, the projected
full-launch opportunity is about 11 real seconds, or 22% of the current
49.93-second wall.

## Exact inputs

The transaction consumes one compact shared hierarchy and twelve distinct
raw physical domains:

1. POLY
2. ACTIVE
3. NPLUS
4. NWELL
5. CONTACT
6. METAL1
7. VIA1
8. METAL2
9. VIA2
10. METAL3
11. VIA3
12. METAL4

The existing KANTM102 six-domain capture is the compatibility baseline.
The twelve-domain format must preserve one source-cell table, one context
stream, one parent stream, per-domain source-cell geometry ranges, exact
orthogonal transforms, and independently bound domain/capture digests.
Naively flattening the input on the host is not admissible.

The production hierarchy census validates that choice.  One compact capture
completed in 20.765 seconds and occupied 227,270,196 stored bytes, while a
naive twelve-domain expansion would contain 141,611,088 polygons,
566,471,360 edges, and 25,490,860,096 geometry bytes.  The shared hierarchy
contains 273 source cells and 849,265 occurrence contexts.  In particular,
the M1 graph has about 67.65 million polygon owners and VIA1/M2 adds another
43.12 million, so the device implementation must stream stage frontiers
rather than retain every expanded domain at once.

## Staged connectivity

Closed-set polygon touch or overlap is electrical connectivity.  Every
stage first settles all newly enabled self and cross-layer relations, then
computes that stage's antenna result before adding the next via/metal pair.

| Checkpoint | Newly enabled graph relations |
|---|---|
| M1 | POLY self, CONTACT self, M1 self, POLY-CONTACT, CONTACT-M1 |
| M2 | VIA1 self, M2 self, M1-VIA1, VIA1-M2 |
| M3 | VIA2 self, M3 self, M2-VIA2, VIA2-M3 |
| M4 | VIA3 self, M4 self, M3-VIA3, VIA3-M4 |

Earlier checks cannot be evaluated from the final M4 partition: an upper
metal bridge can merge roots that were separate at an earlier checkpoint.
The device implementation therefore uses one persistent disjoint set with
four explicit settle-and-reduce barriers.

Component labels are canonicalized to the minimum stable node identity.
Candidate duplication across spatial bins must not change labels, edge
census, or results.

## Rule semantics

GATE is the exact positive-area region:

```text
POLY intersect ACTIVE
```

DIODE is:

```text
NPLUS intersect (ACTIVE subtract NWELL)
```

For FreePDK45 the diode factor is zero.  A component with positive DIODE area
is exempt.  DIODE is an annotation: touching more than one conductor
component marks each component exempt but does not join those components.

For each non-exempt component with GATE area greater than one DBU squared,
checkpoint `k` is clean only when:

```text
merged_area(METALk) / merged_area(GATE) <= 300 + epsilon
```

Area accumulation must be exact union area, not raw polygon-area summation.
The host CPU oracle remains the semantic reference for focused fixtures.

The first production backend may use a stricter one-sided certificate instead
of reconstructing both exact unions:

- sum the exact positive areas of every raw target-metal polygon owned by a
  conductor root; overlap can only make this an upper bound on merged metal
  area;
- enumerate complete positive-area POLY/ACTIVE rectangle intersections and
  retain the largest single intersection owned by each conductor root; this is
  a lower bound on merged GATE area;
- certify a root only with checked integer arithmetic when
  `metal_upper <= 300 * gate_lower`.

The backend must decline the whole transaction if an eligible gate cannot be
bounded, if a polygon cannot be exactly rectangulated, or if any count,
coordinate, multiplication, or reduction overflows.  Diode exemptions may be
ignored by this certificate because doing so is stricter.  These bounds can
prove a clean result but can never report a violation or replace the CPU path
after an uncertain result.

## Memory plan

The production implementation streams adjacent conductor layers:

1. retain the disjoint-set labels and component annotations;
2. retain geometry only for the current metal and the next via/metal join;
3. release older geometry immediately after the cross-layer relation is
   settled;
4. evaluate and compact component annotations at every checkpoint.

This bounds geometry residency by the largest adjacent stage rather than the
sum of all twelve flattened domains.  Device admission is explicit and
fail-closed.  The production launcher must also prevent this transaction
from overlapping another high-residency backend call on the 10-GiB device.

The qualification target is the local RTX 3080 with 10,240 MiB of device
memory.  A 16-GiB card may diagnose a rejected allocation frontier, but it
does not qualify the production path.  The default transaction budget must
leave explicit CUDA/runtime headroom, account for caller-owned staging and
sort/reduction scratch as well as steady-state arrays, and report its
conservative peak in the result.  An unaccounted temporary, a peak above the
request budget, or insufficient free memory declines the entire transaction
to the literal CPU path.

This qualification gate does not block the first end-to-end watershed.  A
high-memory GPU (including a rented 96-GiB RTX Pro 6000) may run a simpler
fully expanded implementation first to establish correctness, real wall time,
and measured allocation frontiers.  Such a result is labeled a high-memory
proof rather than a production qualification; the measured census then guides
the later streaming rewrite needed to check off the 10-GiB gate.

## Atomic result contract

The backend echoes every request identity, digest, capacity, count, and
qualified option.  It returns:

- a four-bit certified-empty mask;
- per-stage component, candidate, edge, gate, exempt, and evaluated counts;
- exact stage digests;
- device flags and bounded timing counters.

The host bypasses literal CPU rules only when all four selected bits are
certified, every echo and digest validates, all counters are internally
consistent, and no hit, uncertainty, capacity, ABI, or device flag is set.
Any other outcome runs the unchanged CPU chain for every owned antenna rule.

## Qualification

Before production timing is booked:

1. directed and seeded randomized CUDA/CPU partition differentials pass;
2. hierarchy, array, transform, cross-context, point-touch, bridge, area,
   diode, one-DBU boundary, and staged-merge fixtures pass;
3. malformed, capacity, missing-symbol, backend-error, and injected-error
   paths leave outputs unchanged and select CPU fallback;
4. clean and deliberately nonempty complete DRC reports match the CPU
   reference;
5. a memory-instrumented production screen passes without another owner's
   GPU fallback;
6. repeated matched full-launch trials demonstrate real critical-path wall
   reduction.

## Retained worklist

- [x] Reconstruct the qualified 49.93-second critical path and repeated
  antenna work census.
- [x] Recover, port, and test the deterministic six-domain CPU conductor
  oracle.
- [x] Specify the atomic staged transaction, semantics, memory plan, and
  fail-closed result contract.
- [x] Add the exact `antenna_m1_m4` CPU owner and balanced-launch scheduling
  mode without changing the default deck.
- [x] Complete and test the compact shared-hierarchy twelve-domain capture.
- [x] Complete the reusable staged CUDA connectivity/DSU core and randomized
  CPU differential.
- [x] Complete the conservative device-resident gate/metal clean certificate,
  bounded spatial census, and randomized CPU differential.
- [ ] Extend the reference oracle through M4, including stage-local area and
  diode annotations.
- [x] Add the production ABI, loader, exact result validation, and GSI method.
- [x] Add the atomic deck fast path; retain all ten literal CPU rules in one
  fallback branch.
- [ ] Pass focused hierarchy, transform, boundary, malformed-result, and
  injected-fallback gates.
- [ ] Pass the clean production report and nonempty hierarchical fixture
  integrity gates.
- [ ] Package the exact compact production capture and standalone replay for
  the high-memory remote watershed.
- [x] Capture and hash the exact FreePDK45 independent-x2 production request;
  retain its canonical 18.7-MiB compressed replay artifact.
- [x] Replace both production gate-census global counter hot spots—ACTIVE
  memberships and POLY query visits—with exact block-reduced admission
  passes.
- [x] Retune the production certificate grid from 125 to 250 DBU using an
  exact grid/membership/query/candidate sweep.
- [x] Filter exact touching pairs before occurrence materialization and
  traverse each occupied cell cooperatively on the GPU.
- [x] Implement memory-bounded logical metal stages as domain sub-appends,
  removing the duplicated assembled geometry frontier.
- [x] Compact exact connectivity memberships from 24 to 16 bytes with a
  lossless signed-64-bit cell key and block-reduced bounds.
- [x] Emit exact rectangle pairs only from their canonical shared cell,
  removing multi-cell duplicates and the exact-path rectangle-key
  sort/unique pass.
- [x] Stream exact owner connectivity with root-only CAS unions and compact
  same-owner/multi-rectangle exception streams, eliminating the
  rectangle-cardinality tile and owner fan-out arrays.
- [x] Compose exact factor-zero DIODE witnesses with a disjoint root-cell
  GATE lower-bound refinement.  On the independent-x2 production replay this
  closes the M1 frontier from 1,566 uncertain roots to zero; the first exact
  integrated replay reached M2.
- [x] Prune staged exact connectivity work before enumeration: never rescan
  old-old rectangle pairs, and generate only new/new or new/allowed-neighbor
  pairs within each cell.  This exact optimization passed 22 directed and 96
  seeded staged differentials.  The production replay showed that it is a
  qualified structural win rather than the expected wall-time win: M2 still
  contains 1,637,538,138 genuine exact owner edges and remained the roof.
- [x] Compact the retained rectangle allocation after the M2 append.  Only
  22,946,444 M2 rectangles remain, while the reserved 84,223,436-rectangle
  frontier strands about 2.28 GiB and prevents the M2 certificate from
  entering under the 9-GiB transaction cap.  The exact integrated replay now
  completes all four checkpoints under the cap.
- [x] Fuse the overwhelmingly singleton M2 edge stream into one exact pass.
  VIA1 has one rectangle for every one of its 20,178,022 owners, and M2 has
  only 468 more rectangles than its 22,945,976 owners.  Union and census
  singleton-singleton edges during the count pass, then restrict the fill
  pass to pairs involving a multi-rectangle owner.  Rejected as a production
  speed optimization: the exact isolated replay was 333.32 seconds versus
  the prior 323.64-second record, 9.68 seconds or 3.0% slower.  It did reduce
  peak device memory from 8,595 to 7,769 MiB, but M2 remained 324.50 seconds.
- [x] Collapse identical single-rectangle conductor geometry into an exact
  weighted quotient before spatial enumeration.  The authenticated
  production census projects 61.1% fewer M1 rectangles, 90.7% fewer VIA1
  rectangles, and 74.2% fewer M2 rectangles while preserving the exact
  weighted internal pair totals and all multi-tile exceptions.  The exact
  device seam and production predictor are banked on
  `fork/antenna-geometry-quotient`; complete phase telemetry showed that
  VIA1 plus M2 connectivity append is only 1.487 seconds, below 0.5% of the
  320.96-second run, so active integration is deferred.
- [x] Replace the comparison sort of 145–173 million 16-byte `CellMember`
  records with a stable radix sort of 64-bit cell keys and 32-bit rectangle
  nodes.  Membership fill is initially ordered by rectangle node, so sort
  stability preserves the exact old/new split required by staged pruning;
  a 50-million-record exact microbenchmark was bit-for-bit identical and
  reduced sort time from 61.71 to 21.58 ms (65.0%).  Because the complete
  connectivity append is only 1.487 seconds, production integration is
  deferred until the certificate roof is removed.
- [x] Reuse hierarchy-local conductor forests.  Precompute the exact
  same-context connectivity forest once for each of the 273 source cells,
  instantiate those forest edges across 849,265 occurrence contexts, and
  leave cross-context touching pairs to the global grid.  Prove transform,
  array, and boundary-touch equivalence before measuring the removed spatial
  work.  Deferred before implementation because complete connectivity now
  measures below 0.5% of whole-run wall.
- [x] Enumerate only canonical-start membership buckets.  A count-only
  production projection removes 682.7 million M2 pair tests (21.1%) by
  requiring a pair's cell to contain its maximum left and bottom coordinate;
  exact focused implementation, oracle, memcheck, and racecheck gates are
  banked at `fork/canonical-bucket-frontier`.  Production integration is
  deferred because the eliminated enumeration kernels are no longer on the
  critical path.
- [ ] Reuse the verified preliminary checkpoint census when root-cell
  refinement begins instead of rerunning the complete preliminary
  certificate.  The exact M2 phase trace measured the duplicated pass at
  102.98 seconds; the reuse path must retain unchanged-output failure and
  identity/counter validation.
- [ ] Retain the preliminary device root annotations through root-cell
  refinement, eliminating the third gate/metal/root reduction while
  preserving the 10-GiB transaction cap.
- [ ] Remove M2 root-reduction contention.  Complete phase telemetry assigns
  102.98 seconds to the first certificate evaluation; instrument and replace
  the contended per-tile/per-owner atomics with exact sorted or segmented
  reductions.
- [ ] Optimize the true root-cell refinement remainder: ACTIVE grid build,
  uncertain POLY/ACTIVE record generation, CUB key/area sort, and equal-key
  maximum-area accumulation.  Measure each phase independently before
  selecting the next kernel.
- [x] Demonstrate an accounted peak within the 10,240-MiB qualification card,
  including caller staging and temporary sort/reduction storage.
- [ ] Run matched repeated full-launch control/candidate trials and book only
  the measured real-seconds and percent reduction.
- [ ] Re-profile the new roof and select the next largest contiguous target.
