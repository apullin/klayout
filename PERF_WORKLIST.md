# KLayout performance worklist

This is the durable queue for the `perf` branch.  Update it whenever an item
is measured, accepted, rejected, or re-ranked.

Never delete an item when its status changes.  Move or annotate it so the
history survives context rollover.  `[ ]` means open and `[x]` means closed;
the section and wording record whether a closed item was completed, abandoned,
bounded, or judged not worth doing.

## Current baseline and acceptance gate

- Workload: FreePDK45 `sram_1rw0r0w_64_1024_freepdk45_aref.gds`
- Binary: clang + full LTO, jemalloc, batch mode
- Legacy pre-gate serial wall time: 128.68 s, 128.98 s; mean 128.83 s
- Process-sharded wall time: 69.78 s, 70.43 s, and 69.689 s through the
  packaged launcher; three-run mean 69.966 s
- Legacy exact-output gain: **+84.1% throughput** versus the serial
  mean, with 1.06% full-range spread.  The packaged run alone is +84.9%.
  These measurements predate the three-independent-observation and runtime-
  identity gate below.  They remain useful engineering evidence, but are not
  automatically eligible as an identity-v3 controlled comparison.
- Baseline raw report SHA-256:
  `e9340a8d3cf3608cff5f87fe5c82aa9b122c215627dc9159a3d59f98bcaaedd0`
- Path-normalized baseline and batched report SHA-256:
  `74e8e6bb6d46e8574bcb8cdf159fee32f52dd49c0f6997db44b2645648357ba9`
- Deterministic merged clean-report SHA-256:
  `c699e50f16d9f460d686191fcbfa6986ededbff96f416ebfd8d55cfdff8bf6b1`
- Shard-aware deck SHA-256:
  `5b05b599f5af878364365136248177b63098e92ff18287d097274f54500ee2ce`
- Deck-bound shard manifest SHA-256:
  `2f0e419ad63654b5cd6987b81662ed0b44b671f3f353f7a2a3dff4f7ef6949a3`
  (`decks/freepdk45-shards-bound.json` in the external corpus)
- Deterministic merged 99-marker fixture SHA-256:
  `9b55135453bbdd9a9a0c28b583bee73d89af20cd499978c052fa991b8d1919ef`
- Iteration target: 3–5 minutes; retain only exact, repeatable wins

## Benchmark-integrity gate

- [x] **Runtime-provenance enforcement and concurrent replication — completed:**
  record the resolved executable and dynamic KLayout bundle hashes, deck/input/
  manifest/report hashes, full child commands, allocator and thread environment,
  host, per-shard timing, and RSS.  Repeat comparisons must match every identity
  field; experiments must explicitly declare the dimensions allowed to differ;
  historical comparisons remain contextual and cannot become accepted results.
  Support 3–5 concurrent repetitions with optional disjoint CPU sets, but label
  their aggregate designs/hour separately from isolated single-job latency.
  The shared fail-closed collector now fingerprints the executable, actual ELF
  loader closure, KLayout plugins and their dependency closure, ordered preloads,
  4,869 loadable Ruby/Python files through compact tree manifests, and code-
  bearing runtime controls twice before a run and again afterward.  It
  rejects unresolved loader state, `LD_AUDIT`, uncovered plugin search paths,
  nonempty implicit KLayout homes, and in-place mutations.  The benchmark uses
  fresh per-sample homes, exclusive output locking, atomic summaries, exact
  suite/argv/artifact identities, three-observation qualification, direct-process
  RSS and average CPU, and distinct isolated-latency versus concurrent-batch
  throughput semantics.  Eighty-three focused tests pass (30 benchmark, 34
  launcher/merger, 19 runtime identity).  Real nonempty sentinels produced the
  exact 99 FreePDK45 and 25 Sky130 items under comparison-identity v3; a real
  two-shard Sky130 launcher smoke also produced all 25 items while recording 113
  selected dependencies, 27 plugins, per-shard timing/RSS, and all three
  orchestrator scripts.  One concurrent batch remains exploratory; acceptance
  requires at least three independent batches.

## Benchmark ladder

The accepted FreePDK45 SRAM remains the historical comparison point, but it is
no longer sufficient by itself.  Use real Sky130 logic to prevent SRAM-,
hierarchy-, or deck-specific tuning, and keep different sizes for different
questions:

- **Smoke/correctness (22–57 s):** hierarchical and flat FP4, DME1, and FP16.
  These are fast enough for launcher, manifest, report-merger, and deliberately
  nonempty-fixture tests.  They are not long enough to accept small performance
  claims.
- **Iteration/acceptance candidate (about 4 minutes):** native hierarchical
  `hg0_s3_asic` (54,744,844 bytes, 155 cells).  Pre-gate no-preload
  current-binary serial measurements are 223.94 and 224.97 s wall (224.455 s
  mean; 0.46% full-range spread).  They need one complete identity-v3
  three-observation cohort before serving as a new controlled baseline.  The
  older 291.561 s signoff record came from
  LibreLane 3.0.4 invoking a different, unrecorded KLayout executable; it used
  the same GDS and underlying deck content, but is historical context only.
  The apparent roughly **+30% throughput** from 291.561 to 224.455 s is
  therefore a cumulative environment/build/code comparison, not statistical
  variation or a controlled performance result.  Source GDS SHA-256:
  `a5aae78efed5a76e5f2c6c03762f4fc60d70132931028bb47f941f08d3184de7`.
  Current serial report SHA-256:
  `8fad6c51f6e96b2e1c79ab98446f8ad536527d33cc974096e41e71689ccb730f`.
- **Heavy qualification:** `vmu_top_asic` (63 MB, historical 403.048 s) and
  FP64 (85 MB, historical 538.245 s).  Run after an optimization passes both
  smoke and the four-minute lane.
- **Pre-overnight stress:** the 139 MB `vmu_v1_top` (historical 1157.014 s,
  4 GiB peak RSS).  This is a release gate, not an iteration benchmark.

Every promoted lane needs a pinned input/deck hash, current-binary serial
baseline, exact category/item/cell payload comparison, and at least one
deliberately nonempty hierarchical fixture.  Record average CPU and peak RSS,
not just wall time.

## Ranked work

1. [x] **Cross-PDK sharding on the Sky130 ladder — completed**
   Make the Sky130 deck shard-aware, use FP4/FP16/DME1 to qualify report
   semantics quickly, then time `hg0_s3_asic`.  This is the first gate because
   it answers whether the +84.1% FreePDK result was a useful general mechanism
   or another one-benchmark optimization.  Preserve all FEOL, BEOL, off-grid,
   seal, and floating-metal option semantics.  The pre-gate no-preload
   controlled result is 224.455 s serial mean versus 126.463 s two-shard mean
   on `hg0_s3_asic`:
   **+77.5% throughput** and 43.7% less wall time.  Serial full-range spread is
   0.46%; sharded spread is 1.26%.  All four reports have exact category, cell,
   and item equality; both serial reports share SHA-256
   `8fad6c51f6e96b2e1c79ab98446f8ad536527d33cc974096e41e71689ccb730f`,
   and all sharded merged reports share SHA-256
   `ae4d2bd9feabf689042d9208e77d8c9a1d4a07c1ee4cc9144e666cc4a1081f9f`.
   A hierarchical nonempty fixture also proved the exact 25-item union and
   deterministic merge.  Durable corpus artifacts are
   `decks/sky130A_mr-sharded.drc` (SHA-256
   `9f8c1cffe597c69cd217c3c8c70cb9ece749e40b4170b10f45fd6dc2b7ddbec9`),
   `decks/sky130A-shards-bound.json` (SHA-256
   `597c872aca92b310c6ffca8aff0993542129c89dce05fd038192eb6dd8797cc3`),
   and `inputs/generated/sky130_shard_violating.gds` (SHA-256
   `d357ccb312ce5c4a6964475c127765ea031ad1148b37c28ffd75d1ec7ff6c236`).
   One explicit-jemalloc sharded run took 108.527 s,
   suggesting another **+16.5% throughput** over the no-preload sharded mean,
   but that is a separate exploratory configuration pending serial and repeated
   measurements.  Never mix the 291.561 s historical record into either
   controlled comparison.  The exact result is not invalidated, but its two
   serial and two sharded observations predate identity v3 and do not satisfy
   the new three-observation promotion rule.
2. [ ] **Measured 3/4-way process scheduling and automatic balancing**
   The launcher already accepts arbitrary shards and bounded jobs.  Profile
   independent category-producing chains and use longest-processing-time-first
   bin packing rather than hand balancing.  Current FreePDK operation timings
   show a roughly 140 s work sum and a 27.7 s largest indivisible chain, giving
   a four-way lower bound near 35 s; 35–45 s is a realistic first target versus
   69.966 s now, or another **+55% to +100% throughput** before contention.
   First run a cheap four-process duplicate-shard contention probe and stop if
   per-job latency regresses more than 10–15%.
3. [ ] **Thread/core-budget sweep and affinity**
   Sweep 1/2/4 inner threads per shard and restrict the complete launch to
   2/4/8 physical CPUs.  Today, two processes request eight worker threads but
   average only about 2.06 CPU cores total, so the +84.1% result probably does
   not require a 32-core machine.  Establish the minimum practical core budget,
   then test disjoint physical-core/CCD pinning only after four-way sharding.
   Always bound outer jobs times inner threads and record RSS.
4. [ ] **Native bounded rule-DAG executor**
   Ruby declares dependencies and output order serially; only C++ terminal
   geometry tasks run in parallel, with private results and serial publication
   by deck ordinal.  Arbitrary Ruby blocks, mutable report databases, layouts,
   and `DeepShapeStore` outputs must never execute concurrently.  This removes
   process supervision and may reduce duplicated setup without weakening the
   isolation proof.
5. [ ] **Fuse enclosure producer/consumer chains**
   `metal1.enclosing(cont)` costs about 22.3 s and materializes roughly 2.6
   million edge pairs before `.second_edges.width(...)` spends another 5.5 s.
   Stream/filter candidates before materializing the intermediate, retaining
   exact predicates and hierarchy reduction.  The 27.8 s ceiling suggests a
   realistic **+4% to +13%** whole-run opportunity after rebalancing.  Require
   a deliberately nonempty, transformed hierarchical fixture.
6. [ ] **Rectangular exact-size contact/via fast path**
   Exact-length edge filters cost about 5.1 s for contacts and 3.5 s for via1.
   For proven Manhattan boxes, compare transformed dimensions directly and
   retain the generic polygon fallback.  Establish the ceiling in an isolated
   microbenchmark before changing engine code.
7. [ ] **Multi-layer off-grid traversal**
   Roughly 28 independent `ongrid` calls total about 6.8 s.  Test an API that
   traverses each hierarchical cell once while dispatching violations to
   separate per-layer report categories.  Confirm the same shape on Sky130 and
   GF180 before implementation.
8. [ ] **Target-specific code generation and cross-PDK PGO**
   Try a separate `znver2`-tuned build, then profile-guided optimization trained
   on FreePDK45 plus Sky130/GF180.  Plausible ranges are **+2% to +8%** for
   target tuning and **+3% to +8%** for PGO, but accept only exact reports on the
   full ladder and keep the portable build supported.  Consider BOLT only if
   counters still show front-end or instruction-cache pressure.
9. [ ] **Generalized exact early pruning**
   Inventory operations whose required target or interaction layer is empty
   and prove that skipping extraction cannot change report categories or
   diagnostics.  Empty-target antenna pruning was valuable; use profiling to
   avoid scattering checks with no measurable ceiling.
10. [ ] **Persistent workers for batches of small peripherals**
    Startup is negligible on the long lanes but material on many tiny blocks.
    Reusing initialized processes across a batch could improve throughput by
    30–50%; aggressively verify layout, report, Ruby, and hierarchy-cache reset
    between designs.  This optimizes developer throughput, not one large DRC.
11. [ ] **Intra-cell parallelism for remaining deep checks**
    Subdivide the few large serial geometry operations that limit ordinary
    operation-level thread scaling.  Re-profile after N-way rule scheduling so
    work is not spent below the new critical path.
12. [ ] **Lazy interpreter initialization**
    Reduce the roughly 1.4 s batch startup floor for short jobs; separate from
    long-run geometry work.
13. [ ] **Incremental antenna edge replay — re-profile first**
    The old estimate assumed ten cumulative rebuilds.  Empty-target pruning
    removed six, so its present ceiling is much smaller and must be measured
    before implementation.
14. [ ] **GPU broad-phase feasibility gate — later, orthogonal project**
    Do not translate the Ruby PDK deck to CUDA.  The credible first kernel is
    spatial bin/sort plus candidate filtering for the millions of M1 enclosure
    edge pairs, with exact predicates and hierarchy reduction retained on CPU.
    Instrument serialized bytes, candidate-reduction ratio, transfer time, and
    CPU fallback first; proceed only if transfer plus kernel time is comfortably
    below the roughly 22 s producer ceiling.  A realistic whole-run opportunity
    is 5–15 s, but CPU rule scheduling has much better evidence and lower risk.

## Measured lower-priority paths

- [ ] **Shared/forked layout setup — deferred:** add phase timers before
  designing it.  The visible common derived-layer prelude is only about 2.7 s,
  page cache already
  removes much raw I/O, and deep inputs are lazy.  A Linux preload/fork proof is
  allowed only before Ruby/Qt worker threads; allocator/fork safety and COW
  dirtying make it an experiment, not an architecture.
- [ ] **Immutable deep-store sharing — deferred:** a real shapes-content
  revision token, frozen-input contract, and per-worker derived/variant/output
  overlays are
  prerequisites.  The existing hierarchy generation ID does not track content,
  and `DeepShapeStore` is deliberately mutable and non-copyable.
- [x] **Report merge optimization — closed as not worth doing:** current clean
  merge overhead is about 4 ms and has no useful ceiling.

## Recently completed

- [x] Empty-target antenna pruning: 161.38 s to 146.50 s, **+10.2% throughput**,
  byte-identical report; commit `0873912` on `perf`.
- [x] Homogeneous multi-output compound DRC: batch the matching M1 and M2
  width/spacing checks into one deep traversal while preserving independent
  report categories.  146.50 s to 128.83 s mean, **+13.7% throughput**,
  path-normalized byte-identical report.  Direct M1 batch: 17.15–17.24 s;
  direct M2 batch: 3.95–4.04 s.
- [x] Isolated process-level rule-DAG scheduling: a balanced `ACTIVE` + `METAL1`
  shard and an all-other-rules shard reduce 128.83 s to a 69.966 s mean,
  **+84.1% throughput**.  `scripts/run_parallel_drc.py` bounds and supervises
  child processes; `scripts/merge_sharded_lyrdb.py` bootstraps a strict
  category-order/ownership manifest and publishes an atomic deterministic
  report.  The clean workload has all 157 categories, and a hierarchical
  violation fixture has the exact 99-marker union (15 + 84), including full
  item payloads and cell references.  KLayout natively loads the merged output;
  merge overhead is 4 ms.  The manifest is bound to the shard-aware deck hash,
  so a mismatched deck fails before expensive children launch.  Thirty-one
  focused launcher/merger tests cover success, failure, SIGTERM cleanup,
  atomic publication, permissions, nested categories, cell variants/reference
  order, cross-layout reuse, determinism, and nonempty payload preservation.

## Rejected or bounded prototypes

- [x] Persistent hierarchy-context caching: context formation was only about
  12–14 s for the entire deck and the contexts contain mutable propagated
  results without a reliable `Shapes` revision token.  Do not revive this as
  a cross-rule cache without a real invalidation design.
- [x] Disparate-rule batching: the aggregate uses the maximum interaction border
  and union of all external inputs.  Batch only rules with similar distance,
  metrics and inputs; unrelated rules can increase preprocessing work.
- [x] M1 enclosure batching: even the apparently ideal pair
  `enclosing(cont, 35nm, projection)` + `enclosing(via1, 35nm, projection)`
  regressed.  The pair took 31.66 s batched versus 30.67 s separately, and the
  full run took 130.24 s versus the 128.83 s accepted mean (about 1.1% slower).
  The report remained exact.  The unioned secondary interactions outweighed
  any shared traversal, so do not revive this pair.
- [x] Mixed hierarchy reducers: sequential reducer composition can under-specify
  variants.  The batch API rejects incompatible reducer combinations rather
  than risking incorrect rotated or magnified hierarchy reuse.

## Guardrails

- Do not disable or weaken rules.
- Preserve report geometry, categories, descriptions, and diagnostic outputs.
- Test both flat and deep modes when engine semantics are affected.
- Record rejected prototypes so they are not rediscovered after context rolls.
