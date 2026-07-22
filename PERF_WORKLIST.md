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
  throughput semantics.  Eighty-seven focused tests pass (31 benchmark, 36
  launcher/merger, 20 runtime identity).  Fresh installs now use one normalized
  environment for both provenance snapshots, every child, and metadata; it
  suppresses first-run Python bytecode writes without excluding `.pyc` from the
  fail-closed runtime identity.  A cache-free real runtime stayed cache-free and
  produced the exact 99-item sentinel (commit `4843f23`).  Real nonempty
  sentinels produced the
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
- [x] **Recognizable non-SRAM head-check baseline — completed (about 3–5
  minutes):** Sky130 HG0 S5, a custom RV32 core with 35,938 standard cells in
  a 788.23 by 798.95 um die.  Its clean final GDS is 54,154,630 bytes, top
  `hg0_s5_asic`, SHA-256
  `03b62132c3f85b66664101fa90faacee6a952cc196cc246633e1c9e298c45911`.
  Three sequential identity-v3 observations are 199.803983, 199.567681, and
  200.649140 s: **199.803983 s median (3m19.804s)**, 200.006935 s mean, and
  0.541% full-range spread.  Mean CPU use is 1.1574 cores and maximum
  direct-process RSS is 1,104,268 KiB.  All reports are raw-byte identical at
  256 categories, one cell, zero items, raw SHA-256
  `87613d325082e7bb52473a24043d8c681aed46b68890d6367092039b3d6df2fd`,
  and normalized SHA-256
  `2c9f660d7b2d7186329c510333083bfe19ab17779fe42d66c936feac0b45fdb4`.
  The nonempty Sky130 sentinel also passed three times at 256 categories, two
  cells, and 25 items.  Runner support is committed at `00a82e3`/`bcf4d1a`;
  durable evidence is under
  `evidence/sky130-hg0-s5-identity-v3-20260721/` in the external corpus.

  The retained historical run is 282.433 s for the direct KLayout process and
  282.843 s for the whole LibreLane step.  Against the direct-process record,
  the controlled current median is **29.3% less wall time** and **+41.4%
  throughput**, but that remains cumulative environment/build/code context,
  not single-patch attribution.  Use this lane before promoting engine-level
  wins as broadly useful.  A FreePDK-only owner/deck scheduling change may
  remain PDK-specific, but must be labeled that way rather than borrowing this
  lane's generality.
- **Heavy qualification:** `vmu_top_asic` (63 MB, historical 403.048 s) and
  FP64 (85 MB, historical 538.245 s).  Run after an optimization passes both
  smoke and the four-minute lane.
- **Pre-overnight stress:** the 139 MB `vmu_v1_top` (historical 1157.014 s,
  4 GiB peak RSS).  This is a release gate, not an iteration benchmark.
- **FreePDK45 scale/capstone lane:** the 512-Kbit two-independent-tree SRAM
  takes 46m09.49s on a fresh upstream-master stock build and 2m55.035s for the
  current best eight-owner DRC plus strict merge.  Use it to demonstrate and
  re-profile cumulative scaling, not for routine iteration or single-patch
  attribution.  The checked record under ranked item 2 binds its identities,
  exactness evidence, and comparison scope.

Every promoted lane needs a pinned input/deck hash, current-binary serial
baseline, exact category/item/cell payload comparison, and at least one
deliberately nonempty hierarchical fixture.  Record average CPU and peak RSS,
not just wall time.

## Search-budget and acceleration policy

- Continue CPU work while a measured model projects more than **+5% whole-run
  throughput**.  Consider +2% to +5% only when implementation and exactness
  risk are low; defer work below +2%.
- Judge each new change against the immediately preceding accepted
  configuration.  Lead with percent wall-time reduction and the corresponding
  throughput change; keep cumulative stock-to-best multipliers as context, not
  patch attribution.
- After two or three misses against those projections, shift primary effort to
  acceleration instead of extending a weak CPU search indefinitely.
- Build a device-neutral spatial-candidate replay harness.  Preserve exact
  predicates on CPU and charge serialization plus host/device transfers to
  every result.  Try CUDA first.
- Proceed beyond the feasibility harness only when it demonstrates at least a
  **+10% whole-run opportunity**.  Then port the proven kernel to TT-Metalium
  for the para N300, using `tt-emule` locally before waking para.  SFPI is open
  source; compiler defects are engineering issues to isolate and fix, not a
  reason to weaken the exactness gate.

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

   - [x] **Four-process duplicate-shard contention gate — passed
     (exploratory):** on 2026-07-21, one fresh isolated `active_m1` run took
     68.52 s, while four unpinned identical copies completed in a 69.61 s
     batch.  Their 69.41–69.60 s per-job times had a 69.512 s mean and 69.520 s
     median: **+1.45% mean latency**, +1.46% median latency, and +1.58% for
     the slowest job versus the isolated reference, all well below the 10–15%
     stop threshold.  This is **+293.7% aggregate throughput**
     (3.937x) for four independent copies, not a projected full-deck speedup;
     it establishes that host contention does not block four-way shard work.
     The batch averaged 4.15 CPU cores total.  Per-process maximum RSS was
     716,724–737,400 KiB; the conservative sum of individual maxima was
     2,892,792 KiB (2.76 GiB).

     All five processes exited zero and produced byte-identical 13-category,
     zero-item reports: raw SHA-256
     `9fedf1534490be8c52b0978ec48d36e423414e14f85d0d2cd6fbef96b5392744`
     and path-normalized SHA-256
     `f0c174fd03ffc5e8a97d98d004a52ea3b6cdacc0cd9244503a05a68a87cde9b7`.
     The input, deck, binary, and jemalloc SHA-256 values were respectively
     `8111ee46a40b34b1b6fb604e25c0fcaff42d574b686d3ebcb68861916e767358`,
     `5b05b599f5af878364365136248177b63098e92ff18287d097274f54500ee2ce`,
     `cd01abf123ddd9c078b9cbf9c700238645266540be04a043f857904e59631745`,
     and
     `88abf640d394354438475ed5978616475fb28d95390f347ecfcee6ddf3beffc8`.
     This direct manual gate has one isolated observation and one concurrent
     batch, so it is not an identity-v3 promoted benchmark.  Its proposed
     rule-chain profiling, partitioning, and exact report-union validation are
     completed by the checked prototype below.

   - [x] **Four useful-rule shards and measured balancing — completed
     (exploratory):** two verbose repeats modeled 135.180 s of unique rule
     work plus 2.8125 s of eager setup per process.  Keeping producer/consumer
     chains atomic, the four-way deck assigns the Metal1/contact enclosure,
     remaining Active/Metal1 work, Metal2–10 plus the cumulative antenna graph,
     and front-end rules to separate processes.  Four independent grid-layer
     checks were moved from the critical enclosure shard to the second shard
     after the first real 45.958 s result.  The rebalanced per-shard walls were
     40.585, 38.578, 37.473, and 37.069 s; merge cost 0.007 s and the complete
     child-plus-merge workload took 40.592 s.  Against the existing 69.966 s
     two-shard mean, this is **+72.4% throughput** and 42.0% less wall time.
     Against the 128.83 s legacy serial mean, the cumulative engineering result
     is **+217.4% throughput** and 68.5% less wall time.  The full provenance-
     checked launcher took 51.499 s separately, including 5.471 s of prehash
     and 5.424 s of post-run verification; never charge that integrity work to
     the DRC child/merge comparison.  The grid rebalance alone delivered
     **+13.2% throughput** over the first four-way result.

     Every child exited zero.  Direct-child peak RSS values were 734,148,
     630,896, 633,228, and 1,213,748 KiB; their conservative, non-time-correlated
     sum is 3.06 GiB.  The full invocation averaged 3.30 CPU cores.  The merged
     real macro has 157 categories, one cell, zero items, and the same
     path-normalized SHA-256 as the established two-way result:
     `305364bd55b0337444adfb482f09e4e139f901c5612e5c030b7aae7fe7ea7f7b`.
     More importantly, the hierarchical nonempty sentinel strictly proved the
     exact 157-category, two-cell, 99-item XML payload union and matched the
     established normalized SHA-256
     `886b50da71b00fb2b3eb4fb118f0b0759a8855b2207f64c0afda3a54dc3ba893`.

     Durable corpus artifacts are `decks/freepdk45-four-way.lydrc` (SHA-256
     `2d4371c4e35d06850c05854cd7c799c8360f77b985b5bdcc756c646a0bf405dd`)
     and `decks/freepdk45-four-way-bound.json` (SHA-256
     `b24b41d7d9068f51b9baed12a9fc106b2f8d52188a08bac77ff8c2bbc6504688`);
     complete run metadata, reports, and shard/sentinel logs are under
     `evidence/freepdk45-four-way-20260721/`.  This remains one provenance-
     enforced four-way observation compared with pre-gate baselines, not a
     matched comparison-identity-v3 cohort, so it is not a promoted benchmark.
     Automatic partition generation, a matched current
     two-shard cohort, and three independent four-way observations keep the
     parent item open.

   - [x] **Five useful-rule shards — completed (exploratory):** the fifth
     process isolates the complete `METAL1.3` enclosure producer/consumer
     chain.  The remaining shards own the rest of Metal1, Metal2–10 plus
     `ACTIVE.3`, the remaining front end, and Active/Grid/Antenna.  Measured
     shard walls were 34.776, 30.509, 33.973, 29.150, and 31.662 s.  Merge cost
     0.007 s and child-plus-merge wall was 34.783 s.  This is another
     **+16.7% throughput** and 14.3% less wall than the 40.592 s four-way run.
     Relative to the existing 69.966 s two-shard mean it is **+101.2%
     throughput** and 50.3% less wall; relative to the 128.83 s legacy serial
     mean the cumulative engineering result is **+270.4% throughput** and
     73.0% less wall.  The provenance-checked launcher took 45.719 s
     separately, including 5.489 s of prehash and 5.435 s of post-run
     verification.  The full invocation averaged 3.85 CPU cores.

     All five children exited zero.  Direct-child peak RSS values were
     715,940, 621,268, 749,760, 1,219,016, and 634,320 KiB; their conservative,
     non-time-correlated sum is 3.76 GiB.  The clean real report again has 157
     categories, one cell, zero items, and normalized SHA-256
     `305364bd55b0337444adfb482f09e4e139f901c5612e5c030b7aae7fe7ea7f7b`.
     A fresh deck-bound manifest strictly proved the exact 157-category,
     two-cell, 99-item hierarchical sentinel union and the established
     normalized SHA-256
     `886b50da71b00fb2b3eb4fb118f0b0759a8855b2207f64c0afda3a54dc3ba893`.

     Durable corpus artifacts are `decks/freepdk45-five-way.lydrc` (SHA-256
     `ac1408f2ba9625529673b79663ae3659dddbbf73bb0ef66ab8cc39596b8210f0`)
     and `decks/freepdk45-five-way-bound.json` (SHA-256
     `f241b031a525353db5fcd3c67209a93f48d0147f7c0a7d85f9a9ba1dbff78885`);
     metadata, reports, and shard/sentinel logs are under
     `evidence/freepdk45-five-way-20260721/`.  This is one provenance-enforced
     observation, not a matched comparison-identity-v3 cohort.  Purely adding
     a sixth process cannot shorten the measured 34.776 s indivisible Metal1
     critical shard, so five is the useful process-count ceiling until that
     chain or its eager setup changes.

   - [x] **Conditional derived-layer setup per shard — completed
     (exploratory):** `well`, `gate`, and `implant` were constructed eagerly in
     every process.  Feature-aware guards now retain the original `all` order
     while constructing only the layers each shard can consume.  The resulting
     child-plus-merge wall is 32.433 s versus the prior 34.783 s five-way run:
     **+7.2% throughput** and 6.8% less wall time.  Shard walls are 32.426,
     27.953, 32.124, 29.210, and 30.265 s, so Metal1 enclosure remains critical
     by only 0.30 s over Metal2–10 plus `ACTIVE.3`.  The full provenance-checked
     launcher took 43.415 s separately, including 5.538 s of prehash and 5.431
     s of verification, and averaged 3.87 CPU cores.  Direct-child peak RSS
     values are 713,300, 547,312, 678,220, 1,213,744, and 610,664 KiB; their
     conservative, non-time-correlated sum is 3.59 GiB.

     The fresh deck-bound sentinel manifest proved the exact 157-category,
     two-cell, 99-item union.  Full mode retained normalized SHA-256
     `e5bf625fda5eea31fc127870f837970396fe422f3940d9f4d65ca9e6da51d4da`;
     the merged sentinel retained
     `886b50da71b00fb2b3eb4fb118f0b0759a8855b2207f64c0afda3a54dc3ba893`;
     and the real clean macro retained
     `305364bd55b0337444adfb482f09e4e139f901c5612e5c030b7aae7fe7ea7f7b`.
     Durable corpus artifacts are `decks/freepdk45-five-way-lazy.lydrc`
     (SHA-256
     `8e7f39eaac4344eff0734703f90b4accc032bda14735253ceaccddca8e0c810f`)
     and `decks/freepdk45-five-way-lazy-bound.json` (SHA-256
     `4e0ce80e34a387afcefdbe8f696c75cb894c11865609afee00ba407caecc4635`);
     reports, logs, timings, metadata, and retained shard artifacts are under
     `evidence/freepdk45-five-way-lazy-20260721/`.  This is one exact,
     provenance-enforced observation compared with the prior exploratory run,
     not a promoted comparison-identity-v3 cohort.

   - [x] **Rebalance after the guarded `METAL1.3` fast path — completed
     (exploratory):** the fast path left roughly 22.5 s of capacity in
     `m1_enclosure`.  Complete Wells/Poly and `ACTIVE.3/4` blocks now run there;
     `ACTIVE.1/2` run with upper metal, and `VIA1.4` joins the other Via1 rules
     in front end.  Fine owner predicates leave every operation in its original
     textual location, so default `all` order is unchanged.  Atomic batches,
     enclosure/classification chains, and cumulative antenna connectivity stay
     intact.  Updated derived-layer predicates cover every shard consumer.
     Every dedicated-M1 legacy fallback reloads pristine file input so moved
     rules cannot leak speculative deep-store state into `METAL1.3` ownership.

     The provenance-enforced shard walls are 25.323, 27.683, 25.774, 25.372,
     and 27.380 s.  Child-plus-merge is 27.691 s versus the preceding lazy-setup
     five-way result of 32.433 s: **+17.1% throughput**, 14.6% less wall, and
     4.743 s saved.  Versus the guarded-but-unbalanced 32.023 s run, the
     rebalance adds **+15.6% throughput**.  A separate exact direct trial took
     27.12 s; its 2.1% spread from the provenance observation is supporting
     repeatability evidence, not a promoted matched cohort.  The real clean
     report retains normalized SHA-256
     `305364bd55b0337444adfb482f09e4e139f901c5612e5c030b7aae7fe7ea7f7b`.

     Full and merged 99-marker sentinel hashes remain
     `e5bf625fda5eea31fc127870f837970396fe422f3940d9f4d65ca9e6da51d4da`
     and
     `886b50da71b00fb2b3eb4fb118f0b0759a8855b2207f64c0afda3a54dc3ba893`.
     A nonempty hierarchical union proves 157 categories, 18 cells, and 257
     items including seven `METAL1.3` items.  A stronger mixed-layer,
     transformed fixture exercises 31 nonempty Well/Poly/Active/Implant/
     Contact/Metal/Via/Grid categories before a nonempty M1 fallback.  Fresh
     manifest construction proves its exact 157-category, 7-cell, 1,587-item
     shard union; a complete semantic normalizer gives identical SHA-256
     `429d631ab89d9e0a54e4f6ed367223974b8caca564c461fda59ebda9dcbcd8d2`
     for full mode and both merges, including the M1 marker's hierarchy owner.

     Durable artifacts are `decks/freepdk45-five-way-m1-rebalanced.lydrc`
     (SHA-256
     `bcdd851dc4631c5f046cf189843bd8b23c7f3a13a8ce728313f62da129f0b825`),
     `decks/freepdk45-five-way-m1-rebalanced-bound.json` (SHA-256
     `270897bc71af1acfa57e01c7e4de50ead2339ffac8230b1eb6c129e2fb0ee47b`),
     and `evidence/freepdk45-five-way-m1-rebalanced-20260721/`.  This remains
     one provenance observation, not an identity-v3 three-observation cohort.

   - [x] **512-Kbit independent-tree stock-vs-best capstone — completed
     (exact cumulative comparison):** two physically independent clones of the
     valid 256-Kbit one-bank SRAM form a 159,997,878-byte workload: 273 cells,
     570,294 instance records, and 159,962,724 recursive shapes.  Input
     SHA-256 is
     `74911a2111a3421912e54538bf55cd12e50164f43cd1a8c47411602e64c91d98`.
     Fresh upstream `7332de72604869f9e4a9235d8621c1a1632859e1`, KLayout
     0.30.9, GCC release, the original deck (SHA-256
     `fa7edcc47d92eee4195693796c5e35b913ca476f8457965029d12372aa187db0`),
     and no allocator preload took 2769.49 s at 143% average CPU and
     6,130,844 KiB peak RSS.

     The clang/full-LTO+jemalloc binary (SHA-256
     `cd01abf123ddd9c078b9cbf9c700238645266540be04a043f857904e59631745`)
     and five-way deck/manifest took 250.685147 s child-plus-strict-merge;
     merge alone was 0.006804 s.  The full provenance launcher took
     261.823847 s internally, or 261.96 s by `/usr/bin/time`, including
     11.126592 s of integrity work.  The DRC-plus-merge comparison is
     **+1004.8% throughput** and 90.9% less wall time; charging all external
     launcher wall to the optimized side still gives **+957.2% throughput**
     and 90.5% less wall time.

     The complete semantic result is exact at 157 categories, one cell, zero
     items, SHA-256
     `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`.
     The nonempty sentinel is also exact at 157 categories, two cells, 99
     items, SHA-256
     `265a2e1c58aed60bc89e8bd2ad2904147a1e53fcd7a2a7c8bfa8ddaa339cd1a2`.
     This is one cumulative best-configuration-versus-fresh-stock observation,
     not single-patch attribution or an identity-v3 promoted cohort.  Evidence
     is under
     `evidence/freepdk45-capstone-stock-7332de7-20260721/` in the external
     corpus.

   - [x] **Eight-way 512-Kbit owner rebalance — completed (three exact
     observations):** split the previous `m1_rest`, `front_end`, and
     `active_grid_antenna` bundles into Metal1 width/space, Metal1 Via1/
     classification, Implant/Contact, Via1, Grid, and Antenna owners.  Keep
     the guarded M1 enclosure and Metal2/upper/Active owner atomic.  Eight
     processes times four requested inner threads match the 32-thread budget;
     default `all` mode retains historical textual rule order.

     Child-plus-strict-merge walls are 208.427864, 207.237012, and
     207.694970 s: mean **207.786616 s (3m27.787s)** and 0.573% full-range
     spread.  Against the immediately preceding 250.685147 s five-way result,
     this is **17.1% less DRC wall time** and **+20.6% throughput**, saving
     42.899 s.  Full provenance-launcher walls average 218.900690 s versus
     261.823847 s previously: **16.4% less wall time** and **+19.6%
     throughput**.  The unchanged Metal2 owner is now critical at a 207.778 s
     three-run shard mean, as projected.

     All three real reports are raw-byte identical and retain the complete
     semantic SHA-256
     `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`.
     The hierarchical sentinel remains exact at 157 categories, two cells,
     and 99 items, semantic SHA-256
     `265a2e1c58aed60bc89e8bd2ad2904147a1e53fcd7a2a7c8bfa8ddaa339cd1a2`.
     The transformed mixed-layer fixture remains exact at 157 categories,
     seven cells, and 1,587 items, semantic SHA-256
     `429d631ab89d9e0a54e4f6ed367223974b8caca564c461fda59ebda9dcbcd8d2`.

     Durable artifacts are `decks/freepdk45-eight-way-x2.lydrc` (SHA-256
     `a5dbd765477f7ae657bd5335eae231f685e605d16cd30e5d5a1058cdd0c3f4b2`),
     `decks/freepdk45-eight-way-x2-bound.json` (SHA-256
     `fde4d137881d497daf96a981530cd4d6f0ca6ffe0c3542e1ebd875c08d249ca0`),
     and `evidence/freepdk45-eight-way-x2-20260721/`.  This is a qualified
     FreePDK45/x2 scheduling result, not cross-design evidence for an engine
     change.

   - [x] **Nine-way Metal2 follow-up — superseded by a 32-thread-budget
     repack:** splitting the current critical owner into its intact
     `METAL2.1-.9` block and the remaining Active1/2 plus Via2/upper metal block
     would create nine four-thread processes and request 36 threads.  Preserve
     the split idea, but do not exceed the user's 32-thread ceiling.

   - [x] **Eight-way Metal2 follow-up — completed (three exact observations):**
     isolate `METAL2.1-.9`, then combine Active1/2 plus Via1 and Via2/upper
     metal into one smaller owner.  The other six owners remain unchanged, so
     eight processes times four requested threads stay within the 32-thread
     budget.

     Child-plus-strict-merge walls are 189.368010, 187.034080, and
     188.383469 s: mean **188.261853 s (3m08.262s)** and 1.240% full-range
     spread.  Against the immediately preceding 207.786616 s eight-way mean,
     this is **9.4% less DRC wall time** and **+10.4% throughput**, saving
     19.525 s.  Full provenance-launcher walls average 199.401136 s versus
     218.900690 s previously: **8.9% less wall time** and **+9.8%
     throughput**.

     All three real reports are raw-byte identical, SHA-256
     `19543eb1328eb73f058caf9cbf4d5f40721337d7fb687e78ef57427aced57992`,
     and retain the complete semantic SHA-256
     `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`
     at 157 categories, one cell, and zero items.  The sentinel remains exact
     at 157 categories, two cells, 99 items, semantic SHA-256
     `265a2e1c58aed60bc89e8bd2ad2904147a1e53fcd7a2a7c8bfa8ddaa339cd1a2`;
     the mixed hierarchy remains exact at 157 categories, seven cells, 1,587
     items, semantic SHA-256
     `429d631ab89d9e0a54e4f6ed367223974b8caca564c461fda59ebda9dcbcd8d2`.

     The new mean critical path is M1 enclosure at 188.254 s; Implant/Contact
     is 180.237 s and Metal2 is 174.642 s.  Normalizing only each required
     fresh-home path makes all three old and all three new runtime,
     environment, orchestrator, and host identities exact; deck, manifest, and
     their two shard assignments are the intended changes.

     Durable artifacts are `decks/freepdk45-eight-way-m2-repack.lydrc`
     (SHA-256
     `ef860b01ea7b8c67437f5e7256f2f88df7d8748ff4e9d5efdacf07978942300f`),
     `decks/freepdk45-eight-way-m2-repack-bound.json` (SHA-256
     `28ab1c3f206ba5c152eec42b47366934c822d53cd2c3154129410696f8eebf8d`),
     and `evidence/freepdk45-eight-way-m2-repack-20260721/` in the external
     corpus.  This is a qualified FreePDK45/x2 scheduling result, not an
     engine-wide claim.

   - [x] **Dual-bottleneck eight-way repack follow-up — completed (three exact
     observations):** move the intact WELL block plus `ACTIVE.4` from M1
     enclosure to the underloaded antenna owner, and move the self-contained
     `CONTACT.6` rule from Implant/Contact to the M1-width owner.  Eight
     processes times four requested threads retain the 32-thread budget.

     Child-plus-strict-merge walls are 175.654973, 174.520844, and
     174.928626 s: mean **175.034815 s (2m55.035s)** and 0.648% full-range
     spread.  Against the immediately preceding accepted 188.261853 s mean,
     this is **7.0% less DRC wall time** and **+7.6% throughput**, saving
     13.227 s.  Full provenance-launcher walls average 186.162139 s versus
     199.401136 s previously: **6.6% less wall time** and **+7.1%
     throughput**.

     All three real reports are raw-byte identical, SHA-256
     `f79d15877d9029b45fe5711ff84d6a1d5f057573aaace98a4d4469933b53caec`,
     and retain the complete semantic SHA-256
     `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`
     at 157 categories, one cell, and zero items.  The sentinel remains exact
     at 157 categories, two cells, 99 items, semantic SHA-256
     `265a2e1c58aed60bc89e8bd2ad2904147a1e53fcd7a2a7c8bfa8ddaa339cd1a2`;
     the mixed hierarchy remains exact at 157 categories, seven cells, 1,587
     items, semantic SHA-256
     `429d631ab89d9e0a54e4f6ed367223974b8caca564c461fda59ebda9dcbcd8d2`.

     Mean Metal2 wall is now the critical path at 174.608 s, with M1
     enclosure close behind at 173.756 s.  Normalizing only each required
     fresh-home path makes all three preceding and all three new runtime and
     environment identities exact; orchestrator and normalized host identities
     also match.  Exactly the declared five category owners change.

     Durable artifacts are `decks/freepdk45-eight-way-dual-repack.lydrc`
     (SHA-256
     `3e981b9389a67c6c1c4b08f0640d8750ca78990c868c5686fa8ebbd401cba72c`),
     `decks/freepdk45-eight-way-dual-repack-bound.json` (SHA-256
     `7d5b09621e8da2c76091001a2a5bcd37210822698f7908d19aab282490b8811b`),
     and `evidence/freepdk45-eight-way-dual-repack-20260721/` in the external
     corpus.  This is a qualified FreePDK45/x2 scheduling result, not an
     engine-wide claim.

   - [ ] **Profile-derived eight-way critical-path rebalance — in progress:**
     retain the existing eight-process/four-inner-thread envelope, but move
     intact `METAL2.5-.9` from `m2_rules` to `via1_upper_active12` and move
     `POLY.1/.2/.3/.5/.6` (not the heavy `POLY.4`) from `m1_enclosure` to
     `antenna`.  Three retained verbose cohorts model Metal2 at about 159.33 s,
     M1 enclosure at about 159.13 s, and full launcher wall near 170.46 s:
     approximately **8.4% less wall time** and **+9.2% throughput** versus the
     immediately preceding accepted dual-repack mean.  The split preserves
     historical all-mode rule order and moves each derived-layer producer,
     consumer, and `forget` together.  Regenerate the deck-bound manifest and
     pass sentinel, transformed mixed-hierarchy, and three-observation x2
     gates before accepting the model.  `POLY.2` is absent from the existing
     157-category manifest because its pre-existing `polygons?` guard is a type
     test on an edge-pair result and is therefore always false; even the mixed
     fixture's 30 nm Poly/Active gap cannot fire it.  Do not fold a signoff-
     semantics repair into this scheduling treatment or claim that category was
     exercised.  Try this lower-overhead rebalance before a ten-owner/three-
     inner-thread sweep.

   - [x] **Optional final five-way balancing nibble — deferred below the search
     threshold:** moving the intact
     `METAL1.5-1.9` classification block (about 0.70 s) from `m1_rest` into the
     enclosure shard projects only about **+1% to +1.4% throughput** before
     Grid/Antenna becomes critical.  The eight-way split supersedes this move,
     and the explicit below-2% policy now defers it.  Preserve this record so
     the nibble is not repeatedly rediscovered.
3. [ ] **Thread/core-budget sweep and affinity**
   Sweep 1/2/4 inner threads per shard and restrict the complete launch to
   2/4/8 physical CPUs.  Today, two processes request eight worker threads but
   average only about 2.06 CPU cores total, so the +84.1% result probably does
   not require a 32-core machine.  Establish the minimum practical core budget,
   then test disjoint physical-core/CCD pinning only after four-way sharding.
   Always bound outer jobs times inner threads and record RSS.  The earlier
   ten-owner/three-inner-thread proposal is only a finer process partition of
   the same exact DRC deck and merged report, not a different workload; 10 x 3
   merely fits 30 requested threads under the 32-thread ceiling.  It duplicates
   ten runtimes and is demoted behind the modeled eight-owner/four-thread
   critical-path rebalance unless this sweep proves three inner threads retain
   enough per-shard performance to justify the extra owners.
4. [ ] **Native bounded rule-DAG executor**
   Ruby declares dependencies and output order serially; only C++ terminal
   geometry tasks run in parallel, with private results and serial publication
   by deck ordinal.  Arbitrary Ruby blocks, mutable report databases, layouts,
   and `DeepShapeStore` outputs must never execute concurrently.  This removes
   process supervision and may reduce duplicated setup without weakening the
   isolation proof.  Current verbose profiles expose genuinely independent
   terminal work on both tied owners: M1 has stable enclosure operations around
   44.65-50.33 and 65.61-66.00 aggregate seconds; M2 has a 41.02-41.15-second
   width/space batch plus roughly 22.2 seconds of classification.  Their
   overlap models a possible double-digit ceiling, but that is not a measured
   whole-run win and implementation risk is high.  Require an exact shadow
   schedule demonstrating at least +5% whole-run opportunity before changing
   the executor; keep Ruby, layouts, reports, shared stores, and ordered
   publication serial.
5. [x] **Fuse enclosure producer/consumer chains — completed with a guarded
   clean-result fast path (exploratory):** `metal1.enclosing(cont)` cost about
   22.3 s and materialized roughly 2.6 million edge pairs before
   `.second_edges.width(...)` spent another 5.5 s.  Instead of weakening that
   generic relation, the dedicated `m1_enclosure` shard now proves that merged
   contacts are isolated exact 65 nm squares, commutes the relation as a
   cheaper negative probe, and uses its result only when it is empty.  A failed
   domain guard runs the historical expression; a nonempty probe reloads a
   pristine source and runs the historical chain so report hierarchy ownership
   is exact.  Default `all` mode executes the original expression directly.

   The real shard fell from 32.425886 s to 9.887987 s: **+227.9% shard
   throughput**, 69.5% less wall time, and 22.538 s saved.  As expected, the
   unbalanced child-plus-merge result moved only from 32.433202 s to 32.023385
   s (**+1.3% whole-run throughput**) because the 32.014800 s
   `m2_upper_active3` shard immediately became critical.  This establishes
   spare capacity for the next rebalance; never report the shard gain as the
   whole-run gain.  The real clean merge retains normalized SHA-256
   `305364bd55b0337444adfb482f09e4e139f901c5612e5c030b7aae7fe7ea7f7b`.

   The deliberately nonempty transformed hierarchy fixture takes the pristine
   legacy fallback and is exact.  Five other deterministic fixtures cover the
   empty fast path and each domain fallback.  A 300,000-case flat audit and a
   10,000-case deep rotated/mirrored audit found no downstream geometry
   mismatch inside the guarded domain.  The full and strict-merged 99-marker
   sentinels retain 157 categories, two cells, 99 items, and normalized hashes
   `e5bf625fda5eea31fc127870f837970396fe422f3940d9f4d65ca9e6da51d4da`
   and
   `886b50da71b00fb2b3eb4fb118f0b0759a8855b2207f64c0afda3a54dc3ba893`.
   Durable artifacts are `decks/freepdk45-five-way-m1-gated.lydrc` (SHA-256
   `03e3cfee2426dca824b8434843c443287faa72a196c8a0e746772fd46258a6ba`),
   `decks/freepdk45-five-way-m1-gated-bound.json` (SHA-256
   `6f849cb08d6a78c45228cccfc306aac68171d6683b9d533af6de058d65b13185`),
   and `evidence/freepdk45-five-way-m1-gated-20260721/`.  This is one exact
   provenance-enforced observation, not a promoted three-observation cohort.

   - [ ] **Generic enclosure-derived width fusion — profiled, not yet
     implemented:** the current
     `enclosing(...).second_edges.width(...).polygons(...).interacting(...)`
     chain materializes four global intermediate collections.  In current
     flat-symbol profiles, the enclosing range plus derived edge-width kernel
     accounts for about 42.2% of M2 samples but only 8.2% of M1 samples.  A
     Ruby expression rewrite is unsafe because shielding is input-order
     asymmetric and deep hierarchy ownership plus merged-edge semantics are
     observable.  If the ceiling remains above +5% after lower-risk work, use
     a narrow local operation that preserves the original relation orientation,
     same-layer neighbors, transforms, shielding, and ordered publication;
     compare every intermediate as well as the final report on flat/deep
     adversarial fixtures before timing it.
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

   - [x] **Zen 2 code-generation trial — completed (two cross-PDK,
     three-observation cohorts):** fresh portable and
     `-march=znver2 -mtune=znver2` bundles use the same `eb6d701` engine source,
     Clang 22 full LTO, allocator, and runtime dependencies.  Target flags are
     present in C, C++, and final LTO-link commands.  FreePDK45/x2
     child-plus-strict-merge means are 172.837420 s portable versus 165.094884
     s tuned: **4.5% less wall time and +4.7% throughput**.  Full provenance
     launcher means are 183.980714 versus 176.230410 s: **4.2% less wall time
     and +4.4% throughput**.  All six reports share raw SHA-256
     `f79d15877d9029b45fe5711ff84d6a1d5f057573aaace98a4d4469933b53caec`
     and semantic SHA-256
     `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`
     at 157 categories, one cell, and zero items.

     Sky130 HG0 S5 means are 196.892068 s portable versus 188.471924 s tuned:
     **4.3% less wall time and +4.5% throughput**.  Full-range spreads are
     0.194% and 0.136%; all six S5 reports retain normalized SHA-256
     `2c9f660d7b2d7186329c510333083bfe19ab17779fe42d66c936feac0b45fdb4`,
     and all six nonempty sentinels retain 25 items.  Comparison identities
     match outside the declared runtime bundle.  Durable evidence is under
     `evidence/klayout-znver2-qualification-20260722/`; summary JSON SHA-256 is
     `c1e8d42679f69a3b6333d0134098f61f9bcd781c3bd88c5492c096dd7be42633`.

     `scripts/build-clang-perf.sh` keeps portable full-LTO as the default.
     `auto` and explicit `znver2` specialize only after exact-host detection
     plus C, C++, full-LTO-link, and execution probes; any unsupported host or
     failed specialized probe warns and falls back to a distinct portable
     directory.  A fail-closed manifest prevents reuse of mixed toolchain/
     flag artifacts.  Twelve fake-tool tests and a real Zen-2 dry-run pass; the
     helper never silently publishes `-march=native` code.

   - [ ] **Cross-PDK PGO — next low-semantic-risk CPU trial:** train with both
     FreePDK45/x2 and Sky130 S5 so one deck cannot dominate the profile.  Use
     same-source portable/full-LTO control and optimized bundles, exact
     real/sentinel/mixed reports, and three interleaved observations per PDK.
     Accept at +5% whole-run throughput, or at +2% to +5% only if both PDKs are
     consistent and neither has a meaningful regression.  Keep portable as
     the default and treat BOLT as contingent on the resulting counters.
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
    operation-level thread scaling.  Current verbose operation `Elapsed` is
    aggregate CPU-like time: M1 totals roughly 289 s and M2 roughly 270 s while
    their shard walls are about 175 s, only 1.65 and 1.55 effective cores despite
    `threads(4)`.  Current code schedules whole cells and then walks sorted
    contexts serially within each cell.  First sweep isolated 1/2/4-thread
    critical shards and instrument per-cell/context durations; proceed only if
    dominant serial tasks model at least +5% whole-run.  Compute independent raw
    context results thread-locally while retaining sorted serial reduction and
    publication, and always cap outer owners times inner threads at 32.
12. [ ] **Lazy interpreter initialization**
    Reduce the roughly 1.4 s batch startup floor for short jobs; separate from
    long-run geometry work.
13. [ ] **Incremental antenna edge replay — re-profile first**
    The old estimate assumed ten cumulative rebuilds.  Empty-target pruning
    removed six, so its present ceiling is much smaller and must be measured
    before implementation.
14. [ ] **Device-neutral accelerator replay gate — later, orthogonal project**
    Do not translate the Ruby PDK deck to an accelerator language.  Build a
    device-neutral replay harness around spatial bin/sort plus candidate
    filtering, initially for a measured high-cardinality geometry kernel.
    Retain exact predicates and hierarchy reduction on CPU.  Instrument
    serialized bytes, candidate-reduction ratio, serialization, transfers,
    kernel time, and CPU fallback.  Try CUDA first; require at least a **+10%
    whole-run opportunity** after all overhead before continuing.  If it
    passes, port the proven kernel to TT-Metalium on the para N300, testing
    locally with `tt-emule` before waking para.  Treat SFPI/compiler defects as
    fixable engineering work while preserving the CPU exactness oracle.

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

- [x] Direct rectangle predicate for `METAL1.3`: a universal
  `cont.drc(if_any(enclosed(metal1, ...)))` prototype ran in about 8.24 s on
  the real clean SRAM but was not equivalent.  The comprehensive fixture
  exposed four false-positive classes, expanding to 135,792 false-positive
  flat markers on the real hierarchy.  Do not substitute a per-contact
  rectangle test for the edge-pair/corner rule.
- [x] Conservative candidate prefilter before legacy `METAL1.3`: the real run
  was exact and took about 13.04 s, but randomized shielding cases produced
  both false positives and false negatives.  It is not a universal early
  filter and must remain rejected.
- [x] Unguarded `enclosing`/`enclosed` relation reversal for `METAL1.3`: the
  real macro and initial fixtures were exact at about 9.52 s, but KLayout's
  default shielding pass is input-order asymmetric.  Random flat cases found
  geometry counterexamples, and a legal nonempty hierarchy produced different
  logical marker ownership despite identical flattened markers.  `NO_SHIELD`
  confirmed the cause.  Only the guarded empty-terminal form recorded in item
  5 is accepted.
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
