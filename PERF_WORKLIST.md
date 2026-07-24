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
  takes 46m09.49s on a fresh upstream-master stock build.  The formally
  qualified current PGO bundle averages 2m25.564s for eight-owner DRC plus
  strict merge, or 2m36.692s for the full provenance launcher including about
  11 s of integrity work.  The conservative stock-to-full-launcher comparison
  is **17.675x throughput and 94.34% less wall time**; comparing DRC process
  wall with child-plus-merge is **19.026x and 94.74% less wall time**.  These
  are cumulative build, engine, scheduling, allocator, and PGO results—not
  single-patch attribution.  Stock is one historical observation while the
  current value is a three-run mean, so this multiplier is not an identity-v3
  promoted cohort.  Use this lane to demonstrate and re-profile cumulative
  scaling, not for routine iteration.  The checked records under ranked items
  2 and 8 bind identities, exactness evidence, and comparison scope.

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

   - [x] **Sky130 S3 FEOL/LI three-way split — completed and rejected
     (exact one-shot head check):** after PGO qualification, the established
     two-way `feol_li`/`ct_up` split and the proposed three-way
     `feol`/`li`/`ct_up` split ran concurrently on disjoint 16-CPU sets with
     the same qualified PGO binary (SHA-256
     `0e0ba390970b4cd4c888d48a2cac2fcbfed57133cf1b4aa0f0bd1bdbda8f27ae`),
     S3 input, allocator, four engine threads per child, and provenance
     harness.  Two-way child-plus-merge wall was 85.199454 s; three-way was
     85.655893 s: **0.54% more wall time and -0.53% throughput**.  Full
     launcher walls were 96.331464 and 96.915400 s respectively.

     The two-way shard walls were 84.537319 s for `feol_li` and 85.189429 s
     for `ct_up`; the three-way walls were 38.985423 s for `feol`, 50.393985 s
     for `li`, and 85.645375 s for the unchanged `ct_up`.  Splitting FEOL/LI
     therefore cannot shorten this workload's critical path.  Both merged
     reports have 256 categories, one cell, and zero items.  Their raw SHA-256
     values are respectively
     `ae4d2bd9feabf689042d9208e77d8c9a1d4a07c1ee4cc9144e666cc4a1081f9f`
     and
     `53589b5b7516b6cb636bcaeed1fd703030f83c8eee3349dcfb7723ce6e45ae9f`;
     generator-normalized SHA-256 is identically
     `8b558a5ab8303a312cab554c3e271371acb1f1e42d0bf55f96e2307ec58c0001`
     and semantic SHA-256 is identically
     `7cba50625f1576cb98ffb79ddb46a93f04d4d5b1d2804aaee01ace34bc75b2ef`.

     Input, three-way deck, and deck-bound manifest SHA-256 values are
     `a5aae78efed5a76e5f2c6c03762f4fc60d70132931028bb47f941f08d3184de7`,
     `0ffa8ea5de0fb4af0b96abf07d6b627487d24cadc9655c9f0d1b872aacf5445e`,
     and
     `f82392cc0581d57e38e826ce90c8794b2f2cd62b53bed0d6ed125c154639927b`.
     Evidence is archived under
     `evidence/klayout-pgo-qualification-20260722/sky130-s3/`.  This is one
     paired exploratory observation, not a promoted cohort.  It is closed by
     the sub-2% stopping rule; do not try alternate FEOL/LI groupings unless a
     new split first reduces the `ct_up` critical shard.

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

   - [x] **Profile-derived eight-way critical-path rebalance — rejected after
     an exact real screen:** retaining eight owners and four inner threads,
     move intact `METAL2.5-.9` from `m2_rules` to
     `via1_upper_active12`, and move `POLY.1/.2/.3/.5/.6` (not the heavy
     `POLY.4`) from `m1_enclosure` to `antenna`.  The candidate passed the
     157-category/99-item sentinel and the 157-category/1,587-item transformed
     mixed-hierarchy gate with exact semantic hashes.  Its real x2 report also
     retained semantic SHA-256
     `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`.
     The screened deck SHA-256 was
     `dad07e76e215e950495855810393a1e860938a351fa8a4cc713ce0a975d18d61`
     and its bound manifest SHA-256 was
     `3fa705c9465f52703809c8e9950f6c0a03cea50bd9cc48c0ad6ec889748f051c`.

     The performance model did not survive a same-runner screen.  The accepted
     deck took 164.889428 s child-plus-merge and 176.007337 s full-launcher wall;
     the candidate took 165.623715 s and 176.763536 s respectively: **0.4%
     more wall time and -0.4% throughput** on both measures.  The critical path
     merely moved to `via1_upper_active12` at 165.613401 s.  The moved Metal2
     classification chain cost 51.700 s of verbose aggregate elapsed time when
     cold, versus 22.08-22.27 s while co-located with `METAL2.1-.4`; its first
     `sized` operation alone rose from 4.24-4.28 s to 34.00 s.  This is lost
     in-process geometry reuse, not measurement noise that merits a cohort.

     Moving only Poly models about **2.8% less wall time and +2.9% throughput**,
     below the search threshold.  A narrower Metal2 split is not independent:
     `classify_by_width` cumulatively reassigns `layer` for each threshold, so
     `METAL2.6-.9` consume the `METAL2.5` morphology result.  Separating the
     suffix either changes semantics or duplicates the expensive cold prefix.
     The existing sentinel and mixed fixture do not prove that altered
     intermediate geometry is equivalent merely because their final reports
     match.
     Retain the accepted dual-repack and do not rediscover this scheduling
     route.  `POLY.2` remains absent from the manifest because its pre-existing
     `polygons?` type-test guard is always false; no signoff repair was mixed
     into this experiment.

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
8. [x] **Target-specific code generation and cross-PDK PGO**
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

   - [x] **Cross-PDK PGO — completed and formally qualified:** source commit
     `575a32b27db9e9474a4f829f277cdc5703afa3c5`, tree
     `d6810f6fe72eddaf4a951a05d03708b373a24a0d`.  Instrumented mixed training
     produced 504 FreePDK45/x2 and 96 Sky130 S5 raw profiles.  The selected
     FreePDK45:Sky130 merge weights are 1:9, leaving 2.214711396% weighted
     imbalance.  The merged profile SHA-256 is
     `a31ad8f30edc10523fb59f3674bcd742a7f685697bd23314a840bb2edf472fce`;
     its manifest SHA-256 is
     `9e0e44d9ac6929fe6f855b4dfa342c92cdc46c768b605f8d313125b73a3177d6`.

     The PGO-use build took 26:24.79 and produced executable SHA-256
     `0e0ba390970b4cd4c888d48a2cac2fcbfed57133cf1b4aa0f0bd1bdbda8f27ae`,
     build-manifest SHA-256
     `4a637551fc7e20b3c2788803f0c201d922af8f6b11a3124517c883febe04526d`,
     and lifecycle-state SHA-256
     `c6b5ab6fe1fba669d7f181d017000a109fa0569a2b2981ec356d1ebd3d9b06cb`.
     A fresh same-source non-PGO control took 28:06.43 and produced executable
     SHA-256
     `a2f707d341b07fe161bca1909d79e99dbf70489454ea105bdd43ce55ae7c627c`,
     manifest SHA-256
     `b37eed60f3bcf47afec3edffc22bcf54b480ecdf97987eae66db85c9ee3f5fc3`,
     and lifecycle-state SHA-256
     `bd8dd8f4bdbdf0a0f1c6c528de3e7e5bdff4fa63c91a5a9c6cfbe871fe07e493`.
     The control receipt independently proves identical non-PGO configuration
     and the absence of profile flags.

     The explicitly preliminary one-pair predecessor screen measured
     FreePDK45/x2 at 175.789468 s versus 158.288782 s, **9.955% less wall time
     and +11.056% throughput**, and Sky130 S5 at 188.376887 s versus
     161.667109 s, **14.179% less wall time and +16.521% throughput**.  It is
     retained only as a screen; the accepted result is the formal same-source
     control cohort below.

     Formal ABBAAB/BAABBA qualification used three observations per treatment
     and PDK.  FreePDK45/x2 control and PGO means were 180.559164 and
     156.691627 s: **13.219% less wall time and +15.232% throughput**.  Its
     control samples were 181.207872, 179.914453, and 180.555168 s; PGO samples
     were 156.725157, 157.448170, and 155.901553 s.  Sky130 S5 control and PGO
     means were 194.036149 and 161.527181 s: **16.754% less wall time and
     +20.126% throughput**.  Its control samples were 193.578250, 194.047442,
     and 194.482754 s; PGO samples were 161.443286, 161.402617, and 161.735640
     s.  Both unrounded throughput gains exceed the +5% gate with sub-1%
     full-range spread in every lane.

     All twelve real reports and all sentinels were exact.  FreePDK45 retained
     raw report SHA-256
     `f79d15877d9029b45fe5711ff84d6a1d5f057573aaace98a4d4469933b53caec`
     and semantic SHA-256
     `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`
     at 157 categories, one cell, and zero items; its 99-item sentinel and
     1,587-item transformed mixed-hierarchy gates also passed.  Sky130 S5
     retained normalized SHA-256
     `2c9f660d7b2d7186329c510333083bfe19ab17779fe42d66c936feac0b45fdb4`
     with zero items, while every nonempty sentinel retained normalized
     SHA-256
     `22057d4a1882b23d7367fb99981c7de52f066c4ab4beebc792be2ca656bcb404`
     and 25 items.

     Training-campaign, profile-selection, PGO-use, screen, control, and formal
     qualification receipt SHA-256 values are respectively
     `a83c8a522bb6fed5d18d6d7c6dec7ca48383cbd0a28cc314be0b441164369f65`,
     `7f79940da93cdcc0935aafe14b825ce718cd0a4f685fd14d5c820cccd5d73f31`,
     `ec98cd9c2c4d1c5395273c0aa4634964401765888a8106089ae9c307daed0455`,
     `6cd37e6a42c7303994d2504a64f2abc2e49511c27c92ffd2032ebbb2167cd298`,
     `71e2080e79ac01291d52bc92a65cdb3ebe68775ab979173b14a14aff1f09c5dd`,
     and
     `fb09a1deb43389b3c85a3fed3e0daddf1a8baf5c3bf5e1ad524052ed40002428`.
     Train, merge, PGO-use, screen, control-build, and qualification wrapper
     SHA-256 values are respectively
     `d27431070c8b8749673a45864cdd7bffd8c35f511233059e089538e29c111d80`,
     `e74bbf1ec5d1282dcf488d4ce1f24edbe63b3cb249de017557f650e01337da97`,
     `4f103fd007f147bdd68f414a293d5120f82c5d1d1ee7fa66c64c57b51728218f`,
     `f642e4826d3c37710f9ba0e975aef001ae2bb48cbcd19f673581153cd0382774`,
     `82222ed4f89ef07ee0a284093101f0f2279e8c47e40d51833929c9d2e8d6d6a8`,
     and
     `7ad7b99b64e280ff78852cdd46e4cde94b117dd35de66cf22bb308faa7aa57bb`.
     Compact receipts, manifests, wrappers, and the final S3 head check are
     archived under `evidence/klayout-pgo-qualification-20260722/`.
     Portable full-LTO remains the default; PGO is a qualified specialized
     bundle, and BOLT remains deferred unless a fresh residual profile clears
     the whole-run threshold.
9. [ ] **Generalized exact early pruning**
   Inventory operations whose required target or interaction layer is empty
   and prove that skipping extraction cannot change report categories or
   diagnostics.  Empty-target antenna pruning was valuable; use profiling to
   avoid scattering checks with no measurable ceiling.

    - [x] **Rectilinear grid early-empty certificate:** deep grid checks now
      prove that raw polygon edges and every reachable simple hierarchy
      transform preserve the requested lattice before materializing merged
      polygons.  The proof is deliberately fail-closed for diagonal/nonpolygon
      geometry, off-grid placements or array vectors, magnification/arbitrary
      rotation, and unknown/custom array delegates; exact concrete array types
      and zero-cardinality metadata avoid a `size()` multiplication-overflow
      escape.  The normal merged path remains the fallback.

      Five simultaneous, CPU-affinity-isolated 64-Kbit FreePDK45 grid pairs
      measured **20.800 s control versus 3.036 s candidate means**: **17.764
      real seconds removed, or 85.4% less shard wall time**.  All ten reports
      were byte-identical (SHA-256
      `fa05238947d072c2de998e82483a153d6ee5a224e131e2721247569184fbfaeb`).
      A nonempty violation sentinel also stayed byte-identical (SHA-256
      `420b02729315839c12594bf25f7390acfb99260e5fa1c869d3067284002f6ead`),
      and all 124 `dbDeepRegionTests` executions pass.  This removes grid as a
      cross-shard bottleneck; it is not by itself a whole-launch or CUDA win
      because M1 remains critical.
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
    - [x] **Split the antenna owner into three CPU processes — completed:**
      this is process scheduling on the CPU `perf` branch, not CUDA work.
      The unchanged PGO executable ran `antenna_feol`, `antenna_m1_m2`, and
      `antenna_m3_m10` as independent owners while preserving the cumulative
      connectivity prefix required by each check.  Against the prior three-run
      `antenna` mean of 114.409553 s, three exact x2 screens made the new
      antenna critical path 55.402531, 56.556236, and 56.764035 s: a
      56.240934 s mean, **50.84% less lane wall time** and **+103.43%
      antenna throughput**, with 2.42% full-range spread.  The component means
      are 22.506895 s FEOL, 50.393452 s M1/M2, and 56.240934 s M3--M10.

      This removes antenna as a future stacked bottleneck; it does not speed
      the present CPU-only full launch because M1 remains near 145 s.  Full
      provenance-launch means changed only 156.691627 -> 156.006005 s
      (**0.44% less**, treated as noise rather than a whole-run win).
      All three clean merges have the trusted semantic SHA-256
      `dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`
      at 157 categories, one cell, and zero items.  The deliberately nonempty
      hierarchical fixture proves full mode, split `all`, and the ten-owner
      merge share semantic SHA-256
      `409085bc15614329421b1b07a6fdd6b867fe8cad83a214a1f32a5d9986a595a4`
      at 157 categories, two cells, and 86 items, including violations on both
      sides of the M2/M3 owner boundary.

      KLayout emits the antenna diagnostic declarations and named values in
      process-dependent order.  The merger now canonicalizes only those named
      fields, preserves positional values exactly, proves the complete tag
      universe when creating a manifest, and permits a later clean layout to
      emit an empty subset while still rejecting unknown or conflicting tags.
      Keep `--jobs 8` with four KLayout threads per process so ten owners never
      request more than 32 threads.
14. [ ] **Device-neutral accelerator replay gate — active, orthogonal project**
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

    - [x] **CUDA environment and first-target audit:** the local RTX 3080
      (compute capability 8.6, 9.64 GiB) ran a CUDA 12.4 kernel and sustained
      about 25.7 GB/s in each direction with pinned host memory.  A packed
      83 MB candidate stream therefore costs only about 3.2 ms one-way when
      batched; serialization, working-set size, and CPU exact replay are the
      meaningful overheads.  The initially proposed edge-only seam was the
      pointer-free candidate list between `box_scanner<Edge, size_t>` and
      `Edge2EdgeCheckBase::add()`, with hierarchy and exact predicates unchanged
      on CPU.  The current 156.69 s
      full-run mean needs to reach at most 142.45 s to clear the +10%
      opportunity gate.  Accelerating M1 alone models only about +3.9%; a
      production win must benefit at least M1 enclosure, implant/contact, and
      M2 together.  Build and qualify the CPU recorder/replayer before adding
      the CUDA broad-phase, and charge packing, transfers, sorting/dedup,
      fallback, and simultaneous-shard contention.

    - [x] **Correct the target ceiling and prove the standalone CUDA kernel:**
      symbolized attribution showed the edge-only seam was too small and it was
      abandoned before production integration.  Defensible removable shares
      are 7.88% of the M1 shard, 11.29% of implant/contact, and 21.30% of M2;
      even perfect removal models a 145.22 s whole run, only **+7.9%
      throughput**, short of the +10% gate.  The larger device-neutral target is
      now generic self/bipartite int64 AABB candidate generation, with a second
      possible shielding-incidence sort/join kernel; exact predicates and
      ordered publication stay on CPU.

      `benchmarks/cuda_spatial_replay` provides a bounded, versioned,
      pointer-free recorder/replayer and exact CPU oracles.  Boundary,
      bipartite, signed-coordinate, replay, density, overflow, repeatability,
      and million-record gates pass.  Five independent million-record launches
      all returned the identical 2,552,846-pair hash; charged pack plus GPU
      pipeline averaged 78.936 ms (78.107-80.236 ms), including per-call device
      buffer teardown, versus 1,349.768 ms for
      its CPU grid oracle.  That is a synthetic stage result, not a KLayout
      speedup.  Three concurrent owner-like processes sustained 95-97% sampled
      GPU activity but did not increase aggregate throughput, favoring a future
      single broker with persistent buffers.  Production integration remains
      gated on finding roughly two more percentage points of removable M1 work
      and charging CPU replay plus broker overhead.

    - [x] **First real CUDA/KLayout vertical slice — proof of concept:** an
      optional CUDA DSO now exposes the versioned POD broad-phase ABI, while a
      normal KLayout build remains CUDA-header- and CUDA-link-free and discovers
      the accelerator only after explicit opt-in.  The first integration seam
      is deliberately limited to the audited different-layer shape scan.  It
      validates sorted IDs and bounds, reruns the exact AABB predicate on CPU,
      and falls back to the untouched CPU scanner before any callback on loader,
      capacity, CUDA, or validation failure.  The RTX 3080 ABI fixture returned
      exactly `(1,3),(2,4)`, and a through-the-DSO 1024-by-1024 gate matched all
      1,245 candidates from 1,048,576 exhaustive CPU checks, including signed
      and strict-boundary cases.  The standalone replay's exhaustive,
      signed-coordinate, replay, million-record, dense-cell, and overflow gates
      also pass.  This lands the integration mechanism, not a whole-run speedup
      claim: host packing and CPU publication are still charged outside the DSO,
      and the next measurement must capture/replay the dominant M2 enclosure
      stream before widening the seam or building the persistent broker.
      Promotion beyond the experimental `cuda` branch also requires a clean
      full KLayout rebuild and an exact CPU/CUDA report gate; the copied
      incremental diagnostic tree was rejected after exposing stale mixed DB
      symbols rather than being treated as integration evidence.

    - [x] **Clean CUDA/KLayout integrity gate and real M2 capture — completed:**
      a fresh build at `b18ff59` produced byte-identical CPU, CUDA, and preserved
      stock-master nonempty reports (raw SHA-256
      `5d0f089a554cd015a5bc24357c3a7fbfe6306d9e7059b3df6b1a69907fb5ceb3`;
      157 categories, two cells, 99 items).  The opt-in `KEDGER1` recorder now
      captures endpoints, full 64-bit properties, effective rule metadata,
      post-property-gate broad pairs, and the independent CPU exact-predicate
      oracle per scanner request.  A validator/analyzer checks the complete
      binary format.  The 64x1024 M2 head-check yielded 6,485 valid requests;
      775 requests with at least 1,024 records contained 98.4% of recorded
      scanner time.  The x2 capstone large-request capture yielded 3,086 valid
      requests, 11,727,284 records, 8,024,490 relevant broad pairs, and
      2,650,190 exact pairs with no unresolved records.  The first recorder
      draft cost 3.1% even while disabled.  Lazy record discovery removed its
      per-edge work, and moving the large receiver into a cold helper restored
      the original 56-byte hot stack frame.  An eight-vs-eight, core-swapped
      crossover now measures only +0.46% wall / +0.50% user time, below the 2%
      pursuit floor; all 16 reports are byte-identical.

    - [x] **Fused projection-overlap edge predicate — completed standalone:**
      the CUDA AABB enumeration now optionally evaluates the exact nondegenerate
      Manhattan subset of the FreePDK45 enclosure predicate before sort/dedup.
      Unsupported geometry passes through to CPU; real `KEDGER1` exact oracles
      guard against false negatives.  The largest 64x1024 request falls from
      3,634 relevant broad pairs to exactly the CPU oracle's 518 pairs
      (**85.7% fewer downstream pairs**).  The million-edge gate falls from
      1,337,218 to 79,838 unique pairs (**94.0% fewer**) and its charged GPU
      pipeline is 4-5% faster across observed runs.  These are component
      results, not KLayout
      whole-run claims.

    - [x] **Cluster-incidence acceleration — measured and rejected:** a
      temporary counter gate observed 2,461,725 candidate implant incidences,
      but only 28,082 duplicates (1.14%); every duplicate was an already-seeded
      self-exclusion, with zero generated-vs-generated repeats for a GPU
      sort/unique to collapse.  Even zero-cost removal models only 2.30% of the
      implant/contact shard and 0.27% of M2 before packing, transfer, kernel, or
      synchronization overhead.  The instrumentation and an earlier
      below-0.91%-ceiling map-hint trial were both reverted.

    - [x] **Persistent/broker gate — bounded; production broker deferred:** an
      aggregate replay isolated 775 real requests by context and globally
      rekeyed 1,490,138 records.  GPU output exactly matched both the 1,023,189
      broad-pair oracle and the 334,887 exact-pair oracle.  After warmup, the
      charged GPU pipeline averaged 123.495 ms; host packing plus GPU averaged
      214.910 ms versus 1,017.226 ms for the capture-instrumented scanner
      (**78.9% less component time**).  Header-first selection reduced offline
      capture loading from 30.436 s to 0.473 s.  However, the capstone records
      only 8.284 s of this scanner work inside the 156.69 s critical M2 shard:
      even extrapolating the favorable component ratio models roughly 4.2%
      less M2 wall time, below the +10% whole-run integration gate.  Do not
      build the AF_UNIX/shared-arena production broker unless a wider measured
      seam first raises the charged opportunity above that threshold.

    - [x] **Cross-shard EdgeProcessor phase census:** an opt-in, bounded
      `KLAYOUT_EDGE_PHASE_PROFILE_MIN_EDGES` probe now separates preparation,
      intersection discovery, cutpoint splitting, and stateful production
      without changing the disabled algorithm.  All 184 EdgeProcessor test
      executions pass; Manhattan, diagonal, and redo probes account exactly
      for their measured phase totals.  A five-pair, ten-repeat disabled-path
      stress gate measured 10.637 s control versus 10.453 s instrumented means,
      so no regression is visible (the favorable difference is not booked as
      a speedup).

      On the x2 capstone, every one of the 16,508 captured M1/M2 calls was
      Manhattan.  M1 accumulated 47.642 s of phase time: 25.723 s
      intersections, 2.118 s splitting, and 19.749 s production.  M2
      accumulated 32.003 s: 15.143 s intersections, 0.673 s splitting, and
      16.152 s production.  These clocks sum work across four worker threads
      and are **not** removable shard or whole-launch seconds.  They establish
      a materially wider device-neutral target for an exact Manhattan
      intersect/split replay, with stateful production as a second kernel only
      if charged A/B evidence warrants it.
    - [x] **Reject the disconnected-only CONTACT result:** the first
      `DeepEdges` certificate accepted only an empty hierarchy-interaction set,
      but the real 64K CONTACT workload was not disconnected.  In the valid
      same-binary isolated A/B, control averaged 25.100 s and the attempted
      certificate averaged 25.377 s, **0.277 s / 1.1% slower**, with identical
      reports.  That result was rejected rather than booked as a win.

    - [x] **Diagnose the apparent hierarchy interactions:** all **201,643**
      returned marker pairs were byte-identical coincident boxes; there were
      zero boundary-only, containment, or partial-area-overlap pairs.  The raw
      GDS has 335,805 duplicate CONTACT-box pairs and local preprocessing
      removes 134,162 of them, exactly accounting for the remaining 201,643.
      They are real duplicate contact rectangles, not AABB false positives.

    - [x] **Prove a strict duplicate-rectangle selected-empty result:** the
      certificate now accepts a nonempty GPU pair set only for a true-only,
      merged-semantics concrete `EdgeLengthFilter`, with no breakout, complex
      transform, nonzero property, overflow, capacity failure, or source-cache
      ambiguity.  Local EdgeOr must preserve the exact oriented edge multiset
      including multiplicity, every provenance group must be exactly four full
      axis-aligned rectangle sides, the filter must reject every canonical
      edge, the GPU must return complete Success, and every interacting
      transformed box must be byte-identical.  It then returns a fresh merged
      empty result without publishing a source merged cache; every unmet guard
      falls back to the legacy CPU path.

    - [x] **Measure the real 64K CONTACT win:** three same-binary control runs
      averaged **25.040 s** and three certificate runs averaged **18.697 s**:
      **6.343 s / 25.3% less CONTACT-shard wall time**.  All six reports were
      identical with SHA-256
      `e29559d81e17f525a7978e8e402235dfa3a7f6ef408f3c7adbe7ef74d76b3c5e`.
      The GPU self request was about 52 ms and the complete proof decision about
      305–309 ms; those component timings are already charged in the candidate
      wall time.

    - [x] **Close the correctness gates:** the focused GPU integration gate
      passed all 10 certificate tests, and the complete `dbDeepEdgesTests`
      matrix passed **70/70** (35 non-editable plus 35 editable).  A nonempty
      CONTACT-violation sentinel retained exactly four CONTACT.1 edge markers
      and matching raw/semantic reports, so the shortcut cannot erase a real
      selected-length violation.

    - [x] **Keep whole-run accounting honest:** this 6.343 s CONTACT-shard
      reduction currently removes **0.000 s** from the 156.692 s parallel
      full-launch wall because M1 remains the critical lane (145.540 s versus
      M2 at 138.225 s).  It creates useful CONTACT slack and can become a
      whole-run saving only after the critical lanes move or at larger scale.

    - [x] **Run the downstream-composed x2 CONTACT scale gate:** the
      same-binary control took **218.32 s** and the certificate candidate took
      **155.10 s**, removing **63.22 s / 28.96%** of this x2 CONTACT-shard wall.
      Reports were identical with SHA-256
      `9511c638ae7ed175e9bca5ece71068602b6c804bd230aecc5e56fa3cbefad305`.
      The 369.611 ms GPU self request and 2,412.654 ms complete certificate
      decision are charged inside the candidate wall.  Evidence:
      `/home/pullin/personal/klayout/cuda-evidence-temp/contact-x2-duplicate-empty-ab-20260723T154225Z`.
      This is explicitly a downstream-composed x2 CONTACT-shard result, not a
      63.22 s saving from the original 156.692 s parallel full launch.

    - [x] **Reject cuSpatial as the production geometry substrate:** its closest
      join maps bounding boxes to leaves of a point quadtree, not KLayout's
      context-qualified AABB/AABB or edge/edge joins.  Its geometry predicates
      use floating-point GIS semantics and do not implement exact signed-int64
      thresholds, KLayout projection/orientation modes, partial markers,
      shielding, or hierarchy publication.  The final 25.04 release is
      archived, and even the header API adds RMM while the full library adds
      cuDF.  Keep using the maintained NVIDIA layer that actually fits this
      workload—CCCL/CUB/Thrust—and implement the small exact integer kernels
      directly.  Algorithmic ideas may be borrowed; no cuSpatial dependency
      should be introduced.

    - [x] **Reject two more M1 deck-only shortcuts:** local POLY.3/.4
      edge-pair-to-polygon fusion changed the x2 M1 wall from 141.09 s to
      140.00 s, only **0.77% less wall time**, below the stopping threshold.
      It also exposed a general magnified-hierarchy defect: the compound
      edge-pair-to-polygon wrapper drops the child check's
      `MagnificationReducer`, so the fused form cannot be claimed equivalent
      without a C++ reducer-composition fix.  The guarded local METAL1.3
      rectangle filter was exact on the adversarial 21-item hierarchy fixture,
      but the clean x2 probe produced 2,125,548 flat intermediate markers and
      correctly fell back.  It regressed wall from **141.73 s to 445.99 s**;
      keep the current guarded reversed-empty path.

    - [x] **Qualify ACTIVE.3 as the first device-resident empty-result target:**
      `well.enclosing(active, 55nm, euclidian)` took 41.58 s in the isolated
      profiling microscope and published zero flat and hierarchical edge
      pairs.  A separate no-shield (`transparent`) census also published
      exactly zero pairs in 39.01 s wall and produced the same normalized
      report.  The two timings are not an A/B performance comparison—the
      shielded run carried `perf record` and ran beside other probes—but the
      no-hit result proves that the first GPU certificate need not implement
      shielding for this workload.  The input has 24,687,816 flat ACTIVE
      polygon occurrences represented by only 716 stored polygons; WELL has
      1,964 flat occurrences and 1,074 stored polygons.  Preserve that
      hierarchy compression rather than serializing the flat universe.

    - [ ] **Move the CUDA ownership boundary outward to a fused DRC plan:**
      the successful ngspice-CUDA project showed that narrow device-evaluator
      and solver-only seams lose to synchronization and Amdahl's law, while a
      resident evaluator/assembly/solve/control pipeline delivered 8.44x
      analysis and 6.73x total speedup.  Apply the same lesson here.  CPU setup
      should lower immutable cell, instance-array, transform, edge-template,
      rule, stable-ID, and report-ownership tables once.  GPU work should retain
      those tables through hierarchy frontier expansion, spatial candidate
      generation, exact predicates, rule-specific waiver/reduction, stable
      sort/dedup, and survivor compaction.  Return only compact final marker
      descriptors (zero records on a clean rule), never the raw candidate
      stream.

      Start with the exact ACTIVE.3 no-hit certificate, then reuse the same
      device IR for METAL1.3's four-bit deficient-side reduction.  The latter
      culls masks 0, singleton, and two-opposite and compacts only disallowed or
      uncertain stable contact IDs.  The first KLayout integration may consume
      only a complete zero-survivor result; the ABI and kernel must nevertheless
      support bounded compact survivors so empty-only behavior is not baked
      into the engine.  Unsupported transforms, properties, breakout cells,
      overflow, queue/capacity exhaustion, or incomplete traversal must fail
      closed to the unchanged CPU operation.  A general nonempty backend may
      require several thousand lines and is not rejected on code volume alone;
      exact oracle gates and whole-run savings decide whether each slice lands.

      - [x] **Land the bounded METAL1.3 device-side analyze/cull kernel:** CUDA
        now performs the context-qualified sparse broad phase, strict
        projection-distance classification, four-side mask reduction, waiver,
        deterministic sort, and survivor compaction without returning raw
        pairs.  The additive ABI preserves the old v1 entry points.  Its
        synthetic gates cover all 16 masks, the 34/35/36 threshold, contexts,
        partial and non-Manhattan uncertainty, signed-coordinate extrema,
        malformed requests, duplicate IDs, and capacity fallback.  Independent
        review caught and fixed two initially fail-open internal-invariant
        branches before commit; the hardened suite passes CUDA memcheck with
        zero errors.  This is commit `93c51a8` on `fork/cuda`, not a production
        KLayout speedup: a caller must still prove and fingerprint the complete
        hierarchy/edge universe.

      - [x] **Make the exact ACTIVE.3 derived scene reproducible:** the capture
        harness derives `nwell.or(pwell)` and ACTIVE through the production
        deep engine, uses `DeepRegion#insert_into` to retain hierarchy, replays
        the rule from the captured operands, records a non-flattening census,
        and publishes only after source/replay checks pass.  Clean tiny and 64K
        gates pass; a deliberate nonempty fixture retains 4/4 markers.  The x2
        scene represents 24,687,816 logical ACTIVE polygons with 716 stored
        shapes and 570,294 instance records.  This is commit `8148bf5` on
        `fork/cuda`; hashes are provenance, not hard-coded correctness values.

      - [x] **Lower the ACTIVE.3 capture to deterministic device POD:** the
        versioned `KACTSCN1` format preserves the cell/instance/array DAG,
        directed Manhattan contours, layer bounds, and scene fingerprint in
        bounded little-endian records.  Independent validation rejects
        malformed, unsupported, truncated, or hash-mismatched scenes.  This is
        commit `40462b3` on `fork/cuda`.

      - [x] **Qualify the exact ACTIVE.3 device predicate:** the fixed 55 nm
        Euclidean enclosure relation now has a shared CPU/CUDA implementation
        with fail-closed uncertainty, exhaustive lattice,
        randomized/extreme-coordinate, and direct KLayout differential gates.
        This is commit `038623d` on `fork/cuda`.

      - [x] **Join the bookends in a standalone contiguous ACTIVE.3 GPU
        island — completed experimentally:** after checked packed-scene loading
        and host context lowering, the device owns WELL expansion and indexing,
        streams 98,754,896 transformed ACTIVE edges without materializing a
        flat edge or candidate array, evaluates the exact predicate, and
        returns only bounded counters/diagnostics.  Five fresh x2 processes had
        an external median of **0.70 s**; the ACTIVE query itself took
        **23.24 ms**, with identical zero-raw-hit/zero-uncertain censuses.
        Against the roughly 41.58 s isolated CPU observation, that is
        contextual evidence of about **40.88 s / 98.3% less wall time (roughly
        59x)**.  It is not yet a same-binary production A/B, M1-shard saving,
        or full-run result: the executable starts from an already captured
        derived scene and has not replaced KLayout's live operation.  This is
        commit `c3ceed2` on `fork/cuda`.  Follow-up commit `b1fd3da` corrected
        the result contract: local ACTIVE union can remove a raw inner-edge
        hit, so only zero raw hits are consumable; every raw hit requires
        pristine CPU fallback.

      - [x] **Integrate at the live ACTIVE.3 operation seam — completed,
        isolated-process scope:** the default-off KLayout hook now lowers the
        live merged-WELL/raw-ACTIVE DeepShapeStore hierarchy into digest-bound
        POD and calls the additive CUDA entry point in-process.  Only a complete
        zero-raw-hit result returns the already-created empty `DeepEdgePairs`;
        every hit, uncertainty, unsupported shape/transform/property, malformed
        echo, capacity limit, loader error, or exception runs the unchanged CPU
        processor.  Three fresh same-binary isolated controls had external
        walls **55.25/55.37/54.80 s**; three frozen candidates had
        **2.99/2.95/2.96 s**.  The medians are **55.25 -> 2.96 s**, removing
        **52.29 real seconds / 94.6% of process wall** (about 18.7x).  All
        reports have canonical SHA-256
        `681cd5f31b2407672e760f718a827721f15a9193f2f7464c12d5ce610d961ed7`.
        Peak host RSS also fell from about 2.31 GiB to 0.55 GiB.  This is an
        isolated derived-scene replay result; the raw-layout M1 result is
        recorded separately below.

        The live and standalone operand universes match exactly: 284 WELL
        contexts / 8,924 edges, 788,174 ACTIVE contexts / 98,754,896 edges, and
        44,623,826 candidates.  Live deep extraction prunes 780 contexts
        containing neither operand, explaining its 848,485 versus packed
        849,265 total without omitting geometry.  Deliberate nonempty and
        raw-ACTIVE-union counterexamples both forced exact CPU fallback;
        nested 3x2 arrays under all eight transforms passed; normal and live x2
        Compute Sanitizer runs reported zero errors.  Independent review
        returned SHIP after fixing a near-`uint32` device-loop wrap.  Commits
        `766871b`, `ee91e1b`, and `a318f6e` are pushed to `fork/cuda`.

      - [x] **Confirm the live certificate inside the actual x2 M1 lane:**
        one isolated same-binary raw-layout A/B at `a318f6e` changed
        `m1_enclosure` from **255.06 s to 203.12 s**, removing **51.94 real
        seconds / 20.36% of lane wall**.  The integration build is slower than
        the specialized formal PGO binary, so this comparison is deliberately
        against its immediate feature-off control rather than the older
        145.5-second PGO record.  Raw reports are byte-identical with SHA-256
        `dd1199719a17c460a188bd05581e294fae597060cfc4cd75c5b53fba81f2c289`;
        generator-stripped reports also match.  The candidate returned the
        exact 44,623,826-candidate clean census with 189.72 ms live lowering,
        267.15 ms backend time, and zero hit/uncertainty/fallback/device flags.
        External process wall is authoritative: KLayout's verbose per-operation
        `Elapsed` is aggregate CPU-like time and is not used as a wall
        denominator.  This first qualified pair is not yet a repeated
        statistical M1 result or a parallel full-launch measurement.

      - [x] **Prove a reusable device-resident VIA1 sandwich on both metal
        boundaries — completed experimentally:** exact packed-scene oracles
        now retain the raw M1/VIA1 and M2/VIA1 hierarchies and execute the
        complete production projection-enclosure chains.  CUDA expands the
        20,178,022 logical VIA1 occurrences once, builds a cut self-grid,
        accepts only exact coincident raw duplicates, rejects every other
        touch/overlap, proves strict 75 nm Euclidean spacing, and streams the
        enclosing-metal hierarchy through an exact positive containment
        certificate.  Simple hole-free Manhattan metal polygons are lowered
        to both exact X- and Y-slab rectangle subsets; unsupported cuts,
        malformed scenes, overflow, capacity exhaustion, incomplete traversal,
        counter mismatch, or any positive uncertainty fails closed.

        Both full x2 scenes certify every VIA with zero enclosure misses,
        unsafe overlaps, spacing violations, or device flags.  M2 expands to
        22,947,380 metal rectangles; M1 expands to 41,109,338.  Five fresh
        standalone processes give a **96.601 ms median M2 GPU plan** and
        **0.633129 s median warm standalone total**, versus the contextual
        126.49-second exact CPU METAL2.3-chain observation: about 125.857 s /
        99.499% less wall and 199.8x throughput.  M1 gives a **118.416 ms
        median GPU plan** and **1.265037 s median warm standalone total**,
        versus the 62.66-second capture/oracle run containing the 58.010-second
        enclosing and 2.930-second width stages: about 61.395 s / 97.981% less
        wall and 49.5x throughput.  These are standalone derived-scene
        comparisons, not a same-binary live or whole-run A/B; packed-scene
        validation and host hierarchy lowering are charged, while live source
        extraction and deck integration are not yet measured.

        The same proof state implies six clean categories when fused:
        METAL1.4, METAL2.3, and VIA1.1--.4.  Independent M1 and M2 medians sum
        to 215.017 ms; reusing the resident VIA expansion/grid/pair pass should
        reduce the combined call further, but that saving remains a projection
        until measured.  Fifteen deterministic/adversarial gates cover exact
        enclosure distance, X/Y slab witnesses, arrays and all orthogonal
        transforms, exact duplicates, nonidentical touching fallback, axial
        and diagonal 75 nm boundaries, missing enclosure, corrupt input, and
        nonrectangular-cut decline.  CUDA memcheck reports zero errors on the
        duplicate and spacing branches.  Capture, island, gate, and hardening
        commits `2385939`, `5b9fd3c`, `8e47aa0`, `5906b80`, and `6df69d6` are
        pushed to `fork/cuda`.

      - [x] **Integrate the six-rule VIA1 sandwich as one atomic live plan:**
        serialize raw M1/VIA1/M2 from their shared DeepShapeStore without
        mutating or merging the CPU layers, invoke one optional DSO symbol,
        retain VIA boxes and their CSR grid while reusing one metal scratch
        allocation for M1 then M2, and return only a digest-bound six-bit clean
        mask.  The deck may consume the result only when all six bits are
        certified; every partial result, unsupported scene, error, or capacity
        decline must run all six historical CPU chains unchanged.  Move those
        categories into one existing shard owner rather than adding a process.
        Qualify six deliberately nonempty sentinels, backend-missing/error
        fallback, nested hierarchy, exact report equality, and a full x2
        same-binary A/B before claiming any whole-run saving.
        **Delivered:** the opt-in path preserves the original shard ownership
        when disabled and moves all six rules into one existing owner only when
        requested; every noncertificate runs the complete local CPU stack.
        On x2, the fused backend handled 849,265 contexts, 41,109,338 M1
        rectangles, 20,178,022 VIA occurrences, 22,947,380 M2 rectangles, and
        672,673,286 VIA candidate pairs in 537.582 ms of device work after
        1.172 seconds of live host lowering.  Against the original eight-shard
        deck, `m1_via_class` fell 259.526 -> 59.832 seconds (**76.95% less**),
        `m2_rules` 230.567 -> 100.867 seconds (**56.25% less**), and
        `via1_upper_active12` 187.276 -> 59.982 seconds (**67.97% less**).
        Co-locating the same six CPU rules for the isolated owner comparison
        took 570.79 seconds versus 58.74 seconds with CUDA (**89.71% less**,
        512.05 seconds saved).  The honest full-launch comparison was only
        278.20 -> 275.97 seconds (**0.80% less**, not treated as a whole-run
        win) because the unchanged `m1_enclosure` lane remained critical.
        Canonical reports were byte-identical with SHA-256
        `01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`.
        Seventeen live geometry/fallback cases, 49 adversarial host-ABI cases,
        both device smokes, independent host/device audits, and CUDA memcheck
        all pass.  Implementation and gate commits `8e911a4` and `eff6057` are
        pushed to `fork/cuda`.

      - [x] **Integrate the qualified live CONTACT/METAL1.3 certificate:**
        reuse the atomic projection backend with raw M1/CONTACT hierarchy
        lowering, a single-rectangle fast path, and an exact integer-DBU
        union-strip proof for split M1.  The same-binary x2
        `m1_enclosure` lane changed 202.73 -> 88.06 seconds (**56.56% less**,
        114.67 seconds saved).  The complete host-to-host transaction took
        3,063.94 ms and the CUDA backend reported 235.58 ms; canonical reports
        were identical with SHA-256
        `d056b808e6f2134e60286e35247a92e3a2f6d2eaa26b463fd572aa7e0652146d`.
        Fifteen live geometry/fallback cases, the existing seventeen-case VIA
        regression, backend/oracle smokes, negative/grid-boundary and one-DBU
        gap cases, and CUDA memcheck pass.  Implementation and gate commits
        `4728aa4` and `3d144c8` are pushed to `fork/cuda`.  This is an
        independently lowered certificate, not yet the resident fused tail
        below; do not report the lane saving as a full-launch reduction.

      - [x] **Reap the live CUDA wins with CPU-owner balancing:** stack the
        qualified three-way antenna split, leave the compound METAL1.1/.2
        traversal intact, and move only independent `CONTACT.6` from the
        overloaded M1 width/space owner into the underloaded grid owner.
        Three same-binary all-CUDA baselines had full-launch walls
        208.568190, 209.288711, and 210.616450 s; three balanced candidates
        took 184.556784, 186.398209, and 184.242187 s.  The means are
        **209.491117 -> 185.065727 s: 24.425390 real seconds / 11.66% less
        full wall time and +13.20% throughput**.  Baseline and candidate
        full-range spreads are 0.98% and 1.17%.

        Child-plus-merge means are 204.965780 -> 180.514787 s (**11.93%
        less**).  M1 width/space becomes the sole pole at a 180.507539 s mean,
        down from 204.958218 s.  The antenna pole falls from 191.356083 s to
        96.299579 s for the slowest new owner (**49.68% less antenna-lane
        wall**); grid rises only to 45.032370 s.  This is the first
        configuration-level result that converts the large independent CUDA
        lane reductions into a double-digit end-to-end win.

        All six clean reports are canonically identical at SHA-256
        `01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`;
        all three candidate raw merges are byte-identical at
        `89fa723caf5ecd620993d62c2d2e63ca6eaf8c14fe0dac25a7abf9c2992c1472`.
        CUDA-enabled shards also match CPU `all` exactly on the 86-item
        antenna fixture, 99-item M1 sentinel, and 1,587-item mixed hierarchy,
        including the moved CONTACT.6 marker and retained M1.1/.2 markers.
        Their semantic SHA-256 values are respectively
        `409085bc15614329421b1b07a6fdd6b867fe8cad83a214a1f32a5d9986a595a4`,
        `265a2e1c58aed60bc89e8bd2ad2904147a1e53fcd7a2a7c8bfa8ddaa339cd1a2`,
        and
        `429d631ab89d9e0a54e4f6ed367223974b8caca564c461fda59ebda9dcbcd8d2`.
        The transformed CUDA deck and bound manifest SHA-256 values are
        `5e32231a9232a98b4bfc7475e2f9a9400d967787734072e5678c870ffde68573`
        and
        `5145c61ca568c05d82675f700f6b95c3b7135880adf081f90e5692492538c59f`.
        The reusable fail-closed transform lives under
        `benchmarks/freepdk45_contact6_split/` on `perf`; its grid mode is the
        CUDA-specific balancing choice.

      - [x] **Rebuild the CUDA host with the production CPU configuration:**
        a clean source-bound Clang 22, full-LTO/LLD, `znver2` control build
        reused the identical CUDA backend, deck, manifest, x2 input, ten-owner
        order, and eight-job limit.  The preceding balanced means are the
        immediate comparison.  Three new full-launch walls were 155.075069,
        156.075237, and 156.136431 s, giving
        **185.065727 -> 155.762246 s: 29.303481 real seconds / 15.83% less
        full wall time and +18.81% throughput**.  Candidate full-range spread
        is 0.68%.

        Child-plus-merge changed 180.514787 -> 147.723113 s (**18.17% less**).
        M1 width/space remains the pole at a 147.715584 s mean, down from
        180.507539 s (**18.17% less**); implant/contact follows at 132.263260
        s, down from 160.457722 s (**17.57% less**).  All three raw reports are
        byte-identical at SHA-256
        `89fa723caf5ecd620993d62c2d2e63ca6eaf8c14fe0dac25a7abf9c2992c1472`
        and all canonical reports retain
        `01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`.
        Every counted run produced the ACTIVE.3, VIA1-stack,
        CONTACT/METAL1.3, and selected-empty certificates.

        The 29:26.31 build is bound to commit
        `f15739559610a8c31b21bf47a185ee22eaeded86`, tree
        `3f5ec23262ed2a2797f5b8a6f47d5fa0ff979337`, executable SHA-256
        `007f6ddc0750d7b6b8d6c8287756a2e7a94dca298e5a4ded0c4e4bb6cb5cc252`,
        build-manifest SHA-256
        `26b7137e6fe9ff45ed0051e302d255af7fc8c1cbc4ff39eef1269fc6f3982fce`,
        and lifecycle-state SHA-256
        `3a9383a9df70cf57df21425941ee83b1eb36aa6beacbefe69d2cf238f0aedb78`.

      - [ ] **Retrain mixed PGO for the current CUDA host:** no profile from
        the pre-CUDA source is valid for this tree.  If the production-build
        result clears the whole-run gate, generate fresh profiles with one
        balanced FreePDK45 CUDA x2 run and one real Sky130 S5 run, weight by
        measured counter totals, build a source-bound PGO-use bundle, and
        require at least 5% on both one-pair screens before formal cohorts.

      - [ ] **Fuse an ACTIVE.3-through-METAL1.3 resident tail plan:** upload
        WELL/ACTIVE/CONT/METAL1 and hierarchy once, keep candidate generation,
        exact predicates, METAL1.3 guards, four-side reduction, and per-rule
        clean/fallback state on device, and return one terminal result.  Source
        and profile mapping projects roughly 95--100 seconds of the 145.5-second
        M1 lane in this interval, but that is overlapping opportunity
        accounting, not an additive measured saving.  The existing ACTIVE.3
        and METAL1.3 kernels are the two proven endpoints.

      - [ ] **Widen the resident plan across the post-derived M1 interval:**
        after CPU construction of WELL and GATE, keep POLY.1/3/4/5/6,
        ACTIVE.3, and METAL1.3 on device through terminal per-category culling.
        This exposes about 95% of profiled M1 operation work without first
        porting polygon Boolean construction.  POLY.3/4 require exact
        edge-pair normalization, edge-pair-to-polygon conversion, zero-area
        culling, and stable hierarchy ownership; a raw-hit-zero certificate is
        insufficient.

      - [ ] **Move hierarchy lowering onto the GPU and amortize residency:**
        replace the current host context lowering with a bounded device
        BFS/wavefront, then add a fingerprint-keyed resident scene cache with a
        persistent CUDA context and reusable buffers.  Charge cold load,
        validation, upload, queueing, fallback, and teardown separately from
        warm reuse.  Extend the plan boundary leftward to raw layers and perform
        WELL/GATE/FIELD-POLY Boolean construction on device only after the
        post-derived executor is exact and beneficial; this is the whole-M1
        endgame rather than a prerequisite for the next measured win.
15. [ ] **Lean headless DRC build and developer-turnaround path**
    The accepted runtime is headless, but the standard build still compiles the
    full KLayout distribution.  The current graph has 1,877 object files; 724
    are Qt bindings that the batch DRC solver does not use.  First measure the
    already-supported, low-risk configuration
    `-without-qtbinding -nopython -nolstream -nolibgit2` while retaining Qt,
    Ruby, the normal `klayout` executable, and GDS/OAS readers.  Inventory
    predicts about 1,068 objects, **43.1% fewer compile actions**, but this is
    explicitly not a wall-time claim.  Qualify it on Ruby DRC smoke, both
    nonempty sentinels, FreePDK45/x2, Sky130 S5, and GDS/OAS format coverage
    before using it for iteration builds.

    If measured build turnaround warrants a source change, add a fail-closed
    `-without-tests` qmake guard.  Roughly 206 test objects remain after the
    supported reductions, for a projected total near 859 objects or **54.2%
    fewer compile actions** than the full graph.  Keep release/full-feature
    builds unchanged.

    A genuinely Qt-free solver is a separate second stage, not a build flag
    flip: `-without-qt` omits the normal `klayout` executable.  Base a dedicated
    `drc-run` target on `strmrun`, retain TL/GSI/DB/Ruby/LYM/RDB/DRC plus only
    GDS2/OASIS plugins, add non-Qt Expat support, and decouple embedded DRC QRC
    generation from runtime Python.  Prove that it does not link Qt, Python,
    GUI, LIB, LVS, or PEX before repeating the exact runtime ladder.  This is a
    build-productivity project and must not be mixed into PGO runtime claims.
16. [ ] **Stable-key hierarchy translation cache**
    `interaction_registration_shape2inst::add_shapes_from_intruder_inst()`
    currently materializes and hashes a transformed polygon before discovering
    that the same source shape and composed transform was already translated.
    Its receiver is 8.14% of the post-PGO M1 profile, 4.61% of implant/contact,
    and 1.88% of M2.  A symbolized non-LTO microscope attributes about 4.91
    profile points to polygon transform, hash, equality, lookup, and insertion;
    those non-LTO points are diagnostic attribution, not a runtime claim.

    First instrument a bounded stable-key cache keyed by source identity,
    composed transform, and properties.  On a hit, reuse the existing ID; on a
    miss, retain today's transformed-geometry lookup so geometrically equal
    shapes still deduplicate exactly.  KLayout's reverse inst2shape path already
    uses `shape_reference_translator_with_trans` as a model.  Proceed to a
    production patch only if the measured hit rate and charged lookup cost
    model at least +5% whole-run opportunity, or if a smaller 2-5% result is
    demonstrably low risk.

    A secondary, overlapping trial is a no-update bounding-box converter after
    an explicit layout update.  The non-LTO build exposes 7.46 profile points
    in repeated `Layout::update()`/dirty checks, but the PGO binary attributes
    only about 1% directly to named bbox conversion in M1.  Do not book that
    ceiling until path-specific counters separate genuine repeated work from
    PGO inlining and the already-counted outer scanner.

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
