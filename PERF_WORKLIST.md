# KLayout performance worklist

This is the durable queue for the `perf` branch.  Update it whenever an item
is measured, accepted, rejected, or re-ranked.

## Current baseline and acceptance gate

- Workload: FreePDK45 `sram_1rw0r0w_64_1024_freepdk45_aref.gds`
- Binary: clang + full LTO, jemalloc, batch mode
- Previous accepted wall time: 146.50 s
- Current wall time: 128.68 s, 128.98 s; mean 128.83 s
- Current gain: **+13.7% throughput** with 0.23% run-to-run spread
- Baseline raw report SHA-256:
  `e9340a8d3cf3608cff5f87fe5c82aa9b122c215627dc9159a3d59f98bcaaedd0`
- Path-normalized baseline and batched report SHA-256:
  `74e8e6bb6d46e8574bcb8cdf159fee32f52dd49c0f6997db44b2645648357ba9`
- Iteration target: 3–5 minutes; retain only exact, repeatable wins

## Ranked work

1. **Rule-DAG scheduling and parallel rule groups — next**
   Schedule independent derived layers and checks concurrently while
   preserving deterministic report contents and resource bounds.
2. **Incremental antenna edge replay — re-profile first**
   The old estimate assumed ten cumulative rebuilds.  Empty-target pruning
   removed six, so its present ceiling is much smaller and must be measured
   before implementation.
3. **Intra-cell parallelism for remaining deep checks**
   Subdivide the few large serial geometry operations that limit ordinary
   deck-level thread scaling.
4. **Lazy interpreter initialization**
   Reduce the roughly 1.4 s batch startup floor for short jobs; separate from
   long-run geometry work.

## Recently completed

- Empty-target antenna pruning: 161.38 s to 146.50 s, **+10.2% throughput**,
  byte-identical report; commit `0873912` on `perf`.
- Homogeneous multi-output compound DRC: batch the matching M1 and M2
  width/spacing checks into one deep traversal while preserving independent
  report categories.  146.50 s to 128.83 s mean, **+13.7% throughput**,
  path-normalized byte-identical report.  Direct M1 batch: 17.15–17.24 s;
  direct M2 batch: 3.95–4.04 s.

## Rejected or bounded prototypes

- Persistent hierarchy-context caching: context formation was only about
  12–14 s for the entire deck and the contexts contain mutable propagated
  results without a reliable `Shapes` revision token.  Do not revive this as
  a cross-rule cache without a real invalidation design.
- Disparate-rule batching: the aggregate uses the maximum interaction border
  and union of all external inputs.  Batch only rules with similar distance,
  metrics and inputs; unrelated rules can increase preprocessing work.
- Mixed hierarchy reducers: sequential reducer composition can under-specify
  variants.  The batch API rejects incompatible reducer combinations rather
  than risking incorrect rotated or magnified hierarchy reuse.

## Guardrails

- Do not disable or weaken rules.
- Preserve report geometry, categories, descriptions, and diagnostic outputs.
- Test both flat and deep modes when engine semantics are affected.
- Record rejected prototypes so they are not rediscovered after context rolls.
