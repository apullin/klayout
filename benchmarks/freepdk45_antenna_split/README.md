# FreePDK45 antenna process split

This CPU-side benchmark transform replaces the single FreePDK45 `antenna`
owner with three independent KLayout processes:

- `antenna_feol`: WELL.1, WELL.4, VT.1, and ACTIVE.4
- `antenna_m1_m2`: METAL1 and METAL2 antenna checks
- `antenna_m3_m10`: METAL3 through METAL10 antenna checks

Passing `--split-upper` replaces the last owner with `antenna_m3` and
`antenna_m4_m10`.  The latter builds the required cumulative prefix through
M3 without evaluating M3, then evaluates M4 through M10.  `drc_shard=all`
still performs every historical connect/check in its original order exactly
once.

Generate a candidate deck with:

```sh
python3 benchmarks/freepdk45_antenna_split/split_deck.py \
  input.lydrc output.lydrc

python3 benchmarks/freepdk45_antenna_split/split_deck.py \
  --split-upper input.lydrc output-upper-split.lydrc
```

The transform preserves `drc_shard=all` order and fails if its expected source
sites differ. Each checking owner rebuilds its required cumulative connection
prefix; the upper owner does not execute lower-metal checks.

Pass `--split-upper` to replace `antenna_m3_m10` with independent
`antenna_m3` and `antenna_m4_m10` owners. This mode keeps the METAL2-to-METAL3
connection prefix shared by both upper owners without executing the METAL3
check in the METAL4-through-METAL10 owner.

Pass `--split-lower` to replace `antenna_m1_m2` with independent `antenna_m1`
and `antenna_m2` owners. The M1-only owner stops after its gate-to-M1 prefix;
the M2 and every upper owner build the cumulative M1-via1-M2 prefix. The lower
and upper modes compose:

```sh
python3 benchmarks/freepdk45_antenna_split/split_deck.py \
  --split-lower --split-upper input.lydrc output.lydrc
```

Use `scripts/merge_sharded_lyrdb.py manifest` with a trusted full report and
the complete shard union to create a new deck-bound manifest: ten shards by
default, eleven with either optional split, or twelve with both. Do not just
reassign the old manifest: the proof must include the nonempty hierarchical
fixture from `antenna_fixture.rb`, because the older FreePDK45 sentinels
contain no antenna markers.

KLayout emits antenna diagnostic tag declarations and tagged values in
process-dependent order. The merger treats those named fields as associative,
sorts them in its output, and still preserves positional values exactly. A
manifest proves the complete trusted tag universe, while a later layout may
emit any subset of it (including none for a clean run); unknown or conflicting
tags still fail closed. `semantic_report_hash.py` applies the same
normalization when comparing a trusted full report with the merged result.

Keep `--jobs 8` while the deck requests four threads per process. CLI shard
order is launch order, so schedule the long antenna owners in the first wave.

For the current eleven-owner CUDA gate, use a deck-bound eleven-owner manifest
and `run_balanced_full_gate.sh --split-upper-antenna --jobs 11`.  The accepted
32-core-budget screen used two inner threads per owner (22 requested workers).

## Qualified x2 result

The three-owner split was screened three times with the same qualified PGO
executable used by the preceding eight-owner baseline. The old single
`antenna` owner averaged 114.409553 seconds. The new critical antenna owner
(`antenna_m3_m10`) took 55.402531, 56.556236, and 56.764035 seconds, averaging
56.240934 seconds: **50.84% less antenna-lane wall time**. The current
CPU-only launch remains M1-bound, so its full provenance-launch mean changed
only from 156.691627 to 156.006005 seconds; do not claim that 0.44% difference
as a whole-run win.

All three clean merged reports match the trusted full report at semantic
SHA-256
`dd7b3a6f3c8303e105d5ac882261caf68f7f119da90ed40f801fb71800c46a47`
(157 categories, one cell, zero items). The nonempty hierarchical fixture's
full, split-`all`, and merged reports match at semantic SHA-256
`409085bc15614329421b1b07a6fdd6b867fe8cad83a214a1f32a5d9986a595a4`
(157 categories, two cells, 86 items).

The measured x2 input, transformed deck, fixture, and deck-bound manifest
SHA-256 values are respectively:

- `74911a2111a3421912e54538bf55cd12e50164f43cd1a8c47411602e64c91d98`
- `be0d3be9fa0fbfb2c0e4591d7e73a6a464b12384cb8d5debc20c4972cf7c8357`
- `ce8df8fb2f557b90e11692d03e46960761693e8fe38182b6baf03b0b92adb48b`
- `c62ea2fe69d4c65247b892bcd78eff959565f1e85080ec97a9e17ab0d93109f5`

## Optional upper split result

On the current x2 CUDA workload, the bounded 10-owner/three-thread control put
`antenna_m3_m10` at 96.431 s.  The eleven-owner/two-thread full gate put
`antenna_m3` at 57.465 s and `antenna_m4_m10` at 57.769 s: the critical
antenna lane is **38.662 real seconds / 40.09% shorter**.  Full wall remained
effectively flat at 101.36 -> 101.66 s because unchanged `m2_rules` became the
96.773-second pole.  Both full reports retained canonical SHA-256
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`.

A nonempty two-owner M3/M4 sniff also proved its exact eight-category union
against the former upper owner.  The original and split `drc_shard=all`
fixtures matched at semantic SHA-256
`409085bc15614329421b1b07a6fdd6b867fe8cad83a214a1f32a5d9986a595a4`
(157 categories, two cells, 86 items).
