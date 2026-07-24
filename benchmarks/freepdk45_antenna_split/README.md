# FreePDK45 antenna process split

This CPU-side benchmark transform replaces the single FreePDK45 `antenna`
owner with three independent KLayout processes:

- `antenna_feol`: WELL.1, WELL.4, VT.1, and ACTIVE.4
- `antenna_m1_m2`: METAL1 and METAL2 antenna checks
- `antenna_m3_m10`: METAL3 through METAL10 antenna checks

Generate a candidate deck with:

```sh
python3 benchmarks/freepdk45_antenna_split/split_deck.py \
  input.lydrc output.lydrc
```

The transform preserves `drc_shard=all` order and fails if its expected source
sites differ. Each checking owner rebuilds its required cumulative connection
prefix; the upper owner does not execute lower-metal checks.

Use `scripts/merge_sharded_lyrdb.py manifest` with a trusted full report and
all ten shard reports to create the deck-bound manifest. Do not just reassign
the old manifest: the proof must include the nonempty hierarchical fixture
from `antenna_fixture.rb`, because the older FreePDK45 sentinels contain no
antenna markers.

KLayout emits antenna diagnostic tag declarations and tagged values in
process-dependent order. The merger treats those named fields as associative,
sorts them in its output, and still preserves positional values exactly. A
manifest proves the complete trusted tag universe, while a later layout may
emit any subset of it (including none for a clean run); unknown or conflicting
tags still fail closed. `semantic_report_hash.py` applies the same
normalization when comparing a trusted full report with the merged result.

Keep `--jobs 8` while the deck requests four threads per process. CLI shard
order is launch order, so schedule the long antenna owners in the first wave.

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
