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
sorts them in its output, and still preserves positional values exactly.
`semantic_report_hash.py` applies the same normalization when comparing a
trusted full report with the merged result.

Keep `--jobs 8` while the deck requests four threads per process. CLI shard
order is launch order, so schedule the long antenna owners in the first wave.
