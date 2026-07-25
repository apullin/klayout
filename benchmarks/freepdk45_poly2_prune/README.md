# FreePDK45 dead POLY.2 prune

The historical FreePDK45 deck computes:

```ruby
poly_sep_active = poly.separation(active, 140.nm, projection)
if poly_sep_active.polygons?
  # POLY.2 output
end
```

`separation` always returns an EdgePairs-backed DRC layer, while `polygons?`
is true only for Region-backed layers. The output branch is therefore
unreachable for every input, but the full separation still executes.

`prune_deck.py` removes exactly that known block and replaces it with a
documenting marker. It rejects source drift, duplicates, extra POLY.2 output
sites, malformed XML, and already-pruned decks.

Generate a candidate without mutating the source:

```sh
python3 benchmarks/freepdk45_poly2_prune/prune_deck.py \
  input.lydrc output.lydrc
```

Run the transform tests and the live DRC API contract gate:

```sh
bash benchmarks/freepdk45_poly2_prune/run_gate.sh /path/to/klayout
```

The runtime gate proves `separation.edge_pairs?`, disproves
`separation.polygons?`, and executes the historical conditional on a real DRC
layer to ensure its body remains unreachable.

## CUDA/balanced-deck composition

The qualified balanced CUDA launcher keeps this optimization independently
opt-in:

```sh
bash benchmarks/cuda_spatial_replay/run_balanced_full_gate.sh \
  ... \
  --prune-poly2
```

The live-CUDA rewrites are applied first.  The exact POLY.2 transform then
accepts that generated deck only if the historical dead block is still present
once, before the antenna and CONTACT.6 owner transforms run.  With no
`--prune-poly2`, the generated deck retains the source block byte-for-byte.

The launcher pins the prune implementation and the pre-prune generated deck in
addition to its normal source, final decks, manifest, executable, backend,
input, and report artifacts.  Its normal deck-bound manifest validation applies
to the final pruned-and-balanced deck.  Source drift therefore stops before
DRC, while a manifest for an unpruned deck cannot validate a pruned candidate.

## Qualified result

An ABBA comparison on the current CUDA-aware FreePDK45 downstream-composed x2
`m1_enclosure` owner measured controls at 67.27 and 67.90 seconds and
candidates at 60.01 and 60.75 seconds.  The means were 67.585 and 60.380
seconds: 7.205 real seconds saved, or 10.66% less wall time (N=2 per lane).
All four semantic report hashes were identical:

```text
67c0e2d632fe87fec7c7c20a66ca0816718c029d96702242b621f003600df434
```

This is an owner-lane result for that exact deck/build, not a separately
measured full-launch claim.
