# FreePDK45 CONTACT.6 process split

This CPU-side transform removes `CONTACT.6` from the overloaded
`m1_width_space` owner without changing the already-qualified compound
`METAL1.1`/`METAL1.2` traversal:

- `m1_contact6`: `CONTACT.6`
- `m1_width_space`: `METAL1.1` and `METAL1.2`

Generate a candidate deck with:

```sh
python3 benchmarks/freepdk45_contact6_split/split_deck.py \
  input.lydrc output.lydrc
```

The transform is composable with
`benchmarks/freepdk45_antenna_split/split_deck.py`. It changes only the owner
predicate for the existing CONTACT.6 block, preserves the original
`drc_shard=all` execution order, validates exact source-site counts, and
parses the generated macro as XML before publication.

Keep the compound M1 batch intact. A direct full-input probe measured 17.52
seconds for the compound traversal, while spacing alone took 26.30 seconds
and separate width plus spacing took 29.84 seconds. Splitting those rules
would discard an existing shared-traversal win.

Exactness requires both retained nonempty gates:

- the 99-item FreePDK45 sentinel contains four `METAL1.1` markers and one
  `METAL1.2` marker;
- the mixed hierarchical fixture contains one `CONTACT.6` marker.

Also run the dynamic-tag antenna fixture when creating a manifest and compare
the clean x2 merge against its trusted full report. When composed with the
three antenna owners there are eleven processes; retain `--jobs 8` and four
KLayout threads per process to cap requested concurrency at 32 threads.
