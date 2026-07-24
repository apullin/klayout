# FreePDK45 CONTACT.6 process split

This CPU-side transform removes `CONTACT.6` from the overloaded
`m1_width_space` owner without changing the already-qualified compound
`METAL1.1`/`METAL1.2` traversal. The default creates an independent owner:

- `m1_contact6`: `CONTACT.6`
- `m1_width_space`: `METAL1.1` and `METAL1.2`

Generate a candidate deck with:

```sh
python3 benchmarks/freepdk45_contact6_split/split_deck.py \
  input.lydrc output.lydrc
```

When an accelerated configuration makes the existing `grid` owner
underloaded, coalesce CONTACT.6 there without adding a process:

```sh
python3 benchmarks/freepdk45_contact6_split/split_deck.py \
  --owner grid input.lydrc output.lydrc
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
the clean x2 merge against its trusted full report. When the independent owner
is composed with the three antenna owners there are eleven processes; the
grid-coalesced variant has ten. Retain `--jobs 8` and four KLayout threads per
process to cap requested concurrency at 32 threads.

## Qualified stacked CUDA result

The grid-coalesced mode was composed with the three-way antenna split and all
three existing live CUDA certificates: ACTIVE.3, the atomic VIA1 stack, and
CONTACT/METAL1.3. The KLayout executable, CUDA backend, input, accelerator
environment, and eight-job limit were held constant. Three preceding
eight-owner observations and three balanced ten-owner observations produced:

- full launcher: 209.491117 -> 185.065727 seconds mean, **24.425390 seconds /
  11.66% less wall time** and **+13.20% throughput**;
- child plus merge: 204.965780 -> 180.514787 seconds mean, **24.450993
  seconds / 11.93% less wall time**;
- M1 width/space: 204.958218 -> 180.507539 seconds mean;
- antenna: 191.356083 seconds for the single owner -> 96.299579 seconds for
  the slowest new owner, **49.68% less antenna-lane wall time**.

Baseline and candidate full-range spreads were 0.98% and 1.17% respectively.
The grid owner grew only to 45.032370 seconds and remained off the critical
path. All six clean reports are canonically identical at SHA-256
`01129a266f1ac2ef14e07def69fc26cc51dafe6beebe237e57c4dc68146a06d3`;
the three candidate raw merged reports are also byte-identical at SHA-256
`89fa723caf5ecd620993d62c2d2e63ca6eaf8c14fe0dac25a7abf9c2992c1472`.

The CUDA-enabled shard union was proven against CPU `all` mode on three
deliberately nonempty fixtures:

- antenna: 157 categories, two cells, 86 items, semantic SHA-256
  `409085bc15614329421b1b07a6fdd6b867fe8cad83a214a1f32a5d9986a595a4`;
- M1 sentinel: 157 categories, two cells, 99 items, semantic SHA-256
  `265a2e1c58aed60bc89e8bd2ad2904147a1e53fcd7a2a7c8bfa8ddaa339cd1a2`;
- mixed hierarchy: 157 categories, seven cells, 1,587 items, semantic SHA-256
  `429d631ab89d9e0a54e4f6ed367223974b8caca564c461fda59ebda9dcbcd8d2`.

The qualified executable, CUDA backend, transformed deck, deck-bound
manifest, and x2 input SHA-256 values are respectively:

- `978cbc79750f7ad97ae2dab26d662eb0a5271a713f97e7c512e0c7db67cad30d`
- `92abc5223c2be5589b2298ba42891afa4f26e589ca94227e3801b14faf11902d`
- `5e32231a9232a98b4bfc7475e2f9a9400d967787734072e5678c870ffde68573`
- `5145c61ca568c05d82675f700f6b95c3b7135880adf081f90e5692492538c59f`
- `74911a2111a3421912e54538bf55cd12e50164f43cd1a8c47411602e64c91d98`
