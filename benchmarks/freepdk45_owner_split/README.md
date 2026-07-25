# FreePDK45 fine-grained owner splits

`split_deck.py` provides two independent, fail-closed CPU process splits for
the current shard-aware FreePDK45 deck:

- retain IMPLANT.1-.5 in `implant_contact` and move CONTACT.1-.5 to `contact`;
- move ACTIVE.1/.2 from `via1_upper_active12` to `active12`.

No rule expression changes. In `drc_shard=all` mode every selected predicate
is true, so the historical rule and output order remains intact. The transform
match-counts declarations, guards, dependencies, block boundaries, and output
sites, and parses the generated macro as XML before publishing it.

Generate either or both variants without changing the input:

```sh
python3 benchmarks/freepdk45_owner_split/split_deck.py \
  --split-implant-contact input.lydrc output.lydrc

python3 benchmarks/freepdk45_owner_split/split_deck.py \
  --split-active12 input.lydrc output.lydrc
```

## Nonempty CONTACT.1-.5 CPU integrity gate

`contact1_5_fixture.rb` creates one hierarchical top with six focused leaves.
At its 0.5-nm DBU:

- a 125-by-130 DBU contact violates CONTACT.1;
- two exact contacts with a 145 DBU gap violate CONTACT.2;
- an isolated exact contact violates CONTACT.3;
- exact contacts with a 5 DBU ACTIVE or POLY margin violate CONTACT.4/.5;
- overlapping NPLUS and PPLUS rectangles violate retained IMPLANT.5.

Run the focused gate against the current fully composed 12-owner deck before
the owner transform:

```sh
python3 benchmarks/freepdk45_owner_split/run_cpu_gate.py \
  --klayout /path/to/klayout \
  --deck /path/to/freepdk45-balanced-cuda.lydrc
```

The gate constructs a clean runtime environment with no `KLAYOUT_CUDA_*`
variables, then:

1. proves the source and transformed CPU `all` reports have identical semantic
   canonical models;
2. runs all 13 transformed owners;
3. creates a fresh deck-bound manifest from the trusted transformed `all`
   report and complete shard union;
4. merges through the production strict merger and proves exact canonical
   equality with `all`;
5. requires nonempty CONTACT.1-.5 and IMPLANT.5 evidence;
6. requires the `contact` category inventory to be exactly CONTACT.1-.5 and
   the `implant_contact` inventory to remain exactly IMPLANT.1-.5; and
7. checks every moved/retained item count and manifest owner explicitly.

KLayout and the merger serialize equivalent cell graphs and item sets in
different non-semantic orders. The canonical comparison therefore uses the
same validated category, tag, cell/reference, and item-fingerprint model as
the strict merger. It retains every meaningful item field and duplicate
multiplicity while normalizing only the generator path, cell encounter order,
item order, and XML formatting.

The focused manifest is temporary integrity evidence. A reusable production
manifest should still be generated with the antenna tag fixture so its dynamic
tag universe is represented.

Failed gates retain their temporary directory automatically. Pass
`--keep-work` to retain successful evidence as well.
