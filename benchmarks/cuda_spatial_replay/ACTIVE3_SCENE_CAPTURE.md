# ACTIVE.3 exact derived-scene capture

This harness turns the FreePDK45 ACTIVE.3 operation into a durable,
self-contained workload:

```ruby
well = polygons(3, 0).or(polygons(2, 0))
active = polygons(1, 0)
well.enclosing(active, 55.nm, euclidian)
```

The capture deck derives `well` exactly as the production deck does, then uses
`DeepRegion#insert_into` for both operands. That operation copies the deep shape
store into a new layout. It preserves the derived region's cell, instance, and
array representation rather than flattening every repeated polygon. It does not
promise byte-identical topology to the original source GDS.

The captured layout has one public top cell,
`KLAYOUT_CUDA_ACTIVE3_SCENE`, and this fixed layer map:

| Layer | Operand |
|---|---|
| 101/0 | `nwell.or(pwell)` |
| 102/0 | `active` |

## Capture, replay, and census

Pass every input, output, and top-cell name explicitly. Outputs must not already
exist:

```sh
bash benchmarks/cuda_spatial_replay/run_active3_scene_capture.sh \
  --klayout /path/to/klayout \
  --input /path/to/design.gds \
  --topcell DESIGN_TOP \
  --scene /tmp/design-active3-scene.gds \
  --source-report /tmp/design-active3-source.lyrdb \
  --replay-report /tmp/design-active3-replay.lyrdb \
  --census /tmp/design-active3-scene.census \
  --threads 4
```

The runner:

1. refuses missing, aliased, or pre-existing outputs;
2. captures the exact deep derived operands and runs ACTIVE.3 on the source;
3. re-runs the same rule on the captured scene;
4. records stored and logical-flat cardinalities without flattening;
5. checks the report category and source/replay item counts;
6. publishes outputs only after those checks pass; and
7. prints observed SHA-256 hashes for provenance.

KLayout selects the stream format from the `--scene` filename suffix. The
private staging file retains that suffix, so `.gds`, `.gds.gz`, `.oas`, and
other formats supported by the selected KLayout build are not mislabeled.

Replay starts from the already-derived `well` operand. It intentionally excludes
the original `nwell.or(pwell)` construction cost; the workload isolates the
hierarchical ACTIVE.3 enclosure/context computation.

An equal report-item count is a useful regression sanity check, not a complete
geometric proof for arbitrary failing layouts. Before using a nonempty report
as an accelerator oracle, compare normalized marker geometry and ownership as
well. The current no-hit workload is suitable for an exact empty-result
certificate only if the accelerator checks every relevant candidate.

## What the census proves

`stored_shapes` and `stored_edges` count geometry physically stored once across
reachable cell definitions. `flat_shapes` and `flat_edges` recursively apply
instance-array multiplicity without materializing flattened geometry.
`instance_records`, `array_records`, `array_elements`, `max_na`, and `max_nb`
make accidental flattening or array expansion visible. `instance_records`
includes scalar SREFs; `array_records` counts records with multiplicity above
one.

For orientation, two previously captured local scenes produced:

```text
# 64 x 1024 SRAM example
cells=132 instance_records=12610 array_records=4 array_elements=78187 complex_instances=0 property_instances=0 max_na=128 max_nb=256
well stored_shapes=283 stored_edges=1162 ... flat_shapes=726 flat_edges=3438
active stored_shapes=354 stored_edges=1440 ... flat_shapes=3139808 flat_edges=12561008

# downstream-composed 2 x (64 x 4096) SRAM example
cells=273 instance_records=570294 array_records=0 array_elements=570294 complex_instances=0 property_instances=0 max_na=0 max_nb=0
well stored_shapes=1074 stored_edges=4356 ... flat_shapes=1964 flat_edges=8924
active stored_shapes=716 stored_edges=2912 ... flat_shapes=24687816 flat_edges=98754896
```

Those figures are examples, not golden acceptance values. The second source
layout contains many individual references rather than GDS arrays, so its large
`instance_records` count is expected; the derived capture still retains those
references instead of expanding 24.7 million active shapes.

Likewise, hashes printed by the runner are examples/provenance only. Raw GDS and
LYRDB hashes can change with serializer versions, macro paths, report metadata,
or source naming even when geometry is equivalent. Do not hard-code them as
correctness criteria.
