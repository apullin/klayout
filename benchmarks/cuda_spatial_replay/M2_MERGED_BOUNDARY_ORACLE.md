# CPU-merged M2 boundary oracle

`m2_merged_boundary_oracle_cli` independently converts the qualified
CPU-merged M2 `KM1WSCN1` capture into the exact canonical segment ABI consumed
by the CUDA Manhattan-union replay. It is the correctness oracle for the raw
M2 GPU union; it does not call or share code with that union.

The loader verifies:

- independent whole-file and scene SHA-256 allowlists;
- transport and canonical-payload SHA-256;
- merged assertion, source provenance, physical layer `101/0`, DBU, rule
  distances, all record counts, canonical offsets, and scene bbox;
- all context, cell, polygon, edge, and nonempty-context prefix ranges;
- clockwise, closed, simple, hole-free Manhattan stored contours using the
  exact sweep validator; and
- checked application of all eight orthogonal hierarchy transforms.

Reflected contexts reverse each transformed edge and traverse the source
contour in reverse order, restoring the clockwise/material-on-right contract.
Each world edge is normalized to the 32-byte
`manhattan_union_format.cuh::DirectedSegmentI64` layout:

```text
i64 fixed, lo, hi
i32 outward_side
u32 axis                 # horizontal=0, vertical=1
```

For horizontal edges, `side=-1/+1` means bottom/top. For vertical edges it
means left/right. Canonical order is axis, side, fixed, lo, hi, matching the
GPU replay.

The oracle additionally proves globally that there are no:

- adjacent collinear fragments;
- clockwise/CCW hole contours;
- repeated contour vertices or point-kissing contours;
- duplicate or opposite boundary fragments;
- positive-length collinear overlaps; or
- unexpected perpendicular crossings.

## Qualified production census

Input identity:

```text
file SHA-256    980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f
scene SHA-256   441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480
source SHA-256  8630e7ca7a2a72d03a4ba4fe61fd9f048a04d5370cd488b297c484bb19754558
```

Exact canonical result:

```text
stored contexts       39573
nonempty contexts       334
stored cells            121
stored contours       13166
stored edges        4380228
flat contours         14222
flat edges/segments 4385384
horizontal          2192692
vertical            2192692
negative side       2190644
positive side       2194740

boundary SHA-256  94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d
boundary FNV-1a   7541395996791771514
```

All topology error counters are exactly zero.

Maximum-contour and aggregate statistics:

```text
maximum edges             2084
flat contour ID             709
context ID                    4
stored source polygon       408
area                    217849000 dbu^2
perimeter                  2977710 dbu
width                        45170 dbu
height                     1415745 dbu
longest segment            1408625 dbu
total area            664939163050 dbu^2
total perimeter         9226168640 dbu
```

The full validation, flatten, global topology sweeps, canonical sorts, and
portable SHA-256 take about 6.0 seconds wall and 639 MiB peak RSS on the
qualification host.

## Reusable comparison API and stream

`m2_merged_boundary_oracle.h` exposes the loader, canonical segment vector,
exact statistics, and `compare_candidate()`. The comparator requires strict
canonical input and reports the first differing segment or count.

The CLI can also read or write a pointer-free candidate stream. Its 128-byte
little-endian header contains:

```text
char magic[8] = "KM2BND01"
u32 version=1, header_bytes=128, endian_tag, record_bytes=32
u64 file_bytes, record_count
u8 payload_sha256[32]
u8 source_scene_sha256[32]
u8 reserved_zero[24]
```

The payload is the canonical sequence of 32-byte segment records. The reader
checks every header field, length calculation, payload digest, scene identity,
record semantic, and canonical-order invariant before comparison.

## Build and run

```sh
cmake -S benchmarks/cuda_spatial_replay -B .scratch-build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build .scratch-build --target \
  m2_merged_boundary_oracle_cli m2_merged_boundary_oracle_test

.scratch-build/m2_merged_boundary_oracle_test
.scratch-build/m2_merged_boundary_oracle_cli \
  --expect-file-sha256=980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f \
  --expect-scene-sha256=441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480 \
  --expect-boundary-sha256=94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d \
  /path/to/m2-width-space-census-exact.km1ws
```
