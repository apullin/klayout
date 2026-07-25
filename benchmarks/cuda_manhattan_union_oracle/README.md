# Exact bounded Manhattan-union oracle

This directory contains a deliberately slow CPU correctness oracle for the
CUDA Manhattan-union work. It does not share the CUDA sweep/scan algorithm:

1. Input `int64` rectangles are expanded into occupied integer unit cells.
2. Cells sharing a positive-length edge receive the same connected-component
   diagnostic label.
3. Exposed cell sides are coalesced into maximal collinear fragments.
4. Every fragment is directed with occupied material on its **right**. This is
   KLayout's canonical convention: outer hulls are clockwise and holes are
   counter-clockwise.

KLayout production defaults to maximum coherence (`min_coherence=false`),
which resolves kissing corners into fewer polygons. This oracle intentionally
does **not** claim that its 4-neighbour occupancy labels reproduce that contour
pairing. Every checkerboard/degree-4 lattice vertex returns
`UnionStatus::UnsupportedKissingVertex` with a canonical vertex/diagonal/
component diagnostic. A qualified production fast path must fail closed on
that status until it implements and validates maximum-coherence pairing
exactly.

Component IDs are canonical: components are sorted by their least occupied
cell, comparing X and then Y. The fragment list is sorted by component and
directed endpoints. The output therefore remains stable under input ordering
and rectangle decomposition.

The bounded unit grid is intentional. This is an obviously correct oracle for
small fixtures and deterministic differential fuzzing, not a candidate
production union engine.

Run it without changing the main KLayout build:

```sh
benchmarks/cuda_manhattan_union_oracle/run.sh
```

The runner keeps its tiny standalone build under an ignored `.build`
subdirectory by default, avoiding both the main CMake tree and system `/tmp`.
Set `KLAYOUT_MANHATTAN_ORACLE_BUILD_DIR` to override it.

The test uses a separately implemented dense bitmap/disjoint-set/lattice-scan
reference for 5,000 deterministic randomized differential cases. Directed
fixtures cover overlap, containment, coincident rectangles, shared edges,
corner touching, an L shape, a hole, a one-cell channel, and a same-component
degree-4 kissing vertex. Both kissing fixtures explicitly assert the
unsupported/fail-closed status. The suite also checks exact
clockwise/counter-clockwise orientation, input-order and rectangle-split
metamorphisms, capacity failures, and coordinates at both ends of the signed
64-bit range.
