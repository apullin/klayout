# Production M2 Manhattan-union bridge

`m2_manhattan_production_census` is a fail-closed bridge from the validated
`KACTSCN1` raw M2 hierarchy to the 48-byte rectangle ABI used by the standalone
Manhattan-union replay. It does not claim that the downstream union is correct.
It establishes the exact production rectangle stream and measures the global
coordinate-compressed x-slab workload on the GPU.

`m2_manhattan_production_loader.h` is the reusable seam. Its static library
returns only 24-byte resolved contexts, local 40-byte rectangle templates,
per-cell spans, canonical M2-context indices, and device output offsets. A CUDA
consumer can therefore expand directly into its resident `RectI64` buffer
without materializing or transferring the 1.026-GiB world stream on the host.

The bridge reuses the complete `KACTSCN1` digest, section, record, contour,
bbox, hierarchy, transform, overflow, and reachability validation from
`active3_scene_island.cu`. It additionally requires:

- logical slot 0 to be physical layer `101/0`;
- every raw M2 contour to be a clockwise four-edge box or six-edge L;
- every six-edge contour to occupy exactly three cells of its 2-by-2
  coordinate-compressed grid;
- exact area conservation for each rectangle decomposition; and
- caller-supplied scene SHA-256 and optional flat polygon/rectangle censuses.

Each L uses one deterministic nonoverlapping decomposition: its complete
x-column, then the occupied cell in the other column. The earlier VIA-stack
lowering emitted both X- and Y-oriented decompositions; that representation is
not used here.

The GPU applies all eight checked orthogonal transforms and emits
`{left,bottom,right,top,source_token,context_token}`. The optional
`--verify-expanded-host` gate copies the complete output back and compares all
six fields of every rectangle against an independent host transform, in
canonical stream order.

## Qualified production result

Input:

```text
file_sha256  67d22644602bcb306e376e99babce764da1242d4a6f14988e3b5fa664d8270bc
scene_sha256 dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2
cells         143
contexts      587201
M2 contexts   568632
```

Exact census:

```text
stored M2 polygons       45960
stored M2 L shapes           6
stored rectangles        45966
flat M2 polygons      22945976
flat M2 L shapes           468
flat rectangles        22946444
unique X coordinates      46384
X slabs                   46383
rectangle/slab memberships 92386704
maximum slabs/rectangle      43
rectangles over 4096 slabs    0
```

The canonical world-coordinate stream is pinned independently:

```text
record ABI bytes  48
stream bytes      1101429312
stream SHA-256    f9a3a3d4bbdadc80531c341361eb4f0dfc18538a5ada303ac909da86cea1c767
```

Thus the exact one-decomposition stream has only 468 more rectangles than
polygons. The old 22,947,380-box VIA-stack census is 936 boxes larger, exactly
two redundant boxes for each of the 468 flat L occurrences.

The global-X approach does not suffer a coordinate-cross-product explosion on
this workload. It does exceed the first union replay's 64-million membership
default: the production limit must cover 92,386,704 memberships and
184,773,408 y events. The largest individual rectangle spans only 43 slabs.

Three full coordinate-oracle runs on the RTX 3080 were identical. Their mean
charged GPU bridge/census time was 23.36 ms:

```text
upload             2.44 ms
expand             3.35 ms
X sort/unique     14.25 ms
membership count   3.32 ms
```

The one-time full 1.026-GiB D2H coordinate-by-coordinate oracle took about
1.25 seconds. Computing the reference SHA-256 with the deliberately local
portable implementation adds about 4.55 seconds. Neither is part of the
charged GPU timing or the resident production path.

## Build and run

Keep nvcc temporary files in a home-backed path on hosts whose `/tmp` quota is
small:

```sh
mkdir -p .scratch-build .scratch-tmp
TMPDIR="$PWD/.scratch-tmp" cmake \
  -S benchmarks/cuda_spatial_replay -B .scratch-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
TMPDIR="$PWD/.scratch-tmp" cmake --build .scratch-build \
  --target m2_manhattan_production_census

.scratch-build/m2_manhattan_production_census \
  --expect-scene-sha256=dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2 \
  --expect-flat-polygons=22945976 \
  --expect-flat-rectangles=22946444 \
  --expect-world-rect-sha256=f9a3a3d4bbdadc80531c341361eb4f0dfc18538a5ada303ac909da86cea1c767 \
  /path/to/m2-via1-x2.kact
```

The reusable loader has an independent link/symbol/census smoke target:

```sh
cmake --build .scratch-build \
  --target m2_manhattan_production_loader_smoke
.scratch-build/m2_manhattan_production_loader_smoke \
  /path/to/m2-via1-x2.kact \
  dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2
```
