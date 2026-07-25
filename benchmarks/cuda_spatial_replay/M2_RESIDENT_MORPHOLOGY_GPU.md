# Exact resident CUDA M2 F90/F270 certificate

`m2_resident_morphology_gpu.cu` is an independently linkable CUDA module for
the FreePDK45 M2 cumulative F90/F270 suffix.  It consumes the explicit
stream-ordered x-slab/interval view exported by `manhattan_union_gpu_core`;
it does not compile or include an executable `.cu` source.

The fixed exact transaction is:

```text
F90  = dilate_90(erode_89(raw M2 union))
long = F90 boundary segments with length >= 600
clean long-edge certificate = exact space(180) over all long pairs
F270 empty certificate = count(erode_269(F90)) == 0
```

Every coordinate is signed int64 and every operation is Manhattan set
geometry.  There is no floating point, raster approximation, contour
reconstruction, or host geometry round trip.  The only production host
transfer is the bounded eight-edge long-space certificate.  The full F90
boundary is copied only when the caller explicitly requests qualification.

## API and fail-closed contract

`m2_resident_morphology_gpu.cuh` exposes:

- a synchronous `consume_f90_f270(stream, view, request)` entry point;
- a checked `ResidentStripHook` adapter for the shared union core;
- independent input-slab, input-interval, output-slab, output-interval,
  raw-boundary, canonical-boundary, active-slab, per-band-work,
  total-work, and long-edge-pair capacities;
- exact counters and timings for both morphology passes, the F90 boundary and
  long-space certificate, and the count-only F270 erosion; and
- a qualification-only arbitrary-radius seam used by the raster differential
  driver.

The module retains no device pointer after return and synchronizes the supplied
stream before publishing a result.  Invalid views, repeated hook invocation,
cap exhaustion, count/product overflow, signed coordinate overflow, CUDA
failure, noncanonical output, emit/count disagreement, or an uncertain exact
predicate throws.  The union owner converts that exception to fallback and
never publishes a partial certificate.

The universal total source-interval-visit cap is 2,000,000,000.  The pinned
production scene's exact count-only `r=269` pass visits 4,681,660,762 source
intervals, so it uses a separate explicitly enabled and exactly
8,000,000,000 production-local cap.  Merely raising the numeric limit without
the production qualification flag is rejected.

The x arithmetic gate uses `__int128` before launching shifted-endpoint
kernels.  It proves that endpoint shifts, doubled midpoint comparisons, and
the full `±2r` active-window expressions fit signed int64.

## Exact gates

The separately compiled `m2_resident_morphology_replay.cc` links
`m2_resident_morphology_gpu`, `manhattan_union_gpu_core`, the compact
production loader, and the independent boundary oracle.

The fresh RTX 3080 gate passed:

- 248 edge-for-edge raster differentials: 14 directed fixtures under four
  operations plus 64 deterministic random overlapping-rectangle scenes under
  three operations;
- four low/high x-limit cases: two exact accepted boundaries and two
  fail-closed overflows;
- eight exact 180-DBU predicate fixtures, including 179/180/181 thresholds,
  diagonal Euclidean distance, exterior-side orientation, and unsafe signed
  span; and
- all 4,254,384 canonical F90 boundary edges against the checked stock KLayout
  golden, with FNV-64 `2057677162565968634`.

The production certificate additionally reproduced exactly:

```text
F90 strip intervals          39,930,010
F90 canonical boundary        4,254,384
F90 long edges                        8
unordered long-edge pairs            28
space violations                       0
uncertain predicates                   0
F270 eroded intervals                  0
```

The checked golden file SHA-256 is
`e7149202ef0ace74618a01f56135ea1cde2b4a0bdb00102f9a5f4a072af2ea49`;
its portable boundary payload SHA-256 is
`233a611bc306126b0292763954aab2d984508f2466a56ced2d1cf08af6c526ff`.

## Timing

A four-run process-local production gate measured:

| path | charged time |
|---|---:|
| stock KLayout F90/F270 suffix | 14.691 s |
| resident CUDA callback, warm median of three | 0.583350 s |
| stock union/stitch/F90/F270 block | 18.592380 s |
| compact load + host expansion + union/resident, warm | 1.989959 s |

The resident suffix therefore used **96.029% less time**, removing
**14.107650 real seconds** from that suffix.  The deliberately conservative
fully charged block used **89.297% less time**, removing **16.602421 real
seconds** from its like-for-like offline denominator.  These are module and
offline-pipeline results, not yet a measured whole-launcher reduction.

Warm resident phase timings were approximately 25 ms for `erode89`, 251 ms
for `dilate90`, 286 ms for boundary extraction plus the exact long-space
certificate, and 11 ms for the count-only `erode269`.  The sampled live
allocation delta was 3,428 MiB on the 10-GiB card.

## Reproduction

```sh
cmake -S benchmarks/cuda_spatial_replay \
  -B /tmp/klayout-m2-resident-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/klayout-m2-resident-build \
  --target m2_resident_morphology_replay -j 32

/tmp/klayout-m2-resident-build/m2_resident_morphology_replay \
  --self-test

/tmp/klayout-m2-resident-build/m2_resident_morphology_replay \
  --production \
  --kact /path/to/m2-via1-x2.kact \
  --gt90-golden /path/to/m2-gt90-stock.km2bnd \
  --repeat 4
```

This milestone stops at the reusable union-to-resident-suffix boundary.  The
live DSO adapter, raw hierarchy-to-resident union input, M2.1/M2.2 and M2.4
certificates, and atomic deck rewrite remain separate integration work.
