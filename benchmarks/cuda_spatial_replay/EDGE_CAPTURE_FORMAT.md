# Edge scanner capture format

`KLAYOUT_EDGE_REPLAY_CAPTURE_DIR=/path` enables a narrowly scoped diagnostic
capture in `poly2poly_check`. It recognizes only the first positive pass of a
shielded, different-polygon, different-layer `OverlapRelation` check with
projection metrics, a 90-degree angle limit, `[0, max)` projection limits,
partial-edge output, and `IncludeZeroDistanceWhenTouching`. This is the profile
used by the FreePDK45 M2 enclosure operation. The effective distance is stored
per request because hierarchy scaling can change it.

`KLAYOUT_EDGE_REPLAY_CAPTURE_MIN_RECORDS=N` suppresses request files with fewer
than `N` scanner records. Its default and minimum effective value are 1. Both
variables are read once, on the first construction of a polygon scanner. The
capture directory is created if necessary.

The instrumentation does not run when `KLAYOUT_EDGE_REPLAY_CAPTURE_DIR` is
unset. A capture uses request-local vectors and has no global lock in the
scanner callback. Each completed scanner invocation writes a unique temporary
file in the target directory and atomically renames it to
`edge-replay-p<PID>-t<THREAD>-r<REQUEST>-<TIME>.ker`.

## Version 1 binary layout

All current supported hosts are little-endian. Integers are stored in native
little-endian form. Each file contains, in order:

1. A 192-byte `KEDGER1\0` header.
2. `record_count` 96-byte records.
3. `broad_pair_count` 8-byte pairs that pass the receiver's cheap
   property/layer gate and reach the exact edge predicate.
4. `exact_pair_count` 8-byte exact-predicate pairs.

The header contains sizes and offsets for every section, the 64-bit effective
distance and signed 64-bit projection bounds, all relation/profile metadata, process,
thread and request identifiers, the request's scanner elapsed time in
nanoseconds, and these counters:

- scanner callbacks;
- scanner finish callbacks;
- callbacks whose edge pointer could not be resolved;
- sorted-unique pairs that reach the exact edge predicate;
- raw exact-predicate acceptances;
- sorted-unique exact-predicate pairs.

Version 1 encodes an unbounded maximum projection as signed `-1`. A capture
flag declares this convention. The in-memory filter uses the unsigned
`distance_type` maximum; readers must translate the sentinel rather than treat
it as a negative geometric limit.

Every 96-byte record contains signed 64-bit `left, bottom, right, top` AABB
coordinates, signed 64-bit `x1, y1, x2, y2` endpoints, the full 64-bit scanner
property (`size_t`), a one-based 32-bit request-local ID, a 32-bit context
(currently zero), and 32-bit flags. Flag bit 0 says endpoints are present and
bit 1 marks layer/side B (`property & 1`). ID zero is reserved as the pair-key
sentinel used by the replay backends.

Each pair is two ascending 32-bit request-local record IDs. Both pair sections
are sorted and unique. “Exact” means the first-pass directional
`EdgeRelationFilter::check` accepted the broad candidate; it is deliberately
before the existing CPU shielding pass. Shielding and final output order remain
CPU responsibilities.

The standalone CUDA replay's older `KSPAT01` format has only a 32-bit property
field. A converter or a native `KEDGER1` reader must explicitly range-check
before narrowing; the capture never truncates the scanner's `size_t` property.
