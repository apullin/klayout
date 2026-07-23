# Resident CUDA geometry pipeline

Status: design proposal. This is an additive successor to the current
stateless spatial-replay ABI; it does not change the existing v1 entry points.

## Objective

Move the expensive part of a deep DRC operation across one accelerator
ownership boundary:

```text
CPU once                                  GPU-resident rule execution
--------                                  ---------------------------
parse layout and rule                     expand/prune hierarchy contexts
validate supported subset                 generate spatial candidates
lower pointer-free scene       upload ->  evaluate exact predicates
                                           apply rule waivers and cull
                               <- result  reduce/deduplicate by hierarchy
validate and atomically publish            compact final markers/survivors
```

The production boundary must not download broad-phase pairs for routine CPU
predicate replay. A clean rule should return a small status/result header; a
nonempty rule should return compact final marker descriptors. Unsupported or
incomplete work must fall back before any KLayout receiver callback.

This follows the successful ngspice CUDA architecture: the host lowered a
complex pointer graph once, the GPU retained all repeatedly consumed state,
and only validated final output/checkpoint data crossed back. Earlier
evaluator-only and solver-only seams were transfer- and Amdahl-bound.

## Proposed additive C ABI

The ABI is POD-only. No `db::Layout`, shape, cell, iterator, receiver, or other
KLayout object pointer crosses it. Input pointers are borrowed only for the
duration of `create_scene`; the backend copies accepted data before returning.
Scene and result handles are backend-owned opaque tokens.

Illustrative entry points:

```c
uint32_t klayout_cuda_scene_abi_version(void);

int klayout_cuda_create_scene_v1(
    const struct klayout_cuda_scene_request_v1 *request,
    struct klayout_cuda_scene_v1 **scene,
    struct klayout_cuda_status_v1 *status);

int klayout_cuda_run_rule_v1(
    struct klayout_cuda_scene_v1 *scene,
    const struct klayout_cuda_rule_request_v1 *request,
    struct klayout_cuda_rule_result_v1 **result,
    struct klayout_cuda_status_v1 *status);

const struct klayout_cuda_rule_result_view_v1 *
klayout_cuda_rule_result_view_v1(
    const struct klayout_cuda_rule_result_v1 *result);

void klayout_cuda_release_rule_result_v1(
    struct klayout_cuda_rule_result_v1 *result);

void klayout_cuda_destroy_scene_v1(
    struct klayout_cuda_scene_v1 *scene);
```

Every request and view starts with `abi_version` and `struct_size`. A scene
request also carries:

- a content/revision fingerprint covering all lowered geometry and hierarchy;
- database-unit and coordinate-width declarations;
- root cell/context IDs;
- array counts and byte sizes;
- explicit resource ceilings; and
- pointers to the immutable arrays described below.

`run_rule` is speculative. It cannot mutate the source layout or publish
callbacks. KLayout validates the complete result and confirms the source
revision/fingerprint is unchanged before materializing any marker. A failed
attempt discards all backend output and runs the pristine CPU operation.

An optional `replace_layer`/`append_layer` entry point can be added later if
profiling proves that CPU-created derived layers must be refreshed without
rebuilding the immutable hierarchy. It is not required for the first proof.

## Pointer-free scene

The first scene schema should use dense indices and bounded offsets:

```text
Cell:
  bbox
  edge_begin, edge_count
  polygon_begin, polygon_count
  instance_begin, instance_count

Instance:
  child_cell
  orthogonal integer transform
  array columns, rows, column_pitch, row_pitch
  stable instance ID

Polygon:
  edge_begin, edge_count
  layer ID, property ID, stable shape ID

Edge:
  x1, y1, x2, y2
  polygon ID, stable edge ID

RootContext:
  cell ID, transform, clip/window, stable occurrence ID
```

Offsets and IDs are checked before upload. Arithmetic that can expand a box,
transform a coordinate, count an array, or size a buffer is overflow-checked
on the host and device. Initially unsupported transforms, properties,
breakout behavior, or polygon classes reject the scene or rule.

Regular arrays remain compressed in the scene. The device computes the
intersecting row/column range from the query window; it must not eagerly emit
every flat occurrence.

## Iterative hierarchy wavefront

Device hierarchy traversal uses bounded frontier arrays rather than recursion:

1. Seed the frontier from root contexts.
2. Classify each item as locally processable, pruned by bounding box, or
   requiring child expansion.
3. Count child/context outputs per item.
4. Use an exclusive scan to allocate the next frontier.
5. Emit transformed child records and compressed array ranges.
6. Sort/unique exact context keys where reuse is semantically valid.
7. Repeat until the frontier is empty or a configured capacity is reached.

A context key must include every value that can affect semantics: subject and
intruder cell IDs, composed transform, clip/window, relevant properties,
rule signature, and hierarchy ownership token. Context deduplication is an
optimization, never an excuse to merge distinguishable marker ownership.

Each terminal context feeds device-resident spatial indexing, exact predicate,
cull, and result-reduction stages. Intermediate candidates remain on device.
Use CCCL/CUB primitives for scans, radix sorting, selection, and reductions
where they preserve the required integer semantics. A fused pipeline does not
require one monolithic kernel.

## Result and survivor semantics

The result view contains:

- status and fallback flags;
- input scene fingerprint and rule fingerprint;
- counts for contexts, candidates, exact hits, culled items, survivors, and
  final markers;
- sorted final marker descriptors;
- sorted survivor descriptors for conservatively undecidable work; and
- charged stage timings.

A final marker descriptor contains stable context/shape/edge IDs, integer
geometry, property/ownership tokens, and the rule-local marker kind. It must
contain enough information for deterministic CPU materialization without
rerunning the broad or exact predicate.

A survivor means “not proven removable,” not “violation.” Partial replay is
allowed only when the host can prove that each survivor forms an interaction-
closed subproblem and that accepted GPU markers cannot interact with replayed
work. Otherwise any survivor requires full pristine CPU fallback.

Valid terminal outcomes are:

- `COMPLETE`, zero markers: clean certificate;
- `COMPLETE`, nonzero markers: exact compact final result;
- `PARTIAL`, survivors: only for an explicitly qualified interaction-closed
  replay contract; or
- `FALLBACK`/`ERROR`: no backend marker may be published.

No capacity limit silently truncates candidates, survivors, or markers.

## Milestone A: ACTIVE.3 resident certificate

Target the measured `well.enclosing(active, 55 nm, euclidean)` clean path
first. It has high wall-time exposure and simpler waiver semantics than
METAL1.3 while still exercising the deep hierarchy.

Initial qualified subset:

- orthogonal integer hierarchy transforms;
- rectilinear, valid polygon contours;
- exact context and property matching;
- checked 55 nm coordinate expansion; and
- an exact or conservative-safe Euclidean enclosure proof.

The device expands/prunes contexts, generates relevant ACTIVE/WELL candidates,
evaluates the supported enclosure proof, and compacts only ACTIVE shapes or
contexts it cannot prove enclosed. An empty survivor set is sufficient to
skip the legacy operation. Nonempty or unsupported output initially triggers
full pristine fallback; exact nonempty marker generation is a later promotion.

Required gates:

- clean x2 input produces zero survivors;
- an ACTIVE.3 violation sentinel forces fallback or returns the exact marker;
- transformed mixed hierarchy preserves cell/category/marker ownership;
- synthetic boundaries cover exactly-at-55-nm, one-unit-short, holes,
  containment, touching, nested arrays, rotations/reflections, and
  non-Manhattan fallback; and
- same-binary control/candidate reports are canonically identical.

## Milestone B: METAL1.3 analyze/cull

Then target the guarded CONTACT-in-METAL1 projection rule with
`one_side_allowed` and `two_opposite_sides_allowed`.

For the first bounded subset, contacts are valid guarded rectangles and M1
edges are directed Manhattan edges in the same exact hierarchy context. The
device returns, per surviving contact:

```text
contact_id
context_id
deficient_side_mask
UNCERTAIN and/or DISALLOWED flags
```

Side masks use the rule's four rectangle sides. Mask zero, a singleton side,
or exactly two opposite sides is waivable. Adjacent pairs, three/four sides,
partial projections, unsupported shielding cases, or non-Manhattan candidates
remain survivors. Only a zero-survivor result can initially bypass the
pristine legacy rule.

Promotion to exact nonempty output requires device-side shielding, marker
ownership, and hierarchy reduction to reproduce canonical CPU edge-pair
markers, not merely the same flattened geometry.

Required gates include the clean x2 workload, the standard violation sentinel,
the qualifying mixed-hierarchy nonempty fixture, exhaustive side-mask cases,
partial projections, corners, holes, shielding/order counterexamples,
nonrectangles, and transformed hierarchy.

## Exactness and fallback contract

Before a result is accepted, the host validates:

- ABI versions, structure sizes, fingerprints, and rule opcode/options;
- all counts, byte sizes, pointers, stable IDs, and sorted/unique guarantees;
- finite/in-range timing and telemetry counters;
- coordinate, transform, multiplication, and allocation overflow flags;
- result disposition consistent with marker/survivor counts;
- every returned descriptor against the original lowered scene; and
- an unchanged live layout/revision boundary.

The backend fails closed on unsupported rule options, transforms, properties,
geometry, resource exhaustion, CUDA/library failure, malformed output, or
user interruption. GPU success is never inferred from an empty pointer alone.

Validation progresses through:

1. a CPU implementation of the packed scene/rule pipeline versus current
   KLayout;
2. standalone GPU replay versus the packed CPU oracle;
3. focused synthetic hierarchy and predicate tests;
4. clean and nonempty live KLayout same-binary gates;
5. cross-design and cross-PDK regression; and
6. fresh-process runs that do not inherit warmed device state.

## Process ownership and a future broker

The accepted DRC launcher uses multiple processes. Independent DSOs would
create separate CUDA contexts, upload duplicate scenes, consume scarce device
memory, and serialize or contend unpredictably. Per-process residency alone
therefore does not deliver whole-launch residency.

The first integration should use one explicit GPU-owner shard and leave
unsupported rules on CPU shards. If reusable ACTIVE/M1/M2 work demonstrates a
charged whole-run opportunity, introduce one broker process owning:

- the CUDA context and persistent allocation pools;
- fingerprint-keyed immutable scenes;
- bounded per-client queues and streams;
- client/scene namespaces;
- shared or pinned staging buffers;
- cancellation and client-death cleanup; and
- deterministic result ownership/release.

No client pointer crosses the broker boundary. Broker construction is gated on
measured scene reuse and at least 10% modeled whole-run opportunity after all
IPC, queueing, packing, and contention costs.

## Charged performance criteria

Report real seconds and percent less wall time versus the immediate same-binary
control. Keep component, shard, critical-path, and full-launch scopes distinct.

Charge and report:

- CPU eligibility, lowering, packing, and fingerprinting;
- cold scene allocation and H2D upload;
- hierarchy wavefront, spatial indexing, exact predicate, cull, reduction,
  sort/unique, and compaction;
- status/marker D2H transfer;
- host validation and marker materialization;
- scene teardown when it is not amortized;
- wasted accelerator work plus the complete CPU rerun on fallback;
- GPU queue/IPC delay under simultaneous shard load; and
- resulting rule, shard, critical path, and full launcher wall time.

Measure both cold one-rule use and warm multi-rule scene reuse. Use at least
three fresh controls and candidates with CPU affinity/resource accounting,
plus simultaneous-shard runs on the real launcher. Report GPU activity and
achieved occupancy separately from wall-time benefit; utilization is a
diagnostic, not the acceptance criterion.

An accelerated component does not qualify as a product win unless the charged
critical path moves. A broker or generalized resident backend proceeds only
after exactness gates pass and the charged full-run opportunity remains at
least 10%.
