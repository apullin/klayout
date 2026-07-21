#!/usr/bin/env python3
"""Run the short hierarchical-network performance and exactness matrix.

The large physical inputs stay in the external klayout-perf corpus.  This
runner pins those inputs and their decks by SHA-256, checks an exact normalized
DRC report fingerprint, and writes only logs, reports, and one summary JSON to
the requested output directory.

The default ``regular`` group is one pass over five clean full-DRC cases, two
nonempty correctness sentinels, and three focused antenna/connectivity cases.
On the development host it takes roughly four minutes.  The larger HG0-S3
acceptance lane and HG0-S5 non-SRAM head-check are selected separately.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .klayout_runtime_provenance import collect_runtime_provenance
else:
    from klayout_runtime_provenance import collect_runtime_provenance


REPO_ROOT = Path(__file__).resolve().parent.parent
ANTENNA_DECK = REPO_ROOT / "testdata" / "drc" / "sky130_antenna_ladder.drc"
RUNTIME_PROVENANCE_SCRIPT = Path(__file__).resolve().with_name(
    "klayout_runtime_provenance.py"
)
SCHEMA_VERSION = 2
COMPARISON_IDENTITY_VERSION = 3
MINIMUM_INDEPENDENT_OBSERVATIONS = 3
RUNTIME_NORMALIZATION_POLICY_ID = "hier-network-relocatable-runtime-v1"
RSS_POLL_INTERVAL_SECONDS = 0.05
PEAK_RSS_CAVEAT = (
    "Linux /proc VmHWM polling covers the direct KLayout process; descendant "
    "processes are excluded and short-lived peaks between polls may be missed."
)

COMPARISON_MODES = ("repeat", "experiment", "historical")
TREATMENT_DIMENSIONS = (
    "runtime-bundle",
    "preload",
    "allocator-env",
    "thread-env",
    "input",
    "deck",
    "manifest",
    "argv",
    "execution-shape",
)

ALLOCATOR_ENV_NAMES = frozenset(
    {
        "GLIBC_TUNABLES",
        "MALLOC_ARENA_MAX",
        "MALLOC_ARENA_TEST",
        "MALLOC_CONF",
    }
)
ALLOCATOR_ENV_PREFIXES = ("MALLOC_", "JEMALLOC_", "TCMALLOC_")
THREAD_ENV_NAMES = frozenset(
    {
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
)
THREAD_ENV_PREFIXES = ("OMP_", "GOMP_", "KMP_", "TBB_", "RAYON_")

OPENRAM_SMALL_SHA256 = (
    "7f5a0ae375110399782667be3a40c9f2c9eb94deb087108ba134508554fd8a2e"
)
SKY130_DME1_SHA256 = (
    "32c2ee99e7defd8ac69223abc1c9cd0ab62d005ee08bbafab1fc1bb18528b851"
)
SKY130_FP16_SHA256 = (
    "80b9e0a3d831deece111c18ab3e20726a670937e11e78b40edef4884d567b91f"
)
SKY130_FP4_SHA256 = (
    "154a4101fe448eacec74d04d01abdeea362b9615378fe4a5e56d5841f13817e1"
)
SKY130_FP4_FLAT_SHA256 = (
    "4ccc828f14d81fe52d38512f3ee2d864ffd660ebd7e6360e4af3ed68dc7e820c"
)
SKY130_FP4_FLAT_MANIFEST_SHA256 = (
    "c915c5a76efd14d3a119d071cc17e7a9ed2a49e1656822574c3f79186b94834e"
)
FREEPDK45_VIOLATIONS_SHA256 = (
    "a2f0accacf3473cab0181ed0531e0cb7fe8be98affa220d22fe413e2d90f4195"
)
SKY130_SHARD_VIOLATIONS_SHA256 = (
    "d357ccb312ce5c4a6964475c127765ea031ad1148b37c28ffd75d1ec7ff6c236"
)
SKY130_HG0_S3_SHA256 = (
    "a5aae78efed5a76e5f2c6c03762f4fc60d70132931028bb47f941f08d3184de7"
)
SKY130_HG0_S5_SHA256 = (
    "03b62132c3f85b66664101fa90faacee6a952cc196cc246633e1c9e298c45911"
)

FREEPDK45_DECK_SHA256 = (
    "fa7edcc47d92eee4195693796c5e35b913ca476f8457965029d12372aa187db0"
)
SKY130_DECK_SHA256 = (
    "caf4a6b08cb12f78d6bb2d120737424c786b3bae8d6489234b9b269e91107bfc"
)
FREEPDK45_SHARDED_DECK_SHA256 = (
    "5b05b599f5af878364365136248177b63098e92ff18287d097274f54500ee2ce"
)
SKY130_SHARDED_DECK_SHA256 = (
    "9f8c1cffe597c69cd217c3c8c70cb9ece749e40b4170b10f45fd6dc2b7ddbec9"
)
ANTENNA_DECK_SHA256 = (
    "2664cb66dfddd515ce04d13e09f39feb0a96647e2170363b513a315c874b8096"
)

GENERATOR_PATTERN = re.compile(
    br"<generator>drc: script='[^']+'</generator>"
)
NORMALIZED_GENERATOR = b"<generator>drc: script='$DECK'</generator>"


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    input_relative: str
    input_sha256: str
    deck_relative: str | None
    deck_sha256: str
    top_cell: str
    normalized_report_sha256: str
    expected_report_item_count: int = 0
    auxiliary_relative: str | None = None
    auxiliary_sha256: str | None = None
    shard_manifest_relative: str | None = None
    shard_manifest_sha256: str | None = None


@dataclass(frozen=True)
class LoadedBaseline:
    summary: dict[str, object]
    timings: dict[str, float]
    metadata: dict[str, object]
    statistical_qualification: dict[str, object] | None = None


CASES = (
    Case(
        name="drc_fpd45_small",
        kind="freepdk45",
        input_relative="inputs/sram_1rw0r0w_2_16_freepdk45.gds",
        input_sha256=OPENRAM_SMALL_SHA256,
        deck_relative="decks/freepdk45.lydrc",
        deck_sha256=FREEPDK45_DECK_SHA256,
        top_cell="sram_1rw0r0w_2_16_freepdk45",
        normalized_report_sha256=(
            "f7358b9dd4b70e2f2e5d975d2fb14da2afb49f04634cb4fcc3bd1ba5b3a7e339"
        ),
    ),
    Case(
        name="sentinel_fpd45_nonempty",
        kind="freepdk45",
        input_relative="inputs/generated/freepdk45_violations.gds",
        input_sha256=FREEPDK45_VIOLATIONS_SHA256,
        deck_relative="decks/freepdk45-sharded.lydrc",
        deck_sha256=FREEPDK45_SHARDED_DECK_SHA256,
        top_cell="freepdk45_violations",
        normalized_report_sha256=(
            "e5bf625fda5eea31fc127870f837970396fe422f3940d9f4d65ca9e6da51d4da"
        ),
        expected_report_item_count=99,
    ),
    Case(
        name="sentinel_sky130_nonempty",
        kind="sky130",
        input_relative="inputs/generated/sky130_shard_violating.gds",
        input_sha256=SKY130_SHARD_VIOLATIONS_SHA256,
        deck_relative="decks/sky130A_mr-sharded.drc",
        deck_sha256=SKY130_SHARDED_DECK_SHA256,
        top_cell="SKY_SHARD_FIXTURE",
        normalized_report_sha256=(
            "22057d4a1882b23d7367fb99981c7de52f066c4ab4beebc792be2ca656bcb404"
        ),
        expected_report_item_count=25,
    ),
    Case(
        name="drc_sky130_fpu_fp4_hier",
        kind="sky130",
        input_relative="inputs/domestic_micro/fpu_fp4_asic.gds",
        input_sha256=SKY130_FP4_SHA256,
        deck_relative="decks/sky130A_mr.drc",
        deck_sha256=SKY130_DECK_SHA256,
        top_cell="fpu_fp4_asic",
        normalized_report_sha256=(
            "283fed7128087485a25d8ac86c9c6682e02b7dd7ec073363dc7cdc8ad578f025"
        ),
    ),
    Case(
        name="drc_sky130_fpu_fp4_flat",
        kind="sky130",
        input_relative=(
            "inputs/domestic_micro/fpu_fp4_asic__fullflat_shapes.gds"
        ),
        input_sha256=SKY130_FP4_FLAT_SHA256,
        deck_relative="decks/sky130A_mr.drc",
        deck_sha256=SKY130_DECK_SHA256,
        top_cell="fpu_fp4_asic__flat",
        normalized_report_sha256=(
            "7792fcee013d1a19c67708cb3374c4465f09d9de0db71cbd8dbb0d06420d6a1b"
        ),
        auxiliary_relative=(
            "inputs/domestic_micro/fpu_fp4_asic__fullflat_shapes.compose.json"
        ),
        auxiliary_sha256=SKY130_FP4_FLAT_MANIFEST_SHA256,
    ),
    Case(
        name="drc_sky130_dme1",
        kind="sky130",
        input_relative="inputs/domestic_micro/dme1_asic.gds",
        input_sha256=SKY130_DME1_SHA256,
        deck_relative="decks/sky130A_mr.drc",
        deck_sha256=SKY130_DECK_SHA256,
        top_cell="dme1_asic",
        normalized_report_sha256=(
            "2604d91cad8ec4af81e411e752fe9e5cca57fb8b366153c0d6063303b46bb07d"
        ),
    ),
    Case(
        name="drc_sky130_fpu_fp16",
        kind="sky130",
        input_relative="inputs/domestic_micro/fpu_fp16_asic.gds",
        input_sha256=SKY130_FP16_SHA256,
        deck_relative="decks/sky130A_mr.drc",
        deck_sha256=SKY130_DECK_SHA256,
        top_cell="fpu_fp16_asic",
        normalized_report_sha256=(
            "bfd79362b7a3ff9176392358066293ca18e7562f9e19eaeeaf4f70adc571ac30"
        ),
    ),
    Case(
        name="drc_sky130_hg0_s3",
        kind="sky130",
        input_relative="inputs/domestic_micro/hg0_s3_asic.gds",
        input_sha256=SKY130_HG0_S3_SHA256,
        deck_relative="decks/sky130A_mr-sharded.drc",
        deck_sha256=SKY130_SHARDED_DECK_SHA256,
        top_cell="hg0_s3_asic",
        normalized_report_sha256=(
            "9e8ce682678678d78a5ffa55ffd7303eb2ab41938a215b382c6ec2eaee2d9150"
        ),
    ),
    Case(
        name="drc_sky130_hg0_s5",
        kind="sky130",
        input_relative="inputs/domestic_micro/hg0_s5_asic.gds",
        input_sha256=SKY130_HG0_S5_SHA256,
        deck_relative="decks/sky130A_mr.drc",
        deck_sha256=SKY130_DECK_SHA256,
        top_cell="hg0_s5_asic",
        normalized_report_sha256=(
            "2c9f660d7b2d7186329c510333083bfe19ab17779fe42d66c936feac0b45fdb4"
        ),
    ),
    Case(
        name="antenna_sky130_fpu_fp4_hier",
        kind="antenna",
        input_relative="inputs/domestic_micro/fpu_fp4_asic.gds",
        input_sha256=SKY130_FP4_SHA256,
        deck_relative=None,
        deck_sha256=ANTENNA_DECK_SHA256,
        top_cell="fpu_fp4_asic",
        normalized_report_sha256=(
            "b892638081744b60f3642ca12db4a2c20346413906476da2d43bd15a9ff99e32"
        ),
    ),
    Case(
        name="antenna_sky130_dme1",
        kind="antenna",
        input_relative="inputs/domestic_micro/dme1_asic.gds",
        input_sha256=SKY130_DME1_SHA256,
        deck_relative=None,
        deck_sha256=ANTENNA_DECK_SHA256,
        top_cell="dme1_asic",
        normalized_report_sha256=(
            "83fb19e45f6e939fd3b0d31e2bd82c19d95e51f720b1c0e0477974ebf303dddf"
        ),
    ),
    Case(
        name="antenna_sky130_fpu_fp16",
        kind="antenna",
        input_relative="inputs/domestic_micro/fpu_fp16_asic.gds",
        input_sha256=SKY130_FP16_SHA256,
        deck_relative=None,
        deck_sha256=ANTENNA_DECK_SHA256,
        top_cell="fpu_fp16_asic",
        normalized_report_sha256=(
            "f938aedda3f78d75a92635e5d57a632391650a791ebcb5cc47622da367e633f3"
        ),
    ),
)

CASE_BY_NAME = {case.name: case for case in CASES}
ACCEPTANCE_CASE_NAMES = ("drc_sky130_hg0_s3",)
HEAD_CHECK_CASE_NAMES = ("drc_sky130_hg0_s5",)
SEPARATE_LANE_CASE_NAMES = ACCEPTANCE_CASE_NAMES + HEAD_CHECK_CASE_NAMES
CASE_GROUPS = {
    "regular": tuple(
        case.name
        for case in CASES
        if case.name not in SEPARATE_LANE_CASE_NAMES
    ),
    "acceptance": ACCEPTANCE_CASE_NAMES,
    "head_check": HEAD_CHECK_CASE_NAMES,
    "full": tuple(case.name for case in CASES if case.kind != "antenna"),
    "antenna": tuple(case.name for case in CASES if case.kind == "antenna"),
    "sky130_full": tuple(case.name for case in CASES if case.kind == "sky130"),
}
CASE_GROUPS["all"] = tuple(case.name for case in CASES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_file_hash(path: Path, expected: str, role: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{role} is missing: {path}")
    metadata = file_meta(path)
    if metadata["sha256"] != expected:
        raise ValueError(
            f"{role} SHA-256 mismatch for {path}: got {metadata['sha256']}, "
            f"expected {expected}"
        )
    return metadata


def resolve_executable(command: str) -> Path:
    expanded = os.path.expanduser(command)
    if os.sep in expanded or (os.altsep and os.altsep in expanded):
        candidate = Path(expanded).resolve()
    else:
        resolved = shutil.which(expanded)
        candidate = Path(resolved).resolve() if resolved else Path(expanded)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError(f"KLayout executable is unavailable: {command}")
    return candidate


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def harness_comparison_identity() -> dict[str, object]:
    """Fingerprint the runner and the schemas that define its output."""

    return {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "runtime_provenance_script_sha256": sha256_file(
            RUNTIME_PROVENANCE_SCRIPT
        ),
        "summary_schema_version": SCHEMA_VERSION,
        "comparison_identity_version": COMPARISON_IDENTITY_VERSION,
    }


def suite_comparison_identity(
    selected_cases: Sequence[Case],
) -> dict[str, object]:
    """Preserve exact suite membership and ordering."""

    ordered_case_names = [case.name for case in selected_cases]
    return {
        "ordered_case_names": ordered_case_names,
        "case_count": len(ordered_case_names),
        "fingerprint_sha256": canonical_json_sha256(ordered_case_names),
    }


def runtime_bundle_identity(
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Extract code and non-treatment controls from shared provenance."""

    identity = runtime_provenance.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("runtime provenance lacks its canonical identity")
    environment = identity.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("runtime provenance lacks normalized environment state")
    required = ("schema", "ld_preload", "environment")
    if any(key not in identity for key in required):
        raise ValueError("runtime provenance identity is incomplete")

    other_environment = {
        key: value
        for key, value in sorted(environment.items())
        if key != "LD_PRELOAD"
        and not _is_allocator_environment_key(key)
        and not _is_thread_environment_key(key)
    }
    # Preserve every current and future shared-identity field except the two
    # dimensions deliberately classified on their own below (preload and
    # allocator/thread environment).  This avoids silently dropping a newly
    # added code-coverage policy from controlled comparisons.
    filtered_runtime_identity = {
        key: value
        for key, value in identity.items()
        if key not in {"canonical_sha256", "ld_preload", "environment"}
    }
    filtered_runtime_identity["environment"] = other_environment
    components = {
        "schema": "hier-network-runtime-bundle-v3",
        "filtered_runtime_identity": filtered_runtime_identity,
    }
    return {
        "components": components,
        "fingerprint_sha256": canonical_json_sha256(components),
    }


def preload_identity(
    runtime_provenance: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Extract ordered preload content identity and diagnostic metadata."""

    raw_identity = runtime_provenance.get("identity")
    raw_metadata = runtime_provenance.get("ld_preload")
    if not isinstance(raw_identity, dict) or not isinstance(raw_metadata, dict):
        raise ValueError("runtime provenance lacks LD_PRELOAD identity")
    identity_entries = raw_identity.get("ld_preload")
    metadata_entries = raw_metadata.get("ordered_files")
    if not isinstance(identity_entries, list) or not isinstance(
        metadata_entries, list
    ):
        raise ValueError("runtime provenance has invalid LD_PRELOAD metadata")
    identity = {
        "ordered_files": list(identity_entries),
        "fingerprint_sha256": canonical_json_sha256(identity_entries),
    }
    return identity, list(metadata_entries)


def _is_allocator_environment_key(key: str) -> bool:
    return key in ALLOCATOR_ENV_NAMES or key.startswith(ALLOCATOR_ENV_PREFIXES)


def _is_thread_environment_key(key: str) -> bool:
    return (
        key in THREAD_ENV_NAMES
        or key.startswith(THREAD_ENV_PREFIXES)
        or key.endswith("_NUM_THREADS")
    )


def relevant_environment_identity(
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    raw_identity = runtime_provenance.get("identity")
    if not isinstance(raw_identity, dict) or not isinstance(
        raw_identity.get("environment"), dict
    ):
        raise ValueError("runtime provenance lacks normalized environment state")
    environment = raw_identity["environment"]
    return {
        "allocator": {
            key: value
            for key, value in sorted(environment.items())
            if _is_allocator_environment_key(key)
        },
        "threads": {
            key: value
            for key, value in sorted(environment.items())
            if _is_thread_environment_key(key)
        },
    }


def normalized_report_sha256(path: Path) -> str:
    contents = path.read_bytes()
    normalized, substitutions = GENERATOR_PATTERN.subn(
        NORMALIZED_GENERATOR, contents, count=1
    )
    if substitutions != 1:
        raise ValueError(
            f"report must contain exactly one recognizable DRC generator: {path}"
        )
    return hashlib.sha256(normalized).hexdigest()


def report_item_count(path: Path) -> int:
    stack: list[str] = []
    items_depth = None
    count = 0
    found_items = False
    for event, element in ET.iterparse(path, events=("start", "end")):
        local_name = element.tag.rsplit("}", 1)[-1]
        if event == "start":
            stack.append(local_name)
            if items_depth is None and local_name == "items":
                items_depth = len(stack)
                found_items = True
        else:
            if items_depth is not None and len(stack) == items_depth + 1:
                count += 1
            if (
                items_depth is not None
                and len(stack) == items_depth
                and local_name == "items"
            ):
                items_depth = None
            stack.pop()
            element.clear()
    if not found_items:
        raise ValueError(f"report has no <items> element: {path}")
    return count


def process_peak_rss_kb(pid: int) -> int | None:
    """Read Linux's kernel-maintained direct-process RSS high-water mark."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        with Path(f"/proc/{pid}/status").open("r", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def case_paths(case: Case, corpus_root: Path) -> tuple[Path, Path, Path | None]:
    input_path = corpus_root / case.input_relative
    deck_path = (
        ANTENNA_DECK
        if case.deck_relative is None
        else corpus_root / case.deck_relative
    )
    auxiliary_path = (
        corpus_root / case.auxiliary_relative if case.auxiliary_relative else None
    )
    return input_path, deck_path, auxiliary_path


def preflight_case(case: Case, corpus_root: Path) -> dict[str, object]:
    input_path, deck_path, auxiliary_path = case_paths(case, corpus_root)
    metadata: dict[str, object] = {
        "input": require_file_hash(
            input_path, case.input_sha256, f"{case.name} input"
        ),
        "deck": require_file_hash(deck_path, case.deck_sha256, f"{case.name} deck"),
    }
    if (auxiliary_path is None) != (case.auxiliary_sha256 is None):
        raise ValueError(
            f"{case.name} composition manifest path and SHA-256 must be paired"
        )
    if auxiliary_path is not None:
        assert case.auxiliary_sha256 is not None
        metadata["auxiliary"] = require_file_hash(
            auxiliary_path, case.auxiliary_sha256, f"{case.name} auxiliary input"
        )
    if (case.shard_manifest_relative is None) != (
        case.shard_manifest_sha256 is None
    ):
        raise ValueError(
            f"{case.name} shard manifest path and SHA-256 must be paired"
        )
    if case.shard_manifest_relative is not None:
        assert case.shard_manifest_sha256 is not None
        metadata["shard_manifest"] = require_file_hash(
            corpus_root / case.shard_manifest_relative,
            case.shard_manifest_sha256,
            f"{case.name} shard manifest",
        )
    return metadata


def verify_pinned_artifacts_unchanged(
    selected_cases: Sequence[Case],
    corpus_root: Path,
    pinned_metadata: Mapping[str, Mapping[str, object]],
) -> None:
    """Re-hash every selected input, deck, and manifest after measurement."""

    failures: list[str] = []
    for case in selected_cases:
        input_path, deck_path, auxiliary_path = case_paths(case, corpus_root)
        artifact_paths: dict[str, Path] = {
            "input": input_path,
            "deck": deck_path,
        }
        if auxiliary_path is not None:
            artifact_paths["auxiliary"] = auxiliary_path
        if case.shard_manifest_relative is not None:
            artifact_paths["shard_manifest"] = (
                corpus_root / case.shard_manifest_relative
            )
        initial_case_metadata = pinned_metadata.get(case.name)
        if not isinstance(initial_case_metadata, Mapping):
            failures.append(f"{case.name}: missing pre-run metadata")
            continue
        if set(artifact_paths) != set(initial_case_metadata):
            failures.append(
                f"{case.name}: artifact set changed from "
                f"{sorted(initial_case_metadata)} to {sorted(artifact_paths)}"
            )
        for role, path in artifact_paths.items():
            initial = initial_case_metadata.get(role)
            try:
                current = file_meta(path)
            except OSError as error:
                failures.append(f"{case.name} {role}: {error}")
                continue
            if current != initial:
                failures.append(
                    f"{case.name} {role}: metadata changed from "
                    f"{initial!r} to {current!r}"
                )
    if failures:
        raise RuntimeError(
            "pinned benchmark artifact(s) changed during the benchmark:\n  - "
            + "\n  - ".join(failures)
        )


def command_for_case(
    klayout: Path,
    case: Case,
    input_path: Path,
    deck_path: Path,
    report_path: Path,
    *,
    cpu_set: tuple[int, ...] | None = None,
    taskset: Path | None = None,
) -> list[str]:
    if case.kind == "freepdk45":
        command = [
            str(klayout),
            "-b",
            "-r",
            str(deck_path),
            "-rd",
            f"input={input_path}",
            "-rd",
            f"topcell={case.top_cell}",
            "-rd",
            f"output={report_path}",
        ]
    else:
        command = [
            str(klayout),
            "-b",
            "-zz",
            "-r",
            str(deck_path),
            "-rd",
            f"input={input_path}",
            "-rd",
            f"report={report_path}",
            "-rd",
            f"top_cell={case.top_cell}",
        ]
        if case.kind == "sky130":
            for setting in (
                "feol=true",
                "beol=true",
                "offgrid=true",
                "seal=true",
                "floating_met=false",
                "sram_exclude=false",
                "thr=4",
            ):
                command.extend(("-rd", setting))

    if cpu_set is not None:
        if taskset is None:
            raise ValueError("taskset is required when a CPU set is selected")
        command = [
            str(taskset),
            "--cpu-list",
            format_cpu_set(cpu_set),
            *command,
        ]
    return command


def format_cpu_set(cpus: Sequence[int]) -> str:
    """Return a stable compact Linux CPU-list representation."""

    if not cpus:
        return ""
    ranges: list[str] = []
    first = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def command_templates_for_case(
    case: Case,
    cpu_sets: list[tuple[int, ...]] | None,
) -> list[list[str]]:
    """Build path-normalized argv templates for every distinct parallel slot."""

    slot_cpu_sets: list[tuple[int, ...] | None] = (
        list(cpu_sets) if cpu_sets is not None else [None]
    )
    return [
        command_for_case(
            Path("$KLAYOUT"),
            case,
            Path("$INPUT"),
            Path("$DECK"),
            Path("$REPORT"),
            cpu_set=cpu_set,
            taskset=Path("$TASKSET") if cpu_set is not None else None,
        )
        for cpu_set in slot_cpu_sets
    ]


def _metadata_sha256(
    metadata: Mapping[str, object], key: str, role: str
) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
        raise ValueError(f"invalid {role} metadata")
    return value["sha256"]


def case_comparison_identity(
    case: Case,
    metadata: Mapping[str, object],
    cpu_sets: list[tuple[int, ...]] | None,
) -> dict[str, object]:
    input_sha256 = _metadata_sha256(metadata, "input", f"{case.name} input")
    deck_sha256 = _metadata_sha256(metadata, "deck", f"{case.name} deck")
    if input_sha256 is None or deck_sha256 is None:
        raise ValueError(f"{case.name} lacks required input or deck metadata")
    return {
        "kind": case.kind,
        "top_cell": case.top_cell,
        "artifacts": {
            "input_sha256": input_sha256,
            "deck_sha256": deck_sha256,
            "manifests": {
                "composition_sha256": _metadata_sha256(
                    metadata, "auxiliary", f"{case.name} composition manifest"
                ),
                "shard_sha256": _metadata_sha256(
                    metadata, "shard_manifest", f"{case.name} shard manifest"
                ),
            },
        },
        "argv_templates": command_templates_for_case(case, cpu_sets),
        "expected_report_item_count": case.expected_report_item_count,
        "expected_normalized_report_sha256": case.normalized_report_sha256,
    }


def current_host_identity() -> dict[str, str]:
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def build_comparison_identity(
    runtime_provenance: Mapping[str, object],
    preload: Mapping[str, object],
    environment: Mapping[str, object],
    host: Mapping[str, str],
    execution_shape: Mapping[str, object],
    selected_cases: Sequence[Case],
    pinned_metadata: Mapping[str, Mapping[str, object]],
    cpu_sets: list[tuple[int, ...]] | None,
) -> dict[str, object]:
    cases = {
        case.name: case_comparison_identity(
            case, pinned_metadata[case.name], cpu_sets
        )
        for case in selected_cases
    }
    return {
        "version": COMPARISON_IDENTITY_VERSION,
        "harness": harness_comparison_identity(),
        "suite": suite_comparison_identity(selected_cases),
        "runtime_bundle": runtime_bundle_identity(runtime_provenance),
        "preload": dict(preload),
        "environment": dict(environment),
        "host": dict(host),
        "execution_shape": dict(execution_shape),
        "cases": cases,
    }


def benchmark_environment(klayout: Path, klayout_home: Path) -> dict[str, str]:
    klayout_home.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(klayout.parent) + (
        os.pathsep + library_path if library_path else ""
    )
    environment["KLAYOUT_HOME"] = str(klayout_home)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    return environment


def collect_benchmark_runtime_provenance(
    klayout: Path,
    environment: Mapping[str, str],
    output_dir: Path,
) -> dict[str, object]:
    """Collect a relocatable identity for one fresh-home benchmark runtime."""

    klayout_home = environment.get("KLAYOUT_HOME")
    if not klayout_home:
        raise ValueError("benchmark runtime requires an explicit KLAYOUT_HOME")
    return collect_runtime_provenance(
        klayout,
        environment,
        cwd=output_dir,
        path_replacements={
            str(klayout.parent): "$KLAYOUT_INSTALL",
            str(Path(klayout_home).resolve()): "$KLAYOUT_HOME",
        },
        normalization_policy_id=RUNTIME_NORMALIZATION_POLICY_ID,
    )


def run_sample(
    klayout: Path,
    case: Case,
    corpus_root: Path,
    report_path: Path,
    log_path: Path,
    environment: dict[str, str],
    *,
    working_directory: Path,
    collect_child_usage: bool = True,
    cpu_set: tuple[int, ...] | None = None,
    taskset: Path | None = None,
) -> dict[str, object]:
    input_path, deck_path, _ = case_paths(case, corpus_root)
    if not working_directory.is_dir():
        raise ValueError(
            f"sample working directory is unavailable: {working_directory}"
        )
    command = command_for_case(
        klayout,
        case,
        input_path,
        deck_path,
        report_path,
        cpu_set=cpu_set,
        taskset=taskset,
    )
    for output_path in (report_path, log_path):
        if output_path.exists():
            output_path.unlink()

    usage_before = (
        resource.getrusage(resource.RUSAGE_CHILDREN) if collect_child_usage else None
    )
    started = time.perf_counter()
    peak_rss_kb: int | None = None
    with log_path.open("wb") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            observed_rss_kb = process_peak_rss_kb(process.pid)
            if observed_rss_kb is not None:
                peak_rss_kb = max(peak_rss_kb or 0, observed_rss_kb)
            time.sleep(RSS_POLL_INTERVAL_SECONDS)
    wall_seconds = time.perf_counter() - started
    usage_after = (
        resource.getrusage(resource.RUSAGE_CHILDREN) if collect_child_usage else None
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"{case.name} exited with status {process.returncode}; log: {log_path}"
        )
    if not report_path.is_file():
        raise RuntimeError(f"{case.name} did not create its report; log: {log_path}")

    items = report_item_count(report_path)
    if items != case.expected_report_item_count:
        raise RuntimeError(
            f"{case.name} produced {items} report items, expected "
            f"{case.expected_report_item_count}; report: {report_path}"
        )
    normalized_sha256 = normalized_report_sha256(report_path)
    if normalized_sha256 != case.normalized_report_sha256:
        raise RuntimeError(
            f"{case.name} normalized report SHA-256 mismatch: got "
            f"{normalized_sha256}, expected {case.normalized_report_sha256}; "
            f"report: {report_path}"
        )

    user_seconds = (
        usage_after.ru_utime - usage_before.ru_utime
        if usage_before is not None and usage_after is not None
        else None
    )
    system_seconds = (
        usage_after.ru_stime - usage_before.ru_stime
        if usage_before is not None and usage_after is not None
        else None
    )
    result = {
        "command": command,
        "wall_seconds": wall_seconds,
        "user_seconds": user_seconds,
        "system_seconds": system_seconds,
        "average_cpu_cores": (
            (user_seconds + system_seconds) / wall_seconds
            if user_seconds is not None and system_seconds is not None
            else None
        ),
        "child_cpu_timing": (
            "per-sample RUSAGE_CHILDREN delta"
            if collect_child_usage
            else "unavailable: concurrent child accounting is process-global"
        ),
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_source": (
            "linux_proc_status_vm_hwm_direct_process_polling"
            if peak_rss_kb is not None
            else "unavailable"
        ),
        "peak_rss_caveat": PEAK_RSS_CAVEAT,
        "cpu_affinity": list(cpu_set) if cpu_set is not None else None,
        "klayout_home": environment.get("KLAYOUT_HOME"),
        "working_directory": str(working_directory),
        "report_item_count": items,
        "report": file_meta(report_path),
        "normalized_report_sha256": normalized_sha256,
        "log": file_meta(log_path),
    }
    return result


def parse_cpu_sets(
    value: str | None,
    parallel_runs: int,
    parser: argparse.ArgumentParser,
) -> list[tuple[int, ...]] | None:
    if value is None:
        return None

    specifications = [specification.strip() for specification in value.split(";")]
    if any(not specification for specification in specifications):
        parser.error("--cpu-sets contains an empty CPU set")
    if len(specifications) != parallel_runs:
        parser.error(
            "--cpu-sets must provide exactly one semicolon-separated set per "
            f"parallel slot ({parallel_runs} required)"
        )

    parsed: list[tuple[int, ...]] = []
    claimed: set[int] = set()
    for specification in specifications:
        cpus: set[int] = set()
        for token in specification.split(","):
            token = token.strip()
            try:
                if "-" in token:
                    first_text, last_text = token.split("-", 1)
                    first = int(first_text)
                    last = int(last_text)
                    if first < 0 or last < first:
                        raise ValueError
                    cpus.update(range(first, last + 1))
                else:
                    cpu = int(token)
                    if cpu < 0:
                        raise ValueError
                    cpus.add(cpu)
            except ValueError:
                parser.error(
                    f"invalid CPU set {specification!r}; use forms such as "
                    "0-3,8"
                )
        if not cpus:
            parser.error("--cpu-sets contains an empty CPU set")
        overlap = claimed.intersection(cpus)
        if overlap:
            parser.error(
                "--cpu-sets must be disjoint; repeated CPUs: "
                + ",".join(str(cpu) for cpu in sorted(overlap))
            )
        claimed.update(cpus)
        parsed.append(tuple(sorted(cpus)))
    return parsed


def run_case_samples(
    klayout: Path,
    case: Case,
    corpus_root: Path,
    reports_dir: Path,
    logs_dir: Path,
    environment: dict[str, str],
    *,
    runs: int,
    parallel_runs: int,
    cpu_sets: list[tuple[int, ...]] | None,
    taskset: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run one case in ordered batches and return samples plus batch timings."""

    samples: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    sample_state_dir = reports_dir.parent / "sample-state"
    sample_state_dir.mkdir(parents=True, exist_ok=True)
    for batch_start in range(1, runs + 1, parallel_runs):
        run_indices = list(
            range(batch_start, min(batch_start + parallel_runs, runs + 1))
        )
        batch_started = time.perf_counter()
        futures: list[concurrent.futures.Future[dict[str, object]]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(run_indices)
        ) as executor:
            for slot, run_index in enumerate(run_indices):
                suffix = "" if runs == 1 else f".run-{run_index}"
                report_path = reports_dir / f"{case.name}{suffix}.drc.report"
                log_path = logs_dir / f"{case.name}{suffix}.log"
                sample_root = Path(
                    tempfile.mkdtemp(
                        prefix=f"{case.name}.run-{run_index}.",
                        dir=sample_state_dir,
                    )
                )
                sample_home = sample_root / "klayout-home"
                working_directory = sample_root / "work"
                sample_home.mkdir()
                working_directory.mkdir()
                sample_environment = dict(environment)
                sample_environment["KLAYOUT_HOME"] = str(sample_home)
                cpu_set = cpu_sets[slot] if cpu_sets is not None else None
                futures.append(
                    executor.submit(
                        run_sample,
                        klayout,
                        case,
                        corpus_root,
                        report_path,
                        log_path,
                        sample_environment,
                        working_directory=working_directory,
                        collect_child_usage=parallel_runs == 1,
                        cpu_set=cpu_set,
                        taskset=taskset,
                    )
                )
            batch_samples = [future.result() for future in futures]
        batch_wall = time.perf_counter() - batch_started
        samples.extend(batch_samples)
        batches.append(
            {
                "batch_index": len(batches) + 1,
                "concurrency": len(run_indices),
                "run_indices": run_indices,
                "wall_seconds": batch_wall,
            }
        )
    return samples, batches


def _positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _validate_controlled_baseline(
    baseline: Mapping[str, object],
    path: Path,
) -> None:
    """Reject summaries that cannot support a controlled timing comparison."""

    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"controlled baseline must use schema version {SCHEMA_VERSION}: {path}"
        )
    for key in ("measurement_mode", "execution_mode"):
        if baseline.get(key) != "isolated_latency":
            raise ValueError(
                f"controlled baseline {key} must be isolated_latency: {path}"
            )
    if baseline.get("parallel_runs") != 1:
        raise ValueError(
            f"controlled baseline parallel_runs must equal one: {path}"
        )
    runs_per_case = baseline.get("runs_per_case")
    if (
        not isinstance(runs_per_case, int)
        or isinstance(runs_per_case, bool)
        or runs_per_case < 1
    ):
        raise ValueError(f"controlled baseline has invalid runs_per_case: {path}")

    identity = baseline.get("comparison_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"controlled baseline lacks comparison identity: {path}")
    harness = identity.get("harness")
    if (
        identity.get("version") != COMPARISON_IDENTITY_VERSION
        or not isinstance(harness, dict)
        or not isinstance(harness.get("script_sha256"), str)
        or not isinstance(
            harness.get("runtime_provenance_script_sha256"), str
        )
        or harness.get("summary_schema_version") != SCHEMA_VERSION
        or harness.get("comparison_identity_version")
        != COMPARISON_IDENTITY_VERSION
    ):
        raise ValueError(
            f"controlled baseline has invalid harness/schema identity: {path}"
        )
    execution_shape = identity.get("execution_shape")
    if (
        not isinstance(execution_shape, dict)
        or execution_shape.get("mode") != "isolated_latency"
        or execution_shape.get("parallel_runs") != 1
    ):
        raise ValueError(
            f"controlled baseline identity is not an isolated measurement: {path}"
        )

    raw_cases = baseline.get("cases")
    if not isinstance(raw_cases, list) or any(
        not isinstance(result, dict) for result in raw_cases
    ):
        raise ValueError(f"controlled baseline has invalid case records: {path}")
    result_names = [result.get("name") for result in raw_cases]
    suite = identity.get("suite")
    identity_cases = identity.get("cases")
    if (
        not all(isinstance(name, str) and name for name in result_names)
        or baseline.get("selected_cases") != result_names
        or not isinstance(suite, dict)
        or suite.get("ordered_case_names") != result_names
        or suite.get("case_count") != len(result_names)
        or suite.get("fingerprint_sha256")
        != canonical_json_sha256(result_names)
        or not isinstance(identity_cases, dict)
        or list(identity_cases) != result_names
    ):
        raise ValueError(
            f"controlled baseline has inconsistent ordered suite/case context: {path}"
        )
    for result in raw_cases:
        assert isinstance(result, dict)
        name = result.get("name")
        label = repr(name) if isinstance(name, str) else "<unnamed>"
        if result.get("measurement_mode") != "isolated_latency":
            raise ValueError(
                f"controlled baseline case {label} is not isolated: {path}"
            )
        if result.get("parallel_runs") != 1:
            raise ValueError(
                f"controlled baseline case {label} parallel_runs is not one: {path}"
            )
        n = result.get("n")
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or n < 1
            or n != runs_per_case
        ):
            raise ValueError(
                f"controlled baseline case {label} has invalid n: {path}"
            )
        samples = result.get("per_job_wall_seconds")
        if not isinstance(samples, list) or len(samples) != n:
            raise ValueError(
                f"controlled baseline case {label} n/sample length mismatch: {path}"
            )
        if not all(_positive_finite_number(sample) for sample in samples):
            raise ValueError(
                f"controlled baseline case {label} has a non-finite or "
                f"non-positive sample wall: {path}"
            )
        stored_median = result.get("wall_median_seconds")
        recomputed_median = statistics.median(float(sample) for sample in samples)
        if (
            not _positive_finite_number(stored_median)
            or float(stored_median) != recomputed_median
        ):
            raise ValueError(
                f"controlled baseline case {label} wall median does not match "
                f"its samples: {path}"
            )

        batch_count = result.get("batch_count")
        batch_samples = result.get("batch_wall_samples_seconds")
        batches = result.get("batches")
        if batch_count != n or not isinstance(batch_samples, list) or len(
            batch_samples
        ) != n:
            raise ValueError(
                f"controlled baseline case {label} has invalid isolated batches: {path}"
            )
        if not all(_positive_finite_number(sample) for sample in batch_samples):
            raise ValueError(
                f"controlled baseline case {label} has invalid batch wall: {path}"
            )
        if (
            not isinstance(batches, list)
            or len(batches) != n
            or any(
                not isinstance(batch, dict)
                or batch.get("concurrency") != 1
                for batch in batches
            )
        ):
            raise ValueError(
                f"controlled baseline case {label} batches are not isolated: {path}"
            )


def load_baseline(
    path: Path | None,
    selected_case_names: Sequence[str],
    comparison_mode: str = "repeat",
) -> LoadedBaseline | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        baseline = json.load(stream)
    if not isinstance(baseline, dict) or not isinstance(baseline.get("cases"), list):
        raise ValueError(f"baseline is not a benchmark_hier_network result: {path}")

    timings: dict[str, float] = {}
    for result in baseline["cases"]:
        if not isinstance(result, dict):
            raise ValueError(f"baseline contains a non-object case: {path}")
        name = result.get("name")
        wall = result.get("wall_median_seconds")
        if not isinstance(name, str) or not name:
            raise ValueError(f"baseline contains a case without a valid name: {path}")
        if name in timings:
            raise ValueError(f"baseline contains duplicate case {name!r}: {path}")
        if not _positive_finite_number(wall):
            raise ValueError(
                f"baseline case {name!r} lacks a finite positive wall median: {path}"
            )
        timings[name] = float(wall)

    missing = [name for name in selected_case_names if name not in timings]
    if missing:
        raise ValueError(
            "baseline is missing selected case(s): " + ", ".join(missing)
        )
    qualification = None
    if comparison_mode in ("repeat", "experiment"):
        if list(timings) != list(selected_case_names):
            raise ValueError(
                "controlled baseline ordered case suite does not exactly match "
                f"the selection: {path}"
            )
        _validate_controlled_baseline(baseline, path)
        qualification = statistical_qualification(
            measurement_mode="isolated_latency",
            sample_count=int(baseline["runs_per_case"]),
            batch_count=int(baseline["runs_per_case"]),
        )
    elif comparison_mode != "historical":
        raise ValueError(f"unknown comparison mode: {comparison_mode}")
    return LoadedBaseline(
        summary=baseline,
        timings=timings,
        metadata=file_meta(path),
        statistical_qualification=qualification,
    )


def _difference(
    dimension: str,
    path: str,
    baseline: object,
    candidate: object,
) -> dict[str, object]:
    return {
        "dimension": dimension,
        "path": path,
        "baseline": baseline,
        "candidate": candidate,
    }


def comparison_identity_differences(
    baseline: object,
    candidate: Mapping[str, object],
) -> list[dict[str, object]]:
    """Describe every controlled-comparison invariant that changed."""

    required_top_level = {
        "version",
        "harness",
        "suite",
        "runtime_bundle",
        "preload",
        "environment",
        "host",
        "execution_shape",
        "cases",
    }
    if not isinstance(baseline, dict) or not required_top_level.issubset(baseline):
        return [
            _difference(
                "provenance",
                "comparison_identity",
                "missing or incomplete",
                f"version {COMPARISON_IDENTITY_VERSION}",
            )
        ]

    differences: list[dict[str, object]] = []
    if baseline.get("version") != candidate.get("version"):
        differences.append(
            _difference(
                "provenance",
                "comparison_identity.version",
                baseline.get("version"),
                candidate.get("version"),
            )
        )

    for key, dimension in (("harness", "provenance"), ("suite", "suite")):
        baseline_component = baseline.get(key)
        candidate_component = candidate.get(key)
        if not isinstance(baseline_component, dict) or not isinstance(
            candidate_component, dict
        ):
            differences.append(
                _difference(
                    "provenance",
                    f"comparison_identity.{key}",
                    "missing or invalid",
                    "complete",
                )
            )
        elif baseline_component != candidate_component:
            differences.append(
                _difference(
                    dimension,
                    f"comparison_identity.{key}",
                    baseline_component,
                    candidate_component,
                )
            )

    for key, dimension, fingerprint_key in (
        ("runtime_bundle", "runtime-bundle", "fingerprint_sha256"),
        ("preload", "preload", "fingerprint_sha256"),
    ):
        baseline_component = baseline.get(key)
        candidate_component = candidate.get(key)
        if (
            not isinstance(baseline_component, dict)
            or not isinstance(candidate_component, dict)
            or not isinstance(baseline_component.get(fingerprint_key), str)
            or not isinstance(candidate_component.get(fingerprint_key), str)
        ):
            differences.append(
                _difference(
                    "provenance",
                    f"comparison_identity.{key}",
                    "missing or invalid",
                    "complete",
                )
            )
        elif baseline_component.get(fingerprint_key) != candidate_component.get(
            fingerprint_key
        ):
            differences.append(
                _difference(
                    dimension,
                    f"comparison_identity.{key}.{fingerprint_key}",
                    baseline_component.get(fingerprint_key),
                    candidate_component.get(fingerprint_key),
                )
            )

    baseline_environment = baseline.get("environment")
    candidate_environment = candidate.get("environment")
    if not isinstance(baseline_environment, dict) or not isinstance(
        candidate_environment, dict
    ):
        differences.append(
            _difference(
                "provenance",
                "comparison_identity.environment",
                "missing or invalid",
                "complete",
            )
        )
    else:
        for key, dimension in (
            ("allocator", "allocator-env"),
            ("threads", "thread-env"),
        ):
            if not isinstance(baseline_environment.get(key), dict):
                differences.append(
                    _difference(
                        "provenance",
                        f"comparison_identity.environment.{key}",
                        "missing or invalid",
                        candidate_environment.get(key),
                    )
                )
            elif baseline_environment.get(key) != candidate_environment.get(key):
                differences.append(
                    _difference(
                        dimension,
                        f"comparison_identity.environment.{key}",
                        baseline_environment.get(key),
                        candidate_environment.get(key),
                    )
                )

    for key, dimension in (
        ("host", "host"),
        ("execution_shape", "execution-shape"),
    ):
        baseline_component = baseline.get(key)
        candidate_component = candidate.get(key)
        if not isinstance(baseline_component, dict) or not isinstance(
            candidate_component, dict
        ):
            differences.append(
                _difference(
                    "provenance",
                    f"comparison_identity.{key}",
                    "missing or invalid",
                    "complete",
                )
            )
        elif baseline_component != candidate_component:
            differences.append(
                _difference(
                    dimension,
                    f"comparison_identity.{key}",
                    baseline_component,
                    candidate_component,
                )
            )

    baseline_cases = baseline.get("cases")
    candidate_cases = candidate.get("cases")
    if not isinstance(baseline_cases, dict) or not isinstance(candidate_cases, dict):
        differences.append(
            _difference(
                "provenance",
                "comparison_identity.cases",
                "missing or invalid",
                "complete",
            )
        )
        return differences

    if set(baseline_cases) != set(candidate_cases):
        differences.append(
            _difference(
                "suite",
                "comparison_identity.cases.membership",
                sorted(baseline_cases),
                sorted(candidate_cases),
            )
        )

    for name, candidate_case in candidate_cases.items():
        baseline_case = baseline_cases.get(name)
        if not isinstance(baseline_case, dict) or not isinstance(candidate_case, dict):
            differences.append(
                _difference(
                    "suite",
                    f"comparison_identity.cases.{name}",
                    "missing or invalid",
                    "present",
                )
            )
            continue

        required_case_keys = {
            "kind",
            "top_cell",
            "artifacts",
            "argv_templates",
            "expected_report_item_count",
            "expected_normalized_report_sha256",
        }
        if not required_case_keys.issubset(baseline_case):
            differences.append(
                _difference(
                    "provenance",
                    f"comparison_identity.cases.{name}",
                    "incomplete",
                    "complete",
                )
            )
            continue

        for key in (
            "kind",
            "top_cell",
            "expected_report_item_count",
            "expected_normalized_report_sha256",
        ):
            if baseline_case.get(key) != candidate_case.get(key):
                differences.append(
                    _difference(
                        "correctness",
                        f"comparison_identity.cases.{name}.{key}",
                        baseline_case.get(key),
                        candidate_case.get(key),
                    )
                )

        baseline_artifacts = baseline_case.get("artifacts")
        candidate_artifacts = candidate_case.get("artifacts")
        if not isinstance(baseline_artifacts, dict) or not isinstance(
            candidate_artifacts, dict
        ):
            differences.append(
                _difference(
                    "provenance",
                    f"comparison_identity.cases.{name}.artifacts",
                    "missing or invalid",
                    "complete",
                )
            )
        else:
            for key, dimension in (
                ("input_sha256", "input"),
                ("deck_sha256", "deck"),
                ("manifests", "manifest"),
            ):
                baseline_value = baseline_artifacts.get(key)
                candidate_value = candidate_artifacts.get(key)
                if key == "manifests":
                    valid = (
                        isinstance(baseline_value, dict)
                        and {
                            "composition_sha256",
                            "shard_sha256",
                        }.issubset(baseline_value)
                        and all(
                            value is None or isinstance(value, str)
                            for value in baseline_value.values()
                        )
                    )
                else:
                    valid = isinstance(baseline_value, str)
                if not valid:
                    differences.append(
                        _difference(
                            "provenance",
                            f"comparison_identity.cases.{name}.artifacts.{key}",
                            "missing or invalid",
                            candidate_value,
                        )
                    )
                elif baseline_value != candidate_value:
                    differences.append(
                        _difference(
                            dimension,
                            f"comparison_identity.cases.{name}.artifacts.{key}",
                            baseline_value,
                            candidate_value,
                        )
                    )

        if not isinstance(baseline_case.get("argv_templates"), list):
            differences.append(
                _difference(
                    "provenance",
                    f"comparison_identity.cases.{name}.argv_templates",
                    "missing or invalid",
                    candidate_case.get("argv_templates"),
                )
            )
        elif baseline_case.get("argv_templates") != candidate_case.get(
            "argv_templates"
        ):
            differences.append(
                _difference(
                    "argv",
                    f"comparison_identity.cases.{name}.argv_templates",
                    baseline_case.get("argv_templates"),
                    candidate_case.get("argv_templates"),
                )
            )
    return differences


def _format_differences(differences: Sequence[Mapping[str, object]]) -> str:
    lines = []
    for difference in differences:
        baseline = json.dumps(difference.get("baseline"), sort_keys=True)
        candidate = json.dumps(difference.get("candidate"), sort_keys=True)
        lines.append(
            f"  - [{difference.get('dimension')}] {difference.get('path')}: "
            f"baseline {baseline}, candidate {candidate}"
        )
    return "\n".join(lines)


def authorize_comparison(
    mode: str,
    declared_treatments: Sequence[str],
    baseline: LoadedBaseline | None,
    candidate_identity: Mapping[str, object],
) -> dict[str, object]:
    if baseline is None:
        return {
            "mode": "none",
            "declared_treatments": [],
            "observed_differences": [],
            "identity_eligible_for_acceptance": False,
            "classification": "no baseline supplied",
        }

    differences = comparison_identity_differences(
        baseline.summary.get("comparison_identity"), candidate_identity
    )
    observed_dimensions = sorted(
        {str(difference["dimension"]) for difference in differences}
    )
    declared = sorted(set(declared_treatments))

    if mode == "repeat" and differences:
        raise ValueError(
            "baseline is not an identity-exact repeat:\n"
            + _format_differences(differences)
            + "\nUse --comparison-mode experiment with exact --treatment "
            "dimensions for a controlled A/B, or --comparison-mode historical "
            "for context-only numbers."
        )
    if mode == "experiment":
        if observed_dimensions != declared:
            raise ValueError(
                "experiment treatment declaration does not exactly match the "
                "observed differences: declared "
                f"{declared}, observed {observed_dimensions}\n"
                + _format_differences(differences)
            )
    elif mode == "historical":
        pass
    elif mode != "repeat":
        raise ValueError(f"unknown comparison mode: {mode}")

    return {
        "mode": mode,
        "declared_treatments": declared,
        "observed_differences": differences,
        "identity_eligible_for_acceptance": mode != "historical",
        "classification": (
            "historical context only; never a controlled or statistical result"
            if mode == "historical"
            else (
                "controlled experiment with exact declared treatments"
                if mode == "experiment"
                else "identity-exact repeat"
            )
        ),
    }


def statistical_qualification(
    *,
    measurement_mode: str,
    sample_count: int,
    batch_count: int,
) -> dict[str, object]:
    """Describe whether a result has enough independent observations."""

    if measurement_mode == "isolated_latency":
        independent_observations = sample_count
        observation_unit = "isolated run"
        observation_unit_plural = "isolated runs"
    elif measurement_mode == "concurrent_replicate_throughput":
        independent_observations = batch_count
        observation_unit = "concurrent batch"
        observation_unit_plural = "concurrent batches"
    else:
        raise ValueError(f"unknown measurement mode: {measurement_mode}")
    qualified = independent_observations >= MINIMUM_INDEPENDENT_OBSERVATIONS
    return {
        "qualified": qualified,
        "independent_observations": independent_observations,
        "observation_unit": observation_unit,
        "minimum_independent_observations": MINIMUM_INDEPENDENT_OBSERVATIONS,
        "reason": (
            "minimum independent-observation count satisfied"
            if qualified
            else (
                f"requires at least {MINIMUM_INDEPENDENT_OBSERVATIONS} "
                f"independent {observation_unit_plural}"
            )
        ),
    }


def percent_faster(baseline_seconds: float, candidate_seconds: float) -> float:
    return 100.0 * (baseline_seconds / candidate_seconds - 1.0)


def format_comparison(
    candidate_seconds: float,
    baseline_seconds: float | None,
    comparison_mode: str = "repeat",
) -> str:
    if baseline_seconds is None:
        return f"{candidate_seconds:.3f}s"
    faster = percent_faster(baseline_seconds, candidate_seconds)
    if faster >= 0.0:
        comparison = f"+{faster:.1f}% faster"
    else:
        comparison = f"{abs(faster):.1f}% slower"
    rendered = (
        f"{candidate_seconds:.3f}s ({comparison}; "
        f"baseline {baseline_seconds:.3f}s)"
    )
    if comparison_mode == "historical":
        return "HISTORICAL CONTEXT: " + rendered
    if comparison_mode == "experiment":
        return "CONTROLLED EXPERIMENT: " + rendered
    return rendered


def write_json_atomic(path: Path, value: object) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextlib.contextmanager
def exclusive_output_directory_lock(output_dir: Path):
    """Hold a non-blocking process lock for one benchmark output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".benchmark.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                f"benchmark output directory is already in use: {output_dir}"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def prepare_output_summary(
    output_dir: Path,
    baseline_path: Path | None,
    *,
    overwrite: bool,
) -> tuple[Path, Path | None]:
    """Invalidate an old active summary before any measured work starts."""

    summary_path = output_dir / "benchmark-summary.json"
    if baseline_path is not None and baseline_path == summary_path.resolve():
        raise ValueError(
            "baseline path must not be the output benchmark-summary.json"
        )
    if not summary_path.exists():
        return summary_path, None
    if not overwrite:
        raise ValueError(
            f"summary already exists: {summary_path}; use --overwrite or another directory"
        )
    archive_path = output_dir / (
        "benchmark-summary.previous-"
        f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-"
        f"{os.getpid()}-{time.time_ns()}.json"
    )
    os.replace(summary_path, archive_path)
    return summary_path, archive_path


def resolve_output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return Path(tempfile.gettempdir()) / (
        f"klayout-hier-network-{timestamp}-{os.getpid()}-{time.time_ns()}"
    )


def select_cases(value: str, parser: argparse.ArgumentParser) -> list[Case]:
    selected_names: set[str] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token in CASE_GROUPS:
            selected_names.update(CASE_GROUPS[token])
        elif token in CASE_BY_NAME:
            selected_names.add(token)
        else:
            parser.error(f"unknown case or group: {token}")
    if not selected_names:
        parser.error("--cases selected no cases")
    return [case for case in CASES if case.name in selected_names]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    case_help = ", ".join(CASE_GROUPS) + "; individual: " + ", ".join(CASE_BY_NAME)
    parser = argparse.ArgumentParser(
        description=(
            "Run the approximately four-minute hierarchical-network regression lane "
            "against an external klayout-perf corpus."
        )
    )
    parser.add_argument(
        "--corpus-root",
        default="~/personal/klayout-perf",
        help="read-only klayout-perf corpus root (default: %(default)s)",
    )
    parser.add_argument(
        "--klayout",
        default=str(REPO_ROOT / "build-release" / "klayout"),
        help="KLayout executable (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        help="output directory; default is a unique directory below /tmp",
    )
    parser.add_argument(
        "--cases",
        default="regular",
        help=f"comma-separated groups or cases (default: %(default)s); {case_help}",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="total measured jobs per case (default: %(default)s)",
    )
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=1,
        help=(
            "concurrent replicates in every batch; 1 measures isolated "
            "latency (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cpu-sets",
        help=(
            "optional explicit disjoint CPU set per parallel slot, separated "
            "by semicolons (example: '0-3;4-7'); no affinity is selected by "
            "default"
        ),
    )
    parser.add_argument(
        "--baseline",
        help="optional prior summary JSON used for +x%% faster reporting",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=COMPARISON_MODES,
        default="repeat",
        help=(
            "baseline policy: exact repeat, explicitly treated experiment, "
            "or context-only historical comparison (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--treatment",
        action="append",
        choices=TREATMENT_DIMENSIONS,
        default=[],
        metavar="DIMENSION",
        help=(
            "identity dimension intentionally changed by an experiment; repeat "
            "for every treatment"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace benchmark-summary.json in an existing output directory",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least one")
    if args.parallel_runs < 1:
        parser.error("--parallel-runs must be at least one")
    if args.parallel_runs > args.runs:
        parser.error("--parallel-runs cannot exceed --runs")
    if args.parallel_runs > 1 and args.runs % args.parallel_runs != 0:
        parser.error(
            "--runs must be a multiple of --parallel-runs so every measured "
            "batch has identical concurrency"
        )
    if args.parallel_runs > 1 and args.baseline:
        parser.error(
            "--baseline compares isolated latency and cannot be used with "
            "concurrent replicate throughput mode"
        )
    duplicate_treatments = sorted(
        treatment
        for treatment in set(args.treatment)
        if args.treatment.count(treatment) > 1
    )
    if duplicate_treatments:
        parser.error(
            "duplicate --treatment dimension(s): "
            + ", ".join(duplicate_treatments)
        )
    if args.comparison_mode == "experiment":
        if not args.baseline:
            parser.error("--comparison-mode experiment requires --baseline")
        if not args.treatment:
            parser.error(
                "--comparison-mode experiment requires at least one --treatment"
            )
    elif args.treatment:
        parser.error("--treatment is only valid with --comparison-mode experiment")
    if args.comparison_mode == "historical" and not args.baseline:
        parser.error("--comparison-mode historical requires --baseline")
    args.cpu_sets = parse_cpu_sets(args.cpu_sets, args.parallel_runs, parser)
    args.selected_cases = select_cases(args.cases, parser)
    return args


def run_benchmark(args: argparse.Namespace, output_dir: Path) -> int:
    baseline_path = (
        Path(args.baseline).expanduser().resolve() if args.baseline else None
    )
    reports_dir = output_dir / "reports"
    logs_dir = output_dir / "logs"
    summary_path, archived_summary_path = prepare_output_summary(
        output_dir, baseline_path, overwrite=args.overwrite
    )

    corpus_root = Path(args.corpus_root).expanduser().resolve()
    klayout = resolve_executable(args.klayout)
    inherited_allowed_cpus = (
        tuple(sorted(os.sched_getaffinity(0)))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    measurement_mode = (
        "isolated_latency"
        if args.parallel_runs == 1
        else "concurrent_replicate_throughput"
    )
    execution_shape = {
        "mode": measurement_mode,
        "parallel_runs": args.parallel_runs,
        "cpu_affinity_mode": (
            "explicit_taskset" if args.cpu_sets is not None else "none"
        ),
        "requested_slot_cpu_sets": (
            [format_cpu_set(cpu_set) for cpu_set in args.cpu_sets]
            if args.cpu_sets is not None
            else None
        ),
        "inherited_allowed_cpu_set": (
            format_cpu_set(inherited_allowed_cpus)
            if inherited_allowed_cpus is not None
            else None
        ),
        "taskset_sha256": None,
    }
    taskset = None
    affinity_validated = False
    if args.cpu_sets is not None:
        taskset_command = shutil.which("taskset")
        if taskset_command is None:
            raise ValueError("--cpu-sets requires the taskset executable")
        taskset = Path(taskset_command).resolve()
        execution_shape["taskset_sha256"] = sha256_file(taskset)
        if inherited_allowed_cpus is not None:
            available_cpus = set(inherited_allowed_cpus)
            requested_cpus = {cpu for cpu_set in args.cpu_sets for cpu in cpu_set}
            unavailable_cpus = requested_cpus.difference(available_cpus)
            if unavailable_cpus:
                unavailable = ",".join(str(cpu) for cpu in sorted(unavailable_cpus))
                raise ValueError(
                    "--cpu-sets requests CPUs unavailable to this process: "
                    f"{unavailable}"
                )
            affinity_validated = True

    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Validate the entire requested corpus before paying for any benchmark run.
    pinned_metadata = {
        case.name: preflight_case(case, corpus_root) for case in args.selected_cases
    }

    environment = benchmark_environment(klayout, output_dir / "klayout-home")
    version_process = subprocess.run(
        [str(klayout), "-v"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if version_process.returncode != 0:
        raise RuntimeError(
            f"KLayout version query failed with status {version_process.returncode}: "
            f"{(version_process.stderr or version_process.stdout).strip()}"
        )
    version = (version_process.stdout or version_process.stderr).strip()
    runtime_provenance = collect_benchmark_runtime_provenance(
        klayout, environment, output_dir
    )
    runtime_provenance["version"] = version
    preload_fingerprint, preload_metadata = preload_identity(runtime_provenance)
    environment_identity = relevant_environment_identity(runtime_provenance)
    host_metadata = current_host_identity()
    candidate_identity = build_comparison_identity(
        runtime_provenance,
        preload_fingerprint,
        environment_identity,
        host_metadata,
        execution_shape,
        args.selected_cases,
        pinned_metadata,
        args.cpu_sets,
    )

    # Refuse an invalid comparison before starting a measured KLayout run.
    loaded_baseline = load_baseline(
        baseline_path,
        [case.name for case in args.selected_cases],
        args.comparison_mode,
    )
    comparison_policy = authorize_comparison(
        args.comparison_mode,
        args.treatment,
        loaded_baseline,
        candidate_identity,
    )
    baseline_timings = loaded_baseline.timings if loaded_baseline else {}
    baseline_meta = loaded_baseline.metadata if loaded_baseline else None
    statistical_policy = statistical_qualification(
        measurement_mode=measurement_mode,
        sample_count=args.runs,
        batch_count=args.runs // args.parallel_runs,
    )
    baseline_statistically_qualified = bool(
        loaded_baseline is not None
        and loaded_baseline.statistical_qualification is not None
        and loaded_baseline.statistical_qualification["qualified"]
    )
    eligible_for_acceptance = bool(
        comparison_policy["identity_eligible_for_acceptance"]
    ) and bool(statistical_policy["qualified"])
    eligible_for_acceptance = (
        eligible_for_acceptance and baseline_statistically_qualified
    )

    results: list[dict[str, object]] = []
    if args.parallel_runs == 1:
        print("measurement mode: isolated latency (one active replicate)")
    else:
        print(
            "measurement mode: concurrent replicate throughput "
            f"(up to {args.parallel_runs} active replicates; per-job wall "
            "includes contention and is not isolated latency)"
        )
    total_started = time.perf_counter()
    for case in args.selected_cases:
        samples, batches = run_case_samples(
            klayout,
            case,
            corpus_root,
            reports_dir,
            logs_dir,
            environment,
            runs=args.runs,
            parallel_runs=args.parallel_runs,
            cpu_sets=args.cpu_sets,
            taskset=taskset,
        )

        wall_samples = [float(sample["wall_seconds"]) for sample in samples]
        rss_samples = [
            int(sample["peak_rss_kb"])
            for sample in samples
            if sample.get("peak_rss_kb") is not None
        ]
        average_cpu_samples = [
            float(sample["average_cpu_cores"])
            for sample in samples
            if sample.get("average_cpu_cores") is not None
        ]
        wall_median = statistics.median(wall_samples)
        batch_wall_samples = [float(batch["wall_seconds"]) for batch in batches]
        batch_wall_sum = sum(batch_wall_samples)
        batch_wall_median = statistics.median(batch_wall_samples)
        batch_throughput_samples = [
            3600.0 * int(batch["concurrency"]) / float(batch["wall_seconds"])
            for batch in batches
        ]
        throughput = 3600.0 * len(samples) / batch_wall_sum
        baseline_seconds = baseline_timings.get(case.name)
        result: dict[str, object] = {
            "name": case.name,
            "kind": case.kind,
            "top_cell": case.top_cell,
            "measurement_mode": measurement_mode,
            "n": len(samples),
            "wall_median_seconds": wall_median,
            "wall_median_scope": (
                "per-job KLayout subprocess latency; contention-affected in "
                "concurrent mode"
            ),
            "wall_mean_seconds": statistics.mean(wall_samples),
            "wall_min_seconds": min(wall_samples),
            "wall_max_seconds": max(wall_samples),
            "wall_stdev_seconds": (
                statistics.stdev(wall_samples) if len(wall_samples) > 1 else 0.0
            ),
            "peak_rss_kb_max": max(rss_samples) if rss_samples else None,
            "peak_rss_kb_samples": rss_samples,
            "average_cpu_cores_mean": (
                statistics.mean(average_cpu_samples)
                if average_cpu_samples
                else None
            ),
            "average_cpu_cores_samples": average_cpu_samples,
            "per_job_wall_seconds": wall_samples,
            "per_job_wall_scope": "KLayout subprocess only",
            "parallel_runs": args.parallel_runs,
            "batch_count": len(batches),
            "batch_wall_samples_seconds": batch_wall_samples,
            "batch_wall_median_seconds": batch_wall_median,
            "batch_wall_sum_seconds": batch_wall_sum,
            "batch_throughput_samples_designs_per_hour": (
                batch_throughput_samples
            ),
            "batch_throughput_median_designs_per_hour": statistics.median(
                batch_throughput_samples
            ),
            "throughput_designs_per_hour": throughput,
            "batches": batches,
            "expected_report_item_count": case.expected_report_item_count,
            "expected_normalized_report_sha256": case.normalized_report_sha256,
            "statistical_qualification": statistical_policy,
            **pinned_metadata[case.name],
            "samples": samples,
        }
        if baseline_seconds is not None:
            result["comparison"] = {
                "mode": comparison_policy["mode"],
                "baseline_wall_seconds": baseline_seconds,
                "candidate_wall_seconds": wall_median,
                "percent_faster": percent_faster(baseline_seconds, wall_median),
                "identity_eligible_for_acceptance": comparison_policy[
                    "identity_eligible_for_acceptance"
                ],
                "candidate_statistically_qualified": statistical_policy[
                    "qualified"
                ],
                "baseline_statistically_qualified": (
                    baseline_statistically_qualified
                ),
                "statistically_qualified": bool(
                    statistical_policy["qualified"]
                )
                and baseline_statistically_qualified,
                "eligible_for_acceptance": eligible_for_acceptance,
            }
        results.append(result)
        if args.parallel_runs == 1:
            print(
                f"{case.name:38s} "
                f"{format_comparison(wall_median, baseline_seconds, args.comparison_mode)}"
            )
        else:
            print(
                f"{case.name:38s} {len(samples)} jobs; "
                f"per-job median {wall_median:.3f}s; "
                f"batch wall {batch_wall_sum:.3f}s; "
                f"throughput {throughput:.2f} designs/hour"
            )

    elapsed = time.perf_counter() - total_started

    # A build or preload replaced in place during a benchmark invalidates its
    # provenance even if the pre-run comparison check was exact.
    post_runtime_provenance = collect_benchmark_runtime_provenance(
        klayout, environment, output_dir
    )
    post_runtime_provenance["version"] = version
    post_preload_fingerprint, _ = preload_identity(post_runtime_provenance)
    if runtime_bundle_identity(post_runtime_provenance) != candidate_identity.get(
        "runtime_bundle"
    ):
        raise RuntimeError("KLayout runtime bundle changed during the benchmark")
    if post_preload_fingerprint != candidate_identity.get("preload"):
        raise RuntimeError("LD_PRELOAD contents changed during the benchmark")
    if harness_comparison_identity() != candidate_identity.get("harness"):
        raise RuntimeError("benchmark harness changed during the benchmark")
    if taskset is not None and sha256_file(taskset) != execution_shape.get(
        "taskset_sha256"
    ):
        raise RuntimeError("taskset executable changed during the benchmark")
    verify_pinned_artifacts_unchanged(
        args.selected_cases, corpus_root, pinned_metadata
    )

    candidate_sum = sum(float(result["wall_median_seconds"]) for result in results)
    aggregate_batch_wall = sum(
        float(result["batch_wall_sum_seconds"]) for result in results
    )
    completed_case_runs = sum(int(result["n"]) for result in results)
    aggregate_throughput = 3600.0 * completed_case_runs / aggregate_batch_wall
    aggregate = None
    if loaded_baseline is not None:
        baseline_sum = sum(
            loaded_baseline.timings[str(result["name"])] for result in results
        )
        aggregate = {
            "mode": comparison_policy["mode"],
            "identity_eligible_for_acceptance": comparison_policy[
                "identity_eligible_for_acceptance"
            ],
            "candidate_statistically_qualified": statistical_policy[
                "qualified"
            ],
            "baseline_statistically_qualified": (
                baseline_statistically_qualified
            ),
            "statistically_qualified": bool(statistical_policy["qualified"])
            and baseline_statistically_qualified,
            "eligible_for_acceptance": eligible_for_acceptance,
            "metric": "isolated_wall_median_seconds",
            "baseline_sum_seconds": baseline_sum,
            "candidate_sum_seconds": candidate_sum,
            "percent_faster": percent_faster(baseline_sum, candidate_sum),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_root": str(corpus_root),
        "output_dir": str(output_dir),
        "selected_cases": [case.name for case in args.selected_cases],
        "execution_mode": measurement_mode,
        "measurement_mode": measurement_mode,
        "measurement_mode_description": (
            "one active replicate; per-job wall is an isolated-latency metric"
            if args.parallel_runs == 1
            else (
                "replicates overlap; per-job wall is contention-affected and "
                "batch throughput, not isolated latency, is the primary metric"
            )
        ),
        "runs_per_case": args.runs,
        "parallel_runs": args.parallel_runs,
        "comparison_identity": candidate_identity,
        "comparison": {
            **comparison_policy,
            "statistical_qualification": statistical_policy,
            "baseline_statistical_qualification": (
                loaded_baseline.statistical_qualification
                if loaded_baseline is not None
                else None
            ),
            "statistically_qualified": bool(statistical_policy["qualified"])
            and baseline_statistically_qualified,
            "eligible_for_acceptance": eligible_for_acceptance,
            "metric": (
                "isolated_wall_median_seconds"
                if loaded_baseline is not None
                else None
            ),
        },
        "elapsed_seconds": elapsed,
        "case_wall_median_sum_seconds": candidate_sum,
        "aggregate_throughput": {
            "completed_case_runs": completed_case_runs,
            "batch_wall_sum_seconds": aggregate_batch_wall,
            "designs_per_hour": aggregate_throughput,
            "batch_wall_scope": (
                "thread-pool launch, process launch, DRC execution, and exact "
                "report validation"
            ),
        },
        "baseline": baseline_meta,
        "archived_previous_summary": (
            file_meta(archived_summary_path)
            if archived_summary_path is not None
            else None
        ),
        "aggregate_comparison": aggregate,
        "runtime_provenance": runtime_provenance,
        "preloaded_libraries": preload_metadata,
        "environment": {
            key: environment.get(key)
            for key in (
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "KLAYOUT_HOME",
                "QT_QPA_PLATFORM",
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
            )
        },
        "cpu_affinity": {
            "mode": "explicit_taskset" if args.cpu_sets is not None else "none",
            "slot_cpu_sets": (
                [list(cpu_set) for cpu_set in args.cpu_sets]
                if args.cpu_sets is not None
                else None
            ),
            "validated_against_process_affinity": affinity_validated,
            "inherited_allowed_cpu_set": (
                format_cpu_set(inherited_allowed_cpus)
                if inherited_allowed_cpus is not None
                else None
            ),
        },
        "host": host_metadata,
        "cases": results,
    }
    write_json_atomic(summary_path, summary)

    if args.parallel_runs == 1:
        if aggregate is None:
            print(f"total isolated median sum: {candidate_sum:.3f}s")
        else:
            print(
                "total isolated median sum: "
                + format_comparison(
                    candidate_sum,
                    float(aggregate["baseline_sum_seconds"]),
                    args.comparison_mode,
                )
            )
    else:
        print(f"total concurrent batch wall: {aggregate_batch_wall:.3f}s")
        print(f"aggregate throughput: {aggregate_throughput:.2f} designs/hour")
    print(f"summary: {summary_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_directory(args)
    with exclusive_output_directory_lock(output_dir):
        return run_benchmark(args, output_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        sys.stderr.write(f"ERROR: {error}\n")
        sys.exit(1)
