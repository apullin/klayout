#!/usr/bin/env python3
"""Run the short hierarchical-network performance and exactness matrix.

The large physical inputs stay in the external klayout-perf corpus.  This
runner pins those inputs and their decks by SHA-256, checks an exact normalized
DRC report fingerprint, and writes only logs, reports, and one summary JSON to
the requested output directory.

The default ``regular`` group is one pass over five full DRC cases and three
focused antenna/connectivity cases.  On the development host it takes roughly
3 minutes 49 seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
ANTENNA_DECK = REPO_ROOT / "testdata" / "drc" / "sky130_antenna_ladder.drc"
SCHEMA_VERSION = 1

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

FREEPDK45_DECK_SHA256 = (
    "fa7edcc47d92eee4195693796c5e35b913ca476f8457965029d12372aa187db0"
)
SKY130_DECK_SHA256 = (
    "caf4a6b08cb12f78d6bb2d120737424c786b3bae8d6489234b9b269e91107bfc"
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
    auxiliary_relative: str | None = None
    auxiliary_sha256: str | None = None


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
CASE_GROUPS = {
    "regular": tuple(case.name for case in CASES),
    "full": tuple(case.name for case in CASES if case.kind != "antenna"),
    "antenna": tuple(case.name for case in CASES if case.kind == "antenna"),
    "sky130_full": tuple(case.name for case in CASES if case.kind == "sky130"),
}
CASE_GROUPS["all"] = CASE_GROUPS["regular"]


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


def adjacent_library_meta(klayout: Path) -> list[dict[str, object]]:
    aliases_by_target: dict[Path, list[str]] = {}
    for candidate in klayout.parent.iterdir():
        if not candidate.name.startswith("libklayout_") or ".so" not in candidate.name:
            continue
        if not candidate.is_file():
            continue
        target = candidate.resolve()
        aliases_by_target.setdefault(target, []).append(candidate.name)

    libraries = []
    for target, aliases in sorted(aliases_by_target.items(), key=lambda item: str(item[0])):
        metadata = file_meta(target)
        metadata["aliases"] = sorted(aliases)
        libraries.append(metadata)
    return libraries


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
    if auxiliary_path is not None and case.auxiliary_sha256 is not None:
        metadata["auxiliary"] = require_file_hash(
            auxiliary_path, case.auxiliary_sha256, f"{case.name} auxiliary input"
        )
    return metadata


def command_for_case(
    klayout: Path, case: Case, input_path: Path, deck_path: Path, report_path: Path
) -> list[str]:
    if case.kind == "freepdk45":
        return [
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
    return command


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


def run_sample(
    klayout: Path,
    case: Case,
    corpus_root: Path,
    report_path: Path,
    log_path: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    input_path, deck_path, _ = case_paths(case, corpus_root)
    command = command_for_case(klayout, case, input_path, deck_path, report_path)
    for output_path in (report_path, log_path):
        if output_path.exists():
            output_path.unlink()

    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with log_path.open("wb") as log_stream:
        process = subprocess.run(
            command,
            cwd=report_path.parent,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)

    if process.returncode != 0:
        raise RuntimeError(
            f"{case.name} exited with status {process.returncode}; log: {log_path}"
        )
    if not report_path.is_file():
        raise RuntimeError(f"{case.name} did not create its report; log: {log_path}")

    items = report_item_count(report_path)
    if items != 0:
        raise RuntimeError(
            f"{case.name} produced {items} report items, expected zero; "
            f"report: {report_path}"
        )
    normalized_sha256 = normalized_report_sha256(report_path)
    if normalized_sha256 != case.normalized_report_sha256:
        raise RuntimeError(
            f"{case.name} normalized report SHA-256 mismatch: got "
            f"{normalized_sha256}, expected {case.normalized_report_sha256}; "
            f"report: {report_path}"
        )

    return {
        "command": command,
        "wall_seconds": wall_seconds,
        "user_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "system_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "report_item_count": items,
        "report": file_meta(report_path),
        "normalized_report_sha256": normalized_sha256,
        "log": file_meta(log_path),
    }


def load_baseline(path: Path | None) -> tuple[dict[str, float], dict[str, object] | None]:
    if path is None:
        return {}, None
    with path.open(encoding="utf-8") as stream:
        baseline = json.load(stream)
    if not isinstance(baseline, dict) or not isinstance(baseline.get("cases"), list):
        raise ValueError(f"baseline is not a benchmark_hier_network result: {path}")
    timings: dict[str, float] = {}
    for result in baseline["cases"]:
        if not isinstance(result, dict):
            continue
        name = result.get("name")
        wall = result.get("wall_median_seconds")
        if isinstance(name, str) and isinstance(wall, (int, float)) and wall > 0:
            timings[name] = float(wall)
    return timings, file_meta(path)


def percent_faster(baseline_seconds: float, candidate_seconds: float) -> float:
    return 100.0 * (baseline_seconds / candidate_seconds - 1.0)


def format_comparison(candidate_seconds: float, baseline_seconds: float | None) -> str:
    if baseline_seconds is None:
        return f"{candidate_seconds:.3f}s"
    faster = percent_faster(baseline_seconds, candidate_seconds)
    if faster >= 0.0:
        comparison = f"+{faster:.1f}% faster"
    else:
        comparison = f"{abs(faster):.1f}% slower"
    return (
        f"{candidate_seconds:.3f}s ({comparison}; "
        f"baseline {baseline_seconds:.3f}s)"
    )


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
            "Run the approximately 3m49 hierarchical-network regression lane "
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
        help="measured runs per case (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline",
        help="optional prior summary JSON used for +x%% faster reporting",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace benchmark-summary.json in an existing output directory",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least one")
    args.selected_cases = select_cases(args.cases, parser)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    corpus_root = Path(args.corpus_root).expanduser().resolve()
    klayout = resolve_executable(args.klayout)
    baseline_path = Path(args.baseline).expanduser().resolve() if args.baseline else None
    baseline_timings, baseline_meta = load_baseline(baseline_path)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output_dir = Path(tempfile.gettempdir()) / (
            f"klayout-hier-network-{timestamp}-{os.getpid()}"
        )
    reports_dir = output_dir / "reports"
    logs_dir = output_dir / "logs"
    summary_path = output_dir / "benchmark-summary.json"
    if summary_path.exists() and not args.overwrite:
        raise ValueError(
            f"summary already exists: {summary_path}; use --overwrite or another directory"
        )
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
    launcher_meta = file_meta(klayout)
    launcher_meta["version"] = version
    launcher_meta["adjacent_libraries"] = adjacent_library_meta(klayout)

    results: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for case in args.selected_cases:
        samples = []
        for run_index in range(1, args.runs + 1):
            suffix = "" if args.runs == 1 else f".run-{run_index}"
            report_path = reports_dir / f"{case.name}{suffix}.drc.report"
            log_path = logs_dir / f"{case.name}{suffix}.log"
            samples.append(
                run_sample(
                    klayout,
                    case,
                    corpus_root,
                    report_path,
                    log_path,
                    environment,
                )
            )

        wall_median = statistics.median(
            float(sample["wall_seconds"]) for sample in samples
        )
        baseline_seconds = baseline_timings.get(case.name)
        result: dict[str, object] = {
            "name": case.name,
            "kind": case.kind,
            "top_cell": case.top_cell,
            "n": len(samples),
            "wall_median_seconds": wall_median,
            "wall_min_seconds": min(float(sample["wall_seconds"]) for sample in samples),
            "expected_report_item_count": 0,
            "expected_normalized_report_sha256": case.normalized_report_sha256,
            **pinned_metadata[case.name],
            "samples": samples,
        }
        if baseline_seconds is not None:
            result["baseline_wall_seconds"] = baseline_seconds
            result["percent_faster"] = percent_faster(
                baseline_seconds, wall_median
            )
        results.append(result)
        print(f"{case.name:38s} {format_comparison(wall_median, baseline_seconds)}")

    elapsed = time.perf_counter() - total_started
    candidate_sum = sum(float(result["wall_median_seconds"]) for result in results)
    comparable_results = [
        result for result in results if "baseline_wall_seconds" in result
    ]
    aggregate = None
    if len(comparable_results) == len(results):
        baseline_sum = sum(
            float(result["baseline_wall_seconds"]) for result in comparable_results
        )
        aggregate = {
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
        "runs_per_case": args.runs,
        "elapsed_seconds": elapsed,
        "case_wall_median_sum_seconds": candidate_sum,
        "baseline": baseline_meta,
        "aggregate_comparison": aggregate,
        "launcher": launcher_meta,
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
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cases": results,
    }
    write_json_atomic(summary_path, summary)

    if aggregate is None:
        print(f"total measured wall: {candidate_sum:.3f}s")
    else:
        print(
            "total measured wall: "
            + format_comparison(
                candidate_sum, float(aggregate["baseline_sum_seconds"])
            )
        )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        sys.stderr.write(f"ERROR: {error}\n")
        sys.exit(1)
