#!/usr/bin/env python3
"""Prove the FreePDK45 CONTACT.1-.5 owner split with nonempty CPU reports.

The supplied deck must be the current fully composed 12-owner FreePDK45 deck
before ``--split-implant-contact`` is applied.  This gate deliberately clears
all CUDA opt-ins, generates a hierarchical violation fixture, and proves:

* source ``all`` mode equals transformed ``all`` mode after removing only the
  generator path;
* the strict deck-bound manifest accepts the complete 13-shard category and
  item union;
* the manifest merge exactly equals transformed ``all`` mode; and
* CONTACT.1-.5 are nonempty and owned only by ``contact``, while the retained
  ``implant_contact`` owner remains nonempty through IMPLANT.5.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import merge_sharded_lyrdb as report_model  # noqa: E402


FIXTURE_SCRIPT = HERE / "contact1_5_fixture.rb"
SPLIT_SCRIPT = HERE / "split_deck.py"
MERGER = ROOT / "scripts" / "merge_sharded_lyrdb.py"
TOP_CELL = "FREEPDK45_CONTACT_OWNER_SPLIT"

SHARDS = (
    "m1_width_space",
    "implant_contact",
    "contact",
    "antenna_m4_m10",
    "antenna_m3",
    "antenna_m2",
    "antenna_m1",
    "m2_rules",
    "m1_enclosure",
    "via1_upper_active12",
    "grid",
    "m1_via_class",
    "antenna_feol",
)
CONTACT_CATEGORIES = tuple(f"CONTACT.{rule}" for rule in range(1, 6))
IMPLANT_CATEGORIES = tuple(f"IMPLANT.{rule}" for rule in range(1, 6))


class GateError(RuntimeError):
    """A fail-closed gate condition was not satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--klayout",
        required=True,
        help="CPU-capable KLayout executable (CUDA runtime opt-ins are cleared)",
    )
    parser.add_argument(
        "--deck",
        required=True,
        type=Path,
        help="fully composed unsplit 12-owner FreePDK45 .lydrc",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="retain the successful temporary evidence directory",
    )
    return parser.parse_args()


def executable(value: str) -> Path:
    candidate = Path(value)
    resolved = (
        shutil.which(value)
        if candidate.name == value
        else str(candidate.expanduser().resolve())
    )
    if resolved is None:
        raise GateError(f"KLayout executable was not found: {value}")
    path = Path(resolved)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise GateError(f"KLayout is not executable: {path}")
    return path.resolve()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise GateError(f"{label} is missing: {resolved}")
    return resolved


def runtime_environment(work: Path) -> dict[str, str]:
    runtime = work / "runtime"
    paths = {
        "HOME": runtime / "home",
        "KLAYOUT_HOME": runtime / "klayout-home",
        "XDG_CONFIG_HOME": runtime / "xdg-config",
        "XDG_CACHE_HOME": runtime / "xdg-cache",
        "XDG_DATA_HOME": runtime / "xdg-data",
        "TMPDIR": runtime / "tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True)

    # Start from a small allowlist.  In particular, no KLAYOUT_CUDA_* variable
    # can cross this boundary, so every generated optional transaction takes
    # its historical CPU path.
    environment = {
        name: str(path) for name, path in paths.items()
    }
    environment.update(
        {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "QT_QPA_PLATFORM": "offscreen",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if os.environ.get("LD_LIBRARY_PATH"):
        environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return environment


def run(
    command: list[str],
    *,
    log: Path,
    environment: dict[str, str],
) -> None:
    with log.open("xb") as output:
        completed = subprocess.run(
            command,
            check=False,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if completed.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        details = "\n".join(tail)
        raise GateError(
            f"command failed with exit {completed.returncode}: "
            f"{' '.join(command)}\n{details}"
        )


def run_drc(
    klayout: Path,
    deck: Path,
    fixture: Path,
    report: Path,
    *,
    shard: str,
    environment: dict[str, str],
) -> None:
    run(
        [
            str(klayout),
            "-b",
            "-r",
            str(deck),
            "-rd",
            f"input={fixture}",
            "-rd",
            f"topcell={TOP_CELL}",
            "-rd",
            f"output={report}",
            "-rd",
            f"drc_shard={shard}",
        ],
        log=report.with_suffix(".log"),
        environment=environment,
    )
    if not report.is_file() or report.stat().st_size == 0:
        raise GateError(f"{shard} produced no report: {report}")


def assert_no_cuda_telemetry(reports: Path) -> None:
    for log in sorted(reports.glob("*.log")):
        for line in log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.lstrip().startswith("CUDA "):
                raise GateError(
                    f"CPU-only gate observed CUDA telemetry in {log}: {line}"
                )


def canonical_report(path: Path) -> bytes:
    # Reuse the strict merger's validated semantic model.  KLayout emits cells
    # in encounter order, while the merger emits the same cell/reference graph
    # in dependency order.  Item order can likewise differ across independent
    # shard processes.  This canonical form ignores those two non-semantic
    # orderings plus the deck path, but retains category order, tag content,
    # the complete cell/reference graph, every meaningful item field, and
    # duplicate-item multiplicity.
    root = report_model._parse_report(path)
    metadata = report_model._metadata(root)
    metadata["generator"] = "DRC_GENERATOR"
    cells = report_model._cell_records(root)
    ordered_cells = tuple(
        (name, cells[name]) for name in report_model._ordered_cells(cells)
    )
    items = tuple(
        sorted(
            (repr(fingerprint), multiplicity)
            for fingerprint, multiplicity in report_model._item_counter(
                root
            ).items()
        )
    )
    model = (
        tuple(metadata.items()),
        tuple(sorted(report_model._tags(root))),
        tuple(report_model._categories(root)),
        ordered_cells,
        items,
    )
    return repr(model).encode("utf-8")


def assert_same_report(left: Path, right: Path, label: str) -> bytes:
    left_bytes = canonical_report(left)
    right_bytes = canonical_report(right)
    if left_bytes != right_bytes:
        raise GateError(
            f"{label} differs after removing only <generator>: "
            f"{hashlib.sha256(left_bytes).hexdigest()} != "
            f"{hashlib.sha256(right_bytes).hexdigest()}"
        )
    return left_bytes


def parse_report(path: Path) -> ET.Element:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GateError(f"invalid report {path}: {exc}") from exc
    if root.tag != "report-database":
        raise GateError(f"unexpected report root in {path}: {root.tag}")
    return root


def category_paths(root: ET.Element) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []

    def visit(container: ET.Element, prefix: tuple[str, ...]) -> None:
        for category in container.findall("category"):
            name = category.findtext("name")
            if name is None:
                raise GateError("report category has no name")
            path = prefix + (name,)
            result.append(path)
            nested = category.find("categories")
            if nested is not None:
                visit(nested, path)

    container = root.find("categories")
    if container is None:
        raise GateError("report has no categories")
    visit(container, ())
    return tuple(result)


def item_counts(root: ET.Element) -> Counter[str]:
    result: Counter[str] = Counter()
    container = root.find("items")
    if container is None:
        raise GateError("report has no items container")
    for item in container.findall("item"):
        category = item.findtext("category")
        if category is None:
            raise GateError("report item has no category")
        result[category.strip("'")] += 1
    return result


def verify_evidence(
    *,
    split_deck: Path,
    split_all: Path,
    merged: Path,
    manifest: Path,
    shard_reports: dict[str, Path],
) -> tuple[str, Counter[str]]:
    canonical = assert_same_report(
        split_all, merged, "transformed all-mode and strict manifest merge"
    )
    all_root = parse_report(split_all)
    counts = item_counts(all_root)
    missing = [category for category in CONTACT_CATEGORIES if counts[category] == 0]
    if missing:
        raise GateError(f"focused fixture left CONTACT categories empty: {missing}")
    if counts["IMPLANT.5"] == 0:
        raise GateError("focused fixture left retained IMPLANT.5 empty")

    top_cell = all_root.findtext("top-cell")
    if top_cell != TOP_CELL:
        raise GateError(f"unexpected report top cell: {top_cell!r}")
    cells = all_root.find("cells")
    if cells is None or len(cells.findall("cell")) < 2:
        raise GateError("focused report did not preserve hierarchical cell evidence")

    contact_root = parse_report(shard_reports["contact"])
    implant_root = parse_report(shard_reports["implant_contact"])
    contact_inventory = category_paths(contact_root)
    implant_inventory = category_paths(implant_root)
    if contact_inventory != tuple((name,) for name in CONTACT_CATEGORIES):
        raise GateError(
            f"contact shard category inventory is not exact: {contact_inventory}"
        )
    if implant_inventory != tuple((name,) for name in IMPLANT_CATEGORIES):
        raise GateError(
            "implant_contact shard category inventory is not exact: "
            f"{implant_inventory}"
        )

    contact_counts = item_counts(contact_root)
    implant_counts = item_counts(implant_root)
    for category in CONTACT_CATEGORIES:
        if contact_counts[category] != counts[category]:
            raise GateError(
                f"{category} item ownership mismatch: "
                f"all={counts[category]} contact={contact_counts[category]}"
            )
    for category in IMPLANT_CATEGORIES:
        if implant_counts[category] != counts[category]:
            raise GateError(
                f"{category} item ownership mismatch: "
                f"all={counts[category]} "
                f"implant_contact={implant_counts[category]}"
            )

    try:
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"invalid manifest {manifest}: {exc}") from exc
    if tuple(manifest_value.get("shards", ())) != SHARDS:
        raise GateError("manifest shard order differs from the qualified owner plan")
    deck_sha256 = hashlib.sha256(split_deck.read_bytes()).hexdigest()
    if manifest_value.get("deck_sha256") != deck_sha256:
        raise GateError("manifest is not bound to the transformed deck")
    owners = {
        tuple(entry["path"]): entry["owner"]
        for entry in manifest_value.get("categories", ())
    }
    for category in CONTACT_CATEGORIES:
        if owners.get((category,)) != "contact":
            raise GateError(f"manifest does not assign {category} to contact")
    for category in IMPLANT_CATEGORIES:
        if owners.get((category,)) != "implant_contact":
            raise GateError(
                f"manifest does not retain {category} in implant_contact"
            )

    return hashlib.sha256(canonical).hexdigest(), counts


def main() -> int:
    args = parse_args()
    work = Path(
        tempfile.mkdtemp(
            prefix="freepdk45-owner-split-cpu-gate.",
            dir=os.environ.get("TMPDIR", "/tmp"),
        )
    )
    success = False
    try:
        klayout = executable(args.klayout)
        source_deck = require_file(args.deck, "source deck")
        fixture_script = require_file(FIXTURE_SCRIPT, "fixture generator")
        split_script = require_file(SPLIT_SCRIPT, "owner transform")
        merger = require_file(MERGER, "strict report merger")
        environment = runtime_environment(work)

        fixture = work / "contact1-5.gds"
        run(
            [
                str(klayout),
                "-b",
                "-r",
                str(fixture_script),
                "-rd",
                f"output={fixture}",
            ],
            log=work / "fixture.log",
            environment=environment,
        )
        if not fixture.is_file() or fixture.stat().st_size == 0:
            raise GateError("fixture generator produced no GDS")

        split_deck = work / "freepdk45-contact-owner-split.lydrc"
        run(
            [
                sys.executable,
                str(split_script),
                "--split-implant-contact",
                str(source_deck),
                str(split_deck),
            ],
            log=work / "transform.log",
            environment=environment,
        )

        reports = work / "reports"
        reports.mkdir()
        source_all = reports / "source-all.lyrdb"
        split_all = reports / "split-all.lyrdb"
        run_drc(
            klayout,
            source_deck,
            fixture,
            source_all,
            shard="all",
            environment=environment,
        )
        run_drc(
            klayout,
            split_deck,
            fixture,
            split_all,
            shard="all",
            environment=environment,
        )
        assert_same_report(
            source_all,
            split_all,
            "source and transformed all-mode reports",
        )

        shard_reports: dict[str, Path] = {}
        for shard in SHARDS:
            report = reports / f"{shard}.lyrdb"
            run_drc(
                klayout,
                split_deck,
                fixture,
                report,
                shard=shard,
                environment=environment,
            )
            shard_reports[shard] = report
        assert_no_cuda_telemetry(reports)

        manifest = work / "manifest.json"
        manifest_command = [
            sys.executable,
            str(merger),
            "manifest",
            "--schema",
            str(split_all),
            "--deck",
            str(split_deck),
        ]
        for shard in SHARDS:
            manifest_command.extend(
                ["--shard", f"{shard}={shard_reports[shard]}"]
            )
        manifest_command.extend(["-o", str(manifest)])
        run(
            manifest_command,
            log=work / "manifest.log",
            environment=environment,
        )

        merged = reports / "merged.lyrdb"
        merge_command = [
            sys.executable,
            str(merger),
            "merge",
            "--manifest",
            str(manifest),
            "--deck",
            str(split_deck),
        ]
        for shard in SHARDS:
            merge_command.extend(
                ["--shard", f"{shard}={shard_reports[shard]}"]
            )
        merge_command.extend(["-o", str(merged)])
        run(
            merge_command,
            log=work / "merge.log",
            environment=environment,
        )

        semantic_sha256, counts = verify_evidence(
            split_deck=split_deck,
            split_all=split_all,
            merged=merged,
            manifest=manifest,
            shard_reports=shard_reports,
        )
        interesting = {
            category: counts[category]
            for category in CONTACT_CATEGORIES + IMPLANT_CATEGORIES
            if counts[category]
        }
        print(
            "FREEPDK45_OWNER_SPLIT_CPU_GATE PASS "
            f"shards={len(SHARDS)} "
            f"semantic_sha256={semantic_sha256} "
            f"nonempty={json.dumps(interesting, sort_keys=True)}"
        )
        success = True
        return 0
    except (GateError, OSError) as exc:
        print(f"FREEPDK45_OWNER_SPLIT_CPU_GATE FAIL: {exc}", file=sys.stderr)
        print(f"FREEPDK45_OWNER_SPLIT_CPU_GATE work={work}", file=sys.stderr)
        return 2
    finally:
        if success and not args.keep_work:
            shutil.rmtree(work)
        elif success:
            print(f"FREEPDK45_OWNER_SPLIT_CPU_GATE work={work}")


if __name__ == "__main__":
    raise SystemExit(main())
