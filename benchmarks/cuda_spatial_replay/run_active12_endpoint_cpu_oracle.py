#!/usr/bin/env python3
"""Run synthetic ACTIVE.1/.2 endpoint fixtures through KLayout's CPU rules."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


ORIENTATIONS = (
    "right_up",
    "left_up",
    "right_down",
    "left_down",
    "swap_right_up",
    "swap_left_up",
    "swap_right_down",
    "swap_left_down",
)
LIMITS = ("minus", "equal", "plus")


def expectations() -> dict[str, tuple[bool, bool]]:
    result: dict[str, tuple[bool, bool]] = {}
    for orientation in ORIENTATIONS:
        for limit in LIMITS:
            result[f"space_corner_{orientation}_{limit}"] = (
                False,
                limit == "minus",
            )
            result[f"width_corner_{orientation}_{limit}"] = (
                limit == "minus",
                False,
            )
            result[
                f"convex_width_facing_space_{orientation}_{limit}"
            ] = (
                False,
                limit == "minus",
            )
            result[f"projection_touch_space_{orientation}_{limit}"] = (
                False,
                limit == "minus",
            )
    for limit in LIMITS:
        result[f"same_rectangle_width_{limit}"] = (
            limit == "minus",
            False,
        )
        result[f"hole_wall_width_{limit}"] = (
            limit == "minus",
            False,
        )
    result["duplicate_union"] = (False, False)
    result["overlap_union"] = (False, False)
    result["edge_touch_union"] = (False, False)
    result["point_touch_union"] = (True, True)
    result["mixed_topology"] = (True, False)
    return result


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--klayout", type=Path, required=True)
    parser.add_argument(
        "--fixture-script",
        type=Path,
        default=here / "active12_endpoint_cpu_fixture.rb",
    )
    parser.add_argument(
        "--deck",
        type=Path,
        default=here / "active12_endpoint_cpu_oracle.drc",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--keep-work", action="store_true")
    return parser.parse_args()


def cpu_environment(work: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("KLAYOUT_CUDA_"):
            environment.pop(name)
    environment.update(
        {
            "HOME": str(work / "home"),
            "KLAYOUT_HOME": str(work / "klayout-home"),
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    return environment


def run_checked(command: list[str], environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def category_presence(report: Path) -> tuple[bool, bool]:
    root = ET.parse(report).getroot()
    found = {"ACTIVE.1": False, "ACTIVE.2": False}
    items = root.find("items")
    if items is None:
        raise RuntimeError(f"report has no items element: {report}")
    for item in items.findall("item"):
        category = item.findtext("category")
        if category is not None:
            category = category.strip("'\"")
        if category in found:
            found[category] = True
    return found["ACTIVE.1"], found["ACTIVE.2"]


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.jobs > 32:
        raise SystemExit("--jobs must be from 1 through 32")
    for path, description in (
        (args.klayout, "KLayout executable"),
        (args.fixture_script, "fixture generator"),
        (args.deck, "CPU oracle deck"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {description}: {path}")

    work = Path(tempfile.mkdtemp(prefix="active12-endpoint-cpu-oracle."))
    preserve_work = args.keep_work
    if preserve_work:
        print(f"ACTIVE12_ENDPOINT_CPU_ORACLE work={work}")
    try:
        (work / "home").mkdir()
        (work / "klayout-home").mkdir()
        environment = cpu_environment(work)
        layout = work / "active12-endpoints.gds"
        generator_log = run_checked(
            [
                str(args.klayout),
                "-b",
                "-r",
                str(args.fixture_script),
                "-rd",
                f"output={layout}",
            ],
            environment,
        )
        (work / "fixture.log").write_text(generator_log, encoding="utf-8")
        if not layout.is_file():
            raise RuntimeError("fixture generator did not publish a layout")

        expected = expectations()

        def run_case(
            item: tuple[str, tuple[bool, bool]],
        ) -> tuple[str, tuple[bool, bool]]:
            name, _ = item
            report = work / f"{name}.lyrdb"
            log = run_checked(
                [
                    str(args.klayout),
                    "-b",
                    "-r",
                    str(args.deck),
                    "-rd",
                    f"input={layout}",
                    "-rd",
                    f"topcell={name}",
                    "-rd",
                    f"output={report}",
                ],
                environment,
            )
            (work / f"{name}.log").write_text(log, encoding="utf-8")
            return name, category_presence(report)

        actual: dict[str, tuple[bool, bool]] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.jobs, len(expected))
        ) as executor:
            for name, presence in executor.map(run_case, expected.items()):
                actual[name] = presence

        failures = [
            f"{name}: expected width={want[0]} space={want[1]}, "
            f"actual width={actual[name][0]} space={actual[name][1]}"
            for name, want in expected.items()
            if actual.get(name) != want
        ]
        if failures:
            raise RuntimeError(
                "ACTIVE.1/.2 CPU oracle mismatch:\n" + "\n".join(failures)
            )
        print(
            "ACTIVE12_ENDPOINT_CPU_ORACLE PASS"
            f" cases={len(expected)}"
            f" orientations={len(ORIENTATIONS)}"
            " thresholds=minus,equal,plus"
            " same_rectangle=3 hole=3"
            " canonical_union=duplicate,overlap,edge-touch,point-touch"
            " mixed_topology=1 projection_touch=24"
        )
        return 0
    finally:
        if not preserve_work and work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    raise SystemExit(main())
