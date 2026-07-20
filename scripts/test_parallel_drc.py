#!/usr/bin/env python3
"""Tests for deterministic sharded DRC report merging and launching."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import xml.etree.ElementTree as ET
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import merge_sharded_lyrdb as merger  # noqa: E402
import run_parallel_drc as launcher  # noqa: E402


def _text(parent: ET.Element, tag: str, value: str | None = None) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = value
    return element


def _cell(
    parent: ET.Element,
    name: str,
    layout_name: str = "layout",
    variant: str = "",
    references: tuple[tuple[str, str], ...] = (),
) -> None:
    element = ET.SubElement(parent, "cell")
    _text(element, "name", name)
    _text(element, "variant", variant)
    _text(element, "layout-name", layout_name)
    reference_root = ET.SubElement(element, "references")
    for parent_name, transform in references:
        reference = ET.SubElement(reference_root, "ref")
        _text(reference, "parent", parent_name)
        _text(reference, "trans", transform)


def _item(parent: ET.Element, item: dict[str, object]) -> None:
    element = ET.SubElement(parent, "item")
    _text(element, "tags", str(item.get("tags", "")))
    # Dotted one-component category names use KLayout's quoted word syntax;
    # an unquoted dot would denote a nested category path.
    _text(
        element,
        "category",
        str(item.get("category_ref", f"'{item['category']}'")),
    )
    _text(element, "cell", str(item.get("cell", "TOP")))
    _text(element, "visited", str(item.get("visited", "false")))
    _text(element, "multiplicity", str(item.get("multiplicity", "1")))
    _text(element, "comment", str(item["marker"]))
    _text(element, "image", str(item.get("image", "")))
    values = ET.SubElement(element, "values")
    for value in item.get("values", ()):
        _text(values, "value", str(value))


def write_report(
    path: Path,
    categories: list[str | tuple[str, ...]],
    items: list[dict[str, object]],
    *,
    category_descriptions: dict[str, str] | None = None,
    cells: list[tuple[object, ...]] | None = None,
    top_cell: str = "TOP",
    original_file: str = "/fixture/design.gds",
) -> None:
    """Write a compact but schema-shaped KLayout report database."""

    descriptions = category_descriptions or {}
    root = ET.Element("report-database")
    _text(root, "description", "parallel DRC fixture")
    _text(root, "original-file", original_file)
    _text(root, "generator", "drc: script='/fixture/deck.lydrc'")
    _text(root, "top-cell", top_cell)

    tags = ET.SubElement(root, "tags")
    tag = ET.SubElement(tags, "tag")
    _text(tag, "name", "review")
    _text(tag, "description", "fixture tag")

    category_root = ET.SubElement(root, "categories")
    category_containers: dict[tuple[str, ...], ET.Element] = {(): category_root}
    for category_spec in categories:
        category_path = (
            (category_spec,) if isinstance(category_spec, str) else category_spec
        )
        parent = category_containers[category_path[:-1]]
        name = category_path[-1]
        category = ET.SubElement(parent, "category")
        _text(category, "name", name)
        description_key = ".".join(category_path)
        _text(
            category,
            "description",
            descriptions.get(description_key, f"description {description_key}"),
        )
        category_containers[category_path] = ET.SubElement(category, "categories")

    cell_root = ET.SubElement(root, "cells")
    for cell_spec in cells or [("TOP", "layout"), ("CHILD", "layout")]:
        name = str(cell_spec[0])
        layout_name = str(cell_spec[1])
        variant = str(cell_spec[2]) if len(cell_spec) > 2 else ""
        references = cell_spec[3] if len(cell_spec) > 3 else ()
        _cell(cell_root, name, layout_name, variant, references)

    item_root = ET.SubElement(root, "items")
    for item in items:
        _item(item_root, item)

    ET.indent(root, space=" ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def category_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        element.findtext("name", default="")
        for element in root.findall("./categories/category")
    ]


def cell_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        element.findtext("name", default="")
        for element in root.findall("./cells/cell")
    ]


def item_markers(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        element.findtext("comment", default="")
        for element in root.findall("./items/item")
    ]


def item_for_marker(path: Path, marker: str) -> ET.Element:
    for element in ET.parse(path).getroot().findall("./items/item"):
        if element.findtext("comment") == marker:
            return element
    raise AssertionError(f"item marker {marker!r} was not found in {path}")


def element_signature(element: ET.Element) -> tuple[object, ...]:
    """Return a whitespace-insensitive signature for one complete XML subtree."""

    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(element_signature(child) for child in element),
    )


class ReportFixture(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.directory = Path(self._temporary_directory.name)

        self.reference = self.directory / "reference.lyrdb"
        self.odd = self.directory / "odd.lyrdb"
        self.even = self.directory / "even.lyrdb"
        self.manifest = self.directory / "manifest.json"

        self.items = {
            "A-1": {
                "marker": "A-1",
                "category": "RULE.A",
                "cell": "TOP",
                "tags": "#review",
                "visited": "true",
                "multiplicity": "3",
                "image": "aW1hZ2UtYnl0ZXM=",
                "values": [
                    "edge-pair: (0,0;1,0)|(1,1;0,1)",
                    "[#review] text: payload <A&1>",
                ],
            },
            "A-2": {
                "marker": "A-2",
                "category": "RULE.A",
                "cell": "CHILD",
                "values": ["polygon: (0,0;0,2;2,2;2,0)"],
            },
            "B-1": {
                "marker": "B-1",
                "category": "RULE.B",
                "cell": "TOP",
                "values": ["box: (3,4;5,6)"],
            },
            "C-1": {
                "marker": "C-1",
                "category": "RULE.C",
                "cell": "CHILD",
                "values": ["edge: (7,8;9,10)"],
            },
            "D-1": {
                "marker": "D-1",
                "category": "RULE.D",
                "cell": "TOP",
                "values": ["text: final marker"],
            },
        }

        canonical_items = [
            self.items[name] for name in ("A-1", "A-2", "B-1", "C-1", "D-1")
        ]
        write_report(
            self.reference,
            ["RULE.A", "RULE.B", "RULE.C", "RULE.D"],
            canonical_items,
        )
        # Deliberately reverse both category and item groups inside each shard.
        write_report(
            self.odd,
            ["RULE.C", "RULE.A"],
            [self.items[name] for name in ("C-1", "A-1", "A-2")],
        )
        write_report(
            self.even,
            ["RULE.D", "RULE.B"],
            [self.items[name] for name in ("D-1", "B-1")],
        )
        merger.create_manifest(
            self.reference,
            [("odd", self.odd), ("even", self.even)],
            self.manifest,
        )

    def merge(
        self,
        reports: list[tuple[str, Path]] | None = None,
        name: str = "merged.lyrdb",
    ) -> Path:
        output = self.directory / name
        merger.merge_reports(
            self.manifest,
            reports or [("odd", self.odd), ("even", self.even)],
            output,
        )
        return output

    def rewrite_report(
        self,
        name: str,
        categories: list[str],
        items: list[dict[str, object]],
        **kwargs: object,
    ) -> Path:
        path = self.directory / name
        write_report(path, categories, items, **kwargs)
        return path


class MergeReportsTests(ReportFixture):
    def test_create_manifest_never_overwrites_an_input(self) -> None:
        before = self.reference.read_bytes()
        with self.assertRaises(ValueError):
            merger.create_manifest(
                self.reference,
                [("odd", self.odd), ("even", self.even)],
                self.reference,
            )
        self.assertEqual(self.reference.read_bytes(), before)

    def test_create_manifest_rejects_a_nonmatching_nonempty_union(self) -> None:
        incomplete = self.rewrite_report(
            "odd-incomplete-union.lyrdb",
            ["RULE.C", "RULE.A"],
            [self.items[name] for name in ("C-1", "A-1")],
        )
        with self.assertRaises(ValueError):
            merger.create_manifest(
                self.reference,
                [("odd", incomplete), ("even", self.even)],
                self.directory / "invalid-manifest.json",
            )

    def test_manifest_merge_interleaved_items_is_input_order_deterministic(self) -> None:
        with self.manifest.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        self.assertIsInstance(manifest, dict)
        manifest_text = json.dumps(manifest, sort_keys=True)
        for value in ("odd", "even", "RULE.A", "RULE.B", "RULE.C", "RULE.D"):
            self.assertIn(value, manifest_text)

        forward = self.merge(name="forward.lyrdb")
        reverse = self.merge(
            [("even", self.even), ("odd", self.odd)],
            name="reverse.lyrdb",
        )

        self.assertEqual(forward.read_bytes(), reverse.read_bytes())
        self.assertEqual(
            category_names(forward),
            ["RULE.A", "RULE.B", "RULE.C", "RULE.D"],
        )
        self.assertEqual(cell_names(forward), ["CHILD", "TOP"])
        self.assertEqual(
            item_markers(forward),
            ["A-1", "A-2", "B-1", "C-1", "D-1"],
        )

    def test_merge_preserves_the_whole_item_payload(self) -> None:
        merged = self.merge()
        source_item = item_for_marker(self.odd, "A-1")
        merged_item = item_for_marker(merged, "A-1")
        self.assertEqual(
            element_signature(merged_item),
            element_signature(source_item),
        )

    def test_merge_rejects_missing_category(self) -> None:
        incomplete = self.rewrite_report(
            "odd-missing.lyrdb",
            ["RULE.A"],
            [self.items[name] for name in ("A-1", "A-2")],
        )
        with self.assertRaises(ValueError):
            self.merge([("odd", incomplete), ("even", self.even)])

    def test_failed_merge_preserves_an_existing_output(self) -> None:
        incomplete = self.rewrite_report(
            "odd-missing-for-atomicity.lyrdb",
            ["RULE.A"],
            [self.items[name] for name in ("A-1", "A-2")],
        )
        output = self.directory / "existing-output.lyrdb"
        output.write_bytes(b"existing report sentinel")
        with self.assertRaises(ValueError):
            self.merge(
                [("odd", incomplete), ("even", self.even)],
                name=output.name,
            )
        self.assertEqual(output.read_bytes(), b"existing report sentinel")

    def test_merge_rejects_overlapping_category_ownership(self) -> None:
        overlapping = self.rewrite_report(
            "even-overlap.lyrdb",
            ["RULE.D", "RULE.B", "RULE.A"],
            [self.items[name] for name in ("D-1", "B-1", "A-1")],
        )
        with self.assertRaises(ValueError):
            self.merge([("odd", self.odd), ("even", overlapping)])

    def test_merge_rejects_conflicting_category_definition(self) -> None:
        conflicting = self.rewrite_report(
            "odd-category-conflict.lyrdb",
            ["RULE.C", "RULE.A"],
            [self.items[name] for name in ("C-1", "A-1", "A-2")],
            category_descriptions={"RULE.A": "a conflicting description"},
        )
        with self.assertRaises(ValueError):
            self.merge([("odd", conflicting), ("even", self.even)])

    def test_merge_accepts_identical_overlapping_cells(self) -> None:
        merged = self.merge()
        self.assertEqual(cell_names(merged), ["CHILD", "TOP"])

    def test_merge_rejects_missing_referenced_cell(self) -> None:
        odd_missing = self.rewrite_report(
            "odd-missing-cell.lyrdb",
            ["RULE.C", "RULE.A"],
            [self.items[name] for name in ("C-1", "A-1", "A-2")],
            cells=[("TOP", "layout")],
        )
        even_missing = self.rewrite_report(
            "even-missing-cell.lyrdb",
            ["RULE.D", "RULE.B"],
            [self.items[name] for name in ("D-1", "B-1")],
            cells=[("TOP", "layout")],
        )
        with self.assertRaises(ValueError):
            self.merge([("odd", odd_missing), ("even", even_missing)])

    def test_merge_rejects_conflicting_cell_definition(self) -> None:
        conflicting = self.rewrite_report(
            "even-cell-conflict.lyrdb",
            ["RULE.D", "RULE.B"],
            [self.items[name] for name in ("D-1", "B-1")],
            cells=[("TOP", "different-layout"), ("CHILD", "layout")],
        )
        with self.assertRaises(ValueError):
            self.merge([("odd", self.odd), ("even", conflicting)])

    def test_manifest_is_reusable_for_a_different_layout(self) -> None:
        new_items = {
            "A-new": {"marker": "A-new", "category": "RULE.A", "cell": "OTHER"},
            "D-new": {"marker": "D-new", "category": "RULE.D", "cell": "NEW"},
        }
        cells = [("OTHER", "other-layout"), ("NEW", "new-layout")]
        odd = self.rewrite_report(
            "other-layout-odd.lyrdb",
            ["RULE.C", "RULE.A"],
            [new_items["A-new"]],
            cells=cells,
            top_cell="OTHER",
            original_file="/fixture/other.gds",
        )
        even = self.rewrite_report(
            "other-layout-even.lyrdb",
            ["RULE.D", "RULE.B"],
            [new_items["D-new"]],
            cells=cells,
            top_cell="OTHER",
            original_file="/fixture/other.gds",
        )
        merged = self.merge([("even", even), ("odd", odd)], "other-layout.lyrdb")
        root = ET.parse(merged).getroot()
        self.assertEqual(root.findtext("top-cell"), "OTHER")
        self.assertEqual(item_markers(merged), ["A-new", "D-new"])

    def test_nested_and_quoted_category_paths_round_trip(self) -> None:
        parent = ("parent.name",)
        child = ("parent.name", "child'quoted")
        other = ("OTHER",)
        nested_item = {
            "marker": "nested",
            "category": child[-1],
            "category_ref": merger._category_name(child),
        }
        other_item = {
            "marker": "other",
            "category": other[-1],
            "category_ref": merger._category_name(other),
        }
        reference = self.directory / "nested-reference.lyrdb"
        first = self.directory / "nested-first.lyrdb"
        second = self.directory / "nested-second.lyrdb"
        manifest = self.directory / "nested-manifest.json"
        output = self.directory / "nested-merged.lyrdb"
        write_report(reference, [parent, child, other], [nested_item, other_item])
        write_report(first, [parent, child], [nested_item])
        write_report(second, [other], [other_item])
        merger.create_manifest(
            reference, [("first", first), ("second", second)], manifest
        )
        merger.merge_reports(
            manifest, [("second", second), ("first", first)], output
        )
        root = ET.parse(output).getroot()
        self.assertEqual(
            root.findtext("./categories/category/name"), "parent.name"
        )
        self.assertEqual(
            root.findtext("./categories/category/categories/category/name"),
            "child'quoted",
        )
        self.assertEqual(item_markers(output), ["nested", "other"])

    def test_cell_variant_and_reference_order_are_preserved(self) -> None:
        references = (
            ("P2", "r0 *1 2,0"),
            ("P1", "r0 *1 1,0"),
        )
        cells = [
            ("P1", "parent-one"),
            ("P2", "parent-two"),
            ("CHILD", "child-layout", "v1", references),
        ]
        item_a = {
            "marker": "variant-a",
            "category": "RULE.A",
            "cell": "CHILD:v1",
        }
        item_b = {
            "marker": "variant-b",
            "category": "RULE.B",
            "cell": "CHILD:v1",
        }
        reference = self.directory / "refs-reference.lyrdb"
        first = self.directory / "refs-first.lyrdb"
        second = self.directory / "refs-second.lyrdb"
        manifest = self.directory / "refs-manifest.json"
        output = self.directory / "refs-merged.lyrdb"
        write_report(reference, ["RULE.A", "RULE.B"], [item_a, item_b], cells=cells, top_cell="P1")
        write_report(first, ["RULE.A"], [item_a], cells=cells, top_cell="P1")
        write_report(second, ["RULE.B"], [item_b], cells=cells, top_cell="P1")
        merger.create_manifest(
            reference, [("first", first), ("second", second)], manifest
        )
        merger.merge_reports(
            manifest, [("second", second), ("first", first)], output
        )
        root = ET.parse(output).getroot()
        child = next(
            cell
            for cell in root.findall("./cells/cell")
            if cell.findtext("name") == "CHILD"
        )
        self.assertEqual(child.findtext("variant"), "v1")
        self.assertEqual(
            [
                (ref.findtext("parent"), ref.findtext("trans"))
                for ref in child.findall("./references/ref")
            ],
            list(references),
        )


class LauncherTests(ReportFixture):
    def setUp(self) -> None:
        super().setUp()
        self.deck = self.directory / "deck.lydrc"
        self.layout = self.directory / "design.gds"
        self.deck.write_text("# fake deck\n", encoding="utf-8")
        self.layout.write_bytes(b"fake-layout")
        merger.create_manifest(
            self.reference,
            [("odd", self.odd), ("even", self.even)],
            self.manifest,
            deck_path=self.deck,
        )

        self.fake_reports = self.directory / "fake-reports"
        self.fake_reports.mkdir()
        (self.fake_reports / "odd.lyrdb").write_bytes(self.odd.read_bytes())
        (self.fake_reports / "even.lyrdb").write_bytes(self.even.read_bytes())

        self.executable = self.directory / "fake-klayout"
        self.executable.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import os
                from pathlib import Path
                import shutil
                import sys
                import time

                arguments = sys.argv[1:]
                runtime = {}
                for index, argument in enumerate(arguments):
                    if argument == "-rd":
                        key, value = arguments[index + 1].split("=", 1)
                        runtime[key] = value

                shard = runtime["drc_shard"]
                print(f"fake shard {shard}")
                if shard == "bad":
                    print("intentional fake failure")
                    raise SystemExit(7)
                if shard == "missing":
                    print("intentional missing report")
                    raise SystemExit(0)
                if shard == "slow":
                    Path(os.environ["FAKE_SLOW_PID"]).write_text(
                        str(os.getpid()), encoding="utf-8"
                    )
                    time.sleep(30)

                source = Path(os.environ["FAKE_DRC_REPORTS"]) / f"{shard}.lyrdb"
                shutil.copyfile(source, runtime["output"])
                """
            ),
            encoding="utf-8",
        )
        self.executable.chmod(
            self.executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

    def arguments(
        self,
        output: Path,
        shards: tuple[str, ...],
        manifest: Path | None = None,
    ) -> list[str]:
        arguments = [
            "--klayout",
            str(self.executable),
            "--deck",
            str(self.deck),
            "--input",
            str(self.layout),
            "--top-cell",
            "TOP",
            "--output",
            str(output),
            "--manifest",
            str(manifest or self.manifest),
            "--jobs",
            "2",
        ]
        for shard in shards:
            arguments.extend(("--shard", shard))
        return arguments

    def test_fake_executable_success_runs_and_merges(self) -> None:
        output = self.directory / "launcher-merged.lyrdb"
        output.write_bytes(b"existing report sentinel")
        output.chmod(0o640)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"FAKE_DRC_REPORTS": str(self.fake_reports)},
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(self.arguments(output, ("even", "odd")))

        self.assertEqual(result, 0, stderr.getvalue())
        self.assertTrue(output.is_file())
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)
        self.assertEqual(
            item_markers(output),
            ["A-1", "A-2", "B-1", "C-1", "D-1"],
        )
        self.assertIn("merged report", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_fake_executable_failure_is_reported_without_output(self) -> None:
        output = self.directory / "must-not-exist.lyrdb"
        bad_manifest = self.directory / "bad-manifest.json"
        merger.create_manifest(
            self.reference,
            [("odd", self.odd), ("bad", self.even)],
            bad_manifest,
            deck_path=self.deck,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"FAKE_DRC_REPORTS": str(self.fake_reports)},
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(
                self.arguments(output, ("odd", "bad"), manifest=bad_manifest)
            )

        self.assertEqual(result, 1)
        self.assertFalse(output.exists())
        self.assertIn("shard 'bad' failed with exit code 7", stderr.getvalue())
        self.assertIn("intentional fake failure", stderr.getvalue())

    def test_deck_mismatch_fails_before_launch(self) -> None:
        output = self.directory / "deck-mismatch.lyrdb"
        self.deck.write_text("# modified fake deck\n", encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = launcher.main(self.arguments(output, ("odd", "even")))
        self.assertEqual(result, 1)
        self.assertFalse(output.exists())
        self.assertIn("deck SHA-256 differs", stderr.getvalue())

    def test_success_without_report_is_a_failure(self) -> None:
        output = self.directory / "missing-report.lyrdb"
        output.write_bytes(b"existing report sentinel")
        missing_manifest = self.directory / "missing-manifest.json"
        merger.create_manifest(
            self.reference,
            [("odd", self.odd), ("missing", self.even)],
            missing_manifest,
            deck_path=self.deck,
        )
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"FAKE_DRC_REPORTS": str(self.fake_reports)}),
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(
                self.arguments(
                    output, ("odd", "missing"), manifest=missing_manifest
                )
            )
        self.assertEqual(result, 1)
        self.assertEqual(output.read_bytes(), b"existing report sentinel")
        self.assertIn("did not produce a report", stderr.getvalue())
        self.assertIn("intentional missing report", stderr.getvalue())

    def test_sigterm_cleans_up_a_running_child(self) -> None:
        output = self.directory / "terminated-output.lyrdb"
        output.write_bytes(b"existing report sentinel")
        slow_manifest = self.directory / "slow-manifest.json"
        merger.create_manifest(
            self.reference,
            [("odd", self.odd), ("slow", self.even)],
            slow_manifest,
            deck_path=self.deck,
        )
        pid_file = self.directory / "slow.pid"
        environment = os.environ.copy()
        environment.update(
            {
                "FAKE_DRC_REPORTS": str(self.fake_reports),
                "FAKE_SLOW_PID": str(pid_file),
            }
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_parallel_drc.py"),
                *self.arguments(
                    output, ("odd", "slow"), manifest=slow_manifest
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        deadline = time.monotonic() + 3.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(pid_file.exists(), "slow child did not start")
        slow_pid = int(pid_file.read_text(encoding="utf-8"))
        process.terminate()
        _stdout, stderr = process.communicate(timeout=8.0)
        self.assertEqual(process.returncode, 128 + signal.SIGTERM, stderr)
        self.assertEqual(output.read_bytes(), b"existing report sentinel")
        self.assertIn("terminated by signal", stderr)
        with self.assertRaises(ProcessLookupError):
            os.kill(slow_pid, 0)


if __name__ == "__main__":
    unittest.main()
