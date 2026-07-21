#!/usr/bin/env python3
"""Tests for deterministic sharded DRC report merging and launching."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
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


def fake_runtime_provenance(
    executable: str,
    environment: object,
    **kwargs: object,
) -> dict[str, object]:
    """Keep launcher tests focused on orchestration, not ELF inspection."""

    del environment, kwargs
    path = Path(executable).resolve(strict=True)
    contents = path.read_bytes()
    executable_sha256 = hashlib.sha256(contents).hexdigest()
    canonical_sha256 = hashlib.sha256(
        b"launcher-test-runtime-v1\0" + contents
    ).hexdigest()
    return {
        "format": "klayout-runtime-provenance",
        "format_version": 1,
        "canonical_sha256": canonical_sha256,
        "identity": {
            "schema": "launcher-test-runtime-v1",
            "canonical_sha256": canonical_sha256,
        },
        "executable": {
            "resolved_path": str(path),
            "size_bytes": len(contents),
            "sha256": executable_sha256,
        },
    }


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


def process_is_running(pid: int) -> bool:
    """Treat a dead-but-not-yet-reaped Linux process as no longer running."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
        except FileNotFoundError:
            return False
        if len(fields) > 2 and fields[2] == "Z":
            return False
    return True


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
        runtime_collector_patcher = mock.patch.object(
            launcher,
            "collect_runtime_provenance",
            side_effect=fake_runtime_provenance,
        )
        self.runtime_collector = runtime_collector_patcher.start()
        self.addCleanup(runtime_collector_patcher.stop)
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
                import subprocess
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
                    if os.environ.get("FAKE_SLOW_GRANDCHILD_PID"):
                        grandchild = subprocess.Popen(
                            [sys.executable, "-c", "import time; time.sleep(30)"]
                        )
                        Path(os.environ["FAKE_SLOW_GRANDCHILD_PID"]).write_text(
                            str(grandchild.pid), encoding="utf-8"
                        )
                    time.sleep(30)
                if os.environ.get("FAKE_DRC_DELAY"):
                    time.sleep(float(os.environ["FAKE_DRC_DELAY"]))

                top_cell_key = os.environ.get("FAKE_TOP_CELL_RD_KEY", "topcell")
                output_key = os.environ.get("FAKE_OUTPUT_RD_KEY", "output")
                if top_cell_key not in runtime:
                    print(f"missing expected top-cell key {top_cell_key}")
                    raise SystemExit(8)
                if output_key not in runtime:
                    print(f"missing expected output key {output_key}")
                    raise SystemExit(9)

                source = Path(os.environ["FAKE_DRC_REPORTS"]) / f"{shard}.lyrdb"
                shutil.copyfile(source, runtime[output_key])
                """
            ),
            encoding="utf-8",
        )
        self.executable.chmod(
            self.executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

    def test_environment_record_captures_performance_controls_stably(self) -> None:
        relevant = {
            "GLIBC_TUNABLES": "glibc.malloc.tcache_count=32",
            "MALLOC_ARENA_MAX": "2",
            "JEMALLOC_BACKGROUND_THREAD": "true",
            "TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES": "1048576",
            "OMP_NUM_THREADS": "4",
            "GOMP_CPU_AFFINITY": "0-3",
            "KMP_AFFINITY": "compact",
            "TBB_NUM_THREADS": "4",
            "RAYON_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUSTOM_NUM_THREADS": "6",
            "OMP_TOKEN": "do-not-publish-me",
            "NOT_PERFORMANCE_RELEVANT": "omit-me",
        }
        with mock.patch.dict(os.environ, relevant, clear=True):
            environment = launcher._environment_record()

        self.assertEqual(list(environment), sorted(environment))
        for key, value in relevant.items():
            if key == "NOT_PERFORMANCE_RELEVANT":
                self.assertNotIn(key, environment)
            elif key == "OMP_TOKEN":
                self.assertEqual(
                    environment[key],
                    {
                        "redacted": True,
                        "value_sha256": hashlib.sha256(
                            value.encode("utf-8")
                        ).hexdigest(),
                    },
                )
            else:
                self.assertEqual(environment[key], value)
        # Fixed controls remain explicit JSON nulls when they are unset.
        self.assertIsNone(environment["LD_PRELOAD"])
        self.assertIsNone(environment["MALLOC_CONF"])
        self.assertIsNone(environment["NUMEXPR_MAX_THREADS"])

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

    def isolated_launcher_command(self, arguments: list[str]) -> list[str]:
        """Run the real launcher process while replacing only ELF collection."""

        bootstrap = (
            "import sys; "
            f"sys.path.insert(0, {str(SCRIPT_DIR)!r}); "
            "from unittest import mock; "
            "import run_parallel_drc as launcher; "
            "from test_parallel_drc import fake_runtime_provenance; "
            "patcher = mock.patch.object(launcher, 'collect_runtime_provenance', "
            "side_effect=fake_runtime_provenance); "
            "patcher.start(); "
            "raise SystemExit(launcher.main(sys.argv[1:]))"
        )
        return [sys.executable, "-c", bootstrap, *arguments]

    def test_build_command_supports_top_cell_and_report_rd_keys(self) -> None:
        output = self.directory / "sky-convention-output.lyrdb"
        arguments = self.arguments(output, ("odd",))
        arguments.extend(
            (
                "--top-cell-rd-key",
                "top_cell",
                "--output-rd-key",
                "report",
            )
        )
        args = launcher.parse_args(arguments)
        private_report = self.directory / "private-report.lyrdb"
        spec = launcher.ShardSpec(
            index=0,
            name="odd",
            report=private_report,
            log=self.directory / "private-report.log",
        )

        command = launcher.build_command(args, spec)
        assignments = {
            command[index + 1].split("=", 1)[0]: command[index + 1].split("=", 1)[1]
            for index, value in enumerate(command)
            if value == "-rd"
        }
        self.assertEqual(assignments["input"], str(self.layout))
        self.assertEqual(assignments["top_cell"], "TOP")
        self.assertEqual(assignments["report"], str(private_report))
        self.assertEqual(assignments["drc_shard"], "odd")
        self.assertNotIn("topcell", assignments)
        self.assertNotIn("output", assignments)

    def test_build_command_forwards_each_generic_rd_exactly_once(self) -> None:
        output = self.directory / "generic-rd-output.lyrdb"
        arguments = self.arguments(output, ("odd",))
        arguments.extend(("--rd", "alpha=one", "--rd", "beta=two"))
        args = launcher.parse_args(arguments)
        spec = launcher.ShardSpec(
            index=0,
            name="odd",
            report=self.directory / "generic-rd-private.lyrdb",
            log=self.directory / "generic-rd-private.log",
        )

        command = launcher.build_command(args, spec)
        self.assertEqual(command.count("alpha=one"), 1)
        self.assertEqual(command.count("beta=two"), 1)
        self.assertEqual(command.count("-rd"), 6)

    def test_alternate_orchestration_rd_keys_are_reserved(self) -> None:
        for key in ("top_cell", "report"):
            with self.subTest(key=key), self.assertRaises(argparse.ArgumentTypeError):
                launcher.parse_rd(f"{key}=must-not-override")

    def test_fake_executable_success_runs_and_merges(self) -> None:
        output = self.directory / "launcher-merged.lyrdb"
        output.write_bytes(b"existing report sentinel")
        output.chmod(0o640)
        arguments = self.arguments(output, ("even", "odd"))
        arguments.extend(
            (
                "--cohort-id",
                "simultaneous-5-way-a",
                "--replicate-index",
                "3",
                "--replicate-count",
                "5",
            )
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "FAKE_DRC_REPORTS": str(self.fake_reports),
                    "FAKE_DRC_DELAY": "0.12",
                    "MALLOC_CONF": "background_thread:true",
                    "OMP_NUM_THREADS": "7",
                },
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(arguments)

        self.assertEqual(result, 0, stderr.getvalue())
        self.assertTrue(output.is_file())
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o640)
        self.assertEqual(
            item_markers(output),
            ["A-1", "A-2", "B-1", "C-1", "D-1"],
        )
        self.assertIn("merged report", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

        sidecar = Path(f"{output}.metadata.json")
        with sidecar.open("r", encoding="utf-8") as stream:
            provenance = json.load(stream)
        self.assertEqual(provenance["status"], "success")
        self.assertEqual(
            provenance["format"], "klayout-parallel-drc-provenance"
        )
        self.assertEqual(provenance["format_version"], 1)
        self.assertEqual(
            provenance["inputs"]["deck"]["sha256"],
            hashlib.sha256(self.deck.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["inputs"]["input_layout"]["sha256"],
            hashlib.sha256(self.layout.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["inputs"]["manifest"]["sha256"],
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["runtime_bundle"]["executable"]["sha256"],
            hashlib.sha256(self.executable.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["output_report"]["sha256"],
            hashlib.sha256(output.read_bytes()).hexdigest(),
        )
        self.assertEqual(provenance["environment"]["OMP_NUM_THREADS"], "7")
        self.assertEqual(
            provenance["environment"]["MALLOC_CONF"],
            "background_thread:true",
        )
        self.assertEqual(provenance["configuration"]["shard_names"], ["even", "odd"])
        self.assertEqual(provenance["configuration"]["jobs_effective"], 2)
        self.assertEqual(
            provenance["configuration"]["cohort"],
            {
                "id": "simultaneous-5-way-a",
                "replicate_count": 5,
                "replicate_index": 3,
            },
        )
        if hasattr(os, "sched_getaffinity"):
            self.assertEqual(
                provenance["host"]["cpu_affinity"],
                sorted(os.sched_getaffinity(0)),
            )
        self.assertEqual(len(provenance["invocation"]["child_commands"]), 2)
        self.assertEqual(len(provenance["shards"]), 2)
        timing = provenance["timing"]
        self.assertGreater(timing["aggregate_wall_seconds"], 0)
        self.assertEqual(
            timing["aggregate_wall_seconds"],
            timing["child_merge_workload_wall_seconds"],
        )
        self.assertGreater(timing["provenance_prehash_wall_seconds"], 0)
        self.assertGreater(timing["provenance_verification_wall_seconds"], 0)
        self.assertGreater(
            timing["launcher_wall_seconds"], timing["children_wall_seconds"]
        )
        self.assertEqual(
            timing["full_launcher_wall_seconds"],
            timing["launcher_wall_seconds"],
        )
        self.assertTrue(
            provenance["measurement_semantics"][
                "prehash_before_child_merge_workload"
            ]
        )
        self.assertIn("warm-cache", provenance["measurement_semantics"]["cache_state"])
        for shard in provenance["shards"]:
            self.assertGreaterEqual(shard["wall_seconds"], 0)
            self.assertEqual(shard["returncode"], 0)
            if sys.platform.startswith("linux"):
                self.assertGreater(shard["peak_rss_kb"], 0)
                self.assertEqual(
                    shard["peak_rss_source"],
                    "linux_proc_status_vm_hwm_direct_process_polling",
                )
            else:
                self.assertIsNone(shard["peak_rss_kb"])
                self.assertEqual(shard["peak_rss_source"], "unavailable")
            self.assertIn("descendants are excluded", shard["peak_rss_caveat"])
            self.assertIsNone(shard["report"]["path"])
            self.assertIsNone(shard["log"]["path"])
            self.assertFalse(shard["report"]["retained"])
            self.assertFalse(shard["log"]["retained"])
            self.assertEqual(
                shard["command"],
                provenance["invocation"]["child_commands"][shard["index"]],
            )

    def test_explicit_metadata_records_retained_child_artifacts(self) -> None:
        output = self.directory / "retained-output.lyrdb"
        metadata = self.directory / "retained-provenance.json"
        arguments = self.arguments(output, ("odd", "even"))
        arguments.extend(("--keep-temp", "--metadata", str(metadata)))
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"FAKE_DRC_REPORTS": str(self.fake_reports)},
            ),
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(arguments)

        self.assertEqual(result, 0, stderr.getvalue())
        provenance = json.loads(metadata.read_text(encoding="utf-8"))
        temporary_directory = Path(
            provenance["temporary_artifacts"]["directory"]
        )
        self.addCleanup(shutil.rmtree, temporary_directory, True)
        self.assertTrue(provenance["temporary_artifacts"]["retained"])
        self.assertTrue(temporary_directory.is_dir())
        for shard in provenance["shards"]:
            self.assertTrue(Path(shard["report"]["path"]).is_file())
            self.assertTrue(Path(shard["log"]["path"]).is_file())
            self.assertTrue(shard["report"]["retained"])
            self.assertTrue(shard["log"]["retained"])

    def test_runtime_bundle_delegates_to_shared_fail_closed_collector(self) -> None:
        before = launcher._runtime_bundle(str(self.executable))
        self.executable.write_bytes(b"different executable implementation")
        self.executable.chmod(self.executable.stat().st_mode | stat.S_IXUSR)
        after = launcher._runtime_bundle(str(self.executable))

        self.assertEqual(self.runtime_collector.call_count, 2)
        self.assertEqual(
            self.runtime_collector.call_args_list[0].args[0],
            str(self.executable),
        )
        self.assertNotEqual(
            before["canonical_sha256"], after["canonical_sha256"]
        )

    def test_runtime_closure_cannot_be_a_publication_target(self) -> None:
        for target_kind in ("output", "metadata", "failure"):
            with self.subTest(target_kind=target_kind):
                output = self.directory / f"{target_kind}-output.lyrdb"
                metadata = self.directory / f"{target_kind}-metadata.json"
                if target_kind == "output":
                    runtime_path = output
                elif target_kind == "metadata":
                    runtime_path = metadata
                else:
                    runtime_path = launcher._failure_metadata_path(metadata)
                runtime_path.write_bytes(b"attested runtime plugin")
                arguments = self.arguments(output, ("odd",))
                arguments.extend(("--metadata", str(metadata)))
                args = launcher.parse_args(arguments)
                runtime_bundle = {
                    "runtime_plugins": {
                        "files": [
                            {"resolved_path": str(runtime_path.resolve())}
                        ]
                    }
                }
                with self.assertRaisesRegex(
                    ValueError, "must not overwrite a collected KLayout runtime"
                ):
                    launcher._validate_runtime_publication_targets(
                        args, runtime_bundle
                    )
                self.assertEqual(
                    runtime_path.read_bytes(), b"attested runtime plugin"
                )

    def test_target_lock_name_cannot_masquerade_as_a_plugin(self) -> None:
        plugin_target = self.directory / "db_plugins" / "libfixture.so.1"
        plugin_target.parent.mkdir()
        lock = launcher._lock_path(plugin_target)
        self.assertEqual(lock.parent, plugin_target.parent)
        self.assertNotIn(".so", lock.name)
        self.assertTrue(lock.name.endswith(".lock"))

    def test_atomic_json_preserves_existing_metadata_on_publish_failure(self) -> None:
        metadata = self.directory / "atomic-metadata.json"
        metadata.write_bytes(b"accepted metadata sentinel")
        with (
            mock.patch.object(launcher.os, "replace", side_effect=OSError("boom")),
            self.assertRaises(OSError),
        ):
            launcher._atomic_json(metadata, {"status": "success"})
        self.assertEqual(metadata.read_bytes(), b"accepted metadata sentinel")
        self.assertEqual(list(self.directory.glob(".atomic-metadata.json.*")), [])

    def test_failure_after_merge_preserves_existing_pair(self) -> None:
        output = self.directory / "prepublication-failure.lyrdb"
        metadata = Path(f"{output}.metadata.json")
        output.write_bytes(b"accepted report sentinel")
        metadata.write_bytes(b"accepted metadata sentinel")
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"FAKE_DRC_REPORTS": str(self.fake_reports)},
            ),
            mock.patch.object(
                launcher,
                "_stage_json",
                side_effect=OSError("metadata serialization failed after merge"),
            ) as stage_json,
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(self.arguments(output, ("odd", "even")))

        self.assertEqual(result, 1)
        self.assertEqual(output.read_bytes(), b"accepted report sentinel")
        self.assertEqual(metadata.read_bytes(), b"accepted metadata sentinel")
        self.assertGreaterEqual(stage_json.call_count, 1)
        self.assertEqual(list(self.directory.glob(f".{output.name}.stage-*")), [])
        self.assertIn("metadata serialization failed after merge", stderr.getvalue())

    def test_metadata_publication_failure_rolls_back_previous_pair(self) -> None:
        output = self.directory / "publication-rollback.lyrdb"
        metadata = Path(f"{output}.metadata.json")
        old_report = b"accepted report sentinel"
        old_metadata = b"accepted metadata sentinel"
        output.write_bytes(old_report)
        metadata.write_bytes(old_metadata)
        real_replace = os.replace
        injected = False
        replace_events: list[tuple[Path, Path]] = []

        def replace_with_one_failure(source: object, destination: object) -> None:
            nonlocal injected
            source_path = Path(source)
            destination_path = Path(destination)
            replace_events.append((source_path, destination_path))
            if (
                not injected
                and destination_path == metadata
                and ".stage-" in source_path.name
            ):
                injected = True
                raise OSError("injected success-metadata publication failure")
            real_replace(source, destination)

        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"FAKE_DRC_REPORTS": str(self.fake_reports)},
            ),
            mock.patch.object(launcher.os, "replace", replace_with_one_failure),
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(self.arguments(output, ("odd", "even")))

        self.assertTrue(injected)
        self.assertEqual(result, 1)
        self.assertEqual(output.read_bytes(), old_report)
        self.assertEqual(metadata.read_bytes(), old_metadata)
        metadata_invalidation = next(
            index
            for index, (source, destination) in enumerate(replace_events)
            if source == metadata and ".stale-" in destination.name
        )
        report_publication = next(
            index
            for index, (source, destination) in enumerate(replace_events)
            if destination == output and ".stage-" in source.name
        )
        self.assertLess(metadata_invalidation, report_publication)
        self.assertEqual(list(self.directory.glob("*.previous-*")), [])
        self.assertEqual(list(self.directory.glob("*.stale-*")), [])
        self.assertIn(
            "injected success-metadata publication failure", stderr.getvalue()
        )

    def test_backup_restore_failure_never_republishes_stale_metadata(self) -> None:
        output = self.directory / "backup-failure.lyrdb"
        metadata = self.directory / "backup-failure.metadata.json"
        report_stage = self.directory / ".report.stage"
        metadata_stage = self.directory / ".metadata.stage"
        output.write_bytes(b"old report")
        metadata.write_bytes(b"old metadata")
        report_stage.write_bytes(b"new report")
        metadata_stage.write_bytes(b"new metadata")
        real_replace = os.replace
        real_fsync_directory = launcher._fsync_directory
        fsync_calls = 0

        def fail_report_backup_fsync(directory: Path) -> None:
            nonlocal fsync_calls
            fsync_calls += 1
            if fsync_calls == 2:
                raise OSError("injected report-backup fsync failure")
            real_fsync_directory(directory)

        def fail_report_restore(source: object, destination: object) -> None:
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                destination_path == output
                and ".previous-" in source_path.name
            ):
                raise OSError("injected report restoration failure")
            real_replace(source, destination)

        with (
            mock.patch.object(
                launcher, "_fsync_directory", fail_report_backup_fsync
            ),
            mock.patch.object(launcher.os, "replace", fail_report_restore),
            self.assertRaisesRegex(
                OSError, "injected report-backup fsync failure"
            ),
        ):
            launcher._publish_staged_pair(
                report_stage, output, metadata_stage, metadata
            )

        self.assertFalse(output.exists())
        self.assertFalse(metadata.exists())
        self.assertEqual(
            len(list(self.directory.glob(".backup-failure.lyrdb.previous-*"))),
            1,
        )
        self.assertEqual(
            len(
                list(
                    self.directory.glob(
                        ".backup-failure.metadata.json.stale-*"
                    )
                )
            ),
            1,
        )

    def test_none_peak_rss_is_reported_as_unavailable_even_on_linux(self) -> None:
        self.assertEqual(launcher._peak_rss_source(None), "unavailable")

    def test_fake_executable_supports_sky130_rd_convention(self) -> None:
        output = self.directory / "sky-convention-merged.lyrdb"
        arguments = self.arguments(output, ("even", "odd"))
        arguments.extend(
            (
                "--top-cell-rd-key",
                "top_cell",
                "--output-rd-key",
                "report",
            )
        )
        for assignment in (
            "feol=true",
            "beol=true",
            "offgrid=true",
            "seal=true",
            "floating_met=false",
            "sram_exclude=false",
            "thr=4",
        ):
            arguments.extend(("--rd", assignment))

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "FAKE_DRC_REPORTS": str(self.fake_reports),
                    "FAKE_TOP_CELL_RD_KEY": "top_cell",
                    "FAKE_OUTPUT_RD_KEY": "report",
                },
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = launcher.main(arguments)

        self.assertEqual(result, 0, stderr.getvalue())
        self.assertIn("merged report", stdout.getvalue())
        self.assertEqual(
            item_markers(output),
            ["A-1", "A-2", "B-1", "C-1", "D-1"],
        )

    def test_fake_executable_failure_is_reported_without_output(self) -> None:
        output = self.directory / "must-not-exist.lyrdb"
        accepted_metadata = Path(f"{output}.metadata.json")
        accepted_metadata.write_bytes(b"accepted metadata sentinel")
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
        self.assertEqual(
            accepted_metadata.read_bytes(), b"accepted metadata sentinel"
        )
        failed_metadata = Path(f"{output}.metadata.failed.json")
        failure = json.loads(failed_metadata.read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["failure"]["type"], "ShardFailure")
        self.assertIsNone(failure["output_report"])
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
        failure_sidecar = Path(f"{output}.metadata.failed.json")
        failure = json.loads(failure_sidecar.read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["failure"]["type"], "ReportError")
        self.assertEqual(
            failure["inputs"]["deck"]["sha256"],
            hashlib.sha256(self.deck.read_bytes()).hexdigest(),
        )

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

    def test_concurrent_launchers_cannot_share_publication_targets(self) -> None:
        output = self.directory / "concurrent-output.lyrdb"
        metadata = Path(f"{output}.metadata.json")
        output.write_bytes(b"accepted report sentinel")
        metadata.write_bytes(b"accepted metadata sentinel")
        slow_manifest = self.directory / "concurrent-manifest.json"
        merger.create_manifest(
            self.reference,
            [("odd", self.odd), ("slow", self.even)],
            slow_manifest,
            deck_path=self.deck,
        )
        pid_file = self.directory / "concurrent-slow.pid"
        environment = os.environ.copy()
        environment.update(
            {
                "FAKE_DRC_REPORTS": str(self.fake_reports),
                "FAKE_SLOW_PID": str(pid_file),
            }
        )
        command = self.isolated_launcher_command(
            self.arguments(output, ("odd", "slow"), manifest=slow_manifest)
        )
        first = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        self.addCleanup(lambda: first.poll() is None and first.kill())
        deadline = time.monotonic() + 3.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(pid_file.exists(), "first launch did not reach child work")

        second = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            timeout=3.0,
            check=False,
        )
        self.assertEqual(second.returncode, 1)
        self.assertIn("already locked", second.stderr)
        self.assertNotIn("failure provenance", second.stderr)
        self.assertEqual(output.read_bytes(), b"accepted report sentinel")
        self.assertEqual(metadata.read_bytes(), b"accepted metadata sentinel")

        first.terminate()
        _stdout, first_stderr = first.communicate(timeout=8.0)
        self.assertEqual(first.returncode, 128 + signal.SIGTERM, first_stderr)
        self.assertEqual(list(self.directory.glob(".parallel-drc-*.lock")), [])

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
        grandchild_pid_file = self.directory / "slow-grandchild.pid"
        environment = os.environ.copy()
        environment.update(
            {
                "FAKE_DRC_REPORTS": str(self.fake_reports),
                "FAKE_SLOW_PID": str(pid_file),
                "FAKE_SLOW_GRANDCHILD_PID": str(grandchild_pid_file),
            }
        )
        process = subprocess.Popen(
            self.isolated_launcher_command(
                self.arguments(
                    output, ("odd", "slow"), manifest=slow_manifest
                )
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        deadline = time.monotonic() + 3.0
        while (
            (not pid_file.exists() or not grandchild_pid_file.exists())
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        self.assertTrue(pid_file.exists(), "slow child did not start")
        self.assertTrue(grandchild_pid_file.exists(), "slow grandchild did not start")
        slow_pid = int(pid_file.read_text(encoding="utf-8"))
        grandchild_pid = int(grandchild_pid_file.read_text(encoding="utf-8"))
        process.terminate()
        _stdout, stderr = process.communicate(timeout=8.0)
        self.assertEqual(process.returncode, 128 + signal.SIGTERM, stderr)
        self.assertEqual(output.read_bytes(), b"existing report sentinel")
        self.assertIn("terminated by signal", stderr)
        deadline = time.monotonic() + 2.0
        while (
            (process_is_running(slow_pid) or process_is_running(grandchild_pid))
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        self.assertFalse(process_is_running(slow_pid))
        self.assertFalse(process_is_running(grandchild_pid))


if __name__ == "__main__":
    unittest.main()
