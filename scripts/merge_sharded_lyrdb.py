#!/usr/bin/env python3
"""Create and apply strict manifests for process-sharded KLayout reports.

Two independent ``.lyrdb`` files contain their local category order, but not
the original order between shards.  This tool bootstraps an explicit *deck
contract* from one trusted full report, while proving that the supplied shard
reports are its exact semantic union.  Later runs use only that contract:
categories are emitted in canonical order and every category has one declared
owner.  The manifest is intentionally reusable for other layouts and top
cells checked by the same deck; it does not pin marker or cell contents.
For a cell present in multiple shards, each shard must reproduce the complete
ordered reference table; partial per-shard reference tables are rejected.

Items are copied as complete XML nodes, so tags, visited state, multiplicity,
comments, images, value order, and marker geometry are retained.  Item order
is canonicalized by category; within one category the owning shard's order is
preserved.  Marker order is not semantically significant in KLayout reports.

Only standard, uncompressed XML reports produced by KLayout are accepted.
The implementation intentionally fails closed on malformed metadata, category
overlap, conflicting cells/references, dangling references, or unknown item
targets.  The process launcher additionally guarantees that each input report
comes directly from the configured KLayout executable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import heapq
import json
import os
from pathlib import Path
import stat
import string
import sys
import tempfile
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


FORMAT_NAME = "klayout-sharded-lyrdb-manifest"
FORMAT_VERSION = 1
ROOT_FIELDS = (
    "description",
    "original-file",
    "generator",
    "top-cell",
    "tags",
    "categories",
    "cells",
    "items",
)
CATEGORY_FIELDS = ("name", "description", "categories")
CELL_FIELDS = ("name", "variant", "layout-name", "references")
REFERENCE_FIELDS = ("parent", "trans")
ITEM_FIELDS = (
    "tags",
    "category",
    "cell",
    "visited",
    "multiplicity",
    "comment",
    "image",
    "values",
)


class ReportError(ValueError):
    """A report, manifest, or shard union violates the strict contract."""


def _children(node: ET.Element) -> tuple[str, ...]:
    return tuple(child.tag for child in node)


def _require_children(node: ET.Element, expected: Sequence[str], context: str) -> None:
    actual = _children(node)
    if actual != tuple(expected):
        raise ReportError(
            f"{context}: expected child sequence {tuple(expected)!r}, got {actual!r}"
        )


def _leaf(node: ET.Element, name: str, context: str) -> str:
    child = node.find(name)
    if child is None:
        raise ReportError(f"{context}: missing {name!r}")
    if len(child):
        raise ReportError(f"{context}: {name!r} must be a leaf element")
    return child.text or ""


def _parse_report(path: str | os.PathLike[str]) -> ET.Element:
    report_path = Path(path)
    try:
        root = ET.parse(report_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ReportError(f"{report_path}: unable to read report: {exc}") from exc
    if root.tag != "report-database":
        raise ReportError(f"{report_path}: root is {root.tag!r}, not 'report-database'")
    if root.attrib:
        raise ReportError(f"{report_path}: report-database attributes are unsupported")
    for element in root.iter():
        if element.attrib:
            raise ReportError(
                f"{report_path}: attributes on {element.tag!r} are unsupported"
            )
    _require_children(root, ROOT_FIELDS, str(report_path))
    categories = _categories(root)
    category_names = {_category_name(path): path for path, _ in categories}
    if len(category_names) != len(categories):
        raise ReportError(f"{report_path}: category path serialization collision")
    cells = _cell_records(root)
    _ordered_cells(cells)
    _items(root, category_names, cells, str(report_path))
    return root


def _metadata(root: ET.Element) -> dict[str, str]:
    return {name: _leaf(root, name, "report metadata") for name in ROOT_FIELDS[:4]}


def _tags(root: ET.Element) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    container = root.find("tags")
    assert container is not None
    for index, tag in enumerate(container):
        if tag.tag != "tag":
            raise ReportError(f"tags[{index}]: expected 'tag', got {tag.tag!r}")
        _require_children(tag, ("name", "description"), f"tags[{index}]")
        result.append(
            (_leaf(tag, "name", f"tags[{index}]"), _leaf(tag, "description", f"tags[{index}]"))
        )
    if len({name for name, _ in result}) != len(result):
        raise ReportError("duplicate tag declaration")
    return tuple(result)


def _categories(root: ET.Element) -> list[tuple[tuple[str, ...], str]]:
    result: list[tuple[tuple[str, ...], str]] = []
    seen: set[tuple[str, ...]] = set()
    container = root.find("categories")
    assert container is not None

    def visit(parent: ET.Element, prefix: tuple[str, ...]) -> None:
        sibling_names: set[str] = set()
        for index, category in enumerate(parent):
            context = f"category {prefix!r}[{index}]"
            if category.tag != "category":
                raise ReportError(f"{context}: expected 'category', got {category.tag!r}")
            _require_children(category, CATEGORY_FIELDS, context)
            name = _leaf(category, "name", context)
            if name in sibling_names:
                raise ReportError(f"{context}: duplicate sibling name {name!r}")
            sibling_names.add(name)
            path = prefix + (name,)
            if path in seen:
                raise ReportError(f"duplicate category path {path!r}")
            seen.add(path)
            result.append((path, _leaf(category, "description", context)))
            nested = category.find("categories")
            assert nested is not None
            visit(nested, path)

    visit(container, ())
    return result


def _quote_component(name: str) -> str:
    """Match ``Category::path`` / ``tl::to_word_or_quoted_string(_, "_$")``."""

    safe = bool(name) and (name[0].isalpha() and name[0].isascii() or name[0] in "_$")
    safe = safe and all(
        (character.isalnum() and character.isascii()) or character in "_$"
        for character in name[1:]
    )
    if safe:
        return name

    quoted = ["'"]
    for byte in name.encode("utf-8"):
        character = chr(byte)
        if character in "'\\":
            quoted.extend(("\\", character))
        elif character == "\n":
            quoted.append("\\n")
        elif character == "\r":
            quoted.append("\\r")
        elif character == "\t":
            quoted.append("\\t")
        elif 0x20 <= byte < 0x7f:
            quoted.append(character)
        else:
            quoted.append(f"\\{byte:03o}")
    quoted.append("'")
    return "".join(quoted)


def _category_name(path: Sequence[str]) -> str:
    return ".".join(_quote_component(component) for component in path)


def _fingerprint(node: ET.Element) -> tuple[object, ...]:
    """Canonicalize meaningful XML while ignoring formatting indentation."""

    text = node.text or ""
    if len(node) and not text.strip():
        text = ""
    return (
        node.tag,
        tuple(sorted(node.attrib.items())),
        text,
        tuple(_fingerprint(child) for child in node),
    )


def _qname(name: str, variant: str) -> str:
    return f"{name}:{variant}" if variant else name


def _cell_records(root: ET.Element) -> dict[str, tuple[str, str, tuple[tuple[str, str], ...]]]:
    records: dict[str, tuple[str, str, tuple[tuple[str, str], ...]]] = {}
    container = root.find("cells")
    assert container is not None
    for index, cell in enumerate(container):
        context = f"cells[{index}]"
        if cell.tag != "cell":
            raise ReportError(f"{context}: expected 'cell', got {cell.tag!r}")
        _require_children(cell, CELL_FIELDS, context)
        name = _leaf(cell, "name", context)
        variant = _leaf(cell, "variant", context)
        layout_name = _leaf(cell, "layout-name", context)
        qname = _qname(name, variant)
        references_node = cell.find("references")
        assert references_node is not None
        references: list[tuple[str, str]] = []
        for reference_index, reference in enumerate(references_node):
            reference_context = f"{context}.references[{reference_index}]"
            if reference.tag != "ref":
                raise ReportError(
                    f"{reference_context}: expected 'ref', got {reference.tag!r}"
                )
            _require_children(reference, REFERENCE_FIELDS, reference_context)
            references.append(
                (
                    _leaf(reference, "parent", reference_context),
                    _leaf(reference, "trans", reference_context),
                )
            )
        # Reference order influences which sample hierarchy path KLayout shows
        # for a marker.  Preserve producer order rather than canonicalizing it.
        record = (name, layout_name, tuple(references))
        previous = records.get(qname)
        if previous is not None and previous != record:
            raise ReportError(f"conflicting duplicate cell {qname!r}")
        if previous is not None:
            raise ReportError(f"duplicate cell {qname!r}")
        records[qname] = record
    return records


def _merge_cells(
    roots: Iterable[ET.Element],
) -> dict[str, tuple[str, str, tuple[tuple[str, str], ...]]]:
    merged: dict[str, tuple[str, str, tuple[tuple[str, str], ...]]] = {}
    for root in roots:
        for qname, record in _cell_records(root).items():
            if qname in merged and merged[qname] != record:
                raise ReportError(f"shards disagree about cell {qname!r}")
            merged[qname] = record
    for qname, (_, _, references) in merged.items():
        for parent, _ in references:
            if parent not in merged:
                raise ReportError(f"cell {qname!r} references unknown parent {parent!r}")
    return merged


def _ordered_cells(
    records: Mapping[str, tuple[str, str, tuple[tuple[str, str], ...]]]
) -> list[str]:
    """Return a deterministic parent-before-child topological order."""

    children: dict[str, set[str]] = {qname: set() for qname in records}
    indegree: dict[str, int] = {qname: 0 for qname in records}
    for child, (_, _, references) in records.items():
        parents = {parent for parent, _ in references}
        indegree[child] = len(parents)
        for parent in parents:
            if parent not in records:
                raise ReportError(
                    f"cell {child!r} references unknown parent {parent!r}"
                )
            children[parent].add(child)
    ready = [qname for qname, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    result: list[str] = []
    while ready:
        parent = heapq.heappop(ready)
        result.append(parent)
        for child in sorted(children[parent]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(result) != len(records):
        raise ReportError("cell reference graph contains a cycle")
    return result


def _items(
    root: ET.Element,
    known_categories: Mapping[str, tuple[str, ...]],
    known_cells: Mapping[str, object],
    context: str,
) -> list[tuple[tuple[str, ...], ET.Element]]:
    result: list[tuple[tuple[str, ...], ET.Element]] = []
    container = root.find("items")
    assert container is not None
    for index, item in enumerate(container):
        item_context = f"{context}.items[{index}]"
        if item.tag != "item":
            raise ReportError(f"{item_context}: expected 'item', got {item.tag!r}")
        _require_children(item, ITEM_FIELDS, item_context)
        scalar_values = {
            field: _leaf(item, field, item_context) for field in ITEM_FIELDS[:-1]
        }
        if scalar_values["visited"] not in {"true", "false"}:
            raise ReportError(f"{item_context}: visited must be true or false")
        try:
            multiplicity = int(scalar_values["multiplicity"])
        except ValueError as exc:
            raise ReportError(f"{item_context}: multiplicity must be an integer") from exc
        if multiplicity < 0:
            raise ReportError(f"{item_context}: multiplicity must not be negative")
        category_name = scalar_values["category"]
        path = known_categories.get(category_name)
        if path is None:
            raise ReportError(f"{item_context}: unknown category {category_name!r}")
        cell_name = scalar_values["cell"]
        if cell_name not in known_cells:
            raise ReportError(f"{item_context}: unknown cell {cell_name!r}")
        values = item.find("values")
        assert values is not None
        if any(value.tag != "value" or len(value) for value in values):
            raise ReportError(f"{item_context}: values must contain leaf 'value' elements")
        result.append((path, item))
    return result


def _item_counter(root: ET.Element) -> Counter[tuple[object, ...]]:
    container = root.find("items")
    assert container is not None
    return Counter(_fingerprint(item) for item in container)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    mode = _publication_mode(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _publication_mode(path: Path) -> int:
    """Preserve an existing mode or apply normal create semantics to a new file."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ReportError(f"output is not a regular file path: {path}")
        return stat.S_IMODE(path.stat().st_mode)
    previous_umask = os.umask(0)
    os.umask(previous_umask)
    return 0o666 & ~previous_umask


def _named_roots(
    shards: Sequence[tuple[str, str | os.PathLike[str]]]
) -> list[tuple[str, Path, ET.Element]]:
    if not shards:
        raise ReportError("at least one shard is required")
    names = [name for name, _ in shards]
    if any(not name for name in names):
        raise ReportError("shard names must not be empty")
    if len(set(names)) != len(names):
        raise ReportError("duplicate shard name")
    return [(name, Path(path), _parse_report(path)) for name, path in shards]


def create_manifest(
    reference_report: str | os.PathLike[str],
    shards: Sequence[tuple[str, str | os.PathLike[str]]],
    manifest_path: str | os.PathLike[str],
    deck_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Prove an exact shard union and write its reusable order/owner manifest."""

    reference_path = Path(reference_report)
    reference = _parse_report(reference_path)
    named = _named_roots(shards)
    reference_metadata = _metadata(reference)
    reference_tags = _tags(reference)
    reference_categories = _categories(reference)
    reference_category_map = dict(reference_categories)

    owners: dict[tuple[str, ...], str] = {}
    supplied: dict[tuple[str, ...], str] = {}
    shard_generator: str | None = None
    for name, path, root in named:
        metadata = _metadata(root)
        for field in ("description", "original-file", "top-cell"):
            if metadata[field] != reference_metadata[field]:
                raise ReportError(
                    f"{path}: metadata field {field!r} differs from reference"
                )
        if shard_generator is None:
            shard_generator = metadata["generator"]
        elif metadata["generator"] != shard_generator:
            raise ReportError(f"{path}: shard generators disagree")
        if _tags(root) != reference_tags:
            raise ReportError(f"{path}: tag declarations differ from reference")
        for category_path, description in _categories(root):
            if category_path in owners:
                raise ReportError(
                    f"category {category_path!r} appears in both {owners[category_path]!r} "
                    f"and {name!r}"
                )
            owners[category_path] = name
            supplied[category_path] = description

    if supplied != reference_category_map:
        missing = sorted(set(reference_category_map) - set(supplied))
        extra = sorted(set(supplied) - set(reference_category_map))
        conflicts = sorted(
            path
            for path in set(supplied) & set(reference_category_map)
            if supplied[path] != reference_category_map[path]
        )
        raise ReportError(
            f"category union differs from reference: missing={missing!r}, "
            f"extra={extra!r}, conflicting={conflicts!r}"
        )

    merged_cells = _merge_cells(root for _, _, root in named)
    reference_cells = _cell_records(reference)
    if merged_cells != reference_cells:
        raise ReportError("cell/reference union differs from reference")

    reference_rdb_categories = {
        _category_name(path): path for path, _ in reference_categories
    }
    _items(reference, reference_rdb_categories, reference_cells, str(reference_path))
    for _, path, root in named:
        shard_rdb_categories = {
            _category_name(category_path): category_path
            for category_path, _ in _categories(root)
        }
        _items(root, shard_rdb_categories, merged_cells, str(path))

    shard_items = sum(
        (_item_counter(root) for _, _, root in named),
        Counter(),
    )
    reference_items = _item_counter(reference)
    if shard_items != reference_items:
        raise ReportError(
            f"item union differs from reference: reference={sum(reference_items.values())}, "
            f"shards={sum(shard_items.values())}"
        )

    deck_sha256: str | None = None
    if deck_path is not None:
        try:
            deck_sha256 = hashlib.sha256(Path(deck_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ReportError(f"unable to hash shard deck {deck_path}: {exc}") from exc

    manifest: dict[str, object] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "deck_sha256": deck_sha256,
        "metadata": {
            "description": reference_metadata["description"],
            "generator": reference_metadata["generator"],
        },
        "shards": [name for name, _, _ in named],
        "tags": [
            {"name": name, "description": description}
            for name, description in reference_tags
        ],
        "categories": [
            {
                "path": list(path),
                "description": description,
                "owner": owners[path],
            }
            for path, description in reference_categories
        ],
    }
    destination = Path(manifest_path)
    protected = {reference_path.resolve()}
    protected.update(path.resolve() for _, path, _ in named)
    if deck_path is not None:
        protected.add(Path(deck_path).resolve())
    if destination.resolve() in protected:
        raise ReportError("manifest must not overwrite the reference or a shard report")
    _atomic_json(destination, manifest)
    return manifest


def _load_manifest(path: str | os.PathLike[str]) -> dict[str, object]:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"{manifest_path}: unable to read manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"{manifest_path}: manifest root must be an object")
    expected_keys = {
        "format",
        "version",
        "reference_sha256",
        "deck_sha256",
        "metadata",
        "shards",
        "tags",
        "categories",
    }
    if set(value) != expected_keys:
        raise ReportError(
            f"{manifest_path}: manifest keys differ: expected {sorted(expected_keys)!r}"
        )
    if value["format"] != FORMAT_NAME or value["version"] != FORMAT_VERSION:
        raise ReportError(f"{manifest_path}: unsupported manifest format/version")
    if (
        not isinstance(value["reference_sha256"], str)
        or len(value["reference_sha256"]) != 64
        or any(character not in string.hexdigits for character in value["reference_sha256"])
    ):
        raise ReportError(f"{manifest_path}: invalid reference_sha256")
    deck_sha256 = value["deck_sha256"]
    if deck_sha256 is not None and (
        not isinstance(deck_sha256, str)
        or len(deck_sha256) != 64
        or any(character not in string.hexdigits for character in deck_sha256)
    ):
        raise ReportError(f"{manifest_path}: invalid deck_sha256")
    metadata = value["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {"description", "generator"}:
        raise ReportError(f"{manifest_path}: invalid metadata object")
    if not all(isinstance(item, str) for item in metadata.values()):
        raise ReportError(f"{manifest_path}: metadata values must be strings")
    shard_names = value["shards"]
    if (
        not isinstance(shard_names, list)
        or not shard_names
        or not all(isinstance(name, str) and name for name in shard_names)
        or len(set(shard_names)) != len(shard_names)
    ):
        raise ReportError(f"{manifest_path}: invalid shard list")
    tags = value["tags"]
    if not isinstance(tags, list):
        raise ReportError(f"{manifest_path}: tags must be a list")
    for tag in tags:
        if (
            not isinstance(tag, dict)
            or set(tag) != {"name", "description"}
            or not all(isinstance(field, str) for field in tag.values())
        ):
            raise ReportError(f"{manifest_path}: invalid tag entry")
    categories = value["categories"]
    if not isinstance(categories, list):
        raise ReportError(f"{manifest_path}: categories must be a list")
    seen: set[tuple[str, ...]] = set()
    ordered_paths: list[tuple[str, ...]] = []
    for category in categories:
        if not isinstance(category, dict) or set(category) != {"path", "description", "owner"}:
            raise ReportError(f"{manifest_path}: invalid category entry")
        category_path = category["path"]
        if (
            not isinstance(category_path, list)
            or not category_path
            or not all(isinstance(component, str) for component in category_path)
            or not isinstance(category["description"], str)
            or category["owner"] not in shard_names
        ):
            raise ReportError(f"{manifest_path}: invalid category fields")
        path_tuple = tuple(category_path)
        if path_tuple in seen:
            raise ReportError(f"{manifest_path}: duplicate category {path_tuple!r}")
        if len(path_tuple) > 1 and path_tuple[:-1] not in seen:
            raise ReportError(
                f"{manifest_path}: category parent must precede child {path_tuple!r}"
            )
        seen.add(path_tuple)
        ordered_paths.append(path_tuple)
    children: dict[tuple[str, ...], list[tuple[str, ...]]] = {(): []}
    for path_tuple in ordered_paths:
        children.setdefault(path_tuple[:-1], []).append(path_tuple)
        children.setdefault(path_tuple, [])
    preorder: list[tuple[str, ...]] = []

    def visit(parent: tuple[str, ...]) -> None:
        for child in children[parent]:
            preorder.append(child)
            visit(child)

    visit(())
    if preorder != ordered_paths:
        raise ReportError(f"{manifest_path}: categories are not in depth-first order")
    return value


def validate_manifest(
    manifest_path: str | os.PathLike[str],
    shard_names: Sequence[str] | None = None,
    deck_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Validate a manifest and optionally bind it to a shard set and deck."""

    manifest = _load_manifest(manifest_path)
    if shard_names is not None:
        if len(shard_names) != len(set(shard_names)):
            raise ReportError("duplicate shard name")
        expected = set(manifest["shards"])
        supplied = set(shard_names)
        if supplied != expected:
            raise ReportError(
                f"shard set differs from manifest: missing={sorted(expected - supplied)!r}, "
                f"extra={sorted(supplied - expected)!r}"
            )
    expected_deck_sha256 = manifest["deck_sha256"]
    if expected_deck_sha256 is not None:
        if deck_path is None:
            raise ReportError("manifest is deck-bound but no deck was supplied")
        try:
            actual_deck_sha256 = hashlib.sha256(Path(deck_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ReportError(f"unable to hash shard deck {deck_path}: {exc}") from exc
        if actual_deck_sha256 != expected_deck_sha256:
            raise ReportError(
                f"shard deck SHA-256 differs: {actual_deck_sha256} != {expected_deck_sha256}"
            )
    return manifest


def _append_leaf(parent: ET.Element, name: str, value: str) -> ET.Element:
    child = ET.SubElement(parent, name)
    child.text = value
    return child


def _build_categories(
    parent: ET.Element, categories: Sequence[Mapping[str, object]]
) -> None:
    containers: dict[tuple[str, ...], ET.Element] = {(): parent}
    for entry in categories:
        path = tuple(entry["path"])
        container = containers[path[:-1]]
        category = ET.SubElement(container, "category")
        _append_leaf(category, "name", path[-1])
        _append_leaf(category, "description", str(entry["description"]))
        nested = ET.SubElement(category, "categories")
        containers[path] = nested


def _build_cells(
    parent: ET.Element,
    records: Mapping[str, tuple[str, str, tuple[tuple[str, str], ...]]],
) -> None:
    for qname in _ordered_cells(records):
        name, layout_name, references = records[qname]
        variant = qname[len(name) + 1 :] if qname != name else ""
        cell = ET.SubElement(parent, "cell")
        _append_leaf(cell, "name", name)
        _append_leaf(cell, "variant", variant)
        _append_leaf(cell, "layout-name", layout_name)
        references_node = ET.SubElement(cell, "references")
        for reference_parent, transform in references:
            reference = ET.SubElement(references_node, "ref")
            _append_leaf(reference, "parent", reference_parent)
            _append_leaf(reference, "trans", transform)


def _write_report(root: ET.Element, output_path: Path) -> None:
    if not output_path.parent.is_dir():
        raise ReportError(f"output parent does not exist: {output_path.parent}")
    mode = _publication_mode(output_path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".lyrdb", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        tree = ET.ElementTree(root)
        ET.indent(tree, space=" ")
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        _parse_report(temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def merge_reports(
    manifest_path: str | os.PathLike[str],
    shards: Sequence[tuple[str, str | os.PathLike[str]]],
    output_path: str | os.PathLike[str],
    deck_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Merge shard reports deterministically according to a strict manifest."""

    named = _named_roots(shards)
    manifest = validate_manifest(
        manifest_path, [name for name, _, _ in named], deck_path
    )
    supplied_names = {name for name, _, _ in named}
    expected_names = set(manifest["shards"])
    if supplied_names != expected_names:
        raise ReportError(
            "shard set differs from manifest: "
            f"missing={sorted(expected_names - supplied_names)!r}, "
            f"extra={sorted(supplied_names - expected_names)!r}"
        )

    metadata_by_name = {name: _metadata(root) for name, _, root in named}
    first_metadata = next(iter(metadata_by_name.values()))
    for name, metadata in metadata_by_name.items():
        for field in ("description", "original-file", "top-cell", "generator"):
            if metadata[field] != first_metadata[field]:
                raise ReportError(f"shards disagree about metadata field {field!r} ({name!r})")
    manifest_metadata = manifest["metadata"]
    if first_metadata["description"] != manifest_metadata["description"]:
        raise ReportError("report description differs from manifest")

    expected_tags = tuple(
        (entry["name"], entry["description"]) for entry in manifest["tags"]
    )
    for name, path, root in named:
        if _tags(root) != expected_tags:
            raise ReportError(f"{path}: tag declarations differ from manifest")

    manifest_categories = manifest["categories"]
    category_entries = {
        tuple(entry["path"]): (entry["description"], entry["owner"])
        for entry in manifest_categories
    }
    expected_per_shard: dict[str, dict[tuple[str, ...], str]] = {
        name: {} for name in expected_names
    }
    for path, (description, owner) in category_entries.items():
        expected_per_shard[owner][path] = description
    roots_by_name = {name: (path, root) for name, path, root in named}
    for name, expected in expected_per_shard.items():
        path, root = roots_by_name[name]
        actual = dict(_categories(root))
        if actual != expected:
            raise ReportError(f"{path}: category inventory differs from manifest owner map")

    cells = _merge_cells(root for _, _, root in named)
    rdb_category_names = {
        _category_name(path): path for path in category_entries
    }
    if len(rdb_category_names) != len(category_entries):
        raise ReportError("category path serialization collision")

    items_by_category: dict[tuple[str, ...], list[ET.Element]] = {
        path: [] for path in category_entries
    }
    for name, path, root in named:
        for category_path, item in _items(root, rdb_category_names, cells, str(path)):
            owner = category_entries[category_path][1]
            if owner != name:
                raise ReportError(
                    f"{path}: item for category {category_path!r} belongs to shard {owner!r}"
                )
            items_by_category[category_path].append(copy.deepcopy(item))

    output = ET.Element("report-database")
    _append_leaf(output, "description", first_metadata["description"])
    _append_leaf(output, "original-file", first_metadata["original-file"])
    _append_leaf(output, "generator", manifest_metadata["generator"])
    _append_leaf(output, "top-cell", first_metadata["top-cell"])
    tags_node = ET.SubElement(output, "tags")
    for name, description in expected_tags:
        tag = ET.SubElement(tags_node, "tag")
        _append_leaf(tag, "name", name)
        _append_leaf(tag, "description", description)
    categories_node = ET.SubElement(output, "categories")
    _build_categories(categories_node, manifest_categories)
    cells_node = ET.SubElement(output, "cells")
    _build_cells(cells_node, cells)
    items_node = ET.SubElement(output, "items")
    item_count = 0
    for entry in manifest_categories:
        path = tuple(entry["path"])
        for item in items_by_category[path]:
            items_node.append(item)
            item_count += 1

    destination = Path(output_path)
    protected = {Path(path).resolve() for _, path, _ in named}
    protected.add(Path(manifest_path).resolve())
    if destination.resolve() in protected:
        raise ReportError("output must not overwrite a shard report or manifest")
    _write_report(output, destination)
    return {
        "output": str(destination),
        "shards": len(named),
        "categories": len(category_entries),
        "cells": len(cells),
        "items": item_count,
    }


def _parse_shard(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=REPORT, got {value!r}")
    return name, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest", help="prove a shard union against a trusted full report"
    )
    manifest.add_argument("--schema", required=True, help="trusted full .lyrdb report")
    manifest.add_argument(
        "--deck", help="optional shard-aware deck to bind by SHA-256"
    )
    manifest.add_argument(
        "--shard", required=True, action="append", type=_parse_shard, metavar="NAME=REPORT"
    )
    manifest.add_argument("-o", "--output", required=True, help="output manifest JSON")

    merge = subparsers.add_parser("merge", help="merge reports using a proven manifest")
    merge.add_argument("--manifest", required=True, help="manifest JSON")
    merge.add_argument("--deck", help="deck required by a deck-bound manifest")
    merge.add_argument(
        "--shard", required=True, action="append", type=_parse_shard, metavar="NAME=REPORT"
    )
    merge.add_argument("-o", "--output", required=True, help="output merged .lyrdb")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "manifest":
            manifest = create_manifest(
                args.schema, args.shard, args.output, deck_path=args.deck
            )
            print(
                f"manifest: {args.output} "
                f"({len(manifest['shards'])} shards, {len(manifest['categories'])} categories)"
            )
        else:
            summary = merge_reports(
                args.manifest, args.shard, args.output, deck_path=args.deck
            )
            print(
                f"merged: {summary['output']} "
                f"({summary['categories']} categories, {summary['cells']} cells, "
                f"{summary['items']} items)"
            )
        return 0
    except (OSError, ReportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
