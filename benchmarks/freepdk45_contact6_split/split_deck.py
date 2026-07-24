#!/usr/bin/env python3
"""Move FreePDK45 CONTACT.6 into an independent CPU process shard.

The input must be a shard-aware FreePDK45 deck with CONTACT.6 still owned by
``m1_width_space``. The transform is deliberately fail-closed and composes
with the separate three-way antenna transform.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


CONTACT_SHARD = "m1_contact6"
GRID_OWNER = "grid"
OWNER_CHOICES = ("independent", GRID_OWNER)
OWNED_CATEGORIES = ("CONTACT.6",)
RETAINED_CATEGORIES = ("METAL1.1", "METAL1.2")


class TransformError(ValueError):
    """The source deck is not the expected unsplit CONTACT.6 variant."""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise TransformError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def split_deck(source: str, owner: str = "independent") -> str:
    """Return a deterministic LF-normalized CONTACT.6-split deck."""

    if owner not in OWNER_CHOICES:
        raise TransformError(f"unsupported CONTACT.6 owner {owner!r}")
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    if (
        f'drc_shard == "{CONTACT_SHARD}"' in text
        or "run_contact6 = run_m1_contact6" in text
        or "run_contact6 = run_grid" in text
    ):
        raise TransformError("source deck is already CONTACT.6-split")

    for category in OWNED_CATEGORIES + RETAINED_CATEGORIES:
        count = text.count(f'.output("{category}"')
        if count != 1:
            raise TransformError(
                f"{category}: expected one output site, found {count}"
            )

    owner_predicate = "run_grid"
    if owner == "independent":
        declaration = (
            'run_m1_width_space = drc_shard == "all" || '
            'drc_shard == "m1_width_space"\n'
        )
        text = _replace_once(
            text,
            declaration,
            declaration
            + 'run_m1_contact6 = drc_shard == "all" || '
            'drc_shard == "m1_contact6"\n',
            "CONTACT.6 owner declaration",
        )
        text = _replace_once(
            text,
            "run_m1_enclosure || run_m1_width_space || run_m1_via_class",
            "run_m1_enclosure || run_m1_width_space || run_m1_contact6 || "
            "run_m1_via_class",
            "valid-shard guard",
        )
        owner_predicate = "run_m1_contact6"
    elif text.count('drc_shard == "grid"') != 1:
        raise TransformError("grid owner: expected one source declaration")

    text = _replace_once(
        text,
        "run_contact6 = run_m1_width_space\n",
        f"run_contact6 = {owner_predicate}\n",
        "CONTACT.6 owner",
    )

    try:
        ET.fromstring(text)
    except ET.ParseError as exc:
        raise TransformError(f"generated deck is not valid XML: {exc}") from exc
    return text


def _write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        handle.flush()
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="shard-aware input .lydrc")
    parser.add_argument("output", type=Path, help="generated split .lydrc")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output"
    )
    parser.add_argument(
        "--owner",
        choices=OWNER_CHOICES,
        default="independent",
        help="create an independent owner or coalesce CONTACT.6 into grid",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input and output must differ")
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    try:
        source = args.input.read_bytes().decode("utf-8")
        transformed = split_deck(source, owner=args.owner)
        _write_atomic(args.output, transformed)
    except (OSError, UnicodeDecodeError, TransformError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
