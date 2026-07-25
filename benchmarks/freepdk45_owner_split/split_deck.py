#!/usr/bin/env python3
"""Split two intact FreePDK45 rule groups into independent CPU owners.

The input must already be a shard-aware FreePDK45 deck.  The transform can:

* retain IMPLANT.1-.5 in ``implant_contact`` while moving CONTACT.1-.5 into
  the independent ``contact`` owner; and
* move ACTIVE.1/.2 from ``via1_upper_active12`` into ``active12``.

No rule expression is rewritten.  In ``drc_shard=all`` mode both new owner
predicates are true, so the historical textual execution and output order are
preserved.  Every source edit is match-counted and the generated macro is
parsed as XML before publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


IMPLANT_CONTACT_SHARD = "implant_contact"
CONTACT_SHARD = "contact"
VIA1_UPPER_ACTIVE12_SHARD = "via1_upper_active12"
ACTIVE12_SHARD = "active12"


class TransformError(ValueError):
    """The source deck is not the expected unsplit owner variant."""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise TransformError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def _require_output_counts(
    text: str, expected: dict[str, tuple[int, ...]]
) -> None:
    for category, allowed in expected.items():
        count = text.count(f'.output("{category}"')
        if count not in allowed:
            choices = " or ".join(str(value) for value in allowed)
            raise TransformError(
                f"{category}: expected {choices} source sites, found {count}"
            )


def split_deck(
    source: str,
    *,
    split_implant_contact: bool = False,
    split_active12: bool = False,
) -> str:
    """Return a deterministic LF-normalized owner-split deck."""

    if not split_implant_contact and not split_active12:
        raise TransformError("at least one owner split must be selected")

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    if (
        f'drc_shard == "{CONTACT_SHARD}"' in text
        or f'drc_shard == "{ACTIVE12_SHARD}"' in text
    ):
        raise TransformError("source deck is already owner-split")

    if split_implant_contact:
        _require_output_counts(
            text,
            {
                "IMPLANT.1": (1, 2),
                "IMPLANT.2": (1, 2),
                "IMPLANT.3": (1,),
                "IMPLANT.4": (1,),
                "IMPLANT.5": (1,),
                "CONTACT.1": (1, 2),
                "CONTACT.2": (1, 2),
                "CONTACT.3": (1, 2),
                "CONTACT.4": (1,),
                "CONTACT.5": (1,),
            },
        )
        old_declaration = (
            'run_implant_contact = drc_shard == "all" || '
            f'drc_shard == "{IMPLANT_CONTACT_SHARD}"\n'
        )
        new_declaration = old_declaration + (
            'run_contact = drc_shard == "all" || '
            f'drc_shard == "{CONTACT_SHARD}"\n'
        )
        text = _replace_once(
            text,
            old_declaration,
            new_declaration,
            "implant/contact owner declarations",
        )
        text = _replace_once(
            text,
            "run_m2_rules || run_implant_contact || "
            "run_via1_upper_active12",
            "run_m2_rules || run_implant_contact || run_contact || "
            "run_via1_upper_active12",
            "implant/contact valid-shard guard",
        )
        text = _replace_once(
            text,
            "(run_implant_contact || run_m1_enclosure)",
            "(run_contact || run_m1_enclosure)",
            "M1-contact certificate owner",
        )
        text = _replace_once(
            text,
            "implant.forget\n\n#   Contact",
            "implant.forget\n\nend\n\n\nif run_contact\n\n#   Contact",
            "implant/contact block boundary",
        )

    if split_active12:
        _require_output_counts(
            text, {"ACTIVE.1": (1,), "ACTIVE.2": (1,)}
        )
        declaration = (
            'run_via1_upper_active12 = drc_shard == "all" || '
            f'drc_shard == "{VIA1_UPPER_ACTIVE12_SHARD}"\n'
        )
        text = _replace_once(
            text,
            declaration,
            declaration
            + 'run_active12_rules = drc_shard == "all" || '
            f'drc_shard == "{ACTIVE12_SHARD}"\n',
            "ACTIVE.1/.2 owner declaration",
        )
        text = _replace_once(
            text,
            "run_via1_upper_active12 || run_grid",
            "run_via1_upper_active12 || run_active12_rules || run_grid",
            "ACTIVE.1/.2 valid-shard guard",
        )
        text = _replace_once(
            text,
            "run_active12 = run_via1_upper_active12\n",
            "run_active12 = run_active12_rules\n",
            "ACTIVE.1/.2 owner",
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
        "--split-implant-contact",
        action="store_true",
        help="create independent implant and contact owners",
    )
    parser.add_argument(
        "--split-active12",
        action="store_true",
        help="create an independent ACTIVE.1/.2 owner",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output"
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
        transformed = split_deck(
            source,
            split_implant_contact=args.split_implant_contact,
            split_active12=args.split_active12,
        )
        _write_atomic(args.output, transformed)
    except (OSError, UnicodeDecodeError, TransformError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
