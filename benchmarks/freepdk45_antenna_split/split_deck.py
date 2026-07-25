#!/usr/bin/env python3
"""Split the FreePDK45 antenna owner into exact CPU process shards.

The input must be one of the shard-aware FreePDK45 decks derived from the
qualified eight-owner deck.  The transform is deliberately fail-closed: every
expected owner predicate and antenna-check site must occur exactly once.

The resulting owners are:

* ``antenna_feol``: WELL.1, WELL.4, VT.1 and ACTIVE.4
* ``antenna_m1_m2``: METAL1_ANTENNA and METAL2_ANTENNA
* ``antenna_m3_m10``: METAL3_ANTENNA through METAL10_ANTENNA

The optional ``--split-upper`` mode replaces the last owner with two:

* ``antenna_m3``: METAL3_ANTENNA
* ``antenna_m4_m10``: METAL4_ANTENNA through METAL10_ANTENNA

Each metal owner rebuilds the exact cumulative connection prefix it needs.
Lower-metal antenna checks are not executed while constructing the M3 prefix.
In ``drc_shard=all`` mode, the historical check/connect order is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


FEOL_SHARD = "antenna_feol"
LOWER_SHARD = "antenna_m1_m2"
UPPER_SHARD = "antenna_m3_m10"
M3_SHARD = "antenna_m3"
M4_UPPER_SHARD = "antenna_m4_m10"
ANTENNA_CATEGORIES = tuple(f"METAL{layer}_ANTENNA" for layer in range(1, 11))


class TransformError(ValueError):
    """The source deck is not the expected unsplit FreePDK45 variant."""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise TransformError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def _antenna_section(split_upper: bool = False) -> str:
    lines = [
        "#   ANTENNA checks",
        "################",
        "if ANTENNA &amp;&amp; run_antenna_checks",
        'info("ANTENNA section")',
        "",
        "diode = nplus &amp; active - nwell # diode recognition layer",
        "",
        "# Every checking owner needs the exact gate-to-M1 connection prefix.",
        "connect(gate, poly)",
        "connect(poly, cont)",
        "connect(diode, cont)",
        "connect(cont, metal1)",
        "",
        "if run_antenna_m1_m2",
        'antenna_check(gate, metal1, 300.0, diode).output("METAL1_ANTENNA", "METAL1_ANTENNA : Ratio of Maximum Allowed (Field poly area or Metal Layer Area) to transistor gate area : 300:1")',
        "end",
        "",
        "# Both checking owners need the connection prefix through metal2.",
        "connect(metal1, via1)",
        "connect(via1, metal2)",
        "",
        "if run_antenna_m1_m2",
        'antenna_check(gate, metal2, 300.0, diode).output("METAL2_ANTENNA", "METAL2_ANTENNA : Ratio of Maximum Allowed (Field poly area or Metal Layer Area) to transistor gate area : 300:1")',
        "end",
        "",
    ]
    if split_upper:
        lines.extend(
            [
                "if run_antenna_m3 || run_antenna_m4_m10",
                "# build connection of poly+gate to metal3",
                "connect(metal2, via2)",
                "connect(via2, metal3)",
                "",
                "if run_antenna_m3",
                'antenna_check(gate, metal3, 300.0, diode).output("METAL3_ANTENNA", "METAL3_ANTENNA : Ratio of Maximum Allowed (Field poly area or Metal Layer Area) to transistor gate area : 300:1")',
                "",
                "end",
                "",
                "end",
                "",
                "if run_antenna_m4_m10",
            ]
        )
        first_upper = 4
    else:
        lines.append("if run_antenna_m3_m10")
        first_upper = 3

    for layer in range(first_upper, 11):
        lower = layer - 1
        lines.extend(
            [
                f"# build connection of poly+gate to metal{layer}",
                f"connect(metal{lower}, via{lower})",
                f"connect(via{lower}, metal{layer})",
                "",
                f'antenna_check(gate, metal{layer}, 300.0, diode).output("METAL{layer}_ANTENNA", "METAL{layer}_ANTENNA : Ratio of Maximum Allowed (Field poly area or Metal Layer Area) to transistor gate area : 300:1")',
                "",
            ]
        )
    lines.extend(["end", "", "end", ""])
    return "\n".join(lines)


def split_deck(source: str, *, split_upper: bool = False) -> str:
    """Return a deterministic LF-normalized antenna-sharded deck."""

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    if any(
        name in text
        for name in (
            FEOL_SHARD,
            LOWER_SHARD,
            UPPER_SHARD,
            M3_SHARD,
            M4_UPPER_SHARD,
        )
    ):
        raise TransformError("source deck is already antenna-split")

    for category in ANTENNA_CATEGORIES:
        count = text.count(f'.output("{category}"')
        if count != 1:
            raise TransformError(
                f"{category}: expected one antenna output site, found {count}"
            )

    if split_upper:
        owner_declaration = (
            'run_antenna_feol = drc_shard == "all" || drc_shard == "antenna_feol"\n'
            'run_antenna_m1_m2 = drc_shard == "all" || drc_shard == "antenna_m1_m2"\n'
            'run_antenna_m3 = drc_shard == "all" || drc_shard == "antenna_m3"\n'
            'run_antenna_m4_m10 = drc_shard == "all" || drc_shard == "antenna_m4_m10"\n'
            "run_antenna_checks = run_antenna_m1_m2 || run_antenna_m3 || "
            "run_antenna_m4_m10\n"
        )
    else:
        owner_declaration = (
            'run_antenna_feol = drc_shard == "all" || drc_shard == "antenna_feol"\n'
            'run_antenna_m1_m2 = drc_shard == "all" || drc_shard == "antenna_m1_m2"\n'
            'run_antenna_m3_m10 = drc_shard == "all" || drc_shard == "antenna_m3_m10"\n'
            "run_antenna_checks = run_antenna_m1_m2 || run_antenna_m3_m10\n"
        )
    text = _replace_once(
        text,
        'run_antenna = drc_shard == "all" || drc_shard == "antenna"\n',
        owner_declaration,
        "antenna owner declaration",
    )
    text = _replace_once(
        text,
        " || run_grid || run_antenna\n",
        " || run_grid || run_antenna_feol || run_antenna_checks\n",
        "valid-shard guard",
    )
    text = _replace_once(
        text,
        "run_well = run_antenna\n",
        "run_well = run_antenna_feol\n",
        "WELL owner",
    )
    text = _replace_once(
        text,
        "run_active4 = run_antenna\n",
        "run_active4 = run_antenna_feol\n",
        "ACTIVE.4 owner",
    )
    text = _replace_once(
        text,
        "(ANTENNA &amp;&amp; run_antenna)\n",
        "(ANTENNA &amp;&amp; run_antenna_checks)\n",
        "gate dependency",
    )

    start_marker = "#   ANTENNA checks\n"
    end_marker = "# time spent for the DRC\n"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise TransformError("antenna section boundaries are not unique")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    text = text[:start] + _antenna_section(split_upper) + text[end:]

    # Parse the macro as XML after transformation.  This catches missed entity
    # escaping before an expensive KLayout run.
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
    parser.add_argument("input", type=Path, help="unsplit shard-aware .lydrc")
    parser.add_argument("output", type=Path, help="generated split .lydrc")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output"
    )
    parser.add_argument(
        "--split-upper",
        action="store_true",
        help="split METAL3 and METAL4-through-METAL10 into separate owners",
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
        transformed = split_deck(source, split_upper=args.split_upper)
        _write_atomic(args.output, transformed)
    except (OSError, UnicodeDecodeError, TransformError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
