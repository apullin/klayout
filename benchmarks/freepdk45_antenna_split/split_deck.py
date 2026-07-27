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

The optional ``--split-lower`` mode replaces the lower owner with two:

* ``antenna_m1``: METAL1_ANTENNA
* ``antenna_m2``: METAL2_ANTENNA

The mutually exclusive ``--fuse-metal`` mode assigns all ten metal checks to
one ``antenna_m1_m4`` owner.  METAL5 through METAL10 remain in that exact CPU
owner even when they are empty on the current workload; the name describes the
last non-empty layer rather than weakening the output contract.

The optional ``--small-first-diode`` mode reassociates the exact diode set
expression from ``nplus & (active - nwell)`` to
``(nplus & active) - nwell`` so the smaller intersection is formed first.

The two modes compose. An M1-only process does not build the M1-via1-M2
connection; every M2-or-higher process rebuilds that exact cumulative prefix.

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
M1_SHARD = "antenna_m1"
M2_SHARD = "antenna_m2"
M3_SHARD = "antenna_m3"
M4_UPPER_SHARD = "antenna_m4_m10"
FUSED_METAL_SHARD = "antenna_m1_m4"
ANTENNA_CATEGORIES = tuple(f"METAL{layer}_ANTENNA" for layer in range(1, 11))
HISTORICAL_DIODE_LINE = (
    "diode = nplus &amp; active - nwell # diode recognition layer"
)
SMALL_FIRST_DIODE_LINE = (
    "diode = (nplus &amp; active) - nwell # diode recognition layer"
)


class TransformError(ValueError):
    """The source deck is not the expected unsplit FreePDK45 variant."""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise TransformError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def metal_owner_manifest(
    *,
    split_lower: bool = False,
    split_upper: bool = False,
    fuse_metal: bool = False,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the deterministic metal-output ownership contract."""

    if fuse_metal:
        if split_lower or split_upper:
            raise TransformError(
                "fused metal owner cannot be combined with lower/upper splits"
            )
        return ((FUSED_METAL_SHARD, ANTENNA_CATEGORIES),)

    lower = (
        (
            (M1_SHARD, (ANTENNA_CATEGORIES[0],)),
            (M2_SHARD, (ANTENNA_CATEGORIES[1],)),
        )
        if split_lower
        else ((LOWER_SHARD, ANTENNA_CATEGORIES[:2]),)
    )
    upper = (
        (
            (M3_SHARD, (ANTENNA_CATEGORIES[2],)),
            (M4_UPPER_SHARD, ANTENNA_CATEGORIES[3:]),
        )
        if split_upper
        else ((UPPER_SHARD, ANTENNA_CATEGORIES[2:]),)
    )
    return lower + upper


def antenna_shards(
    *,
    split_lower: bool = False,
    split_upper: bool = False,
    fuse_metal: bool = False,
) -> tuple[str, ...]:
    """Return the deterministic antenna shard list used by launch harnesses."""

    return (FEOL_SHARD,) + tuple(
        owner
        for owner, _categories in metal_owner_manifest(
            split_lower=split_lower,
            split_upper=split_upper,
            fuse_metal=fuse_metal,
        )
    )


def _owner_predicate(shard: str) -> str:
    return f"run_{shard}"


def _metal_antenna_output(layer: int, receiver: str) -> str:
    category = f"METAL{layer}_ANTENNA"
    description = (
        f"{category} : Ratio of Maximum Allowed "
        "(Field poly area or Metal Layer Area) to transistor gate area : 300:1"
    )
    return f'{receiver}.output("{category}", "{description}")'


def _fused_certificate_prelude() -> str:
    upper_empty = " &amp;&amp; ".join(
        f"metal{layer}.is_empty?" for layer in range(5, 11)
    )
    return "\n".join(
        [
            "# BEGIN KLAYOUT CUDA ANTENNA M1-M4 RAW TRANSACTION",
            "# The certificate is atomic: raw M5-M10 must first be exactly empty,",
            "# then one qualified M1-M4 call must certify every staged metal check.",
            "# Missing capability, a false result, or any exception keeps the flag",
            "# false and executes the complete literal CPU chain below.",
            "antenna_m1_m4_clean = false",
            "antenna_m1_m4_empty = nil",
            'antenna_m1_m4_reason = "not-owner"',
            "if ANTENNA &amp;&amp; run_antenna_m1_m4",
            '  antenna_m1_m4_reason = "upper-metal-nonempty"',
            "  begin",
            f"    antenna_m1_m4_upper_empty = {upper_empty}",
            "    if antenna_m1_m4_upper_empty",
            '      antenna_m1_m4_reason = "method-unavailable"',
            "      if poly.respond_to?(:cuda_antenna_m1_m4_raw_clean?)",
            "        antenna_m1_m4_certified = "
            "poly.cuda_antenna_m1_m4_raw_clean?("
            "active, nplus, nwell, cont, metal1, via1, metal2, via2, "
            "metal3, via3, metal4)",
            "        if antenna_m1_m4_certified",
            "          antenna_m1_m4_empty = polygon_layer",
            "          antenna_m1_m4_clean = true",
            '          antenna_m1_m4_reason = "certified-empty"',
            "        else",
            '          antenna_m1_m4_reason = "certificate-declined"',
            "        end",
            "      end",
            "    end",
            "  rescue StandardError =&gt; antenna_m1_m4_error",
            "    antenna_m1_m4_clean = false",
            "    antenna_m1_m4_empty = nil",
            '    antenna_m1_m4_reason = "exception:#{antenna_m1_m4_error.class}"',
            "  end",
            "end",
            'info("CUDA ANTENNA M1-M4 raw transaction: '
            '#{antenna_m1_m4_clean ? \'certified-empty\' : '
            "'full-cpu-fallback'} reason=#{antenna_m1_m4_reason}\") "
            "if ANTENNA &amp;&amp; run_antenna_m1_m4",
            "# END KLAYOUT CUDA ANTENNA M1-M4 RAW TRANSACTION",
            "",
        ]
    )


def _fused_antenna_section(small_first_diode: bool) -> str:
    lines = [
        "#   ANTENNA checks",
        "################",
        "if ANTENNA &amp;&amp; run_antenna_checks",
        'info("ANTENNA section")',
        "",
        "if antenna_m1_m4_clean",
    ]
    lines.extend(
        _metal_antenna_output(layer, "antenna_m1_m4_empty")
        for layer in range(1, 11)
    )
    lines.extend(
        [
            "else",
            (
                SMALL_FIRST_DIODE_LINE
                if small_first_diode
                else HISTORICAL_DIODE_LINE
            ),
            "",
            "# Exact staged literal fallback; keep this complete and in source order.",
            "connect(gate, poly)",
            "connect(poly, cont)",
            "connect(diode, cont)",
            "connect(cont, metal1)",
            "",
            _metal_antenna_output(
                1, "antenna_check(gate, metal1, 300.0, diode)"
            ),
            "",
        ]
    )
    for layer in range(2, 11):
        lower = layer - 1
        lines.extend(
            [
                f"# build connection of poly+gate to metal{layer}",
                f"connect(metal{lower}, via{lower})",
                f"connect(via{lower}, metal{layer})",
                "",
                _metal_antenna_output(
                    layer,
                    f"antenna_check(gate, metal{layer}, 300.0, diode)",
                ),
                "",
            ]
        )
    lines.extend(["end", "", "end", ""])
    return "\n".join(lines)


def _antenna_section(
    split_upper: bool = False,
    split_lower: bool = False,
    small_first_diode: bool = False,
    fuse_metal: bool = False,
) -> str:
    if fuse_metal:
        if split_lower or split_upper:
            raise TransformError(
                "fused metal owner cannot be combined with lower/upper splits"
            )
        return _fused_antenna_section(small_first_diode)

    lower_m1_predicate = _owner_predicate(
        M1_SHARD if split_lower else LOWER_SHARD
    )
    lower_m2_predicate = _owner_predicate(
        M2_SHARD if split_lower else LOWER_SHARD
    )
    upper_predicates = [
        _owner_predicate(owner)
        for owner, _categories in metal_owner_manifest(
            split_lower=split_lower,
            split_upper=split_upper,
        )
        if owner not in (M1_SHARD, M2_SHARD, LOWER_SHARD)
    ]
    lines = [
        "#   ANTENNA checks",
        "################",
        "if ANTENNA &amp;&amp; run_antenna_checks",
        'info("ANTENNA section")',
        "",
        (
            SMALL_FIRST_DIODE_LINE
            if small_first_diode
            else HISTORICAL_DIODE_LINE
        ),
        "",
        "# Every checking owner needs the exact gate-to-M1 connection prefix.",
        "connect(gate, poly)",
        "connect(poly, cont)",
        "connect(diode, cont)",
        "connect(cont, metal1)",
        "",
        f"if {lower_m1_predicate}",
        'antenna_check(gate, metal1, 300.0, diode).output("METAL1_ANTENNA", "METAL1_ANTENNA : Ratio of Maximum Allowed (Field poly area or Metal Layer Area) to transistor gate area : 300:1")',
        "end",
        "",
    ]
    if split_lower:
        lines.extend(
            [
                "# Only M2-or-higher owners need the connection prefix through metal2.",
                f"if {' || '.join([lower_m2_predicate] + upper_predicates)}",
                "connect(metal1, via1)",
                "connect(via1, metal2)",
                "",
                f"if {lower_m2_predicate}",
                'antenna_check(gate, metal2, 300.0, diode).output("METAL2_ANTENNA", "METAL2_ANTENNA : Ratio of Maximum Allowed (Field poly area or Metal Layer Area) to transistor gate area : 300:1")',
                "end",
                "",
                "end",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "# Both checking owners need the connection prefix through metal2.",
                "connect(metal1, via1)",
                "connect(via1, metal2)",
                "",
                f"if {lower_m2_predicate}",
                'antenna_check(gate, metal2, 300.0, diode).output("METAL2_ANTENNA", "METAL2_ANTENNA : Ratio of Maximum Allowed (Field poly area or Metal Layer Area) to transistor gate area : 300:1")',
                "end",
                "",
            ]
        )

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
        if len(upper_predicates) != 1:
            raise TransformError(
                "unsplit upper antenna section must have exactly one owner"
            )
        lines.append(f"if {upper_predicates[0]}")
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


def split_deck(
    source: str,
    *,
    split_lower: bool = False,
    split_upper: bool = False,
    small_first_diode: bool = False,
    fuse_metal: bool = False,
) -> str:
    """Return a deterministic LF-normalized antenna-sharded deck."""

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    if any(
        name in text
        for name in (
            FEOL_SHARD,
            LOWER_SHARD,
            UPPER_SHARD,
            M1_SHARD,
            M2_SHARD,
            M3_SHARD,
            M4_UPPER_SHARD,
            FUSED_METAL_SHARD,
        )
    ):
        raise TransformError("source deck is already antenna-split")

    diode_line_count = text.splitlines().count(HISTORICAL_DIODE_LINE)
    if diode_line_count != 1:
        raise TransformError(
            "historical diode source: expected exactly one line, "
            f"found {diode_line_count}"
        )

    for category in ANTENNA_CATEGORIES:
        count = text.count(f'.output("{category}"')
        if count != 1:
            raise TransformError(
                f"{category}: expected one antenna output site, found {count}"
            )

    metal_owners = metal_owner_manifest(
        split_lower=split_lower,
        split_upper=split_upper,
        fuse_metal=fuse_metal,
    )
    owner_declaration = (
        'run_antenna_feol = drc_shard == "all" || '
        'drc_shard == "antenna_feol"\n'
        + "".join(
            f'{_owner_predicate(owner)} = drc_shard == "all" || '
            f'drc_shard == "{owner}"\n'
            for owner, _categories in metal_owners
        )
        + "run_antenna_checks = "
        + " || ".join(_owner_predicate(owner) for owner, _ in metal_owners)
        + "\n"
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
    gate_dependency = "(ANTENNA &amp;&amp; run_antenna)\n"
    raw_poly34_owner_count = text.count("poly34_raw_owner =")
    if raw_poly34_owner_count > 1:
        raise TransformError(
            "raw POLY.3/.4 owner: expected at most one marker, "
            f"found {raw_poly34_owner_count}"
        )
    expected_gate_dependencies = 2 if raw_poly34_owner_count else 1
    gate_dependency_count = text.count(gate_dependency)
    if gate_dependency_count != expected_gate_dependencies:
        raise TransformError(
            "gate dependency: expected "
            f"{expected_gate_dependencies} source match"
            f"{'es' if expected_gate_dependencies != 1 else ''}, "
            f"found {gate_dependency_count}"
        )
    plain_gate_dependency = "(ANTENNA &amp;&amp; run_antenna_checks)\n"
    if fuse_metal:
        need_gate_matches = [
            line
            for line in text.splitlines(keepends=True)
            if line.startswith("need_gate = ")
            and gate_dependency.rstrip("\n") in line
        ]
        if len(need_gate_matches) != 1:
            raise TransformError(
                "fused antenna need_gate dependency: expected one source "
                f"match, found {len(need_gate_matches)}"
            )
        fused_gate_dependency = (
            "(ANTENNA &amp;&amp; run_antenna_checks "
            "&amp;&amp; !antenna_m1_m4_clean)\n"
        )
        fused_need_gate = need_gate_matches[0].replace(
            gate_dependency, fused_gate_dependency, 1
        )
        text = _replace_once(
            text,
            need_gate_matches[0],
            _fused_certificate_prelude() + fused_need_gate,
            "fused antenna certificate insertion",
        )
        remaining_dependencies = expected_gate_dependencies - 1
        if text.count(gate_dependency) != remaining_dependencies:
            raise TransformError(
                "fused antenna non-gate dependency: expected "
                f"{remaining_dependencies} source match"
                f"{'es' if remaining_dependencies != 1 else ''}, "
                f"found {text.count(gate_dependency)}"
            )
        text = text.replace(
            gate_dependency,
            plain_gate_dependency,
            remaining_dependencies,
        )
    else:
        text = text.replace(
            gate_dependency,
            plain_gate_dependency,
            expected_gate_dependencies,
        )

    start_marker = "#   ANTENNA checks\n"
    end_marker = "# time spent for the DRC\n"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise TransformError("antenna section boundaries are not unique")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    text = (
        text[:start]
        + _antenna_section(
            split_upper=split_upper,
            split_lower=split_lower,
            small_first_diode=small_first_diode,
            fuse_metal=fuse_metal,
        )
        + text[end:]
    )

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
        "--split-lower",
        action="store_true",
        help="split METAL1 and METAL2 into separate owners",
    )
    parser.add_argument(
        "--split-upper",
        action="store_true",
        help="split METAL3 and METAL4-through-METAL10 into separate owners",
    )
    parser.add_argument(
        "--small-first-diode",
        action="store_true",
        help="form the exact diode recognition layer from its smaller operand first",
    )
    parser.add_argument(
        "--fuse-metal",
        action="store_true",
        help="assign METAL1-through-METAL10 antenna checks to one staged owner",
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
            split_lower=args.split_lower,
            split_upper=args.split_upper,
            small_first_diode=args.small_first_diode,
            fuse_metal=args.fuse_metal,
        )
        _write_atomic(args.output, transformed)
    except (OSError, UnicodeDecodeError, TransformError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
