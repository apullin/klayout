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

WELL_NEED_LINE = (
    "need_well = (DRC &amp;&amp; "
    "(run_well || run_active3 || run_active4)) || "
    "(OFFGRID &amp;&amp; run_grid)\n"
)
GRID_ASSIGNMENT_LINE = "grid = 2.5.nm\n"
WELL_UNION_LITERAL = "well = nwell.or(pwell) if need_well"

GRID_WELL_CERTIFICATE = """\
# BEGIN KLAYOUT RAW WELL GRID CERTIFICATE
# In the grid-only owner, a two-input raw proof may establish that merging
# NWELL and PWELL cannot create an off-grid vertex.  Every decline, missing
# method, or exception retains the literal WELL union and GRID check below.
grid_well_raw_grid = 2.5.nm
grid_well_raw_clean = false
grid_well_raw_reason = "not-owner"
grid_well_raw_owner = OFFGRID &amp;&amp; run_grid &amp;&amp; !run_well &amp;&amp; !run_active3 &amp;&amp; !run_active4
if grid_well_raw_owner
  grid_well_raw_reason = "method-unavailable"
  begin
    if nwell.respond_to?(:raw_union_grid_clean?)
      grid_well_raw_clean = nwell.raw_union_grid_clean?(pwell, grid_well_raw_grid)
      grid_well_raw_reason = grid_well_raw_clean ? "certified-empty" : "certificate-declined"
    end
  rescue StandardError =&gt; grid_well_raw_error
    grid_well_raw_clean = false
    grid_well_raw_reason = "exception:#{grid_well_raw_error.class}"
  end
end
info("RAW WELL grid transaction: #{grid_well_raw_clean ? 'certified-empty' : 'full-cpu-fallback'} reason=#{grid_well_raw_reason}") if grid_well_raw_owner
need_well = ((DRC &amp;&amp; (run_well || run_active3 || run_active4)) || (OFFGRID &amp;&amp; run_grid)) &amp;&amp; !grid_well_raw_clean
well = polygon_layer if grid_well_raw_clean
# END KLAYOUT RAW WELL GRID CERTIFICATE
"""


class TransformError(ValueError):
    """The source deck is not the expected unsplit CONTACT.6 variant."""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise TransformError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def _literal_line_offsets(text: str, literal: str) -> list[int]:
    """Return offsets of lines containing only ``literal`` plus whitespace."""

    offsets: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.strip() == literal:
            offsets.append(offset)
        offset += len(line)
    return offsets


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

    if owner == GRID_OWNER:
        well_union_offsets = _literal_line_offsets(text, WELL_UNION_LITERAL)
        if len(well_union_offsets) != 1:
            raise TransformError(
                "raw WELL grid fallback: expected one literal "
                f"WELL union, found {len(well_union_offsets)}"
            )
        well_need_count = text.count(WELL_NEED_LINE)
        if well_need_count != 1:
            raise TransformError(
                "raw WELL grid certificate: expected one source match, "
                f"found {well_need_count}"
            )
        grid_assignment_count = text.count(GRID_ASSIGNMENT_LINE)
        if grid_assignment_count != 1:
            raise TransformError(
                "raw WELL grid value: expected one source match, "
                f"found {grid_assignment_count}"
            )
        if not (
            text.index(WELL_NEED_LINE)
            < well_union_offsets[0]
            < text.index(GRID_ASSIGNMENT_LINE)
        ):
            raise TransformError(
                "raw WELL grid fallback: literal WELL union must follow "
                "need_well and precede the GRID section"
            )

        text = _replace_once(
            text,
            GRID_ASSIGNMENT_LINE,
            "grid = grid_well_raw_grid\n",
            "raw WELL grid value",
        )
        text = _replace_once(
            text,
            WELL_NEED_LINE,
            GRID_WELL_CERTIFICATE,
            "raw WELL grid certificate",
        )

        transformed_union_offsets = _literal_line_offsets(
            text, WELL_UNION_LITERAL
        )
        certificate_end = text.index(
            "# END KLAYOUT RAW WELL GRID CERTIFICATE\n"
        )
        transformed_grid = text.index("grid = grid_well_raw_grid\n")
        if (
            len(transformed_union_offsets) != 1
            or not (
                certificate_end
                < transformed_union_offsets[0]
                < transformed_grid
            )
        ):
            raise TransformError(
                "raw WELL grid fallback: generated fallback order is invalid"
            )

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
