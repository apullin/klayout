#!/usr/bin/env python3
"""Remove FreePDK45's provably unreachable POLY.2 calculation.

The historical deck computes ``Region#separation`` and then guards its only
consumer with ``polygons?``.  The DRC API always returns an EdgePairs layer
from ``separation``, while ``polygons?`` is true only for Region layers.  The
guard can therefore never publish POLY.2, but the separation still runs.

This transform is deliberately exact and fail-closed: it accepts one known
source block, replaces it with an explanatory marker, and rejects drift,
duplicates, and already-pruned inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


SOURCE_BLOCK = """\
poly_sep_active = poly.separation(active, 140.nm, projection)
if poly_sep_active.polygons?
  poly_sep_active.polygons.without_area(0).output("POLY.2", "POLY.2 : Minimum spacing of poly AND active: 140nm")
end
poly_sep_active.forget
"""

PRUNED_BLOCK = """\
# POLY.2 is intentionally absent.  DRC separation always returns EdgePairs,
# so the historical polygons? guard was unreachable despite paying for the
# full poly/active separation.
"""


class TransformError(ValueError):
    """The source deck is not the qualified unpruned FreePDK45 variant."""


def prune_deck(source: str) -> str:
    """Return a deterministic LF-normalized deck without dead POLY.2 work."""

    text = source.replace("\r\n", "\n").replace("\r", "\n")
    if PRUNED_BLOCK.strip() in text:
        raise TransformError("source deck is already POLY.2-pruned")

    source_count = text.count(SOURCE_BLOCK)
    if source_count != 1:
        raise TransformError(
            f"POLY.2 dead block: expected one exact source match, found {source_count}"
        )
    output_count = text.count('.output("POLY.2"')
    if output_count != 1:
        raise TransformError(
            f"POLY.2 output site: expected one source match, found {output_count}"
        )

    transformed = text.replace(SOURCE_BLOCK, PRUNED_BLOCK, 1)
    if (
        "poly_sep_active" in transformed
        or '.output("POLY.2"' in transformed
        or "poly.separation(active, 140.nm, projection)" in transformed
    ):
        raise TransformError("POLY.2 source survived the exact replacement")

    try:
        ET.fromstring(transformed)
    except ET.ParseError as exc:
        raise TransformError(f"generated deck is not valid XML: {exc}") from exc
    return transformed


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
    parser.add_argument("input", type=Path, help="unpruned FreePDK45 .lydrc")
    parser.add_argument("output", type=Path, help="generated pruned .lydrc")
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
        transformed = prune_deck(source)
        _write_atomic(args.output, transformed)
    except (OSError, UnicodeDecodeError, TransformError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
