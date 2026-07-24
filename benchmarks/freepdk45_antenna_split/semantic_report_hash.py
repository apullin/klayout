#!/usr/bin/env python3
"""Hash report semantics while normalizing unordered tagged diagnostics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "scripts"))

import merge_sharded_lyrdb as reports  # noqa: E402


def semantic_model(path: Path) -> tuple[object, ...]:
    root = reports._parse_report(path)
    metadata = reports._metadata(root)
    metadata["generator"] = "DRC_GENERATOR"
    cells = reports._cell_records(root)
    ordered_cells = tuple(
        (name, cells[name]) for name in reports._ordered_cells(cells)
    )
    items = tuple(
        sorted(
            (repr(fingerprint), multiplicity)
            for fingerprint, multiplicity in reports._item_counter(root).items()
        )
    )
    return (
        tuple(metadata.items()),
        tuple(sorted(reports._tags(root))),
        tuple(reports._categories(root)),
        ordered_cells,
        items,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, metavar="REPORT")
    args = parser.parse_args()
    models = []
    for path in args.reports:
        model = semantic_model(path)
        models.append(model)
        digest = hashlib.sha256(repr(model).encode("utf-8")).hexdigest()
        print(
            f"{path} {digest} categories={len(model[2])} "
            f"cells={len(model[3])} "
            f"items={sum(count for _, count in model[4])}"
        )
    if len(models) > 1:
        print(f"all_equal={all(model == models[0] for model in models[1:])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
