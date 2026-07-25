#!/usr/bin/env python3
"""Focused tests for the fail-closed FreePDK45 POLY.2 prune."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from prune_deck import PRUNED_BLOCK, SOURCE_BLOCK, TransformError, prune_deck


def source_deck() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\r\n'
        "<klayout-macro>\r\n"
        "<text>\r\n"
        'poly.width(50.nm, euclidian).output("POLY.1", "width")\r\n'
        + SOURCE_BLOCK.replace("\n", "\r\n")
        + 'poly.enclosing(gate, 55.nm, projection).output("POLY.3", '
        '"enclosure")\r\n'
        "</text>\r\n"
        "</klayout-macro>\r\n"
    )


class PruneDeckTest(unittest.TestCase):
    def test_prunes_only_the_dead_block_and_preserves_order(self) -> None:
        result = prune_deck(source_deck())
        self.assertNotIn("\r", result)
        self.assertIn(PRUNED_BLOCK, result)
        self.assertNotIn("poly_sep_active", result)
        self.assertNotIn('.output("POLY.2"', result)
        self.assertNotIn(
            "poly.separation(active, 140.nm, projection)", result
        )
        self.assertLess(
            result.index('.output("POLY.1"'),
            result.index(PRUNED_BLOCK.strip()),
        )
        self.assertLess(
            result.index(PRUNED_BLOCK.strip()),
            result.index('.output("POLY.3"'),
        )
        ET.fromstring(result)

    def test_rejects_an_already_pruned_deck(self) -> None:
        with self.assertRaisesRegex(TransformError, "already POLY.2-pruned"):
            prune_deck(prune_deck(source_deck()))

    def test_rejects_source_contract_drift(self) -> None:
        drifted = source_deck().replace(
            "if poly_sep_active.polygons?",
            "if poly_sep_active.edge_pairs?",
        )
        with self.assertRaisesRegex(TransformError, "exact source match"):
            prune_deck(drifted)

    def test_rejects_a_duplicate_dead_block(self) -> None:
        duplicated = source_deck().replace(
            SOURCE_BLOCK.replace("\n", "\r\n"),
            (SOURCE_BLOCK + SOURCE_BLOCK).replace("\n", "\r\n"),
        )
        with self.assertRaisesRegex(TransformError, "found 2"):
            prune_deck(duplicated)

    def test_rejects_an_extra_poly2_output_site(self) -> None:
        source = source_deck().replace(
            "</text>",
            'polygon_layer.output("POLY.2", "extra")\r\n</text>',
        )
        with self.assertRaisesRegex(TransformError, "output site"):
            prune_deck(source)


if __name__ == "__main__":
    unittest.main()
