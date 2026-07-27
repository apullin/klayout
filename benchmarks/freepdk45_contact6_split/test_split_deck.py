#!/usr/bin/env python3
"""Focused tests for the fail-closed FreePDK45 CONTACT.6 transform."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from split_deck import CONTACT_SHARD, TransformError, split_deck


def source_deck(*, antenna_split: bool = False) -> str:
    antenna_guard = (
        "run_antenna_feol || run_antenna_checks"
        if antenna_split
        else "run_antenna"
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\r\n'
        "<klayout-macro>\r\n"
        "<text>\r\n"
        'run_m1_enclosure = drc_shard == "all" || '
        'drc_shard == "m1_enclosure"\r\n'
        'run_m1_width_space = drc_shard == "all" || '
        'drc_shard == "m1_width_space"\r\n'
        'run_m1_via_class = drc_shard == "all" || '
        'drc_shard == "m1_via_class"\r\n'
        'run_grid = drc_shard == "all" || drc_shard == "grid"\r\n'
        "raise unless run_m1_enclosure || run_m1_width_space || "
        f"run_m1_via_class || {antenna_guard}\r\n"
        "run_contact6 = run_m1_width_space\r\n"
        "need_well = (DRC &amp;&amp; "
        "(run_well || run_active3 || run_active4)) || "
        "(OFFGRID &amp;&amp; run_grid)\r\n"
        "well = nwell.or(pwell) if need_well\r\n"
        "grid = 2.5.nm\r\n"
        "if run_contact6\r\n"
        'cont.separation(poly, 35.nm, euclidian).output("CONTACT.6", '
        '"contact description")\r\n'
        "end\r\n"
        "if run_m1_width_space\r\n"
        "metal1_width, metal1_space = metal1.drc_batch([])\r\n"
        'metal1_width.output("METAL1.1", "width description")\r\n'
        'metal1_space.output("METAL1.2", "space description")\r\n'
        "end\r\n"
        "</text>\r\n"
        "</klayout-macro>\r\n"
    )


class SplitDeckTest(unittest.TestCase):
    def test_split_is_composable_and_preserves_all_mode_order(self) -> None:
        for antenna_split in (False, True):
            with self.subTest(antenna_split=antenna_split):
                result = split_deck(source_deck(antenna_split=antenna_split))
                self.assertNotIn("\r", result)
                self.assertIn(f'drc_shard == "{CONTACT_SHARD}"', result)
                self.assertIn("run_contact6 = run_m1_contact6", result)
                self.assertIn(
                    "run_m1_width_space || run_m1_contact6 || run_m1_via_class",
                    result,
                )
                self.assertNotIn("raw_union_grid_clean?", result)
                self.assertLess(
                    result.index('.output("CONTACT.6"'),
                    result.index('.output("METAL1.1"'),
                )
                self.assertLess(
                    result.index('.output("METAL1.1"'),
                    result.index('.output("METAL1.2"'),
                )
                ET.fromstring(result)

    def test_can_coalesce_contact6_into_the_existing_grid_owner(self) -> None:
        result = split_deck(source_deck(antenna_split=True), owner="grid")
        self.assertIn("run_contact6 = run_grid", result)
        self.assertNotIn("run_m1_contact6", result)
        self.assertNotIn(f'drc_shard == "{CONTACT_SHARD}"', result)
        self.assertIn(
            "grid_well_raw_owner = OFFGRID &amp;&amp; run_grid "
            "&amp;&amp; !run_well &amp;&amp; !run_active3 "
            "&amp;&amp; !run_active4",
            result,
        )
        self.assertIn(
            "grid_well_raw_clean = "
            "nwell.raw_union_grid_clean?(pwell, grid_well_raw_grid)",
            result,
        )
        self.assertIn("grid_well_raw_grid = 2.5.nm", result)
        self.assertIn("grid = grid_well_raw_grid", result)
        self.assertNotIn("\ngrid = 2.5.nm\n", result)
        self.assertIn(
            "need_well = ((DRC &amp;&amp; "
            "(run_well || run_active3 || run_active4)) || "
            "(OFFGRID &amp;&amp; run_grid)) "
            "&amp;&amp; !grid_well_raw_clean",
            result,
        )
        self.assertIn("well = polygon_layer if grid_well_raw_clean", result)
        self.assertEqual(result.count("well = nwell.or(pwell) if need_well"), 1)
        self.assertLess(
            result.index("raw_union_grid_clean?"),
            result.index("well = nwell.or(pwell) if need_well"),
        )
        self.assertLess(
            result.index('.output("CONTACT.6"'),
            result.index('.output("METAL1.1"'),
        )

    def test_rejects_an_already_split_deck(self) -> None:
        result = split_deck(source_deck())
        with self.assertRaisesRegex(TransformError, "already CONTACT.6-split"):
            split_deck(result)

    def test_rejects_a_missing_grid_owner(self) -> None:
        source = source_deck().replace(
            'run_grid = drc_shard == "all" || drc_shard == "grid"\r\n', ""
        )
        with self.assertRaisesRegex(TransformError, "grid owner"):
            split_deck(source, owner="grid")

    def test_rejects_a_missing_output_site(self) -> None:
        source = source_deck().replace(
            'metal1_space.output("METAL1.2", "space description")', ""
        )
        with self.assertRaisesRegex(TransformError, "METAL1.2"):
            split_deck(source)

    def test_grid_owner_rejects_a_missing_literal_well_path(self) -> None:
        source = source_deck(antenna_split=True).replace(
            "well = nwell.or(pwell) if need_well\r\n",
            "",
        )
        with self.assertRaisesRegex(TransformError, "raw WELL grid fallback"):
            split_deck(source, owner="grid")

    def test_grid_owner_rejects_duplicate_literal_well_paths(self) -> None:
        literal = "well = nwell.or(pwell) if need_well\r\n"
        source = source_deck(antenna_split=True).replace(
            literal,
            literal + literal,
        )
        with self.assertRaisesRegex(
            TransformError, "expected one literal WELL union, found 2"
        ):
            split_deck(source, owner="grid")

    def test_grid_owner_rejects_a_misordered_literal_well_path(self) -> None:
        need = (
            "need_well = (DRC &amp;&amp; "
            "(run_well || run_active3 || run_active4)) || "
            "(OFFGRID &amp;&amp; run_grid)\r\n"
        )
        literal = "well = nwell.or(pwell) if need_well\r\n"
        source = source_deck(antenna_split=True).replace(
            need + literal,
            literal + need,
        )
        with self.assertRaisesRegex(
            TransformError, "must follow need_well and precede"
        ):
            split_deck(source, owner="grid")

    def test_grid_owner_rejects_grid_value_drift(self) -> None:
        source = source_deck(antenna_split=True).replace(
            "grid = 2.5.nm\r\n",
            "grid = 5.nm\r\n",
        )
        with self.assertRaisesRegex(TransformError, "raw WELL grid value"):
            split_deck(source, owner="grid")


if __name__ == "__main__":
    unittest.main()
