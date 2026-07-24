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
        "raise unless run_m1_enclosure || run_m1_width_space || "
        f"run_m1_via_class || {antenna_guard}\r\n"
        "run_contact6 = run_m1_width_space\r\n"
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
                self.assertLess(
                    result.index('.output("CONTACT.6"'),
                    result.index('.output("METAL1.1"'),
                )
                self.assertLess(
                    result.index('.output("METAL1.1"'),
                    result.index('.output("METAL1.2"'),
                )
                ET.fromstring(result)

    def test_rejects_an_already_split_deck(self) -> None:
        result = split_deck(source_deck())
        with self.assertRaisesRegex(TransformError, "already CONTACT.6-split"):
            split_deck(result)

    def test_rejects_a_missing_output_site(self) -> None:
        source = source_deck().replace(
            'metal1_space.output("METAL1.2", "space description")', ""
        )
        with self.assertRaisesRegex(TransformError, "METAL1.2"):
            split_deck(source)


if __name__ == "__main__":
    unittest.main()
