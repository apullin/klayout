#!/usr/bin/env python3
"""Focused tests for the fail-closed FreePDK45 antenna deck transform."""

from __future__ import annotations

import unittest

from split_deck import ANTENNA_CATEGORIES, TransformError, split_deck


def source_deck() -> str:
    checks = []
    for layer in range(1, 11):
        if layer > 1:
            checks.extend(
                [
                    f"connect(metal{layer - 1}, via{layer - 1})",
                    f"connect(via{layer - 1}, metal{layer})",
                ]
            )
        checks.append(
            f'antenna_check(gate, metal{layer}, 300.0, diode).output('
            f'"METAL{layer}_ANTENNA", "description {layer}")'
        )
    antenna = "\n".join(checks)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\r\n'
        "<klayout-macro>\r\n"
        "<text>\r\n"
        'run_antenna = drc_shard == "all" || drc_shard == "antenna"\r\n'
        "raise unless run_other || run_grid || run_antenna\r\n"
        "run_well = run_antenna\r\n"
        "run_active4 = run_antenna\r\n"
        "need_gate = a || (ANTENNA &amp;&amp; run_antenna)\r\n"
        "#   ANTENNA checks\r\n"
        "################\r\n"
        "if ANTENNA &amp;&amp; run_antenna\r\n"
        "diode = nplus &amp; active - nwell\r\n"
        "connect(gate, poly)\r\n"
        "connect(poly, cont)\r\n"
        "connect(diode, cont)\r\n"
        "connect(cont, metal1)\r\n"
        f"{antenna}\r\n"
        "end\r\n"
        "# time spent for the DRC\r\n"
        "</text>\r\n"
        "</klayout-macro>\r\n"
    )


class SplitDeckTest(unittest.TestCase):
    def test_split_is_lf_normalized_and_preserves_all_check_order(self) -> None:
        result = split_deck(source_deck())
        self.assertNotIn("\r", result)
        self.assertIn('drc_shard == "antenna_feol"', result)
        self.assertIn('drc_shard == "antenna_m1_m2"', result)
        self.assertIn('drc_shard == "antenna_m3_m10"', result)
        self.assertIn("run_well = run_antenna_feol", result)
        self.assertIn("run_active4 = run_antenna_feol", result)
        self.assertIn("ANTENNA &amp;&amp; run_antenna_checks", result)

        positions = [result.index(f'.output("{name}"') for name in ANTENNA_CATEGORIES]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(
            result.index("if run_antenna_m1_m2"),
            result.index('.output("METAL1_ANTENNA"'),
        )
        self.assertLess(
            result.index("if run_antenna_m3_m10"),
            result.index('.output("METAL3_ANTENNA"'),
        )

    def test_rejects_an_already_split_deck(self) -> None:
        with self.assertRaisesRegex(TransformError, "already antenna-split"):
            split_deck(source_deck().replace("antenna", "antenna_feol", 1))

    def test_rejects_a_missing_output_site(self) -> None:
        source = source_deck().replace(
            '.output("METAL7_ANTENNA", "description 7")', ""
        )
        with self.assertRaisesRegex(TransformError, "METAL7_ANTENNA"):
            split_deck(source)


if __name__ == "__main__":
    unittest.main()
