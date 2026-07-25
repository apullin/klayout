#!/usr/bin/env python3
"""Focused tests for the fail-closed FreePDK45 owner split."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

try:
    from .split_deck import TransformError, split_deck
except ImportError:
    from split_deck import TransformError, split_deck


def source_deck() -> str:
    implant_outputs = "\n".join(
        f'implant.output("IMPLANT.{rule}", "implant {rule}")'
        for rule in range(1, 6)
    )
    contact_outputs = "\n".join(
        f'cont.output("CONTACT.{rule}", "contact {rule}")'
        for rule in range(1, 6)
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<klayout-macro>
<text>
run_m2_rules = drc_shard == "all" || drc_shard == "m2_rules"
run_implant_contact = drc_shard == "all" || drc_shard == "implant_contact"
run_via1_upper_active12 = drc_shard == "all" || drc_shard == "via1_upper_active12"
run_grid = drc_shard == "all" || drc_shard == "grid"
raise unless run_m2_rules || run_implant_contact || run_via1_upper_active12 || run_grid
run_poly = run_m1_enclosure
run_active12 = run_via1_upper_active12
need_gate = (DRC &amp;&amp; (run_poly || run_implant_contact))
need_implant = DRC &amp;&amp; run_implant_contact
m1_contact_owner = requested &amp;&amp; (run_implant_contact || run_m1_enclosure)
if run_active12
active.output("ACTIVE.1", "active 1")
active.output("ACTIVE.2", "active 2")
end
if run_implant_contact

#   Implant
{implant_outputs}
implant.forget

#   Contact
{contact_outputs}

end
</text>
</klayout-macro>
"""


class SplitDeckTest(unittest.TestCase):
    def test_both_splits_preserve_textual_rule_order(self) -> None:
        result = split_deck(
            source_deck(),
            split_implant_contact=True,
            split_active12=True,
        )
        self.assertIn('drc_shard == "implant_contact"', result)
        self.assertIn('drc_shard == "contact"', result)
        self.assertIn('drc_shard == "active12"', result)
        self.assertIn(
            "run_m2_rules || run_implant_contact || run_contact || "
            "run_via1_upper_active12 || run_active12_rules || run_grid",
            result,
        )
        self.assertIn("run_active12 = run_active12_rules", result)
        self.assertIn("run_poly || run_implant_contact", result)
        self.assertIn(
            "m1_contact_owner = requested &amp;&amp; "
            "(run_contact || run_m1_enclosure)",
            result,
        )
        self.assertLess(
            result.index('.output("IMPLANT.5"'),
            result.index('.output("CONTACT.1"'),
        )
        self.assertEqual(result.count("if run_implant_contact\n"), 1)
        self.assertEqual(result.count("if run_contact\n"), 1)
        ET.fromstring(result)

    def test_implant_contact_split_is_independent(self) -> None:
        result = split_deck(
            source_deck(), split_implant_contact=True
        )
        self.assertIn('drc_shard == "implant_contact"', result)
        self.assertIn('drc_shard == "contact"', result)
        self.assertNotIn('drc_shard == "active12"', result)
        self.assertIn("run_active12 = run_via1_upper_active12", result)

    def test_active12_split_is_independent(self) -> None:
        result = split_deck(source_deck(), split_active12=True)
        self.assertIn('drc_shard == "active12"', result)
        self.assertIn('drc_shard == "implant_contact"', result)
        self.assertIn("run_active12 = run_active12_rules", result)

    def test_rejects_duplicate_output_sites(self) -> None:
        source = source_deck().replace(
            'active.output("ACTIVE.1", "active 1")',
            'active.output("ACTIVE.1", "active 1")\n'
            'active.output("ACTIVE.1", "duplicate")',
        )
        with self.assertRaisesRegex(
            TransformError, "ACTIVE.1: expected 1 source sites, found 2"
        ):
            split_deck(source, split_active12=True)

    def test_rejects_noop_and_already_split_inputs(self) -> None:
        with self.assertRaisesRegex(TransformError, "at least one"):
            split_deck(source_deck())
        split = split_deck(source_deck(), split_active12=True)
        with self.assertRaisesRegex(TransformError, "already owner-split"):
            split_deck(split, split_active12=True)

    def test_rejects_source_drift(self) -> None:
        source = source_deck().replace("implant.forget\n", "")
        with self.assertRaisesRegex(
            TransformError, "implant/contact block boundary"
        ):
            split_deck(source, split_implant_contact=True)


if __name__ == "__main__":
    unittest.main()
