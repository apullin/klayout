#!/usr/bin/env python3
"""Focused tests for the fail-closed FreePDK45 owner split."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

try:
    from .split_deck import TransformError, split_deck
except ImportError:
    from split_deck import TransformError, split_deck


def source_deck(
    *,
    implant15: bool = False,
    implant12: bool = False,
    active12: bool = False,
) -> str:
    historical_implant_outputs = "\n".join(
        f'implant.output("IMPLANT.{rule}", "implant {rule}")'
        for rule in range(1, 6)
    )
    implant12_marker = ""
    implant_outputs = historical_implant_outputs
    if implant12:
        implant12_marker = (
            'implant12_request = ENV["KLAYOUT_CUDA_IMPLANT12"].to_s\n'
        )
        first_two = "\n".join(
            f'implant.output("IMPLANT.{rule}", "implant {rule}")'
            for rule in range(1, 3)
        )
        suffix = "\n".join(
            f'implant.output("IMPLANT.{rule}", "implant {rule}")'
            for rule in range(3, 6)
        )
        implant_outputs = (
            f"if implant12_clean\n{first_two}\nelse\n{first_two}\nend\n"
            f"{suffix}"
        )
    implant15_marker = ""
    implant15_end = ""
    if implant15:
        implant15_marker = (
            "# BEGIN KLAYOUT CUDA IMPLANT15 RAW OWNER\n"
            "implant15_raw_clean = false\n"
        )
        clean_outputs = "\n".join(
            f'empty.output("IMPLANT.{rule}", "implant {rule}")'
            for rule in range(1, 6)
        )
        implant_outputs = (
            f"if implant15_raw_clean\n{clean_outputs}\nelse\n"
            f"{implant_outputs}"
        )
        implant15_end = "\nend"
    contact_outputs = "\n".join(
        f'cont.output("CONTACT.{rule}", "contact {rule}")'
        for rule in range(1, 6)
    )
    active12_marker = ""
    active12_outputs = (
        'active.output("ACTIVE.1", "active 1")\n'
        'active.output("ACTIVE.2", "active 2")'
    )
    if active12:
        active12_marker = (
            'active12_request = ENV["KLAYOUT_CUDA_ACTIVE12"].to_s\n'
        )
        active12_outputs = (
            'if active12_clean\n'
            'empty.output("ACTIVE.1", "active 1")\n'
            'empty.output("ACTIVE.2", "active 2")\n'
            'else\n'
            f'{active12_outputs}\n'
            'end'
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
{implant12_marker}{implant15_marker}implant = nplus.or(pplus) if need_implant
m1_contact_owner = requested &amp;&amp; (run_implant_contact || run_m1_enclosure)
if run_active12
{active12_marker}{active12_outputs}
end
if run_implant_contact

#   Implant
{implant_outputs}
implant.forget{implant15_end}

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

    def test_raw_implant15_and_nested_implant12_split_exactly(self) -> None:
        for implant12, expected_implant12_count in ((False, 2), (True, 3)):
            with self.subTest(implant12=implant12):
                result = split_deck(
                    source_deck(implant15=True, implant12=implant12),
                    split_implant_contact=True,
                )
                self.assertEqual(
                    result.count('.output("IMPLANT.1"'),
                    expected_implant12_count,
                )
                self.assertEqual(
                    result.count('.output("IMPLANT.2"'),
                    expected_implant12_count,
                )
                for rule in range(3, 6):
                    self.assertEqual(
                        result.count(f'.output("IMPLANT.{rule}"'),
                        2,
                    )
                self.assertIn(
                    "implant.forget\nend\n\nend\n\n\n"
                    "if run_contact\n\n#   Contact",
                    result,
                )
                ET.fromstring(result)

    def test_transaction_markers_bind_the_only_accepted_output_counts(
        self,
    ) -> None:
        missing_raw_marker = source_deck(implant15=True).replace(
            "# BEGIN KLAYOUT CUDA IMPLANT15 RAW OWNER\n",
            "",
        )
        with self.assertRaisesRegex(
            TransformError,
            "IMPLANT.1: expected 1 source sites, found 2",
        ):
            split_deck(
                missing_raw_marker,
                split_implant_contact=True,
            )

        duplicate_raw_marker = source_deck(implant15=True).replace(
            "# BEGIN KLAYOUT CUDA IMPLANT15 RAW OWNER\n",
            "# BEGIN KLAYOUT CUDA IMPLANT15 RAW OWNER\n"
            "# BEGIN KLAYOUT CUDA IMPLANT15 RAW OWNER\n",
        )
        with self.assertRaisesRegex(
            TransformError,
            "raw IMPLANT.1-.5 transaction: expected zero or one exact marker",
        ):
            split_deck(
                duplicate_raw_marker,
                split_implant_contact=True,
            )

        missing_implant12_marker = source_deck(implant12=True).replace(
            'implant12_request = ENV["KLAYOUT_CUDA_IMPLANT12"].to_s\n',
            "",
        )
        with self.assertRaisesRegex(
            TransformError,
            "IMPLANT.1: expected 1 source sites, found 2",
        ):
            split_deck(
                missing_implant12_marker,
                split_implant_contact=True,
            )

    def test_active12_split_is_independent(self) -> None:
        result = split_deck(source_deck(), split_active12=True)
        self.assertIn('drc_shard == "active12"', result)
        self.assertIn('drc_shard == "implant_contact"', result)
        self.assertIn("run_active12 = run_active12_rules", result)

    def test_active12_transaction_marker_accepts_clean_and_fallback_sites(
        self,
    ) -> None:
        result = split_deck(
            source_deck(active12=True),
            split_active12=True,
        )
        self.assertEqual(result.count('.output("ACTIVE.1"'), 2)
        self.assertEqual(result.count('.output("ACTIVE.2"'), 2)
        self.assertIn("run_active12 = run_active12_rules", result)
        ET.fromstring(result)

        missing_marker = source_deck(active12=True).replace(
            'active12_request = ENV["KLAYOUT_CUDA_ACTIVE12"].to_s\n',
            "",
        )
        with self.assertRaisesRegex(
            TransformError,
            "ACTIVE.1: expected 1 source sites, found 2",
        ):
            split_deck(missing_marker, split_active12=True)

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
