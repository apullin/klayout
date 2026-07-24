#!/usr/bin/env python3
"""Focused unit tests for exact live-deck source rewrites."""

from __future__ import annotations

import unittest

import make_via1_stack_live_deck as generator


IMPLANT1 = (
    'implant.separation(gate, 70.nm, projection).polygons.without_area(0)'
    '.output("IMPLANT.1", "IMPLANT.1 : Minimum spacing of nimplant/ '
    'pimplant to channel : 70nm")'
)
IMPLANT2 = (
    'implant.separation(cont, 25.nm, projection).polygons.without_area(0)'
    '.output("IMPLANT.2", "IMPLANT.1 : Minimum spacing of nimplant/ '
    'pimplant to contact : 25nm")'
)
SOURCE_BLOCK = f"{IMPLANT1}\n{IMPLANT2}"


class Implant12TransformTest(unittest.TestCase):
    def test_rewrites_only_the_exact_pair(self) -> None:
        source = f"before\n{SOURCE_BLOCK}\nafter\n"

        transformed = generator.add_implant12(source)

        self.assertTrue(transformed.startswith("before\n"))
        self.assertTrue(transformed.endswith("\nafter\n"))
        self.assertIn(
            "implant.respond_to?(:cuda_implant12_clean?) "
            "&amp;&amp; implant.cuda_implant12_clean?(gate, cont)",
            transformed,
        )
        self.assertIn(
            'implant12_empty.output("IMPLANT.1", '
            '"IMPLANT.1 : Minimum spacing of nimplant/ pimplant to channel : '
            '70nm")',
            transformed,
        )
        self.assertIn(
            'implant12_empty.output("IMPLANT.2", '
            '"IMPLANT.1 : Minimum spacing of nimplant/ pimplant to contact : '
            '25nm")',
            transformed,
        )

        # The only separation calls left are the two historical expressions
        # preserved literally inside the fail-closed CPU branch.
        self.assertEqual(transformed.count(IMPLANT1), 1)
        self.assertEqual(transformed.count(IMPLANT2), 1)
        false_branch = transformed.split("else\n", 1)[1].split("\nend", 1)[0]
        self.assertEqual(false_branch, f"  {IMPLANT1}\n  {IMPLANT2}")

    def test_preserves_order_and_keeps_postprocessing_out_of_clean_path(
        self,
    ) -> None:
        transformed = generator.add_implant12(SOURCE_BLOCK)

        clean_branch = transformed.split(
            "if implant12_clean\n", 1
        )[1].split("\nelse", 1)[0]
        self.assertLess(
            clean_branch.index('output("IMPLANT.1"'),
            clean_branch.index('output("IMPLANT.2"'),
        )
        self.assertNotIn(".polygons", clean_branch)
        self.assertNotIn(".without_area", clean_branch)

    def test_rejects_a_changed_rule(self) -> None:
        changed = SOURCE_BLOCK.replace("25.nm", "26.nm")

        with self.assertRaisesRegex(
            RuntimeError,
            r"IMPLANT\.1/\.2 transaction: expected one source block, found 0",
        ):
            generator.add_implant12(changed)

    def test_rejects_duplicate_source_pairs(self) -> None:
        duplicated = f"{SOURCE_BLOCK}\n{SOURCE_BLOCK}"

        with self.assertRaisesRegex(
            RuntimeError,
            r"IMPLANT\.1/\.2 transaction: expected one source block, found 2",
        ):
            generator.add_implant12(duplicated)


if __name__ == "__main__":
    unittest.main()
