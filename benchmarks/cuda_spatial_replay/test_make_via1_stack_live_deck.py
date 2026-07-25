#!/usr/bin/env python3
"""Focused unit tests for exact live-deck source rewrites."""

from __future__ import annotations

import re
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

M2_CPU12 = """metal2_width, metal2_space = metal2.drc_batch([
  width(euclidian) &lt; 70.nm,
  space(euclidian) &lt; 70.nm
])
metal2_width.output("METAL2.1", "METAL2.1 : Minimum width of  intermediate metal2 : 70nm")
metal2_space.output("METAL2.2", "METAL2.2 : Minimum spacing of  intermediate metal2 : 70nm")"""

M2_3_OWNER = """unless via1_stack_requested
  via1_edges_with_less_enclosure = metal2.enclosing(via1, 35.nm, projection).second_edges
  error_corners = via1_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
  via1_edges_with_less_enclosure.forget
  via1.interacting(error_corners.polygons(1.dbu)).output("METAL2.3", "METAL2.3 : Minimum enclosure around via1 on two opposite sides : 35nm")
  error_corners.forget
end"""

M2_CPU49 = """via2_edges_with_less_enclosure = metal2.enclosing(via2, 35.nm, projection).second_edges
error_corners = via2_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
via2_edges_with_less_enclosure.forget
via2.interacting(error_corners.polygons(1.dbu)).output("METAL2.4", "METAL2.4 : Minimum enclosure around via2 on two opposite sides : 35nm")
error_corners.forget
metal2_gt90, metal2_gt270, metal2_gt500, metal2_gt900, metal2_gt1500 = classify_by_width(metal2, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)
metal2_gt90.edges.with_length(300.nm,nil).space(90.nm,euclidian).output("METAL2.5", "METAL2.5 : Minimum spacing of  intermediate metal2 wider than 90 nm and longer than 300 nm : 90nm")
metal2_gt270.edges.with_length(900.nm,nil).space(270.nm,euclidian).output("METAL2.6", "METAL2.6 : Minimum spacing of  intermediate metal2 wider than 270 nm and longer than 900 nm : 270nm")
metal2_gt500.edges.with_length(1.8.um,nil).space(500.nm,euclidian).output("METAL2.7", "METAL2.7 : Minimum spacing of  intermediate metal2 wider than 500 nm and longer than 1.8 um : 500nm")
metal2_gt900.edges.with_length(2.7.um,nil).space(900.nm,euclidian).output("METAL2.8", "METAL2.8 : Minimum spacing of  intermediate metal2 wider than 900 nm and longer than 2.7 um : 900nm")
metal2_gt1500.edges.with_length(4.um,nil).space(1500.nm,euclidian).output("METAL2.9", "METAL2.9 : Minimum spacing of  intermediate metal2 wider than 1500 nm and longer than 4.0 um : 1500nm")
[ metal2_gt90, metal2_gt270, metal2_gt500, metal2_gt900, metal2_gt1500 ].each { |l| l.forget }"""

VIA2_OWNER = """if run_via1_upper_active12

#   via2
# FreePDK Calibre deck incorrectly has this as 65nm so we are going to be compatible.
via2.edges.without_length(65.nm).output("VIA2.1", "VIA2.1 : Minimum/Maximum width of via2 : 65nm")
via2.space(85.nm, euclidian).output("VIA2.2", "VIA2.2 : Minimum spacing of via2 : 85nm")
via2.not(metal2).output("VIA2.3", "VIA2.3 : via2 must be inside metal2")
via2.not(metal3).output("VIA2.4", "VIA2.4 : via2 must be inside metal3")"""

M2_OWNER_BLOCK = (
    f"if run_m2_rules\n\n#   metal2\n{M2_CPU12}\n"
    f"{M2_3_OWNER}\n\n{M2_CPU49}\n\nend\n\n\n{VIA2_OWNER}"
)


def indent(block: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in block.splitlines())


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


class M2RulesTransformTest(unittest.TestCase):
    def test_exact_atomic_rewrite_is_deterministic(self) -> None:
        source = f"before\n{M2_OWNER_BLOCK}\nafter\n"

        first = generator.add_m2_rules(source)
        second = generator.add_m2_rules(source)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("before\n"))
        self.assertTrue(first.endswith("\nafter\n"))
        self.assertEqual(
            first.count(
                'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s'
            ),
            1,
        )
        self.assertEqual(
            first.count("metal2.respond_to?(:cuda_m2_flat_union)"), 1
        )
        self.assertEqual(
            first.count("metal2.cuda_m2_flat_union(via2)"), 1
        )
        self.assertEqual(first.count("m2_rules_flat_results &lt;&lt;"), 8)
        self.assertEqual(first.count("m2_rules_empty.output"), 8)

        expected_empty = [
            "METAL2.1",
            "METAL2.2",
            "METAL2.4",
            "METAL2.5",
            "METAL2.6",
            "METAL2.7",
            "METAL2.8",
            "METAL2.9",
        ]
        actual_empty = re.findall(
            r'm2_rules_empty\.output\("(METAL2\.[0-9])"', first
        )
        self.assertEqual(actual_empty, expected_empty)
        self.assertNotIn('m2_rules_empty.output("METAL2.3"', first)

    def test_speculative_path_never_publishes_and_cleans_every_result(
        self,
    ) -> None:
        transformed = generator.add_m2_rules(M2_OWNER_BLOCK)
        speculative = transformed.split(
            'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s', 1
        )[1].split("m2_rules_empty = polygon_layer", 1)[0]

        self.assertNotIn(".output(", speculative)
        self.assertIn(
            "m2_rules_flat_results.length == 8 &amp;&amp; "
            "m2_rules_flat_results.all? { |result| result.is_empty? }",
            speculative,
        )
        self.assertIn("rescue StandardError =&gt; m2_rules_error", speculative)
        self.assertIn("m2_rules_flat_temps.reverse_each do |layer|", speculative)
        self.assertIn("layer.forget if layer", speculative)
        self.assertIn(
            "rescue StandardError =&gt; m2_rules_cleanup_error", speculative
        )

        clean_assignment = (
            "m2_rules_clean = m2_rules_flat_results.length == 8 &amp;&amp; "
            "m2_rules_flat_results.all? { |result| result.is_empty? }"
        )
        clean_index = transformed.index(clean_assignment)
        ensure_index = transformed.index("\n  ensure\n", clean_index)
        cleanup_index = transformed.index(
            "m2_rules_flat_temps.reverse_each do |layer|", ensure_index
        )
        certified_index = transformed.index("'certified-empty'", cleanup_index)
        self.assertLess(clean_index, ensure_index)
        self.assertLess(ensure_index, cleanup_index)
        self.assertLess(cleanup_index, certified_index)
        self.assertEqual(transformed.count(clean_assignment), 1)
        self.assertEqual(transformed.count("'certified-empty'"), 1)

        # The pristine source operands are passed once and are never mutated.
        for forbidden in (
            "metal2.dup",
            "metal2.flatten",
            "metal2.forget",
            "via2.dup",
            "via2.flatten",
            "via2.forget",
        ):
            self.assertNotIn(forbidden, transformed)

    def test_preserves_literal_cpu_fallbacks(self) -> None:
        transformed = generator.add_m2_rules(M2_OWNER_BLOCK)

        self.assertIn(f"else\n{indent(M2_CPU12)}\nend", transformed)
        self.assertIn(f"else\n{indent(M2_CPU49)}\nend", transformed)

        # Each historical deep expression survives exactly once. The
        # speculative copies use explicitly prefixed flat operands.
        self.assertEqual(
            transformed.count(
                "metal2_width, metal2_space = metal2.drc_batch(["
            ),
            1,
        )
        self.assertEqual(
            transformed.count(
                "via2_edges_with_less_enclosure = "
                "metal2.enclosing(via2, 35.nm, projection).second_edges"
            ),
            1,
        )
        self.assertEqual(
            transformed.count(
                "classify_by_width(metal2, 90.nm, 270.nm, "
                "500.nm, 900.nm, 1500.nm)"
            ),
            1,
        )

    def test_preserves_m2_3_and_via2_ownership_verbatim(self) -> None:
        transformed = generator.add_m2_rules(M2_OWNER_BLOCK)

        self.assertEqual(transformed.count(M2_3_OWNER), 1)
        self.assertEqual(transformed.count(VIA2_OWNER), 1)
        self.assertEqual(transformed.count('.output("METAL2.3"'), 1)
        for category in ("VIA2.1", "VIA2.2", "VIA2.3", "VIA2.4"):
            self.assertEqual(transformed.count(f'.output("{category}"'), 1)

        decisions = [
            match.start()
            for match in re.finditer(r"(?m)^if m2_rules_clean$", transformed)
        ]
        self.assertEqual(len(decisions), 2)
        first_decision, second_decision = decisions
        m2_3_owner = transformed.index(M2_3_OWNER)
        self.assertLess(first_decision, m2_3_owner)
        self.assertLess(m2_3_owner, second_decision)

    def test_rejects_changed_prefix_and_suffix_rules(self) -> None:
        changed_prefix = M2_OWNER_BLOCK.replace(
            "width(euclidian) &lt; 70.nm",
            "width(euclidian) &lt; 71.nm",
            1,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"M2\.1/\.2 speculative transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_m2_rules(changed_prefix)

        changed_suffix = M2_OWNER_BLOCK.replace(
            "with_length(4.um,nil)", "with_length(4.1.um,nil)", 1
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"M2\.4-\.9 speculative transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_m2_rules(changed_suffix)

    def test_rejects_duplicate_owner_blocks(self) -> None:
        duplicated = f"{M2_OWNER_BLOCK}\n{M2_OWNER_BLOCK}"

        with self.assertRaisesRegex(
            RuntimeError,
            r"M2\.1/\.2 speculative transaction: "
            r"expected one source block, found 2",
        ):
            generator.add_m2_rules(duplicated)


if __name__ == "__main__":
    unittest.main()
