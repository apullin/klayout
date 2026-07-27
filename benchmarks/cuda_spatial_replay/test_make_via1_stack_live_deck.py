#!/usr/bin/env python3
"""Focused unit tests for exact live-deck source rewrites."""

from __future__ import annotations

import inspect
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
IMPLANT3 = (
    'implant.width(45.nm, euclidian).output("IMPLANT.3", '
    '"IMPLANT.3 : Minimum width of nimplant/ pimplant  : 45nm")'
)
IMPLANT4 = (
    'implant.space(45.nm, euclidian).output("IMPLANT.4", '
    '"IMPLANT.4 : Minimum spacing of nimplant/ pimplant  : 45nm")'
)
IMPLANT5 = (
    'nplus.and(pplus).output("IMPLANT.5", '
    '"IMPLANT.5 : Nimplant and pimplant must not overlap")'
)
IMPLANT_UNION = "implant = nplus.or(pplus) if need_implant"
IMPLANT15_RULES = (
    f"{SOURCE_BLOCK}\n{IMPLANT3}\n{IMPLANT4}\n{IMPLANT5}\nimplant.forget"
)
IMPLANT15_SOURCE = (
    f"gate = poly &amp; active if need_gate\n{IMPLANT_UNION}\n"
    f"intervening\n#   Implant\n{IMPLANT15_RULES}"
)
POLY3 = (
    "poly.enclosing(gate, 55.nm, projection).polygons.without_area(0)"
    '.output("POLY.3", "POLY.3 : Minimum poly extension beyond active : '
    '55nm")'
)
POLY4 = (
    "active.enclosing(gate, 70.nm, projection).polygons.without_area(0)"
    '.output("POLY.4", "POLY.4 : Minimum enclosure of active around gate : '
    '70nm")'
)
POLY34_GATE = "gate = poly &amp; active if need_gate"
POLY34_LAZY_GATE = (
    "gate = poly &amp; active if need_gate &amp;&amp; !poly34_raw_clean"
)
POLY34_RAW_OWNER = (
    "poly34_raw_owner = poly34_requested &amp;&amp; DRC &amp;&amp; run_poly "
    "&amp;&amp; !run_implant_contact &amp;&amp; "
    "!(ANTENNA &amp;&amp; run_antenna)"
)
POLY34_RULE_BLOCK = f"{POLY3}\n{POLY4}"
POLY34_SOURCE_BLOCK = (
    f"{POLY34_GATE}\nintervening\n{POLY34_RULE_BLOCK}"
)
ACTIVE3_WELL = "well = nwell.or(pwell) if need_well"
ACTIVE3_RULE = (
    "well.enclosing(active, 55.nm, euclidian)"
    '.output("ACTIVE.3", "ACTIVE.3 : Minimum enclosure/spacing of '
    'nwell/pwell to active: 55nm")'
)
ACTIVE3_SOURCE_BLOCK = f"{ACTIVE3_WELL}\nintervening\n{ACTIVE3_RULE}"
ACTIVE4_RULE = (
    'active.not(well).output("ACTIVE.4", '
    '"ACTIVE.4 : active must be inside nwell or pwell")'
)
ACTIVE4_SOURCE_BLOCK = f"{ACTIVE3_WELL}\nintervening\n{ACTIVE4_RULE}"

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

M1_CPU59 = """metal1_gt90, metal1_gt270, metal1_gt500, metal1_gt900, metal1_gt1500 = classify_by_width(metal1, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)
metal1_gt90.edges.with_length(300.nm,nil).space(90.nm,euclidian).output("METAL1.5", "METAL1.5 : Minimum spacing of metal1 wider than 90 nm and longer than 300 nm : 90nm")
metal1_gt270.edges.with_length(900.nm,nil).space(270.nm,euclidian).output("METAL1.6", "METAL1.6 : Minimum spacing of metal1 wider than 270 nm and longer than 900 nm : 270nm")
metal1_gt500.edges.with_length(1.8.um,nil).space(500.nm,euclidian).output("METAL1.7", "METAL1.7 : Minimum spacing of metal1 wider than 500 nm and longer than 1.8 um : 500nm")
metal1_gt900.edges.with_length(2.7.um,nil).space(900.nm,euclidian).output("METAL1.8", "METAL1.8 : Minimum spacing of metal1 wider than 900 nm and longer than 2.7 um : 900nm")
metal1_gt1500.edges.with_length(4.um,nil).space(1500.nm,euclidian).output("METAL1.9", "METAL1.9 : Minimum spacing of metal1 wider than 1500 nm and longer than 4.0 um : 1500nm")
[ metal1_gt90, metal1_gt270, metal1_gt500, metal1_gt900, metal1_gt1500 ].each { |l| l.forget }"""

ACTIVE12_CPU = """active.width(90.nm, euclidian).output("ACTIVE.1", "ACTIVE.1 : Minimum width of active : 90nm")
active.space(80.nm, euclidian).output("ACTIVE.2", "ACTIVE.2 : Minimum spacing of active : 80nm")"""


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


class Implant15RawTransformTest(unittest.TestCase):
    def test_is_deterministic_ordered_and_bypasses_the_union_only_on_clean(
        self,
    ) -> None:
        source = f"before\n{IMPLANT15_SOURCE}\nafter\n"

        first = generator.add_implant15(source)
        second = generator.add_implant15(source)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("before\n"))
        self.assertTrue(first.endswith("\nafter\n"))
        self.assertEqual(first.count(IMPLANT_UNION), 1)
        self.assertLess(
            first.index("gate = poly &amp; active if need_gate"),
            first.index("nplus.cuda_implant15_raw_clean?(pplus, gate, cont)"),
        )
        self.assertLess(
            first.index("nplus.cuda_implant15_raw_clean?(pplus, gate, cont)"),
            first.index(f"  {IMPLANT_UNION}"),
        )

        clean_branch = first.split(
            "#   Implant\nif implant15_raw_clean\n", 1
        )[1].split("\nelse\n", 1)[0]
        self.assertEqual(
            re.findall(
                r'implant15_empty\.output\("(IMPLANT\.[1-5])"',
                clean_branch,
            ),
            [f"IMPLANT.{rule}" for rule in range(1, 6)],
        )
        self.assertNotIn("nplus.or(pplus)", clean_branch)
        self.assertNotIn("implant.separation", clean_branch)
        self.assertNotIn("implant.width", clean_branch)
        self.assertNotIn("implant.space", clean_branch)
        self.assertNotIn("nplus.and(pplus)", clean_branch)

    def test_every_decline_keeps_one_literal_union_and_five_rule_fallback(
        self,
    ) -> None:
        transformed = generator.add_implant15(IMPLANT15_SOURCE)

        self.assertIn(
            "implant15_owner = implant15_requested &amp;&amp; DRC "
            "&amp;&amp; run_implant_contact",
            transformed,
        )
        self.assertIn(
            "if nplus.respond_to?(:cuda_implant15_raw_clean?)",
            transformed,
        )
        self.assertIn(
            "rescue StandardError =&gt; implant15_raw_error",
            transformed,
        )
        method_index = transformed.index(
            "nplus.cuda_implant15_raw_clean?(pplus, gate, cont)"
        )
        empty_index = transformed.index(
            "implant15_empty = polygon_layer", method_index
        )
        rescue_index = transformed.index(
            "rescue StandardError =&gt; implant15_raw_error", empty_index
        )
        self.assertLess(method_index, empty_index)
        self.assertLess(empty_index, rescue_index)
        self.assertIn("implant15_empty = nil", transformed[rescue_index:])
        self.assertIn(
            'implant15_raw_reason = "exception:'
            '#{implant15_raw_error.class}"',
            transformed,
        )
        self.assertIn(
            "unless implant15_raw_clean\n"
            f"  {IMPLANT_UNION}\n"
            "end",
            transformed,
        )
        for historical_rule in (
            IMPLANT1,
            IMPLANT2,
            IMPLANT3,
            IMPLANT4,
            IMPLANT5,
        ):
            self.assertEqual(transformed.count(historical_rule), 1)
        self.assertEqual(transformed.count("implant.forget"), 1)
        self.assertIn(
            "CUDA IMPLANT.1-.5 raw transaction: "
            "#{implant15_raw_clean ? 'certified-empty' : "
            "'full-cpu-fallback'} reason=#{implant15_raw_reason}",
            transformed,
        )

    def test_implant12_remains_inside_only_the_outer_fallback(self) -> None:
        with_implant12 = generator.add_implant12(IMPLANT15_SOURCE)
        transformed = generator.add_implant15(with_implant12)

        clean_branch, fallback = transformed.split(
            "#   Implant\nif implant15_raw_clean\n", 1
        )[1].split("\nelse\n", 1)
        self.assertNotIn("implant12_", clean_branch)
        self.assertIn("implant12_clean", fallback)
        self.assertEqual(
            transformed.count(
                "implant.respond_to?(:cuda_implant12_clean?)"
            ),
            1,
        )
        self.assertEqual(transformed.count(IMPLANT1), 1)
        self.assertEqual(transformed.count(IMPLANT2), 1)
        self.assertEqual(transformed.count(IMPLANT3), 1)
        self.assertEqual(transformed.count(IMPLANT4), 1)
        self.assertEqual(transformed.count(IMPLANT5), 1)

    def test_composes_after_poly34_and_rejects_missing_or_late_gate(
        self,
    ) -> None:
        with_poly_rules = f"{IMPLANT15_SOURCE}\n{POLY34_RULE_BLOCK}"
        poly_first = generator.add_poly34(with_poly_rules)
        transformed = generator.add_implant15(poly_first)
        self.assertIn(POLY34_LAZY_GATE, transformed)
        self.assertLess(
            transformed.index(POLY34_LAZY_GATE),
            transformed.index(
                "nplus.cuda_implant15_raw_clean?(pplus, gate, cont)"
            ),
        )

        missing_gate = IMPLANT15_SOURCE.replace(
            "gate = poly &amp; active if need_gate\n",
            "",
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "expected one GATE construction followed by one implant union",
        ):
            generator.add_implant15(missing_gate)

        late_gate = IMPLANT15_SOURCE.replace(
            "gate = poly &amp; active if need_gate\n",
            "",
        ).replace(
            "intervening\n",
            "intervening\ngate = poly &amp; active if need_gate\n",
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "expected one GATE construction followed by one implant union",
        ):
            generator.add_implant15(late_gate)

    def test_source_drift_and_duplicates_fail_closed(self) -> None:
        changed = IMPLANT15_SOURCE.replace("45.nm", "46.nm", 1)
        with self.assertRaisesRegex(
            RuntimeError,
            r"IMPLANT\.1-\.5 raw rule transaction: expected one exact",
        ):
            generator.add_implant15(changed)

        duplicated = (
            f"gate = poly &amp; active if need_gate\n{IMPLANT_UNION}\n"
            f"#   Implant\n{IMPLANT15_RULES}\n"
            f"#   Implant\n{IMPLANT15_RULES}"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"IMPLANT\.1-\.5 raw rule transaction: expected one exact",
        ):
            generator.add_implant15(duplicated)


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
        self.assertEqual(first.count("m2_rules_flat_results &lt;&lt;"), 3)
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
            "m2_rules_flat_results.length == 3 &amp;&amp; "
            "m2_rules_flat_results.all? { |result| result.is_empty? }",
            speculative,
        )
        self.assertIn("prefix-clean+suffix-certified", speculative)
        self.assertNotIn("m2_rules_flat_gt90", speculative)
        self.assertNotIn("m2_rules_flat_edges_5", speculative)
        self.assertNotIn("m2_rules_flat_metal2_9", speculative)
        self.assertIn("rescue StandardError =&gt; m2_rules_error", speculative)
        self.assertIn("m2_rules_flat_temps.reverse_each do |layer|", speculative)
        self.assertIn("layer.forget if layer", speculative)
        self.assertIn(
            "rescue StandardError =&gt; m2_rules_cleanup_error", speculative
        )

        clean_assignment = (
            "m2_rules_clean = m2_rules_flat_results.length == 3 &amp;&amp; "
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


class M1ResidentMorphologyTransformTest(unittest.TestCase):
    def test_rewrites_only_exact_suffix_and_is_deterministic(self) -> None:
        source = f"before\nif run_m1_via_class\n{M1_CPU59}\nend\nafter\n"

        first = generator.add_m1_5_9(source)
        second = generator.add_m1_5_9(source)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("before\n"))
        self.assertTrue(first.endswith("\nafter\n"))
        self.assertEqual(
            first.count(
                'ENV["KLAYOUT_CUDA_M1_5_9"].to_s'
            ),
            1,
        )
        self.assertEqual(
            first.count(
                "metal1.respond_to?(:cuda_m1_5_9_clean?)"
            ),
            1,
        )
        self.assertEqual(
            first.count("metal1.cuda_m1_5_9_clean?"), 1
        )
        self.assertEqual(first.count("m1_5_9_empty.output"), 5)
        self.assertEqual(first.count("classify_by_width(metal1,"), 1)
        self.assertEqual(first.count("].each { |l| l.forget }"), 1)

    def test_clean_path_publishes_only_empty_categories(self) -> None:
        transformed = generator.add_m1_5_9(M1_CPU59)
        clean = transformed.split(
            "if m1_5_9_clean\n", 1
        )[1].split("\nelse", 1)[0]

        self.assertNotIn("classify_by_width", clean)
        self.assertNotIn(".edges", clean)
        self.assertNotIn(".space", clean)
        self.assertEqual(
            re.findall(
                r'm1_5_9_empty\.output\("(METAL1\.[5-9])"',
                clean,
            ),
            [f"METAL1.{rule}" for rule in range(5, 10)],
        )

    def test_false_path_retains_literal_cpu_sequence(self) -> None:
        transformed = generator.add_m1_5_9(M1_CPU59)
        fallback = transformed.split(
            "if m1_5_9_clean\n", 1
        )[1].split("\nelse\n", 1)[1].rsplit("\nend", 1)[0]

        self.assertEqual(fallback, indent(M1_CPU59))
        self.assertIn("rescue StandardError", transformed)
        self.assertIn("m1_5_9_clean = false", transformed)

    def test_rejects_source_drift_or_duplicates(self) -> None:
        changed = M1_CPU59.replace("with_length(4.um,nil)", "with_length(4.1.um,nil)")
        with self.assertRaisesRegex(
            RuntimeError,
            r"M1\.5-\.9 exact resident morphology transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_m1_5_9(changed)

        duplicated = f"{M1_CPU59}\n{M1_CPU59}"
        with self.assertRaisesRegex(
            RuntimeError,
            r"M1\.5-\.9 exact resident morphology transaction: "
            r"expected one source block, found 2",
        ):
            generator.add_m1_5_9(duplicated)


class Active12TransformTest(unittest.TestCase):
    def test_exact_atomic_rewrite_is_deterministic(self) -> None:
        source = f"before\n{ACTIVE12_CPU}\nafter\n"

        first = generator.add_active12(source)
        second = generator.add_active12(source)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("before\n"))
        self.assertTrue(first.endswith("\nafter\n"))
        self.assertEqual(
            first.count('ENV["KLAYOUT_CUDA_ACTIVE12"].to_s'),
            1,
        )
        self.assertEqual(
            first.count("active.data.respond_to?(:cuda_active12_clean?)"),
            1,
        )
        self.assertEqual(first.count("active.data.cuda_active12_clean?"), 1)
        self.assertEqual(first.count("active12_empty.output"), 2)
        for cpu_expression in ACTIVE12_CPU.splitlines():
            self.assertEqual(first.count(cpu_expression), 1)

    def test_clean_path_publishes_only_two_empty_categories(self) -> None:
        transformed = generator.add_active12(ACTIVE12_CPU)
        clean = transformed.split(
            "if active12_clean\n", 1
        )[1].split("\nelse", 1)[0]

        self.assertNotIn(".width", clean)
        self.assertNotIn(".space", clean)
        self.assertEqual(
            re.findall(
                r'active12_empty\.output\("(ACTIVE\.[12])"',
                clean,
            ),
            ["ACTIVE.1", "ACTIVE.2"],
        )

    def test_every_decline_retains_literal_cpu_pair(self) -> None:
        transformed = generator.add_active12(ACTIVE12_CPU)
        fallback = transformed.split(
            "if active12_clean\n", 1
        )[1].split("\nelse\n", 1)[1].rsplit("\nend", 1)[0]

        self.assertEqual(fallback, indent(ACTIVE12_CPU))
        self.assertIn("rescue StandardError", transformed)
        self.assertIn("active12_clean = false", transformed)

    def test_rejects_source_drift_or_duplicates(self) -> None:
        changed = ACTIVE12_CPU.replace("80.nm", "81.nm")
        with self.assertRaisesRegex(
            RuntimeError,
            r"ACTIVE\.1/\.2 exact resident transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_active12(changed)

        duplicated = f"{ACTIVE12_CPU}\n{ACTIVE12_CPU}"
        with self.assertRaisesRegex(
            RuntimeError,
            r"ACTIVE\.1/\.2 exact resident transaction: "
            r"expected one source block, found 2",
        ):
            generator.add_active12(duplicated)

    def test_composes_after_active4_without_replacing_prior_win(self) -> None:
        source = (
            f"before\n{ACTIVE12_CPU}\n"
            f"{ACTIVE4_SOURCE_BLOCK}\nafter\n"
        )
        active4 = generator.add_active4_well_union(source)
        composed = generator.add_active12(active4)

        self.assertEqual(
            composed,
            generator.add_active12(
                generator.add_active4_well_union(source)
            ),
        )
        self.assertEqual(
            composed.count(
                "# BEGIN KLAYOUT CUDA ACTIVE4 EXACT WELL UNION TRANSACTION"
            ),
            1,
        )
        self.assertEqual(
            composed.count('ENV["KLAYOUT_CUDA_ACTIVE12"].to_s'),
            1,
        )
        self.assertEqual(composed.count(ACTIVE3_WELL), 1)
        self.assertEqual(composed.count(ACTIVE4_RULE), 1)
        for cpu_expression in ACTIVE12_CPU.splitlines():
            self.assertEqual(composed.count(cpu_expression), 1)
        self.assertEqual(composed.count("active4_well_union_empty.output"), 1)
        self.assertEqual(composed.count("active12_empty.output"), 2)

        main_source = inspect.getsource(generator.main)
        self.assertLess(
            main_source.index("if args.active4_well_union:"),
            main_source.index("if args.active12:"),
        )
        self.assertLess(
            main_source.index(
                "output = add_active4_well_union(output)"
            ),
            main_source.index("output = add_active12(output)"),
        )


class Poly34TransformTest(unittest.TestCase):
    def test_rewrites_exact_gate_and_rule_anchors(self) -> None:
        source = f"before\n{POLY34_SOURCE_BLOCK}\nafter\n"

        transformed = generator.add_poly34(source)

        self.assertTrue(transformed.startswith("before\n"))
        self.assertTrue(transformed.endswith("\nafter\n"))
        self.assertIn(
            "poly34_raw_clean = "
            "poly.respond_to?(:cuda_poly34_raw_clean?) "
            "&amp;&amp; poly.cuda_poly34_raw_clean?(active)",
            transformed,
        )
        self.assertIn(
            "poly34_clean = poly.respond_to?(:cuda_poly34_clean?) "
            "&amp;&amp; poly.cuda_poly34_clean?(active, gate)",
            transformed,
        )
        self.assertIn("rescue StandardError =&gt; error", transformed)
        self.assertIn("poly34_raw_clean = false", transformed)
        self.assertIn(
            'info("CUDA POLY.3/.4 Ruby fallback: #{poly34_error}") '
            "if poly34_error",
            transformed,
        )
        self.assertIn(
            'poly34_empty.output("POLY.3", '
            '"POLY.3 : Minimum poly extension beyond active : 55nm")',
            transformed,
        )
        self.assertIn(
            'poly34_empty.output("POLY.4", '
            '"POLY.4 : Minimum enclosure of active around gate : 70nm")',
            transformed,
        )

        # The only projection-enclosure calls left are the two historical
        # expressions preserved literally inside the fail-closed CPU branch.
        self.assertEqual(transformed.count(POLY34_LAZY_GATE), 1)
        self.assertNotIn(POLY34_GATE + "\n", transformed)
        self.assertEqual(transformed.count(POLY3), 1)
        self.assertEqual(transformed.count(POLY4), 1)
        false_branch = transformed.split(
            "\nif poly34_clean\n", 1
        )[1].split("\nelse\n", 1)[1].split("\nend", 1)[0]
        self.assertEqual(false_branch, f"  {POLY3}\n  {POLY4}")

    def test_raw_owner_is_the_exact_sole_consumer_guard(self) -> None:
        transformed = generator.add_poly34(POLY34_SOURCE_BLOCK)

        owner = next(
            line
            for line in transformed.splitlines()
            if line.startswith("poly34_raw_owner =")
        )
        self.assertEqual(owner, POLY34_RAW_OWNER)

    def test_clean_branch_has_no_gate_or_cpu_postprocessing_dependency(
        self,
    ) -> None:
        transformed = generator.add_poly34(POLY34_SOURCE_BLOCK)

        clean_branch = transformed.split(
            "\nif poly34_clean\n", 1
        )[1].split("\nelse", 1)[0]
        self.assertLess(
            clean_branch.index('output("POLY.3"'),
            clean_branch.index('output("POLY.4"'),
        )
        self.assertNotIn("gate =", clean_branch)
        self.assertNotIn("(gate", clean_branch)
        self.assertNotIn(".enclosing", clean_branch)
        self.assertNotIn(".polygons", clean_branch)
        self.assertNotIn(".without_area", clean_branch)

    def test_raw_decline_or_exception_cannot_try_legacy(self) -> None:
        transformed = generator.add_poly34(POLY34_SOURCE_BLOCK)

        raw_call_at = transformed.index(
            "poly.cuda_poly34_raw_clean?(active)"
        )
        raw_rescue_at = transformed.index(
            "rescue StandardError =&gt; error", raw_call_at
        )
        raw_reset_at = transformed.index(
            "poly34_raw_clean = false", raw_rescue_at
        )
        lazy_gate_at = transformed.index(POLY34_LAZY_GATE)
        legacy_guard = (
            "if poly34_requested &amp;&amp; !poly34_raw_owner"
        )
        legacy_guard_at = transformed.index(legacy_guard)
        legacy_call_at = transformed.index(
            "poly.cuda_poly34_clean?(active, gate)"
        )
        cpu_branch_at = transformed.index(f"  {POLY3}")

        self.assertLess(raw_call_at, raw_rescue_at)
        self.assertLess(raw_rescue_at, raw_reset_at)
        self.assertLess(raw_reset_at, lazy_gate_at)
        self.assertLess(lazy_gate_at, legacy_guard_at)
        self.assertLess(legacy_guard_at, legacy_call_at)
        self.assertLess(legacy_call_at, cpu_branch_at)
        self.assertEqual(transformed.count(legacy_guard), 1)

    def test_legacy_hook_is_retained_only_for_nonowner_modes(self) -> None:
        transformed = generator.add_poly34(POLY34_SOURCE_BLOCK)

        legacy_block = transformed.split(
            "if poly34_requested &amp;&amp; !poly34_raw_owner\n", 1
        )[1].split("\nend", 1)[0]
        self.assertIn(
            "poly.cuda_poly34_clean?(active, gate)", legacy_block
        )
        self.assertEqual(
            transformed.count("poly.cuda_poly34_clean?(active, gate)"), 1
        )

    def test_rejects_a_changed_rule(self) -> None:
        changed = POLY34_SOURCE_BLOCK.replace("70.nm", "71.nm")

        with self.assertRaisesRegex(
            RuntimeError,
            r"POLY\.3/\.4 transaction: expected one source block, found 0",
        ):
            generator.add_poly34(changed)

    def test_rejects_changed_or_duplicate_gate_anchor(self) -> None:
        changed = POLY34_SOURCE_BLOCK.replace(
            "if need_gate", "if need_gate_now"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"POLY\.3/\.4 raw GATE transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_poly34(changed)

        duplicated = (
            f"{POLY34_GATE}\n{POLY34_GATE}\n"
            f"intervening\n{POLY34_RULE_BLOCK}"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"POLY\.3/\.4 raw GATE transaction: "
            r"expected one source block, found 2",
        ):
            generator.add_poly34(duplicated)

    def test_rejects_duplicate_rule_anchor(self) -> None:
        duplicated = (
            f"{POLY34_GATE}\nintervening\n"
            f"{POLY34_RULE_BLOCK}\n{POLY34_RULE_BLOCK}"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"POLY\.3/\.4 transaction: expected one source block, found 2",
        ):
            generator.add_poly34(duplicated)

    def test_injected_ruby_exception_replaces_only_the_hook(self) -> None:
        transformed = generator.add_poly34(POLY34_SOURCE_BLOCK)
        injected = generator.inject_poly34_ruby_exception(transformed)

        self.assertNotIn(
            "poly.cuda_poly34_raw_clean?(active)", injected
        )
        self.assertIn(
            "poly.cuda_poly34_clean?(active, gate)", injected
        )
        self.assertEqual(
            injected.count('raise("injected POLY34 Ruby exception")'), 1
        )
        self.assertIn("rescue StandardError =&gt; error", injected)
        raise_at = injected.index(
            'raise("injected POLY34 Ruby exception")'
        )
        rescue_at = injected.index(
            "rescue StandardError =&gt; error", raise_at
        )
        reset_at = injected.index(
            "poly34_raw_clean = false", rescue_at
        )
        gate_at = injected.index(POLY34_LAZY_GATE)
        cpu_at = injected.index(f"  {POLY3}")
        self.assertLess(raise_at, rescue_at)
        self.assertLess(rescue_at, reset_at)
        self.assertLess(reset_at, gate_at)
        self.assertLess(gate_at, cpu_at)
        self.assertEqual(injected.count(POLY3), 1)
        self.assertEqual(injected.count(POLY4), 1)

    def test_injected_ruby_exception_rejects_missing_hook(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "POLY.3/.4 injected Ruby exception: "
            "expected one source block, found 0",
        ):
            generator.inject_poly34_ruby_exception(POLY34_SOURCE_BLOCK)


class Active3RawWellsTransformTest(unittest.TestCase):
    def test_exact_transaction_is_deterministic_and_fail_closed(self) -> None:
        source = f"before\n{ACTIVE3_SOURCE_BLOCK}\nafter\n"

        first = generator.add_active3_raw_wells(source)
        second = generator.add_active3_raw_wells(source)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("before\n"))
        self.assertTrue(first.endswith("\nafter\n"))
        self.assertEqual(
            first.count(
                'active3_raw_wells_request = '
                'ENV["KLAYOUT_CUDA_ACTIVE3_RAW_WELLS"].to_s'
            ),
            1,
        )
        self.assertIn(
            "nwell.respond_to?(:cuda_active3_raw_wells_clean?)", first
        )
        self.assertIn(
            "nwell.cuda_active3_raw_wells_clean?(pwell, active)", first
        )
        self.assertIn(
            "rescue StandardError =&gt; active3_raw_wells_error", first
        )
        self.assertLess(
            first.index("active3_raw_wells_clean = false"),
            first.index("unless active3_raw_wells_clean"),
        )

        # Each exact source operation remains once, exclusively in the
        # fail-closed branch.  The certified branch only publishes an empty
        # category and never constructs WELL.
        self.assertEqual(first.count(ACTIVE3_WELL), 1)
        self.assertEqual(first.count(ACTIVE3_RULE), 1)
        clean_branch = first.split(
            "\nif active3_raw_wells_clean\n", 1
        )[1].split("\nelse", 1)[0]
        self.assertNotIn("nwell.or(pwell)", clean_branch)
        self.assertNotIn("well.enclosing", clean_branch)
        self.assertIn("active3_raw_wells_empty.output", clean_branch)

    def test_owner_excludes_every_other_well_consumer(self) -> None:
        transformed = generator.add_active3_raw_wells(ACTIVE3_SOURCE_BLOCK)
        owner = next(
            line
            for line in transformed.splitlines()
            if line.startswith("active3_raw_wells_owner =")
        )

        for required in (
            "active3_raw_wells_requested",
            "DRC",
            "run_active3",
            "!run_well",
            "!run_active4",
            "!(OFFGRID &amp;&amp; run_grid)",
        ):
            self.assertIn(required, owner)

    def test_exception_is_caught_before_literal_cpu_fallback(self) -> None:
        transformed = generator.add_active3_raw_wells(ACTIVE3_SOURCE_BLOCK)

        call_at = transformed.index(
            "nwell.cuda_active3_raw_wells_clean?(pwell, active)"
        )
        rescue_at = transformed.index(
            "rescue StandardError =&gt; active3_raw_wells_error"
        )
        reset_at = transformed.index(
            "active3_raw_wells_clean = false", rescue_at
        )
        union_at = transformed.index(f"  {ACTIVE3_WELL}")
        rule_at = transformed.index(f"  {ACTIVE3_RULE}")
        self.assertLess(call_at, rescue_at)
        self.assertLess(rescue_at, reset_at)
        self.assertLess(reset_at, union_at)
        self.assertLess(reset_at, rule_at)

    def test_rejects_changed_or_duplicate_source_anchors(self) -> None:
        changed_union = ACTIVE3_SOURCE_BLOCK.replace(
            "nwell.or(pwell)", "nwell | pwell"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"ACTIVE\.3 raw-WELL union transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_active3_raw_wells(changed_union)

        changed_rule = ACTIVE3_SOURCE_BLOCK.replace("55.nm", "56.nm")
        with self.assertRaisesRegex(
            RuntimeError,
            r"ACTIVE\.3 raw-WELL output transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_active3_raw_wells(changed_rule)

        duplicated = (
            f"{ACTIVE3_WELL}\n{ACTIVE3_WELL}\nintervening\n{ACTIVE3_RULE}"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"ACTIVE\.3 raw-WELL union transaction: "
            r"expected one source block, found 2",
        ):
            generator.add_active3_raw_wells(duplicated)

    def test_composes_with_poly34_without_weakening_either_fallback(
        self,
    ) -> None:
        source = f"{ACTIVE3_SOURCE_BLOCK}\n{POLY34_SOURCE_BLOCK}"

        transformed = generator.add_poly34(
            generator.add_active3_raw_wells(source)
        )

        self.assertEqual(transformed.count(ACTIVE3_WELL), 1)
        self.assertEqual(transformed.count(ACTIVE3_RULE), 1)
        self.assertEqual(transformed.count(POLY3), 1)
        self.assertEqual(transformed.count(POLY4), 1)
        self.assertEqual(
            transformed.count("BEGIN KLAYOUT CUDA ACTIVE3 RAW WELLS"), 1
        )
        self.assertEqual(
            transformed.count("BEGIN KLAYOUT CUDA POLY34 TRANSACTION"), 1
        )


class Active3ExactWellUnionTransformTest(unittest.TestCase):
    def test_transaction_is_deterministic_exact_and_fail_closed(self) -> None:
        source = f"before\n{ACTIVE3_SOURCE_BLOCK}\nafter\n"
        first = generator.add_active3_well_union(source)
        second = generator.add_active3_well_union(source)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("before\n"))
        self.assertTrue(first.endswith("\nafter\n"))
        self.assertEqual(
            first.count(
                'active3_well_union_request = '
                'ENV["KLAYOUT_CUDA_ACTIVE3_WELL_UNION"].to_s'
            ),
            1,
        )
        self.assertIn(
            "nwell.respond_to?(:cuda_active3_well_union_clean?)", first
        )
        self.assertIn(
            "nwell.cuda_active3_well_union_clean?(pwell, active)", first
        )
        self.assertIn(
            "rescue StandardError =&gt; active3_well_union_error", first
        )
        self.assertEqual(first.count(ACTIVE3_WELL), 1)
        self.assertEqual(first.count(ACTIVE3_RULE), 1)
        clean_branch = first.split(
            "\nif active3_well_union_clean\n", 1
        )[1].split("\nelse", 1)[0]
        self.assertNotIn("nwell.or(pwell)", clean_branch)
        self.assertNotIn("well.enclosing", clean_branch)
        self.assertIn("active3_well_union_empty.output", clean_branch)

    def test_owner_retains_all_other_well_consumers(self) -> None:
        transformed = generator.add_active3_well_union(ACTIVE3_SOURCE_BLOCK)
        owner = next(
            line
            for line in transformed.splitlines()
            if line.startswith("active3_well_union_owner =")
        )
        for required in (
            "active3_well_union_requested",
            "DRC",
            "run_active3",
            "!run_well",
            "!run_active4",
            "!(OFFGRID &amp;&amp; run_grid)",
        ):
            self.assertIn(required, owner)


class Active4ExactWellUnionTransformTest(unittest.TestCase):
    def test_transaction_is_deterministic_exact_and_fail_closed(self) -> None:
        source = f"before\n{ACTIVE4_SOURCE_BLOCK}\nafter\n"
        first = generator.add_active4_well_union(source)
        second = generator.add_active4_well_union(source)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("before\n"))
        self.assertTrue(first.endswith("\nafter\n"))
        self.assertEqual(
            first.count(
                'active4_well_union_request = '
                'ENV["KLAYOUT_CUDA_ACTIVE4_WELL_UNION"].to_s'
            ),
            1,
        )
        self.assertIn(
            "nwell.respond_to?(:cuda_active4_well_union_clean?)", first
        )
        self.assertIn(
            "nwell.cuda_active4_well_union_clean?(pwell, active)", first
        )
        self.assertIn(
            "rescue StandardError =&gt; active4_well_union_error", first
        )
        self.assertEqual(first.count(ACTIVE3_WELL), 1)
        self.assertEqual(first.count(ACTIVE4_RULE), 1)
        clean_branch = first.split(
            "\nif active4_well_union_clean\n", 1
        )[1].split("\nelse", 1)[0]
        self.assertNotIn("nwell.or(pwell)", clean_branch)
        self.assertNotIn("active.not(well)", clean_branch)
        self.assertIn("active4_well_union_empty.output", clean_branch)

    def test_owner_can_share_all_other_well_consumers(self) -> None:
        transformed = generator.add_active4_well_union(ACTIVE4_SOURCE_BLOCK)
        owner = next(
            line
            for line in transformed.splitlines()
            if line.startswith("active4_well_union_owner =")
        )
        for required in (
            "active4_well_union_requested",
            "DRC",
            "run_active4",
        ):
            self.assertIn(required, owner)
        for excluded in (
            "!run_well",
            "!run_active3",
            "!(OFFGRID &amp;&amp; run_grid)",
        ):
            self.assertNotIn(excluded, owner)
        needs_well = next(
            line
            for line in transformed.splitlines()
            if line.startswith("active4_well_union_needs_well =")
        )
        for retained_consumer in (
            "run_well",
            "run_active3",
            "OFFGRID",
            "run_grid",
        ):
            self.assertIn(retained_consumer, needs_well)
        self.assertIn(
            "if !active4_well_union_clean || active4_well_union_needs_well",
            transformed,
        )

    def test_source_drift_fails_closed_at_generation(self) -> None:
        changed_union = ACTIVE4_SOURCE_BLOCK.replace(
            "nwell.or(pwell)", "pwell.or(nwell)"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"ACTIVE\.4 exact WELL-union transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_active4_well_union(changed_union)

        changed_rule = ACTIVE4_SOURCE_BLOCK.replace(
            "active.not(well)", "well.not(active)"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"ACTIVE\.4 exact WELL-union output transaction: "
            r"expected one source block, found 0",
        ):
            generator.add_active4_well_union(changed_rule)


class CombinedM2PolyTransformTest(unittest.TestCase):
    def test_combined_transform_is_deterministic_and_preserves_both(self) -> None:
        source = f"before\n{M2_OWNER_BLOCK}\n{POLY34_SOURCE_BLOCK}\nafter\n"

        first = generator.add_poly34(generator.add_m2_rules(source))
        second = generator.add_poly34(generator.add_m2_rules(source))

        self.assertEqual(first, second)
        self.assertEqual(
            first.count(
                'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s'
            ),
            1,
        )
        self.assertEqual(
            first.count('poly34_request = ENV["KLAYOUT_CUDA_POLY34"].to_s'),
            1,
        )
        self.assertEqual(first.count("m2_rules_empty.output"), 8)
        self.assertEqual(first.count("poly34_empty.output"), 2)
        self.assertEqual(first.count(POLY3), 1)
        self.assertEqual(first.count(POLY4), 1)
        self.assertLess(
            first.index(
                'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s'
            ),
            first.index("BEGIN KLAYOUT CUDA POLY34 TRANSACTION"),
        )


if __name__ == "__main__":
    unittest.main()
