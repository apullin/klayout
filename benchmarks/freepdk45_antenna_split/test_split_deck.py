#!/usr/bin/env python3
"""Focused tests for the fail-closed FreePDK45 antenna deck transform."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from split_deck import (
    ANTENNA_CATEGORIES,
    FEOL_SHARD,
    LOWER_SHARD,
    M1_SHARD,
    M2_SHARD,
    M3_SHARD,
    M4_UPPER_SHARD,
    UPPER_SHARD,
    TransformError,
    antenna_shards,
    metal_owner_manifest,
    split_deck,
)


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


def execute_generated_antenna_section(
    deck: str, shard: str
) -> tuple[list[str], list[str]]:
    """Evaluate generated static guards and return connects/output categories."""

    start = deck.index("#   ANTENNA checks\n")
    end = deck.index("# time spent for the DRC\n", start)
    active = [True]
    connects: list[str] = []
    outputs: list[str] = []

    for line in deck[start:end].splitlines():
        if line.startswith("if "):
            tokens = (
                line.removeprefix("if ")
                .replace("ANTENNA &amp;&amp; ", "")
                .split(" || ")
            )
            condition = any(
                token == "ANTENNA"
                or (
                    token == "run_antenna_checks"
                    and (shard == "all" or shard.startswith("antenna_m"))
                )
                or (
                    token.startswith("run_")
                    and (shard == "all" or token == f"run_{shard}")
                )
                for token in tokens
            )
            active.append(active[-1] and condition)
        elif line == "end":
            if len(active) == 1:
                raise AssertionError("unbalanced generated antenna guard")
            active.pop()
        elif active[-1] and line.startswith("connect("):
            connects.append(line)
        elif active[-1] and '.output("METAL' in line:
            outputs.append(line.split('.output("', 1)[1].split('"', 1)[0])

    if active != [True]:
        raise AssertionError("unterminated generated antenna guard")
    return connects, outputs


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

    def test_optional_upper_split_preserves_all_mode_order_and_one_prefix(self) -> None:
        result = split_deck(source_deck(), split_upper=True)

        self.assertIn('drc_shard == "antenna_m3"', result)
        self.assertIn('drc_shard == "antenna_m4_m10"', result)
        self.assertNotIn('drc_shard == "antenna_m3_m10"', result)
        self.assertIn(
            "if run_antenna_m3 || run_antenna_m4_m10",
            result,
        )
        self.assertIn(
            "\nif run_antenna_m3\n"
            'antenna_check(gate, metal3, 300.0, diode).output("METAL3_ANTENNA"',
            result,
        )
        self.assertIn(
            "\nif run_antenna_m4_m10\n"
            "# build connection of poly+gate to metal4\n"
            "connect(metal3, via3)",
            result,
        )
        self.assertEqual(result.count("connect(metal2, via2)"), 1)
        self.assertEqual(result.count("connect(via2, metal3)"), 1)

        positions = [result.index(f'.output("{name}"') for name in ANTENNA_CATEGORIES]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(
            result.index("if run_antenna_m3"),
            result.index('.output("METAL3_ANTENNA"'),
        )
        self.assertLess(
            result.index("if run_antenna_m4_m10"),
            result.index('.output("METAL4_ANTENNA"'),
        )
        self.assertLess(
            result.index('.output("METAL3_ANTENNA"'),
            result.index("connect(metal3, via3)"),
        )

    def test_optional_lower_split_has_exact_owners_and_m2_prefix_guard(self) -> None:
        result = split_deck(source_deck(), split_lower=True)

        self.assertIn('drc_shard == "antenna_m1"', result)
        self.assertIn('drc_shard == "antenna_m2"', result)
        self.assertNotIn('drc_shard == "antenna_m1_m2"', result)
        self.assertIn(
            "if run_antenna_m2 || run_antenna_m3_m10\n"
            "connect(metal1, via1)\n"
            "connect(via1, metal2)",
            result,
        )
        self.assertEqual(result.count("connect(metal1, via1)"), 1)
        self.assertEqual(result.count("connect(via1, metal2)"), 1)

        expected_outputs = dict(
            metal_owner_manifest(split_lower=True)
        )
        for shard in antenna_shards(split_lower=True):
            connects, outputs = execute_generated_antenna_section(result, shard)
            self.assertEqual(outputs, list(expected_outputs.get(shard, ())))
            if shard == M1_SHARD:
                self.assertNotIn("connect(metal1, via1)", connects)
                self.assertNotIn("connect(via1, metal2)", connects)
            elif shard != FEOL_SHARD:
                self.assertIn("connect(metal1, via1)", connects)
                self.assertIn("connect(via1, metal2)", connects)

        all_connects, all_outputs = execute_generated_antenna_section(
            result, "all"
        )
        self.assertEqual(all_outputs, list(ANTENNA_CATEGORIES))
        self.assertEqual(
            all_connects,
            [
                "connect(gate, poly)",
                "connect(poly, cont)",
                "connect(diode, cont)",
                "connect(cont, metal1)",
                "connect(metal1, via1)",
                "connect(via1, metal2)",
                *[
                    connect
                    for layer in range(3, 11)
                    for connect in (
                        f"connect(metal{layer - 1}, via{layer - 1})",
                        f"connect(via{layer - 1}, metal{layer})",
                    )
                ],
            ],
        )

    def test_lower_and_upper_splits_compose_with_complete_owner_manifest(self) -> None:
        result = split_deck(
            source_deck(), split_lower=True, split_upper=True
        )
        expected_manifest = (
            (M1_SHARD, ("METAL1_ANTENNA",)),
            (M2_SHARD, ("METAL2_ANTENNA",)),
            (M3_SHARD, ("METAL3_ANTENNA",)),
            (M4_UPPER_SHARD, ANTENNA_CATEGORIES[3:]),
        )
        self.assertEqual(
            metal_owner_manifest(split_lower=True, split_upper=True),
            expected_manifest,
        )
        self.assertEqual(
            antenna_shards(split_lower=True, split_upper=True),
            (FEOL_SHARD, M1_SHARD, M2_SHARD, M3_SHARD, M4_UPPER_SHARD),
        )
        self.assertIn(
            "if run_antenna_m2 || run_antenna_m3 || run_antenna_m4_m10",
            result,
        )
        for owner, expected_outputs in expected_manifest:
            _connects, outputs = execute_generated_antenna_section(
                result, owner
            )
            self.assertEqual(outputs, list(expected_outputs))

    def test_owner_manifest_contract_is_complete_in_every_mode(self) -> None:
        expected_shards = {
            (False, False): (FEOL_SHARD, LOWER_SHARD, UPPER_SHARD),
            (True, False): (FEOL_SHARD, M1_SHARD, M2_SHARD, UPPER_SHARD),
            (False, True): (
                FEOL_SHARD,
                LOWER_SHARD,
                M3_SHARD,
                M4_UPPER_SHARD,
            ),
            (True, True): (
                FEOL_SHARD,
                M1_SHARD,
                M2_SHARD,
                M3_SHARD,
                M4_UPPER_SHARD,
            ),
        }
        for (split_lower, split_upper), shards in expected_shards.items():
            with self.subTest(
                split_lower=split_lower, split_upper=split_upper
            ):
                manifest = metal_owner_manifest(
                    split_lower=split_lower, split_upper=split_upper
                )
                categories = [
                    category
                    for _owner, owned_categories in manifest
                    for category in owned_categories
                ]
                self.assertEqual(categories, list(ANTENNA_CATEGORIES))
                self.assertEqual(
                    len({category for category in categories}),
                    len(ANTENNA_CATEGORIES),
                )
                self.assertEqual(
                    antenna_shards(
                        split_lower=split_lower, split_upper=split_upper
                    ),
                    shards,
                )

    def test_split_lower_cli_generation_is_byte_deterministic(self) -> None:
        script = Path(__file__).with_name("split_deck.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.lydrc"
            first = root / "first.lydrc"
            second = root / "second.lydrc"
            source.write_bytes(source_deck().encode("utf-8"))
            for output in (first, second):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--split-lower",
                        "--split-upper",
                        str(source),
                        str(output),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(b"\r", first.read_bytes())

    def test_rejects_a_missing_output_site(self) -> None:
        source = source_deck().replace(
            '.output("METAL7_ANTENNA", "description 7")', ""
        )
        with self.assertRaisesRegex(TransformError, "METAL7_ANTENNA"):
            split_deck(source)


if __name__ == "__main__":
    unittest.main()
