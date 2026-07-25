#!/usr/bin/env python3
"""Static contract tests for the balanced full-launch owner plan."""

from __future__ import annotations

from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LAUNCHER = HERE / "run_balanced_full_gate.sh"
sys.path.insert(0, str(ROOT / "benchmarks" / "freepdk45_antenna_split"))

from split_deck import antenna_shards  # noqa: E402


class BalancedFullGateStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher_text = LAUNCHER.read_text(encoding="utf-8")

    def shell_array(self, name: str) -> tuple[str, ...]:
        match = re.search(
            rf"^{re.escape(name)}=\((.*?)\)$",
            self.launcher_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match, f"missing shell array {name}")
        return tuple(shlex.split(match.group(1), comments=True, posix=True))

    def owner_plan(
        self, *, split_lower: bool, split_upper: bool
    ) -> tuple[str, ...]:
        return (
            self.shell_array("owner_prefix")
            + self.shell_array(
                "owner_upper_split" if split_upper else "owner_upper_joined"
            )
            + self.shell_array(
                "owner_lower_split" if split_lower else "owner_lower_joined"
            )
            + self.shell_array("owner_suffix")
        )

    def test_owner_plans_are_exact_deterministic_and_manifest_complete(
        self,
    ) -> None:
        expected = {
            (False, False): (
                "m1_width_space",
                "implant_contact",
                "antenna_m3_m10",
                "antenna_m1_m2",
                "m2_rules",
                "m1_enclosure",
                "via1_upper_active12",
                "grid",
                "m1_via_class",
                "antenna_feol",
            ),
            (True, False): (
                "m1_width_space",
                "implant_contact",
                "antenna_m3_m10",
                "antenna_m2",
                "antenna_m1",
                "m2_rules",
                "m1_enclosure",
                "via1_upper_active12",
                "grid",
                "m1_via_class",
                "antenna_feol",
            ),
            (False, True): (
                "m1_width_space",
                "implant_contact",
                "antenna_m4_m10",
                "antenna_m3",
                "antenna_m1_m2",
                "m2_rules",
                "m1_enclosure",
                "via1_upper_active12",
                "grid",
                "m1_via_class",
                "antenna_feol",
            ),
            (True, True): (
                "m1_width_space",
                "implant_contact",
                "antenna_m4_m10",
                "antenna_m3",
                "antenna_m2",
                "antenna_m1",
                "m2_rules",
                "m1_enclosure",
                "via1_upper_active12",
                "grid",
                "m1_via_class",
                "antenna_feol",
            ),
        }
        for (split_lower, split_upper), exact_plan in expected.items():
            with self.subTest(
                split_lower=split_lower, split_upper=split_upper
            ):
                plan = self.owner_plan(
                    split_lower=split_lower, split_upper=split_upper
                )
                self.assertEqual(plan, exact_plan)
                self.assertEqual(len(plan), len(set(plan)))
                selected_antenna = {
                    owner for owner in plan if owner.startswith("antenna_")
                }
                self.assertEqual(
                    selected_antenna,
                    set(
                        antenna_shards(
                            split_lower=split_lower,
                            split_upper=split_upper,
                        )
                    ),
                )

    def test_splitter_arguments_compose_without_replacing_cuda_options(
        self,
    ) -> None:
        self.assertIn(
            "if ((split_lower_antenna)); then\n"
            "  antenna_split_args+=(--split-lower)\n"
            "fi",
            self.launcher_text,
        )
        self.assertIn(
            "if ((split_upper_antenna)); then\n"
            "  antenna_split_args+=(--split-upper)\n"
            "fi",
            self.launcher_text,
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--split-lower-antenna",
            "--split-upper-antenna",
            "--with-contact4-active-union",
            "--with-m2-rules",
            "--with-m2-width-space",
            "--with-implant12",
            "--with-poly34",
            "--prune-poly2",
        ):
            self.assertIn(option, completed.stderr)

    def run_preflight(self, *options: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(LAUNCHER),
                "--klayout",
                "/balanced-owner-plan-test/missing-klayout",
                "--backend",
                "/balanced-owner-plan-test/missing-backend",
                "--source-deck",
                "/balanced-owner-plan-test/missing-source",
                "--manifest",
                "/balanced-owner-plan-test/missing-manifest",
                "--input",
                "/balanced-owner-plan-test/missing-input",
                "--top-cell",
                "TOP",
                "--reference",
                "/balanced-owner-plan-test/missing-reference",
                *options,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_jobs_never_exceed_selected_owner_manifest(self) -> None:
        rejected = (
            (("--jobs", "11"), "selected 10-owner plan"),
            (
                ("--split-lower-antenna", "--jobs", "12"),
                "selected 11-owner plan",
            ),
            (
                ("--split-upper-antenna", "--jobs", "12"),
                "selected 11-owner plan",
            ),
        )
        for arguments, message in rejected:
            with self.subTest(arguments=arguments):
                completed = self.run_preflight(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr)

        accepted = (
            ("--jobs", "10"),
            ("--split-lower-antenna", "--jobs", "11"),
            ("--split-upper-antenna", "--jobs", "11"),
            (
                "--split-lower-antenna",
                "--split-upper-antenna",
                "--jobs",
                "12",
            ),
        )
        for arguments in accepted:
            with self.subTest(arguments=arguments):
                completed = self.run_preflight(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertNotIn("exceeds selected", completed.stderr)
                self.assertIn("KLayout is not executable", completed.stderr)


if __name__ == "__main__":
    unittest.main()
