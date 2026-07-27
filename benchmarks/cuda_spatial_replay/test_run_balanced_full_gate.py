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
        self,
        *,
        split_lower: bool,
        split_upper: bool,
        fuse_metal: bool = False,
        split_implant_contact: bool,
        split_active12: bool,
        m1_base_mode_explicit: bool = False,
        m1_5_9_mode_explicit: bool = False,
    ) -> tuple[str, ...]:
        if m1_base_mode_explicit and m1_5_9_mode_explicit:
            owner_prefix = ()
            dynamic_suffix = (
                "antenna_feol",
                "m1_width_space",
                "m1_via_class",
            )
        elif m1_base_mode_explicit:
            owner_prefix = ()
            dynamic_suffix = (
                "m1_via_class",
                "antenna_feol",
                "m1_width_space",
            )
        elif m1_5_9_mode_explicit:
            owner_prefix = self.shell_array("owner_prefix")
            dynamic_suffix = ("antenna_feol", "m1_via_class")
        else:
            owner_prefix = self.shell_array("owner_prefix")
            dynamic_suffix = ("m1_via_class", "antenna_feol")
        return (
            owner_prefix
            + self.shell_array(
                "owner_implant_split"
                if split_implant_contact
                else "owner_implant_joined"
            )
            + (
                self.shell_array("owner_metal_fused")
                if fuse_metal
                else self.shell_array(
                    "owner_upper_split"
                    if split_upper
                    else "owner_upper_joined"
                )
                + self.shell_array(
                    "owner_lower_split"
                    if split_lower
                    else "owner_lower_joined"
                )
            )
            + self.shell_array("owner_suffix_pre")
            + (
                self.shell_array("owner_active_split")
                if split_active12
                else ()
            )
            + self.shell_array("owner_suffix_post")
            + dynamic_suffix
        )

    def test_owner_plans_are_exact_deterministic_and_manifest_complete(
        self,
    ) -> None:
        for split_lower in (False, True):
            for split_upper in (False, True):
                for split_implant_contact in (False, True):
                    for split_active12 in (False, True):
                        exact_plan = (
                            ("m1_width_space",)
                            + (
                                ("implant_contact", "contact")
                                if split_implant_contact
                                else ("implant_contact",)
                            )
                            + (
                                ("antenna_m4_m10", "antenna_m3")
                                if split_upper
                                else ("antenna_m3_m10",)
                            )
                            + (
                                ("antenna_m2", "antenna_m1")
                                if split_lower
                                else ("antenna_m1_m2",)
                            )
                            + ("m2_rules", "m1_enclosure")
                            + (("active12",) if split_active12 else ())
                            + (
                                "via1_upper_active12",
                                "grid",
                                "m1_via_class",
                                "antenna_feol",
                            )
                        )
                        with self.subTest(
                            split_lower=split_lower,
                            split_upper=split_upper,
                            split_implant_contact=split_implant_contact,
                            split_active12=split_active12,
                        ):
                            self._assert_owner_plan(
                                exact_plan,
                                split_lower=split_lower,
                                split_upper=split_upper,
                                split_implant_contact=split_implant_contact,
                                split_active12=split_active12,
                            )

    def test_explicit_m1_modes_reorder_only_the_reserved_tail(self) -> None:
        non_m1 = (
            "implant_contact",
            "contact",
            "antenna_m4_m10",
            "antenna_m3",
            "antenna_m2",
            "antenna_m1",
            "m2_rules",
            "m1_enclosure",
            "active12",
            "via1_upper_active12",
            "grid",
        )
        cases = (
            (
                "base only",
                True,
                False,
                non_m1
                + ("m1_via_class", "antenna_feol", "m1_width_space"),
            ),
            (
                "M1.5-.9 only",
                False,
                True,
                ("m1_width_space",)
                + non_m1
                + ("antenna_feol", "m1_via_class"),
            ),
            (
                "both",
                True,
                True,
                non_m1
                + ("antenna_feol", "m1_width_space", "m1_via_class"),
            ),
        )
        for label, base_explicit, m1_5_9_explicit, exact_plan in cases:
            with self.subTest(label=label):
                self._assert_owner_plan(
                    exact_plan,
                    split_lower=True,
                    split_upper=True,
                    split_implant_contact=True,
                    split_active12=True,
                    m1_base_mode_explicit=base_explicit,
                    m1_5_9_mode_explicit=m1_5_9_explicit,
                )

    def test_explicit_fused_metal_owner_starts_before_reserved_m1_wave(
        self,
    ) -> None:
        exact_plan = (
            "implant_contact",
            "contact",
            "antenna_m1_m4",
            "m2_rules",
            "m1_enclosure",
            "active12",
            "via1_upper_active12",
            "grid",
            "antenna_feol",
            "m1_width_space",
            "m1_via_class",
        )
        self._assert_owner_plan(
            exact_plan,
            split_lower=False,
            split_upper=False,
            fuse_metal=True,
            split_implant_contact=True,
            split_active12=True,
            m1_base_mode_explicit=True,
            m1_5_9_mode_explicit=True,
        )
        self.assertNotIn("delay_fused_metal_owner", self.launcher_text)
        self.assertEqual(exact_plan[:9].count("antenna_m1_m4"), 1)
        self.assertNotIn("m1_width_space", exact_plan[:9])
        self.assertNotIn("m1_via_class", exact_plan[:9])

    def test_fused_metal_owner_replaces_four_split_owners_exactly(
        self,
    ) -> None:
        for split_implant_contact in (False, True):
            for split_active12 in (False, True):
                exact_plan = (
                    ("m1_width_space",)
                    + (
                        ("implant_contact", "contact")
                        if split_implant_contact
                        else ("implant_contact",)
                    )
                    + ("antenna_m1_m4", "m2_rules", "m1_enclosure")
                    + (("active12",) if split_active12 else ())
                    + (
                        "via1_upper_active12",
                        "grid",
                        "m1_via_class",
                        "antenna_feol",
                    )
                )
                with self.subTest(
                    split_implant_contact=split_implant_contact,
                    split_active12=split_active12,
                ):
                    self._assert_owner_plan(
                        exact_plan,
                        split_lower=False,
                        split_upper=False,
                        fuse_metal=True,
                        split_implant_contact=split_implant_contact,
                        split_active12=split_active12,
                    )

    def _assert_owner_plan(
        self,
        exact_plan: tuple[str, ...],
        *,
        split_lower: bool,
        split_upper: bool,
        fuse_metal: bool = False,
        split_implant_contact: bool,
        split_active12: bool,
        m1_base_mode_explicit: bool = False,
        m1_5_9_mode_explicit: bool = False,
    ) -> None:
        plan = self.owner_plan(
            split_lower=split_lower,
            split_upper=split_upper,
            fuse_metal=fuse_metal,
            split_implant_contact=split_implant_contact,
            split_active12=split_active12,
            m1_base_mode_explicit=m1_base_mode_explicit,
            m1_5_9_mode_explicit=m1_5_9_mode_explicit,
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
                    fuse_metal=fuse_metal,
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
        self.assertIn(
            "if ((fuse_metal_antenna)); then\n"
            "  antenna_split_args+=(--fuse-metal)\n"
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
            "--fuse-metal-antenna",
            "--with-antenna-m1-m4",
            "--split-implant-contact",
            "--split-active12",
            "--with-active3-well-union",
            "--with-contact4-active-union",
            "--with-m2-rules",
            "--with-m1-base-width-space",
            "--with-m1-5-9",
            "--with-m2-width-space",
            "--with-implant12",
            "--with-implant15",
            "--with-poly34",
            "--prune-poly2",
        ):
            self.assertIn(option, completed.stderr)

    def test_fused_metal_owner_rejects_antenna_split_modes(self) -> None:
        for split_option in (
            "--split-lower-antenna",
            "--split-upper-antenna",
        ):
            with self.subTest(split_option=split_option):
                completed = self.run_preflight(
                    "--fuse-metal-antenna", split_option
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "--fuse-metal-antenna cannot be combined",
                    completed.stderr,
                )

    def test_antenna_m1_m4_control_and_candidate_share_fused_deck(
        self,
    ) -> None:
        self.assertIn(
            '"KLAYOUT_CUDA_ANTENNA_M1_M4=${antenna_m1_m4}"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_ANTENNA_M1_M4_TELEMETRY=1"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_ANTENNA_DEVICE_WAIT_MS=15000"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_DEVICE_LEASE=1"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_DEVICE_LEASE_WAIT_MS=20000"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_DEVICE_LEASE_TELEMETRY=1"',
            self.launcher_text,
        )
        self.assertIn(
            "CUDA ANTENNA M1-M4 raw transaction: certified-empty",
            self.launcher_text,
        )
        self.assertIn(
            "CUDA ANTENNA M1-M4 raw transaction: full-cpu-fallback",
            self.launcher_text,
        )

        for mode in (
            "--with-antenna-m1-m4",
            "--without-antenna-m1-m4",
        ):
            with self.subTest(mode=mode):
                missing_fuse = self.run_preflight(mode)
                self.assertEqual(missing_fuse.returncode, 2)
                self.assertIn(
                    "requires --fuse-metal-antenna",
                    missing_fuse.stderr,
                )

                accepted = self.run_preflight(
                    "--fuse-metal-antenna", mode
                )
                self.assertEqual(accepted.returncode, 2)
                self.assertNotIn(
                    "requires --fuse-metal-antenna",
                    accepted.stderr,
                )
                self.assertIn("KLayout is not executable", accepted.stderr)

        conflicting = self.run_preflight(
            "--fuse-metal-antenna",
            "--with-antenna-m1-m4",
            "--without-antenna-m1-m4",
        )
        self.assertEqual(conflicting.returncode, 2)
        self.assertIn(
            "choose exactly one antenna M1-M4 runtime mode",
            conflicting.stderr,
        )

    def test_active3_well_union_control_and_candidate_share_one_deck(
        self,
    ) -> None:
        self.assertIn(
            'active3_well_union_generator_args=(--active3-well-union)',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_ACTIVE3_WELL_UNION=${active3_well_union}"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_ACTIVE3_WELL_UNION_TELEMETRY=1"',
            self.launcher_text,
        )
        self.assertIn(
            "CUDA ACTIVE.3 exact resident WELL-union certificate:"
            " outcome=certified-empty",
            self.launcher_text,
        )

    def test_full_gate_rejects_any_via1_or_m1_contact_fallback(
        self,
    ) -> None:
        self.assertIn(
            "a VIA1-stack transaction declined or failed during the full gate",
            self.launcher_text,
        )
        self.assertIn(
            "outcome=(error|fallback|uncertain)",
            self.launcher_text,
        )
        self.assertIn(
            "an M1-contact transaction unexpectedly selected CPU fallback",
            self.launcher_text,
        )
        self.assertIn(
            "CUDA M1 contact transaction: full-cpu-fallback",
            self.launcher_text,
        )

    def test_m1_5_9_control_and_candidate_share_one_deck(self) -> None:
        self.assertIn(
            "m1_5_9_generator_args=(--m1-5-9)",
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_M1_5_9=${m1_5_9}"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_M1_5_9_TELEMETRY=1"',
            self.launcher_text,
        )
        self.assertIn(
            "CUDA M1 exact resident morphology certificate:"
            " outcome=certified-empty",
            self.launcher_text,
        )

    def test_implant15_is_atomic_and_composes_with_implant12(self) -> None:
        self.assertIn(
            "implant15_generator_args=(--implant15)",
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_IMPLANT15=${implant15}"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_IMPLANT15_TELEMETRY=1"',
            self.launcher_text,
        )
        self.assertIn(
            "CUDA IMPLANT.1-.5 raw transaction: certified-empty",
            self.launcher_text,
        )
        self.assertIn(
            "implant12 == 1 && implant15 != 1",
            self.launcher_text,
        )
        self.assertIn(
            "raw IMPLANT.1-.5 success unexpectedly entered "
            "IMPLANT.1/.2 fallback",
            self.launcher_text,
        )

        # Selecting both is deliberate: the raw all-five transaction owns
        # the fast path, while the older IMPLANT.1/.2 transaction remains in
        # its literal outer fallback.
        composed = self.run_preflight(
            "--with-implant12",
            "--with-implant15",
        )
        self.assertEqual(composed.returncode, 2)
        self.assertNotIn("choose exactly one IMPLANT", composed.stderr)
        self.assertIn("KLayout is not executable", composed.stderr)

        legacy_only = self.run_preflight("--with-implant12")
        self.assertEqual(legacy_only.returncode, 2)
        self.assertNotIn("IMPLANT.1-.5", legacy_only.stderr)
        self.assertIn("KLayout is not executable", legacy_only.stderr)

        conflicting = self.run_preflight(
            "--with-implant15",
            "--without-implant15",
        )
        self.assertEqual(conflicting.returncode, 2)
        self.assertIn(
            "choose exactly one IMPLANT.1-.5 runtime mode",
            conflicting.stderr,
        )

    def test_m1_base_width_space_is_exact_and_resource_serialized(self) -> None:
        self.assertIn(
            '"KLAYOUT_CUDA_M1_BASE_WIDTH_SPACE=${m1_base_width_space}"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_CUDA_M1_BASE_WIDTH_SPACE_TELEMETRY=1"',
            self.launcher_text,
        )
        self.assertIn(
            '"KLAYOUT_DEEP_REGION_MULTI_TELEMETRY=1"',
            self.launcher_text,
        )
        self.assertIn(
            "CUDA M1.1/M1.2 raw-union exact live lowering:"
            " outcome=certified-empty",
            self.launcher_text,
        )
        self.assertIn(
            "KLAYOUT_DEEP_REGION_MULTI outcome=raw-cuda-certified-empty",
            self.launcher_text,
        )
        self.assertIn(
            "--serialized-shard m1_width_space\n"
            "    --serialized-shard m1_via_class",
            self.launcher_text,
        )
        self.assertIn(
            '"${serialized_shard_args[@]}"',
            self.launcher_text,
        )
        self.assertIn(
            "owner_prefix=()\n"
            "  owner_suffix_post+=(antenna_feol m1_width_space m1_via_class)",
            self.launcher_text,
        )
        self.assertGreaterEqual(
            self.launcher_text.count(
                "m1_base_width_space >= 0 && m1_5_9 >= 0"
            ),
            2,
        )
        self.assertIn(
            "elif ((m1_base_width_space >= 0)); then",
            self.launcher_text,
        )
        self.assertIn(
            "owner_suffix_post+=(m1_via_class antenna_feol m1_width_space)",
            self.launcher_text,
        )
        self.assertIn(
            "elif ((m1_5_9 >= 0)); then",
            self.launcher_text,
        )
        self.assertIn(
            "owner_suffix_post+=(antenna_feol m1_via_class)",
            self.launcher_text,
        )

    def test_combined_raw_m1_mode_reserves_the_first_launch_wave(self) -> None:
        common = (
            "--split-lower-antenna",
            "--split-upper-antenna",
            "--split-implant-contact",
            "--split-active12",
            "--with-m1-base-width-space",
            "--with-m1-5-9",
        )
        rejected = self.run_preflight(*common, "--jobs", "13")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            "reserve 2 owner(s) behind the first wave; require --jobs 12 or fewer",
            rejected.stderr,
        )

        accepted = self.run_preflight(*common, "--jobs", "12")
        self.assertEqual(accepted.returncode, 2)
        self.assertNotIn("explicit M1 runtime modes reserve", accepted.stderr)
        self.assertIn("KLayout is not executable", accepted.stderr)

        unsplit = (
            "--with-m1-base-width-space",
            "--with-m1-5-9",
        )
        rejected = self.run_preflight(*unsplit, "--jobs", "9")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            "reserve 2 owner(s) behind the first wave; require --jobs 8 or fewer",
            rejected.stderr,
        )

        accepted = self.run_preflight(*unsplit, "--jobs", "8")
        self.assertEqual(accepted.returncode, 2)
        self.assertNotIn("explicit M1 runtime modes reserve", accepted.stderr)
        self.assertIn("KLayout is not executable", accepted.stderr)

    def test_explicit_single_m1_modes_share_a_reserved_schedule(self) -> None:
        modes = (
            "--with-m1-base-width-space",
            "--without-m1-base-width-space",
            "--with-m1-5-9",
            "--without-m1-5-9",
        )
        splits = (
            "--split-lower-antenna",
            "--split-upper-antenna",
            "--split-implant-contact",
            "--split-active12",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                rejected = self.run_preflight(*splits, mode, "--jobs", "14")
                self.assertEqual(rejected.returncode, 2)
                self.assertIn(
                    "reserve 1 owner(s) behind the first wave; "
                    "require --jobs 13 or fewer",
                    rejected.stderr,
                )

                accepted = self.run_preflight(*splits, mode, "--jobs", "13")
                self.assertEqual(accepted.returncode, 2)
                self.assertNotIn("reserve 1 owner(s)", accepted.stderr)
                self.assertIn("KLayout is not executable", accepted.stderr)

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
                ("--fuse-metal-antenna", "--jobs", "10"),
                "selected 9-owner plan",
            ),
            (
                ("--split-lower-antenna", "--jobs", "12"),
                "selected 11-owner plan",
            ),
            (
                ("--split-upper-antenna", "--jobs", "12"),
                "selected 11-owner plan",
            ),
            (
                (
                    "--split-implant-contact",
                    "--split-active12",
                    "--jobs",
                    "13",
                ),
                "selected 12-owner plan",
            ),
        )
        for arguments, message in rejected:
            with self.subTest(arguments=arguments):
                completed = self.run_preflight(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(message, completed.stderr)

        accepted = (
            ("--jobs", "10"),
            ("--fuse-metal-antenna", "--jobs", "9"),
            ("--split-lower-antenna", "--jobs", "11"),
            ("--split-upper-antenna", "--jobs", "11"),
            (
                "--fuse-metal-antenna",
                "--split-implant-contact",
                "--split-active12",
                "--jobs",
                "11",
            ),
            (
                "--split-lower-antenna",
                "--split-upper-antenna",
                "--jobs",
                "12",
            ),
            (
                "--split-lower-antenna",
                "--split-upper-antenna",
                "--split-implant-contact",
                "--split-active12",
                "--jobs",
                "14",
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
