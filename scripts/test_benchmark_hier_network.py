#!/usr/bin/env python3
"""Focused tests for hierarchical-network benchmark execution modes."""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_hier_network as benchmark  # noqa: E402


def fixture_runtime_provenance(
    *,
    dependency_sha256: str = "database",
    preload_sha256: str | None = None,
    preload_path: str = "/fixture/liballocator.so",
    environment: dict[str, object] | None = None,
) -> dict[str, object]:
    preload_identity = []
    preload_metadata = []
    if preload_sha256 is not None:
        preload_identity.append(
            {
                "role": "ld_preload",
                "loader_position": 1,
                "size_bytes": 17,
                "sha256": preload_sha256,
            }
        )
        preload_metadata.append(
            {
                "resolved_path": preload_path,
                "loader_position": 1,
                "size_bytes": 17,
                "sha256": preload_sha256,
            }
        )
    return {
        "identity": {
            "schema": "klayout-runtime-identity-v2",
            "executable": {
                "role": "executable",
                "size_bytes": 11,
                "sha256": "launcher",
            },
            "elf_dependencies": {
                "files": [
                    {
                        "role": "elf_dependency",
                        "size_bytes": 23,
                        "sha256": dependency_sha256,
                    }
                ],
                "virtual": [],
            },
            "runtime_plugins": [],
            "language_runtime_roots": [
                {
                    "language": "ruby",
                    "roles": ["selected-libruby-stdlib"],
                    "file_count": 2,
                    "total_bytes": 31,
                    "manifest_sha256": "ruby-tree",
                }
            ],
            "language_runtime_policy": "ruby-python-loadable-tree-v1",
            "plugin_search_policy": {
                "id": "fixture-plugin-policy-v1",
                "mode": "module_adjacent",
                "root_roles": [],
            },
            "ld_preload": preload_identity,
            "environment": environment or {},
            "environment_normalization_policy": {
                "id": "fixture-normalization-v1"
            },
            "environment_search_path_state_policy": "values-only-v1",
        },
        "ld_preload": {"ordered_files": preload_metadata},
    }


def fixture_identity() -> dict[str, object]:
    return {
        "version": benchmark.COMPARISON_IDENTITY_VERSION,
        "harness": {
            "script_sha256": "runner",
            "runtime_provenance_script_sha256": "runtime-collector",
            "summary_schema_version": benchmark.SCHEMA_VERSION,
            "comparison_identity_version": (
                benchmark.COMPARISON_IDENTITY_VERSION
            ),
        },
        "suite": {
            "ordered_case_names": ["fixture"],
            "case_count": 1,
            "fingerprint_sha256": benchmark.canonical_json_sha256(["fixture"]),
        },
        "runtime_bundle": {
            "launcher_sha256": "launcher",
            "version": "KLayout fixture",
            "libraries": [
                {"aliases": ["libklayout_db.so"], "sha256": "database"}
            ],
            "fingerprint_sha256": "bundle-fingerprint",
        },
        "preload": {
            "ordered_files": [],
            "fingerprint_sha256": "empty-preload",
        },
        "environment": {"allocator": {}, "threads": {}},
        "host": {
            "node": "fixture",
            "platform": "Linux-fixture",
            "machine": "x86_64",
            "python": "3.fixture",
        },
        "execution_shape": {
            "mode": "isolated_latency",
            "parallel_runs": 1,
            "cpu_affinity_mode": "none",
            "requested_slot_cpu_sets": None,
            "inherited_allowed_cpu_set": "0-7",
            "taskset_sha256": None,
        },
        "cases": {
            "fixture": {
                "kind": "freepdk45",
                "top_cell": "TOP",
                "artifacts": {
                    "input_sha256": "input",
                    "deck_sha256": "deck",
                    "manifests": {
                        "composition_sha256": None,
                        "shard_sha256": None,
                    },
                },
                "argv_templates": [
                    ["$KLAYOUT", "-r", "$DECK", "-rd", "input=$INPUT"]
                ],
                "expected_report_item_count": 0,
                "expected_normalized_report_sha256": "report",
            }
        },
    }


def fixture_summary(
    samples: list[float] | None = None,
) -> dict[str, object]:
    samples = [1.0, 2.0, 3.0] if samples is None else samples
    batches = [
        {
            "batch_index": index,
            "concurrency": 1,
            "run_indices": [index],
            "wall_seconds": sample,
        }
        for index, sample in enumerate(samples, start=1)
    ]
    return {
        "schema_version": benchmark.SCHEMA_VERSION,
        "measurement_mode": "isolated_latency",
        "execution_mode": "isolated_latency",
        "runs_per_case": len(samples),
        "parallel_runs": 1,
        "selected_cases": ["fixture"],
        "comparison_identity": fixture_identity(),
        "cases": [
            {
                "name": "fixture",
                "measurement_mode": "isolated_latency",
                "parallel_runs": 1,
                "n": len(samples),
                "wall_median_seconds": benchmark.statistics.median(samples),
                "per_job_wall_seconds": samples,
                "batch_count": len(samples),
                "batch_wall_samples_seconds": samples,
                "batches": batches,
            }
        ],
    }


class ArgumentTests(unittest.TestCase):
    def parse_error(self, *arguments: str) -> str:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            benchmark.parse_args(list(arguments))
        return stderr.getvalue()

    def test_default_is_isolated_single_run(self) -> None:
        args = benchmark.parse_args(["--cases", "drc_fpd45_small"])
        self.assertEqual(args.runs, 1)
        self.assertEqual(args.parallel_runs, 1)
        self.assertIsNone(args.cpu_sets)
        self.assertEqual(args.comparison_mode, "repeat")
        self.assertEqual(args.treatment, [])

    def test_s5_head_check_is_pinned_and_not_in_the_regular_smoke_group(self) -> None:
        case = benchmark.CASE_BY_NAME["drc_sky130_hg0_s5"]
        self.assertEqual(case.kind, "sky130")
        self.assertEqual(case.top_cell, "hg0_s5_asic")
        self.assertEqual(
            case.input_sha256,
            "03b62132c3f85b66664101fa90faacee6a952cc196cc246633e1c9e298c45911",
        )
        self.assertEqual(
            case.normalized_report_sha256,
            "2c9f660d7b2d7186329c510333083bfe19ab17779fe42d66c936feac0b45fdb4",
        )
        self.assertEqual(
            benchmark.CASE_GROUPS["head_check"], ("drc_sky130_hg0_s5",)
        )
        self.assertEqual(
            benchmark.CASE_GROUPS["acceptance"], ("drc_sky130_hg0_s3",)
        )
        self.assertNotIn("drc_sky130_hg0_s5", benchmark.CASE_GROUPS["regular"])
        self.assertNotIn("drc_sky130_hg0_s3", benchmark.CASE_GROUPS["regular"])
        self.assertIn("drc_sky130_hg0_s5", benchmark.CASE_GROUPS["sky130_full"])
        self.assertEqual(case.deck_relative, "decks/sky130A_mr.drc")
        self.assertEqual(case.expected_report_item_count, 0)
        self.assertIsNone(case.shard_manifest_relative)

    def test_parallel_runs_are_bounded_and_form_full_batches(self) -> None:
        self.assertIn(
            "cannot exceed --runs",
            self.parse_error("--runs", "2", "--parallel-runs", "3"),
        )
        self.assertIn(
            "must be a multiple",
            self.parse_error("--runs", "5", "--parallel-runs", "3"),
        )

    def test_concurrent_mode_rejects_latency_baseline(self) -> None:
        error = self.parse_error(
            "--runs",
            "2",
            "--parallel-runs",
            "2",
            "--baseline",
            "baseline.json",
        )
        self.assertIn("cannot be used with concurrent replicate", error)

    def test_cpu_sets_are_explicit_disjoint_slots(self) -> None:
        args = benchmark.parse_args(
            [
                "--runs",
                "2",
                "--parallel-runs",
                "2",
                "--cpu-sets",
                "0-2;4,6",
            ]
        )
        self.assertEqual(args.cpu_sets, [(0, 1, 2), (4, 6)])

        error = self.parse_error(
            "--runs",
            "2",
            "--parallel-runs",
            "2",
            "--cpu-sets",
            "0-2;2-4",
        )
        self.assertIn("must be disjoint", error)

    def test_comparison_modes_require_precise_cli_intent(self) -> None:
        self.assertIn(
            "requires --baseline",
            self.parse_error("--comparison-mode", "experiment"),
        )
        self.assertIn(
            "at least one --treatment",
            self.parse_error(
                "--baseline",
                "baseline.json",
                "--comparison-mode",
                "experiment",
            ),
        )
        self.assertIn(
            "only valid",
            self.parse_error("--treatment", "runtime-bundle"),
        )
        self.assertIn(
            "requires --baseline",
            self.parse_error("--comparison-mode", "historical"),
        )

        args = benchmark.parse_args(
            [
                "--baseline",
                "baseline.json",
                "--comparison-mode",
                "experiment",
                "--treatment",
                "runtime-bundle",
            ]
        )
        self.assertEqual(args.treatment, ["runtime-bundle"])


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.reports = self.directory / "reports"
        self.logs = self.directory / "logs"
        self.reports.mkdir()
        self.logs.mkdir()
        self.environment = {
            "KLAYOUT_HOME": str(self.directory / "klayout-home"),
        }
        self.case = benchmark.CASE_BY_NAME["drc_fpd45_small"]

    def test_command_wraps_only_explicit_affinity(self) -> None:
        arguments = (
            Path("/bin/klayout"),
            self.case,
            Path("input.gds"),
            Path("deck.lydrc"),
            Path("result.lyrdb"),
        )
        plain = benchmark.command_for_case(*arguments)
        self.assertEqual(plain[0], "/bin/klayout")

        pinned = benchmark.command_for_case(
            *arguments,
            cpu_set=(1, 2, 7),
            taskset=Path("/usr/bin/taskset"),
        )
        self.assertEqual(
            pinned[:3], ["/usr/bin/taskset", "--cpu-list", "1-2,7"]
        )
        self.assertEqual(pinned[3:], plain)

    def test_cpu_set_format_is_stable_and_compact(self) -> None:
        self.assertEqual(benchmark.format_cpu_set((0, 1, 2, 4, 7, 8)), "0-2,4,7-8")

    def test_baseline_loader_requires_every_selected_case(self) -> None:
        path = self.directory / "baseline.json"
        summary = fixture_summary()
        path.write_text(json.dumps(summary), encoding="utf-8")
        loaded = benchmark.load_baseline(path, ["fixture"])
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.timings, {"fixture": 2.0})

        with self.assertRaisesRegex(ValueError, "missing selected case"):
            benchmark.load_baseline(path, ["fixture", "absent"])

        summary["cases"].append(
            copy.deepcopy(summary["cases"][0])
        )
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            benchmark.load_baseline(path, ["fixture"])

    def test_controlled_baseline_recomputes_isolated_finite_samples(self) -> None:
        path = self.directory / "baseline.json"
        mutations = {
            "schema version": lambda value: value.update(schema_version=1),
            "must be isolated_latency": lambda value: value.update(
                measurement_mode="concurrent_replicate_throughput"
            ),
            "n/sample length mismatch": lambda value: value["cases"][0][
                "per_job_wall_seconds"
            ].pop(),
            "non-finite or non-positive sample wall": lambda value: value[
                "cases"
            ][0]["per_job_wall_seconds"].__setitem__(1, float("nan")),
            "wall median does not match": lambda value: value["cases"][0].update(
                wall_median_seconds=2.5
            ),
            "batches are not isolated": lambda value: value["cases"][0][
                "batches"
            ][0].update(concurrency=2),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected):
                summary = fixture_summary()
                mutate(summary)
                path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    benchmark.load_baseline(path, ["fixture"], "repeat")

    def test_experiment_also_requires_strict_schema_v2_baseline(self) -> None:
        path = self.directory / "legacy.json"
        path.write_text(
            json.dumps(
                {
                    "cases": [
                        {"name": "fixture", "wall_median_seconds": 2.0}
                    ]
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "schema version"):
            benchmark.load_baseline(path, ["fixture"], "experiment")

    def test_one_run_baseline_is_valid_context_but_not_statistically_qualified(
        self,
    ) -> None:
        path = self.directory / "one-run.json"
        path.write_text(json.dumps(fixture_summary([2.0])), encoding="utf-8")
        loaded = benchmark.load_baseline(path, ["fixture"], "repeat")
        assert loaded is not None
        assert loaded.statistical_qualification is not None
        self.assertFalse(loaded.statistical_qualification["qualified"])
        self.assertEqual(
            loaded.statistical_qualification["independent_observations"], 1
        )

    def test_controlled_baseline_requires_exact_ordered_suite_context(self) -> None:
        path = self.directory / "wrong-suite.json"
        summary = fixture_summary()
        summary["selected_cases"] = ["other", "fixture"]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ordered suite/case context"):
            benchmark.load_baseline(path, ["fixture"], "repeat")

    def test_legacy_baseline_requires_explicit_historical_mode(self) -> None:
        path = self.directory / "legacy.json"
        path.write_text(
            json.dumps(
                {
                    "cases": [{"name": "fixture", "wall_median_seconds": 2.0}],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "schema version"):
            benchmark.load_baseline(path, ["fixture"], "repeat")
        loaded = benchmark.load_baseline(path, ["fixture"], "historical")
        assert loaded is not None
        with self.assertRaisesRegex(ValueError, "not an identity-exact repeat"):
            benchmark.authorize_comparison(
                "repeat", [], loaded, fixture_identity()
            )
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            benchmark.authorize_comparison(
                "experiment", ["runtime-bundle"], loaded, fixture_identity()
            )
        policy = benchmark.authorize_comparison(
            "historical", [], loaded, fixture_identity()
        )
        self.assertFalse(policy["identity_eligible_for_acceptance"])
        self.assertEqual(policy["mode"], "historical")
        self.assertEqual(
            {item["dimension"] for item in policy["observed_differences"]},
            {"provenance"},
        )

    def test_parallel_batch_overlaps_and_uses_unique_outputs_and_homes(self) -> None:
        barrier = threading.Barrier(3)
        calls: list[dict[str, object]] = []

        def fake_run_sample(
            klayout: Path,
            case: benchmark.Case,
            corpus_root: Path,
            report_path: Path,
            log_path: Path,
            environment: dict[str, str],
            **options: object,
        ) -> dict[str, object]:
            calls.append(
                {
                    "report": report_path,
                    "log": log_path,
                    "home": environment["KLAYOUT_HOME"],
                    **options,
                }
            )
            barrier.wait(timeout=2.0)
            time.sleep(0.02)
            return {"wall_seconds": 0.02}

        with mock.patch.object(benchmark, "run_sample", side_effect=fake_run_sample):
            samples, batches = benchmark.run_case_samples(
                Path("/bin/klayout"),
                self.case,
                self.directory,
                self.reports,
                self.logs,
                self.environment,
                runs=3,
                parallel_runs=3,
                cpu_sets=[(0,), (1,), (2,)],
                taskset=Path("/usr/bin/taskset"),
            )

        self.assertEqual(len(samples), 3)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["concurrency"], 3)
        self.assertEqual(batches[0]["run_indices"], [1, 2, 3])
        self.assertEqual(len({call["report"] for call in calls}), 3)
        self.assertEqual(len({call["log"] for call in calls}), 3)
        self.assertEqual(len({call["home"] for call in calls}), 3)
        self.assertEqual(len({call["working_directory"] for call in calls}), 3)
        self.assertEqual(
            {tuple(call["cpu_set"]) for call in calls}, {(0,), (1,), (2,)}
        )
        self.assertTrue(all(call["collect_child_usage"] is False for call in calls))

    def test_output_directory_lock_is_exclusive(self) -> None:
        output = self.directory / "locked-output"
        with benchmark.exclusive_output_directory_lock(output):
            with self.assertRaisesRegex(ValueError, "already in use"):
                with benchmark.exclusive_output_directory_lock(output):
                    pass

    def test_overwrite_archives_active_summary_before_work(self) -> None:
        output = self.directory / "output"
        output.mkdir()
        summary = output / "benchmark-summary.json"
        summary.write_text('{"stale": true}\n', encoding="utf-8")
        active, archive = benchmark.prepare_output_summary(
            output, self.directory / "baseline.json", overwrite=True
        )
        self.assertEqual(active, summary)
        self.assertFalse(active.exists())
        self.assertIsNotNone(archive)
        assert archive is not None
        self.assertEqual(archive.read_text(encoding="utf-8"), '{"stale": true}\n')

        with self.assertRaisesRegex(ValueError, "must not be the output"):
            benchmark.prepare_output_summary(output, summary, overwrite=True)

    def test_postflight_rehashes_every_pinned_artifact(self) -> None:
        corpus = self.directory / "corpus"
        corpus.mkdir()
        relative_paths = {
            "input": "input.gds",
            "deck": "deck.drc",
            "auxiliary": "composition.json",
            "shard_manifest": "shards.json",
        }
        for role, relative in relative_paths.items():
            (corpus / relative).write_bytes(f"{role}-before".encode("ascii"))
        case = benchmark.Case(
            name="all_artifacts",
            kind="freepdk45",
            input_relative=relative_paths["input"],
            input_sha256=benchmark.sha256_file(corpus / relative_paths["input"]),
            deck_relative=relative_paths["deck"],
            deck_sha256=benchmark.sha256_file(corpus / relative_paths["deck"]),
            top_cell="TOP",
            normalized_report_sha256="report",
            auxiliary_relative=relative_paths["auxiliary"],
            auxiliary_sha256=benchmark.sha256_file(
                corpus / relative_paths["auxiliary"]
            ),
            shard_manifest_relative=relative_paths["shard_manifest"],
            shard_manifest_sha256=benchmark.sha256_file(
                corpus / relative_paths["shard_manifest"]
            ),
        )
        pinned = {case.name: benchmark.preflight_case(case, corpus)}
        for role, relative in relative_paths.items():
            (corpus / relative).write_bytes(f"{role}-after".encode("ascii"))
        with self.assertRaisesRegex(RuntimeError, r"artifact\(s\) changed") as raised:
            benchmark.verify_pinned_artifacts_unchanged([case], corpus, pinned)
        for role in relative_paths:
            self.assertIn(role, str(raised.exception))

    def test_sequential_mode_uses_fresh_state_and_cpu_accounting(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_run_sample(
            klayout: Path,
            case: benchmark.Case,
            corpus_root: Path,
            report_path: Path,
            log_path: Path,
            environment: dict[str, str],
            **options: object,
        ) -> dict[str, object]:
            calls.append({"environment": environment, **options})
            return {"wall_seconds": 0.01}

        with mock.patch.object(benchmark, "run_sample", side_effect=fake_run_sample):
            samples, batches = benchmark.run_case_samples(
                Path("/bin/klayout"),
                self.case,
                self.directory,
                self.reports,
                self.logs,
                self.environment,
                runs=2,
                parallel_runs=1,
                cpu_sets=None,
                taskset=None,
            )

        self.assertEqual(len(samples), 2)
        self.assertEqual([batch["concurrency"] for batch in batches], [1, 1])
        self.assertTrue(all(call["collect_child_usage"] is True for call in calls))
        self.assertEqual(
            len({call["environment"]["KLAYOUT_HOME"] for call in calls}), 2
        )
        self.assertEqual(len({call["working_directory"] for call in calls}), 2)
        self.assertTrue(
            all(call["environment"] is not self.environment for call in calls)
        )
        self.assertTrue(
            all(Path(call["working_directory"]).is_dir() for call in calls)
        )
        self.assertTrue(
            all(
                Path(call["environment"]["KLAYOUT_HOME"]).is_dir()
                for call in calls
            )
        )


class ComparisonIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)

    @staticmethod
    def loaded(identity: object) -> benchmark.LoadedBaseline:
        return benchmark.LoadedBaseline(
            summary={
                "comparison_identity": identity,
                "cases": [
                    {"name": "fixture", "wall_median_seconds": 2.0}
                ],
            },
            timings={"fixture": 2.0},
            metadata={"path": "/fixture/baseline.json", "sha256": "summary"},
        )

    def test_bundle_identity_uses_shared_objects_not_install_paths(self) -> None:
        first = fixture_runtime_provenance(dependency_sha256="db-before")
        first["executable"] = {"resolved_path": "/first/klayout"}
        relocated = copy.deepcopy(first)
        relocated["executable"]["resolved_path"] = "/second/klayout"
        self.assertEqual(
            benchmark.runtime_bundle_identity(first),
            benchmark.runtime_bundle_identity(relocated),
        )

        rebuilt = fixture_runtime_provenance(dependency_sha256="db-after")
        self.assertNotEqual(
            benchmark.runtime_bundle_identity(first)["fingerprint_sha256"],
            benchmark.runtime_bundle_identity(rebuilt)["fingerprint_sha256"],
        )
        changed_policy = copy.deepcopy(first)
        changed_policy["identity"][
            "environment_search_path_state_policy"
        ] = "content-inventory-v2"
        self.assertNotEqual(
            benchmark.runtime_bundle_identity(first)["fingerprint_sha256"],
            benchmark.runtime_bundle_identity(changed_policy)[
                "fingerprint_sha256"
            ],
        )
        changed_language_source = copy.deepcopy(first)
        changed_language_source["identity"]["language_runtime_roots"][0][
            "manifest_sha256"
        ] = "changed-ruby-tree"
        self.assertNotEqual(
            benchmark.runtime_bundle_identity(first)["fingerprint_sha256"],
            benchmark.runtime_bundle_identity(changed_language_source)[
                "fingerprint_sha256"
            ],
        )

        baseline = fixture_identity()
        candidate = copy.deepcopy(baseline)
        baseline["runtime_bundle"] = benchmark.runtime_bundle_identity(first)
        candidate["runtime_bundle"] = benchmark.runtime_bundle_identity(rebuilt)
        differences = benchmark.comparison_identity_differences(
            baseline, candidate
        )
        self.assertEqual(
            {difference["dimension"] for difference in differences},
            {"runtime-bundle"},
        )

    def test_preload_identity_extracts_contents_but_not_paths(self) -> None:
        first = fixture_runtime_provenance(
            preload_sha256="same-preload", preload_path="/first/override.so"
        )
        second = fixture_runtime_provenance(
            preload_sha256="same-preload", preload_path="/second/override.so"
        )
        first_identity, first_metadata = benchmark.preload_identity(first)
        second_identity, _ = benchmark.preload_identity(second)
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(
            first_metadata[0]["resolved_path"], "/first/override.so"
        )

        changed_identity, _ = benchmark.preload_identity(
            fixture_runtime_provenance(preload_sha256="changed-preload")
        )
        self.assertNotEqual(first_identity, changed_identity)

    def test_environment_identity_captures_allocator_and_threads(self) -> None:
        identity = benchmark.relevant_environment_identity(
            fixture_runtime_provenance(
                environment={
                    "MALLOC_CONF": "background_thread:true",
                    "TCMALLOC_RELEASE_RATE": "7",
                    "OMP_NUM_THREADS": "4",
                    "GOMP_CPU_AFFINITY": "0-3",
                    "UNRELATED": "ignored",
                }
            )
        )
        self.assertEqual(
            identity,
            {
                "allocator": {
                    "MALLOC_CONF": "background_thread:true",
                    "TCMALLOC_RELEASE_RATE": "7",
                },
                "threads": {
                    "GOMP_CPU_AFFINITY": "0-3",
                    "OMP_NUM_THREADS": "4",
                },
            },
        )

    def test_argv_templates_are_path_normalized_and_keep_runtime_options(self) -> None:
        case = benchmark.CASE_BY_NAME["drc_sky130_fpu_fp4_hier"]
        templates = benchmark.command_templates_for_case(case, None)
        rendered = " ".join(templates[0])
        self.assertIn("$KLAYOUT", rendered)
        self.assertIn("$INPUT", rendered)
        self.assertIn("$DECK", rendered)
        self.assertIn("$REPORT", rendered)
        self.assertIn("thr=4", rendered)
        self.assertNotIn("/home/", rendered)

    def test_case_identity_uses_declared_expected_item_count(self) -> None:
        case = dataclasses.replace(
            benchmark.CASE_BY_NAME["drc_fpd45_small"],
            expected_report_item_count=17,
        )
        identity = benchmark.case_comparison_identity(
            case,
            {
                "input": {"sha256": "input"},
                "deck": {"sha256": "deck"},
            },
            None,
        )
        self.assertEqual(identity["expected_report_item_count"], 17)

    def test_runner_and_exact_ordered_suite_are_identity_invariants(self) -> None:
        baseline = fixture_identity()
        runner_changed = copy.deepcopy(baseline)
        runner_changed["harness"]["script_sha256"] = "different-runner"
        self.assertEqual(
            {
                difference["dimension"]
                for difference in benchmark.comparison_identity_differences(
                    baseline, runner_changed
                )
            },
            {"provenance"},
        )
        collector_changed = copy.deepcopy(baseline)
        collector_changed["harness"][
            "runtime_provenance_script_sha256"
        ] = "different-runtime-collector"
        self.assertEqual(
            {
                difference["dimension"]
                for difference in benchmark.comparison_identity_differences(
                    baseline, collector_changed
                )
            },
            {"provenance"},
        )

        reordered = copy.deepcopy(baseline)
        reordered["suite"]["ordered_case_names"] = ["other", "fixture"]
        reordered["suite"]["case_count"] = 2
        reordered["cases"]["other"] = copy.deepcopy(
            reordered["cases"]["fixture"]
        )
        self.assertEqual(
            {
                difference["dimension"]
                for difference in benchmark.comparison_identity_differences(
                    baseline, reordered
                )
            },
            {"suite"},
        )

    def test_repeat_and_exact_declared_experiment_policies(self) -> None:
        baseline_identity = fixture_identity()
        baseline = self.loaded(baseline_identity)
        repeat = benchmark.authorize_comparison(
            "repeat", [], baseline, copy.deepcopy(baseline_identity)
        )
        self.assertEqual(repeat["observed_differences"], [])
        self.assertTrue(repeat["identity_eligible_for_acceptance"])

        candidate = copy.deepcopy(baseline_identity)
        candidate["runtime_bundle"]["fingerprint_sha256"] = "rebuilt"
        experiment = benchmark.authorize_comparison(
            "experiment", ["runtime-bundle"], baseline, candidate
        )
        self.assertEqual(experiment["mode"], "experiment")
        self.assertEqual(experiment["declared_treatments"], ["runtime-bundle"])

        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            benchmark.authorize_comparison(
                "experiment", ["runtime-bundle", "argv"], baseline, candidate
            )
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            benchmark.authorize_comparison(
                "experiment", ["argv"], baseline, candidate
            )

    def test_every_requested_identity_dimension_is_distinct(self) -> None:
        baseline = fixture_identity()
        mutations = {
            "preload": lambda value: value["preload"].update(
                fingerprint_sha256="preloaded"
            ),
            "allocator-env": lambda value: value["environment"][
                "allocator"
            ].update(MALLOC_CONF="dirty_decay_ms:0"),
            "thread-env": lambda value: value["environment"]["threads"].update(
                OMP_NUM_THREADS="8"
            ),
            "input": lambda value: value["cases"]["fixture"]["artifacts"].update(
                input_sha256="other-input"
            ),
            "deck": lambda value: value["cases"]["fixture"]["artifacts"].update(
                deck_sha256="other-deck"
            ),
            "manifest": lambda value: value["cases"]["fixture"]["artifacts"][
                "manifests"
            ].update(shard_sha256="other-manifest"),
            "argv": lambda value: value["cases"]["fixture"].update(
                argv_templates=[["$KLAYOUT", "-rd", "thr=8"]]
            ),
            "execution-shape": lambda value: value["execution_shape"].update(
                inherited_allowed_cpu_set="0-3"
            ),
        }
        for expected_dimension, mutate in mutations.items():
            with self.subTest(expected_dimension=expected_dimension):
                candidate = copy.deepcopy(baseline)
                mutate(candidate)
                differences = benchmark.comparison_identity_differences(
                    baseline, candidate
                )
                self.assertEqual(
                    {difference["dimension"] for difference in differences},
                    {expected_dimension},
                )

    def test_correctness_and_host_are_not_experiment_treatments(self) -> None:
        baseline_identity = fixture_identity()
        loaded = self.loaded(baseline_identity)
        for key, mutation in (
            (
                "correctness",
                lambda value: value["cases"]["fixture"].update(
                    expected_normalized_report_sha256="different-report"
                ),
            ),
            ("host", lambda value: value["host"].update(node="other-host")),
        ):
            with self.subTest(key=key):
                candidate = copy.deepcopy(baseline_identity)
                mutation(candidate)
                with self.assertRaisesRegex(ValueError, "does not exactly match"):
                    benchmark.authorize_comparison(
                        "experiment", ["runtime-bundle"], loaded, candidate
                    )

    def test_historical_output_is_visibly_labeled(self) -> None:
        rendered = benchmark.format_comparison(7.0, 10.0, "historical")
        self.assertTrue(rendered.startswith("HISTORICAL CONTEXT:"))

    def test_statistical_qualification_counts_independent_observations(self) -> None:
        isolated = benchmark.statistical_qualification(
            measurement_mode="isolated_latency",
            sample_count=3,
            batch_count=3,
        )
        self.assertTrue(isolated["qualified"])
        self.assertEqual(isolated["independent_observations"], 3)

        one_concurrent_batch = benchmark.statistical_qualification(
            measurement_mode="concurrent_replicate_throughput",
            sample_count=5,
            batch_count=1,
        )
        self.assertFalse(one_concurrent_batch["qualified"])
        self.assertEqual(one_concurrent_batch["independent_observations"], 1)

        repeated_concurrent_batches = benchmark.statistical_qualification(
            measurement_mode="concurrent_replicate_throughput",
            sample_count=15,
            batch_count=3,
        )
        self.assertTrue(repeated_concurrent_batches["qualified"])


class SummaryAssemblyTests(unittest.TestCase):
    def test_run_benchmark_assembles_and_postverifies_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            output_dir.mkdir()
            executable = root / "klayout"
            executable.write_bytes(b"fixture executable")
            executable.chmod(0o755)
            case = benchmark.CASE_BY_NAME["drc_fpd45_small"]
            args = benchmark.argparse.Namespace(
                baseline=None,
                corpus_root=str(root / "corpus"),
                klayout=str(executable),
                parallel_runs=1,
                cpu_sets=None,
                selected_cases=[case],
                comparison_mode="repeat",
                treatment=[],
                runs=3,
                overwrite=False,
            )
            pinned = {
                "input": {"sha256": case.input_sha256},
                "deck": {"sha256": case.deck_sha256},
            }
            samples = [
                {
                    "wall_seconds": wall,
                    "peak_rss_kb": 1000 + index,
                    "average_cpu_cores": 1.5,
                }
                for index, wall in enumerate((1.0, 1.1, 0.9), start=1)
            ]
            batches = [
                {
                    "batch_index": index,
                    "concurrency": 1,
                    "run_indices": [index],
                    "wall_seconds": sample["wall_seconds"],
                }
                for index, sample in enumerate(samples, start=1)
            ]
            runtime = fixture_runtime_provenance()

            with (
                mock.patch.object(
                    benchmark,
                    "preflight_case",
                    return_value=pinned,
                ),
                mock.patch.object(
                    benchmark,
                    "verify_pinned_artifacts_unchanged",
                ) as verify_artifacts,
                mock.patch.object(
                    benchmark,
                    "collect_benchmark_runtime_provenance",
                    side_effect=[copy.deepcopy(runtime), copy.deepcopy(runtime)],
                ) as collect_runtime,
                mock.patch.object(
                    benchmark,
                    "run_case_samples",
                    return_value=(samples, batches),
                ),
                mock.patch.object(
                    benchmark.subprocess,
                    "run",
                    return_value=benchmark.subprocess.CompletedProcess(
                        [str(executable), "-v"], 0, "KLayout fixture\n", ""
                    ),
                ),
            ):
                result = benchmark.run_benchmark(args, output_dir)

            self.assertEqual(result, 0)
            self.assertEqual(collect_runtime.call_count, 2)
            verify_artifacts.assert_called_once()
            summary = json.loads(
                (output_dir / "benchmark-summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["comparison_identity"]["version"],
                benchmark.COMPARISON_IDENTITY_VERSION,
            )
            self.assertEqual(summary["runs_per_case"], 3)
            self.assertTrue(
                summary["comparison"]["statistical_qualification"]["qualified"]
            )
            self.assertEqual(summary["cases"][0]["peak_rss_kb_max"], 1003)
            self.assertIn("runtime_provenance", summary)


if __name__ == "__main__":
    unittest.main()
