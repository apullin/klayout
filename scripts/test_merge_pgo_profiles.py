#!/usr/bin/env python3
"""Focused fake-tool tests for merge_pgo_profiles.py."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest


MERGER = Path(__file__).with_name("merge_pgo_profiles.py")
VERSION_STDOUT = b"llvm-profdata fake version 22.1.0\n"
RESOLVER_VERSION_STDOUT = b"fake-ldd 1.0\n"


FAKE_PROFDATA = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if "LD_PRELOAD" in os.environ or "LD_LIBRARY_PATH" in os.environ:
    print("loader override leaked into fake llvm-profdata", file=sys.stderr)
    raise SystemExit(86)
log = os.environ.get("FAKE_PROFDATA_LOG")
if log:
    with Path(log).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(args) + "\n")

if args == ["merge", "--version"]:
    sys.stdout.buffer.write(b"llvm-profdata fake version 22.1.0\n")
    raise SystemExit(0)

if not args:
    raise SystemExit(90)

operation = args[0]
if operation == "merge":
    if os.environ.get("FAKE_FAIL_MERGE") == "1":
        print("intentional fake merge failure", file=sys.stderr)
        raise SystemExit(17)
    try:
        output = Path(args[args.index("-o") + 1])
    except (ValueError, IndexError):
        raise SystemExit("missing -o")
    weighted = [item for item in args if item.startswith("--weighted-input=")]
    if weighted:
        total = 0
        for item in weighted:
            weight_text, input_text = item.split("=", 1)[1].split(",", 1)
            count = int(Path(input_text).read_text(encoding="utf-8").split("=", 1)[1])
            total += int(weight_text) * count
    else:
        raw_inputs = [Path(item) for item in args if item.endswith(".profraw")]
        if not raw_inputs:
            raise SystemExit("no raw inputs")
        total = sum(int(path.read_text(encoding="utf-8")) for path in raw_inputs)
    output.write_text(f"count={total}", encoding="utf-8")
    mutation = os.environ.get("FAKE_MUTATE_RAW")
    if mutation and not weighted:
        with Path(mutation).open("a", encoding="utf-8") as stream:
            stream.write("9")
        os.environ.pop("FAKE_MUTATE_RAW", None)
    raise SystemExit(0)

if operation == "show":
    count = int(Path(args[1]).read_text(encoding="utf-8").split("=", 1)[1])
    if os.environ.get("FAKE_ZERO_SUMMARY") == "1":
        count = 0
    print("Instrumentation level: Front-end")
    print("Total functions: 4")
    print(f"Maximum function count: {count}")
    print("Maximum internal block count: 0")
    print(f"Total count: {count}")
    race_output = os.environ.get("FAKE_RACE_OUTPUT")
    if race_output and Path(args[1]).name == "merged.profdata":
        race = Path(race_output)
        race.mkdir()
        (race / "owned-by-racer").write_text("keep", encoding="utf-8")
    raise SystemExit(0)

raise SystemExit(91)
"""


FAKE_RESOLVER = r"""#!/usr/bin/env python3
import os
from pathlib import Path
import sys

if "LD_PRELOAD" in os.environ or "LD_LIBRARY_PATH" in os.environ:
    print("loader override leaked into fake resolver", file=sys.stderr)
    raise SystemExit(87)
if sys.argv[1:] == ["--version"]:
    sys.stdout.buffer.write(b"fake-ldd 1.0\n")
    raise SystemExit(0)
if len(sys.argv) != 2:
    raise SystemExit(88)
library = Path(os.environ["FAKE_LIBLLVM"])
mode = os.environ.get("FAKE_RESOLVER_MODE")
if mode == "missing":
    print("\tlibc.so.6 => /lib/libc.so.6 (0x00007f0011111000)")
    raise SystemExit(0)
print(f"\tlibLLVM.so.22.1 => {library} (0x00007f0012345000)")
if mode == "ambiguous":
    print(f"\tlibLLVM.so.22.1 => {library} (0x00007f0099999000)")
raise SystemExit(0)
"""


class MergePgoProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.tool = self.root / "llvm-profdata-22"
        self.tool.write_text(textwrap.dedent(FAKE_PROFDATA), encoding="utf-8")
        self.tool.chmod(0o755)
        self.tool_sha256 = hashlib.sha256(self.tool.read_bytes()).hexdigest()
        self.version_sha256 = hashlib.sha256(VERSION_STDOUT).hexdigest()
        self.resolver = self.root / "fake-ldd"
        self.resolver.write_text(textwrap.dedent(FAKE_RESOLVER), encoding="utf-8")
        self.resolver.chmod(0o755)
        self.resolver_sha256 = hashlib.sha256(self.resolver.read_bytes()).hexdigest()
        self.resolver_version_sha256 = hashlib.sha256(
            RESOLVER_VERSION_STDOUT
        ).hexdigest()
        self.libllvm = self.root / "libLLVM.so.22.1"
        self.libllvm.write_bytes(b"fake exact libLLVM 22.1")
        self.libllvm_sha256 = hashlib.sha256(self.libllvm.read_bytes()).hexdigest()
        self.log = self.root / "tool.jsonl"
        self.environment = os.environ.copy()
        self.environment["FAKE_PROFDATA_LOG"] = os.fspath(self.log)
        self.environment["FAKE_LIBLLVM"] = os.fspath(self.libllvm)

        self.alpha = self.root / "alpha-raw"
        self.beta = self.root / "beta-raw"
        self.alpha.mkdir()
        self.beta.mkdir()
        (self.alpha / "z.profraw").write_text("2", encoding="utf-8")
        nested = self.alpha / "nested"
        nested.mkdir()
        (nested / "a.profraw").write_text("3", encoding="utf-8")
        (self.beta / "b.profraw").write_text("7", encoding="utf-8")

    def command(
        self,
        output: Path,
        workloads: list[str] | None = None,
        tool: Path | None = None,
        tool_sha256: str | None = None,
        version_sha256: str | None = None,
        resolver: Path | None = None,
        resolver_sha256: str | None = None,
        resolver_version_sha256: str | None = None,
        libllvm_sha256: str | None = None,
    ) -> list[str]:
        selected_workloads = workloads or [
            f"beta={self.beta}:3",
            f"alpha={self.alpha}:2",
        ]
        command = [
            sys.executable,
            os.fspath(MERGER),
            "--llvm-profdata",
            os.fspath(tool or self.tool),
            "--expect-tool-sha256",
            tool_sha256 or self.tool_sha256,
            "--expect-version-sha256",
            version_sha256 or self.version_sha256,
            "--dependency-resolver",
            os.fspath(resolver or self.resolver),
            "--expect-resolver-sha256",
            resolver_sha256 or self.resolver_sha256,
            "--expect-resolver-version-sha256",
            resolver_version_sha256 or self.resolver_version_sha256,
            "--expect-libllvm-sha256",
            libllvm_sha256 or self.libllvm_sha256,
        ]
        for workload in selected_workloads:
            command.extend(["--workload", workload])
        command.extend(["--output-dir", os.fspath(output)])
        return command

    def run_merger(
        self,
        output: Path,
        workloads: list[str] | None = None,
        environment: dict[str, str] | None = None,
        **command_options: object,
    ) -> subprocess.CompletedProcess[str]:
        selected_environment = self.environment.copy()
        if environment:
            selected_environment.update(environment)
        return subprocess.run(
            self.command(output, workloads, **command_options),
            cwd=self.root,
            env=selected_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def read_log(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def assert_no_staging(self, output: Path) -> None:
        self.assertEqual(list(output.parent.glob(f".{output.name}.tmp-*")), [])

    def test_happy_path_is_sorted_weighted_bound_and_read_only(self) -> None:
        output = self.root / "profile-bundle"
        result = self.run_merger(
            output,
            environment={
                "LD_LIBRARY_PATH": "/untrusted/test/path",
                "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libm.so.6",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("published PGO profile bundle", result.stdout)

        profile = output / "merged.profdata"
        manifest_path = output / "manifest.json"
        self.assertEqual(profile.read_text(encoding="utf-8"), "count=31")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "klayout-pgo-profile-bundle-v1")
        self.assertEqual(
            [item["name"] for item in manifest["workloads"]], ["alpha", "beta"]
        )
        self.assertEqual(
            [item["weight"] for item in manifest["workloads"]], [2, 3]
        )
        self.assertEqual(
            [item["aggregate_total_count"] for item in manifest["workloads"]],
            [5, 7],
        )
        self.assertEqual(
            [item["weighted_total_count"] for item in manifest["workloads"]],
            [10, 21],
        )
        self.assertEqual(
            manifest["combined"]["weighting_check"],
            {
                "equal": True,
                "expected_weighted_total_count": 31,
                "observed_total_count": 31,
            },
        )
        alpha_files = manifest["workloads"][0]["raw_inventory"]["files"]
        self.assertEqual(
            [item["path"] for item in alpha_files], ["nested/a.profraw", "z.profraw"]
        )
        for item in alpha_files:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            manifest["workloads"][0]["raw_inventory"]["inventory_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(manifest["tool"]["sha256"], self.tool_sha256)
        self.assertEqual(
            manifest["tool"]["version_stdout_sha256"], self.version_sha256
        )
        self.assertEqual(
            manifest["subprocess_environment_policy"]["unset"],
            ["LD_LIBRARY_PATH", "LD_PRELOAD"],
        )
        dependency = manifest["runtime_dependency_binding"]
        self.assertEqual(dependency["resolver"]["sha256"], self.resolver_sha256)
        self.assertEqual(
            dependency["resolver"]["version_stdout_sha256"],
            self.resolver_version_sha256,
        )
        self.assertEqual(dependency["libllvm"]["sha256"], self.libllvm_sha256)
        self.assertEqual(
            dependency["libllvm"]["resolved_path"], os.fspath(self.libllvm)
        )
        self.assertIn("{staging}/.workloads", manifest["path_tokens"])
        for workload in manifest["workloads"]:
            self.assertIn("{staging}/.workloads", " ".join(workload["merge_command"]))
        self.assertIn(
            "renameat2(RENAME_NOREPLACE)", manifest["publication"]["method"]
        )
        self.assertEqual(
            manifest["combined"]["profile"]["sha256"],
            hashlib.sha256(profile.read_bytes()).hexdigest(),
        )
        weighted = [
            call for call in self.read_log() if any("--weighted-input=" in x for x in call)
        ]
        self.assertEqual(len(weighted), 1)
        weighted_arguments = [
            item for item in weighted[0] if item.startswith("--weighted-input=")
        ]
        self.assertIn("--weighted-input=2,", weighted_arguments[0])
        self.assertIn("alpha.profdata", weighted_arguments[0])
        self.assertIn("--weighted-input=3,", weighted_arguments[1])
        self.assertIn("beta.profdata", weighted_arguments[1])
        merge_calls = [
            call
            for call in self.read_log()
            if call[:1] == ["merge"] and "--version" not in call
        ]
        for merge_call in merge_calls:
            self.assertIn("--failure-mode=any", merge_call)
            self.assertIn("--instr", merge_call)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(profile.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(manifest_path.stat().st_mode), 0o444)
        self.assert_no_staging(output)

    def test_renameat2_noreplace_preserves_racing_destination(self) -> None:
        output = self.root / "racing-output"
        result = self.run_merger(
            output, environment={"FAKE_RACE_OUTPUT": os.fspath(output)}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output path appeared during merge", result.stderr)
        marker = output / "owned-by-racer"
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((output / "merged.profdata").exists())
        self.assert_no_staging(output)

    def test_requires_two_unique_workloads_and_physical_inputs(self) -> None:
        one = self.run_merger(
            self.root / "one", [f"alpha={self.alpha}:1"]
        )
        self.assertNotEqual(one.returncode, 0)
        self.assertIn("at least two", one.stderr)

        duplicate_name = self.run_merger(
            self.root / "duplicate-name",
            [f"alpha={self.alpha}:1", f"alpha={self.beta}:1"],
        )
        self.assertNotEqual(duplicate_name.returncode, 0)
        self.assertIn("names must be unique", duplicate_name.stderr)

        duplicate_directory = self.run_merger(
            self.root / "duplicate-directory",
            [f"alpha={self.alpha}:1", f"beta={self.alpha}:1"],
        )
        self.assertNotEqual(duplicate_directory.returncode, 0)
        self.assertIn("directories must be unique", duplicate_directory.stderr)

        hardlink_dir = self.root / "hardlink-raw"
        hardlink_dir.mkdir()
        os.link(self.alpha / "z.profraw", hardlink_dir / "alias.profraw")
        hardlink = self.run_merger(
            self.root / "hardlink",
            [f"alpha={self.alpha}:1", f"hardlink={hardlink_dir}:1"],
        )
        self.assertNotEqual(hardlink.returncode, 0)
        self.assertIn("reachable through more than one", hardlink.stderr)

    def test_rejects_empty_nonregular_unexpected_and_symlink_inputs(self) -> None:
        cases: list[tuple[str, Path, str]] = []

        empty_directory = self.root / "empty-raw"
        empty_directory.mkdir()
        cases.append(("empty-directory", empty_directory, "no .profraw"))

        empty_file_directory = self.root / "empty-file-raw"
        empty_file_directory.mkdir()
        (empty_file_directory / "empty.profraw").touch()
        cases.append(("empty-file", empty_file_directory, "profile is empty"))

        unexpected_directory = self.root / "unexpected-raw"
        unexpected_directory.mkdir()
        (unexpected_directory / "data.txt").write_text("1", encoding="utf-8")
        cases.append(("unexpected", unexpected_directory, "non-.profraw"))

        fifo_directory = self.root / "fifo-raw"
        fifo_directory.mkdir()
        os.mkfifo(fifo_directory / "pipe.profraw")
        cases.append(("fifo", fifo_directory, "not a regular file"))

        symlink_file_directory = self.root / "symlink-file-raw"
        symlink_file_directory.mkdir()
        (symlink_file_directory / "real.profraw").write_text("1", encoding="utf-8")
        (symlink_file_directory / "link.profraw").symlink_to("real.profraw")
        cases.append(("symlink-file", symlink_file_directory, "is a symlink"))

        symlink_directory = self.root / "symlink-raw"
        symlink_directory.symlink_to(self.alpha, target_is_directory=True)
        cases.append(("symlink-directory", symlink_directory, "traverses a symlink"))

        for name, invalid_directory, message in cases:
            with self.subTest(name=name):
                output = self.root / f"output-{name}"
                result = self.run_merger(
                    output,
                    [f"alpha={self.alpha}:1", f"invalid={invalid_directory}:1"],
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(output.exists())
                self.assert_no_staging(output)

    def test_rejects_wrong_or_symlinked_tool_identity(self) -> None:
        wrong_tool = self.run_merger(
            self.root / "wrong-tool", tool_sha256="0" * 64
        )
        self.assertNotEqual(wrong_tool.returncode, 0)
        self.assertIn("executable SHA-256 mismatch", wrong_tool.stderr)

        wrong_version = self.run_merger(
            self.root / "wrong-version", version_sha256="f" * 64
        )
        self.assertNotEqual(wrong_version.returncode, 0)
        self.assertIn("version-output SHA-256 mismatch", wrong_version.stderr)

        link = self.root / "llvm-profdata-link"
        link.symlink_to(self.tool)
        linked = self.run_merger(self.root / "linked-tool", tool=link)
        self.assertNotEqual(linked.returncode, 0)
        self.assertIn("traverses a symlink", linked.stderr)

    def test_requires_exact_unambiguous_libllvm_resolution(self) -> None:
        wrong_resolver = self.run_merger(
            self.root / "wrong-resolver", resolver_sha256="0" * 64
        )
        self.assertNotEqual(wrong_resolver.returncode, 0)
        self.assertIn("resolver executable SHA-256 mismatch", wrong_resolver.stderr)

        wrong_version = self.run_merger(
            self.root / "wrong-resolver-version",
            resolver_version_sha256="f" * 64,
        )
        self.assertNotEqual(wrong_version.returncode, 0)
        self.assertIn("resolver version-output SHA-256 mismatch", wrong_version.stderr)

        wrong_library = self.run_merger(
            self.root / "wrong-libllvm", libllvm_sha256="a" * 64
        )
        self.assertNotEqual(wrong_library.returncode, 0)
        self.assertIn("libLLVM.so.22.1 SHA-256 mismatch", wrong_library.stderr)

        for mode in ("missing", "ambiguous"):
            with self.subTest(mode=mode):
                output = self.root / f"resolver-{mode}"
                result = self.run_merger(
                    output, environment={"FAKE_RESOLVER_MODE": mode}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("exactly one unambiguous", result.stderr)
                self.assertFalse(output.exists())

    def test_detects_raw_profile_mutation_and_publishes_nothing(self) -> None:
        output = self.root / "mutated"
        mutate = self.alpha / "z.profraw"
        result = self.run_merger(
            output, environment={"FAKE_MUTATE_RAW": os.fspath(mutate)}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutated during merge", result.stderr)
        self.assertFalse(output.exists())
        self.assert_no_staging(output)

    def test_rejects_tool_failure_or_zero_summary_without_partial_output(self) -> None:
        for name, environment, message in (
            ("failure", {"FAKE_FAIL_MERGE": "1"}, "exit status 17"),
            ("zero", {"FAKE_ZERO_SUMMARY": "1"}, "no nonzero counts"),
        ):
            with self.subTest(name=name):
                output = self.root / name
                result = self.run_merger(output, environment=environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(output.exists())
                self.assert_no_staging(output)

    def test_refuses_to_replace_existing_output(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        marker = output / "owned-by-user"
        marker.write_text("keep", encoding="utf-8")
        result = self.run_merger(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_cli_rejects_nonabsolute_paths_and_invalid_weights(self) -> None:
        invalid_cases = (
            ("relative=raw:1", "absolute path"),
            (f"alpha={self.alpha}:0", "between 1"),
            (f"alpha={self.alpha}:1.5", "not an integer"),
        )
        for index, (invalid, message) in enumerate(invalid_cases):
            with self.subTest(invalid=invalid):
                result = self.run_merger(
                    self.root / f"invalid-cli-{index}",
                    [invalid, f"beta={self.beta}:1"],
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
