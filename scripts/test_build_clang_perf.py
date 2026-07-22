#!/usr/bin/env python3
"""Focused fake-tool tests for build-clang-perf.sh."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest


HELPER_SOURCE = Path(__file__).with_name("build-clang-perf.sh")
MANIFEST_NAME = ".klayout-clang-perf-manifest-v1"


FAKE_COMPILER = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import stat
import sys

args = sys.argv[1:]
tool = Path(sys.argv[0]).name
with Path(os.environ["FAKE_COMPILER_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"tool": tool, "args": args}) + "\n")

if "-march=native" in args and "-dM" in args:
    if os.environ.get("FAKE_NATIVE_CPU") == "znver2":
        print("#define __znver2__ 1")
    else:
        print("#define __x86_64__ 1")
    raise SystemExit(0)

targeted = "-march=znver2" in args
compiling = "-c" in args
is_cxx = tool.endswith("++")
failure = os.environ.get("FAKE_FAIL", "")
if compiling and targeted and failure == "znver2-c":
    raise SystemExit(11)
if compiling and targeted and is_cxx and failure == "znver2-cxx":
    raise SystemExit(12)
if not compiling and targeted and failure == "znver2-link":
    raise SystemExit(13)
if not compiling and not targeted and failure == "portable-link":
    raise SystemExit(14)

try:
    output = Path(args[args.index("-o") + 1])
except (ValueError, IndexError):
    raise SystemExit("fake compiler expected -o")
output.parent.mkdir(parents=True, exist_ok=True)
if compiling:
    output.write_bytes(b"fake LLVM bitcode")
else:
    exit_code = 15 if targeted and failure == "znver2-exec" else 0
    output.write_text(
        "#!/usr/bin/env bash\nexit " + str(exit_code) + "\n",
        encoding="utf-8",
    )
    output.chmod(output.stat().st_mode | stat.S_IXUSR)
"""


FAKE_QMAKE = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

with Path(os.environ["FAKE_QMAKE_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\n")
"""


FAKE_BUILD = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
Path(os.environ["FAKE_BUILD_LOG"]).write_text(
    json.dumps(args) + "\n", encoding="utf-8"
)
try:
    wrapper = args[args.index("-qmake") + 1]
except (ValueError, IndexError):
    raise SystemExit("fake build expected -qmake")
raise SystemExit(subprocess.run([wrapper, "-v"], check=False).returncode)
"""


class BuildClangPerfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.root = self.directory / "repo"
        self.script_directory = self.root / "scripts"
        self.tool_directory = self.directory / "tools"
        self.script_directory.mkdir(parents=True)
        self.tool_directory.mkdir()

        self.helper = self.script_directory / "build-clang-perf.sh"
        shutil.copy2(HELPER_SOURCE, self.helper)
        self._make_executable(self.helper)

        self.compiler = self.tool_directory / "fake-clang"
        self.compiler_cxx = self.tool_directory / "fake-clang++"
        self.qmake = self.tool_directory / "fake-qmake6"
        self.build = self.root / "build.sh"
        self._write_executable(self.compiler, FAKE_COMPILER)
        self._write_executable(self.compiler_cxx, FAKE_COMPILER)
        self._write_executable(self.qmake, FAKE_QMAKE)
        self._write_executable(self.build, FAKE_BUILD)

        self.compiler_log = self.directory / "compiler.jsonl"
        self.qmake_log = self.directory / "qmake.jsonl"
        self.build_log = self.directory / "build.json"
        self.output_root = self.directory / "outputs"
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "CC": str(self.compiler),
                "CXX": str(self.compiler_cxx),
                "QMAKE": str(self.qmake),
                "FAKE_COMPILER_LOG": str(self.compiler_log),
                "FAKE_QMAKE_LOG": str(self.qmake_log),
                "FAKE_BUILD_LOG": str(self.build_log),
                "KLAYOUT_PERF_OUTPUT_ROOT": str(self.output_root),
            }
        )
        self.environment.pop("CONDA_PREFIX", None)
        self.environment.pop("KLAYOUT_PERF_PREFIX", None)

    @staticmethod
    def _make_executable(path: Path) -> None:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def _write_executable(self, path: Path, contents: str) -> None:
        path.write_text(textwrap.dedent(contents), encoding="utf-8")
        self._make_executable(path)

    def run_helper(
        self, *arguments: str, environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        selected_environment = self.environment.copy()
        if environment:
            selected_environment.update(environment)
        return subprocess.run(
            [str(self.helper), *arguments],
            cwd=self.root,
            env=selected_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    @staticmethod
    def read_json_lines(path: Path) -> list[object]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def compiler_calls(self) -> list[dict[str, object]]:
        return [dict(item) for item in self.read_json_lines(self.compiler_log)]

    def qmake_arguments(self) -> list[str]:
        records = self.read_json_lines(self.qmake_log)
        self.assertEqual(len(records), 1)
        return list(records[0])

    def build_arguments(self) -> list[str]:
        return list(json.loads(self.build_log.read_text(encoding="utf-8")))

    def profile_build_directory(
        self, profile: str = "portable", output_root: Path | None = None
    ) -> Path:
        return (output_root or self.output_root) / f"build-clang-perf-{profile}"

    def profile_bin_directory(
        self, profile: str = "portable", output_root: Path | None = None
    ) -> Path:
        return (output_root or self.output_root) / f"bin-clang-perf-{profile}"

    def manifest_path(
        self, profile: str = "portable", output_root: Path | None = None
    ) -> Path:
        return self.profile_build_directory(profile, output_root) / MANIFEST_NAME

    def clear_logs(self) -> None:
        for path in (self.compiler_log, self.qmake_log, self.build_log):
            path.unlink(missing_ok=True)

    @staticmethod
    def option_value(arguments: list[str], option: str) -> str:
        return arguments[arguments.index(option) + 1]

    def assert_full_lto_probe(self, calls: list[dict[str, object]]) -> None:
        operational = [
            list(call["args"])
            for call in calls
            if "-march=native" not in list(call["args"])
        ]
        self.assertEqual(len(operational), 3)
        compile_calls = [args for args in operational if "-c" in args]
        link_calls = [args for args in operational if "-c" not in args]
        self.assertEqual(len(compile_calls), 2)
        self.assertEqual(len(link_calls), 1)
        for args in operational:
            self.assertIn("-flto=full", args)
        self.assertIn("-fuse-ld=lld", link_calls[0])

    def test_portable_is_default_and_forwards_arguments_exactly(self) -> None:
        ruby = str(self.directory / "runtime with spaces" / "ruby")
        result = self.run_helper("--", "-ruby", ruby, "-option", "-j7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("requested profile: portable", result.stdout)
        self.assertIn("resolved profile:  portable", result.stdout)
        calls = self.compiler_calls()
        self.assert_full_lto_probe(calls)
        self.assertFalse(
            any("-march=native" in list(call["args"]) for call in calls)
        )
        self.assertFalse(
            any("-march=znver2" in list(call["args"]) for call in calls)
        )

        build_args = self.build_arguments()
        self.assertEqual(build_args[-4:], ["-ruby", ruby, "-option", "-j7"])
        self.assertTrue(
            self.option_value(build_args, "-build").endswith(
                "build-clang-perf-portable"
            )
        )
        self.assertTrue(
            self.option_value(build_args, "-bin").endswith(
                "bin-clang-perf-portable"
            )
        )
        wrapper = Path(self.option_value(build_args, "-qmake"))
        self.assertFalse(wrapper.exists(), "temporary qmake wrapper leaked")

        qmake_args = self.qmake_arguments()
        cflags = next(arg for arg in qmake_args if arg.startswith("QMAKE_CFLAGS+="))
        lflags = next(arg for arg in qmake_args if arg.startswith("QMAKE_LFLAGS+="))
        self.assertIn("-flto=full", cflags)
        self.assertIn("-flto=full", lflags)
        self.assertIn("-fuse-ld=lld", lflags)
        self.assertNotIn("-march", " ".join(qmake_args))

    def test_explicit_znver2_targets_compile_and_lto_link(self) -> None:
        result = self.run_helper(
            "--profile",
            "znver2",
            environment={"FAKE_NATIVE_CPU": "znver2"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resolved profile:  znver2", result.stdout)
        calls = self.compiler_calls()
        native_calls = [
            call for call in calls if "-march=native" in list(call["args"])
        ]
        self.assertEqual(len(native_calls), 1)
        operational = [
            list(call["args"])
            for call in calls
            if "-march=native" not in list(call["args"])
        ]
        self.assertEqual(len(operational), 3)
        for args in operational:
            self.assertIn("-flto=full", args)
            self.assertIn("-march=znver2", args)
            self.assertIn("-mtune=znver2", args)
        link_args = next(args for args in operational if "-c" not in args)
        self.assertIn("-fuse-ld=lld", link_args)

        build_args = self.build_arguments()
        self.assertTrue(
            self.option_value(build_args, "-build").endswith(
                "build-clang-perf-znver2"
            )
        )
        qmake_args = self.qmake_arguments()
        for variable in (
            "QMAKE_CFLAGS+=",
            "QMAKE_CXXFLAGS+=",
            "QMAKE_LFLAGS+=",
        ):
            value = next(arg for arg in qmake_args if arg.startswith(variable))
            self.assertIn("-march=znver2", value)
            self.assertIn("-mtune=znver2", value)

    def test_auto_selects_znver2_on_matching_host(self) -> None:
        result = self.run_helper(
            "--profile=auto", environment={"FAKE_NATIVE_CPU": "znver2"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("requested profile: auto", result.stdout)
        self.assertIn("resolved profile:  znver2", result.stdout)

    def test_host_mismatch_warns_and_uses_portable_directory(self) -> None:
        result = self.run_helper("--profile", "znver2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("does not identify this host as znver2", result.stderr)
        self.assertIn("falling back to portable", result.stderr)
        self.assertIn("resolved profile:  portable", result.stdout)
        self.assertTrue(
            self.option_value(self.build_arguments(), "-build").endswith(
                "build-clang-perf-portable"
            )
        )
        operational = [
            list(call["args"])
            for call in self.compiler_calls()
            if "-march=native" not in list(call["args"])
        ]
        self.assertFalse(any("-march=znver2" in args for args in operational))

    def test_specialized_probe_failures_fall_back_safely(self) -> None:
        cases = (
            ("znver2-cxx", "C++ compilation"),
            ("znver2-link", "full-LTO link"),
            ("znver2-exec", "linked-program execution"),
        )
        for failure, diagnostic in cases:
            with self.subTest(failure=failure):
                nested = BuildClangPerfTests(methodName="runTest")
                nested.setUp()
                try:
                    result = nested.run_helper(
                        "--profile",
                        "auto",
                        environment={
                            "FAKE_NATIVE_CPU": "znver2",
                            "FAKE_FAIL": failure,
                        },
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(diagnostic, result.stderr)
                    self.assertIn("falling back to portable", result.stderr)
                    self.assertIn("resolved profile:  portable", result.stdout)
                    qmake_args = nested.qmake_arguments()
                    self.assertNotIn("-march=znver2", " ".join(qmake_args))
                finally:
                    nested.doCleanups()

    def test_portable_lto_failure_is_fatal_and_does_not_build(self) -> None:
        result = self.run_helper(environment={"FAKE_FAIL": "portable-link"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("portable full-LTO link probe failed", result.stderr)
        self.assertIn("refusing a non-LTO build", result.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())

    def test_manifest_binds_toolchain_configuration_and_allows_resume(self) -> None:
        first = self.run_helper("--", "-noruby")
        self.assertEqual(first.returncode, 0, first.stderr)
        manifest = self.manifest_path()
        self.assertTrue(manifest.is_file())
        manifest_bytes = manifest.read_bytes()
        manifest_text = manifest_bytes.decode("utf-8")
        required_lines = (
            "manifest_schema=klayout-clang-perf-v1",
            "resolved_profile=portable",
            f"cc_path={self.compiler}",
            "cc_sha256=" + hashlib.sha256(self.compiler.read_bytes()).hexdigest(),
            f"cxx_path={self.compiler_cxx}",
            "cxx_sha256="
            + hashlib.sha256(self.compiler_cxx.read_bytes()).hexdigest(),
            f"qmake_path={self.qmake}",
            "qmake_sha256=" + hashlib.sha256(self.qmake.read_bytes()).hexdigest(),
            "qmake_spec=linux-clang",
            "dependency_prefix=",
            "compile_flags.0=-flto=full",
            "link_flags.0=-flto=full",
            "link_flags.1=-fuse-ld=lld",
            "build_arguments.0=-noruby",
        )
        for line in required_lines:
            self.assertIn(line, manifest_text)

        build_artifact = self.profile_build_directory() / "objects" / "db.o"
        bin_artifact = self.profile_bin_directory() / "klayout"
        build_artifact.parent.mkdir()
        bin_artifact.parent.mkdir()
        build_artifact.write_bytes(b"same-config object")
        bin_artifact.write_bytes(b"same-config executable")
        self.clear_logs()

        manifest_mtime = manifest.stat().st_mtime_ns
        dry_run = self.run_helper("--dry-run", "--", "-noruby")
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("manifest status:    matching-resume", dry_run.stdout)
        self.assertEqual(manifest.read_bytes(), manifest_bytes)
        self.assertEqual(manifest.stat().st_mtime_ns, manifest_mtime)
        self.assertEqual(build_artifact.read_bytes(), b"same-config object")
        self.assertEqual(bin_artifact.read_bytes(), b"same-config executable")
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())
        self.clear_logs()

        second = self.run_helper("--", "-noruby")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("manifest status:    matching-resume", second.stdout)
        self.assertEqual(manifest.read_bytes(), manifest_bytes)
        self.assertEqual(build_artifact.read_bytes(), b"same-config object")
        self.assertEqual(bin_artifact.read_bytes(), b"same-config executable")
        self.assertTrue(self.build_log.exists())

    def test_nonempty_build_or_bin_without_manifest_is_rejected(self) -> None:
        for artifact_side in ("build", "bin"):
            with self.subTest(artifact_side=artifact_side):
                output_root = self.directory / f"unmanifested-{artifact_side}"
                if artifact_side == "build":
                    artifact = (
                        self.profile_build_directory(output_root=output_root)
                        / "stale.o"
                    )
                else:
                    artifact = (
                        self.profile_bin_directory(output_root=output_root)
                        / "klayout"
                    )
                artifact.parent.mkdir(parents=True)
                artifact.write_bytes(b"unmanifested artifact")
                self.clear_logs()

                result = self.run_helper(
                    environment={"KLAYOUT_PERF_OUTPUT_ROOT": str(output_root)}
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("have no regular", result.stderr)
                self.assertIn("refusing to mix configurations", result.stderr)
                self.assertFalse(self.build_log.exists())
                self.assertFalse(self.qmake_log.exists())
                self.assertFalse(self.manifest_path(output_root=output_root).exists())
                self.assertEqual(artifact.read_bytes(), b"unmanifested artifact")

    def test_mismatched_manifest_is_rejected_without_rewrite(self) -> None:
        first = self.run_helper()
        self.assertEqual(first.returncode, 0, first.stderr)
        manifest = self.manifest_path()
        manifest.chmod(manifest.stat().st_mode | stat.S_IWUSR)
        mismatched = manifest.read_bytes() + b"unexpected_configuration=1\n"
        manifest.write_bytes(mismatched)
        self.clear_logs()

        second = self.run_helper()
        self.assertEqual(second.returncode, 2)
        self.assertIn("does not exactly match", second.stderr)
        self.assertIn("refusing to mix configurations", second.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())
        self.assertEqual(manifest.read_bytes(), mismatched)

    def test_compiler_file_upgrade_invalidates_existing_manifest(self) -> None:
        first = self.run_helper()
        self.assertEqual(first.returncode, 0, first.stderr)
        original_manifest = self.manifest_path().read_bytes()
        with self.compiler.open("a", encoding="utf-8") as stream:
            stream.write("\n# simulated compiler upgrade\n")
        self.clear_logs()

        second = self.run_helper()
        self.assertEqual(second.returncode, 2)
        self.assertIn("does not exactly match", second.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertEqual(self.manifest_path().read_bytes(), original_manifest)

    def test_dry_run_probes_and_prints_but_invokes_nothing(self) -> None:
        result = self.run_helper(
            "--profile",
            "auto",
            "--dry-run",
            "--",
            "-noruby",
            environment={"FAKE_NATIVE_CPU": "znver2"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resolved profile:  znver2", result.stdout)
        self.assertIn("dry-run (probes passed; build.sh will not run)", result.stdout)
        self.assertIn("qmake invocation additions", result.stdout)
        self.assertIn("QMAKE_LFLAGS", result.stdout)
        self.assertIn("-march=znver2", result.stdout)
        self.assertIn("-noruby", result.stdout)
        self.assertTrue(self.compiler_log.exists())
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())
        self.assertFalse(
            self.output_root.exists(), "dry-run created an artifact directory"
        )

    def test_invalid_and_helper_managed_options_are_rejected(self) -> None:
        invalid = self.run_helper("--profile", "native")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("unsupported profile", invalid.stderr)
        managed = self.run_helper("--", "-build", "/tmp/collision")
        self.assertEqual(managed.returncode, 2)
        self.assertIn("managed by this helper", managed.stderr)
        self.assertFalse(self.build_log.exists())


if __name__ == "__main__":
    unittest.main()
