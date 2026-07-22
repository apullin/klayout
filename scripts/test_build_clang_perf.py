#!/usr/bin/env python3
"""Focused fake-tool tests for build-clang-perf.sh."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tempfile
import textwrap
import unittest


HELPER_SOURCE = Path(__file__).with_name("build-clang-perf.sh")
MANIFEST_NAME = ".klayout-clang-perf-manifest-v2"
PGO_STATE_NAME = ".klayout-clang-perf-pgo-state-v1"
PGO_LOCK_NAME = ".klayout-clang-perf-pgo-lock-v1"


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
pgo_generate = any(arg.startswith("-fprofile-generate=") for arg in args)
pgo_use = any(arg.startswith("-fprofile-use=") for arg in args)
failure = os.environ.get("FAKE_FAIL", "")
if compiling and targeted and failure == "znver2-c":
    raise SystemExit(11)
if compiling and targeted and is_cxx and failure == "znver2-cxx":
    raise SystemExit(12)
if not compiling and targeted and failure == "znver2-link":
    raise SystemExit(13)
if not compiling and not targeted and failure == "portable-link":
    raise SystemExit(14)
if compiling and (pgo_generate or pgo_use) and failure == "pgo-c":
    raise SystemExit(16)
if compiling and is_cxx and (pgo_generate or pgo_use) and failure == "pgo-cxx":
    raise SystemExit(17)
if not compiling and (pgo_generate or pgo_use) and failure == "pgo-link":
    raise SystemExit(18)

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
import signal
import stat
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
qmake_result = subprocess.run([wrapper, "-v"], check=False).returncode
if qmake_result:
    raise SystemExit(qmake_result)

if os.environ.get("FAKE_BUILD_FAIL"):
    raise SystemExit(23)

mutation = os.environ.get("FAKE_BUILD_MUTATE", "")
if mutation == "compiler":
    target = Path(os.environ["CC"])
elif mutation == "build-driver":
    target = Path(sys.argv[0])
elif mutation == "profile-source":
    target = Path(os.environ["PGO_TEST_PROFILE"])
elif mutation == "staged-profile":
    build_directory = Path(args[args.index("-build") + 1])
    matches = list(build_directory.glob("pgo-input-*.profdata"))
    if len(matches) != 1:
        raise SystemExit("fake build expected one staged PGO profile")
    target = matches[0]
elif mutation:
    raise SystemExit("unknown fake mutation: " + mutation)
else:
    target = None
if target is not None:
    target.chmod(target.stat().st_mode | stat.S_IWUSR)
    with target.open("ab") as stream:
        stream.write(b"\nmutated during build\n")

if os.environ.get("FAKE_BUILD_KILL_PARENT"):
    os.kill(os.getppid(), signal.SIGKILL)

raise SystemExit(0)
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

        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.name", "KLayout perf test"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "klayout-perf@example.invalid"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "fake source"],
            cwd=self.root,
            check=True,
        )

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
        self,
        profile: str = "portable",
        output_root: Path | None = None,
        *,
        pgo: str = "none",
        profile_hash: str = "",
        manifest_hash: str = "",
    ) -> Path:
        suffix = profile
        if pgo == "generate":
            suffix += "-pgo-generate"
        elif pgo == "control":
            suffix += "-pgo-control"
        elif pgo == "use":
            suffix += f"-pgo-use-{profile_hash[:16]}-manifest-{manifest_hash[:16]}"
        if pgo != "none":
            commit, tree = self.source_identity()
            suffix += f"-commit-{commit[:16]}-tree-{tree[:16]}"
        return (output_root or self.output_root) / f"build-clang-perf-{suffix}"

    def profile_bin_directory(
        self,
        profile: str = "portable",
        output_root: Path | None = None,
        *,
        pgo: str = "none",
        profile_hash: str = "",
        manifest_hash: str = "",
    ) -> Path:
        suffix = profile
        if pgo == "generate":
            suffix += "-pgo-generate"
        elif pgo == "control":
            suffix += "-pgo-control"
        elif pgo == "use":
            suffix += f"-pgo-use-{profile_hash[:16]}-manifest-{manifest_hash[:16]}"
        if pgo != "none":
            commit, tree = self.source_identity()
            suffix += f"-commit-{commit[:16]}-tree-{tree[:16]}"
        return (output_root or self.output_root) / f"bin-clang-perf-{suffix}"

    def manifest_path(
        self,
        profile: str = "portable",
        output_root: Path | None = None,
        *,
        pgo: str = "none",
        profile_hash: str = "",
        manifest_hash: str = "",
    ) -> Path:
        return (
            self.profile_build_directory(
                profile,
                output_root,
                pgo=pgo,
                profile_hash=profile_hash,
                manifest_hash=manifest_hash,
            )
            / MANIFEST_NAME
        )

    def make_immutable_profile(
        self, name: str = "training.profdata", contents: bytes = b"fake profdata"
    ) -> Path:
        path = self.directory / name
        path.write_bytes(contents)
        path.chmod(0o444)
        return path

    def make_immutable_profile_manifest(
        self,
        name: str = "training-profile-manifest.json",
        contents: bytes = b'{"schema":"fake-pgo-profile-v1"}\n',
    ) -> Path:
        path = self.directory / name
        path.write_bytes(contents)
        path.chmod(0o444)
        return path

    def source_identity(self) -> tuple[str, str]:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=self.root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        return commit, tree

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
            "manifest_schema=klayout-clang-perf-v2",
            "resolved_profile=portable",
            "pgo_mode=none",
            "pgo_profile_source_path=",
            "pgo_profile_sha256=",
            "source_commit=",
            "source_tree=",
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
        manifest.chmod(0o444)
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

    def test_pgo_generate_instruments_compiles_and_links_atomically(self) -> None:
        result = self.run_helper("--pgo", "generate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PGO mode:          generate", result.stdout)

        pgo_calls = [
            list(call["args"])
            for call in self.compiler_calls()
            if any(
                str(arg).startswith("-fprofile-generate=")
                for arg in list(call["args"])
            )
        ]
        self.assertEqual(len(pgo_calls), 3)
        for arguments in pgo_calls:
            self.assertIn("-fprofile-update=atomic", arguments)
            self.assertIn("-flto=full", arguments)

        build_directory = self.profile_build_directory(pgo="generate")
        bin_directory = self.profile_bin_directory(pgo="generate")
        self.assertEqual(
            Path(self.option_value(self.build_arguments(), "-build")),
            build_directory,
        )
        self.assertEqual(
            Path(self.option_value(self.build_arguments(), "-bin")), bin_directory
        )
        default_profiles = build_directory / "pgo-default-profraw-discard"
        self.assertTrue(default_profiles.is_dir())
        commit, tree = self.source_identity()
        self.assertIn(
            f"-commit-{commit[:16]}-tree-{tree[:16]}", build_directory.name
        )
        lifecycle = build_directory / PGO_STATE_NAME
        lifecycle_text = lifecycle.read_text(encoding="utf-8")
        self.assertIn(
            "lifecycle_schema=klayout-clang-perf-pgo-lifecycle-v1",
            lifecycle_text,
        )
        self.assertIn("status=complete", lifecycle_text)
        self.assertIn("manifest_sha256=", lifecycle_text)
        self.assertEqual(lifecycle.stat().st_mode & 0o222, 0)
        self.assertFalse((build_directory / PGO_LOCK_NAME).exists())

        qmake_args = self.qmake_arguments()
        for variable in (
            "QMAKE_CFLAGS+=",
            "QMAKE_CXXFLAGS+=",
            "QMAKE_LFLAGS+=",
        ):
            value = next(arg for arg in qmake_args if arg.startswith(variable))
            self.assertIn(
                f"-fprofile-generate={default_profiles}", value
            )
            self.assertIn("-fprofile-update=atomic", value)

        manifest_text = self.manifest_path(pgo="generate").read_text(
            encoding="utf-8"
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=self.root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        self.assertIn("pgo_mode=generate", manifest_text)
        self.assertIn(f"source_commit={commit}", manifest_text)
        self.assertIn(f"source_tree={tree}", manifest_text)
        self.assertIn("-fprofile-update=atomic", manifest_text)

    def test_pgo_failed_build_leaves_refusing_lifecycle(self) -> None:
        failed = self.run_helper(
            "--pgo", "generate", environment={"FAKE_BUILD_FAIL": "1"}
        )
        self.assertEqual(failed.returncode, 23, failed.stderr)
        build_directory = self.profile_build_directory(pgo="generate")
        state = build_directory / PGO_STATE_NAME
        lock = build_directory / PGO_LOCK_NAME
        self.assertIn("status=building", state.read_text(encoding="utf-8"))
        self.assertEqual(state.stat().st_mode & 0o222, 0)
        self.assertTrue(lock.is_dir())

        self.clear_logs()
        resumed = self.run_helper("--pgo", "generate")
        self.assertEqual(resumed.returncode, 2)
        self.assertIn("lifecycle lock exists", resumed.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())

    def test_pgo_stale_or_concurrent_lock_refuses_complete_artifacts(self) -> None:
        first = self.run_helper("--pgo", "generate")
        self.assertEqual(first.returncode, 0, first.stderr)
        build_directory = self.profile_build_directory(pgo="generate")
        (build_directory / PGO_LOCK_NAME).mkdir()
        self.clear_logs()

        second = self.run_helper("--pgo", "generate")
        self.assertEqual(second.returncode, 2)
        self.assertIn("lifecycle lock exists", second.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())

    def test_pgo_sigkill_leaves_refusing_lifecycle(self) -> None:
        killed = self.run_helper(
            "--pgo", "generate", environment={"FAKE_BUILD_KILL_PARENT": "1"}
        )
        self.assertEqual(killed.returncode, -signal.SIGKILL)
        build_arguments = self.build_arguments()
        leaked_probe = Path(self.option_value(build_arguments, "-qmake")).parent
        self.addCleanup(shutil.rmtree, leaked_probe, ignore_errors=True)
        build_directory = self.profile_build_directory(pgo="generate")
        self.assertIn(
            "status=building",
            (build_directory / PGO_STATE_NAME).read_text(encoding="utf-8"),
        )
        self.assertTrue((build_directory / PGO_LOCK_NAME).is_dir())

        self.clear_logs()
        resumed = self.run_helper("--pgo", "generate")
        self.assertEqual(resumed.returncode, 2)
        self.assertIn("lifecycle lock exists", resumed.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())

    def test_pgo_tool_mutation_cannot_publish_complete(self) -> None:
        result = self.run_helper(
            "--pgo", "generate", environment={"FAKE_BUILD_MUTATE": "compiler"}
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("C compiler changed after configuration", result.stderr)
        build_directory = self.profile_build_directory(pgo="generate")
        self.assertIn(
            "status=building",
            (build_directory / PGO_STATE_NAME).read_text(encoding="utf-8"),
        )
        self.assertTrue((build_directory / PGO_LOCK_NAME).is_dir())

    def test_pgo_staged_profile_mutation_cannot_publish_complete(self) -> None:
        profile = self.make_immutable_profile(contents=b"trusted profile")
        profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
        profile_manifest = self.make_immutable_profile_manifest()
        manifest_hash = hashlib.sha256(profile_manifest.read_bytes()).hexdigest()
        result = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(profile),
            "--pgo-profile-manifest",
            str(profile_manifest),
            environment={"FAKE_BUILD_MUTATE": "staged-profile"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("PGO profile must be immutable", result.stderr)
        build_directory = self.profile_build_directory(
            pgo="use", profile_hash=profile_hash, manifest_hash=manifest_hash
        )
        self.assertIn(
            "status=building",
            (build_directory / PGO_STATE_NAME).read_text(encoding="utf-8"),
        )
        self.assertTrue((build_directory / PGO_LOCK_NAME).is_dir())

    def test_pgo_use_stages_hashed_immutable_profile(self) -> None:
        profile = self.make_immutable_profile(contents=b"profile generation one")
        profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
        profile_manifest = self.make_immutable_profile_manifest()
        manifest_hash = hashlib.sha256(profile_manifest.read_bytes()).hexdigest()
        result = self.run_helper(
            "--profile",
            "znver2",
            "--pgo=use",
            "--pgo-profile",
            str(profile),
            "--pgo-profile-manifest",
            str(profile_manifest),
            environment={"FAKE_NATIVE_CPU": "znver2"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PGO mode:          use", result.stdout)
        self.assertIn(f"PGO profile SHA:    {profile_hash}", result.stdout)
        self.assertIn(
            "PGO probe scope:    profile format/toolchain only",
            result.stdout,
        )

        build_directory = self.profile_build_directory(
            "znver2",
            pgo="use",
            profile_hash=profile_hash,
            manifest_hash=manifest_hash,
        )
        staged = build_directory / f"pgo-input-{profile_hash}.profdata"
        staged_manifest = (
            build_directory / f"pgo-profile-manifest-{manifest_hash}.json"
        )
        self.assertEqual(staged.read_bytes(), profile.read_bytes())
        self.assertEqual(staged.stat().st_mode & 0o222, 0)
        self.assertEqual(staged_manifest.read_bytes(), profile_manifest.read_bytes())
        self.assertEqual(staged_manifest.stat().st_mode & 0o222, 0)
        self.assertEqual(
            Path(self.option_value(self.build_arguments(), "-build")),
            build_directory,
        )

        qmake_args = self.qmake_arguments()
        for variable in (
            "QMAKE_CFLAGS+=",
            "QMAKE_CXXFLAGS+=",
            "QMAKE_LFLAGS+=",
        ):
            value = next(arg for arg in qmake_args if arg.startswith(variable))
            self.assertIn(f"-fprofile-use={staged}", value)
            self.assertIn("-Werror=profile-instr-out-of-date", value)
            self.assertIn("-march=znver2", value)

        manifest_text = self.manifest_path(
            "znver2",
            pgo="use",
            profile_hash=profile_hash,
            manifest_hash=manifest_hash,
        ).read_text(encoding="utf-8")
        self.assertIn("pgo_mode=use", manifest_text)
        self.assertIn(f"pgo_profile_source_path={profile}", manifest_text)
        self.assertIn(f"pgo_profile_sha256={profile_hash}", manifest_text)
        self.assertIn(f"pgo_profile_build_path={staged}", manifest_text)
        self.assertIn(
            f"pgo_profile_manifest_source_path={profile_manifest}", manifest_text
        )
        self.assertIn(f"pgo_profile_manifest_sha256={manifest_hash}", manifest_text)
        self.assertIn(
            f"pgo_profile_manifest_build_path={staged_manifest}", manifest_text
        )

        self.clear_logs()
        resumed = self.run_helper(
            "--profile",
            "znver2",
            "--pgo=use",
            "--pgo-profile",
            str(profile),
            "--pgo-profile-manifest",
            str(profile_manifest),
            environment={"FAKE_NATIVE_CPU": "znver2"},
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("manifest status:    matching-resume", resumed.stdout)
        self.assertEqual(staged.read_bytes(), profile.read_bytes())

    def test_pgo_use_rejects_a_tampered_staged_profile(self) -> None:
        profile = self.make_immutable_profile(contents=b"trusted profile")
        profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
        profile_manifest = self.make_immutable_profile_manifest()
        manifest_hash = hashlib.sha256(profile_manifest.read_bytes()).hexdigest()
        first = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(profile),
            "--pgo-profile-manifest",
            str(profile_manifest),
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        staged = (
            self.profile_build_directory(
                pgo="use",
                profile_hash=profile_hash,
                manifest_hash=manifest_hash,
            )
            / f"pgo-input-{profile_hash}.profdata"
        )
        staged.chmod(0o644)
        staged.write_bytes(b"tampered profile")
        staged.chmod(0o444)
        self.clear_logs()

        second = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(profile),
            "--pgo-profile-manifest",
            str(profile_manifest),
        )
        self.assertEqual(second.returncode, 2)
        self.assertIn(
            "staged PGO profile does not match its manifest identity",
            second.stderr,
        )
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())

    def test_pgo_profile_hash_separates_use_builds(self) -> None:
        first_profile = self.make_immutable_profile(
            "first.profdata", b"first profile"
        )
        second_profile = self.make_immutable_profile(
            "second.profdata", b"second profile"
        )
        first_hash = hashlib.sha256(first_profile.read_bytes()).hexdigest()
        second_hash = hashlib.sha256(second_profile.read_bytes()).hexdigest()
        profile_manifest = self.make_immutable_profile_manifest()
        manifest_hash = hashlib.sha256(profile_manifest.read_bytes()).hexdigest()

        first = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(first_profile),
            "--pgo-profile-manifest",
            str(profile_manifest),
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.clear_logs()
        second = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(second_profile),
            "--pgo-profile-manifest",
            str(profile_manifest),
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotEqual(first_hash, second_hash)
        self.assertTrue(
            self.profile_build_directory(
                pgo="use", profile_hash=first_hash, manifest_hash=manifest_hash
            ).is_dir()
        )
        self.assertTrue(
            self.profile_build_directory(
                pgo="use", profile_hash=second_hash, manifest_hash=manifest_hash
            ).is_dir()
        )

    def test_pgo_phase_directories_do_not_mix_with_default(self) -> None:
        baseline = self.run_helper("--pgo", "none")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertIn("PGO mode:          none", baseline.stdout)
        self.clear_logs()
        generate = self.run_helper("--pgo", "generate")
        self.assertEqual(generate.returncode, 0, generate.stderr)
        self.assertTrue(self.profile_build_directory().is_dir())
        self.assertTrue(self.profile_build_directory(pgo="generate").is_dir())
        self.assertNotEqual(
            self.profile_build_directory(),
            self.profile_build_directory(pgo="generate"),
        )

    def test_pgo_control_is_clean_bound_and_has_no_profile_flags(self) -> None:
        control = self.run_helper("--pgo", "control")
        self.assertEqual(control.returncode, 0, control.stderr)
        self.assertIn("PGO mode:          control", control.stdout)
        build_directory = self.profile_build_directory(pgo="control")
        self.assertEqual(
            Path(self.option_value(self.build_arguments(), "-build")),
            build_directory,
        )
        self.assertIn("status=complete", (build_directory / PGO_STATE_NAME).read_text())
        self.assertFalse((build_directory / PGO_LOCK_NAME).exists())
        qmake_text = " ".join(self.qmake_arguments())
        self.assertIn("-flto=full", qmake_text)
        self.assertNotIn("-fprofile-generate", qmake_text)
        self.assertNotIn("-fprofile-use", qmake_text)
        manifest_text = self.manifest_path(pgo="control").read_text()
        commit, tree = self.source_identity()
        self.assertIn("pgo_mode=control", manifest_text)
        self.assertIn(f"source_commit={commit}", manifest_text)
        self.assertIn(f"source_tree={tree}", manifest_text)

    def test_explicit_pgo_never_falls_back_to_non_pgo(self) -> None:
        result = self.run_helper(
            "--pgo", "generate", environment={"FAKE_FAIL": "pgo-link"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PGO generate full-LTO link probe failed", result.stderr)
        self.assertIn("refusing a non-PGO build", result.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())
        self.assertFalse(self.output_root.exists())

    def test_znver2_fallback_retains_requested_pgo_phase(self) -> None:
        result = self.run_helper("--profile", "znver2", "--pgo", "generate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("falling back to portable", result.stderr)
        self.assertIn("resolved profile:  portable", result.stdout)
        self.assertIn("PGO mode:          generate", result.stdout)
        self.assertEqual(
            Path(self.option_value(self.build_arguments(), "-build")),
            self.profile_build_directory(pgo="generate"),
        )
        qmake_args = self.qmake_arguments()
        self.assertIn("-fprofile-generate=", " ".join(qmake_args))
        self.assertNotIn("-march=znver2", " ".join(qmake_args))

    def test_pgo_dry_run_is_nonmutating(self) -> None:
        result = self.run_helper("--pgo", "generate", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PGO mode:          generate", result.stdout)
        self.assertIn("new (dry-run; not written)", result.stdout)
        self.assertFalse(self.output_root.exists())
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())

    def test_help_requires_process_and_module_profile_placeholders(self) -> None:
        result = self.run_helper("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("containing both %p and %m", result.stdout)
        self.assertIn("%m_%p.profraw", result.stdout)

    def test_pgo_requires_a_clean_committed_source(self) -> None:
        with self.build.open("a", encoding="utf-8") as stream:
            stream.write("\n# uncommitted change\n")
        pgo = self.run_helper("--pgo", "generate", "--dry-run")
        self.assertEqual(pgo.returncode, 2)
        self.assertIn("require a clean source tree", pgo.stderr)
        self.assertFalse(self.output_root.exists())

        # The non-PGO default deliberately retains its existing behavior.
        baseline = self.run_helper("--dry-run")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)

    def test_pgo_use_requires_regular_read_only_profile(self) -> None:
        missing = self.run_helper("--pgo", "use")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("requires --pgo-profile", missing.stderr)

        profile = self.make_immutable_profile()
        missing_manifest = self.run_helper(
            "--pgo", "use", "--pgo-profile", str(profile)
        )
        self.assertEqual(missing_manifest.returncode, 2)
        self.assertIn("requires --pgo-profile-manifest", missing_manifest.stderr)
        profile_manifest = self.make_immutable_profile_manifest()

        writable = self.directory / "writable.profdata"
        writable.write_bytes(b"mutable")
        writable.chmod(0o644)
        mutable = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(writable),
            "--pgo-profile-manifest",
            str(profile_manifest),
        )
        self.assertEqual(mutable.returncode, 2)
        self.assertIn("must be immutable", mutable.stderr)

        symlink = self.directory / "linked.profdata"
        symlink.symlink_to(profile)
        linked = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(symlink),
            "--pgo-profile-manifest",
            str(profile_manifest),
        )
        self.assertEqual(linked.returncode, 2)
        self.assertIn("must not be a symbolic link", linked.stderr)
        self.assertFalse(self.build_log.exists())
        self.assertFalse(self.qmake_log.exists())

    def test_pgo_use_requires_immutable_valid_json_manifest(self) -> None:
        profile = self.make_immutable_profile()
        invalid = self.make_immutable_profile_manifest(
            "invalid.json", b"not json\n"
        )
        invalid_result = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(profile),
            "--pgo-profile-manifest",
            str(invalid),
        )
        self.assertEqual(invalid_result.returncode, 2)
        self.assertIn("not a valid JSON object", invalid_result.stderr)

        writable = self.directory / "writable-manifest.json"
        writable.write_text("{}\n", encoding="utf-8")
        writable.chmod(0o644)
        writable_result = self.run_helper(
            "--pgo",
            "use",
            "--pgo-profile",
            str(profile),
            "--pgo-profile-manifest",
            str(writable),
        )
        self.assertEqual(writable_result.returncode, 2)
        self.assertIn("PGO profile manifest must be immutable", writable_result.stderr)

    def test_invalid_and_helper_managed_options_are_rejected(self) -> None:
        invalid = self.run_helper("--profile", "native")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("unsupported profile", invalid.stderr)
        invalid_pgo = self.run_helper("--pgo", "sample")
        self.assertEqual(invalid_pgo.returncode, 2)
        self.assertIn("unsupported PGO mode", invalid_pgo.stderr)
        unused_profile = self.run_helper(
            "--pgo", "generate", "--pgo-profile", "/tmp/unneeded.profdata"
        )
        self.assertEqual(unused_profile.returncode, 2)
        self.assertIn("are valid only with --pgo use", unused_profile.stderr)
        managed = self.run_helper("--", "-build", "/tmp/collision")
        self.assertEqual(managed.returncode, 2)
        self.assertIn("managed by this helper", managed.stderr)
        self.assertFalse(self.build_log.exists())


if __name__ == "__main__":
    unittest.main()
