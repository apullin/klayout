#!/usr/bin/env python3
"""Focused tests for path-independent KLayout runtime provenance."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import klayout_runtime_provenance as provenance  # noqa: E402


class RuntimeProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.ldd = self.directory / "trusted-ldd"
        self._write_executable(self.ldd, b"test ldd")

    @staticmethod
    def _write_executable(path: Path, contents: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def make_install(
        self,
        name: str,
        *,
        dependency: bytes = b"selected dependency",
        plugin: bytes = b"runtime plugin",
        preload: bytes = b"preloaded allocator",
        adjacent: bytes = b"unloaded adjacent library",
    ) -> dict[str, Path]:
        root = self.directory / name
        root.mkdir(parents=True)
        executable = root / "klayout"
        self._write_executable(executable, b"same klayout front-end")

        dependency_path = root / "loader" / "libselected.so.7"
        dependency_path.parent.mkdir()
        dependency_path.write_bytes(dependency)

        plugin_dependency_path = root / "loader" / "libpluginonly.so.3"
        plugin_dependency_path.write_bytes(b"plugin-only dependency")

        preload_path = root / "runtime" / "libpreload.so"
        preload_path.parent.mkdir()
        preload_path.write_bytes(preload)

        plugin_path = root / "db_plugins" / "nested" / "libformat.so.1"
        plugin_path.parent.mkdir(parents=True)
        plugin_path.write_bytes(plugin)
        (plugin_path.parent / "libformat.so").symlink_to(plugin_path.name)

        adjacent_path = root / "libklayout_unused.so.1"
        adjacent_path.write_bytes(adjacent)
        (root / "libklayout_unused.so").symlink_to(adjacent_path.name)
        return {
            "root": root,
            "executable": executable,
            "dependency": dependency_path,
            "plugin_dependency": plugin_dependency_path,
            "preload": preload_path,
            "plugin": plugin_path,
            "adjacent": adjacent_path,
        }

    @staticmethod
    def fake_ldd(
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        stdout: object,
        stderr: object,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, stdout, stderr, text, check
        root = Path(env["TEST_INSTALL_ROOT"])
        subject = Path(command[-1])
        dependency = Path(
            env.get("TEST_SELECTED_LIBRARY", root / "loader" / "libselected.so.7")
        )
        preload_tokens = [
            token
            for token in __import__("re").split(
                r"[:\s]+", env.get("LD_PRELOAD", "")
            )
            if token
        ]
        lines = ["linux-vdso.so.1 (0x00007ffff7fc0000)"]
        for token in preload_tokens:
            if "/" in token:
                lines.append(f"{Path(token)} (0x00007ffff7fb0000)")
            elif token == "libpreload.so":
                lines.append(
                    f"libpreload.so => {root / 'runtime' / token} "
                    "(0x00007ffff7fb0000)"
                )
        lines.extend(
            [
                f"libselected.so.7 => {dependency} (0x00007ffff7fa0000)",
                "/lib64/ld-linux-x86-64.so.2 (0x00007ffff7ff0000)",
            ]
        )
        if env.get("TEST_RUBY_LIBRARY"):
            lines.insert(
                -1,
                "libruby.so.4.0 => "
                f"{env['TEST_RUBY_LIBRARY']} (0x00007ffff7f80000)",
            )
        if env.get("TEST_PYTHON_LIBRARY"):
            lines.insert(
                -1,
                "libpython3.12.so.1.0 => "
                f"{env['TEST_PYTHON_LIBRARY']} (0x00007ffff7f70000)",
            )
        if subject.name != "klayout":
            lines.insert(
                -1,
                "libpluginonly.so.3 => "
                f"{root / 'loader' / 'libpluginonly.so.3'} "
                "(0x00007ffff7f90000)",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(lines) + "\n",
            stderr="",
        )

    def add_language_runtime(self, install: dict[str, Path]) -> dict[str, Path]:
        library_root = install["root"] / "language" / "lib"
        ruby_library = library_root / "libruby.so.4.0"
        python_library = library_root / "libpython3.12.so.1.0"
        ruby_source = library_root / "ruby" / "4.0.0" / "time.rb"
        ruby_native = library_root / "ruby" / "4.0.0" / "date_core.so"
        python_source = library_root / "python3.12" / "logger.py"
        python_native = library_root / "python3.12" / "_datetime.so"
        for path, contents in (
            (ruby_library, b"ruby interpreter"),
            (python_library, b"python interpreter"),
            (ruby_source, b"ruby time source"),
            (ruby_native, b"ruby native extension"),
            (python_source, b"python logger source"),
            (python_native, b"python native extension"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        return {
            "ruby_library": ruby_library,
            "python_library": python_library,
            "ruby_source": ruby_source,
            "ruby_native": ruby_native,
            "python_source": python_source,
            "python_native": python_native,
        }

    def collect(
        self,
        install: dict[str, Path],
        environment: dict[str, str] | None = None,
        ldd_side_effect: object | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        klayout_home = install["root"] / "empty-klayout-home"
        klayout_home.mkdir(exist_ok=True)
        selected_environment = {
            "LD_PRELOAD": str(install["preload"]),
            "KLAYOUT_HOME": str(klayout_home),
            "TEST_INSTALL_ROOT": str(install["root"]),
            **(environment or {}),
        }
        with mock.patch.object(
            provenance.subprocess,
            "run",
            side_effect=ldd_side_effect or self.fake_ldd,
        ):
            return provenance.collect_runtime_provenance(
                install["executable"],
                selected_environment,
                cwd=install["root"],
                ldd_path=self.ldd,
                **kwargs,
            )

    def test_actual_loader_selection_changes_fingerprint(self) -> None:
        install = self.make_install("selected")
        first_library = self.directory / "selection-a" / "libselected.so.7"
        second_library = self.directory / "selection-b" / "libselected.so.7"
        first_library.parent.mkdir()
        second_library.parent.mkdir()
        first_library.write_bytes(b"first selected implementation")
        second_library.write_bytes(b"second selected implementation")

        first = self.collect(
            install,
            {
                "LD_LIBRARY_PATH": str(first_library.parent),
                "TEST_SELECTED_LIBRARY": str(first_library),
            },
            path_replacements={str(first_library.parent): "$SELECTED"},
            normalization_policy_id="selected-library-path-v1",
        )
        second = self.collect(
            install,
            {
                "LD_LIBRARY_PATH": str(second_library.parent),
                "TEST_SELECTED_LIBRARY": str(second_library),
            },
            path_replacements={str(second_library.parent): "$SELECTED"},
            normalization_policy_id="selected-library-path-v1",
        )

        first_dependency = first["identity"]["elf_dependencies"]["files"][0]
        second_dependency = second["identity"]["elf_dependencies"]["files"][0]
        self.assertNotEqual(first_dependency["sha256"], second_dependency["sha256"])
        self.assertNotEqual(first["canonical_sha256"], second["canonical_sha256"])

    def test_adjacent_but_unloaded_library_is_explicitly_excluded(self) -> None:
        install = self.make_install("adjacent")
        before = self.collect(install)
        install["adjacent"].write_bytes(b"changed but still unloaded")
        after = self.collect(install)

        self.assertEqual(before["canonical_sha256"], after["canonical_sha256"])
        before_unloaded = before["adjacent_klayout_libraries"]["unloaded"]
        after_unloaded = after["adjacent_klayout_libraries"]["unloaded"]
        self.assertEqual(len(before_unloaded), 1)
        self.assertFalse(before_unloaded[0]["included_in_identity"])
        self.assertNotEqual(before_unloaded[0]["sha256"], after_unloaded[0]["sha256"])
        self.assertIn(
            "diagnostic metadata only",
            before["adjacent_klayout_libraries"]["policy"],
        )

    def test_recursive_plugin_change_alters_fingerprint_and_aliases_deduplicate(
        self,
    ) -> None:
        install = self.make_install("plugin")
        before = self.collect(install)
        plugin_files = before["runtime_plugins"]["files"]
        self.assertEqual(len(plugin_files), 1)
        self.assertEqual(
            plugin_files[0]["relative_aliases"],
            [
                "install/db_plugins/nested/libformat.so",
                "install/db_plugins/nested/libformat.so.1",
            ],
        )

        install["plugin"].write_bytes(b"changed runtime plugin")
        after = self.collect(install)
        self.assertNotEqual(before["canonical_sha256"], after["canonical_sha256"])

    def test_plugin_only_dependency_change_alters_fingerprint(self) -> None:
        install = self.make_install("plugin-dependency")
        before = self.collect(install)
        plugin_dependency_sha = hashlib.sha256(
            b"plugin-only dependency"
        ).hexdigest()
        matching = [
            record
            for record in before["identity"]["elf_dependencies"]["files"]
            if record["sha256"] == plugin_dependency_sha
        ]
        self.assertEqual(len(matching), 1)
        self.assertTrue(
            all(
                selection["probe_role"].startswith("runtime_plugin:")
                for selection in matching[0]["selections"]
            )
        )

        install["plugin_dependency"].write_bytes(
            b"changed plugin-only dependency"
        )
        after = self.collect(install)
        self.assertNotEqual(before["canonical_sha256"], after["canonical_sha256"])

    def test_language_source_native_and_install_roots_enter_identity(self) -> None:
        install = self.make_install("language-runtime")
        language = self.add_language_runtime(install)
        install_files = []
        for directory, filename in (
            ("ruby", "klayout.rb"),
            ("python", "klayout.py"),
            ("pymod", "adapter.py"),
        ):
            path = install["root"] / directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{directory} install module".encode())
            install_files.append(path)
        environment = {
            "TEST_RUBY_LIBRARY": str(language["ruby_library"]),
            "TEST_PYTHON_LIBRARY": str(language["python_library"]),
        }

        baseline = self.collect(install, environment)
        self.assertEqual(
            baseline["identity"]["schema"], "klayout-runtime-identity-v2"
        )
        roles = {
            role
            for root in baseline["identity"]["language_runtime_roots"]
            for role in root["roles"]
        }
        self.assertTrue(
            {
                "selected-libruby-stdlib",
                "selected-libpython-3.12-stdlib",
                "klayout-install-ruby",
                "klayout-install-python",
                "klayout-install-pymod",
            }.issubset(roles)
        )
        self.assertTrue(
            all(
                "files" not in root
                for root in baseline["language_runtime"]["roots"]
            )
        )

        for path in (
            language["ruby_source"],
            language["ruby_native"],
            language["python_source"],
            language["python_native"],
            *install_files,
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b" changed")
                changed = self.collect(install, environment)
                self.assertNotEqual(
                    baseline["canonical_sha256"], changed["canonical_sha256"]
                )
                path.write_bytes(original)

    def test_only_klayout_python_path_contents_enter_identity(self) -> None:
        install = self.make_install("python-path")
        language = self.add_language_runtime(install)
        ignored = self.directory / "inherited-python" / "ignored.py"
        effective = self.directory / "klayout-python" / "effective.py"
        for path in (ignored, effective):
            path.parent.mkdir()
            path.write_bytes(b"initial python module")
        base_environment = {
            "TEST_PYTHON_LIBRARY": str(language["python_library"]),
            "PYTHONPATH": str(ignored.parent),
        }

        before_ignored = self.collect(install, base_environment)
        ignored.write_bytes(b"changed but KLayout clears PYTHONPATH")
        after_ignored = self.collect(install, base_environment)
        self.assertEqual(
            before_ignored["canonical_sha256"], after_ignored["canonical_sha256"]
        )

        effective_environment = {
            **base_environment,
            "KLAYOUT_PYTHONPATH": str(effective.parent),
        }
        before_effective = self.collect(install, effective_environment)
        effective.write_bytes(b"changed effective python module")
        after_effective = self.collect(install, effective_environment)
        self.assertNotEqual(
            before_effective["canonical_sha256"],
            after_effective["canonical_sha256"],
        )

    def test_absolute_rubyopt_require_outside_roots_enters_identity(self) -> None:
        install = self.make_install("rubyopt-require")
        language = self.add_language_runtime(install)
        required = self.directory / "external-ruby" / "boot.rb"
        required.parent.mkdir()
        required.write_bytes(b"external boot module")
        environment = {
            "TEST_RUBY_LIBRARY": str(language["ruby_library"]),
            "RUBYOPT": f"-r{required}",
        }

        before = self.collect(install, environment)
        roles = {
            role
            for root in before["identity"]["language_runtime_roots"]
            for role in root["roles"]
        }
        self.assertIn("environment-RUBYOPT-require-1", roles)
        required.write_bytes(b"changed external boot module")
        after = self.collect(install, environment)
        self.assertNotEqual(before["canonical_sha256"], after["canonical_sha256"])

    def test_mid_collection_language_mutation_is_rejected(self) -> None:
        install = self.make_install("language-mutation")
        language = self.add_language_runtime(install)
        calls = 0

        def mutating_ldd(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 3:
                language["ruby_source"].write_bytes(b"changed between passes")
            return self.fake_ldd(*args, **kwargs)

        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError,
            "runtime changed while provenance was collected: "
            "language_runtime_roots",
        ):
            self.collect(
                install,
                {
                    "TEST_RUBY_LIBRARY": str(language["ruby_library"]),
                    "TEST_PYTHON_LIBRARY": str(language["python_library"]),
                },
                ldd_side_effect=mutating_ldd,
            )

    def test_mid_collection_code_mutations_are_rejected(self) -> None:
        mutation_cases = (
            ("executable", "executable", 1, b"changed executable"),
            ("dependency", "elf_dependencies", 3, b"changed dependency"),
            ("plugin", "runtime_plugins", 2, b"changed plugin"),
        )
        for path_key, expected_component, mutation_call, contents in mutation_cases:
            with self.subTest(component=expected_component):
                install = self.make_install(f"mutation-{path_key}")
                calls = 0

                def mutating_ldd(*args: object, **kwargs: object) -> object:
                    nonlocal calls
                    calls += 1
                    if calls == mutation_call:
                        install[path_key].write_bytes(contents)
                    return self.fake_ldd(*args, **kwargs)

                with self.assertRaisesRegex(
                    provenance.RuntimeProvenanceError,
                    "runtime changed while provenance was collected: "
                    + expected_component,
                ):
                    self.collect(install, ldd_side_effect=mutating_ldd)

    def test_unresolved_preload_fails_closed(self) -> None:
        install = self.make_install("preload-error")
        missing = install["root"] / "missing" / "liballocator.so"
        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError,
            "LD_PRELOAD entry.*does not resolve",
        ):
            self.collect(install, {"LD_PRELOAD": str(missing)})

        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError,
            "bare LD_PRELOAD entry.*not found",
        ):
            self.collect(install, {"LD_PRELOAD": "unknown-preload.so"})

        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError,
            "contains a path but is not absolute",
        ):
            self.collect(install, {"LD_PRELOAD": "./runtime/libpreload.so"})

    def test_bare_preload_is_accepted_only_when_ldd_proves_selection(self) -> None:
        install = self.make_install("bare-preload")
        result = self.collect(install, {"LD_PRELOAD": "libpreload.so"})
        ordered = result["ld_preload"]["ordered_files"]
        self.assertEqual(len(ordered), 1)
        self.assertEqual(ordered[0]["loader_position"], 1)
        self.assertEqual(
            ordered[0]["sha256"],
            hashlib.sha256(b"preloaded allocator").hexdigest(),
        )

    def test_install_paths_do_not_affect_canonical_identity(self) -> None:
        first_install = self.make_install("relocated-a")
        second_install = self.make_install("relocated-b")
        first = self.collect(
            first_install,
            {
                "LD_LIBRARY_PATH": str(first_install["root"] / "loader"),
                "PYTHONPATH": str(first_install["root"] / "python"),
                "KLAYOUT_TEST_DATA": str(first_install["root"] / "data"),
                "KLAYOUT_HOME": str(first_install["root"] / "ephemeral-home"),
            },
            path_replacements={str(first_install["root"]): "$INSTALL"},
            normalization_policy_id="relocatable-install-v1",
        )
        second = self.collect(
            second_install,
            {
                "LD_LIBRARY_PATH": str(second_install["root"] / "loader"),
                "PYTHONPATH": str(second_install["root"] / "python"),
                "KLAYOUT_TEST_DATA": str(second_install["root"] / "data"),
                "KLAYOUT_HOME": str(second_install["root"] / "ephemeral-home"),
            },
            path_replacements={str(second_install["root"]): "$INSTALL"},
            normalization_policy_id="relocatable-install-v1",
        )

        self.assertNotEqual(
            first["executable"]["resolved_path"],
            second["executable"]["resolved_path"],
        )
        self.assertEqual(first["identity"], second["identity"])
        self.assertEqual(first["canonical_sha256"], second["canonical_sha256"])
        self.assertEqual(
            first["environment"]["normalized"]["KLAYOUT_HOME"],
            "$INSTALL/ephemeral-home",
        )

    def test_environment_capture_is_broad_and_normalizer_can_omit_values(self) -> None:
        install = self.make_install("environment")
        result = self.collect(
            install,
            {
                "KLAYOUT_DRC_THREADS": "8",
                "RUBYLIB": "/ruby/path",
                "PYTHONHOME": "/python/home",
                "LC_ALL": "C.UTF-8",
                "MALLOC_CONF": "background_thread:true",
                "OMP_NUM_THREADS": "8",
                "KLAYOUT_API_TOKEN": "never print this token",
                "UNRELATED_SECRET": "ignored",
            },
            environment_normalizer=(
                lambda key, value: None if key == "PYTHONHOME" else value
            ),
            normalization_policy_id="omit-python-home-v1",
        )
        captured = result["environment"]["captured"]
        normalized = result["environment"]["normalized"]
        for key in (
            "KLAYOUT_DRC_THREADS",
            "RUBYLIB",
            "PYTHONHOME",
            "LC_ALL",
            "MALLOC_CONF",
            "OMP_NUM_THREADS",
        ):
            self.assertIn(key, captured)
        self.assertNotIn("UNRELATED_SECRET", captured)
        self.assertNotIn("PYTHONHOME", normalized)
        self.assertEqual(result["environment"]["hook_omitted_keys"], ["PYTHONHOME"])
        self.assertNotIn(
            "never print this token", str(result["environment"])
        )
        self.assertTrue(captured["KLAYOUT_API_TOKEN"]["redacted"])
        self.assertIn("redacted_value_sha256", normalized["KLAYOUT_API_TOKEN"])

    def test_normalization_policy_is_required_and_part_of_identity(self) -> None:
        install = self.make_install("normalization-policy")
        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError,
            "require a stable normalization_policy_id",
        ):
            self.collect(
                install,
                path_replacements={str(install["root"]): "$INSTALL"},
            )

        first = self.collect(
            install,
            environment_normalizer=lambda key, value: value,
            normalization_policy_id="identity-hook-v1",
        )
        second = self.collect(
            install,
            environment_normalizer=lambda key, value: value,
            normalization_policy_id="identity-hook-v2",
        )
        self.assertNotEqual(first["canonical_sha256"], second["canonical_sha256"])
        self.assertEqual(
            first["identity"]["environment_normalization_policy"]["id"],
            "identity-hook-v1",
        )

    def test_loader_audit_and_uncovered_plugin_paths_fail_closed(self) -> None:
        install = self.make_install("unsafe-loader-state")
        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError, "nonempty LD_AUDIT is unsupported"
        ):
            self.collect(install, {"LD_AUDIT": "/tmp/audit.so"})
        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError, "nonempty KLAYOUT_PATH"
        ):
            self.collect(install, {"KLAYOUT_PATH": "/tmp/external-klayout"})

    def test_install_architecture_plugins_require_explicit_coverage(self) -> None:
        install = self.make_install("architecture-plugins")
        architecture_root = (
            install["root"] / "x86_64-linux-gcc" / "db_plugins"
        )
        architecture_root.mkdir(parents=True)
        architecture_plugin = architecture_root / "libarchitecture.so"
        architecture_plugin.write_bytes(b"architecture-selected plugin")

        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError,
            "architecture plugin state cannot be selected",
        ):
            self.collect(install)

        covered = self.collect(
            install,
            plugin_directories={
                "install-db": install["root"] / "db_plugins",
                "architecture-db": architecture_root,
            },
            plugin_search_policy_id="fixture-architecture-roots-v1",
        )
        self.assertEqual(len(covered["runtime_plugins"]["files"]), 2)
        self.assertEqual(
            covered["identity"]["plugin_search_policy"]["id"],
            "fixture-architecture-roots-v1",
        )

    def test_install_salt_state_requires_explicit_coverage(self) -> None:
        install = self.make_install("salt-plugins")
        salt_root = install["root"] / "salt"
        salt_root.mkdir()
        # An empty Salt tree cannot contribute code and remains acceptable.
        self.collect(install)

        package = salt_root / "fixture-package"
        package.mkdir()
        (package / "grain.xml").write_text("<grain/>", encoding="utf-8")
        with self.assertRaisesRegex(
            provenance.RuntimeProvenanceError,
            "installation Salt state may add versioned runtime plugins",
        ):
            self.collect(install)

    def test_bare_executable_path_uses_supplied_cwd(self) -> None:
        install = self.make_install("relative-path/tools")
        cwd = install["root"].parent
        klayout_home = cwd / "empty-home"
        klayout_home.mkdir()
        environment = {
            "PATH": "tools",
            "KLAYOUT_HOME": str(klayout_home),
            "TEST_INSTALL_ROOT": str(install["root"]),
        }
        with mock.patch.object(
            provenance.subprocess, "run", side_effect=self.fake_ldd
        ):
            relative = provenance.collect_runtime_provenance(
                "klayout",
                environment,
                cwd=cwd,
                ldd_path=self.ldd,
            )
        self.assertEqual(
            relative["executable"]["resolved_path"],
            str(install["executable"].resolve()),
        )

        environment["PATH"] = ""
        with mock.patch.object(
            provenance.subprocess, "run", side_effect=self.fake_ldd
        ):
            empty_entry = provenance.collect_runtime_provenance(
                "klayout",
                environment,
                cwd=install["root"],
                ldd_path=self.ldd,
            )
        self.assertEqual(
            empty_entry["executable"]["resolved_path"],
            str(install["executable"].resolve()),
        )

    def test_missing_elf_dependency_and_unparseable_ldd_output_reject(self) -> None:
        install = self.make_install("ldd-errors")

        def not_found(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            del args, kwargs
            return subprocess.CompletedProcess(
                [str(self.ldd)], 0, "libmissing.so => not found\n", ""
            )

        with (
            mock.patch.object(provenance.subprocess, "run", side_effect=not_found),
            self.assertRaisesRegex(
                provenance.RuntimeProvenanceError, "ELF dependency was not found"
            ),
        ):
            provenance.collect_runtime_provenance(
                install["executable"], {}, ldd_path=self.ldd
            )

        def malformed(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            del args, kwargs
            return subprocess.CompletedProcess(
                [str(self.ldd)], 0, "surprising loader prose\n", ""
            )

        with (
            mock.patch.object(provenance.subprocess, "run", side_effect=malformed),
            self.assertRaisesRegex(
                provenance.RuntimeProvenanceError, "unrecognized ldd output"
            ),
        ):
            provenance.collect_runtime_provenance(
                install["executable"], {}, ldd_path=self.ldd
            )


if __name__ == "__main__":
    unittest.main()
