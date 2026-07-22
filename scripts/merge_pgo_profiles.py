#!/usr/bin/env python3
"""Fail-closed construction of a weighted LLVM instrumentation profile.

The published object is an atomically renamed, read-only directory containing
``merged.profdata`` and ``manifest.json``.  Raw profiles are never accepted
through globs or implicit discovery outside the explicitly named workload
directories.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Sequence


SCHEMA = "klayout-pgo-profile-bundle-v1"
PROFILE_NAME = "merged.profdata"
MANIFEST_NAME = "manifest.json"
NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
SUMMARY_RE = re.compile(
    r"^\s*(Total functions|Maximum function count|"
    r"Maximum internal block count|Total count):\s*([0-9]+)\s*$"
)
LIBLLVM_SONAME = "libLLVM.so.22.1"
LIBLLVM_LINE_RE = re.compile(
    r"^\s*libLLVM\.so\.22\.1\s+=>\s+(\S+)\s+"
    r"\(0x[0-9a-fA-F]+\)\s*$"
)
LOADER_ENVIRONMENT_VARIABLES = ("LD_LIBRARY_PATH", "LD_PRELOAD")
RENAME_NOREPLACE = 1


class MergeError(RuntimeError):
    """An expected validation or tool failure."""


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    directory: Path
    weight: int


@dataclass(frozen=True)
class Snapshot:
    directory: str
    directories: tuple[dict[str, Any], ...]
    files: tuple[dict[str, Any], ...]
    inventory_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "inventory_sha256": self.inventory_sha256,
            "directories": list(self.directories),
            "files": list(self.files),
        }


@dataclass(frozen=True)
class ToolIdentity:
    path: Path
    sha256: str
    stat_identity: tuple[int, int, int, int, int, int]
    version_stdout: str
    version_stdout_sha256: str


@dataclass(frozen=True)
class DependencyIdentity:
    path: Path
    sha256: str
    stat_identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ResolverIdentity:
    path: Path
    sha256: str
    stat_identity: tuple[int, int, int, int, int, int]
    version_stdout: str
    version_stdout_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IMODE(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stat_manifest(relative_path: str, value: os.stat_result) -> dict[str, Any]:
    return {
        "path": relative_path,
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": f"{stat.S_IMODE(value.st_mode):04o}",
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _normal_absolute(path_text: str, description: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise MergeError(f"{description} must be an absolute path: {path_text}")
    normal = Path(os.path.normpath(os.fspath(path)))
    if os.fspath(normal) != os.fspath(path):
        raise MergeError(
            f"{description} must be lexically normalized: {path_text}"
        )
    return normal


def _lstat_without_symlink_components(path: Path, description: str) -> os.stat_result:
    current = Path(path.anchor)
    try:
        root_stat = os.lstat(current)
        if stat.S_ISLNK(root_stat.st_mode):
            raise MergeError(f"{description} traverses a symlink: {current}")
        for component in path.parts[1:]:
            current = current / component
            value = os.lstat(current)
            if stat.S_ISLNK(value.st_mode):
                raise MergeError(f"{description} traverses a symlink: {current}")
        return value if path.parts[1:] else root_stat
    except FileNotFoundError as error:
        raise MergeError(f"{description} does not exist: {current}") from error
    except OSError as error:
        raise MergeError(f"cannot inspect {description} {current}: {error}") from error


def _hash_open_regular(path: Path, expected: os.stat_result, description: str) -> str:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MergeError(f"cannot open {description} {path}: {error}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MergeError(f"{description} is not a regular file: {path}")
        if _stat_identity(before) != _stat_identity(expected):
            raise MergeError(f"{description} mutated while being opened: {path}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if _stat_identity(after) != _stat_identity(before):
            raise MergeError(f"{description} mutated while being hashed: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _snapshot_workload(directory: Path) -> Snapshot:
    root_stat = _lstat_without_symlink_components(directory, "workload directory")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise MergeError(f"workload path is not a directory: {directory}")

    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    def visit(current: Path, relative: Path) -> None:
        try:
            before = os.lstat(current)
        except OSError as error:
            raise MergeError(f"cannot inspect workload directory {current}: {error}") from error
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise MergeError(f"workload directory became nonregular: {current}")
        relative_text = "." if relative == Path(".") else relative.as_posix()
        directories.append(_stat_manifest(relative_text, before))
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            raise MergeError(f"cannot enumerate workload directory {current}: {error}") from error
        for entry in entries:
            child = current / entry.name
            child_relative = (
                Path(entry.name)
                if relative == Path(".")
                else relative / entry.name
            )
            try:
                value = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise MergeError(f"cannot inspect workload input {child}: {error}") from error
            if stat.S_ISLNK(value.st_mode):
                raise MergeError(f"workload input is a symlink: {child}")
            if stat.S_ISDIR(value.st_mode):
                visit(child, child_relative)
                continue
            if not stat.S_ISREG(value.st_mode):
                raise MergeError(f"workload input is not a regular file: {child}")
            if child.suffix != ".profraw":
                raise MergeError(f"unexpected non-.profraw workload file: {child}")
            if value.st_size == 0:
                raise MergeError(f"workload profile is empty: {child}")
            record = _stat_manifest(child_relative.as_posix(), value)
            record["sha256"] = _hash_open_regular(child, value, "workload profile")
            files.append(record)
        try:
            after = os.lstat(current)
        except OSError as error:
            raise MergeError(f"workload directory vanished: {current}") from error
        if _stat_identity(after) != _stat_identity(before):
            raise MergeError(f"workload directory mutated while inventoried: {current}")

    visit(directory, Path("."))
    files.sort(key=lambda item: item["path"])
    directories.sort(key=lambda item: item["path"])
    if not files:
        raise MergeError(f"workload directory has no .profraw files: {directory}")
    inventory = {"directories": directories, "files": files}
    return Snapshot(
        directory=os.fspath(directory),
        directories=tuple(directories),
        files=tuple(files),
        inventory_sha256=_canonical_sha256(inventory),
    )


def _assert_unchanged(expected: Snapshot) -> None:
    current = _snapshot_workload(Path(expected.directory))
    if current != expected:
        raise MergeError(f"workload input mutated during merge: {expected.directory}")


def _parse_workload(text: str) -> WorkloadSpec:
    try:
        name, remainder = text.split("=", 1)
        directory_text, weight_text = remainder.rsplit(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "workload must have the form NAME=/absolute/raw/directory:WEIGHT"
        ) from error
    if not NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            f"invalid workload name {name!r}; use letters, digits, '.', '_' or '-'"
        )
    try:
        directory = _normal_absolute(directory_text, "workload directory")
    except MergeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    try:
        weight = int(weight_text, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"workload weight is not an integer: {weight_text!r}"
        ) from error
    if weight < 1 or weight > 2**31 - 1:
        raise argparse.ArgumentTypeError(
            "workload weight must be between 1 and 2147483647"
        )
    return WorkloadSpec(name=name, directory=directory, weight=weight)


def _validate_sha256(text: str, option: str) -> str:
    if not SHA256_RE.fullmatch(text):
        raise MergeError(f"{option} must be exactly 64 hexadecimal characters")
    return text.lower()


def _file_identity(path: Path, description: str) -> tuple[str, tuple[int, ...]]:
    value = _lstat_without_symlink_components(path, description)
    if not stat.S_ISREG(value.st_mode):
        raise MergeError(f"{description} is not a regular file: {path}")
    return _hash_open_regular(path, value, description), _stat_identity(value)


def _tool_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in LOADER_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    return environment


def _bounded_output(data: bytes, limit: int = 4000) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + "\n[output truncated]"
    return text


class ExactDependencyResolver:
    """An exact ldd-compatible resolver used only to bind libLLVM."""

    def __init__(
        self,
        path: Path,
        expected_sha256: str,
        expected_version_sha256: str,
    ) -> None:
        self.path = path
        value = _lstat_without_symlink_components(
            path, "dependency resolver executable"
        )
        if not stat.S_ISREG(value.st_mode):
            raise MergeError(f"dependency resolver is not a regular file: {path}")
        if not os.access(path, os.X_OK):
            raise MergeError(f"dependency resolver is not executable: {path}")
        digest = _hash_open_regular(path, value, "dependency resolver executable")
        if digest != expected_sha256:
            raise MergeError(
                "dependency resolver executable SHA-256 mismatch: "
                f"expected {expected_sha256}, found {digest}"
            )
        self._sha256 = digest
        self._stat_identity = _stat_identity(value)
        version = self.run(["--version"], "dependency resolver version probe")
        if version.stderr:
            raise MergeError(
                "dependency resolver --version wrote to stderr: "
                + _bounded_output(version.stderr)
            )
        if not version.stdout:
            raise MergeError("dependency resolver --version returned empty stdout")
        version_digest = _sha256_bytes(version.stdout)
        if version_digest != expected_version_sha256:
            raise MergeError(
                "dependency resolver version-output SHA-256 mismatch: "
                f"expected {expected_version_sha256}, found {version_digest}"
            )
        try:
            version_stdout = version.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise MergeError(
                "dependency resolver --version stdout is not UTF-8"
            ) from error
        self.identity = ResolverIdentity(
            path=path,
            sha256=digest,
            stat_identity=self._stat_identity,
            version_stdout=version_stdout,
            version_stdout_sha256=version_digest,
        )

    def assert_unchanged(self) -> None:
        digest, identity = _file_identity(
            self.path, "dependency resolver executable"
        )
        if digest != self._sha256 or identity != self._stat_identity:
            raise MergeError("dependency resolver mutated during profile merge")

    def run(
        self, arguments: Sequence[str], description: str
    ) -> subprocess.CompletedProcess[bytes]:
        self.assert_unchanged()
        command = [os.fspath(self.path), *arguments]
        try:
            result = subprocess.run(
                command,
                env=_tool_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            raise MergeError(f"could not execute {description}: {error}") from error
        self.assert_unchanged()
        if result.returncode != 0:
            raise MergeError(
                f"{description} failed with exit status {result.returncode}\n"
                f"stdout:\n{_bounded_output(result.stdout)}\n"
                f"stderr:\n{_bounded_output(result.stderr)}"
            )
        return result

    def resolve_libllvm(
        self, tool_path: Path, expected_sha256: str
    ) -> tuple[DependencyIdentity, dict[str, Any]]:
        result = self.run(
            [os.fspath(tool_path)], "llvm-profdata dependency resolution"
        )
        if result.stderr:
            raise MergeError(
                "dependency resolver wrote to stderr: "
                + _bounded_output(result.stderr)
            )
        try:
            stdout = result.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise MergeError("dependency resolver stdout is not UTF-8") from error
        libllvm_lines = [line for line in stdout.splitlines() if "libLLVM" in line]
        matches = [LIBLLVM_LINE_RE.fullmatch(line) for line in libllvm_lines]
        matches = [match for match in matches if match is not None]
        if len(libllvm_lines) != 1 or len(matches) != 1:
            raise MergeError(
                "dependency resolver must report exactly one unambiguous "
                f"{LIBLLVM_SONAME} mapping"
            )
        resolved_text = os.path.normpath(matches[0].group(1))
        resolved = _normal_absolute(resolved_text, f"resolved {LIBLLVM_SONAME}")
        if resolved.name != LIBLLVM_SONAME:
            raise MergeError(
                f"resolved dependency is not named {LIBLLVM_SONAME}: {resolved}"
            )
        value = _lstat_without_symlink_components(
            resolved, f"resolved {LIBLLVM_SONAME}"
        )
        if not stat.S_ISREG(value.st_mode):
            raise MergeError(
                f"resolved {LIBLLVM_SONAME} is not a regular file: {resolved}"
            )
        digest = _hash_open_regular(resolved, value, f"resolved {LIBLLVM_SONAME}")
        if digest != expected_sha256:
            raise MergeError(
                f"resolved {LIBLLVM_SONAME} SHA-256 mismatch: "
                f"expected {expected_sha256}, found {digest}"
            )
        dependency = DependencyIdentity(
            path=resolved,
            sha256=digest,
            stat_identity=_stat_identity(value),
        )
        record = {
            "command": [os.fspath(self.path), os.fspath(tool_path)],
            "stdout": stdout,
            "stdout_sha256": _sha256_bytes(result.stdout),
            "output_policy": {
                "required_soname": LIBLLVM_SONAME,
                "required_exact_mappings": 1,
                "stderr": "must be empty",
            },
        }
        return dependency, record


class ExactTool:
    def __init__(
        self,
        path: Path,
        expected_sha256: str,
        expected_version_sha256: str,
    ) -> None:
        self.path = path
        self._dependency: DependencyIdentity | None = None
        value = _lstat_without_symlink_components(path, "llvm-profdata executable")
        if not stat.S_ISREG(value.st_mode):
            raise MergeError(f"llvm-profdata is not a regular file: {path}")
        if not os.access(path, os.X_OK):
            raise MergeError(f"llvm-profdata is not executable: {path}")
        digest = _hash_open_regular(path, value, "llvm-profdata executable")
        if digest != expected_sha256:
            raise MergeError(
                "llvm-profdata executable SHA-256 mismatch: "
                f"expected {expected_sha256}, found {digest}"
            )
        self._sha256 = digest
        self._stat_identity = _stat_identity(value)
        version = self.run(
            ["merge", "--version"], "llvm-profdata version probe"
        )
        if version.stderr:
            raise MergeError(
                "llvm-profdata merge --version wrote to stderr: "
                + _bounded_output(version.stderr)
            )
        if not version.stdout:
            raise MergeError("llvm-profdata merge --version returned empty stdout")
        version_digest = _sha256_bytes(version.stdout)
        if version_digest != expected_version_sha256:
            raise MergeError(
                "llvm-profdata version-output SHA-256 mismatch: "
                f"expected {expected_version_sha256}, found {version_digest}"
            )
        try:
            version_stdout = version.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise MergeError(
                "llvm-profdata merge --version stdout is not UTF-8"
            ) from error
        self.identity = ToolIdentity(
            path=path,
            sha256=digest,
            stat_identity=self._stat_identity,
            version_stdout=version_stdout,
            version_stdout_sha256=version_digest,
        )

    def bind_dependency(self, dependency: DependencyIdentity) -> None:
        if self._dependency is not None:
            raise MergeError("llvm-profdata dependency was bound more than once")
        self._dependency = dependency
        self.assert_dependency_unchanged()

    def assert_dependency_unchanged(self) -> None:
        if self._dependency is None:
            return
        digest, identity = _file_identity(
            self._dependency.path, f"resolved {LIBLLVM_SONAME}"
        )
        if (
            digest != self._dependency.sha256
            or identity != self._dependency.stat_identity
        ):
            raise MergeError(f"resolved {LIBLLVM_SONAME} mutated during profile merge")

    def assert_unchanged(self) -> None:
        digest, identity = _file_identity(self.path, "llvm-profdata executable")
        if digest != self._sha256 or identity != self._stat_identity:
            raise MergeError("llvm-profdata executable mutated during profile merge")

    def run(
        self, arguments: Sequence[str], description: str
    ) -> subprocess.CompletedProcess[bytes]:
        if hasattr(self, "_sha256"):
            self.assert_unchanged()
            self.assert_dependency_unchanged()
        command = [os.fspath(self.path), *arguments]
        try:
            result = subprocess.run(
                command,
                env=_tool_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            raise MergeError(f"could not execute {description}: {error}") from error
        if hasattr(self, "_sha256"):
            self.assert_unchanged()
            self.assert_dependency_unchanged()
        if result.returncode != 0:
            raise MergeError(
                f"{description} failed with exit status {result.returncode}\n"
                f"stdout:\n{_bounded_output(result.stdout)}\n"
                f"stderr:\n{_bounded_output(result.stderr)}"
            )
        return result


def _generated_profile(path: Path, description: str) -> dict[str, Any]:
    try:
        value = os.lstat(path)
    except OSError as error:
        raise MergeError(f"{description} was not produced: {path}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise MergeError(f"{description} is not a regular file: {path}")
    if value.st_size == 0:
        raise MergeError(f"{description} is empty: {path}")
    digest = _hash_open_regular(path, value, description)
    return {
        "size": value.st_size,
        "sha256": digest,
        "stat_identity": _stat_identity(value),
    }


def _profile_summary(
    tool: ExactTool,
    path: Path,
    description: str,
    recorded_profile_path: str,
) -> dict[str, Any]:
    before = _generated_profile(path, description)
    result = tool.run(["show", os.fspath(path)], f"{description} summary")
    after = _generated_profile(path, description)
    if before != after:
        raise MergeError(f"{description} mutated while its summary was read")
    try:
        text = result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise MergeError(f"{description} summary is not UTF-8") from error
    metrics: dict[str, int] = {}
    for line in text.splitlines():
        match = SUMMARY_RE.match(line)
        if match:
            key = match.group(1).lower().replace(" ", "_")
            metrics[key] = int(match.group(2), 10)
    required = {
        "total_functions",
        "maximum_function_count",
        "maximum_internal_block_count",
        "total_count",
    }
    if not required.issubset(metrics):
        missing = ", ".join(sorted(required - metrics.keys()))
        raise MergeError(f"{description} summary is missing fields: {missing}")
    if metrics["total_functions"] <= 0:
        raise MergeError(f"{description} summary contains no functions")
    if max(
        metrics["maximum_function_count"],
        metrics["maximum_internal_block_count"],
    ) <= 0:
        raise MergeError(f"{description} summary contains no nonzero counts")
    if metrics["total_count"] <= 0:
        raise MergeError(f"{description} summary has a nonpositive total count")
    return {
        "command": [os.fspath(tool.path), "show", recorded_profile_path],
        "stdout_sha256": _sha256_bytes(result.stdout),
        "stdout": text,
        "metrics": metrics,
    }


def _recorded_raw_command(
    tool: ExactTool, snapshot: Snapshot, workload_name: str
) -> list[str]:
    raw_paths = [
        os.fspath(Path(snapshot.directory) / file_record["path"])
        for file_record in snapshot.files
    ]
    return [
        os.fspath(tool.path),
        "merge",
        "--failure-mode=any",
        "--instr",
        "-o",
        f"{{staging}}/.workloads/{workload_name}.profdata",
        *raw_paths,
    ]


def _raw_arguments(snapshot: Snapshot, output: Path) -> list[str]:
    return [
        "merge",
        "--failure-mode=any",
        "--instr",
        "-o",
        os.fspath(output),
        *[
            os.fspath(Path(snapshot.directory) / file_record["path"])
            for file_record in snapshot.files
        ],
    ]


def _remove_private_staging(path: Path) -> None:
    if not path.exists():
        return
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.chmod(Path(root) / name, 0o600, follow_symlinks=False)
            except OSError:
                pass
        for name in directories:
            try:
                os.chmod(Path(root) / name, 0o700, follow_symlinks=False)
            except OSError:
                pass
    try:
        os.chmod(path, 0o700, follow_symlinks=False)
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _renameat2() -> Any:
    if sys.platform != "linux":
        raise MergeError(
            "atomic publication requires Linux renameat2(RENAME_NOREPLACE)"
        )
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = library.renameat2
    except AttributeError as error:
        raise MergeError(
            "libc does not expose renameat2; refusing non-atomic publication"
        ) from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    return function


def _publish_noreplace(staging: Path, output: Path) -> None:
    if staging.parent != output.parent:
        raise MergeError("publication staging and output are not sibling paths")
    parent_stat = _lstat_without_symlink_components(output.parent, "output parent")
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise MergeError(f"output parent is not a directory: {output.parent}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(output.parent, flags)
    except OSError as error:
        raise MergeError(f"cannot open output parent {output.parent}: {error}") from error
    try:
        if _stat_identity(os.fstat(parent_descriptor)) != _stat_identity(parent_stat):
            raise MergeError("output parent mutated while being opened")
        function = _renameat2()
        ctypes.set_errno(0)
        result = function(
            parent_descriptor,
            os.fsencode(staging.name),
            parent_descriptor,
            os.fsencode(output.name),
            RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise MergeError(f"output path appeared during merge: {output}")
            if error_number in {
                errno.EINVAL,
                errno.ENOSYS,
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }:
                raise MergeError(
                    "renameat2(RENAME_NOREPLACE) is unavailable for the output "
                    f"filesystem: {os.strerror(error_number)}"
                )
            raise MergeError(
                f"atomic publication failed: {os.strerror(error_number)}"
            )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def merge_profiles(
    tool: ExactTool,
    resolver: ExactDependencyResolver,
    dependency: DependencyIdentity,
    dependency_resolution: dict[str, Any],
    workloads: Sequence[WorkloadSpec],
    output_directory: Path,
) -> dict[str, Any]:
    if len(workloads) < 2:
        raise MergeError("at least two --workload arguments are required")
    names = [workload.name for workload in workloads]
    if len(set(names)) != len(names):
        raise MergeError("workload names must be unique")
    directories = [os.fspath(workload.directory) for workload in workloads]
    if len(set(directories)) != len(directories):
        raise MergeError("workload directories must be unique")

    output_parent = output_directory.parent
    parent_stat = _lstat_without_symlink_components(output_parent, "output parent")
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise MergeError(f"output parent is not a directory: {output_parent}")
    _renameat2()
    try:
        os.lstat(output_directory)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise MergeError(f"cannot inspect output path {output_directory}: {error}") from error
    else:
        raise MergeError(f"output path already exists: {output_directory}")

    ordered = sorted(workloads, key=lambda workload: workload.name)
    snapshots: dict[str, Snapshot] = {}
    physical_inputs: set[tuple[int, int]] = set()
    for workload in ordered:
        snapshot = _snapshot_workload(workload.directory)
        for file_record in snapshot.files:
            identity = (file_record["device"], file_record["inode"])
            if identity in physical_inputs:
                raise MergeError(
                    "a raw profile is reachable through more than one workload or path: "
                    f"{workload.directory / file_record['path']}"
                )
            physical_inputs.add(identity)
        snapshots[workload.name] = snapshot

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.tmp-", dir=os.fspath(output_parent)
        )
    )
    try:
        work_directory = staging / ".workloads"
        work_directory.mkdir(mode=0o700)
        workload_records: list[dict[str, Any]] = []
        intermediates: dict[str, Path] = {}
        intermediate_identities: dict[str, dict[str, Any]] = {}
        expected_weighted_total_count = 0

        for workload in ordered:
            snapshot = snapshots[workload.name]
            intermediate = work_directory / f"{workload.name}.profdata"
            tool.run(
                _raw_arguments(snapshot, intermediate),
                f"raw-profile merge for workload {workload.name}",
            )
            _assert_unchanged(snapshot)
            summary = _profile_summary(
                tool,
                intermediate,
                f"workload {workload.name} profile",
                f"{{staging}}/.workloads/{workload.name}.profdata",
            )
            profile_identity = _generated_profile(
                intermediate, f"workload {workload.name} profile"
            )
            intermediates[workload.name] = intermediate
            intermediate_identities[workload.name] = profile_identity
            aggregate_total_count = summary["metrics"]["total_count"]
            weighted_total_count = workload.weight * aggregate_total_count
            expected_weighted_total_count += weighted_total_count
            workload_records.append(
                {
                    "name": workload.name,
                    "weight": workload.weight,
                    "aggregate_total_count": aggregate_total_count,
                    "weighted_total_count": weighted_total_count,
                    "raw_inventory": snapshot.manifest(),
                    "merge_command": _recorded_raw_command(
                        tool, snapshot, workload.name
                    ),
                    "merged_profile": {
                        "size": profile_identity["size"],
                        "sha256": profile_identity["sha256"],
                    },
                    "summary": summary,
                }
            )

        combined = staging / PROFILE_NAME
        combined_arguments = [
            "merge",
            "--failure-mode=any",
            "--instr",
            "-o",
            os.fspath(combined),
        ]
        recorded_combined_command = [
            os.fspath(tool.path),
            "merge",
            "--failure-mode=any",
            "--instr",
            "-o",
            f"{{bundle}}/{PROFILE_NAME}",
        ]
        for workload in ordered:
            combined_arguments.append(
                f"--weighted-input={workload.weight},{intermediates[workload.name]}"
            )
            recorded_combined_command.append(
                f"--weighted-input={workload.weight},"
                f"{{staging}}/.workloads/{workload.name}.profdata"
            )
        tool.run(combined_arguments, "weighted profile merge")
        for workload in ordered:
            current = _generated_profile(
                intermediates[workload.name],
                f"workload {workload.name} profile",
            )
            if current != intermediate_identities[workload.name]:
                raise MergeError(
                    f"workload {workload.name} profile mutated during weighted merge"
                )
        combined_summary = _profile_summary(
            tool,
            combined,
            "combined profile",
            f"{{bundle}}/{PROFILE_NAME}",
        )
        combined_identity = _generated_profile(combined, "combined profile")
        observed_total_count = combined_summary["metrics"]["total_count"]
        if observed_total_count != expected_weighted_total_count:
            raise MergeError(
                "combined profile total count does not equal the explicit weighted "
                f"sum: expected {expected_weighted_total_count}, "
                f"found {observed_total_count}"
            )

        manifest = {
            "schema": SCHEMA,
            "path_tokens": {
                "{bundle}": os.fspath(output_directory),
                "{staging}/.workloads": (
                    "private intermediates removed before publication"
                ),
            },
            "subprocess_environment_policy": {
                "set": {"LANG": "C", "LC_ALL": "C"},
                "unset": list(LOADER_ENVIRONMENT_VARIABLES),
                "inherit_all_other_variables": True,
            },
            "tool": {
                "path": os.fspath(tool.identity.path),
                "sha256": tool.identity.sha256,
                "version_command": [
                    os.fspath(tool.path),
                    "merge",
                    "--version",
                ],
                "version_stdout": tool.identity.version_stdout,
                "version_stdout_sha256": tool.identity.version_stdout_sha256,
            },
            "runtime_dependency_binding": {
                "resolver": {
                    "path": os.fspath(resolver.identity.path),
                    "sha256": resolver.identity.sha256,
                    "version_command": [
                        os.fspath(resolver.path),
                        "--version",
                    ],
                    "version_stdout": resolver.identity.version_stdout,
                    "version_stdout_sha256": (
                        resolver.identity.version_stdout_sha256
                    ),
                },
                "resolution": dependency_resolution,
                "libllvm": {
                    "soname": LIBLLVM_SONAME,
                    "resolved_path": os.fspath(dependency.path),
                    "sha256": dependency.sha256,
                },
                "scope": (
                    "libLLVM.so.22.1 is resolved and hashed; other dynamic "
                    "dependencies remain a campaign-manifest responsibility"
                ),
            },
            "workloads": workload_records,
            "combined": {
                "merge_command": recorded_combined_command,
                "summary": combined_summary,
                "weighting_check": {
                    "expected_weighted_total_count": (
                        expected_weighted_total_count
                    ),
                    "observed_total_count": observed_total_count,
                    "equal": True,
                },
                "profile": {
                    "path": PROFILE_NAME,
                    "mode": "0444",
                    "size": combined_identity["size"],
                    "sha256": combined_identity["sha256"],
                },
            },
            "publication": {
                "directory": os.fspath(output_directory),
                "directory_mode": "0555",
                "manifest": MANIFEST_NAME,
                "manifest_mode": "0444",
                "method": (
                    "parent-dirfd-relative Linux "
                    "renameat2(RENAME_NOREPLACE)"
                ),
            },
        }
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.chmod(combined, 0o444)
        os.chmod(manifest_path, 0o444)
        _fsync_file(combined)
        _fsync_file(manifest_path)

        # This is deliberately the last expensive validation before the
        # publication rename.  It closes the window in which a raw input or
        # the exact merge tool could change after producing the manifest.
        for snapshot in snapshots.values():
            _assert_unchanged(snapshot)
        tool.assert_unchanged()
        tool.assert_dependency_unchanged()
        resolver.assert_unchanged()
        for workload in ordered:
            current = _generated_profile(
                intermediates[workload.name],
                f"workload {workload.name} profile",
            )
            if current != intermediate_identities[workload.name]:
                raise MergeError(
                    f"workload {workload.name} profile mutated before publication"
                )
        final_combined = _generated_profile(combined, "combined profile")
        if (
            final_combined["size"] != combined_identity["size"]
            or final_combined["sha256"] != combined_identity["sha256"]
        ):
            raise MergeError("combined profile mutated before publication")

        shutil.rmtree(work_directory)
        _fsync_directory(staging)
        os.chmod(staging, 0o555)
        _publish_noreplace(staging, output_directory)
        staging = Path()
        return manifest
    finally:
        if staging != Path():
            _remove_private_staging(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two or more explicit raw-profile workloads into an atomic, "
            "read-only weighted LLVM PGO profile bundle."
        )
    )
    parser.add_argument(
        "--llvm-profdata",
        required=True,
        metavar="ABSOLUTE_PATH",
        help="exact absolute llvm-profdata executable (symlinks are rejected)",
    )
    parser.add_argument(
        "--expect-tool-sha256",
        required=True,
        metavar="HEX",
        help="required SHA-256 of the llvm-profdata executable",
    )
    parser.add_argument(
        "--expect-version-sha256",
        required=True,
        metavar="HEX",
        help=(
            "required SHA-256 of exact llvm-profdata merge --version stdout bytes"
        ),
    )
    parser.add_argument(
        "--dependency-resolver",
        required=True,
        metavar="ABSOLUTE_PATH",
        help=(
            "exact ldd-compatible resolver used to bind the loaded "
            f"{LIBLLVM_SONAME}"
        ),
    )
    parser.add_argument(
        "--expect-resolver-sha256",
        required=True,
        metavar="HEX",
        help="required SHA-256 of the dependency resolver executable",
    )
    parser.add_argument(
        "--expect-resolver-version-sha256",
        required=True,
        metavar="HEX",
        help="required SHA-256 of exact resolver --version stdout bytes",
    )
    parser.add_argument(
        "--expect-libllvm-sha256",
        required=True,
        metavar="HEX",
        help=f"required SHA-256 of the resolved {LIBLLVM_SONAME}",
    )
    parser.add_argument(
        "--workload",
        action="append",
        required=True,
        type=_parse_workload,
        metavar="NAME=/ABSOLUTE/RAW/DIR:WEIGHT",
        help="explicit workload raw-profile directory and positive integer weight",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="ABSOLUTE_PATH",
        help="new bundle directory to publish atomically; it must not exist",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    try:
        tool_path = _normal_absolute(options.llvm_profdata, "llvm-profdata")
        resolver_path = _normal_absolute(
            options.dependency_resolver, "dependency resolver"
        )
        output_directory = _normal_absolute(options.output_dir, "output directory")
        expected_tool_sha256 = _validate_sha256(
            options.expect_tool_sha256, "--expect-tool-sha256"
        )
        expected_version_sha256 = _validate_sha256(
            options.expect_version_sha256, "--expect-version-sha256"
        )
        expected_resolver_sha256 = _validate_sha256(
            options.expect_resolver_sha256, "--expect-resolver-sha256"
        )
        expected_resolver_version_sha256 = _validate_sha256(
            options.expect_resolver_version_sha256,
            "--expect-resolver-version-sha256",
        )
        expected_libllvm_sha256 = _validate_sha256(
            options.expect_libllvm_sha256, "--expect-libllvm-sha256"
        )
        tool = ExactTool(
            tool_path, expected_tool_sha256, expected_version_sha256
        )
        resolver = ExactDependencyResolver(
            resolver_path,
            expected_resolver_sha256,
            expected_resolver_version_sha256,
        )
        tool.assert_unchanged()
        dependency, dependency_resolution = resolver.resolve_libllvm(
            tool_path, expected_libllvm_sha256
        )
        tool.assert_unchanged()
        tool.bind_dependency(dependency)
        manifest = merge_profiles(
            tool,
            resolver,
            dependency,
            dependency_resolution,
            options.workload,
            output_directory,
        )
    except (MergeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"published PGO profile bundle: {output_directory}")
    print(f"profile SHA-256: {manifest['combined']['profile']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
