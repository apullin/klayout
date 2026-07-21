#!/usr/bin/env python3
"""Deterministic provenance for the code used by a KLayout process.

The small ``klayout`` front-end is not a sufficient runtime identity by
itself.  This module fingerprints the executable, the ELF dependency closure
selected by the dynamic loader, KLayout's runtime plugin libraries, ordered
``LD_PRELOAD`` entries, and environment controls that can change execution.

Filesystem locations are retained in diagnostic metadata but excluded from
the canonical code identity.  Environment search paths are strings, not
implicit directory-content inventories.  A caller that normalizes such a
string or attests to external search-path state must name that policy
explicitly so the policy itself enters the canonical identity.

The executable, environment, and ``ldd`` program must be trusted.  Like
``ldd`` itself, this collector is not a sandbox for inspecting hostile ELF
files.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
from typing import Any


FORMAT = "klayout-runtime-provenance"
FORMAT_VERSION = 2
IDENTITY_SCHEMA = "klayout-runtime-identity-v2"
_HASH_CHUNK_BYTES = 1024 * 1024
_PRELOAD_IDENTITY_SENTINEL = "<represented-by-ordered-content-identity>"
_VIRTUAL_ELF_OBJECTS = frozenset({"linux-vdso.so.1", "linux-gate.so.1"})
_LDD_ADDRESS_RE = re.compile(r"^(?P<value>.+?)\s+\(0x[0-9a-fA-F]+\)\s*$")
_DEFAULT_NORMALIZATION_POLICY_ID = "verbatim-environment-v1"
_DEFAULT_PLUGIN_SEARCH_POLICY_ID = "module-adjacent-klayout-plugins-v2"
_LANGUAGE_TREE_POLICY_ID = "ruby-python-loadable-tree-v1"
_SENSITIVE_ENVIRONMENT_FRAGMENTS = (
    "TOKEN",
    "PASS",
    "SECRET",
    "KEY",
    "CREDENTIAL",
)

EnvironmentNormalizer = Callable[[str, str], str | None]


class RuntimeProvenanceError(ValueError):
    """Raised when a complete, trustworthy runtime identity cannot be made."""


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value with a stable, whitespace-free serialization."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise RuntimeProvenanceError(
            f"{description} does not resolve to a file: {path}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeProvenanceError(
            f"{description} is not a regular file: {resolved}"
        )
    return resolved


def _file_record(path: Path, **extra: object) -> dict[str, object]:
    """Hash a regular file and detect replacement during the read."""

    resolved = _regular_file(path, description="runtime component")
    before = resolved.stat()
    sha256 = _sha256_file(resolved)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeProvenanceError(
            f"runtime component changed while being fingerprinted: {resolved}"
        )
    return {
        "resolved_path": str(resolved),
        "size_bytes": after.st_size,
        "sha256": sha256,
        "filesystem_identity": {
            "device": after.st_dev,
            "inode": after.st_ino,
            "mtime_ns": after.st_mtime_ns,
        },
        **extra,
    }


def _resolve_executable(
    command: str | os.PathLike[str], environment: Mapping[str, str], cwd: Path
) -> Path:
    supplied = os.fspath(command)
    if os.sep in supplied or (os.altsep and os.altsep in supplied):
        candidate = Path(supplied)
        if not candidate.is_absolute():
            candidate = cwd / candidate
    else:
        raw_path = environment.get("PATH", os.defpath)
        path_entries = []
        for entry in raw_path.split(os.pathsep):
            directory = Path(entry) if entry else cwd
            if not directory.is_absolute():
                directory = cwd / directory
            path_entries.append(str(directory))
        resolved = shutil.which(
            supplied, path=os.pathsep.join(path_entries)
        )
        if resolved is None:
            raise RuntimeProvenanceError(
                f"KLayout executable was not found through PATH: {supplied}"
            )
        candidate = Path(resolved)
    executable = _regular_file(candidate, description="KLayout executable")
    if not os.access(executable, os.X_OK):
        raise RuntimeProvenanceError(
            f"KLayout executable is not executable: {executable}"
        )
    return executable


def _resolve_ldd(ldd_path: str | os.PathLike[str] | None) -> Path:
    if ldd_path is not None:
        candidate = Path(ldd_path)
    else:
        candidate = next(
            (
                path
                for path in (Path("/usr/bin/ldd"), Path("/bin/ldd"))
                if path.exists()
            ),
            Path(shutil.which("ldd") or "ldd"),
        )
    resolved = _regular_file(candidate, description="trusted ldd")
    if not os.access(resolved, os.X_OK):
        raise RuntimeProvenanceError(f"trusted ldd is not executable: {resolved}")
    return resolved


def _without_load_address(value: str) -> str:
    match = _LDD_ADDRESS_RE.match(value.strip())
    return match.group("value").strip() if match else value.strip()


def _parse_ldd_output(output: str) -> list[dict[str, object]]:
    """Parse GNU/glibc-style ldd output and reject ambiguous lines."""

    entries: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=>" in line:
            alias_text, selected_text = line.split("=>", 1)
            alias = alias_text.strip()
            selected = _without_load_address(selected_text)
            if not alias:
                raise RuntimeProvenanceError(
                    f"ldd output line {line_number} has an empty library name"
                )
            if selected == "not found":
                raise RuntimeProvenanceError(
                    f"ELF dependency was not found: {alias}"
                )
            if not selected:
                raise RuntimeProvenanceError(
                    f"ldd output line {line_number} has an empty selected path"
                )
            entries.append(
                {
                    "loader_position": len(entries) + 1,
                    "loader_alias": Path(alias).name,
                    "selected_path": selected,
                    "virtual": False,
                }
            )
            continue

        selected = _without_load_address(line)
        if selected in _VIRTUAL_ELF_OBJECTS:
            entries.append(
                {
                    "loader_position": len(entries) + 1,
                    "loader_alias": selected,
                    "selected_path": None,
                    "virtual": True,
                }
            )
            continue
        if selected in {"statically linked", "not a dynamic executable"}:
            raise RuntimeProvenanceError(
                f"KLayout executable has no dynamic ELF closure: {selected}"
            )
        if os.sep in selected or (os.altsep and os.altsep in selected):
            entries.append(
                {
                    "loader_position": len(entries) + 1,
                    "loader_alias": Path(selected).name,
                    "selected_path": selected,
                    "virtual": False,
                }
            )
            continue
        raise RuntimeProvenanceError(
            f"unrecognized ldd output at line {line_number}: {raw_line!r}"
        )
    if not entries:
        raise RuntimeProvenanceError("ldd returned an empty dependency listing")
    return entries


def _run_ldd(
    subject: Path,
    probe_role: str,
    environment: Mapping[str, str],
    cwd: Path,
    ldd_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    probe_environment = dict(environment)
    # Dependency selection still uses the supplied loader environment.  A C
    # locale only makes the diagnostic text parseable and deterministic.
    probe_environment["LC_ALL"] = "C"
    probe_environment["LANG"] = "C"
    command = [str(ldd_path), str(subject)]
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=probe_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeProvenanceError(
            f"ldd failed with exit status {completed.returncode}: {detail}"
        )
    entries = _parse_ldd_output(completed.stdout)
    for entry in entries:
        entry["probe_role"] = probe_role
    return entries, {
        "probe_role": probe_role,
        "subject_path": str(subject),
        "command": command,
        "stderr": completed.stderr,
        "entry_count": len(entries),
    }


def _preload_tokens(value: str | None) -> list[str]:
    """Split LD_PRELOAD according to the ELF loader's space/colon grammar."""

    if not value:
        return []
    return [token for token in re.split(r"[:\s]+", value) if token]


def _path_like_preload(token: str) -> bool:
    return os.sep in token or bool(os.altsep and os.altsep in token)


def _resolve_explicit_preloads(tokens: Sequence[str]) -> dict[int, Path]:
    resolved: dict[int, Path] = {}
    for position, token in enumerate(tokens, start=1):
        if not _path_like_preload(token):
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            raise RuntimeProvenanceError(
                f"LD_PRELOAD entry {token!r} contains a path but is not "
                "absolute; use an absolute path so its contents can be "
                "fingerprinted unambiguously"
            )
        resolved[position] = _regular_file(
            candidate, description=f"LD_PRELOAD entry {token!r}"
        )
    return resolved


def _materialize_ldd_entries(
    parsed: Sequence[Mapping[str, object]],
    cwd: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resolve, hash, and deduplicate ldd-selected regular files."""

    grouped: dict[Path, dict[str, Any]] = {}
    virtual: list[dict[str, object]] = []
    for entry in parsed:
        alias = str(entry["loader_alias"])
        position = int(entry["loader_position"])
        probe_role = str(entry["probe_role"])
        if bool(entry["virtual"]):
            virtual.append(
                {
                    "probe_role": probe_role,
                    "loader_alias": alias,
                    "loader_position": position,
                    "kind": "virtual_elf_object",
                }
            )
            continue
        raw_path = str(entry["selected_path"])
        selected_path = Path(raw_path)
        if not selected_path.is_absolute():
            selected_path = cwd / selected_path
        selected = _regular_file(
            selected_path, description=f"ldd-selected dependency {alias!r}"
        )
        group = grouped.setdefault(
            selected,
            {"selections": {}, "selected_as": set()},
        )
        selection = group["selections"].setdefault(
            probe_role, {"aliases": set(), "positions": []}
        )
        selection["aliases"].add(alias)
        selection["positions"].append(position)
        group["selected_as"].add(raw_path)

    loaded: list[dict[str, object]] = []
    for target, group in grouped.items():
        selections = [
            {
                "probe_role": probe_role,
                "loader_aliases": sorted(selection["aliases"]),
                "loader_positions": sorted(selection["positions"]),
            }
            for probe_role, selection in sorted(group["selections"].items())
        ]
        loaded.append(
            _file_record(
                target,
                selections=selections,
                selected_as=sorted(group["selected_as"]),
            )
        )
    loaded.sort(
        key=lambda record: (
            record["selections"][0]["probe_role"],
            min(record["selections"][0]["loader_positions"]),
            tuple(record["selections"][0]["loader_aliases"]),
        )
    )
    virtual.sort(
        key=lambda record: (
            str(record["probe_role"]), int(record["loader_position"])
        )
    )
    return loaded, virtual


def _resolve_preloads(
    tokens: Sequence[str],
    explicit: Mapping[int, Path],
    loaded: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_path: dict[Path, Mapping[str, object]] = {}
    by_alias: dict[str, set[Path]] = defaultdict(set)
    for record in loaded:
        target = Path(str(record["resolved_path"])).resolve()
        executable_selections = [
            selection
            for selection in record["selections"]
            if selection["probe_role"] == "executable"
        ]
        if not executable_selections:
            continue
        by_path[target] = record
        by_alias[target.name].add(target)
        for selection in executable_selections:
            for alias in selection["loader_aliases"]:
                by_alias[str(alias)].add(target)

    result: list[dict[str, object]] = []
    for position, token in enumerate(tokens, start=1):
        if position in explicit:
            target = explicit[position]
            if target not in by_path:
                raise RuntimeProvenanceError(
                    f"LD_PRELOAD entry {token!r} resolved to {target}, but ldd "
                    "did not prove that the loader selected it"
                )
        else:
            candidates = by_alias.get(token, set())
            if len(candidates) != 1:
                qualifier = "not found" if not candidates else "ambiguous"
                raise RuntimeProvenanceError(
                    f"bare LD_PRELOAD entry {token!r} is {qualifier} in the "
                    "actual ldd selection; use an absolute path"
                )
            target = next(iter(candidates))
        selected = by_path[target]
        executable_aliases = sorted(
            {
                str(alias)
                for selection in selected["selections"]
                if selection["probe_role"] == "executable"
                for alias in selection["loader_aliases"]
            }
        )
        result.append(
            _file_record(
                target,
                loader_position=position,
                supplied_as=token,
                loader_aliases=executable_aliases,
            )
        )
    return result


def _effective_klayout_home(
    environment: Mapping[str, str], cwd: Path
) -> Path | None:
    if "KLAYOUT_HOME" in environment:
        configured = environment["KLAYOUT_HOME"]
        if not configured:
            return None
        candidate = Path(configured)
    else:
        home = environment.get("HOME")
        if not home:
            raise RuntimeProvenanceError(
                "KLAYOUT_HOME and HOME are both unset; the implicit KLayout "
                "plugin search root cannot be proven"
            )
        candidate = Path(home) / ".klayout"
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve()


def _directory_has_entries(path: Path, *, description: str) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        raise RuntimeProvenanceError(f"{description} is not a directory: {path}")
    try:
        return next(path.iterdir(), None) is not None
    except OSError as exc:
        raise RuntimeProvenanceError(f"cannot inspect {description}: {path}") from exc


def _assert_install_expansions_are_empty(executable_dir: Path) -> None:
    """Reject KLayout install search expansions we cannot select exactly.

    ``lay::ApplicationBase`` expands each KLayout path through an architecture
    directory and through versioned Salt package directories before db/lay
    plugin discovery.  The executable does not expose the exact architecture,
    version, and active Salt package order through this API.  Default mode
    therefore accepts only an empty Salt tree and no architecture-child plugin
    objects.  Callers with either kind of state must provide the exhaustive
    selected roots and a named plugin policy.
    """

    salt_root = executable_dir / "salt"
    if _directory_has_entries(salt_root, description="installation Salt root"):
        raise RuntimeProvenanceError(
            "installation Salt state may add versioned runtime plugins; supply "
            "exhaustive plugin_directories and plugin_search_policy_id"
        )

    try:
        children = sorted(executable_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise RuntimeProvenanceError(
            f"cannot inspect KLayout installation path: {executable_dir}"
        ) from exc
    for child in children:
        if child.name in {"db_plugins", "lay_plugins", "salt"} or not child.is_dir():
            continue
        for plugin_kind in ("db_plugins", "lay_plugins"):
            candidate = child / plugin_kind
            if not candidate.exists():
                continue
            if not candidate.is_dir():
                raise RuntimeProvenanceError(
                    "installation architecture plugin root is not a directory: "
                    f"{candidate}"
                )
            try:
                has_plugin_objects = any(candidate.rglob("*.so*"))
            except OSError as exc:
                raise RuntimeProvenanceError(
                    f"cannot inspect architecture plugin root: {candidate}"
                ) from exc
            if has_plugin_objects:
                raise RuntimeProvenanceError(
                    "installation architecture plugin state cannot be selected "
                    "without KLayout's exact architecture; supply exhaustive "
                    "plugin_directories and plugin_search_policy_id"
                )


def _default_plugin_directories(
    executable_dir: Path,
    loaded_dependencies: Sequence[Mapping[str, object]],
    environment: Mapping[str, str],
    cwd: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Cover only plugin roots whose KLayout search semantics are provable."""

    if environment.get("KLAYOUT_PATH"):
        raise RuntimeProvenanceError(
            "nonempty KLAYOUT_PATH enables architecture and Salt plugin search; "
            "supply exhaustive plugin_directories and plugin_search_policy_id"
        )
    klayout_home = _effective_klayout_home(environment, cwd)
    if klayout_home is not None and klayout_home.exists():
        if not klayout_home.is_dir():
            raise RuntimeProvenanceError(
                f"effective KLAYOUT_HOME is not a directory: {klayout_home}"
            )
        has_external_state = _directory_has_entries(
            klayout_home, description="effective KLAYOUT_HOME"
        )
        if has_external_state:
            raise RuntimeProvenanceError(
                "nonempty KLAYOUT_HOME may add root, architecture, or Salt "
                "plugins; supply exhaustive plugin_directories and "
                "plugin_search_policy_id"
            )

    roots_to_consider: dict[str, Path] = {}
    # With KLAYOUT_PATH unset, KLayout adds its installation path.  The
    # executable directory is the deterministic approximation available to a
    # launcher; module-adjacent roots below cover split lib/bin installs.
    if "KLAYOUT_PATH" not in environment:
        _assert_install_expansions_are_empty(executable_dir)
        roots_to_consider["install/db_plugins"] = executable_dir / "db_plugins"
        roots_to_consider["install/lay_plugins"] = executable_dir / "lay_plugins"
    for record in loaded_dependencies:
        aliases = {
            str(alias)
            for selection in record["selections"]
            for alias in selection["loader_aliases"]
        }
        if any(alias.startswith("libklayout_db.so") for alias in aliases):
            roots_to_consider["db-module/db_plugins"] = (
                Path(str(record["resolved_path"])).parent / "db_plugins"
            )
        if any(alias.startswith("libklayout_lay.so") for alias in aliases):
            roots_to_consider["lay-module/lay_plugins"] = (
                Path(str(record["resolved_path"])).parent / "lay_plugins"
            )

    directories: dict[str, Path] = {}
    for role, candidate in sorted(roots_to_consider.items()):
        if candidate.is_dir():
            directories[role] = candidate
    coverage = {
        "mode": "module_adjacent_with_empty_install_expansions",
        "effective_klayout_home": (
            str(klayout_home) if klayout_home is not None else None
        ),
        "klayout_home_state": "absent_or_empty",
        "klayout_path_state": "unset_or_empty",
        "install_architecture_plugin_state": "absent",
        "install_salt_state": "absent_or_empty",
    }
    return directories, coverage


def _plugin_records(
    executable_dir: Path,
    loaded_dependencies: Sequence[Mapping[str, object]],
    environment: Mapping[str, str],
    cwd: Path,
    plugin_directories: Mapping[str, str | os.PathLike[str]] | None,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    if plugin_directories is None:
        roots, coverage = _default_plugin_directories(
            executable_dir, loaded_dependencies, environment, cwd
        )
    else:
        roots = {}
        coverage = {
            "mode": "caller_supplied_exhaustive_roots",
            "root_roles": sorted(plugin_directories),
        }
        for role, configured in plugin_directories.items():
            if not role or not isinstance(role, str):
                raise RuntimeProvenanceError(
                    "plugin directory roles must be nonempty strings"
                )
            candidate = Path(configured)
            if not candidate.is_absolute():
                candidate = executable_dir / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise RuntimeProvenanceError(
                    f"plugin directory {role!r} is unavailable: {candidate}"
                ) from exc
            if not resolved.is_dir():
                raise RuntimeProvenanceError(
                    f"plugin directory {role!r} is not a directory: {resolved}"
                )
            roots[role] = candidate

    aliases_by_target: dict[Path, set[str]] = defaultdict(set)
    root_metadata: list[dict[str, object]] = []
    for role, root_path in sorted(roots.items()):
        resolved_root = root_path.resolve(strict=True)
        root_metadata.append(
            {
                "role": role,
                "path": str(root_path),
                "resolved_path": str(resolved_root),
            }
        )
        for alias in sorted(resolved_root.rglob("*.so*")):
            try:
                target = alias.resolve(strict=True)
            except OSError as exc:
                raise RuntimeProvenanceError(
                    f"runtime plugin alias cannot be resolved: {alias}"
                ) from exc
            if not target.is_file():
                continue
            relative = alias.relative_to(resolved_root).as_posix()
            aliases_by_target[target].add(f"{role}/{relative}")

    plugins = [
        _file_record(
            target,
            relative_aliases=sorted(aliases),
            probe_role="runtime_plugin:" + "|".join(sorted(aliases)),
        )
        for target, aliases in aliases_by_target.items()
    ]
    plugins.sort(key=lambda record: tuple(record["relative_aliases"]))
    return plugins, root_metadata, coverage


def _loader_aliases(record: Mapping[str, object]) -> set[str]:
    return {
        str(alias)
        for selection in record["selections"]
        for alias in selection["loader_aliases"]
    }


def _path_entries(value: str, cwd: Path) -> list[Path]:
    paths: list[Path] = []
    for entry in value.split(os.pathsep):
        candidate = Path(entry) if entry else cwd
        if not candidate.is_absolute():
            candidate = cwd / candidate
        paths.append(candidate)
    return paths


def _ruby_include_paths(value: str, cwd: Path) -> list[Path]:
    try:
        tokens = shlex.split(value)
    except ValueError as exc:
        raise RuntimeProvenanceError(f"cannot parse RUBYOPT: {exc}") from exc
    paths: list[Path] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        include: str | None = None
        if token == "-I":
            index += 1
            if index >= len(tokens):
                raise RuntimeProvenanceError("RUBYOPT ends with an incomplete -I")
            include = tokens[index]
        elif token.startswith("-I") and len(token) > 2:
            include = token[2:]
        if include is not None:
            paths.extend(_path_entries(include, cwd))
        index += 1
    return paths


def _ruby_required_paths(value: str, cwd: Path) -> list[Path]:
    try:
        tokens = shlex.split(value)
    except ValueError as exc:
        raise RuntimeProvenanceError(f"cannot parse RUBYOPT: {exc}") from exc
    required: list[Path] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        requested: str | None = None
        if token in {"-r", "--require"}:
            index += 1
            if index >= len(tokens):
                raise RuntimeProvenanceError(
                    f"RUBYOPT ends with an incomplete {token}"
                )
            requested = tokens[index]
        elif token.startswith("-r") and len(token) > 2:
            requested = token[2:]
        elif token.startswith("--require="):
            requested = token.partition("=")[2]
        if requested and (
            Path(requested).is_absolute()
            or os.sep in requested
            or bool(os.altsep and os.altsep in requested)
        ):
            candidate = Path(requested)
            if not candidate.is_absolute():
                candidate = cwd / candidate
            alternatives = (candidate, Path(f"{candidate}.rb"), Path(f"{candidate}.so"))
            selected = next((path for path in alternatives if path.is_file()), None)
            if selected is None:
                raise RuntimeProvenanceError(
                    f"RUBYOPT required path cannot be resolved: {requested!r}"
                )
            required.append(selected)
        index += 1
    return required


def _language_file_is_loadable(language: str, name: str) -> bool:
    lower = name.lower()
    native_extension = bool(re.search(r"\.so(?:\.|$)", lower))
    if language == "ruby":
        return native_extension or lower.endswith(
            (".rb", ".rbc", ".gemspec", ".bundle")
        )
    return native_extension or lower.endswith(
        (".py", ".pyc", ".pyo", ".pth", ".zip", ".egg", ".whl")
    )


def _language_tree_record(
    language: str, roles: Sequence[str], configured_path: Path
) -> dict[str, object]:
    resolved = configured_path.resolve(strict=True)
    aliases_by_target: dict[Path, set[str]] = defaultdict(set)
    if resolved.is_file():
        if not _language_file_is_loadable(language, resolved.name):
            raise RuntimeProvenanceError(
                f"{language} search-path file is not dynamically loadable: {resolved}"
            )
        aliases_by_target[resolved].add("$ROOT")
    elif resolved.is_dir():
        try:
            entries = sorted(resolved.rglob("*"), key=lambda path: path.as_posix())
        except OSError as exc:
            raise RuntimeProvenanceError(
                f"cannot inventory {language} runtime root: {resolved}"
            ) from exc
        for alias in entries:
            if alias.is_symlink() and alias.resolve(strict=True).is_dir():
                raise RuntimeProvenanceError(
                    f"symlinked directories in {language} runtime roots require "
                    f"explicit expansion: {alias}"
                )
            if not _language_file_is_loadable(language, alias.name):
                continue
            target = _regular_file(
                alias, description=f"{language} dynamically loadable file"
            )
            aliases_by_target[target].add(alias.relative_to(resolved).as_posix())
    else:
        raise RuntimeProvenanceError(
            f"{language} runtime root is neither a file nor directory: {resolved}"
        )

    manifest_entries: list[dict[str, object]] = []
    stability_entries: list[dict[str, object]] = []
    total_bytes = 0
    for target, aliases in aliases_by_target.items():
        record = _file_record(target)
        total_bytes += int(record["size_bytes"])
        manifest_entry = {
            "relative_aliases": sorted(aliases),
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        manifest_entries.append(manifest_entry)
        stability_entries.append(
            {
                **manifest_entry,
                "resolved_path": record["resolved_path"],
                "filesystem_identity": record["filesystem_identity"],
            }
        )
    manifest_entries.sort(key=lambda entry: tuple(entry["relative_aliases"]))
    stability_entries.sort(key=lambda entry: tuple(entry["relative_aliases"]))
    return {
        "language": language,
        "roles": sorted(roles),
        "path": str(configured_path),
        "resolved_path": str(resolved),
        "file_count": len(manifest_entries),
        "total_bytes": total_bytes,
        "manifest_sha256": canonical_json_sha256(manifest_entries),
        "stability_sha256": canonical_json_sha256(stability_entries),
    }


def _language_runtime_roots(
    loaded_dependencies: Sequence[Mapping[str, object]],
    environment: Mapping[str, str],
    cwd: Path,
    executable_dir: Path,
) -> list[dict[str, object]]:
    roots: dict[tuple[str, Path], set[str]] = defaultdict(set)
    python_versions: set[str] = set()
    has_ruby = False
    has_python = False

    def add_root(
        language: str, role: str, path: Path, *, required: bool = False
    ) -> None:
        candidate = path if path.is_absolute() else cwd / path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            if required:
                raise RuntimeProvenanceError(
                    f"required {language} runtime root is unavailable: {candidate}"
                ) from exc
            return
        if not resolved.is_file() and not resolved.is_dir():
            if required:
                raise RuntimeProvenanceError(
                    f"required {language} runtime root is unusable: {resolved}"
                )
            return
        roots[(language, resolved)].add(role)

    for record in loaded_dependencies:
        aliases = _loader_aliases(record)
        target = Path(str(record["resolved_path"]))
        names = aliases | {target.name}
        if any(name.startswith("libruby") for name in names):
            has_ruby = True
            direct = target.parent / "ruby"
            fallback = target.parent.parent / "lib" / "ruby"
            selected = direct if direct.exists() else fallback
            add_root("ruby", "selected-libruby-stdlib", selected, required=True)
        python_matches = [
            match
            for name in names
            if (match := re.search(r"libpython(\d+\.\d+)", name))
        ]
        if python_matches:
            has_python = True
            for match in python_matches:
                version = match.group(1)
                python_versions.add(version)
                direct = target.parent / f"python{version}"
                fallback = target.parent.parent / "lib" / f"python{version}"
                selected = direct if direct.exists() else fallback
                add_root(
                    "python",
                    f"selected-libpython-{version}-stdlib",
                    selected,
                    required=True,
                )
        if any(name.startswith("libklayout_pya.so") for name in names):
            add_root("python", "klayout-pya-pymod", target.parent / "pymod")

    if has_ruby:
        add_root("ruby", "klayout-install-ruby", executable_dir / "ruby")
        for key in ("RUBYLIB", "GEM_PATH"):
            if environment.get(key):
                for position, path in enumerate(
                    _path_entries(environment[key], cwd), start=1
                ):
                    add_root("ruby", f"environment-{key}-{position}", path)
        for key in ("GEM_HOME", "BUNDLE_PATH"):
            if environment.get(key):
                add_root("ruby", f"environment-{key}", Path(environment[key]))
        if environment.get("RUBYOPT"):
            for position, path in enumerate(
                _ruby_include_paths(environment["RUBYOPT"], cwd), start=1
            ):
                add_root("ruby", f"environment-RUBYOPT-I-{position}", path)
            for position, path in enumerate(
                _ruby_required_paths(environment["RUBYOPT"], cwd), start=1
            ):
                add_root(
                    "ruby",
                    f"environment-RUBYOPT-require-{position}",
                    path,
                    required=True,
                )
        home = environment.get("HOME")
        if home:
            add_root(
                "ruby",
                "ruby-user-gems-xdg",
                Path(home) / ".local/share/gem/ruby",
            )
            add_root("ruby", "ruby-user-gems-legacy", Path(home) / ".gem/ruby")

    if has_python:
        add_root("python", "klayout-install-python", executable_dir / "python")
        add_root("python", "klayout-install-pymod", executable_dir / "pymod")
        # pya.cc clears ordinary PYTHONPATH before initializing embedded
        # Python, then maps KLAYOUT_PYTHONPATH into its place.  Keep the raw
        # PYTHONPATH value in the environment identity, but do not inventory
        # a tree that KLayout makes non-executable.
        for key in ("KLAYOUT_PYTHONPATH",):
            if environment.get(key):
                for position, path in enumerate(
                    _path_entries(environment[key], cwd), start=1
                ):
                    add_root("python", f"environment-{key}-{position}", path)
        # KLayout likewise clears PYTHONHOME and honors only its namespaced
        # replacement.
        for key in ("KLAYOUT_PYTHONHOME",):
            if not environment.get(key):
                continue
            configured_home = Path(environment[key])
            for version in sorted(python_versions):
                add_root(
                    "python",
                    f"environment-{key}-python{version}",
                    configured_home / "lib" / f"python{version}",
                    required=True,
                )
        if environment.get("PYTHONNOUSERSITE", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            home = environment.get("HOME")
            if home:
                for version in sorted(python_versions):
                    add_root(
                        "python",
                        f"python-user-site-{version}",
                        Path(home)
                        / ".local"
                        / "lib"
                        / f"python{version}"
                        / "site-packages",
                    )

    klayout_bases: list[tuple[str, Path]] = []
    if environment.get("KLAYOUT_HOME"):
        klayout_bases.append(("klayout-home", Path(environment["KLAYOUT_HOME"])))
    if environment.get("KLAYOUT_PATH"):
        klayout_bases.extend(
            (f"klayout-path-{position}", path)
            for position, path in enumerate(
                _path_entries(environment["KLAYOUT_PATH"], cwd), start=1
            )
        )
    for base_role, base in klayout_bases:
        if has_ruby:
            add_root("ruby", f"{base_role}-ruby", base / "ruby")
        if has_python:
            add_root("python", f"{base_role}-python", base / "python")

    records = [
        _language_tree_record(language, sorted(roles), root)
        for (language, root), roles in roots.items()
    ]
    records.sort(key=lambda record: (record["language"], tuple(record["roles"])))
    return records


def _adjacent_library_records(
    executable_dir: Path,
    selected_targets: set[Path],
) -> dict[str, object]:
    aliases_by_target: dict[Path, set[str]] = defaultdict(set)
    for alias in sorted(executable_dir.glob("libklayout*.so*")):
        try:
            target = alias.resolve(strict=True)
        except OSError as exc:
            raise RuntimeProvenanceError(
                f"adjacent KLayout library alias cannot be resolved: {alias}"
            ) from exc
        if target.is_file():
            aliases_by_target[target].add(alias.name)

    selected: list[dict[str, object]] = []
    unloaded: list[dict[str, object]] = []
    for target, aliases in aliases_by_target.items():
        is_selected = target in selected_targets
        record = _file_record(
            target,
            relative_aliases=sorted(aliases),
            selected_by_loader=is_selected,
            included_in_identity=is_selected,
        )
        (selected if is_selected else unloaded).append(record)
    for records in (selected, unloaded):
        records.sort(key=lambda record: tuple(record["relative_aliases"]))
    return {
        "policy": (
            "Only ldd-selected targets enter the canonical dependency identity; "
            "adjacent unloaded files are diagnostic metadata only."
        ),
        "selected": selected,
        "unloaded": unloaded,
    }


def _is_relevant_environment_key(key: str) -> bool:
    if key.startswith(
        (
            "KLAYOUT_",
            "RUBY",
            "PYTHON",
            "GEM_",
            "BUNDLE_",
            "LC_",
            "QT_",
            "QML_",
        )
    ):
        return True
    if key in {
        "HOME",
        "LANG",
        "LANGUAGE",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_BIND_NOW",
        "LD_HWCAP_MASK",
        "GLIBC_TUNABLES",
        "MALLOC_CONF",
        "TZ",
    }:
        return True
    if key.startswith(
        (
            "MALLOC_",
            "JEMALLOC_",
            "TCMALLOC_",
            "OMP_",
            "GOMP_",
            "KMP_",
            "TBB_",
            "RAYON_",
        )
    ):
        return True
    return key.endswith("_NUM_THREADS")


def _is_sensitive_environment_key(key: str) -> bool:
    upper = key.upper()
    return any(fragment in upper for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS)


def _normalize_environment(
    environment: Mapping[str, str],
    path_replacements: Mapping[str | os.PathLike[str], str] | None,
    normalizer: EnvironmentNormalizer | None,
    normalization_policy_id: str | None,
) -> dict[str, object]:
    raw_captured = {
        key: str(value)
        for key, value in sorted(environment.items())
        if _is_relevant_environment_key(key)
    }
    uses_caller_normalization = bool(path_replacements) or normalizer is not None
    if uses_caller_normalization and not normalization_policy_id:
        raise RuntimeProvenanceError(
            "path replacements and environment normalizers require a stable "
            "normalization_policy_id"
        )
    if normalization_policy_id is not None and not normalization_policy_id.strip():
        raise RuntimeProvenanceError("normalization_policy_id cannot be empty")

    replacements: list[tuple[str, str]] = []
    if path_replacements:
        for old, replacement in path_replacements.items():
            old_text = os.fspath(old)
            if not old_text:
                raise RuntimeProvenanceError(
                    "environment path replacement cannot be empty"
                )
            if not isinstance(replacement, str) or not replacement:
                raise RuntimeProvenanceError(
                    "environment path replacement tokens must be nonempty strings"
                )
            replacements.append((old_text, replacement))
        replacements.sort(key=lambda item: (-len(item[0]), item[0]))

    captured: dict[str, object] = {}
    normalized: dict[str, object] = {}
    hook_omitted: list[str] = []
    redacted_keys: list[str] = []
    for key, raw_value in raw_captured.items():
        if _is_sensitive_environment_key(key):
            value_sha256 = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()
            captured[key] = {
                "redacted": True,
                "value_sha256": value_sha256,
            }
            normalized[key] = {
                "redacted_value_sha256": value_sha256,
            }
            redacted_keys.append(key)
            continue
        captured[key] = raw_value
        if key == "LD_PRELOAD":
            value = _PRELOAD_IDENTITY_SENTINEL
        else:
            value = raw_value
            for old, replacement in replacements:
                value = value.replace(old, replacement)
        if normalizer is not None:
            normalized_value = normalizer(key, value)
            if normalized_value is None:
                hook_omitted.append(key)
                continue
            if not isinstance(normalized_value, str):
                raise RuntimeProvenanceError(
                    f"environment normalizer returned a non-string for {key}"
                )
            value = normalized_value
        normalized[key] = value

    policy = {
        "id": normalization_policy_id or _DEFAULT_NORMALIZATION_POLICY_ID,
        "caller_hook": normalizer is not None,
        "path_replacement_tokens": sorted(
            replacement for _, replacement in replacements
        ),
        "built_in_rules": [
            "ld-preload-is-ordered-content-identity-v1",
            "sensitive-environment-values-are-sha256-v1",
        ],
    }
    return {
        "captured": captured,
        "normalized": normalized,
        "normalization_policy": policy,
        "hook_omitted_keys": hook_omitted,
        "redacted_keys": redacted_keys,
        "path_replacement_count": len(replacements),
        "search_path_state_policy": (
            "Environment search paths are captured as strings only. Directory "
            "contents require a separately named caller policy and identity."
        ),
    }


def _dependency_identity(
    loaded: Sequence[Mapping[str, object]],
    virtual: Sequence[Mapping[str, object]],
    preload_targets: set[Path],
) -> dict[str, object]:
    file_entries = []
    for record in loaded:
        target = Path(str(record["resolved_path"])).resolve()
        if target in preload_targets:
            continue
        file_entries.append(
            {
                "role": "elf_dependency",
                "selections": list(record["selections"]),
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
        )
    virtual_entries = [
        {
            "role": "virtual_elf_object",
            "probe_role": record["probe_role"],
            "loader_alias": record["loader_alias"],
            "loader_position": record["loader_position"],
        }
        for record in virtual
    ]
    return {"files": file_entries, "virtual": virtual_entries}


def _capture_loader_state(
    executable_path: Path,
    environment: Mapping[str, str],
    cwd: Path,
    trusted_ldd: Path,
    plugin_directories: Mapping[str, str | os.PathLike[str]] | None,
    preload_tokens: Sequence[str],
    explicit_preloads: Mapping[int, Path],
) -> dict[str, object]:
    executable_entries, executable_probe = _run_ldd(
        executable_path, "executable", environment, cwd, trusted_ldd
    )
    executable_dependencies, _ = _materialize_ldd_entries(
        executable_entries, cwd
    )
    plugins, plugin_roots, plugin_coverage = _plugin_records(
        executable_path.parent,
        executable_dependencies,
        environment,
        cwd,
        plugin_directories,
    )

    all_entries = list(executable_entries)
    probe_metadata = [executable_probe]
    # A lay plugin can have DT_NEEDED references to db plugins already dlopen'd
    # earlier by KLayout.  Standalone ldd has no existing process namespace, so
    # expose the discovered plugin directories ahead of the caller's loader
    # path while probing.  This lets ldd select the same canonical plugin
    # objects rather than reporting those already-loaded SONAMEs as missing.
    plugin_probe_environment = dict(environment)
    plugin_loader_directories: list[str] = []
    for plugin in plugins:
        directory = str(Path(str(plugin["resolved_path"])).parent)
        if directory not in plugin_loader_directories:
            plugin_loader_directories.append(directory)
    caller_loader_path = environment.get("LD_LIBRARY_PATH")
    if caller_loader_path:
        plugin_loader_directories.append(caller_loader_path)
    if plugin_loader_directories:
        plugin_probe_environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            plugin_loader_directories
        )
    for plugin in plugins:
        plugin_entries, plugin_probe = _run_ldd(
            Path(str(plugin["resolved_path"])),
            str(plugin["probe_role"]),
            plugin_probe_environment,
            cwd,
            trusted_ldd,
        )
        all_entries.extend(plugin_entries)
        probe_metadata.append(plugin_probe)

    loaded, virtual = _materialize_ldd_entries(all_entries, cwd)
    preloads = _resolve_preloads(preload_tokens, explicit_preloads, loaded)
    return {
        "loaded": loaded,
        "virtual": virtual,
        "preloads": preloads,
        "plugins": plugins,
        "plugin_roots": plugin_roots,
        "plugin_coverage": plugin_coverage,
        "plugin_probe_ld_library_path": plugin_probe_environment.get(
            "LD_LIBRARY_PATH"
        ),
        "ldd_probes": probe_metadata,
    }


def _file_stability(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "resolved_path": record["resolved_path"],
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
        "filesystem_identity": record["filesystem_identity"],
    }


def _state_stability(
    executable_record: Mapping[str, object], state: Mapping[str, object]
) -> dict[str, object]:
    return {
        "executable": _file_stability(executable_record),
        "elf_dependencies": [
            {
                **_file_stability(record),
                "selections": record["selections"],
                "selected_as": record["selected_as"],
            }
            for record in state["loaded"]
        ],
        "virtual_elf_dependencies": list(state["virtual"]),
        "runtime_plugins": [
            {
                **_file_stability(record),
                "relative_aliases": record["relative_aliases"],
                "probe_role": record["probe_role"],
            }
            for record in state["plugins"]
        ],
        "language_runtime_roots": [
            {
                "language": record["language"],
                "roles": record["roles"],
                "resolved_path": record["resolved_path"],
                "file_count": record["file_count"],
                "total_bytes": record["total_bytes"],
                "manifest_sha256": record["manifest_sha256"],
                "stability_sha256": record["stability_sha256"],
            }
            for record in state["language_runtime_roots"]
        ],
        "plugin_roots": list(state["plugin_roots"]),
        "plugin_coverage": dict(state["plugin_coverage"]),
        "plugin_probe_ld_library_path": state["plugin_probe_ld_library_path"],
        "ld_preload": [
            {
                **_file_stability(record),
                "loader_position": record["loader_position"],
                "loader_aliases": record["loader_aliases"],
            }
            for record in state["preloads"]
        ],
    }


def _assert_stable_snapshot(
    initial: Mapping[str, object], final: Mapping[str, object]
) -> None:
    for component in (
        "executable",
        "elf_dependencies",
        "virtual_elf_dependencies",
        "runtime_plugins",
        "language_runtime_roots",
        "plugin_roots",
        "plugin_coverage",
        "plugin_probe_ld_library_path",
        "ld_preload",
    ):
        if initial[component] != final[component]:
            raise RuntimeProvenanceError(
                "runtime changed while provenance was collected: " + component
            )


def collect_runtime_provenance(
    executable: str | os.PathLike[str],
    environment: Mapping[str, str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    plugin_directories: Mapping[str, str | os.PathLike[str]] | None = None,
    plugin_search_policy_id: str | None = None,
    path_replacements: Mapping[str | os.PathLike[str], str] | None = None,
    environment_normalizer: EnvironmentNormalizer | None = None,
    normalization_policy_id: str | None = None,
    ldd_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Return rich runtime metadata and a normalized canonical identity.

    ``environment`` is the exact environment intended for KLayout.  Relative
    executable paths use ``cwd`` (the current directory by default).  A
    path-bearing ``LD_PRELOAD`` entry must be absolute.  ``plugin_directories``
    maps stable roles to an exhaustive set of runtime plugin roots and requires
    ``plugin_search_policy_id``.  Without it, only installation/module-adjacent
    ``db_plugins`` and ``lay_plugins`` roots are covered; nonempty external
    KLayout search state and unresolved installation architecture/Salt state
    are rejected.

    ``path_replacements`` and ``environment_normalizer`` affect only the
    normalized environment identity.  Either requires a stable
    ``normalization_policy_id`` which enters that identity alongside the
    replacement tokens and hook-enabled state.  Dynamically loadable Ruby and
    Python files under interpreter-, KLayout-, and environment-selected roots
    enter compact content manifests; non-code search-path data does not.
    Sensitive diagnostic values are redacted; their value hashes remain in the
    canonical identity.  The executable, environment, and ldd implementation
    are trusted inputs.
    """

    working_directory = Path(cwd or os.getcwd()).resolve()
    runtime_environment = {
        str(key): str(value) for key, value in environment.items()
    }
    if runtime_environment.get("LD_AUDIT"):
        raise RuntimeProvenanceError(
            "nonempty LD_AUDIT is unsupported until audit modules and their "
            "dependency closures are content-fingerprinted"
        )
    if plugin_directories is not None and not plugin_search_policy_id:
        raise RuntimeProvenanceError(
            "explicit plugin_directories require a stable "
            "plugin_search_policy_id"
        )
    if plugin_search_policy_id is not None and not plugin_search_policy_id.strip():
        raise RuntimeProvenanceError("plugin_search_policy_id cannot be empty")

    executable_path = _resolve_executable(
        executable, runtime_environment, working_directory
    )
    trusted_ldd = _resolve_ldd(ldd_path)
    preload_tokens = _preload_tokens(runtime_environment.get("LD_PRELOAD"))
    explicit_preloads = _resolve_explicit_preloads(preload_tokens)

    # The initial executable hash precedes every loader probe.  The complete
    # loader selection and every identity-bearing file are then collected a
    # second time.  Exact path/inode/content/inventory comparison rejects a
    # build or install changed during collection.
    initial_executable = _file_record(
        executable_path, supplied_as=os.fspath(executable), role="executable"
    )
    initial_state = _capture_loader_state(
        executable_path,
        runtime_environment,
        working_directory,
        trusted_ldd,
        plugin_directories,
        preload_tokens,
        explicit_preloads,
    )
    initial_state["language_runtime_roots"] = _language_runtime_roots(
        initial_state["loaded"],
        runtime_environment,
        working_directory,
        executable_path.parent,
    )
    final_state = _capture_loader_state(
        executable_path,
        runtime_environment,
        working_directory,
        trusted_ldd,
        plugin_directories,
        preload_tokens,
        explicit_preloads,
    )
    # Plugin files were hashed before their final ldd probes.  Rediscover and
    # hash the final inventory once more after those probes.
    final_plugins, final_plugin_roots, final_plugin_coverage = _plugin_records(
        executable_path.parent,
        [
            record
            for record in final_state["loaded"]
            if any(
                selection["probe_role"] == "executable"
                for selection in record["selections"]
            )
        ],
        runtime_environment,
        working_directory,
        plugin_directories,
    )
    post_plugin_state = {
        **final_state,
        "plugins": final_plugins,
        "plugin_roots": final_plugin_roots,
        "plugin_coverage": final_plugin_coverage,
    }
    post_plugin_state["language_runtime_roots"] = _language_runtime_roots(
        post_plugin_state["loaded"],
        runtime_environment,
        working_directory,
        executable_path.parent,
    )
    final_executable = _file_record(
        executable_path, supplied_as=os.fspath(executable), role="executable"
    )
    _assert_stable_snapshot(
        _state_stability(initial_executable, initial_state),
        _state_stability(final_executable, post_plugin_state),
    )

    loaded_dependencies = post_plugin_state["loaded"]
    virtual_dependencies = post_plugin_state["virtual"]
    preloads = post_plugin_state["preloads"]
    plugins = post_plugin_state["plugins"]
    plugin_roots = post_plugin_state["plugin_roots"]
    plugin_coverage = post_plugin_state["plugin_coverage"]
    language_runtime_roots = post_plugin_state["language_runtime_roots"]
    preload_targets = {
        Path(str(record["resolved_path"])).resolve() for record in preloads
    }
    selected_targets = {
        Path(str(record["resolved_path"])).resolve()
        for record in loaded_dependencies
    }
    adjacent = _adjacent_library_records(executable_path.parent, selected_targets)
    environment_metadata = _normalize_environment(
        runtime_environment,
        path_replacements,
        environment_normalizer,
        normalization_policy_id,
    )

    executable_record = final_executable
    dependency_identity = _dependency_identity(
        loaded_dependencies, virtual_dependencies, preload_targets
    )
    plugin_identity = [
        {
            "role": "runtime_plugin",
            "relative_aliases": list(record["relative_aliases"]),
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in plugins
    ]
    language_runtime_identity = [
        {
            "language": record["language"],
            "roles": record["roles"],
            "file_count": record["file_count"],
            "total_bytes": record["total_bytes"],
            "manifest_sha256": record["manifest_sha256"],
        }
        for record in language_runtime_roots
    ]
    preload_identity = [
        {
            "role": "ld_preload",
            "loader_position": record["loader_position"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in preloads
    ]
    components: dict[str, object] = {
        "schema": IDENTITY_SCHEMA,
        "executable": {
            "role": "executable",
            "size_bytes": executable_record["size_bytes"],
            "sha256": executable_record["sha256"],
        },
        "elf_dependencies": dependency_identity,
        "runtime_plugins": plugin_identity,
        "language_runtime_roots": language_runtime_identity,
        "language_runtime_policy": _LANGUAGE_TREE_POLICY_ID,
        "plugin_search_policy": {
            "id": plugin_search_policy_id or _DEFAULT_PLUGIN_SEARCH_POLICY_ID,
            "mode": plugin_coverage["mode"],
            "root_roles": [root["role"] for root in plugin_roots],
            "dependency_probe": "discovered-plugin-dirs-first-v1",
        },
        "ld_preload": preload_identity,
        "environment": environment_metadata["normalized"],
        "environment_normalization_policy": environment_metadata[
            "normalization_policy"
        ],
        "environment_search_path_state_policy": (
            "language-loadable-content-v1-other-values-only-v1"
        ),
    }
    aggregate_sha256 = canonical_json_sha256(components)
    identity = {**components, "canonical_sha256": aggregate_sha256}

    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "canonical_sha256": aggregate_sha256,
        "identity": identity,
        "executable": executable_record,
        "elf_dependencies": {
            "loaded": loaded_dependencies,
            "virtual": virtual_dependencies,
            "ldd_probes": post_plugin_state["ldd_probes"],
        },
        "runtime_plugins": {
            "roots": plugin_roots,
            "files": plugins,
            "coverage": plugin_coverage,
            "probe_ld_library_path": post_plugin_state[
                "plugin_probe_ld_library_path"
            ],
            "search_policy_id": (
                plugin_search_policy_id or _DEFAULT_PLUGIN_SEARCH_POLICY_ID
            ),
        },
        "language_runtime": {
            "policy_id": _LANGUAGE_TREE_POLICY_ID,
            "roots": language_runtime_roots,
            "diagnostic_policy": (
                "Only aggregate manifests are retained; per-file paths and "
                "hashes are used transiently for two-pass stability."
            ),
        },
        "ld_preload": {
            "ordered_files": preloads,
            "identity_policy": (
                "Loader position and file contents enter the canonical identity; "
                "request paths are diagnostic only."
            ),
        },
        "adjacent_klayout_libraries": adjacent,
        "environment": environment_metadata,
    }


__all__ = [
    "EnvironmentNormalizer",
    "FORMAT",
    "FORMAT_VERSION",
    "IDENTITY_SCHEMA",
    "RuntimeProvenanceError",
    "canonical_json_sha256",
    "collect_runtime_provenance",
]
