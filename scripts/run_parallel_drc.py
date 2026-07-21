#!/usr/bin/env python3
"""Run independent KLayout DRC shards concurrently and merge their reports.

This launcher gives each shard its own KLayout process.  That isolation is
intentional: deep-mode DRC keeps mutable hierarchy and geometry caches that
must not be shared between concurrent rule groups.  Every child reads the same
layout and deck, but receives a distinct ``drc_shard`` run-time variable and a
private report path.  Once *all* children have exited successfully, the shard
reports are merged according to an explicit ownership/order manifest by
``merge_sharded_lyrdb.merge_reports``.

The requested output is never handed to a child, and merging starts only after
all children succeed.  A failed shard therefore cannot publish a partial final
report.  By default, temporary reports and logs are deleted after success or
failure; pass ``--keep-temp`` when diagnosing a run to retain them.

Every successful run atomically publishes a machine-readable provenance file
at ``<output>.metadata.json`` (or ``--metadata``).  It binds timings to hashes
of the layout, deck, manifest, output, loader-selected KLayout dependency and
plugin closure, and preloaded libraries.  Failed attempts use a distinct
``*.failed.json`` file and never replace accepted success metadata.

Example::

    python3 scripts/run_parallel_drc.py \
      --klayout ./bin-clang/klayout \
      --deck /path/to/freepdk45-sharded.lydrc \
      --input /path/to/design.gds \
      --top-cell design \
      --output /path/to/design.drc.report \
      --manifest /path/to/freepdk45-shards.json \
      --shard active_m1 --shard other

Decks using the common ``top_cell``/``report`` run-time variable convention
(including the Sky130 deck in the performance corpus) additionally pass::

    --top-cell-rd-key top_cell --output-rd-key report

For a reproducible benchmark launch, set ``KLAYOUT_HOME`` to a fresh empty
directory and leave ``KLAYOUT_PATH`` unset.  The provenance collector rejects
uncovered user/plugin search state instead of silently omitting loaded code.

Only Python's standard library is used by this launcher.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence, TextIO

if __package__:
    from .klayout_runtime_provenance import collect_runtime_provenance
else:
    from klayout_runtime_provenance import collect_runtime_provenance


_RESERVED_RD_KEYS = frozenset(
    {"input", "topcell", "top_cell", "output", "report", "drc_shard"}
)
_TOP_CELL_RD_KEYS = ("topcell", "top_cell")
_OUTPUT_RD_KEYS = ("output", "report")
_POLL_INTERVAL_SECONDS = 0.05
_TERMINATE_TIMEOUT_SECONDS = 5.0
_PROVENANCE_FORMAT = "klayout-parallel-drc-provenance"
_PROVENANCE_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024
_FIXED_ENVIRONMENT_KEYS = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "GLIBC_TUNABLES",
    "MALLOC_CONF",
    "KLAYOUT_HOME",
    "QT_QPA_PLATFORM",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)
_ENVIRONMENT_PREFIXES = (
    "MALLOC_",
    "JEMALLOC_",
    "TCMALLOC_",
    "OMP_",
    "GOMP_",
    "KMP_",
    "TBB_",
    "RAYON_",
)
_SENSITIVE_ENVIRONMENT_FRAGMENTS = (
    "TOKEN",
    "PASS",
    "SECRET",
    "KEY",
    "CREDENTIAL",
)


@dataclass(frozen=True)
class ShardSpec:
    """One named DRC shard and its private temporary artifacts."""

    index: int
    name: str
    report: Path
    log: Path


@dataclass
class RunningShard:
    """Bookkeeping for a live KLayout child process."""

    spec: ShardSpec
    process: subprocess.Popen[bytes]
    log_file: TextIO
    started: float
    process_group_id: int | None = None
    peak_rss_kb: int | None = None


@dataclass(frozen=True)
class ShardResult:
    """Timing and status returned by one completed shard."""

    spec: ShardSpec
    returncode: int
    wall_seconds: float
    peak_rss_kb: int | None


class ShardFailure(RuntimeError):
    """Raised when a KLayout shard exits unsuccessfully."""


class TargetLockedError(RuntimeError):
    """Raised when another launcher owns one of the publication targets."""


class TerminationRequested(BaseException):
    """Convert SIGTERM into normal child and temporary-artifact cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class _OwnedLock:
    """Identity of one cooperatively owned O_EXCL lock file."""

    path: Path
    device: int
    inode: int


class _TargetLocks:
    """Exclusive locks for all report and provenance publication targets."""

    def __init__(self, owned: Sequence[_OwnedLock]) -> None:
        self._owned = list(owned)

    @classmethod
    def acquire(cls, targets: Sequence[Path]) -> _TargetLocks:
        lock_paths = sorted({_lock_path(target) for target in targets}, key=str)
        owned: list[_OwnedLock] = []
        try:
            for lock_path in lock_paths:
                try:
                    descriptor = os.open(
                        lock_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                except FileExistsError as exc:
                    raise TargetLockedError(
                        f"publication target is already locked: {lock_path}"
                    ) from exc
                identity = os.fstat(descriptor)
                owned.append(
                    _OwnedLock(lock_path, identity.st_dev, identity.st_ino)
                )
                try:
                    payload = (
                        f"pid={os.getpid()}\n"
                        f"created_utc={_utc_now()}\n"
                    ).encode("utf-8")
                    os.write(descriptor, payload)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _fsync_directory(lock_path.parent)
        except BaseException:
            cls(owned).release()
            raise
        return cls(owned)

    def release(self) -> None:
        for lock in reversed(self._owned):
            try:
                current = lock.path.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                print(
                    f"warning: unable to inspect target lock {lock.path}: {exc}",
                    file=sys.stderr,
                )
                continue
            # Never remove a lock that was replaced after we acquired it.
            if (current.st_dev, current.st_ino) != (lock.device, lock.inode):
                continue
            try:
                lock.path.unlink()
                _fsync_directory(lock.path.parent)
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(
                    f"warning: unable to release target lock {lock.path}: {exc}",
                    file=sys.stderr,
                )
        self._owned.clear()


def positive_int(value: str) -> int:
    """Argparse converter accepting strictly positive integers."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_rd(value: str) -> tuple[str, str]:
    """Parse and validate one additional ``--rd KEY=VALUE`` assignment."""

    key, separator, assignment = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(
            f"expected KEY=VALUE for --rd, got {value!r}"
        )
    if key in _RESERVED_RD_KEYS:
        reserved = ", ".join(sorted(_RESERVED_RD_KEYS))
        raise argparse.ArgumentTypeError(
            f"--rd may not override reserved key {key!r} ({reserved})"
        )
    return key, assignment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments without touching the filesystem."""

    parser = argparse.ArgumentParser(
        description=(
            "Run independent shards of one KLayout DRC deck concurrently, "
            "then atomically merge their report databases using an explicit "
            "category ownership/order manifest."
        ),
        epilog=(
            "Each shard receives -rd drc_shard=NAME. Additional --rd values "
            "are forwarded to every child, but orchestration keys (input, "
            "topcell/top_cell, output/report and drc_shard) are reserved. "
            "Child stdout and stderr are combined in one per-shard log."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--klayout",
        required=True,
        metavar="EXE",
        help="KLayout executable (a path or a command resolvable through PATH)",
    )
    parser.add_argument(
        "--deck",
        required=True,
        metavar="FILE",
        help="shard-aware KLayout DRC deck",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="LAYOUT",
        help="input layout read independently by every shard",
    )
    parser.add_argument(
        "--top-cell",
        required=True,
        metavar="CELL",
        help="top cell passed to the deck through --top-cell-rd-key",
    )
    parser.add_argument(
        "--top-cell-rd-key",
        choices=_TOP_CELL_RD_KEYS,
        default="topcell",
        metavar="KEY",
        help="KLayout -rd variable used for the top cell",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="REPORT",
        help="final merged .lyrdb/report path; never written by a child",
    )
    parser.add_argument(
        "--output-rd-key",
        choices=_OUTPUT_RD_KEYS,
        default="output",
        metavar="KEY",
        help="KLayout -rd variable used for each private child report",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        metavar="JSON",
        help="merge manifest defining category ownership and canonical order",
    )
    parser.add_argument(
        "--shard",
        required=True,
        action="append",
        metavar="NAME",
        help="shard name to run; repeat for every shard in the manifest",
    )
    parser.add_argument(
        "--rd",
        action="append",
        default=[],
        type=parse_rd,
        metavar="KEY=VALUE",
        help="additional KLayout run-time variable forwarded to every shard",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        metavar="N",
        help="maximum number of concurrent KLayout processes (all shards if omitted)",
    )
    parser.add_argument(
        "--cohort-id",
        metavar="ID",
        help="label shared by concurrently launched benchmark replicates",
    )
    parser.add_argument(
        "--replicate-index",
        type=positive_int,
        metavar="N",
        help="one-based index of this run within its benchmark cohort",
    )
    parser.add_argument(
        "--replicate-count",
        type=positive_int,
        metavar="N",
        help="declared number of runs in this benchmark cohort",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="retain the temporary directory containing shard reports and logs",
    )
    parser.add_argument(
        "--metadata",
        metavar="JSON",
        help=(
            "successful-run provenance sidecar "
            "(<output>.metadata.json if omitted)"
        ),
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    # Keep the exact spelling and ordering supplied by the caller.  Namespace
    # fields alone lose that information and are insufficient for reproducing
    # or comparing benchmark invocations.
    args.launcher_argv = raw_argv
    args.process_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *raw_argv]
    )
    return args


def metadata_path(args: argparse.Namespace) -> Path:
    """Return the explicit or deterministic successful-run sidecar path."""

    configured = getattr(args, "metadata", None)
    return Path(configured) if configured else Path(f"{args.output}.metadata.json")


def _failure_metadata_path(path: Path) -> Path:
    """Use a distinct name so a failed attempt cannot replace accepted data."""

    if path.suffix == ".json":
        return path.with_name(f"{path.stem}.failed.json")
    return path.with_name(f"{path.name}.failed.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(
    path: str | os.PathLike[str], *, supplied_as: str | None = None
) -> dict[str, object]:
    """Hash one regular file while detecting a concurrent replacement."""

    original = Path(path)
    resolved = original.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"provenance input is not a regular file: {resolved}")
    sha256 = _sha256_file(resolved)
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"file changed while being fingerprinted: {resolved}")
    record: dict[str, object] = {
        "path": str(original),
        "resolved_path": str(resolved),
        "size_bytes": after.st_size,
        "sha256": sha256,
    }
    if supplied_as is not None:
        record["supplied_as"] = supplied_as
    return record


def _resolved_executable(command: str) -> Path:
    if os.sep in command or (os.altsep and os.altsep in command):
        return Path(command).resolve(strict=True)
    resolved = shutil.which(command)
    if resolved is None:
        raise ValueError(f"--klayout command was not found on PATH: {command}")
    return Path(resolved).resolve(strict=True)


def _environment_record() -> dict[str, object]:
    """Capture allocator, loader, OpenMP, and KLayout runtime controls."""

    keys = set(_FIXED_ENVIRONMENT_KEYS)
    keys.update(
        key
        for key in os.environ
        if key.startswith(_ENVIRONMENT_PREFIXES) or key.endswith("_NUM_THREADS")
    )
    captured: dict[str, object] = {}
    for key in sorted(keys):
        value = os.environ.get(key)
        if value is not None and any(
            fragment in key.upper()
            for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS
        ):
            captured[key] = {
                "redacted": True,
                "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
        else:
            captured[key] = value
    return captured


def _preload_entries(value: str | None) -> list[str]:
    if not value:
        return []
    # The ELF loader accepts both spaces and colons as LD_PRELOAD separators.
    return value.replace(":", " ").split()


def _resolve_preload(entry: str, executable_dir: Path) -> Path | None:
    candidate = Path(entry)
    if candidate.is_absolute() or os.sep in entry:
        candidate = candidate if candidate.is_absolute() else Path.cwd() / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None
        return resolved if resolved.is_file() else None

    search_dirs: list[Path] = []
    for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if item:
            search_dirs.append(Path(item))
    search_dirs.append(executable_dir)
    for directory in search_dirs:
        candidate = directory / entry
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _runtime_bundle(command: str) -> dict[str, object]:
    """Fingerprint the exact loader-selected KLayout runtime, fail closed."""

    return collect_runtime_provenance(
        command,
        os.environ,
        cwd=Path.cwd(),
    )


def _runtime_component_paths(value: object) -> set[Path]:
    """Return every content-addressed file path in runtime provenance."""

    paths: set[Path] = set()
    if isinstance(value, Mapping):
        resolved_path = value.get("resolved_path")
        if isinstance(resolved_path, str):
            candidate = Path(resolved_path)
            if candidate.is_file():
                paths.add(candidate.resolve())
        for nested in value.values():
            paths.update(_runtime_component_paths(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            paths.update(_runtime_component_paths(nested))
    return paths


def _validate_runtime_publication_targets(
    args: argparse.Namespace,
    runtime_bundle: Mapping[str, object],
) -> None:
    """Prevent report or metadata publication over attested runtime code."""

    sidecar = metadata_path(args)
    targets = {
        "--output": _canonical_target(Path(args.output)),
        "--metadata": _canonical_target(sidecar),
        "failure metadata": _canonical_target(_failure_metadata_path(sidecar)),
    }
    protected = _runtime_component_paths(runtime_bundle)
    for label, target in targets.items():
        if target in protected:
            raise ValueError(
                f"{label} must not overwrite a collected KLayout runtime "
                f"component: {target}"
            )


def _input_records(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    return {
        "deck": _file_record(args.deck),
        "input_layout": _file_record(args.input),
        "manifest": _file_record(args.manifest),
    }


def _same_file_record(
    before: Mapping[str, object], after: Mapping[str, object]
) -> bool:
    keys = ("resolved_path", "size_bytes", "sha256")
    return all(before.get(key) == after.get(key) for key in keys)


def _verify_runtime_unchanged(
    initial_inputs: Mapping[str, Mapping[str, object]],
    final_inputs: Mapping[str, Mapping[str, object]],
    initial_bundle: Mapping[str, object],
    final_bundle: Mapping[str, object],
    initial_orchestrator: Mapping[str, Mapping[str, object]],
    final_orchestrator: Mapping[str, Mapping[str, object]],
) -> None:
    for name, initial in initial_inputs.items():
        if not _same_file_record(initial, final_inputs[name]):
            raise ShardFailure(f"{name} changed while DRC shards were running")
    if initial_bundle["canonical_sha256"] != final_bundle["canonical_sha256"]:
        raise ShardFailure("KLayout runtime bundle changed while DRC shards were running")
    for name, initial in initial_orchestrator.items():
        if not _same_file_record(initial, final_orchestrator[name]):
            raise ShardFailure(f"{name} changed while DRC shards were running")


def _publication_mode(path: Path) -> int:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"publication target is not a regular file path: {path}")
        return stat.S_IMODE(path.stat().st_mode)
    previous_umask = os.umask(0)
    os.umask(previous_umask)
    return 0o666 & ~previous_umask


def _canonical_target(path: Path) -> Path:
    """Canonicalize a publication name without following the final component."""

    return path.parent.resolve(strict=True) / path.name


def _lock_path(target: Path) -> Path:
    target = _canonical_target(target)
    # Do not embed the target suffix: a report deliberately pointed into a
    # plugin directory must not make its lock look like another ``*.so*``
    # candidate while runtime provenance is being collected.
    digest = hashlib.sha256(target.name.encode("utf-8")).hexdigest()[:24]
    return target.with_name(f".parallel-drc-{digest}.lock")


def _fsync_directory(directory: Path) -> None:
    """Durably record directory-entry changes on POSIX filesystems."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_json(path: Path, value: object) -> Path:
    """Fully serialize and fsync JSON in the target directory without publishing."""

    mode = _publication_mode(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", dir=path.parent
    )
    staged = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return staged
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            staged.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: object) -> None:
    """Publish complete JSON with replacement atomicity in its directory."""

    temporary = _stage_json(path, value)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _unused_neighbor(path: Path, label: str) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.{label}-", dir=path.parent
    )
    os.close(descriptor)
    os.unlink(temporary)
    return Path(temporary)


@dataclass
class _BackupState:
    """Track a rename even when a durability barrier or restoration fails."""

    public_path: Path
    existed: bool = False
    backup: Path | None = None
    moved: bool = False


def _backup_existing(state: _BackupState, label: str) -> None:
    path = state.public_path
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"publication target is not a regular file path: {path}")
    state.existed = True
    backup = _unused_neighbor(path, label)
    state.backup = backup
    os.replace(path, backup)
    state.moved = True
    try:
        _fsync_directory(path.parent)
    except BaseException as exc:
        # Do not make the caller guess whether the rename happened when the
        # durability barrier, rather than the rename itself, failed.
        try:
            os.replace(backup, path)
            state.moved = False
            state.backup = None
            _fsync_directory(path.parent)
        except BaseException as restore_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "backup restoration failed: " + str(restore_exc)
                )
        raise


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _publish_staged_pair(
    report_stage: Path,
    output: Path,
    metadata_stage: Path,
    metadata: Path,
) -> None:
    """Publish a report/metadata pair, invalidating stale metadata first.

    There is no portable two-path atomic rename.  The old success metadata is
    therefore moved out of its public name before the report changes.  If a
    later operation fails, rollback restores the previous pair when possible;
    if rollback itself fails, old metadata remains invalidated rather than
    falsely describing a new report.
    """

    metadata_state = _BackupState(metadata)
    report_state = _BackupState(output)
    report_published = False
    metadata_published = False
    try:
        # This ordering is the core consistency invariant: an old success
        # sidecar must not remain public once a new report can become visible.
        _backup_existing(metadata_state, "stale")
        _backup_existing(report_state, "previous")

        os.replace(report_stage, output)
        report_published = True
        _fsync_directory(output.parent)

        os.replace(metadata_stage, metadata)
        metadata_published = True
        _fsync_directory(metadata.parent)
    except BaseException as exc:
        rollback_errors: list[str] = []
        report_restored = not report_published and not report_state.moved
        metadata_removed = not metadata_published
        try:
            if metadata_published:
                _remove_if_present(metadata)
                metadata_removed = True
        except OSError as rollback_exc:
            rollback_errors.append(f"remove new metadata: {rollback_exc}")

        # If new metadata cannot be removed, retain the matching new report;
        # replacing it with the old report would create the inverse mismatch.
        if metadata_removed:
            try:
                if report_published:
                    _remove_if_present(output)
                if report_state.moved and report_state.backup is not None:
                    os.replace(report_state.backup, output)
                    report_state.moved = False
                    report_state.backup = None
                    _fsync_directory(output.parent)
                report_restored = True
            except OSError as rollback_exc:
                rollback_errors.append(f"restore previous report: {rollback_exc}")

        # Restoring stale metadata is safe only after the corresponding old
        # report is known to be back in place.
        if (
            report_restored
            and metadata_state.moved
            and metadata_state.backup is not None
        ):
            try:
                os.replace(metadata_state.backup, metadata)
                metadata_state.moved = False
                metadata_state.backup = None
                _fsync_directory(metadata.parent)
            except OSError as rollback_exc:
                rollback_errors.append(f"restore previous metadata: {rollback_exc}")
        if rollback_errors and hasattr(exc, "add_note"):
            exc.add_note("publication rollback: " + "; ".join(rollback_errors))
        raise

    # Once both public names and directory entries are durable, backup cleanup
    # is non-critical and must not turn a successful publication into failure.
    for backup in (report_state.backup, metadata_state.backup):
        if backup is None:
            continue
        try:
            _remove_if_present(backup)
        except OSError as exc:
            print(
                f"warning: unable to remove publication backup {backup}: {exc}",
                file=sys.stderr,
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _process_peak_rss_kb(pid: int) -> int | None:
    """Read Linux's kernel-maintained RSS high-water mark when available."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        with Path(f"/proc/{pid}/status").open("r", encoding="ascii") as stream:
            for line in stream:
                if not line.startswith("VmHWM:"):
                    continue
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1])
                return None
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def _peak_rss_source(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return "linux_proc_status_vm_hwm_direct_process_polling"


_PEAK_RSS_CAVEAT = (
    "Direct child process only; descendants are excluded, and polling can miss "
    "a short-lived process before /proc is sampled."
)


def _host_record() -> dict[str, object]:
    uname = platform.uname()
    try:
        cpu_affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_affinity = None
    orchestrator_peak_rss_kb = _process_peak_rss_kb(os.getpid())
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": cpu_affinity,
        "orchestrator_peak_rss_kb": orchestrator_peak_rss_kb,
        "peak_rss_source": _peak_rss_source(orchestrator_peak_rss_kb),
        "peak_rss_caveat": _PEAK_RSS_CAVEAT,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }


def _orchestrator_record() -> dict[str, dict[str, object]]:
    launcher = Path(__file__).resolve()
    merger = launcher.with_name("merge_sharded_lyrdb.py")
    runtime_provenance = launcher.with_name("klayout_runtime_provenance.py")
    return {
        "launcher": _file_record(launcher),
        "merger": _file_record(merger),
        "runtime_provenance": _file_record(runtime_provenance),
    }


def _artifact_record(path: Path, retained: bool) -> dict[str, object]:
    record = _file_record(path)
    return {
        "path": str(path.resolve()) if retained else None,
        "retained": retained,
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def _staged_output_record(stage: Path, output: Path) -> dict[str, object]:
    """Describe staged bytes under the public name they will receive."""

    record = _file_record(stage)
    record["path"] = str(output)
    record["resolved_path"] = str(_canonical_target(output))
    return record


def _measurement_semantics() -> dict[str, object]:
    return {
        "cache_state": (
            "uncontrolled/warm-cache-capable; no cache drop is performed, and "
            "pre-workload SHA-256 reads can warm input/runtime file pages"
        ),
        "prehash_before_child_merge_workload": True,
        "child_merge_workload_boundary": (
            "starts immediately before child launch and ends after the merged "
            "report has been created and fsynced in staging"
        ),
        "provenance_verification_boundary": (
            "post-workload rehash and unchanged-file checks only"
        ),
        "full_launcher_wall_boundary": (
            "starts at launcher run entry and ends immediately before success "
            "metadata serialization/publication; atomic pair publication is excluded"
        ),
        "peak_rss_caveat": _PEAK_RSS_CAVEAT,
    }


def _provenance_prefix(
    args: argparse.Namespace,
    specs: Sequence[ShardSpec],
    jobs: int,
    started_utc: str,
    inputs: Mapping[str, Mapping[str, object]] | None,
    runtime_bundle: Mapping[str, object] | None,
    orchestrator: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    return {
        "format": _PROVENANCE_FORMAT,
        "format_version": _PROVENANCE_VERSION,
        "invocation": {
            "argv": list(getattr(args, "process_argv", [])),
            "launcher_arguments": list(getattr(args, "launcher_argv", [])),
            "working_directory": str(Path.cwd().resolve()),
            "child_commands": [build_command(args, spec) for spec in specs],
        },
        "configuration": {
            "top_cell": args.top_cell,
            "top_cell_rd_key": args.top_cell_rd_key,
            "output_rd_key": args.output_rd_key,
            "additional_rd": [
                {"key": key, "value": value} for key, value in args.rd
            ],
            "shard_names": (
                [spec.name for spec in specs] if specs else list(args.shard)
            ),
            "jobs_requested": args.jobs,
            "jobs_effective": jobs,
            "keep_temp": args.keep_temp,
            "cohort": {
                "id": getattr(args, "cohort_id", None),
                "replicate_index": getattr(args, "replicate_index", None),
                "replicate_count": getattr(args, "replicate_count", None),
            },
        },
        "environment": _environment_record(),
        "host": _host_record(),
        "inputs": dict(inputs) if inputs is not None else None,
        "runtime_bundle": dict(runtime_bundle) if runtime_bundle is not None else None,
        "orchestrator": dict(orchestrator) if orchestrator is not None else None,
        "started_utc": started_utc,
    }


def validate_args(args: argparse.Namespace) -> None:
    """Reject ambiguous or unsafe invocations before starting any child."""

    for option, value in (
        ("--deck", args.deck),
        ("--input", args.input),
        ("--manifest", args.manifest),
    ):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"{option} is not a readable file: {value}")

    executable = args.klayout
    if os.sep in executable or (os.altsep and os.altsep in executable):
        path = Path(executable)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"--klayout is not an executable file: {executable}")
    elif shutil.which(executable) is None:
        raise ValueError(f"--klayout command was not found on PATH: {executable}")

    if not args.top_cell:
        raise ValueError("--top-cell must not be empty")

    if any(not name for name in args.shard):
        raise ValueError("--shard names must not be empty")
    duplicate_shards = sorted(
        name for name in set(args.shard) if args.shard.count(name) > 1
    )
    if duplicate_shards:
        raise ValueError(
            "duplicate --shard name(s): " + ", ".join(duplicate_shards)
        )

    duplicate_rd = sorted(
        key for key in {key for key, _ in args.rd}
        if sum(candidate == key for candidate, _ in args.rd) > 1
    )
    if duplicate_rd:
        raise ValueError("duplicate --rd key(s): " + ", ".join(duplicate_rd))

    cohort_id = getattr(args, "cohort_id", None)
    replicate_index = getattr(args, "replicate_index", None)
    replicate_count = getattr(args, "replicate_count", None)
    if cohort_id == "":
        raise ValueError("--cohort-id must not be empty")
    has_replicate_index = replicate_index is not None
    has_replicate_count = replicate_count is not None
    if has_replicate_index != has_replicate_count:
        raise ValueError(
            "--replicate-index and --replicate-count must be supplied together"
        )
    if has_replicate_index and cohort_id is None:
        raise ValueError("replicate fields require --cohort-id")
    if (
        replicate_index is not None
        and replicate_count is not None
        and replicate_index > replicate_count
    ):
        raise ValueError("--replicate-index must not exceed --replicate-count")

    output = Path(args.output)
    if (output.exists() or output.is_symlink()) and (
        output.is_symlink() or not output.is_file()
    ):
        raise ValueError(f"--output is not a regular file path: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"--output parent directory does not exist: {output.parent}")

    resolved_output = output.resolve()
    executable_path = _resolved_executable(args.klayout)
    protected_inputs = {
        Path(args.deck).resolve(),
        Path(args.input).resolve(),
        Path(args.manifest).resolve(),
        executable_path,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("merge_sharded_lyrdb.py"),
        Path(__file__).resolve().with_name("klayout_runtime_provenance.py"),
    }
    for alias in executable_path.parent.glob("libklayout*.so"):
        try:
            protected_inputs.add(alias.resolve(strict=True))
        except OSError:
            pass
    for entry in _preload_entries(os.environ.get("LD_PRELOAD")):
        preload = _resolve_preload(entry, executable_path.parent)
        if preload is not None:
            protected_inputs.add(preload)
    if resolved_output in protected_inputs:
        raise ValueError(
            "--output must not overwrite a runtime, deck, input, or manifest file"
        )

    metadata = metadata_path(args)
    for label, sidecar in (
        ("--metadata", metadata),
        ("failure metadata", _failure_metadata_path(metadata)),
    ):
        if sidecar.exists() and (sidecar.is_symlink() or not sidecar.is_file()):
            raise ValueError(f"{label} is not a regular file path: {sidecar}")
        if not sidecar.parent.is_dir():
            raise ValueError(
                f"{label} parent directory does not exist: {sidecar.parent}"
            )
        if sidecar.resolve() in protected_inputs | {resolved_output}:
            raise ValueError(
                f"{label} must not overwrite an output, runtime, deck, input, "
                "or manifest file"
            )

    publication_targets = {
        _canonical_target(output),
        _canonical_target(metadata),
        _canonical_target(_failure_metadata_path(metadata)),
    }
    lock_paths = {
        _lock_path(output),
        _lock_path(metadata),
        _lock_path(_failure_metadata_path(metadata)),
    }
    collisions = sorted(publication_targets & lock_paths, key=str)
    if collisions:
        raise ValueError(
            "publication target collides with a launcher lock path: "
            + ", ".join(str(path) for path in collisions)
        )


def build_command(args: argparse.Namespace, spec: ShardSpec) -> list[str]:
    """Build one argv vector in the same order as a normal KLayout DRC run."""

    command = [
        args.klayout,
        "-b",
        "-r",
        args.deck,
        "-rd",
        f"input={args.input}",
        "-rd",
        f"{args.top_cell_rd_key}={args.top_cell}",
        "-rd",
        f"{args.output_rd_key}={spec.report}",
        "-rd",
        f"drc_shard={spec.name}",
    ]
    for key, value in args.rd:
        command.extend(("-rd", f"{key}={value}"))
    return command


def _close_log(running: RunningShard) -> None:
    if not running.log_file.closed:
        running.log_file.close()


def _signal_process_group(state: RunningShard, signum: int) -> None:
    """Signal a private child group, tolerating exit races."""

    try:
        if state.process_group_id is not None and hasattr(os, "killpg"):
            os.killpg(state.process_group_id, signum)
        elif signum == signal.SIGTERM:
            state.process.terminate()
        else:
            state.process.kill()
    except ProcessLookupError:
        pass


def _process_group_exists(state: RunningShard) -> bool:
    # Reap an exited leader promptly; otherwise its zombie can make a private
    # group look live for the entire grace period.
    state.process.poll()
    if state.process_group_id is None or not hasattr(os, "killpg"):
        return state.process.poll() is None
    try:
        os.killpg(state.process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_children(states: Sequence[RunningShard]) -> None:
    """Terminate child process groups, escalating to kill after a grace period."""

    # Signal every launched group, including one whose leader exited just before
    # cleanup: a wrapper may have left descendants running in its private group.
    for state in states:
        _signal_process_group(state, signal.SIGTERM)

    deadline = time.monotonic() + _TERMINATE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not any(_process_group_exists(state) for state in states):
            break
        time.sleep(_POLL_INTERVAL_SECONDS)

    for state in states:
        if _process_group_exists(state):
            _signal_process_group(state, signal.SIGKILL)
    for state in states:
        try:
            state.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _signal_process_group(state, signal.SIGKILL)
            state.process.wait()
        _close_log(state)


def _log_tail(path: Path, lines: int = 20) -> str:
    """Return a bounded diagnostic tail that survives temporary cleanup."""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return "".join(deque(stream, maxlen=lines)).rstrip()
    except OSError as exc:
        return f"<unable to read log: {exc}>"


def run_shards(
    args: argparse.Namespace,
    specs: Sequence[ShardSpec],
    jobs: int,
) -> tuple[list[ShardResult], float]:
    """Run shards with bounded concurrency and stop immediately on failure."""

    pending = deque(specs)
    running: dict[subprocess.Popen[bytes], RunningShard] = {}
    cleanup_states: list[RunningShard] = []
    completed: dict[int, ShardResult] = {}
    aggregate_started = time.monotonic()

    def launch(spec: ShardSpec) -> None:
        log_file = spec.log.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                build_command(args, spec),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=(os.name == "posix"),
            )
        except BaseException:
            log_file.close()
            raise
        state = RunningShard(
            spec=spec,
            process=process,
            process_group_id=(process.pid if os.name == "posix" else None),
            log_file=log_file,
            started=time.monotonic(),
        )
        running[process] = state
        cleanup_states.append(state)

    def sample_memory(state: RunningShard) -> None:
        peak_rss_kb = _process_peak_rss_kb(state.process.pid)
        if peak_rss_kb is not None:
            state.peak_rss_kb = max(state.peak_rss_kb or 0, peak_rss_kb)

    try:
        while pending or running:
            while pending and len(running) < jobs:
                launch(pending.popleft())

            # Sample before poll(), which may reap a completed process and
            # remove its /proc status entry.
            for state in running.values():
                sample_memory(state)
            finished = [process for process in running if process.poll() is not None]
            if not finished:
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # Consume failures first so no new work starts after a failed shard.
            failed = next((process for process in finished if process.returncode), None)
            if failed is not None:
                state = running[failed]
                _close_log(state)
                tail = _log_tail(state.spec.log)
                returncode = failed.returncode
                diagnostic = (
                    f"shard {state.spec.name!r} failed with exit code {returncode}; "
                    f"log: {state.spec.log}"
                )
                if tail:
                    diagnostic += f"\n--- log tail ---\n{tail}"
                raise ShardFailure(diagnostic)

            for process in finished:
                state = running.pop(process)
                _close_log(state)
                try:
                    report_size = state.spec.report.stat().st_size
                except OSError as exc:
                    tail = _log_tail(state.spec.log)
                    diagnostic = (
                        f"shard {state.spec.name!r} exited successfully but did not "
                        f"produce a report: {exc}"
                    )
                    if tail:
                        diagnostic += f"\n--- log tail ---\n{tail}"
                    raise ShardFailure(diagnostic) from exc
                if report_size == 0:
                    tail = _log_tail(state.spec.log)
                    diagnostic = (
                        f"shard {state.spec.name!r} exited successfully but produced "
                        "an empty report file"
                    )
                    if tail:
                        diagnostic += f"\n--- log tail ---\n{tail}"
                    raise ShardFailure(diagnostic)
                completed[state.spec.index] = ShardResult(
                    spec=state.spec,
                    returncode=process.returncode,
                    wall_seconds=time.monotonic() - state.started,
                    peak_rss_kb=state.peak_rss_kb,
                )
                # A successfully validated report completes this child.  Do
                # not retain a dead PGID long enough for unrelated PID reuse.
                cleanup_states.remove(state)
    except BaseException:
        _stop_children(cleanup_states)
        raise

    aggregate_wall = time.monotonic() - aggregate_started
    return [completed[spec.index] for spec in specs], aggregate_wall


def load_merger() -> tuple[Callable[..., dict[str, object]], Callable[..., dict[str, object]]]:
    """Import the sibling merger in direct-script and module execution modes."""

    if __package__:
        from .merge_sharded_lyrdb import merge_reports, validate_manifest
    else:
        from merge_sharded_lyrdb import merge_reports, validate_manifest
    return validate_manifest, merge_reports


def run(args: argparse.Namespace) -> int:
    """Execute one validated parallel run and publish its merged report."""

    started_utc = _utc_now()
    attempt_started = time.monotonic()
    initial_inputs: Mapping[str, Mapping[str, object]] | None = None
    initial_runtime_bundle: Mapping[str, object] | None = None
    initial_orchestrator: Mapping[str, Mapping[str, object]] | None = None
    temp_dir: Path | None = None
    specs: list[ShardSpec] = []
    jobs = min(args.jobs or len(args.shard), len(args.shard))
    results: list[ShardResult] = []
    children_wall: float | None = None
    prehash_wall: float | None = None
    merge_wall: float | None = None
    workload_wall: float | None = None
    verification_wall: float | None = None
    locks: _TargetLocks | None = None
    report_stage: Path | None = None
    metadata_stage: Path | None = None
    runtime_publication_targets_safe = False

    try:
        validate_args(args)
        output = Path(args.output)
        sidecar = metadata_path(args)
        locks = _TargetLocks.acquire(
            (output, sidecar, _failure_metadata_path(sidecar))
        )
        # Resolve the merger before doing expensive child work.  The import
        # remains lazy so --help works if this script is copied in isolation.
        validate_manifest, merge_reports = load_merger()
        prehash_started = time.monotonic()
        initial_inputs = _input_records(args)
        initial_runtime_bundle = _runtime_bundle(args.klayout)
        _validate_runtime_publication_targets(args, initial_runtime_bundle)
        runtime_publication_targets_safe = True
        initial_orchestrator = _orchestrator_record()
        validate_manifest(args.manifest, args.shard, args.deck)
        prehash_wall = time.monotonic() - prehash_started
        temp_dir = Path(tempfile.mkdtemp(prefix="klayout-parallel-drc-"))
        specs = [
            ShardSpec(
                index=index,
                name=name,
                report=temp_dir / f"shard-{index:03d}.lyrdb",
                log=temp_dir / f"shard-{index:03d}.log",
            )
            for index, name in enumerate(args.shard)
        ]
        jobs = min(args.jobs or len(specs), len(specs))
        workload_started = time.monotonic()
        results, children_wall = run_shards(args, specs, jobs)

        merge_started = time.monotonic()
        report_mode = _publication_mode(output)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.stage-", suffix=".lyrdb", dir=output.parent
        )
        os.close(descriptor)
        report_stage = Path(temporary)
        try:
            merge_reports(
                args.manifest,
                [(spec.name, spec.report) for spec in specs],
                report_stage,
                deck_path=args.deck,
            )
            os.chmod(report_stage, report_mode)
            _fsync_file(report_stage)
        except (OSError, ValueError) as exc:
            diagnostics = [f"report merge failed: {exc}"]
            for spec in specs:
                tail = _log_tail(spec.log)
                if tail:
                    diagnostics.append(f"--- shard {spec.name} log tail ---\n{tail}")
            raise ShardFailure("\n".join(diagnostics)) from exc
        merge_wall = time.monotonic() - merge_started
        workload_wall = time.monotonic() - workload_started

        # Refuse to publish a report if any supposedly immutable benchmark
        # input or optimized runtime component changed underneath the workload.
        verification_started = time.monotonic()
        final_inputs = _input_records(args)
        final_runtime_bundle = _runtime_bundle(args.klayout)
        final_orchestrator = _orchestrator_record()
        _verify_runtime_unchanged(
            initial_inputs,
            final_inputs,
            initial_runtime_bundle,
            final_runtime_bundle,
            initial_orchestrator,
            final_orchestrator,
        )
        verification_wall = time.monotonic() - verification_started

        output_report = _staged_output_record(report_stage, output)
        shard_records = [
            {
                "index": result.spec.index,
                "name": result.spec.name,
                "command": build_command(args, result.spec),
                "returncode": result.returncode,
                "wall_seconds": result.wall_seconds,
                "peak_rss_kb": result.peak_rss_kb,
                "peak_rss_source": _peak_rss_source(result.peak_rss_kb),
                "peak_rss_caveat": _PEAK_RSS_CAVEAT,
                "report": _artifact_record(result.spec.report, args.keep_temp),
                "log": _artifact_record(result.spec.log, args.keep_temp),
            }
            for result in results
        ]
        provenance = _provenance_prefix(
            args,
            specs,
            jobs,
            started_utc,
            initial_inputs,
            initial_runtime_bundle,
            initial_orchestrator,
        )
        timing: dict[str, float | None] = {
            "provenance_prehash_wall_seconds": prehash_wall,
            "children_wall_seconds": children_wall,
            "merge_wall_seconds": merge_wall,
            "child_merge_workload_wall_seconds": workload_wall,
            # Compatibility alias retained for consumers of v1 files.
            "aggregate_wall_seconds": workload_wall,
            "provenance_verification_wall_seconds": verification_wall,
            "full_launcher_wall_seconds": None,
            # Compatibility alias for early provenance consumers.
            "launcher_wall_seconds": None,
        }
        provenance.update(
            {
                "status": "success",
                "completed_utc": _utc_now(),
                "timing": timing,
                "measurement_semantics": _measurement_semantics(),
                "shards": shard_records,
                "temporary_artifacts": {
                    "retained": args.keep_temp,
                    "directory": str(temp_dir.resolve()) if args.keep_temp else None,
                },
                "output_report": output_report,
            }
        )
        launcher_wall = time.monotonic() - attempt_started
        timing["full_launcher_wall_seconds"] = launcher_wall
        timing["launcher_wall_seconds"] = launcher_wall
        # Both artifacts are complete and durable in their target directories
        # before either public name changes.
        metadata_stage = _stage_json(sidecar, provenance)
        _publish_staged_pair(report_stage, output, metadata_stage, sidecar)
        report_stage = None
        metadata_stage = None

        for result in results:
            print(f"shard {result.spec.name}: {result.wall_seconds:.3f} s")
        print(
            f"children: {children_wall:.3f} s "
            f"({len(results)} shards, {jobs} jobs)"
        )
        print(f"merge: {merge_wall:.3f} s")
        print(f"child+merge workload: {workload_wall:.3f} s")
        print(f"provenance verification: {verification_wall:.3f} s")
        print(
            "full launcher wall (through pre-publication provenance assembly): "
            f"{launcher_wall:.3f} s"
        )
        print(f"merged report: {args.output}")
        print(f"provenance: {sidecar}")
        return 0
    except BaseException as exc:
        # Failure data uses a distinct path.  A lock-acquisition failure must
        # not write into another launcher's target namespace.
        if locks is not None and runtime_publication_targets_safe:
            try:
                failed = _provenance_prefix(
                    args,
                    specs,
                    jobs,
                    started_utc,
                    initial_inputs,
                    initial_runtime_bundle,
                    initial_orchestrator,
                )
                failed.update(
                    {
                        "status": "failed",
                        "failed_utc": _utc_now(),
                        "failure": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "timing": {
                            "provenance_prehash_wall_seconds": prehash_wall,
                            "children_wall_seconds": children_wall,
                            "merge_wall_seconds": merge_wall,
                            "child_merge_workload_wall_seconds": workload_wall,
                            "provenance_verification_wall_seconds": verification_wall,
                            "attempt_wall_seconds": time.monotonic() - attempt_started,
                        },
                        "measurement_semantics": _measurement_semantics(),
                        "shards": [
                            {
                                "index": spec.index,
                                "name": spec.name,
                                "command": build_command(args, spec),
                                "report_path": (
                                    str(spec.report.resolve())
                                    if args.keep_temp
                                    else None
                                ),
                                "log_path": (
                                    str(spec.log.resolve())
                                    if args.keep_temp
                                    else None
                                ),
                            }
                            for spec in specs
                        ],
                        "temporary_artifacts": {
                            "retained": args.keep_temp,
                            "directory": (
                                str(temp_dir.resolve())
                                if args.keep_temp and temp_dir is not None
                                else None
                            ),
                        },
                        "output_report": None,
                    }
                )
                failure_sidecar = _failure_metadata_path(metadata_path(args))
                _atomic_json(failure_sidecar, failed)
                print(f"failure provenance: {failure_sidecar}", file=sys.stderr)
            except BaseException as metadata_exc:
                print(
                    f"warning: unable to write failure provenance: {metadata_exc}",
                    file=sys.stderr,
                )
        raise
    finally:
        for staged in (report_stage, metadata_stage):
            if staged is None:
                continue
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
        if temp_dir is None:
            pass
        elif args.keep_temp:
            print(f"temporary artifacts: {temp_dir}", file=sys.stderr)
        else:
            try:
                shutil.rmtree(temp_dir)
            except OSError as exc:
                print(
                    f"warning: unable to remove temporary directory {temp_dir}: {exc}",
                    file=sys.stderr,
                )
        if locks is not None:
            locks.release()


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point with concise user-facing errors."""

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_termination(signum: int, _frame: object) -> None:
        raise TerminationRequested(signum)

    signal.signal(signal.SIGTERM, request_termination)
    try:
        try:
            return run(parse_args(argv))
        except (
            ImportError,
            OSError,
            ValueError,
            ShardFailure,
            TargetLockedError,
        ) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except TerminationRequested as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 128 + exc.signum
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
