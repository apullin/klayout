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

Example::

    python3 scripts/run_parallel_drc.py \
      --klayout ./bin-clang/klayout \
      --deck /path/to/freepdk45-sharded.lydrc \
      --input /path/to/design.gds \
      --top-cell design \
      --output /path/to/design.drc.report \
      --manifest /path/to/freepdk45-shards.json \
      --shard active_m1 --shard other

Only Python's standard library is used by this launcher.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence, TextIO


_RESERVED_RD_KEYS = frozenset({"input", "topcell", "output", "drc_shard"})
_POLL_INTERVAL_SECONDS = 0.05
_TERMINATE_TIMEOUT_SECONDS = 5.0


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


@dataclass(frozen=True)
class ShardResult:
    """Timing and status returned by one completed shard."""

    spec: ShardSpec
    returncode: int
    wall_seconds: float


class ShardFailure(RuntimeError):
    """Raised when a KLayout shard exits unsuccessfully."""


class TerminationRequested(BaseException):
    """Convert SIGTERM into normal child and temporary-artifact cleanup."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


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
            "are forwarded to every child, but input, topcell, output and "
            "drc_shard are reserved. Child stdout and stderr are combined in "
            "one per-shard log."
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
        help="top cell passed to the deck as the topcell run-time variable",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="REPORT",
        help="final merged .lyrdb/report path; never written by a child",
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
        "--keep-temp",
        action="store_true",
        help="retain the temporary directory containing shard reports and logs",
    )
    return parser.parse_args(argv)


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

    output = Path(args.output)
    if output.exists() and not output.is_file():
        raise ValueError(f"--output is not a regular file path: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"--output parent directory does not exist: {output.parent}")

    resolved_output = output.resolve()
    protected_inputs = {
        Path(args.deck).resolve(),
        Path(args.input).resolve(),
        Path(args.manifest).resolve(),
    }
    if resolved_output in protected_inputs:
        raise ValueError("--output must not overwrite the deck, input, or manifest")


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
        f"topcell={args.top_cell}",
        "-rd",
        f"output={spec.report}",
        "-rd",
        f"drc_shard={spec.name}",
    ]
    for key, value in args.rd:
        command.extend(("-rd", f"{key}={value}"))
    return command


def _close_log(running: RunningShard) -> None:
    if not running.log_file.closed:
        running.log_file.close()


def _stop_children(running: Mapping[subprocess.Popen[bytes], RunningShard]) -> None:
    """Terminate all live children, escalating to kill after a short grace period."""

    live = [state for state in running.values() if state.process.poll() is None]
    for state in live:
        state.process.terminate()

    deadline = time.monotonic() + _TERMINATE_TIMEOUT_SECONDS
    for state in live:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

    for state in live:
        if state.process.poll() is None:
            state.process.kill()
    for state in live:
        state.process.wait()

    for state in running.values():
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
            )
        except BaseException:
            log_file.close()
            raise
        running[process] = RunningShard(
            spec=spec,
            process=process,
            log_file=log_file,
            started=time.monotonic(),
        )

    try:
        while pending or running:
            while pending and len(running) < jobs:
                launch(pending.popleft())

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
                )
    except BaseException:
        _stop_children(running)
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

    validate_args(args)
    # Resolve the merger before doing expensive child work.  The import remains
    # lazy so ``--help`` is useful even if this script is copied in isolation.
    validate_manifest, merge_reports = load_merger()
    validate_manifest(args.manifest, args.shard, args.deck)
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
    total_started = time.monotonic()

    try:
        results, children_wall = run_shards(args, specs, jobs)
        for result in results:
            print(f"shard {result.spec.name}: {result.wall_seconds:.3f} s")
        print(
            f"children aggregate: {children_wall:.3f} s "
            f"({len(results)} shards, {jobs} jobs)"
        )

        merge_started = time.monotonic()
        try:
            merge_reports(
                args.manifest,
                [(spec.name, spec.report) for spec in specs],
                args.output,
                deck_path=args.deck,
            )
        except (OSError, ValueError) as exc:
            diagnostics = [f"report merge failed: {exc}"]
            for spec in specs:
                tail = _log_tail(spec.log)
                if tail:
                    diagnostics.append(f"--- shard {spec.name} log tail ---\n{tail}")
            raise ShardFailure("\n".join(diagnostics)) from exc
        merge_wall = time.monotonic() - merge_started
        total_wall = time.monotonic() - total_started
        print(f"merge: {merge_wall:.3f} s")
        print(f"aggregate wall: {total_wall:.3f} s")
        print(f"merged report: {args.output}")
        return 0
    finally:
        if args.keep_temp:
            print(f"temporary artifacts: {temp_dir}", file=sys.stderr)
        else:
            try:
                shutil.rmtree(temp_dir)
            except OSError as exc:
                print(
                    f"warning: unable to remove temporary directory {temp_dir}: {exc}",
                    file=sys.stderr,
                )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point with concise user-facing errors."""

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_termination(signum: int, _frame: object) -> None:
        raise TerminationRequested(signum)

    signal.signal(signal.SIGTERM, request_termination)
    try:
        try:
            return run(parse_args(argv))
        except (ImportError, OSError, ValueError, ShardFailure) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except TerminationRequested as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 128 + exc.signum
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
