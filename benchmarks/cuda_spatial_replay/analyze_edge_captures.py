#!/usr/bin/env python3
"""Validate and summarize one or more KEDGER1 edge-scanner captures."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import pathlib
import struct
import sys
from collections import Counter
from typing import Iterable, Sequence


HEADER = struct.Struct("<8s14I3q13Q")
RECORD = struct.Struct("<8qQ4I8x")
PAIR = struct.Struct("<2I")

MAGIC = b"KEDGER1\0"
VERSION = 1
HEADER_SIZE = 192
RECORD_SIZE = 96
PAIR_SIZE = 8

HEADER_FIELDS = (
    "magic",
    "version",
    "header_size",
    "record_size",
    "pair_size",
    "coordinate_bits",
    "property_bits",
    "capture_flags",
    "relation",
    "metrics",
    "zero_distance_mode",
    "ignore_angle_millidegrees",
    "option_flags",
    "process_id",
    "reserved",
    "distance",
    "min_projection",
    "max_projection",
    "request_id",
    "thread_tag",
    "scanner_elapsed_ns",
    "record_count",
    "scanner_callbacks",
    "finish_callbacks",
    "unresolved_callbacks",
    "broad_pair_count",
    "exact_accept_callbacks",
    "exact_pair_count",
    "records_offset",
    "broad_pairs_offset",
    "exact_pairs_offset",
)


class CaptureError(ValueError):
    """A malformed or internally inconsistent capture."""


@dataclasses.dataclass(frozen=True)
class Capture:
    path: pathlib.Path
    process_id: int
    thread_tag: int
    request_id: int
    distance: int
    relation: int
    metrics: int
    zero_distance_mode: int
    ignore_angle_millidegrees: int
    option_flags: int
    capture_flags: int
    scanner_elapsed_ns: int
    record_count: int
    scanner_callbacks: int
    finish_callbacks: int
    unresolved_callbacks: int
    broad_pair_count: int
    exact_accept_callbacks: int
    exact_pair_count: int
    file_bytes: int
    file_sha256: bytes
    records_sha256: bytes
    broad_pairs_sha256: bytes
    exact_pairs_sha256: bytes

    @property
    def request_label(self) -> str:
        return f"p{self.process_id}/t{self.thread_tag:016x}/r{self.request_id}"


def fail(path: pathlib.Path, message: str) -> CaptureError:
    return CaptureError(f"{path}: {message}")


def checked_section_end(
    path: pathlib.Path, offset: int, count: int, size: int, label: str
) -> int:
    if offset < 0 or count < 0:
        raise fail(path, f"negative {label} offset or count")
    end = offset + count * size
    if end < offset:
        raise fail(path, f"{label} section size overflow")
    return end


def validate_pairs(
    path: pathlib.Path,
    section: memoryview,
    count: int,
    record_count: int,
    label: str,
) -> list[int]:
    keys: list[int] = []
    previous = -1
    for index in range(count):
        first, second = PAIR.unpack_from(section, index * PAIR_SIZE)
        if first == 0 or second == 0:
            raise fail(path, f"{label}[{index}] uses reserved record ID zero")
        if first >= second:
            raise fail(path, f"{label}[{index}] is not an ascending ID pair")
        if second > record_count:
            raise fail(
                path,
                f"{label}[{index}] record ID {second} exceeds {record_count}",
            )
        key = (first << 32) | second
        if key <= previous:
            raise fail(path, f"{label} is not strictly sorted and unique at {index}")
        keys.append(key)
        previous = key
    return keys


def validate_exact_subset(
    path: pathlib.Path, broad: Sequence[int], exact: Sequence[int]
) -> None:
    broad_index = 0
    for exact_index, key in enumerate(exact):
        while broad_index < len(broad) and broad[broad_index] < key:
            broad_index += 1
        if broad_index == len(broad) or broad[broad_index] != key:
            raise fail(
                path,
                f"exact pair {exact_index} is absent from the broad-pair oracle",
            )


def parse_capture(path: pathlib.Path) -> Capture:
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise fail(path, f"file is only {len(data)} bytes; header needs {HEADER_SIZE}")

    values = dict(zip(HEADER_FIELDS, HEADER.unpack_from(data)))
    if values["magic"] != MAGIC:
        raise fail(path, f"bad magic {values['magic']!r}")
    if values["version"] != VERSION:
        raise fail(path, f"unsupported version {values['version']}")
    if values["header_size"] != HEADER_SIZE:
        raise fail(path, f"unexpected header size {values['header_size']}")
    if values["record_size"] != RECORD_SIZE:
        raise fail(path, f"unexpected record size {values['record_size']}")
    if values["pair_size"] != PAIR_SIZE:
        raise fail(path, f"unexpected pair size {values['pair_size']}")
    if values["coordinate_bits"] != 64 or values["property_bits"] != 64:
        raise fail(
            path,
            "only 64-bit coordinates and properties are supported "
            f"(got {values['coordinate_bits']}/{values['property_bits']})",
        )
    if values["reserved"] != 0:
        raise fail(path, f"nonzero reserved header field {values['reserved']}")

    records_end = checked_section_end(
        path,
        values["records_offset"],
        values["record_count"],
        RECORD_SIZE,
        "records",
    )
    broad_end = checked_section_end(
        path,
        values["broad_pairs_offset"],
        values["broad_pair_count"],
        PAIR_SIZE,
        "broad pairs",
    )
    exact_end = checked_section_end(
        path,
        values["exact_pairs_offset"],
        values["exact_pair_count"],
        PAIR_SIZE,
        "exact pairs",
    )
    if values["records_offset"] != HEADER_SIZE:
        raise fail(path, f"records start at {values['records_offset']}, not 192")
    if values["broad_pairs_offset"] != records_end:
        raise fail(path, "broad-pair section is not contiguous after records")
    if values["exact_pairs_offset"] != broad_end:
        raise fail(path, "exact-pair section is not contiguous after broad pairs")
    if exact_end != len(data):
        raise fail(
            path,
            f"length is {len(data)} bytes but sections require {exact_end}",
        )

    if values["record_count"] > 0xFFFFFFFF:
        raise fail(path, "record count cannot be represented by one-based uint32 IDs")
    if values["unresolved_callbacks"] > values["scanner_callbacks"]:
        raise fail(path, "unresolved callbacks exceed raw scanner callbacks")
    if values["broad_pair_count"] > values["scanner_callbacks"]:
        raise fail(path, "unique broad pairs exceed raw scanner callbacks")
    if values["exact_accept_callbacks"] > values["scanner_callbacks"]:
        raise fail(path, "raw exact acceptances exceed raw scanner callbacks")
    if values["exact_pair_count"] > values["broad_pair_count"]:
        raise fail(path, "unique exact pairs exceed unique broad pairs")
    if values["exact_pair_count"] > values["exact_accept_callbacks"]:
        raise fail(path, "unique exact pairs exceed raw exact acceptances")

    all_bytes = memoryview(data)
    records = all_bytes[values["records_offset"] : records_end]
    broad_bytes = all_bytes[values["broad_pairs_offset"] : broad_end]
    exact_bytes = all_bytes[values["exact_pairs_offset"] : exact_end]

    for index in range(values["record_count"]):
        (
            left,
            bottom,
            right,
            top,
            x1,
            y1,
            x2,
            y2,
            property_value,
            record_id,
            context,
            flags,
            reserved,
        ) = RECORD.unpack_from(records, index * RECORD_SIZE)
        expected_id = index + 1
        if record_id != expected_id:
            raise fail(
                path,
                f"record {index} has ID {record_id}, expected {expected_id}",
            )
        if (left, bottom, right, top) != (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ):
            raise fail(path, f"record ID {record_id} AABB does not match endpoints")
        if not flags & 1:
            raise fail(path, f"record ID {record_id} has no endpoint flag")
        if bool(flags & 2) != bool(property_value & 1):
            raise fail(path, f"record ID {record_id} side flag disagrees with property")
        if flags & ~3:
            raise fail(path, f"record ID {record_id} has unknown flags 0x{flags:x}")
        if context != 0 or reserved != 0:
            raise fail(path, f"record ID {record_id} has nonzero reserved data")

    broad = validate_pairs(
        path,
        broad_bytes,
        values["broad_pair_count"],
        values["record_count"],
        "broad pairs",
    )
    exact = validate_pairs(
        path,
        exact_bytes,
        values["exact_pair_count"],
        values["record_count"],
        "exact pairs",
    )
    validate_exact_subset(path, broad, exact)

    return Capture(
        path=path,
        process_id=values["process_id"],
        thread_tag=values["thread_tag"],
        request_id=values["request_id"],
        distance=values["distance"],
        relation=values["relation"],
        metrics=values["metrics"],
        zero_distance_mode=values["zero_distance_mode"],
        ignore_angle_millidegrees=values["ignore_angle_millidegrees"],
        option_flags=values["option_flags"],
        capture_flags=values["capture_flags"],
        scanner_elapsed_ns=values["scanner_elapsed_ns"],
        record_count=values["record_count"],
        scanner_callbacks=values["scanner_callbacks"],
        finish_callbacks=values["finish_callbacks"],
        unresolved_callbacks=values["unresolved_callbacks"],
        broad_pair_count=values["broad_pair_count"],
        exact_accept_callbacks=values["exact_accept_callbacks"],
        exact_pair_count=values["exact_pair_count"],
        file_bytes=len(data),
        file_sha256=hashlib.sha256(data).digest(),
        records_sha256=hashlib.sha256(records).digest(),
        broad_pairs_sha256=hashlib.sha256(broad_bytes).digest(),
        exact_pairs_sha256=hashlib.sha256(exact_bytes).digest(),
    )


def discover_paths(arguments: Sequence[str]) -> list[pathlib.Path]:
    found: dict[pathlib.Path, None] = {}
    for argument in arguments:
        path = pathlib.Path(argument)
        if path.is_dir():
            candidates: Iterable[pathlib.Path] = path.rglob("*.ker")
        elif path.is_file():
            candidates = (path,)
        else:
            raise CaptureError(f"{path}: no such file or directory")
        for candidate in candidates:
            found[candidate.resolve()] = None
    return sorted(found)


def percentile(sorted_values: Sequence[int], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def metric_row(
    label: str, values: Sequence[int], scale: float = 1.0, decimals: int = 1
) -> str:
    ordered = sorted(values)
    total = sum(values) / scale
    mean = total / len(values)
    samples = (
        ordered[0] / scale,
        percentile(ordered, 0.50) / scale,
        percentile(ordered, 0.90) / scale,
        percentile(ordered, 0.95) / scale,
        percentile(ordered, 0.99) / scale,
        ordered[-1] / scale,
    )
    number = f"{{:.{decimals}f}}"
    fields = " ".join(f"{number.format(value):>12}" for value in samples)
    return (
        f"{label:<22} {fields} {number.format(mean):>12} "
        f"{number.format(total):>15}"
    )


def requests_for_share(values: Sequence[int], fraction: float) -> int:
    target = sum(values) * fraction
    accumulated = 0
    for count, value in enumerate(sorted(values, reverse=True), start=1):
        accumulated += value
        if accumulated >= target:
            return count
    return 0


def concentration_row(label: str, values: Sequence[int]) -> str:
    ordered = sorted(values, reverse=True)
    total = sum(ordered)

    def top_share(count: int) -> str:
        if total == 0:
            return "n/a"
        return f"{100.0 * sum(ordered[:count]) / total:.2f}%"

    top_one_percent = max(1, math.ceil(len(values) * 0.01))
    return (
        f"{label:<22} {top_share(1):>10} {top_share(10):>10} "
        f"{top_share(top_one_percent):>10} "
        f"{requests_for_share(values, 0.50):>10,d} "
        f"{requests_for_share(values, 0.90):>10,d} "
        f"{requests_for_share(values, 0.99):>10,d}"
    )


def ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.3f}%"


def aggregate_digest(captures: Sequence[Capture], attribute: str) -> str:
    digest = hashlib.sha256()
    for capture in sorted(
        captures,
        key=lambda item: (
            item.process_id,
            item.request_id,
            item.thread_tag,
            item.path.name,
        ),
    ):
        digest.update(struct.pack("<IQQ", capture.process_id, capture.request_id, capture.thread_tag))
        digest.update(getattr(capture, attribute))
    return digest.hexdigest()


def print_top(captures: Sequence[Capture], attribute: str, label: str, count: int) -> None:
    print(f"\nTop {min(count, len(captures))} by {label}")
    print(
        "request                            records       raw_cb        broad"
        "        exact   elapsed_ms        bytes"
    )
    ordered = sorted(
        captures,
        key=lambda item: (
            -getattr(item, attribute),
            item.process_id,
            item.request_id,
            item.thread_tag,
        ),
    )
    for capture in ordered[:count]:
        print(
            f"{capture.request_label:<34} "
            f"{capture.record_count:>10,d} "
            f"{capture.scanner_callbacks:>12,d} "
            f"{capture.broad_pair_count:>12,d} "
            f"{capture.exact_pair_count:>12,d} "
            f"{capture.scanner_elapsed_ns / 1_000_000:>12.3f} "
            f"{capture.file_bytes:>12,d}"
        )


def summarize(captures: Sequence[Capture], top_count: int) -> None:
    metrics = {
        "record_count": [capture.record_count for capture in captures],
        "scanner_callbacks": [capture.scanner_callbacks for capture in captures],
        "broad_pair_count": [capture.broad_pair_count for capture in captures],
        "exact_pair_count": [capture.exact_pair_count for capture in captures],
        "scanner_elapsed_ns": [capture.scanner_elapsed_ns for capture in captures],
        "file_bytes": [capture.file_bytes for capture in captures],
    }
    finish_callbacks = sum(capture.finish_callbacks for capture in captures)
    unresolved_callbacks = sum(capture.unresolved_callbacks for capture in captures)
    exact_accept_callbacks = sum(
        capture.exact_accept_callbacks for capture in captures
    )
    total_records = sum(metrics["record_count"])
    total_callbacks = sum(metrics["scanner_callbacks"])
    total_broad = sum(metrics["broad_pair_count"])
    total_exact = sum(metrics["exact_pair_count"])
    total_bytes = sum(metrics["file_bytes"])

    profiles = Counter(
        (
            capture.distance,
            capture.relation,
            capture.metrics,
            capture.zero_distance_mode,
            capture.ignore_angle_millidegrees,
            capture.option_flags,
            capture.capture_flags,
        )
        for capture in captures
    )
    processes = {capture.process_id for capture in captures}
    threads = {(capture.process_id, capture.thread_tag) for capture in captures}

    print(f"Validated {len(captures):,} KEDGER1 captures")
    print(
        f"Processes/threads: {len(processes):,}/{len(threads):,}; "
        f"profiles: {len(profiles):,}; bytes: {total_bytes:,} "
        f"({total_bytes / (1024 * 1024):.3f} MiB)"
    )
    print(
        "Aggregate: "
        f"records={total_records:,} raw_scanner_callbacks={total_callbacks:,} "
        f"finish_callbacks={finish_callbacks:,} unresolved={unresolved_callbacks:,} "
        f"relevant_broad_pairs={total_broad:,} "
        f"raw_exact_acceptances={exact_accept_callbacks:,} "
        f"unique_exact_pairs={total_exact:,}"
    )
    print(
        "Selectivity: "
        f"broad/raw_callbacks={ratio(total_broad, total_callbacks)} "
        f"exact/broad={ratio(total_exact, total_broad)} "
        f"exact_filter_rejects={ratio(total_broad - total_exact, total_broad)} "
        f"unique/raw_exact={ratio(total_exact, exact_accept_callbacks)}"
    )
    print(
        "Request populations: "
        f"broad_empty={sum(value == 0 for value in metrics['broad_pair_count']):,} "
        f"exact_empty={sum(value == 0 for value in metrics['exact_pair_count']):,} "
        f"zero_elapsed={sum(value == 0 for value in metrics['scanner_elapsed_ns']):,} "
        f"unresolved_nonzero={sum(capture.unresolved_callbacks != 0 for capture in captures):,}"
    )

    print("\nPer-request distribution")
    print(
        f"{'metric':<22} {'min':>12} {'p50':>12} {'p90':>12} {'p95':>12} "
        f"{'p99':>12} {'max':>12} {'mean':>12} {'total':>15}"
    )
    print(metric_row("records", metrics["record_count"], decimals=1))
    print(metric_row("raw scanner callbacks", metrics["scanner_callbacks"], decimals=1))
    print(metric_row("relevant broad pairs", metrics["broad_pair_count"], decimals=1))
    print(metric_row("unique exact pairs", metrics["exact_pair_count"], decimals=1))
    print(metric_row("scanner elapsed (ms)", metrics["scanner_elapsed_ns"], 1_000_000, 3))
    print(metric_row("file bytes (KiB)", metrics["file_bytes"], 1024, 3))

    print("\nWork concentration")
    print(
        f"{'metric':<22} {'top 1':>10} {'top 10':>10} {'top 1%':>10} "
        f"{'reqs@50%':>10} {'reqs@90%':>10} {'reqs@99%':>10}"
    )
    print(concentration_row("records", metrics["record_count"]))
    print(concentration_row("raw scanner callbacks", metrics["scanner_callbacks"]))
    print(concentration_row("relevant broad pairs", metrics["broad_pair_count"]))
    print(concentration_row("unique exact pairs", metrics["exact_pair_count"]))
    print(concentration_row("scanner elapsed", metrics["scanner_elapsed_ns"]))
    print(concentration_row("file bytes", metrics["file_bytes"]))

    print("\nBuckets by records per request")
    print(
        f"{'records':<13} {'requests':>10} {'records':>14} {'raw_cb':>14} "
        f"{'broad':>14} {'exact':>14} {'elapsed_ms':>14}"
    )
    buckets = (
        ("1-15", 1, 16),
        ("16-63", 16, 64),
        ("64-255", 64, 256),
        ("256-1023", 256, 1024),
        ("1024+", 1024, None),
    )
    for label, lower, upper in buckets:
        selected = [
            capture
            for capture in captures
            if capture.record_count >= lower
            and (upper is None or capture.record_count < upper)
        ]
        print(
            f"{label:<13} {len(selected):>10,d} "
            f"{sum(item.record_count for item in selected):>14,d} "
            f"{sum(item.scanner_callbacks for item in selected):>14,d} "
            f"{sum(item.broad_pair_count for item in selected):>14,d} "
            f"{sum(item.exact_pair_count for item in selected):>14,d} "
            f"{sum(item.scanner_elapsed_ns for item in selected) / 1_000_000:>14.3f}"
        )

    if len(threads) <= 16:
        print("\nPer-thread aggregate")
        print(
            f"{'process/thread':<30} {'requests':>10} {'records':>14} "
            f"{'broad':>14} {'exact':>14} {'elapsed_ms':>14}"
        )
        for process_thread in sorted(threads):
            selected = [
                capture
                for capture in captures
                if (capture.process_id, capture.thread_tag) == process_thread
            ]
            process_id, thread_tag = process_thread
            print(
                f"{f'p{process_id}/t{thread_tag:016x}':<30} "
                f"{len(selected):>10,d} "
                f"{sum(item.record_count for item in selected):>14,d} "
                f"{sum(item.broad_pair_count for item in selected):>14,d} "
                f"{sum(item.exact_pair_count for item in selected):>14,d} "
                f"{sum(item.scanner_elapsed_ns for item in selected) / 1_000_000:>14.3f}"
            )

    if len(profiles) <= 12:
        print("\nProfiles (requests)")
        for profile, count in sorted(profiles.items()):
            distance, relation, metrics_value, zero, angle, options, flags = profile
            print(
                f"{count:>7,d}  distance={distance} relation={relation} "
                f"metrics={metrics_value} zero={zero} angle_mdeg={angle} "
                f"options=0x{options:x} capture_flags=0x{flags:x}"
            )

    print("\nAggregate SHA-256 (logical request order)")
    print(f"files:       {aggregate_digest(captures, 'file_sha256')}")
    print(f"records:     {aggregate_digest(captures, 'records_sha256')}")
    print(f"broad pairs: {aggregate_digest(captures, 'broad_pairs_sha256')}")
    print(f"exact pairs: {aggregate_digest(captures, 'exact_pairs_sha256')}")

    for attribute, label in (
        ("record_count", "records"),
        ("scanner_callbacks", "raw scanner callbacks"),
        ("broad_pair_count", "relevant broad pairs"),
        ("exact_pair_count", "unique exact pairs"),
        ("scanner_elapsed_ns", "scanner elapsed"),
        ("file_bytes", "file bytes"),
    ):
        print_top(captures, attribute, label, top_count)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate KEDGER1 headers, section lengths, records, sorted pair "
            "oracles, and exact-pair subset relationships; then report workload "
            "size and scanner-time distributions."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="capture file or directory (directories are searched recursively)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="number of largest requests to print per metric (default: 5)",
    )
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error("--top must be nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        paths = discover_paths(args.paths)
        if not paths:
            raise CaptureError("no .ker capture files found")
        captures = [parse_capture(path) for path in paths]
        summarize(captures, args.top)
    except (CaptureError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
