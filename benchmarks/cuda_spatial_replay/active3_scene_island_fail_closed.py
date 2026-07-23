#!/usr/bin/env python3
"""Adversarial fail-closed gates for the standalone ACTIVE.3 GPU island."""

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import tempfile
from pathlib import Path


HEADER = struct.Struct("<8s14Id18Q32sQ")
INSTANCE = struct.Struct("<4Q6q4I")


def resign(data: bytearray) -> None:
    digest_input = bytearray(data)
    digest_input[216:256] = b"\x00" * 40
    data[216:248] = hashlib.sha256(digest_input).digest()


def run_uncertain(binary: Path, arguments: list[str], label: str) -> None:
    result = subprocess.run(
        [str(binary), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 2 or "verdict=UNCERTAIN" not in result.stdout:
        raise RuntimeError(
            f"{label}: expected fail-closed rc=2, got rc={result.returncode}\n"
            f"{result.stdout}"
        )
    print(f"ACTIVE3_FAIL_CLOSED ok gate={label}")


def expected(data: bytearray) -> str:
    return "--expect-scene-sha256=" + bytes(data[216:248]).hex()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    scene = args.scene.resolve(strict=True)
    original = bytearray(scene.read_bytes())
    values = HEADER.unpack_from(original)
    root_cell = values[16]
    instance_count = values[18]
    instances_offset = values[25]
    edges_offset = values[29]

    with tempfile.TemporaryDirectory(
        prefix="active3-island-fail-closed-", dir="/tmp"
    ) as temporary:
        root = Path(temporary)

        truncated = root / "truncated.kact"
        truncated.write_bytes(original[:100])
        run_uncertain(binary, [expected(original), str(truncated)], "truncated")

        corrupt = bytearray(original)
        corrupt[-1] ^= 1
        corrupt_path = root / "corrupt-hash.kact"
        corrupt_path.write_bytes(corrupt)
        run_uncertain(
            binary, [expected(original), str(corrupt_path)], "corrupt-hash"
        )

        dbu = bytearray(original)
        struct.pack_into("<d", dbu, 64, 0.0007)
        resign(dbu)
        dbu_path = root / "unsupported-dbu.kact"
        dbu_path.write_bytes(dbu)
        run_uncertain(
            binary, [expected(dbu), str(dbu_path)], "nonintegral-distance-dbu"
        )

        reserved = bytearray(original)
        struct.pack_into("<Q", reserved, 248, 1)
        resign(reserved)
        reserved_path = root / "reserved.kact"
        reserved_path.write_bytes(reserved)
        run_uncertain(
            binary, [expected(reserved), str(reserved_path)], "reserved-header"
        )

        cycle = bytearray(original)
        root_instance = None
        for index in range(instance_count):
            offset = instances_offset + index * INSTANCE.size
            record = INSTANCE.unpack_from(cycle, offset)
            if record[1] == root_cell:
                root_instance = offset
                break
        if root_instance is None:
            raise RuntimeError("base scene root has no instance for cycle gate")
        struct.pack_into("<Q", cycle, root_instance + 16, root_cell)
        resign(cycle)
        cycle_path = root / "cycle.kact"
        cycle_path.write_bytes(cycle)
        run_uncertain(
            binary, [expected(cycle), str(cycle_path)], "hierarchy-cycle"
        )

        coordinate = bytearray(original)
        struct.pack_into("<q", coordinate, edges_offset + 16, 1_000_000_000_001)
        resign(coordinate)
        coordinate_path = root / "coordinate-bound.kact"
        coordinate_path.write_bytes(coordinate)
        run_uncertain(
            binary,
            [expected(coordinate), str(coordinate_path)],
            "coordinate-overflow-domain",
        )

        run_uncertain(
            binary,
            [expected(original), "--max-contexts=1", str(scene)],
            "context-capacity",
        )
        run_uncertain(
            binary,
            [expected(original), "--max-grid-cells=1", str(scene)],
            "grid-capacity",
        )
        run_uncertain(
            binary,
            [expected(original), "--max-memberships=1", str(scene)],
            "membership-capacity",
        )
        run_uncertain(
            binary,
            [expected(original), "--max-pair-work=1", str(scene)],
            "pair-work-capacity",
        )
        run_uncertain(
            binary,
            ["--expect-scene-sha256=" + "0" * 64, str(scene)],
            "fingerprint-mismatch",
        )
        run_uncertain(binary, [str(scene)], "fingerprint-missing")

    print("ACTIVE3_FAIL_CLOSED all_ok gates=12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
