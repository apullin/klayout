#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${1:-${here}/build-m2-flat-region-bridge}"
qmake_build="${KLAYOUT_QMAKE_BUILD_DIR:-}"
oracle="${KLAYOUT_M2_FLAT_BRIDGE_ORACLE:-}"

if [[ -z "${qmake_build}" ]]; then
  echo "KLAYOUT_QMAKE_BUILD_DIR must name a built KLayout qmake tree" >&2
  exit 2
fi
if [[ -z "${oracle}" ]]; then
  echo "KLAYOUT_M2_FLAT_BRIDGE_ORACLE must name the pinned KM1WS file" >&2
  exit 2
fi

mkdir -p "${build}/tmp"
env TMPDIR="${build}/tmp" cmake \
  -S "${here}" -B "${build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DKLAYOUT_QMAKE_BUILD_DIR="${qmake_build}"
env TMPDIR="${build}/tmp" cmake --build "${build}" \
  --target m2_flat_region_bridge

export LD_LIBRARY_PATH="${qmake_build}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${build}/m2_flat_region_bridge" --self-test
"${build}/m2_flat_region_bridge" \
  --audit-full-boundary \
  --audit-m2-width-space \
  "${oracle}"
"${build}/m2_flat_region_bridge" "${oracle}"
