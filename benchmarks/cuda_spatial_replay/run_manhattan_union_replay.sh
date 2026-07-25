#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${1:-${here}/build-manhattan-union}"
grid="${KLAYOUT_CUDA_MANHATTAN_UNION_GRID:-1024}"
repeat="${KLAYOUT_CUDA_MANHATTAN_UNION_REPEAT:-5}"
production_kact="${KLAYOUT_CUDA_MANHATTAN_UNION_PRODUCTION_KACT:-}"
production_oracle="${KLAYOUT_CUDA_MANHATTAN_UNION_PRODUCTION_ORACLE:-}"

if [[ -n "${production_kact}" && -z "${production_oracle}" ]] ||
   [[ -z "${production_kact}" && -n "${production_oracle}" ]]; then
  echo "set both production KACT and oracle paths, or neither" >&2
  exit 2
fi

mkdir -p "${build}/tmp"
env TMPDIR="${build}/tmp" cmake \
  -S "${here}" -B "${build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=86
env TMPDIR="${build}/tmp" cmake --build "${build}" \
  --target manhattan_union_replay

"${build}/manhattan_union_replay" --self-test
"${build}/manhattan_union_replay" \
  --benchmark-grid "${grid}" --repeat "${repeat}"

if [[ -n "${production_kact}" ]]; then
  "${build}/manhattan_union_replay" \
    --production-m2-kact "${production_kact}" \
    --production-m2-oracle "${production_oracle}" \
    --repeat "${repeat}"
fi
