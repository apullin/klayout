#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nvcc=${NVCC:-/usr/bin/nvcc}
host_cxx=${CUDAHOSTCXX:-/usr/bin/g++-13}
architecture=${CUDA_ARCH:-sm_86}
build_dir=${BUILD_DIR:-/tmp/klayout-poly34-terminal-empty-island-build}

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CASES.poly34" >&2
  exit 2
fi

mkdir -p "${build_dir}/tmp"
TMPDIR="${build_dir}/tmp" "${nvcc}" \
  -O3 -std=c++17 -arch="${architecture}" -lineinfo \
  -ccbin "${host_cxx}" -Xcompiler=-Wall,-Wextra \
  "${script_dir}/poly34_terminal_empty_island.cu" \
  -o "${build_dir}/poly34_terminal_empty_island"

exec "${build_dir}/poly34_terminal_empty_island" "$1"
