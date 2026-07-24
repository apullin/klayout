#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nvcc=${NVCC:-/usr/bin/nvcc}
host_cxx=${CUDAHOSTCXX:-/usr/bin/g++-13}
architecture=${CUDA_ARCH:-sm_86}
build_dir=${BUILD_DIR:-/tmp/klayout-m1-width-space-island-build}
benchmark_contexts=${M1_WIDTH_SPACE_BENCHMARK_CONTEXTS:-65536}
repetitions=${M1_WIDTH_SPACE_REPETITIONS:-3}
host_scene=${M1_WIDTH_SPACE_HOST_SCENE:-}
qualified_host_scene_sha256=df713200c1271e510ac2ecd1bdc060054e69451658f0c4327d64b228f8925235

mkdir -p "${build_dir}/tmp"
TMPDIR="${build_dir}/tmp" "${nvcc}" \
  -O3 -std=c++17 -arch="${architecture}" -lineinfo \
  -ccbin "${host_cxx}" -Xcompiler=-Wall,-Wextra \
  "${script_dir}/m1_width_space_scene_island.cu" \
  -o "${build_dir}/m1_width_space_scene_island"

if [[ $# -gt 0 ]]; then
  exec "${build_dir}/m1_width_space_scene_island" "$@"
fi

if [[ -n "${host_scene}" ]]; then
  exec "${build_dir}/m1_width_space_scene_island" \
    --host-scene="${host_scene}" \
    --expect-scene-sha256="${qualified_host_scene_sha256}"
fi

"${build_dir}/m1_width_space_scene_island" --self-test
exec "${build_dir}/m1_width_space_scene_island" \
  --benchmark-contexts="${benchmark_contexts}" \
  --repetitions="${repetitions}" \
  --no-host-oracle
