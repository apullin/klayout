#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:-"${here}/build"}
nvcc=${NVCC:-/usr/bin/nvcc}
host_cxx=${CUDAHOSTCXX:-/usr/bin/g++-13}

mkdir -p "${build_dir}/tmp"

cmake -S "${here}" -B "${build_dir}/backend-cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_HOST_COMPILER="${host_cxx}" \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "${build_dir}/backend-cmake" \
  --target cuda_spatial_backend_smoke -j "${BUILD_JOBS:-16}"

echo "== runtime backend ABI gate =="
"${build_dir}/backend-cmake/cuda_spatial_backend_smoke"

env TMPDIR="${build_dir}/tmp" \
  "${nvcc}" -O3 -std=c++17 -arch=sm_86 -ccbin "${host_cxx}" \
  "${here}/spatial_replay.cu" -o "${build_dir}/cuda_spatial_replay"

echo "== exhaustive correctness gate =="
"${build_dir}/cuda_spatial_replay" \
  --records 8192 --contexts 8 --world-size 5000 --object-size 64 \
  --geometry edges --mode self --enlargement 32 --cell-size 128 \
  --reference exhaustive --max-memberships 1000000 \
  --max-pair-work 2000000 --max-candidates 2000000

echo "== strict boundary gates (self and bipartite) =="
for mode in self bipartite; do
  for enlargement in 0 1; do
    "${build_dir}/cuda_spatial_replay" \
      --fixture boundary --mode "${mode}" --enlargement "${enlargement}" \
      --cell-size 8 --reference exhaustive --max-memberships 1000 \
      --max-pair-work 1000 --max-candidates 1000
  done
done

echo "== signed-coordinate extrema and replay round trip =="
"${build_dir}/cuda_spatial_replay" \
  --fixture extrema --mode bipartite --enlargement 1 --cell-size 8 \
  --reference exhaustive --write-input "${build_dir}/extrema.replay" \
  --max-memberships 1000 --max-pair-work 1000 --max-candidates 1000
"${build_dir}/cuda_spatial_replay" \
  --input "${build_dir}/extrema.replay" --mode bipartite --enlargement 1 \
  --cell-size 8 --reference exhaustive --max-memberships 1000 \
  --max-pair-work 1000 --max-candidates 1000

echo "== aggregated throughput sample =="
"${build_dir}/cuda_spatial_replay" \
  --records 1000000 --contexts 4 --world-size 20000 --object-size 64 \
  --geometry aabb --mode bipartite --enlargement 32 --cell-size 128 \
  --reference grid --warmup 1 --repeat 3 --max-memberships 16000000 \
  --max-pair-work 64000000 --max-candidates 16000000

echo "== expected fail-closed dense-cell gate =="
set +e
"${build_dir}/cuda_spatial_replay" \
  --records 2048 --contexts 1 --world-size 32 --object-size 16 \
  --mode bipartite --enlargement 32 --cell-size 128 --reference exhaustive \
  --max-edges-per-cell 64 --max-memberships 1000000 \
  --max-pair-work 4000000 --max-candidates 4000000
dense_status=$?
set -e
if [[ ${dense_status} -ne 3 ]]; then
  echo "dense-cell gate returned ${dense_status}, expected explicit fallback status 3" >&2
  exit 1
fi

echo "== expected fail-closed int64 overflow gate =="
set +e
"${build_dir}/cuda_spatial_replay" \
  --fixture overflow --mode bipartite --enlargement 1 --cell-size 8 \
  --reference exhaustive --max-memberships 1000 --max-pair-work 1000 \
  --max-candidates 1000
overflow_status=$?
set -e
if [[ ${overflow_status} -ne 3 ]]; then
  echo "coordinate-overflow gate returned ${overflow_status}, expected 3" >&2
  exit 1
fi

echo "== expected fail-closed default-grid overflow gate =="
set +e
"${build_dir}/cuda_spatial_replay" \
  --fixture overflow --mode bipartite --enlargement 1 --cell-size 8 \
  --max-memberships 1000 --max-pair-work 1000 --max-candidates 1000
overflow_grid_status=$?
set -e
if [[ ${overflow_grid_status} -ne 3 ]]; then
  echo "default-grid overflow gate returned ${overflow_grid_status}, expected 3" >&2
  exit 1
fi
