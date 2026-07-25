#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

backend_build="${KLAYOUT_CUDA_BACKEND_BUILD:-/tmp/klayout-contact4-union-resident-build}"
qmake_build="${KLAYOUT_QMAKE_BUILD_DIR:-/tmp/klayout-m1ws-live-build}"
gate_build="${KLAYOUT_CONTACT4_REAL_VALIDATOR_BUILD:-/tmp/klayout-contact4-real-validator-gate}"

backend="${backend_build}/libklayout_cuda_spatial_backend.so"
db_library="${qmake_build}/libklayout_db.so"
tl_library="${qmake_build}/libklayout_tl.so"
gsi_library="${qmake_build}/libklayout_gsi.so"

for required in \
  "${backend}" \
  "${db_library}" \
  "${tl_library}" \
  "${gsi_library}"; do
  if [[ ! -r "${required}" ]]; then
    echo "missing required library: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${gate_build}"

"${CXX:-c++}" \
  -std=c++17 -O2 -Wall -Wextra \
  -I"${repo_root}/src/db/db" \
  -I"${repo_root}/src/tl/tl" \
  -I"${repo_root}/src/gsi/gsi" \
  "${script_dir}/contact4_active_union_real_validator_gate.cc" \
  "${repo_root}/src/db/db/dbCudaSpatialBackend.cc" \
  "${backend}" \
  "${db_library}" \
  "${tl_library}" \
  "${gsi_library}" \
  -pthread -ldl \
  -Wl,-rpath,"${backend_build}:${qmake_build}" \
  -o "${gate_build}/contact4_active_union_real_validator_gate"

"${gate_build}/contact4_active_union_real_validator_gate"
