#!/usr/bin/env bash
set -euo pipefail

# Reproducible integration gate for the DeepEdges disconnected-merge
# certificate. Invoke with bash so this file does not depend on executable-bit
# preservation:
#
#   bash benchmarks/cuda_spatial_replay/run_deep_edges_integration.sh \
#     [backend.so] [klayout-build-dir]

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${here}/../.." && pwd)

default_backend="${here}/build-integration/backend-cmake/libklayout_cuda_spatial_backend.so"
backend=${1:-${KLAYOUT_CUDA_SPATIAL_BACKEND:-"${default_backend}"}}
build_dir=${2:-${KLAYOUT_BUILD_DIR:-"${repo_root}/build-release"}}

if [[ ! -f "${backend}" ]]; then
  echo "CUDA spatial backend not found: ${backend}" >&2
  echo "Build it with benchmarks/cuda_spatial_replay/run.sh or pass its path." >&2
  exit 2
fi
if [[ ! -x "${build_dir}/ut_runner" ]]; then
  echo "KLayout unit-test runner not found: ${build_dir}/ut_runner" >&2
  exit 2
fi
if [[ ! -f "${build_dir}/db_tests.ut" ]]; then
  echo "KLayout DB test plugin not found: ${build_dir}/db_tests.ut" >&2
  exit 2
fi

backend_dir=$(cd -- "$(dirname -- "${backend}")" && pwd)
backend="${backend_dir}/$(basename -- "${backend}")"
build_dir=$(cd -- "${build_dir}" && pwd)

gate_tmp=$(mktemp -d "${TMPDIR:-/tmp}/klayout-deep-edges-gate.XXXXXX")
trap 'rm -rf -- "${gate_tmp}"' EXIT
log="${gate_tmp}/integration.log"

tests=(
  "dbDeepEdgesTests:24_LengthFilterDisconnectedHierarchy"
  "dbDeepEdgesTests:25_LengthFilterTouchingHierarchyFallsBack"
  "dbDeepEdgesTests:26_LengthFilterMergedSemanticsOff"
  "dbDeepEdgesTests:27_LengthFilterComplexTransformFallsBack"
  "dbDeepEdgesTests:28_LengthFilterCancellationInteractionFallsBack"
  "dbDeepEdgesTests:29_LengthFilterStoredEdgeLimitFallsBack"
  "dbDeepEdgesTests:30_LengthFilterLowReuseFallsBack"
  "dbDeepEdgesTests:31_LengthFilterDuplicateRectanglesSelectEmpty"
  "dbDeepEdgesTests:32_LengthFilterDuplicateRectanglesUnsafeModesFallBack"
  "dbDeepEdgesTests:33_LengthFilterEqualBoxNonRectangleFallsBack"
)

env \
  QT_QPA_PLATFORM=offscreen \
  LD_LIBRARY_PATH="${build_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  TESTSRC="${repo_root}" \
  TESTTMP="${gate_tmp}" \
  KLAYOUT_CUDA_DISCONNECTED_MERGE=1 \
  KLAYOUT_DEEP_EDGE_CERT_PROFILE=1 \
  KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}" \
  KLAYOUT_CUDA_SPATIAL_TELEMETRY=1 \
  KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=1 \
  KLAYOUT_CUDA_SPATIAL_CELL_SIZE=128 \
  KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64 \
  KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096 \
  KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=100000 \
  KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=100000 \
  KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=100000 \
  "${build_dir}/ut_runner" -ne "${tests[@]}" 2>&1 | tee "${log}"

for test in "${tests[@]}"; do
  if ! grep -Fq -- "${test}" "${log}"; then
    echo "integration gate did not select expected test: ${test}" >&2
    exit 1
  fi
done

required_telemetry=(
  "KLAYOUT_DEEP_EDGE_CERT status=success reason=disconnected"
  "KLAYOUT_DEEP_EDGE_CERT status=fallback reason=marker-interaction"
  "KLAYOUT_DEEP_EDGE_CERT status=fallback reason=complex-transform"
  "KLAYOUT_DEEP_EDGE_CERT status=fallback reason=stored-edge-limit"
  "KLAYOUT_DEEP_EDGE_CERT status=fallback reason=reuse-ratio"
  "KLAYOUT_DEEP_EDGE_CERT status=success reason=selected-empty"
)

for telemetry in "${required_telemetry[@]}"; do
  if ! grep -Fq -- "${telemetry}" "${log}"; then
    echo "integration gate missing telemetry: ${telemetry}" >&2
    exit 1
  fi
done

if ! grep -Eq -- \
  'KLAYOUT_DEEP_EDGE_CERT .*accelerator_ms=[0-9.]+ decision_ms=[0-9.]+' \
  "${log}"; then
  echo "integration gate missing decision-time telemetry" >&2
  exit 1
fi
if grep -Eq -- 'KLAYOUT_DEEP_EDGE_CERT .* total_ms=' "${log}"; then
  echo "integration gate observed obsolete total_ms certificate telemetry" >&2
  exit 1
fi

if [[ $(grep -Fc -- \
  "KLAYOUT_DEEP_EDGE_CERT status=fallback reason=marker-interaction" \
  "${log}") -lt 2 ]]; then
  echo "integration gate did not observe both touching and cancellation fallbacks" >&2
  exit 1
fi

echo "DeepEdges CUDA certificate integration gate passed."
