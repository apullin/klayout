#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${1:-${here}/build-m2-gpu-flat-pipeline}"
qmake_build="${KLAYOUT_QMAKE_BUILD_DIR:-}"
kact="${KLAYOUT_M2_GPU_FLAT_KACT:-}"
oracle="${KLAYOUT_M2_GPU_FLAT_ORACLE:-}"
union_repeat="${KLAYOUT_M2_GPU_FLAT_UNION_REPEAT:-4}"
bridge_repeat="${KLAYOUT_M2_GPU_FLAT_BRIDGE_REPEAT:-3}"
evidence_root="${KLAYOUT_M2_GPU_FLAT_EVIDENCE_ROOT:-${build}/evidence}"

if [[ -z "${qmake_build}" ]]; then
  echo "KLAYOUT_QMAKE_BUILD_DIR must name a built KLayout qmake tree" >&2
  exit 2
fi
if [[ -z "${kact}" || -z "${oracle}" ]]; then
  echo "KLAYOUT_M2_GPU_FLAT_KACT and KLAYOUT_M2_GPU_FLAT_ORACLE are required" >&2
  exit 2
fi
if [[ ! "${union_repeat}" =~ ^[1-9][0-9]*$ ||
      ! "${bridge_repeat}" =~ ^[1-9][0-9]*$ ]]; then
  echo "repeat counts must be positive integers" >&2
  exit 2
fi

mkdir -p "${build}/tmp" "${evidence_root}"
evidence="$(mktemp -d "${evidence_root}/m2-gpu-flat-pipeline.XXXXXX")"
candidate="${evidence}/gpu-boundary.km2bnd"

env TMPDIR="${build}/tmp" cmake \
  -S "${here}" -B "${build}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DKLAYOUT_QMAKE_BUILD_DIR="${qmake_build}"
env TMPDIR="${build}/tmp" cmake --build "${build}" --parallel 32 \
  --target manhattan_union_replay manhattan_union_gpu_core_smoke \
           m2_flat_region_bridge \
           m2_merged_boundary_oracle_cli \
           m2_merged_boundary_oracle_test

export LD_LIBRARY_PATH="${qmake_build}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${build}/manhattan_union_gpu_core_smoke" |
  tee "${evidence}/union-core-smoke.log"
"${build}/m2_merged_boundary_oracle_test" |
  tee "${evidence}/stream-self-test.log"
"${build}/m2_flat_region_bridge" --self-test |
  tee "${evidence}/bridge-self-test.log"

# Live numerator, producer half: the CPU oracle is deliberately absent.
"${build}/manhattan_union_replay" \
  --production-m2-kact "${kact}" \
  --production-m2-candidate-out "${candidate}" \
  --repeat "${union_repeat}" |
  tee "${evidence}/union.log"

# Qualification-only exact comparison.  Its ~6-second CPU-oracle load is
# recorded but never included in the live numerator below.
"${build}/m2_merged_boundary_oracle_cli" \
  --expect-file-sha256=980d439ba40535117505dc4e6d31d866af2f897e29fc46b041cebe9a55de7d0f \
  --expect-scene-sha256=441475a90d0471b886d5f09622d083b29aaa92f9cf47f31f4b7715792cf14480 \
  --expect-boundary-sha256=94b715fc2f9e2ab53f0af0f3dda5a579e9fa4b55b98fc2d04a1a0d9732ad820d \
  --candidate-producer-scene-sha256=dd239a45408a046eece0ca1e4c8759ea4b8539e6b7a51599c2ac9a2996a86bd2 \
  "--candidate=${candidate}" \
  "${oracle}" |
  tee "${evidence}/oracle-qualification.log"

# One excluded full-boundary qualification pins the 4.254-million-edge F90
# census.  The measured live form below uses the exact fused length filter.
"${build}/m2_flat_region_bridge" \
  --audit-full-boundary --audit-m2-width-space \
  --candidate "${candidate}" |
  tee "${evidence}/flat-full-qualification.log"

for ((run = 1; run <= bridge_repeat; ++run)); do
  "${build}/m2_flat_region_bridge" \
    --audit-m2-width-space --candidate "${candidate}" |
    tee "${evidence}/flat-live-${run}.log"
done

# Actual production corruption probes.  Both must be rejected while reading,
# before endpoint stitching or any KLayout Region construction.
cp "${candidate}" "${evidence}/payload-corrupt.km2bnd"
printf '\377' |
  dd of="${evidence}/payload-corrupt.km2bnd" \
     bs=1 seek=160 count=1 conv=notrunc status=none
if "${build}/m2_flat_region_bridge" \
     --candidate "${evidence}/payload-corrupt.km2bnd" \
     >"${evidence}/payload-corrupt.log" 2>&1; then
  echo "corrupted KM2BND payload was accepted" >&2
  exit 3
fi

cp "${candidate}" "${evidence}/truncated.km2bnd"
truncate -s -1 "${evidence}/truncated.km2bnd"
if "${build}/m2_flat_region_bridge" \
     --candidate "${evidence}/truncated.km2bnd" \
     >"${evidence}/truncated.log" 2>&1; then
  echo "truncated KM2BND stream was accepted" >&2
  exit 3
fi

awk '
  /^M2_MANHATTAN_PRODUCTION_UNION PASS/ {
    for (i = 1; i <= NF; ++i) {
      split($i, pair, "=")
      if (pair[1] == "published_candidate_pipeline_ms") print pair[2]
    }
  }
' "${evidence}/union.log" >"${evidence}/union-live-ms.txt"

awk '
  /^TIMING/ {
    for (i = 1; i <= NF; ++i) {
      split($i, pair, "=")
      if (pair[1] == "total_ms") print pair[2]
    }
  }
' "${evidence}"/flat-live-*.log |
  sort -n >"${evidence}/flat-live-ms.txt"

union_ms="$(tail -1 "${evidence}/union-live-ms.txt")"
bridge_median_ms="$(
  awk '
    { value[NR] = $1 }
    END {
      if (!NR) exit 1
      if (NR % 2) {
        printf "%.3f", value[(NR + 1) / 2]
      } else {
        printf "%.3f", (value[NR / 2] + value[NR / 2 + 1]) / 2
      }
    }
  ' "${evidence}/flat-live-ms.txt"
)"
pipeline_ms="$(
  awk -v union_ms="${union_ms}" -v bridge_ms="${bridge_median_ms}" \
    'BEGIN { printf "%.3f", union_ms + bridge_ms }'
)"

sha256sum \
  "${build}/manhattan_union_replay" \
  "${build}/m2_flat_region_bridge" \
  "${candidate}" "${kact}" "${oracle}" \
  >"${evidence}/pinned-artifacts.sha256"

echo "M2_GPU_FLAT_PIPELINE PASS" \
  "producer_live_ms=${union_ms}" \
  "flat_live_median_ms=${bridge_median_ms}" \
  "charged_end_to_end_ms=${pipeline_ms}" \
  "oracle_qualification_excluded=1" \
  "full_boundary_qualification_excluded=1" \
  "serialization_and_read_charged=1" \
  "evidence=${evidence}"
