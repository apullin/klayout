#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_m2_via2_certificate_gate.sh \
    --klayout PATH [--python PATH] [--build-dir PATH] [--keep-work]

Builds deterministic M2/VIA2 fixtures, compares the exact CUDA union-coverage
certificate with both a bounded CPU differential and KLayout's stock
METAL2.4 chain, then exercises digest, census, topology, and capacity fallback.
EOF
}

die() {
  echo "M2/VIA2 certificate gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/m2_via2_certificate_fixture.rb"
oracle_deck="${here}/m2_via2_certificate_oracle.lydrc"
exporter="${here}/active3_packed_scene_export.rb"
validator="${here}/validate_active3_packed_scene.py"
runner="${here}/run_m2_via2_certificate.sh"

klayout=
python=${PYTHON:-python3}
requested_build_dir=
keep_work=0

while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
      shift 2
      ;;
    --build-dir)
      (($# >= 2)) || die "--build-dir requires a value"
      requested_build_dir=$2
      shift 2
      ;;
    --keep-work)
      keep_work=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${klayout}" ]] || die "missing --klayout"
[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${fixture_script}" ]] || die "fixture generator is missing"
[[ -f "${oracle_deck}" ]] || die "CPU oracle deck is missing"
[[ -f "${exporter}" ]] || die "KACTSCN1 exporter is missing"
[[ -f "${validator}" ]] || die "KACTSCN1 validator is missing"
[[ -x "${runner}" ]] || die "certificate runner is not executable"
python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
klayout=$(readlink -f -- "${klayout}")
klayout_dir=$(dirname -- "${klayout}")

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-m2-via2-gate.XXXXXX")
if [[ -n "${requested_build_dir}" ]]; then
  mkdir -p -- "${requested_build_dir}"
  build_dir=$(readlink -f -- "${requested_build_dir}")
else
  build_dir="${work}/build"
fi

cleanup() {
  if ((keep_work)); then
    echo "M2_VIA2_CERTIFICATE_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p \
  "${work}/home" "${work}/config" "${work}/cache" "${build_dir}"
common_env=(
  QT_QPA_PLATFORM=offscreen
  HOME="${work}/home"
  XDG_CONFIG_HOME="${work}/config"
  XDG_CACHE_HOME="${work}/cache"
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)

fixture="${work}/m2-via2-certificate.gds"
env "${common_env[@]}" "${klayout}" -b \
  -r "${fixture_script}" -rd "output=${fixture}" \
  >"${work}/fixture.log" 2>&1
cat "${work}/fixture.log"
grep -Fq "M2_VIA2_CERTIFICATE_FIXTURE ok" "${work}/fixture.log" ||
  die "fixture completion marker is missing"
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"

declare -A scene_path
declare -A scene_digest

export_case() {
  local key=$1
  local top=$2
  local output="${work}/${key}.kact"
  local export_log="${work}/${key}.export.log"
  local validate_log="${work}/${key}.validate.log"
  local digest

  env "${common_env[@]}" "${klayout}" -b \
    -r "${exporter}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${output}" \
    -rd "well_layer=101" \
    -rd "well_datatype=0" \
    -rd "active_layer=102" \
    -rd "active_datatype=0" \
    >"${export_log}" 2>&1
  cat "${export_log}"
  [[ -s "${output}" ]] || die "${key}: exporter produced no scene"
  "${python}" "${validator}" "${output}" >"${validate_log}" 2>&1
  cat "${validate_log}"
  grep -Fq "KACTSCN1 ok" "${validate_log}" ||
    die "${key}: independent packed-scene validation failed"
  digest=$(
    sed -nE \
      's/^KACTSCN1 output=.* scene_sha256=([0-9a-f]{64})$/\1/p' \
      "${export_log}"
  )
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] ||
    die "${key}: missing or ambiguous scene digest"
  scene_path["${key}"]=${output}
  scene_digest["${key}"]=${digest}
}

oracle_case() {
  local key=$1
  local top=$2
  local expectation=$3
  local report="${work}/${key}.oracle.lyrdb"
  local log="${work}/${key}.oracle.log"

  env "${common_env[@]}" "${klayout}" -b \
    -r "${oracle_deck}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    >"${log}" 2>&1
  cat "${log}"
  grep -Fq "M2_VIA2_CPU_ORACLE" "${log}" ||
    die "${key}: CPU oracle completion marker is missing"
  if [[ "${expectation}" == clean ]]; then
    grep -Eq \
      'M2_VIA2_CPU_ORACLE .*corners_flat=0 corners_hier=0 markers_flat=0 markers_hier=0' \
      "${log}" ||
      die "${key}: stock CPU METAL2.4 did not report clean"
  else
    grep -Eq 'M2_VIA2_CPU_ORACLE .*markers_flat=[1-9][0-9]*' "${log}" ||
      die "${key}: stock CPU METAL2.4 did not report a marker"
  fi
}

run_case() {
  local key=$1
  local expected_rc=$2
  shift 2
  local log="${work}/${key}.gpu.log"
  local rc

  set +e
  "${build_dir}/m2_via2_certificate" \
    "--expect-scene-sha256=${scene_digest[${key}]}" \
    --verify-cpu \
    "${scene_path[${key}]}" >"${log}" 2>&1
  rc=$?
  set -e
  cat "${log}"
  [[ "${rc}" == "${expected_rc}" ]] ||
    die "${key}: expected exit ${expected_rc}, got ${rc}"
  for expected in "$@"; do
    grep -Fq "${expected}" "${log}" ||
      die "${key}: missing output: ${expected}"
  done
  echo "M2_VIA2_CERTIFICATE_GATE ok gate=${key} rc=${rc}"
}

cases=(
  "x69 M2VIA2_X_69 violation"
  "x70 M2VIA2_X_70 clean"
  "x71 M2VIA2_X_71 clean"
  "y69 M2VIA2_Y_69 violation"
  "y70 M2VIA2_Y_70 clean"
  "y71 M2VIA2_Y_71 clean"
  "partial M2VIA2_PARTIAL_ADJACENT violation"
  "y_choice M2VIA2_Y_CHOICE clean"
  "tiled M2VIA2_UNION_TILED_X clean"
  "tiled_gap M2VIA2_UNION_TILED_X_GAP clean"
  "corner_only M2VIA2_CORNER_ONLY violation"
  "hierarchy M2VIA2_HIERARCHY_8 clean"
  "nonrect M2VIA2_NONRECT_VIA unsupported"
)

for row in "${cases[@]}"; do
  read -r key top expectation <<<"${row}"
  export_case "${key}" "${top}"
  if [[ "${expectation}" != unsupported ]]; then
    oracle_case "${key}" "${top}" "${expectation}"
  fi
done

# Deterministic repeated export and explicit scene digest.
export_case x70_repeat M2VIA2_X_70
cmp -s "${scene_path[x70]}" "${scene_path[x70_repeat]}" ||
  die "repeat X=70 exports are not byte-identical"
[[ "${scene_digest[x70]}" == "${scene_digest[x70_repeat]}" ]] ||
  die "repeat X=70 scene digests differ"

# Compile once. Every subsequent result is from this exact binary.
BUILD_DIR="${build_dir}" "${runner}" \
  "--expect-scene-sha256=${scene_digest[x70]}" \
  --verify-cpu "${scene_path[x70]}" \
  >"${work}/x70.build.log" 2>&1
cat "${work}/x70.build.log"
[[ -x "${build_dir}/m2_via2_certificate" ]] ||
  die "runner did not produce the certificate binary"
grep -Fq "verdict=CLEAN" "${work}/x70.build.log" ||
  die "first exact-boundary run did not certify clean"

run_case x69 3 "verdict=FALLBACK" " misses=1 "
run_case x70 0 "verdict=CLEAN" " x_certified=1 y_certified=0 misses=0 "
run_case x71 0 "verdict=CLEAN" " x_certified=1 y_certified=0 misses=0 "
run_case y69 3 "verdict=FALLBACK" " misses=1 "
run_case y70 0 "verdict=CLEAN" " x_certified=0 y_certified=1 misses=0 "
run_case y71 0 "verdict=CLEAN" " x_certified=0 y_certified=1 misses=0 "
run_case partial 3 "verdict=FALLBACK" " misses=1 "
run_case y_choice 0 "verdict=CLEAN" " x_certified=0 y_certified=1 misses=0 "
run_case tiled 0 \
  "verdict=CLEAN" \
  " metal_boxes=3 vias=1 " \
  " x_certified=1 y_certified=0 misses=0 "
run_case tiled_gap 3 "verdict=FALLBACK" " misses=1 "
run_case corner_only 3 "verdict=FALLBACK" " misses=1 "
run_case hierarchy 0 \
  "verdict=CLEAN" \
  " contexts=9 metal_contexts=8 via_contexts=8 " \
  " metal_boxes=24 vias=8 " \
  " x_certified=4 y_certified=4 misses=0 "

nonrect_log="${work}/nonrect.gpu.log"
set +e
"${build_dir}/m2_via2_certificate" \
  "--expect-scene-sha256=${scene_digest[nonrect]}" \
  "${scene_path[nonrect]}" >"${nonrect_log}" 2>&1
nonrect_rc=$?
set -e
cat "${nonrect_log}"
[[ "${nonrect_rc}" == 2 ]] || die "nonrect VIA2 did not fail closed"
grep -Fq "verdict=UNCERTAIN" "${nonrect_log}" ||
  die "nonrect VIA2 omitted UNCERTAIN verdict"
grep -Fq "cut layer contains a non-rectangle" "${nonrect_log}" ||
  die "nonrect VIA2 omitted its reason"

# The tiled clean case needs three distinct raw M2 rectangles. A bounded
# candidate budget must decline rather than silently use a partial union.
capacity_log="${work}/capacity.gpu.log"
set +e
"${build_dir}/m2_via2_certificate" \
  "--expect-scene-sha256=${scene_digest[tiled]}" \
  --max-query-candidates=2 \
  "${scene_path[tiled]}" >"${capacity_log}" 2>&1
capacity_rc=$?
set -e
cat "${capacity_log}"
[[ "${capacity_rc}" == 2 ]] || die "candidate capacity did not fail closed"
grep -Fq "verdict=UNCERTAIN" "${capacity_log}" ||
  die "candidate capacity omitted UNCERTAIN verdict"
grep -Fq "device_flags=16" "${capacity_log}" ||
  die "candidate capacity omitted its device flag"

# Census expectations and explicit digest are both fail-closed guards.
census_log="${work}/census.gpu.log"
set +e
"${build_dir}/m2_via2_certificate" \
  "--expect-scene-sha256=${scene_digest[x70]}" \
  --expect-vias=2 \
  "${scene_path[x70]}" >"${census_log}" 2>&1
census_rc=$?
set -e
cat "${census_log}"
[[ "${census_rc}" == 2 ]] || die "wrong VIA2 census did not fail closed"
grep -Fq "logical VIA2 census differs from expectation" "${census_log}" ||
  die "wrong VIA2 census omitted its reason"

digest_log="${work}/digest.gpu.log"
set +e
"${build_dir}/m2_via2_certificate" \
  --expect-scene-sha256=0000000000000000000000000000000000000000000000000000000000000000 \
  "${scene_path[x70]}" >"${digest_log}" 2>&1
digest_rc=$?
set -e
cat "${digest_log}"
[[ "${digest_rc}" == 2 ]] || die "wrong scene digest did not fail closed"
grep -Fq "scene SHA-256 does not match explicit expectation" "${digest_log}" ||
  die "wrong scene digest omitted its reason"

corrupt="${work}/x70-corrupt.kact"
"${python}" -c \
  'import pathlib,sys; d=bytearray(pathlib.Path(sys.argv[1]).read_bytes()); d[256]^=1; pathlib.Path(sys.argv[2]).write_bytes(d)' \
  "${scene_path[x70]}" "${corrupt}"
corrupt_log="${work}/corrupt.gpu.log"
set +e
"${build_dir}/m2_via2_certificate" \
  "--expect-scene-sha256=${scene_digest[x70]}" \
  "${corrupt}" >"${corrupt_log}" 2>&1
corrupt_rc=$?
set -e
cat "${corrupt_log}"
[[ "${corrupt_rc}" == 2 ]] || die "corrupt payload did not fail closed"
grep -Fq "scene SHA-256 mismatch" "${corrupt_log}" ||
  die "corrupt payload omitted its integrity reason"

sha256sum \
  "${build_dir}/m2_via2_certificate" \
  "${fixture}" \
  "${scene_path[x70]}" \
  >"${work}/pinned-inputs.sha256"
cat "${work}/pinned-inputs.sha256"
echo \
  "M2_VIA2_CERTIFICATE_GATE passed clean=7 fallback=5 fail_closed=5 deterministic=1"
