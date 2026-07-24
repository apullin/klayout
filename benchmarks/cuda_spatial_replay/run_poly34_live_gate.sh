#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_poly34_live_gate.sh \
    --klayout PATH --backend PATH [--keep-work]

Runs only the focused live integrity gate for the opt-in atomic POLY.3/.4
terminal-empty certificate.  It compares complete canonical CPU and CUDA
reports for clean, hierarchical-clean, independent POLY.3/POLY.4 positive
    hits and a mixed two-rule fallback.  Feature-disabled, missing-backend, and
    capacity lanes must also preserve the CPU report without unsafe partial
    consumption.  This script is not a production/full-design benchmark.
EOF
}

die() {
  echo "POLY.3/.4 live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/poly34_live_fixture.rb"
deck="${here}/poly34_live_gate.drc"
klayout=
backend=
keep_work=0

while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
      shift 2
      ;;
    --backend)
      (($# >= 2)) || die "--backend requires a value"
      backend=$2
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
[[ -n "${backend}" ]] || die "missing --backend"
[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${backend}" ]] || die "CUDA backend is missing: ${backend}"
[[ -f "${fixture_script}" ]] || die "fixture generator is missing"
[[ -f "${deck}" ]] || die "gate deck is missing"

klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
klayout_dir=$(dirname -- "${klayout}")
work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-poly34-live-gate.XXXXXX")

cleanup() {
  if ((keep_work)); then
    echo "POLY34_LIVE_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- "${work}/reports" "${work}/logs" "${work}/runtime"
missing_backend="${work}/missing-cuda-spatial-backend.so"

run_klayout() {
  local mode=$1
  local lane=$2
  shift 2
  local runtime="${work}/runtime/${lane}"
  mkdir -p -- \
    "${runtime}/home" "${runtime}/config" \
    "${runtime}/cache" "${runtime}/data" "${runtime}/tmp"

  local -a command=(
    env
    -u KLAYOUT_CUDA_SPATIAL_BACKEND
    -u KLAYOUT_CUDA_SPATIAL_DEVICE
    -u KLAYOUT_CUDA_SPATIAL_TELEMETRY
    -u KLAYOUT_CUDA_POLY34
    -u KLAYOUT_CUDA_POLY34_TELEMETRY
    -u KLAYOUT_CUDA_POLY34_MAX_CONTEXTS
    -u KLAYOUT_CUDA_POLY34_MAX_FLAT_BOXES
    -u KLAYOUT_CUDA_POLY34_MAX_GRID_CELLS
    -u KLAYOUT_CUDA_POLY34_MAX_POLY_MEMBERSHIPS
    -u KLAYOUT_CUDA_POLY34_MAX_ACTIVE_MEMBERSHIPS
    -u KLAYOUT_CUDA_POLY34_MAX_QUERY_VISITS
    -u KLAYOUT_CUDA_POLY34_MAX_CANDIDATE_WORK
    QT_QPA_PLATFORM=offscreen
    HOME="${runtime}/home"
    XDG_CONFIG_HOME="${runtime}/config"
    XDG_CACHE_HOME="${runtime}/cache"
    XDG_DATA_HOME="${runtime}/data"
    TMPDIR="${runtime}/tmp"
    LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  )
  case "${mode}" in
    off)
      ;;
    cuda)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_POLY34=1
        KLAYOUT_CUDA_POLY34_TELEMETRY=1
      )
      ;;
    backend-off)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_POLY34_TELEMETRY=1
      )
      ;;
    missing)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${missing_backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_POLY34=1
        KLAYOUT_CUDA_POLY34_TELEMETRY=1
      )
      ;;
    capacity)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_POLY34=1
        KLAYOUT_CUDA_POLY34_TELEMETRY=1
        KLAYOUT_CUDA_POLY34_MAX_GRID_CELLS=1
      )
      ;;
    *)
      die "internal error: unknown mode ${mode}"
      ;;
  esac
  command+=("${klayout}" "$@")
  "${command[@]}"
}

fixture="${work}/poly34-live-fixture.gds"
fixture_log="${work}/logs/fixture.log"
if ! run_klayout off fixture -b -r "${fixture_script}" \
  -rd "output=${fixture}" >"${fixture_log}" 2>&1; then
  cat -- "${fixture_log}" >&2
  die "fixture generation failed"
fi
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
  grep -Fq -- \
  "POLY34_LIVE_FIXTURE ok path=${fixture} tops=7 dbu=0.0005 hierarchy_gate_occurrences=2048 transformed_hit_occurrences=6" \
  "${fixture_log}" ||
  die "fixture completion marker or hierarchy count is wrong"

canonicalize_report() {
  local report=$1
  local output=$2
  sed '/<generator>/d' "${report}" >"${output}"
}

item_count() {
  local report=$1
  local category=$2
  grep -Fc -- "<category>'${category}'</category>" "${report}" || true
}

assert_report() {
  local report=$1
  local top=$2
  local expected=$3
  local poly3_count
  local poly4_count
  grep -Fq -- "<top-cell>${top}</top-cell>" "${report}" ||
    die "${top}: report top-cell marker is missing"
  [[ "$(grep -Fc -- '<name>POLY.3</name>' "${report}" || true)" == 1 ]] ||
    die "${top}: report does not contain exactly one POLY.3 category"
  [[ "$(grep -Fc -- '<name>POLY.4</name>' "${report}" || true)" == 1 ]] ||
    die "${top}: report does not contain exactly one POLY.4 category"
  poly3_count=$(item_count "${report}" POLY.3)
  poly4_count=$(item_count "${report}" POLY.4)
  case "${expected}" in
    clean)
      ((poly3_count == 0 && poly4_count == 0)) ||
        die "${top}: expected no markers, found POLY.3=${poly3_count} POLY.4=${poly4_count}"
      ;;
    poly3)
      ((poly3_count > 0 && poly4_count == 0)) ||
        die "${top}: expected POLY.3-only markers, found POLY.3=${poly3_count} POLY.4=${poly4_count}"
      ;;
    poly4)
      ((poly3_count == 0 && poly4_count > 0)) ||
        die "${top}: expected POLY.4-only markers, found POLY.3=${poly3_count} POLY.4=${poly4_count}"
      ;;
    both)
      ((poly3_count > 0 && poly4_count > 0)) ||
        die "${top}: expected both marker classes, found POLY.3=${poly3_count} POLY.4=${poly4_count}"
      ;;
    *)
      die "internal error: unknown report expectation ${expected}"
      ;;
  esac
}

run_case() {
  local lane=$1
  local mode=$2
  local top=$3
  local expected=$4
  shift 4
  local lane_dir="${work}/reports/${lane}"
  local report="${lane_dir}/${top}.lyrdb"
  local log="${work}/logs/${lane}-${top}.log"
  mkdir -p -- "${lane_dir}"
  if ! run_klayout "${mode}" "${lane}-${top}" -b -r "${deck}" \
    -rd "input=${fixture}" -rd "topcell=${top}" \
    -rd "output=${report}" "$@" >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}/${top}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] || die "${lane}/${top}: DRC produced no report"
  grep -Fq -- "POLY34_LIVE_DECK ok top=${top}" "${log}" ||
    die "${lane}/${top}: deck completion marker is missing"
  assert_report "${report}" "${top}" "${expected}"
}

compare_reports() {
  local reference_lane=$1
  local candidate_lane=$2
  local top=$3
  local reference="${work}/reports/${reference_lane}/${top}.lyrdb"
  local candidate="${work}/reports/${candidate_lane}/${top}.lyrdb"
  local reference_canonical="${reference}.canonical"
  local candidate_canonical="${candidate}.canonical"
  canonicalize_report "${reference}" "${reference_canonical}"
  canonicalize_report "${candidate}" "${candidate_canonical}"
  cmp -s -- "${reference_canonical}" "${candidate_canonical}" ||
    die "${candidate_lane}/${top}: report differs from ${reference_lane}"
}

attempt_count() {
  grep -Fc -- \
    "CUDA POLY.3/.4 terminal-empty certificate:" "$1" || true
}

lowering_count() {
  grep -Fc -- "CUDA POLY.3/.4 live lowering:" "$1" || true
}

assert_one_attempt() {
  local log=$1
  local label=$2
  [[ "$(attempt_count "${log}")" == 1 ]] ||
    die "${label}: expected exactly one backend attempt"
  [[ "$(lowering_count "${log}")" == 1 ]] ||
    die "${label}: expected exactly one live lowering"
}

cases=(
  "POLY34_CLEAN:clean:cert"
  "POLY34_HIER_CLEAN:clean:cert"
  "POLY34_HIER_TRANSFORM_HIT:poly3:hit"
  "POLY34_MANHATTAN_PRIMARY_CLEAN:clean:cert"
  "POLY34_POLY3_HIT:poly3:hit"
  "POLY34_POLY4_HIT:poly4:hit"
  "POLY34_MIXED:both:hit"
)

for spec in "${cases[@]}"; do
  IFS=: read -r top expected disposition <<<"${spec}"
  run_case oracle off "${top}" "${expected}"
  oracle_log="${work}/logs/oracle-${top}.log"
  [[ "$(attempt_count "${oracle_log}")" == 0 ]] ||
    die "oracle/${top}: opt-in-off path invoked the backend"
  [[ "$(lowering_count "${oracle_log}")" == 0 ]] ||
    die "oracle/${top}: opt-in-off path lowered a scene"
done
echo "POLY34_LIVE_GATE ok gate=cpu-oracle cases=7 categories=2"

for spec in "${cases[@]}"; do
  IFS=: read -r top expected disposition <<<"${spec}"
  run_case accelerated cuda "${top}" "${expected}"
  compare_reports oracle accelerated "${top}"
  log="${work}/logs/accelerated-${top}.log"
  assert_one_attempt "${log}" "accelerated/${top}"
  if [[ "${disposition}" == cert ]]; then
    grep -Fq -- \
      "CUDA POLY.3/.4 terminal-empty certificate: outcome=certified-empty" \
      "${log}" ||
      die "accelerated/${top}: expected certified-empty"
    grep -Fq -- "POLY34_LIVE_DECK ok top=${top} accelerated=1" "${log}" ||
      die "accelerated/${top}: clean branch was not consumed"
    if [[ "${top}" == POLY34_HIER_CLEAN ]]; then
      grep -Fq -- \
        "contexts=2051 poly_boxes=2048 active_boxes=2048 gates=2048" \
        "${log}" ||
        die "accelerated/${top}: hierarchy occurrence census is wrong"
      grep -Fq -- "atomic_empty=2048 fallback_gates=0" "${log}" ||
        die "accelerated/${top}: hierarchy certificate count is wrong"
      grep -Fq -- \
        "contexts=2051 cells=3 stored_boxes=3 poly_boxes=2048 active_boxes=2048 gate_boxes=2048" \
        "${log}" ||
        die "accelerated/${top}: compact lowering census is wrong"
    fi
  else
    grep -Fq -- \
      "CUDA POLY.3/.4 terminal-empty certificate: outcome=not-empty-cpu-fallback" \
      "${log}" ||
      die "accelerated/${top}: positive marker did not request CPU fallback"
    grep -Fq -- "POLY34_LIVE_DECK ok top=${top} accelerated=0" "${log}" ||
      die "accelerated/${top}: hit branch did not execute both CPU rules"
    if [[ "${top}" == POLY34_HIER_TRANSFORM_HIT ]]; then
      grep -Fq -- \
        "contexts=7 poly_boxes=6 active_boxes=6 gates=6 certified_mask=2" \
        "${log}" ||
        die "accelerated/${top}: transformed hit census is wrong"
      grep -Fq -- "atomic_empty=0 fallback_gates=6" "${log}" ||
        die "accelerated/${top}: transformed hit was not atomically declined"
      grep -Fq -- \
        "contexts=7 cells=2 stored_boxes=3 poly_boxes=6 active_boxes=6 gate_boxes=6" \
        "${log}" ||
        die "accelerated/${top}: transformed compact lowering census is wrong"
    fi
  fi
done
echo \
  "POLY34_LIVE_GATE ok gate=cuda clean=3 hits=4 reports=cpu-identical atomic=1 transformed-hit=1 manhattan-primary=1"

run_case wrong-layer cuda POLY34_CLEAN clean -rd poly_layer=19
compare_reports oracle wrong-layer POLY34_CLEAN
wrong_layer_log="${work}/logs/wrong-layer-POLY34_CLEAN.log"
[[ "$(attempt_count "${wrong_layer_log}")" == 0 ]] ||
  die "wrong-layer: an unqualified layer reached the backend"
[[ "$(lowering_count "${wrong_layer_log}")" == 0 ]] ||
  die "wrong-layer: an unqualified layer caused scene lowering"
grep -Fq -- "POLY34_LIVE_DECK ok top=POLY34_CLEAN accelerated=0" \
  "${wrong_layer_log}" ||
  die "wrong-layer: pristine CPU rules were not executed"
echo "POLY34_LIVE_GATE ok gate=wrong-layer report=cpu-identical no-lowering=1"

run_case backend-off backend-off POLY34_HIER_CLEAN clean
compare_reports oracle backend-off POLY34_HIER_CLEAN
backend_off_log="${work}/logs/backend-off-POLY34_HIER_CLEAN.log"
[[ "$(attempt_count "${backend_off_log}")" == 0 ]] ||
  die "backend-off: an opt-in-disabled symbol was called"
[[ "$(lowering_count "${backend_off_log}")" == 0 ]] ||
  die "backend-off: opt-in-disabled backend caused scene lowering"
echo "POLY34_LIVE_GATE ok gate=backend-off report=cpu-identical"

run_case missing missing POLY34_MIXED both
compare_reports oracle missing POLY34_MIXED
missing_log="${work}/logs/missing-POLY34_MIXED.log"
[[ "$(attempt_count "${missing_log}")" == 0 ]] ||
  die "missing-backend: an unavailable symbol was called"
[[ "$(lowering_count "${missing_log}")" == 0 ]] ||
  die "missing-backend: unavailable backend caused scene lowering"
grep -Fq -- "unable to load CUDA spatial backend:" "${missing_log}" ||
  die "missing-backend: expected loader warning is absent"
echo "POLY34_LIVE_GATE ok gate=missing-backend report=cpu-identical"

run_case capacity capacity POLY34_HIER_CLEAN clean
compare_reports oracle capacity POLY34_HIER_CLEAN
capacity_log="${work}/logs/capacity-POLY34_HIER_CLEAN.log"
assert_one_attempt "${capacity_log}" capacity
grep -Fq -- \
  "CUDA POLY.3/.4 terminal-empty certificate: outcome=fallback" \
  "${capacity_log}" ||
  die "capacity: backend did not request CPU fallback"
grep -Eq -- "fallback_flags=[1-9][0-9]*" "${capacity_log}" ||
  die "capacity: nonzero fallback flag is absent"
grep -Fq -- \
  "POLY34_LIVE_DECK ok top=POLY34_HIER_CLEAN accelerated=0" \
  "${capacity_log}" ||
  die "capacity: both historical CPU rules were not executed"
echo "POLY34_LIVE_GATE ok gate=capacity report=cpu-identical"

echo \
  "POLY34_LIVE_GATE PASS oracles=7 cuda=7 wrong-layer=1 backend-off=1 missing=1 capacity=1 reports=18"
