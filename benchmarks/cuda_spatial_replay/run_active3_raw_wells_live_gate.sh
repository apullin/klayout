#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ACTIVE.3 raw-WELL live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
klayout=
backend=
library_path=
keep_work=0
while (($#)); do
  case "$1" in
    --klayout) klayout=$2; shift 2 ;;
    --backend) backend=$2; shift 2 ;;
    --library-path) library_path=$2; shift 2 ;;
    --keep-work) keep_work=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -x "${klayout}" ]] || die "missing executable --klayout"
[[ -f "${backend}" ]] || die "missing --backend"
[[ -f "${here}/active3_raw_wells_live_fixture.rb" ]] ||
  die "missing fixture generator"
[[ -f "${here}/active3_raw_wells_live_gate.drc" ]] || die "missing deck"

klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
if [[ -z "${library_path}" ]]; then
  library_path=$(dirname -- "${klayout}")
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-active3-raw-wells-live.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "ACTIVE3_RAW_WELLS_LIVE_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT
mkdir -p -- "${work}/logs" "${work}/reports" "${work}/runtime"

run_klayout() {
  local lane=$1
  shift
  local runtime="${work}/runtime/${lane}"
  mkdir -p -- "${runtime}/home" "${runtime}/config" "${runtime}/cache"
  env \
    -u KLAYOUT_CUDA_SPATIAL_BACKEND \
    -u KLAYOUT_CUDA_ACTIVE3 \
    -u KLAYOUT_CUDA_ACTIVE3_TELEMETRY \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_TELEMETRY \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_CONTEXTS \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_GRID_CELLS \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_MEMBERSHIPS \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_PAIR_WORK \
    QT_QPA_PLATFORM=offscreen \
    HOME="${runtime}/home" \
    XDG_CONFIG_HOME="${runtime}/config" \
    XDG_CACHE_HOME="${runtime}/cache" \
    LD_LIBRARY_PATH="${library_path}" \
    "$@"
}

fixture="${work}/fixture.gds"
run_klayout fixture \
  "${klayout}" -b -r "${here}/active3_raw_wells_live_fixture.rb" \
  -rd "output=${fixture}" >"${work}/logs/fixture.log" 2>&1 ||
  die "fixture generation failed"
grep -Fq "ACTIVE3_RAW_WELLS_LIVE_FIXTURE ok" "${work}/logs/fixture.log" ||
  die "fixture completion marker missing"

canonicalize() {
  sed '/<generator>/d' "$1" >"$2"
}

run_lane() {
  local mode=$1
  local top=$2
  local capacity=${3:-normal}
  local lane="${mode}-${top}-${capacity}"
  local report="${work}/reports/${lane}.lyrdb"
  local log="${work}/logs/${lane}.log"
  local -a prefix=(env)
  if [[ "${mode}" == cuda ]]; then
    prefix+=(
      KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
      KLAYOUT_CUDA_ACTIVE3_RAW_WELLS=1
      KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_TELEMETRY=1
    )
    if [[ "${capacity}" == tiny ]]; then
      prefix+=(KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_MAX_PAIR_WORK=1)
    fi
  fi
  run_klayout "${lane}" "${prefix[@]}" \
    "${klayout}" -b -r "${here}/active3_raw_wells_live_gate.drc" \
    -rd "input=${fixture}" -rd "topcell=${top}" \
    -rd "output=${report}" -rd "mode=${mode}" >"${log}" 2>&1 ||
    die "${lane} execution failed"
  [[ -s "${report}" ]] || die "${lane} produced no report"
  canonicalize "${report}" "${report}.canonical"
}

tops=(
  ACTIVE3_RAW_WELLS_CLEAN
  ACTIVE3_RAW_WELLS_FALSE_POSITIVE
  ACTIVE3_RAW_WELLS_TRUE_HIT
)
for top in "${tops[@]}"; do
  run_lane cpu "${top}"
  run_lane cuda "${top}"
  cmp \
    "${work}/reports/cpu-${top}-normal.lyrdb.canonical" \
    "${work}/reports/cuda-${top}-normal.lyrdb.canonical" ||
    die "${top}: CUDA report differs from pristine CPU"
done

grep -Fq \
  "mode=cuda top=ACTIVE3_RAW_WELLS_CLEAN certified=1 empty=1" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_CLEAN-normal.log" ||
  die "clean scene was not certified"
grep -Eq \
  'nwell_edges=4 pwell_edges=4 active_edges=4 pair_bound=32 .*candidates=' \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_CLEAN-normal.log" ||
  die "clean split-edge/candidate census is missing"
grep -Fq "outcome=raw-hits-cpu-fallback" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_FALSE_POSITIVE-normal.log" ||
  die "internal raw-WELL false positive did not request CPU fallback"
grep -Fq \
  "mode=cuda top=ACTIVE3_RAW_WELLS_FALSE_POSITIVE certified=0 empty=1" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_FALSE_POSITIVE-normal.log" ||
  die "false-positive scene did not preserve an empty CPU result"
grep -Fq \
  "mode=cuda top=ACTIVE3_RAW_WELLS_TRUE_HIT certified=0 empty=0" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_TRUE_HIT-normal.log" ||
  die "true violation did not reach the CPU marker path"

run_lane cuda ACTIVE3_RAW_WELLS_CLEAN tiny
cmp \
  "${work}/reports/cpu-ACTIVE3_RAW_WELLS_CLEAN-normal.lyrdb.canonical" \
  "${work}/reports/cuda-ACTIVE3_RAW_WELLS_CLEAN-tiny.lyrdb.canonical" ||
  die "capacity fallback report differs from pristine CPU"
grep -Fq "reason=pair-work-capacity" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_CLEAN-tiny.log" ||
  die "pair-work capacity fallback was not observed"

echo \
  "ACTIVE3_RAW_WELLS_LIVE_GATE PASS clean=certified" \
  "false_positive=cpu-fallback true_hit=cpu-marker capacity=cpu-fallback"
