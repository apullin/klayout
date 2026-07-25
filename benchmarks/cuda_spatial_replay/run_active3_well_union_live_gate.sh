#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ACTIVE.3 exact WELL-union live gate: $*" >&2
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
[[ -f "${here}/active3_well_union_live_gate.drc" ]] ||
  die "missing deck"

klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
if [[ -z "${library_path}" ]]; then
  library_path=$(dirname -- "${klayout}")
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-active3-well-union-live.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "ACTIVE3_WELL_UNION_LIVE_GATE work=${work}"
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
    -u KLAYOUT_CUDA_SPATIAL_DEVICE \
    -u KLAYOUT_CUDA_ACTIVE3 \
    -u KLAYOUT_CUDA_ACTIVE3_TELEMETRY \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS \
    -u KLAYOUT_CUDA_ACTIVE3_RAW_WELLS_TELEMETRY \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_TELEMETRY \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CELLS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CONTEXTS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_STORED_POLYGONS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_STORED_EDGES \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_FLAT_POLYGONS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_FLAT_EDGES \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_RECTANGLES \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_X_SLABS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_UNION_MEMBERSHIPS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_EVENTS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_RAW_SEGMENTS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_BOUNDARY_SEGMENTS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_SLABS_PER_RECTANGLE \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_ACTIVE_EDGES \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_GRID_CELLS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_ACTIVE_MEMBERSHIPS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_ACTIVE_CELL_VISITS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_MEMBER_VISITS \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_PAIR_WORK \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CELLS_PER_ACTIVE_EDGE \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_CELLS_PER_WELL_EDGE \
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
      KLAYOUT_CUDA_ACTIVE3_WELL_UNION=1
      KLAYOUT_CUDA_ACTIVE3_WELL_UNION_TELEMETRY=1
    )
    if [[ "${capacity}" == tiny ]]; then
      prefix+=(KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_RECTANGLES=1)
    fi
  fi
  run_klayout "${lane}" "${prefix[@]}" \
    "${klayout}" -b -r "${here}/active3_well_union_live_gate.drc" \
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
grep -Fq \
  "mode=cuda top=ACTIVE3_RAW_WELLS_FALSE_POSITIVE certified=1 empty=1" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_FALSE_POSITIVE-normal.log" ||
  die "internal raw-WELL boundary was not removed by the exact union"
grep -Fq "outcome=raw-hits-cpu-fallback" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_TRUE_HIT-normal.log" ||
  die "true violation did not request CPU fallback"
grep -Fq \
  "mode=cuda top=ACTIVE3_RAW_WELLS_TRUE_HIT certified=0 empty=0" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_TRUE_HIT-normal.log" ||
  die "true violation did not reach the CPU marker path"
grep -Fq \
  "complete resident exact-WELL-union ACTIVE3 empty certificate" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_CLEAN-normal.log" ||
  die "clean certificate did not complete the resident no-geometry-D2H path"

run_lane cuda ACTIVE3_RAW_WELLS_FALSE_POSITIVE tiny
cmp \
  "${work}/reports/cpu-ACTIVE3_RAW_WELLS_FALSE_POSITIVE-normal.lyrdb.canonical" \
  "${work}/reports/cuda-ACTIVE3_RAW_WELLS_FALSE_POSITIVE-tiny.lyrdb.canonical" ||
  die "capacity fallback report differs from pristine CPU"
grep -Fq "outcome=fallback" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_FALSE_POSITIVE-tiny.log" ||
  die "rectangle-capacity fallback was not observed"
grep -Fq \
  "mode=cuda top=ACTIVE3_RAW_WELLS_FALSE_POSITIVE certified=0 empty=1" \
  "${work}/logs/cuda-ACTIVE3_RAW_WELLS_FALSE_POSITIVE-tiny.log" ||
  die "rectangle-capacity fallback did not preserve the CPU result"

echo \
  "ACTIVE3_WELL_UNION_LIVE_GATE PASS clean=certified" \
  "internal_boundary=certified true_hit=cpu-fallback" \
  "rectangle_capacity=cpu-fallback no_geometry_d2h=proved"
