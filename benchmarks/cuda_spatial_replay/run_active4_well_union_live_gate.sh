#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ACTIVE.4 exact WELL-union subset live gate: $*" >&2
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
[[ -f "${here}/active4_well_union_live_fixture.rb" ]] ||
  die "missing fixture generator"
[[ -f "${here}/active4_well_union_live_gate.drc" ]] ||
  die "missing deck"

klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
if [[ -z "${library_path}" ]]; then
  library_path=$(dirname -- "${klayout}")
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-active4-well-union-live.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "ACTIVE4_WELL_UNION_LIVE_GATE work=${work}"
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
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION \
    -u KLAYOUT_CUDA_ACTIVE3_WELL_UNION_TELEMETRY \
    -u KLAYOUT_CUDA_ACTIVE4_WELL_UNION \
    -u KLAYOUT_CUDA_ACTIVE4_WELL_UNION_TELEMETRY \
    -u KLAYOUT_CUDA_ACTIVE4_WELL_UNION_MAX_SEARCH_STEPS \
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
  "${klayout}" -b -r "${here}/active4_well_union_live_fixture.rb" \
  -rd "output=${fixture}" >"${work}/logs/fixture.log" 2>&1 ||
  die "fixture generation failed"
grep -Fq "ACTIVE4_WELL_UNION_LIVE_FIXTURE ok" "${work}/logs/fixture.log" ||
  die "fixture completion marker missing"

canonicalize() {
  sed '/<generator>/d' "$1" >"$2"
}

run_lane() {
  local mode=$1
  local top=$2
  local variant=${3:-normal}
  local lane="${mode}-${top}-${variant}"
  local report="${work}/reports/${lane}.lyrdb"
  local log="${work}/logs/${lane}.log"
  local -a prefix=(env)
  if [[ "${mode}" == cuda ]]; then
    prefix+=(KLAYOUT_CUDA_ACTIVE4_WELL_UNION=1)
    if [[ "${variant}" != missing ]]; then
      prefix+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_ACTIVE4_WELL_UNION_TELEMETRY=1
      )
    fi
    if [[ "${variant}" == tiny ]]; then
      prefix+=(KLAYOUT_CUDA_ACTIVE3_WELL_UNION_MAX_RECTANGLES=1)
    fi
    if [[ "${variant}" == search-pass ]]; then
      prefix+=(KLAYOUT_CUDA_ACTIVE4_WELL_UNION_MAX_SEARCH_STEPS=5)
    fi
    if [[ "${variant}" == search-decline ]]; then
      prefix+=(KLAYOUT_CUDA_ACTIVE4_WELL_UNION_MAX_SEARCH_STEPS=4)
    fi
  fi
  run_klayout "${lane}" "${prefix[@]}" \
    "${klayout}" -b -r "${here}/active4_well_union_live_gate.drc" \
    -rd "input=${fixture}" -rd "topcell=${top}" \
    -rd "output=${report}" -rd "mode=${mode}" >"${log}" 2>&1 ||
    die "${lane} execution failed"
  [[ -s "${report}" ]] || die "${lane} produced no report"
  canonicalize "${report}" "${report}.canonical"
}

clean_tops=(
  ACTIVE4_INSIDE
  ACTIVE4_BOUNDARY_TOUCH
  ACTIVE4_OVERLAP_ACTIVE
  ACTIVE4_INTERNAL_WELL_BOUNDARY
  ACTIVE4_HIERARCHY_TRANSFORMS
)
hit_tops=(
  ACTIVE4_PARTIAL_OUTSIDE
  ACTIVE4_FAR_OUTSIDE
  ACTIVE4_WELL_HOLE
)
tops=("${clean_tops[@]}" "${hit_tops[@]}")
for top in "${tops[@]}"; do
  run_lane cpu "${top}"
  run_lane cuda "${top}"
  cmp \
    "${work}/reports/cpu-${top}-normal.lyrdb.canonical" \
    "${work}/reports/cuda-${top}-normal.lyrdb.canonical" ||
    die "${top}: CUDA report differs from pristine CPU"
done

for top in "${clean_tops[@]}"; do
  grep -Fq \
    "mode=cuda top=${top} available=1 certified=1 empty=1" \
    "${work}/logs/cuda-${top}-normal.log" ||
    die "${top}: clean exact set was not certified"
  grep -Fq "outcome=certified-empty" \
    "${work}/logs/cuda-${top}-normal.log" ||
    die "${top}: complete certificate telemetry missing"
done
for top in "${hit_tops[@]}"; do
  grep -Fq \
    "mode=cuda top=${top} available=1 certified=0 empty=0" \
    "${work}/logs/cuda-${top}-normal.log" ||
    die "${top}: witness did not reach literal CPU fallback"
  grep -Fq "outcome=raw-hits-cpu-fallback" \
    "${work}/logs/cuda-${top}-normal.log" ||
    die "${top}: exact non-subset telemetry missing"
done

# Missing backend and bounded capacity are both mandatory fail-closed paths.
run_lane cuda ACTIVE4_INTERNAL_WELL_BOUNDARY missing
cmp \
  "${work}/reports/cpu-ACTIVE4_INTERNAL_WELL_BOUNDARY-normal.lyrdb.canonical" \
  "${work}/reports/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-missing.lyrdb.canonical" ||
  die "missing-backend fallback report differs from pristine CPU"
grep -Fq \
  "mode=cuda top=ACTIVE4_INTERNAL_WELL_BOUNDARY available=1 certified=0 empty=1" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-missing.log" ||
  die "missing backend did not preserve literal CPU result"

run_lane cuda ACTIVE4_INTERNAL_WELL_BOUNDARY tiny
cmp \
  "${work}/reports/cpu-ACTIVE4_INTERNAL_WELL_BOUNDARY-normal.lyrdb.canonical" \
  "${work}/reports/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-tiny.lyrdb.canonical" ||
  die "capacity fallback report differs from pristine CPU"
grep -Fq "outcome=fallback" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-tiny.log" ||
  die "rectangle-capacity fallback was not observed"
grep -Fq \
  "mode=cuda top=ACTIVE4_INTERNAL_WELL_BOUNDARY available=1 certified=0 empty=1" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-tiny.log" ||
  die "capacity fallback did not preserve literal CPU result"

run_lane cuda ACTIVE4_INTERNAL_WELL_BOUNDARY search-pass
cmp \
  "${work}/reports/cpu-ACTIVE4_INTERNAL_WELL_BOUNDARY-normal.lyrdb.canonical" \
  "${work}/reports/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-pass.lyrdb.canonical" ||
  die "exact search-boundary pass report differs from pristine CPU"
grep -Fq "member_visits=5" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-pass.log" ||
  die "exact search-boundary work count drifted"
grep -Fq \
  "mode=cuda top=ACTIVE4_INTERNAL_WELL_BOUNDARY available=1 certified=1 empty=1" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-pass.log" ||
  die "exact search cap N did not certify"

run_lane cuda ACTIVE4_INTERNAL_WELL_BOUNDARY search-decline
cmp \
  "${work}/reports/cpu-ACTIVE4_INTERNAL_WELL_BOUNDARY-normal.lyrdb.canonical" \
  "${work}/reports/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-decline.lyrdb.canonical" ||
  die "search-capacity fallback report differs from pristine CPU"
grep -Fq "outcome=fallback" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-decline.log" ||
  die "search-capacity fallback was not observed"
grep -Fq "member_visits=5" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-decline.log" ||
  die "search-cap N-1 did not complete the exact N work items"
grep -Fq "device_flags=16" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-decline.log" ||
  die "dedicated search-capacity device flag was not observed"
grep -Fq \
  "mode=cuda top=ACTIVE4_INTERNAL_WELL_BOUNDARY available=1 certified=0 empty=1" \
  "${work}/logs/cuda-ACTIVE4_INTERNAL_WELL_BOUNDARY-search-decline.log" ||
  die "search-capacity fallback did not preserve literal CPU result"

echo \
  "ACTIVE4_WELL_UNION_LIVE_GATE PASS clean=5 witness=3" \
  "hierarchy=orthogonal-8 missing_backend=cpu-fallback" \
  "rectangle_capacity=cpu-fallback search_boundary=N-pass,N-1-fallback" \
  "exact_reports=identical"
