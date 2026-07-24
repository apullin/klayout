#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_m1_width_space_live_gate.sh \
    --klayout PATH --backend PATH [--keep-work]

Runs a focused integrity gate for the opt-in live METAL1.1/METAL1.2 CUDA
empty certificate.  The supplied KLayout binary is its own pristine CPU
oracle with the opt-in disabled.  The gate checks:

  * a clean hierarchical merged batch certifies both empty outputs;
  * width and spacing hits invoke CUDA, then preserve complete CPU markers;
  * raw semantics, a changed option, and reversed output order never invoke;
  * a missing backend and a fail-closed capacity limit preserve CPU markers.

All generated layouts, reports, logs, homes, and caches live under a fresh
TMPDIR directory.  They are removed unless --keep-work is specified.
EOF
}

die() {
  echo "M1 width/space live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/m1_width_space_live_fixture.rb"
deck="${here}/m1_width_space_live_gate.drc"

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

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-m1-width-space-live-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "M1_WIDTH_SPACE_LIVE_GATE work=${work}"
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
  mkdir -p -- "${runtime}/home" "${runtime}/config" "${runtime}/cache"

  local -a command=(
    env
    -u KLAYOUT_CUDA_SPATIAL_BACKEND
    -u KLAYOUT_CUDA_SPATIAL_DEVICE
    -u KLAYOUT_CUDA_SPATIAL_TELEMETRY
    -u KLAYOUT_CUDA_SPATIAL_MIN_RECORDS
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_CELLS
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_CONTEXTS
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_STORED_POLYGONS
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_STORED_EDGES
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_FLAT_POLYGONS
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_FLAT_EDGES
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_GRID_CELLS
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_MEMBERSHIPS
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_PAIR_WORK
    QT_QPA_PLATFORM=offscreen
    HOME="${runtime}/home"
    XDG_CONFIG_HOME="${runtime}/config"
    XDG_CACHE_HOME="${runtime}/cache"
  )

  case "${mode}" in
    off)
      ;;
    cuda)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_M1_WIDTH_SPACE=1
        KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1
      )
      ;;
    missing)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${missing_backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_M1_WIDTH_SPACE=1
        KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1
      )
      ;;
    capacity)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_M1_WIDTH_SPACE=1
        KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1
        KLAYOUT_CUDA_M1_WIDTH_SPACE_MAX_GRID_CELLS=1
      )
      ;;
    *)
      die "internal error: unknown mode ${mode}"
      ;;
  esac

  command+=("${klayout}" "$@")
  "${command[@]}"
}

fixture="${work}/m1-width-space-live-fixture.gds"
fixture_log="${work}/logs/fixture.log"
if ! run_klayout off fixture -b -r "${fixture_script}" \
  -rd "output=${fixture}" >"${fixture_log}" 2>&1; then
  cat -- "${fixture_log}" >&2
  die "fixture generation failed"
fi
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
grep -Fq -- \
  "M1_WIDTH_SPACE_LIVE_FIXTURE ok path=${fixture} tops=3" "${fixture_log}" ||
  die "fixture completion marker or top count is wrong"
echo "M1_WIDTH_SPACE_LIVE_GATE ok gate=fixture tops=3 hierarchy=1"

canonicalize_report() {
  local report=$1
  local output=$2
  sed '/<generator>/d' "${report}" >"${output}"
}

category_count() {
  local report=$1
  local category=$2
  grep -Fc -- "<category>'${category}'</category>" "${report}" || true
}

assert_report_shape() {
  local report=$1
  local top=$2
  local width_count
  local space_count
  grep -Fq -- "<top-cell>${top}</top-cell>" "${report}" ||
    die "${top}: report top-cell marker is missing"
  width_count=$(category_count "${report}" METAL1.1)
  space_count=$(category_count "${report}" METAL1.2)
  case "${top}" in
    M1_WIDTH_SPACE_CLEAN)
      ((width_count == 0 && space_count == 0)) ||
        die "${top}: expected no markers, found width=${width_count} space=${space_count}"
      ;;
    M1_WIDTH_SPACE_WIDTH_HIT)
      ((width_count > 0 && space_count == 0)) ||
        die "${top}: expected width-only markers, found width=${width_count} space=${space_count}"
      ;;
    M1_WIDTH_SPACE_SPACE_HIT)
      ((width_count == 0 && space_count > 0)) ||
        die "${top}: expected spacing-only markers, found width=${width_count} space=${space_count}"
      ;;
    *)
      die "internal error: unknown top ${top}"
      ;;
  esac
}

run_case() {
  local lane=$1
  local mode=$2
  local variant=$3
  local top=$4
  local lane_dir="${work}/reports/${lane}"
  local report="${lane_dir}/${top}.lyrdb"
  local log="${work}/logs/${lane}-${top}.log"
  mkdir -p -- "${lane_dir}"

  if ! run_klayout "${mode}" "${lane}-${top}" -b -r "${deck}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    -rd "variant=${variant}" >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}/${top}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] || die "${lane}/${top}: DRC produced no report"
  grep -Fq -- \
    "M1_WIDTH_SPACE_LIVE_DECK ok variant=${variant} top=${top}" "${log}" ||
    die "${lane}/${top}: deck completion marker is missing"
  assert_report_shape "${report}" "${top}"
}

compare_reports() {
  local reference_lane=$1
  local candidate_lane=$2
  local top=$3
  local reference="${work}/reports/${reference_lane}/${top}.lyrdb"
  local candidate="${work}/reports/${candidate_lane}/${top}.lyrdb"
  local reference_canonical="${work}/reports/${reference_lane}/${top}.canonical.lyrdb"
  local candidate_canonical="${work}/reports/${candidate_lane}/${top}.canonical.lyrdb"
  canonicalize_report "${reference}" "${reference_canonical}"
  canonicalize_report "${candidate}" "${candidate_canonical}"
  cmp -s -- "${reference_canonical}" "${candidate_canonical}" ||
    die "${candidate_lane}/${top}: report differs from ${reference_lane}"
}

assert_no_m1_invocation() {
  local log=$1
  local label=$2
  if grep -Fq -- "CUDA M1 width/space empty certificate:" "${log}" ||
     grep -Fq -- "CUDA M1 width/space live lowering:" "${log}"; then
    die "${label}: an unqualified lane invoked the M1 CUDA seam"
  fi
}

certificate_line() {
  local log=$1
  grep -F -- "CUDA M1 width/space empty certificate:" "${log}" || true
}

assert_one_certificate() {
  local log=$1
  local label=$2
  local count
  count=$(grep -Fc -- "CUDA M1 width/space empty certificate:" "${log}" || true)
  [[ "${count}" == 1 ]] ||
    die "${label}: expected one M1 certificate line, found ${count}"
  count=$(grep -Fc -- "CUDA M1 width/space live lowering:" "${log}" || true)
  [[ "${count}" == 1 ]] ||
    die "${label}: expected one live-lowering line, found ${count}"
}

assert_hit_fallback() {
  local log=$1
  local kind=$2
  local label=$3
  local line
  local width_hits
  local space_hits
  assert_one_certificate "${log}" "${label}"
  line=$(certificate_line "${log}")
  [[ "${line}" == *"outcome=fallback"* ||
     "${line}" == *"outcome=raw-hits-cpu-fallback"* ]] ||
    die "${label}: hit lane did not request CPU fallback"
  width_hits=$(sed -n 's/.* width_hits=\([0-9][0-9]*\).*/\1/p' <<<"${line}")
  space_hits=$(sed -n 's/.* space_hits=\([0-9][0-9]*\).*/\1/p' <<<"${line}")
  [[ -n "${width_hits}" && -n "${space_hits}" ]] ||
    die "${label}: hit counters are missing"
  if [[ "${kind}" == width ]]; then
    ((width_hits > 0 && space_hits == 0)) ||
      die "${label}: expected width-only GPU hits, found width=${width_hits} space=${space_hits}"
  else
    ((width_hits == 0 && space_hits > 0)) ||
      die "${label}: expected spacing-only GPU hits, found width=${width_hits} space=${space_hits}"
  fi
}

qualified_tops=(
  M1_WIDTH_SPACE_CLEAN
  M1_WIDTH_SPACE_WIDTH_HIT
  M1_WIDTH_SPACE_SPACE_HIT
)
for top in "${qualified_tops[@]}"; do
  run_case oracle-qualified off qualified "${top}"
  assert_no_m1_invocation \
    "${work}/logs/oracle-qualified-${top}.log" "oracle-qualified/${top}"
done

run_case oracle-raw off raw M1_WIDTH_SPACE_WIDTH_HIT
run_case oracle-changed-option off changed_option M1_WIDTH_SPACE_WIDTH_HIT
run_case oracle-wrong-order off wrong_order M1_WIDTH_SPACE_SPACE_HIT
echo "M1_WIDTH_SPACE_LIVE_GATE ok gate=cpu-oracles reports=6"

for top in "${qualified_tops[@]}"; do
  run_case cuda-qualified cuda qualified "${top}"
  compare_reports oracle-qualified cuda-qualified "${top}"
done

clean_log="${work}/logs/cuda-qualified-M1_WIDTH_SPACE_CLEAN.log"
assert_one_certificate "${clean_log}" "cuda-qualified/clean"
grep -Fq -- \
  "CUDA M1 width/space empty certificate: outcome=certified-empty" "${clean_log}" ||
  die "cuda-qualified/clean: expected a certified-empty transaction"
grep -Fq -- \
  "width_empty=1 spacing_empty=1" "${clean_log}" ||
  die "cuda-qualified/clean: both published outputs were not empty"

assert_hit_fallback \
  "${work}/logs/cuda-qualified-M1_WIDTH_SPACE_WIDTH_HIT.log" \
  width "cuda-qualified/width-hit"
assert_hit_fallback \
  "${work}/logs/cuda-qualified-M1_WIDTH_SPACE_SPACE_HIT.log" \
  space "cuda-qualified/space-hit"
echo \
  "M1_WIDTH_SPACE_LIVE_GATE ok gate=qualified clean=certified-empty hits=2 report=cpu-identical"

run_case cuda-raw cuda raw M1_WIDTH_SPACE_WIDTH_HIT
compare_reports oracle-raw cuda-raw M1_WIDTH_SPACE_WIDTH_HIT
assert_no_m1_invocation \
  "${work}/logs/cuda-raw-M1_WIDTH_SPACE_WIDTH_HIT.log" "cuda-raw"

run_case cuda-changed-option cuda changed_option M1_WIDTH_SPACE_WIDTH_HIT
compare_reports oracle-changed-option cuda-changed-option M1_WIDTH_SPACE_WIDTH_HIT
assert_no_m1_invocation \
  "${work}/logs/cuda-changed-option-M1_WIDTH_SPACE_WIDTH_HIT.log" \
  "cuda-changed-option"

run_case cuda-wrong-order cuda wrong_order M1_WIDTH_SPACE_SPACE_HIT
compare_reports oracle-wrong-order cuda-wrong-order M1_WIDTH_SPACE_SPACE_HIT
assert_no_m1_invocation \
  "${work}/logs/cuda-wrong-order-M1_WIDTH_SPACE_SPACE_HIT.log" \
  "cuda-wrong-order"
echo \
  "M1_WIDTH_SPACE_LIVE_GATE ok gate=qualification raw=declined changed_option=declined wrong_order=declined"

run_case missing-backend missing qualified M1_WIDTH_SPACE_WIDTH_HIT
compare_reports oracle-qualified missing-backend M1_WIDTH_SPACE_WIDTH_HIT
assert_no_m1_invocation \
  "${work}/logs/missing-backend-M1_WIDTH_SPACE_WIDTH_HIT.log" \
  "missing-backend"
grep -Fq -- "unable to load CUDA spatial backend:" \
  "${work}/logs/missing-backend-M1_WIDTH_SPACE_WIDTH_HIT.log" ||
  die "missing-backend: expected loader warning is missing"
echo \
  "M1_WIDTH_SPACE_LIVE_GATE ok gate=missing-backend report=cpu-identical"

run_case capacity capacity qualified M1_WIDTH_SPACE_WIDTH_HIT
compare_reports oracle-qualified capacity M1_WIDTH_SPACE_WIDTH_HIT
capacity_log="${work}/logs/capacity-M1_WIDTH_SPACE_WIDTH_HIT.log"
assert_one_certificate "${capacity_log}" "capacity"
grep -Fq -- \
  "CUDA M1 width/space empty certificate: outcome=fallback" "${capacity_log}" ||
  die "capacity: backend did not decline the certificate"
capacity_line=$(certificate_line "${capacity_log}")
[[ "${capacity_line}" =~ fallback_flags=[1-9][0-9]* ]] ||
  die "capacity: expected a nonzero fail-closed fallback flag"
echo \
  "M1_WIDTH_SPACE_LIVE_GATE ok gate=capacity report=cpu-identical"

echo \
  "M1_WIDTH_SPACE_LIVE_GATE PASS oracles=6 cuda_qualified=3 qualification_declines=3 fail_closed=2"
