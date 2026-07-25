#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_contact4_live_gate.sh \
    --klayout PATH --backend PATH [--keep-work]

Runs the focused integrity gate for the opt-in live CONTACT.4 CUDA empty
certificate.  The supplied KLayout binary is its own pristine CPU oracle with
the opt-in disabled.  Qualified cases cover the strict 9/10-DBU boundary,
Euclidean endpoint 6/7 versus 6/8, partial projections, collinear
touch/overlap/separation, raw geometry whose merge changes edge subsegments,
and transformed arrays.

CUDA raw-hit lanes must preserve byte-equivalent canonical CPU reports.
Wrong DBU, options, operand order, and raw-primary semantics must not invoke
the seam.  Missing-backend and capacity lanes must fail closed to the CPU.

All artifacts live under a fresh TMPDIR directory and are removed unless
--keep-work is specified.
EOF
}

die() {
  echo "CONTACT.4 live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/contact4_live_fixture.rb"
deck="${here}/contact4_live_gate.drc"

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

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-contact4-live-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "CONTACT4_LIVE_GATE work=${work}"
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
    -u KLAYOUT_CUDA_ACTIVE3
    -u KLAYOUT_CUDA_ACTIVE3_TELEMETRY
    -u KLAYOUT_CUDA_M1_CONTACT
    -u KLAYOUT_CUDA_M1_CONTACT_TELEMETRY
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE
    -u KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY
    -u KLAYOUT_CUDA_VIA1_STACK
    -u KLAYOUT_CUDA_VIA1_STACK_TELEMETRY
    -u KLAYOUT_CUDA_CONTACT4
    -u KLAYOUT_CUDA_CONTACT4_TELEMETRY
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY
    -u KLAYOUT_CUDA_CONTACT4_MAX_CONTEXTS
    -u KLAYOUT_CUDA_CONTACT4_MAX_GRID_CELLS
    -u KLAYOUT_CUDA_CONTACT4_MAX_MEMBERSHIPS
    -u KLAYOUT_CUDA_CONTACT4_MAX_PAIR_WORK
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_CONTEXTS
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_GRID_CELLS
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_MEMBERSHIPS
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_PAIR_WORK
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
        KLAYOUT_CUDA_CONTACT4=1
        KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
      )
      ;;
    missing)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${missing_backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4=1
        KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
      )
      ;;
    capacity)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4=1
        KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4_MAX_GRID_CELLS=1
      )
      ;;
    *)
      die "internal error: unknown mode ${mode}"
      ;;
  esac

  command+=("${klayout}" "$@")
  "${command[@]}"
}

generate_fixture() {
  local lane=$1
  local dbu=$2
  local fixture=$3
  local log="${work}/logs/fixture-${lane}.log"
  if ! run_klayout off "fixture-${lane}" -b -r "${fixture_script}" \
    -rd "output=${fixture}" -rd "dbu=${dbu}" >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}: fixture generation failed"
  fi
  [[ -s "${fixture}" ]] || die "${lane}: fixture generator produced no layout"
  grep -Fq -- "CONTACT4_LIVE_FIXTURE ok path=${fixture} tops=14" "${log}" ||
    die "${lane}: fixture completion marker or top count is wrong"
}

fixture="${work}/contact4-live-fixture.gds"
wrong_dbu_fixture="${work}/contact4-live-wrong-dbu-fixture.gds"
generate_fixture default 0.0005 "${fixture}"
generate_fixture wrong-dbu 0.001 "${wrong_dbu_fixture}"
echo \
  "CONTACT4_LIVE_GATE ok gate=fixtures tops=14 dbu=0.0005 wrong_dbu=0.001"

canonicalize_report() {
  local report=$1
  local output=$2
  sed '/<generator>/d' "${report}" >"${output}"
}

category_count() {
  local report=$1
  grep -Fc -- "<category>'CONTACT.4'</category>" "${report}" || true
}

assert_report_shape() {
  local report=$1
  local top=$2
  local expected=$3
  local count
  grep -Fq -- "<top-cell>${top}</top-cell>" "${report}" ||
    die "${top}: report top-cell marker is missing"
  count=$(category_count "${report}")
  case "${expected}" in
    clean)
      ((count == 0)) ||
        die "${top}: expected no CONTACT.4 markers, found ${count}"
      ;;
    hit)
      ((count > 0)) ||
        die "${top}: expected CONTACT.4 markers, found none"
      ;;
    any)
      ;;
    *)
      die "internal error: unknown report expectation ${expected}"
      ;;
  esac
}

run_case() {
  local lane=$1
  local mode=$2
  local variant=$3
  local top=$4
  local input=$5
  local expected=$6
  local lane_dir="${work}/reports/${lane}"
  local report="${lane_dir}/${top}.lyrdb"
  local log="${work}/logs/${lane}-${top}.log"
  mkdir -p -- "${lane_dir}"

  if ! run_klayout "${mode}" "${lane}-${top}" -b -r "${deck}" \
    -rd "input=${input}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    -rd "variant=${variant}" >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}/${top}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] || die "${lane}/${top}: DRC produced no report"
  grep -Fq -- \
    "CONTACT4_LIVE_DECK ok variant=${variant} top=${top}" "${log}" ||
    die "${lane}/${top}: deck completion marker is missing"
  assert_report_shape "${report}" "${top}" "${expected}"
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

assert_no_contact4_invocation() {
  local log=$1
  local label=$2
  if grep -Fq -- "CUDA CONTACT.4 empty certificate:" "${log}" ||
     grep -Fq -- "CUDA CONTACT.4 live lowering:" "${log}"; then
    die "${label}: an unqualified lane invoked the CONTACT.4 CUDA seam"
  fi
}

certificate_line() {
  local log=$1
  grep -F -- "CUDA CONTACT.4 empty certificate:" "${log}" || true
}

assert_one_attempt() {
  local log=$1
  local label=$2
  local count
  count=$(grep -Fc -- "CUDA CONTACT.4 empty certificate:" "${log}" || true)
  [[ "${count}" == 1 ]] ||
    die "${label}: expected one CONTACT.4 certificate line, found ${count}"
  count=$(grep -Fc -- "CUDA CONTACT.4 live lowering:" "${log}" || true)
  [[ "${count}" == 1 ]] ||
    die "${label}: expected one CONTACT.4 live-lowering line, found ${count}"
}

assert_certified_empty() {
  local log=$1
  local label=$2
  assert_one_attempt "${log}" "${label}"
  grep -Fq -- \
    "CUDA CONTACT.4 empty certificate: outcome=certified-empty" "${log}" ||
    die "${label}: expected a certified-empty transaction"
}

assert_hit_fallback() {
  local log=$1
  local label=$2
  local line
  local raw_hits
  assert_one_attempt "${log}" "${label}"
  line=$(certificate_line "${log}")
  [[ "${line}" == *"outcome=raw-hits-cpu-fallback"* ]] ||
    die "${label}: hit lane did not request CPU fallback"
  raw_hits=$(sed -n 's/.* raw_hits=\([0-9][0-9]*\).*/\1/p' <<<"${line}")
  [[ -n "${raw_hits}" ]] || die "${label}: raw_hits counter is missing"
  ((raw_hits > 0)) || die "${label}: expected a nonzero raw_hits counter"
}

qualified_cases=(
  "CONTACT4_CLEAN:clean:cert"
  "CONTACT4_GAP_9:hit:hit"
  "CONTACT4_GAP_10:clean:cert"
  "CONTACT4_ENDPOINT_6_7:clean:hit"
  "CONTACT4_ENDPOINT_6_8:clean:cert"
  "CONTACT4_PARTIAL_PROJECTION_9:hit:hit"
  "CONTACT4_PARTIAL_PROJECTION_10:clean:cert"
  "CONTACT4_COLLINEAR_TOUCH:hit:hit"
  "CONTACT4_COLLINEAR_OVERLAP:hit:hit"
  "CONTACT4_COLLINEAR_SEPARATION:clean:cert"
  "CONTACT4_RAW_DUPLICATE_MERGE:clean:cert"
  "CONTACT4_RAW_OVERLAP_MERGE:clean:cert"
  "CONTACT4_HIERARCHY_CLEAN:clean:cert"
  "CONTACT4_HIERARCHY_HIT:hit:hit"
)

for spec in "${qualified_cases[@]}"; do
  top=${spec%%:*}
  rest=${spec#*:}
  report_expected=${rest%%:*}
  run_case \
    oracle-qualified off qualified "${top}" "${fixture}" "${report_expected}"
  assert_no_contact4_invocation \
    "${work}/logs/oracle-qualified-${top}.log" "oracle-qualified/${top}"
done
echo "CONTACT4_LIVE_GATE ok gate=cpu-oracles reports=14"

for spec in "${qualified_cases[@]}"; do
  top=${spec%%:*}
  rest=${spec#*:}
  report_expected=${rest%%:*}
  cuda_expected=${rest##*:}
  run_case \
    cuda-qualified cuda qualified "${top}" "${fixture}" "${report_expected}"
  compare_reports oracle-qualified cuda-qualified "${top}"
  log="${work}/logs/cuda-qualified-${top}.log"
  if [[ "${cuda_expected}" == cert ]]; then
    assert_certified_empty "${log}" "cuda-qualified/${top}"
  else
    assert_hit_fallback "${log}" "cuda-qualified/${top}"
  fi
done
echo \
  "CONTACT4_LIVE_GATE ok gate=qualified clean=8 hits=6 reports=cpu-identical"

unqualified_cases=(
  "raw-primary-duplicate:raw_primary:CONTACT4_RAW_DUPLICATE_MERGE:${fixture}"
  "raw-primary-overlap:raw_primary:CONTACT4_RAW_OVERLAP_MERGE:${fixture}"
  "changed-option:changed_option:CONTACT4_GAP_9:${fixture}"
  "wrong-order:wrong_order:CONTACT4_GAP_9:${fixture}"
  "wrong-dbu:qualified:CONTACT4_GAP_9:${wrong_dbu_fixture}"
)
for spec in "${unqualified_cases[@]}"; do
  IFS=: read -r lane variant top input <<<"${spec}"
  run_case "oracle-${lane}" off "${variant}" "${top}" "${input}" any
  run_case "cuda-${lane}" cuda "${variant}" "${top}" "${input}" any
  compare_reports "oracle-${lane}" "cuda-${lane}" "${top}"
  assert_no_contact4_invocation \
    "${work}/logs/cuda-${lane}-${top}.log" "cuda-${lane}/${top}"
done
echo \
  "CONTACT4_LIVE_GATE ok gate=qualification raw_primary=2 changed_option=1 wrong_order=1 wrong_dbu=1"

run_case missing-backend missing qualified CONTACT4_GAP_9 "${fixture}" hit
compare_reports oracle-qualified missing-backend CONTACT4_GAP_9
assert_no_contact4_invocation \
  "${work}/logs/missing-backend-CONTACT4_GAP_9.log" "missing-backend"
grep -Fq -- "unable to load CUDA spatial backend:" \
  "${work}/logs/missing-backend-CONTACT4_GAP_9.log" ||
  die "missing-backend: expected loader warning is missing"
echo "CONTACT4_LIVE_GATE ok gate=missing-backend report=cpu-identical"

run_case capacity capacity qualified CONTACT4_HIERARCHY_CLEAN "${fixture}" clean
compare_reports oracle-qualified capacity CONTACT4_HIERARCHY_CLEAN
capacity_log="${work}/logs/capacity-CONTACT4_HIERARCHY_CLEAN.log"
assert_one_attempt "${capacity_log}" "capacity"
grep -Fq -- \
  "CUDA CONTACT.4 empty certificate: outcome=fallback" "${capacity_log}" ||
  die "capacity: backend did not decline the certificate"
capacity_line=$(certificate_line "${capacity_log}")
[[ "${capacity_line}" =~ fallback_flags=[1-9][0-9]* ]] ||
  die "capacity: expected a nonzero fail-closed fallback flag"
echo "CONTACT4_LIVE_GATE ok gate=capacity report=cpu-identical"

echo \
  "CONTACT4_LIVE_GATE PASS oracles=19 cuda_qualified=14 qualification_declines=5 fail_closed=2"
