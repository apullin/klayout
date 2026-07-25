#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_contact4_raw_active_live_gate.sh \
    --klayout PATH --backend PATH \
    [--stale-backend PATH] [--keep-work]

Runs the focused integrity gate for the opt-in early raw-ACTIVE CONTACT.4
empty certificate.  The supplied KLayout binary is its own pristine CPU
oracle with all CONTACT.4 CUDA opt-ins disabled.

The gate verifies:

  * a clean raw scene certifies before merged ACTIVE is constructed;
  * a conservative raw false positive falls through to the established
    merged-ACTIVE certificate without publishing a false DRC marker;
  * a real raw hit falls through to the CPU marker path;
  * raw semantics, changed options, wrong operand order, and wrong DBU decline;
  * raw-profile capacity and loader failures preserve canonical CPU reports.

If --stale-backend is supplied, it must export the established
klayout_cuda_spatial_run_active3_empty_v1 symbol but not the new
klayout_cuda_spatial_run_contact4_raw_active_empty_v1 symbol.  The gate then
proves that missing raw capability is detected before raw scene lowering while
the established late CONTACT.4 certificate remains usable.

All artifacts live under a fresh TMPDIR directory and are removed unless
--keep-work is specified.
EOF
}

die() {
  echo "CONTACT.4 raw-ACTIVE live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/contact4_raw_active_live_fixture.rb"
deck="${here}/contact4_raw_active_live_gate.drc"

klayout=
backend=
stale_backend=
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
    --stale-backend)
      (($# >= 2)) || die "--stale-backend requires a value"
      stale_backend=$2
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
if [[ -n "${stale_backend}" ]]; then
  [[ -f "${stale_backend}" ]] ||
    die "stale CUDA backend is missing: ${stale_backend}"
  stale_backend=$(readlink -f -- "${stale_backend}")
fi

old_contact4_symbol=klayout_cuda_spatial_run_active3_empty_v1
raw_contact4_symbol=klayout_cuda_spatial_run_contact4_raw_active_empty_v1

symbol_present() {
  local library=$1
  local symbol=$2
  nm -D --defined-only "${library}" 2>/dev/null |
    awk '{print $NF}' |
    grep -Fxq -- "${symbol}"
}

if [[ -n "${stale_backend}" ]]; then
  command -v nm >/dev/null 2>&1 ||
    die "--stale-backend validation requires nm"
  symbol_present "${stale_backend}" "${old_contact4_symbol}" ||
    die "stale backend lacks established symbol ${old_contact4_symbol}"
  if symbol_present "${stale_backend}" "${raw_contact4_symbol}"; then
    die "stale backend unexpectedly exports new symbol ${raw_contact4_symbol}"
  fi
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-contact4-raw-active-live-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "CONTACT4_RAW_ACTIVE_LIVE_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- "${work}/reports" "${work}/logs" "${work}/runtime"
missing_backend="${work}/missing-cuda-spatial-backend.so"

raw_certificate_marker="CUDA CONTACT.4 raw-ACTIVE empty certificate:"
raw_lowering_marker="CUDA CONTACT.4 raw-ACTIVE live lowering:"
late_certificate_marker="CUDA CONTACT.4 empty certificate:"
late_lowering_marker="CUDA CONTACT.4 live lowering:"

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
    -u KLAYOUT_CUDA_M2_WIDTH_SPACE
    -u KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY
    -u KLAYOUT_CUDA_M2_RULES
    -u KLAYOUT_CUDA_M2_RULES_TELEMETRY
    -u KLAYOUT_CUDA_POLY34
    -u KLAYOUT_CUDA_POLY34_TELEMETRY
    -u KLAYOUT_CUDA_VIA1_STACK
    -u KLAYOUT_CUDA_VIA1_STACK_TELEMETRY
    -u KLAYOUT_CUDA_IMPLANT12
    -u KLAYOUT_CUDA_IMPLANT12_TELEMETRY
    -u KLAYOUT_CUDA_CONTACT4
    -u KLAYOUT_CUDA_CONTACT4_TELEMETRY
    -u KLAYOUT_CUDA_CONTACT4_MAX_CONTEXTS
    -u KLAYOUT_CUDA_CONTACT4_MAX_GRID_CELLS
    -u KLAYOUT_CUDA_CONTACT4_MAX_MEMBERSHIPS
    -u KLAYOUT_CUDA_CONTACT4_MAX_PAIR_WORK
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_CONTEXTS
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_GRID_CELLS
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_MEMBERSHIPS
    -u KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_PAIR_WORK
    -u KLAYOUT_CUDA_CONTACT4_EARLY_RAW_ACTIVE
    -u KLAYOUT_CUDA_DISCONNECTED_MERGE
    QT_QPA_PLATFORM=offscreen
    HOME="${runtime}/home"
    XDG_CONFIG_HOME="${runtime}/config"
    XDG_CACHE_HOME="${runtime}/cache"
    LD_LIBRARY_PATH="${klayout_dir}"
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
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY=1
      )
      ;;
    capacity)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4=1
        KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_MAX_MEMBERSHIPS=1
      )
      ;;
    missing)
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${missing_backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4=1
        KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY=1
      )
      ;;
    stale)
      [[ -n "${stale_backend}" ]] ||
        die "internal error: stale mode without --stale-backend"
      command+=(
        KLAYOUT_CUDA_SPATIAL_BACKEND="${stale_backend}"
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4=1
        KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE=1
        KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY=1
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
  [[ -s "${fixture}" ]] ||
    die "${lane}: fixture generator produced no layout"
  grep -Fq -- \
    "CONTACT4_RAW_ACTIVE_LIVE_FIXTURE ok path=${fixture} tops=4" "${log}" ||
    die "${lane}: fixture completion marker or top count is wrong"
}

fixture="${work}/contact4-raw-active-live-fixture.gds"
wrong_dbu_fixture="${work}/contact4-raw-active-live-wrong-dbu-fixture.gds"
generate_fixture default 0.0005 "${fixture}"
generate_fixture wrong-dbu 0.001 "${wrong_dbu_fixture}"
echo \
  "CONTACT4_RAW_ACTIVE_LIVE_GATE ok gate=fixtures tops=4 dbu=0.0005 wrong_dbu=0.001"

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
  [[ -s "${report}" ]] ||
    die "${lane}/${top}: DRC produced no report"
  grep -Fq -- \
    "CONTACT4_RAW_ACTIVE_LIVE_DECK ok variant=${variant} top=${top}" \
    "${log}" ||
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

marker_count() {
  local log=$1
  local marker=$2
  grep -Fc -- "${marker}" "${log}" || true
}

assert_marker_count() {
  local log=$1
  local marker=$2
  local expected=$3
  local label=$4
  local count
  count=$(marker_count "${log}" "${marker}")
  [[ "${count}" == "${expected}" ]] ||
    die "${label}: expected ${expected} '${marker}' lines, found ${count}"
}

certificate_line() {
  local log=$1
  local marker=$2
  grep -F -- "${marker}" "${log}" || true
}

assert_raw_attempt() {
  local log=$1
  local label=$2
  assert_marker_count "${log}" "${raw_certificate_marker}" 1 "${label}"
  assert_marker_count "${log}" "${raw_lowering_marker}" 1 "${label}"
}

assert_no_raw_attempt() {
  local log=$1
  local label=$2
  assert_marker_count "${log}" "${raw_certificate_marker}" 0 "${label}"
  assert_marker_count "${log}" "${raw_lowering_marker}" 0 "${label}"
}

assert_late_attempt() {
  local log=$1
  local label=$2
  assert_marker_count "${log}" "${late_certificate_marker}" 1 "${label}"
  assert_marker_count "${log}" "${late_lowering_marker}" 1 "${label}"
}

assert_no_late_attempt() {
  local log=$1
  local label=$2
  assert_marker_count "${log}" "${late_certificate_marker}" 0 "${label}"
  assert_marker_count "${log}" "${late_lowering_marker}" 0 "${label}"
}

assert_no_contact4_attempt() {
  local log=$1
  local label=$2
  assert_no_raw_attempt "${log}" "${label}"
  assert_no_late_attempt "${log}" "${label}"
}

assert_outcome() {
  local log=$1
  local marker=$2
  local outcome=$3
  local label=$4
  local line
  line=$(certificate_line "${log}" "${marker}")
  [[ "${line}" == *"outcome=${outcome}"* ]] ||
    die "${label}: expected outcome=${outcome}; line='${line}'"
}

assert_raw_certified() {
  local log=$1
  local label=$2
  assert_raw_attempt "${log}" "${label}"
  assert_outcome "${log}" "${raw_certificate_marker}" certified-empty "${label}"
}

assert_raw_hit() {
  local log=$1
  local label=$2
  local line
  assert_raw_attempt "${log}" "${label}"
  assert_outcome \
    "${log}" "${raw_certificate_marker}" raw-hits-cpu-fallback "${label}"
  line=$(certificate_line "${log}" "${raw_certificate_marker}")
  [[ "${line}" =~ raw_hits=[1-9][0-9]* ]] ||
    die "${label}: expected a nonzero raw_hits counter; line='${line}'"
}

assert_late_certified() {
  local log=$1
  local label=$2
  assert_late_attempt "${log}" "${label}"
  assert_outcome "${log}" "${late_certificate_marker}" certified-empty "${label}"
}

assert_late_hit() {
  local log=$1
  local label=$2
  local line
  assert_late_attempt "${log}" "${label}"
  assert_outcome \
    "${log}" "${late_certificate_marker}" raw-hits-cpu-fallback "${label}"
  line=$(certificate_line "${log}" "${late_certificate_marker}")
  [[ "${line}" =~ raw_hits=[1-9][0-9]* ]] ||
    die "${label}: expected a nonzero late raw_hits counter; line='${line}'"
}

qualified_cases=(
  "CONTACT4_RAW_ACTIVE_CLEAN:clean:cert:none"
  "CONTACT4_RAW_ACTIVE_HIERARCHY_CLEAN:clean:cert:none"
  "CONTACT4_RAW_ACTIVE_FALSE_POSITIVE:clean:hit:cert"
  "CONTACT4_RAW_ACTIVE_TRUE_HIT:hit:hit:hit"
)

for spec in "${qualified_cases[@]}"; do
  IFS=: read -r top report_expected raw_expected late_expected <<<"${spec}"
  run_case \
    oracle-qualified off qualified "${top}" "${fixture}" "${report_expected}"
  assert_no_contact4_attempt \
    "${work}/logs/oracle-qualified-${top}.log" "oracle-qualified/${top}"
done
echo "CONTACT4_RAW_ACTIVE_LIVE_GATE ok gate=cpu-oracles reports=4"

for spec in "${qualified_cases[@]}"; do
  IFS=: read -r top report_expected raw_expected late_expected <<<"${spec}"
  run_case \
    cuda-qualified cuda qualified "${top}" "${fixture}" "${report_expected}"
  compare_reports oracle-qualified cuda-qualified "${top}"
  log="${work}/logs/cuda-qualified-${top}.log"
  case "${raw_expected}" in
    cert)
      assert_raw_certified "${log}" "cuda-qualified/${top}"
      ;;
    hit)
      assert_raw_hit "${log}" "cuda-qualified/${top}"
      ;;
    *)
      die "internal error: unknown raw expectation ${raw_expected}"
      ;;
  esac
  case "${late_expected}" in
    none)
      assert_no_late_attempt "${log}" "cuda-qualified/${top}"
      ;;
    cert)
      assert_late_certified "${log}" "cuda-qualified/${top}"
      ;;
    hit)
      assert_late_hit "${log}" "cuda-qualified/${top}"
      ;;
    *)
      die "internal error: unknown late expectation ${late_expected}"
      ;;
  esac
done
echo \
  "CONTACT4_RAW_ACTIVE_LIVE_GATE ok gate=qualified early_cert=2 conservative_fallback=1 true_hit=1 reports=cpu-identical"

unqualified_cases=(
  "raw-primary:raw_primary:CONTACT4_RAW_ACTIVE_CLEAN:${fixture}"
  "changed-option:changed_option:CONTACT4_RAW_ACTIVE_TRUE_HIT:${fixture}"
  "wrong-order:wrong_order:CONTACT4_RAW_ACTIVE_TRUE_HIT:${fixture}"
  "wrong-dbu:qualified:CONTACT4_RAW_ACTIVE_TRUE_HIT:${wrong_dbu_fixture}"
)
for spec in "${unqualified_cases[@]}"; do
  IFS=: read -r lane variant top input <<<"${spec}"
  run_case "oracle-${lane}" off "${variant}" "${top}" "${input}" any
  run_case "cuda-${lane}" cuda "${variant}" "${top}" "${input}" any
  compare_reports "oracle-${lane}" "cuda-${lane}" "${top}"
  assert_no_contact4_attempt \
    "${work}/logs/cuda-${lane}-${top}.log" "cuda-${lane}/${top}"
done
echo \
  "CONTACT4_RAW_ACTIVE_LIVE_GATE ok gate=qualification raw_primary=1 changed_option=1 wrong_order=1 wrong_dbu=1"

run_case \
  capacity capacity qualified CONTACT4_RAW_ACTIVE_HIERARCHY_CLEAN \
  "${fixture}" clean
compare_reports \
  oracle-qualified capacity CONTACT4_RAW_ACTIVE_HIERARCHY_CLEAN
capacity_log="${work}/logs/capacity-CONTACT4_RAW_ACTIVE_HIERARCHY_CLEAN.log"
assert_raw_attempt "${capacity_log}" capacity
assert_outcome \
  "${capacity_log}" "${raw_certificate_marker}" fallback capacity
capacity_line=$(certificate_line "${capacity_log}" "${raw_certificate_marker}")
[[ "${capacity_line}" =~ fallback_flags=[1-9][0-9]* ]] ||
  die "capacity: expected nonzero fail-closed flags; line='${capacity_line}'"
assert_late_certified "${capacity_log}" capacity
echo \
  "CONTACT4_RAW_ACTIVE_LIVE_GATE ok gate=raw-capacity fallback=late-merged-cert report=cpu-identical"

run_case \
  missing-backend missing qualified CONTACT4_RAW_ACTIVE_TRUE_HIT \
  "${fixture}" hit
compare_reports \
  oracle-qualified missing-backend CONTACT4_RAW_ACTIVE_TRUE_HIT
missing_log="${work}/logs/missing-backend-CONTACT4_RAW_ACTIVE_TRUE_HIT.log"
assert_no_contact4_attempt "${missing_log}" missing-backend
grep -Fq -- "unable to load CUDA spatial backend:" "${missing_log}" ||
  die "missing-backend: expected loader warning is missing"
echo \
  "CONTACT4_RAW_ACTIVE_LIVE_GATE ok gate=missing-backend report=cpu-identical"

stale_status=skipped
if [[ -n "${stale_backend}" ]]; then
  run_case \
    stale-backend stale qualified CONTACT4_RAW_ACTIVE_CLEAN \
    "${fixture}" clean
  compare_reports \
    oracle-qualified stale-backend CONTACT4_RAW_ACTIVE_CLEAN
  stale_log="${work}/logs/stale-backend-CONTACT4_RAW_ACTIVE_CLEAN.log"
  assert_no_raw_attempt "${stale_log}" stale-backend
  assert_late_certified "${stale_log}" stale-backend
  stale_status=passed
  echo \
    "CONTACT4_RAW_ACTIVE_LIVE_GATE ok gate=stale-backend raw_lowering=0 late_merged_cert=1 report=cpu-identical"
else
  echo \
    "CONTACT4_RAW_ACTIVE_LIVE_GATE SKIP gate=stale-backend reason=no---stale-backend"
fi

echo \
  "CONTACT4_RAW_ACTIVE_LIVE_GATE PASS oracles=8 cuda_qualified=4 qualification_declines=4 fail_closed=2 stale=${stale_status}"
