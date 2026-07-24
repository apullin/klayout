#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_implant12_live_gate.sh \
    --stock-klayout PATH --deck PATH \
    [--live-klayout PATH] [--python PATH] [--keep-work]

Builds deterministic boundary and nonempty IMPLANT.1/.2 fixtures, generates
the opt-in fail-closed transaction deck, and compares complete CPU reports:

  * the source deck with no generated transaction;
  * the generated deck with its runtime opt-in disabled;
  * the generated deck requested on a binary without the optional method;
  * optionally, a live binary whose unavailable backend must decline.

Every candidate must be byte-identical to the source report after removing
the report-generator line. All work remains under a fresh TMPDIR directory
and is removed unless --keep-work is supplied.
EOF
}

die() {
  echo "IMPLANT.1/.2 live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/implant12_live_fixture.rb"
deck_generator="${here}/make_via1_stack_live_deck.py"

stock_klayout=
live_klayout=
source_deck=
python=${PYTHON:-python3}
keep_work=0

while (($#)); do
  case "$1" in
    --stock-klayout)
      (($# >= 2)) || die "--stock-klayout requires a value"
      stock_klayout=$2
      shift 2
      ;;
    --live-klayout)
      (($# >= 2)) || die "--live-klayout requires a value"
      live_klayout=$2
      shift 2
      ;;
    --deck)
      (($# >= 2)) || die "--deck requires a value"
      source_deck=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
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

[[ -n "${stock_klayout}" ]] || die "missing --stock-klayout"
[[ -n "${source_deck}" ]] || die "missing --deck"
[[ -x "${stock_klayout}" ]] ||
  die "stock KLayout is not executable: ${stock_klayout}"
[[ -f "${source_deck}" ]] || die "source deck is missing: ${source_deck}"
[[ -f "${fixture_script}" ]] || die "fixture generator is missing"
[[ -f "${deck_generator}" ]] || die "deck generator is missing"
if [[ -n "${live_klayout}" ]]; then
  [[ -x "${live_klayout}" ]] ||
    die "live KLayout is not executable: ${live_klayout}"
fi
python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
stock_klayout=$(readlink -f -- "${stock_klayout}")
source_deck=$(readlink -f -- "${source_deck}")
if [[ -n "${live_klayout}" ]]; then
  live_klayout=$(readlink -f -- "${live_klayout}")
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-implant12-live-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "IMPLANT12_LIVE_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- "${work}/reports" "${work}/logs" "${work}/runtime"

run_klayout() {
  local mode=$1
  local lane=$2
  local binary=$3
  shift 3
  local runtime="${work}/runtime/${lane}"
  mkdir -p -- \
    "${runtime}/home" \
    "${runtime}/config" \
    "${runtime}/cache" \
    "${runtime}/data"

  local -a command=(
    env
    -u KLAYOUT_CUDA_SPATIAL_BACKEND
    -u KLAYOUT_CUDA_SPATIAL_DEVICE
    -u KLAYOUT_CUDA_SPATIAL_TELEMETRY
    -u KLAYOUT_CUDA_IMPLANT12
    -u KLAYOUT_CUDA_IMPLANT12_TELEMETRY
    QT_QPA_PLATFORM=offscreen
    HOME="${runtime}/home"
    XDG_CONFIG_HOME="${runtime}/config"
    XDG_CACHE_HOME="${runtime}/cache"
    XDG_DATA_HOME="${runtime}/data"
  )
  if [[ "${mode}" == requested ]]; then
    command+=(KLAYOUT_CUDA_IMPLANT12=1)
  elif [[ "${mode}" != off ]]; then
    die "internal error: unknown mode ${mode}"
  fi
  command+=("${binary}" "$@")
  "${command[@]}"
}

live_deck="${work}/freepdk45-implant12-live.lydrc"
generator_log="${work}/logs/generator.log"
if ! "${python}" "${deck_generator}" \
  --input "${source_deck}" \
  --output "${live_deck}" \
  --implant12 >"${generator_log}" 2>&1; then
  cat -- "${generator_log}" >&2
  die "live deck generation failed"
fi
[[ -s "${live_deck}" ]] || die "deck generator produced no output"
grep -Fq -- \
  'implant.cuda_implant12_clean?(gate, cont)' "${live_deck}" ||
  die "generated deck is missing the exact IMPLANT.1/.2 receiver"
[[ "$(grep -Fc -- \
  'implant.separation(gate, 70.nm, projection).polygons.without_area(0).output("IMPLANT.1"' \
  "${live_deck}" || true)" == 1 ]] ||
  die "generated deck did not retain exactly one historical IMPLANT.1 fallback"
[[ "$(grep -Fc -- \
  'implant.separation(cont, 25.nm, projection).polygons.without_area(0).output("IMPLANT.2"' \
  "${live_deck}" || true)" == 1 ]] ||
  die "generated deck did not retain exactly one historical IMPLANT.2 fallback"
echo "IMPLANT12_LIVE_GATE ok gate=generator exact_rewrite=1"

fixture="${work}/implant12-live-fixture.gds"
fixture_log="${work}/logs/fixture.log"
if ! run_klayout off fixture "${stock_klayout}" -b -r "${fixture_script}" \
  -rd "output=${fixture}" >"${fixture_log}" 2>&1; then
  cat -- "${fixture_log}" >&2
  die "fixture generation failed"
fi
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
grep -Fq -- \
  "IMPLANT12_LIVE_FIXTURE ok path=${fixture} tops=4" "${fixture_log}" ||
  die "fixture completion marker or top count is wrong"
echo "IMPLANT12_LIVE_GATE ok gate=fixture tops=4 hierarchy=1"

cases=(
  IMPLANT12_BOUNDARY
  IMPLANT12_IMPLANT1_HIT
  IMPLANT12_IMPLANT2_HIT
  IMPLANT12_BOTH_HIT
)

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

assert_report() {
  local report=$1
  local top=$2
  local implant1_count
  local implant2_count
  grep -Fq -- "<top-cell>${top}</top-cell>" "${report}" ||
    die "${top}: report top-cell marker is missing"
  implant1_count=$(category_count "${report}" IMPLANT.1)
  implant2_count=$(category_count "${report}" IMPLANT.2)
  case "${top}" in
    IMPLANT12_BOUNDARY)
      ((implant1_count == 0 && implant2_count == 0)) ||
        die "${top}: expected no markers, found IMPLANT.1=${implant1_count} IMPLANT.2=${implant2_count}"
      ;;
    IMPLANT12_IMPLANT1_HIT)
      ((implant1_count > 0 && implant2_count == 0)) ||
        die "${top}: expected IMPLANT.1-only markers, found IMPLANT.1=${implant1_count} IMPLANT.2=${implant2_count}"
      ;;
    IMPLANT12_IMPLANT2_HIT)
      ((implant1_count == 0 && implant2_count > 0)) ||
        die "${top}: expected IMPLANT.2-only markers, found IMPLANT.1=${implant1_count} IMPLANT.2=${implant2_count}"
      ;;
    IMPLANT12_BOTH_HIT)
      ((implant1_count > 0 && implant2_count > 0)) ||
        die "${top}: expected both marker classes, found IMPLANT.1=${implant1_count} IMPLANT.2=${implant2_count}"
      ;;
    *)
      die "internal error: unknown fixture ${top}"
      ;;
  esac
}

assert_transaction() {
  local mode=$1
  local top=$2
  local log=$3
  local count
  count=$(
    grep -Fc -- "CUDA IMPLANT.1/.2 transaction:" "${log}" || true
  )
  if [[ "${mode}" == off ]]; then
    ((count == 0)) ||
      die "${top}: opt-in-off path emitted transaction telemetry"
  else
    [[ "${count}" == 1 ]] ||
      die "${top}: expected one transaction line, found ${count}"
    grep -Fq -- \
      "CUDA IMPLANT.1/.2 transaction: full-cpu-fallback" "${log}" ||
      die "${top}: requested transaction did not fail closed"
  fi
}

run_case() {
  local lane=$1
  local binary=$2
  local mode=$3
  local deck=$4
  local top=$5
  local lane_dir="${work}/reports/${lane}"
  local report="${lane_dir}/${top}.lyrdb"
  local log="${work}/logs/${lane}-${top}.log"
  mkdir -p -- "${lane_dir}"

  if ! run_klayout "${mode}" "${lane}-${top}" "${binary}" -b \
    -r "${deck}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    -rd "drc_shard=implant_contact" >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}/${top}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] || die "${lane}/${top}: DRC produced no report"
  assert_report "${report}" "${top}"
  assert_transaction "${mode}" "${top}" "${log}"
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

for top in "${cases[@]}"; do
  run_case source-off "${stock_klayout}" off "${source_deck}" "${top}"
  run_case generated-off "${stock_klayout}" off "${live_deck}" "${top}"
  compare_reports source-off generated-off "${top}"
done
echo \
  "IMPLANT12_LIVE_GATE ok gate=generated-off cases=${#cases[@]} report=source-identical"

for top in "${cases[@]}"; do
  run_case missing-method "${stock_klayout}" requested "${live_deck}" "${top}"
  compare_reports source-off missing-method "${top}"
done
echo \
  "IMPLANT12_LIVE_GATE ok gate=missing-method cases=${#cases[@]} report=source-identical"

lanes=3
if [[ -n "${live_klayout}" ]]; then
  for top in "${cases[@]}"; do
    run_case live-fallback "${live_klayout}" requested "${live_deck}" "${top}"
    compare_reports source-off live-fallback "${top}"
  done
  echo \
    "IMPLANT12_LIVE_GATE ok gate=live-fallback cases=${#cases[@]} report=source-identical"
  ((lanes += 1))
fi

echo \
  "IMPLANT12_LIVE_GATE PASS cases=${#cases[@]} lanes=${lanes} comparisons=$(((${lanes} - 1) * ${#cases[@]}))"
