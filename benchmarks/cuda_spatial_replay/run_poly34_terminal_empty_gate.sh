#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_poly34_terminal_empty_gate.sh \
    --klayout PATH [--random-count N] [--build-dir PATH] [--keep-work]

Generates deterministic boundary/adversarial cases plus seeded randomized
rectangle-union cases. A real KLayout process computes the exact POLY.3 and
POLY.4 terminal result through projection enclosing, normalized edge-pair
polygon conversion, and zero-area exclusion. The CUDA certificate must agree
with the independent conservative oracle and must produce no false-clean
outcome against KLayout.
EOF
}

die() {
  echo "POLY34 terminal-empty gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture="${here}/poly34_terminal_empty_fixture.rb"
runner="${here}/run_poly34_terminal_empty_island.sh"

klayout=
random_count=20000
requested_build_dir=
keep_work=0
while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
      shift 2
      ;;
    --random-count)
      (($# >= 2)) || die "--random-count requires a value"
      random_count=$2
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
[[ "${random_count}" =~ ^[1-9][0-9]*$ ]] ||
  die "--random-count must be a positive integer"
[[ -f "${fixture}" ]] || die "fixture generator is missing"
[[ -x "${runner}" ]] || die "island runner is not executable"

klayout=$(readlink -f -- "${klayout}")
klayout_dir=$(dirname -- "${klayout}")
work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-poly34-terminal-gate.XXXXXX")
if [[ -n "${requested_build_dir}" ]]; then
  mkdir -p -- "${requested_build_dir}"
  build_dir=$(readlink -f -- "${requested_build_dir}")
else
  build_dir="${work}/build"
fi

cleanup() {
  if ((keep_work)); then
    echo "POLY34_TERMINAL_EMPTY_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- "${work}/home" "${work}/config" "${work}/cache" "${build_dir}"
common_env=(
  QT_QPA_PLATFORM=offscreen
  HOME="${work}/home"
  XDG_CONFIG_HOME="${work}/config"
  XDG_CACHE_HOME="${work}/cache"
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)

cases="${work}/cases.poly34"
fixture_log="${work}/fixture.log"
env "${common_env[@]}" "${klayout}" -b \
  -r "${fixture}" \
  -rd "output=${cases}" \
  -rd "random_count=${random_count}" 2>&1 |
  tee "${fixture_log}"
[[ -s "${cases}" ]] || die "fixture generator produced no cases"
grep -Fq -- "POLY34_TERMINAL_EMPTY_FIXTURE ok" "${fixture_log}" ||
  die "fixture completion marker is missing"
grep -Fq -- "randomized=${random_count} " "${fixture_log}" ||
  die "fixture randomized count is wrong"
grep -Fq -- "false_clean=0" "${fixture_log}" ||
  die "independent fixture oracle found a false clean"
grep -Eq -- "zero_area_only_profiles=[1-9][0-9]*" "${fixture_log}" ||
  die "fixture did not exercise nonempty raw edge pairs culled at the terminal"

island_log="${work}/island.log"
BUILD_DIR="${build_dir}" "${runner}" "${cases}" 2>&1 | tee "${island_log}"
grep -Fq -- "POLY34_TERMINAL_EMPTY_ISLAND verdict=GO" "${island_log}" ||
  die "CUDA correctness island did not issue GO"
grep -Fq -- " randomized=${random_count} " "${island_log}" ||
  die "CUDA correctness island randomized count is wrong"
grep -Fq -- " gpu_mismatches=0 false_clean=0 " "${island_log}" ||
  die "CUDA correctness island found a mismatch or false clean"

assert_case() {
  local name=$1
  local expected=$2
  grep -Fq -- "POLY34_CASE name=${name} ${expected}" "${island_log}" ||
    die "${name}: missing expected classification: ${expected}"
}

assert_case all_coincident \
  "poly=TERMINAL_EMPTY active=TERMINAL_EMPTY atomic_terminal_empty=1 klayout_poly_empty=1 klayout_active_empty=1"
assert_case exact_profiles \
  "poly=TERMINAL_EMPTY active=TERMINAL_EMPTY atomic_terminal_empty=1 klayout_poly_empty=1 klayout_active_empty=1"
assert_case profile_split_110_140 \
  "poly=TERMINAL_EMPTY active=FALLBACK atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=0"
assert_case poly_one_dbu_partial \
  "poly=FALLBACK active=TERMINAL_EMPTY atomic_terminal_empty=0 klayout_poly_empty=0 klayout_active_empty=1"
assert_case active_one_dbu_partial \
  "poly=TERMINAL_EMPTY active=FALLBACK atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=0"
assert_case threshold_minus_one \
  "poly=FALLBACK active=FALLBACK atomic_terminal_empty=0 klayout_poly_empty=0 klayout_active_empty=0"
assert_case mixed_zero_full_declined \
  "poly=FALLBACK active=FALLBACK atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=1"
assert_case disconnected_near_declined \
  "poly=FALLBACK active=FALLBACK atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=1"
assert_case disconnected_exact_boundary \
  "poly=TERMINAL_EMPTY active=TERMINAL_EMPTY atomic_terminal_empty=1 klayout_poly_empty=1 klayout_active_empty=1"
assert_case union_tiled_full_bands \
  "poly=TERMINAL_EMPTY active=TERMINAL_EMPTY atomic_terminal_empty=1 klayout_poly_empty=1 klayout_active_empty=1"
assert_case one_dbu_band_gap_declined \
  "poly=FALLBACK active=FALLBACK atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=1"
assert_case gate_not_covered \
  "poly=FALLBACK active=FALLBACK atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=1"
assert_case unsupported_poly_shape \
  "poly=UNSUPPORTED active=TERMINAL_EMPTY atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=1"
assert_case unsupported_active_shape \
  "poly=TERMINAL_EMPTY active=UNSUPPORTED atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=1"
assert_case candidate_capacity \
  "poly=UNSUPPORTED active=UNSUPPORTED atomic_terminal_empty=0 klayout_poly_empty=1 klayout_active_empty=1"

echo \
  "POLY34_TERMINAL_EMPTY_GATE ok deterministic=15 randomized=${random_count} false_clean=0"
