#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_poly34_production_dry_run_gate.sh \
    --klayout PATH [--build-dir DIR] [--keep-work]

Builds a two-gate layout with KLayout, runs the read-only candidate-window
census including exact projection/zero-area terminal evaluation, and requires
complete atomic coverage with no nonzero terminal marker.
EOF
}

die() {
  echo "POLY34 production dry-run gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture="${here}/poly34_production_dry_run_fixture.rb"
runner="${here}/run_poly34_production_dry_run.sh"
klayout=
build_dir=/tmp/klayout-poly34-production-dry-run-gate-build
keep_work=0
while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
      shift 2
      ;;
    --build-dir)
      (($# >= 2)) || die "--build-dir requires a value"
      build_dir=$2
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
[[ -f "${fixture}" ]] || die "fixture is absent: ${fixture}"
[[ -f "${runner}" ]] || die "runner is absent: ${runner}"

klayout=$(readlink -f -- "${klayout}")
klayout_dir=$(dirname -- "${klayout}")
work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-poly34-production-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "POLY34_PRODUCTION_DRY_RUN_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT
mkdir -p -- "${work}/home" "${work}/config" "${work}/cache"

input="${work}/fixture.gds"
env \
  HOME="${work}/home" \
  XDG_CONFIG_HOME="${work}/config" \
  XDG_CACHE_HOME="${work}/cache" \
  QT_QPA_PLATFORM=offscreen \
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${klayout}" -b -r "${fixture}" -rd "output=${input}" \
  >"${work}/fixture.log" 2>&1
grep -Fq 'POLY34_PRODUCTION_DRY_RUN_FIXTURE ok' "${work}/fixture.log" ||
  die "fixture completion marker is missing"

env OMP_NUM_THREADS=2 \
  bash "${runner}" \
    --input "${input}" \
    --top POLY34_PRODUCTION_DRY_RUN \
    --build-dir "${build_dir}" \
    --klayout-lib-dir "${klayout_dir}" \
    --klayout-gds-plugin-dir "${klayout_dir}/db_plugins" \
    >"${work}/dry-run.log" 2>&1

grep -Fq \
  'POLY34_CANDIDATE_WINDOWS profile=poly3 gates=2 raw_candidates=2 maximum=1 ' \
  "${work}/dry-run.log" ||
  die "POLY.3 candidate census is wrong"
grep -Fq \
  'POLY34_CANDIDATE_WINDOWS profile=poly4 gates=2 raw_candidates=2 maximum=1 ' \
  "${work}/dry-run.log" ||
  die "POLY.4 candidate census is wrong"
grep -Fq \
  'POLY34_ATOMIC_COVERAGE gates=2 terminal_empty=2 fallback_or_unsupported=0' \
  "${work}/dry-run.log" ||
  die "atomic certificate coverage is incomplete"
test "$(grep -Fc ' nonzero_area_polygons=0 ' "${work}/dry-run.log")" -eq 2 ||
  die "an exact terminal profile retained a nonzero marker"
grep -Fq 'POLY34_PRODUCTION_DRY_RUN verdict=GO ' "${work}/dry-run.log" ||
  die "dry run did not issue GO"

bad_dbu_input="${work}/bad-dbu.gds"
env \
  HOME="${work}/home" \
  XDG_CONFIG_HOME="${work}/config" \
  XDG_CACHE_HOME="${work}/cache" \
  QT_QPA_PLATFORM=offscreen \
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${klayout}" -b -r "${fixture}" \
    -rd "output=${bad_dbu_input}" -rd dbu=0.001 \
    >"${work}/bad-dbu-fixture.log" 2>&1
if "${build_dir}/poly34_production_dry_run" \
    "--input=${bad_dbu_input}" --top=POLY34_PRODUCTION_DRY_RUN \
    >"${work}/bad-dbu.log" 2>&1; then
  die "mismatched-DBU fixture unexpectedly passed"
fi
grep -Fq \
  'layout DBU differs from qualified 0.0005 micron profile' \
  "${work}/bad-dbu.log" ||
  die "mismatched-DBU fixture did not fail closed"

limit_input="${work}/coordinate-limit.gds"
env \
  HOME="${work}/home" \
  XDG_CONFIG_HOME="${work}/config" \
  XDG_CACHE_HOME="${work}/cache" \
  QT_QPA_PLATFORM=offscreen \
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${klayout}" -b -r "${fixture}" \
    -rd "output=${limit_input}" -rd coordinate_limit=true \
    >"${work}/coordinate-limit-fixture.log" 2>&1
if "${build_dir}/poly34_production_dry_run" \
    "--input=${limit_input}" --top=POLY34_PRODUCTION_DRY_RUN \
    --skip-exact-terminal >"${work}/coordinate-limit.log" 2>&1; then
  die "coordinate-limit fixture unexpectedly passed"
fi
grep -Fq \
  'candidate-window coordinate extent is unsafe for scanner arithmetic' \
  "${work}/coordinate-limit.log" ||
  die "coordinate-limit fixture did not fail closed"

cat "${work}/dry-run.log"
echo "POLY34_PRODUCTION_DRY_RUN_GATE ok dbu_guard=pass coordinate_guard=pass"
