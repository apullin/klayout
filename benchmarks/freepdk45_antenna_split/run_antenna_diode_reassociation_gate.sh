#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_antenna_diode_reassociation_gate.sh \
    --klayout PATH [--keep-work]

Generates a deterministic hierarchical FreePDK45 NPLUS/ACTIVE/NWELL fixture
and proves, by empty KLayout symmetric differences, that:

  nplus & active - nwell
    == nplus & (active - nwell)
    == (nplus & active) - nwell

The fixture includes a nonempty partially cut diode, empty and disjoint
operands, cross-level parent/child overlap, all eight orthogonal transforms,
and a 3-by-2 cell array.
EOF
}

die() {
  echo "antenna diode reassociation gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/antenna_diode_reassociation_fixture.rb"
gate_script="${here}/antenna_diode_reassociation_gate.rb"

klayout=
keep_work=0
while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
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
[[ -f "${fixture_script}" ]] || die "fixture generator is missing"
[[ -f "${gate_script}" ]] || die "gate script is missing"

klayout=$(readlink -f -- "${klayout}")
klayout_dir=$(dirname -- "${klayout}")
work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-antenna-diode-reassociation.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "ANTENNA_DIODE_REASSOCIATION_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- "${work}/home" "${work}/config" "${work}/cache"
common_env=(
  QT_QPA_PLATFORM=offscreen
  HOME="${work}/home"
  XDG_CONFIG_HOME="${work}/config"
  XDG_CACHE_HOME="${work}/cache"
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)

fixture="${work}/antenna-diode-reassociation.gds"
fixture_log="${work}/fixture.log"
gate_log="${work}/gate.log"

env "${common_env[@]}" "${klayout}" -b \
  -r "${fixture_script}" -rd "output=${fixture}" 2>&1 |
  tee "${fixture_log}"
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
grep -Fq -- "ANTENNA_DIODE_REASSOCIATION_FIXTURE ok" "${fixture_log}" ||
  die "fixture completion marker is missing"

env "${common_env[@]}" "${klayout}" -b \
  -r "${gate_script}" -rd "input=${fixture}" 2>&1 |
  tee "${gate_log}"
grep -Fq -- "ANTENNA_DIODE_REASSOCIATION_GATE PASS" "${gate_log}" ||
  die "equivalence completion marker is missing"
