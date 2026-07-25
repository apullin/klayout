#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "${here}/../.." && pwd)
klayout=${1:-klayout}
fixture=${2:-"${repo}/testdata/drc/drctest.gds"}

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s "${here}" -p 'test_*.py' -v

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-poly2-type-contract.XXXXXX")
trap 'rm -rf "${work}"' EXIT
HOME="${work}/home" KLAYOUT_HOME="${work}/klayout-home" \
  QT_QPA_PLATFORM=offscreen \
  "${klayout}" -b -r "${here}/poly2_type_contract.drc" \
  -rd "input=${fixture}" >"${work}/contract.log" 2>&1

grep -F \
  "POLY2_TYPE_CONTRACT PASS separation=EdgePairs polygons=false branch=unreachable" \
  "${work}/contract.log"
echo "POLY2_PRUNE_GATE PASS transform=fail-closed runtime-contract=exact"
