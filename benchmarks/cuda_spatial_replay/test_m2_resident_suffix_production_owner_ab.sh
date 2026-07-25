#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gate="${here}/run_m2_resident_suffix_production_owner_ab.sh"

fail() {
  echo "M2 resident suffix owner A/B static test: $*" >&2
  exit 2
}

bash -n "${gate}" ||
  fail "gate does not pass bash syntax validation"

help=$(bash "${gate}" --help 2>&1)
for marker in \
  "previous accelerated implementation" \
  "percent less time" \
  "freshly generated deck" \
  "idle-GPU census" \
  "loader closure" \
  "evidence directory is" \
  "always retained"; do
  [[ "${help}" == *"${marker}"* ]] ||
    fail "help is missing contract marker: ${marker}"
done

expected_three=$'1\tprevious-1\tprevious\t1\n2\tresident-1\tresident\t1\n3\tresident-2\tresident\t2\n4\tprevious-2\tprevious\t2\n5\tprevious-3\tprevious\t3\n6\tresident-3\tresident\t3'
actual_three=$(bash "${gate}" --print-order --repetitions 3)
[[ "${actual_three}" == "${expected_three}" ]] ||
  fail "default N=3 order is not alternating previous/resident"

expected_four=$'1\tprevious-1\tprevious\t1\n2\tresident-1\tresident\t1\n3\tresident-2\tresident\t2\n4\tprevious-2\tprevious\t2\n5\tprevious-3\tprevious\t3\n6\tresident-3\tresident\t3\n7\tresident-4\tresident\t4\n8\tprevious-4\tprevious\t4'
actual_four=$(bash "${gate}" --print-order --repetitions 4)
[[ "${actual_four}" == "${expected_four}" ]] ||
  fail "configurable N=4 order is not alternating previous/resident"

if bash "${gate}" --print-order --repetitions 0 >/dev/null 2>&1; then
  fail "zero repetitions were incorrectly accepted"
fi

for marker in \
  "previous_klayout_sha256=6147d349166c38dcdba514ebbe23f6ada9a94f1861ebfeff6f38aaf9701ccf9f" \
  "previous_backend_sha256=f764cd4587f71e98cfc540f496aba0096c88991b1c3e1d18376a6227b3961c7a" \
  "previous_deck_sha256=f68ac26ca91bb4c2a8225ce71d93a34b4d8a61d303b96748a1b67f4facc360d3" \
  'run_generator "${current_deck}" --m2-rules' \
  'capture_runtime_closure "${lane}-before" "${mode}"' \
  'capture_runtime_closure "${lane}-after" "${mode}"' \
  'assert_gpu_idle "${lane}-before"' \
  'assert_gpu_idle "${lane}-after"' \
  'less_pct=%.3f'; do
  grep -Fq -- "${marker}" "${gate}" ||
    fail "gate is missing static integrity marker: ${marker}"
done

echo \
  "M2_RESIDENT_SUFFIX_PRODUCTION_OWNER_AB_STATIC_TEST PASS bash-syntax=1 order-N3=exact order-N4=exact legacy-pins=3 deterministic-current-deck=1 runtime-closure=before+after gpu-idle=before+after"
