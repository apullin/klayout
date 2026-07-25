#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_m2_rules_live_gate.sh \
    --stock-klayout PATH --deck PATH \
    [--live-klayout PATH] [--backend PATH] [--python PATH] [--cxx PATH] \
    [--generate-only] [--keep-work]

Generates the exact opt-in M2.1/.2/.4-.9 speculative-flat deck and a bounded
physical FreePDK45 M2/VIA2 fixture. The mandatory stock-host lanes check:

  * deterministic, exact-count deck rewriting with no source-layer mutation;
  * source versus generated behavior with the runtime request disabled; and
  * complete CPU fallback when the generated method is unavailable.

An optional live host checks the same reports with the backend absent, then
builds and loads the CPU-only contract fake. Its exact deck_clean boundary
must certify an empty candidate set while preserving independent M2.3 and
VIA2.1 owners; its live_caps boundary must complete host publication and then
take the pristine CPU fallback on a speculative suffix hit. If a real backend
is also supplied, clean M2 lanes must certify empty while every
M2.1/.2/.4/.5-.9 hit must reproduce the pristine CPU report.

All generated decks, layouts, reports, logs, homes, and caches are created
under a fresh /tmp directory. --generate-only stops after deterministic deck
and fixture validation. Work is removed unless --keep-work is supplied.
EOF
}

die() {
  echo "M2 rules live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/m2_rules_live_fixture.rb"
deck_generator="${here}/make_via1_stack_live_deck.py"
fake_backend_source="${here}/m2_union_contract_fake_backend.cc"

stock_klayout=
live_klayout=
live_library_dir=
source_deck=
backend=
fake_backend=
python=${PYTHON:-python3}
cxx=${CXX:-c++}
generate_only=0
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
    --backend)
      (($# >= 2)) || die "--backend requires a value"
      backend=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
      shift 2
      ;;
    --cxx)
      (($# >= 2)) || die "--cxx requires a value"
      cxx=$2
      shift 2
      ;;
    --generate-only)
      generate_only=1
      shift
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
[[ -f "${fake_backend_source}" ]] || die "fake backend source is missing"
if [[ -n "${live_klayout}" ]]; then
  [[ -x "${live_klayout}" ]] ||
    die "live KLayout is not executable: ${live_klayout}"
fi
if [[ -n "${backend}" ]]; then
  [[ -n "${live_klayout}" ]] ||
    die "--backend requires --live-klayout"
  [[ -f "${backend}" ]] || die "CUDA backend is missing: ${backend}"
fi

python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
stock_klayout=$(readlink -f -- "${stock_klayout}")
source_deck=$(readlink -f -- "${source_deck}")
if [[ -n "${live_klayout}" ]]; then
  live_klayout=$(readlink -f -- "${live_klayout}")
  live_library_dir=$(dirname -- "${live_klayout}")
fi
if [[ -n "${backend}" ]]; then
  backend=$(readlink -f -- "${backend}")
fi

work=$(mktemp -d /tmp/klayout-m2-rules-live-gate.XXXXXX)
cleanup() {
  if ((keep_work)); then
    echo "M2_RULES_LIVE_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- \
  "${work}/decks" \
  "${work}/logs" \
  "${work}/reports" \
  "${work}/runtime"

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
    "${runtime}/data" \
    "${runtime}/tmp"

  local -a command=(
    env
    -u KLAYOUT_CUDA_SPATIAL_BACKEND
    -u KLAYOUT_CUDA_SPATIAL_DEVICE
    -u KLAYOUT_CUDA_SPATIAL_TELEMETRY
    -u KLAYOUT_CUDA_M2_UNION_FAKE_MODE
    -u KLAYOUT_CUDA_M2_RULES
    -u KLAYOUT_CUDA_M2_RULES_TELEMETRY
    -u KLAYOUT_CUDA_M2_WIDTH_SPACE
    -u KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY
    -u KLAYOUT_CUDA_VIA1_STACK
    -u KLAYOUT_CUDA_VIA1_STACK_TELEMETRY
    QT_QPA_PLATFORM=offscreen
    HOME="${runtime}/home"
    XDG_CONFIG_HOME="${runtime}/config"
    XDG_CACHE_HOME="${runtime}/cache"
    XDG_DATA_HOME="${runtime}/data"
    TMPDIR="${runtime}/tmp"
  )
  if [[ -n "${live_klayout}" && "${binary}" == "${live_klayout}" ]]; then
    # A staged executable may retain the original qmake build RUNPATH.
    # Prefer its colocated staged libraries so the requested live seam is
    # exactly the one under test.
    command+=(LD_LIBRARY_PATH="${live_library_dir}")
  fi
  if [[ "${mode}" == requested ]]; then
    command+=(
      KLAYOUT_CUDA_M2_RULES=1
      KLAYOUT_CUDA_M2_RULES_TELEMETRY=1
    )
  elif [[ "${mode}" == cuda ]]; then
    command+=(
      KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
      KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
      KLAYOUT_CUDA_M2_RULES=1
      KLAYOUT_CUDA_M2_RULES_TELEMETRY=1
    )
  elif [[ "${mode}" == fake-deck-clean ]]; then
    command+=(
      KLAYOUT_CUDA_SPATIAL_BACKEND="${fake_backend}"
      KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
      KLAYOUT_CUDA_M2_RULES=1
      KLAYOUT_CUDA_M2_RULES_TELEMETRY=1
      KLAYOUT_CUDA_M2_UNION_FAKE_MODE=deck_clean
    )
  elif [[ "${mode}" == fake-live-caps ]]; then
    command+=(
      KLAYOUT_CUDA_SPATIAL_BACKEND="${fake_backend}"
      KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
      KLAYOUT_CUDA_M2_RULES=1
      KLAYOUT_CUDA_M2_RULES_TELEMETRY=1
      KLAYOUT_CUDA_M2_UNION_FAKE_MODE=live_caps
    )
  elif [[ "${mode}" != off ]]; then
    die "internal error: unknown mode ${mode}"
  fi
  command+=("${binary}" "$@")
  "${command[@]}"
}

live_deck="${work}/decks/freepdk45-m2-rules-live.lydrc"
repeat_deck="${work}/decks/freepdk45-m2-rules-live-repeat.lydrc"
default_deck="${work}/decks/freepdk45-default-live.lydrc"
generator_log="${work}/logs/generator.log"

for output in "${live_deck}" "${repeat_deck}"; do
  if ! "${python}" "${deck_generator}" \
    --input "${source_deck}" \
    --output "${output}" \
    --m2-rules >>"${generator_log}" 2>&1; then
    cat -- "${generator_log}" >&2
    die "M2 rules deck generation failed"
  fi
done
if ! "${python}" "${deck_generator}" \
  --input "${source_deck}" \
  --output "${default_deck}" >>"${generator_log}" 2>&1; then
  cat -- "${generator_log}" >&2
  die "default deck generation failed"
fi

cmp -s -- "${live_deck}" "${repeat_deck}" ||
  die "repeat --m2-rules generation is not byte-identical"
[[ -s "${live_deck}" ]] || die "deck generator produced no M2 rules deck"
grep -Fq -- 'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s' \
  "${live_deck}" || die "generated deck is missing the runtime request"
if grep -Fq -- 'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s' \
  "${default_deck}"; then
  die "default generator output unexpectedly contains the M2 rules rewrite"
fi

assert_deck_count() {
  local expected=$1
  local pattern=$2
  local label=$3
  local actual
  actual=$(grep -Fc -- "${pattern}" "${live_deck}" || true)
  [[ "${actual}" == "${expected}" ]] ||
    die "${label}: expected ${expected} occurrences, found ${actual}"
}

assert_deck_count 1 \
  'metal2.respond_to?(:cuda_m2_flat_union)' \
  "stock compatibility guard"
assert_deck_count 1 \
  'metal2.cuda_m2_flat_union(via2)' \
  "flat operand transaction"
assert_deck_count 8 \
  'm2_rules_flat_results &lt;&lt;' \
  "exact speculative result census"
assert_deck_count 8 \
  'm2_rules_empty.output' \
  "atomic empty output census"
assert_deck_count 1 \
  'metal2_width, metal2_space = metal2.drc_batch([' \
  "pristine M2.1/.2 fallback"
assert_deck_count 1 \
  'via2_edges_with_less_enclosure = metal2.enclosing(via2, 35.nm, projection).second_edges' \
  "pristine M2.4 fallback"
assert_deck_count 1 \
  'classify_by_width(metal2, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)' \
  "pristine M2.5-.9 fallback"

for category in \
  METAL2.1 METAL2.2 METAL2.4 METAL2.5 \
  METAL2.6 METAL2.7 METAL2.8 METAL2.9; do
  assert_deck_count 2 \
    ".output(\"${category}\"" \
    "${category} atomic/fallback output"
done
assert_deck_count 3 '.output("METAL2.3"' "unchanged M2.3 ownership"
assert_deck_count 0 \
  'm2_rules_empty.output("METAL2.3"' \
  "forbidden M2.3 suppression"
for category in VIA2.1 VIA2.2 VIA2.3 VIA2.4; do
  assert_deck_count 1 \
    ".output(\"${category}\"" \
    "${category} original ownership"
done

if grep -Eq -- '(^|[^[:alnum:]_])(metal2|via2)\.(dup|flatten|forget)' \
  "${live_deck}"; then
  die "generated deck mutates or destroys a pristine M2/VIA2 operand"
fi

speculative_block="${work}/decks/m2-speculative-block.txt"
awk '
  /^m2_rules_request = ENV/ { capture = 1 }
  capture { print }
  /^info\("CUDA M2 rules transaction:/ { exit }
' "${live_deck}" >"${speculative_block}"
[[ -s "${speculative_block}" ]] ||
  die "unable to extract the speculative M2 transaction"
if grep -Fq -- '.output(' "${speculative_block}"; then
  die "speculative flat path publishes a non-atomic output"
fi
grep -Fq -- \
  'm2_rules_flat_results.length == 8 &amp;&amp; m2_rules_flat_results.all? { |result| result.is_empty? }' \
  "${speculative_block}" ||
  die "generated deck is missing the exact eight-result clean decision"
grep -Fq -- 'm2_rules_flat_temps.reverse_each do |layer|' \
  "${speculative_block}" ||
  die "generated deck is missing reverse temporary cleanup"

clean_assignment='m2_rules_clean = m2_rules_flat_results.length == 8 &amp;&amp; m2_rules_flat_results.all? { |result| result.is_empty? }'
[[ $(grep -Fc -- "${clean_assignment}" "${speculative_block}" || true) == 1 ]] ||
  die "generated deck must contain one exact eight-result decision"
[[ $(grep -Ec -- '^[[:space:]]*ensure$' "${speculative_block}" || true) == 1 ]] ||
  die "speculative transaction must contain one ensure boundary"
[[ $(grep -Fc -- "'certified-empty'" "${speculative_block}" || true) == 1 ]] ||
  die "speculative transaction must contain one post-cleanup certification"

clean_line=$(grep -nF -- "${clean_assignment}" "${speculative_block}" | cut -d: -f1)
ensure_line=$(
  grep -nE -- '^[[:space:]]*ensure$' "${speculative_block}" | cut -d: -f1
)
cleanup_line=$(
  grep -nF -- 'm2_rules_flat_temps.reverse_each do |layer|' \
    "${speculative_block}" | cut -d: -f1
)
certified_line=$(
  grep -nF -- "'certified-empty'" "${speculative_block}" | cut -d: -f1
)
((clean_line < ensure_line &&
  ensure_line < cleanup_line &&
  cleanup_line < certified_line)) ||
  die "all-eight decision, ensure cleanup, and certification are misordered"

echo \
  "M2_RULES_LIVE_GATE ok gate=deck deterministic=1 exact_results=8 evaluation=before-cleanup certification=after-cleanup atomic_outputs=8 pristine_fallbacks=8 ownership=M2.3,VIA2.1-.4"

fixture="${work}/m2-rules-live-fixture.gds"
fixture_log="${work}/logs/fixture.log"
if ! run_klayout off fixture "${stock_klayout}" -b \
  -r "${fixture_script}" \
  -rd "output=${fixture}" >"${fixture_log}" 2>&1; then
  cat -- "${fixture_log}" >&2
  die "fixture generation failed"
fi
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
grep -Fq -- \
  "M2_RULES_LIVE_FIXTURE ok path=${fixture} tops=14 dbu=0.0005 metal2=13/0 via2=14/0" \
  "${fixture_log}" ||
  die "fixture completion marker, top count, DBU, or physical layers are wrong"
grep -Fq -- \
  "deck_clean=M2[0,0;1000,1000]+VIA2[200,200;330,330] m2_3_owner=VIA1[10,10;140,140] via2_owner=VIA2.1" \
  "${fixture_log}" ||
  die "fixture is missing the exact fake-backend clean/owner geometry"
echo \
  "M2_RULES_LIVE_GATE ok gate=fixture tops=14 dbu=0.0005 metal2=13/0 via2=14/0 deck_clean=exact owners=M2.3,VIA2.1"

if ((generate_only)); then
  echo "M2_RULES_LIVE_GATE ok gate=generate-only"
  exit 0
fi

if [[ -n "${live_klayout}" ]]; then
  cxx=$(command -v -- "${cxx}") ||
    die "C++ compiler is not executable: ${cxx}"
  fake_backend_dir="${work}/fake-backend"
  fake_backend="${fake_backend_dir}/libm2_union_contract_backend_full.so"
  fake_backend_log="${work}/logs/fake-backend-build.log"
  mkdir -p -- "${fake_backend_dir}"
  if ! "${cxx}" \
    -std=c++17 -O2 -Wall -Wextra \
    -fPIC -fvisibility=hidden -shared \
    -DKLAYOUT_CUDA_SPATIAL_BACKEND_BUILD=1 \
    -DKLAYOUT_M2_UNION_FAKE_WITH_RUN=1 \
    -DKLAYOUT_M2_UNION_FAKE_WITH_RELEASE=1 \
    -I"${here}/../../src/db/db" \
    "${fake_backend_source}" \
    -o "${fake_backend}" >"${fake_backend_log}" 2>&1; then
    cat -- "${fake_backend_log}" >&2
    die "fake M2 union backend compilation failed"
  fi
  [[ -s "${fake_backend}" ]] ||
    die "fake M2 union backend compiler produced no DSO"
  echo \
    "M2_RULES_LIVE_GATE ok gate=fake-backend-build compiler=${cxx}"
fi

cases=(
  M2_RULES_CLEAN
  M2_RULES_M2_1_HIT
  M2_RULES_M2_2_HIT
  M2_RULES_M2_3_OWNER
  M2_RULES_M2_4_HIT
  M2_RULES_M2_5_HIT
  M2_RULES_M2_6_HIT
  M2_RULES_M2_7_HIT
  M2_RULES_M2_8_HIT
  M2_RULES_M2_9_HIT
  M2_RULES_VIA2_1_OWNER
  M2_RULES_VIA2_2_OWNER
  M2_RULES_VIA2_3_OWNER
  M2_RULES_VIA2_4_OWNER
)

declare -A expected_category=(
  [M2_RULES_CLEAN]=""
  [M2_RULES_M2_1_HIT]="METAL2.1"
  [M2_RULES_M2_2_HIT]="METAL2.2"
  [M2_RULES_M2_3_OWNER]="METAL2.3"
  [M2_RULES_M2_4_HIT]="METAL2.4"
  [M2_RULES_M2_5_HIT]="METAL2.5"
  [M2_RULES_M2_6_HIT]="METAL2.6"
  [M2_RULES_M2_7_HIT]="METAL2.7"
  [M2_RULES_M2_8_HIT]="METAL2.8"
  [M2_RULES_M2_9_HIT]="METAL2.9"
  [M2_RULES_VIA2_1_OWNER]="VIA2.1"
  [M2_RULES_VIA2_2_OWNER]="VIA2.2"
  [M2_RULES_VIA2_3_OWNER]="VIA2.3"
  [M2_RULES_VIA2_4_OWNER]="VIA2.4"
)

declare -A case_shard=(
  [M2_RULES_CLEAN]="m2_rules"
  [M2_RULES_M2_1_HIT]="m2_rules"
  [M2_RULES_M2_2_HIT]="m2_rules"
  [M2_RULES_M2_3_OWNER]="m2_rules"
  [M2_RULES_M2_4_HIT]="m2_rules"
  [M2_RULES_M2_5_HIT]="m2_rules"
  [M2_RULES_M2_6_HIT]="m2_rules"
  [M2_RULES_M2_7_HIT]="m2_rules"
  [M2_RULES_M2_8_HIT]="m2_rules"
  [M2_RULES_M2_9_HIT]="m2_rules"
  [M2_RULES_VIA2_1_OWNER]="all"
  [M2_RULES_VIA2_2_OWNER]="all"
  [M2_RULES_VIA2_3_OWNER]="all"
  [M2_RULES_VIA2_4_OWNER]="all"
)

controlled_categories=(
  METAL2.1 METAL2.2 METAL2.3 METAL2.4 METAL2.5
  METAL2.6 METAL2.7 METAL2.8 METAL2.9
  VIA2.1 VIA2.2 VIA2.3 VIA2.4
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
  local expected=${expected_category[${top}]}
  local category count
  grep -Fq -- "<top-cell>${top}</top-cell>" "${report}" ||
    die "${top}: report top-cell marker is missing"
  for category in "${controlled_categories[@]}"; do
    count=$(category_count "${report}" "${category}")
    if [[ -n "${expected}" && "${category}" == "${expected}" ]]; then
      ((count > 0)) ||
        die "${top}: expected ${category}, but the report has no marker"
    else
      ((count == 0)) ||
        die "${top}: unexpected ${category} marker count ${count}"
    fi
  done
}

is_atomic_hit_case() {
  case "$1" in
    M2_RULES_M2_1_HIT|M2_RULES_M2_2_HIT|M2_RULES_M2_4_HIT|\
    M2_RULES_M2_5_HIT|M2_RULES_M2_6_HIT|M2_RULES_M2_7_HIT|\
    M2_RULES_M2_8_HIT|M2_RULES_M2_9_HIT)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

assert_transaction() {
  local lane=$1
  local top=$2
  local log=$3
  local transaction_count
  transaction_count=$(
    grep -Fc -- "CUDA M2 rules transaction:" "${log}" || true
  )

  case "${lane}" in
    source-off|generated-off)
      ((transaction_count == 0)) ||
        die "${lane}/${top}: opt-in-off run emitted a transaction"
      ;;
    missing-method)
      [[ "${transaction_count}" == 1 ]] ||
        die "${lane}/${top}: expected one transaction, found ${transaction_count}"
      grep -Fq -- \
        "CUDA M2 rules transaction: full-cpu-fallback reason=method-unavailable" \
        "${log}" ||
        die "${lane}/${top}: stock host did not take missing-method fallback"
      ;;
    live-missing)
      [[ "${transaction_count}" == 1 ]] ||
        die "${lane}/${top}: expected one transaction, found ${transaction_count}"
      grep -Fq -- \
        "CUDA M2 rules transaction: full-cpu-fallback reason=operand-decline" \
        "${log}" ||
        die "${lane}/${top}: missing backend did not decline atomically"
      grep -Fq -- \
        "CUDA M2 live flat operands: disposition=capability-unavailable" \
        "${log}" ||
        die "${lane}/${top}: live host did not reject capability before lowering"
      ;;
    fake-deck-clean)
      [[ "${transaction_count}" == 1 ]] ||
        die "${lane}/${top}: expected one transaction, found ${transaction_count}"
      grep -Fq -- \
        "CUDA M2 live flat operands: disposition=complete" "${log}" ||
        die "${lane}/${top}: deck_clean fake did not publish flat operands"
      grep -Fq -- \
        "CUDA M2 rules transaction: certified-empty reason=all-clean" \
        "${log}" ||
        die "${lane}/${top}: exact clean flat operands did not certify empty"
      ;;
    fake-live-caps)
      [[ "${transaction_count}" == 1 ]] ||
        die "${lane}/${top}: expected one transaction, found ${transaction_count}"
      grep -Fq -- \
        "CUDA M2 live flat operands: disposition=complete" "${log}" ||
        die "${lane}/${top}: live_caps fake did not publish flat operands"
      grep -Fq -- \
        "CUDA M2 rules transaction: full-cpu-fallback reason=rule-hit" \
        "${log}" ||
        die "${lane}/${top}: speculative suffix hit did not select CPU fallback"
      ;;
    cuda)
      [[ "${transaction_count}" == 1 ]] ||
        die "${lane}/${top}: expected one transaction, found ${transaction_count}"
      grep -Fq -- \
        "CUDA M2 live flat operands: disposition=complete" "${log}" ||
        die "${lane}/${top}: real backend did not publish exact flat operands"
      if is_atomic_hit_case "${top}"; then
        grep -Fq -- \
          "CUDA M2 rules transaction: full-cpu-fallback reason=rule-hit" \
          "${log}" ||
          die "${lane}/${top}: flat rule hit did not select pristine fallback"
      else
        grep -Fq -- \
          "CUDA M2 rules transaction: certified-empty reason=all-clean" \
          "${log}" ||
          die "${lane}/${top}: clean atomic M2 set did not certify empty"
      fi
      ;;
    *)
      die "internal error: unknown transaction lane ${lane}"
      ;;
  esac
}

run_case() {
  local lane=$1
  local binary=$2
  local mode=$3
  local deck=$4
  local top=$5
  local shard=${case_shard[${top}]}
  local lane_dir="${work}/reports/${lane}"
  local report="${lane_dir}/${top}.lyrdb"
  local log="${work}/logs/${lane}-${top}.log"
  mkdir -p -- "${lane_dir}"

  if ! run_klayout "${mode}" "${lane}-${top}" "${binary}" -b \
    -r "${deck}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    -rd "drc_shard=${shard}" >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}/${top}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] || die "${lane}/${top}: DRC produced no report"
  assert_report "${report}" "${top}"
  assert_transaction "${lane}" "${top}" "${log}"
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
  "M2_RULES_LIVE_GATE ok gate=generated-off cases=${#cases[@]} report=source-identical"

for top in "${cases[@]}"; do
  run_case missing-method "${stock_klayout}" requested "${live_deck}" "${top}"
  compare_reports source-off missing-method "${top}"
done
echo \
  "M2_RULES_LIVE_GATE ok gate=missing-method cases=${#cases[@]} report=source-identical"

lanes=3
if [[ -n "${live_klayout}" ]]; then
  for top in "${cases[@]}"; do
    run_case live-missing "${live_klayout}" requested "${live_deck}" "${top}"
    compare_reports source-off live-missing "${top}"
  done
  lanes=$((lanes + 1))
  echo \
    "M2_RULES_LIVE_GATE ok gate=live-missing-backend cases=${#cases[@]} report=source-identical"

  fake_clean_cases=(
    M2_RULES_CLEAN
    M2_RULES_M2_3_OWNER
    M2_RULES_VIA2_1_OWNER
  )
  for top in "${fake_clean_cases[@]}"; do
    run_case \
      fake-deck-clean "${live_klayout}" fake-deck-clean "${live_deck}" "${top}"
    compare_reports source-off fake-deck-clean "${top}"
  done
  lanes=$((lanes + 1))
  echo \
    "M2_RULES_LIVE_GATE ok gate=fake-deck-clean certified_empty=${#fake_clean_cases[@]} evaluated_all8=1 cleanup_succeeded=1 preserved=M2.3,VIA2.1 report=source-identical"

  run_case \
    fake-live-caps "${live_klayout}" fake-live-caps \
    "${live_deck}" M2_RULES_CLEAN
  compare_reports source-off fake-live-caps M2_RULES_CLEAN
  lanes=$((lanes + 1))
  echo \
    "M2_RULES_LIVE_GATE ok gate=fake-live-caps complete_to_suffix_hit=1 fallback=pristine report=source-identical"
fi

if [[ -n "${backend}" ]]; then
  for top in "${cases[@]}"; do
    run_case cuda "${live_klayout}" cuda "${live_deck}" "${top}"
    compare_reports source-off cuda "${top}"
  done
  lanes=$((lanes + 1))
  echo \
    "M2_RULES_LIVE_GATE ok gate=real-backend clean_owners=6 hit_fallbacks=8 report=source-identical"
fi

echo \
  "M2_RULES_LIVE_GATE ok cases=${#cases[@]} lanes=${lanes} atomic=1 ownership=M2.3,VIA2.1-.4"
