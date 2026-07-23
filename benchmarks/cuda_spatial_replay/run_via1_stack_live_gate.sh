#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_via1_stack_live_gate.sh \
    --stock-klayout PATH --deck PATH \
    [--live-klayout PATH] [--backend PATH] [--python PATH] \
    [--generate-only] [--keep-work]

Builds a deterministic 17-case M1/VIA1/M2 fixture and an opt-in atomic
live-CUDA variant of the supplied FreePDK45 deck.  With opt-in off, generated
deck reports must be identical to the source deck under every original owner.
With opt-in on, the stock KLayout binary provides the local CPU-fallback oracle
for the co-located six-rule transaction.  An optional live binary must match
that oracle without a backend; an optional CUDA backend must match it while
also producing the expected transaction telemetry.

All generated layouts, decks, reports, logs, homes, and caches live under a
fresh TMPDIR directory.  They are removed unless --keep-work is specified.
--generate-only validates fixture/deck generation without running the DRC
cases, so it does not depend on a completed live backend.
EOF
}

die() {
  echo "VIA1-stack live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/via1_stack_live_fixture.rb"
deck_generator="${here}/make_via1_stack_live_deck.py"

stock_klayout=
live_klayout=
source_deck=
backend=
python=${PYTHON:-python3}
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
fi
if [[ -n "${backend}" ]]; then
  backend=$(readlink -f -- "${backend}")
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-via1-stack-live-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "VIA1_STACK_LIVE_GATE work=${work}"
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
  mkdir -p -- "${runtime}/home" "${runtime}/config" "${runtime}/cache"

  local -a command=(
    env
    -u KLAYOUT_CUDA_SPATIAL_BACKEND
    -u KLAYOUT_CUDA_SPATIAL_DEVICE
    -u KLAYOUT_CUDA_SPATIAL_TELEMETRY
    -u KLAYOUT_CUDA_VIA1_STACK
    -u KLAYOUT_CUDA_VIA1_STACK_TELEMETRY
    QT_QPA_PLATFORM=offscreen
    HOME="${runtime}/home"
    XDG_CONFIG_HOME="${runtime}/config"
    XDG_CACHE_HOME="${runtime}/cache"
  )
  if [[ "${mode}" == requested ]]; then
    command+=(
      KLAYOUT_CUDA_VIA1_STACK=1
    )
  elif [[ "${mode}" == cuda ]]; then
    command+=(
      KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
      KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
      KLAYOUT_CUDA_VIA1_STACK=1
      KLAYOUT_CUDA_VIA1_STACK_TELEMETRY=1
    )
  elif [[ "${mode}" != off ]]; then
    die "internal error: unknown KLayout mode ${mode}"
  fi
  command+=("${binary}" "$@")
  "${command[@]}"
}

fixture="${work}/via1-stack-live-fixture.gds"
fixture_log="${work}/logs/fixture.log"
if ! run_klayout off fixture "${stock_klayout}" -b \
  -r "${fixture_script}" -rd "output=${fixture}" \
  >"${fixture_log}" 2>&1; then
  cat -- "${fixture_log}" >&2
  die "fixture generation failed"
fi
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
grep -Fq -- \
  "VIA1_STACK_LIVE_FIXTURE ok path=${fixture} tops=17" "${fixture_log}" ||
  die "fixture completion marker or top count is wrong"
echo "VIA1_STACK_LIVE_GATE ok gate=fixture tops=17"

live_deck="${work}/via1-stack-live.lydrc"
repeat_deck="${work}/via1-stack-live-repeat.lydrc"
generator_log="${work}/logs/deck-generator.log"
"${python}" "${deck_generator}" \
  --input "${source_deck}" --output "${live_deck}" \
  >"${generator_log}" 2>&1 ||
  {
    cat -- "${generator_log}" >&2
    die "live deck generation failed"
  }
"${python}" "${deck_generator}" \
  --input "${source_deck}" --output "${repeat_deck}" \
  >>"${generator_log}" 2>&1 ||
  {
    cat -- "${generator_log}" >&2
    die "repeat live deck generation failed"
  }
cmp -s -- "${live_deck}" "${repeat_deck}" ||
  die "repeat live deck generation is not byte-identical"

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
  "via1_stack_request = ENV[\"KLAYOUT_CUDA_VIA1_STACK\"].to_s" \
  "Ruby opt-in source"
assert_deck_count 1 \
  "via1_stack_owner = via1_stack_requested &amp;&amp; run_via1_upper_active12" \
  "requested transaction owner"
assert_deck_count 1 \
  "via1.respond_to?(:cuda_via1_stack_clean?)" "stock compatibility guard"
off_owner_guard_count=$(
  grep -Ec -- '^unless via1_stack_requested$' "${live_deck}" || true
)
[[ "${off_owner_guard_count}" == 2 ]] ||
  die "original-owner guards: expected 2, found ${off_owner_guard_count}"
atomic_owner_guard_count=$(
  grep -Ec -- '^if via1_stack_owner$' "${live_deck}" || true
)
[[ "${atomic_owner_guard_count}" == 2 ]] ||
  die "atomic-owner guards: expected 2, found ${atomic_owner_guard_count}"
decision_guard_count=$(
  grep -Ec -- '^[[:space:]]*if via1_stack_clean$' "${live_deck}" || true
)
[[ "${decision_guard_count}" == 4 ]] ||
  die "atomic decision guards: expected 4, found ${decision_guard_count}"
assert_deck_count 6 "via1_stack_empty.output" "empty result outputs"
for category in VIA1.1 VIA1.2 VIA1.3 VIA1.4; do
  assert_deck_count 2 \
    ".output(\"${category}\"" "${category} all-or-nothing output"
done
for category in METAL1.4 METAL2.3; do
  assert_deck_count 3 \
    ".output(\"${category}\"" "${category} off/atomic ownership output"
done
echo \
  "VIA1_STACK_LIVE_GATE ok gate=deck deterministic=1 off_owner_guards=2 atomic_owner_guards=2 atomic_outputs=6"

if ((generate_only)); then
  echo "VIA1_STACK_LIVE_GATE ok gate=generate-only"
  exit 0
fi

cases=(
  VIA1_STACK_CLEAN
  VIA1_STACK_ENCLOSURE_70
  VIA1_STACK_SPACING_150
  VIA1_STACK_DIAGONAL_90_120
  VIA1_STACK_DUPLICATE
  VIA1_STACK_L_SLAB
  VIA1_STACK_PLUS_SLAB
  VIA1_STACK_HIERARCHY
  VIA1_STACK_M1_MISS
  VIA1_STACK_M2_MISS
  VIA1_STACK_OUTSIDE_M1
  VIA1_STACK_OUTSIDE_M2
  VIA1_STACK_BAD_SIZE
  VIA1_STACK_SPACING_149
  VIA1_STACK_DIAGONAL_90_119
  VIA1_STACK_TOUCH
  VIA1_STACK_NONRECT_CUT
)
categories=(METAL1.4 VIA1.1 VIA1.2 VIA1.3 VIA1.4 METAL2.3)

declare -A expected_categories=(
  [VIA1_STACK_CLEAN]=""
  [VIA1_STACK_ENCLOSURE_70]=""
  [VIA1_STACK_SPACING_150]=""
  [VIA1_STACK_DIAGONAL_90_120]=""
  [VIA1_STACK_DUPLICATE]=""
  [VIA1_STACK_L_SLAB]=""
  [VIA1_STACK_PLUS_SLAB]=""
  [VIA1_STACK_HIERARCHY]=""
  [VIA1_STACK_M1_MISS]="METAL1.4"
  [VIA1_STACK_M2_MISS]="METAL2.3"
  [VIA1_STACK_OUTSIDE_M1]="VIA1.3"
  [VIA1_STACK_OUTSIDE_M2]="VIA1.4"
  [VIA1_STACK_BAD_SIZE]="VIA1.1"
  [VIA1_STACK_SPACING_149]="VIA1.2"
  [VIA1_STACK_DIAGONAL_90_119]="VIA1.2"
  [VIA1_STACK_TOUCH]="VIA1.1"
  [VIA1_STACK_NONRECT_CUT]="VIA1.1"
)

declare -A cuda_clean=(
  [VIA1_STACK_CLEAN]=1
  [VIA1_STACK_ENCLOSURE_70]=1
  [VIA1_STACK_SPACING_150]=1
  [VIA1_STACK_DIAGONAL_90_120]=1
  [VIA1_STACK_DUPLICATE]=1
  [VIA1_STACK_L_SLAB]=1
  [VIA1_STACK_PLUS_SLAB]=1
  [VIA1_STACK_HIERARCHY]=1
)

assert_one_line() {
  local log=$1
  local pattern=$2
  local label=$3
  local count
  count=$(grep -Fc -- "${pattern}" "${log}" || true)
  [[ "${count}" == 1 ]] ||
    die "${label}: expected one '${pattern}' line, found ${count}"
}

assert_expected_categories() {
  local report=$1
  local top=$2
  local expected=",${expected_categories[${top}]},"
  local category count

  grep -Fq -- "<top-cell>${top}</top-cell>" "${report}" ||
    die "${top}: report top-cell marker is missing"
  for category in "${categories[@]}"; do
    count=$(
      grep -Fc -- "<category>'${category}'</category>" "${report}" || true
    )
    if [[ "${expected}" == *",${category},"* ]]; then
      ((count > 0)) ||
        die "${top}: expected ${category}, but the report has no item"
    else
      ((count == 0)) ||
        die "${top}: unexpected ${category} item count ${count}"
    fi
  done
}

run_case() {
  local lane=$1
  local binary=$2
  local mode=$3
  local top=$4
  local lane_dir="${work}/reports/${lane}"
  local report="${lane_dir}/${top}.lyrdb"
  local log="${work}/logs/${lane}-${top}.log"
  local transaction_count

  mkdir -p -- "${lane_dir}"
  if ! run_klayout "${mode}" "${lane}-${top}" "${binary}" -b \
    -r "${live_deck}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    -rd "drc_shard=via1_upper_active12" \
    >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}/${top}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] || die "${lane}/${top}: DRC produced no report"
  assert_expected_categories "${report}" "${top}"

  if [[ "${mode}" == off ]]; then
    transaction_count=$(
      grep -Fc -- "CUDA VIA1 stack transaction:" "${log}" || true
    )
    ((transaction_count == 0)) ||
      die "${lane}/${top}: CUDA-off path emitted transaction telemetry"
  elif [[ "${mode}" == requested ]]; then
    assert_one_line \
      "${log}" "CUDA VIA1 stack transaction:" "${lane}/${top} transaction"
    grep -Fq -- \
      "CUDA VIA1 stack transaction: full-cpu-fallback" "${log}" ||
      die "${lane}/${top}: expected the requested local CPU fallback"
  elif [[ -n "${cuda_clean[${top}]:-}" ]]; then
    assert_one_line \
      "${log}" "CUDA VIA1 stack transaction:" "${lane}/${top} transaction"
    grep -Fq -- \
      "CUDA VIA1 stack transaction: certified-empty" "${log}" ||
      die "${lane}/${top}: expected a certified-empty transaction"
    assert_one_line \
      "${log}" \
      "CUDA VIA1 stack empty certificate: outcome=certified-empty" \
      "${lane}/${top} CUDA certificate"
  else
    assert_one_line \
      "${log}" "CUDA VIA1 stack transaction:" "${lane}/${top} transaction"
    grep -Fq -- \
      "CUDA VIA1 stack transaction: full-cpu-fallback" "${log}" ||
      die "${lane}/${top}: expected fail-closed CPU fallback"
    if ! grep -Fq -- "CUDA VIA1 stack empty certificate:" "${log}" &&
       ! grep -Fq -- \
         "CUDA VIA1 stack live lowering: outcome=cpu-fallback" "${log}"; then
      die "${lane}/${top}: fallback has no backend/serializer telemetry"
    fi
  fi
}

canonicalize_report() {
  local report=$1
  local output=$2
  sed '/<generator>/d' "${report}" >"${output}"
}

compare_lane_to_stock() {
  local lane=$1
  local top=$2
  local stock_report="${work}/reports/stock/${top}.lyrdb"
  local lane_report="${work}/reports/${lane}/${top}.lyrdb"
  local stock_canonical="${work}/reports/stock/${top}.canonical.lyrdb"
  local lane_canonical="${work}/reports/${lane}/${top}.canonical.lyrdb"
  canonicalize_report "${stock_report}" "${stock_canonical}"
  canonicalize_report "${lane_report}" "${lane_canonical}"
  cmp -s -- "${stock_canonical}" "${lane_canonical}" ||
    die "${lane}/${top}: report differs from the stock CPU oracle"
}

declare -A probe_reports
declare -A probe_logs

run_ownership_probe() {
  local key=$1
  local mode=$2
  local deck=$3
  local top=$4
  local shard=$5
  local probe_dir="${work}/reports/ownership"
  local report="${probe_dir}/${key}.lyrdb"
  local log="${work}/logs/ownership-${key}.log"

  mkdir -p -- "${probe_dir}"
  if ! run_klayout "${mode}" "ownership-${key}" "${stock_klayout}" -b \
    -r "${deck}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    -rd "drc_shard=${shard}" \
    >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "ownership/${key}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] || die "ownership/${key}: DRC produced no report"
  probe_reports["${key}"]=${report}
  probe_logs["${key}"]=${log}
}

compare_ownership_probes() {
  local source_key=$1
  local generated_key=$2
  local source_canonical="${work}/reports/ownership/${source_key}.canonical.lyrdb"
  local generated_canonical="${work}/reports/ownership/${generated_key}.canonical.lyrdb"
  canonicalize_report "${probe_reports[${source_key}]}" "${source_canonical}"
  canonicalize_report "${probe_reports[${generated_key}]}" "${generated_canonical}"
  cmp -s -- "${source_canonical}" "${generated_canonical}" ||
    die "ownership/${generated_key}: CUDA-off report differs from source deck"
}

assert_probe_category() {
  local key=$1
  local category=$2
  local expected=$3
  local count
  count=$(
    grep -Fc -- \
      "<category>'${category}'</category>" "${probe_reports[${key}]}" || true
  )
  if [[ "${expected}" == present ]]; then
    ((count > 0)) ||
      die "ownership/${key}: expected ${category}, but found no item"
  elif [[ "${expected}" == absent ]]; then
    ((count == 0)) ||
      die "ownership/${key}: unexpected ${category} item count ${count}"
  else
    die "internal error: invalid ownership category expectation ${expected}"
  fi
}

assert_probe_transaction() {
  local key=$1
  local expected=$2
  local count
  count=$(
    grep -Fc -- \
      "CUDA VIA1 stack transaction:" "${probe_logs[${key}]}" || true
  )
  if [[ "${expected}" == none ]]; then
    ((count == 0)) ||
      die "ownership/${key}: CUDA-off/wrong-owner path emitted a transaction"
  else
    [[ "${count}" == 1 ]] ||
      die "ownership/${key}: expected one transaction, found ${count}"
    grep -Fq -- \
      "CUDA VIA1 stack transaction: ${expected}" "${probe_logs[${key}]}" ||
      die "ownership/${key}: expected transaction ${expected}"
  fi
}

# CUDA-off must be observably identical to the source deck under each of the
# six original rule owners, including the two rules redirected when requested.
off_tops=(
  VIA1_STACK_M1_MISS
  VIA1_STACK_BAD_SIZE
  VIA1_STACK_SPACING_149
  VIA1_STACK_OUTSIDE_M1
  VIA1_STACK_OUTSIDE_M2
  VIA1_STACK_M2_MISS
)
off_shards=(
  m1_via_class
  via1_upper_active12
  via1_upper_active12
  via1_upper_active12
  via1_upper_active12
  m2_rules
)
off_categories=(METAL1.4 VIA1.1 VIA1.2 VIA1.3 VIA1.4 METAL2.3)
for ((i = 0; i < ${#off_tops[@]}; ++i)); do
  key=${off_categories[i]//./_}
  source_key="source-off-${key}"
  generated_key="generated-off-${key}"
  run_ownership_probe \
    "${source_key}" off "${source_deck}" \
    "${off_tops[i]}" "${off_shards[i]}"
  run_ownership_probe \
    "${generated_key}" off "${live_deck}" \
    "${off_tops[i]}" "${off_shards[i]}"
  compare_ownership_probes "${source_key}" "${generated_key}"
  assert_probe_category "${generated_key}" "${off_categories[i]}" present
  assert_probe_transaction "${generated_key}" none
done

# With CUDA off, M1.4 and METAL2.3 must not leak into the atomic owner.
run_ownership_probe \
  generated-off-wrong-METAL1_4 off "${live_deck}" \
  VIA1_STACK_M1_MISS via1_upper_active12
assert_probe_category generated-off-wrong-METAL1_4 METAL1.4 absent
assert_probe_transaction generated-off-wrong-METAL1_4 none
run_ownership_probe \
  generated-off-wrong-METAL2_3 off "${live_deck}" \
  VIA1_STACK_M2_MISS via1_upper_active12
assert_probe_category generated-off-wrong-METAL2_3 METAL2.3 absent
assert_probe_transaction generated-off-wrong-METAL2_3 none
echo \
  "VIA1_STACK_LIVE_GATE ok gate=off-ownership source_identical=6 wrong_owner_absent=2"

# With CUDA requested, the historical M1/M2 owners suppress their copies and
# the atomic VIA1 owner performs a complete local CPU fallback without needing
# the optional method or backend.
run_ownership_probe \
  generated-requested-old-METAL1_4 requested "${live_deck}" \
  VIA1_STACK_M1_MISS m1_via_class
assert_probe_category generated-requested-old-METAL1_4 METAL1.4 absent
assert_probe_transaction generated-requested-old-METAL1_4 none
run_ownership_probe \
  generated-requested-old-METAL2_3 requested "${live_deck}" \
  VIA1_STACK_M2_MISS m2_rules
assert_probe_category generated-requested-old-METAL2_3 METAL2.3 absent
assert_probe_transaction generated-requested-old-METAL2_3 none
run_ownership_probe \
  generated-requested-atomic-METAL1_4 requested "${live_deck}" \
  VIA1_STACK_M1_MISS via1_upper_active12
assert_probe_category generated-requested-atomic-METAL1_4 METAL1.4 present
assert_probe_transaction \
  generated-requested-atomic-METAL1_4 full-cpu-fallback
run_ownership_probe \
  generated-requested-atomic-METAL2_3 requested "${live_deck}" \
  VIA1_STACK_M2_MISS via1_upper_active12
assert_probe_category generated-requested-atomic-METAL2_3 METAL2.3 present
assert_probe_transaction \
  generated-requested-atomic-METAL2_3 full-cpu-fallback
echo \
  "VIA1_STACK_LIVE_GATE ok gate=requested-ownership old_owner_absent=2 atomic_owner_present=2"

for top in "${cases[@]}"; do
  run_case stock "${stock_klayout}" requested "${top}"
  echo \
    "VIA1_STACK_LIVE_GATE ok lane=stock-requested-fallback case=${top} categories=${expected_categories[${top}]:-none}"
done
echo \
  "VIA1_STACK_LIVE_GATE ok gate=stock-requested-fallback-oracle cases=${#cases[@]}"

if [[ -n "${live_klayout}" ]]; then
  for top in "${cases[@]}"; do
    run_case live-fallback "${live_klayout}" requested "${top}"
    compare_lane_to_stock live-fallback "${top}"
    echo \
      "VIA1_STACK_LIVE_GATE ok lane=live-requested-fallback case=${top} report=stock-identical"
  done
  echo \
    "VIA1_STACK_LIVE_GATE ok gate=live-requested-fallback cases=${#cases[@]} report=stock-identical"
fi

if [[ -n "${backend}" ]]; then
  for top in "${cases[@]}"; do
    run_case cuda "${live_klayout}" cuda "${top}"
    compare_lane_to_stock cuda "${top}"
    if [[ -n "${cuda_clean[${top}]:-}" ]]; then
      disposition=certified-empty
    else
      disposition=full-cpu-fallback
    fi
    echo \
      "VIA1_STACK_LIVE_GATE ok lane=cuda case=${top} disposition=${disposition} report=stock-identical"
  done
  echo \
    "VIA1_STACK_LIVE_GATE ok gate=cuda clean=${#cuda_clean[@]} fallback=$((${#cases[@]} - ${#cuda_clean[@]})) report=stock-identical"
fi

lanes=1
[[ -z "${live_klayout}" ]] || ((lanes += 1))
[[ -z "${backend}" ]] || ((lanes += 1))
echo \
  "VIA1_STACK_LIVE_GATE PASS cases=${#cases[@]} lanes=${lanes} comparisons=$(((${lanes} - 1) * ${#cases[@]}))"
