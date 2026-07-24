#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_m1_contact_live_gate.sh \
    --stock-klayout PATH --deck PATH \
    [--live-klayout PATH] [--backend PATH] [--python PATH] \
    [--generate-only] [--keep-work]

Builds a deterministic 15-case CONTACT/M1 fixture and an opt-in live-CUDA
variant of the supplied FreePDK45 deck.  Both implant_contact CONTACT.1-.3 and
m1_enclosure METAL1.3 owners are checked.  CUDA-off reports must remain
identical to the source deck.  Requested runs without a usable backend must
select each historical CPU rule exactly once.  With a live binary and backend,
qualified clean cases must receive the shared empty certificate, while rule
violations and unsupported domains must fail closed to stock-identical CPU
reports.

All generated layouts, decks, reports, logs, homes, and caches live under a
fresh TMPDIR directory.  They are removed unless --keep-work is specified.
EOF
}

die() {
  echo "M1-contact live gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/m1_contact_live_fixture.rb"
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
  [[ -n "${live_klayout}" ]] || die "--backend requires --live-klayout"
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

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-m1-contact-live-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "M1_CONTACT_LIVE_GATE work=${work}"
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
    -u KLAYOUT_CUDA_M1_CONTACT
    -u KLAYOUT_CUDA_M1_CONTACT_TELEMETRY
    -u KLAYOUT_CUDA_M1_CONTACT_MAX_CONTEXTS
    -u KLAYOUT_CUDA_M1_CONTACT_MAX_GRID_CELLS
    -u KLAYOUT_CUDA_M1_CONTACT_MAX_METAL_MEMBERSHIPS
    -u KLAYOUT_CUDA_M1_CONTACT_MAX_CUT_MEMBERSHIPS
    -u KLAYOUT_CUDA_M1_CONTACT_MAX_PAIR_WORK
    QT_QPA_PLATFORM=offscreen
    HOME="${runtime}/home"
    XDG_CONFIG_HOME="${runtime}/config"
    XDG_CACHE_HOME="${runtime}/cache"
  )
  if [[ "${mode}" == requested ]]; then
    command+=(KLAYOUT_CUDA_M1_CONTACT=1)
  elif [[ "${mode}" == cuda ]]; then
    command+=(
      KLAYOUT_CUDA_SPATIAL_BACKEND="${backend}"
      KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
      KLAYOUT_CUDA_VIA1_STACK_TELEMETRY=1
      KLAYOUT_CUDA_M1_CONTACT=1
      KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1
    )
  elif [[ "${mode}" != off ]]; then
    die "internal error: unknown KLayout mode ${mode}"
  fi
  command+=("${binary}" "$@")
  "${command[@]}"
}

fixture="${work}/m1-contact-live-fixture.gds"
fixture_log="${work}/logs/fixture.log"
if ! run_klayout off fixture "${stock_klayout}" -b \
  -r "${fixture_script}" -rd "output=${fixture}" \
  >"${fixture_log}" 2>&1; then
  cat -- "${fixture_log}" >&2
  die "fixture generation failed"
fi
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
grep -Fq -- \
  "M1_CONTACT_LIVE_FIXTURE ok path=${fixture} tops=15" "${fixture_log}" ||
  die "fixture completion marker or top count is wrong"
echo "M1_CONTACT_LIVE_GATE ok gate=fixture tops=15"

live_deck="${work}/m1-contact-live.lydrc"
repeat_deck="${work}/m1-contact-live-repeat.lydrc"
generator_log="${work}/logs/deck-generator.log"
"${python}" "${deck_generator}" \
  --input "${source_deck}" --output "${live_deck}" --m1-contact \
  >"${generator_log}" 2>&1 ||
  {
    cat -- "${generator_log}" >&2
    die "live deck generation failed"
  }
"${python}" "${deck_generator}" \
  --input "${source_deck}" --output "${repeat_deck}" --m1-contact \
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
  'm1_contact_request = ENV["KLAYOUT_CUDA_M1_CONTACT"].to_s' \
  "Ruby opt-in source"
assert_deck_count 1 \
  "cont.respond_to?(:cuda_m1_contact_clean?)" \
  "stock compatibility guard"
assert_deck_count 1 \
  "CUDA M1 contact transaction:" \
  "transaction telemetry"
assert_deck_count 1 \
  "m1_contact_owner = m1_contact_requested" \
  "shared owner gate"
assert_deck_count 1 \
  'm1_contact_empty.output("CONTACT.1"' \
  "CONTACT.1 certified-empty output"
assert_deck_count 1 \
  'm1_contact_empty.output("CONTACT.2"' \
  "CONTACT.2 certified-empty output"
assert_deck_count 1 \
  'm1_contact_empty.output("CONTACT.3"' \
  "CONTACT.3 certified-empty output"
assert_deck_count 1 \
  'm1_contact_empty.output("METAL1.3"' \
  "METAL1.3 certified-empty output"
m1_clean_guard_count=$(
  grep -Ec -- '^if m1_contact_clean$' "${live_deck}" || true
)
[[ "${m1_clean_guard_count}" == 2 ]] ||
  die "M1 certificate guards: expected 2, found ${m1_clean_guard_count}"

# The transform itself is the exact rule matcher.  A changed CONTACT.2
# distance must be rejected instead of being silently accelerated.
mutated_source="${work}/m1-contact-mutated-source.lydrc"
mutated_output="${work}/m1-contact-mutated-output.lydrc"
mutation_log="${work}/logs/deck-mutation.log"
sed \
  's/cont\.space(75\.nm, euclidian)\.output("CONTACT\.2"/cont.space(76.nm, euclidian).output("CONTACT.2"/' \
  "${source_deck}" >"${mutated_source}"
cmp -s -- "${source_deck}" "${mutated_source}" &&
  die "CONTACT.2 mutation did not change the source deck"
if "${python}" "${deck_generator}" \
  --input "${mutated_source}" --output "${mutated_output}" --m1-contact \
  >"${mutation_log}" 2>&1; then
  die "changed CONTACT.2 rule was incorrectly accepted"
fi
grep -Fq -- \
  "CONTACT.1-.3 transaction: expected one source block, found 0" \
  "${mutation_log}" ||
  {
    cat -- "${mutation_log}" >&2
    die "changed CONTACT.2 rule did not fail at the exact matcher"
  }
echo \
  "M1_CONTACT_LIVE_GATE ok gate=deck deterministic=1 transaction=1 certificate_outputs=4 exact_matcher=1"

if ((generate_only)); then
  echo "M1_CONTACT_LIVE_GATE ok gate=generate-only"
  exit 0
fi

cases=(
  M1_CONTACT_CLEAN
  M1_CONTACT_ENCLOSURE_70
  M1_CONTACT_ONE_DEFICIENT_SIDE
  M1_CONTACT_OPPOSITE_DEFICIENT
  M1_CONTACT_SPACING_150
  M1_CONTACT_DUPLICATE
  M1_CONTACT_HIERARCHY
  M1_CONTACT_ADJACENT_DEFICIENT
  M1_CONTACT_OUTSIDE_M1
  M1_CONTACT_BAD_SIZE
  M1_CONTACT_SPACING_149
  M1_CONTACT_OVERLAP
  M1_CONTACT_TOUCH
  M1_CONTACT_NONRECT
  M1_CONTACT_SPLIT_M1
)

declare -A expected_error=(
  [M1_CONTACT_ADJACENT_DEFICIENT]=1
)

declare -A expected_contact1=(
  [M1_CONTACT_BAD_SIZE]=2
  [M1_CONTACT_OVERLAP]=2
  [M1_CONTACT_TOUCH]=2
  [M1_CONTACT_NONRECT]=6
)

declare -A expected_contact2=(
  [M1_CONTACT_SPACING_149]=1
)

declare -A expected_contact3=(
  [M1_CONTACT_OUTSIDE_M1]=1
)

declare -A cuda_clean=(
  [M1_CONTACT_CLEAN]=1
  [M1_CONTACT_ENCLOSURE_70]=1
  [M1_CONTACT_ONE_DEFICIENT_SIDE]=1
  [M1_CONTACT_OPPOSITE_DEFICIENT]=1
  [M1_CONTACT_SPACING_150]=1
  [M1_CONTACT_DUPLICATE]=1
  [M1_CONTACT_HIERARCHY]=1
  [M1_CONTACT_SPLIT_M1]=1
)

canonicalize_report() {
  local report=$1
  local output=$2
  sed '/<generator>/d' "${report}" >"${output}"
}

assert_report() {
  local report=$1
  local top=$2
  local owner=$3
  local count
  grep -Fq -- "<top-cell>${top}</top-cell>" "${report}" ||
    die "${top}: report top-cell marker is missing"

  if [[ "${owner}" == m1_enclosure ]]; then
    count=$(
      grep -Fc -- "<category>'METAL1.3'</category>" "${report}" || true
    )
    if [[ -n "${expected_error[${top}]:-}" ]]; then
      ((count > 0)) || die "${owner}/${top}: expected METAL1.3, but found no item"
    else
      ((count == 0)) ||
        die "${owner}/${top}: unexpected METAL1.3 item count ${count}"
    fi
  elif [[ "${owner}" == implant_contact ]]; then
    local expected
    for rule in 1 2 3; do
      count=$(
        grep -Fc -- "<category>'CONTACT.${rule}'</category>" \
          "${report}" || true
      )
      case "${rule}" in
        1) expected=${expected_contact1[${top}]:-0} ;;
        2) expected=${expected_contact2[${top}]:-0} ;;
        3) expected=${expected_contact3[${top}]:-0} ;;
      esac
      [[ "${count}" == "${expected}" ]] ||
        die "${owner}/${top}: CONTACT.${rule} expected ${expected}, found ${count}"
    done
  else
    die "internal error: unknown owner ${owner}"
  fi
}

assert_transaction() {
  local mode=$1
  local top=$2
  local log=$3
  local count
  count=$(grep -Fc -- "CUDA M1 contact transaction:" "${log}" || true)
  if [[ "${mode}" == off ]]; then
    ((count == 0)) ||
      die "${top}: CUDA-off path emitted M1-contact transaction telemetry"
    return
  fi
  [[ "${count}" == 1 ]] ||
    die "${top}: expected one M1-contact transaction, found ${count}"
  if [[ "${mode}" == cuda && -n "${cuda_clean[${top}]:-}" ]]; then
    grep -Fq -- \
      "CUDA M1 contact transaction: certified-empty" "${log}" ||
      die "${top}: expected a certified-empty transaction"
  else
    grep -Fq -- \
      "CUDA M1 contact transaction: full-cpu-fallback" "${log}" ||
      die "${top}: expected a full CPU fallback transaction"
  fi
  if [[ "${mode}" == cuda ]]; then
    count=$(
      grep -Fc -- "CUDA M1 contact live lowering:" "${log}" || true
    )
    [[ "${count}" == 1 ]] ||
      die "${top}: expected one M1-contact live-lowering line, found ${count}"
  fi
}

run_case() {
  local lane=$1
  local binary=$2
  local mode=$3
  local deck=$4
  local top=$5
  local owner=$6
  local lane_dir="${work}/reports/${lane}/${owner}"
  local report="${lane_dir}/${top}.lyrdb"
  local log="${work}/logs/${lane}-${owner}-${top}.log"

  mkdir -p -- "${lane_dir}"
  if ! run_klayout "${mode}" "${lane}-${owner}-${top}" "${binary}" -b \
    -r "${deck}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${report}" \
    -rd "drc_shard=${owner}" \
    >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}/${owner}/${top}: DRC invocation failed"
  fi
  [[ -s "${report}" ]] ||
    die "${lane}/${owner}/${top}: DRC produced no report"
  assert_report "${report}" "${top}" "${owner}"
  assert_transaction "${mode}" "${top}" "${log}"
}

compare_reports() {
  local reference_lane=$1
  local candidate_lane=$2
  local top=$3
  local owner=$4
  local reference="${work}/reports/${reference_lane}/${owner}/${top}.lyrdb"
  local candidate="${work}/reports/${candidate_lane}/${owner}/${top}.lyrdb"
  local reference_canonical="${work}/reports/${reference_lane}/${owner}/${top}.canonical.lyrdb"
  local candidate_canonical="${work}/reports/${candidate_lane}/${owner}/${top}.canonical.lyrdb"
  canonicalize_report "${reference}" "${reference_canonical}"
  canonicalize_report "${candidate}" "${candidate_canonical}"
  cmp -s -- "${reference_canonical}" "${candidate_canonical}" ||
    die "${candidate_lane}/${owner}/${top}: report differs from ${reference_lane}"
}

# Source and generated decks must be observably identical with opt-in off for
# both consumers of the shared certificate.
owners=(implant_contact m1_enclosure)
for owner in "${owners[@]}"; do
  for top in "${cases[@]}"; do
    run_case source-off "${stock_klayout}" off \
      "${source_deck}" "${top}" "${owner}"
    run_case generated-off "${stock_klayout}" off \
      "${live_deck}" "${top}" "${owner}"
    compare_reports source-off generated-off "${top}" "${owner}"
  done
done
echo \
  "M1_CONTACT_LIVE_GATE ok gate=default-off owners=${#owners[@]} cases=${#cases[@]} report=source-identical"

# A binary without the optional method is the stock compatibility oracle.
for owner in "${owners[@]}"; do
  for top in "${cases[@]}"; do
    run_case stock-requested "${stock_klayout}" requested \
      "${live_deck}" "${top}" "${owner}"
    compare_reports source-off stock-requested "${top}" "${owner}"
  done
done
echo \
  "M1_CONTACT_LIVE_GATE ok gate=stock-requested-fallback owners=${#owners[@]} cases=${#cases[@]} report=source-identical"

if [[ -n "${live_klayout}" ]]; then
  for owner in "${owners[@]}"; do
    for top in "${cases[@]}"; do
      run_case live-fallback "${live_klayout}" requested \
        "${live_deck}" "${top}" "${owner}"
      compare_reports source-off live-fallback "${top}" "${owner}"
    done
  done
  echo \
    "M1_CONTACT_LIVE_GATE ok gate=live-requested-fallback owners=${#owners[@]} cases=${#cases[@]} report=source-identical"
fi

if [[ -n "${backend}" ]]; then
  for owner in "${owners[@]}"; do
    for top in "${cases[@]}"; do
      run_case cuda "${live_klayout}" cuda \
        "${live_deck}" "${top}" "${owner}"
      compare_reports source-off cuda "${top}" "${owner}"
      if [[ -n "${cuda_clean[${top}]:-}" ]]; then
        disposition=certified-empty
      else
        disposition=full-cpu-fallback
      fi
      echo \
        "M1_CONTACT_LIVE_GATE ok lane=cuda owner=${owner} case=${top} disposition=${disposition} report=source-identical"
    done
  done
  echo \
    "M1_CONTACT_LIVE_GATE ok gate=cuda owners=${#owners[@]} clean_per_owner=${#cuda_clean[@]} fallback_per_owner=$((${#cases[@]} - ${#cuda_clean[@]})) report=source-identical"
fi

# One source oracle plus generated-off and stock-requested candidates.
lanes=3
[[ -z "${live_klayout}" ]] || ((lanes += 1))
[[ -z "${backend}" ]] || ((lanes += 1))
echo \
  "M1_CONTACT_LIVE_GATE PASS owners=${#owners[@]} cases=${#cases[@]} lanes=${lanes} comparisons=$(((${lanes} - 1) * ${#owners[@]} * ${#cases[@]}))"
