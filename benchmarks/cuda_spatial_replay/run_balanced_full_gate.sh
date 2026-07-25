#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_balanced_full_gate.sh \
    --klayout PATH --backend PATH --source-deck PATH \
    --manifest PATH --input PATH --top-cell NAME --reference PATH \
    [--python PATH] [--timeout-seconds N] [--jobs 8|10|11] \
    [--split-upper-antenna] \
    [--without-contact4] \
    [--with-m2-rules|--without-m2-rules] \
    [--with-m2-width-space|--without-m2-width-space] \
    [--with-implant12|--without-implant12] [--keep-work]

Regenerates the qualified FreePDK45 live-CUDA deck, applies the antenna split,
coalesces CONTACT.6 into the grid owner, and runs the exact balanced full-launch
gate. With --split-upper-antenna, METAL3 and METAL4-through-METAL10 checks use
separate owners, for eleven owners total. CUDA resource limits remain fixed;
the launcher may use eight, ten, or eleven process slots.

The manifest must be bound to the generated deck and selected owner set.
Reference may be either a raw or generator-stripped XML .lyrdb report. The
merged report must match it after removing only the generator element. CUDA
certificate telemetry, the canonical report, timings, hashes, and launcher
provenance are checked before success.

--without-contact4 retains every other qualified CUDA transaction and exists
only to produce a same-binary CONTACT.4-off performance control.

--with-m2-rules and --without-m2-rules both generate the identical
fail-closed METAL2.1/.2/.4-.9 speculative-flat transaction, then toggle only
its runtime environment. The default omits that deck rewrite and preserves
the older qualified gate. The enabled M2-rules owner and the independent
M2-width/space certificate cannot run together; use --without-m2-width-space
for an explicit same-deck M2-rules candidate or control.

--with-m2-width-space and --without-m2-width-space preserve the same deck and
toggle only the separately qualified METAL2.1/.2 runtime transaction.  The
default leaves its environment unset for compatibility with historical gates.

--with-implant12 and --without-implant12 both generate the identical fused
IMPLANT.1/.2 transaction deck, then toggle only its runtime environment. The
default omits that deck rewrite and preserves the older qualified gate.

All generated decks, reports, logs, homes, and caches live under a fresh
TMPDIR directory. They are removed unless --keep-work is specified.
EOF
}

die() {
  echo "balanced full CUDA gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "${here}/../.." && pwd)
deck_generator="${here}/make_via1_stack_live_deck.py"
antenna_split="${root}/benchmarks/freepdk45_antenna_split/split_deck.py"
contact6_split="${root}/benchmarks/freepdk45_contact6_split/split_deck.py"
runner="${root}/scripts/run_parallel_drc.py"

klayout=
backend=
source_deck=
manifest=
input=
top_cell=
reference=
python=${PYTHON:-python3}
timeout_seconds=900
keep_work=0
contact4=1
implant12=-1
m2_rules=-1
m2_width_space=-1
jobs=8
split_upper_antenna=0

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
    --source-deck)
      (($# >= 2)) || die "--source-deck requires a value"
      source_deck=$2
      shift 2
      ;;
    --manifest)
      (($# >= 2)) || die "--manifest requires a value"
      manifest=$2
      shift 2
      ;;
    --input)
      (($# >= 2)) || die "--input requires a value"
      input=$2
      shift 2
      ;;
    --top-cell)
      (($# >= 2)) || die "--top-cell requires a value"
      top_cell=$2
      shift 2
      ;;
    --reference)
      (($# >= 2)) || die "--reference requires a value"
      reference=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
      shift 2
      ;;
    --timeout-seconds)
      (($# >= 2)) || die "--timeout-seconds requires a value"
      timeout_seconds=$2
      shift 2
      ;;
    --jobs)
      (($# >= 2)) || die "--jobs requires a value"
      jobs=$2
      shift 2
      ;;
    --split-upper-antenna)
      split_upper_antenna=1
      shift
      ;;
    --keep-work)
      keep_work=1
      shift
      ;;
    --without-contact4)
      contact4=0
      shift
      ;;
    --with-m2-rules)
      ((m2_rules == -1)) ||
        die "choose exactly one M2-rules runtime mode"
      m2_rules=1
      shift
      ;;
    --without-m2-rules)
      ((m2_rules == -1)) ||
        die "choose exactly one M2-rules runtime mode"
      m2_rules=0
      shift
      ;;
    --with-m2-width-space)
      m2_width_space=1
      shift
      ;;
    --without-m2-width-space)
      m2_width_space=0
      shift
      ;;
    --with-implant12)
      ((implant12 == -1)) ||
        die "choose exactly one IMPLANT.1/.2 runtime mode"
      implant12=1
      shift
      ;;
    --without-implant12)
      ((implant12 == -1)) ||
        die "choose exactly one IMPLANT.1/.2 runtime mode"
      implant12=0
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
[[ -n "${source_deck}" ]] || die "missing --source-deck"
[[ -n "${manifest}" ]] || die "missing --manifest"
[[ -n "${input}" ]] || die "missing --input"
[[ -n "${top_cell}" ]] || die "missing --top-cell"
[[ -n "${reference}" ]] || die "missing --reference"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] ||
  die "--timeout-seconds must be a positive integer"
[[ "${jobs}" == 8 || "${jobs}" == 10 || "${jobs}" == 11 ]] ||
  die "--jobs must be 8, 10, or 11"
if [[ "${jobs}" == 11 ]] && (( ! split_upper_antenna)); then
  die "--jobs 11 requires --split-upper-antenna"
fi
if ((m2_rules == 1 && m2_width_space == 1)); then
  die "--with-m2-rules is incompatible with --with-m2-width-space"
fi

[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${backend}" ]] || die "CUDA backend is missing: ${backend}"
[[ -f "${source_deck}" ]] || die "source deck is missing: ${source_deck}"
[[ -f "${manifest}" ]] || die "manifest is missing: ${manifest}"
[[ -s "${input}" ]] || die "input layout is missing or empty: ${input}"
[[ -s "${reference}" ]] ||
  die "reference report is missing or empty: ${reference}"
[[ -f "${deck_generator}" ]] || die "live deck generator is missing"
[[ -f "${antenna_split}" ]] || die "antenna split transform is missing"
[[ -f "${contact6_split}" ]] || die "CONTACT.6 split transform is missing"
[[ -f "${runner}" ]] || die "parallel DRC runner is missing"
[[ -x /usr/bin/time ]] || die "/usr/bin/time is unavailable"
[[ -x /usr/bin/timeout ]] || die "/usr/bin/timeout is unavailable"

python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
source_deck=$(readlink -f -- "${source_deck}")
manifest=$(readlink -f -- "${manifest}")
input=$(readlink -f -- "${input}")
reference=$(readlink -f -- "${reference}")

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-balanced-full-cuda-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "BALANCED_FULL_CUDA_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

deck_dir="${work}/decks"
runtime="${work}/runtime"
runtime_tmp="${runtime}/tmp"
mkdir -p -- \
  "${deck_dir}" \
  "${runtime}/home" \
  "${runtime}/klayout-home" \
  "${runtime}/xdg-config" \
  "${runtime}/xdg-cache" \
  "${runtime}/xdg-data" \
  "${runtime_tmp}"

live_deck="${deck_dir}/freepdk45-m1-contact-live.lydrc"
antenna_deck="${deck_dir}/freepdk45-m1-contact-antenna.lydrc"
balanced_deck="${deck_dir}/freepdk45-balanced-cuda.lydrc"
transform_log="${work}/deck-transform.log"

run_transform() {
  local label=$1
  shift
  if ! "$@" >>"${transform_log}" 2>&1; then
    cat -- "${transform_log}" >&2
    die "${label} failed"
  fi
}

implant12_generator_args=()
implant12_env=()
if ((implant12 >= 0)); then
  implant12_generator_args=(--implant12)
  implant12_env=(
    "KLAYOUT_CUDA_IMPLANT12=${implant12}"
    "KLAYOUT_CUDA_IMPLANT12_TELEMETRY=${implant12}"
  )
fi
m2_rules_generator_args=()
m2_rules_env=()
if ((m2_rules >= 0)); then
  m2_rules_generator_args=(--m2-rules)
  m2_rules_env=(
    "KLAYOUT_CUDA_M2_RULES=${m2_rules}"
    "KLAYOUT_CUDA_M2_RULES_TELEMETRY=${m2_rules}"
  )
fi
m2_width_space_env=()
if ((m2_width_space >= 0)); then
  m2_width_space_env=(
    "KLAYOUT_CUDA_M2_WIDTH_SPACE=${m2_width_space}"
    "KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY=${m2_width_space}"
  )
fi
run_transform "live CUDA deck generation" \
  "${python}" "${deck_generator}" \
    --input "${source_deck}" --output "${live_deck}" --m1-contact \
    "${m2_rules_generator_args[@]}" \
    "${implant12_generator_args[@]}"
antenna_split_args=()
if ((split_upper_antenna)); then
  antenna_split_args=(--split-upper)
fi
run_transform "antenna split" \
  "${python}" "${antenna_split}" \
    "${antenna_split_args[@]}" "${live_deck}" "${antenna_deck}"
run_transform "CONTACT.6 grid coalescing" \
  "${python}" "${contact6_split}" \
    --owner grid "${antenna_deck}" "${balanced_deck}"

[[ -s "${balanced_deck}" ]] || die "generated balanced deck is empty"

reference_canonical="${work}/reference.canonical.lyrdb"
sed '/<generator>/d' "${reference}" >"${reference_canonical}"
[[ -s "${reference_canonical}" ]] ||
  die "canonical reference report is empty"
grep -Fq -- "<report-database>" "${reference_canonical}" ||
  die "reference is not an XML KLayout report database"

if ((split_upper_antenna)); then
  shards=(
    m1_width_space
    implant_contact
    antenna_m4_m10
    antenna_m3
    antenna_m1_m2
    m2_rules
    m1_enclosure
    via1_upper_active12
    grid
    m1_via_class
    antenna_feol
  )
else
  shards=(
    m1_width_space
    implant_contact
    antenna_m3_m10
    antenna_m1_m2
    m2_rules
    m1_enclosure
    via1_upper_active12
    grid
    m1_via_class
    antenna_feol
  )
fi
shard_args=()
for shard in "${shards[@]}"; do
  shard_args+=(--shard "${shard}")
done

stamp=$(date -u +%Y%m%dT%H%M%SZ)
cohort="freepdk45-balanced-cuda-${stamp}"
launcher_log="${work}/launcher.log"
time_file="${work}/time.txt"
report="${work}/merged.lyrdb"
metadata="${work}/metadata.json"

set +e
/usr/bin/time \
  -f 'wall_s=%e user_s=%U system_s=%S max_rss_kib=%M exit=%x' \
  -o "${time_file}" \
  /usr/bin/timeout \
    --foreground --signal=TERM --kill-after=15s "${timeout_seconds}s" \
    env -i \
      "HOME=${runtime}/home" \
      "KLAYOUT_HOME=${runtime}/klayout-home" \
      "XDG_CONFIG_HOME=${runtime}/xdg-config" \
      "XDG_CACHE_HOME=${runtime}/xdg-cache" \
      "XDG_DATA_HOME=${runtime}/xdg-data" \
      "TMPDIR=${runtime_tmp}" \
      PATH=/usr/bin:/bin \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
      PYTHONDONTWRITEBYTECODE=1 \
      QT_QPA_PLATFORM=offscreen \
      "KLAYOUT_CUDA_SPATIAL_BACKEND=${backend}" \
      KLAYOUT_CUDA_ACTIVE3=1 \
      KLAYOUT_CUDA_ACTIVE3_TELEMETRY=1 \
      KLAYOUT_CUDA_M1_WIDTH_SPACE=1 \
      KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1 \
      "${m2_rules_env[@]}" \
      "${m2_width_space_env[@]}" \
      KLAYOUT_CUDA_VIA1_STACK=1 \
      KLAYOUT_CUDA_VIA1_STACK_TELEMETRY=1 \
      KLAYOUT_CUDA_DISCONNECTED_MERGE=1 \
      KLAYOUT_DEEP_EDGE_CERT_PROFILE=1 \
      KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=100000 \
      KLAYOUT_CUDA_SPATIAL_CELL_SIZE=512 \
      KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64 \
      KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096 \
      KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=80000000 \
      KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=100000000 \
      KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=30000000 \
      KLAYOUT_CUDA_M1_CONTACT=1 \
      KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1 \
      "KLAYOUT_CUDA_CONTACT4=${contact4}" \
      "KLAYOUT_CUDA_CONTACT4_TELEMETRY=${contact4}" \
      "${implant12_env[@]}" \
      "${python}" "${runner}" \
        --klayout "${klayout}" \
        --deck "${balanced_deck}" \
        --input "${input}" \
        --top-cell "${top_cell}" \
        --output "${report}" \
        --metadata "${metadata}" \
        --manifest "${manifest}" \
        "${shard_args[@]}" \
        --jobs "${jobs}" \
        --cohort-id "${cohort}" \
        --replicate-index 1 \
        --replicate-count 1 \
        --keep-temp \
        >"${launcher_log}" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" >"${work}/exit-status.txt"

if ((rc != 0)); then
  echo "BALANCED_FULL_CUDA_GATE failed rc=${rc} work=${work}" >&2
  cat -- "${time_file}" >&2 || true
  tail -n 200 -- "${launcher_log}" >&2 || true
  exit "${rc}"
fi

[[ -s "${report}" ]] || die "parallel launcher produced no merged report"
[[ -s "${metadata}" ]] || die "parallel launcher produced no provenance"

shopt -s nullglob
shard_dirs=("${runtime_tmp}"/klayout-parallel-drc-*)
shopt -u nullglob
((${#shard_dirs[@]} == 1)) ||
  die "expected one retained shard directory, found ${#shard_dirs[@]}"
shard_dir=${shard_dirs[0]}
[[ -d "${shard_dir}" ]] || die "retained shard artifact is not a directory"

require_telemetry() {
  local marker=$1
  local label=$2
  grep -R -Fq --include='*.log' -- "${marker}" "${shard_dir}" ||
    die "missing ${label} telemetry"
}

require_telemetry \
  "CUDA ACTIVE.3 empty certificate: outcome=certified-empty" \
  "ACTIVE.3 certified-empty"
require_telemetry \
  "CUDA M1 width/space empty certificate: outcome=certified-empty" \
  "M1 width/space certified-empty"
if ((m2_width_space == 1)); then
  require_telemetry \
    "CUDA M2 width/space empty certificate: outcome=certified-empty" \
    "M2 width/space certified-empty"
elif ((m2_width_space == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA M2 width/space" "${shard_dir}"; then
  die "M2 width/space-off control unexpectedly invoked M2 CUDA"
fi
if ((m2_rules == 1)); then
  require_telemetry \
    "CUDA M2 exact union boundary: outcome=complete" \
    "M2 exact-union complete"
  require_telemetry \
    "CUDA M2 live flat operands: disposition=complete" \
    "M2 live-flat operand publication"
  require_telemetry \
    "CUDA M2 rules transaction: certified-empty reason=all-clean" \
    "M2 rules certified-empty"
elif ((m2_rules == 0)) &&
     grep -R -Eq --include='*.log' -- \
       'CUDA M2 (exact union boundary:|live flat operands:|rules transaction:)' \
       "${shard_dir}"; then
  die "M2-rules-off control unexpectedly invoked the live M2 transaction"
fi
require_telemetry \
  "CUDA VIA1 stack empty certificate: outcome=certified-empty" \
  "VIA1-stack certified-empty"
require_telemetry \
  "CUDA M1 contact transaction: certified-empty" \
  "M1-contact certified-empty"
if ((contact4)); then
  require_telemetry \
    "CUDA CONTACT.4 empty certificate: outcome=certified-empty" \
    "CONTACT.4 certified-empty"
elif grep -R -Fq --include='*.log' -- \
  "CUDA CONTACT.4" "${shard_dir}"; then
  die "CONTACT.4-off control unexpectedly invoked CONTACT.4 CUDA"
fi
if ((implant12 == 1)); then
  require_telemetry \
    "CUDA IMPLANT.1/.2 transaction: certified-empty" \
    "IMPLANT.1/.2 certified-empty"
elif ((implant12 == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA IMPLANT.1/.2 transaction:" "${shard_dir}"; then
  die "IMPLANT.1/.2-off control unexpectedly invoked IMPLANT CUDA"
fi

shard_count=$(grep -c '^shard ' "${launcher_log}" || true)
[[ "${shard_count}" == "${#shards[@]}" ]] ||
  die "expected ${#shards[@]} shard timing lines, found ${shard_count}"

report_canonical="${work}/merged.canonical.lyrdb"
sed '/<generator>/d' "${report}" >"${report_canonical}"
if ! cmp -s -- "${reference_canonical}" "${report_canonical}"; then
  sha256sum -- "${reference_canonical}" "${report_canonical}" >&2
  die "merged report differs from the canonical reference"
fi

grep -R -nE --include='*.log' -- \
  'CUDA ACTIVE\.3 empty certificate:|CUDA ACTIVE\.3 live lowering:|CUDA M1 width/space empty certificate:|CUDA M1 width/space live lowering:|CUDA M2 width/space empty certificate:|CUDA M2 width/space live lowering:|CUDA M2 exact union boundary:|CUDA M2 live flat operands:|CUDA M2 rules transaction:|CUDA M1 contact transaction:|CUDA M1 contact live lowering:|CUDA CONTACT\.4 empty certificate:|CUDA CONTACT\.4 live lowering:|CUDA IMPLANT\.1/\.2 transaction:|CUDA IMPLANT\.1/\.2 empty certificate:|CUDA IMPLANT\.1/\.2 live lowering:|CUDA VIA1 stack transaction:|CUDA VIA1 stack empty certificate:|CUDA VIA1 stack live lowering:|KLAYOUT_DEEP_EDGE_CERT ' \
  "${shard_dir}" >"${work}/cuda-telemetry.txt"

grep -E \
  '^shard |^children:|^merge:|^child\+merge workload:|^provenance verification:|^full launcher wall ' \
  "${launcher_log}" >"${work}/launcher-summary.txt" ||
  die "launcher timing summary is missing"

sha256sum -- \
  "${klayout}" \
  "${backend}" \
  "${source_deck}" \
  "${live_deck}" \
  "${antenna_deck}" \
  "${balanced_deck}" \
  "${manifest}" \
  "${input}" \
  "${reference}" \
  "${reference_canonical}" \
  "${report}" \
  "${report_canonical}" \
  "${deck_generator}" \
  "${antenna_split}" \
  "${contact6_split}" \
  "${runner}" \
  >"${work}/pinned-artifacts.sha256"
sha256sum -- \
  "${reference_canonical}" "${report_canonical}" \
  >"${work}/canonical-report-sha256.txt"

cat -- "${time_file}"
cat -- "${work}/launcher-summary.txt"
cat -- "${work}/canonical-report-sha256.txt"
cat -- "${work}/cuda-telemetry.txt"
echo \
  "BALANCED_FULL_CUDA_GATE ok owners=${#shards[@]} jobs=${jobs} contact4=${contact4} m2_rules=${m2_rules} m2_width_space=${m2_width_space} implant12=${implant12} split_upper_antenna=${split_upper_antenna}"
