#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_m2_rules_production_owner_gate.sh \
    --klayout PATH --backend PATH --device-smoke PATH --source-deck PATH \
    --default-deck-reference PATH --input PATH --top-cell NAME \
    --reference PATH [--python PATH] [--timeout-seconds N] [--keep-work]

  bash run_m2_rules_production_owner_gate.sh \
    --source-deck PATH --default-deck-reference PATH \
    [--python PATH] --generate-only [--keep-work]

Generates one deterministic FreePDK45 deck containing the atomic
METAL2.1/.2/.4-.9 transaction, then runs only drc_shard=m2_rules. The
feature-off control and live candidate use the same KLayout executable,
backend, generated deck, input, resource limits, and explicit GPU 0 binding.
They run serially, control first. Their generator-stripped reports must be
byte-identical to one another and to the required production-owner reference.

Before each owner lane, the gate rejects pre-existing CUDA compute processes,
captures and hashes the loader-selected runtime closure, verifies that the
device smoke selected the requested backend and that KLayout selected its
colocated libklayout_db and libklayout_drc, executes a no-GPU DRC-layer probe
for the actual cuda_m2_flat_union host seam, checks the two M2 backend ABI
symbols, and runs the bounded M2 production device smoke. The closure is
rehashed after each lane and must remain unchanged.

The candidate must emit exactly one complete exact-union boundary, one
complete live-flat publication with the qualified production topology
fingerprint, and one certified-empty atomic rules transaction. The control
must emit none of those markers. Both lanes must remain isolated from every
unrelated qualified owner transaction.

Generation without --m2-rules must stay byte-identical to the supplied
pre-change reference. Repeated default and opt-in generation must also be
byte-identical. --generate-only performs only these bounded generator checks.

All decks, reports, logs, homes, timings, loader closures, and hashes live
under a fresh TMPDIR directory. They are removed unless --keep-work is set.
EOF
}

die() {
  echo "M2 rules production owner gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
deck_generator="${here}/make_via1_stack_live_deck.py"
host_hook_probe="${here}/m2_rules_host_hook_probe.drc"

klayout=
backend=
device_smoke=
source_deck=
default_deck_reference=
input=
top_cell=
reference=
python=${PYTHON:-python3}
timeout_seconds=600
keep_work=0
generate_only=0

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
    --device-smoke)
      (($# >= 2)) || die "--device-smoke requires a value"
      device_smoke=$2
      shift 2
      ;;
    --source-deck)
      (($# >= 2)) || die "--source-deck requires a value"
      source_deck=$2
      shift 2
      ;;
    --default-deck-reference)
      (($# >= 2)) || die "--default-deck-reference requires a value"
      default_deck_reference=$2
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

[[ -n "${source_deck}" ]] || die "missing --source-deck"
[[ -n "${default_deck_reference}" ]] ||
  die "missing --default-deck-reference"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] ||
  die "--timeout-seconds must be a positive integer"
[[ -f "${source_deck}" ]] || die "source deck is missing: ${source_deck}"
[[ -f "${default_deck_reference}" ]] ||
  die "default deck reference is missing: ${default_deck_reference}"
[[ -f "${deck_generator}" ]] || die "deck generator is missing"
[[ -f "${host_hook_probe}" ]] || die "host hook probe is missing"

if (( ! generate_only)); then
  [[ -n "${klayout}" ]] || die "missing --klayout"
  [[ -n "${backend}" ]] || die "missing --backend"
  [[ -n "${device_smoke}" ]] || die "missing --device-smoke"
  [[ -n "${input}" ]] || die "missing --input"
  [[ -n "${top_cell}" ]] || die "missing --top-cell"
  [[ -n "${reference}" ]] || die "missing --reference"
  [[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
  [[ -f "${backend}" ]] || die "CUDA backend is missing: ${backend}"
  [[ -x "${device_smoke}" ]] ||
    die "CUDA device smoke is not executable: ${device_smoke}"
  [[ -s "${input}" ]] || die "input layout is missing or empty: ${input}"
  [[ -s "${reference}" ]] ||
    die "reference report is missing or empty: ${reference}"
  [[ -x /usr/bin/time ]] || die "/usr/bin/time is unavailable"
  [[ -x /usr/bin/timeout ]] || die "/usr/bin/timeout is unavailable"
  [[ -x /usr/bin/readelf ]] || die "/usr/bin/readelf is unavailable"
  [[ -x /usr/bin/ldd ]] || die "/usr/bin/ldd is unavailable"
  [[ -x /usr/bin/nm ]] || die "/usr/bin/nm is unavailable"
  [[ -x /usr/bin/nvidia-smi ]] || die "/usr/bin/nvidia-smi is unavailable"
fi

python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
source_deck=$(readlink -f -- "${source_deck}")
default_deck_reference=$(readlink -f -- "${default_deck_reference}")
if (( ! generate_only)); then
  klayout=$(readlink -f -- "${klayout}")
  backend=$(readlink -f -- "${backend}")
  device_smoke=$(readlink -f -- "${device_smoke}")
  input=$(readlink -f -- "${input}")
  reference=$(readlink -f -- "${reference}")
  klayout_dir=$(dirname -- "${klayout}")
  backend_dir=$(dirname -- "${backend}")
  runtime_ld_library_path="${backend_dir}:${klayout_dir}"
fi

work=$(mktemp -d \
  "${TMPDIR:-/tmp}/klayout-m2-rules-production-owner.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "M2_RULES_PRODUCTION_OWNER_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- \
  "${work}/decks" \
  "${work}/logs" \
  "${work}/reports" \
  "${work}/runtime" \
  "${work}/timings"

default_deck="${work}/decks/default.lydrc"
default_repeat="${work}/decks/default-repeat.lydrc"
live_deck="${work}/decks/m2-rules-live.lydrc"
live_repeat="${work}/decks/m2-rules-live-repeat.lydrc"
generator_log="${work}/logs/deck-generator.log"

run_generator() {
  local output=$1
  shift
  if ! "${python}" "${deck_generator}" \
    --input "${source_deck}" --output "${output}" \
    --m1-contact --implant12 "$@" >>"${generator_log}" 2>&1; then
    cat -- "${generator_log}" >&2
    die "deck generation failed for ${output}"
  fi
}

run_generator "${default_deck}"
run_generator "${default_repeat}"
cmp -s -- "${default_deck}" "${default_repeat}" ||
  die "repeated default deck generation is not byte-identical"
cmp -s -- "${default_deck_reference}" "${default_deck}" ||
  {
    sha256sum -- "${default_deck_reference}" "${default_deck}" >&2
    die "default deck changed relative to the pre-change reference"
  }

run_generator "${live_deck}" --m2-rules
run_generator "${live_repeat}" --m2-rules
cmp -s -- "${live_deck}" "${live_repeat}" ||
  die "repeated M2-rules deck generation is not byte-identical"
[[ -s "${live_deck}" ]] || die "generated M2-rules deck is empty"
if grep -Fq -- \
  'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s' \
  "${default_deck}"; then
  die "default generator output unexpectedly contains the M2-rules rewrite"
fi

mutated_source="${work}/decks/mutated-m2-source.lydrc"
mutated_output="${work}/decks/mutated-m2-output.lydrc"
mutation_log="${work}/logs/deck-mutation.log"
m2_9_source='metal2_gt1500.edges.with_length(4.um,nil).space(1500.nm,euclidian).output("METAL2.9"'
[[ "$(grep -Fc -- "${m2_9_source}" "${source_deck}" || true)" == 1 ]] ||
  die "source deck does not contain one exact M2.9 matcher"
sed \
  's/metal2_gt1500\.edges\.with_length(4\.um,nil)\.space(1500\.nm,euclidian)\.output("METAL2\.9"/metal2_gt1500.edges.with_length(4.um,nil).space(1501.nm,euclidian).output("METAL2.9"/' \
  "${source_deck}" >"${mutated_source}"
cmp -s -- "${source_deck}" "${mutated_source}" &&
  die "M2.9 mutation did not change the source deck"
if "${python}" "${deck_generator}" \
  --input "${mutated_source}" --output "${mutated_output}" \
  --m1-contact --implant12 --m2-rules >"${mutation_log}" 2>&1; then
  die "changed M2.9 rule was incorrectly accepted"
fi
grep -Fq -- \
  "M2.4-.9 speculative transaction: expected one source block, found 0" \
  "${mutation_log}" ||
  {
    cat -- "${mutation_log}" >&2
    die "changed M2.9 rule did not fail at the exact matcher"
  }

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
  'm2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s' \
  "runtime opt-in"
assert_deck_count 1 \
  'metal2.respond_to?(:cuda_m2_flat_union)' \
  "live method guard"
assert_deck_count 1 \
  'metal2.cuda_m2_flat_union(via2)' \
  "flat operand transaction"
assert_deck_count 3 \
  'm2_rules_flat_results &lt;&lt;' \
  "speculative result census"
assert_deck_count 8 \
  'm2_rules_empty.output' \
  "atomic empty output census"
assert_deck_count 1 \
  'metal2_width, metal2_space = metal2.drc_batch([' \
  "pristine M2.1/.2 CPU fallback"
assert_deck_count 1 \
  'via2_edges_with_less_enclosure = metal2.enclosing(via2, 35.nm, projection).second_edges' \
  "pristine M2.4 CPU fallback"
assert_deck_count 1 \
  'classify_by_width(metal2, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)' \
  "pristine M2.5-.9 CPU fallback"
assert_deck_count 0 \
  'm2_rules_empty.output("METAL2.3"' \
  "forbidden M2.3 ownership"

for category in \
  METAL2.1 METAL2.2 METAL2.4 METAL2.5 \
  METAL2.6 METAL2.7 METAL2.8 METAL2.9; do
  assert_deck_count 2 \
    ".output(\"${category}\"" \
    "${category} atomic/fallback output"
done
assert_deck_count 3 '.output("METAL2.3"' "preserved M2.3 ownership"
for category in VIA2.1 VIA2.2 VIA2.3 VIA2.4; do
  assert_deck_count 1 \
    ".output(\"${category}\"" \
    "${category} preserved ownership"
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
  die "speculative M2 path publishes a non-atomic output"
fi

clean_assignment='m2_rules_clean = m2_rules_flat_results.length == 3 &amp;&amp; m2_rules_flat_results.all? { |result| result.is_empty? }'
[[ "$(grep -Fc -- "${clean_assignment}" "${speculative_block}" || true)" == 1 ]] ||
  die "generated deck must contain one exact three-prefix-result decision"
[[ "$(grep -Ec -- '^[[:space:]]*ensure$' \
  "${speculative_block}" || true)" == 1 ]] ||
  die "speculative transaction must contain one ensure boundary"
[[ "$(grep -Fc -- 'm2_rules_flat_temps.reverse_each do |layer|' \
  "${speculative_block}" || true)" == 1 ]] ||
  die "speculative transaction must contain one reverse cleanup"
[[ "$(grep -Fc -- "'certified-empty'" \
  "${speculative_block}" || true)" == 1 ]] ||
  die "speculative transaction must contain one post-cleanup certification"

clean_line=$(grep -nF -- "${clean_assignment}" \
  "${speculative_block}" | cut -d: -f1)
ensure_line=$(grep -nE -- '^[[:space:]]*ensure$' \
  "${speculative_block}" | cut -d: -f1)
cleanup_line=$(grep -nF -- \
  'm2_rules_flat_temps.reverse_each do |layer|' \
  "${speculative_block}" | cut -d: -f1)
certified_line=$(grep -nF -- "'certified-empty'" \
  "${speculative_block}" | cut -d: -f1)
((clean_line < ensure_line &&
  ensure_line < cleanup_line &&
  cleanup_line < certified_line)) ||
  die "three-prefix decision, cleanup, and certification are misordered"

echo \
  "M2_RULES_PRODUCTION_OWNER_GATE ok gate=generator default-byte-identical=1 repeatable=1 exact-matcher=1 prefix-results=3 suffix-bits=5 atomic-outputs=8 pristine-fallbacks=8 ownership=M2.3,VIA2.1-.4"

if ((generate_only)); then
  sha256sum -- \
    "${source_deck}" \
    "${default_deck_reference}" \
    "${default_deck}" \
    "${live_deck}" \
    "${deck_generator}" \
    "${host_hook_probe}" \
    "${script_path}" \
    >"${work}/pinned-artifacts.sha256"
  cat -- "${work}/pinned-artifacts.sha256"
  echo "M2_RULES_PRODUCTION_OWNER_GATE PASS gate=generate-only"
  exit 0
fi

reference_canonical="${work}/reports/reference.canonical.lyrdb"
sed '/<generator>/d' "${reference}" >"${reference_canonical}"
[[ -s "${reference_canonical}" ]] ||
  die "canonical reference report is empty"
grep -Fq -- "<report-database>" "${reference_canonical}" ||
  die "reference is not an XML KLayout report database"
grep -Fq -- "<top-cell>${top_cell}</top-cell>" "${reference_canonical}" ||
  die "reference top-cell does not match --top-cell"
if grep -Fq -- "<name>METAL2.3</name>" "${reference_canonical}"; then
  die "reference incorrectly assigns METAL2.3 to the M2-rules owner"
fi
[[ "$(grep -Fc -- "<category>" "${reference_canonical}" || true)" == 8 ]] ||
  die "reference must contain exactly eight M2-rules categories"
for category in \
  METAL2.1 METAL2.2 METAL2.4 METAL2.5 \
  METAL2.6 METAL2.7 METAL2.8 METAL2.9; do
  [[ "$(grep -Fc -- \
    "<name>${category}</name>" "${reference_canonical}" || true)" == 1 ]] ||
    die "reference must contain exactly one ${category} category"
done
[[ "$(grep -Fc -- "<item>" "${reference_canonical}" || true)" == 0 ]] ||
  die "reference M2-rules owner report must contain no violation items"

canonicalize_report() {
  local report=$1
  local output=$2
  sed '/<generator>/d' "${report}" >"${output}"
}

resolved_dependency() {
  local artifact=$1
  local soname_pattern=$2
  env -i \
    PATH=/usr/bin:/bin \
    "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
    /usr/bin/ldd "${artifact}" |
    awk -v pattern="${soname_pattern}" \
      '$1 ~ pattern && $2 == "=>" {
         print $3
         exit
       }'
}

capture_runtime_closure() {
  local label=$1
  local prefix="${work}/runtime/${label}-closure"
  local combined="${prefix}.ldd"
  local paths="${prefix}.paths"
  local hashes="${prefix}.sha256"
  local symbols="${prefix}.backend-symbols"
  local selected_backend
  local selected_db
  local selected_drc
  local selected_klayout_library

  : >"${combined}"
  for artifact in "${klayout}" "${device_smoke}" "${backend}"; do
    {
      echo "ARTIFACT ${artifact}"
      /usr/bin/readelf -d "${artifact}"
      env -i \
        PATH=/usr/bin:/bin \
        "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
        /usr/bin/ldd "${artifact}"
    } >>"${combined}" 2>&1
  done
  if grep -Fq -- "not found" "${combined}"; then
    cat -- "${combined}" >&2
    die "${label}: runtime closure has an unresolved dependency"
  fi

  selected_backend=$(
    resolved_dependency \
      "${device_smoke}" '^libklayout_cuda_spatial_backend[.]so'
  )
  [[ -n "${selected_backend}" ]] ||
    die "${label}: device smoke has no loader-selected CUDA backend"
  [[ "$(readlink -f -- "${selected_backend}")" == "${backend}" ]] ||
    die "${label}: device smoke selected ${selected_backend}, expected ${backend}"

  selected_db=$(
    resolved_dependency "${klayout}" '^libklayout_db[.]so'
  )
  [[ -n "${selected_db}" ]] ||
    die "${label}: KLayout has no loader-selected libklayout_db"
  [[ "$(dirname -- "$(readlink -f -- "${selected_db}")")" == \
      "${klayout_dir}" ]] ||
    die "${label}: KLayout selected ${selected_db} outside ${klayout_dir}"

  selected_drc=$(
    resolved_dependency "${klayout}" '^libklayout_drc[.]so'
  )
  [[ -n "${selected_drc}" ]] ||
    die "${label}: KLayout has no loader-selected libklayout_drc"
  [[ "$(dirname -- "$(readlink -f -- "${selected_drc}")")" == \
      "${klayout_dir}" ]] ||
    die "${label}: KLayout selected ${selected_drc} outside ${klayout_dir}"

  while IFS= read -r selected_klayout_library; do
    [[ "$(dirname -- \
      "$(readlink -f -- "${selected_klayout_library}")")" == \
      "${klayout_dir}" ]] ||
      die "${label}: KLayout selected ${selected_klayout_library} outside ${klayout_dir}"
  done < <(
    env -i \
      PATH=/usr/bin:/bin \
      "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
      /usr/bin/ldd "${klayout}" |
      awk \
        '$1 ~ /^libklayout_.*[.]so/ && $2 == "=>" {
           print $3
         }'
  )

  /usr/bin/nm -D --defined-only "${backend}" >"${symbols}"
  for symbol in \
    klayout_cuda_spatial_run_m2_union_boundary_v1 \
    klayout_cuda_spatial_release_m2_union_boundary_v1; do
    [[ "$(awk -v symbol="${symbol}" '$NF == symbol { ++count } END {
      print count + 0
    }' "${symbols}")" == 1 ]] ||
      die "${label}: backend must export exactly one ${symbol}"
  done

  {
    printf '%s\n' "${klayout}" "${device_smoke}" "${backend}"
    awk \
      '$2 == "=>" && $3 ~ /^\// { print $3 }
       $1 ~ /^\// { print $1 }' \
      "${combined}"
  } |
    while IFS= read -r artifact; do
      readlink -f -- "${artifact}"
    done |
    sort -u >"${paths}"

  while IFS= read -r artifact; do
    [[ -f "${artifact}" ]] ||
      die "${label}: resolved closure member is not a file: ${artifact}"
    sha256sum -- "${artifact}"
  done <"${paths}" >"${hashes}"

  printf 'selected_backend=%s\nselected_libdb=%s\nselected_libdrc=%s\n' \
    "$(readlink -f -- "${selected_backend}")" \
    "$(readlink -f -- "${selected_db}")" \
    "$(readlink -f -- "${selected_drc}")" \
    >"${prefix}.selected"
}

run_device_preflight() {
  local lane=$1
  local label="${lane}-before"
  local log="${work}/logs/${lane}-device-preflight.log"
  local hook_log="${work}/logs/${lane}-host-hook-preflight.log"
  local compute_log="${work}/logs/${lane}-preexisting-compute.log"
  local hook_runtime="${work}/runtime/${lane}-hook"

  if ! /usr/bin/nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"${compute_log}" 2>&1; then
    cat -- "${compute_log}" >&2
    die "${lane}: unable to census pre-existing CUDA processes"
  fi
  if grep -Eq -- '^[[:space:]]*[0-9]+' "${compute_log}"; then
    cat -- "${compute_log}" >&2
    die "${lane}: another CUDA compute process would contaminate timing"
  fi
  if ! /usr/bin/nvidia-smi \
    --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}: nvidia-smi device preflight failed"
  fi

  capture_runtime_closure "${label}"
  mkdir -p -- \
    "${hook_runtime}/home" \
    "${hook_runtime}/klayout-home" \
    "${hook_runtime}/xdg-config" \
    "${hook_runtime}/xdg-cache" \
    "${hook_runtime}/xdg-data" \
    "${hook_runtime}/tmp"
  if ! env -i \
    "HOME=${hook_runtime}/home" \
    "KLAYOUT_HOME=${hook_runtime}/klayout-home" \
    "XDG_CONFIG_HOME=${hook_runtime}/xdg-config" \
    "XDG_CACHE_HOME=${hook_runtime}/xdg-cache" \
    "XDG_DATA_HOME=${hook_runtime}/xdg-data" \
    "TMPDIR=${hook_runtime}/tmp" \
    PATH=/usr/bin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
    QT_QPA_PLATFORM=offscreen \
    CUDA_VISIBLE_DEVICES= \
    KLAYOUT_CUDA_M2_RULES=0 \
    KLAYOUT_CUDA_M2_RULES_TELEMETRY=0 \
    "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
    "${klayout}" -b -r "${host_hook_probe}" >"${hook_log}" 2>&1; then
    cat -- "${hook_log}" >&2
    die "${lane}: loader-selected host lacks cuda_m2_flat_union"
  fi
  [[ "$(grep -Fc -- \
    "M2_RULES_HOST_HOOK_PREFLIGHT PASS" "${hook_log}" || true)" == 1 ]] ||
    {
      cat -- "${hook_log}" >&2
      die "${lane}: host hook completion marker is missing"
    }
  if ! env -i \
    PATH=/usr/bin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
    CUDA_VISIBLE_DEVICES=0 \
    KLAYOUT_CUDA_SPATIAL_DEVICE=0 \
    KLAYOUT_CUDA_SPATIAL_TELEMETRY=1 \
    "KLAYOUT_CUDA_SPATIAL_BACKEND=${backend}" \
    "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
    "${device_smoke}" >>"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}: M2 CUDA backend device smoke failed"
  fi
  [[ "$(grep -Fc -- \
    "M2_UNION_PRODUCTION_BACKEND_SMOKE PASS transforms=8 rectangles=16 segments=48 fnv64=5530745522766976907 adversarial=11 release_idempotent=1" \
    "${log}" || true)" == 1 ]] ||
    {
      cat -- "${log}" >&2
      die "${lane}: exact M2 backend smoke completion marker is missing"
    }
  echo \
    "M2_RULES_PRODUCTION_OWNER_GATE ok gate=device-preflight lane=${lane} device=0 runtime-closure=bound gpu-idle=1"
}

pin_static_artifacts() {
  local output=$1
  sha256sum -- \
    "${klayout}" \
    "${backend}" \
    "${device_smoke}" \
    "${source_deck}" \
    "${default_deck_reference}" \
    "${default_deck}" \
    "${live_deck}" \
    "${input}" \
    "${reference}" \
    "${reference_canonical}" \
    "${deck_generator}" \
    "${host_hook_probe}" \
    "${script_path}" \
    >"${output}"
}

run_lane() {
  local lane=$1
  local enabled=$2
  local runtime="${work}/runtime/${lane}"
  local report="${work}/reports/${lane}.lyrdb"
  local log="${work}/logs/${lane}.log"
  local time_file="${work}/timings/${lane}.txt"

  mkdir -p -- \
    "${runtime}/home" \
    "${runtime}/klayout-home" \
    "${runtime}/xdg-config" \
    "${runtime}/xdg-cache" \
    "${runtime}/xdg-data" \
    "${runtime}/tmp"

  printf '%s lane=%s phase=start\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${lane}" \
    >>"${work}/run-order.txt"
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
        "TMPDIR=${runtime}/tmp" \
        PATH=/usr/bin:/bin \
        LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
        PYTHONDONTWRITEBYTECODE=1 \
        QT_QPA_PLATFORM=offscreen \
        CUDA_VISIBLE_DEVICES=0 \
        KLAYOUT_CUDA_SPATIAL_DEVICE=0 \
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1 \
        "KLAYOUT_CUDA_SPATIAL_BACKEND=${backend}" \
        "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
        KLAYOUT_CUDA_ACTIVE3=1 \
        KLAYOUT_CUDA_ACTIVE3_TELEMETRY=1 \
        KLAYOUT_CUDA_M1_WIDTH_SPACE=1 \
        KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1 \
        "KLAYOUT_CUDA_M2_RULES=${enabled}" \
        "KLAYOUT_CUDA_M2_RULES_TELEMETRY=${enabled}" \
        KLAYOUT_CUDA_M2_WIDTH_SPACE=0 \
        KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY=0 \
        KLAYOUT_CUDA_VIA1_STACK=1 \
        KLAYOUT_CUDA_VIA1_STACK_TELEMETRY=1 \
        KLAYOUT_CUDA_M1_CONTACT=1 \
        KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1 \
        KLAYOUT_CUDA_CONTACT4=1 \
        KLAYOUT_CUDA_CONTACT4_TELEMETRY=1 \
        KLAYOUT_CUDA_IMPLANT12=1 \
        KLAYOUT_CUDA_IMPLANT12_TELEMETRY=1 \
        KLAYOUT_CUDA_DISCONNECTED_MERGE=1 \
        KLAYOUT_DEEP_EDGE_CERT_PROFILE=1 \
        KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=100000 \
        KLAYOUT_CUDA_SPATIAL_CELL_SIZE=512 \
        KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64 \
        KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096 \
        KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=80000000 \
        KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=100000000 \
        KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=30000000 \
        "${klayout}" -b \
          -r "${live_deck}" \
          -rd "input=${input}" \
          -rd "topcell=${top_cell}" \
          -rd "output=${report}" \
          -rd drc_shard=m2_rules \
          >"${log}" 2>&1
  local rc=$?
  set -e
  printf '%s lane=%s phase=finish status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${lane}" "${rc}" \
    >>"${work}/run-order.txt"
  if ((rc != 0)); then
    cat -- "${time_file}" >&2 || true
    tail -n 200 -- "${log}" >&2 || true
    die "${lane}: production M2-rules owner failed with status ${rc}"
  fi
  [[ -s "${report}" ]] ||
    die "${lane}: production M2-rules owner produced no report"
  grep -Fq -- "<top-cell>${top_cell}</top-cell>" "${report}" ||
    die "${lane}: report top-cell marker is missing"
  canonicalize_report \
    "${report}" "${work}/reports/${lane}.canonical.lyrdb"

  capture_runtime_closure "${lane}-after"
  cmp -s -- \
    "${work}/runtime/${lane}-before-closure.sha256" \
    "${work}/runtime/${lane}-after-closure.sha256" ||
    die "${lane}: runtime closure changed during the owner run"
}

assert_exact_marker() {
  local lane=$1
  local marker=$2
  local label=$3
  local log="${work}/logs/${lane}.log"
  local count
  count=$(grep -Fc -- "${marker}" "${log}" || true)
  [[ "${count}" == 1 ]] ||
    die "${lane}: expected one ${label} marker, found ${count}"
}

assert_ownership_isolation() {
  local lane=$1
  local log="${work}/logs/${lane}.log"
  if grep -Eq -- \
    'CUDA ACTIVE\.3 |CUDA M1 width/space |CUDA M2 width/space |CUDA VIA1 stack |CUDA M1 contact |CUDA CONTACT\.4 |CUDA IMPLANT\.1/\.2 |CUDA POLY\.3/\.4 ' \
    "${log}"; then
    grep -nE -- \
      'CUDA ACTIVE\.3 |CUDA M1 width/space |CUDA M2 width/space |CUDA VIA1 stack |CUDA M1 contact |CUDA CONTACT\.4 |CUDA IMPLANT\.1/\.2 |CUDA POLY\.3/\.4 ' \
      "${log}" >&2 || true
    die "${lane}: an unrelated qualified owner transaction ran in m2_rules"
  fi
}

pin_static_artifacts "${work}/pinned-artifacts-before.sha256"

# This is deliberately a fixed-order, serial timing comparison.
run_device_preflight control
run_lane control 0
assert_ownership_isolation control
if grep -Eq -- \
  'CUDA M2 (exact union boundary:|live flat operands:|rules transaction:)' \
  "${work}/logs/control.log"; then
  die "feature-off control unexpectedly invoked the live M2 transaction"
fi

run_device_preflight candidate
cmp -s -- \
  "${work}/runtime/control-before-closure.sha256" \
  "${work}/runtime/candidate-before-closure.sha256" ||
  die "control and candidate selected different runtime closures"
run_lane candidate 1
assert_ownership_isolation candidate

assert_exact_marker \
  candidate \
  "CUDA M2 exact union boundary: outcome=complete" \
  "complete exact-union"
assert_exact_marker \
  candidate \
  "CUDA M2 live flat operands: disposition=complete" \
  "complete live-flat"
assert_exact_marker \
  candidate \
  "CUDA M2 rules transaction: certified-empty reason=prefix-clean+suffix-certified" \
  "certified-empty rules transaction"
for family in \
  "CUDA M2 exact union boundary:" \
  "CUDA M2 live flat operands:" \
  "CUDA M2 rules transaction:"; do
  [[ "$(grep -Fc -- "${family}" \
    "${work}/logs/candidate.log" || true)" == 1 ]] ||
    die "candidate: expected one total ${family} marker"
done
grep -Fq -- \
  "CUDA spatial backend loaded: ${backend}" \
  "${work}/logs/candidate.log" ||
  die "candidate: exact CUDA backend load marker is missing"

union_line=$(
  grep -F -- "CUDA M2 exact union boundary: outcome=complete" \
    "${work}/logs/candidate.log"
)
# The live DeepShapeStore retains the complete production hierarchy.  The
# older standalone M2 exporter compacted empty contexts to 587201; every
# M2-bearing context and geometry fingerprint below is identical, but that
# compact-scene total is not the live owner census.
for token in \
  contexts=849265 \
  metal_contexts=568632 \
  rectangles=22946444 \
  slabs=46383 \
  memberships=92386704 \
  events=184773408 \
  strip_intervals=3691466 \
  raw_segments=9575624 \
  segments=4385384 \
  fnv64=7541395996791771514 \
  fallback_flags=0 \
  device_flags=0; do
  [[ " ${union_line} " == *" ${token} "* ]] ||
    die "candidate: exact-union topology fingerprint is missing ${token}"
done

union_line_number=$(
  grep -nF -- "CUDA M2 exact union boundary:" \
    "${work}/logs/candidate.log" | cut -d: -f1
)
flat_line_number=$(
  grep -nF -- "CUDA M2 live flat operands:" \
    "${work}/logs/candidate.log" | cut -d: -f1
)
transaction_line_number=$(
  grep -nF -- "CUDA M2 rules transaction:" \
    "${work}/logs/candidate.log" | cut -d: -f1
)
((union_line_number < flat_line_number &&
  flat_line_number < transaction_line_number)) ||
  die "candidate: union, live-flat, and transaction telemetry are misordered"

flat_line=$(
  grep -F -- "CUDA M2 live flat operands: disposition=complete" \
    "${work}/logs/candidate.log"
)
for token in \
  raw_segments=9575624 \
  boundary_segments=4385384 \
  suffix_mask=31 \
  contours=14222 \
  vertices=4385384 \
  via2_polygons=10128 \
  via2_property_polygons=0 \
  reason=none; do
  [[ " ${flat_line} " == *" ${token} "* ]] ||
    die "candidate: live-flat topology fingerprint is missing ${token}"
done
[[ " ${flat_line} " =~ [[:space:]]suffix_ms=([0-9]+\.[0-9]+)[[:space:]] ]] &&
  [[ "${BASH_REMATCH[1]}" != "0.000" ]] ||
  die "candidate: resident M2.5-.9 timing is missing or zero"

if grep -Eq -- \
  'CUDA M2 rules transaction: full-cpu-fallback|reason=prefix-rule-hit|CUDA M2 exact union boundary: outcome=(fallback|error|invalid-result|disabled)|CUDA M2 live flat operands: disposition=(host-declined|backend-fallback|backend-error|invalid-result|topology-declined|disabled)|cudaSetDevice:' \
  "${work}/logs/candidate.log"; then
  die "candidate: M2 telemetry reported a fallback, decline, or device error"
fi

for lane in control candidate; do
  cmp -s -- \
    "${reference_canonical}" \
    "${work}/reports/${lane}.canonical.lyrdb" ||
    {
      sha256sum -- \
        "${reference_canonical}" \
        "${work}/reports/${lane}.canonical.lyrdb" >&2
      die "${lane}: report differs from the canonical production reference"
    }
done
cmp -s -- \
  "${work}/reports/control.canonical.lyrdb" \
  "${work}/reports/candidate.canonical.lyrdb" ||
  die "candidate report differs from the feature-off control"
cmp -s -- \
  "${work}/reports/control.lyrdb" \
  "${work}/reports/candidate.lyrdb" ||
  die "candidate raw report differs from the same-deck control"

pin_static_artifacts "${work}/pinned-artifacts-after.sha256"
cmp -s -- \
  "${work}/pinned-artifacts-before.sha256" \
  "${work}/pinned-artifacts-after.sha256" ||
  die "a pinned executable, input, deck, reference, or gate changed during A/B"

sha256sum -- \
  "${work}/reports/control.lyrdb" \
  "${work}/reports/candidate.lyrdb" \
  "${work}/reports/control.canonical.lyrdb" \
  "${work}/reports/candidate.canonical.lyrdb" \
  "${work}/runtime/control-before-closure.sha256" \
  "${work}/runtime/control-after-closure.sha256" \
  "${work}/runtime/candidate-before-closure.sha256" \
  "${work}/runtime/candidate-after-closure.sha256" \
  "${work}/logs/control-device-preflight.log" \
  "${work}/logs/candidate-device-preflight.log" \
  "${work}/logs/control-host-hook-preflight.log" \
  "${work}/logs/candidate-host-hook-preflight.log" \
  >"${work}/evidence.sha256"

control_wall=$(
  sed -n 's/^wall_s=\([^ ]*\).*/\1/p' \
    "${work}/timings/control.txt"
)
candidate_wall=$(
  sed -n 's/^wall_s=\([^ ]*\).*/\1/p' \
    "${work}/timings/candidate.txt"
)
[[ -n "${control_wall}" ]] || die "control wall time is missing"
[[ -n "${candidate_wall}" ]] || die "candidate wall time is missing"
awk -v control="${control_wall}" -v candidate="${candidate_wall}" \
  'BEGIN {
    delta = control - candidate
    percent = control == 0 ? 0 : 100 * delta / control
    printf "M2_RULES_PRODUCTION_OWNER_AB control_wall_s=%.2f candidate_wall_s=%.2f delta_s=%.2f reduction_pct=%.3f report=exact\n", control, candidate, delta, percent
  }'

cat -- "${work}/run-order.txt"
cat -- "${work}/timings/control.txt"
cat -- "${work}/timings/candidate.txt"
sha256sum -- \
  "${reference_canonical}" \
  "${work}/reports/control.canonical.lyrdb" \
  "${work}/reports/candidate.canonical.lyrdb"
grep -h -E -- \
  'CUDA M2 exact union boundary:|CUDA M2 live flat operands:|CUDA M2 rules transaction:' \
  "${work}/logs/control.log" \
  "${work}/logs/candidate.log"
echo \
  "M2_RULES_PRODUCTION_OWNER_GATE PASS owner=m2_rules lanes=2 serial=control,candidate canonical-reference=exact runtime-closure=stable device-preflight=each-lane ownership-isolated=1"
