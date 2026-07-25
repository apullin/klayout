#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_m2_resident_suffix_production_owner_ab.sh \
    --current-klayout PATH --current-backend PATH \
    --source-deck PATH --default-deck-reference PATH \
    --input PATH --top-cell NAME --reference PATH \
    --expected-report-sha256 HEX \
    [--previous-klayout PATH] [--previous-backend PATH] \
    [--previous-deck PATH] [--python PATH] [--repetitions N] \
    [--timeout-seconds N] [--evidence-root DIR]

  bash run_m2_resident_suffix_production_owner_ab.sh \
    --print-order [--repetitions N]

Runs a serial, alternating-order A/B of the FreePDK45 m2_rules production
owner.  Both lanes keep the already-accelerated exact raw-M2 union enabled:

  previous: pinned pre-resident host, backend, and eight-host-rule deck
  resident: selected current host/backend and a freshly generated deck which
            consumes the exact resident M2.5-.9 empty certificate

The legacy defaults are the exact artifacts used by the accepted 37.53-second
accelerated owner record.  Their SHA-256s are fixed below; rebuilt or replaced
/tmp artifacts fail before timing.  The resident deck is generated twice from
--source-deck with the checked-in generator and must be byte-identical across
both generations.  Its default-mode generation must match
--default-deck-reference, so this timing gate cannot hide generator drift.

For each replicate, odd pairs run previous/resident and even pairs run
resident/previous.  --repetitions defaults to 3.  Each lane starts and ends
with an idle-GPU census, binds and hashes its complete loader closure before
and after execution, emits the expected lane-specific M2 telemetry, and
produces a generator-stripped report byte-identical to the canonical
production reference.

The summary reports absolute mean owner seconds and percent less time relative
to the previous accelerated implementation.  This is an owner-only result,
not a full-launcher or critical-path claim.  A fresh evidence directory is
always retained under --evidence-root (default: TMPDIR or /tmp), including
decks, reports, logs, timings, order, loader closures, and hashes.
EOF
}

die() {
  echo "M2 resident suffix production owner A/B: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
deck_generator="${here}/make_via1_stack_live_deck.py"

previous_klayout=/tmp/klayout-m2-live-gsi-bin/klayout
previous_backend=/tmp/klayout-m2-combined-real-cmake/libklayout_cuda_spatial_backend.so
previous_deck=/home/pullin/personal/klayout/.scratchpad/cuda-runs/klayout-m2-rules-production-owner.oSbILc/decks/m2-rules-live.lydrc
readonly previous_klayout_sha256=6147d349166c38dcdba514ebbe23f6ada9a94f1861ebfeff6f38aaf9701ccf9f
readonly previous_backend_sha256=f764cd4587f71e98cfc540f496aba0096c88991b1c3e1d18376a6227b3961c7a
readonly previous_deck_sha256=f68ac26ca91bb4c2a8225ce71d93a34b4d8a61d303b96748a1b67f4facc360d3

current_klayout=
current_backend=
source_deck=
default_deck_reference=
input=
top_cell=
reference=
expected_report_sha256=
python=${PYTHON:-python3}
repetitions=3
timeout_seconds=900
evidence_root=${KLAYOUT_CUDA_EVIDENCE_ROOT:-${TMPDIR:-/tmp}}
print_order=0

while (($#)); do
  case "$1" in
    --previous-klayout)
      (($# >= 2)) || die "--previous-klayout requires a value"
      previous_klayout=$2
      shift 2
      ;;
    --previous-backend)
      (($# >= 2)) || die "--previous-backend requires a value"
      previous_backend=$2
      shift 2
      ;;
    --previous-deck)
      (($# >= 2)) || die "--previous-deck requires a value"
      previous_deck=$2
      shift 2
      ;;
    --current-klayout)
      (($# >= 2)) || die "--current-klayout requires a value"
      current_klayout=$2
      shift 2
      ;;
    --current-backend)
      (($# >= 2)) || die "--current-backend requires a value"
      current_backend=$2
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
    --expected-report-sha256)
      (($# >= 2)) || die "--expected-report-sha256 requires a value"
      expected_report_sha256=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
      shift 2
      ;;
    --repetitions)
      (($# >= 2)) || die "--repetitions requires a value"
      repetitions=$2
      shift 2
      ;;
    --timeout-seconds)
      (($# >= 2)) || die "--timeout-seconds requires a value"
      timeout_seconds=$2
      shift 2
      ;;
    --evidence-root)
      (($# >= 2)) || die "--evidence-root requires a value"
      evidence_root=$2
      shift 2
      ;;
    --print-order)
      print_order=1
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

[[ "${repetitions}" =~ ^[1-9][0-9]*$ ]] ||
  die "--repetitions must be a positive integer"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] ||
  die "--timeout-seconds must be a positive integer"

emit_order() {
  local replicate
  local sequence
  local mode
  local index=0
  for ((replicate = 1; replicate <= repetitions; ++replicate)); do
    if ((replicate % 2)); then
      sequence="previous resident"
    else
      sequence="resident previous"
    fi
    for mode in ${sequence}; do
      ((index += 1))
      printf '%s\t%s-%s\t%s\t%s\n' \
        "${index}" "${mode}" "${replicate}" "${mode}" "${replicate}"
    done
  done
}

if ((print_order)); then
  emit_order
  exit 0
fi

[[ -n "${current_klayout}" ]] || die "missing --current-klayout"
[[ -n "${current_backend}" ]] || die "missing --current-backend"
[[ -n "${source_deck}" ]] || die "missing --source-deck"
[[ -n "${default_deck_reference}" ]] ||
  die "missing --default-deck-reference"
[[ -n "${input}" ]] || die "missing --input"
[[ -n "${top_cell}" ]] || die "missing --top-cell"
[[ -n "${reference}" ]] || die "missing --reference"
[[ -n "${expected_report_sha256}" ]] ||
  die "missing --expected-report-sha256"
[[ "${expected_report_sha256}" =~ ^[0-9A-Fa-f]{64}$ ]] ||
  die "--expected-report-sha256 must be exactly 64 hexadecimal digits"

[[ -x "${previous_klayout}" ]] ||
  die "previous KLayout is not executable: ${previous_klayout}"
[[ -f "${previous_backend}" ]] ||
  die "previous CUDA backend is missing: ${previous_backend}"
[[ -f "${previous_deck}" ]] ||
  die "previous M2 deck is missing: ${previous_deck}"
[[ -x "${current_klayout}" ]] ||
  die "current KLayout is not executable: ${current_klayout}"
[[ -f "${current_backend}" ]] ||
  die "current CUDA backend is missing: ${current_backend}"
[[ -f "${source_deck}" ]] || die "source deck is missing: ${source_deck}"
[[ -f "${default_deck_reference}" ]] ||
  die "default deck reference is missing: ${default_deck_reference}"
[[ -s "${input}" ]] || die "input layout is missing or empty: ${input}"
[[ -s "${reference}" ]] ||
  die "reference report is missing or empty: ${reference}"
[[ -f "${deck_generator}" ]] || die "deck generator is missing"
[[ -x /usr/bin/time ]] || die "/usr/bin/time is unavailable"
[[ -x /usr/bin/timeout ]] || die "/usr/bin/timeout is unavailable"
[[ -x /usr/bin/readelf ]] || die "/usr/bin/readelf is unavailable"
[[ -x /usr/bin/ldd ]] || die "/usr/bin/ldd is unavailable"
[[ -x /usr/bin/nm ]] || die "/usr/bin/nm is unavailable"
[[ -x /usr/bin/nvidia-smi ]] || die "/usr/bin/nvidia-smi is unavailable"

python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
previous_klayout=$(readlink -f -- "${previous_klayout}")
previous_backend=$(readlink -f -- "${previous_backend}")
previous_deck=$(readlink -f -- "${previous_deck}")
current_klayout=$(readlink -f -- "${current_klayout}")
current_backend=$(readlink -f -- "${current_backend}")
source_deck=$(readlink -f -- "${source_deck}")
default_deck_reference=$(readlink -f -- "${default_deck_reference}")
input=$(readlink -f -- "${input}")
reference=$(readlink -f -- "${reference}")
expected_report_sha256=$(
  printf '%s' "${expected_report_sha256}" | tr 'A-F' 'a-f'
)

file_sha256() {
  sha256sum -- "$1" | awk '{print $1}'
}

[[ "$(file_sha256 "${previous_klayout}")" == \
    "${previous_klayout_sha256}" ]] ||
  die "previous KLayout does not match the pinned accelerated baseline"
[[ "$(file_sha256 "${previous_backend}")" == \
    "${previous_backend_sha256}" ]] ||
  die "previous backend does not match the pinned accelerated baseline"
[[ "$(file_sha256 "${previous_deck}")" == "${previous_deck_sha256}" ]] ||
  die "previous deck does not match the pinned accelerated baseline"

mkdir -p -- "${evidence_root}"
evidence_root=$(readlink -f -- "${evidence_root}")
work=$(mktemp -d \
  "${evidence_root}/klayout-m2-resident-suffix-owner-ab.XXXXXX")
trap 'echo "M2_RESIDENT_SUFFIX_PRODUCTION_OWNER_AB evidence=${work}"' EXIT
umask 077

mkdir -p -- \
  "${work}/decks" \
  "${work}/logs" \
  "${work}/reports" \
  "${work}/runtime" \
  "${work}/timings"

generator_log="${work}/logs/deck-generator.log"
current_default_deck="${work}/decks/current-default.lydrc"
current_default_repeat="${work}/decks/current-default-repeat.lydrc"
current_deck="${work}/decks/current-m2-rules.lydrc"
current_deck_repeat="${work}/decks/current-m2-rules-repeat.lydrc"

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

run_generator "${current_default_deck}"
run_generator "${current_default_repeat}"
cmp -s -- "${current_default_deck}" "${current_default_repeat}" ||
  die "current default deck generation is not byte-identical"
cmp -s -- "${default_deck_reference}" "${current_default_deck}" ||
  {
    sha256sum -- \
      "${default_deck_reference}" "${current_default_deck}" >&2
    die "current default generation differs from its reference"
  }

run_generator "${current_deck}" --m2-rules
run_generator "${current_deck_repeat}" --m2-rules
cmp -s -- "${current_deck}" "${current_deck_repeat}" ||
  die "current M2 deck generation is not byte-identical"

previous_clean='m2_rules_clean = m2_rules_flat_results.length == 8 &amp;&amp; m2_rules_flat_results.all? { |result| result.is_empty? }'
current_clean='m2_rules_clean = m2_rules_flat_results.length == 3 &amp;&amp; m2_rules_flat_results.all? { |result| result.is_empty? }'
[[ "$(grep -Fc -- "${previous_clean}" "${previous_deck}" || true)" == 1 ]] ||
  die "previous deck lacks its exact eight-result host suffix decision"
[[ "$(grep -Fc -- 'm2_rules_reason = m2_rules_clean ? "all-clean" : "rule-hit"' \
  "${previous_deck}" || true)" == 1 ]] ||
  die "previous deck lacks its exact all-host-rules decision"
[[ "$(grep -Fc -- "${current_clean}" "${current_deck}" || true)" == 1 ]] ||
  die "current deck lacks its exact three-result prefix decision"
[[ "$(grep -Fc -- \
  'm2_rules_reason = m2_rules_clean ? "prefix-clean+suffix-certified" : "prefix-rule-hit"' \
  "${current_deck}" || true)" == 1 ]] ||
  die "current deck lacks its exact resident-suffix decision"
cmp -s -- "${previous_deck}" "${current_deck}" &&
  die "previous and resident decks are unexpectedly byte-identical"

canonicalize_report() {
  local source=$1
  local output=$2
  sed '/<generator>/d' "${source}" >"${output}"
}

reference_canonical="${work}/reports/reference.canonical.lyrdb"
canonicalize_report "${reference}" "${reference_canonical}"
[[ -s "${reference_canonical}" ]] ||
  die "canonical production reference is empty"
reference_sha256=$(file_sha256 "${reference_canonical}")
[[ "${reference_sha256}" == "${expected_report_sha256}" ]] ||
  die "reference canonical SHA-256 is ${reference_sha256}, expected ${expected_report_sha256}"
grep -Fq -- "<top-cell>${top_cell}</top-cell>" "${reference_canonical}" ||
  die "reference top-cell does not match --top-cell"
[[ "$(grep -Fc -- "<category>" "${reference_canonical}" || true)" == 8 ]] ||
  die "reference must contain exactly eight M2-rules categories"
[[ "$(grep -Fc -- "<item>" "${reference_canonical}" || true)" == 0 ]] ||
  die "reference M2 owner report must contain no violation items"

select_mode() {
  local mode=$1
  case "${mode}" in
    previous)
      selected_klayout=${previous_klayout}
      selected_backend=${previous_backend}
      selected_deck=${previous_deck}
      ;;
    resident)
      selected_klayout=${current_klayout}
      selected_backend=${current_backend}
      selected_deck=${current_deck}
      ;;
    *)
      die "internal error: unknown lane mode ${mode}"
      ;;
  esac
  selected_klayout_dir=$(dirname -- "${selected_klayout}")
  selected_backend_dir=$(dirname -- "${selected_backend}")
  selected_ld_library_path="${selected_backend_dir}:${selected_klayout_dir}"
}

resolved_dependency() {
  local artifact=$1
  local soname_pattern=$2
  local ld_library_path=$3
  env -i \
    PATH=/usr/bin:/bin \
    "LD_LIBRARY_PATH=${ld_library_path}" \
    /usr/bin/ldd "${artifact}" |
    awk -v pattern="${soname_pattern}" \
      '$1 ~ pattern && $2 == "=>" {
         print $3
         exit
       }'
}

capture_runtime_closure() {
  local label=$1
  local mode=$2
  local prefix="${work}/runtime/${label}-closure"
  local combined="${prefix}.ldd"
  local paths="${prefix}.paths"
  local hashes="${prefix}.sha256"
  local symbols="${prefix}.backend-symbols"
  local loaded_db
  local loaded_drc
  local loaded_klayout_library
  local artifact

  select_mode "${mode}"
  : >"${combined}"
  for artifact in "${selected_klayout}" "${selected_backend}"; do
    {
      echo "ARTIFACT ${artifact}"
      /usr/bin/readelf -d "${artifact}"
      env -i \
        PATH=/usr/bin:/bin \
        "LD_LIBRARY_PATH=${selected_ld_library_path}" \
        /usr/bin/ldd "${artifact}"
    } >>"${combined}" 2>&1
  done
  if grep -Fq -- "not found" "${combined}"; then
    cat -- "${combined}" >&2
    die "${label}: runtime closure has an unresolved dependency"
  fi

  loaded_db=$(
    resolved_dependency \
      "${selected_klayout}" '^libklayout_db[.]so' \
      "${selected_ld_library_path}"
  )
  loaded_drc=$(
    resolved_dependency \
      "${selected_klayout}" '^libklayout_drc[.]so' \
      "${selected_ld_library_path}"
  )
  [[ -n "${loaded_db}" ]] ||
    die "${label}: KLayout closure has no libklayout_db"
  [[ -n "${loaded_drc}" ]] ||
    die "${label}: KLayout closure has no libklayout_drc"
  loaded_db=$(readlink -f -- "${loaded_db}")
  loaded_drc=$(readlink -f -- "${loaded_drc}")
  [[ "$(dirname -- "${loaded_db}")" == "${selected_klayout_dir}" ]] ||
    die "${label}: KLayout selected ${loaded_db} outside ${selected_klayout_dir}"
  [[ "$(dirname -- "${loaded_drc}")" == "${selected_klayout_dir}" ]] ||
    die "${label}: KLayout selected ${loaded_drc} outside ${selected_klayout_dir}"

  while IFS= read -r loaded_klayout_library; do
    [[ "$(dirname -- \
      "$(readlink -f -- "${loaded_klayout_library}")")" == \
      "${selected_klayout_dir}" ]] ||
      die "${label}: selected ${loaded_klayout_library} outside ${selected_klayout_dir}"
  done < <(
    env -i \
      PATH=/usr/bin:/bin \
      "LD_LIBRARY_PATH=${selected_ld_library_path}" \
      /usr/bin/ldd "${selected_klayout}" |
      awk \
        '$1 ~ /^libklayout_.*[.]so/ && $2 == "=>" {
           print $3
         }'
  )

  /usr/bin/nm -D --defined-only "${selected_backend}" >"${symbols}"
  for symbol in \
    klayout_cuda_spatial_run_m2_union_boundary_v1 \
    klayout_cuda_spatial_release_m2_union_boundary_v1; do
    [[ "$(awk -v symbol="${symbol}" '$NF == symbol { ++count } END {
      print count + 0
    }' "${symbols}")" == 1 ]] ||
      die "${label}: backend must export exactly one ${symbol}"
  done

  {
    printf '%s\n' \
      "${selected_klayout}" \
      "${selected_backend}" \
      "${selected_deck}"
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

  printf \
    'mode=%s\nklayout=%s\nbackend=%s\ndeck=%s\nlibdb=%s\nlibdrc=%s\n' \
    "${mode}" \
    "${selected_klayout}" \
    "${selected_backend}" \
    "${selected_deck}" \
    "${loaded_db}" \
    "${loaded_drc}" \
    >"${prefix}.selected"
}

assert_gpu_idle() {
  local label=$1
  local census="${work}/logs/${label}-compute-processes.csv"
  if ! /usr/bin/nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"${census}" 2>&1; then
    cat -- "${census}" >&2
    die "${label}: unable to census CUDA compute processes"
  fi
  if grep -Eq -- '^[[:space:]]*[0-9]+' "${census}"; then
    cat -- "${census}" >&2
    die "${label}: another CUDA compute process would contaminate timing"
  fi
}

marker_count() {
  local log=$1
  local marker=$2
  grep -Fc -- "${marker}" "${log}" || true
}

assert_marker_count() {
  local log=$1
  local marker=$2
  local expected=$3
  local label=$4
  local actual
  actual=$(marker_count "${log}" "${marker}")
  [[ "${actual}" == "${expected}" ]] ||
    die "${label}: expected ${expected} '${marker}' lines, found ${actual}"
}

assert_m2_telemetry() {
  local lane=$1
  local mode=$2
  local log="${work}/logs/${lane}.log"
  local union_line
  local flat_line

  select_mode "${mode}"
  assert_marker_count \
    "${log}" "CUDA spatial backend loaded: ${selected_backend}" 1 "${lane}"
  assert_marker_count \
    "${log}" "CUDA M2 exact union boundary: outcome=complete" 1 "${lane}"
  assert_marker_count \
    "${log}" "CUDA M2 live flat operands: disposition=complete" 1 "${lane}"
  assert_marker_count \
    "${log}" "CUDA M2 rules transaction: certified-empty" 1 "${lane}"

  union_line=$(
    grep -F -- "CUDA M2 exact union boundary: outcome=complete" "${log}"
  )
  for token in \
    contexts=849265 \
    metal_contexts=568632 \
    rectangles=22946444 \
    raw_segments=9575624 \
    segments=4385384 \
    fnv64=7541395996791771514 \
    fallback_flags=0 \
    device_flags=0; do
    [[ " ${union_line} " == *" ${token} "* ]] ||
      die "${lane}: exact-union topology fingerprint is missing ${token}"
  done

  flat_line=$(
    grep -F -- "CUDA M2 live flat operands: disposition=complete" "${log}"
  )
  for token in \
    raw_segments=9575624 \
    boundary_segments=4385384 \
    contours=14222 \
    vertices=4385384 \
    via2_polygons=10128 \
    via2_property_polygons=0 \
    reason=none; do
    [[ " ${flat_line} " == *" ${token} "* ]] ||
      die "${lane}: live-flat topology fingerprint is missing ${token}"
  done

  case "${mode}" in
    previous)
      assert_marker_count \
        "${log}" \
        "CUDA M2 rules transaction: certified-empty reason=all-clean" \
        1 "${lane}"
      [[ "${flat_line}" != *" suffix_mask="* ]] ||
        die "${lane}: previous lane unexpectedly consumed a suffix certificate"
      ;;
    resident)
      assert_marker_count \
        "${log}" \
        "CUDA M2 rules transaction: certified-empty reason=prefix-clean+suffix-certified" \
        1 "${lane}"
      [[ " ${flat_line} " == *" suffix_mask=31 "* ]] ||
        die "${lane}: resident lane lacks the all-five-bit suffix certificate"
      [[ " ${flat_line} " =~ [[:space:]]suffix_ms=([0-9]+\.[0-9]+)[[:space:]] ]] &&
        [[ "${BASH_REMATCH[1]}" != "0.000" ]] ||
        die "${lane}: resident suffix timing is missing or zero"
      ;;
  esac

  if grep -Eq -- \
    'CUDA M2 rules transaction: full-cpu-fallback|reason=prefix-rule-hit|CUDA M2 exact union boundary: outcome=(fallback|error|invalid-result|disabled)|CUDA M2 live flat operands: disposition=(host-declined|backend-fallback|backend-error|invalid-result|topology-declined|disabled)|cudaSetDevice:' \
    "${log}"; then
    die "${lane}: M2 telemetry reported a fallback, decline, or device error"
  fi
  if grep -Eq -- \
    'CUDA ACTIVE\.3 |CUDA M1 width/space |CUDA M2 width/space |CUDA VIA1 stack |CUDA M1 contact |CUDA CONTACT\.4 |CUDA IMPLANT\.1/\.2 |CUDA POLY\.3/\.4 ' \
    "${log}"; then
    die "${lane}: an unrelated owner transaction ran in m2_rules"
  fi
}

static_artifacts=(
  "${script_path}"
  "${deck_generator}"
  "${previous_klayout}"
  "${previous_backend}"
  "${previous_deck}"
  "${current_klayout}"
  "${current_backend}"
  "${source_deck}"
  "${default_deck_reference}"
  "${current_default_deck}"
  "${current_deck}"
  "${input}"
  "${reference}"
  "${reference_canonical}"
)
sha256sum -- "${static_artifacts[@]}" \
  >"${work}/pinned-artifacts-before.sha256"

common_cuda_env=(
  KLAYOUT_CUDA_SPATIAL_DEVICE=0
  KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
  KLAYOUT_CUDA_ACTIVE3=1
  KLAYOUT_CUDA_ACTIVE3_TELEMETRY=1
  KLAYOUT_CUDA_M1_WIDTH_SPACE=1
  KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1
  KLAYOUT_CUDA_M2_RULES=1
  KLAYOUT_CUDA_M2_RULES_TELEMETRY=1
  KLAYOUT_CUDA_M2_WIDTH_SPACE=0
  KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY=0
  KLAYOUT_CUDA_VIA1_STACK=1
  KLAYOUT_CUDA_VIA1_STACK_TELEMETRY=1
  KLAYOUT_CUDA_M1_CONTACT=1
  KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1
  KLAYOUT_CUDA_CONTACT4=1
  KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
  KLAYOUT_CUDA_IMPLANT12=1
  KLAYOUT_CUDA_IMPLANT12_TELEMETRY=1
  KLAYOUT_CUDA_DISCONNECTED_MERGE=1
  KLAYOUT_DEEP_EDGE_CERT_PROFILE=1
  KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=100000
  KLAYOUT_CUDA_SPATIAL_CELL_SIZE=512
  KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64
  KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096
  KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=80000000
  KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=100000000
  KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=30000000
)
printf '%s\n' "${common_cuda_env[@]}" |
  sort >"${work}/runtime/common-feature-env.txt"

run_lane() {
  local lane=$1
  local mode=$2
  local runtime="${work}/runtime/${lane}"
  local report="${work}/reports/${lane}.lyrdb"
  local canonical="${work}/reports/${lane}.canonical.lyrdb"
  local log="${work}/logs/${lane}.log"
  local time_file="${work}/timings/${lane}.txt"
  local before="${work}/runtime/${lane}-before-closure.sha256"
  local after="${work}/runtime/${lane}-after-closure.sha256"
  local baseline="${work}/runtime/${mode}-closure-baseline.sha256"
  local rc
  local actual_sha256

  select_mode "${mode}"
  mkdir -p -- \
    "${runtime}/home" \
    "${runtime}/klayout-home" \
    "${runtime}/xdg-config" \
    "${runtime}/xdg-cache" \
    "${runtime}/xdg-data" \
    "${runtime}/tmp"

  capture_runtime_closure "${lane}-before" "${mode}"
  if [[ -f "${baseline}" ]]; then
    cmp -s -- "${baseline}" "${before}" ||
      die "${lane}: ${mode} loader closure differs from its first replicate"
  else
    cp -- "${before}" "${baseline}"
  fi
  assert_gpu_idle "${lane}-before"

  printf '%s\t%s\t%s\t%s\n' \
    "${lane}" "${mode}" start "$(date -u +%FT%TZ)" \
    >>"${work}/run-events.tsv"
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
        "LD_LIBRARY_PATH=${selected_ld_library_path}" \
        "KLAYOUT_CUDA_SPATIAL_BACKEND=${selected_backend}" \
        "${common_cuda_env[@]}" \
        "${selected_klayout}" -b -r "${selected_deck}" \
          -rd "input=${input}" \
          -rd "topcell=${top_cell}" \
          -rd "output=${report}" \
          -rd drc_shard=m2_rules \
          >"${log}" 2>&1
  rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${lane}" "${mode}" finish "${rc}" "$(date -u +%FT%TZ)" \
    >>"${work}/run-events.tsv"
  if ((rc != 0)); then
    cat -- "${time_file}" >&2 || true
    tail -n 200 -- "${log}" >&2 || true
    die "${lane}: production M2 owner failed with status ${rc}"
  fi

  [[ -s "${report}" ]] ||
    die "${lane}: production M2 owner produced no report"
  canonicalize_report "${report}" "${canonical}"
  actual_sha256=$(file_sha256 "${canonical}")
  [[ "${actual_sha256}" == "${expected_report_sha256}" ]] ||
    die "${lane}: canonical report SHA-256 is ${actual_sha256}, expected ${expected_report_sha256}"
  cmp -s -- "${reference_canonical}" "${canonical}" ||
    die "${lane}: canonical report differs from the production reference"
  assert_m2_telemetry "${lane}" "${mode}"

  capture_runtime_closure "${lane}-after" "${mode}"
  cmp -s -- "${before}" "${after}" ||
    die "${lane}: runtime closure changed during execution"
  assert_gpu_idle "${lane}-after"
  sha256sum -- "${static_artifacts[@]}" \
    >"${work}/runtime/${lane}-pinned-artifacts-after.sha256"
  cmp -s -- \
    "${work}/pinned-artifacts-before.sha256" \
    "${work}/runtime/${lane}-pinned-artifacts-after.sha256" ||
    die "${lane}: a pinned executable, input, deck, or reference changed"
}

emit_order >"${work}/run-order.tsv"
while IFS=$'\t' read -r index lane mode replicate; do
  printf 'M2_RESIDENT_SUFFIX_PRODUCTION_OWNER_AB lane=%s/%s name=%s mode=%s replicate=%s\n' \
    "${index}" "$((2 * repetitions))" "${lane}" "${mode}" "${replicate}"
  run_lane "${lane}" "${mode}"
done <"${work}/run-order.tsv"

timings_tsv="${work}/timings/owner-wall.tsv"
printf 'replicate\tmode\twall_s\n' >"${timings_tsv}"
for ((replicate = 1; replicate <= repetitions; ++replicate)); do
  for mode in previous resident; do
    lane="${mode}-${replicate}"
    wall=$(
      sed -n 's/^wall_s=\([^ ]*\).*/\1/p' \
        "${work}/timings/${lane}.txt"
    )
    [[ "${wall}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
      die "${lane}: wall time is missing or malformed"
    printf '%s\t%s\t%s\n' \
      "${replicate}" "${mode}" "${wall}" >>"${timings_tsv}"
  done
done

summary="${work}/timings/owner-summary.txt"
awk -v expected_n="${repetitions}" '
  NR == 1 { next }
  $2 == "previous" { previous_sum += $3; ++previous_n }
  $2 == "resident" { resident_sum += $3; ++resident_n }
  END {
    if (previous_n != expected_n || resident_n != expected_n) {
      exit 2
    }
    previous_mean = previous_sum / previous_n
    resident_mean = resident_sum / resident_n
    delta = previous_mean - resident_mean
    less = previous_mean == 0 ? 0 : 100.0 * delta / previous_mean
    printf "M2_RESIDENT_SUFFIX_PRODUCTION_OWNER_AB owner=m2_rules scope=owner-only previous_accelerated_mean_wall_s=%.3f resident_suffix_mean_wall_s=%.3f delta_s=%.3f less_pct=%.3f N=%d alternating_order=1 report=exact\n",
      previous_mean, resident_mean, delta, less, expected_n
  }' "${timings_tsv}" >"${summary}" ||
  die "unable to summarize exactly N timings per lane"

canonical_hashes="${work}/canonical-report-sha256.txt"
sha256sum -- "${reference_canonical}" \
  "${work}"/reports/{previous,resident}-*.canonical.lyrdb \
  >"${canonical_hashes}"

sha256sum -- "${static_artifacts[@]}" \
  "${work}/run-order.tsv" \
  "${work}/run-events.tsv" \
  "${timings_tsv}" \
  "${summary}" \
  "${canonical_hashes}" \
  "${work}"/runtime/*-closure.sha256 \
  "${work}"/logs/*-compute-processes.csv \
  >"${work}/evidence.sha256"

telemetry="${work}/logs/m2-telemetry.txt"
grep -h -E -- \
  'CUDA M2 (exact union boundary:|live flat operands:|rules transaction:)' \
  "${work}"/logs/{previous,resident}-*.log \
  >"${telemetry}"

cat -- "${work}/run-order.tsv"
cat -- "${timings_tsv}"
cat -- "${summary}"
cat -- "${canonical_hashes}"
cat -- "${telemetry}"
echo \
  "M2_RESIDENT_SUFFIX_PRODUCTION_OWNER_GATE PASS owner=m2_rules scope=owner-only lanes=$((2 * repetitions)) N=${repetitions} serial=1 alternating-order=1 canonical-reports=exact runtime-closures=stable gpu-idle=each-lane evidence=${work}"
