#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_poly34_production_owner_gate.sh \
    --klayout PATH --backend PATH --device-smoke PATH --source-deck PATH \
    --default-deck-reference PATH --input PATH --top-cell NAME \
    [--python PATH] [--timeout-seconds N] [--generate-only] [--ab-only] \
    [--keep-work]

Generates the opt-in FreePDK45 POLY.3/.4 transaction deck and runs only the
production m1_enclosure owner. The first two runs are a same-binary,
same-generated-deck feature-off/live-CUDA A/B. Only after that exact report
differential passes, injected-Ruby-exception and forced-capacity lanes prove
that production declines retain both original CPU expressions.

The generator gate requires a pre-change default-deck reference. Generation
without --poly34 must remain byte-identical to that reference; repeated default
and opt-in generations must also be byte-identical. Every owner report is
compared after removing only KLayout's generator element.

All generated decks, reports, logs, homes, caches, timings, and hashes live
under a fresh TMPDIR directory. They are removed unless --keep-work is set.
EOF
}

die() {
  echo "POLY.3/.4 production owner gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
deck_generator="${here}/make_via1_stack_live_deck.py"
hook_probe="${here}/poly34_hook_presence.drc"

klayout=
backend=
device_smoke=
source_deck=
default_deck_reference=
input=
top_cell=
python=${PYTHON:-python3}
timeout_seconds=600
keep_work=0
generate_only=0
ab_only=0

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
    --keep-work)
      keep_work=1
      shift
      ;;
    --generate-only)
      generate_only=1
      shift
      ;;
    --ab-only)
      ab_only=1
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
if (( ! generate_only)); then
  [[ -n "${device_smoke}" ]] || die "missing --device-smoke"
fi
[[ -n "${source_deck}" ]] || die "missing --source-deck"
[[ -n "${default_deck_reference}" ]] ||
  die "missing --default-deck-reference"
[[ -n "${input}" ]] || die "missing --input"
[[ -n "${top_cell}" ]] || die "missing --top-cell"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] ||
  die "--timeout-seconds must be a positive integer"
[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${backend}" ]] || die "CUDA backend is missing: ${backend}"
if (( ! generate_only)); then
  [[ -x "${device_smoke}" ]] ||
    die "CUDA device smoke is not executable: ${device_smoke}"
fi
[[ -f "${source_deck}" ]] || die "source deck is missing: ${source_deck}"
[[ -f "${default_deck_reference}" ]] ||
  die "default deck reference is missing: ${default_deck_reference}"
[[ -s "${input}" ]] || die "input layout is missing or empty: ${input}"
[[ -f "${deck_generator}" ]] || die "deck generator is missing"
[[ -f "${hook_probe}" ]] || die "POLY.3/.4 hook probe is missing"
[[ -x /usr/bin/time ]] || die "/usr/bin/time is unavailable"
[[ -x /usr/bin/timeout ]] || die "/usr/bin/timeout is unavailable"

python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
if (( ! generate_only)); then
  device_smoke=$(readlink -f -- "${device_smoke}")
fi
source_deck=$(readlink -f -- "${source_deck}")
default_deck_reference=$(readlink -f -- "${default_deck_reference}")
input=$(readlink -f -- "${input}")
klayout_dir=$(dirname -- "${klayout}")
backend_dir=$(dirname -- "${backend}")
runtime_ld_library_path="${backend_dir}:${klayout_dir}"

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-poly34-production-owner.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "POLY34_PRODUCTION_OWNER_GATE work=${work}"
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
live_deck="${work}/decks/poly34-live.lydrc"
live_repeat="${work}/decks/poly34-live-repeat.lydrc"
exception_deck="${work}/decks/poly34-ruby-exception.lydrc"
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

run_generator "${live_deck}" --poly34
run_generator "${live_repeat}" --poly34
cmp -s -- "${live_deck}" "${live_repeat}" ||
  die "repeated POLY.3/.4 deck generation is not byte-identical"
run_generator \
  "${exception_deck}" --poly34 --inject-poly34-ruby-exception

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
  'poly34_request = ENV["KLAYOUT_CUDA_POLY34"].to_s' \
  "Ruby opt-in source"
assert_deck_count 1 \
  "poly.respond_to?(:cuda_poly34_clean?)" \
  "stock compatibility guard"
assert_deck_count 1 \
  "CUDA POLY.3/.4 transaction:" \
  "transaction telemetry"
assert_deck_count 1 \
  "rescue StandardError =&gt; error" \
  "Ruby exception guard"
assert_deck_count 1 \
  'poly34_empty.output("POLY.3"' \
  "POLY.3 certified-empty output"
assert_deck_count 1 \
  'poly34_empty.output("POLY.4"' \
  "POLY.4 certified-empty output"
assert_deck_count 1 \
  'poly.enclosing(gate, 55.nm, projection).polygons.without_area(0).output("POLY.3"' \
  "pristine POLY.3 CPU expression"
assert_deck_count 1 \
  'active.enclosing(gate, 70.nm, projection).polygons.without_area(0).output("POLY.4"' \
  "pristine POLY.4 CPU expression"

[[ "$(grep -Fc -- \
  'raise("injected POLY34 Ruby exception")' "${exception_deck}")" == 1 ]] ||
  die "injected-exception deck does not contain exactly one injected exception"
if grep -Fq -- \
  "poly.cuda_poly34_clean?(active, gate)" "${exception_deck}"; then
  die "injected-exception deck retained the live hook call"
fi
for expression in \
  'poly.enclosing(gate, 55.nm, projection).polygons.without_area(0).output("POLY.3"' \
  'active.enclosing(gate, 70.nm, projection).polygons.without_area(0).output("POLY.4"'; do
  [[ "$(grep -Fc -- "${expression}" "${exception_deck}")" == 1 ]] ||
    die "injected-exception deck changed a pristine CPU expression"
done

mutated_source="${work}/decks/mutated-source.lydrc"
mutated_output="${work}/decks/mutated-output.lydrc"
mutation_log="${work}/logs/deck-mutation.log"
sed \
  's/active\.enclosing(gate, 70\.nm, projection)/active.enclosing(gate, 71.nm, projection)/' \
  "${source_deck}" >"${mutated_source}"
cmp -s -- "${source_deck}" "${mutated_source}" &&
  die "POLY.4 mutation did not change the source deck"
if "${python}" "${deck_generator}" \
  --input "${mutated_source}" --output "${mutated_output}" \
  --m1-contact --implant12 --poly34 >"${mutation_log}" 2>&1; then
  die "changed POLY.4 rule was incorrectly accepted"
fi
grep -Fq -- \
  "POLY.3/.4 transaction: expected one source block, found 0" \
  "${mutation_log}" ||
  {
    cat -- "${mutation_log}" >&2
    die "changed POLY.4 rule did not fail at the exact matcher"
  }

echo \
  "POLY34_PRODUCTION_OWNER_GATE ok gate=generator default-byte-identical=1 repeatable=1 exact-matcher=1 ruby-rescue=1 injected-exception=1 pristine-cpu=2 ordered-empty-outputs=2"

if ((generate_only)); then
  sha256sum -- \
    "${source_deck}" \
    "${default_deck_reference}" \
    "${default_deck}" \
    "${live_deck}" \
    "${exception_deck}" \
    "${deck_generator}" \
    "$0" \
    >"${work}/pinned-artifacts.sha256"
  cat -- "${work}/pinned-artifacts.sha256"
  echo "POLY34_PRODUCTION_OWNER_GATE PASS gate=generate-only"
  exit 0
fi

canonicalize_report() {
  local report=$1
  local output=$2
  sed '/<generator>/d' "${report}" >"${output}"
}

run_host_hook_preflight() {
  local runtime="${work}/runtime/hook-probe"
  local log="${work}/logs/hook-probe.log"
  mkdir -p -- \
    "${runtime}/home" "${runtime}/config" "${runtime}/cache" "${runtime}/tmp"
  if ! env -i \
    "HOME=${runtime}/home" \
    "KLAYOUT_HOME=${runtime}/home" \
    "XDG_CONFIG_HOME=${runtime}/config" \
    "XDG_CACHE_HOME=${runtime}/cache" \
    "TMPDIR=${runtime}/tmp" \
    PATH=/usr/bin:/bin \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
    QT_QPA_PLATFORM=offscreen \
    "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
    "${klayout}" -b -r "${hook_probe}" >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "loaded KLayout runtime failed the POLY.3/.4 hook probe"
  fi
  grep -Fq -- "POLY34_HOOK_PRESENCE ok" "${log}" ||
    die "POLY.3/.4 hook probe completion marker is missing"
  echo \
    "POLY34_PRODUCTION_OWNER_GATE ok gate=host-hook-preflight runtime=bound"
}

capture_runtime_closure() {
  local lane=$1
  local prefix="${work}/runtime/${lane}-closure"
  local combined="${prefix}.ldd"
  local paths="${prefix}.paths"
  local hashes="${prefix}.sha256"
  local loaded_backend
  local loaded_db

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
    die "${lane}: runtime closure has an unresolved dependency"
  fi

  loaded_backend=$(
    env -i \
      PATH=/usr/bin:/bin \
      "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
      /usr/bin/ldd "${device_smoke}" |
      awk \
        '$1 == "libklayout_cuda_spatial_backend.so" && $2 == "=>" {
           print $3
           exit
         }'
  )
  [[ -n "${loaded_backend}" ]] ||
    die "${lane}: smoke has no loader-selected CUDA backend"
  [[ "$(readlink -f -- "${loaded_backend}")" == "${backend}" ]] ||
    die "${lane}: smoke selected ${loaded_backend}, expected ${backend}"
  /usr/bin/nm -D "${backend}" >"${prefix}.backend-nm"
  grep -Fq -- \
    "klayout_cuda_spatial_run_poly34_empty_v1" \
    "${prefix}.backend-nm" ||
    die "${lane}: selected backend has no POLY.3/.4 entry point"

  loaded_db=$(
    env -i \
      PATH=/usr/bin:/bin \
      "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
      /usr/bin/ldd "${klayout}" |
      awk \
        '$1 ~ /^libklayout_db\.so/ && $2 == "=>" {
           print $3
           exit
         }'
  )
  [[ -n "${loaded_db}" ]] ||
    die "${lane}: KLayout has no loader-selected libklayout_db"
  [[ "$(dirname -- "$(readlink -f -- "${loaded_db}")")" == \
      "${klayout_dir}" ]] ||
    die "${lane}: KLayout selected ${loaded_db} outside ${klayout_dir}"

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
      die "${lane}: resolved closure member is not a file: ${artifact}"
    sha256sum -- "${artifact}"
  done <"${paths}" >"${hashes}"
}

run_device_preflight() {
  local lane=$1
  local log="${work}/logs/${lane}-device-preflight.log"
  local compute_log="${work}/logs/${lane}-preexisting-compute.log"
  if ! nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"${compute_log}" 2>&1; then
    cat -- "${compute_log}" >&2
    die "${lane}: unable to census pre-existing CUDA processes"
  fi
  if grep -Eq -- '^[[:space:]]*[0-9]+' "${compute_log}"; then
    cat -- "${compute_log}" >&2
    die "${lane}: another CUDA compute process would contaminate timing"
  fi
  if ! nvidia-smi \
    --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader >"${log}" 2>&1; then
    cat -- "${log}" >&2
    die "${lane}: nvidia-smi device preflight failed"
  fi
  capture_runtime_closure "${lane}"
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
    die "${lane}: CUDA backend device smoke failed"
  fi
  grep -Fq -- "CUDA backend ABI smoke passed:" "${log}" ||
    {
      cat -- "${log}" >&2
      die "${lane}: CUDA backend smoke completion marker is missing"
    }
  echo \
    "POLY34_PRODUCTION_OWNER_GATE ok gate=device-preflight lane=${lane} device=0 runtime-closure=bound gpu-idle=1"
}

run_lane() {
  local lane=$1
  local mode=$2
  local lane_deck="${live_deck}"
  local runtime="${work}/runtime/${lane}"
  local report="${work}/reports/${lane}.lyrdb"
  local log="${work}/logs/${lane}.log"
  local time_file="${work}/timings/${lane}.txt"
  local -a cuda_env=(
    "KLAYOUT_CUDA_SPATIAL_BACKEND=${backend}"
    KLAYOUT_CUDA_ACTIVE3=1
    KLAYOUT_CUDA_ACTIVE3_TELEMETRY=1
    KLAYOUT_CUDA_M1_CONTACT=1
    KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1
  )

  mkdir -p -- \
    "${runtime}/home" \
    "${runtime}/klayout-home" \
    "${runtime}/xdg-config" \
    "${runtime}/xdg-cache" \
    "${runtime}/xdg-data" \
    "${runtime}/tmp"

  case "${mode}" in
    control)
      cuda_env+=(
        KLAYOUT_CUDA_POLY34=0
        KLAYOUT_CUDA_POLY34_TELEMETRY=0
      )
      ;;
    cuda)
      cuda_env+=(
        KLAYOUT_CUDA_POLY34=1
        KLAYOUT_CUDA_POLY34_TELEMETRY=1
      )
      ;;
    exception)
      lane_deck="${exception_deck}"
      cuda_env+=(
        KLAYOUT_CUDA_POLY34=1
        KLAYOUT_CUDA_POLY34_TELEMETRY=1
      )
      ;;
    capacity)
      cuda_env+=(
        KLAYOUT_CUDA_POLY34=1
        KLAYOUT_CUDA_POLY34_TELEMETRY=1
        KLAYOUT_CUDA_POLY34_MAX_GRID_CELLS=1
      )
      ;;
    *)
      die "internal error: unknown lane mode ${mode}"
      ;;
  esac

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
        QT_QPA_PLATFORM=offscreen \
        CUDA_VISIBLE_DEVICES=0 \
        KLAYOUT_CUDA_SPATIAL_DEVICE=0 \
        KLAYOUT_CUDA_SPATIAL_TELEMETRY=1 \
        "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
        KLAYOUT_CUDA_DISCONNECTED_MERGE=1 \
        KLAYOUT_DEEP_EDGE_CERT_PROFILE=1 \
        KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=100000 \
        KLAYOUT_CUDA_SPATIAL_CELL_SIZE=512 \
        KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64 \
        KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096 \
        KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=80000000 \
        KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=100000000 \
        KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=30000000 \
        "${cuda_env[@]}" \
        "${klayout}" -b \
          -r "${lane_deck}" \
          -rd "input=${input}" \
          -rd "topcell=${top_cell}" \
          -rd "output=${report}" \
          -rd drc_shard=m1_enclosure \
          >"${log}" 2>&1
  local rc=$?
  set -e
  if ((rc != 0)); then
    cat -- "${time_file}" >&2 || true
    tail -n 200 -- "${log}" >&2 || true
    die "${lane}: production owner failed with status ${rc}"
  fi
  [[ -s "${report}" ]] || die "${lane}: production owner produced no report"
  canonicalize_report \
    "${report}" "${work}/reports/${lane}.canonical.lyrdb"
  grep -Fq -- "<top-cell>${top_cell}</top-cell>" "${report}" ||
    die "${lane}: report top-cell marker is missing"
}

assert_same_report() {
  local candidate=$1
  cmp -s -- \
    "${work}/reports/control.canonical.lyrdb" \
    "${work}/reports/${candidate}.canonical.lyrdb" ||
    {
      sha256sum -- \
        "${work}/reports/control.canonical.lyrdb" \
        "${work}/reports/${candidate}.canonical.lyrdb" >&2
      die "${candidate}: report differs from the feature-off control"
    }
}

assert_transaction() {
  local lane=$1
  local disposition=$2
  local log="${work}/logs/${lane}.log"
  local count
  count=$(grep -Fc -- "CUDA POLY.3/.4 transaction:" "${log}" || true)
  [[ "${count}" == 1 ]] ||
    die "${lane}: expected one transaction marker, found ${count}"
  grep -Fq -- \
    "CUDA POLY.3/.4 transaction: ${disposition}" "${log}" ||
    die "${lane}: expected transaction disposition ${disposition}"
}

# The requested timing comparison is deliberately first and serial.
run_host_hook_preflight
run_device_preflight control
run_lane control control
run_device_preflight cuda
cmp -s -- \
  "${work}/runtime/control-closure.sha256" \
  "${work}/runtime/cuda-closure.sha256" ||
  die "control and CUDA lanes selected different runtime closures"
run_lane cuda cuda
assert_same_report cuda
if grep -Fq -- "CUDA POLY.3/.4" "${work}/logs/control.log"; then
  die "feature-off control unexpectedly invoked POLY.3/.4 CUDA"
fi
assert_transaction cuda certified-empty
cuda_certificate_count=$(
  grep -Fc -- \
    "CUDA POLY.3/.4 terminal-empty certificate:" \
    "${work}/logs/cuda.log" || true
)
[[ "${cuda_certificate_count}" == 1 ]] ||
  die "CUDA lane expected one terminal certificate, found ${cuda_certificate_count}"
grep -Fq -- \
  "CUDA POLY.3/.4 terminal-empty certificate: outcome=certified-empty contexts=849265 poly_boxes=5013998 active_boxes=2684780 gates=3401254 certified_mask=3 poly_candidates=4462594 active_candidates=3403326 atomic_empty=3401254 fallback_gates=0" \
  "${work}/logs/cuda.log" ||
  die "CUDA lane did not report the exact production certificate census"
grep -Fq -- \
  "fallback_flags=0 device_flags=0 message=complete atomic POLY.3/.4 terminal-empty certificate" \
  "${work}/logs/cuda.log" ||
  die "CUDA lane terminal certificate reported fallback or device flags"
cuda_lowering_count=$(
  grep -Fc -- "CUDA POLY.3/.4 live lowering:" "${work}/logs/cuda.log" || true
)
[[ "${cuda_lowering_count}" == 1 ]] ||
  die "CUDA lane expected one live-lowering marker, found ${cuda_lowering_count}"
for lane in control cuda; do
  grep -Fq -- \
    "CUDA spatial backend loaded: ${backend}" \
    "${work}/logs/${lane}.log" ||
    die "${lane}: exact CUDA backend load marker is missing"
  grep -Fq -- \
    "CUDA ACTIVE.3 empty certificate: outcome=certified-empty" \
    "${work}/logs/${lane}.log" ||
    die "${lane}: unrelated ACTIVE.3 CUDA transaction did not certify"
  grep -Fq -- \
    "CUDA M1 contact transaction: certified-empty" \
    "${work}/logs/${lane}.log" ||
    die "${lane}: unrelated M1-contact CUDA transaction did not certify"
  if grep -Fq -- \
    "cudaSetDevice: no CUDA-capable device" "${work}/logs/${lane}.log"; then
    die "${lane}: CUDA device disappeared after the successful preflight"
  fi
  if grep -Eq -- \
    'CUDA .*outcome=(cpu-fallback|error|failed|declined)' \
    "${work}/logs/${lane}.log"; then
    die "${lane}: CUDA telemetry reported an error or fallback outcome"
  fi
done

control_wall=$(
  sed -n 's/^wall_s=\([^ ]*\).*/\1/p' \
    "${work}/timings/control.txt"
)
cuda_wall=$(
  sed -n 's/^wall_s=\([^ ]*\).*/\1/p' \
    "${work}/timings/cuda.txt"
)
[[ -n "${control_wall}" ]] || die "control wall time is missing"
[[ -n "${cuda_wall}" ]] || die "CUDA wall time is missing"
awk -v control="${control_wall}" -v cuda="${cuda_wall}" \
  'BEGIN {
    delta = control - cuda
    percent = control == 0 ? 0 : 100 * delta / control
    printf "POLY34_PRODUCTION_OWNER_AB control_wall_s=%.2f cuda_wall_s=%.2f delta_s=%.2f reduction_pct=%.3f report=exact\n", control, cuda, delta, percent
  }'

if ((ab_only)); then
  sha256sum -- \
    "${klayout}" \
    "${backend}" \
    "${device_smoke}" \
    "${hook_probe}" \
    "${source_deck}" \
    "${default_deck_reference}" \
    "${default_deck}" \
    "${live_deck}" \
    "${exception_deck}" \
    "${input}" \
    "${work}/reports/control.canonical.lyrdb" \
    "${work}/reports/cuda.canonical.lyrdb" \
    "${work}/runtime/control-closure.sha256" \
    "${work}/runtime/cuda-closure.sha256" \
    "${work}/logs/control-device-preflight.log" \
    "${work}/logs/cuda-device-preflight.log" \
    "${work}/logs/hook-probe.log" \
    "${deck_generator}" \
    "$0" \
    >"${work}/pinned-artifacts.sha256"
  sha256sum -- \
    "${work}/reports/control.canonical.lyrdb" \
    "${work}/reports/cuda.canonical.lyrdb" \
    >"${work}/canonical-report-sha256.txt"
  cat -- "${work}/timings/control.txt"
  cat -- "${work}/timings/cuda.txt"
  cat -- "${work}/canonical-report-sha256.txt"
  grep -h -E -- \
    'CUDA POLY\.3/\.4 transaction:|CUDA POLY\.3/\.4 live lowering:|CUDA M1 contact transaction:' \
    "${work}/logs/control.log" \
    "${work}/logs/cuda.log"
  echo \
    "POLY34_PRODUCTION_OWNER_GATE PASS owner=m1_enclosure lanes=2 ab-first=1 canonical-reports=exact default-byte-identical=1 device-preflight=each-lane"
  exit 0
fi

# Only after the production A/B passes do fallback lanes consume the owner.
run_device_preflight exception
cmp -s -- \
  "${work}/runtime/control-closure.sha256" \
  "${work}/runtime/exception-closure.sha256" ||
  die "injected-exception lane selected a different runtime closure"
run_lane exception exception
assert_same_report exception
assert_transaction exception full-cpu-fallback
grep -Fq -- \
  "CUDA spatial backend loaded: ${backend}" \
  "${work}/logs/exception.log" ||
  die "exception: exact CUDA backend load marker is missing"
grep -Fq -- \
  "CUDA ACTIVE.3 empty certificate: outcome=certified-empty" \
  "${work}/logs/exception.log" ||
  die "exception: unrelated ACTIVE.3 CUDA transaction did not certify"
grep -Fq -- \
  "CUDA M1 contact transaction: certified-empty" \
  "${work}/logs/exception.log" ||
  die "exception: unrelated M1-contact CUDA transaction did not certify"
if grep -Eq -- \
  'cudaSetDevice:|CUDA .*outcome=(error|failed|declined)' \
  "${work}/logs/exception.log"; then
  die "exception: CUDA telemetry reported a device or runtime error"
fi
grep -Fq -- \
  "CUDA POLY.3/.4 Ruby fallback: RuntimeError: injected POLY34 Ruby exception" \
  "${work}/logs/exception.log" ||
  die "injected Ruby exception did not reach the fail-closed deck fallback"
if grep -Fq -- \
  "CUDA POLY.3/.4 live lowering:" "${work}/logs/exception.log"; then
  die "injected Ruby exception unexpectedly reached live lowering"
fi

run_device_preflight capacity
cmp -s -- \
  "${work}/runtime/control-closure.sha256" \
  "${work}/runtime/capacity-closure.sha256" ||
  die "forced-capacity lane selected a different runtime closure"
run_lane capacity capacity
assert_same_report capacity
assert_transaction capacity full-cpu-fallback
grep -Fq -- \
  "CUDA spatial backend loaded: ${backend}" \
  "${work}/logs/capacity.log" ||
  die "capacity: exact CUDA backend load marker is missing"
grep -Fq -- \
  "CUDA ACTIVE.3 empty certificate: outcome=certified-empty" \
  "${work}/logs/capacity.log" ||
  die "capacity: unrelated ACTIVE.3 CUDA transaction did not certify"
grep -Fq -- \
  "CUDA M1 contact transaction: certified-empty" \
  "${work}/logs/capacity.log" ||
  die "capacity: unrelated M1-contact CUDA transaction did not certify"
if grep -Eq -- \
  'cudaSetDevice:|CUDA .*outcome=(error|failed|declined)' \
  "${work}/logs/capacity.log"; then
  die "capacity: CUDA telemetry reported a device or runtime error"
fi
capacity_certificate_count=$(
  grep -Fc -- \
    "CUDA POLY.3/.4 terminal-empty certificate:" \
    "${work}/logs/capacity.log" || true
)
[[ "${capacity_certificate_count}" == 1 ]] ||
  die "capacity: expected one terminal certificate, found ${capacity_certificate_count}"
grep -Fq -- \
  "CUDA POLY.3/.4 terminal-empty certificate: outcome=fallback contexts=849265 poly_boxes=5013998 active_boxes=2684780 gates=3401254 certified_mask=0 poly_candidates=0 active_candidates=0 atomic_empty=0 fallback_gates=0" \
  "${work}/logs/capacity.log" ||
  die "forced-capacity lane did not report the exact production fallback census"
grep -Fq -- \
  "fallback_flags=8 device_flags=0 message=POLY34 pipeline requested CPU fallback" \
  "${work}/logs/capacity.log" ||
  die "forced-capacity lane did not exercise the expected backend capacity decline"
capacity_lowering_count=$(
  grep -Fc -- \
    "CUDA POLY.3/.4 live lowering:" "${work}/logs/capacity.log" || true
)
[[ "${capacity_lowering_count}" == 1 ]] ||
  die "capacity: expected one live-lowering marker, found ${capacity_lowering_count}"
if grep -Fq -- \
  "CUDA POLY.3/.4 terminal-empty certificate: outcome=certified-empty" \
  "${work}/logs/capacity.log"; then
  die "forced-capacity lane unexpectedly certified the transaction empty"
fi

sha256sum -- \
  "${klayout}" \
  "${backend}" \
  "${device_smoke}" \
  "${hook_probe}" \
  "${source_deck}" \
  "${default_deck_reference}" \
  "${default_deck}" \
  "${live_deck}" \
  "${exception_deck}" \
  "${input}" \
  "${work}/reports/control.canonical.lyrdb" \
  "${work}/reports/cuda.canonical.lyrdb" \
  "${work}/reports/exception.canonical.lyrdb" \
  "${work}/reports/capacity.canonical.lyrdb" \
  "${work}/runtime/control-closure.sha256" \
  "${work}/runtime/cuda-closure.sha256" \
  "${work}/runtime/exception-closure.sha256" \
  "${work}/runtime/capacity-closure.sha256" \
  "${work}/logs/control-device-preflight.log" \
  "${work}/logs/cuda-device-preflight.log" \
  "${work}/logs/exception-device-preflight.log" \
  "${work}/logs/capacity-device-preflight.log" \
  "${work}/logs/hook-probe.log" \
  "${deck_generator}" \
  "$0" \
  >"${work}/pinned-artifacts.sha256"

sha256sum -- \
  "${work}/reports/control.canonical.lyrdb" \
  "${work}/reports/cuda.canonical.lyrdb" \
  "${work}/reports/exception.canonical.lyrdb" \
  "${work}/reports/capacity.canonical.lyrdb" \
  >"${work}/canonical-report-sha256.txt"

cat -- "${work}/timings/control.txt"
cat -- "${work}/timings/cuda.txt"
cat -- "${work}/timings/exception.txt"
cat -- "${work}/timings/capacity.txt"
cat -- "${work}/canonical-report-sha256.txt"
grep -h -E -- \
  'CUDA POLY\.3/\.4 transaction:|CUDA POLY\.3/\.4 live lowering:' \
  "${work}/logs/cuda.log" \
  "${work}/logs/exception.log" \
  "${work}/logs/capacity.log"
echo \
  "POLY34_PRODUCTION_OWNER_GATE PASS owner=m1_enclosure lanes=4 ab-first=1 canonical-reports=exact default-byte-identical=1"
