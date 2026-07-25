#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_contact4_active_union_production_owner_gate.sh \
    --klayout PATH --backend PATH --deck PATH --input PATH \
    --top-cell NAME --reference PATH --expected-report-sha256 HEX \
    [--timeout-seconds N] [--keep-work]

Runs a serial, order-balanced N=3 A/B of the FreePDK45 implant_contact
production owner. Every lane uses the same executable, CUDA backend, deck,
input, top cell, reference, CUDA feature set, resource limits, and isolated
env -i runtime. The sole feature difference is:

  control:   KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=0
  candidate: KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=1

The six-lane order is control/candidate, candidate/control,
control/candidate. Each generator-stripped report must be byte-identical to
the reference and have the explicitly supplied SHA-256. Strict telemetry
proves that control used the established merged-ACTIVE CONTACT.4 certificate,
while candidate used exactly one fused raw-ACTIVE-union/CONTACT.4 transaction
and skipped both established CONTACT.4 fallbacks.

This gate reports the implant_contact owner delta only. It does not claim a
whole-launcher or critical-path reduction.

All reports, logs, homes, timings, loader evidence, and hashes live under a
fresh TMPDIR directory. They are removed unless --keep-work is supplied.
EOF
}

die() {
  echo "CONTACT.4 ACTIVE-union production owner gate: $*" >&2
  exit 2
}

script_path=$(readlink -f -- "${BASH_SOURCE[0]}")

klayout=
backend=
deck=
input=
top_cell=
reference=
expected_report_sha256=
timeout_seconds=900
keep_work=0

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
    --deck)
      (($# >= 2)) || die "--deck requires a value"
      deck=$2
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
    --timeout-seconds)
      (($# >= 2)) || die "--timeout-seconds requires a value"
      timeout_seconds=$2
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
[[ -n "${backend}" ]] || die "missing --backend"
[[ -n "${deck}" ]] || die "missing --deck"
[[ -n "${input}" ]] || die "missing --input"
[[ -n "${top_cell}" ]] || die "missing --top-cell"
[[ -n "${reference}" ]] || die "missing --reference"
[[ -n "${expected_report_sha256}" ]] ||
  die "missing --expected-report-sha256"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] ||
  die "--timeout-seconds must be a positive integer"
[[ "${expected_report_sha256}" =~ ^[0-9A-Fa-f]{64}$ ]] ||
  die "--expected-report-sha256 must be exactly 64 hexadecimal digits"

[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${backend}" ]] || die "CUDA backend is missing: ${backend}"
[[ -f "${deck}" ]] || die "DRC deck is missing: ${deck}"
[[ -s "${input}" ]] || die "input layout is missing or empty: ${input}"
[[ -s "${reference}" ]] ||
  die "reference report is missing or empty: ${reference}"
[[ -x /usr/bin/time ]] || die "/usr/bin/time is unavailable"
[[ -x /usr/bin/timeout ]] || die "/usr/bin/timeout is unavailable"
[[ -x /usr/bin/ldd ]] || die "/usr/bin/ldd is unavailable"
[[ -x /usr/bin/nm ]] || die "/usr/bin/nm is unavailable"
[[ -x /usr/bin/nvidia-smi ]] || die "/usr/bin/nvidia-smi is unavailable"

klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
deck=$(readlink -f -- "${deck}")
input=$(readlink -f -- "${input}")
reference=$(readlink -f -- "${reference}")
expected_report_sha256=$(
  printf '%s' "${expected_report_sha256}" | tr 'A-F' 'a-f'
)
klayout_dir=$(dirname -- "${klayout}")
backend_dir=$(dirname -- "${backend}")
runtime_ld_library_path="${backend_dir}:${klayout_dir}"

work=$(mktemp -d \
  "${TMPDIR:-/tmp}/klayout-contact4-active-union-owner.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "CONTACT4_ACTIVE_UNION_PRODUCTION_OWNER_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mkdir -p -- \
  "${work}/logs" \
  "${work}/reports" \
  "${work}/runtime" \
  "${work}/timings"

canonicalize_report() {
  local source=$1
  local output=$2
  sed '/<generator>/d' "${source}" >"${output}"
}

file_sha256() {
  sha256sum -- "$1" | awk '{print $1}'
}

reference_canonical="${work}/reports/reference.canonical.lyrdb"
canonicalize_report "${reference}" "${reference_canonical}"
reference_canonical_sha256=$(file_sha256 "${reference_canonical}")
[[ "${reference_canonical_sha256}" == "${expected_report_sha256}" ]] ||
  die "reference canonical SHA-256 is ${reference_canonical_sha256}, expected ${expected_report_sha256}"

loader_closure="${work}/runtime/klayout.ldd"
if ! env -i \
  PATH=/usr/bin:/bin \
  "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
  /usr/bin/ldd "${klayout}" >"${loader_closure}" 2>&1; then
  cat -- "${loader_closure}" >&2
  die "unable to resolve the selected KLayout runtime closure"
fi
if grep -Fq -- "not found" "${loader_closure}"; then
  cat -- "${loader_closure}" >&2
  die "selected KLayout runtime closure has an unresolved dependency"
fi

loaded_db=$(
  awk \
    '$1 ~ /^libklayout_db\.so/ && $2 == "=>" {
       print $3
       exit
     }' \
    "${loader_closure}"
)
loaded_drc=$(
  awk \
    '$1 ~ /^libklayout_drc\.so/ && $2 == "=>" {
       print $3
       exit
     }' \
    "${loader_closure}"
)
[[ -n "${loaded_db}" ]] ||
  die "KLayout loader closure contains no libklayout_db"
[[ -n "${loaded_drc}" ]] ||
  die "KLayout loader closure contains no libklayout_drc"
loaded_db=$(readlink -f -- "${loaded_db}")
loaded_drc=$(readlink -f -- "${loaded_drc}")
[[ "$(dirname -- "${loaded_db}")" == "${klayout_dir}" ]] ||
  die "KLayout selected ${loaded_db}, outside executable directory ${klayout_dir}"
[[ "$(dirname -- "${loaded_drc}")" == "${klayout_dir}" ]] ||
  die "KLayout selected ${loaded_drc}, outside executable directory ${klayout_dir}"

backend_symbols="${work}/runtime/backend-symbols.txt"
/usr/bin/nm -D --defined-only "${backend}" >"${backend_symbols}"
required_backend_symbols=(
  klayout_cuda_spatial_abi_version
  klayout_cuda_spatial_run_active3_empty_v1
  klayout_cuda_spatial_run_contact4_raw_active_empty_v1
  klayout_cuda_spatial_run_contact4_active_union_empty_v1
  klayout_cuda_spatial_run_implant12_empty_v1
  klayout_cuda_spatial_run_m1_width_space_empty_v1
  klayout_cuda_spatial_run_m2_width_space_empty_v1
  klayout_cuda_spatial_run_m2_union_boundary_v1
  klayout_cuda_spatial_run_poly34_empty_v1
  klayout_cuda_spatial_run_via1_stack_empty_v1
)
for symbol in "${required_backend_symbols[@]}"; do
  grep -Eq -- "[[:space:]]${symbol}$" "${backend_symbols}" ||
    die "selected CUDA backend does not export ${symbol}"
done

core_artifacts=(
  "${script_path}"
  "${klayout}"
  "${loaded_db}"
  "${loaded_drc}"
  "${backend}"
  "${deck}"
  "${input}"
  "${reference}"
)

snapshot_core_artifacts() {
  local output=$1
  sha256sum -- "${core_artifacts[@]}" >"${output}"
}

core_before="${work}/runtime/core-artifacts-before.sha256"
snapshot_core_artifacts "${core_before}"

common_cuda_env=(
  "KLAYOUT_CUDA_SPATIAL_BACKEND=${backend}"
  KLAYOUT_CUDA_SPATIAL_DEVICE=0
  KLAYOUT_CUDA_SPATIAL_TELEMETRY=1
  KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=100000
  KLAYOUT_CUDA_SPATIAL_CELL_SIZE=512
  KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64
  KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096
  KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=80000000
  KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=100000000
  KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=30000000
  KLAYOUT_CUDA_ACTIVE3=1
  KLAYOUT_CUDA_ACTIVE3_TELEMETRY=1
  KLAYOUT_CUDA_M1_WIDTH_SPACE=1
  KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1
  KLAYOUT_CUDA_M2_RULES=1
  KLAYOUT_CUDA_M2_RULES_TELEMETRY=1
  KLAYOUT_CUDA_M2_WIDTH_SPACE=0
  KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY=0
  KLAYOUT_CUDA_POLY34=1
  KLAYOUT_CUDA_POLY34_TELEMETRY=1
  KLAYOUT_CUDA_VIA1_STACK=1
  KLAYOUT_CUDA_VIA1_STACK_TELEMETRY=1
  KLAYOUT_CUDA_M1_CONTACT=1
  KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1
  KLAYOUT_CUDA_CONTACT4=1
  KLAYOUT_CUDA_CONTACT4_TELEMETRY=1
  KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE=0
  KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY=0
  KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_TELEMETRY=1
  KLAYOUT_CUDA_IMPLANT12=1
  KLAYOUT_CUDA_IMPLANT12_TELEMETRY=1
  KLAYOUT_CUDA_DISCONNECTED_MERGE=1
  KLAYOUT_DEEP_EDGE_CERT_PROFILE=1
)

write_feature_manifest() {
  local active_union=$1
  local output=$2
  {
    printf '%s\n' "${common_cuda_env[@]}"
    printf 'KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=%s\n' "${active_union}"
  } | sort >"${output}"
}

control_feature_manifest="${work}/runtime/control-feature-env.txt"
candidate_feature_manifest="${work}/runtime/candidate-feature-env.txt"
write_feature_manifest 0 "${control_feature_manifest}"
write_feature_manifest 1 "${candidate_feature_manifest}"
sed \
  's/^KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=[01]$/KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=<lane>/' \
  "${control_feature_manifest}" \
  >"${work}/runtime/control-feature-env.normalized.txt"
sed \
  's/^KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=[01]$/KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=<lane>/' \
  "${candidate_feature_manifest}" \
  >"${work}/runtime/candidate-feature-env.normalized.txt"
cmp -s -- \
  "${work}/runtime/control-feature-env.normalized.txt" \
  "${work}/runtime/candidate-feature-env.normalized.txt" ||
  die "control and candidate feature environments differ beyond ACTIVE_UNION"
[[ "$(grep -Fc -- \
  "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=0" \
  "${control_feature_manifest}")" == 1 ]] ||
  die "control feature manifest does not contain exactly one ACTIVE_UNION=0"
[[ "$(grep -Fc -- \
  "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=1" \
  "${candidate_feature_manifest}")" == 1 ]] ||
  die "candidate feature manifest does not contain exactly one ACTIVE_UNION=1"

assert_gpu_idle() {
  local lane=$1
  local census="${work}/logs/${lane}-preexisting-compute.txt"
  if ! /usr/bin/nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"${census}" 2>&1; then
    cat -- "${census}" >&2
    die "${lane}: unable to census pre-existing CUDA processes"
  fi
  if grep -Eq -- '^[[:space:]]*[0-9]+' "${census}"; then
    cat -- "${census}" >&2
    die "${lane}: another CUDA compute process would contaminate timing"
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

fused_certificate_marker="CUDA CONTACT.4 fused ACTIVE-union empty certificate:"
fused_lowering_marker="CUDA CONTACT.4 fused ACTIVE-union live lowering:"
late_certificate_marker="CUDA CONTACT.4 empty certificate:"
late_lowering_marker="CUDA CONTACT.4 live lowering:"
raw_active_marker="CUDA CONTACT.4 raw-ACTIVE"

assert_common_telemetry() {
  local lane=$1
  local log="${work}/logs/${lane}.log"

  assert_marker_count \
    "${log}" "CUDA spatial backend loaded: ${backend}" 1 "${lane}"
  assert_marker_count \
    "${log}" "CUDA M1 contact transaction: certified-empty" 1 "${lane}"
  assert_marker_count \
    "${log}" "CUDA IMPLANT.1/.2 transaction: certified-empty" 1 "${lane}"
  assert_marker_count "${log}" "${raw_active_marker}" 0 "${lane}"
  if grep -Fq -- "cudaSetDevice: no CUDA-capable device" "${log}"; then
    die "${lane}: CUDA device disappeared after the idle preflight"
  fi
}

assert_control_telemetry() {
  local lane=$1
  local log="${work}/logs/${lane}.log"
  local certificate

  assert_common_telemetry "${lane}"
  assert_marker_count "${log}" "${fused_certificate_marker}" 0 "${lane}"
  assert_marker_count "${log}" "${fused_lowering_marker}" 0 "${lane}"
  assert_marker_count "${log}" "${late_certificate_marker}" 1 "${lane}"
  assert_marker_count "${log}" "${late_lowering_marker}" 1 "${lane}"
  certificate=$(grep -F -- "${late_certificate_marker}" "${log}")
  [[ "${certificate}" == *"outcome=certified-empty"* ]] ||
    die "${lane}: established CONTACT.4 certificate did not certify empty"
  [[ "${certificate}" == *"raw_hits=0 uncertain=0"* ]] ||
    die "${lane}: established CONTACT.4 certificate reported a hit or uncertainty"
  [[ "${certificate}" == *"fallback_flags=0 device_flags=0"* ]] ||
    die "${lane}: established CONTACT.4 certificate reported fallback/device flags"
}

assert_candidate_telemetry() {
  local lane=$1
  local log="${work}/logs/${lane}.log"
  local certificate
  local lowering

  assert_common_telemetry "${lane}"
  assert_marker_count "${log}" "${fused_certificate_marker}" 1 "${lane}"
  assert_marker_count "${log}" "${fused_lowering_marker}" 1 "${lane}"
  assert_marker_count "${log}" "${late_certificate_marker}" 0 "${lane}"
  assert_marker_count "${log}" "${late_lowering_marker}" 0 "${lane}"

  certificate=$(grep -F -- "${fused_certificate_marker}" "${log}")
  lowering=$(grep -F -- "${fused_lowering_marker}" "${log}")
  [[ "${certificate}" == *"outcome=certified-empty"* ]] ||
    die "${lane}: fused CONTACT.4 transaction did not certify empty"
  [[ "${certificate}" == *"raw_hits=0 uncertain=0"* ]] ||
    die "${lane}: fused CONTACT.4 transaction reported a hit or uncertainty"
  [[ "${certificate}" == *"fallback_flags=0 device_flags=0"* ]] ||
    die "${lane}: fused CONTACT.4 transaction reported fallback/device flags"
  for census in \
    active_contexts contact_contexts active_polygons contact_edges \
    rectangles boundary_segments member_visits candidates; do
    [[ "${certificate}" =~ ${census}=[1-9][0-9]* ]] ||
      die "${lane}: fused certificate lacks a positive ${census} census"
  done
  for census in \
    active_contexts contact_contexts active_stored_polygons \
    active_flat_polygons active_flat_edges contact_stored_polygons \
    contact_flat_polygons contact_flat_edges; do
    [[ "${lowering}" =~ ${census}=[1-9][0-9]* ]] ||
      die "${lane}: fused live lowering lacks a positive ${census} census"
  done
}

run_lane() {
  local lane=$1
  local mode=$2
  local active_union
  local runtime="${work}/runtime/${lane}"
  local report="${work}/reports/${lane}.lyrdb"
  local canonical="${work}/reports/${lane}.canonical.lyrdb"
  local log="${work}/logs/${lane}.log"
  local time_file="${work}/timings/${lane}.txt"
  local rc
  local actual_sha256

  case "${mode}" in
    control)
      active_union=0
      ;;
    candidate)
      active_union=1
      ;;
    *)
      die "internal error: unknown lane mode ${mode}"
      ;;
  esac

  mkdir -p -- \
    "${runtime}/home" \
    "${runtime}/klayout-home" \
    "${runtime}/xdg-config" \
    "${runtime}/xdg-cache" \
    "${runtime}/xdg-data" \
    "${runtime}/tmp"

  assert_gpu_idle "${lane}"
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
        "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
        "${common_cuda_env[@]}" \
        "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=${active_union}" \
        "${klayout}" -b -r "${deck}" \
          -rd "input=${input}" \
          -rd "topcell=${top_cell}" \
          -rd "output=${report}" \
          -rd drc_shard=implant_contact \
          >"${log}" 2>&1
  rc=$?
  set -e
  if ((rc != 0)); then
    cat -- "${time_file}" >&2 || true
    tail -n 200 -- "${log}" >&2 || true
    die "${lane}: implant_contact owner failed with status ${rc}"
  fi

  [[ -s "${report}" ]] ||
    die "${lane}: implant_contact owner produced no report"
  grep -Fq -- "<top-cell>${top_cell}</top-cell>" "${report}" ||
    die "${lane}: report top-cell marker is missing"
  canonicalize_report "${report}" "${canonical}"
  actual_sha256=$(file_sha256 "${canonical}")
  [[ "${actual_sha256}" == "${expected_report_sha256}" ]] ||
    die "${lane}: canonical report SHA-256 is ${actual_sha256}, expected ${expected_report_sha256}"
  cmp -s -- "${reference_canonical}" "${canonical}" ||
    die "${lane}: canonical report differs from the production reference"

  if [[ "${mode}" == control ]]; then
    assert_control_telemetry "${lane}"
  else
    assert_candidate_telemetry "${lane}"
  fi

  snapshot_core_artifacts \
    "${work}/runtime/${lane}-core-artifacts-after.sha256"
  cmp -s -- \
    "${core_before}" \
    "${work}/runtime/${lane}-core-artifacts-after.sha256" ||
    die "${lane}: a pinned input/runtime artifact changed during the gate"
}

lane_order=(
  "control-1:control"
  "candidate-1:candidate"
  "candidate-2:candidate"
  "control-2:control"
  "control-3:control"
  "candidate-3:candidate"
)

order_index=0
for lane_spec in "${lane_order[@]}"; do
  IFS=: read -r lane mode <<<"${lane_spec}"
  ((order_index += 1))
  printf '%s\t%s\t%s\t%s\n' \
    "${order_index}" "${lane}" "${mode}" "$(date -u +%FT%TZ)" \
    >>"${work}/run-order.tsv"
  run_lane "${lane}" "${mode}"
done

timings_tsv="${work}/timings/owner-wall.tsv"
: >"${timings_tsv}"
for replicate in 1 2 3; do
  for mode in control candidate; do
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

control_1=$(awk '$1 == 1 && $2 == "control" {print $3}' "${timings_tsv}")
control_2=$(awk '$1 == 2 && $2 == "control" {print $3}' "${timings_tsv}")
control_3=$(awk '$1 == 3 && $2 == "control" {print $3}' "${timings_tsv}")
candidate_1=$(awk '$1 == 1 && $2 == "candidate" {print $3}' "${timings_tsv}")
candidate_2=$(awk '$1 == 2 && $2 == "candidate" {print $3}' "${timings_tsv}")
candidate_3=$(awk '$1 == 3 && $2 == "candidate" {print $3}' "${timings_tsv}")

summary="${work}/timings/owner-summary.txt"
awk \
  -v c1="${control_1}" -v c2="${control_2}" -v c3="${control_3}" \
  -v g1="${candidate_1}" -v g2="${candidate_2}" -v g3="${candidate_3}" '
  function median3(a, b, c, t) {
    if (a > b) { t = a; a = b; b = t }
    if (b > c) { t = b; b = c; c = t }
    if (a > b) { t = a; a = b; b = t }
    return b
  }
  BEGIN {
    control_mean = (c1 + c2 + c3) / 3.0
    candidate_mean = (g1 + g2 + g3) / 3.0
    delta = control_mean - candidate_mean
    percent = control_mean == 0 ? 0 : 100.0 * delta / control_mean
    printf "CONTACT4_ACTIVE_UNION_PRODUCTION_OWNER_AB owner=implant_contact scope=owner-only control_mean_wall_s=%.3f candidate_mean_wall_s=%.3f delta_s=%.3f reduction_pct=%.3f control_median_wall_s=%.3f candidate_median_wall_s=%.3f N=3 order_balanced=1 report=exact\n",
      control_mean, candidate_mean, delta, percent,
      median3(c1, c2, c3), median3(g1, g2, g3)
  }' >"${summary}"

canonical_hashes="${work}/canonical-report-sha256.txt"
sha256sum -- \
  "${reference_canonical}" \
  "${work}/reports/control-1.canonical.lyrdb" \
  "${work}/reports/candidate-1.canonical.lyrdb" \
  "${work}/reports/control-2.canonical.lyrdb" \
  "${work}/reports/candidate-2.canonical.lyrdb" \
  "${work}/reports/control-3.canonical.lyrdb" \
  "${work}/reports/candidate-3.canonical.lyrdb" \
  >"${canonical_hashes}"

pinned_artifacts="${work}/pinned-artifacts.sha256"
sha256sum -- \
  "${core_artifacts[@]}" \
  "${reference_canonical}" \
  "${control_feature_manifest}" \
  "${candidate_feature_manifest}" \
  "${loader_closure}" \
  "${backend_symbols}" \
  "${work}/reports/control-1.canonical.lyrdb" \
  "${work}/reports/candidate-1.canonical.lyrdb" \
  "${work}/reports/control-2.canonical.lyrdb" \
  "${work}/reports/candidate-2.canonical.lyrdb" \
  "${work}/reports/control-3.canonical.lyrdb" \
  "${work}/reports/candidate-3.canonical.lyrdb" \
  >"${pinned_artifacts}"

telemetry="${work}/logs/contact4-telemetry.txt"
grep -h -E -- \
  'CUDA (M1 contact transaction:|IMPLANT\.1/\.2 transaction:|CONTACT\.4 (fused ACTIVE-union |raw-ACTIVE )?(empty certificate:|live lowering:))' \
  "${work}"/logs/{control,candidate}-{1,2,3}.log \
  >"${telemetry}"

cat -- "${work}/run-order.tsv"
cat -- "${timings_tsv}"
cat -- "${summary}"
cat -- "${canonical_hashes}"
cat -- "${pinned_artifacts}"
cat -- "${telemetry}"
echo \
  "CONTACT4_ACTIVE_UNION_PRODUCTION_OWNER_GATE PASS owner=implant_contact scope=owner-only lanes=6 N=3 serial=1 order-balanced=1 canonical-reports=exact telemetry=exact"
