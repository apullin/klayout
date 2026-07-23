#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_active3_scene_capture.sh \
    --klayout PATH --input PATH --topcell NAME --scene PATH \
    --source-report PATH --replay-report PATH --census PATH [--threads N]

All input and output paths are explicit. The runner refuses to overwrite files.
EOF
}

die() {
  echo "active3 scene capture: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
capture_deck="${here}/active3_scene_capture.lydrc"
replay_deck="${here}/active3_scene_replay.lydrc"
census_script="${here}/active3_scene_census.rb"

klayout=
input=
topcell=
scene=
source_report=
replay_report=
census=
threads=1

while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
      shift 2
      ;;
    --input)
      (($# >= 2)) || die "--input requires a value"
      input=$2
      shift 2
      ;;
    --topcell)
      (($# >= 2)) || die "--topcell requires a value"
      topcell=$2
      shift 2
      ;;
    --scene)
      (($# >= 2)) || die "--scene requires a value"
      scene=$2
      shift 2
      ;;
    --source-report)
      (($# >= 2)) || die "--source-report requires a value"
      source_report=$2
      shift 2
      ;;
    --replay-report)
      (($# >= 2)) || die "--replay-report requires a value"
      replay_report=$2
      shift 2
      ;;
    --census)
      (($# >= 2)) || die "--census requires a value"
      census=$2
      shift 2
      ;;
    --threads)
      (($# >= 2)) || die "--threads requires a value"
      threads=$2
      shift 2
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
[[ -n "${input}" ]] || die "missing --input"
[[ -n "${topcell}" ]] || die "missing --topcell"
[[ -n "${scene}" ]] || die "missing --scene"
[[ -n "${source_report}" ]] || die "missing --source-report"
[[ -n "${replay_report}" ]] || die "missing --replay-report"
[[ -n "${census}" ]] || die "missing --census"
[[ "${threads}" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"

[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${input}" ]] || die "input does not exist: ${input}"
[[ -f "${capture_deck}" ]] || die "capture deck is missing: ${capture_deck}"
[[ -f "${replay_deck}" ]] || die "replay deck is missing: ${replay_deck}"
[[ -f "${census_script}" ]] || die "census script is missing: ${census_script}"

canonical_input=$(readlink -f -- "${input}")
outputs=("${scene}" "${source_report}" "${replay_report}" "${census}")
canonical_outputs=()
for output in "${outputs[@]}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] ||
    die "refusing to overwrite output: ${output}"
  parent=$(dirname -- "${output}")
  [[ -d "${parent}" ]] || die "output directory does not exist: ${parent}"
  [[ -w "${parent}" ]] || die "output directory is not writable: ${parent}"
  canonical_parent=$(readlink -f -- "${parent}")
  canonical_output="${canonical_parent}/$(basename -- "${output}")"
  [[ "${canonical_output}" != "${canonical_input}" ]] || die "output aliases input: ${output}"
  canonical_outputs+=("${canonical_output}")
done

for ((i = 0; i < ${#canonical_outputs[@]}; ++i)); do
  for ((j = i + 1; j < ${#canonical_outputs[@]}; ++j)); do
    [[ "${canonical_outputs[i]}" != "${canonical_outputs[j]}" ]] ||
      die "output paths alias each other: ${outputs[i]} and ${outputs[j]}"
  done
done

klayout=$(readlink -f -- "${klayout}")
klayout_dir=$(dirname -- "${klayout}")
work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-active3-capture.XXXXXX")
stage_dirs=()
published=()
publication_complete=0

cleanup() {
  if ((publication_complete == 0)) && ((${#published[@]})); then
    rm -f -- "${published[@]}"
  fi
  rm -rf -- "${work}" "${stage_dirs[@]}"
}
trap cleanup EXIT

new_stage_dir() {
  local destination=$1
  local parent base stage
  parent=$(dirname -- "${destination}")
  base=$(basename -- "${destination}")
  stage=$(mktemp -d "${parent}/.${base}.active3-stage.XXXXXX")
  stage_dirs+=("${stage}")
  stage_result=${stage}
}

new_stage_dir "${canonical_outputs[0]}"
temp_scene="${stage_result}/$(basename -- "${canonical_outputs[0]}")"
new_stage_dir "${canonical_outputs[1]}"
temp_source_report="${stage_result}/source.lyrdb"
new_stage_dir "${canonical_outputs[2]}"
temp_replay_report="${stage_result}/replay.lyrdb"
new_stage_dir "${canonical_outputs[3]}"
temp_census="${stage_result}/scene.census"
capture_log="${work}/capture.log"
replay_log="${work}/replay.log"
census_log="${work}/census.log"
runtime_home="${work}/home"
runtime_config="${work}/config"
runtime_cache="${work}/cache"
mkdir -p -- "${runtime_home}" "${runtime_config}" "${runtime_cache}"

common_env=(
  QT_QPA_PLATFORM=offscreen
  HOME="${runtime_home}"
  XDG_CONFIG_HOME="${runtime_config}"
  XDG_CACHE_HOME="${runtime_cache}"
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)

env "${common_env[@]}" "${klayout}" -b \
  -r "${capture_deck}" \
  -rd "input=${canonical_input}" \
  -rd "topcell=${topcell}" \
  -rd "scene_output=${temp_scene}" \
  -rd "output=${temp_source_report}" \
  -rd "threads=${threads}" 2>&1 | tee "${capture_log}"

[[ -s "${temp_scene}" ]] || die "capture did not produce a nonempty scene"
[[ -s "${temp_source_report}" ]] || die "capture did not produce a nonempty source report"
grep -Fq -- "ACTIVE3_CAPTURE" "${capture_log}" ||
  die "capture did not emit its completion marker"

env "${common_env[@]}" "${klayout}" -b \
  -r "${replay_deck}" \
  -rd "input=${temp_scene}" \
  -rd "topcell=KLAYOUT_CUDA_ACTIVE3_SCENE" \
  -rd "output=${temp_replay_report}" \
  -rd "threads=${threads}" 2>&1 | tee "${replay_log}"

[[ -s "${temp_replay_report}" ]] || die "replay did not produce a nonempty report"
grep -Fq -- "ACTIVE3_REPLAY" "${replay_log}" ||
  die "replay did not emit its start marker"

env "${common_env[@]}" "${klayout}" -b \
  -r "${census_script}" \
  -rd "input=${temp_scene}" \
  -rd "topcell=KLAYOUT_CUDA_ACTIVE3_SCENE" \
  -rd "output=${temp_census}" \
  -rd "well_layer=101" \
  -rd "active_layer=102" 2>&1 | tee "${census_log}"

[[ -s "${temp_census}" ]] || die "census did not produce a nonempty result"
grep -Fq -- "format=klayout-active3-scene-census-v1" "${temp_census}" ||
  die "census format marker is missing"
grep -Eq -- '^cells=[1-9][0-9]* instance_records=' "${temp_census}" ||
  die "census did not find a reachable hierarchy"
grep -Eq -- '^well stored_shapes=[1-9][0-9]* ' "${temp_census}" ||
  die "captured well operand is empty"
grep -Eq -- '^active stored_shapes=[1-9][0-9]* ' "${temp_census}" ||
  die "captured active operand is empty"

source_items=$(grep -c -- '<item>' "${temp_source_report}" || true)
replay_items=$(grep -c -- '<item>' "${temp_replay_report}" || true)
[[ "${source_items}" == "${replay_items}" ]] ||
  die "source/replay ACTIVE.3 item-count mismatch: ${source_items} != ${replay_items}"
grep -Fq -- "<name>ACTIVE.3</name>" "${temp_source_report}" ||
  die "source report is missing ACTIVE.3"
grep -Fq -- "<name>ACTIVE.3</name>" "${temp_replay_report}" ||
  die "replay report is missing ACTIVE.3"

for output in "${outputs[@]}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] ||
    die "output appeared before publication: ${output}"
done

publish() {
  local staged=$1
  local destination=$2
  mv -n -- "${staged}" "${destination}"
  [[ ! -e "${staged}" ]] ||
    die "refusing to replace output that appeared during publication: ${destination}"
  [[ -e "${destination}" ]] ||
    die "publication failed: ${destination}"
  published+=("${destination}")
}

publish "${temp_scene}" "${canonical_outputs[0]}"
publish "${temp_source_report}" "${canonical_outputs[1]}"
publish "${temp_replay_report}" "${canonical_outputs[2]}"
publish "${temp_census}" "${canonical_outputs[3]}"
publication_complete=1

echo "ACTIVE.3 scene capture and replay passed."
echo "source_items=${source_items} replay_items=${replay_items}"
sha256sum -- "${canonical_outputs[@]}"
