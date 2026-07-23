#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_m1_via1_packed_scene_capture.sh \
    --klayout PATH --input PATH --topcell NAME \
    --scene PATH --packed-scene PATH --report PATH \
    [--threads N] [--python PATH]

All input and output paths are explicit. The runner refuses to overwrite files.
The intermediate scene maps METAL1 to 101/0 and VIA1 to 102/0; the packed output
uses those same two slots in the existing KACTSCN1 format.
EOF
}

die() {
  echo "M1/VIA1 packed-scene capture: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
capture_deck="${here}/m1_via1_projection_capture.lydrc"
exporter="${here}/active3_packed_scene_export.rb"
validator="${here}/validate_active3_packed_scene.py"

klayout=
input=
topcell=
scene=
packed_scene=
report=
threads=1
python=${PYTHON:-python3}

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
    --packed-scene)
      (($# >= 2)) || die "--packed-scene requires a value"
      packed_scene=$2
      shift 2
      ;;
    --report)
      (($# >= 2)) || die "--report requires a value"
      report=$2
      shift 2
      ;;
    --threads)
      (($# >= 2)) || die "--threads requires a value"
      threads=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
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
[[ -n "${packed_scene}" ]] || die "missing --packed-scene"
[[ -n "${report}" ]] || die "missing --report"
[[ "${threads}" =~ ^[1-9][0-9]*$ ]] ||
  die "--threads must be a positive integer"

[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${input}" ]] || die "input does not exist: ${input}"
[[ -f "${capture_deck}" ]] || die "capture deck is missing: ${capture_deck}"
[[ -f "${exporter}" ]] || die "packed-scene exporter is missing: ${exporter}"
[[ -f "${validator}" ]] || die "packed-scene validator is missing: ${validator}"
python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"

canonical_input=$(readlink -f -- "${input}")
outputs=("${scene}" "${packed_scene}" "${report}")
canonical_outputs=()
for output in "${outputs[@]}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] ||
    die "refusing to overwrite output: ${output}"
  parent=$(dirname -- "${output}")
  [[ -d "${parent}" ]] || die "output directory does not exist: ${parent}"
  [[ -w "${parent}" ]] || die "output directory is not writable: ${parent}"
  canonical_parent=$(readlink -f -- "${parent}")
  canonical_output="${canonical_parent}/$(basename -- "${output}")"
  [[ "${canonical_output}" != "${canonical_input}" ]] ||
    die "output aliases input: ${output}"
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
work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-m1-via1-capture.XXXXXX")
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
  stage=$(mktemp -d "${parent}/.${base}.m1-via1-stage.XXXXXX")
  stage_dirs+=("${stage}")
  stage_result=${stage}
}

new_stage_dir "${canonical_outputs[0]}"
temp_scene="${stage_result}/$(basename -- "${canonical_outputs[0]}")"
new_stage_dir "${canonical_outputs[1]}"
temp_packed_scene="${stage_result}/$(basename -- "${canonical_outputs[1]}")"
new_stage_dir "${canonical_outputs[2]}"
temp_report="${stage_result}/$(basename -- "${canonical_outputs[2]}")"

capture_log="${work}/capture.log"
export_log="${work}/export.log"
validate_log="${work}/validate.log"
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
  -rd "output=${temp_report}" \
  -rd "threads=${threads}" 2>&1 | tee "${capture_log}"

[[ -s "${temp_scene}" ]] || die "capture did not produce a nonempty scene"
[[ -s "${temp_report}" ]] || die "capture did not produce a nonempty report"
grep -Fq -- "M1_VIA1_CAPTURE" "${capture_log}" ||
  die "capture did not emit its completion marker"
grep -Fq -- "<name>METAL1.4</name>" "${temp_report}" ||
  die "source report is missing METAL1.4"

env "${common_env[@]}" "${klayout}" -b \
  -r "${exporter}" \
  -rd "input=${temp_scene}" \
  -rd "topcell=KLAYOUT_CUDA_M1_VIA1_SCENE" \
  -rd "output=${temp_packed_scene}" \
  -rd "well_layer=101" \
  -rd "well_datatype=0" \
  -rd "active_layer=102" \
  -rd "active_datatype=0" 2>&1 | tee "${export_log}"

[[ -s "${temp_packed_scene}" ]] ||
  die "export did not produce a nonempty packed scene"
grep -Eq -- \
  '^KACTSCN1 output=.* bytes=[1-9][0-9]* cells=[1-9][0-9]* instances=[0-9]+ polygons=[1-9][0-9]* edges=[1-9][0-9]* ' \
  "${export_log}" ||
  die "export did not emit a valid KACTSCN1 summary"

"${python}" "${validator}" "${temp_packed_scene}" 2>&1 |
  tee "${validate_log}"
grep -Eq -- \
  '^KACTSCN1 ok .* cells=[1-9][0-9]* instances=[0-9]+ .* polygons=[1-9][0-9]* edges=[1-9][0-9]* layers=101/0,102/0 ' \
  "${validate_log}" ||
  die "independent KACTSCN1 validation summary is missing"

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
publish "${temp_packed_scene}" "${canonical_outputs[1]}"
publish "${temp_report}" "${canonical_outputs[2]}"
publication_complete=1

report_items=$(grep -c -- '<item>' "${canonical_outputs[2]}" || true)
echo "M1/VIA1 packed-scene capture passed."
echo "report_items=${report_items}"
grep -F -- "M1_VIA1_CAPTURE" "${capture_log}" | tail -n 1
tail -n 1 -- "${export_log}"
tail -n 1 -- "${validate_log}"
sha256sum -- "${canonical_outputs[@]}"
