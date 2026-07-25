#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_contact4_active_contact_kact_capture.sh \
    --klayout PATH --input PATH \
    --topcell sram_1rw0r0w_64_4096_freepdk45__x2 \
    --scene PATH --packed-scene PATH --report PATH --manifest PATH \
    [--threads N] [--python PATH]

Produce the source-bound full-size raw ACTIVE/CONTACT KACTSCN1 qualification
input. All paths are explicit and no output is overwritten. The source GDS,
top cell, capture deck, exporter, validator, physical layers, packed file, and
embedded scene digest are pinned below; any identity drift fails closed.
EOF
}

die() {
  echo "CONTACT.4 ACTIVE/CONTACT KACT capture: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

require_sha256() {
  local label=$1
  local path=$2
  local expected=$3
  local actual
  actual=$(sha256_file "${path}")
  [[ "${actual}" == "${expected}" ]] ||
    die "${label} SHA-256 mismatch: ${actual} != ${expected}"
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
capture_deck="${here}/contact4_active_contact_capture.lydrc"
exporter="${here}/active3_packed_scene_export.rb"
validator="${here}/validate_active3_packed_scene.py"

# Published production-x2 identity and exact qualification producer revisions.
readonly expected_source_sha256=\
74911a2111a3421912e54538bf55cd12e50164f43cd1a8c47411602e64c91d98
readonly expected_topcell=\
sram_1rw0r0w_64_4096_freepdk45__x2
readonly expected_capture_deck_sha256=\
3347422a3b4fd8ca4ed77c0e4711de6f7cc35f68d2d4594f576abed9996f8dc9
readonly expected_exporter_sha256=\
e6f0208b65cb21a142d0e387468520f7cea87af6108a976fef39fde433d04cf0
readonly expected_validator_sha256=\
2b6d655c4f2ad5848e5272f6abf22712635ec18035baa7fc851f7d8ac476d6c7

# Golden identities from two independent source/deck/exporter-qualified
# production captures.
readonly expected_packed_file_sha256=\
4c1609873c64c36b5ecf97796cc43cb4f164c4e872d199514ea07893449772bd
readonly expected_packed_scene_sha256=\
88cf7064183aba158e80486463b21c2481c3861490c0d0d4215311b2491be23e

klayout=
input=
topcell=
scene=
packed_scene=
report=
manifest=
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
    --manifest)
      (($# >= 2)) || die "--manifest requires a value"
      manifest=$2
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
[[ -n "${manifest}" ]] || die "missing --manifest"
[[ "${topcell}" == "${expected_topcell}" ]] ||
  die "top cell mismatch: ${topcell} != ${expected_topcell}"
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
require_sha256 "production source" \
  "${canonical_input}" "${expected_source_sha256}"
require_sha256 "capture deck" \
  "${capture_deck}" "${expected_capture_deck_sha256}"
require_sha256 "KACT exporter" \
  "${exporter}" "${expected_exporter_sha256}"
require_sha256 "KACT validator" \
  "${validator}" "${expected_validator_sha256}"

outputs=("${scene}" "${packed_scene}" "${report}" "${manifest}")
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
work=$(mktemp -d \
  "${TMPDIR:-/tmp}/klayout-contact4-active-contact-capture.XXXXXX")
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
  stage=$(mktemp -d \
    "${parent}/.${base}.contact4-active-contact-stage.XXXXXX")
  stage_dirs+=("${stage}")
  stage_result=${stage}
}

new_stage_dir "${canonical_outputs[0]}"
temp_scene="${stage_result}/$(basename -- "${canonical_outputs[0]}")"
new_stage_dir "${canonical_outputs[1]}"
temp_packed_scene="${stage_result}/$(basename -- "${canonical_outputs[1]}")"
new_stage_dir "${canonical_outputs[2]}"
temp_report="${stage_result}/$(basename -- "${canonical_outputs[2]}")"
new_stage_dir "${canonical_outputs[3]}"
temp_manifest="${stage_result}/$(basename -- "${canonical_outputs[3]}")"

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
  TZ=UTC
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)

env "${common_env[@]}" "${klayout}" -b \
  -r "${capture_deck}" \
  -rd "input=${canonical_input}" \
  -rd "topcell=${topcell}" \
  -rd "scene_output=${temp_scene}" \
  -rd "output=${temp_report}" \
  -rd "threads=${threads}" 2>&1 | tee "${capture_log}"

[[ -s "${temp_scene}" ]] ||
  die "capture did not produce a nonempty scene"
[[ -s "${temp_report}" ]] ||
  die "capture did not produce a nonempty source report"
grep -Eq -- \
  '^.*CONTACT4_ACTIVE_CONTACT_CAPTURE .* active=1/0 contact=10/0 ' \
  "${capture_log}" ||
  die "capture did not bind physical ACTIVE 1/0 and CONTACT 10/0"
grep -Fq -- "<name>CONTACT.4</name>" "${temp_report}" ||
  die "source report is missing CONTACT.4"

env "${common_env[@]}" "${klayout}" -b \
  -r "${exporter}" \
  -rd "input=${temp_scene}" \
  -rd "topcell=KLAYOUT_CUDA_CONTACT4_ACTIVE_CONTACT_SCENE" \
  -rd "output=${temp_packed_scene}" \
  -rd "well_layer=1" \
  -rd "well_datatype=0" \
  -rd "active_layer=10" \
  -rd "active_datatype=0" 2>&1 | tee "${export_log}"

[[ -s "${temp_packed_scene}" ]] ||
  die "export did not produce a nonempty packed scene"
export_summary=$(
  grep -E -- \
    '^KACTSCN1 output=.* bytes=[1-9][0-9]* cells=[1-9][0-9]* instances=[0-9]+ polygons=[1-9][0-9]* edges=[1-9][0-9]* root=[0-9]+ scene_sha256=[0-9a-f]{64}$' \
    "${export_log}" | tail -n 1 || true
)
[[ -n "${export_summary}" ]] ||
  die "export did not emit a valid KACTSCN1 summary"
export_scene_sha256=${export_summary##*scene_sha256=}

"${python}" "${validator}" "${temp_packed_scene}" 2>&1 |
  tee "${validate_log}"
validation_summary=$(
  grep -E -- \
    '^KACTSCN1 ok .* bytes=[1-9][0-9]* dbu=0\.0005 root=[0-9]+ cells=[1-9][0-9]* instances=[0-9]+ array_occurrences=[0-9]+ polygons=[1-9][0-9]* edges=[1-9][0-9]* layers=1/0,10/0 scene_sha256=[0-9a-f]{64} file_sha256=[0-9a-f]{64}$' \
    "${validate_log}" | tail -n 1 || true
)
[[ -n "${validation_summary}" ]] ||
  die "independent KACTSCN1 physical-layer validation is missing"

validated_scene_sha256=$(
  sed -nE \
    's/.* scene_sha256=([0-9a-f]{64}) file_sha256=.*/\1/p' \
    <<<"${validation_summary}"
)
validated_file_sha256=${validation_summary##*file_sha256=}
packed_file_sha256=$(sha256_file "${temp_packed_scene}")
[[ "${validated_scene_sha256}" == "${export_scene_sha256}" ]] ||
  die "exporter/validator scene SHA-256 mismatch"
[[ "${validated_file_sha256}" == "${packed_file_sha256}" ]] ||
  die "validator/external packed-file SHA-256 mismatch"

if [[ -n "${expected_packed_scene_sha256}" ]]; then
  [[ "${validated_scene_sha256}" == "${expected_packed_scene_sha256}" ]] ||
    die "qualified scene SHA-256 mismatch: ${validated_scene_sha256} != ${expected_packed_scene_sha256}"
fi
if [[ -n "${expected_packed_file_sha256}" ]]; then
  [[ "${packed_file_sha256}" == "${expected_packed_file_sha256}" ]] ||
    die "qualified packed-file SHA-256 mismatch: ${packed_file_sha256} != ${expected_packed_file_sha256}"
fi

capture_file_sha256=$(sha256_file "${temp_scene}")
report_sha256=$(sha256_file "${temp_report}")
capture_bytes=$(stat -c %s -- "${temp_scene}")
packed_bytes=$(stat -c %s -- "${temp_packed_scene}")
report_items=$(grep -c -- '<item>' "${temp_report}" || true)
cells=$(
  sed -nE 's/.* cells=([0-9]+) instances=.*/\1/p' \
    <<<"${validation_summary}"
)
instances=$(
  sed -nE 's/.* instances=([0-9]+) array_occurrences=.*/\1/p' \
    <<<"${validation_summary}"
)
array_occurrences=$(
  sed -nE 's/.* array_occurrences=([0-9]+) polygons=.*/\1/p' \
    <<<"${validation_summary}"
)
polygons=$(
  sed -nE 's/.* polygons=([0-9]+) edges=.*/\1/p' \
    <<<"${validation_summary}"
)
edges=$(
  sed -nE 's/.* edges=([0-9]+) layers=.*/\1/p' \
    <<<"${validation_summary}"
)

printf '%s\n' \
  '{' \
  '  "format": "klayout-contact4-active-contact-kact-qualification-v1",' \
  '  "source": {' \
  "    \"sha256\": \"${expected_source_sha256}\"," \
  "    \"top_cell\": \"${expected_topcell}\"," \
  '    "dbu_micrometers": 0.0005,' \
  '    "active": {"layer": 1, "datatype": 0, "kact_slot": 0},' \
  '    "contact": {"layer": 10, "datatype": 0, "kact_slot": 1}' \
  '  },' \
  '  "producer": {' \
  "    \"capture_deck_sha256\": \"${expected_capture_deck_sha256}\"," \
  "    \"exporter_sha256\": \"${expected_exporter_sha256}\"," \
  "    \"validator_sha256\": \"${expected_validator_sha256}\"" \
  '  },' \
  '  "artifacts": {' \
  "    \"capture_gds_sha256\": \"${capture_file_sha256}\"," \
  "    \"capture_gds_bytes\": ${capture_bytes}," \
  "    \"source_report_sha256\": \"${report_sha256}\"," \
  "    \"source_report_items\": ${report_items}," \
  "    \"kact_file_sha256\": \"${packed_file_sha256}\"," \
  "    \"kact_scene_sha256\": \"${validated_scene_sha256}\"," \
  "    \"kact_bytes\": ${packed_bytes}" \
  '  },' \
  '  "census": {' \
  "    \"cells\": ${cells}," \
  "    \"instances\": ${instances}," \
  "    \"array_occurrences\": ${array_occurrences}," \
  "    \"polygons\": ${polygons}," \
  "    \"edges\": ${edges}" \
  '  }' \
  '}' >"${temp_manifest}"
"${python}" -m json.tool "${temp_manifest}" >/dev/null

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
publish "${temp_manifest}" "${canonical_outputs[3]}"
publication_complete=1

echo "CONTACT.4 ACTIVE/CONTACT KACT capture passed."
echo "source_sha256=${expected_source_sha256}"
echo "capture_deck_sha256=${expected_capture_deck_sha256}"
echo "exporter_sha256=${expected_exporter_sha256}"
echo "validator_sha256=${expected_validator_sha256}"
echo "${validation_summary}"
sha256sum -- "${canonical_outputs[@]}"
