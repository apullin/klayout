#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_projection_enclosure_scene_gate.sh \
    --klayout PATH [--python PATH] [--build-dir PATH] [--keep-work]

Generates deterministic projection-enclosure fixtures, exports each named top
through the existing KACTSCN1 exporter, compiles the CUDA island once, and
checks clean, boundary, decomposition, hierarchy, fallback, and fail-closed
outcomes, including duplicate/touching cut handling and the strict VIA1 spacing
boundary. Temporary artifacts are removed unless --keep-work is specified.
EOF
}

die() {
  echo "projection-enclosure scene gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture_script="${here}/projection_enclosure_scene_fixture.rb"
exporter="${here}/active3_packed_scene_export.rb"
validator="${here}/validate_active3_packed_scene.py"
island_runner="${here}/run_projection_enclosure_scene_island.sh"

klayout=
python=${PYTHON:-python3}
requested_build_dir=
keep_work=0

while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
      shift 2
      ;;
    --build-dir)
      (($# >= 2)) || die "--build-dir requires a value"
      requested_build_dir=$2
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
[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${fixture_script}" ]] || die "fixture generator is missing"
[[ -f "${exporter}" ]] || die "KACTSCN1 exporter is missing"
[[ -f "${validator}" ]] || die "KACTSCN1 validator is missing"
[[ -x "${island_runner}" ]] || die "projection island runner is not executable"
python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
klayout=$(readlink -f -- "${klayout}")
klayout_dir=$(dirname -- "${klayout}")

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-projection-scene-gate.XXXXXX")
if [[ -n "${requested_build_dir}" ]]; then
  mkdir -p -- "${requested_build_dir}"
  build_dir=$(readlink -f -- "${requested_build_dir}")
else
  build_dir="${work}/build"
fi

cleanup() {
  if ((keep_work)); then
    echo "PROJECTION_ENCLOSURE_SCENE_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

runtime_home="${work}/home"
runtime_config="${work}/config"
runtime_cache="${work}/cache"
mkdir -p -- \
  "${runtime_home}" "${runtime_config}" "${runtime_cache}" "${build_dir}"
common_env=(
  QT_QPA_PLATFORM=offscreen
  HOME="${runtime_home}"
  XDG_CONFIG_HOME="${runtime_config}"
  XDG_CACHE_HOME="${runtime_cache}"
  LD_LIBRARY_PATH="${klayout_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)

fixture="${work}/projection-enclosure-fixture.gds"
fixture_log="${work}/fixture.log"
env "${common_env[@]}" "${klayout}" -b \
  -r "${fixture_script}" -rd "output=${fixture}" 2>&1 |
  tee "${fixture_log}"
[[ -s "${fixture}" ]] || die "fixture generator produced no layout"
grep -Fq -- "PROJECTION_ENCLOSURE_SCENE_FIXTURE ok" "${fixture_log}" ||
  die "fixture completion marker is missing"

declare -A scene_path
declare -A scene_digest

export_case() {
  local key=$1
  local top=$2
  local output="${work}/${key}.kact"
  local export_log="${work}/${key}.export.log"
  local validate_log="${work}/${key}.validate.log"
  local digest

  env "${common_env[@]}" "${klayout}" -b \
    -r "${exporter}" \
    -rd "input=${fixture}" \
    -rd "topcell=${top}" \
    -rd "output=${output}" \
    -rd "well_layer=101" \
    -rd "well_datatype=0" \
    -rd "active_layer=102" \
    -rd "active_datatype=0" 2>&1 |
    tee "${export_log}"
  [[ -s "${output}" ]] || die "${key}: exporter produced no packed scene"

  "${python}" "${validator}" "${output}" 2>&1 | tee "${validate_log}"
  grep -Fq -- "KACTSCN1 ok" "${validate_log}" ||
    die "${key}: independent packed-scene validation failed"

  digest=$(
    sed -nE \
      's/^KACTSCN1 output=.* scene_sha256=([0-9a-f]{64})$/\1/p' \
      "${export_log}"
  )
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] ||
    die "${key}: exporter scene digest is missing or ambiguous"
  scene_path["${key}"]=${output}
  scene_digest["${key}"]=${digest}
}

export_case clean_box PROJECTION_CLEAN_BOX
export_case clean_box_repeat PROJECTION_CLEAN_BOX
cmp -s -- "${scene_path[clean_box]}" "${scene_path[clean_box_repeat]}" ||
  die "repeat clean-box KACTSCN1 exports are not byte-identical"
[[ "${scene_digest[clean_box]}" == "${scene_digest[clean_box_repeat]}" ]] ||
  die "repeat clean-box scene digests differ"
echo "PROJECTION_ENCLOSURE_SCENE_GATE ok gate=deterministic-export"

export_case exact_boundary PROJECTION_EXACT_BOUNDARY
export_case l_shape PROJECTION_L_SHAPE
export_case x_slab_witness PROJECTION_X_SLAB_WITNESS
export_case missing_enclosure PROJECTION_MISSING_ENCLOSURE
export_case duplicate_cuts PROJECTION_DUPLICATE_CUTS
export_case touching_cuts PROJECTION_TOUCHING_CUTS
export_case spacing_149 PROJECTION_SPACING_149
export_case spacing_150 PROJECTION_SPACING_150
export_case diagonal_90_120 PROJECTION_DIAGONAL_90_120
export_case diagonal_90_119 PROJECTION_DIAGONAL_90_119
export_case hierarchy PROJECTION_HIERARCHY
export_case nonrect_cut PROJECTION_NONRECT_CUT

assert_contains() {
  local log=$1
  local text=$2
  grep -Fq -- "${text}" "${log}" ||
    die "$(basename -- "${log}"): missing output: ${text}"
}

run_expected() {
  local key=$1
  local expected_rc=$2
  shift 2
  local log="${work}/${key}.island.log"
  local rc

  set +e
  "${build_dir}/projection_enclosure_scene_island" \
    "--expect-scene-sha256=${scene_digest[${key}]}" \
    --verify-bruteforce \
    "${scene_path[${key}]}" >"${log}" 2>&1
  rc=$?
  set -e
  cat -- "${log}"
  [[ "${rc}" == "${expected_rc}" ]] ||
    die "${key}: expected exit ${expected_rc}, got ${rc}"
  for expected in "$@"; do
    assert_contains "${log}" "${expected}"
  done
  echo "PROJECTION_ENCLOSURE_SCENE_GATE ok gate=${key} rc=${rc}"
}

# The first clean run also compiles the island. Subsequent cases use exactly
# that binary so the gate pays the compile cost once.
clean_log="${work}/clean_box.island.log"
BUILD_DIR="${build_dir}" "${island_runner}" \
  "--expect-scene-sha256=${scene_digest[clean_box]}" \
  --verify-bruteforce "${scene_path[clean_box]}" 2>&1 |
  tee "${clean_log}"
[[ -x "${build_dir}/projection_enclosure_scene_island" ]] ||
  die "island runner did not produce its binary"
assert_contains "${clean_log}" "verdict=CLEAN"
assert_contains "${clean_log}" " metal_boxes=1 cuts=1 "
assert_contains "${clean_log}" " certified=1 misses=0 "
assert_contains "${clean_log}" " duplicate_cut_pairs=0 "
assert_contains "${clean_log}" " unsafe_touching_cut_pairs=0 "
assert_contains "${clean_log}" " via1_spacing_violations=0 "
assert_contains "${clean_log}" " via1_size_clean=1 "
assert_contains "${clean_log}" " via1_spacing_clean=1 "
assert_contains "${clean_log}" " projection_clean=1 "
assert_contains "${clean_log}" " decomposed_metal_templates=0 "
assert_contains "${clean_log}" " decomposition_rectangles=0 device_flags=0 "
echo "PROJECTION_ENCLOSURE_SCENE_GATE ok gate=clean_box rc=0"

run_expected exact_boundary 0 \
  "verdict=CLEAN" \
  " metal_boxes=1 cuts=1 " \
  " certified=1 misses=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=1 "

run_expected l_shape 0 \
  "verdict=CLEAN" \
  " metal_boxes=4 cuts=1 " \
  " certified=1 misses=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=1 " \
  " decomposed_metal_templates=1 " \
  " decomposition_rectangles=4 device_flags=0 "

run_expected x_slab_witness 0 \
  "verdict=CLEAN" \
  " metal_boxes=6 cuts=1 " \
  " candidate_boxes=5 " \
  " certified=1 misses=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=1 " \
  " decomposed_metal_templates=1 " \
  " decomposition_rectangles=6 device_flags=0 "

run_expected hierarchy 0 \
  "verdict=CLEAN" \
  " contexts=11 metal_contexts=8 cut_contexts=8 " \
  " metal_boxes=8 cuts=8 " \
  " certified=8 misses=0 " \
  " duplicate_cut_pairs=0 " \
  " unsafe_touching_cut_pairs=0 " \
  " via1_spacing_violations=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=1 "

run_expected missing_enclosure 3 \
  "verdict=FALLBACK" \
  " metal_boxes=1 cuts=1 " \
  " certified=0 misses=1 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=0 " \
  "SAMPLE context=0 cut_local=0 box=60,60,190,190"

run_expected duplicate_cuts 0 \
  "verdict=CLEAN" \
  " metal_boxes=1 cuts=2 " \
  " certified=2 misses=0 " \
  " cuts_pair_queried=2 " \
  " cut_pair_candidates=1 " \
  " duplicate_cut_pairs=1 " \
  " unsafe_touching_cut_pairs=0 " \
  " via1_spacing_violations=0 " \
  " cut_pair_clean=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=1 "

run_expected touching_cuts 3 \
  "verdict=FALLBACK" \
  " metal_boxes=1 cuts=2 " \
  " certified=2 misses=0 " \
  " cuts_pair_queried=2 " \
  " cut_pair_candidates=1 " \
  " duplicate_cut_pairs=0 " \
  " unsafe_touching_cut_pairs=1 " \
  " via1_spacing_violations=0 " \
  " cut_pair_clean=0 " \
  " via1_size_clean=0 " \
  " via1_spacing_clean=0 " \
  " projection_clean=0 "

run_expected spacing_149 0 \
  "verdict=CLEAN" \
  " metal_boxes=1 cuts=2 " \
  " certified=2 misses=0 " \
  " cuts_pair_queried=2 " \
  " cut_pair_candidates=1 " \
  " duplicate_cut_pairs=0 " \
  " unsafe_touching_cut_pairs=0 " \
  " via1_spacing_violations=1 " \
  " cut_pair_clean=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=0 " \
  " projection_clean=1 "

run_expected spacing_150 0 \
  "verdict=CLEAN" \
  " metal_boxes=1 cuts=2 " \
  " certified=2 misses=0 " \
  " cuts_pair_queried=2 " \
  " cut_pair_candidates=0 " \
  " duplicate_cut_pairs=0 " \
  " unsafe_touching_cut_pairs=0 " \
  " via1_spacing_violations=0 " \
  " cut_pair_clean=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=1 "

run_expected diagonal_90_120 0 \
  "verdict=CLEAN" \
  " metal_boxes=1 cuts=2 " \
  " certified=2 misses=0 " \
  " cuts_pair_queried=2 " \
  " cut_pair_candidates=1 " \
  " duplicate_cut_pairs=0 " \
  " unsafe_touching_cut_pairs=0 " \
  " via1_spacing_violations=0 " \
  " cut_pair_clean=1 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=1 " \
  " projection_clean=1 "

run_expected diagonal_90_119 0 \
  "verdict=CLEAN" \
  " metal_boxes=1 cuts=2 " \
  " certified=2 misses=0 " \
  " cuts_pair_queried=2 " \
  " cut_pair_candidates=1 " \
  " duplicate_cut_pairs=0 " \
  " unsafe_touching_cut_pairs=0 " \
  " via1_spacing_violations=1 " \
  " cut_pair_clean=0 " \
  " via1_size_clean=1 " \
  " via1_spacing_clean=0 " \
  " projection_clean=1 "

run_expected nonrect_cut 2 \
  "verdict=UNCERTAIN" \
  "projection-enclosure cut layer contains a non-rectangle"

corrupt_scene="${work}/corrupt-clean-box.kact"
"${python}" -c \
  'import pathlib,sys; data=bytearray(pathlib.Path(sys.argv[1]).read_bytes()); data[256]^=1; pathlib.Path(sys.argv[2]).write_bytes(data)' \
  "${scene_path[clean_box]}" "${corrupt_scene}"

corrupt_validate_log="${work}/corrupt.validate.log"
set +e
"${python}" "${validator}" "${corrupt_scene}" \
  >"${corrupt_validate_log}" 2>&1
corrupt_validate_rc=$?
set -e
cat -- "${corrupt_validate_log}"
[[ "${corrupt_validate_rc}" == 2 ]] ||
  die "corrupt scene unexpectedly passed independent validation"
assert_contains "${corrupt_validate_log}" "KACTSCN1 invalid:"

corrupt_log="${work}/corrupt.island.log"
set +e
"${build_dir}/projection_enclosure_scene_island" \
  "--expect-scene-sha256=${scene_digest[clean_box]}" \
  "${corrupt_scene}" >"${corrupt_log}" 2>&1
corrupt_rc=$?
set -e
cat -- "${corrupt_log}"
[[ "${corrupt_rc}" == 2 ]] ||
  die "corrupt scene did not fail closed with exit 2"
assert_contains "${corrupt_log}" "verdict=UNCERTAIN"
assert_contains "${corrupt_log}" "scene SHA-256 mismatch"
echo "PROJECTION_ENCLOSURE_SCENE_GATE ok gate=corrupt-hash rc=2"

echo "PROJECTION_ENCLOSURE_SCENE_GATE passed gates=15 verdict_clean=10 fallback=2 fail_closed=2 deterministic=1"
