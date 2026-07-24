#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_poly34_production_dry_run.sh \
    --input LAYOUT.gds --top TOP [--build-dir DIR] \
    [--klayout-lib-dir DIR] [--klayout-gds-plugin-dir DIR] \
    [--skip-exact-terminal]

Builds and runs the read-only POLY.3/POLY.4 production candidate-window
census. The default library paths select the accepted PGO KLayout artifact.
No host ABI, DRC deck, or production report path is changed.
EOF
}

die() {
  echo "POLY34 production dry run: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_file="${here}/poly34_production_dry_run.cc"
source_root=$(cd -- "${here}/../.." && pwd)
default_lib_dir=/home/pullin/personal/klayout/bin-clang-perf-znver2-pgo-use-a31ad8f30edc1052-manifest-9e0e44d9ac6929fe-commit-575a32b27db9e947-tree-d6810f6fe72eddaf

input=
top=
build_dir=/tmp/klayout-poly34-production-dry-run-build
lib_dir=${KLAYOUT_LIB_DIR:-"${default_lib_dir}"}
gds_plugin_dir=${KLAYOUT_GDS_PLUGIN_DIR:-}
qt_lib_dir=${KLAYOUT_QT_LIB_DIR:-/home/pullin/.local/share/mamba/envs/klay/lib}
qt_include_dir=${KLAYOUT_QT_INCLUDE_DIR:-/home/pullin/.local/share/mamba/envs/klay/include/qt6}
skip_exact_terminal=0
while (($#)); do
  case "$1" in
    --input)
      (($# >= 2)) || die "--input requires a value"
      input=$2
      shift 2
      ;;
    --top)
      (($# >= 2)) || die "--top requires a value"
      top=$2
      shift 2
      ;;
    --build-dir)
      (($# >= 2)) || die "--build-dir requires a value"
      build_dir=$2
      shift 2
      ;;
    --klayout-lib-dir)
      (($# >= 2)) || die "--klayout-lib-dir requires a value"
      lib_dir=$2
      shift 2
      ;;
    --klayout-gds-plugin-dir)
      (($# >= 2)) || die "--klayout-gds-plugin-dir requires a value"
      gds_plugin_dir=$2
      shift 2
      ;;
    --skip-exact-terminal)
      skip_exact_terminal=1
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

if [[ -z "${gds_plugin_dir}" ]]; then
  gds_plugin_dir="${lib_dir}/db_plugins"
fi
[[ -n "${input}" ]] || die "missing --input"
[[ -f "${input}" ]] || die "input is absent: ${input}"
[[ -n "${top}" ]] || die "missing --top"
[[ -f "${source_file}" ]] || die "source is absent: ${source_file}"
[[ -f "${lib_dir}/libklayout_db.so" ]] ||
  die "libklayout_db.so is absent: ${lib_dir}"
[[ -f "${lib_dir}/libklayout_tl.so" ]] ||
  die "libklayout_tl.so is absent: ${lib_dir}"
[[ -f "${gds_plugin_dir}/libgds2.so" ]] ||
  die "libgds2.so is absent: ${gds_plugin_dir}"

mkdir -p -- "${build_dir}"
binary="${build_dir}/poly34_production_dry_run"
cxx=${CXX:-/usr/bin/g++-13}
"${cxx}" \
  -std=c++17 -O3 -fopenmp -Wall -Wextra -Werror \
  -DHAVE_QT \
  -I"${source_root}/src/db/db" \
  -I"${source_root}/src/tl/tl" \
  -I"${source_root}/src/gsi/gsi" \
  -I"${source_root}/src/plugins/common" \
  -I"${source_root}/src/plugins/streamers/gds2/db_plugin" \
  -I"${qt_include_dir}" \
  -I"${qt_include_dir}/QtCore" \
  -I"${qt_include_dir}/QtCore5Compat" \
  "${source_file}" \
  -L"${lib_dir}" -L"${gds_plugin_dir}" -L"${qt_lib_dir}" \
  -Wl,-rpath,"${lib_dir}" \
  -Wl,-rpath,"${gds_plugin_dir}" \
  -Wl,-rpath,"${qt_lib_dir}" \
  -lklayout_db -lklayout_tl -lgds2 -lQt6Core -lQt6Core5Compat \
  -o "${binary}"

arguments=(
  "--input=$(readlink -f -- "${input}")"
  "--top=${top}"
)
if ((skip_exact_terminal)); then
  arguments+=(--skip-exact-terminal)
fi
exec "${binary}" "${arguments[@]}"
