#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build-clang-perf.sh [OPTIONS] [-- BUILD.SH-ARG ...]

Build KLayout with Clang and full LTO. The default profile is CPU-portable.

Options:
  --profile PROFILE  portable (default), auto, or znver2
  --dry-run          resolve and probe the profile, then print without building
  -h, --help         show this help

The auto and znver2 profiles use znver2 code generation only when the build
host is identified as znver2 and C, C++, full-LTO link, and execution probes
all pass. Otherwise they warn and safely fall back to the portable profile.
Nonempty profile build/install directories are reused only when the build
directory contains an exact matching fail-closed configuration manifest.

Tool overrides: CC, CXX, QMAKE
Optional environment:
  KLAYOUT_PERF_OUTPUT_ROOT  parent of profile-specific build/bin directories
  KLAYOUT_PERF_PREFIX       dependency prefix (defaults to CONDA_PREFIX)
  KLAYOUT_QMAKE_SPEC        qmake spec (defaults to linux-clang)
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

shell_join() {
  local item
  printf '  '
  for item in "$@"; do
    printf '%q ' "$item"
  done
  printf '\n'
}

requested_profile=portable
dry_run=0
build_args=()

while (($#)); do
  case "$1" in
    --profile)
      (($# >= 2)) || die '--profile requires an argument'
      requested_profile=$2
      shift 2
      ;;
    --profile=*)
      requested_profile=${1#*=}
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      build_args=("$@")
      break
      ;;
    *)
      die "unknown helper option '$1' (put build.sh arguments after --)"
      ;;
  esac
done

case "$requested_profile" in
  portable|auto|znver2) ;;
  *) die "unsupported profile '$requested_profile'" ;;
esac

# These options would override the helper's safety and output-separation
# guarantees. Keep the escape surface deliberately small.
for argument in "${build_args[@]}"; do
  case "$argument" in
    -qmake|-build|-bin|-prefix|-expert|-debug|-dry-run)
      die "build.sh option '$argument' is managed by this helper"
      ;;
  esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
helper_path=$script_dir/$(basename -- "${BASH_SOURCE[0]}")
source_root=$(cd -- "$script_dir/.." && pwd -P)
build_driver=$source_root/build.sh
[[ -x "$build_driver" ]] || die "build driver is not executable: $build_driver"

resolve_tool() {
  local requested=$1
  local description=$2
  local resolved
  resolved=$(command -v -- "$requested" 2>/dev/null) ||
    die "$description is not executable or not on PATH: $requested"
  [[ -x "$resolved" ]] || die "$description is not executable: $resolved"
  if [[ "$resolved" != /* ]]; then
    resolved=$(cd -- "$(dirname -- "$resolved")" && pwd -P)/$(basename -- "$resolved")
  fi
  printf '%s\n' "$resolved"
}

hash_file() {
  local path=$1
  local output
  if command -v sha256sum >/dev/null 2>&1; then
    output=$(sha256sum -- "$path") || die "could not hash $path"
    printf '%s\n' "${output%% *}"
  elif command -v shasum >/dev/null 2>&1; then
    output=$(shasum -a 256 -- "$path") || die "could not hash $path"
    printf '%s\n' "${output%% *}"
  else
    die 'neither sha256sum nor shasum is available'
  fi
}

cc=$(resolve_tool "${CC:-clang}" 'C compiler')
cxx=$(resolve_tool "${CXX:-clang++}" 'C++ compiler')
qmake=$(resolve_tool "${QMAKE:-qmake6}" 'qmake')
qmake_spec=${KLAYOUT_QMAKE_SPEC:-linux-clang}

if [[ ${KLAYOUT_PERF_PREFIX+x} ]]; then
  dependency_prefix=$KLAYOUT_PERF_PREFIX
else
  dependency_prefix=${CONDA_PREFIX:-}
fi

common_compile_flags=(-flto=full)
common_link_flags=(-flto=full -fuse-ld=lld)
if [[ -n "$dependency_prefix" ]]; then
  if [[ -d "$dependency_prefix/include" ]]; then
    common_compile_flags+=("-I$dependency_prefix/include")
  fi
  if [[ -d "$dependency_prefix/lib" ]]; then
    common_link_flags+=(
      "-L$dependency_prefix/lib"
      "-Wl,-rpath,$dependency_prefix/lib"
    )
  fi
fi

probe_root=$(mktemp -d "${TMPDIR:-/tmp}/klayout-clang-perf-probe.XXXXXX")
cleanup() {
  rm -rf -- "$probe_root"
}
trap cleanup EXIT

cat >"$probe_root/probe.c" <<'EOF'
int klayout_perf_probe_c(void) { return 17; }
EOF
cat >"$probe_root/probe.cc" <<'EOF'
extern "C" int klayout_perf_probe_c(void);
int main() { return klayout_perf_probe_c() == 17 ? 0 : 1; }
EOF

probe_reason=
probe_log=
probe_profile() {
  local profile=$1
  shift
  local -a target_flags=("$@")
  local -a compile_flags=(-O2 "${common_compile_flags[@]}" "${target_flags[@]}")
  local -a link_flags=("${common_link_flags[@]}" "${target_flags[@]}")
  local c_object=$probe_root/$profile-c.o
  local cxx_object=$probe_root/$profile-cxx.o
  local executable=$probe_root/$profile-probe

  probe_log=$probe_root/$profile.log
  : >"$probe_log"
  rm -f -- "$c_object" "$cxx_object" "$executable"

  if ! "$cc" "${compile_flags[@]}" -c "$probe_root/probe.c" \
      -o "$c_object" >>"$probe_log" 2>&1; then
    probe_reason='C compilation'
    return 1
  fi
  if ! "$cxx" "${compile_flags[@]}" -c "$probe_root/probe.cc" \
      -o "$cxx_object" >>"$probe_log" 2>&1; then
    probe_reason='C++ compilation'
    return 1
  fi
  if ! "$cxx" "${link_flags[@]}" "$c_object" "$cxx_object" \
      -o "$executable" >>"$probe_log" 2>&1; then
    probe_reason='full-LTO link'
    return 1
  fi
  if ! "$executable" >>"$probe_log" 2>&1; then
    probe_reason='linked-program execution'
    return 1
  fi
  probe_reason=
  return 0
}

host_reason=
host_is_znver2() {
  local macros
  if ! macros=$("$cc" -march=native -dM -E -x c /dev/null 2>&1); then
    host_reason='the compiler could not identify the native CPU'
    return 1
  fi
  if grep -Fqx '#define __znver2__ 1' <<<"$macros"; then
    host_reason=
    return 0
  fi
  host_reason='the compiler does not identify this host as znver2'
  return 1
}

resolved_profile=portable
if [[ "$requested_profile" != portable ]]; then
  if ! host_is_znver2; then
    warn "requested profile '$requested_profile': $host_reason; falling back to portable"
  elif probe_profile znver2 -march=znver2 -mtune=znver2; then
    resolved_profile=znver2
  else
    warn "requested profile '$requested_profile': znver2 $probe_reason probe failed; falling back to portable"
  fi
fi

if [[ "$resolved_profile" == portable ]]; then
  if ! probe_profile portable; then
    printf 'ERROR: portable %s probe failed; refusing a non-LTO build\n' \
      "$probe_reason" >&2
    if [[ -s "$probe_log" ]]; then
      sed 's/^/  | /' "$probe_log" >&2
    fi
    exit 1
  fi
fi

target_flags=()
if [[ "$resolved_profile" == znver2 ]]; then
  target_flags=(-march=znver2 -mtune=znver2)
fi
compile_flags=("${common_compile_flags[@]}" "${target_flags[@]}")
link_flags=("${common_link_flags[@]}" "${target_flags[@]}")

output_root=${KLAYOUT_PERF_OUTPUT_ROOT:-$source_root}
if [[ "$output_root" != /* ]]; then
  output_root=$source_root/$output_root
fi
build_directory=$output_root/build-clang-perf-$resolved_profile
bin_directory=$output_root/bin-clang-perf-$resolved_profile

manifest_name=.klayout-clang-perf-manifest-v1
manifest_path=$build_directory/$manifest_name
expected_manifest=$probe_root/$manifest_name

manifest_scalar() {
  local key=$1
  local value=$2
  printf '%s=' "$key"
  printf '%q' "$value"
  printf '\n'
}

manifest_array() {
  local key=$1
  shift
  local index=0
  local value
  manifest_scalar "$key.count" "$#"
  for value in "$@"; do
    manifest_scalar "$key.$index" "$value"
    ((index += 1))
  done
}

cc_hash=$(hash_file "$cc")
cxx_hash=$(hash_file "$cxx")
qmake_hash=$(hash_file "$qmake")
helper_hash=$(hash_file "$helper_path")
build_driver_hash=$(hash_file "$build_driver")

{
  manifest_scalar manifest_schema klayout-clang-perf-v1
  manifest_scalar resolved_profile "$resolved_profile"
  manifest_scalar source_root "$source_root"
  manifest_scalar build_directory "$build_directory"
  manifest_scalar bin_directory "$bin_directory"
  manifest_scalar helper_path "$helper_path"
  manifest_scalar helper_sha256 "$helper_hash"
  manifest_scalar build_driver_path "$build_driver"
  manifest_scalar build_driver_sha256 "$build_driver_hash"
  manifest_scalar cc_path "$cc"
  manifest_scalar cc_sha256 "$cc_hash"
  manifest_scalar cxx_path "$cxx"
  manifest_scalar cxx_sha256 "$cxx_hash"
  manifest_scalar qmake_path "$qmake"
  manifest_scalar qmake_sha256 "$qmake_hash"
  manifest_scalar qmake_spec "$qmake_spec"
  manifest_scalar dependency_prefix "$dependency_prefix"
  manifest_array compile_flags "${compile_flags[@]}"
  manifest_array link_flags "${link_flags[@]}"
  manifest_array build_arguments "${build_args[@]}"
} >"$expected_manifest"
manifest_hash=$(hash_file "$expected_manifest")

directory_has_entries() (
  local directory=$1
  local -a entries
  shopt -s nullglob dotglob
  entries=("$directory"/*)
  ((${#entries[@]} > 0))
)

manifest_state=
validate_artifact_directories() {
  local build_nonempty=0
  local bin_nonempty=0

  if [[ -L "$build_directory" ]] ||
      [[ -e "$build_directory" && ! -d "$build_directory" ]]; then
    die "build path is not a real directory: $build_directory"
  fi
  if [[ -L "$bin_directory" ]] ||
      [[ -e "$bin_directory" && ! -d "$bin_directory" ]]; then
    die "install path is not a real directory: $bin_directory"
  fi
  if [[ -d "$build_directory" ]] && directory_has_entries "$build_directory"; then
    build_nonempty=1
  fi
  if [[ -d "$bin_directory" ]] && directory_has_entries "$bin_directory"; then
    bin_nonempty=1
  fi

  if ((build_nonempty == 0 && bin_nonempty == 0)); then
    manifest_state=new
    return 0
  fi
  if [[ ! -f "$manifest_path" || -L "$manifest_path" ]]; then
    die "existing build/install artifacts have no regular $manifest_name; refusing to mix configurations (select a fresh KLAYOUT_PERF_OUTPUT_ROOT or remove them after inspection)"
  fi
  if ! cmp -s -- "$expected_manifest" "$manifest_path"; then
    die "existing $manifest_name does not exactly match this toolchain/configuration; refusing to mix configurations (select a fresh KLAYOUT_PERF_OUTPUT_ROOT or remove the artifacts after inspection)"
  fi
  manifest_state=matching-resume
}

install_manifest_for_new_build() {
  local temporary_manifest
  mkdir -p -- "$build_directory"

  # Recheck both artifact directories after mkdir so a racing creator cannot
  # silently introduce an unmanifested build between validation and writing.
  validate_artifact_directories
  if [[ "$manifest_state" == matching-resume ]]; then
    return 0
  fi
  [[ "$manifest_state" == new ]] ||
    die 'artifact-directory state changed while preparing the manifest'

  temporary_manifest=$build_directory/.$manifest_name.tmp.$$
  (umask 022; cp -- "$expected_manifest" "$temporary_manifest")
  chmod 0444 "$temporary_manifest"
  mv -- "$temporary_manifest" "$manifest_path"
  manifest_state=created
}

validate_artifact_directories
if ((dry_run == 0)) && [[ "$manifest_state" == new ]]; then
  install_manifest_for_new_build
fi

printf 'KLayout Clang performance build\n'
printf '  requested profile: %s\n' "$requested_profile"
printf '  resolved profile:  %s\n' "$resolved_profile"
printf '  C compiler:       %s\n' "$cc"
printf '  C++ compiler:     %s\n' "$cxx"
printf '  qmake:             %s\n' "$qmake"
printf '  compile additions:'
printf ' %q' "${compile_flags[@]}"
printf '\n'
printf '  link additions:   '
printf ' %q' "${link_flags[@]}"
printf '\n'
printf '  build directory:   %s\n' "$build_directory"
printf '  install directory: %s\n' "$bin_directory"
printf '  manifest:           %s\n' "$manifest_path"
printf '  manifest SHA-256:   %s\n' "$manifest_hash"
if ((dry_run)) && [[ "$manifest_state" == new ]]; then
  printf '  manifest status:    new (dry-run; not written)\n'
else
  printf '  manifest status:    %s\n' "$manifest_state"
fi

qmake_prefix=(-spec "$qmake_spec")
qmake_assignments=(
  "QMAKE_CC=$cc"
  "QMAKE_CXX=$cxx"
  "QMAKE_LINK=$cxx"
  "QMAKE_LINK_SHLIB=$cxx"
  "QMAKE_LINK_C=$cc"
  "QMAKE_LINK_C_SHLIB=$cc"
  "QMAKE_CFLAGS+=${compile_flags[*]}"
  "QMAKE_CXXFLAGS+=${compile_flags[*]}"
  "QMAKE_LFLAGS+=${link_flags[*]}"
)

if ((dry_run)); then
  printf '  mode:               dry-run (probes passed; build.sh will not run)\n'
  printf '  qmake invocation additions:\n'
  shell_join "${qmake_prefix[@]}" '<build.sh qmake arguments>' \
    "${qmake_assignments[@]}"
  printf '  build.sh invocation:\n'
  shell_join "$build_driver" -release -qmake '<generated-qmake-wrapper>' \
    -build "$build_directory" -bin "$bin_directory" "${build_args[@]}"
  exit 0
fi

qmake_wrapper=$probe_root/qmake-clang-perf
{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'exec %q' "$qmake"
  local_argument=
  for local_argument in "${qmake_prefix[@]}"; do
    printf ' %q' "$local_argument"
  done
  printf ' "$@"'
  for local_argument in "${qmake_assignments[@]}"; do
    printf ' %q' "$local_argument"
  done
  printf '\n'
} >"$qmake_wrapper"
chmod +x "$qmake_wrapper"

build_command=(
  "$build_driver"
  -release
  -qmake "$qmake_wrapper"
  -build "$build_directory"
  -bin "$bin_directory"
  "${build_args[@]}"
)

printf '  build.sh invocation:\n'
shell_join "${build_command[@]}"
"${build_command[@]}"
