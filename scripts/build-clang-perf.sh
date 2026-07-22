#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build-clang-perf.sh [OPTIONS] [-- BUILD.SH-ARG ...]

Build KLayout with Clang and full LTO. The default profile is CPU-portable.

Options:
  --profile PROFILE  portable (default), auto, or znver2
  --pgo MODE         none (default), control, generate, or use
  --pgo-profile PATH immutable regular .profdata input (required with use)
  --pgo-profile-manifest PATH
                     immutable regular JSON evidence (required with use)
  --dry-run          resolve and probe the profile, then print without building
  -h, --help         show this help

The auto and znver2 profiles use znver2 code generation only when the build
host is identified as znver2 and C, C++, full-LTO link, and execution probes
all pass. Otherwise they warn and safely fall back to the portable profile.
Nonempty profile build/install directories are reused only when the build
directory contains an exact matching fail-closed configuration manifest.
PGO control/generate/use builds additionally require a clean Git source tree
and use identity-qualified, transactional artifact directories. Control has no
PGO flags and is the formal same-source comparison phase. Generate builds use
atomic counters and a build-local default raw-profile directory;
multi-process/module training must override LLVM_PROFILE_FILE with a pattern
containing both %p and %m (for example, /path/to/raw/%m_%p.profraw).

Tool overrides: CC, CXX, QMAKE, JSON_PYTHON
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
pgo_mode=none
pgo_profile_argument=
pgo_profile_manifest_argument=
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
    --pgo)
      (($# >= 2)) || die '--pgo requires an argument'
      pgo_mode=$2
      shift 2
      ;;
    --pgo=*)
      pgo_mode=${1#*=}
      shift
      ;;
    --pgo-profile)
      (($# >= 2)) || die '--pgo-profile requires an argument'
      pgo_profile_argument=$2
      shift 2
      ;;
    --pgo-profile=*)
      pgo_profile_argument=${1#*=}
      shift
      ;;
    --pgo-profile-manifest)
      (($# >= 2)) || die '--pgo-profile-manifest requires an argument'
      pgo_profile_manifest_argument=$2
      shift 2
      ;;
    --pgo-profile-manifest=*)
      pgo_profile_manifest_argument=${1#*=}
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
case "$pgo_mode" in
  none|control|generate|use) ;;
  *) die "unsupported PGO mode '$pgo_mode'" ;;
esac
if [[ "$pgo_mode" == use ]]; then
  [[ -n "$pgo_profile_argument" ]] ||
    die "--pgo use requires --pgo-profile PATH"
  [[ -n "$pgo_profile_manifest_argument" ]] ||
    die "--pgo use requires --pgo-profile-manifest PATH"
elif [[ -n "$pgo_profile_argument" || -n "$pgo_profile_manifest_argument" ]]; then
  die "--pgo-profile and --pgo-profile-manifest are valid only with --pgo use"
fi

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

canonical_regular_file() {
  local path=$1
  local description=$2
  local parent
  [[ ! -L "$path" ]] || die "$description must not be a symbolic link: $path"
  [[ -f "$path" ]] || die "$description is not a regular file: $path"
  parent=$(cd -- "$(dirname -- "$path")" && pwd -P) ||
    die "could not resolve $description parent: $path"
  printf '%s/%s\n' "$parent" "$(basename -- "$path")"
}

file_permissions() {
  local path=$1
  local permissions
  permissions=$(stat -c '%A' -- "$path" 2>/dev/null) ||
    die "could not inspect file permissions: $path"
  printf '%s\n' "$permissions"
}

require_immutable_regular_file() {
  local path=$1
  local description=$2
  local permissions
  [[ ! -L "$path" && -f "$path" ]] ||
    die "$description ceased to be a regular non-symlink file: $path"
  permissions=$(file_permissions "$path")
  [[ "$permissions" != *w* && ! -w "$path" ]] ||
    die "$description must be immutable (remove every write permission): $path"
}

cc=$(resolve_tool "${CC:-clang}" 'C compiler')
cxx=$(resolve_tool "${CXX:-clang++}" 'C++ compiler')
qmake=$(resolve_tool "${QMAKE:-qmake6}" 'qmake')
qmake_spec=${KLAYOUT_QMAKE_SPEC:-linux-clang}

pgo_profile_path=
pgo_profile_hash=
pgo_profile_permissions=
pgo_profile_manifest_path=
pgo_profile_manifest_hash=
pgo_profile_manifest_permissions=
json_python=
json_python_hash=
if [[ "$pgo_mode" == use ]]; then
  pgo_profile_path=$(canonical_regular_file "$pgo_profile_argument" 'PGO profile')
  require_immutable_regular_file "$pgo_profile_path" 'PGO profile'
  pgo_profile_hash=$(hash_file "$pgo_profile_path")
  pgo_profile_permissions=$(file_permissions "$pgo_profile_path")
  pgo_profile_manifest_path=$(canonical_regular_file \
    "$pgo_profile_manifest_argument" 'PGO profile manifest')
  require_immutable_regular_file "$pgo_profile_manifest_path" \
    'PGO profile manifest'
  pgo_profile_manifest_hash=$(hash_file "$pgo_profile_manifest_path")
  pgo_profile_manifest_permissions=$(file_permissions \
    "$pgo_profile_manifest_path")
  json_python=$(resolve_tool "${JSON_PYTHON:-python3}" 'Python JSON validator')
  json_python_hash=$(hash_file "$json_python")
  if ! "$json_python" -c \
      'import json, sys; data = json.load(open(sys.argv[1], "rb")); assert isinstance(data, dict)' \
      "$pgo_profile_manifest_path" >/dev/null 2>&1; then
    die "PGO profile manifest is not a valid JSON object: $pgo_profile_manifest_path"
  fi
fi

git_tool=
git_hash=
source_commit=
source_tree=
require_clean_source_identity() {
  local status
  git_tool=$(resolve_tool "${GIT:-git}" 'Git')
  GIT_OPTIONAL_LOCKS=0 "$git_tool" -C "$source_root" \
    rev-parse --is-inside-work-tree \
    >/dev/null 2>&1 || die "PGO builds require a Git worktree: $source_root"
  source_commit=$(GIT_OPTIONAL_LOCKS=0 "$git_tool" -C "$source_root" \
    rev-parse --verify HEAD) ||
    die 'PGO builds require a valid source commit'
  source_tree=$(GIT_OPTIONAL_LOCKS=0 "$git_tool" -C "$source_root" \
    rev-parse --verify 'HEAD^{tree}') ||
    die 'PGO builds require a valid committed source tree'
  status=$(GIT_OPTIONAL_LOCKS=0 "$git_tool" -C "$source_root" \
    status --porcelain=v1 --untracked-files=all) ||
    die 'could not verify PGO source-tree cleanliness'
  [[ -z "$status" ]] ||
    die 'PGO builds require a clean source tree (tracked and untracked files must be committed, ignored, or removed)'
  git_hash=$(hash_file "$git_tool")
}

if [[ "$pgo_mode" != none ]]; then
  require_clean_source_identity
fi

verify_clean_source_identity() {
  local current_commit
  local current_tree
  local status
  [[ "$pgo_mode" != none ]] || return 0
  current_commit=$(GIT_OPTIONAL_LOCKS=0 "$git_tool" -C "$source_root" \
    rev-parse --verify HEAD) ||
    die 'PGO source commit could not be revalidated'
  current_tree=$(GIT_OPTIONAL_LOCKS=0 "$git_tool" -C "$source_root" \
    rev-parse --verify 'HEAD^{tree}') ||
    die 'PGO source tree could not be revalidated'
  status=$(GIT_OPTIONAL_LOCKS=0 "$git_tool" -C "$source_root" \
    status --porcelain=v1 --untracked-files=all) ||
    die 'could not revalidate PGO source-tree cleanliness'
  [[ -z "$status" && "$current_commit" == "$source_commit" &&
    "$current_tree" == "$source_tree" ]] ||
    die 'PGO source identity changed after configuration; refusing mixed artifacts'
}

verify_pgo_profile_identity() {
  local current_hash
  local current_permissions
  local current_manifest_hash
  local current_manifest_permissions
  [[ "$pgo_mode" == use ]] || return 0
  require_immutable_regular_file "$pgo_profile_path" 'PGO profile'
  current_hash=$(hash_file "$pgo_profile_path")
  current_permissions=$(file_permissions "$pgo_profile_path")
  [[ "$current_hash" == "$pgo_profile_hash" &&
    "$current_permissions" == "$pgo_profile_permissions" ]] ||
    die 'PGO profile identity changed after configuration; refusing mixed artifacts'
  require_immutable_regular_file "$pgo_profile_manifest_path" \
    'PGO profile manifest'
  current_manifest_hash=$(hash_file "$pgo_profile_manifest_path")
  current_manifest_permissions=$(file_permissions "$pgo_profile_manifest_path")
  [[ "$current_manifest_hash" == "$pgo_profile_manifest_hash" &&
    "$current_manifest_permissions" == "$pgo_profile_manifest_permissions" ]] ||
    die 'PGO profile manifest identity changed after configuration; refusing mixed artifacts'
}

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

output_root=${KLAYOUT_PERF_OUTPUT_ROOT:-$source_root}
if [[ "$output_root" != /* ]]; then
  output_root=$source_root/$output_root
fi
artifact_profile=$resolved_profile
case "$pgo_mode" in
  none) ;;
  control)
    artifact_profile+=-pgo-control
    ;;
  generate)
    artifact_profile+=-pgo-generate
    ;;
  use)
    artifact_profile+=-pgo-use-${pgo_profile_hash:0:16}-manifest-${pgo_profile_manifest_hash:0:16}
    ;;
esac
if [[ "$pgo_mode" != none ]]; then
  artifact_profile+=-commit-${source_commit:0:16}-tree-${source_tree:0:16}
fi
build_directory=$output_root/build-clang-perf-$artifact_profile
bin_directory=$output_root/bin-clang-perf-$artifact_profile

pgo_lifecycle_schema=
pgo_lifecycle_path=
pgo_lifecycle_lock_path=
if [[ "$pgo_mode" != none ]]; then
  pgo_lifecycle_schema=klayout-clang-perf-pgo-lifecycle-v1
  pgo_lifecycle_path=$build_directory/.klayout-clang-perf-pgo-state-v1
  pgo_lifecycle_lock_path=$build_directory/.klayout-clang-perf-pgo-lock-v1
fi

pgo_default_profile_directory=
pgo_staged_profile_path=
pgo_staged_profile_manifest_path=
pgo_flags=()
case "$pgo_mode" in
  none|control) ;;
  generate)
    pgo_default_profile_directory=$build_directory/pgo-default-profraw-discard
    pgo_flags=(
      "-fprofile-generate=$pgo_default_profile_directory"
      -fprofile-update=atomic
    )
    ;;
  use)
    pgo_staged_profile_path=$build_directory/pgo-input-$pgo_profile_hash.profdata
    pgo_staged_profile_manifest_path=$build_directory/pgo-profile-manifest-$pgo_profile_manifest_hash.json
    pgo_flags=(
      "-fprofile-use=$pgo_staged_profile_path"
      -Werror=profile-instr-out-of-date
    )
    ;;
esac

# Probe the PGO flag family separately from target selection. A PGO failure is
# fatal: never turn an explicitly requested PGO build into a non-PGO build.
if [[ "$pgo_mode" == generate || "$pgo_mode" == use ]]; then
  probe_pgo_flags=()
  if [[ "$pgo_mode" == generate ]]; then
    probe_profile_directory=$probe_root/pgo-default-profraw
    mkdir -p -- "$probe_profile_directory"
    probe_pgo_flags=(
      "-fprofile-generate=$probe_profile_directory"
      -fprofile-update=atomic
    )
  else
    # The synthetic probe has no KLayout symbols and therefore cannot validate
    # whether the training counts apply to this source tree. It checks only
    # that Clang can read the indexed profile and accept the use-mode flags.
    # The real build retains the out-of-date diagnostic as an error; suppress
    # only the expected synthetic probe's no-profile-data warning here.
    probe_pgo_flags=(
      "-fprofile-use=$pgo_profile_path"
      -Werror=profile-instr-out-of-date
      -Wno-profile-instr-unprofiled
    )
  fi
  if ! probe_profile pgo-${resolved_profile}-${pgo_mode} \
      "${target_flags[@]}" "${probe_pgo_flags[@]}"; then
    printf 'ERROR: PGO %s %s probe failed; refusing a non-PGO build\n' \
      "$pgo_mode" "$probe_reason" >&2
    if [[ -s "$probe_log" ]]; then
      sed 's/^/  | /' "$probe_log" >&2
    fi
    exit 1
  fi
fi

compile_flags=(
  "${common_compile_flags[@]}"
  "${target_flags[@]}"
  "${pgo_flags[@]}"
)
link_flags=(
  "${common_link_flags[@]}"
  "${target_flags[@]}"
  "${pgo_flags[@]}"
)

manifest_name=.klayout-clang-perf-manifest-v2
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
  manifest_scalar manifest_schema klayout-clang-perf-v2
  manifest_scalar resolved_profile "$resolved_profile"
  manifest_scalar pgo_mode "$pgo_mode"
  manifest_scalar pgo_profile_source_path "$pgo_profile_path"
  manifest_scalar pgo_profile_sha256 "$pgo_profile_hash"
  manifest_scalar pgo_profile_source_permissions "$pgo_profile_permissions"
  manifest_scalar pgo_profile_build_path "$pgo_staged_profile_path"
  manifest_scalar pgo_profile_manifest_source_path "$pgo_profile_manifest_path"
  manifest_scalar pgo_profile_manifest_sha256 "$pgo_profile_manifest_hash"
  manifest_scalar pgo_profile_manifest_source_permissions "$pgo_profile_manifest_permissions"
  manifest_scalar pgo_profile_manifest_build_path "$pgo_staged_profile_manifest_path"
  manifest_scalar pgo_default_profile_directory "$pgo_default_profile_directory"
  manifest_scalar pgo_lifecycle_schema "$pgo_lifecycle_schema"
  manifest_scalar pgo_lifecycle_path "$pgo_lifecycle_path"
  manifest_scalar source_commit "$source_commit"
  manifest_scalar source_tree "$source_tree"
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
  manifest_scalar git_path "$git_tool"
  manifest_scalar git_sha256 "$git_hash"
  manifest_scalar json_python_path "$json_python"
  manifest_scalar json_python_sha256 "$json_python_hash"
  manifest_scalar qmake_spec "$qmake_spec"
  manifest_scalar dependency_prefix "$dependency_prefix"
  manifest_array compile_flags "${compile_flags[@]}"
  manifest_array link_flags "${link_flags[@]}"
  manifest_array build_arguments "${build_args[@]}"
} >"$expected_manifest"
manifest_hash=$(hash_file "$expected_manifest")

expected_pgo_building_state=
expected_pgo_complete_state=
if [[ "$pgo_mode" != none ]]; then
  expected_pgo_building_state=$probe_root/pgo-state-building
  expected_pgo_complete_state=$probe_root/pgo-state-complete
  {
    manifest_scalar lifecycle_schema "$pgo_lifecycle_schema"
    manifest_scalar status building
    manifest_scalar manifest_sha256 "$manifest_hash"
  } >"$expected_pgo_building_state"
  {
    manifest_scalar lifecycle_schema "$pgo_lifecycle_schema"
    manifest_scalar status complete
    manifest_scalar manifest_sha256 "$manifest_hash"
  } >"$expected_pgo_complete_state"
fi

directory_has_entries() (
  local directory=$1
  local -a entries
  shopt -s nullglob dotglob
  entries=("$directory"/*)
  ((${#entries[@]} > 0))
)

validate_complete_pgo_lifecycle() {
  [[ "$pgo_mode" != none ]] || return 0
  if [[ -e "$pgo_lifecycle_lock_path" || -L "$pgo_lifecycle_lock_path" ]]; then
    die "PGO build lifecycle lock exists; refusing interrupted or concurrent artifacts: $pgo_lifecycle_lock_path"
  fi
  if [[ ! -f "$pgo_lifecycle_path" || -L "$pgo_lifecycle_path" ]]; then
    die "PGO artifacts have no regular complete lifecycle state; refusing interrupted or incomplete build: $pgo_lifecycle_path"
  fi
  if [[ $(file_permissions "$pgo_lifecycle_path") == *w* ||
      -w "$pgo_lifecycle_path" ]]; then
    die 'PGO build lifecycle state is writable; refusing mutable provenance'
  fi
  if ! cmp -s -- "$expected_pgo_complete_state" "$pgo_lifecycle_path"; then
    die 'PGO build lifecycle state is not exactly complete for this manifest; refusing interrupted or mixed artifacts'
  fi
}

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
  if [[ $(file_permissions "$manifest_path") == *w* || -w "$manifest_path" ]]; then
    die "existing $manifest_name is writable; refusing mutable build provenance"
  fi
  if ! cmp -s -- "$expected_manifest" "$manifest_path"; then
    die "existing $manifest_name does not exactly match this toolchain/configuration; refusing to mix configurations (select a fresh KLAYOUT_PERF_OUTPUT_ROOT or remove the artifacts after inspection)"
  fi
  validate_complete_pgo_lifecycle
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

stage_bound_pgo_file() {
  local source_path=$1
  local expected_hash=$2
  local staged_path=$3
  local description=$4
  local temporary_path
  local staged_hash

  if [[ -e "$staged_path" || -L "$staged_path" ]]; then
    [[ ! -L "$staged_path" && -f "$staged_path" ]] ||
      die "staged $description is not a regular file: $staged_path"
    require_immutable_regular_file "$staged_path" "staged $description"
    staged_hash=$(hash_file "$staged_path")
    [[ "$staged_hash" == "$expected_hash" ]] ||
      die "staged $description does not match its manifest identity"
    return 0
  fi

  temporary_path=$(mktemp "$build_directory/.pgo-input.tmp.XXXXXX") ||
    die "could not create temporary staged $description"
  cp -- "$source_path" "$temporary_path"
  chmod 0444 "$temporary_path"
  staged_hash=$(hash_file "$temporary_path")
  [[ "$staged_hash" == "$expected_hash" ]] || {
    rm -f -- "$temporary_path"
    die "$description changed while staging it for the build"
  }
  mv -- "$temporary_path" "$staged_path"
  require_immutable_regular_file "$staged_path" "staged $description"
}

verify_staged_pgo_file() {
  local staged_path=$1
  local expected_hash=$2
  local description=$3
  local staged_hash
  [[ ! -L "$staged_path" && -f "$staged_path" ]] ||
    die "staged $description changed after configuration"
  require_immutable_regular_file "$staged_path" "staged $description"
  staged_hash=$(hash_file "$staged_path")
  [[ "$staged_hash" == "$expected_hash" ]] ||
    die "staged $description changed after configuration"
}

prepare_pgo_artifacts() {

  case "$pgo_mode" in
    none|control)
      return 0
      ;;
    generate)
      if [[ -L "$pgo_default_profile_directory" ]] ||
          [[ -e "$pgo_default_profile_directory" &&
            ! -d "$pgo_default_profile_directory" ]]; then
        die "PGO default-profile path is not a real directory: $pgo_default_profile_directory"
      fi
      mkdir -p -- "$pgo_default_profile_directory"
      ;;
    use)
      verify_pgo_profile_identity
      stage_bound_pgo_file "$pgo_profile_path" "$pgo_profile_hash" \
        "$pgo_staged_profile_path" 'PGO profile'
      stage_bound_pgo_file "$pgo_profile_manifest_path" \
        "$pgo_profile_manifest_hash" "$pgo_staged_profile_manifest_path" \
        'PGO profile manifest'
      ;;
  esac
}

verify_pgo_build_artifacts() {
  case "$pgo_mode" in
    none|control) ;;
    generate)
      [[ ! -L "$pgo_default_profile_directory" &&
        -d "$pgo_default_profile_directory" ]] ||
        die 'PGO default-profile directory changed after configuration'
      ;;
    use)
      verify_staged_pgo_file "$pgo_staged_profile_path" \
        "$pgo_profile_hash" 'PGO profile'
      verify_staged_pgo_file "$pgo_staged_profile_manifest_path" \
        "$pgo_profile_manifest_hash" 'PGO profile manifest'
      ;;
  esac
}

verify_bound_executable() {
  local description=$1
  local path=$2
  local expected_hash=$3
  local current_hash
  [[ -f "$path" && -x "$path" ]] ||
    die "$description ceased to be an executable regular file: $path"
  current_hash=$(hash_file "$path")
  [[ "$current_hash" == "$expected_hash" ]] ||
    die "$description changed after configuration; refusing mixed artifacts"
}

verify_all_pgo_bound_inputs() {
  [[ "$pgo_mode" != none ]] || return 0
  verify_bound_executable 'build helper' "$helper_path" "$helper_hash"
  verify_bound_executable 'build driver' "$build_driver" "$build_driver_hash"
  verify_bound_executable 'C compiler' "$cc" "$cc_hash"
  verify_bound_executable 'C++ compiler' "$cxx" "$cxx_hash"
  verify_bound_executable 'qmake' "$qmake" "$qmake_hash"
  verify_bound_executable 'Git' "$git_tool" "$git_hash"
  if [[ "$pgo_mode" == use ]]; then
    verify_bound_executable 'Python JSON validator' "$json_python" \
      "$json_python_hash"
  fi
  verify_clean_source_identity
  verify_pgo_profile_identity
  verify_pgo_build_artifacts
  [[ -f "$manifest_path" && ! -L "$manifest_path" ]] ||
    die 'PGO build manifest changed after configuration'
  if [[ $(file_permissions "$manifest_path") == *w* || -w "$manifest_path" ]]; then
    die 'PGO build manifest became writable after configuration'
  fi
  cmp -s -- "$expected_manifest" "$manifest_path" ||
    die 'PGO build manifest changed after configuration'
}

publish_pgo_lifecycle_state() {
  local expected_state=$1
  local temporary_state
  temporary_state=$(mktemp "$build_directory/.pgo-state.tmp.XXXXXX") ||
    die 'could not create temporary PGO lifecycle state'
  cp -- "$expected_state" "$temporary_state"
  chmod 0444 "$temporary_state"
  mv -- "$temporary_state" "$pgo_lifecycle_path"
}

begin_pgo_build_lifecycle() {
  [[ "$pgo_mode" != none ]] || return 0
  if ! mkdir -- "$pgo_lifecycle_lock_path" 2>/dev/null; then
    die "could not acquire PGO build lifecycle lock; refusing concurrent or interrupted artifacts: $pgo_lifecycle_lock_path"
  fi

  # Recheck the state after acquiring the atomic lock. Never remove this lock
  # on failure: a killed or failed process must leave artifacts non-resumable.
  if [[ "$manifest_state" == matching-resume ]]; then
    [[ -f "$pgo_lifecycle_path" && ! -L "$pgo_lifecycle_path" ]] ||
      die 'complete PGO lifecycle state disappeared before rebuild'
    cmp -s -- "$expected_pgo_complete_state" "$pgo_lifecycle_path" ||
      die 'complete PGO lifecycle state changed before rebuild'
  elif [[ "$manifest_state" == created ]]; then
    [[ ! -e "$pgo_lifecycle_path" && ! -L "$pgo_lifecycle_path" ]] ||
      die 'new PGO build unexpectedly already has lifecycle state'
  else
    die 'PGO build lifecycle cannot start from this manifest state'
  fi
  publish_pgo_lifecycle_state "$expected_pgo_building_state"
}

complete_pgo_build_lifecycle() {
  [[ "$pgo_mode" != none ]] || return 0
  [[ -d "$pgo_lifecycle_lock_path" &&
    ! -L "$pgo_lifecycle_lock_path" ]] ||
    die 'PGO build lifecycle lock disappeared during the build'
  [[ -f "$pgo_lifecycle_path" && ! -L "$pgo_lifecycle_path" ]] ||
    die 'PGO building lifecycle state disappeared during the build'
  if [[ $(file_permissions "$pgo_lifecycle_path") == *w* ||
      -w "$pgo_lifecycle_path" ]]; then
    die 'PGO building lifecycle state became writable during the build'
  fi
  cmp -s -- "$expected_pgo_building_state" "$pgo_lifecycle_path" ||
    die 'PGO building lifecycle state changed during the build'
  publish_pgo_lifecycle_state "$expected_pgo_complete_state"
  rmdir -- "$pgo_lifecycle_lock_path" ||
    die 'could not release PGO build lifecycle lock after completion'
  validate_complete_pgo_lifecycle
}

validate_artifact_directories
if ((dry_run == 0)); then
  verify_clean_source_identity
  verify_pgo_profile_identity
  if [[ "$manifest_state" == new ]]; then
    install_manifest_for_new_build
  fi
  prepare_pgo_artifacts
fi

printf 'KLayout Clang performance build\n'
printf '  requested profile: %s\n' "$requested_profile"
printf '  resolved profile:  %s\n' "$resolved_profile"
printf '  PGO mode:          %s\n' "$pgo_mode"
if [[ "$pgo_mode" == generate ]]; then
  printf '  default profiles:   %s (discard; training LLVM_PROFILE_FILE must contain %%m and %%p)\n' \
    "$pgo_default_profile_directory"
elif [[ "$pgo_mode" == use ]]; then
  printf '  PGO profile:        %s\n' "$pgo_profile_path"
  printf '  PGO profile SHA:    %s\n' "$pgo_profile_hash"
  printf '  profile manifest:   %s\n' "$pgo_profile_manifest_path"
  printf '  manifest SHA:       %s\n' "$pgo_profile_manifest_hash"
  printf '  PGO probe scope:    profile format/toolchain only; KLayout applicability is checked during compilation\n'
fi
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
verify_all_pgo_bound_inputs
begin_pgo_build_lifecycle
"${build_command[@]}"
verify_all_pgo_bound_inputs
complete_pgo_build_lifecycle
