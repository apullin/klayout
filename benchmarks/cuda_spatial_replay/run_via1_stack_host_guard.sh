#!/usr/bin/env bash
set -euo pipefail

usage()
{
  cat >&2 <<'EOF'
Usage:
  run_via1_stack_host_guard.sh --klayout-bin DIR [--cxx PATH] [--keep-work]

Builds the CPU-only adversarial VIA1-stack DSO and links the host-consumer
smoke test against an existing KLayout installation. No CUDA toolkit or GPU is
required. Build products live in a fresh temporary directory.
EOF
}

die()
{
  echo "VIA1-stack host guard: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd -- "${here}/../.." && pwd)
klayout_bin=
cxx=${CXX:-g++}
keep_work=0

while (($#)); do
  case "$1" in
    --klayout-bin)
      (($# >= 2)) || die "--klayout-bin requires a value"
      klayout_bin=$2
      shift 2
      ;;
    --cxx)
      (($# >= 2)) || die "--cxx requires a value"
      cxx=$2
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

[[ -n "${klayout_bin}" ]] || die "missing --klayout-bin"
[[ -d "${klayout_bin}" ]] ||
  die "KLayout binary directory is missing: ${klayout_bin}"
[[ -f "${klayout_bin}/libklayout_db.so" ]] ||
  die "libklayout_db.so is missing from ${klayout_bin}"
cxx=$(command -v -- "${cxx}") || die "C++ compiler is not executable"
klayout_bin=$(readlink -f -- "${klayout_bin}")

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-via1-stack-host-guard.XXXXXX")
cleanup()
{
  if ((keep_work)); then
    echo "VIA1_STACK_HOST_GUARD work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

mapfile -t dependency_dirs < <(
  ldd "${klayout_bin}/libklayout_db.so" |
    awk '/ => \// { path=$3; sub(/\/[^/]+$/, "", path); print path }' |
    sort -u
)
rpath_link_args=("-Wl,-rpath-link,${klayout_bin}")
for directory in "${dependency_dirs[@]}"; do
  rpath_link_args+=("-Wl,-rpath-link,${directory}")
done

"${cxx}" \
  -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -fPIC -shared \
  -DKLAYOUT_CUDA_SPATIAL_BACKEND_BUILD \
  -I"${source_root}/src/db/db" \
  "${here}/via1_stack_fake_backend.cc" \
  -o "${work}/libvia1_stack_fake_backend.so"

"${cxx}" \
  -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -I"${source_root}/src/db/db" \
  "${here}/via1_stack_host_guard_smoke.cc" \
  -L"${klayout_bin}" \
  -Wl,-rpath,"${klayout_bin}" \
  "${rpath_link_args[@]}" \
  -lklayout_db -ldl -pthread \
  -o "${work}/via1_stack_host_guard_smoke"

"${work}/via1_stack_host_guard_smoke" \
  "${work}/libvia1_stack_fake_backend.so"
echo "VIA1_STACK_HOST_GUARD build_and_run=PASS"
