#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "${here}/../.." && pwd)
source_file="${here}/contact4_active_union_kact_fused_replay.cu"

backend=${KLAYOUT_CONTACT4_ACTIVE_UNION_BACKEND_DSO:-\
/tmp/klayout-contact4-union-resident-build/libklayout_cuda_spatial_backend.so}
kact=${KLAYOUT_CONTACT4_ACTIVE_UNION_KACT:-\
/tmp/contact4-active-contact-kact.om3nRq/active-contact-x2.kact}
repeat=${KLAYOUT_CONTACT4_ACTIVE_UNION_REPEAT:-2}
device=${KLAYOUT_CUDA_SPATIAL_DEVICE:-0}
build_dir=${KLAYOUT_CONTACT4_ACTIVE_UNION_KACT_BUILD:-\
/tmp/klayout-contact4-active-union-kact-fused-build}
binary="${build_dir}/contact4_active_union_kact_fused_replay"

[[ -f "${source_file}" ]] ||
  { echo "missing replay source: ${source_file}" >&2; exit 2; }
[[ -f "${backend}" ]] ||
  { echo "missing fused backend DSO: ${backend}" >&2; exit 2; }
[[ -f "${kact}" ]] ||
  { echo "missing qualified KACT: ${kact}" >&2; exit 2; }
[[ "${repeat}" =~ ^[0-9]+$ ]] && ((repeat >= 2)) ||
  { echo "repeat must be an integer >=2" >&2; exit 2; }
[[ "${device}" =~ ^[0-9]+$ ]] ||
  { echo "device must be a nonnegative integer" >&2; exit 2; }

mkdir -p -- "${build_dir}"
backend=$(readlink -f -- "${backend}")
backend_dir=$(dirname -- "${backend}")

nvcc -std=c++17 -O2 -lineinfo \
  -I"${here}" \
  -I"${root}/src/db/db" \
  "${source_file}" \
  "${backend}" \
  -Xlinker -rpath -Xlinker "${backend_dir}" \
  -o "${binary}"

"${binary}" "${kact}" "${repeat}" "${device}"
