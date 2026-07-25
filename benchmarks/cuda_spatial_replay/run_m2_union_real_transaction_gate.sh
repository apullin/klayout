#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build="${1:-$root/build}"
scene="${KLAYOUT_M2_UNION_REAL_KACT:-/tmp/m2-via1-x2.39i7kG/m2-via1-x2.kact}"
oracle="${KLAYOUT_M2_UNION_REAL_ORACLE:-/tmp/m2-width-space-census-exact.km1ws}"
scratch="${KLAYOUT_M2_UNION_REAL_EVIDENCE_ROOT:-/home/pullin/personal/klayout/.scratchpad/cuda-runs}"
mkdir -p "$scratch"
out=$(mktemp -d "$scratch/m2-union-real-transaction.XXXXXX")
mkdir -p "$out/tmp"

test -f "$scene"
test -f "$oracle"
cmake --build "$build" --parallel 32 --target \
  klayout_cuda_spatial_backend \
  m2_union_real_transaction_gate

nvidia-smi \
  --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$out/gpu-before.txt"
/usr/bin/time \
  -f 'wall_s=%e user_s=%U system_s=%S max_rss_kib=%M exit=%x' \
  -o "$out/time.txt" \
  env TMPDIR="$out/tmp" \
    "$build/m2_union_real_transaction_gate" \
      "$build/libklayout_cuda_spatial_backend.so" \
      "$scene" "$oracle" >"$out/transaction.log" 2>&1
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$out/gpu-after.txt"
sha256sum \
  "$build/libklayout_cuda_spatial_backend.so" \
  "$build/m2_union_real_transaction_gate" \
  "$scene" "$oracle" >"$out/pinned-artifacts.sha256"

cat "$out/gpu-before.txt"
cat "$out/time.txt"
cat "$out/transaction.log"
cat "$out/pinned-artifacts.sha256"
cat "$out/gpu-after.txt"
printf 'M2_UNION_REAL_TRANSACTION evidence=%s\n' "$out"
