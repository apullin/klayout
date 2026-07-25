#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build="${1:-$root/build}"
scene="${KLAYOUT_M2_UNION_BACKEND_KACT:-/tmp/m2-via1-x2.39i7kG/m2-via1-x2.kact}"
oracle="${KLAYOUT_M2_UNION_BACKEND_ORACLE:-/tmp/m2-width-space-census-exact.km1ws}"
repeat="${KLAYOUT_M2_UNION_BACKEND_REPEAT:-3}"
scratch="${KLAYOUT_M2_UNION_BACKEND_EVIDENCE_ROOT:-/home/pullin/personal/klayout/.scratchpad/cuda-runs}"
out=$(mktemp -d "$scratch/m2-union-production-backend.XXXXXX")
mkdir -p "$out/tmp"

test -f "$scene"
test -f "$oracle"
cmake --build "$build" --parallel 32 --target \
  klayout_cuda_spatial_backend \
  m2_union_production_backend_smoke \
  m2_union_production_backend_gate

nvidia-smi \
  --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$out/gpu-before.txt"
"$build/m2_union_production_backend_smoke" \
  >"$out/small-scene.log" 2>&1
/usr/bin/time \
  -f 'wall_s=%e user_s=%U system_s=%S max_rss_kib=%M exit=%x' \
  -o "$out/time.txt" \
  env TMPDIR="$out/tmp" \
    "$build/m2_union_production_backend_gate" \
      "$scene" "$oracle" "$repeat" \
      >"$out/production.log" 2>&1
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader >"$out/gpu-after.txt"
sha256sum \
  "$build/libklayout_cuda_spatial_backend.so" \
  "$build/m2_union_production_backend_smoke" \
  "$build/m2_union_production_backend_gate" \
  "$scene" "$oracle" >"$out/pinned-artifacts.sha256"

cat "$out/gpu-before.txt"
cat "$out/small-scene.log"
cat "$out/time.txt"
cat "$out/production.log"
cat "$out/pinned-artifacts.sha256"
cat "$out/gpu-after.txt"
printf 'M2_UNION_PRODUCTION_BACKEND evidence=%s\n' "$out"
