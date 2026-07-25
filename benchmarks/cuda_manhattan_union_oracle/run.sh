#!/usr/bin/env bash

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${KLAYOUT_MANHATTAN_ORACLE_BUILD_DIR:-"$here/.build"}"
compiler="${CXX:-c++}"

mkdir -p "$build_dir/tmp"

env TMPDIR="$build_dir/tmp" "$compiler" \
  -std=c++17 \
  -O2 \
  -g \
  -Wall \
  -Wextra \
  -Werror \
  -pedantic \
  "$here/manhattan_union_oracle.cc" \
  "$here/manhattan_union_oracle_test.cc" \
  -o "$build_dir/manhattan_union_oracle_test"

"$build_dir/manhattan_union_oracle_test"
