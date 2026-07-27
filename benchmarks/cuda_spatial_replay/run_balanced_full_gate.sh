#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_balanced_full_gate.sh \
    --klayout PATH --backend PATH --source-deck PATH \
    --manifest PATH --input PATH --top-cell NAME --reference PATH \
    [--python PATH] [--timeout-seconds N] [--jobs 8..14] \
    [--split-lower-antenna] [--split-upper-antenna] \
    [--fuse-metal-antenna] \
    [--with-antenna-m1-m4|--without-antenna-m1-m4] \
    [--split-implant-contact] [--split-active12] \
    [--without-contact4] \
    [--with-active3-well-union|--without-active3-well-union] \
    [--with-active4-well-union|--without-active4-well-union] \
    [--with-contact4-active-union|--without-contact4-active-union] \
    [--with-m2-rules|--without-m2-rules] \
    [--with-active12|--without-active12] \
    [--with-m1-base-width-space|--without-m1-base-width-space] \
    [--with-m1-5-9|--without-m1-5-9] \
    [--with-m2-width-space|--without-m2-width-space] \
    [--with-implant12|--without-implant12] \
    [--with-implant15|--without-implant15] \
    [--with-poly34|--without-poly34] \
    [--prune-poly2] [--keep-work]

Regenerates the qualified FreePDK45 live-CUDA deck, applies the antenna split,
coalesces CONTACT.6 into the grid owner, and runs the exact balanced full-launch
gate. With --split-lower-antenna, METAL1 and METAL2 checks use separate
owners. With --split-upper-antenna, METAL3 and METAL4-through-METAL10 checks
use separate owners. The two split modes compose. The mutually exclusive
--fuse-metal-antenna mode assigns all ten metal checks to one exact staged
owner, providing the literal CPU fallback for the M1-through-M4 CUDA
transaction. The selected plan has nine owners in fused mode, ten by default,
and up to fourteen with all four splits. --split-implant-contact moves the
intact IMPLANT.1-.5 and CONTACT.1-.5 blocks into separate owners.
--split-active12 moves the intact ACTIVE.1/.2 block out of the upper-metal
owner. CUDA resource limits remain fixed; the launcher may use 8 through 14
process slots, never more slots than selected owners.

--with-antenna-m1-m4 and --without-antenna-m1-m4 require
--fuse-metal-antenna.  They retain the identical atomic fused-owner deck and
toggle only the M1-through-M4 CUDA transaction at runtime.  A complete
four-stage empty certificate emits all ten named empty antenna categories;
every decline, exception, missing symbol, or disabled control executes the
complete literal METAL1-through-METAL10 CPU chain.

The manifest must be bound to the generated deck and selected owner set.
Reference may be either a raw or generator-stripped XML .lyrdb report. The
merged report must match it after removing only the generator element. CUDA
certificate telemetry, the canonical report, timings, hashes, and launcher
provenance are checked before success.

--without-contact4 retains every other qualified CUDA transaction and exists
only to produce a same-binary CONTACT.4-off performance control.

--with-active3-well-union and --without-active3-well-union generate the same
exact resident NWELL/PWELL-union ACTIVE.3 transaction deck, then toggle only
its runtime environment.  COMPLETE with zero hits and zero uncertainty is the
sole bypass; every decline executes the literal historical WELL union and
ACTIVE.3 expression.  The default omits this deck rewrite for compatibility
with historical gates.

--with-active4-well-union and --without-active4-well-union generate the same
exact resident ACTIVE-subset-of-WELL transaction deck, then toggle only its
runtime environment.  COMPLETE with all exact rectangles visited/completed,
zero witnesses, zero uncertainty, and zero device flags is the sole bypass;
every decline executes the literal WELL union and active.not(well).  ACTIVE.3
and ACTIVE.4 exact WELL-union rewrites are alternatives because each owns the
same lazy WELL construction site.

--with-contact4-active-union and --without-contact4-active-union retain the
same fused-capable binary, backend, and deck while toggling only the exact
pre-merge raw-ACTIVE union/CONTACT.4 transaction.  Both modes leave the older
merged CONTACT.4 certificate enabled as the candidate's fail-closed fallback
and the control's active implementation.  The default leaves the fused
environment unset for compatibility with historical gates.

--with-m2-rules and --without-m2-rules both generate the identical
fail-closed METAL2.1/.2/.4-.9 speculative-flat transaction, then toggle only
its runtime environment. The default omits that deck rewrite and preserves
the older qualified gate. The enabled M2-rules owner and the independent
M2-width/space certificate cannot run together; use --without-m2-width-space
for an explicit same-deck M2-rules candidate or control.

--with-m1-5-9 and --without-m1-5-9 both generate the identical fail-closed
raw-M1 resident-morphology transaction inside the exact m1_via_class owner,
then toggle only its runtime environment.  The sole bypass is a complete,
all-five-bit empty proof; every decline runs the literal M1.5-.9 CPU block.

--with-m1-base-width-space and --without-m1-base-width-space preserve the
same deck and toggle only the exact pre-merge raw-M1 METAL1.1/.2 transaction.
The sole bypass is a complete two-bit empty proof; every decline runs the
literal historical merge and width/space expressions.  An explicitly selected
M1 runtime mode reserves its owner behind the first launch wave for both the
on and off controls.  When both M1 modes are selected, the launcher also
serializes their owner processes under the same schedule: each candidate can
consume roughly 8 GiB of the device and they must never overlap.  Thus the
fully split 14-owner plan accepts at most 13 jobs with one explicit M1 mode or
12 with both.  An explicitly selected fused antenna M1-M4 transaction starts
in the first wave so its host lowering overlaps the other owners.  Its bounded
device admission waits for a reserved M1 transaction to release sufficient
GPU memory, then retries the resident transaction with the same exact capture.

--with-active12 and --without-active12 both generate the identical fail-closed
raw-ACTIVE resident ACTIVE.1/.2 transaction and toggle only its runtime
environment.  The sole bypass is a complete two-bit empty proof at the fixed
90-nm width and 80-nm spacing thresholds; every decline runs the literal two
historical CPU expressions.  This mode requires --split-active12.

--with-m2-width-space and --without-m2-width-space preserve the same deck and
toggle only the separately qualified METAL2.1/.2 runtime transaction.  The
default leaves its environment unset for compatibility with historical gates.

--with-implant12 and --without-implant12 both generate the identical fused
IMPLANT.1/.2 transaction deck, then toggle only its runtime environment. The
default omits that deck rewrite and preserves the older qualified gate.

--with-implant15 and --without-implant15 both generate the identical atomic
raw-NPLUS/PPLUS IMPLANT.1-.5 transaction deck, then toggle only its runtime
environment.  A complete all-five-bit empty certificate bypasses the
historical implant union and all five CPU rules; every other outcome runs the
literal union and intact CPU block once.  When IMPLANT15 and IMPLANT12 are
both generated, IMPLANT12 remains nested in that fallback and is dormant
after an IMPLANT15 success.  The default omits the new rewrite byte-for-byte.

--with-poly34 and --without-poly34 both generate the identical atomic
POLY.3/.4 transaction deck, then toggle only its runtime environment. The
default omits that deck rewrite and preserves the older qualified gate.

--prune-poly2 applies the independently qualified, fail-closed source
transform which removes FreePDK45's unreachable POLY.2 calculation.  The
historical separation returns an EdgePairs layer, so its polygons? guard can
never publish POLY.2.  The default preserves the source block byte-for-byte.
Source drift, duplicates, an extra POLY.2 output, or an already-pruned input
fail before any DRC child starts.

All generated decks, reports, logs, homes, and caches live under a fresh
TMPDIR directory. They are removed unless --keep-work is specified.
EOF
}

die() {
  echo "balanced full CUDA gate: $*" >&2
  exit 2
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "${here}/../.." && pwd)
deck_generator="${here}/make_via1_stack_live_deck.py"
poly2_prune="${root}/benchmarks/freepdk45_poly2_prune/prune_deck.py"
antenna_split="${root}/benchmarks/freepdk45_antenna_split/split_deck.py"
contact6_split="${root}/benchmarks/freepdk45_contact6_split/split_deck.py"
owner_split="${root}/benchmarks/freepdk45_owner_split/split_deck.py"
runner="${root}/scripts/run_parallel_drc.py"

klayout=
backend=
source_deck=
manifest=
input=
top_cell=
reference=
python=${PYTHON:-python3}
timeout_seconds=900
keep_work=0
contact4=1
active3_well_union=-1
active4_well_union=-1
contact4_active_union=-1
implant12=-1
implant15=-1
m2_rules=-1
active12=-1
m1_base_width_space=-1
m1_5_9=-1
m2_width_space=-1
poly34=-1
prune_poly2=0
jobs=8
split_lower_antenna=0
split_upper_antenna=0
fuse_metal_antenna=0
antenna_m1_m4=-1
split_implant_contact=0
split_active12=0

while (($#)); do
  case "$1" in
    --klayout)
      (($# >= 2)) || die "--klayout requires a value"
      klayout=$2
      shift 2
      ;;
    --backend)
      (($# >= 2)) || die "--backend requires a value"
      backend=$2
      shift 2
      ;;
    --source-deck)
      (($# >= 2)) || die "--source-deck requires a value"
      source_deck=$2
      shift 2
      ;;
    --manifest)
      (($# >= 2)) || die "--manifest requires a value"
      manifest=$2
      shift 2
      ;;
    --input)
      (($# >= 2)) || die "--input requires a value"
      input=$2
      shift 2
      ;;
    --top-cell)
      (($# >= 2)) || die "--top-cell requires a value"
      top_cell=$2
      shift 2
      ;;
    --reference)
      (($# >= 2)) || die "--reference requires a value"
      reference=$2
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python=$2
      shift 2
      ;;
    --timeout-seconds)
      (($# >= 2)) || die "--timeout-seconds requires a value"
      timeout_seconds=$2
      shift 2
      ;;
    --jobs)
      (($# >= 2)) || die "--jobs requires a value"
      jobs=$2
      shift 2
      ;;
    --split-lower-antenna)
      split_lower_antenna=1
      shift
      ;;
    --split-upper-antenna)
      split_upper_antenna=1
      shift
      ;;
    --fuse-metal-antenna)
      fuse_metal_antenna=1
      shift
      ;;
    --with-antenna-m1-m4)
      ((antenna_m1_m4 == -1)) ||
        die "choose exactly one antenna M1-M4 runtime mode"
      antenna_m1_m4=1
      shift
      ;;
    --without-antenna-m1-m4)
      ((antenna_m1_m4 == -1)) ||
        die "choose exactly one antenna M1-M4 runtime mode"
      antenna_m1_m4=0
      shift
      ;;
    --split-implant-contact)
      split_implant_contact=1
      shift
      ;;
    --split-active12)
      split_active12=1
      shift
      ;;
    --keep-work)
      keep_work=1
      shift
      ;;
    --without-contact4)
      contact4=0
      shift
      ;;
    --with-active3-well-union)
      ((active3_well_union == -1)) ||
        die "choose exactly one ACTIVE.3 WELL-union runtime mode"
      active3_well_union=1
      shift
      ;;
    --without-active3-well-union)
      ((active3_well_union == -1)) ||
        die "choose exactly one ACTIVE.3 WELL-union runtime mode"
      active3_well_union=0
      shift
      ;;
    --with-active4-well-union)
      ((active4_well_union == -1)) ||
        die "choose exactly one ACTIVE.4 WELL-union runtime mode"
      active4_well_union=1
      shift
      ;;
    --without-active4-well-union)
      ((active4_well_union == -1)) ||
        die "choose exactly one ACTIVE.4 WELL-union runtime mode"
      active4_well_union=0
      shift
      ;;
    --with-contact4-active-union)
      ((contact4_active_union == -1)) ||
        die "choose exactly one fused CONTACT.4 runtime mode"
      contact4_active_union=1
      shift
      ;;
    --without-contact4-active-union)
      ((contact4_active_union == -1)) ||
        die "choose exactly one fused CONTACT.4 runtime mode"
      contact4_active_union=0
      shift
      ;;
    --with-m2-rules)
      ((m2_rules == -1)) ||
        die "choose exactly one M2-rules runtime mode"
      m2_rules=1
      shift
      ;;
    --without-m2-rules)
      ((m2_rules == -1)) ||
        die "choose exactly one M2-rules runtime mode"
      m2_rules=0
      shift
      ;;
    --with-active12)
      ((active12 == -1)) ||
        die "choose exactly one ACTIVE.1/.2 runtime mode"
      active12=1
      shift
      ;;
    --without-active12)
      ((active12 == -1)) ||
        die "choose exactly one ACTIVE.1/.2 runtime mode"
      active12=0
      shift
      ;;
    --with-m1-base-width-space)
      ((m1_base_width_space == -1)) ||
        die "choose exactly one M1 base-width/space runtime mode"
      m1_base_width_space=1
      shift
      ;;
    --without-m1-base-width-space)
      ((m1_base_width_space == -1)) ||
        die "choose exactly one M1 base-width/space runtime mode"
      m1_base_width_space=0
      shift
      ;;
    --with-m1-5-9)
      ((m1_5_9 == -1)) ||
        die "choose exactly one M1.5-.9 runtime mode"
      m1_5_9=1
      shift
      ;;
    --without-m1-5-9)
      ((m1_5_9 == -1)) ||
        die "choose exactly one M1.5-.9 runtime mode"
      m1_5_9=0
      shift
      ;;
    --with-m2-width-space)
      m2_width_space=1
      shift
      ;;
    --without-m2-width-space)
      m2_width_space=0
      shift
      ;;
    --with-implant12)
      ((implant12 == -1)) ||
        die "choose exactly one IMPLANT.1/.2 runtime mode"
      implant12=1
      shift
      ;;
    --without-implant12)
      ((implant12 == -1)) ||
        die "choose exactly one IMPLANT.1/.2 runtime mode"
      implant12=0
      shift
      ;;
    --with-implant15)
      ((implant15 == -1)) ||
        die "choose exactly one IMPLANT.1-.5 runtime mode"
      implant15=1
      shift
      ;;
    --without-implant15)
      ((implant15 == -1)) ||
        die "choose exactly one IMPLANT.1-.5 runtime mode"
      implant15=0
      shift
      ;;
    --with-poly34)
      ((poly34 == -1)) ||
        die "choose exactly one POLY.3/.4 runtime mode"
      poly34=1
      shift
      ;;
    --without-poly34)
      ((poly34 == -1)) ||
        die "choose exactly one POLY.3/.4 runtime mode"
      poly34=0
      shift
      ;;
    --prune-poly2)
      prune_poly2=1
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
[[ -n "${backend}" ]] || die "missing --backend"
[[ -n "${source_deck}" ]] || die "missing --source-deck"
[[ -n "${manifest}" ]] || die "missing --manifest"
[[ -n "${input}" ]] || die "missing --input"
[[ -n "${top_cell}" ]] || die "missing --top-cell"
[[ -n "${reference}" ]] || die "missing --reference"
[[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] ||
  die "--timeout-seconds must be a positive integer"
[[ "${jobs}" =~ ^[0-9]+$ ]] && ((jobs >= 8 && jobs <= 14)) ||
  die "--jobs must be an integer from 8 through 14"
if ((fuse_metal_antenna &&
      (split_lower_antenna || split_upper_antenna))); then
  die "--fuse-metal-antenna cannot be combined with antenna split modes"
fi
if ((antenna_m1_m4 >= 0 && !fuse_metal_antenna)); then
  die "explicit antenna M1-M4 runtime mode requires --fuse-metal-antenna"
fi
selected_owner_count=$((
  10 + split_lower_antenna + split_upper_antenna +
  split_implant_contact + split_active12 - fuse_metal_antenna
))
((jobs <= selected_owner_count)) ||
  die "--jobs ${jobs} exceeds selected ${selected_owner_count}-owner plan"
if ((m2_rules == 1 && m2_width_space == 1)); then
  die "--with-m2-rules is incompatible with --with-m2-width-space"
fi
if ((contact4 == 0 && contact4_active_union == 1)); then
  die "--with-contact4-active-union requires the fail-closed CONTACT.4 fallback"
fi
if ((active12 >= 0 && ! split_active12)); then
  die "an explicit ACTIVE.1/.2 runtime mode requires --split-active12"
fi
if ((active3_well_union >= 0 && active4_well_union >= 0)); then
  die "ACTIVE.3 and ACTIVE.4 exact WELL-union runtime modes are alternatives"
fi
reserved_m1_owner_count=0
if ((m1_base_width_space >= 0)); then
  ((reserved_m1_owner_count += 1))
fi
if ((m1_5_9 >= 0)); then
  ((reserved_m1_owner_count += 1))
fi
m1_first_wave_reservation=${reserved_m1_owner_count}
if ((antenna_m1_m4 >= 0 && reserved_m1_owner_count == 2)); then
  # The explicit fused-antenna mode enables the cross-process CUDA device
  # lease.  Admit the first serialized M1 owner into the initial wave; the
  # runner still holds the second M1 owner until the first one completes.
  m1_first_wave_reservation=1
fi
if ((m1_first_wave_reservation > 0 &&
      jobs > selected_owner_count - m1_first_wave_reservation)); then
  die "explicit M1 runtime modes reserve ${m1_first_wave_reservation} owner(s) behind the first wave; require --jobs $((selected_owner_count - m1_first_wave_reservation)) or fewer"
fi

[[ -x "${klayout}" ]] || die "KLayout is not executable: ${klayout}"
[[ -f "${backend}" ]] || die "CUDA backend is missing: ${backend}"
[[ -f "${source_deck}" ]] || die "source deck is missing: ${source_deck}"
[[ -f "${manifest}" ]] || die "manifest is missing: ${manifest}"
[[ -s "${input}" ]] || die "input layout is missing or empty: ${input}"
[[ -s "${reference}" ]] ||
  die "reference report is missing or empty: ${reference}"
[[ -f "${deck_generator}" ]] || die "live deck generator is missing"
[[ -f "${poly2_prune}" ]] || die "POLY.2 prune transform is missing"
[[ -f "${antenna_split}" ]] || die "antenna split transform is missing"
[[ -f "${contact6_split}" ]] || die "CONTACT.6 split transform is missing"
[[ -f "${owner_split}" ]] || die "owner split transform is missing"
[[ -f "${runner}" ]] || die "parallel DRC runner is missing"
[[ -x /usr/bin/time ]] || die "/usr/bin/time is unavailable"
[[ -x /usr/bin/timeout ]] || die "/usr/bin/timeout is unavailable"

python=$(command -v -- "${python}") ||
  die "Python interpreter is not executable: ${python}"
klayout=$(readlink -f -- "${klayout}")
backend=$(readlink -f -- "${backend}")
source_deck=$(readlink -f -- "${source_deck}")
manifest=$(readlink -f -- "${manifest}")
input=$(readlink -f -- "${input}")
reference=$(readlink -f -- "${reference}")
klayout_dir=$(dirname -- "${klayout}")
backend_dir=$(dirname -- "${backend}")
runtime_ld_library_path="${backend_dir}:${klayout_dir}"

"${python}" - "${manifest}" "${selected_owner_count}" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
expected_shards = int(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
shards = manifest.get("shards")
categories = manifest.get("categories")
if not isinstance(shards, list) or len(shards) != expected_shards:
    raise SystemExit(
        "balanced full CUDA gate: manifest owner count differs: "
        f"{len(shards) if isinstance(shards, list) else 'invalid'} "
        f"!= {expected_shards}"
    )
if not isinstance(categories, list) or len(categories) != 157:
    raise SystemExit(
        "balanced full CUDA gate: manifest category count differs: "
        f"{len(categories) if isinstance(categories, list) else 'invalid'} "
        "!= 157"
    )
print(
    "BALANCED_FULL_MANIFEST_SHAPE "
    f"shards={len(shards)} categories={len(categories)}"
)
PY

work=$(mktemp -d "${TMPDIR:-/tmp}/klayout-balanced-full-cuda-gate.XXXXXX")
cleanup() {
  if ((keep_work)); then
    echo "BALANCED_FULL_CUDA_GATE work=${work}"
  else
    rm -rf -- "${work}"
  fi
}
trap cleanup EXIT

deck_dir="${work}/decks"
runtime="${work}/runtime"
runtime_tmp="${runtime}/tmp"
mkdir -p -- \
  "${deck_dir}" \
  "${runtime}/home" \
  "${runtime}/klayout-home" \
  "${runtime}/xdg-config" \
  "${runtime}/xdg-cache" \
  "${runtime}/xdg-data" \
  "${runtime_tmp}"

live_deck="${deck_dir}/freepdk45-m1-contact-live.lydrc"
generator_deck="${live_deck}"
if ((prune_poly2)); then
  generator_deck="${deck_dir}/freepdk45-m1-contact-live-unpruned.lydrc"
fi
antenna_deck="${deck_dir}/freepdk45-m1-contact-antenna.lydrc"
coalesced_deck="${deck_dir}/freepdk45-contact6-coalesced.lydrc"
balanced_deck="${deck_dir}/freepdk45-balanced-cuda.lydrc"
transform_log="${work}/deck-transform.log"

run_transform() {
  local label=$1
  shift
  if ! "$@" >>"${transform_log}" 2>&1; then
    cat -- "${transform_log}" >&2
    die "${label} failed"
  fi
}

implant12_generator_args=()
implant12_env=()
if ((implant12 >= 0)); then
  implant12_generator_args=(--implant12)
  implant12_env=(
    "KLAYOUT_CUDA_IMPLANT12=${implant12}"
    "KLAYOUT_CUDA_IMPLANT12_TELEMETRY=${implant12}"
  )
fi
implant15_generator_args=()
implant15_env=()
if ((implant15 >= 0)); then
  implant15_generator_args=(--implant15)
  implant15_env=(
    "KLAYOUT_CUDA_IMPLANT15=${implant15}"
    "KLAYOUT_CUDA_IMPLANT15_TELEMETRY=1"
  )
fi
antenna_m1_m4_env=()
if ((antenna_m1_m4 >= 0)); then
  antenna_m1_m4_env=(
    "KLAYOUT_CUDA_ANTENNA_M1_M4=${antenna_m1_m4}"
    "KLAYOUT_CUDA_ANTENNA_M1_M4_TELEMETRY=1"
    "KLAYOUT_CUDA_ANTENNA_DEVICE_WAIT_MS=15000"
  )
fi
device_lease_env=()
if ((antenna_m1_m4 >= 0 || active4_well_union >= 0 || m2_rules >= 0)); then
  # ACTIVE.4 performs a process-terminal cudaDeviceReset after its exact
  # certificate.  Give it the same cross-process lease as every high-memory
  # M1 owner so reset/reclamation and M1 allocation can never overlap.
  # Enable this for control and candidate modes to preserve one schedule.
  device_lease_env=(
    "KLAYOUT_CUDA_DEVICE_LEASE=1"
    "KLAYOUT_CUDA_DEVICE_LEASE_WAIT_MS=20000"
    "KLAYOUT_CUDA_DEVICE_LEASE_TELEMETRY=1"
  )
fi
m2_rules_generator_args=()
m2_rules_env=()
if ((m2_rules >= 0)); then
  m2_rules_generator_args=(--m2-rules)
  m2_rules_env=(
    "KLAYOUT_CUDA_M2_RULES=${m2_rules}"
    "KLAYOUT_CUDA_M2_RULES_TELEMETRY=${m2_rules}"
  )
fi
m1_5_9_generator_args=()
m1_5_9_env=()
if ((m1_5_9 >= 0)); then
  m1_5_9_generator_args=(--m1-5-9)
  m1_5_9_env=(
    "KLAYOUT_CUDA_M1_5_9=${m1_5_9}"
    "KLAYOUT_CUDA_M1_5_9_TELEMETRY=1"
  )
fi
m1_base_width_space_env=()
if ((m1_base_width_space >= 0)); then
  m1_base_width_space_env=(
    "KLAYOUT_CUDA_M1_BASE_WIDTH_SPACE=${m1_base_width_space}"
    "KLAYOUT_CUDA_M1_BASE_WIDTH_SPACE_TELEMETRY=1"
    "KLAYOUT_DEEP_REGION_MULTI_TELEMETRY=1"
  )
fi
active12_generator_args=()
active12_env=()
if ((active12 >= 0)); then
  active12_generator_args=(--active12)
  active12_env=(
    "KLAYOUT_CUDA_ACTIVE12=${active12}"
    "KLAYOUT_CUDA_ACTIVE12_TELEMETRY=1"
  )
fi
m2_width_space_env=()
if ((m2_width_space >= 0)); then
  m2_width_space_env=(
    "KLAYOUT_CUDA_M2_WIDTH_SPACE=${m2_width_space}"
    "KLAYOUT_CUDA_M2_WIDTH_SPACE_TELEMETRY=${m2_width_space}"
  )
fi
poly34_generator_args=()
poly34_env=()
if ((poly34 >= 0)); then
  poly34_generator_args=(--poly34)
  poly34_env=(
    "KLAYOUT_CUDA_POLY34=${poly34}"
    "KLAYOUT_CUDA_POLY34_TELEMETRY=${poly34}"
  )
fi
contact4_active_union_env=()
if ((contact4_active_union >= 0)); then
  contact4_active_union_env=(
    "KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE=0"
    "KLAYOUT_CUDA_CONTACT4_RAW_ACTIVE_TELEMETRY=0"
    "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION=${contact4_active_union}"
    "KLAYOUT_CUDA_CONTACT4_ACTIVE_UNION_TELEMETRY=1"
  )
fi
active3_well_union_generator_args=()
active3_well_union_env=()
if ((active3_well_union >= 0)); then
  active3_well_union_generator_args=(--active3-well-union)
  active3_well_union_env=(
    "KLAYOUT_CUDA_ACTIVE3_WELL_UNION=${active3_well_union}"
    "KLAYOUT_CUDA_ACTIVE3_WELL_UNION_TELEMETRY=1"
  )
fi
active4_well_union_generator_args=()
active4_well_union_env=()
if ((active4_well_union >= 0)); then
  active4_well_union_generator_args=(--active4-well-union)
  active4_well_union_env=(
    "KLAYOUT_CUDA_ACTIVE4_WELL_UNION=${active4_well_union}"
    "KLAYOUT_CUDA_ACTIVE4_WELL_UNION_TELEMETRY=1"
    "KLAYOUT_CUDA_ACTIVE4_WELL_UNION_TERMINAL_RESET=1"
  )
fi
run_transform "live CUDA deck generation" \
  "${python}" "${deck_generator}" \
    --input "${source_deck}" --output "${generator_deck}" --m1-contact \
    "${active3_well_union_generator_args[@]}" \
    "${active4_well_union_generator_args[@]}" \
    "${active12_generator_args[@]}" \
    "${m1_5_9_generator_args[@]}" \
    "${m2_rules_generator_args[@]}" \
    "${implant12_generator_args[@]}" \
    "${implant15_generator_args[@]}" \
    "${poly34_generator_args[@]}"
if ((prune_poly2)); then
  run_transform "POLY.2 dead-computation prune" \
    "${python}" "${poly2_prune}" "${generator_deck}" "${live_deck}"
fi
antenna_split_args=()
if ((split_lower_antenna)); then
  antenna_split_args+=(--split-lower)
fi
if ((split_upper_antenna)); then
  antenna_split_args+=(--split-upper)
fi
if ((fuse_metal_antenna)); then
  antenna_split_args+=(--fuse-metal)
fi
run_transform "antenna split" \
  "${python}" "${antenna_split}" \
    "${antenna_split_args[@]}" "${live_deck}" "${antenna_deck}"
run_transform "CONTACT.6 grid coalescing" \
  "${python}" "${contact6_split}" \
    --owner grid "${antenna_deck}" "${coalesced_deck}"
owner_split_args=()
if ((split_implant_contact)); then
  owner_split_args+=(--split-implant-contact)
fi
if ((split_active12)); then
  owner_split_args+=(--split-active12)
fi
if ((${#owner_split_args[@]})); then
  run_transform "implant/contact and ACTIVE.1/.2 owner split" \
    "${python}" "${owner_split}" \
      "${owner_split_args[@]}" "${coalesced_deck}" "${balanced_deck}"
else
  cp -- "${coalesced_deck}" "${balanced_deck}"
fi

[[ -s "${balanced_deck}" ]] || die "generated balanced deck is empty"
if ((prune_poly2)); then
  grep -Fq -- \
    "# POLY.2 is intentionally absent.  DRC separation always returns EdgePairs," \
    "${balanced_deck}" ||
    die "POLY.2 prune marker did not survive deck composition"
  if grep -Fq -- \
       "poly.separation(active, 140.nm, projection)" "${balanced_deck}" ||
     grep -Fq -- '.output("POLY.2"' "${balanced_deck}"; then
    die "POLY.2 dead computation survived deck composition"
  fi
fi

reference_canonical="${work}/reference.canonical.lyrdb"
sed '/<generator>/d' "${reference}" >"${reference_canonical}"
[[ -s "${reference_canonical}" ]] ||
  die "canonical reference report is empty"
grep -Fq -- "<report-database>" "${reference_canonical}" ||
  die "reference is not an XML KLayout report database"

owner_prefix=(m1_width_space)
owner_implant_joined=(implant_contact)
owner_implant_split=(implant_contact contact)
owner_upper_joined=(antenna_m3_m10)
owner_upper_split=(antenna_m4_m10 antenna_m3)
owner_lower_joined=(antenna_m1_m2)
owner_lower_split=(antenna_m2 antenna_m1)
owner_metal_fused=(antenna_m1_m4)
owner_suffix_pre=(
  m2_rules
  m1_enclosure
)
owner_active_split=(active12)
owner_suffix_post=(
  via1_upper_active12
  grid
)
if ((m1_base_width_space >= 0 && m1_5_9 >= 0)); then
  # Launch every non-M1 owner first.  The first freed slot starts base
  # width/space; the serialized set then holds M1.5-.9 until base completes.
  # Candidate and control modes retain this identical resource schedule.
  owner_prefix=()
  owner_suffix_post+=(antenna_feol m1_width_space m1_via_class)
elif ((m1_base_width_space >= 0)); then
  # Queue an explicitly selected base-width/space owner last for both its
  # candidate and control, after all owners that do not need its reservation.
  owner_prefix=()
  owner_suffix_post+=(m1_via_class antenna_feol m1_width_space)
elif ((m1_5_9 >= 0)); then
  # Queue an explicitly selected M1.5-.9 owner last for both candidate and
  # control, preserving one schedule when only its runtime mode changes.
  owner_suffix_post+=(antenna_feol m1_via_class)
else
  owner_suffix_post+=(m1_via_class antenna_feol)
fi
serialized_shard_args=()
if ((m1_base_width_space >= 0 && m1_5_9 >= 0)); then
  serialized_shard_args=(
    --serialized-shard m1_width_space
    --serialized-shard m1_via_class
  )
fi
shards=("${owner_prefix[@]}")
if ((split_implant_contact)); then
  shards+=("${owner_implant_split[@]}")
else
  shards+=("${owner_implant_joined[@]}")
fi
if ((fuse_metal_antenna)); then
  shards+=("${owner_metal_fused[@]}")
else
  if ((split_upper_antenna)); then
    shards+=("${owner_upper_split[@]}")
  else
    shards+=("${owner_upper_joined[@]}")
  fi
  if ((split_lower_antenna)); then
    shards+=("${owner_lower_split[@]}")
  else
    shards+=("${owner_lower_joined[@]}")
  fi
fi
shards+=("${owner_suffix_pre[@]}")
if ((split_active12)); then
  shards+=("${owner_active_split[@]}")
fi
shards+=("${owner_suffix_post[@]}")
((${#shards[@]} == selected_owner_count)) ||
  die "internal owner-plan count mismatch"
shard_args=()
for shard in "${shards[@]}"; do
  shard_args+=(--shard "${shard}")
done

stamp=$(date -u +%Y%m%dT%H%M%SZ)
cohort="freepdk45-balanced-cuda-${stamp}"
launcher_log="${work}/launcher.log"
time_file="${work}/time.txt"
report="${work}/merged.lyrdb"
metadata="${work}/metadata.json"

set +e
/usr/bin/time \
  -f 'wall_s=%e user_s=%U system_s=%S max_rss_kib=%M exit=%x' \
  -o "${time_file}" \
  /usr/bin/timeout \
    --foreground --signal=TERM --kill-after=15s "${timeout_seconds}s" \
    env -i \
      "HOME=${runtime}/home" \
      "KLAYOUT_HOME=${runtime}/klayout-home" \
      "XDG_CONFIG_HOME=${runtime}/xdg-config" \
      "XDG_CACHE_HOME=${runtime}/xdg-cache" \
      "XDG_DATA_HOME=${runtime}/xdg-data" \
      "TMPDIR=${runtime_tmp}" \
      PATH=/usr/bin:/bin \
      LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC \
      PYTHONDONTWRITEBYTECODE=1 \
      QT_QPA_PLATFORM=offscreen \
      CUDA_VISIBLE_DEVICES=0 \
      KLAYOUT_CUDA_SPATIAL_DEVICE=0 \
      KLAYOUT_CUDA_SPATIAL_TELEMETRY=1 \
      "${device_lease_env[@]}" \
      "KLAYOUT_CUDA_SPATIAL_BACKEND=${backend}" \
      "LD_LIBRARY_PATH=${runtime_ld_library_path}" \
      KLAYOUT_CUDA_ACTIVE3=1 \
      KLAYOUT_CUDA_ACTIVE3_TELEMETRY=1 \
      "${active3_well_union_env[@]}" \
      "${active4_well_union_env[@]}" \
      "${active12_env[@]}" \
      "${m1_base_width_space_env[@]}" \
      KLAYOUT_CUDA_M1_WIDTH_SPACE=1 \
      KLAYOUT_CUDA_M1_WIDTH_SPACE_TELEMETRY=1 \
      "${m1_5_9_env[@]}" \
      "${m2_rules_env[@]}" \
      "${m2_width_space_env[@]}" \
      KLAYOUT_CUDA_VIA1_STACK=1 \
      KLAYOUT_CUDA_VIA1_STACK_TELEMETRY=1 \
      KLAYOUT_CUDA_DISCONNECTED_MERGE=1 \
      KLAYOUT_DEEP_EDGE_CERT_PROFILE=1 \
      KLAYOUT_CUDA_SPATIAL_MIN_RECORDS=100000 \
      KLAYOUT_CUDA_SPATIAL_CELL_SIZE=512 \
      KLAYOUT_CUDA_SPATIAL_MAX_CELLS_PER_RECORD=64 \
      KLAYOUT_CUDA_SPATIAL_MAX_RECORDS_PER_CELL=4096 \
      KLAYOUT_CUDA_SPATIAL_MAX_MEMBERSHIPS=80000000 \
      KLAYOUT_CUDA_SPATIAL_MAX_PAIR_WORK=100000000 \
      KLAYOUT_CUDA_SPATIAL_MAX_CANDIDATES=30000000 \
      KLAYOUT_CUDA_M1_CONTACT=1 \
      KLAYOUT_CUDA_M1_CONTACT_TELEMETRY=1 \
      "KLAYOUT_CUDA_CONTACT4=${contact4}" \
      "KLAYOUT_CUDA_CONTACT4_TELEMETRY=${contact4}" \
      "${contact4_active_union_env[@]}" \
      "${implant12_env[@]}" \
      "${implant15_env[@]}" \
      "${antenna_m1_m4_env[@]}" \
      "${poly34_env[@]}" \
      "${python}" "${runner}" \
        --klayout "${klayout}" \
        --deck "${balanced_deck}" \
        --input "${input}" \
        --top-cell "${top_cell}" \
        --output "${report}" \
        --metadata "${metadata}" \
        --manifest "${manifest}" \
        "${shard_args[@]}" \
        --jobs "${jobs}" \
        "${serialized_shard_args[@]}" \
        --cohort-id "${cohort}" \
        --replicate-index 1 \
        --replicate-count 1 \
        --keep-temp \
        >"${launcher_log}" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" >"${work}/exit-status.txt"

if ((rc != 0)); then
  echo "BALANCED_FULL_CUDA_GATE failed rc=${rc} work=${work}" >&2
  cat -- "${time_file}" >&2 || true
  tail -n 200 -- "${launcher_log}" >&2 || true
  exit "${rc}"
fi

[[ -s "${report}" ]] || die "parallel launcher produced no merged report"
[[ -s "${metadata}" ]] || die "parallel launcher produced no provenance"

shopt -s nullglob
shard_dirs=("${runtime_tmp}"/klayout-parallel-drc-*)
shopt -u nullglob
((${#shard_dirs[@]} == 1)) ||
  die "expected one retained shard directory, found ${#shard_dirs[@]}"
shard_dir=${shard_dirs[0]}
[[ -d "${shard_dir}" ]] || die "retained shard artifact is not a directory"

require_telemetry() {
  local marker=$1
  local label=$2
  grep -R -Fq --include='*.log' -- "${marker}" "${shard_dir}" ||
    die "missing ${label} telemetry"
}

if ((active3_well_union == 1)); then
  require_telemetry \
    "CUDA ACTIVE.3 exact resident WELL-union certificate: outcome=certified-empty" \
    "exact resident WELL-union ACTIVE.3 certified-empty"
  require_telemetry \
    "CUDA ACTIVE.3 exact WELL-union live lowering:" \
    "exact resident WELL-union ACTIVE.3 live lowering"
  if grep -R -Fq --include='*.log' -- \
       "CUDA ACTIVE.3 empty certificate:" "${shard_dir}"; then
    die "exact ACTIVE.3 WELL-union candidate unexpectedly invoked fallback"
  fi
else
  require_telemetry \
    "CUDA ACTIVE.3 empty certificate: outcome=certified-empty" \
    "ACTIVE.3 certified-empty"
  if ((active3_well_union == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA ACTIVE.3 exact" "${shard_dir}"; then
    die "ACTIVE.3 WELL-union-off control unexpectedly invoked the exact path"
  fi
fi
if ((active4_well_union == 1)); then
  require_telemetry \
    "CUDA ACTIVE.4 exact resident WELL-union subset certificate: outcome=certified-empty" \
    "exact resident WELL-union ACTIVE.4 certified-empty"
  if ! grep -R -Eq --include='*.log' -- \
       'CUDA ACTIVE\.4 exact resident WELL-union subset certificate: outcome=certified-empty.*fallback_flags=0 device_flags=0' \
       "${shard_dir}"; then
    die "exact ACTIVE.4 WELL-union certificate did not close every status flag"
  fi
  require_telemetry \
    "CUDA ACTIVE.4 exact WELL-union live lowering:" \
    "exact resident WELL-union ACTIVE.4 live lowering"
  require_telemetry \
    "CUDA ACTIVE.4 exact WELL-union transaction: certified-empty" \
    "ACTIVE.4 deck transaction certified-empty"
  require_telemetry \
    "KLAYOUT_CUDA_ACTIVE4_DEVICE_RECLAIM " \
    "ACTIVE.4 terminal device reclamation"
  require_telemetry \
    "KLAYOUT_CUDA_DEVICE_LEASE role=active4_well_union " \
    "ACTIVE.4 cross-process device lease"
  if ! grep -R -Eq --include='*.log' -- \
       'KLAYOUT_CUDA_DEVICE_LEASE role=active4_well_union device=[0-9]+ wait_ms=[0-9.]+ disposition=acquired' \
       "${shard_dir}" ||
     ! grep -R -Eq --include='*.log' -- \
       'KLAYOUT_CUDA_DEVICE_LEASE role=active4_well_union hold_ms=[0-9.]+ disposition=released' \
       "${shard_dir}"; then
    die "ACTIVE.4 cross-process device lease was not acquired and released"
  fi
  if ! grep -R -Eq --include='*.log' -- \
       'KLAYOUT_CUDA_ACTIVE4_DEVICE_RECLAIM free_before_bytes=[0-9]+ free_after_bytes=[0-9]+ released_bytes=[0-9]+ disposition=terminal-reset' \
       "${shard_dir}"; then
    die "ACTIVE.4 terminal device reclamation did not complete"
  fi
  if grep -R -Eq --include='*.log' -- \
       'CUDA ACTIVE\.4 exact resident WELL-union subset certificate: outcome=(fallback|error|uncertain|raw-hits-cpu-fallback)' \
       "${shard_dir}"; then
    die "exact ACTIVE.4 WELL-union candidate unexpectedly invoked fallback"
  fi
elif ((active4_well_union == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA ACTIVE.4 exact" "${shard_dir}"; then
  die "ACTIVE.4 WELL-union-off control unexpectedly invoked the exact path"
fi
if ((m1_base_width_space == 1)); then
  require_telemetry \
    "CUDA M1.1/M1.2 raw-union exact live lowering: outcome=certified-empty" \
    "raw-M1 base-width/space certified-empty"
  require_telemetry \
    "KLAYOUT_DEEP_REGION_MULTI outcome=raw-cuda-certified-empty" \
    "raw-M1 base-width/space pre-merge bypass"
  if ! grep -R -Eq --include='*.log' -- \
       'KLAYOUT_DEEP_REGION_MULTI outcome=raw-cuda-certified-empty .*merged_deep_layer_ms=0([.]0+)?([[:space:]]|$)' \
       "${shard_dir}"; then
    die "raw-M1 base-width/space certificate did not bypass merged_deep_layer"
  fi
  if grep -R -Fq --include='*.log' -- \
       "CUDA M1 width/space empty certificate:" "${shard_dir}"; then
    die "raw-M1 base-width/space candidate unexpectedly invoked legacy fallback"
  fi
else
  require_telemetry \
    "CUDA M1 width/space empty certificate: outcome=certified-empty" \
    "M1 width/space certified-empty"
  if ((m1_base_width_space == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA M1.1/M1.2 raw-union" "${shard_dir}"; then
    die "raw-M1 base-width/space-off control unexpectedly invoked exact CUDA"
  fi
fi
if ((active12 == 1)); then
  require_telemetry \
    "CUDA ACTIVE.1/ACTIVE.2 raw-union exact live lowering: outcome=certified-empty" \
    "raw-ACTIVE ACTIVE.1/.2 certified-empty"
  require_telemetry \
    "CUDA ACTIVE.1/ACTIVE.2 transaction: certified-empty" \
    "ACTIVE.1/.2 deck transaction certified-empty"
  if grep -R -Eq --include='*.log' -- \
       'CUDA ACTIVE[.]1/ACTIVE[.]2 .*outcome=(cpu-fallback|fallback|error|uncertain|not-empty)' \
       "${shard_dir}"; then
    die "ACTIVE.1/.2 candidate unexpectedly invoked fallback"
  fi
elif ((active12 == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA ACTIVE.1/ACTIVE.2 raw-union" "${shard_dir}"; then
  die "ACTIVE.1/.2-off control unexpectedly invoked exact CUDA"
fi
if ((m1_5_9 == 1)); then
  require_telemetry \
    "CUDA M1 exact resident morphology certificate: outcome=certified-empty" \
    "M1.5-.9 exact resident morphology certified-empty"
  require_telemetry \
    "CUDA M1.5-.9 exact live lowering: outcome=certified-empty" \
    "M1.5-.9 exact live lowering"
  require_telemetry \
    "CUDA M1.5-.9 transaction: certified-empty" \
    "M1.5-.9 empty-output transaction"
elif ((m1_5_9 == 0)) &&
     grep -R -Eq --include='*.log' -- \
       'CUDA M1(\.5-\.9 exact| exact resident morphology)' \
       "${shard_dir}"; then
  die "M1.5-.9-off control unexpectedly invoked the exact CUDA path"
fi
if ((m2_width_space == 1)); then
  require_telemetry \
    "CUDA M2 width/space empty certificate: outcome=certified-empty" \
    "M2 width/space certified-empty"
elif ((m2_width_space == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA M2 width/space" "${shard_dir}"; then
  die "M2 width/space-off control unexpectedly invoked M2 CUDA"
fi
if ((m2_rules == 1)); then
  require_telemetry \
    "CUDA M2 exact union boundary: outcome=complete" \
    "M2 exact-union complete"
  require_telemetry \
    "CUDA M2 live flat operands: disposition=complete" \
    "M2 live-flat operand publication"
  require_telemetry \
    "CUDA M2 rules transaction: certified-empty reason=prefix-clean+suffix-certified" \
    "M2 rules certified-empty"
  require_telemetry \
    "KLAYOUT_CUDA_DEVICE_LEASE role=m2_union " \
    "M2 cross-process device lease"
  if ! grep -R -Eq --include='*.log' -- \
       'KLAYOUT_CUDA_DEVICE_LEASE role=m2_union device=[0-9]+ wait_ms=[0-9.]+ disposition=acquired' \
       "${shard_dir}" ||
     ! grep -R -Eq --include='*.log' -- \
       'KLAYOUT_CUDA_DEVICE_LEASE role=m2_union hold_ms=[0-9.]+ disposition=released' \
       "${shard_dir}"; then
    die "M2 cross-process device lease was not acquired and released"
  fi
elif ((m2_rules == 0)) &&
     grep -R -Eq --include='*.log' -- \
       'CUDA M2 (exact union boundary:|live flat operands:|rules transaction:)' \
       "${shard_dir}"; then
  die "M2-rules-off control unexpectedly invoked the live M2 transaction"
fi
require_telemetry \
  "CUDA VIA1 stack empty certificate: outcome=certified-empty" \
  "VIA1-stack certified-empty"
require_telemetry \
  "CUDA M1 contact transaction: certified-empty" \
  "M1-contact certified-empty"
if grep -R -Eq --include='*.log' -- \
     'CUDA VIA1 stack empty certificate: outcome=(error|fallback|uncertain)' \
     "${shard_dir}"; then
  die "a VIA1-stack transaction declined or failed during the full gate"
fi
if grep -R -Fq --include='*.log' -- \
     "CUDA M1 contact transaction: full-cpu-fallback" "${shard_dir}"; then
  die "an M1-contact transaction unexpectedly selected CPU fallback"
fi
if ((contact4_active_union == 1)); then
  require_telemetry \
    "CUDA CONTACT.4 fused ACTIVE-union empty certificate: outcome=certified-empty" \
    "fused ACTIVE-union CONTACT.4 certified-empty"
  require_telemetry \
    "CUDA CONTACT.4 fused ACTIVE-union live lowering:" \
    "fused ACTIVE-union CONTACT.4 live lowering"
  if grep -R -Fq --include='*.log' -- \
       "CUDA CONTACT.4 empty certificate:" "${shard_dir}" ||
     grep -R -Fq --include='*.log' -- \
       "CUDA CONTACT.4 live lowering:" "${shard_dir}" ||
     grep -R -Fq --include='*.log' -- \
       "CUDA CONTACT.4 raw-ACTIVE" "${shard_dir}"; then
    die "fused CONTACT.4 candidate unexpectedly invoked a fallback certificate"
  fi
elif ((contact4)); then
  require_telemetry \
    "CUDA CONTACT.4 empty certificate: outcome=certified-empty" \
    "CONTACT.4 certified-empty"
  if ((contact4_active_union == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA CONTACT.4 fused ACTIVE-union" "${shard_dir}"; then
    die "fused CONTACT.4-off control unexpectedly invoked the fused path"
  fi
elif grep -R -Fq --include='*.log' -- \
  "CUDA CONTACT.4" "${shard_dir}"; then
  die "CONTACT.4-off control unexpectedly invoked CONTACT.4 CUDA"
fi
if ((implant15 == 1)); then
  require_telemetry \
    "CUDA IMPLANT.1-.5 raw transaction: certified-empty" \
    "raw IMPLANT.1-.5 certified-empty"
  if grep -R -Fq --include='*.log' -- \
       "CUDA IMPLANT.1/.2 transaction:" "${shard_dir}"; then
    die "raw IMPLANT.1-.5 success unexpectedly entered IMPLANT.1/.2 fallback"
  fi
elif ((implant15 == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA IMPLANT.1-.5 raw" "${shard_dir}"; then
  die "IMPLANT.1-.5-off control unexpectedly invoked raw IMPLANT CUDA"
fi
if ((implant12 == 1 && implant15 != 1)); then
  require_telemetry \
    "CUDA IMPLANT.1/.2 transaction: certified-empty" \
    "IMPLANT.1/.2 certified-empty"
elif ((implant12 == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA IMPLANT.1/.2 transaction:" "${shard_dir}"; then
  die "IMPLANT.1/.2-off control unexpectedly invoked IMPLANT CUDA"
fi
if ((antenna_m1_m4 == 1)); then
  require_telemetry \
    "CUDA ANTENNA M1-M4 raw transaction: certified-empty" \
    "raw antenna M1-M4 certified-empty"
elif ((antenna_m1_m4 == 0)); then
  require_telemetry \
    "CUDA ANTENNA M1-M4 raw transaction: full-cpu-fallback" \
    "raw antenna M1-M4-off full CPU fallback"
fi
if ((poly34 == 1)); then
  require_telemetry \
    "CUDA POLY.3/.4 transaction: certified-empty" \
    "POLY.3/.4 certified-empty"
elif ((poly34 == 0)) &&
     grep -R -Fq --include='*.log' -- \
       "CUDA POLY.3/.4" "${shard_dir}"; then
  die "POLY.3/.4-off control unexpectedly invoked POLY CUDA"
fi

shard_count=$(grep -c '^shard ' "${launcher_log}" || true)
[[ "${shard_count}" == "${#shards[@]}" ]] ||
  die "expected ${#shards[@]} shard timing lines, found ${shard_count}"

report_canonical="${work}/merged.canonical.lyrdb"
sed '/<generator>/d' "${report}" >"${report_canonical}"
if ! cmp -s -- "${reference_canonical}" "${report_canonical}"; then
  sha256sum -- "${reference_canonical}" "${report_canonical}" >&2
  die "merged report differs from the canonical reference"
fi

grep -R -nE --include='*.log' -- \
  'CUDA ACTIVE\.3 exact resident WELL-union certificate:|CUDA ACTIVE\.3 exact WELL-union live lowering:|CUDA ACTIVE\.3 empty certificate:|CUDA ACTIVE\.3 live lowering:|CUDA ACTIVE\.4 exact resident WELL-union subset certificate:|CUDA ACTIVE\.4 exact WELL-union live lowering:|CUDA ACTIVE\.4 exact WELL-union transaction:|CUDA M1\.1/M1\.2 raw-union exact live lowering:|CUDA M1 width/space empty certificate:|CUDA M1 width/space live lowering:|CUDA M1 exact resident morphology certificate:|CUDA M1\.5-\.9 exact live lowering:|CUDA M1\.5-\.9 transaction:|CUDA M2 width/space empty certificate:|CUDA M2 width/space live lowering:|CUDA M2 exact union boundary:|CUDA M2 live flat operands:|CUDA M2 rules transaction:|CUDA M1 contact transaction:|CUDA M1 contact live lowering:|CUDA CONTACT\.4 fused ACTIVE-union empty certificate:|CUDA CONTACT\.4 fused ACTIVE-union live lowering:|CUDA CONTACT\.4 empty certificate:|CUDA CONTACT\.4 live lowering:|CUDA IMPLANT\.1-\.5 raw transaction:|CUDA IMPLANT\.1/\.2 transaction:|CUDA IMPLANT\.1/\.2 empty certificate:|CUDA IMPLANT\.1/\.2 live lowering:|CUDA ANTENNA M1-M4 raw transaction:|CUDA antenna M1-M4 transaction:|CUDA POLY\.3/\.4 transaction:|CUDA POLY\.3/\.4 live lowering:|CUDA VIA1 stack transaction:|CUDA VIA1 stack empty certificate:|CUDA VIA1 stack live lowering:|KLAYOUT_DEEP_REGION_MULTI |KLAYOUT_DEEP_EDGE_CERT ' \
  "${shard_dir}" >"${work}/cuda-telemetry.txt"

grep -E \
  '^shard |^children:|^merge:|^child\+merge workload:|^provenance verification:|^full launcher wall ' \
  "${launcher_log}" >"${work}/launcher-summary.txt" ||
  die "launcher timing summary is missing"

poly2_pins=()
if ((prune_poly2)); then
  poly2_pins=("${poly2_prune}" "${generator_deck}")
fi
sha256sum -- \
  "${klayout}" \
  "${backend}" \
  "${source_deck}" \
  "${poly2_pins[@]}" \
  "${live_deck}" \
  "${antenna_deck}" \
  "${coalesced_deck}" \
  "${balanced_deck}" \
  "${manifest}" \
  "${input}" \
  "${reference}" \
  "${reference_canonical}" \
  "${report}" \
  "${report_canonical}" \
  "${deck_generator}" \
  "${antenna_split}" \
  "${contact6_split}" \
  "${owner_split}" \
  "${runner}" \
  >"${work}/pinned-artifacts.sha256"
sha256sum -- \
  "${reference_canonical}" "${report_canonical}" \
  >"${work}/canonical-report-sha256.txt"

cat -- "${time_file}"
cat -- "${work}/launcher-summary.txt"
cat -- "${work}/canonical-report-sha256.txt"
cat -- "${work}/cuda-telemetry.txt"
echo \
  "BALANCED_FULL_CUDA_GATE ok owners=${#shards[@]} jobs=${jobs} contact4=${contact4} active3_well_union=${active3_well_union} active4_well_union=${active4_well_union} contact4_active_union=${contact4_active_union} active12=${active12} m1_base_width_space=${m1_base_width_space} m1_5_9=${m1_5_9} m2_rules=${m2_rules} m2_width_space=${m2_width_space} implant12=${implant12} implant15=${implant15} antenna_m1_m4=${antenna_m1_m4} poly34=${poly34} prune_poly2=${prune_poly2} split_lower_antenna=${split_lower_antenna} split_upper_antenna=${split_upper_antenna} fuse_metal_antenna=${fuse_metal_antenna} split_implant_contact=${split_implant_contact} split_active12=${split_active12}"
