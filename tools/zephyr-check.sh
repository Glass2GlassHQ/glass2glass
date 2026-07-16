#!/usr/bin/env bash
# Zephyr executor proof (M637) via the g2g Zephyr module (M647): build the
# g2g-noalloc staticlib (the exact pipeline the no-heap / panic-free symbol
# proofs cover) for the qemu_cortex_m3 board's ISA (thumbv7m, soft float),
# build a Zephyr application that consumes the g2g module (no west workspace:
# the pinned zephyr tree + the CMSIS module revision its manifest pins are
# cloned directly; the g2g module is handed to Zephyr via ZEPHYR_EXTRA_MODULES,
# what `west update` would arrange), and boot it on QEMU's lm3s6965evb
# Cortex-M3. The app declares nothing about the library: the module provides
# <g2g.h> and links the archive. Zephyr's main thread calls g2g_noalloc_run()
# and g2g_audio_run() and verifies the wire checksums on-target, completing the
# executor matrix (bare M628 / Embassy M632 / FreeRTOS M633 / Zephyr M637) and
# proving the consumable module packaging.
#
# Usage: tools/zephyr-check.sh
# Requires: rustup target thumbv7m-none-eabi; arm-none-eabi-gcc (its prefix
# dir overridable with $GNUARMEMB_TOOLCHAIN_PATH); cmake + ninja + dtc;
# python3 with Zephyr's base build deps (pyelftools, PyYAML, packaging,
# pykwalify); qemu-system-arm (override with $QEMU_SYSTEM_ARM); network on
# first run (clones the pinned zephyr + cmsis trees).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/examples/g2g-zephyr"
# qemu_cortex_m3 (TI LM3S6965, ARMv7-M, no FPU): soft-float thumbv7m, unlike
# the other executor proofs' MPS2-AN386 Cortex-M4F. The staticlib is pure
# no_std Rust, so the ISA is just a --target flag.
TARGET="thumbv7m-none-eabi"
LIB="$ROOT/examples/g2g-noalloc/target/$TARGET/release/libg2g_noalloc.a"
OUT="$SRC/target"
BUILD="$OUT/build"
QEMU="${QEMU_SYSTEM_ARM:-qemu-system-arm}"

# Zephyr, pinned. Cached under the example's (gitignored) target dir so
# repeat runs are offline, like the FreeRTOS kernel checkout.
ZEPHYR_TAG="v4.2.0"
ZDIR="$OUT/zephyr"

echo "== building g2g-noalloc staticlib for $TARGET =="
rustup target add "$TARGET" >/dev/null 2>&1 || true
cargo build --manifest-path "$ROOT/examples/g2g-noalloc/Cargo.toml" --release --target "$TARGET"

# dtc is deliberately not required: Zephyr's own python devicetree tooling
# generates the headers, the dtc binary only re-validates when present.
for tool in cmake ninja python3; do
  command -v "$tool" >/dev/null 2>&1 \
    || { echo "FAIL: $tool not found (Zephyr build prerequisite)"; exit 1; }
done
python3 -c "import elftools, yaml, packaging" 2>/dev/null \
  || { echo "FAIL: python deps missing (pip install pyelftools PyYAML packaging pykwalify)"; exit 1; }

# GNU Arm Embedded toolchain: Zephyr wants the *prefix* directory (the one
# containing bin/arm-none-eabi-gcc).
if [ -z "${GNUARMEMB_TOOLCHAIN_PATH:-}" ]; then
  GCC_BIN="$(command -v arm-none-eabi-gcc || true)"
  [ -n "$GCC_BIN" ] \
    || { echo "FAIL: arm-none-eabi-gcc not found (install it or set \$GNUARMEMB_TOOLCHAIN_PATH)"; exit 1; }
  GNUARMEMB_TOOLCHAIN_PATH="$(dirname "$(dirname "$GCC_BIN")")"
fi

mkdir -p "$OUT"
if [ ! -d "$ZDIR" ]; then
  echo "== fetching zephyr $ZEPHYR_TAG =="
  git clone --depth 1 --branch "$ZEPHYR_TAG" -q \
    https://github.com/zephyrproject-rtos/zephyr.git "$ZDIR"
fi

# The Cortex-M arch code needs the CMSIS headers, which live in a module
# repo. Fetch whichever CMSIS module(s) this Zephyr's manifest pins (older
# trees pin `cmsis`, newer ones `cmsis_6`), at the pinned revision, and hand
# them to CMake as ZEPHYR_MODULES (what `west update` would have arranged).
MODULES=""
while read -r NAME REV; do
  MDIR="$OUT/modules/$NAME"
  if [ ! -d "$MDIR" ]; then
    echo "== fetching module $NAME @ $REV =="
    mkdir -p "$MDIR"
    git -C "$MDIR" init -q
    git -C "$MDIR" remote add origin "https://github.com/zephyrproject-rtos/$NAME.git"
    git -C "$MDIR" fetch -q --depth 1 origin "$REV"
    git -C "$MDIR" checkout -q FETCH_HEAD
  fi
  MODULES="${MODULES:+$MODULES;}$MDIR"
done < <(python3 - "$ZDIR/west.yml" <<'PY'
import sys, yaml
manifest = yaml.safe_load(open(sys.argv[1]))["manifest"]
for p in manifest["projects"]:
    if p["name"] in ("cmsis", "cmsis_6"):
        print(p["name"], p["revision"])
PY
)
[ -n "$MODULES" ] || { echo "FAIL: no cmsis module found in zephyr's west.yml"; exit 1; }

# The g2g Zephyr module (M647): a Zephyr shop would list it in west.yml; here
# we hand it to Zephyr via ZEPHYR_EXTRA_MODULES (what `west update` arranges),
# alongside the cmsis module. The app links nothing g2g itself; the module
# imports the prebuilt staticlib (path in G2G_STATICLIB) and wires it in.
G2G_MODULE="$ROOT/examples/g2g-zephyr-module"

echo "== configuring the Zephyr application via the g2g module (board qemu_cortex_m3) =="
export ZEPHYR_BASE="$ZDIR"
cmake -S "$SRC" -B "$BUILD" -GNinja \
  -DBOARD=qemu_cortex_m3 \
  -DZEPHYR_MODULES="$MODULES" \
  -DZEPHYR_EXTRA_MODULES="$G2G_MODULE" \
  -DZEPHYR_TOOLCHAIN_VARIANT=gnuarmemb \
  -DGNUARMEMB_TOOLCHAIN_PATH="$GNUARMEMB_TOOLCHAIN_PATH" \
  -DG2G_STATICLIB="$LIB"

echo "== building =="
ninja -C "$BUILD"

ELF="$BUILD/zephyr/zephyr.elf"
[ -f "$ELF" ] || { echo "FAIL: $ELF not produced"; exit 1; }

command -v "${QEMU%% *}" >/dev/null 2>&1 \
  || { echo "FAIL: $QEMU not found (install qemu-system-arm or set \$QEMU_SYSTEM_ARM)"; exit 1; }

echo "== running on QEMU lm3s6965evb (Cortex-M3) =="
OUT_TEXT="$(timeout 120 $QEMU -machine lm3s6965evb -cpu cortex-m3 -nographic \
  -semihosting-config enable=on,target=native -kernel "$ELF" 2>&1)"
echo "$OUT_TEXT"

grep -q "video + flagship audio ran under Zephyr on Cortex-M3, checksums OK" <<<"$OUT_TEXT" \
  || { echo "FAIL: expected banner missing from the Zephyr QEMU run"; exit 1; }

echo "PASS: the g2g-noalloc pipeline executed under Zephyr's main thread on an"
echo "      emulated Cortex-M3 (wire checksum verified on-target, semihosting"
echo "      exit 0)."
