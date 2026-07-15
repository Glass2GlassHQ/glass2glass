#!/usr/bin/env bash
# Build-time worst-case RAM / stack / ROM report (M627) for the heap-free
# pipeline: link `examples/g2g-noalloc` into a real gc-sectioned ELF for a bare
# Cortex-M target and have tools/footprint.py report ROM, static RAM, and the
# worst-case stack (computed from the disassembly call graph, so it includes
# the capture ring + pipeline state machine, which live in the entry frame).
# The budgets below are a regression guard: CI fails if the pipeline outgrows
# them. Appends a markdown table to $GITHUB_STEP_SUMMARY when set.
#
# Usage: tools/footprint-report.sh
# Requires: rustup target thumbv7em-none-eabihf; python3; llvm-size +
# llvm-objdump (from PATH or the rustup llvm-tools component, which this
# script installs; a host GNU objdump cannot disassemble the ARM ELF);
# rust-lld (ships with the rust toolchain).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/examples/g2g-noalloc/Cargo.toml"
TARGET="thumbv7em-none-eabihf"

# The regression budgets (bytes), one row per proof pipeline (each linked as
# its own gc-sectioned ELF from the shared archive, so the numbers stay
# per-pipeline). Static RAM has no headroom on purpose: the no-alloc
# pipelines own no globals, and that claim should break loudly.
#
# Video pipeline (real camera seam M630 + SPI display element M629, exposed
# as a plain future M632, safe drive_ready executor + const-init ring M634,
# Caps::Tensor negotiated on the transform link M636): measured 2026-07 at
# 4096 ROM / 0 static / 1388 stack (opt-level "s" since M644: "z" stopped
# inlining the grown archive's async state machines, which reintroduced
# panic paths; "s" is also smaller here).
MAX_ROM=5120
MAX_STACK=2048
MAX_STATIC_RAM=0
# Flagship audio graph (M644: two captures -> convert -> resample -> mix ->
# G.711 encode -> RTP): measured 2026-07 at 10572 ROM / 0 static / 6504
# stack. The resampler tables + polyphase MAC and the u128 media-clock
# division dominate the extra ROM; the 48 kHz capture ring dominates the
# stack.
AUDIO_MAX_ROM=12288
AUDIO_MAX_STACK=8192
AUDIO_MAX_STATIC_RAM=0

echo "== building g2g-noalloc for $TARGET =="
rustup target add "$TARGET" >/dev/null 2>&1 || true
cargo build --manifest-path "$MANIFEST" --release --target "$TARGET"

AR="$ROOT/examples/g2g-noalloc/target/$TARGET/release/libg2g_noalloc.a"

# rust-lld ships inside the toolchain's sysroot, not on PATH.
SYSROOT="$(rustc --print sysroot)"
LLD="$(find "$SYSROOT" -name rust-lld -type f | head -1)"
[ -n "$LLD" ] || { echo "FAIL: rust-lld not found in the rust sysroot"; exit 1; }

# llvm-objdump / llvm-size: PATH if present, else the rustup llvm-tools
# component (same sysroot as rust-lld). A host GNU objdump can't disassemble
# the ARM ELF, so there is no binutils fallback.
OBJDUMP="$(command -v llvm-objdump || true)"
SIZE="$(command -v llvm-size || true)"
if [ -z "$OBJDUMP" ] || [ -z "$SIZE" ]; then
  rustup component add llvm-tools >/dev/null 2>&1 \
    || rustup component add llvm-tools-preview >/dev/null 2>&1 || true
  OBJDUMP="${OBJDUMP:-$(find "$SYSROOT" -name llvm-objdump -type f | head -1)}"
  SIZE="${SIZE:-$(find "$SYSROOT" -name llvm-size -type f | head -1)}"
fi
[ -n "$OBJDUMP" ] && [ -n "$SIZE" ] \
  || { echo "FAIL: llvm-objdump / llvm-size not found (PATH or llvm-tools)"; exit 1; }

# Link and report one pipeline entry against its budgets; accumulates a
# non-zero status so every row prints before a budget failure exits.
STATUS=0
report_entry() {
  local title="$1" entry="$2" ar="$3" isa="$4" tgt="$5" elf="$6" max_rom="$7" max_stack="$8" max_static="$9"
  echo "== linking $elf (gc-sections, entry $entry) =="
  "$LLD" -flavor gnu --gc-sections -e "$entry" -o "$elf" "$ar"
  echo "== footprint: $title ($tgt) =="
  # `|| STATUS=$?` so a budget failure still prints + publishes before exiting
  # (a plain assignment would abort here under `set -e`).
  local report
  report="$(python3 "$ROOT/tools/footprint.py" "$elf" --entry "$entry" --isa "$isa" \
    --objdump "$OBJDUMP" --size "$SIZE" \
    --max-rom "$max_rom" --max-stack "$max_stack" --max-static-ram "$max_static")" || STATUS=$?
  echo "$report"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "### $title footprint ($tgt)"
      echo
      echo '```'
      echo "$report"
      echo '```'
    } >> "$GITHUB_STEP_SUMMARY"
  fi
}

ARM_OUT="$ROOT/examples/g2g-noalloc/target/$TARGET/release"
report_entry "Heap-free video pipeline" g2g_noalloc_run "$AR" arm "$TARGET" \
  "$ARM_OUT/noalloc.elf" "$MAX_ROM" "$MAX_STACK" "$MAX_STATIC_RAM"
report_entry "Flagship audio graph" g2g_audio_run "$AR" arm "$TARGET" \
  "$ARM_OUT/audio.elf" "$AUDIO_MAX_ROM" "$AUDIO_MAX_STACK" "$AUDIO_MAX_STATIC_RAM"

# RISC-V footprint (M656): the same heap-free video pipeline linked for a bare
# RISC-V target (riscv32imafc, the ESP32-P4 class), proving the footprint
# guarantee is not an ARM accident. Measured 2026-07 at 3718 ROM / 0 static /
# 1328 stack, within the same budgets as the ARM build. (The flagship audio
# graph's RISC-V stack is budgeted too as of M657, below.)
RV_TARGET="riscv32imafc-unknown-none-elf"
echo "== building g2g-noalloc for $RV_TARGET =="
rustup target add "$RV_TARGET" >/dev/null 2>&1 || true
cargo build --manifest-path "$MANIFEST" --release --target "$RV_TARGET"
RV_AR="$ROOT/examples/g2g-noalloc/target/$RV_TARGET/release/libg2g_noalloc.a"
RV_OUT="$ROOT/examples/g2g-noalloc/target/$RV_TARGET/release"
report_entry "Heap-free video pipeline" g2g_noalloc_run "$RV_AR" riscv "$RV_TARGET" \
  "$RV_OUT/noalloc.elf" "$MAX_ROM" "$MAX_STACK" "$MAX_STATIC_RAM"

# Flagship audio graph on RISC-V (M657): rustc encodes this frame as a constant
# too large for addi's 12-bit immediate, so it materializes the size into a
# register and does `sub sp, sp, <reg>`. footprint.py now resolves that register
# to its compile-time constant (and fails rather than under-report if it ever
# is not one), so the RISC-V audio stack is budgeted like the others. Measured
# 2026-07 at 10852 ROM / 0 static / 6432 stack, within the ARM audio budgets.
report_entry "Flagship audio graph" g2g_audio_run "$RV_AR" riscv "$RV_TARGET" \
  "$RV_OUT/audio.elf" "$AUDIO_MAX_ROM" "$AUDIO_MAX_STACK" "$AUDIO_MAX_STATIC_RAM"

exit "$STATUS"
