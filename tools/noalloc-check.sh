#!/usr/bin/env bash
# Link-time no-heap proof (M625) + panic-free proof (M626): build the
# `g2g-noalloc` static pipeline for a bare embedded target and assert the linked
# archive references no allocator symbols AND no `core::panicking` machinery.
# This is the machine-checkable form of the heap-free guarantee: a whole g2g
# source -> transform -> sink pipeline that links with no `#[global_allocator]`,
# no `alloc` crate dependency, and no reachable panic path (no unwraps, no
# bounds-check panics, no overflow panics anywhere the pipeline can execute).
# Finally, the same crate is built for the host and actually run (via
# host-harness.c) so the symbol proofs are backed by a real execution.
#
# The proof is run for BOTH a Cortex-M (ARM) and a RISC-V bare target (M656):
# the guarantees are ISA-independent Rust properties, so proving them on two
# ISAs shows the portability claim is real, not an ARM accident. The RISC-V
# target (riscv32imafc-unknown-none-elf) matches the ESP32-P4 class.
#
# Usage: tools/noalloc-check.sh
# Requires: rustup targets (added automatically); llvm-nm or nm; cc (host run).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/examples/g2g-noalloc/Cargo.toml"
TARGETS=("thumbv7em-none-eabihf" "riscv32imafc-unknown-none-elf")

if command -v llvm-nm >/dev/null 2>&1; then NM=llvm-nm; else NM=nm; fi

# Assert the g2g-noalloc archive for $1 has zero allocator symbols, zero panic
# machinery, and the pipeline entry points (so the proof is about emitted code,
# not a dead-code-eliminated nothing).
check_target() {
  local target="$1"
  echo "== building g2g-noalloc for $target (default-features=false: no alloc crate) =="
  rustup target add "$target" >/dev/null 2>&1 || true
  cargo build --manifest-path "$MANIFEST" --release --target "$target"

  local ar="$ROOT/examples/g2g-noalloc/target/$target/release/libg2g_noalloc.a"
  [ -f "$ar" ] || { echo "FAIL: archive not produced at $ar"; exit 1; }

  echo "== checking $target archive for allocator + panic symbols with $NM =="
  # Capture the symbol table once (ignore nm's exit code: it can warn non-zero on
  # archive members with no symbols, which pipefail would otherwise propagate).
  local syms
  syms="$($NM "$ar" 2>/dev/null || true)"

  # The Rust allocator shims / alloc-crate symbols that would appear if any
  # reachable code used the heap. None of these may be present.
  local alloc_pattern='__rust_alloc|__rust_dealloc|__rust_realloc|__rust_alloc_zeroed|__rustc.*alloc|__rg_alloc|_ZN5alloc|rust_oom|handle_alloc_error'
  # Here-strings (not `printf | grep`): with `pipefail`, `grep -q` closing the
  # pipe early would SIGPIPE the producer and be misread as failure.
  local hits
  hits="$(grep -E "$alloc_pattern" <<<"$syms" || true)"
  if [ -n "$hits" ]; then
    echo "FAIL ($target): allocator symbols present in the no-alloc pipeline:"
    echo "$hits"
    exit 1
  fi

  # Panic-free (M626): no `core::panicking` machinery may remain either. Every
  # reachable path avoids unwrap / slice-index / overflow panics, and the
  # single-poll executor lets the optimizer discharge the compiler's
  # resumed-after-completion guard, so the optimized archive has no panic
  # symbols at all (which also proves the `#[panic_handler]` is dead code). The
  # v0 mangling keeps identifier strings verbatim, so a plain grep catches any
  # panic entry point (panic_fmt, panic_bounds_check, unwrap_failed, ...).
  local panic_hits
  panic_hits="$(grep -iE 'panic|unwrap_failed|expect_failed' <<<"$syms" || true)"
  if [ -n "$panic_hits" ]; then
    echo "FAIL ($target): panic machinery present in the no-alloc pipeline:"
    echo "$panic_hits"
    exit 1
  fi

  # Sanity: the pipeline entry symbols must exist, so we proved something real.
  for sym in g2g_noalloc_run g2g_audio_run; do
    if ! grep -q "$sym" <<<"$syms"; then
      echo "FAIL ($target): $sym symbol missing; the pipeline was eliminated, nothing proven"
      exit 1
    fi
  done
  echo "OK ($target): zero allocator symbols, zero panic symbols, entry points present"
}

for t in "${TARGETS[@]}"; do
  check_target "$t"
done

# Behavioral sanity (ISA-independent, run once): build the same crate for the
# host, link the C harness, and run the pipeline for real (64 frames through the
# SPI display element onto a stub bus, wire checksum asserted), so the symbol
# proofs above are about code that demonstrably executes to completion.
if command -v cc >/dev/null 2>&1; then
  echo "== running the pipeline on the host via host-harness.c =="
  cargo build --manifest-path "$MANIFEST" --release
  HOSTDIR="$ROOT/examples/g2g-noalloc/target/release"
  cc "$ROOT/examples/g2g-noalloc/host-harness.c" "$HOSTDIR/libg2g_noalloc.a" \
     -o "$HOSTDIR/noalloc-harness"
  "$HOSTDIR/noalloc-harness"
else
  echo "SKIP: no host C compiler; skipping the behavioral run"
fi

# The flagship audio graph's host validation (M644): wire structure, DSP
# semantics vs an independent float reference, and the pinned cross-target
# checksum, all against the same crate the symbol proofs cover.
echo "== validating the flagship audio graph (noalloc-pipeline tests) =="
cargo test --manifest-path "$ROOT/examples/noalloc-pipeline/Cargo.toml" --quiet

echo "PASS: g2g-noalloc links the video pipeline AND the flagship audio graph"
echo "      (capture->convert->resample->mix->encode->RTP) with zero allocator"
echo "      symbols and zero panic symbols on BOTH ${TARGETS[*]}."
echo "      The heap-free + panic-free guarantees hold at link time, cross-ISA."
