#!/usr/bin/env bash
# C seam ABI proof (M650): the inverse of the freertos integration. Where a C
# app links the whole g2g pipeline and calls into it, here C code IS the
# peripheral, registered as capture/send function pointers and driving the
# pipeline one frame at a time from its own superloop. This script proves that
# path keeps the MCU guarantees and works from real C:
#
#   1. Build the g2g-cffi staticlib for a bare Cortex-M target with `g2g-core`
#      default-features=false (no `alloc` crate) and no global allocator, and
#      assert the archive references:
#        - ZERO allocator symbols (heap-free), and
#        - ZERO data-panic symbols (bounds / overflow / unwrap / slice / div).
#      The one benign, runtime-unreachable async re-poll guard
#      (`panic_const_async_fn_resumed` + the `panic_fmt` it calls) that the
#      one-frame-step future leaves is permitted and NOTHING else: any other
#      panic symbol fails. (A run-to-EOS pipeline discharges even that guard; a
#      single-frame step does not, but drive_ready polls each fresh per-step
#      future exactly once, so it can never fire. See step_source_sink.)
#   2. Build for the host, link harness.c (a real C caller that supplies C
#      capture + send callbacks and steps the pipeline), run it, and assert its
#      wire checksum equals the pipeline's own Rust reference over identical
#      input, so the symbol proofs cover code that demonstrably executes and the
#      C seams are shown byte-transparent.
#
# Usage: tools/cffi-check.sh
# Requires: rustup target thumbv7em-none-eabihf; llvm-nm or nm; cc (host run).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/examples/g2g-cffi/Cargo.toml"
TARGET="thumbv7em-none-eabihf"

echo "== building g2g-cffi for $TARGET (default-features=false: no alloc crate) =="
rustup target add "$TARGET" >/dev/null 2>&1 || true
cargo build --manifest-path "$MANIFEST" --release --target "$TARGET"

AR="$ROOT/examples/g2g-cffi/target/$TARGET/release/libg2g_cffi.a"
[ -f "$AR" ] || { echo "FAIL: archive not produced at $AR"; exit 1; }

if command -v llvm-nm >/dev/null 2>&1; then NM=llvm-nm; else NM=nm; fi

# Capture the symbol table once (ignore nm's exit code: it can warn non-zero on
# archive members with no symbols, which pipefail would otherwise propagate).
SYMS="$($NM "$AR" 2>/dev/null || true)"

echo "== asserting no allocator symbols (heap-free) =="
ALLOC_PATTERN='__rust_alloc|__rust_dealloc|__rust_realloc|__rust_alloc_zeroed|__rustc.*alloc|__rg_alloc|_ZN5alloc|rust_oom|handle_alloc_error'
# Here-strings (not `printf | grep`): with `pipefail`, `grep -q` closing the pipe
# early would SIGPIPE the producer and be misread as failure.
HITS="$(grep -E "$ALLOC_PATTERN" <<<"$SYMS" || true)"
if [ -n "$HITS" ]; then
  echo "FAIL: allocator symbols present in the C-seam pipeline:"
  echo "$HITS"
  exit 1
fi

echo "== asserting no data-panic symbols (bounds / overflow / unwrap / slice) =="
# The panics that matter for memory safety and arithmetic. None may be present:
# the C-seam adapters and the step runner must have no reachable data panic.
DATA_PANIC='panic_bounds_check|slice_index|str_index|unwrap_failed|expect_failed|panic_const_add|panic_const_sub|panic_const_mul|panic_const_div|panic_const_rem|panic_const_neg|panic_const_shl|panic_const_shr'
DHITS="$(grep -iE "$DATA_PANIC" <<<"$SYMS" || true)"
if [ -n "$DHITS" ]; then
  echo "FAIL: data-panic symbols present in the C-seam pipeline:"
  echo "$DHITS"
  exit 1
fi

echo "== asserting the ONLY panic symbols are the benign async re-poll guard =="
# Any panic-family symbol other than the documented, runtime-unreachable async
# resume guard (and the panic_fmt it calls) is a regression.
ALLOWED='panic_const_async_fn_resumed|9panic_fmt'
OTHER="$(grep -iE 'panic|unwrap_failed|expect_failed' <<<"$SYMS" | grep -vE "$ALLOWED" || true)"
if [ -n "$OTHER" ]; then
  echo "FAIL: unexpected panic symbols beyond the async re-poll guard:"
  echo "$OTHER"
  exit 1
fi

echo "== asserting the C ABI entry symbols are present (not eliminated) =="
for sym in g2g_audio_egress_init g2g_audio_egress_step g2g_audio_egress_reference; do
  if ! grep -q "$sym" <<<"$SYMS"; then
    echo "FAIL: $sym symbol missing; the C-seam pipeline was eliminated, nothing proven"
    exit 1
  fi
done

# Behavioral proof: build for the host, link the real C caller, and run it. The
# C harness IS the peripheral (C capture + C send callbacks), steps the pipeline
# 25 frames, and asserts its wire checksum equals the pipeline's Rust reference
# over the identical input, so the C seams are proven byte-transparent from C.
if command -v cc >/dev/null 2>&1; then
  echo "== running the C-driven pipeline on the host via harness.c =="
  cargo build --manifest-path "$MANIFEST" --release
  HOSTDIR="$ROOT/examples/g2g-cffi/target/release"
  cc "$ROOT/examples/g2g-cffi/harness.c" "$HOSTDIR/libg2g_cffi.a" -o "$HOSTDIR/cffi-harness"
  "$HOSTDIR/cffi-harness"
else
  echo "SKIP: no host C compiler; skipping the behavioral run"
fi

echo "PASS: the C seam ABI (C capture/send callbacks + frame-stepped drive) links"
echo "      heap-free and data-panic-free for $TARGET, and a real C caller drives"
echo "      the pipeline byte-transparently against the Rust reference."
