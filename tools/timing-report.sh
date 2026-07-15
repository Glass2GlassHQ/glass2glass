#!/usr/bin/env bash
# Per-frame timing / jitter report (M645) for the flagship audio graph, the
# timing sibling of tools/footprint-report.sh: boot examples/g2g-qemu's
# `timing` binary on QEMU MPS2-AN386 under `-icount shift=0,sleep=off`, where
# virtual time is a pure function of the executed instruction stream, so the
# per-frame numbers are deterministic (asserted: two boots must report
# identical lines) and can be budget-enforced in CI like the memory numbers.
# Without icount the same binary measures host scheduling noise (~1000x the
# jitter), which is exactly what this report exists to exclude.
#
# The binary stamps SysTick (25 MHz emulated system clock) on every emitted
# RTP packet: `first` carries the one-time cost (caps negotiation, resampler
# warm-up), `steady_*` are frames 2..N, `jitter` = steady_max - steady_min.
# Ticks convert to microseconds at /25. Emulated instruction timing is not
# cycle-accurate silicon timing (that is the on-device Hardware row's job);
# what it does prove is that the pipeline's execution cost is bounded and
# data-stable, and CI catches any regression that grows it.
#
# Usage: tools/timing-report.sh
# Requires: rustup target thumbv7em-none-eabihf; qemu-system-arm (override
# with $QEMU_SYSTEM_ARM).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="thumbv7em-none-eabihf"
QEMU="${QEMU_SYSTEM_ARM:-qemu-system-arm}"
ELF="$ROOT/examples/g2g-qemu/target/$TARGET/release/timing"

# The regression budgets (SysTick ticks at the emulated 25 MHz). Measured
# 2026-07: first=17136, steady_max=19093 (~764 us of a 10 ms frame period,
# ~7.6% of an emulated 25 MHz core), jitter=9 (~360 ns; the residual spread
# is the G.711 encoder's amplitude-dependent segment search). Budgets leave
# headroom for toolchain drift, not for algorithmic regressions.
MAX_FIRST_TICKS=24576
MAX_STEADY_TICKS=24576
MAX_JITTER_TICKS=256

command -v "${QEMU%% *}" >/dev/null 2>&1 \
  || { echo "FAIL: $QEMU not found (install qemu-system-arm or set \$QEMU_SYSTEM_ARM)"; exit 1; }

echo "== building the timing binary for $TARGET =="
rustup target add "$TARGET" >/dev/null 2>&1 || true
(cd "$ROOT/examples/g2g-qemu" && cargo build --release --target "$TARGET" --bin timing)

run_once() {
  timeout 120 $QEMU -machine mps2-an386 -cpu cortex-m4 -nographic \
    -semihosting-config enable=on,target=native \
    -icount shift=0,sleep=off -kernel "$ELF"
}

echo "== running on QEMU MPS2-AN386 (Cortex-M4, icount) =="
OUT1="$(run_once)"
echo "$OUT1"
OUT2="$(run_once)"

# Determinism: under icount the report is a pure function of the binary; a
# difference means the measurement (not the pipeline) is unsound.
[ "$OUT1" = "$OUT2" ] \
  || { echo "FAIL: two icount runs differ:"; echo "run1: $OUT1"; echo "run2: $OUT2"; exit 1; }

LINE="$(grep '^g2g-timing: frames=' <<<"$OUT1")" \
  || { echo "FAIL: timing line missing from the QEMU run"; exit 1; }

field() { sed -n "s/.*$1=\([0-9]*\).*/\1/p" <<<"$LINE"; }
FRAMES="$(field frames)"
FIRST="$(field first)"
STEADY_MAX="$(field steady_max)"
JITTER="$(field jitter)"

STATUS=0
check() { # name value budget
  if [ "$2" -gt "$3" ]; then
    echo "FAIL: $1 = $2 ticks exceeds the budget of $3"
    STATUS=1
  fi
}
[ "$FRAMES" = "50" ] || { echo "FAIL: expected 50 frames, got $FRAMES"; exit 1; }
check "first frame" "$FIRST" "$MAX_FIRST_TICKS"
check "steady-state worst case" "$STEADY_MAX" "$MAX_STEADY_TICKS"
check "steady-state jitter" "$JITTER" "$MAX_JITTER_TICKS"

US_STEADY="$(awk -v t="$STEADY_MAX" 'BEGIN { printf "%.1f", t / 25 }')"
US_JITTER="$(awk -v t="$JITTER" 'BEGIN { printf "%.2f", t / 25 }')"
SUMMARY="steady-state worst case ${STEADY_MAX} ticks (${US_STEADY} us of a 10 ms frame), jitter ${JITTER} ticks (${US_JITTER} us), deterministic across boots"
echo "$SUMMARY"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Flagship audio graph timing (QEMU icount, 25 MHz SysTick)"
    echo
    echo '```'
    echo "$LINE"
    echo "$SUMMARY"
    echo '```'
  } >> "$GITHUB_STEP_SUMMARY"
fi

[ "$STATUS" = 0 ] && echo "PASS: per-frame execution cost is bounded and deterministic under icount."
exit "$STATUS"
