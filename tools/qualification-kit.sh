#!/usr/bin/env bash
# Qualification kit runner (M655): run the full set of MCU safety proofs and
# print a consolidated requirement -> evidence -> result report, the evidence
# package an integrator attaches to a product safety case. Each proof also runs
# individually in CI; this gathers them into one verdict on demand.
#
# The lightweight proofs (traceability, no-heap/panic-free, footprint, host
# tests) always run. The on-target proofs (QEMU execution, timing, FreeRTOS,
# Zephyr) run only if their toolchain is present (qemu-system-arm, override with
# $QEMU_SYSTEM_ARM; arm-none-eabi-gcc), and are reported SKIP otherwise, so the
# kit runs anywhere and never reports a false FAIL for a missing tool.
#
# Usage: tools/qualification-kit.sh
# Exit 0 if nothing FAILed (SKIPs are allowed), 1 otherwise.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QEMU="${QEMU_SYSTEM_ARM:-qemu-system-arm}"
have_qemu() { command -v "${QEMU%% *}" >/dev/null 2>&1; }
have_armgcc() { command -v arm-none-eabi-gcc >/dev/null 2>&1; }

declare -a NAMES REQS RESULTS
FAILED=0

# run_step <name> <req-ids> <gate> <command...>
# gate: "always", "qemu", or "armgcc" (a toolchain precondition).
run_step() {
  local name="$1" reqs="$2" gate="$3"; shift 3
  local result
  case "$gate" in
    qemu)   have_qemu   || { record "$name" "$reqs" "SKIP (no qemu)"; return; } ;;
    armgcc) { have_qemu && have_armgcc; } || { record "$name" "$reqs" "SKIP (no arm toolchain)"; return; } ;;
  esac
  echo "== $name =="
  if "$@" >/tmp/qualkit.$$.log 2>&1; then
    result="PASS"
  else
    result="FAIL"
    FAILED=1
    tail -n 15 /tmp/qualkit.$$.log
  fi
  rm -f /tmp/qualkit.$$.log
  record "$name" "$reqs" "$result"
}

record() {
  NAMES+=("$1"); REQS+=("$2"); RESULTS+=("$3")
}

run_step "requirements traceability" "REQ-TRACE-01"                    always bash tools/traceability-check.sh
run_step "no-heap + panic-free"      "REQ-MEM-01, REQ-MEM-02"          always bash tools/noalloc-check.sh
run_step "footprint budgets"         "REQ-MEM-03"                      always bash tools/footprint-report.sh
run_step "host safety tests"         "REQ-MEM-04, REQ-UNSAFE-01, REQ-FAULT-01, REQ-INPUT-01, REQ-CONC-01, REQ-RECV-01, REQ-INTEG-01, REQ-FAULT-02, REQ-FAULT-03" \
                                     always cargo test -p g2g-mcu --quiet
run_step "on-target execution"       "REQ-EXEC-01, REQ-FAULT-02, REQ-FAULT-03, REQ-CONC-01, REQ-RECV-01" qemu bash tools/qemu-check.sh
run_step "deterministic timing"      "REQ-TIME-01"                     qemu   bash tools/timing-report.sh
run_step "FreeRTOS executor"         "REQ-EXEC-01"                     armgcc bash tools/freertos-check.sh
run_step "Zephyr executor"           "REQ-EXEC-01"                     armgcc bash tools/zephyr-check.sh

echo
echo "================ qualification report ================"
printf "%-26s %-10s %s\n" "PROOF" "RESULT" "REQUIREMENTS"
printf "%-26s %-10s %s\n" "-----" "------" "------------"
for i in "${!NAMES[@]}"; do
  printf "%-26s %-10s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}" "${REQS[$i]}"
done
echo "======================================================"

if [ "$FAILED" -ne 0 ]; then
  echo "QUALIFICATION: FAIL (one or more proofs failed)"
  exit 1
fi
echo "QUALIFICATION: PASS (all runnable proofs passed; see SKIPs for tools not present)"
