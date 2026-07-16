#!/usr/bin/env bash
# Requirements-traceability check (M655): verify docs/safety/REQUIREMENTS.md is
# fully backed by evidence that exists in the repo (proof scripts wired into CI,
# named tests, CI jobs). A thin wrapper over tools/traceability-check.py (stdlib
# only, no dependencies), so it runs in the same CI job as the other proofs.
#
# Usage: tools/traceability-check.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/tools/traceability-check.py" "$ROOT"
