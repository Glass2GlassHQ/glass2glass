#!/usr/bin/env python3
"""Requirements-traceability checker (M655): verify that every safety
requirement in docs/safety/REQUIREMENTS.md is backed by evidence that actually
exists in the repository, and that every proof it cites is wired into CI.

This turns the traceability matrix from a document into a checkable claim (the
validation-first posture applied to the safety case): if a requirement's cited
test is renamed, its proof script deleted, or its CI job removed, this fails, so
the matrix cannot silently drift from the code that satisfies it.

Evidence tokens in the matrix's Evidence column (whitespace-separated):
  script:tools/foo-check.sh   a proof script; must exist AND run in a CI workflow
  test:some_test_fn           a `#[test]` fn; must exist in a .rs file
  job:features-linux          a CI job key; must exist in a workflow file

Usage: tools/traceability-check.py [repo-root]
Exit 0 if every requirement is fully traced, 1 otherwise. Stdlib only.
"""

import os
import re
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
MATRIX = os.path.join(ROOT, "docs", "safety", "REQUIREMENTS.md")
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")

# A floor so the matrix cannot be quietly gutted to pass.
MIN_REQUIREMENTS = 12


def collect_test_fns():
    """Every `fn <name>` defined in a .rs file (test fns live among them)."""
    names = set()
    fn_re = re.compile(r"\bfn\s+([a-zA-Z0-9_]+)")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Skip build output and vendored trees.
        dirnames[:] = [d for d in dirnames if d not in ("target", ".git")]
        for fn in filenames:
            if fn.endswith(".rs"):
                try:
                    with open(os.path.join(dirpath, fn), "r", errors="ignore") as f:
                        for m in fn_re.finditer(f.read()):
                            names.add(m.group(1))
                except OSError:
                    pass
    return names


def read_workflows():
    text = ""
    keys = set()
    key_re = re.compile(r"^  ([A-Za-z0-9_-]+):", re.MULTILINE)
    if os.path.isdir(WORKFLOWS):
        for fn in os.listdir(WORKFLOWS):
            if fn.endswith((".yml", ".yaml")):
                with open(os.path.join(WORKFLOWS, fn), "r", errors="ignore") as f:
                    body = f.read()
                text += body
                keys.update(key_re.findall(body))
    return text, keys


def parse_matrix():
    """Return list of (id, evidence_tokens) from the markdown table."""
    rows = []
    with open(MATRIX, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5:
                continue
            rid = cells[0]
            if not rid.startswith("REQ-"):
                continue  # header / separator / prose rows
            # Tokens may be wrapped in markdown code spans; strip backticks.
            tokens = [t.strip("`") for t in cells[4].split() if t.strip("`")]
            rows.append((rid, tokens))
    return rows


def main():
    if not os.path.isfile(MATRIX):
        print(f"FAIL: requirements matrix not found at {MATRIX}")
        return 1

    test_fns = collect_test_fns()
    wf_text, wf_keys = read_workflows()
    rows = parse_matrix()

    errors = []
    seen_ids = set()

    if len(rows) < MIN_REQUIREMENTS:
        errors.append(f"only {len(rows)} requirements; expected at least {MIN_REQUIREMENTS}")

    for rid, tokens in rows:
        if rid in seen_ids:
            errors.append(f"{rid}: duplicate requirement id")
        seen_ids.add(rid)
        if not tokens:
            errors.append(f"{rid}: no evidence cited")
            continue
        for tok in tokens:
            if ":" not in tok:
                errors.append(f"{rid}: malformed evidence token '{tok}'")
                continue
            kind, ref = tok.split(":", 1)
            if kind == "script":
                path = os.path.join(ROOT, ref)
                if not os.path.isfile(path):
                    errors.append(f"{rid}: script evidence '{ref}' does not exist")
                elif os.path.basename(ref) not in wf_text:
                    errors.append(f"{rid}: script '{ref}' is not invoked by any CI workflow")
            elif kind == "test":
                if ref not in test_fns:
                    errors.append(f"{rid}: test evidence '{ref}' not found in any .rs file")
            elif kind == "job":
                if ref not in wf_keys:
                    errors.append(f"{rid}: CI job '{ref}' not found in any workflow")
            else:
                errors.append(f"{rid}: unknown evidence kind '{kind}'")

    if errors:
        print("FAIL: requirements traceability is incomplete:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"PASS: {len(rows)} safety requirements, every evidence link resolves")
    print("      (proof scripts exist and run in CI, cited tests exist, jobs exist).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
