#!/usr/bin/env python3
"""Fail if any criterion benchmark's `head` baseline regressed beyond a ratio
vs its `base` baseline. Used by `.github/workflows/bench.yml`.

Usage: bench_compare.py <criterion-dir> <max-ratio>
  e.g. bench_compare.py g2g-bench/target/criterion 1.5  (fail on > 50% slower)

Compares the `mean.point_estimate` (nanoseconds) of each benchmark that has both
a `base/` and a `head/` baseline. Benchmarks present in only one (e.g. added in
the PR) are skipped. Exit 1 if any regressed past the ratio, else 0.
"""
import glob
import json
import os
import sys


def mean_ns(estimates_path):
    with open(estimates_path) as f:
        return json.load(f)["mean"]["point_estimate"]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    root, threshold = sys.argv[1], float(sys.argv[2])

    failures = []
    checked = 0
    for base_est in sorted(glob.glob(os.path.join(root, "**", "base", "estimates.json"), recursive=True)):
        bench_dir = os.path.dirname(os.path.dirname(base_est))
        head_est = os.path.join(bench_dir, "head", "estimates.json")
        if not os.path.exists(head_est):
            continue
        base = mean_ns(base_est)
        head = mean_ns(head_est)
        ratio = head / base if base else 1.0
        checked += 1
        regressed = ratio > threshold
        status = "REGRESSION" if regressed else "ok"
        name = os.path.relpath(bench_dir, root)
        print(f"{status:11} {name}: base={base:.0f}ns head={head:.0f}ns ({ratio:.2f}x)")
        if regressed:
            failures.append((name, ratio))

    if checked == 0:
        print("no comparable benchmarks (base baseline missing); skipping")
        return 0
    if failures:
        print(f"\n{len(failures)} benchmark(s) regressed past {threshold}x:")
        for name, ratio in failures:
            print(f"  {name}: {ratio:.2f}x")
        return 1
    print(f"\nall {checked} benchmark(s) within {threshold}x of base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
