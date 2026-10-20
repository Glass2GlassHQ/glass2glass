#!/usr/bin/env bash
# Runs the gst-python-ml people.mp4 clip through YOLO11m twice per side:
# native `ortinfer execution-provider=cuda ! detectionpostprocess` and the hosted
# gst-python-ml `ObjectDetector` on cuda:0, both in g2g-launch-py, and prints one
# row per run from the run summary and /usr/bin/time.
#
# Build first (PYO3_PYTHON must be the venv the hosted element imports from):
#   PYO3_PYTHON=$HOME/src/gst-python-ml/.venv/bin/python \
#     cargo build --release -p g2g-python --features ml-cuda --bin g2g-launch-py
#
# G2G_PYML_DIR is the gst-python-ml checkout (default ~/src/gst-python-ml),
# G2G_LAUNCH_PY the binary (default target/release/g2g-launch-py), RUNS the
# runs per side (default 2).
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
pyml=${G2G_PYML_DIR:-$HOME/src/gst-python-ml}
launch=${G2G_LAUNCH_PY:-$repo/target/release/g2g-launch-py}
runs=${RUNS:-2}
clip=$pyml/data/people.mp4
model=$pyml/yolo11m.onnx
# the size YOLO11m was exported at
model_side=640

for path in "$launch" "$clip" "$model" "$pyml/.venv/bin/python"; do
    if [ ! -e "$path" ]; then
        echo "ortinfer-bench: missing $path" >&2
        exit 1
    fi
done

python_version=$("$pyml/.venv/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
export PYTHONPATH="$pyml/plugins/python:$pyml/.venv/lib/$python_version/site-packages:$pyml/.venv/lib64/$python_version/site-packages:$HOME/.local/lib/$python_version/site-packages"

decode="filesrc location=$clip ! qtdemux ! h264parse ! ffmpegdec ! videoconvert ! videoscale ! video/x-raw,format=RGBA,width=$model_side,height=$model_side"
native="$decode ! ortinfer model=$model execution-provider=cuda ! detectionpostprocess ! fakesink"
hosted="$decode ! pyelement module=objectdetector class=ObjectDetector engine-name=onnx model-name=$model device=cuda:0 ! fakesink"

scratch=$(mktemp -d "$HOME/tmp/ortinfer-bench.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# one row: side, run, frames, fps, wall ms per frame, inference p50 and p99 ms, peak RSS MiB
measure() {
    local side=$1 run=$2 element=$3 pipeline=$4
    local log=$scratch/$side-$run.log
    # word splitting of $pipeline is the launch line's own tokenization
    # shellcheck disable=SC2086
    if ! /usr/bin/time -f 'peak-rss-kib %M' "$launch" $pipeline >"$log" 2>&1; then
        echo "ortinfer-bench: $side run $run failed, log follows" >&2
        cat "$log" >&2
        exit 1
    fi
    awk -v side="$side" -v run="$run" -v element="$element" '
        /^  frames:/ { frames = $3; sub(",", "", frames) }
        /^  run:/ { wall = $2; fps = $5 }
        $1 ~ "^" element && $2 == "proc" { p50 = $4; p99 = $8 }
        /^peak-rss-kib/ { rss = $2 / 1024 }
        END {
            if (fps == "" || p50 == "") { print "ortinfer-bench: no run summary for " side > "/dev/stderr"; exit 1 }
            printf "%-7s %3s %6s %7s %13.1f %9s %9s %13.0f\n", side, run, frames, fps, 1000 / fps, p50, p99, rss
        }' "$log"
}

echo "$(date -I) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), load $(cut -d' ' -f1-3 /proc/loadavg)"
printf "%-7s %3s %6s %7s %13s %9s %9s %13s\n" side run frames fps "wall ms/frame" "p50 ms" "p99 ms" "peak RSS MiB"
for run in $(seq "$runs"); do
    measure native "$run" ortinfer "$native"
    measure hosted "$run" pyelement "$hosted"
done
