#!/usr/bin/env bash
# Run the whole g2g-plugins test suite on a Windows host with a small disk: one
# test binary at a time, each deleted once it has run, so the ~800 integration
# test binaries never sit on disk together (a plain `cargo test` builds all of
# them first, about 40 GB of debug binaries).
#
# A test file whose crate-level #![cfg(...)] cannot hold on Windows with the
# chosen features, or whose [[test]] required-features are not all on, is skipped
# without being built and logged as "### skip <name> <reason>". The run resumes:
# a target the log already records as done is not run again.
#
# usage (Git Bash):
#   tools/windows-test-suite.sh <log> [features]
# features default to every Windows element that needs no extra install:
#   mf-decode mf-encode mf-aac wasapi-sink wasapi-src d3d11-sink mf-video-src
#   plugin-loader plugin-signing
# Add `ffmpeg` with FFMPEG_DIR and LIBCLANG_PATH set (see the `ffmpeg` feature in
# g2g-plugins/Cargo.toml) and the ffmpeg DLLs on PATH.
#
# Summarise a finished log with:
#   grep '^test result' <log> | awk '{p+=$4; f+=$6; i+=$8} END {print p, f, i}'
set -u
LOG="${1:?usage: $0 <log> [features]}"
FEATURES="${2:-mf-decode mf-encode mf-aac wasapi-sink wasapi-src d3d11-sink mf-video-src plugin-loader plugin-signing}"
TOOLS=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$TOOLS")
# no incremental cache, and line tables only: a one-off run needs neither
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
touch "$LOG"
LOG=$(cd "$(dirname "$LOG")" && pwd)/$(basename "$LOG")
cd "$REPO" || exit 1
DEPS="${CARGO_TARGET_DIR:-$REPO/target}/debug/deps"

free_gb() { powershell -NoProfile -c "[math]::Round((Get-PSDrive C).Free/1GB,1)"; }

# every feature cargo resolves on for g2g-plugins, implied ones (std) included
ENABLED=$(cargo tree -p g2g-plugins --features "$FEATURES" -e features -i g2g-plugins --prefix none 2>/dev/null \
    | sed -n 's/^g2g-plugins feature "\([^"]*\)".*/\1/p' | sort -u | tr '\n' ' ')

# one "name feature feature..." line per [[test]] with required-features
REQUIRED=$(awk '
    /^\[\[test\]\]/ { name = ""; next }
    /^\[/ { name = "" }
    /^name = / { gsub(/"/, "", $3); name = $3 }
    /^required-features = / && name != "" {
        line = $0; sub(/^required-features = \[/, "", line); sub(/\].*/, "", line)
        gsub(/[",]/, " ", line); print name, line
    }' g2g-plugins/Cargo.toml)

missing_required() {
    local want f
    want=$(printf '%s\n' "$REQUIRED" | awk -v n="$1" '$1 == n { $1 = ""; print }')
    for f in $want; do
        case " $ENABLED " in *" $f "*) ;; *) echo "$f" ;; esac
    done
}

# drop a finished test binary (and its pdb and dep-info)
reap() {
    local f
    for f in "$DEPS/$1"-*.exe; do
        [ -e "$f" ] || continue
        rm -f "$f" "${f%.exe}.pdb" "${f%.exe}.d"
    done
}

finished() { grep -qx -- "### $* done" "$LOG"; }

run() {
    finished "$@" && return
    echo "### $*" >> "$LOG"
    cargo test -p g2g-plugins --features "$FEATURES" "$@" >> "$LOG" 2>&1
    echo "### exit $?" >> "$LOG"
    echo "### $* done" >> "$LOG"
}

echo "### enabled: $ENABLED" >> "$LOG"
run --lib
reap g2g_plugins
run --bins
for b in g2g-plugins/src/bin/*.rs; do reap "$(basename "$b" .rs | tr - _)"; done

GATES=$(perl "$TOOLS/windows-test-gate.pl" "$ENABLED" g2g-plugins/tests/*.rs)
n=0
total=$(printf '%s\n' "$GATES" | wc -l)
while read -r name verdict reason; do
    n=$((n + 1))
    if [ "$verdict" = skip ]; then
        finished --test "$name" || echo "### skip $name $reason" >> "$LOG"
        continue
    fi
    need=$(missing_required "$name" | tr '\n' ' ')
    if [ -n "$need" ]; then
        finished --test "$name" || echo "### skip $name required-features: $need" >> "$LOG"
        continue
    fi
    run --test "$name"
    reap "$name"
    if [ $((n % 50)) -eq 0 ]; then
        echo "### progress $n/$total free $(free_gb) GB" >> "$LOG"
    fi
done <<< "$GATES"

run --doc
echo "### all done free $(free_gb) GB" >> "$LOG"
