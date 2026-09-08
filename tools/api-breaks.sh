#!/usr/bin/env bash
# The API breaks g2g-core's unreleased version carries against the last release
# on crates.io. `record` rewrites the list, `check` fails when it has drifted.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
record="$root/api-breaks-since-release.txt"
mode="${1:-check}"

header() {
    cat <<'EOF'
# Every API break g2g-core carries against the last release on crates.io, one
# per line as `<lint>: <item>`. Regenerate with `tools/api-breaks.sh record`.
#
# cargo-semver-checks derives its release type from the version, so once the
# workspace carries the pre-1.0 breaking bump every lint skips and the `semver`
# CI job verifies nothing until the next publish. This list is what still holds
# a break to a deliberate act during that window, and it is the raw material for
# the release notes when the bump ships.
EOF
}

# `--release-type patch` forbids breakage rather than deriving that it is
# allowed, which is what makes the lints run and name what broke.
raw=$(cd "$root" && cargo semver-checks --package g2g-core --release-type patch --color never 2>&1) || true

if ! printf '%s\n' "$raw" | grep -q '^ *Summary '; then
    echo "tools/api-breaks.sh: cargo semver-checks did not finish" >&2
    printf '%s\n' "$raw" >&2
    exit 1
fi

found=$(printf '%s\n' "$raw" \
    | awk '/^--- failure /{lint=$3; sub(/:$/,"",lint); next}
           /^---/{lint=""; next}
           lint!="" && /^  [^ ]/{sub(/^ */,""); print lint": "$0}' \
    | sed -E 's/,?[[:space:]]*(previously[[:space:]]+)?in[[:space:]]+(file[[:space:]]+)?\/.*$//' \
    | sort -u)

case "$mode" in
record)
    { header; printf '%s\n' "$found"; } >"$record"
    echo "recorded $(printf '%s\n' "$found" | grep -c .) breaks in $record"
    ;;
check)
    if diff -u <({ header; printf '%s\n' "$found"; }) "$record"; then
        echo "api breaks match $record"
    else
        echo "::error::$record is stale. Regenerate with: tools/api-breaks.sh record" >&2
        exit 1
    fi
    ;;
*)
    echo "usage: tools/api-breaks.sh [record|check]" >&2
    exit 2
    ;;
esac
