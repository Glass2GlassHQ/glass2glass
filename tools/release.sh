#!/usr/bin/env bash
# Release the workspace version to crates.io. See RELEASING.md.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# dependency order: a crate is published after every g2g crate it depends on
CRATES=(g2g-core g2g-mcugen g2g-mcu g2g-plugin g2g-plugins g2g-bridge g2g-ml g2g-capi g2g-python)
CRATES_IO_API="https://crates.io/api/v1/crates"
USER_AGENT="g2g-release (https://github.com/Glass2GlassHQ/glass2glass)"
RATE_LIMIT_WAIT_SECONDS=600
RELEASE_BRANCH=master

workspace_version() {
    sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1
}

is_published() {
    local status
    status=$(curl -s -o /dev/null -w '%{http_code}' -A "$USER_AGENT" "$CRATES_IO_API/$1/$2")
    [ "$status" = 200 ]
}

require_clean_release_branch() {
    [ "$(git branch --show-current)" = "$RELEASE_BRANCH" ] || { echo "not on $RELEASE_BRANCH" >&2; exit 1; }
    [ -z "$(git status --porcelain)" ] || { echo "working tree is not clean" >&2; exit 1; }
    git fetch -q origin
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$RELEASE_BRANCH")" ] \
        || { echo "$RELEASE_BRANCH differs from origin/$RELEASE_BRANCH" >&2; exit 1; }
}

cmd_check() {
    require_clean_release_branch
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    cargo test --workspace --features linux-full --no-fail-fast
    cargo run -q -p g2g-python --features launch,g2g-plugins/linux-full --bin g2g-launch-py -- --docgen docs/elements.html
    git diff --exit-code docs/elements.html
    tools/api-breaks.sh check
    cargo publish --dry-run --workspace
    # the dry run's verify builds leave a second g2g_core that breaks the next g2g-bridge test build
    cargo clean -p g2g-bridge
    echo "release checks passed for $(workspace_version)"
}

publish_crate() {
    local crate=$1 version=$2
    until is_published "$crate" "$version"; do
        cargo publish -p "$crate" && break
        echo "publishing $crate failed, retrying in $RATE_LIMIT_WAIT_SECONDS s (crates.io rate limit)" >&2
        sleep "$RATE_LIMIT_WAIT_SECONDS"
    done
    echo "$crate $version is on crates.io"
}

cmd_publish() {
    local version tag
    version=$(workspace_version)
    tag="v$version"
    if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
        require_clean_release_branch
        sed -i "0,/^## Unreleased$/s//## Unreleased\n\n## $version/" CHANGELOG.md
        # the quick start's dependency lines
        sed -i -E "s/^(g2g-[a-z]+ += (\{ version = )?)\"[0-9]+\.[0-9]+\"/\1\"${version%.*}\"/" README.md
        git commit -q -am "release $version"
        git tag "$tag"
    fi
    for crate in "${CRATES[@]}"; do
        publish_crate "$crate" "$version"
    done
    cargo clean -p g2g-bridge
    git push origin "$RELEASE_BRANCH" "$tag"
}

cmd_bump() {
    local next=$1 current
    current=$(workspace_version)
    # only the workspace version and the g2g-* path dependencies, never a third-party pin
    sed -i -E "/^(version|g2g-[a-z]+) = /s/\"$current\"/\"$next\"/" Cargo.toml g2g-plugins/Cargo.toml
    # the default user-agent property shows the version on the element cards
    sed -i "s|<code>g2g/$current</code>|<code>g2g/$next</code>|" docs/elements.html
    cargo update -q --workspace
    tools/api-breaks.sh record
    git commit -q -am "rev to $next"
    echo "bumped $current to $next, push when ready"
}

case "${1:-}" in
check) cmd_check ;;
publish) cmd_publish ;;
bump)
    [ -n "${2:-}" ] || { echo "usage: tools/release.sh bump <next-version>" >&2; exit 2; }
    cmd_bump "$2"
    ;;
*)
    echo "usage: tools/release.sh check | publish | bump <next-version>" >&2
    exit 2
    ;;
esac
