#!/usr/bin/env bash
set -euo pipefail

checkout=$(cd "$(dirname "$0")/.." && pwd)
readme=$checkout/README.md
scratch=$(mktemp -d)
export CARGO_TARGET_DIR=$checkout/target
export CARGO_INSTALL_ROOT=$scratch/install
export PATH=$CARGO_INSTALL_ROOT/bin:$PATH

fail() {
    echo "readme-quickstart-check: $*" >&2
    exit 1
}

readme_block() {
    awk -v marker="<!-- quickstart-check: $1 -->" '
        $0 == marker { found = 1; next }
        found && /^```/ { if (inside) { closed = 1; exit } inside = 1; next }
        inside { print }
        END { if (!closed) exit 1 }
    ' "$readme" || fail "no fenced block after the '$1' marker in README.md"
}

# installs from this checkout instead of crates.io
install_from_checkout() {
    local commands=${1//cargo install g2g-plugins /cargo install --path $checkout/g2g-plugins }
    if grep 'cargo install' <<<"$commands" | grep -qv -- '--path'; then
        fail "a cargo install line does not install g2g-plugins: $commands"
    fi
    echo "$commands"
}

run_commands() {
    echo "+ $1"
    bash -euo pipefail -c "$1"
}

# the Debian / Ubuntu row of the System dependencies table, Wayland sink column
packages=$(awk -F'|' '
    /^\| Distro / { for (i = 1; i <= NF; i++) if ($i ~ /^ *Wayland sink *$/) column = i }
    /^\| Debian \/ Ubuntu / && column { print $column }
    END { if (!column) exit 1 }
' "$readme" | tr -d '`') || fail "no Wayland sink column in the System dependencies table"
read -ra packages <<<"$packages"
[ "${#packages[@]}" -gt 0 ] || fail "no Debian / Ubuntu packages listed for the Wayland sink"
apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends "${packages[@]}"

cd "$scratch"
tools_block=$(readme_block tools)
tools_commands=$(install_from_checkout "$tools_block")
run_commands "$tools_commands"
migrate_commands=$(readme_block migrate)
run_commands "$migrate_commands"
g2g-inspect videotestsrc

mcp_install=$(grep -o 'cargo install [^`]*--bin g2g-mcp' "$readme") || fail "no g2g-mcp install line in README.md"
mcp_install=$(install_from_checkout "$mcp_install")
run_commands "$mcp_install"
mcp_reply=$(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | g2g-mcp)
echo "$mcp_reply"
grep -q '"protocolVersion"' <<<"$mcp_reply" || fail "g2g-mcp did not answer initialize"

cargo new --quiet quickstart
cd quickstart
dependencies=$(readme_block Cargo.toml)
workspace_version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$checkout/Cargo.toml" | head -n 1)
# the README pins the last release until tools/release.sh moves it
dependencies=$("$checkout/tools/set-doc-dependency-version.sh" "${workspace_version%.*}" <<<"$dependencies")
sed -i '/^\[dependencies\]$/d' Cargo.toml
{
    echo "$dependencies"
    echo '[patch.crates-io]'
    grep -o '^g2g-[a-z-]*' <<<"$dependencies" | while read -r crate; do
        echo "$crate = { path = \"$checkout/$crate\" }"
    done
} >>Cargo.toml
readme_block main.rs >src/main.rs
cat Cargo.toml
cargo generate-lockfile
if grep -q '^\[\[patch.unused\]\]' Cargo.lock; then
    fail "the quick start does not build against this checkout"
fi
cargo run
