#!/usr/bin/env bash
# Stage the plugin SDK a distribution installs beside g2g-launch, so a plugin builds with no network.
# Usage: tools/plugin-sdk-bundle.sh <prefix>
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <prefix>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$1"
PREFIX="$(cd "$1" && pwd)"
SDK="$PREFIX/share/g2g/plugin-sdk"
VENDOR="$SDK/vendor"
SDK_CRATES=(g2g-core g2g-plugin)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

package_arguments=()
for crate in "${SDK_CRATES[@]}"; do
  package_arguments+=(-p "$crate")
done
# --allow-dirty: the SDK must match the tree the host binary was built from, committed or not.
cargo package --manifest-path "$ROOT/Cargo.toml" "${package_arguments[@]}" \
  --no-verify --allow-dirty --target-dir "$WORK/target"

mkdir -p "$WORK/crates" "$WORK/scratch/src"
for crate_file in "$WORK/target/package/"*.crate; do
  tar -xzf "$crate_file" -C "$WORK/crates"
done

# `cargo vendor` collects the dependency closure of this throwaway crate.
{
  printf '[package]\nname = "g2g-plugin-sdk-closure"\nversion = "0.0.0"\nedition = "2021"\npublish = false\n\n[workspace]\n\n'
  printf '[features]\nmetadata = ["g2g-core/metadata"]\nmulti-thread = ["g2g-core/multi-thread"]\n'
  # The packaged g2g-plugin names g2g-core by version, which crates.io may not have yet.
  for section in dependencies patch.crates-io; do
    printf '\n[%s]\n' "$section"
    for crate in "${SDK_CRATES[@]}"; do
      crate_dir="$(echo "$WORK/crates/$crate"-[0-9]*)"
      printf '%s = { path = "%s" }\n' "$crate" "$crate_dir"
    done
  done
} >"$WORK/scratch/Cargo.toml"
touch "$WORK/scratch/src/lib.rs"
# The vendored versions must be the ones the host linked.
cp "$ROOT/Cargo.lock" "$WORK/scratch/Cargo.lock"

rm -rf "$VENDOR"
cargo vendor --quiet --versioned-dirs --respect-source-config --manifest-path "$WORK/scratch/Cargo.toml" "$VENDOR" >/dev/null

for crate_file in "$WORK/target/package/"*.crate; do
  crate_dir_name="$(basename "$crate_file" .crate)"
  mv "$WORK/crates/$crate_dir_name" "$VENDOR/"
  checksum="$(sha256sum "$crate_file" | cut -d' ' -f1)"
  printf '{"files":{},"package":"%s"}\n' "$checksum" >"$VENDOR/$crate_dir_name/.cargo-checksum.json"
done

cat >"$SDK/config.toml" <<'EOF'
[source.crates-io]
replace-with = "g2g-plugin-sdk"

[source.g2g-plugin-sdk]
directory = "plugin-sdk/vendor"
EOF

mkdir -p "$PREFIX/include/g2g" "$PREFIX/share/pkgconfig"
cp "$ROOT/g2g-plugin/include/g2g_plugin_v2.h" "$PREFIX/include/g2g/"
sdk_version="$(basename "$(echo "$VENDOR"/g2g-plugin-[0-9]*)")"
sdk_version="${sdk_version#g2g-plugin-}"
cat >"$PREFIX/share/pkgconfig/g2g-plugin.pc" <<EOF
prefix=\${pcfiledir}/../..
includedir=\${prefix}/include

Name: g2g-plugin
Description: C header for glass2glass v2 plugins
Version: $sdk_version
Cflags: -I\${includedir}/g2g
EOF

echo "plugin SDK $sdk_version staged under $PREFIX"
