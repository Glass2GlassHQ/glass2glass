#!/usr/bin/env bash
# sets the g2g version requirement in the docs' Cargo.toml lines, in the given files or on stdin
set -euo pipefail
version=${1:?usage: set-doc-dependency-version.sh <major.minor> [file...]}
shift
expression="s/^(g2g-[a-z]+ += (\{ version = )?)\"[0-9]+\.[0-9]+\"/\1\"$version\"/"
if [ "$#" -eq 0 ]; then
    sed -E "$expression"
else
    sed -i -E "$expression" "$@"
fi
