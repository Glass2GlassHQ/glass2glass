#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
cd "$(dirname "$0")/.."

missing=0
# vendored DeckLink headers and g2g-mcugen output are excluded, neither is g2g's to license
while read -r file; do
  header=$(head -n 20 "$file")
  if ! grep -q '^\(//\|#\) Copyright (C) ' <<<"$header" ||
    ! grep -q '^\(//\|#\) SPDX-License-Identifier: MPL-2.0$' <<<"$header"; then
    echo "$file"
    missing=1
  fi
done < <(git ls-files -- \
  '*.rs' '*.c' '*.h' '*.cpp' '*.hpp' '*.cs' '*.kt' '*.swift' '*.comp' \
  '*.js' '*.mjs' '*.jsx' '*.py' '*.pyi' '*.sh' '*.bash' '*.pl' \
  ':!g2g-plugins/vendor/' \
  ':!examples/mcugen-graphs/src/audio.rs' ':!examples/mcugen-graphs/src/video.rs')

if [ "$missing" -ne 0 ]; then
  echo "::error::the files above have no copyright header, see CONTRIBUTING.md"
  exit 1
fi
